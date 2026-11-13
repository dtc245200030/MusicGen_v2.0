"""MusicGen web server using only Python's standard-library HTTP stack.

Run with: python server.py
"""

from __future__ import annotations

import json
import importlib.util
import os
import re
import sys
import threading
import traceback
import uuid
import wave
from typing import Any
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_musicgen_model.pt"
RUNTIME_DIR = ROOT / "runtime"
AUDIO_DIR = RUNTIME_DIR / "audio"
HISTORY_PATH = RUNTIME_DIR / "history.json"
MAX_PROMPT_LENGTH = 500
SESSION_RE = re.compile(r"^[0-9a-f-]{36}$")

RUNTIME_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

history_lock = threading.RLock()
generation_lock = threading.Lock()
model_lock = threading.Lock()
model_instance: Any | None = None
model_device = "not loaded"


def ensure_ml_runtime() -> None:
    """Re-launch with the project venv when `python server.py` uses global Python."""
    if importlib.util.find_spec("torch") is not None:
        return
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.is_file() and Path(sys.executable).resolve() != venv_python.resolve():
        print(f"PyTorch is not installed in {sys.executable}")
        print(f"Switching to the project environment: {venv_python}", flush=True)
        os.execv(str(venv_python), [str(venv_python), *sys.argv])
    raise RuntimeError(
        "Không tìm thấy PyTorch. Hãy chạy bằng .venv\\Scripts\\python.exe server.py"
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_history() -> list[dict]:
    with history_lock:
        if not HISTORY_PATH.exists():
            return []
        try:
            value = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
            return value if isinstance(value, list) else []
        except (OSError, json.JSONDecodeError):
            return []


def write_history(sessions: list[dict]) -> None:
    with history_lock:
        temporary = HISTORY_PATH.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(HISTORY_PATH)


def get_model():
    global model_instance, model_device
    if model_instance is not None:
        return model_instance

    with model_lock:
        if model_instance is not None:
            return model_instance
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {CHECKPOINT_PATH}")

        # Heavy ML imports stay lazy so the web UI and health endpoint start fast.
        import torch
        from model import MusicTransformerLM

        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading checkpoint on {model_device} ...", flush=True)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        config = checkpoint.get("model_config", {})
        model = MusicTransformerLM(
            num_codebooks=int(config.get("num_codebooks", 4)),
            codebook_size=int(config.get("codebook_size", 1024)),
            d_model=int(config.get("d_model", 512)),
            nhead=int(config.get("nhead", 8)),
            num_layers=int(config.get("num_layers", 6)),
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        # FP16 is well supported on RTX GPUs and keeps this model within 4 GB VRAM.
        if model_device.type == "cuda":
            model.to(device=model_device, dtype=torch.float16).eval()
        else:
            model.to(model_device).eval()
        model_instance = model
        print("Model ready.", flush=True)
        return model


def save_waveform(waveform, destination: Path) -> float:
    import torch

    audio = waveform.detach().float().cpu().squeeze().clamp(-1, 1)
    if audio.ndim > 1:
        audio = audio.mean(dim=0)
    pcm = (audio * 32767).round().to(torch.int16).numpy().tobytes()
    with wave.open(str(destination), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm)
    return audio.numel() / 24000


def generate_music(prompt: str, audio_id: str) -> tuple[str, float, dict]:
    import torch

    model = get_model()
    # EnCodec emits 75 token frames per second. Keep the default short because the
    # current decoder generates autoregressively; users can override it per machine.
    steps = max(25, min(int(os.getenv("MUSICGEN_STEPS", "375")), 750))
    temperature = float(os.getenv("MUSICGEN_TEMPERATURE", "1.0"))
    top_k = int(os.getenv("MUSICGEN_TOP_K", "250"))
    cfg_scale = float(os.getenv("MUSICGEN_CFG_SCALE", "3.0"))

    with generation_lock, torch.inference_mode():
        tokens = model.generate(
            [prompt], max_steps=steps, temperature=temperature,
            top_k=top_k, cfg_scale=cfg_scale,
        )
        waveform = model.tokens_to_audio(tokens)

    filename = f"{audio_id}.wav"
    duration = save_waveform(waveform[0], AUDIO_DIR / filename)
    return f"/audio/{filename}", duration, {
        "mode": "model", "device": str(model_device), "steps": steps,
        "temperature": temperature, "top_k": top_k, "cfg_scale": cfg_scale,
    }


class MusicGenHandler(SimpleHTTPRequestHandler):
    server_version = "MusicGenWeb/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_json(self, payload, status=HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Content-Length không hợp lệ") from exc
        if length <= 0 or length > 16_384:
            raise ValueError("Nội dung yêu cầu rỗng hoặc quá lớn")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("JSON không hợp lệ") from exc
        if not isinstance(value, dict):
            raise ValueError("JSON phải là một object")
        return value

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/api/health":
            self.send_json({
                "status": "ok", "device": str(model_device),
                "checkpoint": CHECKPOINT_PATH.exists(),
                "model_loaded": model_instance is not None,
            })
            return
        if path == "/api/history":
            sessions = read_history()
            summaries = [{
                "id": item["id"], "title": item["title"],
                "updated_at": item["updated_at"],
                "message_count": sum(m.get("role") == "assistant" for m in item.get("messages", [])),
            } for item in sessions]
            self.send_json(summaries)
            return
        if path.startswith("/api/history/"):
            session_id = path.rsplit("/", 1)[-1]
            session = next((s for s in read_history() if s.get("id") == session_id), None)
            self.send_json(session if session else {"error": "Không tìm thấy hội thoại"},
                           HTTPStatus.OK if session else HTTPStatus.NOT_FOUND)
            return
        if path.startswith("/audio/"):
            name = path.rsplit("/", 1)[-1]
            if not re.fullmatch(r"[0-9a-f-]{36}\.wav", name):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.serve_file(AUDIO_DIR / name, "audio/wav")
            return
        if path in ("/", "/index.html"):
            self.serve_file(FRONTEND_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path == "/static/css/style.css":
            self.serve_file(FRONTEND_DIR / "style.css", "text/css; charset=utf-8")
            return
        if path == "/static/js/app.js":
            self.serve_file(FRONTEND_DIR / "app.js", "text/javascript; charset=utf-8")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def serve_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        stat = path.stat()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(stat.st_size))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                self.wfile.write(chunk)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/generate":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            data = self.read_json()
            prompt = str(data.get("prompt", "")).strip()
            if not prompt:
                raise ValueError("Vui lòng nhập mô tả bản nhạc")
            if len(prompt) > MAX_PROMPT_LENGTH:
                raise ValueError(f"Mô tả không được quá {MAX_PROMPT_LENGTH} ký tự")
            session_id = data.get("session_id")
            if session_id is not None and not SESSION_RE.fullmatch(str(session_id)):
                raise ValueError("Mã hội thoại không hợp lệ")

            audio_id = str(uuid.uuid4())
            audio_url, duration, metadata = generate_music(prompt, audio_id)
            sessions = read_history()
            session = next((s for s in sessions if s.get("id") == session_id), None)
            now = utc_now()
            if session is None:
                session_id = str(uuid.uuid4())
                words = prompt.split()
                title = " ".join(words[:7]) + ("…" if len(words) > 7 else "")
                session = {"id": session_id, "title": title, "updated_at": now, "messages": []}
                sessions.insert(0, session)
            session["messages"].extend([
                {"role": "user", "content": prompt, "created_at": now},
                {"role": "assistant", "audio_url": audio_url, "duration": duration,
                 "metadata": metadata, "created_at": now},
            ])
            session["updated_at"] = now
            sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
            write_history(sessions)
            self.send_json({"status": "success", "session_id": session_id,
                            "audio_url": audio_url, "duration": duration,
                            "metadata": metadata})
        except ValueError as exc:
            self.send_json({"status": "error", "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            traceback.print_exc()
            self.send_json({"status": "error", "error": f"Không thể tạo nhạc: {exc}"},
                           HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        sessions = read_history()
        if path == "/api/history":
            for wav in AUDIO_DIR.glob("*.wav"):
                wav.unlink(missing_ok=True)
            write_history([])
            self.send_json({"status": "success"})
            return
        if path.startswith("/api/history/"):
            session_id = path.rsplit("/", 1)[-1]
            target = next((s for s in sessions if s.get("id") == session_id), None)
            if target is None:
                self.send_json({"error": "Không tìm thấy hội thoại"}, HTTPStatus.NOT_FOUND)
                return
            for message in target.get("messages", []):
                url = message.get("audio_url", "")
                if url.startswith("/audio/"):
                    (AUDIO_DIR / url.rsplit("/", 1)[-1]).unlink(missing_ok=True)
            write_history([s for s in sessions if s.get("id") != session_id])
            self.send_json({"status": "success"})
            return
        self.send_error(HTTPStatus.NOT_FOUND)


def main() -> None:
    ensure_ml_runtime()
    host = os.getenv("MUSICGEN_HOST", "127.0.0.1")
    port = int(os.getenv("MUSICGEN_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), MusicGenHandler)
    print(f"MusicGen webchat: http://{host}:{port}")
    print(f"Device: auto (CUDA preferred, detected on model load); checkpoint: {CHECKPOINT_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
