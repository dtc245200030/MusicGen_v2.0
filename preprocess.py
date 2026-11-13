import os
import ast
import html
import re
import unicodedata
from collections import Counter
import torch
import pandas as pd
import soundfile as sf
import torchaudio
from tqdm import tqdm
from encodec import EncodecModel
from encodec.utils import convert_audio

AUDIO_DIR = "./data/fma_small"
METADATA_DIR = "./data/fma_metadata"
OUTPUT_DIR = "./data/processed_tokens"
SAMPLE_RATE = 24000
DURATION = 10  # 10 giây
TARGET_SAMPLES = SAMPLE_RATE * DURATION
MAX_TEXT_LENGTH = 120
MAX_PROMPT_LENGTH = 500


def metadata_value(row, group, field):
    """Lấy giá trị từ Series có MultiIndex 3 tầng của FMA."""
    for column in row.index:
        if isinstance(column, tuple) and column[:2] == (group, field):
            value = row[column]
            return value.iloc[0] if isinstance(value, pd.Series) else value
    return None


def clean_text(value, max_length=MAX_TEXT_LENGTH):
    """Chuẩn hóa metadata thành văn bản an toàn để ghép prompt."""
    if value is None or isinstance(value, (list, tuple, dict, set)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    text = html.unescape(str(value))
    text = re.sub(r"<[^>]*>", " ", text)
    text = unicodedata.normalize("NFKC", text)
    text = "".join(char if char.isprintable() else " " for char in text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" \t,;|/-")
    if not text or text.casefold() in {"nan", "none", "null", "unknown", "n/a"}:
        return None
    return text[:max_length].rstrip()


def clean_number(value, minimum, maximum):
    """Trả về float chỉ khi giá trị hữu hạn và nằm trong miền hợp lệ."""
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not torch.isfinite(torch.tensor(number)) or not minimum <= number <= maximum:
        return None
    return number


def clean_tags(value, limit=4):
    """Làm sạch tags FMA được lưu dưới dạng chuỗi list/dict."""
    if value is None:
        return []
    parsed = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = re.split(r"[,;|]", value)
    if isinstance(parsed, dict):
        parsed = list(parsed.values())
    if not isinstance(parsed, (list, tuple, set)):
        parsed = [parsed]

    result = []
    seen = set()
    for item in parsed:
        if isinstance(item, dict):
            item = item.get("tag_title") or item.get("genre_title") or item.get("name")
        elif isinstance(item, (list, tuple)) and item:
            item = item[-1]
        item = clean_text(item, 40)
        key = item.casefold() if item else None
        if item and key not in seen:
            seen.add(key)
            result.append(item)
        if len(result) >= limit:
            break
    return result


def unique_parts(parts):
    result = []
    seen = set()
    for part in parts:
        part = clean_text(part, MAX_PROMPT_LENGTH)
        key = part.casefold() if part else None
        if part and key not in seen:
            seen.add(key)
            result.append(part)
    return result


def build_prompt(row, echo_row=None):
    genre = clean_text(metadata_value(row, "track", "genre_top"), 60)
    title = clean_text(metadata_value(row, "track", "title"))
    artist = clean_text(metadata_value(row, "artist", "name"))
    album = clean_text(metadata_value(row, "album", "title"))
    composer = clean_text(metadata_value(row, "track", "composer"))
    tags = clean_tags(metadata_value(row, "track", "tags"))

    parts = [f"{genre} track" if genre else "music track"]
    if artist:
        parts.append(f"by {artist}")
    if composer and (not artist or composer.casefold() != artist.casefold()):
        parts.append(f"composed by {composer}")
    if album and (not title or album.casefold() != title.casefold()):
        parts.append(f"from the album {album}")

    if echo_row is not None:
        bpm = clean_number(echo_row.get("tempo"), 20, 300)
        energy = clean_number(echo_row.get("energy"), 0, 1)
        danceability = clean_number(echo_row.get("danceability"), 0, 1)
        if bpm is not None:
            parts.append(f"{round(bpm)} BPM")
        if energy is not None:
            if energy >= 0.7:
                parts.append("high energy")
            elif energy <= 0.3:
                parts.append("calm and low energy")
        if danceability is not None and danceability >= 0.7:
            parts.append("danceable")

    if tags:
        parts.append("with " + ", ".join(tags))
    if title:
        safe_title = title.replace('"', "'")
        parts.append(f'titled "{safe_title}"')

    cleaned_parts = unique_parts(parts)
    article = "An" if cleaned_parts and cleaned_parts[0][0].casefold() in "aeiou" else "A"
    prompt = article + " " + ", ".join(cleaned_parts)
    return prompt[:MAX_PROMPT_LENGTH].rstrip(" ,"), genre, title, artist, album, tags

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. LOAD METADATA TỐI ƯU
print("Loading tracks.csv & echonest.csv...")
tracks = pd.read_csv(os.path.join(METADATA_DIR, "tracks.csv"), index_col=0, header=[0, 1, 2])

# tracks.csv có header 3 tầng. Truy cập bằng tuple 2 phần trả về
# DataFrame (không phải Series), khiến phép lọc giữ nguyên 106.574 dòng.
subset_column = next(col for col in tracks.columns if col[:2] == ('set', 'subset'))
small_tracks = tracks.loc[tracks[subset_column].eq('small')]
print(f"Tìm thấy {len(small_tracks)} track thuộc subset 'small'.")

# Đọc echonest.csv và loại bỏ trùng lặp index
echonest_path = os.path.join(METADATA_DIR, "echonest.csv")
echonest_data = None
if os.path.exists(echonest_path):
    try:
        echonest_df = pd.read_csv(echonest_path, index_col=0, header=[0, 1, 2])['echonest', 'audio_features']
        echonest_data = echonest_df[~echonest_df.index.duplicated(keep='first')]
        print("Đã tích hợp thành công dữ liệu từ echonest.csv!")
    except Exception as e:
        print(f"Không thể đọc echonest.csv ({e}), sẽ dùng prompt mặc định từ tracks.csv.")

# 2. LOAD ENCODEC MODEL
print("Loading EnCodec model...")
model = EncodecModel.encodec_model_24khz()
# 3.0 kbps sinh 4 codebook, khớp với num_codebooks=4 trong model.py.
model.set_target_bandwidth(3.0)
model.to(device)
model.eval()

# 3. PREPROCESS AND SAVE
print("Pre-extracting tokens...")
saved_count = 0
skipped_count = 0
refreshed_count = 0
not_found_count = 0
error_count = 0
error_types = Counter()
error_examples = []

with torch.no_grad():
    for track_id, row in tqdm(small_tracks.iterrows(), total=len(small_tracks)):
        # Định dạng chuẩn 6 chữ số: 2 -> 000002
        track_str = f"{int(track_id):06d}"
        folder_name = track_str[:3]
        file_name = f"{track_str}.mp3"

        # Đường dẫn tệp âm thanh thực tế
        file_path = os.path.join(AUDIO_DIR, folder_name, file_name)

        # Nếu tệp âm thanh không tồn tại -> Bỏ qua
        if not os.path.exists(file_path):
            not_found_count += 1
            continue

        save_path = os.path.join(OUTPUT_DIR, f"{track_str}.pt")
        split = clean_text(metadata_value(row, "set", "split"), 20) or "training"
        echo_row = None
        if echonest_data is not None and track_id in echonest_data.index:
            echo_row = echonest_data.loc[track_id]
        prompt, genre, title, artist, album, tags = build_prompt(row, echo_row)

        # Cập nhật prompt/metadata cho file token cũ mà không encode audio lại.
        if os.path.exists(save_path):
            try:
                saved_data = torch.load(save_path, map_location="cpu", weights_only=False)
                saved_data.update({
                    "prompt": prompt,
                    "genre": genre or "",
                    "title": title or "",
                    "artist": artist or "",
                    "album": album or "",
                    "tags": tags,
                    "split": split,
                })
                torch.save(saved_data, save_path)
                skipped_count += 1
                refreshed_count += 1
            except Exception as e:
                error_count += 1
                error_types[type(e).__name__] += 1
                if len(error_examples) < 5:
                    error_examples.append(f"{save_path}: {type(e).__name__}: {e}")
            continue

        try:
            # Load & Preprocess Audio
            # soundfile đọc MP3 trực tiếp, không phụ thuộc TorchCodec/FFmpeg DLL.
            audio, sr = sf.read(file_path, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(audio.T)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            waveform = convert_audio(waveform, sr, SAMPLE_RATE, model.channels)

            # ĐÃ SỬA LỖI CÚ PHÁP: 5 * SAMPLE_RATE
            start_sample = int(5 * SAMPLE_RATE)
            end_sample = start_sample + TARGET_SAMPLES

            if waveform.shape[-1] >= end_sample:
                waveform = waveform[:, start_sample:end_sample]
            elif waveform.shape[-1] > TARGET_SAMPLES:
                waveform = waveform[:, :TARGET_SAMPLES]
            else:
                pad_size = TARGET_SAMPLES - waveform.shape[-1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_size))

            # Encode to tokens qua EnCodec
            waveform = waveform.unsqueeze(0).to(device)
            encoded_frames = model.encode(waveform)
            codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze(0).cpu()

            # Save token & info
            torch.save({
                "audio_tokens": codes,  # Shape: [n_q, seq_len]
                "prompt": prompt,
                "genre": genre or "",
                "title": title or "",
                "artist": artist or "",
                "album": album or "",
                "tags": tags,
                "split": split
            }, save_path)
            saved_count += 1

        except Exception as e:
            error_count += 1
            error_types[type(e).__name__] += 1
            if len(error_examples) < 5:
                error_examples.append(f"{file_path}: {type(e).__name__}: {e}")
            continue

print("\nKết quả preprocess:")
print(f"  - Mới xử lý và lưu: {saved_count}")
print(f"  - Đã tồn tại, bỏ qua: {skipped_count}")
print(f"  - Đã làm sạch lại prompt cũ: {refreshed_count}")
print(f"  - Không tìm thấy audio: {not_found_count}")
print(f"  - Lỗi khi xử lý: {error_count}")

if error_count:
    print(f"  - Loại lỗi: {dict(error_types)}")
    print("\nMột số lỗi đầu tiên:")
    for error in error_examples:
        print(f"    {error}")

if saved_count == 0 and skipped_count == 0:
    raise RuntimeError(
        "Không tạo được file token nào. Hãy xem các lỗi chi tiết phía trên."
    )

print(f"\nHoàn tất! Hiện có {saved_count + skipped_count} file token trong {OUTPUT_DIR}")
