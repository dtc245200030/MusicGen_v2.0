import os
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import PreprocessedMusicDataset
from model import MusicTransformerLM

# ==========================================
# CẤU HÌNH DÀNH RIÊNG CHO GOOGLE COLAB (T4 15GB VRAM)
# ==========================================
TOKEN_DIR = "./data/processed_tokens"
CHECKPOINT_DIR = "./checkpoints"

BATCH_SIZE = 4
ACCUMULATION_STEPS = 4  # Effective Batch Size = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1.5e-4
MIN_LEARNING_RATE = 1e-6
RESTART_EPOCHS = 20
WEIGHT_DECAY = 0.01
CFG_DROP_PROB = 0.2  # Tỉ lệ drop text prompt (20%) để phục vụ Classifier-Free Guidance
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CODEBOOKS = 4
CODEBOOK_SIZE = 1024
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 6

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ==========================================
# HÀM TÍNH LOSS VÀ ACCURACY CHO AUDIO TOKENS
# ==========================================
def compute_loss_and_accuracy(logits, targets):
    """
    logits:  [B, K, T, Codebook_Size]
    targets: [B, K, T_raw] (Audio Tokens từ EnCodec)
    """
    B, K, T_target = targets.shape
    total_loss = 0.0
    total_correct = 0
    total_top5_correct = 0
    total_top10_correct = 0
    total_tokens = 0

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for k in range(K):
        # Input trong model đã được dịch phải một bước.
        logit_k = logits[:, k, :T_target, :]  # [B, T_target, Codebook_Size]
        target_k = targets[:, k, :]  # [B, T_target]

        # 1. Tính Loss
        # Cross entropy ở FP32 để tránh overflow/NaN khi forward dùng AMP FP16.
        loss_k = criterion(logit_k.float().reshape(-1, CODEBOOK_SIZE), target_k.reshape(-1))
        total_loss += loss_k

        # 2. Tính Accuracy
        preds_k = torch.argmax(logit_k, dim=-1)  # [B, T_target]
        correct_k = (preds_k == target_k).sum().item()
        with torch.no_grad():
            top10 = torch.topk(logit_k.detach(), k=10, dim=-1).indices
            matches = top10.eq(target_k.unsqueeze(-1))
            total_top5_correct += matches[..., :5].any(dim=-1).sum().item()
            total_top10_correct += matches.any(dim=-1).sum().item()

        total_correct += correct_k
        total_tokens += target_k.numel()

    avg_loss = total_loss / K
    accuracy = (total_correct / total_tokens) * 100.0
    top5_accuracy = (total_top5_correct / total_tokens) * 100.0
    top10_accuracy = (total_top10_correct / total_tokens) * 100.0

    return avg_loss, {
        "top1": accuracy,
        "top5": top5_accuracy,
        "top10": top10_accuracy,
    }


def apply_cfg_dropout(prompts, drop_prob=0.2):
    """
    BỔ SUNG MỚI: Ngẫu nhiên thay thế prompt bằng chuỗi rỗng "" với xác suất drop_prob.
    Giúp mô hình học được Unconditioned Generation cho CFG.
    """
    return [p if random.random() > drop_prob else "" for p in prompts]


def main():
    print(
        f"--- HUẤN LUYỆN MÔ HÌNH TRÊN THIẾT BỊ: ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) ---")

    # 1. Load Dataset theo split chuẩn FMA
    train_dataset = PreprocessedMusicDataset(token_dir=TOKEN_DIR, split="training")
    val_dataset = PreprocessedMusicDataset(token_dir=TOKEN_DIR, split="validation")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Tổng mẫu Train: {len(train_dataset)} | Tổng mẫu Val: {len(val_dataset)}")

    # 2. Khởi tạo Model
    model = MusicTransformerLM(
        num_codebooks=NUM_CODEBOOKS,
        codebook_size=CODEBOOK_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    # 3. Optimizer & Scheduler & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=RESTART_EPOCHS,
        T_mult=2,
        eta_min=MIN_LEARNING_RATE,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == "cuda"))

    best_val_loss = float("inf")

    # 4. Training Loop
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        # --- PHASE 1: TRAIN ---
        model.train()
        train_loss = 0.0
        train_metrics = {"top1": 0.0, "top5": 0.0, "top10": 0.0}
        valid_train_batches = 0
        skipped_train_batches = 0
        accumulation_count = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS:02d} [Train]")
        for i, batch in enumerate(pbar):
            audio_tokens = batch["audio_tokens"].to(DEVICE)
            raw_prompts = batch["prompt"]

            # Áp dụng Condition Dropout (Bắt buộc cho CFG)
            prompts = apply_cfg_dropout(raw_prompts, drop_prob=CFG_DROP_PROB)

            with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
                logits = model(audio_tokens, prompts)
                raw_loss, metrics = compute_loss_and_accuracy(logits, audio_tokens)
                loss_scaled = raw_loss / ACCUMULATION_STEPS

            if not torch.isfinite(raw_loss):
                skipped_train_batches += 1
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
                pbar.set_postfix({"loss": "non-finite (skipped)", "acc": "-"})
                continue

            scaler.scale(loss_scaled).backward()
            accumulation_count += 1

            if accumulation_count >= ACCUMULATION_STEPS:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
                scheduler.step((epoch - 1) + (i + 1) / len(train_loader))

            train_loss += raw_loss.item()
            for name in train_metrics:
                train_metrics[name] += metrics[name]
            valid_train_batches += 1
            pbar.set_postfix({
                "loss": f"{raw_loss.item():.4f}",
                "top1": f"{metrics['top1']:.2f}%",
                "top10": f"{metrics['top10']:.2f}%",
            })

        # Cập nhật nếu epoch kết thúc khi chưa đủ accumulation steps.
        if accumulation_count > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if valid_train_batches == 0:
            raise RuntimeError("Tất cả batch train đều có loss NaN/Inf; dừng huấn luyện.")
        avg_train_loss = train_loss / valid_train_batches
        avg_train_metrics = {
            name: value / valid_train_batches for name, value in train_metrics.items()
        }

        # --- PHASE 2: VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_metrics = {"top1": 0.0, "top5": 0.0, "top10": 0.0}
        valid_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                audio_tokens = batch["audio_tokens"].to(DEVICE)
                prompts = batch["prompt"]  # Trong Val không áp dụng Condition Dropout

                with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
                    logits = model(audio_tokens, prompts)
                    loss, metrics = compute_loss_and_accuracy(logits, audio_tokens)

                if torch.isfinite(loss):
                    val_loss += loss.item()
                    for name in val_metrics:
                        val_metrics[name] += metrics[name]
                    valid_val_batches += 1

        if valid_val_batches == 0:
            raise RuntimeError("Tất cả batch validation đều có loss NaN/Inf.")
        avg_val_loss = val_loss / valid_val_batches
        avg_val_metrics = {
            name: value / valid_val_batches for name, value in val_metrics.items()
        }

        current_lr = scheduler.get_last_lr()[0]

        # Giải phóng cache GPU sau mỗi epoch
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        elapsed_time = time.time() - start_time

        # --- IN KẾT QUẢ TỔNG HỢP CỦA EPOCH ---
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS:02d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Top-1/5/10: {avg_train_metrics['top1']:.2f}%/"
              f"{avg_train_metrics['top5']:.2f}%/{avg_train_metrics['top10']:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Top-1/5/10: {avg_val_metrics['top1']:.2f}%/"
              f"{avg_val_metrics['top5']:.2f}%/{avg_val_metrics['top10']:.2f}% | "
              f"LR: {current_lr:.6f} | Time: {elapsed_time:.1f}s")
        if skipped_train_batches:
            print(f"   Cảnh báo: đã bỏ qua {skipped_train_batches} batch có loss NaN/Inf.")

        # --- LƯU MODEL TỐT NHẤT THEO VAL LOSS ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_musicgen_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": avg_val_metrics["top1"],
                "val_metrics": avg_val_metrics,
                "model_config": {
                    "num_codebooks": NUM_CODEBOOKS,
                    "codebook_size": CODEBOOK_SIZE,
                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "num_layers": NUM_LAYERS,
                },
            }, checkpoint_path)
            print(
                f"   Đã lưu checkpoint tốt nhất! "
                f"(Val Loss: {best_val_loss:.4f} | "
                f"Top-1: {avg_val_metrics['top1']:.2f}% | "
                f"Top-10: {avg_val_metrics['top10']:.2f}%)"
            )


if __name__ == "__main__":
    main()
