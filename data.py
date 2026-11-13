import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader


class PreprocessedMusicDataset(Dataset):
    """
    Dataset tối ưu nạp dữ liệu siêu nhanh từ các file .pt đã tiền xử lý.
    """

    def __init__(self, token_dir="./data/processed_tokens", split=None):
        """
        Args:
            token_dir (str): Thư mục chứa các file .pt
            split (str, optional): Chọn 'training', 'validation', hoặc 'test'.
                                   Nếu để None sẽ load toàn bộ dữ liệu.
        """
        all_files = sorted(glob.glob(os.path.join(token_dir, "*.pt")))
        if len(all_files) == 0:
            raise FileNotFoundError(f"Không tìm thấy file .pt nào trong {token_dir}. Hãy chạy preprocess.py trước!")

        self.files = []

        if split is not None:
            # Lọc nhanh thông tin split mà KHÔNG đọc toàn bộ nội dung file .pt bằng torch.load
            # Đọc metadata nhanh bằng cách chỉ nạp khi cần hoặc tạo file danh sách cache
            index_cache_path = os.path.join(token_dir, "split_index.pt")

            if os.path.exists(index_cache_path):
                # Nếu đã có file index cache -> Load tức thì trong 0.01 giây
                index_map = torch.load(index_cache_path, map_location="cpu", weights_only=False)
                self.files = index_map.get(split, [])
            else:
                # Nếu chưa có -> Tạo cache lần đầu tiên cho các lần sau
                print("Đang tạo danh sách chỉ mục (Index Cache) cho các tập split...")
                index_map = {"training": [], "validation": [], "test": []}
                for file_path in all_files:
                    data = torch.load(file_path, map_location="cpu", weights_only=False)
                    file_split = data.get("split", "unknown")
                    if file_split in index_map:
                        index_map[file_split].append(file_path)

                torch.save(index_map, index_cache_path)
                self.files = index_map.get(split, [])

            print(f"Loaded {len(self.files)} samples for split: '{split}'")
        else:
            # Lọc bỏ file split_index.pt nếu có
            self.files = [f for f in all_files if not f.endswith("split_index.pt")]
            print(f"Loaded ALL {len(self.files)} samples.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Đọc dữ liệu đã nén sẵn
        data = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        return {
            "audio_tokens": data["audio_tokens"],  # Tensor [n_q, seq_len]
            "prompt": data["prompt"],  # String Text Prompt
            "genre": data.get("genre", "unknown"),  # Genre
            "split": data.get("split", "unknown")  # Split ('training', 'validation', 'test')
        }


# ==========================================
# Code kiểm thử Dataloader
# ==========================================
if __name__ == "__main__":
    TOKEN_DIR = "./data/processed_tokens"

    if os.path.exists(TOKEN_DIR):
        # Load thử tập Training
        train_dataset = PreprocessedMusicDataset(token_dir=TOKEN_DIR, split="training")
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            pin_memory=True  # Giúp truyền Tensor sang GPU nhanh hơn
        )

        for batch in train_loader:
            print("\n--- KẾT QUẢ LOAD 1 BATCH TRAIN ---")
            print("Audio Tokens Batch Shape:", batch["audio_tokens"].shape)
            print("Ví dụ Prompt:", batch["prompt"][0])
            print("Ví dụ Genre:", batch["genre"][0])
            print("Ví dụ Split:", batch["split"][0])
            break
    else:
        print(f"Chưa tìm thấy thư mục {TOKEN_DIR}. Hãy chạy file preprocess.py trước!")