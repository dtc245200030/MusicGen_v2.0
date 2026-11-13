# MusicGenAI — sinh nhạc từ mô tả văn bản

MusicGenAI là một dự án text-to-music tự huấn luyện, gồm mô hình ngôn ngữ cho audio token và giao diện web dạng hội thoại. Người dùng nhập mô tả bản nhạc, hệ thống sinh audio WAV 24 kHz, phát trực tiếp trên trình duyệt và lưu lại lịch sử cục bộ.

> Dự án này không sử dụng checkpoint MusicGen chính thức của Meta. Kiến trúc trong repository kết hợp **T5** để mã hóa văn bản, **Transformer Decoder** để sinh token tự hồi quy và **EnCodec** để chuyển đổi giữa token và waveform.

## Tính năng

- Sinh nhạc từ prompt dài tối đa 500 ký tự.
- Tự động dùng CUDA khi có GPU NVIDIA tương thích, nếu không sẽ chạy bằng CPU.
- Hỗ trợ Classifier-Free Guidance (CFG), temperature và top-k sampling.
- Giao diện web responsive, có trình phát và tải file WAV.
- Lưu, tìm kiếm, mở lại và xóa từng cuộc hội thoại.
- API HTTP đơn giản, không phụ thuộc web framework.
- Quy trình đầy đủ để tải FMA Small, tiền xử lý audio và huấn luyện lại mô hình.

## Kiến trúc

Luồng suy luận:

```text
Prompt
  └─ T5 Encoder (t5-base)
       └─ Transformer Decoder + CFG
            └─ 4 luồng token EnCodec (codebook 1.024 phần tử)
                 └─ EnCodec 24 kHz / 3 kbps
                      └─ WAV mono 24 kHz
```

Checkpoint hiện tại dùng cấu hình:

| Thành phần | Giá trị |
|---|---:|
| Số codebook | 4 |
| Kích thước codebook | 1.024 |
| Kích thước ẩn | 512 |
| Attention heads | 8 |
| Transformer layers | 6 |
| Text encoder | `t5-base` |
| Audio codec | EnCodec 24 kHz, 3 kbps |

## Yêu cầu hệ thống

- Python **3.12 trở lên**.
- Khoảng trống đĩa đủ cho dependencies, checkpoint và dữ liệu. Riêng checkpoint hiện tại khoảng 806 MB; FMA Small sau khi giải nén chiếm nhiều GB.
- Khuyến nghị GPU NVIDIA có CUDA để suy luận và huấn luyện nhanh hơn. CPU vẫn được hỗ trợ nhưng sinh nhạc sẽ chậm đáng kể.
- Kết nối Internet ở lần chạy đầu để tải model `t5-base` và trọng số EnCodec nếu chúng chưa có trong cache.

Cấu hình trong `pyproject.toml` đang cố định PyTorch `2.14.1` từ index CUDA 13.0. Nếu máy không hỗ trợ cấu hình này, cần chọn bản PyTorch phù hợp với hệ điều hành/GPU rồi điều chỉnh dependency tương ứng.

## Cài đặt

### Cách 1: dùng uv (khuyến nghị)

```powershell
uv sync
```

Sau đó chạy lệnh trong môi trường dự án:

```powershell
uv run python server.py
```

### Cách 2: dùng virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python server.py
```

Nếu PowerShell chặn script kích hoạt, có thể chạy trực tiếp:

```powershell
.\.venv\Scripts\python.exe server.py
```

## Chạy ứng dụng web

Đảm bảo checkpoint nằm đúng tại:

```text
checkpoints/best_musicgen_model.pt
```

Khởi động server:

```powershell
python server.py
```

Mở [http://127.0.0.1:8000](http://127.0.0.1:8000) trên trình duyệt. Server khởi động giao diện trước và chỉ nạp model khi có yêu cầu sinh nhạc đầu tiên, vì vậy yêu cầu đầu tiên thường mất nhiều thời gian hơn.

Kiểm tra CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Khi dùng CUDA, model được chuyển sang FP16 để giảm VRAM. Khi dùng CPU, model giữ độ chính xác mặc định.

## Cấu hình suy luận

Các tham số được cấu hình bằng biến môi trường trước khi chạy server:

```powershell
$env:MUSICGEN_HOST = "127.0.0.1"
$env:MUSICGEN_PORT = "8000"
$env:MUSICGEN_STEPS = "375"
$env:MUSICGEN_TEMPERATURE = "1.0"
$env:MUSICGEN_TOP_K = "250"
$env:MUSICGEN_CFG_SCALE = "3.0"
python server.py
```

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `MUSICGEN_HOST` | `127.0.0.1` | Địa chỉ server lắng nghe. Dùng `0.0.0.0` nếu muốn truy cập từ máy khác trong mạng. |
| `MUSICGEN_PORT` | `8000` | Cổng HTTP. |
| `MUSICGEN_STEPS` | `375` | Số frame token cần sinh; server giới hạn trong khoảng 25–750. EnCodec tạo khoảng 75 frame/giây, nên 375 bước tương đương khoảng 5 giây. |
| `MUSICGEN_TEMPERATURE` | `1.0` | Độ ngẫu nhiên khi sampling. Giá trị thấp thường ổn định hơn. |
| `MUSICGEN_TOP_K` | `250` | Chỉ sampling trong k token có xác suất cao nhất. Đặt `0` để tắt lọc top-k. |
| `MUSICGEN_CFG_SCALE` | `3.0` | Mức độ bám theo prompt. Giá trị quá cao có thể làm giảm chất lượng âm thanh. |

Thời gian sinh tăng gần tuyến tính theo `MUSICGEN_STEPS`. Mô hình hiện sinh tự hồi quy nên đoạn nhạc dài có thể mất nhiều thời gian, đặc biệt trên CPU.

## API

### Kiểm tra trạng thái

```http
GET /api/health
```

Ví dụ phản hồi:

```json
{
  "status": "ok",
  "device": "cuda:0",
  "checkpoint": true,
  "model_loaded": true
}
```

Trước lần sinh đầu tiên, `device` có thể là `not loaded` và `model_loaded` là `false`.

### Sinh nhạc

```http
POST /api/generate
Content-Type: application/json

{
  "prompt": "A calm lo-fi track with warm piano and soft rain",
  "session_id": null
}
```

`session_id` là tùy chọn. Gửi lại ID nhận được từ lần trước để nối thêm bản nhạc vào cùng cuộc hội thoại.

Ví dụ với PowerShell:

```powershell
$body = @{ prompt = "Epic cinematic music with heavy drums" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/generate -ContentType "application/json" -Body $body
```

### Lịch sử

| Method | Endpoint | Chức năng |
|---|---|---|
| `GET` | `/api/history` | Lấy danh sách tóm tắt các cuộc hội thoại. |
| `GET` | `/api/history/{id}` | Lấy đầy đủ một cuộc hội thoại. |
| `DELETE` | `/api/history/{id}` | Xóa cuộc hội thoại và các file audio liên quan. |
| `DELETE` | `/api/history` | Xóa toàn bộ lịch sử và audio đã sinh. |
| `GET` | `/audio/{uuid}.wav` | Phát hoặc tải một file audio đã sinh. |

## Chuẩn bị dữ liệu và huấn luyện

Phần này chỉ cần thiết khi muốn huấn luyện lại model. Để chạy web với checkpoint có sẵn, có thể bỏ qua.

### 1. Tải FMA Small và metadata

```powershell
python download-musiccaps.py
```

Mặc dù tên script là `download-musiccaps.py`, script hiện tải hai gói **FMA Small** từ máy chủ FMA:

```text
data/
├── fma_small/       # các file MP3
└── fma_metadata/    # tracks.csv, echonest.csv, ...
```

Script bỏ qua gói đã giải nén, kiểm tra file ZIP có hợp lệ và tải lại nếu archive bị hỏng.

### 2. Tiền xử lý

```powershell
python preprocess.py
```

Quy trình tiền xử lý:

1. Chọn các track thuộc subset `small` trong `tracks.csv`.
2. Tạo prompt tiếng Anh từ thể loại, nghệ sĩ, album, tiêu đề, tag và đặc trưng Echo Nest (tempo, energy, danceability) nếu có.
3. Đọc audio, chuyển về mono 24 kHz.
4. Lấy đoạn 10 giây, ưu tiên bắt đầu từ giây thứ 5; audio ngắn được zero-pad.
5. Mã hóa bằng EnCodec 24 kHz ở 3 kbps thành 4 codebook token.
6. Lưu từng mẫu vào `data/processed_tokens/{track_id}.pt` cùng metadata và split chuẩn của FMA.

Nếu file token đã tồn tại, script chỉ làm mới prompt/metadata mà không encode lại audio. Sau khi thay đổi dữ liệu hoặc split, nên xóa `data/processed_tokens/split_index.pt` để dataset tạo lại cache chỉ mục.

Kiểm tra DataLoader:

```powershell
python data.py
```

### 3. Huấn luyện

```powershell
python train.py
```

Cấu hình huấn luyện nằm trực tiếp ở đầu `train.py`:

| Tham số | Giá trị hiện tại |
|---|---:|
| Batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 16 |
| Epochs | 100 |
| Learning rate | `1.5e-4` |
| Weight decay | `0.01` |
| CFG prompt dropout | `0.2` |
| Scheduler | Cosine Annealing Warm Restarts |

Training sử dụng AMP trên CUDA, gradient clipping, theo dõi top-1/top-5/top-10 accuracy và validation loss. Checkpoint có validation loss tốt nhất được ghi vào `checkpoints/best_musicgen_model.pt`, gồm trọng số model, optimizer, scheduler, epoch và `model_config` dùng khi server nạp model.

Thiết lập mặc định được ghi chú cho GPU Google Colab T4 15 GB. Nếu thiếu VRAM, hãy giảm `BATCH_SIZE`, tăng `ACCUMULATION_STEPS` để giữ effective batch size, hoặc giảm kích thước model.

## Cấu trúc thư mục

```text
MusicGen/
├── checkpoints/
│   └── best_musicgen_model.pt  # checkpoint tốt nhất
├── data/
│   ├── fma_small/              # audio nguồn
│   ├── fma_metadata/           # metadata FMA
│   └── processed_tokens/       # token EnCodec sau tiền xử lý
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── runtime/                    # tự tạo khi chạy server
│   ├── audio/                  # WAV đã sinh
│   └── history.json            # lịch sử hội thoại
├── data.py                     # PyTorch Dataset/DataLoader
├── download-musiccaps.py       # tải FMA Small và metadata
├── model.py                    # T5 + Transformer + EnCodec
├── preprocess.py               # tạo token và prompt huấn luyện
├── server.py                   # HTTP server và API
├── train.py                    # vòng lặp huấn luyện
├── pyproject.toml
└── uv.lock
```

## Dữ liệu runtime

Server tự tạo thư mục `runtime/`:

- `runtime/history.json` chứa lịch sử hội thoại.
- `runtime/audio/*.wav` chứa audio đã sinh.

Xóa lịch sử qua giao diện hoặc API cũng xóa các file WAV tương ứng. Nếu cần giữ kết quả, hãy tải hoặc sao lưu file trước khi xóa.

## Xử lý sự cố

### Không tìm thấy PyTorch

Nếu `.venv` tồn tại, `server.py` sẽ thử tự khởi chạy lại bằng `.venv\Scripts\python.exe`. Nếu vẫn lỗi, đồng bộ lại dependency bằng `uv sync` hoặc cài project vào môi trường đang dùng.

### Không tìm thấy checkpoint

Kiểm tra file `checkpoints/best_musicgen_model.pt`. Tên và đường dẫn này đang được cố định trong `server.py`.

### CUDA không khả dụng

Kiểm tra `torch.cuda.is_available()`, driver NVIDIA và phiên bản PyTorch/CUDA. Dự án vẫn chạy bằng CPU nhưng chậm hơn nhiều.

### Hết VRAM

Giảm `MUSICGEN_STEPS` khi suy luận. Khi huấn luyện, giảm `BATCH_SIZE`. Đóng các tiến trình đang sử dụng GPU trước khi chạy lại.

### Lần chạy đầu bị chậm hoặc lỗi tải model

T5 và EnCodec có thể cần tải trọng số từ Internet. Kiểm tra kết nối mạng và cache của Hugging Face/PyTorch, sau đó chạy lại.

### Port 8000 đang được sử dụng

```powershell
$env:MUSICGEN_PORT = "8080"
python server.py
```

## Giới hạn hiện tại

- Sinh token theo kiểu tự hồi quy nên chưa phù hợp để tạo audio dài theo thời gian thực.
- Mỗi lần server chỉ xử lý một lượt sinh nhạc để tránh tranh chấp model/GPU.
- Lịch sử được lưu bằng một file JSON cục bộ, chưa phù hợp cho triển khai nhiều người dùng hoặc nhiều worker.
- Server chưa có xác thực, HTTPS, rate limiting hay cơ chế cô lập người dùng. Không nên mở trực tiếp ra Internet.
- Chất lượng phụ thuộc vào checkpoint, quy mô FMA Small và chất lượng prompt; đây là dự án nghiên cứu/thử nghiệm, không phải dịch vụ sản xuất.

## Bản quyền và giấy phép

Repository hiện chưa có file `LICENSE`. Vì vậy, chưa có quyền sử dụng, chỉnh sửa hoặc phân phối mã nguồn nào được cấp một cách rõ ràng. Dữ liệu FMA, model T5, EnCodec và các dependency có giấy phép/điều khoản riêng; hãy kiểm tra các điều khoản đó trước khi sử dụng cho mục đích thương mại hoặc phân phối sản phẩm.
