import os
import zipfile
import requests
from tqdm import tqdm

AUDIO_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
CHECKSUM_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
DATA_DIR = "./data"

os.makedirs(DATA_DIR, exist_ok=True)


def download_file_with_progress(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as file, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        # Đã sửa lỗi: 1024 * 1024 và xóa dấu '-' thừa
        for data in response.iter_content(chunk_size=1024 * 1024):  # Tải từng block 1MB
            size = file.write(data)
            bar.update(size)


def safe_download_and_extract(url, extract_to):
    file_name = url.split("/")[-1]
    folder_name = file_name.replace(".zip", "")
    archive_path = os.path.join(extract_to, file_name)
    extracted_folder_path = os.path.join(extract_to, folder_name)

    # 1. Kiểm tra xem thư mục đã giải nén sẵn chưa
    if os.path.exists(extracted_folder_path) and os.path.isdir(extracted_folder_path):
        print(f"Thư mục '{folder_name}' đã được giải nén sẵn. Bỏ qua hoàn toàn!")
        return

    # 2. Kiểm tra tệp zip xem có sẵn và hợp lệ không
    need_download = True
    if os.path.exists(archive_path):
        if zipfile.is_zipfile(archive_path):
            print(f"Tệp {file_name} hợp lệ đã tồn tại. Bỏ qua tải xuống.")
            need_download = False
        else:
            print(f"Tệp {file_name} cũ bị lỗi/không hoàn chỉnh. Đang xóa để tải lại...")
            os.remove(archive_path)

    # 3. Tiến hành tải nếu chưa có file zip hợp lệ
    if need_download:
        print(f"Đang tải {file_name}...")
        download_file_with_progress(url, archive_path)

    # 4. Giải nén file zip
    print(f"Đang giải nén {file_name}...")
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Giải nén hoàn tất!\n")


# Tải và giải nén fma_small
safe_download_and_extract(AUDIO_URL, DATA_DIR)

# Tải và giải nén fma_metadata
safe_download_and_extract(CHECKSUM_URL, DATA_DIR)