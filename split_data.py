import os
import shutil
import math
from pathlib import Path


def split_pest24_standard(input_root, yaml_file, output_root, num_rounds, num_clients):
    # 1. Thiết lập đường dẫn gốc
    input_path = Path(input_root)
    yaml_path = Path(yaml_file)
    out_path = Path(output_root)
    
    # Các loại dữ liệu cần xử lý
    data_types = ['train', 'valid', 'test']

    for d_type in data_types:
        img_dir = input_path / d_type / "images"
        lbl_dir = input_path / d_type / "labels"
        
        if not img_dir.exists():
            print(f"Bỏ qua {d_type} vì không tìm thấy thư mục images.")
            continue

        # Lấy danh sách ảnh của loại này (train/valid/hoặc test)
        all_images = sorted(list(img_dir.glob("*.jpg")))
        total_files = len(all_images)
        files_per_round = math.ceil(total_files / num_rounds)

        for r in range(num_rounds):
            round_name = f"pest24_{r}"
            
            # Chia subset cho Round
            r_start = r * files_per_round
            r_end = min(r_start + files_per_round, total_files)
            round_subset = all_images[r_start:r_end]
            
            num_files_in_round = len(round_subset)
            if num_files_in_round == 0: continue
                
            files_per_client = math.ceil(num_files_in_round / num_clients)

            for c in range(num_clients):
                client_root = out_path / round_name / "partitions" / f"client_{c}"
                
                # Tạo folder đích: client_x / {train/valid/test} / {images/labels}
                dest_img_dir = client_root / d_type / "images"
                dest_lbl_dir = client_root / d_type / "labels"
                dest_img_dir.mkdir(parents=True, exist_ok=True)
                dest_lbl_dir.mkdir(parents=True, exist_ok=True)

                # Copy file data.yaml vào root của client (chỉ copy 1 lần)
                if yaml_path.exists() and not (client_root / "data.yaml").exists():
                    shutil.copy(yaml_path, client_root / "data.yaml")

                # Chia file cho Client
                c_start = c * files_per_client
                c_end = min(c_start + files_per_client, num_files_in_round)
                client_subset = round_subset[c_start:c_end]

                for img_file in client_subset:
                    # Copy Image
                    shutil.copy(img_file, dest_img_dir / img_file.name)
                    
                    # Tìm và copy Label tương ứng
                    txt_file = lbl_dir / f"{img_file.stem}.txt"
                    if txt_file.exists():
                        shutil.copy(txt_file, dest_lbl_dir / txt_file.name)

        print(f"Đã chia xong dữ liệu loại: {d_type}")
def merge_clients_to_round_level(output_root, num_rounds):
    out_path = Path(output_root)
    data_types = ['train', 'valid', 'test']

    for r in range(num_rounds):
        round_path = out_path / f"pest24_{r}"
        partitions_path = round_path / "partitions"
        
        if not partitions_path.exists():
            print(f"Không tìm thấy partitions trong {round_path.name}, bỏ qua.")
            continue

        # print(f"Đang gộp dữ liệu cho {round_name}...")

        # Duyệt qua từng loại dữ liệu (train, valid, test)
        for d_type in data_types:
            # Tạo folder đích ở cấp Round (ngang hàng với partitions)
            global_dest_img = round_path / d_type / "images"
            global_dest_lbl = round_path / d_type / "labels"
            
            global_dest_img.mkdir(parents=True, exist_ok=True)
            global_dest_lbl.mkdir(parents=True, exist_ok=True)

            # Tìm tất cả các folder client_x trong partitions
            for client_dir in partitions_path.iterdir():
                if client_dir.is_dir() and client_dir.name.startswith("client_"):
                    src_img_dir = client_dir / d_type / "images"
                    src_lbl_dir = client_dir / d_type / "labels"

                    # Copy ảnh từ client ra folder chung của Round
                    if src_img_dir.exists():
                        for img_file in src_img_dir.glob("*.jpg"):
                            shutil.copy(img_file, global_dest_img / img_file.name)

                    # Copy nhãn từ client ra folder chung của Round
                    if src_lbl_dir.exists():
                        for lbl_file in src_lbl_dir.glob("*.txt"):
                            shutil.copy(lbl_file, global_dest_lbl / lbl_file.name)

    print("--- Hoàn tất gộp dữ liệu! ---")
def create_accumulative_expe2(input_root, output_root, max_n=5):
    input_path = Path(input_root)
    output_path = Path(output_root)

    # Lặp qua từng round để tạo folder tương ứng trong expe2
    for current_max in range(max_n + 1):
        target_folder = output_path / f"pest24_{current_max}"
        print(f"--- Đang tạo {target_folder} (tích lũy từ 0 đến {current_max}) ---")
        
        # Gọi lại logic gộp cho mỗi folder mới
        # Gom các pest24_i (với i chạy từ 0 đến current_max) vào target_folder
        merge_rounds_to_final(input_root, target_folder, max_n=current_max)

def merge_rounds_to_final(input_root, dest_path, max_n):
    input_path = Path(input_root)
    dest_path = Path(dest_path)
    data_types = ['train', 'valid', 'test']

    for n in range(max_n + 1):
        round_folder = input_path / f"pest24_{n}"
        if not round_folder.exists():
            continue

        # 1. Gộp folder chung (train, valid, test)
        for d_type in data_types:
            src_type_dir = round_folder / d_type
            if src_type_dir.exists():
                for sub in ['images', 'labels']:
                    src_sub = src_type_dir / sub
                    dest_sub = dest_path / d_type / sub
                    dest_sub.mkdir(parents=True, exist_ok=True)
                    for file in src_sub.glob("*.*"):
                        shutil.copy(file, dest_sub / file.name)

        # 2. Gộp trong partitions/client_x
        partitions_src = round_folder / "partitions"
        if partitions_src.exists():
            for client_dir in partitions_src.iterdir():
                if client_dir.is_dir() and client_dir.name.startswith("client_"):
                    for d_type in data_types:
                        for sub in ['images', 'labels']:
                            src_client_sub = client_dir / d_type / sub
                            dest_client_sub = dest_path / "partitions" / client_dir.name / d_type / sub
                            if src_client_sub.exists():
                                dest_client_sub.mkdir(parents=True, exist_ok=True)
                                for file in src_client_sub.glob("*.*"):
                                    shutil.copy(file, dest_client_sub / file.name)
                    
                    # Copy data.yaml của client
                    client_yaml = client_dir / "data.yaml"
                    if client_yaml.exists():
                        shutil.copy(client_yaml, dest_path / "partitions" / client_dir.name / "data.yaml")

        # 3. Copy data.yaml cấp độ Round
        round_yaml = round_folder / "data.yaml"
        if round_yaml.exists():
            shutil.copy(round_yaml, dest_path / "data.yaml")
def create_sliding_window_expe2(input_root, output_root, max_n=5, window_size=4):
    input_path = Path(input_root)
    output_path = Path(output_root)

    # Lặp qua từng round để tạo folder tương ứng trong expe2
    for current_max in range(max_n + 1):
        target_folder = output_path / f"pest24_{current_max}"
        
        # Tính toán round bắt đầu để đảm bảo chỉ lấy tối đa 'window_size' rounds
        # Ví dụ: current_max = 5, window_size = 4 => start_round = 5 - 4 + 1 = 2 (Lấy 2, 3, 4, 5)
        start_round = max(0, current_max - window_size + 1)
        
        print(f"--- Đang tạo {target_folder.name} ---")
        print(f"    Tích lũy từ Round {start_round} đến {current_max} (Window size: {window_size})")
        
        # Gọi hàm gộp với dải round đã xác định
        merge_limited_rounds(input_root, target_folder, start_round, current_max)

def merge_limited_rounds(input_root, dest_path, start_n, end_n):
    input_path = Path(input_root)
    dest_path = Path(dest_path)
    data_types = ['train', 'valid', 'test']

    for n in range(start_n, end_n + 1):
        round_folder = input_path / f"pest24_{n}"
        if not round_folder.exists():
            continue

        # 1. Gộp folder chung (train, valid, test)
        for d_type in data_types:
            src_type_dir = round_folder / d_type
            if src_type_dir.exists():
                for sub in ['images', 'labels']:
                    src_sub = src_type_dir / sub
                    dest_sub = dest_path / d_type / sub
                    dest_sub.mkdir(parents=True, exist_ok=True)
                    for file in src_sub.glob("*.*"):
                        shutil.copy(file, dest_sub / file.name)

        # 2. Gộp trong partitions/client_x
        partitions_src = round_folder / "partitions"
        if partitions_src.exists():
            for client_dir in partitions_src.iterdir():
                if client_dir.is_dir() and client_dir.name.startswith("client_"):
                    for d_type in data_types:
                        for sub in ['images', 'labels']:
                            src_client_sub = client_dir / d_type / sub
                            dest_client_sub = dest_path / "partitions" / client_dir.name / d_type / sub
                            if src_client_sub.exists():
                                dest_client_sub.mkdir(parents=True, exist_ok=True)
                                for file in src_client_sub.glob("*.*"):
                                    shutil.copy(file, dest_client_sub / file.name)
                    
                    # Copy data.yaml của client
                    client_yaml = client_dir / "data.yaml"
                    if client_yaml.exists():
                        shutil.copy(client_yaml, dest_path / "partitions" / client_dir.name / "data.yaml")

        # 3. Copy data.yaml cấp độ Round (lấy cái mới nhất)
        round_yaml = round_folder / "data.yaml"
        if round_yaml.exists():
            shutil.copy(round_yaml, dest_path / "data.yaml")

# --- Thực thi ---

# --- Thực thi ---

# --- Cấu hình ---
split_pest24_standard(
    input_root='C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch/pest24_1',
    yaml_file='C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch/pest24_1/data.yaml', 
    output_root='C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch/expe3_1', 
    num_rounds=2, 
    num_clients=2
)
# --- Cách sử dụng ---
# Sau khi bạn chạy hàm chia (split) ở trên, hãy gọi hàm này:
merge_clients_to_round_level(
    output_root='C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch/expe3_1', 
    num_rounds=2
)
create_accumulative_expe2(
    input_root='C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch/expe3_1', 
    output_root='C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch/expe3_2', 
    max_n=5
)
create_sliding_window_expe2(
    input_root='C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch/expe3_1', 
    output_root='C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch/expe3_3', 
    max_n=10,        # Giả sử bạn có 10 round
    window_size=4    # Chỉ lấy tối đa 4 round gần nhất
)


