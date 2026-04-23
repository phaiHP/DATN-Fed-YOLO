"""pest24example: A Flower / YOLOv8 Server App."""

import torch
from ultralytics import YOLO
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
# import logging
# import sys

# # Cấu hình logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("flwr_yolo_log.txt", encoding='utf-8'), # Ghi vào file
#         logging.StreamHandler(sys.stdout) # Vẫn hiển thị ra màn hình Terminal
#     ]
# )
import sys
import os
from datetime import datetime

# class Logger(object):
#     def __init__(self):
#         self.terminal = sys.stdout
#         # Tạo thư mục 'logs' nếu chưa có
#         if not os.path.exists("logs"):
#             os.makedirs("logs")
            
#         # Tạo tên file dựa trên thời gian hiện tại
#         timestamp = datetime.now().strftime("%Y%m%d_%HH%MM%SS")
#         log_filename = f"logs/training_{timestamp}.log"
        
#         self.log = open(log_filename, "a", encoding='utf-8')
#         print(f"--- Đang ghi log vào file: {log_filename} ---")

#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)

#     def flush(self):
#         self.terminal.flush()
#         self.log.flush()

# # Kích hoạt Logger cho cả đầu ra thông thường và lỗi
# sys.stdout = Logger()
# sys.stderr = sys.stdout 

# # Test thử
# print("Bắt đầu chạy Flower Simulation...")

# Ví dụ sử dụng trong hàm của bạn
# logging.info(f"Results Dict Keys: {results.results_dict.keys()}")
# from flwr.server.strategy import FedAvg
# ── init model with 24 class ──
import datetime
import sys
import os
import logging 
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        if not os.path.exists("logs"):
            os.makedirs("logs")
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/training_{timestamp}.log"
        
        self.log = open(log_filename, "a", encoding='utf-8')
        print(f"--- Đang ghi log vào file: {log_filename} ---")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def fileno(self):
        """Trả về file descriptor của terminal gốc."""
        return self.terminal.fileno()

    def isatty(self):
        """Kiểm tra xem có phải là terminal thực hay không."""
        return self.terminal.isatty()

    @property
    def encoding(self):
        """Trả về encoding của terminal gốc."""
        return self.terminal.encoding

# Kích hoạt Logger
sys.stdout = Logger()
sys.stderr = sys.stdout
log_file_path = sys.stdout.log.name 

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s : %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'), # Ghi vào cùng file log đó
        logging.StreamHandler(sys.stdout.terminal)           # Vẫn đẩy ra màn hình console gốc
    ]
)

# Chỉ định rõ cho Flower và Ray sử dụng cấu hình này
logging.getLogger("flwr").setLevel(logging.INFO)
MODEL_PATH = "C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch/pest24_init.pt"
# net = YOLO(MODEL_PATH)
# MODEL_PATH = "/home/btldevteam/data/han-experiment/RAG/flwr/quickstart-pytorch/pest24_init.pt"
# CENTRAL_YAML = "/home/btldevteam/data/han-experiment/RAG/flwr/quickstart-pytorch/pest24/data.yaml"
CENTRAL_YAML = "C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch/pest24_{round}/data.yaml"

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate aggregated YOLO model on centralized val data."""
    print(f"\n[Round {server_round}] Running global evaluation...")

    net = YOLO(MODEL_PATH)
    state_dict = arrays.to_torch_state_dict()
    net.model.load_state_dict(state_dict, strict=True)  
    central_yaml = CENTRAL_YAML.format(round=server_round)
    metrics = net.val(
        data=central_yaml,
        device= "cpu",
        verbose=False,
        plots=True,
    )

    map50_95 = float(metrics.box.map)
    map50    = float(metrics.box.map50)
    print(f"[Round {server_round}] mAP50-95: {map50_95:.4f} | mAP50: {map50:.4f}")

    return MetricRecord({
        "mAP50-95": map50_95,
        "mAP50":    map50,
    })

def get_on_fit_config_fn():
    """Hàm này trả về một hàm con được gọi mỗi round."""
    def fit_config(server_round: int):
        # Trả về ConfigRecord chứa số round hiện tại
        return ConfigRecord({
            "current_round": server_round, 
            "learning_rate": 0.01  # Bạn có thể lấy lr từ context.run_config nếu muốn
        })
    return fit_config

app = ServerApp()
@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int       = context.run_config["num-server-rounds"]
    lr: float                = context.run_config["learning-rate"]

    print("Initializing global YOLOv8 model (nc=24)...")
    net = YOLO(MODEL_PATH)                          
    arrays = ArrayRecord(net.model.state_dict())
    # Sửa đoạn này trong hàm main()
    # strategy = FedAvg(
    #     fraction_fit=1.0,             # Sử dụng 100% client khả dụng
    #     min_fit_clients=2,            # Cần tối thiểu 2 client để huấn luyện
    #     min_available_clients=2,      # Đợi đủ 2 client mới bắt đầu vòng lặp
    #     fraction_evaluate=1.0,        # Đánh giá trên cả 2 client
    #     min_evaluate_clients=2,       # Tối thiểu 2 client để đánh giá
    # )
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)
#     strategy = FedAvg(
#     fraction_fit=1.0,           # dùng hết client
#     min_fit_clients=2,          # 👈 phải = num-clients
#     min_available_clients=2,    # 👈 phải = num-clients
#     fraction_evaluate=1.0,
#     min_evaluate_clients=2,
# )
    print(f"Starting federated training for {num_rounds} rounds...")
    # result = strategy.start(
    #     grid=grid,
    #     initial_arrays=arrays,
    #     train_config=ConfigRecord({"lr": lr}),
    #     num_rounds=num_rounds,
    #     evaluate_fn=global_evaluate,
        
    # )
    result = strategy.start(
    grid=grid,
    initial_arrays=arrays,
    train_config=ConfigRecord({
        "lr": lr,
        "round": 0,   # 👈 sẽ được update mỗi round
    }),
    num_rounds=num_rounds,
    # on_fit_config_fn=get_on_fit_config_fn(),
    evaluate_fn=global_evaluate,
)

    # ── Save final model ──
    print("\nSaving final aggregated model to disk...")
    final_net = YOLO(MODEL_PATH)
    final_net.model.load_state_dict(
        result.arrays.to_torch_state_dict(),
        strict=True,                                
    )
    final_net.save("pest24_federated_final.pt")
    print("✅ Model saved to pest24_federated_final.pt")