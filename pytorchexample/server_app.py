"""pest24example: A Flower / YOLOv8 Server App."""

import torch
from ultralytics import YOLO
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
# from flwr.server.strategy import FedAvg
# ── init model with 24 class ──
MODEL_PATH = "C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch/pest24_init.pt"
# net = YOLO(MODEL_PATH)
# MODEL_PATH = "/home/btldevteam/data/han-experiment/RAG/flwr/quickstart-pytorch/pest24_init.pt"
# CENTRAL_YAML = "/home/btldevteam/data/han-experiment/RAG/flwr/quickstart-pytorch/pest24/data.yaml"
CENTRAL_YAML = "C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch/pest24/data.yaml"

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate aggregated YOLO model on centralized val data."""
    print(f"\n[Round {server_round}] Running global evaluation...")

    net = YOLO(MODEL_PATH)
    state_dict = arrays.to_torch_state_dict()
    net.model.load_state_dict(state_dict, strict=True)  

    metrics = net.val(
        data=CENTRAL_YAML,
        device="0" if torch.cuda.is_available() else "cpu",
        verbose=False,
        plots=False,
    )

    map50_95 = float(metrics.box.map)
    map50    = float(metrics.box.map50)
    print(f"[Round {server_round}] mAP50-95: {map50_95:.4f} | mAP50: {map50:.4f}")

    return MetricRecord({
        "mAP50-95": map50_95,
        "mAP50":    map50,
    })


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int          = context.run_config["num-server-rounds"]
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