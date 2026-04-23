"""pytorchexample: A Flower / YOLOv8 Client App."""

import os
import torch
from ultralytics import YOLO
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from pytorchexample.yolo_task import train as train_fn, test as test_fn

app = ClientApp()

MODEL_PATH = "C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch/pest24_init.pt"


def count_images(folder: str) -> int:
    if not os.path.exists(folder):
        return 0
    return len([
        f for f in os.listdir(folder)
        if f.endswith((".jpg", ".jpeg", ".png"))
    ])


@app.train()
def train(msg: Message, context: Context):
    """Train the YOLOv8 model on local pest24 partition."""

    partition_id = context.node_config["partition-id"]
    # current_round = msg.content["config"].get("server_round", 0)
    current_round = msg.content["config"].get("server-round", msg.content["config"].get("round", 0))
    print(f"\n[Client {partition_id}] Starting training for round {current_round}...")
    device       = "cpu"
#
    net = YOLO(MODEL_PATH)
    state_dict = msg.content["arrays"].to_torch_state_dict()
    net.model.load_state_dict(state_dict, strict=False)

    train_loss = train_fn(
        net,
        partition_id=partition_id,
        epochs=context.run_config["local-epochs"],
        lr=msg.content["config"]["lr"],
        numround=current_round,
        device=device,
    )
    # print(f"[Client {partition_id}] Finished training with loss: {train_loss:.4f}")

    train_img_dir = (
        f"C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch"
        f"/pest24_{current_round}/partitions/client_{partition_id}/train/images"
    )
    num_examples = count_images(train_img_dir)

    model_record  = ArrayRecord(net.model.state_dict())  
    metric_record = MetricRecord({
        "train_loss":   float(train_loss[0]),
          "val_loss": float(train_loss[1]),  # 👈 tổng loss
        "num-examples": num_examples,
        "map5095":   float(train_loss[2]),  # 👈 thêm train_map
         "map50": float(train_loss[3]),  # 👈 thêm train_map50
    })
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the YOLOv8 model on local pest24 val split."""

    partition_id = context.node_config["partition-id"]
    device       = "cpu"
    # current_round = msg.content["config"].get("server_round", 0)
    current_round = msg.content["config"].get("server-round", msg.content["config"].get("round", 0))
    print(f"msg.content['config']: {msg.content['config']}")  # Debug: xem config thực tế
    print(f"\n[Client {partition_id}] Starting evaluation for round {current_round}...")
    net = YOLO(MODEL_PATH)
    state_dict = msg.content["arrays"].to_torch_state_dict()
    net.model.load_state_dict(state_dict, strict=False)

    eval_map, eval_map50 = test_fn(
        net,
        partition_id=partition_id,
        numround=current_round,
        device=device,
    )

    val_img_dir = (
        f"C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch"
        f"/pest24_{current_round}/partitions/client_{partition_id}/valid/images"
    )
    num_examples = count_images(val_img_dir)

    metric_record = MetricRecord({
        # "eval_loss":    float(eval_loss),
        "eval_map":     float(eval_map),
        "eval_map50":   float(eval_map50),
        "num-examples": num_examples,
    })
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
def fit(self, parameters, config):
    round_num = config["server_round"]   # 👈 lấy round

    save_dir = f"runs/round_{round_num}"

    results = self.model.train(
        data=...,
        epochs=1,
        project="runs",
        name=f"round_{round_num}",   # 👈 KEY
        exist_ok=True               # không overwrite
    )