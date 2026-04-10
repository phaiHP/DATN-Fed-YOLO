import os
import pandas as pd
import torch
from ultralytics import YOLO  


def train(net: YOLO, partition_id: int, epochs: int, lr: float, numround: int, device):
    """Train YOLOv8 on the client's partition using local data.yaml."""

    yaml_path = f"C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch/pest24_{numround}/partitions/client_{partition_id}/data.yaml"

    results = net.train(
        data=yaml_path,
        epochs=epochs,
        lr0=lr,
        device=device,
        optimizer="SGD",
        momentum=0.9,
        imgsz=640,
        batch=4,
        verbose=True,
        plots=True,
        save=True, 
        amp=False,           
    )
    box_loss=0
    cls_loss=0
    dfl_loss=0
    save_dir = results.save_dir 
    results_csv = os.path.join(save_dir, "results.csv")
    print(f"save_dir: {save_dir}")
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        # Lấy hàng cuối cùng (epoch cuối)
        last_row = df.iloc[-1]
    
    # Tên cột trong YOLOv8 thường là: '         train/box_loss', '         train/cls_loss', ...
    # Lưu ý: YOLO thường thêm khoảng trắng vào tên cột trong CSV
        box_loss = last_row.get('train/box_loss', 0.0)
        cls_loss = last_row.get('train/cls_loss', 0.0)
        dfl_loss = last_row.get('train/dfl_loss', 0.0)
        print(f"Extracted Losses - Box: {box_loss}, Cls: {cls_loss}, DFL: {dfl_loss}")
    # box_loss = results.results_dict.get("train/box_loss", 0.0)
    # cls_loss = results.results_dict.get("train/cls_loss", 0.0)
    # dfl_loss = results.results_dict.get("train/dfl_loss", 0.0)
    print(f"Results Dict Keys: {results.results_dict.keys()}") # Xem YOLO thực sự trả về những gì
    # box_loss = results.results_dict["train/box_loss"]
    # cls_loss = results.results_dict["train/cls_loss"]
    # dfl_loss = results.results_dict["train/dfl_loss"]

    return box_loss + cls_loss + dfl_loss


def test(net: YOLO, partition_id: int,numround: int, device):
    """Evaluate YOLOv8 on the client's local val split."""

    yaml_path = f"C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch/pest24_{numround}/partitions/client_{partition_id}/data.yaml"

    metrics = net.val(
        data=yaml_path,
        device=device,
        verbose=True,
        plots=True,
         save=True,  
    )

    eval_loss = metrics.box.map        
    eval_acc  = metrics.box.map50     

    return eval_loss, eval_acc