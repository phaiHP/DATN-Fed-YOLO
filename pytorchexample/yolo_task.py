import torch
from ultralytics import YOLO  
import os
import pandas as pd

def train(net: YOLO, partition_id: int, epochs: int, lr: float, device):
    """Train YOLOv8 on the client's partition using local data.yaml."""

    yaml_path = f"C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch/pest24/partitions/client_{partition_id}/data.yaml"

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

    map5095 = results.results_dict.get("metrics/mAP50-95(B)", 1.1)     
    map50  = results.results_dict.get("metrics/mAP50(B)", 1.1)    
    box_loss=0
    cls_loss=0
    dfl_loss=0
    save_dir = results.save_dir 
    results_csv = os.path.join(save_dir, "results.csv")
    print(f"save_dir: {save_dir}")
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        last_row = df.iloc[-1]
        box_loss = last_row.get('train/box_loss', 0.0)
        cls_loss = last_row.get('train/cls_loss', 0.0)
        dfl_loss = last_row.get('train/dfl_loss', 0.0)
        val_box_loss = last_row.get('val/box_loss', 0.0)
        val_cls_loss = last_row.get('val/cls_loss', 0.0)
        val_dfl_loss = last_row.get('val/dfl_loss', 0.0)
        print(f"Extracted Losses - Box: {box_loss}, Cls: {cls_loss}, DFL: {dfl_loss}")
    print(f"Results Dict Keys: {results.results_dict.keys()}")
    train_loss = box_loss + cls_loss + dfl_loss
    val_loss = val_box_loss + val_cls_loss + val_dfl_loss
    print(f"Total Loss: {train_loss}")
    return train_loss, val_loss, map5095, map50


def test(net: YOLO, partition_id: int, device):
    """Evaluate YOLOv8 on the client's local val split."""

    yaml_path = f"C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch/pest24/partitions/client_{partition_id}/data.yaml"

    metrics = net.val(
        data=yaml_path,
        device=device,
        verbose=True,
        plots=True,
         save=True,  
    )

    eval_map = metrics.box.map        
    eval_map50  = metrics.box.map50     

    return eval_map, eval_map50