import torch
from ultralytics import YOLO  


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

    box_loss = results.results_dict.get("train/box_loss", 0.0)
    cls_loss = results.results_dict.get("train/cls_loss", 0.0)
    dfl_loss = results.results_dict.get("train/dfl_loss", 0.0)

    return box_loss + cls_loss + dfl_loss


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

    eval_loss = metrics.box.map        
    eval_acc  = metrics.box.map50     

    return eval_loss, eval_acc