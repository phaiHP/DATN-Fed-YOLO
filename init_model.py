import torch
from ultralytics import YOLO
import torch
print(f"CUDA khả dụng: {torch.cuda.is_available()}")
print(f"Tên GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
net = YOLO("yolov8n.pt")
results = net.train(
    data="C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch/pest24/data.yaml",
    epochs=1,
    imgsz=640,
    batch=4,
    device="0" if torch.cuda.is_available() else "cpu",
    amp=False,
    project="runs/init",
    name="pest24_init",
)
# def train_model():
#     net = YOLO("yolov8n.pt")
#     results = net.train(
#         data="pest24/data.yaml",
#         epochs=1,
#         imgsz=800,
#         batch=4,
#         device="0" if torch.cuda.is_available() else "cpu",
#         amp=False,
#         project="runs/init",
#         name="pest24_init",
#         workers=2, # Bạn có thể giảm số này xuống nếu máy bị treo
#     )
# if __name__ == '__main__':
#     train_model()

#Lấy best.pt từ kết quả train
# import shutil

# shutil.copy(
#     "C:/Users/PRECISION/Downloads/quickstart-pytorch/quickstart-pytorch/runs/init/pest24_init18/weights/best.pt",
#     "pest24_init.pt",
# )
# print("✅ Saved pest24_init.pt")

# net2 = YOLO("pest24_init.pt")
# print("nc =", net2.model.nc)  # phải ra 24
