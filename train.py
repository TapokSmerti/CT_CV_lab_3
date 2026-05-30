# train_yolo.py
from ultralytics import YOLO

model = YOLO("yolo11s-seg.pt")  # скачается автоматически

results = model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device=0,
    project="runs/seg",
    name="yolo11_signs",
    patience=10,
    save=True,
)
