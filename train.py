# train_yolo.py
from ultralytics import YOLO

model = YOLO("yolo11s-seg.pt")  # скачается автоматически

results = model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=1280,
    batch=8,
    device=0,
    project="runs/seg",
    name="yolo11_signs_v2",
    scale=0.9,
    mosaic=1.0,
    copy_paste=0.5,
    mixup=0.1,
    degrees=15,
    perspective=0.001,
    patience=10,
    save=True,
)
