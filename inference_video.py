# inference_video_yolo.py
from ultralytics import YOLO
from pathlib import Path
import cv2

model = YOLO(" runs/segment/runs/seg/yolo11_signs/weights/best.pt")

for video in Path("videos").glob("*.mp4"):
    model.predict(
        source=str(video),
        save=True,
        conf=0.5,
        project="videos/output",
        name=video.stem,
    )
    print(f"Готово: {video.name}")