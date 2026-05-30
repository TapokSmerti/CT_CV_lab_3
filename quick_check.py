# draw_labels.py
import cv2
import numpy as np
from pathlib import Path
import yaml

with open("dataset/rrs/data.yaml") as f:
    cfg = yaml.safe_load(f)

img_path = sorted(Path("dataset/rrs/train/images").glob("*"))[0]
lbl_path = Path(str(img_path).replace("images", "labels").rsplit(".", 1)[0] + ".txt")

img = cv2.imread(str(img_path))
H, W = img.shape[:2]

with open(lbl_path) as f:
    lines = f.readlines()

for line in lines:
    parts = list(map(float, line.strip().split()))
    cls   = int(parts[0])
    coords = parts[1:]  # x y x y x y ... нормализованные

    # Полигон сегментации
    pts = np.array(coords).reshape(-1, 2)
    pts[:, 0] *= W
    pts[:, 1] *= H
    pts = pts.astype(np.int32)

    color = (0, 255, 0) if cls == 1 else (255, 0, 0)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=3)

    x, y = pts[0]
    cv2.putText(img, cfg["names"][cls], (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

cv2.imwrite("draw_labels.jpg", img)
print("Сохранено draw_labels.jpg")