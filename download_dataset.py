import os
import zipfile
import kaggle

dataset = "viacheslavshalamov/russian-road-signs-segmentation-dataset"

os.makedirs("dataset", exist_ok=True)

kaggle.api.dataset_download_files(dataset, path="dataset", unzip=True)

print("Dataset downloaded and extracted to 'dataset' folder")

# visualize_classes.py
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

data_root = Path("dataset/sign_dataset/train")
jsons     = sorted(data_root.glob("*.json"))

# Собираем по одному примеру каждого класса
class_examples = defaultdict(list)
for jp in jsons[:500]:
    with open(jp) as f:
        d = json.load(f)
    for cid in d.get("class_ids", []):
        class_examples[cid].append(jp)

print("Классы найдены:", sorted(class_examples.keys()))

# Для каждого класса сохраняем картинку с боксом
for cid in sorted(class_examples.keys()):
    jp       = class_examples[cid][0]
    img_path = Path(str(jp).replace("_coco.json", ""))
    img      = cv2.imread(str(img_path))
    if img is None:
        continue

    with open(jp) as f:
        d = json.load(f)

    for i, c in enumerate(d["class_ids"]):
        if c != cid:
            continue
        # bbox формат — проверяем оба варианта
        box = d["bbox"][i]
        if len(box) == 4:
            # пробуем y1,x1,y2,x2
            a, b, c2, dd = [int(v) for v in box]
            # если первые два меньше вторых — это y1,x1,y2,x2
            cv2.rectangle(img, (b, a), (dd, c2), (0, 255, 0), 3)
            cv2.putText(img, f"class={cid}", (b, a - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        break

    out = f"class_vis/class_{cid:02d}.jpg"
    Path("class_vis").mkdir(exist_ok=True)
    # Ресайз чтобы не было огромных картинок
    h, w = img.shape[:2]
    if w > 1280:
        img = cv2.resize(img, (1280, int(h * 1280 / w)))
    cv2.imwrite(out, img)
    print(f"  class {cid:2d}: сохранено → {out}  (примеров: {len(class_examples[cid])})")