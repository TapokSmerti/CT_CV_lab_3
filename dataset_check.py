import json
import cv2
import numpy as np


img = cv2.imread("dataset/sign_dataset/train/1.jpg")

print("image shape:", img.shape)

with open("dataset/sign_dataset/train/1.jpg_coco.json") as f:
    d = json.load(f)

print("num masks:", len(d["masks"]))
print("mask shape:", len(d["masks"][0]), len(d["masks"][0][0]))
print("classes:", d["class_ids"])
print("bbox:", d["bbox"][0])


with open(
    "dataset/sign_dataset/train/1.jpg_coco.json"
) as f:
    d = json.load(f)

m = np.array(d["masks"])

print('json data')
print(m.shape)

from dataset import RoadSignsDataset

ds = RoadSignsDataset(
    "dataset/sign_dataset/train"
)

img, target = ds[0]

print(img.shape)

print(target["boxes"].shape)
print(target["labels"].shape)
print(target["masks"].shape)

# Добавь в конец dataset_check.py или запусти отдельно

from dataset import RoadSignsDataset
import torch

ds = RoadSignsDataset("dataset/sign_dataset/train")

print("chech first 20 samples validity...")
for i in range(20):
    img, target = ds[i]
    labels = target["labels"]
    assert labels.min() >= 1, f"Сэмпл {i}: метка < 1: {labels}"
    assert labels.max() <= 8, f"Сэмпл {i}: метка > 8: {labels}"
    boxes = target["boxes"]
    assert (boxes[:, 2] > boxes[:, 0]).all(), f"sample {i}: invalid bbox (x2<=x1)"
    assert (boxes[:, 3] > boxes[:, 1]).all(), f"sample {i}: invalid bbox (y2<=y1)"
    print(f"  [{i}] labels={labels.tolist()}  boxes_shape={boxes.shape}  masks_shape={target['masks'].shape}")

print("OK!")

# check_classes.py
import json
from pathlib import Path
from collections import Counter

data_root = Path("dataset/sign_dataset/train")
jsons = sorted(data_root.glob("*.json"))[:200]

all_class_ids = []
for jp in jsons:
    with open(jp) as f:
        d = json.load(f)
    all_class_ids.extend(d.get("class_ids", []))

print("Уникальные class_ids в датасете:", sorted(set(all_class_ids)))
print("Распределение:", Counter(all_class_ids).most_common())

# check_one.py
import json
import cv2
import numpy as np
from pathlib import Path

# берём первый попавшийся файл
jp = sorted(Path("dataset/sign_dataset/train").glob("*.json"))[0]
img_path = Path(str(jp).replace("_coco.json", ""))

with open(jp) as f:
    d = json.load(f)

print("Файл:", jp.name)
print("class_ids:", d.get("class_ids"))
print("bbox (первый):", d.get("bbox", [[]])[0])
print("Все ключи JSON:", list(d.keys()))

# Визуализируем боксы на картинке
img = cv2.imread(str(img_path))
if img is not None:
    for box, cid in zip(d.get("bbox", []), d.get("class_ids", [])):
        y1, x1, y2, x2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(cid), (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imwrite("check_boxes.jpg", img)
    print("Сохранено: check_boxes.jpg — посмотри что там нарисовано")