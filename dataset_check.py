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


# find_sign_classes.py
import json
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np

data_root = Path("dataset/sign_dataset/train")
jsons = sorted(data_root.glob("*.json"))

# Смотрим все ключи и структуру
with open(jsons[0]) as f:
    d = json.load(f)
print("Ключи JSON:", list(d.keys()))
print("class_ids пример:", d.get("class_ids"))

# Проверяем есть ли поле с именами классов
for key in d.keys():
    print(f"\n{key}:", d[key] if key != "masks" else f"<masks shape {np.array(d[key]).shape}>")

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