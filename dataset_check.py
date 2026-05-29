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