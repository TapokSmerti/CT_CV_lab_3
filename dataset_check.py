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