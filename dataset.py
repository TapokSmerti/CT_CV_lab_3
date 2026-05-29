import json
import cv2
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset


class RoadSignsDataset(Dataset):

    def __init__(self, root):

        self.root = Path(root)

        self.images = sorted(
            self.root.glob("*.jpg")
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]

        json_path = Path(
            str(img_path) + "_coco.json"
        )

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        H, W = image.shape[:2]

        with open(json_path) as f:
            data = json.load(f)

        masks = np.array(
            data["masks"],
            dtype=np.uint8
        )

        if masks.shape[0] != len(data["class_ids"]):
            masks = np.transpose(
                masks,
                (2, 0, 1)
            )

        full_masks = []

        for mask, bbox in zip(
            masks,
            data["bbox"]
        ):

            y1, x1, y2, x2 = bbox

            full = np.zeros(
                (H, W),
                dtype=np.uint8
            )

            mask = cv2.resize(
                mask.astype(np.uint8),
                (x2 - x1, y2 - y1)
            )

            full[
                y1:y2,
                x1:x2
            ] = mask

            full_masks.append(full)

        full_masks = np.array(
            full_masks,
            dtype=np.uint8
        )

        boxes = torch.tensor(
            data["bbox"],
            dtype=torch.float32
        )

        labels = torch.tensor(
            data["class_ids"],
            dtype=torch.int64
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).permute(2,0,1) / 255.

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": torch.tensor(
                full_masks
            )
        }

        return image, target