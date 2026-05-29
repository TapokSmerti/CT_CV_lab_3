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

        print(f"Found {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        while True:

            img_path = self.images[idx]

            json_path = Path(
                str(img_path) + "_coco.json"
            )

            image = cv2.imread(str(img_path))

            if image is None:
                idx = (idx + 1) % len(self)
                continue

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            H, W = image.shape[:2]

            try:
                with open(json_path) as f:
                    data = json.load(f)
            except Exception:
                idx = (idx + 1) % len(self)
                continue

            boxes_raw = data.get("bbox", [])
            labels_raw = data.get("class_ids", [])

            if len(boxes_raw) == 0:
                idx = (idx + 1) % len(self)
                continue

            boxes_np = np.array(
                boxes_raw,
                dtype=np.float32
            )

            if boxes_np.ndim != 2:
                idx = (idx + 1) % len(self)
                continue

            if boxes_np.shape[1] != 4:
                idx = (idx + 1) % len(self)
                continue

            masks = np.array(
                data["masks"],
                dtype=np.uint8
            )

            #
            # у тебя masks=(56,56,N)
            #
            if masks.ndim == 3:

                if masks.shape[-1] == len(labels_raw):
                    masks = np.transpose(
                        masks,
                        (2, 0, 1)
                    )

            else:
                idx = (idx + 1) % len(self)
                continue

            full_masks = []

            for i in range(len(labels_raw)):

                mask = masks[i]

                y1, x1, y2, x2 = boxes_raw[i]

                y1 = int(y1)
                x1 = int(x1)
                y2 = int(y2)
                x2 = int(x2)

                if y2 <= y1 or x2 <= x1:
                    continue

                full = np.zeros(
                    (H, W),
                    dtype=np.uint8
                )

                roi_mask = cv2.resize(
                    mask.astype(np.uint8),
                    (x2 - x1, y2 - y1),
                    interpolation=cv2.INTER_NEAREST
                )

                full[
                    y1:y2,
                    x1:x2
                ] = roi_mask

                full_masks.append(full)

            if len(full_masks) == 0:
                idx = (idx + 1) % len(self)
                continue

            full_masks = np.stack(
                full_masks,
                axis=0
            )

            image = torch.tensor(
                image,
                dtype=torch.float32
            ).permute(2, 0, 1) / 255.0

            target = {
                "boxes": torch.tensor(
                    boxes_np,
                    dtype=torch.float32
                ),
                "labels": torch.tensor(
                    labels_raw,
                    dtype=torch.int64
                ),
                "masks": torch.tensor(
                    full_masks,
                    dtype=torch.uint8
                )
            }

            return image, target