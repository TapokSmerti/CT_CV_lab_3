import json
import cv2
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset

NUM_CLASSES = 8  # stop, pedestrian, parking, no_entry, priority, crossing, speed_limit, traffic_lights


class RoadSignsDataset(Dataset):

    def __init__(self, root):
        self.root = Path(root)
        self.images = sorted(self.root.glob("*.jpg"))
        print(f"Found {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        while True:

            img_path = self.images[idx]
            json_path = Path(str(img_path) + "_coco.json")

            image = cv2.imread(str(img_path))
            if image is None:
                idx = (idx + 1) % len(self)
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            H, W = image.shape[:2]

            try:
                with open(json_path) as f:
                    data = json.load(f)
            except Exception:
                idx = (idx + 1) % len(self)
                continue

            boxes_raw  = data.get("bbox", [])
            labels_raw = data.get("class_ids", [])

            if len(boxes_raw) == 0 or len(labels_raw) == 0:
                idx = (idx + 1) % len(self)
                continue

            # Mask R-CNN: класс 0 = фон, поэтому сдвигаем 0-based → 1-based
            labels_1based = [int(l) + 1 for l in labels_raw]

            # Оставляем только валидные метки [1, NUM_CLASSES]
            valid_idx = [
                i for i, l in enumerate(labels_1based)
                if 1 <= l <= NUM_CLASSES
            ]

            if len(valid_idx) == 0:
                idx = (idx + 1) % len(self)
                continue

            boxes_raw     = [boxes_raw[i]     for i in valid_idx]
            labels_1based = [labels_1based[i]  for i in valid_idx]

            boxes_np = np.array(boxes_raw, dtype=np.float32)
            if boxes_np.ndim != 2 or boxes_np.shape[1] != 4:
                idx = (idx + 1) % len(self)
                continue

            # Маски: исходный формат (56, 56, N) → (N, 56, 56)
            try:
                masks = np.array(data["masks"], dtype=np.uint8)
            except Exception:
                idx = (idx + 1) % len(self)
                continue

            if masks.ndim == 3:
                if masks.shape[-1] == len(labels_raw):
                    masks = np.transpose(masks, (2, 0, 1))
                else:
                    idx = (idx + 1) % len(self)
                    continue
            else:
                idx = (idx + 1) % len(self)
                continue

            # Фильтруем маски по valid_idx
            masks = masks[valid_idx]

            # Разворачиваем маски (56x56) → полный размер изображения
            full_masks = []
            kept_boxes  = []
            kept_labels = []

            for i in range(len(labels_1based)):

                y1, x1, y2, x2 = boxes_raw[i]
                y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)

                # Клипуем по границам изображения
                y1 = max(0, min(y1, H - 1))
                x1 = max(0, min(x1, W - 1))
                y2 = max(0, min(y2, H))
                x2 = max(0, min(x2, W))

                if y2 <= y1 or x2 <= x1:
                    continue

                full = np.zeros((H, W), dtype=np.uint8)
                roi_mask = cv2.resize(
                    masks[i].astype(np.uint8),
                    (x2 - x1, y2 - y1),
                    interpolation=cv2.INTER_NEAREST
                )
                full[y1:y2, x1:x2] = roi_mask
                full_masks.append(full)
                kept_boxes.append([x1, y1, x2, y2])  # формат xyxy для Mask R-CNN
                kept_labels.append(labels_1based[i])

            if len(full_masks) == 0:
                idx = (idx + 1) % len(self)
                continue

            full_masks = np.stack(full_masks, axis=0)

            image_tensor = torch.tensor(
                image, dtype=torch.float32
            ).permute(2, 0, 1) / 255.0

            target = {
                "boxes": torch.tensor(kept_boxes, dtype=torch.float32),
                "labels": torch.tensor(kept_labels, dtype=torch.int64),
                "masks": torch.tensor(full_masks, dtype=torch.uint8),
            }

            return image_tensor, target