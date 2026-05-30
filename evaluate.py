# evaluate.py
import numpy as np
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset
import yaml

# ── Конфиг ────────────────────────────────────────────────────────
WEIGHTS   = "runs/segment/runs/seg/yolo11_signs_v2/weights/best.pt"
DATA_YAML = "dataset/data.yaml"
VAL_DIR   = "dataset/rrs/valid"   # папка с images/ и labels/
CONF      = 0.5
IOU_THR   = 0.5
IMG_SIZE  = 1280
# ──────────────────────────────────────────────────────────────────


def load_val_items(val_dir: str, class_names: dict):
    """Читает все изображения и YOLO-seg лейблы из val_dir."""
    img_dir = Path(val_dir) / "images"
    lbl_dir = Path(val_dir) / "labels"
    items = []
    for img_path in sorted(img_dir.glob("*")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        gt_masks, gt_labels = [], []
        for line in lbl_path.read_text().strip().splitlines():
            parts = list(map(float, line.split()))
            cls   = int(parts[0])
            coords = np.array(parts[1:]).reshape(-1, 2)
            coords[:, 0] *= W
            coords[:, 1] *= H
            poly = coords.astype(np.int32)
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 1)
            gt_masks.append(mask)
            gt_labels.append(cls)
        if gt_masks:
            items.append({
                "path":      img_path,
                "gt_masks":  gt_masks,
                "gt_labels": gt_labels,
                "H": H, "W": W,
            })
    return items


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred,  gt).sum()
    return float(inter / union) if union > 0 else 0.0


def centroid(m: np.ndarray):
    ys, xs = np.where(m)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else (0.0, 0.0)


def l2(pred_mask, gt_mask) -> float:
    cx1, cy1 = centroid(pred_mask)
    cx2, cy2 = centroid(gt_mask)
    return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))


def evaluate():
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    class_names = cfg["names"]          # {0: 'road sign'} или {0: ..., 1: ...}
    n_classes   = len(class_names)

    model = YOLO(WEIGHTS)
    items = load_val_items(VAL_DIR, class_names)
    print(f"Validation images: {len(items)}")

    # Накопители
    per_cls_iou  = {c: [] for c in range(n_classes)}
    per_cls_prec = {c: [] for c in range(n_classes)}
    per_cls_rec  = {c: [] for c in range(n_classes)}
    per_cls_l2   = {c: [] for c in range(n_classes)}

    iou_thresholds = [0.5, 0.75, 0.9]
    # считаем по изображениям (есть хоть один объект с IoU >= t)
    img_iou_counts = {t: 0 for t in iou_thresholds}
    total_images_with_gt = 0

    for item in items:
        H, W = item["H"], item["W"]
        gt_masks  = item["gt_masks"]
        gt_labels = item["gt_labels"]

        result = model(str(item["path"]), conf=CONF, imgsz=IMG_SIZE, verbose=False)[0]

        pred_masks, pred_labels = [], []
        if result.masks is not None and result.boxes is not None:
            for i in range(len(result.boxes)):
                pm = cv2.resize(
                    result.masks.data[i].cpu().numpy(),
                    (W, H), interpolation=cv2.INTER_NEAREST
                ) > 0.5
                pred_masks.append(pm)
                pred_labels.append(int(result.boxes.cls[i].item()))

        matched_gt = set()
        best_ious_this_image = []

        for pi, (pm, pl) in enumerate(zip(pred_masks, pred_labels)):
            best_iou_val, best_gi = 0.0, -1
            for gi, (gm, gl) in enumerate(zip(gt_masks, gt_labels)):
                if gi in matched_gt or gl != pl:
                    continue
                v = mask_iou(pm, gm)
                if v > best_iou_val:
                    best_iou_val, best_gi = v, gi

            if best_gi >= 0 and best_iou_val > 0.1:
                matched_gt.add(best_gi)
                gm = gt_masks[best_gi]

                per_cls_iou[pl].append(best_iou_val)
                per_cls_l2[pl].append(l2(pm, gm))
                best_ious_this_image.append(best_iou_val)

                tp = np.logical_and(pm, gm).sum()
                fp = np.logical_and(pm, ~gm.astype(bool)).sum()
                fn = np.logical_and(~pm, gm.astype(bool)).sum()
                per_cls_prec[pl].append(float(tp / (tp + fp + 1e-8)))
                per_cls_rec[pl].append(float(tp / (tp + fn + 1e-8)))

        # Порог по изображению: хоть один объект достигает IoU >= t
        if gt_masks:
            total_images_with_gt += 1
            for t in iou_thresholds:
                if any(v >= t for v in best_ious_this_image):
                    img_iou_counts[t] += 1

    # ── Вывод ──────────────────────────────────────────────────────
    W1, W2, W3, W4, W5, W6 = 16, 7, 8, 7, 8, 6
    sep = "=" * (W1 + W2 + W3 + W4 + W5 + W6 + 5)

    print(f"\n{sep}")
    print(f"{'Class':<{W1}} {'mIoU':>{W2}} {'Prec':>{W3}} {'Rec':>{W4}} {'L2(px)':>{W5}} {'N':>{W6}}")
    print(sep)

    all_ious = []
    for c in range(n_classes):
        ious = per_cls_iou[c]
        if not ious:
            continue
        miou = np.mean(ious)
        prec = np.mean(per_cls_prec[c])
        rec  = np.mean(per_cls_rec[c])
        l2v  = np.mean(per_cls_l2[c])
        name = class_names[c] if isinstance(class_names, dict) else class_names[c]
        all_ious.extend(ious)
        print(f"{name:<{W1}} {miou:>{W2}.3f} {prec:>{W3}.3f} {rec:>{W4}.3f} {l2v:>{W5}.1f} {len(ious):>{W6}}")

    print(sep)
    overall_miou = np.mean(all_ious) if all_ious else 0.0
    print(f"{'mIoU overall':<{W1}} {overall_miou:>{W2}.3f}")
    print()

    print("IoU threshold coverage (% images with at least one match):")
    for t in iou_thresholds:
        pct = 100 * img_iou_counts[t] / max(total_images_with_gt, 1)
        print(f"  IoU >= {t}: {pct:5.1f}%  ({img_iou_counts[t]}/{total_images_with_gt})")

    print(sep)

    # Также быстрые метрики через встроенный val ultralytics
    print("\n── Ultralytics built-in val ──────────────────────────────")
    metrics = model.val(data=DATA_YAML, split="val", conf=CONF, iou=IOU_THR, imgsz=IMG_SIZE)
    print(f"mAP50:      {metrics.seg.map50:.4f}")
    print(f"mAP50-95:   {metrics.seg.map:.4f}")
    print(f"Precision:  {metrics.seg.mp:.4f}")
    print(f"Recall:     {metrics.seg.mr:.4f}")


if __name__ == "__main__":
    evaluate()