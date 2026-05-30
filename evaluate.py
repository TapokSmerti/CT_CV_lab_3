# evaluate.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

from dataset import RoadSignsDataset

CLASS_NAMES = [
    "background",
    "stop", "pedestrian", "parking", "no_entry",
    "priority", "crossing", "speed_limit", "traffic_lights"
]
NUM_CLASSES = 9


def collate(batch):
    return tuple(zip(*batch))


def build_model(weights_path, device):
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred,  gt).sum()
    return float(inter / union) if union > 0 else 0.0


def evaluate(weights_path: str, data_root: str, conf: float = 0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(weights_path, device)

    ds     = RoadSignsDataset(data_root)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=collate, num_workers=0)

    # Метрики по классам
    per_class_iou       = {i: [] for i in range(1, NUM_CLASSES)}
    per_class_tp        = {i: 0  for i in range(1, NUM_CLASSES)}
    per_class_fp        = {i: 0  for i in range(1, NUM_CLASSES)}
    per_class_fn        = {i: 0  for i in range(1, NUM_CLASSES)}
    per_class_l2        = {i: [] for i in range(1, NUM_CLASSES)}

    iou_thresholds      = [0.5, 0.75, 0.9]
    iou_counts          = {t: 0 for t in iou_thresholds}
    total_gt            = 0

    for images, targets in tqdm(loader, desc="Evaluating"):
        images = [x.to(device) for x in images]

        with torch.no_grad():
            preds = model(images)

        for pred, target in zip(preds, targets):

            # Фильтруем предсказания по confidence
            keep = pred["scores"] >= conf
            pred_masks  = pred["masks"][keep].squeeze(1).cpu().numpy() > 0.5
            pred_labels = pred["labels"][keep].cpu().numpy()
            pred_scores = pred["scores"][keep].cpu().numpy()

            gt_masks    = target["masks"].cpu().numpy().astype(bool)
            gt_labels   = target["labels"].cpu().numpy()

            matched_gt  = set()

            for pi, (pm, pl) in enumerate(zip(pred_masks, pred_labels)):

                best_iou, best_gi = 0.0, -1

                for gi, (gm, gl) in enumerate(zip(gt_masks, gt_labels)):
                    if gi in matched_gt:
                        continue
                    if gl != pl:
                        continue
                    iou = mask_iou(pm, gm)
                    if iou > best_iou:
                        best_iou, best_gi = iou, gi

                if best_gi >= 0 and best_iou > 0.1:
                    matched_gt.add(best_gi)
                    per_class_iou[pl].append(best_iou)
                    per_class_tp[pl] += 1

                    # L2 между центроидами
                    def centroid(m):
                        ys, xs = np.where(m)
                        return (xs.mean(), ys.mean()) if len(xs) else (0, 0)

                    cx1, cy1 = centroid(pm)
                    cx2, cy2 = centroid(gt_masks[best_gi])
                    l2 = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                    per_class_l2[pl].append(float(l2))

                    # IoU thresholds
                    for t in iou_thresholds:
                        if best_iou >= t:
                            iou_counts[t] += 1
                    total_gt += 1
                else:
                    per_class_fp[pl] += 1

            # FN: GT-объекты без совпадения
            for gi, gl in enumerate(gt_labels):
                if gi not in matched_gt:
                    per_class_fn[gl] += 1
                    per_class_iou[gl].append(0.0)
                    total_gt += 1

    # Печатаем результаты
    print("\n" + "="*65)
    print(f"{'Class':<16} {'mIoU':>6} {'Prec':>6} {'Rec':>6} {'L2':>7} {'N':>5}")
    print("="*65)

    all_ious = []
    for cls_id in range(1, NUM_CLASSES):
        ious  = per_class_iou[cls_id]
        tp    = per_class_tp[cls_id]
        fp    = per_class_fp[cls_id]
        fn    = per_class_fn[cls_id]
        l2s   = per_class_l2[cls_id]

        miou  = np.mean(ious) if ious else 0.0
        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        l2    = np.mean(l2s)  if l2s  else 0.0

        all_ious.extend(ious)
        name = CLASS_NAMES[cls_id]
        print(f"{name:<16} {miou:>6.3f} {prec:>6.3f} {rec:>6.3f} {l2:>7.1f} {len(ious):>5}")

    print("="*65)
    print(f"{'mIoU (all)':<16} {np.mean(all_ious):>6.3f}")
    print()
    print("IoU threshold coverage:")
    for t in iou_thresholds:
        pct = 100 * iou_counts[t] / max(total_gt, 1)
        print(f"  IoU >= {t}: {pct:.1f}%  ({iou_counts[t]}/{total_gt})")


if __name__ == "__main__":
    evaluate(
        weights_path="maskrcnn_signs_best.pth",
        data_root="dataset/sign_dataset/val",  # или test
        conf=0.5
    )