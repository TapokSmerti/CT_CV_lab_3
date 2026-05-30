# inference_video.py
import cv2
import torch
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pathlib import Path

CLASS_NAMES = [
    "background",
    "stop", "pedestrian", "parking", "no_entry",
    "priority", "crossing", "speed_limit", "traffic_lights"
]

COLORS = [
    (0,   0,   0  ),  # background — не используется
    (255, 56,  56 ),  # stop         — красный
    (255, 157, 151),  # pedestrian   — розовый
    (255, 112, 31 ),  # parking      — оранжевый
    (255, 178, 29 ),  # no_entry     — жёлтый
    (207, 210, 49 ),  # priority     — жёлто-зелёный
    (72,  249, 10 ),  # crossing     — зелёный
    (146, 204, 23 ),  # speed_limit  — салатовый
    (61,  219, 134),  # traffic_lights — бирюзовый
]

NUM_CLASSES = 9
IMG_SIZE    = (640, 640)


def build_model(weights_path, device):
    model = maskrcnn_resnet50_fpn(weights=None)
    in_f  = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES)
    in_fm = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_fm, 256, NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def overlay_mask(frame, mask, color, alpha=0.45):
    colored = np.zeros_like(frame, dtype=np.uint8)
    colored[mask] = color
    return cv2.addWeighted(frame, 1.0, colored, alpha, 0)


def process_frame(model, frame_rgb, device, conf=0.4):
    """Инференс одного кадра. frame_rgb — numpy HxWx3 RGB."""
    H, W = frame_rgb.shape[:2]

    # Ресайз до размера обучения
    inp = cv2.resize(frame_rgb, IMG_SIZE)
    tensor = torch.tensor(inp, dtype=torch.float32).permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor)[0]

    keep        = pred["scores"] >= conf
    masks_raw   = pred["masks"][keep].squeeze(1).cpu().numpy()   # N x 640 x 640
    labels      = pred["labels"][keep].cpu().numpy()
    scores      = pred["scores"][keep].cpu().numpy()
    boxes_raw   = pred["boxes"][keep].cpu().numpy()              # N x 4, в 640x640

    # Масштабируем боксы обратно в размер кадра
    sx, sy = W / IMG_SIZE[0], H / IMG_SIZE[1]

    detections = []
    for mask_s, label, score, box_s in zip(masks_raw, labels, scores, boxes_raw):

        # Маска → оригинальный размер
        mask_full = cv2.resize(
            (mask_s > 0.5).astype(np.uint8),
            (W, H),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # Бокс → оригинальный размер
        x1, y1, x2, y2 = box_s
        x1, y1, x2, y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)

        detections.append({
            "mask":   mask_full,
            "label":  int(label),
            "score":  float(score),
            "box":    (x1, y1, x2, y2),
        })

    return detections


def draw_detections(frame_bgr, detections):
    """Рисует маски и боксы на BGR кадре."""
    out = frame_bgr.copy()

    for det in detections:
        label  = det["label"]
        score  = det["score"]
        mask   = det["mask"]
        x1, y1, x2, y2 = det["box"]
        color  = COLORS[label]

        # Полупрозрачная маска
        out = overlay_mask(out, mask, color)

        # Контур маски
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color, 2)

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Подпись
        label_text = f"{CLASS_NAMES[label]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, label_text,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 2
        )

    return out


def process_video(video_path: str, weights_path: str, out_path: str, conf=0.4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(weights_path, device)

    cap = cv2.VideoCapture(video_path)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS, (W, H)
    )

    frame_idx = 0
    print(f"Processing {Path(video_path).name}  ({total_frames} frames, {FPS:.0f} fps)")

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = process_frame(model, frame_rgb, device, conf)
        result     = draw_detections(frame_bgr, detections)

        # Счётчик кадра
        cv2.putText(
            result, f"{frame_idx+1}/{total_frames}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )

        out.write(result)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"  {frame_idx}/{total_frames} frames done")

    cap.release()
    out.release()
    print(f"Saved → {out_path}\n")


if __name__ == "__main__":
    WEIGHTS = "maskrcnn_signs_best.pth"

    videos = list(Path("videos").glob("*.mp4"))
    if not videos:
        print("Положи видео в папку videos/")
    else:
        Path("videos/output").mkdir(parents=True, exist_ok=True)
        for v in videos:
            process_video(
                str(v), WEIGHTS,
                f"videos/output/{v.stem}_seg.mp4"
            )