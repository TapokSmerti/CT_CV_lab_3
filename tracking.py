# tracking.py
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

model = YOLO("runs/segment/runs/seg/yolo11_signs_v2/weights/best.pt")


def count_id_switches(history: dict) -> int:
    """
    history: {frame_id: [(track_id, x1, y1, x2, y2), ...]}
    ID Switch = объект (IoU > 0.5 с предыдущим кадром) сменил track_id.
    """
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / (ua + 1e-6)

    switches = 0
    frames = sorted(history.keys())
    for i in range(1, len(frames)):
        prev = {t[0]: t[1:] for t in history[frames[i - 1]]}
        for tid, box in [(t[0], t[1:]) for t in history[frames[i]]]:
            best_iou, best_id = 0.0, None
            for pid, pbox in prev.items():
                v = iou(box, pbox)
                if v > best_iou:
                    best_iou, best_id = v, pid
            if best_iou > 0.5 and best_id != tid:
                switches += 1
    return switches


def run_tracker(video_path: str, tracker_name: str) -> tuple[int, int]:
    """Запускает трекер на видео, сохраняет результат. Возвращает (id_switches, unique_ids)."""
    cap = cv2.VideoCapture(video_path)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    out_path = f"videos/output/{Path(video_path).stem}_{tracker_name}.mp4"
    Path("videos/output").mkdir(parents=True, exist_ok=True)

    results_gen = model.track(
        source=video_path,
        tracker=f"{tracker_name}.yaml",
        conf=0.5,
        iou=0.5,
        persist=True,
        stream=True,
        imgsz=1280,
    )

    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)
    )

    history   = {}
    frame_idx = 0
    id_colors = {}
    total_ids = set()

    for result in results_gen:
        frame      = result.orig_img.copy()
        frame_data = []

        if result.boxes is not None and result.boxes.id is not None:
            boxes  = result.boxes.xyxy.cpu().numpy()
            ids    = result.boxes.id.cpu().numpy().astype(int)
            labels = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()

            masks = None
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()

            for i, (box, tid, label, score) in enumerate(zip(boxes, ids, labels, scores)):
                x1, y1, x2, y2 = map(int, box)
                total_ids.add(tid)

                if tid not in id_colors:
                    np.random.seed(tid * 17)
                    id_colors[tid] = tuple(np.random.randint(80, 230, 3).tolist())
                color = id_colors[tid]

                if masks is not None and i < len(masks):
                    mask = cv2.resize(
                        masks[i], (W, H), interpolation=cv2.INTER_NEAREST
                    ) > 0.5
                    overlay       = frame.copy()
                    overlay[mask] = color
                    frame         = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
                    contours, _   = cv2.findContours(
                        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(frame, contours, -1, color, 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID:{tid} {model.names[label]} {score:.2f}",
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )

                frame_data.append((tid, x1, y1, x2, y2))

        history[frame_idx] = frame_data

        cv2.putText(
            frame,
            f"{tracker_name} | frame {frame_idx} | unique IDs: {len(total_ids)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        writer.write(frame)
        frame_idx += 1

    writer.release()

    id_sw = count_id_switches(history)
    return id_sw, len(total_ids)


# ── Запуск ────────────────────────────────────────────────────────

videos = sorted(Path("videos").glob("*.mp4"))

if not videos:
    print("Положи видео в папку videos/*.mp4")
else:
    # Собираем результаты
    rows = []
    for v in videos:
        print(f"Processing {v.name} ...")
        sw_byte, ids_byte = run_tracker(str(v), "bytetrack")
        sw_bot,  ids_bot  = run_tracker(str(v), "botsort")
        rows.append((v.name, sw_byte, ids_byte, sw_bot, ids_bot))

    # Итоговая таблица
    W1, W2, W3, W4 = 38, 12, 8, 8
    sep = "=" * (W1 + W2 + W3 + W4 + 2)

    print(f"\n{sep}")
    print(f"{'Video':<{W1}} {'Tracker':<{W2}} {'ID Sw':>{W3}} {'IDs':>{W4}}")
    print(sep)

    for video_name, sw_byte, ids_byte, sw_bot, ids_bot in rows:
        print(f"{video_name:<{W1}} {'bytetrack':<{W2}} {sw_byte:>{W3}} {ids_byte:>{W4}}")
        print(f"{'':<{W1}} {'botsort':<{W2}} {sw_bot:>{W3}} {ids_bot:>{W4}}")

    print(sep)
    print("Видео сохранены в videos/output/")