# tracking.py
import cv2
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from inference_video import build_model, process_frame, draw_detections, COLORS, CLASS_NAMES

NUM_CLASSES = 9


# ──────────────────────────────────────────────
# SORT
# ──────────────────────────────────────────────

class KalmanTracker:
    _next_id = 1

    def __init__(self, box, label):
        from filterpy.kalman import KalmanFilter
        self.id    = KalmanTracker._next_id
        self.label = label
        KalmanTracker._next_id += 1

        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.F[0, 4] = self.kf.F[1, 5] = self.kf.F[2, 6] = 1

        self.kf.H      = np.eye(4, 7)
        self.kf.R[2:,2:] *= 10
        self.kf.P[4:,4:] *= 1000
        self.kf.P        *= 10
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        x1, y1, x2, y2 = box
        cx, cy = (x1+x2)/2, (y1+y2)/2
        s = (x2-x1) * (y2-y1)
        r = (x2-x1) / (y2-y1 + 1e-6)
        self.kf.x[:4] = [[cx], [cy], [s], [r]]

        self.hits      = 1
        self.no_losses = 0

    def predict(self):
        self.kf.predict()
        self.no_losses += 1
        return self._bbox()

    def update(self, box):
        x1, y1, x2, y2 = box
        cx, cy = (x1+x2)/2, (y1+y2)/2
        s = (x2-x1) * (y2-y1)
        r = (x2-x1) / (y2-y1 + 1e-6)
        self.kf.update([[cx], [cy], [s], [r]])
        self.hits      += 1
        self.no_losses  = 0

    def _bbox(self):
        cx, cy, s, r = self.kf.x[:4].flatten()
        w = np.sqrt(abs(s * r))
        h = abs(s) / (w + 1e-6)
        return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / (ua + 1e-6)


class SORT:
    def __init__(self, max_age=5, min_hits=1, iou_thr=0.3):
        self.max_age  = max_age
        self.min_hits = min_hits
        self.iou_thr  = iou_thr
        self.trackers = []
        KalmanTracker._next_id = 1

    def update(self, detections):
        """detections: list of {"box": [x1,y1,x2,y2], "label": int}"""

        # Предсказываем позиции
        predicted = [t.predict() for t in self.trackers]

        # Матрица IoU
        matched_t, matched_d = set(), set()
        if self.trackers and detections:
            iou_mat = np.zeros((len(detections), len(self.trackers)))
            for di, det in enumerate(detections):
                for ti, pred_box in enumerate(predicted):
                    if det["label"] == self.trackers[ti].label:
                        iou_mat[di, ti] = box_iou(det["box"], pred_box)

            row_ind, col_ind = linear_sum_assignment(-iou_mat)
            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] >= self.iou_thr:
                    self.trackers[c].update(detections[r]["box"])
                    matched_t.add(c)
                    matched_d.add(r)

        # Новые треки для несопоставленных детекций
        for di, det in enumerate(detections):
            if di not in matched_d:
                self.trackers.append(KalmanTracker(det["box"], det["label"]))

        # Удаляем старые треки
        self.trackers = [t for t in self.trackers if t.no_losses <= self.max_age]

        # Возвращаем активные треки
        results = []
        for t in self.trackers:
            if t.hits >= self.min_hits:
                results.append({
                    "id":    t.id,
                    "label": t.label,
                    "box":   t._bbox(),
                })
        return results


# ──────────────────────────────────────────────
# ByteTrack (упрощённая версия)
# ──────────────────────────────────────────────

class ByteTracker:
    """
    Упрощённый ByteTrack: two-stage matching.
    High-conf detections → IoU match.
    Low-conf detections  → second-pass match с unmatched треками.
    """
    def __init__(self, high_thr=0.5, low_thr=0.2, max_age=10, iou_thr=0.3):
        self.high_thr = high_thr
        self.low_thr  = low_thr
        self.max_age  = max_age
        self.iou_thr  = iou_thr
        self.trackers = []
        self._next_id = 1

    def _new_id(self):
        i = self._next_id
        self._next_id += 1
        return i

    def _match(self, dets, trk_indices, iou_thr):
        if not dets or not trk_indices:
            return [], list(range(len(dets))), list(trk_indices)

        iou_mat = np.zeros((len(dets), len(trk_indices)))
        for di, det in enumerate(dets):
            for j, ti in enumerate(trk_indices):
                if det["label"] == self.trackers[ti]["label"]:
                    iou_mat[di, j] = box_iou(det["box"], self.trackers[ti]["box"])

        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        matched, unmatched_d, unmatched_t = [], [], list(range(len(dets)))

        matched_t_set = set()
        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= iou_thr:
                matched.append((r, trk_indices[c]))
                matched_t_set.add(c)
                if r in unmatched_d:
                    unmatched_d.remove(r)

        unmatched_t = [trk_indices[c] for c in range(len(trk_indices))
                       if c not in matched_t_set]
        unmatched_d = [di for di in range(len(dets))
                       if di not in {r for r, _ in matched}]

        return matched, unmatched_d, unmatched_t

    def update(self, detections):
        high = [d for d in detections if d["score"] >= self.high_thr]
        low  = [d for d in detections if self.low_thr <= d["score"] < self.high_thr]

        all_t = list(range(len(self.trackers)))

        # Stage 1: high-conf ↔ все треки
        matched1, unmatched_d1, unmatched_t1 = self._match(high, all_t, self.iou_thr)

        for r, ti in matched1:
            self.trackers[ti]["box"]      = high[r]["box"]
            self.trackers[ti]["no_losses"] = 0
            self.trackers[ti]["hits"]     += 1

        # Stage 2: low-conf ↔ unmatched треки
        matched2, unmatched_d2, unmatched_t2 = self._match(low, unmatched_t1, self.iou_thr)

        for r, ti in matched2:
            self.trackers[ti]["box"]       = low[r]["box"]
            self.trackers[ti]["no_losses"] = 0
            self.trackers[ti]["hits"]     += 1

        # Новые треки из high-conf
        for di in unmatched_d1:
            self.trackers.append({
                "id":        self._new_id(),
                "label":     high[di]["label"],
                "box":       high[di]["box"],
                "hits":      1,
                "no_losses": 0,
            })

        # Удаляем старые
        for ti in sorted(set(unmatched_t2), reverse=True):
            self.trackers[ti]["no_losses"] += 1

        self.trackers = [t for t in self.trackers if t["no_losses"] <= self.max_age]

        return [t for t in self.trackers if t["hits"] >= 1]


# ──────────────────────────────────────────────
# Подсчёт ID Switches
# ──────────────────────────────────────────────

def count_id_switches(history: dict) -> int:
    """
    history: {frame_id: [{"id": int, "box": [...], "label": int}]}
    ID Switch = объект (определяется по IoU > 0.5 с предыдущим кадром)
                сменил track_id.
    """
    switches = 0
    frames   = sorted(history.keys())

    for i in range(1, len(frames)):
        prev = history[frames[i-1]]
        curr = history[frames[i]]

        prev_by_id = {t["id"]: t["box"] for t in prev}

        for ct in curr:
            best_iou, best_id = 0.0, None
            for pid, pbox in prev_by_id.items():
                iou = box_iou(ct["box"], pbox)
                if iou > best_iou:
                    best_iou, best_id = iou, pid

            # Объект переместился к этому треку от другого ID
            if best_iou > 0.5 and best_id != ct["id"]:
                switches += 1

    return switches


# ──────────────────────────────────────────────
# Запуск на видео
# ──────────────────────────────────────────────

def run_tracking(video_path, weights_path, tracker_name="sort", conf=0.4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(weights_path, device)

    if tracker_name == "sort":
        tracker = SORT(max_age=5, min_hits=1, iou_thr=0.3)
    else:
        tracker = ByteTracker(high_thr=0.5, low_thr=0.2, max_age=10)

    cap = cv2.VideoCapture(video_path)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)

    out_path = f"videos/output/{Path(video_path).stem}_{tracker_name}.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    history   = {}
    frame_idx = 0
    id_colors = {}

    print(f"\n[{tracker_name.upper()}] {Path(video_path).name}")

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = process_frame(model, frame_rgb, device, conf)

        # Формат для трекера
        trk_input = [
            {"box": list(d["box"]), "label": d["label"], "score": d["score"]}
            for d in detections
        ]

        tracks = tracker.update(trk_input)

        # Рисуем сегментацию
        result = draw_detections(frame_bgr, detections)

        # Рисуем треки поверх
        frame_track_data = []
        for t in tracks:
            tid   = t["id"]
            label = t["label"]
            x1, y1, x2, y2 = [int(v) for v in t["box"]]

            if tid not in id_colors:
                np.random.seed(tid * 31)
                id_colors[tid] = tuple(np.random.randint(80, 255, 3).tolist())
            color = id_colors[tid]

            cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                result, f"ID:{tid} {CLASS_NAMES[label]}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

            frame_track_data.append({"id": tid, "box": [x1,y1,x2,y2], "label": label})

        history[frame_idx] = frame_track_data

        # Подпись трекера
        cv2.putText(
            result, f"{tracker_name.upper()} | frame {frame_idx}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )

        out.write(result)
        frame_idx += 1

    cap.release()
    out.release()

    id_sw = count_id_switches(history)
    total_tracks = max((t["id"] for f in history.values() for t in f), default=0)
    print(f"  Frames:      {frame_idx}")
    print(f"  Total IDs:   {total_tracks}")
    print(f"  ID Switches: {id_sw}")
    return id_sw, frame_idx


if __name__ == "__main__":
    import pip
    try:
        from filterpy.kalman import KalmanFilter
    except ImportError:
        print("Устанавливаю filterpy...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "filterpy"])

    WEIGHTS = "maskrcnn_signs_best.pth"
    Path("videos/output").mkdir(parents=True, exist_ok=True)

    videos = sorted(Path("videos").glob("*.mp4"))
    if not videos:
        print("Положи видео в папку videos/*.mp4")
    else:
        print("\n" + "="*55)
        print(f"{'Video':<25} {'Tracker':>10} {'ID Sw':>8} {'Frames':>8}")
        print("="*55)

        for v in videos:
            sw_sort, fr = run_tracking(str(v), WEIGHTS, "sort")
            sw_byte, _  = run_tracking(str(v), WEIGHTS, "bytetrack")
            print(f"{v.name:<25} {'SORT':>10} {sw_sort:>8} {fr:>8}")
            print(f"{'':<25} {'ByteTrack':>10} {sw_byte:>8}")

        print("="*55)
        print("Видео сохранены в videos/output/")