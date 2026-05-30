# evaluate_yolo.py
from ultralytics import YOLO

model = YOLO(" runs/segment/runs/seg/yolo11_signs/weights/best.pt")

metrics = model.val(
    data="dataset/data.yaml",
    split="val",
    conf=0.35,
    iou=0.5,
)

print(f"mAP50:      {metrics.seg.map50:.4f}")
print(f"mAP50-95:   {metrics.seg.map:.4f}")
print(f"Precision:  {metrics.seg.mp:.4f}")
print(f"Recall:     {metrics.seg.mr:.4f}")

# Per-class IoU
for i, name in enumerate(model.names.values()):
    print(f"  {name}: AP={metrics.seg.ap[i]:.3f}")