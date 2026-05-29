from ultralytics import YOLO

def main():

    # предобученная segmentation модель
    model = YOLO("yolo11n-seg.pt")

    model.train(
        data="data.yaml",

        epochs=100,
        imgsz=640,
        batch=16,

        device=0,

        workers=8,

        optimizer="AdamW",

        lr0=1e-3,

        project="runs",
        name="russian_signs_seg",

        patience=20,

        save=True,
        save_period=10,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=10,
        translate=0.1,
        scale=0.3,
        shear=2.0,

        mosaic=1.0,
        mixup=0.1,

        pretrained=True
    )

if __name__ == "__main__":
    main()