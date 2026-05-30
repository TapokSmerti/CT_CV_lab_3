# CT_CV_lab_3 — Road Sign Segmentation & Tracking

Сегментация дорожных знаков на видео с использованием YOLOv11-seg,
оценка качества по IoU/Precision/Recall/L2, трекинг объектов (ByteTrack и BotSORT).

---

## Структура проекта

```
CT_CV_lab_3/
├── dataset/                  # датасет (в .gitignore)
│   └── rrs/
│       ├── train/images/
│       ├── train/labels/
│       ├── valid/images/
│       ├── valid/labels/
│       └── data.yaml
├── videos/                   # исходные видео для тестирования
│   └── output/               # сюда сохраняются результаты
├── runs/                     # веса моделей после обучения
├── train.py                  # обучение YOLOv11-seg
├── evaluate.py               # метрики IoU/Precision/Recall/L2
├── inference_video.py        # инференс на видео
├── tracking.py               # трекинг ByteTrack + BotSORT
├── download_dataset.py       # скачать датасет с Roboflow
├── requirements.txt
└── README.md
```

---

## Быстрый старт

### 1. Установка зависимостей

```bash
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

`requirements.txt`:
```
ultralytics
roboflow
torch
torchvision
numpy
opencv-python
tqdm
pillow
matplotlib
pandas
filterpy
scipy
pyyaml
```

### 2. Скачать датасет

Зарегистрируйся на [roboflow.com](https://roboflow.com), получи API key в настройках аккаунта,
вставь его в `download_dataset.py` и запусти:

```bash
python3 download_dataset.py
```

Датасет скачается в `dataset/rrs/`. Убедись что `dataset/data.yaml` указывает на правильные пути.

### 3. Обучение модели

```bash
python3 train.py
```

Веса сохранятся в `runs/seg/yolo11_signs_v2/weights/best.pt`.

Параметры обучения (`train.py`):
- модель: `yolo11s-seg`
- 50 эпох, `imgsz=1280`, `batch=8`
- аугментации: mosaic, copy_paste, scale, degrees, mixup

### 4. Оценка качества на валидационной выборке

```bash
python3 evaluate.py
```

Выводит таблицу метрик по классам:

```
================================================================
Class            mIoU    Prec     Rec   L2(px)      N
================================================================
road sign       0.731   0.812   0.774     6.2    412
================================================================
mIoU overall    0.731

IoU threshold coverage (% images with at least one match):
  IoU >= 0.5:  88.4%  (354/400)
  IoU >= 0.75: 71.2%  (285/400)
  IoU >= 0.9:  39.5%  (158/400)
```

Перед запуском проверь пути в начале файла:
```python
WEIGHTS   = "runs/segment/runs/seg/yolo11_signs_v2/weights/best.pt"
DATA_YAML = "dataset/data.yaml"
VAL_DIR   = "dataset/rrs/valid"
```

### 5. Инференс на видео

Положи видео в папку `videos/` и запусти:

```bash
python3 inference_video.py
```

Результаты сохранятся в `videos/output/<имя_видео>/`.

### 6. Трекинг и подсчёт ID Switches

```bash
python3 tracking.py
```

Запускает два трекера (ByteTrack и BotSORT) на всех видео из `videos/`
и выводит итоговую таблицу:

```
======================================================
Video                                  Tracker    ID Sw    IDs
======================================================
video1.mp4                             bytetrack      3     12
                                       botsort        1     12
video2.mp4                             bytetrack      5     18
                                       botsort        2     18
======================================================
```

Результирующие видео сохраняются в `videos/output/`.

---

## Описание метрик

| Метрика | Описание |
|---|---|
| mIoU | Intersection over Union между предсказанной и GT маской |
| Precision | TP / (TP + FP) по пикселям маски |
| Recall | TP / (TP + FN) по пикселям маски |
| L2 | Расстояние между центроидами предсказанной и GT маски (пиксели) |
| IoU >= t | % изображений, на которых хотя бы один объект имеет IoU >= t |
| ID Switches | Число смен track_id у одного объекта между соседними кадрами |

---

## Используемые модели и алгоритмы

- **YOLOv11s-seg** — основная модель сегментации, дообученная на датасете Russian Road Signs
- **ByteTrack** — двухэтапное сопоставление детекций (high + low confidence)
- **BotSORT** — ByteTrack + ReID признаки, как правило даёт меньше ID Switches

---

## Требования к железу

- GPU: рекомендуется NVIDIA с 8+ GB VRAM (обучение при `imgsz=1280`)
- CPU-инференс работает, но медленно (~2–3 fps)
- RAM: 16+ GB

---

## Датасет

[Russian Road Signs — Roboflow Universe](https://universe.roboflow.com/buda-vampilov/russian-road-signs-m4lzc)

~2000 изображений дорожных сцен с разметкой сегментации знаков в формате YOLO-seg.