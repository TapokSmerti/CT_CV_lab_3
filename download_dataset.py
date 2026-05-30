# download_roboflow.py
from roboflow import Roboflow

rf = Roboflow(api_key="lPpWX1mnE2k3R4I1YpAR")  # из настроек аккаунта на roboflow.com

project = rf.workspace("buda-vampilov").project("russian-road-signs-m4lzc")
dataset = project.version(1).download(
    "yolov8",                    # формат для YOLOv11-seg
    location="dataset_roboflow"
)
print("Скачано в:", dataset.location)