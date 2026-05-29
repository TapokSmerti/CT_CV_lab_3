import os
import zipfile

dataset = "viacheslavshalamov/russian-road-signs-segmentation-dataset"

os.system(
    f'kaggle datasets download -d {dataset} -p data'
)

zip_path = "dataset/russian-road-signs-segmentation-dataset.zip"

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall("dataset")

print("Dataset downloaded")