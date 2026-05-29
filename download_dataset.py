import os
import zipfile
import kaggle

dataset = "viacheslavshalamov/russian-road-signs-segmentation-dataset"

os.makedirs("dataset", exist_ok=True)

kaggle.api.dataset_download_files(dataset, path="dataset", unzip=True)

print("Dataset downloaded and extracted to 'dataset' folder")

