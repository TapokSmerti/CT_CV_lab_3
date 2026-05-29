# train.py — полная версия с прогресс-баром и сохранением лучшей модели

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

from dataset import RoadSignsDataset


def collate(batch):
    return tuple(zip(*batch))


train_ds = RoadSignsDataset("dataset/sign_dataset/train")

loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True,
    collate_fn=collate,
    num_workers=0
)

model = maskrcnn_resnet50_fpn(weights="DEFAULT")

num_classes = 9  # 8 классов + фон

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

device = "cuda"
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

epochs = 20
best_loss = float("inf")

for epoch in range(epochs):

    model.train()
    loss_sum = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

    for images, targets in pbar:

        images = [x.to(device) for x in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        losses = model(images, targets)

        loss_cls      = losses["loss_classifier"].item()
        loss_box      = losses["loss_box_reg"].item()
        loss_mask     = losses["loss_mask"].item()
        loss_obj      = losses["loss_objectness"].item()
        loss_rpn_box  = losses["loss_rpn_box_reg"].item()

        loss = sum(losses.values())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        loss_sum += loss.item()

        # Детальный прогресс-бар
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "cls":  f"{loss_cls:.3f}",
            "mask": f"{loss_mask:.3f}",
            "box":  f"{loss_box:.3f}",
        })

    scheduler.step()

    epoch_loss = loss_sum / len(loader)
    print(f"\nEpoch {epoch+1}: avg_loss={epoch_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    # Сохраняем лучшую модель
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "maskrcnn_signs_best.pth")
        print(f"  ✓ Saved best model (loss={best_loss:.4f})")

    # Чекпоинт каждые 5 эпох
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"maskrcnn_signs_epoch{epoch+1}.pth")

print("Training complete!")