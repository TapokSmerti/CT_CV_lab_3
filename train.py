import torch

from torch.utils.data import DataLoader

from dataset import RoadSignsDataset

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn
)

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)

from torchvision.models.detection.mask_rcnn import (
    MaskRCNNPredictor
)


def collate(batch):
    return tuple(zip(*batch))


train_ds = RoadSignsDataset(
    "dataset/sign_dataset/train"
)

loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
    collate_fn=collate,
    num_workers=4
)

model = maskrcnn_resnet50_fpn(
    weights="DEFAULT"
)

num_classes = 9

in_features = (
    model.roi_heads.box_predictor
    .cls_score.in_features
)

model.roi_heads.box_predictor = (
    FastRCNNPredictor(
        in_features,
        num_classes
    )
)

in_features_mask = (
    model.roi_heads.mask_predictor
    .conv5_mask.in_channels
)

model.roi_heads.mask_predictor = (
    MaskRCNNPredictor(
        in_features_mask,
        256,
        num_classes
    )
)

device = "cuda"

model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

epochs = 20

for epoch in range(epochs):

    model.train()

    loss_sum = 0

    for images, targets in loader:

        images = [
            x.to(device)
            for x in images
        ]

        targets = [
            {
                k: v.to(device)
                for k,v in t.items()
            }
            for t in targets
        ]

        losses = model(
            images,
            targets
        )

        loss = sum(
            losses.values()
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        loss_sum += loss.item()

    print(
        f"epoch {epoch}:",
        loss_sum / len(loader)
    )

torch.save(
    model.state_dict(),
    "maskrcnn_signs.pth"
)