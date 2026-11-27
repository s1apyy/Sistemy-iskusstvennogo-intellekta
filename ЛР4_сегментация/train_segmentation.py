import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "shapes"
SAVE_DIR = Path(__file__).resolve().parent / "results"
SAVE_DIR.mkdir(exist_ok=True)

NUM_CLASSES = 4  # фон + 3 фигуры
BATCH_SIZE = 16
EPOCHS = 8
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Устройство:", device)

CLASS_NAMES = ["фон", "круг", "квадрат", "треугольник"]
# цвета для визуализации масок
PALETTE = np.array([
    [0, 0, 0],
    [220, 60, 60],
    [60, 120, 220],
    [60, 180, 80],
], dtype=np.uint8)


class ShapesDataset(Dataset):
    def __init__(self, split="train"):
        self.img_dir = DATA_DIR / split / "images"
        self.mask_dir = DATA_DIR / split / "masks"
        self.files = sorted(self.img_dir.glob("*.png"))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx].name
        img = Image.open(self.img_dir / name).convert("RGB")
        mask = Image.open(self.mask_dir / name)

        img = self.to_tensor(img)  # [0,1]
        # небольшая нормализация "на глаз"
        img = (img - 0.5) / 0.5
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return img, mask


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Классический U-Net, но поменьше по каналам."""

    def __init__(self, n_classes=NUM_CLASSES):
        super().__init__()
        self.down1 = DoubleConv(3, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        bn = self.bottleneck(self.pool(d3))

        x = self.up3(bn)
        x = self.conv3(torch.cat([x, d3], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, d2], dim=1))
        x = self.up1(x)
        x = self.conv1(torch.cat([x, d1], dim=1))
        return self.out(x)


def pixel_accuracy(logits, masks):
    preds = logits.argmax(1)
    return (preds == masks).float().mean().item()


def mean_iou(logits, masks, n_classes=NUM_CLASSES):
    preds = logits.argmax(1)
    ious = []
    for c in range(n_classes):
        pred_c = preds == c
        true_c = masks == c
        inter = (pred_c & true_c).sum().item()
        union = (pred_c | true_c).sum().item()
        if union == 0:
            continue
        ious.append(inter / union)
    if not ious:
        return 0.0
    return sum(ious) / len(ious)


def colorize(mask_np):
    h, w = mask_np.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        out[mask_np == c] = PALETTE[c]
    return out


def visualize(model, loader, n=4):
    model.eval()
    imgs, masks = next(iter(loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        logits = model(imgs)
        preds = logits.argmax(1).cpu().numpy()

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for i in range(n):
        # denorm
        img = imgs[i].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * 0.5 + 0.5, 0, 1)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("изображение")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(colorize(masks[i].numpy()))
        axes[i, 1].set_title("маска (истина)")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(colorize(preds[i]))
        axes[i, 2].set_title("предсказание")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "segmentation_examples.png", dpi=130)
    plt.close()
    print("Визуализация:", SAVE_DIR / "segmentation_examples.png")


def train():
    if not (DATA_DIR / "train" / "images").exists():
        print("Сначала запустите prepare_dataset.py")
        return

    train_ds = ShapesDataset("train")
    val_ds = ShapesDataset("val")
    print("Train:", len(train_ds), "Val:", len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UNet().to(device)
    # фон часто доминирует — чуть уменьшим его вес
    weights = torch.tensor([0.4, 1.2, 1.2, 1.2], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_acc": [], "val_miou": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        n = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
            n += imgs.size(0)

        train_loss = running / n

        # валидация
        model.eval()
        accs, ious = [], []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                accs.append(pixel_accuracy(logits, masks))
                ious.append(mean_iou(logits, masks))

        val_acc = sum(accs) / len(accs)
        val_miou = sum(ious) / len(ious)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_miou"].append(val_miou)

        print(
            f"Epoch {epoch}/{EPOCHS} | loss={train_loss:.4f} | "
            f"val_pix_acc={val_acc:.3f} | val_mIoU={val_miou:.3f}"
        )

    torch.save(model.state_dict(), SAVE_DIR / "unet_shapes.pth")
    with open(SAVE_DIR / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"])
    ax[0].set_title("Train loss")
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(history["val_acc"], label="pixel acc")
    ax[1].plot(history["val_miou"], label="mIoU")
    ax[1].set_title("Validation")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "curves.png", dpi=120)
    plt.close()

    visualize(model, val_loader, n=4)
    print("Модель:", SAVE_DIR / "unet_shapes.pth")


if __name__ == "__main__":
    train()
