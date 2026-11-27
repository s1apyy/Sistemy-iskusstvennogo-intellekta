import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = Path(__file__).resolve().parent / "results"
SAVE_DIR.mkdir(exist_ok=True)


def resolve_data_dir():
    marker = ROOT / "datasets" / "active_dataset.txt"
    if marker.exists():
        return Path(marker.read_text(encoding="utf-8").strip())
    # запасные варианты
    for p in [ROOT / "datasets" / "cifar10", ROOT / "datasets" / "minishapes"]:
        if (p / "train").exists():
            return p
    return ROOT / "datasets" / "cifar10"


DATA_DIR = resolve_data_dir()

# гиперпараметры (подобрал примерно, чтобы училось не слишком долго)
BATCH_SIZE = 64
EPOCHS = 5  # без аугментации побыстрее, в ЛР2 сравним
LR = 0.001
IMG_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Устройство:", device)


class SimpleCNN(nn.Module):
    """Небольшая свёрточная сеть — без ResNet и прочего."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_loaders():
    global DATA_DIR
    DATA_DIR = resolve_data_dir()
    print("Датасет:", DATA_DIR)

    # без аугментации — только нормализация
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_ds = datasets.ImageFolder(str(DATA_DIR / "train"), transform=transform)
    test_ds = datasets.ImageFolder(str(DATA_DIR / "test"), transform=transform)

    print("Классы:", train_ds.classes)
    print("Train:", len(train_ds), "Test:", len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader, train_ds.classes


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total


def train():
    if not (DATA_DIR / "train").exists():
        print("Датасет не найден. Сначала запустите download_dataset.py")
        return

    train_loader, test_loader, classes = get_loaders()
    model = SimpleCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, test_loader)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"test_loss={test_loss:.4f} acc={test_acc:.3f}"
        )

    # сохраняем модель и историю
    torch.save({
        "model_state": model.state_dict(),
        "classes": classes,
    }, SAVE_DIR / "model_lab1.pth")

    with open(SAVE_DIR / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # графики обучения
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["test_loss"], label="test")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["test_acc"], label="test")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "training_curves.png", dpi=120)
    plt.close()
    print("Модель сохранена в", SAVE_DIR / "model_lab1.pth")


if __name__ == "__main__":
    train()
