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
    for p in [ROOT / "datasets" / "cifar10", ROOT / "datasets" / "minishapes"]:
        if (p / "train").exists():
            return p
    return ROOT / "datasets" / "cifar10"


DATA_DIR = resolve_data_dir()

# чуть больше эпох, т.к. с аугментацией обычно дольше сходится
BATCH_SIZE = 64
EPOCHS = 12
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Устройство:", device)


class SimpleCNN(nn.Module):
    # та же сеть что в ЛР1, чтобы сравнение было честным
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_loaders():
    global DATA_DIR
    DATA_DIR = resolve_data_dir()
    print("Датасет:", DATA_DIR)

    # --- аугментация только для train ---
    train_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # тест без аугментации!
    test_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_ds = datasets.ImageFolder(str(DATA_DIR / "train"), transform=train_tf)
    test_ds = datasets.ImageFolder(str(DATA_DIR / "test"), transform=test_tf)

    print("Классы:", train_ds.classes)
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
            pred = outputs.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total


def show_augmentation_examples(classes):
    """Показать как выглядят аугментированные картинки."""
    train_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.ToTensor(),
    ])
    # без нормализации, чтобы было видно
    ds = datasets.ImageFolder(str(DATA_DIR / "train"), transform=train_tf)

    fig, axes = plt.subplots(2, 5, figsize=(11, 5))
    for ax in axes.flat:
        img, label = ds[np_rand_idx(len(ds))]
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(classes[label], fontsize=9)
        ax.axis("off")
    plt.suptitle("Примеры аугментации (каждый раз картинка чуть другая)")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "augmentation_examples.png", dpi=120)
    plt.close()
    print("Примеры аугментации сохранены")


def np_rand_idx(n):
    import random
    return random.randint(0, n - 1)


def train():
    if not (DATA_DIR / "train").exists():
        print("Нет датасета. Сначала: python ../ЛР1_классификация/download_dataset.py")
        return

    train_loader, test_loader, classes = get_loaders()
    show_augmentation_examples(classes)

    model = SimpleCNN(len(classes)).to(device)
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
            pred = outputs.argmax(1)
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

    torch.save({"model_state": model.state_dict(), "classes": classes},
               SAVE_DIR / "model_lab2.pth")
    with open(SAVE_DIR / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # сравнение с ЛР1 если есть история
    lab1_hist = ROOT / "ЛР1_классификация" / "results" / "history.json"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(history["test_acc"], label="ЛР2 (с аугментацией)")
    if lab1_hist.exists():
        with open(lab1_hist, encoding="utf-8") as f:
            h1 = json.load(f)
        ax[0].plot(h1["test_acc"], label="ЛР1 (без аугментации)")
    ax[0].set_title("Test accuracy")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(history["train_loss"], label="train")
    ax[1].plot(history["test_loss"], label="test")
    ax[1].set_title("Loss (ЛР2)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "comparison.png", dpi=120)
    plt.close()

    print("\nИтоговая test accuracy (с аугментацией):", round(history["test_acc"][-1], 3))
    if lab1_hist.exists():
        print("Для сравнения ЛР1 test accuracy:", round(h1["test_acc"][-1], 3))
    print("Результаты в", SAVE_DIR)


if __name__ == "__main__":
    train()
