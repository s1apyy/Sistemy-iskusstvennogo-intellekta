from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = Path(__file__).resolve().parent / "results"
MODEL_PATH = SAVE_DIR / "model_lab1.pth"


def resolve_data_dir():
    marker = ROOT / "datasets" / "active_dataset.txt"
    if marker.exists():
        return Path(marker.read_text(encoding="utf-8").strip())
    for p in [ROOT / "datasets" / "cifar10", ROOT / "datasets" / "minishapes"]:
        if (p / "train").exists():
            return p
    return ROOT / "datasets" / "cifar10"


DATA_DIR = resolve_data_dir()


# та же архитектура, что в train.py (чтобы загрузить веса)
class SimpleCNN(nn.Module):
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


def denorm(img_tensor):
    # обратная нормализация для показа
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * 0.5 + 0.5
    return np.clip(img, 0, 1)


def main():
    if not MODEL_PATH.exists():
        print("Сначала обучите модель: python train.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    classes = ckpt["classes"]

    model = SimpleCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_ds = datasets.ImageFolder(str(DATA_DIR / "test"), transform=transform)
    loader = DataLoader(test_ds, batch_size=16, shuffle=True)

    images, labels = next(iter(loader))
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(1).cpu()
        conf = probs.max(1).values.cpu()

    # сетка 4x4
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(denorm(images[i]))
        true_c = classes[labels[i]]
        pred_c = classes[preds[i]]
        color = "green" if true_c == pred_c else "red"
        ax.set_title(f"ист: {true_c}\nпред: {pred_c} ({conf[i]:.2f})",
                     color=color, fontsize=8)
        ax.axis("off")

    plt.suptitle("Примеры классификации (зелёный = верно, красный = ошибка)", fontsize=11)
    plt.tight_layout()
    out = SAVE_DIR / "classification_examples.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print("Сохранено:", out)

    # ещё отдельно посчитаем accuracy на всём тесте
    correct = 0
    total = 0
    full_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    with torch.no_grad():
        for x, y in full_loader:
            x = x.to(device)
            pred = model(x).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"Accuracy на тесте: {correct / total:.3f}")


if __name__ == "__main__":
    main()
