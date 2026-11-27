"""
Загрузка CIFAR-10 на диск (вариант для ЛР по свёрточным сетям).

Данные сохраняются в datasets/cifar10/ как .npz массивы,
чтобы обучение читало выборку с диска, а не только из кеша Keras.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "cifar10"
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _ready(root: Path) -> bool:
    needed = [
        "x_train.npy",
        "y_train.npy",
        "x_test.npy",
        "y_test.npy",
        "class_names.txt",
    ]
    return all((root / name).exists() for name in needed)


def download() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if _ready(DATA_DIR):
        print(f"Датасет уже на диске: {DATA_DIR}")
        return DATA_DIR

    # Keras скачает во внутренний кеш, затем мы сохраним копию в проект
    from tensorflow.keras.datasets import cifar10

    print("Скачивание CIFAR-10 через Keras...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    np.save(DATA_DIR / "x_train.npy", x_train)
    np.save(DATA_DIR / "y_train.npy", y_train)
    np.save(DATA_DIR / "x_test.npy", x_test)
    np.save(DATA_DIR / "y_test.npy", y_test)
    (DATA_DIR / "class_names.txt").write_text(
        "\n".join(CLASS_NAMES), encoding="utf-8"
    )

    print(f"Сохранено в {DATA_DIR}")
    print(f"train={x_train.shape}, test={x_test.shape}")
    return DATA_DIR


def load_from_disk(
    root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    root = root or download()
    if not _ready(root):
        root = download()

    x_train = np.load(root / "x_train.npy")
    y_train = np.load(root / "y_train.npy")
    x_test = np.load(root / "x_test.npy")
    y_test = np.load(root / "y_test.npy")
    names = (root / "class_names.txt").read_text(encoding="utf-8").strip().splitlines()
    return x_train, y_train, x_test, y_test, names


if __name__ == "__main__":
    download()
