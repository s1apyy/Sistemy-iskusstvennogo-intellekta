from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "HAPT"
ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/341/"
    "smartphone+based+recognition+of+human+activities+and+postural+transitions.zip"
)
MIRROR_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00341/"
    "HAPT%20Data%20Set.zip"
)


def _has_train_test(root: Path) -> bool:
    return (root / "Train" / "X_train.txt").exists() and (
        root / "Test" / "X_test.txt"
    ).exists()


def _find_dataset_root(base: Path) -> Path | None:
    if _has_train_test(base):
        return base
    for child in base.iterdir():
        if child.is_dir() and _has_train_test(child):
            return child
    return None


def download() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = _find_dataset_root(DATA_DIR)
    if existing is not None:
        print(f"Датасет уже на диске: {existing}")
        return existing

    zip_path = DATA_DIR / "HAPT_Data_Set.zip"
    for url in (ZIP_URL, MIRROR_URL):
        try:
            print(f"Скачивание: {url}")
            urlretrieve(url, zip_path)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"Не удалось скачать с {url}: {exc}")
    else:
        raise RuntimeError(
            "Не удалось скачать датасет. Скачайте вручную HAPT Data Set.zip "
            f"и распакуйте в {DATA_DIR}"
        )

    print(f"Распаковка в {DATA_DIR} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    root = _find_dataset_root(DATA_DIR)
    if root is None:
        raise RuntimeError(f"После распаковки не найдены Train/Test в {DATA_DIR}")

    print(f"Готово: {root}")
    return root


if __name__ == "__main__":
    download()
