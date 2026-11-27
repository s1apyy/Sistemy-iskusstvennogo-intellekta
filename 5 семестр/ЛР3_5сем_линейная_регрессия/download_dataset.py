from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "BikeSharing"
ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
)
MIRROR_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/"
    "Bike-Sharing-Dataset.zip"
)


def _has_csv(root: Path) -> bool:
    return (root / "day.csv").exists() or (root / "hour.csv").exists()


def _find_root(base: Path) -> Path | None:
    if _has_csv(base):
        return base
    for child in base.iterdir():
        if child.is_dir() and _has_csv(child):
            return child
    return None


def download() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = _find_root(DATA_DIR)
    if existing is not None:
        print(f"Датасет уже на диске: {existing}")
        return existing

    zip_path = DATA_DIR / "Bike-Sharing-Dataset.zip"
    for url in (ZIP_URL, MIRROR_URL):
        try:
            print(f"Скачивание: {url}")
            urlretrieve(url, zip_path)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"Не удалось скачать с {url}: {exc}")
    else:
        raise RuntimeError(
            "Не удалось скачать датасет. Скачайте Bike-Sharing-Dataset.zip "
            f"вручную и распакуйте в {DATA_DIR}"
        )

    print(f"Распаковка в {DATA_DIR} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    root = _find_root(DATA_DIR)
    if root is None:
        raise RuntimeError(f"После распаковки не найдены CSV в {DATA_DIR}")

    print(f"Готово: {root}")
    return root


if __name__ == "__main__":
    download()
