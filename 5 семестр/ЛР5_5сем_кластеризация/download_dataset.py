from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "wiki4HE"
ZIP_URL = "https://archive.ics.uci.edu/static/public/334/wiki4he.zip"
CSV_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00334/wiki4HE.csv"
)


def _find_csv(base: Path) -> Path | None:
    direct = base / "wiki4HE.csv"
    if direct.exists():
        return direct
    matches = list(base.rglob("wiki4HE.csv"))
    return matches[0] if matches else None


def download() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = _find_csv(DATA_DIR)
    if existing is not None:
        print(f"Датасет уже на диске: {existing}")
        return existing

    zip_path = DATA_DIR / "wiki4he.zip"
    try:
        print(f"Скачивание: {ZIP_URL}")
        urlretrieve(ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
    except Exception as exc:  # noqa: BLE001
        print(f"ZIP не скачался ({exc}), пробуем прямой CSV...")
        csv_path = DATA_DIR / "wiki4HE.csv"
        urlretrieve(CSV_URL, csv_path)

    found = _find_csv(DATA_DIR)
    if found is None:
        raise RuntimeError(f"wiki4HE.csv не найден в {DATA_DIR}")

    print(f"Готово: {found}")
    return found


if __name__ == "__main__":
    download()
