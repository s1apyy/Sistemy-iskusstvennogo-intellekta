import random
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "datasets" / "cifar10"
CACHE = ROOT / "datasets" / "_cifar_cache"
FALLBACK_DIR = ROOT / "datasets" / "minishapes"

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_TGZ = CACHE / "cifar-10-python.tar.gz"

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

FALLBACK_CLASSES = ["circle", "square", "triangle", "cross", "line"]


def try_download_cifar():
    CACHE.mkdir(parents=True, exist_ok=True)
    if not CIFAR_TGZ.exists() or CIFAR_TGZ.stat().st_size < 1_000_000:
        print("Качаю CIFAR-10...")
        last_err = None
        for attempt in range(1, 4):
            try:
                urllib.request.urlretrieve(CIFAR_URL, CIFAR_TGZ)
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"  попытка {attempt} не вышла: {e}")
        if last_err is not None:
            raise last_err

    from torchvision.datasets import CIFAR10
    train_set = CIFAR10(root=str(CACHE), train=True, download=True)
    test_set = CIFAR10(root=str(CACHE), train=False, download=True)
    return train_set, test_set


def save_cifar_split(dataset, split_name: str):
    for name in CLASSES:
        (OUT_DIR / split_name / name).mkdir(parents=True, exist_ok=True)

    print(f"Сохраняю {split_name} на диск...")
    for idx, (img, label) in enumerate(tqdm(dataset)):
        path = OUT_DIR / split_name / CLASSES[label] / f"{idx:05d}.png"
        img.save(path)


def make_shape(cls_name, size=64):
    # пёстрый фон
    bg = np.random.randint(30, 120, (size, size, 3), dtype=np.uint8)
    img = Image.fromarray(bg)
    d = ImageDraw.Draw(img)
    color = tuple(int(x) for x in np.random.randint(100, 255, 3))
    m = random.randint(6, 16)
    thick = random.randint(3, 8)

    # иногда рисуем "отвлекающую" точку
    if random.random() < 0.35:
        ox, oy = random.randint(0, size - 8), random.randint(0, size - 8)
        d.ellipse([ox, oy, ox + 8, oy + 8], fill=tuple(np.random.randint(50, 200, 3)))

    if cls_name == "circle":
        # иногда почти эллипс — путается с другими
        pad = random.randint(0, 8)
        d.ellipse([m, m + pad, size - m, size - m - pad], outline=color, width=thick)
        if random.random() < 0.5:
            d.ellipse([m + 4, m + 4 + pad, size - m - 4, size - m - 4 - pad], fill=color)
    elif cls_name == "square":
        d.rectangle([m, m, size - m, size - m], outline=color, width=thick)
        if random.random() < 0.45:
            d.rectangle([m + 5, m + 5, size - m - 5, size - m - 5], fill=color)
    elif cls_name == "triangle":
        pts = [(size // 2, m), (size - m, size - m), (m, size - m)]
        if random.random() < 0.5:
            d.polygon(pts, outline=color)
            d.line(pts + [pts[0]], fill=color, width=thick)
        else:
            d.polygon(pts, fill=color)
    elif cls_name == "cross":
        w = size // random.choice([5, 6, 7, 8])
        d.rectangle([size // 2 - w, m, size // 2 + w, size - m], fill=color)
        d.rectangle([m, size // 2 - w, size - m, size // 2 + w], fill=color)
        # иногда чуть кривой крест
        if random.random() < 0.3:
            d.line([(m, m), (size - m, size - m)], fill=color, width=2)
    else:  # line — может быть почти как сторона квадрата
        if random.random() < 0.5:
            d.line([(m, size - m), (size - m, m)], fill=color, width=thick + 2)
        else:
            y = random.randint(size // 3, 2 * size // 3)
            d.line([(m, y), (size - m, y + random.randint(-10, 10))], fill=color, width=thick + 2)

    # поворот
    angle = random.uniform(-35, 35)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=tuple(bg[0, 0]))
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))

    arr = np.array(img, dtype=np.int16)
    noise = np.random.randint(-28, 29, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def build_fallback(n_train=1000, n_test=250, force=False):
    if force and FALLBACK_DIR.exists():
        shutil.rmtree(FALLBACK_DIR)

    print("Собираю локальный датасет MiniShapes (с шумом/поворотами)")
    for split, n in [("train", n_train), ("test", n_test)]:
        per_class = n // len(FALLBACK_CLASSES)
        for cls in FALLBACK_CLASSES:
            out = FALLBACK_DIR / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                make_shape(cls).save(out / f"{i:04d}.png")
    set_active(FALLBACK_DIR)
    print(f"Готово: {FALLBACK_DIR}")
    print("Классы:", ", ".join(FALLBACK_CLASSES))


def set_active(path: Path):
    marker = ROOT / "datasets" / "active_dataset.txt"
    marker.write_text(str(path), encoding="utf-8")


def main():
    random.seed(42)
    np.random.seed(42)

    if (OUT_DIR / "train" / "airplane").exists():
        n = len(list((OUT_DIR / "train" / "airplane").glob("*.png")))
        if n > 100:
            print(f"CIFAR-10 уже есть: {OUT_DIR}")
            set_active(OUT_DIR)
            return

    try:
        train_set, test_set = try_download_cifar()
        save_cifar_split(train_set, "train")
        save_cifar_split(test_set, "test")
        set_active(OUT_DIR)
        print(f"CIFAR-10 сохранён в {OUT_DIR}")
    except Exception as e:
        print("Не удалось скачать CIFAR-10:", e)
        # пересобираем fallback всегда заново, если cifar нет
        build_fallback(force=True)


if __name__ == "__main__":
    main()
