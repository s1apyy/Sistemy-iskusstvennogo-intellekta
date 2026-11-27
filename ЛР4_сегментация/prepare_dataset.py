import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / "shapes"
IMG_SIZE = 128
N_TRAIN = 400
N_VAL = 80

# 0 = фон, 1 = круг, 2 = квадрат, 3 = треугольник
CLASS_COLORS = {
    1: (220, 60, 60),
    2: (60, 120, 220),
    3: (60, 180, 80),
}


def rand_color():
    return tuple(random.randint(40, 220) for _ in range(3))


def draw_one(draw_img, draw_mask, cls):
    """Рисуем одну фигуру на картинке и на маске."""
    margin = 15
    x1 = random.randint(margin, IMG_SIZE - 50)
    y1 = random.randint(margin, IMG_SIZE - 50)
    size = random.randint(30, 55)
    x2, y2 = x1 + size, y1 + size
    color = CLASS_COLORS[cls]

    if cls == 1:  # круг
        draw_img.ellipse([x1, y1, x2, y2], fill=color)
        draw_mask.ellipse([x1, y1, x2, y2], fill=cls)
    elif cls == 2:  # квадрат
        draw_img.rectangle([x1, y1, x2, y2], fill=color)
        draw_mask.rectangle([x1, y1, x2, y2], fill=cls)
    else:  # треугольник
        pts = [(x1, y2), (x2, y2), ((x1 + x2) // 2, y1)]
        draw_img.polygon(pts, fill=color)
        draw_mask.polygon(pts, fill=cls)


def make_sample():
    # фон шумный + иногда градиент, чтобы модель не читерила по цвету фона
    base = random.randint(150, 220)
    bg = np.random.randint(base - 40, base + 40, (IMG_SIZE, IMG_SIZE, 3), dtype=np.int16)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    img = Image.fromarray(bg)
    mask = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)

    d_img = ImageDraw.Draw(img)
    d_mask = ImageDraw.Draw(mask)

    # 2-4 фигуры, чаще перекрываются — сегментация сложнее
    n_shapes = random.randint(2, 4)
    for _ in range(n_shapes):
        cls = random.randint(1, 3)
        draw_one(d_img, d_mask, cls)

    # шум поверх картинки (маску не трогаем)
    arr = np.array(img, dtype=np.int16)
    arr = np.clip(arr + np.random.randint(-20, 21, arr.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img, mask


def save_split(split, n):
    img_dir = OUT / split / "images"
    mask_dir = OUT / split / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        img, mask = make_sample()
        img.save(img_dir / f"{i:04d}.png")
        mask.save(mask_dir / f"{i:04d}.png")
    print(f"{split}: {n} пар image/mask -> {OUT / split}")


def main():
    random.seed(42)
    np.random.seed(42)

    # всегда пересобираем — датасет лёгкий
    save_split("train", N_TRAIN)
    save_split("val", N_VAL)
    print("Готово.")


if __name__ == "__main__":
    main()
