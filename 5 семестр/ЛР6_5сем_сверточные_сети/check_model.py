"""
Дополнительная программа ЛР6.

Загружает обученную CNN из файла и позволяет:
1) оценить точность на тестовой выборке с диска;
2) классифицировать одно изображение (файл PNG/JPG).

Примеры:
  python check_model.py
  python check_model.py path/to/image.png
  python check_model.py --image path/to/image.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

from download_dataset import CLASS_NAMES, load_from_disk

RESULTS = Path(__file__).resolve().parent / "results"
MODEL_PATH = RESULTS / "cnn_cifar10.keras"


def load_model(path: Path = MODEL_PATH) -> keras.Model:
    if not path.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {path}\nСначала запустите: python train_cnn.py"
        )
    print(f"Загрузка модели: {path}")
    return keras.models.load_model(path)


def evaluate_on_test(model: keras.Model) -> tuple[float, float]:
    print("Загрузка тестовой выборки с диска...")
    _, _, x_test, y_test, class_names = load_from_disk()
    x_test = x_test.astype("float32") / 255.0
    y_cat = keras.utils.to_categorical(y_test, 10)
    loss, acc = model.evaluate(x_test, y_cat, verbose=0)
    print(f"Точность на test: {acc:.4f} (loss={loss:.4f})")

    # несколько случайных примеров
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x_test), size=8, replace=False)
    preds = model.predict(x_test[idx], verbose=0)
    print("\nПримеры предсказаний:")
    for i, p in zip(idx, preds):
        true_id = int(y_test[i][0]) if y_test.ndim > 1 else int(y_test[i])
        pred_id = int(np.argmax(p))
        conf = float(p[pred_id])
        mark = "OK" if pred_id == true_id else "ERR"
        print(
            f"  [{mark}] true={class_names[true_id]:12s}  "
            f"pred={class_names[pred_id]:12s}  conf={conf:.3f}"
        )
    return float(loss), float(acc)


def predict_image(model: keras.Model, image_path: Path) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Файл не найден: {image_path}")

    img = Image.open(image_path).convert("RGB").resize((32, 32))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    pred_id = int(np.argmax(probs))
    print(f"\nФайл: {image_path}")
    print(f"Класс: {CLASS_NAMES[pred_id]}  (вероятность {probs[pred_id]:.4f})")
    print("Топ-5:")
    for i in np.argsort(probs)[::-1][:5]:
        print(f"  {CLASS_NAMES[i]:12s}  {probs[i]:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Проверка сохранённой CNN (CIFAR-10)"
    )
    p.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Путь к изображению для классификации",
    )
    p.add_argument("--image", dest="image_opt", default=None, help="То же, что positional")
    p.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="Путь к файлу модели .keras",
    )
    p.add_argument(
        "--skip-test",
        action="store_true",
        help="Не считать accuracy на test, только predict по файлу",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image_opt or args.image
    model = load_model(Path(args.model))

    if not args.skip_test and image_path is None:
        evaluate_on_test(model)
        print(
            "\nЧтобы проверить своё изображение:\n"
            "  python check_model.py path\\to\\image.png"
        )
        return

    if image_path is None:
        print("Укажите путь к изображению.", file=sys.stderr)
        sys.exit(1)

    if not args.skip_test:
        # при наличии файла всё равно можно быстро глянуть test — опционально пропускаем
        pass
    predict_image(model, Path(image_path))


if __name__ == "__main__":
    main()
