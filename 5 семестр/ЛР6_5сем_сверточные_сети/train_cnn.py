"""
ЛР6. Свёрточные сети (Keras)

1. Загрузка обучающей выборки с диска
2. Создание CNN
3. Обучение
4. Сохранение модели в файл
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from download_dataset import load_from_disk

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
MODEL_PATH = RESULTS / "cnn_cifar10.keras"
NUM_CLASSES = 10
EPOCHS = 12
BATCH_SIZE = 64


def preprocess(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
):
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    return x_train, y_train, x_test, y_test


def build_cnn(input_shape=(32, 32, 3)) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="cifar10_cnn",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["accuracy"], label="train")
    axes[0].plot(history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["loss"], label="train")
    axes[1].plot(history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("=== Загрузка CIFAR-10 с диска ===")
    x_train, y_train, x_test, y_test, class_names = load_from_disk()
    print(f"Классы: {class_names}")
    print(f"train={x_train.shape}, test={x_test.shape}")

    x_train, y_train, x_test, y_test = preprocess(x_train, y_train, x_test, y_test)

    print("\n=== Создание модели CNN ===")
    model = build_cnn()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    print("\n=== Обучение ===")
    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # гарантированно сохраняем финальную (best weights уже в model)
    model.save(MODEL_PATH)
    print(f"\nМодель сохранена: {MODEL_PATH}")

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")

    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    plot_history(history, RESULTS / "training_curves.png")

    summary = {
        "dataset": "CIFAR-10",
        "model_file": str(MODEL_PATH.name),
        "epochs_ran": len(history["loss"]),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "class_names": class_names,
    }
    with (RESULTS / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    (RESULTS / "report_notes.txt").write_text(
        "\n".join(
            [
                "ЛР6. Свёрточная сеть (Keras) — CIFAR-10",
                f"Модель: {MODEL_PATH.name}",
                f"Test accuracy: {test_acc:.4f}",
                "Проверка: python check_model.py",
                "Проверка изображения: python check_model.py path/to/image.png",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Кривые обучения: {RESULTS / 'training_curves.png'}")


if __name__ == "__main__":
    main()
