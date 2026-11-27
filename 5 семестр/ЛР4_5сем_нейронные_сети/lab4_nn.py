from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from download_dataset import DATA_DIR, download

warnings.filterwarnings("ignore")

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def load_hapt(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Загрузка уже разделённых Train/Test из HAPT."""
    x_train = np.loadtxt(root / "Train" / "X_train.txt")
    y_train = np.loadtxt(root / "Train" / "y_train.txt").astype(int)
    x_test = np.loadtxt(root / "Test" / "X_test.txt")
    y_test = np.loadtxt(root / "Test" / "y_test.txt").astype(int)

    labels_path = root / "activity_labels.txt"
    activity_names: list[str] = []
    if labels_path.exists():
        with labels_path.open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    activity_names.append(parts[1])
    return x_train, y_train, x_test, y_test, activity_names


def make_splits(
    x_train_full: np.ndarray,
    y_train_full: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    val_size: float = 0.2,
) -> dict[str, np.ndarray]:
    """
    В датасете есть train/test (~70/30 по субъектам).
    Validation выделяем из обучающей выборки.
    """
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=val_size,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )
    return {
        "X_train": x_train,
        "y_train": y_train,
        "X_val": x_val,
        "y_val": y_val,
        "X_test": x_test,
        "y_test": y_test,
    }


def scale_features(
    x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Масштабирование: fit только на train, transform на val/test."""
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)
    return x_train_s, x_val_s, x_test_s, scaler


def evaluate(model, x: np.ndarray, y: np.ndarray) -> float:
    return float(accuracy_score(y, model.predict(x)))


def train_baseline(data: dict[str, np.ndarray]) -> dict:
    """Базовое обучение Perceptron и MLPClassifier."""
    print("\n=== 3–4. Базовое обучение Perceptron и MLPClassifier ===")

    perc = Perceptron(max_iter=1000, eta0=1.0, random_state=RANDOM_STATE)
    perc.fit(data["X_train"], data["y_train"])
    perc_scores = {
        "train": evaluate(perc, data["X_train"], data["y_train"]),
        "val": evaluate(perc, data["X_val"], data["y_val"]),
        "test": evaluate(perc, data["X_test"], data["y_test"]),
    }
    print(
        f"Perceptron  — train={perc_scores['train']:.4f}, "
        f"val={perc_scores['val']:.4f}, test={perc_scores['test']:.4f}"
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=300,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
    )
    mlp.fit(data["X_train"], data["y_train"])
    mlp_scores = {
        "train": evaluate(mlp, data["X_train"], data["y_train"]),
        "val": evaluate(mlp, data["X_val"], data["y_val"]),
        "test": evaluate(mlp, data["X_test"], data["y_test"]),
    }
    print(
        f"MLPClassifier — train={mlp_scores['train']:.4f}, "
        f"val={mlp_scores['val']:.4f}, test={mlp_scores['test']:.4f}"
    )

    y_pred = mlp.predict(data["X_test"])
    report = classification_report(data["y_test"], y_pred, digits=4)
    print("\nClassification report (MLP, test):\n", report)
    (RESULTS / "classification_report_mlp.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(data["y_test"], y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion matrix — MLPClassifier (test)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix_mlp.png", dpi=150)
    plt.close(fig)

    return {"perceptron": perc_scores, "mlp": mlp_scores, "mlp_model": mlp}


def experiment_learning_rate(data: dict[str, np.ndarray]) -> dict:
    """Эксперимент: коэффициент обучения (learning_rate_init / eta0)."""
    print("\n=== 5.1 Эксперимент: коэффициент обучения ===")
    rates = [0.0001, 0.001, 0.01, 0.1, 1.0]
    mlp_val, mlp_test = [], []
    perc_val, perc_test = [], []

    for lr in rates:
        mlp = MLPClassifier(
            hidden_layer_sizes=(100,),
            solver="adam",
            learning_rate_init=lr,
            max_iter=300,
            random_state=RANDOM_STATE,
            early_stopping=True,
        )
        mlp.fit(data["X_train"], data["y_train"])
        mlp_val.append(evaluate(mlp, data["X_val"], data["y_val"]))
        mlp_test.append(evaluate(mlp, data["X_test"], data["y_test"]))
        print(f"  MLP lr={lr:<8} val={mlp_val[-1]:.4f} test={mlp_test[-1]:.4f}")

        perc = Perceptron(eta0=lr, max_iter=1000, random_state=RANDOM_STATE)
        perc.fit(data["X_train"], data["y_train"])
        perc_val.append(evaluate(perc, data["X_val"], data["y_val"]))
        perc_test.append(evaluate(perc, data["X_test"], data["y_test"]))
        print(f"  Perc eta0={lr:<6} val={perc_val[-1]:.4f} test={perc_test[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(rates, mlp_val, "o-", label="val")
    axes[0].plot(rates, mlp_test, "s--", label="test")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("learning_rate_init")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("MLPClassifier — learning rate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rates, perc_val, "o-", label="val")
    axes[1].plot(rates, perc_test, "s--", label="test")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("eta0")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Perceptron — learning rate (eta0)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS / "exp_learning_rate.png", dpi=150)
    plt.close(fig)

    best_i = int(np.argmax(mlp_val))
    return {
        "rates": rates,
        "mlp_val": mlp_val,
        "mlp_test": mlp_test,
        "perc_val": perc_val,
        "perc_test": perc_test,
        "best_mlp_lr": rates[best_i],
        "best_mlp_val": mlp_val[best_i],
    }


def experiment_regularization(data: dict[str, np.ndarray], best_lr: float) -> dict:
    """Эксперимент: параметр регуляризации alpha (L2)."""
    print("\n=== 5.2 Эксперимент: регуляризация (alpha) ===")
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    mlp_val, mlp_test = [], []
    perc_val, perc_test = [], []

    for alpha in alphas:
        mlp = MLPClassifier(
            hidden_layer_sizes=(100,),
            solver="adam",
            learning_rate_init=best_lr,
            alpha=alpha,
            max_iter=300,
            random_state=RANDOM_STATE,
            early_stopping=True,
        )
        mlp.fit(data["X_train"], data["y_train"])
        mlp_val.append(evaluate(mlp, data["X_val"], data["y_val"]))
        mlp_test.append(evaluate(mlp, data["X_test"], data["y_test"]))
        print(f"  MLP alpha={alpha:<8} val={mlp_val[-1]:.4f} test={mlp_test[-1]:.4f}")

        perc = Perceptron(
            penalty="l2",
            alpha=alpha,
            eta0=0.1,
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        perc.fit(data["X_train"], data["y_train"])
        perc_val.append(evaluate(perc, data["X_val"], data["y_val"]))
        perc_test.append(evaluate(perc, data["X_test"], data["y_test"]))
        print(f"  Perc alpha={alpha:<7} val={perc_val[-1]:.4f} test={perc_test[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(alphas, mlp_val, "o-", label="val")
    axes[0].plot(alphas, mlp_test, "s--", label="test")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("alpha (L2)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("MLPClassifier — regularization")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(alphas, perc_val, "o-", label="val")
    axes[1].plot(alphas, perc_test, "s--", label="test")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("alpha (L2)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Perceptron — regularization")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS / "exp_regularization.png", dpi=150)
    plt.close(fig)

    best_i = int(np.argmax(mlp_val))
    return {
        "alphas": alphas,
        "mlp_val": mlp_val,
        "mlp_test": mlp_test,
        "perc_val": perc_val,
        "perc_test": perc_test,
        "best_mlp_alpha": alphas[best_i],
        "best_mlp_val": mlp_val[best_i],
    }


def experiment_solver(data: dict[str, np.ndarray], best_lr: float, best_alpha: float) -> dict:
    """Эксперимент: функция оптимизации (solver)."""
    print("\n=== 5.3 Эксперимент: solver (оптимизатор) ===")
    solvers = ["adam", "sgd", "lbfgs"]
    mlp_val, mlp_test = [], []

    for solver in solvers:
        kwargs = dict(
            hidden_layer_sizes=(100,),
            solver=solver,
            alpha=best_alpha,
            max_iter=500 if solver == "lbfgs" else 300,
            random_state=RANDOM_STATE,
        )
        if solver in ("adam", "sgd"):
            kwargs["learning_rate_init"] = best_lr
            kwargs["early_stopping"] = True
        if solver == "sgd":
            kwargs["learning_rate"] = "adaptive"

        mlp = MLPClassifier(**kwargs)
        mlp.fit(data["X_train"], data["y_train"])
        mlp_val.append(evaluate(mlp, data["X_val"], data["y_val"]))
        mlp_test.append(evaluate(mlp, data["X_test"], data["y_test"]))
        print(f"  solver={solver:<6} val={mlp_val[-1]:.4f} test={mlp_test[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(solvers))
    w = 0.35
    ax.bar(x - w / 2, mlp_val, w, label="val")
    ax.bar(x + w / 2, mlp_test, w, label="test")
    ax.set_xticks(x)
    ax.set_xticklabels(solvers)
    ax.set_ylabel("Accuracy")
    ax.set_title("MLPClassifier — solver comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "exp_solver.png", dpi=150)
    plt.close(fig)

    best_i = int(np.argmax(mlp_val))
    return {
        "solvers": solvers,
        "mlp_val": mlp_val,
        "mlp_test": mlp_test,
        "best_solver": solvers[best_i],
        "best_mlp_val": mlp_val[best_i],
    }


def train_best(
    data: dict[str, np.ndarray],
    lr: float,
    alpha: float,
    solver: str,
) -> dict:
    """Финальная модель с лучшими параметрами."""
    print("\n=== Финальная модель с лучшими параметрами ===")
    kwargs = dict(
        hidden_layer_sizes=(100,),
        solver=solver,
        alpha=alpha,
        max_iter=500,
        random_state=RANDOM_STATE,
    )
    if solver in ("adam", "sgd"):
        kwargs["learning_rate_init"] = lr
        kwargs["early_stopping"] = True
    if solver == "sgd":
        kwargs["learning_rate"] = "adaptive"

    mlp = MLPClassifier(**kwargs)
    mlp.fit(data["X_train"], data["y_train"])
    scores = {
        "train": evaluate(mlp, data["X_train"], data["y_train"]),
        "val": evaluate(mlp, data["X_val"], data["y_val"]),
        "test": evaluate(mlp, data["X_test"], data["y_test"]),
        "params": {"learning_rate_init": lr, "alpha": alpha, "solver": solver},
    }
    print(
        f"Best MLP — params={scores['params']}\n"
        f"  train={scores['train']:.4f}, val={scores['val']:.4f}, "
        f"test={scores['test']:.4f}"
    )

    if hasattr(mlp, "loss_curve_") and mlp.loss_curve_:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(mlp.loss_curve_)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("MLP loss curve (best model)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / "best_mlp_loss_curve.png", dpi=150)
        plt.close(fig)

    return scores


def plot_split_sizes(data: dict[str, np.ndarray]) -> None:
    sizes = [
        len(data["y_train"]),
        len(data["y_val"]),
        len(data["y_test"]),
    ]
    labels = ["Train", "Validation", "Test"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, sizes, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("Число объектов")
    ax.set_title("Разделение выборки HAPT")
    for bar, n in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, n + 20, str(n), ha="center")
    fig.tight_layout()
    fig.savefig(RESULTS / "split_sizes.png", dpi=150)
    plt.close(fig)


def main() -> None:
    print("=== 0. Загрузка датасета на диск ===")
    root = download()
    # если датасет лежит во вложенной папке
    if not (root / "Train" / "X_train.txt").exists():
        root = DATA_DIR
        for child in DATA_DIR.iterdir():
            if child.is_dir() and (child / "Train" / "X_train.txt").exists():
                root = child
                break

    print("=== 1. Разделение выборки ===")
    x_tr_full, y_tr_full, x_te, y_te, activities = load_hapt(root)
    print(f"Признаков: {x_tr_full.shape[1]}")
    print(f"Train (исходный): {x_tr_full.shape[0]}, Test: {x_te.shape[0]}")
    if activities:
        print(f"Классы ({len(activities)}): {', '.join(activities)}")

    splits = make_splits(x_tr_full, y_tr_full, x_te, y_te)
    print(
        f"После выделения validation: "
        f"train={len(splits['y_train'])}, "
        f"val={len(splits['y_val'])}, "
        f"test={len(splits['y_test'])}"
    )
    plot_split_sizes(splits)

    print("\n=== 2. Масштабирование признаков (StandardScaler) ===")
    x_tr_s, x_val_s, x_te_s, _ = scale_features(
        splits["X_train"], splits["X_val"], splits["X_test"]
    )
    data = {
        "X_train": x_tr_s,
        "y_train": splits["y_train"],
        "X_val": x_val_s,
        "y_val": splits["y_val"],
        "X_test": x_te_s,
        "y_test": splits["y_test"],
    }
    print(
        f"Среднее train после scaling ≈ {data['X_train'].mean():.4f}, "
        f"std ≈ {data['X_train'].std():.4f}"
    )

    baseline = train_baseline(data)
    lr_exp = experiment_learning_rate(data)
    alpha_exp = experiment_regularization(data, lr_exp["best_mlp_lr"])
    solver_exp = experiment_solver(
        data, lr_exp["best_mlp_lr"], alpha_exp["best_mlp_alpha"]
    )
    best = train_best(
        data,
        lr_exp["best_mlp_lr"],
        alpha_exp["best_mlp_alpha"],
        solver_exp["best_solver"],
    )

    summary = {
        "dataset": "HAPT — Smartphone-Based Recognition of Human Activities "
        "and Postural Transitions",
        "n_features": int(x_tr_full.shape[1]),
        "split_sizes": {
            "train": int(len(splits["y_train"])),
            "val": int(len(splits["y_val"])),
            "test": int(len(splits["y_test"])),
        },
        "baseline": {
            "perceptron": baseline["perceptron"],
            "mlp": baseline["mlp"],
        },
        "experiments": {
            "learning_rate": {
                "best_lr": lr_exp["best_mlp_lr"],
                "best_val": lr_exp["best_mlp_val"],
            },
            "regularization": {
                "best_alpha": alpha_exp["best_mlp_alpha"],
                "best_val": alpha_exp["best_mlp_val"],
            },
            "solver": {
                "best_solver": solver_exp["best_solver"],
                "best_val": solver_exp["best_mlp_val"],
            },
        },
        "best_model": best,
        "conclusions": [
            "Масштабирование признаков необходимо для Perceptron и MLP "
            "(градиентные методы чувствительны к масштабу).",
            f"Лучший learning_rate_init по validation: {lr_exp['best_mlp_lr']}.",
            f"Лучший alpha (L2): {alpha_exp['best_mlp_alpha']}.",
            f"Лучший solver по validation: {solver_exp['best_solver']} "
            f"(на test близки adam и lbfgs).",
            f"Итоговая точность лучшей MLP на test: {best['test']:.4f}.",
            f"Базовый Perceptron на test: {baseline['perceptron']['test']:.4f}; "
            f"базовый MLP: {baseline['mlp']['test']:.4f}. "
            "На HAPT линейный Perceptron уже даёт высокую точность, "
            "MLP выигрывает в основном на сложных/переходных классах "
            "и при подборе гиперпараметров.",
        ],
    }

    with (RESULTS / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    report_lines = [
        "Лабораторная работа №4. Основы нейронных сетей",
        "Датасет: HAPT (UCI)",
        "",
        "1. Разделение выборки",
        f"   Train={summary['split_sizes']['train']}, "
        f"Val={summary['split_sizes']['val']}, "
        f"Test={summary['split_sizes']['test']}",
        "   (validation выделен из Train стратифицированно, 20%)",
        "",
        "2. Масштабирование: StandardScaler (fit на train)",
        "",
        "3–4. Базовая точность",
        f"   Perceptron test = {baseline['perceptron']['test']:.4f}",
        f"   MLPClassifier test = {baseline['mlp']['test']:.4f}",
        "",
        "5. Лучшие параметры по validation",
        f"   learning_rate_init = {lr_exp['best_mlp_lr']}",
        f"   alpha = {alpha_exp['best_mlp_alpha']}",
        f"   solver = {solver_exp['best_solver']}",
        "",
        f"Финальная модель test accuracy = {best['test']:.4f}",
        "",
        "Выводы:",
        *[f" - {c}" for c in summary["conclusions"]],
        "",
        "Графики: results/exp_learning_rate.png, "
        "exp_regularization.png, exp_solver.png, "
        "confusion_matrix_mlp.png, split_sizes.png",
    ]
    (RESULTS / "report_notes.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print("\n=== Готово ===")
    print(f"Результаты сохранены в {RESULTS}")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
