from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from download_dataset import download

warnings.filterwarnings("ignore")

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Признаки без утечки: casual + registered = cnt
FEATURE_COLS = [
    "season",
    "yr",
    "mnth",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]
# Для полиномов — непрерывные признаки (иначе число колонок взрывается)
POLY_COLS = ["temp", "atemp", "hum", "windspeed"]


def manual_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Разделение выборки без sklearn.model_selection.train_test_split.
    Перемешиваем индексы через numpy.RandomState и режем вручную.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size должен быть в (0, 1)")

    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    indices = rng.permutation(n)
    n_test = int(round(n * test_size))
    if n_test < 1 or n_test >= n:
        raise ValueError("Некорректный размер тестовой выборки")

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def load_data(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Используем day.csv (731 день): удобнее для полиномов и наглядных графиков.
    Цель — cnt (суммарное число аренд).
    """
    csv_path = root / "day.csv"
    if not csv_path.exists():
        # на случай вложенной папки
        matches = list(root.rglob("day.csv"))
        if not matches:
            raise FileNotFoundError(f"day.csv не найден в {root}")
        csv_path = matches[0]

    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    X_poly = df[POLY_COLS].to_numpy(dtype=float)
    y = df["cnt"].to_numpy(dtype=float)
    print(f"Файл: {csv_path}")
    print(f"Объектов: {len(df)}, признаков: {len(FEATURE_COLS)}, цель: cnt")
    return X, X_poly, y, FEATURE_COLS


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, s=18, edgecolors="none")
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "r--", lw=1.5, label="идеал y = x")
    ax.set_xlabel("Истинные значения (cnt)")
    ax.set_ylabel("Предсказания")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_linear(X_train, X_test, y_train, y_test) -> dict:
    print("\n=== 2–3. Линейная регрессия ===")
    model = LinearRegression()
    model.fit(X_train, y_train)

    train_m = metrics(y_train, model.predict(X_train))
    test_m = metrics(y_test, model.predict(X_test))
    print(
        f"LinearRegression — "
        f"R² train={train_m['r2']:.4f}, test={test_m['r2']:.4f}; "
        f"RMSE test={test_m['rmse']:.2f}"
    )

    plot_pred_vs_true(
        y_test,
        model.predict(X_test),
        "Линейная регрессия: pred vs true (test)",
        RESULTS / "linear_pred_vs_true.png",
    )

    # коэффициенты
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(FEATURE_COLS, model.coef_)
    ax.set_title("Коэффициенты LinearRegression")
    ax.set_ylabel("Вес")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "linear_coefficients.png", dpi=150)
    plt.close(fig)

    return {"train": train_m, "test": test_m, "intercept": float(model.intercept_)}


def experiment_polynomial(X_poly, y) -> dict:
    """
    Полиномиальная регрессия по непрерывным признакам.
    График R²(train/test) от степени полинома.
    """
    print("\n=== 4. Полиномиальная регрессия ===")
    X_train, X_test, y_train, y_test = manual_train_test_split(
        X_poly, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    degrees = list(range(1, 8))
    train_r2, test_r2 = [], []
    train_rmse, test_rmse = [], []

    for deg in degrees:
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        X_tr = poly.fit_transform(X_train)
        X_te = poly.transform(X_test)

        # масштабирование после полиномизации — стабильнее для высоких степеней
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        model = LinearRegression()
        model.fit(X_tr, y_train)
        tr = metrics(y_train, model.predict(X_tr))
        te = metrics(y_test, model.predict(X_te))
        train_r2.append(tr["r2"])
        test_r2.append(te["r2"])
        train_rmse.append(tr["rmse"])
        test_rmse.append(te["rmse"])
        print(
            f"  degree={deg}: features={X_tr.shape[1]}, "
            f"R² train={tr['r2']:.4f}, test={te['r2']:.4f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(degrees, train_r2, "o-", label="train")
    axes[0].plot(degrees, test_r2, "s--", label="test")
    axes[0].set_xlabel("Степень полинома")
    axes[0].set_ylabel("R² (точность)")
    axes[0].set_title("Полиномиальная регрессия: R² от степени")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(degrees, train_rmse, "o-", label="train")
    axes[1].plot(degrees, test_rmse, "s--", label="test")
    axes[1].set_xlabel("Степень полинома")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Полиномиальная регрессия: RMSE от степени")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS / "poly_degree_accuracy.png", dpi=150)
    plt.close(fig)

    best_i = int(np.argmax(test_r2))
    return {
        "degrees": degrees,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "best_degree": degrees[best_i],
        "best_test_r2": test_r2[best_i],
        "features_used": POLY_COLS,
    }


def experiment_regularization(X, y) -> dict:
    """
    Ridge и Lasso: зависимость R² от коэффициента регуляризации alpha.
    """
    print("\n=== 5. Регуляризация (Ridge / Lasso) ===")
    X_train, X_test, y_train, y_test = manual_train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    alphas = np.logspace(-3, 3, 13)  # 0.001 ... 1000
    ridge_train, ridge_test = [], []
    lasso_train, lasso_test = [], []

    for alpha in alphas:
        ridge = Ridge(alpha=float(alpha), random_state=RANDOM_STATE)
        ridge.fit(X_train_s, y_train)
        ridge_train.append(r2_score(y_train, ridge.predict(X_train_s)))
        ridge_test.append(r2_score(y_test, ridge.predict(X_test_s)))

        lasso = Lasso(alpha=float(alpha), max_iter=10000, random_state=RANDOM_STATE)
        lasso.fit(X_train_s, y_train)
        lasso_train.append(r2_score(y_train, lasso.predict(X_train_s)))
        lasso_test.append(r2_score(y_test, lasso.predict(X_test_s)))

        print(
            f"  alpha={alpha:<8.4g}  "
            f"Ridge test R²={ridge_test[-1]:.4f}  "
            f"Lasso test R²={lasso_test[-1]:.4f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(alphas, ridge_train, "o-", label="train")
    axes[0].plot(alphas, ridge_test, "s--", label="test")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Коэффициент регуляризации α")
    axes[0].set_ylabel("R² (точность)")
    axes[0].set_title("Ridge: R² от α")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(alphas, lasso_train, "o-", label="train")
    axes[1].plot(alphas, lasso_test, "s--", label="test")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Коэффициент регуляризации α")
    axes[1].set_ylabel("R² (точность)")
    axes[1].set_title("Lasso: R² от α")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS / "regularization_alpha.png", dpi=150)
    plt.close(fig)

    best_ridge_i = int(np.argmax(ridge_test))
    best_lasso_i = int(np.argmax(lasso_test))

    # финальная модель с лучшим alpha (по test среди Ridge/Lasso)
    if ridge_test[best_ridge_i] >= lasso_test[best_lasso_i]:
        best_name = "Ridge"
        best_alpha = float(alphas[best_ridge_i])
        best_r2 = float(ridge_test[best_ridge_i])
        final = Ridge(alpha=best_alpha, random_state=RANDOM_STATE)
    else:
        best_name = "Lasso"
        best_alpha = float(alphas[best_lasso_i])
        best_r2 = float(lasso_test[best_lasso_i])
        final = Lasso(alpha=best_alpha, max_iter=10000, random_state=RANDOM_STATE)

    final.fit(X_train_s, y_train)
    final_test = metrics(y_test, final.predict(X_test_s))
    plot_pred_vs_true(
        y_test,
        final.predict(X_test_s),
        f"{best_name}(α={best_alpha:g}): pred vs true (test)",
        RESULTS / "best_regularized_pred_vs_true.png",
    )

    print(
        f"Лучшая регуляризация: {best_name}, α={best_alpha:g}, "
        f"test R²={best_r2:.4f}"
    )

    return {
        "alphas": [float(a) for a in alphas],
        "ridge_train_r2": [float(v) for v in ridge_train],
        "ridge_test_r2": [float(v) for v in ridge_test],
        "lasso_train_r2": [float(v) for v in lasso_train],
        "lasso_test_r2": [float(v) for v in lasso_test],
        "best_model": best_name,
        "best_alpha": best_alpha,
        "best_test_r2": best_r2,
        "best_test_metrics": final_test,
    }


def plot_split_sizes(n_train: int, n_test: int) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Train", "Test"], [n_train, n_test], color=["#4C72B0", "#C44E52"])
    ax.set_ylabel("Число объектов")
    ax.set_title("Разделение выборки (ручное)")
    for bar, n in zip(bars, [n_train, n_test]):
        ax.text(bar.get_x() + bar.get_width() / 2, n + 5, str(n), ha="center")
    fig.tight_layout()
    fig.savefig(RESULTS / "split_sizes.png", dpi=150)
    plt.close(fig)


def main() -> None:
    print("=== 0. Загрузка датасета на диск ===")
    root = download()

    print("\n=== 1. Ручное разделение train/test ===")
    X, X_poly, y, _ = load_data(root)
    X_train, X_test, y_train, y_test = manual_train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(
        f"train={len(y_train)}, test={len(y_test)} "
        f"(test_size={TEST_SIZE}, без train_test_split)"
    )
    plot_split_sizes(len(y_train), len(y_test))

    linear = train_linear(X_train, X_test, y_train, y_test)
    poly = experiment_polynomial(X_poly, y)
    reg = experiment_regularization(X, y)

    summary = {
        "dataset": "Bike Sharing Dataset (day.csv, UCI)",
        "target": "cnt",
        "features": FEATURE_COLS,
        "split": {
            "method": "manual permutation + slice (без train_test_split)",
            "test_size": TEST_SIZE,
            "train": int(len(y_train)),
            "test": int(len(y_test)),
            "random_state": RANDOM_STATE,
        },
        "linear_regression": linear,
        "polynomial": poly,
        "regularization": reg,
        "conclusions": [
            "Разделение train/test выполнено вручную через np.random.permutation.",
            f"Линейная регрессия: R² test = {linear['test']['r2']:.4f}, "
            f"RMSE = {linear['test']['rmse']:.2f}.",
            f"Лучшая степень полинома по test R²: {poly['best_degree']} "
            f"(R² = {poly['best_test_r2']:.4f}). "
            "Слишком высокая степень ведёт к переобучению.",
            f"Лучшая регуляризация: {reg['best_model']} "
            f"с α = {reg['best_alpha']}, test R² = {reg['best_test_r2']:.4f}.",
            "Малый α ≈ обычная линейная регрессия; слишком большой α "
            "занижает веса и ухудшает качество.",
        ],
    }

    with (RESULTS / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    report = [
        "Лабораторная работа №3. Линейная регрессия",
        "Датасет: Bike Sharing Dataset (day.csv)",
        "",
        "1. Разделение выборки (вручную, без train_test_split)",
        f"   Train={len(y_train)}, Test={len(y_test)}",
        "",
        "2–3. LinearRegression",
        f"   R² train={linear['train']['r2']:.4f}, test={linear['test']['r2']:.4f}",
        f"   RMSE test={linear['test']['rmse']:.2f}, MAE test={linear['test']['mae']:.2f}",
        "",
        "4. Полиномиальная регрессия",
        f"   Лучшая степень: {poly['best_degree']}, test R²={poly['best_test_r2']:.4f}",
        "   График: results/poly_degree_accuracy.png",
        "",
        "5. Регуляризация",
        f"   Лучшая модель: {reg['best_model']}, α={reg['best_alpha']}",
        f"   test R²={reg['best_test_r2']:.4f}",
        "   График: results/regularization_alpha.png",
        "",
        "Выводы:",
        *[f" - {c}" for c in summary["conclusions"]],
    ]
    (RESULTS / "report_notes.txt").write_text("\n".join(report), encoding="utf-8")

    print("\n=== Готово ===")
    print(f"Результаты: {RESULTS}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
