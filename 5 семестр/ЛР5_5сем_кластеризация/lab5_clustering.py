from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from download_dataset import download

warnings.filterwarnings("ignore")

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# Likert-пункты опроса (основной сигнал для кластеров «отношения к Wikipedia»)
LIKERT_PREFIXES = (
    "PU",
    "PEU",
    "ENJ",
    "Qu",
    "QU",
    "Vis",
    "VIS",
    "Im",
    "IM",
    "SA",
    "Use",
    "USE",
    "Pf",
    "PF",
    "JR",
    "BI",
    "Inc",
    "INC",
    "Exp",
    "EXP",
)


def load_wiki4he(csv_path: Path) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """
    CSV разделён ';', пропуски часто как '?'.
    Берём Likert-вопросы (+ AGE, YEARSEXP при наличии) как числовые признаки.
    """
    df = pd.read_csv(csv_path, sep=";", na_values=["?", ""])
    print(f"Загружено: {csv_path.name}, shape={df.shape}")

    # нормализуем имена колонок
    df.columns = [c.strip() for c in df.columns]

    likert_cols = [
        c
        for c in df.columns
        if any(c.upper().startswith(p.upper()) for p in LIKERT_PREFIXES)
        and c.upper() not in {"UNIVERSITY", "USERWIKI"}
    ]
    # дополнительные числовые демографические
    extra = [c for c in ("AGE", "YEARSEXP") if c in df.columns]
    feature_cols = likert_cols + extra

    data = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    # imputed median — чтобы не терять слишком много строк
    data = data.fillna(data.median(numeric_only=True))
    # если колонка полностью пустая — убрать
    data = data.dropna(axis=1, how="all")
    feature_cols = list(data.columns)

    # отбросить строки, где всё ещё NaN
    mask = ~data.isna().any(axis=1)
    data = data.loc[mask]
    print(f"Признаков: {len(feature_cols)}, объектов после очистки: {len(data)}")
    return data.to_numpy(dtype=float), data, feature_cols


def scale_features(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    print(
        f"После scaling: mean≈{Xs.mean():.4f}, std≈{Xs.std():.4f}"
    )
    return Xs, scaler


def safe_scores(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Метрики качества кластеризации (без меток классов)."""
    labels = np.asarray(labels)
    # DBSCAN может пометить шум как -1
    mask = labels >= 0
    n_clusters = len(set(labels[mask])) if mask.any() else 0
    n_noise = int((labels < 0).sum())

    result = {
        "n_clusters": float(n_clusters),
        "n_noise": float(n_noise),
        "silhouette": float("nan"),
        "davies_bouldin": float("nan"),
        "calinski_harabasz": float("nan"),
    }
    if n_clusters < 2 or mask.sum() < n_clusters + 1:
        return result

    Xm, lm = X[mask], labels[mask]
    if len(set(lm)) < 2:
        return result

    result["silhouette"] = float(silhouette_score(Xm, lm))
    result["davies_bouldin"] = float(davies_bouldin_score(Xm, lm))
    result["calinski_harabasz"] = float(calinski_harabasz_score(Xm, lm))
    return result


def plot_pca_clusters(X: np.ndarray, labels: np.ndarray, title: str, path: Path) -> None:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pts = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    uniq = sorted(set(labels))
    cmap = plt.get_cmap("tab10")
    for i, lab in enumerate(uniq):
        m = labels == lab
        name = "noise" if lab < 0 else f"cluster {lab}"
        ax.scatter(
            pts[m, 0],
            pts[m, 1],
            s=18,
            alpha=0.7,
            label=name,
            color="lightgray" if lab < 0 else cmap(i % 10),
            edgecolors="none",
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title)
    ax.legend(markerscale=1.5, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def experiment_kmeans(X: np.ndarray) -> dict:
    print("\n=== Эксперимент: KMeans (число кластеров) ===")
    ks = list(range(2, 11))
    sil, dbi, ch, inertia = [], [], [], []

    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = model.fit_predict(X)
        sc = safe_scores(X, labels)
        sil.append(sc["silhouette"])
        dbi.append(sc["davies_bouldin"])
        ch.append(sc["calinski_harabasz"])
        inertia.append(float(model.inertia_))
        print(
            f"  k={k}: silhouette={sc['silhouette']:.4f}, "
            f"DBI={sc['davies_bouldin']:.4f}, CH={sc['calinski_harabasz']:.1f}"
        )

    best_i = int(np.nanargmax(sil))
    best_k = ks[best_i]
    best = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    best_labels = best.fit_predict(X)
    plot_pca_clusters(
        X,
        best_labels,
        f"KMeans, k={best_k}",
        RESULTS / "kmeans_pca.png",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ks, sil, "o-", label="silhouette ↑")
    axes[0].plot(ks, dbi, "s--", label="Davies-Bouldin ↓")
    axes[0].set_xlabel("k (число кластеров)")
    axes[0].set_ylabel("Метрика")
    axes[0].set_title("KMeans: качество от k")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ks, inertia, "o-", color="#C44E52")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Inertia")
    axes[1].set_title("KMeans: elbow (inertia)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "exp_kmeans_k.png", dpi=150)
    plt.close(fig)

    return {
        "algorithm": "KMeans",
        "ks": ks,
        "silhouette": sil,
        "davies_bouldin": dbi,
        "calinski_harabasz": ch,
        "inertia": inertia,
        "best_k": best_k,
        "best_silhouette": float(sil[best_i]),
        "labels": best_labels,
    }


def experiment_agglomerative(X: np.ndarray) -> dict:
    print("\n=== Эксперимент: AgglomerativeClustering ===")
    ks = list(range(2, 11))
    linkages = ["ward", "average", "complete"]
    results_table = []

    fig, axes = plt.subplots(1, len(linkages), figsize=(12, 4), sharey=True)
    best = {"silhouette": -1.0}

    for ax, link in zip(axes, linkages):
        sil = []
        for k in ks:
            model = AgglomerativeClustering(n_clusters=k, linkage=link)
            labels = model.fit_predict(X)
            sc = safe_scores(X, labels)
            sil.append(sc["silhouette"])
            results_table.append(
                {
                    "linkage": link,
                    "k": k,
                    "silhouette": sc["silhouette"],
                    "davies_bouldin": sc["davies_bouldin"],
                }
            )
            if sc["silhouette"] > best["silhouette"]:
                best = {
                    "silhouette": sc["silhouette"],
                    "linkage": link,
                    "k": k,
                    "labels": labels,
                    "davies_bouldin": sc["davies_bouldin"],
                }
        ax.plot(ks, sil, "o-")
        ax.set_title(f"linkage={link}")
        ax.set_xlabel("k")
        ax.set_ylabel("Silhouette")
        ax.grid(True, alpha=0.3)
        print(
            f"  linkage={link}: best k by sil ≈ "
            f"{ks[int(np.nanargmax(sil))]} ({max(sil):.4f})"
        )

    fig.suptitle("AgglomerativeClustering: silhouette от k и linkage")
    fig.tight_layout()
    fig.savefig(RESULTS / "exp_agglomerative.png", dpi=150)
    plt.close(fig)

    plot_pca_clusters(
        X,
        best["labels"],
        f"Agglomerative, k={best['k']}, linkage={best['linkage']}",
        RESULTS / "agglomerative_pca.png",
    )

    return {
        "algorithm": "AgglomerativeClustering",
        "table": results_table,
        "best_k": int(best["k"]),
        "best_linkage": best["linkage"],
        "best_silhouette": float(best["silhouette"]),
        "labels": best["labels"],
    }


def experiment_dbscan(X: np.ndarray) -> dict:
    print("\n=== Эксперимент: DBSCAN (eps, min_samples) ===")
    # эвристика eps по k-distance (k=min_samples)
    from sklearn.neighbors import NearestNeighbors

    min_samples_grid = [3, 5, 8, 12]
    # подбор eps вокруг типичных дистанций
    nn = NearestNeighbors(n_neighbors=5).fit(X)
    dists, _ = nn.kneighbors(X)
    kth = np.sort(dists[:, -1])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(kth)
    ax.set_title("k-distance graph (k=5) для выбора eps")
    ax.set_xlabel("Точки (отсортированы)")
    ax.set_ylabel("Расстояние до 5-го соседа")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "dbscan_kdistance.png", dpi=150)
    plt.close(fig)

    eps_grid = np.unique(
        np.round(
            np.quantile(kth, [0.5, 0.7, 0.8, 0.85, 0.9, 0.95]),
            3,
        )
    )
    table = []
    best = {"silhouette": -1.0}

    for eps in eps_grid:
        for ms in min_samples_grid:
            model = DBSCAN(eps=float(eps), min_samples=ms)
            labels = model.fit_predict(X)
            sc = safe_scores(X, labels)
            row = {
                "eps": float(eps),
                "min_samples": ms,
                "n_clusters": int(sc["n_clusters"]),
                "n_noise": int(sc["n_noise"]),
                "silhouette": sc["silhouette"],
                "davies_bouldin": sc["davies_bouldin"],
            }
            table.append(row)
            print(
                f"  eps={eps:.3f}, min_samples={ms}: "
                f"clusters={row['n_clusters']}, noise={row['n_noise']}, "
                f"sil={sc['silhouette']:.4f}"
                if not np.isnan(sc["silhouette"])
                else f"  eps={eps:.3f}, min_samples={ms}: "
                f"clusters={row['n_clusters']}, noise={row['n_noise']}, sil=n/a"
            )
            if not np.isnan(sc["silhouette"]) and sc["silhouette"] > best["silhouette"]:
                best = {**row, "labels": labels, "silhouette": sc["silhouette"]}

    # heatmap silhouette для валидных конфигураций
    pivot_eps = sorted({r["eps"] for r in table})
    pivot_ms = sorted({r["min_samples"] for r in table})
    mat = np.full((len(pivot_ms), len(pivot_eps)), np.nan)
    for r in table:
        i = pivot_ms.index(r["min_samples"])
        j = pivot_eps.index(r["eps"])
        mat[i, j] = r["silhouette"]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot_eps)))
    ax.set_xticklabels([f"{e:.2f}" for e in pivot_eps], rotation=45)
    ax.set_yticks(range(len(pivot_ms)))
    ax.set_yticklabels(pivot_ms)
    ax.set_xlabel("eps")
    ax.set_ylabel("min_samples")
    ax.set_title("DBSCAN: silhouette")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(RESULTS / "exp_dbscan.png", dpi=150)
    plt.close(fig)

    if "labels" in best:
        plot_pca_clusters(
            X,
            best["labels"],
            f"DBSCAN, eps={best['eps']}, min_samples={best['min_samples']}",
            RESULTS / "dbscan_pca.png",
        )
    else:
        # fallback: взять конфиг с наибольшим числом кластеров >=2
        candidates = [r for r in table if r["n_clusters"] >= 2]
        if candidates:
            best = max(candidates, key=lambda r: r["n_clusters"])
            model = DBSCAN(eps=best["eps"], min_samples=best["min_samples"])
            best["labels"] = model.fit_predict(X)
            best["silhouette"] = float("nan")
            plot_pca_clusters(
                X,
                best["labels"],
                f"DBSCAN, eps={best['eps']}, min_samples={best['min_samples']}",
                RESULTS / "dbscan_pca.png",
            )

    return {
        "algorithm": "DBSCAN",
        "table": [
            {k: v for k, v in r.items() if k != "labels"}
            for r in table
        ],
        "best_eps": best.get("eps"),
        "best_min_samples": best.get("min_samples"),
        "best_silhouette": float(best.get("silhouette", float("nan"))),
        "best_n_clusters": int(best.get("n_clusters", 0)),
        "labels": best.get("labels"),
    }


def experiment_gmm(X: np.ndarray) -> dict:
    print("\n=== Эксперимент: GaussianMixture ===")
    ks = list(range(2, 11))
    sil, bics, aics = [], [], []
    best = {"silhouette": -1.0}

    for k in ks:
        model = GaussianMixture(
            n_components=k, covariance_type="full", random_state=RANDOM_STATE
        )
        labels = model.fit_predict(X)
        sc = safe_scores(X, labels)
        sil.append(sc["silhouette"])
        bics.append(float(model.bic(X)))
        aics.append(float(model.aic(X)))
        print(
            f"  components={k}: sil={sc['silhouette']:.4f}, "
            f"BIC={bics[-1]:.1f}"
        )
        if sc["silhouette"] > best["silhouette"]:
            best = {
                "silhouette": sc["silhouette"],
                "k": k,
                "labels": labels,
            }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ks, sil, "o-")
    axes[0].set_xlabel("n_components")
    axes[0].set_ylabel("Silhouette")
    axes[0].set_title("GMM: silhouette")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ks, bics, "o-", label="BIC ↓")
    axes[1].plot(ks, aics, "s--", label="AIC ↓")
    axes[1].set_xlabel("n_components")
    axes[1].set_ylabel("Критерий")
    axes[1].set_title("GMM: BIC / AIC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "exp_gmm.png", dpi=150)
    plt.close(fig)

    plot_pca_clusters(
        X,
        best["labels"],
        f"GaussianMixture, n={best['k']}",
        RESULTS / "gmm_pca.png",
    )

    return {
        "algorithm": "GaussianMixture",
        "ks": ks,
        "silhouette": sil,
        "bic": bics,
        "aic": aics,
        "best_k": int(best["k"]),
        "best_silhouette": float(best["silhouette"]),
        "labels": best["labels"],
    }


def compare_algorithms(results: list[dict]) -> dict:
    print("\n=== Сравнение лучших конфигураций ===")
    rows = []
    for r in results:
        rows.append(
            {
                "algorithm": r["algorithm"],
                "silhouette": r["best_silhouette"],
                "params": {
                    k: r[k]
                    for k in r
                    if k.startswith("best_") and k != "best_silhouette"
                },
            }
        )
        print(
            f"  {r['algorithm']}: silhouette={r['best_silhouette']:.4f}, "
            f"params={rows[-1]['params']}"
        )

    valid = [row for row in rows if not np.isnan(row["silhouette"])]
    winner = max(valid, key=lambda x: x["silhouette"]) if valid else rows[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r["algorithm"] for r in rows]
    vals = [r["silhouette"] if not np.isnan(r["silhouette"]) else 0 for r in rows]
    colors = ["#55A868" if n == winner["algorithm"] else "#4C72B0" for n in names]
    ax.bar(names, vals, color=colors)
    ax.set_ylabel("Silhouette (чем выше, тем лучше)")
    ax.set_title("Сравнение лучших конфигураций алгоритмов")
    ax.set_ylim(0, max(0.1, max(vals) * 1.15))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "algorithms_comparison.png", dpi=150)
    plt.close(fig)

    return {"rows": rows, "winner": winner}


def cluster_profiles(df_feat: pd.DataFrame, labels: np.ndarray, name: str) -> None:
    """Средние значения признаков по кластерам — для интерпретации."""
    if labels is None:
        return
    tmp = df_feat.copy()
    tmp["cluster"] = labels
    tmp = tmp[tmp["cluster"] >= 0]
    means = tmp.groupby("cluster").mean()

    # топ-вариативных признаков для heatmap
    var = means.var().sort_values(ascending=False)
    top = list(var.head(12).index)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(means[top].T, aspect="auto", cmap="coolwarm")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top)
    ax.set_xticks(range(len(means.index)))
    ax.set_xticklabels([f"C{c}" for c in means.index])
    ax.set_title(f"Профили кластеров ({name})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(RESULTS / "cluster_profiles.png", dpi=150)
    plt.close(fig)


def main() -> None:
    print("=== 0. Загрузка датасета на диск ===")
    csv_path = download()

    print("\n=== Загрузка и подготовка ===")
    X, df_feat, feature_cols = load_wiki4he(csv_path)

    print("\n=== 1. Масштабирование признаков ===")
    Xs, _ = scale_features(X)

    # PCA overview
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pts = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.6)
    ax.set_title("wiki4HE после scaling (PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "data_pca_overview.png", dpi=150)
    plt.close(fig)

    print("\n=== 2–3. Кластеризация и эксперименты ===")
    km = experiment_kmeans(Xs)
    agg = experiment_agglomerative(Xs)
    db = experiment_dbscan(Xs)
    gmm = experiment_gmm(Xs)

    comparison = compare_algorithms([km, agg, db, gmm])
    winner_name = comparison["winner"]["algorithm"]
    winner_labels = {
        "KMeans": km["labels"],
        "AgglomerativeClustering": agg["labels"],
        "DBSCAN": db.get("labels"),
        "GaussianMixture": gmm["labels"],
    }[winner_name]
    cluster_profiles(df_feat, winner_labels, winner_name)

    summary = {
        "dataset": "wiki4HE (UCI)",
        "n_samples": int(Xs.shape[0]),
        "n_features": int(Xs.shape[1]),
        "features": feature_cols,
        "scaling": "StandardScaler",
        "algorithms": ["KMeans", "AgglomerativeClustering", "DBSCAN", "GaussianMixture"],
        "best": comparison["winner"],
        "kmeans": {
            "best_k": km["best_k"],
            "best_silhouette": km["best_silhouette"],
        },
        "agglomerative": {
            "best_k": agg["best_k"],
            "best_linkage": agg["best_linkage"],
            "best_silhouette": agg["best_silhouette"],
        },
        "dbscan": {
            "best_eps": db["best_eps"],
            "best_min_samples": db["best_min_samples"],
            "best_silhouette": db["best_silhouette"],
            "best_n_clusters": db["best_n_clusters"],
        },
        "gmm": {
            "best_k": gmm["best_k"],
            "best_silhouette": gmm["best_silhouette"],
        },
        "conclusions": [
            "Признаки Likert-опроса масштабированы StandardScaler "
            "(обязательно для KMeans/DBSCAN/GMM).",
            f"Лучший алгоритм по silhouette: {winner_name} "
            f"({comparison['winner']['silhouette']:.4f}).",
            f"KMeans: оптимальное k≈{km['best_k']}.",
            f"Agglomerative: k={agg['best_k']}, linkage={agg['best_linkage']}.",
            f"DBSCAN: eps={db['best_eps']}, min_samples={db['best_min_samples']} "
            "(чувствителен к плотности, часть точек может быть noise).",
            f"GMM: n_components={gmm['best_k']}.",
            "Кластеры интерпретируются как группы преподавателей "
            "с похожим отношением/практикой использования Wikipedia.",
        ],
    }

    # убрать nondumpy labels из json
    with (RESULTS / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    report = [
        "Лабораторная работа №5. Алгоритмы кластеризации",
        "Датасет: wiki4HE",
        "",
        f"Объектов: {Xs.shape[0]}, признаков: {Xs.shape[1]}",
        "Scaling: StandardScaler",
        "",
        "Алгоритмы (≥3): KMeans, AgglomerativeClustering, DBSCAN, GaussianMixture",
        "",
        "Лучшие параметры:",
        f"  KMeans: k={km['best_k']}, silhouette={km['best_silhouette']:.4f}",
        f"  Agglomerative: k={agg['best_k']}, linkage={agg['best_linkage']}, "
        f"silhouette={agg['best_silhouette']:.4f}",
        f"  DBSCAN: eps={db['best_eps']}, min_samples={db['best_min_samples']}, "
        f"silhouette={db['best_silhouette']:.4f}",
        f"  GMM: n={gmm['best_k']}, silhouette={gmm['best_silhouette']:.4f}",
        "",
        f"Итог: лучший алгоритм — {winner_name}",
        "",
        "Выводы:",
        *[f" - {c}" for c in summary["conclusions"]],
        "",
        "Графики: exp_kmeans_k.png, exp_agglomerative.png, exp_dbscan.png,",
        "exp_gmm.png, algorithms_comparison.png, *_pca.png",
    ]
    (RESULTS / "report_notes.txt").write_text("\n".join(report), encoding="utf-8")

    print("\n=== Готово ===")
    print(f"Результаты: {RESULTS}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
