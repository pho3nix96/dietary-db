"""Compare data-driven clusters with the official GIFT grouping."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def agreement(clusters, reference) -> dict:
    return {
        "ARI": float(adjusted_rand_score(reference, clusters)),
        "NMI": float(normalized_mutual_info_score(reference, clusters)),
        "n_clusters": int(len(set(clusters))),
        "n_reference_groups": int(len(set(reference))),
    }


def nearest(D, k: int) -> np.ndarray:
    """Indices of each row's k nearest other items (rows x k).

    The item itself is never among them, even when another item is at distance 0;
    `np.argsort(D)[:, 1:k+1]` would assume the item sorts first, which ties break."""
    D = np.array(D, dtype=float, copy=True)
    np.fill_diagonal(D, np.inf)
    k = max(0, min(k, D.shape[0] - 1))
    return np.argsort(D, axis=1, kind="stable")[:, :k]


def knn_agreement(D, labels, k: int = 5) -> float:
    """Share of each item's k nearest neighbours that carry the item's own label."""
    labels = np.asarray(labels)
    idx = nearest(D, k)
    return float((labels[idx] == labels[:, None]).mean())


def chance_agreement(labels) -> float:
    """knn_agreement expected if neighbours were drawn at random (without the item itself)."""
    counts = pd.Series(np.asarray(labels)).value_counts().to_numpy(dtype=float)
    n = counts.sum()
    return float((counts * (counts - 1)).sum() / (n * (n - 1))) if n > 1 else float("nan")


def misfits(df: pd.DataFrame, clusters, group_col: str, id_col: str, name_col: str) -> pd.DataFrame:
    """Foods whose cluster is dominated by a different GIFT group.

    For each cluster, the dominant group is the most common GIFT group in it.
    A food is a misfit when its own group differs from that dominant group.
    `cluster_purity` shows how clear-cut the cluster's identity is.
    """
    t = pd.DataFrame({
        "id": df[id_col].to_numpy(),
        "name": df[name_col].to_numpy(),
        "gift_group": df[group_col].astype(str).to_numpy(),
        "cluster": np.asarray(clusters),
    })
    dom = t.groupby("cluster")["gift_group"].agg(lambda s: s.value_counts().index[0])
    purity = t.groupby("cluster")["gift_group"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    t["cluster_dominant_group"] = t["cluster"].map(dom)
    t["cluster_purity"] = t["cluster"].map(purity).round(3)
    out = t[t["gift_group"] != t["cluster_dominant_group"]]
    return out.sort_values(["cluster_purity", "cluster"], ascending=[False, True])


def resolution_scan(W, reference, resolutions, seeds=(0, 1, 2)) -> pd.DataFrame:
    """Agreement with GIFT across Leiden resolutions and seeds (stability check)."""
    from .fuse import leiden

    rows = []
    for r in resolutions:
        labels = [leiden(W, r, s) for s in seeds]
        between_seeds = [adjusted_rand_score(labels[0], l) for l in labels[1:]] or [1.0]
        a = agreement(labels[0], reference)
        rows.append({"resolution": r, **a, "seed_stability_ARI": float(np.mean(between_seeds))})
    return pd.DataFrame(rows)
