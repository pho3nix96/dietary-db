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
