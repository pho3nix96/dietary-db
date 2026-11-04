#!/usr/bin/env python3
"""Build a food tree with the treelet transform.

    python scripts/food_tree.py [--foods results/cereals/clusters.csv] [--k 8]
        [--blocks totals fat_type sugar_type carbohydrate_type protein_osborne]

Foods are the variables; their standardized composition profiles are the
observations (macros per 100 g on a log scale, plus centred log-ratios of each
fraction block, every block given equal total weight). The treelet transform
merges the two most correlated foods (or branches) at each level, rotates them
into a sum variable (the branch prototype) and a difference variable, and
continues to the root.

Cut level. In dietary-pattern studies (Gorst-Rasmussen et al. 2011, summarized
by Zhao et al. 2021, Nutr J 20:37) k is fixed, the share of variance carried by
the k largest sum variables is computed at every level (cross-validated over
people), and the tree is cut where that curve stops rising. On composition
profiles the curve keeps rising to the root, so here the default cut is the
level where k branches remain (L = p - k); the curve is still written out.

Writes results/food_tree/{nodes.csv, level_curve.csv, leaves.csv,
tree.json, dendrogram.png}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score as ari

from foodspace import fractions, treelet as tl

OSBORNE = ["albumin_globulin", "prolamin", "glutelin"]
GRAINS = ["wheat", "rye", "oats", "barley", "maize", "rice"]


def blocks_for(sub: pd.DataFrame, cfg: dict, wanted: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    if "totals" in wanted:
        cols = cfg["columns"]["nutrients"]
        out["totals"] = np.log1p(sub[cols].apply(pd.to_numeric, errors="coerce").clip(lower=0))
    for name, b in cfg["fractions"]["blocks"].items():
        if name in wanted and b.get("source") != "curated":
            S = fractions.block_shares(sub, b["parts"], b.get("parent"), b.get("min_parent", 0.5))
            out[name] = fractions.clr(S)
    if "protein_osborne" in wanted:
        out["protein_osborne"] = fractions.clr(sub[OSBORNE])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--foods", default="results/cereals/clusters.csv",
                    help="foods to build the tree on (output of within_group_cereals.py)")
    ap.add_argument("--blocks", nargs="+",
                    default=["totals", "fat_type", "sugar_type", "carbohydrate_type", "protein_osborne"])
    ap.add_argument("--k", type=int, default=8, help="number of factors / branches to report")
    ap.add_argument("--level", type=int, default=None, help="cut level; default p - k (k branches remain)")
    ap.add_argument("--out", default="results/food_tree")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    foods = pd.read_csv(args.foods)
    table = pd.read_csv(cfg["input"]).set_index("food_id")
    sub = table.loc[foods["food_id"]].reset_index()
    for c in OSBORNE:
        sub[c] = foods[c].to_numpy()
    blocks = blocks_for(sub, cfg, args.blocks)
    F = tl.feature_matrix(blocks)
    C = tl.food_covariance(F)
    p = len(sub)
    res = tl.treelet(C, kmax=max(args.k, 20))

    curve = tl.explained_by_level(res, args.k, np.trace(C))
    p = len(sub)
    level = args.level if args.level is not None else p - args.k
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"level": np.arange(p), f"explained_top{args.k}": curve}).round(4).to_csv(out / "level_curve.csv", index=False)

    # nodes
    names = sub["name"].to_numpy()
    grain = foods["dominant_grain"].to_numpy()
    group = foods["main_group"].to_numpy()
    rows = []
    for nd in res["nodes"]:
        lv = np.array(nd["leaves"])
        w = np.abs(nd["loading"]); w = w / w.sum()
        top = lv[np.argsort(-w)[:4]]
        g = pd.Series(grain[lv]).value_counts(normalize=True)
        rows.append({"node": nd["node"], "level": nd["level"], "n": len(lv), "similarity": nd["similarity"],
                     **{f: float(w @ foods[f].to_numpy()[lv]) for f in OSBORNE},
                     "grains": "; ".join(f"{k} {v:.0%}" for k, v in g.head(3).items()),
                     "food_groups": "; ".join(f"{k} ({v})" for k, v in pd.Series(group[lv]).value_counts().head(3).items()),
                     "prototype_foods": " | ".join(names[top])})
    nodes = pd.DataFrame(rows)
    nodes.round(3).to_csv(out / "nodes.csv", index=False)

    # branches at the cut: stop after `level` merges
    labels = tl.cut(res["linkage"], p, p - level)
    sizes = pd.Series(labels).value_counts()
    leaves = foods[["food_id", "name", "main_group", "dominant_grain", *OSBORNE]].assign(branch=labels)
    leaves.to_csv(out / "leaves.csv", index=False)

    # compare with the references at k branches (cutting near the root)
    lab_k = tl.cut(res["linkage"], p, args.k)
    report = {"foods": p, "features": F.shape[1], "blocks": list(blocks), "k": args.k, "cut_level": int(level),
              "branches_at_cut": int(len(sizes)), "branch_sizes": sizes.tolist(),
              "explained_at_cut": float(curve[level]),
              "ari_k_branches_vs_dominant_grain": ari(grain, lab_k),
              "ari_k_branches_vs_food_group": ari(group, lab_k)}
    from scipy.cluster.hierarchy import fcluster, linkage as hlink
    from scipy.spatial.distance import squareform
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    D = 1 - C / np.outer(d, d)
    np.fill_diagonal(D, 0)
    hc = fcluster(hlink(squareform(np.clip(D, 0, None), checks=False), "average"), args.k, "maxclust")
    report["baseline_average_linkage_ari_vs_dominant_grain"] = ari(grain, hc)
    report["baseline_average_linkage_ari_vs_food_group"] = ari(group, hc)
    report["baseline_branch_sizes"] = pd.Series(hc).value_counts().tolist()
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # tree for a viewer
    tree = {"leaves": [{"id": i, "name": names[i], "group": group[i], "grain": grain[i],
                        **{f: round(float(foods[f][i]), 3) for f in OSBORNE}} for i in range(p)],
            "merges": [{"node": int(n["node"]), "a": int(z[0]), "b": int(z[1]), "level": int(n["level"]),
                        "similarity": round(float(n["similarity"]), 4)} for n, z in zip(res["nodes"], res["linkage"])],
            "curve": [round(float(x), 4) for x in curve], "k": args.k, "cut_level": int(level)}
    (out / "tree.json").write_text(json.dumps(tree))

    # dendrogram
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    Z = res["linkage"].copy()
    Z[:, 2] = np.arange(1, p)                          # height = treelet level, as in the review's Fig. 4
    colors = {"wheat": "#c9822b", "rye": "#6b4f9e", "oats": "#3d8c5c", "barley": "#b8a03a",
              "maize": "#d64b4b", "rice": "#3b7bbf", "mixed": "#888888"}
    fig, ax = plt.subplots(figsize=(18, 7))
    dn = dendrogram(Z, ax=ax, no_labels=True, color_threshold=0, above_threshold_color="#9aa3a0")
    for x, leaf in zip(np.arange(5, 10 * p, 10), dn["leaves"]):
        ax.scatter(x, -4, s=10, c=colors.get(grain[leaf], "#888"), marker="s")
    ax.axhline(level, ls=":", c="k", lw=0.8)
    ax.text(10, level + 3, f"cut level L = {level}", fontsize=9)
    ax.set_ylabel("treelet level")
    ax.set_title(f"Food tree of {p} cereal foods (treelet transform; leaves coloured by dominant grain)")
    for g, c in colors.items():
        ax.scatter([], [], c=c, marker="s", label=g)
    ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
    ax.set_ylim(-8, p)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "dendrogram.png", dpi=130)
    print(f"Wrote {out}/")


if __name__ == "__main__":
    main()
