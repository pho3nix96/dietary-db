#!/usr/bin/env python3
"""Cluster foods inside each FAO/WHO GIFT group (groups are fixed, never mixed).

    python scripts/within_gift_groups.py [--config config/config.yaml] [--out results/within_gift]

Top level: the 19 GIFT nutrition-sensitive groups, taken as given from FoodEx2
(scripts/map_gift.py). Inside each group, foods are compared on composition and fractions, each
scaled to [0, 1] and averaged with equal weight. Food names are NOT part of
the similarity: they are used only to place new foods (src/foodspace/placement.py)
and are reported here as a diagnostic.

    (names        TF-IDF on the English food name: diagnostic only, not fused)
    composition   macros per 100 g (log1p, z-scored within the group, cosine)
    fractions     fat (SFA/MUFA/PUFA; omega-6/ALA/long-chain omega-3;
                  C4-C10 vs longer saturated), sugars (mono-/sucrose/other di-),
                  carbohydrate (starch/sugars/fibre) and protein fractions
                  (recipe-based protfrac_* columns when the table has them;
                  otherwise curated categories plus cereal Osborne profiles)
    (each fraction block counts only for pairs of foods that both have it)

Then, per group: Leiden clusters on a kNN graph of the fused similarity, a
treelet tree on the fused similarity matrix, a 2D layout, and agreement with
the GIFT subgroups (ARI, NMI, share of 5 nearest neighbours in the same
subgroup for each view).

Writes results/within_gift/{foods.csv, groups.csv, trees.json, map.json}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from foodspace import fractions, fuse, layout, treelet as tl, views

OSBORNE_TO_COMMON = {"albumin_globulin": "mixed", "prolamin": "alcohol_soluble", "glutelin": "insoluble_storage"}
MIN_FOODS = 8


def protein_block(df, cfg, curated, cereal_profiles):
    b = cfg["fractions"]["blocks"]["protein_type"]
    S = fractions.curated_block(df, curated, b["key_col"], b["scheme"])
    for c in OSBORNE_TO_COMMON.values():
        if c not in S:
            S[c] = 0.0
    if cereal_profiles is not None:
        cp = cereal_profiles.set_index("food_id")
        ids = df["food_id"].to_numpy()
        for i, fid in enumerate(ids):
            if fid in cp.index:
                S.iloc[i] = 0.0
                for o, c in OSBORNE_TO_COMMON.items():
                    S.iloc[i, S.columns.get_loc(c)] = cp.loc[fid, o]
    low = pd.to_numeric(df[b["parent"]], errors="coerce").fillna(0) < b.get("min_parent", 2.0)
    S[low.to_numpy()] = np.nan
    return S


def cereal_profiles_ok(path):
    p = Path(path)
    if not p.exists():
        return None
    t = pd.read_csv(p)
    ok = (t["grain_mass_share"] >= 0.5) & t["coverage"].between(0.5, 2.0) & (t["profiled_share"] >= 0.9)
    return t[ok][["food_id", "albumin_globulin", "prolamin", "glutelin"]]


def knn_same(D, labels, k=5):
    k = min(k, len(labels) - 1)
    idx = np.argsort(D, axis=1)[:, 1 : k + 1]
    return float((labels[idx] == labels[:, None]).mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--cereal-profiles", default="results/cereals/cereal_profiles.csv")
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--input", default=None, help="food table; default: the config's input (Swedish)")
    ap.add_argument("--out", default="results/within_gift")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    df = pd.read_csv(args.input or cfg["input"])
    df = df[df["gift_group"] != "unassigned"].reset_index(drop=True)
    curated = pd.read_csv(cfg["fractions"]["curated_table"])
    # Names only: the FoodEx2 description or the database's own class name would leak the
    # GIFT subgroup (it is assigned from them), inflating agreement with the reference.
    texts = df["name"].fillna("").str.lower().tolist()
    E = views.tfidf_embeddings(texts)

    measured = {k: v for k, v in cfg["fractions"]["blocks"].items() if v.get("source") != "curated"}
    blocks_all = fractions.build_blocks(df, {"blocks": measured}, None)
    pf = [c for c in df.columns if c.startswith("protfrac_")]
    if pf:  # recipe-expanded profiles already in the table (Fineli)
        P = df[pf].rename(columns=lambda c: c[len("protfrac_"):]).astype(float)
        low = pd.to_numeric(df["PROT_g"], errors="coerce").fillna(0) < cfg["fractions"]["blocks"]["protein_type"].get("min_parent", 2.0)
        P[low.to_numpy()] = np.nan
        blocks_all["protein_type"] = P
    else:
        blocks_all["protein_type"] = protein_block(df, cfg, curated, cereal_profiles_ok(args.cereal_profiles))
    print("fraction blocks:", {k: int(B.notna().all(axis=1).sum()) for k, B in blocks_all.items()})
    cols = cfg["columns"]["nutrients"]

    food_rows, group_rows, trees, groups_out = [], [], {}, []
    has_src = "source" in df.columns
    cross_rows = []
    for g, sub in df.groupby("gift_group", sort=False):
        ix = sub.index.to_numpy()
        n = len(ix)
        sg = sub["gift_subgroup"].fillna("?").to_numpy()
        if n < MIN_FOODS:
            labels = np.zeros(n, int)
            xy = np.column_stack([np.linspace(-0.5, 0.5, n), np.zeros(n)]) if n > 1 else np.zeros((1, 2))
            Dv = None
        else:
            Xp, _ = views.prepare_nutrients(sub[cols].reset_index(drop=True))
            Dn = views.nutrient_distance(Xp, cfg["nutrients"]["metric"])
            Fd, frep = fractions.fraction_distance({k: B.iloc[ix].reset_index(drop=True) for k, B in blocks_all.items()})
            Ds = views.embedding_distance(E[ix])
            # names do not shape the space; they are used only to place new foods (foodspace.placement)
            D = fuse.weighted_fusion({"comp": Dn, "frac": Fd}, {"comp": 1, "frac": 1})
            k = min(10, n - 1)
            W = fuse.knn_sparsify(fuse.affinity(D, k=k), k)
            labels = np.asarray(fuse.leiden(W, resolution=args.resolution, seed=0))
            xy = layout.embed_2d(D, "umap" if n >= 15 else "tsne")
            xy = (xy - xy.mean(0)) / (np.abs(xy - xy.mean(0)).max() or 1)
            Dv = {"semantics": Ds, "composition": Dn, "fractions": Fd, "fused": D}
            res = tl.treelet(1.0 - D)
            trees[g] = {"food_ids": sub["food_id"].tolist(),
                        "linkage": np.round(res["linkage"], 4).tolist()}
        row = {"gift_group": g, "foods": n, "gift_subgroups": int(len(set(sg))), "clusters": int(len(set(labels)))}
        if has_src and Dv is not None:
            src = sub["source"].to_numpy()
            if len(set(src)) > 1:
                D = Dv["fused"]
                k = min(5, n - 1)
                idx = np.argsort(D, axis=1)[:, 1 : k + 1]
                other = (src[idx] != src[:, None]).mean()
                # expected share of other-country neighbours if country did not matter
                exp = np.mean([(src != s_).sum() / (n - 1) for s_ in src])
                row.update({"other_country_5nn": float(other), "other_country_expected": float(exp),
                            "country_mixing": float(other / exp) if exp > 0 else np.nan})
                for v, Dm in Dv.items():
                    ix5 = np.argsort(Dm, axis=1)[:, 1 : k + 1]
                    row[f"country_mixing_{v}"] = float((src[ix5] != src[:, None]).mean() / exp) if exp > 0 else np.nan
                ids = sub["food_id"].to_numpy()
                names = sub["name"].to_numpy()
                for j in range(n):
                    o = np.where(src != src[j])[0]
                    b = o[np.argmin(D[j, o])]
                    cross_rows.append({"gift_group": g, "food_id": int(ids[j]), "source": src[j], "name": names[j],
                                       "match_food_id": int(ids[b]), "match_source": src[b], "match_name": names[b],
                                       "distance": round(float(D[j, b]), 4),
                                       "same_cluster": bool(labels[j] == labels[b])})
        if Dv is not None and len(set(sg)) > 1:
            row.update({"ari_vs_subgroup": adjusted_rand_score(sg, labels),
                        "nmi_vs_subgroup": normalized_mutual_info_score(sg, labels),
                        **{f"same_subgroup_5nn_{v}": knn_same(Dm, sg) for v, Dm in Dv.items()}})
        group_rows.append(row)
        for j, i in enumerate(ix):
            food_rows.append({"food_id": int(df.at[i, "food_id"]), "name": df.at[i, "name"], "gift_group": g,
                              "gift_subgroup": sg[j], "main_group": str(df.at[i, "main_group"]).strip(),
                              "cluster": int(labels[j]), "x": float(xy[j, 0]), "y": float(xy[j, 1]),
                              "protein_profile": bool(blocks_all["protein_type"].iloc[i].notna().all()),
                              **({"source": df.at[i, "source"]} if has_src else {})})

    out = Path(args.out)
    if cross_rows:
        out.mkdir(parents=True, exist_ok=True)
        cw = pd.DataFrame(cross_rows)
        # mutual best matches are the most reliable equivalent foods
        best = cw.set_index("food_id")["match_food_id"]
        cw["mutual"] = cw["match_food_id"].map(best) == cw["food_id"]
        cw.to_csv(out / "crosswalk.csv", index=False)
    out.mkdir(parents=True, exist_ok=True)
    foods = pd.DataFrame(food_rows)
    foods.to_csv(out / "foods.csv", index=False)
    groups = pd.DataFrame(group_rows).sort_values("foods", ascending=False)
    groups.round(3).to_csv(out / "groups.csv", index=False)
    (out / "trees.json").write_text(json.dumps(trees))
    show = ["gift_group", "foods", "gift_subgroups", "clusters", "ari_vs_subgroup", "other_country_5nn",
            "other_country_expected", "country_mixing",
            "same_subgroup_5nn_composition", "same_subgroup_5nn_fractions", "same_subgroup_5nn_semantics",
            "same_subgroup_5nn_fused"]
    print(groups[[c for c in show if c in groups]].round(2).to_string(index=False))
    print(f"Wrote {out}/")


if __name__ == "__main__":
    main()
