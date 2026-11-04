#!/usr/bin/env python3
"""Treelet trees for the food groups that dietary recommendations are set for.

    python scripts/recommendation_groups.py [--records data/simulated/food_records]
        [--out results/recommendation_groups/groups.json]

Nine groups, built from GIFT groups and subgroups:
    Grains            Cereals and their products
    Legumes           Pulses (excluding soybeans) + Soybean and soy-based products
    Nuts / Seeds      Nuts, seeds and their products, split by name (oilseeds -> Seeds)
    Milk products     Milk and milk products
    Meat and eggs     Meat and meat products + Eggs and their products
    Seafood           Fish, shellfish and their products
    Fruits            Fruits and their products
    Vegetables        Vegetables and their products

Servings: 7 g protein for milk products, nuts, seeds and legumes; 14 g protein for
meat and eggs and for seafood; 30 g starch for grains; 80 g of food for fruits and
vegetables.

Within each group: macronutrient distance (protein, fat, carbohydrate, fibre per
100 g; log1p, z-scored within the group, cosine), fraction distance (fat, sugar,
carbohydrate and protein fractions) and their equal-weight fusion, as in
within_gift_groups.py. The treelet tree is built on 1 - fused distance; cutting it
after p - k merges gives k branches. Exported per group: foods (with intake from the
food records, in grams and in servings), tree linkage, a 2D and a 3D layout and each food's nearest neighbours under
both distances.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from foodspace import fractions, fuse, layout, treelet as tl, views

SEEDS = re.compile(r"\b(seed|seeds|sesame|tahini|sunflower|pumpkin|linseeds?|flax(?:seeds?)?|chia|hemp|poppy|psyllium)\b", re.I)
REC = {  # (GIFT group, GIFT subgroup or None for all) -> recommendation group
    ("Cereals and their products", None): "Grains",
    ("Pulses, nuts and seeds and their products", "Pulses (excluding soybeans) and their products"): "Legumes",
    ("Pulses, nuts and seeds and their products", "Soybean and soy-based products"): "Legumes",
    ("Pulses, nuts and seeds and their products", "Nuts, seeds and their products"): "Nuts",
    ("Milk and milk products", None): "Milk products",
    ("Meat and meat products", None): "Meat and eggs",
    ("Eggs and their products", None): "Meat and eggs",
    ("Fish, shellfish and their products", None): "Seafood",
    ("Fruits and their products", None): "Fruits",
    ("Vegetables and their products", None): "Vegetables",
}
# one serving, by what is counted: protein (g), starch (g) or food weight (g)
SERVING = {"Milk products": ("PROT_g", 7), "Meat and eggs": ("PROT_g", 14), "Seafood": ("PROT_g", 14),
           "Nuts": ("PROT_g", 7), "Seeds": ("PROT_g", 7), "Legumes": ("PROT_g", 7),
           "Grains": ("starch_est_g", 30), "Fruits": (None, 80), "Vegetables": (None, 80)}
ORDER = ["Grains", "Legumes", "Nuts", "Seeds", "Milk products", "Meat and eggs", "Seafood", "Fruits", "Vegetables"]
MACROS = ["PROT_g", "FAT_g", "CHO_g", "FIBT_g", "SUGAR_g", "ENERC_kcal"]


FIXES = [  # GIFT assignments that do not fit a recommendation group (from FoodEx2 coding)
    (re.compile(r"\bsea buckthorn\b", re.I), "Fruits"),
    (re.compile(r"\bsubstitute\b.*\boat based\b|\boat based\b.*\bsubstitute\b", re.I), None),  # oat yoghurts coded as nuts
]


def rec_group(row):
    g = REC.get((row["gift_group"], row["gift_subgroup"])) or REC.get((row["gift_group"], None))
    for rx, to in FIXES:
        if g and rx.search(str(row["name"])):
            return to
    if g == "Nuts" and SEEDS.search(str(row["name"])) and not re.search(r"\bnut", str(row["name"]), re.I):
        g = "Seeds"
    return g


def knn(D, k):
    idx = np.argsort(D, axis=1)[:, 1 : k + 1]
    return [[[int(j), round(float(D[i, j]), 4)] for j in idx[i]] for i in range(len(D))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--records", default="data/simulated/food_records")
    ap.add_argument("--k", type=int, default=8, help="nearest neighbours exported per food")
    ap.add_argument("--out", default="results/recommendation_groups/groups.json")
    args = ap.parse_args(argv)

    spec = importlib.util.spec_from_file_location("wgg", Path(__file__).with_name("within_gift_groups.py"))
    wgg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wgg)

    cfg = yaml.safe_load(Path(args.config).read_text())
    df = pd.read_csv(cfg["input"])
    df["rec_group"] = df.apply(rec_group, axis=1)
    df = df[df["rec_group"].notna()].reset_index(drop=True)
    curated = pd.read_csv(cfg["fractions"]["curated_table"])
    measured = {k: v for k, v in cfg["fractions"]["blocks"].items() if v.get("source") != "curated"}
    blocks = fractions.build_blocks(df, {"blocks": measured}, None)
    blocks["protein_type"] = wgg.protein_block(df, cfg, curated, wgg.cereal_profiles_ok("results/cereals/cereal_profiles.csv"))
    cols = cfg["columns"]["nutrients"]

    c = pd.read_csv(Path(args.records) / "consumption.csv")
    c = c[c["in_database"]]
    pdays = c.groupby(["SUBJECT", "SURVEY_DAY"]).ngroups
    ig = c.groupby("food_id")
    intake = pd.DataFrame({"g": ig["FOOD_AMOUNT_CONS"].sum() / pdays, "people": ig["SUBJECT"].nunique()})
    intake.index = intake.index.astype(int)

    out_groups = []
    for name in ORDER:
        sub = df[df["rec_group"] == name]
        ix = sub.index.to_numpy()
        n = len(ix)
        if n < 3:
            continue
        Xp, _ = views.prepare_nutrients(sub[cols].reset_index(drop=True))
        Dn = views.nutrient_distance(Xp, cfg["nutrients"]["metric"])
        Fd, _ = fractions.fraction_distance({k: B.iloc[ix].reset_index(drop=True) for k, B in blocks.items()})
        D = fuse.weighted_fusion({"comp": Dn, "frac": Fd}, {"comp": 1, "frac": 1})
        res = tl.treelet(1.0 - D)
        xy = layout.embed_2d(D, "umap" if n >= 15 else "tsne")
        xy = (xy - xy.mean(0)) / (np.abs(xy - xy.mean(0)).max() or 1)
        xyz = layout.embed(D, "umap" if n >= 15 else "tsne", dim=3, min_dist=0.25)
        # robust scaling per axis so a few far-out foods do not squash the rest; clipped at 1.15
        xyz = xyz - np.median(xyz, 0)
        xyz = np.clip(xyz / np.maximum(np.percentile(np.abs(xyz), 95, axis=0), 1e-9), -1.15, 1.15)
        foods = []
        basis, size = SERVING[name]
        for j, (_, r) in enumerate(sub.iterrows()):
            fid = int(r["food_id"])
            it = intake.loc[fid] if fid in intake.index else None
            per100 = 100.0 if basis is None else (float(r[basis]) if pd.notna(r[basis]) else 0.0)
            sv100 = per100 / size  # servings in 100 g of the food
            foods.append({"id": fid, "name": str(r["name"]).strip(), "sub": r["gift_subgroup"],
                          "x": round(float(xy[j, 0]), 4), "y": round(float(xy[j, 1]), 4),
                          "xyz": [round(float(v), 4) for v in xyz[j]],
                          "g": round(float(it["g"]), 2) if it is not None else 0.0,
                          "people": int(it["people"]) if it is not None else 0,
                          "sv100": round(sv100, 3),
                          "sv": round(float(it["g"]) * sv100 / 100, 3) if it is not None else 0.0,
                          **{m.replace("_g", "").replace("ENERC_", ""): (round(float(r[m]), 1) if pd.notna(r[m]) else None)
                             for m in MACROS}})
        k = min(args.k, n - 1)
        out_groups.append({"name": name, "n": n, "foods": foods,
                           "serving": {"basis": {"PROT_g": "protein", "starch_est_g": "starch", None: "food weight"}[basis], "g": size},
                           "linkage": np.round(res["linkage"][:, :2], 0).astype(int).tolist(),
                           "knn_macro": knn(Dn, k), "knn_fused": knn(D, k)})
        print(f"{name:14s} {n:4d} foods, eaten: {sum(f['g'] > 0 for f in foods):3d}, "
              f"servings/person-day: {sum(f['sv'] for f in foods):.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"person_days": int(pdays), "people": int(c["SUBJECT"].nunique()), "groups": out_groups},
                              separators=(",", ":")))
    print(f"-> {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
