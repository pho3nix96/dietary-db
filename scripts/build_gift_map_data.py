#!/usr/bin/env python3
"""Compact JSON for the within-GIFT-group map page (from results/within_gift)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

R = Path(sys.argv[1] if len(sys.argv) > 1 else "results/within_gift")
TABLE = sys.argv[2] if len(sys.argv) > 2 else "data/processed/foods_lv.csv"
foods = pd.read_csv(R / "foods.csv")
groups = pd.read_csv(R / "groups.csv")
lv = pd.read_csv(TABLE)[["food_id", "PROT_g", "FAT_g", "CHO_g", "FIBT_g", "SUGAR_g"]]
if "source" in foods and "source" in lv:
    lv = lv.drop(columns="source")
foods = foods.merge(lv, on="food_id", how="left")
m = pd.read_csv("data/raw/gift/foodex2_gift_nut.csv", dtype=str)
short_sub = dict(zip(m["gift_subgroup"], m["gift_subgroup_short"]))
short_grp = dict(zip(m["gift_group"], m["gift_group_short"]))
out = []
for _, g in groups.iterrows():
    f = foods[foods["gift_group"] == g["gift_group"]]
    subs = f["gift_subgroup"].value_counts().index.tolist()
    rows = []
    for _, r in f.iterrows():
        nm = r["name"] if not str(r["name"]).isupper() else str(r["name"]).capitalize()
        rows.append([nm, subs.index(r["gift_subgroup"]), int(r["cluster"]), round(r["x"], 4), round(r["y"], 4),
                     *[None if pd.isna(r[c]) else round(float(r[c]), 1) for c in ("PROT_g", "FAT_g", "CHO_g", "FIBT_g", "SUGAR_g")],
                     1 if r["protein_profile"] else 0,
                     ({"SE": 0, "FI": 1}.get(r["source"], -1) if "source" in r else -1)])
    stats = {k: (None if pd.isna(g.get(k)) else round(float(g[k]), 3)) for k in
             ("ari_vs_subgroup", "nmi_vs_subgroup", "same_subgroup_5nn_semantics", "same_subgroup_5nn_composition",
              "same_subgroup_5nn_fractions", "same_subgroup_5nn_fused", "country_mixing", "country_mixing_semantics",
              "country_mixing_composition", "country_mixing_fractions") if k in g}
    out.append({"name": g["gift_group"], "short": short_grp.get(g["gift_group"], g["gift_group"]),
                "subgroups": [short_sub.get(s, s) for s in subs], "subgroups_full": subs,
                "clusters": int(g["clusters"]), "foods": rows, "stats": stats})
p = R / "map.json"  # one database
p.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
print(len(out), "groups", p.stat().st_size // 1024, "KB")
