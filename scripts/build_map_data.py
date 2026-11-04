#!/usr/bin/env python3
"""Build the data file for the interactive Swedish food space map.

    python scripts/build_map_data.py [--out results/map/lv_map.json]

For every food: name, food group, family (for highlighting), 2D positions on
totals-only and totals+fractions similarity (UMAP, the second rotated onto the
first with Procrustes so the views share orientation), the nutrient values
shown in the side panel, the 3 nearest neighbours in each view and the protein
category. Also writes the same-group neighbour comparison per food group
(5 nearest neighbours), as in compare_totals_vs_fractions.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.linalg import orthogonal_procrustes

sys.path.insert(0, str(Path(__file__).parent))
from compare_totals_vs_fractions import distances  # noqa: E402

from foodspace import layout  # noqa: E402

FAMILIES = {
    "Dairy": ["Milk", "Hard cheese", "Soft cheese", "Fresh cheese and quark", "Natural fermented products",
              "Whey products", "Cream, creme fraiche", "Flavoured fermented milk", "Processed cheese", "Butter"],
    "Meat": ["Red meat fresh frozen cooked", "Processed meat", "Offals", "Sausages", "Poultry", "Brawn"],
    "Fish": ["Fish fresh frozen cooked", "Fish smoked", "Fish fried", "Shellfish squid fresh frozen cooked",
             "Roe, roe spread"],
    "Cereals": ["Soft bread", "Crisp bread", "Flour starch bran", "Breakfast cereals", "Porridge", "Gruel",
                "Pasta", "Rice rice noodles", "Grain bulgur, couscous", "Salty biscuits", "Rice cakes"],
}
PANEL = ["PROT_g", "FAT_g", "CHO_g", "FIBT_g", "SUGAR_g", "FASAT_g", "FAMS_g", "FAPU_g", "SUCS_g",
         "other_disac_g", "starch_est_g"]


def scaled(xy):
    xy = xy - xy.mean(axis=0)
    return xy / np.abs(xy).max()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--out", default="results/map/lv_map.json")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    df = pd.read_csv(cfg["input"])
    df["main_group"] = df["main_group"].astype(str).str.strip()
    fam_of = {g: f for f, gs in FAMILIES.items() for g in gs}
    fam = df["main_group"].map(fam_of).fillna("")
    Ds = distances(df, cfg)

    xy_t = scaled(layout.embed_2d(Ds["totals"], "umap"))
    xy_f = scaled(layout.embed_2d(Ds["all_fractions"], "umap"))
    R, _ = orthogonal_procrustes(xy_f, xy_t)
    xy_f = scaled(xy_f @ R)

    nn = {v: np.argsort(Ds[v], axis=1)[:, 1:4] for v in ("totals", "all_fractions")}
    vals = df[PANEL].apply(pd.to_numeric, errors="coerce").round(1)
    pc = df.get("protein_fraction_category", pd.Series([None] * len(df)))
    foods = []
    for i in range(len(df)):
        foods.append([df["name"][i], df["main_group"][i], fam[i],
                      round(float(xy_t[i, 0]), 4), round(float(xy_t[i, 1]), 4),
                      round(float(xy_f[i, 0]), 4), round(float(xy_f[i, 1]), 4),
                      [None if pd.isna(v) else float(v) for v in vals.iloc[i]],
                      nn["totals"][i].tolist(), nn["all_fractions"][i].tolist(),
                      None if pd.isna(pc[i]) else pc[i]])

    grp = df["main_group"].to_numpy()
    rows = []
    for v, D in Ds.items():
        idx = np.argsort(D, axis=1)[:, 1 : args.k + 1]
        same = grp[idx] == grp[:, None]
        for g in pd.unique(grp):
            m = grp == g
            rows.append({"group": g, "n_foods": int(m.sum()), "view": v, "s": same[m].mean()})
        rows.append({"group": "ALL FOODS", "n_foods": len(df), "view": v, "s": same.mean()})
        for f, gs in FAMILIES.items():
            m = np.isin(grp, gs)
            rows.append({"group": f"{f.upper()} (neighbours in family)", "n_foods": int(m.sum()), "view": v,
                         "s": np.isin(grp[idx[m]], gs).mean()})
    t = pd.DataFrame(rows).pivot_table(index=["group", "n_foods"], columns="view", values="s").reset_index()
    comparison = t.round(3).to_dict(orient="records")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"foods": foods, "families": list(FAMILIES), "comparison": comparison},
                              separators=(",", ":"), ensure_ascii=False))
    fam_rows = [r for r in comparison if "family" in r["group"] or r["group"] == "ALL FOODS"]
    print(pd.DataFrame(fam_rows).to_string(index=False))
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
