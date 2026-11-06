#!/usr/bin/env python3
"""How much do nutrient fractions change which foods sit together?

    python scripts/compare_totals_vs_fractions.py [--config config/config.yaml] [--k 5]

For three similarity definitions (totals only; totals + measured fractions;
totals + measured fractions + curated protein fractions), reports for every
food group the share of each food's k nearest neighbours that belong to the
same group, plus the share of dairy foods' neighbours that are dairy.
The curated protein profiles are assigned per food category, so part of their
gain in group cohesion is built in; the middle column shows the effect of the
measured fractions alone.

Writes results/comparison_by_group.csv and results/neighbours_examples.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from foodspace import evaluate, fractions, fuse, views

DAIRY = ["Milk", "Hard cheese", "Soft cheese", "Fresh cheese and quark", "Natural fermented products",
         "Whey products", "Cream, creme fraiche", "Flavoured fermented milk", "Processed cheese", "Butter"]


def distances(df, cfg):
    cols = cfg["columns"]
    Xp, _ = views.prepare_nutrients(df[cols["nutrients"]], groups=df[cols["gift_group"]])
    Dn = views.nutrient_distance(Xp, cfg["nutrients"]["metric"])
    fcfg = cfg["fractions"]
    curated = pd.read_csv(fcfg["curated_table"]) if fcfg.get("curated_table") else None
    measured = dict(fcfg, blocks={k: v for k, v in fcfg["blocks"].items() if v.get("source") != "curated"})
    Dm, _ = fractions.fraction_distance(fractions.build_blocks(df, measured, None))
    Da, _ = fractions.fraction_distance(fractions.build_blocks(df, fcfg, curated))
    both = lambda Df: fuse.weighted_fusion({"n": Dn, "f": Df}, {"n": 1, "f": 1})
    return {"totals": Dn, "measured_fractions": both(Dm), "all_fractions": both(Da)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--group-col", default="main_group", help="reference grouping to score against")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    df = pd.read_csv(cfg["input"])
    df[args.group_col] = df[args.group_col].astype(str).str.strip()
    grp = df[args.group_col].to_numpy()
    is_dairy = np.isin(grp, DAIRY)
    Ds = distances(df, cfg)

    rows, nn = [], {}
    for name, D in Ds.items():
        idx = evaluate.nearest(D, args.k)
        nn[name] = idx
        same = grp[idx] == grp[:, None]
        for g in pd.unique(grp):
            m = grp == g
            rows.append({"group": g, "n_foods": int(m.sum()), "view": name, "same_group_neighbours": same[m].mean()})
        rows.append({"group": "ALL FOODS", "n_foods": len(df), "view": name, "same_group_neighbours": same.mean()})
        rows.append({"group": "DAIRY (neighbours that are dairy)", "n_foods": int(is_dairy.sum()), "view": name,
                     "same_group_neighbours": is_dairy[idx[is_dairy]].mean()})
    t = pd.DataFrame(rows).pivot_table(index=["group", "n_foods"], columns="view", values="same_group_neighbours").reset_index()
    t = t[["group", "n_foods", "totals", "measured_fractions", "all_fractions"]].sort_values("n_foods", ascending=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t.round(3).to_csv(out / "comparison_by_group.csv", index=False)

    names = df["name"].to_numpy()
    ex = pd.DataFrame({"food": names, "group": grp,
                       **{f"neighbours_{v}": [" | ".join(names[j] for j in row) for row in nn[v]] for v in Ds}})
    ex.to_csv(out / "neighbours_examples.csv", index=False)
    print(t.head(25).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
