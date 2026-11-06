#!/usr/bin/env python3
"""Cluster cereal foods among themselves by protein-fraction composition.

    python scripts/within_group_cereals.py [--config config/config.yaml]
        [--commodities data/raw/livsmedelsverket/raw_commodities.csv]
        [--min-coverage 0.5] [--min-profiled 0.9] [--k 5] [--resolution 1.0]

Steps
1. For every food, work out how much of its protein comes from each grain
   (raw-commodity recipe x commodity protein), and the protein-weighted
   Osborne profile (albumin+globulin / prolamin / glutelin) from the curated
   table.
2. Keep cereal foods: grains make up at least `min_grain_mass` of the recipe
   (water excluded), they supply `min_coverage`-`max_coverage` of the food's
   protein (above 1 happens when the grain was refined to starch or the
   recipe factors are rough), and at least `min_profiled` of that grain
   protein comes from grains with a curated profile.
3. Compare three similarity views within that subset, as in
   compare_totals_vs_fractions.py:
       totals                macros per 100 g
       measured_fractions    + fat, sugar and carbohydrate type blocks
       protein_fractions     + measured blocks + the Osborne profile
   scored against two references: the dominant grain (>= 60 % of grain
   protein, else "mixed") and the database's food group.
4. Leiden clusters on the protein_fractions view, with a summary per cluster.

The Osborne profile is computed from the grain mix, so agreement with the
dominant grain is partly built in. What the profile adds is a chemical scale:
grains with similar fractions (wheat and barley: prolamin-rich) sit close,
oats and rice (glutelin-rich) sit close, rye (albumin-rich) stands apart, and
wheat-rye blends fall in between.

Writes results/cereals/{cereal_profiles.csv, comparison.csv, clusters.csv,
cluster_summary.csv, map.json}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from foodspace import commodities as cm
from foodspace import evaluate, fractions, fuse, layout, views

OSBORNE = cm.OSBORNE


def knn_share(D, labels, k):
    idx = evaluate.nearest(D, k)
    return (labels[idx] == labels[:, None]).mean(), idx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--commodities", default="data/raw/livsmedelsverket/raw_commodities.csv")
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--min-grain-mass", type=float, default=0.5,
                    help="grain commodities' share of the recipe (water excluded)")
    ap.add_argument("--max-coverage", type=float, default=2.0,
                    help="above this the recipe lists grain that was refined to starch, so grain protein overstates the food's")
    ap.add_argument("--min-profiled", type=float, default=0.9)
    ap.add_argument("--min-protein", type=float, default=1.0, help="g/100 g; below this the profile is noise")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--out", default="results/cereals")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    df = pd.read_csv(cfg["input"])
    df["main_group"] = df["main_group"].astype(str).str.strip()
    table = pd.read_csv(cfg["fractions"]["curated_table"])

    mixes = cm.load_mixes(args.commodities)
    recipe = cm.load_mixes(args.commodities, "share_pct")
    prot = cm.commodity_protein(df, mixes, list(cm.GRAIN_OF))
    shares = cm.grain_shares(table)
    prof = cm.product_profiles(df, mixes, prot, shares)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prof["grain_mass_share"] = prof["food_id"].map(lambda i: cm.grain_mass_share(recipe.get(i, {})))
    allp = df[["food_id", "name", "main_group", "PROT_g"]].merge(prof, on="food_id")
    allp.round(4).to_csv(out / "cereal_profiles.csv", index=False)

    keep = (allp["grain_mass_share"] >= args.min_grain_mass) & allp["coverage"].between(args.min_coverage, args.max_coverage) & (allp["profiled_share"] >= args.min_profiled) \
        & (allp["PROT_g"] >= args.min_protein)
    sel = allp[keep].reset_index(drop=True)
    sub = df.set_index("food_id").loc[sel["food_id"]].reset_index()
    print(f"{len(allp)} foods contain grain; {len(sel)} kept as cereal foods "
          f"(grain >= {args.min_grain_mass} of recipe, coverage {args.min_coverage}-{args.max_coverage}, profiled >= {args.min_profiled}, protein >= {args.min_protein} g)")

    # views
    cols = cfg["columns"]["nutrients"]
    Xp, _ = views.prepare_nutrients(sub[cols], groups=sub["main_group"])
    Dn = views.nutrient_distance(Xp, cfg["nutrients"]["metric"])
    fcfg = cfg["fractions"]
    measured = {k: v for k, v in fcfg["blocks"].items() if v.get("source") != "curated"}
    mblocks = fractions.build_blocks(sub, dict(fcfg, blocks=measured), None)
    Dm, _ = fractions.fraction_distance(mblocks)
    pblock = sel[OSBORNE].copy()
    pblock.index = sub.index
    Dp, _ = fractions.fraction_distance({**mblocks, "protein_osborne": pblock})
    Dprot_only, _ = fractions.fraction_distance({"protein_osborne": pblock})
    both = lambda Df: fuse.weighted_fusion({"n": Dn, "f": Df}, {"n": 1, "f": 1})
    Ds = {"totals": Dn, "measured_fractions": both(Dm), "protein_fractions": both(Dp)}

    grain = sel["dominant_grain"].to_numpy()
    group = sel["main_group"].to_numpy()
    rows = []
    for v, D in Ds.items():
        g_all, _ = knn_share(D, grain, args.k)
        m_all, _ = knn_share(D, group, args.k)
        rows.append({"view": v, "same_dominant_grain": g_all, "same_food_group": m_all})
        idx = evaluate.nearest(D, args.k)
        for gname in pd.unique(grain):
            m = grain == gname
            rows.append({"view": v, "subset": f"grain={gname}", "n": int(m.sum()),
                         "same_dominant_grain": (grain[idx[m]] == gname).mean()})
    comp = pd.DataFrame(rows)
    comp.round(3).to_csv(out / "comparison.csv", index=False)
    print(comp[comp["subset"].isna()].drop(columns=["subset", "n"]).round(2).to_string(index=False))
    print(comp[comp["subset"].notna()].pivot_table(index=["subset", "n"], columns="view",
                                                   values="same_dominant_grain").round(2).to_string())

    # clusters on the protein view
    D = Ds["protein_fractions"]
    W = fuse.knn_sparsify(fuse.affinity(D, k=min(15, len(sel) - 1)), min(15, len(sel) - 1))
    cl = fuse.leiden(W, resolution=args.resolution, seed=cfg.get("seed", 0))
    xy = layout.embed_2d(D, method="umap", seed=cfg.get("seed", 0))
    res = sel.assign(cluster=cl, x=xy[:, 0], y=xy[:, 1])
    res.round(4).to_csv(out / "clusters.csv", index=False)

    grains = ["wheat", "rye", "oats", "barley", "maize", "rice"]
    summ = res.groupby("cluster").agg(
        n=("food_id", "size"),
        **{f: (f, "mean") for f in OSBORNE},
        **{f"{g}_share": (f"{g}_protein_share", "mean") for g in grains},
        food_groups=("main_group", lambda s: "; ".join(f"{k} ({v})" for k, v in s.value_counts().head(3).items())),
        examples=("name", lambda s: " | ".join(s.head(4))),
    ).reset_index()
    summ.round(3).to_csv(out / "cluster_summary.csv", index=False)
    print(summ.drop(columns=["examples"]).round(2).to_string(index=False))

    # small JSON for a map page
    json.dump({"foods": res[["food_id", "name", "main_group", "dominant_grain", "cluster", "x", "y", "PROT_g", "coverage",
                             *OSBORNE, *[f"{g}_protein_share" for g in grains]]].round(3).to_dict(orient="records"),
               "comparison": comp.round(3).replace({np.nan: None}).to_dict(orient="records"),
               "settings": vars(args)},
              open(out / "map.json", "w"))
    print(f"Wrote {out}/")


if __name__ == "__main__":
    main()
