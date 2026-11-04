#!/usr/bin/env python3
"""Pool the Swedish and Finnish food tables into one.

    python scripts/combine_databases.py [--out data/processed/foods_nordic.csv]

Keeps the columns both tables have in the same meaning (per 100 g edible
portion): macros, sugar parts, starch, fat classes, omega-6/ALA/long-chain
omega-3, recipe-based protein fractions, GIFT group and subgroup, names.
Swedish-only columns (C4-C10 milk-fat marker) are dropped so neither country
gets an extra block. food_id becomes 1xxxxxx for Sweden and 2xxxxxx for
Finland; `source` says which.

Known method differences, reported by this script on matched food types:
starch is measured in Fineli and estimated (CHO - sugars) in the Swedish table;
the omega-3/omega-6 figures are summed from individual fatty acids in Sweden
and given as totals in Fineli.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEEP = ["name", "main_group", "gift_group", "gift_subgroup", "gift_source",
        "PROT_g", "FAT_g", "CHO_g", "FIBT_g", "SUGAR_g", "MNSAC_g", "SUCS_g", "other_disac_g", "starch_est_g",
        "FASAT_g", "FAMS_g", "FAPU_g", "n6_g", "ala_g", "lc_n3_g",
        "protein_profiled_share", "protein_main_source"]

CHECKS = {  # food type: (Swedish name regex, Finnish name regex)
    "milk ~1.5% fat": (r"^Milk 1\.5% fat", r"^MILK, 1\.5% FAT"),
    "cod, raw": (r"^Cod raw", r"^COD$"),
    "rolled oats": (r"^Rolled oats", r"^OAT FLAKES|^ROLLED OATS"),
    "rye crisp bread": (r"^Crisp bread wholegrain rye", r"CRISPBREAD, RYE|RYE CRISPBREAD"),
    "salmon, raw": (r"^Salmon farmed", r"^SALMON, FARMED|^SALMON$"),
    "rapeseed oil": (r"^Rapeseed oil", r"^RAPESEED OIL"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--se", default="data/processed/foods_lv.csv")
    ap.add_argument("--fi", default="data/processed/foods_fi.csv")
    ap.add_argument("--out", default="data/processed/foods_nordic.csv")
    args = ap.parse_args(argv)

    parts = []
    for src, path, base in (("SE", args.se, 1_000_000), ("FI", args.fi, 2_000_000)):
        d = pd.read_csv(path)
        pf = [c for c in d.columns if c.startswith("protfrac_")]
        t = d[[c for c in KEEP if c in d] + pf].copy()
        t.insert(0, "food_id", base + d["food_id"].astype(int))
        t.insert(1, "source", src)
        t.insert(2, "source_food_id", d["food_id"].astype(int))
        if src == "FI":
            t["name"] = t["name"].str.capitalize()
        parts.append(t)
    df = pd.concat(parts, ignore_index=True)
    df = df[df["gift_group"].notna() & (df["gift_group"] != "unassigned")].reset_index(drop=True)
    pf = [c for c in df.columns if c.startswith("protfrac_")]
    df[pf] = df[pf].where(df[pf].notna().all(axis=1), axis=0)  # all-or-nothing per food

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} foods ({df['source'].value_counts().to_dict()}) -> {out}")

    cols = ["PROT_g", "FAT_g", "CHO_g", "FIBT_g", "SUGAR_g", "starch_est_g", "other_disac_g", "FAPU_g", "n6_g", "lc_n3_g"]
    rows = []
    for label, (se_rx, fi_rx) in CHECKS.items():
        for src, rx in (("SE", se_rx), ("FI", fi_rx)):
            m = df[(df["source"] == src) & df["name"].str.contains(rx, case=False, regex=True, na=False)]
            if len(m):
                rows.append({"food": label, "db": src, "n": len(m), **m[cols].median().round(2).to_dict()})
    if rows:
        print("\nSame food types in both databases (medians, g/100 g):")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
