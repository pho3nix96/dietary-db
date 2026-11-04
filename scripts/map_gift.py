#!/usr/bin/env python3
"""Build the FoodEx2 -> GIFT nutrition-sensitive grouping table.

    python scripts/map_gift.py [--matrix data/raw/gift/foodex2_gift_nut.csv]
        [--out data/curated/foodex2_to_gift.csv]

Input is the "03 Matching table" sheet of FoodEx2_MTX.xlsx (FAO/WHO GIFT,
FoodEx2 matrix version 16.2; https://openknowledge.fao.org/handle/20.500.14283/cd7985en),
reduced to the nutrition-sensitive columns (19 groups, 95 subgroups).

Base terms that GIFT leaves unmapped ("N/A": broad, non-reportable or
deprecated terms) inherit the grouping of their nearest mapped ancestor in the
FoodEx2 exposure hierarchy; the `gift_source` column says which.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", default="data/raw/gift/foodex2_gift_nut.csv")
    ap.add_argument("--out", default="data/curated/foodex2_to_gift.csv")
    args = ap.parse_args(argv)

    m = pd.read_csv(args.matrix, dtype=str).fillna("")
    by_expo = m.set_index("expo_hierarchy_code")
    cols = ["gift_group_code", "gift_group", "gift_subgroup_code", "gift_subgroup"]
    rows = []
    for _, r in m.iterrows():
        src = "matrix"
        vals = r[cols]
        code = r["expo_hierarchy_code"]
        while not vals["gift_group_code"] and "." in code:
            code = code.rsplit(".", 1)[0]
            if code in by_expo.index:
                vals = by_expo.loc[code, cols]
                src = f"ancestor {by_expo.loc[code, 'foodex2_code']}"
        rows.append({"foodex2_code": r["foodex2_code"], "foodex2_desc": r["foodex2_desc"], **vals.to_dict(),
                     "gift_source": src if vals["gift_group_code"] else "unmapped"})
    out = pd.DataFrame(rows).drop_duplicates("foodex2_code")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out["gift_source"].str.split().str[0].value_counts().to_string())
    print(f"Wrote {len(out)} FoodEx2 terms -> {args.out}")


if __name__ == "__main__":
    main()
