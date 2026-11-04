#!/usr/bin/env python3
"""Place the unique products of a food-record dataset in the food space.

    python scripts/place_unique_foods.py [--records data/simulated/food_records] [--db se]

Unique products are rows whose food is not in the composition table
(in_database = False). Each is placed with foodspace.placement: its name picks
the GIFT group and name neighbours; its recorded nutrient values (if any)
decide its position and cluster. Output: placed_unique_foods.csv next to the
records, and, when unique_foods_truth.csv is present (simulated data), how
often placement recovers the true GIFT group, subgroup and base food.
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

from foodspace.placement import FoodSpace

DBS = {"se": ("data/processed/foods_lv.csv", "results/within_gift"),
       "fi": ("data/processed/foods_fi.csv", "results/within_gift_fi"),
       "nordic": ("data/processed/foods_nordic.csv", "results/within_gift_nordic")}
VALUE_COLS = ["PROT_g", "FAT_g", "CHO_g", "FIBT_g", "SUGAR_g", "FASAT_g", "FAMS_g", "FAPU_g"]


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default="data/simulated/food_records")
    ap.add_argument("--db", choices=DBS, default="se")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args(argv)
    rec = Path(args.records)
    truth_path = rec / "unique_foods_truth.csv"
    if truth_path.exists():
        u = pd.read_csv(truth_path)
    else:  # real data: unique products = rows not in the database, values from the row per 100 g
        c = pd.read_csv(rec / "consumption.csv")
        u = c[~c["in_database"]].rename(columns={"INGREDIENT_ENG": "name"})[["unique_food_id", "name"]]
    fs = FoodSpace(*DBS[args.db])
    rows = []
    for _, r in u.iterrows():
        vals = {c: r[f"{c}_per100"] for c in VALUE_COLS if f"{c}_per100" in r and pd.notna(r[f"{c}_per100"])}
        p = fs.place(r["name"], vals, k=args.k)
        rows.append({"unique_food_id": r["unique_food_id"], "name": r["name"], "has_values": bool(vals),
                     "gift_group": p.gift_group, "gift_subgroup": p.gift_subgroup,
                     "group_confidence": round(p.group_confidence, 3), "cluster": p.cluster,
                     "x": round(p.x, 4), "y": round(p.y, 4), "basis": p.basis,
                     "nearest_foods": " | ".join(n for n, _ in p.composition_neighbours[:3]),
                     "imputed": ",".join(p.imputed), "notes": " ".join(p.notes)})
    out = pd.DataFrame(rows)
    out.to_csv(rec / "placed_unique_foods.csv", index=False)
    print(f"Placed {len(out)} unique products -> {rec / 'placed_unique_foods.csv'}")

    if truth_path.exists():
        m = out.merge(u[["unique_food_id", "base_name", "true_gift_group", "true_gift_subgroup"]], on="unique_food_id")
        m["group_ok"] = m["gift_group"] == m["true_gift_group"]
        m["subgroup_ok"] = m["gift_subgroup"] == m["true_gift_subgroup"]
        m["base_in_top3"] = [str(b) in str(n) for b, n in zip(m["base_name"], m["nearest_foods"])]
        s = m.groupby("has_values")[["group_ok", "subgroup_ok", "base_in_top3"]].mean().round(2)
        s.index = s.index.map({True: "name + values", False: "name only"})
        s["n"] = m.groupby("has_values").size().values
        print(s.to_string())
        low = m[m["group_confidence"] < 0.6]
        print(f"low-confidence group calls: {len(low)} (of which wrong: {int((~low['group_ok']).sum())}); "
              f"high-confidence wrong: {int(((m['group_confidence'] >= 0.6) & ~m['group_ok']).sum())}")
        m.to_csv(rec / "placed_unique_foods_vs_truth.csv", index=False)


if __name__ == "__main__":
    main()
