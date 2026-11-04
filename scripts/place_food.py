#!/usr/bin/env python3
"""Place a new food in the food space by its name and, if known, its nutrients.

    python scripts/place_food.py "Oat yoghurt, strawberry" --db nordic \\
        --set PROT_g=1.2 FAT_g=2.5 CHO_g=11 FIBT_g=0.8 SUGAR_g=8
    python scripts/place_food.py --evaluate --db se

The name picks the GIFT group (unless --group is given) and the closest
existing foods; the nutrients decide the position and cluster inside the
group. Values not given are filled from the name neighbours and listed as
imputed. --evaluate reports how often names alone recover the GIFT group and
subgroup (leave-one-out).
"""

import argparse
import json

from foodspace.placement import FoodSpace

DBS = {"se": ("data/processed/foods_lv.csv", "results/within_gift"),
       "fi": ("data/processed/foods_fi.csv", "results/within_gift_fi"),
       "nordic": ("data/processed/foods_nordic.csv", "results/within_gift_nordic")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?")
    ap.add_argument("--db", choices=DBS, default="nordic")
    ap.add_argument("--set", nargs="*", default=[], help="nutrient values, e.g. PROT_g=3.4 FAT_g=1.5")
    ap.add_argument("--group", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args(argv)
    fs = FoodSpace(*DBS[args.db])
    if args.evaluate:
        print(json.dumps(fs.evaluate_names(), indent=2))
        return
    vals = {k: float(v) for k, v in (s.split("=", 1) for s in args.set)}
    p = fs.place(args.name, vals, gift_group=args.group, k=args.k)
    print(json.dumps(p.__dict__, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
