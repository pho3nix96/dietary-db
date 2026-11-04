#!/usr/bin/env python3
"""Compare people's diets against the food reference.

    python scripts/compare_diets.py [--records data/simulated/food_records]
        [--foods data/processed/foods_lv.csv] [--out results/diet_similarity]
        [--weight ENERGY_kcal] [--dish-shares PATH]

Writes
    matching_report.json   how record lines were linked to the reference
    person_distance.csv    people x people distance (food-tree view + nutrient view)
    person_clusters.csv    Leiden clusters of people
    summary.json           settings and, if truth.csv exists, how well clusters recover the
                           hidden dietary patterns (simulated records only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foodspace import diet, evaluate, fuse  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default="data/simulated/food_records")
    ap.add_argument("--foods", default="data/processed/foods_lv.csv")
    ap.add_argument("--out", default="results/diet_similarity")
    ap.add_argument("--weight", default="ENERGY_kcal", help="ENERGY_kcal or FOOD_AMOUNT_CONS (grams)")
    ap.add_argument("--dish-shares", default=None, help="CSV food_id, leaf, share: spreads composite dishes over ingredient leaves")
    ap.add_argument("--k", type=int, default=10, help="neighbours per person for clustering")
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--nutrient-weight", type=float, default=0.25,
                    help="weight of the nutrient view (food-tree view = 1); on the simulated records the food-tree view alone recovered patterns best")
    args = ap.parse_args(argv)

    rec_dir, out = Path(args.records), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    foods = pd.read_csv(args.foods, low_memory=False)
    foods = foods[foods["gift_group"].notna() & (foods["gift_group"] != "unassigned")]
    records = pd.read_csv(rec_dir / "consumption.csv", low_memory=False)
    plc_path = rec_dir / "placed_unique_foods.csv"
    placements = pd.read_csv(plc_path) if plc_path.exists() else None

    matched = diet.match_records(records, foods, placements)
    report = diet.match_report(matched, args.weight)
    (out / "matching_report.json").write_text(json.dumps(report, indent=2))
    print("Matching:", json.dumps(report["by_match"]))

    leaf_index, parent, length, _ = diet.food_tree(foods, matched)
    dish = pd.read_csv(args.dish_shares) if args.dish_shares else None
    P = diet.diet_profiles(matched, leaf_index, args.weight, dish_shares=dish)
    people = P.index.to_numpy()

    views = {
        "food_tree": diet.tree_distance(P, leaf_index, parent, length),
        "nutrients": diet.nutrient_distance(matched, people=people),
    }
    D = diet.person_distance(views, {"food_tree": 1.0, "nutrients": args.nutrient_weight})
    pd.DataFrame(D, index=people, columns=people).to_csv(out / "person_distance.csv")

    W = fuse.knn_sparsify(fuse.affinity(D, args.k), args.k)
    clusters = fuse.leiden(W, args.resolution, 0)
    pd.DataFrame({"SUBJECT": people, "cluster": clusters}).to_csv(out / "person_clusters.csv", index=False)

    summary = {"people": int(len(people)), "weight": args.weight, "k": args.k, "resolution": args.resolution,
               "n_clusters": int(len(set(clusters))), "views": list(views), "dish_shares": bool(dish is not None)}
    truth_path = rec_dir / "truth.csv"
    if truth_path.exists():
        truth = pd.read_csv(truth_path).set_index("SUBJECT").reindex(people)
        summary["vs_hidden_pattern"] = evaluate.agreement(clusters, truth["pattern"].astype(str))
        for name, Dv in views.items():  # which view recovers the patterns on its own
            Wv = fuse.knn_sparsify(fuse.affinity(Dv, args.k), args.k)
            summary.setdefault("view_ARI", {})[name] = round(float(
                evaluate.agreement(fuse.leiden(Wv, args.resolution, 0), truth["pattern"].astype(str))["ARI"]), 3)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
