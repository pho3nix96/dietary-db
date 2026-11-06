#!/usr/bin/env python3
"""Compare people's diets against the food reference.

    python scripts/compare_diets.py [--records data/simulated/food_records]
        [--foods data/processed/foods_lv.csv] [--out results/diet_similarity]
        [--weight ENERGY_kcal] [--dish-shares PATH] [--ignore-ids]

Record lines are matched to the reference (food id, FoodEx2 code, name), lines without
nutrient values take them from their reference food, and each person becomes a profile
of intake shares over the GIFT food tree (see src/foodspace/diet.py).

Real records: their food ids are not ids of the reference table, so pass --ignore-ids,
and write outputs outside the repository's tracked folders (e.g. --out results/private/...),
since person-level results derived from real records are personal data.

Writes
    matching_report.json   how record lines were linked to the reference, and where nutrient values came from
    person_distance.csv    people x people distance (food-tree view + nutrient view)
    person_clusters.csv    Leiden clusters of people
    summary.json           settings and, if truth.csv exists, how well the distances recover the
                           hidden dietary patterns (simulated records only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    ap.add_argument("--ignore-ids", action="store_true",
                    help="match on FoodEx2 code and name only: use when the records' food_id values are not ids of --foods "
                         "(real records), or to test that path on the simulated ones")
    args = ap.parse_args(argv)

    rec_dir, out = Path(args.records), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    foods = pd.read_csv(args.foods, low_memory=False)
    # foods without a GIFT group stay in the reference (group "unassigned"), so their record lines still match
    foods["gift_group"] = foods["gift_group"].fillna("unassigned")
    foods["gift_subgroup"] = foods["gift_subgroup"].fillna("unassigned")
    records = pd.read_csv(rec_dir / "consumption.csv", low_memory=False)
    plc_path = rec_dir / "placed_unique_foods.csv"
    placements = pd.read_csv(plc_path) if plc_path.exists() else None

    matched = diet.match_records(records, foods, placements, id_col=None if args.ignore_ids else "food_id")
    matched = diet.fill_nutrients(matched, foods, placements)
    report = diet.match_report(matched, args.weight, foods)
    (out / "matching_report.json").write_text(json.dumps(report, indent=2))
    print("Matching:", json.dumps(report["by_match"]), "| nutrients:", json.dumps(report["nutrients_from"]))

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
               "nutrient_weight": args.nutrient_weight, "ignore_ids": bool(args.ignore_ids),
               "n_clusters": int(len(set(clusters))), "views": list(views), "dish_shares": bool(dish is not None)}
    truth_path = rec_dir / "truth.csv"
    if truth_path.exists():  # simulated records only: how well do the distances recover the hidden patterns?
        truth = pd.read_csv(truth_path).set_index("SUBJECT").reindex(people)
        pattern = truth["pattern"].astype(str).to_numpy()
        summary["vs_hidden_pattern"] = evaluate.agreement(clusters, pattern)
        summary["same_pattern_5nn"] = {"fused": round(evaluate.knn_agreement(D, pattern, 5), 3),
                                       "chance": round(evaluate.chance_agreement(pattern), 3)}
        for name, Dv in views.items():  # each view on its own
            Wv = fuse.knn_sparsify(fuse.affinity(Dv, args.k), args.k)
            summary.setdefault("view_ARI", {})[name] = round(float(
                evaluate.agreement(fuse.leiden(Wv, args.resolution, 0), pattern)["ARI"]), 3)
            summary["same_pattern_5nn"][name] = round(evaluate.knn_agreement(Dv, pattern, 5), 3)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
