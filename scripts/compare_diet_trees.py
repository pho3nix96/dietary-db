#!/usr/bin/env python3
"""Which food tree makes the person distance recover dietary patterns best?

    python scripts/compare_diet_trees.py [--sets 5] [--out results/diet_similarity/tree_comparison.csv]

Runs on the committed simulated records plus `--sets` new record sets simulated with seeds
1..sets (without unique products; written to a temporary folder and deleted afterwards).
For each record set and food tree, people are compared with the tree distance alone and with
the nutrient view added (weight 0.25), and scored against the hidden dietary pattern:

    same_pattern_5nn, _10nn   share of a person's 5 (10) most similar people with the same pattern
    chance                    the same share for neighbours drawn at random
    ari                       agreement of Leiden clusters of people (k = 10) with the patterns

Food trees (src/foodspace/diet.py):
    flat                  every food on its own: no similarity between different foods
    gift_groups           GIFT groups only: foods of one group count as the same
    gift_subgroups        GIFT group -> subgroup -> food (the default of compare_diets.py)
    subgroup_composition  GIFT group -> subgroup -> composition tree of the subgroup's foods
    composition           GIFT group -> composition tree (subgroups not used)

Only simulated records carry patterns to score against, and the simulated patterns are
defined by which kinds of foods people choose; real records need an outside criterion.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
from pathlib import Path

import pandas as pd

from foodspace import diet, evaluate, fuse

TREES = ["flat", "gift_groups", "gift_subgroups", "subgroup_composition", "composition"]


def load_simulator():
    spec = importlib.util.spec_from_file_location("sim", Path(__file__).with_name("simulate_food_records.py"))
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    return sim


def food_trees(foods, matched, trees, placements):
    return {
        "flat": diet.flat_tree(foods, matched),
        "gift_groups": diet.food_tree(foods, matched, (1.0, 1e-9, 1e-9)),
        "gift_subgroups": diet.food_tree(foods, matched),
        "subgroup_composition": diet.composition_tree(foods, matched, trees, placements, within="subgroup"),
        "composition": diet.composition_tree(foods, matched, trees, placements, within="group"),
    }


def score(name, records_dir, foods, trees, nutrient_weight, k):
    rec = Path(records_dir)
    consumption = pd.read_csv(rec / "consumption.csv")
    plc_path = rec / "placed_unique_foods.csv"
    placements = pd.read_csv(plc_path) if plc_path.exists() else None
    matched = diet.fill_nutrients(diet.match_records(consumption, foods, placements), foods, placements)
    truth = pd.read_csv(rec / "truth.csv").set_index("SUBJECT")["pattern"].astype(str)
    rows, Dn = [], None
    for tree, (leaf_index, parent, length, _) in food_trees(foods, matched, trees, placements).items():
        P = diet.diet_profiles(matched, leaf_index)
        people = P.index.to_numpy()
        y = truth.reindex(people).to_numpy()
        if Dn is None:
            Dn = diet.nutrient_distance(matched, people=people)
        Dt = diet.tree_distance(P, leaf_index, parent, length)
        for view, D in (("tree", Dt), ("tree+nutrients", (Dt + nutrient_weight * Dn) / (1 + nutrient_weight))):
            W = fuse.knn_sparsify(fuse.affinity(D, k), k)
            rows.append({"records": name, "tree": tree, "view": view,
                         "same_pattern_5nn": evaluate.knn_agreement(D, y, 5),
                         "same_pattern_10nn": evaluate.knn_agreement(D, y, 10),
                         "chance": evaluate.chance_agreement(y),
                         "ari": evaluate.agreement(fuse.leiden(W, 1.0, 0), y)["ARI"]})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default="data/simulated/food_records", help="committed simulated records")
    ap.add_argument("--foods", default="data/processed/foods_lv.csv")
    ap.add_argument("--trees", default="results/within_gift/trees.json")
    ap.add_argument("--sets", type=int, default=5, help="extra simulated record sets (seeds 1..sets)")
    ap.add_argument("--nutrient-weight", type=float, default=0.25)
    ap.add_argument("--k", type=int, default=10, help="neighbours per person for the Leiden graph")
    ap.add_argument("--out", default="results/diet_similarity/tree_comparison.csv")
    args = ap.parse_args(argv)

    foods = pd.read_csv(args.foods, low_memory=False)
    foods[["gift_group", "gift_subgroup"]] = foods[["gift_group", "gift_subgroup"]].fillna("unassigned")
    trees = json.loads(Path(args.trees).read_text())
    rows = score("committed (seed 42)", args.records, foods, trees, args.nutrient_weight, args.k)
    sim = load_simulator()
    with tempfile.TemporaryDirectory() as tmp:
        for seed in range(1, args.sets + 1):
            out = Path(tmp) / f"seed{seed}"
            with contextlib.redirect_stdout(io.StringIO()):
                sim.main(["--foods", args.foods, "--seed", str(seed), "--unique-share", "0", "--out", str(out)])
            rows += score(f"seed {seed}", out, foods, trees, args.nutrient_weight, args.k)
            print(f"record set {seed}/{args.sets} done")

    r = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    r.round(4).to_csv(out, index=False)
    metrics = ["same_pattern_5nn", "same_pattern_10nn", "ari"]
    summary = r.groupby(["view", "tree"], sort=False)[metrics].agg(["mean", "std"]).round(3)
    print(f"\n{r['records'].nunique()} record sets; chance level of same_pattern_5nn: {r['chance'].mean():.3f}")
    print(summary.to_string())
    base = r[r["tree"] == "gift_subgroups"].set_index(["records", "view"])["same_pattern_5nn"]
    print("\nsame_pattern_5nn minus gift_subgroups, per record set (min / mean / max):")
    for tree in TREES:
        if tree == "gift_subgroups":
            continue
        d = r[r["tree"] == tree].set_index(["records", "view"])["same_pattern_5nn"] - base
        for view, dv in d.groupby(level="view"):
            print(f"  {tree:21s} {view:15s} {dv.min():+.3f} / {dv.mean():+.3f} / {dv.max():+.3f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
