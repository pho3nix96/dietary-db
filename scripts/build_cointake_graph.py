#!/usr/bin/env python3
"""Co-intake graph of a food-record dataset, laid out on the composition treelet trees.

    python scripts/build_cointake_graph.py [--records data/simulated/food_records]
        [--trees results/within_gift/trees.json] [--out results/cointake/graph.json]

Nodes are the foods reported in the records (database foods plus unique products).
Their hierarchy is: all foods -> GIFT group -> that group's composition treelet tree
(pruned to the foods actually eaten) -> food. A unique product is hung next to the
database food that placement found nearest to it.

Edges are co-intake: two foods eaten in the same meal (person + day + eating occasion).
Edge weight = sum over meals of min(grams A, grams B), divided by person-days, i.e.
grams per person-day eaten together (the same with kcal in `kcal`).
Node weight = grams (and kcal) per person-day.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import to_tree

from foodspace.records import day_keys, meal_keys


def pruned_tree(linkage, food_ids, keep):
    """Nested dict of the treelet tree with only leaves in `keep`; one-child nodes collapsed.

    Treelet merges are not monotonic (a later merge can have a higher similarity than an
    earlier one), so each node's height `h` is raised to at least its children's heights:
    drawn trees then never have a parent below its child."""
    root = to_tree(np.asarray(linkage, float))

    def walk(n):
        if n.is_leaf():
            fid = int(food_ids[n.id])
            return {"id": f"f{fid}"} if fid in keep else None
        kids = [k for k in (walk(n.left), walk(n.right)) if k]
        if not kids:
            return None
        if len(kids) == 1:
            return kids[0]
        h = max([float(n.dist)] + [k.get("h", 0.0) for k in kids])
        return {"h": round(h, 3), "children": kids}

    return walk(root)


def find_leaf(tree, leaf_id):
    """Return (parent, index) of the leaf with this id."""
    stack = [tree]
    while stack:
        n = stack.pop()
        for i, c in enumerate(n.get("children", [])):
            if c.get("id") == leaf_id:
                return n, i
            stack.append(c)
    return None, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default="data/simulated/food_records")
    ap.add_argument("--trees", default="results/within_gift/trees.json")
    ap.add_argument("--out", default="results/cointake/graph.json")
    args = ap.parse_args(argv)
    rec = Path(args.records)
    c = pd.read_csv(rec / "consumption.csv")
    trees = json.loads(Path(args.trees).read_text())
    placed = pd.read_csv(rec / "placed_unique_foods.csv").set_index("unique_food_id")

    c["node"] = np.where(c["in_database"], "f" + c["food_id"].fillna(0).astype(int).astype(str),
                         "u" + c["unique_food_id"].astype(str))
    # unique products: GIFT group from placement (not from the hidden truth)
    uq = ~c["in_database"]
    c.loc[uq, "gift_group"] = c.loc[uq, "unique_food_id"].map(placed["gift_group"])
    c.loc[uq, "gift_subgroup"] = c.loc[uq, "unique_food_id"].map(placed["gift_subgroup"])
    person_days = c.groupby(day_keys(c)).ngroups
    people = c["SUBJECT"].nunique()

    g = c.groupby("node")
    nodes = pd.DataFrame({"name": g["INGREDIENT_ENG"].first(), "group": g["gift_group"].first(),
                          "subgroup": g["gift_subgroup"].first(),
                          "g": g["FOOD_AMOUNT_CONS"].sum() / person_days,
                          "kcal": g["ENERGY_kcal"].sum() / person_days,
                          "people": g["SUBJECT"].nunique(), "times": g.size(),
                          "unique": ~g["in_database"].first()})

    # edges: same person, day and eating occasion; one row per food per meal (amounts summed)
    mk = meal_keys(c)
    m = c.groupby(mk + ["node"])[["FOOD_AMOUNT_CONS", "ENERGY_kcal"]].sum().reset_index()
    w, e, k = defaultdict(float), defaultdict(float), defaultdict(int)
    for _, meal in m.groupby(mk):
        items = sorted(zip(meal["node"], meal["FOOD_AMOUNT_CONS"], meal["ENERGY_kcal"].fillna(0)))
        for (a, ga, ea), (b, gb, eb) in itertools.combinations(items, 2):
            w[(a, b)] += min(ga, gb)
            e[(a, b)] += min(ea, eb)
            k[(a, b)] += 1
    edges = [{"s": a, "t": b, "w": round(v / person_days, 3), "kcal": round(e[(a, b)] / person_days, 3),
              "meals": k[(a, b)]} for (a, b), v in w.items()]

    # hierarchy: GIFT group -> pruned composition treelet tree -> foods
    groups = []
    for grp, sub in nodes.groupby("group", sort=False):
        db_ids = {int(i[1:]) for i in sub.index if i.startswith("f")}
        if grp in trees:
            t = pruned_tree(trees[grp]["linkage"], trees[grp]["food_ids"], db_ids)
        else:
            t = None
        if t is None:
            t = {"children": []}
        elif "children" not in t:
            t = {"children": [t]}
        placed_ids = {x.get("id") for x in _leaves(t)}
        for i in sub.index:  # database foods not in the tree (e.g. too few foods in the group)
            if i.startswith("f") and i not in placed_ids:
                t["children"].append({"id": i})
        for i in sub.index[sub["unique"].to_numpy()]:
            near = str(placed.loc[i[1:], "nearest_foods"]).split(" | ")[0].strip()
            hit = nodes.index[(nodes["name"].str.strip() == near) & ~nodes["unique"]]
            parent, j = find_leaf(t, hit[0]) if len(hit) else (None, None)
            if parent is not None:
                parent["children"][j] = {"h": 0.0, "children": [parent["children"][j], {"id": i}]}
            else:
                t["children"].append({"id": i})
        t["name"] = grp
        groups.append(t)
    order = nodes.groupby("group")["g"].sum().sort_values(ascending=False).index.tolist()
    groups.sort(key=lambda t: order.index(t["name"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    nd = nodes.reset_index().rename(columns={"node": "id"})
    nd["g"] = nd["g"].round(2)
    nd["kcal"] = nd["kcal"].round(1)
    payload = {"meta": {"people": int(people), "person_days": int(person_days),
                        "meals": int(m.groupby(mk).ngroups),
                        "foods": int((~nodes["unique"]).sum()), "unique": int(nodes["unique"].sum()),
                        "edges": len(edges)},
               "tree": {"name": "all foods", "children": groups},
               "nodes": nd.to_dict("records"), "edges": edges}
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"{len(nd)} nodes, {len(edges)} edges, {len(groups)} groups -> {out}")


def _leaves(t):
    if "children" not in t:
        return [t]
    return [x for c in t["children"] for x in _leaves(c)]


if __name__ == "__main__":
    main()
