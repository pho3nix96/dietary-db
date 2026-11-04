#!/usr/bin/env python3
"""One co-intake graph per person, weighted in servings.

    python scripts/person_graphs.py [--records data/simulated/food_records]
        [--out results/person_graphs] [--graphml]

Nodes: the foods a person ate from the nine recommendation groups (grains, legumes,
nuts, seeds, milk products, meat and eggs, seafood, fruits, vegetables). Node weight =
servings per recorded day.

Edges: two of those foods eaten in the same meal (same day and eating occasion). Edge
weight = min(servings of A, servings of B) in that meal, summed over the person's
meals and divided by their recorded days, i.e. servings eaten together per day.

Servings (as defined in recommendation_groups.py): 7 g protein for milk products,
nuts, seeds and legumes; 14 g protein for meat and eggs and for seafood; 30 g starch
(carbohydrate - sugars) for grains; 80 g of food for fruits and vegetables. They are
computed from each record row's own nutrient amounts, so unique products with label
values are counted the same way. Name-only unique products take the values of the
table food placement found nearest.

Every person uses the same node keys (f<food_id> for table foods, u<unique id> for
unique products), so graphs can be joined: summing edge weights over people gives a
population graph, and mapping food ids to treelet branches
(results/recommendation_groups/groups.json) gives graphs at any tree level.

Writes person_nodes.csv, person_edges.csv, person_summary.csv
(and one GraphML file per person with --graphml).
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

MEAL = ["SUBJECT", "SURVEY_DAY", "MEAL_NAME"]


def load_rg():
    spec = importlib.util.spec_from_file_location("rg", Path(__file__).with_name("recommendation_groups.py"))
    rg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rg)
    return rg


def servings(rows: pd.DataFrame, rg) -> pd.Series:
    sv = pd.Series(np.nan, index=rows.index)
    for grp, (basis, size) in rg.SERVING.items():
        m = rows["rec_group"] == grp
        if basis is None:
            amount = rows.loc[m, "FOOD_AMOUNT_CONS"]
        elif basis == "PROT_g":
            amount = rows.loc[m, "PROTEIN_g"]
        else:  # starch = available carbohydrate - sugars
            amount = (rows.loc[m, "CARBOH_g"] - rows.loc[m, "SUGAR_g"]).clip(lower=0)
        sv[m] = amount / size
    return sv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default="data/simulated/food_records")
    ap.add_argument("--foods", default="data/processed/foods_lv.csv")
    ap.add_argument("--out", default="results/person_graphs")
    ap.add_argument("--graphml", action="store_true")
    args = ap.parse_args(argv)
    rg = load_rg()
    rec = Path(args.records)
    c = pd.read_csv(rec / "consumption.csv")
    foods = pd.read_csv(args.foods)

    # unique products: GIFT group from placement; missing values from the nearest table food
    uq = ~c["in_database"]
    placed = pd.read_csv(rec / "placed_unique_foods.csv").set_index("unique_food_id")
    c.loc[uq, "gift_group"] = c.loc[uq, "unique_food_id"].map(placed["gift_group"])
    c.loc[uq, "gift_subgroup"] = c.loc[uq, "unique_food_id"].map(placed["gift_subgroup"])
    c["values_from"] = np.where(uq, "label", "table")
    byname = foods.assign(key=foods["name"].str.strip()).drop_duplicates("key").set_index("key")
    nv = uq & c["PROTEIN_g"].isna()
    for i in c.index[nv]:
        near = str(placed.loc[c.at[i, "unique_food_id"], "nearest_foods"]).split(" | ")[0].strip()
        if near in byname.index:
            f, g = byname.loc[near], c.at[i, "FOOD_AMOUNT_CONS"] / 100
            c.loc[i, ["PROTEIN_g", "CARBOH_g", "SUGAR_g"]] = [f["PROT_g"] * g, f["CHO_g"] * g, f["SUGAR_g"] * g]
            c.at[i, "values_from"] = "nearest food"

    c["node"] = np.where(c["in_database"], "f" + c["food_id"].fillna(0).astype(int).astype(str),
                         "u" + c["unique_food_id"].astype(str))
    c["rec_group"] = c.rename(columns={"INGREDIENT_ENG": "name"}).apply(rg.rec_group, axis=1)
    c = c[c["rec_group"].notna()].copy()
    c["servings"] = servings(c, rg)
    days = pd.read_csv(rec / "consumption.csv").groupby("SUBJECT")["SURVEY_DAY"].nunique()

    # nodes
    g = c.groupby(["SUBJECT", "node"])
    nodes = g.agg(name=("INGREDIENT_ENG", "first"), rec_group=("rec_group", "first"),
                  food_id=("food_id", "first"), unique_food_id=("unique_food_id", "first"),
                  servings=("servings", "sum"), grams=("FOOD_AMOUNT_CONS", "sum"),
                  meals=("MEAL_NAME", "size"), days_eaten=("SURVEY_DAY", "nunique"),
                  values_from=("values_from", "first")).reset_index()
    nodes["days_recorded"] = nodes["SUBJECT"].map(days)
    nodes["servings_per_day"] = (nodes["servings"] / nodes["days_recorded"]).round(4)
    nodes["grams_per_day"] = (nodes["grams"] / nodes["days_recorded"]).round(2)
    nodes = nodes.drop(columns=["servings", "grams"])

    # edges: same person, day and eating occasion
    meal = c.groupby(MEAL + ["node"], as_index=False)["servings"].sum()
    rows = []
    for (s, d, m), grp in meal.groupby(MEAL):
        items = sorted(zip(grp["node"], grp["servings"]))
        for (a, sa), (b, sb) in itertools.combinations(items, 2):
            rows.append((s, a, b, min(sa, sb)))
    e = pd.DataFrame(rows, columns=["SUBJECT", "source", "target", "w"])
    edges = e.groupby(["SUBJECT", "source", "target"]).agg(servings_together=("w", "sum"), meals_together=("w", "size")).reset_index()
    edges["servings_together_per_day"] = (edges["servings_together"] / edges["SUBJECT"].map(days)).round(4)
    edges = edges.drop(columns="servings_together")
    grp_of = nodes.drop_duplicates("node").set_index("node")["rec_group"]
    edges["source_group"], edges["target_group"] = edges["source"].map(grp_of), edges["target"].map(grp_of)

    # per-person summary: servings per day by group, graph size
    summ = nodes.pivot_table(index="SUBJECT", columns="rec_group", values="servings_per_day", aggfunc="sum", fill_value=0)
    summ = summ.reindex(columns=[x for x in rg.ORDER if x in summ.columns]).round(2)
    summ.columns = [f"servings_{x.lower().replace(' ', '_')}" for x in summ.columns]
    summ.insert(0, "foods", nodes.groupby("SUBJECT").size())
    summ.insert(1, "edges", edges.groupby("SUBJECT").size().reindex(summ.index, fill_value=0))
    summ.insert(2, "days_recorded", days.reindex(summ.index))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(out / "person_nodes.csv", index=False)
    edges.to_csv(out / "person_edges.csv", index=False)
    summ.reset_index().to_csv(out / "person_summary.csv", index=False)
    if args.graphml:
        import networkx as nx
        gdir = out / "graphml"
        gdir.mkdir(exist_ok=True)
        for s, nd in nodes.groupby("SUBJECT"):
            G = nx.Graph(subject=int(s))
            for r in nd.itertuples():
                G.add_node(r.node, name=r.name, rec_group=r.rec_group, servings_per_day=float(r.servings_per_day),
                           grams_per_day=float(r.grams_per_day))
            for r in edges[edges["SUBJECT"] == s].itertuples():
                G.add_edge(r.source, r.target, weight=float(r.servings_together_per_day), meals=int(r.meals_together))
            nx.write_graphml(G, gdir / f"person_{int(s):03d}.graphml")
    print(f"{nodes['SUBJECT'].nunique()} people, {len(nodes)} person-food nodes, {len(edges)} person edges -> {out}/")
    print(summ.describe().loc[["mean", "50%", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
