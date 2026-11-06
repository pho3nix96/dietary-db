"""Tests for small shared helpers: neighbours, record keys, curated statuses, commodity rules, tree heights."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from foodspace import evaluate, fractions, records
from foodspace.commodities import commodity_category

ROOT = Path(__file__).parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_nearest_never_returns_the_item_itself():
    # items 0 and 1 are identical (distance 0): a plain argsort may put 1 before 0 in row 0
    D = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.5, 0.5, 0.0]])
    idx = evaluate.nearest(D, 2)
    assert all(i not in row for i, row in enumerate(idx))
    assert evaluate.knn_agreement(D, ["a", "a", "b"], 1) == pytest.approx(2 / 3)
    assert evaluate.chance_agreement(["a", "a", "b", "b"]) == pytest.approx(1 / 3)


def test_record_keys_use_round_when_present():
    r = pd.DataFrame({"SUBJECT": [1, 1, 1], "ROUND": [1, 2, 2], "SURVEY_DAY": [1, 1, 1], "MEAL_NAME": ["L", "L", "D"]})
    assert records.day_keys(r) == ["SUBJECT", "ROUND", "SURVEY_DAY"]
    assert records.days_per_subject(r).loc[1] == 2  # day 1 of round 1 and day 1 of round 2
    assert records.meal_keys(r.drop(columns="ROUND")) == ["SUBJECT", "SURVEY_DAY", "MEAL_NAME"]


def test_curated_statuses_filter_whole_categories():
    t = pd.DataFrame({"category": ["a", "a", "b", "b"], "scheme": ["s"] * 4, "fraction": ["x", "y", "x", "y"],
                      "common_class": ["x", "y", "x", "y"], "share": [0.5, 0.5, 0.3, 0.7],
                      "status": ["sourced", "sourced", "sourced", "provisional"]})
    df = pd.DataFrame({"cat": ["a", "b"]})
    allowed = fractions.curated_block(df, t, "cat", "common", ["sourced"])
    assert allowed.loc[0].notna().all() and allowed.loc[1].isna().all()
    assert fractions.category_status(t).to_dict() == {"a": "sourced", "b": "provisional"}


@pytest.mark.parametrize("name, category", [
    ("Buttermilk", "milk_bovine"), ("Milk, low fat", "milk_bovine"), ("Fat-free milk", "milk_bovine"),
    ("Herring, salted", "fish"), ("Salmon, fatty", "fish"), ("Egg, boiled", "egg"), ("Peanut butter", "peanut"),
    ("Butter", None), ("Rapeseed oil", None), ("Coconut oil", None), ("Milk fat", None), ("Eggplant", None),
])
def test_commodity_rules_match_whole_words(name, category):
    assert commodity_category("X", name)[0] == category


def test_cointake_tree_heights_never_drop_below_children():
    cg = load_script("build_cointake_graph")
    # merge 3 (the root) is lower than merge 2 below it: a treelet inversion
    Z = np.array([[0, 1, 0.1, 2], [2, 3, 0.6, 2], [4, 5, 0.3, 4]], float)
    t = cg.pruned_tree(Z, [10, 11, 12, 13], {10, 11, 12, 13})

    def check(n):
        for c in n.get("children", []):
            assert c.get("h", 0.0) <= n["h"]
            check(c)
    check(t)
    assert t["h"] == pytest.approx(0.6)


def test_unique_products_values_per_100g_from_records():
    pu = load_script("place_unique_foods")
    c = pd.DataFrame({"in_database": [False, False, True], "unique_food_id": ["U1", "U1", None],
                      "INGREDIENT_ENG": ["Homemade bread", "Homemade bread", "Milk"],
                      "FOOD_AMOUNT_CONS": [50.0, 150.0, 200.0], "PROTEIN_g": [5.0, np.nan, 6.8]})
    u = pu.unique_products_from_records(c)
    assert len(u) == 1 and u.loc[0, "PROT_g_per100"] == pytest.approx(10.0)  # 5 g in the 50 g line that has a value
