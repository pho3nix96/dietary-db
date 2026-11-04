"""Tests for matching records to the food reference and comparing diets."""

import numpy as np
import pandas as pd
import pytest

from foodspace import diet


def reference():
    return pd.DataFrame({
        "food_id": [1, 2, 3, 4, 5],
        "name": ["Whole milk", "Semi-skimmed milk", "Oat porridge", "Rye bread", "Apple"],
        "gift_group": ["Milk", "Milk", "Cereals", "Cereals", "Fruit"],
        "gift_subgroup": ["Fresh milk", "Fresh milk", "Porridge", "Bread", "Pome fruit"],
        "foodex2_code": ["A02LY", "A02LZ", "A0F0T", "A00ZZ", "A01DJ"],
    })


def records():
    return pd.DataFrame({
        "SUBJECT": [1, 1, 2, 2, 3, 3],
        "food_id": [1, np.nan, np.nan, np.nan, np.nan, np.nan],
        "unique_food_id": [np.nan, np.nan, np.nan, np.nan, "U1", np.nan],
        "FOODEX2_INGR_CODE": ["A02LY", "A0F0T", "A01DJ", "ZZZZZ", "ZZZZZ", "ZZZZZ"],
        "INGREDIENT_ENG": ["x", "porridge", "apple", "semi skimmed milk", "homemade thing", "qqqq"],
        "ENERGY_kcal": [100.0, 200.0, 50.0, 100.0, 80.0, 10.0],
        "CARBOH_g": [5, 30, 12, 5, 10, 1], "SUGAR_g": [5, 2, 10, 5, 3, 1],
        "PROTEIN_g": [3, 6, 0.3, 3, 1, 0], "FAT_g": [4, 4, 0.2, 2, 2, 0], "FIBTG_g": [0, 3, 2, 0, 1, 0],
    })


def placed():
    return pd.DataFrame({"unique_food_id": ["U1"], "gift_group": ["Fruit"], "gift_subgroup": ["Pome fruit"]})


def test_match_types():
    m = diet.match_records(records(), reference(), placed())
    assert m["match"].tolist() == ["id", "code", "code", "name", "placed", "unmatched"]
    assert m["leaf"].tolist()[:5] == ["f1", "f3", "f5", "f2", "uU1"]
    assert pd.isna(m["leaf"].iloc[5])
    rep = diet.match_report(m, "ENERGY_kcal")
    assert rep["by_match"]["unmatched"] == 1 and rep["weight_share_by_match"]["unmatched"] < 0.05


def profiles(dish_shares=None):
    foods, m = reference(), diet.match_records(records(), reference(), placed())
    li, par, ln, _ = diet.food_tree(foods, m)
    return m, li, par, ln, diet.diet_profiles(m, li, dish_shares=dish_shares)


def test_profiles_are_shares_and_scale_free():
    m, li, _, _, P = profiles()
    assert np.allclose(P.sum(axis=1), 1.0)
    m2 = m.copy()
    m2.loc[m2["SUBJECT"] == 1, "ENERGY_kcal"] *= 0.5  # person 1 under-reports everything by half
    assert np.allclose(diet.diet_profiles(m2, li).loc[1], P.loc[1])


def test_tree_distance_is_a_metric_and_respects_food_similarity():
    foods = reference()
    recs = pd.DataFrame({"SUBJECT": [1, 2, 3, 4], "food_id": [1, 2, 3, 5], "ENERGY_kcal": [100.0] * 4})
    m = diet.match_records(recs, foods)
    li, par, ln, _ = diet.food_tree(foods, m)
    P = diet.diet_profiles(m, li)
    D = diet.tree_distance(P, li, par, ln)
    assert np.allclose(D, D.T) and np.allclose(np.diag(D), 0)
    assert D[0, 1] < D[0, 2] < D[0, 3] + 1e-9   # whole vs semi-skimmed milk < milk vs porridge
    assert D[0, 1] == pytest.approx(2 * ln[li["f1"]] / (2 * (1.0 + 0.5 + 0.25)))
    assert D.max() <= 1 + 1e-9
    assert D[0, 2] == pytest.approx(1.0)       # different groups: the largest possible distance


def test_dish_shares_spread_a_dish_over_ingredient_leaves():
    foods = reference()
    recs = pd.DataFrame({"SUBJECT": [1, 2], "food_id": [3, 1], "ENERGY_kcal": [100.0, 100.0]})
    m = diet.match_records(recs, foods)
    li, par, ln, _ = diet.food_tree(foods, m)
    shares = pd.DataFrame({"food_id": [3, 3], "leaf": ["f1", "f5"], "share": [0.5, 0.5]})  # "dish" = half milk, half apple
    P = diet.diet_profiles(m, li, dish_shares=shares)
    assert P.loc[1, "f1"] == pytest.approx(0.5) and P.loc[1, "f5"] == pytest.approx(0.5) and P.loc[1, "f3"] == 0
    D = diet.tree_distance(P, li, par, ln)
    assert D[0, 1] < 1.0  # the dish eater now overlaps with the milk drinker


def test_nutrient_distance_and_person_distance():
    m, *_ = profiles()
    Dn = diet.nutrient_distance(m, people=[1, 2, 3])
    assert Dn.shape == (3, 3) and np.allclose(np.diag(Dn), 0) and Dn.max() <= 1 + 1e-9
    D = diet.person_distance({"a": Dn, "b": np.zeros((3, 3))}, {"a": 1.0, "b": 3.0})
    assert np.allclose(D, Dn / 4)
    with pytest.raises(ValueError):
        diet.person_distance({"a": Dn}, {"a": 0})
