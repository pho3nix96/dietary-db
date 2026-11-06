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


def faceted_reference():
    return pd.DataFrame({
        "food_id": [10, 11, 12, 13],
        "name": ["Rye bread", "Wheat bread white", "Wheat bread wholegrain", "Apple raw"],
        "gift_group": ["Cereals", "Cereals", "Cereals", "Fruit"],
        "gift_subgroup": ["Bread", "Bread", "Bread", "Pome fruit"],
        # 11 and 12 share the base term A004V; 12 is coded with facets, 11 with the bare term
        "foodex2_code": ["A004Y#F04.A001K", "A004V", "A004V#F01.A0F2S", "A01DJ#F28.A07JS"],
    })


def test_match_uses_base_terms_and_names_among_shared_codes():
    recs = pd.DataFrame({
        "SUBJECT": [1, 1, 1, 1, 1, 1],
        "food_id": [99, 99, 99, 99, 99, 99],  # ids from another database: must be ignored
        "FOODEX2_INGR_CODE": ["A004Y", "A004V", "A004V", "A004V#F01.A0F2S", "A01DJ", "XXXXX"],
        "INGREDIENT_ENG": ["rye bread", "wholegrain wheat bread", None, "bread", "apple", "zzzz"],
        "ENERGY_kcal": [100.0] * 6,
    })
    m = diet.match_records(recs, faceted_reference(), id_col=None)
    assert m["match"].tolist() == ["code", "code+name", "code_ambiguous", "code", "code", "unmatched"]
    assert m["leaf"].tolist()[:5] == ["f10", "f12", "f11", "f12", "f13"]  # no name: the bare-coded food


def test_id_mismatch_is_reported():
    foods = faceted_reference()
    recs = pd.DataFrame({"SUBJECT": [1, 2], "food_id": [10, 13], "INGREDIENT_ENG": ["Milk", "Cheese"],
                         "ENERGY_kcal": [1.0, 1.0]})
    rep = diet.match_report(diet.match_records(recs, foods), "ENERGY_kcal", foods)
    assert rep["id_matches_with_same_name"] == 0.0


def test_fill_nutrients_from_reference_and_placement():
    foods = reference().assign(ENERC_kcal=[64, 46, 70, 250, 52], PROT_g=[3.4, 3.5, 2.5, 8.5, 0.3],
                               FAT_g=[3.0, 1.5, 1.4, 1.2, 0.2], CHO_g=[4.7, 4.8, 12.0, 45.0, 12.0],
                               SUGAR_g=[4.7, 4.8, 0.5, 3.0, 10.0], FIBT_g=[0, 0, 1.5, 6.0, 2.0])
    recs = pd.DataFrame({"SUBJECT": [1, 1, 2], "food_id": [1, np.nan, np.nan], "unique_food_id": [np.nan, "U1", "U9"],
                         "INGREDIENT_ENG": ["x", "homemade apple thing", "qqqq"], "FOOD_AMOUNT_CONS": [200.0, 50.0, 10.0]})
    plc = placed().assign(nearest_foods=["Apple | Rye bread"])
    m = diet.fill_nutrients(diet.match_records(recs, foods, plc), foods, plc)
    assert m["nutrients_from"].tolist() == ["reference", "reference", "missing"]
    assert m.loc[0, "ENERGY_kcal"] == pytest.approx(128.0) and m.loc[1, "PROTEIN_g"] == pytest.approx(0.15)


def test_nutrient_distance_self_is_zero_without_data():
    m, *_ = profiles()
    m.loc[m["SUBJECT"] == 3, ["PROTEIN_g", "FAT_g", "CARBOH_g", "SUGAR_g", "FIBTG_g"]] = np.nan
    D = diet.nutrient_distance(m, people=[1, 2, 3])
    assert np.allclose(np.diag(D), 0) and D[0, 2] == 1.0


def test_unknown_dish_share_leaves_are_reported(capsys):
    foods = reference()
    recs = pd.DataFrame({"SUBJECT": [1], "food_id": [3], "ENERGY_kcal": [100.0]})
    m = diet.match_records(recs, foods)
    li, *_ = diet.food_tree(foods, m)
    shares = pd.DataFrame({"food_id": [3, 3], "leaf": ["f1", "f999"], "share": [0.5, 0.5]})
    P = diet.diet_profiles(m, li, dish_shares=shares)
    assert "f999" in capsys.readouterr().out and P.loc[1, "f1"] == pytest.approx(1.0)
