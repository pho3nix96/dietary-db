import numpy as np
import pandas as pd

from foodspace import freetext as ft


def test_clean_label_drops_brand_and_translates_once():
    assert ft.clean_label("(Kellogg's) Frosted Flakes, Cereal").startswith("corn flakes breakfast cereal")
    assert "crisps" in ft.clean_label("Lay's Wavy Potato Chips")
    # a replacement is not translated again ("cereal" inside "breakfast cereal")
    assert ft.clean_label("Cereal") == "breakfast cereal"


def test_head_label_is_the_last_noun_before_with():
    assert ft.head_label("Salad with Fried Chicken") == "salad"
    assert ft.head_label("Lemon Loaf") == "cake"
    assert ft.head_label("Peanut Butter") == "peanut butter"
    assert ft.head_label("2 cups") == ""


def test_profile_is_portion_free_and_skips_tiny_energy():
    p1 = ft.profile([10], [5], [20], [5], [2])
    p2 = ft.profile([20], [10], [40], [10], [4])
    assert np.allclose(p1, p2) and np.isclose(p1.sum(), 1)
    assert np.isnan(ft.profile([0.1], [0], [0.5], [0], [0])).all()


def _foods():
    return pd.DataFrame({
        "food_id": [1, 2, 3, 4],
        "name": ["Milk whole 3% fat", "Semi-skimmed milk 1.5% fat", "Apple pie", "Apple w/ skin"],
        "gift_group": ["Milk", "Milk", "Sweets", "Fruits"], "gift_subgroup": ["m", "m", "s", "f"],
        "PROT_g": [3.4, 3.5, 3.0, 0.3], "FAT_g": [3.0, 1.5, 12.0, 0.2], "CHO_g": [4.7, 4.8, 35.0, 11.0],
        "SUGAR_g": [4.7, 4.8, 18.0, 10.0], "FIBT_g": [0, 0, 1.5, 2.0]})


def test_composition_picks_the_milk_that_fits_the_report():
    # 8 fl oz of 2% milk: 8 g protein, 5 g fat, 12 g carbohydrate (all sugar)
    items = pd.DataFrame({"label": ["Milk"], "protein": [8.0], "total_fat": [5.0], "total_carb": [12.0],
                          "sugar": [12.0], "dietary_fiber": [0.0]})
    m = ft.match_items(items, _foods(), head_weight=0, form_penalty=0, comp_weight=2.0, min_name_sim=0.1)
    assert m.loc[0, "hybrid_food_id"] == 2


def test_form_words_keep_a_basic_food_away_from_its_dish():
    items = pd.DataFrame({"label": ["Apple"], "protein": [0.5], "total_fat": [0.3], "total_carb": [25.0],
                          "sugar": [19.0], "dietary_fiber": [4.4]})
    m = ft.match_items(items, _foods(), min_name_sim=0.1)
    assert m.loc[0, "name_food_id"] == 4 and m.loc[0, "hybrid_food_id"] == 4
    assert m.loc[0, "hybrid_status"] == "matched"


def test_non_food_and_empty_labels_are_flagged():
    items = pd.DataFrame({"label": ["Ibuprofen", None], "protein": [0, 0], "total_fat": [0, 0], "total_carb": [0, 0],
                          "sugar": [0, 0], "dietary_fiber": [0, 0]})
    m = ft.match_items(items, _foods())
    assert m["hybrid_status"].tolist() == ["non_food", "no_label"]
