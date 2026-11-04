import numpy as np
import pandas as pd

from foodspace import commodities as cm

TABLE = pd.DataFrame([
    ("wheat", "osborne", "albumin_globulin", 0.2), ("wheat", "osborne", "prolamin", 0.6), ("wheat", "osborne", "glutelin", 0.2),
    ("rye", "osborne", "albumin_globulin", 0.5), ("rye", "osborne", "prolamin", 0.2), ("rye", "osborne", "glutelin", 0.3),
    ("oats", "osborne", "albumin_globulin", np.nan), ("oats", "osborne", "prolamin", np.nan), ("oats", "osborne", "glutelin", np.nan),
    ("faba_bean", "osborne", "albumin", 0.2),
], columns=["category", "scheme", "fraction", "share"])


def test_grain_shares_keeps_complete_grains_only():
    S = cm.grain_shares(TABLE)
    assert list(S.index) == ["rye", "wheat"] and list(S.columns) == cm.OSBORNE


def test_load_mixes_and_grain_mass_share(tmp_path):
    p = tmp_path / "rav.csv"
    p.write_text("food_id,foodex2_code,commodity,share_pct,factor,raw_equivalent_g,preparation\n"
                 "1,A001K,Rye grain,90,0.4,36,Cooked\n1,A03DL,Tap water,6,1,6,Uncooked\n1,A042P,Salt,4,1,4,Uncooked\n")
    assert cm.load_mixes(p) == {1: {"A001K": 36.0, "A03DL": 6.0, "A042P": 4.0}}
    share = cm.load_mixes(p, "share_pct")[1]
    assert abs(cm.grain_mass_share(share) - 90 / 94) < 1e-9   # water left out


def test_product_profile_is_protein_weighted_mix():
    df = pd.DataFrame({"food_id": [1, 2, 3, 4], "PROT_g": [10.0, 9.0, 10.0, 20.0]})
    mixes = {1: {"A001M": 100.0},                       # pure wheat (single-commodity: sets wheat protein = 10)
             2: {"A001K": 100.0},                       # pure rye (sets rye protein = 9)
             3: {"A001M": 50.0, "A001K": 50.0},         # blend
             4: {"A000G": 100.0}}                       # oats: no curated profile
    prot = cm.commodity_protein(df, mixes, ["A001M", "A001K", "A000G"])
    assert prot["A001M"] == 10 and prot["A001K"] == 9
    S = cm.grain_shares(TABLE)
    P = cm.product_profiles(df, mixes, prot, S).set_index("food_id")
    w = 5.0 / (5.0 + 4.5)                                # wheat's share of the blend's grain protein
    assert abs(P.loc[3, "prolamin"] - (w * 0.6 + (1 - w) * 0.2)) < 1e-9
    assert abs(P.loc[3, "coverage"] - 0.95) < 1e-9
    assert P.loc[3, "dominant_grain"] == "mixed" and P.loc[1, "dominant_grain"] == "wheat"
    assert P.loc[4, "profiled_share"] == 0 and np.isnan(P.loc[4, "prolamin"])


def test_recipe_protein_profile_mixes_categories_by_protein():
    rav = pd.DataFrame({"food_id": [1, 2, 3, 3], "foodex2_code": ["A001M", "A02LY", "A001M", "A02LY"],
                        "commodity": ["Wheat grains", "Cow milk, whole", "Wheat grains", "Cow milk, whole"],
                        "share_pct": [100, 100, 50, 50], "raw_equivalent_g": [100, 100, 50, 50]})
    prot = pd.Series({1: 10.0, 2: 3.4, 3: 6.7})
    cur = pd.DataFrame([("wheat", "prolamin", "alcohol_soluble", 0.6), ("wheat", "glutelin", "insoluble_storage", 0.2),
                        ("wheat", "albumin_globulin", "mixed", 0.2), ("milk_bovine", "casein", "casein", 0.8),
                        ("milk_bovine", "whey", "water_soluble", 0.2)],
                       columns=["category", "fraction", "common_class", "share"])
    P = cm.recipe_protein_profiles(rav, prot, cur).set_index("food_id")
    w = 5.0 / (5.0 + 1.7)                       # wheat share of the blend's protein
    assert abs(P.loc[3, "protfrac_alcohol_soluble"] - 0.6 * w) < 1e-3
    assert abs(P.loc[3, "protfrac_casein"] - 0.8 * (1 - w)) < 1e-3
    assert cm.commodity_category("A0XXX", "Rapeseed oil")[0] is None
    assert cm.commodity_category("A0XXX", "Ripened cheese")[0] == "cheese_rennet"
