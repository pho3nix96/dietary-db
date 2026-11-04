"""Tests for the fraction-profile view.

Uses a small synthetic set of dairy and meat foods (values are illustrative,
not from a composition table) to check the property the view exists for:
cheese should sit closer to milk than to salami once nutrient fractions are
included, even though on totals cheese looks like a fatty, high-protein food.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from foodspace import fractions, run, views

CURATED = Path(__file__).resolve().parents[1] / "data" / "curated" / "protein_fractions.csv"


def dairy_meat():
    milk_fat = (0.65, 0.28, 0.04)   # SFA, MUFA, PUFA shares of fat (illustrative)
    meat_fat = (0.40, 0.48, 0.10)
    rows = [
        # name, group, protein, fat, carb, fibre, sugars, lactose, sucrose, fat profile, curated category
        ("Milk 3.5%", "Milk and dairy", 3.3, 3.5, 4.8, 0, 4.8, 4.8, 0, milk_fat, "milk_bovine"),
        ("Milk 1.5%", "Milk and dairy", 3.4, 1.5, 4.9, 0, 4.9, 4.9, 0, milk_fat, "milk_bovine"),
        ("Gouda", "Milk and dairy", 25, 29, 0.1, 0, 0.1, 0.1, 0, milk_fat, "cheese_rennet"),
        ("Edam", "Milk and dairy", 26, 26, 0.1, 0, 0.1, 0.1, 0, milk_fat, "cheese_rennet"),
        ("Mozzarella", "Milk and dairy", 22, 20, 0.5, 0, 0.5, 0.5, 0, milk_fat, "cheese_rennet"),
        ("Whey protein", "Milk and dairy", 80, 5, 8, 0, 8, 8, 0, milk_fat, "whey_protein"),
        ("Salami", "Meat", 22, 34, 1.2, 0, 1.0, 0, 1.0, meat_fat, None),
        ("Beef steak", "Meat", 25, 19, 0, 0, 0, 0, 0, meat_fat, None),
        ("Pork chop", "Meat", 26, 13, 0, 0, 0, 0, 0, meat_fat, None),
    ]
    out = []
    for i, (name, g, p, f, c, fi, s, lac, suc, fp, cat) in enumerate(rows):
        out.append({
            "food_id": f"f{i}", "name": name, "gift_group": g, "gift_subgroup": g,
            "protein_g": p, "fat_g": f, "carbohydrate_g": c, "fibre_g": fi, "sugars_g": s,
            "lactose_g": lac, "sucrose_g": suc,
            "sfa_g": f * fp[0], "mufa_g": f * fp[1], "pufa_g": f * fp[2],
            "protein_fraction_category": cat,
        })
    return pd.DataFrame(out)


def test_block_shares_undefined_below_threshold():
    df = pd.DataFrame({"fat_g": [10, 0.2], "sfa_g": [6, 0.1], "mufa_g": [3, 0.05], "pufa_g": [1, 0.05]})
    S = fractions.block_shares(df, ["sfa_g", "mufa_g", "pufa_g"], parent="fat_g", min_parent=0.5)
    assert np.allclose(S.iloc[0].sum(), 1.0)
    assert S.iloc[1].isna().all()


def test_concentration_does_not_change_profile():
    # milk and a 10x concentrated milk have the same fat profile, so distance 0
    df = pd.DataFrame({"fat_g": [3.5, 35], "sfa_g": [2.3, 23], "mufa_g": [1.0, 10], "pufa_g": [0.15, 1.5]})
    S = fractions.block_shares(df, ["sfa_g", "mufa_g", "pufa_g"], parent="fat_g")
    assert fractions.aitchison_distance(S)[0, 1] < 1e-9


def test_clr_handles_zeros():
    S = pd.DataFrame({"a": [0.5, 1.0], "b": [0.5, 0.0]})
    C = fractions.clr(S)
    assert np.isfinite(C.to_numpy()).all()
    assert np.allclose(C.sum(axis=1), 0)


def test_curated_block_reads_sourced_rows_only():
    table = pd.read_csv(CURATED)
    df = pd.DataFrame({"protein_fraction_category": ["milk_bovine", "yogurt", None]})
    S = fractions.curated_block(df, table, "protein_fraction_category", "casein_whey")
    assert np.isclose(S.loc[0, "casein"], 0.80)
    assert S.loc[1].isna().all()   # yogurt values not sourced yet
    assert S.loc[2].isna().all()


def test_cheese_moves_from_salami_to_milk():
    df = dairy_meat()
    name = df["name"].tolist()
    i = {n: k for k, n in enumerate(name)}

    # Totals only: cheese looks more like salami than like milk
    Xp, _ = views.prepare_nutrients(df[["protein_g", "fat_g", "carbohydrate_g", "fibre_g"]], groups=df.gift_group)
    Dn = views.nutrient_distance(Xp)
    assert Dn[i["Gouda"], i["Salami"]] < Dn[i["Gouda"], i["Milk 3.5%"]]

    # Fractions: fat profile and protein type say cheese is dairy
    cfg = {"blocks": {
        "fat_type": {"parent": "fat_g", "parts": ["sfa_g", "mufa_g", "pufa_g"]},
        "sugar_type": {"parent": "sugars_g", "parts": ["lactose_g", "sucrose_g"]},
        "protein_type": {"source": "curated", "key_col": "protein_fraction_category", "scheme": "casein_whey"},
    }}
    blocks = fractions.build_blocks(df, cfg, pd.read_csv(CURATED))
    Df, report = fractions.fraction_distance(blocks)
    assert Df[i["Gouda"], i["Milk 3.5%"]] < Df[i["Gouda"], i["Salami"]]
    # cheeses are closer to each other than to milk (casein share differs)
    assert Df[i["Gouda"], i["Edam"]] < Df[i["Gouda"], i["Milk 3.5%"]]
    # whey protein is set apart from cheese by protein type
    assert Df[i["Gouda"], i["Whey protein"]] > Df[i["Gouda"], i["Edam"]]
    assert report["protein_type"]["foods_with_block"] == 6


def test_pipeline_with_fractions(tmp_path):
    df = dairy_meat()
    path = tmp_path / "foods.csv"
    df.to_csv(path, index=False)
    cfg = {
        "input": str(path), "output_dir": str(tmp_path / "out"), "seed": 0,
        "columns": {"id": "food_id", "name": "name", "description": None, "gift_group": "gift_group",
                    "gift_subgroup": "gift_subgroup", "nutrients": ["protein_g", "fat_g", "carbohydrate_g", "fibre_g"]},
        "nutrients": {"basis": "per_100g", "max_missing": 0.3, "impute": "group_median", "log1p": True, "metric": "cosine"},
        "semantics": {"enabled": False, "model": ""},
        "graph": {"k": 3},
        "fusion": {"method": "weighted", "gift_in_fusion": False, "snf_iterations": 10,
                   "weights": {"nutrients": 1.0, "fractions": 1.0}},
        "fractions": {"enabled": True, "curated_table": str(CURATED), "blocks": {
            "fat_type": {"parent": "fat_g", "parts": ["sfa_g", "mufa_g", "pufa_g"]},
            "protein_type": {"source": "curated", "key_col": "protein_fraction_category", "scheme": "casein_whey"},
            "not_in_table": {"parent": "fat_g", "parts": ["c18_3_g"]},
        }},
        "clustering": {"resolution": 1.0, "scan": [1.0]},
        "layout": {"method": "tsne"},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    run.main(str(tmp_path / "config.yaml"))
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert "fractions" in metrics["views"] and "gift" not in metrics["views"]
    assert set(metrics["fraction_blocks"]) >= {"fat_type", "protein_type"}
    assert "not_in_table" not in metrics["fraction_blocks"]


def test_common_scheme_puts_families_on_shared_axes():
    table = pd.read_csv(CURATED)
    df = pd.DataFrame({"protein_fraction_category": ["milk_bovine", "cheese_rennet", "wheat", "soybean", "meat_muscle", "lentil"]})
    S = fractions.curated_block(df, table, "protein_fraction_category", "common")
    assert np.allclose(S.dropna().sum(axis=1), 1.0)
    assert S.loc[5].isna().all()                      # lentil not sourced yet
    assert S.loc[0, "casein"] > 0.7 and S.loc[2, "alcohol_soluble"] > 0.5
    D = fractions.aitchison_distance(S)
    # milk is closer to cheese than to wheat or meat
    assert D[0, 1] < D[0, 2] and D[0, 1] < D[0, 4]


def test_protein_block_skips_low_protein_foods():
    df = pd.DataFrame({"protein_g": [3.3, 0.5, 25], "protein_fraction_category": ["milk_bovine", "milk_bovine", "cheese_rennet"]})
    cfg = {"blocks": {"protein_type": {"source": "curated", "key_col": "protein_fraction_category",
                                       "scheme": "common", "parent": "protein_g", "min_parent": 2.0}}}
    S = fractions.build_blocks(df, cfg, pd.read_csv(CURATED))["protein_type"]
    assert S.loc[1].isna().all() and S.loc[0].notna().all()
