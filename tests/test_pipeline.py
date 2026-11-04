"""Runs the pipeline on synthetic foods with known structure.

Three GIFT groups with distinct nutrient profiles, plus one planted food whose
nutrients match another group. The pipeline should recover the groups and
flag the planted food as a misfit.
"""

import json

import numpy as np
import pandas as pd
import yaml

from foodspace import run


def make_foods(n_per=20, seed=1):
    rng = np.random.default_rng(seed)
    profiles = {  # protein, fat, carbohydrate, fibre (synthetic, not real foods)
        "Meat": [20, 10, 0.5, 0.1],
        "Cereals": [10, 2, 70, 8],
        "Fruits": [0.8, 0.3, 12, 2.5],
    }
    rows = []
    for g, p in profiles.items():
        for i in range(n_per):
            vals = np.clip(rng.normal(p, np.array(p) * 0.15 + 0.05), 0, None)
            rows.append({"food_id": f"{g}_{i}", "name": f"{g} item {i}", "gift_group": g,
                         "gift_subgroup": f"{g}_{i % 2}", **dict(zip(["protein_g", "fat_g", "carbohydrate_g", "fibre_g"], vals))})
    # Planted misfit: labelled Cereals, nutrients like Meat
    rows.append({"food_id": "planted", "name": "planted item", "gift_group": "Cereals", "gift_subgroup": "Cereals_0",
                 "protein_g": 21, "fat_g": 9, "carbohydrate_g": 0.6, "fibre_g": 0.1})
    return pd.DataFrame(rows)


def test_pipeline(tmp_path):
    foods = make_foods()
    foods.loc[3, "fat_g"] = np.nan  # one missing value to exercise imputation
    path = tmp_path / "foods.csv"
    foods.to_csv(path, index=False)
    cfg = {
        "input": str(path), "output_dir": str(tmp_path / "out"), "seed": 0,
        "columns": {"id": "food_id", "name": "name", "description": None, "gift_group": "gift_group",
                    "gift_subgroup": "gift_subgroup", "nutrients": ["protein_g", "fat_g", "carbohydrate_g", "fibre_g"]},
        "nutrients": {"basis": "per_100g", "max_missing": 0.3, "impute": "group_median", "log1p": True, "metric": "cosine"},
        "semantics": {"enabled": False, "model": ""},
        "graph": {"k": 8},
        "fusion": {"method": "weighted", "snf_iterations": 20, "weights": {"gift": 0.0, "nutrients": 1.0}},
        "clustering": {"resolution": 0.2, "scan": [0.2, 1.0]},
        "layout": {"method": "tsne"},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    run.main(str(cfg_path))

    out = tmp_path / "out"
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["vs_gift_group"]["ARI"] > 0.8
    assert metrics["nutrient_preparation"]["values_imputed"] == 1
    mis = pd.read_csv(out / "misfits.csv")
    assert "planted" in mis["id"].tolist()
    fm = pd.read_csv(out / "food_map.csv")
    assert len(fm) == len(foods) and fm[["x", "y"]].notna().all().all()


def test_snf_runs(tmp_path):
    from foodspace import fuse, views

    foods = make_foods(10)
    Xp, _ = views.prepare_nutrients(foods[["protein_g", "fat_g", "carbohydrate_g", "fibre_g"]])
    Ds = [views.gift_distance(foods.gift_group, foods.gift_subgroup), views.nutrient_distance(Xp)]
    W = fuse.snf([fuse.affinity(D, 5) for D in Ds], k=5, t=10)
    assert W.shape == (len(foods), len(foods)) and np.allclose(W, W.T)
    labels = fuse.leiden(fuse.knn_sparsify(W, 5), 0.5)
    assert len(set(labels)) >= 2
