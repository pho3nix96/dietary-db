import numpy as np
import pandas as pd
import yaml

from foodspace.placement import FoodSpace


def _space(tmp_path):
    rows = []
    for i, (name, grp, p, f, c) in enumerate([
        ("Milk 3% fat", "Milk", 3.4, 3.0, 4.8), ("Milk 1% fat", "Milk", 3.4, 1.0, 4.8), ("Milk skimmed", "Milk", 3.5, 0.1, 4.9),
        ("Cheese hard 28% fat", "Milk", 26, 28, 0.5), ("Cheese hard 17% fat", "Milk", 28, 17, 0.5), ("Cheese soft 20%", "Milk", 20, 20, 1),
        ("Rye bread", "Cereals", 8, 2, 45), ("Wheat bread", "Cereals", 9, 3, 50), ("Oat flakes", "Cereals", 13, 7, 60),
        ("Rice boiled", "Cereals", 2.5, 0.3, 28)]):
        rows.append({"food_id": i, "name": name, "gift_group": grp, "gift_subgroup": grp + " sub",
                     "PROT_g": p, "FAT_g": f, "CHO_g": c, "FIBT_g": 1.0, "SUGAR_g": 1.0})
    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "t.csv", index=False)
    res = tmp_path / "res"
    res.mkdir()
    df.assign(cluster=[0, 0, 0, 1, 1, 1, 0, 0, 0, 1], x=np.linspace(-1, 1, 10), y=0.0)[["food_id", "cluster", "x", "y"]] \
        .to_csv(res / "foods.csv", index=False)
    cfg = {"columns": {"nutrients": ["PROT_g", "FAT_g", "CHO_g", "FIBT_g"]},
           "nutrients": {"metric": "cosine"}, "fractions": {"blocks": {}}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    return FoodSpace(tmp_path / "t.csv", res, tmp_path / "c.yaml")


def test_name_picks_group_and_nutrients_pick_neighbours(tmp_path):
    fs = _space(tmp_path)
    p = fs.place("Cheese hard 31% fat", {"PROT_g": 27, "FAT_g": 31, "CHO_g": 0.4, "FIBT_g": 0})
    assert p.gift_group == "Milk" and p.cluster == 1
    assert p.composition_neighbours[0][0].startswith("Cheese")
    q = fs.place("Milk 2% fat")  # no values: placed by name only
    assert q.gift_group == "Milk" and q.basis.startswith("name only")


def test_name_evaluation_runs(tmp_path):
    r = _space(tmp_path).evaluate_names(n=10, k=3)
    assert 0 <= r["group_accuracy"] <= 1 and r["foods_tested"] == 10
