import pandas as pd

import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("map_gift", pathlib.Path(__file__).parents[1] / "scripts" / "map_gift.py")
mg = importlib.util.module_from_spec(spec); spec.loader.exec_module(mg)


def test_unmapped_terms_inherit_nearest_mapped_ancestor(tmp_path):
    m = pd.DataFrame([
        ["Z1", "1", "A0", "Grains", "", "", "", "", "", ""],
        ["Z1.1", "2", "A1", "Cereal grains", "1", "Cereals and their products", "Cereals", "106", "Other cereals", "Other"],
        ["Z1.1.1", "3", "A2", "Deprecated grain", "", "", "", "", "", ""],
    ], columns=["expo_hierarchy_code", "level", "foodex2_code", "foodex2_desc", "gift_group_code", "gift_group",
                "gift_group_short", "gift_subgroup_code", "gift_subgroup", "gift_subgroup_short"])
    m.to_csv(tmp_path / "m.csv", index=False)
    mg.main(["--matrix", str(tmp_path / "m.csv"), "--out", str(tmp_path / "o.csv")])
    o = pd.read_csv(tmp_path / "o.csv", dtype=str).fillna("").set_index("foodex2_code")
    assert o.loc["A2", "gift_group"] == "Cereals and their products" and o.loc["A2", "gift_source"] == "ancestor A1"
    assert o.loc["A0", "gift_source"] == "unmapped"
