"""Prepare step plus a full pipeline run with the default config on synthetic
foods that use Livsmedelsverket column names."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from foodspace import run

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prep_lv", ROOT / "scripts" / "prepare_livsmedelsverket.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def synthetic_raw(folder: Path, n_per=8, seed=0):
    rng = np.random.default_rng(seed)
    kinds = {  # PROT, FAT, CHO, FIBT, SUGAR, MNSAC, SUCS, DISAC, FASAT, FAMS, FAPU (illustrative)
        "Milk": [3.4, 3.0, 4.8, 0, 4.8, 0, 0, 4.8, 1.9, 0.8, 0.1],
        "Cheese": [25, 28, 0.1, 0, 0.1, 0, 0, 0.1, 18, 7, 1],
        "Bread": [9, 3, 45, 6, 3, 1, 1.5, 0.5, 0.6, 1.0, 1.2],
        "Apple": [0.3, 0.2, 12, 2.4, 10, 7, 3, 3, 0.03, 0.01, 0.06],
    }
    cols = ["PROT_g", "FAT_g", "CHO_g", "FIBT_g", "SUGAR_g", "MNSAC_g", "SUCS_g", "DISAC_g", "FASAT_g", "FAMS_g", "FAPU_g"]
    rows, foods, i = [], [], 0
    for k, v in kinds.items():
        for j in range(n_per):
            vals = np.clip(rng.normal(v, np.array(v) * 0.08 + 1e-3), 0, None).round(3)
            rows.append({"food_id": i, "name": f"{k} {j}", "foodex2_code": f"X{k[:3]}", "main_group": k, **dict(zip(cols, vals))})
            foods.append({"food_id": i, "name_sv": f"{k} sv {j}", "food_type": "Analysed", "scientific_name": ""})
            i += 1
    folder.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(folder / "nutrients_wide.csv", index=False)
    pd.DataFrame(foods).to_csv(folder / "foods.csv", index=False)


def test_prepare_and_run_default_config(tmp_path):
    raw = tmp_path / "raw"
    synthetic_raw(raw)
    gift = tmp_path / "gift.csv"
    pd.DataFrame({"foodex2_code": ["XMil", "XChe", "XBre", "XApp"],
                  "gift_group": ["Milk and milk products", "Milk and milk products", "Cereals", "Fruits"],
                  "gift_subgroup": ["Milk", "Cheese", "Bread", "Fruit"]}).to_csv(gift, index=False)
    pmap = tmp_path / "pmap.csv"
    pd.DataFrame({"food_id": list(range(0, 8)) + list(range(8, 16)),
                  "protein_fraction_category": ["milk_bovine"] * 8 + ["cheese_rennet"] * 8}).to_csv(pmap, index=False)
    out = tmp_path / "foods_lv.csv"
    prep.main(["--raw", str(raw), "--out", str(out), "--gift-map", str(gift), "--protein-map", str(pmap)])
    df = pd.read_csv(out)
    assert (df["starch_est_g"] >= 0).all() and "other_disac_g" in df
    milk = df[df.name.str.startswith("Milk")]
    assert (milk["other_disac_g"] > 4).all()          # lactose shows up as "other disaccharides"
    assert (df["gift_group"] != "unassigned").all()

    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    cfg["input"] = str(out)
    cfg["output_dir"] = str(tmp_path / "out")
    cfg["graph"]["k"] = 5
    cfg["layout"]["method"] = "tsne"
    cfg["fractions"]["curated_table"] = str(ROOT / "data" / "curated" / "protein_fractions.csv")
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    run.main(str(tmp_path / "cfg.yaml"))
    m = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert {"fat_type", "sugar_type", "carbohydrate_type", "protein_type"} <= set(m["fraction_blocks"])
    assert m["vs_gift_subgroup"]["ARI"] > 0.8
