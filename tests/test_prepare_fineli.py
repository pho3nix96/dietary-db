import importlib.util
import pathlib

import pandas as pd

spec = importlib.util.spec_from_file_location("pf", pathlib.Path(__file__).parents[1] / "scripts" / "prepare_fineli.py")
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)


def w(d, name, text):
    (d / name).write_text(text, encoding="latin1")


def test_fineli_gift_and_recipe_protein_profile(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    w(raw, "food.csv", "FOODID;FOODNAME;FOODTYPE;PROCESS;EDPORT;IGCLASS;IGCLASSP;FUCLASS;FUCLASSP\n"
      "1;VEHNÄJAUHO;FOOD;IND;100;WHEAT;CERTOT;FLOUR;CEBAKTOT\n"
      "2;MAITO;FOOD;IND;100;MILKLF;MILKTOT;MILKLF;MILKDTOT\n"
      "3;PULLA;DISH;BAK;100;DISH;NONINGR;BUN;CEBAKTOT\n")
    w(raw, "foodname_EN.csv", "FOODID;FOODNAME;LANG\n1;FLOUR, WHEAT;EN\n2;MILK, 1% FAT;EN\n3;BUN;EN\n")
    w(raw, "fuclass_EN.csv", "THSCODE;DESCRIPT;LANG\nFLOUR;Flour;EN\nMILKLF;Milks <2% fat;EN\nBUN;Buns;EN\n")
    w(raw, "igclass_EN.csv", "THSCODE;DESCRIPT;LANG\nWHEAT;Wheat;EN\nMILKLF;Milks;EN\nDISH;Dish;EN\n")
    w(raw, "component_value.csv", "FOODID;EUFDNAME;BESTLOC;ACQTYPE;METHTYPE;METHIND\n"
      "1;PROT;10,0;S;S;x\n1;FAT;1,0;S;S;x\n1;CHOAVL;70;S;S;x\n1;STARCH;69;S;S;x\n"
      "2;PROT;3,4;S;S;x\n2;FAT;1,0;S;S;x\n2;CHOAVL;4,8;S;S;x\n2;LACS;4,8;S;S;x\n"
      "3;PROT;8,0;S;S;x\n3;FAT;10;S;S;x\n3;CHOAVL;50;S;S;x\n")
    # bun: 100 g flour (10 g protein) + 100 g milk (3.4 g protein)
    w(raw, "contribfood.csv", "FOODID;CONFDID;AMOUNT;FOODUNIT;MASS;EVREMAIN;RECYEAR\n3;1;100;G;100,00;100;\n3;2;100;G;100,00;100;\n")
    gift = pd.DataFrame({"gift_group": ["Cereals and their products", "Cereals and their products", "Milk and milk products", "Sweets and sugars"],
                         "gift_subgroup": ["Other and mixed cereals", "Wheat", "Milk", "Dough-based sweets"],
                         "gift_subgroup_code": ["106", "103", "401", "1201"]})
    gift.to_csv(tmp_path / "gift.csv", index=False)
    pd.DataFrame({"fuclass": ["FLOUR", "MILKLF", "BUN"], "gift_subgroup_code": ["106", "401", "1201"]}).to_csv(tmp_path / "fu.csv", index=False)
    prot = pd.DataFrame([("wheat", "prolamin", "alcohol_soluble", 0.6), ("wheat", "glutelin", "insoluble_storage", 0.2),
                         ("wheat", "albumin_globulin", "mixed", 0.2), ("milk_bovine", "casein", "casein", 0.8),
                         ("milk_bovine", "whey", "water_soluble", 0.2)],
                        columns=["category", "fraction", "common_class", "share"])
    prot.to_csv(tmp_path / "prot.csv", index=False)
    out = tmp_path / "fi.csv"
    pf.main(["--raw", str(raw), "--out", str(out), "--gift-matrix", str(tmp_path / "gift.csv"),
             "--fuclass-map", str(tmp_path / "fu.csv"), "--protein-table", str(tmp_path / "prot.csv")])
    df = pd.read_csv(out).set_index("food_id")
    assert df.loc[1, "gift_subgroup"] == "Wheat" and df.loc[1, "gift_source"] == "fineli_fuclass+name_rule" and df.loc[3, "gift_group"] == "Sweets and sugars"
    assert df.loc[2, "other_disac_g"] == 4.8 and df.loc[1, "starch_est_g"] == 69
    wheat = 10 / 13.4
    assert abs(df.loc[3, "protfrac_alcohol_soluble"] - 0.6 * wheat) < 1e-3
    assert abs(df.loc[3, "protfrac_casein"] - 0.8 * (1 - wheat)) < 1e-3
