#!/usr/bin/env python3
"""Turn the Fineli open-data package into the pipeline's food table.

    python scripts/prepare_fineli.py [--raw data/raw/fineli/unz] [--out data/processed/foods_fi.csv]

Input: Fineli Release 20, basic package 2 (4,232 foods, 74 components; THL,
CC BY 4.0), unzipped. Files are ';'-separated, Latin-1, decimal comma.

Output columns follow the Swedish table so the same config and scripts run:
    macros      PROT_g FAT_g CHO_g (available) FIBT_g (FIBC, total fibre) SUGAR_g ...
    sugars      MNSAC_g = glucose + fructose + galactose; SUCS_g; LACS_g; MALS_g;
                other_disac_g = lactose + maltose
    starch      starch_est_g = measured STARCH (not CHO - sugars as in Sweden)
    fat         FASAT_g FAMS_g (cis-MUFA) FAPU_g; n6_g (total n-6); ala_g (18:3 n-3);
                lc_n3_g = total n-3 - ALA (EPA + DHA + others)
    fibre       FIBINS_g (insoluble), PSACNCS_g (soluble non-cellulosic polysaccharides)
Fineli has no C4-C10 fatty acids, so the milk-fat block is skipped.

GIFT group/subgroup: from Fineli's food-use class (FUCLASS) through the curated
data/curated/fineli_fuclass_to_gift.csv, refined by a few name rules
(gift_source says which). Fineli has no FoodEx2 codes.

Protein fractions: every recipe is expanded recursively to raw ingredients
(contribfood.csv). Each raw ingredient's ingredient class (IGCLASS, with name
rules for "other grains") points to a curated protein category; a food's
profile is the protein-weighted mix of its ingredients' profiles, on the
common solubility classes. `protein_profiled_share` is the share of the
food's ingredient protein that has a sourced or provisional profile; foods
under --min-profiled get no profile.
"""

from __future__ import annotations

import argparse
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

NUTRIENTS = {  # Fineli EUFDNAME -> (column, factor to g/100 g)
    "ENERC": ("ENERC_kJ", 1), "PROT": ("PROT_g", 1), "FAT": ("FAT_g", 1), "CHOAVL": ("CHO_g", 1),
    "FIBC": ("FIBT_g", 1), "SUGAR": ("SUGAR_g", 1), "STARCH": ("starch_est_g", 1), "ALC": ("ALC_g", 1),
    "WATER": ("WATER_g", 1), "GLUS": ("GLUS_g", 1), "FRUS": ("FRUS_g", 1), "GALS": ("GALS_g", 1),
    "SUCS": ("SUCS_g", 1), "LACS": ("LACS_g", 1), "MALS": ("MALS_g", 1), "SUGOH": ("SUGOH_g", 1),
    "FIBINS": ("FIBINS_g", 1), "PSACNCS": ("PSACNCS_g", 1), "FASAT": ("FASAT_g", 1), "FAMCIS": ("FAMS_g", 1),
    "FAPU": ("FAPU_g", 1), "FATRN": ("FATRN_g", 1), "FAPUN3": ("FAPUN3_g", 1), "FAPUN6": ("n6_g", 1),
    "F18D3N3": ("ala_g", 0.001), "F20D5N3": ("EPA_g", 0.001), "F22D6N3": ("DHA_g", 0.001),
    "F20D4N6": ("AA_g", 0.001), "F18D2CN6": ("LA_g", 0.001), "CHOLE": ("CHORL_mg", 1), "NA": ("NA_mg", 1),
}

# Fineli ingredient class -> curated protein category (data/curated/protein_fractions.csv)
IG_TO_PROTEIN = {
    "WHEAT": "wheat", "RYE": "rye", "OAT": "oats", "OATLIQ": "oats", "BARLEY": "barley", "RICE": "rice",
    "MILKFF": "milk_bovine", "MILKLF": "milk_bovine", "MILKHF": "milk_bovine", "MILKINGR": "milk_bovine",
    "CREAM": "cream", "SOURMILK": "kefir_fermented_milk", "MILKCURL": "kefir_fermented_milk",
    "YOGHURHI": "yogurt", "YOGHURLO": "yogurt", "CURD": "quark_cottage", "SOUCREAM": "cream",
    "CHEECUHI": "cheese_rennet", "CHEECULO": "cheese_rennet", "CHEEUNHI": "quark_cottage",
    "CHEEUNLO": "quark_cottage", "CHEEPRHI": "cheese_rennet", "CHEEPRLO": "cheese_rennet",
    "FISH": "fish", "FISHPROD": "fish", "BEEF": "meat_muscle", "PORK": "meat_muscle", "LAMM": "meat_muscle",
    "POULTRY": "meat_muscle", "GAME": "meat_muscle", "MEATCUTS": "meat_processed", "SAUSAGE": "meat_processed",
    "SAUSCUTS": "meat_processed", "MEATPROD": "meat_processed", "OFFAL": "offal", "EGG": "egg", "EGGOTH": "egg",
    "SOYAPROD": "soybean", "NUTSEED": "tree_nuts", "POTATO": "potato", "POTAPROD": "potato",
}
NAME_RULES_GRAIN = [(r"\bMAIZE|\bCORN\b|POLENTA", "maize"), (r"SOY", "soybean"), (r"PEA\b|PEAS\b", "pea"),
                    (r"LENTIL", "lentil"), (r"CHICKPEA", "chickpea"), (r"FABA|BROAD BEAN", "faba_bean"),
                    (r"KIDNEY", "kidney_bean"), (r"PEANUT", "peanut")]

NAME_RULES_GIFT = [  # (fuclass list or None, regex on English name, subgroup code)
    (["CERBRKF", "FLOUR", "PORR", "BISCUSAV"], r"\bMAIZE|\bCORN\b|POLENTA", "102"),
    (["CERBRKF", "FLOUR", "PORR"], r"\bRICE\b", "101"),
    (["CERBRKF", "FLOUR", "PORR", "BISCUSAV"], r"WHEAT|SEMOLINA|GRAHAM|SPELT|BULGUR|COUSCOUS", "103"),
    (["FLOUR", "PORR"], r"MILLET", "105"),
    (["PASTAADD"], r"RICE NOODLE|RICE VERMICELLI", "101"),
    (["FISH"], r"PIKE|PERCH|ZANDER|BREAM|ROACH|BURBOT|VENDACE|IDE\b|CARP|CATFISH", "601"),
    (["FISH"], r"SALMON|TROUT|WHITEFISH|CHAR\b|EEL", "602"),
    (["FISH"], r"HERRING|COD|SAITHE|HADDOCK|TUNA|MACKEREL|SARDINE|SPRAT|PLAICE|FLOUNDER|HALIBUT|ANCHOV|POLLOCK|HAKE|REDFISH|POLLACK|SOLE\b|SEA BASS|SHARK", "603"),
    (["FRUFRESH"], r"ORANGE|LEMON|LIME|GRAPEFRUIT|MANDARIN|CLEMENTINE|POMELO", "1002"),
    (["FRUFRESH"], r"APRICOT|MANGO|PAPAYA|PEACH|NECTARINE|PERSIMMON|MELON, MUSK|CANTALOUPE", "1001"),
    (["VEGFRESH"], r"CARROT|PUMPKIN|SQUASH|SWEET POTATO|PEPPER, RED|SWEET PEPPER, RED", "902"),
    (["VEGFRESH"], r"SPINACH|LETTUCE|KALE|CHARD|CABBAGE|ROCKET|PARSLEY|DILL|NETTLE|LEAF", "901"),
    (["LEGUMES"], r"SOY", "302"),
    (["LEGUMES"], r"GREEN BEAN|STRING BEAN|PEA, GREEN\b(?!, DRIED)|SUGAR PEA|MANGETOUT", "903"),
    (["SNACKVEG"], r"RAISIN|DRIED|PRUNE|DATE\b|FIG\b|APRICOT", "1005"),
    (["SNACKSAV"], r"\bNUT\b|PEANUT|ALMOND|CASHEW|SEED", "303"),
    (["VEGJUICE"], r"OAT DRINK|RICE DRINK|SOY DRINK|SOYA DRINK|ALMOND DRINK|OAT BEVERAGE|SOYA BEVERAGE", "1407"),
    (["INGRMISC"], r"COCOA", "1403"),
]


def read(raw: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(raw / name, sep=";", encoding="latin1", dtype=str)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")


def gift_columns(food: pd.DataFrame, names: pd.Series, fu_map: pd.DataFrame, gift: pd.DataFrame) -> pd.DataFrame:
    sub = food["FUCLASS"].map(fu_map.set_index("fuclass")["gift_subgroup_code"].astype(str))
    src = pd.Series(np.where(sub.notna(), "fineli_fuclass", "unmapped"), index=food.index)
    for classes, rx, code in NAME_RULES_GIFT:
        m = food["FUCLASS"].isin(classes) & names.str.contains(rx, regex=True, na=False)
        sub[m] = code
        src[m] = "fineli_fuclass+name_rule"
    g = gift.dropna(subset=["gift_subgroup_code"]).drop_duplicates("gift_subgroup_code").set_index("gift_subgroup_code")
    return pd.DataFrame({"gift_subgroup_code": sub, "gift_group": sub.map(g["gift_group"]),
                         "gift_subgroup": sub.map(g["gift_subgroup"]), "gift_source": src}, index=food.index)


def protein_profiles(food, names, comp, recipes, curated, min_profiled):
    """Recipe-expanded, protein-weighted protein-fraction profile per food (common classes)."""
    common = (curated[curated["share"].notna()]
              .pivot_table(index="category", columns="common_class", values="share", aggfunc="sum").fillna(0.0))
    common = common[(common.sum(axis=1) - 1).abs() < 0.02]
    classes = list(common.columns)
    prot = comp.get("PROT_g", pd.Series(dtype=float))
    ig = food.set_index("FOODID")["IGCLASS"]
    nm = names
    kids = recipes.groupby("FOODID").apply(lambda d: list(zip(d["CONFDID"], d["MASS"])), include_groups=False).to_dict()

    def leaf_category(fid):
        c = IG_TO_PROTEIN.get(ig.get(fid))
        if c is None and ig.get(fid) in ("CEROTH", "PEABEAN", "LEGUPROD", "STARCH"):
            for rx, cat in NAME_RULES_GRAIN:
                if re.search(rx, str(nm.get(fid, ""))):
                    return cat
        return c

    @lru_cache(maxsize=None)
    def expand(fid, depth=0):
        """dict category -> grams protein per 100 g of this food's recipe mass (unscaled weights)."""
        if fid in kids and depth < 8:
            tot = {}
            mass = sum(m for _, m in kids[fid] if m == m)
            for cid, m in kids[fid]:
                if not m or m != m:
                    continue
                for c, p in expand(cid, depth + 1).items():
                    tot[c] = tot.get(c, 0.0) + p * m / max(mass, 1e-9)
            return tot
        p = prot.get(fid, np.nan)
        if not p or p != p:
            return {}
        return {leaf_category(fid) or "_unprofiled": float(p)}

    rows = []
    for fid in food["FOODID"]:
        e = expand(fid)
        tot = sum(e.values())
        prof = {c: v for c, v in e.items() if c in common.index}
        ptot = sum(prof.values())
        row = {"FOODID": fid, "protein_profiled_share": ptot / tot if tot > 0 else np.nan,
               "protein_main_source": max(e, key=e.get) if e else None}
        if tot > 0 and ptot / tot >= min_profiled:
            vec = sum(common.loc[c] * v for c, v in prof.items()) / ptot
            row.update({f"protfrac_{k}": round(float(vec[k]), 4) for k in classes})
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/fineli/unz")
    ap.add_argument("--out", default="data/processed/foods_fi.csv")
    ap.add_argument("--gift-matrix", default="data/raw/gift/foodex2_gift_nut.csv")
    ap.add_argument("--fuclass-map", default="data/curated/fineli_fuclass_to_gift.csv")
    ap.add_argument("--protein-table", default="data/curated/protein_fractions.csv")
    ap.add_argument("--min-profiled", type=float, default=0.7)
    args = ap.parse_args(argv)
    raw = Path(args.raw)

    food = read(raw, "food.csv")
    names = read(raw, "foodname_EN.csv").set_index("FOODID")["FOODNAME"]
    fu = read(raw, "fuclass_EN.csv").set_index("THSCODE")["DESCRIPT"]
    igd = read(raw, "igclass_EN.csv").set_index("THSCODE")["DESCRIPT"]
    cv = read(raw, "component_value.csv")
    cv["v"] = num(cv["BESTLOC"])
    cv = cv[cv["EUFDNAME"].isin(NUTRIENTS)]
    wide = cv.pivot_table(index="FOODID", columns="EUFDNAME", values="v", aggfunc="first")
    comp = pd.DataFrame({NUTRIENTS[k][0]: wide[k] * NUTRIENTS[k][1] for k in NUTRIENTS if k in wide})

    df = pd.DataFrame({"food_id": food["FOODID"].astype(int), "name": food["FOODID"].map(names),
                       "name_fi": food["FOODNAME"], "food_type": food["FOODTYPE"], "process": food["PROCESS"],
                       "fuclass": food["FUCLASS"], "igclass": food["IGCLASS"],
                       "main_group": food["FUCLASS"].map(fu), "ingredient_class": food["IGCLASS"].map(igd)})
    df = pd.concat([df, comp.reindex(food["FOODID"]).reset_index(drop=True)], axis=1)
    zero = pd.Series(0.0, index=df.index)
    z = lambda c: df[c].fillna(0) if c in df else zero
    df["MNSAC_g"] = (z("GLUS_g") + z("FRUS_g") + z("GALS_g")).round(3)
    df["DISAC_g"] = (z("SUCS_g") + z("LACS_g") + z("MALS_g")).round(3)
    df["other_disac_g"] = (z("LACS_g") + z("MALS_g")).round(3)
    df["lc_n3_g"] = (z("FAPUN3_g") - z("ala_g")).clip(lower=0).round(4)
    df["description"] = df["main_group"].fillna("") + " | " + df["ingredient_class"].fillna("")

    gift = pd.read_csv(args.gift_matrix, dtype=str)
    g = gift_columns(food, df["name"], pd.read_csv(args.fuclass_map, dtype=str), gift)
    df = pd.concat([df, g.reset_index(drop=True)], axis=1)
    df["gift_group"] = df["gift_group"].fillna("unassigned")
    df["gift_subgroup"] = df["gift_subgroup"].fillna("unassigned")

    rec = read(raw, "contribfood.csv")
    rec["MASS"] = num(rec["MASS"])
    pp = protein_profiles(food, names, comp, rec,
                          pd.read_csv(args.protein_table), args.min_profiled)
    df = pd.concat([df, pp.drop(columns="FOODID").reset_index(drop=True)], axis=1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    has = df.filter(like="protfrac_").notna().all(axis=1)
    print(f"Wrote {len(df)} foods, {df.shape[1]} columns -> {out}")
    print(f"GIFT group assigned: {(df['gift_group'] != 'unassigned').sum()}; by source:",
          df["gift_source"].value_counts().to_dict())
    print(f"Protein-fraction profile: {has.sum()} foods (profiled share >= {args.min_profiled})")


if __name__ == "__main__":
    main()
