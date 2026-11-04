#!/usr/bin/env python3
"""Turn the Livsmedelsverket download into the pipeline's food table.

    python scripts/prepare_livsmedelsverket.py
        [--raw data/raw/livsmedelsverket] [--out data/processed/foods_lv.csv]

Adds derived fraction columns that the database does not report directly:
    starch_est_g        available carbohydrate minus total sugars (CHO_g - SUGAR_g),
                        an estimate of starch (plus minor other carbohydrates);
                        differences under 0.5 g are set to 0 (rounding noise)
    other_disac_g       disaccharides other than sucrose (DISAC_g - SUCS_g), i.e.
                        mainly lactose and maltose: it separates dairy sugars from
                        table sugar
    n6_g, ala_g, lc_n3_g  omega-6 (18:2 + 20:4), alpha-linolenic acid (18:3) and
                        long-chain omega-3 (EPA + DPA + DHA)
    sc_mc_sfa_g, long_sfa_g
                        C4-C10 saturated fatty acids (milk-fat marker) and the
                        rest of the saturated fat
Negative differences from rounding are set to 0.

Joins two optional curated tables when present:
    data/curated/foodex2_to_gift.csv          foodex2_code, gift_group, gift_subgroup
                                              (built by scripts/map_gift.py from the GIFT matrix)
    data/curated/protein_category_map.csv     food_id, protein_fraction_category
Foods without a GIFT mapping get "unassigned", so the pipeline still runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def derive(df: pd.DataFrame) -> pd.DataFrame:
    def diff(a, b):
        if a in df and b in df:
            return (pd.to_numeric(df[a], errors="coerce") - pd.to_numeric(df[b], errors="coerce")).clip(lower=0)
        return None

    s = diff("CHO_g", "SUGAR_g")
    if s is not None:
        # differences under 0.5 g are rounding noise (e.g. milk), not starch
        df["starch_est_g"] = s.where(s >= 0.5, 0.0).round(3)
    s = diff("DISAC_g", "SUCS_g")
    if s is not None:
        df["other_disac_g"] = s.round(3)

    def num(c):
        return pd.to_numeric(df[c], errors="coerce").fillna(0) if c in df else None

    # polyunsaturated fat by family; the remainder (other PUFA) is added by the block itself
    if all(c in df for c in ("F18:2_g", "F20:4_g", "F18:3_g", "F20:5_g", "F22:5_g", "F22:6_g")):
        df["n6_g"] = (num("F18:2_g") + num("F20:4_g")).round(3)
        df["ala_g"] = num("F18:3_g").round(3)
        df["lc_n3_g"] = (num("F20:5_g") + num("F22:5_g") + num("F22:6_g")).round(3)
    # short- and medium-chain saturated fatty acids (C4-C10) occur in milk fat only
    if "F4-10:0_g" in df and "FASAT_g" in df:
        df["sc_mc_sfa_g"] = num("F4-10:0_g").round(3)
        df["long_sfa_g"] = (num("FASAT_g") - num("F4-10:0_g")).clip(lower=0).round(3)
    return df


def first_code(codes: str) -> str:
    """FoodEx2 base term of the first code: 'A037M#F28.A07LT|...' -> 'A037M' (facets dropped)."""
    return str(codes).split("|")[0].split("#")[0].strip() if isinstance(codes, str) and codes else ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/livsmedelsverket")
    ap.add_argument("--out", default="data/processed/foods_lv.csv")
    ap.add_argument("--gift-map", default="data/curated/foodex2_to_gift.csv")
    ap.add_argument("--protein-map", default="data/curated/protein_category_map.csv")
    ap.add_argument("--protein-table", default="data/curated/protein_fractions.csv")
    args = ap.parse_args(argv)

    raw = Path(args.raw)
    wide = pd.read_csv(raw / "nutrients_wide.csv")
    foods = pd.read_csv(raw / "foods.csv")[["food_id", "name_sv", "food_type", "scientific_name"]]
    df = wide.merge(foods, on="food_id", how="left")
    df["main_group"] = df["main_group"].astype("string").str.strip()
    df["description"] = df["main_group"].fillna("")
    df = derive(df)

    df["foodex2_base"] = df["foodex2_code"].map(first_code)
    gift = Path(args.gift_map)
    if gift.exists():
        g = pd.read_csv(gift, dtype=str)
        g = g[[c for c in ("foodex2_code", "gift_group", "gift_subgroup", "gift_source") if c in g]]
        df = df.merge(g, left_on="foodex2_base", right_on="foodex2_code", how="left", suffixes=("", "_map"))
        df = df.drop(columns=[c for c in df.columns if c.endswith("_map")])
        print(f"GIFT groups assigned to {df['gift_group'].notna().sum()} of {len(df)} foods")
    else:
        print(f"No GIFT mapping at {gift}; all foods marked 'unassigned' for now")
    for c in ("gift_group", "gift_subgroup"):
        if c not in df:
            df[c] = pd.NA
        df[c] = df[c].fillna("unassigned")

    pmap = Path(args.protein_map)
    if pmap.exists():
        p = pd.read_csv(pmap, dtype={"food_id": int, "protein_fraction_category": str})[["food_id", "protein_fraction_category"]]
        df = df.merge(p, on="food_id", how="left")
        print(f"Protein-fraction categories set for {df['protein_fraction_category'].notna().sum()} foods")
    elif "protein_fraction_category" not in df:
        df["protein_fraction_category"] = pd.NA

    rav = raw / "raw_commodities.csv"
    if rav.exists() and Path(args.protein_table).exists():
        from foodspace.commodities import recipe_protein_profiles

        pp = recipe_protein_profiles(pd.read_csv(rav, dtype={"foodex2_code": str}),
                                     pd.to_numeric(df.set_index("food_id")["PROT_g"], errors="coerce"),
                                     pd.read_csv(args.protein_table))
        df = df.merge(pp, on="food_id", how="left")
        print(f"Recipe-based protein profiles for {df.filter(like='protfrac_').notna().all(axis=1).sum()} foods")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} foods with {df.shape[1]} columns -> {out}")


if __name__ == "__main__":
    main()
