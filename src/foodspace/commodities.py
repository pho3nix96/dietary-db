"""Protein-fraction profiles of composite foods from their raw-commodity recipes.

Livsmedelsverket lists, for every food, the raw commodities it is made of
(``/livsmedel/{nr}/ravaror``): each commodity's share of the recipe and a
factor converting it to grams of raw commodity per 100 g of food.
For cereal foods this tells us how much wheat, rye, oat ... went into a bread
or a porridge. Combining that with

* the protein content of each raw commodity (taken from the single-commodity
  foods in the same database), and
* the Osborne fraction shares of each grain (``data/curated/protein_fractions.csv``),

gives a protein-weighted fraction profile for each product:

    protein_from_grain_g   = amount_g / 100 * commodity_protein_g_per_100g
    fraction_share[f]      = sum_g protein_from_grain_g * share[g, f] / sum_g protein_from_grain_g

``coverage`` is the cereal protein divided by the food's total protein, so a
fish gratin with breadcrumbs gets a low coverage and can be left out.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# FoodEx2 raw-commodity codes used by Livsmedelsverket -> grain.
# Brans, germs and flours count as their grain; their fraction mix differs from
# the whole grain (bran and germ are richer in albumins and globulins), which is
# a known simplification.
GRAIN_OF = {
    "A001M": "wheat", "A003Y": "wheat", "A004P": "wheat", "A004Q": "wheat", "A004M": "wheat",
    "A001K": "rye", "A003K": "rye", "A003Q": "rye",
    "A000G": "oats", "A002Y": "oats", "A003B": "oats",
    "A000P": "barley", "A002L": "barley",
    "A000T": "maize", "A002Q": "maize", "A002P": "maize",
    "A001C": "rice", "A003F": "rice",
    "A000N": "buckwheat", "A000Y": "millet", "A001L": "sorghum", "A000A": "teff",
    "A000R": "quinoa", "A000M": "amaranth",
}

# Fallback protein (g per 100 g raw commodity) where the database has no clean
# single-commodity food. Typical table values; replace when better data exist.
PROTEIN_FALLBACK = {
    "A004M": 0.3,   # wheat starch
    "A000T": 8.5,   # maize grain (the only single-commodity food is corn starch)
    "A000R": 14.0,  # quinoa
    "A003K": 6.5,   # light rye flour
    "A002L": 9.0,   # barley flour
    "A002P": 8.0,   # maize meal
    "A002Y": 11.3,  # oat flour (taken as oat grain)
}


def parse_mix(s: str) -> dict[str, float]:
    """'A001K:36,A03DL:6' -> {'A001K': 36.0, 'A03DL': 6.0}"""
    if not isinstance(s, str) or not s:
        return {}
    out = {}
    for part in s.split(","):
        code, _, amt = part.partition(":")
        try:
            out[code] = out.get(code, 0.0) + float(amt)
        except ValueError:
            continue
    return out


WATER = {"A03DL"}  # tap water: left out of recipe mass shares


def load_mixes(path: str | Path, value: str = "raw_equivalent_g") -> dict[int, dict[str, float]]:
    """raw_commodities.csv -> {food_id: {foodex2_code: value}}.

    `raw_equivalent_g` is grams of raw commodity per 100 g food (share x factor);
    `share_pct` is the commodity's share of the recipe as listed.
    """
    t = pd.read_csv(path, dtype={"foodex2_code": str})
    t[value] = pd.to_numeric(t[value], errors="coerce").fillna(0.0)
    g = t.groupby(["food_id", "foodex2_code"])[value].sum()
    out: dict[int, dict[str, float]] = {}
    for (fid, code), v in g.items():
        out.setdefault(int(fid), {})[code] = float(v)
    return out


def grain_mass_share(recipe: dict[str, float]) -> float:
    """Share of grain commodities in a recipe (by listed share, water excluded)."""
    tot = sum(v for c, v in recipe.items() if c not in WATER)
    return sum(v for c, v in recipe.items() if c in GRAIN_OF) / tot if tot > 0 else 0.0


def commodity_protein(df: pd.DataFrame, mixes: dict[int, dict[str, float]], codes,
                      id_col="food_id", prot_col="PROT_g") -> pd.Series:
    """Protein per 100 g of each raw commodity, from foods made of 100 g of that commodity only."""
    rows = []
    for fid, p in zip(df[id_col], pd.to_numeric(df[prot_col], errors="coerce")):
        m = mixes.get(int(fid), {})
        if len(m) == 1:
            (code, amt), = m.items()
            if code in codes and abs(amt - 100) < 1e-9 and p > 0:
                rows.append((code, p))
    est = pd.DataFrame(rows, columns=["code", "prot"]).groupby("code")["prot"].median() if rows else pd.Series(dtype=float)
    out = pd.Series({c: est.get(c, np.nan) for c in codes}, dtype=float)
    for c, v in PROTEIN_FALLBACK.items():
        if c in out.index and (np.isnan(out[c]) or c == "A000T"):
            out[c] = v
    return out


OSBORNE = ["albumin_globulin", "prolamin", "glutelin"]


def grain_shares(table: pd.DataFrame, fractions=OSBORNE, scheme: str = "osborne") -> pd.DataFrame:
    """Curated table -> grains x fractions matrix of shares.

    Keeps the categories (grains) that have every one of `fractions` filled in.
    """
    t = table[(table["scheme"] == scheme) & table["share"].notna() & table["fraction"].isin(fractions)]
    S = t.pivot_table(index="category", columns="fraction", values="share", aggfunc="sum")
    return S.reindex(columns=list(fractions)).dropna()


def product_profiles(df: pd.DataFrame, mixes, prot_per_commodity: pd.Series, shares: pd.DataFrame,
                     id_col="food_id", prot_col="PROT_g", dominant_at: float = 0.6) -> pd.DataFrame:
    """One row per food with any cereal commodity.

    Columns: cereal_protein_g, coverage, profiled_share, <grain>_protein_share for
    every grain, <fraction> shares, dominant_grain.
    """
    grains = sorted(set(GRAIN_OF.values()))
    recs = []
    for fid, total in zip(df[id_col], pd.to_numeric(df[prot_col], errors="coerce")):
        m = mixes.get(int(fid), {})
        by_grain = dict.fromkeys(grains, 0.0)
        for code, amt in m.items():
            g = GRAIN_OF.get(code)
            if g is not None and not np.isnan(prot_per_commodity.get(code, np.nan)):
                by_grain[g] += amt / 100.0 * prot_per_commodity[code]
        cp = sum(by_grain.values())
        if cp <= 0:
            continue
        rec = {id_col: int(fid), "cereal_protein_g": cp,
               "coverage": cp / total if total and total > 0 else np.nan}
        prof = sum(v for g, v in by_grain.items() if g in shares.index)
        rec["profiled_share"] = prof / cp
        for g, v in by_grain.items():
            rec[f"{g}_protein_share"] = v / cp
        if prof > 0:
            mix = sum(by_grain[g] * shares.loc[g] for g in shares.index if by_grain[g] > 0) / prof
            rec.update(mix.to_dict())
        top = max(by_grain, key=by_grain.get)
        rec["dominant_grain"] = top if by_grain[top] / cp >= dominant_at else "mixed"
        recs.append(rec)
    return pd.DataFrame(recs)


# --- protein categories for any raw commodity (recipe-based profiles for all foods) ---
# (regex on the commodity name, curated protein category, typical raw protein g/100 g)
COMMODITY_RULES = [
    (r"whey protein|milk protein", "whey_protein", 80.0),
    (r"whey powdered|whey and whey", "whey_protein", 12.0),
    (r"milk powder|dairy concentrate", "milk_bovine", 35.0),
    (r"cow milk|milk, goat|^milk\b|buttermilk|human milk|\b(?:skimmed|semi-skimmed|whole|fat[- ]free|low[- ]fat) milk\b",
     "milk_bovine", 3.4),
    (r"cream|créme fraiche|creme fraiche|sour cream", "cream", 2.5),
    (r"yoghurt", "yogurt", 3.6),
    (r"filmjölk|fermented milk", "kefir_fermented_milk", 3.2),
    (r"fresh uncured cheese", "quark_cottage", 11.0),
    (r"ripened cheese|hard cheese|brined cheese|processed cheese", "cheese_rennet", 24.0),
    (r"liver|kidney|offal|organs", "offal", 18.0),
    (r"fresh meat|\bmeat\b|chicken|turkey|duck|goose|poultry|pheasant|pigeon|ptarmigan|ratites|frog", "meat_muscle", 20.0),
    (r"shrimp|prawn|crab|lobster|mussel|oyster|scallop|crayfish|cephalopod|snail", "fish", 16.0),
    (r"roe", None, 25.0),
    (r"salmon|cod|herring|pollack|plaice|catfish|fish|tuna|mackerel|flounder|haddock|sprat|halibut|trout|perch|pike|whitefish|eel|whiting|tilapia|turbot|bass|bream|char|sardine|ling|monkfish|stockfish|hake", "fish", 19.0),
    (r"egg  yolk|egg yolk", "egg", 16.0),
    (r"egg  white|egg white", "egg", 11.0),
    (r"hen egg|\beggs?\b", "egg", 12.5),
    (r"soya bean|soy", "soybean", 36.0),
    (r"peas \(dry\)", "pea", 22.0),
    (r"lentil", "lentil", 24.0),
    (r"beans \(dry\)", "kidney_bean", 21.0),
    (r"peanut", "peanut", 25.0),
    (r"almond|hazelnut|walnut|cashew|pistachio|pecan|brazil nut|macadamia|pine nut|chestnut|coconut", "tree_nuts", 18.0),
    (r"seed(?! oil)|sesame|linseed|chia|hemp", "oil_seeds", 20.0),
    (r"potato(?! starch)|sweet potato", "potato", 2.0),
]


# commodities with (almost) no protein; whole words only, so buttermilk, salted fish,
# fatty fish and boiled eggs are not caught
NO_PROTEIN = re.compile(r"\b(oils?|fats?|butter|starch|sugars?|syrups?|honey|salt|water|vinegar|juices?|wine|beer|"
                        r"spirits?|liqueur)\b")
# fat-content wording on protein foods ("milk, low fat", "fat-free yoghurt", "milk 3% fat") is not a fat product
FAT_WORDING = re.compile(r"\b(?:low|reduced|non|full|half)[- ]fat\b|\bfat[- ]free\b|\d+(?:[.,]\d+)?\s?%\s?fat\b")
NUT_BUTTER = re.compile(r"\b(?:peanut|nut|almond|cashew|hazelnut|seed|sesame)s? butter\b")


def commodity_category(code: str, name: str):
    """Curated protein category and typical protein (g/100 g raw) for a raw commodity."""
    g = GRAIN_OF.get(code)
    if g is not None:
        return g, PROTEIN_FALLBACK.get(code, 10.0)
    n = str(name).lower()
    if not NUT_BUTTER.search(n) and NO_PROTEIN.search(FAT_WORDING.sub(" ", n)):
        return None, 0.0
    for rx, cat, p in COMMODITY_RULES:
        if re.search(rx, n):
            return cat, p
    return None, 1.5  # vegetables, fruit, mushrooms, spices, cocoa ...: unprofiled, little protein


def recipe_protein_profiles(raw_commodities: pd.DataFrame, food_protein: pd.Series, curated: pd.DataFrame,
                            min_profiled: float = 0.7) -> pd.DataFrame:
    """Protein-fraction profile per food from its raw-commodity recipe (common solubility classes).

    raw_commodities: food_id, foodex2_code, commodity, share_pct, raw_equivalent_g
    food_protein:    PROT_g indexed by food_id (used to estimate each commodity's protein from
                     foods made of that commodity alone; typical values otherwise)
    """
    common = (curated[curated["share"].notna()]
              .pivot_table(index="category", columns="common_class", values="share", aggfunc="sum").fillna(0.0))
    common = common[(common.sum(axis=1) - 1).abs() < 0.02]
    t = raw_commodities.copy()
    t["raw_equivalent_g"] = pd.to_numeric(t["raw_equivalent_g"], errors="coerce").fillna(0.0)
    names = t.drop_duplicates("foodex2_code").set_index("foodex2_code")["commodity"]
    cat = {c: commodity_category(c, n) for c, n in names.items()}
    # commodity protein from single-commodity foods (100 g raw equivalent)
    single = t.groupby("food_id").filter(lambda d: len(d) == 1)
    single = single[(single["raw_equivalent_g"] - 100).abs() < 1e-6]
    est = single.assign(p=single["food_id"].map(food_protein)).groupby("foodex2_code")["p"].median()
    prot = {c: (est[c] if c in est.index and est[c] > 0 else cat[c][1]) for c in names.index}
    rows = []
    for fid, d in t.groupby("food_id"):
        e = {}
        for code, g in zip(d["foodex2_code"], d["raw_equivalent_g"]):
            c, _ = cat[code]
            e[c or "_unprofiled"] = e.get(c or "_unprofiled", 0.0) + g * prot[code] / 100.0
        tot = sum(e.values())
        prof = {c: v for c, v in e.items() if c in common.index}
        ptot = sum(prof.values())
        row = {"food_id": int(fid), "protein_profiled_share": ptot / tot if tot > 0 else np.nan,
               "protein_main_source": max(e, key=e.get) if tot > 0 else None}
        if tot > 0 and ptot / tot >= min_profiled:
            vec = sum(common.loc[c] * v for c, v in prof.items()) / ptot
            row.update({f"protfrac_{k}": round(float(vec[k]), 4) for k in common.columns})
        rows.append(row)
    return pd.DataFrame(rows)
