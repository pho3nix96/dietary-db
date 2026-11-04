"""Load and validate the food table."""

from __future__ import annotations

import pandas as pd


def load_foods(path: str, cols: dict) -> pd.DataFrame:
    """Read the food table and check that the configured columns exist.

    `cols` comes from config: id, name, description, foodex2, gift_group,
    gift_subgroup, and `nutrients` (a list of nutrient column names).
    """
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep)
    required = [cols["id"], cols["name"], cols["gift_group"], cols["gift_subgroup"], *cols["nutrients"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from {path}: {missing}")
    if df[cols["id"]].duplicated().any():
        dups = df.loc[df[cols["id"]].duplicated(), cols["id"]].tolist()[:10]
        raise ValueError(f"Duplicate food ids, first ones: {dups}")
    return df.reset_index(drop=True)


def food_texts(df: pd.DataFrame, cols: dict) -> list[str]:
    """Name plus description, the input for semantic embeddings."""
    name = df[cols["name"]].fillna("").astype(str)
    desc_col = cols.get("description")
    if desc_col and desc_col in df.columns:
        desc = df[desc_col].fillna("").astype(str)
        return [f"{n}. {d}".strip(". ") for n, d in zip(name, desc)]
    return name.tolist()
