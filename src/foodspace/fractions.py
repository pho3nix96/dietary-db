"""The "what kind" layer: nutrient fractions as profiles within their parent.

Each block describes one macronutrient by its parts, for example

    fat:      SFA, MUFA, PUFA            (parent: total fat)
    sugars:   glucose, fructose, sucrose, lactose
    protein:  casein, whey, ...          (from the curated table)

Within a block the parts are turned into shares (they sum to 1), so a food
and a concentrated form of it (milk vs milk powder) get the same profile.
Shares are compositional data, so they are compared with the Aitchison
distance: zeros are replaced by a small value, the centred log-ratio (CLR)
is taken, and Euclidean distance is computed on the CLR values.

Each block is scaled to [0, 1] and the blocks are averaged with equal weight,
so a block with 20 fatty acids does not outweigh a block with 4 sugars.
A block only counts for a pair of foods when both foods have it: a food
with almost no fat has no meaningful fat profile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def block_shares(
    df: pd.DataFrame,
    parts: list[str],
    parent: str | None = None,
    min_parent: float = 0.5,
    remainder: bool = False,
) -> pd.DataFrame:
    """Shares of each part within one block; NaN rows where the block is undefined.

    parent:     column with the total (e.g. fat_g). If given, foods with less than
                `min_parent` (g/100 g) get NaN for the whole block.
    remainder:  add an "other" share = parent - sum(parts), for when the parts
                don't cover the whole total (e.g. only some fatty acids measured).
    """
    X = df[parts].apply(pd.to_numeric, errors="coerce").astype(float).clip(lower=0)
    if remainder:
        if parent is None:
            raise ValueError("remainder needs a parent column")
        X["other"] = (pd.to_numeric(df[parent], errors="coerce") - X.sum(axis=1, min_count=1)).clip(lower=0)
    total = X.sum(axis=1, min_count=len(X.columns))  # NaN if any part is missing
    S = X.div(total, axis=0)
    S[total <= 0] = np.nan
    if parent is not None:
        low = pd.to_numeric(df[parent], errors="coerce").fillna(0) < min_parent
        S[low] = np.nan
    return S


def clr(S: pd.DataFrame, delta: float = 1e-3) -> pd.DataFrame:
    """Centred log-ratio after multiplicative zero replacement.

    Zeros become `delta` and the other shares are shrunk so each row still
    sums to 1 (Martín-Fernández et al. 2003). Rows with NaN stay NaN.
    """
    A = S.to_numpy(dtype=float)
    out = np.full_like(A, np.nan)
    ok = ~np.isnan(A).any(axis=1)
    R = A[ok]
    zeros = R <= 0
    nz = zeros.sum(axis=1, keepdims=True)
    R = np.where(zeros, delta, R * (1 - nz * delta))
    L = np.log(R)
    out[ok] = L - L.mean(axis=1, keepdims=True)
    return pd.DataFrame(out, index=S.index, columns=S.columns)


def aitchison_distance(S: pd.DataFrame) -> np.ndarray:
    """Pairwise Aitchison distance; NaN where either food lacks the block."""
    C = clr(S).to_numpy()
    acc = np.zeros((C.shape[0], C.shape[0]))
    for j in range(C.shape[1]):  # one part at a time: n x n memory instead of n x n x p
        d = C[:, j][:, None] - C[:, j][None, :]
        acc += d * d
    return np.sqrt(acc)


def fraction_distance(blocks: dict[str, pd.DataFrame], fill: float = 1.0, min_shared: int = 1) -> tuple[np.ndarray, dict]:
    """Combine block distances with equal weight.

    Each block's distances are scaled to [0, 1] by that block's largest value.
    For each pair, the result is the mean over blocks both foods have. Pairs
    that share fewer than `min_shared` blocks get `fill` (1 = treated as unlike);
    raise `min_shared` so pairs compared on a single block are not mixed with
    pairs compared on many.

    Returns the distance matrix and a coverage report per block.
    """
    n = next(iter(blocks.values())).shape[0]
    total = np.zeros((n, n))
    count = np.zeros((n, n))
    report = {}
    for name, S in blocks.items():
        D = aitchison_distance(S)
        hi = np.nanmax(D) if np.isfinite(D).any() else 0
        if hi > 0:
            D = D / hi
        have = np.isfinite(D)
        total += np.where(have, D, 0)
        count += have
        report[name] = {
            "foods_with_block": int((~S.isna().any(axis=1)).sum()),
            "parts": list(S.columns),
        }
    with np.errstate(invalid="ignore"):
        F = np.where(count >= max(min_shared, 1), total / np.maximum(count, 1), fill)
    np.fill_diagonal(F, 0)
    report["pairs_without_shared_block"] = int(((count < max(min_shared, 1)) & ~np.eye(n, dtype=bool)).sum() // 2)
    return F, report


def curated_block(
    df: pd.DataFrame,
    table: pd.DataFrame,
    key_col: str,
    scheme: str,
) -> pd.DataFrame:
    """Shares from the curated table (long format) for one fraction scheme.

    The food table needs a column `key_col` naming each food's curated category
    (e.g. "cheese_rennet"). Foods without a category, or categories without
    sourced values for this scheme, get NaN, so the block is skipped for them.

    scheme="common" compares foods across schemes: each fraction is summed into
    its `common_class` (solubility-based classes such as water_soluble,
    salt_soluble, alcohol_soluble, insoluble_storage, casein,
    structural_insoluble, mixed), so wheat, soybean, meat and milk land on one
    set of axes.
    """
    t = table[table["share"].notna()]
    if scheme == "common":
        wide = t.pivot_table(index="category", columns="common_class", values="share", aggfunc="sum").fillna(0.0)
    else:
        t = t[t["scheme"] == scheme]
        wide = t.pivot_table(index="category", columns="fraction", values="share", aggfunc="first")
    # rows must cover the whole protein; normalise in case rounding leaves 0.99 or 1.01
    wide = wide.div(wide.sum(axis=1), axis=0)
    keys = df[key_col] if key_col in df.columns else pd.Series([None] * len(df))
    out = wide.reindex(keys.to_numpy())
    out.index = df.index
    return out if not out.empty else pd.DataFrame(index=df.index)


def build_blocks(df: pd.DataFrame, cfg: dict, curated: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """All blocks named in the config, skipping blocks whose columns are absent."""
    blocks = {}
    for name, b in cfg.get("blocks", {}).items():
        if b.get("source") == "curated":
            if curated is None:
                continue
            S = curated_block(df, curated, b["key_col"], b["scheme"])
            if b.get("parent") and b["parent"] in df.columns:
                low = pd.to_numeric(df[b["parent"]], errors="coerce").fillna(0) < b.get("min_parent", 0.5)
                S[low.to_numpy()] = np.nan
        else:
            missing = [c for c in b["parts"] + ([b["parent"]] if b.get("parent") else []) if c not in df.columns]
            if missing:
                print(f"Fraction block '{name}' skipped; columns not in the food table: {missing}")
                continue
            S = block_shares(df, b["parts"], b.get("parent"), b.get("min_parent", 0.5), b.get("remainder", False))
        if S.notna().all(axis=1).sum() >= 2:
            blocks[name] = S
        else:
            print(f"Fraction block '{name}' skipped; fewer than two foods have it.")
    return blocks
