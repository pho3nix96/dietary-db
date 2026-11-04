"""Compare people's diets using the food reference.

Steps
1. match_records: link every record line to a leaf of the food reference
   (by food id, then FoodEx2 code, then name; unique products keep their placed GIFT group).
2. diet_profiles: per person, the share of intake (energy or grams) on each leaf.
3. tree_distance: distance between two people = earth mover's distance on the food tree
   GIFT group -> GIFT subgroup -> food. Moving intake between two similar foods (whole
   milk to semi-skimmed milk) is cheap; moving it to a different group is expensive.
   This is the weighted UniFrac idea from microbiome work, applied to foods.
4. nutrient_distance: Aitchison distance between macro-nutrient profiles.
5. person_distance: weighted mean of the two views.

Profiles are shares, so a person who under-reports everything by the same factor keeps
the same profile. Composite dishes can be spread over ingredient groups by passing
`dish_shares` (see `diet_profiles`); without it a dish counts as one food of its group.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from . import fractions
from .placement import NameIndex

DEFAULT_LENGTHS = (1.0, 0.5, 0.25)  # edge lengths: root->group, group->subgroup, subgroup->food


# -- 1. matching ---------------------------------------------------------------------------

def _codes(s) -> list[str]:
    return [c for c in re.split(r"[|,;\s]+", str(s)) if c and c != "nan"]


def match_records(records: pd.DataFrame, foods: pd.DataFrame, placements: pd.DataFrame | None = None,
                  min_name_sim: float = 0.5, id_col: str = "food_id", code_col: str = "FOODEX2_INGR_CODE",
                  name_col: str = "INGREDIENT_ENG", unique_col: str = "unique_food_id") -> pd.DataFrame:
    """Add `leaf`, `match`, `gift_group`, `gift_subgroup` to the record lines.

    match is one of: id, code, name, placed (unique product placed in the food space),
    unmatched. Unmatched lines get leaf NaN and are left out of the profiles, so check
    the report from `match_report`.

    foods:       reference table with food_id, name, gift_group, gift_subgroup, foodex2_code (optional)
    placements:  placed unique products (unique_food_id, gift_group, gift_subgroup)
    """
    foods = foods.reset_index(drop=True)
    by_id = foods.set_index("food_id")
    code_to_ids: dict[str, list] = {}
    if "foodex2_code" in foods:
        for fid, c in zip(foods["food_id"], foods["foodex2_code"]):
            for code in _codes(c):
                code_to_ids.setdefault(code, []).append(fid)
    index = NameIndex(foods["name"].fillna(""))
    plc = placements.set_index(unique_col) if placements is not None else None

    leaf, how, grp, sub = [], [], [], []
    for _, r in records.iterrows():
        fid = r.get(id_col)
        uid = r.get(unique_col) if unique_col in records else None
        g = s = None
        if pd.notna(fid) and fid in by_id.index:
            leaf.append(f"f{int(fid)}"); how.append("id")
            g, s = by_id.at[fid, "gift_group"], by_id.at[fid, "gift_subgroup"]
        elif plc is not None and pd.notna(uid) and uid in plc.index:
            leaf.append(f"u{uid}"); how.append("placed")
            g, s = plc.at[uid, "gift_group"], plc.at[uid, "gift_subgroup"]
        else:
            ids = code_to_ids.get(str(r.get(code_col)), [])
            name = r.get(name_col)
            pick = None
            if len(ids) == 1:
                pick, kind = ids[0], "code"
            elif isinstance(name, str) and name:
                sims = index.similarity(name)
                if len(ids) > 1:  # several foods share the code: choose among them by name
                    cand = foods.index[foods["food_id"].isin(ids)].to_numpy()
                    j = cand[int(np.argmax(sims[cand]))]
                else:
                    j = int(np.argmax(sims))
                if sims[j] >= min_name_sim:
                    pick, kind = foods.at[j, "food_id"], "name"
            if pick is None:
                leaf.append(np.nan); how.append("unmatched")
            else:
                leaf.append(f"f{int(pick)}"); how.append(kind)
                g, s = by_id.at[pick, "gift_group"], by_id.at[pick, "gift_subgroup"]
        grp.append(g); sub.append(s)
    out = records.copy()
    out["leaf"], out["match"], out["gift_group"], out["gift_subgroup"] = leaf, how, grp, sub
    return out


def match_report(matched: pd.DataFrame, weight: str | None = None) -> dict:
    """Share of record lines (and of the weight, e.g. kcal) per match type."""
    n = matched["match"].value_counts()
    rep = {"lines": int(len(matched)), "by_match": {k: int(v) for k, v in n.items()}}
    if weight and weight in matched:
        w = matched.groupby("match")[weight].sum()
        rep["weight_share_by_match"] = {k: round(float(v / w.sum()), 4) for k, v in w.items()}
    return rep


# -- 2. profiles -----------------------------------------------------------------------------

def food_tree(foods: pd.DataFrame, matched: pd.DataFrame, lengths=DEFAULT_LENGTHS):
    """Tree GIFT group -> subgroup -> leaf, with leaves from the reference and from
    matched unique products. Returns (leaves, parent, length, depth_of_leaf).

    Nodes are numbered with the root as 0. `parent[i]` is the parent of node i."""
    lg, ls, lf = lengths
    leaves = {f"f{int(i)}": (g, s) for i, g, s in zip(foods["food_id"], foods["gift_group"], foods["gift_subgroup"])}
    extra = matched[matched["leaf"].notna()].drop_duplicates("leaf")
    for k, g, s in zip(extra["leaf"], extra["gift_group"], extra["gift_subgroup"]):
        leaves.setdefault(k, (g, s))
    nodes = {"root": 0}
    parent, length = [-1], [0.0]

    def add(key, par, ln):
        if key not in nodes:
            nodes[key] = len(parent)
            parent.append(nodes[par]); length.append(ln)
        return key

    for k, (g, s) in leaves.items():
        gk = add(("g", g), "root", lg)
        sk = add(("s", g, s), gk, ls)
        add(("l", k), sk, lf)
    leaf_index = {k: nodes[("l", k)] for k in leaves}
    return leaf_index, np.array(parent), np.array(length), len(nodes)


def diet_profiles(matched: pd.DataFrame, leaf_index: dict, weight: str = "ENERGY_kcal", subject: str = "SUBJECT",
                  dish_shares: pd.DataFrame | None = None) -> pd.DataFrame:
    """Share of each person's intake on each leaf (rows sum to 1; people with no matched intake are dropped).

    weight:       column to sum, e.g. ENERGY_kcal (default) or FOOD_AMOUNT_CONS (grams)
    dish_shares:  optional table food_id, leaf, share. A record line whose food_id appears
                  here has its weight spread over those leaves (e.g. the ingredient groups
                  of a composite dish), instead of counting as one food.
    """
    m = matched[matched["leaf"].notna() & (matched[weight] > 0)]
    rows = []
    split = {}
    if dish_shares is not None:
        for fid, t in dish_shares.groupby("food_id"):
            split[f"f{int(fid)}"] = list(zip(t["leaf"], t["share"] / t["share"].sum()))
    for s, leaf, w in zip(m[subject], m["leaf"], m[weight]):
        if leaf in split:
            rows += [(s, tgt, w * sh) for tgt, sh in split[leaf] if tgt in leaf_index]
        else:
            rows.append((s, leaf, w))
    t = pd.DataFrame(rows, columns=["subject", "leaf", "w"])
    P = t.pivot_table(index="subject", columns="leaf", values="w", aggfunc="sum", fill_value=0.0)
    P = P.reindex(columns=list(leaf_index), fill_value=0.0)
    return P.div(P.sum(axis=1), axis=0)


# -- 3. distances ------------------------------------------------------------------------------

def tree_distance(P: pd.DataFrame, leaf_index: dict, parent: np.ndarray, length: np.ndarray) -> np.ndarray:
    """Earth mover's distance between profiles on the food tree, scaled to [0, 1].

    D(p, q) = sum over tree edges of length * |mass below the edge in p - mass below it in q|.
    Scaled by twice the longest root-to-leaf path (the largest possible distance)."""
    order = list(leaf_index)
    rows, cols, longest = [], [], 0.0
    for j, k in enumerate(order):
        v, path = leaf_index[k], 0.0
        while v != 0:  # every node on the way up has this leaf below it
            rows.append(v); cols.append(j)
            path += length[v]; v = parent[v]
        longest = max(longest, path)
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(parent), len(order)))
    M = (A @ P.reindex(columns=order, fill_value=0.0).to_numpy().T).T  # people x nodes: mass below each node
    D = cdist(M * length[None, :], M * length[None, :], "cityblock")
    return D / (2 * longest)


def nutrient_distance(matched: pd.DataFrame, subject: str = "SUBJECT", people=None,
                      parts=("PROTEIN_g", "FAT_g", "SUGAR_g", "starch_g", "FIBTG_g")) -> np.ndarray:
    """Aitchison distance between people's macro-nutrient profiles (shares of the parts), scaled to [0, 1].

    starch_g is carbohydrate minus sugars, clipped at 0. `people` fixes the row order."""
    m = matched.copy()
    m["starch_g"] = (m["CARBOH_g"] - m["SUGAR_g"]).clip(lower=0)
    T = m.groupby(subject)[list(parts)].sum()
    if people is not None:
        T = T.reindex(people)
    S = T.div(T.sum(axis=1), axis=0)
    D = fractions.aitchison_distance(S)
    hi = np.nanmax(D)
    return np.nan_to_num(D / hi if hi > 0 else D, nan=1.0)


def person_distance(views: dict[str, np.ndarray], weights: dict[str, float] | None = None) -> np.ndarray:
    """Weighted mean of the person-to-person distance matrices (equal weights by default)."""
    w = weights or {k: 1.0 for k in views}
    total = sum(w.get(k, 0.0) for k in views)
    if total <= 0:
        raise ValueError("At least one view needs a positive weight.")
    return sum(w.get(k, 0.0) * D for k, D in views.items()) / total
