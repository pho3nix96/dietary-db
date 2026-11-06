"""Compare people's diets using the food reference.

Steps
1. match_records: link every record line to a leaf of the food reference: by food id
   (only when the records use the reference's own ids), then FoodEx2 code, then name.
   Unique products that were placed in the food space keep their placed GIFT group.
2. fill_nutrients: lines without nutrient values (e.g. name-only products) take them
   from the reference food they were matched or placed next to.
3. diet_profiles: per person, the share of intake (energy or grams) on each leaf.
4. tree_distance: distance between two people = earth mover's distance on the food tree
   GIFT group -> GIFT subgroup -> food. Moving intake between two similar foods (whole
   milk to semi-skimmed milk) is cheap; moving it to a different group is expensive.
   This is the weighted UniFrac idea from microbiome work, applied to foods.
5. nutrient_distance: Aitchison distance between macro-nutrient profiles.
6. person_distance: weighted mean of the two views.

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

# record column (GIFT names) -> food-table column (per 100 g)
RECORD_TO_TABLE = {"ENERGY_kcal": "ENERC_kcal", "PROTEIN_g": "PROT_g", "FAT_g": "FAT_g",
                   "CARBOH_g": "CHO_g", "SUGAR_g": "SUGAR_g", "FIBTG_g": "FIBT_g"}


# -- 1. matching ---------------------------------------------------------------------------

def base_term(code) -> str:
    """FoodEx2 base term of one code: 'A03VA#F04.A01QF$F04.A0F0N' -> 'A03VA' (facets dropped)."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return ""
    return str(code).split("#")[0].strip()


def _split_codes(s) -> list[str]:
    """All codes in a cell, facets kept: 'A01|A02#F01.X' -> ['A01', 'A02#F01.X']."""
    return [c for c in re.split(r"[|,;\s]+", str(s)) if c and c != "nan"]


def match_records(records: pd.DataFrame, foods: pd.DataFrame, placements: pd.DataFrame | None = None,
                  min_name_sim: float = 0.5, id_col: str | None = "food_id", code_col: str = "FOODEX2_INGR_CODE",
                  name_col: str = "INGREDIENT_ENG", unique_col: str = "unique_food_id") -> pd.DataFrame:
    """Add `leaf`, `match`, `gift_group`, `gift_subgroup` to the record lines.

    Order of evidence for each line:
      id              the line's `id_col` is a food id of this reference table. Pass
                      id_col=None when the records' ids come from another database.
      placed          a unique product placed in the food space (placements table)
      code            its FoodEx2 code (with facets) or base term points to one reference food
      code+name       several reference foods share the base term; the most similar name wins
      code_ambiguous  as above but the line has no name; the food coded with the bare base term
                      (no facets) is taken if there is one
      name            no code match; the most similar reference name (>= min_name_sim)
      unmatched       none of the above; leaf is NaN and the line is left out of the profiles

    foods:       reference table with food_id, name, gift_group, gift_subgroup, foodex2_code (optional)
    placements:  placed unique products (unique_food_id, gift_group, gift_subgroup)
    """
    foods = foods.reset_index(drop=True)
    by_id = foods.set_index("food_id")
    full_to_rows: dict[str, list[int]] = {}
    base_to_rows: dict[str, list[int]] = {}
    bare_rows: dict[str, set[int]] = {}  # foods coded with the bare base term
    if "foodex2_code" in foods:
        for j, cell in enumerate(foods["foodex2_code"]):
            for c in _split_codes(cell):
                b = base_term(c)
                full_to_rows.setdefault(c, []).append(j)
                if j not in base_to_rows.setdefault(b, []):
                    base_to_rows[b].append(j)
                if c == b:
                    bare_rows.setdefault(b, set()).add(j)
    index = NameIndex(foods["name"].fillna(""))
    plc = placements.set_index(unique_col) if placements is not None else None
    use_ids = id_col is not None and id_col in records

    def by_code_and_name(code: str, name: str):
        b = base_term(code)
        if code != b:  # the record carries facets: an exact code + facets match decides
            rows = full_to_rows.get(code, [])
            if len(rows) == 1:
                return rows[0], "code"
        rows = base_to_rows.get(b, [])
        if len(rows) == 1:
            return rows[0], "code"
        if rows:
            if name:
                sims = index.similarity(name)
                return rows[int(np.argmax(sims[rows]))], "code+name"
            bare = [j for j in rows if j in bare_rows.get(b, set())]
            return (bare or rows)[0], "code_ambiguous"
        if name:
            sims = index.similarity(name)
            j = int(np.argmax(sims))
            if sims[j] >= min_name_sim:
                return j, "name"
        return None, "unmatched"

    cache: dict[tuple[str, str], tuple] = {}
    leaf, how, grp, sub = [], [], [], []
    for _, r in records.iterrows():
        fid = r.get(id_col) if use_ids else None
        uid = r.get(unique_col) if unique_col in records else None
        g = s = None
        if use_ids and pd.notna(fid) and fid in by_id.index:
            leaf.append(f"f{int(fid)}"); how.append("id")
            g, s = by_id.at[fid, "gift_group"], by_id.at[fid, "gift_subgroup"]
        elif plc is not None and pd.notna(uid) and uid in plc.index:
            leaf.append(f"u{uid}"); how.append("placed")
            g, s = plc.at[uid, "gift_group"], plc.at[uid, "gift_subgroup"]
        else:
            code = r.get(code_col)
            code = "" if code is None or (isinstance(code, float) and np.isnan(code)) else str(code).strip()
            name = r.get(name_col)
            name = name.strip() if isinstance(name, str) else ""
            if (code, name) not in cache:
                cache[(code, name)] = by_code_and_name(code, name)
            j, kind = cache[(code, name)]
            if j is None:
                leaf.append(np.nan); how.append("unmatched")
            else:
                leaf.append(f"f{int(foods.at[j, 'food_id'])}"); how.append(kind)
                g, s = foods.at[j, "gift_group"], foods.at[j, "gift_subgroup"]
        grp.append(g); sub.append(s)
    out = records.copy()
    out["leaf"], out["match"], out["gift_group"], out["gift_subgroup"] = leaf, how, grp, sub
    return out


def match_report(matched: pd.DataFrame, weight: str | None = None, foods: pd.DataFrame | None = None,
                 name_col: str = "INGREDIENT_ENG") -> dict:
    """Share of record lines (and of the weight, e.g. kcal) per match type.

    With `foods`, also checks the id matches: the share whose record name equals the
    reference name. A low share means the records' ids belong to another database."""
    n = matched["match"].value_counts()
    rep = {"lines": int(len(matched)), "by_match": {k: int(v) for k, v in n.items()}}
    if weight and weight in matched:
        w = matched.groupby("match")[weight].sum()
        rep["weight_share_by_match"] = {k: round(float(v / w.sum()), 4) for k, v in w.items()}
    if foods is not None and name_col in matched and (matched["match"] == "id").any():
        ids = matched[matched["match"] == "id"]
        ref = foods.set_index(foods["food_id"].map(lambda i: f"f{int(i)}"))["name"]
        same = ids[name_col].astype(str).str.strip().str.lower() == ids["leaf"].map(ref).astype(str).str.strip().str.lower()
        rep["id_matches_with_same_name"] = round(float(same.mean()), 4)
        if same.mean() < 0.5:
            print("Warning: most id matches have a different name in the reference; "
                  "the records' ids may come from another database (use id_col=None).")
    if "nutrients_from" in matched:
        rep["nutrients_from"] = {k: int(v) for k, v in matched["nutrients_from"].value_counts().items()}
    return rep


def fill_nutrients(matched: pd.DataFrame, foods: pd.DataFrame, placements: pd.DataFrame | None = None,
                   amount_col: str = "FOOD_AMOUNT_CONS", columns: dict[str, str] = RECORD_TO_TABLE) -> pd.DataFrame:
    """Fill a line's missing nutrient values from its reference food: amount / 100 x value per 100 g.

    The reference food is the matched leaf; a placed unique product uses the first of its
    `nearest_foods` in the placements table. Adds `nutrients_from`: record (all values were
    in the record), reference (some were filled) or missing (could not be filled)."""
    out = matched.copy()
    ref = foods.copy()
    if "ENERC_kcal" not in ref and "ENERC_kJ" in ref:
        ref["ENERC_kcal"] = pd.to_numeric(ref["ENERC_kJ"], errors="coerce") / 4.184
    per100 = ref.set_index("food_id")
    near = nearest_food_of_products(placements, ref)
    for c in columns:
        if c not in out:
            out[c] = np.nan
    cols = list(columns)
    missing = out[cols].isna().any(axis=1).to_numpy()
    src = np.where(missing, "missing", "record").astype(object)
    pos = {c: out.columns.get_loc(c) for c in cols}
    for i in np.flatnonzero(missing):
        leaf = out["leaf"].iat[i]
        fid = int(leaf[1:]) if isinstance(leaf, str) and leaf.startswith("f") else near.get(leaf)
        amt = pd.to_numeric(out[amount_col].iat[i], errors="coerce") if amount_col in out else np.nan
        if fid is None or fid not in per100.index or pd.isna(amt):
            continue
        for rc, tc in columns.items():
            if pd.isna(out.iat[i, pos[rc]]) and tc in per100.columns:
                v = pd.to_numeric(per100.at[fid, tc], errors="coerce")
                if pd.notna(v):
                    out.iat[i, pos[rc]] = float(v) * float(amt) / 100.0
        src[i] = "reference" if out.iloc[i][cols].notna().all() else "missing"
    out["nutrients_from"] = src
    return out


# -- 2. profiles -----------------------------------------------------------------------------

def _label(x) -> str:
    return "unassigned" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


class _Tree:
    """Rooted tree as parent/length arrays; node 0 is the root."""

    def __init__(self):
        self.ids, self.parent, self.length = {"root": 0}, [-1], [0.0]

    def add(self, key, parent_key, length: float) -> int:
        if key not in self.ids:
            self.ids[key] = len(self.parent)
            self.parent.append(self.ids[parent_key])
            self.length.append(float(length))
        return self.ids[key]

    def result(self, leaf_keys):
        leaf_index = {k: self.ids[("l", k)] for k in leaf_keys}
        return leaf_index, np.array(self.parent), np.array(self.length), len(self.parent)


def _leaves(foods: pd.DataFrame, matched: pd.DataFrame) -> dict:
    """leaf key -> (GIFT group, subgroup): every reference food plus matched unique products."""
    leaves = {f"f{int(i)}": (_label(g), _label(s))
              for i, g, s in zip(foods["food_id"], foods["gift_group"], foods["gift_subgroup"])}
    extra = matched[matched["leaf"].notna()].drop_duplicates("leaf")
    for k, g, s in zip(extra["leaf"], extra["gift_group"], extra["gift_subgroup"]):
        leaves.setdefault(k, (_label(g), _label(s)))
    return leaves


def food_tree(foods: pd.DataFrame, matched: pd.DataFrame, lengths=DEFAULT_LENGTHS):
    """Tree GIFT group -> subgroup -> leaf, with leaves from the reference and from
    matched unique products. Returns (leaf_index, parent, length, n_nodes).

    Nodes are numbered with the root as 0. `parent[i]` is the parent of node i.
    Missing groups and subgroups are labelled "unassigned"."""
    lg, ls, lf = lengths
    leaves = _leaves(foods, matched)
    t = _Tree()
    for k, (g, s) in leaves.items():
        t.add(("g", g), "root", lg)
        t.add(("s", g, s), ("g", g), ls)
        t.add(("l", k), ("s", g, s), lf)
    return t.result(leaves)


def flat_tree(foods: pd.DataFrame, matched: pd.DataFrame):
    """Every food directly under the root: no similarity between different foods (a baseline).
    The tree distance is then half the summed absolute difference in food shares."""
    leaves = _leaves(foods, matched)
    t = _Tree()
    for k in leaves:
        t.add(("l", k), "root", 1.0)
    return t.result(leaves)


def nearest_food_of_products(placements: pd.DataFrame | None, foods: pd.DataFrame) -> dict:
    """Leaf of each placed unique product ("u<id>") -> food_id of the first of its nearest foods."""
    if placements is None or "nearest_foods" not in placements:
        return {}
    key = foods["name"].astype(str).str.strip().str.lower()
    by_name = pd.Series(foods["food_id"].to_numpy(), index=key).groupby(level=0).first()
    out = {}
    for uid, s in zip(placements["unique_food_id"], placements["nearest_foods"]):
        first = str(s).split(" | ")[0].strip().lower()
        if first in by_name.index:
            out[f"u{uid}"] = int(by_name[first])
    return out


def _treelet_parts(info: dict):
    """(food ids, n leaves, monotonic node heights, children of each merge, root) of one trees.json entry."""
    ids = [int(i) for i in info["food_ids"]]
    Z = np.asarray(info["linkage"], dtype=float).reshape(-1, 4)
    n = len(ids)
    H = np.zeros(n + len(Z))
    kids = {}
    for k, (a, b, h, _) in enumerate(Z):  # children are always merged before their parent
        H[n + k] = max(h, H[int(a)], H[int(b)])  # treelet merges are not monotonic
        kids[n + k] = (int(a), int(b))
    return ids, n, H, kids, n + len(Z) - 1


def _hang(t: _Tree, anchor, parts, keep: set, depth: float, tag: tuple, leaf_parent: dict) -> None:
    """Hang a treelet tree, pruned to the leaf positions in `keep`, under node `anchor`.

    Nodes left with one child are dropped, the pruned root becomes `anchor`, and heights are
    scaled so every kept leaf sits `depth` below the anchor. Records each kept leaf's
    (parent key, edge length) in `leaf_parent`; leaves are added to the tree later."""
    ids, n, H, kids, top = parts
    rep = {i: (("leaf", i) if i in keep else None) for i in range(n)}
    children = {}
    for v in range(n, top + 1):
        c = [rep[x] for x in kids[v] if rep[x] is not None]
        rep[v] = None if not c else (c[0] if len(c) == 1 else ("node", v))
        if len(c) == 2:
            children[v] = c
    r = rep.get(top)
    if r is None:
        return
    if r[0] == "leaf" or H[r[1]] <= 0:  # one food, or identical foods: all at full depth
        for i in keep:
            leaf_parent[f"f{ids[i]}"] = (anchor, depth)
        return
    scale = depth / H[r[1]]
    stack = [(r[1], anchor)]
    while stack:  # iterative: large groups give deep trees
        v, key_v = stack.pop()
        for c in children[v]:
            if c[0] == "leaf":
                leaf_parent[f"f{ids[c[1]]}"] = (key_v, H[v] * scale)
            else:
                key_c = (*tag, c[1])
                t.add(key_c, key_v, (H[v] - H[c[1]]) * scale)
                stack.append((c[1], key_c))


def composition_tree(foods: pd.DataFrame, matched: pd.DataFrame, trees: dict,
                     placements: pd.DataFrame | None = None, lengths=DEFAULT_LENGTHS, within: str = "subgroup"):
    """Food tree that uses the reference's composition similarity inside GIFT groups.

    `trees` is the trees.json written by scripts/within_gift_groups.py for the same food table:
    per GIFT group, the food ids and the treelet linkage on composition and fraction similarity.
    Branch lengths come from the merge distances (1 - similarity), made monotonic and scaled so
    the depths match food_tree's `lengths` (group, subgroup, food); so does the largest distance.

    within="subgroup": GIFT group -> subgroup -> the group's composition tree pruned to that
        subgroup's foods. Foods of one subgroup are as close as their composition allows;
        foods of different subgroups are as far apart as in food_tree.
    within="group": GIFT group -> composition tree -> food; the subgroups are not used.

    Groups without a tree (fewer than 8 foods) and foods missing from their group's tree keep
    the subgroup structure of food_tree. A placed unique product hangs next to its nearest
    reference food (the first of `nearest_foods` in the placements table), at the same height.
    Returns (leaf_index, parent, length, n_nodes) like food_tree."""
    if within not in ("subgroup", "group"):
        raise ValueError("within must be 'subgroup' or 'group'")
    lg, ls, lf = lengths
    leaves = _leaves(foods, matched)
    t = _Tree()
    leaf_parent: dict = {}
    for g in sorted({g for g, _ in leaves.values()}):
        t.add(("g", g), "root", lg)
        info = trees.get(g)
        if not info or len(info.get("food_ids", [])) < 2 or not info.get("linkage"):
            continue
        parts = _treelet_parts(info)
        ids = parts[0]
        if within == "group":
            keep = {i for i, fid in enumerate(ids) if f"f{fid}" in leaves}
            _hang(t, ("g", g), parts, keep, ls + lf, ("t", g), leaf_parent)
            continue
        by_sub: dict = {}
        for i, fid in enumerate(ids):
            if f"f{fid}" in leaves:
                by_sub.setdefault(leaves[f"f{fid}"][1], set()).add(i)
        for s, keep in by_sub.items():
            t.add(("s", g, s), ("g", g), ls)
            _hang(t, ("s", g, s), parts, keep, lf, ("t", g, s), leaf_parent)
    near = nearest_food_of_products(placements, foods)
    for k, (g, s) in leaves.items():
        if k not in leaf_parent and k.startswith("u") and f"f{near.get(k, -1)}" in leaf_parent:
            leaf_parent[k] = leaf_parent[f"f{near[k]}"]  # sibling of its nearest food, same height
        if k in leaf_parent:
            t.add(("l", k), *leaf_parent[k])
        else:
            t.add(("s", g, s), ("g", g), ls)
            t.add(("l", k), ("s", g, s), lf)
    return t.result(leaves)


def diet_profiles(matched: pd.DataFrame, leaf_index: dict, weight: str = "ENERGY_kcal", subject: str = "SUBJECT",
                  dish_shares: pd.DataFrame | None = None) -> pd.DataFrame:
    """Share of each person's intake on each leaf (rows sum to 1; people with no matched intake are dropped).

    weight:       column to sum, e.g. ENERGY_kcal (default) or FOOD_AMOUNT_CONS (grams)
    dish_shares:  optional table food_id, leaf, share. A record line whose food_id appears
                  here has its weight spread over those leaves (e.g. the ingredient groups
                  of a composite dish), instead of counting as one food.
    """
    m = matched[matched["leaf"].notna() & (pd.to_numeric(matched[weight], errors="coerce") > 0)]
    rows = []
    split = {}
    if dish_shares is not None:
        unknown = ~dish_shares["leaf"].isin(list(leaf_index))
        if unknown.any():
            print(f"Warning: {int(unknown.sum())} dish-share rows point to leaves not in the food tree "
                  f"and are ignored: {sorted(dish_shares.loc[unknown, 'leaf'].astype(str).unique())[:5]}")
        ds = dish_shares[~unknown]
        for fid, t in ds.groupby("food_id"):
            split[f"f{int(fid)}"] = list(zip(t["leaf"], t["share"] / t["share"].sum()))
    for s, leaf, w in zip(m[subject], m["leaf"], m[weight]):
        if leaf in split:
            rows += [(s, tgt, w * sh) for tgt, sh in split[leaf]]
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

    starch_g is carbohydrate minus sugars, clipped at 0. Lines missing any part are left
    out (run fill_nutrients first to fill them). People without nutrient data get
    distance 1 to everyone else. `people` fixes the row order."""
    m = matched.copy()
    m["starch_g"] = (pd.to_numeric(m["CARBOH_g"], errors="coerce") - pd.to_numeric(m["SUGAR_g"], errors="coerce")).clip(lower=0)
    m = m.dropna(subset=list(parts))
    T = m.groupby(subject)[list(parts)].sum()
    if people is not None:
        T = T.reindex(people)
    S = T.div(T.sum(axis=1), axis=0)
    D = fractions.aitchison_distance(S)
    hi = np.nanmax(D) if np.isfinite(D).any() else 0.0
    D = np.nan_to_num(D / hi if hi > 0 else D, nan=1.0)
    np.fill_diagonal(D, 0.0)
    return D


def person_distance(views: dict[str, np.ndarray], weights: dict[str, float] | None = None) -> np.ndarray:
    """Weighted mean of the person-to-person distance matrices (equal weights by default)."""
    w = weights or {k: 1.0 for k in views}
    total = sum(w.get(k, 0.0) for k in views)
    if total <= 0:
        raise ValueError("At least one view needs a positive weight.")
    return sum(w.get(k, 0.0) * D for k, D in views.items()) / total
