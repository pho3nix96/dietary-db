"""Place a new food in the food space.

Names do not shape the space (clusters and layouts use composition and
fractions only). They are used here, to put a new food in the right place:

1. GIFT group and subgroup: weighted vote of the k foods with the most similar
   names (TF-IDF on word and character n-grams), unless the caller gives them.
2. Missing values: anything the new food lacks that its name neighbours in the
   same group have (protein-fraction profile, fatty-acid or sugar detail) is
   filled with their similarity-weighted mean and flagged as imputed.
3. Position and cluster: the new food is compared with the foods of its group
   on composition and fractions, exactly as they are compared with each other;
   its 2D position is the distance-weighted mean of its k nearest foods'
   positions and its cluster is their weighted majority. With no nutrient
   values at all, the name neighbours stand in for the composition neighbours.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from foodspace import fractions, fuse, views


# product claims and origin words: they say how a food is sold or made, not what it is,
# so they are dropped before names are compared (they still stay in the food's name)
CLAIMS = re.compile(r"\b(organic|eco|bio|home-?made|store-?brand|own-?brand|brand|artisan|farm-?shop|farm|"
                    r"grandma'?s|family recipe|small-?batch|market stall|premium|traditional|classic|"
                    r"reduced-?sugar|sugar-?free|no added sugar|high-?protein|lactose-?free|low-?lactose|gluten-?free|"
                    r"extra crispy|industrial|diff\. brands?|fortified|e\.g\.)\b", re.I)


def clean_name(name: str) -> str:
    return re.sub(r"\s+", " ", CLAIMS.sub(" ", str(name).lower())).strip()


class NameIndex:
    """TF-IDF index of food names that can score names not seen when it was built."""

    def __init__(self, names):
        texts = [clean_name(n) for n in names]
        self.w = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(texts)
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True).fit(texts)
        self.M = self._vec(texts)

    def _vec(self, texts):
        return normalize(hstack([self.w.transform(texts), self.c.transform(texts)]).tocsr())

    def similarity(self, name: str) -> np.ndarray:
        return (self.M @ self._vec([clean_name(name)]).T).toarray().ravel()


@dataclass
class Placement:
    name: str
    gift_group: str
    gift_subgroup: str
    group_confidence: float
    name_neighbours: list
    imputed: list = field(default_factory=list)
    composition_neighbours: list = field(default_factory=list)
    cluster: int | None = None
    cluster_confidence: float | None = None
    x: float | None = None
    y: float | None = None
    basis: str = ""
    notes: list = field(default_factory=list)


class FoodSpace:
    def __init__(self, table: str | Path, results: str | Path, config: str | Path = "config/config.yaml"):
        self.cfg = yaml.safe_load(Path(config).read_text())
        df = pd.read_csv(table)
        self.df = df[df["gift_group"].notna() & (df["gift_group"] != "unassigned")].reset_index(drop=True)
        pos = pd.read_csv(Path(results) / "foods.csv")[["food_id", "cluster", "x", "y"]]
        self.df = self.df.merge(pos, on="food_id", how="left")
        self.names = NameIndex(self.df["name"].fillna(""))
        self.nutrients = self.cfg["columns"]["nutrients"]
        self.blocks = {k: v for k, v in self.cfg["fractions"]["blocks"].items() if v.get("source") != "curated"}
        self.protfrac = [c for c in self.df.columns if c.startswith("protfrac_")]

    # -- 1. group from the name ---------------------------------------------
    def _vote(self, sims, labels, k):
        idx = np.argsort(-sims)[:k]
        w = np.clip(sims[idx], 1e-6, None)
        votes = pd.Series(w, index=labels[idx]).groupby(level=0).sum().sort_values(ascending=False)
        return votes.index[0], float(votes.iloc[0] / votes.sum()), idx

    def place(self, name: str, values: dict | None = None, gift_group: str | None = None,
              gift_subgroup: str | None = None, k: int = 10) -> Placement:
        values = dict(values or {})
        sims = self.names.similarity(name)
        if gift_group is None:
            gift_group, conf, _ = self._vote(sims, self.df["gift_group"].to_numpy(), k)
        else:
            conf = 1.0
        g = self.df[self.df["gift_group"] == gift_group]
        gs = sims[g.index]
        if gift_subgroup is None:
            gift_subgroup, _, _ = self._vote(gs, g["gift_subgroup"].fillna("?").to_numpy(), k)
        order = np.argsort(-gs)[:k]
        nn = g.iloc[order]
        p = Placement(name, gift_group, gift_subgroup, conf,
                      [(r["name"], round(float(gs[i]), 3)) for i, (_, r) in zip(order, nn.iterrows())])
        if conf < 0.6:
            p.notes.append(f"Low confidence in the GIFT group from the name ({conf:.2f}); "
                           "check it and pass the group explicitly if it is wrong.")

        # -- 2. fill what the new food lacks from its name neighbours --------
        w = np.clip(gs[order], 1e-6, None)
        detail = sorted({c for b in self.blocks.values() for c in b["parts"] + [b.get("parent")] if c} | set(self.protfrac))
        has_any = any(c in values for c in self.nutrients)
        for c in detail:
            if c in values or c not in g:
                continue
            if c in self.nutrients and has_any:
                continue
            v = nn[c].to_numpy(dtype=float)
            ok = ~np.isnan(v)
            if ok.any() and has_any:  # only details are imputed when macros are given
                values[c] = float(np.average(v[ok], weights=w[ok]))
                p.imputed.append(c)

        # -- 3. compare with the group on composition + fractions ------------
        if has_any:
            row = {c: values.get(c, np.nan) for c in set(self.nutrients) | set(detail)}
            sub = pd.concat([g, pd.DataFrame([row])], ignore_index=True)
            Xp, _ = views.prepare_nutrients(sub[self.nutrients])
            Dn = views.nutrient_distance(Xp, self.cfg["nutrients"]["metric"])
            B = fractions.build_blocks(sub, {"blocks": self.blocks}, None)
            if self.protfrac:
                P = sub[self.protfrac].astype(float)
                if P.notna().all(axis=1).sum() >= 2:
                    B["protein_type"] = P
            if B:
                Fd, _ = fractions.fraction_distance(B)
                D = fuse.weighted_fusion({"comp": Dn, "frac": Fd}, {"comp": 1, "frac": 1})
            else:
                D = Dn
            d = D[-1, :-1]
            idx = np.argsort(d)[:k]
            wts = 1.0 / (d[idx] + 1e-3)
            p.basis = "composition and fractions"
        else:
            idx, wts, d = order, w, 1 - gs
            p.basis = "name only (no nutrient values given)"
        ref = g.iloc[idx]
        p.composition_neighbours = [(r["name"], round(float(d[i]), 3)) for i, (_, r) in zip(idx, ref.iterrows())]
        p.x = float(np.average(ref["x"], weights=wts))
        p.y = float(np.average(ref["y"], weights=wts))
        cv = pd.Series(wts, index=ref["cluster"].to_numpy()).groupby(level=0).sum().sort_values(ascending=False)
        p.cluster, p.cluster_confidence = int(cv.index[0]), float(cv.iloc[0] / cv.sum())
        if p.imputed:
            p.notes.append(f"{len(p.imputed)} detail values were filled from the name neighbours: " + ", ".join(p.imputed))
        return p

    def evaluate_names(self, n: int = 600, k: int = 10, seed: int = 0) -> dict:
        """Leave-one-out: how often do names alone recover the GIFT group / subgroup?"""
        rng = np.random.default_rng(seed)
        ids = rng.choice(len(self.df), size=min(n, len(self.df)), replace=False)
        grp, sub = self.df["gift_group"].to_numpy(), self.df["gift_subgroup"].fillna("?").to_numpy()
        hit_g = hit_s = 0
        for i in ids:
            s = (self.names.M @ self.names.M[i].T).toarray().ravel()
            s[i] = -1  # leave the food itself out
            gg, _, _ = self._vote(s, grp, k)
            ss, _, _ = self._vote(s, sub, k)
            hit_g += gg == grp[i]
            hit_s += ss == sub[i]
        return {"foods_tested": int(len(ids)), "group_accuracy": hit_g / len(ids), "subgroup_accuracy": hit_s / len(ids)}
