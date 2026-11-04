"""Treelet transform (Lee, Nadler & Wasserman 2008, Ann. Appl. Stat. 2:435-471).

A treelet is a hierarchical, local PCA. Starting from p variables, each step

1. finds the two active variables with the highest similarity
   (here signed correlation, so negatively correlated variables never merge),
2. rotates that pair with a Jacobi rotation so they become uncorrelated:
   a "sum" variable (larger variance, the pair's first principal component)
   and a "difference" variable,
3. keeps the sum variable active and retires the difference variable.

After p - 1 steps this gives a binary tree over the variables and an
orthonormal basis. Every internal node has a loading vector over the original
variables that is non-zero only on that node's leaves.

Food tree use: foods are the variables and their standardized composition
profiles (macros, fraction log-ratios) are the observations, so two foods are
similar when their profiles correlate. A node's sum loading weights its member
foods; the sum variable is the branch's prototype composition.
Gorst-Rasmussen et al. (2011) built the tree the other way round, from
correlations of food intakes across people; `treelet` works on either matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def treelet(C: np.ndarray, kmax: int = 20) -> dict:
    """Run the treelet transform to the root on a covariance matrix C (p x p).

    Returns
        linkage   (p-1) x 4 array in scipy format: child ids, 1 - similarity, size
        nodes     list of dicts per merge: level, similarity, theta, leaves, loading
                  (loading: weights over `leaves`, unit norm)
        basis     final orthonormal basis (p x p)
        cov       final rotated covariance
        topvar    p x kmax array: row L holds the kmax largest variances among
                  the active (sum) variables after L merges, in decreasing order;
                  used to choose the cut level (`explained_by_level`)
    """
    C = np.array(C, dtype=float, copy=True)
    p = C.shape[0]
    B = np.eye(p)
    active = np.ones(p, bool)
    leaves = {i: [i] for i in range(p)}
    node_of = {i: i for i in range(p)}
    Z, nodes = [], []
    kmax = min(kmax, p)
    topvar = np.full((p, kmax), np.nan)
    topvar[0] = np.sort(np.diag(C))[::-1][:kmax]

    def corr():
        d = np.sqrt(np.clip(np.diag(C), 1e-300, None))
        R = C / np.outer(d, d)
        R[~active] = -np.inf
        R[:, ~active] = -np.inf
        np.fill_diagonal(R, -np.inf)
        return R

    R = corr()
    for level in range(p - 1):
        a, b = np.unravel_index(int(np.argmax(R)), R.shape)
        sim = R[a, b]
        theta = 0.5 * np.arctan2(2 * C[a, b], C[a, a] - C[b, b])
        c, s = np.cos(theta), np.sin(theta)
        ra, rb = C[a].copy(), C[b].copy()
        C[a], C[b] = c * ra + s * rb, -s * ra + c * rb
        ca, cb = C[:, a].copy(), C[:, b].copy()
        C[:, a], C[:, b] = c * ca + s * cb, -s * ca + c * cb
        C[a, b] = C[b, a] = 0.0
        Ba, Bb = B[:, a].copy(), B[:, b].copy()
        B[:, a], B[:, b] = c * Ba + s * Bb, -s * Ba + c * Bb
        # slot a now holds the sum variable (larger variance), slot b the difference
        active[b] = False
        lv = leaves[a] + leaves.pop(b)
        leaves[a] = lv
        Z.append([node_of[a], node_of[b], 1.0 - sim, len(lv)])
        node_of[a] = p + level
        nodes.append({"node": p + level, "level": level + 1, "similarity": float(sim), "theta": float(theta),
                      "leaves": lv, "loading": B[lv, a].copy()})
        # update similarities: only row/column a changed, row/column b retired
        d = np.sqrt(np.clip(np.diag(C), 1e-300, None))
        ra = C[a] / (d[a] * d)
        ra[~active] = -np.inf
        ra[a] = -np.inf
        R[a], R[:, a] = ra, ra
        R[b], R[:, b] = -np.inf, -np.inf
        v = np.sort(np.diag(C)[active])[::-1][:kmax]
        topvar[level + 1, : len(v)] = v
    return {"linkage": np.array(Z), "nodes": nodes, "basis": B, "cov": C, "topvar": topvar}


def explained_by_level(result: dict, k: int, total_variance: float) -> np.ndarray:
    """Share of total variance carried by the k largest sum variables at each
    cut level L = 0 .. p-1. The review's rule (Gorst-Rasmussen et al. 2011):
    fix k, then take the level where this curve stops increasing."""
    tv = result["topvar"][:, :k]
    return np.nansum(tv, axis=1) / total_variance


def feature_matrix(blocks: dict[str, pd.DataFrame], weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Stack feature blocks (foods x features). Each column is standardized and
    each block scaled to the same total variance (times its weight), so a block
    with many columns does not dominate."""
    out = []
    for name, X in blocks.items():
        X = X.astype(float)
        Z = (X - X.mean()) / X.std(ddof=0).replace(0, 1.0)
        Z = Z.fillna(0.0) / np.sqrt(max(Z.shape[1], 1))
        w = 1.0 if weights is None else weights.get(name, 1.0)
        out.append(np.sqrt(w) * Z.add_prefix(f"{name}:"))
    return pd.concat(out, axis=1)


def food_covariance(F: pd.DataFrame) -> np.ndarray:
    """Covariance between foods (rows) across features (columns), each food's
    profile centred over its features."""
    X = F.to_numpy(dtype=float)
    Xc = X - X.mean(axis=1, keepdims=True)
    return Xc @ Xc.T / X.shape[1]


def cut(linkage: np.ndarray, p: int, n_clusters: int) -> np.ndarray:
    """Labels from stopping the merges when n_clusters branches remain."""
    parent = list(range(p))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    rep = {i: i for i in range(p)}                 # cluster id -> one leaf
    for k in range(p - n_clusters):
        a, b = int(linkage[k, 0]), int(linkage[k, 1])
        ra, rb = find(rep[a]), find(rep[b])
        parent[rb] = ra
        rep[p + k] = ra
    roots = [find(i) for i in range(p)]
    _, labels = np.unique(roots, return_inverse=True)
    return labels
