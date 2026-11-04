"""Turn distances into graphs, fuse the views, and cluster."""

from __future__ import annotations

import numpy as np


def affinity(D: np.ndarray, k: int = 15, mu: float = 0.5) -> np.ndarray:
    """Scaled exponential kernel (as in SNF) on a distance matrix.

    Bandwidth per pair adapts to each food's local neighbourhood, so dense
    and sparse regions of food space are treated comparably.
    """
    n = D.shape[0]
    k = min(k, n - 1)
    Ds = np.sort(D, axis=1)
    local = Ds[:, 1 : k + 1].mean(axis=1)
    eps = (local[:, None] + local[None, :] + D) / 3
    W = np.exp(-(D**2) / (mu * eps + 1e-12))
    np.fill_diagonal(W, 0)
    return (W + W.T) / 2


def knn_sparsify(W: np.ndarray, k: int = 15) -> np.ndarray:
    """Keep each food's k strongest edges; symmetrize."""
    n = W.shape[0]
    k = min(k, n - 1)
    S = np.zeros_like(W)
    idx = np.argsort(-W, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    S[rows, idx.ravel()] = W[rows, idx.ravel()]
    return np.maximum(S, S.T)


def weighted_fusion(dists: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted mean of distance matrices (the adjustable baseline)."""
    total = sum(weights.get(v, 0) for v in dists)
    if total <= 0:
        raise ValueError("At least one view needs a positive weight.")
    return sum(weights.get(v, 0) * D for v, D in dists.items()) / total


def snf(affinities: list[np.ndarray], k: int = 15, t: int = 20) -> np.ndarray:
    """Similarity network fusion (Wang et al. 2014, Nature Methods).

    Iteratively diffuses each view's similarity through the others' local
    kNN structure, so edges supported by several views are reinforced.
    """
    def norm_full(W):
        W = W.copy()
        np.fill_diagonal(W, 0)
        rs = W.sum(axis=1, keepdims=True)
        P = W / (2 * np.where(rs == 0, 1, rs))
        np.fill_diagonal(P, 0.5)
        return P

    def norm_local(W):
        S = knn_sparsify(W, k)
        rs = S.sum(axis=1, keepdims=True)
        return S / np.where(rs == 0, 1, rs)

    Ps = [norm_full(W) for W in affinities]
    Ss = [norm_local(W) for W in affinities]
    m = len(Ps)
    for _ in range(t):
        new = []
        for v in range(m):
            others = sum(Ps[u] for u in range(m) if u != v) / max(m - 1, 1)
            P = Ss[v] @ others @ Ss[v].T
            new.append(norm_full((P + P.T) / 2))
        Ps = new
    F = sum(Ps) / m
    F = (F + F.T) / 2
    np.fill_diagonal(F, 0)
    return F


def leiden(W: np.ndarray, resolution: float = 1.0, seed: int = 0) -> np.ndarray:
    """Leiden communities on a weighted graph (RBConfiguration, i.e. modularity with resolution)."""
    import igraph as ig
    import leidenalg as la

    iu = np.triu_indices_from(W, k=1)
    mask = W[iu] > 0
    edges = list(zip(iu[0][mask], iu[1][mask]))
    g = ig.Graph(n=W.shape[0], edges=edges)
    g.es["weight"] = W[iu][mask].tolist()
    part = la.find_partition(
        g, la.RBConfigurationVertexPartition, weights="weight",
        resolution_parameter=resolution, seed=seed,
    )
    return np.asarray(part.membership)


def n_components(W: np.ndarray) -> int:
    """Number of connected parts in the graph."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    return int(connected_components(csr_matrix(W > 0), directed=False)[0])
