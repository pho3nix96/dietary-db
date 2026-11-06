"""2D or 3D coordinates for the map."""

from __future__ import annotations

import numpy as np


def embed_2d(D: np.ndarray, method: str = "umap", seed: int = 0) -> np.ndarray:
    """2D layout from a precomputed distance matrix (see embed)."""
    return embed(D, method, seed, dim=2)


def embed(D: np.ndarray, method: str = "umap", seed: int = 0, dim: int = 2, min_dist: float = 0.1) -> np.ndarray:
    """Layout in `dim` dimensions (2 or 3) from a precomputed distance matrix.

    UMAP (umap-learn) is preferred; t-SNE from scikit-learn is the fallback. A larger
    min_dist spreads points within a cluster (useful for 3D views).
    """
    if method == "umap":
        try:
            import umap

            n_neighbors = max(2, min(15, D.shape[0] - 1))
            return umap.UMAP(n_components=dim, metric="precomputed", n_neighbors=n_neighbors, min_dist=min_dist, spread=max(1.0, min_dist * 1.5), random_state=seed).fit_transform(D)
        except ImportError:
            import warnings

            warnings.warn("umap-learn is not installed; using t-SNE instead, so layouts differ from UMAP ones "
                          "(pip install umap-learn)", stacklevel=2)
            method = "tsne"
    from sklearn.manifold import TSNE

    perplexity = max(2.0, min(30.0, (D.shape[0] - 1) / 3))
    return TSNE(n_components=dim, metric="precomputed", init="random", perplexity=perplexity, random_state=seed).fit_transform(D)


def similarity_to_distance(W: np.ndarray) -> np.ndarray:
    """Convert a fused similarity matrix to a distance in [0, 1]."""
    W = W / W.max() if W.max() > 0 else W
    D = 1 - W
    np.fill_diagonal(D, 0)
    return D
