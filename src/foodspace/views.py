"""The three similarity views: classification, nutrients, semantics.

Each function returns an n x n distance matrix scaled to [0, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def _scale01(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    hi = np.nanmax(D)
    return D / hi if hi > 0 else D


def gift_distance(groups, subgroups, same_group: float = 0.5) -> np.ndarray:
    """Tree distance in the GIFT hierarchy.

    0 = same subgroup, `same_group` = same group but different subgroup,
    1 = different group.
    """
    g = np.asarray(pd.Series(groups).astype(str))
    s = np.asarray(pd.Series(subgroups).astype(str))
    D = np.ones((len(g), len(g)))
    D[g[:, None] == g[None, :]] = same_group
    D[(g[:, None] == g[None, :]) & (s[:, None] == s[None, :])] = 0.0
    return D


def prepare_nutrients(
    X: pd.DataFrame,
    max_missing: float = 0.3,
    impute: str = "group_median",
    groups=None,
    log: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Clean nutrient columns before computing distances.

    - drops nutrients missing in more than `max_missing` of foods
    - imputes the rest (median within GIFT group, falling back to overall median)
    - log1p-transforms (composition values are right-skewed)
    - z-scores each nutrient

    Returns the prepared matrix and a report of what was done, so the
    choices stay visible in the results.
    """
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    frac_missing = X.isna().mean()
    dropped = frac_missing[frac_missing > max_missing].index.tolist()
    X = X.drop(columns=dropped)
    n_imputed = int(X.isna().sum().sum())
    if impute == "group_median" and groups is not None:
        X = X.groupby(np.asarray(groups)).transform(lambda c: c.fillna(c.median()))
    X = X.fillna(X.median())
    if (X < 0).any().any():
        raise ValueError("Negative nutrient values found; check units or coding errors.")
    if log:
        X = np.log1p(X)
    sd = X.std(ddof=0).replace(0, 1)
    X = (X - X.mean()) / sd
    report = {
        "nutrients_used": X.columns.tolist(),
        "nutrients_dropped_for_missingness": dropped,
        "values_imputed": n_imputed,
        "imputation": impute,
        "log1p": log,
    }
    return X, report


def nutrient_distance(Xp: pd.DataFrame, metric: str = "cosine") -> np.ndarray:
    """Distance between prepared nutrient profiles (cosine or correlation)."""
    D = squareform(pdist(Xp.to_numpy(), metric=metric))
    return _scale01(np.nan_to_num(D, nan=1.0))


def semantic_embeddings(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    """Sentence embeddings for food names and descriptions.

    Needs `sentence-transformers`. A multilingual model such as
    `paraphrase-multilingual-MiniLM-L12-v2` handles non-English food names
    without translation.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError("Install sentence-transformers to use the semantic view.") from e
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)


def tfidf_embeddings(texts: list[str]) -> np.ndarray:
    """Offline fallback for food-name semantics: TF-IDF on word unigrams/bigrams
    plus character 3-5-grams (catches shared stems such as "cheese"/"cheesecake"),
    L2-normalised, so cosine distance compares wording rather than meaning."""
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    words = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform(texts)
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True).fit_transform(texts)
    return normalize(hstack([words, chars]).tocsr()).toarray()


def embedding_distance(E: np.ndarray) -> np.ndarray:
    return _scale01(squareform(pdist(E, metric="cosine")))
