"""Unit tests for fuse, views, evaluate and the fraction distance."""

import numpy as np
import pandas as pd
import pytest

from foodspace import evaluate, fractions, fuse, io, layout, views


def two_blobs(n=15, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 0.1, (n, 3)), rng.normal(3, 0.1, (n, 3))])
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    return D / D.max(), np.array([0] * n + [1] * n)


def test_affinity_symmetric_zero_diagonal():
    D, _ = two_blobs()
    W = fuse.affinity(D, k=5)
    assert np.allclose(W, W.T) and np.all(np.diag(W) == 0) and (W >= 0).all()


def test_knn_sparsify_keeps_at_least_k_and_is_symmetric():
    D, _ = two_blobs()
    S = fuse.knn_sparsify(fuse.affinity(D, 5), 5)
    assert np.allclose(S, S.T)
    assert ((S > 0).sum(axis=1) >= 5).all()


def test_weighted_fusion_weights_and_errors():
    A, B = np.zeros((3, 3)), np.ones((3, 3))
    assert np.allclose(fuse.weighted_fusion({"a": A, "b": B}, {"a": 1, "b": 3}), 0.75)
    with pytest.raises(ValueError):
        fuse.weighted_fusion({"a": A}, {"a": 0})


def test_leiden_recovers_blobs_and_components():
    D, truth = two_blobs()
    W = fuse.knn_sparsify(fuse.affinity(D, 5), 5)
    labels = fuse.leiden(W, 1.0, 0)
    assert evaluate.agreement(labels, truth)["ARI"] > 0.9
    assert fuse.n_components(W) >= 1


def test_agreement_perfect_and_misfits():
    ref = pd.Series(["a"] * 5 + ["b"] * 5)
    assert evaluate.agreement(np.array([0] * 5 + [1] * 5), ref)["ARI"] == pytest.approx(1.0)
    df = pd.DataFrame({"id": range(10), "name": list("abcdefghij"), "g": ref})
    clusters = np.array([0] * 5 + [1] * 4 + [0])  # last food sits with the wrong group
    mis = evaluate.misfits(df, clusters, "g", "id", "name")
    assert mis["id"].tolist() == [9]


def test_prepare_nutrients_median_ignores_groups():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 100.0], "b": [1.0, 2.0, 3.0, 4.0]})
    groups = ["x", "x", "y", "y"]
    med, rep = views.prepare_nutrients(X, impute="median", groups=groups, log=False)
    grp, _ = views.prepare_nutrients(X, impute="group_median", groups=groups, log=False)
    assert rep["values_imputed"] == 1
    assert not np.isclose(med.loc[1, "a"], grp.loc[1, "a"])  # group_median uses the group, median does not


def test_prepare_nutrients_rejects_negative_and_drops_sparse():
    with pytest.raises(ValueError):
        views.prepare_nutrients(pd.DataFrame({"a": [-1.0, 2.0]}))
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [np.nan, np.nan, np.nan, 1.0]})
    _, rep = views.prepare_nutrients(X, max_missing=0.3)
    assert rep["nutrients_dropped_for_missingness"] == ["b"]


def test_gift_distance_levels():
    D = views.gift_distance(["a", "a", "a", "b"], ["x", "x", "y", "z"])
    assert D[0, 1] == 0 and D[0, 2] == 0.5 and D[0, 3] == 1


def test_similarity_to_distance_range():
    D = layout.similarity_to_distance(fuse.affinity(two_blobs()[0], 5))
    assert D.min() >= 0 and D.max() <= 1 and np.all(np.diag(D) == 0)


def test_aitchison_matches_direct_formula_and_nan():
    S = pd.DataFrame({"a": [0.5, 0.2, np.nan], "b": [0.3, 0.3, 0.3], "c": [0.2, 0.5, 0.4]})
    D = fractions.aitchison_distance(S)
    C = fractions.clr(S).to_numpy()
    assert D[0, 1] == pytest.approx(np.linalg.norm(C[0] - C[1]))
    assert np.isnan(D[0, 2]) and np.isnan(D[2, 2])


def test_fraction_distance_min_shared():
    full = pd.DataFrame({"a": [0.5, 0.2, 0.3], "b": [0.5, 0.8, 0.7]})
    partial = full.copy()
    partial.iloc[2] = np.nan
    blocks = {"one": full, "two": partial}
    D1, _ = fractions.fraction_distance(blocks, min_shared=1)
    D2, rep = fractions.fraction_distance(blocks, min_shared=2)
    assert D1[0, 2] < 1.0 and D2[0, 2] == 1.0 and D2[0, 1] == pytest.approx(D1[0, 1])
    assert rep["pairs_without_shared_block"] == 2


def test_load_foods_checks_columns_and_duplicates(tmp_path):
    cols = {"id": "id", "name": "name", "gift_group": "g", "gift_subgroup": "s", "nutrients": ["p"]}
    ok = tmp_path / "ok.csv"
    pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "g": ["x", "x"], "s": ["y", "y"], "p": [1, 2]}).to_csv(ok, index=False)
    assert len(io.load_foods(str(ok), cols)) == 2
    dup = tmp_path / "dup.csv"
    pd.DataFrame({"id": [1, 1], "name": ["a", "b"], "g": ["x", "x"], "s": ["y", "y"], "p": [1, 2]}).to_csv(dup, index=False)
    with pytest.raises(ValueError):
        io.load_foods(str(dup), cols)
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"id": [1], "name": ["a"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        io.load_foods(str(bad), cols)
