import numpy as np
import pandas as pd

from foodspace import treelet as tl


def test_treelet_recovers_blocks_and_stays_orthonormal():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(500, 3))
    X = np.hstack([f[:, [i]] + 0.3 * rng.normal(size=(500, 4)) for i in range(3)])
    C = np.cov(X.T)
    r = tl.treelet(C)
    assert list(tl.cut(r["linkage"], 12, 3)) == [0] * 4 + [1] * 4 + [2] * 4
    B = r["basis"]
    assert np.allclose(B.T @ B, np.eye(12))
    assert np.allclose(B.T @ C @ B, r["cov"])
    node = r["nodes"][-3]                      # one block of four, roughly equal loadings
    assert sorted(node["leaves"]) in ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11])
    assert np.allclose(np.abs(node["loading"]), 0.5, atol=0.05)
    curve = tl.explained_by_level(r, 3, np.trace(C))
    assert curve[8] > curve[0]                 # merging blocks concentrates variance


def test_foods_with_similar_profiles_merge_first():
    F = pd.DataFrame({"a": [1, 1.1, -1, -1.2], "b": [0, 0.1, 2, 2.1], "c": [-1, -1.1, -1, -0.8]},
                     index=["gouda", "edam", "milk", "kefir"])
    r = tl.treelet(tl.food_covariance(tl.feature_matrix({"x": F})))
    first_two = {frozenset(n["leaves"]) for n in r["nodes"][:2]}
    assert first_two == {frozenset({0, 1}), frozenset({2, 3})}
