"""The two association tests of scripts/similarity_power.py: calibrated under no effect, and
they find an effect carried by a branch of the food tree."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("similarity_power", ROOT / "scripts" / "similarity_power.py")
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)


def toy(n=120, seed=0):
    """Two groups of three foods; leaf keys a0..a2 (group a) and b0..b2 (group b)."""
    rng = np.random.default_rng(seed)
    keys = [f"{g}{i}" for g in "ab" for i in range(3)]
    parent = np.array([-1, 0, 0, 1, 1, 1, 2, 2, 2])  # root, group a, group b, six leaves
    length = np.array([0, 1, 1, .5, .5, .5, .5, .5, .5])
    leaf_index = {k: 3 + j for j, k in enumerate(keys)}
    P = pd.DataFrame(rng.dirichlet(np.ones(6), size=n), columns=keys)
    return P, leaf_index, parent, length, rng


def test_tests_are_calibrated_under_no_effect():
    from foodspace import diet

    P, li, parent, length, rng = toy()
    kern = sp.Kernel(diet.tree_distance(P, li, parent, length), rng, draws=4000)
    nodes = sp.NodeTest(P, li, parent, rng, draws=4000)
    p_k = np.array([kern.p(rng.standard_normal(len(P))) for _ in range(400)])
    p_n = np.array([nodes.p(rng.standard_normal(len(P))) for _ in range(400)])
    assert abs((p_k < 0.05).mean() - 0.05) < 0.035
    assert abs((p_n < 0.05).mean() - 0.05) < 0.035


def test_branch_effect_is_found():
    from foodspace import diet

    P, li, parent, length, rng = toy(n=200)
    x = P[["a0", "a1", "a2"]].sum(axis=1).to_numpy()
    y = (x - x.mean()) / x.std() + rng.standard_normal(len(P))
    assert sp.Kernel(diet.tree_distance(P, li, parent, length), rng).p(y) < 0.01
    assert sp.NodeTest(P, li, parent, rng).p(y) < 0.01


def test_clades_lists_leaves_below_each_node():
    _, li, parent, _, _ = toy()
    below = sp.clades(parent, {v: j for j, v in enumerate(li.values())})
    assert below[1] == {0, 1, 2} and below[0] == set(range(6))
    assert sp.depth(parent, 4) == 2
