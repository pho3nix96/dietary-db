#!/usr/bin/env python3
"""Does food similarity help find diet-outcome associations? A power simulation.

    python scripts/similarity_power.py [--sets 5] [--subjects 300] [--reps 100]
        [--r2 0.02 0.05] [--out results/similarity_power]

The question: an analysis that treats every food as unique splits the signal of closely
related foods (whole and semi-skimmed milk) over many sparse variables. Does an analysis that
knows which foods are related find a real effect more often?

Design
- Record sets: `--sets` sets of `--subjects` people x 3 days from simulate_food_records.py
  (seeds 1..sets, no unique products). Each person's diet is their energy share on each food.
- Outcome: y = sqrt(R2) * z(x) + sqrt(1 - R2) * noise, where z(x) is the standardised exposure
  and noise is standard normal, so the exposure explains R2 of the outcome. Scenarios for x:
      subgroup     energy share from one GIFT subgroup (drawn at random per replicate)
      clade        energy share from one branch of a composition tree inside a GIFT group
                   (3 to 40 foods, never a whole subgroup)
      single_food  energy share from one food
      sfa          share of energy from saturated fat (a nutrient that crosses food groups)
      lc_n3        long-chain omega-3 per 1000 kcal
      null         no effect (R2 = 0), to check the false-positive rate
  Sets and foods are drawn among those that at least `--min-exposed` of people eat.
- Tests, for each food tree (as in compare_diet_trees.py):
      nodes   the intake share below every node of the tree is correlated with y, and the
              largest |correlation| is the statistic (its null simulated per tree). With the
              flat tree this is the usual "test every food" analysis; with a tree, related
              foods are also tested together (as subgroups, branches, groups).
      kernel  a whole-diet kernel association test (the MiRKAT idea): K = -1/2 J D^2 J from
              the tree distance D between people, negative eigenvalues set to 0, statistic
              Q = y'Ky / y'y; its null for normal errors is simulated once per kernel.
  "flat" treats every food as unique. "oracle" is a correlation test on the true exposure:
  the best any method could do, since it knows what drives the outcome.
- Power: share of replicates with p < 0.05.

Caveat: the simulated people choose foods by dietary pattern and habit, and the outcome is
defined on the same food table, so this shows when similarity can help, not how much it helps
in real data.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from foodspace import diet

TREES = ["flat", "gift_groups", "gift_subgroups", "subgroup_composition", "composition"]
SCENARIOS = ["subgroup", "clade", "single_food", "sfa", "lc_n3", "null"]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Kernel:
    """Kernel test for one person-by-person distance matrix."""

    def __init__(self, D: np.ndarray, rng, draws: int = 20000):
        n = D.shape[0]
        J = np.eye(n) - 1.0 / n
        K = -0.5 * J @ (D ** 2) @ J
        lam, U = np.linalg.eigh((K + K.T) / 2)
        lam = np.clip(lam, 0, None)
        self.K = (U * lam) @ U.T
        # K1 = 0, so one zero eigenvalue belongs to the constant vector; y is centred, so it
        # lives in the other n-1 dimensions (eigh sorts ascending: drop one zero)
        lam = lam[1:]
        C = rng.chisquare(1, size=(draws, n - 1))
        self.null = np.sort(C @ lam / C.sum(axis=1))

    def p(self, y: np.ndarray) -> float:
        y = y - y.mean()
        q = y @ self.K @ y / (y @ y)
        return (np.sum(self.null >= q) + 1) / (len(self.null) + 1)


class NodeTest:
    """Test every node of a food tree: correlation of y with each person's intake share below
    the node (for the flat tree: each food alone). Statistic: the largest |correlation|; its null
    distribution for normal errors is simulated once per tree, so it accounts for testing many
    correlated nodes. Nodes eaten by fewer than `min_eaten` of people are not tested."""

    def __init__(self, P: pd.DataFrame, leaf_index: dict, parent: np.ndarray, rng,
                 min_eaten: float = 0.05, draws: int = 5000):
        rows, cols = [], []
        for j, k in enumerate(P.columns):
            v = leaf_index[k]
            while v != 0:
                rows.append(v)
                cols.append(j)
                v = parent[v]
        A = np.zeros((len(parent), P.shape[1]))
        A[rows, cols] = 1.0
        M = P.to_numpy() @ A.T  # people x nodes
        keep = (M > 0).mean(axis=0) >= min_eaten
        keep &= M.std(axis=0) > 0
        M = M[:, keep]
        self.Z = (M - M.mean(axis=0)) / M.std(axis=0)
        self.n_nodes = int(keep.sum())
        Y = rng.standard_normal((self.Z.shape[0], draws))
        self.null = np.sort(self._stat(Y))

    def _stat(self, Y: np.ndarray) -> np.ndarray:
        Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
        return np.abs(self.Z.T @ Y).max(axis=0) / Y.shape[0]

    def p(self, y: np.ndarray) -> float:
        q = self._stat(y[:, None])[0]
        return (np.sum(self.null >= q) + 1) / (len(self.null) + 1)


def clades(parent: np.ndarray, leaf_cols: dict[int, int]) -> dict[int, set]:
    """Leaf columns below every node (node -> set of column indices)."""
    below: dict[int, set] = {}
    for node, col in leaf_cols.items():
        v = node
        while v != -1:
            below.setdefault(v, set()).add(col)
            v = parent[v]
    return below


def depth(parent: np.ndarray, v: int) -> int:
    d = 0
    while parent[v] != -1:
        v, d = parent[v], d + 1
    return d


def exposure_sets(trees_built, leaves, P, min_exposed):
    """Candidate food sets: GIFT subgroups and composition clades, as column-index arrays."""
    eaten = (P > 0).to_numpy()
    prevalent = lambda cols: eaten[:, sorted(cols)].any(axis=1).mean() >= min_exposed
    col = {k: j for j, k in enumerate(leaves)}

    li, parent, _, _ = trees_built["gift_subgroups"]
    below = clades(parent, {li[k]: col[k] for k in leaves})
    subgroups = [frozenset(c) for v, c in below.items() if depth(parent, v) == 2 and prevalent(c)]

    li, parent, _, _ = trees_built["composition"]
    below = clades(parent, {li[k]: col[k] for k in leaves})
    used = eaten.any(axis=0)
    sub_eaten = {frozenset(j for j in s if used[j]) for s in subgroups}
    leaf_nodes = set(li.values())
    out = []
    for v, c in below.items():
        if v in leaf_nodes or depth(parent, v) < 2:
            continue
        c_eaten = frozenset(j for j in c if used[j])
        if 3 <= len(c_eaten) <= 40 and c_eaten not in sub_eaten and prevalent(c):
            out.append(frozenset(c))
    comp = sorted(set(out), key=sorted)
    foods = [frozenset([j]) for j in np.flatnonzero(eaten.mean(axis=0) >= min_exposed)]
    return {"subgroup": sorted(set(subgroups), key=sorted), "clade": comp, "single_food": foods}


def nutrient_exposures(P, leaves, foods):
    """Person-level nutrient exposures from energy shares and the food table (per 100 g)."""
    f = foods.set_index(foods["food_id"].map(lambda i: f"f{int(i)}"))
    f = f.reindex(leaves)
    kcal = pd.to_numeric(f["ENERC_kcal"], errors="coerce").to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        sfa = np.where(kcal > 0, 9 * pd.to_numeric(f["FASAT_g"], errors="coerce").to_numpy() / kcal, 0.0)
        n3 = np.where(kcal > 0, 1000 * pd.to_numeric(f["lc_n3_g"], errors="coerce").to_numpy() / kcal, 0.0)
    X = P.to_numpy()
    return {"sfa": X @ np.nan_to_num(sfa), "lc_n3": X @ np.nan_to_num(n3)}


def run_set(seed, records_dir, foods, trees_json, args, rng):
    consumption = pd.read_csv(Path(records_dir) / "consumption.csv")
    matched = diet.match_records(consumption, foods)
    sim = load_module("compare_diet_trees")
    built = sim.food_trees(foods, matched, trees_json, None)
    leaves = list(built["gift_subgroups"][0])
    P = diet.diet_profiles(matched, built["gift_subgroups"][0]).reindex(columns=leaves, fill_value=0.0)
    tests = {}
    for t, (li, parent, length, _) in built.items():
        Pt = P.reindex(columns=list(li), fill_value=0.0)
        tests[f"kernel_{t}"] = Kernel(diet.tree_distance(Pt, li, parent, length), rng)
        tests[f"nodes_{t}"] = NodeTest(Pt, li, parent, rng)
    sets = exposure_sets(built, leaves, P, args.min_exposed)
    nutr = nutrient_exposures(P, leaves, foods)
    X = P.to_numpy()
    n = X.shape[0]
    rows = []
    for scen in SCENARIOS:
        r2s = [0.0] if scen == "null" else args.r2
        for r2 in r2s:
            for rep in range(args.reps if scen != "null" else args.reps * len(args.r2)):
                if scen in sets:
                    cols = sorted(sets[scen][rng.integers(len(sets[scen]))])
                    x = X[:, cols].sum(axis=1)
                elif scen in nutr:
                    x = nutr[scen]
                else:
                    x = rng.standard_normal(n)
                z = (x - x.mean()) / (x.std() or 1.0)
                y = np.sqrt(r2) * z + np.sqrt(1 - r2) * rng.standard_normal(n)
                row = {"seed": seed, "scenario": scen, "r2": r2, "rep": rep,
                       "oracle": stats.pearsonr(x, y)[1] if x.std() > 0 else 1.0}
                row.update({t: k.p(y) for t, k in tests.items()})
                rows.append(row)
    counts = {f"candidate_{k}s": len(v) for k, v in sets.items()}
    nodes = {f"nodes_tested_{t}": tests[f"nodes_{t}"].n_nodes for t in built}
    return rows, {"seed": seed, "people": n, "foods_eaten": int((X > 0).any(axis=0).sum()), **counts, **nodes}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--foods", default="data/processed/foods_lv.csv")
    ap.add_argument("--trees", default="results/within_gift/trees.json")
    ap.add_argument("--sets", type=int, default=5)
    ap.add_argument("--subjects", type=int, default=300)
    ap.add_argument("--reps", type=int, default=100, help="replicates per scenario, R2 and record set")
    ap.add_argument("--r2", type=float, nargs="+", default=[0.02, 0.05])
    ap.add_argument("--min-exposed", type=float, default=0.15, help="share of people who must eat the food set")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="results/similarity_power")
    ap.add_argument("--save-pvalues", action="store_true", help="also write every p-value (pvalues.csv.gz)")
    args = ap.parse_args(argv)

    foods = pd.read_csv(args.foods, low_memory=False)
    foods[["gift_group", "gift_subgroup"]] = foods[["gift_group", "gift_subgroup"]].fillna("unassigned")
    trees_json = json.loads(Path(args.trees).read_text())
    simulator = load_module("simulate_food_records")
    rng = np.random.default_rng(0)
    rows, info = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in range(1, args.sets + 1):
            rec = Path(tmp) / f"seed{seed}"
            with contextlib.redirect_stdout(io.StringIO()):
                simulator.main(["--foods", args.foods, "--seed", str(seed), "--subjects", str(args.subjects),
                                "--unique-share", "0", "--out", str(rec)])
            r, i = run_set(seed, rec, foods, trees_json, args, rng)
            rows += r
            info.append(i)
            print(f"record set {seed}/{args.sets}: {i}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    r = pd.DataFrame(rows)
    if args.save_pvalues:
        r.to_csv(out / "pvalues.csv.gz", index=False, float_format="%.5g")
    tests = ["oracle", *[f"{m}_{t}" for m in ("nodes", "kernel") for t in TREES]]
    power = r.groupby(["scenario", "r2"], sort=False)[tests].agg(lambda p: (p < args.alpha).mean()).round(3)
    per_set = r.groupby(["scenario", "r2", "seed"], sort=False)[tests].agg(lambda p: (p < args.alpha).mean())
    spread = per_set.groupby(level=["scenario", "r2"], sort=False).agg(["min", "max"]).round(2)
    power.to_csv(out / "power.csv")
    spread.to_csv(out / "power_range_over_sets.csv")
    (out / "record_sets.json").write_text(json.dumps({"settings": vars(args), "record_sets": info}, indent=2))
    print(f"\nShare of replicates with p < {args.alpha} (null row = false-positive rate):")
    for m in ("nodes", "kernel"):
        cols = ["oracle", *[f"{m}_{t}" for t in TREES]]
        print(f"\n{m} test:")
        print(power[cols].rename(columns=lambda c: c.replace(f"{m}_", "")).to_string())
    print(f"Wrote {out}/")


if __name__ == "__main__":
    main()
