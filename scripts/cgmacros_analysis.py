"""Meal similarity, relative dose and glycaemic subgroups on the CGMacros data.

First test of the project's ideas on real records. CGMacros (PhysioNet, CC BY-NC-SA 4.0;
Das et al. 2025, Sci Data) has 45 adults (healthy, prediabetes, type 2 diabetes),
about 10 free-living days each, two CGMs (Abbott Libre Pro, Dexcom G6), Fitbit, a baseline
blood panel and meals logged with calories, carbohydrate, protein, fat and fibre.
Breakfasts and lunches were mostly standardised meals given by the study; dinners and
snacks were free-living. There are no food codes or food items, only meal-level macros.

What it does:

1. Builds one row per logged meal with the 2-hour incremental area under the Libre curve
   (iAUC, above the mean of the 15 minutes before the meal) and the peak rise.
2. Person level: HbA1c group (<5.7, 5.7-6.4, >=6.5 %), Harris-Benedict (Roza-Shizgal)
   and Mifflin-St Jeor energy requirement, and Fitbit measured daily expenditure.
3. H5/H6: absolute grams vs intake relative to energy requirement vs grams with body weight
   as a covariate, by AIC of mixed models and by leave-one-person-out prediction.
4. Meal similarity: a meal distance built like the food distance in `foodspace`
   ("how much": z-scored log amounts; "what kind": CLR of energy shares, Aitchison),
   then: do similar meals give similar responses, and more so within a person (H7)?
5. Personal prediction of held-out meals: population model, exact-meal match ("every
   meal unique"), similar meals of the same person, similar meals of the same subgroup.
6. Do person phenotypes from free-living meals agree with those from standardised meals?

Raw data stay in data/raw/cgmacros/ (git-ignored). Only aggregate tables are written to
results/cgmacros/. Run:  python scripts/cgmacros_analysis.py [--raw DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

WINDOW = 120          # minutes after the meal
BASE = 15             # minutes before the meal for the baseline
MACROS = ["carbs", "protein", "fat", "fiber"]
KCAL_PER_G = {"carbs": 4.0, "protein": 4.0, "fat": 9.0, "fiber": 2.0}
SEED = 0


# ---------------------------------------------------------------- loading

def load(raw: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for f in sorted(glob.glob(os.path.join(raw, "CGMacros-*.csv"))):
        d = pd.read_csv(f)
        d["subject"] = int(os.path.basename(f)[9:12])
        frames.append(d)
    if not frames:
        raise SystemExit(f"No CGMacros-*.csv files in {raw}. See scripts/fetch_cgmacros.sh.")
    cgm = pd.concat(frames, ignore_index=True)
    cgm.columns = [c.strip() for c in cgm.columns]
    cgm["Timestamp"] = pd.to_datetime(cgm["Timestamp"])
    bio = pd.read_csv(os.path.join(raw, "bio.csv"))
    bio.columns = [c.strip() for c in bio.columns]
    return cgm, bio


def persons(bio: pd.DataFrame, cgm: pd.DataFrame) -> pd.DataFrame:
    p = pd.DataFrame({
        "subject": bio["subject"],
        "age": bio["Age"],
        "male": (bio["Gender"].str.upper().str[0] == "M").astype(int),
        "weight_kg": bio["Body weight"] * 0.45359237,   # recorded in pounds
        "height_cm": bio["Height"] * 2.54,             # recorded in inches
        "bmi": bio["BMI"],
        "a1c": bio["A1c PDL (Lab)"],
    })
    p["group"] = pd.cut(p["a1c"], [0, 5.7, 6.5, 99], right=False,
                        labels=["healthy", "prediabetes", "T2D"]).astype(str)
    w, h, a, m = p.weight_kg, p.height_cm, p.age, p.male
    # Harris-Benedict as revised by Roza & Shizgal (1984)
    p["bmr_hb"] = np.where(m == 1, 88.362 + 13.397 * w + 4.799 * h - 5.677 * a,
                           447.593 + 9.247 * w + 3.098 * h - 4.330 * a)
    p["bmr_msj"] = 10 * w + 6.25 * h - 5 * a + np.where(m == 1, 5, -161)
    pal = 1.5  # no individual activity level is recorded; a common sedentary-light value
    p["eer_hb"] = p.bmr_hb * pal
    p["eer_msj"] = p.bmr_msj * pal
    # Fitbit total energy expenditure per full day (>= 1400 of 1440 minutes recorded)
    day = cgm.assign(day=cgm.Timestamp.dt.date).groupby(["subject", "day"])["Calories (Activity)"]
    full = day.agg(["count", "sum"])
    full = full[full["count"] >= 1400]
    p = p.merge(full.groupby("subject")["sum"].median().rename("tee_fitbit").reset_index(),
                on="subject", how="left")
    return p


def meals(cgm: pd.DataFrame) -> pd.DataFrame:
    out = []
    for s, d in cgm.groupby("subject"):
        d = d.sort_values("Timestamp").set_index("Timestamp")
        g = d["Libre GL"].interpolate(limit=20)
        gd = d["Dexcom GL"].interpolate(limit=20)
        # entries with 0 kcal (water, black coffee, empty logs) are not meals
        ml = d[d["Meal Type"].notna() & (d["Calories"].fillna(0) > 0)]
        times = ml.index
        for i, (t, r) in enumerate(ml.iterrows()):
            pre = g[t - pd.Timedelta(minutes=BASE): t]
            post = g[t: t + pd.Timedelta(minutes=WINDOW)]
            rec = {
                "subject": s, "time": t,
                "meal_type": str(r["Meal Type"]).lower().strip().rstrip(" 1").rstrip("s"),
                "kcal": r["Calories"], "carbs": r["Carbs"], "protein": r["Protein"],
                "fat": r["Fat"], "fiber": r["Fiber"],
            }
            rec["coverage"] = post.notna().sum() / (WINDOW + 1)
            if len(pre.dropna()) and rec["coverage"] >= 0.9:
                base = pre.mean()
                inc = (post - base).clip(lower=0).fillna(0).to_numpy()
                rec["base"] = base
                rec["iauc"] = np.trapezoid(inc, dx=1.0)
                rec["peak_rise"] = (post - base).max()
            # Dexcom as a second sensor
            pre_d = gd[t - pd.Timedelta(minutes=BASE): t]
            post_d = gd[t: t + pd.Timedelta(minutes=WINDOW)]
            if len(pre_d.dropna()) and post_d.notna().mean() >= 0.9:
                rec["iauc_dexcom"] = np.trapezoid((post_d - pre_d.mean()).clip(lower=0).fillna(0), dx=1.0)
            nxt = times[i + 1] if i + 1 < len(times) else None
            prv = times[i - 1] if i > 0 else None
            rec["clean"] = nxt is None or (nxt - t) > pd.Timedelta(minutes=WINDOW)
            rec["hours_since_prev"] = (t - prv).total_seconds() / 3600 if prv is not None else np.nan
            if rec["hours_since_prev"] > 24:   # a gap in logging, not a 1-day fast
                rec["hours_since_prev"] = np.nan
            prev4 = ml[(ml.index < t) & (ml.index >= t - pd.Timedelta(hours=4))]
            rec["kcal_prev4h"] = prev4["Calories"].sum()
            rec["hour"] = t.hour + t.minute / 60
            out.append(rec)
    m = pd.DataFrame(out)
    m["meal_type"] = m["meal_type"].replace({"snack": "snack", "snacks": "snack"})
    # Fibre entries above 60 g in one meal are logging errors (one participant logs
    # hundreds of grams); treat them as missing.
    m.loc[m["fiber"] > 60, "fiber"] = np.nan
    m["fiber_ok"] = m["fiber"].notna()
    m["fiber"] = m["fiber"].fillna(m.groupby("meal_type")["fiber"].transform("median"))
    # A macro profile shared by at least 20 people is one of the study's standardised meals.
    m["key"] = m[MACROS].round(0).astype(int).astype(str).agg("|".join, axis=1)
    shared = m.groupby("key")["subject"].nunique()
    m["standard"] = m["key"].map(shared) >= 20
    m["y"] = np.sqrt(m["iauc"])   # 7% of meals have iAUC 0; sqrt keeps them and tames the skew
    return m


# ------------------------------------------------------------ meal distance

def meal_distance(m: pd.DataFrame, delta: float = 1e-3) -> dict[str, np.ndarray]:
    """Composition ("how much") and fraction ("what kind") distances between meals,
    following the food distances in foodspace: both scaled to 0-1 by their maximum,
    fused as their mean."""
    from foodspace.fractions import aitchison_distance, clr

    X = np.log1p(m[MACROS].to_numpy(float))
    Z = (X - X.mean(0)) / X.std(0)
    Dc = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    E = pd.DataFrame({k: m[k].to_numpy(float) * KCAL_PER_G[k] for k in MACROS})
    S = E.div(E.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.25)
    Df = aitchison_distance(clr(S, delta=delta))
    Dc = Dc / Dc.max()
    Df = Df / Df.max()
    return {"composition": Dc, "fraction": Df, "fused": (Dc + Df) / 2}


# ---------------------------------------------------------------- modelling

def design(df: pd.DataFrame, scale: str) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for k in MACROS:
        if scale == "abs":
            X[k] = df[k]
        else:  # % of the daily energy requirement from this macro
            X[k] = 100 * df[k] * KCAL_PER_G[k] / df[scale]
    for g in ["prediabetes", "T2D"]:
        X[g] = (df["group"] == g).astype(float)
    return X


def model_set(df: pd.DataFrame):
    sets = {
        "groups only": lambda d: design(d, "abs")[["prediabetes", "T2D"]],
        "absolute grams": lambda d: design(d, "abs"),
        "grams + body weight": lambda d: design(d, "abs").assign(weight_kg=d["weight_kg"]),
        "grams + body weight + sex + age": lambda d: design(d, "abs").assign(
            weight_kg=d["weight_kg"], male=d["male"], age=d["age"]),
        "% of EER (Harris-Benedict)": lambda d: design(d, "eer_hb"),
        "% of EER (Mifflin-St Jeor)": lambda d: design(d, "eer_msj"),
        "% of Fitbit expenditure": lambda d: design(d, "tee_fitbit"),
    }
    return sets


def fit_mixed(y, X, groups):
    import statsmodels.api as sm
    Xs = (X - X.mean()) / X.std().replace(0, 1)   # scaling only helps the optimiser
    Xc = sm.add_constant(Xs, has_constant="add")
    md = sm.MixedLM(y, Xc, groups=groups)
    return _fit(md, "mixed model")


def fit_formula(formula, df):
    import statsmodels.formula.api as smf
    return _fit(smf.mixedlm(formula, df, groups=df["subject"]), formula)


def _fit(md, label):
    # statsmodels' default lbfgs sometimes returns an infinite likelihood here; bfgs and cg
    # agree with each other, so try those first and keep the first finite fit.
    for method in ["bfgs", "cg", "lbfgs", "powell"]:
        try:
            r = md.fit(reml=False, method=method)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if np.isfinite(r.llf) and r.converged:
            return r
    raise RuntimeError("mixed model failed: " + label)


def loso_r2(df, X, y):
    """Leave one person out: OLS on everyone else, predict the left-out person."""
    pred = np.full(len(y), np.nan)
    Xa = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
    subj = df["subject"].to_numpy()
    for s in np.unique(subj):
        tr = subj != s
        beta, *_ = np.linalg.lstsq(Xa[tr], y[tr], rcond=None)
        pred[~tr] = Xa[~tr] @ beta
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def compare_scalings(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subset, df in [("all clean meals", m), ("standardised meals", m[m.standard]),
                       ("free-living meals", m[~m.standard])]:
        df = df.dropna(subset=["y", "tee_fitbit"])
        y = df["y"].to_numpy()
        for name, f in model_set(df).items():
            X = f(df)
            r = fit_mixed(y, X, df["subject"].to_numpy())
            k = X.shape[1] + 3  # + intercept, random-intercept variance, residual variance
            rows.append({"meals": subset, "model": name, "n_meals": len(df),
                         "n_people": df.subject.nunique(),
                         "AIC": -2 * r.llf + 2 * k, "LOSO_R2": loso_r2(df, X, y)})
    out = pd.DataFrame(rows)
    out["dAIC"] = out["AIC"] - out.groupby("meals")["AIC"].transform("min")
    return out


def subgroup_slopes(m: pd.DataFrame) -> pd.DataFrame:
    """Carbohydrate dose-response by HbA1c group (mixed model, random intercept)."""
    import statsmodels.formula.api as smf
    rows = []
    for subset, df in [("all clean meals", m), ("standardised meals", m[m.standard]),
                       ("free-living meals", m[~m.standard])]:
        df = df.dropna(subset=["y"]).copy()
        df["carbs10"] = df["carbs"] / 10
        r = fit_formula("y ~ C(group, Treatment('healthy')) * carbs10 + protein + fat + fiber", df)
        for g in ["healthy", "prediabetes", "T2D"]:
            sub = df[df.group == g]
            rr = fit_formula("y ~ carbs10 + protein + fat + fiber", sub)
            ci = rr.conf_int().loc["carbs10"]
            rows.append({"meals": subset, "group": g, "n_people": sub.subject.nunique(),
                         "n_meals": len(sub),
                         "median_iAUC": sub["iauc"].median(),
                         "sqrt_iAUC_per_10g_carbs": rr.params["carbs10"],
                         "ci_low": ci[0], "ci_high": ci[1],
                         "protein_per_g": rr.params["protein"], "fat_per_g": rr.params["fat"],
                         "fiber_per_g": rr.params["fiber"]})
        inter = [c for c in r.params.index if ":carbs10" in c]
        for c in inter:
            rows.append({"meals": subset, "group": "interaction " + c.split("[T.")[1].split("]")[0],
                         "sqrt_iAUC_per_10g_carbs": r.params[c], "p": r.pvalues[c]})
    return pd.DataFrame(rows)


# ---------------------------------------------- similarity and responses

def pair_table(m: pd.DataFrame, D: np.ndarray, rng) -> pd.DataFrame:
    """|difference in response| against meal distance, within vs between people."""
    y = m["y"].to_numpy()
    s = m["subject"].to_numpy()
    iu = np.triu_indices(len(m), 1)
    dd = D[iu]
    dy = np.abs(y[iu[0]] - y[iu[1]])
    same = s[iu[0]] == s[iu[1]]
    bins = np.quantile(dd, np.linspace(0, 1, 11))
    bins[0] = -1e-9
    b = np.digitize(dd, bins[1:-1])
    rows = []
    for k in range(10):
        for w, lab in [(True, "same person"), (False, "different people")]:
            sel = (b == k) & (same == w)
            rows.append({"distance_decile": k + 1, "pairs": lab,
                         "distance_mid": float(np.median(dd[b == k])),
                         "n_pairs": int(sel.sum()),
                         "mean_abs_diff_sqrt_iAUC": float(dy[sel].mean()) if sel.any() else np.nan})
    return pd.DataFrame(rows), dd, dy, same


def mantel_within(m, D, n_perm=999, rng=None):
    """Spearman correlation of meal distance and response difference among pairs of meals
    of the same person; null by permuting responses within each person."""
    y = m["y"].to_numpy()
    s = m["subject"].to_numpy()
    iu = np.triu_indices(len(m), 1)
    same = s[iu[0]] == s[iu[1]]
    a, b = iu[0][same], iu[1][same]
    dd = D[a, b]

    def rho(yy):
        return stats.spearmanr(dd, np.abs(yy[a] - yy[b]))[0]

    obs = rho(y)
    null = []
    idx = {k: np.where(s == k)[0] for k in np.unique(s)}
    for _ in range(n_perm):
        yp = y.copy()
        for k, ii in idx.items():
            yp[ii] = y[rng.permutation(ii)]
        null.append(rho(yp))
    null = np.array(null)
    return obs, (1 + np.sum(null >= obs)) / (n_perm + 1)


def icc_repeats(m: pd.DataFrame) -> pd.DataFrame:
    """Test-retest of identical standardised meals eaten twice by the same person."""
    rows = []
    for key, df in m[m.standard].groupby("key"):
        rep = df.groupby("subject").filter(lambda g: len(g) >= 2)
        if rep.subject.nunique() < 10:
            continue
        rep = rep.sort_values("time").groupby("subject").head(2)
        a = rep.groupby("subject")["y"].agg(list)
        x = np.array([v[0] for v in a]); z = np.array([v[1] for v in a])
        # ICC(1): one-way random effects
        k = 2
        grand = np.concatenate([x, z]).mean()
        msb = k * np.var((x + z) / 2, ddof=1)
        msw = np.sum((x - (x + z) / 2) ** 2 + (z - (x + z) / 2) ** 2) / len(x)
        rows.append({"meal": key, "kcal": df.kcal.median(), "n_people": len(x),
                     "test_retest_r": stats.pearsonr(x, z)[0],
                     "ICC1": (msb - msw) / (msb + (k - 1) * msw)})
    return pd.DataFrame(rows)


def personal_prediction(m: pd.DataFrame, D: np.ndarray, k: int = 5) -> pd.DataFrame:
    """Predict each meal's sqrt iAUC without using that meal, several ways."""
    y = m["y"].to_numpy()
    s = m["subject"].to_numpy()
    grp = m["group"].to_numpy()
    key = m["key"].to_numpy()
    n = len(m)
    X = np.column_stack([np.ones(n), m[MACROS].to_numpy(float),
                         (grp == "prediabetes").astype(float), (grp == "T2D").astype(float)])
    pop = np.full(n, np.nan)
    for p in np.unique(s):
        tr = s != p
        beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        pop[~tr] = X[~tr] @ beta
    res = y - pop  # residual against a model that never saw this person
    preds = {name: np.full(n, np.nan) for name in [
        "population (macros + group)", "population + person offset",
        "exact same meal, same person", "similar meals, same person (kNN)",
        "population + similar meals' residuals, same person",
        "similar meals, same subgroup (kNN, other people)"]}
    for i in range(n):
        me = (s == s[i]); me[i] = False
        preds["population (macros + group)"][i] = pop[i]
        if me.sum() >= 3:
            preds["population + person offset"][i] = pop[i] + res[me].mean()
            ex = me & (key == key[i])
            if ex.any():
                preds["exact same meal, same person"][i] = y[ex].mean()
            idx = np.where(me)[0]
            nn = idx[np.argsort(D[i, idx])[:k]]
            preds["similar meals, same person (kNN)"][i] = y[nn].mean()
            preds["population + similar meals' residuals, same person"][i] = pop[i] + res[nn].mean()
        other = (s != s[i]) & (grp == grp[i])
        idx = np.where(other)[0]
        nn = idx[np.argsort(D[i, idx])[:k * 5]]
        preds["similar meals, same subgroup (kNN, other people)"][i] = y[nn].mean()
    rows = []
    has_exact = ~np.isnan(preds["exact same meal, same person"])
    for subset, sel in [("all", np.ones(n, bool)), ("standardised", m.standard.to_numpy()),
                        ("free-living", ~m.standard.to_numpy()),
                        ("meals with an exact repeat (all methods on the same meals)", has_exact)]:
        for name, p in preds.items():
            ok = sel & ~np.isnan(p)
            if ok.sum() < 20:
                rows.append({"meals": subset, "method": name, "coverage": ok.sum() / sel.sum()})
                continue
            rows.append({"meals": subset, "method": name,
                         "coverage": ok.sum() / sel.sum(),
                         "r": stats.pearsonr(p[ok], y[ok])[0],
                         "RMSE": float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2)))})
    return pd.DataFrame(rows), preds


def bootstrap_gain(m, preds, a, b, subset, n_boot=2000, rng=None):
    """Difference in correlation (method a minus method b) with a CI from resampling people."""
    y = m["y"].to_numpy(); s = m["subject"].to_numpy()
    sel = subset & ~np.isnan(preds[a]) & ~np.isnan(preds[b])
    people = np.unique(s[sel])
    rows_by = {p: np.where(sel & (s == p))[0] for p in people}

    def diff(idx):
        return np.corrcoef(preds[a][idx], y[idx])[0, 1] - np.corrcoef(preds[b][idx], y[idx])[0, 1]

    obs = diff(np.where(sel)[0])
    bs = [diff(np.concatenate([rows_by[p] for p in rng.choice(people, len(people))]))
          for _ in range(n_boot)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"a": a, "b": b, "r_diff": obs, "ci_low": lo, "ci_high": hi, "n_meals": int(sel.sum())}


def knn_sensitivity(m, Ds, ks=(1, 3, 5, 10, 20)):
    """Free-living meals, predicted from the same person's k most similar other meals,
    for each distance view; plus random meals of the same person as the null."""
    y = m["y"].to_numpy(); s = m["subject"].to_numpy()
    free = ~m["standard"].to_numpy()
    rng = np.random.default_rng(SEED)
    rows = []
    for view, D in list(Ds.items()) + [("random meals (null)", None)]:
        for k in ks:
            p = np.full(len(m), np.nan)
            for i in np.where(free)[0]:
                idx = np.where((s == s[i]) & (np.arange(len(m)) != i))[0]
                if len(idx) < 3:
                    continue
                nn = rng.choice(idx, min(k, len(idx)), replace=False) if D is None \
                    else idx[np.argsort(D[i, idx])[:k]]
                p[i] = y[nn].mean()
            ok = ~np.isnan(p)
            rows.append({"distance": view, "k": k, "r": stats.pearsonr(p[ok], y[ok])[0],
                         "RMSE": float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2)))})
    return pd.DataFrame(rows)


def phenotype_agreement(m: pd.DataFrame, cross_sensor: bool = False) -> dict:
    """Person offset (mean residual vs population model) from standardised meals vs from
    free-living meals, per person."""
    n = len(m)
    y = m["y"].to_numpy(); s = m["subject"].to_numpy()
    grp = m["group"].to_numpy()
    X = np.column_stack([np.ones(n), m[MACROS].to_numpy(float)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)   # no group term: the offset carries it
    m = m.assign(res=y - X @ beta)
    if cross_sensor:
        # free-living offsets from the Dexcom instead, so a person-level calibration bias of
        # one sensor cannot make the two offsets agree
        ok = m["iauc_dexcom"].notna().to_numpy()
        yd = np.sqrt(m["iauc_dexcom"].to_numpy())
        bd, *_ = np.linalg.lstsq(X[ok], yd[ok], rcond=None)
        m = m.assign(res_d=yd - X @ bd)
    a = m[m.standard].groupby("subject")["res"].agg(["mean", "size"])
    b = m[~m.standard].groupby("subject")["res_d" if cross_sensor else "res"].agg(["mean", "size"])
    j = a.join(b, lsuffix="_std", rsuffix="_free").query("size_std >= 4 and size_free >= 4")
    g = m.groupby("subject")["group"].first().reindex(j.index)
    r = stats.spearmanr(j.mean_std, j.mean_free)[0]
    out = {"n_people": len(j), "spearman_standard_vs_free": r}
    # same, after removing the group mean (is there a person signal beyond HbA1c group?)
    js = j.assign(group=g)
    for c in ["mean_std", "mean_free"]:
        js[c] = js[c] - js.groupby("group")[c].transform("mean")
    out["spearman_within_group"] = stats.spearmanr(js.mean_std, js.mean_free)[0]
    out["p_within_group"] = stats.spearmanr(js.mean_std, js.mean_free)[1]
    return out


# ------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    ap.add_argument("--raw", default=os.path.join(here, "..", "data", "raw", "cgmacros"))
    ap.add_argument("--out", default=os.path.join(here, "..", "results", "cgmacros"))
    a = ap.parse_args(argv)
    import statsmodels.api  # noqa: F401  (it resets warning filters on import)
    warnings.filterwarnings("ignore")
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    cgm, bio = load(a.raw)
    P = persons(bio, cgm)
    M_all = meals(cgm).merge(P, on="subject")
    summary = {
        "people": int(P.subject.nunique()),
        "people_per_group": P.group.value_counts().to_dict(),
        "meals_logged": int(len(M_all)),
        "meals_by_type": M_all.meal_type.value_counts().to_dict(),
        "meals_standardised": int(M_all.standard.sum()),
        "standardised_profiles": int(M_all.loc[M_all.standard, "key"].nunique()),
        "meals_with_glucose_window": int(M_all.iauc.notna().sum()),
        "meals_clean_window": int((M_all.iauc.notna() & M_all.clean).sum()),
        "fiber_entries_set_missing": int((~M_all.fiber_ok).sum()),
        "libre_vs_dexcom_spearman": float(stats.spearmanr(
            M_all.iauc, M_all.iauc_dexcom, nan_policy="omit")[0]),
        "eer_hb_median": float(P.eer_hb.median()),
        "tee_fitbit_median": float(P.tee_fitbit.median()),
        "eer_hb_vs_fitbit_spearman": float(stats.spearmanr(P.eer_hb, P.tee_fitbit, nan_policy="omit")[0]),
    }
    m = M_all[M_all.iauc.notna() & M_all.clean].reset_index(drop=True)
    summary["meals_analysed"] = int(len(m))
    summary["standard_analysed"] = int(m.standard.sum())

    # energy-state covariates
    import statsmodels.formula.api as smf
    es = m.dropna(subset=["hours_since_prev"]).copy()
    es["carbs10"] = es.carbs / 10
    es["kcal_prev4h_100"] = es.kcal_prev4h / 100
    r = fit_formula("y ~ carbs10 + protein + fat + fiber + C(group) + np.log1p(hours_since_prev)"
                    " + kcal_prev4h_100 + base", es)
    energy_state = pd.DataFrame({"coef": r.params, "p": r.pvalues}).drop(index="Group Var")

    scal = compare_scalings(m)
    slopes = subgroup_slopes(m)
    icc = icc_repeats(m)
    Ds = meal_distance(m)
    D = Ds["fused"]
    pairs, dd, dy, same = pair_table(m, D, rng)
    mantel = {v: mantel_within(m, Dv, rng=rng) for v, Dv in Ds.items()}
    pred, preds = personal_prediction(m, D)
    free_sel = ~m["standard"].to_numpy()
    gains = pd.DataFrame([
        bootstrap_gain(m, preds, "similar meals, same person (kNN)", "population + person offset",
                       free_sel, rng=rng),
        bootstrap_gain(m, preds, "similar meals, same person (kNN)", "population (macros + group)",
                       free_sel, rng=rng),
        bootstrap_gain(m, preds, "population + person offset", "population (macros + group)",
                       free_sel, rng=rng),
        bootstrap_gain(m, preds, "similar meals, same subgroup (kNN, other people)",
                       "population (macros + group)", free_sel, rng=rng),
    ])
    sens = knn_sensitivity(m, Ds)
    pheno = phenotype_agreement(m)
    pheno.update({k + "_libre_std_vs_dexcom_free": v
                  for k, v in phenotype_agreement(m, cross_sensor=True).items()})

    # how many free-living meals have an exact repeat by the same person?
    free = m[~m.standard]
    rep = free.groupby(["subject", "key"]).size()
    summary["free_meals_with_exact_repeat_same_person"] = float((free.set_index(["subject", "key"]).index.map(rep) > 1).mean())
    summary["mantel_within_person"] = {v: {"rho": float(o), "p": float(p)} for v, (o, p) in mantel.items()}
    summary["phenotype_agreement"] = {k: float(v) for k, v in pheno.items()}
    # identical meals: response difference within vs between people
    st = m[m.standard]
    w, bt = [], []
    for key, df in st.groupby("key"):
        yy = df.y.to_numpy(); ss = df.subject.to_numpy()
        iu = np.triu_indices(len(df), 1)
        d = np.abs(yy[iu[0]] - yy[iu[1]])
        w.append(d[ss[iu[0]] == ss[iu[1]]]); bt.append(d[ss[iu[0]] != ss[iu[1]]])
    w = np.concatenate(w); bt = np.concatenate(bt)
    summary["identical_meal_abs_diff_within"] = float(w.mean())
    summary["identical_meal_abs_diff_between"] = float(bt.mean())
    summary["identical_meal_pairs_within"] = int(len(w))

    scal.to_csv(os.path.join(a.out, "scaling_models.csv"), index=False)
    slopes.to_csv(os.path.join(a.out, "subgroup_carb_slopes.csv"), index=False)
    icc.to_csv(os.path.join(a.out, "standard_meal_test_retest.csv"), index=False)
    pairs.to_csv(os.path.join(a.out, "response_vs_meal_distance.csv"), index=False)
    pred.to_csv(os.path.join(a.out, "personal_prediction.csv"), index=False)
    gains.to_csv(os.path.join(a.out, "prediction_gains_free_living.csv"), index=False)
    sens.to_csv(os.path.join(a.out, "knn_sensitivity_free_living.csv"), index=False)
    energy_state.to_csv(os.path.join(a.out, "energy_state_model.csv"))
    (m.groupby(["group", "meal_type", "standard"]).agg(
        n_meals=("y", "size"), n_people=("subject", "nunique"),
        median_kcal=("kcal", "median"), median_carbs=("carbs", "median"),
        median_iAUC=("iauc", "median"), median_peak_rise=("peak_rise", "median"))
     .reset_index().to_csv(os.path.join(a.out, "meal_summary.csv"), index=False))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    try:
        plot(pairs, pred, sens, a.out)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    pd.set_option("display.width", 200)
    print(json.dumps(summary, indent=2, default=str))
    for t in [scal, slopes, icc, pairs, pred, gains, sens, energy_state]:
        print(t.round(3).to_string()); print()


def plot(pairs, pred, sens, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for lab, c in [("same person", "#2a6f97"), ("different people", "#c44536")]:
        d = pairs[pairs.pairs == lab]
        ax[0].plot(d.distance_decile, d.mean_abs_diff_sqrt_iAUC, "o-", color=c, label=lab)
    ax[0].set_xlabel("meal distance decile (1 = most similar)")
    ax[0].set_ylabel("mean |difference| in sqrt iAUC")
    ax[0].set_title("Similar meals, similar responses?")
    ax[0].legend(frameon=False)
    p = pred[(pred.meals == "free-living") & pred.r.notna()]
    null = sens[(sens.distance == "random meals (null)") & (sens.k == 5)]
    p = pd.concat([p, pd.DataFrame({"method": ["5 random meals, same person (null)"],
                                    "r": null.r.to_numpy(), "coverage": [1.0]})]).sort_values("r")
    ax[1].barh(p.method, p.r, color=["#9aa5b1" if "null" in x else "#2a6f97" for x in p.method])
    for i, (r, c) in enumerate(zip(p.r, p.coverage)):
        ax[1].text(r + 0.01, i, f"{c:.0%} of meals", va="center", fontsize=8)
    ax[1].set_xlabel("correlation with observed sqrt iAUC (held-out meal)")
    ax[1].set_title("Predicting free-living meals")
    ax[1].set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "cgmacros_similarity.png"), dpi=150)


if __name__ == "__main__":
    main()
