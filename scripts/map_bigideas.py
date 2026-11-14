#!/usr/bin/env python3
"""Map the BIG IDEAs food logs (US, free text) onto the Nordic food reference.

    python scripts/map_bigideas.py --data /path/to/big-ideas-glycemic-wearable/1.1.2 \
        [--foods data/processed/foods_nordic.csv] [--out results/bigideas] [--private results/private/bigideas]

The dataset (PhysioNet, ODC-By 1.0) is not in the repository; download it yourself:
https://physionet.org/content/big-ideas-glycemic-wearable/

What it does
  1. reads the 16 food logs (src/foodspace/bigideas.py evens out their formats);
  2. maps each distinct item name to a reference food twice: by name only, and by name +
     reported composition (src/foodspace/freetext.py);
  3. reports coverage by lines and by energy, the GIFT groups reached, and how the reported
     macros compare with the reference composition of the matched foods;
  4. scores both matchers against the hand review in data/curated/bigideas_match_review.csv;
  5. glucose: incremental area under the curve 0-120 min after each meal (Dexcom), and whether
     the GIFT group of the meal's main carbohydrate source explains it beyond the grams of carbohydrate.

Writes (aggregate only) to --out:
    summary.json            every number in the report
    item_map.csv            one row per distinct item name: matched reference food, status, distances
    coverage_by_group.csv   lines and energy per GIFT group reached
    macro_agreement.csv     reported vs reference composition per matched weighed line, summarised
    glucose_by_group.csv    meal iAUC per g carbohydrate by GIFT group of the main carbohydrate source
Writes per-meal and per-line tables to --private (outside git: results/private/ is ignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foodspace import bigideas, freetext  # noqa: E402

GRAMS_PER_UNIT = {"gram": 1.0, "grams": 1.0, "g": 1.0, "ounce": 28.35, "ounces": 28.35, "oz": 28.35}
MACROS = [("protein", "PROT_g"), ("total_fat", "FAT_g"), ("avail_carb", "CHO_g"), ("sugar", "SUGAR_g"),
          ("dietary_fiber", "FIBT_g")]


def label_of(df: pd.DataFrame) -> pd.Series:
    """The app's matched description when there is one, else what the person typed."""
    return df["searched_food"].fillna(df["logged_food"])


def map_items(lines: pd.DataFrame, foods: pd.DataFrame, **kw) -> pd.DataFrame:
    """One row per distinct label; the composition check uses the label's summed macros over all its lines."""
    nut = ["protein", "total_fat", "total_carb", "sugar", "dietary_fiber", "calorie"]
    items = lines.groupby("label")[nut].sum(min_count=1).reset_index()
    items["n_lines"] = lines.groupby("label").size().to_numpy()
    items["kcal_total"] = items["calorie"]
    m = freetext.match_items(items, foods, **kw)
    return items.join(m)


def coverage(lines: pd.DataFrame, how: str) -> dict:
    st = lines[f"{how}_status"]
    kcal = lines["calorie"].fillna(0)
    return {"lines": {k: round(float(v), 4) for k, v in st.value_counts(normalize=True).items()},
            "energy": {k: round(float(kcal[st == k].sum() / kcal.sum()), 4) for k in st.unique()}}


def review_scores(items: pd.DataFrame, review: pd.DataFrame) -> dict:
    """Agreement with the hand review: C = right food or a close stand-in, G = right GIFT group only, W = wrong.
    A verdict counts only while the reviewed food is still what the matcher picks today."""
    now = items[["label", "hybrid_food_id", "name_food_id", "hybrid_status", "name_status"]].add_prefix("now_")
    out = {}
    for rs, r in review.groupby("review_set"):
        r = r[r["hybrid_verdict"] != "U"].merge(now, left_on="label", right_on="now_label", how="left")
        res = {"n_items": int(len(r)), "matcher_version_reviewed": r["matcher_version"].iloc[0]}
        for how, vcol, idcol in (("hybrid", "hybrid_verdict", "hybrid_food_id"),
                                 ("name", "name_only_verdict", "name_only_food_id")):
            same = r[f"now_{how}_food_id"] == r[idcol]
            v = r.loc[same, vcol]
            acc = r.loc[same, f"now_{how}_status"] == "matched"
            res[how] = {"items_scored": int(same.sum()),
                        "top_candidate_right_food": round(float((v == "C").mean()), 3),
                        "top_candidate_right_group": round(float((v != "W").mean()), 3),
                        "accepted": round(float(acc.mean()), 3),
                        "precision_food_among_accepted": round(float((v[acc] == "C").mean()), 3),
                        "precision_group_among_accepted": round(float((v[acc] != "W").mean()), 3),
                        "recall_of_right_food": round(float(((v == "C") & acc).sum() / max((v == "C").sum(), 1)), 3)}
        res["failure_types"] = r.loc[r["failure_type"].notna() & (r["failure_type"] != ""), "failure_type"] \
            .value_counts().to_dict()
        out[rs] = res
    return out


def macro_agreement(lines: pd.DataFrame, foods: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Matched lines logged by weight (grams or ounces, not drinks): reported per 100 g vs the reference food."""
    ref = foods.set_index("food_id")
    unit = lines["unit"].str.lower().str.strip()
    g = lines["amount"] * unit.map(GRAMS_PER_UNIT)
    ok = (lines["hybrid_status"] == "matched") & g.gt(0) & (lines["hybrid_gift_group"] != "Beverages")
    d = lines[ok].copy()
    d["grams"] = g[ok]
    d["avail_carb"] = d["total_carb"] - d["dietary_fiber"].fillna(0)
    rows = []
    for rec, col in MACROS + [("calorie", "kcal")]:
        rep = d[rec] / d["grams"] * 100
        refv = d["hybrid_food_id"].map(ref[col]) if col != "kcal" else d["hybrid_food_id"].map(
            pd.Series(freetext.reference_energy(foods), index=foods["food_id"]))
        diff = rep - refv
        okv = rep.notna() & refv.notna()
        rows.append({"component": rec, "n_lines": int(okv.sum()),
                     "median_reported_per100g": round(float(rep[okv].median()), 2),
                     "median_reference_per100g": round(float(refv[okv].median()), 2),
                     "median_abs_diff_per100g": round(float(diff[okv].abs().median()), 2),
                     "spearman": round(float(pd.Series(rep[okv]).corr(pd.Series(refv[okv]), method="spearman")), 3),
                     "share_within_25pct_or_1g": round(float(((diff[okv].abs() <= np.maximum(0.25 * refv[okv].abs(), 1.0))).mean()), 3)})
    # portion-free view on every matched line: energy-share profile distance
    prof = lines.loc[lines["hybrid_status"] == "matched", "hybrid_profile_dist"].dropna()
    summary = {"weighed_matched_lines": int(len(d)),
               "profile_distance_matched_items": {"median": round(float(prof.median()), 3),
                                                  "p75": round(float(prof.quantile(.75)), 3),
                                                  "p90": round(float(prof.quantile(.9)), 3)}}
    return pd.DataFrame(rows), summary


# -- glucose -------------------------------------------------------------------------------
def meal_iauc(meals: pd.DataFrame, cgm: pd.DataFrame, window: int = 120, gap: int = 120) -> pd.DataFrame:
    """iAUC (mg/dL x min, trapezoid above the pre-meal baseline = mean of the 30 min before)
    for meals with no other meal in the next `gap` minutes and >= 80% of CGM readings present."""
    out = []
    for p, m in meals.groupby("participant"):
        c = cgm[cgm["participant"] == p].set_index("time")["glucose"].sort_index()
        t = m["time_begin"].sort_values().to_numpy()
        for idx, row in m.iterrows():
            t0 = row["time_begin"]
            nxt = t[t > t0]
            if len(nxt) and (nxt[0] - np.datetime64(t0)) < np.timedelta64(gap, "m"):
                continue
            base = c[t0 - pd.Timedelta(minutes=30):t0]
            post = c[t0:t0 + pd.Timedelta(minutes=window)]
            if len(base) < 3 or len(post) < 0.8 * window / 5:
                continue
            x = (post.index - t0).total_seconds().to_numpy() / 60
            y = np.clip(post.to_numpy() - base.mean(), 0, None)
            out.append({"meal": idx, "iauc": float(np.trapezoid(y, x)), "baseline": float(base.mean()),
                        "peak_rise": float((post - base.mean()).max())})
    return pd.DataFrame(out).set_index("meal")


def glucose_analysis(lines: pd.DataFrame, cgm: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    lines = lines.copy()
    lines["avail_carb"] = (lines["total_carb"] - lines["dietary_fiber"].fillna(0)).clip(lower=0)
    lines = lines[lines["time_begin"].notna()]
    meals = (lines.groupby(["participant", "time_begin"])
             .agg(carb=("avail_carb", "sum"), sugar=("sugar", "sum"), fat=("total_fat", "sum"),
                  protein=("protein", "sum"), kcal=("calorie", "sum"), n_items=("line_id", "size"))
             .reset_index())
    # main carbohydrate source: the line carrying most available carbohydrate; its GIFT group if matched
    top = lines.sort_values("avail_carb", ascending=False).drop_duplicates(["participant", "time_begin"])
    top = top.assign(main_group=np.where(top["hybrid_status"] == "matched", top["hybrid_gift_group"], "unmatched"))
    meals = meals.merge(top[["participant", "time_begin", "main_group", "label"]], on=["participant", "time_begin"])
    meals = meals.join(meal_iauc(meals, cgm))
    m = meals.dropna(subset=["iauc"])
    m = m[m["carb"] >= 10].copy()
    m["iauc_per_g_carb"] = m["iauc"] / m["carb"]
    by = (m.groupby("main_group")
          .agg(meals=("iauc", "size"), people=("participant", "nunique"),
               median_carb_g=("carb", "median"), median_sugar_share=("sugar", lambda s: float(np.nan)),
               median_iauc=("iauc", "median"), median_iauc_per_g_carb=("iauc_per_g_carb", "median"))
          .sort_values("meals", ascending=False))
    by["median_sugar_share"] = m.assign(ss=m["sugar"] / m["carb"]).groupby("main_group")["ss"].median()
    by = by.round(2)

    # does the main-source group explain iAUC beyond carbohydrate grams and the person?
    import statsmodels.formula.api as smf
    d = m.copy()
    d["log_iauc"] = np.log1p(d["iauc"])
    keep = d["main_group"].map(d["main_group"].value_counts()) >= 8
    d = d[keep]
    base = smf.ols("log_iauc ~ np.log1p(carb) + np.log1p(fat) + np.log1p(protein) + C(participant)", d).fit()
    full = smf.ols("log_iauc ~ np.log1p(carb) + np.log1p(fat) + np.log1p(protein) + C(participant) + C(main_group)", d).fit()
    sug = smf.ols("log_iauc ~ np.log1p(carb) + np.log1p(fat) + np.log1p(protein) + C(participant) + I(sugar / carb)", d).fit()
    f, pval, df = full.compare_f_test(base)
    stats = {"meals_with_cgm_window": int(meals["iauc"].notna().sum()), "meals_analysed_carb_ge_10g": int(len(m)),
             "meals_in_model": int(len(d)), "groups_in_model": int(d["main_group"].nunique()),
             "r2_carb_fat_protein_person": round(float(base.rsquared), 3),
             "r2_plus_main_source_group": round(float(full.rsquared), 3),
             "f_test_group": {"F": round(float(f), 2), "p": float(f"{pval:.3g}"), "df": int(df)},
             "r2_plus_sugar_share_instead": round(float(sug.rsquared), 3),
             "spearman_iauc_vs_carb": round(float(m["iauc"].corr(m["carb"], method="spearman")), 3)}
    return by, stats, meals


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with the BIG IDEAs files (as downloaded, or flat)")
    ap.add_argument("--foods", default="data/processed/foods_nordic.csv")
    ap.add_argument("--review", default="data/curated/bigideas_match_review.csv")
    ap.add_argument("--out", default="results/bigideas")
    ap.add_argument("--private", default="results/private/bigideas")
    ap.add_argument("--no-glucose", action="store_true")
    args = ap.parse_args(argv)
    out, priv = Path(args.out), Path(args.private)
    out.mkdir(parents=True, exist_ok=True)
    priv.mkdir(parents=True, exist_ok=True)

    foods = pd.read_csv(args.foods, low_memory=False)
    foods = foods[foods["gift_group"].notna() & (foods["gift_group"] != "unassigned")].reset_index(drop=True)
    lines = bigideas.read_food_logs(args.data)
    lines["label"] = label_of(lines)

    items = map_items(lines, foods)
    items_plain = map_items(lines, foods, translate=False)  # ablation: no US -> Nordic wording table
    lines = lines.merge(items.drop(columns=["protein", "total_fat", "total_carb", "sugar", "dietary_fiber", "calorie"]),
                        on="label", how="left")

    cols = ["label", "n_lines", "kcal_total", "clean_label", "head"] + [
        f"{h}_{c}" for h in ("hybrid", "name") for c in ("food_id", "name", "gift_group", "gift_subgroup",
                                                         "name_sim", "profile_dist", "status")]
    items[cols].sort_values("n_lines", ascending=False).round(3).to_csv(out / "item_map.csv", index=False)

    matched = lines["hybrid_status"] == "matched"
    cov = (lines[matched].groupby("hybrid_gift_group")
           .agg(lines=("line_id", "size"), kcal=("calorie", "sum"), items=("label", "nunique"))
           .sort_values("kcal", ascending=False))
    cov["energy_share_of_matched"] = (cov["kcal"] / cov["kcal"].sum()).round(3)
    cov.round(1).to_csv(out / "coverage_by_group.csv")

    review = pd.read_csv(args.review) if Path(args.review).exists() else None
    ma, ma_summary = macro_agreement(lines, foods)
    ma.to_csv(out / "macro_agreement.csv", index=False)

    summary = {
        "dataset": "BIG IDEAs Lab Glycemic Variability and Wearable Device Data 1.1.2 (PhysioNet, ODC-By 1.0)",
        "reference": args.foods, "reference_foods": int(len(foods)),
        "participants": int(lines["participant"].nunique()), "lines": int(len(lines)),
        "distinct_item_names": int(lines["label"].nunique()),
        "lines_with_app_description": round(float(lines["searched_food"].notna().mean()), 3),
        "coverage_hybrid": coverage(lines, "hybrid"), "coverage_name_only": coverage(lines, "name"),
        "items_matched": {"hybrid": round(float((items["hybrid_status"] == "matched").mean()), 3),
                          "name_only": round(float((items["name_status"] == "matched").mean()), 3),
                          "hybrid_without_wording_table": round(float((items_plain["hybrid_status"] == "matched").mean()), 3)},
        "hybrid_changed_name_only_choice": round(float((items["hybrid_food_id"] != items["name_food_id"]).mean()), 3),
        "macro_agreement": ma_summary,
    }
    if review is not None:
        summary["hand_review"] = review_scores(items, review)

    lines.to_csv(priv / "lines_mapped.csv", index=False)
    if not args.no_glucose:
        cgm = bigideas.read_dexcom(args.data)
        by, gstats, meals = glucose_analysis(lines, cgm)
        by.to_csv(out / "glucose_by_group.csv")
        meals.to_csv(priv / "meals_iauc.csv", index=False)
        summary["glucose"] = gstats
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
