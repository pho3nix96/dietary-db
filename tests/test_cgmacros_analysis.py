"""Tests for the CGMacros meal-response analysis on small synthetic CGM traces."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("cga", ROOT / "scripts" / "cgmacros_analysis.py")
cga = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cga)


def trace(subject, meals, minutes=600, base=100.0):
    """One-minute CGM trace with a triangular 60-minute rise of `rise` mg/dL after each meal."""
    t = pd.date_range("2024-01-01 06:00", periods=minutes, freq="min")
    g = np.full(minutes, base)
    rows = {"Meal Type": [None] * minutes, "Calories": [np.nan] * minutes}
    for k in ["Carbs", "Protein", "Fat", "Fiber"]:
        rows[k] = [np.nan] * minutes
    for start, rise, macros in meals:
        bump = np.interp(np.arange(120), [0, 60, 120], [0, rise, 0])
        g[start:start + 120] += bump
        rows["Meal Type"][start] = "Lunch"
        rows["Calories"][start] = 4 * macros[0] + 4 * macros[1] + 9 * macros[2]
        for k, v in zip(["Carbs", "Protein", "Fat", "Fiber"], macros):
            rows[k][start] = v
    d = pd.DataFrame({"Timestamp": t, "Libre GL": g, "Dexcom GL": g, "subject": subject,
                      "Calories (Activity)": 1.5, **rows})
    return d


def test_iauc_of_a_triangle_and_clean_flag():
    d = trace(1, [(60, 40, (50, 20, 10, 5)), (400, 40, (50, 20, 10, 5)), (450, 20, (10, 5, 5, 1))])
    m = cga.meals(d).sort_values("time").reset_index(drop=True)
    # triangle of height 40 over 120 min = 2400 mg/dL*min (the baseline is flat at 100)
    assert m.loc[0, "iauc"] == pytest.approx(2400, rel=0.02)
    assert m.loc[0, "peak_rise"] == pytest.approx(40, abs=1)
    # a second meal 50 minutes later makes the first window unclean
    assert bool(m.loc[0, "clean"]) and not bool(m.loc[1, "clean"]) and bool(m.loc[2, "clean"])
    assert m.loc[1, "hours_since_prev"] == pytest.approx(340 / 60)


def test_zero_kcal_entries_are_not_meals():
    d = trace(1, [(60, 40, (50, 20, 10, 5)), (100, 0, (0, 0, 0, 0))])
    m = cga.meals(d)
    assert len(m) == 1 and bool(m["clean"].iloc[0])


def test_meal_distance_is_zero_for_identical_meals_and_bounded():
    m = pd.DataFrame({"carbs": [50, 50, 10, 80], "protein": [20, 20, 40, 5],
                      "fat": [10, 10, 30, 2], "fiber": [5, 5, 1, 0]})
    D = cga.meal_distance(m)
    for v in ["composition", "fraction", "fused"]:
        assert D[v][0, 1] == pytest.approx(0)
        assert D[v].max() == pytest.approx(1)
        assert np.allclose(D[v], D[v].T)


def test_energy_requirement_formulas():
    bio = pd.DataFrame({"subject": [1], "Age": [30], "Gender": ["M"], "BMI": [24.0],
                        "Body weight": [70 / 0.45359237], "Height": [175 / 2.54],
                        "A1c PDL (Lab)": [5.2]})
    cgm = pd.DataFrame({"subject": [1], "Timestamp": pd.to_datetime(["2024-01-01"]),
                        "Calories (Activity)": [1.0]})
    p = cga.persons(bio, cgm).iloc[0]
    assert p.bmr_hb == pytest.approx(88.362 + 13.397 * 70 + 4.799 * 175 - 5.677 * 30)
    assert p.bmr_msj == pytest.approx(10 * 70 + 6.25 * 175 - 5 * 30 + 5)
    assert p.group == "healthy"
