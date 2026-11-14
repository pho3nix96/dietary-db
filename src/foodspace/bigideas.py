"""Read the BIG IDEAs Lab glycemic variability dataset (PhysioNet, ODC-By 1.0).

Cho, P., Kim, J., Bent, B., & Dunn, J. (2023). BIG IDEAs Lab Glycemic Variability and
Wearable Device Data (version 1.1.2). PhysioNet. https://doi.org/10.13026/zthx-s306

The raw files stay outside the repository. Expected layout (as distributed):

    <root>/001/Food_Log_001.csv, <root>/001/Dexcom_001.csv, ...   or all files flat in <root>/

The 16 food logs are not all written the same way, and `read_food_logs` evens them out:
  * 007 and 013 name the columns Food_Item, Amount_Consumed, Units, Food_Item_Description;
  * 015 and 016 have no header row;
  * 003 has no header and only 11 columns: no time_end, dietary_fiber or total_fat
    (checked on its rows: 393 kcal chicken nuggets = 19 g carb, 0.1 g sugar, 20 g protein).
Nutrients are per logged portion (US label values: total_carb includes fibre).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["date", "time", "time_begin", "time_end", "logged_food", "amount", "unit", "searched_food",
           "calorie", "total_carb", "dietary_fiber", "sugar", "protein", "total_fat"]
RENAME = {"Food_Item": "logged_food", "Amount_Consumed": "amount", "Units": "unit",
          "Food_Item_Description": "searched_food"}
SHORT_003 = ["date", "time", "time_begin", "logged_food", "amount", "unit", "searched_food",
             "calorie", "total_carb", "sugar", "protein"]
NUTRIENTS = ["calorie", "total_carb", "dietary_fiber", "sugar", "protein", "total_fat"]


def _find(root: Path, pattern: str) -> list[Path]:
    files = sorted(root.glob(pattern)) or sorted(root.glob(f"*/{pattern}"))
    if not files:
        raise FileNotFoundError(f"No {pattern} under {root}")
    return files


def read_food_log(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    first = path.open(encoding="utf-8-sig").readline()
    if first.startswith("date,"):
        df = pd.read_csv(path, encoding="utf-8-sig").rename(columns=RENAME)
    else:
        n = len(pd.read_csv(path, header=None, nrows=5).columns)
        names = SHORT_003 if n == len(SHORT_003) else COLUMNS
        df = pd.read_csv(path, header=None, names=names, encoding="utf-8-sig")
    for c in COLUMNS:
        if c not in df:
            df[c] = np.nan
    df = df[COLUMNS].copy()
    for c in NUTRIENTS + ["amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["logged_food", "searched_food", "unit"]:
        df[c] = df[c].where(df[c].isna(), df[c].astype(str).str.strip()).replace("", np.nan)
    df["time_begin"] = pd.to_datetime(df["time_begin"], errors="coerce")
    df["participant"] = path.stem.split("_")[-1]
    return df


def read_food_logs(root: str | Path) -> pd.DataFrame:
    """All 16 food logs as one table, one row per logged item, with a `line_id`."""
    df = pd.concat([read_food_log(p) for p in _find(Path(root), "Food_Log_*.csv")], ignore_index=True)
    df = df[df["logged_food"].notna() | df["searched_food"].notna() | df["calorie"].notna()].reset_index(drop=True)
    df.insert(0, "line_id", np.arange(len(df)))
    return df


def read_dexcom(root: str | Path) -> pd.DataFrame:
    """Glucose readings (EGV rows only): participant, time, glucose (mg/dL)."""
    out = []
    for p in _find(Path(root), "Dexcom_*.csv"):
        d = pd.read_csv(p, low_memory=False)
        d = d[d["Event Type"] == "EGV"]
        g = pd.to_numeric(d["Glucose Value (mg/dL)"], errors="coerce")
        out.append(pd.DataFrame({"participant": p.stem.split("_")[-1],
                                 "time": pd.to_datetime(d["Timestamp (YYYY-MM-DDThh:mm:ss)"], errors="coerce"),
                                 "glucose": g}).dropna())
    return pd.concat(out, ignore_index=True).sort_values(["participant", "time"]).reset_index(drop=True)
