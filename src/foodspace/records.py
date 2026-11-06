"""Keys for food-record tables in the FAO/WHO GIFT layout.

A recorded day is one SURVEY_DAY of one subject in one ROUND (repeat recalls or record
periods may restart day numbering, so ROUND is part of the key when the table has it).
An eating occasion is one MEAL_NAME on such a day.
"""

from __future__ import annotations

import pandas as pd


def day_keys(records: pd.DataFrame) -> list[str]:
    """Columns that identify one recorded day of one person."""
    return ["SUBJECT"] + (["ROUND"] if "ROUND" in records.columns else []) + ["SURVEY_DAY"]


def meal_keys(records: pd.DataFrame) -> list[str]:
    """Columns that identify one eating occasion."""
    return day_keys(records) + ["MEAL_NAME"]


def days_per_subject(records: pd.DataFrame) -> pd.Series:
    """Number of recorded days per subject."""
    return records[day_keys(records)].drop_duplicates().groupby("SUBJECT").size()
