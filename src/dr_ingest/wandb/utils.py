from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd

from .config_registry import wandb_value_converters


def convert_timestamp(ts_str: Any) -> Optional[pd.Timestamp]:
    """Convert a timestamp string (6- or 8-component) to a pandas Timestamp."""
    if pd.isna(ts_str):
        return None
    ts_str = str(ts_str)
    if "_" in ts_str:
        try:
            return pd.to_datetime(ts_str, format="%Y_%m_%d-%H_%M_%S")
        except (ValueError, TypeError):
            return None
    try:
        return pd.to_datetime(ts_str, format="%y%m%d-%H%M%S")
    except (ValueError, TypeError):
        return None


def coerce_to_numeric(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Coerce a column to numeric."""
    if column not in df.columns:
        return df
    df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


SUFFIX_MULTIPLIERS = {
    "M": 1e6,
    "G": 1e9,
    "B": 1e9,
    "T": 1e12,
}

_NUMBER_SUFFIX_PATTERN = re.compile(
    r"^\s*(?P<num>[+-]?\d+(?:\.\d+)?)\s*(?P<suffix>[A-Z]*)\s*$"
)


def convert_string_to_number(value_str: Any) -> Optional[float]:
    """Convert token strings with suffixes to numeric counts."""

    if pd.isna(value_str):  # handles pd.NA, NaN
        return None

    text = str(value_str).strip().upper()
    if text in {"", "N/A"}:
        return None

    match = _NUMBER_SUFFIX_PATTERN.match(text)
    if not match:
        return None

    try:
        base_value = float(match.group("num"))
    except (TypeError, ValueError):
        return None

    suffix = match.group("suffix") or ""
    if not suffix:
        return base_value

    multiplier = SUFFIX_MULTIPLIERS.get(suffix[0])
    if multiplier is None:
        return None

    return base_value * multiplier


wandb_value_converters.register("timestamp.v1")(convert_timestamp)
wandb_value_converters.register("tokens_to_number.v1")(convert_string_to_number)
wandb_value_converters.register("coerce_to_numeric.v1")(coerce_to_numeric)


__all__ = ["convert_timestamp", "convert_string_to_number"]
