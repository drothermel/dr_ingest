from __future__ import annotations

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


def convert_string_to_number(value_str: Any) -> Optional[float]:
    """Convert token strings with suffixes to numeric counts."""
    if pd.isna(value_str):
        return None
    value_str = str(value_str).strip().upper()
    if value_str in {"N/A", ""}:
        return None
    try:
        if value_str.endswith("M"):
            return float(value_str[:-1]) * 1e6
        if value_str.endswith(("G", "B")):
            return float(value_str[:-1]) * 1e9
        if value_str.endswith("T"):
            return float(value_str[:-2]) * 1e12
        return float(value_str)
    except (ValueError, TypeError):
        return None


wandb_value_converters.register("timestamp.v1")(convert_timestamp)
wandb_value_converters.register("tokens_to_number.v1")(convert_string_to_number)


__all__ = ["convert_timestamp", "convert_string_to_number"]
