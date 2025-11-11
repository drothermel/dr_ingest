from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

DELIMITERS = ("-", "_", "/", "(", ")", "%", "+", ",")
SPACE_NORM = re.compile(r"\s+")


def is_nully(value: Any) -> bool:
    if value is None or (isinstance(value, str) and not value.strip()):
        return True
    try:
        return pd.isna(value)
    except Exception:  # noqa: S110
        pass
    return isinstance(value, float) and math.isnan(value)


def normalize_str(value: Any, final_delim: str = " ") -> str | None:
    if is_nully(value):
        return None
    text = str(value).strip().lower()
    for delimiter in DELIMITERS:
        text = text.replace(delimiter, " ")
    text = SPACE_NORM.sub(" ", text).strip()
    if final_delim != " ":
        text = text.replace(" ", final_delim)
    return text or None


def convert_timestamp(ts_str: Any) -> pd.Timestamp | None:
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


def df_coerce_to_numeric(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return df
    df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


CONVERSION_MAP = {
    "timestamp.v1": convert_timestamp,
}


__all__ = [
    "CONVERSION_MAP",
    "convert_timestamp",
    "df_coerce_to_numeric",
    "is_nully",
    "normalize_str",
]
