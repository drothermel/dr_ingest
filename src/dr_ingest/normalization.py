from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

DELIMITERS = ("-", "_", "/")
SPACE_NORM = re.compile(r"\s+")
SUFFIX_MULTIPLIERS = {
    "m": 1e6,
    "g": 1e9,
    "b": 1e9,
    "t": 1e12,
}

_NUMBER_SUFFIX_PATTERN = re.compile(
    r"^\s*(?P<num>[+-]?\d+(?:\.\d+)?)\s*(?P<suffix>[a-z]*)\s*$"
)


def is_nully(value: Any) -> bool:
    if value is None or (isinstance(value, str) and not value.strip()):
        return True
    try:
        return pd.isna(value)
    except Exception:  # noqa: S110
        pass
    return isinstance(value, float) and math.isnan(value)


def normalize_str(value: Any) -> str | None:
    if is_nully(value):
        return None
    text = str(value).strip().lower()
    for delimiter in DELIMITERS:
        text = text.replace(delimiter, " ")
    text = SPACE_NORM.sub(" ", text).strip()
    return text or None


def normalize_numeric(value: Any) -> float | None:
    if is_nully(value):
        return None
    value = str(value).strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def convert_string_to_number(value_str: Any) -> float | None:
    if is_nully(value_str):
        return None

    text = normalize_str(value_str)
    if text is None:
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
    "tokens_to_number.v1": convert_string_to_number,
    "timestamp.v1": convert_timestamp,
}


__all__ = [
    "CONVERSION_MAP",
    "convert_string_to_number",
    "convert_timestamp",
    "df_coerce_to_numeric",
    "is_nully",
    "normalize_numeric",
    "normalize_str",
]
