from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

DELIMITERS = ("-", "_", "/")
SPACE_NORM = re.compile(r"\s+")


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


__all__ = ["is_nully", "normalize_numeric", "normalize_str"]
