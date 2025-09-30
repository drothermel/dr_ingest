from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

DELIMITERS = ("-", "_", "/", "(", ")", "%", "+", ",")
SPACE_NORM = re.compile(r"\s+")
SUFFIX_MULTIPLIERS = {
    "m": 1e6,
    "g": 1e9,
    "b": 1e9,
    "t": 1e12,
}
COMPUTE_UNITS = "e15"
TOKENS_UNITS = "million"
DS_MIX_TO_DS_NAME = {
    "falcon_and_cc": "falcon_cc",
    "falcon_and_cc_eli5_oh_top10p": "falcon_cc_qc_10",
    "falcon_and_cc_eli5_oh_top20p": "falcon_cc_qc_20",
    "falcon_and_cc_og_eli5_oh_top10p": "falcon_cc_qc_orig_10",
    "falcon_and_cc_tulu_qc_top10": "falcon_cc_qc_tulu_10",
    "dolma_v1_6_and_sources_baseline": "dolma16",
    "no_code": "dolma17_no_code",
    "no_reddit": "dolma17_no_reddit",
    "no_math_no_code": "dolma17_no_math_code",
    "no_flan": "dolma17_no_flan",
    "dolma17_50p_dclm_baseline_50p": "dclm_baseline_50_dolma_50",
    "dolma17_25p_dclm_baseline_75p": "dclm_baseline_75_dolma_25",
    "dolma17_75p_dclm_baseline_25p": "dclm_baseline_25_dolma_75",
    "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p": "dclm_baseline_qc_10",
    "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p": "dclm_baseline_qc_20",
    "dclm_ft7percentile_fw3": "dclm_baseline_qc_7_fw3",
    "dclm_ft7percentile_fw2": "dclm_baseline_qc_7_fw2",
    "prox_fineweb_pro": "fineweb_pro",
    "fineweb_edu_dedup": "fineweb_edu",
    "dclm_fw_top10": "dclm_baseline_qc_fw_10",
    "dclm_fw_top3": "dclm_baseline_qc_fw_3",
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


def convert_number(
    value: float | int | str | None, unit: str
) -> tuple[float | None, str]:
    value = normalize_numeric(value)
    if value is not None:
        if unit == "million":
            return to_millions(value)
        elif unit == "billion":
            return to_billions(value)
        elif unit == "trillion":
            return to_trillions(value)
        elif unit == "e15":
            return to_e15(value)
    return None, unit


def to_millions(value: float | int) -> tuple[float, str]:
    return value / 1e6, "million"


def to_billions(value: float | int) -> tuple[float, str]:
    return value / 1e9, "billion"


def to_trillions(value: float | int) -> tuple[float, str]:
    return value / 1e12, "trillion"


def to_e15(value: float | int) -> tuple[float, str]:
    return value / 1e15, "e15"


def normalize_ds_str(value: Any) -> str | None:
    norm_str = normalize_str(value, final_delim="_")
    if norm_str is None:
        return None
    norm_str = norm_str.replace(".", "")
    norm_str = DS_MIX_TO_DS_NAME.get(norm_str, norm_str)
    return norm_str


def normalize_tokens(value: int | float | str | None) -> float | None:
    number = normalize_numeric(value)
    return convert_number(number, TOKENS_UNITS)[0]


def normalize_compute(value: int | float | str | None) -> float | None:
    number = normalize_numeric(value)
    return convert_number(number, COMPUTE_UNITS)[0]


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
