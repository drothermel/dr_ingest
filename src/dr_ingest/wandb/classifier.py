"""Run-ID classification utilities built on the pattern registry."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
from memo import memlist

from .patterns import PATTERN_SPECS

CLASSIFICATION_LOG: List[Dict[str, Any]] = []
_record_classification = memlist(data=CLASSIFICATION_LOG)


@_record_classification
def _log_event(**kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - memo hook
    return kwargs


def classify_run_id(run_id: str) -> Tuple[str, Dict[str, str | None]]:
    """Return the run type and extracted metadata for a run identifier."""
    run_type, extracted = classify_run_id_type_and_extract(run_id)
    _log_event(run_id=run_id, run_type=run_type, pattern=extracted.get("pattern_name"))
    return run_type, extracted


def classify_run_id_type_and_extract(run_id: str) -> Tuple[str, Dict[str, str | None]]:
    """Match a run ID against the registered patterns."""
    for spec in PATTERN_SPECS:
        match = spec.regex.match(run_id)
        if match:
            extracted = match.groupdict()
            extracted["pattern_name"] = spec.name
            return spec.run_type, extracted

    if (
        "main_default" in run_id
        and "dpo" not in run_id
        and "--reduce_loss=" not in run_id
    ):
        return "old", {}

    return "other", {}


def parse_and_group_run_ids(
    df: pd.DataFrame, run_id_col: str = "run_id"
) -> Dict[str, List[Dict[str, str]]]:
    """Group run IDs by run type and attach extracted fields."""
    if run_id_col not in df.columns:
        raise KeyError(f"Column '{run_id_col}' not found in DataFrame")

    type_data: Dict[str, List[Dict[str, str]]] = {}
    for run_id in df[run_id_col].astype(str):
        run_type, extracted_data = classify_run_id(run_id)
        type_data.setdefault(run_type, [])
        if run_type != "old":
            extracted_data["run_id"] = run_id
            type_data[run_type].append(extracted_data)
    for run_type, records in type_data.items():
        records.sort(key=lambda x: x.get("run_id", ""))
    return type_data


def convert_groups_to_dataframes(
    grouped_data: Dict[str, List[Dict[str, str]]],
) -> Dict[str, pd.DataFrame]:
    """Convert grouped run metadata to DataFrames keyed by run type."""
    dataframes: Dict[str, pd.DataFrame] = {}
    for run_type, records in grouped_data.items():
        if records:
            df = pd.DataFrame(records)
            if "pattern_name" in df.columns:
                df = df.sort_values("pattern_name")
            columns = ["run_id"] + [col for col in df.columns if col != "run_id"]
            dataframes[run_type] = df[columns]
    return dataframes


__all__ = [
    "CLASSIFICATION_LOG",
    "classify_run_id",
    "classify_run_id_type_and_extract",
    "parse_and_group_run_ids",
    "convert_groups_to_dataframes",
]
