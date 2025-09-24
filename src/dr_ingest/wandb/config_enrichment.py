"""Helpers for merging WandB config and summary fields into run data."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable

import pandas as pd

from ..df_ops import ensure_column, maybe_update_cell


def extract_config_fields(
    runs_df: pd.DataFrame, run_ids: Iterable[str], field_mapping: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    """Pull selected config/summary fields for the provided run IDs."""

    config_data: Dict[str, Dict[str, Any]] = {}
    for run_id in run_ids:
        run_row = runs_df[runs_df["run_id"] == run_id]
        if run_row.empty:
            continue

        config = _safe_load_json(run_row.iloc[0].get("config")) or {}
        for target_field, config_field in field_mapping.items():
            value = config.get(config_field)
            if value is not None:
                config_data.setdefault(run_id, {})[target_field] = value

        summary = _safe_load_json(run_row.iloc[0].get("summary"))
        if summary and summary.get("total_tokens") is not None:
            config_data.setdefault(run_id, {})["num_finetuned_tokens_real"] = summary[
                "total_tokens"
            ]

    return config_data


def merge_config_fields(
    frame: pd.DataFrame,
    config_data: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Merge config-derived fields into the processed DataFrame."""

    result = frame.copy()
    if "run_id" not in result.columns:
        return result

    for run_id, fields in config_data.items():
        run_idx = result.index[result["run_id"] == run_id]
        if run_idx.empty:
            continue
        row_idx = run_idx[0]
        for field, value in fields.items():
            if field == "num_finetuned_tokens_real":
                result = ensure_column(result, field, None, inplace=True)
                result.loc[row_idx, field] = value
            elif field in result.columns:
                result = maybe_update_cell(
                    result,
                    row_idx,
                    field,
                    str(value),
                    inplace=True,
                )
    return result


def _safe_load_json(payload: Any) -> Dict[str, Any] | None:
    if not payload or (isinstance(payload, float) and pd.isna(payload)):
        return None
    try:
        return json.loads(payload) if isinstance(payload, str) else dict(payload)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


__all__ = ["extract_config_fields", "merge_config_fields"]
