"""Post-processing pipeline for classified WandB runs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import pandas as pd

from ..df_ops import (
    apply_row_updates,
    force_setter,
    maybe_update_setter,
    require_row_index,
)
from ..json_utils import safe_load_json
from .processing_context import ProcessingContext
from .tokens import (
    ensure_full_finetune_defaults,
    fill_missing_token_totals,
)


def apply_processing(
    dataframes: Dict[str, pd.DataFrame],
    defaults: Optional[Dict[str, Any]] = None,
    column_map: Optional[Dict[str, str]] = None,
    runs_df: Optional[pd.DataFrame] = None,
    history_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Normalise extracted run data across run types."""

    context = ProcessingContext.from_config(
        overrides=defaults or {}, column_renames_override=column_map or {}
    )
    processed: Dict[str, pd.DataFrame] = {}
    recipe_columns = ["comparison_model_recipe", "initial_checkpoint_recipe"]

    for run_type, df in dataframes.items():
        frame = (
            df.copy()
            .pipe(context.apply_defaults)
            .pipe(context.map_recipes, recipe_columns)
            .pipe(merge_config_fields_from_runs, runs_df=runs_df, context=context)
            .pipe(context.apply_converters)
            .pipe(context.rename_columns)
            .pipe(ensure_full_finetune_defaults)
            .pipe(fill_missing_token_totals)
        )
        frame = context.apply_hook(run_type, frame)
        processed[run_type] = frame
    return processed


def extract_config_fields(
    runs_df: pd.DataFrame,
    run_ids: Iterable[str],
    field_mapping: dict[str, str],
    *,
    source_column: str = "config",
) -> dict[str, dict[str, Any]]:
    """Return updates for the provided run IDs from the requested payload."""

    if not field_mapping:
        return {}

    updates: dict[str, dict[str, Any]] = {}
    for run_id in run_ids:
        row_idx = require_row_index(runs_df, "run_id", run_id)
        run_row = runs_df.iloc[row_idx]
        payload = safe_load_json(run_row.get(source_column)) or {}
        if not isinstance(payload, dict):
            continue
        for target_field, source_field in field_mapping.items():
            if source_field in payload and payload[source_field] is not None:
                updates.setdefault(run_id, {})[target_field] = payload[source_field]
    return updates


def merge_config_fields_from_runs(
    frame: pd.DataFrame,
    runs_df: Optional[pd.DataFrame],
    context: ProcessingContext,
) -> pd.DataFrame:
    if runs_df is None or (
        not context.config_field_mapping and not context.summary_field_mapping
    ):
        return frame
    assert all("run_id" in d.columns for d in [frame, runs_df]), "run_id required"

    run_ids = frame["run_id"].tolist()
    summary_updates = extract_config_fields(
        runs_df,
        run_ids,
        context.summary_field_mapping,
        source_column="summary",
    )
    frame = apply_row_updates(frame, summary_updates, force_setter)
    optional_updates = extract_config_fields(
        runs_df,
        run_ids,
        context.config_field_mapping,
    )
    frame = apply_row_updates(frame, optional_updates, maybe_update_setter)
    return frame


__all__ = ["apply_processing", "extract_config_fields", "merge_config_fields_from_runs"]
