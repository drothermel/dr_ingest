"""Post-processing pipeline for classified WandB runs."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .config_enrichment import extract_config_fields, merge_config_fields
from .processing_context import ProcessingContext
from .tokens import (
    compute_token_delta_percent,
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
        frame = df.copy()
        frame = context.rename_columns(frame)
        frame = context.apply_defaults(frame)
        frame = context.map_recipes(frame, recipe_columns)

        if runs_df is not None and "run_id" in frame.columns:
            run_ids = frame["run_id"].tolist()
            if context.config_field_mapping:
                config_data = extract_config_fields(
                    runs_df, run_ids, context.config_field_mapping
                )
                frame = merge_config_fields(frame, config_data)

        frame = context.apply_converters(frame)
        frame = ensure_comparison_recipe_default(frame)
        frame = ensure_full_finetune_defaults(frame)
        frame = fill_missing_token_totals(frame)
        frame = compute_token_delta_percent(frame)
        frame = context.apply_hook(run_type, frame)

        processed[run_type] = frame

    return processed


def ensure_comparison_recipe_default(frame: pd.DataFrame) -> pd.DataFrame:
    if (
        "comparison_model_size" in frame.columns
        and "comparison_model_recipe" in frame.columns
    ):
        result = frame.copy()
        result["comparison_model_recipe"] = result["comparison_model_recipe"].fillna(
            "Dolma1.7"
        )
        return result
    return frame


__all__ = ["apply_processing", "extract_config_fields", "merge_config_fields"]
