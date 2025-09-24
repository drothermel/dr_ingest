"""Post-processing pipeline for classified WandB runs."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .processing_context import ProcessingContext
from .tokens import (
    ensure_full_finetune_defaults,
    fill_missing_token_totals,
)
from .hydration import HydrationExecutor


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
    hydrator = HydrationExecutor.from_context(context)
    processed: Dict[str, pd.DataFrame] = {}
    recipe_columns = ["comparison_model_recipe", "initial_checkpoint_recipe", "ckpt_data"]

    for run_type, df in dataframes.items():
        frame = df.copy()
        frame = hydrator.apply(frame, ground_truth_source=runs_df)
        frame = (
            frame.pipe(context.apply_defaults)
            .pipe(context.map_recipes, recipe_columns)
            .pipe(context.apply_converters)
            .pipe(context.rename_columns)
            .pipe(ensure_full_finetune_defaults)
            .pipe(fill_missing_token_totals)
        )
        frame = context.apply_hook(run_type, frame)
        processed[run_type] = frame
    return processed
__all__ = ["apply_processing"]
