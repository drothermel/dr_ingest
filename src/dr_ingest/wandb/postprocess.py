"""Post-processing pipeline for classified WandB runs."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .processing_context import ProcessingContext
from .hydration import HydrationExecutor
from .normalization_pipeline import RunNormalizationExecutor


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
    normalizer = RunNormalizationExecutor.from_context(context)
    processed: Dict[str, pd.DataFrame] = {}
    for run_type, df in dataframes.items():
        frame = df.copy()
        frame = hydrator.apply(frame, ground_truth_source=runs_df)
        frame = normalizer.normalize(frame, run_type=run_type)
        processed[run_type] = frame
    return processed


__all__ = ["apply_processing"]
