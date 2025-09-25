from __future__ import annotations

from typing import Any

import pandas as pd

from .hydration import HydrationExecutor
from .normalization_pipeline import RunNormalizationExecutor
from .processing_context import ProcessingContext


def apply_processing(
    dataframes: dict[str, pd.DataFrame],
    defaults: dict[str, Any] | None = None,
    column_map: dict[str, str] | None = None,
    runs_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    context = ProcessingContext.from_config(
        overrides=defaults or {}, column_renames_override=column_map or {}
    )
    hydrator = HydrationExecutor.from_context(context)
    normalizer = RunNormalizationExecutor.from_context(context)
    processed: dict[str, pd.DataFrame] = {}
    for run_type, df in dataframes.items():
        frame = df.copy()
        frame = hydrator.apply(frame, ground_truth_source=runs_df)
        frame = normalizer.normalize(frame, run_type=run_type)
        processed[run_type] = frame
    return processed


__all__ = ["apply_processing"]
