from __future__ import annotations

import pandas as pd

from dr_ingest.utils.pandas import apply_if_column, ensure_column
from dr_ingest.wandb.metrics import canonicalize_metric_label


def normalize_matched_run_type(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    default_label = canonicalize_metric_label(None)
    result = ensure_column(result, "comparison_metric", default_label, inplace=True)
    result = apply_if_column(
        result,
        "comparison_metric",
        lambda series: series.apply(canonicalize_metric_label),
        inplace=True,
    )
    return result


__all__ = ["normalize_matched_run_type"]
