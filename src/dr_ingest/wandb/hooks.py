from __future__ import annotations

from typing import Callable

import pandas as pd

from dr_ingest.df_ops import apply_if_column, ensure_column
from dr_ingest.wandb.config_registry import wandb_hooks
from dr_ingest.wandb.metrics import canonicalize_metric_label


def normalize_matched_run_type(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure matched runs have default comparison metrics and suffix normalisation."""

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


@wandb_hooks.register("normalize_matched_run_type")
def _make_normalize_matched_run_type() -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Registry factory returning the normalisation hook."""

    return normalize_matched_run_type


__all__ = ["normalize_matched_run_type"]
