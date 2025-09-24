from __future__ import annotations

from typing import Callable

import pandas as pd

from dr_ingest.df_ops import apply_if_column, ensure_column
from dr_ingest.wandb.config_registry import wandb_hooks


def _normalise_metric_label(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "pile-valppl"
    value_str = str(value)
    if value_str.endswith("-valppl") or value_str.endswith("_en-valppl"):
        return value_str
    if value_str.lower() == "c4":
        return "c4_en-valppl"
    return f"{value_str}-valppl"


def normalize_matched_run_type(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure matched runs have default comparison metrics and suffix normalisation."""

    result = df.copy()
    result = ensure_column(result, "comparison_metric", "pile", inplace=True)
    result = apply_if_column(
        result,
        "comparison_metric",
        lambda series: series.apply(_normalise_metric_label),
        inplace=True,
    )
    return result


@wandb_hooks.register("normalize_matched_run_type")
def _make_normalize_matched_run_type() -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Registry factory returning the normalisation hook."""

    return normalize_matched_run_type


__all__ = ["normalize_matched_run_type"]
