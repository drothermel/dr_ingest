"""Post-processing hooks for run-type specific tweaks."""

from __future__ import annotations

import pandas as pd


def normalize_matched(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure matched runs have normalised comparison metrics."""
    result = df.copy()
    if "comparison_metric" not in result.columns:
        result["comparison_metric"] = "pile"
    result["comparison_metric"] = result["comparison_metric"].fillna("pile")

    def _normalise_metric(value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "pile-valppl"
        value_str = str(value)
        if value_str.endswith("-valppl") or value_str.endswith("_en-valppl"):
            return value_str
        if value_str.lower() == "c4":
            return "c4_en-valppl"
        return f"{value_str}-valppl"

    result["comparison_metric"] = result["comparison_metric"].apply(_normalise_metric)
    return result


__all__ = ["normalize_matched"]
