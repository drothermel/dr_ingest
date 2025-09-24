"""Utilities for normalising fine-tuning token values."""

from __future__ import annotations

import pandas as pd

from .constants import ALL_FT_TOKENS, DEFAULT_FULL_FT_EPOCHS


def ensure_full_finetune_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Fill token columns for runs that contain the ``_Ft_`` marker."""

    if "run_id" not in df.columns:
        return df.copy()

    result = df.copy()
    mask = result["run_id"].str.contains("_Ft_")
    if isinstance(mask, pd.Series) and mask.any():
        result.loc[mask, "num_finetune_tokens_per_epoch"] = ALL_FT_TOKENS
        result.loc[mask, "num_finetune_epochs"] = DEFAULT_FULL_FT_EPOCHS
        full_total = DEFAULT_FULL_FT_EPOCHS * ALL_FT_TOKENS
        result.loc[mask, "num_finetune_tokens"] = full_total
        result.loc[mask, "num_finetuned_tokens_real"] = full_total
    return result


def fill_missing_token_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute total fine-tune tokens when epoch and per-epoch values exist."""

    required_cols = {"num_finetune_tokens_per_epoch", "num_finetune_epochs"}
    if not required_cols.issubset(df.columns):
        return df.copy()

    result = df.copy()
    if "num_finetune_tokens" not in result.columns:
        result["num_finetune_tokens"] = None

    result["num_finetune_epochs"] = pd.to_numeric(
        result["num_finetune_epochs"], errors="coerce"
    )

    fill_mask = (
        result["num_finetune_tokens_per_epoch"].notna()
        & result["num_finetune_epochs"].notna()
        & result["num_finetune_tokens"].isna()
    )
    result.loc[fill_mask, "num_finetune_tokens"] = (
        result.loc[fill_mask, "num_finetune_tokens_per_epoch"]
        * result.loc[fill_mask, "num_finetune_epochs"]
    )
    return result


def compute_token_delta_percent(df: pd.DataFrame) -> pd.DataFrame:
    """Compute percentage difference between planned and actual token usage."""

    required_cols = {"num_finetune_tokens", "num_finetuned_tokens_real"}
    if not required_cols.issubset(df.columns):
        return df.copy()

    result = df.copy()
    mask = (
        result["num_finetune_tokens"].notna()
        & result["num_finetuned_tokens_real"].notna()
        & (result["num_finetune_tokens"] != 0)
    )
    result["abs_difference_ft_tokens_pct"] = None
    result.loc[mask, "abs_difference_ft_tokens_pct"] = (
        (
            result.loc[mask, "num_finetune_tokens"]
            - result.loc[mask, "num_finetuned_tokens_real"]
        ).abs()
        / result.loc[mask, "num_finetune_tokens"]
        * 100
    )
    return result


__all__ = [
    "ensure_full_finetune_defaults",
    "fill_missing_token_totals",
    "compute_token_delta_percent",
]
