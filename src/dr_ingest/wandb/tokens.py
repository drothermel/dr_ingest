"""Utilities for normalising fine-tuning token values."""

from __future__ import annotations

import pandas as pd

from ..df_ops import ensure_column, masked_setter
from .constants import ALL_FT_TOKENS, DEFAULT_FULL_FT_EPOCHS
from .utils import coerce_to_numeric

FULL_TOTAL_TOKENS = DEFAULT_FULL_FT_EPOCHS * ALL_FT_TOKENS
REQUIRED_TOKEN_COLS = {"num_finetune_tokens_per_epoch", "num_finetune_epochs"}
TOK_DEFAULT_VALS = {
    "num_finetune_tokens_per_epoch": ALL_FT_TOKENS,
    "num_finetune_epochs": DEFAULT_FULL_FT_EPOCHS,
    "num_finetune_tokens": FULL_TOTAL_TOKENS,
    "num_finetuned_tokens_real": FULL_TOTAL_TOKENS,
}


def ensure_full_finetune_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Fill token columns for runs that contain the ``_Ft_`` marker."""

    if "run_id" not in df.columns:
        return df

    result = df.copy()
    mask = result["run_id"].str.contains("_Ft_")
    if isinstance(mask, pd.Series) and mask.any():
        for col, val in TOK_DEFAULT_VALS.items():
            result = masked_setter(result, mask, col, val)
    return result


def fill_missing_token_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute total fine-tune tokens when epoch and per-epoch values exist."""

    if not REQUIRED_TOKEN_COLS.issubset(df.columns):
        return df

    result = df.copy()
    result = ensure_column(result, "num_finetune_tokens", None)
    result = coerce_to_numeric(result, "num_finetune_tokens")
    result = coerce_to_numeric(result, "num_finetune_tokens_per_epoch")
    result = coerce_to_numeric(result, "num_finetune_epochs")

    calc_ft_toks_mask = (
        result["num_finetune_tokens_per_epoch"].notna()
        & result["num_finetune_epochs"].notna()
        & result["num_finetune_tokens"].isna()
    )
    if calc_ft_toks_mask.any():
        per_epoch = result.loc[calc_ft_toks_mask, "num_finetune_tokens_per_epoch"]
        epochs = result.loc[calc_ft_toks_mask, "num_finetune_epochs"]
        result.loc[calc_ft_toks_mask, "num_finetune_tokens"] = per_epoch * epochs
    return result


__all__ = [
    "ensure_full_finetune_defaults",
    "fill_missing_token_totals",
]
