from __future__ import annotations

import pandas as pd

from dr_ingest.normalization import df_coerce_to_numeric
from dr_ingest.utils.pandas import ensure_column, masked_setter
from dr_ingest.wandb.constants import ALL_FT_TOKENS, DEFAULT_FULL_FT_EPOCHS

FULL_TOTAL_TOKENS = DEFAULT_FULL_FT_EPOCHS * ALL_FT_TOKENS
REQUIRED_TOKEN_COLS = {"num_finetune_tokens_per_epoch", "num_finetune_epochs"}
TOK_DEFAULT_VALS = {
    "num_finetune_tokens_per_epoch": ALL_FT_TOKENS,
    "num_finetune_epochs": DEFAULT_FULL_FT_EPOCHS,
    "num_finetune_tokens": FULL_TOTAL_TOKENS,
    "num_finetuned_tokens_real": FULL_TOTAL_TOKENS,
}


def ensure_full_finetune_defaults(df: pd.DataFrame) -> pd.DataFrame:
    if "run_id" not in df.columns:
        return df

    result = df.copy()
    mask = result["run_id"].str.contains("_Ft_")
    if isinstance(mask, pd.Series) and mask.any():
        for col, val in TOK_DEFAULT_VALS.items():
            result = masked_setter(result, mask, col, val)
    return result


def fill_missing_token_totals(df: pd.DataFrame) -> pd.DataFrame:
    if not REQUIRED_TOKEN_COLS.issubset(df.columns):
        return df

    result = df.copy()
    result = ensure_column(result, "num_finetune_tokens", None)
    result = df_coerce_to_numeric(result, "num_finetune_tokens")
    result = df_coerce_to_numeric(result, "num_finetune_tokens_per_epoch")
    result = df_coerce_to_numeric(result, "num_finetune_epochs")

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
