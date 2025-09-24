from __future__ import annotations

import pandas as pd

from dr_ingest.wandb.tokens import fill_missing_token_totals


def test_fill_missing_token_totals_handles_empty_mask() -> None:
    df = pd.DataFrame(
        {
            "run_id": ["ft-a", "ft-b"],
            "num_finetune_tokens_per_epoch": [pd.NA, pd.NA],
            "num_finetune_epochs": [pd.NA, pd.NA],
        }
    )

    result = fill_missing_token_totals(df)
    assert result["num_finetune_tokens"].isna().all()


def test_fill_missing_token_totals_computes_per_row() -> None:
    df = pd.DataFrame(
        {
            "run_id": ["ft-a", "ft-b"],
            "num_finetune_tokens_per_epoch": [10_000_000, 12_000_000],
            "num_finetune_epochs": [2, 3],
        }
    )

    result = fill_missing_token_totals(df)

    assert (
        result.loc[result["run_id"] == "ft-a", "num_finetune_tokens"].item()
        == 20_000_000
    )
    assert (
        result.loc[result["run_id"] == "ft-b", "num_finetune_tokens"].item()
        == 36_000_000
    )
