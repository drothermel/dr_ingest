from __future__ import annotations

import pandas as pd

from dr_ingest.wandb.postprocess import apply_processing, extract_config_fields


def test_apply_processing_value_converters() -> None:
    df = pd.DataFrame(
        {
            "run_id": ["2025_08_30-16_54_48_test_run"],
            "timestamp": ["2025_08_30-16_54_48"],
            "num_finetune_tokens": ["10M"],
            "num_finetune_tokens_per_epoch": ["5M"],
            "num_finetuned_tokens_real": ["8M"],
        }
    )

    processed = apply_processing({"simple_ft": df})["simple_ft"]

    assert processed.loc[0, "timestamp"] == pd.Timestamp("2025-08-30 16:54:48")
    assert processed.loc[0, "num_finetune_tokens"] == 10_000_000
    assert processed.loc[0, "num_finetune_tokens_per_epoch"] == 5_000_000
    assert processed.loc[0, "num_finetuned_tokens_real"] == 8_000_000


def test_extract_config_fields_reads_summary_payload() -> None:
    runs_df = pd.DataFrame(
        {
            "run_id": ["run-1", "run-2"],
            "config": ["{}", "{}"],
            "summary": ['{"total_tokens": 123}', None],
        }
    )

    mapping = {"num_finetuned_tokens_real": "total_tokens"}
    updates = extract_config_fields(
        runs_df,
        ["run-1", "run-2"],
        mapping,
        source_column="summary",
    )

    assert updates == {"run-1": {"num_finetuned_tokens_real": 123}}
