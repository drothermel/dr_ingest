from __future__ import annotations

from pathlib import Path

import pandas as pd

from dr_ingest.wandb.classifier import (
    CLASSIFICATION_LOG,
    convert_groups_to_dataframes,
    parse_and_group_run_ids,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "data"
RUNS_FIXTURE = FIXTURES / "wandb_runs_sample.parquet"


def load_runs_dataframe() -> pd.DataFrame:
    return pd.read_parquet(RUNS_FIXTURE)


def test_classify_sample_runs() -> None:
    runs_df = load_runs_dataframe()
    grouped = parse_and_group_run_ids(runs_df)
    frames = convert_groups_to_dataframes(grouped)

    # We expect known run IDs to fall into the right run types
    ft_run = "2025_08_30-16_54_48_test_finetune_DD-dolma1_7-4M_Ft_--learning_rate=5e-05"
    dpo_run = "2025_08_27-07_52_26_test_dpo_tune_cache_dd__dolma1_7-4M__main__100Mt_lr=5e-06_default_--learning_rate=2e-07"  # noqa: E501
    matched_run = "250912-172812_match_150M_c4_finetune_100Mtx1_DD-d17-530M-6250-2"

    assert ft_run in frames["simple_ft"].run_id.to_numpy()
    assert dpo_run in frames["dpo"].run_id.to_numpy()
    assert matched_run in frames["matched"].run_id.to_numpy()


def test_classification_log_records_events() -> None:
    CLASSIFICATION_LOG.clear()
    runs_df = load_runs_dataframe()
    parse_and_group_run_ids(runs_df)
    assert CLASSIFICATION_LOG  # log should capture entries
