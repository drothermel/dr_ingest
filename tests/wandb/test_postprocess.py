from __future__ import annotations

import pandas as pd

from dr_ingest.wandb.hydration import HydrationExecutor
from dr_ingest.wandb.normalization_pipeline import RunNormalizationExecutor
from dr_ingest.wandb.postprocess import apply_processing
from dr_ingest.wandb.processing_context import ProcessingContext


def test_apply_processing_value_converters() -> None:
    df = pd.DataFrame(
        {
            "run_id": ["2025_08_30-16_54_48_test_run"],
            "timestamp": ["2025_08_30-16_54_48"],
            "num_finetune_tokens": ["10M"],
            "num_finetune_tokens_per_epoch": ["5M"],
            "num_finetuned_tokens_real": ["8M"],
            "initial_checkpoint_recipe": ["d17"],
            "initial_checkpoint_steps": [None],
        }
    )

    runs_df = pd.DataFrame(
        {
            "run_id": ["2025_08_30-16_54_48_test_run"],
            "config": ["{}"],
            "summary": ['{"total_tokens": 123}'],
        }
    )

    processed = apply_processing({"simple_ft": df}, runs_df=runs_df)["simple_ft"]

    assert processed.loc[0, "timestamp"] == pd.Timestamp("2025-08-30 16:54:48")
    assert processed.loc[0, "num_finetune_tokens"] == 10_000_000
    assert processed.loc[0, "num_finetune_tokens_per_epoch"] == 5_000_000
    assert processed.loc[0, "num_finetuned_tokens_real"] == 123
    assert processed.loc[0, "ckpt_steps"] == "main"
    assert processed.loc[0, "ckpt_data"] == "Dolma1.7"


def test_hydration_executor_applies_summary_and_config_targets() -> None:
    df = pd.DataFrame({"run_id": ["run-1"]})
    runs_df = pd.DataFrame(
        {
            "run_id": ["run-1"],
            "config": ['{"learning_rate": 1e-5}'],
            "summary": ['{"total_tokens": 999}'],
        }
    )

    context = ProcessingContext.from_config()
    executor = HydrationExecutor.from_context(context)
    hydrated = executor.apply(df, ground_truth_source=runs_df)

    assert hydrated.loc[0, "num_finetuned_tokens_real"] == 999
    assert hydrated.loc[0, "lr"] == "1e-05"


def test_run_normalization_executor_applies_defaults_and_recipes() -> None:
    context = ProcessingContext.from_config()
    executor = RunNormalizationExecutor.from_context(context)
    frame = pd.DataFrame(
        {
            "run_id": ["run-1_Ft_"],
            "initial_checkpoint_recipe": ["d17"],
            "initial_checkpoint_steps": [None],
            "comparison_model_recipe": ["d16"],
            "comparison_metric": [None],
            "num_finetune_tokens_per_epoch": ["5M"],
            "num_finetune_epochs": ["2"],
        }
    )

    normalized = executor.normalize(frame, run_type="matched")

    assert normalized.loc[0, "ckpt_steps"] == "main"
    assert normalized.loc[0, "ckpt_data"] == "Dolma1.7"
    assert normalized.loc[0, "num_finetune_tokens"] == 1_330_254_868
    # matched hook currently leaves metric empty when unmatched
    assert normalized.loc[0, "comparison_metric"] == "pile None"


def test_apply_processing_maps_recipes_after_config_merge() -> None:
    df = pd.DataFrame(
        {
            "run_id": ["2025_08_30-16_54_48_test_run"],
            "initial_checkpoint_recipe": [None],
        }
    )

    runs_df = pd.DataFrame(
        {
            "run_id": ["2025_08_30-16_54_48_test_run"],
            "config": ['{"initial_checkpoint_recipe": "d17"}'],
            "summary": ["{}"],
        }
    )

    processed = apply_processing({"simple_ft": df}, runs_df=runs_df)["simple_ft"]

    assert "ckpt_data" in processed.columns
    assert processed.loc[0, "ckpt_data"] == "Dolma1.7"
