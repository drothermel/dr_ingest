from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from dr_ingest.pipelines.dd_results import (
    SCALING_LAW_FILENAMES,
    parse_dd_results_train,
    parse_scaling_law_dir,
    parse_train_df,
)

RAW_TO_FINAL_KEYS: dict[str, str] = {
    "acc_raw": "accuracy.raw",
    "acc_per_token": "accuracy.per_token",
    "acc_per_char": "accuracy.per_char",
    "acc_per_byte": "accuracy.per_byte",
    "acc_uncond": "accuracy.uncond",
    "sum_logits_corr": "sum_logits_corr.raw",
    "logits_per_token_corr": "sum_logits_corr.per_token",
    "logits_per_char_corr": "sum_logits_corr.per_char",
    "correct_prob": "correct_prob.raw",
    "correct_prob_per_token": "correct_prob.per_token",
    "correct_prob_per_char": "correct_prob.per_char",
    "margin": "margin.raw",
    "margin_per_token": "margin.per_token",
    "margin_per_char": "margin.per_char",
    "total_prob": "total_prob.raw",
    "total_prob_per_token": "total_prob.per_token",
    "total_prob_per_char": "total_prob.per_char",
    "uncond_correct_prob": "uncond_correct_prob.raw",
    "uncond_correct_prob_per_token": "uncond_correct_prob.per_token",
    "uncond_correct_prob_per_char": "uncond_correct_prob.per_char",
    "norm_correct_prob": "norm_correct_prob.raw",
    "norm_correct_prob_per_token": "norm_correct_prob.per_token",
    "norm_correct_prob_per_char": "norm_correct_prob.per_char",
    "bits_per_byte_corr": "bits_per_byte_correct",
    "primary_metric": "primary_metric",
}


def flatten_metrics(node: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in node.items():
        dotted = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, dotted))
        else:
            flat[dotted] = value
    return flat

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "data" / "dd_results"


@pytest.fixture(scope="module")
def train_fixture() -> pl.DataFrame:
    path = FIXTURE_DIR / "train-00000-of-00004.parquet"
    if not path.exists():
        pytest.skip("train fixture parquet not available")
    return pl.read_parquet(path)


@pytest.fixture(scope="module")
def scaling_law_fixture_dir() -> Path:
    required = [FIXTURE_DIR / filename for filename in SCALING_LAW_FILENAMES]
    missing = [path for path in required if not path.exists()]
    if missing:
        pytest.skip(f"Scaling-law fixture files missing: {missing}")
    return FIXTURE_DIR


def test_parse_dd_results_train_preserves_metrics(train_fixture: pl.DataFrame) -> None:
    parsed = parse_dd_results_train(train_fixture)
    for idx, raw_literal in enumerate(train_fixture["metrics"].to_list()):
        raw_metrics = ast.literal_eval(raw_literal)
        parsed_struct = parsed[idx, "metrics"]
        for key, expected in raw_metrics.items():
            actual = parsed_struct[key]
            if expected is None:
                assert actual is None
            else:
                assert actual == pytest.approx(expected)


def test_parse_train_df_nested_metrics_match(train_fixture: pl.DataFrame) -> None:
    final_df = parse_train_df(train_fixture)
    final_metrics = final_df[0, "metrics"]
    raw_metrics = ast.literal_eval(train_fixture[0, "metrics"])

    assert final_metrics["primary_metric"] == pytest.approx(raw_metrics["primary_metric"])

    for family, raw_key_prefix in [
        ("correct_prob", "correct_prob"),
        ("norm_correct_prob", "norm_correct_prob"),
        ("margin", "margin"),
        ("total_prob", "total_prob"),
        ("uncond_correct_prob", "uncond_correct_prob"),
    ]:
        family_metrics = final_metrics[family]
        assert family_metrics["raw"] == pytest.approx(raw_metrics[raw_key_prefix])
        assert family_metrics["per_token"] == pytest.approx(
            raw_metrics[f"{raw_key_prefix}_per_token"]
        )
        assert family_metrics["per_char"] == pytest.approx(
            raw_metrics[f"{raw_key_prefix}_per_char"]
        )

    accuracy_metrics = final_metrics["accuracy"]
    assert accuracy_metrics["raw"] == pytest.approx(raw_metrics["acc_raw"])
    assert accuracy_metrics["per_token"] == pytest.approx(raw_metrics["acc_per_token"])
    assert accuracy_metrics["per_char"] == pytest.approx(raw_metrics["acc_per_char"])
    assert accuracy_metrics["per_byte"] == pytest.approx(raw_metrics["acc_per_byte"])
    assert accuracy_metrics["uncond"] == pytest.approx(raw_metrics["acc_uncond"])

    assert final_metrics["bits_per_byte_correct"] == pytest.approx(
        raw_metrics["bits_per_byte_corr"]
    )


@pytest.mark.parametrize("row_index", [0, 123, 2048])
def test_parse_train_df_preserves_full_metric_mapping(
    train_fixture: pl.DataFrame,
    row_index: int,
) -> None:
    if row_index >= train_fixture.height:
        pytest.skip(f"Fixture has only {train_fixture.height} rows")

    parsed = parse_train_df(train_fixture)
    flattened = flatten_metrics(parsed[row_index, "metrics"])
    raw_metrics = ast.literal_eval(train_fixture[row_index, "metrics"])

    for raw_key, final_key in RAW_TO_FINAL_KEYS.items():
        assert raw_key in raw_metrics, f"missing raw metric {raw_key}"
        assert final_key in flattened, f"missing mapped metric {final_key}"
        expected = raw_metrics[raw_key]
        actual = flattened[final_key]
        if expected is None:
            assert actual is None
        else:
            assert actual == pytest.approx(expected)


def test_parse_scaling_law_dir_outputs(scaling_law_fixture_dir: Path) -> None:
    outputs = parse_scaling_law_dir(scaling_law_fixture_dir)
    expected_keys = {
        "macro_avg_raw",
        "scaling_law_true_raw",
        "scaling_law_pred_one_step_raw",
        "scaling_law_pred_two_step_raw",
    }

    assert set(outputs) == expected_keys

    macro_output = outputs["macro_avg_raw"]
    macro_expected = pl.read_parquet(
        scaling_law_fixture_dir / "macro_avg-00000-of-00001.parquet"
    )
    assert_frame_equal(macro_output, macro_expected)

    two_step = outputs["scaling_law_pred_two_step_raw"]
    assert not two_step.is_empty()
    required_columns = {
        "id",
        "task",
        "recipe",
        "fit_config",
        "pred_task_losses",
        "pred_task_loss_to_metrics",
        "pred_task_metrics",
    }
    assert required_columns.issubset(two_step.columns)
