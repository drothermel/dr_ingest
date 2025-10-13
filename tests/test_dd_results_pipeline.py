from __future__ import annotations

import ast
from pathlib import Path

import polars as pl
import pytest

from dr_ingest.pipelines.dd_results import parse_dd_results_train, parse_train_df

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "data" / "dd_results"


@pytest.fixture(scope="module")
def train_fixture() -> pl.DataFrame:
    path = FIXTURE_DIR / "train-00000-of-00004.parquet"
    if not path.exists():
        pytest.skip("train fixture parquet not available")
    return pl.read_parquet(path)


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
