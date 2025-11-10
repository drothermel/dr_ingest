from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from dr_ingest.configs.datadecide import DataDecideConfig
from dr_ingest.pipelines.dd_results import (
    parse_dd_results_train,
    parse_train_df,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "data" / "dd_results"


@pytest.fixture(scope="module")
def train_fixture() -> pd.DataFrame:
    path = FIXTURE_DIR / "train-00000-of-00004.parquet"
    if not path.exists():
        pytest.skip("train fixture parquet not available")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def config() -> DataDecideConfig:
    return DataDecideConfig()


def test_parse_dd_results_train_preserves_metrics(
    train_fixture: pd.DataFrame, config: DataDecideConfig
) -> None:
    parsed = parse_dd_results_train(train_fixture, config=config)

    for idx in range(len(train_fixture)):
        raw_literal = train_fixture.iloc[idx]["metrics"]
        raw_metrics = ast.literal_eval(raw_literal)

        for raw_key, normalized_key in config.metric_column_renames.items():
            if raw_key in raw_metrics:
                expected = raw_metrics[raw_key]
                actual = parsed.iloc[idx][normalized_key]

                if expected is None or pd.isna(expected):
                    assert pd.isna(actual) or actual is None
                else:
                    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("row_index", [0, 123, 2048])
def test_parse_train_df_preserves_full_metric_mapping(
    train_fixture: pd.DataFrame,
    config: DataDecideConfig,
    row_index: int,
) -> None:
    if row_index >= len(train_fixture):
        pytest.skip(f"Fixture has only {len(train_fixture)} rows")

    parsed = parse_train_df(train_fixture, config=config)
    raw_metrics = ast.literal_eval(train_fixture.iloc[row_index]["metrics"])

    for raw_key, normalized_key in config.metric_column_renames.items():
        assert raw_key in raw_metrics, f"missing raw metric {raw_key}"
        assert normalized_key in parsed.columns, f"missing mapped metric {normalized_key}"

        expected = raw_metrics[raw_key]
        actual = parsed.iloc[row_index][normalized_key]

        if expected is None or pd.isna(expected):
            assert pd.isna(actual) or actual is None
        else:
            assert actual == pytest.approx(expected)


def test_parse_train_df_has_required_columns(
    train_fixture: pd.DataFrame, config: DataDecideConfig
) -> None:
    parsed = parse_train_df(train_fixture, config=config)

    required_columns = {"id", "recipe", "tokens_millions", "compute_e15"}
    assert required_columns.issubset(parsed.columns)

    assert "data" not in parsed.columns
    assert "chinchilla" not in parsed.columns
    assert "tokens" not in parsed.columns
    assert "compute" not in parsed.columns
    assert "metrics" not in parsed.columns


def test_parse_train_df_normalizations(
    train_fixture: pd.DataFrame, config: DataDecideConfig
) -> None:
    parsed = parse_train_df(train_fixture, config=config)

    assert "recipe" in parsed.columns
    assert "tokens_millions" in parsed.columns
    assert "compute_e15" in parsed.columns

    assert parsed["tokens_millions"].dtype in [float, "float64"]
    assert parsed["compute_e15"].dtype in [float, "float64"]
