from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from dr_ingest.df_ops import (
    apply_column_converters,
    apply_if_column,
    ensure_column,
    fill_missing_values,
    masked_setter,
    force_set_cell,
    map_column_with_fallback,
    maybe_update_cell,
    rename_columns,
    require_row_index,
)


def _df(data: dict[str, list[Any]]) -> pd.DataFrame:
    return pd.DataFrame(data)


def test_ensure_column_creates_and_fills() -> None:
    df = _df({"a": [1, None]})
    result = ensure_column(df, "b", 0)
    assert "b" in result.columns
    assert result["b"].tolist() == [0, 0]
    result2 = ensure_column(df, "a", 5)
    assert result2["a"].tolist() == [1, 5]


def test_fill_missing_values_applies_defaults() -> None:
    df = _df({"a": [1, None], "b": [None, "x"]})
    result = fill_missing_values(df, {"a": 0, "b": "default"})
    assert result["a"].tolist() == [1, 0]
    assert result["b"].tolist() == ["default", "x"]


def test_rename_columns_only_existing() -> None:
    df = _df({"old": [1], "keep": [2]})
    result = rename_columns(df, {"old": "new", "missing": "ignored"})
    assert "new" in result.columns and "old" not in result.columns
    assert "keep" in result.columns


def test_map_column_with_fallback_preserves_unknowns() -> None:
    df = _df({"col": ["x", "y", None]})
    result = map_column_with_fallback(df, "col", {"x": "mapped"})
    assert result["col"].tolist() == ["mapped", "y", None]


def test_apply_column_converters() -> None:
    df = _df({"a": ["1", "2"], "b": ["upper", "case"]})
    result = apply_column_converters(df, {"a": int, "b": str.upper})
    assert result["a"].tolist() == [1, 2]
    assert result["b"].tolist() == ["UPPER", "CASE"]


def test_maybe_update_cell_respects_missing_markers() -> None:
    df = _df({"col": [None, "value", "N/A"]})
    result = maybe_update_cell(df, 0, "col", "filled")
    assert result.iloc[0]["col"] == "filled"
    result = maybe_update_cell(result, 1, "col", "new")
    assert result.iloc[1]["col"] == "value"
    result = maybe_update_cell(result, 2, "col", "new")
    assert result.iloc[2]["col"] == "new"


def test_apply_if_column_runs_function() -> None:
    df = _df({"exists": [1, 2], "other": [3, 4]})
    result = apply_if_column(df, "exists", lambda s: s * 10)
    assert result["exists"].tolist() == [10, 20]
    # missing column should leave DataFrame unchanged
    result2 = apply_if_column(df, "unknown", lambda s: s * 10)
    pd.testing.assert_frame_equal(result2, df)


def test_require_row_index_success_and_errors() -> None:
    df = _df({"run_id": ["a", "b"]})
    assert require_row_index(df, "run_id", "a") == 0
    with pytest.raises(ValueError):
        require_row_index(df, "run_id", "missing")
    dup_df = _df({"run_id": ["a", "a"]})
    with pytest.raises(ValueError):
        require_row_index(dup_df, "run_id", "a")


def test_force_set_cell_ensures_column_and_sets_value() -> None:
    df = _df({"run_id": ["a"]})
    updated = force_set_cell(df, 0, "col", 5)
    assert updated.loc[0, "col"] == 5
    # inplace update
    force_set_cell(df, 0, "col", 10, inplace=True)
    assert df.loc[0, "col"] == 10


def test_masked_setter_creates_column_and_sets_matches() -> None:
    df = _df({"run_id": ["a", "b"], "flag": [False, True]})
    mask = df["flag"]
    result = masked_setter(df.copy(), mask, "new_col", "value")
    assert result.loc[result["run_id"] == "b", "new_col"].item() == "value"
    assert pd.isna(result.loc[result["run_id"] == "a", "new_col"].item())
