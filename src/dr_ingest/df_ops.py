"""Reusable pandas DataFrame helper utilities."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import pandas as pd


MissingMarkers = Iterable[Any]


def ensure_column(
    df: pd.DataFrame,
    column: str,
    default: Any,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Ensure a column exists and fill missing entries with a default value."""

    target = df if inplace else df.copy()
    if column not in target.columns:
        target[column] = default
    else:
        target[column] = target[column].fillna(default)
    return target


def fill_missing_values(
    df: pd.DataFrame,
    defaults: Mapping[str, Any],
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Fill NaNs/None in multiple columns with provided defaults."""

    target = df if inplace else df.copy()
    for column, default in defaults.items():
        if column in target.columns:
            target[column] = target[column].fillna(default)
    return target


def rename_columns(
    df: pd.DataFrame,
    mapping: Mapping[str, str],
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Rename columns using a provided mapping without failing on missing keys."""

    target = df if inplace else df.copy()
    existing_map = {old: new for old, new in mapping.items() if old in target.columns}
    if existing_map:
        target = target.rename(columns=existing_map)
    return target


def map_column_with_fallback(
    df: pd.DataFrame,
    column: str,
    mapping: Mapping[str, Any],
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Map values in a column via dict while preserving unknown entries."""

    if column not in df.columns:
        return df if inplace else df.copy()

    target = df if inplace else df.copy()

    def _mapper(value: Any) -> Any:
        if pd.isna(value):
            return value
        return mapping.get(value, value)

    target[column] = target[column].map(_mapper)
    return target


def apply_column_converters(
    df: pd.DataFrame,
    converters: Mapping[str, Callable[[Any], Any]],
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Apply a mapping of column -> converter callable to a DataFrame."""

    target = df if inplace else df.copy()
    for column, converter in converters.items():
        if column in target.columns:
            target[column] = target[column].apply(converter)
    return target


def maybe_update_cell(
    df: pd.DataFrame,
    row_index: int,
    column: str,
    value: Any,
    *,
    missing_markers: MissingMarkers = (None, "N/A"),
    inplace: bool = False,
) -> pd.DataFrame:
    """Update a single cell only if its current value is considered missing."""

    target = df if inplace else df.copy()
    if column not in target.columns or row_index not in target.index:
        return target

    current = target.at[row_index, column]
    is_missing = pd.isna(current) or current in missing_markers
    if is_missing:
        target.at[row_index, column] = value
    return target


def apply_if_column(
    df: pd.DataFrame,
    column: str,
    func: Callable[[pd.Series], pd.Series],
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Apply a function to a column only if it exists."""

    if column not in df.columns:
        return df if inplace else df.copy()

    target = df if inplace else df.copy()
    target[column] = func(target[column])
    return target


__all__ = [
    "ensure_column",
    "fill_missing_values",
    "rename_columns",
    "map_column_with_fallback",
    "apply_column_converters",
    "maybe_update_cell",
    "apply_if_column",
]
