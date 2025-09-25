from __future__ import annotations

import pandas as pd

from dr_ingest.normalization import (
    convert_string_to_number,
    convert_timestamp,
    is_nully,
    normalize_numeric,
    normalize_str,
)


def test_convert_timestamp_formats() -> None:
    assert convert_timestamp("250901-155734") == pd.Timestamp("2025-09-01 15:57:34")
    assert convert_timestamp("2025_08_30-16_54_48") == pd.Timestamp(
        "2025-08-30 16:54:48"
    )
    assert convert_timestamp("not-a-ts") is None


def test_convert_string_to_number_suffixes() -> None:
    assert convert_string_to_number("10M") == 10_000_000
    assert convert_string_to_number("1.5G") == 1_500_000_000
    assert convert_string_to_number(" ") is None
    assert convert_string_to_number(None) is None
    assert convert_string_to_number("2.5Mt") == 2_500_000
    assert convert_string_to_number("4gt") == 4_000_000_000
    assert convert_string_to_number("7B") == 7_000_000_000
    assert convert_string_to_number("5T") == 5_000_000_000_000
    assert convert_string_to_number("100XYZ") is None


def test_is_nully_detects_values() -> None:
    assert is_nully(None)
    assert is_nully(" ")
    assert is_nully("\t\n")
    assert is_nully(float("nan"))
    assert is_nully(object()) is False


def test_normalize_str_basic() -> None:
    assert normalize_str("  Pile-ValPPL  ") == "pile valppl"
    assert normalize_str("foo/bar") == "foo bar"
    assert normalize_str(None) is None
    assert normalize_str("  ") is None


def test_normalize_numeric_basic() -> None:
    assert normalize_numeric("1.23") == 1.23
    assert normalize_numeric(5) == 5.0
    assert normalize_numeric("  ") is None
    assert normalize_numeric("not a number") is None
