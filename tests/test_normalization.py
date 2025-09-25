from __future__ import annotations

from dr_ingest.normalization import (
    is_nully,
    normalize_key,
    key_variants,
    normalize_numeric,
    normalize_str,
    split_by_known_prefix,
)


def test_normalize_key_basic() -> None:
    assert normalize_key("Pile-ValPPL") == "pile_valppl"
    assert normalize_key("  c4/en  ") == "c4_en"


def test_key_variants_generates_forms() -> None:
    variants = key_variants("Pile-ValPPL")
    assert "pile_valppl" in variants
    assert "pile-valppl" in variants
    assert "pilevalppl" in variants


def test_split_by_known_prefix_matches_longest() -> None:
    prefix, remainder = split_by_known_prefix(
        "arc_easy_correct_prob", ["arc", "arc_easy", "mmlu_average"]
    )
    assert prefix == "arc_easy"
    assert remainder == "correct_prob"


def test_split_by_known_prefix_handles_unknown() -> None:
    prefix, remainder = split_by_known_prefix("unknown_metric", ["arc_easy"])
    assert prefix == "unknown_metric"
    assert remainder is None


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
