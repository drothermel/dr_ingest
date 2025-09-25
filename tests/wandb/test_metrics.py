from __future__ import annotations

import pandas as pd

from dr_ingest.wandb.metrics import canonicalize_metric_label, parse_metric_label


EXPECTED_LABEL_PARSE_CASES = {
    "pile": ("pile", None, None),
    "pile valppl": ("pile", "valppl", None),
    "pile weird_metric": ("pile", None, "weird metric"),
    "arc_easy_correct_prob": ("arc_easy", "correct_prob", None),
    "arc easy correct prob per token": (
        "arc_easy",
        "correct_prob_per_token",
        None,
    ),
    "eval/pile-validation/Perplexity": (
        None,
        None,
        "eval pile validation perplexity",
    ),
    "mmlu_high_school_biology primary_metric": (
        "mmlu_high_school_biology",
        "primary_metric",
        None,
    ),
    "boolq margin": ("boolq", "margin", None),
    "dolma_common-crawl": ("dolma_common_crawl", None, None),
    "unknown_metric_name": (None, None, "unknown metric name"),
    None: (None, None, None),
    "": (None, None, None),
    "   ": (None, None, None),
}


def test_canonicalize_metric_label_perplexity_aliases() -> None:
    assert canonicalize_metric_label("pile") == "pile-valppl"
    assert canonicalize_metric_label("C4") == "c4_en-valppl"
    assert canonicalize_metric_label("eval/pile-validation/Perplexity") == "pile-valppl"
    assert canonicalize_metric_label("c4_en-valppl") == "c4_en-valppl"


def test_canonicalize_metric_label_olmes_pairings() -> None:
    assert canonicalize_metric_label("arc_easy_correct_prob") == "arc_easy_correct_prob"
    assert canonicalize_metric_label("arc_easy-correct_prob") == "arc_easy_correct_prob"
    assert (
        canonicalize_metric_label("arc_easy-correct_prob_per_token")
        == "arc_easy_correct_prob_per_token"
    )


def test_canonicalize_metric_label_missing_returns_default() -> None:
    assert canonicalize_metric_label(None) == "pile-valppl"
    assert canonicalize_metric_label(pd.NA) == "pile-valppl"


def test_canonicalize_metric_label_unknown_falls_back() -> None:
    assert canonicalize_metric_label("unknown_metric") == "pile-valppl"


def test_canonicalize_metric_label_with_custom_default() -> None:
    assert canonicalize_metric_label("unknown_metric", default="fallback") == "fallback"


def test_canonicalize_metric_label_handles_embedded_tokens() -> None:
    assert canonicalize_metric_label("eval/pile-validation/Perplexity") == "pile-valppl"


def test_canonicalize_metric_label_unmatched_metric_defaults() -> None:
    assert canonicalize_metric_label("pile weird_metric") == "pile-valppl"


def test_parse_metric_label_expected_cases() -> None:
    for raw, expected in EXPECTED_LABEL_PARSE_CASES.items():
        assert parse_metric_label(raw) == expected
