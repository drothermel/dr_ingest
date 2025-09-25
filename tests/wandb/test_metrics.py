from __future__ import annotations

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


def test_canonicalize_metric_label_exact_match() -> None:
    assert canonicalize_metric_label("pile valppl") == "pile valppl"
    assert canonicalize_metric_label("arc_easy correct_prob") == "arc easy correct prob"


def test_canonicalize_metric_label_empty_when_incomplete() -> None:
    assert canonicalize_metric_label("pile") == "pile"
    assert canonicalize_metric_label("unknown metric") == "unknown metric"


def test_parse_metric_label_expected_cases() -> None:
    for raw, expected in EXPECTED_LABEL_PARSE_CASES.items():
        assert parse_metric_label(raw) == expected
