from __future__ import annotations

import pandas as pd

from dr_ingest.wandb.metrics import canonicalize_metric_label


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
