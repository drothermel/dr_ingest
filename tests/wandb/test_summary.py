from __future__ import annotations

from dr_ingest.wandb.summary import normalize_oe_summary


def test_normalize_oe_summary_basic() -> None:
    summary = {
        "other_metric": 1,
        "oe_eval_metrics/task1/accuracy": 0.8,
        "oe_eval_metrics/task1/loss": 0.3,
        "oe_eval_metrics/task2": 0.5,
        "oe_eval_metrics/task3/extra": None,
    }

    result = normalize_oe_summary(summary)
    assert "task1" in result
    assert result["task1"]["accuracy"] == 0.8
    assert result["task2"] == 0.5
    assert "task3" not in result  # dropped because only None values


def test_normalize_oe_summary_with_extra_metrics() -> None:
    summary = {
        "oe_eval_metrics/task1/accuracy": 0.9,
        "oe_eval_metrics/task1/extra_metrics": {"precision": 0.7},
    }

    result = normalize_oe_summary(summary)
    assert result["task1"]["precision"] == 0.7


def test_normalize_oe_summary_drops_null_extra_metrics() -> None:
    summary = {
        "oe_eval_metrics/task1/accuracy": 0.9,
        "oe_eval_metrics/task1/extra_metrics": {"precision": None},
    }

    result = normalize_oe_summary(summary)
    assert "precision" not in result["task1"]


def test_normalize_oe_summary_mixed_scalar_and_nested() -> None:
    summary = {
        "oe_eval_metrics/task1": 0.5,
        "oe_eval_metrics/task1/accuracy": 0.8,
    }

    result = normalize_oe_summary(summary)
    assert result["task1"]["value"] == 0.5
    assert result["task1"]["accuracy"] == 0.8
