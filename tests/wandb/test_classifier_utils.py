from __future__ import annotations

import pandas as pd

from dr_ingest.wandb.classifier import (
    RunClassification,
    group_classifications_by_type,
    iter_classified_runs,
    parse_and_group_run_ids,
)


def test_iter_classified_runs_generates_records() -> None:
    df = pd.DataFrame({"run_id": ["abc", "main_default_run"]})
    classifications = list(iter_classified_runs(df))
    assert classifications[0].run_id == "abc"
    assert classifications[1].run_type == "old"


def test_group_classifications_by_type_filters_dropped() -> None:
    classifications = [
        RunClassification(run_id="run-new", run_type="simple_ft", metadata={}),
        RunClassification(run_id="run-old", run_type="old", metadata={}),
    ]
    grouped = group_classifications_by_type(classifications, drop_run_types={"old"})
    assert "old" not in grouped
    assert grouped["simple_ft"][0]["run_id"] == "run-new"


def test_parse_and_group_respects_drop_set() -> None:
    df = pd.DataFrame({"run_id": ["main_default_old", "abc"]})
    grouped = parse_and_group_run_ids(df, drop_run_types=("old",))
    assert "old" not in grouped
    assert "abc" in [rec["run_id"] for rec in grouped["other"]]


def test_parse_and_group_includes_legacy_when_not_dropped() -> None:
    df = pd.DataFrame({"run_id": ["main_default_old"]})
    grouped = parse_and_group_run_ids(df, drop_run_types=None)
    assert grouped["old"][0]["run_id"] == "main_default_old"
