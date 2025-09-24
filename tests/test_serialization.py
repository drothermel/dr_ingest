from __future__ import annotations

from pathlib import Path

import pandas as pd

from dr_ingest.serialization import (
    compare_sizes,
    dump_runs_and_history,
    ensure_parquet,
)


def test_dump_and_convert(tmp_path: Path) -> None:
    runs = [
        {"run_id": "run-1", "config": {"lr": 1e-4}},
        {"run_id": "run-2", "config": {"lr": 5e-5}},
    ]
    histories = [[{"run_id": "run-1", "metrics": {"loss": 0.5}}]]

    dump_runs_and_history(tmp_path, "runs", "history", runs, histories)

    runs_json = tmp_path / "runs.jsonl"
    history_json = tmp_path / "history.jsonl"
    assert runs_json.exists()
    assert history_json.exists()

    runs_parquet = ensure_parquet(runs_json)
    history_parquet = ensure_parquet(history_json)

    assert runs_parquet.exists()
    assert history_parquet.exists()

    runs_df = pd.read_parquet(runs_parquet)
    assert set(runs_df["run_id"]) == {"run-1", "run-2"}

    sizes = compare_sizes(runs_json, runs_parquet, history_json, history_parquet)
    assert str(runs_json) in sizes
    assert str(runs_parquet) in sizes
