from __future__ import annotations

from pathlib import Path

import pytest

from dr_ingest.configs import Paths
from dr_ingest.metrics_all.constants import LoadMetricsAllConfig
from dr_ingest.metrics_all.load_results import load_all_results


def _require_metrics_root() -> Path:
    metrics_root = Paths().metrics_all_dir
    if not metrics_root.exists():
        pytest.skip(f"metrics root does not exist: {metrics_root}")
    return metrics_root


def test_load_all_results_enriches_records() -> None:
    metrics_root = _require_metrics_root()
    records = load_all_results(root_paths=[metrics_root])
    assert records, "Expected at least one record from metrics-all ingest"
    first_record = records[0]
    assert "result_dir" in first_record
    assert "eval_results_path" in first_record
    assert Path(first_record["eval_results_path"]).exists()


def test_artifact_paths_exist_when_reported() -> None:
    metrics_root = _require_metrics_root()
    cfg = LoadMetricsAllConfig(root_paths=[metrics_root])
    records = load_all_results(root_paths=[metrics_root], config=cfg)
    artifact_keys = [artifact_type.value for artifact_type in cfg.artifact_types]
    validated = 0
    for record in records:
        for key in artifact_keys:
            path_str = record.get(key)
            if path_str is None:
                continue
            assert Path(path_str).exists()
            validated += 1
        if validated >= len(artifact_keys):
            break
    if validated == 0:
        pytest.skip("No artifact files were referenced in the sampled records")
