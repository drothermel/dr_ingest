from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dr_ingest.utils.io import iter_file_glob_from_roots

from .constants import LoadMetricsAllConfig
from .eval_record import EvalRecordSet

__all__ = ["load_all_results"]


def load_all_results(
    *,
    root_paths: Path | str | Iterable[Path | str] | None = None,
    config: LoadMetricsAllConfig | None = None,
) -> list[dict[str, Any]]:
    assert root_paths or config, (
        "Either root_paths or LoadMetricsallConfig must be provided"
    )
    cfg = config or LoadMetricsAllConfig(root_paths=root_paths)  # type: ignore
    records: list[dict[str, Any]] = []
    for metrics_path in iter_file_glob_from_roots(
        cfg.root_paths, file_glob=cfg.results_filename
    ):
        record_set = EvalRecordSet(cfg=cfg, metrics_all_file=metrics_path)
        records.extend(record_set.load())
    return records
