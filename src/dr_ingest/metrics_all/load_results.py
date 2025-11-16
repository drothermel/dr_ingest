from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Iterator, Sequence

from dr_ingest.utils.io import iter_file_glob_from_roots

from .constants import LoadMetricsAllConfig
from .eval_record import EvalRecordSet

__all__ = ["iter_metrics_files", "load_all_results"]


def _normalize_roots(
    root_paths: Path | str | Iterable[Path | str] | None,
    *,
    fallback: Sequence[Path | str],
) -> list[Path | str]:
    if root_paths is None:
        return list(fallback)
    if isinstance(root_paths, (str, Path)):
        return [root_paths]
    return list(root_paths)


def iter_metrics_files(
    root_paths: Path | str | Iterable[Path | str] | None,
    *,
    config: LoadMetricsAllConfig,
) -> Iterator[Path]:
    normalized_roots = _normalize_roots(root_paths, fallback=config.root_paths)
    yield from iter_file_glob_from_roots(
        normalized_roots,
        file_glob=config.results_filename,
    )


def load_all_results(
    root_paths: Path | str | Iterable[Path | str] | None = None,
    *,
    config: LoadMetricsAllConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = config or LoadMetricsAllConfig()
    normalized_roots = _normalize_roots(root_paths, fallback=cfg.root_paths)
    records: list[dict[str, Any]] = []
    for metrics_path in iter_file_glob_from_roots(
        normalized_roots,
        file_glob=cfg.results_filename,
    ):
        record_set = EvalRecordSet(cfg=cfg, metrics_all_file=metrics_path)
        records.extend(record_set.load())
    return records
