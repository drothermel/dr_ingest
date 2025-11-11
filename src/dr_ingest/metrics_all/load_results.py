from collections.abc import Iterable
from pathlib import Path
from typing import Any

import srsly

from dr_ingest.utils.io import iter_file_glob_from_roots

from .constants import LoadMetricsAllConfig
from .eval_record import EvalRecordSet

__all__ = ["iter_file_glob_from_roots", "load_all_results"]


def load_all_results(
    root_paths: Iterable[Path | str],
    config: LoadMetricsAllConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = config or LoadMetricsAllConfig(root_paths=root_paths)
    records = []
    for eval_path in iter_file_glob_from_roots(
        root_paths,
        file_glob=cfg.results_filename,
    ):
        _ = EvalRecordSet(metrics_all_file=str(eval_path))
        # TODO: finish implementing
    raise NotImplementedError("Implement this function")
    return records
