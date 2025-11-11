from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import srsly
from pydantic import BaseModel, Field

from dr_ingest.metrics_all.constants import LoadMetricsAllConfig
from dr_ingest.types import TaskArtifactType
from dr_ingest.utils.io import iter_file_glob_from_roots


def get_artifacts_by_results_dir(data: dict[str, Any]) -> dict[TaskArtifactType, str]:
    cfg = data["cfg"]
    root_paths = [data["results_dir"]]

    artifacts_by_type = {}
    for artifact_type, suffix in cfg.task_file_suffixes.items():
        artifacts_by_type[artifact_type] = iter_file_glob_from_roots(
            root_paths=root_paths,
            file_glob=f"{cfg.task_file_prefix}*{suffix}",
        )
    return artifacts_by_type


class EvalRecordSet(BaseModel):
    cfg: LoadMetricsAllConfig = Field(default_factory=LoadMetricsAllConfig)
    metrics_all_file: str
    results_dir: str = Field(
        default_factory=lambda data: Path(data["metrics_all_file"]).parent
    )
    artifacts: dict[TaskArtifactType, str] = Field(
        default_factory=lambda data: get_artifacts_by_results_dir(data)
    )


class EvalRecord(BaseModel):
    metrics_all_file: str
    results_dir: str = Field(
        default_factory=lambda data: Path(data["metrics_all_file"].parent)
    )
    eval_data: dict[str, Any]

    @classmethod
    def dedupe_by_task(
        cls,
        metrics_all_file: str,
        records: Iterable[dict[str, Any]],
    ) -> list[EvalRecord]:
        serialized_records = set()
        unique_records = []
        for record in records:
            serialized = srsly.json_dumps(record)
            if serialized not in serialized_records:
                serialized_records.add(serialized)
                unique_records.append(
                    cls(
                        metrics_all_file=metrics_all_file,
                        eval_data=record,
                    )
                )
        return unique_records
