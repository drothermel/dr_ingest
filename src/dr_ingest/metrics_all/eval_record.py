from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

import srsly

from dr_ingest.metrics_all.constants import LoadMetricsAllConfig
from dr_ingest.types import TaskArtifactType

__all__ = [
    "ArtifactIndex",
    "EvalRecord",
    "EvalRecordSet",
]


@dataclass(frozen=True)
class EvalRecord:
    """Wrapper around raw metrics-all rows with helpers for enrichment."""

    data: dict[str, Any]

    @property
    def task_name(self) -> str:
        value = self.data["task_name"]
        if not isinstance(value, str):
            raise TypeError("task_name must be a string")
        return value

    @property
    def trimmed_task_name(self) -> str:
        return self.task_name.strip()

    @property
    def task_idx(self) -> int | None:
        value = self.data.get("task_idx")
        return value if isinstance(value, int) else None

    def build_task_stem(self, cfg: LoadMetricsAllConfig) -> str | None:
        return cfg.build_task_stem(
            task_idx=self.task_idx,
            trimmed_task_name=self.trimmed_task_name,
        )

    def to_enriched_dict(
        self,
        *,
        cfg: LoadMetricsAllConfig,
        run_dir: Path,
        metrics_all_path: Path,
        artifact_index: "ArtifactIndex",
    ) -> dict[str, Any]:
        stem = self.build_task_stem(cfg)
        artifact_paths = artifact_index.role_paths(stem, cfg.artifact_types)
        resolved_artifacts = {
            role: str(path) if path is not None else None
            for role, path in artifact_paths.items()
        }
        return {
            **self.data,
            "result_dir": str(run_dir),
            "eval_results_path": str(metrics_all_path),
            "task_file_stem": stem,
            **resolved_artifacts,
        }

    @classmethod
    def dedupe_by_task(cls, records: Iterable[dict[str, Any]]) -> list["EvalRecord"]:
        seen_serialized: set[str] = set()
        first_by_task: dict[str, EvalRecord] = {}
        for raw_record in records:
            serialized = srsly.json_dumps(raw_record, sort_keys=True)
            if serialized in seen_serialized:
                continue
            seen_serialized.add(serialized)
            candidate = cls(data=raw_record)
            if candidate.task_name not in first_by_task:
                first_by_task[candidate.task_name] = candidate
        return list(first_by_task.values())


@dataclass(frozen=True)
class TaskArtifacts:
    stem: str
    paths_by_role: Mapping[TaskArtifactType, Path]

    def to_role_mapping(
        self, roles: Iterable[TaskArtifactType]
    ) -> dict[str, Path | None]:
        return {role.value: self.paths_by_role.get(role) for role in roles}


@dataclass(frozen=True)
class ArtifactIndex:
    entries: Mapping[str, TaskArtifacts]

    def role_paths(
        self,
        stem: str | None,
        roles: Iterable[TaskArtifactType],
    ) -> dict[str, Path | None]:
        if stem is None:
            return {role.value: None for role in roles}
        artifact = self.entries.get(stem)
        return artifact.to_role_mapping(roles) if artifact else {
            role.value: None for role in roles
        }

    @classmethod
    def build(cls, directory: Path, cfg: LoadMetricsAllConfig) -> "ArtifactIndex":
        directory = Path(directory)
        index: MutableMapping[str, dict[TaskArtifactType, Path]] = defaultdict(dict)
        for role, suffix in cfg.task_file_suffixes.items():
            pattern = f"{cfg.task_file_prefix}*{suffix}"
            for path in directory.glob(pattern):
                stem = path.name.removesuffix(suffix)
                index[stem][role] = path
        return cls(
            {
                stem: TaskArtifacts(stem=stem, paths_by_role=dict(paths))
                for stem, paths in index.items()
            }
        )


@dataclass(frozen=True)
class EvalRecordSet:
    cfg: LoadMetricsAllConfig
    metrics_all_file: Path

    @property
    def results_dir(self) -> Path:
        return self.metrics_all_file.parent

    def load(self) -> list[dict[str, Any]]:
        raw_records = srsly.read_jsonl(self.metrics_all_file)
        deduped = EvalRecord.dedupe_by_task(raw_records)
        artifact_index = ArtifactIndex.build(self.results_dir, self.cfg)
        return [
            record.to_enriched_dict(
                cfg=self.cfg,
                run_dir=self.results_dir,
                metrics_all_path=self.metrics_all_file,
                artifact_index=artifact_index,
            )
            for record in deduped
        ]
