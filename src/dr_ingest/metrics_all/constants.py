from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from dr_ingest.types import TaskArtifactType
from dr_ingest.utils.display import add_marimo_display


@add_marimo_display()
class LoadMetricsAllConfig(BaseModel):
    """Configuration shared by all metrics-all ingestion utilities."""

    root_paths: list[Path]
    results_filename: str = "metrics-all.jsonl"

    cache_subdir: str = "new_ingest"
    prefix_map: dict[TaskArtifactType, str] = Field(
        default_factory=lambda: {
            TaskArtifactType.PREDICTIONS: "prd",
            TaskArtifactType.RECORDED_INPUTS: "inp",
            TaskArtifactType.REQUESTS: "req",
            TaskArtifactType.CONFIG: "cfg",
            TaskArtifactType.METRICS: "met",
        }
    )

    skip_map: dict[TaskArtifactType, set[str]] = Field(
        default_factory=lambda: {
            TaskArtifactType.PREDICTIONS: {"file_prefix", "doc_id", "idx"},
            TaskArtifactType.RECORDED_INPUTS: {"file_prefix", "doc_id", "idx"},
            TaskArtifactType.REQUESTS: {"file_prefix", "doc_id", "idx"},
            TaskArtifactType.CONFIG: {"file_prefix", "doc_id"},
            TaskArtifactType.METRICS: {"file_prefix", "doc_id"},
        }
    )

    select_parts: list[str] = Field(
        default_factory=lambda: [
            "dk.file_prefix",
            "dk.doc_id",
            "dk.idx",
        ]
    )

    task_file_prefix: str = "task-"
    task_idx_width: int = 3
    stem_separator: str = "-"
    task_file_suffixes: dict[TaskArtifactType, str] = Field(
        default_factory=lambda: {
            TaskArtifactType.PREDICTIONS: "-predictions.jsonl",
            TaskArtifactType.RECORDED_INPUTS: "-recorded-inputs.jsonl",
            TaskArtifactType.REQUESTS: "-requests.jsonl",
        }
    )
    path_cols: list[str] = Field(
        default_factory=lambda: [
            "result_dir",
            "eval_results_path",
            "predictions",
            "recorded_inputs",
            "requests",
        ]
    )
    dict_cols: list[str] = Field(
        default_factory=lambda: [
            "beaker_info",
            "model_config",
            "task_config",
            "compute_config",
            "metrics",
        ]
    )

    @field_validator("root_paths", mode="before")
    def validate_root_paths(cls, value: Any) -> list[Path]:
        if not isinstance(value, list | str | Path):
            raise ValueError("root_paths must be a list, string or Path")
        if isinstance(value, str | Path):
            value = [value]
        if not all(isinstance(path, str | Path) for path in value):
            raise ValueError("root_paths must be a list of Path objects")
        return [Path(p_or_s) for p_or_s in value]

    @property
    def artifact_types(self) -> tuple[TaskArtifactType, ...]:
        return tuple(self.task_file_suffixes.keys())

    def build_task_stem(
        self, *, task_idx: int | None, trimmed_task_name: str | None
    ) -> str | None:
        """Return canonical ``task-XXX-<name>`` stems when possible."""

        if task_idx is None or not trimmed_task_name:
            return None
        idx = f"{task_idx:0{self.task_idx_width}d}"
        return f"{self.task_file_prefix}{idx}{self.stem_separator}{trimmed_task_name}"
