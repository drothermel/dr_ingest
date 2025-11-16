from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dr_ingest.configs import Paths
from dr_ingest.types import TaskArtifactType
from dr_ingest.utils.display import add_marimo_display


@add_marimo_display()
class LoadMetricsAllConfig(BaseModel):
    """Configuration shared by all metrics-all ingestion utilities."""

    model_config = ConfigDict(validate_default=True, frozen=True)

    root_paths: Iterable[Path | str] = Field(
        default_factory=lambda: [Paths().metrics_all_dir]
    )
    results_filename: str = "metrics-all.jsonl"
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

    @field_validator("results_filename", "task_file_prefix", "stem_separator")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("string fields must be non-empty")
        return value

    @field_validator("task_idx_width")
    @classmethod
    def _positive_width(cls, value: int) -> int:
        if value < 1:
            raise ValueError("task_idx_width must be at least 1")
        return value

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
