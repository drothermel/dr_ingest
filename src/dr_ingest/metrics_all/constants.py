from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from dr_ingest.configs import Paths
from dr_ingest.types import TaskArtifactType


class LoadMetricsAllConfig(BaseModel):
    model_config = ConfigDict(validate_default=True)
    root_paths: Iterable[Path | str] = Field(
        default_factory=lambda: [Paths().metrics_all_dir]
    )

    results_filename: str = "metrics-all.jsonl"
    task_file_prefix: str = "task-"
    task_file_suffixes: dict[TaskArtifactType, str] = Field(
        default_factory=lambda: {
            "predictions": "-predictions.jsonl",
            "recorded_inputs": "-recorded-inputs.jsonl",
            "requests": "-requests.jsonl",
        }
    )
