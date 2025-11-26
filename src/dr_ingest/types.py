from enum import StrEnum


class TaskArtifactType(StrEnum):
    PREDICTIONS = "predictions"
    RECORDED_INPUTS = "recorded-inputs"
    REQUESTS = "requests"
    CONFIG = "config"
    METRICS = "metrics"

    @property
    def extension(self) -> str:
        mapping = {
            TaskArtifactType.PREDICTIONS: "jsonl",
            TaskArtifactType.RECORDED_INPUTS: "jsonl",
            TaskArtifactType.REQUESTS: "jsonl",
            TaskArtifactType.CONFIG: "json",
            TaskArtifactType.METRICS: "json",
        }
        return mapping[self]

    @property
    def filename_pattern(self) -> str:
        return f"*-{self.value}.{self.extension}"
