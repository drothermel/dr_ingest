from enum import StrEnum


class TaskArtifactType(StrEnum):
    PREDICTIONS = "predictions"
    RECORDED_INPUTS = "recorded-inputs"
    REQUESTS = "requests"
    CONFIG = "config"
    METRICS = "metrics"

    @property
    def extension(self) -> str:
        match self:
            case (
                TaskArtifactType.PREDICTIONS
                | TaskArtifactType.RECORDED_INPUTS
                | TaskArtifactType.REQUESTS
            ):
                return "jsonl"
            case TaskArtifactType.CONFIG | TaskArtifactType.METRICS:
                return "json"
            case _:
                raise ValueError(f"Invalid task artifact type: {self}")

    @property
    def filename_pattern(self) -> str:
        return f"*-{self.value}.{self.extension}"
