from enum import StrEnum


class TaskArtifactType(StrEnum):
    PREDICTIONS = "predictions"
    RECORDED_INPUTS = "recorded_inputs"
    REQUESTS = "requests"
