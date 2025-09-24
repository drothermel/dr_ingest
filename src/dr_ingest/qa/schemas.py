"""Typed schemas for QA evaluation instances."""

from __future__ import annotations

from attrs import define


@define
class ModelAnswerOutput:
    is_greedy: bool
    logits_per_byte: float
    logits_per_char: float
    logits_per_token: float
    num_chars: int
    num_tokens: int
    num_tokens_all: int
    sum_logits: float
    sum_logits_uncond: float


@define
class QuestionOutputData:
    doc_id: int
    native_id: int
    label: int
    answer_outputs: list[ModelAnswerOutput]


@define
class TaskOutputData:
    task_hash: str
    model_hash: str
    data: str
    params: str
    seed: int
    task: str
    step: int
    question_outputs: list[QuestionOutputData]


__all__ = [
    "ModelAnswerOutput",
    "QuestionOutputData",
    "TaskOutputData",
]
