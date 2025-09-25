from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from attrs import define

from dr_ingest.normalization import normalize_key, normalize_str

TokenSeq = tuple[str, ...]


@define(frozen=True)
class TaskMetricUnmatched:
    task: str | None
    metric: str | None
    unmatched: str | None


@define(frozen=True)
class TaskEntry:
    canonical: str
    tokens: TokenSeq


@define(frozen=True)
class MetricEntry:
    canonical: str
    tokens: TokenSeq


def _tokenize(value: Any) -> TokenSeq:
    normalized = normalize_str(value)
    if not normalized:
        return ()
    return tuple(part for part in normalized.split(" ") if part)


@define(frozen=True)
class MetricCatalog:
    tasks: Sequence[TaskEntry]
    metrics: dict[TokenSeq, MetricEntry]
    tasks_sorted: Sequence[TaskEntry]

    @classmethod
    def from_config(cls) -> MetricCatalog:
        from .config import load_metric_names, load_metric_task_groups

        metric_entries: dict[TokenSeq, MetricEntry] = {}
        for raw_metric in load_metric_names():
            tokens = _tokenize(raw_metric)
            if tokens:
                canonical = normalize_key(raw_metric)
                metric_entries.setdefault(tokens, MetricEntry(canonical, tokens))

        tasks: list[TaskEntry] = []
        for names in load_metric_task_groups().values():
            for raw_task in names:
                tokens = _tokenize(raw_task)
                if tokens:
                    tasks.append(TaskEntry(normalize_key(raw_task), tokens))

        tasks_sorted = sorted(tasks, key=lambda entry: len(entry.tokens), reverse=True)
        return cls(
            tasks=tuple(tasks), metrics=metric_entries, tasks_sorted=tuple(tasks_sorted)
        )

    def match_task(self, tokens: TokenSeq) -> tuple[TaskEntry, int, int] | None:
        for entry in self.tasks_sorted:
            length = len(entry.tokens)
            if length == 0 or length > len(tokens):
                continue
            if tokens[:length] == entry.tokens:
                return entry, 0, length
        return None

    def match_metric(self, tokens: TokenSeq) -> MetricEntry | None:
        if not tokens:
            return None
        return self.metrics.get(tokens)


@lru_cache(maxsize=1)
def _catalog() -> MetricCatalog:
    return MetricCatalog.from_config()


def parse_metric_label(value: Any) -> TaskMetricUnmatched:
    catalog = _catalog()
    tokens = _tokenize(value)
    if not tokens:
        return TaskMetricUnmatched(None, None, None)

    match = catalog.match_task(tokens)
    if match is None:
        return TaskMetricUnmatched(None, None, " ".join(tokens))

    task_entry, _, end_index = match
    remaining = tokens[end_index:]

    metric_entry = catalog.match_metric(remaining)
    if metric_entry is not None:
        return TaskMetricUnmatched(task_entry.canonical, metric_entry.canonical, None)

    unmatched = " ".join(remaining) if remaining else None
    return task_entry.canonical, None, unmatched


def canonicalize_metric_label(value: Any, strict: bool = False) -> str:
    tmu = parse_metric_label(value)
    if strict and (tmu.unmatched is not None or tmu.metric is None or tmu.task is None):
        raise ValueError(f"Invalid metric label: {value}")
    base = f"{tmu.task} f{tmu.metric}"
    if tmu.unmatched:
        base = f"{base} {tmu.unmatched}"
    return base


__all__ = ["canonicalize_metric_label", "parse_metric_label"]
