from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from attrs import define

from dr_ingest.normalization import normalize_str
from dr_ingest.wandb.config import load_metric_names, load_metric_task_groups

TokenSeq = tuple[str, ...]


@define(frozen=True)
class TaskMetricUnmatched:
    task: str | None
    metric: str | None
    unmatched: str | None


@define(frozen=True)
class Entry:
    canonical: str
    tokens: TokenSeq

    def __str__(self) -> str:
        return f"Entry({self.canonical})"

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def from_raw_str(cls, raw_str: str) -> Entry | None:
        normalized = normalize_str(raw_str)
        if not normalized:
            return None
        tokens = tuple(part for part in normalized.split(" ") if part)
        return cls(normalized, tokens)


def _tokenize(value: Any) -> TokenSeq:
    normalized = normalize_str(value)
    if not normalized:
        return ()
    return tuple(part for part in normalized.split(" ") if part)


@define(frozen=True)
class MetricCatalog:
    tasks: Sequence[Entry]
    metrics: dict[TokenSeq, Entry]
    tasks_sorted: Sequence[Entry]

    @classmethod
    def from_config(cls) -> MetricCatalog:
        metric_entries: dict[TokenSeq, Entry] = {}
        for raw_metric in load_metric_names():
            metric_entry = Entry.from_raw_str(raw_metric)
            assert metric_entry is not None, f"Invalid metric name: {raw_metric}"
            metric_entries[metric_entry.tokens] = metric_entry

        tasks: list[Entry] = []
        for names in load_metric_task_groups().values():
            for raw_task in names:
                task_entry = Entry.from_raw_str(raw_task)
                assert task_entry is not None, f"Invalid task name: {raw_task}"
                tasks.append(task_entry)
        tasks_sorted = sorted(tasks, key=lambda entry: len(entry.tokens), reverse=True)
        return cls(
            tasks=tuple(tasks), metrics=metric_entries, tasks_sorted=tuple(tasks_sorted)
        )

    def match_task(self, tokens: TokenSeq) -> tuple[Entry, int, int] | None:
        for entry in self.tasks_sorted:
            length = len(entry.tokens)
            if length == 0 or length > len(tokens):
                continue
            if tokens[:length] == entry.tokens:
                return entry, 0, length
        return None

    def match_metric(self, tokens: TokenSeq) -> Entry | None:
        if not tokens:
            return None
        return self.metrics.get(tokens)


@lru_cache(maxsize=1)
def _catalog() -> MetricCatalog:
    return MetricCatalog.from_config()


def parse_metric_label(value: Any) -> TaskMetricUnmatched:
    catalog = _catalog()
    tokens = _tokenize(value)
    print(f"{value=} {tokens=}")
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
    return TaskMetricUnmatched(task_entry.canonical, None, unmatched)


def canonicalize_metric_label(value: Any, strict: bool = False) -> str:
    tmu = parse_metric_label(value)
    if strict and (tmu.unmatched is not None or tmu.metric is None or tmu.task is None):
        raise ValueError(f"Invalid metric label: {value}")
    base = f"{tmu.task} {tmu.metric}"
    if tmu.unmatched:
        base = f"{base} {tmu.unmatched}"
    return base


__all__ = ["canonicalize_metric_label", "parse_metric_label"]
