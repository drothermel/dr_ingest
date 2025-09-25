"""Metric label normalization built on normalized string tokens."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from collections.abc import Sequence

import pandas as pd

from dr_ingest.normalization import normalize_key, normalize_str

TokenSeq = tuple[str, ...]


def _tokenize(value: str) -> TokenSeq:
    normalized = normalize_str(value)
    if not normalized:
        return ()
    return tuple(part for part in normalized.split(" ") if part)


@dataclass(frozen=True)
class TaskEntry:
    canonical: str
    tokens: TokenSeq
    category: str


@dataclass(frozen=True)
class MetricEntry:
    canonical: str
    tokens: TokenSeq


@dataclass(frozen=True)
class MetricCatalog:
    tasks: Sequence[TaskEntry]
    metrics_by_tokens: dict[TokenSeq, MetricEntry]
    tasks_sorted: Sequence[TaskEntry]
    default_task: TaskEntry
    default_metric: MetricEntry
    perplexity_tasks: set[str]

    @classmethod
    def from_config(cls) -> MetricCatalog:
        from .config import load_metric_names, load_metric_task_groups

        task_groups = load_metric_task_groups()
        metric_names = list(load_metric_names())
        if not metric_names:
            raise ValueError(
                "metrics.metric_names.names must define at least one metric"
            )

        metric_entries: dict[TokenSeq, MetricEntry] = {}
        for raw_metric in metric_names:
            tokens = _tokenize(raw_metric)
            if not tokens:
                continue
            canonical = normalize_key(raw_metric)
            metric_entries.setdefault(tokens, MetricEntry(canonical, tokens))

        default_metric_tokens = _tokenize(metric_names[0])
        default_metric_entry = metric_entries.get(default_metric_tokens)
        if default_metric_entry is None:
            default_metric_entry = MetricEntry(
                canonical=normalize_key(metric_names[0]),
                tokens=default_metric_tokens,
            )

        tasks: list[TaskEntry] = []
        perplexity_tasks: set[str] = set()
        default_task_entry: TaskEntry | None = None

        for category, names in task_groups.items():
            for raw_task in names:
                tokens = _tokenize(raw_task)
                if not tokens:
                    continue
                canonical = normalize_key(raw_task)
                entry = TaskEntry(
                    canonical=canonical, tokens=tokens, category=str(category)
                )
                tasks.append(entry)
                if category == "perplexity":
                    perplexity_tasks.add(canonical)
                    if default_task_entry is None:
                        default_task_entry = entry
                if default_task_entry is None:
                    default_task_entry = entry

        if not tasks:
            raise ValueError("metrics.tasks must define at least one task")
        if default_task_entry is None:
            default_task_entry = tasks[0]

        tasks_sorted = sorted(tasks, key=lambda entry: len(entry.tokens), reverse=True)

        return cls(
            tasks=tuple(tasks),
            metrics_by_tokens=metric_entries,
            tasks_sorted=tuple(tasks_sorted),
            default_task=default_task_entry,
            default_metric=default_metric_entry,
            perplexity_tasks=perplexity_tasks,
        )

    def match_task(self, tokens: TokenSeq) -> tuple[TaskEntry, int, int] | None:
        for start in range(len(tokens)):
            for entry in self.tasks_sorted:
                length = len(entry.tokens)
                if length == 0 or start + length > len(tokens):
                    continue
                if tokens[start : start + length] == entry.tokens:
                    return entry, start, start + length
        return None

    def match_metric(self, tokens: TokenSeq) -> MetricEntry | None:
        if not tokens:
            return None
        return self.metrics_by_tokens.get(tokens)

    def format_label(self, task: TaskEntry, metric: MetricEntry) -> str:
        if (
            task.canonical in self.perplexity_tasks
            and metric.canonical == self.default_metric.canonical
        ):
            return f"{task.canonical}-{metric.canonical}"
        return f"{task.canonical}_{metric.canonical}"

    @property
    def default_label(self) -> str:
        return self.format_label(self.default_task, self.default_metric)


@dataclass(frozen=True)
class MetricLabelResolver:
    catalog: MetricCatalog

    @classmethod
    def from_config(cls) -> MetricLabelResolver:
        return cls(MetricCatalog.from_config())

    def resolve(self, value: Any, default: str | None = None) -> str:
        default_label = default or self.catalog.default_label
        text = self._normalize_input(value)
        if text is None:
            return default_label

        tokens = _tokenize(text)
        if not tokens:
            return default_label

        match = self.catalog.match_task(tokens)
        if match is None:
            return default_label

        task_entry, _, end_index = match
        remaining_tokens = tokens[end_index:]

        metric_entry = self.catalog.match_metric(remaining_tokens)
        if metric_entry is not None:
            return self.catalog.format_label(task_entry, metric_entry)

        if (
            not remaining_tokens
            and task_entry.canonical in self.catalog.perplexity_tasks
        ):
            return self.catalog.format_label(task_entry, self.catalog.default_metric)

        return default_label

    @staticmethod
    def _normalize_input(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        text = str(value).strip()
        return text or None


@lru_cache(maxsize=1)
def _default_resolver() -> MetricLabelResolver:
    return MetricLabelResolver.from_config()


def canonicalize_metric_label(value: Any, default: str | None = None) -> str:
    return _default_resolver().resolve(value, default=default)


def parse_metric_label(value: Any) -> tuple[str | None, str | None, str | None]:
    """Parse a metric label into (task, metric, unmatched) components."""

    raise NotImplementedError


__all__ = ["MetricLabelResolver", "canonicalize_metric_label", "parse_metric_label"]
