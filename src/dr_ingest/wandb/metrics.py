"""Metric label normalisation utilities."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import attrs
import pandas as pd

from dr_ingest.normalization import key_variants, normalize_key, split_by_known_prefix


@lru_cache(maxsize=1)
def _default_resolver() -> "MetricLabelResolver":
    return MetricLabelResolver.from_config()


def canonicalize_metric_label(value: Any, default: Optional[str] = None) -> str:
    """Return a canonical metric label given a raw value."""

    return _default_resolver().resolve(value, default=default)


@lru_cache(maxsize=1)
def _default_label() -> str:
    from .config import load_metric_defaults

    defaults = load_metric_defaults()
    return str(defaults.get("default_label", "pile-valppl"))


@lru_cache(maxsize=1)
def _perplexity_metric() -> str:
    from .config import load_metric_defaults

    defaults = load_metric_defaults()
    return str(defaults.get("perplexity_default_metric", "valppl"))


@lru_cache(maxsize=1)
def _perplexity_label_map() -> Dict[str, str]:
    from .config import load_perplexity_label_map

    mapping = load_perplexity_label_map()
    return {key.lower(): value for key, value in mapping.items()}


@lru_cache(maxsize=1)
def _alias_lookup() -> Dict[str, str]:
    from .config import load_metric_aliases

    aliases = {key.lower(): value for key, value in load_metric_aliases().items()}
    for raw_key, canonical in _perplexity_label_map().items():
        for variant in key_variants(raw_key):
            aliases.setdefault(variant, canonical)
        for variant in key_variants(canonical):
            aliases.setdefault(variant, canonical)
    return aliases


@lru_cache(maxsize=1)
def _perplexity_labels() -> Set[str]:
    labels = set(_perplexity_label_map().values())
    labels.update(_alias_lookup().values())
    return {label for label in labels}


@lru_cache(maxsize=1)
def _metric_name_set() -> Set[str]:
    from .config import load_metric_names

    return {name.lower() for name in load_metric_names()}


@lru_cache(maxsize=1)
def _olmes_task_set() -> Set[str]:
    from .config import load_olmes_tasks

    return {task.lower() for task in load_olmes_tasks()}


@lru_cache(maxsize=1)
def _mmlu_task_set() -> Set[str]:
    from .config import load_mmlu_tasks

    return {task.lower() for task in load_mmlu_tasks()}


@lru_cache(maxsize=1)
def _perplexity_task_set() -> Set[str]:
    from .config import load_perplexity_tasks

    tasks = {task.lower() for task in load_perplexity_tasks()}
    for label in _perplexity_labels():
        task_part = label.split("-")[0]
        tasks.add(task_part.lower())
        tasks.add(task_part.replace("-", "_").lower())
    return tasks


@lru_cache(maxsize=1)
def _known_tasks() -> Set[str]:
    tasks = set(_olmes_task_set())
    tasks.update(_mmlu_task_set())
    tasks.update(_perplexity_task_set())
    return tasks


@lru_cache(maxsize=1)
def _all_known_labels() -> Set[str]:
    labels = set(_perplexity_labels())
    tasks = _known_tasks()
    metrics = _metric_name_set()
    for task in tasks:
        for metric in metrics:
            labels.add(f"{task}_{metric}")
    return labels


@lru_cache(maxsize=1)
def _canonical_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for label in _all_known_labels():
        for variant in key_variants(label):
            lookup.setdefault(variant, label)
    return lookup


def _format_task_metric(task: str, metric: str) -> str:
    task_norm = normalize_key(task)
    metric_norm = normalize_key(metric)
    return f"{task_norm}_{metric_norm}" if metric_norm else task_norm


def _split_task_metric(value: str) -> Tuple[str, str | None]:
    prefix, remainder = split_by_known_prefix(value, _known_tasks())

    if remainder:
        return prefix, normalize_key(remainder)

    lowered = normalize_key(value)
    if lowered in _perplexity_task_set():
        return lowered, None

    return lowered, None


@attrs.define(frozen=True)
class MetricLabelResolver:
    """Resolve raw metric labels to canonical names."""

    default_label: str

    @classmethod
    def from_config(cls) -> "MetricLabelResolver":
        return cls(default_label=_default_label())

    def resolve(self, value: Any, *, default: Optional[str] = None) -> str:
        default_label = default or self.default_label
        text = self._normalize_input(value)
        if text is None:
            return default_label

        variants = list(key_variants(text))
        canonical_lookup = _canonical_lookup()
        alias_lookup = _alias_lookup()
        known_tasks = tuple(_known_tasks())
        perplexity_task_set = _perplexity_task_set()
        perplexity_metric = _perplexity_metric()

        canonical = self._lookup_canonical_variants(variants, canonical_lookup)
        if canonical:
            return canonical

        alias_target = self._resolve_alias(variants, alias_lookup)
        if alias_target:
            canonical_alias = self._lookup_canonical(alias_target, canonical_lookup)
            return canonical_alias or alias_target

        task, metric = self._split_task_metric(text, known_tasks, perplexity_task_set)
        if metric is None:
            task_normalized = normalize_key(task)
            if (
                task_normalized in perplexity_task_set
                or task.lower() in perplexity_task_set
            ):
                label = f"{task_normalized.replace('_', '-')}-{perplexity_metric}"
                canonical_label = self._lookup_canonical(label, canonical_lookup)
                return canonical_label or label
            return default_label

        candidate = _format_task_metric(task, metric)
        canonical_candidate = self._lookup_canonical(candidate, canonical_lookup)
        return canonical_candidate or default_label

    @staticmethod
    def _normalize_input(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _lookup_canonical_variants(
        variants: Iterable[str], lookup: Dict[str, str]
    ) -> Optional[str]:
        for variant in variants:
            if variant in lookup:
                return lookup[variant]
            lowered = variant.lower()
            if lowered in lookup:
                return lookup[lowered]
        return None

    @staticmethod
    def _resolve_alias(
        variants: Iterable[str], alias_lookup: Dict[str, str]
    ) -> Optional[str]:
        for variant in variants:
            alias_target = alias_lookup.get(variant)
            if alias_target:
                return alias_target
            alias_target = alias_lookup.get(variant.lower())
            if alias_target:
                return alias_target
        return None

    @staticmethod
    def _lookup_canonical(label: str, lookup: Dict[str, str]) -> Optional[str]:
        for variant in key_variants(label):
            if variant in lookup:
                return lookup[variant]
            result = lookup.get(variant.lower())
            if result:
                return result
        return None

    @staticmethod
    def _split_task_metric(
        value: str,
        known_tasks: Iterable[str],
        perplexity_task_set: Set[str],
    ) -> Tuple[str, str | None]:
        prefix, remainder = split_by_known_prefix(value, known_tasks)

        if remainder:
            return prefix, normalize_key(remainder)

        lowered = normalize_key(value)
        if lowered in perplexity_task_set:
            return lowered, None

        return lowered, None


__all__ = ["canonicalize_metric_label", "MetricLabelResolver"]
