"""Metric label normalisation utilities."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd

from dr_ingest.normalization import key_variants, normalize_key, split_by_known_prefix


def canonicalize_metric_label(value: Any, default: Optional[str] = None) -> str:
    """Return a canonical metric label given a raw value.

    Parameters
    ----------
    value:
        The raw label to normalise (may be a string, None, or pandas NA).
    default:
        Optional override for the fallback label when the value is missing or
        cannot be resolved. Falls back to the project-wide default when not
        supplied.
    """

    default_label = default or _default_label()
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default_label

    text = str(value).strip()
    if not text:
        return default_label

    for candidate in key_variants(text):
        if candidate in _canonical_lookup():
            return _canonical_lookup()[candidate]
        if candidate in _alias_lookup():
            alias_target = _alias_lookup()[candidate]
            for alias_key in key_variants(alias_target):
                if alias_key in _canonical_lookup():
                    return _canonical_lookup()[alias_key]
            return alias_target

    task, metric = _split_task_metric(text)
    if metric is None:
        if task.lower() in _perplexity_task_set():
            label = f"{task.replace('_', '-')}-{_perplexity_metric()}"
            for key in key_variants(label):
                if key in _canonical_lookup():
                    return _canonical_lookup()[key]
            return label
        return default_label

    candidate = _format_task_metric(task, metric)
    for key in key_variants(candidate):
        if key in _canonical_lookup():
            return _canonical_lookup()[key]

    return default_label


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


__all__ = ["canonicalize_metric_label"]
