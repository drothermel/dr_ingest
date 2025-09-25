"""String normalisation helpers for label and key processing."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

DEFAULT_DELIMITERS = ("-", "_", " ", "/")
DELIMITERS = ("-", "_", "/")
SPACE_NORM = re.compile(r"\s+")
SPLIT_ALPHANUMERIC = re.compile(r"[A-Za-z]+|[0-9]+(?:\.[0-9]+)?")


def normalize_str(value: str) -> list[str]:
    result = value.strip().lower()
    for delimiter in DELIMITERS:
        result = result.replace(delimiter, " ")
    result = SPACE_NORM.sub(" ", result)
    return result


def any_key_lookup(keys: Iterable[str], mapping: dict[str, Any]) -> Any | None:
    return next(filter_none(mapping.get(k) for k in keys), None)


def filter_none(iter_obj: Iterable[Any]) -> Iterable[Any]:
    if isinstance(iter_obj, dict):
        return {k: v for k, v in iter_obj.items() if v is not None}
    return filter(lambda x: x is not None, iter_obj)


def normalize_key(value: str, *, delimiters: Iterable[str] = DEFAULT_DELIMITERS) -> str:
    result = value.strip().lower()
    for delimiter in delimiters:
        result = result.replace(delimiter, "_")
    while "__" in result:
        result = result.replace("__", "_")
    return result.strip("_")


def key_variants(
    value: str, *, delimiters: Iterable[str] = DEFAULT_DELIMITERS
) -> set[str]:
    variants: set[str] = set()
    base = normalize_key(value, delimiters=delimiters)
    variants.add(base)
    variants.add(base.replace("_", "-"))
    variants.add(base.replace("_", ""))

    raw_lower = value.strip().lower()
    variants.add(raw_lower)
    variants.add(raw_lower.replace("-", "_"))
    variants.add(raw_lower.replace("_", "-"))
    variants.add(raw_lower.replace(" ", ""))
    variants.add(raw_lower.replace(" ", "_"))
    variants.add(raw_lower.replace("/", "_"))
    variants.add(raw_lower.replace("/", "-"))

    return {variant for variant in variants if variant}


def split_by_known_prefix(
    value: str,
    known_prefixes: Iterable[str],
    *,
    delimiters: Iterable[str] = DEFAULT_DELIMITERS,
) -> tuple[str, str | None]:
    normalized_value = normalize_key(value, delimiters=delimiters)
    prefix_set = {
        normalize_key(prefix, delimiters=delimiters) for prefix in known_prefixes
    }

    if normalized_value in prefix_set:
        return normalized_value, None

    parts = normalized_value.split("_")
    for i in range(len(parts), 0, -1):
        candidate_prefix = "_".join(parts[:i])
        if candidate_prefix in prefix_set:
            remainder = "_".join(parts[i:]) if i < len(parts) else None
            return candidate_prefix, remainder or None

    return normalized_value, None


__all__ = ["key_variants", "normalize_key", "split_by_known_prefix"]
