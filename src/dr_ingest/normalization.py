"""String normalisation helpers for label and key processing."""

from __future__ import annotations

from typing import Iterable, Set

DEFAULT_DELIMITERS = ("-", "_", " ", "/")


def normalize_key(value: str, *, delimiters: Iterable[str] = DEFAULT_DELIMITERS) -> str:
    """Lowercase and replace common delimiters with underscores."""

    result = value.strip().lower()
    for delimiter in delimiters:
        result = result.replace(delimiter, "_")
    while "__" in result:
        result = result.replace("__", "_")
    return result.strip("_")


def key_variants(
    value: str, *, delimiters: Iterable[str] = DEFAULT_DELIMITERS
) -> Set[str]:
    """Generate common equivalent forms for a key or label."""

    variants: Set[str] = set()
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
    """Split a value into (prefix, remainder) using the longest matching prefix.

    Parameters
    ----------
    value:
        The raw string to split.
    known_prefixes:
        Collection of candidate prefixes that should be matched case-insensitively.
    delimiters:
        Delimiters to normalise before attempting the prefix match.
    """

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


__all__ = ["normalize_key", "key_variants", "split_by_known_prefix"]
