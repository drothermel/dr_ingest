"""Regex pattern registry for WandB run classification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import catalogue

from .config import load_pattern_specs

pattern_registry = catalogue.create("dr_ingest", "wandb", "patterns")


@dataclass(frozen=True)
class PatternSpec:
    """Specification for a single regex pattern."""

    name: str
    run_type: str
    regex: re.Pattern[str]


PATTERN_SPECS: list[PatternSpec] = []


def _register_pattern(name: str, run_type: str, pattern: re.Pattern[str]) -> None:
    spec = PatternSpec(name=name, run_type=run_type, regex=pattern)
    PATTERN_SPECS.append(spec)

    @pattern_registry.register(name)
    def factory() -> PatternSpec:  # pragma: no cover - registry hook
        return spec


def _ensure_compiled(regex: object) -> re.Pattern[str]:
    if isinstance(regex, re.Pattern):
        return regex
    if isinstance(regex, str):
        return re.compile(regex)
    raise TypeError(f"Unsupported regex type: {type(regex)!r}")


def _initialise_patterns(pattern_specs: Iterable[tuple[str, str, object]]) -> None:
    for name, run_type, regex in pattern_specs:
        compiled = _ensure_compiled(regex)
        _register_pattern(name, run_type, compiled)


_initialise_patterns(load_pattern_specs())


__all__ = [
    "PatternSpec",
    "PATTERN_SPECS",
    "pattern_registry",
]
