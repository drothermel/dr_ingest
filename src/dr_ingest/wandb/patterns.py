"""Regex pattern registry for WandB run classification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import catalogue

from .config import load_pattern_specs

pattern_registry = catalogue.create("dr_ingest", "wandb", "patterns")


@dataclass(frozen=True)
class PatternSpec:
    """Specification for a single regex pattern."""

    name: str
    run_type: str
    regex: re.Pattern[str]


PATTERN_SPECS: List[PatternSpec] = []


def _register_pattern(name: str, run_type: str, pattern: str) -> None:
    spec = PatternSpec(name=name, run_type=run_type, regex=re.compile(pattern))
    PATTERN_SPECS.append(spec)

    @pattern_registry.register(name)
    def factory() -> PatternSpec:  # pragma: no cover - registry hook
        return spec


for pattern_name, run_type, regex in load_pattern_specs():
    _register_pattern(pattern_name, run_type, regex)


__all__ = [
    "PatternSpec",
    "PATTERN_SPECS",
    "pattern_registry",
]
