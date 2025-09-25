from __future__ import annotations

import re

from dr_ingest.wandb import constants as const

LITERAL_PREFIX = "literal:"
EXTRA_COMPONENTS = {
    "ANCHOR_START": "^",
    "ANCHOR_END": "$",
}


def _constant_map() -> dict[str, str]:
    return {name: getattr(const, name) for name in dir(const) if name.isupper()}


def _fragment_map() -> dict[str, str]:
    mapping = _constant_map()
    mapping.update(EXTRA_COMPONENTS)
    return mapping


def _resolve_fragment(name: str, fragments: dict[str, str]) -> str:
    if name.startswith(LITERAL_PREFIX):
        return name[len(LITERAL_PREFIX) :]
    if name in fragments:
        return fragments[name]
    raise KeyError(f"Unknown pattern fragment '{name}'")


def build_composite_pattern(
    *, run_type: str, components: list[str], name: str | None = None
) -> tuple[str, str, re.Pattern[str]]:
    fragments = _fragment_map()
    regex_parts = [_resolve_fragment(component, fragments) for component in components]
    regex = "".join(regex_parts)
    compiled = re.compile(regex)
    pattern_name = name or f"pattern_{abs(hash((run_type, tuple(components))))}"
    return pattern_name, run_type, compiled
