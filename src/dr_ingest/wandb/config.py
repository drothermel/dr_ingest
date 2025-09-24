"""Configuration utilities for WandB ingestion."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

from confection import Config

from . import constants as const

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

CONFIG_PATH = Path(__file__).resolve().parent / "config.cfg"


@lru_cache(maxsize=1)
def load_config() -> Config:
    """Load the Confection configuration from disk."""
    return Config().from_disk(CONFIG_PATH, interpolate=False)


@lru_cache(maxsize=1)
def load_defaults() -> Dict[str, str]:
    """Return default values used during post-processing."""
    cfg = load_config()
    return dict(cfg["defaults"])  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def load_recipe_mapping() -> Dict[str, str]:
    """Return recipe name mapping used to normalise run metadata."""
    cfg = load_config()
    return dict(cfg["recipe_mapping"])  # type: ignore[arg-type]


def _constant_values() -> Dict[str, str]:
    return {name: getattr(const, name) for name in dir(const) if name.isupper()}


def _format_regex(template: str) -> str:
    return template.format(**_constant_values())


@lru_cache(maxsize=1)
def load_pattern_specs() -> List[Tuple[str, str, str]]:
    cfg = load_config()
    patterns_cfg = dict(cfg.get("patterns", {}))  # type: ignore[arg-type]
    specs: List[Tuple[str, str, str]] = []
    for key, section in patterns_cfg.items():
        run_type = section["run_type"]
        regex_template = section["regex"]
        regex = _format_regex(regex_template)
        pattern_name = f"{key.upper()}_PATTERN"
        specs.append((pattern_name, run_type, regex))
    return specs


@lru_cache(maxsize=1)
def load_column_renames() -> Dict[str, str]:
    cfg = load_config()
    processing_cfg = cfg.get("processing", {})  # type: ignore[arg-type]
    column_cfg = processing_cfg.get("column_renames", {})
    return dict(column_cfg)


@lru_cache(maxsize=1)
def load_fill_from_config_map() -> Dict[str, str]:
    cfg = load_config()
    processing_cfg = cfg.get("processing", {})  # type: ignore[arg-type]
    fill_cfg = processing_cfg.get("fill_from_config", {})
    return dict(fill_cfg)


@lru_cache(maxsize=1)
def load_run_type_hooks() -> Dict[str, Callable[["pd.DataFrame"], "pd.DataFrame"]]:
    cfg = load_config()
    processing_cfg = cfg.get("processing", {})  # type: ignore[arg-type]
    hooks_cfg = processing_cfg.get("run_type_hooks", {})
    hooks: Dict[str, Callable[["pd.DataFrame"], "pd.DataFrame"]] = {}
    for run_type, path in hooks_cfg.items():
        module_name, func_name = path.split(":")
        module = import_module(module_name)
        hook = getattr(module, func_name)
        hooks[run_type] = hook
    return hooks


__all__ = [
    "CONFIG_PATH",
    "load_config",
    "load_defaults",
    "load_recipe_mapping",
    "load_pattern_specs",
    "load_column_renames",
    "load_fill_from_config_map",
    "load_run_type_hooks",
]
