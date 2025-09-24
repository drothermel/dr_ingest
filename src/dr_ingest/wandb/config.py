"""Configuration utilities for WandB ingestion."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Tuple

from confection import Config, registry

from . import config_registry  # noqa: F401
from . import pattern_builders  # noqa: F401
from . import hooks  # noqa: F401
from .config_registry import wandb_value_converters

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "wandb"
CONFIG_FILES: Tuple[Path, ...] = (
    CONFIG_DIR / "base.cfg",
    CONFIG_DIR / "patterns.cfg",
    CONFIG_DIR / "processing.cfg",
    CONFIG_DIR / "hooks.cfg",
)


def _load_single_config(path: Path) -> Config:
    return Config().from_disk(path, interpolate=False)


@lru_cache(maxsize=1)
def load_raw_config() -> Config:
    configs = [_load_single_config(path) for path in CONFIG_FILES]
    merged = configs[0]
    for extra in configs[1:]:
        merged = Config(merged).merge(extra)
    return merged


@lru_cache(maxsize=1)
def load_resolved_config() -> Dict[str, object]:
    cfg = load_raw_config()
    return registry.resolve(cfg)


@lru_cache(maxsize=1)
def load_defaults() -> Dict[str, str]:
    cfg = load_raw_config()
    return dict(cfg["defaults"])  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def load_recipe_mapping() -> Dict[str, str]:
    cfg = load_raw_config()
    return dict(cfg["recipe_mapping"])  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def load_pattern_specs() -> Iterable[Tuple[str, str, object]]:
    resolved = load_resolved_config()
    patterns = resolved.get("patterns", {})
    if isinstance(patterns, dict):
        return patterns.values()
    return patterns


@lru_cache(maxsize=1)
def load_column_renames() -> Dict[str, str]:
    cfg = load_raw_config()
    processing_cfg = cfg.get("processing", {})  # type: ignore[arg-type]
    return dict(processing_cfg.get("column_renames", {}))


@lru_cache(maxsize=1)
def load_fill_from_config_map() -> Dict[str, str]:
    cfg = load_raw_config()
    processing_cfg = cfg.get("processing", {})  # type: ignore[arg-type]
    return dict(processing_cfg.get("fill_from_config", {}))


@lru_cache(maxsize=1)
def load_run_type_hooks() -> Dict[str, Callable[["pd.DataFrame"], "pd.DataFrame"]]:
    resolved = load_resolved_config()
    rt_hooks = resolved.get("run_type_hooks", {})
    if isinstance(rt_hooks, dict):
        return rt_hooks
    return {}


@lru_cache(maxsize=1)
def load_value_converter_map() -> Dict[str, str]:
    cfg = load_raw_config()
    processing_cfg = cfg.get("processing", {})  # type: ignore[arg-type]
    return dict(processing_cfg.get("value_converter_map", {}))


@lru_cache(maxsize=1)
def load_column_converters() -> Dict[str, Callable[[object], object]]:
    mapping = load_value_converter_map()
    return {
        column: wandb_value_converters.get(name) for column, name in mapping.items()
    }


__all__ = [
    "CONFIG_DIR",
    "CONFIG_FILES",
    "load_raw_config",
    "load_resolved_config",
    "load_defaults",
    "load_recipe_mapping",
    "load_pattern_specs",
    "load_column_renames",
    "load_fill_from_config_map",
    "load_run_type_hooks",
    "load_value_converter_map",
    "load_column_converters",
]
