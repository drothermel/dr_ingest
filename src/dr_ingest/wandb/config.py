from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from confection import Config
from pydantic import BaseModel

if TYPE_CHECKING:  # pragma: no cover
    pass

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "wandb"
CONFIG_FILES: tuple[Path, ...] = (
    CONFIG_DIR / "patterns.cfg",
    CONFIG_DIR / "processing.cfg",
    CONFIG_DIR / "metrics.cfg",
)


class IngestWandbDefaults(BaseModel):
    num_finetune_epochs: int = 1
    initial_checkpoint_steps: str = "main"
    comparison_metric: str = "pile"
    comparison_model_recipe: str = "Dolma1.7"


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
def load_pattern_specs() -> Iterable[tuple[str, str, object]]:
    cfg = load_raw_config()
    patterns = cfg["patterns"]
    if isinstance(patterns, dict):
        return patterns.values()
    return patterns


@lru_cache(maxsize=1)
def load_column_renames() -> dict[str, str]:
    cfg = load_raw_config()
    return cfg["processing"]["column_renames"]


@lru_cache(maxsize=1)
def load_fill_from_config_map() -> dict[str, str]:
    cfg = load_raw_config()
    return cfg["processing"]["fill_from_config"]


@lru_cache(maxsize=1)
def load_value_converter_map() -> dict[str, str]:
    cfg = load_raw_config()
    return cfg["processing"]["value_converter_map"]


@lru_cache(maxsize=1)
def load_metric_names() -> Iterable[str]:
    cfg = load_raw_config()
    names = cfg["metrics"]["metric_names"]["names"]
    return [str(name) for name in names]


@lru_cache(maxsize=1)
def load_metric_task_groups() -> dict[str, Iterable[str]]:
    cfg = load_raw_config()
    tasks_section = cfg["metrics"]["tasks"]
    groups: dict[str, Iterable[str]] = {}
    for category, names in tasks_section.items():
        groups[str(category)] = [str(name) for name in names]
    return groups


@lru_cache(maxsize=1)
def load_summary_field_map() -> dict[str, str]:
    cfg = load_raw_config()
    return cfg["processing"]["summary_field_map"]


__all__ = [
    "CONFIG_DIR",
    "CONFIG_FILES",
    "load_column_renames",
    "load_fill_from_config_map",
    "load_metric_names",
    "load_metric_task_groups",
    "load_pattern_specs",
    "load_raw_config",
    "load_summary_field_map",
    "load_value_converter_map",
]
