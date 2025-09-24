"""Configuration utilities for WandB ingestion."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

from confection import Config

CONFIG_PATH = Path(__file__).resolve().parent / "config.cfg"


@lru_cache(maxsize=1)
def load_config() -> Config:
    """Load the Confection configuration from disk."""
    return Config().from_disk(CONFIG_PATH)


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


__all__ = [
    "CONFIG_PATH",
    "load_config",
    "load_defaults",
    "load_recipe_mapping",
]
