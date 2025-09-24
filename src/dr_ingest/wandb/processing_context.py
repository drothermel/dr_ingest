"""Configuration-backed context for WandB run post processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from .config import (
    load_column_converters,
    load_column_renames,
    load_defaults,
    load_fill_from_config_map,
    load_recipe_mapping,
    load_run_type_hooks,
)


@dataclass
class ProcessingContext:
    column_renames: Dict[str, str]
    defaults: Dict[str, Any]
    recipe_mapping: Dict[str, str]
    config_field_mapping: Dict[str, str]
    column_converters: Dict[str, Any]
    run_type_hooks: Dict[str, Any]

    @classmethod
    def from_config(
        cls,
        *,
        overrides: Dict[str, Any] | None = None,
        column_renames_override: Dict[str, str] | None = None,
        config_field_mapping_override: Dict[str, str] | None = None,
    ) -> "ProcessingContext":
        defaults = dict(load_defaults())
        if overrides:
            defaults.update(overrides)

        column_renames = dict(load_column_renames())
        if column_renames_override:
            column_renames.update(column_renames_override)

        config_field_mapping = dict(load_fill_from_config_map())
        if config_field_mapping_override:
            config_field_mapping.update(config_field_mapping_override)

        return cls(
            column_renames=column_renames,
            defaults=defaults,
            recipe_mapping=dict(load_recipe_mapping()),
            config_field_mapping=config_field_mapping,
            column_converters=dict(load_column_converters()),
            run_type_hooks=dict(load_run_type_hooks()),
        )

    def apply_defaults(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column, default_value in self.defaults.items():
            if column in result.columns:
                result[column] = result[column].fillna(default_value)
        return result

    def rename_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        existing = {
            old: new for old, new in self.column_renames.items() if old in frame.columns
        }
        return frame.rename(columns=existing) if existing else frame.copy()

    def map_recipes(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        result = frame.copy()
        for column in columns:
            if column not in result.columns:
                continue
            result[column] = result[column].map(
                lambda value: self.recipe_mapping.get(value, value)
                if pd.notna(value)
                else value
            )
        return result

    def apply_converters(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column, converter in self.column_converters.items():
            if column in result.columns:
                result[column] = result[column].apply(converter)
        return result

    def apply_hook(self, run_type: str, frame: pd.DataFrame) -> pd.DataFrame:
        hook = self.run_type_hooks.get(run_type)
        if hook:
            return hook(frame)
        return frame


__all__ = ["ProcessingContext"]
