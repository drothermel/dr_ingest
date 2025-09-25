from __future__ import annotations

from typing import Any

import pandas as pd
from attrs import define

from dr_ingest.normalization import CONVERSION_MAP
from dr_ingest.wandb.config import (
    load_column_renames,
    load_defaults,
    load_fill_from_config_map,
    load_recipe_columns,
    load_recipe_mapping,
    load_summary_field_map,
    load_value_converter_map,
)
from dr_ingest.wandb.hooks import normalize_matched_run_type

RUN_TYPE_HOOKS: dict[str, Any] = {
    "matched": normalize_matched_run_type,
}


@define
class ProcessingContext:
    column_renames: dict[str, str]
    defaults: dict[str, Any]
    recipe_mapping: dict[str, str]
    recipe_columns: list[str]
    config_field_mapping: dict[str, str]
    summary_field_mapping: dict[str, str]
    value_converter_map: dict[str, str]
    run_type_hooks: dict[str, Any]

    @classmethod
    def from_config(
        cls,
        *,
        overrides: dict[str, Any] | None = None,
        column_renames_override: dict[str, str] | None = None,
        config_field_mapping_override: dict[str, str] | None = None,
        summary_field_mapping_override: dict[str, str] | None = None,
    ) -> ProcessingContext:
        defaults = dict(load_defaults())
        if overrides:
            defaults.update(overrides)

        column_renames = dict(load_column_renames())
        if column_renames_override:
            column_renames.update(column_renames_override)

        config_field_mapping = dict(load_fill_from_config_map())
        if config_field_mapping_override:
            config_field_mapping.update(config_field_mapping_override)

        summary_field_mapping = dict(load_summary_field_map())
        if summary_field_mapping_override:
            summary_field_mapping.update(summary_field_mapping_override)

        value_converter_map = dict(load_value_converter_map())

        return cls(
            column_renames=column_renames,
            defaults=defaults,
            recipe_mapping=dict(load_recipe_mapping()),
            recipe_columns=list(load_recipe_columns()),
            config_field_mapping=config_field_mapping,
            summary_field_mapping=summary_field_mapping,
            value_converter_map=value_converter_map,
            run_type_hooks=RUN_TYPE_HOOKS,
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

    def map_recipes(
        self, frame: pd.DataFrame, columns: list[str] | None = None
    ) -> pd.DataFrame:
        result = frame.copy()
        target_columns = columns or self.recipe_columns
        for column in target_columns:
            if column not in result.columns:
                continue
            result[column] = result[column].map(
                lambda value: self.recipe_mapping.get(value, value)
                if pd.notna(value)
                else value
            )
        return result

    def apply_value_converters(self, frame: pd.DataFrame) -> pd.DataFrame:
        for column, converter in self.value_converter_map.items():
            print(f" {column=} {converter=}")
            if column not in frame.columns:
                continue
            frame[column] = frame[column].apply(CONVERSION_MAP[converter])
        return frame

    def apply_hook(self, run_type: str, frame: pd.DataFrame) -> pd.DataFrame:
        hook = self.run_type_hooks.get(run_type)
        if hook:
            return hook(frame)
        return frame


__all__ = ["ProcessingContext"]
