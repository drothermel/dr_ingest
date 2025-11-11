from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import attrs
import pandas as pd

from dr_ingest.utils.pandas import (
    apply_row_updates,
    ensure_column,
    force_setter,
    maybe_update_setter,
    require_row_index,
)

if TYPE_CHECKING:  # pragma: no cover
    from .processing_context import ProcessingContext


Setter = Callable[[pd.DataFrame, int, str, Any], pd.DataFrame]


@attrs.define(frozen=True)
class HydrationStageConfig:
    source_column: str
    field_map: dict[str, str]
    setter: Setter


@attrs.define(frozen=True)
class HydrationPlan:
    stages: tuple[HydrationStageConfig, ...]

    @classmethod
    def from_context(cls, context: ProcessingContext) -> HydrationPlan:
        stages: list[HydrationStageConfig] = []
        if context.summary_field_mapping:
            stages.append(
                HydrationStageConfig(
                    source_column="summary",
                    field_map=context.summary_field_mapping,
                    setter=force_setter,
                )
            )
        if context.config_field_mapping:
            stages.append(
                HydrationStageConfig(
                    source_column="config",
                    field_map=context.config_field_mapping,
                    setter=maybe_update_setter,
                )
            )
        return cls(tuple(stages))


@attrs.define(frozen=True)
class HydrationExecutor:
    plan: HydrationPlan

    @classmethod
    def from_context(cls, context: ProcessingContext) -> HydrationExecutor:
        return cls(HydrationPlan.from_context(context))

    def apply(
        self,
        frame: pd.DataFrame,
        *,
        ground_truth_source: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if ground_truth_source is None or not self.plan.stages:
            return frame
        assert "run_id" in frame.columns, "expected 'run_id' in extracted runs"
        assert "run_id" in ground_truth_source.columns, (
            "expected 'run_id' in payload source"
        )

        run_ids = frame["run_id"].tolist()
        result = frame
        for stage in self.plan.stages:
            for target_field in stage.field_map:
                result = ensure_column(result, target_field, None, inplace=True)
            updates = _collect_stage_updates(
                ground_truth_source, run_ids, stage.source_column, stage.field_map
            )
            result = apply_row_updates(result, updates, stage.setter)
        return result


def _collect_stage_updates(
    runs_df: pd.DataFrame,
    run_ids: Iterable[str],
    source_column: str,
    field_map: dict[str, str],
) -> dict[str, dict[str, Any]]:
    if not field_map:
        return {}

    updates: dict[str, dict[str, Any]] = {}
    for run_id in run_ids:
        row_idx = require_row_index(runs_df, "run_id", run_id)
        payload = safe_load_json(runs_df.iloc[row_idx].get(source_column)) or {}
        if not isinstance(payload, dict):
            continue
        for target_field, source_field in field_map.items():
            value = payload.get(source_field)
            if value is not None:
                updates.setdefault(run_id, {})[target_field] = value
    return updates


def safe_load_json(payload: Any) -> dict[str, Any] | None:
    """Load JSON from strings or mappings, returning ``None`` on failure."""

    if payload is None or (isinstance(payload, float) and pd.isna(payload)):
        return None
    try:
        if isinstance(payload, str):
            return json.loads(payload)
        if isinstance(payload, dict):
            return payload
        return dict(payload)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


__all__ = [
    "HydrationExecutor",
    "HydrationPlan",
    "HydrationStageConfig",
]
