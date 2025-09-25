"""Helpers for normalizing WandB summary blobs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from clumper import Clumper

TASK_METRIC_NUM_PARTS = 3
TASK_ONLY_NUM_PARTS = 2


def select_oe_eval_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in summary.items() if k.startswith("oe_eval_metrics")}


def group_oe_metrics_by_task(metrics: Mapping[str, Any]) -> dict[str, Any]:
    grouped: dict[str, Any] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        parts = key.split("/")
        if len(parts) not in {TASK_ONLY_NUM_PARTS, TASK_METRIC_NUM_PARTS}:
            continue
        if len(parts) == TASK_METRIC_NUM_PARTS:
            _, task, metric = parts
            existing = grouped.get(task)
            if not isinstance(existing, dict):
                existing = {"value": existing} if existing is not None else {}
            existing[metric] = value
            grouped[task] = existing
        else:
            _, task = parts
            current = grouped.get(task)
            if isinstance(current, dict):
                current["value"] = value
            else:
                grouped[task] = value
    return grouped


def normalize_oe_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    metrics_by_task = group_oe_metrics_by_task(select_oe_eval_metrics(summary))
    records = []
    for task, metrics in metrics_by_task.items():
        if isinstance(metrics, Mapping):
            records.append({"task": task, **metrics})
        else:
            records.append({"task": task, "value": metrics})
    cleaned = (
        Clumper(records)
        .drop("task_config")
        .map(lambda row: {k: v for k, v in row.items() if v is not None})
        .keep(lambda row: len(row) > 1)
        .map(
            lambda row: {
                **row,
                **{
                    k: v
                    for k, v in (row.get("extra_metrics", {}) or {}).items()
                    if v is not None
                },
            }
        )
        .drop("extra_metrics")
        .collect()
    )
    normalised: dict[str, Any] = {}
    for row in cleaned:
        task = row["task"]
        payload = {k: v for k, v in row.items() if k != "task"}
        if payload == {"value": None}:
            continue
        if "value" in payload:
            scalar = payload.pop("value")
            if not payload:
                normalised[task] = scalar
                continue
            payload["value"] = scalar
        normalised[task] = payload
    return normalised


__all__ = [
    "group_oe_metrics_by_task",
    "normalize_oe_summary",
    "select_oe_eval_metrics",
]
