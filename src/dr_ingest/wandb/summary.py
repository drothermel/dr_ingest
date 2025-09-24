"""Helpers for normalizing WandB summary blobs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from clumper import Clumper


def select_oe_eval_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in summary.items() if k.startswith("oe_eval_metrics")}


def group_oe_metrics_by_task(metrics: Mapping[str, Any]) -> dict[str, Any]:
    grouped: dict[str, Any] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        parts = key.split("/")
        if len(parts) not in {2, 3}:
            continue
        if len(parts) == 3:
            _, task, metric = parts
            grouped.setdefault(task, {})[metric] = value
        else:
            _, task = parts
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
        .map(lambda row: {**row, **row.get("extra_metrics", {})})
        .drop("extra_metrics")
        .collect()
    )
    normalised: dict[str, Any] = {}
    for row in cleaned:
        task = row["task"]
        payload = {k: v for k, v in row.items() if k != "task"}
        if payload == {"value": None}:
            continue
        if "value" in payload and len(payload) == 1:
            normalised[task] = payload["value"]
        else:
            payload.pop("value", None)
            normalised[task] = payload
    return normalised


__all__ = [
    "normalize_oe_summary",
    "select_oe_eval_metrics",
    "group_oe_metrics_by_task",
]
