"""Transformation helpers for QA evaluation records."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from clumper import Clumper


def build_file_metadata(
    records: Iterable[dict[str, Any]],
    *,
    data: str,
    params: str,
    seed: int,
    task: str,
    step: int,
) -> list[dict[str, Any]]:
    """Deduplicate task/model metadata for a QA JSONL file."""

    return (
        Clumper(list(records))
        .select("task_hash", "model_hash")
        .mutate(
            data=lambda _: data,
            params=lambda _: params,
            seed=lambda _: seed,
            task=lambda _: task,
            step=lambda _: step,
        )
        .drop_duplicates()
        .collect()
    )


def extract_question_payloads(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prepare per-question payloads by stripping metrics and renaming."""

    payloads: list[dict[str, Any]] = []
    for record in records:
        item = {
            k: v
            for k, v in record.items()
            if k not in {"metrics", "task_hash", "model_hash"}
        }
        if "answer_outputs" in item:
            item["model_output"] = item.pop("answer_outputs")
        payloads.append(item)
    return payloads


def preview_agg_metrics(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return aggregated metric rows for inspection."""

    def _to_row(d: dict[str, Any]) -> dict[str, Any]:
        metrics = d.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        return {**metrics, "doc_id": d.get("doc_id")}

    return Clumper(list(records)).map(_to_row).collect()


def model_output_keys(records: Iterable[dict[str, Any]]) -> list[str]:
    """Return sorted keys present in the first model output payload."""

    if not records:
        return []
    first = next((d for d in records if d.get("model_output")), None)
    if not first or not first.get("model_output"):
        return []
    return sorted(first["model_output"][0].keys())


__all__ = [
    "build_file_metadata",
    "extract_question_payloads",
    "model_output_keys",
    "preview_agg_metrics",
]
