from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import attrs
import pandas as pd
from memo import memlist

from .patterns import PATTERN_SPECS


@dataclass(frozen=True)
class RunClassification:
    run_id: str
    run_type: str
    metadata: dict[str, str | None]


CLASSIFICATION_LOG: list[dict[str, Any]] = []
_record_classification = memlist(data=CLASSIFICATION_LOG)


@_record_classification
def _log_event(**kwargs: Any) -> dict[str, Any]:  # pragma: no cover - memo hook
    return kwargs


def classify_run_id(run_id: str) -> tuple[str, dict[str, str | None]]:
    run_type, extracted = classify_run_id_type_and_extract(run_id)
    _log_event(run_id=run_id, run_type=run_type, pattern=extracted.get("pattern_name"))
    return run_type, extracted


def classify_run_id_type_and_extract(run_id: str) -> tuple[str, dict[str, str | None]]:
    pattern_match = _match_registered_pattern(run_id)
    if pattern_match is not None:
        return pattern_match

    legacy_match = _classify_legacy_run(run_id)
    if legacy_match is not None:
        return legacy_match

    return "other", {}


def parse_and_group_run_ids(
    df: pd.DataFrame,
    run_id_col: str = "run_id",
    drop_run_types: Iterable[str] | None = ("old",),
) -> dict[str, list[dict[str, str | None]]]:
    pipeline = RunClassificationPipeline()
    return pipeline.classify_and_group(
        df, run_id_col=run_id_col, drop_run_types=drop_run_types
    )


def convert_groups_to_dataframes(
    grouped_data: dict[str, list[dict[str, str | None]]],
) -> dict[str, pd.DataFrame]:
    pipeline = RunClassificationPipeline()
    return pipeline.grouped_to_dataframes(grouped_data)


def iter_classified_runs(
    df: pd.DataFrame, *, run_id_col: str = "run_id"
) -> Iterator[RunClassification]:
    pipeline = RunClassificationPipeline()
    return pipeline.iter_classifications(df, run_id_col=run_id_col)


def group_classifications_by_type(
    classifications: Iterable[RunClassification],
    *,
    drop_run_types: Iterable[str] | None = None,
) -> dict[str, list[dict[str, str | None]]]:
    pipeline = RunClassificationPipeline()
    return pipeline.group_classifications(
        classifications, drop_run_types=drop_run_types
    )


def _sorted_metadata(
    records: list[dict[str, str | None]],
) -> list[dict[str, str | None]]:
    return sorted(
        records, key=lambda x: (x.get("run_id") or "", x.get("pattern_name") or "")
    )


@attrs.define(frozen=True)
class RunClassificationPipeline:
    """Orchestrate the run classification flow from DataFrame to grouped tables."""

    classifier: Callable[[str], tuple[str, dict[str, str | None]]] = classify_run_id
    run_id_column: str = "run_id"

    def iter_classifications(
        self,
        df: pd.DataFrame,
        *,
        run_id_col: str | None = None,
    ) -> Iterator[RunClassification]:
        column = run_id_col or self.run_id_column
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")

        for run_id in df[column].astype(str):
            run_type, metadata = self.classifier(run_id)
            yield RunClassification(run_id=run_id, run_type=run_type, metadata=metadata)

    def group_classifications(
        self,
        classifications: Iterable[RunClassification],
        *,
        drop_run_types: Iterable[str] | None = None,
    ) -> dict[str, list[dict[str, str | None]]]:
        drop_set = set(drop_run_types or [])
        grouped: dict[str, list[dict[str, str | None]]] = {}

        for classification in classifications:
            if classification.run_type in drop_set:
                continue
            enriched_metadata = dict(classification.metadata)
            enriched_metadata["run_id"] = classification.run_id
            grouped.setdefault(classification.run_type, []).append(enriched_metadata)

        return grouped

    def classify_and_group(
        self,
        df: pd.DataFrame,
        *,
        run_id_col: str | None = None,
        drop_run_types: Iterable[str] | None = ("old",),
    ) -> dict[str, list[dict[str, str | None]]]:
        classifications = list(self.iter_classifications(df, run_id_col=run_id_col))
        grouped = self.group_classifications(
            classifications, drop_run_types=drop_run_types
        )
        return {
            run_type: _sorted_metadata(records) for run_type, records in grouped.items()
        }

    def grouped_to_dataframes(
        self, grouped_data: dict[str, list[dict[str, str | None]]]
    ) -> dict[str, pd.DataFrame]:
        dataframes: dict[str, pd.DataFrame] = {}
        for run_type, records in grouped_data.items():
            if not records:
                continue
            df = pd.DataFrame(records)
            if "pattern_name" in df.columns:
                df = df.sort_values("pattern_name")
            columns = ["run_id"] + [col for col in df.columns if col != "run_id"]
            dataframes[run_type] = df[columns]
        return dataframes

    def classify_group_to_dataframes(
        self,
        df: pd.DataFrame,
        *,
        run_id_col: str | None = None,
        drop_run_types: Iterable[str] | None = ("old",),
    ) -> dict[str, pd.DataFrame]:
        grouped = self.classify_and_group(
            df, run_id_col=run_id_col, drop_run_types=drop_run_types
        )
        return self.grouped_to_dataframes(grouped)


def _match_registered_pattern(run_id: str) -> tuple[str, dict[str, str | None]] | None:
    for spec in PATTERN_SPECS:
        match = spec.regex.match(run_id)
        if match:
            extracted = match.groupdict()
            extracted["pattern_name"] = spec.name
            return spec.run_type, extracted
    return None


def _classify_legacy_run(run_id: str) -> tuple[str, dict[str, str | None]] | None:
    if (
        "main_default" in run_id
        and "dpo" not in run_id
        and "--reduce_loss=" not in run_id
    ):
        return "old", {}
    return None


__all__ = [
    "CLASSIFICATION_LOG",
    "RunClassificationPipeline",
    "classify_run_id",
    "classify_run_id_type_and_extract",
    "convert_groups_to_dataframes",
]
