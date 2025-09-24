"""Run-ID classification utilities built on the pattern registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import pandas as pd
from memo import memlist

from .patterns import PATTERN_SPECS


@dataclass(frozen=True)
class RunClassification:
    run_id: str
    run_type: str
    metadata: Dict[str, str | None]


CLASSIFICATION_LOG: List[Dict[str, Any]] = []
_record_classification = memlist(data=CLASSIFICATION_LOG)


@_record_classification
def _log_event(**kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - memo hook
    return kwargs


def classify_run_id(run_id: str) -> Tuple[str, Dict[str, str | None]]:
    """Return the run type and extracted metadata for a run identifier."""
    run_type, extracted = classify_run_id_type_and_extract(run_id)
    _log_event(run_id=run_id, run_type=run_type, pattern=extracted.get("pattern_name"))
    return run_type, extracted


def classify_run_id_type_and_extract(run_id: str) -> Tuple[str, Dict[str, str | None]]:
    """Match a run ID against the registered patterns."""
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
) -> Dict[str, List[Dict[str, str | None]]]:
    """Group run IDs by run type and attach extracted fields."""

    classifications = list(iter_classified_runs(df, run_id_col=run_id_col))
    grouped = group_classifications_by_type(
        classifications, drop_run_types=drop_run_types
    )
    return {
        run_type: _sorted_metadata(records) for run_type, records in grouped.items()
    }


def convert_groups_to_dataframes(
    grouped_data: Dict[str, List[Dict[str, str | None]]],
) -> Dict[str, pd.DataFrame]:
    """Convert grouped run metadata to DataFrames keyed by run type."""
    dataframes: Dict[str, pd.DataFrame] = {}
    for run_type, records in grouped_data.items():
        if records:
            df = pd.DataFrame(records)
            if "pattern_name" in df.columns:
                df = df.sort_values("pattern_name")
            columns = ["run_id"] + [col for col in df.columns if col != "run_id"]
            dataframes[run_type] = df[columns]
    return dataframes


def iter_classified_runs(
    df: pd.DataFrame, *, run_id_col: str = "run_id"
) -> Iterator[RunClassification]:
    """Yield run classifications from a DataFrame."""

    if run_id_col not in df.columns:
        raise KeyError(f"Column '{run_id_col}' not found in DataFrame")

    for run_id in df[run_id_col].astype(str):
        run_type, metadata = classify_run_id(run_id)
        yield RunClassification(run_id=run_id, run_type=run_type, metadata=metadata)


def group_classifications_by_type(
    classifications: Iterable[RunClassification],
    *,
    drop_run_types: Iterable[str] | None = None,
) -> Dict[str, List[Dict[str, str | None]]]:
    """Group classifications by run type, optionally dropping some types."""

    drop_set = {rtype for rtype in (drop_run_types or [])}
    grouped: Dict[str, List[Dict[str, str | None]]] = {}

    for classification in classifications:
        if classification.run_type in drop_set:
            continue
        enriched_metadata = dict(classification.metadata)
        enriched_metadata["run_id"] = classification.run_id
        grouped.setdefault(classification.run_type, []).append(enriched_metadata)

    return grouped


def _sorted_metadata(
    records: List[Dict[str, str | None]],
) -> List[Dict[str, str | None]]:
    return sorted(
        records, key=lambda x: (x.get("run_id") or "", x.get("pattern_name") or "")
    )


def _match_registered_pattern(run_id: str) -> Tuple[str, Dict[str, str | None]] | None:
    for spec in PATTERN_SPECS:
        match = spec.regex.match(run_id)
        if match:
            extracted = match.groupdict()
            extracted["pattern_name"] = spec.name
            return spec.run_type, extracted
    return None


def _classify_legacy_run(run_id: str) -> Tuple[str, Dict[str, str | None]] | None:
    if (
        "main_default" in run_id
        and "dpo" not in run_id
        and "--reduce_loss=" not in run_id
    ):
        return "old", {}
    return None


__all__ = [
    "CLASSIFICATION_LOG",
    "classify_run_id",
    "classify_run_id_type_and_extract",
    "iter_classified_runs",
    "group_classifications_by_type",
    "parse_and_group_run_ids",
    "convert_groups_to_dataframes",
]
