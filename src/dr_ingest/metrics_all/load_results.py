from __future__ import annotations

from collections.abc import Hashable, Iterable
from pathlib import Path
from typing import Any

from dr_ingest.utils.io import iter_file_glob_from_roots

from .constants import LoadMetricsAllConfig
from .eval_record import EvalRecordSet

__all__ = ["load_all_results"]


def load_all_results(
    *,
    root_paths: Path | str | Iterable[Path | str] | None = None,
    config: LoadMetricsAllConfig | None = None,
) -> list[dict[str, Any]]:
    assert root_paths or config, (
        "Either root_paths or LoadMetricsallConfig must be provided"
    )
    cfg = config or LoadMetricsAllConfig(root_paths=root_paths)  # type: ignore
    records: list[dict[str, Any]] = []
    for metrics_path in iter_file_glob_from_roots(
        cfg.root_paths, file_glob=cfg.results_filename
    ):
        record_set = EvalRecordSet(cfg=cfg, metrics_all_file=metrics_path)
        records.extend(record_set.load())

    return dedup_records(cfg, records)


def dedup_records(
    cfg: LoadMetricsAllConfig, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not records:
        return []

    path_cols = set(cfg.path_cols)
    # group by *all* non-path columns
    all_cols = set().union(*(r.keys() for r in records))
    group_cols = [c for c in all_cols if c not in path_cols]

    # 2. Group records by their non-path values (including dict-valued ones)
    groups = {}
    for rec in records:
        key = tuple((col, _freeze_value(rec.get(col))) for col in group_cols)
        groups.setdefault(key, []).append(rec)

    # 3. For each group, keep one record + aggregate path columns
    deduped = []
    for recs in groups.values():
        # Start from the first record in the group
        merged = recs[0].copy()

        # For each path col:
        #   - keep the first value as the main column
        #   - add all_{col} with the list of all seen values
        for col in path_cols:
            vals = [r.get(col) for r in recs if col in r]
            if not vals:
                continue

            # (1) first value as canonical
            merged[col] = vals[0]

            # (2) all values (deduped but order-preserving)
            merged[f"all_{col}"] = _unique_preserve_order(vals)

        deduped.append(merged)

    # deduped is now:
    #   - one dict per unique "data" (all non-path cols)
    #   - dict-valued columns still dicts, not strings
    #   - path columns collapsed with all_{col} lists added
    return deduped


def _freeze_value(v: Any) -> Hashable:
    """
    Turn (possibly nested) structures into something hashable so we can group.
    Dicts -> sorted tuple of (key, value), lists/tuples -> tuple, sets -> sorted tuple.
    Leaves scalars (str, int, float, Path, etc.) as-is.
    """
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze_value(vv)) for k, vv in v.items()))
    elif isinstance(v, (list, tuple)):
        return tuple(_freeze_value(x) for x in v)
    elif isinstance(v, set):
        return tuple(sorted(_freeze_value(x) for x in v))
    else:
        return v  # assumed hashable


def _unique_preserve_order(seq: Iterable[Any]) -> list[Any]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
