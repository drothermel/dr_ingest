"""Streaming ingest utilities for DataDec eval directories."""

from __future__ import annotations

import json
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any

import duckdb

import dr_ingest.utils as du
from dr_ingest.metrics_all.constants import LoadMetricsAllConfig
from dr_ingest.types import TaskArtifactType


def collect_all_eval_paths(config: LoadMetricsAllConfig) -> list[dict[str, Any]]:
    all_paths: list[dict[str, Any]] = []
    for metrics_all_path in du.iter_file_glob_from_roots(
        config.root_paths, file_glob=config.results_filename
    ):
        results_dir = metrics_all_path.parent
        entry = {
            "metrics-all": metrics_all_path,
            "results_dir": results_dir,
        }
        entry.update(
            {
                t.value: list(
                    du.iter_file_glob_from_roots(
                        results_dir, file_glob=t.filename_pattern
                    )
                )
                for t in TaskArtifactType
            }
        )
        all_paths.append(entry)
    return all_paths


def sanitize_column_name(name: str) -> str:
    import re

    sanitized = re.sub(r"\W+", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "col"


def build_prefixed_column_aliases(
    columns: Iterable[str],
    table_alias: str,
    prefix: str,
    skip_cols: Iterable[str] = (),
) -> list[str]:
    skip = set(skip_cols)
    prefix = prefix.rstrip("_") + "_"
    aliases: list[str] = []
    for col in columns:
        if col in skip:
            continue
        escaped = col.replace('"', '""')
        safe_col = sanitize_column_name(col)
        aliases.append(f'{table_alias}."{escaped}" AS {prefix}{safe_col}')
    return aliases


def careful_read_json_files(
    conn: duckdb.DuckDBPyConnection,
    paths: list[Path],
    *,
    fmt: str,
) -> duckdb.DuckDBPyRelation:
    return conn.read_json(
        [str(p) for p in paths],
        format=fmt,
        sample_size=-1,
        maximum_sample_files=len(paths),
        union_by_name=True,
        filename=True,
    )


def empty_join_relation(conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyRelation:
    return conn.sql(
        """
        SELECT
            CAST(NULL AS VARCHAR) AS file_prefix,
            CAST(NULL AS VARCHAR) AS doc_id
        WHERE 0=1
        """
    )


def prep_artifact_relation(
    conn: duckdb.DuckDBPyConnection,
    rel: duckdb.DuckDBPyRelation,
    artifact_value: str,
    *,
    keep_doc_id: bool,
) -> duckdb.DuckDBPyRelation:
    suffix_len = len(f"-{artifact_value}")
    safe_name = f"raw_{artifact_value.replace('-', '_')}"
    conn.register(safe_name, rel)
    doc_id_clause = "" if keep_doc_id else "CAST(NULL AS VARCHAR) AS doc_id,"
    return conn.sql(
        f"""
        WITH extracted AS (
            SELECT
                {safe_name}.*,
                regexp_replace(CAST(filename AS VARCHAR), '^.*/', '') AS basename,
                regexp_replace(
                    regexp_replace(CAST(filename AS VARCHAR), '^.*/', ''),
                    '\\.[^.]*$', ''
                ) AS stem
            FROM {safe_name}
        )
        SELECT
            left(stem, length(stem) - {suffix_len}) AS file_prefix,
            {doc_id_clause}
            extracted.* EXCLUDE (filename, basename, stem)
        FROM extracted
        """
    )


def load_jsonl_artifact(
    conn: duckdb.DuckDBPyConnection,
    paths: list[Path],
    artifact_value: str,
) -> duckdb.DuckDBPyRelation:
    if not paths:
        return empty_join_relation(conn)
    base_rel = careful_read_json_files(conn, paths, fmt="newline_delimited")
    return prep_artifact_relation(conn, base_rel, artifact_value, keep_doc_id=True)


def load_json_artifact(
    conn: duckdb.DuckDBPyConnection,
    paths: list[Path],
    artifact_value: str,
) -> duckdb.DuckDBPyRelation:
    if not paths:
        return empty_join_relation(conn)
    base_rel = careful_read_json_files(conn, paths, fmt="auto")
    return prep_artifact_relation(conn, base_rel, artifact_value, keep_doc_id=False)


def _register_relations(
    conn: duckdb.DuckDBPyConnection,
    all_paths: dict[str, Any],
) -> dict[TaskArtifactType, duckdb.DuckDBPyRelation]:
    rels: dict[TaskArtifactType, duckdb.DuckDBPyRelation] = {}
    for artifact in TaskArtifactType:
        paths = all_paths.get(artifact.value, [])
        if isinstance(paths, Path):
            paths = [paths]
        loader = (
            load_json_artifact
            if artifact in (TaskArtifactType.CONFIG, TaskArtifactType.METRICS)
            else load_jsonl_artifact
        )
        rels[artifact] = loader(conn, paths, artifact.value)
    return rels


def _build_select_clause(
    preds_rel,
    recs_rel,
    reqs_rel,
    cfg_rel,
    met_rel,
) -> str:
    prefix_map = {
        TaskArtifactType.PREDICTIONS: "prd",
        TaskArtifactType.RECORDED_INPUTS: "rin",
        TaskArtifactType.REQUESTS: "req",
        TaskArtifactType.CONFIG: "cfg",
        TaskArtifactType.METRICS: "met",
    }
    skip_map = {
        TaskArtifactType.PREDICTIONS: {"file_prefix", "doc_id", "idx"},
        TaskArtifactType.RECORDED_INPUTS: {"file_prefix", "doc_id", "idx"},
        TaskArtifactType.REQUESTS: {"file_prefix", "doc_id", "idx"},
        TaskArtifactType.CONFIG: {"file_prefix", "doc_id"},
        TaskArtifactType.METRICS: {"file_prefix", "doc_id"},
    }

    select_parts: list[str] = ["dk.file_prefix", "dk.doc_id", "dk.idx"]
    select_parts += build_prefixed_column_aliases(
        preds_rel.columns,
        "preds",
        prefix_map[TaskArtifactType.PREDICTIONS],
        skip_map[TaskArtifactType.PREDICTIONS],
    )
    select_parts += build_prefixed_column_aliases(
        recs_rel.columns,
        "recs",
        prefix_map[TaskArtifactType.RECORDED_INPUTS],
        skip_map[TaskArtifactType.RECORDED_INPUTS],
    )
    select_parts += build_prefixed_column_aliases(
        reqs_rel.columns,
        "reqs",
        prefix_map[TaskArtifactType.REQUESTS],
        skip_map[TaskArtifactType.REQUESTS],
    )
    select_parts += build_prefixed_column_aliases(
        cfg_rel.columns,
        "cfg",
        prefix_map[TaskArtifactType.CONFIG],
        skip_map[TaskArtifactType.CONFIG],
    )
    select_parts += build_prefixed_column_aliases(
        met_rel.columns,
        "met",
        prefix_map[TaskArtifactType.METRICS],
        skip_map[TaskArtifactType.METRICS],
    )
    return ",\n                ".join(select_parts)


def export_eval_dir_to_parquet(
    conn: duckdb.DuckDBPyConnection,
    entry: dict[str, Any],
    cache_dir: Path,
    *,
    force: bool = False,
    doc_id_value: int | None = None,
    slug_override: str | None = None,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(entry["results_dir"])
    slug = slug_override or results_dir.name
    parquet_path = cache_dir / f"{slug}.parquet"
    metadata_path = cache_dir / f"{slug}.json"

    if parquet_path.exists() and metadata_path.exists() and not force:
        return parquet_path

    rels = _register_relations(conn, entry)

    preds_rel = rels[TaskArtifactType.PREDICTIONS]
    recs_rel = rels[TaskArtifactType.RECORDED_INPUTS]
    reqs_rel = rels[TaskArtifactType.REQUESTS]
    cfg_rel = rels[TaskArtifactType.CONFIG]
    met_rel = rels[TaskArtifactType.METRICS]

    conn.register("preds", preds_rel)
    conn.register("recs", recs_rel)
    if "idx" not in reqs_rel.columns:
        conn.register("reqs_base", reqs_rel)
        reqs_rel = conn.sql(
            "SELECT reqs_base.*, CAST(NULL AS BIGINT) AS idx FROM reqs_base"
        )
    conn.register("reqs", reqs_rel)
    conn.register("cfg", cfg_rel)
    conn.register("met", met_rel)

    select_clause = _build_select_clause(
        preds_rel,
        recs_rel,
        reqs_rel,
        cfg_rel,
        met_rel,
    )

    doc_filter = (
        ""
        if doc_id_value is None
        else f" WHERE doc_id = {doc_id_value}"
    )
    req_filter = (
        ""
        if doc_id_value is None
        else f" WHERE doc_id = {doc_id_value}"
    )

    select_sql = f"""
        WITH doc_keys AS (
            SELECT DISTINCT file_prefix, doc_id, CAST(NULL AS BIGINT) AS idx FROM preds{doc_filter}
            UNION
            SELECT DISTINCT file_prefix, doc_id, CAST(NULL AS BIGINT) AS idx FROM recs{doc_filter}
            UNION
            SELECT DISTINCT file_prefix, doc_id, idx FROM reqs{req_filter}
        )
        SELECT
            {select_clause}
        FROM doc_keys dk
        LEFT JOIN preds USING (file_prefix, doc_id)
        LEFT JOIN recs USING (file_prefix, doc_id)
        LEFT JOIN reqs
            ON reqs.file_prefix = dk.file_prefix
           AND reqs.doc_id = dk.doc_id
           AND reqs.idx IS NOT DISTINCT FROM dk.idx
        LEFT JOIN cfg USING (file_prefix)
        LEFT JOIN met USING (file_prefix)
    """

    escaped_path = str(parquet_path).replace("'", "''")
    copy_sql = f"COPY ({select_sql}) TO '{escaped_path}' (FORMAT PARQUET);"
    conn.sql(copy_sql)

    def _stringify(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _stringify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_stringify(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    metadata = _stringify(entry)
    if doc_id_value is not None:
        metadata = {**metadata, "doc_id_filter": doc_id_value}
    metadata_path.write_text(json.dumps(metadata, indent=2))

    for name in ("preds", "recs", "reqs", "cfg", "met", "reqs_base"):
        with suppress(Exception):
            conn.unregister(name)

    return parquet_path


def clean_cached_results(
    conn: duckdb.DuckDBPyConnection,
    cache_dir: Path,
    *,
    drop_columns: Iterable[str] | None = None,
    output_name: str = "deduped.parquet",
) -> Path:
    cache_dir = Path(cache_dir)
    parquet_files = sorted(cache_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {cache_dir}")

    rel = conn.read_parquet([str(p) for p in parquet_files])
    conn.register("cached_runs", rel)
    columns = rel.columns

    drop_set = set(drop_columns or {"file_prefix", "task_name"})
    drop_candidates = {
        col
        for col in columns
        if col in drop_set or col.endswith("_path") or col.endswith("_dir")
    }
    keep_columns = [col for col in columns if col not in drop_candidates]
    if not keep_columns:
        raise ValueError("No columns left after dropping path-related fields")

    projection = ", ".join(f'"{c}"' for c in keep_columns)
    escaped_out = str(cache_dir / output_name).replace("'", "''")
    conn.sql(
        f"COPY (SELECT DISTINCT {projection} FROM cached_runs) TO '{escaped_out}' (FORMAT PARQUET);"
    )

    with suppress(Exception):
        conn.unregister("cached_runs")

    return cache_dir / output_name


__all__ = [
    "collect_all_eval_paths",
    "export_eval_dir_to_parquet",
    "clean_cached_results",
]
