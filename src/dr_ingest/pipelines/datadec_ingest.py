"""Streaming ingest utilities for DataDec eval directories."""

from __future__ import annotations

import json
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any

import duckdb
from duckdb.typing import VARCHAR

import dr_ingest.utils as du
from dr_ingest.metrics_all.constants import LoadMetricsAllConfig
from dr_ingest.types import TaskArtifactType


DOC_ARTIFACT_TYPES = (
    TaskArtifactType.PREDICTIONS,
    TaskArtifactType.RECORDED_INPUTS,
    TaskArtifactType.REQUESTS,
)

JSON_SERIALIZED_COLUMNS = {
    "metrics",
    "model_output",
    "doc",
    "request",
    "requests",
    "model_config",
    "task_config",
    "compute_config",
    "beaker_info",
}


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
                for t in DOC_ARTIFACT_TYPES
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
    column_types: Iterable[Any],
    table_alias: str,
    prefix: str,
    skip_cols: Iterable[str] = (),
) -> list[str]:
    skip = set(skip_cols)
    prefix = prefix.rstrip("_") + "_"
    aliases: list[str] = []
    for col, col_type in zip(columns, column_types):
        if col in skip:
            continue
        escaped = col.replace('"', '""')
        safe_col = sanitize_column_name(col)
        expr = f'{table_alias}."{escaped}"'
        if isinstance(col_type, str):
            dtype = col_type.upper()
        else:
            dtype = str(col_type).upper()
        if col in JSON_SERIALIZED_COLUMNS:
            expr = f"to_json({expr})"
            dtype = "JSON"
        if dtype == "JSON":
            expr = f"CAST({expr} AS VARCHAR)"
        aliases.append(f"{expr} AS {prefix}{safe_col}")
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
    for artifact in DOC_ARTIFACT_TYPES:
        paths = all_paths.get(artifact.value, [])
        if isinstance(paths, Path):
            paths = [paths]
        rels[artifact] = load_jsonl_artifact(conn, paths, artifact.value)
    return rels


def _build_select_clause(
    preds_rel,
    recs_rel,
    reqs_rel,
    met_rel,
) -> str:
    prefix_map = {
        TaskArtifactType.PREDICTIONS: "prd",
        TaskArtifactType.RECORDED_INPUTS: "rin",
        TaskArtifactType.REQUESTS: "req",
        TaskArtifactType.METRICS: "met",
    }
    skip_map = {
        TaskArtifactType.PREDICTIONS: {"file_prefix", "doc_id", "idx"},
        TaskArtifactType.RECORDED_INPUTS: {"file_prefix", "doc_id", "idx"},
        TaskArtifactType.REQUESTS: {"file_prefix", "doc_id", "idx"},
        TaskArtifactType.METRICS: {"file_prefix", "doc_id"},
    }

    select_parts: list[str] = ["dk.file_prefix", "dk.doc_id", "dk.idx"]
    select_parts += build_prefixed_column_aliases(
        preds_rel.columns,
        preds_rel.types,
        "preds",
        prefix_map[TaskArtifactType.PREDICTIONS],
        skip_map[TaskArtifactType.PREDICTIONS],
    )
    select_parts += build_prefixed_column_aliases(
        recs_rel.columns,
        recs_rel.types,
        "recs",
        prefix_map[TaskArtifactType.RECORDED_INPUTS],
        skip_map[TaskArtifactType.RECORDED_INPUTS],
    )
    select_parts += build_prefixed_column_aliases(
        reqs_rel.columns,
        reqs_rel.types,
        "reqs",
        prefix_map[TaskArtifactType.REQUESTS],
        skip_map[TaskArtifactType.REQUESTS],
    )
    select_parts += build_prefixed_column_aliases(
        met_rel.columns,
        met_rel.types,
        "met",
        prefix_map[TaskArtifactType.METRICS],
        skip_map[TaskArtifactType.METRICS],
    )
    return ",\n                ".join(select_parts)


def _empty_meta_relation(conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyRelation:
    return conn.sql("SELECT CAST(NULL AS VARCHAR) AS file_prefix WHERE 0=1")


def _load_metrics_all_relation(
    conn: duckdb.DuckDBPyConnection,
    metrics_all_path: Path | None,
) -> duckdb.DuckDBPyRelation | None:
    if not metrics_all_path or not metrics_all_path.exists():
        return None
    return careful_read_json_files(
        conn,
        [metrics_all_path],
        fmt="newline_delimited",
    )


def _build_task_name_map(conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyRelation:
    return conn.sql(
        """
        WITH prefixes AS (
            SELECT file_prefix FROM preds
            UNION
            SELECT file_prefix FROM recs
            UNION
            SELECT file_prefix FROM reqs
        ),
        named AS (
            SELECT
                file_prefix,
                max(task_name) AS task_name
            FROM (
                SELECT file_prefix, task_name FROM recs WHERE task_name IS NOT NULL
                UNION ALL
                SELECT file_prefix, task_name FROM reqs WHERE task_name IS NOT NULL
            )
            GROUP BY file_prefix
        )
        SELECT
            prefixes.file_prefix,
            COALESCE(
                named.task_name,
                regexp_replace(prefixes.file_prefix, '^task-\\d+-', '')
            ) AS task_name
        FROM prefixes
        LEFT JOIN named USING (file_prefix)
        """
    )


def _build_metrics_relation(
    conn: duckdb.DuckDBPyConnection,
    metrics_all_path: Path | None,
    task_name_map: duckdb.DuckDBPyRelation,
) -> tuple[duckdb.DuckDBPyRelation, list[str]]:
    cleanup: list[str] = []
    raw_rel = _load_metrics_all_relation(conn, metrics_all_path)
    if raw_rel is None or "task_name" not in raw_rel.columns:
        return _empty_meta_relation(conn), cleanup
    conn.register("metrics_all_raw", raw_rel)
    cleanup.append("metrics_all_raw")
    conn.register("task_name_map", task_name_map)
    cleanup.append("task_name_map")
    rel = conn.sql(
        """
        SELECT
            task_name_map.file_prefix,
            metrics_all_raw.*
        FROM metrics_all_raw
        LEFT JOIN task_name_map
          ON task_name_map.task_name = metrics_all_raw.task_name
        WHERE task_name_map.file_prefix IS NOT NULL
        """
    )
    return rel, cleanup


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
    conn.register("preds", preds_rel)
    conn.register("recs", recs_rel)
    if "idx" not in reqs_rel.columns:
        conn.register("reqs_base", reqs_rel)
        reqs_rel = conn.sql(
            "SELECT reqs_base.*, CAST(NULL AS BIGINT) AS idx FROM reqs_base"
        )
    conn.register("reqs", reqs_rel)

    task_name_map_rel = _build_task_name_map(conn)
    metrics_all_path = entry.get("metrics-all")
    met_rel, meta_cleanup = _build_metrics_relation(
        conn, metrics_all_path, task_name_map_rel
    )
    conn.register("met", met_rel)

    select_clause = _build_select_clause(
        preds_rel,
        recs_rel,
        reqs_rel,
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

    cleanup_names = ["preds", "recs", "reqs", "met", "reqs_base", *meta_cleanup]
    for name in cleanup_names:
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

    rel = conn.read_parquet(
        [str(p) for p in parquet_files],
        union_by_name=True,
    )
    columns = rel.columns
    has_config = "met_task_config" in columns
    if has_config:
        def _normalize_config(value):
            if value is None:
                return None
            if isinstance(value, dict):
                filtered = {k: value[k] for k in value if k != "task_name"}
            else:
                filtered = value
            try:
                return json.dumps(filtered, sort_keys=True)
            except TypeError:
                return json.dumps(filtered, sort_keys=True, default=str)

        conn.create_function(
            "normalize_config",
            _normalize_config,
            return_type=VARCHAR,
        )

    drop_set = set(drop_columns or {"file_prefix", "task_name"})
    drop_candidates = {
        col
        for col in columns
        if col in drop_set or col.endswith("_path") or col.endswith("_dir")
    }
    keep_columns = [col for col in columns if col not in drop_candidates]
    if not keep_columns:
        raise ValueError("No columns left after dropping path-related fields")

    required_cols = set(keep_columns)
    required_cols.add("file_prefix")
    if has_config:
        required_cols.add("met_task_config")
    projection_expr = ", ".join(f'"{c}"' for c in required_cols)
    rel = rel.project(projection_expr)
    conn.register("cached_runs", rel)
    columns = rel.columns

    projection = ", ".join(f'"{c}"' for c in keep_columns)
    escaped_out = str(cache_dir / output_name).replace("'", "''")
    if has_config:
        copy_sql = f"""
            COPY (
                WITH base AS (
                    SELECT
                        cached_runs.*,
                        normalize_config(met_task_config) AS met_task_config_key
                    FROM cached_runs
                ),
                ranked AS (
                    SELECT
                        base.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY met_task_config_key
                            ORDER BY file_prefix
                        ) AS cfg_rank
                    FROM base
                )
                SELECT DISTINCT {projection}
                FROM ranked
                WHERE met_task_config_key IS NULL OR cfg_rank = 1
            ) TO '{escaped_out}' (FORMAT PARQUET)
        """
    else:
        copy_sql = (
            f"COPY (SELECT DISTINCT {projection} FROM cached_runs) TO '{escaped_out}' (FORMAT PARQUET)"
        )
    conn.sql(copy_sql)

    with suppress(Exception):
        conn.unregister("cached_runs")

    return cache_dir / output_name


__all__ = [
    "collect_all_eval_paths",
    "export_eval_dir_to_parquet",
    "clean_cached_results",
]
