import marimo

__generated_with = "0.17.7"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    from pathlib import Path
    from typing import Any
    from collections.abc import Iterable

    import json
    import marimo as mo
    import pandas as pd
    import duckdb
    from uuid import UUID

    import dr_ingest.utils as du
    from dr_ingest.metrics_all.constants import LoadMetricsAllConfig
    from dr_ingest.types import TaskArtifactType
    from dr_ingest.configs.paths import Paths

    paths = Paths(
        metrics_all_dir="/Users/daniellerothermel/drotherm/data/datadec/2025-11-03_posttrain/",
    )
    load_cfg = LoadMetricsAllConfig(
        root_paths=[paths.metrics_all_dir],
    )
    notebook_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    error_log_path = notebook_dir / "new_ingest_cache_errors.log"
    cache_root = paths.data_cache_dir / load_cfg.cache_subdir
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_root
    return (
        Any,
        Iterable,
        LoadMetricsAllConfig,
        Path,
        TaskArtifactType,
        UUID,
        cache_root,
        du,
        duckdb,
        error_log_path,
        json,
        load_cfg,
        mo,
        paths,
        pd,
    )


@app.cell
def _(paths):
    paths
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Next Steps
    - test and then run caching
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### To Pandas Utils
    """)
    return


@app.cell
def _(Iterable, pd):
    def add_column_prefix(
        df: pd.DataFrame,
        prefix: str,
        skip: Iterable[str] = ("file_prefix", "doc_id"),
    ) -> pd.DataFrame:
        """
        Prefix all columns except keys like file_prefix/doc_id,
        so merges don't turn into a soup of suffixes.
        """
        df = df.copy()
        rename = {
            col: f"{prefix}{col}"
            for col in df.columns
            if col not in skip
        }
        return df.rename(columns=rename)
    return


@app.function
def sanitize_column_name(name: str) -> str:
    import re

    sanitized = re.sub(r"\W+", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "col"


@app.function
def build_prefixed_column_aliases(columns, table_alias, prefix, skip_cols=()):
    skip = set(skip_cols)
    prefix = prefix.rstrip("_") + "_"
    aliases = []
    for col in columns:
        if col in skip:
            continue
        escaped = col.replace('"', '""')
        safe_col = sanitize_column_name(col)
        aliases.append(f'{table_alias}."{escaped}" AS {prefix}{safe_col}')
    return aliases


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Parsing Helpers
    """)
    return


@app.cell
def _(Any, LoadMetricsAllConfig, TaskArtifactType, du):
    def collect_all_eval_paths(config: LoadMetricsAllConfig) -> list[dict[str, Any]]:
        all_paths = []
        for metrics_all_path in du.iter_file_glob_from_roots(
            config.root_paths, file_glob=config.results_filename
        ):
            results_dir = metrics_all_path.parent
            all_paths.append(
                {
                    "metrics-all": metrics_all_path,
                    "results_dir": results_dir,
                    **{
                        t.value: list(
                            du.iter_file_glob_from_roots(
                                results_dir, file_glob=t.filename_pattern
                            )
                        )
                        for t in TaskArtifactType
                    },
                }
            )
        return all_paths
    return (collect_all_eval_paths,)


@app.cell
def _(Path):
    def file_prefix_from_path(path: Path, artifact_val: str) -> str:
        """
        Given a path like:
          task-038-mmlu_management:0shot_cot::tulu3-metrics.json
        and artifact_value="metrics",
        return:
          task-038-mmlu_management:0shot_cot::tulu3
        """
        stem = path.stem  # e.g. "task-038-...::tulu3-metrics"
        suffix = "-" + artifact_val
        if not stem.endswith(suffix):
            raise ValueError(f"Unexpected filename for {artifact_val}: {path.name}")
        return stem[: -len(suffix)]
    return


@app.function
def prep_artifact_relation(conn, rel, artifact_value, keep_doc_id):
    suffix_len = len(f"-{artifact_value}")
    safe_name = f"raw_{artifact_value.replace('-', '_')}"
    conn.register(safe_name, rel)
    doc_id_clause = "" if keep_doc_id else "CAST(NULL AS VARCHAR) AS doc_id,\n            "
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
            {doc_id_clause}extracted.* EXCLUDE (filename, basename, stem)
        FROM extracted
        """
    )


@app.function
def careful_read_json_files(conn, paths, format):
    return conn.read_json(
        [str(p) for p in paths],
        format=format,
        sample_size=-1, 
        maximum_sample_files=len(paths),
        union_by_name=True,
        filename=True,
    )


@app.function
def empty_join_duckdb_relation(conn):
    return conn.sql("""
        SELECT 
            CAST(NULL AS VARCHAR) AS file_prefix,
            CAST(NULL AS VARCHAR) AS doc_id
        WHERE 0=1
    """)


@app.cell
def _(Path, duckdb):
    def load_jsonl_artifact(
        conn: duckdb.DuckDBPyConnection,
        paths: list[Path],
        artifact_value: str,
    ) -> duckdb.DuckDBPyRelation:
        if not paths:
            return empty_join_duckdb_relation(conn)
        base_rel = careful_read_json_files(conn, paths, format="newline_delimited")
        return prep_artifact_relation(conn, base_rel, artifact_value, keep_doc_id=True)
    return (load_jsonl_artifact,)


@app.cell
def _(Path, duckdb):
    def load_json_artifact(
        conn: duckdb.DuckDBPyConnection,
        paths: list[Path],
        artifact_value: str,
    ):
        if not paths:
            return empty_join_duckdb_relation(conn)
        base_rel = careful_read_json_files(conn, paths, format="auto")
        return prep_artifact_relation(conn, base_rel, artifact_value, keep_doc_id=False)
    return (load_json_artifact,)


@app.cell
def _(
    LoadMetricsAllConfig,
    Path,
    TaskArtifactType,
    UUID,
    duckdb,
    json,
    load_json_artifact,
    load_jsonl_artifact,
    pd,
):
    def build_big_eval_df_for_results_dir_duckdb(
        all_paths: dict[str, Path | list[Path]],
        cache_dir: Path,
        load_cfg: LoadMetricsAllConfig,
        force: bool = False,
        conn: duckdb.DuckDBPyConnection | None = None,
    ) -> pd.DataFrame:
        results_dir = Path(all_paths["results_dir"])
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        slug = results_dir.name
        parquet_path = cache_dir / f"{slug}.parquet"
        metadata_path = cache_dir / f"{slug}.json"

        if parquet_path.exists() and metadata_path.exists() and not force:
            return pd.read_parquet(parquet_path)

        close_conn = conn is None
        conn = conn or duckdb.connect()

        # Load every artifact family into DuckDB relations
        rels = {}
        for artifact in TaskArtifactType:
            paths = all_paths.get(artifact.value, [])
            print(f">> - loading {artifact}: {len(paths)} files")
            if isinstance(paths, Path):
                paths = [paths]
            loader = (
                load_json_artifact
                if artifact in (TaskArtifactType.CONFIG, TaskArtifactType.METRICS)
                else load_jsonl_artifact
            )
            rels[artifact] = loader(conn, paths, artifact.value)

        preds_rel = rels[TaskArtifactType.PREDICTIONS]
        conn.register("preds", preds_rel)

        recs_rel = rels[TaskArtifactType.RECORDED_INPUTS]
        conn.register("recs", recs_rel)

        reqs_rel = rels[TaskArtifactType.REQUESTS]
        if "idx" not in reqs_rel.columns:
            conn.register("reqs_base", reqs_rel)
            reqs_rel = conn.sql("SELECT reqs_base.*, CAST(NULL AS BIGINT) AS idx FROM reqs_base")
        conn.register("reqs", reqs_rel)

        cfg_rel = rels[TaskArtifactType.CONFIG]
        conn.register("cfg", cfg_rel)

        met_rel = rels[TaskArtifactType.METRICS]
        conn.register("met", met_rel)

        select_parts = [*load_cfg.select_parts]
        select_parts += build_prefixed_column_aliases(
            preds_rel.columns, "preds", load_cfg.prefix_map[TaskArtifactType.PREDICTIONS], load_cfg.skip_map[TaskArtifactType.PREDICTIONS]
        )
        select_parts += build_prefixed_column_aliases(
            recs_rel.columns, "recs", load_cfg.prefix_map[TaskArtifactType.RECORDED_INPUTS], load_cfg.skip_map[TaskArtifactType.RECORDED_INPUTS]
        )
        select_parts += build_prefixed_column_aliases(
            reqs_rel.columns, "reqs", load_cfg.prefix_map[TaskArtifactType.REQUESTS], load_cfg.skip_map[TaskArtifactType.REQUESTS]
        )
        select_parts += build_prefixed_column_aliases(
            cfg_rel.columns, "cfg", load_cfg.prefix_map[TaskArtifactType.CONFIG], load_cfg.skip_map[TaskArtifactType.CONFIG]
        )
        select_parts += build_prefixed_column_aliases(
            met_rel.columns, "met", load_cfg.prefix_map[TaskArtifactType.METRICS], load_cfg.skip_map[TaskArtifactType.METRICS]
        )

        select_clause = ",\n                ".join(select_parts)

        result_rel = conn.sql(
            f"""
            WITH doc_keys AS (
                SELECT DISTINCT file_prefix, doc_id, CAST(NULL AS BIGINT) AS idx FROM preds
                UNION
                SELECT DISTINCT file_prefix, doc_id, CAST(NULL AS BIGINT) AS idx FROM recs
                UNION
                SELECT DISTINCT file_prefix, doc_id, idx FROM reqs
            )
            SELECT
                {select_clause}
            FROM doc_keys dk
            LEFT JOIN preds USING (file_prefix, doc_id)
            LEFT JOIN recs  USING (file_prefix, doc_id)
            LEFT JOIN reqs
                ON reqs.file_prefix = dk.file_prefix
               AND reqs.doc_id = dk.doc_id
               AND reqs.idx IS NOT DISTINCT FROM dk.idx
            LEFT JOIN cfg   USING (file_prefix)
            LEFT JOIN met   USING (file_prefix)
            """
        )


        combined_df = result_rel.df()

        def _col_contains_uuid(series) -> bool:
            try:
                return series.map(lambda v: isinstance(v, UUID)).any()
            except Exception:  # noqa: BLE001
                return False

        for col in combined_df.columns:
            series = combined_df[col]
            if _col_contains_uuid(series):
                combined_df[col] = series.map(lambda v: str(v) if isinstance(v, UUID) else v)

        def _force_object_to_string(df):
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].astype("string").fillna(pd.NA)
            return df

        def _stringify_paths(obj):
            if isinstance(obj, dict):
                return {k: _stringify_paths(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_stringify_paths(v) for v in obj]
            if isinstance(obj, Path):
                return str(obj)
            return obj

        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            combined_df.to_parquet(parquet_path, index=False)
        except Exception:
            combined_df = _force_object_to_string(combined_df)
            combined_df.to_parquet(parquet_path, index=False)
        metadata_path.write_text(json.dumps(_stringify_paths(all_paths), indent=2))

        if close_conn:
            conn.close()

        return combined_df
    return (build_big_eval_df_for_results_dir_duckdb,)


@app.cell(column=1)
def _(
    LoadMetricsAllConfig,
    build_big_eval_df_for_results_dir_duckdb,
    cache_root,
    collect_all_eval_paths,
    duckdb,
    error_log_path,
):
    def cache_all_eval_dirs(load_cfg: LoadMetricsAllConfig, force: bool = False):
        from datetime import datetime
        import traceback

        print(":: Start Cache All Eval Dirs ::")
        conn = duckdb.connect()
        cached = []
        try:
            for entry in collect_all_eval_paths(load_cfg):
                results_dir = (entry or {}).get("results_dir")
                print()
                print(f">> Loading artifacts from dir: {results_dir}")
                try:
                    cached.append(
                        build_big_eval_df_for_results_dir_duckdb(
                            entry,
                            cache_dir=cache_root,
                            load_cfg=load_cfg,
                            force=force,
                            conn=conn,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    timestamp = datetime.now().isoformat()
                    log_message = (
                        f"[{timestamp}] Failed for {results_dir}: {exc}\n"
                        f"{traceback.format_exc()}\n"
                    )
                    with error_log_path.open("a", encoding="utf-8") as fh:
                        fh.write(log_message)
                    print(f"!! Error logged for {results_dir}. Continuing...")
        finally:
            conn.close()
        return cached
    return (cache_all_eval_dirs,)


@app.cell
def _(Path, cache_root, pd):
    def clean_cached_results(
        cache_dir: Path = cache_root,
        drop_columns: set[str] | None = None,
        output_name: str = "deduped.parquet",
    ):
        print()
        print(":: Start Clean Cached Results ::")
        cache_dir = Path(cache_dir)
        parquet_files = sorted(cache_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {cache_dir}")

        frames = [pd.read_parquet(p) for p in parquet_files]
        combined = pd.concat(frames, ignore_index=True)

        drop_set = set(drop_columns or {"file_prefix", "task_name"})
        drop_candidates = {
            col
            for col in combined.columns
            if col in drop_set or col.endswith("_path") or col.endswith("_dir")
        }
        if drop_candidates:
            combined = combined.drop(columns=sorted(drop_candidates))

        for col in combined.select_dtypes(include="object").columns:
            combined[col] = combined[col].astype("string")

        deduped = combined.drop_duplicates().reset_index(drop=True)
        output_path = cache_dir / output_name
        deduped.to_parquet(output_path, index=False)
        return deduped, output_path
    return (clean_cached_results,)


@app.cell
def _(duckdb):
    conn = duckdb.connect()
    return (conn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CLI ingestion

    To run the full pipeline without exhausting notebook memory, use the CLI:

    ```bash
    uv run ingest-datadec cache --metrics-dir <metrics_dir> --cache-dir <cache_dir>
    uv run ingest-datadec dedupe --cache-dir <cache_dir>
    ```

    The CLI reuses the same helpers defined above but streams each directory in a
    fresh DuckDB process, writing straight to parquet.
    """)
    return


if __name__ == "__main__":
    app.run()
