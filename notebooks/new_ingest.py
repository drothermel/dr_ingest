import marimo

__generated_with = "0.17.7"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    from pathlib import Path
    from typing import Any
    from collections.abc import Iterable

    import marimo as mo
    import pandas as pd
    import duckdb

    import dr_ingest.utils as du
    from dr_ingest.metrics_all.constants import LoadMetricsAllConfig
    from dr_ingest.types import TaskArtifactType

    ex_dir = Path(
        "/Users/daniellerothermel/"
        "drotherm/data/datadec/"
        "2025-11-03_posttrain/_eval_results/"
        "meta-llama_Llama-3.1-8B__main"
    )

    load_cfg = LoadMetricsAllConfig(
        root_paths=[ex_dir],
    )
    return (
        Any,
        Iterable,
        LoadMetricsAllConfig,
        Path,
        TaskArtifactType,
        du,
        duckdb,
        load_cfg,
        mo,
        pd,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Next Steps
    - Pull in the suggested helpers from chatgpt
    - Setup dumping to cache dir
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
    return (file_prefix_from_path,)


@app.function
def prep_artifact_relation(conn, rel, artifact_value, keep_doc_id):
    suffix_len = len(f"-{artifact_value}")
    temp_name = f"raw_{artifact_value}"
    conn.register(temp_name, rel)
    doc_id_expr = "doc_id" if keep_doc_id else "CAST(NULL AS VARCHAR) AS doc_id"
    return conn.sql(f"""
        WITH extracted AS (
            SELECT
                {temp_name}.*,
                regexp_replace(CAST(filename AS VARCHAR), '^.*/', '') AS basename,
                regexp_replace(
                    regexp_replace(CAST(filename AS VARCHAR), '^.*/', ''),
                    '\\.[^.]*$', ''
                ) AS stem
            FROM {temp_name}
        )
        SELECT
            left(stem, length(stem) - {suffix_len}) AS file_prefix,
            {doc_id_expr},
            extracted.* EXCLUDE (filename, basename, stem)
        FROM extracted
    """)


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
        conn.register("raw_jsonl", base_rel)
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
        conn.register("raw_jsonl", base_rel)
        return prep_artifact_relation(conn, base_rel, artifact_value, keep_doc_id=False)
    return


@app.cell(column=1)
def _(load_cfg):
    load_cfg
    return


@app.cell
def _(collect_all_eval_paths, load_cfg):
    all_paths = collect_all_eval_paths(load_cfg)[0]
    list(all_paths.keys())
    return (all_paths,)


@app.cell
def _(Any, Path):
    def convert_dir_paths_to_df_and_md(dir_paths: dict[str, Path | list[Path]]) -> dict[str, Any]:
        pass
    return


@app.cell
def _(Path, TaskArtifactType, all_paths, file_prefix_from_path):
    for type_str, path_list in all_paths.items():
        if isinstance(path_list, str | Path):
            print(type_str, path_list)
            print()
            continue

        if len(path_list) == 0:
            continue
        print(type_str, file_prefix_from_path(path_list[0], TaskArtifactType(type_str)), path_list[0])
        print()
    return


@app.cell
def _(duckdb):
    conn = duckdb.connect()
    return (conn,)


@app.cell
def _(Path, all_paths, conn, load_jsonl_artifact):
    sample_dfs = {}
    for atype, apaths in all_paths.items():
        if not apaths or isinstance(apaths, str | Path):
            continue
        sample_dfs[atype] = load_jsonl_artifact(conn, apaths, atype).df()
    return


@app.cell
def _():
    return


@app.cell(column=2)
def _():
    return


if __name__ == "__main__":
    app.run()
