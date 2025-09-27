import ast
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import srsly
from dr_wandb import fetch_project_runs
from duckdb.typing import VARCHAR

# -------------------------------------------------
# General Utils for Basic Splitting and Raw Dumping
# -------------------------------------------------


def is_nested(df: pd.DataFrame, col: str) -> bool:
    return df[col].apply(lambda x: isinstance(x, list | dict)).any()


def literal_eval_udf(x: str | None) -> str | None:
    # Return JSON string; you could return dict, but use str for compatibility
    if x is None:
        return None
    try:
        return json.dumps(ast.literal_eval(x))
    except Exception:
        return None


def get_variable_json_from_duckdb(var_name: str) -> Any:
    return json.loads(duckdb.sql(f"SELECT getvariable('{var_name}')").fetchone()[0])


def split_df_to_db_by_object_cols(
    df: pd.DataFrame, name_prefix: str = ""
) -> Iterator[tuple[str, pd.DataFrame]]:
    obj_cols, non_obj_cols = [], []
    for col in df.columns:
        if is_nested(df, col):
            obj_cols.append(col)
        else:
            non_obj_cols.append(col)
    for obj_col in obj_cols:
        obj_df = pd.DataFrame(df[obj_col].tolist())
        missing_id_cols = [col for col in non_obj_cols if col not in obj_df.columns]
        id_df = df[missing_id_cols]
        obj_df = pd.concat([id_df, obj_df], axis=1)
        obj_col_name = f"{name_prefix}{obj_col}"
        yield obj_col_name, obj_df


def load_parse_write_duckdb(
    df_name: str,
    load_fxn: Callable,
    parse_fxn: Callable,
    out_dir: str,
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    con = duckdb.connect(":memory:")
    sub_dfs_gen = parse_fxn(load_fxn(**kwargs), **kwargs)
    sub_dfs = {}
    for sub_name, sub_df in sub_dfs_gen:
        name = f"{df_name}_{sub_name}"
        sub_dfs[name] = sub_df
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM sub_df")  # noqa: S608
        con.execute(
            f"COPY {name} to '{out_dir}/{name}.parquet'"
            " (FORMAT parquet, PARQUET_VERSION v2)"
        )
        print(">> Wrote:", f"{out_dir}/{name}.parquet")
    return sub_dfs


# -------------------------------------------------
# Specific Setting Helpers
# -------------------------------------------------


def wandb_load_fxn(**kwargs: Any) -> tuple[list[dict], list[dict]]:
    entity = kwargs.get("entity")
    project = kwargs.get("project")
    runs_per_page = kwargs.get("runs_per_page", 500)
    log_every = kwargs.get("log_every", 10)
    source_dir = kwargs.get("source_dir", "notebooks")
    redownload = (
        kwargs.get("redownload", False) and entity is not None and project is not None
    )
    if redownload:
        print(">> Redownloading from wandb...")
        return fetch_project_runs(
            entity,
            project,
            runs_per_page=runs_per_page,
            include_history=True,
            progress_callback=lambda i, total, name: print(
                f">> Processing run {i}/{total}: {name}"
            )
            if i % log_every == 0
            else None,
        )
    print(">> Loading locally...")
    runs = list(srsly.read_jsonl(f"{source_dir}/wandb_runs.jsonl"))
    history = list(srsly.read_jsonl(f"{source_dir}/wandb_history.jsonl"))
    return runs, history


def wandb_parse_fxn(
    runs_history: tuple[list[dict], list[dict]], **kwargs: Any
) -> Iterator[tuple[str, pd.DataFrame]]:
    runs, history = runs_history
    runs_df = pd.DataFrame(runs)
    history_df = pd.DataFrame(history)
    print(">> Parsing runs...")
    yield from split_df_to_db_by_object_cols(runs_df, name_prefix="runs_")
    print(">> Parsing history...")
    yield "history", history_df


def dd_results_load_fxn(**kwargs: Any) -> list[dict]:
    source_dir = Path(kwargs.get("source_dir"))
    files = [str(p) for p in source_dir.glob("train-*.parquet")]
    print(files)
    results_table = duckdb.read_parquet(files)
    return results_table


def old_fxn() -> None:
    duckdb.sql("""
    SET VARIABLE json_schema = (
        SELECT json_structure(py_literal_eval(metrics)::JSON)
        FROM df
        LIMIT 1
    );
    """)
    dd_results_df = duckdb.sql("""
    SELECT
        * EXCLUDE(metrics),
        from_json(
            py_literal_eval(metrics)::JSON,
            getvariable('json_schema')
        ) AS metrics
    FROM df
    """).df()
    print(dd_results_df)


def dd_results_parse_fxn(
    df: pd.DataFrame, **kwargs: Any
) -> Iterator[tuple[str, pd.DataFrame]]:
    duckdb.create_function("py_literal_eval", literal_eval_udf, [VARCHAR], VARCHAR)
    dd_results_df = duckdb.sql("""
    WITH parsed_metrics AS (
        SELECT
            * EXCLUDE(metrics),
            py_literal_eval(metrics)::JSON AS parsed_json
        FROM df
    )
    SELECT
        * EXCLUDE(parsed_json),
        from_json(
            parsed_json,
            (SELECT json_structure(parsed_json) FROM parsed_metrics LIMIT 1)
        ) AS metrics
    FROM parsed_metrics
    """).df()
    yield "dd_results", dd_results_df


# -------------------------------------------------
