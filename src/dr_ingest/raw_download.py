from collections.abc import Callable, Iterator
from typing import Any

import duckdb
import pandas as pd
import srsly
from dr_wandb import fetch_project_runs

# -------------------------------------------------
# General Utils for Basic Splitting and Raw Dumping
# -------------------------------------------------


def is_nested(df: pd.DataFrame, col: str) -> bool:
    return df[col].apply(lambda x: isinstance(x, list | dict)).any()


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
