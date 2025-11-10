import ast
import json
import time
from collections.abc import Callable, Iterator
from typing import Any

import duckdb
import pandas as pd
import srsly
from dr_wandb import fetch_project_runs
from huggingface_hub import HfFileSystem

from dr_ingest.pipelines.dd_results import (
    parse_dd_results_train,
)

DD_RESULTS_REPO = "allenai/DataDecide-eval-results"
DD_NUM_TRAIN_FILES = 4
DD_TRAIN_FILE_PATH_FORMAT_STR = "data/train-0000{}-of-00004.parquet"
DD_RES_NAMES = [
    "macro_avg-00000-of-00001.parquet",
    "scaling_law_fit-00000-of-00001.parquet",
]
DD_RES_OTHER_PATH_FORMAT_STR = "data/{}"

type ColName = str

# -------------------------------------------------
# Huggingface Download Utils
# -------------------------------------------------


def get_hf_fs() -> HfFileSystem:
    return HfFileSystem()


def get_hf_download_path(repo: str, filepath: str) -> str:
    return f"hf://datasets/{repo}/{filepath}"


def load_parquet_from_hf(fs: HfFileSystem, path: str) -> pd.DataFrame:
    with fs.open(path, "rb") as f:
        return pd.read_parquet(f)


# -------------------------------------------------
# Utils for Converting Literals to Structs
# -------------------------------------------------


def str_to_literal(x: str) -> Any:
    return ast.literal_eval(x)


def literal_str_to_json_str(x: str) -> str:
    return json.dumps(ast.literal_eval(x))


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


def dd_results_load_fxn(**kwargs: Any) -> dict[str, pd.DataFrame]:
    fs = get_hf_fs()
    outputs = {}
    train_dfs = []
    for i in range(DD_NUM_TRAIN_FILES):
        start = time.time()
        fp = DD_TRAIN_FILE_PATH_FORMAT_STR.format(i)
        hf_path = get_hf_download_path(DD_RESULTS_REPO, fp)
        train_dfs.append(load_parquet_from_hf(fs, hf_path))
        print(f">> Downloaded {fp} in {time.time() - start:.2f} seconds")
    outputs["train"] = pd.concat(train_dfs, ignore_index=True)

    for fp in DD_RES_NAMES:
        hf_path = get_hf_download_path(
            DD_RESULTS_REPO, DD_RES_OTHER_PATH_FORMAT_STR.format(fp)
        )
        name = fp.split("-")[0]
        outputs[name] = load_parquet_from_hf(fs, hf_path)
    return outputs


def dd_results_parse_fxn(
    outputs: dict[str, pd.DataFrame], **kwargs: Any
) -> Iterator[tuple[str, pd.DataFrame]]:
    for name, df in outputs.items():
        if name == "train":
            yield "train", outputs["train"].pipe(parse_dd_results_train)
            continue
        yield name, df


# -------------------------------------------------
# -------------------------------------------------
