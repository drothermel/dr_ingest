import ast
import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any

import duckdb
import pandas as pd
import polars as pl
import srsly
from dr_wandb import fetch_project_runs
from huggingface_hub import HfFileSystem

from dr_ingest.normalization import (
    normalize_compute,
    normalize_ds_str,
    normalize_tokens,
)
from dr_ingest.parallel import list_merge, parallel_process, set_merge
from dr_ingest.parse import parse_sl_setup_to_config

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


def pl_load_parquet_from_hf(fs: HfFileSystem, path: str) -> pl.DataFrame:
    with fs.open(path, "rb") as f:
        return pl.read_parquet(f)


# -------------------------------------------------
# Utils for Converting Literals to Structs
# -------------------------------------------------


def str_to_literal(x: str) -> Any:
    return ast.literal_eval(x)


def str_list_to_dicts(x: list[str]) -> list[dict[str, Any]]:
    return [ast.literal_eval(item) for item in x]


def dict_list_to_all_keys(x: list[dict[str, Any]]) -> set[str]:
    all_keys = set()
    for d in x:
        all_keys.update(d.keys())
    return all_keys


def literal_str_to_json_str(x: str) -> str:
    return json.dumps(ast.literal_eval(x))


def make_struct_dtype(keys: list[str], ftypes: list[pl.DataType]) -> pl.DataType:
    return pl.Struct(
        [pl.Field(key, ftype) for key, ftype in zip(keys, ftypes, strict=False)]
    )


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


def dd_results_load_fxn(**kwargs: Any) -> dict[str, pl.DataFrame]:
    fs = get_hf_fs()
    outputs = {}
    train_dfs = []
    for i in range(DD_NUM_TRAIN_FILES):
        start = time.time()
        fp = DD_TRAIN_FILE_PATH_FORMAT_STR.format(i)
        hf_path = get_hf_download_path(DD_RESULTS_REPO, fp)
        train_dfs.append(pl_load_parquet_from_hf(fs, hf_path))
        print(f">> Downloaded {fp} in {time.time() - start:.2f} seconds")
    outputs["train"] = pl.concat(train_dfs, how="vertical")

    for fp in DD_RES_NAMES:
        hf_path = get_hf_download_path(
            DD_RESULTS_REPO, DD_RES_OTHER_PATH_FORMAT_STR.format(fp)
        )
        name = fp.split("-")[0]
        outputs[name] = pl_load_parquet_from_hf(fs, hf_path)
    return outputs


def parse_dd_results_train(df: pl.DataFrame) -> pl.DataFrame:
    train_metrics = df["metrics"].to_list()
    tm_dicts = parallel_process(train_metrics, str_list_to_dicts, list_merge)
    tm_keys = parallel_process(tm_dicts, dict_list_to_all_keys, set_merge)
    tm_dtype = make_struct_dtype(tm_keys, [pl.Float64] * len(tm_keys))
    return df.drop("metrics").with_columns(
        pl.Series("metrics", tm_dicts, dtype=tm_dtype)
    )


def dd_results_parse_fxn(
    outputs: dict[str, pl.DataFrame], **kwargs: Any
) -> Iterator[tuple[str, pl.DataFrame]]:
    for name, df in outputs.items():
        if name == "train":
            yield "train", outputs["train"].pipe(parse_dd_results_train)
            continue
        yield name, df


# -------------------------------------------------


def parse_train_df(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.pipe(parse_dd_results_train)
        .with_columns(
            pl.col("data").map_elements(normalize_ds_str).alias("recipe"),
            pl.col("tokens").map_elements(normalize_tokens).alias("tokens_millions"),
            pl.col("compute").map_elements(normalize_compute).alias("compute_e15"),
            pl.struct(
                accuracy=pl.struct(
                    raw=pl.col("metrics").struct.field("acc_raw"),
                    per_token=pl.col("metrics").struct.field("acc_per_token"),
                    per_char=pl.col("metrics").struct.field("acc_per_char"),
                    per_byte=pl.col("metrics").struct.field("acc_per_byte"),
                    uncond=pl.col("metrics").struct.field("acc_uncond"),
                ),
                sum_logits_corr=pl.struct(
                    raw=pl.col("metrics").struct.field("sum_logits_corr"),
                    per_token=pl.col("metrics").struct.field("logits_per_token_corr"),
                    per_char=pl.col("metrics").struct.field("logits_per_char_corr"),
                ),
                correct_prob=pl.struct(
                    raw=pl.col("metrics").struct.field("correct_prob"),
                    per_token=pl.col("metrics").struct.field("correct_prob_per_token"),
                    per_char=pl.col("metrics").struct.field("correct_prob_per_char"),
                ),
                margin=pl.struct(
                    raw=pl.col("metrics").struct.field("margin"),
                    per_token=pl.col("metrics").struct.field("margin_per_token"),
                    per_char=pl.col("metrics").struct.field("margin_per_char"),
                ),
                total_prob=pl.struct(
                    raw=pl.col("metrics").struct.field("total_prob"),
                    per_token=pl.col("metrics").struct.field("total_prob_per_token"),
                    per_char=pl.col("metrics").struct.field("total_prob_per_char"),
                ),
                uncond_correct_prob=pl.struct(
                    raw=pl.col("metrics").struct.field("uncond_correct_prob"),
                    per_token=pl.col("metrics").struct.field(
                        "uncond_correct_prob_per_token"
                    ),
                    per_char=pl.col("metrics").struct.field(
                        "uncond_correct_prob_per_char"
                    ),
                ),
                norm_correct_prob=pl.struct(
                    raw=pl.col("metrics").struct.field("norm_correct_prob"),
                    per_token=pl.col("metrics").struct.field(
                        "norm_correct_prob_per_token"
                    ),
                    per_char=pl.col("metrics").struct.field(
                        "norm_correct_prob_per_char"
                    ),
                ),
                bits_per_byte_correct=pl.col("metrics").struct.field(
                    "bits_per_byte_corr"
                ),
                primary_metric=pl.col("metrics").struct.field("primary_metric"),
            ).alias("metrics_struct"),
        )
        .drop("data", "chinchilla", "tokens", "compute", "metrics")
        .rename({"metrics_struct": "metrics"})
        .with_row_count("id")
    )


def parse_sl_results(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    sl_w_cfg = prep_sl_cfg(df)
    sl_one_step_rows = [row for row in sl_w_cfg if row["fit_config"]["one_step"]]
    sl_two_step_rows = [row for row in sl_w_cfg if not row["fit_config"]["one_step"]]
    sl_one_step_df = extract_one_step_preds(sl_one_step_rows)
    sl_two_step_df = extract_two_step_preds(sl_two_step_rows)
    sl_true_df = extract_true_metrics(sl_two_step_rows)
    return {
        "scaling_law_pred_one_step_raw": pl.DataFrame(sl_one_step_df),
        "scaling_law_pred_two_step_raw": pl.DataFrame(sl_two_step_df),
        "scaling_law_true_raw": pl.DataFrame(sl_true_df),
    }


def prep_sl_cfg(df: pl.DataFrame) -> list[dict[str, Any]]:
    col_list = df.to_dicts()
    for d in col_list:
        d["fit_config"] = parse_sl_setup_to_config(d["setup"])
        d["recipe"] = normalize_ds_str(d["mix"])
        del d["mix"], d["setup"]
    return col_list


def extract_metric_struct(mets_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": {
            "raw": mets_dict.get("acc_raw"),
            "per_char": mets_dict.get("acc_per_char"),
            "per_token": mets_dict.get("acc_per_token"),
        },
        "margin": {
            "raw": mets_dict.get("margin"),
            "per_char": mets_dict.get("margin_per_char"),
            "per_token": mets_dict.get("margin_per_token"),
        },
        "norm_correct_prob": {
            "raw": mets_dict.get("norm_correct_prob"),
            "per_char": mets_dict.get("norm_correct_prob_per_char"),
            "per_token": mets_dict.get("norm_correct_prob_per_token"),
        },
        "total_prob": {
            "raw": mets_dict.get("total_prob"),
            "per_char": mets_dict.get("total_prob_per_char"),
            "per_token": mets_dict.get("total_prob_per_token"),
        },
        "correct_prob": {
            "raw": mets_dict.get("correct_prob"),
            "per_char": mets_dict.get("correct_prob_per_char"),
            "per_token": mets_dict.get("correct_prob_per_token"),
        },
    }


def extract_true_metrics(col_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    true_loss = defaultdict(dict)
    true_metrics = defaultdict(dict)
    configs = {}
    for d in col_list:
        eid = (d["task"], d["recipe"], d["fit_config"]["name"])
        true_loss[eid][d["metric"]] = d["step_1_y"]
        true_metrics[eid][d["metric"]] = d["step_2_y"]
        configs[eid] = d["fit_config"]

    output = []
    for i, (eid, cfg) in enumerate(configs.items()):
        if cfg["name"] != "3_param-default":
            continue
        output.append(
            {
                "id": i,
                "task": eid[0],
                "recipe": eid[1],
                "task_losses": extract_metric_struct(true_loss[eid]),
                "task_metrics": extract_metric_struct(true_metrics[eid]),
            }
        )
    return output


def extract_two_step_preds(col_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    loss_preds = defaultdict(dict)
    loss_to_metric_preds = defaultdict(dict)
    metric_preds = defaultdict(dict)
    configs = {}
    for d in col_list:
        eid = (d["task"], d["recipe"], d["fit_config"]["name"])
        loss_preds[eid][d["metric"]] = d["step_1_pred"]
        loss_to_metric_preds[eid][d["metric"]] = d["step_2_pred"]
        metric_preds[eid][d["metric"]] = d["stacked_pred"]
        configs[eid] = d["fit_config"]

    output = []
    for i, (eid, cfg) in enumerate(configs.items()):
        output.append(
            {
                "id": i,
                "task": eid[0],
                "recipe": eid[1],
                "fit_config": cfg,
                "pred_task_losses": extract_metric_struct(loss_preds[eid]),
                "pred_task_loss_to_metrics": extract_metric_struct(
                    loss_to_metric_preds[eid]
                ),
                "pred_task_metrics": extract_metric_struct(metric_preds[eid]),
            }
        )
    return output


def extract_one_step_preds(col_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = defaultdict(dict)
    configs = {}

    for d in col_list:
        eid = (d["task"], d["recipe"], d["fit_config"]["name"])
        metrics[eid][d["metric"]] = d["stacked_pred"]
        configs[eid] = d["fit_config"]

    output = []
    for i, (eid, mets) in enumerate(metrics.items()):
        output.append(
            {
                "id": i,
                "task": eid[0],
                "recipe": eid[1],
                "fit_config": configs[eid],
                "pred_task_metrics": extract_metric_struct(mets),
            }
        )
    return output


# -------------------------------------------------
