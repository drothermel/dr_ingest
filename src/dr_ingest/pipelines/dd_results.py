from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl

from dr_ingest.normalization import (
    normalize_compute,
    normalize_ds_str,
    normalize_tokens,
)
from dr_ingest.parse import parse_sl_setup_to_config
from dr_ingest.parallel import list_merge, parallel_process, set_merge

SCALING_LAW_MACRO_FILENAME = "macro_avg-00000-of-00001.parquet"
SCALING_LAW_FIT_FILENAME = "scaling_law_fit-00000-of-00001.parquet"
SCALING_LAW_FILENAMES = [SCALING_LAW_MACRO_FILENAME, SCALING_LAW_FIT_FILENAME]


def str_list_to_dicts(items: list[str]) -> list[dict[str, Any]]:
    """Convert a list of literal strings into dictionaries."""

    return [ast.literal_eval(item) for item in items]


def dict_list_to_all_keys(dicts: list[dict[str, Any]]) -> set[str]:
    """Return the union of keys across dictionaries."""

    all_keys: set[str] = set()
    for mapping in dicts:
        all_keys.update(mapping.keys())
    return all_keys


def make_struct_dtype(keys: list[str], field_types: list[pl.DataType]) -> pl.Struct:
    """Build a Polars struct dtype for the provided keys."""

    return pl.Struct([pl.Field(key, dtype) for key, dtype in zip(keys, field_types, strict=False)])


def parse_dd_results_train(df: pl.DataFrame) -> pl.DataFrame:
    """Expand the literal metrics column into a typed struct."""

    train_metrics = df["metrics"].to_list()
    tm_dicts = parallel_process(train_metrics, str_list_to_dicts, list_merge)
    tm_keys = parallel_process(tm_dicts, dict_list_to_all_keys, set_merge)
    tm_dtype = make_struct_dtype(tm_keys, [pl.Float64] * len(tm_keys))
    return df.drop("metrics").with_columns(pl.Series("metrics", tm_dicts, dtype=tm_dtype))


def parse_train_df(df: pl.DataFrame) -> pl.DataFrame:
    """Produce the cleaned train dataframe with normalized helper columns."""

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
                    per_token=pl.col("metrics").struct.field("uncond_correct_prob_per_token"),
                    per_char=pl.col("metrics").struct.field("uncond_correct_prob_per_char"),
                ),
                norm_correct_prob=pl.struct(
                    raw=pl.col("metrics").struct.field("norm_correct_prob"),
                    per_token=pl.col("metrics").struct.field("norm_correct_prob_per_token"),
                    per_char=pl.col("metrics").struct.field("norm_correct_prob_per_char"),
                ),
                bits_per_byte_correct=pl.col("metrics").struct.field("bits_per_byte_corr"),
                primary_metric=pl.col("metrics").struct.field("primary_metric"),
            ).alias("metrics_struct"),
        )
        .drop("data", "chinchilla", "tokens", "compute", "metrics")
        .rename({"metrics_struct": "metrics"})
        .with_row_index("id")
    )


def parse_sl_results(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Split scaling-law results into the three downstream datasets."""

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
    """Attach parsed config information to each scaling-law row."""

    col_list = df.to_dicts()
    for mapping in col_list:
        mapping["fit_config"] = parse_sl_setup_to_config(mapping["setup"])
        mapping["recipe"] = normalize_ds_str(mapping["mix"])
        del mapping["mix"], mapping["setup"]
    return col_list


def extract_metric_struct(mets_dict: dict[str, Any]) -> dict[str, Any]:
    """Collect the raw/per-token/per-char variants for a metric family."""

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
    true_loss: defaultdict[tuple[Any, Any, Any], dict[str, Any]] = defaultdict(dict)
    true_metrics: defaultdict[tuple[Any, Any, Any], dict[str, Any]] = defaultdict(dict)
    configs: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for mapping in col_list:
        eid = (mapping["task"], mapping["recipe"], mapping["fit_config"]["name"])
        true_loss[eid][mapping["metric"]] = mapping["step_1_y"]
        true_metrics[eid][mapping["metric"]] = mapping["step_2_y"]
        configs[eid] = mapping["fit_config"]

    output: list[dict[str, Any]] = []
    for idx, (eid, cfg) in enumerate(configs.items()):
        if cfg["name"] != "3_param-default":
            continue
        output.append(
            {
                "id": idx,
                "task": eid[0],
                "recipe": eid[1],
                "task_losses": extract_metric_struct(true_loss[eid]),
                "task_metrics": extract_metric_struct(true_metrics[eid]),
            }
        )
    return output


def extract_two_step_preds(col_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    loss_preds: defaultdict[tuple[Any, Any, Any], dict[str, Any]] = defaultdict(dict)
    loss_to_metric_preds: defaultdict[tuple[Any, Any, Any], dict[str, Any]] = defaultdict(dict)
    metric_preds: defaultdict[tuple[Any, Any, Any], dict[str, Any]] = defaultdict(dict)
    configs: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for mapping in col_list:
        eid = (mapping["task"], mapping["recipe"], mapping["fit_config"]["name"])
        loss_preds[eid][mapping["metric"]] = mapping["step_1_pred"]
        loss_to_metric_preds[eid][mapping["metric"]] = mapping["step_2_pred"]
        metric_preds[eid][mapping["metric"]] = mapping["stacked_pred"]
        configs[eid] = mapping["fit_config"]

    output: list[dict[str, Any]] = []
    for idx, (eid, cfg) in enumerate(configs.items()):
        output.append(
            {
                "id": idx,
                "task": eid[0],
                "recipe": eid[1],
                "fit_config": cfg,
                "pred_task_losses": extract_metric_struct(loss_preds[eid]),
                "pred_task_loss_to_metrics": extract_metric_struct(loss_to_metric_preds[eid]),
                "pred_task_metrics": extract_metric_struct(metric_preds[eid]),
            }
        )
    return output


def extract_one_step_preds(col_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics: defaultdict[tuple[Any, Any, Any], dict[str, Any]] = defaultdict(dict)
    configs: dict[tuple[Any, Any, Any], dict[str, Any]] = {}

    for mapping in col_list:
        eid = (mapping["task"], mapping["recipe"], mapping["fit_config"]["name"])
        metrics[eid][mapping["metric"]] = mapping["stacked_pred"]
        configs[eid] = mapping["fit_config"]

    output: list[dict[str, Any]] = []
    for idx, (eid, mets) in enumerate(metrics.items()):
        output.append(
            {
                "id": idx,
                "task": eid[0],
                "recipe": eid[1],
                "fit_config": configs[eid],
                "pred_task_metrics": extract_metric_struct(mets),
            }
        )
    return output


def parse_scaling_law_dir(source_dir: Path) -> dict[str, pl.DataFrame]:
    """Load and parse scaling-law parquet files from a directory.

    Parameters
    ----------
    source_dir:
        Directory containing the macro-average and scaling-law fit parquet files.

    Returns
    -------
    dict[str, pl.DataFrame]
        Dictionary including the macro-average dataframe and parsed scaling-law outputs.
    """

    macro_path = (source_dir / SCALING_LAW_MACRO_FILENAME).resolve()
    fit_path = (source_dir / SCALING_LAW_FIT_FILENAME).resolve()

    missing: list[Path] = [path for path in (macro_path, fit_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing scaling-law parquet files: {missing}")

    macro_df = pl.read_parquet(macro_path)
    scaling_law_df = pl.read_parquet(fit_path)
    outputs = parse_sl_results(scaling_law_df)
    outputs["macro_avg_raw"] = macro_df
    return outputs
