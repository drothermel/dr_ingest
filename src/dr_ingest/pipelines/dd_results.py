from __future__ import annotations

import ast

import pandas as pd

from dr_ingest.datadec.datadecide import DataDecideConfig
from dr_ingest.utils import group_col_by_prefix

__all__ = [
    "add_task_group",
    "parse_metrics_col",
    "parse_size_string",
    "parse_train_df",
]


def parse_train_df(
    df: pd.DataFrame, config: DataDecideConfig | None = None
) -> pd.DataFrame:
    cfg = config or DataDecideConfig()

    return (
        df.pipe(parse_metrics_col, config=cfg)
        .assign(
            recipe=df["data"].apply(
                lambda x: cfg.recipe_config.normalized_recipe_map.get(x, x)
            ),
            tokens_millions=df["tokens"].apply(lambda x: x / 1e6),
            compute_e15=df["compute"].apply(lambda x: x / 1e15),
            acc_baseline=df["task"].apply(
                lambda x: cfg.task_baselines.get(x, cfg.mmlu_baseline)
            ),
            prob_baseline=cfg.prob_baseline,
            params_numeric=df["params"].apply(lambda x: parse_size_string(x)),
        )
        .pipe(add_task_group, config=cfg)
        .drop(columns=["chinchilla", *cfg.metrics_cols_to_drop])
        .sort_values(by=["params_numeric", "step"])
        .reset_index(drop=True)
    )


def parse_metrics_col(
    df: pd.DataFrame, config: DataDecideConfig | None = None
) -> pd.DataFrame:
    cfg = config or DataDecideConfig()
    metrics_dicts = df["metrics"].apply(ast.literal_eval)
    metrics_df = pd.DataFrame(metrics_dicts.tolist())
    metrics_df = metrics_df.rename(columns=cfg.metric_column_renames)
    return df.drop(columns=["metrics"]).join(metrics_df)


def add_task_group(
    df: pd.DataFrame, config: DataDecideConfig | None = None
) -> pd.DataFrame:
    cfg = config or DataDecideConfig()
    df[cfg.task_group_col] = group_col_by_prefix(
        df,
        column=cfg.task_col,
        prefix_map=cfg.task_group_map,
        output_col=cfg.task_group_col,
    )
    return df


def parse_size_string(
    val: object,
    *,
    allow_plain: bool = False,
) -> float:
    """Parse a size string like \"10M\" or \"1B\" into a float."""

    val_str_original = str(val).strip()

    if not val_str_original or pd.isna(val_str_original):
        raise ValueError(f"Invalid size string: {val}")

    val_str = val_str_original.upper()
    if val_str.endswith("T"):
        unit = 1e12
        val_str = val_str[:-1]
    elif val_str.endswith("B"):
        unit = 1e9
        val_str = val_str[:-1]
    elif val_str.endswith("M"):
        unit = 1e6
        val_str = val_str[:-1]
    elif val_str.endswith("K"):
        unit = 1e3
        val_str = val_str[:-1]
    else:
        raise ValueError(
            f"Invalid size string: {val_str_original} (requires K/M/B/T suffix)"
        )
    return float(val_str) * unit
