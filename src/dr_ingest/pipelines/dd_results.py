from __future__ import annotations

import ast

import pandas as pd

from dr_ingest.configs.datadecide import DataDecideConfig
from dr_ingest.normalization import (
    normalize_compute,
    normalize_ds_str,
    normalize_tokens,
)

__all__ = [
    "parse_metrics_col",
    "parse_train_df",
]


def parse_metrics_col(
    df: pd.DataFrame, config: DataDecideConfig | None = None
) -> pd.DataFrame:
    cfg = config or DataDecideConfig()
    metrics_dicts = df["metrics"].apply(ast.literal_eval)
    metrics_df = pd.DataFrame(metrics_dicts.tolist())
    metrics_df = metrics_df.rename(columns=cfg.metric_column_renames)
    return df.drop(columns=["metrics"]).join(metrics_df)


def parse_train_df(
    df: pd.DataFrame, config: DataDecideConfig | None = None
) -> pd.DataFrame:
    if config is None:
        config = DataDecideConfig()

    return (
        df.pipe(parse_metrics_col, config=config)
        .assign(
            recipe=df["data"].apply(normalize_ds_str),
            tokens_millions=df["tokens"].apply(normalize_tokens),
            compute_e15=df["compute"].apply(normalize_compute),
        )
        .drop(columns=["data", "chinchilla", "tokens", "compute"])
        .reset_index(drop=False)
        .rename(columns={"index": "id"})
    )
