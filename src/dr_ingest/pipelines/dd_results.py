from __future__ import annotations

import ast
from typing import Any

import polars as pl

from dr_ingest.normalization import (
    normalize_compute,
    normalize_ds_str,
    normalize_tokens,
)
from dr_ingest.parallel import list_merge, parallel_process, set_merge

__all__ = [
    "dict_list_to_all_keys",
    "make_struct_dtype",
    "parse_dd_results_train",
    "parse_train_df",
    "str_list_to_dicts",
]


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

    return pl.Struct(
        [pl.Field(key, dtype) for key, dtype in zip(keys, field_types, strict=False)]
    )


def parse_dd_results_train(df: pl.DataFrame) -> pl.DataFrame:
    """Expand the literal metrics column into a typed struct."""

    train_metrics = df["metrics"].to_list()
    tm_dicts = parallel_process(train_metrics, str_list_to_dicts, list_merge)
    tm_keys = parallel_process(tm_dicts, dict_list_to_all_keys, set_merge)
    tm_dtype = make_struct_dtype(tm_keys, [pl.Float64] * len(tm_keys))
    return df.drop("metrics").with_columns(
        pl.Series("metrics", tm_dicts, dtype=tm_dtype)
    )


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
        .with_row_index("id")
    )
