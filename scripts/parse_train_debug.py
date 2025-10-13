from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

from dr_ingest.pipelines.dd_results import parse_dd_results_train, parse_train_df

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_SOURCE_DIR = (
    REPO_ROOT / "datadec/data/raw_downloads/DD-eval-results/data"
).resolve()
DEFAULT_FIXTURE_DIR = (REPO_ROOT / "dr_ingest/tests/fixtures/data/dd_results").resolve()
DEFAULT_OUTPUT_PATH = Path("data/train_results.parquet")

RAW_TO_FINAL_KEYS: dict[str, str] = {
    "acc_raw": "accuracy.raw",
    "acc_per_token": "accuracy.per_token",
    "acc_per_char": "accuracy.per_char",
    "acc_per_byte": "accuracy.per_byte",
    "acc_uncond": "accuracy.uncond",
    "sum_logits_corr": "sum_logits_corr.raw",
    "logits_per_token_corr": "sum_logits_corr.per_token",
    "logits_per_char_corr": "sum_logits_corr.per_char",
    "correct_prob": "correct_prob.raw",
    "correct_prob_per_token": "correct_prob.per_token",
    "correct_prob_per_char": "correct_prob.per_char",
    "margin": "margin.raw",
    "margin_per_token": "margin.per_token",
    "margin_per_char": "margin.per_char",
    "total_prob": "total_prob.raw",
    "total_prob_per_token": "total_prob.per_token",
    "total_prob_per_char": "total_prob.per_char",
    "uncond_correct_prob": "uncond_correct_prob.raw",
    "uncond_correct_prob_per_token": "uncond_correct_prob.per_token",
    "uncond_correct_prob_per_char": "uncond_correct_prob.per_char",
    "norm_correct_prob": "norm_correct_prob.raw",
    "norm_correct_prob_per_token": "norm_correct_prob.per_token",
    "norm_correct_prob_per_char": "norm_correct_prob.per_char",
    "bits_per_byte_corr": "bits_per_byte_correct",
    "primary_metric": "primary_metric",
}

TRAIN_GLOB = "train-*.parquet"


def default_source_dir() -> Path:
    if DEFAULT_SOURCE_DIR.exists():
        return DEFAULT_SOURCE_DIR
    return DEFAULT_FIXTURE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse DD train shards into a consolidated parquet dataset."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=default_source_dir(),
        help="Directory containing DD results parquet shards.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="Row index to inspect throughout the pipeline.",
    )
    parser.add_argument(
        "--params", type=str, default=None, help="Filter by params value (e.g. 1B)."
    )
    parser.add_argument(
        "--task", type=str, default=None, help="Filter by task name (exact match)."
    )
    parser.add_argument(
        "--data-contains",
        type=str,
        default=None,
        help="Filter rows where the data column contains this substring.",
    )
    parser.add_argument(
        "--metrics-key",
        type=str,
        default="norm_correct_prob",
        help="Metric key to highlight in comparisons.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination parquet path for consolidated train results.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print row-level debug information after writing the output parquet.",
    )
    return parser.parse_args()


def read_shards(
    source_dir: Path,
) -> tuple[list[Path], list[pl.DataFrame], pl.DataFrame]:
    shard_paths = sorted(source_dir.glob(TRAIN_GLOB))
    if not shard_paths:
        msg = f"No train shards matching {TRAIN_GLOB} in {source_dir}"
        raise FileNotFoundError(msg)
    shard_dfs = [pl.read_parquet(path) for path in shard_paths]
    combined = pl.concat(shard_dfs, how="vertical")
    return shard_paths, shard_dfs, combined


def filter_dataframe(
    df: pl.DataFrame,
    params: str | None,
    task: str | None,
    data_contains: str | None,
) -> pl.DataFrame:
    expr = None
    if params is not None:
        expr = (
            pl.col("params") == params
            if expr is None
            else expr & (pl.col("params") == params)
        )
    if task is not None:
        expr = (
            pl.col("task") == task if expr is None else expr & (pl.col("task") == task)
        )
    if data_contains is not None:
        expr = (
            pl.col("data").str.contains(data_contains)
            if expr is None
            else expr & pl.col("data").str.contains(data_contains)
        )
    return df if expr is None else df.filter(expr)


def literal_metrics(value: Any) -> Mapping[str, Any]:
    if isinstance(value, str):
        return ast.literal_eval(value)
    if isinstance(value, Mapping):
        return value
    msg = f"Unsupported metrics literal type: {type(value)!r}"
    raise TypeError(msg)


def safe_parse_dd_results(df: pl.DataFrame) -> pl.DataFrame:
    try:
        return parse_dd_results_train(df)
    except TypeError:
        metrics_dicts = [literal_metrics(item) for item in df["metrics"].to_list()]
        all_keys = sorted({key for item in metrics_dicts for key in item})
        struct_dtype = pl.Struct([pl.Field(key, pl.Float64) for key in all_keys])
        metrics_series = pl.Series("metrics", metrics_dicts, dtype=struct_dtype)
        return df.drop("metrics").with_columns(metrics_series)


def flatten_metrics(node: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in node.items():
        dotted = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_metrics(value, dotted))
        else:
            flat[dotted] = value
    return flat


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    paths, shards, combined = read_shards(source_dir)
    print("Loaded shards:")
    for path, shard in zip(paths, shards, strict=False):
        print(f"- {path.name}: {shard.shape[0]} rows")

    parsed_all = parse_train_df(combined)
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Output file {output_path} already exists. Use --force to overwrite."
        )

    parsed_all.write_parquet(output_path)
    print(
        f"\nWrote parsed train results to {output_path} "
        f"({parsed_all.height} rows, {parsed_all.width} columns)."
    )

    if not args.inspect:
        return

    filtered = filter_dataframe(combined, args.params, args.task, args.data_contains)
    if filtered.is_empty():
        print("\nNo rows matched the provided filters for inspection.")
        return

    if args.row_index >= filtered.height:
        raise IndexError(
            f"Row index {args.row_index} out of range for "
            f"filtered frame with {filtered.height} rows."
        )

    row = filtered.row(args.row_index, named=True)
    print("\nSelected row metadata:")
    print(
        f"  params={row['params']} task={row['task']} "
        f"data={row['data']} step={row['step']} seed={row['seed']}"
    )

    raw_metrics = literal_metrics(row["metrics"])
    print("\nRaw metrics (literal):")
    for key in sorted(raw_metrics):
        print(f"  {key}: {raw_metrics[key]}")

    parsed_subset = safe_parse_dd_results(filtered)
    struct_metrics = parsed_subset[args.row_index, "metrics"]
    print("\nParsed metrics struct (post parse_dd_results_train):")
    for key in sorted(struct_metrics.keys()):
        print(f"  {key}: {struct_metrics[key]}")

    final_df = parse_train_df(filtered)
    final_struct = final_df[args.row_index, "metrics"]
    flattened = flatten_metrics(final_struct)
    print("\nFinal nested metrics (post parse_train_df):")
    for key in sorted(flattened):
        print(f"  {key}: {flattened[key]}")

    print("\nComparison vs raw literal:")
    for key in sorted(raw_metrics):
        raw_value = raw_metrics[key]
        final_key = RAW_TO_FINAL_KEYS.get(key)
        if final_key is None:
            print(f"  {key}: missing mapping (value={raw_value})")
            continue
        final_value = flattened.get(final_key)
        if final_value is None:
            print(f"  {key} -> {final_key}: missing in final struct")
            continue
        status = "OK" if raw_value == final_value else "DIFF"
        print(f"  {key} -> {final_key}: {raw_value} vs {final_value} [{status}]")

    highlight = args.metrics_key
    highlight_mapping = RAW_TO_FINAL_KEYS.get(highlight)
    if highlight_mapping:
        print("\nHighlighted metric:")
        print(f"  raw[{highlight}] = {raw_metrics.get(highlight)}")
        print(f"  parsed[{highlight}] = {struct_metrics.get(highlight)}")
        print(f"  final[{highlight_mapping}] = {flattened.get(highlight_mapping)}")


if __name__ == "__main__":
    main()
