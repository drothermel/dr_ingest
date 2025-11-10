from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl
import typer
from pydantic import Undefined

from dr_ingest.configs import (
    DataDecideConfig,
    DataDecideSourceConfig,
    ParsedSourceConfig,
    Paths,
)
from dr_ingest.hf import download_tables_from_hf, upload_file_to_hf
from dr_ingest.pipelines.dd_results import parse_train_df

app = typer.Typer()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse DD train shards into a consolidated parquet dataset."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the parsed parquet to Hugging Face using DuckDB COPY.",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip parsing and only upload the existing output parquet.",
    )
    return parser.parse_args()


def resolve_local_datadecide_filepaths(
    *,
    paths: Paths | None = None,
    source_config: DataDecideSourceConfig | None = None,
) -> list[str]:
    paths = paths or Paths()
    source_cfg = source_config or DataDecideSourceConfig()

    return [
        f"{paths.data_cache_dir}/{fp}"
        for fp in source_cfg.results_hf.resolve_filepaths()
    ]


def resolve_parsed_output_path(
    *,
    paths: Paths | None = None,
    parsed_config: ParsedSourceConfig | None = None,
) -> Path:
    paths = paths or Paths()
    parsed_cfg = parsed_config or ParsedSourceConfig()
    parsed_hf_loc = parsed_cfg.pretrain
    results_filename = parsed_hf_loc.get_the_single_filepath()
    output_path = Path(paths.data_cache_dir / results_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def validate_and_merge_tables(expected_paths: list[str]) -> pd.DataFrame:
    shard_dfs: list[pd.DataFrame] = []
    for path in expected_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing downloaded shard for {path}")
        shard_dfs.append(pd.read_parquet(path))
    return pd.concat(shard_dfs, ignore_index=True)


@app.command()
def download(
    force: bool = False,
    data_cache_dir: str | None = None,
) -> None:
    """Download raw Data Decide Results from HF to Local"""
    paths = Paths(data_cache_dir=data_cache_dir or Undefined)  # type: ignore
    dd_cfg = DataDecideConfig()
    dd_source_hf_loc = dd_cfg.source_config.results_hf
    table_paths = download_tables_from_hf(
        dd_source_hf_loc,
        local_dir=paths.data_cache_dir,
        force_download=force,
    )
    print(f">> Downloaded tables: {table_paths} to {paths.data_cache_dir}")


@app.command()
def parse(
    force: bool = False,
    data_cache_dir: str | None = None,
) -> None:
    """Parse already downloaded Data Decide Results"""
    paths = Paths(data_cache_dir=data_cache_dir or Undefined)  # type: ignore
    source_filepaths = resolve_local_datadecide_filepaths(paths=paths)
    output_path = resolve_parsed_output_path(paths=paths)

    # Load and parse the tables
    source_df = validate_and_merge_tables(source_filepaths)
    parsed_df = (parse_train_df(pl.from_pandas(source_df))).to_pandas()
    parsed_df.to_parquet(output_path, index=False)
    print(f">> Wrote parsed train results to {output_path}")


@app.command()
def upload(
    data_cache_dir: str | None = None,
) -> None:
    """Upload parsed Data Decide Results from local to HF"""
    paths = Paths(data_cache_dir=data_cache_dir or Undefined)  # type: ignore
    parsed_pretrain_loc = ParsedSourceConfig().pretrain
    output_path = resolve_parsed_output_path(paths=paths)
    if not output_path.exists():
        raise FileNotFoundError(
            f"Output file {output_path} not found; cannot upload-only."
        )
    print(f">> Upload Only: {output_path} to {parsed_pretrain_loc}")
    upload_file_to_hf(local_path=output_path, hf_loc=parsed_pretrain_loc)


@app.command()
def full_pipeline(
    force: bool = False,
    data_cache_dir: str | None = None,
) -> None:
    download(force, data_cache_dir)
    parse(force, data_cache_dir)
    upload(data_cache_dir)


if __name__ == "__main__":
    app()
