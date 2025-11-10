from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import polars as pl

from dr_ingest.configs import DataDecideConfig, ParsedSourceConfig, Paths
from dr_ingest.hf import HFLocation, download_tables_from_hf, upload_file_to_hf
from dr_ingest.pipelines.dd_results import parse_train_df


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


def upload_parquet_to_hf(local_path: Path, hf_loc: HFLocation) -> None:
    if not local_path.exists():
        raise FileNotFoundError(f"Parquet file {local_path} does not exist for upload.")
    print(f"Uploading {local_path} to huggingface {hf_loc}")
    start = time.time()
    upload_file_to_hf(
        local_path=local_path,
        hf_loc=hf_loc,
    )
    elapsed = time.time() - start
    print(f"Upload completed in {elapsed:.2f} seconds.")


def main() -> None:
    args = parse_args()
    paths = Paths()

    dd_cfg = DataDecideConfig()
    dd_source_hf_loc = dd_cfg.source_config.results_hf

    parsed_cfg = ParsedSourceConfig()
    parsed_hf_loc = parsed_cfg.pretrain
    results_filename = parsed_hf_loc.get_the_single_filepath()

    output_path = Path(paths.data_cache_dir / results_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.upload_only:
        if not output_path.exists():
            raise FileNotFoundError(
                f"Output file {output_path} not found; cannot upload-only."
            )
        upload_parquet_to_hf(output_path, dd_source_hf_loc)
        return

    filepaths = dd_source_hf_loc.filepaths or []
    if not filepaths:
        raise ValueError("DataDecide config missing train shard filepaths.")

    tables = download_tables_from_hf(
        hf_loc=dd_source_hf_loc,
        target_dir=paths.data_cache_dir,
        force_download=args.force,
    )

    shard_frames_pd: list[pd.DataFrame] = []
    print("Loaded shards:")
    for relative_path in filepaths:
        stem = Path(relative_path).stem
        frame = tables.get(stem)
        if frame is None:
            raise KeyError(f"Missing downloaded shard for {relative_path}")
        shard_frames_pd.append(frame)
        print(f"- {relative_path}: {len(frame)} rows")

    combined_pd = pd.concat(shard_frames_pd, ignore_index=True)
    combined_pl = pl.from_pandas(combined_pd)
    parsed_pl = parse_train_df(combined_pl)
    parsed = parsed_pl.to_pandas()

    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Output file {output_path} already exists. Use --force to overwrite."
        )
    parsed.to_parquet(output_path, index=False)
    print(
        f"Wrote parsed train results to {output_path} "
        f"({parsed.shape[0]} rows, {parsed.shape[1]} columns)."
    )

    if args.upload:
        upload_parquet_to_hf(output_path, parsed_hf_loc)


if __name__ == "__main__":
    main()
