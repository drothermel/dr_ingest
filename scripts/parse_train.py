from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from huggingface_hub import HfFileSystem

from dr_ingest import HFLocation
from dr_ingest.configs import DataDecideConfig, Paths
from dr_ingest.hf.hf_upload import upload_file_to_hf
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


def download_train_shards(
    destination: Path,
    redownload: bool,
    hf_location: HFLocation,
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    filepaths = hf_location.filepaths or []
    if not filepaths:
        raise ValueError(
            "HF location must include explicit filepaths for train shard download."
        )

    local_paths: list[Path] = []
    for filepath in filepaths:
        filename = Path(filepath).name
        local_path = destination / filename
        remote_path = hf_location.get_path_uri(filepath)
        if local_path.exists() and not redownload:
            print(f"Skipping download for {filename} (already exists).")
        else:
            print(f"Downloading {remote_path} -> {local_path}")
            with fs.open(remote_path, "rb") as src, local_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        local_paths.append(local_path)
    return local_paths


def read_shards(
    file_paths: list[Path],
) -> tuple[list[Path], list[pl.DataFrame], pl.DataFrame]:
    if not file_paths:
        raise FileNotFoundError("No train shard paths provided for parsing.")
    shard_frames = [pl.read_parquet(path) for path in file_paths]
    combined = pl.concat(shard_frames, how="vertical")
    return file_paths, shard_frames, combined


def upload_parquet_to_hf(local_path: Path) -> None:
    if not local_path.exists():
        raise FileNotFoundError(f"Parquet file {local_path} does not exist for upload.")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise OSError(
            "HF_TOKEN environment variable is required for uploading to Hugging Face."
        )

    print(f"Uploading {local_path} to huggingface")
    start = time.time()
    upload_file_to_hf(
        local_path,
        repo_id="drotherm/dd_parsed",
        path_in_repo="train_results.parquet",
        token=hf_token,
        repo_type="dataset",
    )
    elapsed = time.time() - start
    print(f"Upload completed in {elapsed:.2f} seconds.")


def main() -> None:
    # Old Config System
    load_dotenv()
    args = parse_args()

    # New Config System
    paths = Paths()
    dd_cfg = DataDecideConfig()

    output_path = Path(paths.data_cache_dir / dd_cfg.results_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.upload_only:
        if not output_path.exists():
            raise FileNotFoundError(
                f"Output file {output_path} not found; cannot upload-only."
            )
        upload_parquet_to_hf(output_path)
        return

    shard_paths = download_train_shards(
        paths.data_cache_dir,
        args.force,
        dd_cfg.source_config.results_hf,
    )
    shard_paths, shard_frames, combined = read_shards(shard_paths)
    print("Loaded shards:")
    for path, frame in zip(shard_paths, shard_frames, strict=False):
        print(f"- {path.name}: {frame.shape[0]} rows")

    parsed = parse_train_df(combined)

    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Output file {output_path} already exists. Use --force to overwrite."
        )
    parsed.write_parquet(output_path)
    print(
        f"Wrote parsed train results to {output_path} "
        f"({parsed.height} rows, {parsed.width} columns)."
    )

    if args.upload:
        upload_parquet_to_hf(output_path)


if __name__ == "__main__":
    main()
