from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import duckdb
import polars as pl
from dotenv import load_dotenv

from dr_ingest.pipelines.dd_results import parse_train_df
from dr_ingest.raw_download import (
    DD_NUM_TRAIN_FILES,
    DD_RESULTS_REPO,
    DD_TRAIN_FILE_PATH_FORMAT_STR,
    get_hf_download_path,
    get_hf_fs,
)

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_SOURCE_DIR = Path("data")
DEFAULT_OUTPUT_PATH = Path("data/train_results.parquet")
HF_TRAIN_RESULTS_URI = "hf://datasets/drotherm/dd_parsed/train_results.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse DD train shards into a consolidated parquet dataset."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Local directory where raw train shards will be stored.",
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
        "--redownload",
        action="store_true",
        help="Force redownload of train shards even if they already exist.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the parsed parquet to Hugging Face using DuckDB COPY.",
    )
    return parser.parse_args()


def download_train_shards(destination: Path, redownload: bool) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    fs = get_hf_fs()
    local_paths: list[Path] = []
    for idx in range(DD_NUM_TRAIN_FILES):
        filename = DD_TRAIN_FILE_PATH_FORMAT_STR.format(idx).split("/")[-1]
        local_path = destination / filename
        remote_path = get_hf_download_path(
            DD_RESULTS_REPO, f"data/{filename}"
        )
        if local_path.exists() and not redownload:
            print(f"Skipping download for {filename} (already exists).")
        else:
            print(f"Downloading {remote_path} -> {local_path}")
            with fs.open(remote_path, "rb") as src, local_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        local_paths.append(local_path)
    return local_paths


def read_shards(file_paths: list[Path]) -> tuple[list[Path], list[pl.DataFrame], pl.DataFrame]:
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
        raise EnvironmentError(
            "HF_TOKEN environment variable is required for uploading to Hugging Face."
        )

    print(f"Uploading {local_path} -> {HF_TRAIN_RESULTS_URI}")
    with duckdb.connect() as conn:
        # Ensure HTTPFS extension is available for remote COPY operations.
        conn.execute("INSTALL httpfs;")
        conn.execute("LOAD httpfs;")

        token_literal = hf_token.replace("'", "''")
        conn.execute(
            f"""
            CREATE SECRET IF NOT EXISTS hf_token (
                TYPE HUGGINGFACE,
                TOKEN '{token_literal}'
            );
            """
        )

        start = time.time()
        conn.execute(
            "COPY (SELECT * FROM read_parquet(?)) TO ? (FORMAT PARQUET);",
            [str(local_path), HF_TRAIN_RESULTS_URI],
        )
        elapsed = time.time() - start
    print(f"Upload completed in {elapsed:.2f} seconds.")


def main() -> None:
    load_dotenv()
    args = parse_args()
    source_dir = args.source_dir.expanduser()
    shard_paths = download_train_shards(source_dir, args.redownload)
    shard_paths, shard_frames, combined = read_shards(shard_paths)
    print("Loaded shards:")
    for path, frame in zip(shard_paths, shard_frames, strict=False):
        print(f"- {path.name}: {frame.shape[0]} rows")

    parsed = parse_train_df(combined)

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
