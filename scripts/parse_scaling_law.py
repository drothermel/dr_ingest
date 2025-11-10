from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

from dr_ingest.hf import HFLocation, upload_file_to_hf
from dr_ingest.pipelines.dd_results import (
    SCALING_LAW_FILENAMES,
    parse_scaling_law_dir,
)
from dr_ingest.raw_download import (
    DD_RESULTS_REPO,
    DD_RES_OTHER_PATH_FORMAT_STR,
    get_hf_download_path,
    get_hf_fs,
)
from dotenv import load_dotenv

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_SOURCE_DIR = (REPO_ROOT / "data").resolve()
DEFAULT_OUTPUT_DIR = (REPO_ROOT / "data/scaling_law").resolve()
HF_SCALING_LAW_LOCATION = HFLocation(org="drotherm", repo_name="dd_parsed")
HF_SCALING_LAW_REPO_ID = HF_SCALING_LAW_LOCATION.repo_id

OUTPUT_NAME_MAP: dict[str, str] = {
    "macro_avg_raw": "macro_avg.parquet",
    "scaling_law_pred_one_step_raw": "scaling_law_pred_one_step.parquet",
    "scaling_law_pred_two_step_raw": "scaling_law_pred_two_step.parquet",
    "scaling_law_true_raw": "scaling_law_true.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and parse DataDecide scaling-law parquet files."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory where raw scaling-law parquet files will be stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write parsed scaling-law parquet outputs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files if they are present.",
    )
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Force redownload of raw scaling-law parquet files.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload parsed scaling-law parquet files to Hugging Face.",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip parsing and upload existing outputs only.",
    )
    return parser.parse_args()


def download_scaling_law_files(destination: Path, redownload: bool) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    fs = get_hf_fs()
    local_paths: list[Path] = []
    for filename in SCALING_LAW_FILENAMES:
        local_path = destination / filename
        remote_path = get_hf_download_path(
            DD_RESULTS_REPO,
            DD_RES_OTHER_PATH_FORMAT_STR.format(filename),
        )
        if local_path.exists() and not redownload:
            print(f"Skipping download for {filename} (already exists).")
        else:
            print(f"Downloading {remote_path} -> {local_path}")
            with fs.open(remote_path, "rb") as src, local_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        local_paths.append(local_path)
    return local_paths


def write_outputs(
    outputs: dict[str, "polars.DataFrame"],
    destination: Path,
    force: bool,
) -> None:
    import polars

    destination.mkdir(parents=True, exist_ok=True)
    for key, df in outputs.items():
        filename = OUTPUT_NAME_MAP.get(key, f"{key}.parquet")
        output_path = destination / filename
        if output_path.exists() and not force:
            raise FileExistsError(
                f"Output file {output_path} already exists. Use --force to overwrite."
            )
        df.write_parquet(output_path)
        print(
            f"Wrote {output_path} ({df.height if hasattr(df, 'height') else 'n/a'} rows, "
            f"{df.width if hasattr(df, 'width') else 'n/a'} columns)."
        )


def upload_outputs(directory: Path) -> None:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN environment variable is required for uploading to Hugging Face."
        )

    for key, filename in OUTPUT_NAME_MAP.items():
        local_path = directory / filename
        if not local_path.exists():
            print(f"Skipping upload for {filename} (file not found).")
            continue
        print(f"Uploading {local_path} -> {HF_SCALING_LAW_REPO_ID}:{filename}")
        start = time.time()
        upload_file_to_hf(
            local_path=local_path,
            hf_loc=HF_SCALING_LAW_LOCATION,
            path_in_repo=filename,
            hf_token=hf_token,
        )
        elapsed = time.time() - start
        print(f"Upload completed in {elapsed:.2f} seconds.")


def main() -> None:
    load_dotenv()
    args = parse_args()
    source_dir = args.source_dir.expanduser()
    output_dir = args.output_dir.expanduser()

    if args.upload_only:
        if not output_dir.exists():
            raise FileNotFoundError(
                f"Output directory {output_dir} not found; cannot upload-only."
            )
        upload_outputs(output_dir)
        return

    download_scaling_law_files(source_dir, args.redownload)

    outputs = parse_scaling_law_dir(source_dir)
    print("Parsed scaling-law outputs:")
    for name, df in outputs.items():
        rows = getattr(df, "height", "n/a")
        cols = getattr(df, "width", "n/a")
        print(f"- {name}: {rows} rows x {cols} columns")

    write_outputs(outputs, output_dir, args.force)

    if args.upload:
        upload_outputs(output_dir)


if __name__ == "__main__":
    main()
