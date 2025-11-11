"""Helpers to retrieve tables from Hugging Face-hosted parquet datasets."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from dr_ingest.configs.auth import AuthSettings
from dr_ingest.configs.paths import Paths

from .location import HFLocation

__all__ = [
    "cached_download_tables_from_hf",
    "get_tables_from_cache",
    "query_hf_with_duckdb",
    "read_local_parquet_paths",
    "upload_file_to_hf",
]


def upload_file_to_hf(
    local_path: str | Path,
    hf_loc: HFLocation,
    *,
    hf_token: str | None = None,
) -> None:
    """Upload a single file to Hugging Face Hub."""
    api = HfApi(token=AuthSettings().resolve("hf", hf_token))
    api.upload_file(
        path_or_fileobj=str(local_path),
        repo_id=hf_loc.repo_id,
        path_in_repo=hf_loc.get_the_single_filepath(),
        repo_type=hf_loc.hf_hub_repo_type,
    )


def get_tables_from_cache(
    hf_loc: HFLocation,
    local_dir: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    local_dir = Path(local_dir or Paths().data_cache_dir)
    local_paths = hf_loc.resolve_filepaths(local_dir=local_dir)
    for fp in local_paths:
        if not Path(fp).exists():
            raise FileNotFoundError(f"Local file not found: {fp}")
    return read_local_parquet_paths(local_paths)  # type: ignore


def query_hf_with_duckdb(
    hf_loc: HFLocation,
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, pd.DataFrame]:
    resolved_paths = hf_loc.resolve_filepaths()
    hf_uris = hf_loc.get_uris_for_files(resolved_paths, ignore_cfg_files=True)
    results: dict[str, pd.DataFrame] = {}
    for filepath, uri in zip(resolved_paths, hf_uris, strict=False):
        hf_id = uri.removeprefix("hf://")
        results[Path(filepath).stem] = connection.execute(
            f"SELECT * FROM '{hf_id}'"  # noqa: S608
        ).df()
    return results


def cached_download_tables_from_hf(
    hf_loc: HFLocation,
    *,
    local_dir: Path | str | None = None,
    hf_token: str | None = None,
    force_download: bool = False,
    verbose: bool = True,
) -> dict[str, str | Path]:
    """Download tables directly from Hugging Face storage."""

    local_dir = Path(local_dir or Paths().data_cache_dir)
    local_paths = hf_loc.resolve_filepaths(local_dir=local_dir)
    if not force_download and all(Path(fp).exists() for fp in local_paths):
        if verbose:
            print(f">> All tables already cached:\n - {'\n - '.join(local_paths)}")
        return {Path(fp).stem: fp for fp in local_paths}

    local_dir.mkdir(parents=True, exist_ok=True)
    remote_paths = hf_loc.resolve_filepaths()
    tables: dict[str, str | Path] = {}
    for remote_path in remote_paths:
        local_path = hf_hub_download(
            repo_id=hf_loc.repo_id,
            filename=remote_path,
            repo_type=hf_loc.hf_hub_repo_type,
            token=AuthSettings().resolve("hf", hf_token),
            local_dir=hf_loc.build_local_dir(local_dir),
            force_download=force_download,
        )
        tables[remote_path] = local_path

    if verbose:
        print(f">> Downloaded {hf_loc.org}/{hf_loc.repo_name} tables:")
        print("\n".join([f" - {rem} -> {loc}" for rem, loc in tables.items()]))
    return tables


def read_local_parquet_paths(local_paths: list[Path | str]) -> dict[str, pd.DataFrame]:
    return {Path(fp).stem: pd.read_parquet(fp) for fp in local_paths}
