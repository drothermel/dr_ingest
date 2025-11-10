"""Helpers to retrieve tables from Hugging Face-hosted parquet datasets."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from dr_ingest.configs import AuthSettings, Paths

from .location import HFLocation

__all__ = [
    "download_tables_from_hf",
    "query_data_from_hf",
    "query_with_duckdb",
    "upload_file_to_hf",
]


def upload_file_to_hf(
    local_path: str | Path,
    hf_loc: HFLocation,
    path_in_repo: str,
    *,
    hf_token: str | None = None,
) -> None:
    """Upload a single file to Hugging Face Hub."""
    api = HfApi(token=AuthSettings().resolve("hf", hf_token))
    api.upload_file(
        path_or_fileobj=str(local_path),
        repo_id=hf_loc.repo_id,
        path_in_repo=hf_loc.norm_posix(path_in_repo),
        repo_type=hf_loc.repo_type,
    )


def query_data_from_hf(
    hf_loc: HFLocation,
    *,
    filepaths: list[str | Path] | None = None,
    target_dir: Path | None = None,
    connection: duckdb.DuckDBPyConnection | None = None,
    hf_token: str | None = None,
    force_download: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load tables from a Hugging Face dataset as Pandas DataFrames."""
    if connection is None:
        return download_tables_from_hf(
            hf_loc=hf_loc,
            filepaths=filepaths,
            target_dir=target_dir,
            hf_token=hf_token,
            force_download=force_download,
        )
    return query_with_duckdb(
        hf_loc=hf_loc,
        connection=connection,
        filepaths=filepaths,
        target_dir=target_dir,
        hf_token=hf_token,
    )


def query_with_duckdb(
    hf_loc: HFLocation,
    connection: duckdb.DuckDBPyConnection,
    *,
    filepaths: list[str | Path] | None = None,
    target_dir: Path | None = None,
    hf_token: str | None = None,
    force_download: bool = False,
) -> dict[str, pd.DataFrame]:
    resolved_paths = hf_loc.resolve_filepaths(extra_paths=filepaths)
    hf_uris = hf_loc.get_uris_for_files(resolved_paths, ignore_cfg_files=True)
    results: dict[str, pd.DataFrame] = {}
    for filepath, uri in zip(resolved_paths, hf_uris, strict=True):
        hf_id = uri.removeprefix("hf://")
        results[Path(filepath).stem] = connection.execute(
            f"SELECT * FROM '{hf_id}'"  # noqa: S608
        ).df()
    return results


def download_tables_from_hf(
    hf_loc: HFLocation,
    *,
    filepaths: list[str | Path] | None = None,
    target_dir: Path | None = None,
    hf_token: str | None = None,
    force_download: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download tables directly from Hugging Face storage."""
    resolved_paths = hf_loc.resolve_filepaths(extra_paths=filepaths)
    token = AuthSettings().resolve("hf", hf_token)

    target_dir = target_dir or Paths().data_cache_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, pd.DataFrame] = {}
    for filepath in resolved_paths:
        local_path = hf_hub_download(
            repo_id=hf_loc.repo_id,
            filename=filepath,
            repo_type=hf_loc.repo_type,
            token=token,
            local_dir=str(target_dir),
            force_download=force_download,
        )
        tables[Path(filepath).stem] = pd.read_parquet(local_path)
    return tables
