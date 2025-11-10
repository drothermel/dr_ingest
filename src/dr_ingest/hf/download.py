"""Helpers to retrieve tables from Hugging Face-hosted parquet datasets."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import duckdb
import pandas as pd
from huggingface_hub import hf_hub_download

from dr_ingest.configs import AuthSettings, Paths

from .hf_location import HFLocation

__all__ = [
    "download_tables_from_hf",
    "query_data_from_hf",
]


def query_data_from_hf(
    hf_loc: HFLocation,
    *,
    filepaths: Iterable[str] | None = None,
    target_dir: Path | None = None,
    connection: duckdb.DuckDBPyConnection | None = None,
    hf_token: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load tables from a Hugging Face dataset as Pandas DataFrames."""
    resolved_paths = _resolve_filepaths(hf_loc, filepaths)

    if connection is not None:
        hf_uris = hf_loc.get_uris_for_files(resolved_paths, ignore_cfg_files=True)
        results: dict[str, pd.DataFrame] = {}
        for filepath, uri in zip(resolved_paths, hf_uris, strict=True):
            hf_id = uri.removeprefix("hf://")
            results[Path(filepath).stem] = connection.execute(
                f"SELECT * FROM '{hf_id}'"  # noqa: S608
            ).df()
        return results

    return download_tables_from_hf(
        hf_loc=hf_loc,
        filepaths=resolved_paths,
        target_dir=target_dir,
        hf_token=hf_token,
    )


def download_tables_from_hf(
    hf_loc: HFLocation,
    *,
    filepaths: Iterable[str] | None = None,
    target_dir: Path | None = None,
    hf_token: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Download tables directly from Hugging Face storage."""
    resolved_paths = _resolve_filepaths(hf_loc, filepaths)
    token = _resolve_hf_token(hf_token)

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
        )
        tables[Path(filepath).stem] = pd.read_parquet(local_path)
    return tables


def _resolve_filepaths(
    hf_loc: HFLocation,
    filepaths: Iterable[str] | None,
) -> list[str]:
    resolved = list(filepaths or hf_loc.filepaths or [])
    if not resolved:
        raise ValueError(
            "HFLocation must define `filepaths` or an explicit `filepaths` "
            "argument must be provided."
        )
    return resolved


def _resolve_hf_token(explicit_token: str | None) -> str | None:
    if explicit_token:
        return explicit_token
    auth = AuthSettings()
    return os.getenv(auth.hf_env_var) if auth.hf_env_var else None
