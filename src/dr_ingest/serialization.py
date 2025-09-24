"""Serialization helpers for reading/writing WandB exports."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Mapping

import duckdb
import pandas as pd
import srsly

logger = logging.getLogger(__name__)


def dump_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    """Write an iterable of mappings to a JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    srsly.write_jsonl(path, rows)
    logger.info("Wrote %s", path)


def dump_runs_and_history(
    out_dir: Path,
    runs_filename: str,
    history_filename: str,
    runs: Iterable[Mapping],
    histories: Iterable[Iterable[Mapping]],
) -> None:
    """Persist runs and history payloads to JSONL files."""

    dump_jsonl(out_dir / f"{runs_filename}.jsonl", runs)
    dump_jsonl(out_dir / f"{history_filename}.jsonl", histories)


def json_to_parquet(json_path: Path, parquet_path: Path) -> None:
    """Convert a JSON (array or JSONL) file to Parquet via DuckDB."""

    table_name = json_path.stem
    duckdb.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_json('{json_path}')")
    duckdb.execute(
        f"COPY {table_name} TO '{parquet_path}' (FORMAT parquet, PARQUET_VERSION v2)"
    )
    logger.info("Converted %s to %s", json_path, parquet_path)


def ensure_parquet(json_path: Path) -> Path:
    """Create a Parquet file alongside json_path if it does not exist."""

    parquet_path = json_path.with_suffix(".parquet")
    if not parquet_path.exists():
        json_to_parquet(json_path, parquet_path)
    return parquet_path


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def compare_sizes(*paths: Path) -> dict[str, float]:
    """Return a mapping of filename -> size (MB)."""

    return {str(path): file_size_mb(path) for path in paths if path.exists()}

