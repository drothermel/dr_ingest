from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path

import duckdb
import srsly

logger = logging.getLogger(__name__)


def dump_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
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
    dump_jsonl(out_dir / f"{runs_filename}.jsonl", runs)
    dump_jsonl(out_dir / f"{history_filename}.jsonl", histories)


def _temp_table_name(json_path: Path) -> str:
    digest = hashlib.md5(
        str(json_path).encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    return f"json_{digest}"


def json_to_parquet(json_path: Path, parquet_path: Path) -> None:
    table_name = _temp_table_name(json_path)
    try:
        duckdb.execute(
            f'CREATE OR REPLACE TEMP TABLE "{table_name}" AS SELECT * FROM read_json(?)',  # noqa: S608 E501
            [str(json_path)],
        )
        duckdb.execute(
            f'COPY "{table_name}" TO ? (FORMAT parquet, PARQUET_VERSION v2)',
            [str(parquet_path)],
        )
        logger.info("Converted %s to %s", json_path, parquet_path)
    finally:
        duckdb.execute(f'DROP TABLE IF EXISTS "{table_name}"')


def ensure_parquet(json_path: Path) -> Path:
    parquet_path = json_path.with_suffix(".parquet")
    if not parquet_path.exists():
        json_to_parquet(json_path, parquet_path)
    return parquet_path


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def compare_sizes(*paths: Path) -> dict[str, float]:
    return {str(path): file_size_mb(path) for path in paths if path.exists()}
