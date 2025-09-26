from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path

import duckdb
import pandas as pd
import srsly

logger = logging.getLogger(__name__)


def dump_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    srsly.write_jsonl(path, rows)
    logger.info("Wrote %s", path)


def _temp_table_name(json_path: Path) -> str:
    digest = hashlib.md5(
        str(json_path).encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    return f"json_{digest}"


def write_parquet(df: pd.DataFrame, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = Path(out_dir / f"{name}.parquet")
    with duckdb.connect(":memory:") as con:
        con.execute(
            f'CREATE OR REPLACE TEMP TABLE "{name}" AS SELECT * FROM df',  # noqa: S608
        )
        con.execute(
            f'COPY "{name}" TO ? (FORMAT parquet, PARQUET_VERSION v2)',
            [str(target_path)],
        )
    print(f"Wrote df {name} to {target_path}")


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


def ensure_convert_json_to_parquet(json_path: Path) -> Path:
    parquet_path = json_path.with_suffix(".parquet")
    if not parquet_path.exists():
        json_to_parquet(json_path, parquet_path)
    return parquet_path


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def compare_sizes(*paths: Path) -> dict[Path, float]:
    return {path: file_size_mb(path) for path in paths if path.exists()}
