"""CLI for streaming DataDec eval directory ingestion."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import typer

from dr_ingest.configs.paths import Paths
from dr_ingest.metrics_all.constants import LoadMetricsAllConfig
from dr_ingest.pipelines.datadec_ingest import (
    clean_cached_results,
    collect_all_eval_paths,
    export_eval_dir_to_parquet,
)


app = typer.Typer(help=__doc__)


def _resolve_defaults(
    metrics_dir: Optional[Path], cache_dir: Optional[Path]
) -> tuple[Path, Path]:
    defaults = Paths()
    metrics = Path(metrics_dir) if metrics_dir else defaults.metrics_all_dir
    cache = Path(cache_dir) if cache_dir else defaults.data_cache_dir / "new_ingest"
    cache.mkdir(parents=True, exist_ok=True)
    return metrics, cache


@app.command()
def cache(
    metrics_dir: Optional[Path] = typer.Option(
        None,
        help="Root directory that contains *_eval_results folders",
        file_okay=False,
        dir_okay=True,
        exists=True,
    ),
    cache_dir: Optional[Path] = typer.Option(
        None,
        help="Directory to write combined parquet + metadata files",
        file_okay=False,
        dir_okay=True,
    ),
    results_dir: Optional[list[Path]] = typer.Option(
        None,
        help="Specific results directories to process (default all)",
    ),
    force: bool = typer.Option(False, help="Overwrite existing cache files"),
    log_path: Optional[Path] = typer.Option(
        None,
        help="Path to append ingestion errors (default cache_dir/ingest_errors.log)",
    ),
    doc_id: Optional[int] = typer.Option(
        None,
        help="If provided, only materialize rows with this doc_id",
    ),
    slug_prefix: str = typer.Option(
        "",
        help="Prefix to prepend to each output slug (e.g., 'nov03_')",
    ),
    memory_limit: str = typer.Option(
        "8GB",
        help="DuckDB memory limit (e.g., '8GB'); data beyond this spills to disk",
    ),
    threads: int = typer.Option(
        2,
        help="Maximum DuckDB threads per directory (set <= CPU cores)",
    ),
):
    """Stream each eval directory into a cached parquet file."""

    metrics_dir, cache_dir = _resolve_defaults(metrics_dir, cache_dir)
    log_path = log_path or (cache_dir / "ingest_errors.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    load_cfg = LoadMetricsAllConfig(root_paths=[metrics_dir])
    entries = collect_all_eval_paths(load_cfg)

    if results_dir:
        wanted = {Path(p).resolve() for p in results_dir}
        entries = [
            entry
            for entry in entries
            if Path(entry["results_dir"]).resolve() in wanted
        ]

    if not entries:
        typer.echo("No eval directories found", err=True)
        raise typer.Exit(code=1)

    processed = 0
    skipped = 0
    for entry in entries:
        results_path = Path(entry["results_dir"])
        try:
            relative = results_path.relative_to(metrics_dir)
            relative_slug = str(relative).strip("/") or results_path.name
        except ValueError:
            relative_slug = results_path.name

        safe_slug = slug_prefix + relative_slug.replace("/", "__")
        if doc_id is not None:
            safe_slug = f"{safe_slug}__doc{doc_id}"
        parquet_path = cache_dir / f"{safe_slug}.parquet"
        metadata_path = cache_dir / f"{safe_slug}.json"

        if parquet_path.exists() and metadata_path.exists() and not force:
            typer.echo(f"-> Skipping {results_path} (already cached as {safe_slug})")
            skipped += 1
            continue

        typer.echo(f"-> Processing {results_path} -> {safe_slug}")
        spill_dir = cache_dir / "spill"
        spill_dir.mkdir(parents=True, exist_ok=True)

        try:
            with duckdb.connect(
                config={
                    "memory_limit": memory_limit,
                    "temp_directory": str(spill_dir),
                    "threads": str(max(1, threads)),
                }
            ) as conn:
                conn.execute("PRAGMA preserve_insertion_order=false;")
                export_eval_dir_to_parquet(
                    conn,
                    entry,
                    cache_dir,
                    force=force,
                    doc_id_value=doc_id,
                    slug_override=safe_slug,
                )
            processed += 1
        except Exception as exc:  # noqa: BLE001
            timestamp = datetime.now().isoformat()
            message = f"[{timestamp}] Failed for {results_path}: {exc}\n"
            typer.echo(f"!! Error for {results_path}, see {log_path}", err=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(message)

    typer.echo(f"Finished {processed} directories (skipped {skipped}).")


@app.command()
def dedupe(
    cache_dir: Optional[Path] = typer.Option(
        None,
        help="Directory that contains cached parquet files",
        file_okay=False,
        dir_okay=True,
    ),
    output_name: str = typer.Option(
        "deduped.parquet",
        help="Name of the deduplicated parquet file",
    ),
    drop_column: list[str] = typer.Option(
        [],
        help="Additional columns to drop before deduplicating",
    ),
):
    """Deduplicate cached parquet files and drop path-related columns."""

    _, cache_dir = _resolve_defaults(None, cache_dir)
    conn = duckdb.connect()
    try:
        output_path = clean_cached_results(
            conn,
            cache_dir,
            drop_columns=drop_column or None,
            output_name=output_name,
        )
    finally:
        conn.close()

    typer.echo(f"Wrote {output_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
