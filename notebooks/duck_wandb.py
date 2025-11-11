import marimo

__generated_with = "0.16.2"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    # Keeping: Seems like a plausible base for wandb fetching
    import hashlib
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    import duckdb
    import srsly
    from dr_wandb import fetch_project_runs

    from dr_ingest.normalization import normalize_str

    def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        srsly.write_jsonl(path, rows)
        print(f"Wrote {path}")

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
            print(f"Converted {json_path} to {parquet_path}")
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

    def make_progress_callback(log_every: int = 10) -> Callable:
        def progress(idx: int, total: int, name: str) -> None:
            if idx % log_every == 0:
                print(f"Processing run {idx}/{total}: {name}")

        return progress

    return (
        compare_sizes,
        dump_jsonl,
        ensure_convert_json_to_parquet,
        fetch_project_runs,
        make_progress_callback,
        normalize_str,
    )


@app.cell
def _(Path):
    def get_file_size_mb(file_path):
        return Path(file_path).stat().st_size / (1024 * 1024)

    return


@app.cell
def _(
    dump_jsonl,
    ensure_convert_json_to_parquet,
    fetch_project_runs,
    history_json,
    make_progress_callback,
    pd,
    runs_json,
):
    def load_or_download_all_from_wandb(
        entity,
        project,
        runs_jsonl_path,
        history_jsonl_path,
        runs_per_page=500,
        log_every: int = 10,
    ):
        if not runs_json.exists() or not history_json.exists():
            progress = make_progress_callback(log_every)
            runs, histories = fetch_project_runs(
                entity,
                project,
                runs_per_page=runs_per_page,
                include_history=True,
                progress_callback=progress,
            )
            print(
                f">> Finished downloading, {len(runs)} runs and {len(histories)} histories"
            )
            dump_jsonl(runs_jsonl_path, runs)
            dump_jsonl(history_jsonl_path, histories)

        runs_parquet_path = ensure_convert_json_to_parquet(runs_jsonl_path)
        history_parquet_path = ensure_convert_json_to_parquet(history_jsonl_path)
        return (
            pd.read_parquet(runs_parquet_path),
            pd.read_parquet(history_parquet_path),
        )

    return


@app.cell
def _(
    OUT_DIR,
    Path,
    WANDB_HISTORY_FILENAME,
    WANDB_RUNS_FILENAME,
    compare_sizes,
):
    runs_json = Path(OUT_DIR, f"{WANDB_RUNS_FILENAME}.jsonl")
    history_json = Path(OUT_DIR, f"{WANDB_HISTORY_FILENAME}.jsonl")
    runs_parquet = runs_json.with_suffix(".parquet")
    history_parquet = history_json.with_suffix(".parquet")

    for path, size in compare_sizes(
        runs_json, runs_parquet, history_json, history_parquet
    ).items():
        print(f"{path.name}: {size:.2f} MB")
    return history_json, runs_json, runs_parquet


@app.cell
def _():
    return


@app.cell(column=1)
def _():
    from collections import defaultdict
    from pathlib import Path
    from typing import Any

    import duckdb
    import pandas as pd
    import quak
    import srsly

    return Path, pd


@app.cell
def _():
    ENTITY = "ml-moe"
    PROJECT = "ft-scaling"
    RUNS_PER_PAGE = 500
    WANDB_RUNS_FILENAME = "wandb_runs"
    WANDB_HISTORY_FILENAME = "wandb_history"
    OUT_DIR = "/Users/daniellerothermel/drotherm/repos/dr_ingest/notebooks/"
    return OUT_DIR, WANDB_HISTORY_FILENAME, WANDB_RUNS_FILENAME


@app.cell
def _(drop_bad_dates, pd, runs_parquet):
    runs_df = pd.read_parquet(runs_parquet)
    runs_df = drop_bad_dates(runs_df, "created_at", "2025-08-21")
    print(runs_df.columns)
    runs_df["created_at"]

    return (runs_df,)


app._unparsable_cell(
    r"""
    def drop_bad_dates(df: pd.DataFrame, date_col: str, min_date: str) -> pd.DataFrame:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        min_date = pd.to_datetime(min_date)
        filtered_df = df[df[date_col] >= min_date]
        return filtered_df

    def split_then_norm_rid(rid: str) -> list[str]:
        parts = rid.split(\"-\")
        return [normalize_str(p) for p in parts]

    def extract_date_prefix(str_list: str) -> tuple[str, list[str]]:
    """,
    name="_",
)


@app.cell
def _(normalize_str, runs_df):
    # runs_df['rid2'] = runs_df['run_id'].apply(split_then_norm_rid)
    runs_df["rid2"] = runs_df["run_id"].apply(normalize_str)
    runs_df[["run_id", "rid2"]]
    return


@app.cell
def _():
    return


@app.cell(column=2)
def _():
    return


if __name__ == "__main__":
    app.run()
