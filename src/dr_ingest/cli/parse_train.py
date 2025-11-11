from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from dr_ingest.configs import (
    ParsedSourceConfig,
    Paths,
)
from dr_ingest.datadec.datadecide import DataDecideSourceConfig
from dr_ingest.hf.io import (
    cached_download_tables_from_hf,
    get_tables_from_cache,
    upload_file_to_hf,
)
from dr_ingest.pipelines.dd_results import parse_train_df

app = typer.Typer()


def resolve_parsed_output_path(
    *,
    paths: Paths | None = None,
    parsed_config: ParsedSourceConfig | None = None,
) -> Path:
    paths = paths or Paths()
    parsed_cfg = parsed_config or ParsedSourceConfig()
    parsed_hf_loc = parsed_cfg.pretrain
    results_filename = parsed_hf_loc.get_the_single_filepath()
    output_path = Path(paths.data_cache_dir / results_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def validate_and_merge_tables(expected_paths: list[str]) -> pd.DataFrame:
    shard_dfs: list[pd.DataFrame] = []
    for path in expected_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing downloaded shard for {path}")
        shard_dfs.append(pd.read_parquet(path))
    return pd.concat(shard_dfs, ignore_index=True)


@app.command()
def download(
    force_download: bool = False,
    data_cache_dir: str | None = None,
) -> None:
    """Download raw Data Decide Results from HF to Local"""
    paths = Paths(data_cache_dir=data_cache_dir) if data_cache_dir else Paths()  # type: ignore
    cached_download_tables_from_hf(
        DataDecideSourceConfig().results_hf,
        local_dir=paths.data_cache_dir,
        force_download=force_download,
    )


@app.command()
def parse(
    data_cache_dir: str | None = None,
) -> None:
    """Parse already downloaded Data Decide Results"""
    paths = Paths(data_cache_dir=data_cache_dir) if data_cache_dir else Paths()  # type: ignore
    output_path = resolve_parsed_output_path(paths=paths)
    source_loc = DataDecideSourceConfig().results_hf
    source_df = pd.concat(
        get_tables_from_cache(source_loc, local_dir=paths.data_cache_dir),
        ignore_index=True,
    )
    print(">> Begin parsing, this will take 2min+")
    parsed_df = parse_train_df(source_df)
    parsed_df.to_parquet(output_path, index=False)
    print(f">> Wrote parsed train results to {output_path}")


@app.command()
def upload(
    data_cache_dir: str | None = None,
) -> None:
    """Upload parsed Data Decide Results from local to HF"""
    paths = Paths(data_cache_dir=data_cache_dir) if data_cache_dir else Paths()  # type: ignore
    local_parsed_path = resolve_parsed_output_path(paths=paths)
    if not local_parsed_path.exists():
        raise FileNotFoundError(
            f"Output file {local_parsed_path} not found; cannot upload-only."
        )
    parsed_pretrain_hf_loc = ParsedSourceConfig().pretrain
    print(">> Upload Only")
    print(f" - from: {local_parsed_path}")
    print(" - to: ")
    print(parsed_pretrain_hf_loc.model_dump_json(indent=4))
    upload_file_to_hf(local_parsed_path, parsed_pretrain_hf_loc)


@app.command()
def full_pipeline(
    force_download: bool = False,
    data_cache_dir: str | None = None,
) -> None:
    """Download, parse, parse and upload Data Decide results"""
    download(force_download, data_cache_dir)
    parse(data_cache_dir)
    upload(data_cache_dir)


if __name__ == "__main__":
    app()
