from __future__ import annotations

from pathlib import Path

import typer

from dr_ingest.configs import ParsedSourceConfig, Paths
from dr_ingest.datadec.datadecide import DataDecideSourceConfig
from dr_ingest.hf.io import cached_download_tables_from_hf, upload_file_to_hf
from dr_ingest.pipelines.dd_scaling_laws import parse_scaling_law_dir

app = typer.Typer()


@app.command()
def download(
    force_download: bool = False,
    data_cache_dir: str | None = None,
) -> None:
    paths = Paths(data_cache_dir=data_cache_dir) if data_cache_dir else Paths()  # type: ignore
    dd_cfg = DataDecideSourceConfig()
    cached_download_tables_from_hf(
        dd_cfg.macro_avg_hf,
        local_dir=paths.data_cache_dir,
        force_download=force_download,
    )
    cached_download_tables_from_hf(
        dd_cfg.scaling_laws_hf,
        local_dir=paths.data_cache_dir,
        force_download=force_download,
    )


@app.command()
def parse(
    data_cache_dir: str | None = None,
) -> None:
    paths = Paths(data_cache_dir=data_cache_dir) if data_cache_dir else Paths()  # type: ignore
    parsed_outputs = parse_scaling_law_dir(paths.data_cache_dir)
    for filepath, df in parsed_outputs.items():
        out_path = f"{paths.data_cache_dir}/{filepath}"
        df.write_parquet(out_path)
        print(f">> Wrote {out_path}")


@app.command()
def upload(
    data_cache_dir: str | None = None,
) -> None:
    paths = Paths(data_cache_dir=data_cache_dir) if data_cache_dir else Paths()  # type: ignore
    hf_loc = ParsedSourceConfig().scaling_laws
    local_paths = hf_loc.resolve_filepaths(local_dir=paths.data_cache_dir)
    remote_paths = hf_loc.resolve_filepaths()

    # Verify all before starting to upload
    for path in local_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found, cannot upload")

    for local_path, remote_path in zip(local_paths, remote_paths, strict=True):
        print(f">> Uploading {local_path} to {hf_loc.repo_id}: {remote_path}")
        upload_file_to_hf(
            local_path=local_path,
            hf_loc=hf_loc.model_copy(update={"filepaths": [remote_path]}),
        )
        print()


@app.command()
def full_pipeline(
    force: bool = False,
    data_cache_dir: str | None = None,
) -> None:
    download(force, data_cache_dir)
    parse(data_cache_dir)
    upload(data_cache_dir)


if __name__ == "__main__":
    app()
