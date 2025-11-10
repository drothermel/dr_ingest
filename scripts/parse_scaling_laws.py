from __future__ import annotations

from pathlib import Path

import typer
from pydantic import BaseModel, Field
from pydantic.experimental.missing_sentinel import MISSING

from dr_ingest.configs import DataDecideSourceConfig, Paths
from dr_ingest.hf import HFLocation, download_tables_from_hf, upload_file_to_hf
from dr_ingest.pipelines.dd_scaling_laws import parse_scaling_law_dir

app = typer.Typer()


class ScalingLawOutputConfig(BaseModel):
    output_name_map: dict[str, str] = Field(
        default_factory=lambda: {
            "macro_avg_raw": "macro_avg.parquet",
            "scaling_law_pred_one_step_raw": "scaling_law_pred_one_step.parquet",
            "scaling_law_pred_two_step_raw": "scaling_law_pred_two_step.parquet",
            "scaling_law_true_raw": "scaling_law_true.parquet",
        }
    )
    hf_location: HFLocation = Field(
        default_factory=lambda: HFLocation(org="drotherm", repo_name="dd_parsed")
    )


def resolve_local_scaling_law_filepaths(
    *,
    paths: Paths | None = None,
    source_config: DataDecideSourceConfig | None = None,
) -> list[str]:
    paths = paths or Paths()
    source_cfg = source_config or DataDecideSourceConfig()

    macro_pathname = source_cfg.macro_avg_hf.get_the_single_filepath()
    fit_pathname = source_cfg.scaling_laws_hf.get_the_single_filepath()

    return [
        f"{paths.data_cache_dir}/{macro_pathname}",
        f"{paths.data_cache_dir}/{fit_pathname}",
    ]


def resolve_parsed_output_paths(
    *,
    paths: Paths | None = None,
    output_config: ScalingLawOutputConfig | None = None,
) -> dict[str, Path]:
    paths = paths or Paths()
    output_cfg = output_config or ScalingLawOutputConfig()

    output_paths = {}
    for key, filename in output_cfg.output_name_map.items():
        output_path = Path(paths.data_cache_dir / filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_paths[key] = output_path

    return output_paths


def validate_source_files(expected_paths: list[str]) -> None:
    for path in expected_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing downloaded file: {path}")


@app.command()
def download(
    force: bool = False,
    data_cache_dir: str | None = None,
) -> None:
    paths = Paths(data_cache_dir=data_cache_dir or MISSING)  # type: ignore
    dd_cfg = DataDecideSourceConfig()

    macro_hf_loc = dd_cfg.macro_avg_hf
    fit_hf_loc = dd_cfg.scaling_laws_hf

    macro_paths = download_tables_from_hf(
        macro_hf_loc,
        local_dir=paths.data_cache_dir,
        force_download=force,
    )
    fit_paths = download_tables_from_hf(
        fit_hf_loc,
        local_dir=paths.data_cache_dir,
        force_download=force,
    )

    print(f">> Downloaded macro avg: {macro_paths} to {paths.data_cache_dir}")
    print(f">> Downloaded scaling law fit: {fit_paths} to {paths.data_cache_dir}")


@app.command()
def parse(
    force: bool = False,
    data_cache_dir: str | None = None,
) -> None:
    paths = Paths(data_cache_dir=data_cache_dir or MISSING)  # type: ignore
    source_filepaths = resolve_local_scaling_law_filepaths(paths=paths)
    output_paths = resolve_parsed_output_paths(paths=paths)

    validate_source_files(source_filepaths)

    parsed_outputs = parse_scaling_law_dir(paths.data_cache_dir)

    for key, df in parsed_outputs.items():
        output_path = output_paths[key]
        if output_path.exists() and not force:
            raise FileExistsError(
                f"Output file {output_path} already exists. Use --force to overwrite."
            )
        df.write_parquet(output_path)
        print(f">> Wrote {output_path} ({df.height} rows, {df.width} columns)")


@app.command()
def upload(
    data_cache_dir: str | None = None,
) -> None:
    paths = Paths(data_cache_dir=data_cache_dir or MISSING)  # type: ignore
    output_cfg = ScalingLawOutputConfig()
    output_paths = resolve_parsed_output_paths(paths=paths, output_config=output_cfg)

    for output_path in output_paths.values():
        if not output_path.exists():
            raise FileNotFoundError(
                f"Output file {output_path} not found; cannot upload."
            )

    for key, output_path in output_paths.items():
        filename = output_cfg.output_name_map[key]
        print(
            f">> Uploading {output_path} to {output_cfg.hf_location.repo_id}:{filename}"
        )
        upload_file_to_hf(
            local_path=output_path,
            hf_loc=output_cfg.hf_location,
            path_in_repo=filename,
        )


@app.command()
def full_pipeline(
    force: bool = False,
    data_cache_dir: str | None = None,
) -> None:
    download(force, data_cache_dir)
    parse(force, data_cache_dir)
    upload(data_cache_dir)


if __name__ == "__main__":
    app()
