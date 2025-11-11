from pydantic import BaseModel, Field

from dr_ingest.hf.location import HFLocation
from dr_ingest.utils import add_marimo_display


@add_marimo_display()
class ParsedSourceConfig(BaseModel):
    pretrain: HFLocation = Field(
        default_factory=lambda: HFLocation(
            org="drotherm",
            repo_name="dd_parsed",
            filepaths=[
                "train_results.parquet",
            ],
        )
    )
    scaling_laws: HFLocation = Field(
        default_factory=lambda: HFLocation(
            org="drotherm",
            repo_name="dd_parsed",
            filepaths=[
                "scaling_law_fit.parquet",
            ],
        )
    )
    wandb: HFLocation = Field(
        default_factory=lambda: HFLocation(
            org="drotherm",
            repo_name="dd_parsed",
            filepaths=[
                "wandb_history.parquet",
                "wandb_runs_config.parquet",
                "wandb_runs_summary.parquet",
                "wandb_runs_sweep_info.parquet",
                "wandb_runs_system_attrs.parquet",
                "wandb_runs_system_metrics.parquet",
                "wandb_runs_wandb_metadata.parquet",
            ],
        )
    )
