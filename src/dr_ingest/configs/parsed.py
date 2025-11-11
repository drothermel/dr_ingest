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
            local_path_include_org=False,
            local_path_include_repo=False,
        )
    )
    scaling_laws: HFLocation = Field(
        default_factory=lambda: HFLocation(
            org="drotherm",
            repo_name="dd_parsed",
            filepaths=[
                "macro_avg.parquet",
                "scaling_law_pred_one_step.parquet",
                "scaling_law_pred_two_step.parquet",
                "scaling_law_true.parquet",
            ],
            local_path_include_org=False,
            local_path_include_repo=False,
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
            local_path_include_org=False,
            local_path_include_repo=False,
        )
    )
