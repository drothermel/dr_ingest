from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, computed_field

from dr_ingest.hf.location import HFLocation
from dr_ingest.utils.display import add_marimo_display

from .recipes import DataDecideRecipeConfig


@add_marimo_display()
class DataDecideSourceConfig(BaseModel):
    google_drive_folder: HttpUrl = HttpUrl(
        "https://drive.google.com/drive/folders/1weYlEOlHrA_fzT2OsRa40uLc4EKTGz1D"
    )
    perplexity_metrics_csv: HttpUrl = HttpUrl(
        "https://github.com/allenai/DataDecide/blob/main/perplexity_metrics_by_group.csv"
    )
    local_path_include_org: bool = True
    local_path_include_repo: bool = True

    results_hf: HFLocation = Field(
        default_factory=lambda: HFLocation(
            org="allenai",
            repo_name="DataDecide-eval-results",
            filepaths=[
                "data/train-00000-of-00004.parquet",
                "data/train-00001-of-00004.parquet",
                "data/train-00002-of-00004.parquet",
                "data/train-00003-of-00004.parquet",
            ],
        )
    )
    scaling_laws_hf: HFLocation = Field(
        default_factory=lambda: HFLocation(
            org="allenai",
            repo_name="DataDecide-eval-results",
            filepaths=[
                "data/scaling_law_fit-00000-of-00001.parquet",
            ],
        )
    )
    macro_avg_hf: HFLocation = Field(
        default_factory=lambda: HFLocation(
            org="allenai",
            repo_name="DataDecide-eval-results",
            filepaths=[
                "data/macro_avg-00000-of-00001.parquet",
            ],
        )
    )
    instances_hf: HFLocation = Field(
        default_factory=lambda: HFLocation(
            org="allenai",
            repo_name="DataDecide-eval-instances",
        )
    )


@add_marimo_display()
class DataDecideConfig(BaseModel):
    source_config: DataDecideSourceConfig = Field(
        default_factory=DataDecideSourceConfig
    )
    recipe_config: DataDecideRecipeConfig = Field(
        default_factory=DataDecideRecipeConfig
    )

    ## Downloaded Col Names
    task_col: Literal["task"] = "task"
    step_col: Literal["step"] = "step"
    seed_col: Literal["seed"] = "seed"
    params_col: Literal["params"] = "params"
    acc_baseline_col: Literal["acc_baseline"] = "acc_baseline"
    prob_baseline_col: Literal["prob_baseline"] = "prob_baseline"

    ## Task Group Mapping
    task_group_col: Literal["task_group"] = "task_group"
    task_group_map: dict[str, str] = Field(
        default_factory=lambda: {
            "mmlu_": "mmlu",
        }
    )

    ## Baseline Values
    prob_baseline: float = 0.0
    mmlu_baseline: float = 0.25
    task_baselines: dict[str, float] = Field(
        default_factory=lambda: {
            "winogrande": 0.5,
            "socialiqa": 0.3333333333,
            "piqa": 0.5,
            "openbookqa": 0.25,
            "hellaswag": 0.25,
            "csqa": 0.2,
            "boolq": 0.5,
            "arc_easy": 0.25,
            "arc_challenge": 0.2,
        }
    )

    ## Metric Column Remapping (raw -> normalized)
    metric_column_renames: dict[str, str] = Field(
        default_factory=lambda: {
            "acc_raw": "metrics_acc_raw",
            "acc_per_token": "metrics_acc_per_token",
            "acc_per_char": "metrics_acc_per_char",
            "acc_per_byte": "metrics_acc_per_byte",
            "acc_uncond": "metrics_acc_uncond",
            "sum_logits_corr": "metrics_sum_logits_correct_raw",
            "logits_per_token_corr": "metrics_sum_logits_correct_per_token",
            "logits_per_char_corr": "metrics_sum_logits_correct_per_char",
            "logits_per_byte_corr": "metrics_sum_logits_correct_per_byte",
            "correct_prob": "metrics_correct_prob_raw",
            "correct_prob_per_token": "metrics_correct_prob_per_token",
            "correct_prob_per_char": "metrics_correct_prob_per_char",
            "margin": "metrics_margin_raw",
            "margin_per_token": "metrics_margin_per_token",
            "margin_per_char": "metrics_margin_per_char",
            "total_prob": "metrics_total_prob_raw",
            "total_prob_per_token": "metrics_total_prob_per_token",
            "total_prob_per_char": "metrics_total_prob_per_char",
            "uncond_correct_prob": "metrics_uncond_correct_prob_raw",
            "uncond_correct_prob_per_token": "metrics_uncond_correct_prob_per_token",
            "uncond_correct_prob_per_char": "metrics_uncond_correct_prob_per_char",
            "uncond_total_prob": "metrics_uncond_total_prob_raw",
            "norm_correct_prob": "metrics_norm_correct_prob_raw",
            "norm_correct_prob_per_token": "metrics_norm_correct_prob_per_token",
            "norm_correct_prob_per_char": "metrics_norm_correct_prob_per_char",
            "bits_per_byte_corr": "metrics_bits_per_byte_correct",
            "primary_metric": "metrics_primary_metric",
        }
    )

    metrics_cols_to_drop: list[str] = Field(
        default_factory=lambda: [
            "no_answer",  # Always null or zero
            # Not meaningful in aggregate
            "predicted_index_raw",
            "predicted_index_per_token",
            "predicted_index_per_char",
            "predicted_index_per_byte",
            "predicted_index_uncond",
            "correct_choice",
        ]
    )

    ## Ordering Configs
    param_order: tuple[str, ...] = Field(
        default_factory=lambda: (
            "4M",
            "6M",
            "8M",
            "10M",
            "14M",
            "16M",
            "20M",
            "60M",
            "90M",
            "150M",
            "300M",
            "530M",
            "750M",
            "1B",
        )
    )

    seed_order: tuple[str, ...] = Field(
        default_factory=lambda: (
            "default",
            "large_aux_2",
            "large_aux_3",
            "small_aux_2",
            "small_aux_3",
        )
    )

    @computed_field
    @property
    def recipe_order(self) -> list[str]:
        return self.recipe_config.recipe_order
