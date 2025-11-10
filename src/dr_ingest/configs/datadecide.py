from typing import Literal

from pydantic import BaseModel, Field, HttpUrl

from dr_ingest import HFLocation
from dr_ingest.utils.display import add_marimo_display


@add_marimo_display()
class DataDecideSourceConfig(BaseModel):
    google_drive_folder: HttpUrl = HttpUrl(
        "https://drive.google.com/drive/folders/1weYlEOlHrA_fzT2OsRa40uLc4EKTGz1D"
    )
    perplexity_metrics_csv: HttpUrl = HttpUrl(
        "https://github.com/allenai/DataDecide/blob/main/perplexity_metrics_by_group.csv"
    )

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
            repo_name="DataDecide-eval-scaling-laws",
            filepaths=[
                "data/scaling_law_fit-00000-of-00001.parquet",
            ],
        )
    )
    macro_avg_hf: HFLocation = Field(
        default_factory=lambda: HFLocation(
            org="allenai",
            repo_name="DataDecide-eval-macro-avg",
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

    ## Default Filenames
    results_filename: str = "train_results.parquet"

    ## Downloaded Col Names
    task_col: Literal["task"] = "task"
    step_col: Literal["step"] = "step"
    seed_col: Literal["seed"] = "seed"
    params_col: Literal["params"] = "params"

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

    ## Ordering Configs
    recipe_order: tuple[str, ...] = Field(
        default_factory=lambda: (
            "dolma17_no_reddit",
            "dolma17_no_flan",
            "dolma17_no_code",
            "dolma17_no_math_code",
            "c4",
            "falcon",
            "falcon_cc",
            "falcon_cc_qc_20",
            "falcon_cc_qc_orig_10",
            "falcon_cc_qc_10",
            "falcon_cc_qc_tulu_10",
            "dolma16",
            "fineweb_pro",
            "fineweb_edu",
            "dolma17",
            "dclm_baseline_25_dolma_75",
            "dclm_baseline_50_dolma_50",
            "dclm_baseline_75_dolma_25",
            "dclm_baseline",
            "dclm_baseline_qc_fw_10",
            "dclm_baseline_qc_fw_3",
            "dclm_baseline_qc_10",
            "dclm_baseline_qc_20",
            "dclm_baseline_qc_7_fw3",
            "dclm_baseline_qc_7_fw2",
        )
    )

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
