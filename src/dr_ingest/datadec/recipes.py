from pydantic import BaseModel, Field

from dr_ingest.utils.display import add_marimo_display

type RecipeString = str
type NormalizedRecipeName = str


@add_marimo_display()
class DataDecideRecipeConfig(BaseModel):
    original_recipe_names: list[RecipeString] = Field(
        default_factory=lambda: [
            "Dolma1.7 (no Reddit)",
            "Dolma1.7 (no Flan)",
            "Dolma1.7 (no code)",
            "Dolma1.7 (no math, code)",
            "C4",
            "Falcon",
            "Falcon+CC",
            "Falcon+CC (QC 20%)",
            "Falcon+CC (QC Orig 10%)",
            "Falcon+CC (QC 10%)",
            "Falcon+CC (QC Tulu 10%)",
            "Dolma1.6++",
            "FineWeb-Pro",
            "FineWeb-Edu",
            "Dolma1.7",
            "DCLM-Baseline 25% / Dolma 75%",
            "DCLM-Baseline 50% / Dolma 50%",
            "DCLM-Baseline 75% / Dolma 25%",
            "DCLM-Baseline",
            "DCLM-Baseline (QC FW 10%)",
            "DCLM-Baseline (QC FW 3%)",
            "DCLM-Baseline (QC 10%)",
            "DCLM-Baseline (QC 20%)",
            "DCLM-Baseline (QC 7%, FW3)",
            "DCLM-Baseline (QC 7%, FW2)",
        ]
    )
    recipe_order: list[NormalizedRecipeName] = Field(
        default_factory=lambda: [
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
        ]
    )
    normalized_recipe_map: dict[RecipeString, NormalizedRecipeName] = Field(
        default_factory=lambda: {
            # ----------- Original Strings -----------
            "Dolma1.7 (no Reddit)": "dolma17_no_reddit",
            "Dolma1.7 (no Flan)": "dolma17_no_flan",
            "Dolma1.7 (no code)": "dolma17_no_code",
            "Dolma1.7 (no math, code)": "dolma17_no_math_code",
            "C4": "c4",
            "Falcon": "falcon",
            "Falcon+CC": "falcon_cc",
            "Falcon+CC (QC 20%)": "falcon_cc_qc_20",
            "Falcon+CC (QC Orig 10%)": "falcon_cc_qc_orig_10",
            "Falcon+CC (QC 10%)": "falcon_cc_qc_10",
            "Falcon+CC (QC Tulu 10%)": "falcon_cc_qc_tulu_10",
            "Dolma1.6++": "dolma16",
            "FineWeb-Pro": "fineweb_pro",
            "FineWeb-Edu": "fineweb_edu",
            "Dolma1.7": "dolma17",
            "DCLM-Baseline 25% / Dolma 75%": "dclm_baseline_25_dolma_75",
            "DCLM-Baseline 50% / Dolma 50%": "dclm_baseline_50_dolma_50",
            "DCLM-Baseline 75% / Dolma 25%": "dclm_baseline_75_dolma_25",
            "DCLM-Baseline": "dclm_baseline",
            "DCLM-Baseline (QC FW 10%)": "dclm_baseline_qc_fw_10",
            "DCLM-Baseline (QC FW 3%)": "dclm_baseline_qc_fw_3",
            "DCLM-Baseline (QC 10%)": "dclm_baseline_qc_10",
            "DCLM-Baseline (QC 20%)": "dclm_baseline_qc_20",
            "DCLM-Baseline (QC 7%, FW3)": "dclm_baseline_qc_7_fw3",
            "DCLM-Baseline (QC 7%, FW2)": "dclm_baseline_qc_7_fw2",
            # ----------- Scaling "Mix" Values -----------
            "DCLM-baseline": "dclm_baseline",
            "dclm_ft7percentile_fw3": "dclm_baseline_qc_7_fw3",
            "dclm_ft7percentile_fw2": "dclm_baseline_qc_7_fw2",
            "dclm_fw_top10": "dclm_baseline_qc_fw_10",
            "dclm_fw_top3": "dclm_baseline_qc_fw_3",
            "dolma_v1_6_and_sources_baseline": "dolma16",
            "dolma17_50p_dclm_baseline_50p": "dclm_baseline_50_dolma_50",
            "dolma17_25p_dclm_baseline_75p": "dclm_baseline_75_dolma_25",
            "dolma17_75p_dclm_baseline_25p": "dclm_baseline_25_dolma_75",
            "falcon_and_cc": "falcon_cc",
            "falcon_and_cc_eli5_oh_top10p": "falcon_cc_qc_10",
            "falcon_and_cc_eli5_oh_top20p": "falcon_cc_qc_20",
            "falcon_and_cc_og_eli5_oh_top10p": "falcon_cc_qc_orig_10",
            "falcon_and_cc_tulu_qc_top10": "falcon_cc_qc_tulu_10",
            "fineweb_edu_dedup": "fineweb_edu",
            "no_code": "dolma17_no_code",
            "no_flan": "dolma17_no_flan",
            "no_math_no_code": "dolma17_no_math_code",
            "no_reddit": "dolma17_no_reddit",
            "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p": (
                "dclm_baseline_qc_10"
            ),
            "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p": (
                "dclm_baseline_qc_20"
            ),
            "prox_fineweb_pro": "fineweb_pro",
        }
    )
