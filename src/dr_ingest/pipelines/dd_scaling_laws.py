from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl

from dr_ingest.datadec.datadecide import DataDecideSourceConfig
from dr_ingest.datadec.recipes import DataDecideRecipeConfig

__all__ = ["parse_scaling_law_dir"]


def parse_scaling_law_dir(
    source_dir: Path,
    recipe_config: DataDecideRecipeConfig | None = None,
) -> dict[str, pl.DataFrame]:
    """Load and parse scaling-law parquet files from a directory.

    Parameters
    ----------
    source_dir:
        Directory containing the macro-average and scaling-law fit parquet files.

    Returns
    -------
    dict[str, pl.DataFrame]
        Dictionary including the macro-average dataframe and parsed scaling-law outputs.
    """
    recipe_cfg = recipe_config or DataDecideRecipeConfig()
    source_cfg = DataDecideSourceConfig()
    macro_path = source_cfg.macro_avg_hf.get_the_single_filepath(local_dir=source_dir)
    fit_path = source_cfg.scaling_laws_hf.get_the_single_filepath(local_dir=source_dir)

    if missing := [path for path in (macro_path, fit_path) if not Path(path).exists()]:
        raise FileNotFoundError(f"Missing scaling-law parquet files: {missing}")

    macro_df = pl.read_parquet(macro_path)
    scaling_law_df = pl.read_parquet(fit_path)
    output_paths_to_dfs = parse_sl_results(scaling_law_df, recipe_cfg)
    output_paths_to_dfs["macro_avg.parquet"] = macro_df
    return output_paths_to_dfs


def parse_sl_results(
    df: pl.DataFrame, cfg: DataDecideRecipeConfig
) -> dict[str, pl.DataFrame]:
    """Split scaling-law results into the three downstream datasets."""

    sl_w_cfg = _prep_sl_cfg(df, cfg)
    sl_one_step_rows = [row for row in sl_w_cfg if row["fit_config"]["one_step"]]
    sl_two_step_rows = [row for row in sl_w_cfg if not row["fit_config"]["one_step"]]
    sl_one_step_df = _extract_one_step_preds(sl_one_step_rows)
    sl_two_step_df = _extract_two_step_preds(sl_two_step_rows)
    sl_true_df = _extract_true_metrics(sl_two_step_rows)
    return {
        "scaling_law_pred_one_step.parquet": pl.DataFrame(sl_one_step_df),
        "scaling_law_pred_two_step.parquet": pl.DataFrame(sl_two_step_df),
        "scaling_law_true.parquet": pl.DataFrame(sl_true_df),
    }


def _prep_sl_cfg(df: pl.DataFrame, cfg: DataDecideRecipeConfig) -> list[dict[str, Any]]:
    """Attach parsed config information to each scaling-law row."""

    col_list = df.to_dicts()
    for mapping in col_list:
        setup_val = mapping["setup"]
        mapping["fit_config"] = {
            "name": setup_val,
            "one_step": _get_sl_one_step_bool(setup_val),
            "params": _get_sl_num_fit_params(setup_val),
            "filtering": _get_sl_filter(setup_val),
            "helper_point": _get_sl_helper_point(setup_val),
            "heldout": _get_heldout(setup_val),
        }
        mapping["recipe"] = cfg.normalized_recipe_map.get(
            mapping["mix"],
            mapping["mix"],
        )
        del mapping["mix"], mapping["setup"]
    return col_list


def _extract_true_metrics(col_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    true_loss = defaultdict(dict)
    true_metrics = defaultdict(dict)
    configs = {}
    for mapping in col_list:
        eid = (mapping["task"], mapping["recipe"], mapping["fit_config"]["name"])
        true_loss[eid][mapping["metric"]] = mapping["step_1_y"]
        true_metrics[eid][mapping["metric"]] = mapping["step_2_y"]
        configs[eid] = mapping["fit_config"]

    output: list[dict[str, Any]] = []
    for idx, (eid, cfg) in enumerate(configs.items()):
        if cfg["name"] != "3_param-default":
            continue
        output.append(
            {
                "id": idx,
                "task": eid[0],
                "recipe": eid[1],
                "task_losses": _extract_metric_struct(true_loss[eid]),
                "task_metrics": _extract_metric_struct(true_metrics[eid]),
            }
        )
    return output


def _extract_two_step_preds(col_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    loss_preds = defaultdict(dict)
    loss_to_metric_preds = defaultdict(dict)
    metric_preds = defaultdict(dict)
    configs = {}
    for mapping in col_list:
        eid = (mapping["task"], mapping["recipe"], mapping["fit_config"]["name"])
        loss_preds[eid][mapping["metric"]] = mapping["step_1_pred"]
        loss_to_metric_preds[eid][mapping["metric"]] = mapping["step_2_pred"]
        metric_preds[eid][mapping["metric"]] = mapping["stacked_pred"]
        configs[eid] = mapping["fit_config"]

    output: list[dict[str, Any]] = []
    for idx, (eid, cfg) in enumerate(configs.items()):
        output.append(
            {
                "id": idx,
                "task": eid[0],
                "recipe": eid[1],
                "fit_config": cfg,
                "pred_task_losses": _extract_metric_struct(loss_preds[eid]),
                "pred_task_loss_to_metrics": _extract_metric_struct(
                    loss_to_metric_preds[eid]
                ),
                "pred_task_metrics": _extract_metric_struct(metric_preds[eid]),
            }
        )
    return output


def _extract_one_step_preds(col_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = defaultdict(dict)
    configs = {}

    for mapping in col_list:
        eid = (mapping["task"], mapping["recipe"], mapping["fit_config"]["name"])
        metrics[eid][mapping["metric"]] = mapping["stacked_pred"]
        configs[eid] = mapping["fit_config"]

    output = []
    for idx, (eid, mets) in enumerate(metrics.items()):
        output.append(
            {
                "id": idx,
                "task": eid[0],
                "recipe": eid[1],
                "fit_config": configs[eid],
                "pred_task_metrics": _extract_metric_struct(mets),
            }
        )
    return output


def _extract_metric_struct(metrics_dict: dict[str, Any]) -> dict[str, Any]:
    """Collect the raw/per-token/per-char variants for a metric family."""

    return {
        "acc_raw": metrics_dict.get("acc_raw"),
        "acc_per_char": metrics_dict.get("acc_per_char"),
        "acc_per_token": metrics_dict.get("acc_per_token"),
        "margin_raw": metrics_dict.get("margin"),
        "margin_per_char": metrics_dict.get("margin_per_char"),
        "margin_per_token": metrics_dict.get("margin_per_token"),
        "norm_correct_prob_raw": metrics_dict.get("norm_correct_prob"),
        "norm_correct_prob_per_char": metrics_dict.get("norm_correct_prob_per_char"),
        "norm_correct_prob_per_token": metrics_dict.get("norm_correct_prob_per_token"),
        "total_prob_raw": metrics_dict.get("total_prob"),
        "total_prob_per_char": metrics_dict.get("total_prob_per_char"),
        "total_prob_per_token": metrics_dict.get("total_prob_per_token"),
        "correct_prob_raw": metrics_dict.get("correct_prob"),
        "correct_prob_per_char": metrics_dict.get("correct_prob_per_char"),
        "correct_prob_per_token": metrics_dict.get("correct_prob_per_token"),
    }


def _get_sl_num_fit_params(value: str) -> str:
    fp_str = value[0]
    return fp_str[0]


def _get_sl_one_step_bool(value: str) -> bool:
    return "1_step" in value


def _get_sl_filter(value: str) -> str:
    if "step2=0.5" in value:
        return "50_Percent"
    return "None"


def _get_sl_helper_point(value: str) -> bool:
    return "helper_points" in value


def _get_heldout(value: str) -> list[str]:
    heldout_list = ["1B"]
    vsplit = value.split("-")
    for item in vsplit:
        if "no_" in item:
            heldout_list.extend(item.split("no_"))
    return [h.strip("_") for h in heldout_list if h != ""]
