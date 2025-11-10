from collections import defaultdict
from typing import Any

from dr_ingest.parse import parse_sl_setup_to_config

SCALING_LAW_MACRO_FILENAME = "macro_avg-00000-of-00001.parquet"
SCALING_LAW_FIT_FILENAME = "scaling_law_fit-00000-of-00001.parquet"
SCALING_LAW_FILENAMES = [SCALING_LAW_MACRO_FILENAME, SCALING_LAW_FIT_FILENAME]


def extract_metric_struct(metrics_dict: dict[str, Any]) -> dict[str, Any]:
    """Collect the raw/per-token/per-char variants for a metric family."""

    return {
        "accuracy": {
            "raw": metrics_dict.get("acc_raw"),
            "per_char": metrics_dict.get("acc_per_char"),
            "per_token": metrics_dict.get("acc_per_token"),
        },
        "margin": {
            "raw": metrics_dict.get("margin"),
            "per_char": metrics_dict.get("margin_per_char"),
            "per_token": metrics_dict.get("margin_per_token"),
        },
        "norm_correct_prob": {
            "raw": metrics_dict.get("norm_correct_prob"),
            "per_char": metrics_dict.get("norm_correct_prob_per_char"),
            "per_token": metrics_dict.get("norm_correct_prob_per_token"),
        },
        "total_prob": {
            "raw": metrics_dict.get("total_prob"),
            "per_char": metrics_dict.get("total_prob_per_char"),
            "per_token": metrics_dict.get("total_prob_per_token"),
        },
        "correct_prob": {
            "raw": metrics_dict.get("correct_prob"),
            "per_char": metrics_dict.get("correct_prob_per_char"),
            "per_token": metrics_dict.get("correct_prob_per_token"),
        },
    }


def extract_true_metrics(col_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    true_loss: defaultdict[tuple[Any, Any, Any], dict[str, Any]] = defaultdict(dict)
    true_metrics: defaultdict[tuple[Any, Any, Any], dict[str, Any]] = defaultdict(dict)
    configs: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
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
                "task_losses": extract_metric_struct(true_loss[eid]),
                "task_metrics": extract_metric_struct(true_metrics[eid]),
            }
        )
    return output


def parse_sl_results(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Split scaling-law results into the three downstream datasets."""

    sl_w_cfg = prep_sl_cfg(df)
    sl_one_step_rows = [row for row in sl_w_cfg if row["fit_config"]["one_step"]]
    sl_two_step_rows = [row for row in sl_w_cfg if not row["fit_config"]["one_step"]]
    sl_one_step_df = extract_one_step_preds(sl_one_step_rows)
    sl_two_step_df = extract_two_step_preds(sl_two_step_rows)
    sl_true_df = extract_true_metrics(sl_two_step_rows)
    return {
        "scaling_law_pred_one_step_raw": pl.DataFrame(sl_one_step_df),
        "scaling_law_pred_two_step_raw": pl.DataFrame(sl_two_step_df),
        "scaling_law_true_raw": pl.DataFrame(sl_true_df),
    }


def prep_sl_cfg(df: pl.DataFrame) -> list[dict[str, Any]]:
    """Attach parsed config information to each scaling-law row."""

    col_list = df.to_dicts()
    for mapping in col_list:
        mapping["fit_config"] = parse_sl_setup_to_config(mapping["setup"])
        mapping["recipe"] = normalize_ds_str(mapping["mix"])
        del mapping["mix"], mapping["setup"]
    return col_list


def extract_two_step_preds(col_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    loss_preds: defaultdict[tuple[Any, Any, Any], dict[str, Any]] = defaultdict(dict)
    loss_to_metric_preds: defaultdict[tuple[Any, Any, Any], dict[str, Any]] = (
        defaultdict(dict)
    )
    metric_preds: defaultdict[tuple[Any, Any, Any], dict[str, Any]] = defaultdict(dict)
    configs: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
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
                "pred_task_losses": extract_metric_struct(loss_preds[eid]),
                "pred_task_loss_to_metrics": extract_metric_struct(
                    loss_to_metric_preds[eid]
                ),
                "pred_task_metrics": extract_metric_struct(metric_preds[eid]),
            }
        )
    return output


def extract_one_step_preds(col_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics: defaultdict[tuple[Any, Any, Any], dict[str, Any]] = defaultdict(dict)
    configs: dict[tuple[Any, Any, Any], dict[str, Any]] = {}

    for mapping in col_list:
        eid = (mapping["task"], mapping["recipe"], mapping["fit_config"]["name"])
        metrics[eid][mapping["metric"]] = mapping["stacked_pred"]
        configs[eid] = mapping["fit_config"]

    output: list[dict[str, Any]] = []
    for idx, (eid, mets) in enumerate(metrics.items()):
        output.append(
            {
                "id": idx,
                "task": eid[0],
                "recipe": eid[1],
                "fit_config": configs[eid],
                "pred_task_metrics": extract_metric_struct(mets),
            }
        )
    return output


def parse_scaling_law_dir(source_dir: Path) -> dict[str, pl.DataFrame]:
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

    macro_path = (source_dir / SCALING_LAW_MACRO_FILENAME).resolve()
    fit_path = (source_dir / SCALING_LAW_FIT_FILENAME).resolve()

    missing: list[Path] = [path for path in (macro_path, fit_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing scaling-law parquet files: {missing}")

    macro_df = pl.read_parquet(macro_path)
    scaling_law_df = pl.read_parquet(fit_path)
    outputs = parse_sl_results(scaling_law_df)
    outputs["macro_avg_raw"] = macro_df
    return outputs
