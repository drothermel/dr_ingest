"""Post-processing utilities for classified WandB runs."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from .config import load_defaults, load_recipe_mapping
from .constants import ALL_FT_TOKENS, DEFAULT_FULL_FT_EPOCHS
from .utils import convert_string_to_number, convert_timestamp


def extract_config_fields(
    runs_df: pd.DataFrame, run_ids: Iterable[str], field_mapping: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    """Extract selected configuration/summary fields for runs."""
    config_data: Dict[str, Dict[str, Any]] = {}
    for run_id in run_ids:
        run_row = runs_df[runs_df["run_id"] == run_id]
        if run_row.empty:
            continue
        try:
            config = json.loads(run_row.iloc[0]["config"])
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            config = {}
        for target_field, config_field in field_mapping.items():
            if config_field in config and config[config_field] is not None:
                config_data.setdefault(run_id, {})[target_field] = config[config_field]
        summary_payload = run_row.iloc[0].get("summary")
        if summary_payload and not pd.isna(summary_payload):
            try:
                summary = json.loads(summary_payload)
            except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                summary = None
            if summary and summary.get("total_tokens") is not None:
                config_data.setdefault(run_id, {})["num_finetuned_tokens_real"] = (
                    summary["total_tokens"]
                )
    return config_data


def apply_processing(
    dataframes: Dict[str, pd.DataFrame],
    defaults: Optional[Dict[str, Any]] = None,
    column_map: Optional[Dict[str, str]] = None,
    runs_df: Optional[pd.DataFrame] = None,
    history_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Normalise extracted run data across run types."""
    defaults_map = dict(defaults or load_defaults())
    column_map = column_map or {}
    recipe_mapping = load_recipe_mapping()

    processed: Dict[str, pd.DataFrame] = {}
    recipe_columns = ["comparison_model_recipe", "initial_checkpoint_recipe"]
    config_field_mapping = {
        "lr": "learning_rate",
        "seed": "seed",
        "num_finetune_epochs": "num_train_epochs",
    }

    for run_type, df in dataframes.items():
        processed_df = df.copy()

        for old_col, new_col in column_map.items():
            if old_col in processed_df.columns:
                processed_df = processed_df.rename(columns={old_col: new_col})

        for col, default_val in defaults_map.items():
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].fillna(default_val)

        for recipe_col in recipe_columns:
            if recipe_col in processed_df.columns:
                processed_df[recipe_col] = processed_df[recipe_col].map(
                    lambda x: recipe_mapping.get(x, x) if pd.notna(x) else x
                )

        if runs_df is not None and "run_id" in processed_df.columns:
            run_ids = processed_df["run_id"].tolist()
            config_data = extract_config_fields(runs_df, run_ids, config_field_mapping)
            for run_id, fields in config_data.items():
                run_idx = processed_df.index[processed_df["run_id"] == run_id]
                if run_idx.empty:
                    continue
                for field, value in fields.items():
                    if field == "num_finetuned_tokens_real":
                        processed_df.loc[run_idx[0], field] = value
                    elif field in processed_df.columns:
                        current_val = processed_df.loc[run_idx[0], field]
                        if pd.isna(current_val) or current_val == "N/A":
                            processed_df.loc[run_idx[0], field] = str(value)

        if "timestamp" in processed_df.columns:
            processed_df["timestamp"] = processed_df["timestamp"].apply(
                convert_timestamp
            )

        if "comparison_model_size" in processed_df.columns:
            processed_df["comparison_model_recipe"] = processed_df[
                "comparison_model_recipe"
            ].fillna("Dolma1.7")

        if run_type == "matched":
            if "comparison_metric" not in processed_df.columns:
                processed_df["comparison_metric"] = "pile"
            processed_df["comparison_metric"] = processed_df[
                "comparison_metric"
            ].fillna("pile")
            processed_df["comparison_metric"] = processed_df["comparison_metric"].map(
                lambda x: x + "_en-valppl" if x == "c4" else x + "-valppl"
            )

        for col in [
            "num_finetune_tokens",
            "num_finetune_tokens_per_epoch",
            "num_finetuned_tokens_real",
        ]:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].apply(convert_string_to_number)

        mask = (
            processed_df["run_id"].str.contains("_Ft_")
            if "run_id" in processed_df
            else False
        )
        if isinstance(mask, pd.Series) and mask.any():
            processed_df.loc[mask, "num_finetune_tokens_per_epoch"] = ALL_FT_TOKENS
            processed_df.loc[mask, "num_finetune_epochs"] = DEFAULT_FULL_FT_EPOCHS
            processed_df.loc[mask, "num_finetune_tokens"] = (
                DEFAULT_FULL_FT_EPOCHS * ALL_FT_TOKENS
            )
            processed_df.loc[mask, "num_finetuned_tokens_real"] = (
                DEFAULT_FULL_FT_EPOCHS * ALL_FT_TOKENS
            )

        if (
            "num_finetune_tokens_per_epoch" in processed_df.columns
            and "num_finetune_epochs" in processed_df.columns
        ):
            if "num_finetune_tokens" not in processed_df.columns:
                processed_df["num_finetune_tokens"] = None
            processed_df["num_finetune_epochs"] = pd.to_numeric(
                processed_df["num_finetune_epochs"], errors="coerce"
            )
            fill_mask = (
                processed_df["num_finetune_tokens_per_epoch"].notna()
                & processed_df["num_finetune_epochs"].notna()
                & processed_df["num_finetune_tokens"].isna()
            )
            processed_df.loc[fill_mask, "num_finetune_tokens"] = (
                processed_df.loc[fill_mask, "num_finetune_tokens_per_epoch"]
                * processed_df.loc[fill_mask, "num_finetune_epochs"]
            )

        if (
            "num_finetune_tokens" in processed_df.columns
            and "num_finetuned_tokens_real" in processed_df.columns
        ):
            mask = (
                processed_df["num_finetune_tokens"].notna()
                & processed_df["num_finetuned_tokens_real"].notna()
                & (processed_df["num_finetune_tokens"] != 0)
            )
            processed_df["abs_difference_ft_tokens_pct"] = None
            processed_df.loc[mask, "abs_difference_ft_tokens_pct"] = (
                (
                    processed_df.loc[mask, "num_finetune_tokens"]
                    - processed_df.loc[mask, "num_finetuned_tokens_real"]
                ).abs()
                / processed_df.loc[mask, "num_finetune_tokens"]
                * 100
            )

        processed[run_type] = processed_df

    return processed


__all__ = ["extract_config_fields", "apply_processing"]
