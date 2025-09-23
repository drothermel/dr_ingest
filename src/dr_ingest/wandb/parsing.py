from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import catalogue
import pandas as pd
from confection import Config
from memo import memlist

ALL_FT_TOKENS = 665_127_434
DEFAULT_FULL_FT_EPOCHS = 2

TIMESTAMP_6 = r"(?P<timestamp>\d{6}-\d{6})"
TIMESTAMP_8 = r"(?P<timestamp>\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})"
EXP_NAME = r"(?P<exp_name>[\w_]+)"
COMPARISON_MODEL_SIZE = r"(?P<comparison_model_size>\d+[MB])"
COMPARISON_METRIC = r"(?P<comparison_metric>[\w_]+)"
INITIAL_CHECKPOINT_RECIPE = r"(?P<initial_checkpoint_recipe>[\w_]+)"
INITIAL_CHECKPOINT_RECIPE_DASH = r"(?P<initial_checkpoint_recipe>[\w-]+)"
INITIAL_CHECKPOINT_SIZE = r"(?P<initial_checkpoint_size>\d+[MB])"
INITIAL_CHECKPOINT_STEPS = r"(?P<initial_checkpoint_steps>\d+)"
INITIAL_CHECKPOINT_STEPS_WORD = r"(?P<initial_checkpoint_steps>\w+)"
SEED = r"(?P<seed>\d+)"
LEARNING_RATE = r"(?P<lr>[0-9.e-]+)"
LEARNING_RATE_1 = r"(?P<lr1>[0-9.e-]+)"
LEARNING_RATE_2 = r"(?P<lr2>[0-9.e-]+)"
FINETUNE_TOKENS_EPOCHS_8 = r"(?P<num_finetune_tokens_per_epoch>\d+[MB])tx(?P<num_finetune_epochs>\d+)"
FINETUNE_TOKENS_EPOCHS_6 = r"(?P<num_finetune_tokens_per_epoch>\d+[MG])tx(?P<num_finetune_epochs>\d+)"
FINETUNE_TOKENS_8 = r"(?P<num_finetune_tokens>\d+[MB])"
FINETUNE_TOKENS_GT = r"(?P<num_finetune_tokens>\d+[MG]t)"
FINETUNE_TOKENS_SIMPLE = r"(?P<num_finetune_tokens>\d+)"
REDUCE_LOSS = r"(?P<reduce_loss>\w+)"

TIMESTAMP_6_EXP_NAME = rf"{TIMESTAMP_6}_{EXP_NAME}"
TIMESTAMP_8_EXP_NAME = rf"{TIMESTAMP_8}_{EXP_NAME}"
LR_SUFFIX = rf"_lr={LEARNING_RATE}"
LEARNING_RATE_FLAG = rf"_--learning_rate={LEARNING_RATE}"
LEARNING_RATE_EQUAL = rf"_learning_rate={LEARNING_RATE}"
FINETUNE_FT = "_finetune_Ft"
FINETUNE_TOKENS_6 = rf"_finetune_{FINETUNE_TOKENS_EPOCHS_6}"
DD_BLOCK_STEPS_WORD = rf"DD-{INITIAL_CHECKPOINT_RECIPE}-{INITIAL_CHECKPOINT_SIZE}"
DD_BLOCK_FULL = rf"DD-{INITIAL_CHECKPOINT_RECIPE}-{INITIAL_CHECKPOINT_SIZE}-{INITIAL_CHECKPOINT_STEPS}-{SEED}"
DD_COMPARISON_6 = rf"DD-[\w-]+-{COMPARISON_MODEL_SIZE}"
MATCHED_PREFIX_6 = rf"{TIMESTAMP_6_EXP_NAME}_{COMPARISON_MODEL_SIZE}"
MATCHED_PREFIX_WITH_METRIC = rf"{TIMESTAMP_6_EXP_NAME}_{COMPARISON_MODEL_SIZE}_{COMPARISON_METRIC}"

MIN_VALID_RUN_ID_SEGMENTS = 2
EXPECTED_DATE_OR_TIME_RAW_LEN = 6
EXPECTED_DATETIME_RAW_LEN = 2 * EXPECTED_DATE_OR_TIME_RAW_LEN + 1
ALT_COMPARISON_MODEL_RECIPE_STR = "c4"
ALT_COMPARISON_MODEL_RECIPE = "C4"
FINETUNE_PATTERN = r"(\d+M)tx(\d+)"
DD_PATTERN = r"DD-([^-]+)-(\d+M)-(\d+)-(\d+)"

CONFIG_PATH = Path(__file__).resolve().parent / "config.cfg"
_config = Config().from_disk(CONFIG_PATH)
DEFAULTS: Dict[str, str] = dict(_config["defaults"])  # type: ignore[arg-type]
RECIPE_MAPPING: Dict[str, str] = dict(_config["recipe_mapping"])  # type: ignore[arg-type]

pattern_registry = catalogue.create("dr_ingest", "wandb", "patterns")


@dataclass(frozen=True)
class PatternSpec:
    name: str
    run_type: str
    regex: re.Pattern[str]


PATTERN_SPECS: List[PatternSpec] = []


def _register_pattern(name: str, run_type: str, pattern: str) -> None:
    spec = PatternSpec(name=name, run_type=run_type, regex=re.compile(pattern))
    PATTERN_SPECS.append(spec)

    @pattern_registry.register(name)
    def factory() -> PatternSpec:  # pragma: no cover - registry hook
        return spec


_register_pattern(
    "FT1_PATTERN",
    "simple_ft_vary_tokens",
    rf"^{TIMESTAMP_8_EXP_NAME}_{DD_BLOCK_STEPS_WORD}_{INITIAL_CHECKPOINT_STEPS_WORD}_{FINETUNE_TOKENS_EPOCHS_8}{LEARNING_RATE_FLAG}$",
)
_register_pattern(
    "FT3_PATTERN",
    "simple_ft_vary_tokens",
    rf"^{TIMESTAMP_8_EXP_NAME}_{DD_BLOCK_STEPS_WORD}_{INITIAL_CHECKPOINT_STEPS_WORD}_{FINETUNE_TOKENS_8}_toks{LEARNING_RATE_FLAG}$",
)
_register_pattern(
    "FT4_PATTERN",
    "simple_ft",
    rf"^{TIMESTAMP_8_EXP_NAME}_{DD_BLOCK_STEPS_WORD}_Ft{LEARNING_RATE_FLAG}$",
)
_register_pattern(
    "FT5_PATTERN",
    "simple_ft",
    rf"^{TIMESTAMP_6_EXP_NAME}_DD-{INITIAL_CHECKPOINT_RECIPE_DASH}-{INITIAL_CHECKPOINT_SIZE}_Ft{LEARNING_RATE_EQUAL}$",
)
_register_pattern(
    "FT6_PATTERN",
    "simple_ft_vary_tokens",
    rf"^{TIMESTAMP_6_EXP_NAME}_{FINETUNE_TOKENS_EPOCHS_8}_{DD_BLOCK_FULL}{LR_SUFFIX}$",
)
_register_pattern(
    "FT7_PATTERN",
    "simple_ft",
    rf"^{TIMESTAMP_6_EXP_NAME}_Ft_{DD_BLOCK_FULL}{LR_SUFFIX}$",
)
_register_pattern(
    "MATCHED6_PATTERN",
    "matched",
    rf"^{TIMESTAMP_6_EXP_NAME}_{DD_COMPARISON_6}_Ft_{DD_BLOCK_FULL}{LR_SUFFIX}$",
)
_register_pattern(
    "MATCHED7_PATTERN",
    "matched",
    rf"^{TIMESTAMP_8_EXP_NAME}_{DD_COMPARISON_6}_Ft_{DD_BLOCK_FULL}{LR_SUFFIX}$",
)
_register_pattern(
    "REDUCE_LOSS_PATTERN",
    "reduce_type",
    rf"^{TIMESTAMP_8_EXP_NAME}_{DD_BLOCK_STEPS_WORD}_{INITIAL_CHECKPOINT_STEPS_WORD}_default_--max_train_samples={FINETUNE_TOKENS_SIMPLE}_--reduce_loss={REDUCE_LOSS}$",
)
_register_pattern(
    "DPO1_PATTERN",
    "dpo",
    rf"^{TIMESTAMP_8_EXP_NAME}_{DD_BLOCK_STEPS_WORD}_{INITIAL_CHECKPOINT_STEPS_WORD}_default$",
)
_register_pattern(
    "DPO2_PATTERN",
    "dpo",
    rf"^{TIMESTAMP_8_EXP_NAME}_dd__{INITIAL_CHECKPOINT_RECIPE}-{INITIAL_CHECKPOINT_SIZE}__{INITIAL_CHECKPOINT_STEPS_WORD}__{FINETUNE_TOKENS_GT}_lr={LEARNING_RATE_1}_default_--learning_rate={LEARNING_RATE_2}$",
)
_register_pattern(
    "MATCHED1_PATTERN",
    "matched",
    rf"^{MATCHED_PREFIX_WITH_METRIC}{FINETUNE_TOKENS_6}_{DD_BLOCK_FULL}$",
)
_register_pattern(
    "MATCHED2_PATTERN",
    "matched",
    rf"^{MATCHED_PREFIX_WITH_METRIC}{FINETUNE_FT}_{DD_BLOCK_FULL}$",
)
_register_pattern(
    "MATCHED3_PATTERN",
    "matched",
    rf"^{MATCHED_PREFIX_6}{FINETUNE_FT}_{DD_BLOCK_FULL}{LR_SUFFIX}$",
)
_register_pattern(
    "MATCHED4_PATTERN",
    "matched",
    rf"^{MATCHED_PREFIX_6}{FINETUNE_TOKENS_6}_{DD_BLOCK_FULL}$",
)
_register_pattern(
    "MATCHED5_PATTERN",
    "matched",
    rf"^{MATCHED_PREFIX_6}{FINETUNE_FT}_{DD_BLOCK_FULL}$",
)
_register_pattern(
    "MATCHED8_PATTERN",
    "matched",
    rf"^{MATCHED_PREFIX_6}{FINETUNE_TOKENS_6}_{DD_BLOCK_FULL}{LR_SUFFIX}$",
)

CLASSIFICATION_LOG: List[Dict[str, Any]] = []
_record_classification = memlist(data=CLASSIFICATION_LOG)


@_record_classification
def _log_event(**kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - memo hook
    return kwargs


def classify_run_id(run_id: str) -> Tuple[str, Dict[str, str | None]]:
    run_type, extracted = classify_run_id_type_and_extract(run_id)
    _log_event(run_id=run_id, run_type=run_type, pattern=extracted.get("pattern_name"))
    return run_type, extracted


def classify_run_id_type_and_extract(run_id: str) -> Tuple[str, Dict[str, str | None]]:
    for spec in PATTERN_SPECS:
        match = spec.regex.match(run_id)
        if match:
            extracted = match.groupdict()
            extracted["pattern_name"] = spec.name
            return spec.run_type, extracted

    if (
        "main_default" in run_id
        and "dpo" not in run_id
        and "--reduce_loss=" not in run_id
    ):
        return "old", {}

    return "other", {}


def parse_and_group_run_ids(
    df: pd.DataFrame, run_id_col: str = "run_id"
) -> Dict[str, List[Dict[str, str]]]:
    if run_id_col not in df.columns:
        raise KeyError(f"Column '{run_id_col}' not found in DataFrame")

    type_data: Dict[str, List[Dict[str, str]]] = {}
    for run_id in df[run_id_col].astype(str):
        run_type, extracted_data = classify_run_id(run_id)
        type_data.setdefault(run_type, [])
        if run_type != "old":
            extracted_data["run_id"] = run_id
            type_data[run_type].append(extracted_data)
    for run_type, records in type_data.items():
        records.sort(key=lambda x: x.get("run_id", ""))
    return type_data


def convert_groups_to_dataframes(
    grouped_data: Dict[str, List[Dict[str, str]]]
) -> Dict[str, pd.DataFrame]:
    dataframes: Dict[str, pd.DataFrame] = {}
    for run_type, records in grouped_data.items():
        if records:
            df = pd.DataFrame(records)
            if "pattern_name" in df.columns:
                df = df.sort_values("pattern_name")
            columns = ["run_id"] + [col for col in df.columns if col != "run_id"]
            dataframes[run_type] = df[columns]
    return dataframes


def convert_timestamp(ts_str: Any) -> Optional[pd.Timestamp]:
    if pd.isna(ts_str):
        return None
    ts_str = str(ts_str)
    if "_" in ts_str:
        try:
            return pd.to_datetime(ts_str, format="%Y_%m_%d-%H_%M_%S")
        except (ValueError, TypeError):
            return None
    try:
        return pd.to_datetime(ts_str, format="%y%m%d-%H%M%S")
    except (ValueError, TypeError):
        return None


def convert_string_to_number(value_str: Any) -> Optional[float]:
    if pd.isna(value_str):
        return None
    value_str = str(value_str).strip().upper()
    if value_str in {"N/A", ""}:
        return None
    try:
        if value_str.endswith("M"):
            return float(value_str[:-1]) * 1e6
        if value_str.endswith(("G", "B")):
            return float(value_str[:-1]) * 1e9
        if value_str.endswith("T"):
            return float(value_str[:-2]) * 1e12
        return float(value_str)
    except (ValueError, TypeError):
        return None


def extract_config_fields(
    runs_df: pd.DataFrame, run_ids: Iterable[str], field_mapping: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
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
            except (json.JSONDecodeError, ValueError, TypeError):
                summary = None
            if summary and summary.get("total_tokens") is not None:
                config_data.setdefault(run_id, {})["num_finetuned_tokens_real"] = summary[
                    "total_tokens"
                ]
    return config_data


def apply_processing(
    dataframes: Dict[str, pd.DataFrame],
    defaults: Optional[Dict[str, Any]] = None,
    column_map: Optional[Dict[str, str]] = None,
    runs_df: Optional[pd.DataFrame] = None,
    history_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    defaults = defaults or DEFAULTS
    column_map = column_map or {}

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

        for col, default_val in defaults.items():
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].fillna(default_val)

        for recipe_col in recipe_columns:
            if recipe_col in processed_df.columns:
                processed_df[recipe_col] = processed_df[recipe_col].map(
                    lambda x: RECIPE_MAPPING.get(x, x) if pd.notna(x) else x
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
            processed_df["timestamp"] = processed_df["timestamp"].apply(convert_timestamp)

        if "comparison_model_size" in processed_df.columns:
            processed_df["comparison_model_recipe"] = processed_df[
                "comparison_model_recipe"
            ].fillna("Dolma1.7")

        if run_type == "matched":
            if "comparison_metric" not in processed_df.columns:
                processed_df["comparison_metric"] = "pile"
            processed_df["comparison_metric"] = processed_df["comparison_metric"].fillna(
                "pile"
            )
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

        mask = processed_df["run_id"].str.contains("_Ft_") if "run_id" in processed_df else False
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
                (processed_df.loc[mask, "num_finetune_tokens"]
                 - processed_df.loc[mask, "num_finetuned_tokens_real"])
                .abs()
                / processed_df.loc[mask, "num_finetune_tokens"]
                * 100
            )

        processed[run_type] = processed_df

    return processed

