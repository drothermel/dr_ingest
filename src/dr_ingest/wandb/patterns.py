"""Regex pattern registry for WandB run classification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import catalogue

from . import constants as c

pattern_registry = catalogue.create("dr_ingest", "wandb", "patterns")


@dataclass(frozen=True)
class PatternSpec:
    """Specification for a single regex pattern."""

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


# Default pattern registrations.
_register_pattern(
    "FT1_PATTERN",
    "simple_ft_vary_tokens",
    rf"^{c.TIMESTAMP_8_EXP_NAME}_{c.DD_BLOCK_STEPS_WORD}_{c.INITIAL_CHECKPOINT_STEPS_WORD}_{c.FINETUNE_TOKENS_EPOCHS_8}{c.LEARNING_RATE_FLAG}$",
)
_register_pattern(
    "FT3_PATTERN",
    "simple_ft_vary_tokens",
    rf"^{c.TIMESTAMP_8_EXP_NAME}_{c.DD_BLOCK_STEPS_WORD}_{c.INITIAL_CHECKPOINT_STEPS_WORD}_{c.FINETUNE_TOKENS_8}_toks{c.LEARNING_RATE_FLAG}$",
)
_register_pattern(
    "FT4_PATTERN",
    "simple_ft",
    rf"^{c.TIMESTAMP_8_EXP_NAME}_{c.DD_BLOCK_STEPS_WORD}_Ft{c.LEARNING_RATE_FLAG}$",
)
_register_pattern(
    "FT5_PATTERN",
    "simple_ft",
    rf"^{c.TIMESTAMP_6_EXP_NAME}_DD-{c.INITIAL_CHECKPOINT_RECIPE_DASH}-{c.INITIAL_CHECKPOINT_SIZE}_Ft{c.LEARNING_RATE_EQUAL}$",
)
_register_pattern(
    "FT6_PATTERN",
    "simple_ft_vary_tokens",
    rf"^{c.TIMESTAMP_6_EXP_NAME}_{c.FINETUNE_TOKENS_EPOCHS_8}_{c.DD_BLOCK_FULL}{c.LR_SUFFIX}$",
)
_register_pattern(
    "FT7_PATTERN",
    "simple_ft",
    rf"^{c.TIMESTAMP_6_EXP_NAME}_Ft_{c.DD_BLOCK_FULL}{c.LR_SUFFIX}$",
)
_register_pattern(
    "MATCHED6_PATTERN",
    "matched",
    rf"^{c.TIMESTAMP_6_EXP_NAME}_{c.DD_COMPARISON_6}_Ft_{c.DD_BLOCK_FULL}{c.LR_SUFFIX}$",
)
_register_pattern(
    "MATCHED7_PATTERN",
    "matched",
    rf"^{c.TIMESTAMP_8_EXP_NAME}_{c.DD_COMPARISON_6}_Ft_{c.DD_BLOCK_FULL}{c.LR_SUFFIX}$",
)
_register_pattern(
    "REDUCE_LOSS_PATTERN",
    "reduce_type",
    rf"^{c.TIMESTAMP_8_EXP_NAME}_{c.DD_BLOCK_STEPS_WORD}_{c.INITIAL_CHECKPOINT_STEPS_WORD}_default_--max_train_samples={c.FINETUNE_TOKENS_SIMPLE}_--reduce_loss={c.REDUCE_LOSS}$",
)
_register_pattern(
    "DPO1_PATTERN",
    "dpo",
    rf"^{c.TIMESTAMP_8_EXP_NAME}_{c.DD_BLOCK_STEPS_WORD}_{c.INITIAL_CHECKPOINT_STEPS_WORD}_default$",
)
_register_pattern(
    "DPO2_PATTERN",
    "dpo",
    rf"^{c.TIMESTAMP_8_EXP_NAME}_dd__{c.INITIAL_CHECKPOINT_RECIPE}-{c.INITIAL_CHECKPOINT_SIZE}__{c.INITIAL_CHECKPOINT_STEPS_WORD}__{c.FINETUNE_TOKENS_GT}_lr={c.LEARNING_RATE_1}_default_--learning_rate={c.LEARNING_RATE_2}$",
)
_register_pattern(
    "MATCHED1_PATTERN",
    "matched",
    rf"^{c.MATCHED_PREFIX_WITH_METRIC}{c.FINETUNE_TOKENS_6}_{c.DD_BLOCK_FULL}$",
)
_register_pattern(
    "MATCHED2_PATTERN",
    "matched",
    rf"^{c.MATCHED_PREFIX_WITH_METRIC}{c.FINETUNE_FT}_{c.DD_BLOCK_FULL}$",
)
_register_pattern(
    "MATCHED3_PATTERN",
    "matched",
    rf"^{c.MATCHED_PREFIX_6}{c.FINETUNE_FT}_{c.DD_BLOCK_FULL}{c.LR_SUFFIX}$",
)
_register_pattern(
    "MATCHED4_PATTERN",
    "matched",
    rf"^{c.MATCHED_PREFIX_6}{c.FINETUNE_TOKENS_6}_{c.DD_BLOCK_FULL}$",
)
_register_pattern(
    "MATCHED5_PATTERN",
    "matched",
    rf"^{c.MATCHED_PREFIX_6}{c.FINETUNE_FT}_{c.DD_BLOCK_FULL}$",
)
_register_pattern(
    "MATCHED8_PATTERN",
    "matched",
    rf"^{c.MATCHED_PREFIX_6}{c.FINETUNE_TOKENS_6}_{c.DD_BLOCK_FULL}{c.LR_SUFFIX}$",
)

__all__ = [
    "PatternSpec",
    "PATTERN_SPECS",
    "pattern_registry",
]
