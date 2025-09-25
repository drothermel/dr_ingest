from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import pandas as pd

from dr_ingest.wandb.tokens import (
    ensure_full_finetune_defaults,
    fill_missing_token_totals,
)

if TYPE_CHECKING:  # pragma: no cover
    from dr_ingest.wandb.processing_context import ProcessingContext


@attrs.define(frozen=True)
class RunNormalizationExecutor:
    context: ProcessingContext

    @classmethod
    def from_context(cls, context: ProcessingContext) -> RunNormalizationExecutor:
        return cls(context)

    def normalize(
        self,
        frame: pd.DataFrame,
        *,
        run_type: str | None = None,
    ) -> pd.DataFrame:
        result = frame.copy()
        result = self.context.apply_defaults(result)
        result = self.context.map_recipes(result)
        result = self.context.rename_columns(result)
        result = self.context.apply_value_converters(result)
        result = ensure_full_finetune_defaults(result)
        result = fill_missing_token_totals(result)
        if run_type is not None:
            result = self.context.apply_hook(run_type, result)
        return result


__all__ = ["RunNormalizationExecutor"]
