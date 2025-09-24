"""Public exports for dr_ingest."""

from .normalization import key_variants, normalize_key, split_by_known_prefix
from .serialization import compare_sizes, dump_runs_and_history, ensure_parquet

try:  # pragma: no cover - optional dependency
    from .wandb.summary import normalize_oe_summary
except ImportError:  # pragma: no cover
    normalize_oe_summary = None  # type: ignore[assignment]

__all__ = [
    "key_variants",
    "normalize_key",
    "split_by_known_prefix",
    "compare_sizes",
    "dump_runs_and_history",
    "ensure_parquet",
    "normalize_oe_summary",
]
