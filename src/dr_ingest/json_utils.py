"""Utility helpers for tolerant JSON parsing."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


def safe_load_json(payload: Any) -> dict[str, Any] | None:
    """Load JSON from strings or mappings, returning ``None`` on failure."""

    if payload is None or (isinstance(payload, float) and pd.isna(payload)):
        return None
    try:
        if isinstance(payload, str):
            return json.loads(payload)
        if isinstance(payload, dict):
            return payload
        return dict(payload)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


__all__ = ["safe_load_json"]
