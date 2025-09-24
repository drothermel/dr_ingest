from __future__ import annotations

import pandas as pd

from dr_ingest.wandb.utils import convert_string_to_number, convert_timestamp


def test_convert_timestamp_formats() -> None:
    assert convert_timestamp("250901-155734") == pd.Timestamp("2025-09-01 15:57:34")
    assert convert_timestamp("2025_08_30-16_54_48") == pd.Timestamp(
        "2025-08-30 16:54:48"
    )
    assert convert_timestamp("not-a-ts") is None


def test_convert_string_to_number_suffixes() -> None:
    assert convert_string_to_number("10M") == 10_000_000
    assert convert_string_to_number("1.5G") == 1_500_000_000
    assert convert_string_to_number(" ") is None
    assert convert_string_to_number(None) is None
