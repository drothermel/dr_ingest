from typing import Any

import polars as pl

NUM_FIT_PARAMS_MAP = {
    "2": "2_A_alpha",
    "3": "3_A_alpha_E",
    "5": "5_A_alpha_B_beta_E",
}


def get_sl_num_fit_params(value: str) -> str:
    fp_str = value[0]
    return NUM_FIT_PARAMS_MAP.get(fp_str, "Unknown")


def get_sl_one_step_bool(value: str) -> bool:
    return "1_step" in value


def get_sl_filter(value: str) -> str:
    if "step2=0.5" in value:
        return "50_Percent"
    return "None"


def get_sl_helper_point(value: str) -> bool:
    return "helper_points" in value


def get_heldout(value: str) -> list[str]:
    heldout_list = ["1B"]
    vsplit = value.split("-")
    for item in vsplit:
        if "no_" in item:
            heldout_list.extend(item.split("no_"))
    return [h.strip("_") for h in heldout_list if h != ""]


def parse_sl_setup_to_config(value: str) -> dict[str, Any]:
    return {
        "name": value,
        "one_step": get_sl_one_step_bool(value),
        "params": get_sl_num_fit_params(value),
        "filtering": get_sl_filter(value),
        "helper_point": get_sl_helper_point(value),
        "heldout": get_heldout(value),
    }


def parse_all_sl_setups_to_configs(values: list[str]) -> pl.DataFrame:
    return pl.DataFrame([parse_sl_setup_to_config(value) for value in values])
