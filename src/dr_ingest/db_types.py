import contextlib

import duckdb


def create_scaling_law_params_enum(conn: duckdb.DuckDBPyConnection) -> None:
    conn.sql("""
    CREATE TYPE scaling_law_params_enum AS ENUM (
        '2_A_alpha',
        '3_A_alpha_E',
        '5_A_alpha_B_beta_E'
    );
    """)


def create_scaling_law_filter_enum(conn: duckdb.DuckDBPyConnection) -> None:
    conn.sql("""
    CREATE TYPE scaling_law_filter_enum AS ENUM (
        '50_Percent',
        'None'
    );
    """)


def create_qa_task_enum(conn: duckdb.DuckDBPyConnection) -> None:
    conn.sql("""
    CREATE TYPE qa_task_enum AS ENUM (
        'arc_challenge',
        'arc_easy',
        'boolq',
        'csqa',
        'hellaswag',
        'openbookqa',
        'piqa',
        'socialiqa',
        'winogrande',
        'olmes_10_macro_avg',
        'mmlu'
    );
    """)


def create_mmlu_task_enum(conn: duckdb.DuckDBPyConnection) -> None:
    conn.sql("""
    CREATE TYPE mmlu_task_enum AS ENUM (
        'mmlu_average',
        'mmlu_abstract_algebra',
        'mmlu_anatomy',
        'mmlu_astronomy',
        'mmlu_business_ethics',
        'mmlu_clinical_knowledge',
        'mmlu_college_biology',
        'mmlu_college_chemistry',
        'mmlu_college_computer_science',
        'mmlu_college_mathematics',
        'mmlu_college_medicine',
        'mmlu_college_physics',
        'mmlu_computer_security',
        'mmlu_conceptual_physics',
        'mmlu_econometrics',
        'mmlu_electrical_engineering',
        'mmlu_elementary_mathematics',
        'mmlu_formal_logic',
        'mmlu_global_facts',
        'mmlu_high_school_biology',
        'mmlu_high_school_chemistry',
        'mmlu_high_school_computer_science',
        'mmlu_high_school_european_history',
        'mmlu_high_school_geography',
        'mmlu_high_school_government_and_politics',
        'mmlu_high_school_macroeconomics',
        'mmlu_high_school_mathematics',
        'mmlu_high_school_microeconomics',
        'mmlu_high_school_physics',
        'mmlu_high_school_psychology',
        'mmlu_high_school_statistics',
        'mmlu_high_school_us_history',
        'mmlu_high_school_world_history',
        'mmlu_human_aging',
        'mmlu_human_sexuality',
        'mmlu_international_law',
        'mmlu_jurisprudence',
        'mmlu_logical_fallacies',
        'mmlu_machine_learning',
        'mmlu_management',
        'mmlu_marketing',
        'mmlu_medical_genetics',
        'mmlu_miscellaneous',
        'mmlu_moral_disputes',
        'mmlu_moral_scenarios',
        'mmlu_nutrition',
        'mmlu_philosophy',
        'mmlu_prehistory',
        'mmlu_professional_accounting',
        'mmlu_professional_law',
        'mmlu_professional_medicine',
        'mmlu_professional_psychology',
        'mmlu_public_relations',
        'mmlu_security_studies',
        'mmlu_sociology',
        'mmlu_us_foreign_policy',
        'mmlu_virology',
        'mmlu_world_religions'
    );
    """)


def create_recipe_enum(conn: duckdb.DuckDBPyConnection) -> None:
    conn.sql("""
    CREATE TYPE recipe_enum AS ENUM (
        'c4',
        'falcon',
        'falcon_cc',
        'falcon_cc_qc_tulu_10',
        'falcon_cc_qc_orig_10',
        'falcon_cc_qc_10',
        'falcon_cc_qc_20',
        'dolma16',
        'dolma17',
        'dolma17_no_flan',
        'dolma17_no_reddit',
        'dolma17_no_code',
        'dolma17_no_math_code',
        'fineweb_edu',
        'fineweb_pro',
        'dclm_baseline',
        'dclm_baseline_qc_10',
        'dclm_baseline_qc_20',
        'dclm_baseline_qc_fw_3',
        'dclm_baseline_qc_fw_10',
        'dclm_baseline_qc_7_fw2',
        'dclm_baseline_qc_7_fw3',
        'dclm_baseline_25_dolma_75',
        'dclm_baseline_50_dolma_50',
        'dclm_baseline_75_dolma_25'
    );
    """)


def create_param_enum(conn: duckdb.DuckDBPyConnection) -> None:
    conn.sql("""
    CREATE TYPE param_enum AS ENUM (
        '4M',
        '6M',
        '8M',
        '10M',
        '14M',
        '16M',
        '20M',
        '60M',
        '90M',
        '150M',
        '300M',
        '530M',
        '750M',
        '1B'
    );
    """)


def create_seed_enum(conn: duckdb.DuckDBPyConnection) -> None:
    conn.sql("""
    CREATE TYPE seed_enum AS ENUM (
        'small_aux_2',
        'small_aux_3',
        'large_aux_2',
        'large_aux_e',
        'default'
    );
    """)


def create_all_eval_metrics_struct(conn: duckdb.DuckDBPyConnection) -> None:
    conn.sql("""
    CREATE TYPE all_eval_metrics_struct AS STRUCT (
        accuracy full_metric_struct,
        correct_prob proxy_metric_struct,
        sum_correct_logits proxy_metric_struct,
        norm_correct_prob proxy_metric_struct,
        uncond_correct_prob proxy_metric_struct,
        total_prob proxy_metric_struct,
        margin proxy_metric_struct,
        bits_per_byte_correct REAL,
        primary_metric REAL
    );
    """)


def create_scaling_law_config_struct(conn: duckdb.DuckDBPyConnection) -> None:
    conn.sql("""
    CREATE TYPE scaling_law_config_struct AS STRUCT (
        params scaling_law_params_enum,
        filtering scaling_law_filter_enum,
        heldout VARCHAR[],
        one_step BOOLEAN,
        helper_point BOOLEAN
    );
    """)


def create_proxy_metric_struct(conn: duckdb.DuckDBPyConnection) -> None:
    conn.sql("""
    CREATE TYPE proxy_metric_struct AS STRUCT (
        raw FLOAT,
        per_char FLOAT,
        per_token FLOAT
    );
    """)


def create_full_metric_struct(conn: duckdb.DuckDBPyConnection) -> None:
    conn.sql("""
    CREATE TYPE full_metric_struct AS STRUCT (
        raw FLOAT,
        per_char FLOAT,
        per_token FLOAT,
        per_byte FLOAT,
        uncond FLOAT
    );
    """)


def create_scaling_law_metrics_struct(conn: duckdb.DuckDBPyConnection) -> None:
    conn.sql("""
    CREATE TYPE scaling_law_metrics_struct AS STRUCT (
        margin proxy_metric_struct,
        norm_correct_prob proxy_metric_struct,
        total_prob proxy_metric_struct,
        accuracy proxy_metric_struct,
        correct_prob proxy_metric_struct
    );
    """)


ALL_ENUM_CREATION_FUNCTIONS = {
    "scaling_law_params_enum": create_scaling_law_params_enum,
    "scaling_law_filter_enum": create_scaling_law_filter_enum,
    "qa_task_enum": create_qa_task_enum,
    "mmlu_task_enum": create_mmlu_task_enum,
    "recipe_enum": create_recipe_enum,
    "param_enum": create_param_enum,
    "seed_enum": create_seed_enum,
}

ALL_STRUCT_CREATION_FUNCTIONS = {
    "scaling_law_config_struct": create_scaling_law_config_struct,
    "proxy_metric_struct": create_proxy_metric_struct,
    "full_metric_struct": create_full_metric_struct,
    "scaling_law_metrics_struct": create_scaling_law_metrics_struct,
    "all_eval_metrics_struct": create_all_eval_metrics_struct,
}


def create_all_enums(conn: duckdb.DuckDBPyConnection) -> None:
    for name, func in ALL_ENUM_CREATION_FUNCTIONS.items():
        with contextlib.suppress(duckdb.Error):
            func(conn)
            print(f">> Created enum {name}")


def create_all_structs(conn: duckdb.DuckDBPyConnection) -> None:
    for name, func in ALL_STRUCT_CREATION_FUNCTIONS.items():
        try:
            func(conn)
            print(f">> Created struct {name}")
        except Exception as e:
            print(f">> Failed to create struct {name}: {e}")
