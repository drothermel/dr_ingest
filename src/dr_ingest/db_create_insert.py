import duckdb
import polars as pl


def create_train_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.sql("""
    CREATE TABLE train_results (
        id INTEGER PRIMARY KEY,
        params VARCHAR,
        task VARCHAR,
        recipe recipe_enum,
        seed VARCHAR,
        step REAL,
        tokens_million REAL,
        compute_e15 REAL,
        metrics all_eval_metrics_struct
    );
    """)


def create_scaling_law_pred_one_step_raw(conn: duckdb.DuckDBPyConnection) -> None:
    conn.sql("""
    CREATE TABLE scaling_law_pred_one_step_raw (
        id INTEGER PRIMARY KEY,
        task qa_task_enum,
        recipe recipe_enum,
        fit_config scaling_law_config_struct,
        pred_task_metrics scaling_law_metrics_struct
    );
    """)


def create_scaling_law_pred_two_step_raw(conn: duckdb.DuckDBPyConnection) -> None:
    conn.sql("""
    CREATE TABLE scaling_law_pred_two_step_raw (
        id INTEGER PRIMARY KEY,
        task qa_task_enum,
        recipe recipe_enum,
        fit_config scaling_law_config_struct,
        pred_task_losses scaling_law_metrics_struct,
        pred_task_loss_to_metrics scaling_law_metrics_struct,
        pred_task_metrics scaling_law_metrics_struct
    );
    """)


def create_scaling_law_true_raw(conn: duckdb.DuckDBPyConnection) -> None:
    conn.sql("""
    CREATE TABLE scaling_law_true_raw (
        id INTEGER PRIMARY KEY,
        task qa_task_enum,
        recipe recipe_enum,
        task_losses scaling_law_metrics_struct,
        task_metrics scaling_law_metrics_struct
    );
    """)


SCALING_LAW_TABLES = {
    "scaling_law_pred_one_step_raw": create_scaling_law_pred_one_step_raw,
    "scaling_law_pred_two_step_raw": create_scaling_law_pred_two_step_raw,
    "scaling_law_true_raw": create_scaling_law_true_raw,
}


def create_all_scaling_law_tables(conn: duckdb.DuckDBPyConnection) -> None:
    for table_name, create_func in SCALING_LAW_TABLES.items():
        create_func(conn)
        print(f">> Created table {table_name}")


def insert_into_scaling_law_one_step_raw(
    conn: duckdb.DuckDBPyConnection,
    df: pl.DataFrame,
) -> None:
    conn.register("df", df)
    conn.execute("""
    INSERT INTO scaling_law_pred_one_step_raw
    SELECT
        id,
        task::qa_task_enum,
        recipe::recipe_enum,
        fit_config::scaling_law_config_struct,
        pred_task_metrics::scaling_law_metrics_struct
    FROM df;
    """)
    conn.unregister("df")


def insert_into_scaling_law_pred_two_step_raw(
    conn: duckdb.DuckDBPyConnection,
    df: pl.DataFrame,
) -> None:
    conn.register("df", df)
    conn.execute("""
    INSERT INTO scaling_law_pred_two_step_raw
    SELECT
        id,
        task::qa_task_enum,
        recipe::recipe_enum,
        fit_config::scaling_law_config_struct,
        pred_task_losses::scaling_law_metrics_struct,
        pred_task_loss_to_metrics::scaling_law_metrics_struct,
        pred_task_metrics::scaling_law_metrics_struct
    FROM df;
    """)
    conn.unregister("df")


def insert_into_scaling_law_true_raw(
    conn: duckdb.DuckDBPyConnection,
    df: pl.DataFrame,
) -> None:
    conn.register("df", df)
    conn.execute("""
    INSERT INTO scaling_law_true_raw
    SELECT
        id,
        task::qa_task_enum,
        recipe::recipe_enum,
        task_losses::scaling_law_metrics_struct,
        task_metrics::scaling_law_metrics_struct
    FROM df;
    """)
    conn.unregister("df")


def insert_into_train_table(
    conn: duckdb.DuckDBPyConnection, train_df: pl.DataFrame
) -> None:
    conn.register("train_df", train_df)
    conn.execute("""
    INSERT INTO train_results
    SELECT
        id,
        params,
        task,
        recipe::recipe_enum,
        seed,
        step,
        tokens_millions,
        compute_e15,
        metrics::all_eval_metrics_struct
    FROM train_df;
    """)
    conn.unregister("train_df")


def insert_wandb_data_into_db(
    conn: duckdb.DuckDBPyConnection, wandb_dfs: dict[str, pl.DataFrame]
) -> None:
    for df_name, wdf in wandb_dfs:
        conn.register(f"{df_name}_df", wdf)
        conn.execute(f"""
        CREATE TABLE wandb_{df_name} AS SELECT * FROM {df_name}_df
        """)  # noqa: S608
        conn.unregister(f"{df_name}_df")


SCALING_LAW_INSERT_TABLES = {
    "scaling_law_pred_one_step_raw": insert_into_scaling_law_one_step_raw,
    "scaling_law_pred_two_step_raw": insert_into_scaling_law_pred_two_step_raw,
    "scaling_law_true_raw": insert_into_scaling_law_true_raw,
}


def insert_into_scaling_law_tables(
    conn: duckdb.DuckDBPyConnection, sl_results: list[tuple[str, pl.DataFrame]]
) -> None:
    for table_name, df in sl_results.items():
        insert_func = SCALING_LAW_INSERT_TABLES[table_name]
        insert_func(conn, df)
        print(f">> Inserted into table {table_name}")
