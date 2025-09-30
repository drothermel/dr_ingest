import marimo

__generated_with = "0.16.2"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo
    import duckdb
    import os
    from dotenv import load_dotenv

    load_dotenv()
    return duckdb, os


@app.cell
def _():
    from dr_ingest.db_types import (
        create_all_enums,
        create_all_structs,
    )
    from dr_ingest.db_create_insert import (
        create_train_table,
        create_all_scaling_law_tables,
    )
    return


@app.cell
def _(os):
    motherduck_token = os.getenv("MOTHERDUCK_TOKEN")
    huggingface_token = os.getenv("HF_TOKEN")
    motherduck_token, huggingface_token
    return huggingface_token, motherduck_token


@app.cell
def _(duckdb, motherduck_token):
    conn_local = duckdb.connect("./data/dd_eval_results.duckdb")
    conn_md = duckdb.connect(f"md:?motherduck_token={motherduck_token}")
    return conn_local, conn_md


@app.cell
def _(conn_md, huggingface_token):
    conn_md.execute(f"""
        CREATE SECRET hf_token (
            TYPE HUGGINGFACE,
            TOKEN '{huggingface_token}'
        );
    """)
    return


@app.cell
def _(conn_md):
    conn_md.execute("""
        CREATE DATABASE dd_ (
            TYPE DUCKLAKE,
            DATA_PATH 'hf://datasets/drotherm/dd_parsed/'
        );
    """)
    return


@app.cell
def _():
    #conn_md.execute("USE dd_ducklake")
    return


@app.cell
def _(conn_local):
    conn_local.execute("SHOW TABLES").df()
    return


@app.cell
def _(conn_local):
    scaling_law_true_raw_df = conn_local.execute("SELECT * FROM scaling_law_true_raw").fetch_df()
    scaling_law_true_raw_df['task'] = scaling_law_true_raw_df['task'].astype('string')
    scaling_law_true_raw_df['recipe'] = scaling_law_true_raw_df['task'].astype('string')
    scaling_law_true_raw_df
    return


@app.cell
def _(conn_local):
    conn_local.execute("COPY (SELECT * FROM scaling_law_true_raw) TO 'data/scaling_law_true_raw.parquet' (FORMAT PARQUET);")
    return


@app.cell
def _(conn_local):
    for name in [
        "scaling_law_pred_one_step_raw",
        "scaling_law_pred_two_step_raw",
        "scaling_law_true_raw",
        "train_results",
        "wandb_history",
        "wandb_runs_config",
        "wandb_runs_summary",
        "wandb_runs_sweep_info",
        "wandb_runs_system_attrs",
        "wandb_runs_system_metrics",
        "wandb_runs_wandb_metadata",
    
    ]:
        conn_local.execute(f"COPY (SELECT * FROM {name}) TO 'data/{name}.parquet' (FORMAT PARQUET);")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
