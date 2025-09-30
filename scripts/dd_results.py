import marimo

__generated_with = "0.16.2"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from collections import defaultdict
    from pathlib import Path
    import duckdb
    return Path, duckdb, mo, pl


@app.cell
def _():
    from dr_ingest.normalization import (
        normalize_ds_str,
        normalize_compute,
        normalize_tokens,
    )
    from dr_ingest.db_types import (
        create_all_enums,
        create_all_structs,
    )
    from dr_ingest.db_create_insert import (
        create_train_table,
        create_all_scaling_law_tables,
        insert_into_scaling_law_tables,
        insert_into_train_table,
        insert_wandb_data_into_db,
    )
    from dr_ingest.raw_download import (
        wandb_load_fxn,
        wandb_parse_fxn,
        parse_sl_results,
        parse_train_df,
    )
    return (
        create_all_enums,
        create_all_scaling_law_tables,
        create_all_structs,
        create_train_table,
        insert_into_scaling_law_tables,
        insert_into_train_table,
        insert_wandb_data_into_db,
        parse_sl_results,
        parse_train_df,
        wandb_load_fxn,
        wandb_parse_fxn,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load DFs
    train_df, sl_df, ma_df
    """
    )
    return


@app.cell
def _():
    redownload = True
    reparse_train = True
    return redownload, reparse_train


@app.cell
def _(Path, parse_train_df, pl, reparse_train):
    train_parsed_path = Path("./train_parsed.parquet")
    if reparse_train:
        train_df = parse_train_df(
            pl.read_parquet(
                [
                    "/Users/daniellerothermel/drotherm/repos/datadec/data/raw_downloads/DD-eval-results/data/train-00000-of-00004.parquet",
                    "/Users/daniellerothermel/drotherm/repos/datadec/data/raw_downloads/DD-eval-results/data/train-00001-of-00004.parquet",
                    "/Users/daniellerothermel/drotherm/repos/datadec/data/raw_downloads/DD-eval-results/data/train-00002-of-00004.parquet",
                    "/Users/daniellerothermel/drotherm/repos/datadec/data/raw_downloads/DD-eval-results/data/train-00003-of-00004.parquet",
                ]
            )
        )
        train_df.write_parquet(train_parsed_path)
    else:
        train_df = pl.read_parquet(train_parsed_path)
    return (train_df,)


@app.cell
def _(pl):
    sl_df = pl.read_parquet(
        "/Users/daniellerothermel/drotherm/repos/datadec/data/raw_downloads/DD-eval-results/data/scaling_law_fit-00000-of-00001.parquet"
    )
    return (sl_df,)


@app.cell
def _(sl_df):
    sl_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load Scaling Laws""")
    return


@app.cell
def _(duckdb):
    conn = duckdb.connect("./data/dd_eval_results.duckdb")
    return (conn,)


@app.cell
def _(
    conn,
    create_all_enums,
    create_all_scaling_law_tables,
    create_all_structs,
):
    create_all_enums(conn)
    create_all_structs(conn)
    create_all_scaling_law_tables(conn)
    return


@app.cell
def _(conn, insert_into_scaling_law_tables, parse_sl_results, sl_df):
    sl_results = parse_sl_results(sl_df)
    insert_into_scaling_law_tables(conn, sl_results)
    return


@app.cell
def _(conn, create_train_table):
    create_train_table(conn)
    return


@app.cell
def _(conn, insert_into_train_table, train_df):
    insert_into_train_table(conn, train_df)
    return


@app.cell
def _(
    conn,
    insert_wandb_data_into_db,
    redownload,
    wandb_load_fxn,
    wandb_parse_fxn,
):
    wandb_loaded = wandb_load_fxn(
        entity="ml-moe", project="ft-scaling", redownload=redownload
    )
    wandb_dfs = list(wandb_parse_fxn(wandb_loaded))
    insert_wandb_data_into_db(conn, wandb_dfs)
    return


@app.cell
def _(conn):
    conn.sql("SHOW TABLES")
    return


@app.cell
def _(conn):
    conn.close()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
