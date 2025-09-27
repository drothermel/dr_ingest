import marimo

__generated_with = "0.16.2"
app = marimo.App(width="full")


@app.cell
def _():
    from dr_wandb import fetch_project_runs
    import marimo as mo
    import pandas as pd
    import duckdb
    import srsly
    import sh

    from dr_ingest.raw_download import (
        is_nested,
        split_df_to_db_by_object_cols,
        load_parse_write_duckdb,
        wandb_load_fxn,
        wandb_parse_fxn,
    )

    ENTITY, PROJECT = "ml-moe", "ft-scaling"
    RUNS_NAME, HIST_NAME = "runs_t2", "hist_t2"
    return (
        ENTITY,
        PROJECT,
        load_parse_write_duckdb,
        mo,
        wandb_load_fxn,
        wandb_parse_fxn,
    )


@app.cell(hide_code=True)
def _(mo):
    switch = mo.ui.switch(label="Enable wandb download", value=False)
    parsing_switch = mo.ui.switch(label="Run parsing pipeline", value=False)
    mo.vstack(
        [
            mo.md(r"""
    # Exploring WandB Download & Processing

    The process is:

    $$\textrm{WandB} \rightarrow \textrm{Pandas DataFrame} \rightarrow \textrm{DuckDB Table} \rightarrow \textrm{Parquet File}$$

    The first step is to get the current data which you can do in two ways:

    1. Flip the following switch to actually download the latest from WandB (~15mins). 
    2. Load a local file to see the rest of the process in action! (default)
    """),
            switch,
            parsing_switch,
        ]
    )
    return parsing_switch, switch


@app.cell(hide_code=True)
def _(
    ENTITY,
    PROJECT,
    load_parse_write_duckdb,
    mo,
    parsing_switch,
    switch,
    wandb_load_fxn,
    wandb_parse_fxn,
):
    mo.stop(not parsing_switch.value, mo.md("Flip the switch to run the pipeline!"))
    all_wandb_dfs = load_parse_write_duckdb(
        df_name="wandb",
        load_fxn=wandb_load_fxn,
        parse_fxn=wandb_parse_fxn,
        out_dir="notebooks",
        entity=ENTITY,
        project=PROJECT,
        redownload=switch.value,
        log_every=1,
    )
    return (all_wandb_dfs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## View Wandb Raw Run Tables""")
    return


@app.cell
def _(all_wandb_dfs):
    wandb_dfs = iter(all_wandb_dfs.items())
    return (wandb_dfs,)


@app.cell
def _(wandb_dfs):
    next(wandb_dfs)
    return


@app.cell
def _(wandb_dfs):
    next(wandb_dfs)
    return


@app.cell
def _(wandb_dfs):
    next(wandb_dfs)
    return


@app.cell
def _(wandb_dfs):
    next(wandb_dfs)
    return


@app.cell
def _(wandb_dfs):
    next(wandb_dfs)
    return


@app.cell
def _(wandb_dfs):
    next(wandb_dfs)
    return


@app.cell
def _(wandb_dfs):
    next(wandb_dfs)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
