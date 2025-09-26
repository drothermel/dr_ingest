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

    ENTITY, PROJECT = "ml-moe", "ft-scaling"
    RUNS_NAME, HIST_NAME = "runs_t2", "hist_t2"
    return ENTITY, PROJECT, duckdb, fetch_project_runs, mo, pd, srsly


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


@app.cell
def _(pd):
    def is_nested(df, col):
        return df[col].apply(lambda x: isinstance(x, (list, dict))).any()


    def split_df_to_db_by_object_cols(
        df: pd.DataFrame, name_prefix: str = ""
    ) -> tuple[str, pd.DataFrame]:
        obj_cols, non_obj_cols = [], []
        for col in df.columns:
            if is_nested(df, col):
                obj_cols.append(col)
            else:
                non_obj_cols.append(col)
        for obj_col in obj_cols:
            obj_df = pd.DataFrame(df[obj_col].tolist())
            missing_id_cols = [col for col in non_obj_cols if col not in obj_df.columns]
            id_df = df[missing_id_cols]
            obj_df = pd.concat([id_df, obj_df], axis=1)
            obj_col_name = f"{name_prefix}{obj_col}"
            yield obj_col_name, obj_df
    return (split_df_to_db_by_object_cols,)


@app.cell
def _(duckdb):
    def load_parse_write_duckdb(df_name, load_fxn, parse_fxn, out_dir, **kwargs):
        con = duckdb.connect(":memory:")
        sub_dfs_gen = parse_fxn(load_fxn(**kwargs), **kwargs)
        sub_dfs = {}
        for sub_name, sub_df in sub_dfs_gen:
            name = f"{df_name}_{sub_name}"
            sub_dfs[name] = sub_df
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM sub_df")
            con.execute(
                f"COPY {name} to '{out_dir}/{name}.parquet' (FORMAT parquet, PARQUET_VERSION v2)"
            )
            print(">> Wrote:", f"{out_dir}/{name}.parquet")
        return sub_dfs
    return (load_parse_write_duckdb,)


@app.cell
def _(fetch_project_runs, pd, split_df_to_db_by_object_cols, srsly):
    def wandb_load_fxn(**kwargs):
        entity = kwargs.get("entity", None)
        project = kwargs.get("project", None)
        runs_per_page = kwargs.get("runs_per_page", 500)
        log_every = kwargs.get("log_every", 10)
        source_dir = kwargs.get("source_dir", "notebooks")
        redownload = (
            kwargs.get("redownload", False) and entity is not None and project is not None
        )
        if redownload:
            print(">> Redownloading from wandb...")
            return fetch_project_runs(
                entity,
                project,
                runs_per_page=runs_per_page,
                include_history=True,
                progress_callback=lambda i, total, name: print(
                    f">> Processing run {i}/{total}: {name}"
                )
                if i % log_every == 0
                else None,
            )
        print(">> Loading locally...")
        runs = list(srsly.read_jsonl(f"{source_dir}/wandb_runs.jsonl"))
        history = list(srsly.read_jsonl(f"{source_dir}/wandb_history.jsonl"))
        return runs, history


    def wandb_parse_fxn(runs_history, **kwargs):
        runs, history = runs_history
        runs_df = pd.DataFrame(runs)
        history_df = pd.DataFrame(history)
        print(">> Parsing runs...")
        yield from split_df_to_db_by_object_cols(runs_df, name_prefix="runs_")
        print(">> Parsing history...")
        yield "history", history_df
    return wandb_load_fxn, wandb_parse_fxn


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
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
