import marimo

__generated_with = "0.16.2"
app = marimo.App()


@app.cell
def _():
    return


@app.cell
def _():
    from dr_wandb import fetch_project_runs
    import marimo as mo
    import pandas as pd
    import duckdb
    from duckdb.typing import VARCHAR
    import srsly
    import sh
    import json
    import ast

    from dr_ingest.raw_download import (
        is_nested,
        literal_eval_udf,
        split_df_to_db_by_object_cols,
        load_parse_write_duckdb,
        dd_results_load_fxn,
        dd_results_parse_fxn,
        wandb_load_fxn,
        wandb_parse_fxn,
    )

    ENTITY, PROJECT = "ml-moe", "ft-scaling"
    RUNS_NAME, HIST_NAME = "runs_t2", "hist_t2"
    raw_downloads_dir = (
        "/Users/daniellerothermel/drotherm/repos/datadec/data/raw_downloads/"
    )
    dd_res_dir = f"{raw_downloads_dir}DD-eval-results/data/"
    dd_res_names = [f"train-0000{i}-of-00004.parquet" for i in range(4)]
    dd_res_other = [
        "macro_avg-00000-of-00001.parquet",
        "scaling_law_fit-00000-of-00001.parquet",
    ]
    return (
        ENTITY,
        PROJECT,
        dd_res_dir,
        dd_res_names,
        dd_res_other,
        dd_results_load_fxn,
        dd_results_parse_fxn,
        duckdb,
        load_parse_write_duckdb,
        mo,
        pd,
        wandb_load_fxn,
        wandb_parse_fxn,
    )


@app.cell
def _(dd_res_dir, dd_res_names, duckdb, mo):
    files = [f"{dd_res_dir}{name}" for name in dd_res_names]
    ex_t = duckdb.read_parquet(files)
    mo.vstack(
        [
            mo.md(f"""
            ### Load Results Table with duckdb
            Load from: {files} into variable `ex_t`

            Then we can run: `ex_t.explain()` to get a string back describing what the `ex_t` object is.  (Note: we need `mo.plain_text(...)` to render the newlines correctly.)
            """),
            mo.plain_text(ex_t.explain()),
            mo.md(
                "And then `ex_t.describe()` where .describe() returns a table which will only be properly formatted as a string if we use `mo.plain_text(str(...))`"
            ),
            mo.plain_text(str(ex_t.describe())),
            mo.md(
                "... but we could also just convert it to a df with `ex_t.describe().df()`"
            ),
            ex_t.describe().df(),
            mo.md("And finally, we can see the table itself:"),
            duckdb.sql("SELECT * FROM ex_t LIMIT 5").df(),
        ]
    )
    return (ex_t,)


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("""
            ## Processing metrics col
            The metrics column contains a python literal string of a dictionary.  First we need to create a function to convert the string to a json string:
            ```python
            import ast
            import json

            def literal_eval_udf(x: str) -> str:
                # Return JSON string for compatibility
                if x is None:
                    return None
                try:
                    return json.dumps(ast.literal_eval(x))
                except Exception:
                    return None
            ```
            Then we can add this to duckdb as a new function, where the try/catch and remove is to avoid redefining the fxn if we rerun the cell:
            ```python
            from duckdb.types import VARCHAR

            try:
                duckdb.remove_function("py_literal_eval")
                duckdb.create_function("py_literal_eval", literal_eval_udf, [VARCHAR], VARCHAR)
            except Exception:
                pass
            ```
            Next we can extract the schema from the metrics columna fter processing with this UDF:
            ```python
            duckdb.sql(\"""
            SET VARIABLE json_schema = (
                SELECT json_structure(py_literal_eval(metrics)::JSON)
                FROM ex_t
                LIMIT 1
            );
            \""")
            ```
            Which we can then use to see the schema:
            ```python
            json.loads(
                duckdb.sql(
                    "SELECT getvariable('json_schema')"
                ).fetchone()[0]
            )
            ```
            """)
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    extract_button = mo.ui.run_button(label="Run Metrics Extraction (Takes ~2m)")
    mo.md("""
    And now we can use this schema to create a struct by selecting a new table:
    ```python
    duckdb.sql(f\"""
    CREATE TABLE ex_new AS
    SELECT
        * EXCLUDE(metrics),
        from_json(
            py_literal_eval(metrics)::JSON,
            getvariable('json_schema')
        ) AS metrics
    FROM ex_t
    \""")
    ```
    """)
    return (extract_button,)


@app.cell
def _(
    dd_res_dir,
    dd_results_load_fxn,
    dd_results_parse_fxn,
    duckdb,
    extract_button,
    mo,
):
    mo.stop(not extract_button.value, extract_button)
    final_table_name, final_table = next(
        iter(dd_results_parse_fxn(dd_results_load_fxn(source_dir=dd_res_dir)))
    )
    mo.vstack(
        [
            mo.md("Original table:"),
            duckdb.sql("DESCRIBE ex_t").df(),
            mo.md(f"New table {final_table_name}:"),
            duckdb.sql("DESCRIBE final_table").df(),
        ]
    )
    return


@app.cell
def _(dd_res_dir, dd_res_other, mo, pd):
    macro_avg_path = f"{dd_res_dir}{dd_res_other[0]}"
    macro_avg_df = pd.read_parquet(macro_avg_path)
    mo.vstack([
        mo.md(f"""## Macro Average Data
        Loading: {macro_avg_path}
        """)
    ])
    macro_avg_df
    return


@app.cell
def _(dd_res_dir, dd_res_other, mo, pd):
    scaling_path = f"{dd_res_dir}{dd_res_other[1]}"
    scaling_df = pd.read_parquet(scaling_path)
    mo.vstack([
        mo.md(f"""## Scaling Law Fit Data
        Loading: {scaling_path}
        """),
        scaling_df,
    ])
    return


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


@app.cell(hide_code=True)
def _(dd_res_dir, dd_res_name_selector, dd_res_names, mo):
    qa_parsing_switch = mo.ui.switch(label="Run QA parsing pipeline", value=False)
    mo.vstack(
        [
            mo.md(
                f"""
    # Exploring QA Data Parsing

    The process is:

    """
                + r"$$\textrm{HF} \rightarrow \textrm{DuckDB Table} \rightarrow \textrm{Parquet File}$$"
                + f"""

    First we start with the aggregated results from dir: {dd_res_dir}

    Files: '{dd_res_name_selector}'
            """
            ),
            *[mo.md(f"- {name}") for name in dd_res_names],
            qa_parsing_switch,
        ]
    )
    return (qa_parsing_switch,)


@app.cell
def _(mo, qa_parsing_switch):
    mo.stop(
        not qa_parsing_switch.value, mo.md("Flip the switch to parse the results qa data.")
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
