import marimo

__generated_with = "0.16.2"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    return


@app.cell(column=1)
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
        split_df_to_db_by_object_cols,
        load_parse_write_duckdb,
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
    dd_res_name_selector = "train-*.parquet"
    return (
        ENTITY,
        PROJECT,
        VARCHAR,
        ast,
        dd_res_dir,
        dd_res_name_selector,
        dd_res_names,
        duckdb,
        json,
        load_parse_write_duckdb,
        mo,
        pd,
        wandb_load_fxn,
        wandb_parse_fxn,
    )


@app.cell
def _(ast, duckdb, json):
    def literal_eval_udf(x: str) -> str:
        # Return JSON string; you could return dict, but use str for compatibility
        if x is None:
            return None
        try:
            return json.dumps(ast.literal_eval(x))
        except Exception:
            return None


    def get_variable_from_duckdb(var_name: str):
        return duckdb.sql(f"SELECT getvariable('{var_name}')").fetchone()[0]


    def schema_variable_to_struct_def(var_name: str) -> str:
        schema = get_variable_from_duckdb(var_name)
        json_schema = json.loads(schema)
        struct_def = ", ".join([f"{k} {v}" for k, v in json_schema.items()])
        return f"STRUCT({struct_def})"
    return literal_eval_udf, schema_variable_to_struct_def


@app.cell(hide_code=True)
def _(dd_res_dir, dd_res_names, mo, pd):
    ex_df_path = f"{dd_res_dir}{dd_res_names[0]}"
    ex_df = pd.read_parquet(ex_df_path)
    mo.vstack(
        [
            mo.md(f"""
            ### Load Results Table (part 0/4) with pandas.
            Load from: {ex_df_path}
            """),
            ex_df,
        ]
    )
    return


@app.cell(hide_code=True)
def _(dd_res_dir, dd_res_names, duckdb, mo):
    ex_duckdb_path = f"{dd_res_dir}{dd_res_names[0]}"
    ex_t = duckdb.read_parquet(ex_duckdb_path)
    mo.vstack(
        [
            mo.md(f"""
            ### Load Results Table with duckdb
            Load from: {ex_duckdb_path} into variable `ex_t`

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
def _(VARCHAR, duckdb, json, literal_eval_udf, mo):
    # Avoid duplicate definitions when rerunning by removing and re-adding
    try:
        duckdb.remove_function("py_literal_eval")
        duckdb.create_function("py_literal_eval", literal_eval_udf, [VARCHAR], VARCHAR)
    except Exception:
        pass

    duckdb.sql("""
    SET VARIABLE json_schema = (
        SELECT json_structure(py_literal_eval(metrics)::JSON)
        FROM ex_t
        LIMIT 1
    );
    """)
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
        Which produces:
        """),
            json.loads(duckdb.sql("SELECT getvariable('json_schema')").fetchone()[0]),
        ]
    )
    return


@app.cell
def _(mo, schema_variable_to_struct_def):
    alter_table_button = mo.ui.run_button(label="Run Alter Table")
    mo.vstack(
        [
            mo.md("""
        Finally, lets alter the table to add a metrics_struct col 
        (better storage efficiency and easier access), 
        then drop the old one and rename. We could also just select everything into a new table and rename it, 
        but the alter table method will use less memory which will be relevant in the future.
    
        First we need to convert the schema to a struct def of form:
        ```SQL
        STRUCT(
            key1 TYPE, 
            key2 TYPE, 
            ...
        )
        ```
        Which for us produces:
        """),
            schema_variable_to_struct_def("json_schema"),
            mo.md("""
        Then we can do the alter table:
        ```python
        duckdb.sql(\"""
            ALTER TABLE ex_t
            ADD COLUMN metrics_struct STRUCT(...);
        
            UPDATE ex_t 
            SET metrics_struct = from_json(
                py_literal_eval(metrics)::JSON, 
                getvariable('json_schema')
            );
        
            ALTER TABLE ex_t DROP COLUMN metrics;
        
            ALTER TABLE ex_t RENAME COLUMN metrics_struct TO metrics;
        \""")
        ```
        """),
        ]
    )
    return (alter_table_button,)


@app.cell
def _(alter_table_button, duckdb, ex_t, mo):
    mo.vstack(
        mo.stop(not alter_table_button.value, alter_table_button),
        duckdb.sql("""
        ALTER TABLE ex_t 
        ADD COLUMN metrics_struct {}
        """),
        duckdb.sql("""
        UPDATE ex_t 
        SET metrics_struct = from_json(
            py_literal_eval(metrics)::JSON, 
            getvariable('json_schema')
        )
        """),
        duckdb.sql("ALTER TABLE ex_t DROP COLUMN metrics"),
        duckdb.sql("ALTER TABLE ex_t RENAME COLUMN metrics_struct TO metrics"),
        mo.plain_text(ex_t.describe()),
    )
    return


@app.cell
def _():
    return


@app.cell(column=2)
def _():
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


@app.cell(column=3)
def _():
    return


if __name__ == "__main__":
    app.run()
