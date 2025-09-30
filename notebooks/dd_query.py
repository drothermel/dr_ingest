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
    return duckdb, mo, os


@app.cell
def _():
    import altair as alt
    return (alt,)


@app.cell
def _():
    import polars as pl
    import pandas as pd
    return (pd,)


@app.cell
def _(duckdb, os):
    motherduck_token = os.getenv("MOTHERDUCK_TOKEN")
    huggingface_token = os.getenv("HF_TOKEN")
    conn_md = duckdb.connect(f"md:?motherduck_token={motherduck_token}")
    conn_md.execute(f"""
        CREATE SECRET IF NOT EXISTS hf_token (
            TYPE HUGGINGFACE,
            TOKEN '{huggingface_token}'
        );
    """)
    return (conn_md,)


@app.cell
def _(mo):
    mo.md("## Load all Tables")
    return


@app.cell
def _():
    dataset_base = "hf://datasets/drotherm/dd_parsed"
    tables = [
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
    ]
    return dataset_base, tables


@app.cell
def _(conn_md, dataset_base, tables):
    dfs = {}
    for table in tables:
        table_path = f"{dataset_base}/{table}.parquet"
        print(table_path)
        dfs[table] = conn_md.execute(f"""SELECT * FROM '{table_path}'""").fetch_df()
    dfs
    return (dfs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Plotting""")
    return


@app.cell
def _(alt, dfs, mo, pd):
    tr_df = pd.DataFrame(dfs["train_results"])
    tr_df["primary_metric"] = tr_df["metrics"].apply(lambda x: x["primary_metric"])
    tr_filtered_df = tr_df[
        (tr_df["params"].isin(["1B", "530M", "150M"]))
        & (tr_df["task"] == "arc_challenge")
        & (tr_df["seed"] == "default")
        & (tr_df["recipe"] == "dclm_baseline")
    ]
    tr_chart = mo.ui.altair_chart(
        alt.Chart(tr_filtered_df)
        .mark_point()
        .encode(x="step", y="primary_metric", color="params")
    )
    return tr_chart, tr_filtered_df


@app.cell
def _(tr_chart):
    tr_chart
    return


@app.cell
def _(tr_filtered_df):
    tr_filtered_df
    return


@app.cell
def _(alt, tr_filtered_df):
    _chart = (
        alt.Chart(tr_filtered_df)
        .mark_line()
        .encode(
            x=alt.X(field='tokens_million', type='quantitative'),
            y=alt.Y(field='primary_metric', type='quantitative', aggregate='mean'),
            color=alt.Color(field='params', type='nominal'),
            tooltip=[
                alt.Tooltip(field='tokens_million', format=',.2f'),
                alt.Tooltip(field='primary_metric', aggregate='mean', format=',.2f'),
                alt.Tooltip(field='params')
            ]
        )
        .properties(
            title='',
            height=290,
            width='container',
            config={
                'axis': {
                    'grid': True
                }
            }
        )
    )
    _chart
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
