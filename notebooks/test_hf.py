import marimo

__generated_with = "0.17.7"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    import pandas as pd
    from dr_ingest.hf.io import get_tables_from_cache
    from dr_ingest.configs import DataDecideSourceConfig, Paths
    from dr_ingest.pipelines.dd_results import parse_train_df


@app.cell
def _():
    dd_cfg = DataDecideSourceConfig()
    return (dd_cfg,)


@app.cell
def _():
    paths = Paths()
    return (paths,)


@app.cell
def _(dd_cfg):
    dd_cfg.results_hf
    return


@app.cell
def _(dd_cfg, paths):
    dd_cfg.results_hf.resolve_filepaths(local_dir=paths.data_cache_dir)
    return


@app.cell
def _():
    parsed_df = pd.read_parquet("/Users/daniellerothermel/drotherm/data/cache/train_results.parquet")
    parsed_df
    return


@app.cell
def _():
    mo.md(r"""
    def wandb_load_fxn(**kwargs: Any) -> tuple[list[dict], list[dict]]:
        entity = kwargs.get("entity")
        project = kwargs.get("project")
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


    def wandb_parse_fxn(
        runs_history: tuple[list[dict], list[dict]], **kwargs: Any
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        runs, history = runs_history
        runs_df = pd.DataFrame(runs)
        history_df = pd.DataFrame(history)
        print(">> Parsing runs...")
        yield from split_df_to_db_by_object_cols(runs_df, name_prefix="runs_")
        print(">> Parsing history...")
        yield "history", history_df
    """)
    return


@app.cell
def _():
    mo.md(r"""
    wandb_loaded = wandb_load_fxn(
        entity="ml-moe", project="ft-scaling", redownload=redownload
    )
    wandb_dfs = list(wandb_parse_fxn(wandb_loaded))
    insert_wandb_data_into_db(conn, wandb_dfs)
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
