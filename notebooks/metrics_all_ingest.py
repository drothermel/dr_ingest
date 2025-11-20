import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import pandas as pd

    from dr_ingest.configs import Paths
    from dr_ingest.metrics_all.constants import LoadMetricsAllConfig
    from dr_ingest.metrics_all.load_results import load_all_results

    paths = Paths(
        #metrics_all_dir="/Users/daniellerothermel/drotherm/data/datadec/2025-10-08_posttrain"
    )


@app.cell
def _():
    mo.vstack(
        [
            mo.md("The metrics-all root to ingest."),
            paths.metrics_all_dir,
        ]
    )
    return


@app.cell
def _():
    load_cfg = LoadMetricsAllConfig(root_paths=[paths.metrics_all_dir])
    raw_records = load_all_results(config=load_cfg)
    # Alternatively: load_all_results(root_paths=["any_old_path/"])
    return (raw_records,)


@app.cell
def _(raw_records):
    results_df = pd.DataFrame(raw_records)
    mo.vstack(
        [
            mo.md(f"Loaded {len(results_df)} records from `{paths.metrics_all_dir}`"),
            results_df,
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
