import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path

    from dr_ingest.configs import Paths
    from dr_ingest.metrics_all.constants import LoadMetricsAllConfig
    from dr_ingest.metrics_all.load_results import load_all_results
    return LoadMetricsAllConfig, Path, Paths, load_all_results, mo, pd


@app.cell
def _(Paths, mo):
    paths = Paths()
    default_root = paths.metrics_all_dir
    metrics_root_input = mo.ui.text(
        value=str(default_root),
        label="Metrics root directory",
        full_width=True,
    )
    mo.vstack(
        [
            mo.md("Select the metrics-all root to ingest."),
            metrics_root_input,
        ]
    )
    return (metrics_root_input,)


@app.cell
def _(LoadMetricsAllConfig, Path, load_all_results, metrics_root_input):
    selected_root = Path(metrics_root_input.value).expanduser()
    cfg = LoadMetricsAllConfig(root_paths=[selected_root])
    raw_records = load_all_results(root_paths=[selected_root], config=cfg)
    return raw_records, selected_root


@app.cell
def _(mo, pd, raw_records, selected_root):
    results_df = pd.DataFrame(raw_records)
    mo.vstack(
        [
            mo.md(
                f"Loaded {len(results_df)} records from `{selected_root}`"
            ),
            results_df,
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
