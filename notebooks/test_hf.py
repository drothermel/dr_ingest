import marimo

__generated_with = "0.17.7"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo
    from dr_ingest.hf.location import HFLocation
    from dr_ingest.configs import DataDecideSourceConfig
    return (DataDecideSourceConfig,)


@app.cell
def _(DataDecideSourceConfig):
    dd_cfg = DataDecideSourceConfig()
    return (dd_cfg,)


@app.cell
def _(dd_cfg):
    dd_cfg.results_hf
    return


@app.cell
def _(dd_cfg):
    dd_cfg.results_hf.resolve_filepaths()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
