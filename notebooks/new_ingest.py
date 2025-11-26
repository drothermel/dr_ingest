import marimo

__generated_with = "0.17.7"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    from pathlib import Path
    from typing import Any

    import dr_ingest.utils as du
    from dr_ingest.metrics_all.constants import LoadMetricsAllConfig
    from dr_ingest.types import TaskArtifactType

    ex_dir = Path(
        "/Users/daniellerothermel/"
        "drotherm/data/datadec/"
        "2025-11-03_posttrain/_eval_results/"
        "meta-llama_Llama-3.1-8B__main"
    )

    load_cfg = LoadMetricsAllConfig(
        root_paths=[ex_dir],
    )
    return Any, LoadMetricsAllConfig, TaskArtifactType, du, load_cfg, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Next Steps
    - Setup file loading, metadata extraction and bundling
    - Setup dumping to cache dir
    """)
    return


@app.cell
def _(Any, LoadMetricsAllConfig, TaskArtifactType, du):
    def collect_all_eval_paths(config: LoadMetricsAllConfig) -> list[dict[str, Any]]:
        metrics_all_paths = list(
            du.iter_file_glob_from_roots(
                config.root_paths, file_glob=config.results_filename
            )
        )
        result_dirs = {map: map.parent for map in metrics_all_paths}
        all_paths = []
        for metrics_all_path in du.iter_file_glob_from_roots(
            config.root_paths, file_glob=config.results_filename

        ):
            results_dir = metrics_all_path.parent
            all_paths.append({
                "metrics-all": metrics_all_path,
                "results_dir": results_dir,
                **{
                    t: list(du.iter_file_glob_from_roots(
                        results_dir, file_glob=t.filename_pattern
                    )) for t in TaskArtifactType
                }
            })
        return all_paths
    return (collect_all_eval_paths,)


@app.cell(column=1)
def _(load_cfg):
    load_cfg
    return


@app.cell
def _(collect_all_eval_paths, load_cfg):
    all_paths = collect_all_eval_paths(load_cfg)
    list(all_paths[0].keys())
    return


@app.cell
def _():
    return


@app.cell(column=2)
def _():
    return


if __name__ == "__main__":
    app.run()
