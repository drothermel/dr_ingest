import marimo

__generated_with = "0.17.7"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path
    import srsly

    ex_dir = Path(
        "/Users/daniellerothermel/"
        "drotherm/data/datadec/" 
        "2025-11-03_posttrain/_eval_results/"
        "meta-llama_Llama-3.1-8B__main"
    )
    return (ex_dir,)


@app.cell
def _(ex_dir):
    ex_dir.exists()
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _():
    return


@app.cell(column=2)
def _():
    return


if __name__ == "__main__":
    app.run()
