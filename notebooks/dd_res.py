import marimo

__generated_with = "0.16.2"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    import polars as pl
    import pandas as pd

    from pathlib import Path

    from dr_ingest.normalization import (
        normalize_str,
        normalize_ds_str,
        to_millions,
        to_billions,
        to_trillions,
        normalize_compute,
        normalize_tokens,
    )
    from dr_ingest.parallel import parallel_process, list_merge, set_merge
    from dr_ingest.raw_download import (
        str_list_to_dicts,
        dict_list_to_all_keys,
        make_struct_dtype,
        parse_dd_results_train,
        dd_results_parse_fxn,
        get_hf_download_path,
        get_hf_fs,
        pl_load_parquet_from_hf,
        DD_RESULTS_REPO,
        DD_RES_NAMES,
        DD_RES_OTHER_PATH_FORMAT_STR,
    )


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""## Results Parse""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""### Load Data from Local Download (datadec/data/raw_downloads/)""")
    return


@app.cell
def _():
    run_results_parse = True
    if run_results_parse:
        repos_dir = Path("/Users/daniellerothermel/drotherm/repos")
        raw_downloads_dir = Path(repos_dir / "datadec/data/raw_downloads")
        eval_res_dir = Path(raw_downloads_dir / "DD-eval-results/data" )
        all_res_files = [
            str(Path(eval_res_dir / f"train-0000{i}-of-00004.parquet")) for i in range(4)
        ]
        mo.output.append(all_res_files)
        all_res_dfs = [pd.read_parquet(f) for f in all_res_files]

        test_df = pl.read_parquet(all_res_files)
        test_df_pd = pd.concat(all_res_dfs, ignore_index=True)
        mo.output.append(test_df)
    return run_results_parse, test_df


@app.cell(hide_code=True)
def _():
    mo.md(r"""### See Normalization Utils on All Unique""")
    return


@app.cell
def _(run_results_parse, test_df):
    if run_results_parse:
        print(">> Unique Tokens (and normalized)")
        for tok in test_df["tokens"].unique().to_list():
            nt = normalize_tokens(tok)
            print(f"{tok:e} | {nt:.2f} M")

        print()
        print(">> Unique Compute (and normalized)")
        for comp in test_df["compute"].unique().to_list():
            nc = normalize_compute(comp)
            print(f"{comp:e} | {comp / 1e15:.2f} | {nc:.2f} e15")

        print()
        print(">> Unique recipes (and normalized)")
        for _ds in test_df["data"].unique().to_list():
            print(f"{_ds:35} | {normalize_str(_ds):35} | {normalize_ds_str(_ds):35}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""### Parse Results -> this is where something breaks for everything but primary_metric""")
    return


@app.cell
def _(run_results_parse, test_df):
    if run_results_parse:
        tdf = test_df.to_pandas()
        df_short = tdf[tdf['params'] == '1B']
        mo.output.append(df_short)
        #metrics_short = df_short["metrics"].to_list()
        #outputs_tmp = {"train": test_df.slice(0, 10)}
        #parsed_gen = dd_results_parse_fxn(outputs_tmp)
        #mo.vstack([
        #    outputs_tmp,
        #    len(metrics_short),
        #    metrics_short[:3],
        #    next(iter(parsed_gen))
        #])
    return


@app.cell
def _():
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""## Non-Results DD Parse""")
    return


@app.cell
def _():
    run_non_results_parse = False
    if run_non_results_parse:
        test_ma_df = pl.read_parquet(
            "/Users/daniellerothermel/drotherm/repos/datadec/data/raw_downloads/DD-eval-results/data/macro_avg-00000-of-00001.parquet",
        )
        test_ma_df
    return run_non_results_parse, test_ma_df


@app.cell
def _(run_non_results_parse, test_sl_df):
    if run_non_results_parse:
        for mix in test_sl_df["mix"].unique().to_list():
            print(f'"{normalize_ds_str(mix)}",')
    return


@app.cell
def _(run_non_results_parse, test_ma_df):
    if run_non_results_parse:
        test_sl_df = pl.read_parquet(
            "/Users/daniellerothermel/drotherm/repos/datadec/data/raw_downloads/DD-eval-results/data/scaling_law_fit-00000-of-00001.parquet",
        )
        mo.vstack([
            test_sl_df,
            test_sl_df["setup"].unique().to_list()
        ])
        for _s in test_ma_df["seed"].unique().to_list():
            print(f"{_s:15} | {normalize_str(_s, final_delim='_'):15}")
    return (test_sl_df,)


@app.cell(column=3, hide_code=True)
def _():
    mo.md(r"""## Load from HF""")
    return


@app.cell
def _():
    run_load_from_hf = False
    if run_load_from_hf:
        fs = get_hf_fs()
        other_0_df = pl_load_parquet_from_hf(fs, hfp)
        other_1_df = pl_load_parquet_from_hf(fs, hfp2)
        filtered = other_1_df.filter(
            (~pl.col("setup").str.contains("intermediate"))
            & (pl.col("setup").str.contains("1_step"))
            & (~pl.col("step_1_pred").is_close(pl.col("stacked_pred")))
        )
        hfp = get_hf_download_path(
            DD_RESULTS_REPO, DD_RES_OTHER_PATH_FORMAT_STR.format(DD_RES_NAMES[0])
        )
        hfp2 = get_hf_download_path(
            DD_RESULTS_REPO, DD_RES_OTHER_PATH_FORMAT_STR.format(DD_RES_NAMES[1])
        )


        mo.vstack([
            other_1_df,
            filtered,
            hfp,
            hfp2,
        ])
    return hfp, hfp2


@app.cell
def _():
    return


@app.cell(column=4)
def _():
    return


if __name__ == "__main__":
    app.run()
