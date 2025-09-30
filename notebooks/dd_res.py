import marimo

__generated_with = "0.16.2"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    return


@app.cell(column=1)
def _():
    import marimo as mo
    import polars as pl
    return mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Helpers""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Execution""")
    return


@app.cell
def _(pl):
    test_ma_df = pl.read_parquet(
        "/Users/daniellerothermel/drotherm/repos/datadec/data/raw_downloads/DD-eval-results/data/macro_avg-00000-of-00001.parquet",
    )
    test_ma_df
    return (test_ma_df,)


@app.cell
def _(pl):
    test_sl_df = pl.read_parquet(
        "/Users/daniellerothermel/drotherm/repos/datadec/data/raw_downloads/DD-eval-results/data/scaling_law_fit-00000-of-00001.parquet",
    )
    test_sl_df
    return (test_sl_df,)


@app.cell
def _(test_sl_df):
    test_sl_df["setup"].unique().to_list()
    return


@app.cell
def _(normalize_str, test_ma_df):
    for s in test_ma_df["seed"].unique().to_list():
        print(f"{s:15} | {normalize_str(s, final_delim='_'):15}")
    return


@app.cell
def _(pl):
    test_df = pl.read_parquet(
        [
            "/Users/daniellerothermel/drotherm/repos/datadec/data/raw_downloads/DD-eval-results/data/train-00000-of-00004.parquet",
            "/Users/daniellerothermel/drotherm/repos/datadec/data/raw_downloads/DD-eval-results/data/train-00001-of-00004.parquet",
            "/Users/daniellerothermel/drotherm/repos/datadec/data/raw_downloads/DD-eval-results/data/train-00002-of-00004.parquet",
            "/Users/daniellerothermel/drotherm/repos/datadec/data/raw_downloads/DD-eval-results/data/train-00003-of-00004.parquet",
        ]
    )
    test_df
    return (test_df,)


@app.cell
def _():
    from dr_ingest.normalization import (
        normalize_str,
        normalize_ds_str,
        to_millions,
        to_billions,
        to_trillions,
        normalize_compute,
        normalize_tokens,
    )
    return normalize_compute, normalize_ds_str, normalize_str, normalize_tokens


@app.cell
def _(normalize_compute, test_df):
    for comp in test_df["compute"].unique().to_list():
        nc, usm = normalize_compute(comp)
        print(f"{comp:e} | {comp / 1e15:.2f} | {nc:.2f} {usm}")
    return


@app.cell
def _(normalize_tokens, test_df):
    for tok in test_df["tokens"].unique().to_list():
        nt, ust = normalize_tokens(tok)
        print(f"{tok:e} | {nt:.2f} {ust}")
    return


@app.cell
def _(normalize_ds_str, test_sl_df):
    for mix in test_sl_df["mix"].unique().to_list():
        print(f'"{normalize_ds_str(mix)}",')
    return


@app.cell
def _(normalize_ds_str, normalize_str, test_df):
    for ds in test_df["data"].unique().to_list():
        print(f"{ds:35} | {normalize_str(ds):35} | {normalize_ds_str(ds):35}")
    return


@app.cell
def _(test_df):
    for ms in test_df["params"].unique().to_list():
        print(f'"{ms}",')
    return


@app.cell
def _(test_df):
    df_short = test_df.slice(0, 10_000)
    return (df_short,)


@app.cell
def _(df_short):
    metrics_short = df_short["metrics"].to_list()
    len(metrics_short), metrics_short[:3]
    return


@app.cell
def _():
    from dr_ingest.parallel import parallel_process, list_merge, set_merge
    return


@app.cell
def _():
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
    return (
        DD_RESULTS_REPO,
        DD_RES_NAMES,
        DD_RES_OTHER_PATH_FORMAT_STR,
        dd_results_parse_fxn,
        get_hf_download_path,
        get_hf_fs,
        pl_load_parquet_from_hf,
    )


@app.cell
def _(test_df):
    outputs_tmp = {"train": test_df.slice(0, 10)}
    outputs_tmp
    return (outputs_tmp,)


@app.cell
def _(dd_results_parse_fxn, outputs_tmp):
    parsed_gen = dd_results_parse_fxn(outputs_tmp)
    return (parsed_gen,)


@app.cell
def _(parsed_gen):
    next(iter(parsed_gen))
    return


@app.cell
def _(
    DD_RESULTS_REPO,
    DD_RES_NAMES,
    DD_RES_OTHER_PATH_FORMAT_STR,
    get_hf_download_path,
    get_hf_fs,
):
    fs = get_hf_fs()
    hfp = get_hf_download_path(
        DD_RESULTS_REPO, DD_RES_OTHER_PATH_FORMAT_STR.format(DD_RES_NAMES[0])
    )
    print(hfp)
    return fs, hfp


@app.cell
def _(fs, hfp, mo, pl_load_parquet_from_hf):
    mo.stop(True, "pause")
    other_0_df = pl_load_parquet_from_hf(fs, hfp)
    return (other_0_df,)


@app.cell
def _(other_0_df):
    other_0_df
    return


@app.cell
def _(
    DD_RESULTS_REPO,
    DD_RES_NAMES,
    DD_RES_OTHER_PATH_FORMAT_STR,
    get_hf_download_path,
):
    hfp2 = get_hf_download_path(
        DD_RESULTS_REPO, DD_RES_OTHER_PATH_FORMAT_STR.format(DD_RES_NAMES[1])
    )
    print(hfp2)
    return (hfp2,)


@app.cell
def _(fs, hfp2, mo, pl_load_parquet_from_hf):
    mo.stop(True, "pause")
    other_1_df = pl_load_parquet_from_hf(fs, hfp2)
    other_1_df
    return (other_1_df,)


@app.cell
def _(other_1_df, pl):
    filtered = other_1_df.filter(
        (~pl.col("setup").str.contains("intermediate"))
        & (pl.col("setup").str.contains("1_step"))
        & (~pl.col("step_1_pred").is_close(pl.col("stacked_pred")))
    )
    filtered
    return


@app.cell
def _():
    return


@app.cell(column=2)
def _():
    return


if __name__ == "__main__":
    app.run()
