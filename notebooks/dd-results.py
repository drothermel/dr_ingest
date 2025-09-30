import marimo

__generated_with = "0.16.2"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    from collections.abc import Generator, Callable
    from typing import Any

    from dr_ingest.serilization import (
        ipc_serialize,
        ipc_deserialize,
        ipc_deserialize_all,
    )

    import polars as pl
    from datasets import Dataset, load_dataset
    from huggingface_hub import login, hf_hub_download, HfFileSystem
    import time
    import ast
    import json
    import multiprocess as mp
    from functools import partial
    import io

    DD_RESULTS_REPO = "allenai/DataDecide-eval-results"
    DD_NUM_TRAIN_FILES = 4
    DD_TRAIN_FILE_PATH_FORMAT_STR = "data/train-0000{}-of-00004.parquet"
    return (
        Any,
        Callable,
        DD_NUM_TRAIN_FILES,
        DD_RESULTS_REPO,
        DD_TRAIN_FILE_PATH_FORMAT_STR,
        Generator,
        HfFileSystem,
        ast,
        ipc_deserialize,
        ipc_deserialize_all,
        ipc_serialize,
        json,
        mo,
        mp,
        partial,
        pl,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Running Download + Parse Pipeline""")
    return


@app.cell
def _(mo):
    allow_hf_download = mo.ui.switch(
        value=False, label="Allow Re-download Train Results from HF"
    )
    return (allow_hf_download,)


@app.cell
def _(allow_hf_download):
    allow_hf_download
    return


@app.cell
def _(HfFileSystem, allow_hf_download, download_dd_train_res, mo, pl):
    mo.stop(
        not allow_hf_download.value,
        mo.md("**Update Detected:** Enable HF reactive re-download by toggling above."),
    )
    fs = HfFileSystem()
    all_dfs = []
    all_info = []
    for df, info in download_dd_train_res(fs):
        all_dfs.append(df)
        all_info.append(info)
    final_df = pl.concat(all_dfs, how="vertical")
    all_elapsed_times = [t["download_time"] for t in all_info]
    return all_dfs, all_elapsed_times, final_df


@app.cell
def _(final_df, parse_dd_train_res):
    parse_dd_train_res(final_df.head(20))
    return


@app.cell
def _(all_dfs, all_elapsed_times, final_df, mo):
    full_shape = f"{final_df.shape[0]:,}"
    partial_shapes = [f"{pdf.shape[0]:,}" for pdf in all_dfs]
    sum_elapsed_str = sum(all_elapsed_times)
    elapsed_strs = [f"{t:.2f}" for t in all_elapsed_times]
    mo.vstack(
        [
            mo.md(f"""
            **Total rows:** `{full_shape}` `[{", ".join(partial_shapes)}]`
        
            **Total time:** `{sum_elapsed_str:.2f}` seconds `[{", ".join(elapsed_strs)}]`
            """),
            final_df,
        ]
    )
    return


@app.cell
def _(chunk_literal_to_json, final_df):
    literal_small = chunk_literal_to_json(final_df.head(10_000), "metrics")
    literal_small
    return (literal_small,)


@app.cell
def _(chunk_json_to_keys, literal_small):
    keys_small = chunk_json_to_keys(literal_small, "metrics_json")
    keys_small
    return (keys_small,)


@app.cell
def _(keys_small, make_struct_dtype, pl):
    struct_dtype = make_struct_dtype(keys_small, [pl.Float64] * len(keys_small))
    struct_dtype
    return (struct_dtype,)


@app.cell
def _(decode_to_struct, literal_small, struct_dtype):
    decode_to_struct(literal_small, "metrics_json", struct_dtype)
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""## Helper Functions""")
    return


@app.function
def get_hf_download_path(repo: str, filepath: str) -> str:
    return f"hf://datasets/{repo}/{filepath}"


@app.cell
def _(HfFileSystem, pl):
    def pl_load_parquet_from_hf(fs: HfFileSystem, path: str) -> pl.DataFrame:
        with fs.open(path, "rb") as f:
            return pl.read_parquet(f)
    return (pl_load_parquet_from_hf,)


@app.cell
def _(
    DD_NUM_TRAIN_FILES,
    DD_RESULTS_REPO,
    DD_TRAIN_FILE_PATH_FORMAT_STR,
    Generator,
    HfFileSystem,
    pl,
    pl_load_parquet_from_hf,
    time,
):
    def download_dd_train_res(
        fs: HfFileSystem,
    ) -> Generator[tuple[pl.DataFrame, dict], None, None]:
        for i in range(DD_NUM_TRAIN_FILES):
            fp = DD_TRAIN_FILE_PATH_FORMAT_STR.format(i)
            hf_path = get_hf_download_path(DD_RESULTS_REPO, fp)
            start = time.time()
            df = pl_load_parquet_from_hf(fs, hf_path)
            elapsed = time.time() - start
            print(f">> Read {hf_path} in {elapsed:.2f} seconds")
            yield df, {"download_time": elapsed}
    return (download_dd_train_res,)


@app.cell
def _(ast, json, pl):
    def pl_literal_to_json(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
        return df.with_columns(
            pl.col(col_name)
            .map_elements(lambda x: json.dumps(ast.literal_eval(x)), return_dtype=pl.String)
            .alias(f"{col_name}_json")
        )
    return (pl_literal_to_json,)


@app.cell
def _(json, pl):
    def pl_json_to_keys(df: pl.DataFrame, col_name: str) -> set[str]:
        all_keys = set()
        for metrics in df[col_name]:
            all_keys.update(json.loads(metrics).keys())
        return pl.DataFrame({"keys": all_keys})
    return (pl_json_to_keys,)


@app.cell
def _(Callable, ipc_deserialize, ipc_serialize):
    def chunk_process_polars(chunk_bytes, fxn: Callable):
        return ipc_serialize(fxn(ipc_deserialize(chunk_bytes)))
    return (chunk_process_polars,)


@app.cell
def _(
    Callable,
    chunk_process_polars,
    ipc_deserialize_all,
    ipc_serialize,
    mp,
    partial,
    pl,
):
    def parallel_process_polars(df: pl.DataFrame, fxn: Callable, n_cores=None):
        if n_cores is None:
            n_cores = mp.cpu_count()
        chunk_size = df.shape[0] // n_cores
        chunks = [df.slice(i * chunk_size, chunk_size) for i in range(n_cores)]
        ipc_chunks = [ipc_serialize(ch) for ch in chunks]
        ipc_fxn = partial(chunk_process_polars, fxn=fxn)

        with mp.get_context("spawn").Pool(n_cores) as pool:
            results_ipc = pool.map(ipc_fxn, ipc_chunks)
        return ipc_deserialize_all(results_ipc)
    return (parallel_process_polars,)


@app.cell
def _(Any, pl):
    def make_struct_dtype(keys: list[str], types: Any) -> pl.DataType:
        return pl.Struct([pl.Field(key, type) for key, type in zip(keys, types)])
    return (make_struct_dtype,)


@app.cell
def _(
    make_struct_dtype,
    parallel_process_polars,
    partial,
    pl,
    pl_json_to_keys,
):
    def parallel_extract_float_struct_dtype(df: pl.DataFrame, col_name: str) -> pl.DataType:
        json_to_keys_fxn = partial(pl_json_to_keys, col_name=col_name)
        keys = parallel_process_polars(df, json_to_keys_fxn).unique().to_list()
        return make_struct_dtype(keys, [pl.Float64] * len(keys))
    return (parallel_extract_float_struct_dtype,)


@app.cell
def _(pl):
    def decode_to_struct(
        df: pl.DataFrame, col_name: str, struct_dtype: pl.DataType
    ) -> pl.DataFrame:
        return df.with_columns(
            pl.col(col_name).str.json_decode(struct_dtype).alias(f"{col_name}_struct")
        )
    return (decode_to_struct,)


@app.cell
def _(pl):
    def replace_col_careful(
        df: pl.DataFrame,
        core_col: str,
        new_col_suffix: str,
    ):
        return df.drop(core_col).rename({f"{core_col}{new_col_suffix}": core_col})
    return (replace_col_careful,)


@app.cell
def _(
    decode_to_struct,
    parallel_extract_float_struct_dtype,
    parallel_process_polars,
    partial,
    pl,
    pl_literal_to_json,
    replace_col_careful,
):
    def parse_dd_train_res(
        df: pl.DataFrame,
    ):
        target_col = "metrics"
        literal_to_json_fxn = partial(pl_literal_to_json, col_name=target_col)
        rename_json_to_target = partial(replace_col_careful, target_col, "_json")
        rename_struct_to_target = partial(replace_col_careful, target_col, "_struct")

        df_json = df.pipe(parallel_process_polars, literal_to_json_fxn).pipe(
            rename_json_to_target
        )

        struct_dtype = parallel_extract_float_struct_dtype(df_json, target_col)

        return df_json.pipe(decode_to_struct, target_col, struct_dtype).pipe(
            rename_struct_to_target
        )
    return (parse_dd_train_res,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
