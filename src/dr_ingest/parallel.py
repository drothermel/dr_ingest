from __future__ import annotations

from collections.abc import Callable

import multiprocess as mp
from more_itertools import chunked


def get_chunk_setup(
    len_input: int, num_cores: int | None = None
) -> tuple[int, list[int]]:
    if num_cores is None:
        num_cores = mp.cpu_count()
    chunk_size = len_input // num_cores
    return num_cores, chunk_size


def parallel_process(
    input_list: list,
    process_fxn: Callable,
    merge_fxn: Callable,
    num_cores: int | None = None,
) -> list:
    num_items = len(input_list)
    num_cores, chunk_size = get_chunk_setup(num_items, num_cores)
    print(f">> Processing {num_items} items in {num_cores} chunks of size {chunk_size}")
    chunks = list(chunked(input_list, chunk_size))
    with mp.get_context("spawn").Pool(num_cores) as pool:
        results = pool.map(process_fxn, chunks)
    return merge_fxn(results)


def list_merge(x: list[list]) -> list:
    return [item for sublist in x for item in sublist]


def set_merge(x: list[set]) -> set:
    return set.union(*x)
