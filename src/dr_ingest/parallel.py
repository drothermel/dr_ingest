from __future__ import annotations

from collections.abc import Callable
import math

import multiprocess as mp
from more_itertools import chunked


def get_chunk_setup(
    len_input: int, num_cores: int | None = None
) -> tuple[int, int]:
    if num_cores is None:
        num_cores = mp.cpu_count()
    if len_input == 0:
        return 1, 1
    num_cores = max(1, min(num_cores, len_input))
    if len_input <= 32:
        num_cores = 1
    chunk_size = max(1, math.ceil(len_input / num_cores))
    return num_cores, chunk_size


def parallel_process(
    input_list: list,
    process_fxn: Callable,
    merge_fxn: Callable,
    num_cores: int | None = None,
) -> list:
    num_items = len(input_list)
    if num_items == 0:
        return merge_fxn([])
    num_cores, chunk_size = get_chunk_setup(num_items, num_cores)
    print(f">> Processing {num_items} items in {num_cores} chunks of size {chunk_size}")
    chunks = list(chunked(input_list, chunk_size))
    if len(chunks) == 1:
        results = [process_fxn(chunks[0])]
    else:
        with mp.get_context("spawn").Pool(num_cores) as pool:
            results = pool.map(process_fxn, chunks)
    return merge_fxn(results)


def list_merge(x: list[list]) -> list:
    return [item for sublist in x for item in sublist]


def set_merge(x: list[set]) -> set:
    if not x:
        return set()
    return set.union(*x)
