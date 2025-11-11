from collections.abc import Iterable, Iterator
from pathlib import Path

__all__ = ["iter_file_glob_from_roots"]


def iter_file_glob_from_roots(
    root_paths: Iterable[Path | str],
    file_glob: str,
) -> Iterator[Path]:
    for path in root_paths:
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"Root path not found: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Root path is not a directory: {root}")
        yield from root.rglob(file_glob)
