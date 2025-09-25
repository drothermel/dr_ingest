"""Utilities for working with QA evaluation tarballs."""

from __future__ import annotations

import tarfile
from pathlib import Path


def list_tarballs(root_dir: Path, data: str, params: str, seed: int) -> list[Path]:
    """Return sorted tar.gz files for the given dataset/config tuple."""

    target_dir = root_dir / data / params / f"seed-{seed}"
    if not target_dir.exists():
        return []
    return sorted(target_dir.glob("*.tar.gz"))


def ensure_extracted(tar_path: Path, dest_root: Path) -> Path:
    """Extract ``tar_path`` into ``dest_root`` if not already present."""

    target_dir = dest_root / tar_path.stem.replace(".tar", "")
    if target_dir.exists():
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as archive:
        archive.extractall(path=target_dir, filter="data")
    return target_dir


__all__ = ["ensure_extracted", "list_tarballs"]
