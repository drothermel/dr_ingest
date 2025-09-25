from __future__ import annotations

import tarfile
from pathlib import Path

from dr_ingest.qa import ensure_extracted, list_tarballs


def _create_tarball(tmp_path: Path, path: Path, name: str) -> Path:
    target_dir = tmp_path / path
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / f"{name}.txt"
    file_path.write_text("payload")
    tar_path = target_dir / f"{name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(file_path, arcname=file_path.name)
    return tar_path


def test_list_tarballs(tmp_path: Path) -> None:
    root = tmp_path / "root"
    tar_path = _create_tarball(tmp_path, Path("root/data/params/seed-1"), "sample")
    tarballs = list_tarballs(root, "data", "params", 1)
    assert tarballs == [tar_path]


def test_ensure_extracted(tmp_path: Path) -> None:
    tar_path = _create_tarball(tmp_path, Path("data.tar"), "example")
    dest_root = tmp_path / "extracted"
    extracted_dir = ensure_extracted(tar_path, dest_root)
    assert extracted_dir.exists()
    assert any(extracted_dir.iterdir())
    # second call should not re-extract but still return same path
    extracted_dir_2 = ensure_extracted(tar_path, dest_root)
    assert extracted_dir_2 == extracted_dir
