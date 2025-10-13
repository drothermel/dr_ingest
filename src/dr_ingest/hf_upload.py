from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi


def upload_file_to_hf(
    local_path: Path,
    repo_id: str,
    path_in_repo: str,
    token: str,
    repo_type: str = "dataset",
) -> None:
    """Upload a single file to Hugging Face Hub."""
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(local_path),
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        repo_type=repo_type,
        token=token,
    )
