from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from dr_ingest.utils import add_marimo_display

__all__ = ["Paths"]


@add_marimo_display()
class Paths(BaseModel):
    username: str = "drotherm"

    # Base directories: used to build actually used paths
    repos_dir: Path = Field(
        default_factory=lambda data: Path.home() / data["username"] / "repos"
    )
    data_dir: Path = Field(
        default_factory=lambda data: Path.home() / data["username"] / "data"
    )

    # Actually used paths, repo_root and data_cache_dir must exist
    repo_root: Path = Field(
        default_factory=lambda data: data["repos_dir"] / "dr_ingest"
    )
    data_cache_dir: Path = Field(
        default_factory=lambda data: data["data_dir"] / "cache"
    )
    metrics_all_dir: Path = Field(
        default_factory=lambda data: data["data_dir"] / "datadec"
    )

    @model_validator(mode="before")
    def validate_path_types(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(value, Path | str):
                    raise ValueError(f"Invalid path type for {key}: {type(value)}")
                data[key] = Path(value)
        return data
