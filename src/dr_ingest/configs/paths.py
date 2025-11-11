from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler
from pydantic_core import core_schema

from dr_ingest.utils import add_marimo_display

__all__ = ["ExistingPath", "Paths"]


# TODO: Add this to a shared utils repo
class ExistingPath(Path):
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: type[Any],
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        base_schema = handler(Path)
        return core_schema.no_info_after_validator_function(
            cls.validate_exists, base_schema
        )

    @classmethod
    def validate_exists(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"Path does not exist: {value}")
        return value


@add_marimo_display()
class Paths(BaseModel):
    model_config = ConfigDict(validate_default=True)
    username: str = "drotherm"

    # Base directories: used to build actually used paths
    repos_dir: Path = Field(
        default_factory=lambda data: Path.home() / data["username"] / "repos"
    )
    data_dir: Path = Field(
        default_factory=lambda data: Path.home() / data["username"] / "data"
    )

    # Actually used paths, repo_root and data_cache_dir must exist
    repo_root: ExistingPath = Field(
        default_factory=lambda data: data["repos_dir"] / "dr_ingest"
    )
    data_cache_dir: ExistingPath = Field(
        default_factory=lambda data: data["data_dir"] / "cache"
    )
    metrics_all_dir: Path = Field(
        default_factory=lambda data: data["data_dir"]
        / "datadec"
        / "2025-10-08_posttrain"
    )
