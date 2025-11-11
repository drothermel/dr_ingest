"""Pydantic configuration helpers for Hugging Face repositories."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Annotated, ClassVar

from pydantic import BaseModel, Field, HttpUrl, computed_field, field_validator

from dr_ingest.utils import add_marimo_display

__all__ = ["HFLocation", "HFRepoID", "HFResource"]

HFRepoID = Annotated[
    str,
    Field(description="Hugging Face repo ID (e.g. allenai/DataDecide-eval-results)"),
]

HFResource = Annotated[
    str,
    Field(description="Hugging Face resource URI (e.g. hf://datasets/...)"),
]


@add_marimo_display()
class HFLocation(BaseModel):
    uri_prefix: ClassVar[str] = "hf://"
    repo_type: ClassVar[str] = "datasets"
    repo_type_aliases: ClassVar[tuple[str, ...]] = ("dataset", "datasets")

    org: str
    repo_name: str
    filepaths: list[str] | None = None

    local_path_include_org: bool = True
    local_path_include_repo: bool = True

    # --- computed fields -------------------------------------------------

    @computed_field
    @property
    def repo_id(self) -> HFRepoID:
        return f"{self.org}/{self.repo_name}"

    @computed_field
    @property
    def repo_uri(self) -> HFResource:
        return f"hf://{self.repo_type}/{self.repo_id}"

    @computed_field
    @property
    def repo_link(self) -> HttpUrl:
        return HttpUrl(f"https://huggingface.co/{self.repo_type}/{self.repo_id}")

    @computed_field
    @property
    def rest_api_repo_url(self) -> HttpUrl:
        api_base_url = "https://huggingface.co/api"
        return HttpUrl(f"{api_base_url}/{self.repo_type}/{self.repo_id}")

    @computed_field
    @property
    def hf_hub_repo_type(self) -> str:
        if self.repo_type == "datasets":
            return "dataset"
        raise ValueError(f"Invalid repo type: {self.repo_type}")

    # --- constructors ----------------------------------------------------
    @classmethod
    def from_uri(
        cls,
        uri: str,
        *,
        filepaths: Iterable[str] | None = None,
    ) -> HFLocation:
        """Construct a location from an ``hf://`` URI."""
        if not uri or not isinstance(uri, str):
            raise ValueError("HF URI must be a non-empty string.")
        prefix = cls.uri_prefix
        if uri.startswith(prefix):
            stripped = uri[len(prefix) :]
        else:
            raise ValueError(f"HF URI must start with '{prefix}'. Got: {uri!r}")

        parts = [part for part in stripped.split("/") if part]
        if len(parts) < 2:
            raise ValueError(
                "HF URI must include at least org and repo, e.g. "
                "'hf://datasets/org/repo'."
            )

        expected_repo_type = cls.repo_type
        alias_map = {expected_repo_type: expected_repo_type}
        alias_map.update(dict.fromkeys(cls.repo_type_aliases, expected_repo_type))

        potential_repo_type = parts[0]
        normalized = alias_map.get(potential_repo_type)
        if normalized:
            if normalized != expected_repo_type:
                raise ValueError(
                    f"HF URI repo type '{potential_repo_type}' does not match expected "
                    f"'{expected_repo_type}'."
                )
            org_repo_parts = parts[1:]
        elif len(parts) == 2:
            org_repo_parts = parts
        else:
            raise ValueError(
                "HF URI with nested paths must start with 'hf://datasets/' or 'hf://dataset/'."
            )

        if len(org_repo_parts) < 2:
            raise ValueError(
                "HF URI must include both org and repo names, e.g. "
                "'hf://datasets/org/repo'."
            )
        org, repo_name, *sub_path = org_repo_parts
        resolved_paths: list[str] = list(filepaths or [])
        if sub_path:
            resolved_paths.append("/".join(sub_path))
        return cls(
            org=org,
            repo_name=repo_name,
            filepaths=resolved_paths or None,
        )

    @field_validator("filepaths")
    @classmethod
    def posix_norm_filepaths(cls, v: list[str | Path] | None) -> list[str] | None:
        if v is None:
            return None
        return [cls.norm_posix(p) for p in v]

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def norm_posix(path: str | Path) -> str:
        """Ensure repo-internal, forward-slashed, no leading slash."""
        p = PurePosixPath(str(path))
        return str(p).lstrip("/")

    @staticmethod
    def _is_dir(path: str | Path) -> bool:
        return str(path).endswith("/") or not Path(path).suffix

    def get_the_single_filepath(self, local_dir: str | Path | None = None) -> str:
        return self.resolve_filepaths(local_dir=local_dir, expect_one=True)[0]

    def build_local_dir(self, local_dir: str | Path | None = None) -> str | None:
        dir_parts = [str(local_dir or "")]
        if self.local_path_include_org:
            dir_parts.append(self.org)
        if self.local_path_include_repo:
            dir_parts.append(self.repo_name)
        return "/".join(dir_parts)

    def resolve_filepaths(
        self,
        extra_paths: list[str | Path] | None = None,
        local_dir: str | Path | None = None,
        required: bool = True,
        expect_one: bool = False,
    ) -> list[str]:
        # Preserve order and deduplicate
        seen = {}
        for path in [
            *(self.filepaths or []),
            *[self.norm_posix(p) for p in extra_paths or []],
        ]:
            if path in seen:
                continue
            seen[path] = path
        paths = list(seen.keys())
        if local_dir:
            paths = [f"{self.build_local_dir(local_dir)}/{path}" for path in paths]
        if required and not paths:
            raise ValueError("No filepaths found")
        if expect_one and len(paths) != 1:
            raise ValueError("Expected exactly one filepath")
        return paths

    def get_path_uri(self, path: str | Path) -> HFResource:
        """URI to reference the file in python APIs."""
        return f"{self.repo_uri}/{self.norm_posix(path)}"

    def get_path_link(self, path: str | Path) -> HttpUrl:
        """Web URL to view the file in the browser."""
        path = self.norm_posix(path)
        path_type = "tree" if self._is_dir(path) else "blob"
        return HttpUrl(f"{self.repo_link}/{path_type}/main/{path}")

    def get_rest_path_url(self, path: str | Path) -> HttpUrl:
        """REST API URL to view directory info."""
        path = self.norm_posix(path)
        if not self._is_dir(path):
            raise ValueError("REST endpoint only supports directories, not files.")
        return HttpUrl(f"{self.rest_api_repo_url}/tree/main/{path}")

    def get_file_download_link(self, filepath: str | Path) -> HttpUrl:
        """Direct download link for a single file."""
        path = self.norm_posix(filepath)
        if self._is_dir(path):
            raise ValueError("Download link only supports files, not directories.")
        return HttpUrl(f"{self.repo_link}/resolve/main/{path}")

    def get_uris_for_files(
        self,
        filepaths: Iterable[str | Path] | None = None,
        ignore_cfg_files: bool = False,
    ) -> list[HFResource]:
        items: list[str | Path] = list(filepaths or [])
        if not ignore_cfg_files:
            items.extend(self.filepaths or [])

        seen: dict[str, None] = {}
        for item in items:
            uri = self.get_path_uri(self.norm_posix(item))
            seen.setdefault(uri, None)
        return list(seen.keys())
