from __future__ import annotations

import os
from collections.abc import Mapping

from dotenv import load_dotenv
from pydantic import BaseModel

from dr_ingest.utils import add_marimo_display


@add_marimo_display()
class AuthSettings(BaseModel):
    motherduck_env_var: str = "MOTHERDUCK_TOKEN"
    hf_env_var: str = "HF_TOKEN"

    def resolve(
        self,
        which: str,
        explicit: str | None = None,
        env: Mapping[str, str] = os.environ,
        dotfile_loc: str | None = None,
    ) -> str | None:
        if explicit:
            return explicit
        if dotfile_loc:
            load_dotenv(dotfile_loc)
        name = getattr(self, f"{which}_env_var", "")
        return env.get(name) if name else None
