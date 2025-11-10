from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel, Literal

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
    ) -> str | None:
        if explicit:
            return explicit
        name = getattr(self, f"{which}_env_var", "")
        return env.get(name) if name else None
