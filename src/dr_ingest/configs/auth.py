from pydantic import BaseModel

from dr_ingest.utils import add_marimo_display


@add_marimo_display()
class AuthSettings(BaseModel):
    motherduck_env_var: str = "MOTHERDUCK_TOKEN"
    hf_env_var: str = "HF_TOKEN"
