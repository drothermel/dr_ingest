"""Connection helpers for external data sources."""

from __future__ import annotations

import duckdb

from dr_ingest.configs import AuthSettings, Paths

__all__ = [
    "open_motherduck_connection",
]


def open_motherduck_connection(
    *,
    paths: Paths | None = None,
    auth: AuthSettings | None = None,
) -> duckdb.DuckDBPyConnection:
    """Open a MotherDuck connection if credentials are available."""

    env_file = (paths or Paths()).repo_root / ".env"
    auth = auth or AuthSettings()
    md_token = auth.resolve("motherduck", dotfile_loc=str(env_file))
    assert md_token, f"MotherDuck token not found in .env file: {env_file}"
    conn = duckdb.connect(f"md:?motherduck_token={md_token}")

    hf_token = auth.resolve("hf", dotfile_loc=str(env_file))
    conn.execute(
        f"""
        CREATE SECRET IF NOT EXISTS hf_token (
            TYPE HUGGINGFACE,
            TOKEN '{hf_token}'
        );
        """
    )
    return conn
