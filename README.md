# dr_ingest

Shared ingestion utilities for the DataDecide ecosystem. This package will host canonical WandB parsing, DuckDB ingestion helpers, and supporting CLIs.

Refer to the documentation hub in `dr_ref` for onboarding details and design notes.

## Configuration

- WandB ingestion settings live under `configs/wandb/` and are merged in this order: `base.cfg`, `patterns.cfg`, `processing.cfg`, `hooks.cfg`.
- Add new sections to the appropriate file (e.g. regex specs in `patterns.cfg`) to keep the Python loader declarative.
