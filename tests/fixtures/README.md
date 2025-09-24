# dr_ingest Fixture Assets

Fixtures in this directory are distilled slices of the canonical WandB and pretraining datasets. They are regenerated via scripts in the same folder so unit tests and demos can exercise real-world patterns without loading the full corpora.

## Contents
- `build_wandb_samples.py`: Extracts a curated subset of runs, history entries, and pretraining rows from the upstream datasets.
- `data/`:
  - `wandb_runs_sample.parquet`
  - `wandb_history_sample.parquet`
  - `pretrain_metrics_sample.parquet`
  - `manifest.json`

## Regeneration
```bash
uv run python tests/fixtures/build_wandb_samples.py
```
Ensure the source data exists at:
- `../datadec/notebooks/wandb_runs.jsonl`
- `../datadec/notebooks/wandb_history.parquet`
- `../datadec/data/datadecide/full_eval.parquet`

## Pending Work
Automated “expected output” snapshots (classification JSON and processed tables) are not yet generated because the downstream enrichment logic is still in flux. Once the parsing/merge pipeline stabilizes, add a script that runs the shared module on these fixtures and stores the expected outputs under `data/` for regression tests.
