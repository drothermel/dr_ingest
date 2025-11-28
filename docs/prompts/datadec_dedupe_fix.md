# Prompt: Fix DataDec export so dedupe works

## Context
- We're caching DataDec eval directories via `uv run ingest-datadec cache ...` into `/Users/daniellerothermel/drotherm/data/cache/new_ingest_v1` (Nov 03 dump) and `/Users/daniellerothermel/drotherm/data/cache/new_ingest_v1_old` (Oct 08 dump).
- The pipeline now joins per-task metadata from `metrics-all.jsonl` only; no `config.json` / `metrics.json` files are needed.
- Dedupe logic (`uv run ingest-datadec dedupe ...`) reads all Parquets, normalizes `met_task_config` (dropping `task_name`), and keeps the first row per normalized config.
- Dedupe currently fails because some columns DuckDB stores as `JSON` (e.g., `prd_native_id`, `req_native_id`, etc.) contain plain strings like `allenai/DataDecide-dclm-baseline-1B`. When dedupe queries those columns, DuckDB tries to parse them as JSON and raises “Malformed JSON at byte 0 …”.
- Those columns originate from the export stage in `src/dr_ingest/pipelines/datadec_ingest.py` when we copy per-task artifacts directly; we never coerce strings into proper JSON.

## Task for the follow-up agent
1. Update the export logic (in `export_eval_dir_to_parquet` or earlier helpers) to ensure any column that DuckDB writes with `JSON` type contains valid JSON text (e.g., use DuckDB’s `TO_JSON`/`sql` functions, or wrap Python-side via `json.dumps`). Alternatively, convert those problematic fields to `VARCHAR`/`STRUCT` before writing Parquet.
2. Re-run the doc-id 0 cache commands:
   - `uv run ingest-datadec cache --metrics-dir /Users/daniellerothermel/drotherm/data/datadec/2025-11-03_posttrain/ --cache-dir /Users/daniellerothermel/drotherm/data/cache/new_ingest_v1 --doc-id 0 --slug-prefix 11-03_ --force`
   - `uv run ingest-datadec cache --metrics-dir /Users/daniellerothermel/drotherm/data/datadec/2025-10-08_posttrain --cache-dir /Users/daniellerothermel/drotherm/data/cache/new_ingest_v1_old --doc-id 0 --slug-prefix 10_08_ --force`
3. Re-run dedupe (`uv run ingest-datadec dedupe --cache-dir <cache> --output-name deduped.parquet`) to confirm it succeeds with the new exports.
4. Update documentation (e.g., `docs/notes/datadec_metrics_all/ARTIFACT_NOTES.md`) if the export assumptions change.

## Notes
- Inspect columns via `duckdb`: `DESCRIBE SELECT * FROM read_parquet('path.parquet')` to confirm types.
- Focus on columns typed as `JSON`; these currently contain plain strings and need serialization fixes.
- Keep doc-id filtering behavior intact.
