# DataDec Metrics-All Artifact Notes

Context below is based on the directory `/Users/daniellerothermel/drotherm/data/datadec/2025-11-03_posttrain/_eval_results/meta-llama_Llama-3.1-8B__main/`, which is representative of recent dumps.

## Artifact Types

### `metrics-all.jsonl`
- One JSON object per evaluated task (including aggregate bundles like `core_9mcqa:rc::olmes:full`).
- Top-level keys: `task_name`, `task_hash`, `model_hash`, `model_config`, `task_config`, `compute_config`, `processing_time`, `current_date`, `num_instances`, `beaker_info`, `metrics`, `task_idx`.
- Each object is exactly duplicated in the corresponding per-task `*-metrics.json` file when such a file exists. `metrics-all.jsonl` is therefore sufficient to reconstruct both `config.json` and `metrics.json` contents without touching the per-task artifacts.
- Aggregate entries (e.g., `core_9mcqa`, `mmlu`, `bbh`, `minerva`) appear **only** in this file—no matching `task-XXX` directories are emitted for them.
- Old-format dumps (e.g., `2025-10-08_posttrain/...`) only shipped `metrics-all.jsonl`; the schema is identical to the new dumps. You can recreate missing per-task `config.json` and `metrics.json` files by extracting `task_config` (for configs) or writing the entire record (for metrics). Mapping a record back to its artifact directory should be done via `task_name`, since folder prefixes like `task-002-...` are reused across unrelated benchmarks.
- Those older dumps may have fewer rows—only the “base” task for a regime might be present, even if multiple task variants were executed. In such cases there is no metadata entry to reconstruct, and the per-task artifacts remain the only source of information.

### Per-task `*-metrics.json`
- Stored alongside predictions/requests for each task shard (e.g., `task-002-boolq:mc::olmes-metrics.json`).
- Byte-for-byte identical to the matching JSON line in `metrics-all.jsonl`. This includes nested `model_config`, `task_config`, and measurement stats.
- Top-level keys are the same twelve keys listed above for `metrics-all.jsonl`.

### Per-task `*-config.json`
- Contains only the `task_config` portion of the metrics entry.
- Observed keys: `task_name`, `task_core`, `limit`, `split`, `num_shots`, `fewshot_seed`, `primary_metric`, `random_subsample_seed`, `context_kwargs`, `generation_kwargs`, `metric_kwargs`, `native_id_field`, `fewshot_source`, `dataset_path`, `dataset_name`, `use_chat_format`, `version`, `revision`, `compute_gold_bpb`, `external_eval`, `custom_kwargs`, `skip_model_judges`, `model_max_length`, `metadata`.
- Because this payload is already embedded inside every metrics record and `metrics-all.jsonl`, it does **not** add unique information.

### `metrics.jsonl`
- Current dump contains 36 newline-delimited strings repeating `"all_primary_scores"`, `"tasks"`, `"model_config"`. It appears to be a truncated or vestigial file; no structured JSON objects are available here. Nothing in this file is needed if `metrics-all.jsonl` is present.

### Per-task `*-predictions.jsonl`
- Row-level scores and model outputs per document.
- Top-level keys: `doc_id`, `native_id`, `metrics` (per-instance variants of the task metrics), `model_output` (list of token-level stats per choice/loglik request), `label`, `task_hash`, `model_hash`.

### Per-task `*-recorded-inputs.jsonl`
- Captures the prompts/context fed to the model. Each line has: `doc` (original dataset entry with question/choices), `task_name`, `doc_id`, `native_id`, `label`, `requests` (list mirroring what went over the wire, each with `request_type`, `request`, `idx`).

### Per-task `*-requests.jsonl`
- Raw request payloads, top-level keys: `request_type`, `doc` (same schema as above), `request` (serialized message sent to the model runner), `idx`, `task_name`, `doc_id`, `native_id`, `label`.

## Key Findings
1. `metrics-all.jsonl` is the canonical source: every per-task `*-metrics.json` object and `*-config.json` object is directly derivable from it, so storing all three is redundant.
2. Aggregate tasks only exist in `metrics-all.jsonl`; relying on per-task folders would miss those cross-task scores.
3. Other per-task artifacts (`predictions`, `recorded-inputs`, `requests`) contain additional row-level data not represented in the metrics files, so they remain necessary when analyzing individual examples.
4. The ingest cache now serializes nested objects such as `doc`, `request`, `requests`, `model_output`, `metrics`, and the various `*_config` maps into JSON strings before writing Parquet so DuckDB sees consistent `VARCHAR` types across dumps; downstream consumers should `json.loads` these columns if they need the structured payloads.
