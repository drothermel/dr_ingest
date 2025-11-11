# dr_ingest

Shared ingestion utilities:
- Huggingface Downloads, Parsing and Parsed Uploads (demo'd by allenai/DataDecide-eval-results)
- [coming soon] LLM eval dumps with a central `metrics-all.jsonl` and then task artifact files
- [coming soon] wandb ingestion via `dr_wandb`

### Setup:
```
uv tool install dr_ingest
# -or-
uv add dr_ingest
uv sync
```

### Current entrypoints:
```
» ingest-parse-train --help

 Usage: ingest-parse-train [OPTIONS] COMMAND [ARGS]...

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                                                                          │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                                   │
│ --help                        Show this message and exit.                                                                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ download        Download raw Data Decide Results from HF to Local                                                                                                │
│ parse           Parse already downloaded Data Decide Results                                                                                                     │
│ upload          Upload parsed Data Decide Results from local to HF                                                                                               │
│ full-pipeline   Download, parse, parse and upload Data Decide results                                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
```
» ingest-parse-scaling --help

 Usage: ingest-parse-scaling [OPTIONS] COMMAND [ARGS]...

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                                                                          │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                                   │
│ --help                        Show this message and exit.                                                                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ download                                                                                                                                                         │
│ parse                                                                                                                                                            │
│ upload                                                                                                                                                           │
│ full-pipeline                                                                                                                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```



