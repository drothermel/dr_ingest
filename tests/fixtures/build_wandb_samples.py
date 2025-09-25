from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterable

import pandas as pd

RUN_IDS = [
    "2025_08_21-08_24_43_test_finetune_DD-dolma1_7-4M_main_1Mtx1_--learning_rate=5e-07",
    "2025_08_30-16_54_48_test_finetune_DD-dolma1_7-4M_Ft_--learning_rate=5e-05",
    "250901-155734_test_finetune_DD-dclm-baseline-150M_Ft_learning_rate=5e-05",
    "250912-172812_match_150M_c4_finetune_100Mtx1_DD-d17-530M-6250-2",
    "2025_08_27-07_52_26_test_dpo_tune_cache_dd__dolma1_7-4M__main__100Mt_lr=5e-06_default_--learning_rate=2e-07",
    "2025_08_18-22_30_05_allenai--DataDecide-dolma1_7-4M_main_default",
]

# Parameters/datasets needed for the pretraining fixture
PRETRAIN_COMBOS = {
    ("4M", "Dolma1.7"),
    ("150M", "DCLM-Baseline"),
    ("150M", "C4"),
    ("530M", "Dolma1.7"),
}
PRETRAIN_ALLOWED_SEEDS = {0}
PRETRAIN_MAX_STEP = 10_000


def _repo_paths() -> dict[str, Path]:
    fixtures_dir = Path(__file__).resolve().parent
    repo_root = fixtures_dir.parent.parent
    repos_root = repo_root.parent
    datadec_root = repos_root / "datadec"

    paths = {
        "fixtures_root": fixtures_dir,
        "data_dir": fixtures_dir / "data",
        "raw_runs": datadec_root / "notebooks" / "wandb_runs.jsonl",
        "raw_history": datadec_root / "notebooks" / "wandb_history.parquet",
        "pretrain_full": datadec_root / "data" / "datadecide" / "full_eval.parquet",
    }

    for key, path in paths.items():
        if key.endswith("root") or key == "data_dir":
            continue
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")

    paths["data_dir"].mkdir(parents=True, exist_ok=True)
    return paths


def _normalize_serializable(record: dict) -> dict:
    record = record.copy()
    for key, value in list(record.items()):
        if isinstance(value, (dict, list)):
            record[key] = json.dumps(value)
        elif value is None:
            record[key] = None
        elif isinstance(value, (int, float, str, bool)):
            # already serialisable
            continue
        else:
            record[key] = str(value)
    return record


def _collect_runs(raw_path: Path, run_ids: Iterable[str]) -> pd.DataFrame:
    selected_records: list[dict] = []
    with raw_path.open() as fh:
        for line in fh:
            record = json.loads(line)
            if record["run_id"] in run_ids:
                selected_records.append(_normalize_serializable(record))
    if not selected_records:
        raise ValueError("No matching runs found for requested fixture IDs")
    df = pd.DataFrame(selected_records)
    return df


def _collect_history(raw_path: Path, run_ids: set[str]) -> pd.DataFrame:
    history_df = pd.read_parquet(raw_path)
    if "json" not in history_df.columns:
        raise ValueError("Unexpected history parquet schema; expected 'json' column")

    rows: list[dict] = []
    for payload in history_df["json"]:
        if payload is None:
            continue
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            if entry.get("run_id") not in run_ids:
                continue
            metrics = entry.get("metrics") or {}
            rows.append(
                {
                    "run_id": entry.get("run_id"),
                    "step": entry.get("step"),
                    "timestamp": entry.get("timestamp"),
                    "runtime": entry.get("runtime"),
                    "total_tokens": metrics.get("total_tokens"),
                    "train_loss": metrics.get("train_loss"),
                    "learning_rate": metrics.get("learning_rate"),
                }
            )

    if not rows:
        raise ValueError("No matching history rows found for requested run IDs")

    history = pd.DataFrame(rows)
    return history.sort_values(["run_id", "step"], ignore_index=True)


def _collect_pretrain(full_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(full_path)
    combo_mask = df.apply(
        lambda row: (row["params"], row["data"]) in PRETRAIN_COMBOS, axis=1
    )
    seed_mask = df["seed"].isin(PRETRAIN_ALLOWED_SEEDS)
    step_mask = df["step"].fillna(0) <= PRETRAIN_MAX_STEP
    mask = combo_mask & seed_mask & step_mask
    filtered = df.loc[mask].copy()
    if filtered.empty:
        raise ValueError("Pretraining filter produced an empty fixture subset")
    return filtered


def main() -> None:
    paths = _repo_paths()
    run_ids = set(RUN_IDS)

    runs_df = _collect_runs(paths["raw_runs"], run_ids)
    runs_out = paths["data_dir"] / "wandb_runs_sample.parquet"
    runs_df.to_parquet(runs_out, index=False)

    history_df = _collect_history(paths["raw_history"], run_ids)
    history_out = paths["data_dir"] / "wandb_history_sample.parquet"
    history_df.to_parquet(history_out, index=False)

    pretrain_df = _collect_pretrain(paths["pretrain_full"])
    pretrain_out = paths["data_dir"] / "pretrain_metrics_sample.parquet"
    pretrain_df.to_parquet(pretrain_out, index=False)

    manifest = {
        "runs": {
            "path": runs_out.name,
            "rows": len(runs_df),
            "columns": list(runs_df.columns),
        },
        "history": {
            "path": history_out.name,
            "rows": len(history_df),
            "columns": list(history_df.columns),
        },
        "pretrain": {
            "path": pretrain_out.name,
            "rows": len(pretrain_df),
            "columns": list(pretrain_df.columns[:15]),
        },
    }

    manifest_path = paths["data_dir"] / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("Fixture extraction complete")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
