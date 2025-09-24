from __future__ import annotations

from dr_ingest.qa.transform import (
    build_file_metadata,
    extract_question_payloads,
    model_output_keys,
    preview_agg_metrics,
)


SAMPLE_RECORDS = [
    {
        "doc_id": 1,
        "task_hash": "task123",
        "model_hash": "modelA",
        "metrics": {"accuracy": 0.8, "loss": 0.3},
        "answer_outputs": [
            {
                "is_greedy": True,
                "logits_per_token": 1.2,
                "extra": "ignored",
            }
        ],
    },
    {
        "doc_id": 2,
        "task_hash": "task123",
        "model_hash": "modelA",
        "metrics": {"accuracy": 0.7},
        "answer_outputs": [
            {
                "is_greedy": False,
                "logits_per_token": 0.9,
            }
        ],
    },
]


def test_build_file_metadata() -> None:
    result = build_file_metadata(
        SAMPLE_RECORDS,
        data="c4",
        params="4M",
        seed=2,
        task="qa_task",
        step=10,
    )
    assert len(result) == 1
    row = result[0]
    assert row["task_hash"] == "task123"
    assert row["data"] == "c4"
    assert row["step"] == 10


def test_extract_question_payloads() -> None:
    payloads = extract_question_payloads(SAMPLE_RECORDS)
    assert len(payloads) == 2
    assert "metrics" not in payloads[0]
    assert "model_output" in payloads[0]


def test_preview_agg_metrics() -> None:
    metrics = preview_agg_metrics(SAMPLE_RECORDS)
    assert metrics[0]["doc_id"] == 1
    assert metrics[0]["loss"] == 0.3


def test_preview_agg_metrics_handles_missing() -> None:
    records = [
        {"doc_id": 3, "metrics": None},
        {"doc_id": 4},
    ]
    metrics = preview_agg_metrics(records)
    assert metrics[0] == {"doc_id": 3}
    assert metrics[1] == {"doc_id": 4}


def test_model_output_keys() -> None:
    keys = model_output_keys(extract_question_payloads(SAMPLE_RECORDS))
    assert keys == ["extra", "is_greedy", "logits_per_token"]
