from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.retrieval_goldset import evaluate_retrieval_goldset


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _report(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema": "vkp.sqlite_vec_benchmark.v1",
            "queries": [
                {
                    "query_id": "q1",
                    "bundle_id": "bundle",
                    "top_10": [
                        {"stable_id": f"chunk-{index}"}
                        for index in range(1, 11)
                    ],
                },
                {
                    "query_id": "q2",
                    "bundle_id": "bundle",
                    "top_10": [{"stable_id": "other"}],
                },
            ],
        },
    )


def test_goldset_evaluation_distinguishes_human_labels_from_pending_rows(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path / "report.json")
    review = _write_json(
        tmp_path / "review.json",
        {
            "schema": "vkp.sqlite_vec_human_gold_review.v1",
            "queries": [
                {
                    "query_id": "q1",
                    "bundle_id": "bundle",
                    "review_status": "human_confirmed",
                    "relevant_stable_ids": ["chunk-2", "chunk-8"],
                    "review_notes": "two relevant passages",
                },
                {
                    "query_id": "q2",
                    "bundle_id": "bundle",
                    "review_status": "pending_human_review",
                    "relevant_stable_ids": ["other"],
                },
            ],
        },
    )

    result = evaluate_retrieval_goldset(review, report)

    assert result["status"] == "partial"
    assert result["evidence_status"] == "partial_human_review"
    assert result["aggregate"]["reviewed_query_count"] == 1
    assert result["aggregate"]["pending_query_count"] == 1
    assert result["aggregate"]["recall_at_5"] == 0.5
    assert result["aggregate"]["recall_at_10"] == 1.0
    assert result["pending_query_ids"] == ["q2"]


def test_goldset_requires_complete_human_review_when_requested(tmp_path: Path) -> None:
    report = _report(tmp_path / "report.json")
    review = _write_json(
        tmp_path / "review.json",
        {
            "schema": "vkp.sqlite_vec_human_gold_review.v1",
            "queries": [
                {
                    "query_id": "q1",
                    "review_status": "pending_human_review",
                    "relevant_stable_ids": ["chunk-1"],
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="human review is incomplete"):
        evaluate_retrieval_goldset(review, report, require_complete=True)


def test_goldset_rejects_unbound_query(tmp_path: Path) -> None:
    report = _report(tmp_path / "report.json")
    review = _write_json(
        tmp_path / "review.json",
        {
            "schema": "vkp.sqlite_vec_human_gold_review.v1",
            "queries": [
                {
                    "query_id": "missing",
                    "review_status": "human_confirmed",
                    "relevant_stable_ids": ["chunk-1"],
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="absent from source report"):
        evaluate_retrieval_goldset(review, report)
