from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.retrieval_goldset import evaluate_retrieval_goldset


def test_zero_human_labels_are_explicitly_blocked_not_partial(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "vkp.sqlite_vec_benchmark.v1",
                "queries": [
                    {
                        "query_id": "q1",
                        "bundle_id": "bundle",
                        "top_10": [{"stable_id": "chunk-1"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "schema": "vkp.sqlite_vec_human_gold_review.v1",
                "queries": [
                    {
                        "query_id": "q1",
                        "review_status": "pending_human_review",
                        "relevant_stable_ids": ["chunk-1"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_retrieval_goldset(review, report)

    assert result["status"] == "blocked_human_review"
    assert result["evidence_status"] == "blocked_human_review"
