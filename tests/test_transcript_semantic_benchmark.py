from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.transcript_semantic_benchmark import (
    evaluate_transcript_semantic_gold,
)


FIXTURE = Path(__file__).parent / "fixtures" / "transcript-semantic-gold-v1.json"


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def test_deidentified_gold_metrics_cover_recall_safety_absorption_and_scope(
    tmp_path: Path,
) -> None:
    pack = _write(
        tmp_path / "pack.json",
        {
            "candidates": [
                {
                    "candidate_id": "c1",
                    "segment_index": 1,
                    "original_text": "中医人寿",
                    "correction_type": "organization",
                },
                {
                    "candidate_id": "c2",
                    "segment_index": 2,
                    "original_text": "可以赔付",
                    "correction_type": "negation",
                },
            ]
        },
    )
    validation = _write(
        tmp_path / "validation.json",
        {
            "accepted_decisions": [
                {
                    "candidate_id": "c1",
                    "correction_type": "organization",
                    "corrected_text": "中意人寿",
                    "human_confirmed": False,
                },
                {
                    "candidate_id": "c2",
                    "correction_type": "negation",
                    "corrected_text": "不可以赔付",
                    "human_confirmed": True,
                },
            ]
        },
    )

    result = evaluate_transcript_semantic_gold(
        FIXTURE,
        pack,
        validation,
        full_body_text="中意人寿说明：不可以赔付。",
        smart_summary_text="中意人寿说明：不可以赔付。",
    )

    assert result["ok"] is True
    assert result["metrics"] == {
        "candidate_recall": 1.0,
        "candidate_precision": 1.0,
        "auto_apply_accuracy": 1.0,
        "dangerous_misedit_rate": 0.0,
        "full_body_absorption_rate": 1.0,
        "summary_no_stale_original_rate": 1.0,
        "excluded_content_candidate_budget_consumed": 0,
        "excluded_content_leak_count": 0,
    }
    assert result["boundary"]["deidentified_fixture_only"] is True


def test_gold_benchmark_fails_dangerous_misedit_and_excluded_content_leak(
    tmp_path: Path,
) -> None:
    pack = _write(
        tmp_path / "pack.json",
        {
            "candidates": [
                {
                    "candidate_id": "c1",
                    "segment_index": 2,
                    "original_text": "可以赔付",
                    "correction_type": "negation",
                },
                {
                    "candidate_id": "c2",
                    "segment_index": 3,
                    "original_text": "欢迎大家点赞关注",
                    "correction_type": "content_scope",
                },
            ]
        },
    )
    validation = _write(
        tmp_path / "validation.json",
        {
            "accepted_decisions": [
                {
                    "candidate_id": "c1",
                    "correction_type": "negation",
                    "corrected_text": "一定可以赔付",
                    "human_confirmed": False,
                }
            ]
        },
    )

    result = evaluate_transcript_semantic_gold(
        FIXTURE,
        pack,
        validation,
        full_body_text="欢迎大家点赞关注。",
        smart_summary_text="欢迎大家点赞关注。",
    )

    assert result["ok"] is False
    assert result["metrics"]["dangerous_misedit_rate"] == 1.0
    assert result["metrics"]["excluded_content_candidate_budget_consumed"] == 1
    assert result["metrics"]["excluded_content_leak_count"] == 1


def test_summary_may_omit_corrected_detail_but_must_not_repeat_stale_text(
    tmp_path: Path,
) -> None:
    pack = _write(
        tmp_path / "pack.json",
        {
            "candidates": [
                {
                    "candidate_id": "c1",
                    "segment_index": 1,
                    "original_text": "中医人寿",
                    "correction_type": "organization",
                }
            ]
        },
    )
    validation = _write(
        tmp_path / "validation.json",
        {
            "accepted_decisions": [
                {
                    "candidate_id": "c1",
                    "correction_type": "organization",
                    "corrected_text": "中意人寿",
                    "human_confirmed": True,
                }
            ]
        },
    )

    omitted = evaluate_transcript_semantic_gold(
        FIXTURE,
        pack,
        validation,
        full_body_text="中意人寿提供保障。",
        smart_summary_text="本节概述保险保障原则。",
    )
    stale = evaluate_transcript_semantic_gold(
        FIXTURE,
        pack,
        validation,
        full_body_text="中意人寿提供保障。",
        smart_summary_text="中医人寿提供保障。",
    )

    assert omitted["metrics"]["full_body_absorption_rate"] == 1.0
    assert omitted["metrics"]["summary_no_stale_original_rate"] == 1.0
    assert stale["metrics"]["summary_no_stale_original_rate"] == 0.0
    assert stale["ok"] is False
