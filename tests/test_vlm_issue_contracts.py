from __future__ import annotations

from video_knowledge_pipeline.acceptance_check import _review_queue_summary
from video_knowledge_pipeline.knowledge_note_export import _export_status
from video_knowledge_pipeline.smart_summary_codex import (
    _quality_number_evidence,
    _structural_heading_numbers,
)
from video_knowledge_pipeline.visual_evidence import evidence_index_sets


def test_visual_evidence_counts_share_strict_model_and_consumable_sets() -> None:
    timeline = [
        {
            "index": 1,
            "assets": [{"path": "frame.jpg"}],
            "visual_understanding": {
                "objects": ["window"],
                "actions": [],
            },
        },
        {
            "index": 2,
            "assets": [{"path": "frame.jpg"}],
            "visual_understanding": {
                "objects": ["dialog"],
                "validation_status": "incomplete",
            },
        },
        {
            "index": 3,
            "review_status": "accepted",
            "human_review": {"status": "accepted"},
        },
    ]

    result = evidence_index_sets(timeline)

    assert result["visual_model_complete"] == [1]
    assert result["visual_export_consumable"] == [1, 3]


def test_export_status_separates_file_write_from_quality_gate() -> None:
    assert _export_status(write_status="exported", quality_ready=True) == "exported_quality_ready"
    assert _export_status(write_status="exported", quality_ready=False) == "exported_quality_blocked"
    assert _export_status(write_status="blocked_canonical_export_mismatch", quality_ready=False) == "blocked_canonical_export_mismatch"


def test_heading_ordinals_are_structural_not_factual_numbers() -> None:
    markdown = "## 8、 高频话术\n正文包含保费 3000 万。\n### 8.1.2 示例"

    evidence = _quality_number_evidence(markdown)

    assert _structural_heading_numbers(markdown) == ["8"]
    assert not any("8" in key for key in evidence)
    assert any("30000000" in key for key in evidence)


def test_acceptance_review_queue_keeps_channels_separate_and_deduplicated() -> None:
    result = _review_queue_summary(
        {
            "semantic_frame_without_analysis": 613,
            "temporal_sequence_without_analysis": 7,
            "samples": {
                "missing_transcript": [1, 2],
                "ocr_gap": [2, 3],
                "missing_visual_understanding": [3, 4],
                "temporal_sequence_without_analysis": [4, 5],
            },
        },
        {
            "speech": {"expected_count": 674, "covered_count": 670},
            "screen_text": {"expected_count": 100, "covered_count": 80, "blocker_count": 25},
        },
    )

    assert result["transcript_review_targets"] == 4
    assert result["ocr_review_targets"] == 25
    assert result["semantic_visual_missing"] == 613
    assert result["temporal_visual_missing"] == 7
    assert result["deduplicated_sample_indexes"] == [1, 2, 3, 4, 5]
