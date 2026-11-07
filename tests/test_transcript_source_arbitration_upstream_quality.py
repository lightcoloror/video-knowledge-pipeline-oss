from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.transcript_source_arbitration import (
    arbitrate_transcript_sources,
)
from video_knowledge_pipeline.transcript import parse_transcript


def test_arbitration_preserves_upstream_asr_review_chunk(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    write_json(root / "manifest.json", {"title": "fixture"})
    transcript = root / "normalized-transcript.json"
    write_json(
        transcript,
        {
            "schema": "video_knowledge_pipeline.normalized_transcript.v1",
            "source_execution": {
                "status": "review_required",
                "asr_quality": {
                    "status": "review_required",
                    "segment_count": 3,
                    "passed_segment_count": 2,
                    "review_segment_count": 1,
                    "failed_segment_count": 0,
                    "review_chunks": [
                        {
                            "segment_id": "2",
                            "position": 99,
                            "start": 2.0,
                            "end": 4.0,
                            "reasons": ["low_text_density"],
                            "original_text": "第二段",
                            "preserve_original_text": True,
                        }
                    ],
                    "failed_chunks": [],
                },
            },
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "第一段"},
                {
                    "segment_id": "2",
                    "start": 2.0,
                    "end": 4.0,
                    "text": "",
                    "empty_text_preserved": True,
                },
                {"segment_id": "3", "start": 4.0, "end": 6.0, "text": "第三段"},
            ],
        },
    )

    result = arbitrate_transcript_sources(root, asr_json=transcript)

    assert result["status"] == "completed_with_review_items"
    assert result["quality_summary"]["status"] == "needs_review"
    assert result["quality_summary"]["review_segment_indexes"] == [2]
    assert result["quality_summary"]["can_use_as_summary_input"] is False
    assert result["upstream_asr_quality"]["review_segment_count"] == 1
    corrected = read_json(root / "source-arbitrated-transcript.json")
    assert corrected["segments"][0]["needs_human_review"] is False
    assert corrected["segments"][1]["needs_human_review"] is True
    assert corrected["segments"][1]["segment_id"] == "2"
    assert corrected["segments"][1]["empty_text_preserved"] is True
    assert corrected["segments"][1]["start"] == 2.0
    assert corrected["segments"][1]["end"] == 4.0
    assert corrected["segments"][2]["needs_human_review"] is False
    reparsed = parse_transcript(root / "source-arbitrated-transcript.json")
    assert len(reparsed) == 3
    assert reparsed[1].segment_id == "2"
    assert reparsed[1].text == ""
    assert (
        corrected["segments"][1]["review_reason"]
        == "upstream_asr_review:low_text_density"
    )
