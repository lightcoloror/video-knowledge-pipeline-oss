from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.lecture_package import (
    _align_item_to_transcript,
    _review_transcript_segments,
)


def test_review_accepts_reliable_short_canonical_segment() -> None:
    """A cue with reliable temporal coverage may supply canonical review text.

    Intent: keep precise review playback. Decision: require the existing overlap
    hard gate before text matching. Reason: short visual windows must not inherit
    nearby long-form speech. Evidence: the established review-session regression
    treats long-cue attachment as unsafe. Effective scope: review projection only.
    """

    aligned = _align_item_to_transcript(
        {"start": 2.0, "end": 4.0, "transcript": "canonical 正文"},
        [{"start": 1.5, "end": 4.5, "text": "canonical 正文"}],
    )

    assert aligned is not None
    assert aligned["review_start"] == 1.5
    assert aligned["review_start_source"] == "asr_segment_start"


def test_review_rejects_matching_long_cue_without_word_timestamps() -> None:
    """Text equality alone cannot locate speech inside one long canonical cue."""

    aligned = _align_item_to_transcript(
        {"start": 2.0, "end": 4.0, "transcript": "canonical 正文"},
        [{"start": 0.0, "end": 30.0, "text": "这是一段较长的 canonical 正文"}],
    )

    assert aligned is None

def test_review_projects_only_words_inside_short_visual_window() -> None:
    """Timed words may safely localize a long cue without changing the cue."""

    aligned = _align_item_to_transcript(
        {"start": 2.0, "end": 4.0, "transcript": "第二部分"},
        [
            {
                "id": "segment-0001",
                "start": 0.0,
                "end": 10.0,
                "text": "第一部分第二部分第三部分",
                "words": [
                    {"word": "第一部分", "start": 0.0, "end": 1.0},
                    {"word": "第二部分", "start": 2.2, "end": 3.2},
                    {"word": "第三部分", "start": 8.0, "end": 9.0},
                ],
            }
        ],
    )

    assert aligned is not None
    assert aligned["review_start"] == 2.2
    assert aligned["review_start_source"] == "asr_word_timestamp_start"
    assert aligned["review_transcript_excerpt"] == "第二部分"
    assert aligned["review_transcript_source_segment_ids"] == ["segment-0001"]


def test_review_rejects_partial_word_timestamps_that_do_not_cover_canonical_text() -> None:
    """Partial provider timing must not masquerade as complete cue alignment."""

    aligned = _align_item_to_transcript(
        {"start": 2.0, "end": 4.0, "transcript": "第二部分"},
        [
            {
                "id": "segment-0001",
                "start": 0.0,
                "end": 10.0,
                "text": "第一部分第二部分第三部分",
                "metadata": {
                    "words": [
                        {"word": "第二部分", "start": 2.2, "end": 3.2},
                    ]
                },
            }
        ],
    )

    assert aligned is None


def test_review_loads_validated_qwen_alignment_sidecar(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    canonical = bundle / "normalized-transcript.json"
    canonical.write_text(
        json.dumps({"segments": [{"id": "segment-1", "start": 0, "end": 10, "text": "第一部分第二部分第三部分"}]}),
        encoding="utf-8",
    )
    sidecar = bundle / "qwen3-forced-aligner-output.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.qwen3_forced_aligner_output.v1",
                "status": "completed",
                "transcript_path": str(canonical),
                "timestamps_monotonic": True,
                "segments": [
                    {
                        "index": 1,
                        "start": 0,
                        "end": 10,
                        "text": "第一部分第二部分第三部分",
                        "source_cue_indexes": [0],
                        "words": [
                            {"text": "第一部分", "start": 0, "end": 1},
                            {"text": "第二部分", "start": 2.2, "end": 3.2},
                            {"text": "第三部分", "start": 8, "end": 9},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    package = {
        "review_artifacts": {
            "bundle_dir": str(bundle),
            "normalized_transcript_json": canonical.name,
            "qwen3_forced_alignment_json": sidecar.name,
        }
    }

    segments = _review_transcript_segments(package)
    aligned = _align_item_to_transcript(
        {"start": 2, "end": 4, "transcript": "第二部分"}, segments
    )

    assert aligned is not None
    assert aligned["review_start"] == 2.2
    assert aligned["review_start_source"] == "asr_word_timestamp_start"
    assert aligned["review_transcript_source_segment_ids"] == ["cue-index:0"]


def test_review_rejects_qwen_sidecar_when_text_identity_changed(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    canonical = bundle / "normalized-transcript.json"
    canonical.write_text(
        json.dumps({"segments": [{"start": 0, "end": 10, "text": "canonical 正文"}]}),
        encoding="utf-8",
    )
    sidecar = bundle / "qwen3-forced-aligner-output.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.qwen3_forced_aligner_output.v1",
                "status": "completed",
                "transcript_path": str(canonical),
                "timestamps_monotonic": True,
                "segments": [{"start": 0, "end": 10, "text": "changed", "words": [{"text": "changed", "start": 2, "end": 3}]}],
            }
        ),
        encoding="utf-8",
    )
    package = {
        "review_artifacts": {
            "bundle_dir": str(bundle),
            "normalized_transcript_json": canonical.name,
            "qwen3_forced_alignment_json": sidecar.name,
        }
    }

    segments = _review_transcript_segments(package)

    assert segments[0]["text"] == "canonical 正文"
    assert "words" not in segments[0]
