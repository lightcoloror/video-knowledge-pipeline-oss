from __future__ import annotations

import hashlib
import json
from pathlib import Path

from video_knowledge_pipeline.smart_summary_chapters import (
    _chapter_citation_digest,
    _chapter_evidence_trace,
)
from video_knowledge_pipeline.smart_summary_input_pack import (
    build_smart_summary_input_pack,
)
from video_knowledge_pipeline.content_scope import CONTENT_SCOPE_SCHEMA


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def test_summary_and_citations_keep_semantic_correction_artifact_identity(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    transcript = bundle / "normalized-transcript.json"
    _write_json(
        transcript,
        {
            "segments": [
                {
                    "segment_id": "cue-0001",
                    "start": 0,
                    "end": 10,
                    "text": "中意人寿提供保障。",
                }
            ]
        },
    )
    _write_json(
        bundle / "manifest.json",
        {
            "title": "Synthetic",
            "normalized_transcript_json": "normalized-transcript.json",
        },
    )
    _write_json(
        bundle / "timeline.json",
        [
            {
                "index": 1,
                "segment_id": "timeline-0001",
                "start": 0,
                "end": 10,
                "transcript": "中意人寿提供保障。",
            }
        ],
    )
    _write_json(
        bundle / "transcript-semantic-correction-pack.json",
        {
            "status": "ready_for_review",
            "candidates": [
                {
                    "candidate_id": "semcorr-0001",
                    "segment_index": 0,
                    "correction_type": "organization",
                    "evidence_ids": ["ocr-1", "asr-2"],
                }
            ],
        },
    )
    _write_json(
        bundle / "transcript-semantic-correction-validation.json",
        {
            "status": "validated",
            "accepted_decisions": [
                {
                    "candidate_id": "semcorr-0001",
                    "segment_index": 0,
                    "action": "replace",
                    "correction_type": "organization",
                    "evidence_ids": ["ocr-1", "asr-2"],
                    "human_confirmed": True,
                }
            ],
        },
    )
    _write_json(
        bundle / "transcript-semantic-correction-closure.json",
        {"status": "completed", "pack_sha256": "a" * 64},
    )

    pack = build_smart_summary_input_pack(bundle, write=False)
    segment = pack["transcript_segments"][0]
    provenance = segment["semantic_correction_provenance"]
    expected_transcript_sha = hashlib.sha256(transcript.read_bytes()).hexdigest()

    assert provenance["candidate_ids"] == ["semcorr-0001"]
    assert provenance["correction_types"] == ["organization"]
    assert provenance["evidence_ids"] == ["ocr-1", "asr-2"]
    assert provenance["human_confirmed"] is True
    assert provenance["artifact_hashes"]["pack"]["sha256"]
    assert pack["evidence_trace"]["transcript_sha256"] == expected_transcript_sha
    assert pack["evidence_trace"]["semantic_correction"]["status"] == "closed"

    trace = _chapter_evidence_trace(
        pack["evidence_trace"],
        pack["transcript_segments"],
        0,
        10,
    )
    citation = _chapter_citation_digest(pack["transcript_segments"], trace)[0]

    assert trace["transcript_sha256"] == expected_transcript_sha
    assert trace["summary"]["semantically_corrected_segments"] == 1
    assert citation["source_sha256"] == expected_transcript_sha
    assert citation["segment_id"] == "cue-0001"
    assert citation["semantic_correction"]["candidate_ids"] == ["semcorr-0001"]


def test_semantic_provenance_uses_zero_based_source_identity_after_scope_filter(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    transcript = bundle / "normalized-transcript.json"
    _write_json(
        transcript,
        {
            "segments": [
                {"segment_id": "warmup", "start": 0, "end": 5, "text": "暖场歌词。"},
                {"segment_id": "body", "start": 5, "end": 10, "text": "中意人寿提供保障。"},
            ]
        },
    )
    transcript_sha = hashlib.sha256(transcript.read_bytes()).hexdigest()
    _write_json(
        bundle / "manifest.json",
        {"title": "Synthetic", "normalized_transcript_json": "normalized-transcript.json"},
    )
    _write_json(
        bundle / "timeline.json",
        [
            {"index": 1, "start": 0, "end": 5, "transcript": "暖场歌词。"},
            {"index": 2, "start": 5, "end": 10, "transcript": "中意人寿提供保障。"},
        ],
    )
    _write_json(
        bundle / "content-scope.json",
        {
            "schema": CONTENT_SCOPE_SCHEMA,
            "status": "confirmed",
            "transcript_source": "normalized-transcript.json",
            "transcript_source_sha256": transcript_sha,
            "excluded_intervals": [
                {
                    "start": 0,
                    "end": 5,
                    "reason": "warmup_music",
                    "category": "warmup_music",
                    "disposition": "excluded_confirmed",
                    "decision": "operator_confirmed",
                    "exclude_from": ["smart_summary"],
                    "retain_in": ["full_transcript", "intermediate_evidence"],
                }
            ],
        },
    )
    _write_json(
        bundle / "transcript-semantic-correction-pack.json",
        {
            "candidates": [
                {"candidate_id": "warmup-fix", "segment_index": 0},
                {"candidate_id": "body-fix", "segment_index": 1},
            ]
        },
    )
    _write_json(
        bundle / "transcript-semantic-correction-validation.json",
        {
            "accepted_decisions": [
                {"candidate_id": "warmup-fix", "segment_index": 0, "action": "replace"},
                {"candidate_id": "body-fix", "segment_index": 1, "action": "replace"},
            ]
        },
    )
    _write_json(
        bundle / "transcript-semantic-correction-closure.json",
        {"status": "completed"},
    )

    pack = build_smart_summary_input_pack(bundle, write=False)

    assert len(pack["transcript_segments"]) == 1
    segment = pack["transcript_segments"][0]
    assert segment["segment_id"] == "body"
    assert segment["source_segment_indexes"] == [1]
    assert segment["semantic_correction_provenance"]["candidate_ids"] == ["body-fix"]
    assert pack["evidence_trace"]["semantic_correction"]["accepted_decision_count"] == 2
    assert pack["evidence_trace"]["semantic_correction"]["applied_segment_count"] == 1
