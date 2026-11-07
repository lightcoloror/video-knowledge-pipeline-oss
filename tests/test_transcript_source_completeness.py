from __future__ import annotations

import json
from pathlib import Path
import pytest

from video_knowledge_pipeline.funasr_chunked_runner import _records_have_content
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.transcript_quality_gate import (
    run_transcript_quality_gate,
)
from video_knowledge_pipeline.transcript_source_completeness import (
    assess_transcript_source_completeness,
)


@pytest.mark.parametrize('status,ok,segments', [
    ('runtime_unavailable', False, []), ('failed', False, []),
    ('completed', False, []), ('completed', True, [{'start': 'bad', 'end': 2}]),
    ('completed', True, 3), (['completed'], True, []),
])
def test_failed_or_malformed_vad_never_establishes_silence(tmp_path, status, ok, segments):
    from video_knowledge_pipeline.transcript_source_completeness import _load_bound_independent_vad, SILERO_CANDIDATE_SCHEMA
    media = tmp_path / 'audio.wav'
    media.write_bytes(b'synthetic media')
    _write_json(tmp_path / 'silero-vad-candidate.json', {
        'schema': SILERO_CANDIDATE_SCHEMA, 'status': status, 'ok': ok,
        'candidate_only': True, 'segments': segments,
        'source_media': {'path': str(media), 'sha256': sha256_file(media)},
    })
    result = _load_bound_independent_vad(tmp_path, tmp_path / 'raw.json', {'input': str(media)}, {})
    assert result['verified'] is False
    assert result['segments'] == []
    assert result['summary']['status'] != 'completed'
    if not isinstance(segments, list):
        assert result['summary']['segment_count'] is None


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _bundle(tmp_path: Path, raw: dict[str, object]) -> tuple[Path, Path]:
    root = tmp_path / "bundle"
    raw_path = root / "transcripts" / "raw-asr-output.json"
    normalized = root / "normalized-transcript.json"
    _write_json(raw_path, raw)
    _write_json(
        normalized,
        {
            "source_path": str(raw_path),
            "segments": [
                {"start": 0, "end": 300, "text": "这是一段有内容的转写。"},
                {"start": 300, "end": 600, "text": "第二段也保留时间范围。"},
            ],
        },
    )
    _write_json(
        root / "manifest.json",
        {
            "duration_seconds": 600,
            "normalized_transcript_json": "normalized-transcript.json",
        },
    )
    return root, normalized


def test_empty_text_record_is_not_completed_chunk_evidence() -> None:
    assert _records_have_content([{"text": "", "timestamp": []}]) is False
    assert _records_have_content([{"text": "真实语音"}]) is True


def test_chunked_source_rejects_nominally_successful_empty_chunk(
    tmp_path: Path,
) -> None:
    root, normalized = _bundle(
        tmp_path,
        {
            "schema": "video_knowledge_funasr_chunked_raw_output.v1",
            "status": "completed",
            "ok": True,
            "duration_seconds": 600,
            "chunk_seconds": 300,
            "chunk_count": 2,
            "successful_chunk_count": 2,
            "successful_chunk_indexes": [0, 1],
            "failed_chunk_count": 0,
            "failed_chunks": [],
            "gaps": [],
            "chunk_results": [
                {"chunk_index": 0, "text": "有内容", "chunk_offset_seconds": 0},
                {
                    "chunk_index": 1,
                    "text": "",
                    "timestamp": [],
                    "chunk_offset_seconds": 300,
                },
            ],
        },
    )

    result = assess_transcript_source_completeness(root, normalized)

    assert result["status"] == "failed"
    assert result["chunk_integrity"]["unverified_empty_chunk_indexes"] == [1]
    assert "unverified_empty_asr_chunks" in {
        row["kind"] for row in result["issues"]
    }


def test_complete_chunks_do_not_claim_speech_completeness_without_vad(
    tmp_path: Path,
) -> None:
    root, normalized = _bundle(
        tmp_path,
        {
            "schema": "video_knowledge_funasr_chunked_raw_output.v1",
            "status": "completed",
            "ok": True,
            "duration_seconds": 600,
            "chunk_seconds": 300,
            "chunk_count": 2,
            "successful_chunk_count": 2,
            "successful_chunk_indexes": [0, 1],
            "failed_chunk_count": 0,
            "failed_chunks": [],
            "gaps": [],
            "chunk_results": [
                {"chunk_index": 0, "text": "第一块", "chunk_offset_seconds": 0},
                {"chunk_index": 1, "text": "第二块", "chunk_offset_seconds": 300},
            ],
        },
    )

    result = assess_transcript_source_completeness(root, normalized)

    assert result["execution_integrity"] == "passed"
    assert result["speech_completeness_verified"] is False
    assert result["speech_coverage"] == "unverified"
    assert result["status"] == "warning"


def test_readable_projection_follows_normalized_lineage_to_raw_source(
    tmp_path: Path,
) -> None:
    root, normalized = _bundle(
        tmp_path,
        {
            "schema": "video_knowledge_funasr_chunked_raw_output.v1",
            "status": "completed",
            "quality_status": "degraded",
            "ok": True,
            "duration_seconds": 600,
            "chunk_seconds": 300,
            "chunk_count": 2,
            "successful_chunk_count": 2,
            "successful_chunk_indexes": [0, 1],
            "failed_chunk_count": 0,
            "failed_chunks": [],
            "gaps": [],
            "overlap_merge": {
                "status": "review_required",
                "boundary_review_required_count": 1,
            },
            "chunk_results": [
                {"chunk_index": 0, "text": "第一块", "chunk_offset_seconds": 0},
                {"chunk_index": 1, "text": "第二块", "chunk_offset_seconds": 300},
            ],
        },
    )
    readable = root / "readable-transcript.json"
    _write_json(
        readable,
        {
            "source_path": str(normalized),
            "segments": json.loads(normalized.read_text(encoding="utf-8"))["segments"],
        },
    )

    result = assess_transcript_source_completeness(root, readable)

    assert result["source_schema"] == "video_knowledge_funasr_chunked_raw_output.v1"
    assert result["status"] == "failed"
    assert result["chunk_integrity"]["quality_status"] == "degraded"
    assert result["chunk_integrity"]["boundary_review_required_count"] == 1
    assert "asr_chunk_boundary_review_required" in {
        row["kind"] for row in result["issues"]
    }


def test_source_bound_complete_boundary_review_resolves_raw_review_marker(
    tmp_path: Path,
) -> None:
    root, normalized = _bundle(
        tmp_path,
        {
            "schema": "video_knowledge_funasr_chunked_raw_output.v1",
            "status": "completed",
            "quality_status": "degraded",
            "ok": True,
            "duration_seconds": 600,
            "chunk_seconds": 300,
            "chunk_count": 2,
            "successful_chunk_count": 2,
            "successful_chunk_indexes": [0, 1],
            "failed_chunk_count": 0,
            "failed_chunks": [],
            "gaps": [],
            "overlap_merge": {
                "status": "review_required",
                "boundary_review_required_count": 1,
                "boundaries": [
                    {
                        "left_chunk_index": 0,
                        "right_chunk_index": 1,
                        "requires_human_review": True,
                    }
                ],
            },
            "chunk_results": [
                {"chunk_index": 0, "text": "第一块", "chunk_offset_seconds": 0},
                {"chunk_index": 1, "text": "第二块", "chunk_offset_seconds": 300},
            ],
        },
    )
    raw_path = root / "transcripts" / "raw-asr-output.json"
    _write_json(
        root / "asr-boundary-review.json",
        {
            "schema": "video_knowledge_pipeline.asr_boundary_review.v1",
            "status": "completed",
            "source": {
                "path": str(raw_path),
                "sha256": sha256_file(raw_path),
            },
            "decisions": [
                {
                    "left_chunk_index": 0,
                    "right_chunk_index": 1,
                    "status": "no_omission_detected",
                    "action": "retain_merged_output",
                    "missing_content_detected": False,
                }
            ],
        },
    )

    result = assess_transcript_source_completeness(root, normalized)

    assert result["boundary_review"]["status"] == "accepted"
    assert result["chunk_integrity"]["boundary_review_original_count"] == 1
    assert result["chunk_integrity"]["boundary_review_required_count"] == 0
    assert "asr_chunk_boundary_review_required" not in {
        row["kind"] for row in result["issues"]
    }


def test_stale_boundary_review_hash_cannot_resolve_raw_review_marker(
    tmp_path: Path,
) -> None:
    root, normalized = _bundle(
        tmp_path,
        {
            "schema": "video_knowledge_funasr_chunked_raw_output.v1",
            "status": "completed",
            "quality_status": "degraded",
            "ok": True,
            "duration_seconds": 600,
            "chunk_seconds": 300,
            "chunk_count": 2,
            "successful_chunk_count": 2,
            "successful_chunk_indexes": [0, 1],
            "failed_chunk_count": 0,
            "failed_chunks": [],
            "gaps": [],
            "overlap_merge": {
                "status": "review_required",
                "boundary_review_required_count": 1,
                "boundaries": [
                    {
                        "left_chunk_index": 0,
                        "right_chunk_index": 1,
                        "requires_human_review": True,
                    }
                ],
            },
            "chunk_results": [
                {"chunk_index": 0, "text": "第一块", "chunk_offset_seconds": 0},
                {"chunk_index": 1, "text": "第二块", "chunk_offset_seconds": 300},
            ],
        },
    )
    raw_path = root / "transcripts" / "raw-asr-output.json"
    _write_json(
        root / "asr-boundary-review.json",
        {
            "schema": "video_knowledge_pipeline.asr_boundary_review.v1",
            "status": "completed",
            "source": {"path": str(raw_path), "sha256": "0" * 64},
            "decisions": [
                {
                    "left_chunk_index": 0,
                    "right_chunk_index": 1,
                    "status": "no_omission_detected",
                    "action": "retain_merged_output",
                    "missing_content_detected": False,
                }
            ],
        },
    )

    result = assess_transcript_source_completeness(root, normalized)

    assert result["boundary_review"]["status"] == "invalid_or_stale"
    assert result["boundary_review"]["source_sha256_matches"] is False
    assert result["chunk_integrity"]["boundary_review_required_count"] == 1
    assert "asr_chunk_boundary_review_required" in {
        row["kind"] for row in result["issues"]
    }


def test_transcript_gate_labels_span_coverage_without_overclaiming(
    tmp_path: Path,
) -> None:
    root, _normalized = _bundle(
        tmp_path,
        {
            "schema": "video_knowledge_funasr_raw_output.v1",
            "duration_seconds": 1800,
            "result": [{"text": "整段单次结果"}],
        },
    )

    result = run_transcript_quality_gate(root, write=False)

    assert result["audio_coverage"] == result["timeline_span_coverage"]
    assert (
        result["audio_coverage_semantics"]
        == "timeline_span_only_not_speech_completeness"
    )
    assert result["source_completeness"]["execution_mode"] == "legacy_single_pass"
    assert result["source_completeness"]["speech_completeness_verified"] is False
    assert "legacy_single_pass_long_media" in {
        row["kind"] for row in result["issues"]
    }


def test_source_completeness_reports_coarse_estimated_timing_without_density_retry(
    tmp_path: Path,
) -> None:
    """Keep content completeness and timestamp precision as separate facts."""

    root, normalized = _bundle(
        tmp_path,
        {
            "schema": "video_knowledge_funasr_chunked_raw_output.v1",
            "status": "completed",
            "ok": True,
            "duration_seconds": 600,
            "chunk_seconds": 300,
            "chunk_count": 2,
            "successful_chunk_count": 2,
            "successful_chunk_indexes": [0, 1],
            "failed_chunk_count": 0,
            "failed_chunks": [],
            "gaps": [],
            "chunk_results": [
                {"chunk_index": 0, "text": "第一块", "chunk_offset_seconds": 0},
                {"chunk_index": 1, "text": "第二块", "chunk_offset_seconds": 300},
            ],
        },
    )
    payload = json.loads(normalized.read_text(encoding="utf-8"))
    for segment in payload["segments"]:
        segment["transformations"] = [
            {
                "type": "timing_estimation",
                "method": "character_proportional_within_source_window",
            }
        ]
    _write_json(normalized, payload)

    result = assess_transcript_source_completeness(root, normalized)

    assert result["timing_precision"] == "coarse_estimated"
    assert result["estimated_timing_segment_count"] == 2
    assert result["response_quality"]["review_segment_count"] == 0
    assert result["response_quality"]["retry_plan"]["windows"] == []
    assert "asr_timing_estimated" in {row["kind"] for row in result["issues"]}
