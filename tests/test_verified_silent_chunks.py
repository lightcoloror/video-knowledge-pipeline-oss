from __future__ import annotations

import hashlib
import json
from pathlib import Path

from video_knowledge_pipeline.transcript_source_completeness import (
    assess_transcript_source_completeness,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _fixture(
    tmp_path: Path,
    *,
    independent_speech: list[dict[str, float]],
) -> tuple[Path, Path]:
    root = tmp_path / "bundle"
    media = tmp_path / "lesson.wav"
    media.write_bytes(b"source-bound-audio")
    raw = root / "transcripts" / "raw-asr-output.json"
    normalized = root / "normalized-transcript.json"
    _write_json(
        raw,
        {
            "schema": "video_knowledge_funasr_chunked_raw_output.v1",
            "input": str(media),
            "duration_seconds": 600,
            "chunk_seconds": 300,
            "chunk_count": 2,
            "successful_chunk_count": 1,
            "successful_chunk_indexes": [0],
            "failed_chunk_count": 1,
            "failed_chunks": [
                {
                    "chunk_index": 1,
                    "start": 300,
                    "end": 600,
                    "reason": "unverified_empty_chunk",
                }
            ],
            "gaps": [
                {
                    "chunk_index": 1,
                    "start": 300,
                    "end": 600,
                    "reason": "unverified_empty_chunk",
                }
            ],
            "unresolved_chunk_indexes": [1],
            "chunk_results": [
                {
                    "chunk_index": 0,
                    "text": "第一块有讲话。",
                    "chunk_offset_seconds": 0,
                }
            ],
            "status": "degraded",
            "ok": False,
        },
    )
    _write_json(
        normalized,
        {
            "source_path": str(raw),
            "segments": [{"start": 0, "end": 300, "text": "第一块有讲话。"}],
        },
    )
    _write_json(
        root / "manifest.json",
        {
            "duration_seconds": 600,
            "normalized_transcript_json": "normalized-transcript.json",
        },
    )
    _write_json(
        root / "silero-vad-candidate.json",
        {
            "schema": "video_knowledge_pipeline.silero_vad_candidate.v1",
            "status": "completed",
            "candidate_only": True,
            "source_media": {
                "path": str(media),
                "sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
            },
            "upstream": {
                "project": "SYSTRAN/faster-whisper",
                "model": "silero_v5_bundled_onnx",
            },
            "segments": independent_speech,
        },
    )
    return root, normalized


def test_source_bound_silero_can_verify_an_empty_chunk_is_silence(
    tmp_path: Path,
) -> None:
    root, normalized = _fixture(
        tmp_path,
        independent_speech=[{"start": 0.0, "end": 120.0}],
    )

    result = assess_transcript_source_completeness(root, normalized)

    assert result["execution_integrity"] == "passed_with_verified_silence"
    assert result["speech_completeness_verified"] is True
    assert result["chunk_integrity"]["verified_silent_chunk_indexes"] == [1]
    assert result["chunk_integrity"]["unresolved_chunk_indexes"] == []
    assert "asr_chunk_integrity_failed" not in {
        row["kind"] for row in result["issues"]
    }
    assert "empty_asr_chunks_verified_silence" in {
        row["kind"] for row in result["issues"]
    }


def test_independent_speech_inside_empty_chunk_keeps_failure(
    tmp_path: Path,
) -> None:
    root, normalized = _fixture(
        tmp_path,
        independent_speech=[
            {"start": 0.0, "end": 120.0},
            {"start": 350.0, "end": 400.0},
        ],
    )

    result = assess_transcript_source_completeness(root, normalized)

    assert result["status"] == "failed"
    assert result["execution_integrity"] == "failed"
    assert result["speech_completeness_verified"] is False
    assert "asr_chunk_integrity_failed" in {
        row["kind"] for row in result["issues"]
    }
    assert "independent_vad_speech_gap" in {
        row["kind"] for row in result["issues"]
    }

def test_source_bound_silero_reconciles_legacy_successful_empty_chunk(
    tmp_path: Path,
) -> None:
    """Older runners must not require fabricated text for proven silence.

    Intent: cover the historical contract where an empty child response was
    counted as a successful chunk.
    Decision: reuse the same source-hash-bound Silero evidence as the modern
    failed-chunk path.
    Reason: production Bundles created before the strict runner fix have no
    ``failed_chunks`` row to reconcile.
    Evidence: the fixture mirrors the 2026-07-24 21/21 legacy receipt shape.
    Effective scope: quality interpretation only; the raw fixture is not
    rewritten by the assessment.
    """

    root, normalized = _fixture(
        tmp_path,
        independent_speech=[{"start": 0.0, "end": 120.0}],
    )
    raw = root / "transcripts" / "raw-asr-output.json"
    payload = json.loads(raw.read_text(encoding="utf-8"))
    payload.update(
        {
            "successful_chunk_count": 2,
            "successful_chunk_indexes": [0, 1],
            "failed_chunk_count": 0,
            "failed_chunks": [],
            "gaps": [],
            "unresolved_chunk_indexes": [],
            "chunk_results": [
                {
                    "chunk_index": 0,
                    "text": "第一块有讲话。",
                    "chunk_offset_seconds": 0,
                },
                {
                    "chunk_index": 1,
                    "text": "",
                    "timestamp": [],
                    "chunk_offset_seconds": 300,
                },
            ],
            "status": "completed",
            "ok": True,
        }
    )
    _write_json(raw, payload)

    result = assess_transcript_source_completeness(root, normalized)

    assert result["execution_integrity"] == "passed_with_verified_silence"
    assert result["speech_completeness_verified"] is True
    assert result["chunk_integrity"]["verified_silent_chunk_indexes"] == [1]
    assert result["chunk_integrity"]["unresolved_chunk_indexes"] == []
    assert "unverified_empty_asr_chunks" not in {
        row["kind"] for row in result["issues"]
    }
