from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.asr_chunk_batch_merge import SCHEMA as MERGE_SCHEMA
from video_knowledge_pipeline.asr_response_quality import SCHEMA as QUALITY_SCHEMA
from video_knowledge_pipeline.asr_retry_snippets import prepare_asr_retry_snippets


def test_retry_snippet_plan_preserves_vad_windows_and_requires_new_consent(tmp_path: Path) -> None:
    media = tmp_path / "audio.wav"
    media.write_bytes(b"local-audio-fixture")
    quality = {
        "retry_plan": {
            "windows": [
                {
                    "retry_id": "retry-0001",
                    "source_segment_ids": ["segment-7"],
                    "start": 123.82,
                    "end": 153.8,
                    "alignment_source": "vad_boundary",
                    "reasons": ["task_instruction_leak"],
                }
            ]
        }
    }

    result = prepare_asr_retry_snippets(media, quality, tmp_path / "retry", execute=False)

    assert result["status"] == "planned"
    assert result["window_count"] == 1
    assert result["completed_count"] == 0
    assert result["failed_count"] == 0
    assert result["media_sha256"] == hashlib.sha256(media.read_bytes()).hexdigest()
    assert result["artifacts"][0]["alignment_source"] == "vad_boundary"
    assert result["artifacts"][0]["command"][-1].endswith(".wav")
    assert result["remote_retry_policy"]["requires_new_exact_consent"] is True
    assert result["remote_retry_policy"]["silent_provider_fallback_allowed"] is False
    saved = json.loads((tmp_path / "retry" / "asr-retry-snippets.json").read_text(encoding="utf-8"))
    assert saved["artifacts"][0]["source_segment_ids"] == ["segment-7"]


def test_retry_snippets_consume_chunk_merge_quality_and_verify_source_media(
    tmp_path: Path,
) -> None:
    media = tmp_path / "source.mp4"
    media.write_bytes(b"content-addressed-source-media")
    digest = hashlib.sha256(media.read_bytes()).hexdigest()
    merge_report = {
        "schema": MERGE_SCHEMA,
        "source_media": {
            "path": str(media),
            "bytes": media.stat().st_size,
            "sha256": digest,
        },
        "asr_quality": {
            "schema": QUALITY_SCHEMA,
            "retry_plan": {
                "windows": [
                    {
                        "retry_id": "vad-gap-0001",
                        "start": 10.0,
                        "end": 14.0,
                        "alignment_source": "vad_uncovered_speech",
                        "reasons": ["missing_speech_coverage"],
                    }
                ]
            },
        },
    }

    result = prepare_asr_retry_snippets(
        media, merge_report, tmp_path / "retry", execute=False
    )

    assert result["status"] == "planned"
    assert result["source_media_verified"] is True
    assert result["quality_report_schema"] == MERGE_SCHEMA
    assert result["quality_schema"] == QUALITY_SCHEMA
    assert result["artifacts"][0]["retry_id"] == "vad-gap-0001"
    assert result["artifacts"][0]["reasons"] == ["missing_speech_coverage"]


def test_retry_snippets_reject_chunk_merge_source_media_mismatch(
    tmp_path: Path,
) -> None:
    media = tmp_path / "source.mp4"
    media.write_bytes(b"current-source")
    report = {
        "schema": MERGE_SCHEMA,
        "source_media": {"bytes": media.stat().st_size, "sha256": "0" * 64},
        "asr_quality": {"retry_plan": {"windows": []}},
    }

    with pytest.raises(ValueError, match="SHA-256 does not match"):
        prepare_asr_retry_snippets(media, report, tmp_path / "retry", execute=False)
