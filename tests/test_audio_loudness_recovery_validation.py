from __future__ import annotations

from pathlib import Path

import pytest

from video_knowledge_pipeline import audio_loudness_recovery_validation as validation
from video_knowledge_pipeline.audio_chunk_manifest import (
    SCHEMA as CHUNK_MANIFEST_SCHEMA,
    compute_audio_chunk_manifest_revision,
)
from video_knowledge_pipeline.audio_loudness_recovery import SCHEMA as RECOVERY_SCHEMA
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.silero_vad_candidate import SCHEMA as SILERO_SCHEMA
from video_knowledge_pipeline.storage import read_json, write_json


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _recovery_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "chunk-0001.wav"
    source.write_bytes(b"low-level-source")
    candidate = tmp_path / "chunk-0001.loudness-recovery-candidate.wav"
    candidate.write_bytes(b"normalized-candidate")
    report = tmp_path / "recovery.json"
    write_json(
        report,
        {
            "schema": RECOVERY_SCHEMA,
            "ok": True,
            "status": "candidate_requires_speech_vad",
            "candidate_only": True,
            "speech_proven": False,
            "asr_retry_authorized": False,
            "duration_seconds": 10.0,
            "source_media": _identity(source),
            "candidate_output": {
                **_identity(candidate),
                "exists": True,
                "produced_this_run": True,
            },
        },
    )
    return report, source, candidate


def _vad_report(
    path: Path,
    candidate: Path,
    *,
    segments: list[dict[str, float]],
) -> Path:
    write_json(
        path,
        {
            "schema": SILERO_SCHEMA,
            "ok": True,
            "status": "completed",
            "candidate_only": True,
            "source_media": _identity(candidate),
            "segments": segments,
            "segment_count": len(segments),
        },
    )
    return path


def _chunk_manifest(
    path: Path,
    *,
    parent: Path,
    chunk: Path,
    include_parent_sha256: bool = True,
    include_chunk_sha256: bool = True,
) -> Path:
    source = {
        "path": str(parent.resolve()),
        "bytes": parent.stat().st_size,
        "mtime_ns": parent.stat().st_mtime_ns,
        "duration_seconds": 30.0,
    }
    if include_parent_sha256:
        source["sha256"] = sha256_file(parent)
    chunk_row = {
        "index": 1,
        "artifact_path": str(chunk.resolve()),
        "bytes": chunk.stat().st_size,
        "start_seconds": 20.0,
        "end_seconds": 30.0,
        "duration_seconds": 10.0,
        "overlap_before_seconds": 0.0,
        "overlap_after_seconds": 0.0,
    }
    if include_chunk_sha256:
        chunk_row["sha256"] = sha256_file(chunk)
    manifest = {
        "schema": CHUNK_MANIFEST_SCHEMA,
        "source": source,
        "strategy": {"mode": "fixed"},
        "silence_detection": {},
        "chunks": [chunk_row],
    }
    manifest["revision"] = compute_audio_chunk_manifest_revision(manifest)
    write_json(path, manifest)
    return path


def test_validator_requires_independent_vad_before_retry(
    tmp_path: Path,
) -> None:
    report, _, _ = _recovery_fixture(tmp_path)

    result = validation.validate_low_level_audio_candidate(report, write=False)

    assert result["status"] == "vad_required"
    assert result["ok"] is True
    assert result["targeted_retry_recommended"] is False
    assert result["operator_boundary"]["automatic_asr_execution"] is False


def test_validator_rejects_vad_from_a_different_candidate(
    tmp_path: Path,
) -> None:
    report, _, candidate = _recovery_fixture(tmp_path)
    other = tmp_path / "other.wav"
    other.write_bytes(b"other")
    vad = _vad_report(
        tmp_path / "vad.json",
        other,
        segments=[{"start": 1.0, "end": 2.0}],
    )

    result = validation.validate_low_level_audio_candidate(
        report,
        vad_report_path=vad,
        write=False,
    )

    assert result["status"] == "invalid_vad_lineage"
    assert "vad_candidate_path_mismatch" in result["errors"]
    assert "vad_candidate_sha256_mismatch" in result["errors"]
    assert candidate.exists()


def test_no_speech_vad_does_not_plan_asr(
    tmp_path: Path,
) -> None:
    report, _, candidate = _recovery_fixture(tmp_path)
    vad = _vad_report(tmp_path / "vad.json", candidate, segments=[])

    result = validation.validate_low_level_audio_candidate(
        report,
        vad_report_path=vad,
        write=False,
    )

    assert result["status"] == "no_speech_detected"
    assert result["speech_evidence"]["passed"] is False
    assert result["targeted_retry_recommended"] is False
    assert result["targeted_retry_plan"] == {}


def test_speech_without_parent_lineage_is_confirmed_but_not_merge_ready(
    tmp_path: Path,
) -> None:
    report, _, candidate = _recovery_fixture(tmp_path)
    vad = _vad_report(
        tmp_path / "vad.json",
        candidate,
        segments=[{"start": 1.0, "end": 3.0}],
    )

    result = validation.validate_low_level_audio_candidate(
        report,
        vad_report_path=vad,
        write=False,
    )

    assert result["status"] == "speech_candidate_confirmed"
    assert result["speech_evidence"]["passed"] is True
    assert result["targeted_retry_recommended"] is False
    assert result["targeted_retry_plan"] == {}


def test_exact_manifest_maps_candidate_speech_to_global_retry_windows(
    tmp_path: Path,
) -> None:
    report, source, candidate = _recovery_fixture(tmp_path)
    parent = tmp_path / "parent.wav"
    parent.write_bytes(b"parent-media")
    vad = _vad_report(
        tmp_path / "vad.json",
        candidate,
        segments=[
            {"start": 1.0, "end": 2.0},
            {"start": 1.8, "end": 4.0},
        ],
    )
    manifest = _chunk_manifest(
        tmp_path / "audio-chunk-manifest.json",
        parent=parent,
        chunk=source,
    )

    result = validation.validate_low_level_audio_candidate(
        report,
        vad_report_path=vad,
        chunk_manifest_path=manifest,
        chunk_index=1,
        write=False,
    )

    assert result["status"] == "targeted_retry_planned"
    assert result["targeted_retry_recommended"] is True
    plan = result["targeted_retry_plan"]
    assert plan["input_audio"]["sha256"] == sha256_file(candidate)
    assert plan["parent_source"]["sha256"] == sha256_file(parent)
    assert plan["global_offset_seconds"] == 20.0
    assert plan["speech_intervals_local"] == [
        {"start": 1.0, "end": 4.0, "duration_seconds": 3.0}
    ]
    assert plan["speech_intervals_global"] == [
        {"start": 21.0, "end": 24.0, "duration_seconds": 3.0}
    ]
    assert plan["automatic_execution"] is False
    assert plan["canonical_transcript_modified"] is False


def test_legacy_manifest_without_parent_sha_is_not_merge_ready(
    tmp_path: Path,
) -> None:
    report, source, candidate = _recovery_fixture(tmp_path)
    parent = tmp_path / "parent.wav"
    parent.write_bytes(b"parent-media")
    vad = _vad_report(
        tmp_path / "vad.json",
        candidate,
        segments=[{"start": 1.0, "end": 3.0}],
    )
    manifest = _chunk_manifest(
        tmp_path / "legacy-manifest.json",
        parent=parent,
        chunk=source,
        include_parent_sha256=False,
    )

    result = validation.validate_low_level_audio_candidate(
        report,
        vad_report_path=vad,
        chunk_manifest_path=manifest,
        chunk_index=1,
        write=False,
    )

    assert result["status"] == "invalid_chunk_lineage"
    assert "parent_source_sha256_missing" in result["errors"]
    assert result["targeted_retry_recommended"] is False


def test_tampered_manifest_revision_is_not_merge_ready(tmp_path: Path) -> None:
    report, source, candidate = _recovery_fixture(tmp_path)
    parent = tmp_path / "parent.wav"
    parent.write_bytes(b"parent")
    vad = _vad_report(
        tmp_path / "vad.json",
        candidate,
        segments=[{"start": 1.0, "end": 2.0}],
    )
    manifest_path = _chunk_manifest(
        tmp_path / "manifest.json",
        parent=parent,
        chunk=source,
    )
    manifest = read_json(manifest_path)
    manifest["revision"] = "f" * 64
    write_json(manifest_path, manifest)

    result = validation.validate_low_level_audio_candidate(
        report,
        vad_report_path=vad,
        chunk_manifest_path=manifest_path,
        chunk_index=1,
        write=False,
    )

    assert result["status"] == "invalid_chunk_lineage"
    assert "chunk_manifest_revision_mismatch" in result["errors"]
    assert result["targeted_retry_recommended"] is False


def test_manifest_chunk_without_sha_is_not_merge_ready(tmp_path: Path) -> None:
    report, source, candidate = _recovery_fixture(tmp_path)
    parent = tmp_path / "parent.wav"
    parent.write_bytes(b"parent")
    vad = _vad_report(
        tmp_path / "vad.json",
        candidate,
        segments=[{"start": 1.0, "end": 2.0}],
    )
    manifest_path = _chunk_manifest(
        tmp_path / "manifest.json",
        parent=parent,
        chunk=source,
        include_chunk_sha256=False,
    )

    result = validation.validate_low_level_audio_candidate(
        report,
        vad_report_path=vad,
        chunk_manifest_path=manifest_path,
        chunk_index=1,
        write=False,
    )

    assert result["status"] == "invalid_chunk_lineage"
    assert "manifest_chunk_sha256_missing" in result["errors"]
    assert result["targeted_retry_recommended"] is False


def test_parent_mutation_invalidates_chunk_lineage(tmp_path: Path) -> None:
    report, source, candidate = _recovery_fixture(tmp_path)
    parent = tmp_path / "parent.wav"
    parent.write_bytes(b"parent")
    vad = _vad_report(
        tmp_path / "vad.json",
        candidate,
        segments=[{"start": 1.0, "end": 2.0}],
    )
    manifest_path = _chunk_manifest(
        tmp_path / "manifest.json",
        parent=parent,
        chunk=source,
    )
    parent.write_bytes(b"changed-parent")

    result = validation.validate_low_level_audio_candidate(
        report,
        vad_report_path=vad,
        chunk_manifest_path=manifest_path,
        chunk_index=1,
        write=False,
    )

    assert result["status"] == "invalid_chunk_lineage"
    assert "parent_source_sha256_changed" in result["errors"]
    assert result["targeted_retry_recommended"] is False


def test_manifest_chunk_hash_cannot_be_rebound_without_recovery_match(
    tmp_path: Path,
) -> None:
    report, source, candidate = _recovery_fixture(tmp_path)
    parent = tmp_path / "parent.wav"
    parent.write_bytes(b"parent")
    vad = _vad_report(
        tmp_path / "vad.json",
        candidate,
        segments=[{"start": 1.0, "end": 2.0}],
    )
    manifest_path = _chunk_manifest(
        tmp_path / "manifest.json",
        parent=parent,
        chunk=source,
    )
    manifest = read_json(manifest_path)
    manifest["chunks"][0]["sha256"] = "e" * 64
    manifest["revision"] = compute_audio_chunk_manifest_revision(manifest)
    write_json(manifest_path, manifest)

    result = validation.validate_low_level_audio_candidate(
        report,
        vad_report_path=vad,
        chunk_manifest_path=manifest_path,
        chunk_index=1,
        write=False,
    )

    assert result["status"] == "invalid_chunk_lineage"
    assert "manifest_chunk_recovery_sha256_mismatch" in result["errors"]
    assert result["targeted_retry_recommended"] is False


def test_mutated_candidate_invalidates_recovery_lineage(
    tmp_path: Path,
) -> None:
    report, _, candidate = _recovery_fixture(tmp_path)
    candidate.write_bytes(b"mutated-after-recovery")

    result = validation.validate_low_level_audio_candidate(report, write=False)

    assert result["status"] == "invalid_recovery_lineage"
    assert "candidate_sha256_changed" in result["errors"]


def test_execute_vad_delegates_to_existing_silero_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, _, candidate = _recovery_fixture(tmp_path)
    captured: dict[str, object] = {}

    def fake_silero(
        media_path: str | Path,
        **kwargs: object,
    ) -> dict[str, object]:
        captured["media_path"] = str(Path(media_path).resolve())
        captured.update(kwargs)
        return {
            "schema": SILERO_SCHEMA,
            "ok": True,
            "status": "completed",
            "candidate_only": True,
            "source_media": _identity(candidate),
            "segments": [{"start": 1.0, "end": 2.0}],
            "segment_count": 1,
        }

    monkeypatch.setattr(validation, "run_silero_vad_candidate", fake_silero)

    result = validation.validate_low_level_audio_candidate(
        report,
        execute_vad=True,
        output_path=tmp_path / "validation.json",
        write=True,
    )

    assert result["status"] == "speech_candidate_confirmed"
    assert captured["media_path"] == str(candidate.resolve())
    assert captured["execute"] is True
    assert captured["write"] is True
