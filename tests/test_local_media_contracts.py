from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.canonical_json import canonical_json_sha256
from video_knowledge_pipeline.local_media_contracts import (
    CRISPASR_COMMIT,
    CRISPASR_VERSION,
    FFMPEG_OUTLET_ID,
    LLAMA_CPP_COMMIT,
    ROUGH_CUT_FINALIZE_RECEIPT_SCHEMA,
    SQLITE_VEC_COMMIT,
    build_ffmpeg_execution_receipt,
    build_speech_execution_receipt,
    import_rough_cut_finalize_receipt,
    local_media_provider_status,
    validate_ffmpeg_execution_receipt,
    validate_speech_execution_receipt,
)
from video_knowledge_pipeline.storage import write_json


def _write(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")
    return path


def _payload_sha256(value: dict[str, object], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return canonical_json_sha256(payload)


def _artifact_reference(path: Path) -> dict[str, object]:
    from video_knowledge_pipeline.file_hash import sha256_file

    content_kind = "json" if path.suffix == ".json" else "binary"
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "content_kind": content_kind,
        "canonical_sha256": (
            canonical_json_sha256(json.loads(path.read_text(encoding="utf-8")))
            if content_kind == "json"
            else None
        ),
    }


def _speech_files(tmp_path: Path) -> dict[str, Path]:
    files = {
        "binary_path": _write(tmp_path / "crispasr.exe", b"binary"),
        "model_path": _write(tmp_path / "crispasr-model.bin", b"model"),
        "input_audio_path": _write(tmp_path / "audio.wav", b"RIFF-audio"),
        "transcript_path": _write(tmp_path / "transcript.txt", "测试逐字稿"),
        "word_timestamps_path": tmp_path / "words.json",
        "arbitration_path": tmp_path / "arbitration.json",
    }
    write_json(
        files["word_timestamps_path"],
        {"schema": "video_knowledge_pipeline.asr_word_timestamps.v1", "words": []},
    )
    write_json(
        files["arbitration_path"],
        {
            "schema": "video_knowledge_pipeline.transcript_source_arbitration.v1",
            "status": "accepted",
        },
    )
    return files


def test_speech_receipt_binds_crispasr_attempts_and_registry(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    files = _speech_files(tmp_path)
    receipt = build_speech_execution_receipt(
        bundle,
        **files,
        requested_backend="cuda",
        used_backend="cpu",
        device="CPU",
        attempts=[
            {
                "backend": "cuda",
                "status": "failed",
                "duration_ms": 1250,
                "error": "out of memory",
            },
            {"backend": "cpu", "status": "completed", "duration_ms": 2400},
        ],
        chunk_seconds=30,
        overlap_seconds=5,
        allowed_roots=[tmp_path],
        write=True,
    )

    assert receipt["engine"]["version"] == CRISPASR_VERSION
    assert receipt["engine"]["source_commit"] == CRISPASR_COMMIT
    assert receipt["engine"]["cpu_retry_performed"] is True
    assert receipt["chunking"]["lcs_deduplication"] is True
    assert receipt["arbitration"]["transcript_is_authoritative"] is True
    assert receipt["execution_boundary"]["automatic_local_cloud_fallback"] is False
    validate_speech_execution_receipt(receipt, [tmp_path])

    saved = json.loads(
        (bundle / "exports/media-execution/speech-execution-receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved == receipt
    registry = json.loads((bundle / "run-artifact-registry.json").read_text(encoding="utf-8"))
    row = next(item for item in registry["runs"] if item["run_type"] == "speech_execution_receipt")
    assert row["status"] == "completed"


def test_speech_receipt_rejects_unaccepted_arbitration(tmp_path: Path) -> None:
    files = _speech_files(tmp_path)
    write_json(
        files["arbitration_path"],
        {
            "schema": "video_knowledge_pipeline.transcript_source_arbitration.v1",
            "status": "needs_review",
        },
    )
    with pytest.raises(ValueError, match="not accepted"):
        build_speech_execution_receipt(
            tmp_path / "bundle",
            **files,
            requested_backend="cuda",
            used_backend="cuda",
            device="GPU 0",
            attempts=[{"backend": "cuda", "status": "completed", "duration_ms": 100}],
            chunk_seconds=30,
            overlap_seconds=5,
            allowed_roots=[tmp_path],
            write=False,
        )


def test_ffmpeg_receipt_records_actual_argv_and_explicit_fallback(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    ffmpeg = _write(tmp_path / "ffmpeg.exe", b"ffmpeg-binary")
    source = _write(tmp_path / "source.mp4", b"source")
    frame = _write(tmp_path / "frame.jpg", b"frame")
    receipt = build_ffmpeg_execution_receipt(
        bundle,
        operation="frame_extract",
        ffmpeg_path=ffmpeg,
        actual_argv=[str(ffmpeg), "-ss", "1.5", "-i", str(source), str(frame)],
        inputs=[{"role": "source_video", "path": source}],
        outputs=[{"role": "frame", "path": frame, "media_time_s": 1.5}],
        requested_backend="nvenc",
        selected_backend="cpu",
        hardware_accelerated=False,
        fallback_used=True,
        fallback_reason="NVENC unavailable after explicit probe",
        allowed_roots=[tmp_path],
        write=True,
    )

    assert receipt["outlet_id"] == FFMPEG_OUTLET_ID
    assert receipt["command"][0] == "ffmpeg"
    assert receipt["actual_argv"][0] == str(ffmpeg)
    assert receipt["execution_profile"]["fallback_reason"].startswith("NVENC")
    validate_ffmpeg_execution_receipt(receipt, [tmp_path])
    registry = json.loads((bundle / "run-artifact-registry.json").read_text(encoding="utf-8"))
    row = next(item for item in registry["runs"] if item["run_type"] == "ffmpeg_execution_receipt")
    assert row["operator_boundary"]["second_ffmpeg_orchestrator_created"] is False


def test_ffmpeg_receipt_rejects_silent_fallback(tmp_path: Path) -> None:
    ffmpeg = _write(tmp_path / "ffmpeg.exe", b"ffmpeg")
    source = _write(tmp_path / "source.mp4", b"source")
    output = _write(tmp_path / "output.mp4", b"output")
    with pytest.raises(ValueError, match="visible reason"):
        build_ffmpeg_execution_receipt(
            tmp_path / "bundle",
            operation="transcode",
            ffmpeg_path=ffmpeg,
            actual_argv=[str(ffmpeg), "-i", str(source), str(output)],
            inputs=[{"role": "source_video", "path": source}],
            outputs=[{"role": "transcoded_video", "path": output}],
            requested_backend="qsv",
            selected_backend="cpu",
            hardware_accelerated=False,
            fallback_used=True,
            allowed_roots=[tmp_path],
            write=False,
        )


def _rough_cut_receipt(tmp_path: Path) -> Path:
    workspace = tmp_path / "rough-workspace.json"
    selection = tmp_path / "rough-selection.json"
    write_json(workspace, {"schema": "video_creation_pipeline.rough_cut_workspace.v1"})
    write_json(selection, {"schema": "video_creation_pipeline.rough_cut_selection.v1"})
    receipt: dict[str, object] = {
        "schema": ROUGH_CUT_FINALIZE_RECEIPT_SCHEMA,
        "created_at": "2026-07-23T00:00:00+00:00",
        "source_workspace": _artifact_reference(workspace),
        "round_one_selection": _artifact_reference(selection),
        "round_two": None,
        "workspace_sha256": "1" * 64,
        "final_decisions": [
            {
                "shot_id": "shot-001",
                "status": "accepted",
                "candidate_id": "candidate-001",
                "reason": "",
                "human_confirmed": True,
                "source_round": 1,
            }
        ],
        "unresolved_shots": [],
        "final_timeline_sha256": "",
        "operator_confirmation": {
            "confirmed": True,
            "method": "visible_operator_finalization",
            "confirmed_by": "operator",
            "confirmed_at": "2026-07-23T00:00:00+00:00",
        },
        "readiness": {
            "status": "ready_for_boundary_refinement",
            "render_eligible": True,
            "next_action": "videocut_kit_boundary_refinement",
        },
        "execution_boundary": {
            "candidate_only": True,
            "timeline_truth_created": False,
            "ffmpeg_executed": False,
            "boundary_refinement_executed": False,
            "automatic_fallback": False,
            "automatic_publish": False,
        },
    }
    receipt["final_timeline_sha256"] = canonical_json_sha256(receipt["final_decisions"])
    receipt["finalize_receipt_sha256"] = _payload_sha256(
        receipt, "finalize_receipt_sha256"
    )
    path = tmp_path / "rough-cut-finalize-receipt.json"
    write_json(path, receipt)
    return path


def test_rough_cut_import_preserves_provenance_and_timeline_truth(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    timeline = bundle / "timeline.json"
    timeline.write_bytes(b'{"timeline_truth":"unchanged"}')
    before = timeline.read_bytes()
    receipt = _rough_cut_receipt(tmp_path)
    transcript = _write(tmp_path / "transcript.json", '{"segments":[]}')
    ocr = _write(tmp_path / "ocr.json", '{"pages":[]}')
    temporal = _write(tmp_path / "temporal.json", '{"groups":[]}')

    result = import_rough_cut_finalize_receipt(
        bundle,
        receipt_path=receipt,
        transcript_evidence=[transcript],
        ocr_evidence=[ocr],
        temporal_evidence=[temporal],
        allowed_roots=[tmp_path],
        write=True,
    )

    assert timeline.read_bytes() == before
    assert result["authority_boundary"]["candidate_only"] is True
    assert result["authority_boundary"]["mutates_vkp_timeline"] is False
    assert set(result["evidence_provenance"]["channels"]) == {
        "transcript",
        "ocr",
        "temporal",
    }
    saved = json.loads(
        (
            bundle
            / "exports/video-creation-contracts/rough-cut-candidate-import.json"
        ).read_text(encoding="utf-8")
    )
    assert saved["import_sha256"] == _payload_sha256(saved, "import_sha256")
    registry = json.loads((bundle / "run-artifact-registry.json").read_text(encoding="utf-8"))
    row = next(item for item in registry["runs"] if item["run_type"] == "rough_cut_candidate_import")
    assert row["parameters"]["candidate_only"] is True


def test_rough_cut_import_requires_explicit_gap_for_missing_channel(tmp_path: Path) -> None:
    receipt = _rough_cut_receipt(tmp_path)
    transcript = _write(tmp_path / "transcript.json", '{"segments":[]}')
    with pytest.raises(ValueError, match="ocr provenance"):
        import_rough_cut_finalize_receipt(
            tmp_path / "bundle",
            receipt_path=receipt,
            transcript_evidence=[transcript],
            evidence_gaps={"temporal": "no temporal evidence generated"},
            allowed_roots=[tmp_path],
            write=False,
        )


def test_local_provider_registry_is_candidate_only_and_nonexecuting() -> None:
    status = local_media_provider_status()
    by_id = {row["provider_id"]: row for row in status["providers"]}
    assert by_id["llama-cpp-b8644"]["source_commit"] == LLAMA_CPP_COMMIT
    assert by_id["llama-cpp-b8644"]["candidate_only"] is True
    assert by_id["sqlite-vec-v0-1-7"]["source_commit"] == SQLITE_VEC_COMMIT
    assert by_id["sqlite-vec-v0-1-7"]["may_replace_human_labels"] is False
    assert status["authority_boundary"]["downloads_models"] is False
    assert status["authority_boundary"]["executes_models"] is False

def test_speech_receipt_rejects_backend_attempt_mismatch(tmp_path: Path) -> None:
    files = _speech_files(tmp_path)
    with pytest.raises(ValueError, match="used_backend"):
        build_speech_execution_receipt(
            tmp_path / "bundle",
            **files,
            requested_backend="cuda",
            used_backend="cpu",
            device="CPU",
            attempts=[{"backend": "cuda", "status": "completed", "duration_ms": 100}],
            chunk_seconds=30,
            overlap_seconds=5,
            allowed_roots=[tmp_path],
            write=False,
        )


def test_ffmpeg_receipt_rejects_in_place_output(tmp_path: Path) -> None:
    ffmpeg = _write(tmp_path / "ffmpeg.exe", b"ffmpeg")
    source = _write(tmp_path / "source.mp4", b"source")
    with pytest.raises(ValueError, match="overwrites an input"):
        build_ffmpeg_execution_receipt(
            tmp_path / "bundle",
            operation="transcode",
            ffmpeg_path=ffmpeg,
            actual_argv=[str(ffmpeg), "-i", str(source), str(source)],
            inputs=[{"role": "source_video", "path": source}],
            outputs=[{"role": "transcoded_video", "path": source}],
            requested_backend="cpu",
            selected_backend="cpu",
            hardware_accelerated=False,
            fallback_used=False,
            allowed_roots=[tmp_path],
            write=False,
        )
