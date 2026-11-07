from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file
from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .storage import write_json

SPEECH_EXECUTION_RECEIPT_SCHEMA = "video_creation_pipeline.speech_execution_receipt.v1"
FFMPEG_EXECUTION_RECEIPT_SCHEMA = "video_creation_pipeline.ffmpeg_execution_receipt.v1"
ROUGH_CUT_FINALIZE_RECEIPT_SCHEMA = "video_creation_pipeline.rough_cut_finalize_receipt.v1"
ROUGH_CUT_PROVENANCE_SCHEMA = "video_knowledge_pipeline.rough_cut_evidence_provenance.v1"
ROUGH_CUT_IMPORT_SCHEMA = "video_knowledge_pipeline.rough_cut_candidate_import.v1"
LOCAL_PROVIDER_STATUS_SCHEMA = "video_knowledge_pipeline.local_media_provider_status.v1"

CRISPASR_VERSION = "v0.8.18"
CRISPASR_COMMIT = "9deefe8f47273722415e4b4be5d87361b96177c9"
FFMPEG_OUTLET_ID = "video_creation_pipeline.single_ffmpeg_outlet"
FFMPEG_OPERATIONS = {
    "frame_extract",
    "render",
    "concat",
    "trim",
    "transcode",
    "audio_qa",
}

LLAMA_CPP_BUILD = "b8644"
LLAMA_CPP_COMMIT = "39b27f0da0271c06986cb31b68bc0fe68e780616"
SQLITE_VEC_VERSION = "v0.1.7"
SQLITE_VEC_COMMIT = "633eecf5067ab12ef331b3c4500c765f8e6d6da0"


def build_speech_execution_receipt(
    bundle_dir: str | Path,
    *,
    binary_path: str | Path,
    model_path: str | Path,
    input_audio_path: str | Path,
    transcript_path: str | Path,
    word_timestamps_path: str | Path,
    arbitration_path: str | Path,
    requested_backend: str,
    used_backend: str,
    device: str,
    attempts: list[dict[str, Any]],
    chunk_seconds: float,
    overlap_seconds: float,
    srt_path: str | Path | None = None,
    lcs_deduplication: bool = True,
    punctuation_split: bool = True,
    allowed_roots: Iterable[str | Path] | None = None,
    output_path: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Create the VKP-owned receipt for an already completed CrispASR run.

    This function never launches ASR. It binds the reviewed engine, exact
    artifacts, recovery policy, visible attempts, and VKP arbitration into the
    contract consumed read-only by the video-creation pipeline.
    """

    root = _bundle_root(bundle_dir)
    roots = _allowed_roots(root, allowed_roots)
    binary = _artifact_reference(binary_path, roots, "CrispASR binary")
    model = _artifact_reference(model_path, roots, "CrispASR model")
    audio = _artifact_reference(input_audio_path, roots, "speech input audio")
    transcript = _artifact_reference(transcript_path, roots, "arbitrated transcript")
    words = _artifact_reference(word_timestamps_path, roots, "word timestamps")
    arbitration = _artifact_reference(arbitration_path, roots, "VKP arbitration")
    _validate_arbitration_document(Path(arbitration["path"]))
    normalized_attempts = _validate_attempts(attempts)

    requested = _required_text(requested_backend, "requested_backend")
    used = _required_text(used_backend, "used_backend")
    resolved_device = _required_text(device, "device")
    if normalized_attempts[-1]["status"] != "completed":
        raise ValueError("final speech execution attempt must be completed")
    if normalized_attempts[-1]["backend"].lower() != used.lower():
        raise ValueError("used_backend must match the final completed attempt")
    cpu_retry = used.lower() == "cpu" and any(
        row["backend"].lower() != "cpu" and row["status"] == "failed"
        for row in normalized_attempts[:-1]
    )
    if cpu_retry and normalized_attempts[-1]["backend"].lower() != "cpu":
        raise ValueError("CPU retry must be the final visible attempt")

    chunk = _nonnegative_number(chunk_seconds, "chunk_seconds")
    overlap = _nonnegative_number(overlap_seconds, "overlap_seconds")
    if chunk and overlap >= chunk:
        raise ValueError("overlap_seconds must be shorter than chunk_seconds")
    if not lcs_deduplication or not punctuation_split:
        raise ValueError("CrispASR receipt requires LCS deduplication and punctuation splitting")

    outputs: dict[str, Any] = {
        "transcript": transcript,
        "word_timestamps": words,
    }
    if srt_path is not None:
        outputs["srt"] = _artifact_reference(srt_path, roots, "speech SRT")

    receipt: dict[str, Any] = {
        "schema": SPEECH_EXECUTION_RECEIPT_SCHEMA,
        "created_at": now_iso(),
        "owner": "video-knowledge-pipeline",
        "status": "completed",
        "engine": {
            "name": "CrispASR",
            "version": CRISPASR_VERSION,
            "source_commit": CRISPASR_COMMIT,
            "binary_sha256": binary["sha256"],
            "model_sha256": model["sha256"],
            "binary": binary,
            "model": model,
            "requested_backend": requested,
            "used_backend": used,
            "device": resolved_device,
            "cpu_retry_performed": cpu_retry,
        },
        "attempts": normalized_attempts,
        "input_audio": audio,
        "outputs": outputs,
        "chunking": {
            "seconds": chunk,
            "overlap_seconds": overlap,
            "lcs_deduplication": True,
            "punctuation_split": True,
        },
        "arbitration": {
            "performed_by": "video-knowledge-pipeline",
            "status": "accepted",
            "transcript_is_authoritative": True,
            "overwrites_source_evidence": False,
            "evidence": arbitration,
        },
        "execution_boundary": {
            "video_creation_asr_executed": False,
            "automatic_local_cloud_fallback": False,
            "second_asr_truth_created": False,
        },
    }
    receipt["receipt_sha256"] = _payload_sha256(receipt, "receipt_sha256")
    validate_speech_execution_receipt(receipt, roots)

    if write:
        destination = _output_path(
            root,
            output_path,
            "exports/media-execution/speech-execution-receipt.json",
        )
        write_json(destination, receipt)
        register_bundle_run(
            root,
            run_type="speech_execution_receipt",
            run_id=f"speech-receipt-{receipt['receipt_sha256'][:12]}",
            status="completed",
            title="VKP 本地 ASR 执行回执",
            summary=(
                f"CrispASR {CRISPASR_VERSION} completed with {len(normalized_attempts)} "
                "visible attempt(s); transcript remains VKP-authoritative."
            ),
            inputs={"input_audio": audio, "arbitration": arbitration},
            parameters={
                "requested_backend": requested,
                "used_backend": used,
                "cpu_retry_performed": cpu_retry,
                "chunk_seconds": chunk,
                "overlap_seconds": overlap,
            },
            artifacts=[destination, transcript["path"], words["path"]],
            operator_boundary=receipt["execution_boundary"],
            write=True,
        )
    return receipt


def validate_speech_execution_receipt(
    receipt: dict[str, Any],
    allowed_roots: Iterable[str | Path],
) -> None:
    if receipt.get("schema") != SPEECH_EXECUTION_RECEIPT_SCHEMA:
        raise ValueError("unsupported speech execution receipt schema")
    if receipt.get("receipt_sha256") != _payload_sha256(receipt, "receipt_sha256"):
        raise ValueError("speech execution receipt integrity check failed")
    if receipt.get("owner") != "video-knowledge-pipeline" or receipt.get("status") != "completed":
        raise ValueError("speech receipt must be a completed VKP-owned execution")
    engine = _object(receipt.get("engine"))
    if (
        engine.get("name") != "CrispASR"
        or engine.get("version") != CRISPASR_VERSION
        or engine.get("source_commit") != CRISPASR_COMMIT
    ):
        raise ValueError("speech receipt does not bind the reviewed CrispASR candidate")
    for field in ("binary", "model"):
        reference = _validate_artifact_reference(_object(engine.get(field)), allowed_roots, field)
        if engine.get(f"{field}_sha256") != reference["sha256"]:
            raise ValueError(f"speech engine {field} hash drifted")
    for field in ("requested_backend", "used_backend", "device"):
        _required_text(engine.get(field), field)
    if not isinstance(engine.get("cpu_retry_performed"), bool):
        raise ValueError("speech receipt must report CPU retry")
    attempts = _validate_attempts(receipt.get("attempts"))
    if engine["cpu_retry_performed"] and (
        len(attempts) < 2 or attempts[-1]["backend"].lower() != "cpu"
    ):
        raise ValueError("CrispASR CPU retry must be visible in attempts")
    _validate_artifact_reference(_object(receipt.get("input_audio")), allowed_roots, "input audio")
    outputs = _object(receipt.get("outputs"))
    _validate_artifact_reference(_object(outputs.get("transcript")), allowed_roots, "transcript")
    _validate_artifact_reference(
        _object(outputs.get("word_timestamps")), allowed_roots, "word timestamps"
    )
    if outputs.get("srt") is not None:
        _validate_artifact_reference(_object(outputs.get("srt")), allowed_roots, "SRT")
    chunking = _object(receipt.get("chunking"))
    seconds = _nonnegative_number(chunking.get("seconds"), "chunk seconds")
    overlap = _nonnegative_number(chunking.get("overlap_seconds"), "chunk overlap")
    if seconds and overlap >= seconds:
        raise ValueError("speech overlap must be shorter than a chunk")
    if chunking.get("lcs_deduplication") is not True or chunking.get("punctuation_split") is not True:
        raise ValueError("speech chunk recovery policy is incomplete")
    arbitration = _object(receipt.get("arbitration"))
    if (
        arbitration.get("performed_by") != "video-knowledge-pipeline"
        or arbitration.get("status") != "accepted"
        or arbitration.get("transcript_is_authoritative") is not True
        or arbitration.get("overwrites_source_evidence") is not False
    ):
        raise ValueError("speech receipt lacks accepted VKP arbitration")
    arbitration_ref = _validate_artifact_reference(
        _object(arbitration.get("evidence")), allowed_roots, "arbitration evidence"
    )
    _validate_arbitration_document(Path(arbitration_ref["path"]))
    boundary = _object(receipt.get("execution_boundary"))
    if any(
        boundary.get(field) is not False
        for field in (
            "video_creation_asr_executed",
            "automatic_local_cloud_fallback",
            "second_asr_truth_created",
        )
    ):
        raise ValueError("speech execution boundary is invalid")


def build_ffmpeg_execution_receipt(
    bundle_dir: str | Path,
    *,
    operation: str,
    ffmpeg_path: str | Path,
    actual_argv: list[str],
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    requested_backend: str,
    selected_backend: str,
    hardware_accelerated: bool,
    fallback_used: bool,
    fallback_reason: str = "",
    allowed_roots: Iterable[str | Path] | None = None,
    output_path: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Bind one already completed call from VKP's existing FFmpeg outlet."""

    root = _bundle_root(bundle_dir)
    roots = _allowed_roots(root, allowed_roots)
    if operation not in FFMPEG_OPERATIONS:
        raise ValueError(f"unsupported FFmpeg operation: {operation}")
    executable = _artifact_reference(ffmpeg_path, roots, "FFmpeg executable")
    if not isinstance(actual_argv, list) or not actual_argv:
        raise ValueError("actual_argv must be a non-empty list")
    resolved_argv = [_required_text(value, "FFmpeg argv item") for value in actual_argv]
    if Path(resolved_argv[0]).name.lower() not in {"ffmpeg", "ffmpeg.exe"}:
        raise ValueError("actual_argv must identify FFmpeg as argv[0]")
    canonical_command = ["ffmpeg", *resolved_argv[1:]]
    normalized_inputs = _artifact_rows(inputs, roots, "FFmpeg input")
    normalized_outputs = _artifact_rows(outputs, roots, "FFmpeg output")
    input_paths = {row["artifact"]["path"] for row in normalized_inputs}
    if any(row["artifact"]["path"] in input_paths for row in normalized_outputs):
        raise ValueError("FFmpeg receipt cannot claim source_media_modified=false when output overwrites an input")
    if operation == "frame_extract":
        for row in normalized_outputs:
            if row["role"] != "frame":
                raise ValueError("frame extraction outputs must use role=frame")
            row["media_time_s"] = _nonnegative_number(row.get("media_time_s"), "media_time_s")
    reason = str(fallback_reason or "").strip()
    if fallback_used and not reason:
        raise ValueError("FFmpeg fallback requires a visible reason")

    receipt: dict[str, Any] = {
        "schema": FFMPEG_EXECUTION_RECEIPT_SCHEMA,
        "created_at": now_iso(),
        "outlet_id": FFMPEG_OUTLET_ID,
        "status": "completed",
        "operation": operation,
        "execution_profile": {
            "selected_by": FFMPEG_OUTLET_ID,
            "requested_backend": _required_text(requested_backend, "requested_backend"),
            "selected_backend": _required_text(selected_backend, "selected_backend"),
            "hardware_accelerated": bool(hardware_accelerated),
            "fallback_used": bool(fallback_used),
            "fallback_reason": reason,
            "automatic_local_cloud_fallback": False,
            "ffmpeg_executable": executable,
        },
        "command": canonical_command,
        "actual_argv": resolved_argv,
        "inputs": normalized_inputs,
        "outputs": normalized_outputs,
        "execution_boundary": {
            "orchestration_owner": FFMPEG_OUTLET_ID,
            "second_ffmpeg_orchestrator_created": False,
            "source_media_modified": False,
            "automatic_publish": False,
        },
    }
    receipt["receipt_sha256"] = _payload_sha256(receipt, "receipt_sha256")
    validate_ffmpeg_execution_receipt(receipt, roots)

    if write:
        destination = _output_path(
            root,
            output_path,
            f"exports/media-execution/ffmpeg-{operation}-execution-receipt.json",
        )
        write_json(destination, receipt)
        register_bundle_run(
            root,
            run_type="ffmpeg_execution_receipt",
            run_id=f"ffmpeg-{operation}-{receipt['receipt_sha256'][:12]}",
            status="completed",
            title="VKP FFmpeg 单一出口执行回执",
            summary=(
                f"{operation} completed with backend {selected_backend}; "
                f"fallback_used={bool(fallback_used)}."
            ),
            inputs={"artifacts": normalized_inputs},
            parameters=receipt["execution_profile"],
            artifacts=[destination, *[row["artifact"]["path"] for row in normalized_outputs]],
            operator_boundary=receipt["execution_boundary"],
            write=True,
        )
    return receipt


def validate_ffmpeg_execution_receipt(
    receipt: dict[str, Any],
    allowed_roots: Iterable[str | Path],
) -> None:
    if receipt.get("schema") != FFMPEG_EXECUTION_RECEIPT_SCHEMA:
        raise ValueError("unsupported FFmpeg execution receipt schema")
    if receipt.get("receipt_sha256") != _payload_sha256(receipt, "receipt_sha256"):
        raise ValueError("FFmpeg execution receipt integrity check failed")
    if receipt.get("outlet_id") != FFMPEG_OUTLET_ID or receipt.get("status") != "completed":
        raise ValueError("FFmpeg receipt must come from the existing completed outlet")
    if receipt.get("operation") not in FFMPEG_OPERATIONS:
        raise ValueError("FFmpeg receipt operation is unsupported")
    profile = _object(receipt.get("execution_profile"))
    if profile.get("selected_by") != FFMPEG_OUTLET_ID:
        raise ValueError("FFmpeg profile was not selected by the outlet")
    _required_text(profile.get("requested_backend"), "requested_backend")
    _required_text(profile.get("selected_backend"), "selected_backend")
    if not isinstance(profile.get("hardware_accelerated"), bool):
        raise ValueError("FFmpeg profile must report acceleration")
    if not isinstance(profile.get("fallback_used"), bool):
        raise ValueError("FFmpeg profile must report fallback")
    if profile["fallback_used"] and not str(profile.get("fallback_reason") or "").strip():
        raise ValueError("FFmpeg fallback requires a visible reason")
    if profile.get("automatic_local_cloud_fallback") is not False:
        raise ValueError("FFmpeg cannot use local/cloud fallback")
    executable = _validate_artifact_reference(
        _object(profile.get("ffmpeg_executable")), allowed_roots, "FFmpeg executable"
    )
    actual = receipt.get("actual_argv")
    command = receipt.get("command")
    if not isinstance(actual, list) or not actual:
        raise ValueError("FFmpeg receipt lacks actual argv")
    if Path(str(actual[0])).resolve() != Path(executable["path"]).resolve():
        raise ValueError("FFmpeg actual argv does not bind the executable")
    if not isinstance(command, list) or not command or command[0] != "ffmpeg":
        raise ValueError("FFmpeg receipt lacks the canonical consumer argv")
    if command[1:] != actual[1:]:
        raise ValueError("FFmpeg canonical and actual argv drifted")
    inputs = receipt.get("inputs")
    outputs = receipt.get("outputs")
    if not isinstance(inputs, list) or not inputs or not isinstance(outputs, list) or not outputs:
        raise ValueError("FFmpeg receipt requires input and output artifacts")
    _validate_artifact_rows(inputs, allowed_roots, "FFmpeg input")
    _validate_artifact_rows(outputs, allowed_roots, "FFmpeg output")
    if receipt["operation"] == "frame_extract":
        for row in outputs:
            if row.get("role") != "frame":
                raise ValueError("frame extraction outputs must use role=frame")
            _nonnegative_number(row.get("media_time_s"), "media_time_s")
    boundary = _object(receipt.get("execution_boundary"))
    if boundary.get("orchestration_owner") != FFMPEG_OUTLET_ID or any(
        boundary.get(field) is not False
        for field in (
            "second_ffmpeg_orchestrator_created",
            "source_media_modified",
            "automatic_publish",
        )
    ):
        raise ValueError("FFmpeg execution boundary is invalid")


def import_rough_cut_finalize_receipt(
    bundle_dir: str | Path,
    *,
    receipt_path: str | Path,
    transcript_evidence: Iterable[str | Path] = (),
    ocr_evidence: Iterable[str | Path] = (),
    temporal_evidence: Iterable[str | Path] = (),
    evidence_gaps: dict[str, str] | None = None,
    allowed_roots: Iterable[str | Path] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Import an operator-finalized rough-cut receipt as candidate evidence only."""

    root = _bundle_root(bundle_dir)
    roots = _allowed_roots(root, allowed_roots)
    source = _require_file(receipt_path, roots, "rough-cut finalize receipt")
    receipt = _read_json(source, "rough-cut finalize receipt")
    _validate_rough_cut_finalize_receipt(receipt, roots)
    provenance = _build_provenance(
        transcript_evidence=transcript_evidence,
        ocr_evidence=ocr_evidence,
        temporal_evidence=temporal_evidence,
        evidence_gaps=evidence_gaps or {},
        allowed_roots=roots,
    )
    status = str(_object(receipt.get("readiness")).get("status") or "")
    result: dict[str, Any] = {
        "schema": ROUGH_CUT_IMPORT_SCHEMA,
        "generated_at": now_iso(),
        "status": "completed" if status == "ready_for_boundary_refinement" else "needs_review",
        "source_receipt": _artifact_reference(source, roots, "rough-cut finalize receipt"),
        "source_receipt_sha256": receipt["finalize_receipt_sha256"],
        "workspace_sha256": receipt["workspace_sha256"],
        "final_timeline_sha256": receipt["final_timeline_sha256"],
        "readiness": dict(receipt["readiness"]),
        "final_decisions": list(receipt["final_decisions"]),
        "unresolved_shots": list(receipt["unresolved_shots"]),
        "evidence_provenance": provenance,
        "authority_boundary": {
            "candidate_only": True,
            "mutates_vkp_timeline": False,
            "rough_cut_workspace_is_truth": False,
            "automatic_fallback": False,
            "automatic_publish": False,
            "next_executor": "videocut-kit boundary refinement after human confirmation",
        },
    }
    result["import_sha256"] = _payload_sha256(result, "import_sha256")
    if write:
        contract_dir = root / "exports" / "video-creation-contracts"
        contract_dir.mkdir(parents=True, exist_ok=True)
        receipt_copy = contract_dir / "rough-cut-finalize-receipt.json"
        if source != receipt_copy:
            shutil.copy2(source, receipt_copy)
        provenance_path = contract_dir / "rough-cut-evidence-provenance.json"
        import_path = contract_dir / "rough-cut-candidate-import.json"
        write_json(provenance_path, provenance)
        result["source_receipt"] = _artifact_reference(
            receipt_copy, [root], "copied rough-cut finalize receipt"
        )
        result["import_sha256"] = _payload_sha256(result, "import_sha256")
        write_json(import_path, result)
        register_bundle_run(
            root,
            run_type="rough_cut_candidate_import",
            run_id=f"rough-cut-{result['import_sha256'][:12]}",
            status=result["status"],
            title="粗剪人工确认候选导入",
            summary=(
                f"Imported {len(result['final_decisions'])} human-confirmed candidate "
                f"decision(s); Timeline truth was not changed."
            ),
            inputs={
                "source_receipt_sha256": result["source_receipt_sha256"],
                "workspace_sha256": result["workspace_sha256"],
            },
            parameters={"candidate_only": True, "readiness": status},
            artifacts=[import_path, provenance_path, receipt_copy],
            failed_items=[
                {"id": row.get("shot_id"), "reason": row.get("reason")}
                for row in result["unresolved_shots"]
            ],
            retry_command="video-creative-contracts finalize-rough-cut <reviewed inputs>",
            next_actions=[
                (
                    "交给 videocut-kit 做边界精修。"
                    if result["status"] == "completed"
                    else "先完成人工素材复核，再生成新的 finalize receipt。"
                )
            ],
            operator_boundary=result["authority_boundary"],
            write=True,
        )

    return result


def local_media_provider_status() -> dict[str, Any]:
    """Return reviewed candidates without claiming they are installed or healthy."""

    return {
        "schema": LOCAL_PROVIDER_STATUS_SCHEMA,
        "generated_at": now_iso(),
        "providers": [
            {
                "provider_id": "crispasr-v0-8-18",
                "capability": "asr",
                "version": CRISPASR_VERSION,
                "source_commit": CRISPASR_COMMIT,
                "status": "candidate_requires_real_benchmark",
                "execution_owner": "video-knowledge-pipeline",
                "candidate_only": False,
                "may_replace_timeline": False,
                "automatic_online_fallback": False,
            },
            {
                "provider_id": "llama-cpp-b8644",
                "capability": "local_multimodal_candidate_evidence",
                "version": LLAMA_CPP_BUILD,
                "source_commit": LLAMA_CPP_COMMIT,
                "status": "candidate_not_installed_by_vkp",
                "execution_owner": "video-knowledge-pipeline provider gateway",
                "candidate_only": True,
                "may_replace_timeline": False,
                "automatic_online_fallback": False,
            },
            {
                "provider_id": "sqlite-vec-v0-1-7",
                "capability": "material_candidate_index",
                "version": SQLITE_VEC_VERSION,
                "source_commit": SQLITE_VEC_COMMIT,
                "status": "candidate_requires_recall_benchmark",
                "execution_owner": "video-knowledge-pipeline run registry",
                "candidate_only": True,
                "may_replace_timeline": False,
                "may_replace_human_labels": False,
                "may_replace_metadata": False,
                "automatic_online_fallback": False,
            },
        ],
        "authority_boundary": {
            "starts_services": False,
            "downloads_models": False,
            "executes_models": False,
            "executes_ffmpeg": False,
            "publishes": False,
        },
    }


def _validate_rough_cut_finalize_receipt(
    receipt: dict[str, Any],
    allowed_roots: Iterable[str | Path],
) -> None:
    if receipt.get("schema") != ROUGH_CUT_FINALIZE_RECEIPT_SCHEMA:
        raise ValueError("unsupported rough-cut finalize receipt schema")
    if receipt.get("finalize_receipt_sha256") != _payload_sha256(
        receipt, "finalize_receipt_sha256"
    ):
        raise ValueError("rough-cut finalize receipt integrity check failed")
    _validate_artifact_reference(
        _object(receipt.get("source_workspace")), allowed_roots, "source workspace"
    )
    _validate_artifact_reference(
        _object(receipt.get("round_one_selection")), allowed_roots, "round-one selection"
    )
    second = receipt.get("round_two")
    if second is not None:
        for key in ("retry_plan", "candidates", "selection"):
            _validate_artifact_reference(
                _object(_object(second).get(key)), allowed_roots, f"round-two {key}"
            )
    decisions = receipt.get("final_decisions")
    unresolved = receipt.get("unresolved_shots")
    if not isinstance(decisions, list):
        raise ValueError("rough-cut finalize receipt lacks final decisions")
    if not isinstance(unresolved, list):
        raise ValueError("rough-cut finalize receipt lacks unresolved shots")
    accepted = [row for row in decisions if isinstance(row, dict) and row.get("status") == "accepted"]
    if receipt.get("final_timeline_sha256") != canonical_json_sha256(accepted):
        raise ValueError("rough-cut accepted-decision digest is invalid")
    readiness = _object(receipt.get("readiness"))
    ready = not unresolved
    if (
        readiness.get("render_eligible") is not ready
        or readiness.get("status")
        != ("ready_for_boundary_refinement" if ready else "unresolved_material")
        or readiness.get("next_action")
        != ("videocut_kit_boundary_refinement" if ready else "human_material_review")
    ):
        raise ValueError("rough-cut finalize readiness is inconsistent")
    confirmation = _object(receipt.get("operator_confirmation"))
    if (
        confirmation.get("confirmed") is not True
        or confirmation.get("method") != "visible_operator_finalization"
        or not str(confirmation.get("confirmed_by") or "").strip()
        or not str(confirmation.get("confirmed_at") or "").strip()
    ):
        raise ValueError("rough-cut finalize receipt lacks operator confirmation")
    boundary = _object(receipt.get("execution_boundary"))
    if boundary.get("candidate_only") is not True or any(
        boundary.get(field) is not False
        for field in (
            "timeline_truth_created",
            "ffmpeg_executed",
            "boundary_refinement_executed",
            "automatic_fallback",
            "automatic_publish",
        )
    ):
        raise ValueError("rough-cut finalize receipt boundary is invalid")


def _build_provenance(
    *,
    transcript_evidence: Iterable[str | Path],
    ocr_evidence: Iterable[str | Path],
    temporal_evidence: Iterable[str | Path],
    evidence_gaps: dict[str, str],
    allowed_roots: Iterable[str | Path],
) -> dict[str, Any]:
    channels: dict[str, Any] = {}
    for name, values in (
        ("transcript", transcript_evidence),
        ("ocr", ocr_evidence),
        ("temporal", temporal_evidence),
    ):
        artifacts = [
            _artifact_reference(path, allowed_roots, f"{name} evidence") for path in values
        ]
        gap = str(evidence_gaps.get(name) or "").strip()
        if not artifacts and not gap:
            raise ValueError(f"{name} provenance requires artifacts or an explicit gap reason")
        channels[name] = {
            "status": "available" if artifacts else "known_gap",
            "artifacts": artifacts,
            "gap_reason": gap,
        }
    result = {
        "schema": ROUGH_CUT_PROVENANCE_SCHEMA,
        "generated_at": now_iso(),
        "channels": channels,
        "policy": {
            "candidate_evidence_only": True,
            "timeline_truth_unchanged": True,
            "human_confirmation_required": True,
        },
    }
    result["provenance_sha256"] = _payload_sha256(result, "provenance_sha256")
    return result


def _validate_arbitration_document(path: Path) -> None:
    data = _read_json(path, "VKP arbitration")
    schema = str(data.get("schema") or "")
    status = str(data.get("status") or "")
    if not schema.startswith("video_knowledge_pipeline."):
        raise ValueError("arbitration evidence must be produced by VKP")
    if status not in {"accepted", "completed", "passed", "ready_for_import"}:
        raise ValueError("arbitration evidence is not accepted/completed")


def _artifact_rows(
    rows: list[dict[str, Any]],
    allowed_roots: Iterable[str | Path],
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} rows must be non-empty")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {index} must be an object")
        role = _required_text(row.get("role"), f"{label} role")
        normalized = {
            "role": role,
            "artifact": _artifact_reference(row.get("path"), allowed_roots, f"{label} {index}"),
        }
        for key, value in row.items():
            if key not in {"role", "path", "artifact"}:
                normalized[key] = value
        result.append(normalized)
    return result


def _validate_artifact_rows(
    rows: list[Any],
    allowed_roots: Iterable[str | Path],
    label: str,
) -> None:
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not str(row.get("role") or "").strip():
            raise ValueError(f"{label} row {index} is incomplete")
        _validate_artifact_reference(
            _object(row.get("artifact")), allowed_roots, f"{label} {index}"
        )


def _artifact_reference(
    value: str | Path | Any,
    allowed_roots: Iterable[str | Path],
    label: str,
) -> dict[str, Any]:
    path = _require_file(value, allowed_roots, label)
    content_kind = "json" if path.suffix.lower() == ".json" else "binary"
    reference: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "content_kind": content_kind,
        "canonical_sha256": None,
    }
    if content_kind == "json":
        reference["canonical_sha256"] = canonical_json_sha256(_read_json(path, label))
    return reference


def _validate_artifact_reference(
    reference: dict[str, Any],
    allowed_roots: Iterable[str | Path],
    label: str,
) -> dict[str, Any]:
    path = _require_file(reference.get("path"), allowed_roots, label)
    expected = _artifact_reference(path, allowed_roots, label)
    if reference != expected:
        raise ValueError(f"{label} artifact reference is stale")
    return expected


def _allowed_roots(
    bundle_root: Path,
    values: Iterable[str | Path] | None,
) -> tuple[Path, ...]:
    roots = [bundle_root]
    for value in values or ():
        root = Path(value).expanduser().resolve()
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def _require_file(
    value: str | Path | Any,
    allowed_roots: Iterable[str | Path],
    label: str,
) -> Path:
    path = Path(str(value or "")).expanduser().resolve()
    roots = tuple(Path(root).expanduser().resolve() for root in allowed_roots)
    if not any(path == root or path.is_relative_to(root) for root in roots):
        raise ValueError(f"{label} is outside allowed roots: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _bundle_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _output_path(root: Path, value: str | Path | None, default: str) -> Path:
    path = Path(value).expanduser().resolve() if value is not None else root / default
    if not (path == root or path.is_relative_to(root)):
        raise ValueError("receipt output must stay inside the Bundle")
    return path


def _validate_attempts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("speech execution requires visible attempts")
    result: list[dict[str, Any]] = []
    for index, attempt in enumerate(value, 1):
        if not isinstance(attempt, dict):
            raise ValueError(f"attempt {index} must be an object")
        row = {
            "backend": _required_text(attempt.get("backend"), f"attempt {index} backend"),
            "status": _required_text(attempt.get("status"), f"attempt {index} status"),
            "duration_ms": _nonnegative_number(
                attempt.get("duration_ms"), f"attempt {index} duration_ms"
            ),
        }
        if row["status"] not in {"completed", "failed"}:
            raise ValueError(f"attempt {index} status is unsupported")
        if attempt.get("error"):
            row["error"] = str(attempt["error"])
        result.append(row)
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object")
    return data


def _payload_sha256(value: dict[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return canonical_json_sha256(payload)


def _required_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    return text


def _nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative number") from exc
    if number < 0 or not math.isfinite(number):
        raise ValueError(f"{label} must be a non-negative number")
    return number


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _request(path: str | Path) -> dict[str, Any]:
    return _read_json(Path(path).expanduser().resolve(), "request")


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VKP local media contract front door")
    parser.add_argument(
        "command",
        choices=("speech-receipt", "ffmpeg-receipt", "rough-cut-import", "provider-status"),
    )
    parser.add_argument("--request")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "provider-status":
        result = local_media_provider_status()
    else:
        if not args.request:
            parser.error("--request is required for this command")
        request = _request(args.request)
        request["write"] = bool(args.write)
        if args.command == "speech-receipt":
            result = build_speech_execution_receipt(**request)
        elif args.command == "ffmpeg-receipt":
            result = build_ffmpeg_execution_receipt(**request)
        else:
            result = import_rough_cut_finalize_receipt(**request)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
