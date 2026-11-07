from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file
from .long_video_fast_segment import build_long_video_fast_segment_plan
from .long_video_memory_pack import build_long_video_memory_pack
from .models import now_iso
from .path_defaults import workspace_root
from .storage import read_json, write_json
from .video_edit_review_pack import build_video_edit_review_pack


CLOSURE_SCHEMA = "video_knowledge_pipeline.small_model_video_closure.v1"
DOCTOR_SCHEMA = "video_knowledge_pipeline.video_closure_doctor.v1"
EVIDENCE_SCHEMA = "video_knowledge_pipeline.video_closure_evidence.v1"
SEGMENT_PLAN_SCHEMA = "video_knowledge_pipeline.video_closure_segment_plan.v1"
HANDOFF_SCHEMA = "video_knowledge_pipeline.candidate_editing_handoff.v1"
REVIEW_PACKET_SCHEMA = "video_knowledge_pipeline.video_closure_review_packet.v1"
STAGE_RECEIPT_SCHEMA = "video_knowledge_pipeline.video_closure_stage_receipt.v1"
FAILURE_RECEIPT_SCHEMA = "video_knowledge_pipeline.video_closure_failure_receipt.v1"
CONTENT_BRIEF_SCHEMA = "content-video-brief.v1"
MVP_EDIT_JOB_SCHEMA = "video_creation_pipeline.vkp_edit_job.v3"
CONTRACT_VERSION = "1.0.0"
AUTO_RETRY_COUNT = 0
TIMESTAMP_TOLERANCE_SECONDS = 0.001
LOW_CONFIDENCE_THRESHOLD = 0.60
STAGES = ("evidence", "segment_plan", "content_brief", "candidate_handoff", "review_packet")
STAGE_FILENAMES = {
    "evidence": "evidence-pack.json",
    "segment_plan": "segment-plan.json",
    "content_brief": "content-video-brief.json",
    "candidate_handoff": "candidate-editing-handoff.json",
    "review_packet": "review-packet.json",
}
MVP_CONTRACT_FILES = (
    "src/video_creation_pipeline/vkp_bridge.py",
    "src/video_creation_pipeline/cli.py",
    "tests/fixtures/content-video-brief-v1/upstream-canonical/content-video-brief.v1.schema.json",
)
SYNTHETIC_MEDIA_MAGIC = b"VKP_SYNTHETIC_MEDIA_FIXTURE_V1\n"


class VideoClosureError(RuntimeError):
    """Fail-closed error carrying a stable code for small-model recovery."""

    def __init__(self, error_code: str, detail: str = "") -> None:
        self.error_code = str(error_code or "video_closure_failed")
        self.detail = str(detail or "")
        super().__init__(f"{self.error_code}: {self.detail}".rstrip(": "))


def video_closure_doctor(
    bundle_dir: str | Path | None = None,
    *,
    mvp_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a bounded no-write readiness and command catalog."""

    bundle = Path(bundle_dir).expanduser().resolve() if bundle_dir else None
    input_status: dict[str, Any] = {"status": "not_supplied", "blockers": []}
    if bundle is not None:
        try:
            context = _build_input_context(bundle)
            input_status = {
                "status": "ready",
                "bundle": str(bundle),
                "input_identity_sha256": context["input_identity_sha256"],
                "duration_seconds": context["duration_seconds"],
                "timeline_count": len(context["timeline"]),
                "transcript_segment_count": len(context["segments"]),
                "blockers": [],
            }
        except VideoClosureError as exc:
            input_status = {
                "status": "blocked",
                "bundle": str(bundle),
                "blockers": [{"error_code": exc.error_code, "detail": exc.detail}],
            }
    downstream = _mvp_contract_snapshot(_resolve_mvp_root(mvp_root))
    bundle_arg = _ps_quote(bundle) if bundle else "<bundle>"
    output_arg = "<closure-output>"
    commands = {
        "doctor": f".\\scripts\\video-knowledge.ps1 video-closure-doctor {bundle_arg}",
        "run": f".\\scripts\\video-knowledge.ps1 video-closure-run {bundle_arg} --output-dir {output_arg}",
        "resume": f".\\scripts\\video-knowledge.ps1 video-closure-resume {output_arg}",
        "status": f".\\scripts\\video-knowledge.ps1 video-closure-status {output_arg}",
    }
    return {
        "schema": DOCTOR_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "status": "ready" if input_status["status"] in {"ready", "not_supplied"} else "blocked",
        "ready_to_run": input_status["status"] == "ready" and downstream["status"] != "dirty",
        "ready_for_mvp_candidate_handoff": downstream["status"] == "ready",
        "input": input_status,
        "downstream_contract": _compact_downstream_snapshot(downstream),
        "commands": commands,
        "stage_order": list(STAGES),
        "output_boundary": ["candidate_only", "review_required"],
        "auto_retry_count": AUTO_RETRY_COUNT,
        "resume_policy": "explicit_only_hash_bound_idempotent",
        "operator_boundary": {
            "reads_source_media_for_identity": True,
            "decodes_source_media": False,
            "probes_media_metadata_only_if_manifest_duration_missing": True,
            "executes_ffmpeg": False,
            "executes_asr_or_ocr": False,
            "calls_provider": False,
            "uploads_or_publishes": False,
            "overwrites_source_media": False,
        },
    }


def run_video_closure(
    bundle_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    closure_run_id: str | None = None,
    mvp_root: str | Path | None = None,
    pause_after: str | None = None,
    execution_mode: str = "candidate_only",
    _resume: bool = False,
) -> dict[str, Any]:
    """Run the five-stage candidate-only closure with zero automatic retries."""

    bundle = Path(bundle_dir).expanduser().resolve()
    run_id = _validated_run_id(closure_run_id or _default_run_id(bundle))
    run_root = _run_root(bundle, output_dir, run_id)
    if execution_mode != "candidate_only":
        error = VideoClosureError(
            "real_execution_not_authorized",
            "video closure emits plans and review artifacts only; rendering requires a separate human-approved MVP action",
        )
        _safe_record_failure(run_root, run_id, "preflight", error)
        raise error
    if pause_after is not None and pause_after not in STAGES:
        error = VideoClosureError("invalid_pause_stage", str(pause_after))
        _safe_record_failure(run_root, run_id, "preflight", error)
        raise error

    manifest_path = run_root / "closure-run.json"
    try:
        if manifest_path.is_file():
            manifest = _read_object(manifest_path, "closure manifest", "corrupt_closure_manifest")
            _validate_manifest_identity(manifest, bundle, run_id)
            _validate_current_input_identity(manifest)
            if not _resume:
                if manifest.get("status") == "review_required":
                    result = _public_status(manifest, run_root)
                    result["idempotent_no_op"] = True
                    return result
                raise VideoClosureError(
                    "explicit_resume_required",
                    str(manifest.get("recovery", {}).get("resume_command") or _resume_command(run_root)),
                )
        else:
            context = _build_input_context(bundle)
            downstream = _mvp_contract_snapshot(_resolve_mvp_root(mvp_root))
            if downstream["status"] == "dirty":
                raise VideoClosureError(
                    "downstream_contract_dirty",
                    "MVP bridge or content-video-brief contract has uncommitted changes",
                )
            run_root.mkdir(parents=True, exist_ok=True)
            manifest = {
                "schema": CLOSURE_SCHEMA,
                "contract_version": CONTRACT_VERSION,
                "closure_run_id": run_id,
                "created_at": now_iso(),
                "bundle_dir": str(bundle),
                "run_root": str(run_root),
                "mode": "candidate_only",
                "status": "running",
                "completed_stages": [],
                "next_stage": STAGES[0],
                "input_snapshot": context["input_snapshot"],
                "input_identity_sha256": context["input_identity_sha256"],
                "downstream_contract_snapshot": downstream,
                "auto_retry_count": AUTO_RETRY_COUNT,
                "automatic_retry_allowed": False,
                "accepted": False,
                "published": False,
                "source_media_modified": False,
                "recovery": {"resume_command": _resume_command(run_root)},
            }
            write_json(manifest_path, manifest)
        return _advance_closure(
            run_root,
            manifest,
            mvp_root=mvp_root,
            pause_after=pause_after,
        )
    except VideoClosureError as exc:
        _mark_blocked_if_possible(manifest_path, run_root, exc)
        _safe_record_failure(run_root, run_id, _failure_stage(manifest_path), exc)
        raise
    except Exception as exc:  # noqa: BLE001 - converted to a stable fail-closed receipt.
        wrapped = VideoClosureError("unexpected_closure_failure", str(exc))
        _mark_blocked_if_possible(manifest_path, run_root, wrapped)
        _safe_record_failure(run_root, run_id, _failure_stage(manifest_path), wrapped)
        raise wrapped from exc


def resume_video_closure(
    closure_dir: str | Path,
    *,
    mvp_root: str | Path | None = None,
) -> dict[str, Any]:
    run_root = Path(closure_dir).expanduser().resolve()
    manifest = _read_object(run_root / "closure-run.json", "closure manifest", "corrupt_closure_manifest")
    _validate_current_input_identity(manifest)
    if manifest.get("status") == "review_required":
        _validate_all_receipts(run_root, manifest)
        result = _public_status(manifest, run_root)
        result["idempotent_no_op"] = True
        return result
    return run_video_closure(
        str(manifest.get("bundle_dir") or ""),
        output_dir=run_root,
        closure_run_id=str(manifest.get("closure_run_id") or ""),
        mvp_root=mvp_root,
        _resume=True,
    )


def video_closure_status(closure_dir: str | Path) -> dict[str, Any]:
    run_root = Path(closure_dir).expanduser().resolve()
    manifest = _read_object(run_root / "closure-run.json", "closure manifest", "corrupt_closure_manifest")
    _validate_current_input_identity(manifest)
    _validate_all_receipts(run_root, manifest)
    return _public_status(manifest, run_root)


def finalize_video_closure(closure_dir: str | Path) -> dict[str, Any]:
    run_root = Path(closure_dir).expanduser().resolve()
    manifest_path = run_root / "closure-run.json"
    manifest = _read_object(manifest_path, "closure manifest", "corrupt_closure_manifest")
    completed = [str(value) for value in manifest.get("completed_stages") or []]
    if completed != list(STAGES):
        raise VideoClosureError(
            "premature_finalize",
            f"expected stages {','.join(STAGES)}; completed {','.join(completed)}",
        )
    _validate_all_receipts(run_root, manifest)
    manifest.update(
        {
            "status": "review_required",
            "mode": "candidate_only",
            "next_stage": None,
            "finalized_at": manifest.get("finalized_at") or now_iso(),
            "accepted": False,
            "published": False,
            "source_media_modified": False,
        }
    )
    write_json(manifest_path, manifest)
    return _public_status(manifest, run_root)


def _advance_closure(
    run_root: Path,
    manifest: dict[str, Any],
    *,
    mvp_root: str | Path | None,
    pause_after: str | None,
) -> dict[str, Any]:
    bundle = Path(str(manifest["bundle_dir"])).resolve()
    context = _build_input_context(bundle)
    if context["input_identity_sha256"] != manifest.get("input_identity_sha256"):
        raise VideoClosureError("input_drift", "Bundle evidence changed after closure start")
    downstream = _validated_downstream_for_resume(manifest, mvp_root)
    _validate_all_receipts(run_root, manifest)
    completed = [str(value) for value in manifest.get("completed_stages") or []]

    for stage in STAGES[len(completed) :]:
        artifacts = _execute_stage(stage, run_root, manifest, context, downstream)
        receipt = _stage_receipt(run_root, manifest, stage, artifacts)
        receipt_path = _receipt_path(run_root, stage)
        if receipt_path.exists():
            raise VideoClosureError("duplicate_stage_receipt", stage)
        write_json(receipt_path, receipt)
        completed.append(stage)
        manifest["completed_stages"] = list(completed)
        manifest["next_stage"] = STAGES[len(completed)] if len(completed) < len(STAGES) else None
        manifest["status"] = "running"
        manifest["last_stage_receipt"] = str(receipt_path)
        write_json(run_root / "closure-run.json", manifest)
        if pause_after == stage and stage != STAGES[-1]:
            manifest["status"] = "paused"
            manifest["paused_after"] = stage
            manifest["recovery"] = {"resume_command": _resume_command(run_root)}
            write_json(run_root / "closure-run.json", manifest)
            return _public_status(manifest, run_root)
    return finalize_video_closure(run_root)


def _execute_stage(
    stage: str,
    run_root: Path,
    manifest: dict[str, Any],
    context: dict[str, Any],
    downstream: dict[str, Any],
) -> list[Path]:
    output = run_root / "artifacts" / STAGE_FILENAMES[stage]
    if output.exists():
        raise VideoClosureError("orphan_stage_artifact", str(output))
    if stage == "evidence":
        payload = _build_evidence_pack(manifest, context)
        write_json(output, payload)
        return [output]
    if stage == "segment_plan":
        evidence = _read_stage_artifact(run_root, "evidence")
        payload = _build_segment_plan(manifest, context, evidence)
        write_json(output, payload)
        return [output]
    if stage == "content_brief":
        evidence = _read_stage_artifact(run_root, "evidence")
        plan = _read_stage_artifact(run_root, "segment_plan")
        payload = _build_content_brief(manifest, evidence, plan)
        _validate_content_brief(payload, downstream)
        write_json(output, payload)
        return [output]
    if stage == "candidate_handoff":
        brief_path = run_root / "artifacts" / STAGE_FILENAMES["content_brief"]
        plan_path = run_root / "artifacts" / STAGE_FILENAMES["segment_plan"]
        extra: list[Path] = []
        edit_job: dict[str, Any] | None = None
        edit_job_path = run_root / "artifacts" / "mvp-vkp-edit-job.json"
        if downstream["status"] == "ready":
            edit_job = _invoke_mvp_bridge(Path(downstream["root"]), Path(manifest["bundle_dir"]), edit_job_path)
            _validate_mvp_edit_job(edit_job, Path(manifest["bundle_dir"]))
            if not edit_job_path.is_file():
                write_json(edit_job_path, edit_job)
            extra.append(edit_job_path)
        payload = _build_candidate_handoff(manifest, brief_path, plan_path, downstream, edit_job_path if edit_job else None)
        write_json(output, payload)
        return [output, *extra]
    if stage == "review_packet":
        payload = _build_review_packet(run_root, manifest, context)
        write_json(output, payload)
        return [output]
    raise VideoClosureError("unknown_stage", stage)


def _build_input_context(bundle: Path) -> dict[str, Any]:
    if not bundle.is_dir():
        raise VideoClosureError("bundle_not_found", str(bundle))
    manifest_path = bundle / "manifest.json"
    timeline_path = bundle / "timeline.json"
    manifest = _read_object(manifest_path, "manifest.json", "corrupt_manifest")
    timeline = _read_array(timeline_path, "timeline.json", "corrupt_timeline")
    if not timeline:
        raise VideoClosureError("missing_evidence", "timeline.json contains no evidence rows")
    transcript_path = _bundle_artifact(bundle, manifest.get("normalized_transcript"), "normalized-transcript.json", "transcript")
    transcript = _read_object(transcript_path, transcript_path.name, "corrupt_transcript")
    segments = [row for row in transcript.get("segments") or [] if isinstance(row, dict)]
    if not segments:
        raise VideoClosureError("missing_evidence", "normalized transcript contains no segments")
    media_path = _bundle_artifact(bundle, manifest.get("source_video"), "source.mp4", "source media")
    if media_path.stat().st_size <= 0:
        raise VideoClosureError("corrupt_media", "source media is empty")
    expected_media_hash = str(manifest.get("source_video_sha256") or "").lower()
    actual_media_hash = sha256_file(media_path)
    if expected_media_hash and expected_media_hash != actual_media_hash:
        raise VideoClosureError("source_media_hash_mismatch", str(media_path))

    media_meta = manifest.get("media") if isinstance(manifest.get("media"), dict) else {}
    duration = _number(media_meta.get("duration_seconds"), "media_duration_invalid")
    if manifest.get("fixture_only") is True:
        if str(manifest.get("synthetic_media_contract") or "") != SYNTHETIC_MEDIA_MAGIC.decode().strip():
            raise VideoClosureError("synthetic_media_contract_invalid", str(media_path))
        if not media_path.read_bytes().startswith(SYNTHETIC_MEDIA_MAGIC):
            raise VideoClosureError("corrupt_media", "synthetic fixture magic does not match")
    elif duration <= 0:
        duration = _probe_real_media_duration(media_path)
    if duration <= 0:
        raise VideoClosureError("media_duration_invalid", str(duration))
    _validate_timestamps(timeline, duration, label="timeline")
    _validate_timestamps(segments, duration, label="transcript")

    refs: dict[str, Any] = {
        "manifest": _artifact_ref(manifest_path),
        "timeline": _artifact_ref(timeline_path),
        "transcript": _artifact_ref(transcript_path),
        "source_media": _artifact_ref(media_path),
    }
    optional_keys = {
        "vad": manifest.get("silero_vad_candidate"),
        "chapters": manifest.get("smart_summary_chapters"),
        "run_registry": manifest.get("run_artifact_registry_json"),
        "workbench": manifest.get("video_workbench_json"),
    }
    for key, value in optional_keys.items():
        if not value:
            continue
        path = _bundle_artifact(bundle, value, "", key)
        refs[key] = _artifact_ref(path)
    identity = canonical_json_sha256({key: {"sha256": value["sha256"], "bytes": value["bytes"]} for key, value in refs.items()})
    return {
        "bundle": bundle,
        "manifest": manifest,
        "timeline": timeline,
        "transcript": transcript,
        "segments": segments,
        "media_path": media_path,
        "duration_seconds": duration,
        "input_snapshot": refs,
        "input_identity_sha256": identity,
    }


def _build_evidence_pack(manifest: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    timeline = context["timeline"]
    evidence_rows = []
    for position, row in enumerate(timeline, start=1):
        confidence = _confidence(row.get("confidence"))
        evidence_rows.append(
            {
                "evidence_id": f"timeline:{int(row.get('index') or position)}",
                "timeline_index": int(row.get("index") or position),
                "start": round(float(row["start"]), 3),
                "end": round(float(row["end"]), 3),
                "transcript": str(row.get("transcript") or "").strip(),
                "visual_text": str(row.get("visual_text") or row.get("ocr_text") or "").strip(),
                "confidence": confidence,
                "evidence_level": "review_required" if confidence < LOW_CONFIDENCE_THRESHOLD or row.get("needs_human_review") else "candidate",
                "default_action": "keep",
                "human_review_required": bool(confidence < LOW_CONFIDENCE_THRESHOLD or row.get("needs_human_review")),
            }
        )
    return {
        "schema": EVIDENCE_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "closure_run_id": manifest["closure_run_id"],
        "created_at": now_iso(),
        "source_bundle": str(context["bundle"]),
        "input_identity_sha256": context["input_identity_sha256"],
        "source_media": context["input_snapshot"]["source_media"],
        "time_base": {"unit": "seconds", "origin": "media_start", "precision_ms": 1, "offset_applied_twice": False},
        "duration_seconds": context["duration_seconds"],
        "transcript_segments": [
            {
                "segment_id": str(row.get("segment_id") or f"segment-{index:04d}"),
                "start": round(float(row["start"]), 3),
                "end": round(float(row["end"]), 3),
                "text": str(row.get("text") or "").strip(),
                "confidence": _confidence(row.get("confidence")),
                "default_action": "keep",
            }
            for index, row in enumerate(context["segments"], start=1)
        ],
        "timeline_evidence": evidence_rows,
        "low_confidence_policy": "keep_and_require_human_review",
        "source_media_modified": False,
        "provider_called": False,
    }


def _build_segment_plan(manifest: dict[str, Any], context: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    try:
        fast = build_long_video_fast_segment_plan(context["bundle"], write=False)
        memory = build_long_video_memory_pack(context["bundle"], target_window_seconds=300.0, write=False)
    except Exception as exc:  # noqa: BLE001 - stable wrapper around reused modules.
        raise VideoClosureError("segment_plan_failed", str(exc)) from exc
    windows = [
        {
            "window_id": str(row.get("memory_id") or f"window-{index:04d}"),
            "start": round(float(row.get("start") or 0.0), 3),
            "end": round(float(row.get("end") or 0.0), 3),
            "timeline_indexes": [int(value) for value in row.get("timeline_indexes") or []],
            "checkpoint_status": "planned",
        }
        for index, row in enumerate(memory.get("short_memories") or [], start=1)
        if isinstance(row, dict)
    ]
    if not windows:
        raise VideoClosureError("missing_window_plan", "long_video_memory_pack returned no bounded windows")
    low = [
        {
            "evidence_id": row["evidence_id"],
            "start": row["start"],
            "end": row["end"],
            "confidence": row["confidence"],
            "default_action": "keep",
            "reason": "low_confidence_requires_human_review",
        }
        for row in evidence["timeline_evidence"]
        if float(row.get("confidence") or 0.0) < LOW_CONFIDENCE_THRESHOLD or row.get("human_review_required")
    ]
    chapters = _chapter_boundaries(context, windows)
    candidates = []
    for index, row in enumerate(fast.get("segments") or [], start=1):
        if not isinstance(row, dict):
            continue
        candidates.append(
            {
                "candidate_id": str(row.get("segment_id") or row.get("id") or f"segment-candidate-{index:04d}"),
                "start": round(float(row.get("start") or 0.0), 3),
                "end": round(float(row.get("end") or 0.0), 3),
                "classification": str(row.get("classification") or row.get("kind") or "review_required"),
                "default_action": "keep",
                "human_confirmation_required": True,
                "evidence_ids": [str(value) for value in row.get("evidence_ids") or []],
            }
        )
    return {
        "schema": SEGMENT_PLAN_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "closure_run_id": manifest["closure_run_id"],
        "created_at": now_iso(),
        "input_evidence_sha256": _stage_sha(Path(manifest["run_root"]) / "artifacts" / STAGE_FILENAMES["evidence"]),
        "time_base": evidence["time_base"],
        "duration_seconds": evidence["duration_seconds"],
        "window_plan": windows,
        "segment_candidates": candidates,
        "low_confidence_items": low,
        "human_chapter_boundaries": chapters,
        "plan_precedes_editing": True,
        "editing_authorized": False,
        "original_media_overwrite_allowed": False,
        "source_media_modified": False,
        "automatic_delete_allowed": False,
        "reused_modules": [
            "video_knowledge_pipeline.long_video_memory_pack",
            "video_knowledge_pipeline.long_video_fast_segment",
        ],
        "underlying_contracts": {
            "memory": str(memory.get("schema") or ""),
            "segment": str(fast.get("schema") or ""),
        },
    }


def _build_content_brief(manifest: dict[str, Any], evidence: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    digest = canonical_json_sha256({"evidence": evidence.get("input_identity_sha256"), "plan": plan.get("window_plan")})
    chapters = []
    scenes = []
    for index, window in enumerate(plan.get("window_plan") or [], start=1):
        start = float(window.get("start") or 0.0)
        end = float(window.get("end") or 0.0)
        rows = [row for row in evidence.get("timeline_evidence") or [] if float(row.get("end") or 0.0) > start and float(row.get("start") or 0.0) < end]
        refs = [{"ref_type": "evidence", "ref_id": str(row["evidence_id"])} for row in rows]
        if not refs:
            refs = [{"ref_type": "unknown", "ref_id": f"window:{index:04d}:missing-evidence"}]
        narration = " ".join(str(row.get("transcript") or row.get("visual_text") or "").strip() for row in rows).strip()
        if not narration:
            narration = "该时间窗证据不足，必须保留为待复核候选。"
        chapter_id = f"chapter-{index:02d}"
        chapters.append(
            {
                "chapter_id": chapter_id,
                "title": f"候选章节 {index}",
                "purpose": "按时间窗组织可追溯的视频证据，章节边界待人工确认",
                "target_duration_seconds": max(5, int(round(end - start))),
                "narration_draft": narration[:1200],
                "evidence_refs": refs,
            }
        )
        scenes.append(
            {
                "scene_id": f"scene-{index:02d}",
                "chapter_ref": chapter_id,
                "purpose": "把原视频证据投影为候选剪辑场景，不承诺素材权利或发布",
                "narration_draft": narration[:1200],
                "evidence_refs": refs,
                "visual_leads": [
                    {
                        "lead_id": f"visual-lead-{index:02d}",
                        "description": "沿用该时间窗的 VKP 画面证据；具体入点、出点和画面须人工复核",
                        "status": "candidate",
                        "upstream_status": "candidate",
                        "evidence_role": "explanatory",
                        "evidence_refs": refs,
                        "rights_note": "只读引用原 Bundle；未授予复制、上传或发布权。",
                        "boundary_note": "计划先于剪辑，低置信内容默认保留。",
                    }
                ],
                "subtitle_candidates": [narration[:160]],
            }
        )
    unknowns = [
        {
            "unknown_id": f"low-confidence-{index:04d}",
            "text": f"{row['evidence_id']} 置信度不足，默认保留并等待人工决定。",
            "handling": "requires-review",
        }
        for index, row in enumerate(plan.get("low_confidence_items") or [], start=1)
    ]
    evidence_path = Path(manifest["run_root"]) / "artifacts" / STAGE_FILENAMES["evidence"]
    plan_path = Path(manifest["run_root"]) / "artifacts" / STAGE_FILENAMES["segment_plan"]
    target_duration = max(15, min(7200, int(round(float(evidence.get("duration_seconds") or 0.0)))))
    thesis = str((evidence.get("timeline_evidence") or [{}])[0].get("transcript") or "基于 VKP 证据形成候选视频结构，所有边界等待人工复核。")
    return {
        "schema_version": CONTENT_BRIEF_SCHEMA,
        "brief_id": f"vbrief-{digest[:20]}",
        "content_id": f"vkp-closure-{digest[:20]}",
        "mode": "draft_only",
        "title": str(evidence.get("source_bundle") or "VKP 视频候选简报")[-160:],
        "thesis": thesis[:800],
        "audience": "human-video-reviewer",
        "target_duration_seconds": target_duration,
        "chapters": chapters,
        "scenes": scenes,
        "forbidden_expressions": [
            "不得把候选剪辑表述为已经剪完或已经发布。",
            "不得自动删除低置信内容或覆盖原片。",
            "不得把未复核章节边界表述为人工确认事实。",
        ],
        "unknowns": unknowns,
        "review_gate": {
            "status": "required",
            "reasons": ["章节边界、低置信内容、剪辑点位和发布权均需人工复核。"],
            "required_reviews": ["evidence", "chapter-boundary", "edit-selection", "privacy", "rights"],
            "publication_boundary": "manual-only",
        },
        "provenance": {
            "research_id": manifest["closure_run_id"],
            "input_schema_id": EVIDENCE_SCHEMA,
            "input_sha256": str(evidence.get("input_identity_sha256") or digest),
            "adapter": "research-output-to-content",
            "adapter_version": "vkp-video-closure-1.0.0",
            "generated_at": now_iso(),
            "manifest_ref": str(Path(manifest["run_root"]) / "closure-run.json"),
            "offline_only": True,
            "external_api_called": False,
            "model_memory_used_as_evidence": False,
            "upstream_input_files": [
                {"path": str(evidence_path), "sha256": sha256_file(evidence_path), "status": "present"},
                {"path": str(plan_path), "sha256": sha256_file(plan_path), "status": "present"},
            ],
        },
    }


def _validate_content_brief(payload: dict[str, Any], downstream: dict[str, Any]) -> None:
    required = {"schema_version", "brief_id", "content_id", "mode", "title", "thesis", "audience", "target_duration_seconds", "chapters", "scenes", "review_gate", "provenance"}
    if not required.issubset(payload):
        raise VideoClosureError("content_brief_invalid", "required fields are missing")
    if downstream.get("status") != "ready":
        return
    schema_path = Path(downstream["root"]) / MVP_CONTRACT_FILES[-1]
    try:
        schema = _read_object(schema_path, "content-video-brief schema", "downstream_contract_unreadable")
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(payload)
    except (SchemaError, ValidationError) as exc:
        raise VideoClosureError("content_brief_contract_rejected", str(exc)) from exc


def _build_candidate_handoff(
    manifest: dict[str, Any],
    brief_path: Path,
    plan_path: Path,
    downstream: dict[str, Any],
    edit_job_path: Path | None,
) -> dict[str, Any]:
    status = "ready_for_human_selection" if edit_job_path else "downstream_unavailable_review_required"
    return {
        "schema": HANDOFF_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "closure_run_id": manifest["closure_run_id"],
        "created_at": now_iso(),
        "status": status,
        "candidate_only": True,
        "review_required": True,
        "execute": False,
        "plan_precedes_editing": True,
        "source_bundle_read_only": True,
        "original_media_overwrite_allowed": False,
        "content_video_brief": _artifact_ref(brief_path),
        "segment_plan": _artifact_ref(plan_path),
        "mvp_contract": MVP_EDIT_JOB_SCHEMA,
        "mvp_contract_snapshot": _compact_downstream_snapshot(downstream),
        "mvp_edit_job": _artifact_ref(edit_job_path) if edit_job_path else None,
        "recovery": {
            "command": _resume_command(Path(manifest["run_root"])),
            "reason": "provide a clean compatible MVP root and resume" if not edit_job_path else "human selection required",
        },
        "operator_boundary": {
            "ffmpeg_executed": False,
            "asr_or_ocr_executed": False,
            "provider_called": False,
            "media_uploaded": False,
            "media_published": False,
            "source_media_modified": False,
        },
    }


def _build_review_packet(run_root: Path, manifest: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    handoff = _read_stage_artifact(run_root, "candidate_handoff")
    plan = _read_stage_artifact(run_root, "segment_plan")
    try:
        existing_review = build_video_edit_review_pack(context["bundle"], write=False)
    except Exception as exc:  # noqa: BLE001 - retain a review gap instead of inventing readiness.
        existing_review = {"schema": "video_knowledge_pipeline.video_edit_review_pack.v1", "blockers": [{"reason": "existing_review_pack_unavailable", "detail": str(exc)}], "storyboard_scene_count": 0}
    stage_refs = {
        stage: _artifact_ref(run_root / "artifacts" / STAGE_FILENAMES[stage])
        for stage in STAGES[:-1]
    }
    return {
        "schema": REVIEW_PACKET_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "closure_run_id": manifest["closure_run_id"],
        "created_at": now_iso(),
        "status": "review_required",
        "mode": "candidate_only",
        "accepted": False,
        "published": False,
        "source_media_modified": False,
        "stage_artifacts": stage_refs,
        "candidate_handoff_status": handoff.get("status"),
        "review_items": {
            "human_chapter_boundaries": plan.get("human_chapter_boundaries") or [],
            "low_confidence_items": plan.get("low_confidence_items") or [],
            "candidate_edit_selection": "required",
            "privacy_and_rights": "required",
        },
        "existing_video_edit_review": {
            "schema": existing_review.get("schema"),
            "storyboard_scene_count": int(existing_review.get("storyboard_scene_count") or 0),
            "blockers": existing_review.get("blockers") or [],
            "ready_for_single_ffmpeg": bool(existing_review.get("ready_for_single_ffmpeg")),
        },
        "finalization_gate": {
            "candidate_artifacts_complete": True,
            "real_render_authorized": False,
            "accepted": False,
            "publication_authorized": False,
        },
        "resume_command": _resume_command(run_root),
        "auto_retry_count": AUTO_RETRY_COUNT,
    }


def _invoke_mvp_bridge(mvp_root: Path, bundle: Path, output_path: Path) -> dict[str, Any]:
    env = os.environ.copy()
    source_root = str((mvp_root / "src").resolve())
    env["PYTHONPATH"] = source_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "video_creation_pipeline.cli", str(bundle), "--output", str(output_path)],
        cwd=mvp_root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "MVP bridge failed")[-2000:]
        raise VideoClosureError("mvp_candidate_handoff_failed", detail)
    return _read_object(output_path, "MVP edit job", "mvp_candidate_handoff_invalid")


def _validate_mvp_edit_job(job: dict[str, Any], bundle: Path) -> None:
    if job.get("schema") != MVP_EDIT_JOB_SCHEMA:
        raise VideoClosureError("mvp_candidate_handoff_invalid", "unexpected schema")
    source = job.get("source_bundle") if isinstance(job.get("source_bundle"), dict) else {}
    if Path(str(source.get("path") or "")).resolve() != bundle.resolve():
        raise VideoClosureError("mvp_candidate_handoff_invalid", "source bundle identity mismatch")
    for row in job.get("candidate_clips") or []:
        if not isinstance(row, dict) or row.get("selection_status") != "candidate_only" or row.get("human_confirmed") is True:
            raise VideoClosureError("mvp_candidate_handoff_invalid", "candidate authority was widened")
    operator = job.get("operator_boundary") if isinstance(job.get("operator_boundary"), dict) else {}
    remote = job.get("remote_capabilities") if isinstance(job.get("remote_capabilities"), dict) else {}
    upload = remote.get("upload") if isinstance(remote.get("upload"), dict) else {}
    if any(bool(value) for value in (operator.get("automatic_publish"), operator.get("bridge_executes_ffmpeg"), operator.get("bridge_executes_asr"), remote.get("execute"), remote.get("automatic_publish"), upload.get("allowed"), upload.get("automatic"))):
        raise VideoClosureError("mvp_candidate_handoff_invalid", "execution or upload authority is forbidden")


def _mvp_contract_snapshot(root: Path | None) -> dict[str, Any]:
    if root is None or not root.is_dir():
        return {"status": "unavailable", "root": str(root or ""), "files": [], "git_head": "", "identity_sha256": ""}
    files = []
    for relative in MVP_CONTRACT_FILES:
        path = root / relative
        if not path.is_file():
            return {"status": "unavailable", "root": str(root), "missing": relative, "files": files, "git_head": "", "identity_sha256": ""}
        files.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    git_head = ""
    git_clean: bool | None = None
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False)
        if head.returncode == 0:
            git_head = head.stdout.strip()
            worktree_diff = subprocess.run(
                ["git", "-C", str(root), "diff", "--quiet", "--", *MVP_CONTRACT_FILES],
                timeout=10,
                check=False,
            )
            index_diff = subprocess.run(
                ["git", "-C", str(root), "diff", "--cached", "--quiet", "--", *MVP_CONTRACT_FILES],
                timeout=10,
                check=False,
            )
            git_clean = worktree_diff.returncode == 0 and index_diff.returncode == 0
    except (OSError, subprocess.SubprocessError):
        git_clean = None
    identity = canonical_json_sha256({"git_head": git_head, "files": files})
    return {
        "status": "dirty" if git_clean is False else "ready",
        "root": str(root),
        "git_head": git_head,
        "git_clean": git_clean,
        "files": files,
        "identity_sha256": identity,
        "contract": MVP_EDIT_JOB_SCHEMA,
    }


def _validated_downstream_for_resume(manifest: dict[str, Any], mvp_root: str | Path | None) -> dict[str, Any]:
    stored = manifest.get("downstream_contract_snapshot") if isinstance(manifest.get("downstream_contract_snapshot"), dict) else {"status": "unavailable"}
    requested = _resolve_mvp_root(mvp_root or stored.get("root"))
    current = _mvp_contract_snapshot(requested)
    if stored.get("status") == "ready":
        if current.get("status") != "ready" or current.get("identity_sha256") != stored.get("identity_sha256"):
            raise VideoClosureError("downstream_contract_drift", "MVP bridge/schema identity changed after closure start")
        return current
    if stored.get("status") == "dirty":
        raise VideoClosureError("downstream_contract_dirty", "stored MVP contract was dirty")
    if current.get("status") == "dirty":
        raise VideoClosureError("downstream_contract_dirty", "MVP bridge/schema has uncommitted changes")
    if current.get("status") == "ready":
        manifest["downstream_contract_snapshot"] = current
        write_json(Path(manifest["run_root"]) / "closure-run.json", manifest)
    return current


def _resolve_mvp_root(value: str | Path | None) -> Path | None:
    if value and str(value).strip():
        return Path(value).expanduser().resolve()
    configured = str(os.environ.get("MVP_VIDEO_PIPELINE_ROOT") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    registry_path = workspace_root() / "tool-registry.json"
    if registry_path.is_file():
        try:
            registry = read_json(registry_path)
            tools = registry.get("tools") if isinstance(registry, dict) else None
            record = tools.get("mvp-video-pipeline") if isinstance(tools, dict) else None
            if isinstance(record, dict) and str(record.get("path") or "").strip():
                return Path(str(record["path"])).expanduser().resolve()
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return None


def _validate_all_receipts(run_root: Path, manifest: dict[str, Any]) -> None:
    completed = [str(value) for value in manifest.get("completed_stages") or []]
    if completed != list(STAGES[: len(completed)]):
        raise VideoClosureError("stage_order_invalid", ",".join(completed))
    for stage in completed:
        receipt = _read_object(_receipt_path(run_root, stage), f"{stage} receipt", "stage_receipt_missing_or_corrupt")
        if receipt.get("schema") != STAGE_RECEIPT_SCHEMA or receipt.get("closure_run_id") != manifest.get("closure_run_id") or receipt.get("stage") != stage:
            raise VideoClosureError("stage_receipt_identity_mismatch", stage)
        for artifact in receipt.get("artifacts") or []:
            path = Path(str(artifact.get("path") or "")).resolve()
            if not path.is_file() or sha256_file(path) != artifact.get("sha256") or path.stat().st_size != artifact.get("bytes"):
                raise VideoClosureError("stage_artifact_drift", str(path))


def _stage_receipt(run_root: Path, manifest: dict[str, Any], stage: str, artifacts: list[Path]) -> dict[str, Any]:
    previous = [
        _artifact_ref(run_root / "artifacts" / STAGE_FILENAMES[value])
        for value in STAGES[: STAGES.index(stage)]
    ]
    return {
        "schema": STAGE_RECEIPT_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "closure_run_id": manifest["closure_run_id"],
        "stage": stage,
        "status": "succeeded",
        "completed_at": now_iso(),
        "input_sha256": canonical_json_sha256({"bundle": manifest["input_identity_sha256"], "previous": previous}),
        "artifacts": [_artifact_ref(path) for path in artifacts],
        "auto_retry_count": AUTO_RETRY_COUNT,
        "failure_point": None,
        "recovery": {"resume_command": _resume_command(run_root)},
        "operator_boundary": {"candidate_only": True, "source_media_modified": False, "provider_called": False, "published": False},
    }


def _record_failure(run_root: Path, run_id: str, stage: str, error: VideoClosureError) -> Path:
    path = run_root / "failure-receipts" / f"failure-{time.time_ns()}.json"
    write_json(
        path,
        {
            "schema": FAILURE_RECEIPT_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "closure_run_id": run_id,
            "status": "blocked",
            "failed_at": now_iso(),
            "failure_point": stage,
            "error_code": error.error_code,
            "detail": error.detail,
            "auto_retry_count": AUTO_RETRY_COUNT,
            "automatic_retry_allowed": False,
            "recovery": {"resume_command": _resume_command(run_root), "explicit_resume_required": True},
            "source_media_modified": False,
        },
    )
    return path


def _safe_record_failure(run_root: Path, run_id: str, stage: str, error: VideoClosureError) -> Path | None:
    try:
        return _record_failure(run_root, run_id, stage, error)
    except OSError:
        return None


def _mark_blocked_if_possible(manifest_path: Path, run_root: Path, error: VideoClosureError) -> None:
    if not manifest_path.is_file():
        return
    try:
        manifest = _read_object(manifest_path, "closure manifest", "corrupt_closure_manifest")
        manifest["status"] = "blocked"
        manifest["last_error"] = {"error_code": error.error_code, "detail": error.detail, "failed_at": now_iso()}
        manifest["recovery"] = {"resume_command": _resume_command(run_root)}
        write_json(manifest_path, manifest)
    except Exception:  # noqa: BLE001 - preserve the original failure.
        return


def _public_status(manifest: dict[str, Any], run_root: Path) -> dict[str, Any]:
    return {
        "schema": CLOSURE_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "closure_run_id": manifest.get("closure_run_id"),
        "status": manifest.get("status"),
        "mode": "candidate_only",
        "bundle_dir": manifest.get("bundle_dir"),
        "run_root": str(run_root),
        "completed_stages": list(manifest.get("completed_stages") or []),
        "next_stage": manifest.get("next_stage"),
        "auto_retry_count": AUTO_RETRY_COUNT,
        "accepted": False,
        "published": False,
        "source_media_modified": False,
        "review_packet": str(run_root / "artifacts" / STAGE_FILENAMES["review_packet"]) if (run_root / "artifacts" / STAGE_FILENAMES["review_packet"]).is_file() else "",
        "resume_command": _resume_command(run_root),
    }


def _chapter_boundaries(context: dict[str, Any], windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chapters_ref = context["input_snapshot"].get("chapters")
    rows: list[dict[str, Any]] = []
    if chapters_ref:
        chapters = _read_object(Path(chapters_ref["path"]), "chapters", "corrupt_chapters")
        rows = [row for row in chapters.get("chapters") or [] if isinstance(row, dict)]
    if not rows:
        rows = [{"index": index, "start": row["start"], "end": row["end"]} for index, row in enumerate(windows, start=1)]
    return [
        {
            "boundary_id": f"chapter-boundary-{index:03d}",
            "start": round(float(row.get("start") or 0.0), 3),
            "end": round(float(row.get("end") or 0.0), 3),
            "title": str(row.get("title") or f"候选章节 {index}"),
            "status": "review_required",
            "human_confirmed": False,
        }
        for index, row in enumerate(rows, start=1)
    ]


def _probe_real_media_duration(path: Path) -> float:
    try:
        from .media_equivalence_audit import probe_media

        result = probe_media(path)
    except Exception as exc:  # noqa: BLE001 - exact ffprobe failures vary by host.
        raise VideoClosureError("corrupt_media", str(exc)) from exc
    duration = _number(result.get("duration_seconds"), "media_duration_invalid")
    if duration <= 0:
        raise VideoClosureError("corrupt_media", "ffprobe returned no positive duration")
    return duration


def _validate_timestamps(rows: list[Any], duration: float, *, label: str) -> None:
    previous_start = -1.0
    seen: set[int] = set()
    for position, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            raise VideoClosureError("timestamp_invalid", f"{label} row {position} is not an object")
        start = _number(raw.get("start"), "timestamp_invalid")
        end = _number(raw.get("end"), "timestamp_invalid")
        if start < -TIMESTAMP_TOLERANCE_SECONDS or end <= start or end > duration + TIMESTAMP_TOLERANCE_SECONDS or start + TIMESTAMP_TOLERANCE_SECONDS < previous_start:
            raise VideoClosureError("timestamp_outside_media", f"{label} row {position}: {start}..{end} / {duration}")
        previous_start = start
        if label == "timeline":
            index = int(raw.get("index") or position)
            if index in seen:
                raise VideoClosureError("timeline_index_duplicated", str(index))
            seen.add(index)


def _bundle_artifact(bundle: Path, value: Any, fallback: str, label: str) -> Path:
    raw = str(value or fallback).strip()
    if not raw:
        raise VideoClosureError("missing_evidence", f"{label} path is missing")
    path = Path(raw).expanduser()
    path = (path if path.is_absolute() else bundle / path).resolve()
    if not path.is_relative_to(bundle):
        raise VideoClosureError("bundle_path_escape", raw)
    if not path.is_file():
        raise VideoClosureError("missing_evidence", f"{label} not found: {path}")
    return path


def _artifact_ref(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise VideoClosureError("artifact_missing", str(resolved))
    return {"path": str(resolved), "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def _read_stage_artifact(run_root: Path, stage: str) -> dict[str, Any]:
    return _read_object(run_root / "artifacts" / STAGE_FILENAMES[stage], f"{stage} artifact", "stage_artifact_missing_or_corrupt")


def _read_object(path: Path, label: str, code: str) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise VideoClosureError(code, f"{label}: {exc}") from exc
    if not isinstance(value, dict):
        raise VideoClosureError(code, f"{label} must be a JSON object")
    return value


def _read_array(path: Path, label: str, code: str) -> list[Any]:
    try:
        value = read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise VideoClosureError(code, f"{label}: {exc}") from exc
    if not isinstance(value, list):
        raise VideoClosureError(code, f"{label} must be a JSON array")
    return value


def _validated_run_id(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{7,79}", normalized):
        raise VideoClosureError("closure_run_id_invalid", normalized)
    return normalized


def _default_run_id(bundle: Path) -> str:
    refs = {"bundle": str(bundle.resolve())}
    for name in ("manifest.json", "timeline.json"):
        path = bundle / name
        if path.is_file():
            refs[name] = sha256_file(path)
    manifest_path = bundle / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = read_json(manifest_path)
            transcript_value = manifest.get("normalized_transcript") if isinstance(manifest, dict) else None
            transcript = Path(str(transcript_value or "normalized-transcript.json"))
            transcript = (transcript if transcript.is_absolute() else bundle / transcript).resolve()
            if transcript.is_relative_to(bundle.resolve()) and transcript.is_file():
                refs["normalized_transcript"] = sha256_file(transcript)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return f"video-closure-{canonical_json_sha256(refs)[:16]}"


def _run_root(bundle: Path, output_dir: str | Path | None, run_id: str) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return (bundle.parent / ".vkp-closure-runs" / run_id).resolve()


def _validate_manifest_identity(manifest: dict[str, Any], bundle: Path, run_id: str) -> None:
    if manifest.get("schema") != CLOSURE_SCHEMA or manifest.get("contract_version") != CONTRACT_VERSION:
        raise VideoClosureError("closure_contract_stale", "closure manifest contract does not match current implementation")
    if manifest.get("closure_run_id") != run_id or Path(str(manifest.get("bundle_dir") or "")).resolve() != bundle.resolve():
        raise VideoClosureError("closure_identity_mismatch", str(bundle))


def _validate_current_input_identity(manifest: dict[str, Any]) -> None:
    bundle = Path(str(manifest.get("bundle_dir") or "")).resolve()
    current = _build_input_context(bundle)
    if current["input_identity_sha256"] != manifest.get("input_identity_sha256"):
        raise VideoClosureError("input_drift", "Bundle evidence changed after closure start")


def _failure_stage(manifest_path: Path) -> str:
    if not manifest_path.is_file():
        return "preflight"
    try:
        value = read_json(manifest_path)
        return str(value.get("next_stage") or "finalize") if isinstance(value, dict) else "preflight"
    except Exception:  # noqa: BLE001
        return "preflight"


def _receipt_path(run_root: Path, stage: str) -> Path:
    return run_root / "receipts" / f"{STAGES.index(stage) + 1:02d}-{stage}.json"


def _stage_sha(path: Path) -> str:
    if not path.is_file():
        raise VideoClosureError("stage_artifact_missing_or_corrupt", str(path))
    return sha256_file(path)


def _number(value: Any, code: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VideoClosureError(code, str(value)) from exc
    if number != number or number in {float("inf"), float("-inf")}:
        raise VideoClosureError(code, str(value))
    return number


def _confidence(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, result)), 4)


def _resume_command(run_root: Path) -> str:
    return f".\\scripts\\video-knowledge.ps1 video-closure-resume {_ps_quote(run_root)}"


def _ps_quote(value: Path | str) -> str:
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def _compact_downstream_snapshot(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": value.get("status"),
        "root": value.get("root"),
        "git_head": value.get("git_head"),
        "git_clean": value.get("git_clean"),
        "identity_sha256": value.get("identity_sha256"),
        "contract": value.get("contract", MVP_EDIT_JOB_SCHEMA),
        "missing": value.get("missing", ""),
    }
