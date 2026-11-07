from __future__ import annotations

import difflib
import hashlib
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .file_hash import sha256_file as _sha256_file
from .asr_adapter import read_asr_cues, read_asr_segment_dicts, render_srt
from .asr_consensus import build_asr_consensus
from .asr_diff_adjudication import build_asr_diff_adjudication
from .asr_response_quality import assess_asr_response
from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .storage import bundle_write_lock, read_json, write_json
from .transcript import parse_transcript


SCHEMA = "video_knowledge_pipeline.asr_secondary_evidence_closure.v1"
PREPARED_SUITE_SCHEMA = (
    "video_knowledge_pipeline.model_candidate_fixed_suite_prepared.v1"
)
CONNECTOR_SCHEMA = "video_knowledge_pipeline.trusted_model_connector.v1"
UNTIMED_ALIGNMENT_MIN_SIMILARITY = 0.55


def close_secondary_asr_evidence(
    bundle_dir: str | Path,
    *,
    connector_execution: str | Path,
    prepared_suite: str | Path,
    candidate_id: str = "",
    primary_transcript: str | Path | None = None,
    media_path: str | Path | None = None,
    agreement_threshold: float = 0.86,
    write: bool = True,
) -> dict[str, Any]:
    """Turn one exact connector execution into local-only secondary ASR evidence.

    This function never calls a provider and never promotes or edits the canonical
    transcript.  It validates the saved execution against the prepared candidate,
    re-runs the ASR quality gate, keeps usable segments, and reuses VKP consensus
    and anonymous adjudication artifacts.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"bundle manifest not found: {manifest_path}")
    execution_path = _required_json_path(connector_execution, "connector execution")
    suite_path = _required_json_path(prepared_suite, "prepared suite")
    execution = _read_object(execution_path, "connector execution")
    suite = _read_object(suite_path, "prepared suite")
    candidate = _select_candidate(suite, candidate_id)
    safe_candidate_id = _safe_id(str(candidate.get("candidate_id") or "secondary-asr"))
    canonical_path = _canonical_path(root, primary_transcript)
    canonical_before = _sha256_file(canonical_path)
    evidence_dir = root / "asr-secondary-evidence" / safe_candidate_id
    report_path = evidence_dir / "secondary-asr-evidence-closure.json"
    quality_path = evidence_dir / "secondary-asr-quality.json"
    raw_path = evidence_dir / "secondary-raw-asr-output.json"
    accepted_path = evidence_dir / "secondary-accepted-asr-output.json"
    normalized_path = evidence_dir / "secondary-normalized-transcript.json"
    normalized_srt_path = evidence_dir / "secondary-normalized-transcript.srt"
    inferred_path = evidence_dir / "secondary-inferred-timing-transcript.json"

    identity = _validate_identity(execution, suite, candidate)
    base_result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "connector_execution": str(execution_path),
        "connector_execution_sha256": _sha256_file(execution_path),
        "prepared_suite": str(suite_path),
        "prepared_suite_sha256": _sha256_file(suite_path),
        "identity_validation": identity,
        "canonical_integrity": {
            "path": str(canonical_path),
            "before_sha256": canonical_before,
            "after_sha256": canonical_before,
            "unchanged": True,
        },
        "operator_boundary": {
            "local_postprocessing_only": True,
            "no_network_call": True,
            "no_provider_fallback": True,
            "no_location_fallback": True,
            "secondary_never_auto_promoted": True,
            "does_not_apply_patches": True,
            "human_confirmation_required_for_changes": True,
            "evaluation_reference_used": False,
            "inferred_timing_never_treated_as_provider_timestamp": True,
        },
        "updated_at": now_iso(),
    }
    if not identity["ok"]:
        result = {
            **base_result,
            "ok": False,
            "status": "blocked",
            "reason": "connector_execution_identity_mismatch",
            "quality": {},
            "artifacts": {},
        }
        return _finish(
            root,
            result,
            report_path=report_path,
            canonical_path=canonical_path,
            write=write,
        )

    raw_payload = _extract_asr_payload(execution)
    task_instructions = str(candidate.get("instructions") or "")
    asr_prompt = str(candidate.get("asr_prompt") or "")
    quality = assess_asr_response(
        raw_payload,
        task_instructions=task_instructions,
        asr_prompt=asr_prompt,
    )
    accepted_segments = _accepted_segments(raw_payload, quality)
    timing_inference = (
        _infer_untimed_secondary_segments(raw_payload, canonical_path)
        if not accepted_segments
        else {"status": "not_needed", "segments": []}
    )
    quality_summary = {
        "status": quality.get("status"),
        "quality_gate_passed": bool(quality.get("quality_gate_passed")),
        "segment_count": int(quality.get("segment_count") or 0),
        "accepted_segment_count": len(accepted_segments),
        "review_segment_count": int(quality.get("review_segment_count") or 0),
        "failed_segment_count": int(quality.get("failed_segment_count") or 0),
        "successful_segments_preserved": True,
        "task_instructions_sha256": _text_sha256(task_instructions),
        "asr_prompt_sha256": _text_sha256(asr_prompt),
        "plaintext_prompts_persisted_in_closure_report": False,
        "timing_inference": {
            "status": timing_inference.get("status"),
            "method": timing_inference.get("method", ""),
            "overall_similarity": timing_inference.get("overall_similarity"),
            "segment_count": len(timing_inference.get("segments") or []),
            "timing_inferred": bool(timing_inference.get("segments")),
            "provider_timestamps_present": bool(
                raw_payload.get("segments")
            ),
        },
    }
    artifacts: dict[str, str] = {}
    if write:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        with bundle_write_lock(
            root, operation="asr_secondary_evidence_inputs", timeout_seconds=1.0
        ):
            write_json(raw_path, raw_payload)
            write_json(quality_path, quality)
        artifacts.update(
            {
                "raw_asr_output": str(raw_path),
                "quality_report": str(quality_path),
            }
        )

    if not accepted_segments and timing_inference.get("segments"):
        inferred_payload = {
            "schema": "video_knowledge_pipeline.inferred_timing_secondary_transcript.v1",
            "id": f"secondary-asr-inferred-{safe_candidate_id}",
            "title": f"Inferred-timing secondary ASR evidence: {safe_candidate_id}",
            "provider": str(candidate.get("provider") or "auto"),
            "source_execution_sha256": _sha256_file(execution_path),
            "source_text_sha256": timing_inference.get("source_text_sha256"),
            "timing_source": timing_inference.get("method"),
            "overall_similarity": timing_inference.get("overall_similarity"),
            "segments": timing_inference["segments"],
            "operator_boundary": {
                "candidate_evidence_only": True,
                "timing_inferred_from_primary_boundaries": True,
                "provider_timestamps_present": False,
                "automatic_promotion_forbidden": True,
                "human_confirmation_required": True,
            },
        }
        if write:
            with bundle_write_lock(
                root,
                operation="asr_secondary_inferred_timing",
                timeout_seconds=1.0,
            ):
                write_json(inferred_path, inferred_payload)
            consensus = build_asr_consensus(
                evidence_dir,
                primary_transcript=canonical_path,
                secondary_transcript=inferred_path,
                media_path=media_path,
                agreement_threshold=agreement_threshold,
                execute_clips=False,
                write=True,
            )
            adjudication = build_asr_diff_adjudication(evidence_dir, write=True)
            artifacts.update(
                {
                    "inferred_timing_transcript": str(inferred_path),
                    "inferred_consensus_json": str(evidence_dir / "asr-consensus.json"),
                    "inferred_consensus_markdown": str(evidence_dir / "asr-consensus.md"),
                    "inferred_adjudication_pack": str(
                        evidence_dir / "asr-consensus-adjudication-pack.json"
                    ),
                    "inferred_adjudication_todo": str(
                        evidence_dir / "asr-consensus-adjudication.todo.json"
                    ),
                }
            )
        else:
            consensus = {
                "status": "preview_not_written",
                "conflict_count": None,
                "primary_segment_count": len(parse_transcript(canonical_path)),
                "secondary_segment_count": len(timing_inference["segments"]),
            }
            adjudication = {
                "status": "preview_not_written",
                "difference_count": None,
                "cluster_count": None,
            }
        result = {
            **base_result,
            "ok": False,
            "status": "degraded" if write else "preview_degraded",
            "reason": "untimed_secondary_text_preserved_as_inferred_candidates",
            "quality": quality_summary,
            "consensus": {
                "status": consensus.get("status"),
                "conflict_count": consensus.get("conflict_count"),
                "primary_segment_count": consensus.get("primary_segment_count"),
                "secondary_segment_count": consensus.get("secondary_segment_count"),
            },
            "adjudication": {
                "status": adjudication.get("status"),
                "difference_count": adjudication.get("difference_count"),
                "cluster_count": adjudication.get("cluster_count"),
                "patches_applied": 0,
            },
            "artifacts": artifacts,
        }
        return _finish(
            root,
            result,
            report_path=report_path,
            canonical_path=canonical_path,
            write=write,
        )

    if not accepted_segments:
        result = {
            **base_result,
            "ok": False,
            "status": "failed",
            "reason": "no_quality_accepted_secondary_segments",
            "quality": quality_summary,
            "artifacts": artifacts,
        }
        return _finish(
            root,
            result,
            report_path=report_path,
            canonical_path=canonical_path,
            write=write,
        )

    accepted_payload = dict(raw_payload)
    accepted_payload["segments"] = accepted_segments
    if write:
        with bundle_write_lock(
            root, operation="asr_secondary_evidence_normalize", timeout_seconds=1.0
        ):
            write_json(accepted_path, accepted_payload)
            cues = read_asr_cues(accepted_path, provider="auto")
            normalized = {
                "id": f"secondary-asr-{safe_candidate_id}",
                "title": f"Secondary ASR evidence: {safe_candidate_id}",
                "provider": str(candidate.get("provider") or "auto"),
                "source_path": str(accepted_path),
                "source_execution_sha256": _sha256_file(execution_path),
                "segments": read_asr_segment_dicts(
                    accepted_path, provider="auto", cues=cues
                ),
            }
            write_json(normalized_path, normalized)
            normalized_srt_path.write_text(render_srt(cues), encoding="utf-8")
        artifacts.update(
            {
                "accepted_asr_output": str(accepted_path),
                "normalized_transcript": str(normalized_path),
                "normalized_srt": str(normalized_srt_path),
            }
        )
        consensus = build_asr_consensus(
            root,
            primary_transcript=canonical_path,
            secondary_transcript=normalized_path,
            media_path=media_path,
            agreement_threshold=agreement_threshold,
            execute_clips=False,
            write=True,
        )
        adjudication = build_asr_diff_adjudication(root, write=True)
        artifacts.update(
            {
                "consensus_json": str(root / "asr-consensus.json"),
                "consensus_markdown": str(root / "asr-consensus.md"),
                "adjudication_pack": str(root / "asr-consensus-adjudication-pack.json"),
                "adjudication_todo": str(root / "asr-consensus-adjudication.todo.json"),
            }
        )
    else:
        cues = read_asr_cues_from_payload(accepted_payload)
        consensus = {
            "status": "preview_not_written",
            "conflict_count": None,
            "secondary_segment_count": len(cues),
        }
        adjudication = {
            "status": "preview_not_written",
            "difference_count": None,
            "cluster_count": None,
        }

    degraded = bool(quality.get("failed_segment_count"))
    quality_review = bool(quality.get("review_segment_count"))
    conflict_count = int(consensus.get("conflict_count") or 0)
    needs_review = quality_review or conflict_count > 0
    status = (
        "preview"
        if not write
        else "degraded"
        if degraded
        else "needs_review"
        if needs_review
        else "completed"
    )
    reason = (
        "preview_local_closure"
        if not write
        else "secondary_asr_contains_blocking_segments"
        if degraded
        else "human_adjudication_required"
        if needs_review
        else "secondary_asr_consensus_complete"
    )
    result = {
        **base_result,
        "ok": not degraded,
        "status": status,
        "reason": reason,
        "quality": quality_summary,
        "consensus": {
            "status": consensus.get("status"),
            "conflict_count": consensus.get("conflict_count"),
            "primary_segment_count": consensus.get("primary_segment_count"),
            "secondary_segment_count": consensus.get("secondary_segment_count"),
        },
        "adjudication": {
            "status": adjudication.get("status"),
            "difference_count": adjudication.get("difference_count"),
            "cluster_count": adjudication.get("cluster_count"),
            "patches_applied": 0,
        },
        "artifacts": artifacts,
    }
    return _finish(
        root,
        result,
        report_path=report_path,
        canonical_path=canonical_path,
        write=write,
    )


def read_asr_cues_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("segments") or []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _finish(
    root: Path,
    result: dict[str, Any],
    *,
    report_path: Path,
    canonical_path: Path,
    write: bool,
) -> dict[str, Any]:
    canonical_after = _sha256_file(canonical_path)
    canonical = result["canonical_integrity"]
    canonical["after_sha256"] = canonical_after
    canonical["unchanged"] = canonical_after == canonical["before_sha256"]
    if not canonical["unchanged"]:
        result["ok"] = False
        result["status"] = "failed"
        result["reason"] = "canonical_transcript_changed_during_secondary_evidence"
    result["updated_at"] = now_iso()
    if not write:
        return result

    report_path.parent.mkdir(parents=True, exist_ok=True)
    result.setdefault("artifacts", {})["closure_report"] = str(report_path)
    write_json(report_path, result)
    run = register_bundle_run(
        root,
        run_type="asr_secondary_evidence_closure",
        run_id="asr-secondary-evidence-closure",
        status=str(result.get("status") or "unknown"),
        title="Secondary ASR evidence closure",
        summary=(
            f"Candidate={result.get('candidate_id', '')}; "
            f"status={result.get('status', '')}; canonical unchanged="
            f"{bool(canonical.get('unchanged'))}."
        ),
        inputs={
            "connector_execution": result.get("connector_execution"),
            "prepared_suite": result.get("prepared_suite"),
            "canonical_transcript": canonical.get("path"),
        },
        parameters={
            "candidate_id": result.get("candidate_id"),
            "no_network_call": True,
            "automatic_promotion": False,
        },
        artifacts=list((result.get("artifacts") or {}).values()),
        failed_items=_run_failed_items(result),
        next_actions=_next_actions(result),
        operator_boundary=dict(result.get("operator_boundary") or {}),
        write=True,
    )
    result["run_registry"] = run
    write_json(report_path, result)
    return result


def _run_failed_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identity = result.get("identity_validation") or {}
    for mismatch in identity.get("mismatches") or []:
        rows.append(
            {
                "id": str(mismatch.get("field") or "identity"),
                "reason": "identity_mismatch",
                "detail": str(mismatch.get("detail") or ""),
            }
        )
    quality = result.get("quality") or {}
    failed_count = int(quality.get("failed_segment_count") or 0)
    if failed_count:
        rows.append(
            {
                "id": "secondary_asr_quality",
                "reason": "blocking_segments",
                "detail": f"{failed_count} secondary ASR segments failed quality checks.",
            }
        )
    elif quality and not bool(quality.get("quality_gate_passed")):
        rows.append(
            {
                "id": "secondary_asr_quality",
                "reason": "quality_gate_failed",
                "detail": "Secondary ASR output did not pass the response quality gate.",
            }
        )
    if not (result.get("canonical_integrity") or {}).get("unchanged"):
        rows.append(
            {
                "id": "canonical_integrity",
                "reason": "canonical_changed",
                "detail": "Canonical transcript hash changed during local closure.",
            }
        )
    return rows


def _next_actions(result: dict[str, Any]) -> list[str]:
    status = str(result.get("status") or "")
    timing_inference = (result.get("quality") or {}).get("timing_inference") or {}
    if status in {"degraded", "preview_degraded"} and timing_inference.get(
        "timing_inferred"
    ):
        return [
            "Review the inferred-timing disagreement pack as candidate evidence only.",
            "Require provider timestamps, local forced alignment, or human confirmation before applying any change.",
            "Any remote retry requires a new exact artifact consent; no fallback is allowed.",
        ]
    if status == "needs_review":
        return [
            "Review anonymous ASR disagreement clusters.",
            "Apply only evidence-backed, human-confirmed decisions with the existing adjudication command.",
        ]
    if status == "degraded":
        return [
            "Review the secondary ASR quality report and retain only accepted segments.",
            "Any remote retry requires a new exact artifact consent; no fallback is allowed.",
        ]
    if status == "blocked":
        return ["Use the connector execution produced by the exact prepared candidate."]
    if status == "failed":
        return ["Inspect the closure report before preparing any new exact retry."]
    return []


def _validate_identity(
    execution: dict[str, Any],
    suite: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    mismatches: list[dict[str, str]] = []
    if str(suite.get("schema") or "") != PREPARED_SUITE_SCHEMA:
        _mismatch(mismatches, "prepared_suite.schema", "unexpected schema")
    if str(execution.get("schema") or "") != CONNECTOR_SCHEMA:
        _mismatch(mismatches, "connector_execution.schema", "unexpected schema")
    execution_status = str(execution.get("status") or "")
    accepted_statuses = {"completed", "review_required", "degraded", "failed"}
    if not bool(execution.get("ok")) or execution_status not in accepted_statuses:
        _mismatch(
            mismatches,
            "connector_execution.status",
            "transport execution is not consumable",
        )
    if candidate.get("model_type"):
        _expect_equal(
            mismatches,
            "model_type",
            execution.get("model_type"),
            candidate.get("model_type"),
        )
    for field in ("transport_ok", "contract_ok"):
        if execution.get(field) is not True:
            _mismatch(mismatches, field, "connector execution gate did not pass")
    if str(execution.get("task") or "") != str(candidate.get("connector_task") or ""):
        _mismatch(mismatches, "task", "connector task differs from prepared candidate")

    route = execution.get("route") if isinstance(execution.get("route"), dict) else {}
    _expect_equal(
        mismatches, "route_id", route.get("route_id"), candidate.get("route_id")
    )
    _expect_equal(
        mismatches,
        "route_revision",
        route.get("route_revision"),
        candidate.get("route_revision"),
    )
    if candidate.get("virtual_model"):
        _expect_equal(
            mismatches,
            "virtual_model",
            route.get("virtual_model"),
            candidate.get("virtual_model"),
        )
    deployments = route.get("deployments") or []
    deployment = (
        deployments[0] if deployments and isinstance(deployments[0], dict) else {}
    )
    if len(deployments) != 1:
        _mismatch(
            mismatches, "deployments", "exactly one prepared deployment is required"
        )
    _expect_equal(
        mismatches, "provider", deployment.get("provider"), candidate.get("provider")
    )
    _expect_equal(mismatches, "model", deployment.get("model"), candidate.get("model"))
    actual_destination = _destination(deployment.get("base_url"))
    _expect_equal(
        mismatches,
        "destination",
        actual_destination,
        str(candidate.get("destination") or "").lower(),
    )

    expected_artifacts = _artifact_map(candidate.get("artifacts") or [])

    runtime = _runtime_result(execution)
    if not runtime:
        _mismatch(mismatches, "runtime_result", "runtime result is missing")
    else:
        _expect_equal(
            mismatches,
            "runtime.route_id",
            runtime.get("route_id"),
            candidate.get("route_id"),
        )
        _expect_equal(
            mismatches,
            "runtime.route_revision",
            runtime.get("route_revision"),
            candidate.get("route_revision"),
        )
        if candidate.get("route_task"):
            _expect_equal(
                mismatches,
                "runtime.task",
                runtime.get("task"),
                candidate.get("route_task"),
            )
        if candidate.get("virtual_model"):
            _expect_equal(
                mismatches,
                "runtime.virtual_model",
                runtime.get("virtual_model"),
                candidate.get("virtual_model"),
            )
        _expect_equal(
            mismatches,
            "runtime.provider",
            runtime.get("provider"),
            candidate.get("provider"),
        )
        runtime_deployment = (
            runtime.get("deployment")
            if isinstance(runtime.get("deployment"), dict)
            else {}
        )
        _expect_equal(
            mismatches,
            "runtime.model",
            runtime_deployment.get("model"),
            candidate.get("model"),
        )
    upload_manifest = execution.get("upload_manifest")
    uploaded = upload_manifest.get("files") if isinstance(upload_manifest, dict) else []
    actual_artifacts = _artifact_map(uploaded or [])
    if expected_artifacts != actual_artifacts:
        _mismatch(
            mismatches,
            "upload_manifest",
            "uploaded path/hash/size set differs from prepared candidate",
        )
    return {
        "ok": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "route_id": str(candidate.get("route_id") or ""),
        "route_revision": str(candidate.get("route_revision") or ""),
        "provider": str(candidate.get("provider") or ""),
        "model": str(candidate.get("model") or ""),
        "destination": str(candidate.get("destination") or ""),
        "artifact_count": len(expected_artifacts),
    }


def _select_candidate(suite: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    rows = [row for row in suite.get("candidates") or [] if isinstance(row, dict)]
    if candidate_id:
        matches = [
            row for row in rows if str(row.get("candidate_id") or "") == candidate_id
        ]
        if len(matches) != 1:
            raise ValueError(f"prepared candidate not found: {candidate_id}")
        return matches[0]
    if len(rows) != 1:
        raise ValueError(
            "candidate_id is required when prepared suite has multiple candidates"
        )
    return rows[0]


def _extract_asr_payload(execution: dict[str, Any]) -> dict[str, Any]:
    queue: list[dict[str, Any]] = [execution]
    visited: set[int] = set()
    text_payload: dict[str, Any] = {}
    priority_keys = (
        "raw_output",
        "raw_response",
        "runtime_result",
        "model_result",
        "response",
    )
    while queue:
        value = queue.pop(0)
        identity = id(value)
        if identity in visited:
            continue
        visited.add(identity)
        if not text_payload and str(value.get("text") or "").strip():
            text_payload = dict(value)
        if isinstance(value.get("segments"), list):
            return dict(value)
        for key in priority_keys:
            nested = value.get(key)
            if isinstance(nested, dict):
                queue.append(nested)
    return text_payload


def _infer_untimed_secondary_segments(
    raw_payload: dict[str, Any], canonical_path: Path
) -> dict[str, Any]:
    source_text = str(raw_payload.get("text") or "").strip()
    primary_cues = parse_transcript(canonical_path)
    base = {
        "status": "unavailable",
        "method": "primary_segment_monotonic_text_projection",
        "overall_similarity": 0.0,
        "source_text_sha256": _text_sha256(source_text),
        "segments": [],
    }
    if not source_text or not primary_cues:
        return base

    primary_parts = [_compact_alignment_text(cue.text) for cue in primary_cues]
    primary_text = "".join(primary_parts)
    secondary_text, secondary_offsets = _compact_alignment_with_offsets(source_text)
    if not primary_text or not secondary_text:
        return base

    matcher = difflib.SequenceMatcher(
        a=primary_text,
        b=secondary_text,
        autojunk=False,
    )
    overall_similarity = round(float(matcher.ratio()), 6)
    result = {**base, "overall_similarity": overall_similarity}
    if overall_similarity < UNTIMED_ALIGNMENT_MIN_SIMILARITY:
        result["status"] = "rejected_low_similarity"
        return result

    primary_boundaries = [0]
    for part in primary_parts:
        primary_boundaries.append(primary_boundaries[-1] + len(part))
    projected = [
        _project_alignment_boundary(
            boundary,
            matcher.get_opcodes(),
            secondary_length=len(secondary_text),
        )
        for boundary in primary_boundaries
    ]
    for index in range(1, len(projected)):
        projected[index] = max(projected[index], projected[index - 1])
    projected[0] = 0
    projected[-1] = len(secondary_text)
    raw_boundaries = [
        _raw_alignment_boundary(
            boundary,
            secondary_offsets,
            source_length=len(source_text),
        )
        for boundary in projected
    ]
    raw_boundaries[0] = 0
    raw_boundaries[-1] = len(source_text)

    segments: list[dict[str, Any]] = []
    for index, cue in enumerate(primary_cues):
        text = source_text[raw_boundaries[index] : raw_boundaries[index + 1]].strip()
        if not text:
            continue
        primary_id = str(cue.segment_id or f"primary-{index + 1:06d}")
        text_similarity = round(
            difflib.SequenceMatcher(
                a=primary_parts[index],
                b=_compact_alignment_text(text),
                autojunk=False,
            ).ratio(),
            6,
        )
        segment_id = f"secondary-inferred-{index + 1:06d}"
        segments.append(
            {
                "id": segment_id,
                "segment_id": segment_id,
                "source_segment_ids": ["provider-full-text"],
                "start": float(cue.start),
                "end": float(cue.end),
                "text": text,
                "timing_inferred": True,
                "timing_source": "primary_segment_boundaries",
                "primary_segment_id": primary_id,
                "alignment_similarity": text_similarity,
                "transformations": [
                    {
                        "type": "monotonic_text_projection",
                        "primary_segment_id": primary_id,
                        "provider_timestamps_present": False,
                    }
                ],
            }
        )
    if not segments:
        return result
    result.update(
        {
            "status": "candidate_alignment_available",
            "primary_segment_count": len(primary_cues),
            "segment_count": len(segments),
            "segments": segments,
        }
    )
    return result


def _compact_alignment_text(value: Any) -> str:
    return _compact_alignment_with_offsets(value)[0]


def _compact_alignment_with_offsets(value: Any) -> tuple[str, list[int]]:
    text = str(value or "")
    characters: list[str] = []
    offsets: list[int] = []
    for index, character in enumerate(text):
        lowered = character.lower()
        if re.fullmatch(r"[0-9a-z\u4e00-\u9fff]", lowered):
            characters.append(lowered)
            offsets.append(index)
    return "".join(characters), offsets


def _project_alignment_boundary(
    position: int,
    opcodes: list[tuple[str, int, int, int, int]],
    *,
    secondary_length: int,
) -> int:
    position = max(0, int(position))
    for _tag, primary_start, primary_end, secondary_start, secondary_end in opcodes:
        if position < primary_start:
            return max(0, min(secondary_length, secondary_start))
        if primary_start <= position <= primary_end:
            if primary_end <= primary_start:
                projected = secondary_end
            else:
                fraction = (position - primary_start) / (primary_end - primary_start)
                projected = round(
                    secondary_start + fraction * (secondary_end - secondary_start)
                )
            return max(0, min(secondary_length, int(projected)))
    return max(0, int(secondary_length))


def _raw_alignment_boundary(
    normalized_boundary: int,
    offsets: list[int],
    *,
    source_length: int,
) -> int:
    if normalized_boundary <= 0:
        return 0
    if normalized_boundary >= len(offsets):
        return max(0, int(source_length))
    return max(0, min(int(source_length), int(offsets[normalized_boundary])))


def _runtime_result(execution: dict[str, Any]) -> dict[str, Any]:
    model_result = execution.get("model_result")
    if not isinstance(model_result, dict):
        return {}
    runtime = model_result.get("runtime_result")
    return dict(runtime) if isinstance(runtime, dict) else {}


def _accepted_segments(
    raw_payload: dict[str, Any], quality: dict[str, Any]
) -> list[dict[str, Any]]:
    raw_rows = [
        dict(row) for row in raw_payload.get("segments") or [] if isinstance(row, dict)
    ]
    assessed = [row for row in quality.get("segments") or [] if isinstance(row, dict)]
    return [
        row
        for index, row in enumerate(raw_rows)
        if index < len(assessed) and not bool(assessed[index].get("blocking"))
    ]


def _artifact_map(rows: list[Any]) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        path = _normal_path(row.get("path"))
        if not path:
            continue
        result[path] = (
            int(row.get("bytes") or 0),
            str(row.get("sha256") or "").lower(),
        )
    return result


def _canonical_path(root: Path, value: str | Path | None) -> Path:
    path = Path(value or root / "source-arbitrated-transcript.json").expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("primary transcript must stay inside the bundle") from exc
    if not path.is_file():
        raise FileNotFoundError(f"canonical transcript not found: {path}")
    return path


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _required_json_path(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _normal_path(value: Any) -> str:
    if not value:
        return ""
    return str(Path(str(value)).expanduser().resolve()).replace("\\", "/").casefold()


def _destination(value: Any) -> str:
    text = str(value or "").strip()
    parsed = urlsplit(text if "://" in text else f"https://{text}")
    return str(parsed.hostname or "").lower()


def _expect_equal(
    rows: list[dict[str, str]], field: str, actual: Any, expected: Any
) -> None:
    if str(actual or "") != str(expected or ""):
        _mismatch(rows, field, "actual value differs from prepared candidate")


def _mismatch(rows: list[dict[str, str]], field: str, detail: str) -> None:
    rows.append({"field": field, "detail": detail})


def _safe_id(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "")).strip("-._")
    return clean[:120] or "secondary-asr"

def _text_sha256(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest() if value else ""
