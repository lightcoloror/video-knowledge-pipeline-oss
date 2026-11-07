from __future__ import annotations

from pathlib import Path
from typing import Any

from .content_asset_batch import _discover_bundles
from .models import now_iso
from .storage import read_json, write_json
from .transcript_semantic_correction import import_transcript_semantic_review_notes, transcript_semantic_correction_status

ACCEPTED_STATES = {"accepted", "accepted_no_candidates"}


def transcript_semantic_acceptance(
    bundle_dir: str | Path,
    *,
    output_dir: str | Path = "",
    write: bool = True,
) -> dict[str, Any]:
    """Read-only single-bundle semantic correction acceptance proof."""

    root = Path(bundle_dir).expanduser().resolve()
    row = _row_for_bundle(root)
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else root / "exports"
    json_path = out_dir / "transcript-semantic-acceptance.json"
    markdown_path = out_dir / "transcript-semantic-acceptance.md"
    result = {
        "schema": "video_knowledge_pipeline.transcript_semantic_acceptance.v1",
        "created_at": now_iso(),
        "bundle_dir": str(root),
        "output_dir": str(out_dir),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "ok": bool(row.get("accepted")),
        "status": "accepted" if row.get("accepted") else "needs_semantic_correction_action",
        "acceptance_state": row.get("acceptance_state", "unknown"),
        "semantic_status": row.get("semantic_status", "unknown"),
        "next_action_key": row.get("next_action_key", "none"),
        "item": row,
        "canonical_transcript_integrity": row.get("canonical_transcript_integrity") or {},
        "next_actions": row.get("next_actions") or [],
        "operator_boundary": {
            "read_only": True,
            "no_asr_run": True,
            "no_vision_or_cloud_call": True,
            "no_download": True,
            "does_not_modify_raw_sources": True,
            "does_not_execute_closure_or_export": True,
            "accepted_requires_canonical_export_hash_match": True,
        },
        "write": write,
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(json_path, result)
        markdown_path.write_text(_render_single_acceptance_markdown(result), encoding="utf-8")
    return result

def transcript_semantic_batch_acceptance(
    batch_input: str | Path,
    *,
    output_dir: str | Path = "",
    target_bundle_count: int = 3,
    limit: int = 0,
    write: bool = True,
) -> dict[str, Any]:
    """Summarize transcript semantic correction readiness across bundles.

    This is a read-only acceptance dashboard. It does not run ASR, vision,
    download, validation, closure, or export. It only inspects current bundle
    artifacts and turns per-bundle semantic correction status into a batch gate.
    """

    bundles = _discover_bundles(batch_input)
    all_bundle_count = len(bundles)
    if int(limit or 0) > 0:
        bundles = bundles[: int(limit or 0)]
    rows = [_row_for_bundle(bundle) for bundle in bundles]
    summary = _summary(rows, target_bundle_count=target_bundle_count)
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else _default_output_dir(batch_input)
    json_path = out_dir / "transcript-semantic-batch-acceptance.json"
    markdown_path = out_dir / "transcript-semantic-batch-acceptance.md"
    result = {
        "schema": "video_knowledge_pipeline.transcript_semantic_batch_acceptance.v1",
        "created_at": now_iso(),
        "batch_input": str(Path(batch_input).expanduser()),
        "output_dir": str(out_dir),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "target_bundle_count": int(target_bundle_count or 0),
        "limit": int(limit or 0),
        "limited": bool(int(limit or 0) > 0 and all_bundle_count > len(rows)),
        "discovered_bundle_count": all_bundle_count,
        "bundle_count": len(rows),
        "ok": bool(rows) and bool(summary.get("target_bundle_count_met")) and int(summary.get("not_accepted_count") or 0) == 0,
        "status": _batch_status(rows, summary),
        "summary": summary,
        "items": rows,
        "next_actions": _next_actions(rows, target_bundle_count=target_bundle_count),
        "operator_boundary": {
            "read_only": True,
            "no_asr_run": True,
            "no_vision_or_cloud_call": True,
            "no_download": True,
            "does_not_modify_raw_sources": True,
        },
        "write": write,
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(json_path, result)
        markdown_path.write_text(_render_markdown(result), encoding="utf-8")
    return result


def _row_for_bundle(bundle: Path) -> dict[str, Any]:
    root = bundle.expanduser().resolve()
    try:
        status = transcript_semantic_correction_status(root, write=False)
    except Exception as exc:
        return {
            "bundle_dir": str(root),
            "accepted": False,
            "acceptance_state": "status_error",
            "semantic_status": "status_error",
            "next_action_key": "inspect_bundle",
            "error": str(exc),
            "next_actions": [f"Inspect semantic correction artifacts for {root}: {exc}"],
        }
    canonical_integrity = _canonical_integrity_state(root)
    semantic_status = str(status.get("status") or "missing")
    candidate_discovery_status = str(status.get("candidate_discovery_status") or "not_planned")
    next_action_key = str(status.get("next_action_key") or "none")
    required_next_action_key = str(
        status.get("required_next_action_key") or next_action_key
    )
    completion_gate = (
        status.get("completion_gate")
        if isinstance(status.get("completion_gate"), dict)
        else {}
    )
    residual = int(status.get("final_residual_error_total") or 0)
    review_required = int(status.get("review_required_count") or 0)
    accepted_count = int(status.get("accepted_decision_count") or 0)
    summary_impact = _summary_impact_state(root, accepted_count=accepted_count)
    artifacts = status.get("artifacts") if isinstance(status.get("artifacts"), dict) else {}
    acceptance_state = _acceptance_state(semantic_status, residual=residual, review_required=review_required, summary_impact=summary_impact)
    if semantic_status == "no_candidates" and candidate_discovery_status in {"not_planned", "prompt_ready", "llm_prompt_ready", "suggestions_ready", "model_output_parse_failed"}:
        acceptance_state = "needs_candidate_discovery"
    if acceptance_state in ACCEPTED_STATES and not bool(status.get("ok")):
        acceptance_state = _completion_gate_acceptance_state(
            required_next_action_key,
            review_required=review_required,
        )
    if not bool(canonical_integrity.get("passed")):
        acceptance_state = "needs_canonical_export_refresh"
        required_next_action_key = "refresh_exports_or_review"
    return {
        "bundle_dir": str(root),
        "accepted": acceptance_state in ACCEPTED_STATES and bool(status.get("ok")),
        "acceptance_state": acceptance_state,
        "semantic_status": semantic_status,
        "next_action_key": next_action_key,
        "required_next_action_key": required_next_action_key,
        "completion_gate": completion_gate,
        "candidate_count": int(status.get("candidate_count") or 0),
        "accepted_decision_count": accepted_count,
        "review_required_count": review_required,
        "final_residual_error_total": residual,
        "summary_impact_status": summary_impact["status"],
        "summary_impact_required": summary_impact["required"],
        "summary_impact_residual_total": summary_impact["residual_total"],
        "summary_impact_corrected_hit_total": summary_impact["corrected_hit_total"],
        "summary_impact_absorption_rate": summary_impact["absorption_rate"],
        "candidate_type_counts": status.get("candidate_type_counts") if isinstance(status.get("candidate_type_counts"), dict) else {},
        "canonical_transcript_integrity": canonical_integrity,
        "risk_level_counts": status.get("risk_level_counts") if isinstance(status.get("risk_level_counts"), dict) else {},
        "evidence_source_counts": status.get("evidence_source_counts") if isinstance(status.get("evidence_source_counts"), dict) else {},
        "validation_rejection_reason_counts": status.get("validation_rejection_reason_counts") if isinstance(status.get("validation_rejection_reason_counts"), dict) else {},
        "review_closure_summary": status.get("review_closure_summary") if isinstance(status.get("review_closure_summary"), dict) else {},
        "chapter_risk_summary": status.get("chapter_risk_summary") if isinstance(status.get("chapter_risk_summary"), list) else [],
        "artifacts": artifacts,
        "evidence_files": _evidence_files(root, artifacts),
        "commands": status.get("commands") if isinstance(status.get("commands"), dict) else {},
        "next_actions": _row_next_actions(
            root,
            semantic_status,
            required_next_action_key,
            acceptance_state,
            summary_impact=summary_impact,
        ),
    }


def _canonical_integrity_state(root: Path) -> dict[str, Any]:
    try:
        from .knowledge_note_export import canonical_export_integrity_status

        return canonical_export_integrity_status(root)
    except Exception as exc:
        return {
            "status": "canonical_export_integrity_error",
            "passed": False,
            "canonical_path": "",
            "issues": [
                {
                    "key": "canonical_export_integrity_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            ],
        }


def _acceptance_state(semantic_status: str, *, residual: int, review_required: int, summary_impact: dict[str, Any] | None = None) -> str:
    summary = summary_impact or {"required": False, "status": "not_required"}
    if semantic_status == "impact_passed" and residual == 0:
        if not summary.get("required") or str(summary.get("status") or "") in {"passed", "no_accepted_decisions", "no_evaluable_replacements"}:
            return "accepted"
        if str(summary.get("status") or "") == "missing_report":
            return "needs_summary_impact_report"
        return "needs_summary_refresh_or_review"
    if semantic_status == "no_candidates":
        return "accepted_no_candidates"
    if semantic_status == "missing_pack":
        return "needs_pack"
    if semantic_status == "needs_closure":
        return "needs_closure"
    if semantic_status == "needs_impact_report":
        return "needs_impact_report"
    if semantic_status == "needs_summary_impact_report":
        return "needs_summary_impact_report"
    if semantic_status == "summary_impact_needs_fix":
        return "needs_summary_refresh_or_review"
    if semantic_status == "impact_needs_fix" or residual > 0:
        return "needs_export_refresh_or_review"
    if semantic_status in {"needs_llm_or_codex_review", "needs_human_review_or_new_result"} or review_required > 0:
        return "needs_review"
    return "needs_inspection"


def _completion_gate_acceptance_state(
    required_next_action_key: str,
    *,
    review_required: int,
) -> str:
    """Translate the additive completion gate without changing legacy status."""

    key = str(required_next_action_key or "inspect_bundle")
    if key == "run_candidate_discovery":
        return "needs_candidate_discovery"
    if key in {"validate_current_pack", "reimport_or_revalidate_current_pack"}:
        return "needs_validation"
    if key == "review_candidates" or int(review_required or 0) > 0:
        return "needs_review"
    if key == "run_closure":
        return "needs_closure"
    if key == "run_transcript_quality_gate":
        return "quality_gate_needs_fix"
    return "needs_completion_gate"


def _summary_impact_state(root: Path, *, accepted_count: int) -> dict[str, Any]:
    if int(accepted_count or 0) <= 0:
        return {"required": False, "status": "not_required", "residual_total": 0, "corrected_hit_total": 0, "absorption_rate": 0.0}
    path = root / "transcript-semantic-summary-impact-report.json"
    if not path.exists():
        return {"required": True, "status": "missing_report", "residual_total": 0, "corrected_hit_total": 0, "absorption_rate": 0.0}
    try:
        data = read_json(path)
    except Exception:
        return {"required": True, "status": "invalid_report", "residual_total": 0, "corrected_hit_total": 0, "absorption_rate": 0.0}
    if not isinstance(data, dict):
        return {"required": True, "status": "invalid_report", "residual_total": 0, "corrected_hit_total": 0, "absorption_rate": 0.0}
    return {
        "required": True,
        "status": str(data.get("status") or "missing_status"),
        "residual_total": int(data.get("summary_residual_original_total") or 0),
        "corrected_hit_total": int(data.get("summary_corrected_hit_total") or 0),
        "absorption_rate": float(data.get("summary_absorption_rate") or 0.0),
    }

def _evidence_files(root: Path, artifacts: dict[str, Any]) -> dict[str, bool]:
    keys = {
        "pack_json": "pack_json",
        "validation_json": "validation_json",
        "review_json": "review_json",
        "closure_json": "closure_json",
        "impact_json": "impact_json",
        "corrected_transcript_json": "corrected_transcript_json",
    }
    result: dict[str, bool] = {}
    for out_key, artifact_key in keys.items():
        value = str(artifacts.get(artifact_key) or "")
        path = Path(value).expanduser() if value else root / _fallback_name(out_key)
        if not path.is_absolute():
            path = root / path
        result[out_key] = path.exists()
    result["full_transcript_md"] = (root / "exports" / "full-transcript.md").exists()
    result["smart_summary_md"] = (root / "exports" / "smart-summary.md").exists()
    return result


def _fallback_name(key: str) -> str:
    return {
        "pack_json": "transcript-semantic-correction-pack.json",
        "validation_json": "transcript-semantic-correction-validation.json",
        "review_json": "transcript-semantic-correction-review.json",
        "closure_json": "transcript-semantic-correction-closure.json",
        "impact_json": "transcript-semantic-correction-impact-report.json",
        "corrected_transcript_json": "source-arbitrated-transcript.json",
    }.get(key, key)


def _summary(rows: list[dict[str, Any]], *, target_bundle_count: int) -> dict[str, Any]:
    by_acceptance: dict[str, int] = {}
    by_status: dict[str, int] = {}
    by_next_action: dict[str, int] = {}
    by_candidate_type: dict[str, int] = {}
    by_risk: dict[str, int] = {}
    by_rejection: dict[str, int] = {}
    totals = {
        "candidate_count": 0,
        "accepted_decision_count": 0,
        "review_required_count": 0,
        "final_residual_error_total": 0,
    }
    not_accepted: list[dict[str, Any]] = []
    for row in rows:
        acceptance = str(row.get("acceptance_state") or "unknown")
        status = str(row.get("semantic_status") or "unknown")
        next_action = str(row.get("next_action_key") or "none")
        by_acceptance[acceptance] = by_acceptance.get(acceptance, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1
        by_next_action[next_action] = by_next_action.get(next_action, 0) + 1
        totals["candidate_count"] += int(row.get("candidate_count") or 0)
        totals["accepted_decision_count"] += int(row.get("accepted_decision_count") or 0)
        totals["review_required_count"] += int(row.get("review_required_count") or 0)
        totals["final_residual_error_total"] += int(row.get("final_residual_error_total") or 0)
        _merge_counts(by_candidate_type, row.get("candidate_type_counts"))
        _merge_counts(by_risk, row.get("risk_level_counts"))
        _merge_counts(by_rejection, row.get("validation_rejection_reason_counts"))
        if not row.get("accepted"):
            not_accepted.append({
                "bundle_dir": row.get("bundle_dir", ""),
                "acceptance_state": acceptance,
                "semantic_status": status,
                "next_action_key": next_action,
                "review_required_count": int(row.get("review_required_count") or 0),
                "final_residual_error_total": int(row.get("final_residual_error_total") or 0),
            })
    accepted_count = sum(1 for row in rows if row.get("accepted"))
    return {
        "bundle_count": len(rows),
        "target_bundle_count": int(target_bundle_count or 0),
        "target_bundle_count_met": len(rows) >= int(target_bundle_count or 0),
        "accepted_count": accepted_count,
        "not_accepted_count": len(rows) - accepted_count,
        **totals,
        "by_acceptance_state": by_acceptance,
        "by_semantic_status": by_status,
        "by_next_action": by_next_action,
        "by_candidate_type": by_candidate_type,
        "by_risk_level": by_risk,
        "by_rejection_reason": by_rejection,
        "not_accepted_items": not_accepted,
    }


def _merge_counts(target: dict[str, int], values: Any) -> None:
    if not isinstance(values, dict):
        return
    for key, value in values.items():
        name = str(key or "unknown")
        target[name] = target.get(name, 0) + int(value or 0)


def _batch_status(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if not rows:
        return "no_bundles_found"
    if not summary.get("target_bundle_count_met"):
        return "needs_more_bundles_for_batch_acceptance"
    if int(summary.get("not_accepted_count") or 0) == 0:
        return "accepted"
    return "needs_semantic_correction_action"


def _row_next_actions(root: Path, semantic_status: str, next_action_key: str, acceptance_state: str, *, summary_impact: dict[str, Any] | None = None) -> list[str]:
    q = f"'{root}'"
    if acceptance_state in ACCEPTED_STATES:
        return []
    if acceptance_state == "needs_canonical_export_refresh":
        return [
            f".\\scripts\\video-knowledge.ps1 export-knowledge-note {q}",
            f".\\scripts\\video-knowledge.ps1 transcript-semantic-acceptance {q}",
            "Do not mark accepted/completed until canonical and derived export hashes match.",
        ]
    if acceptance_state == "needs_candidate_discovery":
        return [
            f".\\scripts\\video-knowledge.ps1 transcript-semantic-candidate-discovery-pack {q} --limit 40",
            f".\\scripts\\video-knowledge.ps1 transcript-semantic-candidate-discovery-codex-draft {q} --limit 40 --max-suggestions 40",
            f".\\scripts\\video-knowledge.ps1 import-transcript-semantic-candidate-suggestions {q} --input-json '{root / 'transcript-semantic-candidate-suggestions.codex.md'}'",
            f"Run candidate discovery before accepting no-candidate status for {q}.",
        ]
    if acceptance_state == "needs_summary_impact_report":
        return [f".\\scripts\\video-knowledge.ps1 transcript-semantic-summary-impact-report {q}"]
    if acceptance_state == "needs_summary_refresh_or_review":
        status = str((summary_impact or {}).get("status") or "")
        return [f".\\scripts\\video-knowledge.ps1 export-knowledge-note {q}", f".\\scripts\\video-knowledge.ps1 transcript-semantic-summary-impact-report {q}", f"Review smart-summary.md because summary impact status is {status}."]
    if next_action_key == "run_transcript_quality_gate":
        return [f".\\scripts\\video-knowledge.ps1 transcript-quality-gate {q}"]
    if next_action_key == "validate_current_pack":
        return [
            f"Review the current semantic pack and validate a hash-bound result for {q}.",
            f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-status {q}",
        ]
    if next_action_key == "reimport_or_revalidate_current_pack":
        return [
            f"Reimport or revalidate semantic decisions against the current pack for {q}.",
            "Do not reuse a stale validation result.",
        ]
    if semantic_status == "missing_pack":
        return [f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-pack {q}"]
    if acceptance_state == "needs_candidate_discovery":
        return [f".\\scripts\\video-knowledge.ps1 transcript-semantic-candidate-discovery-pack {q} --limit 40", f".\\scripts\\video-knowledge.ps1 transcript-semantic-candidate-discovery-llm-draft {q} --limit 40", f"Import candidate suggestions for {q} before accepting no-candidate status."]
    if next_action_key in {"validate_result", "run_llm_draft_preview", "execute_llm_or_use_codex"} or semantic_status == "needs_llm_or_codex_review":
        return [f"Review transcript-semantic-correction-pack, generate or fill transcript-semantic-correction-result.codex.md, then run validate-transcript-semantic-correction for {q}."]
    if next_action_key == "review_candidates":
        return [f"Open task-console.html or review pack, export review notes, then run import-transcript-semantic-review-notes for {q}."]
    if next_action_key == "run_closure":
        return [f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-closure {q} --input-json '<validated-result-json>'"]
    if next_action_key == "run_impact":
        return [f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-impact-report {q}"]
    if next_action_key == "refresh_exports_or_review":
        return [f".\\scripts\\video-knowledge.ps1 export-knowledge-note {q}", f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-impact-report {q}"]
    return [f"Inspect transcript semantic correction status for {q}: {semantic_status}/{next_action_key}"]


def _next_actions(rows: list[dict[str, Any]], *, target_bundle_count: int) -> list[str]:
    actions: list[str] = []
    if len(rows) < int(target_bundle_count or 0):
        actions.append(f"Add more bundles to reach target batch validation count: {len(rows)}/{int(target_bundle_count or 0)}.")
    for row in rows:
        actions.extend(str(action) for action in row.get("next_actions") or [] if str(action).strip())
    return _dedupe(actions)




def transcript_semantic_repair_queue(
    batch_input: str | Path,
    *,
    output_dir: str | Path = "",
    target_bundle_count: int = 3,
    limit: int = 0,
    write: bool = True,
) -> dict[str, Any]:
    """Build a preview-first repair queue for transcript semantic correction.

    The queue does not execute ASR, LLM, validation, closure, export, or writes to
    source media. It turns the current semantic-correction status of each bundle
    into operator-visible retry actions for UI/OpenClaw.
    """

    bundles = _discover_bundles(batch_input)
    all_bundle_count = len(bundles)
    if int(limit or 0) > 0:
        bundles = bundles[: int(limit or 0)]
    rows = [_queue_row_for_bundle(bundle) for bundle in bundles]
    summary = _queue_summary(rows, target_bundle_count=target_bundle_count)
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else _default_output_dir(batch_input)
    json_path = out_dir / "transcript-semantic-repair-queue.json"
    markdown_path = out_dir / "transcript-semantic-repair-queue.md"
    result = {
        "schema": "video_knowledge_pipeline.transcript_semantic_repair_queue.v1",
        "created_at": now_iso(),
        "batch_input": str(Path(batch_input).expanduser()),
        "output_dir": str(out_dir),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "target_bundle_count": int(target_bundle_count or 0),
        "limit": int(limit or 0),
        "limited": bool(int(limit or 0) > 0 and all_bundle_count > len(rows)),
        "discovered_bundle_count": all_bundle_count,
        "bundle_count": len(rows),
        "ok": int(summary.get("action_required_count") or 0) == 0 and len(rows) >= int(target_bundle_count or 0),
        "status": _queue_status(rows, summary),
        "summary": summary,
        "items": rows,
        "operator_boundary": {
            "preview_only": True,
            "does_not_execute_actions": True,
            "no_asr_run": True,
            "no_vision_or_cloud_call": True,
            "no_download": True,
            "does_not_modify_raw_sources": True,
            "llm_execution_requires_explicit_command": True,
        },
        "write": write,
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(json_path, result)
        markdown_path.write_text(_render_queue_markdown(result), encoding="utf-8")
    return result


def _queue_row_for_bundle(bundle: Path) -> dict[str, Any]:
    root = bundle.expanduser().resolve()
    try:
        status = transcript_semantic_correction_status(root, write=False)
    except Exception as exc:
        return {
            "bundle_dir": str(root),
            "semantic_status": "status_error",
            "acceptance_state": "status_error",
            "action_key": "inspect_bundle",
            "action_status": "blocked_or_failed",
            "action_kind": "operator_review_required",
            "machine_action_available": False,
            "human_review_required": True,
            "retry_command": f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-status '{root}'",
            "error": str(exc),
            "progress": _queue_progress({}, acceptance_state="status_error", action_status="blocked_or_failed"),
        }
    canonical_integrity = _canonical_integrity_state(root)
    semantic_status = str(status.get("status") or "missing")
    candidate_discovery_status = str(status.get("candidate_discovery_status") or "not_planned")
    legacy_next_action_key = str(status.get("next_action_key") or "none")
    required_next_action_key = str(
        status.get("required_next_action_key") or legacy_next_action_key
    )
    completion_gate = (
        status.get("completion_gate")
        if isinstance(status.get("completion_gate"), dict)
        else {}
    )
    residual = int(status.get("final_residual_error_total") or 0)
    review_required = int(status.get("review_required_count") or 0)
    accepted_count = int(status.get("accepted_decision_count") or 0)
    summary_impact = _summary_impact_state(root, accepted_count=accepted_count)
    acceptance_state = _acceptance_state(semantic_status, residual=residual, review_required=review_required, summary_impact=summary_impact)
    if semantic_status == "no_candidates" and candidate_discovery_status in {"not_planned", "prompt_ready", "llm_prompt_ready", "suggestions_ready", "model_output_parse_failed"}:
        acceptance_state = "needs_candidate_discovery"
    if acceptance_state in ACCEPTED_STATES and not bool(status.get("ok")):
        acceptance_state = _completion_gate_acceptance_state(
            required_next_action_key,
            review_required=review_required,
        )
    if not bool(canonical_integrity.get("passed")):
        acceptance_state = "needs_canonical_export_refresh"
        required_next_action_key = "refresh_exports_or_review"
    status_for_action = dict(status)
    status_for_action["required_next_action_key"] = required_next_action_key
    action = _queue_action(root, status_for_action, acceptance_state=acceptance_state, summary_impact=summary_impact)
    return {
        "bundle_dir": str(root),
        "semantic_status": semantic_status,
        "acceptance_state": acceptance_state,
        "accepted": acceptance_state in ACCEPTED_STATES and bool(status.get("ok")),
        "candidate_count": int(status.get("candidate_count") or 0),
        "accepted_decision_count": accepted_count,
        "review_required_count": review_required,
        "final_residual_error_total": residual,
        "summary_impact_status": summary_impact["status"],
        "summary_impact_required": summary_impact["required"],
        "summary_impact_residual_total": summary_impact["residual_total"],
        "summary_impact_corrected_hit_total": summary_impact["corrected_hit_total"],
        "summary_impact_absorption_rate": summary_impact["absorption_rate"],
        "llm_draft_status": str(status.get("llm_draft_status") or "not_planned"),
        "canonical_transcript_integrity": canonical_integrity,
        "llm_draft_next_action": str(status.get("llm_draft_next_action") or "run_llm_draft_preview"),
        "candidate_discovery_status": candidate_discovery_status,
        "candidate_discovery_next_action": str(status.get("candidate_discovery_next_action") or "run_candidate_discovery"),
        "candidate_discovery_segment_count": int(status.get("candidate_discovery_segment_count") or 0),
        "candidate_discovery_suggestion_count": int(status.get("candidate_discovery_suggestion_count") or 0),
        "candidate_discovery_imported_candidate_count": int(status.get("candidate_discovery_imported_candidate_count") or 0),
        "readable_impact_status": str(status.get("readable_impact_status") or "missing"),
        "readable_required_residual_total": int(status.get("readable_required_residual_total") or 0),
        **action,
        "next_action_key": legacy_next_action_key,
        "required_next_action_key": required_next_action_key,
        "completion_gate": completion_gate,
        "progress": _queue_progress(status, acceptance_state=acceptance_state, action_status=str(action.get("action_status") or "unknown")),
    }


def _queue_action(root: Path, status: dict[str, Any], *, acceptance_state: str, summary_impact: dict[str, Any] | None = None) -> dict[str, Any]:
    q = f"'{root}'"
    commands = status.get("commands") if isinstance(status.get("commands"), dict) else {}
    semantic_status = str(status.get("status") or "missing")
    next_action_key = str(
        status.get("required_next_action_key")
        or status.get("next_action_key")
        or "none"
    )
    llm_status = str(status.get("llm_draft_status") or "not_planned")
    llm_next = str(status.get("llm_draft_next_action") or "run_llm_draft_preview")
    review_required = int(status.get("review_required_count") or 0)
    residual = int(status.get("final_residual_error_total") or 0)
    if acceptance_state in ACCEPTED_STATES:
        return _action("none", "completed", "skip", False, False, "", "Already accepted.")
    if acceptance_state == "needs_canonical_export_refresh":
        return _action("refresh_exports_or_review", "needs_execution", "local_export_refresh", True, False, f".\\scripts\\video-knowledge.ps1 export-knowledge-note {q}", "Canonical and derived export hashes must match before acceptance.")
    if acceptance_state == "needs_candidate_discovery":
        discovery_status = str(status.get("candidate_discovery_status") or "not_planned")
        discovery_next = str(status.get("candidate_discovery_next_action") or "run_candidate_discovery")
        if discovery_status == "not_planned":
            return _action("run_candidate_discovery", "needs_execution", "local_preview", True, False, commands.get("run_candidate_discovery") or f".\\scripts\\video-knowledge.ps1 transcript-semantic-candidate-discovery-pack {q} --limit 40", "No initial candidates found; run local candidate discovery over ASR/subtitle/visual evidence before accepting no-candidate state.")
        if discovery_status == "prompt_ready":
            return _action("run_candidate_discovery_llm_preview", "needs_execution", "llm_preview", True, False, commands.get("run_candidate_discovery_llm_preview") or f".\\scripts\\video-knowledge.ps1 transcript-semantic-candidate-discovery-llm-draft {q} --limit 40", "Candidate discovery prompt is ready; generate the LLM/Codex suggestions prompt without cloud execution.")
        if discovery_status == "llm_prompt_ready":
            return _action("execute_candidate_discovery_llm_or_use_codex", "operator_confirmation_required", "llm_or_codex_review", False, True, commands.get("execute_candidate_discovery_llm_or_use_codex") or "", "Fill candidate suggestions with Codex or explicitly execute a configured LLM provider, then import suggestions.")
        if discovery_status == "suggestions_ready":
            return _action("import_candidate_suggestions", "needs_execution", "safe_import_candidates_only", True, False, commands.get("import_candidate_suggestions") or f".\\scripts\\video-knowledge.ps1 import-transcript-semantic-candidate-suggestions {q} --input-json '{root / 'transcript-semantic-candidate-suggestions.llm.md'}'", "Import discovered suggestions as normal semantic candidates; validation and closure still required.")
        if discovery_status == "model_output_parse_failed":
            return _action("retry_candidate_discovery_llm_or_manual_review", "needs_execution_or_review", "llm_preview_or_manual_review", True, True, commands.get("run_candidate_discovery_llm_preview") or f".\\scripts\\video-knowledge.ps1 transcript-semantic-candidate-discovery-llm-draft {q} --limit 40", "Candidate discovery model output could not be parsed; retry preview or fill suggestions manually.")
        return _action(discovery_next, "operator_review_required", "human_review_required", False, True, commands.get(discovery_next) or "", f"Inspect candidate discovery state before accepting no-candidate status: {discovery_status}.")
    if acceptance_state == "needs_summary_impact_report":
        return _action("run_summary_impact", "needs_execution", "local_report", True, False, f".\\scripts\\video-knowledge.ps1 transcript-semantic-summary-impact-report {q}", "Check whether accepted corrections are visible in smart-summary.")
    if acceptance_state == "needs_summary_refresh_or_review":
        status = str((summary_impact or {}).get("status") or "")
        human = status in {"needs_fix", "not_proven", "invalid_report"}
        return _action("refresh_summary_impact_or_review", "needs_execution_or_review", "local_export_or_review", True, human, f".\\scripts\\video-knowledge.ps1 export-knowledge-note {q}; .\\scripts\\video-knowledge.ps1 transcript-semantic-summary-impact-report {q}", f"Refresh smart-summary and rerun summary impact; current summary impact status={status}.")
    if next_action_key == "run_transcript_quality_gate":
        return _action(
            "run_transcript_quality_gate",
            "needs_execution",
            "local_quality_gate",
            True,
            False,
            commands.get("run_transcript_quality_gate")
            or f".\\scripts\\video-knowledge.ps1 transcript-quality-gate {q}",
            "Run the transcript quality gate against the formal corrected transcript before acceptance.",
        )
    if next_action_key == "validate_current_pack":
        if llm_status == "executed":
            command = commands.get("validate_llm_result") or ""
        elif llm_status == "codex_draft_ready":
            command = commands.get("validate_result") or ""
        else:
            command = commands.get("run_llm_draft_preview") or commands.get("llm_draft_preview") or ""
        return _action(
            "validate_current_pack",
            "needs_execution",
            "safe_validation_or_review_preparation",
            True,
            False,
            command,
            "Current-pack candidates remain unadjudicated; prepare or validate a hash-bound result.",
        )
    if next_action_key == "reimport_or_revalidate_current_pack":
        return _action(
            "reimport_or_revalidate_current_pack",
            "needs_execution_or_review",
            "safe_revalidation",
            True,
            True,
            commands.get("validate_result") or commands.get("status") or "",
            "The validation identity is stale; reimport or revalidate only against the current pack.",
        )
    if semantic_status == "missing_pack":
        return _action("build_pack", "needs_execution", "local_preview", True, False, commands.get("pack") or f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-pack {q}", "Build the semantic correction evidence pack.")
    if next_action_key in {"validate_result", "run_llm_draft_preview", "execute_llm_or_use_codex"} or semantic_status == "needs_llm_or_codex_review":
        if llm_status == "executed":
            return _action("validate_llm_result", "needs_execution", "safe_parse_only", True, False, commands.get("validate_llm_result") or f".\\scripts\\video-knowledge.ps1 validate-transcript-semantic-correction {q} --input-json '{root / 'transcript-semantic-correction-result.llm.md'}'", "Validate existing LLM result before closure.")
        if llm_status == "codex_draft_ready":
            return _action("validate_result", "needs_execution", "safe_parse_only", True, False, commands.get("validate_result") or f".\\scripts\\video-knowledge.ps1 validate-transcript-semantic-correction {q} --input-json '{root / 'transcript-semantic-correction-result.codex.md'}'", "Validate existing local Codex result before closure.")
        if llm_status in {"not_planned", "model_output_parse_failed"}:
            key = "run_llm_draft_preview" if llm_status == "not_planned" else "retry_llm_or_manual_review"
            return _action(key, "needs_execution", "llm_preview", True, False, commands.get(key) or commands.get("llm_draft_preview") or f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-llm-draft {q} --limit 80", "Generate or refresh the LLM/Codex review prompt.")
        if llm_status == "prompt_ready":
            return _action("execute_llm_or_use_codex", "operator_confirmation_required", "llm_or_codex_review", False, True, commands.get("execute_llm_or_use_codex") or commands.get("codex_draft") or "", "Run Codex review locally or explicitly execute a configured LLM provider, then validate the result.")
        return _action("review_candidates", "operator_review_required", "human_review_required", False, True, commands.get("codex_draft") or "", f"Review candidates; semantic={semantic_status}, llm={llm_status}, next={llm_next}.")
    if next_action_key == "run_closure":
        human_after = review_required > 0
        reason = "Apply validated corrections, refresh readable exports, then rerun impact reports."
        if human_after:
            reason += " Some low-confidence candidates remain open for later review."
        return _action("run_closure", "needs_execution", "local_write_after_validation", True, human_after, commands.get("closure") or f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-closure {q} --input-json PATH_TO_VALIDATED_RESULT", reason)
    if next_action_key == "run_impact":
        return _action("run_impact", "needs_execution", "local_report", True, bool(review_required), commands.get("run_impact") or commands.get("impact") or f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-impact-report {q}", "Check residual errors after closure/export; unresolved review candidates remain separate.")
    if next_action_key == "run_readable_impact":
        return _action("run_readable_impact", "needs_execution", "local_report", True, bool(review_required), commands.get("run_readable_impact") or commands.get("readable_impact") or f".\\scripts\\video-knowledge.ps1 transcript-semantic-readable-impact-report {q}", "Check whether accepted corrections reached full-transcript and smart-summary; unresolved review candidates remain separate.")
    if next_action_key == "refresh_exports_or_review" or residual > 0:
        return _action("refresh_exports_or_review", "needs_execution_or_review", "local_export_or_review", True, bool(residual), f".\\scripts\\video-knowledge.ps1 export-knowledge-note {q}; .\\scripts\\video-knowledge.ps1 transcript-semantic-correction-impact-report {q}", "Refresh exports, then rerun impact; review remaining residuals if any.")
    if review_required > 0 or next_action_key == "review_candidates":
        return _action("review_candidates", "operator_review_required", "human_review_required", False, True, f"Open {root / 'task-console.html'} and export transcript-semantic-correction-review-notes.json", "Human review is required for low-confidence or conflicting candidates.")
    return _action("inspect_bundle", "operator_review_required", "unknown", False, True, commands.get("status") or f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-status {q}", f"Inspect semantic correction state: {semantic_status}/{next_action_key}.")


def _action(action_key: str, action_status: str, action_kind: str, machine: bool, human: bool, command: str, reason: str) -> dict[str, Any]:
    return {
        "action_key": action_key,
        "next_action_key": action_key,
        "action_status": action_status,
        "action_kind": action_kind,
        "machine_action_available": bool(machine),
        "human_review_required": bool(human),
        "retry_command": command,
        "reason": reason,
    }


def _queue_progress(status: dict[str, Any], *, acceptance_state: str, action_status: str) -> dict[str, Any]:
    if acceptance_state in ACCEPTED_STATES:
        step = 8
    elif acceptance_state in {"needs_summary_impact_report", "needs_summary_refresh_or_review"}:
        step = 7
    else:
        semantic_status = str(status.get("status") or "missing")
        if semantic_status == "missing_pack":
            step = 1
        elif acceptance_state == "needs_candidate_discovery":
            step = 2
        elif semantic_status == "needs_llm_or_codex_review":
            step = 2
        elif semantic_status == "needs_human_review_or_new_result":
            step = 3
        elif semantic_status == "needs_closure":
            step = 4
        elif semantic_status == "needs_impact_report":
            step = 5
        elif semantic_status == "needs_readable_impact_report":
            step = 6
        else:
            step = 0 if action_status == "blocked_or_failed" else 3
    total = 8
    return {"step": step, "total_steps": total, "percent": round((step / total) * 100, 1) if total else 0.0}

def _queue_summary(rows: list[dict[str, Any]], *, target_bundle_count: int) -> dict[str, Any]:
    by_action: dict[str, int] = {}
    by_status: dict[str, int] = {}
    by_acceptance: dict[str, int] = {}
    by_action_status: dict[str, int] = {}
    machine = 0
    human = 0
    action_required = 0
    for row in rows:
        by_action[str(row.get("action_key") or "unknown")] = by_action.get(str(row.get("action_key") or "unknown"), 0) + 1
        by_status[str(row.get("semantic_status") or "unknown")] = by_status.get(str(row.get("semantic_status") or "unknown"), 0) + 1
        by_acceptance[str(row.get("acceptance_state") or "unknown")] = by_acceptance.get(str(row.get("acceptance_state") or "unknown"), 0) + 1
        by_action_status[str(row.get("action_status") or "unknown")] = by_action_status.get(str(row.get("action_status") or "unknown"), 0) + 1
        if row.get("machine_action_available"):
            machine += 1
        if row.get("human_review_required"):
            human += 1
        if str(row.get("action_key") or "none") != "none":
            action_required += 1
    return {
        "bundle_count": len(rows),
        "target_bundle_count": int(target_bundle_count or 0),
        "target_bundle_count_met": len(rows) >= int(target_bundle_count or 0),
        "action_required_count": action_required,
        "machine_action_available_count": machine,
        "human_review_required_count": human,
        "by_action_key": by_action,
        "by_action_status": by_action_status,
        "by_semantic_status": by_status,
        "by_acceptance_state": by_acceptance,
    }


def _queue_status(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if not rows:
        return "no_bundles_found"
    if not summary.get("target_bundle_count_met"):
        return "needs_more_bundles_for_queue"
    if int(summary.get("action_required_count") or 0) == 0:
        return "complete"
    if int(summary.get("human_review_required_count") or 0) > 0:
        return "needs_human_review"
    return "machine_actions_available"



def transcript_semantic_batch_review_pack(
    batch_input: str | Path,
    *,
    output_dir: str | Path = "",
    target_bundle_count: int = 3,
    limit: int = 0,
    max_candidates_per_bundle: int = 0,
    write: bool = True,
) -> dict[str, Any]:
    """Build a cross-bundle review pack for semantic correction candidates."""

    bundles = _discover_bundles(batch_input)
    all_bundle_count = len(bundles)
    if int(limit or 0) > 0:
        bundles = bundles[: int(limit or 0)]
    items: list[dict[str, Any]] = []
    bundle_summaries: list[dict[str, Any]] = []
    for bundle in bundles:
        root = bundle.expanduser().resolve()
        rows = _review_rows_for_bundle(root, max_candidates=max_candidates_per_bundle)
        bundle_summaries.append({"bundle_dir": str(root), "review_item_count": len(rows)})
        items.extend(rows)
    todo_reviews = [_todo_review_row(row) for row in items]
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else _default_output_dir(batch_input)
    json_path = out_dir / "transcript-semantic-batch-review-pack.json"
    markdown_path = out_dir / "transcript-semantic-batch-review-pack.md"
    todo_path = out_dir / "transcript-semantic-batch-review-notes.todo.json"
    prompt_path = out_dir / "transcript-semantic-batch-codex-review-prompt.md"
    todo = {
        "schema": "video_knowledge_pipeline.transcript_semantic_batch_review_notes.v1",
        "created_at": now_iso(),
        "source_review_pack": str(json_path),
        "reviews": todo_reviews,
    }
    result = {
        "schema": "video_knowledge_pipeline.transcript_semantic_batch_review_pack.v1",
        "created_at": now_iso(),
        "batch_input": str(Path(batch_input).expanduser()),
        "output_dir": str(out_dir),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "todo_json_path": str(todo_path),
        "codex_prompt_path": str(prompt_path),
        "target_bundle_count": int(target_bundle_count or 0),
        "limit": int(limit or 0),
        "limited": bool(int(limit or 0) > 0 and all_bundle_count > len(bundles)),
        "discovered_bundle_count": all_bundle_count,
        "bundle_count": len(bundles),
        "status": "review_pack_ready" if items else "no_review_items",
        "ok": True,
        "review_item_count": len(items),
        "todo_review_count": len(todo_reviews),
        "editable_reviews": todo_reviews[:80],
        "editable_review_truncated": len(todo_reviews) > 80,
        "items": items,
        "todo": todo,
        "bundle_summaries": bundle_summaries,
        "summary": {
            "review_item_count": len(items),
            "todo_review_count": len(todo_reviews),
            "by_correction_type": _count_by_key(items, "correction_type"),
            "by_risk_level": _count_by_key(items, "risk_level"),
        },
        "paths": {
            "json": str(json_path),
            "markdown": str(markdown_path),
            "todo_json": str(todo_path),
            "codex_prompt": str(prompt_path),
        },
        "commands": {
            "codex_review_draft": f".\\scripts\\video-knowledge.ps1 transcript-semantic-batch-codex-review-draft {json_path}",
            "import_review_notes": f".\\scripts\\video-knowledge.ps1 transcript-semantic-batch-import-review-notes {todo_path}",
        },
        "operator_boundary": {
            "local_only": True,
            "review_pack_only": True,
            "no_cloud_call": True,
            "no_asr_run": True,
            "no_vision_or_cloud_call": True,
            "no_download": True,
            "does_not_modify_raw_sources": True,
        },
        "write": write,
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(json_path, result)
        write_json(todo_path, todo)
        markdown_path.write_text(_render_batch_review_pack_markdown(result), encoding="utf-8")
        prompt_path.write_text(_render_batch_review_prompt(result), encoding="utf-8")
    return result


def transcript_semantic_batch_import_review_notes(
    review_json: str | Path,
    *,
    output_dir: str | Path = "",
    min_confidence: float = 0.88,
    write: bool = True,
) -> dict[str, Any]:
    """Import a cross-bundle review-notes JSON into each referenced bundle."""

    review_path = Path(review_json).expanduser().resolve()
    payload = read_json(review_path)
    rows = [row for row in (payload.get("reviews") if isinstance(payload, dict) else []) or [] if isinstance(row, dict)]
    bundles: dict[str, list[dict[str, Any]]] = {}
    skipped: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        bundle_dir = str(row.get("bundle_dir") or "").strip()
        if not bundle_dir:
            skipped.append({"row_number": idx, "candidate_id": row.get("candidate_id"), "reason": "missing_bundle_dir"})
            continue
        bundles.setdefault(str(Path(bundle_dir).expanduser().resolve()), []).append(row)
    imports: list[dict[str, Any]] = []
    bundle_summaries: list[dict[str, Any]] = []
    by_validation: dict[str, int] = {}
    post_next_actions: dict[str, int] = {}
    imported_decisions = 0
    accepted_decisions = 0
    review_required = 0
    closure_ready = 0
    open_review = 0
    for bundle_dir, bundle_rows in sorted(bundles.items()):
        root = Path(bundle_dir)
        import_result = import_transcript_semantic_review_notes(root, review_json=review_path, min_confidence=min_confidence, write=write)
        imports.append(import_result)
        validation = import_result.get("validation") if isinstance(import_result.get("validation"), dict) else {}
        validation_status = str(validation.get("status") or "unknown")
        by_validation[validation_status] = by_validation.get(validation_status, 0) + 1
        decision_count = int(import_result.get("decision_count") or 0)
        accepted_count = int(validation.get("accepted_decision_count") or 0)
        review_count = int(validation.get("review_required_count") or 0)
        imported_decisions += decision_count
        accepted_decisions += accepted_count
        review_required += review_count
        queue = transcript_semantic_repair_queue(root, target_bundle_count=1, write=False)
        next_action = "none"
        if queue.get("items"):
            first = queue["items"][0]
            if isinstance(first, dict):
                next_action = str(first.get("action_key") or "none")
        post_next_actions[next_action] = post_next_actions.get(next_action, 0) + 1
        if next_action == "run_closure":
            closure_ready += 1
        if review_count > 0:
            open_review += 1
        bundle_summaries.append(
            {
                "bundle_dir": bundle_dir,
                "input_review_count": len(bundle_rows),
                "status": import_result.get("status"),
                "decision_count": decision_count,
                "accepted_decision_count": accepted_count,
                "review_required_count": review_count,
                "validation_status": validation_status,
                "post_import_next_action": next_action,
            }
        )
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else review_path.parent
    json_path = out_dir / "transcript-semantic-batch-review-import.json"
    markdown_path = out_dir / "transcript-semantic-batch-review-import.md"
    status = _batch_import_status(imported_decisions, accepted_decisions, review_required)
    result = {
        "schema": "video_knowledge_pipeline.transcript_semantic_batch_review_import.v1",
        "created_at": now_iso(),
        "review_json": str(review_path),
        "output_dir": str(out_dir),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "status": status,
        "ok": imported_decisions > 0,
        "bundle_count": len(bundles),
        "review_row_count": len(rows),
        "imported_decision_count": imported_decisions,
        "accepted_decision_count": accepted_decisions,
        "review_required_count": review_required,
        "closure_ready_bundle_count": closure_ready,
        "open_review_bundle_count": open_review,
        "skipped_count": len(skipped),
        "skipped": skipped,
        "by_validation_status": by_validation,
        "post_import_next_action_counts": post_next_actions,
        "imports": imports,
        "bundle_summaries": bundle_summaries,
        "operator_boundary": {
            "local_only": True,
            "no_cloud_call": True,
            "no_asr_run": True,
            "no_vision_or_cloud_call": True,
            "no_download": True,
            "closure_still_required": True,
        },
        "write": write,
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(json_path, result)
        markdown_path.write_text(_render_batch_import_markdown(result), encoding="utf-8")
    return result


def transcript_semantic_batch_codex_review_draft(
    review_pack_json: str | Path,
    *,
    output_dir: str | Path = "",
    write: bool = True,
) -> dict[str, Any]:
    """Generate conservative local Codex-substitute review notes from a batch review pack."""

    pack_path = Path(review_pack_json).expanduser().resolve()
    pack = read_json(pack_path)
    rows = [row for row in (pack.get("items") if isinstance(pack, dict) else []) or [] if isinstance(row, dict)]
    reviews = [_codex_review_draft_row(row) for row in rows]
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else pack_path.parent
    json_path = out_dir / "transcript-semantic-batch-review-notes.codex-draft.json"
    markdown_path = out_dir / "transcript-semantic-batch-review-notes.codex-draft.md"
    result = {
        "schema": "video_knowledge_pipeline.transcript_semantic_batch_review_notes.v1",
        "created_at": now_iso(),
        "source_review_pack": str(pack_path),
        "status": "codex_draft_ready" if reviews else "no_review_items",
        "ok": True,
        "review_count": len(reviews),
        "by_review_status": _count_by_key(reviews, "review_status"),
        "reviews": reviews,
        "operator_boundary": {
            "local_codex_substitute": True,
            "no_cloud_call": True,
            "draft_requires_human_or_validation_review": True,
        },
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(json_path, result)
        markdown_path.write_text(_render_batch_codex_review_draft_markdown(result), encoding="utf-8")
    return result


def transcript_semantic_repair_run(
    batch_input: str | Path,
    *,
    output_dir: str | Path = "",
    target_bundle_count: int = 3,
    limit: int = 0,
    execute_safe_actions: bool = False,
    max_actions: int = 0,
    max_rounds: int = 1,
    allow_closure: bool = False,
    allow_llm: bool = False,
    provider_config: dict[str, Any] | None = None,
    llm_limit: int = 80,
    business_authorization_path: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Preview or execute safe local semantic-correction repair actions."""

    queue = transcript_semantic_repair_queue(
        batch_input,
        output_dir=output_dir,
        target_bundle_count=target_bundle_count,
        limit=limit,
        write=write,
    )
    executions: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    remaining_actions = int(max_actions or 0)
    round_count = max(1, int(max_rounds or 1)) if execute_safe_actions else 1
    current_queue = queue
    for round_index in range(1, round_count + 1):
        actionable = [row for row in current_queue.get("items") or [] if isinstance(row, dict) and str(row.get("action_key") or "none") != "none"]
        if remaining_actions > 0:
            actionable = actionable[:remaining_actions]
        round_executions = [
            _repair_run_item(
                row,
                execute=execute_safe_actions,
                allow_closure=allow_closure,
                allow_llm=allow_llm,
                provider_config=provider_config,
                llm_limit=llm_limit,
                business_authorization_path=business_authorization_path,
            )
            for row in actionable
        ]
        for item in round_executions:
            item["round"] = round_index
        executions.extend(round_executions)
        rounds.append({"round": round_index, "action_count": len(actionable), "executed_count": sum(1 for item in round_executions if item.get("executed")), "action_keys": [str(item.get("action_key") or "") for item in round_executions]})
        if remaining_actions > 0:
            remaining_actions = max(0, remaining_actions - len(actionable))
            if remaining_actions == 0:
                break
        if not execute_safe_actions or not actionable:
            break
        if not any(item.get("executed") for item in round_executions):
            break
        if round_index >= round_count:
            break
        current_queue = transcript_semantic_repair_queue(
            batch_input,
            output_dir=output_dir,
            target_bundle_count=target_bundle_count,
            limit=limit,
            write=write,
        )
    after_queue: dict[str, Any] | None = None
    if execute_safe_actions:
        after_queue = transcript_semantic_repair_queue(
            batch_input,
            output_dir=output_dir,
            target_bundle_count=target_bundle_count,
            limit=limit,
            write=write,
        )
    summary = _repair_run_summary(executions)
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else _default_output_dir(batch_input)
    json_path = out_dir / "transcript-semantic-repair-run.json"
    markdown_path = out_dir / "transcript-semantic-repair-run.md"
    result = {
        "schema": "video_knowledge_pipeline.transcript_semantic_repair_run.v1",
        "created_at": now_iso(),
        "batch_input": str(Path(batch_input).expanduser()),
        "output_dir": str(out_dir),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "execute_safe_actions": bool(execute_safe_actions),
        "allow_closure": bool(allow_closure),
        "allow_llm": bool(allow_llm),
        "llm_limit": int(llm_limit or 0),
        "max_actions": int(max_actions or 0),
        "max_rounds": int(max_rounds or 1),
        "target_bundle_count": int(target_bundle_count or 0),
        "limit": int(limit or 0),
        "status": _repair_run_status(executions, execute=execute_safe_actions),
        "ok": _repair_run_ok(executions, execute=execute_safe_actions),
        "summary": summary,
        "before_queue": queue,
        "after_queue": after_queue,
        "executions": executions,
        "rounds": rounds,
        "operator_boundary": {
            "preview_by_default": True,
            "execute_requires_execute_safe_actions": True,
            "llm_provider_call_requires_allow_llm": True,
            "provider_config_not_persisted": True,
            "no_asr_run": True,
            "no_vision_or_cloud_call": True,
            "no_download": True,
            "does_not_modify_raw_sources": True,
            "closure_requires_allow_closure": True,
            "max_rounds_only_repeats_safe_queue_actions": True,
        },
        "write": write,
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(json_path, result)
        markdown_path.write_text(_render_repair_run_markdown(result), encoding="utf-8")
    return result


def _review_rows_for_bundle(root: Path, *, max_candidates: int = 0) -> list[dict[str, Any]]:
    manifest = read_json(root / "manifest.json") if (root / "manifest.json").exists() else {}
    title = str(manifest.get("title") or _fallback_name(root)) if isinstance(manifest, dict) else _fallback_name(root)
    pack = read_json(root / "transcript-semantic-correction-pack.json") if (root / "transcript-semantic-correction-pack.json").exists() else {}
    candidates = [row for row in (pack.get("candidates") if isinstance(pack, dict) else []) or [] if isinstance(row, dict)]
    if int(max_candidates or 0) > 0:
        candidates = candidates[: int(max_candidates or 0)]
    return [_review_item_from_candidate(root, title, candidate, index) for index, candidate in enumerate(candidates, start=1)]


def _review_item_from_candidate(root: Path, title: str, candidate: dict[str, Any], index: int) -> dict[str, Any]:
    evidence = [row for row in candidate.get("evidence") or [] if isinstance(row, dict)]
    evidence_ids = [str(item) for item in candidate.get("evidence_ids") or [] if str(item)]
    for row in evidence:
        evidence_id = str(row.get("evidence_id") or "")
        if evidence_id and evidence_id not in evidence_ids:
            evidence_ids.append(evidence_id)
    candidate_id = str(candidate.get("candidate_id") or f"semcorr-{index:04d}")
    start = candidate.get("start", candidate.get("segment_start", 0.0))
    end = candidate.get("end", candidate.get("segment_end", start))
    return {
        "review_id": f"{root.name}-{candidate_id}",
        "bundle_dir": str(root),
        "bundle_title": title,
        "candidate_id": candidate_id,
        "correction_type": str(candidate.get("correction_type") or "ordinary_word"),
        "risk_level": str(candidate.get("risk_level") or "unknown"),
        "time_range": str(candidate.get("time_range") or ""),
        "start": start,
        "end": end,
        "original_text": str(candidate.get("original_text") or ""),
        "suggested_text": str(candidate.get("suggested_text") or candidate.get("candidate_text") or candidate.get("canonical_hint") or candidate.get("corrected_text") or ""),
        "context_text": str(candidate.get("context_text") or candidate.get("transcript_context") or ""),
        "reason": str(candidate.get("reason") or ""),
        "evidence_ids": evidence_ids,
        "evidence": evidence,
        "evidence_source_types": [str(item) for item in candidate.get("evidence_source_types") or [] if str(item)],
        "timeline_indexes": candidate.get("timeline_indexes") or [],
        "segment_index": candidate.get("segment_index"),
    }


def _todo_review_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload.setdefault("review_status", "needs_more_evidence")
    payload.setdefault("corrected_text", row.get("suggested_text") or "")
    payload.setdefault("confidence", 0.0)
    payload.setdefault("review_note", "")
    return payload


def _codex_review_draft_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = _todo_review_row(row)
    correction_type = str(row.get("correction_type") or "")
    suggested = str(row.get("suggested_text") or row.get("corrected_text") or "").strip()
    original = str(row.get("original_text") or "").strip()
    evidence_text = " ".join(str(item.get("text") or item.get("content") or "") for item in row.get("evidence") or [] if isinstance(item, dict))
    if correction_type == "segment_boundary":
        status = "needs_more_evidence"
        confidence = 0.25
        note = "Segment boundary changes need source transcript review before applying."
    elif suggested:
        status = "accept_correction"
        confidence = 0.92 if suggested.lower() in evidence_text.lower() or suggested in evidence_text else 0.88
        note = "Suggested correction has supporting evidence in the review pack."
    elif original and len(original) <= 2:
        status = "keep_original"
        confidence = 0.9
        note = "Low-information filler or short token; keep original unless stronger evidence appears."
    else:
        status = "needs_more_evidence"
        confidence = 0.35
        note = "No safe corrected text is available in local evidence."
    payload["review_status"] = status
    payload["confidence"] = confidence
    payload["review_note"] = note
    if status == "keep_original":
        payload["corrected_text"] = original
    return payload


def _batch_import_status(imported_decisions: int, accepted_decisions: int, review_required: int) -> str:
    if imported_decisions <= 0:
        return "no_importable_decisions"
    if accepted_decisions > 0 and review_required > 0:
        return "imported_partial_review_remaining"
    if review_required > 0:
        return "imported_review_remaining"
    if accepted_decisions > 0:
        return "imported_closure_ready"
    return "imported"


def _count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _render_batch_review_pack_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Transcript Semantic Correction Batch Review Pack",
        "",
        f"- Created: `{result.get('created_at')}`",
        f"- Status: `{result.get('status')}`",
        f"- Review items: `{int(result.get('review_item_count') or 0)}`",
        f"- Todo notes: `{result.get('todo_json_path')}`",
        "",
        "| Bundle | Candidate | Type | Risk | Original | Suggested |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in result.get("items") or []:
        lines.append(
            "| {bundle} | `{candidate}` | `{kind}` | `{risk}` | {original} | {suggested} |".format(
                bundle=str(row.get("bundle_title") or row.get("bundle_dir") or "").replace("|", "/"),
                candidate=str(row.get("candidate_id") or ""),
                kind=str(row.get("correction_type") or ""),
                risk=str(row.get("risk_level") or ""),
                original=str(row.get("original_text") or "").replace("|", "/")[:120],
                suggested=str(row.get("suggested_text") or "").replace("|", "/")[:120],
            )
        )
    lines.extend(["", "## Operator Boundary", "", "- Local review pack only; no cloud/API call, ASR, vision, download, or source mutation."])
    return "\n".join(lines).rstrip() + "\n"


def _render_batch_review_prompt(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Codex Review Prompt: Transcript Semantic Batch",
            "",
            "Review the JSON pack and fill `review_status`, `corrected_text`, `confidence`, and `review_note` conservatively.",
            "Only accept a correction when evidence clearly supports it. Use `needs_more_evidence` for uncertain cases.",
            "",
            f"Review pack: `{result.get('json_path')}`",
            f"Todo notes: `{result.get('todo_json_path')}`",
        ]
    ).rstrip() + "\n"


def _render_batch_import_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Transcript Semantic Correction Batch Review Import",
        "",
        f"- Created: `{result.get('created_at')}`",
        f"- Status: `{result.get('status')}`",
        f"- Imported decisions: `{int(result.get('imported_decision_count') or 0)}`",
        f"- Accepted decisions: `{int(result.get('accepted_decision_count') or 0)}`",
        f"- Review required: `{int(result.get('review_required_count') or 0)}`",
        "",
    ]
    lines.extend(_counts_section("Validation Status", result.get("by_validation_status")))
    lines.extend(_counts_section("Post Import Next Action", result.get("post_import_next_action_counts")))
    return "\n".join(lines).rstrip() + "\n"


def _render_batch_codex_review_draft_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Transcript Semantic Correction Batch Codex Review Draft",
        "",
        f"- Created: `{result.get('created_at')}`",
        f"- Status: `{result.get('status')}`",
        f"- Reviews: `{int(result.get('review_count') or 0)}`",
        "",
    ]
    lines.extend(_counts_section("Review Status", result.get("by_review_status")))
    lines.extend(["| Candidate | Status | Confidence | Note |", "| --- | --- | ---: | --- |"])
    for row in result.get("reviews") or []:
        lines.append(
            "| `{candidate}` | `{status}` | {confidence:.2f} | {note} |".format(
                candidate=str(row.get("candidate_id") or ""),
                status=str(row.get("review_status") or ""),
                confidence=float(row.get("confidence") or 0.0),
                note=str(row.get("review_note") or "").replace("|", "/"),
            )
        )
    return "\n".join(lines).rstrip() + "\n"

def _repair_run_item(
    row: dict[str, Any],
    *,
    execute: bool,
    allow_closure: bool,
    allow_llm: bool,
    provider_config: dict[str, Any] | None,
    llm_limit: int,
    business_authorization_path: str | Path | None,
) -> dict[str, Any]:
    root = Path(str(row.get("bundle_dir") or "")).expanduser().resolve()
    action = str(row.get("action_key") or "none")
    base = {
        "bundle_dir": str(root),
        "action_key": action,
        "action_status": str(row.get("action_status") or ""),
        "action_kind": str(row.get("action_kind") or ""),
        "machine_action_available": bool(row.get("machine_action_available")),
        "human_review_required": bool(row.get("human_review_required")),
        "retry_command": str(row.get("retry_command") or ""),
    }
    if action == "none":
        return {**base, "run_status": "skipped_completed", "executed": False, "reason": "Already complete."}
    if not execute:
        return {**base, "run_status": "planned", "executed": False, "reason": "Preview only; pass execute_safe_actions to run local safe actions."}
    if action == "execute_llm_or_use_codex" and allow_llm and not provider_config and not business_authorization_path:
        return {**base, "run_status": "skipped_requires_provider_config", "executed": False, "reason": "Text LLM execution requires a configured route or business authorization."}
    if not row.get("machine_action_available") and action != "execute_llm_or_use_codex":
        return {**base, "run_status": "skipped_operator_required", "executed": False, "reason": str(row.get("reason") or "Requires Codex/LLM/human review.")}
    try:
        payload = _execute_safe_queue_action(root, action, allow_closure=allow_closure, allow_llm=allow_llm, provider_config=provider_config, llm_limit=llm_limit, business_authorization_path=business_authorization_path)
        return {**base, "run_status": "executed", "executed": True, "result_status": str(payload.get("status") or ""), "result_ok": bool(payload.get("ok", True)), "result": payload}
    except Exception as exc:  # noqa: BLE001
        return {**base, "run_status": "failed", "executed": False, "error": f"{type(exc).__name__}: {exc}"}


def _execute_safe_queue_action(
    root: Path,
    action: str,
    *,
    allow_closure: bool,
    allow_llm: bool,
    provider_config: dict[str, Any] | None,
    llm_limit: int,
    business_authorization_path: str | Path | None,
) -> dict[str, Any]:
    from .knowledge_note_export import export_knowledge_note
    from .transcript_semantic_summary_impact import transcript_semantic_summary_impact_report
    from .transcript_semantic_correction import (
        build_transcript_semantic_correction_codex_draft,
        build_transcript_semantic_correction_llm_draft,
        build_transcript_semantic_candidate_discovery_llm_draft,
        build_transcript_semantic_candidate_discovery_pack,
        import_transcript_semantic_candidate_suggestions,
        build_transcript_semantic_correction_pack,
        transcript_semantic_correction_closure,
        transcript_semantic_correction_impact_report,
        transcript_semantic_correction_readable_impact_report,
        validate_transcript_semantic_correction,
    )

    if action == "build_pack":
        return build_transcript_semantic_correction_pack(root, write=True)
    if action == "run_candidate_discovery":
        return build_transcript_semantic_candidate_discovery_pack(root, limit=40, write=True)
    if action in {"run_candidate_discovery_llm_preview", "retry_candidate_discovery_llm_or_manual_review"}:
        return build_transcript_semantic_candidate_discovery_llm_draft(root, execute=False, limit=40, write=True)
    if action == "import_candidate_suggestions":
        input_path = _first_existing(root, ["transcript-semantic-candidate-suggestions.codex.md", "transcript-semantic-candidate-suggestions.llm.md", "transcript-semantic-candidate-suggestions.llm.json"])
        return import_transcript_semantic_candidate_suggestions(root, input_json=input_path, write=True)
    if action in {"run_llm_draft_preview", "retry_llm_or_manual_review"}:
        return build_transcript_semantic_correction_llm_draft(root, execute=False, limit=int(llm_limit or 80), write=True)
    if action == "execute_llm_or_use_codex":
        if not allow_llm:
            return build_transcript_semantic_correction_codex_draft(root, write=True)
        if not provider_config and not business_authorization_path:
            return {"status": "skipped_requires_provider_config", "ok": False, "reason": "Pass a configured route or business authorization to execute text LLM review."}
        return build_transcript_semantic_correction_llm_draft(root, provider_config=provider_config, execute=True, limit=int(llm_limit or 80), business_authorization_path=business_authorization_path, write=True)
    if action == "validate_llm_result":
        input_path = _first_existing(root, ["transcript-semantic-correction-result.llm.md", "transcript-semantic-correction-result.llm.json"])
        return validate_transcript_semantic_correction(root, input_json=input_path, write=True)
    if action == "validate_result":
        input_path = _first_existing(root, ["transcript-semantic-correction-result.codex.md", "transcript-semantic-correction-result.codex.json"])
        return validate_transcript_semantic_correction(root, input_json=input_path, write=True)
    if action == "run_closure":
        if not allow_closure:
            return {"status": "skipped_requires_allow_closure", "ok": False, "reason": "Pass allow_closure=true to write source-arbitrated transcript."}
        input_path = _first_existing(root, ["transcript-semantic-correction-result.review.json", "transcript-semantic-correction-result.review.md", "transcript-semantic-correction-result.codex.md", "transcript-semantic-correction-result.llm.md", "transcript-semantic-correction-result.codex.json", "transcript-semantic-correction-result.llm.json"])
        closure = transcript_semantic_correction_closure(root, input_json=input_path, write=True)
        if not closure.get("ok"):
            return {"status": "closure_completed_without_refresh", "ok": False, "closure": closure, "reason": "Closure did not produce a usable corrected transcript."}
        export_result = export_knowledge_note(root, run_transcript_evidence_check=False, write=True)
        impact = transcript_semantic_correction_impact_report(root, write=True)
        readable = transcript_semantic_correction_readable_impact_report(root, write=True)
        summary_impact = transcript_semantic_summary_impact_report(root, write=True)
        summary_status = str(summary_impact.get("status") or "")
        summary_ok = summary_status in {"passed", "no_evaluable_replacements", "no_accepted_decisions"}
        return {
            "status": "closed_and_refreshed_exports",
            "ok": bool(impact.get("status") == "passed" and readable.get("status") == "passed" and summary_ok),
            "closure": closure,
            "export": export_result,
            "impact": impact,
            "readable_impact": readable,
            "summary_impact": summary_impact,
        }
    if action == "run_impact":
        return transcript_semantic_correction_impact_report(root, write=True)
    if action == "run_readable_impact":
        return transcript_semantic_correction_readable_impact_report(root, write=True)
    if action == "run_summary_impact":
        return transcript_semantic_summary_impact_report(root, write=True)
    if action == "refresh_summary_impact_or_review":
        export_result = export_knowledge_note(root, run_transcript_evidence_check=False, write=True)
        summary_impact = transcript_semantic_summary_impact_report(root, write=True)
        return {"status": "refreshed_summary_and_summary_impact", "ok": bool(summary_impact.get("status") == "passed"), "export": export_result, "summary_impact": summary_impact}
    if action == "refresh_exports_or_review":
        export_result = export_knowledge_note(root, run_transcript_evidence_check=False, write=True)
        impact = transcript_semantic_correction_impact_report(root, write=True)
        readable = transcript_semantic_correction_readable_impact_report(root, write=True)
        summary_impact = transcript_semantic_summary_impact_report(root, write=True)
        return {"status": "refreshed_exports_and_reports", "ok": bool(impact.get("status") == "passed" and readable.get("status") == "passed" and summary_impact.get("status") == "passed"), "export": export_result, "impact": impact, "readable_impact": readable, "summary_impact": summary_impact}
    return {"status": "skipped_unsupported_safe_action", "ok": False, "reason": f"Action {action} is not in the safe local action set."}


def _first_existing(root: Path, names: list[str]) -> Path:
    for name in names:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError("none of the expected semantic correction result files exist: " + ", ".join(names))


def _repair_run_summary(executions: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_action: dict[str, int] = {}
    for row in executions:
        by_status[str(row.get("run_status") or "unknown")] = by_status.get(str(row.get("run_status") or "unknown"), 0) + 1
        by_action[str(row.get("action_key") or "unknown")] = by_action.get(str(row.get("action_key") or "unknown"), 0) + 1
    return {
        "action_count": len(executions),
        "executed_count": sum(1 for row in executions if row.get("executed")),
        "planned_count": sum(1 for row in executions if row.get("run_status") == "planned"),
        "failed_count": sum(1 for row in executions if row.get("run_status") == "failed"),
        "operator_required_count": sum(1 for row in executions if str(row.get("run_status") or "") in {"skipped_operator_required", "skipped_requires_allow_llm", "skipped_requires_provider_config"}),
        "by_run_status": by_status,
        "by_action_key": by_action,
    }


def _repair_run_status(executions: list[dict[str, Any]], *, execute: bool) -> str:
    if not executions:
        return "no_actions_required"
    if not execute:
        return "planned"
    if any(row.get("run_status") == "failed" for row in executions):
        return "completed_with_errors"
    if any(str(row.get("run_status") or "") in {"skipped_operator_required", "skipped_requires_allow_llm", "skipped_requires_provider_config"} for row in executions):
        return "completed_with_operator_required"
    return "completed"


def _repair_run_ok(executions: list[dict[str, Any]], *, execute: bool) -> bool:
    if not execute:
        return True
    return not any(row.get("run_status") == "failed" for row in executions)


def _render_repair_run_markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    lines = [
        "# Transcript Semantic Correction Repair Run",
        "",
        f"- Created: `{result.get('created_at')}`",
        f"- Status: `{result.get('status')}`",
        f"- Execute safe actions: `{bool(result.get('execute_safe_actions'))}`",
        f"- Allow closure: `{bool(result.get('allow_closure'))}`",
        f"- Actions: `{int(summary.get('action_count') or 0)}`",
        f"- Executed: `{int(summary.get('executed_count') or 0)}`",
        f"- Planned: `{int(summary.get('planned_count') or 0)}`",
        f"- Failed: `{int(summary.get('failed_count') or 0)}`",
        f"- Operator required: `{int(summary.get('operator_required_count') or 0)}`",
        "",
    ]
    lines.extend(_counts_section("Run Status", summary.get("by_run_status")))
    lines.extend(_counts_section("Action Key", summary.get("by_action_key")))
    lines.extend(["## Executions", "", "| Bundle | Action | Run status | Result |", "| --- | --- | --- | --- |"])
    for row in result.get("executions") or []:
        lines.append(
            "| {bundle} | `{action}` | `{status}` | {result_status} |".format(
                bundle=str(row.get("bundle_dir") or "").replace("|", "/"),
                action=str(row.get("action_key") or ""),
                status=str(row.get("run_status") or ""),
                result_status=str(row.get("result_status") or row.get("error") or row.get("reason") or "").replace("|", "/"),
            )
        )
    lines.extend(["", "## Operator Boundary", "", "- Default is preview only.", "- Safe execution does not call ASR, vision, download, or modify raw sources.", "- Text LLM execution requires both `execute_safe_actions=true` and `allow_llm=true` plus runtime provider config.", "- Closure requires `allow_closure=true` because it writes corrected transcript sidecars."])
    return "\n".join(lines).rstrip() + "\n"
def _render_queue_markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    lines = [
        "# Transcript Semantic Correction Repair Queue",
        "",
        f"- Created: `{result.get('created_at')}`",
        f"- Status: `{result.get('status')}`",
        f"- OK: `{bool(result.get('ok'))}`",
        f"- Bundles: `{int(result.get('bundle_count') or 0)}` / target `{int(result.get('target_bundle_count') or 0)}`",
        f"- Action required: `{int(summary.get('action_required_count') or 0)}`",
        f"- Machine actions: `{int(summary.get('machine_action_available_count') or 0)}`",
        f"- Human review: `{int(summary.get('human_review_required_count') or 0)}`",
        "",
    ]
    lines.extend(_counts_section("Action Key", summary.get("by_action_key")))
    lines.extend(_counts_section("Action Status", summary.get("by_action_status")))
    lines.extend(_counts_section("Action Kind", summary.get("by_action_kind")))
    lines.extend(["## Queue", "", "| Bundle | Action | Status | Kind | Machine | Human | Retry |", "| --- | --- | --- | --- | --- | --- | --- |"])
    for row in result.get("items") or []:
        lines.append(
            "| {bundle} | `{action}` | `{status}` | `{kind}` | `{machine}` | `{human}` | {retry} |".format(
                bundle=str(row.get("bundle_dir") or "").replace("|", "/"),
                action=str(row.get("action_key") or ""),
                status=str(row.get("action_status") or ""),
                kind=str(row.get("action_kind") or ""),
                machine=bool(row.get("machine_action_available")),
                human=bool(row.get("human_review_required")),
                retry=str(row.get("retry_command") or "").replace("|", "/"),
            )
        )
    lines.extend(["", "## Operator Boundary", "", "- Preview-only queue; does not execute actions.", "- Cloud LLM, closure, ASR, vision, and download remain explicit gates."])
    return "\n".join(lines).rstrip() + "\n"
def _default_output_dir(batch_input: str | Path) -> Path:
    path = Path(batch_input).expanduser()
    if path.is_dir():
        return path.resolve() if not (path / "manifest.json").exists() else path.resolve() / "exports"
    return path.parent.resolve() if path.parent else Path.cwd()


def _render_single_acceptance_markdown(result: dict[str, Any]) -> str:
    row = result.get("item") if isinstance(result.get("item"), dict) else {}
    evidence = row.get("evidence_files") if isinstance(row.get("evidence_files"), dict) else {}
    lines = [
        "# Transcript Semantic Correction Acceptance",
        "",
        f"- Created: `{result.get('created_at')}`",
        f"- Bundle: `{result.get('bundle_dir')}`",
        f"- Status: `{result.get('status')}`",
        f"- OK: `{bool(result.get('ok'))}`",
        f"- Acceptance state: `{result.get('acceptance_state')}`",
        f"- Semantic status: `{result.get('semantic_status')}`",
        f"- Next action: `{result.get('next_action_key')}`",
        "",
        "## Canonical Export Integrity",
        "",
        f"- Status: `{(result.get('canonical_transcript_integrity') or {}).get('status', 'unknown')}`",
        f"- Passed: `{bool((result.get('canonical_transcript_integrity') or {}).get('passed'))}`",
        f"- Canonical SHA-256: `{(result.get('canonical_transcript_integrity') or {}).get('canonical_sha256', '')}`",
        "",
        "## Counts",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Candidates | {int(row.get('candidate_count') or 0)} |",
        f"| Accepted decisions | {int(row.get('accepted_decision_count') or 0)} |",
        f"| Review required | {int(row.get('review_required_count') or 0)} |",
        f"| Final residual errors | {int(row.get('final_residual_error_total') or 0)} |",
        f"| Summary residual originals | {int(row.get('summary_impact_residual_total') or 0)} |",
        f"| Summary corrected hits | {int(row.get('summary_impact_corrected_hit_total') or 0)} |",
        "",
    ]
    lines.extend(_counts_section("Candidate Type", row.get("candidate_type_counts")))
    lines.extend(_counts_section("Risk Level", row.get("risk_level_counts")))
    lines.extend(_counts_section("Evidence Source", row.get("evidence_source_counts")))
    if evidence:
        lines.extend(["## Evidence Files", "", "| Artifact | Exists |", "| --- | --- |"])
        for key, exists in sorted(evidence.items()):
            lines.append(f"| `{key}` | `{bool(exists)}` |")
        lines.append("")
    if result.get("next_actions"):
        lines.extend(["## Next Actions", ""])
        for action in result.get("next_actions") or []:
            lines.append(f"- {action}")
        lines.append("")
    lines.extend([
        "## Operator Boundary",
        "",
        "- Read-only report; does not run ASR, vision, download, validation, closure, export, or cloud/API calls.",
        "- `accepted` only proves transcript semantic correction acceptance; it is not publication permission.",
    ])
    return "\n".join(lines).rstrip() + "\n"

def _render_markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    lines = [
        "# Transcript Semantic Correction Batch Acceptance",
        "",
        f"- Created: `{result.get('created_at')}`",
        f"- Status: `{result.get('status')}`",
        f"- OK: `{bool(result.get('ok'))}`",
        f"- Bundles: `{int(result.get('bundle_count') or 0)}` / target `{int(result.get('target_bundle_count') or 0)}`",
        f"- Accepted: `{int(summary.get('accepted_count') or 0)}`",
        f"- Not accepted: `{int(summary.get('not_accepted_count') or 0)}`",
        f"- Candidates: `{int(summary.get('candidate_count') or 0)}`",
        f"- Review required: `{int(summary.get('review_required_count') or 0)}`",
        f"- Residual errors: `{int(summary.get('final_residual_error_total') or 0)}`",
        "",
    ]
    lines.extend(_counts_section("Acceptance State", summary.get("by_acceptance_state")))
    lines.extend(_counts_section("Semantic Status", summary.get("by_semantic_status")))
    lines.extend(_counts_section("Next Action", summary.get("by_next_action")))
    lines.extend(_counts_section("Candidate Type", summary.get("by_candidate_type")))
    lines.extend(_counts_section("Risk Level", summary.get("by_risk_level")))
    lines.extend(["## Bundles", "", "| Bundle | Acceptance | Semantic status | Candidates | Accepted | Review | Residual | Next action |", "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"])
    for row in result.get("items") or []:
        lines.append(
            "| {bundle} | `{acceptance}` | `{status}` | {candidates} | {accepted} | {review} | {residual} | `{next_action}` |".format(
                bundle=str(row.get("bundle_dir") or "").replace("|", "/"),
                acceptance=str(row.get("acceptance_state") or ""),
                status=str(row.get("semantic_status") or ""),
                candidates=int(row.get("candidate_count") or 0),
                accepted=int(row.get("accepted_decision_count") or 0),
                review=int(row.get("review_required_count") or 0),
                residual=int(row.get("final_residual_error_total") or 0),
                next_action=str(row.get("next_action_key") or "none"),
            )
        )
    if result.get("next_actions"):
        lines.extend(["", "## Next Actions", ""])
        for action in result.get("next_actions") or []:
            lines.append(f"- {action}")
    lines.extend(["", "## Operator Boundary", "", "- Read-only report; does not run ASR, vision, download, validation, closure, export, or cloud/API calls."])
    return "\n".join(lines).rstrip() + "\n"


def _counts_section(title: str, values: Any) -> list[str]:
    if not isinstance(values, dict) or not values:
        return []
    lines = [f"## {title}", "", "| Key | Count |", "| --- | ---: |"]
    for key, value in sorted(values.items()):
        lines.append(f"| `{key}` | {int(value or 0)} |")
    lines.append("")
    return lines


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
