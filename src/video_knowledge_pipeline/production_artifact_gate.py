from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .content_profile import resolve_content_profile
from .models import now_iso
from .media_applicability import resolve_media_applicability
from .source_review_lineage import (
    discover_source_review_lineage,
    load_bound_source_review_lineage,
    validate_bound_source_review_lineage,
)
from .storage import read_json, write_json


SCHEMA = "video_knowledge_pipeline.production_artifact_gate.v1"


def transcript_completeness_status(bundle_dir: str | Path) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    path = root / "transcript-quality-gate.json"
    if not path.is_file():
        return {
            "passed": True,
            "status": "not_available_legacy_compatible",
            "path": str(path),
            "detail": "transcript quality artifact is absent; legacy compatibility is retained outside strict profiles",
        }
    try:
        value = read_json(path)
    except Exception as exc:
        return {"passed": False, "status": "invalid", "path": str(path), "detail": f"unreadable: {exc}"}
    if not isinstance(value, dict):
        return {"passed": False, "status": "invalid", "path": str(path), "detail": "not a JSON object"}
    status = str(value.get("status") or "").strip().lower()
    completeness = value.get("source_completeness") if isinstance(value.get("source_completeness"), dict) else {}
    completeness_status = str(completeness.get("status") or "").strip().lower()
    verified = completeness.get("speech_completeness_verified")
    failed = status == "failed" or value.get("ok") is False or int(value.get("fail_count") or 0) > 0 or completeness_status == "failed"
    if failed:
        return {
            "passed": False,
            "status": "failed",
            "path": str(path),
            "transcript_quality_status": status,
            "source_completeness_status": completeness_status,
            "speech_completeness_verified": verified,
            "detail": f"status={status}; source_completeness={completeness_status}; verified={verified}",
        }
    if completeness.get("applicable") is True and verified is not True:
        return {
            "passed": False,
            "status": "unverified",
            "path": str(path),
            "speech_completeness_verified": verified,
            "detail": "speech completeness is not independently verified",
        }
    return {
        "passed": True,
        "status": "verified" if verified is True else status or "passed",
        "path": str(path),
        "speech_completeness_verified": verified,
        "detail": "no unresolved speech-completeness gap is exposed",
    }


def evaluate_production_artifact_gate(
    bundle_dir: str | Path,
    *,
    artifact_kind: str = "smart_summary",
    search_roots: Iterable[str | Path] | None = None,
    discover_prior_reviews: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Single formal-output gate; a failed check cannot be reduced to a warning."""

    root = Path(bundle_dir).expanduser().resolve()
    manifest = _mapping(root / "manifest.json")
    profile = resolve_content_profile(root, manifest=manifest)
    requirements = profile["requirements"]
    transcript = transcript_completeness_status(root)
    speaker = _speaker_status(root, requirements)
    semantic_fact = _human_review_status(
        root,
        names=("medical-insurance-fact-review.json", "fact-review.json"),
        required=bool(requirements["semantic_fact_review_required"]),
        missing_status="required_human_fact_review_missing",
    )
    privacy = _human_review_status(
        root,
        names=("privacy-review.json",),
        required=bool(requirements["privacy_review_required"]),
        missing_status="required_privacy_review_missing",
    )
    prior = load_bound_source_review_lineage(root)
    if discover_prior_reviews and not prior.get("applied"):
        prior = discover_source_review_lineage(root, search_roots=search_roots, apply=False, write=write)
    lineage_validation = (
        validate_bound_source_review_lineage(root)
        if prior.get("applied")
        else {"status": "not_bound", "passed": False, "failures": []}
    )
    prior_review_blocked = (
        (bool(prior.get("candidate_count")) and not bool(prior.get("applied")))
        or (bool(prior.get("applied")) and not bool(lineage_validation.get("passed")))
    )
    strict_profile = profile["profile_id"] == "medical-insurance-interview-v1"
    transcript_passed = bool(transcript.get("passed")) and not (
        strict_profile and transcript.get("status") == "not_available_legacy_compatible"
    )
    blocking_reasons: list[str] = []
    if not transcript_passed:
        blocking_reasons.append(f"transcript_quality:{transcript.get('status')}")
    if not speaker["passed"]:
        blocking_reasons.append(f"speaker_review:{speaker.get('status')}")
    if not semantic_fact["passed"]:
        blocking_reasons.append(f"semantic_fact:{semantic_fact.get('status')}")
    if not privacy["passed"]:
        blocking_reasons.append(f"privacy:{privacy.get('status')}")
    if prior_review_blocked:
        blocking_reasons.append(
            "prior_review_lineage_invalid"
            if prior.get("applied")
            else "prior_human_review_available_not_bound"
        )
    allowed = not blocking_reasons
    publication_approval = _publication_approval(root)
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "artifact_kind": artifact_kind,
        "media_applicability": resolve_media_applicability(root, manifest),
        "status": "formal_generation_allowed" if allowed else "blocked_review_required",
        "ok": allowed,
        "formal_generation_allowed": allowed,
        "machine_draft_allowed": True,
        "artifact_state": "human-reviewed-input" if allowed else "review-required",
        "execution_status": "ready" if allowed else "blocked_by_quality_gate",
        "transcript_quality": transcript,
        "speaker_review_status": speaker,
        "semantic_fact_status": semantic_fact,
        "privacy_status": privacy,
        "publication_readiness": {
            "status": "publication_approved" if publication_approval else "human_approval_required",
            "passed": publication_approval,
        },
        "content_profile": profile,
        "source_review_lineage": {
            "status": prior.get("status") or "not_checked",
            "candidate_count": int(prior.get("candidate_count") or 0),
            "applied": bool(prior.get("applied")),
            "validation_status": lineage_validation.get("status"),
            "validation_failures": lineage_validation.get("failures") or [],
            "path": str(root / "source-review-lineage.json"),
        },
        "blocking_reasons": blocking_reasons,
        "operator_boundary": {
            "formal_filename_must_not_be_written_when_blocked": True,
            "machine_draft_requires_visible_watermark": True,
            "generation_success_does_not_equal_publication_approval": True,
            "does_not_call_models_or_modify_evidence": True,
        },
        "updated_at": now_iso(),
    }
    if write:
        write_json(root / "production-artifact-gate.json", result)
        (root / "production-artifact-gate.md").write_text(_render_markdown(result), encoding="utf-8")
        if manifest:
            manifest["production_artifact_gate_json"] = "production-artifact-gate.json"
            manifest["production_artifact_gate_status"] = result["status"]
            manifest["artifact_state"] = result["artifact_state"]
            write_json(root / "manifest.json", manifest)
    return result


def _speaker_status(root: Path, requirements: dict[str, Any]) -> dict[str, Any]:
    gate = _mapping(root / "transcript-quality-gate.json")
    diarization = gate.get("speaker_diarization") if isinstance(gate.get("speaker_diarization"), dict) else {}
    required = bool(requirements["speaker_diarization_required"])
    diarization_passed = bool(diarization.get("passed")) if required else True
    if not required:
        return {"status": "not_required", "passed": True, "diarization": diarization}
    if not diarization_passed:
        return {"status": "speaker_diarization_required", "passed": False, "diarization": diarization}
    if not requirements["speaker_role_review_required"]:
        return {"status": "diarization_passed", "passed": True, "diarization": diarization}
    review_path = _review_path(root, "speaker_review", ("speaker-review.json",))
    review = _mapping(review_path) if review_path else {}
    if not review:
        return {"status": "human_speaker_role_review_missing", "passed": False, "diarization": diarization}
    status = str(review.get("status") or "").lower()
    mappings = review.get("speaker_mappings") or review.get("mappings") or []
    passed = status in {"human_confirmed", "human_confirmed_roles", "complete", "completed"} and bool(mappings)
    return {"status": status or "human_speaker_role_review_missing", "passed": passed, "diarization": diarization}


def _human_review_status(root: Path, *, names: tuple[str, ...], required: bool, missing_status: str) -> dict[str, Any]:
    if not required:
        return {"status": "not_required", "passed": True, "path": ""}
    kind = "privacy_review" if names == ("privacy-review.json",) else "semantic_fact_review"
    path = _review_path(root, kind, names)
    if path is None:
        return {"status": missing_status, "passed": False, "path": str(root / names[0])}
    payload = _mapping(path)
    status = str(payload.get("status") or "").lower()
    passed = payload.get("human_confirmed") is True or status in {"human_confirmed", "approved", "complete", "completed"}
    return {"status": status or "invalid", "passed": passed, "path": str(path)}


def _review_path(root: Path, kind: str, names: tuple[str, ...]) -> Path | None:
    local = next((root / name for name in names if (root / name).is_file()), None)
    if local is not None and _review_payload_is_human_confirmed(_mapping(local)):
        return local
    manifest = _mapping(root / "manifest.json")
    inherited = manifest.get("inherited_review_artifacts") if isinstance(manifest.get("inherited_review_artifacts"), list) else []
    for row in inherited:
        if not isinstance(row, dict) or row.get("kind") != kind or row.get("human_confirmed") is not True:
            continue
        path = Path(str(row.get("path") or "")).expanduser()
        if path.is_file():
            return path
    return local


def _review_payload_is_human_confirmed(payload: dict[str, Any]) -> bool:
    status = str(payload.get("status") or "").lower()
    return payload.get("human_confirmed") is True or status in {
        "human_confirmed",
        "human_confirmed_roles",
        "approved",
        "complete",
        "completed",
    }


def _publication_approval(root: Path) -> bool:
    payload = _mapping(root / "publication-approval.json")
    return payload.get("human_confirmed") is True and str(payload.get("status") or "").lower() in {"approved", "human_confirmed"}


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Production Artifact Gate",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Artifact state: `{result.get('artifact_state')}`",
        f"- Formal generation allowed: `{result.get('formal_generation_allowed')}`",
        f"- Transcript quality: `{result.get('transcript_quality', {}).get('status')}`",
        f"- Speaker review: `{result.get('speaker_review_status', {}).get('status')}`",
        f"- Semantic fact review: `{result.get('semantic_fact_status', {}).get('status')}`",
        f"- Privacy review: `{result.get('privacy_status', {}).get('status')}`",
        f"- Publication: `{result.get('publication_readiness', {}).get('status')}`",
        "",
        "## Blocking reasons",
        "",
    ]
    reasons = result.get("blocking_reasons") or []
    lines.extend(f"- `{reason}`" for reason in reasons)
    if not reasons:
        lines.append("- None")
    return "\n".join(lines).rstrip() + "\n"


def _mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}
