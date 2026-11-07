from __future__ import annotations

from pathlib import Path
from typing import Any

from .asr_vad_chunking import read_vad_intervals
from .file_hash import sha256_file
from .interval_coverage import interval_coverage
from .models import now_iso
from .silero_vad_candidate import SCHEMA as SILERO_SCHEMA
from .storage import read_json, write_json


SCHEMA = "video_knowledge_pipeline.asr_vad_independent_crosscheck.v1"


def crosscheck_asr_vad_with_independent_candidate(
    authoritative_vad_path: str | Path,
    candidate_vad_path: str | Path,
    *,
    activity_audit_path: str | Path | None = None,
    output_path: str | Path | None = None,
    minimum_gap_seconds: float = 2.0,
    write: bool = True,
) -> dict[str, Any]:
    """Find Silero speech intervals not covered by the authoritative FunASR VAD."""

    authoritative_path = Path(authoritative_vad_path).expanduser().resolve()
    candidate_path = Path(candidate_vad_path).expanduser().resolve()
    authoritative = _read_object(authoritative_path, "authoritative VAD")
    candidate = _read_object(candidate_path, "candidate VAD")
    minimum_gap = float(minimum_gap_seconds)
    if minimum_gap < 0:
        raise ValueError("minimum_gap_seconds must not be negative")
    _validate_pair(authoritative_path, authoritative, candidate)

    authoritative_intervals = [
        (float(row["start"]), float(row["end"]))
        for row in read_vad_intervals(authoritative_path)
    ]
    candidate_intervals = (
        [
            (float(row["start"]), float(row["end"]))
            for row in read_vad_intervals(candidate_path)
        ]
        if candidate.get("segments")
        else []
    )
    uncovered = interval_coverage(
        candidate_intervals,
        authoritative_intervals,
        minimum_gap_seconds=minimum_gap,
    )
    activity = _validated_activity_audit(
        activity_audit_path,
        authoritative_vad_sha256=sha256_file(authoritative_path),
        source_sha256=str(candidate["source_media"]["sha256"]),
    )
    activity_intervals = [
        (float(row["start"]), float(row["end"]))
        for row in (activity.get("audio_probe") or {}).get("activity_intervals") or []
        if isinstance(row, dict)
    ]
    candidates: list[dict[str, Any]] = []
    for position, gap in enumerate(uncovered["gaps"], start=1):
        start = float(gap["start"])
        end = float(gap["end"])
        activity_support = (
            interval_coverage(
                [(start, end)],
                activity_intervals,
                minimum_gap_seconds=0.0,
            )["coverage_ratio"]
            if activity_intervals
            else None
        )
        candidates.append(
            {
                "candidate_id": f"independent-vad-gap-{position:04d}",
                **gap,
                "reason": "independent_silero_speech_without_authoritative_vad_coverage",
                "evidence_type": "candidate_speech_activity",
                "independent_model_support": True,
                "audio_activity_support_ratio": activity_support,
                "candidate_only": True,
                "automatic_acceptance_allowed": False,
                "needs_targeted_asr_or_human_confirmation": True,
            }
        )
    target = (
        Path(output_path).expanduser().resolve()
        if output_path
        else candidate_path.with_name("asr-vad-independent-crosscheck.json")
    )
    result = {
        "schema": SCHEMA,
        "ok": True,
        "status": "review_required" if candidates else "passed",
        "authoritative_vad": {
            "path": str(authoritative_path),
            "sha256": sha256_file(authoritative_path),
            "model": authoritative.get("model"),
            "model_revision": authoritative.get("model_revision"),
        },
        "candidate_vad": {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
            "model": candidate["upstream"]["model"],
            "installed_version": candidate["upstream"]["installed_version"],
        },
        "activity_audit": {
            "path": str(Path(activity_audit_path).expanduser().resolve())
            if activity_audit_path
            else "",
            "used": bool(activity),
        },
        "candidate_vs_authoritative_coverage": uncovered,
        "candidate_gap_count": len(candidates),
        "candidate_gaps": candidates,
        "decision_boundary": {
            "independent_vad_is_authoritative": False,
            "candidate_only": True,
            "authoritative_vad_modified": False,
            "chunk_manifest_modified": False,
            "canonical_transcript_modified": False,
            "automatic_remote_retry": False,
            "automatic_fallback": False,
            "network_call": False,
        },
        "recommended_action": (
            "run targeted ASR or human review only for candidate gaps"
            if candidates
            else "retain the authoritative VAD for this sample"
        ),
        "output_path": str(target),
        "updated_at": now_iso(),
    }
    if write:
        write_json(target, result)
    return result


def _validate_pair(
    authoritative_path: Path,
    authoritative: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    if authoritative.get("candidate_only") is True:
        raise ValueError("authoritative VAD cannot be candidate-only")
    if str(candidate.get("schema") or "") != SILERO_SCHEMA:
        raise ValueError("candidate VAD schema is not supported")
    if candidate.get("candidate_only") is not True:
        raise ValueError("independent VAD must be explicitly candidate-only")
    if str(candidate.get("status") or "") != "completed":
        raise ValueError("independent VAD must be completed")
    source = candidate.get("source_media")
    if not isinstance(source, dict):
        raise ValueError("independent VAD source_media is missing")
    authoritative_media = Path(str(authoritative.get("input") or "")).expanduser().resolve()
    candidate_media = Path(str(source.get("path") or "")).expanduser().resolve()
    if authoritative_media != candidate_media:
        raise ValueError("VAD evidence does not reference the same media path")
    if not authoritative_media.is_file():
        raise FileNotFoundError(f"VAD source media not found: {authoritative_media}")
    current_sha = sha256_file(authoritative_media)
    if str(source.get("sha256") or "").lower() != current_sha.lower():
        raise ValueError("independent VAD source media hash is stale")
    if not authoritative_path.is_file():
        raise FileNotFoundError(f"authoritative VAD not found: {authoritative_path}")


def _validated_activity_audit(
    activity_audit_path: str | Path | None,
    *,
    authoritative_vad_sha256: str,
    source_sha256: str,
) -> dict[str, Any]:
    if not activity_audit_path:
        return {}
    path = Path(activity_audit_path).expanduser().resolve()
    audit = _read_object(path, "activity audit")
    if str(audit.get("schema") or "") != "video_knowledge_pipeline.asr_vad_activity_audit.v1":
        raise ValueError("activity audit schema is not supported")
    if str(audit.get("vad_sha256") or "").lower() != authoritative_vad_sha256.lower():
        raise ValueError("activity audit does not bind the authoritative VAD hash")
    audit_source = audit.get("source_media")
    if not isinstance(audit_source, dict):
        raise ValueError("activity audit source_media is missing")
    if str(audit_source.get("sha256") or "").lower() != source_sha256.lower():
        raise ValueError("activity audit does not bind the source media hash")
    if str(audit.get("status") or "") not in {"passed", "review_required"}:
        raise ValueError("activity audit must be completed")
    return audit


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload
