from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifact_validation import sha256_file
from .storage import read_json


DISCOVERY_PACK_SCHEMA = (
    "video_knowledge_pipeline.transcript_semantic_candidate_discovery.v1"
)
DISCOVERY_SCAN_RECEIPT_NAME = (
    "transcript-semantic-candidate-discovery-scan-receipt.json"
)
DISCOVERY_SCAN_RECEIPT_SCHEMA = (
    "video_knowledge_pipeline.transcript_semantic_candidate_discovery_scan_receipt.v1"
)

_FORMAL_ADJUDICATION_ARRAYS = (
    "decisions",
    "accepted_decisions",
    "arbitrated_no_change_decisions",
    "rejected_decisions",
    "review_required_items",
    "review_required_decisions",
    "review_rows",
)


def semantic_discovery_scan_state(bundle_dir: str | Path) -> dict[str, Any]:
    """Project paged discovery from explicit, hash-bound completion proof.

    A prepared discovery pack is not completion evidence. Completion may come
    from either the bundle-local v1 scan receipt or the existing Autopilot job
    state. The bundle-local receipt contract is intentionally reader-only here:
    no writer is inferred from legacy manifest summaries or import artifacts.
    """

    root = Path(bundle_dir).expanduser().resolve()
    pack_path = root / "transcript-semantic-candidate-discovery-pack.json"
    if not pack_path.is_file():
        residue = _discovery_residue_exists(root)
        state = _empty_discovery_state("manifest_residue" if residue else "missing")
        if residue:
            state.update(
                {
                    "hash_drift": True,
                    "identity_valid": False,
                    "identity_issues": [
                        "discovery_pack_missing_with_residual_artifacts"
                    ],
                }
            )
        return state

    pack = _read_json_object(pack_path)
    if not pack:
        state = _empty_discovery_state("invalid")
        state.update(
            {
                "discovery_pack_sha256": sha256_file(pack_path),
                "hash_drift": True,
                "identity_valid": False,
                "identity_issues": ["discovery_pack_invalid"],
            }
        )
        return state

    pack_sha256 = sha256_file(pack_path)
    cursor = max(0, int(pack.get("cursor") or 0))
    pack_coverage = (
        pack.get("coverage") if isinstance(pack.get("coverage"), dict) else {}
    )
    transcript_path = _resolve_bundle_path(root, pack.get("transcript_path"))
    recorded_transcript_sha256 = str(pack.get("transcript_sha256") or "")
    current_transcript_sha256 = (
        sha256_file(transcript_path)
        if transcript_path is not None and transcript_path.is_file()
        else ""
    )
    pack_identity_issues = _discovery_pack_identity_issues(
        root,
        pack,
        transcript_path=transcript_path,
        current_transcript_sha256=current_transcript_sha256,
    )
    if pack_identity_issues:
        return {
            **_empty_discovery_state("pack_identity_invalid"),
            "cursor": cursor,
            "next_cursor": 0,
            "coverage": pack_coverage,
            "hash_drift": True,
            "transcript_sha256": recorded_transcript_sha256,
            "current_transcript_sha256": current_transcript_sha256,
            "discovery_pack_sha256": pack_sha256,
            "identity_valid": False,
            "identity_issues": pack_identity_issues,
        }

    receipt = _bundle_scan_receipt(
        root,
        pack,
        pack_sha256=pack_sha256,
        transcript_sha256=recorded_transcript_sha256,
    )
    job = _matching_autopilot_discovery_state(
        root,
        pack_sha256=pack_sha256,
        transcript_sha256=recorded_transcript_sha256,
    )

    completion_source = ""
    proof: dict[str, Any] | None = None
    proof_path = ""
    if receipt["valid"]:
        completion_source = "bundle_scan_receipt"
        proof = receipt["value"]
        proof_path = str(receipt["path"])
    elif job["valid"]:
        completion_source = "autopilot_job"
        proof = job["value"]
        proof_path = str(job["path"])

    proof_coverage = (
        proof.get("coverage")
        if isinstance(proof, dict) and isinstance(proof.get("coverage"), dict)
        else {}
    )
    coverage = proof_coverage or pack_coverage
    retained_count, covered_count, coverage_valid = _coverage_counts(coverage)
    coverage_complete = coverage_valid and covered_count >= retained_count
    coverage_matches_pack = _coverage_matches(coverage, pack_coverage)
    complete = bool(
        proof is not None
        and pack.get("complete") is True
        and proof.get("complete") is True
        and pack.get("next_cursor") is None
        and proof.get("next_cursor") is None
        and coverage_complete
        and coverage_matches_pack
    )

    if complete:
        next_cursor: int | None = None
        status = "complete"
    elif isinstance(proof, dict):
        raw_next_cursor = proof.get("next_cursor")
        next_cursor = int(raw_next_cursor) if raw_next_cursor is not None else cursor
        status = "page_imported"
    else:
        next_cursor = cursor
        status = (
            "receipt_invalid"
            if receipt["status"] == "invalid"
            else "page_not_imported"
        )

    identity_issues = list(receipt["issues"])
    hash_drift = False
    if (
        not complete
        and pack.get("complete") is True
        and retained_count == 0
        and covered_count == 0
    ):
        # The legacy status fallback treated an empty manifest/pack summary as
        # completion. Missing completion identity must remain fail-closed.
        status = "empty_completion_receipt_missing"
        hash_drift = True
        identity_issues.append("empty_completion_receipt_missing")

    return {
        "status": status,
        "complete": complete,
        "cursor": cursor,
        "next_cursor": next_cursor,
        "coverage": coverage,
        "page_imported": proof is not None,
        "hash_drift": hash_drift,
        "transcript_sha256": recorded_transcript_sha256,
        "current_transcript_sha256": current_transcript_sha256,
        "source_pack_sha256": receipt["source_pack_sha256"],
        "discovery_pack_sha256": pack_sha256,
        "completion_source": completion_source,
        "job_path": proof_path if completion_source == "autopilot_job" else "",
        "receipt_path": str(receipt["path"]),
        "receipt_sha256": receipt["sha256"],
        "receipt_status": receipt["status"],
        "identity_valid": not identity_issues,
        "identity_issues": identity_issues,
    }


def semantic_candidate_resolution(bundle_dir: str | Path) -> dict[str, Any]:
    """Count candidates that have entered any formal adjudication outcome."""

    root = Path(bundle_dir).expanduser().resolve()
    pack_path = root / "transcript-semantic-correction-pack.json"
    pack = _read_json_object(pack_path)
    candidates = [
        row
        for row in (pack.get("candidates") if isinstance(pack, dict) else []) or []
        if isinstance(row, dict) and str(row.get("candidate_id") or "")
    ]
    all_ids = {str(row["candidate_id"]) for row in candidates}
    source_sha256 = sha256_file(pack_path) if pack_path.is_file() else ""
    adjudicated: set[str] = set()
    source_counts: dict[str, int] = {}

    validation = _read_json_object(
        root / "transcript-semantic-correction-validation.json"
    )
    validation_pack_sha256 = str(validation.get("pack_sha256") or "")
    validation_current = not validation_pack_sha256 or (
        bool(source_sha256) and validation_pack_sha256 == source_sha256
    )
    if validation_current:
        for key in _FORMAL_ADJUDICATION_ARRAYS:
            ids = _candidate_ids(validation.get(key))
            if ids:
                adjudicated.update(ids)
                source_counts[f"validation.{key}"] = len(ids)

    ledger = _read_json_object(
        root / "transcript-semantic-correction-decision-ledger.json"
    )
    ledger_ids: set[str] = set()
    for row in ledger.get("decisions") or []:
        if not isinstance(row, dict):
            continue
        recorded_sha256 = str(row.get("validation_pack_sha256") or "")
        if recorded_sha256 and recorded_sha256 != source_sha256:
            continue
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id:
            ledger_ids.add(candidate_id)
    if ledger_ids:
        adjudicated.update(ledger_ids)
        source_counts["decision_ledger.decisions"] = len(ledger_ids)

    for job_path in _recent_autopilot_jobs(root):
        job = _read_json_object(job_path)
        if (
            not job
            or str(job.get("semantic_source_pack_sha256") or "") != source_sha256
        ):
            continue
        job_ids = {
            str(value)
            for value in job.get("semantic_adjudicated_candidate_ids") or []
            if str(value)
        }
        if job_ids:
            adjudicated.update(job_ids)
            source_counts["autopilot.semantic_adjudicated_candidate_ids"] = len(
                job_ids
            )

    adjudicated &= all_ids
    unresolved = all_ids - adjudicated
    return {
        "candidate_count": len(all_ids),
        "adjudicated_candidate_count": len(adjudicated),
        "unadjudicated_candidate_count": len(unresolved),
        "adjudicated_candidate_ids": sorted(adjudicated),
        "unadjudicated_candidate_ids": sorted(unresolved),
        "adjudication_source_counts": source_counts,
        "source_pack_sha256": source_sha256,
        "validation_pack_sha256": validation_pack_sha256,
        "validation_current": validation_current,
    }


def _bundle_scan_receipt(
    root: Path,
    pack: dict[str, Any],
    *,
    pack_sha256: str,
    transcript_sha256: str,
) -> dict[str, Any]:
    path = root / DISCOVERY_SCAN_RECEIPT_NAME
    source_pack_path = root / "transcript-semantic-correction-pack.json"
    source_pack_sha256 = (
        sha256_file(source_pack_path) if source_pack_path.is_file() else ""
    )
    if not path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "value": {},
            "path": path,
            "sha256": "",
            "source_pack_sha256": source_pack_sha256,
            "issues": [],
        }
    value = _read_json_object(path)
    issues: list[str] = []
    if not value:
        issues.append("scan_receipt_invalid")
    if str(value.get("schema") or "") != DISCOVERY_SCAN_RECEIPT_SCHEMA:
        issues.append("receipt_schema_mismatch")
    if str(value.get("status") or "") != "completed":
        issues.append("receipt_status_not_completed")
    if value.get("complete") is not True:
        issues.append("receipt_complete_not_true")
    if not str(value.get("discovery_pack_sha256") or ""):
        issues.append("receipt_discovery_pack_sha256_missing")
    elif str(value.get("discovery_pack_sha256")) != pack_sha256:
        issues.append("receipt_discovery_pack_sha256_mismatch")
    if not str(value.get("transcript_sha256") or ""):
        issues.append("receipt_transcript_sha256_missing")
    elif str(value.get("transcript_sha256")) != transcript_sha256:
        issues.append("receipt_transcript_sha256_mismatch")
    if not str(value.get("source_pack_sha256") or ""):
        issues.append("receipt_source_pack_sha256_missing")
    elif str(value.get("source_pack_sha256")) != source_pack_sha256:
        issues.append("receipt_source_pack_sha256_mismatch")
    if int(value.get("cursor") or 0) != int(pack.get("cursor") or 0):
        issues.append("receipt_cursor_mismatch")
    if value.get("next_cursor") is not None:
        issues.append("receipt_next_cursor_not_terminal")
    receipt_coverage = (
        value.get("coverage") if isinstance(value.get("coverage"), dict) else {}
    )
    if not _coverage_matches(receipt_coverage, pack.get("coverage")):
        issues.append("receipt_coverage_mismatch")
    batches = value.get("batches") if isinstance(value.get("batches"), list) else []
    if not batches or not isinstance(batches[-1], dict):
        issues.append("receipt_batches_missing")
    else:
        final_batch = batches[-1]
        if str(final_batch.get("discovery_pack_sha256") or "") != pack_sha256:
            issues.append("receipt_final_batch_hash_mismatch")
        if int(final_batch.get("cursor") or 0) != int(pack.get("cursor") or 0):
            issues.append("receipt_final_batch_cursor_mismatch")
        if final_batch.get("next_cursor") is not None:
            issues.append("receipt_final_batch_not_terminal")
        if final_batch.get("complete") is not True:
            issues.append("receipt_final_batch_incomplete")
    if pack.get("complete") is not True or pack.get("next_cursor") is not None:
        issues.append("discovery_pack_not_terminal")
    _, _, coverage_valid = _coverage_counts(receipt_coverage)
    if not coverage_valid:
        issues.append("receipt_coverage_invalid")
    return {
        "status": "valid" if not issues else "invalid",
        "valid": not issues,
        "value": value,
        "path": path,
        "sha256": sha256_file(path),
        "source_pack_sha256": source_pack_sha256,
        "issues": issues,
    }


def _matching_autopilot_discovery_state(
    root: Path,
    *,
    pack_sha256: str,
    transcript_sha256: str,
) -> dict[str, Any]:
    for job_path in _recent_autopilot_jobs(root):
        job = _read_json_object(job_path)
        state = job.get("semantic_candidate_discovery") if job else None
        if not isinstance(state, dict):
            continue
        if not str(state.get("transcript_sha256") or ""):
            continue
        if str(state.get("transcript_sha256")) != transcript_sha256:
            continue
        if not str(state.get("last_discovery_pack_sha256") or ""):
            continue
        if str(state.get("last_discovery_pack_sha256")) != pack_sha256:
            continue
        coverage = (
            state.get("coverage") if isinstance(state.get("coverage"), dict) else {}
        )
        _, _, coverage_valid = _coverage_counts(coverage)
        return {
            "valid": bool(coverage_valid),
            "value": state,
            "path": job_path,
        }
    return {"valid": False, "value": {}, "path": ""}


def _discovery_pack_identity_issues(
    root: Path,
    pack: dict[str, Any],
    *,
    transcript_path: Path | None,
    current_transcript_sha256: str,
) -> list[str]:
    issues: list[str] = []
    if str(pack.get("schema") or "") != DISCOVERY_PACK_SCHEMA:
        issues.append("discovery_pack_schema_mismatch")
    if transcript_path is None:
        issues.append("discovery_transcript_path_missing")
    elif not transcript_path.is_relative_to(root):
        issues.append("discovery_transcript_path_outside_bundle")
    elif not transcript_path.is_file():
        issues.append("discovery_transcript_file_missing")
    recorded_sha256 = str(pack.get("transcript_sha256") or "")
    if not recorded_sha256:
        issues.append("discovery_transcript_sha256_missing")
    elif not current_transcript_sha256:
        issues.append("discovery_transcript_sha256_unverifiable")
    elif recorded_sha256 != current_transcript_sha256:
        issues.append("discovery_transcript_sha256_mismatch")
    _, _, coverage_valid = _coverage_counts(pack.get("coverage"))
    if not coverage_valid:
        issues.append("discovery_coverage_invalid")
    return issues


def _coverage_counts(value: Any) -> tuple[int, int, bool]:
    if not isinstance(value, dict):
        return 0, 0, False
    try:
        retained = int(value.get("retained_cue_count"))
        covered = int(value.get("covered_retained_cue_count"))
    except (TypeError, ValueError):
        return 0, 0, False
    valid = retained >= 0 and covered >= 0 and covered <= retained
    return retained, covered, valid


def _coverage_matches(left: Any, right: Any) -> bool:
    left_retained, left_covered, left_valid = _coverage_counts(left)
    right_retained, right_covered, right_valid = _coverage_counts(right)
    return bool(
        left_valid
        and right_valid
        and left_retained == right_retained
        and left_covered == right_covered
        and left_covered >= left_retained
    )


def _candidate_ids(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {
        str(row.get("candidate_id") or "")
        for row in value
        if isinstance(row, dict) and str(row.get("candidate_id") or "")
    }


def _empty_discovery_state(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "complete": False,
        "cursor": 0,
        "next_cursor": 0,
        "coverage": {},
        "page_imported": False,
        "hash_drift": False,
        "transcript_sha256": "",
        "current_transcript_sha256": "",
        "source_pack_sha256": "",
        "discovery_pack_sha256": "",
        "completion_source": "",
        "job_path": "",
        "receipt_path": "",
        "receipt_sha256": "",
        "receipt_status": "missing",
        "identity_valid": status == "missing",
        "identity_issues": [],
    }


def _recent_autopilot_jobs(root: Path) -> list[Path]:
    runs_dir = root / "runs"
    if not runs_dir.is_dir():
        return []
    try:
        return sorted(
            runs_dir.glob("autopilot-*/autopilot-job.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )[:20]
    except OSError:
        return []


def _discovery_residue_exists(root: Path) -> bool:
    manifest = _read_json_object(root / "manifest.json")
    manifest_keys = {
        "transcript_semantic_candidate_discovery_pack_json",
        "transcript_semantic_candidate_discovery_prompt_markdown",
        "transcript_semantic_candidate_discovery_template_json",
        "transcript_semantic_candidate_discovery_summary",
        "transcript_semantic_candidate_discovery_llm_draft_summary",
        "transcript_semantic_candidate_suggestions_import_json",
    }
    if any(key in manifest for key in manifest_keys):
        return True
    return any(
        path.exists()
        for path in (
            root / "transcript-semantic-candidate-discovery-prompt.md",
            root / "transcript-semantic-candidate-discovery-template.json",
            root / "transcript-semantic-candidate-discovery-llm-prompt.md",
            root / "transcript-semantic-candidate-suggestions-import.json",
        )
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _resolve_bundle_path(root: Path, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()
