"""Separate readable-draft delivery from native formal quality and billing."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file
from .storage import read_json

SCHEMA = "video_knowledge_pipeline.server_delivery.v1"
REVIEWABLE_BLOCKERS = frozenset({
    "transcript_quality_failed", "transcript_semantic_high_risk_review",
    "smart_summary_quality_failed", "screen_text", "export_stale",
})
FORMATS = frozenset({"html", "pdf", "md", "srt", "csv"})


def map_server_delivery(
    native: dict[str, Any], *, execution_status: str, artifacts: list[dict[str, Any]],
    artifact_root: str | Path, policy: str = "strict", review_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Map bound native evidence; never rewrite a failed native gate as passed.

    This v1 contract covers speech video only. N/A and charging require their
    own workbench contracts; neither is inferred from absent files or speech.
    """
    if policy not in {"strict", "reviewable_draft"}:
        raise ValueError("unknown delivery policy")
    root = Path(artifact_root).resolve()
    blockers, verified, formats, warnings = [], [], set(), list(review_items or [])
    if not isinstance(native, dict) or native.get("schema") != "lecture_acceptance_check.v1":
        raise ValueError("native acceptance schema required")
    native_blockers = native.get("blockers")
    if not isinstance(native_blockers, list) or any(not isinstance(x, dict) for x in native_blockers):
        raise ValueError("native blockers must be objects")
    if not isinstance(native.get("summary", {}), dict):
        raise ValueError("native summary must be an object")
    if execution_status != "completed":
        blockers.append("execution_not_completed")
    if native.get("status") not in {"complete", "incomplete", "human_review_required", "machine_action_available", "accepted_with_known_gaps"}:
        blockers.append("native_status_unknown_or_failed")
    if native.get("status") != "complete" and not native_blockers:
        blockers.append("native_incomplete_without_explained_blockers")
    for row in artifacts:
        if not isinstance(row, dict):
            raise ValueError("artifact must be an object")
        name = row.get("path")
        raw_path = root / name if isinstance(name, str) else root
        path = raw_path.resolve()
        kind, role = row.get("format"), row.get("role")
        if (not isinstance(name, str) or Path(name).is_absolute() or not path.is_relative_to(root)
                or raw_path.is_symlink() or not path.is_file() or path.stat().st_size == 0
                or row.get("sha256") != sha256_file(path)):
            blockers.append("artifact_missing_empty_or_drifted")
            continue
        if kind not in FORMATS or path.suffix.lower() != "." + kind:
            blockers.append("artifact_format_invalid")
            continue
        receipt_name = row.get("validation_receipt")
        receipt_path = (root / receipt_name).resolve() if isinstance(receipt_name, str) else root
        if (not isinstance(receipt_name, str) or Path(receipt_name).is_absolute()
                or not receipt_path.is_relative_to(root) or not receipt_path.is_file()
                or sha256_file(receipt_path) != row.get("validation_receipt_sha256")):
            blockers.append("artifact_content_not_validated")
            continue
        receipt = read_json(receipt_path)
        required_checks = {"content", "source_binding"} | ({"font", "images", "pagination"} if kind == "pdf" else
                          {"timecodes", "shot_boundaries", "audiovisual_evidence"} if kind == "csv" else
                          {"timecodes"} if kind == "srt" else {"links"} if kind == "html" else set())
        if (not isinstance(receipt, dict) or receipt.get("schema") != "video_knowledge_pipeline.delivery_validation.v1"
                or receipt.get("status") != "passed" or receipt.get("artifact_sha256") != row["sha256"]
                or receipt.get("native_acceptance_sha256") != canonical_json_sha256(native)
                or not all(receipt.get("checks", {}).get(k) is True for k in required_checks)):
            blockers.append("artifact_content_not_validated")
            continue
        if kind == "csv" and role != "shot_table":
            blockers.append("shot_csv_required_chapter_csv_is_different")
            continue
        if kind == "srt" and role not in {"canonical_transcript", "machine_draft_transcript"}:
            blockers.append("subtitle_role_invalid")
            continue
        verified.append(deepcopy(row))
        formats.add(kind)
    if formats != FORMATS:
        blockers.append("five_format_delivery_incomplete")
    unknown = [x for x in native_blockers if x.get("key") not in REVIEWABLE_BLOCKERS]
    if unknown:
        blockers.append("native_blocker_not_draft_allowlisted")
    if native_blockers:
        if policy == "strict":
            blockers.append("strict_quality_not_passed")
        elif not warnings:
            blockers.append("draft_requires_visible_review_items")
        elif not {x.get("key") for x in native_blockers} <= {x.get("key") for x in warnings if isinstance(x, dict) and x.get("reason")}:
            blockers.append("draft_review_items_do_not_cover_native_blockers")
    # Native formal acceptance needs its actual explicit quality evidence.
    formal = (not native_blockers and native.get("status") == "complete"
              and native.get("summary", {}).get("smart_summary_production_ready") is True
              and native.get("candidate_only") is not True
              and native.get("publication_allowed") is not False
              and native.get("formal_completed") is not False)
    if policy == "strict" and not formal:
        blockers.append("formal_production_readiness_missing")
    allowed = not blockers
    return {
        "schema": SCHEMA, "policy": policy, "execution_status": execution_status,
        "delivery_status": "blocked" if blockers else "formal_ready" if formal else "reviewable_draft",
        "workbench_status": "failed" if blockers else "delivered" if formal else "review_required",
        "download_allowed": allowed, "formal_completed": allowed and formal,
        "candidate_only": not (allowed and formal), "publication_allowed": allowed and formal,
        "settlement_eligible": False, "settlement_reason": "requires_separate_workbench_billing_contract",
        "native_acceptance": deepcopy(native), "native_acceptance_sha256": canonical_json_sha256(native),
        "review_items": warnings, "artifacts": verified, "blockers": sorted(set(blockers)),
        "quality_gate_bypassed": False,
    }


def inspect_server_delivery(
    bundle_dir: str | Path, *, validate_artifacts: Callable[[dict[str, Any]], list[dict[str, Any]]], **kwargs: Any,
) -> dict[str, Any]:
    """Capture once, then let owner QA bind its receipts to this exact snapshot."""
    from .acceptance_check import acceptance_check
    native = acceptance_check(bundle_dir, refresh=True, write=False)
    return map_server_delivery(native, artifacts=validate_artifacts(deepcopy(native)), **kwargs)
