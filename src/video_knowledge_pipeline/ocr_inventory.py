"""Whole-bundle OCR obligations, separate from routing and recognition quality.

No fuzzy reuse: receipts are bound to the exact image bytes. Legacy text remains
usable evidence but is never proof that the current executor ran.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .artifact_freshness import canonical_json_sha256
from .file_hash import sha256_file
from .models import now_iso
from .media_applicability import resolve_media_applicability
from .storage import read_json, write_json


def build_ocr_inventory(root: Path, timeline: list[dict[str, Any]], *, applicability: dict[str, Any] | None = None) -> dict[str, Any]:
    # Lazy import avoids the backfill/coverage dependency cycle.
    from .ocr_backfill import _first_existing_image, _is_wrapper_only_visual_text
    from .visual_evidence import human_review_accepted
    applicability = applicability or resolve_media_applicability(root, timeline=timeline)
    if applicability["visual_status"] == "not_applicable":
        return {
            "schema": "video_knowledge_pipeline.ocr_inventory.v1", "updated_at": now_iso(),
            "status": "not_applicable", "applicable": False, "scope": "verified_audio_only",
            "media_applicability": applicability, "full_video_page_coverage_verified": None,
            "summary": {**{key: 0 for key in (
                "required", "executed", "completed", "existing_evidence", "human_confirmed_evidence",
                "failed", "blocked", "not_run", "empty", "usable_text", "suspicious", "remaining")},
                "execution_complete": None},
            "pending_indexes": [], "empty_indexes": [], "items": [],
            "sampling_audit": {"status": "not_applicable", "full_page_coverage_verified": None},
        }
    path = root / "ocr-execution-inventory.json"
    previous = read_json(path) if path.is_file() else {}
    prior = {r["index"]: r for r in previous.get("items", [])}
    rows = []
    for position, item in enumerate(timeline, 1):
        index = int(item.get("index") or position)
        image = _first_existing_image(root, item)
        visual_obligation = bool(
            image or item.get("frame_paths") or item.get("needs_keyframe") or item.get("visual_text")
            or item.get("visual_route") in {"document_visual", "semantic_frame", "temporal_sequence"}
            or set(item.get("quality_issues") or []) & {
                "missing_frame", "missing_visual_text", "structured_visual_without_ocr",
                "missing_ocr", "low_ocr_confidence", "ocr_text_empty", "screen_text_low_confidence"}
        )
        if not visual_obligation:
            continue  # Pure audio/ASR intervals do not require an OCR image.
        digest = sha256_file(image) if image else ""
        old = prior.get(index, {})
        text = str(item.get("visual_text") or "").strip()
        usable = bool(text) and not _is_wrapper_only_visual_text(item)
        human_review = item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
        human_text = str(
            item.get("human_corrected_visual_text")
            or human_review.get("corrected_visual_text")
            or ""
        ).strip()
        human_confirmed_visual_text = bool(human_text) and human_review_accepted(item)
        human_binding = None
        if human_confirmed_visual_text:
            signature = canonical_json_sha256({"text": human_text, "review": human_review,
                                               "review_status": item.get("review_status")})
            prior_binding = old.get("human_review_binding")
            if isinstance(prior_binding, dict) and prior_binding.get("review_sha256") == signature:
                human_binding = prior_binding
            else:
                # Legacy reviews can bind the first observed frame, but a
                # known changed image cannot inherit an old unbound review.
                review_image = (old.get("image_sha256") if not prior_binding and old else digest)
                human_binding = {"review_sha256": signature, "image_sha256": review_image}
            human_confirmed_visual_text = bool(digest) and human_binding["image_sha256"] == digest
        # A missing frame is an obligation too; selection by VLM is irrelevant.
        row = {"index": index, "start": item.get("start", 0), "end": item.get("end", 0),
               "image_path": str(image or ""), "image_sha256": digest,
               "execution_status": "not_run", "recognition_status": "unknown",
               "usable_text": usable, "attempts": 0, "error": "",
               "human_confirmed_visual_text": human_confirmed_visual_text}
        candidates = item.get("ocr_block_candidates")
        candidates = candidates if isinstance(candidates, dict) else {}
        reference_bound = bool(digest and digest == candidates.get("source_image_sha256")
            and digest == candidates.get("observed_source_sha256"))
        row["reference_candidate_blocks"] = sum(
            isinstance(block, dict) and block.get("reference_usable") is True
            for block in candidates.get("blocks", [])) if reference_bound else 0
        if human_binding:
            # Preserve the review's image identity even when OCR later observes
            # new pixels; repeated inventory writes must not refresh the review.
            row["human_review_binding"] = human_binding
        row["quality_suspicious"] = bool(set(item.get("quality_issues") or []) & {"screen_text_low_confidence", "low_ocr_confidence", "ocr_text_low_information", "ocr_wrapper_only"})
        if human_confirmed_visual_text:
            # This closes the coverage obligation while retaining the distinction
            # between a verified human correction and an OCR executor run.
            row.update(execution_status="human_confirmed_evidence", recognition_status="text_available", usable_text=True)
        elif digest and old.get("image_sha256") == digest:
            row.update({k: old[k] for k in ("execution_status", "recognition_status", "attempts", "error", "last_attempt_at", "engine", "language", "empty_recovery_attempted", "ocr_confidence") if k in old})
            if old.get("error") == "image_changed_or_missing" and row["execution_status"] != "completed":
                row["usable_text"] = False
        elif old:
            row["previous_image_sha256"] = old.get("image_sha256", "")
            row["usable_text"] = False  # Old OCR does not describe changed pixels.
            row["error"] = "image_changed_or_missing"
        elif usable:
            row.update(execution_status="existing_evidence", recognition_status="text_available")
        if not image:
            row.update(execution_status="blocked", error="image_missing")
        if row["recognition_status"] == "suspicious":
            row.update(usable_text=False, quality_suspicious=True)
        rows.append(row)
    counts = Counter(r["execution_status"] for r in rows)
    pending = [r["index"] for r in rows if r["execution_status"] in {"not_run", "running", "blocked", "failed"}]
    empty = [r["index"] for r in rows if r["recognition_status"] == "empty"]
    return {"schema": "video_knowledge_pipeline.ocr_inventory.v1", "updated_at": now_iso(),
            "scope": "sampled_timeline_frames_only", "full_video_page_coverage_verified": False,
            "summary": {"required": len(rows), "executed": counts["completed"] + counts["failed"],
                        "completed": counts["completed"],
                        "existing_evidence": counts["existing_evidence"], "human_confirmed_evidence": counts["human_confirmed_evidence"], "failed": counts["failed"],
                        "blocked": counts["blocked"], "not_run": counts["not_run"] + counts["running"],
                        "empty": len(empty), "usable_text": sum(r["usable_text"] for r in rows),
                        "suspicious": sum(r["quality_suspicious"] for r in rows),
                        "reference_candidate_blocks": sum(r["reference_candidate_blocks"] for r in rows),
                        "reference_candidate_frames": sum(bool(r["reference_candidate_blocks"]) for r in rows),
                        "remaining": len(pending), "execution_complete": not pending},
            "pending_indexes": pending, "empty_indexes": empty,
            "sampling_audit": sampling_audit(root), "items": rows}


def sampling_audit(root: Path) -> dict[str, Any]:
    """Reuse the all-decoded-frame courseware scanner, never infer page coverage
    from OCR success on a sparse timeline or from candidate count alone.
    """
    if not (root / "courseware/scan.json").is_file():
        return {"status": "not_assessed", "next_action": "courseware-prepare then courseware-scan --execute --resume", "full_page_coverage_verified": False}
    from .courseware_material import read_scan
    try:
        scan = read_scan(root)
        pending = [w for i, w in enumerate(scan["windows"]) if i not in scan["completed_windows"]]
        return {"status": "blocked" if scan.get("failure") else ("scan_incomplete" if pending else "scan_completed_page_validation_pending"),
                "required_windows": len(scan["windows"]), "completed_windows": len(scan["completed_windows"]),
                "pending_intervals": pending, "candidate_count": len(scan["candidates"]),
                "failure": scan.get("failure"), "detector": scan.get("detector"),
                "full_page_coverage_verified": False,
                "next_action": "courseware-extract and OCR; scan completion alone does not validate all pages"}
    except (OSError, ValueError, KeyError) as exc:
        return {"status": "invalid_scan", "error": str(exc), "full_page_coverage_verified": False}


def record_ocr_result(root: Path, timeline: list[dict[str, Any]], result: dict[str, Any], *, engine: str, language: str) -> None:
    inventory = build_ocr_inventory(root, timeline)
    for row in inventory["items"]:
        if row["index"] != result.get("index"):
            continue
        if row.get("engine") != engine:
            row.pop("ocr_confidence", None)  # A prior engine's score is not this observation's score.
        completed = bool(result.get("authoritative_observation") or result.get("ok"))
        text = str(result.get("text") or "").strip()
        row.update(execution_status="completed" if completed else ("failed" if result.get("executed") else "blocked"),
                   recognition_status=("suspicious" if result.get("recognition_status") == "suspicious" else ("text_available" if text else "empty")) if completed else "unknown",
                   attempts=row["attempts"] + int(bool(result.get("executed"))),
                   error=str(result.get("stderr") or ""), last_attempt_at=now_iso(), engine=engine, language=language)
        row["empty_recovery_attempted"] = bool(result.get("empty_recovery_attempted"))
        if isinstance(result.get("ocr_confidence"), dict):
            row["ocr_confidence"] = result["ocr_confidence"]
    write_json(root / "ocr-execution-inventory.json", inventory)


def persist_ocr_inventory(root: Path, timeline: list[dict[str, Any]]) -> dict[str, Any]:
    inventory = build_ocr_inventory(root, timeline)
    write_json(root / "ocr-execution-inventory.json", inventory)
    return inventory
