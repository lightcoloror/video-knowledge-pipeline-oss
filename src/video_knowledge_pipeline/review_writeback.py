from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .human_keypoint_review import build_human_keypoint_goldset
from .models import now_iso
from .review_session import apply_review_notes_to_bundle, validate_review_notes_for_bundle
from .storage import bundle_write_lock, read_json, write_json
from .transcript_semantic_correction import (
    import_transcript_semantic_review_notes,
    transcript_semantic_correction_closure,
)


WRITEBACK_SCHEMA = "video_knowledge_pipeline.review_writeback.v1"
SEMANTIC_REVIEW_SCHEMA = "video_knowledge_pipeline.transcript_semantic_correction_review_notes.v1"
_COMMENT_CORRECTION = re.compile(r"^\s*(?P<original>.+?)\s*(?:应为|改为|->|→)\s*(?P<corrected>.+?)\s*$")


def apply_review_payload_to_bundle(
    bundle_dir: str | Path,
    payload: dict[str, Any],
    *,
    write: bool = True,
    refresh_exports: bool = True,
) -> dict[str, Any]:
    """Persist and apply one browser review payload through existing VKP front doors.

    The bundle path is supplied by the local server, never by the browser payload.
    Human semantic corrections are converted to candidate-bound decisions before
    the existing strict validator and closure are invoked.
    """

    root = Path(bundle_dir).expanduser().resolve()
    _require_bundle(root)
    normalized = _normalize_review_payload(payload)
    semantic = semantic_review_notes_from_payload(root, normalized)
    human_key_points = build_human_keypoint_goldset(
        root,
        normalized,
        write=False,
    )
    review_path = root / "review-notes.json"
    semantic_path = root / "transcript-semantic-correction-review-notes.json"
    result: dict[str, Any] = {
        "schema": WRITEBACK_SCHEMA,
        "bundle_dir": str(root),
        "review_json": str(review_path),
        "semantic_review_json": str(semantic_path),
        "write": bool(write),
        "review_count": len(normalized["reviews"]),
        "semantic_review_count": len(semantic["reviews"]),
        "semantic_unresolved": semantic["unresolved"],
        "human_key_points": {
            key: value
            for key, value in human_key_points.items()
            if key != "payload"
        },
        "updated_at": now_iso(),
    }
    if not write:
        result["validation"] = validate_review_notes_for_bundle(root, review_json=None)
        result["status"] = "preview"
        result["ok"] = not semantic["unresolved"]
        return result

    with bundle_write_lock(root, operation="review_writeback", timeout_seconds=2.0):
        write_json(review_path, normalized)
        result["timeline_import"] = apply_review_notes_to_bundle(root, review_json=review_path, write=True)
        if human_key_points["status"] == "ready_to_write":
            result["human_key_points"] = {
                key: value
                for key, value in build_human_keypoint_goldset(
                    root,
                    normalized,
                    write=True,
                ).items()
                if key != "payload"
            }
        if semantic["reviews"]:
            write_json(
                semantic_path,
                {
                    "schema": SEMANTIC_REVIEW_SCHEMA,
                    "source": "review_webui_loopback",
                    "source_review_json": str(review_path),
                    "created_at": now_iso(),
                    "reviews": semantic["reviews"],
                },
            )
            imported = import_transcript_semantic_review_notes(root, review_json=semantic_path, write=True)
            result["semantic_import"] = imported
            validation = imported.get("validation") if isinstance(imported.get("validation"), dict) else {}
            accepted = int(validation.get("accepted_decision_count") or 0)
            if accepted:
                result["semantic_closure"] = transcript_semantic_correction_closure(
                    root,
                    input_json=imported["result_json"],
                    refresh_exports=refresh_exports,
                    write=True,
                )
        result["status"] = "applied"
        result["ok"] = _writeback_ok(result)
        result["updated_at"] = now_iso()
        write_json(root / "review-writeback-report.json", result)
    return result


def semantic_review_notes_from_payload(bundle_dir: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Resolve structured or narrowly formatted human notes to exact semantic candidates."""

    root = Path(bundle_dir).expanduser().resolve()
    pack = _read_optional_dict(root / "transcript-semantic-correction-pack.json")
    candidates = [row for row in pack.get("candidates", []) if isinstance(row, dict)]
    candidate_by_id = {str(row.get("candidate_id") or ""): row for row in candidates if str(row.get("candidate_id") or "")}
    timeline = _read_optional_list(root / "timeline.json")
    timeline_by_index = {_int(row.get("index")): row for row in timeline if _int(row.get("index"))}
    decisions: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    seen: set[str] = set()

    for review in _review_rows(payload):
        timeline_index = _int(review.get("timeline_index") or review.get("index"))
        timeline_item = timeline_by_index.get(timeline_index, {})
        explicit = review.get("transcript_semantic_corrections")
        structured = [row for row in explicit if isinstance(row, dict)] if isinstance(explicit, list) else []
        if review.get("candidate_id") or review.get("semantic_candidate_id"):
            structured.append(review)
        for row in structured:
            resolved = _structured_semantic_decision(row, candidate_by_id, review)
            if resolved is None:
                unresolved.append(
                    {
                        "timeline_index": timeline_index,
                        "candidate_id": row.get("candidate_id") or row.get("semantic_candidate_id") or "",
                        "reason": "unknown_candidate_or_missing_corrected_text",
                    }
                )
                continue
            candidate_id = str(resolved["candidate_id"])
            if candidate_id not in seen:
                decisions.append(resolved)
                seen.add(candidate_id)

        if structured or not _looks_like_asr_correction(review):
            continue
        parsed = _parse_comment_correction(str(review.get("comment") or review.get("notes") or ""))
        if parsed is None:
            continue
        original, corrected = parsed
        matches = _candidate_matches_for_timeline(candidates, timeline_item, original)
        if len(matches) != 1:
            unresolved.append(
                {
                    "timeline_index": timeline_index,
                    "original_text": original,
                    "corrected_text": corrected,
                    "reason": "comment_correction_candidate_not_unique",
                    "candidate_count": len(matches),
                }
            )
            continue
        candidate = matches[0]
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id in seen:
            continue
        decisions.append(_semantic_decision(candidate, corrected, review))
        seen.add(candidate_id)

    return {"reviews": decisions, "unresolved": unresolved}


def _structured_semantic_decision(
    row: dict[str, Any],
    candidate_by_id: dict[str, dict[str, Any]],
    parent_review: dict[str, Any],
) -> dict[str, Any] | None:
    candidate_id = str(row.get("candidate_id") or row.get("semantic_candidate_id") or "").strip()
    candidate = candidate_by_id.get(candidate_id)
    corrected = str(
        row.get("corrected_text")
        or row.get("corrected_transcript")
        or row.get("suggested_text")
        or ""
    ).strip()
    if not candidate or not corrected:
        return None
    return _semantic_decision(candidate, corrected, row, parent_review=parent_review)


def _semantic_decision(
    candidate: dict[str, Any],
    corrected: str,
    review: dict[str, Any],
    *,
    parent_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parent = parent_review or review
    comment = str(
        review.get("comment")
        or review.get("review_note")
        or parent.get("comment")
        or "人工复核确认逐字稿纠正"
    ).strip()
    evidence_ids = [str(value) for value in candidate.get("evidence_ids") or [] if str(value).strip()]
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "status": "corrected_transcript",
        "original_text": str(candidate.get("original_text") or ""),
        "corrected_text": corrected,
        "confidence": 1.0,
        "evidence_ids": evidence_ids,
        "comment": comment,
        "human_confirmed": True,
        "source_timeline_index": _int(parent.get("timeline_index") or parent.get("index")),
    }


def _candidate_matches_for_timeline(
    candidates: list[dict[str, Any]], timeline_item: dict[str, Any], original: str
) -> list[dict[str, Any]]:
    wanted = original.casefold().strip()
    transcript = str(
        timeline_item.get("review_transcript_excerpt")
        or timeline_item.get("human_corrected_transcript")
        or timeline_item.get("transcript")
        or ""
    ).casefold()
    if not wanted or wanted not in transcript:
        return []
    start = _float(timeline_item.get("start"), -1.0)
    end = _float(timeline_item.get("end"), start)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if str(candidate.get("original_text") or "").casefold().strip() != wanted:
            continue
        candidate_start = _float(candidate.get("start"), -1.0)
        candidate_end = _float(candidate.get("end"), candidate_start)
        overlaps = start < 0 or candidate_start < 0 or max(start, candidate_start) <= min(end + 30.0, candidate_end)
        if overlaps:
            rows.append(candidate)
    return rows


def _parse_comment_correction(value: str) -> tuple[str, str] | None:
    match = _COMMENT_CORRECTION.fullmatch(value or "")
    if not match:
        return None
    original = match.group("original").strip(" `\"'，。；;：:")
    corrected = match.group("corrected").strip(" `\"'，。；;：:")
    if not original or not corrected or original.casefold() == corrected.casefold():
        return None
    return original, corrected


def _looks_like_asr_correction(review: dict[str, Any]) -> bool:
    tags = {str(value).strip().lower() for value in review.get("tags") or []}
    return "asr_ocr_error" in tags or bool(review.get("corrected_transcript"))


def _normalize_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("review payload must be a JSON object")
    reviews = _review_rows(payload)
    return {
        "schema": "lecture_review_notes.v1",
        "package_title": str(payload.get("package_title") or ""),
        "exported_at": str(payload.get("exported_at") or now_iso()),
        "saved_to_vkp_at": now_iso(),
        "reviews": reviews,
    }


def _review_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("reviews", "items", "notes"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
    return []


def _writeback_ok(result: dict[str, Any]) -> bool:
    timeline_import = result.get("timeline_import") if isinstance(result.get("timeline_import"), dict) else {}
    validation = timeline_import.get("validation") if isinstance(timeline_import.get("validation"), dict) else {}
    if validation.get("status") == "has_errors":
        return False
    semantic_import = result.get("semantic_import") if isinstance(result.get("semantic_import"), dict) else {}
    if semantic_import and not semantic_import.get("ok"):
        return False
    closure = result.get("semantic_closure") if isinstance(result.get("semantic_closure"), dict) else {}
    if closure and not closure.get("ok"):
        return False
    return True


def _require_bundle(root: Path) -> None:
    if not (root / "manifest.json").is_file() or not (root / "timeline.json").is_file():
        raise ValueError(f"not a VKP webui bundle: {root}")


def _read_optional_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def _read_optional_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    value = read_json(path)
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
