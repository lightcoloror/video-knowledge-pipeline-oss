from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .bundle_next import bundle_next_action
from .bundle_readiness import STRUCTURED_TYPES, audit_bundle_readiness
from .bundle_source_artifacts import bundle_source_artifacts
from .knowledge_coverage import audit_knowledge_coverage
from .markdown_text import markdown_table_cell as _markdown_cell
from .models import now_iso
from .storage import read_json, write_json
from .transcript import format_timestamp

REVIEW_SESSION_SCHEMA = "lecture_review_session.v1"
REVIEW_NOTES_SCHEMA = "lecture_review_notes.v1"
ACCEPTED_REVIEW_STATUSES = {
    "accepted",
    "reviewed",
    "keep_image",
    "accepted_known_gap",
    "accepted_no_visual_content",
    "accepted_provider_blocked",
    "corrected_visual_text",
    "corrected_visual_understanding",
    "corrected_temporal_visual_understanding",
    "corrected_review_start",
}


def apply_review_notes_to_bundle(
    bundle_dir: str | Path,
    *,
    review_json: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Import reviewed timeline notes into a WebUI bundle without overwriting model outputs."""
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline not found: {timeline_path}")
    manifest = read_json(manifest_path)
    timeline_data = read_json(timeline_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    if not isinstance(timeline_data, list):
        raise ValueError("timeline.json must be a JSON array")
    timeline = [item for item in timeline_data if isinstance(item, dict)]
    review_path = Path(review_json).expanduser().resolve() if review_json else root / str(manifest.get("review_notes") or "review-notes.json")
    if not review_path.exists():
        payload: Any = {"schema": REVIEW_NOTES_SCHEMA, "reviews": [], "created_at": now_iso()}
        if write:
            write_json(review_path, payload)
    else:
        payload = read_json(review_path)
    reviews = _review_note_rows(payload)
    validation = validate_review_notes_payload(root, timeline, reviews)
    invalid_rows = set(validation.get("invalid_row_numbers") or [])
    updated_indexes: list[int] = []
    skipped: list[dict[str, Any]] = []
    by_index = {_int_value(item.get("index")): item for item in timeline if _int_value(item.get("index"))}
    for row_number, review in enumerate(reviews, start=1):
        index = _int_value(review.get("timeline_index") or review.get("index"))
        if row_number in invalid_rows:
            skipped.append({"timeline_index": index, "row_number": row_number, "reason": "validation_error"})
            continue
        item = by_index.get(index)
        if not item:
            skipped.append({"timeline_index": index, "row_number": row_number, "reason": "timeline_index_not_found"})
            continue
        human_review = _normalise_review_note(review)
        item["human_review"] = human_review
        item["review_status"] = human_review["status"]
        item["needs_human_review"] = human_review["status"] not in ACCEPTED_REVIEW_STATUSES
        _apply_non_destructive_corrections(item, human_review)
        _mark_review_quality_issues(item, human_review)
        updated_indexes.append(index)
    result = {
        "schema": "lecture_review_notes_import.v1",
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "timeline_path": str(timeline_path),
        "review_json": str(review_path),
        "write": write,
        "review_count": len(reviews),
        "updated": len(updated_indexes),
        "updated_indexes": updated_indexes,
        "skipped": skipped,
        "validation": validation,
        "status_summary": _review_status_summary(reviews, skipped=skipped),
        "updated_at": now_iso(),
    }
    if write:
        write_json(timeline_path, timeline)
        manifest["review_notes_last_import"] = result
        write_json(manifest_path, manifest)
        audit_knowledge_coverage(root, write=True)
        audit_bundle_readiness(root, write=True)
        result["post_apply_refresh"] = _post_apply_refresh(root)
    return result


def validate_review_notes_for_bundle(bundle_dir: str | Path, *, review_json: str | Path | None = None) -> dict[str, Any]:
    """Validate review notes without applying them to the bundle."""
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline not found: {timeline_path}")
    manifest = read_json(manifest_path)
    timeline_data = read_json(timeline_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    if not isinstance(timeline_data, list):
        raise ValueError("timeline.json must be a JSON array")
    timeline = [item for item in timeline_data if isinstance(item, dict)]
    review_path = Path(review_json).expanduser().resolve() if review_json else root / str(manifest.get("review_notes") or "review-notes.json")
    payload: Any = read_json(review_path) if review_path.exists() else {"schema": REVIEW_NOTES_SCHEMA, "reviews": []}
    reviews = _review_note_rows(payload)
    validation = validate_review_notes_payload(root, timeline, reviews)
    validation.update(
        {
            "schema": "lecture_review_notes_validation.v1",
            "bundle_dir": str(root),
            "review_json": str(review_path),
            "review_count": len(reviews),
            "checked_at": now_iso(),
        }
    )
    return validation


def validate_review_notes_payload(root: Path, timeline: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> dict[str, Any]:
    by_index = {_int_value(item.get("index")): item for item in timeline if _int_value(item.get("index"))}
    seen: set[int] = set()
    invalid_rows: set[int] = set()
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for row_number, review in enumerate(reviews, start=1):
        index = _int_value(review.get("timeline_index") or review.get("index"))
        status = _canonical_review_status(review.get("status"))
        if not index:
            errors.append({"row_number": row_number, "timeline_index": index, "key": "missing_timeline_index", "message": "Review row is missing timeline_index."})
            invalid_rows.add(row_number)
            continue
        if index not in by_index:
            errors.append({"row_number": row_number, "timeline_index": index, "key": "timeline_index_not_found", "message": "timeline_index does not exist in timeline.json."})
            invalid_rows.add(row_number)
            continue
        if index in seen:
            errors.append({"row_number": row_number, "timeline_index": index, "key": "duplicate_timeline_index", "message": "Only one review row per timeline_index is allowed in one import."})
            invalid_rows.add(row_number)
        seen.add(index)
        if status == "corrected_visual_understanding" and not _non_empty_dict(review.get("corrected_visual_understanding")):
            errors.append({"row_number": row_number, "timeline_index": index, "key": "missing_corrected_visual_understanding", "message": "status=corrected_visual_understanding requires a non-empty corrected_visual_understanding object."})
            invalid_rows.add(row_number)
        if status == "corrected_temporal_visual_understanding" and not _non_empty_dict(review.get("corrected_temporal_visual_understanding")):
            errors.append({"row_number": row_number, "timeline_index": index, "key": "missing_corrected_temporal_visual_understanding", "message": "status=corrected_temporal_visual_understanding requires a non-empty corrected_temporal_visual_understanding object."})
            invalid_rows.add(row_number)
        if status == "corrected_visual_text" and not str(review.get("corrected_visual_text") or "").strip() and not _has_tile_correction_text(review):
            errors.append({"row_number": row_number, "timeline_index": index, "key": "missing_corrected_visual_text", "message": "status=corrected_visual_text requires corrected_visual_text or tile_corrections[].corrected_text."})
            invalid_rows.add(row_number)
        if status == "corrected_transcript" and not str(review.get("corrected_transcript") or "").strip() and not _has_semantic_correction_text(review):
            errors.append(
                {"row_number": row_number, "timeline_index": index, "key": "missing_corrected_transcript", "message": "status=corrected_transcript requires corrected_transcript or transcript_semantic_corrections[].corrected_text."}
            )
            invalid_rows.add(row_number)
        if status == "corrected_review_start" and _optional_float_value(review.get("corrected_review_start")) is None:
            errors.append({"row_number": row_number, "timeline_index": index, "key": "missing_corrected_review_start", "message": "status=corrected_review_start requires numeric corrected_review_start seconds."})
            invalid_rows.add(row_number)
        evidence_paths = _review_evidence_paths(review, by_index[index])
        if status in ACCEPTED_REVIEW_STATUSES and not evidence_paths:
            warnings.append({"row_number": row_number, "timeline_index": index, "key": "missing_evidence_path", "message": "Accepted review has no evidence_frame_paths or timeline asset paths."})
        for evidence_path in evidence_paths:
            candidate = Path(evidence_path)
            if not candidate.is_absolute():
                candidate = root / candidate
            if not candidate.exists():
                warnings.append({"row_number": row_number, "timeline_index": index, "key": "evidence_path_not_found", "path": str(evidence_path), "message": "Evidence path does not exist on disk."})
    status = "ok"
    if errors:
        status = "has_errors"
    elif warnings:
        status = "has_warnings"
    return {
        "schema": "lecture_review_notes_validation.v1",
        "status": status,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "invalid_row_numbers": sorted(invalid_rows),
        "errors": errors,
        "warnings": warnings,
    }


def review_notes_template_from_timeline(timeline: list[dict[str, Any]], *, created_at: str | None = None) -> dict[str, Any]:
    """Build a fillable review-notes template directly from bundle timeline rows."""
    rows: list[dict[str, Any]] = []
    for position, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        index = _int_value(item.get("index")) or position
        rows.append(
            {
                "timeline_index": index,
                "time_range": _time_range_label(item),
                "route": item.get("visual_route", ""),
                "reason": ", ".join(str(value) for value in item.get("quality_issues") or []),
                "suggested_status": _suggested_status_from_timeline_item(item),
                "evidence_frame_paths": _asset_paths(item),
                "transcript_excerpt": _truncate_text(item.get("transcript"), 220),
                "visual_text_excerpt": _truncate_text(item.get("visual_text"), 220),
                "model_output_excerpt": _model_output_excerpt(item, 260),
                "tags": [],
                "comment": "",
                "corrected_transcript": "",
                "corrected_visual_text": "",
                "corrected_visual_understanding": {},
                "corrected_temporal_visual_understanding": {},
                "corrected_review_start": "",
                "reviewed_at": "",
            }
        )
    return {"schema": REVIEW_NOTES_SCHEMA, "created_at": created_at or now_iso(), "reviews": rows}


def _review_note_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        raise ValueError("review JSON must be an object or array")
    for key in ("reviews", "items", "notes"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [item for item in rows if isinstance(item, dict)]
    return []


def _normalise_review_note(review: dict[str, Any]) -> dict[str, Any]:
    status = _canonical_review_status(review.get("status"))
    tags = review.get("tags") if isinstance(review.get("tags"), list) else []
    return {
        "schema": REVIEW_NOTES_SCHEMA,
        "timeline_index": _int_value(review.get("timeline_index") or review.get("index")),
        "status": status,
        "tags": [str(tag) for tag in tags if str(tag)],
        "comment": str(review.get("comment") or review.get("notes") or "").strip(),
        "human_key_point_confirmed": bool(
            review.get("human_key_point_confirmed")
        ),
        "human_key_point_text": str(review.get("human_key_point_text") or "").strip(),
        "human_key_point_aliases": _normalise_human_key_point_aliases(review.get("human_key_point_aliases")),
        "corrected_transcript": str(review.get("corrected_transcript") or "").strip(),
        "transcript_semantic_corrections": _normalise_transcript_semantic_corrections(review.get("transcript_semantic_corrections")),
        "corrected_visual_text": str(review.get("corrected_visual_text") or "").strip(),
        "tile_corrections": _normalise_tile_corrections(review.get("tile_corrections")),
        "corrected_visual_understanding": review.get("corrected_visual_understanding") if isinstance(review.get("corrected_visual_understanding"), dict) else {},
        "corrected_temporal_visual_understanding": review.get("corrected_temporal_visual_understanding")
        if isinstance(review.get("corrected_temporal_visual_understanding"), dict)
        else {},
        "corrected_review_start": _optional_float_value(review.get("corrected_review_start")),
        "reviewed_at": str(review.get("reviewed_at") or now_iso()),
        "keep_image": bool(review.get("keep_image")) or str(review.get("status") or "").strip().lower() == "keep_image",
        "needs_rerun_ocr": status == "needs_rerun_ocr",
        "accepted_known_gap": status == "accepted_known_gap",
    }


def _canonical_review_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    return {
        "reviewed": "accepted",
        "ok": "accepted",
        "needs_revision": "needs_fix",
        "fix": "needs_fix",
    }.get(status, status or "needs_human_review")


def _normalise_human_key_point_aliases(value: Any) -> list[str]:
    if isinstance(value, str):
        values = value.split("|")
    elif isinstance(value, list):
        values = value
    else:
        values = []
    return [
        str(item).strip()
        for item in values
        if str(item).strip()
    ]


def _non_empty_dict(value: Any) -> bool:
    return isinstance(value, dict) and bool(value)


def _normalise_transcript_semantic_corrections(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id") or row.get("semantic_candidate_id") or "").strip()
        corrected = str(row.get("corrected_text") or row.get("corrected_transcript") or "").strip()
        if not candidate_id or not corrected:
            continue
        rows.append(
            {
                "candidate_id": candidate_id,
                "status": "corrected_transcript",
                "original_text": str(row.get("original_text") or "").strip(),
                "corrected_text": corrected,
                "evidence_ids": [str(item) for item in row.get("evidence_ids") or [] if str(item).strip()],
                "comment": str(row.get("comment") or "").strip(),
                "human_confirmed": True,
            }
        )
    return rows


def _has_semantic_correction_text(review: dict[str, Any]) -> bool:
    return bool(_normalise_transcript_semantic_corrections(review.get("transcript_semantic_corrections")))



def _normalise_tile_corrections(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        corrected = str(row.get("corrected_text") or row.get("text") or "").strip()
        comment = str(row.get("comment") or "").strip()
        status = str(row.get("status") or "").strip()
        if not corrected and not comment and not status:
            continue
        rows.append(
            {
                "tile_id": str(row.get("tile_id") or "").strip(),
                "status": status or ("corrected" if corrected else "reviewed"),
                "corrected_text": corrected,
                "comment": comment,
                "evidence_path": str(row.get("evidence_path") or row.get("tile_path") or "").strip(),
            }
        )
    return rows


def _has_tile_correction_text(review: dict[str, Any]) -> bool:
    return bool(_tile_corrections_text(_normalise_tile_corrections(review.get("tile_corrections"))))


def _tile_corrections_text(corrections: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for row in corrections:
        text = str(row.get("corrected_text") or "").strip()
        if not text:
            continue
        tile_id = str(row.get("tile_id") or "").strip()
        prefix = f"[{tile_id}] " if tile_id else ""
        parts.append(prefix + text)
    return "\n".join(parts).strip()

def _review_evidence_paths(review: dict[str, Any], item: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    explicit = review.get("evidence_frame_paths")
    if isinstance(explicit, list):
        paths.extend(str(path).strip() for path in explicit if str(path).strip())
    if not paths:
        paths.extend(_asset_paths(item))
    return paths


def _apply_non_destructive_corrections(item: dict[str, Any], human_review: dict[str, Any]) -> None:
    if human_review.get("corrected_transcript"):
        item["human_corrected_transcript"] = human_review["corrected_transcript"]
    tile_corrections = human_review.get("tile_corrections") if isinstance(human_review.get("tile_corrections"), list) else []
    tile_text = _tile_corrections_text(tile_corrections)
    if human_review.get("corrected_visual_text"):
        item["human_corrected_visual_text"] = human_review["corrected_visual_text"]
    elif tile_text:
        item["human_corrected_visual_text"] = tile_text
    if tile_corrections:
        item["human_tile_corrections"] = tile_corrections
    corrected_visual = human_review.get("corrected_visual_understanding")
    if isinstance(corrected_visual, dict) and corrected_visual:
        item["human_corrected_visual_understanding"] = corrected_visual
    corrected_temporal = human_review.get("corrected_temporal_visual_understanding")
    if isinstance(corrected_temporal, dict) and corrected_temporal:
        item["human_corrected_temporal_visual_understanding"] = corrected_temporal
    corrected_review_start = human_review.get("corrected_review_start")
    if corrected_review_start is not None:
        item["review_start"] = corrected_review_start
        item["review_start_source"] = "human_review_note"
        item["human_corrected_review_start"] = corrected_review_start
    if human_review.get("keep_image"):
        item["human_keep_image"] = True


def _mark_review_quality_issues(item: dict[str, Any], human_review: dict[str, Any]) -> None:
    if human_review.get("status") not in ACCEPTED_REVIEW_STATUSES:
        return
    issues = [str(issue) for issue in item.get("quality_issues") or []]
    accepted = {
        "needs_human_review",
        "missing_visual_text",
        "missing_ocr",
        "low_ocr_confidence",
        "ocr_text_empty",
        "structured_visual_without_structure",
        "missing_visual_understanding",
        "semantic_frame_without_analysis",
        "temporal_sequence_without_analysis",
        "provider_blocked_visual_understanding",
        "screen_text_low_confidence",
        "timeline_alignment_issue",
    }
    item["quality_issues"] = [issue for issue in issues if issue not in accepted]


def _post_apply_refresh(root: Path) -> dict[str, Any]:
    refreshed: dict[str, Any] = {}
    try:
        from .knowledge_note_export import export_knowledge_note

        export = export_knowledge_note(root, write=True)
        refreshed["knowledge_note"] = export.get("note_path", "")
    except Exception as exc:  # pragma: no cover - best-effort refresh report
        refreshed["export_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from .acceptance_check import acceptance_check

        acceptance = acceptance_check(root, refresh=True, write=True)
        refreshed["acceptance_check"] = acceptance.get("report_path", "")
        refreshed["acceptance_status"] = acceptance.get("status", "")
    except Exception as exc:  # pragma: no cover - best-effort refresh report
        refreshed["acceptance_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from .bundle_status import bundle_status_report

        status = bundle_status_report(root, refresh=True, write=True)
        refreshed["bundle_status"] = status.get("report_path", "")
    except Exception as exc:  # pragma: no cover - best-effort refresh report
        refreshed["bundle_status_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from .webui_bridge import refresh_bundle_review_html

        html = refresh_bundle_review_html(root, write=True)
        refreshed["review_html"] = html.get("review_html_path", "")
    except Exception as exc:  # pragma: no cover - best-effort refresh report
        refreshed["review_html_error"] = f"{type(exc).__name__}: {exc}"
    try:
        closure = review_closure_status(root, write=True)
        refreshed["review_closure_status"] = closure.get("report_markdown_path", "")
    except Exception as exc:  # pragma: no cover - best-effort refresh report
        refreshed["review_closure_status_error"] = f"{type(exc).__name__}: {exc}"
    return refreshed


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_float_value(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def prepare_review_session(
    bundle_dir: str | Path,
    *,
    refresh: bool = True,
    limit: int = 30,
    offset: int = 0,
    reason: str = "",
    group_by: str = "reason",
    include_closed: bool = False,
    output_prefix: str = "review-pack",
    run_id: str | None = None,
    dependency_versions: dict | None = None,
) -> dict[str, Any]:
    """Write a human/agent handoff for reviewing a WebUI lecture bundle."""
    if run_id:
        if offset or reason or include_closed or output_prefix != "review-pack" or limit != 30 or group_by != "reason":
            raise ValueError("versioned_review_filters_not_supported")
        from .versioned_semantic_run import review
        return review(bundle_dir, run_id, dependency_versions)
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    review_html_path = root / "review.html"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not review_html_path.exists():
        raise FileNotFoundError(f"review html not found: {review_html_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")

    next_result = bundle_next_action(root, refresh=refresh)
    readiness = manifest.get("review_readiness") if isinstance(manifest.get("review_readiness"), dict) else {}
    if refresh:
        refreshed_manifest = read_json(manifest_path)
        if isinstance(refreshed_manifest, dict):
            manifest = refreshed_manifest
            readiness = manifest.get("review_readiness") if isinstance(manifest.get("review_readiness"), dict) else readiness
    timeline_data = read_json(timeline_path) if timeline_path.exists() else []
    timeline = [item for item in timeline_data if isinstance(item, dict)] if isinstance(timeline_data, list) else []
    timeline_alignment = _timeline_alignment_by_index(root)
    review_targets = _review_targets(
        timeline,
        readiness=readiness,
        manifest=manifest,
        timeline_alignment=timeline_alignment,
        bundle_dir=root,
        limit=limit,
        offset=offset,
        reason=reason,
        include_closed=include_closed,
    )
    source_artifacts = bundle_source_artifacts(root, refresh=False, write=False)
    knowledge_coverage = audit_knowledge_coverage(root, write=False)

    post_review = manifest.get("post_review") if isinstance(manifest.get("post_review"), dict) else {}
    review_notes_path = root / str(manifest.get("review_notes") or "review-notes.json")
    refresh_args_path = root / str(manifest.get("mcp_refresh_args") or "mcp-refresh-lecture-review.args.json")
    session_path = root / "review-session.json"
    markdown_path = root / "review-session.md"
    template_path = root / "review-notes.template.json"
    fill_guide_path = root / "review-fill-guide.md"
    review_pack_json_path = root / f"{_safe_output_prefix(output_prefix)}.json"
    review_pack_markdown_path = root / f"{_safe_output_prefix(output_prefix)}.md"
    todo_path = root / "review-notes.todo.json"
    closure_status = review_closure_status(root, write=False)
    args_path = root / "mcp-prepare-review-session.args.json"
    review_template = _review_notes_template(review_targets)
    review_pack = _build_review_pack(
        root=root,
        manifest=manifest,
        review_targets=review_targets,
        review_template=review_template,
        closure_status=closure_status,
        group_by=group_by,
        output_prefix=_safe_output_prefix(output_prefix),
        include_closed=include_closed,
    )

    session = {
        "schema": REVIEW_SESSION_SCHEMA,
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "timeline_path": str(timeline_path),
        "review_html_path": str(review_html_path),
        "review_file_url": review_html_path.as_uri(),
        "review_notes_path": str(review_notes_path),
        "review_notes_template_path": str(template_path),
        "review_fill_guide_path": str(fill_guide_path),
        "review_pack_path": str(review_pack_markdown_path),
        "review_pack_json_path": str(review_pack_json_path),
        "review_notes_todo_path": str(todo_path),
        "review_closure_status_path": str(root / "review-closure-status.md"),
        "review_closure_status_json_path": str(root / "review-closure-status.json"),
        "refresh_args_path": str(refresh_args_path),
        "refreshed": bool(refresh),
        "title": str(manifest.get("title") or "Lecture Review Session"),
        "coverage": knowledge_coverage.get("coverage", {}) if isinstance(knowledge_coverage.get("coverage"), dict) else {},
        "readiness": readiness,
        "next_action": next_result.get("next_action") if isinstance(next_result.get("next_action"), dict) else {},
        "review_targets": review_targets,
        "review_pack": {
            "path": str(review_pack_markdown_path),
            "json_path": str(review_pack_json_path),
            "todo_path": str(todo_path),
            "group_by": group_by,
            "include_closed": bool(include_closed),
            "group_count": len(review_pack.get("groups") or []),
        },
        "review_closure_status": closure_status.get("summary", {}),
        "source_artifacts": {
            "summary": source_artifacts.get("summary", {}),
            "source_artifacts_path": source_artifacts.get("source_artifacts_path", ""),
            "source_artifacts_json_path": source_artifacts.get("source_artifacts_json_path", ""),
            "source_artifacts_exists": source_artifacts.get("source_artifacts_exists", False),
            "source_artifacts_markdown_exists": source_artifacts.get("source_artifacts_markdown_exists", False),
            "mcp_args_path": source_artifacts.get("mcp_args_path", ""),
            "next_action": source_artifacts.get("next_action", {}),
            "missing": source_artifacts.get("missing", [])[:10],
        },
        "knowledge_coverage": {
            "summary": _knowledge_coverage_summary(knowledge_coverage.get("coverage", {})),
            "coverage_path": knowledge_coverage.get("coverage_path", ""),
            "coverage_markdown_path": knowledge_coverage.get("coverage_markdown_path", ""),
            "mcp_args_path": knowledge_coverage.get("mcp_args_path", ""),
            "next_action": (knowledge_coverage.get("coverage", {}) or {}).get("next_action", {}),
            "channels": (knowledge_coverage.get("coverage", {}) or {}).get("channels", [])[:10],
            "blockers": (knowledge_coverage.get("coverage", {}) or {}).get("blockers", [])[:10],
            "weak_channels": (knowledge_coverage.get("coverage", {}) or {}).get("weak_channels", [])[:10],
        },
        "post_review": {
            "mcp_tool": post_review.get("mcp_tool") or "refresh_lecture_review_outputs",
            "mcp_args_path": str(refresh_args_path),
            "command": post_review.get("refresh_command") or "",
        },
        "human_steps": [
            "Open review_file_url in a browser.",
            "Review every timeline item, especially blockers and structured visual material.",
            "Export or save review notes to review_notes_path.",
            "Run post_review.mcp_tool with refresh_args_path, then audit bundle readiness again.",
        ],
        "session_path": str(session_path),
        "markdown_path": str(markdown_path),
        "mcp_args_path": str(args_path),
    }
    write_json(session_path, session)
    write_json(template_path, review_template)
    write_json(review_pack_json_path, review_pack)
    review_pack_markdown_path.write_text(render_review_pack_markdown(review_pack), encoding="utf-8")
    write_json(todo_path, review_template)
    closure_status = review_closure_status(root, write=True)
    session["review_closure_status"] = closure_status.get("summary", {})
    write_json(session_path, session)
    markdown_path.write_text(render_review_session_markdown(session), encoding="utf-8")
    fill_guide_path.write_text(render_review_fill_guide_markdown(session, review_template), encoding="utf-8")
    write_json(
        args_path,
        {
            "bundle_dir": str(root),
            "refresh": True,
            "limit": int(limit or 0),
            "offset": int(offset or 0),
            "reason": str(reason or ""),
            "group_by": str(group_by or "reason"),
            "include_closed": bool(include_closed),
            "output_prefix": _safe_output_prefix(output_prefix),
        },
    )
    manifest["review_session"] = "review-session.md"
    manifest["review_session_json"] = "review-session.json"
    manifest["review_notes_template"] = "review-notes.template.json"
    manifest["review_fill_guide"] = "review-fill-guide.md"
    manifest["review_pack"] = str(review_pack_markdown_path.name)
    manifest["review_pack_json"] = str(review_pack_json_path.name)
    manifest["review_notes_todo"] = str(todo_path.name)
    manifest["review_closure_status"] = "review-closure-status.md"
    manifest["review_closure_status_json"] = "review-closure-status.json"
    manifest["mcp_prepare_review_session_args"] = "mcp-prepare-review-session.args.json"
    manifest["mcp_review_closure_status_args"] = "mcp-review-closure-status.args.json"
    write_json(manifest_path, manifest)
    return session


def review_closure_status(bundle_dir: str | Path, *, write: bool = True) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline not found: {timeline_path}")
    manifest = read_json(manifest_path)
    timeline_data = read_json(timeline_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    if not isinstance(timeline_data, list):
        raise ValueError("timeline.json must be a JSON array")
    timeline = [item for item in timeline_data if isinstance(item, dict)]
    readiness = manifest.get("review_readiness") if isinstance(manifest.get("review_readiness"), dict) else {}
    targets = _review_targets(timeline, readiness=readiness, manifest=manifest, bundle_dir=root, limit=0, include_closed=False)
    all_targets = _review_targets(timeline, readiness=readiness, manifest=manifest, bundle_dir=root, limit=0, include_closed=True)
    review_notes_path = root / str(manifest.get("review_notes") or "review-notes.json")
    review_rows = _review_note_rows(read_json(review_notes_path)) if review_notes_path.exists() else []
    validation = validate_review_notes_payload(root, timeline, review_rows) if review_rows else {
        "schema": "lecture_review_notes_validation.v1",
        "status": "ok",
        "error_count": 0,
        "warning_count": 0,
        "invalid_row_numbers": [],
        "errors": [],
        "warnings": [],
    }
    closed_items = [item for item in timeline if _is_reviewed(item)]
    all_review_targets = [item for item in all_targets.get("items") or [] if isinstance(item, dict)]
    closed_targets = [item for item in all_review_targets if item.get("closed")]
    last_import = manifest.get("review_notes_last_import") if isinstance(manifest.get("review_notes_last_import"), dict) else {}
    summary = {
        "total_reviewable": len(all_review_targets) or len(targets.get("items") or []) + len(closed_items),
        "open": int(targets.get("total_open") or 0),
        "closed": len(closed_targets) if all_review_targets else len(closed_items),
        "review_notes_rows": len(review_rows),
        "imported": int(last_import.get("updated") or 0),
        "invalid": int(validation.get("error_count") or 0),
        "warnings": int(validation.get("warning_count") or 0),
    }
    result = {
        "schema": "lecture_review_closure_status.v1",
        "checked_at": now_iso(),
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "timeline_path": str(timeline_path),
        "review_notes_path": str(review_notes_path),
        "summary": summary,
        "open_by_reason": targets.get("by_reason", {}),
        "closed_by_reason": _closed_target_reason_counts(closed_targets),
        "closed_targets": closed_targets,
        "closed_by_status": _timeline_review_status_counts(timeline),
        "suggested_statuses": _suggested_status_counts(targets),
        "validation": validation,
        "next_batch": _next_review_batch(root, targets),
        "report_path": str(root / "review-closure-status.json"),
        "report_markdown_path": str(root / "review-closure-status.md"),
        "mcp_args_path": str(root / "mcp-review-closure-status.args.json"),
    }
    if write:
        manifest["review_closure_status"] = "review-closure-status.md"
        manifest["review_closure_status_json"] = "review-closure-status.json"
        manifest["mcp_review_closure_status_args"] = "mcp-review-closure-status.args.json"
        write_json(manifest_path, manifest)
        write_json(root / "review-closure-status.json", result)
        (root / "review-closure-status.md").write_text(render_review_closure_status_markdown(result), encoding="utf-8")
        write_json(root / "mcp-review-closure-status.args.json", {"bundle_dir": str(root), "write": True})
    return result


def _closed_target_reason_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        for reason in item.get("reasons") or ["closed"]:
            key = str(reason or "closed")
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))

def render_review_session_markdown(session: dict[str, Any]) -> str:
    readiness = session.get("readiness") if isinstance(session.get("readiness"), dict) else {}
    counts = readiness.get("counts") if isinstance(readiness.get("counts"), dict) else {}
    next_action = session.get("next_action") if isinstance(session.get("next_action"), dict) else {}
    lines = [
        "# Lecture Review Session",
        "",
        f"- Title: {session.get('title', '')}",
        f"- Bundle: `{session.get('bundle_dir', '')}`",
        f"- Review UI: {session.get('review_file_url', '')}",
        f"- Review notes JSON: `{session.get('review_notes_path', '')}`",
        f"- Review notes template: `{session.get('review_notes_template_path', '')}`",
        f"- Review pack: `{session.get('review_pack_path', '')}`",
        f"- Review closure status: `{session.get('review_closure_status_path', '')}`",
        f"- Manifest: `{session.get('manifest_path', '')}`",
        "",
        "## Current Gate",
        "",
        f"- Status: `{readiness.get('status', 'unknown')}`",
        f"- Ready: `{readiness.get('ready', False)}`",
        f"- Next action: `{next_action.get('key', '')}` / {next_action.get('label', '')}",
        f"- Reason: {next_action.get('reason') or next_action.get('hint') or ''}",
        f"- Human required: `{next_action.get('human_required', '')}`",
        "",
        "## Counts",
        "",
    ]
    for key in [
        "timeline_items",
        "reviewed_items",
        "pending_review",
        "risk_items",
        "unreviewed_risk_items",
        "structured_items",
        "pending_structured",
        "frame_gap_items",
        "asset_gap_items",
        "structure_gap_items",
        "time_gap_count",
    ]:
        if key in counts:
            lines.append(f"- `{key}`: {counts.get(key)}")
    blockers = readiness.get("blockers") if isinstance(readiness.get("blockers"), list) else []
    lines.extend(["", "## Blockers", ""])
    if blockers:
        for blocker in blockers:
            if not isinstance(blocker, dict):
                continue
            lines.append(f"- `{blocker.get('key', '')}`: {blocker.get('message', '')} ({blocker.get('count', 0)})")
    else:
        lines.append("No blockers recorded.")
    lines.extend(
        [
            "",
            "## Agent Handoff",
            "",
            f"- Next MCP tool: `{next_action.get('mcp_tool', '')}`",
            f"- Next MCP args: `{next_action.get('mcp_args_path', '')}`",
            f"- Post-review MCP tool: `{(session.get('post_review') or {}).get('mcp_tool', '')}`",
            f"- Post-review MCP args: `{(session.get('post_review') or {}).get('mcp_args_path', '')}`",
            "",
            "## Human Steps",
            "",
        ]
    )
    for step in session.get("human_steps") or []:
        lines.append(f"- [ ] {step}")
    targets = session.get("review_targets") if isinstance(session.get("review_targets"), dict) else {}
    items = targets.get("items") if isinstance(targets.get("items"), list) else []
    source_artifacts = session.get("source_artifacts") if isinstance(session.get("source_artifacts"), dict) else {}
    source_summary = source_artifacts.get("summary") if isinstance(source_artifacts.get("summary"), dict) else {}
    source_next = source_artifacts.get("next_action") if isinstance(source_artifacts.get("next_action"), dict) else {}
    missing_artifacts = source_artifacts.get("missing") if isinstance(source_artifacts.get("missing"), list) else []
    knowledge_coverage = session.get("knowledge_coverage") if isinstance(session.get("knowledge_coverage"), dict) else {}
    closure = session.get("review_closure_status") if isinstance(session.get("review_closure_status"), dict) else {}
    knowledge_summary = knowledge_coverage.get("summary") if isinstance(knowledge_coverage.get("summary"), dict) else {}
    knowledge_next = knowledge_coverage.get("next_action") if isinstance(knowledge_coverage.get("next_action"), dict) else {}
    knowledge_channels = knowledge_coverage.get("channels") if isinstance(knowledge_coverage.get("channels"), list) else []
    lines.extend(
        [
            "",
            "## Review Closure",
            "",
            f"- Open: `{closure.get('open', targets.get('total_open', 0))}`",
            f"- Closed: `{closure.get('closed', 0)}`",
            f"- Imported: `{closure.get('imported', 0)}`",
            f"- Invalid rows: `{closure.get('invalid', 0)}`",
            f"- Review pack: `{session.get('review_pack_path', '')}`",
            f"- Todo JSON: `{session.get('review_notes_todo_path', '')}`",
            "",
            "## Knowledge Coverage",
            "",
            f"- Status: `{knowledge_summary.get('status', 'unknown')}`",
            f"- Timeline items: `{knowledge_summary.get('timeline_items', 0)}`",
            f"- Channels: `{knowledge_summary.get('channel_count', 0)}`",
            f"- Blocked channels: `{knowledge_summary.get('blocked_count', 0)}`",
            f"- Weak channels: `{knowledge_summary.get('weak_count', 0)}`",
            f"- Markdown report: `{knowledge_coverage.get('coverage_markdown_path', '')}`",
            f"- JSON report: `{knowledge_coverage.get('coverage_path', '')}`",
            f"- MCP args: `{knowledge_coverage.get('mcp_args_path', '')}`",
            f"- Next coverage action: `{knowledge_next.get('key', '')}` / {knowledge_next.get('label', '')}",
            "",
        ]
    )
    if knowledge_channels:
        lines.extend(["| Channel | Status | Covered | Blockers | MCP Tool |", "|---|---|---:|---:|---|"])
        for channel in knowledge_channels:
            if not isinstance(channel, dict):
                continue
            lines.append(
                f"| {_markdown_cell(str(channel.get('label') or channel.get('key') or ''))} | `{channel.get('status', '')}` | {channel.get('covered_count', 0)}/{channel.get('expected_count', 0)} | {channel.get('blocker_count', 0)} | `{channel.get('mcp_tool', '')}` |"
            )
        lines.append("")
    lines.extend(
        [
            "",
            "## Source Artifacts",
            "",
            f"- Markdown index: `{source_artifacts.get('source_artifacts_path', '')}`",
            f"- JSON index: `{source_artifacts.get('source_artifacts_json_path', '')}`",
            f"- MCP args: `{source_artifacts.get('mcp_args_path', '')}`",
            f"- Available: `{source_summary.get('available_count', 0)}` / `{source_summary.get('artifact_count', 0)}`",
            f"- Missing: `{source_summary.get('missing_count', 0)}`",
            f"- Tools: `{', '.join(str(value) for value in source_summary.get('tools') or [])}`",
            f"- Next source-artifact action: `{source_next.get('key', '')}` / {source_next.get('label', '')}",
            "",
        ]
    )
    if missing_artifacts:
        lines.extend(["Missing source-artifact samples:", ""])
        for item in missing_artifacts[:5]:
            if not isinstance(item, dict):
                continue
            lines.append(f"- {item.get('label') or item.get('key')}: `{item.get('path', '')}`")
        lines.append("")
    lines.extend(
        [
            "## Review Targets",
            "",
            f"- Total open targets: `{targets.get('total_open', 0)}`",
            f"- Filtered open targets: `{targets.get('filtered_open', targets.get('total_open', 0))}`",
            f"- Listed targets: `{len(items)}`",
            f"- Offset: `{targets.get('offset', 0)}`",
            f"- Limit: `{targets.get('limit', 0)}`",
            f"- Reason filter: `{targets.get('reason_filter', '')}`",
            "",
        ]
    )
    ocr_empty_items = [item for item in items if "ocr_text_empty" in (item.get("reasons") or [])]
    if ocr_empty_items:
        lines.extend(
            [
                "### OCR Empty Targets",
                "",
                "ebook pipeline returned no meaningful text for these evidence frames. Review the frame directly, then either keep the image as evidence, add corrected visual text, or send it to multimodal supplementation.",
                "",
                "| Index | Time | Frame | Suggested action |",
                "|---:|---|---|---|",
            ]
        )
        for item in ocr_empty_items:
            ebook_result = item.get("ebook_result") if isinstance(item.get("ebook_result"), dict) else {}
            frame_path = ebook_result.get("frame_path") or (item.get("asset_paths") or [""])[0]
            lines.append(
                f"| {item.get('index', '')} | {item.get('time_range', '')} | `{_markdown_cell(str(frame_path or ''))}` | {_markdown_cell(str(item.get('fallback_suggestion') or ''))} |"
            )
        lines.append("")
    if items:
        lines.extend(["| Index | Time | Reasons | Issues | Evidence |", "|---:|---|---|---|---|"])
        for item in items:
            reasons = ", ".join(str(value) for value in item.get("reasons") or [])
            issues = ", ".join(str(value) for value in item.get("quality_issues") or [])
            evidence = _markdown_cell(str(item.get("evidence_excerpt") or ""))
            lines.append(
                f"| {item.get('index', '')} | {item.get('time_range', '')} | {_markdown_cell(reasons)} | {_markdown_cell(issues)} | {evidence} |"
            )
    else:
        lines.append("No open review targets detected.")
    return "\n".join(lines).rstrip() + "\n"


def render_review_pack_markdown(pack: dict[str, Any]) -> str:
    summary = pack.get("summary") if isinstance(pack.get("summary"), dict) else {}
    closure = pack.get("closure_status") if isinstance(pack.get("closure_status"), dict) else {}
    closure_summary = closure.get("summary") if isinstance(closure.get("summary"), dict) else {}
    lines = [
        "# Review Pack",
        "",
        f"- Bundle: `{pack.get('bundle_dir', '')}`",
        f"- Open targets: `{summary.get('open_targets', 0)}`",
        f"- Listed targets: `{summary.get('listed_targets', 0)}`",
        f"- Closed: `{closure_summary.get('closed', 0)}`",
        f"- Todo JSON: `{pack.get('todo_path', '')}`",
        f"- Closure status: `{pack.get('closure_status_path', '')}`",
        "",
        "## Groups",
        "",
    ]
    groups = pack.get("groups") if isinstance(pack.get("groups"), list) else []
    if not groups:
        lines.append("当前没有待人工复核条目。")
        return "\n".join(lines).rstrip() + "\n"
    for group in groups:
        if not isinstance(group, dict):
            continue
        lines.extend([f"### {group.get('label', group.get('key', 'unknown'))}", "", f"- Count: `{group.get('count', 0)}`", ""])
        lines.extend(["| Index | Time | Suggested | Reason | Alignment | Tile review | Transcript | Visual/OCR | Evidence |", "|---:|---|---|---|---|---|---|---|---|"])
        for item in group.get("items") or []:
            if not isinstance(item, dict):
                continue
            evidence = "; ".join(str(path) for path in item.get("evidence_paths") or item.get("asset_paths") or [])
            if item.get("crop_paths"):
                evidence = "; ".join([evidence, *[str(path) for path in item.get("crop_paths") or []]]).strip("; ")
            lines.append(
                "| {index} | `{time}` | `{status}` | {reason} | {alignment} | {tile_review} | {transcript} | {visual} | {evidence} |".format(
                    index=item.get("index", ""),
                    time=item.get("time_range", ""),
                    status=item.get("suggested_status", ""),
                    reason=_markdown_cell(", ".join(str(value) for value in item.get("reasons") or [])),
                    alignment=_markdown_cell(_timeline_alignment_cell(item.get("timeline_alignment") if isinstance(item.get("timeline_alignment"), dict) else {})),
                    tile_review=_markdown_cell(_tile_review_summary(item)),
                    transcript=_markdown_cell(str(item.get("transcript_excerpt") or "")),
                    visual=_markdown_cell(str(item.get("visual_text_excerpt") or item.get("model_output_excerpt") or "")),
                    evidence=_markdown_cell(evidence),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"



def _timeline_alignment_cell(alignment: dict[str, Any]) -> str:
    issues = [str(value) for value in alignment.get("issues") or []]
    if not issues:
        return ""
    return (
        f"issues={','.join(issues)}; "
        f"review_start={alignment.get('review_start', '-')}; "
        f"asr_start={alignment.get('asr_first_start', '-')}; "
        f"suggested={alignment.get('suggested_review_start', '-')}"
    )

def render_review_closure_status_markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    next_batch = result.get("next_batch") if isinstance(result.get("next_batch"), dict) else {}
    lines = [
        "# Review Closure Status",
        "",
        f"- Bundle: `{result.get('bundle_dir', '')}`",
        f"- Open: `{summary.get('open', 0)}`",
        f"- Closed: `{summary.get('closed', 0)}`",
        f"- Imported: `{summary.get('imported', 0)}`",
        f"- Invalid rows: `{summary.get('invalid', 0)}`",
        f"- Warnings: `{summary.get('warnings', 0)}`",
        f"- Next batch command: `{_markdown_cell(str(next_batch.get('command') or ''))}`",
        "",
        "## Open By Reason",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    open_by_reason = result.get("open_by_reason") if isinstance(result.get("open_by_reason"), dict) else {}
    if open_by_reason:
        for key, count in open_by_reason.items():
            lines.append(f"| `{key}` | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.extend(["", "## Closed By Reason", "", "| Reason | Count |", "|---|---:|"])
    closed_by_reason = result.get("closed_by_reason") if isinstance(result.get("closed_by_reason"), dict) else {}
    if closed_by_reason:
        for key, count in closed_by_reason.items():
            lines.append(f"| `{key}` | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.extend(["", "## Closed By Status", "", "| Status | Count |", "|---|---:|"])
    closed_by_status = result.get("closed_by_status") if isinstance(result.get("closed_by_status"), dict) else {}
    if closed_by_status:
        for key, count in closed_by_status.items():
            lines.append(f"| `{key}` | {count} |")
    else:
        lines.append("| none | 0 |")
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    errors = validation.get("errors") if isinstance(validation.get("errors"), list) else []
    if errors:
        lines.extend(["", "## Invalid Review Rows", "", "| Row | Index | Key | Message |", "|---:|---:|---|---|"])
        for row in errors[:50]:
            if isinstance(row, dict):
                lines.append(
                    f"| {row.get('row_number', '')} | {row.get('timeline_index', '')} | `{row.get('key', '')}` | {_markdown_cell(str(row.get('message') or ''))} |"
                )
    return "\n".join(lines).rstrip() + "\n"


def render_review_fill_guide_markdown(session: dict[str, Any], review_template: dict[str, Any]) -> str:
    rows = [row for row in review_template.get("reviews") or [] if isinstance(row, dict)]
    grouped = _group_review_rows(rows)
    lines = [
        "# Review Notes 填写指南",
        "",
        f"- Bundle: `{session.get('bundle_dir', '')}`",
        f"- Review UI: {session.get('review_file_url', '')}",
        f"- Template JSON: `{session.get('review_notes_template_path', '')}`",
        f"- Write reviewed JSON to: `{session.get('review_notes_path', '')}`",
        "",
        "## 填写规则",
        "",
        "- 不要删除 `timeline_index`、`time_range`、`route`、`evidence_frame_paths`。",
        "- `suggested_status=corrected_visual_understanding` 时，必须填写非空 `corrected_visual_understanding`。",
        "- `suggested_status=corrected_temporal_visual_understanding` 时，必须填写非空 `corrected_temporal_visual_understanding`。",
        "- 如果画面没有额外视觉信息，可以把 `status` 改为 `accepted`，并在 `comment` 说明“无额外画面信息”。",
        "- 如果必须保留图片但不适合降维成文字，把 `status` 设为 `keep_image`，并说明保留原因。",
        "- 如果缺口客观存在但当前可以接受，把 `status` 设为 `accepted_known_gap`，并写清保留缺口的原因。",
        "- 如果需要重新跑裁剪/OCR，把 `status` 设为 `needs_rerun_ocr`；这不会清除 OCR blocker。",
        "- 导入前运行 `validate-review-notes`，确认没有空修正、重复 index 或证据路径问题。",
        "",
        "## 建议字段模板",
        "",
        "### corrected_visual_understanding",
        "",
        "```json",
        '{ "objects": [], "actions": [], "interface_state": "", "spatial_relations": [], "instructor_focus": "", "non_text_information": [], "keep_image_reason": "", "confidence": 0.0, "evidence_frame_paths": [] }',
        "```",
        "",
        "### corrected_temporal_visual_understanding",
        "",
        "```json",
        '{ "event_sequence": [], "state_changes": [], "operation_steps": [], "causal_links": [], "possible_missing_points": [], "confidence": 0.0, "evidence_frame_paths": [] }',
        "```",
        "",
    ]
    for title, key in [
        ("连续片段待补", "temporal"),
        ("单帧视觉待补", "semantic"),
        ("其他人工确认", "accepted"),
    ]:
        section_rows = grouped.get(key, [])
        lines.extend([f"## {title}", "", f"- Count: `{len(section_rows)}`", ""])
        if not section_rows:
            lines.append("- 无")
            lines.append("")
            continue
        for row in section_rows:
            lines.extend(_render_fill_guide_row(row))
    return "\n".join(lines).rstrip() + "\n"


def _group_review_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {"temporal": [], "semantic": [], "accepted": []}
    for row in rows:
        status = str(row.get("suggested_status") or "")
        if status == "corrected_temporal_visual_understanding":
            grouped["temporal"].append(row)
        elif status == "corrected_visual_understanding":
            grouped["semantic"].append(row)
        else:
            grouped["accepted"].append(row)
    return grouped


def _render_fill_guide_row(row: dict[str, Any]) -> list[str]:
    index = row.get("timeline_index", "")
    evidence_paths = [str(path) for path in row.get("evidence_frame_paths") or [] if str(path)]
    status = str(row.get("suggested_status") or "accepted")
    lines = [
        f"### Index {index} `{row.get('time_range', '')}`",
        "",
        f"- Route: `{row.get('route', '')}`",
        f"- Suggested status: `{status}`",
        f"- Reason: `{row.get('reason', '')}`",
        f"- Evidence: `{', '.join(evidence_paths)}`",
        f"- Transcript: {row.get('transcript_excerpt', '')}",
        "",
    ]
    if evidence_paths:
        first = evidence_paths[0].replace("\\", "/")
        lines.extend([f"![evidence]({first})", ""])
    sample = {
        "timeline_index": row.get("timeline_index", 0),
        "status": status,
        "tags": row.get("tags") or [],
        "comment": row.get("comment") or "",
        "corrected_transcript": row.get("corrected_transcript") or "",
        "corrected_visual_text": row.get("corrected_visual_text") or "",
        "corrected_visual_understanding": row.get("corrected_visual_understanding") or {},
        "corrected_temporal_visual_understanding": row.get("corrected_temporal_visual_understanding") or {},
        "evidence_frame_paths": evidence_paths,
        "reviewed_at": row.get("reviewed_at") or "",
    }
    lines.extend(["```json", _compact_json(sample), "```", ""])
    return lines


def _knowledge_coverage_summary(report: dict[str, Any]) -> dict[str, Any]:
    channels = [item for item in report.get("channels") or [] if isinstance(item, dict)]
    return {
        "schema": report.get("schema", ""),
        "status": report.get("status", "unknown"),
        "timeline_items": report.get("timeline_items", 0),
        "channel_count": len(channels),
        "blocked_count": len([item for item in channels if str(item.get("status") or "") == "blocked"]),
        "weak_count": len([item for item in channels if str(item.get("status") or "") == "weak"]),
    }



def _timeline_alignment_by_index(root: Path) -> dict[int, dict[str, Any]]:
    path = root / "timeline-alignment-audit.json"
    if not path.exists():
        return {}
    try:
        data = read_json(path)
    except Exception:
        return {}
    rows = data.get("items") if isinstance(data, dict) else []
    result: dict[int, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return result
    for row in rows:
        if not isinstance(row, dict) or not row.get("issues"):
            continue
        index = _int(row.get("index"))
        if index:
            result[index] = row
    return result


def _compact_timeline_alignment(alignment: dict[str, Any]) -> dict[str, Any]:
    return {
        "issues": [str(value) for value in alignment.get("issues") or []],
        "review_start": _number(alignment.get("review_start")),
        "asr_first_start": _number(alignment.get("asr_first_start")),
        "frame_time": _number(alignment.get("frame_time")),
        "tagger_times": [_number(value) for value in alignment.get("tagger_times") or []],
        "asr_overlap_count": _int(alignment.get("asr_overlap_count")),
        "asr_excerpt": _meaningful_excerpt_text(alignment.get("asr_excerpt")),
        "suggested_review_start": _number(alignment.get("asr_first_start")),
        "suggestion": "Preview only: verify against video, then use ASR segment start as review_start if reliable.",
    }

def _review_targets(
    timeline: list[dict[str, Any]],
    *,
    readiness: dict[str, Any],
    manifest: dict[str, Any] | None = None,
    timeline_alignment: dict[int, dict[str, Any]] | None = None,
    bundle_dir: str | Path | None = None,
    limit: int = 30,
    offset: int = 0,
    reason: str = "",
    include_closed: bool = False,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    from .media_applicability import project_timeline_for_media, resolve_media_applicability
    applicability = resolve_media_applicability(bundle_dir, manifest or {}, timeline)
    timeline = project_timeline_for_media(timeline, applicability)
    ocr_empty_by_index = (_ocr_empty_results_by_index(manifest or {})
                          if applicability['visual_required'] else {})
    alignment_by_index = timeline_alignment or {}
    for item in timeline:
        index = _int(item.get("index"))
        ocr_empty_result = ocr_empty_by_index.get(index)
        alignment = alignment_by_index.get(index) or {}
        reasons = _target_reasons(item, ocr_empty=bool(ocr_empty_result), timeline_alignment=alignment)
        reviewed = _is_reviewed(item)
        if include_closed and reviewed and not reasons:
            reasons = ["closed"]
        if not reasons:
            continue
        crop_paths = _crop_paths(item)
        asset_paths = _asset_paths(item)
        target = {
            "index": index,
            "start": _number(item.get("start")),
            "end": _number(item.get("end")),
            "time_range": _time_range(item),
            "visual_route": str(item.get("visual_route") or ""),
            "review_status": _review_status(item),
            "closed": reviewed,
            "reasons": reasons,
            "quality_issues": [str(issue) for issue in item.get("quality_issues") or []],
            "material_types": [str(value) for value in item.get("material_types") or []],
            "source_segment_ids": [str(value) for value in item.get("source_segment_ids") or []],
            "evidence_excerpt": _evidence_excerpt(item),
            "transcript_excerpt": _meaningful_excerpt_text(item.get("transcript")),
            "visual_text_excerpt": _meaningful_excerpt_text(item.get("visual_text")),
            "model_output_excerpt": _model_output_excerpt(item),
            "asset_paths": asset_paths,
            "crop_paths": crop_paths,
            "evidence_paths": _dedupe([*asset_paths, *crop_paths, *_tile_evidence_paths(item)]),
            "tile_review_targets": _tile_review_targets(item),
            "suggested_filter": _suggested_filter(reasons),
            "suggested_status": "",
            "suggested_action": "",
            "priority": _target_priority(reasons),
        }
        if alignment:
            target["timeline_alignment"] = _compact_timeline_alignment(alignment)
        if ocr_empty_result:
            target["ocr_empty"] = True
            target["ebook_blocker"] = "ocr_text_empty"
            target["ebook_result"] = {
                "index": ocr_empty_result.get("index"),
                "frame_path": ocr_empty_result.get("frame_path") or ocr_empty_result.get("source_path") or "",
                "artifact_path": ocr_empty_result.get("artifact_path") or ocr_empty_result.get("source_artifact_path") or "",
                "status": ocr_empty_result.get("status") or "failed",
                "ok": bool(ocr_empty_result.get("ok")),
            }
            target["fallback_suggestion"] = (
                "ebook pipeline returned no meaningful text; mark as accepted/keep_image "
                "after human review, or send the evidence frame to multimodal supplementation."
            )
        items.append(target)
        target["suggested_status"] = _suggested_review_status(target)
        target["suggested_action"] = _suggested_review_action(target)
    if bundle_dir is not None:
        bundle_path = Path(bundle_dir)
        items.extend(_transcript_arbitration_review_targets(bundle_path))
        items.extend(_transcript_semantic_correction_review_targets(bundle_path))
    items.sort(key=lambda row: (row["priority"], row["index"] or 0, row["start"]))
    open_items = [item for item in items if not item.get("closed")]
    by_reason: dict[str, int] = {}
    for target in open_items:
        for target_reason in target["reasons"]:
            by_reason[target_reason] = by_reason.get(target_reason, 0) + 1
    reason_filter = str(reason or "").strip()
    filtered_items = items
    if reason_filter:
        filters = {part.strip() for part in reason_filter.split(",") if part.strip()}
        filtered_items = [item for item in items if filters & {str(value) for value in item.get("reasons") or []}]
    safe_offset = max(int(offset or 0), 0)
    safe_limit = int(limit or 0)
    if safe_limit <= 0:
        listed_items = filtered_items[safe_offset:]
    else:
        listed_items = filtered_items[safe_offset : safe_offset + safe_limit]
    samples = readiness.get("samples") if isinstance(readiness.get("samples"), dict) else {}
    return {
        "schema": "lecture_review_targets.v1",
        "media_applicability": applicability,
        "total_items": len(items),
        "total_open": len(open_items),
        "filtered_open": len(filtered_items),
        "listed_count": len(listed_items),
        "limit": safe_limit,
        "offset": safe_offset,
        "reason_filter": reason_filter,
        "by_reason": dict(sorted(by_reason.items())),
        "readiness_samples": samples,
        "items": listed_items,
    }



def _transcript_arbitration_review_targets(root: Path) -> list[dict[str, Any]]:
    manifest = _read_manifest(root)
    candidates = []
    raw = str(manifest.get("transcript_source_arbitration_json") or "").strip()
    if raw:
        candidates.append(_resolve_bundle_path(root, raw))
    candidates.append(root / "transcript-source-arbitration.json")
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return []
    payload = read_json(path)
    if not isinstance(payload, dict):
        return []
    rows = payload.get("review_rows") if isinstance(payload.get("review_rows"), list) else []
    human_corrected_segments, human_corrected_path = _human_corrected_transcript_segments(root, manifest)
    targets: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        reasons = _transcript_arbitration_reasons(row)
        if not reasons:
            reasons = ["transcript_source_conflict"]
        alternatives = _transcript_alternatives(row)
        index = _int(row.get("index"))
        human_text = human_corrected_segments.get(index, "") if index is not None else ""
        closed = bool(human_text)
        evidence_paths = [str(path)]
        if human_corrected_path is not None:
            evidence_paths.append(str(human_corrected_path))
        target = {
            "target_type": "transcript_arbitration",
            "index": index,
            "timeline_index": None,
            "transcript_segment_index": index,
            "start": _number(row.get("start")),
            "end": _number(row.get("end")),
            "time_range": f"{format_timestamp(_number(row.get('start')))} - {format_timestamp(_number(row.get('end')))}",
            "visual_route": "transcript",
            "review_status": "corrected_transcript" if closed else "",
            "closed": closed,
            "reasons": reasons,
            "quality_issues": [] if closed else reasons,
            "material_types": ["transcript"],
            "source_segment_ids": [],
            "evidence_excerpt": _meaningful_excerpt_text(human_text or row.get("corrected_text") or row.get("text") or row.get("original_text")),
            "transcript_excerpt": _meaningful_excerpt_text(human_text or row.get("corrected_text") or row.get("text") or row.get("original_text")),
            "visual_text_excerpt": "",
            "model_output_excerpt": "",
            "asset_paths": [],
            "crop_paths": [],
            "evidence_paths": _dedupe(evidence_paths),
            "tile_review_targets": [],
            "transcript_arbitration": {
                "segment_index": index,
                "confidence": row.get("confidence"),
                "review_reason": row.get("review_reason") or "",
                "chosen_source": row.get("chosen_source") or "",
                "chosen_source_type": row.get("chosen_source_type") or "",
                "original_text": row.get("original_text") or row.get("raw_text") or "",
                "corrected_text": row.get("corrected_text") or row.get("text") or "",
                "human_corrected_text": human_text,
                "human_corrected_transcript_path": str(human_corrected_path) if human_corrected_path is not None else "",
                "alternatives": alternatives,
                "report_path": str(path),
            },
            "suggested_filter": "transcript_arbitration",
            "suggested_status": "corrected_transcript",
            "suggested_action": "已通过人工纠正版转写关闭。" if closed else "打开转录编辑器核对候选字幕/ASR来源，导出 transcript-edits.json 后用 apply-transcript-edits 导入。",
            "priority": 52,
        }
        targets.append(target)
    return targets



def _transcript_semantic_correction_review_targets(root: Path) -> list[dict[str, Any]]:
    candidates = [root / "transcript-semantic-correction-review.json"]
    manifest = _read_manifest(root)
    raw = str(manifest.get("transcript_semantic_correction_review_json") or "").strip()
    if raw:
        candidates.insert(0, _resolve_bundle_path(root, raw))
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return []
    payload = read_json(path)
    if not isinstance(payload, dict):
        return []
    rows = payload.get("items") if isinstance(payload.get("items"), list) else []
    targets: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        reasons = [str(item) for item in row.get("quality_issues") or row.get("reject_reasons") or [] if str(item)]
        if not reasons:
            reasons = ["semantic_correction_needs_review"]
        timeline_indexes = [item for item in row.get("timeline_indexes") or [] if item is not None]
        timeline_index = _int(timeline_indexes[0]) if timeline_indexes else None
        target = {
            "target_type": "transcript_semantic_correction",
            "index": timeline_index,
            "timeline_index": timeline_index,
            "transcript_segment_index": _int(row.get("segment_index")),
            "candidate_id": row.get("candidate_id") or "",
            "start": _number(row.get("start")),
            "end": _number(row.get("end")),
            "time_range": row.get("time_range") or f"{format_timestamp(_number(row.get('start')))} - {format_timestamp(_number(row.get('end')))}",
            "visual_route": "transcript_semantic_correction",
            "review_status": "",
            "closed": False,
            "reasons": reasons,
            "quality_issues": reasons,
            "material_types": ["transcript"],
            "source_segment_ids": [],
            "evidence_excerpt": _meaningful_excerpt_text(row.get("evidence_excerpt") or row.get("context_text") or row.get("original_text")),
            "transcript_excerpt": _meaningful_excerpt_text(row.get("context_text") or row.get("original_text")),
            "visual_text_excerpt": "",
            "model_output_excerpt": _meaningful_excerpt_text(row.get("semantic_rationale")),
            "asset_paths": [],
            "crop_paths": [],
            "evidence_paths": [str(path)],
            "tile_review_targets": [],
            "transcript_semantic_correction": {
                "candidate_id": row.get("candidate_id"),
                "correction_type": row.get("correction_type"),
                "original_text": row.get("original_text") or "",
                "suggested_text": row.get("suggested_text") or "",
                "confidence": row.get("confidence"),
                "reject_reasons": row.get("reject_reasons") or [],
                "evidence_ids": row.get("evidence_ids") or [],
                "report_path": str(path),
            },
            "suggested_filter": "transcript_semantic_correction",
            "suggested_status": row.get("suggested_status") or "corrected_transcript",
            "suggested_action": row.get("suggested_action") or "人工核对语义纠错候选，确认后重新导入纠错结果或保留原文。",
            "priority": 50,
        }
        targets.append(target)
    return targets

def _human_corrected_transcript_segments(root: Path, manifest: dict[str, Any]) -> tuple[dict[int, str], Path | None]:
    candidates: list[Path] = []
    raw = str(manifest.get("human_corrected_transcript_json") or "").strip()
    if raw:
        candidates.append(_resolve_bundle_path(root, raw))
    candidates.append(root / "human-corrected-transcript.json")
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return {}, None
    payload = read_json(path)
    if not isinstance(payload, dict) or str(payload.get("source") or "") != "human_transcript_editor":
        return {}, None
    rows = payload.get("segments") if isinstance(payload.get("segments"), list) else []
    segments: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        index = _int(row.get("index"))
        text = str(row.get("corrected_text") or row.get("text") or "").strip()
        if index is not None and text:
            segments[index] = text
    return segments, path

def _transcript_arbitration_reasons(row: dict[str, Any]) -> list[str]:
    reason = str(row.get("review_reason") or "").strip()
    reasons = ["transcript_source_conflict"]
    if reason == "low_arbitration_confidence":
        reasons.append("low_arbitration_confidence")
    elif reason and reason not in reasons:
        reasons.append(reason)
    return reasons


def _transcript_alternatives(row: dict[str, Any]) -> list[dict[str, Any]]:
    values = row.get("alternatives") if isinstance(row.get("alternatives"), list) else []
    alternatives: list[dict[str, Any]] = []
    for value in values[:8]:
        if not isinstance(value, dict):
            continue
        alternatives.append(
            {
                "source_id": str(value.get("source_id") or ""),
                "source_type": str(value.get("source_type") or ""),
                "text": str(value.get("text") or value.get("raw_text") or ""),
                "score": value.get("score"),
                "overlap": value.get("overlap"),
                "similarity_to_base": value.get("similarity_to_base"),
            }
        )
    return alternatives


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.exists():
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def _resolve_bundle_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path

def _target_reasons(item: dict[str, Any], *, ocr_empty: bool = False, timeline_alignment: dict[str, Any] | None = None) -> list[str]:
    reasons: list[str] = []
    issues = {str(issue) for issue in item.get("quality_issues") or []}
    material_types = {str(value) for value in item.get("material_types") or []}
    reviewed = _is_reviewed(item)
    if reviewed:
        return []
    alignment_issues = [str(value) for value in (timeline_alignment or {}).get("issues") or []]
    if alignment_issues:
        reasons.append("timeline_alignment_issue")
    if ocr_empty and not reviewed:
        reasons.append("ocr_text_empty")
    if not reviewed:
        reasons.append("pending_review")
    if issues:
        reasons.append("quality_issues")
    if "tile_result_needs_review" in issues or _tile_review_targets(item):
        reasons.append("tile_result_needs_review")
    if "missing_visual_text" in issues:
        reasons.append("missing_visual_text")
    if material_types & STRUCTURED_TYPES and not reviewed:
        reasons.append("pending_structured")
    if "structured_visual_without_structure" in issues:
        reasons.append("structure_gap")
    if "screen_text_low_confidence" in issues:
        reasons.append("screen_text_low_confidence")
    if issues & {"keep_image_without_frame", "structured_visual_without_frame", "missing_frame"}:
        reasons.append("frame_gap")
    if _has_asset_gap(item):
        reasons.append("asset_gap")
    route = str(item.get("visual_route") or "")
    if route in {"semantic_frame", "mixed"} and not reviewed and not _has_visual_understanding(item):
        reasons.append("semantic_frame_without_analysis")
    if route in {"temporal_sequence", "mixed"} and not reviewed and not _has_temporal_understanding(item):
        reasons.append("temporal_sequence_without_analysis")
    return reasons


def _is_reviewed(item: dict[str, Any]) -> bool:
    return _review_status(item).lower() in ACCEPTED_REVIEW_STATUSES


def _review_status(item: dict[str, Any]) -> str:
    human_review = item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
    return str(item.get("review_status") or human_review.get("status") or "")


def _has_visual_understanding(item: dict[str, Any]) -> bool:
    for key in ("visual_understanding", "human_corrected_visual_understanding"):
        value = item.get(key)
        if isinstance(value, dict) and value and not value.get("parse_failed"):
            return True
    human_review = item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
    corrected = human_review.get("corrected_visual_understanding")
    return isinstance(corrected, dict) and bool(corrected)


def _has_temporal_understanding(item: dict[str, Any]) -> bool:
    for key in ("temporal_visual_understanding", "human_corrected_temporal_visual_understanding"):
        value = item.get(key)
        if isinstance(value, dict) and value and not value.get("parse_failed"):
            return True
    human_review = item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
    corrected = human_review.get("corrected_temporal_visual_understanding")
    return isinstance(corrected, dict) and bool(corrected)


def _has_asset_gap(item: dict[str, Any]) -> bool:
    for asset in item.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        if str(asset.get("copied") or "").lower() == "false" or asset.get("exists") is False:
            return True
    return False


def _tile_review_targets(item: dict[str, Any]) -> list[dict[str, Any]]:
    values = item.get("tile_review_targets")
    if not isinstance(values, list):
        return []
    targets: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, dict):
            targets.append(value)
    return targets



def _tile_corrections_template(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        if not isinstance(target, dict):
            continue
        rows.append(
            {
                "tile_id": str(target.get("tile_id") or "").strip(),
                "status": "needs_review",
                "corrected_text": "",
                "comment": "",
                "confidence": target.get("confidence"),
                "reasons": [str(reason) for reason in target.get("reasons") or []],
                "evidence_path": str(target.get("evidence_path") or target.get("tile_path") or "").strip(),
            }
        )
    return rows

def _tile_evidence_paths(item: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for target in _tile_review_targets(item):
        path = str(target.get("evidence_path") or target.get("tile_path") or "").strip()
        if path:
            paths.append(path)
    return _dedupe(paths)


def _tile_review_summary(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for target in _tile_review_targets(item):
        tile_id = str(target.get("tile_id") or "").strip()
        confidence = target.get("confidence")
        reasons = ",".join(str(reason) for reason in target.get("reasons") or [])
        evidence = str(target.get("evidence_path") or target.get("tile_path") or "").strip()
        bits = []
        if tile_id:
            bits.append(f"tile={tile_id}")
        if confidence is not None:
            bits.append(f"conf={confidence}")
        if reasons:
            bits.append(f"reasons={reasons}")
        if evidence:
            bits.append(f"evidence={evidence}")
        if bits:
            parts.append("; ".join(bits))
    return " | ".join(parts)


def _target_priority(reasons: list[str]) -> int:
    order = {
        "timeline_alignment_issue": 8,
        "asset_gap": 10,
        "frame_gap": 20,
        "ocr_text_empty": 25,
        "tile_result_needs_review": 26,
        "screen_text_low_confidence": 27,
        "missing_visual_text": 28,
        "structure_gap": 30,
        "pending_structured": 40,
        "temporal_sequence_without_analysis": 45,
        "semantic_frame_without_analysis": 46,
        "quality_issues": 50,
        "pending_review": 60,
    }
    return min((order.get(reason, 90) for reason in reasons), default=90)


def _suggested_filter(reasons: list[str]) -> str:
    if "timeline_alignment_issue" in reasons:
        return "timeline_alignment"
    if "asset_gap" in reasons or "frame_gap" in reasons:
        return "missing_frame"
    if "ocr_text_empty" in reasons:
        return "ocr_empty"
    if "tile_result_needs_review" in reasons:
        return "tile_review"
    if "screen_text_low_confidence" in reasons:
        return "screen_text"
    if "structure_gap" in reasons or "pending_structured" in reasons:
        return "structured"
    if "temporal_sequence_without_analysis" in reasons:
        return "temporal_visual"
    if "semantic_frame_without_analysis" in reasons:
        return "semantic_visual"
    if "transcript_source_conflict" in reasons or "low_arbitration_confidence" in reasons:
        return "transcript_arbitration"
    if "quality_issues" in reasons:
        return "risk"
    return "pending"


def _review_notes_template(review_targets: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in review_targets.get("items") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("target_type") or "") == "transcript_arbitration":
            continue
        rows.append(
            {
                "timeline_index": item.get("index", 0),
                "time_range": item.get("time_range", ""),
                "route": item.get("visual_route", ""),
                "reason": ", ".join(str(value) for value in item.get("reasons") or []),
                "suggested_status": _suggested_review_status(item),
                "evidence_frame_paths": item.get("evidence_paths") or item.get("asset_paths") or [],
                "transcript_excerpt": item.get("transcript_excerpt", ""),
                "visual_text_excerpt": item.get("visual_text_excerpt", ""),
                "model_output_excerpt": item.get("model_output_excerpt", ""),
                "timeline_alignment": item.get("timeline_alignment") or {},
                "tile_review_targets": item.get("tile_review_targets") or [],
                "tile_corrections": _tile_corrections_template(item.get("tile_review_targets") if isinstance(item.get("tile_review_targets"), list) else []),
                "corrected_review_start": "",
                "tags": [],
                "comment": "",
                "corrected_transcript": "",
                "corrected_visual_text": "",
                "corrected_visual_understanding": {},
                "corrected_temporal_visual_understanding": {},
                "reviewed_at": "",
            }
        )
    return {"schema": REVIEW_NOTES_SCHEMA, "created_at": now_iso(), "reviews": rows}


def _build_review_pack(
    *,
    root: Path,
    manifest: dict[str, Any],
    review_targets: dict[str, Any],
    review_template: dict[str, Any],
    closure_status: dict[str, Any],
    group_by: str,
    output_prefix: str,
    include_closed: bool,
) -> dict[str, Any]:
    items = [item for item in review_targets.get("items") or [] if isinstance(item, dict)]
    groups = _review_pack_groups(items, group_by=group_by)
    return {
        "schema": "lecture_review_pack.v1",
        "created_at": now_iso(),
        "bundle_dir": str(root),
        "title": str(manifest.get("title") or root.name),
        "output_prefix": output_prefix,
        "group_by": group_by,
        "include_closed": bool(include_closed),
        "summary": {
            "open_targets": int(review_targets.get("total_open") or 0),
            "filtered_open": int(review_targets.get("filtered_open") or 0),
            "listed_targets": len(items),
            "group_count": len(groups),
        },
        "review_targets": review_targets,
        "review_template": review_template,
        "groups": groups,
        "closure_status": closure_status,
        "todo_path": str(root / "review-notes.todo.json"),
        "closure_status_path": str(root / "review-closure-status.md"),
    }


def _review_pack_groups(items: list[dict[str, Any]], *, group_by: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        key = _review_pack_group_key(item, group_by=group_by)
        buckets.setdefault(key, []).append(item)
    order = ["timeline_alignment_issue", "ocr_text_empty", "tile_result_needs_review", "missing_visual_text", "keep_image", "semantic_or_temporal", "transcript_arbitration", "transcript_semantic_correction", "accepted_known_gap", "quality_issues", "pending_review", "closed", "other"]
    groups = []
    for key in sorted(buckets, key=lambda value: (order.index(value) if value in order else len(order), value)):
        rows = sorted(buckets[key], key=lambda row: (int(row.get("priority") or 90), int(row.get("index") or 0)))
        groups.append({"key": key, "label": _review_group_label(key), "count": len(rows), "items": rows})
    return groups


def _review_pack_group_key(item: dict[str, Any], *, group_by: str) -> str:
    if group_by not in {"reason", "suggested_status", "route"}:
        group_by = "reason"
    if group_by == "suggested_status":
        return str(item.get("suggested_status") or "other")
    if group_by == "route":
        return str(item.get("visual_route") or "unknown")
    reasons = {str(value) for value in item.get("reasons") or []}
    status = str(item.get("review_status") or "")
    suggested = str(item.get("suggested_status") or "")
    if "timeline_alignment_issue" in reasons:
        return "timeline_alignment_issue"
    if "ocr_text_empty" in reasons:
        return "ocr_text_empty"
    if "tile_result_needs_review" in reasons:
        return "tile_result_needs_review"
    if "missing_visual_text" in reasons or "screen_text_low_confidence" in reasons or "structure_gap" in reasons:
        return "missing_visual_text"
    if suggested == "keep_image" or status == "keep_image":
        return "keep_image"
    if reasons & {"semantic_frame_without_analysis", "temporal_sequence_without_analysis"}:
        return "semantic_or_temporal"
    if status == "accepted_known_gap":
        return "accepted_known_gap"
    if "quality_issues" in reasons:
        return "quality_issues"
    if "pending_review" in reasons:
        return "pending_review"
    if "closed" in reasons:
        return "closed"
    return "other"


def _review_group_label(key: str) -> str:
    return {
        "timeline_alignment_issue": "时间轴错位",
        "ocr_text_empty": "OCR 失败/空结果",
        "tile_result_needs_review": "Tile 结果待复核",
        "missing_visual_text": "缺屏幕文字/图文结构",
        "keep_image": "需要保留图片",
        "semantic_or_temporal": "语义/连续视觉待确认",
        "accepted_known_gap": "已接受已知缺口",
        "transcript_arbitration": "字幕/ASR 仲裁待复核",
        "quality_issues": "其他质量风险",
        "pending_review": "普通待审核",
        "closed": "已关闭",
        "other": "其他",
    }.get(key, key)


def _suggested_review_status(item: dict[str, Any]) -> str:
    reasons = set(str(value) for value in item.get("reasons") or [])
    if "timeline_alignment_issue" in reasons:
        return "corrected_review_start"
    if "ocr_text_empty" in reasons:
        return "keep_image"
    if "tile_result_needs_review" in reasons:
        return "corrected_visual_text"
    if "screen_text_low_confidence" in reasons:
        return "corrected_visual_text"
    if "temporal_sequence_without_analysis" in reasons:
        return "corrected_temporal_visual_understanding"
    if "semantic_frame_without_analysis" in reasons:
        return "corrected_visual_understanding"
    return "accepted"


def _suggested_review_action(item: dict[str, Any]) -> str:
    status = str(item.get("suggested_status") or _suggested_review_status(item))
    return {
        "keep_image": "人工看图后保留截图证据，必要时补一句说明。",
        "corrected_visual_text": "人工补齐或修正画面文字/OCR；如是 tile 低置信结果，优先核对 tile 证据图。",
        "corrected_visual_understanding": "人工补充对象、动作、界面状态或空间关系。",
        "corrected_temporal_visual_understanding": "人工补充事件序列、状态变化或操作步骤。",
        "needs_rerun_ocr": "重新裁剪或换 OCR 工具后再导入。",
        "accepted_known_gap": "确认缺口可接受，并说明原因。",
        "needs_fix": "打开视频核对 ASR 起点、抽帧时间和打标时间；确认后再手动修正 review_start 或接受当前时间。",
        "corrected_review_start": "打开视频核对 ASR 起点、抽帧时间和打标时间；确认后把人工认可的秒数填入 corrected_review_start。",
        "corrected_transcript": "打开转录编辑器核对候选字幕/ASR来源，导出 transcript-edits.json 后用 apply-transcript-edits 导入。",
        "accepted": "确认该片段无需额外修正。",
    }.get(status, "人工判断后选择合适状态。")


def _suggested_status_from_timeline_item(item: dict[str, Any]) -> str:
    issues = set(str(value) for value in item.get("quality_issues") or [])
    if "temporal_sequence_without_analysis" in issues:
        return "corrected_temporal_visual_understanding"
    if "semantic_frame_without_analysis" in issues:
        return "corrected_visual_understanding"
    if "missing_visual_text" in issues or "ocr_text_empty" in issues:
        return "corrected_visual_text"
    if "screen_text_low_confidence" in issues:
        return "needs_rerun_ocr"
    retention = item.get("visual_retention") if isinstance(item.get("visual_retention"), dict) else {}
    if str(retention.get("recommendation") or "") in {"keep_image", "review_image"}:
        return "keep_image"
    if item.get("needs_human_review"):
        return "needs_human_review"
    return "accepted"


def _time_range_label(item: dict[str, Any]) -> str:
    start = _float_value(item.get("start"))
    end = _float_value(item.get("end"))
    return f"{start:.3f}-{end:.3f}s"


def _truncate_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _evidence_excerpt(item: dict[str, Any], limit: int = 140) -> str:
    text = " ".join(
        value
        for value in [_meaningful_excerpt_text(item.get("transcript")), _meaningful_excerpt_text(item.get("visual_text"))]
        if value
    )
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _model_output_excerpt(item: dict[str, Any], limit: int = 180) -> str:
    parts: list[str] = []
    for key in ("visual_understanding", "temporal_visual_understanding"):
        value = item.get(key)
        if isinstance(value, dict) and value:
            parts.append(str({k: v for k, v in value.items() if k not in {"raw_response", "raw_content"}}))
    text = " ".join(" ".join(parts).split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _ocr_empty_results_by_index(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    visual_structure = manifest.get("visual_structure") if isinstance(manifest.get("visual_structure"), dict) else {}
    results = visual_structure.get("ebook_pipeline_results") if isinstance(visual_structure.get("ebook_pipeline_results"), list) else []
    by_index: dict[int, dict[str, Any]] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        if str(result.get("blocker") or "") != "ocr_text_empty":
            continue
        index = _int(result.get("index"))
        if index:
            by_index[index] = result
    return by_index


def _meaningful_excerpt_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("<!--") and line.endswith("-->"):
            continue
        if line.startswith("# "):
            heading = line[2:].strip()
            if _looks_like_frame_stem(heading):
                continue
        lines.append(line)
    return " ".join(" ".join(lines).split())


def _looks_like_frame_stem(value: str) -> bool:
    stem = Path(value).stem
    if "_000" in stem and stem.endswith("ms"):
        return True
    digits = sum(1 for char in stem if char.isdigit())
    return digits >= 8 and stem.endswith("ms")


def _review_status_summary(reviews: list[dict[str, Any]], *, skipped: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for review in reviews:
        status = _canonical_review_status(review.get("status"))
        counts[status] = counts.get(status, 0) + 1
    return {
        "by_status": dict(sorted(counts.items())),
        "accepted_known_gap": counts.get("accepted_known_gap", 0),
        "keep_image": counts.get("keep_image", 0),
        "corrected_visual_text": counts.get("corrected_visual_text", 0),
        "needs_rerun_ocr": counts.get("needs_rerun_ocr", 0),
        "skipped": len(skipped),
    }


def _timeline_review_status_counts(timeline: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in timeline:
        status = _review_status(item).lower()
        if status in ACCEPTED_REVIEW_STATUSES or status == "needs_rerun_ocr":
            counts[status or "unknown"] = counts.get(status or "unknown", 0) + 1
    return dict(sorted(counts.items()))


def _suggested_status_counts(targets: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in targets.get("items") or []:
        if isinstance(item, dict):
            status = str(item.get("suggested_status") or "unknown")
            counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _next_review_batch(root: Path, targets: dict[str, Any]) -> dict[str, Any]:
    by_reason = targets.get("by_reason") if isinstance(targets.get("by_reason"), dict) else {}
    reason = next(iter(by_reason), "")
    command = f".\\scripts\\video-knowledge.ps1 prepare-review-session {root}"
    if reason:
        command += f" --reason {reason}"
    command += " --limit 30"
    return {
        "reason": reason,
        "limit": 30,
        "command": command,
        "review_pack": str(root / "review-pack.md"),
        "todo_json": str(root / "review-notes.todo.json"),
    }


def _safe_output_prefix(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in str(value or "review-pack").strip())
    return cleaned.strip("-_") or "review-pack"


def _asset_paths(item: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for asset in item.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        path = str(asset.get("path") or asset.get("source") or "").strip()
        if path:
            paths.append(path)
    return _dedupe(paths)


def _crop_paths(item: dict[str, Any]) -> list[str]:
    recovery = item.get("screen_text_recovery") if isinstance(item.get("screen_text_recovery"), dict) else {}
    paths = [str(path).strip() for path in recovery.get("crop_paths") or [] if str(path).strip()]
    return _dedupe(paths)


def _time_range(item: dict[str, Any]) -> str:
    return f"{_format_seconds(_number(item.get('start')))}-{_format_seconds(_number(item.get('end')))}"


def _format_seconds(value: float) -> str:
    seconds = max(float(value or 0), 0.0)
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    if millis == 1000:
        whole += 1
        millis = 0
    minutes, sec = divmod(whole, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{sec:02d}.{millis:03d}"


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0




def _compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
