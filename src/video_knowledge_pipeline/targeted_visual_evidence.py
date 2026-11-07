from __future__ import annotations

from pathlib import Path
from typing import Any

from .high_res_tile_plan import run_high_res_tile_plan
from .models import now_iso
from .screen_text_recovery import run_screen_text_recovery
from .storage import read_json, write_json
from .visual_structure import run_visual_structure_plan
from .vision_review_triage import vision_review_triage


SCHEMA = "video_knowledge_pipeline.targeted_visual_evidence.v1"
BLOCKERS = {
    "ocr_text_empty",
    "ocr_wrapper_only",
    "ocr_text_low_information",
    "screen_text_low_confidence",
    "missing_visual_text",
    "structured_visual_without_structure",
    "ebook_pipeline_failed",
}


def run_targeted_visual_evidence(
    bundle_dir: str | Path,
    *,
    tagger_json: str | Path | None = None,
    min_score: int = 3,
    limit: int = 0,
    execute_ebook: bool = False,
    execute_crops: bool = False,
    execute_ocr: bool = False,
    execute_tiles: bool = False,
    allow_online_review: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    if not write and any((execute_ebook, execute_crops, execute_ocr, execute_tiles)):
        raise ValueError("--no-write cannot be combined with --execute-ebook/--execute-crops/--execute-ocr/--execute-tiles")
    triage = vision_review_triage(
        root,
        mode="triage",
        tagger_json=tagger_json,
        semantic_limit=limit or None,
        temporal_limit=limit or None,
        visual_structure_limit=limit or None,
        min_score=min_score,
        write=write,
    )
    document_indexes = [int(value) for value in triage.get("visual_structure_first_indexes") or []]
    if not write:
        document = {"summary": {"selected_indexes": document_indexes, "execute_ebook_pipeline": False, "status": "preview_no_write"}}
        unresolved = _unresolved_document_indexes(root, document_indexes)
        screen = {"selected_indexes": unresolved, "crop_summary": {}, "ocr_summary": {}, "status": "preview_no_write"}
        unresolved_after_ocr = list(unresolved)
        tiles = {"selected_indexes": unresolved_after_ocr, "summary": {}, "status": "preview_no_write"}
    else:
        document = run_visual_structure_plan(
            root,
            execute_ebook_pipeline=execute_ebook,
            indexes=document_indexes,
            limit=limit or None,
        ) if document_indexes else {"summary": {"selected_indexes": [], "execute_ebook_pipeline": False}}
        unresolved = _unresolved_document_indexes(root, document_indexes)
        screen = run_screen_text_recovery(
            root,
            execute_crops=execute_crops,
            execute_ocr=execute_ocr,
            indexes=unresolved,
            limit=limit,
            write=write,
        ) if unresolved else {"selected_indexes": [], "crop_summary": {}, "ocr_summary": {}}
        unresolved_after_ocr = _unresolved_document_indexes(root, unresolved)
        tiles = run_high_res_tile_plan(
            root,
            execute_tiles=execute_tiles,
            indexes=unresolved_after_ocr,
            limit=limit,
            write=write,
        ) if unresolved_after_ocr else {"selected_indexes": [], "summary": {}}

    should_retriage = bool(write and any((execute_ebook, execute_crops, execute_ocr, execute_tiles)))
    post_local_triage = (
        vision_review_triage(
            root,
            mode="triage",
            tagger_json=tagger_json,
            semantic_limit=limit or None,
            temporal_limit=limit or None,
            visual_structure_limit=limit or None,
            min_score=min_score,
            write=write,
        )
        if should_retriage
        else triage
    )
    semantic_indexes = [int(value) for value in post_local_triage.get("semantic_indexes") or []]
    temporal_indexes = [int(value) for value in post_local_triage.get("temporal_indexes") or []]
    online_semantic = sorted(set(semantic_indexes + unresolved_after_ocr))
    human_review = [] if allow_online_review else online_semantic
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "status": _status(
            document_indexes=document_indexes,
            unresolved=unresolved_after_ocr,
            semantic=online_semantic,
            temporal=temporal_indexes,
            allow_online_review=allow_online_review,
        ),
        "triage": {
            "mode": post_local_triage.get("mode"),
            "selected_counts": post_local_triage.get("selected_counts") or {},
            "pre_local_selected_counts": triage.get("selected_counts") or {},
            "post_local_selected_counts": post_local_triage.get("selected_counts") or {},
            "retriaged_after_local_evidence": should_retriage,
            "document_indexes": document_indexes,
            "semantic_indexes": semantic_indexes,
            "temporal_indexes": temporal_indexes,
        },
        "local_document_stage": {
            "execute_ebook": bool(execute_ebook),
            "selected_indexes": document_indexes,
            "summary": document.get("summary") or {},
        },
        "local_screen_text_stage": {
            "execute_crops": bool(execute_crops),
            "execute_ocr": bool(execute_ocr),
            "selected_indexes": unresolved,
            "crop_summary": screen.get("crop_summary") or {},
            "ocr_summary": screen.get("ocr_summary") or {},
        },
        "local_tile_stage": {
            "execute_tiles": bool(execute_tiles),
            "selected_indexes": unresolved_after_ocr,
            "summary": tiles.get("summary") or {},
        },
        "unresolved_document_indexes": unresolved_after_ocr,
        "online_review": {
            "allowed": bool(allow_online_review),
            "executed": False,
            "semantic_indexes": online_semantic,
            "temporal_indexes": temporal_indexes,
            "preflight_required": bool(online_semantic or temporal_indexes),
            "reason": "Only items unresolved after local document/OCR/tiling or semantic/temporal triage are eligible.",
        },
        "human_review_indexes": human_review,
        "operator_boundary": {
            "local_first": True,
            "ebook_pipeline_is_document_primary": True,
            "empty_wrapper_or_low_information_remains_blocked": True,
            "no_online_vision_call": True,
            "online_review_requires_existing_vision_preflight_and_confirmation": True,
            "does_not_clear_visual_gap_without_evidence": True,
        },
        "artifacts": {
            "triage_json": str(root / "vision-review-triage.json"),
            "visual_structure_report": str(root / "visual-structure-report.md"),
            "screen_text_report": str(root / "screen-text-recovery.md"),
            "tile_report": str(root / "high-res-tile-plan.md"),
            "report_json": str(root / "targeted-visual-evidence.json"),
            "report_markdown": str(root / "targeted-visual-evidence.md"),
        },
        "next_actions": _next_actions(
            root,
            unresolved=unresolved_after_ocr,
            semantic=online_semantic,
            temporal=temporal_indexes,
            allow_online_review=allow_online_review,
        ),
        "updated_at": now_iso(),
    }
    if write:
        write_json(root / "targeted-visual-evidence.json", result)
        (root / "targeted-visual-evidence.md").write_text(_render_markdown(result), encoding="utf-8")
        manifest_path = root / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        manifest = manifest if isinstance(manifest, dict) else {}
        manifest["targeted_visual_evidence_json"] = "targeted-visual-evidence.json"
        manifest["targeted_visual_evidence_markdown"] = "targeted-visual-evidence.md"
        manifest["targeted_visual_evidence_summary"] = {
            "status": result["status"],
            "unresolved_document_count": len(unresolved_after_ocr),
            "online_semantic_count": len(online_semantic),
            "online_temporal_count": len(temporal_indexes),
            "updated_at": result["updated_at"],
        }
        write_json(manifest_path, manifest)
    return result


def _unresolved_document_indexes(root: Path, indexes: list[int]) -> list[int]:
    if not indexes:
        return []
    timeline = read_json(root / "timeline.json")
    timeline = timeline if isinstance(timeline, list) else []
    requested = set(indexes)
    unresolved: list[int] = []
    for position, raw in enumerate(timeline, start=1):
        if not isinstance(raw, dict):
            continue
        index = int(raw.get("index") or position)
        if index not in requested:
            continue
        visual_text = str(raw.get("visual_text") or raw.get("ocr_text") or "").strip()
        structured = raw.get("structured_visual")
        issue_text = " ".join(_strings(raw.get("quality_issues")) + _strings(raw.get("issues")) + _strings(raw.get("ebook_pipeline_status")))
        confidence = _ocr_confidence(raw)
        blocked = any(value in issue_text for value in BLOCKERS) or (confidence is not None and confidence < 0.85)
        if not visual_text or not structured or blocked:
            unresolved.append(index)
    return unresolved


def _ocr_confidence(item: dict[str, Any]) -> float | None:
    for key in ("ocr_confidence", "visual_text_confidence", "screen_text_confidence"):
        value = item.get(key)
        if value is None or value == "":
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if 1 < numeric <= 100:
            numeric /= 100
        if 0 <= numeric <= 1:
            return numeric
    return None


def _strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        rows = []
        for key, item in value.items():
            rows.append(str(key))
            rows.extend(_strings(item))
        return rows
    if isinstance(value, list):
        return [token for item in value for token in _strings(item)]
    return [str(value or "")]


def _status(
    *,
    document_indexes: list[int],
    unresolved: list[int],
    semantic: list[int],
    temporal: list[int],
    allow_online_review: bool,
) -> str:
    if not document_indexes and not semantic and not temporal:
        return "completed_no_candidates"
    if unresolved or semantic or temporal:
        return "online_review_planned" if allow_online_review else "needs_local_or_human_review"
    return "completed_local_evidence"


def _next_actions(
    root: Path,
    *,
    unresolved: list[int],
    semantic: list[int],
    temporal: list[int],
    allow_online_review: bool,
) -> list[str]:
    actions: list[str] = []
    if unresolved:
        csv = ",".join(str(value) for value in unresolved)
        actions.append(f".\\scripts\\video-knowledge.ps1 high-res-tile-plan '{root}' --indexes {csv} --execute-tiles")
    if (semantic or temporal) and allow_online_review:
        actions.append("Run vision-execution-preflight for only the listed semantic/temporal indexes; do not send all frames.")
    elif semantic or temporal:
        actions.append("Review listed unresolved indexes locally or explicitly allow a preflighted online vision batch.")
    return actions


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Targeted Visual Evidence",
        "",
        f"- Status: {result.get('status')}",
        f"- Local-first: {result.get('operator_boundary', {}).get('local_first')}",
        f"- Unresolved document indexes: {result.get('unresolved_document_indexes')}",
        f"- Online semantic candidates: {result.get('online_review', {}).get('semantic_indexes')}",
        f"- Online temporal candidates: {result.get('online_review', {}).get('temporal_indexes')}",
        "",
        "## Route",
        "",
        "document frame -> ebook_markdown_pipeline -> crop OCR -> high-resolution tiles -> targeted online/human review",
        "",
        "## Next Actions",
        "",
    ]
    lines.extend(f"- {action}" for action in result.get("next_actions") or [])
    return "\n".join(lines).rstrip() + "\n"
