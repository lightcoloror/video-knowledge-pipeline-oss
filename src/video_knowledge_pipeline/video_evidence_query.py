from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import read_json, write_json
from .video_rag_search import search_video_rag


PLAN_SCHEMA = "video_knowledge_pipeline.video_evidence_query_plan.v1"
RESULT_SCHEMA = "video_knowledge_pipeline.video_evidence_confirmation.v1"
ALLOWED_CONFIRMATION_STATUSES = {"confirmed", "rejected", "needs_more_evidence"}


def build_video_evidence_query_plan(
    bundle_dir: str | Path,
    *,
    query: str,
    coarse_top_k: int = 12,
    fine_top_k: int = 4,
    write: bool = True,
) -> dict[str, Any]:
    """Build a DeepVideoDiscovery/VideoLucy-style coarse-to-fine review plan."""

    root = Path(bundle_dir).expanduser().resolve()
    search = search_video_rag(root, query=query, top_k=max(coarse_top_k, fine_top_k), ensure_pack=True, retrieval_backend="keyword", write=write)
    coarse_hits = [row for row in search.get("hits") or [] if isinstance(row, dict)]
    timeline = _timeline(root)
    fine = [
        _fine_candidate(root, hit, timeline=timeline, index=index)
        for index, hit in enumerate(coarse_hits[: max(0, int(fine_top_k))], start=1)
    ]
    plan = {
        "schema": PLAN_SCHEMA,
        "bundle_dir": str(root),
        "query": query,
        "status": "ready_for_fine_review" if fine else "no_retrieval_hits",
        "ok": bool(fine),
        "coarse_stage": {
            "backend": search.get("retrieval_backend"),
            "hit_count": len(coarse_hits),
            "hits": coarse_hits,
        },
        "fine_stage": {
            "candidate_count": len(fine),
            "candidates": fine,
        },
        "confirm_stage": {
            "required": True,
            "allowed_statuses": sorted(ALLOWED_CONFIRMATION_STATUSES),
            "completed": 0,
        },
        "operator_boundary": {
            "local_retrieval_only": True,
            "no_cloud_or_model_call": True,
            "retrieval_hit_is_not_confirmation": True,
            "no_evidence_requires_recapture_or_review": True,
            "does_not_modify_timeline": True,
        },
        "artifacts": {
            "json": "exports/video-evidence-query-plan.json",
            "markdown": "exports/video-evidence-query-plan.md",
            "todo": "exports/video-evidence-confirmation.todo.json",
        },
        "updated_at": now_iso(),
    }
    if write:
        exports = root / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        write_json(exports / "video-evidence-query-plan.json", plan)
        (exports / "video-evidence-query-plan.md").write_text(_render_plan(plan), encoding="utf-8")
        write_json(
            exports / "video-evidence-confirmation.todo.json",
            {
                "schema": "video_knowledge_pipeline.video_evidence_confirmation_decisions.v1",
                "query": query,
                "rows": [
                    {
                        "candidate_id": row["candidate_id"],
                        "status": "",
                        "evidence_paths": row["evidence_paths"],
                        "notes": "",
                    }
                    for row in fine
                ],
            },
        )
        manifest_path = root / "manifest.json"
        manifest = _mapping(manifest_path)
        manifest["video_evidence_query_plan_json"] = "exports/video-evidence-query-plan.json"
        manifest["video_evidence_query_plan_markdown"] = "exports/video-evidence-query-plan.md"
        manifest["video_evidence_confirmation_todo_json"] = "exports/video-evidence-confirmation.todo.json"
        write_json(manifest_path, manifest)
    return plan


def apply_video_evidence_confirmation(
    bundle_dir: str | Path,
    *,
    decisions_json: str | Path,
    plan_json: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    plan_path = _bundle_path(root, plan_json or "exports/video-evidence-query-plan.json")
    decisions_path = _bundle_path(root, decisions_json)
    plan = _mapping(plan_path)
    decisions = _mapping(decisions_path)
    candidates = {
        str(row.get("candidate_id") or ""): row
        for row in (plan.get("fine_stage") or {}).get("candidates") or []
        if isinstance(row, dict) and row.get("candidate_id")
    }
    accepted: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for row in decisions.get("rows") or []:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id") or "")
        candidate = candidates.get(candidate_id)
        status = str(row.get("status") or "").strip().lower()
        evidence_paths = [str(value) for value in row.get("evidence_paths") or [] if str(value).strip()]
        if candidate is None:
            rejected_rows.append({"candidate_id": candidate_id, "reason": "unknown_candidate"})
            continue
        if status not in ALLOWED_CONFIRMATION_STATUSES:
            rejected_rows.append({"candidate_id": candidate_id, "reason": "invalid_status"})
            continue
        if status == "confirmed" and not evidence_paths:
            rejected_rows.append({"candidate_id": candidate_id, "reason": "confirmed_requires_evidence"})
            continue
        accepted.append(
            {
                "candidate_id": candidate_id,
                "status": status,
                "evidence_paths": evidence_paths,
                "notes": str(row.get("notes") or ""),
                "time_range": candidate.get("time_range"),
                "query": plan.get("query"),
            }
        )
    unresolved = [
        candidate_id
        for candidate_id in candidates
        if candidate_id not in {row["candidate_id"] for row in accepted}
    ]
    result = {
        "schema": RESULT_SCHEMA,
        "bundle_dir": str(root),
        "query": plan.get("query"),
        "status": "completed" if not rejected_rows and not unresolved else "completed_with_open_items",
        "ok": bool(accepted) and not rejected_rows,
        "confirmed_count": sum(row["status"] == "confirmed" for row in accepted),
        "rejected_count": sum(row["status"] == "rejected" for row in accepted),
        "needs_more_evidence_count": sum(row["status"] == "needs_more_evidence" for row in accepted),
        "decisions": accepted,
        "invalid_rows": rejected_rows,
        "unresolved_candidate_ids": unresolved,
        "operator_boundary": {
            "confirmation_requires_evidence": True,
            "needs_more_evidence_remains_open": True,
            "does_not_promote_to_fact_store": True,
            "no_cloud_or_model_call": True,
        },
        "updated_at": now_iso(),
    }
    if write:
        exports = root / "exports"
        write_json(exports / "video-evidence-confirmation.json", result)
        (exports / "video-evidence-confirmation.md").write_text(_render_confirmation(result), encoding="utf-8")
        manifest_path = root / "manifest.json"
        manifest = _mapping(manifest_path)
        manifest["video_evidence_confirmation_json"] = "exports/video-evidence-confirmation.json"
        manifest["video_evidence_confirmation_markdown"] = "exports/video-evidence-confirmation.md"
        write_json(manifest_path, manifest)
    return result


def _fine_candidate(root: Path, hit: dict[str, Any], *, timeline: list[dict[str, Any]], index: int) -> dict[str, Any]:
    evidence = [str(value) for value in hit.get("evidence_paths") or [] if str(value).strip()]
    timeline_indexes = [int(value) for value in hit.get("timeline_indexes") or [] if _int(value) > 0]
    for item in timeline:
        if _int(item.get("index")) not in timeline_indexes:
            continue
        evidence.extend(_item_evidence_paths(item))
    evidence = _dedupe(evidence)
    has_understanding = bool(hit.get("has_visual_evidence") or hit.get("has_temporal_evidence"))
    if has_understanding and evidence:
        suggested = "confirm_or_reject_claim"
    elif evidence:
        suggested = "inspect_frames_then_confirm"
    else:
        suggested = "recapture_frames_or_mark_needs_more_evidence"
    return {
        "candidate_id": f"evidence-query-{index:03d}",
        "chunk_id": hit.get("chunk_id"),
        "score": hit.get("score"),
        "start": hit.get("start"),
        "end": hit.get("end"),
        "time_range": f"{hit.get('start_time')} - {hit.get('end_time')}",
        "snippet": hit.get("snippet"),
        "timeline_indexes": timeline_indexes,
        "evidence_paths": evidence,
        "has_visual_understanding": bool(hit.get("has_visual_evidence")),
        "has_temporal_understanding": bool(hit.get("has_temporal_evidence")),
        "suggested_next_action": suggested,
        "confirmation_status": "pending",
    }


def _timeline(root: Path) -> list[dict[str, Any]]:
    for path in (root / "timeline.json", root / "timeline-items.json"):
        if path.exists():
            value = read_json(path)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
            if isinstance(value, dict) and isinstance(value.get("items"), list):
                return [row for row in value["items"] if isinstance(row, dict)]
    return []


def _item_evidence_paths(item: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("frame_path", "crop_path"):
        if item.get(key):
            paths.append(str(item[key]))
    for key in ("frame_paths", "temporal_frame_paths", "evidence_frame_paths"):
        paths.extend(str(value) for value in item.get(key) or [] if str(value).strip())
    return paths


def _bundle_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(f"artifact not found: {path}")
    return path.resolve()


def _mapping(path: Path) -> dict[str, Any]:
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _render_plan(plan: dict[str, Any]) -> str:
    lines = [
        "# Video Evidence Query Plan",
        "",
        f"- Query: {plan.get('query')}",
        f"- Coarse hits: {(plan.get('coarse_stage') or {}).get('hit_count', 0)}",
        f"- Fine candidates: {(plan.get('fine_stage') or {}).get('candidate_count', 0)}",
        "",
        "| Candidate | Time | Evidence | Next action |",
        "| --- | --- | ---: | --- |",
    ]
    for row in (plan.get("fine_stage") or {}).get("candidates") or []:
        lines.append(f"| {row.get('candidate_id')} | {row.get('time_range')} | {len(row.get('evidence_paths') or [])} | {row.get('suggested_next_action')} |")
    return "\n".join(lines).rstrip() + "\n"


def _render_confirmation(result: dict[str, Any]) -> str:
    lines = [
        "# Video Evidence Confirmation",
        "",
        f"- Status: {result.get('status')}",
        f"- Confirmed: {result.get('confirmed_count')}",
        f"- Needs more evidence: {result.get('needs_more_evidence_count')}",
        "",
    ]
    for row in result.get("decisions") or []:
        lines.append(f"- {row.get('candidate_id')}: {row.get('status')} ({row.get('time_range')})")
    return "\n".join(lines).rstrip() + "\n"