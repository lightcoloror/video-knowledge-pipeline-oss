from __future__ import annotations

from pathlib import Path
from typing import Any

from .bundle_readiness import build_bundle_readiness
from .models import now_iso
from .media_applicability import project_timeline_for_media, resolve_media_applicability
from .storage import read_json, write_json


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
}


def build_repair_status(manifest: dict[str, Any], timeline: list[dict[str, Any]] | None = None,
                        *, bundle_dir: str | Path | None = None) -> dict[str, Any]:
    """Summarize repair task state for WebUI and MCP agents."""
    applicability = resolve_media_applicability(bundle_dir, manifest, timeline or [])
    timeline = project_timeline_for_media(timeline or [], applicability)
    rows = [
        _frame_recapture_status(manifest),
        _time_gap_recapture_status(manifest),
        _visual_structure_status(manifest, timeline or []),
        _ocr_backfill_status(manifest, timeline or []),
        _video_frame_router_status(manifest, timeline or []),
        _multimodal_frame_analysis_status(manifest, timeline or []),
        _temporal_visual_analysis_status(manifest, timeline or []),
    ]
    if applicability["visual_status"] == "not_applicable":
        for row in rows:
            if row.get("key") != "time_gap_recapture":
                row.update(status="not_applicable", count=0, next_action="", mcp_tool="",
                           action_hint="纯音频无附加视觉材料，此通道不适用。")
    pending = [row for row in rows if row.get("status") in {"pending", "ran_no_update", "failed"}]
    return {
        "schema": "lecture_repair_status.v1",
        "media_applicability": applicability,
        "total": len(rows),
        "pending": len(pending),
        "items": rows,
        "next": pending[0] if pending else None,
    }


def refresh_bundle_repair_status(bundle_dir: str | Path) -> dict[str, Any]:
    """Refresh manifest.repair_status for an existing WebUI bundle."""
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    timeline_data = read_json(timeline_path) if timeline_path.exists() else []
    timeline = [item for item in timeline_data if isinstance(item, dict)] if isinstance(timeline_data, list) else []
    repair_status = build_repair_status(manifest, timeline, bundle_dir=root)
    repair_status["refreshed_at"] = now_iso()
    manifest["repair_status"] = repair_status
    manifest["review_readiness"] = build_bundle_readiness(manifest, timeline, bundle_dir=root)
    write_json(manifest_path, manifest)
    return {
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "timeline_path": str(timeline_path) if timeline_path.exists() else "",
        "repair_status": repair_status,
    }


def _frame_recapture_status(manifest: dict[str, Any]) -> dict[str, Any]:
    plan = _dict_value(manifest.get("frame_recapture"))
    return _status_from_plan(
        manifest,
        key="frame_recapture",
        label="关键帧补采样",
        mcp_tool="run_frame_recapture_plan",
        count=_int_value(plan.get("count")),
        last_run=_dict_value(plan.get("last_run")),
        last_backfill=_dict_value(plan.get("last_backfill")),
        next_action="run_frame_recapture_plan",
        action_hint="先预览缺失关键帧；确认后再把 args JSON 的 execute 改为 true。",
    )


def _time_gap_recapture_status(manifest: dict[str, Any]) -> dict[str, Any]:
    plan = _dict_value(manifest.get("time_gap_recapture"))
    count = _int_value(plan.get("count"))
    status = "not_needed" if count <= 0 else "manual_required"
    return {
        "key": "time_gap_recapture",
        "label": "时间盲区补采样",
        "status": status,
        "count": count,
        "mcp_tool": "",
        "mcp_args_path": "",
        "next_action": "inspect_time_gap_frames" if count > 0 else "",
        "action_hint": "时间盲区需要人工检查中点帧，再决定补录 timeline 或重跑更密采样。",
        "last_run": {},
        "last_backfill": {},
    }


def _ocr_backfill_status(manifest: dict[str, Any], timeline: list[dict[str, Any]]) -> dict[str, Any]:
    plan = _dict_value(manifest.get("ocr_backfill"))
    if timeline:
        count = sum(
            1
            for item in timeline
            if not _is_review_closed(item)
            and _has_any_issue(
                item,
                {
                    "ocr_text_empty",
                    "screen_text_low_confidence",
                    "structured_visual_without_ocr",
                },
            )
        )
    else:
        count = _int_value(plan.get("count"))
    return _status_from_plan(
        manifest,
        key="ocr_backfill",
        label="OCR 备用回填",
        mcp_tool="run_ocr_backfill",
        count=count,
        last_run=_dict_value(plan.get("last_run")),
        last_backfill=_dict_value(plan.get("last_backfill")),
        next_action="run_ocr_backfill",
        action_hint="备用通道：主通道是 run_visual_structure_plan -> ebook_markdown_pipeline；只有需要手动 OCR JSON、CaptiOCR 或 Tesseract fallback 时才用这里。",
    )


def _visual_structure_status(manifest: dict[str, Any], timeline: list[dict[str, Any]]) -> dict[str, Any]:
    plan = _dict_value(manifest.get("visual_structure"))
    count = sum(
        1
        for item in timeline
        if str(item.get("visual_route") or "") in {"document_visual", "mixed"}
        and not _is_review_closed(item)
        and not _has_meaningful_visual_structure(item)
    )
    if count <= 0 and not timeline:
        count = _int_value(plan.get("count"))
    return _status_from_plan(
        manifest,
        key="visual_structure",
        label="图文截图解析",
        mcp_tool="run_visual_structure_plan",
        count=count,
        last_run=_dict_value(plan.get("last_run")),
        last_backfill=_dict_value(plan.get("last_import") or plan.get("last_backfill")),
        next_action="run_visual_structure_plan",
        action_hint="通过 ebook_markdown_pipeline 提取已路由截图中的文字/版面；document_visual 是主分支，semantic_frame/temporal_sequence 则与多模态理解并行后整合。",
    )


def _video_frame_router_status(manifest: dict[str, Any], timeline: list[dict[str, Any]]) -> dict[str, Any]:
    plan = _dict_value(manifest.get("video_frame_router"))
    last_run = _dict_value(plan.get("last_run"))
    count = _int_value(plan.get("count")) if last_run else sum(1 for item in timeline if (item.get("frame_paths") or item.get("assets")) and not item.get("visual_route"))
    return _status_from_plan(
        manifest,
        key="video_frame_router",
        label="画面路由",
        mcp_tool="run_video_frame_router",
        count=count,
        last_run=last_run,
        last_backfill=last_run,
        next_action="run_video_frame_router",
        action_hint="先把帧分到图文截图、单帧语义、连续变化或混合分支；低置信度保留人工复核。",
    )


def _multimodal_frame_analysis_status(manifest: dict[str, Any], timeline: list[dict[str, Any]]) -> dict[str, Any]:
    plan = _dict_value(manifest.get("multimodal_frame_analysis"))
    count = _int_value(plan.get("count"))
    if count <= 0:
        count = sum(1 for item in timeline if str(item.get("visual_route") or "") in {"semantic_frame", "mixed"} and not item.get("visual_understanding"))
    return _status_from_plan(
        manifest,
        key="multimodal_frame_analysis",
        label="多模态单帧理解",
        mcp_tool="run_multimodal_frame_analysis",
        count=count,
        last_run=_dict_value(plan.get("last_run")),
        last_backfill=_dict_value(plan.get("last_run")),
        next_action="run_multimodal_frame_analysis",
        action_hint="默认预览/导入；设置 execute=true 后调用配置好的多模态 API 看非纯图文画面。",
    )


def _temporal_visual_analysis_status(manifest: dict[str, Any], timeline: list[dict[str, Any]]) -> dict[str, Any]:
    plan = _dict_value(manifest.get("temporal_visual_analysis"))
    count = _int_value(plan.get("count"))
    if count <= 0:
        count = sum(1 for item in timeline if str(item.get("visual_route") or "") in {"temporal_sequence", "mixed"} and not item.get("temporal_visual_understanding"))
    return _status_from_plan(
        manifest,
        key="temporal_visual_analysis",
        label="连续片段理解",
        mcp_tool="run_temporal_visual_analysis",
        count=count,
        last_run=_dict_value(plan.get("last_run")),
        last_backfill=_dict_value(plan.get("last_run")),
        next_action="run_temporal_visual_analysis",
        action_hint="默认预览/导入 5-12 帧序列；设置 execute=true 后让多模态 API 描述状态变化和操作链。",
    )


def _status_from_plan(
    manifest: dict[str, Any],
    *,
    key: str,
    label: str,
    mcp_tool: str,
    count: int,
    last_run: dict[str, Any],
    last_backfill: dict[str, Any],
    next_action: str,
    action_hint: str,
) -> dict[str, Any]:
    if count <= 0:
        status = "not_needed"
    elif _int_value(last_backfill.get("updated")) > 0:
        status = "updated"
    elif _int_value(last_run.get("failed")) > 0 and _int_value(last_run.get("succeeded")) <= 0:
        status = "failed"
    elif last_run:
        status = "ran_no_update"
    else:
        status = "pending"
    repair_tool = _dict_value(_dict_value(manifest.get("repair_tools")).get(key))
    return {
        "key": key,
        "label": label,
        "status": status,
        "count": count,
        "mcp_tool": str(repair_tool.get("mcp_tool") or mcp_tool),
        "mcp_args_path": str(repair_tool.get("mcp_args_path") or ""),
        "next_action": next_action if status in {"pending", "ran_no_update", "failed"} else "",
        "action_hint": action_hint,
        "last_run": last_run,
        "last_backfill": last_backfill,
    }


def _has_any_issue(item: dict[str, Any], issues: set[str]) -> bool:
    return bool({str(issue) for issue in item.get("quality_issues") or []} & issues)


def _has_non_empty(value: Any) -> bool:
    if isinstance(value, dict):
        return any(item not in (None, "", [], {}) for item in value.values())
    if isinstance(value, list):
        return any(item not in (None, "", [], {}) for item in value)
    return value not in (None, "", [], {})


def _has_meaningful_visual_structure(item: dict[str, Any]) -> bool:
    return _has_non_empty(item.get("structured_visual")) or bool(str(item.get("visual_text") or "").strip())


def _is_review_closed(item: dict[str, Any]) -> bool:
    human_review = item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
    status = str(item.get("review_status") or human_review.get("status") or "").strip().lower()
    return status in ACCEPTED_REVIEW_STATUSES


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
