from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import now_iso
from .media_applicability import project_timeline_for_media, resolve_media_applicability
from .source_artifacts import summarize_manifest_source_artifacts
from .storage import read_json, write_json


STRUCTURED_TYPES = {"formula", "table", "code"}
ACCEPTED_REVIEW_STATUSES = {
    "reviewed",
    "accepted",
    "accepted_known_gap",
    "keep_image",
    "accepted_no_visual_content",
    "accepted_provider_blocked",
    "corrected_visual_text",
    "corrected_visual_understanding",
    "corrected_temporal_visual_understanding",
}
FRAME_GAP_ISSUES = {"keep_image_without_frame", "structured_visual_without_frame", "missing_frame"}
STRUCTURE_GAP_ISSUES = {"structured_visual_without_structure"}
VISUAL_ROUTE_ISSUES = {"missing_visual_route"}
VISUAL_ANALYSIS_ISSUES = {"missing_visual_understanding", "semantic_frame_without_analysis", "temporal_sequence_without_analysis"}


def audit_bundle_readiness(bundle_dir: str | Path, *, write: bool = True) -> dict[str, Any]:
    """Assess whether a WebUI lecture bundle is ready for final export."""
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
    readiness = build_bundle_readiness(manifest, timeline, bundle_dir=root)
    if write:
        manifest["review_readiness"] = readiness
        write_json(manifest_path, manifest)
    return {
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "timeline_path": str(timeline_path),
        "review_readiness": readiness,
    }


def build_bundle_readiness(
    manifest: dict[str, Any],
    timeline: list[dict[str, Any]],
    *,
    bundle_dir: str | Path | None = None,
) -> dict[str, Any]:
    applicability = resolve_media_applicability(bundle_dir, manifest, timeline)
    timeline = project_timeline_for_media(timeline, applicability)
    reviewed = [item for item in timeline if _is_reviewed(item)]
    risk_items = [item for item in timeline if _quality_issues(item)]
    unreviewed_risk_items = [item for item in risk_items if not _is_reviewed(item)]
    structured_items = [item for item in timeline if _has_structured_material(item)]
    pending_structured = [item for item in structured_items if not _is_reviewed(item)]
    frame_gap_items = [item for item in timeline if set(_quality_issues(item)) & FRAME_GAP_ISSUES]
    asset_gap_items = [item for item in timeline if _has_asset_gap(item, bundle_dir=bundle_dir)]
    structure_gap_items = [item for item in timeline if set(_quality_issues(item)) & STRUCTURE_GAP_ISSUES]
    visual_route_gap_items = [item for item in timeline if set(_quality_issues(item)) & VISUAL_ROUTE_ISSUES]
    visual_analysis_gap_items = [item for item in timeline if set(_quality_issues(item)) & VISUAL_ANALYSIS_ISSUES]
    repair_items = _repair_items(manifest)
    if applicability["visual_status"] == "not_applicable":
        repair_items = [item for item in repair_items if item.get("key") == "time_gap_recapture"]
    pending_repair_items = [
        item for item in repair_items if str(item.get("status") or "") in {"pending", "ran_no_update", "failed"}
    ]
    manual_repair_items = [item for item in repair_items if str(item.get("status") or "") == "manual_required"]
    coverage = manifest.get("coverage") if isinstance(manifest.get("coverage"), dict) else {}
    source_artifacts = summarize_manifest_source_artifacts(manifest)
    time_gap_count = int(_number(coverage.get("time_gap_count")))
    max_time_gap_seconds = _number(coverage.get("max_time_gap_seconds"))
    blockers = _blockers(
        frame_gap_count=len(frame_gap_items),
        asset_gap_count=len(asset_gap_items),
        structure_gap_count=len(structure_gap_items),
        visual_route_gap_count=len(visual_route_gap_items),
        visual_analysis_gap_count=len(visual_analysis_gap_items),
        pending_repair_count=len(pending_repair_items),
        time_gap_count=time_gap_count,
        max_time_gap_seconds=max_time_gap_seconds,
    )
    optional_review_items = _optional_review_items(
        pending_review_count=max(len(timeline) - len(reviewed), 0),
        pending_structured_count=len(pending_structured),
        unreviewed_risk_count=len(unreviewed_risk_items),
        manual_repair_count=len(manual_repair_items),
    )
    warnings = []
    timeline_coverage_percent = _number(coverage.get("timeline_coverage_percent"))
    if timeline_coverage_percent and timeline_coverage_percent < 95:
        warnings.append(
            {
                "key": "low_timeline_coverage",
                "count": timeline_coverage_percent,
                "message": f"时间线覆盖率只有 {timeline_coverage_percent}%",
            }
        )
    if source_artifacts.get("source_count") and source_artifacts.get("sources_with_artifacts") < source_artifacts.get("source_count"):
        warnings.append(
            {
                "key": "missing_source_artifacts",
                "count": source_artifacts.get("source_count", 0) - source_artifacts.get("sources_with_artifacts", 0),
                "message": "有来源缺少原始抽取物回溯路径",
            }
        )
    ready = not blockers
    next_action = _with_executable_next_action(
        _next_action(blockers),
        blockers[0] if blockers else None,
        manifest=manifest,
        bundle_dir=bundle_dir,
    )
    return {
        "schema": "lecture_review_readiness.v1",
        "media_applicability": applicability,
        "checked_at": now_iso(),
        "ready": ready,
        "status": "ready" if ready else "blocked",
        "blockers": blockers,
        "optional_review_items": optional_review_items,
        "warnings": warnings,
        "counts": {
            "timeline_items": len(timeline),
            "reviewed_items": len(reviewed),
            "pending_review": max(len(timeline) - len(reviewed), 0),
            "risk_items": len(risk_items),
            "unreviewed_risk_items": len(unreviewed_risk_items),
            "structured_items": len(structured_items),
            "pending_structured": len(pending_structured),
            "frame_gap_items": len(frame_gap_items),
            "asset_gap_items": len(asset_gap_items),
            "structure_gap_items": len(structure_gap_items),
            "visual_route_gap_items": len(visual_route_gap_items),
            "visual_analysis_gap_items": len(visual_analysis_gap_items),
            "pending_repair_paths": len(pending_repair_items),
            "manual_repair_paths": len(manual_repair_items),
            "time_gap_count": time_gap_count,
            "max_time_gap_seconds": max_time_gap_seconds,
            "timeline_coverage_percent": timeline_coverage_percent,
            "source_count": source_artifacts.get("source_count", 0),
            "sources_with_artifacts": source_artifacts.get("sources_with_artifacts", 0),
            "source_artifact_count": source_artifacts.get("artifact_count", 0),
        },
        "source_artifacts": source_artifacts,
        "samples": {
            "pending_review": _sample_indexes([item for item in timeline if not _is_reviewed(item)]),
            "unreviewed_risk": _sample_indexes(unreviewed_risk_items),
            "pending_structured": _sample_indexes(pending_structured),
            "frame_gap": _sample_indexes(frame_gap_items),
            "asset_gap": _sample_indexes(asset_gap_items),
            "structure_gap": _sample_indexes(structure_gap_items),
            "visual_route_gap": _sample_indexes(visual_route_gap_items),
            "visual_analysis_gap": _sample_indexes(visual_analysis_gap_items),
        },
        "next_action": next_action,
    }


def _blockers(**counts: Any) -> list[dict[str, Any]]:
    rules = [
        ("frame_gap", counts["frame_gap_count"], "仍有关键帧或需保留图片缺口", "run_frame_recapture_or_manual_review"),
        ("asset_gap", counts["asset_gap_count"], "仍有关键帧资产未复制或文件不可用", "run_frame_recapture_or_manual_review"),
        ("visual_route_gap", counts["visual_route_gap_count"], "仍有画面未完成图文/语义/连续变化路由", "run_video_frame_router"),
        ("visual_analysis_gap", counts["visual_analysis_gap_count"], "仍有视频画面未完成多模态视觉理解", "run_visual_analysis"),
        ("structure_gap", counts["structure_gap_count"], "仍有视觉材料未结构化", "run_visual_structure_plan"),
        ("pending_repair", counts["pending_repair_count"], "仍有修复路径待运行或失败", "run_repair_tools"),
        ("time_gap", counts["time_gap_count"], "仍有时间轴空白段需要确认", "inspect_time_gap_frames"),
    ]
    return [
        {"key": key, "count": count, "message": message, "next_action": next_action}
        for key, count, message, next_action in rules
        if _number(count) > 0
    ]


def _optional_review_items(**counts: Any) -> list[dict[str, Any]]:
    rules = [
        ("pending_review", counts["pending_review_count"], "仍有时间线片段未人工确认", "review_pending_timeline"),
        ("pending_structured", counts["pending_structured_count"], "仍有公式/表格/代码片段未人工确认", "review_structured_material"),
        ("unreviewed_risk", counts["unreviewed_risk_count"], "仍有带缺口风险的片段未确认", "review_quality_gaps"),
        ("manual_repair", counts["manual_repair_count"], "仍有修复路径需要人工判断", "manual_repair_review"),
    ]
    return [
        {
            "key": key,
            "count": count,
            "message": message,
            "next_action": next_action,
            "required": False,
            "blocking": False,
        }
        for key, count, message, next_action in rules
        if _number(count) > 0
    ]


def _next_action(blockers: list[dict[str, Any]]) -> dict[str, Any]:
    if not blockers:
        return {
            "key": "export",
            "label": "可以导出",
            "hint": "复核 gate 已收口，可以刷新 WebUI/Obsidian 输出。",
            "human_required": False,
        }
    first = blockers[0]
    return {
        "key": str(first.get("next_action") or first.get("key") or "review"),
        "label": str(first.get("message") or "继续复核"),
        "hint": "先处理最高优先级 blocker，再重新运行 audit_bundle_readiness。",
        "human_required": _requires_human(first),
    }


def _with_executable_next_action(
    action: dict[str, Any],
    blocker: dict[str, Any] | None,
    *,
    manifest: dict[str, Any],
    bundle_dir: str | Path | None,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve() if bundle_dir else Path.cwd().resolve()
    tool, args_name = _action_target(action, blocker, manifest)
    args_path = _resolve_bundle_path(root, args_name)
    return {
        **action,
        "mcp_tool": tool,
        "mcp_args_path": str(args_path),
        "command": _mcp_command(tool, args_path),
    }


def _action_target(action: dict[str, Any], blocker: dict[str, Any] | None, manifest: dict[str, Any]) -> tuple[str, str]:
    key = str((blocker or {}).get("key") or action.get("key") or "")
    next_key = str(action.get("key") or "")
    if next_key == "export":
        return "refresh_lecture_review_outputs", str(manifest.get("mcp_refresh_args") or "mcp-refresh-lecture-review.args.json")
    if key in {"pending_review", "pending_structured", "unreviewed_risk", "manual_repair"}:
        return "prepare_review_session", str(manifest.get("mcp_review_session_args") or "mcp-prepare-review-session.args.json")
    if key == "asset_gap":
        return "repair_bundle_assets", str(manifest.get("mcp_asset_repair_args") or "mcp-repair-bundle-assets.args.json")
    if key in {"frame_gap", "time_gap"}:
        return "run_frame_recapture_plan", str(manifest.get("mcp_frame_recapture_args") or "mcp-run-frame-recapture.args.json")
    if key == "visual_route_gap":
        return "run_video_frame_router", str(manifest.get("mcp_video_frame_router_args") or "mcp-run-video-frame-router.args.json")
    if key == "visual_analysis_gap":
        route = _first_missing_visual_route(manifest)
        if route == "temporal_sequence":
            return "run_temporal_visual_analysis", str(manifest.get("mcp_temporal_visual_analysis_args") or "mcp-run-temporal-visual-analysis.args.json")
        return "run_multimodal_frame_analysis", str(manifest.get("mcp_multimodal_frame_analysis_args") or "mcp-run-multimodal-frame-analysis.args.json")
    if key == "structure_gap":
        return "run_visual_structure_plan", str(manifest.get("mcp_visual_structure_args") or "mcp-run-visual-structure.args.json")
    if key == "pending_repair":
        return "bundle_next_action", str(manifest.get("mcp_next_action_args") or "mcp-bundle-next-action.args.json")
    return "audit_bundle_readiness", str(manifest.get("mcp_readiness_args") or "mcp-audit-bundle-readiness.args.json")


def _requires_human(blocker: dict[str, Any]) -> bool:
    return str(blocker.get("key") or "") in {"pending_review", "pending_structured", "unreviewed_risk", "manual_repair"}


def _first_missing_visual_route(manifest: dict[str, Any]) -> str:
    quality = manifest.get("quality_audit") if isinstance(manifest.get("quality_audit"), dict) else {}
    for item in quality.get("priority_items") or []:
        if not isinstance(item, dict):
            continue
        issues = {str(issue) for issue in item.get("issues") or []}
        if "temporal_sequence_without_analysis" in issues:
            return "temporal_sequence"
    return "semantic_frame"


def _resolve_bundle_path(root: Path, value: Any) -> Path:
    text = str(value or "").strip()
    if not text:
        return root
    path = Path(text)
    return path if path.is_absolute() else root / path


def _mcp_command(tool: str, args_path: Path) -> str:
    escaped_args_path = str(args_path).replace("'", "''")
    return f".\\scripts\\video-knowledge.ps1 mcp-call {tool} '{escaped_args_path}'"


def _is_reviewed(item: dict[str, Any]) -> bool:
    human_review = item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
    return str(item.get("review_status") or human_review.get("status") or "").lower() in ACCEPTED_REVIEW_STATUSES


def _has_structured_material(item: dict[str, Any]) -> bool:
    material_types = {str(value) for value in item.get("material_types") or []}
    return bool(material_types & STRUCTURED_TYPES)


def _quality_issues(item: dict[str, Any]) -> list[str]:
    return [str(issue) for issue in item.get("quality_issues") or []]


def _has_asset_gap(item: dict[str, Any], *, bundle_dir: str | Path | None = None) -> bool:
    assets = item.get("assets") if isinstance(item.get("assets"), list) else []
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        if str(asset.get("copied") or "").lower() == "false":
            return True
        asset_path = str(asset.get("path") or "").strip()
        if bundle_dir and asset_path and not asset_path.startswith(("http://", "https://")):
            path = Path(asset_path)
            candidate = path if path.is_absolute() else Path(bundle_dir) / path
            if not candidate.exists():
                return True
    return False


def _repair_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    repair_status = manifest.get("repair_status") if isinstance(manifest.get("repair_status"), dict) else {}
    return [item for item in repair_status.get("items") or [] if isinstance(item, dict)]


def _sample_indexes(items: list[dict[str, Any]], limit: int = 10) -> list[int]:
    return [int(_number(item.get("index"))) for item in items[:limit] if _number(item.get("index")) > 0]


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

