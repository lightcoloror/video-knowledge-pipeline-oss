from __future__ import annotations

from pathlib import Path
from typing import Any

from .bundle_assets import repair_bundle_assets
from .bundle_readiness import build_bundle_readiness
from .config import public_vision_provider_profile, resolve_vision_execution_profile
from .frame_recapture import run_frame_recapture_plan
from .knowledge_coverage import audit_knowledge_coverage
from .lecture_workflow import refresh_lecture_review_outputs
from .models import now_iso
from .ocr_backfill import run_ocr_backfill
from .multimodal_frame_analyzer import run_multimodal_frame_analysis
from .panorama_analysis import PANORAMA_ANALYSIS_PLAN_SCHEMA
from .panorama_projection import (
    PANORAMA_PROJECTION_PLAN_SCHEMA,
    PANORAMA_VIEWPORT_PACK_SCHEMA,
    viewport_pack_sha256,
)
from .repair_status import build_repair_status, refresh_bundle_repair_status
from .storage import append_jsonl, read_json, read_jsonl, write_json
from .temporal_frame_groups import run_temporal_frame_groups
from .temporal_tag_delta import run_temporal_tag_delta
from .temporal_visual_analyzer import run_temporal_visual_analysis
from .timeline_alignment_audit import timeline_alignment_audit
from .transcript_semantic_correction import (
    import_transcript_semantic_candidate_suggestions,
    transcript_semantic_correction_closure,
    transcript_semantic_correction_status,
)
from .transcript_semantic_state import (
    semantic_candidate_resolution,
    semantic_discovery_scan_state,
)
from .video_frame_router import run_video_frame_router
from .vision_preflight import vision_execution_preflight
from .visual_evidence import has_visual_evidence
from .visual_structure import run_visual_structure_plan


def bundle_next_action(
    bundle_dir: str | Path,
    *,
    refresh: bool = True,
    include_panorama: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Recommend the next safe action; write=False also keeps refresh in memory."""
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if refresh and write:
        refresh_bundle_repair_status(root)
        audit_knowledge_coverage(root, write=True)
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")

    if refresh and not write:
        manifest = _readonly_refreshed_manifest(root, manifest)

    planner_options = {} if write else {"write": False}

    action = (
        (_next_panorama_action(root, manifest, **planner_options) if include_panorama else None)
        or _next_knowledge_coverage_action(root, manifest, **planner_options)
        or _next_repair_action(root, manifest)
        or _next_semantic_correction_action(root, manifest, **planner_options)
        or _next_summary_claim_review_action(root, manifest)
        or _next_readiness_action(root, manifest, **planner_options)
        or _export_action(root, manifest)
    )
    action = _provider_health_override(root, manifest, action, **planner_options)
    from .smart_summary_handoff import summary_handoff_action
    summary_action = summary_handoff_action(root)
    if summary_action and (action.get("status") == "ready" or action.get("human_required")
                           or action.get("kind") == "provider_repair"):
        action = {**summary_action, "upstream_next_action": action}
    from .workflow_completion import workflow_completion_status
    result = {
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "refreshed": bool(refresh),
        "status": action["status"],
        "next_action": action,
        "safe_smoke_action": _safe_smoke_action(root, manifest, action),
        "completion": workflow_completion_status(root),
    }
    args_path = root / "mcp-bundle-next-action.args.json"
    _write_action_args(args_path, {"bundle_dir": str(root), "refresh": True}, write=write)
    result["mcp_args_path"] = str(args_path)
    result["mcp_args_written"] = bool(write)
    return result


def _readonly_refreshed_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Reuse normal refresh builders without persisting or exposing the manifest."""
    snapshot = dict(manifest)
    timeline_path = root / "timeline.json"
    data = read_json(timeline_path) if timeline_path.exists() else []
    timeline = [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []
    snapshot["repair_status"] = build_repair_status(snapshot, timeline, bundle_dir=root)
    snapshot["repair_status"]["refreshed_at"] = now_iso()
    snapshot["review_readiness"] = build_bundle_readiness(snapshot, timeline, bundle_dir=root)
    coverage = audit_knowledge_coverage(root, write=False)["coverage"]
    snapshot["knowledge_coverage"] = coverage
    snapshot["media_applicability"] = coverage["media_applicability"]
    return snapshot


def _write_action_args(path: Path, payload: dict[str, Any], *, write: bool) -> None:
    """Persist planner arguments only in the explicitly writing mode."""
    if write:
        write_json(path, payload)


def _next_panorama_action(root: Path, manifest: dict[str, Any], *, write: bool = True) -> dict[str, Any] | None:
    """Project panorama artifacts into the existing one-action controller."""

    profile = manifest.get("panorama_profile")
    if not isinstance(profile, dict):
        return None
    if profile.get("schema") != "video_knowledge_pipeline.panorama_profile.v1" or profile.get("ready") is not True:
        return {
            "status": "blocked",
            "kind": "panorama_blocker",
            "key": "panorama_projection_confirmation",
            "label": "确认全景投影类型",
            "reason": "全景投影尚未由元数据或用户显式确认；在投影明确前禁止生成视口或调用视觉模型。",
            "blockers": list(profile.get("issues") or []),
            "mcp_tool": "panorama_doctor",
            "mcp_args_path": "",
            "human_required": True,
        }

    projection_ref = manifest.get("panorama_projection_plan")
    if not isinstance(projection_ref, dict):
        args_path = root / "mcp-panorama-prepare.args.json"
        _write_action_args(
            args_path,
            {
                "bundle_dir": str(root),
                "projection": str(profile.get("projection") or ""),
                "input_fov": profile.get("input_fov"),
                "lens_order": str(profile.get("lens_order") or ""),
                "frame_count": 8,
                "limit": 16,
                "write": True,
            },
            write=write,
        )
        return {
            "status": "panorama_pending",
            "kind": "panorama",
            "key": "panorama_prepare",
            "label": "准备全景投影视口计划",
            "reason": "投影类型已确认；先生成有界、Hash 绑定的六方向视口计划。",
            "mcp_tool": "panorama_prepare",
            "mcp_args_path": str(args_path),
            "human_required": False,
        }

    run_id = str(projection_ref.get("run_id") or "")
    plan_path = _panorama_bundle_file(root, projection_ref.get("path"))
    plan = _panorama_mapping(plan_path)
    if (
        not run_id
        or plan.get("schema") != PANORAMA_PROJECTION_PLAN_SCHEMA
        or str(plan.get("run_id") or "") != run_id
        or str(plan.get("plan_sha256") or "") != str(projection_ref.get("plan_sha256") or "")
    ):
        return _panorama_artifact_blocker("panorama_projection_plan_invalid", "全景投影计划缺失、越界或身份不匹配。")

    pack_path = root / "runs" / f"panorama-{run_id}" / "viewport-pack.json"
    pack = _panorama_mapping(pack_path)
    if not pack:
        args_path = root / "mcp-panorama-run.args.json"
        _write_action_args(
            args_path,
            {"bundle_dir": str(root), "run_id": run_id, "execute_projection": True},
            write=write,
        )
        return {
            "status": "panorama_pending",
            "kind": "panorama",
            "key": "panorama_projection_execute",
            "label": "执行有界全景视口投影",
            "reason": "投影计划已冻结；生成 Bundle 内六方向低分辨率派生帧，不读取模型或 Provider。",
            "mcp_tool": "panorama_run",
            "mcp_args_path": str(args_path),
            "run_id": run_id,
            "human_required": False,
        }
    if pack.get("schema") != PANORAMA_VIEWPORT_PACK_SCHEMA or viewport_pack_sha256(pack) != str(
        pack.get("task_pack_sha256") or ""
    ):
        return _panorama_artifact_blocker("panorama_viewport_pack_invalid", "全景视口包 Schema 或内容 Hash 不匹配。")

    analysis_ref = manifest.get("panorama_analysis_plan")
    analysis_path = _panorama_bundle_file(root, analysis_ref.get("path")) if isinstance(analysis_ref, dict) else None
    analysis = _panorama_mapping(analysis_path)
    if not analysis:
        args_path = root / "mcp-panorama-run.args.json"
        _write_action_args(
            args_path,
            {"bundle_dir": str(root), "run_id": run_id, "execute_projection": False},
            write=write,
        )
        return {
            "status": "panorama_pending",
            "kind": "panorama",
            "key": "panorama_analysis_triage",
            "label": "规划全景分级视觉任务",
            "reason": "视口包已验证；先按六面语义覆盖和最多三个方向连续帧生成有界任务计划。",
            "mcp_tool": "panorama_run",
            "mcp_args_path": str(args_path),
            "run_id": run_id,
            "human_required": False,
        }
    if (
        not isinstance(analysis_ref, dict)
        or analysis.get("schema") != PANORAMA_ANALYSIS_PLAN_SCHEMA
        or str(analysis.get("plan_sha256") or "") != str(analysis_ref.get("plan_sha256") or "")
        or str(analysis.get("viewport_pack_sha256") or "") != str(pack.get("task_pack_sha256") or "")
    ):
        return _panorama_artifact_blocker("panorama_analysis_plan_invalid", "全景分析计划与视口包身份不匹配。")

    timeline = read_json(root / "timeline.json")
    if not isinstance(timeline, list):
        return _panorama_artifact_blocker("panorama_timeline_invalid", "timeline.json 不是数组。")
    for task in analysis.get("tasks") or []:
        if not isinstance(task, dict) or _panorama_task_complete(timeline, task):
            continue
        route = task.get("route") if isinstance(task.get("route"), dict) else {}
        if route.get("status") == "unavailable" or route.get("route") == "review_gap":
            return {
                "status": "panorama_review_gap",
                "kind": "panorama_review_gap",
                "key": "panorama_visual_backend_unavailable",
                "label": "记录全景视觉后端缺口",
                "reason": str(route.get("reason") or "没有可用的本地、已授权在线或交互式视觉执行通道。"),
                "panorama_task_id": str(task.get("task_id") or ""),
                "human_required": False,
            }
        task_type = str(task.get("task_type") or "")
        tool = "prepare_panorama_vision_session" if task_type == "panorama_six_view_semantic" else "prepare_panorama_temporal_session"
        args_path = root / f"mcp-{tool.replace('_', '-')}.args.json"
        args = {
            "bundle_dir": str(root),
            "model_requested": "luna",
            "quality": "low",
            "write": True,
        }
        if task_type == "panorama_six_view_semantic":
            args.update(
                {
                    "viewport_pack_json": str(pack_path),
                    "timeline_index": int(task.get("timeline_index") or 0),
                    "sample_index": int(task.get("sample_index") or 0),
                }
            )
        elif task_type == "panorama_temporal_direction":
            args.update({"analysis_plan_json": str(analysis_path), "task_id": str(task.get("task_id") or "")})
        else:
            return _panorama_artifact_blocker("panorama_task_type_invalid", f"未知全景任务类型：{task_type}")
        _write_action_args(args_path, args, write=write)
        return {
            "status": "panorama_analysis_pending",
            "kind": "codex_session",
            "key": str(task.get("task_id") or "panorama_model_task"),
            "label": "分析六方向全景视口" if task_type == "panorama_six_view_semantic" else "分析单方向连续全景帧",
            "reason": "只处理当前一个有界全景任务；提交后由现有 Codex 会话导入器原子校验并写回。",
            "mcp_tool": tool,
            "mcp_args_path": str(args_path),
            "panorama_task": task,
            "analysis_plan_path": str(analysis_path),
            "viewport_pack_path": str(pack_path),
            "human_required": False,
        }
    return None


def _panorama_task_complete(timeline: list[Any], task: dict[str, Any]) -> bool:
    index = int(task.get("timeline_index") or 0)
    if not 1 <= index <= len(timeline) or not isinstance(timeline[index - 1], dict):
        return False
    item = timeline[index - 1]
    if str(task.get("task_type") or "") == "panorama_six_view_semantic":
        value = item.get("panorama_visual_understanding")
        sessions = value.get("sessions") if isinstance(value, dict) else []
        sample = int(task.get("sample_index") or 0)
        return any(int(row.get("sample_index") or 0) == sample for row in sessions or [] if isinstance(row, dict))
    if str(task.get("task_type") or "") == "panorama_temporal_direction":
        value = item.get("panorama_temporal_understanding")
        tracks = value.get("tracks") if isinstance(value, dict) else []
        direction = str(task.get("direction") or "")
        return any(str(row.get("direction") or "") == direction for row in tracks or [] if isinstance(row, dict))
    return False


def _panorama_bundle_file(root: Path, value: Any) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    return path if path.is_relative_to(root) and path.is_file() else None


def _panorama_mapping(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def _panorama_artifact_blocker(key: str, reason: str) -> dict[str, Any]:
    return {
        "status": "blocked",
        "kind": "panorama_blocker",
        "key": key,
        "label": "修复全景产物身份",
        "reason": reason,
        "human_required": True,
    }


def bundle_advance(
    bundle_dir: str | Path,
    *,
    execute: bool = False,
    refresh_outputs: bool = False,
    vault: str | Path | None = None,
    folder: str = "00_Inbox/AI/课程视频知识包",
    timeout_seconds: int = 30,
    ocr_input_json: str | Path | None = None,
    ocr_language: str = "chi_sim",
    captiocr_root: str | Path | None = None,
    visual_structure_input_json: str | Path | None = None,
    provider_config: dict[str, Any] | None = None,
    multimodal_limit: int | None = None,
    temporal_limit: int | None = None,
    frame_count: int | None = None,
    confirm_vision_calls: int | None = None,
    confirm_vision_indexes: str = "",
) -> dict[str, Any]:
    """Advance one safe bundle step based on bundle_next_action."""
    root = Path(bundle_dir).expanduser().resolve()
    profile = resolve_vision_execution_profile(
        provider_config=provider_config,
        multimodal_limit=multimodal_limit,
        temporal_limit=temporal_limit,
        frame_count=frame_count,
    )
    before = bundle_next_action(root, refresh=True)
    action = before.get("next_action") if isinstance(before.get("next_action"), dict) else {}
    if provider_config is not None and str(action.get("kind") or "") == "codex_session":
        # Explicit runtime Provider settings supersede a stale preflight that
        # previously lacked a key. The fresh execution preflight below remains
        # authoritative for this attempted Provider call.
        manifest = read_json(root / "manifest.json")
        if isinstance(manifest, dict):
            direct_action = (
                _next_knowledge_coverage_action(root, manifest)
                or _next_repair_action(root, manifest)
                or _next_semantic_correction_action(root, manifest)
                or _next_summary_claim_review_action(root, manifest)
                or _next_readiness_action(root, manifest)
                or _export_action(root, manifest)
            )
            if direct_action:
                action = direct_action
                before["next_action"] = direct_action
                before["status"] = direct_action["status"]
    status = str(action.get("status") or "")
    result: dict[str, Any] | None = None
    advanced = False
    blocked_reason = ""

    if status in {"repair_pending", "coverage_blocked", "semantic_correction_pending"}:
        result, advanced, blocked_reason = _advance_machine_action(
            root,
            action,
            execute=execute,
            timeout_seconds=timeout_seconds,
            ocr_input_json=ocr_input_json,
            ocr_language=ocr_language,
            captiocr_root=captiocr_root,
            visual_structure_input_json=visual_structure_input_json,
            provider_config=profile["provider_config"],
            multimodal_limit=profile["multimodal_limit"],
            temporal_limit=profile["temporal_limit"],
            frame_count=profile["frame_count"],
            confirm_vision_calls=confirm_vision_calls,
            confirm_vision_indexes=confirm_vision_indexes,
        )
    elif status == "summary_claim_review_pending":
        if not execute:
            blocked_reason = "pass execute=true to prepare the offline task; no model will be called"
        elif (action.get("next_task") or {}).get("prepared"):
            blocked_reason = "prepared review task awaits explicit result import; re-preparing is not progress"
        else:
            from .summary_claim_review import prepare_summary_claim_review
            result = prepare_summary_claim_review(root)
            advanced = bool(result.get("task_id"))
    elif status == "summary_claim_review_ready":
        if execute:
            from .smart_summary_global_reduce import run_smart_summary_global_reduce
            result = run_smart_summary_global_reduce(root, reuse_candidate=True, execute=False)
            advanced = bool(result.get('ok'))
            if not advanced:
                blocked_reason = 'existing summary candidate did not pass current source/quality gates'
        else:
            blocked_reason = 'pass execute=true to resume the existing candidate without a model call'
    elif status == "awaiting-agent":
        from .smart_summary_codex import generate_smart_summary_with_codex
        result = generate_smart_summary_with_codex(root)
        advanced = not result.get("reused", False) and result.get("workflow_status") == "awaiting-agent"
        blocked_reason = "Codex handoff awaits candidate Markdown; no model or background session was invoked"
    elif status == "ready":
        if refresh_outputs:
            result = _refresh_ready_outputs(root, vault=vault, folder=folder)
            advanced = True
        else:
            blocked_reason = "bundle is ready; pass refresh_outputs=true to refresh downstream outputs"
    else:
        blocked_reason = "human review or manual repair is required before machine advance"

    after = bundle_next_action(root, refresh=True)
    args_path = root / "mcp-bundle-advance.args.json"
    write_json(
        args_path,
        {
            "bundle_dir": str(root),
            "execute": False,
            "refresh_outputs": False,
            "folder": folder,
            "timeout_seconds": timeout_seconds,
            "ocr_language": ocr_language,
            "provider_config": public_vision_provider_profile(profile["provider_config"]),
            "multimodal_limit": profile["multimodal_limit"],
            "temporal_limit": profile["temporal_limit"],
            "frame_count": profile["frame_count"],
            "confirm_vision_calls": 0,
            "confirm_vision_indexes": "",
        },
    )
    output = {
        "bundle_dir": str(root),
        "status": "advanced" if advanced else "blocked",
        "advanced": advanced,
        "execute": bool(execute),
        "refresh_outputs": bool(refresh_outputs),
        "execution_profile": {
            "provider": public_vision_provider_profile(profile["provider_config"]),
            "multimodal_limit": profile["multimodal_limit"],
            "temporal_limit": profile["temporal_limit"],
            "frame_count": profile["frame_count"],
        },
        "blocked_reason": blocked_reason,
        "before": _compact_next_result(before),
        "action_result": _compact_action_result(result),
        "after": _compact_next_result(after),
        "mcp_args_path": str(args_path),
    }
    continuation = ((result or {}).get("summary") or {}).get("continuation") or {}
    if continuation:
        output["continuation_path"] = continuation.get("task_path", "")
        output["pending_count"] = continuation.get("pending_count", 0)
    return _write_advance_log(root, output)


def bundle_advance_log(bundle_dir: str | Path) -> dict[str, Any]:
    """Read and render the persisted bundle advance history."""
    root = Path(bundle_dir).expanduser().resolve()
    log_path = root / "bundle-advance-runs.jsonl"
    markdown_path = root / "bundle-advance-runs.md"
    rows = read_jsonl(log_path)
    markdown_path.write_text(_render_advance_log_markdown(root, rows), encoding="utf-8")
    return {
        "bundle_dir": str(root),
        "log_path": str(log_path),
        "markdown_path": str(markdown_path),
        "count": len(rows),
        "advances": rows,
        "last": rows[-1] if rows else {},
    }


def bundle_advance_queue(
    bundle_dir: str | Path,
    *,
    max_steps: int = 4,
    execute: bool = False,
    refresh_outputs: bool = False,
    vault: str | Path | None = None,
    folder: str = "00_Inbox/AI/课程视频知识包",
    timeout_seconds: int = 30,
    ocr_input_json: str | Path | None = None,
    ocr_language: str = "chi_sim",
    captiocr_root: str | Path | None = None,
    visual_structure_input_json: str | Path | None = None,
    provider_config: dict[str, Any] | None = None,
    multimodal_limit: int | None = None,
    temporal_limit: int | None = None,
    frame_count: int | None = None,
    confirm_vision_calls: int | None = None,
    confirm_vision_indexes: str = "",
) -> dict[str, Any]:
    """Advance a bundle until it is blocked, stalled, ready, or reaches max_steps."""
    root = Path(bundle_dir).expanduser().resolve()
    profile = resolve_vision_execution_profile(
        provider_config=provider_config,
        multimodal_limit=multimodal_limit,
        temporal_limit=temporal_limit,
        frame_count=frame_count,
    )
    steps: list[dict[str, Any]] = []
    stop_reason = "max_steps"
    max_count = max(int(max_steps or 1), 1)
    for _ in range(max_count):
        step = bundle_advance(
            root,
            execute=execute,
            refresh_outputs=refresh_outputs,
            vault=vault,
            folder=folder,
            timeout_seconds=timeout_seconds,
            ocr_input_json=ocr_input_json,
            ocr_language=ocr_language,
            captiocr_root=captiocr_root,
            visual_structure_input_json=visual_structure_input_json,
            provider_config=profile["provider_config"],
            multimodal_limit=profile["multimodal_limit"],
            temporal_limit=profile["temporal_limit"],
            frame_count=profile["frame_count"],
            confirm_vision_calls=confirm_vision_calls,
            confirm_vision_indexes=confirm_vision_indexes,
        )
        steps.append(step)
        before_action = _action(step.get("before"))
        after_action = _action(step.get("after"))
        if not step.get("advanced"):
            stop_reason = "blocked"
            break
        continuation = ((step.get("action_result") or {}).get("summary") or {}).get("continuation") or {}
        if continuation.get("pending_count"):
            # A successful bounded call does not authorize reusing its old
            # calls/index confirmation for a different batch.
            stop_reason = "next_batch_confirmation_required"
            break
        if before_action.get("status") == "ready":
            stop_reason = "refreshed_outputs" if refresh_outputs else "ready"
            break
        if _same_action(before_action, after_action):
            stop_reason = "stalled"
            break
    final = bundle_next_action(root, refresh=True)
    args_path = root / "mcp-bundle-advance-queue.args.json"
    write_json(
        args_path,
        {
            "bundle_dir": str(root),
            "max_steps": max_count,
            "execute": False,
            "refresh_outputs": False,
            "folder": folder,
            "timeout_seconds": timeout_seconds,
            "ocr_language": ocr_language,
            "provider_config": public_vision_provider_profile(profile["provider_config"]),
            "multimodal_limit": profile["multimodal_limit"],
            "temporal_limit": profile["temporal_limit"],
            "frame_count": profile["frame_count"],
            "confirm_vision_calls": 0,
            "confirm_vision_indexes": "",
        },
    )
    return {
        "bundle_dir": str(root),
        "status": "advanced" if any(step.get("advanced") for step in steps) else "blocked",
        "stop_reason": stop_reason,
        "continuation": (((steps[-1].get("action_result") or {}).get("summary") or {}).get("continuation") or {}) if steps else {},
        "continuation_path": (steps[-1].get("continuation_path") or "") if steps else "",
        "pending_count": (steps[-1].get("pending_count") or 0) if steps else 0,
        "max_steps": max_count,
        "step_count": len(steps),
        "execution_profile": {
            "provider": public_vision_provider_profile(profile["provider_config"]),
            "multimodal_limit": profile["multimodal_limit"],
            "temporal_limit": profile["temporal_limit"],
            "frame_count": profile["frame_count"],
        },
        "steps": steps,
        "final": final,
        "mcp_args_path": str(args_path),
        "advance_log": bundle_advance_log(root),
    }


def _next_repair_action(root: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    repair_status = manifest.get("repair_status") if isinstance(manifest.get("repair_status"), dict) else {}
    items = [item for item in repair_status.get("items") or [] if isinstance(item, dict)]
    next_item = next((item for item in items if str(item.get("status") or "") in {"pending", "ran_no_update", "failed"}), {})
    if not next_item and not items:
        next_item = repair_status.get("next") if isinstance(repair_status.get("next"), dict) else {}
    if str(next_item.get("status") or "") not in {"pending", "ran_no_update", "failed"}:
        return None
    if not next_item:
        return None
    tool = str(next_item.get("mcp_tool") or "")
    args_path = _resolve_bundle_path(root, next_item.get("mcp_args_path"))
    return {
        "status": "repair_pending",
        "kind": "repair",
        "key": str(next_item.get("key") or "repair"),
        "label": str(next_item.get("label") or "修复任务"),
        "reason": str(next_item.get("action_hint") or "先处理机器可执行的修复任务。"),
        "count": next_item.get("count", 0),
        "mcp_tool": tool,
        "mcp_args_path": str(args_path) if args_path else "",
        "command": _mcp_command(tool, args_path),
        "human_required": False,
    }


def _next_semantic_correction_action(root: Path, manifest: dict[str, Any], *, write: bool = True) -> dict[str, Any] | None:
    """Keep candidate discovery/review/closure ahead of final export.

    The semantic correction module remains the source of truth.  This adapter
    only translates its read-only status into one bounded next action; it does
    not infer that "no accepted replacement" means the correction cycle is
    closed.
    """

    status = transcript_semantic_correction_status(root, write=False)
    discovery_status = str(status.get("candidate_discovery_status") or "not_planned")
    closure_status = str(status.get("closure_status") or "missing")
    identity = status.get("artifact_identity") if isinstance(status.get("artifact_identity"), dict) else {}
    compact_status = _compact_semantic_correction_status(status)
    resolution = _semantic_candidate_resolution(root)
    discovery_scan = _semantic_discovery_scan_state(root)
    compact_status.update(
        {
            "adjudicated_candidate_count": resolution["adjudicated_candidate_count"],
            "unadjudicated_candidate_count": resolution["unadjudicated_candidate_count"],
            "candidate_discovery_scan": discovery_scan,
        }
    )

    discovery_complete = bool(discovery_scan["complete"])
    if not discovery_complete and discovery_status == "suggestions_ready":
        suggestions_path = _semantic_status_artifact_path(
            root,
            status.get("candidate_discovery_artifacts"),
            ("llm_suggestions_json", "llm_suggestions_markdown", "codex_suggestions_markdown"),
            "transcript-semantic-candidate-suggestions.llm.json",
        )
        args_path = root / "mcp-import-transcript-semantic-candidate-suggestions.args.json"
        _write_action_args(args_path, {"bundle_dir": str(root), "input_json": str(suggestions_path), "write": True}, write=write)
        return {
            "status": "semantic_correction_pending",
            "kind": "semantic_correction",
            "key": "semantic_candidate_import",
            "label": "导入语义纠错候选发现结果",
            "reason": "候选发现结果已生成但尚未进入权威 correction pack；先走现有候选导入校验，不能直接进入最终导出。",
            "semantic_status": compact_status,
            "input_json": str(suggestions_path),
            "mcp_tool": "import_transcript_semantic_candidate_suggestions",
            "mcp_args_path": str(args_path),
            "command": _mcp_command("import_transcript_semantic_candidate_suggestions", args_path),
            "human_required": False,
        }
    if not discovery_complete:
        args_path = root / "mcp-transcript-semantic-candidate-discovery-pack.args.json"
        _write_action_args(
            args_path,
            {
                "bundle_dir": str(root),
                "input_json": str(root / "transcript-semantic-correction-pack.json"),
                "limit": 8,
                "cursor": int(discovery_scan["next_cursor"] or 0),
                "write": True,
            },
            write=write,
        )
        return {
            "status": "semantic_correction_pending",
            "kind": "semantic_correction",
            "key": "semantic_candidate_discovery",
            "label": "扫描遗漏的语义纠错候选",
            "reason": "语义纠错候选发现尚未完成；先生成最多 8 个有界、Hash 绑定的文本窗口任务，再导入候选建议。",
            "semantic_status": compact_status,
            "discovery_cursor": int(discovery_scan["next_cursor"] or 0),
            "discovery_scan": discovery_scan,
            "mcp_tool": "transcript_semantic_candidate_discovery_pack",
            "mcp_args_path": str(args_path),
            "command": _mcp_command("transcript_semantic_candidate_discovery_pack", args_path),
            "human_required": False,
        }

    closure_current = (
        closure_status in {"completed", "completed_no_text_changes"}
        and identity.get("current") is not False
        and int(resolution["unadjudicated_candidate_count"]) == 0
    )
    if closure_current:
        # Rejected/deferred candidates remain review gaps, but a current formal
        # closure means no machine semantic action is left unscheduled.
        return None

    semantic_status = str(status.get("status") or "missing_pack")
    review_required_count = int(status.get("review_required_count") or 0)
    if (
        semantic_status == "needs_human_review_or_new_result"
        and review_required_count > 0
        and int(resolution["unadjudicated_candidate_count"]) == 0
    ):
        review_path = _semantic_status_artifact_path(
            root,
            status.get("artifacts"),
            ("review_markdown", "review_json"),
            "transcript-semantic-correction-review.md",
        )
        return {
            "status": "human_review_required",
            "kind": "human_review",
            "key": "semantic_correction_review_gap",
            "label": "复核高风险语义纠错候选",
            "reason": "现有强校验把候选留在 review_required；可作为 complete_draft 的明确缺口，但不得写为 accepted 或自动覆盖转写。",
            "semantic_status": compact_status,
            "review_path": str(review_path),
            "mcp_tool": "transcript_semantic_correction_status",
            "mcp_args_path": str(root / "mcp-transcript-semantic-correction-status.args.json"),
            "command": _mcp_command(
                "transcript_semantic_correction_status",
                root / "mcp-transcript-semantic-correction-status.args.json",
            ),
            "human_required": True,
        }

    if semantic_status == "needs_closure":
        result_path = _semantic_status_artifact_path(
            root,
            status.get("artifacts"),
            ("result_llm_json", "result_codex_markdown", "result_llm_markdown"),
            "",
        )
        if result_path is not None:
            args_path = root / "mcp-transcript-semantic-correction-closure.args.json"
            _write_action_args(
                args_path,
                {
                    "bundle_dir": str(root),
                    "input_json": str(result_path),
                    "min_confidence": 0.88,
                    "auto_apply": True,
                    "refresh_exports": False,
                    "write": True,
                },
                write=write,
            )
            return {
                "status": "semantic_correction_pending",
                "kind": "semantic_correction",
                "key": "semantic_correction_closure",
                "label": "闭合已验证的语义纠错结果",
                "reason": "语义结果已通过现有验证但尚未形成正式 closure；先原子闭合，再刷新最终导出。",
                "semantic_status": compact_status,
                "input_json": str(result_path),
                "mcp_tool": "transcript_semantic_correction_closure",
                "mcp_args_path": str(args_path),
                "command": _mcp_command("transcript_semantic_correction_closure", args_path),
                "human_required": False,
            }

    args_path = root / "mcp-transcript-semantic-correction-llm-draft.args.json"
    _write_action_args(
        args_path,
        {
            "bundle_dir": str(root),
            "input_json": str(root / "transcript-semantic-correction-pack.json"),
            "provider_config": {},
            "execute": False,
            "limit": 8,
            "min_confidence": 0.88,
            "write": True,
        },
        write=write,
    )
    return {
        "status": "semantic_correction_pending",
        "kind": "semantic_correction",
        "key": "semantic_correction_review",
        "label": "审查并闭合语义纠错候选",
        "reason": "候选已扫描但仍未完成强校验和正式 closure；仅向小模型发送最多 8 条 compact 候选，提交后走现有验证器。",
        "semantic_status": compact_status,
        "mcp_tool": "transcript_semantic_correction_llm_draft",
        "mcp_args_path": str(args_path),
        "command": _mcp_command("transcript_semantic_correction_llm_draft", args_path),
        "human_required": False,
    }


def _compact_semantic_correction_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        key: status.get(key)
        for key in (
            "status",
            "candidate_discovery_status",
            "candidate_count",
            "accepted_decision_count",
            "arbitrated_no_change_count",
            "review_required_count",
            "closure_status",
            "corrected_transcript_exists",
            "next_action_key",
        )
    }


def _semantic_discovery_scan_state(root: Path) -> dict[str, Any]:
    return semantic_discovery_scan_state(root)


def _semantic_candidate_resolution(root: Path) -> dict[str, Any]:
    return semantic_candidate_resolution(root)


def _semantic_status_artifact_path(
    root: Path,
    artifacts: Any,
    keys: tuple[str, ...],
    fallback: str,
) -> Path | None:
    values = artifacts if isinstance(artifacts, dict) else {}
    candidates = [
        _resolve_bundle_path(root, values.get(key))
        for key in keys
        if str(values.get(key) or "").strip()
    ]
    for path in candidates:
        if path and path.is_file():
            return path
    if candidates and candidates[0]:
        return candidates[0]
    return root / fallback if fallback else None


def _next_readiness_action(root: Path, manifest: dict[str, Any], *, write: bool = True) -> dict[str, Any] | None:
    readiness = manifest.get("review_readiness") if isinstance(manifest.get("review_readiness"), dict) else {}
    if readiness.get("ready") is True:
        return None
    next_action = readiness.get("next_action") if isinstance(readiness.get("next_action"), dict) else {}
    blockers = [item for item in readiness.get("blockers") or [] if isinstance(item, dict)]
    first = blockers[0] if blockers else {}
    key = str(next_action.get("key") or first.get("next_action") or first.get("key") or "review")
    human_required = key.startswith("review_") or key in {"manual_repair_review", "inspect_time_gap_frames", "run_frame_recapture_or_manual_review"}
    if _has_blocker(blockers, "asset_gap"):
        args_path = root / "mcp-repair-bundle-assets.args.json"
        _write_action_args(args_path, {"bundle_dir": str(root)}, write=write)
        return {
            "status": "repair_pending",
            "kind": "repair",
            "key": "bundle_assets",
            "label": "修复关键帧资产",
            "reason": "存在关键帧资产未复制或文件不可用，先尝试从记录的 source 路径重新复制。",
            "blockers": blockers,
            "mcp_tool": "repair_bundle_assets",
            "mcp_args_path": str(args_path),
            "command": _mcp_command("repair_bundle_assets", args_path),
            "human_required": False,
        }
    args_path = _resolve_bundle_path(root, manifest.get("mcp_readiness_args") or "mcp-audit-bundle-readiness.args.json")
    return {
        "status": "review_blocked",
        "kind": "human_review" if human_required else "readiness",
        "key": key,
        "label": str(next_action.get("label") or first.get("message") or "继续复核"),
        "reason": str(next_action.get("hint") or "先处理最高优先级 blocker，再重新检查 readiness。"),
        "blockers": blockers,
        "mcp_tool": "audit_bundle_readiness",
        "mcp_args_path": str(args_path) if args_path else "",
        "command": _mcp_command("audit_bundle_readiness", args_path),
        "human_required": human_required,
    }


def _next_knowledge_coverage_action(root: Path, manifest: dict[str, Any], *, write: bool = True) -> dict[str, Any] | None:
    coverage = manifest.get("knowledge_coverage") if isinstance(manifest.get("knowledge_coverage"), dict) else {}
    status = str(coverage.get("status") or "").strip().lower()
    if status != "blocked":
        return None
    next_action = coverage.get("next_action") if isinstance(coverage.get("next_action"), dict) else {}
    tool = str(next_action.get("mcp_tool") or "audit_knowledge_coverage")
    args_path = _resolve_bundle_path(root, next_action.get("mcp_args_path") or manifest.get("mcp_knowledge_coverage_args") or "mcp-audit-knowledge-coverage.args.json")
    key = str(next_action.get("key") or "knowledge_coverage")
    if key == "speech" and tool == "timeline_alignment_audit":
        args_path = root / "mcp-timeline-alignment-audit.args.json"
        _write_action_args(
            args_path,
            {"bundle_dir": str(root), "tolerance_seconds": 2.0, "write": True},
            write=write,
        )
    if key in {"screen_text", "structured_visual"} and (
        _visual_structure_ocr_empty_exhausted(manifest)
        or _visual_structure_has_no_candidates(manifest)
    ):
        if _has_pending_semantic_visual(root):
            args_path = _resolve_bundle_path(
                root,
                manifest.get("mcp_multimodal_frame_analysis_args")
                or "mcp-run-multimodal-frame-analysis.args.json",
            )
            return {
                "status": "coverage_blocked",
                "kind": "knowledge_coverage",
                "key": "multimodal_frame_analysis",
                "label": "OCR 无可执行候选，先补语义视觉理解",
                "reason": "结构化 OCR 当前没有可执行候选，但仍有未观察的语义帧；先调用现有视觉通道，屏幕文字缺口继续保留为 review gap。",
                "coverage_status": status,
                "blockers": coverage.get("blockers") if isinstance(coverage.get("blockers"), list) else [],
                "weak_channels": coverage.get("weak_channels") if isinstance(coverage.get("weak_channels"), list) else [],
                "mcp_tool": "run_multimodal_frame_analysis",
                "mcp_args_path": str(args_path) if args_path else "",
                "command": _mcp_command("run_multimodal_frame_analysis", args_path),
                "human_required": False,
                "evidence_boundary": "visual_understanding_does_not_close_screen_text",
            }
        args_path = _resolve_bundle_path(root, manifest.get("mcp_review_session_args") or "mcp-prepare-review-session.args.json")
        return {
            "status": "human_review_required",
            "kind": "human_review",
            "key": "ocr_text_empty_review",
            "label": "OCR 空结果需要人工审核或多模态补充",
            "reason": "ebook_markdown_pipeline 已经处理图文候选，但只返回包装/source 元数据，没有提取到真实屏幕文字；继续重跑 OCR 很可能空转。请保留证据帧，进入人工审核，或对这些帧执行多模态理解补充。",
            "coverage_status": status,
            "blockers": coverage.get("blockers") if isinstance(coverage.get("blockers"), list) else [],
            "weak_channels": coverage.get("weak_channels") if isinstance(coverage.get("weak_channels"), list) else [],
            "mcp_tool": "prepare_review_session",
            "mcp_args_path": str(args_path) if args_path else "",
            "command": _mcp_command("prepare_review_session", args_path),
            "fallback_mcp_tool": "run_multimodal_frame_analysis",
            "fallback_mcp_args_path": str(_resolve_bundle_path(root, manifest.get("mcp_multimodal_frame_analysis_args") or "mcp-run-multimodal-frame-analysis.args.json")),
            "human_required": True,
        }
    if key == "temporal_visual_understanding" and _needs_temporal_frame_groups(root):
        tool = "run_temporal_frame_groups"
        args_path = _resolve_bundle_path(root, manifest.get("mcp_temporal_frame_groups_args") or "mcp-run-temporal-frame-groups.args.json")
        key = "temporal_frame_groups"
        next_action = {
            **next_action,
            "label": "生成连续片段帧组",
            "hint": "连续片段理解前需要先为 temporal_sequence/mixed 时间片生成 5-12 帧顺序证据组。",
        }
    elif key == "temporal_visual_understanding" and _needs_temporal_tag_delta(root):
        tool = "run_temporal_tag_delta"
        args_path = root / "mcp-run-temporal-tag-delta.args.json"
        _write_action_args(
            args_path,
            {
                "bundle_dir": str(root),
                "indexes": "",
                "frame_count": 8,
                "limit": 16,
                "min_frames": 5,
                "execute_tagger": True,
                "write": True,
            },
            write=write,
        )
        key = "temporal_tag_delta"
        next_action = {
            **next_action,
            "label": "先做连续多帧低成本打标",
            "hint": "先用 Bundle 内 5-12 帧做候选弱证据；静态充分片段可弱证据收口，动态或冲突片段再升级连续帧多模态。",
        }
    return {
        "status": "coverage_blocked" if status == "blocked" else "coverage_weak",
        "kind": "knowledge_coverage",
        "key": key,
        "label": str(next_action.get("label") or "补齐知识覆盖"),
        "reason": str(next_action.get("hint") or "先检查知识通道覆盖缺口，避免在最终导出前漏掉语言、屏幕文字、图像或结构化视觉材料。"),
        "coverage_status": status,
        "blockers": coverage.get("blockers") if isinstance(coverage.get("blockers"), list) else [],
        "weak_channels": coverage.get("weak_channels") if isinstance(coverage.get("weak_channels"), list) else [],
        "mcp_tool": tool,
        "mcp_args_path": str(args_path) if args_path else "",
        "command": _mcp_command(tool, args_path),
        "human_required": tool == "audit_knowledge_coverage" or key in {"source_artifacts", "time_axis"},
    }


def _provider_health_override(root: Path, manifest: dict[str, Any], action: dict[str, Any], *, write: bool = True) -> dict[str, Any]:
    blocked_tool = str(action.get("mcp_tool") or "")
    if blocked_tool not in {"run_multimodal_frame_analysis", "run_temporal_visual_analysis"}:
        return action
    preflight = _latest_preflight_json(root, manifest)
    provider_health = preflight.get("provider_health") if isinstance(preflight.get("provider_health"), dict) else {}
    interactive = preflight.get("interactive_session") if isinstance(preflight.get("interactive_session"), dict) else {}
    blockers = preflight.get("blockers") if isinstance(preflight.get("blockers"), list) else []
    blocker_keys = [str(item.get("key") or "") for item in blockers if isinstance(item, dict)]
    provider_blocked = provider_health.get("safe_to_execute") is False or any(
        key in blocker_keys
        for key in {"missing_api_key", "provider_health_failed", "gateway_route_missing"}
    )
    if not provider_blocked:
        return action
    if interactive.get("ready_to_prepare"):
        kind = "semantic" if blocked_tool == "run_multimodal_frame_analysis" else "temporal"
        selected = preflight.get("selected_indexes") if isinstance(preflight.get("selected_indexes"), dict) else {}
        indexes = [int(value) for value in selected.get(kind) or [] if int(value) > 0]
        if kind == "temporal":
            indexes = indexes[:1]
        args_path = root / "mcp-prepare-codex-vision-session.args.json"
        _write_action_args(
            args_path,
            {
                "bundle_dir": str(root),
                "kind": kind,
                "indexes": indexes,
                "model_requested": "luna",
                "quality": "low",
                "executor": "codex_session",
                "write": True,
            },
            write=write,
        )
        index_arg = f" --indexes {','.join(str(value) for value in indexes)}" if indexes else ""
        root_quoted = "'" + str(root).replace("'", "''") + "'"
        return {
            "status": "interactive_session_ready",
            "kind": "codex_session",
            "key": f"prepare_codex_{kind}_vision_session",
            "label": "准备 Codex 单帧视觉包" if kind == "semantic" else "准备 Codex 连续帧包",
            "reason": "Provider 的 Key/health 当前阻塞，但 Bundle 内帧已经可以生成交互式 Codex 会话任务包；该路径不需要 API Key，也不会无人值守执行。",
            "for_blocked_action": {"key": action.get("key", ""), "mcp_tool": blocked_tool, "mcp_args_path": action.get("mcp_args_path", "")},
            "provider_health": {"status": provider_health.get("status", "not_checked"), "safe_to_execute": provider_health.get("safe_to_execute"), "error_class": provider_health.get("error_class", "")},
            "blocker_keys": blocker_keys,
            "mcp_tool": "prepare_codex_vision_session",
            "mcp_args_path": str(args_path),
            "command": f".\\scripts\\video-knowledge.ps1 prepare-codex-vision-session {root_quoted} --kind {kind}{index_arg}",
            "human_required": True,
            "unattended_execution": False,
        }
    matrix_args_path = _resolve_bundle_path(root, manifest.get("mcp_vision_provider_matrix_args") or "mcp-vision-provider-matrix.args.json")
    smoke_args_path = _resolve_bundle_path(root, manifest.get("mcp_vision_provider_smoke_args") or "mcp-vision-provider-smoke.args.json")
    tool = "vision_provider_matrix" if matrix_args_path else "vision_provider_smoke"
    args_path = matrix_args_path or smoke_args_path
    if tool == "vision_provider_matrix" and args_path:
        _write_action_args(
            args_path,
            {
                "providers": ["volcengine_coding_plan", "gemini", "openai", "agnes"],
                "bundle_dir": str(root),
                "output_dir": str(root),
                "timeout_seconds": 8,
                "write": True,
            },
            write=write,
        )
    return {
        "status": "provider_blocked",
        "kind": "provider_repair",
        "key": "provider_matrix_repair" if tool == "vision_provider_matrix" else "provider_repair",
        "label": "比较并修复多模态 Provider" if tool == "vision_provider_matrix" else "修复或切换多模态 Provider",
        "reason": "知识覆盖需要多模态视觉分析，但最新 provider health 显示当前 provider 不安全；先比较 provider matrix 或运行 provider smoke，确认文本/单图/多图 JSON 能通过后再执行真实模型写回。",
        "for_blocked_action": {
            "key": action.get("key", ""),
            "mcp_tool": blocked_tool,
            "mcp_args_path": action.get("mcp_args_path", ""),
        },
        "provider_health": {
            "status": provider_health.get("status", "not_checked"),
            "safe_to_execute": provider_health.get("safe_to_execute"),
            "error_class": provider_health.get("error_class", ""),
        },
        "blocker_keys": blocker_keys,
        "mcp_tool": tool,
        "mcp_args_path": str(args_path) if args_path else "",
        "command": _mcp_command(tool, args_path),
        "human_required": False,
    }


def _latest_preflight_json(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_bundle_path(root, manifest.get("vision_execution_preflight_json") or "vision-execution-preflight.json")
    if not path or not path.exists():
        return {}
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def _visual_structure_ocr_empty_exhausted(manifest: dict[str, Any]) -> bool:
    visual_structure = manifest.get("visual_structure") if isinstance(manifest.get("visual_structure"), dict) else {}
    last_run = visual_structure.get("last_run") if isinstance(visual_structure.get("last_run"), dict) else {}
    blockers = last_run.get("ebook_pipeline_blockers") if isinstance(last_run.get("ebook_pipeline_blockers"), dict) else {}
    total = int(last_run.get("ebook_pipeline_total") or 0)
    succeeded = int(last_run.get("ebook_pipeline_succeeded") or 0)
    empty_count = int(blockers.get("ocr_text_empty") or 0)
    if total > 0 and succeeded == 0 and empty_count == total:
        return True
    results = [item for item in visual_structure.get("ebook_pipeline_results") or [] if isinstance(item, dict)]
    if not results:
        return False
    return all(str(item.get("blocker") or "") == "ocr_text_empty" for item in results)


def _visual_structure_has_no_candidates(manifest: dict[str, Any]) -> bool:
    visual_structure = manifest.get("visual_structure") if isinstance(manifest.get("visual_structure"), dict) else {}
    last_run = visual_structure.get("last_run") if isinstance(visual_structure.get("last_run"), dict) else {}
    return bool(
        "total_candidates" in last_run
        and int(last_run.get("total_candidates") or 0) == 0
        and not list(last_run.get("selected_indexes") or [])
    )


def _has_pending_semantic_visual(root: Path) -> bool:
    timeline_path = root / "timeline.json"
    if not timeline_path.is_file():
        return False
    value = read_json(timeline_path)
    if not isinstance(value, list):
        return False
    return any(
        isinstance(item, dict)
        and str(item.get("visual_route") or "") in {"semantic_frame", "mixed"}
        and not has_visual_evidence(item)
        for item in value
    )


def _next_summary_claim_review_action(root: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    # Lazy import: lecture_workflow and UI adapters share this projection.
    from .summary_claim_review import summary_claim_review_status
    review = summary_claim_review_status(root)
    if not review['required']:
        return None
    if review['passed'] is True:
        if not review.get('resume_summary_required'):
            return None
        return {
            'status': 'summary_claim_review_ready', 'kind': 'summary_claim_review',
            'key': 'resume_reviewed_summary', 'label': '复用已审核候选完成总结',
            'reason': '来源审核已通过；调用既有 Reduce 恢复入口重新核对新鲜度，不重新生成或调用模型。',
            'mcp_tool': 'smart_summary_global_reduce',
            'mcp_args': {'bundle_dir': str(root), 'reuse_candidate': True, 'execute': False},
            'command': review['commands']['resume_summary'], 'human_required': False,
            'unattended_execution': False, 'review_status': review['status'],
        }
    pending = bool(review['next_task'])
    return {
        'status': 'summary_claim_review_pending' if pending else 'summary_claim_review_blocked',
        'kind': 'summary_claim_review', 'key': 'summary_claim_review',
        'label': '完成摘要逐条证据审核' if pending else '修复摘要证据审核缺口',
        'reason': '摘要审核尚未完成；保留来源、条件和反证，不得跳过审核直接导出。',
        'mcp_tool': 'prepare_summary_claim_review' if pending else 'summary_claim_review_status',
        'command': review['next_command'], 'mcp_args': {'bundle_dir': str(root)},
        'next_task': review['next_task'], 'review_counts': review['counts'],
        'review_status': review['status'], 'errors': review.get('errors', []),
        'human_required': not pending, 'unattended_execution': False,
    }


def _export_action(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    post_review = manifest.get("post_review") if isinstance(manifest.get("post_review"), dict) else {}
    tool = str(post_review.get("mcp_tool") or "refresh_lecture_review_outputs")
    args_path = _resolve_bundle_path(root, post_review.get("mcp_args_path") or manifest.get("mcp_refresh_args"))
    return {
        "status": "ready",
        "kind": "export",
        "key": "refresh_review_outputs",
        "label": "可以刷新导出",
        "reason": "repair_status 和 review_readiness 已收口，可以刷新 WebUI/Obsidian 输出。",
        "mcp_tool": tool,
        "mcp_args_path": str(args_path) if args_path else "",
        "command": _mcp_command(tool, args_path),
        "human_required": False,
    }


def _safe_smoke_action(root: Path, manifest: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    tool = str(action.get("mcp_tool") or "")
    key = str(action.get("key") or "")
    if tool not in {"run_multimodal_frame_analysis", "run_temporal_visual_analysis", "run_temporal_frame_groups"}:
        return {}
    args_path = _resolve_bundle_path(root, manifest.get("mcp_controlled_execution_smoke_args") or "mcp-controlled-execution-smoke.args.json")
    return {
        "status": "available",
        "kind": "controlled_execution_smoke",
        "key": "controlled_execution_smoke",
        "label": "本地执行演练",
        "reason": "在真实 provider/key 或确认参数就绪前，可先用本地 fixture 演练 preflight、确认写入、审计和可选回滚。",
        "for_next_action_key": key,
        "mcp_tool": "controlled_execution_smoke",
        "mcp_args_path": str(args_path) if args_path else "",
        "command": _mcp_command("controlled_execution_smoke", args_path),
        "human_required": False,
    }


def _resolve_bundle_path(root: Path, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else root / path


def _mcp_command(tool: str, args_path: Path | None) -> str:
    if not tool or not args_path:
        return ""
    return f".\\scripts\\video-knowledge.ps1 mcp-call {tool} {args_path}"


def _has_blocker(blockers: list[dict[str, Any]], key: str) -> bool:
    return any(str(blocker.get("key") or "") == key for blocker in blockers)


def _action(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    action = value.get("next_action")
    return action if isinstance(action, dict) else {}


def _same_action(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not left or not right:
        return False
    return (
        str(left.get("status") or "") == str(right.get("status") or "")
        and str(left.get("key") or "") == str(right.get("key") or "")
        and str(left.get("mcp_tool") or "") == str(right.get("mcp_tool") or "")
    )


def _advance_machine_action(
    root: Path,
    action: dict[str, Any],
    *,
    execute: bool,
    timeout_seconds: int,
    ocr_input_json: str | Path | None,
    ocr_language: str,
    captiocr_root: str | Path | None,
    visual_structure_input_json: str | Path | None,
    provider_config: dict[str, Any] | None,
    multimodal_limit: int,
    temporal_limit: int,
    frame_count: int,
    confirm_vision_calls: int | None,
    confirm_vision_indexes: str,
) -> tuple[dict[str, Any] | None, bool, str]:
    key = str(action.get("key") or "")
    tool = str(action.get("mcp_tool") or "")
    if key == "semantic_candidate_import" or tool == "import_transcript_semantic_candidate_suggestions":
        input_path = _resolve_bundle_path(root, action.get("input_json"))
        if input_path is None or not input_path.is_file():
            return None, False, "semantic candidate suggestions are missing or stale"
        result = import_transcript_semantic_candidate_suggestions(root, input_json=input_path, write=True)
        return result, str(result.get("status") or "") in {"imported", "no_candidates_imported"}, ""
    if key == "semantic_correction_closure" or tool == "transcript_semantic_correction_closure":
        input_path = _resolve_bundle_path(root, action.get("input_json"))
        if input_path is None or not input_path.is_file():
            return None, False, "validated semantic correction result is missing or stale"
        result = transcript_semantic_correction_closure(
            root,
            input_json=input_path,
            min_confidence=0.88,
            auto_apply=True,
            refresh_exports=False,
            write=True,
        )
        closed = str(result.get("status") or "") in {"completed", "completed_no_text_changes"}
        return result, closed, "" if closed else "semantic closure requires review"
    if key == "speech" and tool == "timeline_alignment_audit":
        return timeline_alignment_audit(root, write=True), True, ""
    if key in {"frame_recapture", "visual_frames"} or tool == "run_frame_recapture_plan":
        return run_frame_recapture_plan(root, execute=execute, timeout_seconds=timeout_seconds), True, ""
    if key == "screen_text" or tool == "run_visual_structure_plan":
        return run_visual_structure_plan(root, input_json=visual_structure_input_json), True, ""
    if key == "ocr_backfill" or tool == "run_ocr_backfill":
        return (
            run_ocr_backfill(
                root,
                input_json=ocr_input_json,
                execute=execute,
                language=ocr_language,
                captiocr_root=captiocr_root,
            ),
            True,
            "",
        )
    if key in {"visual_structure", "structured_visual"}:
        return run_visual_structure_plan(root, input_json=visual_structure_input_json), True, ""
    if key in {"video_frame_router", "visual_route_gap"} or tool == "run_video_frame_router":
        return run_video_frame_router(root, write=True), True, ""
    if key in {"multimodal_frame_analysis", "visual_analysis_gap"} or tool == "run_multimodal_frame_analysis":
        if execute:
            gate = _vision_execution_preflight_gate(
                root,
                provider_config,
                semantic_limit=multimodal_limit,
                temporal_limit=0,
                frame_count=frame_count,
                confirm_vision_calls=confirm_vision_calls,
                confirm_vision_indexes=confirm_vision_indexes,
            )
            if gate:
                return gate, False, _vision_gate_blocked_reason(gate)
        return (
            run_multimodal_frame_analysis(
                root,
                execute=execute,
                provider_config=provider_config,
                limit=multimodal_limit,
                confirm_vision_calls=confirm_vision_calls,
                confirm_vision_indexes=confirm_vision_indexes,
            ),
            True,
            "",
        )
    if key in {"temporal_frame_groups"} or tool == "run_temporal_frame_groups":
        return run_temporal_frame_groups(root, execute=execute, frame_count=frame_count, limit=temporal_limit, timeout_seconds=timeout_seconds), True, ""
    if key == "temporal_tag_delta" or tool == "run_temporal_tag_delta":
        return (
            run_temporal_tag_delta(
                root,
                execute_tagger=execute,
                frame_count=max(5, min(12, int(frame_count))),
                limit=min(16, max(1, int(temporal_limit))) if int(temporal_limit) > 0 else 16,
                min_frames=5,
                write=True,
            ),
            True,
            "",
        )
    if key in {"temporal_visual_analysis"} or tool == "run_temporal_visual_analysis":
        if execute:
            gate = _vision_execution_preflight_gate(
                root,
                provider_config,
                semantic_limit=0,
                temporal_limit=temporal_limit,
                frame_count=frame_count,
                confirm_vision_calls=confirm_vision_calls,
                confirm_vision_indexes=confirm_vision_indexes,
            )
            if gate:
                return gate, False, _vision_gate_blocked_reason(gate)
        return (
            run_temporal_visual_analysis(
                root,
                execute=execute,
                provider_config=provider_config,
                limit=temporal_limit,
                frame_count=frame_count,
                confirm_vision_calls=confirm_vision_calls,
                confirm_vision_indexes=confirm_vision_indexes,
            ),
            True,
            "",
        )
    if key == "bundle_assets" or tool == "repair_bundle_assets":
        return repair_bundle_assets(root), True, ""
    if key == "knowledge_coverage" or tool == "audit_knowledge_coverage":
        return audit_knowledge_coverage(root, write=True), True, ""
    return None, False, f"unsupported machine action: {key or tool}"


def _write_advance_log(root: Path, result: dict[str, Any]) -> dict[str, Any]:
    record = _advance_record(result)
    append_jsonl(root / "bundle-advance-runs.jsonl", [record])
    log = bundle_advance_log(root)
    result["advance_log"] = {
        "record": record,
        "log_path": log["log_path"],
        "markdown_path": log["markdown_path"],
        "count": log["count"],
    }
    return result


def _vision_execution_preflight_gate(
    root: Path,
    provider_config: dict[str, Any] | None,
    *,
    semantic_limit: int,
    temporal_limit: int,
    frame_count: int,
    confirm_vision_calls: int | None,
    confirm_vision_indexes: str,
) -> dict[str, Any] | None:
    preflight = vision_execution_preflight(
        root,
        provider_config=provider_config,
        semantic_limit=semantic_limit,
        temporal_limit=temporal_limit,
        frame_count=frame_count,
        include_semantic=semantic_limit > 0,
        include_temporal=temporal_limit > 0,
        write=True,
    )
    if preflight.get("ready_to_execute"):
        confirmation = _vision_execution_confirmation_gate(
            preflight,
            confirm_vision_calls=confirm_vision_calls,
            confirm_vision_indexes=confirm_vision_indexes,
        )
        if not confirmation:
            return None
        return confirmation
    return {
        "schema": "lecture_bundle_advance_vision_preflight_gate.v1",
        "status": "vision_preflight_blocked",
        "bundle_dir": str(root),
        "preflight_path": preflight.get("preflight_path", ""),
        "preflight_json_path": preflight.get("preflight_json_path", ""),
        "blockers": preflight.get("blockers") if isinstance(preflight.get("blockers"), list) else [],
        "summary": {
            "provider": preflight.get("provider") if isinstance(preflight.get("provider"), dict) else {},
            "ready_to_execute": False,
            "expected_api_calls": preflight.get("expected_api_calls", 0),
            "candidate_counts": preflight.get("candidate_counts") if isinstance(preflight.get("candidate_counts"), dict) else {},
            "blocker_keys": [str(item.get("key") or "") for item in preflight.get("blockers", []) if isinstance(item, dict)],
            "interactive_session": preflight.get("interactive_session") if isinstance(preflight.get("interactive_session"), dict) else {},
            "hint": "Prepare the interactive Codex vision task pack when interactive_session.ready_to_prepare=true; otherwise inspect vision-execution-preflight.md and fix Provider blockers.",
        },
    }


def _vision_execution_confirmation_gate(
    preflight: dict[str, Any],
    *,
    confirm_vision_calls: int | None,
    confirm_vision_indexes: str,
) -> dict[str, Any] | None:
    expected_calls = int(preflight.get("expected_api_calls") or 0)
    expected_indexes = _preflight_selected_index_string(preflight)
    calls_ok = confirm_vision_calls is not None and int(confirm_vision_calls) == expected_calls
    indexes_ok = str(confirm_vision_indexes or "").strip() == expected_indexes
    if calls_ok and indexes_ok:
        return None
    return {
        "schema": "lecture_bundle_advance_vision_confirmation_gate.v1",
        "status": "vision_confirmation_required",
        "bundle_dir": str(preflight.get("bundle_dir") or ""),
        "preflight_path": preflight.get("preflight_path", ""),
        "preflight_json_path": preflight.get("preflight_json_path", ""),
        "summary": {
            "ready_to_execute": True,
            "expected_api_calls": expected_calls,
            "expected_indexes": expected_indexes,
            "received_confirm_vision_calls": confirm_vision_calls,
            "received_confirm_vision_indexes": str(confirm_vision_indexes or ""),
            "hint": "Retry with confirm_vision_calls and confirm_vision_indexes exactly matching the preflight output.",
        },
    }


def _preflight_selected_index_string(preflight: dict[str, Any]) -> str:
    selected = preflight.get("selected_indexes") if isinstance(preflight.get("selected_indexes"), dict) else {}
    indexes: list[int] = []
    for key in ("semantic", "temporal"):
        for value in selected.get(key) or []:
            try:
                indexes.append(int(value))
            except (TypeError, ValueError):
                continue
    return ",".join(str(index) for index in indexes)


def _vision_gate_blocked_reason(gate: dict[str, Any]) -> str:
    if str(gate.get("status") or "") == "vision_confirmation_required":
        return "vision execution confirmation required; match confirm_vision_calls and confirm_vision_indexes from preflight"
    return "vision execution preflight blocked; inspect the preflight report before execute"


def _needs_temporal_frame_groups(root: Path) -> bool:
    timeline_path = root / "timeline.json"
    if not timeline_path.exists():
        return False
    data = read_json(timeline_path)
    timeline = data if isinstance(data, list) else []
    for item in timeline:
        if not isinstance(item, dict):
            continue
        if str(item.get("visual_route") or "") not in {"temporal_sequence", "mixed"}:
            continue
        paths = item.get("temporal_frame_paths")
        if not isinstance(paths, list) or len(paths) < 5:
            return True
    return False


def _needs_temporal_tag_delta(root: Path) -> bool:
    timeline_path = root / "timeline.json"
    if not timeline_path.is_file():
        return False
    timeline = read_json(timeline_path)
    if not isinstance(timeline, list):
        return False
    for item in timeline:
        if not isinstance(item, dict):
            continue
        route = str(item.get("visual_route") or "")
        if route not in {"temporal_sequence", "mixed"}:
            continue
        if isinstance(item.get("temporal_visual_understanding"), dict) and item["temporal_visual_understanding"]:
            continue
        paths = item.get("temporal_frame_paths")
        tag_delta = item.get("temporal_tag_delta")
        if isinstance(paths, list) and len(paths) >= 5 and not (
            isinstance(tag_delta, dict)
            and str(tag_delta.get("schema") or "") == "video_knowledge_pipeline.temporal_tag_delta.v2"
        ):
            return True
    return False


def _advance_record(result: dict[str, Any]) -> dict[str, Any]:
    before = result.get("before") if isinstance(result.get("before"), dict) else {}
    after = result.get("after") if isinstance(result.get("after"), dict) else {}
    before_action = before.get("next_action") if isinstance(before.get("next_action"), dict) else {}
    after_action = after.get("next_action") if isinstance(after.get("next_action"), dict) else {}
    action_result = result.get("action_result") if isinstance(result.get("action_result"), dict) else {}
    summary = action_result.get("summary") if isinstance(action_result.get("summary"), dict) else {}
    artifacts = _advance_artifacts(action_result)
    return {
        "created_at": now_iso(),
        "status": result.get("status", ""),
        "advanced": bool(result.get("advanced")),
        "execute": bool(result.get("execute")),
        "refresh_outputs": bool(result.get("refresh_outputs")),
        "blocked_reason": result.get("blocked_reason", ""),
        "before_status": before_action.get("status", ""),
        "before_key": before_action.get("key", ""),
        "before_tool": before_action.get("mcp_tool", ""),
        "after_status": after_action.get("status", ""),
        "after_key": after_action.get("key", ""),
        "after_tool": after_action.get("mcp_tool", ""),
        "action_summary": summary,
        "action_artifacts": artifacts,
    }


def _advance_artifacts(action_result: dict[str, Any]) -> dict[str, Any]:
    if not action_result:
        return {}
    artifacts: dict[str, Any] = {}
    for key in ("report_path", "manifest_path", "timeline_path", "asset_manifest_path", "preflight_path", "preflight_json_path"):
        value = action_result.get(key)
        if value:
            artifacts[key] = str(value)
    backfill = action_result.get("backfill") if isinstance(action_result.get("backfill"), dict) else {}
    if backfill:
        artifacts["updated_indexes"] = list(backfill.get("updated_indexes") or [])[:20]
        artifacts["updated_count"] = backfill.get("updated", 0)
        artifacts["source_package_updated"] = bool(backfill.get("source_package_updated"))
    if isinstance(action_result.get("missing"), list):
        artifacts["missing_paths"] = [_artifact_path(item, prefer="source") for item in action_result.get("missing", [])[:10]]
    if isinstance(action_result.get("copied"), list):
        artifacts["copied_paths"] = [_artifact_path(item, prefer="path") for item in action_result.get("copied", [])[:10]]
    webui_bundle = action_result.get("webui_bundle") if isinstance(action_result.get("webui_bundle"), dict) else {}
    if webui_bundle:
        for key in ("bundle_dir", "manifest_path", "review_html_path", "note_path"):
            value = webui_bundle.get(key)
            if value:
                artifacts[f"webui_{key}"] = str(value)
    obsidian_export = action_result.get("obsidian_export") if isinstance(action_result.get("obsidian_export"), dict) else {}
    if obsidian_export:
        for key in (
            "path",
            "note_path",
            "output_path",
            "vault",
            "folder",
            "export_manifest_path",
            "export_status_path",
            "export_status_markdown_path",
            "asset_manifest_path",
            "source_artifact_index_path",
        ):
            value = obsidian_export.get(key)
            if value:
                artifacts[f"obsidian_{key}"] = str(value)
        if isinstance(obsidian_export.get("export_status"), dict):
            artifacts["obsidian_export_status"] = obsidian_export["export_status"]
        if obsidian_export.get("mcp_status_args_path"):
            artifacts["obsidian_mcp_status_args_path"] = str(obsidian_export.get("mcp_status_args_path"))
        if obsidian_export.get("mcp_export_args_path"):
            artifacts["obsidian_mcp_export_args_path"] = str(obsidian_export.get("mcp_export_args_path"))
        entrypoints = obsidian_export.get("agent_entrypoints") if isinstance(obsidian_export.get("agent_entrypoints"), dict) else {}
        if not entrypoints and isinstance(obsidian_export.get("pages_by_name"), dict):
            folder = str(obsidian_export.get("folder") or "")
            if folder:
                entrypoints = {
                    "full_speech_text": str(Path(folder) / "transcript.md"),
                    "full_screen_text": str(Path(folder) / "screen-text.md"),
                    "evidence_map": str(Path(folder) / "evidence-map.md"),
                    "structured_materials": str(Path(folder) / "structured-materials.md"),
                    "source_artifacts_json": str(Path(folder) / "source-artifacts.json"),
                }
        for key in (
            "full_speech_text",
            "full_screen_text",
            "timeline",
            "evidence_map",
            "structured_materials",
            "source_artifacts_json",
            "asset_manifest",
        ):
            value = entrypoints.get(key)
            if value:
                artifacts[f"obsidian_{key}"] = str(value)
    if isinstance(action_result.get("obsidian_export_blocked"), dict):
        artifacts["obsidian_export_blocked"] = action_result["obsidian_export_blocked"]
    if isinstance(action_result.get("vision_restore_hint"), dict):
        hint = action_result["vision_restore_hint"]
        for key in ("run_id", "kind", "restore_plan_command", "audit_jsonl_path", "audit_markdown_path"):
            value = hint.get(key)
            if value:
                artifacts[f"vision_{key}"] = value
    if action_result.get("review_json"):
        artifacts["review_json"] = str(action_result.get("review_json"))
    if action_result.get("default_review_notes_created"):
        artifacts["default_review_notes_created"] = True
    return artifacts


def _compact_action_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    compact: dict[str, Any] = {}
    for key in (
        "schema",
        "status",
        "bundle_dir",
        "manifest_path",
        "timeline_path",
        "report_path",
        "report_markdown_path",
        "coverage_path",
        "coverage_markdown_path",
        "preflight_path",
        "preflight_json_path",
        "input_template_json",
        "note_path",
        "full_transcript_path",
        "full_body_path",
        "mcp_args_path",
        "task_id",
        "task_pack_sha256",
        "phase",
    ):
        if key in result:
            compact[key] = result[key]
    if isinstance(result.get("summary"), dict):
        compact["summary"] = result["summary"]
    if isinstance(result.get("run_audit"), dict):
        compact["run_audit"] = _compact_run_audit(result["run_audit"])
        hint = result.get("vision_restore_hint") if isinstance(result.get("vision_restore_hint"), dict) else _vision_restore_hint(result)
        if hint:
            compact["vision_restore_hint"] = hint
    if isinstance(result.get("coverage"), dict):
        compact["coverage"] = _compact_coverage(result["coverage"])
    if isinstance(result.get("backfill"), dict):
        compact["backfill"] = {
            "updated": result["backfill"].get("updated", 0),
            "updated_indexes": list(result["backfill"].get("updated_indexes") or [])[:20],
            "source_package_updated": bool(result["backfill"].get("source_package_updated")),
        }
    if isinstance(result.get("items"), list):
        compact["item_count"] = len(result["items"])
    return compact


def _compact_run_audit(run_audit: dict[str, Any]) -> dict[str, Any]:
    record = run_audit.get("record") if isinstance(run_audit.get("record"), dict) else {}
    return {
        "run_id": str(record.get("run_id") or ""),
        "kind": str(record.get("kind") or ""),
        "execute": bool(record.get("execute")),
        "status": str(record.get("status") or ""),
        "updated_count": int(record.get("updated_count") or 0),
        "timeline_diff_count": int(record.get("timeline_diff_count") or 0),
        "execution_control": record.get("execution_control") if isinstance(record.get("execution_control"), dict) else {},
        "jsonl_path": str(run_audit.get("jsonl_path") or ""),
        "markdown_path": str(run_audit.get("markdown_path") or ""),
    }


def _vision_restore_hint(result: dict[str, Any]) -> dict[str, Any]:
    run_audit = result.get("run_audit") if isinstance(result.get("run_audit"), dict) else {}
    record = run_audit.get("record") if isinstance(run_audit.get("record"), dict) else {}
    run_id = str(record.get("run_id") or "")
    if not run_id:
        return {}
    bundle_dir = str(result.get("bundle_dir") or record.get("bundle_dir") or "")
    command = f'python -m video_knowledge_pipeline.cli vision-analysis-restore-plan "{bundle_dir}" --run-id {run_id}' if bundle_dir else ""
    return {
        "run_id": run_id,
        "kind": str(record.get("kind") or ""),
        "updated_count": int(record.get("updated_count") or 0),
        "timeline_diff_count": int(record.get("timeline_diff_count") or 0),
        "audit_jsonl_path": str(run_audit.get("jsonl_path") or ""),
        "audit_markdown_path": str(run_audit.get("markdown_path") or ""),
        "restore_plan_command": command,
    }


def _compact_next_result(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact = {
        "bundle_dir": value.get("bundle_dir", ""),
        "manifest_path": value.get("manifest_path", ""),
        "refreshed": bool(value.get("refreshed")),
        "status": value.get("status", ""),
        "mcp_args_path": value.get("mcp_args_path", ""),
    }
    if isinstance(value.get("next_action"), dict):
        compact["next_action"] = _compact_next_action(value["next_action"])
    return compact


def _compact_next_action(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keep = [
        "status",
        "kind",
        "key",
        "label",
        "reason",
        "coverage_status",
        "mcp_tool",
        "mcp_args_path",
        "command",
        "human_required",
        "unattended_execution",
        "next_task",
        "mcp_args",
        "review_counts",
        "review_status",
        "errors",
    ]
    compact = {key: value[key] for key in keep if key in value}
    blockers = value.get("blockers") if isinstance(value.get("blockers"), list) else []
    weak = value.get("weak_channels") if isinstance(value.get("weak_channels"), list) else []
    if blockers:
        compact["blocker_count"] = len(blockers)
        compact["blocker_keys"] = [str(item.get("key") or "") for item in blockers[:5] if isinstance(item, dict)]
    if weak:
        compact["weak_count"] = len(weak)
        compact["weak_keys"] = [str(item.get("key") or "") for item in weak[:5] if isinstance(item, dict)]
    return compact


def _compact_coverage(value: dict[str, Any]) -> dict[str, Any]:
    blockers = value.get("blockers") if isinstance(value.get("blockers"), list) else []
    weak = value.get("weak_channels") if isinstance(value.get("weak_channels"), list) else []
    return {
        "status": value.get("status", ""),
        "timeline_items": value.get("timeline_items", 0),
        "items_with_visual_route": value.get("items_with_visual_route", 0),
        "items_with_visual_understanding": value.get("items_with_visual_understanding", 0),
        "items_with_temporal_understanding": value.get("items_with_temporal_understanding", 0),
        "missing_visual_understanding": value.get("missing_visual_understanding", 0),
        "blocker_count": len(blockers),
        "weak_count": len(weak),
        "blocker_keys": [str(item.get("key") or "") for item in blockers[:5] if isinstance(item, dict)],
        "weak_keys": [str(item.get("key") or "") for item in weak[:5] if isinstance(item, dict)],
    }


def _artifact_path(item: Any, *, prefer: str) -> str:
    if isinstance(item, dict):
        fallback = "source" if prefer == "path" else "path"
        return str(item.get(prefer) or item.get(fallback) or "")
    return str(item or "")


def _render_advance_log_markdown(root: Path, rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Bundle Advance Runs",
        "",
        f"- Bundle: `{root}`",
        f"- Count: {len(rows)}",
        "",
        "| Time | Status | From | To | Execute | Refresh | Reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        execute = "yes" if row.get("execute") else "no"
        refresh = "yes" if row.get("refresh_outputs") else "no"
        before = row.get("before_key") or row.get("before_status") or "-"
        after = row.get("after_key") or row.get("after_status") or "-"
        reason = str(row.get("blocked_reason") or "")
        lines.append(
            f"| {row.get('created_at', '')} | {row.get('status', '')} | `{before}` | `{after}` | {execute} | {refresh} | {reason} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _refresh_ready_outputs(root: Path, *, vault: str | Path | None, folder: str) -> dict[str, Any]:
    args_path = root / "mcp-refresh-lecture-review.args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"refresh args not found: {args_path}")
    args = read_json(args_path)
    if not isinstance(args, dict):
        raise ValueError("mcp-refresh-lecture-review.args.json must be a JSON object")
    result = refresh_lecture_review_outputs(
        args["project"],
        args["review_json"],
        webui_output_dir=args.get("webui_output_dir") or str(root),
        vault=vault,
        folder=folder,
        target=args.get("target") or "bilinote",
        allow_blocked_export=True,
    )
    return result

