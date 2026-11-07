from __future__ import annotations

from pathlib import Path
from typing import Any

from .bundle_assets import repair_bundle_assets
from .bundle_next import (
    _readonly_refreshed_manifest,
    bundle_advance,
    bundle_advance_log,
    bundle_advance_queue,
    bundle_next_action,
)
from .bundle_readiness import audit_bundle_readiness
from .bundle_source_artifacts import bundle_source_artifacts
from .frame_recapture import run_frame_recapture_plan
from .knowledge_coverage import audit_knowledge_coverage
from .knowledge_note_export import export_knowledge_note
from .lecture_workflow import refresh_lecture_review_outputs
from .markdown_text import markdown_table_cell as _md_cell
from .mcp_args_audit import audit_bundle_mcp_args
from .models import now_iso
from .multimodal_frame_analyzer import run_multimodal_frame_analysis
from .ocr_backfill import run_ocr_backfill
from .repair_status import refresh_bundle_repair_status
from .review_session import (
    apply_review_notes_to_bundle,
    prepare_review_session,
    review_closure_status,
)
from .run_artifact_registry import build_run_artifact_registry
from .screen_text_recovery import run_screen_text_recovery
from .storage import bundle_write_lock, read_json, read_jsonl, write_json
from .temporal_frame_groups import run_temporal_frame_groups
from .temporal_visual_analyzer import run_temporal_visual_analysis
from .video_frame_router import run_video_frame_router
from .vision_acceptance import vision_acceptance_plan
from .vision_preflight import vision_execution_preflight
from .visual_structure import run_visual_structure_plan

STATUS_SCHEMA = "lecture_bundle_status_report.v1"
CONTROLLED_EXECUTION_CHECK_SCHEMA = "lecture_controlled_execution_check.v1"


def bundle_status_report(bundle_dir: str | Path, *, refresh: bool = True, write: bool = True) -> dict[str, Any]:
    """Build a compact status report; persist it only when write=True."""
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    next_result = bundle_next_action(root, refresh=refresh, write=write)
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    if refresh and not write:
        manifest = _readonly_refreshed_manifest(root, manifest)

    report_path = root / "bundle-status.json"
    markdown_path = root / "bundle-status.md"
    args_path = root / "mcp-bundle-status-report.args.json"
    report = {
        "schema": STATUS_SCHEMA,
        "checked_at": now_iso(),
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "title": str(manifest.get("title") or root.name),
        "target": str(manifest.get("target") or ""),
        "status": _overall_status(next_result),
        "completion": next_result.get("completion", {}),
        "completion_status": (next_result.get("completion") or {}).get("status", "blocked"),
        "refreshed": bool(refresh),
        "next_action": next_result.get("next_action") if isinstance(next_result.get("next_action"), dict) else {},
        "safe_smoke_action": next_result.get("safe_smoke_action") if isinstance(next_result.get("safe_smoke_action"), dict) else {},
        "review_readiness": _review_readiness_summary(manifest),
        "review_closure": _review_closure_summary(root),
        "knowledge_coverage": _knowledge_coverage_summary(manifest),
        "source_artifacts": _source_artifact_summary(root, manifest),
        "entrypoints": _entrypoints(root, manifest),
        "repair_handoffs": _repair_handoffs(root, manifest),
        "mcp_entrypoints": _mcp_entrypoints(root, manifest),
        "mcp_commands": _mcp_commands(root, manifest),
        "mcp_args_audit": _mcp_args_audit_summary(root),
        "latest_advance": _latest_advance(root),
        "controlled_execution": _controlled_execution_summary(root, manifest),
        "obsidian": _obsidian_summary(root),
        "report_path": str(report_path),
        "report_markdown_path": str(markdown_path),
        "mcp_args_path": str(args_path),
    }
    if write:
        with bundle_write_lock(root, operation="bundle_status_report"):
            manifest["bundle_status_report"] = "bundle-status.md"
            manifest["bundle_status_report_json"] = "bundle-status.json"
            manifest["mcp_status_report_args"] = "mcp-bundle-status-report.args.json"
            write_json(manifest_path, manifest)
            write_json(report_path, report)
            markdown_path.write_text(render_bundle_status_report_markdown(report), encoding="utf-8")
            write_json(args_path, {"bundle_dir": str(root), "refresh": True, "write": True})
    return report


def controlled_execution_check(bundle_dir: str | Path, *, refresh: bool = False, write: bool = True) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    status = bundle_status_report(root, refresh=refresh, write=False)
    controlled = status.get("controlled_execution") if isinstance(status.get("controlled_execution"), dict) else {}
    mcp_args_audit = status.get("mcp_args_audit") if isinstance(status.get("mcp_args_audit"), dict) else {}
    checklist = _controlled_execution_checklist(controlled, mcp_args_audit=mcp_args_audit)
    ready = all(item.get("ok") for item in checklist)
    report_path = root / "controlled-execution-check.json"
    markdown_path = root / "controlled-execution-check.md"
    args_path = root / "mcp-controlled-execution-check.args.json"
    report = {
        "schema": CONTROLLED_EXECUTION_CHECK_SCHEMA,
        "checked_at": now_iso(),
        "bundle_dir": str(root),
        "ready_for_real_vision_execution": bool(ready),
        "status": "ready" if ready else "blocked",
        "checklist": checklist,
        "controlled_execution": controlled,
        "mcp_args_audit": mcp_args_audit,
        "next_steps": _controlled_execution_next_steps(controlled, checklist),
        "report_path": str(report_path),
        "report_markdown_path": str(markdown_path),
        "mcp_args_path": str(args_path),
    }
    if write:
        with bundle_write_lock(root, operation="controlled_execution_check"):
            manifest_path = root / "manifest.json"
            manifest = read_json(manifest_path) if manifest_path.exists() else {}
            if isinstance(manifest, dict):
                manifest["controlled_execution_check"] = "controlled-execution-check.md"
                manifest["controlled_execution_check_json"] = "controlled-execution-check.json"
                manifest["mcp_controlled_execution_check_args"] = "mcp-controlled-execution-check.args.json"
                write_json(manifest_path, manifest)
            write_json(report_path, report)
            markdown_path.write_text(render_controlled_execution_check_markdown(report), encoding="utf-8")
            write_json(args_path, {"bundle_dir": str(root), "refresh": False, "write": True})
    return report


def render_controlled_execution_check_markdown(report: dict[str, Any]) -> str:
    controlled = report.get("controlled_execution") if isinstance(report.get("controlled_execution"), dict) else {}
    mcp_audit = report.get("mcp_args_audit") if isinstance(report.get("mcp_args_audit"), dict) else {}
    lines = [
        "# Controlled Execution Check",
        "",
        f"- Bundle: `{report.get('bundle_dir', '')}`",
        f"- Ready: `{report.get('ready_for_real_vision_execution', False)}`",
        f"- Status: `{report.get('status', '')}`",
        f"- Confirm calls: `{controlled.get('latest_execution_control_expected_calls', '')}`",
        f"- Confirm indexes: `{controlled.get('latest_execution_control_expected_indexes', '')}`",
        f"- Provider health: `{controlled.get('provider_health_status', 'not_checked')}` / safe `{controlled.get('provider_health_safe_to_execute', None)}` / `{controlled.get('provider_health_error_class', '')}`",
        f"- Latest write: updated `{controlled.get('latest_vision_run_updated_count', 0)}`, changed `{controlled.get('latest_vision_run_timeline_diff_count', 0)}`",
        f"- MCP args audit: `{mcp_audit.get('status', 'unknown')}` ({mcp_audit.get('ok_count', 0)}/{mcp_audit.get('total', 0)} ok)",
        "",
        "| Check | OK | Detail |",
        "|---|---:|---|",
    ]
    for item in report.get("checklist") or []:
        if isinstance(item, dict):
            lines.append(f"| `{item.get('key', '')}` | {item.get('ok', False)} | {_md_cell(str(item.get('detail') or ''))} |")
    lines.extend(["", "## Next Steps", ""])
    for item in report.get("next_steps") or []:
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def render_bundle_status_report_markdown(report: dict[str, Any]) -> str:
    next_action = report.get("next_action") if isinstance(report.get("next_action"), dict) else {}
    safe_smoke = report.get("safe_smoke_action") if isinstance(report.get("safe_smoke_action"), dict) else {}
    readiness = report.get("review_readiness") if isinstance(report.get("review_readiness"), dict) else {}
    closure = report.get("review_closure") if isinstance(report.get("review_closure"), dict) else {}
    coverage = report.get("knowledge_coverage") if isinstance(report.get("knowledge_coverage"), dict) else {}
    source = report.get("source_artifacts") if isinstance(report.get("source_artifacts"), dict) else {}
    obsidian = report.get("obsidian") if isinstance(report.get("obsidian"), dict) else {}
    controlled = report.get("controlled_execution") if isinstance(report.get("controlled_execution"), dict) else {}
    mcp_audit = report.get("mcp_args_audit") if isinstance(report.get("mcp_args_audit"), dict) else {}
    lines = [
        "---",
        "type: lecture-bundle-status",
        f'created: "{report.get("checked_at", now_iso())}"',
        "---",
        "",
        f"# Bundle 状态：{report.get('title', '')}",
        "",
        f"- 动作状态：`{report.get('status', 'unknown')}`",
        f"- 全流程验收：`{report.get('completion_status', 'blocked')}`",
        f"- Bundle：`{report.get('bundle_dir', '')}`",
        f"- 下一步：`{next_action.get('key', '')}` / {next_action.get('label', '')}",
        f"- 建议工具：`{next_action.get('mcp_tool', '')}`",
        f"- MCP args：`{next_action.get('mcp_args_path', '')}`",
        f"- 本地演练：`{safe_smoke.get('mcp_tool', '')}` / `{safe_smoke.get('mcp_args_path', '')}`",
        f"- 复核 gate：`{readiness.get('status', 'unknown')}`，blockers {readiness.get('blocker_count', 0)}",
        f"- 可选人工复核：{readiness.get('optional_review_count', 0)} 项，不阻塞导出",
        f"- 复核进度：open `{closure.get('open', 0)}`，closed `{closure.get('closed', 0)}`，imported `{closure.get('imported', 0)}`",
        f"- 知识覆盖：`{coverage.get('status', 'unknown')}`，blockers {coverage.get('blocker_count', 0)}，weak {coverage.get('weak_count', 0)}",
        f"- 原始抽取物：{source.get('available_count', 0)}/{source.get('artifact_count', 0)} available，missing {source.get('missing_count', 0)}",
        f"- Obsidian：`{obsidian.get('status', 'not_exported')}`",
        f"- 可控执行：`{controlled.get('status', 'unknown')}`，preflight `{controlled.get('preflight_status', 'missing')}`，restore `{controlled.get('restore_status', 'missing')}`",
        f"- Provider health：`{controlled.get('provider_health_status', 'not_checked')}` / safe `{controlled.get('provider_health_safe_to_execute', None)}` / `{controlled.get('provider_health_error_class', '')}`",
        f"- MCP args 审计：`{mcp_audit.get('status', 'unknown')}`，ok {mcp_audit.get('ok_count', 0)}/{mcp_audit.get('total', 0)}，blocked {mcp_audit.get('blocked_count', 0)}",
        "",
        "## Agent 入口",
        "",
        "| Key | Exists | Path |",
        "|---|---:|---|",
    ]
    for row in report.get("entrypoints") or []:
        if isinstance(row, dict):
            lines.append(f"| {row.get('key', '')} | {row.get('exists', False)} | `{row.get('path', '')}` |")
    completion = report.get("completion") or {}
    lines.extend(["", "## 逐阶段完成判断", "", "| 阶段 | 状态 | 下一动作 |", "|---|---|---|"])
    for row in completion.get("stages", []):
        lines.append(f"| {row['key']} | `{row['status']}` | {_md_cell(str(row.get('next_action') or ''))} |")
    lines.extend(["", "## Repair Handoffs", "", "| Key | Kind | Exists | Path |", "|---|---|---:|---|"])
    handoffs = [row for row in report.get("repair_handoffs") or [] if isinstance(row, dict)]
    if handoffs:
        for row in handoffs:
            lines.append(
                f"| {row.get('key', '')} | `{row.get('kind', '')}` | {row.get('exists', False)} | `{row.get('path', '')}` |"
            )
    else:
        lines.append("| none |  | False |  |")
    if closure:
        lines.extend(
            [
                "",
                "## Review Closure",
                "",
                f"- Status report: `{closure.get('report_markdown_path', '')}`",
                f"- Todo JSON: `{closure.get('todo_json', '')}`",
                f"- Next batch: `{_md_cell(str(closure.get('next_batch_command') or ''))}`",
            ]
        )
    lines.extend(["", "## MCP 入口", "", "| Key | Exists | Path |", "|---|---:|---|"])
    for row in report.get("mcp_entrypoints") or []:
        if isinstance(row, dict):
            lines.append(f"| {row.get('key', '')} | {row.get('exists', False)} | `{row.get('path', '')}` |")
    lines.extend(["", "## MCP 命令", "", "| Key | Tool | Exists | Command |", "|---|---|---:|---|"])
    for row in report.get("mcp_commands") or []:
        if isinstance(row, dict):
            lines.append(
                "| {key} | `{tool}` | {exists} | `{command}` |".format(
                    key=row.get("key", ""),
                    tool=row.get("mcp_tool", ""),
                    exists=row.get("exists", False),
                    command=_md_cell(str(row.get("command") or "")),
                )
            )
    latest = report.get("latest_advance") if isinstance(report.get("latest_advance"), dict) else {}
    if latest:
        lines.extend(
            [
                "",
                "## 最近推进",
                "",
                f"- 时间：`{latest.get('created_at', '')}`",
                f"- 状态：`{latest.get('status', '')}`",
                f"- 从：`{latest.get('before_key', '')}` 到 `{latest.get('after_key', '')}`",
            ]
        )
    if controlled:
        lines.extend(
            [
                "",
                "## 可控真实执行",
                "",
                f"- 状态：`{controlled.get('status', '')}`",
                f"- Preflight：`{controlled.get('preflight_status', '')}`",
                f"- Provider health：`{controlled.get('provider_health_status', 'not_checked')}` / safe `{controlled.get('provider_health_safe_to_execute', None)}` / `{controlled.get('provider_health_error_class', '')}`",
                f"- 确认 gate：`{controlled.get('confirmation_status', '')}`",
                f"- 审计：`{controlled.get('audit_status', '')}`",
                f"- 恢复：`{controlled.get('restore_status', '')}`",
                f"- 最近视觉 run：`{controlled.get('latest_vision_run_id', '')}`",
                f"- 最近写入：updated `{controlled.get('latest_vision_run_updated_count', 0)}`，changed `{controlled.get('latest_vision_run_timeline_diff_count', 0)}`，recoverable `{controlled.get('latest_vision_write_recoverable', False)}`",
                f"- 确认值：calls `{controlled.get('latest_execution_control_expected_calls', '')}`，indexes `{controlled.get('latest_execution_control_expected_indexes', '')}`",
                f"- 收到确认：calls `{controlled.get('latest_execution_control_received_calls', '')}`，indexes `{controlled.get('latest_execution_control_received_indexes', '')}`",
                f"- 恢复计划命令：`{_md_cell(str(controlled.get('restore_plan_command') or ''))}`",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _overall_status(next_result: dict[str, Any]) -> str:
    action = next_result.get("next_action") if isinstance(next_result.get("next_action"), dict) else {}
    status = str(action.get("status") or "").strip()
    if status == "ready":
        return "ready"
    if status in {"repair_pending", "coverage_blocked"}:
        return "machine_action_available"
    if status in {"review_blocked", "coverage_weak"}:
        return "needs_human_review"
    return status or "unknown"


def _review_readiness_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    readiness = manifest.get("review_readiness") if isinstance(manifest.get("review_readiness"), dict) else {}
    blockers = readiness.get("blockers") if isinstance(readiness.get("blockers"), list) else []
    optional = readiness.get("optional_review_items") if isinstance(readiness.get("optional_review_items"), list) else []
    counts = readiness.get("counts") if isinstance(readiness.get("counts"), dict) else {}
    return {
        "status": str(readiness.get("status") or "unknown"),
        "ready": bool(readiness.get("ready")),
        "blocker_count": len(blockers),
        "optional_review_count": len(optional),
        "optional_review_items": optional,
        "next_action": readiness.get("next_action") if isinstance(readiness.get("next_action"), dict) else {},
        "counts": counts,
    }


def _knowledge_coverage_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    coverage = manifest.get("knowledge_coverage") if isinstance(manifest.get("knowledge_coverage"), dict) else {}
    blockers = coverage.get("blockers") if isinstance(coverage.get("blockers"), list) else []
    weak = coverage.get("weak_channels") if isinstance(coverage.get("weak_channels"), list) else []
    return {
        "status": str(coverage.get("status") or "unknown"),
        "timeline_items": coverage.get("timeline_items", 0),
        "blocker_count": len(blockers),
        "weak_count": len(weak),
        "next_action": coverage.get("next_action") if isinstance(coverage.get("next_action"), dict) else {},
    }


def _review_closure_summary(root: Path) -> dict[str, Any]:
    try:
        result = review_closure_status(root, write=False)
    except Exception:
        return {}
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    next_batch = result.get("next_batch") if isinstance(result.get("next_batch"), dict) else {}
    return {
        "open": int(summary.get("open") or 0),
        "closed": int(summary.get("closed") or 0),
        "imported": int(summary.get("imported") or 0),
        "invalid": int(summary.get("invalid") or 0),
        "report_markdown_path": result.get("report_markdown_path", ""),
        "report_path": result.get("report_path", ""),
        "todo_json": next_batch.get("todo_json", ""),
        "next_batch_command": next_batch.get("command", ""),
    }


def _source_artifact_summary(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = _resolve(root, manifest.get("source_artifacts_json") or "source-artifacts.json")
    if not path.exists():
        return {"path": str(path), "exists": False, "artifact_count": 0, "available_count": 0, "missing_count": 0}
    data = read_json(path)
    if not isinstance(data, dict):
        return {"path": str(path), "exists": True, "artifact_count": 0, "available_count": 0, "missing_count": 0}
    return {
        "path": str(path),
        "exists": True,
        "source_count": data.get("source_count", 0),
        "artifact_count": data.get("artifact_count", 0),
        "available_count": data.get("available_count", 0),
        "missing_count": data.get("missing_count", 0),
        "tools": data.get("tools") if isinstance(data.get("tools"), list) else [],
    }


def _entrypoints(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [
        "entry_note",
        "timeline_json",
        "review_html",
        "review_notes",
        "knowledge_coverage_markdown",
        "knowledge_coverage_json",
        "source_artifacts",
        "source_artifacts_json",
        "asset_manifest",
    ]
    return [_path_row(key, _resolve(root, manifest.get(key))) for key in keys]


def _repair_handoffs(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("frame_recapture", "ocr_backfill", "visual_structure"):
        section = manifest.get(key) if isinstance(manifest.get(key), dict) else {}
        last_run = section.get("last_run") if isinstance(section.get("last_run"), dict) else {}
        for kind, field in (("markdown", "handoff_markdown"), ("json", "handoff_json")):
            path_text = str(section.get(field) or "").strip()
            if not path_text:
                continue
            path = _resolve(root, path_text)
            rows.append(
                {
                    "key": key,
                    "kind": kind,
                    "field": field,
                    "path": str(path),
                    "exists": path.exists(),
                    "status": str(last_run.get("status") or section.get("status") or ""),
                }
            )
    return rows


def _mcp_entrypoints(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [key for key in sorted(manifest) if key.startswith("mcp_") and key.endswith("_args")]
    return [_path_row(key, _resolve(root, manifest.get(key))) for key in keys]


def _mcp_commands(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, tool in _MCP_TOOL_BY_MANIFEST_KEY.items():
        if key not in manifest:
            continue
        args_path = _resolve(root, manifest.get(key))
        rows.append(
            {
                "key": key,
                "mcp_tool": tool,
                "mcp_args_path": str(args_path),
                "exists": args_path.exists(),
                "command": _mcp_call_command(tool, args_path),
            }
        )
    return rows


def _latest_advance(root: Path) -> dict[str, Any]:
    rows = read_jsonl(root / "bundle-advance-runs.jsonl")
    return rows[-1] if rows else {}


def _controlled_execution_summary(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    from .media_applicability import resolve_media_applicability
    applicability = resolve_media_applicability(root, manifest)
    if applicability["visual_status"] == "not_applicable":
        return {"status": "not_applicable", "media_applicability": applicability,
                "preflight_status": "not_applicable", "preflight_ready_to_execute": False,
                "provider_health_status": "not_applicable", "provider_health_safe_to_execute": None,
                "confirmation_status": "not_applicable", "audit_status": "not_applicable",
                "restore_status": "not_applicable", "blockers": [],
                "latest_vision_run_execute": False, "latest_vision_write_recoverable": False}
    latest = _latest_advance(root)
    artifacts = latest.get("action_artifacts") if isinstance(latest.get("action_artifacts"), dict) else {}
    action_summary = latest.get("action_summary") if isinstance(latest.get("action_summary"), dict) else {}
    vision_runs_path = root / "vision-analysis-runs.jsonl"
    vision_rows = read_jsonl(vision_runs_path) if vision_runs_path.exists() else []
    latest_vision = vision_rows[-1] if vision_rows else {}
    latest_control = latest_vision.get("execution_control") if isinstance(latest_vision.get("execution_control"), dict) else {}
    preflight_path = _resolve(
        root,
        manifest.get("vision_execution_preflight")
        or artifacts.get("preflight_path")
        or latest_control.get("preflight_path")
        or "vision-execution-preflight.md",
    )
    preflight_json_path = _resolve(
        root,
        manifest.get("vision_execution_preflight_json")
        or artifacts.get("preflight_json_path")
        or latest_control.get("preflight_json_path")
        or "vision-execution-preflight.json",
    )
    restore_plan_path = root / "vision-restore-plan.json"
    restore_runs_path = root / "vision-restore-runs.jsonl"
    restore_rows = read_jsonl(restore_runs_path) if restore_runs_path.exists() else []
    latest_status = str(latest.get("status") or "")
    blocked_reason = str(latest.get("blocked_reason") or "")
    latest_vision_status = str(latest_vision.get("status") or "")
    latest_vision_execute = bool(latest_vision.get("execute"))
    latest_vision_updated_count = _int_value(latest_vision.get("updated_count"))
    latest_vision_diff_count = _int_value(latest_vision.get("timeline_diff_count"))
    latest_vision_write_recoverable = latest_vision_updated_count > 0 or latest_vision_diff_count > 0
    latest_control_status = str(latest_control.get("status") or "")
    latest_control_confirmed = bool(latest_control.get("confirmed")) if latest_control else False
    preflight_status = "present" if preflight_path.exists() and preflight_json_path.exists() else "missing"
    preflight_json = read_json(preflight_json_path) if preflight_json_path.exists() else {}
    preflight_blockers = preflight_json.get("blockers") if isinstance(preflight_json, dict) and isinstance(preflight_json.get("blockers"), list) else []
    preflight_blocker_keys = [str(item.get("key") or "") for item in preflight_blockers if isinstance(item, dict)]
    provider_health = preflight_json.get("provider_health") if isinstance(preflight_json, dict) and isinstance(preflight_json.get("provider_health"), dict) else {}
    provider_health_status = str(provider_health.get("status") or "not_checked")
    provider_health_safe = provider_health.get("safe_to_execute")
    provider_health_error_class = str(provider_health.get("error_class") or "")
    direct_confirmation_required = latest_control_status == "vision_confirmation_required"
    advance_confirmation_required = "confirmation required" in blocked_reason and not latest_control_status
    provider_not_ready = latest_control_status == "vision_provider_not_ready" or latest_vision_status == "vision_provider_not_ready"
    if advance_confirmation_required or direct_confirmation_required:
        confirmation_status = "required"
    elif latest_control_confirmed:
        confirmation_status = "confirmed"
    else:
        confirmation_status = "passed_or_not_needed"
    audit_status = "present" if vision_rows else "missing"
    restore_status = "available" if restore_plan_path.exists() or restore_rows else "not_planned"
    latest_run_id = str(artifacts.get("vision_run_id") or latest_vision.get("run_id") or "")
    restore_command = str(artifacts.get("vision_restore_plan_command") or "")
    if not restore_command and latest_run_id:
        restore_command = f'python -m video_knowledge_pipeline.cli vision-analysis-restore-plan "{root}" --run-id {latest_run_id}'
    blockers: list[str] = []
    if preflight_status == "missing":
        blockers.append("missing_preflight")
    if latest_status == "blocked" and "preflight blocked" in blocked_reason:
        blockers.append("preflight_blocked")
    if "provider_health_failed" in preflight_blocker_keys or provider_health_safe is False:
        blockers.append("provider_health_failed")
    if (latest_status == "blocked" and advance_confirmation_required) or direct_confirmation_required:
        blockers.append("confirmation_required")
    if provider_not_ready:
        blockers.append("vision_provider_not_ready")
    if not vision_rows:
        blockers.append("no_vision_run_audit")
    elif latest_vision_execute and not latest_vision_write_recoverable:
        blockers.append("no_recoverable_vision_write")
    status = "ready_for_controlled_execution" if preflight_status == "present" and not blockers else "controls_incomplete"
    if latest_status == "blocked" and blockers:
        status = "blocked"
    if "confirmation_required" in blockers:
        status = "blocked"
    if "provider_health_failed" in blockers:
        status = "blocked"
    if latest_run_id and restore_command and latest_vision_write_recoverable:
        status = "vision_run_recoverable" if status != "blocked" else status
    return {
        "status": status,
        "preflight_status": preflight_status,
        "preflight_ready_to_execute": bool(preflight_json.get("ready_to_execute")) if isinstance(preflight_json, dict) else False,
        "preflight_blocker_keys": preflight_blocker_keys,
        "provider_health_status": provider_health_status,
        "provider_health_safe_to_execute": provider_health_safe,
        "provider_health_error_class": provider_health_error_class,
        "confirmation_status": confirmation_status,
        "audit_status": audit_status,
        "restore_status": restore_status,
        "blockers": blockers,
        "preflight_path": str(preflight_path),
        "preflight_json_path": str(preflight_json_path),
        "vision_runs_path": str(vision_runs_path),
        "restore_plan_path": str(restore_plan_path),
        "restore_runs_path": str(restore_runs_path),
        "latest_vision_run_id": latest_run_id,
        "latest_vision_run_status": latest_vision_status,
        "latest_vision_run_execute": latest_vision_execute,
        "latest_vision_run_updated_count": latest_vision_updated_count,
        "latest_vision_run_timeline_diff_count": latest_vision_diff_count,
        "latest_vision_write_recoverable": latest_vision_write_recoverable,
        "latest_advance_blocked_reason": blocked_reason,
        "latest_execution_control_status": latest_control_status,
        "latest_execution_control_confirmed": latest_control_confirmed,
        "latest_execution_control_expected_calls": latest_control.get("expected_api_calls", ""),
        "latest_execution_control_expected_indexes": latest_control.get("expected_indexes", ""),
        "latest_execution_control_received_calls": latest_control.get("received_confirm_vision_calls", ""),
        "latest_execution_control_received_indexes": latest_control.get("received_confirm_vision_indexes", ""),
        "latest_action_blocker_keys": action_summary.get("blocker_keys") if isinstance(action_summary.get("blocker_keys"), list) else [],
        "restore_plan_command": restore_command,
        "restore_apply_runs_logged": len(restore_rows),
    }


def _mcp_args_audit_summary(root: Path) -> dict[str, Any]:
    try:
        audit = audit_bundle_mcp_args(
            root,
            callables=_mcp_status_callables(),
            tool_by_manifest_key=_MCP_TOOL_BY_MANIFEST_KEY,
            arg_aliases=_MCP_ARG_ALIASES,
        )
    except Exception as exc:
        return {
            "schema": "video_knowledge_pipeline.mcp_args_audit_summary.v1",
            "status": "error",
            "error": str(exc),
            "total": 0,
            "ok_count": 0,
            "blocked_count": 1,
            "blocked_keys": [],
        }
    rows = audit.get("rows") if isinstance(audit.get("rows"), list) else []
    blocked = [row for row in rows if isinstance(row, dict) and not row.get("ok")]
    return {
        "schema": "video_knowledge_pipeline.mcp_args_audit_summary.v1",
        "status": str(audit.get("status") or ""),
        "total": int(audit.get("total") or 0),
        "ok_count": int(audit.get("ok_count") or 0),
        "blocked_count": int(audit.get("blocked_count") or 0),
        "blocked_keys": [str(row.get("key") or "") for row in blocked[:20]],
    }


def _controlled_execution_checklist(controlled: dict[str, Any], *, mcp_args_audit: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if controlled.get("status") == "not_applicable":
        return [{"key": "visual_execution", "status": "not_applicable", "ok": True,
                 "detail": "Verified audio-only source without supplemental visuals; no vision execution required."}]
    blockers = controlled.get("blockers") if isinstance(controlled.get("blockers"), list) else []
    audit = mcp_args_audit if isinstance(mcp_args_audit, dict) else {}
    return [
        {
            "key": "mcp_args_usable",
            "ok": audit.get("status") in {"ok", ""},
            "detail": "status={status}; ok={ok}/{total}; blocked={blocked}; keys={keys}".format(
                status=audit.get("status", ""),
                ok=audit.get("ok_count", 0),
                total=audit.get("total", 0),
                blocked=audit.get("blocked_count", 0),
                keys=",".join(audit.get("blocked_keys") or []),
            ),
        },
        {
            "key": "preflight_present",
            "ok": controlled.get("preflight_status") == "present",
            "detail": str(controlled.get("preflight_path") or ""),
        },
        {
            "key": "preflight_not_blocked",
            "ok": "preflight_blocked" not in blockers and "provider_health_failed" not in blockers,
            "detail": ", ".join(str(item) for item in blockers),
        },
        {
            "key": "provider_health_ok",
            "ok": "provider_health_failed" not in blockers,
            "detail": "status={status}; safe={safe}; error_class={error_class}".format(
                status=controlled.get("provider_health_status") or "not_checked",
                safe=controlled.get("provider_health_safe_to_execute"),
                error_class=controlled.get("provider_health_error_class") or "",
            ),
        },
        {
            "key": "batch_confirmed_or_not_pending",
            "ok": "confirmation_required" not in blockers,
            "detail": str(
                controlled.get("latest_advance_blocked_reason")
                or controlled.get("latest_execution_control_status")
                or ""
            ),
        },
        {
            "key": "vision_audit_available",
            "ok": controlled.get("audit_status") == "present",
            "detail": str(controlled.get("vision_runs_path") or ""),
        },
        {
            "key": "recoverable_vision_write_available",
            "ok": bool(controlled.get("latest_vision_write_recoverable")),
            "detail": "status={status}; updated={updated}; changed={changed}".format(
                status=controlled.get("latest_vision_run_status") or "",
                updated=controlled.get("latest_vision_run_updated_count") or 0,
                changed=controlled.get("latest_vision_run_timeline_diff_count") or 0,
            ),
        },
        {
            "key": "restore_path_available",
            "ok": bool(controlled.get("restore_plan_command")) or controlled.get("restore_status") == "available",
            "detail": str(controlled.get("restore_plan_command") or controlled.get("restore_plan_path") or ""),
        },
        {
            "key": "restore_apply_available",
            "ok": str(controlled.get("restore_runs_path") or "").endswith("vision-restore-runs.jsonl"),
            "detail": str(controlled.get("restore_runs_path") or ""),
        },
    ]


def _controlled_execution_next_steps(controlled: dict[str, Any], checklist: list[dict[str, Any]]) -> list[str]:
    failed = {str(item.get("key") or "") for item in checklist if not item.get("ok")}
    steps: list[str] = []
    if "mcp_args_usable" in failed:
        steps.append("Run mcp-audit-bundle and fix missing or unsupported MCP args before real execution.")
    if "preflight_present" in failed or "preflight_not_blocked" in failed:
        steps.append("Run vision-execution-preflight and resolve provider/key/candidate blockers.")
    if "provider_health_ok" in failed:
        steps.append("Fix the selected vision provider health check, or switch provider, then rerun vision-execution-preflight with --check-provider.")
    if "vision_provider_not_ready" in controlled.get("blockers", []):
        steps.append("Configure the selected vision provider API key in the environment or explicit provider config, then rerun the same confirmed vision command.")
    if "batch_confirmed_or_not_pending" in failed:
        steps.append("Copy confirm_vision_calls and confirm_vision_indexes from preflight into the same vision execution command.")
    if "vision_audit_available" in failed:
        steps.append("Run a confirmed vision execution or inspect vision-analysis-runs.jsonl.")
    if "recoverable_vision_write_available" in failed:
        steps.append("Run a confirmed vision execution that writes controlled timeline fields before planning restore.")
    if "restore_path_available" in failed:
        run_id = str(controlled.get("latest_vision_run_id") or "<run_id>")
        steps.append(f"Run vision-analysis-restore-plan --run-id {run_id}.")
    if "restore_apply_available" in failed:
        steps.append("Use vision-analysis-apply-restore after reviewing the restore plan.")
    return steps or ["Controlled execution chain is ready."]


def _obsidian_summary(root: Path) -> dict[str, Any]:
    latest = _latest_advance(root)
    artifacts = latest.get("action_artifacts") if isinstance(latest.get("action_artifacts"), dict) else {}
    status = artifacts.get("obsidian_export_status") if isinstance(artifacts.get("obsidian_export_status"), dict) else {}
    status_path = str(artifacts.get("obsidian_export_status_path") or artifacts.get("obsidian_export_status_markdown_path") or "")
    entrypoints = {
        key.removeprefix("obsidian_"): value
        for key, value in artifacts.items()
        if key.startswith("obsidian_") and isinstance(value, str)
    }
    return {
        "status": status.get("status") if status else ("exported" if entrypoints else "not_exported"),
        "export_status": status,
        "status_path": status_path,
        "entrypoints": entrypoints,
    }


def _path_row(key: str, path: Path) -> dict[str, Any]:
    return {"key": key, "path": str(path), "exists": path.exists()}


def _resolve(root: Path, value: Any) -> Path:
    text = str(value or "").strip()
    if not text:
        return root
    path = Path(text)
    return path if path.is_absolute() else root / path


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _mcp_call_command(tool: str, args_path: Path) -> str:
    escaped = str(args_path).replace("'", "''")
    return f".\\scripts\\video-knowledge.ps1 mcp-call {tool} '{escaped}'"




_MCP_TOOL_BY_MANIFEST_KEY = {
    "mcp_refresh_args": "refresh_lecture_review_outputs",
    "mcp_frame_recapture_args": "run_frame_recapture_plan",
    "mcp_ocr_backfill_args": "run_ocr_backfill",
    "mcp_screen_text_recovery_args": "run_screen_text_recovery",
    "mcp_visual_structure_args": "run_visual_structure_plan",
    "mcp_video_frame_router_args": "run_video_frame_router",
    "mcp_multimodal_frame_analysis_args": "run_multimodal_frame_analysis",
    "mcp_temporal_frame_groups_args": "run_temporal_frame_groups",
    "mcp_temporal_visual_analysis_args": "run_temporal_visual_analysis",
    "mcp_asset_repair_args": "repair_bundle_assets",
    "mcp_repair_status_args": "refresh_bundle_repair_status",
    "mcp_readiness_args": "audit_bundle_readiness",
    "mcp_knowledge_coverage_args": "audit_knowledge_coverage",
    "mcp_next_action_args": "bundle_next_action",
    "mcp_source_artifacts_args": "bundle_source_artifacts",
    "mcp_advance_args": "bundle_advance",
    "mcp_advance_log_args": "bundle_advance_log",
    "mcp_advance_queue_args": "bundle_advance_queue",
    "mcp_review_session_args": "prepare_review_session",
    "mcp_apply_review_notes_args": "apply_review_notes",
    "mcp_review_closure_status_args": "review_closure_status",
    "mcp_status_report_args": "bundle_status_report",
    "mcp_acceptance_check_args": "acceptance_check",
    "mcp_vision_provider_smoke_args": "vision_provider_smoke",
    "mcp_vision_provider_matrix_args": "vision_provider_matrix",
    "mcp_controlled_execution_check_args": "controlled_execution_check",
    "mcp_controlled_execution_smoke_args": "controlled_execution_smoke",
    "mcp_run_artifact_registry_args": "run_artifact_registry",
    "mcp_export_knowledge_note_args": "export_knowledge_note",
    "mcp_vision_acceptance_plan_args": "vision_acceptance_plan",
    "mcp_vision_execution_preflight_args": "vision_execution_preflight",
    "mcp_multimodal_frame_analysis_confirmed_args": "run_multimodal_frame_analysis",
    "mcp_temporal_visual_analysis_confirmed_args": "run_temporal_visual_analysis",
}


_MCP_ARG_ALIASES = {
    "refresh_lecture_review_outputs": {"project": "root"},
    "refresh_lecture_review_outputs_tool": {"project": "root"},
}


def _mcp_status_callables() -> dict[str, Any]:
    from .acceptance_check import acceptance_check
    from .controlled_execution_smoke import controlled_execution_smoke
    from .vision_provider_smoke import vision_provider_matrix, vision_provider_smoke

    mapping: dict[str, Any] = {
        "acceptance_check": acceptance_check,
        "vision_provider_smoke": vision_provider_smoke,
        "vision_provider_matrix": vision_provider_matrix,
        "refresh_lecture_review_outputs": refresh_lecture_review_outputs,
        "run_frame_recapture_plan": run_frame_recapture_plan,
        "run_visual_structure_plan": run_visual_structure_plan,
        "run_ocr_backfill": run_ocr_backfill,
        "run_video_frame_router": run_video_frame_router,
        "run_multimodal_frame_analysis": run_multimodal_frame_analysis,
        "run_temporal_frame_groups": run_temporal_frame_groups,
        "run_temporal_visual_analysis": run_temporal_visual_analysis,
        "repair_bundle_assets": repair_bundle_assets,
        "refresh_bundle_repair_status": refresh_bundle_repair_status,
        "audit_bundle_readiness": audit_bundle_readiness,
        "audit_knowledge_coverage": audit_knowledge_coverage,
        "bundle_next_action": bundle_next_action,
        "bundle_source_artifacts": bundle_source_artifacts,
        "bundle_advance": bundle_advance,
        "bundle_advance_log": bundle_advance_log,
        "bundle_advance_queue": bundle_advance_queue,
        "apply_review_notes": apply_review_notes_to_bundle,
        "prepare_review_session": prepare_review_session,
        "review_closure_status": review_closure_status,
        "run_screen_text_recovery": run_screen_text_recovery,
        "bundle_status_report": bundle_status_report,
        "controlled_execution_check": controlled_execution_check,
        "controlled_execution_smoke": controlled_execution_smoke,
        "run_artifact_registry": build_run_artifact_registry,
        "export_knowledge_note": export_knowledge_note,
        "vision_acceptance_plan": vision_acceptance_plan,
        "vision_execution_preflight": vision_execution_preflight,
    }
    for name, func in list(mapping.items()):
        mapping[f"{name}_tool"] = func
    return mapping
