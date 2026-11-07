from __future__ import annotations

from pathlib import Path
from typing import Any

from .asr_runner import detect_asr_runners
from .bilinote_patch import DEFAULT_BILINOTE_ROOT, bilinote_patch_status
from .captiocr_resolver import resolve_captiocr_root
from .lecture_pipeline import lecture_import_status, status_lecture_pipeline_plan
from .path_defaults import tool_source_review_root
from .storage import ensure_project_dirs, read_json, read_jsonl
from .tool_research import recommended_trial_order


def lecture_project_health(
    root: str | Path,
    *,
    plan_json: str | Path | None = None,
    transcript: str | Path | None = None,
    webui_output_dir: str | Path | None = None,
    bilinote_root: str | Path | None = None,
    bilinote_patch_root: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect the current lecture extraction project state without running heavy tools."""
    root_path = Path(root)
    paths = ensure_project_dirs(root_path)
    project_path = paths["project"]
    package_path = paths["lecture_packages"] / "lecture-package.json"
    quality_report_path = paths["notes"] / "lecture-quality-report.md"
    knowledge_note_path = paths["notes"] / "lecture-knowledge-package.md"
    review_html_path = paths["notes"] / "lecture-review.html"
    action_log_path = paths["lecture_packages"] / "lecture-action-log.jsonl"
    action_log_markdown_path = paths["notes"] / "lecture-action-log.md"
    asr_log_path = paths["lecture_packages"] / "asr-command-runs.jsonl"
    asr_log_markdown_path = paths["notes"] / "asr-command-runs.md"
    extractor_log_path = paths["lecture_packages"] / "extractor-command-runs.jsonl"
    extractor_log_markdown_path = paths["notes"] / "extractor-command-runs.md"
    plan = _read_json_object(Path(plan_json)) if plan_json else None
    planned_outputs = plan.get("planned_outputs") if isinstance(plan, dict) and isinstance(plan.get("planned_outputs"), dict) else {}
    planned_bundle = planned_outputs.get("webui_output_dir") if isinstance(planned_outputs, dict) else None
    bundle_dir = Path(webui_output_dir or planned_bundle) if (webui_output_dir or planned_bundle) else paths["lecture_packages"] / "webui-bundle"
    manifest_path = bundle_dir / "manifest.json"
    review_notes_path = bundle_dir / "review-notes.json"

    import_status = lecture_import_status(root_path)
    package = _read_json_object(package_path)
    plan_status = status_lecture_pipeline_plan(plan_json, transcript=transcript) if plan_json else None
    asr_status = detect_asr_runners()
    files = {
        "project": _file_status(project_path),
        "import_registry": _file_status(Path(import_status["registry_path"])),
        "import_report": _file_status(Path(import_status["markdown_path"])),
        "package": _file_status(package_path),
        "knowledge_note": _file_status(knowledge_note_path),
        "review_html": _file_status(review_html_path),
        "quality_report": _file_status(quality_report_path),
        "action_log": _file_status(action_log_path),
        "action_log_markdown": _file_status(action_log_markdown_path),
        "asr_log": _file_status(asr_log_path),
        "asr_log_markdown": _file_status(asr_log_markdown_path),
        "extractor_log": _file_status(extractor_log_path),
        "extractor_log_markdown": _file_status(extractor_log_markdown_path),
        "webui_manifest": _file_status(manifest_path),
        "webui_timeline": _file_status(bundle_dir / "timeline.json"),
        "webui_review_notes": _file_status(review_notes_path),
    }
    package_status = _package_status(package)
    action_log = _action_log_status(action_log_path)
    asr_log = _run_log_status(asr_log_path)
    extractor_log = _run_log_status(extractor_log_path)
    recommended_tools = _recommended_tool_status(plan, bilinote_root=bilinote_root, bilinote_patch_root=bilinote_patch_root)
    level = _health_level(files, import_status, package_status)
    next_actions = _next_actions(level, plan_status, import_status, package_status, files, recommended_tools)
    result = {
        "project": str(root_path),
        "level": level,
        "files": files,
        "asr": {
            "available_tools": [tool["name"] for tool in asr_status.get("tools", []) if tool.get("available")],
            "recommended_order": asr_status.get("recommended_order", []),
        },
        "imports": import_status.get("summary", {}),
        "package": package_status,
        "action_log": action_log,
        "asr_log": asr_log,
        "extractor_log": extractor_log,
        "recommended_tools": recommended_tools,
        "plan_status": plan_status,
        "next_actions": next_actions,
        "markdown_path": str(paths["notes"] / "lecture-project-health.md"),
    }
    Path(result["markdown_path"]).write_text(_render_health_markdown(result), encoding="utf-8")
    return result


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = read_json(path)
    return data if isinstance(data, dict) else None


def _file_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def _package_status(package: dict[str, Any] | None) -> dict[str, Any]:
    if not package:
        return {"exists": False, "timeline_count": 0, "coverage": {}, "quality_summary": {}}
    audit = package.get("quality_audit") if isinstance(package.get("quality_audit"), dict) else {}
    return {
        "exists": True,
        "title": package.get("title", ""),
        "source_count": package.get("source_count", 0),
        "timeline_count": package.get("timeline_count", 0),
        "coverage": package.get("coverage", {}) if isinstance(package.get("coverage"), dict) else {},
        "quality_summary": audit.get("summary", {}) if isinstance(audit.get("summary"), dict) else {},
    }


def _action_log_status(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    last_action = rows[-1] if rows else {}
    return {
        "count": len(rows),
        "last_action": last_action,
    }


def _run_log_status(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    return {
        "count": len(rows),
        "last": rows[-1] if rows else {},
    }


def _recommended_tool_status(
    plan: dict[str, Any] | None = None,
    *,
    bilinote_root: str | Path | None = None,
    bilinote_patch_root: str | Path | None = None,
) -> dict[str, Any]:
    matrix = {str(row.get("name", "")).lower(): row for row in recommended_trial_order()}
    commands = plan.get("commands") if isinstance(plan, dict) and isinstance(plan.get("commands"), dict) else {}
    tools = [
        _bilinote_tool(bilinote_root=bilinote_root, patch_root=bilinote_patch_root),
        _matrix_tool(matrix, "vidclaude", "visual_timeline_extractor", command_hint=str(commands.get("vidclaude") or "")),
        _matrix_tool(matrix, "peepshow", "fast_frame_ocr_report", command_hint=str(commands.get("peepshow") or "")),
        _matrix_tool(matrix, "vidwise", "lightweight_fallback_extractor", command_hint=str(commands.get("vidwise") or "")),
        _local_tool(
            "content-core",
            "cli_mcp_architecture_reference",
            str(tool_source_review_root() / "content-core"),
            command_hint="architecture reference only; no lecture extraction command",
        ),
        _captiocr_tool(),
    ]
    return {
        "available_count": sum(1 for tool in tools if tool.get("available")),
        "tools": tools,
    }


def _captiocr_tool() -> dict[str, Any]:
    resolved = resolve_captiocr_root()
    root = str(resolved.get("root") or tool_source_review_root() / "captiocr")
    return {
        "name": "CaptiOCR",
        "role": "ocr_review_widget_reference",
        "available": bool(resolved.get("available")),
        "paths": [root, *list(resolved.get("evidence") or [])] if resolved.get("available") else [],
        "source": "",
        "install_hint": str(resolved.get("configure_hint") or ""),
        "command_hint": str(resolved.get("command_hint") or f"python {Path(root) / 'CaptiOCR.py'}"),
        "notes": "Tk screenshot OCR/manual correction helper for OCR backfill.",
        "checked_paths": list(resolved.get("checked") or []),
    }


def _bilinote_tool(*, bilinote_root: str | Path | None = None, patch_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(bilinote_root) if bilinote_root else DEFAULT_BILINOTE_ROOT
    command_hint = ".\\scripts\\start-bilinote-lecture.ps1 -CheckOnly"
    if bilinote_root:
        command_hint = f".\\scripts\\start-bilinote-lecture.ps1 -BiliNoteRoot \"{root}\" -CheckOnly"
    try:
        patch = bilinote_patch_status(root, patch_root)
        available = bool(patch.get("ok"))
        notes = (
            f"packaged patch installed={patch.get('installed', 0)}/{patch.get('total', 0)}, "
            f"missing={patch.get('missing', 0)}, drift={patch.get('drift', 0)}"
        )
        paths = [str(root.resolve()), str(Path(str(patch.get("patch_root", ""))).resolve())]
    except (FileNotFoundError, ValueError) as exc:
        patch = {
            "ok": False,
            "error": str(exc),
            "bilinote_root": str(root),
            "install_command": ".\\scripts\\install-bilinote-lecture-patch.ps1 -Install -Backup",
            "check_command": ".\\scripts\\install-bilinote-lecture-patch.ps1 -FailOnDrift",
        }
        available = False
        notes = str(exc)
        paths = [str(root)] if root.exists() else []

    return {
        "name": "BiliNote",
        "role": "product_shell_ui",
        "available": available,
        "paths": paths,
        "source": "",
        "install_hint": f"expected local path: {root}",
        "command_hint": command_hint,
        "notes": notes,
        "patch_status": patch,
    }


def _matrix_tool(matrix: dict[str, dict[str, Any]], name: str, role: str, *, command_hint: str = "") -> dict[str, Any]:
    row = matrix.get(name.lower()) or {}
    installed_paths = [str(path) for path in row.get("installed_paths") or []]
    return {
        "name": name,
        "role": role,
        "available": bool(installed_paths),
        "paths": installed_paths,
        "source": row.get("url", ""),
        "install_hint": row.get("install_hint", ""),
        "command_hint": command_hint,
        "notes": row.get("notes", ""),
    }


def _local_tool(name: str, role: str, root: str, *, required: list[str] | None = None, command_hint: str = "") -> dict[str, Any]:
    root_path = Path(root)
    required_paths = [root_path / item for item in (required or [])]
    available = root_path.exists() and all(path.exists() for path in required_paths)
    paths = [str(root_path)] if root_path.exists() else []
    paths.extend(str(path) for path in required_paths if path.exists())
    return {
        "name": name,
        "role": role,
        "available": available,
        "paths": paths,
        "source": "",
        "install_hint": f"expected local path: {root}",
        "command_hint": command_hint,
        "notes": "local reviewed source tree" if root_path.exists() else "not found at expected local path",
    }


def _health_level(files: dict[str, dict[str, Any]], import_status: dict[str, Any], package_status: dict[str, Any]) -> str:
    if files["webui_review_notes"]["exists"]:
        return "review_ready_to_refresh"
    if files["webui_manifest"]["exists"]:
        return "webui_ready"
    if package_status.get("exists"):
        return "packaged"
    if int((import_status.get("summary") or {}).get("import_count") or 0) > 0:
        return "imported"
    if files["project"]["exists"]:
        return "initialized"
    return "empty"


def _next_actions(
    level: str,
    plan_status: dict[str, Any] | None,
    import_status: dict[str, Any],
    package_status: dict[str, Any],
    files: dict[str, dict[str, Any]],
    recommended_tools: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    webui_bilinote_setup = _bilinote_setup_action(level, recommended_tools)
    if webui_bilinote_setup:
        actions.append(webui_bilinote_setup)
    if plan_status and (plan_status.get("recommended_pipeline_command") or ""):
        actions.append(
            {
                "action": "run_ready_pipeline",
                "command": str(plan_status.get("recommended_pipeline_command") or ""),
                "reason": "planned extractor output is ready",
            }
        )
    if level == "empty":
        actions.append({"action": "init_or_prepare", "command": "video-knowledge prepare-lecture-workspace ...", "reason": "project is not initialized"})
    elif level == "initialized" and int((import_status.get("summary") or {}).get("import_count") or 0) == 0:
        command = "run vidclaude/peepshow/ASR, then run-ready-lecture-pipeline"
        if plan_status and isinstance(plan_status.get("plan_path"), str):
            command = f"video-knowledge run-extractor-plan {plan_status.get('plan_path')} peepshow --execute"
        actions.append({"action": "run_extractors", "command": command, "reason": "no extractor output has been imported"})
    elif level == "imported":
        actions.append({"action": "build_package", "command": "video-knowledge build-lecture-package <project>", "reason": "imported segments exist but lecture-package.json is missing"})
    elif level == "packaged":
        actions.append({"action": "export_webui", "command": "video-knowledge export-webui-bundle <project>", "reason": "package exists but WebUI bundle is missing"})
    elif level == "webui_ready":
        if not webui_bilinote_setup:
            actions.append({"action": "human_review", "command": "open BiliNote and use 课程复核", "reason": "WebUI bundle is ready for human review"})
    elif level == "review_ready_to_refresh":
        actions.append({"action": "refresh_review", "command": "video-knowledge refresh-lecture-review <project> <bundle>/review-notes.json", "reason": "review-notes.json exists"})

    quality = package_status.get("quality_summary") or {}
    if int(quality.get("items_with_issues") or 0) > 0 and files["quality_report"]["exists"]:
        actions.append({"action": "audit_gaps", "command": "open notes/lecture-quality-report.md", "reason": "quality audit still has high-priority review items"})
    return actions


def _bilinote_setup_action(level: str, recommended_tools: dict[str, Any] | None) -> dict[str, str] | None:
    if level != "webui_ready":
        return None
    bilinote = _recommended_tool_by_name(recommended_tools, "BiliNote")
    if not bilinote or bilinote.get("available"):
        return None
    patch = bilinote.get("patch_status") if isinstance(bilinote.get("patch_status"), dict) else {}
    command = str(patch.get("install_command") or ".\\scripts\\install-bilinote-lecture-patch.ps1 -Install -Backup")
    return {
        "action": "setup_bilinote_patch",
        "command": command,
        "reason": "WebUI bundle is ready but BiliNote lecture patch is missing or drifted",
    }


def _recommended_tool_by_name(recommended_tools: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if not isinstance(recommended_tools, dict):
        return None
    wanted = name.lower()
    for tool in recommended_tools.get("tools") or []:
        if isinstance(tool, dict) and str(tool.get("name") or "").lower() == wanted:
            return tool
    return None


def _render_health_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Lecture Project Health",
        "",
        f"- Project: `{result['project']}`",
        f"- Level: `{result['level']}`",
        "",
        "## Next Actions",
        "",
    ]
    actions = result.get("next_actions") or []
    if actions:
        for action in actions:
            lines.append(f"- **{action.get('action', '')}**: `{action.get('command', '')}`")
            lines.append(f"  - Reason: {action.get('reason', '')}")
    else:
        lines.append("- No immediate action detected.")
    lines.extend(["", "## Files", "", "| File | Exists | Path |", "|---|---|---|"])
    for name, status in (result.get("files") or {}).items():
        exists = "yes" if status.get("exists") else "no"
        lines.append(f"| {name} | {exists} | `{status.get('path', '')}` |")
    lines.extend(["", "## Imports", ""])
    imports = result.get("imports") or {}
    for key in ["import_count", "total_segments", "forced_reimport_count", "kinds"]:
        lines.append(f"- {key}: `{imports.get(key, '')}`")
    lines.extend(["", "## Package", ""])
    package = result.get("package") or {}
    lines.append(f"- timeline_count: `{package.get('timeline_count', 0)}`")
    lines.append(f"- source_count: `{package.get('source_count', 0)}`")
    quality = package.get("quality_summary") or {}
    lines.append(f"- quality items_with_issues: `{quality.get('items_with_issues', 0)}`")
    lines.append(f"- quality max_score: `{quality.get('max_score', 0)}`")
    lines.extend(["", "## Action Log", ""])
    action_log = result.get("action_log") if isinstance(result.get("action_log"), dict) else {}
    last_action = action_log.get("last_action") if isinstance(action_log.get("last_action"), dict) else {}
    lines.append(f"- count: `{action_log.get('count', 0)}`")
    if last_action:
        lines.append(f"- last_action: `{last_action.get('action', '')}`")
        lines.append(f"- last_mode: `{last_action.get('mode', '')}`")
        lines.append(f"- last_executed: `{last_action.get('executed', False)}`")
        lines.append(f"- last_level: `{last_action.get('before_level', '')} -> {last_action.get('after_level', '')}`")
    lines.extend(["", "## Runner Logs", ""])
    asr_log = result.get("asr_log") if isinstance(result.get("asr_log"), dict) else {}
    extractor_log = result.get("extractor_log") if isinstance(result.get("extractor_log"), dict) else {}
    lines.append(f"- ASR runs: `{asr_log.get('count', 0)}`")
    if isinstance(asr_log.get("last"), dict) and asr_log.get("last"):
        lines.append(f"- Last ASR: `{asr_log['last'].get('preset', '')}` / `{asr_log['last'].get('status', '')}`")
    lines.append(f"- Visual extractor runs: `{extractor_log.get('count', 0)}`")
    if isinstance(extractor_log.get("last"), dict) and extractor_log.get("last"):
        lines.append(f"- Last extractor: `{extractor_log['last'].get('extractor', '')}` / `{extractor_log['last'].get('status', '')}`")
    lines.extend(["", "## Recommended Tools", ""])
    recommended = result.get("recommended_tools") if isinstance(result.get("recommended_tools"), dict) else {}
    lines.append(f"- available_count: `{recommended.get('available_count', 0)}`")
    lines.extend(["", "| Tool | Role | Available | Command | Notes | Paths |", "|---|---|---|---|---|---|"])
    for tool in recommended.get("tools") or []:
        available = "yes" if tool.get("available") else "no"
        paths = "<br>".join(f"`{path}`" for path in tool.get("paths") or [])
        command = str(tool.get("command_hint") or "")
        command_cell = f"`{command}`" if command else ""
        notes = str(tool.get("notes") or "").replace("|", "\\|")
        lines.append(f"| {tool.get('name', '')} | {tool.get('role', '')} | {available} | {command_cell} | {notes} | {paths} |")
    lines.extend(["", "## ASR", ""])
    asr = result.get("asr") or {}
    lines.append(f"- available_tools: `{asr.get('available_tools', [])}`")
    lines.append(f"- recommended_order: `{asr.get('recommended_order', [])}`")
    return "\n".join(lines).rstrip() + "\n"

