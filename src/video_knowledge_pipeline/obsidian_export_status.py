from __future__ import annotations

from pathlib import Path
from typing import Any

from .powershell import quote_powershell_argument as _ps_quote
from .models import now_iso
from .storage import read_json, write_json


STATUS_SCHEMA = "lecture_obsidian_export_status.v1"


def obsidian_export_status(export: str | Path, *, write: bool = False) -> dict[str, Any]:
    """Validate a lecture Obsidian export manifest and its agent entrypoints."""
    root = Path(export).expanduser()
    manifest_path = root if root.name == "obsidian-export.json" else root / "obsidian-export.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"obsidian export manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("obsidian-export.json must be a JSON object")
    folder = Path(str(manifest.get("folder") or manifest_path.parent)).expanduser()
    pages = manifest.get("pages") if isinstance(manifest.get("pages"), list) else []
    entrypoints = manifest.get("agent_entrypoints") if isinstance(manifest.get("agent_entrypoints"), dict) else {}
    page_rows = [_path_status(page.get("name"), page.get("path")) for page in pages if isinstance(page, dict)]
    entrypoint_rows = [_path_status(key, value) for key, value in entrypoints.items()]
    missing_pages = [row for row in page_rows if not row["exists"]]
    missing_entrypoints = [row for row in entrypoint_rows if not row["exists"]]
    core_entrypoints = ["full_speech_text", "full_screen_text", "timeline", "evidence_map", "source_artifacts_json"]
    missing_core = [key for key in core_entrypoints if key not in entrypoints or not _exists(entrypoints.get(key))]
    asset_validation = _asset_validation(entrypoints.get("asset_manifest"), folder=folder)
    source_validation = _source_artifact_validation(entrypoints.get("source_artifacts_json"))
    status = _status(missing_pages, missing_entrypoints, missing_core, asset_validation, source_validation)
    result = {
        "schema": STATUS_SCHEMA,
        "checked_at": now_iso(),
        "status": status,
        "manifest_path": str(manifest_path),
        "folder": str(folder),
        "page_count": len(page_rows),
        "missing_page_count": len(missing_pages),
        "entrypoint_count": len(entrypoint_rows),
        "missing_entrypoint_count": len(missing_entrypoints),
        "missing_core_entrypoints": missing_core,
        "pages": page_rows,
        "entrypoints": entrypoint_rows,
        "asset_summary": manifest.get("asset_summary") if isinstance(manifest.get("asset_summary"), dict) else {},
        "source_artifact_summary": manifest.get("source_artifact_summary") if isinstance(manifest.get("source_artifact_summary"), dict) else {},
        "asset_validation": asset_validation,
        "source_artifact_validation": source_validation,
        "next_action": _next_action(missing_pages, missing_entrypoints, missing_core, asset_validation, source_validation, entrypoints),
    }
    if write:
        report_path = folder / "obsidian-export-status.md"
        json_path = folder / "obsidian-export-status.json"
        write_json(json_path, result)
        report_path.write_text(render_obsidian_export_status_markdown(result), encoding="utf-8")
        result["status_json_path"] = str(json_path)
        result["status_markdown_path"] = str(report_path)
    return result


def render_obsidian_export_status_markdown(result: dict[str, Any]) -> str:
    lines = [
        "---",
        "type: lecture-obsidian-export-status",
        f'created: "{result.get("checked_at", now_iso())}"',
        "---",
        "",
        "# Obsidian 导出状态",
        "",
        f"- 状态：`{result.get('status', 'unknown')}`",
        f"- Manifest：`{result.get('manifest_path', '')}`",
        f"- 导出目录：`{result.get('folder', '')}`",
        f"- 页面：{result.get('page_count', 0)}，缺失 {result.get('missing_page_count', 0)}",
        f"- Agent 入口：{result.get('entrypoint_count', 0)}，缺失 {result.get('missing_entrypoint_count', 0)}",
        f"- 缺核心入口：{', '.join(result.get('missing_core_entrypoints') or []) or 'none'}",
        f"- 关键帧资产缺失：{(result.get('asset_validation') or {}).get('missing_visual_asset_count', 0)}",
        f"- 原始抽取物缺失：{(result.get('source_artifact_validation') or {}).get('missing_source_artifact_count', 0)}",
        f"- 下一步：`{(result.get('next_action') or {}).get('key', '')}` / {(result.get('next_action') or {}).get('label', '')}",
        "",
    ]
    command = _next_action_command(result.get("next_action") if isinstance(result.get("next_action"), dict) else {})
    if command:
        lines.extend(["## 下一步命令", "", "```powershell", command, "```", ""])
    references = _next_action_references(result.get("next_action") if isinstance(result.get("next_action"), dict) else {})
    if references:
        lines.extend(["## 下一步参考文件", ""])
        for label, path in references:
            lines.append(f"- {label}: `{path}`")
        lines.append("")
    lines.extend(["## Agent 入口", "", "| Key | Exists | Path |", "|---|---:|---|"])
    for row in result.get("entrypoints") or []:
        if isinstance(row, dict):
            lines.append(f"| {row.get('key', '')} | {row.get('exists', False)} | `{row.get('path', '')}` |")
    lines.extend(["", "## 页面", "", "| Page | Exists | Path |", "|---|---:|---|"])
    for row in result.get("pages") or []:
        if isinstance(row, dict):
            lines.append(f"| {row.get('key', '')} | {row.get('exists', False)} | `{row.get('path', '')}` |")
    assets = result.get("asset_validation") if isinstance(result.get("asset_validation"), dict) else {}
    if assets.get("missing_visual_assets"):
        lines.extend(["", "## 缺失关键帧资产", ""])
        for row in assets.get("missing_visual_assets") or []:
            if isinstance(row, dict):
                lines.append(f"- `{row.get('path', '')}`")
    source = result.get("source_artifact_validation") if isinstance(result.get("source_artifact_validation"), dict) else {}
    if source.get("missing_source_artifacts"):
        lines.extend(["", "## 缺失原始抽取物", ""])
        for row in source.get("missing_source_artifacts") or []:
            if isinstance(row, dict):
                lines.append(f"- {row.get('label') or row.get('key')}: `{row.get('path', '')}`")
    return "\n".join(lines).rstrip() + "\n"


def _next_action_command(next_action: dict[str, Any]) -> str:
    tool = str(next_action.get("mcp_tool") or "").strip()
    args_path = str(next_action.get("mcp_args_path") or "").strip()
    if not tool or not args_path:
        return ""
    return "video-knowledge mcp-call " + _ps_quote(tool) + " " + _ps_quote(args_path)


def _next_action_references(next_action: dict[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key, label in (
        ("reference_markdown_path", "Markdown"),
        ("reference_json_path", "JSON"),
        ("reference_path", "参考文件"),
    ):
        value = str(next_action.get(key) or "").strip()
        if value and value not in {path for _, path in rows}:
            rows.append((label, value))
    return rows



def _path_status(key: Any, path: Any) -> dict[str, Any]:
    text = str(path or "").strip()
    return {"key": str(key or ""), "path": text, "exists": _exists(text)}


def _exists(path: Any) -> bool:
    text = str(path or "").strip()
    return bool(text) and Path(text).expanduser().exists()


def _asset_validation(asset_manifest_path: Any, *, folder: Path) -> dict[str, Any]:
    path = Path(str(asset_manifest_path or "")).expanduser()
    if not str(asset_manifest_path or "").strip() or not path.exists():
        return {
            "asset_manifest_path": str(asset_manifest_path or ""),
            "status": "blocked",
            "missing_visual_asset_count": 1,
            "missing_visual_assets": [{"path": str(asset_manifest_path or ""), "reason": "asset_manifest_missing"}],
        }
    data = read_json(path)
    if not isinstance(data, dict):
        return {
            "asset_manifest_path": str(path),
            "status": "blocked",
            "missing_visual_asset_count": 1,
            "missing_visual_assets": [{"path": str(path), "reason": "asset_manifest_invalid"}],
        }
    missing: list[dict[str, Any]] = []
    for original in data.get("missing") or []:
        missing.append({"path": str(original), "reason": "not_copied"})
    for row in data.get("copied") or []:
        if not isinstance(row, dict):
            continue
        target = str(row.get("target") or "").strip()
        relative = str(row.get("relative") or "").strip()
        candidate = Path(target).expanduser() if target else folder / relative
        if not candidate.exists():
            missing.append({"path": str(candidate), "reason": "copied_asset_missing"})
    return {
        "asset_manifest_path": str(path),
        "status": "blocked" if missing else "ok",
        "copied_count": int(data.get("copied_count") or 0),
        "declared_missing_count": int(data.get("missing_count") or 0),
        "missing_visual_asset_count": len(missing),
        "missing_visual_assets": missing[:20],
    }


def _source_artifact_validation(source_artifacts_path: Any) -> dict[str, Any]:
    path = Path(str(source_artifacts_path or "")).expanduser()
    if not str(source_artifacts_path or "").strip() or not path.exists():
        return {
            "source_artifacts_path": str(source_artifacts_path or ""),
            "status": "weak",
            "missing_source_artifact_count": 1,
            "missing_source_artifacts": [{"path": str(source_artifacts_path or ""), "reason": "source_artifacts_index_missing"}],
        }
    data = read_json(path)
    if not isinstance(data, dict):
        return {
            "source_artifacts_path": str(path),
            "status": "weak",
            "missing_source_artifact_count": 1,
            "missing_source_artifacts": [{"path": str(path), "reason": "source_artifacts_index_invalid"}],
        }
    missing = [row for row in data.get("missing") or [] if isinstance(row, dict)]
    return {
        "source_artifacts_path": str(path),
        "status": "weak" if missing else "ok",
        "available_count": int(data.get("available_count") or 0),
        "declared_missing_count": int(data.get("missing_count") or 0),
        "missing_source_artifact_count": len(missing),
        "missing_source_artifacts": missing[:20],
    }


def _status(
    missing_pages: list[dict[str, Any]],
    missing_entrypoints: list[dict[str, Any]],
    missing_core: list[str],
    asset_validation: dict[str, Any],
    source_validation: dict[str, Any],
) -> str:
    if missing_pages or missing_entrypoints or missing_core or asset_validation.get("status") == "blocked":
        return "blocked"
    if source_validation.get("status") == "weak":
        return "weak"
    return "ok"


def _next_action(
    missing_pages: list[dict[str, Any]],
    missing_entrypoints: list[dict[str, Any]],
    missing_core: list[str],
    asset_validation: dict[str, Any],
    source_validation: dict[str, Any],
    entrypoints: dict[str, Any],
) -> dict[str, str]:
    export_args = str(entrypoints.get("mcp_export_lecture_obsidian_args") or "")
    if missing_core:
        return {
            "key": "rerun_obsidian_export",
            "label": "重新导出 Obsidian",
            "mcp_tool": "export_lecture_obsidian",
            "mcp_args_path": export_args,
            "hint": "核心底稿或证据入口缺失，重新运行 export_lecture_obsidian 或 refresh_lecture_review_outputs。",
        }
    if asset_validation.get("status") == "blocked":
        return {
            "key": "repair_visual_assets",
            "label": "修复关键帧资产",
            "mcp_tool": "export_lecture_obsidian",
            "mcp_args_path": export_args,
            "hint": "关键帧资产缺失会破坏图表、板书和必须保留图片的证据链；重新导出或修复 assets/asset-manifest.json 中的缺口。",
        }
    if missing_pages or missing_entrypoints:
        return {
            "key": "repair_export_files",
            "label": "修复导出文件",
            "mcp_tool": "export_lecture_obsidian",
            "mcp_args_path": export_args,
            "hint": "部分页面或入口缺失，优先检查导出目录是否被移动或部分删除。",
        }
    if source_validation.get("status") == "weak":
        return {
            "key": "inspect_source_artifacts",
            "label": "检查原始抽取物",
            "mcp_tool": "",
            "mcp_args_path": "",
            "reference_markdown_path": str(entrypoints.get("source_artifacts_markdown") or ""),
            "reference_json_path": str(entrypoints.get("source_artifacts_json") or ""),
            "hint": "原始 vidclaude/peepshow/vidwise 产物有缺失记录；可阅读 source-artifacts.md/json 决定是否需要重新抽取。",
        }
    return {"key": "ready", "label": "导出可用", "mcp_tool": "", "mcp_args_path": "", "hint": "可以交给人类阅读或 agent 检索复核。"}

