from __future__ import annotations

from pathlib import Path
from typing import Any

from .tool_research import recommended_trial_order


VISUAL_EXTRACTORS = {"vidclaude", "peepshow", "vidwise"}


def visual_tool_matrix() -> dict[str, dict[str, Any]]:
    """Return current local visual extractor rows keyed by tool name."""
    return {
        str(row.get("name") or "").lower(): row
        for row in recommended_trial_order()
        if str(row.get("name") or "").lower() in VISUAL_EXTRACTORS
    }


def resolve_visual_extractor_command(name: str) -> dict[str, Any]:
    """Resolve the preferred command prefix for a visual extractor."""
    tool_name = name.lower().strip()
    row = visual_tool_matrix().get(tool_name) or {}
    paths = [str(path) for path in row.get("installed_paths") or [] if path]
    if tool_name == "peepshow":
        cli = _first_path(paths, suffix="cli.js")
        if cli:
            return _resolved(tool_name, row, ["node", cli], "npx_cache_cli_js", cli)
        return _resolved(tool_name, row, ["npx", "peepshow"], "npx_fallback", "")
    if tool_name == "vidwise":
        command = _first_path(paths) or "vidwise"
        return _resolved(tool_name, row, [command], "installed_command" if paths else "path_fallback", command if paths else "")
    if tool_name == "vidclaude":
        command = _first_path(paths) or ""
        return _resolved(tool_name, row, [command] if command else [], "installed_command" if command else "project_script", command)
    return _resolved(tool_name, row, [], "unsupported", "")


def _resolved(name: str, row: dict[str, Any], command_prefix: list[str], source: str, path: str) -> dict[str, Any]:
    return {
        "name": name,
        "installed": bool(row.get("installed")),
        "installed_paths": list(row.get("installed_paths") or []),
        "command_prefix": command_prefix,
        "command_source": source,
        "command_path": path,
        "install_hint": str(row.get("install_hint") or ""),
        "reuse_role": str(row.get("reuse_role") or ""),
    }


def _first_path(paths: list[str], *, suffix: str | None = None) -> str:
    for value in paths:
        if suffix and not value.lower().endswith(suffix.lower()):
            continue
        path = Path(value)
        if path.exists():
            return str(path)
    return ""
