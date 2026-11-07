from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any


def audit_bundle_mcp_args(
    bundle_dir: str | Path,
    *,
    callables: dict[str, Any],
    tool_by_manifest_key: dict[str, str],
    arg_aliases: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"bundle missing manifest.json: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest.json must contain an object: {manifest_path}")
    aliases = arg_aliases or {}
    rows = []
    for key in sorted(item for item in manifest if item.startswith("mcp_") and item.endswith("_args")):
        tool = _mcp_tool_for_manifest_key(key, tool_by_manifest_key=tool_by_manifest_key)
        raw_path = str(manifest.get(key) or "").strip()
        args_path = _resolve_bundle_mcp_path(root, raw_path)
        row = _audit_mcp_args_row(key=key, tool=tool, args_path=args_path, callables=callables, arg_aliases=aliases)
        rows.append(row)
    blockers = [row for row in rows if not row.get("ok")]
    return {
        "schema": "video_knowledge_pipeline.mcp_args_audit.v1",
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "status": "ok" if not blockers else "blocked",
        "total": len(rows),
        "ok_count": sum(1 for row in rows if row.get("ok")),
        "blocked_count": len(blockers),
        "rows": rows,
    }


def read_mcp_args(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"MCP args JSON must contain an object: {path}")
    return data


def filter_mcp_payload(func: Any, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    signature = inspect.signature(func)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(payload), []
    accepted = set(signature.parameters)
    filtered = {key: value for key, value in payload.items() if key in accepted}
    ignored = [key for key in payload if key not in accepted]
    return filtered, ignored


def normalise_mcp_payload(tool: str, payload: dict[str, Any], arg_aliases: dict[str, dict[str, str]]) -> dict[str, Any]:
    normalised = dict(payload)
    for old_key, new_key in arg_aliases.get(tool, {}).items():
        if old_key in normalised and new_key not in normalised:
            normalised[new_key] = normalised[old_key]
    return normalised


def missing_required_args(func: Any, payload: dict[str, Any]) -> list[str]:
    signature = inspect.signature(func)
    missing = []
    for name, param in signature.parameters.items():
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        if param.default is inspect.Parameter.empty and name not in payload:
            missing.append(name)
    return missing


def _audit_mcp_args_row(
    *,
    key: str,
    tool: str,
    args_path: Path,
    callables: dict[str, Any],
    arg_aliases: dict[str, dict[str, str]],
) -> dict[str, Any]:
    issues = []
    payload: dict[str, Any] = {}
    json_object = False
    if not tool:
        issues.append("unknown_manifest_key")
    if tool and tool not in callables:
        issues.append("unsupported_tool")
    if not args_path.exists():
        issues.append("missing_args_file")
    else:
        try:
            payload = read_mcp_args(args_path)
            json_object = True
        except Exception as exc:
            issues.append(f"invalid_args_json:{exc}")
    ignored_args: list[str] = []
    missing_required: list[str] = []
    if tool in callables and json_object:
        payload = normalise_mcp_payload(tool, payload, arg_aliases)
        call_payload, ignored_args = filter_mcp_payload(callables[tool], payload)
        missing_required = missing_required_args(callables[tool], call_payload)
        if missing_required:
            issues.append("missing_required_args")
    return {
        "key": key,
        "tool": tool,
        "args_path": str(args_path),
        "exists": args_path.exists(),
        "json_object": json_object,
        "ignored_args": ignored_args,
        "missing_required_args": missing_required,
        "ok": not issues,
        "issues": issues,
    }


def _resolve_bundle_mcp_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return root / path


def _mcp_tool_for_manifest_key(key: str, *, tool_by_manifest_key: dict[str, str]) -> str:
    explicit = tool_by_manifest_key.get(key)
    if explicit:
        return explicit
    return key.removeprefix("mcp_").removesuffix("_args")
