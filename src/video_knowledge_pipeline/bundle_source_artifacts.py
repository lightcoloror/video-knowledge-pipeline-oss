from __future__ import annotations

from pathlib import Path
from typing import Any

from .source_artifacts import build_source_artifact_index, render_source_artifact_index_markdown
from .storage import read_json, write_json


def bundle_source_artifacts(bundle_dir: str | Path, *, refresh: bool = False, write: bool = False) -> dict[str, Any]:
    """Read or rebuild the source-artifact traceability index for a WebUI bundle."""
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")

    json_path = root / str(manifest.get("source_artifacts_json") or "source-artifacts.json")
    markdown_path = root / str(manifest.get("source_artifacts") or "source-artifacts.md")
    generated_from_manifest = False
    index: dict[str, Any]
    if json_path.exists() and not refresh:
        loaded = read_json(json_path)
        index = loaded if isinstance(loaded, dict) else {}
    else:
        index = build_source_artifact_index(
            {
                "title": manifest.get("title") or root.name,
                "sources": manifest.get("sources") if isinstance(manifest.get("sources"), list) else [],
                "bundle_dir": str(root),
                "source_package": manifest.get("source_package") or "",
                "manifest_path": str(manifest_path),
                "timeline_path": str(root / str(manifest.get("timeline_json") or "timeline.json")),
                "asset_manifest": str(root / str(manifest.get("asset_manifest") or "assets/asset-manifest.json")),
                "page_metadata_json": str(root / str(manifest.get("page_metadata_json"))) if manifest.get("page_metadata_json") else "",
                "page_metadata_markdown": str(root / str(manifest.get("page_metadata_markdown"))) if manifest.get("page_metadata_markdown") else "",
            }
        )
        generated_from_manifest = True

    markdown = render_source_artifact_index_markdown(index) if index else ""
    if write:
        write_json(json_path, index)
        markdown_path.write_text(markdown, encoding="utf-8")
        if manifest.get("source_artifacts") != markdown_path.name or manifest.get("source_artifacts_json") != json_path.name:
            manifest["source_artifacts"] = markdown_path.name
            manifest["source_artifacts_json"] = json_path.name
            write_json(manifest_path, manifest)

    artifacts = index.get("artifacts") if isinstance(index.get("artifacts"), list) else []
    available = [item for item in artifacts if isinstance(item, dict) and item.get("available")]
    missing = [item for item in artifacts if isinstance(item, dict) and not item.get("available")]
    args_path = root / "mcp-bundle-source-artifacts.args.json"
    write_json(args_path, {"bundle_dir": str(root), "refresh": False, "write": False})
    return {
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "source_artifacts_path": str(markdown_path),
        "source_artifacts_json_path": str(json_path),
        "source_artifacts_exists": json_path.exists(),
        "source_artifacts_markdown_exists": markdown_path.exists(),
        "generated_from_manifest": generated_from_manifest,
        "written": bool(write),
        "mcp_args_path": str(args_path),
        "summary": {
            "source_count": index.get("source_count", 0),
            "artifact_count": index.get("artifact_count", 0),
            "available_count": len(available),
            "missing_count": len(missing),
            "tools": index.get("tools") or [],
        },
        "source_artifacts": index,
        "available": available,
        "missing": missing,
        "markdown_preview": markdown[:4000],
        "next_action": _next_action(index, json_path, markdown_path, args_path),
    }


def _next_action(index: dict[str, Any], json_path: Path, markdown_path: Path, args_path: Path) -> dict[str, Any]:
    missing_count = int(index.get("missing_count") or 0) if isinstance(index, dict) else 0
    source_count = int(index.get("source_count") or 0) if isinstance(index, dict) else 0
    artifact_count = int(index.get("artifact_count") or 0) if isinstance(index, dict) else 0
    if not json_path.exists() or not markdown_path.exists():
        return _with_mcp_action(
            {
            "key": "write_source_artifact_index",
            "label": "写入原始抽取物索引",
            "command_hint": "Run bundle-source-artifacts --write for this bundle.",
            "human_required": False,
            },
            args_path,
        )
    if source_count and artifact_count == 0:
        return _with_mcp_action(
            {
            "key": "inspect_missing_source_artifacts",
            "label": "检查缺失原始抽取物",
            "command_hint": "The bundle has sources but no source artifacts; inspect imports or rebuild the index after source artifacts are present.",
            "human_required": True,
            },
            args_path,
        )
    if missing_count:
        return _with_mcp_action(
            {
            "key": "inspect_missing_source_artifacts",
            "label": "检查缺失原始抽取物",
            "command_hint": "Open source-artifacts.md and inspect the missing section.",
            "human_required": True,
            },
            args_path,
        )
    return _with_mcp_action(
        {
        "key": "ready",
        "label": "原始抽取物索引可用",
        "command_hint": "Use source-artifacts.md/json for traceability checks.",
        "human_required": False,
        },
        args_path,
    )


def _with_mcp_action(action: dict[str, Any], args_path: Path) -> dict[str, Any]:
    escaped_args_path = str(args_path).replace("'", "''")
    return {
        **action,
        "mcp_tool": "bundle_source_artifacts",
        "mcp_args_path": str(args_path),
        "command": f".\\scripts\\video-knowledge.ps1 mcp-call bundle_source_artifacts '{escaped_args_path}'",
    }

