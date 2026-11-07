from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import now_iso


def collect_source_artifacts(
    tool: str,
    source_dir: str | Path,
    *,
    copied_paths: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Build a small manifest of original extractor outputs for audit/review."""
    root = Path(source_dir)
    copied = {key: str(value) for key, value in (copied_paths or {}).items() if str(value)}
    artifacts = []
    for key, relative, kind, label in _artifact_specs(tool):
        path = root / relative
        artifacts.append(
            {
                "key": key,
                "label": label,
                "kind": kind,
                "path": str(path),
                "exists": path.exists(),
                "copied_path": copied.get(key, ""),
            }
        )
    return {
        "schema": "lecture_source_artifacts.v1",
        "tool": tool,
        "source_dir": str(root),
        "artifacts": artifacts,
        "available_count": len([item for item in artifacts if item["exists"] or item.get("copied_path")]),
        "missing_count": len([item for item in artifacts if not item["exists"] and not item.get("copied_path")]),
    }


def summarize_manifest_source_artifacts(manifest: dict[str, Any], *, expected_tool: str = "") -> dict[str, Any]:
    """Summarize source artifact availability from a WebUI bundle manifest."""
    sources = manifest.get("sources") if isinstance(manifest.get("sources"), list) else []
    artifact_count = 0
    source_count = 0
    tools: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        source_artifacts = source.get("source_artifacts") if isinstance(source.get("source_artifacts"), dict) else {}
        artifacts = source_artifacts.get("artifacts") if isinstance(source_artifacts.get("artifacts"), list) else []
        if not artifacts:
            artifacts = _fallback_source_artifacts(source)
        available = [item for item in artifacts if isinstance(item, dict) and (item.get("exists") or item.get("copied_path"))]
        if available:
            source_count += 1
            artifact_count += len(available)
        tool = str(source_artifacts.get("tool") or ("local_video" if artifacts else "")).strip()
        if tool and tool not in tools:
            tools.append(tool)
    expected = expected_tool.strip().lower()
    return {
        "source_count": len([source for source in sources if isinstance(source, dict)]),
        "sources_with_artifacts": source_count,
        "artifact_count": artifact_count,
        "tools": tools,
        "expected_tool": expected,
        "expected_tool_present": not expected or expected in {tool.lower() for tool in tools},
        "ok": artifact_count > 0 and (not expected or expected in {tool.lower() for tool in tools}),
    }


def build_source_artifact_index(package: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic index of original extractor outputs kept for traceability."""
    sources = package.get("sources") if isinstance(package.get("sources"), list) else []
    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    tools: list[str] = []
    for source_index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            continue
        source_artifacts = source.get("source_artifacts") if isinstance(source.get("source_artifacts"), dict) else {}
        tool = str(source_artifacts.get("tool") or "").strip()
        artifacts = source_artifacts.get("artifacts") if isinstance(source_artifacts.get("artifacts"), list) else []
        if not artifacts:
            artifacts = _fallback_source_artifacts(source)
            if artifacts:
                tool = "local_video"
        if tool and tool not in tools:
            tools.append(tool)
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            row = {
                "source_index": source_index,
                "video_id": str(source.get("video_id") or ""),
                "title": str(source.get("title") or ""),
                "video_path": str(source.get("path") or ""),
                "tool": tool,
                "source_dir": str(source_artifacts.get("source_dir") or ""),
                "key": str(artifact.get("key") or ""),
                "label": str(artifact.get("label") or artifact.get("key") or "artifact"),
                "kind": str(artifact.get("kind") or ""),
                "path": str(artifact.get("path") or ""),
                "copied_path": str(artifact.get("copied_path") or ""),
                "exists": bool(artifact.get("exists")),
                "available": bool(artifact.get("exists") or artifact.get("copied_path")),
            }
            rows.append(row)
            if not row["available"]:
                missing_rows.append(row)
    for artifact in _package_artifacts(package):
        tool = str(artifact.get("tool") or "local_bundle")
        if tool not in tools:
            tools.append(tool)
        row = {
            "source_index": 0,
            "video_id": "",
            "title": str(package.get("title") or "Lecture Package"),
            "video_path": "",
            "tool": tool,
            "source_dir": str(package.get("bundle_dir") or ""),
            "key": str(artifact.get("key") or ""),
            "label": str(artifact.get("label") or artifact.get("key") or "artifact"),
            "kind": str(artifact.get("kind") or ""),
            "path": str(artifact.get("path") or ""),
            "copied_path": str(artifact.get("copied_path") or ""),
            "exists": bool(artifact.get("exists")),
            "available": bool(artifact.get("exists") or artifact.get("copied_path")),
        }
        rows.append(row)
        if not row["available"]:
            missing_rows.append(row)
    available_rows = [row for row in rows if row["available"]]
    return {
        "schema": "lecture_source_artifact_index.v1",
        "title": str(package.get("title") or "Lecture Package"),
        "created_at": now_iso(),
        "source_count": len([source for source in sources if isinstance(source, dict)]),
        "artifact_count": len(rows),
        "available_count": len(available_rows),
        "missing_count": len(missing_rows),
        "tools": tools,
        "artifacts": rows,
        "missing": missing_rows,
    }


def render_source_artifact_index_markdown(index: dict[str, Any]) -> str:
    """Render original extractor artifacts as a Markdown traceability index."""
    lines = [
        "---",
        "type: lecture-source-artifacts",
        f'title: "{index.get("title", "Lecture Package")} - 原始抽取物"',
        "tags: [lecture-video, source-artifacts, traceability]",
        f'created: "{index.get("created_at", now_iso())}"',
        "---",
        "",
        f"# 原始抽取物：{index.get('title', 'Lecture Package')}",
        "",
        "这个索引用来追溯 vidclaude / peepshow / vidwise 等现成工具的真实输出。它不是摘要；当结构化笔记可疑时，优先回到这里列出的原始文件核对。",
        "",
        "## 总览",
        "",
        f"- 来源视频数：{index.get('source_count', 0)}",
        f"- 抽取物总数：{index.get('artifact_count', 0)}",
        f"- 可用抽取物：{index.get('available_count', 0)}",
        f"- 缺失抽取物：{index.get('missing_count', 0)}",
        f"- 工具：{', '.join(index.get('tools') or []) or 'unknown'}",
        "",
        "## 可用抽取物",
        "",
    ]
    available = [row for row in index.get("artifacts", []) if isinstance(row, dict) and row.get("available")]
    if not available:
        lines.append("当前没有可用原始抽取物。")
    else:
        current_source = ""
        for row in available:
            source_key = f"{row.get('source_index', '')}. {row.get('title') or row.get('video_id') or 'source'}"
            if source_key != current_source:
                current_source = source_key
                lines.extend(["", f"### {source_key}", ""])
                video_path = str(row.get("video_path") or "").strip()
                source_dir = str(row.get("source_dir") or "").strip()
                if video_path:
                    lines.append(f"- 视频：`{video_path}`")
                if source_dir:
                    lines.append(f"- 工具输出目录：`{source_dir}`")
                lines.append("")
            location = str(row.get("copied_path") or row.get("path") or "").strip()
            lines.append(
                f"- **{row.get('label') or row.get('key')}** `{row.get('kind', '')}` `{row.get('tool', '')}`: `{location}`"
            )
    missing = [row for row in index.get("missing", []) if isinstance(row, dict)]
    lines.extend(["", "## 缺失抽取物", ""])
    if not missing:
        lines.append("当前没有记录到缺失的原始抽取物。")
    else:
        for row in missing:
            lines.append(
                f"- {row.get('title') or row.get('video_id') or 'source'} / {row.get('label') or row.get('key')}: `{row.get('path', '')}`"
            )
    return "\n".join(lines).rstrip() + "\n"


def _artifact_specs(tool: str) -> list[tuple[str, Path, str, str]]:
    specs: dict[str, list[tuple[str, str, str, str]]] = {
        "vidclaude": [
            ("meta", "meta.json", "metadata", "vidclaude metadata"),
            ("transcript", "transcript.json", "transcript", "vidclaude transcript cues"),
            ("timeline", "timeline.json", "timeline", "vidclaude fused timeline"),
            ("evidence", "evidence.md", "evidence_markdown", "vidclaude evidence markdown"),
            ("frames", "frames", "frame_directory", "vidclaude frame directory"),
        ],
        "peepshow": [
            ("manifest", "manifest.json", "metadata", "peepshow manifest"),
            ("report", "report.html", "html_report", "peepshow HTML report"),
            ("frames", "frames", "frame_directory", "peepshow frame directory"),
            ("ocr", "ocr.json", "ocr", "peepshow OCR JSON"),
        ],
        "vidwise": [
            ("video", "video.mp4", "media", "vidwise copied video"),
            ("transcript", "transcript.json", "transcript", "vidwise transcript JSON"),
            ("srt", "transcript.srt", "transcript", "vidwise SRT transcript"),
            ("guide", "guide.md", "guide_markdown", "vidwise guide markdown"),
            ("frames", "frames", "frame_directory", "vidwise frame directory"),
        ],
    }
    return [(key, Path(relative), kind, label) for key, relative, kind, label in specs.get(tool, [])]


def _fallback_source_artifacts(source: dict[str, Any]) -> list[dict[str, Any]]:
    video_path = Path(str(source.get("path") or "")).expanduser()
    if not str(source.get("path") or "").strip():
        return []
    return [
        {
            "key": "video",
            "label": "local source video",
            "kind": "media",
            "path": str(video_path),
            "exists": video_path.exists(),
            "copied_path": "",
        }
    ]


def _package_artifacts(package: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [
        ("source_package", package.get("source_package"), "package_json", "lecture source package"),
        ("manifest", package.get("manifest_path"), "manifest_json", "WebUI bundle manifest"),
        ("timeline", package.get("timeline_path"), "timeline_json", "WebUI timeline"),
        ("asset_manifest", package.get("asset_manifest"), "asset_manifest_json", "frame asset manifest"),
        ("page_metadata_json", package.get("page_metadata_json"), "page_metadata_json", "normalized webpage/source metadata"),
        ("page_metadata_markdown", package.get("page_metadata_markdown"), "page_metadata_markdown", "readable webpage/source metadata"),
    ]
    artifacts: list[dict[str, Any]] = []
    for key, raw_path, kind, label in specs:
        path_text = str(raw_path or "").strip()
        if not path_text:
            continue
        path = Path(path_text).expanduser()
        artifacts.append(
            {
                "key": key,
                "label": label,
                "kind": kind,
                "path": str(path),
                "exists": path.exists(),
                "copied_path": "",
                "tool": "local_bundle",
            }
        )
    return artifacts
