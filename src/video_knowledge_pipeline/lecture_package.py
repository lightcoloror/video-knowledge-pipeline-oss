from __future__ import annotations
from .review_export_naming import review_export_script

import html
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .asr_adapter import read_asr_word_timestamps
from .interval_coverage import merge_nonnegative_intervals as _merge_intervals
from .lecture_outline import (
    generate_lecture_outline,
    render_lecture_outline_markdown,
    write_lecture_outline,
)
from .lecture_review_queue import (
    generate_lecture_review_queue,
    render_lecture_review_queue_markdown,
    write_lecture_review_queue,
)
from .lecture_search import (
    generate_lecture_search_index,
    render_lecture_search_index_markdown,
    write_lecture_search_index,
)
from .lecture_study_index import (
    generate_lecture_study_cards,
    generate_lecture_study_index,
    render_lecture_study_cards_markdown,
    render_lecture_study_index_markdown,
    write_lecture_study_index,
)
from .markdown_text import markdown_table_cell as _md_cell
from .models import now_iso
from .obsidian_export_status import obsidian_export_status
from .path_utils import file_uri_or_empty as _file_uri
from .source_artifacts import (
    build_source_artifact_index,
    render_source_artifact_index_markdown,
)
from .storage import ensure_project_dirs, read_json, write_json
from .transcript import format_timestamp


def build_lecture_package(root: str | Path, *, title: str | None = None, merge_window: float = 1.0) -> dict[str, Any]:
    """Build a no-summary lecture knowledge package from imported video extractor outputs."""
    paths = ensure_project_dirs(root)
    package_dir = paths["lecture_packages"]
    package_dir.mkdir(parents=True, exist_ok=True)

    videos = _load_imported_videos(paths["videos"])
    if not videos:
        raise ValueError("no imported video segments found; run import-vidclaude/import-peepshow/import-vidwise first")

    timeline = _merged_timeline(videos, merge_window=merge_window)
    time_gap_audit = _time_gap_audit(timeline)
    package = {
        "schema": "lecture_knowledge_package.v1",
        "title": title or _default_title(videos),
        "created_at": now_iso(),
        "source_count": len(videos),
        "timeline_count": len(timeline),
        "sources": [
            {
                "video_id": item["metadata"].get("id", ""),
                "title": item["metadata"].get("title", ""),
                "path": item["metadata"].get("path", ""),
                "duration_seconds": item["metadata"].get("duration_seconds", 0),
                "segment_count": len(item["segments"]),
                "source_artifacts": item.get("source_artifacts", {}),
                "source_artifact_count": (item.get("source_artifacts") or {}).get("available_count", 0),
            }
            for item in videos
        ],
        "time_gap_audit": time_gap_audit,
        "coverage": _coverage_audit(timeline, time_gap_audit=time_gap_audit),
        "quality_audit": _quality_audit(timeline),
        "timeline": timeline,
    }
    return _write_package_outputs(paths, package)


def import_lecture_review(root: str | Path, review_path: str | Path) -> dict[str, Any]:
    """Import human review notes exported by lecture-review.html."""
    paths = ensure_project_dirs(root)
    package_path = paths["lecture_packages"] / "lecture-package.json"
    if not package_path.exists():
        raise FileNotFoundError(f"lecture package not found: {package_path}")
    package = read_json(package_path)
    if not isinstance(package, dict):
        raise ValueError("lecture package must be a JSON object")

    review_payload = read_json(Path(review_path))
    reviews = review_payload.get("reviews") if isinstance(review_payload, dict) else None
    if not isinstance(reviews, list):
        raise ValueError("review JSON must contain a reviews list")

    timeline = package.get("timeline")
    if not isinstance(timeline, list):
        raise ValueError("lecture package timeline must be a list")

    updated = 0
    for review in reviews:
        if not isinstance(review, dict):
            continue
        if review.get("manualTimelineItem"):
            manual_item = _manual_timeline_item(review)
            existing = _find_timeline_item(timeline, {"sourceSegmentIds": manual_item.get("source_segment_ids", [])})
            if existing is not None:
                existing.clear()
                existing.update(manual_item)
            else:
                timeline.append(manual_item)
            updated += 1
            continue
        item = _find_timeline_item(timeline, review)
        if item is None:
            continue
        human_review = _normalise_human_review(review)
        _apply_text_corrections(item, human_review)
        _apply_structured_visual_corrections(item, human_review)
        item["human_review"] = human_review
        item["review_status"] = human_review["status"]
        item["needs_human_review"] = human_review["status"] != "reviewed"
        _refresh_visual_retention(item)
        updated += 1
    timeline.sort(
        key=lambda item: (
            str(item.get("video_key") or "") if isinstance(item, dict) else "",
            _safe_float(item.get("start")) if isinstance(item, dict) else 0.0,
        )
    )

    package["review_imported_at"] = now_iso()
    time_gap_audit = _time_gap_audit(timeline)
    package["time_gap_audit"] = time_gap_audit
    package["coverage"] = _coverage_audit(timeline, time_gap_audit=time_gap_audit)
    package["quality_audit"] = _quality_audit(timeline)
    package["timeline_count"] = len(timeline)
    return _write_package_outputs(paths, package, updated=updated)


def audit_lecture_package(root: str | Path) -> dict[str, Any]:
    """Write a gap-focused quality report for the current lecture package."""
    paths = ensure_project_dirs(root)
    package_path = paths["lecture_packages"] / "lecture-package.json"
    if not package_path.exists():
        raise FileNotFoundError(f"lecture package not found: {package_path}")
    package = read_json(package_path)
    if not isinstance(package, dict):
        raise ValueError("lecture package must be a JSON object")
    timeline = package.get("timeline")
    if not isinstance(timeline, list):
        raise ValueError("lecture package timeline must be a list")

    audit = _quality_audit(timeline)
    time_gap_audit = _time_gap_audit(timeline)
    package["quality_audit"] = audit
    package["time_gap_audit"] = time_gap_audit
    package["coverage"] = _coverage_audit(timeline, time_gap_audit=time_gap_audit)
    write_json(package_path, package)
    report_path = paths["notes"] / "lecture-quality-report.md"
    audit_path = paths["lecture_packages"] / "lecture-quality-audit.json"
    write_json(audit_path, audit)
    report_path.write_text(render_lecture_quality_report_markdown(package), encoding="utf-8")
    return {
        "package_path": str(package_path),
        "audit_path": str(audit_path),
        "report_path": str(report_path),
        "quality_audit": audit,
    }


def export_lecture_obsidian(
    root: str | Path,
    vault: str | Path,
    folder: str = "00_Inbox/AI/课程视频知识包",
) -> dict[str, Any]:
    """Export the lecture package as an Obsidian-friendly course folder."""
    paths = ensure_project_dirs(root)
    package_path = paths["lecture_packages"] / "lecture-package.json"
    if not package_path.exists():
        raise FileNotFoundError(f"lecture package not found: {package_path}")
    package = read_json(package_path)
    if not isinstance(package, dict):
        raise ValueError("lecture package must be a JSON object")

    output_dir = Path(vault) / _safe_relative_folder(folder) / _safe_filename(str(package.get("title") or "lecture-package"))
    output_dir.mkdir(parents=True, exist_ok=True)
    export_package, asset_export = _prepare_obsidian_asset_package(package, root=paths["root"], output_dir=output_dir)
    study_index = generate_lecture_study_index(export_package)
    study_cards = generate_lecture_study_cards(study_index)
    study_review_queue = generate_lecture_review_queue(study_cards)
    outline = generate_lecture_outline(export_package)
    search_index = generate_lecture_search_index(export_package)
    source_artifact_index = build_source_artifact_index(export_package)
    files = {
        "index.md": render_obsidian_course_index(export_package),
        "outline.md": render_lecture_outline_markdown(outline),
        "search-index.md": render_lecture_search_index_markdown(search_index),
        "transcript.md": render_obsidian_transcript(export_package),
        "screen-text.md": render_obsidian_screen_text(export_package),
        "timeline.md": render_obsidian_timeline(export_package),
        "study-index.md": render_lecture_study_index_markdown(study_index),
        "study-cards.md": render_lecture_study_cards_markdown(study_cards),
        "study-review-queue.md": render_lecture_review_queue_markdown(study_review_queue),
        "review-queue.md": render_obsidian_review_queue(export_package),
        "quality-report.md": render_lecture_quality_report_markdown(export_package),
        "visual-assets.md": render_obsidian_visual_assets(export_package),
        "structured-materials.md": render_obsidian_structured_materials(export_package),
        "evidence-map.md": render_obsidian_evidence_map(export_package, source_artifact_index),
        "source-artifacts.md": render_source_artifact_index_markdown(source_artifact_index),
        "agent-handoff.md": "",
    }
    exported = []
    for name, content in files.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        exported.append(str(path))
    asset_manifest_path = output_dir / "assets" / "asset-manifest.json"
    write_json(asset_manifest_path, asset_export)
    source_artifact_index_path = output_dir / "source-artifacts.json"
    write_json(source_artifact_index_path, source_artifact_index)
    export_args_path = output_dir / "mcp-export-lecture-obsidian.args.json"
    status_args_path = output_dir / "mcp-obsidian-export-status.args.json"
    export_manifest = _obsidian_export_manifest(
        export_package,
        output_dir=output_dir,
        files=files,
        asset_export=asset_export,
        source_artifact_index=source_artifact_index,
        project_root=paths["root"],
        vault=Path(vault),
        folder_arg=folder,
        export_args_path=export_args_path,
        status_args_path=status_args_path,
    )
    export_manifest_path = output_dir / "obsidian-export.json"
    write_json(export_manifest_path, export_manifest)
    write_json(export_args_path, {"project": str(paths["root"]), "vault": str(Path(vault)), "folder": folder})
    write_json(status_args_path, {"export": str(export_manifest_path), "write": True})
    handoff_json_path = output_dir / "agent-handoff.json"
    write_json(handoff_json_path, _obsidian_agent_handoff(export_manifest))
    (output_dir / "agent-handoff.md").write_text(render_obsidian_agent_handoff_markdown(export_manifest), encoding="utf-8")
    export_status = obsidian_export_status(export_manifest_path, write=True)
    handoff = _obsidian_agent_handoff(export_manifest, export_status=export_status)
    write_json(handoff_json_path, handoff)
    (output_dir / "agent-handoff.md").write_text(render_obsidian_agent_handoff_markdown(export_manifest, export_status=export_status), encoding="utf-8")
    index_path = output_dir / "index.md"
    index_path.write_text(render_obsidian_course_index(export_package, export_status=export_status), encoding="utf-8")
    return {
        "folder": str(output_dir),
        "exported": exported,
        "count": len(exported),
        "export_manifest_path": str(export_manifest_path),
        "mcp_export_args_path": str(export_args_path),
        "mcp_status_args_path": str(status_args_path),
        "agent_handoff_path": str(output_dir / "agent-handoff.md"),
        "agent_handoff_json_path": str(handoff_json_path),
        "export_status_path": str(output_dir / "obsidian-export-status.json"),
        "export_status_markdown_path": str(output_dir / "obsidian-export-status.md"),
        "export_status": {
            "schema": export_status.get("schema"),
            "status": export_status.get("status"),
            "missing_page_count": export_status.get("missing_page_count", 0),
            "missing_entrypoint_count": export_status.get("missing_entrypoint_count", 0),
            "missing_core_entrypoints": export_status.get("missing_core_entrypoints", []),
        },
        "pages": export_manifest["pages"],
        "pages_by_name": export_manifest["pages_by_name"],
        "agent_entrypoints": export_manifest["agent_entrypoints"],
        "assets": asset_export,
        "asset_manifest_path": str(asset_manifest_path),
        "source_artifact_index_path": str(source_artifact_index_path),
    }


def _obsidian_export_manifest(
    package: dict[str, Any],
    *,
    output_dir: Path,
    files: dict[str, str],
    asset_export: dict[str, Any],
    source_artifact_index: dict[str, Any],
    project_root: Path,
    vault: Path,
    folder_arg: str,
    export_args_path: Path,
    status_args_path: Path,
) -> dict[str, Any]:
    """Build a machine-readable map of the Obsidian course export."""
    page_specs = _obsidian_page_specs()
    pages = []
    for name in files:
        spec = page_specs.get(name, {})
        pages.append(
            {
                "name": name,
                "path": str(output_dir / name),
                "role": spec.get("role", "support"),
                "label": spec.get("label", name),
                "purpose": spec.get("purpose", ""),
                "agent_use": spec.get("agent_use", ""),
                "core_full_information_layer": bool(spec.get("core_full_information_layer")),
                "review_layer": bool(spec.get("review_layer")),
                "traceability_layer": bool(spec.get("traceability_layer")),
            }
        )
    by_name = {page["name"]: page for page in pages}
    return {
        "schema": "lecture_obsidian_export.v1",
        "title": str(package.get("title") or "Lecture Course"),
        "created_at": now_iso(),
        "folder": str(output_dir),
        "project": str(project_root),
        "vault": str(vault),
        "folder_arg": folder_arg,
        "page_count": len(pages),
        "pages": pages,
        "pages_by_name": by_name,
        "recommended_read_order": [
            "index.md",
            "quality-report.md",
            "review-queue.md",
            "transcript.md",
            "screen-text.md",
            "evidence-map.md",
            "structured-materials.md",
            "visual-assets.md",
            "source-artifacts.md",
            "agent-handoff.md",
        ],
        "agent_entrypoints": {
            "course_index": str(output_dir / "index.md"),
            "agent_handoff_markdown": str(output_dir / "agent-handoff.md"),
            "agent_handoff_json": str(output_dir / "agent-handoff.json"),
            "full_speech_text": str(output_dir / "transcript.md"),
            "full_screen_text": str(output_dir / "screen-text.md"),
            "timeline": str(output_dir / "timeline.md"),
            "evidence_map": str(output_dir / "evidence-map.md"),
            "structured_materials": str(output_dir / "structured-materials.md"),
            "source_artifacts_markdown": str(output_dir / "source-artifacts.md"),
            "source_artifacts_json": str(output_dir / "source-artifacts.json"),
            "asset_manifest": str(output_dir / "assets" / "asset-manifest.json"),
            "mcp_export_lecture_obsidian_args": str(export_args_path),
            "mcp_obsidian_export_status_args": str(status_args_path),
        },
        "asset_summary": {
            "copied_count": asset_export.get("copied_count", 0),
            "missing_count": asset_export.get("missing_count", 0),
        },
        "source_artifact_summary": {
            "available_count": source_artifact_index.get("available_count", 0),
            "missing_count": source_artifact_index.get("missing_count", 0),
            "tools": source_artifact_index.get("tools", []),
        },
    }


def _obsidian_page_specs() -> dict[str, dict[str, Any]]:
    return {
        "index.md": {
            "role": "navigation",
            "label": "课程首页",
            "purpose": "课程导出总入口。",
            "agent_use": "从这里发现所有 Obsidian 页面。",
        },
        "outline.md": {
            "role": "navigation",
            "label": "课程导航大纲",
            "purpose": "基于时间线组织章节和主题入口。",
            "agent_use": "用于按章节定位内容，不作为全量证据。",
        },
        "search-index.md": {
            "role": "retrieval",
            "label": "检索索引",
            "purpose": "提供面向检索的关键词和片段索引。",
            "agent_use": "先用它快速定位候选片段，再回到 evidence-map 或底稿页核对。",
        },
        "transcript.md": {
            "role": "full_information",
            "label": "全量口语/字幕",
            "purpose": "保留口语/字幕底稿，不做摘要。",
            "agent_use": "核对口头知识、定义、论证链时优先读取。",
            "core_full_information_layer": True,
        },
        "screen-text.md": {
            "role": "full_information",
            "label": "全量屏幕文字/OCR",
            "purpose": "保留屏幕文字、OCR、视觉观察和结构化视觉底稿。",
            "agent_use": "核对课件文字、公式、代码、表格标签时优先读取。",
            "core_full_information_layer": True,
        },
        "timeline.md": {
            "role": "full_information",
            "label": "全量时间线",
            "purpose": "按时间片融合语言、视觉、帧和复核状态。",
            "agent_use": "需要时间顺序和多模态融合视图时读取。",
            "core_full_information_layer": True,
        },
        "study-index.md": {
            "role": "study",
            "label": "学习索引",
            "purpose": "将时间线转成学习入口。",
            "agent_use": "生成复习材料前可参考，但必须回到底稿核对。",
        },
        "study-cards.md": {
            "role": "study",
            "label": "学习卡片草稿",
            "purpose": "复习卡片草稿。",
            "agent_use": "作为草稿继续加工，不作为证据源。",
        },
        "study-review-queue.md": {
            "role": "study",
            "label": "复习队列",
            "purpose": "复习卡片队列。",
            "agent_use": "安排复习时读取。",
        },
        "review-queue.md": {
            "role": "review",
            "label": "人工复核队列",
            "purpose": "列出仍需人工确认或修订的片段。",
            "agent_use": "决定下一步人工审核顺序。",
            "review_layer": True,
        },
        "quality-report.md": {
            "role": "review",
            "label": "缺口审计",
            "purpose": "暴露转写、OCR、关键帧、时间轴等缺口。",
            "agent_use": "决定是否继续补 ASR/OCR/抽帧。",
            "review_layer": True,
        },
        "visual-assets.md": {
            "role": "visual",
            "label": "视觉资产索引",
            "purpose": "集中列出关键帧和视觉观察。",
            "agent_use": "核对必须看图的内容。",
            "traceability_layer": True,
        },
        "structured-materials.md": {
            "role": "structured_materials",
            "label": "结构材料索引",
            "purpose": "集中列出公式、表格、代码、结构化视觉和保留图片片段。",
            "agent_use": "处理不能只靠自然语言表达的知识材料。",
            "core_full_information_layer": True,
        },
        "evidence-map.md": {
            "role": "traceability",
            "label": "逐片段证据地图",
            "purpose": "按时间片回溯语言、OCR、关键帧和原始抽取物。",
            "agent_use": "发现可疑笔记或卡片时，从这里回到真实证据。",
            "traceability_layer": True,
        },
        "source-artifacts.md": {
            "role": "traceability",
            "label": "原始抽取物",
            "purpose": "按工具输出文件索引 vidclaude/peepshow/vidwise 真实产物。",
            "agent_use": "需要检查工具原始输出时读取。",
            "traceability_layer": True,
        },
        "agent-handoff.md": {
            "role": "agent_handoff",
            "label": "Agent handoff",
            "purpose": "汇总 Obsidian 导出后的 agent 入口、状态和 MCP 命令。",
            "agent_use": "AI agent 从这里继续验证、重新导出或回到底稿证据。",
        },
    }


def _obsidian_agent_handoff(export_manifest: dict[str, Any], *, export_status: dict[str, Any] | None = None) -> dict[str, Any]:
    entrypoints = export_manifest.get("agent_entrypoints") if isinstance(export_manifest.get("agent_entrypoints"), dict) else {}
    return {
        "schema": "lecture_obsidian_agent_handoff.v1",
        "title": export_manifest.get("title", ""),
        "created_at": now_iso(),
        "folder": export_manifest.get("folder", ""),
        "project": export_manifest.get("project", ""),
        "status": _compact_export_status(export_status),
        "recommended_read_order": export_manifest.get("recommended_read_order") or [],
        "agent_entrypoints": entrypoints,
        "mcp_commands": _obsidian_handoff_mcp_commands(entrypoints),
        "core_pages": [
            page
            for page in export_manifest.get("pages") or []
            if isinstance(page, dict) and page.get("core_full_information_layer")
        ],
        "traceability_pages": [
            page
            for page in export_manifest.get("pages") or []
            if isinstance(page, dict) and page.get("traceability_layer")
        ],
    }


def render_obsidian_agent_handoff_markdown(
    export_manifest: dict[str, Any],
    *,
    export_status: dict[str, Any] | None = None,
) -> str:
    handoff = _obsidian_agent_handoff(export_manifest, export_status=export_status)
    status = handoff.get("status") if isinstance(handoff.get("status"), dict) else {}
    entrypoints = handoff.get("agent_entrypoints") if isinstance(handoff.get("agent_entrypoints"), dict) else {}
    lines = [
        "---",
        "type: lecture-agent-handoff",
        f'title: "{handoff.get("title", "Lecture Agent Handoff")}"',
        "tags: [lecture-video, agent-handoff, mcp]",
        f'created: "{handoff.get("created_at", now_iso())}"',
        "---",
        "",
        f"# Agent handoff：{handoff.get('title', '')}",
        "",
        "这个页面用于让 AI agent 在 Obsidian 知识库内继续工作：先检查导出状态，再按需读取全量底稿、证据地图或调用现有 MCP 工具。它不是摘要。",
        "",
        f"- 导出状态：`{status.get('status', 'unknown')}`",
        f"- 导出目录：`{handoff.get('folder', '')}`",
        f"- 项目目录：`{handoff.get('project', '')}`",
        f"- 缺核心入口：{', '.join(status.get('missing_core_entrypoints') or []) or 'none'}",
        "",
        "## MCP 命令",
        "",
        "| Key | Tool | Command |",
        "|---|---|---|",
    ]
    for row in handoff.get("mcp_commands") or []:
        if isinstance(row, dict):
            lines.append(f"| {row.get('key', '')} | `{row.get('mcp_tool', '')}` | `{_md_cell(str(row.get('command') or ''))}` |")
    lines.extend(["", "## 核心底稿入口", "", "| Key | Path |", "|---|---|"])
    for key in ("full_speech_text", "full_screen_text", "timeline", "structured_materials", "evidence_map", "source_artifacts_json"):
        value = entrypoints.get(key)
        if value:
            lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## 推荐读取顺序", ""])
    for page in handoff.get("recommended_read_order") or []:
        lines.append(f"- [[{Path(str(page)).stem}|{page}]]")
    next_action = status.get("next_action") if isinstance(status.get("next_action"), dict) else {}
    if next_action:
        lines.extend(
            [
                "",
                "## 下一步",
                "",
                f"- `{next_action.get('key', '')}` / {next_action.get('label', '')}",
                f"- {next_action.get('hint', '')}",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _obsidian_handoff_mcp_commands(entrypoints: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for key, tool in (
        ("mcp_export_lecture_obsidian_args", "export_lecture_obsidian"),
        ("mcp_obsidian_export_status_args", "obsidian_export_status"),
    ):
        args_path = str(entrypoints.get(key) or "").strip()
        if args_path:
            rows.append(
                {
                    "key": key,
                    "mcp_tool": tool,
                    "mcp_args_path": args_path,
                    "command": _mcp_call_command(tool, args_path),
                }
            )
    return rows


def _compact_export_status(export_status: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(export_status, dict):
        return {"status": "pending", "next_action": {}}
    return {
        "schema": export_status.get("schema"),
        "status": export_status.get("status"),
        "missing_page_count": export_status.get("missing_page_count", 0),
        "missing_entrypoint_count": export_status.get("missing_entrypoint_count", 0),
        "missing_core_entrypoints": export_status.get("missing_core_entrypoints", []),
        "next_action": export_status.get("next_action") if isinstance(export_status.get("next_action"), dict) else {},
    }


def _mcp_call_command(tool: str, args_path: str) -> str:
    escaped = str(args_path).replace("'", "''")
    return f".\\scripts\\video-knowledge.ps1 mcp-call {tool} '{escaped}'"




def _prepare_obsidian_asset_package(
    package: dict[str, Any],
    *,
    root: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy frame assets into the exported course folder and rewrite links."""
    export_package = json.loads(json.dumps(package, ensure_ascii=False))
    timeline = export_package.get("timeline") if isinstance(export_package.get("timeline"), list) else []
    assets_dir = output_dir / "assets"
    copied: list[dict[str, str]] = []
    missing: list[str] = []
    for item_index, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        frames = item.get("frame_paths") if isinstance(item.get("frame_paths"), list) else []
        rewritten: list[str] = []
        for frame_index, frame in enumerate(frames, start=1):
            source = _resolve_export_asset(frame, root=root)
            if not source:
                original = str(frame)
                rewritten.append(original)
                missing.append(original)
                continue
            assets_dir.mkdir(parents=True, exist_ok=True)
            target_name = _export_asset_name(source, item_index=item_index, frame_index=frame_index)
            target = assets_dir / target_name
            shutil.copy2(source, target)
            relative = f"assets/{target_name}"
            rewritten.append(relative)
            copied.append({"source": str(source), "target": str(target), "relative": relative})
        item["frame_paths"] = rewritten
    return export_package, {
        "schema": "lecture_obsidian_asset_export.v1",
        "copied_count": len(copied),
        "missing_count": len(missing),
        "copied": copied,
        "missing": missing,
    }


def _resolve_export_asset(frame: Any, *, root: Path) -> Path | None:
    value = str(frame or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(root / path)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _export_asset_name(source: Path, *, item_index: int, frame_index: int) -> str:
    suffix = source.suffix or ".jpg"
    stem = _safe_filename(source.stem)[:80] or "frame"
    return f"frame-{item_index:04d}-{frame_index:02d}-{stem}{suffix}"


def _write_package_outputs(paths: dict[str, Path], package: dict[str, Any], *, updated: int | None = None) -> dict[str, Any]:
    package_dir = paths["lecture_packages"]
    package_dir.mkdir(parents=True, exist_ok=True)
    package_path = package_dir / "lecture-package.json"
    markdown_path = paths["notes"] / "lecture-knowledge-package.md"
    html_path = paths["notes"] / "lecture-review.html"
    quality_report_path = paths["notes"] / "lecture-quality-report.md"
    quality_audit_path = package_dir / "lecture-quality-audit.json"
    timeline = package.get("timeline", [])
    if isinstance(timeline, list):
        time_gap_audit = _time_gap_audit(timeline)
        package["time_gap_audit"] = time_gap_audit
        package["coverage"] = _coverage_audit(timeline, time_gap_audit=time_gap_audit)
        package["quality_audit"] = _quality_audit(timeline)
    write_json(package_path, package)
    write_json(quality_audit_path, package.get("quality_audit", {}))
    outline = write_lecture_outline(paths, package)
    search_index = write_lecture_search_index(paths, package)
    study_index = write_lecture_study_index(paths, package)
    study_review_queue = write_lecture_review_queue(paths, package)
    markdown_path.write_text(render_lecture_package_markdown(package), encoding="utf-8")
    html_path.write_text(render_lecture_review_html(package), encoding="utf-8")
    quality_report_path.write_text(render_lecture_quality_report_markdown(package), encoding="utf-8")
    result = {
        "package_path": str(package_path),
        "markdown_path": str(markdown_path),
        "html_path": str(html_path),
        "quality_report_path": str(quality_report_path),
        "quality_audit_path": str(quality_audit_path),
        "outline_path": outline["outline_path"],
        "outline_markdown_path": outline["outline_markdown_path"],
        "search_index_path": search_index["search_index_path"],
        "search_index_markdown_path": search_index["search_index_markdown_path"],
        "study_index_path": study_index["study_index_path"],
        "study_index_markdown_path": study_index["study_index_markdown_path"],
        "study_cards_path": study_index["study_cards_path"],
        "study_cards_markdown_path": study_index["study_cards_markdown_path"],
        "review_queue_path": study_review_queue["review_queue_path"],
        "review_queue_markdown_path": study_review_queue["review_queue_markdown_path"],
        "review_tasks_markdown_path": study_review_queue["review_tasks_markdown_path"],
        "review_anki_csv_path": study_review_queue["review_anki_csv_path"],
        "timeline_count": len(package.get("timeline", [])),
        "coverage": package.get("coverage", {}),
        "quality_audit": package.get("quality_audit", {}),
    }
    if updated is not None:
        result["updated"] = updated
    return result


def render_obsidian_course_index(package: dict[str, Any], *, export_status: dict[str, Any] | None = None) -> str:
    coverage = package.get("coverage", {})
    lines = [
        "---",
        "type: lecture-course",
        f'title: "{package.get("title", "Lecture Course")}"',
        "tags: [lecture-video, knowledge-package, ai-extracted]",
        f'created: "{now_iso()}"',
        "---",
        "",
        f"# {package.get('title', 'Lecture Course')}",
        "",
        "## 入口",
        "",
        "- [[timeline|全量时间线]]",
        "- [[transcript|全量口语/字幕]]",
        "- [[screen-text|全量屏幕文字/OCR]]",
        "- [[outline|课程导航大纲]]",
        "- [[search-index|检索索引]]",
        "- [[study-index|学习索引]]",
        "- [[study-cards|学习卡片草稿]]",
        "- [[study-review-queue|复习队列]]",
        "- [[review-queue|人工复核队列]]",
        "- [[quality-report|缺口审计]]",
        "- [[visual-assets|视觉资产索引]]",
        "- [[structured-materials|结构材料索引]]",
        "- [[evidence-map|逐片段证据地图]]",
        "- [[source-artifacts|原始抽取物]]",
        "- [[agent-handoff|Agent handoff]]",
        "- [[obsidian-export-status|导出状态]]",
        "",
        "## 覆盖率审计",
        "",
    ]
    for key, value in coverage.items():
        lines.append(f"- `{key}`: {value}")
    if export_status is not None:
        lines.extend(_obsidian_export_status_summary_lines(export_status))
    lines.extend(["", "## 来源", ""])
    for source in package.get("sources", []):
        lines.append(_source_markdown(source))
    lines.extend(
        [
            "",
            "## 使用说明",
            "",
            "- 先看 `review-queue`，处理所有待复核和需修订片段。",
            "- 对必须保留图片的内容，不要强行改写成纯文本。",
            "- 对已确认片段，可以继续拆成概念卡、例题卡或章节笔记。",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _obsidian_export_status_summary_lines(export_status: dict[str, Any]) -> list[str]:
    asset_validation = export_status.get("asset_validation") if isinstance(export_status.get("asset_validation"), dict) else {}
    source_validation = (
        export_status.get("source_artifact_validation")
        if isinstance(export_status.get("source_artifact_validation"), dict)
        else {}
    )
    next_action = export_status.get("next_action") if isinstance(export_status.get("next_action"), dict) else {}
    return [
        "",
        "## 导出状态",
        "",
        f"- 状态：`{export_status.get('status', 'unknown')}`",
        f"- 页面：{export_status.get('page_count', 0)}，缺失 {export_status.get('missing_page_count', 0)}",
        f"- Agent 入口：{export_status.get('entrypoint_count', 0)}，缺失 {export_status.get('missing_entrypoint_count', 0)}",
        f"- 缺核心入口：{', '.join(export_status.get('missing_core_entrypoints') or []) or 'none'}",
        f"- 关键帧资产缺失：{asset_validation.get('missing_visual_asset_count', 0)}",
        f"- 原始抽取物缺失：{source_validation.get('missing_source_artifact_count', 0)}",
        f"- 下一步：`{next_action.get('key', '')}` / {next_action.get('label', '')}",
        "- 详情：[[obsidian-export-status|导出状态]]",
    ]


def render_obsidian_timeline(package: dict[str, Any]) -> str:
    lines = [
        "---",
        "type: lecture-timeline",
        f'title: "{package.get("title", "Lecture Course")} - 全量时间线"',
        "tags: [lecture-video, timeline]",
        f'created: "{now_iso()}"',
        "---",
        "",
        f"# 全量时间线：{package.get('title', 'Lecture Course')}",
        "",
    ]
    for index, item in enumerate(package.get("timeline", []), start=1):
        lines.extend(_obsidian_timeline_item(index, item, include_review=True))
    return "\n".join(lines).rstrip() + "\n"


def render_obsidian_transcript(package: dict[str, Any]) -> str:
    """Render the full spoken/subtitle layer without summarising it."""
    timeline = package.get("timeline") if isinstance(package.get("timeline"), list) else []
    items = [item for item in timeline if isinstance(item, dict)]
    lines = [
        "---",
        "type: lecture-transcript",
        f'title: "{package.get("title", "Lecture Course")} - 全量口语/字幕"',
        "tags: [lecture-video, transcript, no-summary]",
        f'created: "{now_iso()}"',
        "---",
        "",
        f"# 全量口语/字幕：{package.get('title', 'Lecture Course')}",
        "",
        "这个页面只承载口语/字幕层，按时间顺序保留原始或人工修正后的文本。不要把它当摘要；缺失片段会显式标出，方便回到 ASR 或视频复核。",
        "",
        f"- 片段数：{len(items)}",
        f"- 含口语/字幕：{sum(1 for item in items if str(item.get('transcript') or '').strip())}",
        f"- 缺口语/字幕：{sum(1 for item in items if not str(item.get('transcript') or '').strip())}",
        "",
    ]
    if not items:
        lines.append("当前没有时间线片段。")
        return "\n".join(lines).rstrip() + "\n"

    current_video = None
    for index, item in enumerate(items, start=1):
        video_key = str(item.get("video_key") or "unknown")
        if video_key != current_video:
            current_video = video_key
            lines.extend([f"## 视频：{video_key}", ""])
        start = format_timestamp(float(item.get("start", 0)))
        end = format_timestamp(float(item.get("end", 0)))
        transcript = str(item.get("transcript") or "").strip()
        original = str(item.get("original_transcript") or "").strip()
        lines.extend(
            [
                f"### {index}. {start} - {end}",
                "",
                f"- 审核状态：`{item.get('review_status', 'pending')}`",
                f"- 来源片段：{', '.join(item.get('source_segment_ids') or [])}",
                f"- 已人工修正：{bool(original and original != transcript)}",
                "",
                transcript or "（缺失或未命中）",
                "",
            ]
        )
        if original and original != transcript:
            lines.extend(["原始口语/字幕：", "", original, ""])
    return "\n".join(lines).rstrip() + "\n"


def render_obsidian_screen_text(package: dict[str, Any]) -> str:
    """Render the full screen-text/OCR layer without collapsing it into notes."""
    timeline = package.get("timeline") if isinstance(package.get("timeline"), list) else []
    items = [item for item in timeline if isinstance(item, dict)]
    lines = [
        "---",
        "type: lecture-screen-text",
        f'title: "{package.get("title", "Lecture Course")} - 全量屏幕文字/OCR"',
        "tags: [lecture-video, screen-text, ocr, no-summary]",
        f'created: "{now_iso()}"',
        "---",
        "",
        f"# 全量屏幕文字/OCR：{package.get('title', 'Lecture Course')}",
        "",
        "这个页面只承载画面文字、OCR、视觉观察和结构化视觉层。它不是摘要；缺失片段会显式标出，方便回到截图、CaptiOCR、PaddleOCR、Docling/MinerU/Marker 或原始抽取物复核。",
        "",
        f"- 片段数：{len(items)}",
        f"- 含屏幕文字/OCR：{sum(1 for item in items if str(item.get('visual_text') or '').strip())}",
        f"- 含结构化视觉：{sum(1 for item in items if _structured_visual_markdown(item))}",
        f"- 缺屏幕文字/OCR：{sum(1 for item in items if not str(item.get('visual_text') or '').strip())}",
        "",
    ]
    if not items:
        lines.append("当前没有时间线片段。")
        return "\n".join(lines).rstrip() + "\n"

    current_video = None
    for index, item in enumerate(items, start=1):
        video_key = str(item.get("video_key") or "unknown")
        if video_key != current_video:
            current_video = video_key
            lines.extend([f"## 视频：{video_key}", ""])
        start = format_timestamp(float(item.get("start", 0)))
        end = format_timestamp(float(item.get("end", 0)))
        visual_text = str(item.get("visual_text") or "").strip()
        original = str(item.get("original_visual_text") or "").strip()
        structured = _structured_visual_markdown(item)
        frames = item.get("frame_paths") if isinstance(item.get("frame_paths"), list) else []
        lines.extend(
            [
                f"### {index}. {start} - {end}",
                "",
                f"- 审核状态：`{item.get('review_status', 'pending')}`",
                f"- 材料类型：{', '.join(str(value) for value in item.get('material_types') or []) or 'unknown'}",
                f"- 来源片段：{', '.join(item.get('source_segment_ids') or [])}",
                f"- 已人工修正：{bool(original and original != visual_text)}",
                f"- 关键帧数：{len(frames)}",
                "",
                "### OCR / 视觉观察",
                "",
                visual_text or "（缺失或未命中）",
                "",
            ]
        )
        if original and original != visual_text:
            lines.extend(["原始屏幕文字/OCR：", "", original, ""])
        if structured:
            lines.extend(["### 结构化视觉", "", structured, ""])
        if frames:
            lines.extend(["### 关键帧", ""])
            for frame in frames:
                lines.extend([_markdown_image(str(frame)), "", f"`{frame}`", ""])
    return "\n".join(lines).rstrip() + "\n"


def render_obsidian_review_queue(package: dict[str, Any]) -> str:
    items = sorted(
        [
            item
            for item in package.get("timeline", [])
            if item.get("needs_human_review", True) or item.get("review_status") == "needs_revision"
        ],
        key=lambda item: (-_quality_score(_quality_issues(item)), float(item.get("start", 0))),
    )
    lines = [
        "---",
        "type: lecture-review-queue",
        f'title: "{package.get("title", "Lecture Course")} - 人工复核队列"',
        "tags: [lecture-video, review]",
        f'created: "{now_iso()}"',
        "---",
        "",
        f"# 人工复核队列：{package.get('title', 'Lecture Course')}",
        "",
        f"- 待处理片段：{len(items)}",
        "- 排序：缺口风险优先，其次按时间顺序。",
        "",
    ]
    if not items:
        lines.append("当前没有待复核片段。")
    for index, item in enumerate(items, start=1):
        lines.extend(_obsidian_timeline_item(index, item, include_review=True))
    return "\n".join(lines).rstrip() + "\n"


def render_lecture_quality_report_markdown(package: dict[str, Any]) -> str:
    audit = package.get("quality_audit") if isinstance(package.get("quality_audit"), dict) else {}
    if not audit:
        audit = _quality_audit(package.get("timeline", []) if isinstance(package.get("timeline"), list) else [])
    summary = audit.get("summary") if isinstance(audit.get("summary"), dict) else {}
    priority_items = audit.get("priority_items") if isinstance(audit.get("priority_items"), list) else []
    lines = [
        "---",
        "type: lecture-quality-report",
        f'title: "{package.get("title", "Lecture Course")} - 缺口审计"',
        "tags: [lecture-video, quality-audit, review]",
        f'created: "{now_iso()}"',
        "---",
        "",
        f"# 缺口审计：{package.get('title', 'Lecture Course')}",
        "",
        "这个报告不是摘要，也不是自动判定正确性。它只把最可能漏信息、误识别、需要保留图片的片段排到前面，作为人工复核优先级。",
        "",
        "## 总览",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    gap_audit = package.get("time_gap_audit") if isinstance(package.get("time_gap_audit"), dict) else {}
    gap_summary = gap_audit.get("summary") if isinstance(gap_audit.get("summary"), dict) else {}
    gaps = gap_audit.get("gaps") if isinstance(gap_audit.get("gaps"), list) else []
    lines.extend(["", "## 时间覆盖盲区", ""])
    if not gaps:
        lines.append("当前没有超过阈值的时间空白段。")
    else:
        lines.extend(
            [
                f"- 覆盖率：{gap_summary.get('coverage_percent', 0)}%",
                f"- 空白段：{gap_summary.get('gap_count', len(gaps))}",
                f"- 未覆盖秒数：{gap_summary.get('uncovered_seconds', 0)}",
                "",
            ]
        )
        for gap in gaps[:20]:
            start = format_timestamp(float(gap.get("start", 0)))
            end = format_timestamp(float(gap.get("end", 0)))
            lines.append(
                f"- `{gap.get('video_key', 'unknown')}` {start} - {end}，{gap.get('duration_seconds', 0)} 秒，{gap.get('position', 'gap')}"
            )
        if len(gaps) > 20:
            lines.append(f"- 还有 {len(gaps) - 20} 个空白段未列出。")
    lines.extend(["", "## 优先复核片段", ""])
    if not priority_items:
        lines.append("当前没有明显缺口。")
    for item in priority_items:
        start = format_timestamp(float(item.get("start", 0)))
        end = format_timestamp(float(item.get("end", 0)))
        lines.extend(
            [
                f"### {item.get('index', '?')}. {start} - {end}",
                "",
                f"- 风险分：{item.get('score', 0)}",
                f"- 问题：{', '.join(item.get('issues') or [])}",
                f"- 材料类型：{', '.join(item.get('material_types') or [])}",
                f"- 来源片段：{', '.join(item.get('source_segment_ids') or [])}",
                "",
                "口语/字幕：",
                "",
                str(item.get("transcript") or "（缺失或未命中）"),
                "",
                "画面文字 / OCR / 视觉观察：",
                "",
                str(item.get("visual_text") or "（缺失或未命中）"),
                "",
            ]
        )
        frames = item.get("frame_paths") or []
        if frames:
            lines.extend(["关键帧：", ""])
            for frame in frames:
                lines.extend([f"- `{frame}`"])
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_obsidian_visual_assets(package: dict[str, Any]) -> str:
    items = [item for item in package.get("timeline", []) if item.get("frame_paths")]
    lines = [
        "---",
        "type: lecture-visual-assets",
        f'title: "{package.get("title", "Lecture Course")} - 视觉资产索引"',
        "tags: [lecture-video, visual-assets]",
        f'created: "{now_iso()}"',
        "---",
        "",
        f"# 视觉资产索引：{package.get('title', 'Lecture Course')}",
        "",
    ]
    if not items:
        lines.append("当前没有关键帧。")
    for index, item in enumerate(items, start=1):
        start = format_timestamp(float(item.get("start", 0)))
        end = format_timestamp(float(item.get("end", 0)))
        human_review = item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
        keep_images = bool(human_review.get("keep_images"))
        lines.extend(
            [
                f"## {index}. {start} - {end}",
                "",
                f"- 审核状态：`{item.get('review_status', 'pending')}`",
                f"- 必须保留图片：{keep_images}",
                f"- 材料类型：{', '.join(item.get('material_types') or [])}",
                "",
            ]
        )
        for frame in item.get("frame_paths") or []:
            lines.extend([_markdown_image(str(frame)), "", f"`{frame}`", ""])
        visual_text = str(item.get("visual_text") or "").strip()
        if visual_text:
            lines.extend(["### OCR / 视觉观察", "", visual_text, ""])
        structured = _structured_visual_markdown(item)
        if structured:
            lines.extend(["### 结构化视觉", "", structured, ""])
    return "\n".join(lines).rstrip() + "\n"


def render_obsidian_structured_materials(package: dict[str, Any]) -> str:
    """Render formula/table/code/keep-image items as a focused Obsidian index."""
    items = [
        item
        for item in package.get("timeline", [])
        if _is_structured_material_item(item)
    ]
    lines = [
        "---",
        "type: lecture-structured-materials",
        f'title: "{package.get("title", "Lecture Course")} - 结构材料索引"',
        "tags: [lecture-video, structured-materials, formula, table, code]",
        f'created: "{now_iso()}"',
        "---",
        "",
        f"# 结构材料索引：{package.get('title', 'Lecture Course')}",
        "",
        "这个索引集中放置公式、表格、代码、结构化视觉和必须保留图片的片段。能可靠降维成文字的内容保留文字；必须看图或表格的内容保留关键帧。",
        "",
        f"- 条目数：{len(items)}",
        "",
    ]
    if not items:
        lines.append("当前没有命中的结构材料片段。")
        return "\n".join(lines).rstrip() + "\n"

    for index, item in enumerate(items, start=1):
        start = format_timestamp(float(item.get("start", 0)))
        end = format_timestamp(float(item.get("end", 0)))
        material_types = item.get("material_types") if isinstance(item.get("material_types"), list) else []
        review = item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
        retention = item.get("visual_retention") if isinstance(item.get("visual_retention"), dict) else {}
        lines.extend(
            [
                f"## {index}. {start} - {end}",
                "",
                f"- 材料类型：{', '.join(str(value) for value in material_types) or 'unknown'}",
                f"- 审核状态：`{item.get('review_status', 'pending')}`",
                f"- 必须保留图片：{bool(review.get('keep_images') or retention.get('keep_image_required'))}",
                f"- 来源片段：{', '.join(item.get('source_segment_ids') or [])}",
                "",
            ]
        )
        transcript = str(item.get("transcript") or "").strip()
        if transcript:
            lines.extend(["### 口语/字幕", "", transcript, ""])
        visual_text = str(item.get("visual_text") or "").strip()
        if visual_text:
            lines.extend(["### OCR / 视觉观察", "", visual_text, ""])
        structured = _structured_visual_markdown(item)
        if structured:
            lines.extend(["### 结构化内容", "", structured, ""])
        frames = item.get("frame_paths") if isinstance(item.get("frame_paths"), list) else []
        if frames:
            lines.extend(["### 必要图像 / 关键帧", ""])
            for frame in frames:
                lines.extend([_markdown_image(str(frame)), "", f"`{frame}`", ""])
        issues = _quality_issues(item)
        if issues:
            lines.extend(["### 复核提示", "", f"- {', '.join(_quality_issue_label(issue) for issue in issues)}", ""])
    return "\n".join(lines).rstrip() + "\n"


def render_obsidian_evidence_map(package: dict[str, Any], source_artifact_index: dict[str, Any]) -> str:
    """Render a per-timeline traceability map back to frames and source artifacts."""
    timeline = package.get("timeline") if isinstance(package.get("timeline"), list) else []
    items = [item for item in timeline if isinstance(item, dict)]
    artifacts_by_video = _source_artifacts_by_video(source_artifact_index)
    lines = [
        "---",
        "type: lecture-evidence-map",
        f'title: "{package.get("title", "Lecture Course")} - 逐片段证据地图"',
        "tags: [lecture-video, evidence-map, traceability]",
        f'created: "{now_iso()}"',
        "---",
        "",
        f"# 逐片段证据地图：{package.get('title', 'Lecture Course')}",
        "",
        "这个页面按时间线片段组织证据链：口语、屏幕文字、关键帧、触发信号、来源片段和原始抽取物。它不是摘要；当笔记、结构化结果或复习卡可疑时，优先从这里回到真实视频抽取证据。",
        "",
        f"- 片段数：{len(items)}",
        f"- 有关键帧片段：{sum(1 for item in items if item.get('frame_paths'))}",
        f"- 有可回溯原始抽取物的视频源：{len(artifacts_by_video)}",
        "",
    ]
    if not items:
        lines.append("当前没有时间线片段。")
        return "\n".join(lines).rstrip() + "\n"

    for index, item in enumerate(items, start=1):
        video_key = str(item.get("video_key") or "unknown")
        start = format_timestamp(float(item.get("start", 0)))
        end = format_timestamp(float(item.get("end", 0)))
        lines.extend(
            [
                f"## {index}. {video_key} / {start} - {end}",
                "",
                f"- 来源片段：{', '.join(item.get('source_segment_ids') or [])}",
                f"- 触发信号：{', '.join(item.get('signals') or []) or 'unknown'}",
                f"- 材料类型：{', '.join(str(value) for value in item.get('material_types') or []) or 'unknown'}",
                f"- 审核状态：`{item.get('review_status', 'pending')}`",
                f"- 需要人工复核：{bool(item.get('needs_human_review', True))}",
                "",
                "### 信息层",
                "",
                f"- 口语/字幕：{_short_text(item.get('transcript'))}",
                f"- 屏幕文字/OCR：{_short_text(item.get('visual_text'))}",
                f"- 结构化视觉：{len(item.get('structured_visual') if isinstance(item.get('structured_visual'), list) else [])} 条",
                "",
            ]
        )
        frames = item.get("frame_paths") if isinstance(item.get("frame_paths"), list) else []
        if frames:
            lines.extend(["### 关键帧", ""])
            for frame in frames:
                lines.extend([_markdown_image(str(frame)), "", f"`{frame}`", ""])
        issues = _quality_issues(item)
        if issues:
            lines.extend(["### 复核提示", "", f"- {', '.join(_quality_issue_label(issue) for issue in issues)}", ""])
        artifacts = artifacts_by_video.get(video_key) or []
        if artifacts:
            lines.extend(["### 可回查原始抽取物", ""])
            for artifact in artifacts:
                location = str(artifact.get("copied_path") or artifact.get("path") or "").strip()
                lines.append(
                    f"- `{artifact.get('tool', '')}` / {artifact.get('label') or artifact.get('key')}: `{location}`"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _is_structured_material_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    material_types = {str(value) for value in item.get("material_types") or []}
    if material_types & {"formula", "table", "code"}:
        return True
    if item.get("structured_visual"):
        return True
    review = item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
    if review.get("keep_images"):
        return True
    retention = item.get("visual_retention") if isinstance(item.get("visual_retention"), dict) else {}
    return bool(retention.get("keep_image_required"))


def _source_artifacts_by_video(source_artifact_index: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    artifacts = source_artifact_index.get("artifacts") if isinstance(source_artifact_index.get("artifacts"), list) else []
    for row in artifacts:
        if not isinstance(row, dict) or not row.get("available"):
            continue
        keys = [str(row.get("video_id") or "").strip(), str(row.get("title") or "").strip()]
        for key in keys:
            if key:
                result.setdefault(key, []).append(row)
    return result


def _short_text(value: Any, *, limit: int = 140) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return "（缺失或未命中）"
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def render_lecture_package_markdown(package: dict[str, Any]) -> str:
    lines = [
        "---",
        "type: note",
        f'title: "{package.get("title", "lecture knowledge package")}"',
        "tags: [ai-research, lecture-video, knowledge-package]",
        "status: candidate",
        f'created: "{package.get("created_at", now_iso())}"',
        "---",
        "",
        f"# {package.get('title', 'Lecture Knowledge Package')}",
        "",
        "> 这是全量提取工作台，不是摘要。任何缺帧、缺转录、OCR 不确定或视觉信息未解释的片段，都默认保留在人工复核队列里。",
        "",
        "## 覆盖率审计",
        "",
    ]
    coverage = package.get("coverage", {})
    for key in [
        "timeline_items",
        "items_with_transcript",
        "items_with_visual_text",
        "items_with_frames",
        "items_needing_review",
        "items_reviewed",
        "items_needing_revision",
        "items_with_human_notes",
        "items_with_corrected_transcript",
        "items_with_corrected_visual_text",
        "items_with_structured_visual",
        "structured_visual_entries",
        "possible_code_items",
        "possible_formula_items",
        "possible_table_items",
    ]:
        lines.append(f"- {key}: {coverage.get(key, 0)}")

    lines.extend(["", "## 来源", ""])
    for source in package.get("sources", []):
        lines.append(_source_markdown(source))

    lines.extend(["", "## 全量时间线", ""])
    for index, item in enumerate(package.get("timeline", []), start=1):
        start = format_timestamp(float(item.get("start", 0)))
        end = format_timestamp(float(item.get("end", 0)))
        material = ", ".join(item.get("material_types") or ["unknown"])
        signals = ", ".join(item.get("signals") or [])
        lines.extend(
            [
                f"### {index}. {start} - {end}",
                "",
                f"- 来源片段：{', '.join(item.get('source_segment_ids') or [])}",
                f"- 材料类型：{material}",
                f"- 触发信号：{signals}",
                f"- 审核状态：{item.get('review_status', 'pending')}",
                f"- 需要人工复核：{item.get('needs_human_review', True)}",
                "",
            ]
        )
        evidence_lines = _evidence_markdown(item)
        if evidence_lines:
            lines.extend(["#### 证据链", "", *evidence_lines, ""])
        lines.extend(
            [
                "#### 口语/字幕",
                "",
                item.get("transcript") or "（缺失或未命中）",
                "",
                "#### 画面文字 / OCR / 视觉观察",
                "",
                item.get("visual_text") or "（缺失或未命中）",
                "",
            ]
        )
        structured = _structured_visual_markdown(item)
        if structured:
            lines.extend(["#### 结构化视觉", "", structured, ""])
        lines.extend(
            [
                "#### 关键帧",
                "",
            ]
        )
        frames = item.get("frame_paths") or []
        if frames:
            lines.extend(f"- `{frame}`" for frame in frames)
        else:
            lines.append("- （无关键帧）")
        lines.extend(
            [
                "",
                "#### 人工复核",
                "",
                *_human_review_markdown(item.get("human_review") if isinstance(item.get("human_review"), dict) else {}),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_lecture_review_html(package: dict[str, Any]) -> str:
    review_timeline = _timeline_for_review(package)
    review_package = dict(package)
    review_package["timeline"] = review_timeline
    review_coverage = dict(package.get("coverage") or {}) if isinstance(package.get("coverage"), dict) else {}
    review_coverage["items_with_transcript"] = sum(
        1
        for item in review_timeline
        if item.get("transcript")
        and item.get("review_transcript_source") != "canonical_alignment_missing"
    )
    review_package["coverage"] = review_coverage
    data = json.dumps(review_package, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(str(package.get("title", "Lecture Review")))
    coverage_cards = _coverage_cards(review_coverage)
    workflow_cards = _workflow_cards(package)
    header_task_console = _header_task_console_link(package)
    review_workbench = _review_workbench_html(package)
    timeline_html = "\n".join(_timeline_card(index, item) for index, item in enumerate(review_timeline, start=1))
    video_review_panel = _review_video_panel(package)
    sources_html = "\n".join(_source_row(source) for source in package.get("sources", []))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #202124;
      --muted: #626866;
      --line: #d8ddd8;
      --accent: #276f86;
      --warn: #9a4d00;
      --ok: #2f6f3e;
      --chip: #eef3f2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      line-height: 1.5;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 5;
      border-bottom: 1px solid var(--line);
      background: rgba(247, 247, 244, 0.96);
      backdrop-filter: blur(8px);
    }}
    .topbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      max-width: 1280px;
      margin: 0 auto;
      padding: 16px 20px;
    }}
    h1 {{
      margin: 0;
      font-size: 22px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    button, .button {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--ink);
      cursor: pointer;
      font: inherit;
      padding: 8px 11px;
    }}
    button.primary, .button.primary {{
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
      text-decoration: none;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 18px 20px 40px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .metric strong {{
      display: block;
      font-size: 22px;
    }}
    .metric span {{
      color: var(--muted);
      font-size: 13px;
    }}
    .actions {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .action {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .action strong {{
      display: block;
      margin-bottom: 4px;
    }}
    .action code {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
      margin-top: 4px;
    }}
    .action a {{
      color: var(--accent);
      font-weight: 600;
      text-decoration: none;
    }}
    .action button {{
      margin-top: 8px;
      padding: 6px 9px;
      font-size: 13px;
    }}
    .action .hint {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 12px;
    }}
    .gap-list {{
      margin: 8px 0 0;
      padding-left: 18px;
      color: var(--warn);
      font-size: 13px;
    }}
    .review-workbench {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(260px, 420px);
      gap: 12px;
      margin-bottom: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px;
    }}
    .review-workbench h2 {{
      margin: 0 0 8px;
      font-size: 16px;
    }}
    .review-paths {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 8px;
      margin-bottom: 10px;
    }}
    .review-path {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fbfbf9;
    }}
    .review-path span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
    }}
    .review-path code {{
      display: block;
      overflow-wrap: anywhere;
      font-size: 12px;
    }}
    .review-path a {{
      color: var(--accent);
      text-decoration: none;
    }}
    .filter-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 8px 0 10px;
    }}
    .filter-bar button.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}
    .review-draft {{
      min-height: 220px;
      font-family: "Cascadia Code", Consolas, monospace;
      font-size: 12px;
    }}
    .writeback-actions {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      margin-top: 8px;
    }}
    .draft-status {{
      display: block;
      width: 100%;
      color: var(--muted);
      font-size: 12px;
    }}
    .draft-status.saved {{ color: var(--warn); }}
    .draft-status.applied {{ color: var(--ok); font-weight: 600; }}
    .draft-status.error {{ color: #a12622; font-weight: 600; }}
    .transcript-candidate {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      margin: 8px 0;
      background: #fbfbf9;
    }}
    .transcript-candidate strong {{ display: block; }}
    .transcript-candidate code {{ overflow-wrap: anywhere; }}
    .field-hint {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(280px, var(--review-sidebar-width, 420px)) minmax(0, 1fr);
      gap: 16px;
    }}
    .layout.video-wide {{
      grid-template-columns: minmax(520px, var(--review-sidebar-width, 640px)) minmax(0, 1fr);
    }}
    aside {{
      align-self: start;
      position: sticky;
      top: 74px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px;
    }}
    .filter {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      margin-bottom: 10px;
    }}
    .source {{
      border-top: 1px solid var(--line);
      padding: 9px 0;
      font-size: 13px;
      word-break: break-word;
    }}
    .source code {{
      display: block;
      color: var(--muted);
      margin-top: 3px;
    }}
    .source-artifacts {{
      margin: 8px 0 0;
      padding-left: 18px;
    }}
    .source-artifacts li {{
      margin: 4px 0;
    }}
    .review-video {{
      margin-bottom: 12px;
    }}
    .review-video video {{
      width: 100%;
      height: var(--review-video-height, 220px);
      max-height: 70vh;
      display: block;
      background: #000;
      border-radius: 8px;
      margin: 8px 0;
      object-fit: contain;
      resize: vertical;
      overflow: auto;
    }}
    .review-video .video-status {{
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .review-video .video-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 8px 0;
    }}
    .review-video .video-size-control {{
      display: grid;
      grid-template-columns: auto minmax(120px, 1fr) auto;
      align-items: center;
      gap: 8px;
      margin: 8px 0;
      color: var(--muted);
      font-size: 12px;
    }}
    .review-video .video-size-control + .video-size-control {{
      margin-top: 4px;
    }}
    .review-video input[type="range"] {{
      width: 100%;
    }}
    .jump-time {{
      border: 0;
      background: transparent;
      color: var(--accent);
      padding: 0;
      font-weight: 650;
      text-align: left;
    }}
    .item.active {{
      outline: 3px solid rgba(39, 111, 134, 0.25);
      box-shadow: 0 0 0 4px rgba(39, 111, 134, 0.08);
    }}
    .timeline {{
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .item {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .item-header {{
      display: grid;
      grid-template-columns: minmax(130px, auto) 1fr auto;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #fbfbf9;
    }}
    .time {{
      font-variant-numeric: tabular-nums;
      font-weight: 650;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .chip {{
      border-radius: 999px;
      background: var(--chip);
      color: var(--muted);
      padding: 2px 8px;
      font-size: 12px;
    }}
    .needs-review {{
      color: var(--warn);
      font-weight: 600;
    }}
    .reviewed {{
      color: var(--ok);
    }}
    .item-body {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(240px, 360px);
      gap: 12px;
      padding: 12px;
    }}
    h2, h3 {{
      margin: 0 0 8px;
      font-size: 15px;
    }}
    .section {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      margin-bottom: 10px;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "Cascadia Code", Consolas, monospace;
      font-size: 13px;
    }}
    textarea {{
      width: 100%;
      min-height: 96px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      font: inherit;
    }}
    .frames {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }}
    .frame {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #fafafa;
      word-break: break-word;
    }}
    .frame img {{
      width: 100%;
      max-height: 260px;
      object-fit: contain;
      display: block;
      border-radius: 6px;
      background: #eee;
      margin-bottom: 6px;
    }}
    .checks label {{
      display: block;
      margin: 6px 0;
      font-size: 14px;
    }}
    .review-editor label {{
      display: block;
      margin: 8px 0;
      font-size: 13px;
    }}
    .review-editor textarea {{
      min-height: 70px;
      margin-top: 4px;
    }}
    .review-editor .snippet-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }}
    .quick-statuses {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 8px 0;
    }}
    .quick-statuses button {{
      font-size: 12px;
      padding: 6px 8px;
    }}
    .quick-statuses button.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}
    .quality-list {{
      margin: 6px 0 0;
      padding-left: 18px;
      color: var(--warn);
      font-size: 13px;
    }}
    @media (max-width: 900px) {{
      .layout, .item-body, .review-workbench {{ grid-template-columns: 1fr; }}
      aside {{ position: static; }}
      .item-header {{ grid-template-columns: 1fr; }}
    }}
  </style>
  {review_export_script(package)}
</head>
<body>
  <header>
    <div class="topbar">
      <h1>{title}</h1>
      <div>
        {header_task_console}
        <button type="button" onclick="toggleReviewed()">隐藏已确认</button>
        <button type="button" class="primary" onclick="downloadReview()">导出审核 JSON</button>
      </div>
    </div>
  </header>
  <main>
    <section class="summary">{coverage_cards}</section>
    <section class="actions">{workflow_cards}</section>
    {review_workbench}
    <section class="layout" id="review-layout">
      <aside>
        {video_review_panel}
        <input class="filter" id="filter" placeholder="筛选时间线 / OCR / 转写" oninput="applyFilter()">
        <h2>来源</h2>
        {sources_html}
      </aside>
      <section class="timeline" id="timeline">
        {timeline_html}
      </section>
    </section>
  </main>
  <script id="package-data" type="application/json">{data}</script>
  <script>
    const packageData = JSON.parse(document.getElementById('package-data').textContent);
    let hideReviewed = false;
    let activeReviewFilter = 'all';
    let restoringReviewDraft = false;
    const reviewApi = window.VKP_REVIEW_API || null;
    const REVIEW_DRAFT_SCHEMA = 'vkp_review_draft.v2';
    const REVIEW_DRAFT_PREFIX = 'vkpReviewDraft.v2:';

    function toggleReviewed() {{
      hideReviewed = !hideReviewed;
      applyFilter();
    }}

    function setReviewFilter(filter) {{
      activeReviewFilter = filter;
      document.querySelectorAll('[data-review-filter]').forEach((button) => {{
        button.classList.toggle('active', button.dataset.reviewFilter === filter);
      }});
      applyFilter();
    }}

    function applyFilter() {{
      const query = document.getElementById('filter').value.trim().toLowerCase();
      for (const item of document.querySelectorAll('.item')) {{
        const matches = !query || item.textContent.toLowerCase().includes(query);
        const reviewed = item.querySelector('.review-check')?.checked;
        const matchesKind = matchesReviewFilter(item, activeReviewFilter);
        item.style.display = matches && matchesKind && !(hideReviewed && reviewed) ? '' : 'none';
      }}
    }}

    function matchesReviewFilter(item, filter) {{
      if (filter === 'all') return true;
      if (filter === 'needs-human-review') return item.dataset.needsHumanReview === 'true';
      if (filter === 'missing-visual-text') return (item.dataset.qualityIssues || '').split(',').includes('missing_visual_text');
      if (filter === 'screen-text') return ['missing_visual_text', 'ocr_text_empty', 'screen_text_low_confidence', 'low_ocr_confidence'].some((issue) => (item.dataset.qualityIssues || '').split(',').includes(issue));
      if (filter === 'low-confidence') return item.dataset.lowConfidence === 'true';
      if (filter === 'keep-image') return item.dataset.keepImage === 'true' || item.querySelector('.keep-images')?.checked;
      if (filter === 'human-key-point') return item.querySelector('.human-key-point-confirmed')?.checked;
      if (filter === 'corrected') return item.dataset.corrected === 'true' || hasCorrections(item);
      return true;
    }}

    function hasCorrections(item) {{
      return Boolean(
        item.querySelector('.corrected-transcript')?.value.trim() ||
        item.querySelector('.semantic-corrected-text')?.value.trim() ||
        item.querySelector('.human-key-point-text')?.value.trim() ||
        item.querySelector('.corrected-visual-text')?.value.trim() ||
        item.querySelector('.corrected-visual-understanding')?.value.trim() ||
        item.querySelector('.corrected-temporal-visual-understanding')?.value.trim()
      );
    }}

    function buildReviewPayload(selectedOnly = true) {{
      const reviews = [];
      for (const item of document.querySelectorAll('.item')) {{
        if (selectedOnly && !item.querySelector('.include-review-row')?.checked) continue;
        reviews.push(buildReviewSnippet(item));
      }}
      return {{
        schema: 'lecture_review_notes.v1',
        package_title: packageData.title || '',
        exported_at: new Date().toISOString(),
        reviews: reviews
      }};
    }}

    function buildReviewSnippet(item) {{
      const comment = item.querySelector('.review-comment')?.value.trim() || '';
      const correctedTranscript = item.querySelector('.corrected-transcript')?.value.trim() || '';
      const semanticCorrections = Array.from(item.querySelectorAll('.transcript-candidate')).map((candidate) => ({{
        candidate_id: candidate.dataset.candidateId || '',
        status: 'corrected_transcript',
        original_text: candidate.dataset.originalText || '',
        corrected_text: candidate.querySelector('.semantic-corrected-text')?.value.trim() || '',
        evidence_ids: (candidate.dataset.evidenceIds || '').split('|').filter(Boolean),
        comment: comment,
        human_confirmed: true
      }})).filter((row) => row.candidate_id && row.corrected_text && row.corrected_text !== row.original_text);
      const correctedVisualText = item.querySelector('.corrected-visual-text')?.value.trim() || '';
      const correctedVisualUnderstanding = parseCorrectionObject(item.querySelector('.corrected-visual-understanding')?.value.trim() || '');
      const correctedTemporalUnderstanding = parseCorrectionObject(item.querySelector('.corrected-temporal-visual-understanding')?.value.trim() || '');
      const keepImage = item.querySelector('.keep-images')?.checked || false;
      const humanKeyPointConfirmed = item.querySelector('.human-key-point-confirmed')?.checked || false;
      const humanKeyPointText = item.querySelector('.human-key-point-text')?.value.trim() || '';
      const humanKeyPointAliases = (item.querySelector('.human-key-point-aliases')?.value || '')
        .split('|')
        .map((value) => value.trim())
        .filter(Boolean);
      const forcedStatus = item.dataset.reviewStatusSet || '';
      const tags = [];
      if (item.querySelector('.asr-ocr-error')?.checked) tags.push('asr_ocr_error');
      if (item.querySelector('.missing-info')?.checked) tags.push('missing_info');
      if (keepImage) tags.push('keep_image');
      if (humanKeyPointConfirmed) tags.push('human_key_point');
      let status = forcedStatus || (item.querySelector('.review-check')?.checked ? 'accepted' : 'needs_human_review');
      if (!forcedStatus && correctedVisualText) status = 'corrected_visual_text';
      if (!forcedStatus && Object.keys(correctedVisualUnderstanding).length) status = 'corrected_visual_understanding';
      if (!forcedStatus && Object.keys(correctedTemporalUnderstanding).length) status = 'corrected_temporal_visual_understanding';
      if (!forcedStatus && (correctedTranscript || semanticCorrections.length)) status = 'corrected_transcript';
      if (!forcedStatus && keepImage && status === 'accepted') status = 'keep_image';
      if (!forcedStatus && (item.querySelector('.asr-ocr-error')?.checked || item.querySelector('.missing-info')?.checked) && status === 'needs_human_review') status = 'needs_fix';
      const snippet = {{
        timeline_index: Number(item.dataset.index),
        time_range: item.dataset.time,
        route: item.dataset.visualRoute || '',
        status: status,
        tags: tags,
        comment: comment,
        evidence_frame_paths: item.dataset.evidencePaths ? item.dataset.evidencePaths.split('|').filter(Boolean) : [],
        reviewed_at: new Date().toISOString()
      }};
      if (humanKeyPointConfirmed) {{
        snippet.human_key_point_confirmed = true;
        snippet.human_key_point_text = humanKeyPointText;
        snippet.human_key_point_aliases = humanKeyPointAliases;
      }}
      if (correctedTranscript) snippet.corrected_transcript = correctedTranscript;
      if (semanticCorrections.length) snippet.transcript_semantic_corrections = semanticCorrections;
      if (correctedVisualText) snippet.corrected_visual_text = correctedVisualText;
      if (Object.keys(correctedVisualUnderstanding).length) snippet.corrected_visual_understanding = correctedVisualUnderstanding;
      if (Object.keys(correctedTemporalUnderstanding).length) snippet.corrected_temporal_visual_understanding = correctedTemporalUnderstanding;
      return snippet;
    }}

    function parseCorrectionObject(text) {{
      if (!text) return {{}};
      try {{
        const value = JSON.parse(text);
        if (value && typeof value === 'object' && !Array.isArray(value)) return value;
        return {{ value: value }};
      }} catch (error) {{
        return {{ human_summary: text }};
      }}
    }}

    function setQuickReviewStatus(button, status) {{
      const item = button.closest('.item');
      if (!item) return;
      item.dataset.reviewStatusSet = status;
      const include = item.querySelector('.include-review-row');
      const checked = item.querySelector('.review-check');
      const keep = item.querySelector('.keep-images');
      if (include) include.checked = true;
      if (checked) checked.checked = ['accepted', 'accepted_known_gap', 'keep_image', 'corrected_visual_text', 'corrected_transcript'].includes(status);
      if (keep && status === 'keep_image') keep.checked = true;
      item.querySelectorAll('[data-quick-status]').forEach((node) => {{
        node.classList.toggle('active', node === button);
      }});
      refreshReviewDraft();
    }}

    function refreshReviewDraft() {{
      const area = document.getElementById('review-json-draft');
      if (!area) return;
      area.value = JSON.stringify(buildReviewPayload(true), null, 2);
      applyFilter();
      if (!restoringReviewDraft) persistReviewDraft();
    }}

    async function copyReviewSnippet(button) {{
      const item = button.closest('.item');
      const text = JSON.stringify(buildReviewSnippet(item), null, 2);
      try {{
        await navigator.clipboard.writeText(text);
        const old = button.textContent;
        button.textContent = '已复制';
        setTimeout(() => {{ button.textContent = old; }}, 1200);
      }} catch (error) {{
        button.textContent = '复制失败';
      }}
    }}

    function downloadReview() {{
      const payload = buildReviewPayload(true);
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = vkpReviewExportName('综合审核', payload);
      link.click();
      URL.revokeObjectURL(url);
      setReviewDraftStatus('审核 JSON 已下载；浏览器草稿仍保留，尚未自动写回 VKP。', 'saved');
    }}

    function reviewDraftStorageKey() {{
      const identity = `${{location.pathname}}|${{packageData.title || ''}}|${{document.querySelectorAll('.item').length}}`;
      return REVIEW_DRAFT_PREFIX + identity;
    }}

    function captureReviewDraft() {{
      const items = {{}};
      document.querySelectorAll('.item').forEach((item) => {{
        const semantic = {{}};
        item.querySelectorAll('.transcript-candidate').forEach((candidate) => {{
          semantic[candidate.dataset.candidateId || ''] = candidate.querySelector('.semantic-corrected-text')?.value || '';
        }});
        items[item.dataset.index] = {{
          include: item.querySelector('.include-review-row')?.checked || false,
          reviewed: item.querySelector('.review-check')?.checked || false,
          keepImage: item.querySelector('.keep-images')?.checked || false,
          asrOcrError: item.querySelector('.asr-ocr-error')?.checked || false,
          missingInfo: item.querySelector('.missing-info')?.checked || false,
          humanKeyPointConfirmed: item.querySelector('.human-key-point-confirmed')?.checked || false,
          humanKeyPointText: item.querySelector('.human-key-point-text')?.value || '',
          humanKeyPointAliases: item.querySelector('.human-key-point-aliases')?.value || '',
          comment: item.querySelector('.review-comment')?.value || '',
          correctedTranscript: item.querySelector('.corrected-transcript')?.value || '',
          correctedVisualText: item.querySelector('.corrected-visual-text')?.value || '',
          correctedVisualUnderstanding: item.querySelector('.corrected-visual-understanding')?.value || '',
          correctedTemporalVisualUnderstanding: item.querySelector('.corrected-temporal-visual-understanding')?.value || '',
          forcedStatus: item.dataset.reviewStatusSet || '',
          semantic: semantic
        }};
      }});
      return {{ schema: REVIEW_DRAFT_SCHEMA, saved_at: new Date().toISOString(), items: items }};
    }}

    function persistReviewDraft() {{
      try {{
        const draft = captureReviewDraft();
        localStorage.setItem(reviewDraftStorageKey(), JSON.stringify(draft));
        setReviewDraftStatus(`草稿已自动保存于本浏览器（${{new Date().toLocaleTimeString()}}），尚未写回 VKP。`, 'saved');
      }} catch (error) {{
        setReviewDraftStatus(`草稿自动保存失败：${{error.message || error}}`, 'error');
      }}
    }}

    function restoreReviewDraft() {{
      let draft = null;
      try {{ draft = JSON.parse(localStorage.getItem(reviewDraftStorageKey()) || 'null'); }} catch (error) {{}}
      if (!draft || draft.schema !== REVIEW_DRAFT_SCHEMA || !draft.items) {{
        setReviewDraftStatus('尚无浏览器草稿；填写后会自动保存，但不会自动写回 VKP。', '');
        return;
      }}
      restoringReviewDraft = true;
      document.querySelectorAll('.item').forEach((item) => {{
        const state = draft.items[item.dataset.index];
        if (!state) return;
        const setChecked = (selector, value) => {{ const node = item.querySelector(selector); if (node) node.checked = Boolean(value); }};
        const setValue = (selector, value) => {{ const node = item.querySelector(selector); if (node) node.value = value || ''; }};
        setChecked('.include-review-row', state.include);
        setChecked('.review-check', state.reviewed);
        setChecked('.keep-images', state.keepImage);
        setChecked('.asr-ocr-error', state.asrOcrError);
        setChecked('.missing-info', state.missingInfo);
        setChecked('.human-key-point-confirmed', state.humanKeyPointConfirmed);
        setValue('.human-key-point-text', state.humanKeyPointText);
        setValue('.human-key-point-aliases', state.humanKeyPointAliases);
        setValue('.review-comment', state.comment);
        setValue('.corrected-transcript', state.correctedTranscript);
        setValue('.corrected-visual-text', state.correctedVisualText);
        setValue('.corrected-visual-understanding', state.correctedVisualUnderstanding);
        setValue('.corrected-temporal-visual-understanding', state.correctedTemporalVisualUnderstanding);
        item.dataset.reviewStatusSet = state.forcedStatus || '';
        item.querySelectorAll('.transcript-candidate').forEach((candidate) => {{
          const input = candidate.querySelector('.semantic-corrected-text');
          if (input && state.semantic) input.value = state.semantic[candidate.dataset.candidateId || ''] || input.value;
        }});
      }});
      restoringReviewDraft = false;
      setReviewDraftStatus(`已恢复浏览器草稿（${{new Date(draft.saved_at).toLocaleString()}}），尚未写回 VKP。`, 'saved');
    }}

    function clearReviewDraft() {{
      if (!confirm('仅清除本浏览器中的审核草稿？已经正式写回 VKP 的内容不会被撤销。')) return;
      try {{ localStorage.removeItem(reviewDraftStorageKey()); }} catch (error) {{}}
      setReviewDraftStatus('浏览器草稿已清除；页面现有输入尚未改变。', '');
    }}

    function setReviewDraftStatus(text, kind) {{
      const status = document.getElementById('review-draft-status');
      if (!status) return;
      status.textContent = text;
      status.className = `draft-status ${{kind || ''}}`;
    }}

    function includeEditedItem(input) {{
      const item = input.closest('.item');
      if (!item || input.classList.contains('include-review-row')) return;
      const include = item.querySelector('.include-review-row');
      if (include) include.checked = true;
    }}

    async function initReviewWriteback() {{
      const button = document.getElementById('save-to-vkp');
      if (!button || !reviewApi) {{
        setReviewDraftStatus('当前是静态审核页：草稿会自动保存；要一键正式写回，请用页面所示命令启动本地审核服务。', 'saved');
        return;
      }}
      try {{
        const response = await fetch(reviewApi.status_url, {{ cache: 'no-store' }});
        const status = await response.json();
        if (!response.ok || !status.ok) throw new Error(status.error || `HTTP ${{response.status}}`);
        reviewApi.bundle_revision = status.bundle_revision;
        button.disabled = false;
        setReviewDraftStatus('本地写回服务已连接；草稿仍只在浏览器，点击“保存到 VKP”才会正式应用。', 'saved');
      }} catch (error) {{
        setReviewDraftStatus(`本地写回服务不可用：${{error.message || error}}`, 'error');
      }}
    }}

    async function saveReviewToVkp() {{
      const button = document.getElementById('save-to-vkp');
      if (!reviewApi || !button) return;
      const payload = buildReviewPayload(true);
      if (!payload.reviews.length) {{
        setReviewDraftStatus('没有选入审核 JSON 的条目，未执行写回。', 'error');
        return;
      }}
      button.disabled = true;
      setReviewDraftStatus('正在进行本地校验并写回 VKP……', 'saved');
      try {{
        const response = await fetch(reviewApi.apply_url, {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json', 'X-VKP-Review-Token': reviewApi.token }},
          body: JSON.stringify({{ bundle_revision: reviewApi.bundle_revision, review_notes: payload }})
        }});
        const result = await response.json();
        if (!response.ok || !result.ok) throw new Error(result.error || result.writeback?.status || `HTTP ${{response.status}}`);
        reviewApi.bundle_revision = result.bundle_revision;
        persistReviewDraft();
        setReviewDraftStatus('已正式写回 VKP，并刷新逐字稿/摘要相关产物；可重新加载页面查看最新状态。', 'applied');
      }} catch (error) {{
        setReviewDraftStatus(`写回失败：${{error.message || error}}`, 'error');
      }} finally {{
        button.disabled = false;
      }}
    }}

    document.querySelectorAll('[data-copy]').forEach((button) => {{
      button.addEventListener('click', async () => {{
        const text = button.getAttribute('data-copy') || '';
        try {{
          await navigator.clipboard.writeText(text);
          const old = button.textContent;
          button.textContent = '已复制';
          setTimeout(() => {{ button.textContent = old; }}, 1200);
        }} catch (error) {{
          button.textContent = '复制失败';
        }}
      }});
    }});
    document.querySelectorAll('[data-review-filter]').forEach((button) => {{
      button.addEventListener('click', () => setReviewFilter(button.dataset.reviewFilter || 'all'));
    }});
    initReviewSidebarWidth();
    initReviewVideoWide();
    initReviewVideoHeight();

    function setReviewSidebarWidth(value) {{
      const width = Math.max(280, Math.min(900, Number(value) || 420));
      document.documentElement.style.setProperty('--review-sidebar-width', `${{width}}px`);
      const label = document.getElementById('review-sidebar-width-label');
      if (label) label.textContent = `${{width}}px`;
      try {{ localStorage.setItem('vkpReviewSidebarWidth', String(width)); }} catch (error) {{}}
    }}

    function initReviewSidebarWidth() {{
      let saved = 420;
      try {{ saved = Number(localStorage.getItem('vkpReviewSidebarWidth') || 420); }} catch (error) {{}}
      setReviewSidebarWidth(saved);
      const slider = document.getElementById('review-sidebar-width-slider');
      if (slider) slider.value = String(Math.max(280, Math.min(900, saved)));
    }}

    function toggleReviewVideoWide() {{
      const layout = document.getElementById('review-layout');
      if (!layout) return;
      const enabled = !layout.classList.contains('video-wide');
      layout.classList.toggle('video-wide', enabled);
      if (enabled) {{
        setReviewSidebarWidth(Math.max(640, Number(localStorage.getItem('vkpReviewSidebarWidth') || 420)));
      }}
      try {{ localStorage.setItem('vkpReviewVideoWide', enabled ? '1' : '0'); }} catch (error) {{}}
    }}

    function initReviewVideoWide() {{
      let enabled = false;
      try {{ enabled = localStorage.getItem('vkpReviewVideoWide') === '1'; }} catch (error) {{}}
      const layout = document.getElementById('review-layout');
      if (layout && enabled) layout.classList.add('video-wide');
    }}

    function setReviewVideoHeight(value) {{
      const height = Math.max(160, Math.min(720, Number(value) || 220));
      document.documentElement.style.setProperty('--review-video-height', `${{height}}px`);
      const label = document.getElementById('review-video-height-label');
      if (label) label.textContent = `${{height}}px`;
      try {{ localStorage.setItem('vkpReviewVideoHeight', String(height)); }} catch (error) {{}}
    }}

    function initReviewVideoHeight() {{
      let saved = 220;
      try {{ saved = Number(localStorage.getItem('vkpReviewVideoHeight') || 220); }} catch (error) {{}}
      setReviewVideoHeight(saved);
      const slider = document.getElementById('review-video-height-slider');
      if (slider) slider.value = String(Math.max(160, Math.min(720, saved)));
    }}

    function loadReviewVideo(input) {{
      const file = input.files && input.files[0];
      if (!file) return;
      const video = document.getElementById('review-video-player');
      if (!video) return;
      video.src = URL.createObjectURL(file);
      video.load();
      setVideoStatus(`已加载本地视频：${{file.name}}。现在可以点击时间戳跳转。`);
    }}

    function setVideoStatus(text) {{
      const status = document.getElementById('review-video-status');
      if (status) status.textContent = text;
    }}

    function jumpToTimelineItem(index, seconds) {{
      const item = document.querySelector(`.item[data-index="${{index}}"]`);
      document.querySelectorAll('.item.active').forEach((node) => node.classList.remove('active'));
      if (item) {{
        item.classList.add('active');
        item.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
      }}
      const video = document.getElementById('review-video-player');
      const start = Math.max(0, Number(seconds || 0));
      if (!video || !video.src) {{
        setVideoStatus('已定位到审核条目，但浏览器还没有加载视频。请先点“选择本地视频文件”。');
        return;
      }}
      video.currentTime = start;
      video.play().catch(() => {{}});
      setVideoStatus(`已跳转到 #${{index}}，时间戳 ${{formatSeconds(start)}}。`);
    }}

    function formatSeconds(value) {{
      const total = Math.max(0, Math.floor(Number(value) || 0));
      const h = String(Math.floor(total / 3600)).padStart(2, '0');
      const m = String(Math.floor((total % 3600) / 60)).padStart(2, '0');
      const s = String(total % 60).padStart(2, '0');
      return `${{h}}:${{m}}:${{s}}`;
    }}

    function jumpToFirstVisibleReviewItem() {{
      const item = Array.from(document.querySelectorAll('.item')).find((node) => node.style.display !== 'none');
      if (!item) return;
      const timeButton = item.querySelector('.jump-time');
      if (timeButton) timeButton.click();
    }}
    document.querySelectorAll('.include-review-row, .review-check, .keep-images, .asr-ocr-error, .missing-info, .human-key-point-confirmed').forEach((input) => {{
      input.addEventListener('change', () => {{ includeEditedItem(input); refreshReviewDraft(); }});
    }});
    document.querySelectorAll('.review-comment, .corrected-transcript, .semantic-corrected-text, .human-key-point-text, .human-key-point-aliases, .corrected-visual-text, .corrected-visual-understanding, .corrected-temporal-visual-understanding').forEach((input) => {{
      input.addEventListener('input', () => {{ includeEditedItem(input); refreshReviewDraft(); }});
    }});
    restoreReviewDraft();
    refreshReviewDraft();
    initReviewWriteback();
  </script>
</body>
</html>
"""


def _load_imported_videos(videos_dir: Path) -> list[dict[str, Any]]:
    videos = []
    for video_dir in sorted(videos_dir.glob("*")):
        metadata_path = video_dir / "metadata.json"
        segments_path = video_dir / "segments.json"
        artifacts_path = video_dir / "source-artifacts.json"
        if not metadata_path.exists() or not segments_path.exists():
            continue
        metadata = read_json(metadata_path)
        segments = read_json(segments_path)
        source_artifacts = read_json(artifacts_path) if artifacts_path.exists() else {}
        if isinstance(metadata, dict) and isinstance(segments, list):
            videos.append(
                {
                    "metadata": metadata,
                    "segments": [s for s in segments if isinstance(s, dict)],
                    "source_artifacts": source_artifacts if isinstance(source_artifacts, dict) else {},
                }
            )
    return videos


def _merged_timeline(videos: list[dict[str, Any]], *, merge_window: float) -> list[dict[str, Any]]:
    rows = []
    for video in videos:
        metadata = video["metadata"]
        for segment in video["segments"]:
            rows.append(_timeline_item(metadata, segment))
    rows.sort(key=lambda item: (item["video_key"], item["midpoint"], item["start"]))

    merged: list[dict[str, Any]] = []
    for item in rows:
        if not merged or not _can_merge(merged[-1], item, merge_window):
            merged.append(item)
            continue
        _merge_item(merged[-1], item)
    return merged


def _timeline_item(metadata: dict[str, Any], segment: dict[str, Any]) -> dict[str, Any]:
    transcript = str(segment.get("transcript_excerpt") or "").strip()
    visual_text = str(segment.get("visual_observation") or "").strip()
    frame_paths = [str(path) for path in segment.get("frame_paths") or [] if str(path)]
    signals = [str(signal) for signal in segment.get("signals") or [] if str(signal)]
    text_for_type = "\n".join(value for value in [transcript, visual_text] if value)
    item = {
        "video_key": str(metadata.get("path") or metadata.get("id") or ""),
        "video_id": str(metadata.get("id") or ""),
        "video_duration_seconds": float(metadata.get("duration_seconds") or 0),
        "start": float(segment.get("start") or 0),
        "end": float(segment.get("end") or segment.get("start") or 0),
        "midpoint": float(segment.get("midpoint") or segment.get("start") or 0),
        "source_segment_ids": [str(segment.get("id") or "")],
        "signals": signals,
        "transcript": transcript,
        "visual_text": visual_text,
        "frame_paths": frame_paths,
        "material_types": _material_types(text_for_type, frame_paths, signals),
        "needs_human_review": bool(segment.get("needs_human_review", True)),
        "uncertainties": [str(segment.get("uncertainty") or "").strip()],
    }
    _refresh_visual_retention(item)
    return item


def _can_merge(left: dict[str, Any], right: dict[str, Any], merge_window: float) -> bool:
    return left["video_key"] == right["video_key"] and abs(float(left["midpoint"]) - float(right["midpoint"])) <= merge_window


def _merge_item(left: dict[str, Any], right: dict[str, Any]) -> None:
    left["start"] = min(float(left["start"]), float(right["start"]))
    left["end"] = max(float(left["end"]), float(right["end"]))
    left["midpoint"] = (float(left["start"]) + float(left["end"])) / 2
    for key in ["source_segment_ids", "signals", "frame_paths", "material_types", "uncertainties"]:
        left[key] = _dedupe([*(left.get(key) or []), *(right.get(key) or [])])
    left["transcript"] = _join_unique(left.get("transcript", ""), right.get("transcript", ""))
    left["visual_text"] = _join_unique(left.get("visual_text", ""), right.get("visual_text", ""))
    left["needs_human_review"] = bool(left.get("needs_human_review", True) or right.get("needs_human_review", True))
    _refresh_visual_retention(left)


def _refresh_visual_retention(item: dict[str, Any]) -> None:
    review = item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
    item["visual_retention"] = _visual_retention(item, review)


def _visual_retention(item: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    frame_paths = [str(path) for path in item.get("frame_paths") or [] if str(path)]
    structured = item.get("structured_visual") if isinstance(item.get("structured_visual"), list) else []
    material_types = {str(value) for value in item.get("material_types") or []}
    keep_images = bool(review.get("keep_images"))
    visual_heavy = bool(material_types & {"formula", "table", "code"}) or bool(structured)
    if keep_images:
        recommendation = "keep_image"
        reason = "人工复核标记为必须保留图片，不能完全降维成文字。"
    elif visual_heavy and frame_paths:
        recommendation = "review_image"
        reason = "包含公式/表格/代码或结构化视觉，需人工确认是否已完整降维。"
    elif frame_paths:
        recommendation = "image_available"
        reason = "已有关键帧，可作为证据保留；是否必须保留需人工判断。"
    else:
        recommendation = "text_only_candidate"
        reason = "当前没有关键帧，暂按文字证据处理；若画面承载信息应补帧。"
    return {
        "has_frames": bool(frame_paths),
        "frame_count": len(frame_paths),
        "keep_image_required": keep_images,
        "structured_visual_count": len([row for row in structured if isinstance(row, dict)]),
        "visual_material_types": sorted(material_types & {"formula", "table", "code"}),
        "recommendation": recommendation,
        "reason": reason,
    }


def _coverage_audit(timeline: list[dict[str, Any]], *, time_gap_audit: dict[str, Any] | None = None) -> dict[str, Any]:
    gap_audit = time_gap_audit or _time_gap_audit(timeline)
    gap_summary = gap_audit.get("summary") if isinstance(gap_audit.get("summary"), dict) else {}
    return {
        "timeline_items": len(timeline),
        "items_with_transcript": sum(1 for item in timeline if item.get("transcript")),
        "items_with_visual_text": sum(1 for item in timeline if item.get("visual_text")),
        "items_with_frames": sum(1 for item in timeline if item.get("frame_paths")),
        "items_needing_review": sum(1 for item in timeline if item.get("needs_human_review")),
        "items_reviewed": sum(1 for item in timeline if item.get("review_status") == "reviewed"),
        "items_needing_revision": sum(1 for item in timeline if item.get("review_status") == "needs_revision"),
        "items_with_human_notes": sum(1 for item in timeline if (item.get("human_review") or {}).get("notes")),
        "items_with_corrected_transcript": sum(1 for item in timeline if "original_transcript" in item),
        "items_with_corrected_visual_text": sum(1 for item in timeline if "original_visual_text" in item),
        "items_with_structured_visual": sum(1 for item in timeline if item.get("structured_visual")),
        "items_with_visual_route": sum(1 for item in timeline if item.get("visual_route")),
        "items_with_visual_understanding": sum(1 for item in timeline if item.get("visual_understanding")),
        "items_with_temporal_understanding": sum(1 for item in timeline if item.get("temporal_visual_understanding")),
        "structured_visual_entries": sum(
            len(item.get("structured_visual") or [])
            for item in timeline
            if isinstance(item.get("structured_visual"), list)
        ),
        "possible_code_items": sum(1 for item in timeline if "code" in item.get("material_types", [])),
        "possible_formula_items": sum(1 for item in timeline if "formula" in item.get("material_types", [])),
        "possible_table_items": sum(1 for item in timeline if "table" in item.get("material_types", [])),
        "timeline_duration_seconds": round(float(gap_summary.get("duration_seconds") or 0), 3),
        "timeline_covered_seconds": round(float(gap_summary.get("covered_seconds") or 0), 3),
        "timeline_uncovered_seconds": round(float(gap_summary.get("uncovered_seconds") or 0), 3),
        "timeline_coverage_percent": round(float(gap_summary.get("coverage_percent") or 0), 1),
        "time_gap_count": int(gap_summary.get("gap_count") or 0),
        "max_time_gap_seconds": round(float(gap_summary.get("max_gap_seconds") or 0), 3),
    }


def _time_gap_audit(timeline: list[dict[str, Any]], *, gap_threshold: float = 0.5) -> dict[str, Any]:
    by_video: dict[str, list[dict[str, Any]]] = {}
    for item in timeline:
        key = str(item.get("video_key") or item.get("video_id") or "unknown")
        by_video.setdefault(key, []).append(item)

    videos = []
    gaps = []
    total_duration = 0.0
    total_covered = 0.0
    for video_key, items in sorted(by_video.items()):
        duration = max([_safe_float(item.get("video_duration_seconds")) for item in items] + [_safe_float(item.get("end")) for item in items] + [0.0])
        intervals = _merge_intervals(
            [
                (_safe_float(item.get("start")), _safe_float(item.get("end")))
                for item in items
                if _safe_float(item.get("end")) > _safe_float(item.get("start"))
            ]
        )
        covered = sum(end - start for start, end in intervals)
        video_gaps = _interval_gaps(intervals, duration=duration, threshold=gap_threshold)
        for gap in video_gaps:
            gaps.append({"video_key": video_key, **gap})
        total_duration += duration
        total_covered += min(covered, duration) if duration > 0 else covered
        videos.append(
            {
                "video_key": video_key,
                "duration_seconds": round(duration, 3),
                "covered_seconds": round(min(covered, duration) if duration > 0 else covered, 3),
                "uncovered_seconds": round(sum(gap["duration_seconds"] for gap in video_gaps), 3),
                "gap_count": len(video_gaps),
                "gaps": video_gaps,
            }
        )
    total_uncovered = sum(float(gap["duration_seconds"]) for gap in gaps)
    coverage_percent = (total_covered / total_duration * 100) if total_duration > 0 else 0.0
    return {
        "schema": "lecture_time_gap_audit.v1",
        "gap_threshold_seconds": gap_threshold,
        "summary": {
            "video_count": len(videos),
            "duration_seconds": round(total_duration, 3),
            "covered_seconds": round(total_covered, 3),
            "uncovered_seconds": round(total_uncovered, 3),
            "coverage_percent": round(coverage_percent, 1),
            "gap_count": len(gaps),
            "max_gap_seconds": round(max([float(gap["duration_seconds"]) for gap in gaps], default=0.0), 3),
        },
        "videos": videos,
        "gaps": gaps,
    }


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _interval_gaps(intervals: list[tuple[float, float]], *, duration: float, threshold: float) -> list[dict[str, float | str]]:
    gaps: list[dict[str, float | str]] = []
    cursor = 0.0
    for start, end in intervals:
        if start - cursor > threshold:
            gaps.append(_gap_row(cursor, start, "between_segments" if cursor else "leading"))
        cursor = max(cursor, end)
    if duration - cursor > threshold:
        gaps.append(_gap_row(cursor, duration, "trailing"))
    return gaps


def _gap_row(start: float, end: float, position: str) -> dict[str, float | str]:
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "duration_seconds": round(max(end - start, 0.0), 3),
        "position": position,
    }


def _quality_audit(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    priority_items = []
    issue_counts: dict[str, int] = {}
    for index, item in enumerate(timeline, start=1):
        issues = _quality_issues(item)
        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        score = _quality_score(issues)
        if score:
            priority_items.append(
                {
                    "index": index,
                    "start": item.get("start", 0),
                    "end": item.get("end", 0),
                    "score": score,
                    "issues": issues,
                    "material_types": item.get("material_types", []),
                    "source_segment_ids": item.get("source_segment_ids", []),
                    "transcript": item.get("transcript", ""),
                    "visual_text": item.get("visual_text", ""),
                    "frame_paths": item.get("frame_paths", []),
                    "review_status": item.get("review_status", "pending"),
                }
            )
    priority_items.sort(key=lambda row: (-int(row["score"]), float(row.get("start", 0)), int(row.get("index", 0))))
    return {
        "summary": {
            "timeline_items": len(timeline),
            "items_with_issues": len(priority_items),
            "max_score": max([int(item["score"]) for item in priority_items], default=0),
            **{f"issue_{key}": value for key, value in sorted(issue_counts.items())},
        },
        "priority_items": priority_items,
    }


def _quality_issues(item: dict[str, Any]) -> list[str]:
    issues = []
    transcript = str(item.get("transcript") or "").strip()
    visual_text = str(item.get("visual_text") or "").strip()
    frames = item.get("frame_paths") or []
    material_types = set(item.get("material_types") or [])
    structured_visual = item.get("structured_visual") if isinstance(item.get("structured_visual"), list) else []
    visual_route = str(item.get("visual_route") or "")
    review = item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
    if item.get("needs_human_review", True):
        issues.append("needs_human_review")
    if item.get("review_status") == "needs_revision":
        issues.append("needs_revision")
    if not transcript:
        issues.append("missing_transcript")
    if not visual_text:
        issues.append("missing_visual_text")
    if not frames:
        issues.append("missing_frame")
    if material_types & {"formula", "table", "code"} and not frames:
        issues.append("structured_visual_without_frame")
    if material_types & {"formula", "table", "code"} and not visual_text:
        issues.append("structured_visual_without_ocr")
    if material_types & {"formula", "table", "code"} and not structured_visual:
        issues.append("structured_visual_without_structure")
    if frames and not visual_route:
        issues.append("missing_visual_route")
    if visual_route in {"semantic_frame", "mixed"} and not item.get("visual_understanding"):
        issues.append("semantic_frame_without_analysis")
        issues.append("missing_visual_understanding")
    if visual_route in {"temporal_sequence", "mixed"} and not item.get("temporal_visual_understanding"):
        issues.append("temporal_sequence_without_analysis")
        issues.append("missing_visual_understanding")
    if review.get("asr_ocr_error"):
        issues.append("reported_asr_ocr_error")
    if review.get("missing_info"):
        issues.append("reported_missing_info")
    if review.get("keep_images") and not frames:
        issues.append("keep_image_without_frame")
    return _dedupe(issues)


def _quality_score(issues: list[str]) -> int:
    weights = {
        "reported_missing_info": 8,
        "keep_image_without_frame": 8,
        "reported_asr_ocr_error": 7,
        "structured_visual_without_frame": 6,
        "structured_visual_without_ocr": 6,
        "structured_visual_without_structure": 5,
        "temporal_sequence_without_analysis": 6,
        "semantic_frame_without_analysis": 5,
        "missing_visual_understanding": 4,
        "missing_visual_route": 3,
        "needs_revision": 5,
        "missing_frame": 3,
        "missing_visual_text": 3,
        "missing_transcript": 3,
        "needs_human_review": 1,
    }
    return sum(weights.get(issue, 1) for issue in issues)


def _quality_issue_label(issue: str) -> str:
    labels = {
        "needs_human_review": "待人工复核",
        "needs_revision": "需修订",
        "missing_transcript": "缺口语/字幕",
        "missing_visual_text": "缺画面文字",
        "missing_frame": "缺关键帧",
        "structured_visual_without_frame": "结构化视觉无截图",
        "structured_visual_without_ocr": "结构化视觉无 OCR",
        "structured_visual_without_structure": "结构化视觉未结构化",
        "missing_visual_route": "缺画面类型路由",
        "semantic_frame_without_analysis": "语义画面未理解",
        "temporal_sequence_without_analysis": "连续画面未理解",
        "missing_visual_understanding": "缺视频视觉理解",
        "reported_asr_ocr_error": "已标记识别错误",
        "reported_missing_info": "已标记疑似遗漏",
        "keep_image_without_frame": "需保留图但无帧",
    }
    return labels.get(issue, issue)


def _material_types(text: str, frame_paths: list[str], signals: list[str]) -> list[str]:
    types = []
    lowered = text.lower()
    lowered_signals = " ".join(str(signal).lower() for signal in signals)
    if text.strip():
        types.append("text")
    if frame_paths:
        types.append("image")
    if _looks_like_code(text, lowered, lowered_signals):
        types.append("code")
    if _looks_like_formula(text, lowered, lowered_signals):
        types.append("formula")
    if _looks_like_table(text, lowered_signals):
        types.append("table")
    if _looks_like_diagram(text, lowered_signals):
        types.append("diagram")
    if _looks_like_board(text, lowered_signals):
        types.append("board")
    return _dedupe(types or ["unknown"])


def _looks_like_code(text: str, lowered: str, lowered_signals: str) -> bool:
    if any(token in lowered_signals for token in ["code", "terminal", "notebook", "ide"]):
        return True
    code_tokens = [
        "def ",
        "class ",
        "import ",
        "from ",
        "const ",
        "let ",
        "var ",
        "function ",
        "return ",
        "=>",
        "```",
        "#include",
        "public static",
        "console.log",
    ]
    if any(token in lowered for token in code_tokens):
        return True
    return bool(re.search(r"^\s*(for|while|if|else|elif|try|catch)\s*\(.+\)\s*[{:]?", text, flags=re.MULTILINE))


def _looks_like_formula(text: str, lowered: str, lowered_signals: str) -> bool:
    if any(token in lowered_signals for token in ["formula", "equation", "latex", "math"]):
        return True
    formula_tokens = ["≈", "∑", "√", "≤", "≥", "≠", "\\frac", "\\sum", "\\int", "^2", "^", "=>"]
    if any(token in text for token in formula_tokens) and any(char.isdigit() for char in text):
        return True
    if re.search(r"\b[a-zA-Z]\s*\([^)]*\)\s*=", text):
        return True
    return bool(re.search(r"\b[a-zA-Z]\s*[=+\-*/]\s*[\da-zA-Z]", text) and any(char.isdigit() for char in text))


def _looks_like_table(text: str, lowered_signals: str) -> bool:
    if any(token in lowered_signals for token in ["table", "spreadsheet", "sheet"]):
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    pipe_rows = [line for line in lines if line.count("|") >= 2]
    if len(pipe_rows) >= 1 and any("---" in line or len(pipe_rows) >= 2 for line in pipe_rows):
        return True
    if any(len([cell for cell in line.split("|") if cell.strip()]) >= 3 for line in pipe_rows):
        return True
    tab_rows = [line for line in lines if line.count("\t") >= 1]
    if len(tab_rows) >= 2:
        return True
    return bool(re.search(r"(列|行|表格|对比表|数据表|matrix|row|column)", text, flags=re.IGNORECASE))


def _looks_like_diagram(text: str, lowered_signals: str) -> bool:
    if any(token in lowered_signals for token in ["diagram", "chart", "graph", "plot", "flow", "mindmap"]):
        return True
    return bool(re.search(r"(图像|图表|流程图|结构图|示意图|坐标系|曲线|节点|箭头|graph|chart|diagram|plot)", text, flags=re.IGNORECASE))


def _looks_like_board(text: str, lowered_signals: str) -> bool:
    if any(token in lowered_signals for token in ["whiteboard", "blackboard", "board", "handwriting"]):
        return True
    return bool(re.search(r"(板书|白板|黑板|手写|推导过程|chalkboard|whiteboard|handwritten)", text, flags=re.IGNORECASE))


def _join_unique(left: str, right: str) -> str:
    left = str(left or "").strip()
    right = str(right or "").strip()
    if not left:
        return right
    if not right or right in left:
        return left
    if left in right:
        return right
    return f"{left}\n\n{right}"


def _dedupe(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _default_title(videos: list[dict[str, Any]]) -> str:
    first = videos[0]["metadata"]
    return str(first.get("title") or first.get("path") or "Lecture Knowledge Package")


def _find_timeline_item(timeline: list[dict[str, Any]], review: dict[str, Any]) -> dict[str, Any] | None:
    review_ids = {str(value) for value in review.get("sourceSegmentIds", []) if str(value)}
    if review_ids:
        for item in timeline:
            item_ids = {str(value) for value in item.get("source_segment_ids", []) if str(value)}
            if review_ids & item_ids:
                return item
    index = review.get("index")
    if isinstance(index, int) and 1 <= index <= len(timeline):
        return timeline[index - 1]
    return None


def _manual_timeline_item(review: dict[str, Any]) -> dict[str, Any]:
    start = _safe_float(review.get("start"))
    end = _safe_float(review.get("end")) or start
    midpoint = _safe_float(review.get("midpoint")) or ((start + end) / 2)
    source_ids = [str(value) for value in review.get("sourceSegmentIds", []) if str(value)]
    manual_id = str(review.get("manualTimelineId") or f"manual-gap-{start:.3f}-{end:.3f}")
    if not source_ids:
        source_ids = [manual_id]
    frame_paths = [str(path) for path in review.get("framePaths", []) if str(path)]
    material_types = [str(value).strip() for value in review.get("materialTypes", []) if str(value).strip()]
    if frame_paths and "image" not in material_types:
        material_types.append("image")
    human_review = _normalise_human_review(review)
    human_review["manual_timeline_item"] = True
    human_review["manual_timeline_id"] = manual_id
    evidence = _manual_timeline_evidence(review, manual_id=manual_id, frame_paths=frame_paths)
    item = {
        "video_key": str(review.get("videoKey") or review.get("video_key") or ""),
        "video_id": str(review.get("videoId") or review.get("video_id") or ""),
        "video_duration_seconds": _safe_float(review.get("videoDurationSeconds")),
        "start": start,
        "end": max(end, start),
        "midpoint": max(midpoint, 0.0),
        "source_segment_ids": source_ids,
        "signals": ["manual_time_gap_entry"],
        "transcript": str(review.get("transcript") or review.get("correctedTranscript") or "").strip(),
        "visual_text": str(review.get("visualText") or review.get("correctedVisualText") or "").strip(),
        "frame_paths": frame_paths,
        "material_types": material_types or ["manual"],
        "needs_human_review": True,
        "review_status": "needs_revision",
        "human_review": human_review,
        "evidence": evidence,
        "uncertainties": ["manual timeline item created from time gap"],
    }
    _refresh_visual_retention(item)
    return item


def _manual_timeline_evidence(review: dict[str, Any], *, manual_id: str, frame_paths: list[str]) -> dict[str, Any]:
    return {
        "source": "manual_time_gap_review",
        "manual_timeline_id": manual_id,
        "gap": {
            "start": _safe_float(review.get("start")),
            "end": _safe_float(review.get("end")),
            "midpoint": _safe_float(review.get("midpoint")),
            "duration_seconds": _safe_float(review.get("gapDurationSeconds")),
            "position": str(review.get("gapPosition") or "gap"),
        },
        "video_key": str(review.get("videoKey") or review.get("video_key") or ""),
        "recapture_frame_paths": frame_paths,
        "recapture_frame_path": str(review.get("recaptureFramePath") or ""),
        "recapture_relative_output": str(review.get("recaptureRelativeOutput") or ""),
        "created_from": "BiliNote time gap manual entry",
    }


def _normalise_human_review(review: dict[str, Any]) -> dict[str, Any]:
    reviewed = bool(review.get("reviewed", False))
    keep_images = bool(review.get("keepImages", False))
    asr_ocr_error = bool(review.get("asrOcrError", False))
    missing_info = bool(review.get("missingInfo", False))
    notes = str(review.get("notes") or "").strip()
    corrected_transcript = str(review.get("correctedTranscript") or "").strip()
    corrected_visual_text = str(review.get("correctedVisualText") or "").strip()
    structured_visual = _normalise_structured_visual(review)
    final_card = str(review.get("finalCard") or "").strip()
    card_questions = _normalise_review_list(review.get("cardQuestions"))
    card_links = _normalise_review_list(review.get("cardLinks"))
    next_review = str(review.get("nextReview") or "").strip()
    card_kind = str(review.get("cardKind") or "").strip()
    mastery_level = str(review.get("masteryLevel") or "").strip()
    review_state = str(review.get("reviewState") or "").strip()
    confusion_points = _normalise_review_list(review.get("confusionPoints"))
    structured_visual_text = "\n\n".join(
        str(entry.get("markdown") or "").strip()
        for entry in structured_visual
        if str(entry.get("markdown") or "").strip()
    )
    structured_visual_type = str(review.get("structuredVisualType") or "").strip()
    status = "pending"
    if reviewed and not (asr_ocr_error or missing_info):
        status = "reviewed"
    elif (
        reviewed
        or keep_images
        or asr_ocr_error
        or missing_info
        or notes
        or corrected_transcript
        or corrected_visual_text
        or structured_visual
        or final_card
        or card_questions
        or card_links
        or next_review
        or card_kind
        or mastery_level
        or review_state
        or confusion_points
    ):
        status = "needs_revision"
    return {
        "status": status,
        "reviewed": reviewed,
        "keep_images": keep_images,
        "asr_ocr_error": asr_ocr_error,
        "missing_info": missing_info,
        "notes": notes,
        "corrected_transcript": corrected_transcript,
        "corrected_visual_text": corrected_visual_text,
        "structured_visual": structured_visual,
        "structured_visual_text": structured_visual_text,
        "structured_visual_type": structured_visual_type,
        "final_card": final_card,
        "card_questions": card_questions,
        "card_links": card_links,
        "next_review": next_review,
        "card_kind": card_kind,
        "mastery_level": mastery_level,
        "review_state": review_state,
        "confusion_points": confusion_points,
        "reviewed_at": now_iso(),
    }


def _normalise_review_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    return [line.strip(" -\t") for line in text.splitlines() if line.strip(" -\t")]


def _apply_text_corrections(item: dict[str, Any], human_review: dict[str, Any]) -> None:
    corrected_transcript = str(human_review.get("corrected_transcript") or "").strip()
    corrected_visual_text = str(human_review.get("corrected_visual_text") or "").strip()
    if corrected_transcript and corrected_transcript != str(item.get("transcript") or "").strip():
        item.setdefault("original_transcript", item.get("transcript", ""))
        item["transcript"] = corrected_transcript
    if corrected_visual_text and corrected_visual_text != str(item.get("visual_text") or "").strip():
        item.setdefault("original_visual_text", item.get("visual_text", ""))
        item["visual_text"] = corrected_visual_text


def _normalise_structured_visual(review: dict[str, Any]) -> list[dict[str, str]]:
    entries = review.get("structuredVisual")
    if isinstance(entries, list):
        rows = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            markdown = str(entry.get("markdown") or entry.get("text") or entry.get("latex") or "").strip()
            if not markdown:
                continue
            rows.append(
                {
                    "type": str(entry.get("type") or entry.get("material_type") or "structured_visual").strip() or "structured_visual",
                    "source": str(entry.get("source") or "manual_review").strip() or "manual_review",
                    "markdown": markdown,
                }
            )
        return rows
    markdown = str(review.get("structuredVisualText") or "").strip()
    if not markdown:
        return []
    return [
        {
            "type": str(review.get("structuredVisualType") or "structured_visual").strip() or "structured_visual",
            "source": "manual_review",
            "markdown": markdown,
        }
    ]


def _apply_structured_visual_corrections(item: dict[str, Any], human_review: dict[str, Any]) -> None:
    entries = human_review.get("structured_visual")
    if not isinstance(entries, list) or not entries:
        return
    item["structured_visual"] = entries
    additions = [str(entry.get("markdown") or "").strip() for entry in entries if str(entry.get("markdown") or "").strip()]
    if additions:
        visual_text = str(item.get("visual_text") or "").strip()
        for addition in additions:
            if addition not in visual_text:
                visual_text = f"{visual_text}\n\n{addition}".strip() if visual_text else addition
        item["visual_text"] = visual_text


def _human_review_markdown(review: dict[str, Any]) -> list[str]:
    if not review:
        return [
            "- [ ] 是否有遗漏的屏幕文字、板书、表格、公式或代码？",
            "- [ ] 是否需要保留原图而不是降维成文字？",
            "- [ ] 是否存在 ASR/OCR 误识别？",
            "- [ ] 是否可以拆成概念、步骤、例题、定义或结论？",
        ]
    checked = {
        True: "x",
        False: " ",
    }
    lines = [
        f"- 状态：`{review.get('status', 'pending')}`",
        f"- [{checked[bool(review.get('reviewed'))]}] 已确认这一段",
        f"- [{checked[bool(review.get('keep_images'))]}] 必须保留图片，不能完全降维成文字",
        f"- [{checked[bool(review.get('asr_ocr_error'))]}] 存在 ASR/OCR 错误",
        f"- [{checked[bool(review.get('missing_info'))]}] 疑似有遗漏信息",
    ]
    if review.get("notes"):
        lines.extend(["", "人工备注：", "", str(review.get("notes"))])
    if review.get("corrected_transcript"):
        lines.extend(["", "修正后的口语/字幕：", "", str(review.get("corrected_transcript"))])
    if review.get("corrected_visual_text"):
        lines.extend(["", "修正后的画面文字 / OCR / 视觉观察：", "", str(review.get("corrected_visual_text"))])
    if review.get("final_card"):
        lines.extend(["", "最终学习卡片：", "", str(review.get("final_card"))])
    if review.get("card_questions"):
        lines.extend(["", "复习问题：", ""])
        lines.extend([f"- {question}" for question in review.get("card_questions") or []])
    if review.get("card_links"):
        lines.extend(["", "关联链接：", ""])
        lines.extend([f"- {link}" for link in review.get("card_links") or []])
    if review.get("next_review"):
        lines.extend(["", f"下次复习：{review.get('next_review')}"])
    if review.get("card_kind") or review.get("mastery_level") or review.get("review_state") or review.get("confusion_points"):
        lines.extend(["", "复习管理：", ""])
        if review.get("card_kind"):
            lines.append(f"- 卡片类型：`{review.get('card_kind')}`")
        if review.get("mastery_level"):
            lines.append(f"- 掌握度：`{review.get('mastery_level')}`")
        if review.get("review_state"):
            lines.append(f"- 复习状态：`{review.get('review_state')}`")
        for point in review.get("confusion_points") or []:
            lines.append(f"- 混淆点：{point}")
    return lines


def _obsidian_timeline_item(index: int, item: dict[str, Any], *, include_review: bool) -> list[str]:
    start = format_timestamp(float(item.get("start", 0)))
    end = format_timestamp(float(item.get("end", 0)))
    material = ", ".join(item.get("material_types") or ["unknown"])
    signals = ", ".join(item.get("signals") or [])
    lines = [
        f"## {index}. {start} - {end}",
        "",
        f"- 来源片段：{', '.join(item.get('source_segment_ids') or [])}",
        f"- 材料类型：{material}",
        f"- 触发信号：{signals}",
        f"- 审核状态：`{item.get('review_status', 'pending')}`",
        f"- 需要人工复核：{item.get('needs_human_review', True)}",
        "",
    ]
    issues = _quality_issues(item)
    if issues:
        lines.extend(
            [
                "### 缺口审计",
                "",
                f"- 风险分：{_quality_score(issues)}",
                f"- 问题：{', '.join(_quality_issue_label(issue) for issue in issues)}",
                "",
            ]
        )
    evidence_lines = _evidence_markdown(item)
    if evidence_lines:
        lines.extend(["### 证据链", "", *evidence_lines, ""])
    retention = item.get("visual_retention") if isinstance(item.get("visual_retention"), dict) else {}
    if retention:
        lines.extend(
            [
                "### 视觉保留判断",
                "",
                f"- 建议：`{retention.get('recommendation', '')}`",
                f"- 有关键帧：{retention.get('has_frames', False)}",
                f"- 关键帧数量：{retention.get('frame_count', 0)}",
                f"- 必须保留图片：{retention.get('keep_image_required', False)}",
                f"- 结构化视觉条目：{retention.get('structured_visual_count', 0)}",
                f"- 原因：{retention.get('reason', '')}",
                "",
            ]
        )
    lines.extend(
        [
        "### 口语/字幕",
        "",
        str(item.get("transcript") or "（缺失或未命中）"),
        "",
        "### 画面文字 / OCR / 视觉观察",
        "",
        str(item.get("visual_text") or "（缺失或未命中）"),
        "",
        ]
    )
    structured = _structured_visual_markdown(item)
    if structured:
        lines.extend(["### 结构化视觉", "", structured, ""])
    frames = item.get("frame_paths") or []
    if frames:
        lines.extend(["### 关键帧", ""])
        for frame in frames:
            lines.extend([_markdown_image(str(frame)), "", f"`{frame}`", ""])
    if include_review:
        review = item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
        lines.extend(["### 人工复核", "", *_human_review_markdown(review), ""])
    return lines


def _markdown_image(path: str) -> str:
    escaped = path.replace("\\", "/")
    return f"![关键帧]({escaped})"


def _structured_visual_markdown(item: dict[str, Any]) -> str:
    rows = item.get("structured_visual") if isinstance(item.get("structured_visual"), list) else []
    blocks = []
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        markdown = str(entry.get("markdown") or "").strip()
        if not markdown:
            continue
        label = str(entry.get("type") or entry.get("source") or "structured_visual").strip()
        source = str(entry.get("source") or "").strip()
        header = f"#### {label}" if label else "#### structured_visual"
        if source and source != label:
            header = f"{header} ({source})"
        blocks.append("\n\n".join([header, markdown]))
    return "\n\n".join(blocks)


def _evidence_markdown(item: dict[str, Any]) -> list[str]:
    evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
    if not evidence:
        return []
    lines: list[str] = []
    source = str(evidence.get("source") or "").strip()
    if source:
        lines.append(f"- 证据来源：`{source}`")
    manual_id = str(evidence.get("manual_timeline_id") or "").strip()
    if manual_id:
        lines.append(f"- 人工片段 ID：`{manual_id}`")
    gap = evidence.get("gap") if isinstance(evidence.get("gap"), dict) else {}
    if gap:
        start = _safe_float(gap.get("start"))
        end = _safe_float(gap.get("end"))
        midpoint = _safe_float(gap.get("midpoint"))
        duration = _safe_float(gap.get("duration_seconds"))
        position = str(gap.get("position") or "gap").strip()
        lines.append(
            "- 时间盲区："
            f"{format_timestamp(start)} - {format_timestamp(end)}"
            f"，中点 {format_timestamp(midpoint)}"
            f"，{duration:g} 秒，{position}"
        )
    video_key = str(evidence.get("video_key") or "").strip()
    if video_key:
        lines.append(f"- 视频键：`{video_key}`")
    frame = str(evidence.get("recapture_frame_path") or "").strip()
    if frame:
        lines.append(f"- 盲区中点帧：`{frame}`")
    relative_output = str(evidence.get("recapture_relative_output") or "").strip()
    if relative_output:
        lines.append(f"- 相对补帧输出：`{relative_output}`")
    created_from = str(evidence.get("created_from") or "").strip()
    if created_from:
        lines.append(f"- 创建方式：{created_from}")
    return lines


def _structured_visual_html(item: dict[str, Any]) -> str:
    markdown = _structured_visual_markdown(item)
    if not markdown:
        return ""
    return f"""
      <section class="section">
        <h3>结构化视觉</h3>
        <pre>{html.escape(markdown)}</pre>
      </section>
"""


def _visual_gap_html(item: dict[str, Any]) -> str:
    labels = _visual_gap_labels(item)
    if not labels:
        return ""
    rows = "".join(f"<li>{html.escape(label)}</li>" for label in labels)
    return f"""
      <section class="section">
        <h3>待补齐</h3>
        <ul class="gap-list">{rows}</ul>
      </section>
"""


def _visual_gap_labels(item: dict[str, Any]) -> list[str]:
    route = str(item.get("visual_route") or "")
    labels: list[str] = []
    if route in {"document_visual", "mixed"} and not _has_non_empty(item.get("structured_visual")):
        labels.append("图文截图未解析：调用 run_visual_structure_plan / ebook_markdown_pipeline")
    if route in {"semantic_frame", "mixed"} and not _has_mapping(item.get("visual_understanding")):
        labels.append("单帧视觉未理解：调用 run_multimodal_frame_analysis")
    if route in {"temporal_sequence", "mixed"} and not _has_mapping(item.get("temporal_visual_understanding")):
        labels.append("连续片段未理解：先确认 temporal_frame_paths，再调用 run_temporal_visual_analysis")
    return labels


def _visual_understanding_html(item: dict[str, Any]) -> str:
    visual = item.get("visual_understanding") if isinstance(item.get("visual_understanding"), dict) else {}
    if not visual:
        return ""
    return _mapping_section_html(
        "单帧视觉理解",
        visual,
        [
            "objects",
            "actions",
            "screen_state",
            "spatial_relations",
            "instructor_focus",
            "non_text_information",
            "confidence",
            "retain_frame_reason",
            "source",
        ],
    )


def _temporal_understanding_html(item: dict[str, Any]) -> str:
    temporal = item.get("temporal_visual_understanding") if isinstance(item.get("temporal_visual_understanding"), dict) else {}
    if not temporal:
        return ""
    return _mapping_section_html(
        "连续片段理解",
        temporal,
        [
            "event_sequence",
            "state_changes",
            "operation_steps",
            "causal_links",
            "possible_missing_points",
            "confidence",
            "source",
        ],
    )


def _mapping_section_html(title: str, payload: dict[str, Any], keys: list[str]) -> str:
    lines: list[str] = []
    for key in keys:
        value = payload.get(key)
        if value in (None, "", [], {}):
            continue
        lines.append(f"- {key}: {_display_value(value)}")
    if not lines:
        return ""
    text = "\n".join(lines)
    return f"""
      <section class="section">
        <h3>{html.escape(title)}</h3>
        <pre>{html.escape(text)}</pre>
      </section>
"""


def _display_value(value: object) -> str:
    if isinstance(value, list):
        return "；".join(_display_value(item) for item in value)
    if isinstance(value, dict):
        return "；".join(f"{key}={_display_value(item)}" for key, item in value.items() if item not in (None, "", [], {}))
    return str(value)


def _safe_relative_folder(folder: str) -> Path:
    cleaned = Path(folder.strip() or "00_Inbox/AI")
    if cleaned.is_absolute() or any(part == ".." for part in cleaned.parts):
        raise ValueError("Obsidian folder must be relative to the vault")
    return cleaned


def _safe_filename(value: str) -> str:
    cleaned = "".join(char if char not in '<>:"/\\|?*' else "_" for char in value.strip())
    cleaned = cleaned.rstrip(". ")
    return cleaned or "lecture-package"


def _header_task_console_link(package: dict[str, Any]) -> str:
    artifacts = package.get("review_artifacts") if isinstance(package.get("review_artifacts"), dict) else {}
    task_console = str(artifacts.get("task_console") or "").strip()
    if not task_console:
        return ""
    return f'<a class="button primary" href="{html.escape(task_console)}">任务控制台</a>'


def _coverage_cards(coverage: dict[str, Any]) -> str:
    labels = {
        "timeline_items": "时间线片段",
        "items_with_transcript": "含转写",
        "items_with_visual_text": "含视觉文字",
        "items_with_frames": "含关键帧",
        "items_needing_review": "待复核",
        "items_reviewed": "已确认",
        "items_needing_revision": "需修订",
        "items_with_human_notes": "人工备注",
        "items_with_corrected_transcript": "已修正转写",
        "items_with_corrected_visual_text": "已修正视觉文字",
        "possible_code_items": "疑似代码",
        "possible_formula_items": "疑似公式",
        "possible_table_items": "疑似表格",
    }
    return "\n".join(
        f'<div class="metric"><strong>{html.escape(str(coverage.get(key, 0)))}</strong><span>{label}</span></div>'
        for key, label in labels.items()
    )


def _workflow_cards(package: dict[str, Any]) -> str:
    artifacts = package.get("review_artifacts") if isinstance(package.get("review_artifacts"), dict) else {}
    timeline = package.get("timeline") if isinstance(package.get("timeline"), list) else []
    pending_document = []
    pending_semantic = []
    pending_temporal = []
    for index, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        route = str(item.get("visual_route") or "")
        if route in {"document_visual", "mixed"} and not _has_non_empty(item.get("structured_visual")):
            pending_document.append(index)
        if route in {"semantic_frame", "mixed"} and not _has_mapping(item.get("visual_understanding")):
            pending_semantic.append(index)
        if route in {"temporal_sequence", "mixed"} and not _has_mapping(item.get("temporal_visual_understanding")):
            pending_temporal.append(index)
    export_ready = bool(timeline)
    cards = [
        _workflow_card(
            "任务控制台",
            "入口",
            "集中查看状态、推荐下一步和可复制命令",
            str(artifacts.get("mcp_export_task_console_args") or "mcp-export-task-console.args.json"),
            [],
            link=str(artifacts.get("task_console") or "task-console.html"),
            command=".\\scripts\\open-task-console.cmd",
            link_label="打开任务控制台",
        ),
        _workflow_card(
            "待图文解析",
            len(pending_document),
            "走 ebook_markdown_pipeline",
            "mcp-run-visual-structure.args.json",
            pending_document,
        ),
        _workflow_card(
            "待单帧理解",
            len(pending_semantic),
            "走多模态单帧 API；真实执行前先跑 preflight",
            "mcp-run-multimodal-frame-analysis.args.json",
            pending_semantic,
            command=".\\scripts\\video-knowledge.ps1 mcp-call run_multimodal_frame_analysis mcp-run-multimodal-frame-analysis.args.json",
            confirmed_args="mcp-run-multimodal-frame-analysis-confirmed.args.json",
        ),
        _workflow_card(
            "待连续片段理解",
            len(pending_temporal),
            "先抽 5-12 帧再走多模态 API；真实执行前先跑 preflight",
            "mcp-run-temporal-visual-analysis.args.json",
            pending_temporal,
            command=".\\scripts\\video-knowledge.ps1 mcp-call run_temporal_visual_analysis mcp-run-temporal-visual-analysis.args.json",
            confirmed_args="mcp-run-temporal-visual-analysis-confirmed.args.json",
        ),
        _workflow_card(
            "视觉执行 preflight",
            "必跑",
            "检查 provider/key、候选、确认值和恢复链",
            "mcp-vision-execution-preflight.args.json",
            [],
            command=".\\scripts\\video-knowledge.ps1 mcp-call vision_execution_preflight mcp-vision-execution-preflight.args.json",
        ),
        _workflow_card(
            "可控执行状态",
            "必看",
            "查看门禁、审计、可恢复写入和下一步",
            "mcp-controlled-execution-check.args.json",
            [],
            link="controlled-execution-check.md",
            command=".\\scripts\\video-knowledge.ps1 mcp-call controlled_execution_check mcp-controlled-execution-check.args.json",
            link_label="打开可控执行状态",
        ),
        _workflow_card(
            "Bundle 验收",
            "必看",
            "查看 provider、review、coverage、export freshness 和下一步",
            "mcp-acceptance-check.args.json",
            [],
            link="acceptance-check.md",
            command=".\\scripts\\video-knowledge.ps1 mcp-call acceptance_check mcp-acceptance-check.args.json",
            link_label="打开验收报告",
        ),
        _workflow_card(
            "人工审核入口",
            "模板",
            "生成或查看 review-notes.template.json，填好后导入 review-notes.json",
            "mcp-prepare-review-session.args.json",
            [],
            link="review-session.md",
            command=".\\scripts\\video-knowledge.ps1 mcp-call prepare_review_session mcp-prepare-review-session.args.json",
            link_label="打开审核会话",
        ),
        _workflow_card(
            "审核模板 JSON",
            "可填",
            "人工或 agent 填写后用 apply_review_notes 导入",
            "review-notes.template.json",
            [],
            link="review-notes.template.json",
            command=".\\scripts\\video-knowledge.ps1 mcp-call apply_review_notes mcp-apply-review-notes.args.json",
            link_label="打开模板 JSON",
        ),
        _workflow_card(
            "本地执行演练",
            "fixture",
            "用本地 fixture 跑 preflight、确认写入、审计和可选回滚，不需要 API key",
            "mcp-controlled-execution-smoke.args.json",
            [],
            link="controlled-execution-smoke.md",
            command=".\\scripts\\video-knowledge.ps1 mcp-call controlled_execution_smoke mcp-controlled-execution-smoke.args.json",
            link_label="打开执行演练",
        ),
        _workflow_card(
            "一键生成知识库 Markdown",
            "是" if export_ready else "否",
            "生成 Obsidian 友好的知识库 Markdown",
            "mcp-export-knowledge-note.args.json",
            [],
            link="exports/knowledge-note.md",
            command=".\\scripts\\video-knowledge.ps1 mcp-call export_knowledge_note mcp-export-knowledge-note.args.json",
            link_label="打开知识笔记",
        ),
        _workflow_card(
            "完整逐字稿",
            "文本",
            "查看每个时间段说了什么",
            "exports/full-transcript.md",
            [],
            link="exports/full-transcript.md",
            link_label="打开逐字稿",
        ),
        _workflow_card(
            "提取审计",
            "漏项",
            "查看覆盖、缺口、人工审核状态和证据路径",
            "exports/extraction-audit.md",
            [],
            link="exports/extraction-audit.md",
            link_label="打开提取审计",
        ),
        _workflow_card(
            "当前下一步",
            "命令",
            "让 agent 读取当前 bundle 的下一步动作",
            str(artifacts.get("mcp_next_action_args") or "mcp-bundle-next-action.args.json"),
            [],
            command=str(
                artifacts.get("next_action_command")
                or ".\\scripts\\video-knowledge.ps1 mcp-call bundle_next_action mcp-bundle-next-action.args.json"
            ),
        ),
        _workflow_card(
            "批量验收看板",
            "batch",
            "查看多视频 acceptance / screen text / next action 汇总",
            "batch-acceptance-summary.md",
            [],
            link=str(artifacts.get("batch_acceptance_summary") or "") or None,
            link_label="打开批量验收",
        ),
        _workflow_card(
            "批量修复队列",
            "batch",
            "预览或执行 batch-repair-run；默认不会调用长任务或 API",
            "batch-repair-run.md",
            [],
            link=str(artifacts.get("batch_repair_run") or "") or None,
            command=str(artifacts.get("batch_repair_command") or ".\\scripts\\video-knowledge.ps1 batch-repair-run <batch-acceptance-summary.json>"),
            link_label="打开批量修复报告",
        ),
        _workflow_card(
            "批量人工复核",
            "review",
            "集中查看所有视频仍需人工判断的条目",
            "batch-human-review.md",
            [],
            link=str(artifacts.get("batch_human_review") or "") or None,
            link_label="打开批量人工复核",
        ),
    ]
    return "\n".join(cards)


def _review_workbench_html(package: dict[str, Any]) -> str:
    artifacts = package.get("review_artifacts") if isinstance(package.get("review_artifacts"), dict) else {}
    paths = [
        ("任务控制台", artifacts.get("task_console") or "task-console.html"),
        ("审核模板", artifacts.get("review_notes_template") or "review-notes.template.json"),
        ("Review Pack", artifacts.get("review_pack") or "review-pack.md"),
        ("Todo JSON", artifacts.get("review_notes_todo") or "review-notes.todo.json"),
        ("复核进度", artifacts.get("review_closure_status") or "review-closure-status.md"),
        ("审核草稿", artifacts.get("review_notes") or "review-notes.json"),
        ("填写指南", artifacts.get("review_fill_guide") or "review-fill-guide.md"),
        ("导入参数", artifacts.get("mcp_apply_review_notes_args") or "mcp-apply-review-notes.args.json"),
    ]
    path_html = "".join(_review_path_link(label, str(path)) for label, path in paths)
    apply_command = str(artifacts.get("apply_command") or ".\\scripts\\video-knowledge.ps1 mcp-call apply_review_notes mcp-apply-review-notes.args.json")
    validate_command = str(
        artifacts.get("validate_command") or ".\\scripts\\video-knowledge.ps1 validate-review-notes <bundle_dir> --review-json review-notes.json"
    )
    review_server_command = str(
        artifacts.get("review_server_command") or ".\\scripts\\start-review-webui.ps1 -BundleDir '<webui-bundle>'"
    )
    return f"""
    <section class="review-workbench" id="review-workbench">
      <div>
        <h2>审核编辑区</h2>
        <div class="review-paths">{path_html}</div>
        <div class="filter-bar" aria-label="审核筛选">
          <button type="button" class="active" data-review-filter="all">全部</button>
          <button type="button" data-review-filter="needs-human-review">待人工复核</button>
          <button type="button" data-review-filter="missing-visual-text">缺画面文字</button>
          <button type="button" data-review-filter="screen-text">屏幕文字优先</button>
          <button type="button" data-review-filter="low-confidence">低置信度</button>
          <button type="button" data-review-filter="keep-image">需保留图片</button>
          <button type="button" data-review-filter="human-key-point">人工关键点</button>
          <button type="button" data-review-filter="corrected">已填修正</button>
        </div>
        <p class="hint">勾选每段的“选入审核 JSON 草稿”，右侧会生成可保存为 <code>review-notes.json</code> 的内容。先验证再导入，导入后再刷新验收报告。</p>
        <p class="hint">一键写回服务：<code>{html.escape(review_server_command)}</code></p>
        <p class="hint">验证：<code>{html.escape(validate_command)}</code></p>
        <p class="hint">导入：<code>{html.escape(apply_command)}</code></p>
        {_review_closure_panel_html(package)}
        {_screen_text_review_queue_html(package)}
      </div>
      <div>
        <h2>review-notes.json 草稿</h2>
        <textarea class="review-draft" id="review-json-draft" spellcheck="false"></textarea>
        <div class="writeback-actions">
          <button type="button" onclick="downloadReview()">下载 review-notes.json</button>
          <button type="button" class="primary" id="save-to-vkp" onclick="saveReviewToVkp()" disabled>保存到 VKP</button>
          <button type="button" onclick="clearReviewDraft()">清除浏览器草稿</button>
          <span class="draft-status" id="review-draft-status">正在检查草稿状态……</span>
        </div>
      </div>
    </section>
"""


def _review_path_link(label: str, path: str) -> str:
    safe_label = html.escape(label)
    safe_path = html.escape(path)
    if not path:
        return f'<div class="review-path"><span>{safe_label}</span><code></code></div>'
    return f'<div class="review-path"><span>{safe_label}</span><a href="{safe_path}"><code>{safe_path}</code></a></div>'


def _review_closure_panel_html(package: dict[str, Any]) -> str:
    artifacts = package.get("review_artifacts") if isinstance(package.get("review_artifacts"), dict) else {}
    timeline = package.get("timeline") if isinstance(package.get("timeline"), list) else []
    open_count = 0
    closed_count = 0
    by_reason: dict[str, int] = {}
    for item in timeline:
        if not isinstance(item, dict):
            continue
        review = item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
        status = str(item.get("review_status") or review.get("status") or "").lower()
        closed = status in {
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
        if closed:
            closed_count += 1
            continue
        issues = [str(issue) for issue in item.get("quality_issues") or [] if str(issue)]
        if issues or item.get("needs_human_review"):
            open_count += 1
            for issue in issues or ["needs_human_review"]:
                by_reason[issue] = by_reason.get(issue, 0) + 1
    reason_rows = "".join(
        f"<tr><td><code>{html.escape(key)}</code></td><td>{count}</td></tr>"
        for key, count in sorted(by_reason.items(), key=lambda pair: (-pair[1], pair[0]))[:8]
    )
    if not reason_rows:
        reason_rows = "<tr><td>无</td><td>0</td></tr>"
    closure_command = str(artifacts.get("review_closure_command") or ".\\scripts\\video-knowledge.ps1 mcp-call review_closure_status mcp-review-closure-status.args.json")
    return f"""
        <div class="review-priority">
          <h3>复核进度</h3>
          <div class="review-paths">
            <div class="review-path"><span>Open</span><code>{open_count}</code></div>
            <div class="review-path"><span>Closed</span><code>{closed_count}</code></div>
            <div class="review-path"><span>命令</span><code>{html.escape(closure_command)}</code></div>
          </div>
          <table><thead><tr><th>原因</th><th>数量</th></tr></thead><tbody>{reason_rows}</tbody></table>
        </div>
"""


def _screen_text_review_queue_html(package: dict[str, Any]) -> str:
    timeline = package.get("timeline") if isinstance(package.get("timeline"), list) else []
    groups = {
        "OCR 失败/空结果": [],
        "小字或低置信度 UI": [],
        "图文截图": [],
        "需保留图片": [],
    }
    for fallback_index, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        issues = {str(issue) for issue in item.get("quality_issues") or []}
        route = str(item.get("visual_route") or "")
        material_types = {str(value) for value in item.get("material_types") or []}
        review = item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
        index = int(item.get("index") or fallback_index)
        label = f"{index} {format_timestamp(_safe_float(item.get('start')))}"
        if issues & {"ocr_text_empty", "missing_ocr"}:
            groups["OCR 失败/空结果"].append(label)
        if issues & {"screen_text_low_confidence", "low_ocr_confidence"}:
            groups["小字或低置信度 UI"].append(label)
        if route in {"document_visual", "mixed"} or material_types & {"table", "formula", "code", "text"}:
            if issues & {"missing_visual_text", "structured_visual_without_structure", "ocr_text_empty"}:
                groups["图文截图"].append(label)
        if item.get("human_keep_image") or review.get("keep_image") or issues & {"keep_image_without_frame"}:
            groups["需保留图片"].append(label)
    rows = []
    for name, values in groups.items():
        sample = ", ".join(values[:12]) + (" ..." if len(values) > 12 else "")
        rows.append(f"<tr><td>{html.escape(name)}</td><td>{len(values)}</td><td>{html.escape(sample or '无')}</td></tr>")
    return (
        '<div class="review-priority">'
        "<h3>屏幕文字复核优先队列</h3>"
        "<table><thead><tr><th>分组</th><th>数量</th><th>样例</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _workflow_card(
    title: str,
    value: int | str,
    detail: str,
    args_path: str,
    samples: list[int],
    *,
    link: str | None = None,
    command: str | None = None,
    confirmed_args: str | None = None,
    link_label: str = "打开导出 Markdown",
) -> str:
    sample_text = ""
    if samples:
        preview = ", ".join(str(value) for value in samples[:10])
        suffix = " ..." if len(samples) > 10 else ""
        sample_text = f'<ul class="gap-list"><li>样例 index: {html.escape(preview + suffix)}</li></ul>'
    action = f'<a href="{html.escape(link)}">{html.escape(link_label)}</a>' if link else html.escape(detail)
    command_button = (
        f'<button type="button" data-copy="{html.escape(command)}">复制 MCP 命令</button>'
        if command
        else ""
    )
    confirmed = (
        f'<p class="hint">preflight 后可用确认版：<code>{html.escape(confirmed_args)}</code></p>'
        if confirmed_args
        else ""
    )
    return (
        '<div class="action">'
        f"<strong>{html.escape(title)}：{html.escape(str(value))}</strong>"
        f"<span>{action}</span>"
        f"<code>{html.escape(args_path)}</code>"
        f"{confirmed}"
        f"{command_button}"
        f"{sample_text}"
        "</div>"
    )


def _has_mapping(value: object) -> bool:
    return isinstance(value, dict) and any(v not in (None, "", [], {}) for v in value.values())


def _has_non_empty(value: object) -> bool:
    if isinstance(value, dict):
        return any(v not in (None, "", [], {}) for v in value.values())
    if isinstance(value, list):
        return any(item not in (None, "", [], {}) for item in value)
    return value not in (None, "", [], {})


def _timeline_for_review(package: dict[str, Any]) -> list[dict[str, Any]]:
    timeline = [dict(item) for item in package.get("timeline", []) if isinstance(item, dict)]
    segments = _review_transcript_segments(package)
    if not segments:
        return timeline
    for item in timeline:
        original = str(item.get("transcript") or "").strip()
        alignment = _align_item_to_transcript(item, segments)
        if alignment:
            if original:
                item.setdefault("original_transcript", original)
            item.update(alignment)
            item["transcript"] = str(alignment.get("review_transcript_excerpt") or "").strip()
            item["review_transcript_source"] = "source_arbitrated_transcript"
        else:
            item["review_start"] = _float_or_none(item.get("start")) or 0.0
            item["review_start_source"] = "visual_segment_start"
            if original:
                item.setdefault("original_transcript", original)
            notice = "（未找到可靠对齐的 ASR 语音；canonical 来源缺失，旧 timeline 文本仅保留为审计证据）"
            item["transcript"] = notice
            item["review_transcript_excerpt"] = notice
            item["review_transcript_source"] = "canonical_alignment_missing"
    return timeline


def _review_transcript_segments(package: dict[str, Any]) -> list[dict[str, Any]]:
    inline = package.get("review_transcript_segments")
    if isinstance(inline, list):
        return [_normalised_transcript_segment(row) for row in inline if isinstance(row, dict)]
    artifacts = package.get("review_artifacts") if isinstance(package.get("review_artifacts"), dict) else {}
    bundle_dir = str(artifacts.get("bundle_dir") or "").strip()
    if not bundle_dir:
        return []
    root = Path(bundle_dir).expanduser().resolve()
    candidates = []
    for key in (
        "source_arbitrated_transcript_json",
        "corrected_transcript_json",
        "normalized_transcript_json",
        "transcript_json",
    ):
        value = artifacts.get(key)
        if value:
            candidates.append(root / str(value))
    candidates.extend(
        [
            root / "source-arbitrated-transcript.json",
            root / "corrected-transcript.json",
            root / "normalized-transcript.json",
            root / "transcript.json",
        ]
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        rows = data.get("segments") if isinstance(data, dict) else []
        if isinstance(rows, list):
            segments = [_normalised_transcript_segment(row) for row in rows if isinstance(row, dict)]
            segments = [row for row in segments if row.get("text")]
            if segments:
                canonical = sorted(segments, key=lambda row: float(row.get("start") or 0))
                alignment = _validated_review_alignment_segments(
                    root, artifacts, canonical_path=path, canonical_segments=canonical
                )
                return alignment or canonical
    return []


def _validated_review_alignment_segments(
    root: Path,
    artifacts: dict[str, Any],
    *,
    canonical_path: Path,
    canonical_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    value = str(artifacts.get("qwen3_forced_alignment_json") or "").strip()
    if not value:
        return []
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return []
    if (
        not isinstance(data, dict)
        or str(data.get("schema") or "")
        != "video_knowledge_pipeline.qwen3_forced_aligner_output.v1"
        or str(data.get("status") or "") != "completed"
        or data.get("timestamps_monotonic") is not True
    ):
        return []
    transcript_value = str(data.get("transcript_path") or "").strip()
    if not transcript_value:
        return []
    if Path(transcript_value).expanduser().resolve() != canonical_path.resolve():
        return []
    rows = data.get("segments") if isinstance(data.get("segments"), list) else []
    aligned = [
        _normalised_transcript_segment(row)
        for row in rows
        if isinstance(row, dict)
    ]
    aligned = [row for row in aligned if row.get("text") and row.get("words")]
    canonical_text = _normalise_review_match_text(
        "".join(str(row.get("text") or "") for row in canonical_segments)
    )
    aligned_text = _normalise_review_match_text(
        "".join(str(row.get("text") or "") for row in aligned)
    )
    if not canonical_text or aligned_text != canonical_text:
        return []
    return sorted(aligned, key=lambda row: float(row.get("start") or 0))


def _normalised_transcript_segment(row: dict[str, Any]) -> dict[str, Any]:
    start = _float_or_none(row.get("start"))
    end = _float_or_none(row.get("end"))
    if start is None:
        start = 0.0
    if end is None or end < start:
        end = start
    result: dict[str, Any] = {
        "start": float(start),
        "end": float(end),
        "text": str(row.get("text") or "").strip(),
        "source_segment_ids": _review_source_segment_ids(row),
    }
    words = _review_word_timestamps(row)
    if words:
        result["words"] = words
    return result


def _align_item_to_transcript(
    item: dict[str, Any], segments: list[dict[str, Any]]
) -> dict[str, Any] | None:
    start = _float_or_none(item.get("start")) or 0.0
    end = _float_or_none(item.get("end"))
    if end is None or end <= start:
        end = start
    overlapping = []
    for row in segments:
        if (
            float(row.get("end") or 0) < start
            or float(row.get("start") or 0) > end
            or not str(row.get("text") or "").strip()
        ):
            continue
        if _transcript_overlap_is_reliable(row, start, end):
            overlapping.append(row)
            continue
        projection = _word_timed_review_projection(row, start=start, end=end)
        if projection:
            overlapping.append(projection)
    if not overlapping:
        return None
    excerpt = _normalise_review_match_text(
        item.get("corrected_transcript")
        or item.get("transcript")
        or item.get("text")
        or ""
    )
    matched = None
    if excerpt:
        for row in sorted(
            overlapping, key=lambda value: float(value.get("start") or 0)
        ):
            text = _normalise_review_match_text(row.get("text") or "")
            if _review_text_matches(text, excerpt):
                matched = row
                break
    exact = matched is not None
    first = matched or sorted(
        overlapping, key=lambda value: float(value.get("start") or 0)
    )[0]
    joined = " ".join(
        str(row.get("text") or "").strip()
        for row in sorted(
            overlapping, key=lambda value: float(value.get("start") or 0)
        )
        if str(row.get("text") or "").strip()
    )
    source_segment_ids = []
    for row in overlapping:
        for source_id in _review_source_segment_ids(row):
            if source_id not in source_segment_ids:
                source_segment_ids.append(source_id)
    return {
        "review_start": max(0.0, float(first.get("start") or 0)),
        "review_start_source": str(first.get("review_start_source") or "")
        or ("asr_segment_start" if exact else "asr_overlap_start"),
        "review_transcript_excerpt": joined or str(first.get("text") or ""),
        "review_transcript_source_segment_ids": source_segment_ids,
    }


def _review_word_timestamps(row: dict[str, Any]) -> list[dict[str, Any]]:
    words = read_asr_word_timestamps(row)
    if not words and isinstance(row.get("metadata"), dict):
        words = read_asr_word_timestamps(row["metadata"])
    return [
        word
        for word in words
        if "start" in word
        and "end" in word
        and float(word.get("end") or 0) >= float(word.get("start") or 0)
    ]


def _review_source_segment_ids(row: dict[str, Any]) -> list[str]:
    values = row.get("source_segment_ids")
    if not isinstance(values, list):
        values = []
    cue_indexes = row.get("source_cue_indexes")
    if not isinstance(cue_indexes, list):
        cue_indexes = []
    candidates = [
        *values,
        *(f"cue-index:{value}" for value in cue_indexes),
        row.get("id"),
        row.get("segment_id"),
        row.get("source_segment_id"),
    ]
    return list(dict.fromkeys(str(value) for value in candidates if str(value or "").strip()))


def _word_timed_review_projection(
    row: dict[str, Any], *, start: float, end: float
) -> dict[str, Any] | None:
    words = _review_word_timestamps(row)
    if not words:
        return None
    cue_text = _normalise_review_match_text(row.get("text") or "")
    timed_text = _normalise_review_match_text(_join_review_words(words))
    if not cue_text or timed_text != cue_text:
        return None
    selected = [
        word
        for word in words
        if start
        <= (
            float(word.get("start") or 0)
            + float(word.get("end") or word.get("start") or 0)
        )
        / 2
        < end
    ]
    if not selected:
        return None
    return {
        "start": float(selected[0].get("start") or start),
        "end": float(selected[-1].get("end") or end),
        "text": _join_review_words(selected),
        "source_segment_ids": _review_source_segment_ids(row),
        "review_start_source": "asr_word_timestamp_start",
    }


def _join_review_words(words: list[dict[str, Any]]) -> str:
    result = ""
    for row in words:
        token = str(row.get("word") or row.get("text") or "").strip()
        if not token:
            continue
        if (
            result
            and result[-1].isascii()
            and result[-1].isalnum()
            and token[0].isascii()
            and token[0].isalnum()
        ):
            result += " "
        result += token
    return result


def _transcript_overlap_is_reliable(row: dict[str, Any], start: float, end: float) -> bool:
    cue_start = float(row.get("start") or 0)
    cue_end = float(row.get("end") or cue_start)
    duration = max(0.001, cue_end - cue_start)
    overlap = max(0.0, min(cue_end, end) - max(cue_start, start))
    segment_duration = max(0.001, end - start)
    # Avoid assigning a long ASR segment to a short visual frame window. This
    # is the common source of "timestamp says 0-4s but text is from nearby speech".
    if (overlap / duration) >= 0.6:
        return True
    # Only allow segment-coverage fallback for short cues. Long cues that spill
    # far outside the visual window must stay unbound and be reviewed manually.
    return duration <= segment_duration * 1.5 and (overlap / segment_duration) >= 0.75


def _normalise_review_match_text(value: Any) -> str:
    text = str(value or "").lower()
    return "".join(ch for ch in text if ch.isalnum() or "一" <= ch <= "鿿")


def _review_text_matches(cue_text: str, excerpt: str) -> bool:
    if not cue_text or not excerpt:
        return False
    if len(cue_text) >= 4 and cue_text in excerpt:
        return True
    if len(excerpt) >= 4 and excerpt in cue_text:
        return True
    return False


def _transcript_source_chip(source: str) -> str:
    if not source:
        return ""
    return f'<span class="chip">time:{html.escape(source)}</span>'


def _review_video_panel(package: dict[str, Any]) -> str:
    media_path = _review_media_path(package)
    media_src = _file_uri(media_path)
    escaped_path = html.escape(media_path or "未找到原视频路径")
    src_attr = f' src="{html.escape(media_src)}"' if media_src else ""
    return f"""
        <section class="review-video">
          <h2>视频审核</h2>
          <div class="video-status">原视频：<code>{escaped_path}</code></div>
          <video id="review-video-player" controls preload="metadata"{src_attr}></video>
          <div class="video-size-control"><span>视频栏宽</span><input id="review-sidebar-width-slider" type="range" min="280" max="900" step="20" value="420" oninput="setReviewSidebarWidth(this.value)"><span id="review-sidebar-width-label">420px</span></div>
          <div class="video-size-control"><span>视频高度</span><input id="review-video-height-slider" type="range" min="160" max="720" step="20" value="220" oninput="setReviewVideoHeight(this.value)"><span id="review-video-height-label">220px</span></div>
          <div class="video-actions">
            <label><button type="button" onclick="document.getElementById('review-video-picker').click()">选择本地视频文件</button><input id="review-video-picker" type="file" accept="video/*" style="display:none" onchange="loadReviewVideo(this)"></label>
            <button type="button" onclick="jumpToFirstVisibleReviewItem()">跳到当前第一条</button>
            <button type="button" onclick="toggleReviewVideoWide()">切换宽屏审核</button>
          </div>
          <div id="review-video-status" class="video-status">点击时间戳会跳到视频对应位置并高亮审核条目。若浏览器不能直接读取 file:// 视频，请点“选择本地视频文件”。</div>
        </section>
"""


def review_media_reference(package: dict[str, Any]) -> str:
    """Return the registered media reference already used by lecture review.

    This intentionally performs no directory scan. Bundle consumers can reuse
    the same source precedence and then resolve the returned reference relative
    to the owning manifest/package.
    """

    artifacts = package.get("review_artifacts") if isinstance(package.get("review_artifacts"), dict) else {}
    for key in ("media_path", "video_path", "multimodal_sample_review_media_path"):
        value = artifacts.get(key) or package.get(key)
        if value:
            return str(value)
    for source in package.get("sources") or []:
        if not isinstance(source, dict):
            continue
        for key in ("media_path", "path", "source_path", "video_path"):
            value = source.get(key)
            if value:
                return str(value)
    for item in package.get("timeline") or []:
        if not isinstance(item, dict):
            continue
        for key in ("media_path", "video_path", "source_video_path", "video_key"):
            value = item.get(key)
            if value and str(value).lower().endswith((".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v")):
                return str(value)
    return ""


def resolve_review_media_path(bundle_dir: str | Path, manifest: dict[str, Any]) -> Path | None:
    """Resolve review media from a Bundle manifest or its source package.

    Intent: keep every local review surface on the lecture package's existing
    media-selection contract. Decision: inspect only registered manifest and
    ``source_package`` references. Reason: scanning nearby folders can bind the
    wrong interview or a stale transcode. Evidence: real VKP interview Bundles
    register media under ``sources[].path`` in lecture-package.json rather than
    ``manifest.media_path``. Effective scope: local review playback only.
    """

    root = Path(bundle_dir).expanduser().resolve()
    containers: list[tuple[dict[str, Any], Path]] = [(manifest, root)]
    package_ref = str(manifest.get("source_package") or "").strip()
    if package_ref:
        package_path = Path(package_ref).expanduser()
        package_path = package_path if package_path.is_absolute() else root / package_path
        package_path = package_path.resolve()
        if package_path.is_file():
            package = read_json(package_path)
            if isinstance(package, dict):
                containers.append((package, package_path.parent))
    for container, base in containers:
        raw = review_media_reference(container).strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        path = path if path.is_absolute() else base / path
        path = path.resolve()
        if path.is_file():
            return path
    return None


def _review_media_path(package: dict[str, Any]) -> str:
    """Backward-compatible private alias for existing rendered review pages."""

    return review_media_reference(package)


def _source_row(source: dict[str, Any]) -> str:
    title = html.escape(str(source.get("title", "")))
    video_id = html.escape(str(source.get("video_id", "")))
    path = html.escape(str(source.get("path", "")))
    count = html.escape(str(source.get("segment_count", 0)))
    artifact_count = html.escape(str(source.get("source_artifact_count", 0)))
    artifacts = _source_artifacts_html(source)
    return f'<div class="source"><strong>{title}</strong><code>{video_id}</code><code>{path}</code><span>{count} segments</span><span>{artifact_count} source artifacts</span>{artifacts}</div>'


def _source_markdown(source: dict[str, Any]) -> str:
    parts = [
        f"- `{source.get('video_id', '')}` {source.get('title', '')} "
        f"({source.get('segment_count', 0)} segments, {source.get('source_artifact_count', 0)} source artifacts) "
        f"`{source.get('path', '')}`"
    ]
    source_artifacts = source.get("source_artifacts") if isinstance(source.get("source_artifacts"), dict) else {}
    for artifact in source_artifacts.get("artifacts") or []:
        if not isinstance(artifact, dict) or not (artifact.get("exists") or artifact.get("copied_path")):
            continue
        copied = str(artifact.get("copied_path") or "").strip()
        original = str(artifact.get("path") or "").strip()
        location = copied or original
        parts.append(f"  - {artifact.get('label', artifact.get('key', 'artifact'))}: `{location}`")
    return "\n".join(parts)


def _source_artifacts_html(source: dict[str, Any]) -> str:
    source_artifacts = source.get("source_artifacts") if isinstance(source.get("source_artifacts"), dict) else {}
    rows = []
    for artifact in source_artifacts.get("artifacts") or []:
        if not isinstance(artifact, dict) or not (artifact.get("exists") or artifact.get("copied_path")):
            continue
        label = html.escape(str(artifact.get("label") or artifact.get("key") or "artifact"))
        path = html.escape(str(artifact.get("copied_path") or artifact.get("path") or ""))
        rows.append(f"<li><span>{label}</span><code>{path}</code></li>")
    if not rows:
        return ""
    return f'<ul class="source-artifacts">{"".join(rows)}</ul>'


def _timeline_card(index: int, item: dict[str, Any]) -> str:
    start_seconds = float(item.get("review_start") or item.get("start") or 0)
    start = format_timestamp(float(item.get("start", 0)))
    end = format_timestamp(float(item.get("end", 0)))
    time_label = f"{start} - {end}"
    jump_time = _seconds_js(start_seconds)
    route = str(item.get("visual_route") or "").strip()
    route_chip = f'<span class="chip">route:{html.escape(route)}</span>' if route else ""
    gap_html = _visual_gap_html(item)
    material = "".join(f'<span class="chip">{html.escape(str(value))}</span>' for value in item.get("material_types", []))
    signals = "".join(f'<span class="chip">{html.escape(str(value))}</span>' for value in item.get("signals", []))
    transcript_source = str(item.get("review_start_source") or "")
    transcript_for_review = str(item.get("review_transcript_excerpt") or item.get("transcript") or "（缺失或未命中）")
    transcript = html.escape(transcript_for_review)
    visual_text = html.escape(str(item.get("visual_text") or "（缺失或未命中）"))
    segment_ids = ",".join(str(value) for value in item.get("source_segment_ids", []))
    human_review = item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
    review_status = str(item.get("review_status") or human_review.get("status") or "pending")
    review_label = _review_status_label(review_status)
    review_class = "reviewed" if review_status in {"reviewed", "accepted"} else "needs-review"
    notes = html.escape(str(human_review.get("comment") or human_review.get("notes") or ""))
    frames = "\n".join(_frame_html(path) for path in item.get("frame_paths", [])) or '<div class="frame">无关键帧</div>'
    temporal_frames = "\n".join(_frame_html(path) for path in item.get("temporal_frame_paths", []))
    temporal_frame_section = f"<h3>连续帧组</h3>{temporal_frames}" if temporal_frames else ""
    structured_visual = _structured_visual_html(item)
    visual_understanding = _visual_understanding_html(item)
    temporal_understanding = _temporal_understanding_html(item)
    quality_issues = _quality_issues(item)
    quality_html = _quality_issues_html(quality_issues)
    timeline_alignment = item.get("timeline_alignment") if isinstance(item.get("timeline_alignment"), dict) else {}
    timeline_alignment_chip = _timeline_alignment_chip(timeline_alignment)
    timeline_alignment_html = _timeline_alignment_html(timeline_alignment)
    timeline_alignment_issues = [str(value) for value in timeline_alignment.get("issues") or []]
    evidence_paths = _item_evidence_paths(item)
    suggested_visual_text = html.escape(str(human_review.get("corrected_visual_text") or item.get("human_corrected_visual_text") or ""))
    suggested_visual_understanding = html.escape(_json_text(human_review.get("corrected_visual_understanding") or item.get("human_corrected_visual_understanding") or {}))
    suggested_temporal_understanding = html.escape(
        _json_text(human_review.get("corrected_temporal_visual_understanding") or item.get("human_corrected_temporal_visual_understanding") or {})
    )
    suggested_transcript = html.escape(str(human_review.get("corrected_transcript") or item.get("human_corrected_transcript") or ""))
    human_key_point_confirmed = bool(human_review.get("human_key_point_confirmed"))
    human_key_point_text = html.escape(str(human_review.get("human_key_point_text") or ""))
    raw_human_key_point_aliases = human_review.get("human_key_point_aliases") or []
    if isinstance(raw_human_key_point_aliases, str):
        raw_human_key_point_aliases = raw_human_key_point_aliases.split("|")
    elif not isinstance(raw_human_key_point_aliases, list):
        raw_human_key_point_aliases = []
    human_key_point_aliases = html.escape(
        "|".join(str(value).strip() for value in raw_human_key_point_aliases if str(value).strip())
    )
    semantic_candidates = _transcript_semantic_candidates_html(item)
    low_confidence = _has_low_confidence(item, quality_issues)
    keep_image = _should_keep_image(item, human_review)
    corrected = _has_human_correction(item, human_review)
    needs_human_review = bool(item.get("needs_human_review", True))
    return f"""
<article class="item" data-index="{index}" data-time="{html.escape(time_label)}" data-segment-ids="{html.escape(segment_ids)}" data-review-status="{html.escape(review_status)}" data-needs-human-review="{str(needs_human_review).lower()}" data-quality-issues="{html.escape(','.join(quality_issues))}" data-timeline-alignment-issues="{html.escape(','.join(timeline_alignment_issues))}" data-visual-route="{html.escape(route)}" data-low-confidence="{str(low_confidence).lower()}" data-keep-image="{str(keep_image).lower()}" data-corrected="{str(corrected).lower()}" data-evidence-paths="{html.escape('|'.join(evidence_paths))}">
  <div class="item-header">
    <div class="time"><button type="button" class="jump-time" onclick="jumpToTimelineItem({index}, {jump_time})">{html.escape(time_label)}</button></div>
    <div class="chips">{route_chip}<span class="chip">review:{html.escape(review_status)}</span>{_transcript_source_chip(transcript_source)}{timeline_alignment_chip}{material}{signals}</div>
    <div class="{review_class}">{review_label}</div>
  </div>
  <div class="item-body">
    <div>
      {gap_html}
      {quality_html}
      {timeline_alignment_html}
      <section class="section">
        <h3>口语/字幕</h3>
        <pre>{transcript}</pre>
      </section>
      <section class="section">
        <h3>画面文字 / OCR / 视觉观察</h3>
        <pre>{visual_text}</pre>
      </section>
      {structured_visual}
      {visual_understanding}
      {temporal_understanding}
      <section class="section checks">
        <h3>人工复核</h3>
        <label><input type="checkbox" class="review-check" onchange="applyFilter()" {_checked(review_status in {"reviewed", "accepted"} or human_review.get("reviewed"))}> 已确认这一段</label>
        <label><input type="checkbox" class="keep-images" {_checked(keep_image)}> 必须保留图片，不能完全降维成文字</label>
        <label><input type="checkbox" class="asr-ocr-error" {_checked(human_review.get("asr_ocr_error"))}> 存在 ASR/OCR 错误</label>
        <label><input type="checkbox" class="missing-info" {_checked(human_review.get("missing_info"))}> 疑似有遗漏信息</label>
        <textarea class="review-comment" placeholder="补充修正、遗漏、概念拆分、图片保留原因">{notes}</textarea>
      </section>
      <section class="section review-editor">
        <h3>审核 JSON 草稿字段</h3>
        <label><input type="checkbox" class="include-review-row"> 选入审核 JSON 草稿</label>
        <label>纠正后逐字稿 <code>corrected_transcript</code>
          <textarea class="corrected-transcript" data-json-field="corrected_transcript" placeholder="仅在需要替换这一整条时间线逐字稿时填写；普通备注请填上方备注框">{suggested_transcript}</textarea>
          <span class="field-hint">这个字段保留完整的人工作品；若下方存在候选，请优先填写候选纠正，才能精确绑定并刷新正式逐字稿。</span>
        </label>
        {semantic_candidates}
        <div class="human-key-point-editor">
          <strong>智能总结人工关键点</strong>
          <label><input type="checkbox" class="human-key-point-confirmed" {_checked(human_key_point_confirmed)}> 明确确认本段是总结必须覆盖的关键点</label>
          <label>关键点原意
            <textarea class="human-key-point-text" placeholder="用自己的话写出本段必须进入总结的原意">{human_key_point_text}</textarea>
          </label>
          <label>语义等价别名（用 | 分隔）
            <input class="human-key-point-aliases" value="{human_key_point_aliases}" placeholder="例如：儿童方案不含身故责任|孩子不带身故">
          </label>
          <span class="field-hint">只有人工勾选并正式“保存到 VKP”后才会进入 human-key-points.json；候选或模型输出不能自行转正。</span>
        </div>
        <label>corrected_visual_text
          <textarea class="corrected-visual-text" data-json-field="corrected_visual_text" placeholder="修正或补齐画面文字 / OCR">{suggested_visual_text}</textarea>
        </label>
        <label>corrected_visual_understanding
          <textarea class="corrected-visual-understanding" data-json-field="corrected_visual_understanding" placeholder='可填 JSON 对象；普通文字会自动包成 {{"human_summary": "..."}}'>{suggested_visual_understanding}</textarea>
        </label>
        <label>corrected_temporal_visual_understanding
          <textarea class="corrected-temporal-visual-understanding" data-json-field="corrected_temporal_visual_understanding" placeholder='可填 JSON 对象；普通文字会自动包成 {{"human_summary": "..."}}'>{suggested_temporal_understanding}</textarea>
        </label>
        <div class="quick-statuses" aria-label="快速审核状态">
          <button type="button" data-quick-status="accepted_known_gap" onclick="setQuickReviewStatus(this, 'accepted_known_gap')">accepted_known_gap</button>
          <button type="button" data-quick-status="keep_image" onclick="setQuickReviewStatus(this, 'keep_image')">keep_image</button>
          <button type="button" data-quick-status="needs_rerun_ocr" onclick="setQuickReviewStatus(this, 'needs_rerun_ocr')">needs_rerun_ocr</button>
          <button type="button" data-quick-status="corrected_transcript" onclick="setQuickReviewStatus(this, 'corrected_transcript')">corrected_transcript</button>
          <button type="button" data-quick-status="corrected_visual_text" onclick="setQuickReviewStatus(this, 'corrected_visual_text')">corrected_visual_text</button>
        </div>
        <div class="snippet-actions">
          <button type="button" onclick="copyReviewSnippet(this)">复制这一段 JSON</button>
        </div>
      </section>
    </div>
    <div class="frames">
      <h3>关键帧</h3>
      {frames}
      {temporal_frame_section}
    </div>
  </div>
</article>
"""


def _transcript_semantic_candidates_html(item: dict[str, Any]) -> str:
    rows = item.get("transcript_semantic_candidates") if isinstance(item.get("transcript_semantic_candidates"), list) else []
    if not rows:
        return ""
    cards: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate_id = html.escape(str(row.get("candidate_id") or ""), quote=True)
        original = html.escape(str(row.get("original_text") or ""), quote=True)
        evidence_ids = html.escape("|".join(str(value) for value in row.get("evidence_ids") or []), quote=True)
        corrected = html.escape(str(row.get("corrected_text") or row.get("suggested_text") or ""))
        time_range = html.escape(str(row.get("time_range") or ""))
        applied = '<span class="chip reviewed">已应用</span>' if row.get("applied") else '<span class="chip needs-review">待确认</span>'
        cards.append(
            f'<div class="transcript-candidate" data-candidate-id="{candidate_id}" data-original-text="{original}" data-evidence-ids="{evidence_ids}">'
            f'<strong>逐字稿候选 <code>{candidate_id}</code> {applied}</strong>'
            f'<span class="field-hint">时间 {time_range}；原文：<code>{original}</code></span>'
            f'<textarea class="semantic-corrected-text" placeholder="填写该候选的准确替换文字">{corrected}</textarea>'
            '<span class="field-hint">保存到 VKP 时会绑定 candidate_id、原文和证据，经严格校验后刷新正式逐字稿。</span>'
            '</div>'
        )
    return "".join(cards)



def _timeline_alignment_chip(alignment: dict[str, Any]) -> str:
    issues = [str(value) for value in alignment.get("issues") or []]
    if not issues:
        return ""
    label = f"time-align:{len(issues)}"
    return f'<span class="chip timeline-alignment-chip">{html.escape(label)}</span>'


def _timeline_alignment_html(alignment: dict[str, Any]) -> str:
    issues = [str(value) for value in alignment.get("issues") or []]
    if not issues:
        return ""
    issue_rows = "".join(f"<li><code>{html.escape(issue)}</code> {html.escape(_timeline_alignment_issue_label(issue))}</li>" for issue in issues)
    tagger_times = alignment.get("tagger_times") if isinstance(alignment.get("tagger_times"), list) else []
    tagger_label = ", ".join(format_timestamp(float(value)) for value in tagger_times if _float_or_none(value) is not None) or "-"
    review_start = _format_optional_seconds(alignment.get("review_start"))
    asr_start = _format_optional_seconds(alignment.get("asr_first_start"))
    frame_time = _format_optional_seconds(alignment.get("frame_time"))
    suggested = _format_optional_seconds(alignment.get("suggested_review_start"))
    asr_excerpt = html.escape(str(alignment.get("asr_excerpt") or ""))
    suggestion = html.escape(str(alignment.get("suggestion") or "Preview only; confirm against video before editing review_start."))
    return f"""
      <section class="section timeline-alignment-warning">
        <h3>时间轴对齐风险</h3>
        <ul class="quality-list">{issue_rows}</ul>
        <table class="mini-table"><tbody>
          <tr><th>当前审核跳转</th><td><code>{html.escape(review_start)}</code></td></tr>
          <tr><th>ASR 建议起点</th><td><code>{html.escape(asr_start)}</code></td></tr>
          <tr><th>抽帧/截图时间</th><td><code>{html.escape(frame_time)}</code></td></tr>
          <tr><th>打标器时间</th><td><code>{html.escape(tagger_label)}</code></td></tr>
          <tr><th>建议 review_start</th><td><code>{html.escape(suggested)}</code></td></tr>
        </tbody></table>
        <p class="hint">{suggestion}</p>
        <pre>{asr_excerpt}</pre>
      </section>
"""


def _timeline_alignment_issue_label(issue: str) -> str:
    return {
        "missing_asr_overlap": "没有找到重叠 ASR 段，当前时间片可能缺少可核对语音。",
        "review_start_outside_segment": "审核跳转点落在当前 timeline 段外。",
        "review_start_mismatch": "审核跳转点和 ASR 起点差距超过容差。",
        "frame_time_outside_segment": "抽帧时间落在当前 timeline 段外。",
        "tagger_time_conflict": "青龙打标器时间与当前段或 ASR 对齐存在冲突。",
    }.get(issue, "需要人工核对时间轴。")


def _format_optional_seconds(value: Any) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return "-"
    return format_timestamp(float(numeric))

def _seconds_js(value: float) -> str:
    try:
        return f"{max(0.0, float(value)):.3f}".rstrip("0").rstrip(".") or "0"
    except Exception:
        return "0"


def _review_status_label(status: str) -> str:
    return {
        "accepted": "已确认",
        "reviewed": "已确认",
        "needs_revision": "需修订",
        "needs_fix": "需修订",
        "keep_image": "保留图片",
        "corrected_visual_text": "已修正画面文字",
        "corrected_visual_understanding": "已修正单帧理解",
        "corrected_temporal_visual_understanding": "已修正连续理解",
    }.get(status, "待复核")


def _quality_issues_html(issues: list[str]) -> str:
    if not issues:
        return ""
    rows = "".join(f"<li>{html.escape(_quality_issue_label(issue))} <code>{html.escape(issue)}</code></li>" for issue in issues)
    return f"""
      <section class="section">
        <h3>质量缺口</h3>
        <ul class="quality-list">{rows}</ul>
      </section>
"""


def _item_evidence_paths(item: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("frame_paths", "temporal_frame_paths", "original_frame_paths"):
        value = item.get(key)
        if isinstance(value, list):
            paths.extend(str(path) for path in value if str(path))
    for asset in item.get("assets") or []:
        if isinstance(asset, dict):
            paths.append(str(asset.get("path") or asset.get("source") or ""))
    return _dedupe([path for path in paths if path])


def _has_low_confidence(item: dict[str, Any], issues: list[str]) -> bool:
    if any(issue in {"low_ocr_confidence", "model_output_parse_failed"} for issue in issues):
        return True
    route_confidence = _float_or_none(item.get("visual_route_confidence"))
    if route_confidence is not None and route_confidence < 0.6:
        return True
    for key in ("visual_understanding", "temporal_visual_understanding"):
        payload = item.get(key) if isinstance(item.get(key), dict) else {}
        confidence = payload.get("confidence") if isinstance(payload, dict) else None
        numeric = _float_or_none(confidence)
        if numeric is not None and numeric < 0.65:
            return True
        if str(confidence or "").strip().lower() in {"low", "低", "低置信度", "uncertain"}:
            return True
    return False


def _should_keep_image(item: dict[str, Any], human_review: dict[str, Any]) -> bool:
    if human_review.get("keep_images") or human_review.get("keep_image") or item.get("human_keep_image"):
        return True
    retention = item.get("visual_retention") if isinstance(item.get("visual_retention"), dict) else {}
    return str(retention.get("recommendation") or "") in {"keep_image", "review_image"}


def _has_human_correction(item: dict[str, Any], human_review: dict[str, Any]) -> bool:
    if str(item.get("review_status") or "").startswith("corrected_"):
        return True
    correction_keys = [
        "corrected_transcript",
        "corrected_visual_text",
        "corrected_visual_understanding",
        "corrected_temporal_visual_understanding",
    ]
    if any(_has_non_empty(human_review.get(key)) for key in correction_keys):
        return True
    return any(
        _has_non_empty(item.get(key))
        for key in [
            "human_corrected_transcript",
            "human_corrected_visual_text",
            "human_corrected_visual_understanding",
            "human_corrected_temporal_visual_understanding",
        ]
    )


def _json_text(value: object) -> str:
    if not _has_non_empty(value):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _frame_html(path: str) -> str:
    escaped_path = html.escape(str(path))
    suffix = Path(path).suffix.lower()
    image = f'<img src="{escaped_path}" alt="关键帧">' if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"} else ""
    return f'<div class="frame">{image}<code>{escaped_path}</code></div>'


def _checked(value: object) -> str:
    return "checked" if bool(value) else ""
