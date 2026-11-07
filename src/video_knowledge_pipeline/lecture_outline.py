from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import ensure_project_dirs, read_json, write_json
from .transcript import format_timestamp


LECTURE_OUTLINE_SCHEMA = "lecture_outline.v1"
STRUCTURED_TYPES = {"formula", "table", "code"}


def build_lecture_outline(root: str | Path) -> dict[str, Any]:
    """Build a navigation outline from the full lecture timeline without summarizing it."""
    paths = ensure_project_dirs(root)
    package_path = paths["lecture_packages"] / "lecture-package.json"
    if not package_path.exists():
        raise FileNotFoundError(f"lecture package not found: {package_path}")
    package = read_json(package_path)
    if not isinstance(package, dict):
        raise ValueError("lecture package must be a JSON object")
    return write_lecture_outline(paths, package)


def write_lecture_outline(paths: dict[str, Path], package: dict[str, Any]) -> dict[str, Any]:
    outline = generate_lecture_outline(package)
    json_path = paths["lecture_packages"] / "lecture-outline.json"
    markdown_path = paths["notes"] / "lecture-outline.md"
    write_json(json_path, outline)
    markdown_path.write_text(render_lecture_outline_markdown(outline), encoding="utf-8")
    return {
        "outline_path": str(json_path),
        "outline_markdown_path": str(markdown_path),
        "outline": outline,
    }


def generate_lecture_outline(package: dict[str, Any], *, gap_threshold_seconds: float = 45.0) -> dict[str, Any]:
    timeline = package.get("timeline") if isinstance(package.get("timeline"), list) else []
    rows = [
        _outline_entry(index, item)
        for index, item in enumerate(timeline, start=1)
        if isinstance(item, dict)
    ]
    chapters = _chapterize(rows, gap_threshold_seconds=gap_threshold_seconds)
    return {
        "schema": LECTURE_OUTLINE_SCHEMA,
        "title": str(package.get("title") or "Lecture Outline"),
        "created_at": now_iso(),
        "source_package_schema": str(package.get("schema") or ""),
        "timeline_count": len(rows),
        "chapter_count": len(chapters),
        "gap_threshold_seconds": gap_threshold_seconds,
        "chapters": chapters,
    }


def render_lecture_outline_markdown(outline: dict[str, Any]) -> str:
    lines = [
        "---",
        "type: lecture-outline",
        f'title: "{outline.get("title", "Lecture Outline")}"',
        "tags: [lecture-video, outline, knowledge-package]",
        f'created: "{outline.get("created_at", now_iso())}"',
        "---",
        "",
        f"# 课程导航大纲：{outline.get('title', 'Lecture Outline')}",
        "",
        "> 这不是摘要。它只把全量时间线分组为可导航章节，每个条目仍指回原始片段、时间戳、转写、OCR/视觉证据和复核状态。",
        "",
        "## 总览",
        "",
        f"- 全量时间线片段：{outline.get('timeline_count', 0)}",
        f"- 章节段落：{outline.get('chapter_count', 0)}",
        f"- 分章时间间隔阈值：{outline.get('gap_threshold_seconds', 0)} 秒",
        "",
        "## 目录",
        "",
    ]
    chapters = outline.get("chapters") if isinstance(outline.get("chapters"), list) else []
    if not chapters:
        lines.append("暂无章节。")
        return "\n".join(lines).rstrip() + "\n"
    for chapter in chapters:
        lines.append(
            f"- {chapter.get('chapter_id', '')} {chapter.get('title', '')} "
            f"`{chapter.get('time_label', '')}` "
            f"片段 {chapter.get('item_count', 0)} / 待复核 {chapter.get('pending_review_count', 0)}"
        )
    for chapter in chapters:
        lines.extend(["", f"## {chapter.get('chapter_id', '')} {chapter.get('title', '')}", ""])
        lines.extend(
            [
                f"- 时间：{chapter.get('time_label', '')}",
                f"- 片段数：{chapter.get('item_count', 0)}",
                f"- 待复核：{chapter.get('pending_review_count', 0)}",
                f"- 视觉需确认/保留：{chapter.get('visual_review_count', 0)}",
                f"- 材料类型：{_format_counts(chapter.get('material_counts'))}",
                "",
            ]
        )
        for entry in chapter.get("entries") or []:
            if isinstance(entry, dict):
                lines.extend(_entry_markdown(entry))
    return "\n".join(lines).rstrip() + "\n"


def _chapterize(entries: list[dict[str, Any]], *, gap_threshold_seconds: float) -> list[dict[str, Any]]:
    chapters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for entry in entries:
        if previous and _starts_new_chapter(previous, entry, gap_threshold_seconds=gap_threshold_seconds):
            chapters.append(current)
            current = []
        current.append(entry)
        previous = entry
    if current:
        chapters.append(current)
    return [_chapter_from_entries(index, rows) for index, rows in enumerate(chapters, start=1)]


def _starts_new_chapter(previous: dict[str, Any], entry: dict[str, Any], *, gap_threshold_seconds: float) -> bool:
    if entry.get("video_key") != previous.get("video_key"):
        return True
    gap = float(entry.get("start") or 0) - float(previous.get("end") or 0)
    if gap >= gap_threshold_seconds:
        return True
    if _heading_score(entry) >= 2 and len(str(entry.get("headline") or "")) >= 4:
        return True
    return False


def _chapter_from_entries(chapter_index: int, entries: list[dict[str, Any]]) -> dict[str, Any]:
    start = min(float(entry.get("start") or 0) for entry in entries)
    end = max(float(entry.get("end") or 0) for entry in entries)
    material_counts = _count_values(value for entry in entries for value in entry.get("material_types") or [])
    retention_counts = _count_values(str((entry.get("visual_retention") or {}).get("recommendation") or "unknown") for entry in entries)
    return {
        "chapter_id": f"C{chapter_index:02d}",
        "title": _chapter_title(chapter_index, entries),
        "start": round(start, 3),
        "end": round(end, 3),
        "time_label": f"{format_timestamp(start)} - {format_timestamp(end)}",
        "video_key": str(entries[0].get("video_key") or ""),
        "timeline_indices": [int(entry.get("index") or 0) for entry in entries],
        "source_segment_ids": _dedupe(str(value) for entry in entries for value in entry.get("source_segment_ids") or []),
        "item_count": len(entries),
        "pending_review_count": sum(1 for entry in entries if entry.get("needs_human_review") or entry.get("review_status") == "needs_revision"),
        "quality_issue_count": sum(1 for entry in entries if entry.get("quality_issues")),
        "visual_review_count": sum(
            1
            for entry in entries
            if (entry.get("visual_retention") or {}).get("recommendation") in {"keep_image", "review_image"}
        ),
        "material_counts": material_counts,
        "visual_retention_counts": retention_counts,
        "entries": entries,
    }


def _outline_entry(index: int, item: dict[str, Any]) -> dict[str, Any]:
    transcript = str(item.get("transcript") or "").strip()
    visual_text = str(item.get("visual_text") or "").strip()
    structured_visual = item.get("structured_visual") if isinstance(item.get("structured_visual"), list) else []
    material_types = [str(value) for value in item.get("material_types") or [] if str(value)]
    retention = item.get("visual_retention") if isinstance(item.get("visual_retention"), dict) else {}
    start = _float_value(item.get("start"))
    end = _float_value(item.get("end"))
    return {
        "index": index,
        "start": start,
        "end": end,
        "time_label": f"{format_timestamp(start)} - {format_timestamp(end)}",
        "video_key": str(item.get("video_key") or item.get("video_id") or ""),
        "source_segment_ids": [str(value) for value in item.get("source_segment_ids") or [] if str(value)],
        "headline": _headline(transcript, visual_text, material_types),
        "transcript": transcript,
        "visual_text": visual_text,
        "structured_visual_count": len([row for row in structured_visual if isinstance(row, dict)]),
        "has_frames": bool(item.get("frame_paths")),
        "frame_count": len([path for path in item.get("frame_paths") or [] if str(path)]),
        "material_types": material_types,
        "review_status": str(item.get("review_status") or "pending"),
        "needs_human_review": bool(item.get("needs_human_review", True)),
        "quality_issues": [str(value) for value in item.get("quality_issues") or [] if str(value)],
        "visual_retention": retention,
    }


def _entry_markdown(entry: dict[str, Any]) -> list[str]:
    retention = entry.get("visual_retention") if isinstance(entry.get("visual_retention"), dict) else {}
    lines = [
        f"### #{entry.get('index', '')} {entry.get('time_label', '')} {entry.get('headline', '')}",
        "",
        f"- 来源片段：{', '.join(entry.get('source_segment_ids') or [])}",
        f"- 材料类型：{', '.join(entry.get('material_types') or []) or 'unknown'}",
        f"- 复核状态：`{entry.get('review_status', 'pending')}`",
        f"- 视觉保留建议：`{retention.get('recommendation', 'unknown')}`",
        f"- 关键帧：{entry.get('frame_count', 0)}",
        f"- 结构化视觉条目：{entry.get('structured_visual_count', 0)}",
        "",
        "口语/字幕：",
        "",
        _clip(str(entry.get("transcript") or "（缺失或未命中）"), 360),
        "",
        "画面文字 / OCR / 视觉观察：",
        "",
        _clip(str(entry.get("visual_text") or "（缺失或未命中）"), 360),
        "",
    ]
    return lines


def _chapter_title(chapter_index: int, entries: list[dict[str, Any]]) -> str:
    for entry in entries[:3]:
        headline = str(entry.get("headline") or "").strip()
        if headline:
            return headline
    return f"章节 {chapter_index}"


def _headline(transcript: str, visual_text: str, material_types: list[str]) -> str:
    for text in [visual_text, transcript]:
        for line in str(text).splitlines():
            cleaned = _clean_heading(line)
            if cleaned:
                return cleaned
    material = ", ".join(material_types[:3])
    return f"未命名片段{f'（{material}）' if material else ''}"


def _clean_heading(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip(" -#\t")
    if not value:
        return ""
    value = re.sub(r"^#+\s*", "", value)
    value = value.strip()
    if len(value) > 80:
        value = value[:77].rstrip() + "..."
    return value


def _heading_score(entry: dict[str, Any]) -> int:
    text = f"{entry.get('headline', '')}\n{entry.get('visual_text', '')}\n{entry.get('transcript', '')}"
    score = 0
    if re.search(r"(第[一二三四五六七八九十\d]+[章节讲课部分]|^\d+[.、]\s*|^[一二三四五六七八九十]+[、.])", text):
        score += 2
    if any(keyword in text for keyword in ["本节", "这一讲", "接下来", "下面我们", "总结", "小结", "Chapter", "Section"]):
        score += 1
    if set(entry.get("material_types") or []) & STRUCTURED_TYPES:
        score += 1
    return score


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "").strip()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _format_counts(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "unknown"
    return ", ".join(f"{key}={count}" for key, count in value.items())


def _dedupe(values: Any) -> list[str]:
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _float_value(value: Any) -> float:
    try:
        return round(float(value or 0), 3)
    except (TypeError, ValueError):
        return 0.0


def _clip(text: str, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."
