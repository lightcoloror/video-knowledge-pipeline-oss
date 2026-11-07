from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import ensure_project_dirs, read_json, write_json
from .transcript import format_timestamp


LECTURE_SEARCH_INDEX_SCHEMA = "lecture_search_index.v1"


def build_lecture_search_index(root: str | Path) -> dict[str, Any]:
    """Build a local text search index over the full lecture timeline."""
    paths = ensure_project_dirs(root)
    package_path = paths["lecture_packages"] / "lecture-package.json"
    if not package_path.exists():
        raise FileNotFoundError(f"lecture package not found: {package_path}")
    package = read_json(package_path)
    if not isinstance(package, dict):
        raise ValueError("lecture package must be a JSON object")
    return write_lecture_search_index(paths, package)


def write_lecture_search_index(paths: dict[str, Path], package: dict[str, Any]) -> dict[str, Any]:
    index = generate_lecture_search_index(package)
    json_path = paths["lecture_packages"] / "lecture-search-index.json"
    markdown_path = paths["notes"] / "lecture-search-index.md"
    write_json(json_path, index)
    markdown_path.write_text(render_lecture_search_index_markdown(index), encoding="utf-8")
    return {
        "search_index_path": str(json_path),
        "search_index_markdown_path": str(markdown_path),
        "search_index": index,
    }


def search_lecture_index(root: str | Path, query: str, *, limit: int = 8) -> dict[str, Any]:
    """Search the persisted lecture-search-index.json with deterministic local scoring."""
    paths = ensure_project_dirs(root)
    index_path = paths["lecture_packages"] / "lecture-search-index.json"
    if not index_path.exists():
        package_path = paths["lecture_packages"] / "lecture-package.json"
        if not package_path.exists():
            raise FileNotFoundError(f"lecture package not found: {package_path}")
        package = read_json(package_path)
        if not isinstance(package, dict):
            raise ValueError("lecture package must be a JSON object")
        index = generate_lecture_search_index(package)
        write_json(index_path, index)
    else:
        index = read_json(index_path)
    if not isinstance(index, dict):
        raise ValueError("lecture search index must be a JSON object")
    results = search_lecture_entries(index, query, limit=limit)
    return {
        "schema": "lecture_search_results.v1",
        "query": query,
        "limit": limit,
        "count": len(results),
        "results": results,
    }


def generate_lecture_search_index(package: dict[str, Any]) -> dict[str, Any]:
    timeline = package.get("timeline") if isinstance(package.get("timeline"), list) else []
    entries = [
        _search_entry(index, item)
        for index, item in enumerate(timeline, start=1)
        if isinstance(item, dict)
    ]
    vocabulary: dict[str, int] = {}
    for entry in entries:
        for token in set(entry.get("tokens") or []):
            vocabulary[token] = vocabulary.get(token, 0) + 1
    return {
        "schema": LECTURE_SEARCH_INDEX_SCHEMA,
        "title": str(package.get("title") or "Lecture Search Index"),
        "created_at": now_iso(),
        "source_package_schema": str(package.get("schema") or ""),
        "entry_count": len(entries),
        "vocabulary_size": len(vocabulary),
        "entries": entries,
    }


def search_lecture_entries(index: dict[str, Any], query: str, *, limit: int = 8) -> list[dict[str, Any]]:
    terms = _query_terms(query)
    if not terms:
        return []
    scored = []
    for entry in index.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        score, matched = _score_entry(entry, terms)
        if score <= 0:
            continue
        scored.append(
            {
                "score": score,
                "matched_terms": matched,
                "index": entry.get("index", 0),
                "time_label": entry.get("time_label", ""),
                "headline": entry.get("headline", ""),
                "snippet": _snippet(str(entry.get("search_text") or ""), matched),
                "source_segment_ids": entry.get("source_segment_ids") or [],
                "material_types": entry.get("material_types") or [],
                "review_status": entry.get("review_status", "pending"),
                "needs_human_review": bool(entry.get("needs_human_review", True)),
                "visual_retention": entry.get("visual_retention") or {},
            }
        )
    scored.sort(key=lambda row: (-int(row.get("score") or 0), int(row.get("index") or 0)))
    return scored[: max(int(limit), 0)]


def render_lecture_search_index_markdown(index: dict[str, Any]) -> str:
    lines = [
        "---",
        "type: lecture-search-index",
        f'title: "{index.get("title", "Lecture Search Index")}"',
        "tags: [lecture-video, search-index, knowledge-package]",
        f'created: "{index.get("created_at", now_iso())}"',
        "---",
        "",
        f"# 检索索引：{index.get('title', 'Lecture Search Index')}",
        "",
        "> 这是本地关键词检索索引，不是摘要。每条记录对应全量时间线片段，保留时间戳、来源片段、材料类型、复核状态和视觉保留判断。",
        "",
        f"- 条目数：{index.get('entry_count', 0)}",
        f"- 词项数：{index.get('vocabulary_size', 0)}",
        "",
        "## 条目",
        "",
    ]
    for entry in index.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        retention = entry.get("visual_retention") if isinstance(entry.get("visual_retention"), dict) else {}
        lines.extend(
            [
                f"### #{entry.get('index', '')} {entry.get('time_label', '')} {entry.get('headline', '')}",
                "",
                f"- 来源片段：{', '.join(entry.get('source_segment_ids') or [])}",
                f"- 材料类型：{', '.join(entry.get('material_types') or []) or 'unknown'}",
                f"- 复核状态：`{entry.get('review_status', 'pending')}`",
                f"- 视觉保留建议：`{retention.get('recommendation', 'unknown')}`",
                "",
                _clip(str(entry.get("search_text") or ""), 520),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _search_entry(index: int, item: dict[str, Any]) -> dict[str, Any]:
    transcript = str(item.get("transcript") or "").strip()
    visual_text = str(item.get("visual_text") or "").strip()
    structured = item.get("structured_visual") if isinstance(item.get("structured_visual"), list) else []
    structured_text = "\n".join(
        str(row.get("markdown") or row.get("text") or "").strip()
        for row in structured
        if isinstance(row, dict) and str(row.get("markdown") or row.get("text") or "").strip()
    )
    material_types = [str(value) for value in item.get("material_types") or [] if str(value)]
    start = _float_value(item.get("start"))
    end = _float_value(item.get("end"))
    headline = _headline(transcript, visual_text, structured_text, material_types)
    search_text = "\n".join(
        value
        for value in [
            headline,
            transcript,
            visual_text,
            structured_text,
            " ".join(material_types),
            " ".join(str(value) for value in item.get("signals") or [] if str(value)),
        ]
        if value
    )
    return {
        "index": index,
        "start": start,
        "end": end,
        "time_label": f"{format_timestamp(start)} - {format_timestamp(end)}",
        "video_key": str(item.get("video_key") or item.get("video_id") or ""),
        "source_segment_ids": [str(value) for value in item.get("source_segment_ids") or [] if str(value)],
        "headline": headline,
        "search_text": search_text,
        "tokens": _tokens(search_text),
        "material_types": material_types,
        "review_status": str(item.get("review_status") or "pending"),
        "needs_human_review": bool(item.get("needs_human_review", True)),
        "quality_issues": [str(value) for value in item.get("quality_issues") or [] if str(value)],
        "visual_retention": item.get("visual_retention") if isinstance(item.get("visual_retention"), dict) else {},
    }


def _score_entry(entry: dict[str, Any], terms: list[str]) -> tuple[int, list[str]]:
    text = str(entry.get("search_text") or "").lower()
    headline = str(entry.get("headline") or "").lower()
    tokens = set(entry.get("tokens") or [])
    score = 0
    matched = []
    for term in terms:
        lower = term.lower()
        if lower in headline:
            score += 8
            matched.append(term)
            continue
        if lower in text:
            score += 4
            matched.append(term)
            continue
        if lower in tokens:
            score += 2
            matched.append(term)
    if entry.get("needs_human_review"):
        score += 1
    if (entry.get("visual_retention") or {}).get("recommendation") in {"keep_image", "review_image"}:
        score += 1
    return score, _dedupe(matched)


def _query_terms(query: str) -> list[str]:
    quoted = re.findall(r'"([^"]+)"', str(query or ""))
    remainder = re.sub(r'"[^"]+"', " ", str(query or ""))
    return _dedupe([*_tokens(remainder), *[value.strip() for value in quoted if value.strip()]])


def _tokens(text: str) -> list[str]:
    value = str(text or "").lower()
    ascii_tokens = re.findall(r"[a-z0-9_+\-*/=]{2,}", value)
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", value)
    cjk_bigrams = []
    for token in cjk_tokens:
        cjk_bigrams.extend(token[index : index + 2] for index in range(max(len(token) - 1, 0)))
    return _dedupe([*ascii_tokens, *cjk_tokens, *cjk_bigrams])


def _headline(*values: Any) -> str:
    for value in values:
        if isinstance(value, list):
            text = ", ".join(str(item) for item in value if str(item))
        else:
            text = str(value or "")
        for line in text.splitlines():
            cleaned = re.sub(r"\s+", " ", line).strip(" -#\t")
            if cleaned:
                return _clip(cleaned, 80)
    return "未命名片段"


def _snippet(text: str, matched: list[str], *, radius: int = 90) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    lower = value.lower()
    positions = [lower.find(term.lower()) for term in matched if term and lower.find(term.lower()) >= 0]
    if not positions:
        return _clip(value, radius * 2)
    start = max(min(positions) - radius, 0)
    end = min(min(positions) + radius, len(value))
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(value) else ""
    return f"{prefix}{value[start:end].strip()}{suffix}"


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


def _dedupe(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
