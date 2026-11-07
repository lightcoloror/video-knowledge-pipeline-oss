from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import ensure_project_dirs, read_json, write_json
from .transcript import format_timestamp


STUDY_INDEX_SCHEMA = "lecture_study_index.v1"


def build_lecture_study_index(root: str | Path) -> dict[str, Any]:
    """Build a searchable, no-summary study index from the full lecture timeline."""
    paths = ensure_project_dirs(root)
    package_path = paths["lecture_packages"] / "lecture-package.json"
    if not package_path.exists():
        raise FileNotFoundError(f"lecture package not found: {package_path}")
    package = read_json(package_path)
    if not isinstance(package, dict):
        raise ValueError("lecture package must be a JSON object")
    return write_lecture_study_index(paths, package)


def write_lecture_study_index(paths: dict[str, Path], package: dict[str, Any]) -> dict[str, Any]:
    """Write JSON and Markdown study-index artifacts for an existing package."""
    index = generate_lecture_study_index(package)
    cards = generate_lecture_study_cards(index)
    json_path = paths["lecture_packages"] / "lecture-study-index.json"
    markdown_path = paths["notes"] / "lecture-study-index.md"
    cards_json_path = paths["lecture_packages"] / "lecture-study-cards.json"
    cards_markdown_path = paths["notes"] / "lecture-study-cards.md"
    write_json(json_path, index)
    write_json(cards_json_path, cards)
    markdown_path.write_text(render_lecture_study_index_markdown(index), encoding="utf-8")
    cards_markdown_path.write_text(render_lecture_study_cards_markdown(cards), encoding="utf-8")
    return {
        "study_index_path": str(json_path),
        "study_index_markdown_path": str(markdown_path),
        "study_cards_path": str(cards_json_path),
        "study_cards_markdown_path": str(cards_markdown_path),
        "study_index": index,
        "study_cards": cards,
    }


def generate_lecture_study_index(package: dict[str, Any]) -> dict[str, Any]:
    """Create evidence-preserving study entry buckets from package timeline items."""
    timeline = package.get("timeline") if isinstance(package.get("timeline"), list) else []
    entries = [_entry_from_item(index, item) for index, item in enumerate(timeline, start=1) if isinstance(item, dict)]
    buckets = {
        "concepts": [entry for entry in entries if _has_any(entry, {"definition", "concept"})],
        "procedures": [entry for entry in entries if _has_any(entry, {"procedure", "step"})],
        "examples": [entry for entry in entries if _has_any(entry, {"example"})],
        "formulas": [entry for entry in entries if _has_any(entry, {"formula"})],
        "tables": [entry for entry in entries if _has_any(entry, {"table"})],
        "code": [entry for entry in entries if _has_any(entry, {"code"})],
        "visuals_to_keep": [entry for entry in entries if entry.get("has_frames") or "keep_image" in entry.get("labels", [])],
        "review_queue": [entry for entry in entries if entry.get("needs_human_review") or entry.get("review_status") == "needs_revision"],
    }
    return {
        "schema": STUDY_INDEX_SCHEMA,
        "title": str(package.get("title") or "Lecture Study Index"),
        "created_at": now_iso(),
        "source_package_schema": str(package.get("schema") or ""),
        "timeline_count": len(entries),
        "summary": {key: len(value) for key, value in buckets.items()},
        "buckets": buckets,
        "all_entries": entries,
    }


def render_lecture_study_index_markdown(index: dict[str, Any]) -> str:
    lines = [
        "---",
        "type: lecture-study-index",
        f'title: "{index.get("title", "Lecture Study Index")}"',
        "tags: [lecture-video, study-index, knowledge-package]",
        f'created: "{index.get("created_at", now_iso())}"',
        "---",
        "",
        f"# 学习索引：{index.get('title', 'Lecture Study Index')}",
        "",
        "> 这个索引只重组全量时间线，不生成替代摘要。每个条目都保留时间戳、来源片段和复核状态。",
        "",
        "## 总览",
        "",
        f"- 全量时间线片段：{index.get('timeline_count', 0)}",
    ]
    summary = index.get("summary") if isinstance(index.get("summary"), dict) else {}
    for key in ("concepts", "procedures", "examples", "formulas", "tables", "code", "visuals_to_keep", "review_queue"):
        lines.append(f"- `{key}`: {summary.get(key, 0)}")
    buckets = index.get("buckets") if isinstance(index.get("buckets"), dict) else {}
    for key, title in [
        ("concepts", "概念 / 定义候选"),
        ("procedures", "步骤 / 方法候选"),
        ("examples", "例题 / 案例候选"),
        ("formulas", "公式"),
        ("tables", "表格"),
        ("code", "代码"),
        ("visuals_to_keep", "必须保留或建议保留的视觉材料"),
        ("review_queue", "人工复核入口"),
    ]:
        lines.extend(["", f"## {title}", ""])
        entries = buckets.get(key) if isinstance(buckets.get(key), list) else []
        if not entries:
            lines.append("暂无。")
            continue
        for entry in entries:
            lines.extend(_entry_markdown(entry))
    return "\n".join(lines).rstrip() + "\n"


def generate_lecture_study_cards(index: dict[str, Any]) -> dict[str, Any]:
    """Generate evidence-preserving study cards from study-index entries."""
    entries = index.get("all_entries") if isinstance(index.get("all_entries"), list) else []
    cards = [_study_card_from_entry(entry) for entry in entries if isinstance(entry, dict)]
    return {
        "schema": "lecture_study_cards.v1",
        "title": str(index.get("title") or "Lecture Study Cards"),
        "created_at": now_iso(),
        "source_index_schema": str(index.get("schema") or ""),
        "card_count": len(cards),
        "cards": cards,
    }


def render_lecture_study_cards_markdown(cards: dict[str, Any]) -> str:
    lines = [
        "---",
        "type: lecture-study-cards",
        f'title: "{cards.get("title", "Lecture Study Cards")}"',
        "tags: [lecture-video, study-cards, knowledge-package]",
        f'created: "{cards.get("created_at", now_iso())}"',
        "---",
        "",
        f"# 学习卡片草稿：{cards.get('title', 'Lecture Study Cards')}",
        "",
        "> 这些卡片不是摘要，也不是最终知识卡。它们是从全量时间线派生的证据卡，用来保留原文、画面文字、结构化视觉和关键帧，供人工继续拆成概念卡、例题卡或步骤卡。",
        "",
        f"- 卡片数量：{cards.get('card_count', 0)}",
        "",
    ]
    for card in cards.get("cards") or []:
        if isinstance(card, dict):
            lines.extend(_study_card_markdown(card))
    return "\n".join(lines).rstrip() + "\n"


def _entry_from_item(index: int, item: dict[str, Any]) -> dict[str, Any]:
    transcript = str(item.get("transcript") or item.get("transcript_excerpt") or "").strip()
    visual_text = str(item.get("visual_text") or item.get("visual_observation") or "").strip()
    material_types = [str(value) for value in item.get("material_types") or [] if str(value)]
    signals = [str(value) for value in item.get("signals") or [] if str(value)]
    review = item.get("human_review") if isinstance(item.get("human_review"), dict) else {}
    labels = _labels_for_item(transcript, visual_text, material_types, review)
    return {
        "index": index,
        "start": _float_value(item.get("start")),
        "end": _float_value(item.get("end")),
        "time_label": f"{format_timestamp(_float_value(item.get('start')))} - {format_timestamp(_float_value(item.get('end')))}",
        "source_segment_ids": [str(value) for value in item.get("source_segment_ids") or [] if str(value)],
        "video_key": str(item.get("video_key") or item.get("video_id") or ""),
        "material_types": material_types,
        "signals": signals,
        "labels": labels,
        "headline": _headline(transcript, visual_text, labels),
        "transcript": transcript,
        "visual_text": visual_text,
        "structured_visual": item.get("structured_visual") if isinstance(item.get("structured_visual"), list) else [],
        "frame_paths": [str(path) for path in item.get("frame_paths") or [] if str(path)],
        "has_frames": bool(item.get("frame_paths")),
        "visual_retention": _visual_retention(item, review),
        "human_review": review,
        "review_status": str(item.get("review_status") or review.get("status") or "pending"),
        "needs_human_review": bool(item.get("needs_human_review", True)),
        "quality_issues": [str(value) for value in item.get("quality_issues") or [] if str(value)],
    }


def _study_card_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    review = entry.get("human_review") or {}
    card_type = str(review.get("card_kind") or "").strip() or _card_type(entry.get("labels") or [])
    return {
        "id": f"study-card-{int(entry.get('index') or 0):04d}",
        "timeline_index": int(entry.get("index") or 0),
        "card_type": card_type,
        "title": str(entry.get("headline") or f"Timeline {entry.get('index', '')}"),
        "time_label": str(entry.get("time_label") or ""),
        "start": entry.get("start", 0),
        "end": entry.get("end", 0),
        "source_segment_ids": entry.get("source_segment_ids") or [],
        "labels": entry.get("labels") or [],
        "material_types": entry.get("material_types") or [],
        "review_status": entry.get("review_status", "pending"),
        "needs_human_review": bool(entry.get("needs_human_review", True)),
        "quality_issues": entry.get("quality_issues") or [],
        "prompt": _card_prompt(card_type, entry),
        "evidence": {
            "transcript": entry.get("transcript", ""),
            "visual_text": entry.get("visual_text", ""),
            "structured_visual": entry.get("structured_visual") or [],
            "frame_paths": entry.get("frame_paths") or [],
            "visual_retention": entry.get("visual_retention") or {},
        },
        "human_fields": {
            "final_card": str(review.get("final_card") or ""),
            "links": review.get("card_links") or [],
            "questions": review.get("card_questions") or [],
            "next_review": str(review.get("next_review") or ""),
            "card_kind": str(review.get("card_kind") or ""),
            "mastery_level": str(review.get("mastery_level") or ""),
            "review_state": str(review.get("review_state") or ""),
            "confusion_points": review.get("confusion_points") or [],
        },
    }


def _study_card_markdown(card: dict[str, Any]) -> list[str]:
    evidence = card.get("evidence") if isinstance(card.get("evidence"), dict) else {}
    lines = [
        f"## {card.get('id', '')} {card.get('title', '')}",
        "",
        f"- 类型：`{card.get('card_type', 'evidence')}`",
        f"- 时间：{card.get('time_label', '')}",
        f"- 时间线序号：{card.get('timeline_index', '')}",
        f"- 来源片段：{', '.join(card.get('source_segment_ids') or [])}",
        f"- 标签：{', '.join(card.get('labels') or [])}",
        f"- 材料类型：{', '.join(card.get('material_types') or [])}",
        f"- 复核状态：`{card.get('review_status', 'pending')}`",
        "",
        "### 卡片加工位",
        "",
        f"- 提示：{card.get('prompt', '')}",
        "- [ ] 已确认事实没有来自误识别",
        "- [ ] 已决定是否需要保留图片",
        "- [ ] 已写成最终卡片或拆分成多张卡",
        "",
        "最终卡片：",
        "",
        str((card.get("human_fields") or {}).get("final_card") or "> "),
        "",
        "### 证据",
        "",
    ]
    questions = (card.get("human_fields") or {}).get("questions") or []
    links = (card.get("human_fields") or {}).get("links") or []
    next_review = str((card.get("human_fields") or {}).get("next_review") or "").strip()
    card_kind = str((card.get("human_fields") or {}).get("card_kind") or "").strip()
    mastery_level = str((card.get("human_fields") or {}).get("mastery_level") or "").strip()
    review_state = str((card.get("human_fields") or {}).get("review_state") or "").strip()
    confusion_points = (card.get("human_fields") or {}).get("confusion_points") or []
    if questions or links or next_review or card_kind or mastery_level or review_state or confusion_points:
        lines.extend(["### 人工复习字段", ""])
        if card_kind:
            lines.append(f"- 卡片类型：`{card_kind}`")
        if mastery_level:
            lines.append(f"- 掌握度：`{mastery_level}`")
        if review_state:
            lines.append(f"- 复习状态：`{review_state}`")
        if card_kind or mastery_level or review_state:
            lines.append("")
        if questions:
            lines.extend(["复习问题：", ""])
            lines.extend([f"- {question}" for question in questions])
            lines.append("")
        if confusion_points:
            lines.extend(["混淆点 / 错因：", ""])
            lines.extend([f"- {point}" for point in confusion_points])
            lines.append("")
        if links:
            lines.extend(["关联链接：", ""])
            lines.extend([f"- {link}" for link in links])
            lines.append("")
        if next_review:
            lines.extend([f"下次复习：{next_review}", ""])
    transcript = str(evidence.get("transcript") or "").strip()
    if transcript:
        lines.extend(["口语/字幕：", "", transcript, ""])
    visual_text = str(evidence.get("visual_text") or "").strip()
    if visual_text:
        lines.extend(["画面文字 / OCR / 视觉观察：", "", visual_text, ""])
    structured = evidence.get("structured_visual") if isinstance(evidence.get("structured_visual"), list) else []
    for row in structured:
        if not isinstance(row, dict) or not str(row.get("markdown") or "").strip():
            continue
        lines.extend([f"结构化视觉（{row.get('type', 'structured_visual')}）：", "", str(row.get("markdown")).strip(), ""])
    visual_retention = evidence.get("visual_retention") if isinstance(evidence.get("visual_retention"), dict) else {}
    if visual_retention:
        lines.extend(
            [
                "视觉保留判断：",
                "",
                f"- 建议：`{visual_retention.get('recommendation', 'review')}`",
                f"- 有关键帧：{bool(visual_retention.get('has_frames'))}",
                f"- 人工标记必须保留图片：{bool(visual_retention.get('keep_image_required'))}",
                f"- 结构化视觉条目：{visual_retention.get('structured_visual_count', 0)}",
                f"- 原因：{visual_retention.get('reason', '')}",
                "",
            ]
        )
    frame_paths = evidence.get("frame_paths") if isinstance(evidence.get("frame_paths"), list) else []
    if frame_paths:
        lines.extend(["关键帧：", ""])
        for path in frame_paths:
            lines.extend([_markdown_image(str(path)), "", f"- `{path}`", ""])
    issues = card.get("quality_issues") if isinstance(card.get("quality_issues"), list) else []
    if issues:
        lines.extend(["### 缺口", "", f"- {', '.join(str(issue) for issue in issues)}", ""])
    return lines


def _labels_for_item(transcript: str, visual_text: str, material_types: list[str], review: dict[str, Any]) -> list[str]:
    text = f"{transcript}\n{visual_text}".lower()
    labels = set(material_types)
    if any(keyword in text for keyword in ["定义", "所谓", "是什么", "称为", "概念", "define", "definition"]):
        labels.update({"concept", "definition"})
    if any(keyword in text for keyword in ["首先", "然后", "接着", "最后", "步骤", "step", "procedure"]):
        labels.update({"procedure", "step"})
    if any(keyword in text for keyword in ["例如", "比如", "举例", "例题", "案例", "example", "case"]):
        labels.add("example")
    if any(keyword in text for keyword in ["公式", "方程", "定理", "=", "∑", "∫", "lim", "frac"]):
        labels.add("formula")
    if any(keyword in text for keyword in ["|", "表格", "列", "行", "table"]):
        labels.add("table")
    if any(keyword in text for keyword in ["代码", "函数", "class ", "def ", "import ", "return ", "```", "code"]):
        labels.add("code")
    if review.get("keep_images"):
        labels.add("keep_image")
    return sorted(labels)


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


def _card_type(labels: list[str]) -> str:
    label_set = {str(label) for label in labels}
    for value in ("definition", "concept", "procedure", "step", "example", "formula", "table", "code", "keep_image"):
        if value in label_set:
            return value
    return "evidence"


def _card_prompt(card_type: str, entry: dict[str, Any]) -> str:
    prompts = {
        "definition": "把这一段整理成定义卡，保留原始表述中的限定条件。",
        "concept": "把这一段整理成概念卡，说明概念、用途和边界。",
        "procedure": "把这一段整理成步骤卡，按先后顺序列出动作。",
        "step": "把这一段整理成步骤卡，按先后顺序列出动作。",
        "example": "把这一段整理成例题/案例卡，保留题设、推理和结论。",
        "formula": "把这一段整理成公式卡，保留符号定义、适用条件和截图证据。",
        "table": "把这一段整理成表格卡，优先保留 Markdown 表格或截图证据。",
        "code": "把这一段整理成代码卡，保留代码块、输入输出和解释。",
        "keep_image": "判断这张图为什么不能完全降维成文字，并保留必要截图。",
    }
    return prompts.get(card_type, "把这一段整理成证据卡，先核对信息完整性，再决定是否拆分。")


def _headline(transcript: str, visual_text: str, labels: list[str]) -> str:
    source = transcript or visual_text or "无文本条目"
    cleaned = " ".join(source.split())
    if len(cleaned) > 80:
        cleaned = cleaned[:77].rstrip() + "..."
    prefix = "/".join(labels[:3])
    return f"{prefix}: {cleaned}" if prefix else cleaned


def _entry_markdown(entry: dict[str, Any]) -> list[str]:
    lines = [
        f"### {entry.get('index')}. {entry.get('time_label', '')}",
        "",
        f"- 标题：{entry.get('headline', '')}",
        f"- 标签：{', '.join(entry.get('labels') or [])}",
        f"- 材料类型：{', '.join(entry.get('material_types') or [])}",
        f"- 来源片段：{', '.join(entry.get('source_segment_ids') or [])}",
        f"- 复核状态：`{entry.get('review_status', 'pending')}`",
        "",
        "复习 / 复核：",
        "",
        "- [ ] 已核对口语/字幕没有漏听或误识别",
        "- [ ] 已核对屏幕文字、板书、图表、公式或代码",
        "- [ ] 已判断能否降维成文字，不能降维的图片已保留",
        "- [ ] 已按需要拆成概念卡、例题卡、步骤卡或结构化笔记",
        "",
    ]
    if entry.get("transcript"):
        lines.extend(["口语/字幕：", "", str(entry["transcript"]), ""])
    if entry.get("visual_text"):
        lines.extend(["画面文字 / OCR / 视觉观察：", "", str(entry["visual_text"]), ""])
    structured = entry.get("structured_visual") if isinstance(entry.get("structured_visual"), list) else []
    for row in structured:
        if not isinstance(row, dict) or not str(row.get("markdown") or "").strip():
            continue
        lines.extend([f"结构化视觉（{row.get('type', 'structured_visual')}）：", "", str(row.get("markdown")).strip(), ""])
    visual_retention = entry.get("visual_retention") if isinstance(entry.get("visual_retention"), dict) else {}
    if visual_retention:
        lines.extend(
            [
                "视觉保留判断：",
                "",
                f"- 建议：`{visual_retention.get('recommendation', 'review')}`",
                f"- 有关键帧：{bool(visual_retention.get('has_frames'))}",
                f"- 人工标记必须保留图片：{bool(visual_retention.get('keep_image_required'))}",
                f"- 结构化视觉条目：{visual_retention.get('structured_visual_count', 0)}",
                f"- 原因：{visual_retention.get('reason', '')}",
                "",
            ]
        )
    frame_paths = entry.get("frame_paths") if isinstance(entry.get("frame_paths"), list) else []
    if frame_paths:
        lines.append("关键帧：")
        lines.append("")
        for path in frame_paths:
            lines.append(_markdown_image(str(path)))
            lines.append("")
            lines.append(f"- `{path}`")
            lines.append("")
    return lines


def _has_any(entry: dict[str, Any], labels: set[str]) -> bool:
    return bool(labels & set(entry.get("labels") or []))


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _markdown_image(path: str) -> str:
    escaped = path.replace("\\", "/")
    return f"![关键帧]({escaped})"
