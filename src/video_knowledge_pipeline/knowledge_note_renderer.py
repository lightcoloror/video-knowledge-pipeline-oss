from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .models import now_iso
from .term_text import apply_high_confidence_term_replacements, is_high_confidence_term_candidate


def render_polished_knowledge_note(
    *,
    title: str,
    timeline: list[dict[str, Any]],
    coverage: dict[str, Any],
    source_summary: dict[str, Any],
    full_transcript_relpath: str,
    term_correction_status: dict[str, Any] | None = None,
) -> str:
    """Render the human reading note.

    The extraction audit keeps raw model structures. This renderer keeps the main
    note compact, evidence-linked, and readable without dumping nested JSON.
    """
    chapters = _build_chapters(timeline)
    lines: list[str] = [
        "---",
        "type: video-knowledge-note",
        f'title: "{_frontmatter_escape(title)}"',
        f'created: "{now_iso()}"',
        "status: draft",
        "tags: [video-knowledge, lecture-note]",
        "---",
        "",
        f"# {title}",
        "",
        "> 这是知识类视频的结构化整理稿。主文档只保留可读知识与证据入口；完整逐字稿和逐项审计见对应导出文件。",
        "",
        "## 视频概要",
        "",
        *_overview_lines(title, timeline, coverage, source_summary),
        "",
        "## 本视频剩余风险",
        "",
        *_remaining_risk_lines(timeline, coverage),
        "",
        "## 术语仲裁",
        "",
        *_term_correction_status_lines(term_correction_status or {}),
        "",
        "### 片段术语候选",
        "",
        *_term_resolution_lines(timeline),
        "",
        "## 核心知识结构",
        "",
        *_core_structure_lines(chapters),
        "",
        "## 分段讲解",
        "",
    ]
    for chapter in chapters:
        lines.extend(_chapter_lines(chapter))
    lines.extend(
        [
            "",
            "## 关键演示与屏幕证据",
            "",
            *_visual_evidence_lines(timeline),
            "",
            "## 表格、代码、公式、图片保留清单",
            "",
            *_retained_media_lines(timeline),
            "",
            "## 逐字稿",
            "",
            f"- 完整逐字稿：`{full_transcript_relpath}`",
            "",
            *_transcript_index_lines(timeline),
            "",
            "## 质量审计与待复核",
            "",
            *_quality_lines(timeline, coverage),
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _overview_lines(title: str, timeline: list[dict[str, Any]], coverage: dict[str, Any], source_summary: dict[str, Any]) -> list[str]:
    source = _source_label(source_summary)
    duration = _format_time(max((_float(item.get("end")) for item in timeline), default=0.0))
    topic_lines = _first_meaningful_transcripts(timeline, limit=3)
    summary = [
        f"- 视频：{title}",
        f"- 来源：{source}",
        f"- 时长覆盖：`{duration}`，时间线片段：`{len(timeline)}`",
        (
            "- 提取状态："
            f"转写 `{_channel_count(coverage, 'speech')}`，"
            f"单帧视觉 `{_channel_count(coverage, 'semantic_frame_understanding')}`，"
            f"连续片段 `{_channel_count(coverage, 'temporal_visual_understanding')}`，"
            f"图文结构 `{_channel_count(coverage, 'structured_visual')}`。"
        ),
    ]
    if topic_lines:
        summary.append("- 内容线索：" + " / ".join(topic_lines))
    weak = _weak_channel_labels(coverage)
    if weak:
        summary.append("- 已知弱项：" + "；".join(weak))
    return summary



def _term_correction_status_lines(status: dict[str, Any]) -> list[str]:
    if not status:
        return ["- 暂无术语纠错闭环状态。"]
    artifacts = status.get("artifacts") if isinstance(status.get("artifacts"), dict) else {}
    lines = [
        "| 项目 | 状态 |",
        "| --- | --- |",
        f"| 闭环状态 | `{_text(status.get('status') or 'unknown')}` |",
        f"| Codex语义预检 | `{_text(status.get('term_validation_status') or 'missing')}` |",
        f"| 预检接受/拒绝 | `{int(status.get('accepted_validation_decisions') or 0)}/{int(status.get('rejected_validation_decisions') or 0)}` |",
        f"| Codex/大模型仲裁已接受术语 | `{int(status.get('accepted_term_count') or 0)}` |",
        f"| 纠正版转写参与 | `{_yes_no(bool(status.get('source_arbitrated_transcript_exists')))}` |",
        f"| 最终导出错词残留 | `{int(status.get('final_export_alias_total') or 0)}` |",
        f"| 智能总结质量门禁 | `{_yes_no(bool(status.get('smart_summary_quality_passed')))}` |",
        f"| 影响检查 | `{_text(status.get('impact_status') or 'missing')}` |",
    ]
    next_action = _text(status.get("next_action_key"))
    if next_action:
        lines.append(f"| 下一步 | `{next_action}` |")
    evidence = []
    for key in ("term_validation_markdown", "glossary_json", "source_arbitrated_transcript_json", "impact_report_markdown", "closure_markdown"):
        raw = _text(artifacts.get(key))
        if raw:
            evidence.append(f"`{raw}`")
    accepted_terms = _accepted_term_labels(status)
    if accepted_terms:
        lines.extend(["", "- 已接受术语/工具名：" + "、".join(accepted_terms[:30])])
    if evidence:
        lines.extend(["", "- 术语闭环证据：" + "；".join(evidence)])
    return lines


def _accepted_term_labels(status: dict[str, Any]) -> list[str]:
    rows = status.get("accepted_terms") if isinstance(status.get("accepted_terms"), list) else []
    labels: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        canonical = _text(row.get("canonical_term"))
        if canonical:
            labels.append(canonical)
    return _dedupe(labels)

def _term_resolution_lines(timeline: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in timeline:
        candidates = item.get("term_candidates") if isinstance(item.get("term_candidates"), list) else []
        for row in candidates:
            if not isinstance(row, dict):
                continue
            canonical = _text(row.get("canonical_term")) or "未定"
            raw_mentions = row.get("raw_mentions") if isinstance(row.get("raw_mentions"), list) else []
            raw_text = ", ".join(_text(value) for value in raw_mentions if _text(value))
            key = (canonical.lower(), raw_text.lower())
            if key in seen:
                continue
            seen.add(key)
            status = "已自动采用" if is_high_confidence_term_candidate(row) else ("需人工复核" if row.get("needs_human_review") else "可暂用")
            rows.append(
                f"| `{_item_index(item)}` | `{_time_range(item)}` | {_table_cell(canonical)} | "
                f"{_table_cell(raw_text or canonical)} | `{row.get('confidence', '')}` | {status} |"
            )
    if not rows:
        return ["- 暂无跨来源术语仲裁结果。"]
    return ["| Index | 时间 | 建议术语 | 原始候选 | 置信度 | 状态 |", "| ---: | --- | --- | --- | ---: | --- |", *rows[:80]]


def _core_structure_lines(chapters: list[dict[str, Any]]) -> list[str]:
    if not chapters:
        return ["- 暂无可用章节。"]
    lines = ["| 章节 | 时间 | 片段 | 主题线索 | 视觉重点 | 待复核 |", "| ---: | --- | ---: | --- | --- | --- |"]
    for chapter in chapters:
        lines.append(
            "| {number} | `{time_range}` | {count} | {title} | {visual} | {review} |".format(
                number=chapter["number"],
                time_range=chapter["time_range"],
                count=len(chapter["items"]),
                title=_table_cell(chapter["title"]),
                visual=_table_cell(chapter["visual_focus"] or "无"),
                review=_table_cell(", ".join(chapter["issues"][:4]) if chapter["issues"] else "无"),
            )
        )
    return lines


def _chapter_lines(chapter: dict[str, Any]) -> list[str]:
    lines = [
        f"### {chapter['time_range']} {chapter['title']}",
        "",
        "#### 信息来源",
        "",
        *_chapter_source_channel_lines(chapter["items"]),
        "",
        "#### 讲了什么",
        "",
        *_chapter_speech_lines(chapter["items"]),
        "",
        "#### 演示了什么",
        "",
        *_chapter_demo_lines(chapter["items"]),
        "",
        "#### 人工审核",
        "",
        *_chapter_human_review_lines(chapter["items"]),
        "",
        "#### 必须保留的证据",
        "",
        *_chapter_evidence_lines(chapter["items"]),
        "",
        "#### 待复核",
        "",
        *_chapter_review_lines(chapter["items"]),
        "",
    ]
    return lines


def _chapter_speech_lines(items: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    previous = ""
    for item in items:
        text = _clip(_item_transcript(item), 180)
        if text and text != previous:
            lines.append(f"- `{_item_index(item)}` {text}")
            previous = text
    return lines or ["- 暂无转写。"]


def _chapter_source_channel_lines(items: list[dict[str, Any]]) -> list[str]:
    counts = {
        "ASR": sum(1 for item in items if _text(item.get("transcript") or item.get("original_transcript"))),
        "OCR/图文": sum(1 for item in items if _item_visual_text(item) or _structured_text(item)),
        "单帧视觉": sum(1 for item in items if _visual_understanding(item)),
        "连续片段": sum(1 for item in items if _temporal_understanding(item)),
        "人工审核": sum(1 for item in items if _human_review(item)),
    }
    lines = ["| 来源 | 参与片段 |", "| --- | ---: |"]
    lines.extend(f"| {label} | {count} |" for label, count in counts.items())
    return lines


def _chapter_demo_lines(items: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for item in items:
        demo = _item_demo_summary(item)
        if demo:
            lines.append(f"- `{_item_index(item)}` {demo}")
    return lines or ["- 这一章没有明确的视觉演示记录。"]


def _chapter_evidence_lines(items: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for item in items:
        reason = _keep_image_reason(item) or _item_visual_text(item) or _item_demo_summary(item)
        paths = _evidence_paths(item)
        if not paths and not reason:
            continue
        clean_paths = []
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            clean_paths.append(path)
        path_text = "; ".join(f"`{path}`" for path in clean_paths[:3]) if clean_paths else "证据路径见审计表"
        rows.append(f"- `{_item_index(item)}` {_clip(reason or '需要保留画面证据。', 120)}；{path_text}")
    return rows[:12] or ["- 无必须单独保留的证据。"]


def _chapter_human_review_lines(items: list[dict[str, Any]]) -> list[str]:
    rows = []
    for item in items:
        review = _human_review(item)
        status = _text(item.get("review_status") or review.get("status"))
        comment = _text(review.get("comment") or review.get("notes"))
        corrected = _item_visual_text(item) or _item_demo_summary(item)
        keep = _keep_image_reason(item)
        if not (status or comment or corrected or keep):
            continue
        parts = []
        if status:
            parts.append(f"状态 `{status}`")
        if comment:
            parts.append(comment)
        corrected_transcript = _text(item.get("human_corrected_transcript") or review.get("corrected_transcript"))
        if corrected_transcript:
            parts.append(f"修正转写：{_clip(corrected_transcript, 120)}")
        corrected_visual = _mapping(item.get("human_corrected_visual_understanding") or review.get("corrected_visual_understanding"))
        if corrected_visual:
            parts.append(f"修正视觉理解：{_clip(_value_text(corrected_visual), 140)}")
        corrected_temporal = _mapping(item.get("human_corrected_temporal_visual_understanding") or review.get("corrected_temporal_visual_understanding"))
        if corrected_temporal:
            parts.append(f"修正连续片段理解：{_clip(_value_text(corrected_temporal), 140)}")
        if keep:
            parts.append(keep)
        if corrected:
            parts.append(_clip(corrected, 120))
        rows.append(f"- `{_item_index(item)}` " + "；".join(parts))
    return rows or ["- 无人工审核记录。"]


def _chapter_review_lines(items: list[dict[str, Any]]) -> list[str]:
    issues = []
    for item in items:
        row_issues = [str(value) for value in item.get("quality_issues") or [] if str(value).strip()]
        if row_issues:
            issues.append(f"- `{_item_index(item)}` " + "、".join(row_issues[:5]))
        elif item.get("needs_human_review"):
            issues.append(f"- `{_item_index(item)}` needs_human_review")
    return issues[:12] or ["- 无。"]


def _visual_evidence_lines(timeline: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for item in timeline:
        visual = _visual_understanding(item)
        temporal = _temporal_understanding(item)
        visual_text = _item_visual_text(item)
        if not (visual or temporal or visual_text or _human_keep_image(item)):
            continue
        description = _item_demo_summary(item) or visual_text or _keep_image_reason(item) or "已保留视觉证据。"
        paths = "; ".join(_evidence_paths(item)[:3])
        rows.append(
            f"| {_item_index(item)} | `{_time_range(item)}` | `{item.get('visual_route') or 'unknown'}` | "
            f"{_table_cell(description)} | {_table_cell(paths or '见审计表')} |"
        )
    if not rows:
        return ["（暂无关键演示或屏幕证据。）"]
    return ["| Index | 时间 | 路由 | 画面信息 | 证据 |", "| ---: | --- | --- | --- | --- |", *rows[:120]]


def _retained_media_lines(timeline: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for item in timeline:
        material_types = [str(value) for value in item.get("material_types") or [] if str(value).strip()]
        structured = _structured_text(item)
        keep_reason = _keep_image_reason(item)
        if not material_types and not structured and not keep_reason:
            continue
        rows.append(
            f"| {_item_index(item)} | `{_time_range(item)}` | {_table_cell(', '.join(material_types) or '图片/界面')} | "
            f"{_table_cell(_clip(structured or keep_reason or '需要保留原图核对。', 160))} | "
            f"{_table_cell('; '.join(_evidence_paths(item)[:3]) or '见审计表')} |"
        )
    if not rows:
        return ["（暂无表格、代码、公式或必须保留图片记录。）"]
    return ["| Index | 时间 | 类型 | 内容/保留理由 | 证据 |", "| ---: | --- | --- | --- | --- |", *rows]


def _transcript_index_lines(timeline: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    previous = ""
    for item in timeline:
        text = _clip(_item_transcript(item), 140)
        if not text or text == previous:
            continue
        previous = text
        rows.append(f"| {_item_index(item)} | `{_time_range(item)}` | {_table_cell(text)} |")
    if not rows:
        return ["（无转写。）"]
    return ["| Index | 时间 | 转写摘录 |", "| ---: | --- | --- |", *rows]


def _quality_lines(timeline: list[dict[str, Any]], coverage: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    blockers = coverage.get("blockers") if isinstance(coverage.get("blockers"), list) else []
    weak = _weak_channel_labels(coverage)
    document_gaps = _coverage_gap_indexes(coverage, "structured_visual")
    visual_gaps = _coverage_gap_indexes(coverage, "semantic_frame_understanding") + _coverage_gap_indexes(coverage, "temporal_visual_understanding")
    lines.append(f"- 覆盖状态：`{coverage.get('status') or 'unknown'}`")
    lines.append(f"- 阻塞项：{'; '.join(_blocker_label(item) for item in blockers) if blockers else '无'}")
    lines.append(f"- 弱项：{'; '.join(weak) if weak else '无'}")
    lines.extend(["", "### 图文截图/表格/代码/公式待解析", "", f"- {', '.join(document_gaps) if document_gaps else '无'}"])
    lines.extend(["", "### 视频视觉理解待补齐", "", f"- {', '.join(visual_gaps) if visual_gaps else '无'}"])
    review_rows = []
    for item in timeline:
        issues = [str(value) for value in item.get("quality_issues") or [] if str(value).strip()]
        if issues or item.get("needs_human_review"):
            review_rows.append(f"| {_item_index(item)} | `{_time_range(item)}` | {_table_cell(', '.join(issues) or 'needs_human_review')} |")
    if review_rows:
        lines.extend(["", "| Index | 时间 | 待复核原因 |", "| ---: | --- | --- |", *review_rows[:80]])
    return lines


def _remaining_risk_lines(timeline: list[dict[str, Any]], coverage: dict[str, Any]) -> list[str]:
    risks: list[str] = []
    status = _text(coverage.get("status") or "unknown")
    if status not in {"accepted", "ready"}:
        risks.append(f"- 覆盖状态仍为 `{status}`，需要结合审计文件判断是否可接受。")
    blockers = coverage.get("blockers") if isinstance(coverage.get("blockers"), list) else []
    if blockers:
        risks.append("- 阻塞项：" + "；".join(_blocker_label(item) for item in blockers[:6]))
    weak = _weak_channel_labels(coverage)
    if weak:
        risks.append("- 弱项：" + "；".join(weak[:6]))
    issue_counts: dict[str, int] = {}
    for item in timeline:
        for issue in item.get("quality_issues") or []:
            key = _text(issue)
            if key:
                issue_counts[key] = issue_counts.get(key, 0) + 1
    for issue, count in sorted(issue_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:8]:
        risks.append(f"- `{issue}`：{count} 个片段。")
    if not risks:
        return ["- 当前没有机器可识别的剩余风险；仍建议抽查证据帧。"]
    return risks


def _build_chapters(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_route = ""
    previous_end = 0.0
    for item in timeline:
        route = str(item.get("visual_route") or "unknown")
        start = _float(item.get("start"))
        gap = start - previous_end if current else 0.0
        should_split = bool(current) and (route != previous_route or gap > 45 or len(current) >= 8)
        if should_split:
            groups.append(current)
            current = []
        current.append(item)
        previous_route = route
        previous_end = _float(item.get("end"))
    if current:
        groups.append(current)
    chapters = []
    for number, items in enumerate(groups, start=1):
        start = min((_float(item.get("start")) for item in items), default=0.0)
        end = max((_float(item.get("end")) for item in items), default=start)
        chapters.append(
            {
                "number": number,
                "items": items,
                "time_range": f"{_format_time(start)} - {_format_time(end)}",
                "title": _chapter_title(number, items),
                "visual_focus": _chapter_visual_focus(items),
                "issues": _dedupe(str(issue) for item in items for issue in item.get("quality_issues") or [] if str(issue).strip()),
            }
        )
    return chapters


def _chapter_title(number: int, items: list[dict[str, Any]]) -> str:
    for item in items:
        text = _display_text(item, item.get("visual_text")) or _item_transcript(item)
        if text:
            return _clip(text.splitlines()[0], 42)
    return f"第 {number} 段"


def _chapter_visual_focus(items: list[dict[str, Any]]) -> str:
    for item in items:
        value = _item_demo_summary(item) or _item_visual_text(item) or _keep_image_reason(item)
        if value:
            return _clip(value, 80)
    return ""


def _item_demo_summary(item: dict[str, Any]) -> str:
    visual = _visual_understanding(item)
    temporal = _temporal_understanding(item)
    parts = []
    parts.extend(_value_snippets(visual.get("actions"), label="动作"))
    interface_state = _value_text(visual.get("interface_state"))
    if interface_state:
        parts.append(f"界面：{interface_state}")
    instructor_focus = _value_text(visual.get("instructor_focus"))
    if instructor_focus:
        parts.append(f"讲师强调：{instructor_focus}")
    parts.extend(_value_snippets(temporal.get("event_sequence"), label="事件"))
    parts.extend(_value_snippets(temporal.get("operation_steps"), label="步骤"))
    parts.extend(_value_snippets(temporal.get("state_changes"), label="变化"))
    return _clip("；".join(part for part in parts if part), 220)


def _value_snippets(value: Any, *, label: str) -> list[str]:
    text = _value_text(value)
    return [f"{label}：{text}"] if text else []


def _value_text(value: Any) -> str:
    if isinstance(value, str):
        parsed = _parse_literal_string(value)
        if parsed is not None:
            return _value_text(parsed)
        return _text(value)
    if isinstance(value, list):
        parts = [_value_text(item) for item in value]
        return "；".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("description", "action", "type", "text", "label", "name"):
            if _text(value.get(key)):
                return _text(value.get(key))
        parts = []
        for key, val in value.items():
            text = _value_text(val)
            if text:
                parts.append(f"{key}: {text}")
        return "；".join(parts[:4])
    return _text(value)


def _structured_text(item: dict[str, Any]) -> str:
    rows = []
    structured = item.get("structured_visual") if isinstance(item.get("structured_visual"), list) else []
    for entry in structured:
        if not isinstance(entry, dict):
            continue
        value = _meaningful_visual_text(entry.get("markdown") or entry.get("text"), item)
        if value:
            rows.append(_clip(value, 120))
    return "；".join(rows[:3])


def _visual_understanding(item: dict[str, Any]) -> dict[str, Any]:
    return _mapping(item.get("human_corrected_visual_understanding")) or _mapping(_human_review(item).get("corrected_visual_understanding")) or _mapping(item.get("visual_understanding"))


def _temporal_understanding(item: dict[str, Any]) -> dict[str, Any]:
    return _mapping(item.get("human_corrected_temporal_visual_understanding")) or _mapping(_human_review(item).get("corrected_temporal_visual_understanding")) or _mapping(item.get("temporal_visual_understanding"))


def _item_transcript(item: dict[str, Any]) -> str:
    corrected = _text(item.get("corrected_transcript"))
    if corrected:
        return corrected
    return _display_text(item, item.get("transcript") or item.get("original_transcript"))


def _item_visual_text(item: dict[str, Any]) -> str:
    value = (
        _meaningful_visual_text(item.get("human_corrected_visual_text"), item)
        or _meaningful_visual_text(_human_review(item).get("corrected_visual_text"), item)
        or _meaningful_visual_text(item.get("visual_text") or item.get("original_visual_text"), item)
    )
    return apply_high_confidence_term_replacements(value, item)


def _keep_image_reason(item: dict[str, Any]) -> str:
    visual = _visual_understanding(item)
    reason = _value_text(visual.get("keep_image_reason"))
    if reason:
        return reason
    if _human_keep_image(item):
        return "人工保留图片：人工确认保留图片证据，未强行降维为文字。"
    return ""


def _evidence_paths(item: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for source in (_visual_understanding(item).get("evidence_frame_paths") or []):
        if _text(source):
            paths.append(_text(source))
    for source in (_temporal_understanding(item).get("evidence_frame_paths") or []):
        if _text(source):
            paths.append(_text(source))
    for asset in item.get("assets") or []:
        if isinstance(asset, dict):
            path = _text(asset.get("path") or asset.get("source"))
            if path:
                paths.append(path)
    for source in item.get("frame_paths") or []:
        if _text(source):
            paths.append(_text(source))
    recovery = item.get("screen_text_recovery") if isinstance(item.get("screen_text_recovery"), dict) else {}
    for source in recovery.get("crop_paths") or []:
        if _text(source):
            paths.append(_text(source))
    return _dedupe(paths)


def _channel_count(coverage: dict[str, Any], key: str) -> str:
    for channel in coverage.get("channels") or []:
        if isinstance(channel, dict) and channel.get("key") == key:
            return f"{channel.get('covered_count', 0)} / {channel.get('expected_count', 0)}"
    return "unknown"


def _weak_channel_labels(coverage: dict[str, Any]) -> list[str]:
    labels = []
    for channel in coverage.get("weak_channels") or []:
        if isinstance(channel, dict):
            labels.append(
                f"{channel.get('label') or channel.get('key')} `{channel.get('covered_count', 0)} / {channel.get('expected_count', 0)}`"
            )
    return labels


def _coverage_gap_indexes(coverage: dict[str, Any], key: str) -> list[str]:
    samples = coverage.get("samples") if isinstance(coverage.get("samples"), dict) else {}
    mapping = {
        "structured_visual": "structured_gap",
        "semantic_frame_understanding": "missing_visual_understanding",
        "temporal_visual_understanding": "temporal_sequence_without_analysis",
    }
    values = samples.get(mapping.get(key, key), [])
    return [str(value) for value in values] if isinstance(values, list) else []


def _blocker_label(value: Any) -> str:
    if isinstance(value, dict):
        return _text(value.get("label") or value.get("key") or value.get("reason") or value.get("message") or "unknown")
    return _text(value)


def _source_label(source_summary: dict[str, Any]) -> str:
    sources = source_summary.get("sources") if isinstance(source_summary.get("sources"), list) else []
    labels = []
    for source in sources:
        if isinstance(source, dict):
            label = _text(source.get("title") or source.get("id") or source.get("source_id") or source.get("path") or source.get("url"))
            if label:
                labels.append(label)
    return "；".join(labels[:3]) if labels else "本地视频"


def _first_meaningful_transcripts(timeline: list[dict[str, Any]], *, limit: int) -> list[str]:
    values = []
    previous = ""
    for item in timeline:
        text = _clip(_item_transcript(item), 80)
        if text and text != previous:
            values.append(text)
            previous = text
        if len(values) >= limit:
            break
    return values


def _human_review(item: dict[str, Any]) -> dict[str, Any]:
    return item.get("human_review") if isinstance(item.get("human_review"), dict) else {}


def _human_keep_image(item: dict[str, Any]) -> bool:
    return bool(item.get("human_keep_image") or _human_review(item).get("keep_image"))


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) and value else {}


def _meaningful_visual_text(value: Any, item: dict[str, Any]) -> str:
    text = _text(value)
    if not text:
        return ""
    stems = _frame_stems(item)
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("<!--") and stripped.endswith("-->") and "source:" in stripped.lower():
            continue
        if stripped.startswith("# ") and stripped[2:].strip() in stems:
            continue
        kept.append(line.rstrip())
    return "\n".join(kept).strip()


def _frame_stems(item: dict[str, Any]) -> set[str]:
    stems: set[str] = set()
    for key in ("frame_paths", "evidence_paths"):
        values = item.get(key)
        if isinstance(values, list):
            for value in values:
                text = _text(value)
                if text:
                    stems.add(Path(text).stem)
    for asset in item.get("assets") or []:
        if isinstance(asset, dict):
            text = _text(asset.get("path") or asset.get("source"))
            if text:
                stems.add(Path(text).stem)
    return {stem for stem in stems if stem}


def _time_range(item: dict[str, Any]) -> str:
    return f"{_format_time(item.get('start'))} - {_format_time(item.get('end'))}"


def _format_time(value: Any) -> str:
    seconds = _float(value)
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    sec = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{minutes:02d}:{sec:02d}.{ms:03d}"


def _float(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _item_index(item: dict[str, Any]) -> int:
    try:
        return int(item.get("index") or 0)
    except (TypeError, ValueError):
        return 0


def _clip(value: str, limit: int) -> str:
    value = _text(value).replace("\n", " ")
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."



def _yes_no(value: bool) -> str:
    return "是" if value else "否"
def _table_cell(value: str) -> str:
    return _text(value).replace("|", "\\|").replace("\n", "<br>")


def _display_text(item: dict[str, Any], value: Any) -> str:
    return apply_high_confidence_term_replacements(_text(value), item)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _parse_literal_string(value: str) -> Any | None:
    stripped = value.strip()
    if not ((stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]"))):
        return None
    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, (dict, list, tuple)) else None


def _dedupe(values: Any) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _frontmatter_escape(value: str) -> str:
    return value.replace('"', '\\"')
