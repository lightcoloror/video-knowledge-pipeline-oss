from __future__ import annotations

import re
from typing import Any

from .transcript import format_timestamp


_CANONICAL_READER_ORDER = (
    ("一句话概览", "录音总结"),
    ("基本信息", "基本信息"),
    ("核心主题", "核心主题"),
    ("关键观点", "关键观点"),
    ("分段总结", "📅 章节概要"),
    ("高频话术", "✨ 金句精选"),
    ("可执行动作清单", "📋 待办事项"),
    ("待复核点", "待复核点"),
)
_MARKDOWN_HEADING = re.compile(r"^(?P<rank>#{1,6})\s+(?P<content>.+?)\s*$")
_MARKDOWN_LIST_ITEM = re.compile(
    r"^(?P<indent>[ \t]*)(?P<marker>[-*+]|\d+\.)\s+(?P<content>.+?)\s*$"
)
_SPEAKER_LINE = re.compile(
    r"^\*\*(?P<speaker>[^*]+)\*\*\s*[:：]?\s*(?P<text>.*)$"
)
_TIMESTAMP_START = re.compile(r"(?P<timestamp>\d{1,2}:\d{2}:\d{2}(?:\.\d{1,3})?)")
_SPEAKER_COLOURS = ("🟢", "🟣", "🔵", "🟠", "🟡", "⚪")
_READER_BASIC_INFO_OPERATIONAL_LINE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?"
    r"(?:处理时间|来源路径|章节修订来源|视觉证据状态|生成时间|生成方式|"
    r"来源状态|仲裁状态|canonical|schema|provider|model|route|consent)"
    r"(?:\*\*)?\s*[:：]",
    flags=re.IGNORECASE,
)
_OPERATIONAL_LINE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?"
    r"(?:生成方式|来源状态|仲裁状态|schema|provider|model|"
    r"route(?:_id|_revision)?|consent(?:_id)?|source_status|"
    r"arbitration_status)"
    r"(?:\*\*)?\s*[:：]",
    flags=re.IGNORECASE,
)


def render_final_reading_note(
    title: str,
    *,
    smart_summary_markdown: str,
    transcript_markdown: str,
    full_body_markdown: str = "",
    timeline: list[dict[str, Any]],
    content_type: str = "视频整理",
    participant_count: int = 0,
) -> str:
    """Render the single reader-facing document as a Logseq block tree.

    Intent: make the combined Smart Summary + continuous body + transcript
    artifact directly consumable by Logseq while retaining the GetBrain reader
    hierarchy.
    Decision: reuse the local getnote-logseq-sync convention: every semantic
    node is a ``- `` block, each child adds two spaces, transcript entries use
    a speaker/time parent and a text child, no default ``collapsed::`` property,
    and no raw Markdown heading remains.
    Reason: ordinary Markdown headings flatten or split when imported through
    Logseq block APIs; omitting the property preserves Logseq's default expanded
    state and leaves collapse/expand state under the reader's control.
    Evidence: local getnote-logseq-sync good fixtures
    ``03-section-hierarchy.md`` and ``08-transcript-timestamps.md`` plus its
    ``check_logseq_file_format.py`` preflight.
    Effective scope: final reader Markdown only; canonical summary/evidence,
    transcript JSON, timestamps, speaker attribution and quality gates are not
    changed. The body is a presentation-only projection of the same transcript.
    """

    duration = max((float(item.get("end") or 0.0) for item in timeline), default=0.0)
    safe_content_type = str(content_type or "音视频整理").replace("\r", " ").replace("\n", " ").strip()
    safe_participant_count = max(0, int(participant_count or 0))
    recording_info = [
        f"**时长**：约 {format_timestamp(duration)}",
        f"**内容类型**：{safe_content_type}",
    ]
    if safe_participant_count:
        recording_info.append(f"**参与人数**：约 {safe_participant_count} 人")
    raw_summary = _reading_body(smart_summary_markdown)
    if "生成状态：needs_llm_summary" in raw_summary:
        summary_overview = "（智能总结尚未生成；完成总结后重新运行导出即可刷新本页。）"
        summary_sections: list[tuple[str, str]] = []
    else:
        summary_overview, summary_sections = _reader_summary_components(raw_summary)
    full_body = _reading_body(full_body_markdown)
    transcript = _reading_body(transcript_markdown)
    lines = [
        "- 摘要",
        "  - 📑 智能总结",
        "    - 录音信息",
        *(f"      - {item}" for item in recording_info),
        "    - 录音总结",
        *_markdown_fragment_to_logseq_blocks(
            summary_overview or "（暂无智能总结。）",
            base_level=3,
        ),
    ]
    for section_title, section_content in summary_sections:
        lines.append(f"    - {section_title}")
        lines.extend(_markdown_fragment_to_logseq_blocks(section_content, base_level=3))
    lines.append("- 正文")
    lines.extend(
        _markdown_fragment_to_logseq_blocks(
            full_body or "（暂无正文。）",
            base_level=1,
        )
    )
    lines.append("- 逐字稿")
    lines.extend(_transcript_to_logseq_blocks(transcript, base_level=1))
    return "\n".join(lines).rstrip() + "\n"


def _reader_summary_body(markdown: str) -> str:
    """Project the canonical summary into a GetBrain-style reading layout.

    Intent: keep the final document concise while retaining every substantive
    summary section.
    Decision: reuse the canonical Smart Summary as the only content source and
    apply a deterministic presentation projection inspired by BiliNote's
    selectable note formats.
    Reason: regenerating a second summary would split the evidence truth and
    quality gate; a heading projection is reversible and model-free.
    Evidence: VKP canonical headings and pinned BiliNote prompt-builder source
    at commit 095d772c7d0f2f4ba1e65c36b7ceb1e2db34723d.
    Effective scope: reader-facing Markdown only; canonical summary, evidence,
    transcript, timestamps, and review decisions remain unchanged.
    """

    overview, sections = _reader_summary_components(markdown)
    blocks = [overview] if overview else []
    blocks.extend(
        f"## {section_title}\n\n{content}".strip()
        for section_title, content in sections
        if content
    )
    return "\n\n".join(blocks).strip()


def _reader_summary_components(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    """Return reader overview and ordered sections without creating new facts."""

    cleaned = _strip_reader_operational_lines(_reading_body(markdown))
    preamble, sections = _split_level_two_sections(cleaned)
    if not any(name in sections for name, _label in _CANONICAL_READER_ORDER):
        return cleaned, []

    overview = sections.get("一句话概览", "").strip() or preamble
    projected: list[tuple[str, str]] = []
    consumed = {"一句话概览"}
    for name, label in _CANONICAL_READER_ORDER:
        if name == "一句话概览":
            continue
        content = sections.get(name, "").strip()
        if name == "基本信息":
            content = _strip_reader_basic_info_lines(content)
        if not content:
            continue
        consumed.add(name)
        projected.append((label, content))
    for name, content in sections.items():
        if name in consumed or not content.strip():
            continue
        projected.append((name, content.strip()))
    return overview, projected


def _strip_reader_basic_info_lines(markdown: str) -> str:
    """Hide export mechanics from the reader projection, not from evidence."""

    return "\n".join(
        line
        for line in str(markdown or "").splitlines()
        if not _READER_BASIC_INFO_OPERATIONAL_LINE.match(line)
    ).strip()


def _markdown_fragment_to_logseq_blocks(
    markdown: str,
    *,
    base_level: int,
) -> list[str]:
    """Adapt an ordinary Markdown fragment to reviewed Logseq blocks.

    Intent: reuse the local getnote-logseq-sync block convention.
    Decision: implement only the heading/list/paragraph projection needed by
    final reader artifacts; canonical Markdown remains unchanged.
    Reason: importing raw headings inside a Logseq block flattens hierarchy.
    Evidence: getnote-logseq-sync format fixtures and file-format checker.
    Effective scope: presentation-only conversion with no content generation.
    """

    rendered: list[str] = []
    heading_stack: list[tuple[int, int]] = []
    normalized = str(markdown or "").replace("\r\n", "\n").replace("\r", "\n")
    for raw_line in normalized.splitlines():
        if not raw_line.strip():
            continue
        stripped = raw_line.strip()
        heading = _MARKDOWN_HEADING.match(stripped)
        if heading:
            rank = len(heading.group("rank"))
            while heading_stack and rank <= heading_stack[-1][0]:
                heading_stack.pop()
            level = base_level + len(heading_stack)
            rendered.append(_logseq_block(level, heading.group("content")))
            heading_stack.append((rank, level))
            continue
        list_item = _MARKDOWN_LIST_ITEM.match(raw_line)
        content_level = heading_stack[-1][1] + 1 if heading_stack else base_level
        if list_item:
            indent = list_item.group("indent").replace("\t", "  ")
            level = content_level + (len(indent) // 2)
            marker = list_item.group("marker")
            content = list_item.group("content").strip()
            if marker[0].isdigit():
                content = f"{marker} {content}"
            rendered.append(_logseq_block(level, content))
            continue
        rendered.append(_logseq_block(content_level, stripped))
    return rendered


def _transcript_to_logseq_blocks(
    markdown: str,
    *,
    base_level: int,
) -> list[str]:
    """Render timestamp/speaker transcript sections as collapsible blocks."""

    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_lines: list[str] = []
    normalized = str(markdown or "").replace("\r\n", "\n").replace("\r", "\n")
    for raw_line in normalized.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        heading = _MARKDOWN_HEADING.match(stripped)
        if heading:
            if current_heading or current_lines:
                sections.append((current_heading, current_lines))
            current_heading = heading.group("content").strip()
            current_lines = []
        else:
            current_lines.append(stripped)
    if current_heading or current_lines:
        sections.append((current_heading, current_lines))
    if not sections:
        return [_logseq_block(base_level, "（暂无逐字稿。）")]

    rendered: list[str] = []
    for heading, content_lines in sections:
        speaker = ""
        speaker_text = ""
        if content_lines:
            speaker_match = _SPEAKER_LINE.match(content_lines[0])
            if speaker_match:
                speaker = speaker_match.group("speaker").strip()
                speaker_text = speaker_match.group("text").strip()
                content_lines = content_lines[1:]
        timestamp_match = _TIMESTAMP_START.search(heading)
        timestamp = timestamp_match.group("timestamp") if timestamp_match else ""
        if speaker:
            label = f"{_speaker_colour(speaker)} {speaker}"
            if timestamp:
                label += f" [{timestamp}]"
        elif timestamp:
            label = f"[{timestamp}]"
        else:
            label = "逐字稿片段"
        rendered.append(_logseq_block(base_level, label))
        body = [speaker_text, *content_lines] if speaker_text else content_lines
        if not body:
            body = ["（无转写文本。）"]
        for line in body:
            rendered.extend(
                _markdown_fragment_to_logseq_blocks(
                    line,
                    base_level=base_level + 1,
                )
            )
    return rendered


def _speaker_colour(speaker: str) -> str:
    match = re.search(r"(\d+)\s*$", str(speaker or ""))
    if not match:
        return _SPEAKER_COLOURS[0]
    index = max(1, int(match.group(1))) - 1
    return _SPEAKER_COLOURS[index % len(_SPEAKER_COLOURS)]


def _logseq_block(level: int, content: str) -> str:
    safe_content = str(content or "").strip() or "（空）"
    return f"{'  ' * max(0, int(level))}- {safe_content}"


def _strip_reader_operational_lines(markdown: str) -> str:
    kept: list[str] = []
    in_summary_section = False
    for line in str(markdown or "").splitlines():
        stripped = line.strip()
        if re.match(r"^##\s+", line):
            in_summary_section = True
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue
        if stripped.startswith("> 证据边界：") or stripped.startswith("> 证据边界:"):
            continue
        if not in_summary_section and _OPERATIONAL_LINE.match(line):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _split_level_two_sections(markdown: str) -> tuple[str, dict[str, str]]:
    preamble: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in str(markdown or "").splitlines():
        match = re.match(r"^##\s+(.+?)\s*$", line)
        if match:
            current = match.group(1).strip()
            sections.setdefault(current, [])
            continue
        if current is None:
            preamble.append(line)
        else:
            sections[current].append(line)
    return (
        "\n".join(preamble).strip(),
        {name: "\n".join(lines).strip() for name, lines in sections.items()},
    )


def _reading_body(markdown: str) -> str:
    lines = str(markdown or "").replace("\r\n", "\n").split("\n")
    if lines[:1] == ["---"]:
        try:
            closing = lines.index("---", 1)
        except ValueError:
            closing = -1
        if closing >= 0:
            lines = lines[closing + 1 :]
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and re.match(r"^#\s+", lines[0].lstrip()):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).strip()
