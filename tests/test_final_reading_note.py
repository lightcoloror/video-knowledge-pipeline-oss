from video_knowledge_pipeline.final_reading_note import render_final_reading_note


def test_final_reading_note_combines_summary_then_transcript_without_internal_metadata() -> None:
    note = render_final_reading_note(
        "示例课程",
        smart_summary_markdown="# 示例课程 - 智能总结\n\n课程讲解了项目沟通的三个原则。\n",
        transcript_markdown=(
            "# 示例课程 - 原始转录\n\n"
            "### 00:00:00.000 - 00:00:05.000\n\n"
            "项目沟通先明确目标。\n"
        ),
        timeline=[{"start": 0, "end": 5}],
    )

    assert note.index("  - 📑 智能总结") < note.index("- 逐字稿")
    assert "    - 录音信息" in note
    assert "约 00:00:05.000" in note
    assert "课程讲解了项目沟通的三个原则。" in note
    assert "项目沟通先明确目标。" in note
    assert "source_arbitrated_transcript" not in note
    assert "arbitrated_or_reviewed" not in note
    assert "处理时间" not in note
    assert "来源路径" not in note
    assert "章节修订来源" not in note
    assert "视觉证据状态" not in note
    assert "Transcript quality gate:" not in note
    assert note.startswith("- 摘要\n  - 📑 智能总结")
    assert "  - [00:00:00.000]" in note
    assert "collapsed::" not in note
    assert not any(line.lstrip().startswith("#") for line in note.splitlines())


def test_final_reading_note_replaces_operational_summary_placeholder() -> None:
    note = render_final_reading_note(
        "示例课程",
        smart_summary_markdown="# 示例课程 - 智能总结\n\n生成状态：needs_llm_summary。\n\n.\\scripts\\video-knowledge.ps1 ...\n",
        transcript_markdown="# 示例课程 - 原始转录\n\n逐字稿正文。\n",
        timeline=[],
    )

    assert "智能总结尚未生成" in note
    assert "needs_llm_summary" not in note
    assert ".\\scripts\\video-knowledge.ps1" not in note


def test_final_reading_note_projects_canonical_summary_without_regenerating_content() -> None:
    note = render_final_reading_note(
        "双人项目沟通记录",
        smart_summary_markdown="""# 双人项目沟通记录 - 智能总结

生成方式：`codex_llm_rewrite_final`。
> 证据边界：本总结仅依据已入库证据。
schema: internal.summary.v1
provider: test-provider
route_revision: abc123
来源状态：source_arbitrated_transcript
仲裁状态：arbitrated_or_reviewed

## 基本信息

- 内容类型：项目访谈
- 处理时间：2026-07-29T17:00:00
- 来源路径：D:/internal/bundle
- 章节修订来源：D:/internal/revisions.json
- 视觉证据状态：内部质量门状态

## 一句话概览

双方确认了项目交付方向。

## 核心主题

- 不同阶段采用不同的交付检查策略。
- model: 讲者在正文中讨论的模型名称必须保留。

## 分段总结

### 00:00:00 项目需求

参与者说明了排期与资源。

## 关键观点

- 先明确交付目标，再比较方案。

## 可执行动作清单

- 项目负责人次日提供对比方案。

## 高频话术

- “先把需求梳理清楚。”

## 待复核点

- 一处功能名称需要听音确认。
""",
        transcript_markdown="# 原始转录\n\n**说话人1**：根据排期来的嘛。\n",
        timeline=[{"start": 0, "end": 5}],
        content_type="项目访谈",
        participant_count=2,
    )

    assert "双方确认了项目交付方向。" in note
    assert "讲者在正文中讨论的模型名称必须保留。" in note
    assert "    - 📅 章节概要" in note
    assert "    - ✨ 金句精选" in note
    assert "    - 📋 待办事项" in note
    assert "    - 待复核点" in note
    assert note.index("    - 📅 章节概要") < note.index("    - ✨ 金句精选")
    assert note.index("    - ✨ 金句精选") < note.index("    - 📋 待办事项")
    assert "- 逐字稿" in note
    assert "  - 🟢 说话人1" in note
    assert "    - 根据排期来的嘛。" in note
    assert "collapsed::" not in note
    assert "生成方式" not in note
    assert "证据边界" not in note
    assert "schema:" not in note
    assert "provider:" not in note
    assert "route_revision" not in note
    assert "source_arbitrated_transcript" not in note
    assert "arbitrated_or_reviewed" not in note
    assert "处理时间" not in note
    assert "来源路径" not in note
    assert "章节修订来源" not in note
    assert "视觉证据状态" not in note