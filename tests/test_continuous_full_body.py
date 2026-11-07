from pathlib import Path

from video_knowledge_pipeline.final_reading_note import render_final_reading_note
from video_knowledge_pipeline.knowledge_note_export import _continuous_body_lines_from_cues
from video_knowledge_pipeline.models import TranscriptCue
from video_knowledge_pipeline.reader_export_receipt import (
    build_reader_export_receipt,
    receipt_matches_reader_files,
)


def test_continuous_body_preserves_cue_text_order_without_timestamps_or_speakers() -> None:
    cues = [
        TranscriptCue(start=0.0, end=1.0, text="第一句。", speaker="说话人1"),
        TranscriptCue(start=1.0, end=2.0, text="第二句保持原文", speaker="说话人2"),
        TranscriptCue(start=2.0, end=3.0, text="第三句。", speaker="说话人1"),
    ]

    paragraphs = _continuous_body_lines_from_cues(cues, target_chars=10)
    body = "\n\n".join(paragraphs)

    assert body == "第一句。 第二句保持原文\n\n第三句。"
    assert body.index("第一句。") < body.index("第二句保持原文") < body.index("第三句。")
    assert "00:00:" not in body
    assert "说话人1" not in body
    assert "说话人2" not in body


def test_final_reading_note_places_body_between_summary_and_transcript() -> None:
    note = render_final_reading_note(
        "示例课程",
        smart_summary_markdown="# 示例课程 - 智能总结\n\n一句话总结。\n",
        full_body_markdown="# 示例课程 - 正文\n\n第一句。 第二句。\n\n第三句。\n",
        transcript_markdown=(
            "# 示例课程 - 原始转录\n\n"
            "### 00:00:00.000 - 00:00:05.000\n\n"
            "**说话人1**\n\n第一句。\n"
        ),
        timeline=[{"start": 0, "end": 5}],
    )

    assert note.index("  - 📑 智能总结") < note.index("- 正文") < note.index("- 逐字稿")
    body_section = note[note.index("- 正文") : note.index("- 逐字稿")]
    assert "第一句。 第二句。" in body_section
    assert "第三句。" in body_section
    assert "00:00:" not in body_section
    assert "说话人1" not in body_section


def test_reader_export_receipt_binds_full_body(tmp_path: Path) -> None:
    canonical = tmp_path / "source-arbitrated-transcript.json"
    full_transcript = tmp_path / "full-transcript.md"
    full_body = tmp_path / "full-body.md"
    reading_note = tmp_path / "knowledge-note.md"
    canonical.write_text('{"segments":[{"text":"正文证据"}]}', encoding="utf-8")
    full_transcript.write_text("# 逐字稿\n\n正文证据\n", encoding="utf-8")
    full_body.write_text("# 正文\n\n正文证据\n", encoding="utf-8")
    reading_note.write_text("- 正文\n  - 正文证据\n", encoding="utf-8")

    receipt = build_reader_export_receipt(
        canonical_transcript=canonical,
        full_transcript=full_transcript,
        full_body=full_body,
        reading_note=reading_note,
    )

    assert receipt_matches_reader_files(
        receipt,
        canonical_transcript=canonical,
        full_transcript=full_transcript,
        full_body=full_body,
        reading_note=reading_note,
    )
    full_body.write_text("# 正文\n\n已被修改\n", encoding="utf-8")
    assert not receipt_matches_reader_files(
        receipt,
        canonical_transcript=canonical,
        full_transcript=full_transcript,
        full_body=full_body,
        reading_note=reading_note,
    )
