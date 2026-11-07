from video_knowledge_pipeline.bilinote_transcript_tools import (
    clean_subtitle_text,
    merge_transcript_segments,
    parse_subtitle_text,
    segments_to_srt,
    segments_to_txt,
)
from video_knowledge_pipeline.models import TranscriptCue
from video_knowledge_pipeline.transcript import parse_transcript


def test_parse_subtitle_text_cleans_vtt_html_and_ass_tags(tmp_path):
    subtitle = """WEBVTT

00:00.000 --> 00:02.500 align:start
<c.yellow>{\\an8}第一&nbsp;段&amp;内容</c>

00:02.600 --> 00:04.000
第二段
"""
    cues = parse_subtitle_text(subtitle)

    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        (0.0, 2.5, "第一 段&内容"),
        (2.6, 4.0, "第二段"),
    ]

    path = tmp_path / "demo.vtt"
    path.write_text(subtitle, encoding="utf-8")
    parsed = parse_transcript(path)
    assert parsed[0].text == "第一 段&内容"


def test_merge_transcript_segments_joins_short_chinese_and_latin_segments():
    merged = merge_transcript_segments(
        [
            TranscriptCue(start=0.0, end=1.0, text="第一步"),
            TranscriptCue(start=1.5, end=2.2, text="建立信任"),
            TranscriptCue(start=2.8, end=3.2, text="AI"),
            TranscriptCue(start=3.7, end=4.0, text="agent"),
        ]
    )

    assert len(merged) == 1
    assert merged[0].text == "第一步，建立信任AI agent"


def test_merge_transcript_segments_respects_strong_punctuation():
    merged = merge_transcript_segments(
        [
            TranscriptCue(start=0.0, end=1.0, text="这是结论。"),
            TranscriptCue(start=1.2, end=2.0, text="下一段"),
        ]
    )

    assert [cue.text for cue in merged] == ["这是结论。", "下一段"]


def test_segments_to_srt_and_txt():
    cues = [TranscriptCue(start=65.25, end=67.5, text="测试内容")]

    assert "00:01:05,250 --> 00:01:07,500" in segments_to_srt(cues)
    assert segments_to_txt(cues) == "[01:05] 测试内容\n"


def test_clean_subtitle_text_decodes_entities_and_removes_tags():
    assert clean_subtitle_text("<b>{\\an8}A&nbsp;&amp;&nbsp;B</b>") == "A & B"
