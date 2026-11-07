from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline.transcript import parse_transcript


def test_getbrain_speaker_timeline_preserves_speaker_time_and_raw_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "consultation.txt"
    source.write_text(
        "\n".join(
            [
                "双人项目沟通记录",
                "",
                "说话人1 00:00:00",
                "根排期来的嘛你们现在有哪些准备？",
                "",
                "说话人2 00:00:05",
                "目前还没有准备。",
                "",
                "说话人1 00:00:07",
                "就是那天下午开了个会，",
                "然后给你发了一分材料。",
            ]
        ),
        encoding="utf-8",
    )

    cues = parse_transcript(source)

    assert len(cues) == 3
    assert [cue.speaker for cue in cues] == ["说话人1", "说话人2", "说话人1"]
    assert [(cue.start, cue.end) for cue in cues] == [
        (0.0, 5.0),
        (5.0, 7.0),
        (7.0, 7.0),
    ]
    assert cues[0].text == "根排期来的嘛你们现在有哪些准备？"
    assert cues[2].text == "就是那天下午开了个会， 然后给你发了一分材料。"
    assert cues[0].segment_id == "segment-000001"
    assert cues[0].source_segment_ids == ["segment-000001"]
    assert cues[0].metadata["end_inferred_from_next_start"] is True
    assert cues[-1].metadata["end_unknown"] is True


def test_getbrain_parser_does_not_apply_unconfirmed_term_corrections(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw-asr.md"
    source.write_text(
        "\n".join(
            [
                "# 摘要标题不会进入逐字稿",
                "",
                "说话人2 00:03:07",
                "记录我就会义纪要啊。",
                "",
                "说话人1 00:03:09",
                "记录是会义纪要。",
                "",
                "说话人1 00:23:40",
                "星合系统的操作手册。",
            ]
        ),
        encoding="utf-8",
    )

    cues = parse_transcript(source)

    assert [cue.text for cue in cues] == [
        "记录我就会义纪要啊。",
        "记录是会义纪要。",
        "星合系统的操作手册。",
    ]
    assert all("会议纪要" not in cue.text for cue in cues)
    assert all("星河系统" not in cue.text for cue in cues)
    assert [cue.speaker for cue in cues] == ["说话人2", "说话人1", "说话人1"]


def test_plain_text_without_speaker_headers_keeps_legacy_fallback(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plain.txt"
    source.write_text("第一行\n第二行\n", encoding="utf-8")

    cues = parse_transcript(source)

    assert [cue.text for cue in cues] == ["第一行", "第二行"]
    assert [cue.speaker for cue in cues] == ["", ""]
    assert [(cue.start, cue.end) for cue in cues] == [(0.0, 0.0), (1.0, 1.0)]


def test_non_monotonic_speaker_header_is_flagged_without_reordering(
    tmp_path: Path,
) -> None:
    source = tmp_path / "non-monotonic.txt"
    source.write_text(
        "\n".join(
            [
                "Speaker_1 00:00:10",
                "第一段",
                "SPK-2 00:00:09",
                "第二段",
            ]
        ),
        encoding="utf-8",
    )

    cues = parse_transcript(source)

    assert [cue.speaker for cue in cues] == ["Speaker_1", "SPK-2"]
    assert cues[0].start == 10.0
    assert cues[0].end == 10.0
    assert cues[0].metadata["non_monotonic_next_start"] is True
    assert cues[1].start == 9.0


def test_getbrain_final_markdown_emoji_headers_are_imported(
    tmp_path: Path,
) -> None:
    source = tmp_path / "getbrain-final.md"
    source.write_text(
        "\n".join(
            [
                "# 📑 智能总结",
                "这部分是摘要，不应混入逐字稿。",
                "# 逐字稿",
                "🟢 说话人1 [00:00:00]",
                "第一段原文。",
                "🟣 说话人2 [00:00:05]",
                "第二段原文。",
            ]
        ),
        encoding="utf-8",
    )

    cues = parse_transcript(source)

    assert [cue.text for cue in cues] == ["第一段原文。", "第二段原文。"]
    assert [cue.speaker for cue in cues] == ["说话人1", "说话人2"]
    assert [(cue.start, cue.end) for cue in cues] == [(0.0, 5.0), (5.0, 5.0)]
