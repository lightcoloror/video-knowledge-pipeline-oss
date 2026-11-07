from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.knowledge_note_export import (
    _reader_participant_count,
    _speaker_review_notice,
    export_knowledge_note,
)


def test_two_speaker_source_fidelity_reaches_single_reader_document(
    tmp_path: Path,
) -> None:
    """Prove the existing export owners carry the speaker contract end to end.

    Intent: verify that a corrected two-person transcript reaches the single
    Smart Summary + transcript reader document.
    Decision: exercise ``export_knowledge_note`` rather than add a second test
    renderer or synthetic export path.
    Reason: unit-level speaker preservation is insufficient if a downstream
    exporter deduplicates or strips the labels.
    Evidence: the input follows pinned MOSS ``start/end/text/speaker`` segments.
    Effective scope: local contract verification only; no model, network,
    provider, upload, or production Bundle mutation.
    """

    bundle = tmp_path / "two-speaker-reader-export"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "lecture_webui_bundle.v1",
                "title": "双人项目沟通记录",
                "media_path": "D:/recordings/双人项目沟通记录.ogg",
                "content_type": "项目访谈",
                "source_arbitrated_transcript_json": (
                    "source-arbitrated-transcript.json"
                ),
                "participant_count": 2,
                "transcript_requirements": {
                    "speaker_diarization_required": True,
                    "expected_speaker_count": 2,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "source-arbitrated-transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "id": "seg-1",
                        "start": 0,
                        "end": 4,
                        "text": "根据排期来的嘛，你们现在有哪些准备？",
                        "speaker": "S01",
                    },
                    {
                        "id": "seg-2",
                        "start": 4,
                        "end": 8,
                        "text": "目前只有会议纪要，还没有整理行动项。",
                        "speaker": "S02",
                    },
                    {
                        "id": "seg-3",
                        "start": 8,
                        "end": 12,
                        "text": "后续可以由星河系统协助梳理。",
                        "speaker": "S01",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 12,
                    "transcript": "双方讨论现有材料与后续安排。",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = export_knowledge_note(
        bundle,
        run_transcript_evidence_check=False,
    )

    note = Path(result["note_path"]).read_text(encoding="utf-8")
    transcript = Path(result["full_transcript_path"]).read_text(encoding="utf-8")
    gate = result["transcript_quality_gate"]

    assert note.index("  - 📑 智能总结") < note.index("- 逐字稿")
    assert "      - **内容类型**：项目访谈" in note
    assert "      - **参与人数**：约 2 人" in note
    assert "🟢 说话人1" in note
    assert "🟣 说话人2" in note
    assert "根据排期来的嘛" in note
    assert "会议纪要" in note
    assert "星河系统" in note
    assert "  - 🟢 说话人1 [00:00:00.000]" in note
    assert "  - 🟣 说话人2 [00:00:04.000]" in note
    assert "collapsed::" not in note
    assert not any(line.lstrip().startswith("#") for line in note.splitlines())
    assert "source_arbitrated_transcript" not in note
    assert "arbitrated_or_reviewed" not in note
    assert transcript.count("**说话人1**") == 2
    assert transcript.count("**说话人2**") == 1
    assert gate["speaker_diarization"]["passed"] is True
    assert gate["speaker_diarization"]["distinct_speaker_count"] == 2


def test_human_confirmed_speaker_count_overrides_chunk_local_clusters() -> None:
    """Keep cluster evidence while using the confirmed participant count.

    Intent: prevent chunk-local cluster IDs from being reported as people.
    Decision: use a reviewed sidecar for reader metadata and show a notice,
    without rewriting individual speaker attributions.
    Reason: merging labels without global voice evidence can misattribute text.
    Evidence: production yielded five labels while the user confirmed three.
    Effective scope: reader exports only; canonical transcript is unchanged.
    """

    review = {
        "schema": "video_knowledge_pipeline.speaker_review.v1",
        "status": "human_confirmed_count",
        "confirmed_participant_count": 3,
        "observed_cluster_count": 5,
        "cluster_mapping_status": "unresolved_chunk_local_labels",
    }
    gate = {"speaker_diarization": {"distinct_speaker_count": 5}}

    assert _reader_participant_count({}, gate, review) == 3
    assert _reader_participant_count({}, gate, {}) == 5
    notice = _speaker_review_notice(review)
    assert "人工确认实际有 3 位主要说话人" in notice
    assert "5 个‘说话人’编号是分块声纹聚类标签" in notice
