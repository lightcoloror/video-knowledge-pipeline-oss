from __future__ import annotations

from video_knowledge_pipeline.knowledge_note_export import (
    _reader_content_type,
    _reader_participant_count,
)


def test_reader_content_type_prefers_manifest_then_stdlib_mime() -> None:
    """Cover the existing-manifest/stdlib adapter without probing media.

    Intent: prevent audio-only runs from being labeled as video.
    Decision: explicit business content type wins; otherwise infer only the
    broad recording/video label from the existing source path and ``mimetypes``.
    Reason: FFprobe or another media registry would duplicate existing owners.
    Evidence: `.ogg` is an audio MIME type in the Python standard library.
    Effective scope: final reader metadata only.
    """

    assert (
        _reader_content_type(
            {
                "content_type": "客户沟通",
                "media_path": "D:/recordings/client-consultation.ogg",
            }
        )
        == "客户沟通"
    )
    assert (
        _reader_content_type(
            {"media_path": "D:/recordings/client-consultation.ogg"}
        )
        == "录音整理"
    )
    assert (
        _reader_content_type({"media_path": "D:/videos/lecture.mp4"})
        == "视频整理"
    )


def test_reader_participant_count_prefers_observed_diarization() -> None:
    """Keep speaker evidence authoritative over a stale declared count.

    Intent: display a participant count derived from actual speaker clusters.
    Decision: prefer transcript-quality-gate observations, then manifest fields.
    Reason: Smart Summary prose must not invent or override diarization.
    Evidence: the quality gate owns `distinct_speaker_count`.
    Effective scope: final reader metadata only.
    """

    assert (
        _reader_participant_count(
            {
                "participant_count": 3,
                "transcript_requirements": {"expected_speaker_count": 3},
            },
            {"speaker_diarization": {"distinct_speaker_count": 2}},
        )
        == 2
    )
    assert (
        _reader_participant_count(
            {"transcript_requirements": {"expected_speaker_count": 2}},
            {},
        )
        == 2
    )
