from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from video_knowledge_pipeline.audio_silence_probe import (
    parse_audio_silence_output,
    probe_audio_silence,
)
from video_knowledge_pipeline.quality_benchmark import (
    _align_window_to_audio_silence,
)


def test_parse_audio_silence_output_returns_activity_complement() -> None:
    result = parse_audio_silence_output(
        "silence_start: 0\nsilence_end: 2 | silence_duration: 2\n"
        "silence_start: 5\nsilence_end: 7 | silence_duration: 2\n",
        window_start=10,
        window_end=20,
    )

    assert result["silence_intervals"] == [
        {"start": 10.0, "end": 12.0, "duration_seconds": 2.0},
        {"start": 15.0, "end": 17.0, "duration_seconds": 2.0},
    ]
    assert result["activity_intervals"] == [
        {"start": 12.0, "end": 15.0, "duration_seconds": 3.0},
        {"start": 17.0, "end": 20.0, "duration_seconds": 3.0},
    ]


def test_probe_audio_silence_reuses_registered_ffmpeg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    media = tmp_path / "source.mp4"
    media.write_bytes(b"media")
    monkeypatch.setattr(
        "video_knowledge_pipeline.audio_silence_probe.resolve_media_tool",
        lambda _: "ffmpeg-test",
    )
    seen: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        seen["command"] = command
        seen["env"] = kwargs.get("env")
        return SimpleNamespace(
            returncode=0,
            stderr="silence_start: 2\nsilence_end: 4 | silence_duration: 2",
            stdout="",
        )

    monkeypatch.setattr(
        "video_knowledge_pipeline.audio_silence_probe.subprocess.run", fake_run
    )

    result = probe_audio_silence(media, start=5, end=15)

    assert result["status"] == "completed"
    assert result["command"][0] == "ffmpeg-test"
    assert seen["command"] == result["command"]
    assert isinstance(seen["env"], dict)
    assert result["silence_intervals"][0]["start"] == 7.0
    assert result["silence_intervals"][0]["end"] == 9.0


def test_quality_benchmark_alignment_reuses_shared_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    media = tmp_path / "source.mp4"
    media.write_bytes(b"media")
    monkeypatch.setattr(
        "video_knowledge_pipeline.quality_benchmark.probe_audio_silence",
        lambda *_args, **_kwargs: {
            "ok": True,
            "status": "completed",
            "silence_starts": [0.0, 20.0],
            "silence_ends": [5.0, 25.0],
            "silence_intervals": [{}, {}],
        },
    )

    start, end, evidence = _align_window_to_audio_silence(
        media, start=10, end=15, duration=30
    )

    assert start == 5.05
    assert end == 19.95
    assert evidence["ready"] is True
    assert evidence["source"] == "ffmpeg_silencedetect"
