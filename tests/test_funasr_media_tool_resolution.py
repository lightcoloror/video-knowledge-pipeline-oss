from __future__ import annotations

import subprocess
from pathlib import Path

import video_knowledge_pipeline.funasr_python_runner as runner


def test_duration_probe_reuses_shared_media_tool_resolver(
    tmp_path: Path,
    monkeypatch,
) -> None:
    media = tmp_path / "audio.wav"
    media.write_bytes(b"wav")
    resolved_probe = tmp_path / "ffprobe.exe"
    resolved_probe.write_bytes(b"tool")
    calls: list[list[str]] = []

    monkeypatch.delenv("FFPROBE", raising=False)
    monkeypatch.setattr(
        runner,
        "resolve_media_tool",
        lambda name: str(resolved_probe) if name == "ffprobe" else "",
    )

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, "6259.583667\n", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner._media_duration_seconds(media) == 6259.583667
    assert calls[0][0] == str(resolved_probe)


def test_duration_probe_preserves_legacy_ffprobe_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    media = tmp_path / "audio.wav"
    media.write_bytes(b"wav")
    monkeypatch.setenv("FFPROBE", "operator-ffprobe")
    monkeypatch.setattr(
        runner,
        "resolve_media_tool",
        lambda _name: (_ for _ in ()).throw(AssertionError("resolver not expected")),
    )
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, "10.25\n", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner._media_duration_seconds(media) == 10.25
    assert calls[0][0] == "operator-ffprobe"
