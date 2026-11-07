from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_knowledge_pipeline import audio_loudness_recovery as recovery


MEASUREMENTS = {
    "input_i": -51.2,
    "input_tp": -34.1,
    "input_lra": 1.8,
    "input_thresh": -61.3,
    "target_offset": 0.1,
}


def _loudnorm_output(values: dict[str, float] | None = None) -> str:
    payload = values or MEASUREMENTS
    return "ffmpeg preamble\n" + json.dumps(payload, indent=2) + "\nffmpeg trailer"


def _active_probe(duration: float = 10.0) -> dict[str, object]:
    return {
        "ok": True,
        "status": "completed",
        "activity_intervals": [
            {"start": 0.0, "end": duration, "duration_seconds": duration}
        ],
        "silence_intervals": [],
    }


def _patch_local_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    duration: float = 10.0,
    silence_probe: dict[str, object] | None = None,
) -> None:
    monkeypatch.setattr(recovery, "resolve_media_tool", lambda _name: "C:/ffmpeg.exe")
    monkeypatch.setattr(
        recovery,
        "probe_video",
        lambda _path: SimpleNamespace(duration_seconds=duration),
    )
    monkeypatch.setattr(
        recovery,
        "probe_audio_silence",
        lambda *_args, **_kwargs: silence_probe or _active_probe(duration),
    )
    monkeypatch.setattr(recovery, "local_tool_subprocess_env", lambda: {})


def _write_wav(path: Path, *, sample_rate: int = 16000, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(b"\x00\x00" * sample_rate * channels)


def test_parse_loudnorm_measurements_ignores_surrounding_ffmpeg_logs() -> None:
    assert recovery.parse_loudnorm_measurements(_loudnorm_output()) == MEASUREMENTS


def test_parse_loudnorm_measurements_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        recovery.parse_loudnorm_measurements(
            _loudnorm_output({**MEASUREMENTS, "input_i": float("-inf")})
        )


def test_plan_is_candidate_only_and_does_not_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source-audio")
    monkeypatch.setattr(recovery, "resolve_media_tool", lambda _name: "C:/ffmpeg.exe")
    monkeypatch.setattr(
        recovery.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("plan must not run FFmpeg"),
    )

    result = recovery.prepare_low_level_audio_candidate(
        source,
        execute=False,
        write=False,
    )

    assert result["status"] == "planned"
    assert result["candidate_only"] is True
    assert result["speech_proven"] is False
    assert result["asr_retry_authorized"] is False
    assert result["operator_boundary"]["automatic_asr_retry"] is False
    assert result["upstream"]["commit"] == recovery.UPSTREAM_COMMIT


def test_near_silence_blocks_normalization_and_preserves_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "near-silence.wav"
    source.write_bytes(b"unchanged-source")
    before = recovery.sha256_file(source)
    _patch_local_runtime(
        monkeypatch,
        silence_probe={
            "ok": True,
            "status": "completed",
            "activity_intervals": [
                {"start": 0.0, "end": 0.3, "duration_seconds": 0.3}
            ],
            "silence_intervals": [
                {"start": 0.3, "end": 10.0, "duration_seconds": 9.7}
            ],
        },
    )
    monkeypatch.setattr(
        recovery.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("near-silence must not run loudnorm"),
    )

    result = recovery.prepare_low_level_audio_candidate(
        source,
        execute=True,
        render_candidate=True,
        write=False,
    )

    assert result["status"] == "near_silence_blocked"
    assert result["ok"] is True
    assert result["activity_gate"]["passed"] is False
    assert result["asr_retry_authorized"] is False
    assert recovery.sha256_file(source) == before
    assert not (tmp_path / "near-silence.loudness-recovery-candidate.wav").exists()


def test_low_level_analysis_never_authorizes_asr_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "low-level.wav"
    source.write_bytes(b"low-level-source")
    _patch_local_runtime(monkeypatch)
    monkeypatch.setattr(
        recovery.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr=_loudnorm_output(),
        ),
    )

    result = recovery.prepare_low_level_audio_candidate(
        source,
        execute=True,
        render_candidate=False,
        write=False,
    )

    assert result["status"] == "low_level_candidate_detected"
    assert result["loudness_measurements"]["input_i"] == -51.2
    assert result["speech_proven"] is False
    assert result["asr_retry_authorized"] is False
    assert "speech VAD" in result["recommended_action"]


def test_normal_loudness_bypasses_candidate_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "normal.wav"
    source.write_bytes(b"normal-source")
    _patch_local_runtime(monkeypatch)
    calls = 0

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr=_loudnorm_output({**MEASUREMENTS, "input_i": -24.0}),
        )

    monkeypatch.setattr(recovery.subprocess, "run", fake_run)

    result = recovery.prepare_low_level_audio_candidate(
        source,
        execute=True,
        render_candidate=True,
        write=False,
    )

    assert result["status"] == "normalization_not_needed"
    assert calls == 1
    assert result["asr_retry_authorized"] is False


def test_measured_second_pass_writes_valid_sidecar_and_keeps_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "low-level.wav"
    source.write_bytes(b"original-source-bytes")
    candidate = tmp_path / "candidate.wav"
    before = recovery.sha256_file(source)
    _patch_local_runtime(monkeypatch)
    commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if len(commands) == 1:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="",
                stderr=_loudnorm_output(),
            )
        _write_wav(Path(command[-1]))
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="",
            stderr="rendered",
        )

    monkeypatch.setattr(recovery.subprocess, "run", fake_run)

    result = recovery.prepare_low_level_audio_candidate(
        source,
        output_path=candidate,
        execute=True,
        render_candidate=True,
        write=False,
    )

    assert result["status"] == "candidate_requires_speech_vad"
    assert result["candidate_format_validation"]["valid"] is True
    assert result["candidate_output"]["sha256"] == recovery.sha256_file(candidate)
    assert result["asr_retry_authorized"] is False
    assert recovery.sha256_file(source) == before
    render_filter = commands[1][commands[1].index("-af") + 1]
    assert "measured_I=-51.2" in render_filter
    assert "linear=true" in render_filter
    assert commands[1][commands[1].index("-ar") + 1] == "16000"
    assert commands[1][commands[1].index("-ac") + 1] == "1"
    assert commands[1][commands[1].index("-c:a") + 1] == "pcm_s16le"


def test_existing_candidate_is_not_silently_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "low-level.wav"
    source.write_bytes(b"source")
    candidate = tmp_path / "candidate.wav"
    candidate.write_bytes(b"existing-candidate")
    before = candidate.read_bytes()
    _patch_local_runtime(monkeypatch)
    calls = 0

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr=_loudnorm_output(),
        )

    monkeypatch.setattr(recovery.subprocess, "run", fake_run)

    result = recovery.prepare_low_level_audio_candidate(
        source,
        output_path=candidate,
        execute=True,
        render_candidate=True,
        write=False,
    )

    assert result["status"] == "candidate_output_exists"
    assert result["candidate_output"]["exists"] is True
    assert result["candidate_output"]["produced_this_run"] is False
    assert candidate.read_bytes() == before
    assert calls == 1


def test_invalid_first_pass_fails_closed_without_simple_gain_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "invalid.wav"
    source.write_bytes(b"source")
    _patch_local_runtime(monkeypatch)
    monkeypatch.setattr(
        recovery.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="no loudnorm json",
        ),
    )

    result = recovery.prepare_low_level_audio_candidate(
        source,
        execute=True,
        render_candidate=True,
        write=False,
    )

    assert result["status"] == "loudness_measurements_invalid"
    assert result["ok"] is False
    assert result["candidate_output"]["exists"] is False
    assert result["upstream"]["rejected_behavior"] == [
        "simple_gain_fallback",
        "mp3_reencode_fallback",
    ]
