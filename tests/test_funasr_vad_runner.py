from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import video_knowledge_pipeline.funasr_python_runner as python_runner
import video_knowledge_pipeline.funasr_vad_runner as runner


def test_funasr_vad_runner_locks_threshold_profile_and_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    media = tmp_path / "audio.wav"
    media.write_bytes(b"audio")
    output = tmp_path / "vad.json"
    captured: dict[str, object] = {}

    class FakeAutoModel:
        def __init__(self, **kwargs: object) -> None:
            captured["model_kwargs"] = kwargs

        def generate(self, **kwargs: object) -> list[dict[str, object]]:
            captured["generate_kwargs"] = kwargs
            return [{"value": [[1000, 2400], [3000, 4500]]}]

    monkeypatch.setitem(
        sys.modules, "funasr", type("FakeFunASR", (), {"AutoModel": FakeAutoModel})
    )
    monkeypatch.setattr(runner, "_select_device", lambda _device: "cuda")
    monkeypatch.setattr(
        python_runner, "_resolve_local_model", lambda _model: "D:/models/fsmn-vad"
    )

    result = runner.run_vad(
        input_path=str(media),
        output_path=str(output),
        speech_noise_threshold=0.35,
        max_end_silence_time_ms=1100,
        max_single_segment_time_ms=24000,
        evidence_profile="candidate-permissive",
    )

    assert result["ok"] is True
    assert result["candidate_only"] is True
    assert captured["model_kwargs"] == {
        "model": "D:/models/fsmn-vad",
        "model_revision": "v2.0.4",
        "disable_update": True,
        "speech_noise_thres": 0.35,
        "max_end_silence_time": 1100,
        "device": "cuda",
    }
    assert captured["generate_kwargs"] == {
        "input": str(media.resolve()),
        "cache": {},
        "is_final": True,
        "max_single_segment_time": 24000,
    }
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["evidence_profile"] == "candidate-permissive"
    assert payload["candidate_only"] is True
    assert payload["vad_settings"] == {
        "max_single_segment_time_ms": 24000,
        "speech_noise_threshold": 0.35,
        "max_end_silence_time_ms": 1100,
    }
    assert payload["segments"] == [
        {"start": 1.0, "end": 2.4},
        {"start": 3.0, "end": 4.5},
    ]


def test_funasr_vad_runner_rejects_unsafe_threshold_without_loading_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    media = tmp_path / "audio.wav"
    media.write_bytes(b"audio")
    monkeypatch.delitem(sys.modules, "funasr", raising=False)

    result = runner.run_vad(
        input_path=str(media),
        output_path=str(tmp_path / "vad.json"),
        speech_noise_threshold=-0.1,
    )

    assert result["ok"] is False
    assert result["error"] == "speech_noise_threshold must be between 0 and 1"
    assert not (tmp_path / "vad.json").exists()
