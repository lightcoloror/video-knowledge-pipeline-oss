from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline import silero_vad_candidate as candidate_module
from video_knowledge_pipeline.asr_vad_independent_crosscheck import (
    crosscheck_asr_vad_with_independent_candidate,
)
from video_knowledge_pipeline.cli import main as cli_main
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.silero_vad_candidate import (
    SCHEMA as SILERO_SCHEMA,
    run_silero_vad_candidate,
)
from video_knowledge_pipeline.storage import write_json


@pytest.mark.parametrize("rows", [None, {}, [None], [{"start": 9, "end": 2}]] + [
    [{"start": value, "end": 16000}] for value in [None, 'bad', True, -1, 0.5, float('nan'), float('inf')]])
def test_invalid_vad_intervals_do_not_become_silence(rows):
    with pytest.raises(ValueError):
        candidate_module._normalise_segments(rows)


def test_v6_asset_provenance_follows_installed_factory(monkeypatch, tmp_path):
    import sys
    from types import ModuleType
    modules = {name: ModuleType(name) for name in ["faster_whisper", "faster_whisper.audio", "faster_whisper.utils", "faster_whisper.vad"]}
    modules["faster_whisper.audio"].decode_audio = lambda *a, **k: None
    modules["faster_whisper.utils"].get_assets_path = lambda: str(tmp_path)
    def factory():
        return "silero_vad_v6.onnx"
    modules["faster_whisper.vad"].get_vad_model = factory
    modules["faster_whisper.vad"].VadOptions = object
    modules["faster_whisper.vad"].get_speech_timestamps = lambda *a, **k: []
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    asset = tmp_path / "silero_vad_v6.onnx"
    asset.write_bytes(b"synthetic ONNX asset")
    _, _, _, assets = candidate_module._load_faster_whisper_vad_api()
    assert assets[0]["name"] == asset.name
    assert assets[0]["sha256"] == sha256_file(asset)


def test_cli_failed_vad_is_nonzero_and_not_silence(monkeypatch, tmp_path, capsys):
    from video_knowledge_pipeline import cli
    monkeypatch.setattr(cli, "run_silero_vad_candidate", lambda *a, **kw: {
        "ok": False, "status": "runtime_unavailable", "segments": [], "segment_count": 0})
    assert cli_main(["silero-vad-candidate", str(tmp_path / "audio.wav"), "--execute", "--no-write"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "runtime_unavailable"


def test_silero_candidate_preview_does_not_load_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    media = tmp_path / "lesson.wav"
    media.write_bytes(b"audio fixture")

    def forbidden_loader() -> object:
        raise AssertionError("preview must not import or load the VAD model")

    monkeypatch.setattr(candidate_module, "_load_faster_whisper_vad_api", forbidden_loader)

    result = run_silero_vad_candidate(media, execute=False, write=False)

    assert result["status"] == "planned"
    assert result["candidate_only"] is True
    assert result["network_call"] is False
    assert result["operator_boundary"]["model_download_allowed"] is False


def test_silero_candidate_reuses_faster_whisper_vad_api_and_records_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    media = tmp_path / "lesson.wav"
    media.write_bytes(b"audio fixture")
    output = tmp_path / "silero.json"
    calls: dict[str, object] = {}

    class FakeVadOptions:
        def __init__(self, **kwargs: object) -> None:
            calls["options"] = kwargs

    def fake_decode(path: str, *, sampling_rate: int) -> list[float]:
        calls["decode"] = (path, sampling_rate)
        return [0.0] * 48000

    def fake_timestamps(
        audio: object, *, vad_options: object, sampling_rate: int
    ) -> list[dict[str, int]]:
        calls["timestamps"] = (audio, vad_options, sampling_rate)
        return [
            {"start": 8000, "end": 24000},
            {"start": 32000, "end": 48000},
        ]

    monkeypatch.setattr(
        candidate_module,
        "_runtime_status",
        lambda: {
            "available": True,
            "installed_version": "1.1.1",
            "bundled_model_expected": "silero v5",
            "download_required": False,
        },
    )
    monkeypatch.setattr(
        candidate_module,
        "_load_faster_whisper_vad_api",
        lambda: (
            fake_decode,
            fake_timestamps,
            FakeVadOptions,
            [{"name": "silero_encoder_v5.onnx", "sha256": "a" * 64}],
        ),
    )

    result = run_silero_vad_candidate(
        media,
        output_path=output,
        execute=True,
        speech_pad_ms=300,
    )

    assert result["status"] == "completed"
    assert result["segment_count"] == 2
    assert result["segments"][0] == {
        "segment_id": "silero-vad-0001",
        "start": 0.5,
        "end": 1.5,
        "start_sample": 8000,
        "end_sample": 24000,
    }
    assert calls["decode"] == (str(media.resolve()), 16000)
    assert calls["options"]["speech_pad_ms"] == 300  # type: ignore[index]
    assert result["upstream"]["installed_version"] == "1.1.1"
    assert output.is_file()


def _crosscheck_fixtures(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    media = tmp_path / "lesson.wav"
    media.write_bytes(b"crosscheck audio")
    authoritative = tmp_path / "funasr-vad.json"
    write_json(
        authoritative,
        {
            "schema": "video_knowledge_pipeline.funasr_vad_segments.v1",
            "input": str(media.resolve()),
            "model": "fsmn-vad",
            "model_revision": "v2.0.4",
            "evidence_profile": "authoritative",
            "candidate_only": False,
            "segments": [
                {"start": 0.0, "end": 5.0},
                {"start": 10.0, "end": 15.0},
            ],
        },
    )
    candidate = tmp_path / "silero-vad.json"
    write_json(
        candidate,
        {
            "schema": SILERO_SCHEMA,
            "status": "completed",
            "candidate_only": True,
            "source_media": {
                "path": str(media.resolve()),
                "sha256": sha256_file(media),
            },
            "upstream": {
                "model": "silero_v5_bundled_onnx",
                "installed_version": "1.1.1",
            },
            "segments": [
                {"start": 0.0, "end": 5.0},
                {"start": 6.0, "end": 8.0},
                {"start": 10.0, "end": 15.0},
            ],
        },
    )
    audit = tmp_path / "activity-audit.json"
    write_json(
        audit,
        {
            "schema": "video_knowledge_pipeline.asr_vad_activity_audit.v1",
            "status": "review_required",
            "vad_sha256": sha256_file(authoritative),
            "source_media": {"sha256": sha256_file(media)},
            "audio_probe": {"activity_intervals": [{"start": 0.0, "end": 20.0}]},
        },
    )
    return media, authoritative, candidate, audit


def test_independent_crosscheck_surfaces_only_uncovered_silero_speech(
    tmp_path: Path,
) -> None:
    _, authoritative, candidate, audit = _crosscheck_fixtures(tmp_path)

    result = crosscheck_asr_vad_with_independent_candidate(
        authoritative,
        candidate,
        activity_audit_path=audit,
        write=False,
    )

    assert result["status"] == "review_required"
    assert result["candidate_gap_count"] == 1
    gap = result["candidate_gaps"][0]
    assert (gap["start"], gap["end"]) == (6.0, 8.0)
    assert gap["independent_model_support"] is True
    assert gap["audio_activity_support_ratio"] == 1.0
    assert gap["automatic_acceptance_allowed"] is False
    assert result["decision_boundary"]["canonical_transcript_modified"] is False


def test_independent_crosscheck_rejects_stale_media_hash(tmp_path: Path) -> None:
    media, authoritative, candidate, _ = _crosscheck_fixtures(tmp_path)
    media.write_bytes(b"mutated audio")

    with pytest.raises(ValueError, match="source media hash is stale"):
        crosscheck_asr_vad_with_independent_candidate(
            authoritative,
            candidate,
            write=False,
        )


def test_independent_crosscheck_accepts_completed_silero_with_no_speech(
    tmp_path: Path,
) -> None:
    _, authoritative, candidate, _ = _crosscheck_fixtures(tmp_path)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["segments"] = []
    write_json(candidate, payload)

    result = crosscheck_asr_vad_with_independent_candidate(
        authoritative,
        candidate,
        write=False,
    )

    assert result["status"] == "passed"
    assert result["candidate_gap_count"] == 0


def test_silero_candidate_and_crosscheck_cli_are_local_preview_commands(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    media, authoritative, candidate, _ = _crosscheck_fixtures(tmp_path)

    assert (
        cli_main(["silero-vad-candidate", str(media), "--no-write"])
        == 0
    )
    preview = json.loads(capsys.readouterr().out)
    assert preview["status"] == "planned"
    assert preview["network_call"] is False

    assert (
        cli_main(
            [
                "asr-vad-independent-crosscheck",
                str(authoritative),
                str(candidate),
                "--no-write",
            ]
        )
        == 0
    )
    crosscheck = json.loads(capsys.readouterr().out)
    assert crosscheck["status"] == "review_required"
    assert crosscheck["decision_boundary"]["network_call"] is False
