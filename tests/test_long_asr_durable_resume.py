from __future__ import annotations

import json
import os
import sys
import types
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

import video_knowledge_pipeline.qwen3_asr_python_runner as runner
from video_knowledge_pipeline import asr_execution
from video_knowledge_pipeline.asr_runner import _command_for_preset
from video_knowledge_pipeline.file_hash import sha256_file


def _contract() -> dict:
    return {
        "input_identity": {
            "path": "synthetic://eight-hour-timeline",
            "bytes": 32,
            "mtime_ns": 1,
            "sha256": "a" * 64,
        },
        "model": "fixture/qwen3-asr",
        "forced_aligner": "fixture/aligner",
        "language": "Chinese",
        "context_sha256": "b" * 64,
        "chunk_seconds": 300,
        "max_new_tokens": 1024,
        "dtype": "auto",
        "chunk_indexes": [],
        "timestamp_input_basis": "chunk_relative",
        "vad_evidence_sha256": "c" * 64,
    }


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    duration_seconds: float,
    extraction_calls: list[int],
    fail_extraction_indexes: set[int] | None = None,
) -> None:
    failures = fail_extraction_indexes if fail_extraction_indexes is not None else set()
    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    torch.cuda = SimpleNamespace(is_available=lambda: False)

    class Runtime:
        def transcribe(self, *, audio, context, language, return_time_stamps):
            index = int(Path(audio).stem.rsplit("-", 1)[-1])
            return [
                SimpleNamespace(
                    text=f"synthetic window {index}",
                    language=language or "Chinese",
                    time_stamps=[
                        {
                            "text": f"synthetic window {index}",
                            "start_time": 1.0,
                            "end_time": 2.0,
                        }
                    ],
                )
            ]

    class Qwen3ASRModel:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            return Runtime()

    qwen_asr = types.ModuleType("qwen_asr")
    qwen_asr.Qwen3ASRModel = Qwen3ASRModel
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "qwen_asr", qwen_asr)
    monkeypatch.setattr(runner, "_media_duration_seconds", lambda _media: duration_seconds)
    monkeypatch.setattr(
        runner,
        "_model_ready",
        lambda *, preset, model: {
            "ready": True,
            "status": "synthetic_ready",
            "preset": preset,
            "cache_matches": [model],
        },
    )

    def fake_extract(
        media: Path,
        output_dir: Path,
        *,
        index: int,
        start_seconds: float,
        end_seconds: float,
        ffmpeg_path=None,
        timeout_seconds: int = 900,
    ) -> dict:
        extraction_calls.append(index)
        if index in failures:
            raise TimeoutError(f"synthetic timeout for window {index}")
        target = output_dir / f"chunk-{index:04d}.wav"
        target.write_bytes(f"synthetic-{index}".encode())
        return {
            "index": index,
            "path": str(target),
            "bytes": target.stat().st_size,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "duration_seconds": end_seconds - start_seconds,
            "command": ["synthetic-ffmpeg", str(target)],
        }

    monkeypatch.setattr(runner, "extract_audio_chunk_window", fake_extract)


def _run(
    media: Path,
    output: Path,
    *,
    chunk_seconds: int = 300,
    vad_json: Path | None = None,
) -> dict:
    return runner.run_qwen3_asr(
        input_path=str(media),
        output_path=str(output),
        model="fixture/qwen3-asr",
        forced_aligner="fixture/aligner",
        chunk_seconds=chunk_seconds,
        timestamp_basis="chunk_relative",
        vad_json=str(vad_json) if vad_json else None,
    )


def test_eight_hour_window_plan_has_stable_identity_and_vad_priority() -> None:
    plan = runner.build_qwen_window_plan(
        duration_seconds=8 * 60 * 60,
        chunk_seconds=300,
        execution_contract=_contract(),
        vad_intervals=[
            {"start": 0, "end": 300, "confidence": 0.25, "segment_id": "low"},
            {"start": 600, "end": 900, "confidence": 0.95, "segment_id": "speech"},
        ],
        audio_activity_intervals=[
            {"start": 300, "end": 600, "candidate_only": True},
        ],
    )

    assert plan["schema"] == "video_knowledge_pipeline.qwen_asr_window_plan.v2"
    assert plan["duration_ms"] == 28_800_000
    assert len(plan["windows"]) == 96
    assert len({row["window_id"] for row in plan["windows"]}) == 96
    assert plan["windows"][0]["start_ms"] == 0
    assert plan["windows"][-1]["end_ms"] == 28_800_000
    assert all(row["input_sha256"] == "a" * 64 for row in plan["windows"])
    assert all(row["window_input_sha256"] for row in plan["windows"])
    assert all(row["dependencies"] for row in plan["windows"])
    assert all(row["retention"] == "keep" for row in plan["windows"])
    assert plan["windows"][2]["priority"] == "vad_speech_first"
    assert plan["windows"][2]["execution_order"] == 0
    assert plan["windows"][0]["priority"] == "low_confidence_review_keep"
    assert plan["windows"][0]["review_required"] is True
    assert plan["windows"][1]["priority"] == "audio_activity_review_keep"
    assert plan["windows"][1]["speech_proven"] is False
    assert plan["revision"]


def test_timestamp_basis_converts_chunk_relative_once_and_preserves_absolute() -> None:
    relative = runner.normalize_qwen_window_timestamps(
        [{"text": "x", "start": 1.0, "end": 2.0}],
        input_basis="chunk_relative",
        window_start_seconds=30,
        window_end_seconds=60,
    )
    absolute = runner.normalize_qwen_window_timestamps(
        relative["timestamps"],
        input_basis="media_absolute",
        window_start_seconds=30,
        window_end_seconds=60,
    )

    assert relative["timestamps"] == [{"text": "x", "start": 31.0, "end": 32.0}]
    assert relative["offset_applied"] is True
    assert absolute["timestamps"] == relative["timestamps"]
    assert absolute["offset_applied"] is False
    assert absolute["output_basis"] == "media_absolute"
    with pytest.raises(ValueError, match="outside_window"):
        runner.normalize_qwen_window_timestamps(
            [{"text": "x", "start": 61.0, "end": 62.0}],
            input_basis="media_absolute",
            window_start_seconds=30,
            window_end_seconds=60,
        )


def test_eight_hour_timeout_resume_matches_uninterrupted_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "synthetic-eight-hour.wav"
    media.write_bytes(b"synthetic eight-hour timeline; not media content")
    baseline_calls: list[int] = []
    _install_runtime(
        monkeypatch,
        duration_seconds=8 * 60 * 60,
        extraction_calls=baseline_calls,
    )
    baseline = _run(media, tmp_path / "baseline.json")

    interrupted_calls: list[int] = []
    _install_runtime(
        monkeypatch,
        duration_seconds=8 * 60 * 60,
        extraction_calls=interrupted_calls,
        fail_extraction_indexes={39},
    )
    interrupted = _run(media, tmp_path / "resumed.json")
    resume_calls: list[int] = []
    _install_runtime(
        monkeypatch,
        duration_seconds=8 * 60 * 60,
        extraction_calls=resume_calls,
    )
    resumed = _run(media, tmp_path / "resumed.json")

    assert baseline["status"] == "completed"
    assert baseline_calls == list(range(96))
    assert interrupted["status"] == "degraded"
    assert interrupted["successful_chunk_count"] == 95
    assert interrupted["failed_chunk_count"] == 1
    assert resume_calls == [39]
    assert resumed["status"] == "completed"
    assert resumed["resumed_from_checkpoint"] is True
    assert resumed["results"] == baseline["results"]
    assert resumed["segments"] == baseline["segments"]
    assert resumed["text"] == baseline["text"]
    assert resumed["successful_chunk_indexes"] == baseline["successful_chunk_indexes"]
    assert Counter(interrupted_calls + resume_calls) == Counter(
        {index: (2 if index == 39 else 1) for index in range(96)}
    )
    checkpoint = json.loads(Path(resumed["checkpoint_path"]).read_text(encoding="utf-8"))
    assert checkpoint["schema"] == "video_knowledge_pipeline.qwen3_asr_checkpoint.v2"
    assert checkpoint["input_identity"]["sha256"] == sha256_file(media)
    assert checkpoint["window_plan"]["revision"]
    assert checkpoint["checkpoint_sha256"] == runner.qwen_checkpoint_payload_sha256(checkpoint)
    assert all(row["timestamp_basis"] == "media_absolute" for row in resumed["results"])


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("missing", "checkpoint_success_index_mismatch"),
        ("duplicate", "checkpoint_duplicate_result"),
        ("out_of_order", "checkpoint_result_order_invalid"),
    ],
)
def test_structurally_invalid_checkpoint_is_rejected_without_new_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_reason: str,
) -> None:
    case = tmp_path / mutation
    case.mkdir()
    media = case / "input.wav"
    media.write_bytes(b"synthetic")
    first_calls: list[int] = []
    _install_runtime(monkeypatch, duration_seconds=90, extraction_calls=first_calls)
    first = _run(media, case / "output.json", chunk_seconds=30)
    checkpoint_path = Path(first["checkpoint_path"])
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        payload["results"].pop()
    elif mutation == "duplicate":
        payload["results"].append(dict(payload["results"][0]))
    else:
        payload["results"] = list(reversed(payload["results"]))
    payload["checkpoint_sha256"] = runner.qwen_checkpoint_payload_sha256(payload)
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")

    resume_calls: list[int] = []
    _install_runtime(monkeypatch, duration_seconds=90, extraction_calls=resume_calls)
    rejected = _run(media, case / "output.json", chunk_seconds=30)

    assert rejected["status"] == "failed"
    assert rejected["error_code"] == "qwen3_asr_checkpoint_rejected"
    assert rejected["checkpoint_resume_rejected_reason"] == expected_reason
    assert resume_calls == []


def test_corrupt_or_parameter_drift_checkpoint_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "input.wav"
    media.write_bytes(b"synthetic")
    first_calls: list[int] = []
    _install_runtime(monkeypatch, duration_seconds=90, extraction_calls=first_calls)
    first = _run(media, tmp_path / "corrupt.json", chunk_seconds=30)
    checkpoint = Path(first["checkpoint_path"])
    checkpoint.write_text("{broken", encoding="utf-8")

    corrupt_calls: list[int] = []
    _install_runtime(monkeypatch, duration_seconds=90, extraction_calls=corrupt_calls)
    corrupt = _run(media, tmp_path / "corrupt.json", chunk_seconds=30)
    assert corrupt["error_code"] == "qwen3_asr_checkpoint_rejected"
    assert corrupt["checkpoint_resume_rejected_reason"] == "checkpoint_unreadable"
    assert corrupt_calls == []

    drift_calls: list[int] = []
    _install_runtime(monkeypatch, duration_seconds=90, extraction_calls=drift_calls)
    valid = _run(media, tmp_path / "drift.json", chunk_seconds=30)
    assert valid["status"] == "completed"
    drift_calls.clear()
    _install_runtime(monkeypatch, duration_seconds=90, extraction_calls=drift_calls)
    drift = _run(media, tmp_path / "drift.json", chunk_seconds=60)
    assert drift["error_code"] == "qwen3_asr_checkpoint_rejected"
    assert drift["checkpoint_resume_rejected_reason"] == "execution_contract_mismatch"
    assert drift_calls == []


def test_same_size_and_mtime_input_hash_drift_is_rejected_before_new_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "input.wav"
    media.write_bytes(b"synthetic-A")
    original_stat = media.stat()
    first_calls: list[int] = []
    _install_runtime(monkeypatch, duration_seconds=90, extraction_calls=first_calls)
    first = _run(media, tmp_path / "output.json", chunk_seconds=30)
    assert first["status"] == "completed"

    media.write_bytes(b"synthetic-B")
    os.utime(media, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert media.stat().st_size == original_stat.st_size
    drift_calls: list[int] = []
    _install_runtime(monkeypatch, duration_seconds=90, extraction_calls=drift_calls)
    rejected = _run(media, tmp_path / "output.json", chunk_seconds=30)

    assert rejected["error_code"] == "qwen3_asr_checkpoint_rejected"
    assert rejected["checkpoint_resume_rejected_reason"] == "execution_contract_mismatch"
    assert drift_calls == []


def test_vad_speech_windows_execute_first_without_dropping_other_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "input.wav"
    media.write_bytes(b"synthetic")
    vad = tmp_path / "vad.json"
    vad.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.silero_vad_candidate.v1",
                "segments": [
                    {"id": "low", "start": 0, "end": 30, "confidence": 0.2},
                    {"id": "speech", "start": 30, "end": 60, "confidence": 0.95},
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: list[int] = []
    _install_runtime(monkeypatch, duration_seconds=90, extraction_calls=calls)
    result = _run(media, tmp_path / "output.json", chunk_seconds=30, vad_json=vad)

    assert result["status"] == "completed"
    assert calls == [1, 0, 2]
    assert result["successful_chunk_indexes"] == [0, 1, 2]
    windows = result["window_plan"]["windows"]
    assert windows[1]["priority"] == "vad_speech_first"
    assert windows[0]["priority"] == "low_confidence_review_keep"
    assert windows[2]["retention"] == "keep"


def test_qwen_plan_command_exposes_stable_vad_and_timestamp_contract(tmp_path: Path) -> None:
    media = tmp_path / "input.wav"
    media.write_bytes(b"synthetic")
    vad = tmp_path / "vad.json"
    vad.write_text('{"segments": []}', encoding="utf-8")
    command = _command_for_preset(
        preset="qwen3-asr-0.6b",
        command="unused",
        media=media,
        output_dir=tmp_path,
        output_json=tmp_path / "output.json",
        language="Chinese",
        model="fixture/qwen3-asr",
        punc_model=None,
        spk_model=None,
        use_itn=True,
        merge_vad=True,
        merge_length_s=15,
        vad_max_single_segment_time_ms=30000,
        chunk_boundary_mode="fixed_duration",
        chunk_overlap_seconds=5.0,
        python_executable=sys.executable,
        local_device="cpu",
        use_python_runner=True,
        qwen_timestamps=True,
        alignment_transcript=None,
        qwen_timestamp_basis="chunk_relative",
        vad_json=vad,
    )

    assert command[command.index("--timestamp-basis") + 1] == "chunk_relative"
    assert command[command.index("--vad-json") + 1] == str(vad)


def test_asr_plan_recovery_rebuilds_output_from_complete_v2_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "input.wav"
    media.write_bytes(b"synthetic")
    output = tmp_path / "output.json"
    calls: list[int] = []
    _install_runtime(monkeypatch, duration_seconds=90, extraction_calls=calls)
    completed = _run(media, output, chunk_seconds=30)
    expected_results = completed["results"]
    output.unlink()
    command = [
        sys.executable,
        "-m",
        "video_knowledge_pipeline.qwen3_asr_python_runner",
        "--input",
        str(media.resolve()),
        "--output",
        str(output.resolve()),
        "--model",
        "fixture/qwen3-asr",
        "--forced-aligner",
        "fixture/aligner",
        "--language",
        "Chinese",
        "--chunk-seconds",
        "30",
        "--timestamp-basis",
        "chunk_relative",
    ]

    details = asr_execution._qwen_checkpoint_resume_details(output, command)
    assert details["checkpoint_resume_eligible"] is True
    assert details["checkpoint_complete"] is True
    asr_execution._restore_qwen_output_from_checkpoint(output, command, details)
    rebuilt = json.loads(output.read_text(encoding="utf-8"))
    assert rebuilt["results"] == expected_results
    assert rebuilt["checkpoint_recovery"] == "raw_output_rebuilt_without_model_execution"
    assert rebuilt["timestamp_output_basis"] == "media_absolute"
