from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import video_knowledge_pipeline.qwen3_asr_python_runner as runner


def _install_synthetic_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_extraction_indexes: set[int] | None = None,
    extracted_indexes: list[int] | None = None,
    local_timestamp: tuple[float, float] = (0.0, 1.0),
) -> None:
    failures = fail_extraction_indexes if fail_extraction_indexes is not None else set()
    extraction_calls = extracted_indexes if extracted_indexes is not None else []

    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    torch.cuda = SimpleNamespace(is_available=lambda: False)

    class Runtime:
        def transcribe(self, *, audio, context, language, return_time_stamps):
            assert isinstance(audio, tuple) and audio[1] == 16000
            chunk_index = audio[0][0]
            return [
                SimpleNamespace(
                    text=f"synthetic chunk {chunk_index}",
                    language=language or "Chinese",
                    time_stamps=[
                        {
                            "text": f"synthetic chunk {chunk_index}",
                            "start_time": local_timestamp[0],
                            "end_time": local_timestamp[1],
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
    monkeypatch.setattr(runner, "_media_duration_seconds", lambda _media: 90.0)
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
            raise RuntimeError(f"synthetic extraction failure for {index}")
        path = output_dir / f"chunk-{index:04d}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic-{index}".encode("ascii"))
        return {
            "index": index,
            "path": str(path),
            "bytes": path.stat().st_size,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "duration_seconds": end_seconds - start_seconds,
            "command": ["synthetic-ffmpeg", str(path)],
        }

    monkeypatch.setattr(runner, "extract_audio_chunk_window", fake_extract)
    # Runner now uses Qwen's waveform API. Keep model/decoder I/O synthetic;
    # extraction/resume/absolute-offset assertions below remain unchanged.
    monkeypatch.setattr(runner, "_read_pcm_chunk", lambda path: (
        [int(Path(path).stem.rsplit('-', 1)[-1])], 16000))


def test_middle_window_extraction_failure_preserves_successes_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted: list[int] = []
    _install_synthetic_runtime(
        monkeypatch,
        fail_extraction_indexes={1},
        extracted_indexes=extracted,
    )
    media = tmp_path / "synthetic-input.wav"
    output = tmp_path / "raw-asr-output.json"
    media.write_bytes(b"synthetic-media")

    result = runner.run_qwen3_asr(
        input_path=str(media),
        output_path=str(output),
        model="fixture/qwen3-asr",
        forced_aligner="",
        chunk_seconds=30,
    )

    assert extracted == [0, 1, 2]
    assert result["status"] == "degraded"
    assert result["usable"] is True
    assert result["successful_chunk_indexes"] == [0, 2]
    assert result["gaps"] == [
        {
            "chunk_index": 1,
            "start": 30.0,
            "end": 60.0,
            "reason": "chunk_extraction_failed",
        }
    ]
    checkpoint = json.loads(
        Path(result["checkpoint_path"]).read_text(encoding="utf-8")
    )
    assert checkpoint["successful_chunk_indexes"] == [0, 2]
    assert checkpoint["failed_chunks"][0]["reason"] == "chunk_extraction_failed"
    assert checkpoint["execution_contract"]["window_plan_revision"]


def test_resume_extracts_only_the_previously_failed_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "synthetic-input.wav"
    output = tmp_path / "raw-asr-output.json"
    media.write_bytes(b"synthetic-media")
    first_calls: list[int] = []
    _install_synthetic_runtime(
        monkeypatch,
        fail_extraction_indexes={1},
        extracted_indexes=first_calls,
    )
    first = runner.run_qwen3_asr(
        input_path=str(media),
        output_path=str(output),
        model="fixture/qwen3-asr",
        forced_aligner="",
        chunk_seconds=30,
    )
    assert first["successful_chunk_indexes"] == [0, 2]

    second_calls: list[int] = []
    _install_synthetic_runtime(monkeypatch, extracted_indexes=second_calls)
    second = runner.run_qwen3_asr(
        input_path=str(media),
        output_path=str(output),
        model="fixture/qwen3-asr",
        forced_aligner="",
        chunk_seconds=30,
    )

    assert second_calls == [1]
    assert second["resumed_from_checkpoint"] is True
    assert second["status"] == "completed"
    assert second["successful_chunk_indexes"] == [0, 1, 2]


def test_window_local_timestamps_receive_exact_start_offset_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted: list[int] = []
    _install_synthetic_runtime(
        monkeypatch,
        extracted_indexes=extracted,
        local_timestamp=(1.0, 2.0),
    )
    media = tmp_path / "synthetic-input.wav"
    output = tmp_path / "raw-asr-output.json"
    media.write_bytes(b"synthetic-media")

    result = runner.run_qwen3_asr(
        input_path=str(media),
        output_path=str(output),
        model="fixture/qwen3-asr",
        forced_aligner="fixture/aligner",
        chunk_seconds=30,
        chunk_indexes=[1],
    )

    assert extracted == [1]
    assert result["results"][0]["timestamps"] == [
        {"text": "synthetic chunk 1", "start": 31.0, "end": 32.0}
    ]
    assert result["segments"][0]["start"] == 31.0
    assert result["segments"][0]["end"] == 32.0
