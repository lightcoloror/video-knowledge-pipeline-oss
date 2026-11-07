from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_knowledge_pipeline.local_media_progress import LocalMediaProgress, render_progress_line
from video_knowledge_pipeline.qwen3_asr_python_runner import run_qwen3_asr
from video_knowledge_pipeline.smart_summary_input_pack import _transcript_segments
from video_knowledge_pipeline.transcript import parse_transcript
from video_knowledge_pipeline.transcript_postprocess import postprocess_asr_transcript


def _install_fake_qwen_runtime(monkeypatch: pytest.MonkeyPatch, *, fail_chunk: int | None) -> None:
    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    torch.cuda = SimpleNamespace(is_available=lambda: False)

    class Runtime:
        def transcribe(self, *, audio, context, language, return_time_stamps):
            chunk_index = int(Path(audio).stem.rsplit("-", 1)[-1])
            if chunk_index == fail_chunk:
                raise RuntimeError("fixture chunk failure")
            return [
                SimpleNamespace(
                    text=f"chunk {chunk_index}",
                    language=language or "Chinese",
                    time_stamps=[
                        {"text": f"chunk {chunk_index}", "start_time": 0.0, "end_time": 1.0}
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
        path = output_dir / f"chunk-{index:04d}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"chunk-{index}".encode("ascii"))
        return {
            "index": index,
            "path": str(path),
            "bytes": path.stat().st_size,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "duration_seconds": end_seconds - start_seconds,
            "command": ["synthetic-ffmpeg", str(path)],
        }

    monkeypatch.setattr(
        "video_knowledge_pipeline.qwen3_asr_python_runner.extract_audio_chunk_window",
        fake_extract,
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.qwen3_asr_python_runner._media_duration_seconds",
        lambda _media: 90.0,
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.qwen3_asr_python_runner._model_ready",
        lambda *, preset, model: {
            "ready": True,
            "status": "fixture_ready",
            "preset": preset,
            "cache_matches": [model],
        },
    )


def test_local_qwen_asr_multi_chunk_completed_and_progress_monotonic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qwen_runtime(monkeypatch, fail_chunk=None)
    media = tmp_path / "input.wav"
    output = tmp_path / "raw-asr-output.json"
    media.write_bytes(b"fixture")
    callback_events: list[dict] = []

    result = run_qwen3_asr(
        input_path=str(media),
        output_path=str(output),
        model="fixture/qwen3-asr",
        forced_aligner="",
        chunk_seconds=30,
        progress_callback=callback_events.append,
    )

    assert result["status"] == "completed"
    assert result["successful_chunk_count"] == 3
    assert result["failed_chunk_count"] == 0
    assert [row["chunk_index"] for row in result["results"]] == [0, 1, 2]
    assert [row["segment_id"] for row in result["segments"]] == [
        "chunk-0000-result-0001-0001",
        "chunk-0001-result-0001-0001",
        "chunk-0002-result-0001-0001",
    ]
    percents = [event["percent"] for event in callback_events]
    assert percents == sorted(percents)
    assert callback_events[-1]["status"] == "completed"
    assert json.loads(Path(result["progress"]["progress_json"]).read_text(encoding="utf-8"))["status"] == "completed"
    assert render_progress_line(callback_events[-1]).startswith("[COMPLETED]")



def test_local_qwen_asr_resumes_matching_checkpoint_without_repeating_completed_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qwen_runtime(monkeypatch, fail_chunk=None)
    media = tmp_path / "input.wav"
    output = tmp_path / "raw-asr-output.json"
    media.write_bytes(b"fixture")
    checkpoint = output.with_name(f"{output.stem}-checkpoint.json")
    checkpoint.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.qwen3_asr_checkpoint.v1",
                "status": "running",
                "input_identity": {"path": str(media.resolve()), "bytes": media.stat().st_size},
                "model": "fixture/qwen3-asr",
                "chunk_seconds": 30,
                "results": [
                    {
                        "chunk_index": 0,
                        "chunk_offset_seconds": 0.0,
                        "text": "checkpointed chunk 0",
                        "language": "Chinese",
                        "timestamps": [{"text": "checkpointed chunk 0", "start": 0.0, "end": 1.0}],
                        "segments": [{"segment_id": "chunk-0000-result-0001-0001", "source_segment_ids": ["chunk-0000-result-0001-0001"], "start": 0.0, "end": 1.0, "text": "checkpointed chunk 0"}],
                    }
                ],
                "failed_chunks": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_qwen3_asr(
        input_path=str(media),
        output_path=str(output),
        model="fixture/qwen3-asr",
        forced_aligner="",
        chunk_seconds=30,
    )

    assert result["status"] == "completed"
    assert result["resumed_from_checkpoint"] is True
    assert result["checkpointed_successful_chunk_count"] == 1
    assert result["successful_chunk_indexes"] == [0, 1, 2]
    assert [row["chunk_index"] for row in result["results"]] == [0, 1, 2]



def test_local_qwen_asr_single_chunk_failure_is_degraded_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qwen_runtime(monkeypatch, fail_chunk=1)
    media = tmp_path / "input.wav"
    output = tmp_path / "raw-asr-output.json"
    media.write_bytes(b"fixture")

    result = run_qwen3_asr(
        input_path=str(media),
        output_path=str(output),
        model="fixture/qwen3-asr",
        forced_aligner="",
        chunk_seconds=30,
    )

    assert result["ok"] is False
    assert result["usable"] is True
    assert result["status"] == "degraded"
    assert [row["chunk_index"] for row in result["results"]] == [0, 2]
    assert result["failed_chunk_count"] == 1
    failed = result["failed_chunks"][0]
    assert failed["chunk_index"] == 1
    assert failed["reason"] == "chunk_transcription_failed"
    assert Path(failed["artifact_path"]).exists()
    command = failed["retry_command"]["command"]
    chunk_option = command.index("--chunk-indexes")
    assert command[chunk_option : chunk_option + 2] == ["--chunk-indexes", "1"]
    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    assert report["status"] == "degraded"
    assert report["raw_successful_content_preserved"] is True
    progress_rows = [
        json.loads(line)
        for line in Path(result["progress"]["progress_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["percent"] for row in progress_rows] == sorted(row["percent"] for row in progress_rows)
    assert progress_rows[-1]["status"] == "degraded"



def test_local_qwen_asr_stops_retrying_a_chunk_after_configured_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qwen_runtime(monkeypatch, fail_chunk=1)
    media = tmp_path / "input.wav"
    output = tmp_path / "raw-asr-output.json"
    media.write_bytes(b"fixture")

    first = run_qwen3_asr(
        input_path=str(media),
        output_path=str(output),
        model="fixture/qwen3-asr",
        forced_aligner="",
        chunk_seconds=30,
        max_chunk_attempts=1,
    )
    second = run_qwen3_asr(
        input_path=str(media),
        output_path=str(output),
        model="fixture/qwen3-asr",
        forced_aligner="",
        chunk_seconds=30,
        max_chunk_attempts=1,
    )

    assert first["failed_chunks"][0]["retry_exhausted"] is True
    assert first["failed_chunks"][0]["attempt_count"] == 1
    assert second["resumed_from_checkpoint"] is True
    assert second["retry_exhausted_chunk_count"] == 1
    assert second["failed_chunks"][0]["attempt_count"] == 1



def test_transcript_postprocess_preserves_ids_order_timestamps_and_boundaries(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source = bundle / "normalized-transcript.json"
    source.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": "s-03", "start": 8.0, "end": 12.0, "text": "第三段"},
                    {"id": "s-07", "start": 12.5, "end": 16.0, "text": "第七段"},
                    {"id": "s-09", "start": 18.0, "end": 21.0, "text": "第九段"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "manifest.json").write_text(
        json.dumps({"normalized_transcript_json": source.name}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = postprocess_asr_transcript(bundle)
    payload = json.loads((bundle / "postprocessed-transcript.json").read_text(encoding="utf-8"))
    segments = payload["segments"]

    assert result["segment_policy"] == "preserve"
    assert [row["segment_id"] for row in segments] == ["s-03", "s-07", "s-09"]
    assert [(row["start"], row["end"]) for row in segments] == [(8.0, 12.0), (12.5, 16.0), (18.0, 21.0)]
    assert [row["source_segment_ids"] for row in segments] == [["s-03"], ["s-07"], ["s-09"]]
    assert result["source_segment_count"] == result["postprocessed_segment_count"] == 3
    progress = json.loads((bundle / "asr-transcript-postprocess-progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "completed"

    summary_segments = _transcript_segments(parse_transcript(bundle / "corrected-transcript.json"), [])
    assert [row["segment_id"] for row in summary_segments] == ["s-03", "s-07", "s-09"]
    assert [(row["start"], row["end"]) for row in summary_segments] == [(8.0, 12.0), (12.5, 16.0), (18.0, 21.0)]


def test_explicit_readable_merge_records_source_segments(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source = bundle / "normalized-transcript.json"
    source.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": "s-1", "start": 0, "end": 2, "text": "第一段"},
                    {"id": "s-2", "start": 2, "end": 4, "text": "第二段"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "manifest.json").write_text(
        json.dumps({"normalized_transcript_json": source.name}),
        encoding="utf-8",
    )

    result = postprocess_asr_transcript(bundle, segment_policy="readable_merge", target_seconds=30)

    assert result["postprocessed_segment_count"] == 1
    assert result["transformations"]
    assert result["transformations"][-1]["type"] == "explicit_merge"
    assert result["transformations"][-1]["source_segment_ids"] == ["s-1", "s-2"]


def test_progress_protocol_rejects_regression_and_post_terminal_events(tmp_path: Path) -> None:
    progress = LocalMediaProgress(
        pipeline="fixture",
        snapshot_path=tmp_path / "progress.json",
        events_path=tmp_path / "progress.jsonl",
    )
    progress.emit(stage="one", percent=25, message="one")
    with pytest.raises(ValueError, match="monotonic"):
        progress.emit(stage="two", percent=24, message="two")
    progress.emit(stage="done", percent=90, message="done", status="degraded")
    with pytest.raises(ValueError, match="terminal"):
        progress.emit(stage="late", percent=100, message="late", status="completed")


def test_progress_protocol_resets_jsonl_by_atomic_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events_path = tmp_path / "progress.jsonl"
    events_path.write_text("stale\n", encoding="utf-8")
    original_write_text = Path.write_text

    def reject_direct_target_write(path: Path, *args: object, **kwargs: object) -> int:
        if path == events_path:
            raise AssertionError("progress reset must not truncate the existing target directly")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", reject_direct_target_write)

    LocalMediaProgress(
        pipeline="fixture",
        snapshot_path=tmp_path / "progress.json",
        events_path=events_path,
    )

    assert events_path.read_text(encoding="utf-8") == ""
