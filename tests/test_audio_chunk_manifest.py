from __future__ import annotations

import subprocess

import pytest
from pathlib import Path

import video_knowledge_pipeline.audio_chunk_manifest as chunking
import video_knowledge_pipeline.funasr_chunked_runner as runner


def test_fixed_chunk_boundaries_keep_exact_final_window() -> None:
    assert chunking.fixed_chunk_boundaries(65, 30) == [
        (0.0, 30.0),
        (30.0, 60.0),
        (60.0, 65.0),
    ]


def test_single_window_extraction_failure_preserves_sibling_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "synthetic-input.wav"
    media.write_bytes(b"synthetic-media")
    output = tmp_path / "chunks"
    output.mkdir()
    sibling = output / "chunk-0000.wav"
    sibling.write_bytes(b"completed-window")

    monkeypatch.setattr(
        chunking.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 1, "", "synthetic decode failure"
        ),
    )

    with pytest.raises(RuntimeError, match="chunk_index=1; returncode=1"):
        chunking.extract_audio_chunk_window(
            media,
            output,
            index=1,
            start_seconds=30,
            end_seconds=60,
            ffmpeg_path=tmp_path / "ffmpeg.exe",
        )

    assert sibling.read_bytes() == b"completed-window"


def test_subtitle_edit_boundaries_fall_back_to_even_split() -> None:
    assert chunking.compute_silence_adjusted_boundaries(120, 3, []) == [
        (0.0, 40.0),
        (40.0, 80.0),
        (80.0, 120.0),
    ]


def test_subtitle_edit_boundaries_choose_nearest_unused_silence() -> None:
    boundaries = chunking.compute_silence_adjusted_boundaries(
        120,
        3,
        [(59, 61)],
        max_offset_seconds=25,
    )

    assert boundaries == [(0.0, 60.0), (60.0, 80.0), (80.0, 120.0)]


def test_subtitle_edit_boundary_window_is_inclusive_and_strictly_increasing() -> None:
    boundaries = chunking.compute_silence_adjusted_boundaries(
        120,
        2,
        [(69, 71)],
    )

    assert boundaries == [(0.0, 70.0), (70.0, 120.0)]
    assert all(end > start for start, end in boundaries)


def test_subtitle_edit_parser_drops_unmatched_trailing_silence() -> None:
    parsed = chunking.parse_subtitle_edit_silence_intervals(
        "[silencedetect] silence_start: 5.0\n"
        "[silencedetect] silence_end: 6.0 | silence_duration: 1.0\n"
        "[silencedetect] silence_start: 100.0\n"
    )

    assert parsed == [(5.0, 6.0)]


def test_fixed_manifest_records_exact_last_chunk_and_zero_overlap(
    tmp_path: Path,
) -> None:
    media = tmp_path / "lesson.wav"
    media.write_bytes(b"media")
    chunks = [tmp_path / f"chunk-{index:04d}.wav" for index in range(2)]
    for chunk in chunks:
        chunk.write_bytes(b"chunk")

    manifest = chunking.build_fixed_chunk_manifest(
        media,
        chunks,
        target_chunk_seconds=300,
        media_duration_seconds=550,
    )

    assert manifest["chunks"][1]["start_seconds"] == 300
    assert manifest["chunks"][1]["end_seconds"] == 550
    assert manifest["chunks"][1]["overlap_before_seconds"] == 0
    assert manifest["source"]["sha256"] == chunking.sha256_file(media)
    assert manifest["chunks"][0]["bytes"] == chunks[0].stat().st_size
    assert manifest["chunks"][0]["sha256"] == chunking.sha256_file(chunks[0])
    assert manifest["revision"] == chunking.compute_audio_chunk_manifest_revision(
        manifest
    )
    assert len(manifest["revision"]) == 64


def test_silence_snapped_split_uses_exact_extraction_windows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    media = tmp_path / "lesson.wav"
    media.write_bytes(b"media")
    output = tmp_path / "chunks"
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        command = list(command)
        calls.append(command)
        if "silencedetect=noise=-30dB:d=0.5" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                "",
                "silence_start: 39\nsilence_end: 41\n",
            )
        Path(command[-1]).write_bytes(b"chunk")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(chunking.subprocess, "run", fake_run)
    manifest = chunking.prepare_silence_snapped_chunks(
        media,
        output,
        target_chunk_seconds=40,
        media_duration_seconds=120,
        ffmpeg_path=tmp_path / "ffmpeg.exe",
    )

    assert manifest["strategy"]["mode"] == "silence_snap"
    assert manifest["chunks"][0]["end_seconds"] == 40
    assert manifest["chunks"][1]["start_seconds"] == 40
    assert len(calls) == 4
    second_extraction = calls[2]
    assert second_extraction[second_extraction.index("-ss") + 1] == "40.000000"
    assert second_extraction[second_extraction.index("-t") + 1] == "40.000000"
    assert manifest["strategy"]["boundary_semantics"] == (
        "requested_source_media_window"
    )


def _runner_manifest(
    media: Path,
    chunks: list[Path],
    *,
    revision: str,
) -> dict:
    return {
        "schema": chunking.SCHEMA,
        "source": {
            "path": str(media.resolve()),
            "bytes": media.stat().st_size,
            "mtime_ns": media.stat().st_mtime_ns,
            "duration_seconds": 600.0,
        },
        "strategy": {"mode": "silence_snap", "target_chunk_seconds": 300.0},
        "silence_detection": {"status": "completed", "interval_count": 1},
        "chunks": [
            {
                "index": 0,
                "artifact_path": str(chunks[0]),
                "start_seconds": 0.0,
                "end_seconds": 280.0,
                "duration_seconds": 280.0,
                "overlap_before_seconds": 0.0,
                "overlap_after_seconds": 0.0,
                "boundary_source": "subtitle_edit_nearest_silence",
            },
            {
                "index": 1,
                "artifact_path": str(chunks[1]),
                "start_seconds": 280.0,
                "end_seconds": 600.0,
                "duration_seconds": 320.0,
                "overlap_before_seconds": 0.0,
                "overlap_after_seconds": 0.0,
                "boundary_source": "subtitle_edit_nearest_silence",
            },
        ],
        "revision": revision,
    }


def test_funasr_runner_uses_exact_silence_manifest_offsets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    media = tmp_path / "lesson.wav"
    media.write_bytes(b"media")
    output = tmp_path / "raw-asr-output.json"
    chunks = [tmp_path / f"chunk-{index:04d}.wav" for index in range(2)]
    for path in chunks:
        path.write_bytes(b"chunk")
    manifest = _runner_manifest(media, chunks, revision="a" * 64)
    monkeypatch.setattr(
        runner,
        "prepare_silence_snapped_chunks",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(runner, "_media_duration_seconds", lambda _path: 600.0)

    def fake_child(command, **_kwargs):
        command = list(command)
        child_output = Path(command[command.index("--output") + 1])
        child_output.write_text(
            '{"result":[{"text":"正文","sentence_info":'
            '[{"text":"正文","start":0,"end":1000}]}]}',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_child)
    result = runner.run_funasr_chunked(
        input_path=str(media),
        output_path=str(output),
        provider="sensevoice",
        model="model",
        chunk_boundary_mode="silence_snap",
    )

    assert result["status"] == "completed"
    assert result["chunk_manifest_revision"] == "a" * 64
    assert result["result"][1]["sentence_info"][0]["start"] == 280000.0
    assert result["result"][1]["chunk_end_seconds"] == 600.0
    checkpoint = Path(result["checkpoint_path"]).read_text(encoding="utf-8")
    assert '"chunk_manifest_revision": "' + ("a" * 64) + '"' in checkpoint


def test_funasr_retry_preserves_silence_boundary_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    media = tmp_path / "lesson.wav"
    media.write_bytes(b"media")
    output = tmp_path / "raw-asr-output.json"
    chunks = [tmp_path / f"chunk-{index:04d}.wav" for index in range(2)]
    for path in chunks:
        path.write_bytes(b"chunk")
    manifest = _runner_manifest(media, chunks, revision="b" * 64)
    monkeypatch.setattr(
        runner,
        "prepare_silence_snapped_chunks",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(runner, "_media_duration_seconds", lambda _path: 600.0)

    def fake_child(command, **_kwargs):
        command = list(command)
        source = Path(command[command.index("--input") + 1])
        if source.name == "chunk-0001.wav":
            return subprocess.CompletedProcess(command, 1, "", "failed")
        child_output = Path(command[command.index("--output") + 1])
        child_output.write_text(
            '{"result":[{"text":"正文"}]}',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_child)
    result = runner.run_funasr_chunked(
        input_path=str(media),
        output_path=str(output),
        provider="sensevoice",
        model="model",
        chunk_boundary_mode="silence_snap",
    )

    assert result["status"] == "degraded"
    assert result["gaps"] == [
        {
            "chunk_index": 1,
            "start": 280.0,
            "end": 600.0,
            "reason": "chunk_asr_failed",
        }
    ]
    retry = result["retry_commands"][0]["command"]
    assert retry[retry.index("--chunk-boundary-mode") + 1] == "silence_snap"
    assert retry[retry.index("--chunk-indexes") + 1] == "1"
def test_silence_snapped_extraction_failure_names_exact_chunk(
    tmp_path: Path,
    monkeypatch,
) -> None:
    media = tmp_path / "lesson.wav"
    media.write_bytes(b"media")
    output = tmp_path / "chunks"
    extraction_count = 0

    def fake_run(command, **_kwargs):
        nonlocal extraction_count
        command = list(command)
        if "silencedetect=noise=-30dB:d=0.5" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        extraction_count += 1
        if extraction_count == 2:
            return subprocess.CompletedProcess(command, 1, "", "decode failed")
        Path(command[-1]).write_bytes(b"chunk")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(chunking.subprocess, "run", fake_run)
    with pytest.raises(
        RuntimeError,
        match="chunk_index=1; returncode=1",
    ):
        chunking.prepare_silence_snapped_chunks(
            media,
            output,
            target_chunk_seconds=40,
            media_duration_seconds=120,
            ffmpeg_path=tmp_path / "ffmpeg.exe",
        )


def test_silence_checkpoint_rejects_changed_manifest_revision(
    tmp_path: Path,
) -> None:
    media = tmp_path / "lesson.wav"
    media.write_bytes(b"media")
    checkpoint = tmp_path / "checkpoint.json"
    runner._write_checkpoint(
        checkpoint,
        media=media,
        model="model",
        chunk_seconds=300,
        chunk_manifest_revision="a" * 64,
        execution_contract_revision="c" * 64,
        results=[{"chunk_index": 0, "text": "old"}],
        failed_chunks=[],
    )

    loaded = runner._load_checkpoint(
        checkpoint,
        media=media,
        model="model",
        chunk_seconds=300,
        chunk_manifest_revision="b" * 64,
        execution_contract_revision="c" * 64,
        allow_legacy_fixed=False,
    )

    assert loaded == {"resumed": False, "results": [], "failed_chunks": []}

def test_fixed_overlap_manifest_preserves_core_ownership(
    tmp_path: Path,
    monkeypatch,
) -> None:
    media = tmp_path / "lesson.wav"
    media.write_bytes(b"media")
    output = tmp_path / "chunks"
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        command = list(command)
        calls.append(command)
        Path(command[-1]).write_bytes(b"chunk")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(chunking.subprocess, "run", fake_run)
    manifest = chunking.prepare_fixed_overlap_chunks(
        media,
        output,
        target_chunk_seconds=40,
        media_duration_seconds=120,
        overlap_seconds=5,
        ffmpeg_path=tmp_path / "ffmpeg.exe",
    )

    assert len(calls) == 3
    assert manifest["strategy"]["mode"] == "fixed_overlap"
    assert manifest["strategy"]["overlap_seconds"] == 5
    assert manifest["chunks"][0]["start_seconds"] == 0
    assert manifest["chunks"][0]["end_seconds"] == 45
    assert manifest["chunks"][0]["core_start_seconds"] == 0
    assert manifest["chunks"][0]["core_end_seconds"] == 40
    assert manifest["chunks"][0]["overlap_after_seconds"] == 5
    assert manifest["chunks"][1]["start_seconds"] == 35
    assert manifest["chunks"][1]["end_seconds"] == 85
    assert manifest["chunks"][1]["overlap_before_seconds"] == 5
    assert calls[1][calls[1].index("-ss") + 1] == "35.000000"
    assert calls[1][calls[1].index("-t") + 1] == "50.000000"


def test_silence_snapped_overlap_reuses_same_extraction_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    media = tmp_path / "lesson.wav"
    media.write_bytes(b"media")
    output = tmp_path / "chunks"
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        command = list(command)
        calls.append(command)
        if "silencedetect=noise=-30dB:d=0.5" in command:
            return subprocess.CompletedProcess(
                command, 0, "", "silence_start: 39\nsilence_end: 41\n"
            )
        Path(command[-1]).write_bytes(b"chunk")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(chunking.subprocess, "run", fake_run)
    manifest = chunking.prepare_silence_snapped_chunks(
        media,
        output,
        target_chunk_seconds=40,
        media_duration_seconds=120,
        overlap_seconds=5,
        ffmpeg_path=tmp_path / "ffmpeg.exe",
    )

    assert manifest["chunks"][1]["start_seconds"] == 35
    assert manifest["chunks"][1]["core_start_seconds"] == 40
    assert manifest["chunks"][1]["core_end_seconds"] == 80
    assert manifest["chunks"][1]["end_seconds"] == 85
    assert manifest["strategy"]["overlap_seconds"] == 5
