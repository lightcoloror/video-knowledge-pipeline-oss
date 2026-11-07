from __future__ import annotations

import json
import subprocess
from pathlib import Path

import video_knowledge_pipeline.funasr_chunked_runner as runner


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _fixture(tmp_path: Path, count: int = 3) -> tuple[Path, list[Path], Path]:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"video")
    chunks = [tmp_path / f"chunk-{index:04d}.wav" for index in range(count)]
    for chunk in chunks:
        chunk.write_bytes(b"audio")
    return media, chunks, tmp_path / "raw-asr-output.json"


def test_timestamps_without_transcript_text_remain_unverified_empty() -> None:
    assert runner._records_have_content([{"timestamp": [[0, 1000]]}]) is False
    assert (
        runner._records_have_content(
            [{"sentence_info": [{"text": "", "start": 0, "end": 1000}]}]
        )
        is False
    )
    assert (
        runner._records_have_content(
            [{"sentence_info": [{"text": "spoken", "start": 0, "end": 1000}]}]
        )
        is True
    )


def test_failed_child_cannot_reuse_stale_json(
    tmp_path: Path, monkeypatch
) -> None:
    media, chunks, output = _fixture(tmp_path, count=1)
    monkeypatch.setattr(runner, "_audio_chunks", lambda *_args, **_kwargs: chunks)
    stale_dir = output.with_name(f"{output.stem}-chunks")
    stale_dir.mkdir()
    stale = stale_dir / "chunk-0000.json"
    stale.write_text(
        json.dumps({"result": [{"text": "stale transcript"}]}),
        encoding="utf-8",
    )

    def fail(command, **_kwargs):
        return subprocess.CompletedProcess(list(command), 1, "", "child failed")

    monkeypatch.setattr(runner.subprocess, "run", fail)
    result = runner.run_funasr_chunked(
        input_path=str(media),
        output_path=str(output),
        provider="sensevoice",
        model="model",
    )

    assert result["status"] == "failed"
    assert result["result"] == []
    assert stale.exists() is False


def test_selected_repair_success_is_separate_from_other_unresolved_chunks(
    tmp_path: Path, monkeypatch
) -> None:
    media, chunks, output = _fixture(tmp_path)
    monkeypatch.setattr(runner, "_audio_chunks", lambda *_args, **_kwargs: chunks)

    def initial(command, **_kwargs):
        command = list(command)
        source = Path(_option(command, "--input"))
        if source.name != "chunk-0000.wav":
            return subprocess.CompletedProcess(command, 1, "", "failed")
        Path(_option(command, "--output")).write_text(
            json.dumps({"result": [{"text": "chunk zero"}]}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", initial)
    first = runner.run_funasr_chunked(
        input_path=str(media),
        output_path=str(output),
        provider="sensevoice",
        model="model",
    )
    assert first["status"] == "degraded"
    assert first["unresolved_chunk_indexes"] == [1, 2]

    def repair(command, **_kwargs):
        command = list(command)
        assert Path(_option(command, "--input")).name == "chunk-0001.wav"
        Path(_option(command, "--output")).write_text(
            json.dumps({"result": [{"text": "repaired one"}]}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", repair)
    second = runner.run_funasr_chunked(
        input_path=str(media),
        output_path=str(output),
        provider="sensevoice",
        model="model",
        chunk_indexes=[1],
    )

    assert second["status"] == "partial_targeted_completed"
    assert second["ok"] is True
    assert second["canonical_complete"] is False
    assert second["quality_status"] == "degraded"
    assert second["unresolved_chunk_indexes"] == [2]
    snapshot = json.loads(
        Path(second["progress"]["progress_json"]).read_text(encoding="utf-8")
    )
    assert snapshot["status"] == "completed"
    assert "targeted subset completed" in snapshot["message"]
