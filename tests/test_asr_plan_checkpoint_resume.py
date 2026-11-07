from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from video_knowledge_pipeline import asr_execution
from video_knowledge_pipeline.asr_execution import run_asr_plan
from video_knowledge_pipeline.qwen3_asr_python_runner import (
    _load_checkpoint,
    qwen_checkpoint_execution_contract,
)


def _row(index: int, text: str | None = None) -> dict:
    value = text or f"chunk {index}"
    segment_id = f"chunk-{index:04d}-result-0001-0001"
    return {
        "chunk_index": index,
        "chunk_offset_seconds": float(index * 30),
        "text": value,
        "language": "Chinese",
        "timestamps": [
            {
                "text": value,
                "start": float(index * 30),
                "end": float(index * 30 + 1),
            }
        ],
        "segments": [
            {
                "segment_id": segment_id,
                "source_segment_ids": [segment_id],
                "start": float(index * 30),
                "end": float(index * 30 + 1),
                "text": value,
            }
        ],
    }


def _fixture_plan(tmp_path: Path) -> tuple[Path, Path, Path, list[str]]:
    project = tmp_path / "workspace"
    output_dir = project / "transcripts" / "asr-run-fixture"
    output_dir.mkdir(parents=True)
    media = tmp_path / "input.wav"
    media.write_bytes(b"synthetic-audio-fixture")
    output = output_dir / "raw-asr-output.json"
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
        "--language",
        "Chinese",
        "--device",
        "cpu",
        "--chunk-seconds",
        "30",
        "--no-timestamps",
    ]
    plan = {
        "project": str(project),
        "preset": "qwen3-asr-0.6b",
        "provider": "qwen3-asr",
        "media_path": str(media.resolve()),
        "output_dir": str(output_dir),
        "expected_output_json": str(output),
        "command": command,
        "runner": "qwen3_asr_python",
        "asr_mode": "full",
        "available": True,
        "model_ready": {"ready": True},
        "pythonpath": "",
    }
    plan_path = output_dir / "asr-run-plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    return plan_path, media, output, command


def _checkpoint(
    output: Path,
    media: Path,
    *,
    indexes: list[int],
    requested: int = 2,
    status: str = "running",
    execution_contract: dict | None = None,
) -> Path:
    path = output.with_name(f"{output.stem}-checkpoint.json")
    payload = {
        "schema": "video_knowledge_pipeline.qwen3_asr_checkpoint.v1",
        "status": status,
        "input_identity": {
            "path": str(media.resolve()),
            "bytes": media.stat().st_size,
        },
        "model": "fixture/qwen3-asr",
        "forced_aligner": "",
        "language": "Chinese",
        "chunk_seconds": 30,
        "requested_chunk_count": requested,
        "successful_chunk_indexes": indexes,
        "successful_chunk_count": len(indexes),
        "failed_chunk_count": 0,
        "results": [_row(index) for index in indexes],
        "failed_chunks": [],
    }
    if execution_contract is not None:
        payload["execution_contract"] = execution_contract
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _completed_payload(media: Path, indexes: list[int]) -> dict:
    rows = [_row(index) for index in indexes]
    return {
        "schema": "video_knowledge_pipeline.qwen3_asr_raw_output.v1",
        "provider": "qwen3-asr",
        "model": "fixture/qwen3-asr",
        "forced_aligner": "",
        "chunk_seconds": 30,
        "chunk_count": len(indexes),
        "successful_chunk_count": len(indexes),
        "failed_chunk_count": 0,
        "successful_chunk_indexes": indexes,
        "input_path": str(media.resolve()),
        "ok": True,
        "usable": True,
        "status": "completed",
        "results": rows,
        "segments": [segment for row in rows for segment in row["segments"]],
        "text": "\n".join(row["text"] for row in rows),
        "failed_chunks": [],
        "gaps": [],
        "retry_commands": [],
    }


def test_run_asr_plan_restores_complete_checkpoint_and_rerun_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, media, output, _command = _fixture_plan(tmp_path)
    checkpoint = _checkpoint(output, media, indexes=[0, 1])

    def fail_if_executed(*_args, **_kwargs):
        raise AssertionError("completed checkpoint must not launch the ASR child")

    monkeypatch.setattr(asr_execution, "_run_command_with_cuda_oom_recovery", fail_if_executed)

    first = run_asr_plan(plan_path, execute=True, normalize=False)
    first_bytes = output.read_bytes()
    second = run_asr_plan(plan_path, execute=True, normalize=False)

    assert first["status"] == "ok"
    assert first["execution_skipped_reason"] == "checkpoint_complete"
    assert first["checkpoint_recovery"] == "raw_output_rebuilt"
    assert first["resumed_from_checkpoint"] is True
    assert first["checkpoint_successful_chunk_count"] == 2
    assert first["raw_output_json"] == str(output.resolve())
    assert second["execution_skipped_reason"] == "output_already_complete"
    assert second["checkpoint_recovery"] == "raw_output_reused"
    assert output.read_bytes() == first_bytes
    assert json.loads(output.read_text(encoding="utf-8"))["checkpoint_path"] == str(checkpoint)


def test_run_asr_plan_restores_complete_window_revision_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, media, output, _command = _fixture_plan(tmp_path)
    contract = qwen_checkpoint_execution_contract(
        media=media,
        model="fixture/qwen3-asr",
        forced_aligner="",
        language="Chinese",
        context="",
        chunk_seconds=30,
        max_new_tokens=1024,
        dtype_name="auto",
        chunk_indexes=[],
        window_plan_revision="d" * 64,
    )
    _checkpoint(
        output,
        media,
        indexes=[0, 1],
        status="completed",
        execution_contract=contract,
    )

    def fail_if_executed(*_args, **_kwargs):
        raise AssertionError("complete window checkpoint must not launch ASR")

    monkeypatch.setattr(
        asr_execution,
        "_run_command_with_cuda_oom_recovery",
        fail_if_executed,
    )

    result = run_asr_plan(plan_path, execute=True, normalize=False)

    assert result["execution_skipped_reason"] == "checkpoint_complete"
    assert result["checkpoint_window_plan_revision"] == "d" * 64
    assert result["checkpoint_recovery"] == "raw_output_rebuilt"


def test_run_asr_plan_surfaces_partial_checkpoint_before_child_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, media, output, _command = _fixture_plan(tmp_path)
    _checkpoint(output, media, indexes=[0])
    captured: dict[str, list[str]] = {}

    def fake_run(command, **_kwargs):
        captured["command"] = list(command)
        output.write_text(
            json.dumps(_completed_payload(media, [0, 1]), ensure_ascii=False),
            encoding="utf-8",
        )
        completed = subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return completed, [{"stage": "initial", "command": list(command), "returncode": 0}]

    monkeypatch.setattr(asr_execution, "_run_command_with_cuda_oom_recovery", fake_run)

    result = run_asr_plan(plan_path, execute=True, normalize=False)

    assert result["status"] == "ok"
    assert result["resumed_from_checkpoint"] is True
    assert result["checkpoint_status"] == "running"
    assert result["checkpoint_successful_chunk_indexes"] == [0]
    assert "--no-resume" not in captured["command"]


def test_run_asr_plan_no_resume_forces_qwen_child_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, media, output, _command = _fixture_plan(tmp_path)
    _checkpoint(output, media, indexes=[0, 1], status="completed")
    captured: dict[str, list[str]] = {}

    def fake_run(command, **_kwargs):
        captured["command"] = list(command)
        output.write_text(
            json.dumps(_completed_payload(media, [0, 1]), ensure_ascii=False),
            encoding="utf-8",
        )
        completed = subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return completed, [{"stage": "initial", "command": list(command), "returncode": 0}]

    monkeypatch.setattr(asr_execution, "_run_command_with_cuda_oom_recovery", fake_run)

    result = run_asr_plan(plan_path, execute=True, normalize=False, resume=False)

    assert result["status"] == "ok"
    assert result["resume"] is False
    assert result["resumed_from_checkpoint"] is False
    assert captured["command"].count("--no-resume") == 1


def test_qwen_checkpoint_contract_rejects_context_drift(tmp_path: Path) -> None:
    media = tmp_path / "input.wav"
    media.write_bytes(b"fixture")
    checkpoint = tmp_path / "checkpoint.json"
    identity = {
        "path": str(media.resolve()),
        "bytes": media.stat().st_size,
        "mtime_ns": media.stat().st_mtime_ns,
    }
    _checkpoint(
        tmp_path / "raw-asr-output.json",
        media,
        indexes=[0],
        execution_contract={
            "input_identity": identity,
            "model": "fixture/qwen3-asr",
            "forced_aligner": "",
            "language": "Chinese",
            "context_sha256": hashlib.sha256(b"old reviewed terms").hexdigest(),
            "chunk_seconds": 30,
            "max_new_tokens": 1024,
            "dtype": "auto",
            "chunk_indexes": [],
        },
    ).replace(checkpoint)

    loaded = _load_checkpoint(
        checkpoint,
        media=media.resolve(),
        model="fixture/qwen3-asr",
        chunk_seconds=30,
        forced_aligner="",
        language="Chinese",
        context="new reviewed terms",
        max_new_tokens=1024,
        dtype_name="auto",
        chunk_indexes=[],
        resume=True,
    )

    assert loaded["resumed"] is False
    assert loaded["results"] == []


def test_qwen_checkpoint_contract_rejects_window_plan_revision_drift(
    tmp_path: Path,
) -> None:
    media = tmp_path / "input.wav"
    media.write_bytes(b"fixture")
    contract = qwen_checkpoint_execution_contract(
        media=media,
        model="fixture/qwen3-asr",
        forced_aligner="",
        language="Chinese",
        context="",
        chunk_seconds=30,
        max_new_tokens=1024,
        dtype_name="auto",
        chunk_indexes=[],
        window_plan_revision="a" * 64,
    )
    checkpoint = _checkpoint(
        tmp_path / "raw-asr-output.json",
        media,
        indexes=[0],
        execution_contract=contract,
    )

    loaded = _load_checkpoint(
        checkpoint,
        media=media.resolve(),
        model="fixture/qwen3-asr",
        chunk_seconds=30,
        forced_aligner="",
        language="Chinese",
        context="",
        max_new_tokens=1024,
        dtype_name="auto",
        chunk_indexes=[],
        window_plan_revision="b" * 64,
        resume=True,
    )

    assert loaded["resumed"] is False
    assert loaded["results"] == []


def test_qwen_legacy_checkpoint_remains_resumable(tmp_path: Path) -> None:
    media = tmp_path / "input.wav"
    media.write_bytes(b"fixture")
    checkpoint = _checkpoint(
        tmp_path / "raw-asr-output.json",
        media,
        indexes=[0],
    )

    loaded = _load_checkpoint(
        checkpoint,
        media=media.resolve(),
        model="fixture/qwen3-asr",
        chunk_seconds=30,
        forced_aligner="",
        language="Chinese",
        context="",
        max_new_tokens=1024,
        dtype_name="auto",
        chunk_indexes=[],
        resume=True,
    )

    assert loaded["resumed"] is True
    assert loaded["successful_chunk_indexes"] == [0]
