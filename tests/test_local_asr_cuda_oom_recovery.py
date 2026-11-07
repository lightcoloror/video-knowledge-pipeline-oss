from __future__ import annotations

import json
import subprocess
from pathlib import Path

import video_knowledge_pipeline.asr_execution as asr_execution
import video_knowledge_pipeline.asr_runner as asr_runner


def _command_option(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _sensevoice_plan(tmp_path: Path) -> tuple[Path, Path]:
    output_dir = tmp_path / "asr-run"
    output_dir.mkdir()
    raw = output_dir / "raw-asr-output.json"
    command = [
        "python",
        "-m",
        "video_knowledge_pipeline.funasr_python_runner",
        "--device",
        "cuda",
        "--batch-size-s",
        "60",
        "--vad-max-single-segment-time-ms",
        "30000",
    ]
    plan = {
        "project": str(tmp_path),
        "preset": "sensevoice",
        "provider": "sensevoice",
        "runner": "funasr_python",
        "available": True,
        "local_asr_device": "cuda",
        "media_path": str(tmp_path / "lesson.mp4"),
        "output_dir": str(output_dir),
        "expected_output_json": str(raw),
        "command": command,
    }
    plan_path = tmp_path / "asr-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path, raw


def test_sensevoice_cuda_oom_retries_tuned_gpu_then_cpu(tmp_path: Path, monkeypatch) -> None:
    plan_path, raw = _sensevoice_plan(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        command = list(command)
        calls.append(command)
        device = _command_option(command, "--device")
        if device == "cpu":
            raw.write_text(json.dumps({"result": [{"text": "本地转写"}]}), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 1, "", "CUDA out of memory")

    monkeypatch.setattr(asr_execution, "_run_command", fake_run)
    result = asr_execution.run_asr_plan(plan_path, execute=True, normalize=False)

    assert result["status"] == "ok"
    assert [attempt["stage"] for attempt in result["execution_attempts"]] == [
        "initial",
        "gpu_oom_tuned_retry",
        "cpu_after_second_cuda_oom",
    ]
    assert [_command_option(command, "--device") for command in calls] == ["cuda", "cuda", "cpu"]
    assert [_command_option(command, "--batch-size-s") for command in calls] == ["60", "10", "10"]
    assert [_command_option(command, "--vad-max-single-segment-time-ms") for command in calls] == ["30000", "15000", "15000"]
    report = Path(result["asr_log"]["report_path"]).read_text(encoding="utf-8")
    assert "gpu_oom_tuned_retry" in report


def test_non_oom_failure_does_not_change_device(tmp_path: Path, monkeypatch) -> None:
    plan_path, _raw = _sensevoice_plan(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 1, "", "model cache missing")

    monkeypatch.setattr(asr_execution, "_run_command", fake_run)
    result = asr_execution.run_asr_plan(plan_path, execute=True, normalize=False)

    assert result["status"] == "failed"
    assert len(calls) == 1
    assert result["execution_attempts"][0]["stage"] == "initial"


def test_sensevoice_plan_declares_oom_recovery_settings(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"video")
    monkeypatch.setattr(asr_runner, "asr_runtime_profile", lambda: {"device": "cuda"})
    monkeypatch.setattr(asr_runner, "_module_available_in_python", lambda *_args: True)
    monkeypatch.setattr(asr_runner, "_model_ready", lambda **_kwargs: {"ready": True})

    plan = asr_runner.plan_asr_run(tmp_path / "workspace", media, preset="sensevoice")

    assert _command_option(plan["command"], "--batch-size-s") == "60"
    assert plan["oom_recovery"]["gpu_retry"] == {"batch_size_s": 10, "vad_max_single_segment_time_ms": 15000}
    assert plan["oom_recovery"]["cpu_retry_after"] == "second_cuda_out_of_memory_only"
