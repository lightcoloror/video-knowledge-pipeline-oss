from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.asr_ab_compare import compare_asr_ab_sample
from video_knowledge_pipeline.asr_ab_plan import plan_asr_ab_sample
from video_knowledge_pipeline.asr_ab_run import run_asr_ab_sample


def _existing_sample(workspace: Path) -> Path:
    sample_dir = workspace / "transcripts" / "asr-ab-sample"
    sample_dir.mkdir(parents=True)
    sample = sample_dir / "dialogue.sample-0-300.wav"
    sample.write_bytes(b"local sample")
    return sample


def test_asr_ab_plan_registers_moss_through_existing_asr_front_door(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    sample = _existing_sample(workspace)

    plan = plan_asr_ab_sample(workspace, sample, write=False)

    moss = {row["key"]: row for row in plan["variants"]}[
        "moss_transcribe_diarize"
    ]
    assert moss["runner"] == "existing_asr_plan_front_door"
    assert moss["preset"] == "moss-transcribe-diarize"
    assert moss["upstream"] == {
        "project": "OpenMOSS/MOSS-Transcribe-Diarize",
        "commit": "eda4b9f13f1574765a80438c9797780a9bd48112",
        "entrypoint": "mtd-subtitle",
        "output_contract": "segments.json:start,end,text,speaker",
        "postprocess": False,
    }
    assert moss["operator_boundary"]["does_not_download_model"] is True
    assert moss["operator_boundary"]["does_not_fallback_to_another_asr"] is True
    assert "command" not in moss


def test_moss_command_discovery_includes_dedicated_reviewed_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from video_knowledge_pipeline import asr_runner

    scripts = tmp_path / "moss-runtime-py312" / "Scripts"
    scripts.mkdir(parents=True)
    command = scripts / "mtd-subtitle.exe"
    command.write_bytes(b"launcher")
    monkeypatch.delenv("LECTURE_MOSS_TRANSCRIBE_COMMAND", raising=False)
    monkeypatch.delenv("LECTURE_ASR_BIN_DIR", raising=False)
    monkeypatch.setattr(
        asr_runner,
        "_default_command_bin_dirs",
        lambda _root, _preset: [scripts],
    )
    monkeypatch.setattr(asr_runner.shutil, "which", lambda _command: None)

    resolved = asr_runner._resolve_command_path(
        {
            "command": "mtd-subtitle",
            "env_command": "LECTURE_MOSS_TRANSCRIBE_COMMAND",
        }
    )

    assert resolved == str(command.resolve())


def test_asr_environment_reuses_shared_command_resolution_and_runtime_probe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from video_knowledge_pipeline import asr_environment

    command = tmp_path / "mtd-subtitle.exe"
    command.write_bytes(b"launcher")
    monkeypatch.delenv("LECTURE_MOSS_TRANSCRIBE_COMMAND", raising=False)
    monkeypatch.setattr(
        asr_environment,
        "_resolve_command_path",
        lambda _preset: str(command),
    )
    monkeypatch.setattr(
        asr_environment,
        "_module_status",
        lambda *_args, **_kwargs: {
            "module": "moss_transcribe_diarize",
            "available": False,
            "returncode": 1,
        },
    )
    monkeypatch.setattr(
        asr_environment,
        "_command_runtime_probe",
        lambda **_kwargs: {
            "required": True,
            "ready": False,
            "status": "dependency_not_ready",
            "blocker": "missing_python_dependency:transformers",
            "exit_code": 1,
        },
    )

    status = asr_environment._tool_status(
        {
            "name": "moss-transcribe-diarize",
            "role": "speaker challenger",
            "module": "moss_transcribe_diarize",
            "command_name": "mtd-subtitle",
            "env_command": "LECTURE_MOSS_TRANSCRIBE_COMMAND",
            "install_switch": "",
            "install_command": "documented",
        },
        scripts_dir=tmp_path / "shared" / "Scripts",
        python_path=tmp_path / "python.exe",
        venv_exists=False,
    )

    assert status["command"] == str(command)
    assert status["command_exists"] is True
    assert status["configured_by_environment"] is False
    assert status["runtime_ready"] is False
    assert status["runtime_probe"]["blocker"] == (
        "missing_python_dependency:transformers"
    )


def test_asr_ab_moss_preview_reuses_plan_readiness_without_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from video_knowledge_pipeline import asr_ab_run

    workspace = tmp_path / "workspace"
    sample = _existing_sample(workspace)
    plan = plan_asr_ab_sample(workspace, sample, write=True)
    captured: dict[str, object] = {}

    def fake_plan(root, media, **kwargs):
        captured.update({"root": root, "media": media, **kwargs})
        return {
            "plan_path": str(tmp_path / "moss-plan.json"),
            "available": False,
            "availability": {
                "command_path": "",
                "runtime_probe": {
                    "ready": False,
                    "blocker": "command_not_found",
                },
            },
            "model_ready": {
                "ready": False,
                "status": "unknown_or_not_downloaded",
            },
        }

    monkeypatch.setattr(asr_ab_run, "plan_asr_run", fake_plan)
    monkeypatch.setattr(
        asr_ab_run,
        "run_asr_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preview must not execute MOSS")
        ),
    )

    result = run_asr_ab_sample(
        workspace,
        plan_json=plan["artifacts"]["json"],
        variants=["moss_transcribe_diarize"],
        execute_local=False,
        write=False,
    )

    row = result["variants"][0]
    assert row["status"] == "preview"
    assert row["runtime_ready"] is False
    assert row["model_ready"]["ready"] is False
    assert captured["preset"] == "moss-transcribe-diarize"
    assert captured["media"] == sample.resolve()


def test_asr_ab_moss_execute_blocks_without_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from video_knowledge_pipeline import asr_ab_run

    workspace = tmp_path / "workspace"
    sample = _existing_sample(workspace)
    plan = plan_asr_ab_sample(workspace, sample, write=True)
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        asr_ab_run,
        "plan_asr_run",
        lambda *_args, **_kwargs: {
            "plan_path": str(tmp_path / "moss-plan.json"),
            "available": False,
            "availability": {
                "command_path": "",
                "runtime_probe": {
                    "ready": False,
                    "blocker": "command_not_found",
                },
            },
            "model_ready": {
                "ready": False,
                "status": "unknown_or_not_downloaded",
            },
        },
    )

    def fake_run(plan_path, **kwargs):
        calls.append({"plan_path": plan_path, **kwargs})
        return {
            "status": "blocked",
            "stderr": "ASR runner is not marked available",
            "raw_output_json": "",
        }

    monkeypatch.setattr(asr_ab_run, "run_asr_plan", fake_run)

    result = run_asr_ab_sample(
        workspace,
        plan_json=plan["artifacts"]["json"],
        variants=["moss_transcribe_diarize"],
        execute_local=True,
        write=False,
    )

    row = result["variants"][0]
    assert row["status"] == "blocked"
    assert row["provider"] == "moss-transcribe-diarize"
    assert row["raw_output_json"] == ""
    assert row["operator_boundary"]["does_not_fallback_to_another_asr"] is True
    assert calls == [
        {
            "plan_path": str(tmp_path / "moss-plan.json"),
            "execute": True,
            "timeout_seconds": 1800,
        }
    ]


def test_multiple_speaker_candidates_require_metric_selection(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 4.0, "text": "你好", "speaker": "S01"},
                    {"start": 4.0, "end": 8.0, "text": "请讲", "speaker": "S02"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    run = tmp_path / "run.json"
    common_metrics = {
        "segment_count": 2,
        "char_count": 4,
        "punctuation_count": 1,
        "duration_seconds": 8.0,
        "speaker_count": 2,
        "speaker_labeled_segment_count": 2,
    }
    run.write_text(
        json.dumps(
            {
                "workspace_dir": str(tmp_path),
                "sample_media_path": str(tmp_path / "sample.wav"),
                "variants": [
                    {
                        "key": "sensevoice_full_punc",
                        "status": "ok",
                        "metrics": {
                            **common_metrics,
                            "speaker_count": 0,
                            "speaker_labeled_segment_count": 0,
                        },
                    },
                    {
                        "key": "sensevoice_full_punc_campp",
                        "status": "ok",
                        "metrics": common_metrics,
                    },
                    {
                        "key": "moss_transcribe_diarize",
                        "status": "ok",
                        "metrics": common_metrics,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = compare_asr_ab_sample(
        run,
        reference_transcript=reference,
        start_seconds=0.0,
        end_seconds=8.0,
        write=False,
    )

    assert result["production_recommendation"] == (
        "blocked_until_speaker_quality_evaluation_passes"
    )
    assert result["speaker_evaluation_candidate"] == ""
    assert result["speaker_evaluation_candidates"] == [
        "sensevoice_full_punc_campp",
        "moss_transcribe_diarize",
    ]
    assert result["speaker_candidate_selection"] == (
        "multiple_candidates_require_der_cpcer_tcpcer_comparison"
    )
    assert result["gates"]["moss_transcribe_diarize_ready"] is True
    assert result["gates"]["can_decide_second_asr"] is True
