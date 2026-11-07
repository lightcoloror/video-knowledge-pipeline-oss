from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.acceptance_check import acceptance_check
from video_knowledge_pipeline.acceptance_run import run_acceptance_bundle, run_acceptance_run
from video_knowledge_pipeline.asr_adapter import normalize_asr_output
from video_knowledge_pipeline.asr_environment import asr_environment_status
from video_knowledge_pipeline.asr_execution import asr_smoke, run_asr_plan
from video_knowledge_pipeline.asr_runner import plan_asr_run, plan_whisperx_alignment
from video_knowledge_pipeline.local_asr_service_adapter import plan_local_asr_service_run, run_local_asr_service_plan
from video_knowledge_pipeline.batch_run import batch_video_knowledge_run
from video_knowledge_pipeline.bundle_next import bundle_advance, bundle_advance_log, bundle_advance_queue, bundle_next_action
from video_knowledge_pipeline.bundle_status import bundle_status_report, controlled_execution_check
from video_knowledge_pipeline.cli import audit_bundle_mcp_args, build_parser, main as cli_main, resolve_mcp_args_path, run_mcp_call
from video_knowledge_pipeline.config import config_status, resolve_vision_execution_profile, service_url, vision_execution_profile
from video_knowledge_pipeline.controlled_execution_smoke import controlled_execution_smoke
from video_knowledge_pipeline.knowledge_coverage import build_knowledge_coverage
from video_knowledge_pipeline.knowledge_note_export import export_knowledge_note
from video_knowledge_pipeline.lecture_package import render_lecture_review_html
from video_knowledge_pipeline.local_video_run import prepare_local_video_run
import video_knowledge_pipeline.local_video_run as local_video_run
from video_knowledge_pipeline.local_vlm_server_adapter import local_vlm_adapter_plan
from video_knowledge_pipeline.ocr_backfill import run_ocr_backfill
from video_knowledge_pipeline.multimodal_frame_analyzer import (
    _normalise_visual_understanding,
    run_multimodal_frame_analysis,
    vision_analysis_apply_restore,
    vision_analysis_restore_plan,
    vision_analysis_run_log,
)
from video_knowledge_pipeline.peepshow_adapter import attach_peepshow_output_to_bundle
from video_knowledge_pipeline.review_session import apply_review_notes_to_bundle, prepare_review_session, validate_review_notes_for_bundle
from video_knowledge_pipeline.source_artifacts import build_source_artifact_index, summarize_manifest_source_artifacts
from video_knowledge_pipeline.storage import bundle_write_lock, write_json
from video_knowledge_pipeline.temporal_frame_groups import run_temporal_frame_groups
from video_knowledge_pipeline.temporal_visual_analyzer import _normalise_temporal_understanding, run_temporal_visual_analysis
from video_knowledge_pipeline.transcript_resegment import resegment_transcript
from video_knowledge_pipeline.vision_acceptance import vision_acceptance_plan
from video_knowledge_pipeline.video_frame_router import run_video_frame_router
from video_knowledge_pipeline.video_source import prepare_video_source
from video_knowledge_pipeline.vision_api import parse_model_json, resolve_provider_config, test_vision_provider as run_vision_provider_test
from video_knowledge_pipeline.vision_environment import vision_environment_status
from video_knowledge_pipeline.vision_preflight import vision_execution_preflight
from video_knowledge_pipeline.vision_provider_smoke import rank_vision_providers, vision_provider_matrix, vision_provider_smoke
from video_knowledge_pipeline.webui_bridge import export_webui_bundle, refresh_bundle_review_html
import video_knowledge_pipeline.visual_structure as visual_structure
from video_knowledge_pipeline.visual_structure import run_visual_structure_plan



# Moved from test_video_pipeline_smoke.py during Phase 10 split.

def test_asr_env_and_smoke_cli_contract(tmp_path: Path) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")
    env_args = build_parser().parse_args(["asr-env-status", "--venv-dir", str(tmp_path / "venv"), "--output-dir", str(tmp_path / "env"), "--write"])
    smoke_args = build_parser().parse_args(["asr-smoke", str(media), "--output-dir", str(tmp_path / "smoke"), "--duration-seconds", "5", "--no-execute"])
    smoke_code = cli_main(["asr-smoke", str(media), "--output-dir", str(tmp_path / "smoke"), "--duration-seconds", "5", "--no-execute"])

    assert env_args.command == "asr-env-status"
    assert env_args.write is True
    assert smoke_args.command == "asr-smoke"
    assert smoke_args.duration_seconds == 5
    assert smoke_code == 0


def test_sensevoice_funasr_normalization_preserves_tags(tmp_path: Path) -> None:
    raw = tmp_path / "sensevoice.json"
    raw.write_text(
        json.dumps(
            {
                "sentence_info": [
                    {
                        "start": 0,
                        "end": 1200,
                        "text": "<|zh|><|NEUTRAL|><|Speech|>你好，开始讲课",
                        "spk": "speaker-1",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = normalize_asr_output(tmp_path / "workspace", raw, provider="sensevoice", title="测试")

    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    segment = payload["segments"][0]
    assert segment["text"] == "你好，开始讲课"
    assert segment["metadata"]["speaker"] == "speaker-1"
    assert segment["metadata"]["language"] == "zh"
    assert "Speech" in segment["metadata"]["audio_events"]


def test_moss_normalization_preserves_speaker_segments(tmp_path: Path) -> None:
    raw = tmp_path / "segments.json"
    raw.write_text(
        json.dumps(
            [
                {"id": "seg_0001", "start": 0.0, "end": 1.5, "speaker": "S01", "text": "你好"},
                {"id": "seg_0002", "start": 1.5, "end": 3.0, "speaker": "S02", "text": "开始访谈"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = normalize_asr_output(
        tmp_path / "workspace",
        raw,
        provider="moss-transcribe-diarize",
        title="访谈",
    )
    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

    assert result["segment_count"] == 2
    assert payload["segments"][0]["metadata"]["speaker"] == "S01"
    assert payload["segments"][1]["metadata"]["speaker"] == "S02"


def test_moss_normalization_preserves_source_order_ids_and_boundaries(tmp_path: Path) -> None:
    raw = tmp_path / "segments.json"
    source_segments = [
        {
            "id": "moss-0002",
            "start": 4.0,
            "end": 5.0,
            "speaker": "S01",
            "text": "第二句",
        },
        {
            "id": "moss-0001",
            "start": 1.0,
            "end": 1.2,
            "speaker": "S01",
            "text": "第一句",
        },
    ]
    raw.write_text(
        json.dumps(source_segments, ensure_ascii=False),
        encoding="utf-8",
    )

    result = normalize_asr_output(
        tmp_path / "workspace",
        raw,
        provider="moss-transcribe-diarize",
        title="边界保持",
    )
    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

    assert [
        (row["id"], row["start"], row["end"], row["text"])
        for row in payload["segments"]
    ] == [
        ("moss-0002", 4.0, 5.0, "第二句"),
        ("moss-0001", 1.0, 1.2, "第一句"),
    ]
    assert all(row["metadata"]["speaker"] == "S01" for row in payload["segments"])


def test_moss_plan_reuses_cli_and_blocks_missing_model(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "interview.wav"
    media.write_bytes(b"fake audio")

    import video_knowledge_pipeline.asr_runner as asr_runner

    monkeypatch.setattr(
        asr_runner,
        "_resolve_command_path",
        lambda preset: "D:/tools/mtd-subtitle.exe"
        if preset.get("command") == "mtd-subtitle"
        else "",
    )
    monkeypatch.setattr(
        asr_runner,
        "_model_ready",
        lambda preset, model: {
            "model": model,
            "ready": False,
            "cache_matches": [],
            "status": "unknown_or_not_downloaded",
        },
    )
    monkeypatch.setattr(
        asr_runner,
        "_command_runtime_probe",
        lambda **kwargs: {
            "required": True,
            "ready": True,
            "status": "ready",
            "blocker": "",
            "exit_code": 0,
        },
    )

    plan = plan_asr_run(
        tmp_path / "workspace",
        media,
        preset="moss-transcribe-diarize",
    )

    assert plan["provider"] == "moss-transcribe-diarize"
    assert plan["runner"] == "moss_transcribe_diarize_cli"
    assert plan["available"] is True
    assert plan["expected_output_json"].endswith("segments.json")
    assert "--backend" in plan["command"]
    assert "--out-dir" in plan["command"]
    assert plan["command"][plan["command"].index("--max-new-tokens") + 1] == "8192"

    executed = run_asr_plan(plan["plan_path"], execute=True)
    assert executed["status"] == "asr_model_not_ready"


def test_moss_plan_honors_explicit_generation_budget(
    tmp_path: Path, monkeypatch
) -> None:
    """Keep the reused upstream MOSS token budget operator-configurable.

    Intent: make long-media completion testable without changing other ASR presets.
    Decision: expose the upstream CLI parameter through one bounded environment value.
    Reason: a real 300-second sample hit the upstream 2048-token ceiling.
    Evidence: the planned command is the executable contract used by the runner.
    Scope: MOSS command planning only.
    """
    media = tmp_path / "interview.wav"
    media.write_bytes(b"fake audio")

    import video_knowledge_pipeline.asr_runner as asr_runner

    monkeypatch.setenv("LECTURE_MOSS_MAX_NEW_TOKENS", "16384")
    monkeypatch.setattr(
        asr_runner,
        "_resolve_command_path",
        lambda preset: "D:/tools/mtd-subtitle.exe"
        if preset.get("command") == "mtd-subtitle"
        else "",
    )
    monkeypatch.setattr(
        asr_runner,
        "_model_ready",
        lambda preset, model: {
            "model": model,
            "ready": True,
            "cache_matches": ["D:/models/moss"],
            "status": "ready",
        },
    )
    monkeypatch.setattr(
        asr_runner,
        "_command_runtime_probe",
        lambda **kwargs: {
            "required": True,
            "ready": True,
            "status": "ready",
            "blocker": "",
            "exit_code": 0,
        },
    )

    plan = plan_asr_run(
        tmp_path / "workspace",
        media,
        preset="moss-transcribe-diarize",
    )

    assert plan["command"][plan["command"].index("--max-new-tokens") + 1] == "16384"


def test_moss_plan_blocks_broken_cli_before_model_execution(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "interview.wav"
    media.write_bytes(b"fake audio")

    import video_knowledge_pipeline.asr_runner as asr_runner

    monkeypatch.setattr(
        asr_runner,
        "_resolve_command_path",
        lambda preset: "D:/tools/mtd-subtitle.exe"
        if preset.get("command") == "mtd-subtitle"
        else "",
    )
    monkeypatch.setattr(
        asr_runner,
        "_command_runtime_probe",
        lambda **kwargs: {
            "required": True,
            "ready": False,
            "status": "dependency_not_ready",
            "blocker": "missing_python_dependency:transformers",
            "exit_code": 1,
        },
    )
    monkeypatch.setattr(
        asr_runner,
        "_model_ready",
        lambda preset, model: {
            "model": model,
            "ready": True,
            "cache_matches": ["D:/models/moss"],
            "status": "ready",
        },
    )

    plan = plan_asr_run(
        tmp_path / "workspace",
        media,
        preset="moss-transcribe-diarize",
    )

    assert plan["available"] is False
    assert plan["availability"]["entrypoint_available"] is True
    assert plan["availability"]["runtime_probe"]["status"] == "dependency_not_ready"
    executed = run_asr_plan(plan["plan_path"], execute=True)
    assert executed["status"] == "blocked"
    assert "runtime_blocker='missing_python_dependency:transformers'" in executed["stderr"]
    assert executed["availability"]["runtime_probe"]["blocker"] == (
        "missing_python_dependency:transformers"
    )


def test_moss_runtime_probe_failure_is_normalized_without_traceback() -> None:
    import video_knowledge_pipeline.asr_runner as asr_runner

    traceback = (
        "Traceback (most recent call last):\n"
        "  File \"private-path\", line 1\n"
        "ModuleNotFoundError: No module named 'transformers'\n"
    )

    assert asr_runner._classify_command_probe_failure(traceback) == (
        "missing_python_dependency:transformers"
    )


def test_sensevoice_untimed_long_text_is_split_for_timeline(tmp_path: Path) -> None:
    raw = tmp_path / "sensevoice-untimed.json"
    raw.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_funasr_raw_output.v1",
                "duration_seconds": 30,
                "result": [
                    {
                        "key": "lesson",
                        "text": "<|zh|><|NEUTRAL|><|Speech|>第一段内容"
                        "<|zh|><|HAPPY|><|Speech|>第二段内容更长一点"
                        "<|zh|><|NEUTRAL|><|Speech|>第三段",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = normalize_asr_output(tmp_path / "workspace", raw, provider="sensevoice", title="测试")
    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

    assert result["segment_count"] == 3
    assert payload["segments"][0]["start"] == 0
    assert payload["segments"][1]["start"] > payload["segments"][0]["start"]
    assert payload["segments"][-1]["end"] == 30
    assert "第一段内容" in Path(result["srt_path"]).read_text(encoding="utf-8")


def test_sensevoice_untimed_long_video_duration_seconds_is_not_treated_as_milliseconds(tmp_path: Path) -> None:
    raw = tmp_path / "sensevoice-untimed-long.json"
    raw.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_funasr_raw_output.v1",
                "duration_seconds": 10214.035,
                "result": [
                    {
                        "key": "lesson",
                        "text": "<|zh|><|NEUTRAL|><|Speech|>第一段"
                        "<|zh|><|HAPPY|><|Speech|>第二段",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = normalize_asr_output(tmp_path / "workspace", raw, provider="sensevoice", title="长视频")
    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

    assert result["segment_count"] == 2
    assert payload["segments"][-1]["end"] == 10214.035






def test_sensevoice_duplicate_punctuation_is_normalized(tmp_path: Path) -> None:
    raw = tmp_path / "sensevoice-duplicate-punc.json"
    raw.write_text(
        json.dumps(
            {
                "sentence_info": [
                    {
                        "start": 0,
                        "end": 1000,
                        "text": "<|zh|><|NEUTRAL|><|Speech|>大家好，，欢迎学习。。那我们开始。，。",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = normalize_asr_output(tmp_path / "workspace", raw, provider="sensevoice", title="测试")
    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

    assert payload["segments"][0]["text"] == "大家好，欢迎学习。那我们开始。"


def test_sensevoice_spaced_tags_are_cleaned_and_split(tmp_path: Path) -> None:
    raw = tmp_path / "sensevoice-spaced-tags.json"
    raw.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_funasr_raw_output.v1",
                "duration_seconds": 60,
                "result": [
                    {
                        "key": "lesson",
                        "text": "< | zh | > < | NEUTRAL | > < | S pe ech | > < | withi tn | >第一段，有标点。"
                        "< | zh | > < | HAPPY | > < | S pe ech | >第二段，也有标点。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = normalize_asr_output(tmp_path / "workspace", raw, provider="sensevoice", title="测试")
    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

    assert result["segment_count"] == 2
    assert payload["segments"][0]["text"] == "第一段，有标点。"
    assert payload["segments"][1]["text"] == "第二段，也有标点。"
    assert payload["segments"][-1]["end"] == 60
    assert "< |" not in Path(result["srt_path"]).read_text(encoding="utf-8")


def test_funasr_python_plan_and_model_gate(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")

    import video_knowledge_pipeline.asr_runner as asr_runner

    monkeypatch.setattr(asr_runner, "_module_available_in_python", lambda module, python: True)
    monkeypatch.setattr(asr_runner, "_model_ready", lambda preset, model: {"model": model, "ready": False, "status": "unknown_or_not_downloaded"})

    plan = plan_asr_run(tmp_path / "workspace", media, preset="sensevoice", model="iic/SenseVoiceSmall")
    assert plan["runner"] == "funasr_python"
    assert plan["available"] is True
    assert plan["expected_output_json"].endswith("raw-asr-output.json")
    assert Path(plan["project"]).is_absolute()
    assert Path(plan["media_path"]).is_absolute()
    assert Path(plan["output_dir"]).is_absolute()
    assert Path(plan["expected_output_json"]).is_absolute()
    output_arg_index = plan["command"].index("--output") + 1
    assert Path(plan["command"][output_arg_index]).is_absolute()
    assert "-m" in plan["command"]
    assert "video_knowledge_pipeline.funasr_chunked_runner" in plan["command"]
    assert plan["command"][plan["command"].index("--chunk-overlap-seconds") + 1] == "5"
    assert plan["command"][plan["command"].index("--chunk-boundary-mode") + 1] == "fixed_duration"
    assert plan["full_mode"]["chunk_overlap_seconds"] == 5.0
    assert plan["full_mode"]["chunk_boundary_mode"] == "fixed_duration"

    executed = run_asr_plan(plan["plan_path"], execute=True)
    assert executed["status"] == "asr_model_not_ready"
    assert Path(executed["asr_log"]["report_path"]).exists()


def test_plan_asr_run_exposes_silence_snap_and_overlap_controls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")

    import video_knowledge_pipeline.asr_runner as asr_runner

    monkeypatch.setattr(
        asr_runner, "_module_available_in_python", lambda module, python: True
    )
    monkeypatch.setattr(
        asr_runner,
        "_model_ready",
        lambda preset, model: {"model": model, "ready": True, "status": "ready"},
    )

    plan = plan_asr_run(
        tmp_path / "workspace",
        media,
        preset="sensevoice",
        chunk_boundary_mode="silence_snap",
        chunk_overlap_seconds=7.5,
    )

    command = plan["command"]
    assert command[command.index("--chunk-boundary-mode") + 1] == "silence_snap"
    assert command[command.index("--chunk-overlap-seconds") + 1] == "7.5"
    assert plan["full_mode"]["chunk_boundary_mode"] == "silence_snap"
    assert plan["full_mode"]["chunk_overlap_seconds"] == 7.5


def test_funasr_python_runner_prefers_local_model_cache(tmp_path: Path, monkeypatch) -> None:
    import video_knowledge_pipeline.funasr_python_runner as runner

    cache = tmp_path / "modelscope"
    sensevoice = cache / "hub" / "models" / "iic" / "SenseVoiceSmall"
    vad = cache / "hub" / "models" / "iic" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
    sensevoice.mkdir(parents=True)
    vad.mkdir(parents=True)
    monkeypatch.setenv("MODELSCOPE_CACHE", str(cache))
    monkeypatch.delenv("LECTURE_ASR_FORCE_REMOTE_MODEL", raising=False)

    assert runner._resolve_local_model("iic/SenseVoiceSmall") == str(sensevoice.resolve())
    assert runner._resolve_local_model("fsmn-vad") == str(vad.resolve())

    monkeypatch.setenv("LECTURE_ASR_FORCE_REMOTE_MODEL", "1")
    assert runner._resolve_local_model("iic/SenseVoiceSmall") == "iic/SenseVoiceSmall"


def test_asr_env_status_reports_readiness_and_actionable_next_steps(tmp_path: Path, monkeypatch) -> None:
    import video_knowledge_pipeline.asr_environment as asr_environment

    venv = tmp_path / "asr-env"
    scripts = venv / "Scripts"
    scripts.mkdir(parents=True)
    python = scripts / "python.exe"
    python.write_text("fake", encoding="utf-8")
    monkeypatch.setattr(asr_environment, "_python_info", lambda path: {"version": "3.11.9", "major": 3, "minor": 11, "asr_recommended": True, "warning": ""})
    monkeypatch.setattr(asr_environment, "_runtime_info", lambda path, venv_exists: {"torch_available": True, "cuda_available": False, "device": "cpu"})
    monkeypatch.setattr(asr_environment, "_tool_status", lambda tool, **kwargs: {
        "name": tool["name"],
        "role": tool["role"],
        "module": {"available": tool["name"] in {"funasr", "sensevoice"}},
        "command_exists": tool["name"] in {"funasr", "sensevoice"},
        "install_command": "install",
    })
    monkeypatch.setattr(asr_environment, "resolve_media_tool", lambda name: "ffmpeg.exe")
    monkeypatch.setattr(asr_environment, "_model_ready", lambda preset, model: {"model": model, "ready": False, "status": "unknown_or_not_downloaded"})
    monkeypatch.delenv("LECTURE_ASR_ALLOW_MODEL_DOWNLOAD", raising=False)

    status = asr_environment_status(venv, output_dir=tmp_path / "out", write=True)

    checks = {item["key"]: item for item in status["readiness"]}
    assert status["module_ready"] is True
    assert status["cuda_dll"]["schema"] == "video_knowledge_pipeline.cuda_dll_discovery.v1"
    assert checks["python_package_missing"]["ok"] is True
    assert checks["ffmpeg_missing"]["ok"] is True
    assert checks["model_cache_missing"]["ok"] is False
    assert checks["model_download_disabled"]["ok"] is False
    assert checks["cpu_ready"]["ok"] is True
    assert checks["cuda_ready"]["ok"] is False
    assert status["next_action"]["key"] == "prepare_asr_model_cache"
    assert "Audio is not uploaded" in status["privacy"]
    assert Path(status["output_markdown"]).exists()
    assert "ASR Environment Status" in Path(status["output_markdown"]).read_text(encoding="utf-8")


def test_asr_env_status_reports_cuda_ready_when_available(tmp_path: Path, monkeypatch) -> None:
    import video_knowledge_pipeline.asr_environment as asr_environment

    venv = tmp_path / "asr-env"
    scripts = venv / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").write_text("fake", encoding="utf-8")
    monkeypatch.setattr(asr_environment, "_python_info", lambda path: {"version": "3.11.9", "major": 3, "minor": 11, "asr_recommended": True, "warning": ""})
    monkeypatch.setattr(asr_environment, "_runtime_info", lambda path, venv_exists: {"torch_available": True, "cuda_available": True, "device": "cuda"})
    monkeypatch.setattr(asr_environment, "_tool_status", lambda tool, **kwargs: {
        "name": tool["name"],
        "role": tool["role"],
        "module": {"available": tool["name"] in {"funasr", "sensevoice"}},
        "command_exists": tool["name"] in {"funasr", "sensevoice"},
        "install_command": "install",
    })
    monkeypatch.setattr(asr_environment, "resolve_media_tool", lambda name: "ffmpeg.exe")
    monkeypatch.setattr(asr_environment, "_model_ready", lambda preset, model: {"model": model, "ready": True, "status": "ready"})

    status = asr_environment_status(venv)

    checks = {item["key"]: item for item in status["readiness"]}
    assert checks["cuda_ready"]["ok"] is True
    assert status["next_action"]["key"] == "run_asr_smoke"


def test_asr_env_status_exposes_broken_moss_runtime_without_marking_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import video_knowledge_pipeline.asr_environment as asr_environment

    venv = tmp_path / "asr-env"
    scripts = venv / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").write_text("fake", encoding="utf-8")
    moss_command = tmp_path / "mtd-subtitle.exe"
    moss_command.write_text("stub", encoding="utf-8")
    monkeypatch.setenv("LECTURE_MOSS_TRANSCRIBE_COMMAND", str(moss_command))
    monkeypatch.setattr(
        asr_environment,
        "_python_info",
        lambda path: {
            "version": "3.12.11",
            "major": 3,
            "minor": 12,
            "asr_recommended": True,
            "warning": "",
        },
    )
    monkeypatch.setattr(
        asr_environment,
        "_runtime_info",
        lambda path, venv_exists: {
            "torch_available": False,
            "cuda_available": False,
            "device": "cpu",
        },
    )
    monkeypatch.setattr(
        asr_environment,
        "_module_status",
        lambda module, **kwargs: {
            "module": module,
            "available": False,
            "returncode": 1,
        },
    )
    monkeypatch.setattr(
        asr_environment,
        "_command_runtime_probe",
        lambda preset, command_path: {
            "required": preset == "moss-transcribe-diarize",
            "ready": False,
            "status": "dependency_not_ready"
            if preset == "moss-transcribe-diarize"
            else "entrypoint_missing",
            "blocker": "missing_python_dependency:transformers"
            if preset == "moss-transcribe-diarize"
            else "command_not_found",
            "exit_code": 1 if preset == "moss-transcribe-diarize" else None,
        },
    )
    monkeypatch.setattr(asr_environment, "resolve_media_tool", lambda name: "ffmpeg.exe")
    monkeypatch.setattr(
        asr_environment,
        "_model_ready",
        lambda preset, model: {
            "model": model,
            "ready": False,
            "status": "unknown_or_not_downloaded",
        },
    )

    status = asr_environment_status(
        venv,
        output_dir=tmp_path / "out",
        write=True,
    )

    moss = next(
        row for row in status["tools"] if row["name"] == "moss-transcribe-diarize"
    )
    assert moss["command_exists"] is True
    assert moss["configured_by_environment"] is True
    assert moss["runtime_ready"] is False
    assert moss["runtime_probe"]["blocker"] == (
        "missing_python_dependency:transformers"
    )
    assert "moss-transcribe-diarize" in status["runtime_blocked_tools"]
    assert "moss-transcribe-diarize" not in status["command_tools"]
    markdown = Path(status["output_markdown"]).read_text(encoding="utf-8")
    assert "missing_python_dependency:transformers" in markdown


def test_asr_smoke_execute_blocks_when_ffmpeg_missing(tmp_path: Path, monkeypatch) -> None:
    import video_knowledge_pipeline.asr_execution as asr_execution

    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")
    monkeypatch.setattr(asr_execution, "resolve_media_tool", lambda name: "")

    result = asr_smoke(media, output_dir=tmp_path / "smoke", execute=True)

    assert result["status"] == "ffmpeg_missing"
    assert Path(result["report_path"]).exists()


def test_faster_whisper_python_runner_plan_prefers_gpu_defaults(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")

    import video_knowledge_pipeline.asr_runner as asr_runner

    monkeypatch.delenv("LECTURE_ASR_DEVICE", raising=False)
    monkeypatch.delenv("LECTURE_ASR_COMPUTE_TYPE", raising=False)
    monkeypatch.setattr(asr_runner, "_module_available_in_python", lambda module, python: module == "faster_whisper")
    monkeypatch.setattr(asr_runner, "_cuda_available_in_python", lambda python: True)
    monkeypatch.setattr(asr_runner, "_model_ready", lambda preset, model: {"model": model, "ready": True, "status": "ready"})

    plan = plan_asr_run(tmp_path / "workspace", media, preset="faster-whisper", model="large-v3")

    assert plan["runner"] == "faster_whisper_python"
    assert plan["local_asr_device"] == "cuda"
    assert "video_knowledge_pipeline.faster_whisper_python_runner" in plan["command"]
    assert "--device" in plan["command"]
    assert "cuda" in plan["command"]
    assert "--compute-type" in plan["command"]
    assert "float16" in plan["command"]
    assert "--vad-filter" in plan["command"]


def test_whisperx_alignment_plan_uses_whisper_model_language_and_gpu_device(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")

    import video_knowledge_pipeline.asr_runner as asr_runner

    monkeypatch.delenv("LECTURE_ASR_DEVICE", raising=False)
    monkeypatch.setattr(asr_runner, "_resolve_command_path", lambda preset: "whisperx.exe" if preset["provider"] == "whisperx" else "")
    monkeypatch.setattr(asr_runner, "_cuda_available_in_python", lambda python: True)
    monkeypatch.setattr(asr_runner, "_model_ready", lambda preset, model: {"model": model, "ready": True, "status": "ready"})

    plan = plan_whisperx_alignment(tmp_path / "workspace", media, language="zh")

    assert plan["preset"] == "whisperx"
    assert plan["provider"] == "whisperx"
    assert plan["runner"] == "subprocess_command"
    assert "--model" in plan["command"]
    assert "large-v3" in plan["command"]
    assert "iic/SenseVoiceSmall" not in plan["command"]
    assert "--language" in plan["command"]
    assert "zh" in plan["command"]
    assert "--device" in plan["command"]
    assert "cuda" in plan["command"]



def test_whisperx_alignment_plan_can_use_python_module_when_command_missing(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")

    import video_knowledge_pipeline.asr_runner as asr_runner

    monkeypatch.delenv("LECTURE_ASR_DEVICE", raising=False)
    monkeypatch.setattr(asr_runner, "_resolve_command_path", lambda preset: "")
    monkeypatch.setattr(asr_runner, "_module_available_in_python", lambda module, python: module == "whisperx")
    monkeypatch.setattr(asr_runner, "_cuda_available_in_python", lambda python: False)
    monkeypatch.setattr(asr_runner, "_model_ready", lambda preset, model: {"model": model, "ready": True, "status": "ready"})

    plan = plan_whisperx_alignment(tmp_path / "workspace", media, language="zh")

    assert plan["available"] is True
    assert plan["runner"] == "whisperx_python_module"
    assert plan["command"][:3] == [plan["python_executable"], "-m", "whisperx"]
    assert plan["availability"]["command_path"] == ""
    assert plan["availability"]["module_available"] is True


def test_cli_plan_asr_whisperx_default_does_not_use_sensevoice_model(tmp_path: Path, monkeypatch, capsys) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")

    import video_knowledge_pipeline.asr_runner as asr_runner

    monkeypatch.setattr(asr_runner, "_resolve_command_path", lambda preset: "whisperx.exe" if preset["provider"] == "whisperx" else "")
    monkeypatch.setattr(asr_runner, "_model_ready", lambda preset, model: {"model": model, "ready": True, "status": "ready"})

    code = cli_main(["plan-asr", str(tmp_path / "workspace"), str(media), "--preset", "whisperx"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "large-v3" in payload["command"]
    assert "iic/SenseVoiceSmall" not in payload["command"]


def test_whisperx_normalization_preserves_word_timestamps_and_speakers(tmp_path: Path) -> None:
    raw = tmp_path / "whisperx.json"
    raw.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.5,
                        "text": "你好 世界",
                        "speaker": "SPEAKER_00",
                        "language": "zh",
                        "words": [
                            {"word": "你好", "start": 0.1, "end": 0.5, "score": 0.98, "speaker": "SPEAKER_00"},
                            {"word": "世界", "start": 0.7, "end": 1.2, "score": 0.97, "speaker": "SPEAKER_00"},
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = normalize_asr_output(tmp_path / "workspace", raw, provider="whisperx", title="测试")

    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    metadata = payload["segments"][0]["metadata"]
    assert metadata["speaker"] == "SPEAKER_00"
    assert metadata["language"] == "zh"
    assert metadata["alignment"] == "word_level"
    assert metadata["word_count"] == 2
    assert metadata["words"][0]["word"] == "你好"
    assert metadata["words"][0]["start"] == 0.1



def test_run_asr_plan_blocked_includes_availability_detail(tmp_path: Path) -> None:
    project = tmp_path / "workspace"
    output_dir = project / "transcripts" / "asr_run_fake"
    output_dir.mkdir(parents=True)
    plan = output_dir / "asr-run-plan.json"
    plan.write_text(
        json.dumps(
            {
                "project": str(project),
                "preset": "whisperx",
                "provider": "whisperx",
                "media_path": str(tmp_path / "lesson.mp4"),
                "output_dir": str(output_dir),
                "expected_output_json": str(output_dir / "raw-asr-output.json"),
                "command": ["whisperx", "lesson.mp4"],
                "available": False,
                "availability": {"command_path": "", "module": "whisperx", "module_available": False},
            }
        ),
        encoding="utf-8",
    )

    result = run_asr_plan(plan, execute=True)

    assert result["status"] == "blocked"
    assert result["availability"]["module"] == "whisperx"
    assert "module_available=False" in result["stderr"]


def test_run_asr_plan_mirrors_provider_named_json_to_raw_output(tmp_path: Path, monkeypatch) -> None:
    import subprocess
    import video_knowledge_pipeline.asr_execution as asr_execution

    project = tmp_path / "workspace"
    output_dir = project / "transcripts" / "asr_run_fake"
    output_dir.mkdir(parents=True)
    expected = output_dir / "raw-asr-output.json"
    provider_named = output_dir / "lesson.json"
    plan = output_dir / "asr-run-plan.json"
    plan.write_text(
        json.dumps(
            {
                "project": str(project),
                "preset": "whisperx",
                "provider": "whisperx",
                "media_path": str(tmp_path / "lesson.mp4"),
                "output_dir": str(output_dir),
                "expected_output_json": str(expected),
                "command": ["whisperx", "lesson.mp4"],
                "available": True,
            }
        ),
        encoding="utf-8",
    )

    def fake_run(*_args, **_kwargs):
        provider_named.write_text(
            json.dumps(
                {
                    "segments": [{"start": 0, "end": 1, "text": "hello"}],
                    "runtime_metrics": {
                        "status": "available",
                        "measured_chunk_count": 1,
                        "total_child_elapsed_seconds": 2.5,
                        "max_cuda_peak_memory_allocated_mib": 512.0,
                        "max_cuda_peak_memory_reserved_mib": 768.0,
                        "missing_chunk_indexes": [],
                    },
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(["whisperx"], 0, "", "")

    monkeypatch.setattr(asr_execution, "_run_command", fake_run)

    result = run_asr_plan(plan, execute=True)

    assert result["status"] == "ok"
    assert result["raw_output_json"] == str(expected.resolve())
    assert expected.exists()
    assert result["normalized"]["segment_count"] == 1
    assert result["runtime_metrics"]["total_child_elapsed_seconds"] == 2.5
    report = (output_dir / "asr-run-report.md").read_text(encoding="utf-8")
    assert "## Runtime Metrics" in report
    assert "Max CUDA peak allocated MiB: `512.0`" in report



def test_run_asr_plan_alignment_sidecar_never_replaces_normalized_transcript(tmp_path: Path, monkeypatch) -> None:
    import subprocess
    import video_knowledge_pipeline.asr_execution as asr_execution

    project = tmp_path / "workspace"
    project.mkdir(parents=True)
    (project / "manifest.json").write_text("{}", encoding="utf-8")
    transcript = project / "normalized-transcript.json"
    transcript.write_text(
        json.dumps({"segments": [{"start": 0, "end": 1, "text": "hello"}]}),
        encoding="utf-8",
    )
    output_dir = project / "transcripts" / "asr_run_aligner"
    output_dir.mkdir(parents=True)
    expected = output_dir / "raw-asr-output.json"
    plan = output_dir / "asr-run-plan.json"
    plan.write_text(
        json.dumps(
            {
                "project": str(project),
                "preset": "qwen3-forced-aligner",
                "provider": "qwen3_forced_aligner",
                "asr_mode": "alignment",
                "media_path": str(tmp_path / "lesson.wav"),
                "alignment_transcript": str(transcript),
                "output_dir": str(output_dir),
                "expected_output_json": str(expected),
                "command": ["python", "aligner.py"],
                "available": True,
            }
        ),
        encoding="utf-8",
    )
    normalize_calls: list[object] = []

    def fake_run(*_args, **_kwargs):
        expected.write_text(
            json.dumps(
                {
                    "schema": "video_knowledge_pipeline.qwen3_forced_aligner_output.v1",
                    "status": "completed",
                    "provider": "qwen3-forced-aligner",
                    "model": "Qwen/Qwen3-ForcedAligner-0.6B",
                    "transcript_path": str(transcript),
                    "timestamps_monotonic": True,
                    "words": [{"text": "hello", "start": 0, "end": 0.4}],
                    "segments": [{"start": 0, "end": 1, "text": "hello", "words": [{"text": "hello", "start": 0, "end": 0.4}]}],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(["python"], 0, "", "")

    def fake_normalize(*args, **kwargs):
        normalize_calls.append((args, kwargs))
        return {"unexpected": True}

    monkeypatch.setattr(asr_execution, "_run_command", fake_run)
    monkeypatch.setattr(asr_execution, "normalize_asr_output", fake_normalize)

    result = run_asr_plan(plan, execute=True, normalize=True)

    assert result["status"] == "ok"
    assert result["normalize_requested"] is True
    assert result["normalize"] is False
    assert result["normalization_skipped_reason"] == "alignment_sidecar"
    assert result["normalized"] is None
    assert normalize_calls == []
    assert expected.exists()
    assert result["alignment_sidecar_registration"]["status"] == "registered_read_only_sidecar"
    manifest = json.loads((project / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["qwen3_forced_alignment_json"] == str(expected.relative_to(project))
    assert manifest["asr_alignment_sidecars"][0]["does_not_replace_canonical_transcript"] is True

def test_run_asr_plan_recovers_nested_raw_output(tmp_path: Path, monkeypatch) -> None:
    import subprocess
    import video_knowledge_pipeline.asr_execution as asr_execution

    project = tmp_path / "workspace"
    output_dir = project / "transcripts" / "asr_run_fake"
    nested_dir = output_dir / "workspace" / "transcripts" / "asr_run_fake"
    output_dir.mkdir(parents=True)
    nested_dir.mkdir(parents=True)
    expected = output_dir / "raw-asr-output.json"
    nested_raw = nested_dir / "raw-asr-output.json"
    plan = output_dir / "asr-run-plan.json"
    plan.write_text(
        json.dumps(
            {
                "project": str(project),
                "preset": "sensevoice",
                "provider": "sensevoice",
                "media_path": str(tmp_path / "lesson.mp4"),
                "output_dir": str(output_dir),
                "expected_output_json": str(expected),
                "command": ["funasr", "lesson.mp4"],
                "available": True,
            }
        ),
        encoding="utf-8",
    )

    def fake_run(*_args, **_kwargs):
        nested_raw.write_text(json.dumps({"sentence_info": [{"start": 0, "end": 1000, "text": "hello"}]}), encoding="utf-8")
        return subprocess.CompletedProcess(["funasr"], 0, "", "")

    monkeypatch.setattr(asr_execution, "_run_command", fake_run)

    result = run_asr_plan(plan, execute=True)

    assert result["status"] == "ok"
    assert result["raw_output_json"] == str(expected.resolve())
    assert expected.exists()
    assert result["normalized"]["segment_count"] == 1



def test_prepare_local_video_run_injects_metadata_hotwords_before_asr_planning(
    tmp_path: Path,
    monkeypatch,
) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        local_video_run,
        "build_entity_lexicon",
        lambda *_args, **_kwargs: {"hotword_text": "钟巍 王道", "hotword_variant_count": 2},
    )

    def fake_plan(_root, _media, **kwargs):
        captured.update(kwargs)
        return {"preset": kwargs["preset"], "plan_path": str(tmp_path / "asr-run-plan.json")}

    monkeypatch.setattr(local_video_run, "plan_asr_run", fake_plan)
    result = local_video_run.prepare_local_video_run(
        media,
        tmp_path / "run",
        title="短视频营销：获客只是开始，持续经营才是王道！-钟巍",
    )

    assert captured["hotword"] == "钟巍 王道"
    assert result["pre_asr_context"]["hotword_count"] == 2
    assert result["pre_asr_context"]["evaluation_reference_used"] is False
    assert "- Pre-ASR hotwords: `2`" in Path(result["markdown_path"]).read_text(encoding="utf-8")



def test_prepare_local_video_run_with_transcript_does_not_require_asr_execute(tmp_path: Path) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps({"segments": []}), encoding="utf-8")

    result = prepare_local_video_run(media, tmp_path / "run", title="课程测试", transcript_path=transcript)

    keys = [step.get("key") for step in result["next_steps"]]
    assert "review_transcript" in keys
    assert "run_asr_plan" not in keys
    markdown = Path(result["markdown_path"]).read_text(encoding="utf-8")
    assert "Selected transcript" in markdown


def test_resegment_transcript_marks_estimated_timing(tmp_path: Path) -> None:
    source = tmp_path / "coarse.json"
    source.write_text(
        json.dumps(
            {
                "title": "粗转写",
                "provider": "sensevoice",
                "segments": [
                    {
                        "start": 0,
                        "end": 0,
                        "text": "第一段内容很长需要切开，里面包含背景、问题和限制条件。第二段继续说明流程，包含操作步骤、判断标准和复核方式。第三段给出结论，并说明后续要怎么验证。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = resegment_transcript(tmp_path / "workspace", source, duration_seconds=30, max_chars=12)

    assert result["timing_estimated"] is True
    assert result["segment_count"] > 1
    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    assert payload["timing_estimated"] is True
    assert payload["segments"][-1]["end"] == 30


def test_run_asr_plan_registers_bundle_preview_run(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    output_dir = bundle / "transcripts" / "asr_run_preview"
    output_dir.mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}, ensure_ascii=False), encoding="utf-8")
    expected = output_dir / "raw-asr-output.json"
    plan = output_dir / "asr-run-plan.json"
    plan.write_text(
        json.dumps(
            {
                "project": str(bundle),
                "preset": "sensevoice",
                "provider": "sensevoice",
                "media_path": str(tmp_path / "lesson.mp4"),
                "output_dir": str(output_dir),
                "expected_output_json": str(expected),
                "command": ["python", "-m", "fake_asr"],
                "available": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_asr_plan(plan, execute=False)

    assert result["status"] == "preview"
    assert result["run_registry"]["run_type"] == "asr_run_plan"
    assert result["run_registry"]["status"] == "needs_execution"
    run = json.loads((bundle / "runs" / "asr-run-plan" / "run.json").read_text(encoding="utf-8"))
    registry = json.loads((bundle / "run-artifact-registry.json").read_text(encoding="utf-8"))
    assert run["retry_command"].startswith(".\\scripts\\video-knowledge.ps1 run-asr-plan")
    assert registry["status_counts"] == {"needs_execution": 1}


def test_sensevoice_plan_defaults_to_full_funasr_mode(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")

    import video_knowledge_pipeline.asr_runner as asr_runner

    monkeypatch.setattr(asr_runner, "_module_available_in_python", lambda module, python: True)
    monkeypatch.setattr(asr_runner, "_model_ready", lambda preset, model: {"model": model, "ready": True, "status": "ready"})

    plan = plan_asr_run(tmp_path / "workspace", media, preset="sensevoice", model="iic/SenseVoiceSmall")

    assert plan["asr_mode"] == "full"
    assert plan["full_mode"]["vad_model"] == "fsmn-vad"
    assert plan["full_mode"]["punc_model"] == "ct-punc"
    assert plan["full_mode"]["use_itn"] is True
    assert plan["full_mode"]["merge_vad"] is True
    assert "--punc-model" in plan["command"]
    assert "ct-punc" in plan["command"]
    assert "--use-itn" in plan["command"]
    assert "--merge-vad" in plan["command"]
    assert "--vad-max-single-segment-time-ms" in plan["command"]


def test_funasr_python_runner_passes_full_mode_options(tmp_path: Path, monkeypatch) -> None:
    import video_knowledge_pipeline.funasr_python_runner as runner

    media = tmp_path / "lesson.wav"
    media.write_bytes(b"fake audio")
    output = tmp_path / "raw.json"
    captured = {}

    class FakeAutoModel:
        def __init__(self, **kwargs):
            captured["model_kwargs"] = kwargs

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return [{"text": "你好"}]

    monkeypatch.setitem(__import__("sys").modules, "funasr", type("FakeFunASR", (), {"AutoModel": FakeAutoModel}))
    monkeypatch.setattr(runner, "_select_device", lambda device: "cuda")
    monkeypatch.setattr(
        runner,
        "_start_runtime_metrics",
        lambda device: {"device": device, "started_perf_counter": 0.0},
    )
    monkeypatch.setattr(
        runner,
        "_finish_runtime_metrics",
        lambda state: {
            "device": state["device"],
            "elapsed_seconds": 1.25,
            "cuda_peak_memory_allocated_mib": 4096.0,
        },
    )

    result = runner.run_funasr(
        input_path=str(media),
        output_path=str(output),
        provider="sensevoice",
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        spk_model="",
        use_itn=True,
        merge_vad=True,
        merge_length_s=12,
        vad_max_single_segment_time_ms=24000,
        device="auto",
    )

    assert result["ok"] is True
    assert captured["model_kwargs"]["vad_kwargs"] == {"max_single_segment_time": 24000}
    assert captured["model_kwargs"]["punc_model"]
    assert captured["model_kwargs"]["device"] == "cuda"
    assert captured["generate_kwargs"]["use_itn"] is True
    assert captured["generate_kwargs"]["merge_vad"] is True
    assert captured["generate_kwargs"]["merge_length_s"] == 12
    assert captured["generate_kwargs"]["sentence_timestamp"] is True
    assert captured["generate_kwargs"]["output_timestamp"] is True
    assert captured["generate_kwargs"]["return_time_stamps"] is True
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["runtime_metrics"] == {
        "device": "cuda",
        "elapsed_seconds": 1.25,
        "cuda_peak_memory_allocated_mib": 4096.0,
    }



def test_run_whisperx_alignment_preview_keeps_primary_asr_boundary(tmp_path: Path) -> None:
    from video_knowledge_pipeline.whisperx_alignment import run_whisperx_alignment

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"title": "lesson"}, ensure_ascii=False), encoding="utf-8")
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")

    result = run_whisperx_alignment(bundle, media, execute=False, write=True)

    assert result["status"] == "preview"
    assert result["operator_boundary"]["does_not_replace_primary_asr"] is True
    assert result["operator_boundary"]["does_not_promote_corrected_transcript"] is True
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["whisperx_alignment_run_json"] == "transcripts\\whisperx-alignment-run.json" or manifest["whisperx_alignment_run_json"] == "transcripts/whisperx-alignment-run.json"
    assert "corrected_transcript_json" not in manifest
    assert (bundle / "transcripts" / "whisperx-alignment-run.json").exists()
    assert (bundle / "notes" / "whisperx-alignment-run.md").exists()




def test_dolphin_raw_output_normalizes_to_transcript(tmp_path: Path) -> None:
    raw = tmp_path / "dolphin.json"
    raw.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_dolphin_raw_output.v1",
                "provider": "dolphin",
                "duration_seconds": 30,
                "result": {
                    "segments": [
                        {"start": 0.0, "end": 5.0, "text": "第一段内容"},
                        {"start": 5.0, "end": 12.0, "text": "第二段内容"},
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = normalize_asr_output(tmp_path / "workspace", raw, provider="dolphin", title="Dolphin")
    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

    assert result["segment_count"] == 2
    assert payload["segments"][0]["text"] == "第一段内容"
    assert payload["segments"][1]["end"] == 12.0




def test_asr_ab_sample_plan_reuses_existing_sample_file(tmp_path: Path) -> None:
    from video_knowledge_pipeline.asr_ab_plan import plan_asr_ab_sample

    sample_dir = tmp_path / "workspace" / "transcripts" / "asr-ab-sample"
    sample_dir.mkdir(parents=True)
    media = sample_dir / "lesson.sample-0-300.mp4"
    media.write_bytes(b"fake sample")

    plan = plan_asr_ab_sample(tmp_path / "workspace", media, write=False)

    assert plan["sample_reused_existing"] is True
    assert plan["sample_media_path"] == str(media.resolve())
    assert plan["sample_extract_command"] == []
    dolphin = {row["key"]: row for row in plan["variants"]}["dolphin"]
    assert media.name in " ".join(dolphin["command"])


def test_asr_ab_sample_plan_includes_runnable_dolphin_adapter(tmp_path: Path) -> None:
    from video_knowledge_pipeline.asr_ab_plan import plan_asr_ab_sample

    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")

    plan = plan_asr_ab_sample(tmp_path / "workspace", media, write=False)

    dolphin = {row["key"]: row for row in plan["variants"]}["dolphin"]
    assert dolphin["status"] == "planned_optional_local_adapter"
    assert dolphin["runner"] == "dolphin_python"
    assert "video_knowledge_pipeline.dolphin_python_runner" in dolphin["command"]
    assert dolphin["operator_boundary"]["second_evidence_source_only"] is True


def test_asr_ab_sample_plan_includes_campp_gpu_speaker_variant(
    tmp_path: Path,
) -> None:
    from video_knowledge_pipeline.asr_ab_plan import plan_asr_ab_sample

    media = tmp_path / "dialogue.wav"
    media.write_bytes(b"fake audio")

    plan = plan_asr_ab_sample(tmp_path / "workspace", media, write=False)

    campp = {row["key"]: row for row in plan["variants"]}[
        "sensevoice_full_punc_campp"
    ]
    command = campp["command"]
    assert campp["role"] == "local_primary_asr_with_candidate_speaker_labels"
    assert command[command.index("--device") + 1] == "cuda"
    assert command[command.index("--spk-model") + 1] == "cam++"
    assert command[command.index("--punc-model") + 1] == "ct-punc"
    assert campp["operator_boundary"] == {
        "does_not_promote_any_transcript": True,
        "speaker_labels_are_anonymous_candidates": True,
        "speaker_roles_are_not_inferred": True,
        "requires_prepared_local_speaker_model": True,
        "gpu_only": True,
    }
    assert plan["operator_boundary"]["campp_variant_is_gpu_only"] is True


def test_asr_ab_campp_execute_blocks_before_runner_when_model_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from video_knowledge_pipeline import asr_ab_run
    from video_knowledge_pipeline.asr_ab_run import run_asr_ab_sample

    media = tmp_path / "dialogue.wav"
    media.write_bytes(b"fake audio")
    punc_model = tmp_path / "ct-punc"
    punc_model.mkdir()
    monkeypatch.setattr(
        asr_ab_run,
        "_resolve_local_model",
        lambda model: str(punc_model) if model == "ct-punc" else model,
    )

    def fail_if_executed(*args, **kwargs):
        raise AssertionError("missing CAM++ must block before the ASR runner")

    monkeypatch.setattr(asr_ab_run, "run_asr_plan", fail_if_executed)

    result = run_asr_ab_sample(
        tmp_path / "workspace",
        media,
        variants=["sensevoice_full_punc_campp"],
        execute_local=True,
        write=False,
    )

    row = result["variants"][0]
    readiness = {item["flag"]: item for item in row["model_ready"]["required"]}
    assert row["status"] == "asr_model_not_ready"
    assert row["reason"] == "adjunct_model_not_ready"
    assert readiness["--punc-model"]["ready"] is True
    assert readiness["--spk-model"] == {
        "flag": "--spk-model",
        "model": "cam++",
        "resolved": "cam++",
        "ready": False,
    }


def test_asr_ab_metrics_report_anonymous_speaker_coverage(tmp_path: Path) -> None:
    from video_knowledge_pipeline.asr_ab_run import _transcript_metrics

    transcript = tmp_path / "candidate.json"
    transcript.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 5.0, "text": "你好", "speaker": "S01"},
                    {
                        "start": 5.0,
                        "end": 10.0,
                        "text": "请讲",
                        "metadata": {"speaker": "S02"},
                    },
                    {"start": 10.0, "end": 12.0, "text": "未标注"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    metrics = _transcript_metrics(str(transcript))

    assert metrics["speaker_count"] == 2
    assert metrics["speaker_labeled_segment_count"] == 2
    assert metrics["speaker_labeled_duration_seconds"] == 10.0
    assert metrics["speaker_labeled_duration_ratio"] == 0.833333




def test_asr_ab_dolphin_project_python_env_includes_src_path() -> None:
    from video_knowledge_pipeline import asr_ab_run

    env = asr_ab_run._project_python_env()

    assert "PYTHONPATH" in env
    assert str(Path("src").resolve()) in env["PYTHONPATH"]


def test_asr_ab_dolphin_execute_blocks_when_package_missing(tmp_path: Path, monkeypatch) -> None:
    from video_knowledge_pipeline import asr_ab_run
    from video_knowledge_pipeline.asr_ab_run import run_asr_ab_sample

    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")
    monkeypatch.setattr(asr_ab_run, "_python_module_ready", lambda python, module: {"ready": False, "python": python, "module": module, "error": "missing"})

    result = run_asr_ab_sample(tmp_path / "workspace", media, variants=["dolphin"], execute_local=True, write=False)

    row = result["variants"][0]
    assert row["key"] == "dolphin"
    assert row["status"] == "asr_module_not_ready"
    assert row["reason"] == "dolphin_python_package_missing"
    assert row["operator_boundary"]["second_evidence_source_only"] is True




def test_asr_ab_compare_recommends_full_punc_but_blocks_second_asr(tmp_path: Path) -> None:
    from video_knowledge_pipeline.asr_ab_compare import compare_asr_ab_sample

    run = tmp_path / "asr-ab-sample-run.json"
    run.write_text(
        json.dumps(
            {
                "workspace_dir": str(tmp_path),
                "sample_media_path": str(tmp_path / "sample.mp4"),
                "variants": [
                    {"key": "sensevoice_basic", "status": "ok", "metrics": {"segment_count": 10, "char_count": 1000, "punctuation_count": 0, "duration_seconds": 300}},
                    {"key": "sensevoice_full_punc", "status": "ok", "metrics": {"segment_count": 12, "char_count": 1100, "punctuation_count": 90, "duration_seconds": 300}},
                    {"key": "dolphin", "status": "asr_module_not_ready", "metrics": {}},
                    {"key": "openai_cloud_asr", "status": "preview", "metrics": {}},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = compare_asr_ab_sample(run, write=True)

    assert result["primary_recommendation"] == "sensevoice_full_punc"
    assert result["second_asr_recommendation"] == "do_not_introduce_second_asr_by_default_yet"
    assert result["gates"]["sensevoice_full_punc_ready"] is True
    assert result["gates"]["can_decide_second_asr"] is False
    assert Path(result["json_path"]).exists()
    assert "Dolphin" in Path(result["markdown_path"]).read_text(encoding="utf-8")



def test_asr_ab_compare_reports_dolphin_torchcodec_blocker(tmp_path: Path) -> None:
    from video_knowledge_pipeline.asr_ab_compare import compare_asr_ab_sample

    run = tmp_path / "asr-ab-sample-run.json"
    run.write_text(
        json.dumps(
            {
                "workspace_dir": str(tmp_path),
                "sample_media_path": str(tmp_path / "sample.mp4"),
                "variants": [
                    {"key": "sensevoice_full_punc", "status": "ok", "metrics": {"segment_count": 12, "char_count": 1100, "punctuation_count": 90, "duration_seconds": 300}},
                    {
                        "key": "dolphin",
                        "status": "failed",
                        "stdout_tail": "Could not load libtorchcodec_core6.dll",
                        "audio_extract": {"status": "ok", "audio_path": str(tmp_path / "dolphin-input.wav")},
                        "module_ready": {"ready": True},
                        "metrics": {},
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = compare_asr_ab_sample(run, write=True)

    dolphin = {row["key"]: row for row in result["variants"]}["dolphin"]
    assert "audio_extract_ok_but_dolphin_runtime_failed" in dolphin["risks"]
    assert dolphin["blockers"][0]["code"] == "dolphin_torchcodec_runtime_not_ready"
    assert dolphin["blockers"][0]["evidence"]["audio_extract_status"] == "ok"
    assert "## Blockers" in Path(result["markdown_path"]).read_text(encoding="utf-8")


def test_asr_ab_compare_reports_whisperx_availability_blocker(tmp_path: Path) -> None:
    from video_knowledge_pipeline.asr_ab_compare import compare_asr_ab_sample

    plan = tmp_path / "asr-run-plan.json"
    plan.write_text(
        json.dumps(
            {
                "availability": {
                    "command_path": "",
                    "module": "whisperx",
                    "module_available": False,
                    "python_executable": "python.exe",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    alignment = tmp_path / "whisperx-alignment-run.json"
    alignment.write_text(json.dumps({"plan_path": str(plan)}, ensure_ascii=False), encoding="utf-8")
    run = tmp_path / "asr-ab-sample-run.json"
    run.write_text(
        json.dumps(
            {
                "workspace_dir": str(tmp_path),
                "sample_media_path": str(tmp_path / "sample.mp4"),
                "variants": [
                    {"key": "sensevoice_full_punc", "status": "ok", "metrics": {"segment_count": 12, "char_count": 1100, "punctuation_count": 90, "duration_seconds": 300}},
                    {"key": "whisperx_alignment", "status": "blocked", "alignment_run_json": str(alignment), "metrics": {}},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = compare_asr_ab_sample(run, write=False)

    whisperx = {row["key"]: row for row in result["variants"]}["whisperx_alignment"]
    assert whisperx["blockers"][0]["code"] == "whisperx_command_and_module_unavailable"
    assert whisperx["blockers"][0]["evidence"]["module"] == "whisperx"
    assert whisperx["blockers"][0]["evidence"]["module_available"] is False


def test_asr_ab_sample_plan_includes_whisperx_alignment_boundary(tmp_path: Path) -> None:
    from video_knowledge_pipeline.asr_ab_plan import plan_asr_ab_sample

    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")

    result = plan_asr_ab_sample(tmp_path / "workspace", media, write=False)

    rows = {row["key"]: row for row in result["variants"]}
    assert "whisperx_alignment" in rows
    assert rows["whisperx_alignment"]["role"] == "timestamp_speaker_alignment_evidence"
    assert rows["whisperx_alignment"]["operator_boundary"]["alignment_evidence_only"] is True
    assert result["operator_boundary"]["whisperx_alignment_is_timestamp_evidence_only"] is True


def test_asr_ab_compare_scores_reference_similarity_without_importing_reference(tmp_path: Path) -> None:
    from video_knowledge_pipeline.asr_ab_compare import compare_asr_ab_sample

    sensevoice = tmp_path / "sensevoice.json"
    dolphin = tmp_path / "dolphin.json"
    reference = tmp_path / "reference.json"
    sensevoice.write_text(json.dumps({"segments": [{"start": 0, "end": 10, "text": "明晚八点 o 我找一下保单"}]}, ensure_ascii=False), encoding="utf-8")
    dolphin.write_text(json.dumps({"segments": [{"start": 0, "end": 10, "text": "明晚八点 OK 我找一下保单"}]}, ensure_ascii=False), encoding="utf-8")
    reference.write_text(json.dumps({"segments": [{"start": 0, "end": 10, "text": "明晚八点 OK 我找一下我的保单"}]}, ensure_ascii=False), encoding="utf-8")
    run = tmp_path / "asr-ab-sample-run.json"
    run.write_text(
        json.dumps(
            {
                "workspace_dir": str(tmp_path),
                "sample_media_path": str(tmp_path / "sample.mp4"),
                "variants": [
                    {"key": "sensevoice_full_punc", "status": "ok", "normalized_json": str(sensevoice), "metrics": {"segment_count": 1, "char_count": 14, "punctuation_count": 0, "duration_seconds": 10}},
                    {"key": "dolphin", "status": "ok", "normalized_json": str(dolphin), "metrics": {"segment_count": 1, "char_count": 15, "punctuation_count": 0, "duration_seconds": 10}},
                    {"key": "whisperx_alignment", "status": "preview", "metrics": {}},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = compare_asr_ab_sample(run, reference_transcript=reference, write=False)

    rows = {row["key"]: row for row in result["variants"]}
    assert result["reference_role"] == "evaluation_only_not_correction_evidence"
    assert result["operator_boundary"]["does_not_import_reference_as_evidence"] is True
    assert result["best_reference_variant"] == "dolphin"
    assert rows["dolphin"]["reference_similarity"] > rows["sensevoice_full_punc"]["reference_similarity"]


def test_cli_asr_ab_compare_command_parse() -> None:
    parser = build_parser()
    args = parser.parse_args(["asr-ab-compare", "run.json", "--reference-transcript", "reference.json", "--start-seconds", "1", "--end-seconds", "2", "--no-write"])

    assert args.command == "asr-ab-compare"
    assert args.run_json == "run.json"
    assert args.reference_transcript == "reference.json"
    assert args.start_seconds == 1
    assert args.end_seconds == 2
    assert args.no_write is True




def test_asr_ab_sample_run_merges_selected_variant_with_previous_results(tmp_path: Path) -> None:
    from video_knowledge_pipeline.asr_ab_plan import plan_asr_ab_sample
    from video_knowledge_pipeline.asr_ab_run import run_asr_ab_sample

    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")
    plan = plan_asr_ab_sample(tmp_path / "workspace", media, write=True)
    run_path = Path(plan["artifacts"]["json"]).parent / "asr-ab-sample-run.json"
    run_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.asr_ab_sample_run.v1",
                "variants": [
                    {"key": "sensevoice_full_punc", "status": "ok", "metrics": {"segment_count": 3, "char_count": 100, "punctuation_count": 10, "duration_seconds": 30}}
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_asr_ab_sample(tmp_path / "workspace", plan_json=plan["artifacts"]["json"], variants=["openai_cloud_asr"], execute_cloud=False, write=True)

    rows = {row["key"]: row for row in result["variants"]}
    assert result["merged_previous_variants"] is True
    assert rows["sensevoice_full_punc"]["status"] == "ok"
    assert rows["openai_cloud_asr"]["status"] == "preview"


def test_asr_ab_sample_run_preview_does_not_execute_or_promote(tmp_path: Path) -> None:
    from video_knowledge_pipeline.asr_ab_run import run_asr_ab_sample

    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")

    result = run_asr_ab_sample(tmp_path / "workspace", media, variants=["sensevoice_basic", "openai_cloud_asr"], write=True)

    assert result["status"] == "preview"
    assert result["operator_boundary"]["does_not_promote_any_transcript"] is True
    assert result["operator_boundary"]["cloud_upload_sample_only_when_execute_cloud"] is False
    by_key = {row["key"]: row for row in result["variants"]}
    assert by_key["sensevoice_basic"]["execute"] is False
    assert by_key["openai_cloud_asr"]["execute"] is False
    assert by_key["openai_cloud_asr"]["operator_boundary"]["cloud_upload"] is False
    assert (tmp_path / "workspace" / "transcripts" / "asr-ab-sample" / "asr-ab-sample-run.json").exists()
    assert not (tmp_path / "workspace" / "corrected-transcript.json").exists()



def test_asr_ab_full_punc_blocks_when_punc_model_not_ready(tmp_path: Path, monkeypatch) -> None:
    from video_knowledge_pipeline import asr_ab_run
    from video_knowledge_pipeline.asr_ab_run import run_asr_ab_sample

    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")
    monkeypatch.setattr(asr_ab_run, "_resolve_local_model", lambda model: model)
    monkeypatch.delenv("LECTURE_ASR_ALLOW_MODEL_DOWNLOAD", raising=False)

    called = {"run_asr": False}

    def fake_run_asr_plan(*_args, **_kwargs):
        called["run_asr"] = True
        return {"status": "should_not_run"}

    monkeypatch.setattr(asr_ab_run, "run_asr_plan", fake_run_asr_plan)

    result = run_asr_ab_sample(
        tmp_path / "workspace",
        media,
        variants=["sensevoice_full_punc"],
        execute_local=True,
        write=False,
    )

    row = result["variants"][0]
    assert row["status"] == "asr_model_not_ready"
    assert row["reason"] == "adjunct_model_not_ready"
    assert row["model_ready"]["required"][0]["model"] == "ct-punc"
    assert called["run_asr"] is False



def test_asr_model_cache_status_and_prepare_require_download_permission(tmp_path: Path, monkeypatch) -> None:
    from video_knowledge_pipeline import asr_model_cache
    from video_knowledge_pipeline.asr_model_cache import asr_model_cache_status, prepare_asr_model_cache

    monkeypatch.setattr(asr_model_cache, "_resolve_local_model", lambda model: str(tmp_path / "cached" / model) if model != "ct-punc" else model)
    monkeypatch.setattr(asr_model_cache, "_local_model_candidates", lambda model: [tmp_path / "candidate" / model])
    monkeypatch.delenv("LECTURE_ASR_ALLOW_MODEL_DOWNLOAD", raising=False)
    (tmp_path / "cached" / "iic" / "SenseVoiceSmall").mkdir(parents=True)
    (tmp_path / "cached" / "fsmn-vad").mkdir(parents=True)

    status = asr_model_cache_status(tmp_path / "workspace", models=["iic/SenseVoiceSmall", "fsmn-vad", "ct-punc"], write=False)

    rows = {row["model"]: row for row in status["models"]}
    assert rows["iic/SenseVoiceSmall"]["ready"] is True
    assert rows["fsmn-vad"]["ready"] is True
    assert rows["ct-punc"]["ready"] is False
    assert status["ready"] is False
    assert status["source_policy"] == {
        "hub": "modelscope",
        "china_accessible": True,
        "uses_funasr_native_downloader": True,
        "arbitrary_download_url": False,
    }

    result = prepare_asr_model_cache(tmp_path / "workspace", models=["iic/SenseVoiceSmall", "fsmn-vad", "ct-punc"], execute=True, allow_download=False, write=False)

    assert result["status"] == "download_not_allowed"
    assert result["operator_boundary"]["requires_allow_download"] is True
    assert result["operator_boundary"]["network_access"] == "disabled"


def test_campp_model_cache_preview_uses_funasr_modelscope_alias_without_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from video_knowledge_pipeline import asr_model_cache
    from video_knowledge_pipeline.asr_model_cache import prepare_asr_model_cache

    monkeypatch.setenv(
        "LECTURE_ASR_PYTHON",
        str(tmp_path / "lecture-asr" / "python.exe"),
    )
    monkeypatch.setattr(asr_model_cache, "_resolve_local_model", lambda model: model)
    monkeypatch.setattr(
        asr_model_cache,
        "_local_model_candidates",
        lambda model: [tmp_path / "modelscope" / model],
    )

    def fail_if_executed(*args, **kwargs):
        raise AssertionError("preview must not launch a downloader subprocess")

    monkeypatch.setattr(asr_model_cache.subprocess, "run", fail_if_executed)

    result = prepare_asr_model_cache(
        tmp_path / "workspace",
        models=["cam++"],
        execute=False,
        allow_download=False,
        device="cuda",
        write=False,
    )

    row = result["before"]["models"][0]
    assert result["status"] == "preview"
    assert result["operator_boundary"]["network_access"] == "disabled"
    assert result["source_policy"]["hub"] == "modelscope"
    assert row["model"] == "cam++"
    assert row["official_model_ids"][0] == (
        "iic/speech_campplus_sv_zh-cn_16k-common"
    )
    assert result["command"][0] == str(
        tmp_path / "lecture-asr" / "python.exe"
    )
    assert result["command"][result["command"].index("--device") + 1] == "cuda"
    assert result["command"][result["command"].index("--spk-model") + 1] == "cam++"


def test_prepare_asr_model_cache_uses_requested_primary_model_and_creates_workspace(tmp_path: Path, monkeypatch) -> None:
    from video_knowledge_pipeline import asr_model_cache
    from video_knowledge_pipeline.asr_model_cache import prepare_asr_model_cache

    requested = "iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"
    workspace = tmp_path / "new-workspace"
    calls: list[dict] = []

    monkeypatch.setattr(asr_model_cache, "_resolve_local_model", lambda model: model)
    monkeypatch.setattr(asr_model_cache, "_local_model_candidates", lambda model: [tmp_path / "missing" / model])

    class Completed:
        returncode = 0
        stdout = "prepared"
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        assert workspace.exists()
        return Completed()

    monkeypatch.setattr(asr_model_cache.subprocess, "run", fake_run)

    result = prepare_asr_model_cache(
        workspace,
        models=[requested],
        execute=True,
        allow_download=True,
        write=False,
    )

    command = calls[0]["command"]
    assert command[command.index("--model") + 1] == requested
    assert calls[0]["cwd"] == str(workspace.resolve())
    assert result["status"] == "prepare_failed"

def test_cli_asr_model_cache_commands_parse() -> None:
    parser = build_parser()
    status = parser.parse_args(["asr-model-cache-status", "workspace", "--models", "ct-punc", "--no-write"])
    prepare = parser.parse_args(["prepare-asr-model-cache", "workspace", "--models", "ct-punc", "--execute", "--allow-download", "--device", "cpu"])

    assert status.command == "asr-model-cache-status"
    assert prepare.command == "prepare-asr-model-cache"
    assert prepare.execute is True
    assert prepare.allow_download is True


def _write_test_config(path: Path, *, asr_runtime: dict | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.config.v1",
                "services": {
                    "review_webui": {"type": "static_file", "entrypoint": "webui-bundle/review.html"},
                    "ebook_markdown_pipeline_http": {"host": "127.0.0.1", "port": 8765, "path": "/call"},
                    "openclaw_http": {"host": "127.0.0.1", "port": 8931, "path": "/call", "docker_host": "host.docker.internal"},
                    "mcp": {"transport": "stdio"},
                },
                "asr_runtime": asr_runtime or {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_plan_asr_run_reads_unified_runtime_profile(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")
    config_path = tmp_path / "video-knowledge-pipeline.json"
    _write_test_config(
        config_path,
        asr_runtime={
            "provider": "funasr_sensevoice",
            "model": "iic/SenseVoiceSmall",
            "device": "cpu",
            "punc_model": "ct-punc",
            "spk_model": "cam++",
            "enable_diarization": True,
            "merge_length_s": 24,
        },
    )
    monkeypatch.setenv("VIDEO_KNOWLEDGE_PIPELINE_CONFIG", str(config_path))

    import video_knowledge_pipeline.asr_runner as asr_runner

    monkeypatch.setattr(asr_runner, "_module_available_in_python", lambda module, python: True)
    monkeypatch.setattr(asr_runner, "_model_ready", lambda preset, model: {"model": model, "ready": True, "status": "ready"})
    speaker_model = tmp_path / "campp"
    speaker_model.mkdir()
    monkeypatch.setattr(
        asr_runner,
        "_resolve_local_model",
        lambda model: str(speaker_model) if model == "cam++" else model,
    )

    plan = plan_asr_run(tmp_path / "workspace", media, preset="sensevoice")

    assert plan["available"] is True
    assert plan["local_asr_device"] == "cpu"
    assert plan["asr_runtime_profile"]["device"] == "cpu"
    assert plan["full_mode"]["spk_model"] == "cam++"
    assert plan["full_mode"]["merge_length_s"] == 24
    assert "--spk-model" in plan["command"]
    assert "cam++" in plan["command"]
    assert plan["model_readiness"]["speaker"]["ready"] is True


def test_plan_asr_run_blocks_missing_requested_speaker_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")
    config_path = tmp_path / "video-knowledge-pipeline.json"
    _write_test_config(
        config_path,
        asr_runtime={
            "provider": "funasr_sensevoice",
            "model": "iic/SenseVoiceSmall",
            "device": "cpu",
            "punc_model": "ct-punc",
            "spk_model": "cam++",
            "enable_diarization": True,
        },
    )
    monkeypatch.setenv("VIDEO_KNOWLEDGE_PIPELINE_CONFIG", str(config_path))

    import video_knowledge_pipeline.asr_execution as asr_execution
    import video_knowledge_pipeline.asr_runner as asr_runner

    monkeypatch.setattr(asr_runner, "_module_available_in_python", lambda module, python: True)
    monkeypatch.setattr(
        asr_runner,
        "_model_ready",
        lambda preset, model: {"model": model, "ready": True, "status": "ready"},
    )
    monkeypatch.setattr(asr_runner, "_resolve_local_model", lambda model: model)

    plan = plan_asr_run(tmp_path / "workspace", media, preset="sensevoice")

    assert plan["available"] is False
    assert plan["availability"]["runtime_ready"] is True
    assert plan["availability"]["required_models_ready"] is False
    assert plan["availability"]["blockers"] == [
        "speaker_model_missing_or_not_downloaded:cam++"
    ]
    assert plan["model_readiness"]["speaker"] == {
        "model": "cam++",
        "resolved_model": "cam++",
        "required": True,
        "ready": False,
        "status": "missing_or_not_downloaded",
    }
    assert "--spk-model" in plan["command"]
    assert "cam++" in plan["command"]

    def fail_if_executed(*args, **kwargs):
        raise AssertionError("missing speaker model must block before subprocess execution")

    monkeypatch.setattr(asr_execution, "_run_command_with_cuda_oom_recovery", fail_if_executed)
    result = run_asr_plan(plan["plan_path"], execute=True)

    assert result["status"] == "blocked"
    assert "speaker_model_missing_or_not_downloaded:cam++" in result["stderr"]


def test_local_asr_service_plan_and_preview_are_localhost_gated(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")
    config_path = tmp_path / "video-knowledge-pipeline.json"
    _write_test_config(
        config_path,
        asr_runtime={
            "provider": "speaches_openai_compatible",
            "openai_compatible": {
                "base_url": "http://127.0.0.1:8000/v1",
                "model": "Systran/faster-whisper-large-v3",
                "timeout_seconds": 600,
            },
        },
    )
    monkeypatch.setenv("VIDEO_KNOWLEDGE_PIPELINE_CONFIG", str(config_path))

    plan = plan_local_asr_service_run(tmp_path / "workspace", media, language="zh")

    assert plan["schema"] == "video_knowledge_pipeline.local_asr_service_plan.v1"
    assert plan["provider"] == "speaches_openai_compatible"
    assert plan["local_service"] is True
    assert plan["upload_required"] is False
    assert plan["provider_config"]["model"] == "Systran/faster-whisper-large-v3"
    assert "api_key" not in json.dumps(plan, ensure_ascii=False)

    preview = run_local_asr_service_plan(plan["plan_path"], execute=False)
    assert preview["schema"] == "video_knowledge_pipeline.local_asr_service_run.v1"
    assert preview["status"] == "preview"
    assert preview["execute"] is False
    assert preview["local_service"] is True


def test_local_asr_service_blocks_remote_endpoint_without_allow_remote(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake video")
    config_path = tmp_path / "video-knowledge-pipeline.json"
    _write_test_config(
        config_path,
        asr_runtime={
            "provider": "custom_openai_compatible",
            "openai_compatible": {
                "base_url": "https://example.com/v1",
                "model": "remote-asr",
                "timeout_seconds": 600,
            },
        },
    )
    monkeypatch.setenv("VIDEO_KNOWLEDGE_PIPELINE_CONFIG", str(config_path))

    plan = plan_local_asr_service_run(tmp_path / "workspace", media, language="zh")
    blocked = run_local_asr_service_plan(plan["plan_path"], execute=True)

    assert plan["local_service"] is False
    assert plan["upload_required"] is True
    assert blocked["status"] == "blocked_remote_asr_service"


@pytest.fixture
def alignment_plan_inputs(tmp_path: Path, monkeypatch):
    import video_knowledge_pipeline.asr_runner as asr_runner

    media = tmp_path / "synthetic.wav"
    media.write_bytes(b"synthetic audio fixture, never decoded")
    transcript = tmp_path / "canonical.json"
    write_json(transcript, {"segments": [{"start": 0, "end": 3, "text": "保留正文"}]})
    monkeypatch.setattr(asr_runner, "asr_runtime_profile", lambda: {})
    monkeypatch.setattr(asr_runner, "_resolve_python_executable", lambda: "fixture-python")
    monkeypatch.setattr(asr_runner, "default_local_asr_device", lambda *_: "cpu")
    monkeypatch.setattr(asr_runner, "_module_available_in_python", lambda *_: True)
    monkeypatch.setattr(asr_runner, "_resolve_command_path", lambda *_: "")
    monkeypatch.setattr(asr_runner, "_local_qwen_aligner_path", lambda: None)
    monkeypatch.setattr(asr_runner, "_model_ready", lambda **_: {"ready": True})
    return media, transcript


@pytest.mark.parametrize("chunk_seconds", [None, 30, 60])
def test_alignment_chunk_plan_preserves_default_and_forwards_explicit_chunks(
    tmp_path: Path, alignment_plan_inputs, chunk_seconds
):
    media, transcript = alignment_plan_inputs
    options = {} if chunk_seconds is None else {"alignment_chunk_seconds": chunk_seconds}
    plan = plan_asr_run(tmp_path / "workspace", media, preset="qwen3-forced-aligner",
                        transcript_path=transcript, **options)
    expected = 300 if chunk_seconds is None else chunk_seconds
    assert plan["alignment_chunk_seconds"] == expected
    assert plan["command"][plan["command"].index("--chunk-seconds") + 1] == str(expected)
    assert plan["asr_mode"] == "alignment"
    saved = json.loads(Path(plan["plan_path"]).read_text(encoding="utf-8"))
    assert saved["alignment_chunk_seconds"] == expected
    assert saved["command"] == plan["command"]


def test_alignment_chunk_change_gets_a_separate_plan_and_output(
    tmp_path: Path, alignment_plan_inputs
):
    media, transcript = alignment_plan_inputs
    original = plan_asr_run(tmp_path / "workspace", media, preset="qwen3-forced-aligner",
                            transcript_path=transcript)
    old_output = Path(original["expected_output_json"])
    old_output.write_text('{"retained":"300-second output"}', encoding="utf-8")
    changed = plan_asr_run(tmp_path / "workspace", media, preset="qwen3-forced-aligner",
                           transcript_path=transcript, alignment_chunk_seconds=60)
    assert changed["run_id"] != original["run_id"]
    assert changed["plan_path"] != original["plan_path"]
    assert changed["expected_output_json"] != original["expected_output_json"]
    assert not Path(changed["expected_output_json"]).exists()
    assert json.loads(old_output.read_text(encoding="utf-8"))["retained"] == "300-second output"
    assert json.loads(Path(original["plan_path"]).read_text(encoding="utf-8"))["alignment_chunk_seconds"] == 300


@pytest.mark.parametrize("invalid", [0, 29, True, 60.5, float("inf"), float("nan"), "60", None])
def test_alignment_chunk_plan_rejects_invalid_values_before_writing(tmp_path: Path, invalid):
    workspace = tmp_path / "workspace"
    with pytest.raises(ValueError, match="alignment_chunk_seconds must be an integer of at least 30"):
        plan_asr_run(workspace, tmp_path / "absent.wav", preset="qwen3-forced-aligner",
                     alignment_chunk_seconds=invalid)
    assert not workspace.exists()


def test_alignment_chunk_plan_rejects_non_aligner_override_but_keeps_default(
    tmp_path: Path, alignment_plan_inputs
):
    media, _ = alignment_plan_inputs
    with pytest.raises(ValueError, match="only supported by qwen3-forced-aligner"):
        plan_asr_run(tmp_path / "rejected", media, preset="sensevoice", alignment_chunk_seconds=60)
    assert not (tmp_path / "rejected").exists()
    plan = plan_asr_run(tmp_path / "default", media, preset="sensevoice")
    assert plan["alignment_chunk_seconds"] is None
    assert plan["command"][plan["command"].index("--chunk-seconds") + 1] == "300"


@pytest.mark.parametrize("chunk_seconds,expected", [(None, 300), (60, 60)])
def test_alignment_chunk_cli_forwards_default_and_explicit_values(monkeypatch, chunk_seconds, expected):
    import video_knowledge_pipeline.cli as cli

    captured = []
    monkeypatch.setattr(cli, "plan_asr_run", lambda *_, **kwargs: captured.append(kwargs) or {"status": "planned"})
    argv = ["plan-asr", "fixture-workspace", "fixture.wav", "--preset", "qwen3-forced-aligner"]
    if chunk_seconds is not None:
        argv.extend(["--alignment-chunk-seconds", str(chunk_seconds)])
    assert cli_main(argv) == 0
    assert captured[0]["alignment_chunk_seconds"] == expected


@pytest.mark.parametrize("chunk_seconds", [60, 300])
def test_alignment_chunk_cli_rejects_explicit_flag_for_other_presets(monkeypatch, chunk_seconds):
    import video_knowledge_pipeline.cli as cli

    monkeypatch.setattr(cli, "plan_asr_run", lambda *a, **k: pytest.fail("unsupported option reached planning"))
    with pytest.raises(ValueError, match="only supported by qwen3-forced-aligner"):
        cli_main(["plan-asr", "fixture-workspace", "fixture.wav", "--preset", "sensevoice",
                  "--alignment-chunk-seconds", str(chunk_seconds)])


@pytest.mark.parametrize("invalid", ["60.5", "not-an-integer"])
def test_alignment_chunk_cli_rejects_non_integer_flags(invalid):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["plan-asr", "fixture-workspace", "fixture.wav",
                                  "--alignment-chunk-seconds", invalid])
    assert exc.value.code == 2
