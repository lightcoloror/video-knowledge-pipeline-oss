from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from video_knowledge_pipeline.dashscope_filetrans_adapter import (
    UPSTREAM_COMMIT,
    build_dashscope_filetrans_plan,
    call_dashscope_filetrans_asr,
)
from video_knowledge_pipeline.model_runtime_client import (
    authorise_consented_remote_runtime,
)
from video_knowledge_pipeline.model_provider_catalog import provider_preset
from video_knowledge_pipeline.online_model_gateway import online_model_api_call


ROUTE_REVISION = "a" * 64


def _provider() -> dict[str, object]:
    return {
        "provider": "dashscope_filetrans",
        "api_key": "fixture-secret",
        "base_url": "https://dashscope.aliyuncs.com",
        "model": "qwen-audio-3.0-asr-flash-filetrans",
        "route_id": "fixture-asr-route",
        "route_revision": ROUTE_REVISION,
        "consent_id": "fixture-consent",
        "provider_options": {
            "language": "yue",
            "speaker_diarization": True,
            "poll_interval_seconds": 1,
            "poll_timeout_seconds": 60,
        },
    }


def _ready_status(tmp_path: Path) -> dict[str, object]:
    return {
        "ready": True,
        "source_root": str(tmp_path),
        "actual_commit": UPSTREAM_COMMIT,
        "expected_commit": UPSTREAM_COMMIT,
        "error": "",
    }


def test_catalog_declares_async_filetrans_as_asr_only() -> None:
    preset = provider_preset("dashscope_filetrans")

    assert preset["default_model"] == "qwen-audio-3.0-asr-flash-filetrans"
    assert preset["supported_capabilities"] == ["asr"]
    assert preset["default_base_url"] == "https://dashscope.aliyuncs.com"
    assert "text" not in preset["supported_capabilities"]


def test_plan_uses_fixed_upstream_cli_without_secret_in_command(tmp_path: Path) -> None:
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"RIFF-fixture")
    plan = build_dashscope_filetrans_plan(
        provider_config=_provider(),
        audio_path=audio,
        output_srt=tmp_path / "result.srt",
        prompt="保险 医疗",
        source_root=tmp_path,
    )

    assert plan["provider"] == "dashscope_filetrans"
    assert plan["automatic_fallback"] is False
    assert plan["api_key_in_command"] is False
    assert "fixture-secret" not in plan["command"]
    assert plan["environment"]["DASHSCOPE_DEFAULT_LANGUAGE"] == "yue"


def test_adapter_refuses_execution_without_broker_reservation(tmp_path: Path) -> None:
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"RIFF-fixture")
    runner_called = False

    def runner(*args, **kwargs):  # pragma: no cover - must stay unreachable.
        nonlocal runner_called
        runner_called = True
        raise AssertionError("runner must not be called")

    result = call_dashscope_filetrans_asr(
        provider_config=_provider(),
        audio_path=str(audio),
        source_root=tmp_path,
        _runner=runner,
        _source_status=_ready_status(tmp_path),
    )

    assert result["ok"] is False
    assert result["status"] == "consent_required"
    assert result["remote_requests_made"] is False
    assert runner_called is False


def test_adapter_consumes_exact_reservation_and_normalises_output(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"RIFF-fixture")
    calls = 0

    def runner(command, **kwargs):
        nonlocal calls
        calls += 1
        output = Path(command[command.index("--output") + 1])
        output.with_suffix(".mosp").write_text(
            json.dumps(
                {
                    "schema": "moys_asr_workflow.project.v1",
                    "language": "yue",
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 1.2,
                            "text": "你好。",
                            "speaker": "speaker-1",
                            "items": [],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        assert kwargs["env"]["DASHSCOPE_API_KEY"] == "fixture-secret"
        assert "fixture-secret" not in command
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with authorise_consented_remote_runtime(
        consent_id="fixture-consent",
        route_revision=ROUTE_REVISION,
        max_calls=1,
    ):
        result = call_dashscope_filetrans_asr(
            provider_config=_provider(),
            audio_path=str(audio),
            source_root=tmp_path,
            _runner=runner,
            _source_status=_ready_status(tmp_path),
        )
        exhausted = call_dashscope_filetrans_asr(
            provider_config=_provider(),
            audio_path=str(audio),
            source_root=tmp_path,
            _runner=runner,
            _source_status=_ready_status(tmp_path),
        )

    assert result["ok"] is True
    assert result["status"] == "completed"
    assert result["content"] == "你好。"
    assert result["raw_response"]["segments"][0]["speaker"] == "speaker-1"
    assert result["remote_requests_made"] is True
    assert exhausted["status"] == "consent_required"
    assert "exhausted" in exhausted["error"]
    assert calls == 1


def test_online_gateway_preview_declares_async_filetrans_and_execute_is_blocked(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"RIFF-fixture")
    provider = _provider()

    preview = online_model_api_call(
        "asr",
        provider_config=provider,
        audio_path=str(audio),
        execute=False,
        write=False,
    )
    blocked = online_model_api_call(
        "asr",
        provider_config=provider,
        audio_path=str(audio),
        execute=True,
        write=False,
    )

    assert preview["request_plan"]["interface"] == "dashscope_async_filetrans"
    assert preview["request_plan"]["adapter_backend"] == "moys_asr_workflow_fixed_cli"
    assert blocked["status"] == "consent_required"
    assert blocked["remote_requests_made"] is False


def test_timeout_is_classified_without_fallback(tmp_path: Path) -> None:
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"RIFF-fixture")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="fixture", timeout=1)

    with authorise_consented_remote_runtime(
        consent_id="fixture-consent",
        route_revision=ROUTE_REVISION,
        max_calls=1,
    ):
        result = call_dashscope_filetrans_asr(
            provider_config=_provider(),
            audio_path=str(audio),
            source_root=tmp_path,
            _runner=timeout,
            _source_status=_ready_status(tmp_path),
        )

    assert result["status"] == "timeout"
    assert result["automatic_fallback"] is False
    assert result["remote_requests_made"] is True
