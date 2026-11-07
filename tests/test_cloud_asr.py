from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from video_knowledge_pipeline.cloud_asr import plan_cloud_asr_run, run_cloud_asr_plan
from video_knowledge_pipeline.online_model_gateway import asr_transcriptions_url


def _case_root(name: str) -> Path:
    root = Path("semantic-refresh-smoke-run") / f"{name}-{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root




def test_asr_transcriptions_url_respects_openai_compatible_api_base() -> None:
    assert asr_transcriptions_url({"base_url": "https://api.openai.com/v1"}) == "https://api.openai.com/v1/audio/transcriptions"
    assert asr_transcriptions_url({"base_url": "https://api.example.com"}) == "https://api.example.com/v1/audio/transcriptions"
    assert asr_transcriptions_url({"base_url": "https://ark.cn-beijing.volces.com/api/coding/v3"}) == "https://ark.cn-beijing.volces.com/api/coding/v3/audio/transcriptions"
    assert asr_transcriptions_url({"base_url": "https://api.example.com/v1/audio/transcriptions"}) == "https://api.example.com/v1/audio/transcriptions"

def test_plan_cloud_asr_redacts_key_and_defaults_to_preview() -> None:
    root = _case_root("cloud-asr-plan")
    media = root / "lesson.mp4"
    media.write_bytes(b"fake")

    plan = plan_cloud_asr_run(
        root,
        media,
        provider_config={"provider": "openai_compatible_asr", "api_key": "actual-api-key", "base_url": "https://api.example.com/v1"},
    )

    assert plan["preset"] == "cloud-asr"
    assert plan["execute"] is False
    assert plan["default_upload"] is False
    assert plan["upload_required"] is True
    assert plan["provider_config"]["model"] == "gpt-4o-transcribe"
    assert "api_key" not in plan["provider_config"]
    assert "api_key" not in str(plan["request_plan"])
    assert "actual-api-key" not in str(plan)
    assert Path(plan["plan_path"]).exists()


def test_run_cloud_asr_plan_preview_does_not_upload() -> None:
    root = _case_root("cloud-asr-preview")
    media = root / "lesson.mp4"
    media.write_bytes(b"fake")
    plan = plan_cloud_asr_run(root, media)

    result = run_cloud_asr_plan(plan["plan_path"], execute=False)

    assert result["status"] == "preview"
    assert result["operator_boundary"]["will_upload_audio_only_with_execute"] is False
    assert result["cloud_call"] == {}


def test_local_proxy_cloud_asr_preserves_raw_provider_output(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "cloud-asr-proxy"
    root.mkdir()
    media = root / "lesson.wav"
    media.write_bytes(b"fake-wave")
    provider = {
        "provider": "openai_compatible_asr",
        "adapter_backend": "proxy",
        "base_url": "http://127.0.0.1:8776/v1",
        "model": "local-asr",
        "location": "local",
        "execution_location": "local",
        "route_id": "pool-local-asr",
        "route_revision": "b" * 64,
    }
    plan = plan_cloud_asr_run(root, media, provider_config=provider, prompt="保险 领航计划")

    import video_knowledge_pipeline.online_model_gateway as online_gateway

    calls: list[dict[str, object]] = []

    def keep_explicit(model_type: str, explicit: dict[str, object] | None = None, **kwargs: object) -> dict[str, object]:
        return dict(explicit or {})

    def fake_runtime(task: str, **kwargs: object) -> dict[str, object]:
        calls.append({"task": task, **kwargs})
        return {
            "ok": True,
            "status": "completed",
            "content": "你好保险",
            "raw_output": {
                "text": "你好保险",
                "segments": [{"start": 0.0, "end": 1.2, "text": "你好保险"}],
                "provider_trace": "raw-kept",
            },
        }

    monkeypatch.setattr(online_gateway, "resolve_model_api_provider_config", keep_explicit)
    monkeypatch.setattr(online_gateway, "model_runtime_request", fake_runtime)
    result = run_cloud_asr_plan(plan["plan_path"], execute=True)

    raw = json.loads(Path(result["raw_output_json"]).read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert calls[0]["task"] == "asr"
    assert calls[0]["execution_location"] == "local"
    assert calls[0]["prompt"] == "保险 领航计划"
    assert raw["provider_trace"] == "raw-kept"
    assert raw["segments"][0]["text"] == "你好保险"
    assert result["normalized"]["segment_count"] == 1

def test_run_cloud_asr_plan_execute_normalizes_openai_segments(monkeypatch) -> None:
    root = _case_root("cloud-asr-execute")
    media = root / "lesson.mp4"
    media.write_bytes(b"fake")
    plan = plan_cloud_asr_run(root, media)

    def fake_online_model_api_call(*args, **kwargs):
        assert args[0] == "asr"
        assert kwargs["execute"] is True
        return {
            "ok": True,
            "status": "ok",
            "content": "你好世界",
            "raw_response": {"segments": [{"start": 0.0, "end": 1.2, "text": "你好世界"}]},
            "request_plan": {"model_type": "asr"},
        }

    monkeypatch.setattr("video_knowledge_pipeline.cloud_asr.online_model_api_call", fake_online_model_api_call)

    result = run_cloud_asr_plan(plan["plan_path"], execute=True)

    assert result["status"] == "ok"
    assert Path(result["raw_output_json"]).exists()
    assert result["normalized"]["segment_count"] == 1
    assert Path(result["normalized"]["json_path"]).exists()
