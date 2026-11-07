from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline import vision_gateway_profile
from video_knowledge_pipeline import vision_preflight
from video_knowledge_pipeline.vision_gateway_readiness import route_based_gateway_provider_test


ROUTE_ID = "pool-online-production-existing-apis-google-gemini-3-6-flash-vision"
ROUTE_REVISION = "53b54018543ef19ae2cc91f64cc07b85d356eb590b7c2a952640245a47f100a0"


def _provider_config() -> dict[str, object]:
    return {
        "provider": "gemini",
        "base_url": "https://generativelanguage.googleapis.com",
        "model": "gemini-3.6-flash",
        "timeout_seconds": 60,
        "adapter_backend": "proxy",
        "execution_location": "remote",
        "route_id": ROUTE_ID,
        "route_revision": ROUTE_REVISION,
        "virtual_model": "vkp-remote-vision-example",
        "profile_id": "google-gemini-3-6-flash",
        "credential_status": "ready",
        "credential_ready": True,
        "gateway_configured": True,
        "gateway_ready": True,
        "gateway_status": "ready",
    }


def _profile(task: str, *, configured: bool = True) -> dict[str, object]:
    config = _provider_config() if configured else {}
    return {
        "schema": "video_knowledge_pipeline.vision_gateway_profile.v1",
        "task": task,
        "status": "gateway_ready" if configured else "route_missing",
        "route_configured": configured,
        "provider_config": config,
        "gateway_configured": configured,
        "gateway_ready": configured,
        "credential_ready": configured,
        "remote_requests_made": False,
    }


def test_route_profile_uses_dpapi_readiness_without_reading_key(monkeypatch) -> None:
    monkeypatch.setattr(
        vision_gateway_profile,
        "resolve_model_api_provider_config",
        lambda *args, **kwargs: {
            **_provider_config(),
            "api_key": "",
        },
    )
    monkeypatch.setattr(
        vision_gateway_profile,
        "public_model_api_settings_status",
        lambda *args, **kwargs: {
            "profiles": [{"id": "google-gemini-3-6-flash", "credential_status": "ready"}]
        },
    )
    monkeypatch.setattr(
        vision_gateway_profile,
        "model_gateway_runtime_readiness",
        lambda **kwargs: {"ready": True, "status": "ready", "gateway": {"host": "127.0.0.1", "port": 18776}},
    )

    result = vision_gateway_profile.resolve_route_based_vision_gateway_profile("temporal_sequence")

    assert result["status"] == "gateway_ready"
    assert result["credential_ready"] is True
    assert result["gateway_ready"] is True
    assert result["provider_config"]["api_key"] == ""
    assert result["remote_requests_made"] is False
    assert result["secret_values_accessed"] is False


def test_gateway_readiness_is_not_a_provider_smoke_call() -> None:
    report = route_based_gateway_provider_test(_provider_config(), task="temporal_sequence")

    assert report["status"] == "gateway_ready"
    assert report["safe_to_execute"] is True
    assert report["provider"]["api_key_required"] is False
    assert report["provider"]["api_key_configured"] is True
    assert report["remote_requests_made"] is False
    assert report["checks"][0]["name"] == "gateway_route_readiness"


def test_preflight_prefers_route_gateway_and_blocks_missing_task_without_fallback(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "frame.jpg").write_bytes(b"fixture")
    (tmp_path / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "visual_route": "temporal_sequence",
                    "temporal_frame_paths": ["frame.jpg", "frame.jpg"],
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        vision_preflight,
        "resolve_route_based_vision_gateway_profile",
        lambda task: _profile(task, configured=task == "temporal_sequence"),
    )

    report = vision_preflight.vision_execution_preflight(
        tmp_path,
        include_semantic=True,
        include_temporal=True,
        check_provider=False,
        write=False,
    )

    assert report["execution_profile"]["provider_config_source"] == "route_based_gateway"
    assert report["provider"]["gateway_configured"] is True
    assert report["provider"]["gateway_ready"] is True
    assert report["provider"]["api_key_configured"] is True
    assert report["provider"]["route_id"] == ROUTE_ID
    assert report["provider_health"]["status"] == "gateway_ready"
    assert any(item["key"] == "gateway_route_missing" for item in report["blockers"])
    assert not any(item["key"] == "missing_api_key" for item in report["blockers"])
