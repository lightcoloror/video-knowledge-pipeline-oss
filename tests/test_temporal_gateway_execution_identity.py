from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import video_knowledge_pipeline.temporal_visual_analyzer as temporal
import video_knowledge_pipeline.online_model_gateway as online_gateway
import video_knowledge_pipeline.vision_execution_route as execution_route
import video_knowledge_pipeline.vision_preflight as preflight
from video_knowledge_pipeline.vision_export_consent import create_vision_export_consent


GEMINI_ROUTE = {
    "provider": "gemini",
    "base_url": "http://127.0.0.1:18776/v1",
    "model": "gemini-3.6-flash",
    "timeout_seconds": 60,
    "adapter_backend": "proxy",
    "execution_location": "remote",
    "route_id": "pool-online-production-existing-apis-google-gemini-3-6-flash-vision",
    "route_revision": "53b54018543ef19ae2cc91f64cc07b85d356eb590b7c2a952640245a47f100a0",
    "virtual_model": "vkp-remote-vision-gemini-identity-test",
    "profile_id": "google-gemini-3-6-flash",
    "credential_status": "ready",
    "credential_ready": True,
    "gateway_configured": True,
    "gateway_ready": True,
    "gateway_status": "ready",
}


def _route_profile(task: str) -> dict[str, object]:
    return {
        "schema": "video_knowledge_pipeline.vision_gateway_profile.v1",
        "task": task,
        "status": "gateway_ready",
        "route_configured": True,
        "gateway_configured": True,
        "gateway_ready": True,
        "credential_ready": True,
        "provider_config": dict(GEMINI_ROUTE),
        "remote_requests_made": False,
        "secret_values_accessed": False,
    }


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    assets = root / "assets"
    assets.mkdir(parents=True)
    (assets / "frame-1.jpg").write_bytes(b"fixture-frame-1")
    (assets / "frame-2.jpg").write_bytes(b"fixture-frame-2")
    (root / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}), encoding="utf-8")
    (root / "timeline.json").write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "start": 0,
                    "end": 8,
                    "text": "演示界面发生变化。",
                    "visual_route": "temporal_sequence",
                    "frame_paths": ["assets/frame-1.jpg"],
                    "temporal_frame_paths": ["assets/frame-1.jpg", "assets/frame-2.jpg"],
                    "quality_issues": ["temporal_sequence_without_analysis"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root


def test_temporal_execute_uses_same_gateway_identity_as_preflight_and_consent(monkeypatch, tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    # This case tests the configured gateway without a competing local endpoint.
    monkeypatch.setattr(execution_route, "probe_lm_studio_endpoint", lambda: {"listening": False})
    monkeypatch.setattr(execution_route, "resolve_route_based_vision_gateway_profile", _route_profile)
    monkeypatch.setattr(preflight, "resolve_route_based_vision_gateway_profile", _route_profile)

    readiness = preflight.vision_execution_preflight(
        root,
        include_semantic=False,
        include_temporal=True,
        temporal_indexes=[1],
        temporal_limit=1,
        write=False,
    )
    assert readiness["execution_profile"]["provider_config_source"] == "route_based_gateway"
    assert readiness["provider"]["route_id"] == GEMINI_ROUTE["route_id"]

    consent = create_vision_export_consent(
        root,
        temporal_indexes=[1],
        image_max_edge=320,
        image_jpeg_quality=44,
        confirm_data_export=True,
    )
    assert consent["provider"]["route_id"] == GEMINI_ROUTE["route_id"]
    assert consent["provider"]["route_revision"] == GEMINI_ROUTE["route_revision"]

    captured: dict[str, object] = {}

    def fake_gate(_root, _execute, cfg, **kwargs):
        captured["gate_provider"] = dict(cfg)
        captured["gate_kwargs"] = dict(kwargs)
        return {}, {"status": "confirmed", "confirmed": True, "export_consent": {"valid": True, "consent_id": consent["consent_id"]}}

    def fake_probe(paths, **kwargs):
        captured["probe_kwargs"] = dict(kwargs)
        return {"image_paths": list(paths)}

    def fake_model_call(**kwargs):
        captured["model_provider"] = dict(kwargs["provider_config"])
        captured["model_allowed_roots"] = list(kwargs["call_kwargs"]["allowed_roots"])
        return {
            "ok": True,
            "error": "",
            "content": json.dumps(
                {
                    "event_sequence": ["画面从第一张切换到第二张"],
                    "state_changes": ["界面状态已变化"],
                    "evidence_frame_paths": ["assets/frame-1.jpg", "assets/frame-2.jpg"],
                },
                ensure_ascii=False,
            ),
            "attempts": [],
            "attempt_count": 1,
        }

    monkeypatch.setattr(temporal, "_execution_control", fake_gate)
    monkeypatch.setattr(temporal, "prepare_image_probe", fake_probe)
    monkeypatch.setattr(temporal, "call_vision_model_with_retries", fake_model_call)

    result = temporal.run_temporal_visual_analysis(
        root,
        execute=True,
        limit=1,
        indexes=[1],
        execution_actor="agent",
        export_consent=consent["artifacts"]["consent_json"],
    )

    for key in ("provider", "model", "base_url", "route_id", "route_revision", "virtual_model", "profile_id"):
        assert captured["gate_provider"][key] == GEMINI_ROUTE[key]
        assert captured["model_provider"][key] == GEMINI_ROUTE[key]
    assert captured["probe_kwargs"]["max_edge"] == 320
    assert captured["model_allowed_roots"] == [str(root)]
    assert captured["probe_kwargs"]["jpeg_quality"] == 44
    assert captured["gate_kwargs"]["image_probe_max_edge"] == 320
    assert captured["gate_kwargs"]["image_probe_jpeg_quality"] == 44
    assert result["summary"]["provider"]["route_id"] == GEMINI_ROUTE["route_id"]
    assert result["summary"]["provider"]["route_revision"] == GEMINI_ROUTE["route_revision"]
    assert result["summary"]["provider"]["provider_config_source"] == "route_based_gateway"
    assert result["summary"]["image_probe_max_edge"] == 320
    assert result["summary"]["image_probe_jpeg_quality"] == 44


def test_proxy_runtime_receives_only_caller_scoped_allowed_roots(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "probe.jpg"
    image.write_bytes(b"probe")
    captured: dict[str, object] = {}

    monkeypatch.setattr(online_gateway, "resolve_model_api_provider_config", lambda _task, config: dict(config))

    def fake_runtime(task, **kwargs):
        captured["task"] = task
        captured["allowed_roots"] = list(kwargs["allowed_roots"])
        return {"ok": True, "status": "completed", "content": "{}", "raw_output": {}, "error": ""}

    monkeypatch.setattr(online_gateway, "model_runtime_request", fake_runtime)
    result = online_gateway.online_model_api_call(
        "temporal_sequence",
        provider_config=GEMINI_ROUTE,
        image_paths=[str(image)],
        allowed_roots=[tmp_path],
        execute=True,
        write=False,
    )

    assert result["ok"] is True
    assert captured["task"] == "temporal_sequence"


def test_temporal_proxy_call_enters_matching_broker_reservation(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    @contextmanager
    def fake_reservation(**kwargs):
        captured["reservation"] = dict(kwargs)
        yield

    def fake_direct_model(**kwargs):
        captured["provider"] = dict(kwargs["provider_config"])
        captured["allowed_roots"] = list(kwargs["allowed_roots"])
        return {"ok": True, "content": "{}"}

    config = {**GEMINI_ROUTE, "consent_id": "consent-temporal-identity"}
    monkeypatch.setattr(temporal, "authorise_consented_remote_runtime", fake_reservation)
    monkeypatch.setattr(temporal, "call_vision_model", fake_direct_model)
    result = temporal.call_vision_model_with_broker_reservation(
        provider_config=config,
        prompt="fixture",
        image_paths=[str(tmp_path / "probe.jpg")],
        allowed_roots=[str(tmp_path)],
    )

    assert result["ok"] is True
    assert captured["reservation"] == {
        "consent_id": "consent-temporal-identity",
        "route_revision": GEMINI_ROUTE["route_revision"],
        "max_calls": 1,
    }
    assert captured["allowed_roots"] == [str(tmp_path)]
