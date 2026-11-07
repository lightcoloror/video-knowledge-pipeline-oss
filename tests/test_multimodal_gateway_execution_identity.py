from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import video_knowledge_pipeline.multimodal_frame_analyzer as multimodal


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


def test_semantic_proxy_call_enters_matching_broker_reservation(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    @contextmanager
    def fake_reservation(**kwargs):
        captured["reservation"] = dict(kwargs)
        yield

    def fake_direct_model(**kwargs):
        captured["provider"] = dict(kwargs["provider_config"])
        captured["allowed_roots"] = list(kwargs["allowed_roots"])
        return {"ok": True, "content": "{}"}

    monkeypatch.setattr(multimodal, "authorise_consented_remote_runtime", fake_reservation)
    monkeypatch.setattr(multimodal, "call_vision_model", fake_direct_model)

    config = {**GEMINI_ROUTE, "consent_id": "consent-semantic-identity"}
    result = multimodal.call_vision_model_with_broker_reservation(
        provider_config=config,
        prompt="fixture",
        image_paths=[str(tmp_path / "probe.jpg")],
        allowed_roots=[str(tmp_path)],
    )

    assert result["ok"] is True
    assert captured["reservation"] == {
        "consent_id": "consent-semantic-identity",
        "route_revision": GEMINI_ROUTE["route_revision"],
        "max_calls": 1,
    }
    assert captured["provider"]["route_id"] == GEMINI_ROUTE["route_id"]
    assert captured["allowed_roots"] == [str(tmp_path)]


def test_semantic_execute_keeps_gateway_route_identity_and_consent_image_limits(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    assets = root / "assets"
    assets.mkdir(parents=True)
    (assets / "frame.jpg").write_bytes(b"fixture-frame")
    (root / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}), encoding="utf-8")
    (root / "timeline.json").write_text(
        json.dumps(
            [{"start": 0, "end": 8, "transcript": "讲师展示关键画面", "visual_route": "semantic_frame", "frame_paths": ["assets/frame.jpg"]}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        multimodal,
        "resolve_vision_task_execution_route",
        lambda _task, provider_config=None: {
            "status": "gateway_ready",
            "provider_config_source": "route_based_gateway",
            "provider_config": dict(GEMINI_ROUTE),
            "legacy_fallback_blocked": False,
        },
    )
    monkeypatch.setattr(
        multimodal,
        "resolve_vision_execution_profile",
        lambda *, provider_config, multimodal_limit: {"provider_config": dict(provider_config), "multimodal_limit": multimodal_limit or 1, "frame_count": 1},
    )
    monkeypatch.setattr(multimodal, "resolve_provider_config", lambda config: dict(config))

    def fake_gate(_root, _execute, cfg, **kwargs):
        captured["gate_provider"] = dict(cfg)
        captured["gate_kwargs"] = dict(kwargs)
        return {}, {
            "status": "confirmed",
            "confirmed": True,
            "export_consent": {"valid": True, "consent_id": "consent-semantic-route"},
        }

    def fake_limits(_path):
        return {"image_max_edge": 320, "image_jpeg_quality": 44}

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
                {"objects": ["投影片"], "actions": ["展示"], "confidence": 0.9, "evidence_frame_paths": ["assets/frame.jpg"]},
                ensure_ascii=False,
            ),
            "attempts": [],
            "attempt_count": 1,
        }

    monkeypatch.setattr(multimodal, "_execution_control", fake_gate)
    monkeypatch.setattr(multimodal, "vision_export_consent_image_limits", fake_limits)
    monkeypatch.setattr(multimodal, "prepare_image_probe", fake_probe)
    monkeypatch.setattr(multimodal, "call_vision_model_with_retries", fake_model_call)

    result = multimodal.run_multimodal_frame_analysis(
        root,
        execute=True,
        limit=1,
        indexes=[1],
        execution_actor="agent",
        export_consent=root / "fixture-consent.json",
    )

    for key in ("provider", "model", "base_url", "route_id", "route_revision", "virtual_model", "profile_id"):
        assert captured["gate_provider"][key] == GEMINI_ROUTE[key]
        assert captured["model_provider"][key] == GEMINI_ROUTE[key]
    assert captured["model_provider"]["consent_id"] == "consent-semantic-route"
    assert captured["model_allowed_roots"] == [str(root)]
    assert captured["probe_kwargs"] == {"output_dir": root / "vision-analysis-image-probes" / "semantic" / "1", "max_edge": 320, "jpeg_quality": 44}
    assert captured["gate_kwargs"]["image_probe_max_edge"] == 320
    assert captured["gate_kwargs"]["image_probe_jpeg_quality"] == 44
    assert result["summary"]["provider"]["route_id"] == GEMINI_ROUTE["route_id"]
    assert result["summary"]["provider"]["provider_config_source"] == "route_based_gateway"


def test_semantic_batch_stops_after_terminal_provider_error(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    assets = root / "assets"
    assets.mkdir(parents=True)
    for name in ("one.jpg", "two.jpg"):
        (assets / name).write_bytes(b"fixture-frame")
    (root / "manifest.json").write_text(json.dumps({"schema": "lecture_webui_bundle.v1"}), encoding="utf-8")
    (root / "timeline.json").write_text(
        json.dumps(
            [
                {"start": 0, "end": 8, "visual_route": "semantic_frame", "frame_paths": ["assets/one.jpg"]},
                {"start": 8, "end": 16, "visual_route": "semantic_frame", "frame_paths": ["assets/two.jpg"]},
            ]
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        multimodal,
        "resolve_vision_task_execution_route",
        lambda _task, provider_config=None: {
            "status": "gateway_ready",
            "provider_config_source": "route_based_gateway",
            "provider_config": dict(GEMINI_ROUTE),
            "legacy_fallback_blocked": False,
        },
    )
    monkeypatch.setattr(
        multimodal,
        "resolve_vision_execution_profile",
        lambda *, provider_config, multimodal_limit: {"provider_config": dict(provider_config), "multimodal_limit": multimodal_limit or 2, "frame_count": 1},
    )
    monkeypatch.setattr(multimodal, "resolve_provider_config", lambda config: dict(config))
    monkeypatch.setattr(
        multimodal,
        "_execution_control",
        lambda *_args, **_kwargs: ({}, {"status": "confirmed", "confirmed": True, "export_consent": {"valid": True, "consent_id": "fixture-consent"}}),
    )
    monkeypatch.setattr(multimodal, "vision_export_consent_image_limits", lambda _path: {"image_max_edge": 320, "image_jpeg_quality": 44})
    monkeypatch.setattr(multimodal, "prepare_image_probe", lambda paths, **_kwargs: {"image_paths": list(paths)})

    def fake_model_call(**kwargs):
        calls.append(dict(kwargs))
        return {"ok": False, "error": "Gemini FAILED_PRECONDITION: User location is not supported for the API use.", "content": "", "attempts": [], "attempt_count": 1}

    monkeypatch.setattr(multimodal, "call_vision_model_with_retries", fake_model_call)
    result = multimodal.run_multimodal_frame_analysis(
        root,
        execute=True,
        limit=2,
        indexes=[1, 2],
        execution_actor="agent",
        export_consent=root / "fixture-consent.json",
    )

    assert len(calls) == 1
    assert result["summary"]["status"] == "vision_batch_aborted"
    assert result["summary"]["error"] == "vision_batch_aborted_provider_location_unsupported"
    assert result["items"][0]["batch_abort_trigger"] is True
    assert result["items"][1]["executed"] is False
    assert result["items"][1]["batch_aborted"] is True
    assert result["items"][1]["error"] == "vision_batch_aborted_provider_location_unsupported"