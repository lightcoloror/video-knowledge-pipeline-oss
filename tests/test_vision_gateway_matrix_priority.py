from __future__ import annotations

from video_knowledge_pipeline import vision_provider_smoke as smoke_module


def test_default_matrix_prefers_configured_gateway_without_legacy_fallback(monkeypatch) -> None:
    route_config = {
        "provider": "gemini",
        "model": "gemini-3.6-flash",
        "adapter_backend": "proxy",
        "execution_location": "remote",
        "route_id": "route-gemini",
        "route_revision": "a" * 64,
        "credential_ready": True,
        "gateway_configured": True,
        "gateway_ready": True,
    }
    monkeypatch.setattr(
        smoke_module,
        "configured_gateway_vision_profiles",
        lambda: [{"task": "temporal_sequence", "route_configured": True, "provider_config": route_config}],
    )

    def fake_smoke(*, provider_config, **kwargs):
        config = dict(provider_config)
        is_gateway = config.get("adapter_backend") == "proxy"
        provider = {
            "provider": config.get("provider"),
            "model": config.get("model", "legacy"),
            "api_key_required": not is_gateway,
            "api_key_configured": is_gateway,
            "credential_ready": is_gateway,
            "gateway_configured": is_gateway,
            "gateway_ready": is_gateway,
        }
        return {
            "provider": provider,
            "status": "gateway_ready" if is_gateway else "missing_api_key",
            "safe_to_execute": is_gateway,
            "error_class": "" if is_gateway else "missing_api_key",
            "error_summary": "",
            "diagnostics": {},
            "failure_diagnosis": {},
            "image_selection": {},
            "image_probe": {},
            "checks": [],
            "recommended_provider_config": route_config if is_gateway else {},
            "recovery_suggestion": "",
        }

    monkeypatch.setattr(smoke_module, "vision_provider_smoke", fake_smoke)

    report = smoke_module.vision_provider_matrix(providers=None, write=False)

    assert report["recommended_provider"] == "gemini"
    assert report["recommended_provider_config"]["route_id"] == "route-gemini"
    assert report["provider_ranking"][0]["provider_config_source"] == "route_based_gateway"
    assert report["provider_ranking"][0]["reason"].startswith("gateway configured and ready")
