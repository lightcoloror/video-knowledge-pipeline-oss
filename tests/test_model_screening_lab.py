from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from video_knowledge_pipeline.model_api_settings_http import build_server
from video_knowledge_pipeline.model_screening_lab import (
    LAB_SCHEMA,
    SIMULATION_SCHEMA,
    model_screening_lab_status,
    simulate_offline_gateway_contract,
)


def _onboarding() -> dict[str, object]:
    return {
        "entries": [
            {
                "id": "free-provider",
                "label": "Free Provider",
                "priority": "P0",
                "capabilities": ["text", "vision"],
                "runtime_integration": "openai_compatible",
                "status": "ready_for_consent",
                "matching_profile_ids": ["free-provider-profile"],
                "credential_configured": True,
                "route_configured": True,
                "consent_status": "not_checked",
            }
        ]
    }


def test_screening_lab_status_is_offline_and_never_quality_evidence() -> None:
    status = model_screening_lab_status(
        [{"id": "free-provider-profile"}],
        _onboarding(),
    )

    assert status["schema"] == LAB_SCHEMA
    assert status["mode"] == "offline_contract_simulation"
    assert status["criteria_weight_total"] == 100
    assert status["candidate_count"] == 1
    assert status["candidates"][0]["eligible_for_real_smoke"] is True
    boundary = status["operator_boundary"]
    assert boundary == {
        "provider_requests_made": 0,
        "source_artifacts_read": False,
        "payloads_generated": False,
        "credentials_read": False,
        "simulation_is_quality_evidence": False,
        "real_smoke_requires_route_and_consent": True,
        "cross_location_fallback_allowed": False,
    }


@pytest.mark.parametrize(
    ("scenario", "expected_status", "retryable"),
    [
        ("completed", "completed", False),
        ("rate_limited", "rate_limited", True),
        ("provider_unavailable", "provider_unavailable", True),
        ("gateway_timeout", "gateway_timeout", True),
        ("gateway_response_invalid", "gateway_response_invalid", False),
        ("local_gateway_unavailable", "local_gateway_unavailable", False),
    ],
)
def test_simulation_covers_gateway_contracts_without_provider_io(
    scenario: str,
    expected_status: str,
    retryable: bool,
) -> None:
    result = simulate_offline_gateway_contract("vision", scenario)

    assert result["schema"] == SIMULATION_SCHEMA
    assert result["simulation"] is True
    assert result["endpoint"] == "/v1/chat/completions"
    assert result["status"] == expected_status
    assert result["retryable"] is retryable
    assert result["quality_score"] is None
    assert result["content"] is None
    boundary = result["operator_boundary"]
    assert boundary["provider_requests_made"] == 0
    assert boundary["source_artifacts_read"] is False
    assert boundary["payloads_generated"] is False
    assert boundary["credentials_read"] is False
    assert boundary["eligible_for_quality_ranking"] is False
    assert boundary["cross_location_fallback_used"] is False


def test_simulation_maps_standard_endpoints_and_rejects_unknown_values() -> None:
    assert simulate_offline_gateway_contract("text_llm", "completed")["endpoint"] == "/v1/chat/completions"
    assert simulate_offline_gateway_contract("asr", "completed")["endpoint"] == "/v1/audio/transcriptions"
    assert simulate_offline_gateway_contract("ocr", "completed")["endpoint"] == "/v1/ocr"
    with pytest.raises(ValueError, match="screening task"):
        simulate_offline_gateway_contract("video_publish", "completed")
    with pytest.raises(ValueError, match="unsupported screening scenario"):
        simulate_offline_gateway_contract("text", "silent_fallback")


def test_screening_http_is_loopback_csrf_protected_and_secretless(tmp_path: Path) -> None:
    server = build_server(
        host="127.0.0.1",
        port=0,
        settings_path=tmp_path / "settings.json",
        secrets_path=tmp_path / "secrets.json",
        csrf_token="screening-token",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        page = urllib.request.urlopen(base + "/", timeout=5).read().decode("utf-8")
        assert "Offline Model Screening Lab" in page
        assert 'id="screeningTask"' in page
        assert 'id="runScreening"' in page
        assert "never count as model-quality evidence" in page

        status = json.loads(
            urllib.request.urlopen(base + "/api/settings", timeout=15)
            .read()
            .decode("utf-8")
        )
        lab = status["model_screening_lab"]
        assert lab["schema"] == LAB_SCHEMA
        assert lab["criteria_weight_total"] == 100
        assert lab["candidate_count"] == 10
        assert lab["operator_boundary"]["provider_requests_made"] == 0

        body = json.dumps(
            {"task": "vision", "scenario": "rate_limited"}
        ).encode("utf-8")
        request = urllib.request.Request(
            base + "/api/screening/simulate",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-VKP-Settings-Token": "screening-token",
            },
        )
        response = json.loads(
            urllib.request.urlopen(request, timeout=5).read().decode("utf-8")
        )
        simulation = response["simulation"]
        assert response["ok"] is True
        assert simulation["status"] == "rate_limited"
        assert simulation["endpoint"] == "/v1/chat/completions"
        assert simulation["operator_boundary"]["provider_requests_made"] == 0

        bad_request = urllib.request.Request(
            base + "/api/screening/simulate",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-VKP-Settings-Token": "wrong",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(bad_request, timeout=5)
        assert exc.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
