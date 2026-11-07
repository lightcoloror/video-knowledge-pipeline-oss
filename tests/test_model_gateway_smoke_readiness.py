from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline import model_api_settings as settings_module
from video_knowledge_pipeline import model_gateway_smoke_readiness as readiness_module
from video_knowledge_pipeline.model_api_settings import upsert_model_api_profile
from video_knowledge_pipeline.model_gateway_smoke_readiness import (
    ONLINE_PIPELINE_TASKS,
    _configuration_warnings,
    _consent_readiness,
    _online_pipeline_route_readiness,
    _route_readiness,
    model_gateway_smoke_readiness,
)


def _frames(bundle: Path) -> None:
    group = bundle / "temporal-frames" / "0006"
    group.mkdir(parents=True)
    for index in range(1, 9):
        (group / f"frame_{index:02d}.jpg").write_bytes(f"frame-{index}".encode())


def _profile(
    profile_id: str,
    *,
    location: str,
    provider: str,
    base_url: str,
    model: str,
    capabilities: list[str],
) -> dict[str, object]:
    return {
        "id": profile_id,
        "name": profile_id,
        "provider": provider,
        "adapter_backend": "proxy",
        "location": location,
        "capabilities": capabilities,
        "base_url": base_url,
        "model": model,
        "enabled": True,
    }


def test_smoke_readiness_reports_empty_configuration_without_network(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _frames(bundle)
    result = model_gateway_smoke_readiness(
        bundle,
        indexes=[6],
        settings_path=tmp_path / "settings.json",
        secrets_path=tmp_path / "secrets.json",
        port_record_path=tmp_path / "ports.md",
        output_dir=tmp_path / "report",
    )

    assert result["status"] == "configuration_required"
    assert result["route_ready_count"] == 0
    assert result["route_required_count"] == 6
    assert result["temporal_sample"]["status"] == "ready"
    assert result["operator_boundary"]["remote_requests_made"] is False
    assert (tmp_path / "report" / "model-gateway-smoke-readiness.json").is_file()


def test_smoke_readiness_accepts_complete_routes_but_requires_start(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "bundle"
    _frames(bundle)
    settings = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    monkeypatch.setenv(
        "VKP_MODEL_CONNECTOR_ALLOWED_DESTINATIONS",
        "text.example,vision.example,asr.example,ocr.example",
    )
    monkeypatch.setattr(
        settings_module,
        "_protect_secret",
        lambda value: "cipher:" + value.encode().hex(),
    )
    monkeypatch.setattr(
        settings_module,
        "_unprotect_secret",
        lambda value: bytes.fromhex(value.removeprefix("cipher:")).decode(),
    )
    profiles = [
        (
            _profile(
                "local-vlm",
                location="local",
                provider="local_vlm",
                base_url="http://127.0.0.1:8000/v1",
                model="vlm",
                capabilities=["vision"],
            ),
            ["temporal_sequence"],
        ),
        (
            _profile(
                "local-asr",
                location="local",
                provider="openai_compatible_asr",
                base_url="http://127.0.0.1:8001/v1",
                model="asr",
                capabilities=["asr"],
            ),
            ["asr"],
        ),
        (
            _profile(
                "remote-text",
                location="remote",
                provider="openai_compatible",
                base_url="https://text.example/v1",
                model="text",
                capabilities=["text"],
            ),
            ["summary_rewrite"],
        ),
        (
            _profile(
                "remote-vlm",
                location="remote",
                provider="openai_compatible",
                base_url="https://vision.example/v1",
                model="vision",
                capabilities=["vision"],
            ),
            ["temporal_sequence"],
        ),
        (
            _profile(
                "remote-asr",
                location="remote",
                provider="openai_compatible_asr",
                base_url="https://asr.example/v1",
                model="asr",
                capabilities=["asr"],
            ),
            ["asr"],
        ),
        (
            _profile(
                "remote-ocr",
                location="remote",
                provider="mistral",
                base_url="https://ocr.example/v1",
                model="ocr",
                capabilities=["ocr"],
            ),
            ["ocr"],
        ),
    ]
    for profile, tasks in profiles:
        upsert_model_api_profile(
            profile,
            tasks=tasks,
            api_key="fake-remote-key" if profile["location"] == "remote" else "",
            settings_path=settings,
            secrets_path=secrets,
        )
    port_record = tmp_path / "ports.md"
    port_record.write_text("8776 VKP LiteLLM Proxy\n", encoding="utf-8")
    monkeypatch.setattr(
        readiness_module,
        "model_gateway_doctor",
        lambda **kwargs: {
            "status": "ready",
            "http_status": "not_probed",
            "gateway": {"host": "127.0.0.1", "port": 8776},
            "checks": [
                {"key": "port_record", "ok": True},
                {"key": "live_listener", "ok": False},
                {"key": "bind_available", "ok": True},
            ],
        },
    )

    result = model_gateway_smoke_readiness(
        bundle,
        indexes=[6],
        settings_path=settings,
        secrets_path=secrets,
        port_record_path=port_record,
        output_dir=tmp_path / "report",
    )

    assert result["status"] == "operator_start_required"
    assert result["route_ready_count"] == 6
    assert result["gateway"]["port_recorded_and_owned"] is True
    assert result["gateway"]["live"] is False
    assert all(row["adapter_backend"] == "proxy" for row in result["route_requirements"])

def test_remote_route_readiness_requires_allowlist_and_credentials() -> None:
    rows = _route_readiness(
        {
            "profiles": [
                {
                    "id": "remote-text",
                    "provider": "openai_compatible",
                    "adapter_backend": "proxy",
                    "api_key_configured": False,
                }
            ],
            "route_status": [
                {
                    "task": "summary_rewrite",
                    "execution_location": "remote",
                    "deployments": ["remote-text"],
                    "route_id": "remote-text",
                    "route_revision": "a" * 64,
                    "virtual_model": "vkp-remote-text",
                    "allowlist_status": "unknown",
                }
            ],
        }
    )
    remote_text = next(row for row in rows if row["key"] == "remote_text")

    assert remote_text["ready"] is False
    assert remote_text["allowlist_ready"] is False
    assert remote_text["credentials_ready"] is False
    assert remote_text["blockers"] == [
        "remote_allowlist_not_approved",
        "remote_credentials_missing",
    ]


def test_local_vlm_and_speaches_shared_origin_is_an_explicit_warning() -> None:
    warnings = _configuration_warnings(
        [
            {
                "key": "local_vlm",
                "deployment_origins": ["http://127.0.0.1:8000"],
            },
            {
                "key": "local_speaches",
                "deployment_origins": ["http://127.0.0.1:8000"],
            },
        ]
    )

    assert warnings == [
        {
            "key": "local_service_endpoint_shared",
            "severity": "warning",
            "origins": ["http://127.0.0.1:8000"],
            "message": (
                "Local VLM and Speaches routes share an origin. This is valid only "
                "when one operator-managed service intentionally exposes both chat "
                "completions and audio transcriptions; otherwise configure distinct ports."
            ),
        }
    ]


def test_local_vlm_and_speaches_distinct_ports_have_no_warning() -> None:
    warnings = _configuration_warnings(
        [
            {
                "key": "local_vlm",
                "deployment_origins": ["http://127.0.0.1:8000"],
            },
            {
                "key": "local_speaches",
                "deployment_origins": ["http://127.0.0.1:8001"],
            },
        ]
    )

    assert warnings == []


def test_temporal_readiness_requires_exact_six_group_consent_coverage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame_a = tmp_path / "temporal-frames" / "0006" / "frame_01.jpg"
    frame_b = tmp_path / "temporal-frames" / "0080" / "frame_01.jpg"
    frame_a.parent.mkdir(parents=True)
    frame_b.parent.mkdir(parents=True)
    frame_a.write_bytes(b"a")
    frame_b.write_bytes(b"b")
    consent_path = tmp_path / "consent.json"
    consent_path.write_text(
        __import__("json").dumps(
            {
                "task": "temporal_visual_analysis",
                "artifacts": [{"path": str(frame_a)}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.model_gateway_smoke_readiness.trusted_model_connector_status",
        lambda *args, **kwargs: {
            "valid": True,
            "status": "active",
            "remaining_calls": 6,
            "blockers": [],
        },
    )
    routes = [
        {
            "key": "remote_vision",
            "consent_task": "temporal_visual_analysis",
            "route_revision": "a" * 64,
        }
    ]
    temporal = {
        "group_count": 2,
        "groups": [
            {"frames": [{"path": str(frame_a)}]},
            {"frames": [{"path": str(frame_b)}]},
        ],
    }

    incomplete = _consent_readiness([consent_path], routes, temporal)[0]
    assert incomplete["valid"] is False
    assert incomplete["expected_calls"] == 2
    assert "temporal_artifact_coverage_mismatch" in incomplete["blockers"]

    consent_path.write_text(
        __import__("json").dumps(
            {
                "task": "temporal_visual_analysis",
                "artifacts": [{"path": str(frame_a)}, {"path": str(frame_b)}],
            }
        ),
        encoding="utf-8",
    )
    complete = _consent_readiness([consent_path], routes, temporal)[0]
    assert complete["valid"] is True
    assert complete["artifact_coverage_ready"] is True


def test_online_pipeline_readiness_requires_all_tasks_and_rejects_coding_plan() -> None:
    profiles = []
    routes = []
    for task in ONLINE_PIPELINE_TASKS:
        profile_id = f"profile-{task}"
        profiles.append(
            {
                "id": profile_id,
                "provider": "openai_compatible",
                "adapter_backend": "proxy",
                "enabled": True,
                "api_key_configured": True,
                "base_url": "https://models.example/v1",
            }
        )
        routes.append(
            {
                "task": task,
                "execution_location": "remote",
                "deployments": [profile_id],
                "route_id": f"route-{task}",
                "route_revision": task.ljust(64, "a")[:64],
                "virtual_model": f"vkp-remote-{task}",
                "allowlist_status": "approved",
            }
        )

    ready = _online_pipeline_route_readiness(
        {"profiles": profiles, "route_status": routes}
    )

    assert len(ready) == 8
    assert all(row["ready"] for row in ready)

    summary_profile = next(
        profile for profile in profiles if profile["id"] == "profile-summary_rewrite"
    )
    summary_profile["provider"] = "volcengine_coding_plan"
    blocked = _online_pipeline_route_readiness(
        {"profiles": profiles, "route_status": routes}
    )
    summary = next(row for row in blocked if row["task"] == "summary_rewrite")
    assert summary["ready"] is True
    assert summary["blockers"] == []


def test_remote_asr_representative_route_accepts_groq_asr_provider() -> None:
    rows = _route_readiness(
        {
            "profiles": [
                {
                    "id": "groq-asr",
                    "provider": "groq_asr",
                    "adapter_backend": "proxy",
                    "api_key_configured": True,
                    "base_url": "https://api.groq.com/openai/v1",
                }
            ],
            "route_status": [
                {
                    "task": "asr",
                    "execution_location": "remote",
                    "deployments": ["groq-asr"],
                    "route_id": "groq-asr-route",
                    "route_revision": "a" * 64,
                    "virtual_model": "vkp-remote-asr",
                    "allowlist_status": "approved",
                }
            ],
        }
    )
    remote_asr = next(row for row in rows if row["key"] == "remote_asr")

    assert remote_asr["ready"] is True
    assert remote_asr["route_revision"] == "a" * 64
    assert remote_asr["providers"] == ["groq_asr"]