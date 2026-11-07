from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.model_connector_consent import (
    SCHEMA,
    SCHEMA_V1,
    _payload_sha256,
    _normalise_deployment,
    create_model_connector_consent,
    reserve_model_connector_attempt,
    resolve_model_connector_consent_route,
    validate_model_connector_consent,
)
from video_knowledge_pipeline.model_api_settings import (
    resolve_model_api_route,
    upsert_model_api_profile,
)
from video_knowledge_pipeline.storage import write_json


PROVIDER = {
    "provider": "custom_openai_compatible",
    "base_url": "https://a.example/v1",
    "model": "model-a",
}


def _artifact(root: Path) -> Path:
    path = root / "source.md"
    path.write_text("approved source", encoding="utf-8")
    return path


def _route() -> dict[str, object]:
    return {
        "route_id": "pool-remote-text",
        "route_revision": "a" * 64,
        "virtual_model": "vkp-remote-text-pool-remote-text-aaaaaaaaaaaa",
        "execution_location": "remote",
        "deployments": [
            {
                "id": "remote-a",
                "provider": "openai_compatible",
                "model": "model-a",
                "base_url": "https://a.example/v1",
                "interface": "openai_compatible",
            },
            {
                "id": "remote-b",
                "provider": "openai_compatible",
                "model": "model-b",
                "base_url": "https://b.example/v1",
                "interface": "openai_compatible",
            },
        ],
    }


def test_new_consent_is_v2_and_locks_singleton_route(tmp_path: Path) -> None:
    consent = create_model_connector_consent(
        tmp_path,
        task="smart_summary_rewrite",
        artifact_paths=[_artifact(tmp_path)],
        provider_config=PROVIDER,
        output_contract={"format": "json", "required_keys": {"title": "string"}},
        max_estimated_cost_usd=1.0,
        confirm_data_export=True,
    )

    assert consent["schema"] == SCHEMA
    assert consent["schema"].endswith(".v2")
    assert consent["authorized_deployments"] == [consent["provider"]]
    assert consent["route"]["route_revision"]
    assert consent["output_contract"]["required_keys"] == {"title": "string"}
    assert len(consent["output_contract_sha256"]) == 64
    status = validate_model_connector_consent(
        consent["consent_path"],
        provider_config=PROVIDER,
        expected_route_revision=consent["route"]["route_revision"],
    )
    assert status["valid"] is True

    stale = validate_model_connector_consent(
        consent["consent_path"],
        provider_config=PROVIDER,
        expected_route_revision="different",
    )
    assert stale["valid"] is False
    assert any(
        row["key"] == "consent_route_revision_mismatch" for row in stale["blockers"]
    )


def test_v1_consent_remains_valid_without_rewrite(tmp_path: Path) -> None:
    consent = create_model_connector_consent(
        tmp_path,
        task="smart_summary_rewrite",
        artifact_paths=[_artifact(tmp_path)],
        provider_config=PROVIDER,
        max_estimated_cost_usd=1.0,
        confirm_data_export=True,
    )
    path = Path(consent["consent_path"])
    current_route = {
        **consent["route"],
        "deployments": consent["authorized_deployments"],
    }
    legacy = dict(consent)
    legacy["schema"] = SCHEMA_V1
    legacy.pop("authorized_deployments", None)
    legacy.pop("route", None)
    legacy["consent_sha256"] = _payload_sha256(legacy)
    write_json(path, legacy)

    status = validate_model_connector_consent(path, provider_config=PROVIDER)

    assert status["valid"] is True
    assert status["consent_schema"] == SCHEMA_V1
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == SCHEMA_V1

    routed = validate_model_connector_consent(path, route_snapshot=current_route)
    assert routed["valid"] is True
    changed_route = {
        **current_route,
        "deployments": [dict(current_route["deployments"][0])],
    }
    changed_route["deployments"][0]["model"] = "other-model"
    blocked = validate_model_connector_consent(path, route_snapshot=changed_route)
    assert blocked["valid"] is False
    assert any(
        row["key"] == "consent_provider_model_mismatch" for row in blocked["blockers"]
    )


def test_v2_consent_locks_every_remote_deployment(tmp_path: Path) -> None:
    route = _route()
    consent = create_model_connector_consent(
        tmp_path,
        task="smart_summary_rewrite",
        artifact_paths=[_artifact(tmp_path)],
        route_snapshot=route,
        max_estimated_cost_usd=1.0,
        confirm_data_export=True,
    )

    assert len(consent["authorized_deployments"]) == 2
    valid = validate_model_connector_consent(
        consent["consent_path"],
        route_snapshot=route,
        expected_route_revision=str(route["route_revision"]),
    )
    assert valid["valid"] is True

    changed = _route()
    changed["deployments"] = [dict(row) for row in changed["deployments"]]
    changed["deployments"][1]["model"] = "model-c"
    blocked = validate_model_connector_consent(
        consent["consent_path"],
        route_snapshot=changed,
        expected_route_revision=str(route["route_revision"]),
    )
    assert blocked["valid"] is False
    assert any(
        row["key"] == "consent_authorized_deployments_mismatch"
        for row in blocked["blockers"]
    )


def test_consent_route_resolution_locks_revision_and_validates_every_destination(
    tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    upsert_model_api_profile(
        {
            "id": "remote-text",
            "name": "Remote text",
            "provider": "openai_compatible",
            "adapter_backend": "proxy",
            "location": "remote",
            "capabilities": ["text"],
            "base_url": "https://a.example/v1",
            "model": "model-a",
            "timeout_seconds": 30,
            "enabled": True,
        },
        tasks=["summary_rewrite"],
        settings_path=settings,
        secrets_path=secrets,
    )
    route = resolve_model_api_route(
        "summary_rewrite", execution_location="remote", settings_path=settings
    )

    class Policy:
        def __init__(self) -> None:
            self.destinations: list[str] = []

        def require_destination_identity(self, deployment: dict[str, object]) -> None:
            self.destinations.append(str(deployment["base_url"]))

    policy = Policy()
    resolved = resolve_model_connector_consent_route(
        "smart_summary_rewrite",
        route_id=str(route["route_id"]),
        route_revision=str(route["route_revision"]),
        settings_path=settings,
        policy=policy,
    )

    assert resolved == route
    assert policy.destinations == ["https://a.example/v1"]
    with pytest.raises(ValueError, match="revision"):
        resolve_model_connector_consent_route(
            "smart_summary_rewrite",
            route_id=str(route["route_id"]),
            route_revision="stale",
            settings_path=settings,
            policy=policy,
        )


def test_remote_mcp_execution_surface_no_longer_accepts_provider_config() -> None:
    source = Path(
        "src/video_knowledge_pipeline/trusted_model_connector_remote_mcp.py"
    ).read_text(encoding="utf-8")
    start = source.index("def execute_consented_model_task_tool(")
    end = source.index("    @server.tool", start + 20)
    block = source[start:end]

    assert "route_revision" in block
    assert "provider_config" not in block


def test_route_based_v2_consent_requires_current_route_snapshot(tmp_path: Path) -> None:
    route = _route()
    consent = create_model_connector_consent(
        tmp_path,
        task="smart_summary_rewrite",
        artifact_paths=[_artifact(tmp_path)],
        route_snapshot=route,
        max_estimated_cost_usd=1.0,
        confirm_data_export=True,
    )
    first = route["deployments"][0]

    status = validate_model_connector_consent(
        consent["consent_path"],
        provider_config={
            "provider": first["provider"],
            "base_url": first["base_url"],
            "model": first["model"],
        },
    )

    assert status["valid"] is False
    assert any(
        row["key"] == "consent_route_snapshot_required" for row in status["blockers"]
    )


def test_confirmed_consent_requires_explicit_cost_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_estimated_cost_usd"):
        create_model_connector_consent(
            tmp_path,
            task="smart_summary_rewrite",
            artifact_paths=[_artifact(tmp_path)],
            provider_config=PROVIDER,
            confirm_data_export=True,
        )


def test_v2_consent_exposes_exact_upload_manifest_operator_confirmation_and_cost_limits(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    consent = create_model_connector_consent(
        tmp_path,
        task="smart_summary_rewrite",
        artifact_paths=[artifact],
        provider_config=PROVIDER,
        max_calls=2,
        max_estimated_cost_usd=0.2,
        max_cost_per_call_usd=0.1,
        confirm_data_export=True,
    )

    manifest = consent["upload_manifest"]
    assert manifest["file_count"] == 1
    assert manifest["files"] == consent["artifacts"]
    assert manifest["files"][0]["sha256"]
    assert (
        consent["operator_confirmation"]["exact_manifest_sha256"]
        == manifest["manifest_sha256"]
    )
    assert consent["scope"]["max_estimated_cost_usd"] == 0.2
    assert consent["scope"]["max_cost_per_call_usd"] == 0.1

    payload = json.loads(Path(consent["consent_path"]).read_text(encoding="utf-8"))
    payload["upload_manifest"]["files"] = []
    payload["consent_sha256"] = _payload_sha256(payload)
    write_json(Path(consent["consent_path"]), payload)
    blocked = validate_model_connector_consent(
        consent["consent_path"], provider_config=PROVIDER
    )
    assert blocked["valid"] is False
    assert any(
        row["key"] == "consent_upload_manifest_mismatch" for row in blocked["blockers"]
    )


def test_cost_reservation_blocks_before_call_limit_is_exhausted(tmp_path: Path) -> None:
    consent = create_model_connector_consent(
        tmp_path,
        task="smart_summary_rewrite",
        artifact_paths=[_artifact(tmp_path)],
        provider_config=PROVIDER,
        max_calls=4,
        max_estimated_cost_usd=0.2,
        max_cost_per_call_usd=0.1,
        confirm_data_export=True,
    )

    first = reserve_model_connector_attempt(
        consent["consent_path"],
        provider_config=PROVIDER,
        expected_calls=2,
    )
    assert first["reserved"] is True
    assert first["reserved_cost_usd"] == 0.2
    assert first["remaining_calls"] == 2

    blocked = reserve_model_connector_attempt(
        consent["consent_path"],
        provider_config=PROVIDER,
        expected_calls=1,
    )
    assert blocked["reserved"] is False
    assert blocked["remaining_calls"] == 2
    assert any(
        row["key"] == "consent_cost_limit_exceeded" for row in blocked["blockers"]
    )


def test_v1_consent_is_status_compatible_but_not_executable(tmp_path: Path) -> None:
    consent = create_model_connector_consent(
        tmp_path,
        task="smart_summary_rewrite",
        artifact_paths=[_artifact(tmp_path)],
        provider_config=PROVIDER,
        max_estimated_cost_usd=0.1,
        confirm_data_export=True,
    )
    path = Path(consent["consent_path"])
    legacy = dict(consent)
    legacy["schema"] = SCHEMA_V1
    legacy.pop("authorized_deployments", None)
    legacy.pop("route", None)
    legacy["consent_sha256"] = _payload_sha256(legacy)
    write_json(path, legacy)

    status = validate_model_connector_consent(path, provider_config=PROVIDER)
    assert status["valid"] is True
    blocked = reserve_model_connector_attempt(path, provider_config=PROVIDER)
    assert blocked["reserved"] is False
    assert any(row["key"] == "consent_v2_required" for row in blocked["blockers"])


def test_consent_rejects_missing_or_secret_required_provider_options() -> None:
    deployment = {
        "provider": "azure_openai",
        "model": "deployment-name",
        "base_url": "https://example.openai.azure.com",
        "interface": "chat_completions",
        "auth_mode": "api_key_dpapi",
        "provider_options": {"api_version": "2025-01-01-preview"},
        "required_provider_options": ["api_version"],
        "environment_bindings": [],
    }
    normalised = _normalise_deployment(deployment)
    assert normalised["required_provider_options"] == ["api_version"]

    with pytest.raises(ValueError, match="is missing"):
        _normalise_deployment(
            {
                **deployment,
                "provider_options": {},
            }
        )
    with pytest.raises(ValueError, match="must not contain secrets"):
        _normalise_deployment(
            {
                **deployment,
                "required_provider_options": ["client_secret"],
            }
        )


def test_consent_allows_token_limits_without_weakening_secret_rejection() -> None:
    deployment = {
        "provider": "siliconflow",
        "model": "Qwen/Qwen3.5-397B-A17B",
        "base_url": "https://api.siliconflow.cn/v1",
        "interface": "chat_completions",
        "auth_mode": "api_key_dpapi",
        "provider_options": {
            "enable_thinking": False,
            "thinking_budget": 256,
            "max_tokens": 256,
            "stream": True,
        },
        "required_provider_options": [],
        "environment_bindings": [],
    }

    normalised = _normalise_deployment(deployment)

    assert normalised["provider_options"]["max_tokens"] == 256
    with pytest.raises(ValueError, match="must not contain secrets"):
        _normalise_deployment(
            {
                **deployment,
                "provider_options": {"access_token": "not-a-real-secret"},
            }
        )
