from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.model_api_settings import (
    SETTINGS_SCHEMA,
    load_model_api_settings,
    resolve_model_api_route,
    replace_model_api_route_configuration,
    upsert_model_api_profile,
    validate_model_api_profile,
)


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
        "timeout_seconds": 120,
        "enabled": True,
    }


def test_v1_settings_migrate_in_memory_without_rewriting_source(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    legacy = {
        "schema": "video_knowledge_pipeline.local_model_api_settings.v1",
        "profiles": [
            {
                "id": "remote-main",
                "name": "Remote",
                "provider": "openai_compatible",
                "adapter_backend": "builtin",
                "base_url": "https://models.example.com/v1",
                "model": "vision-main",
                "timeout_seconds": 90,
                "enabled": True,
            }
        ],
        "task_routes": {"semantic_frame": "remote-main"},
        "updated_at": "2026-07-14T00:00:00+00:00",
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")

    migrated = load_model_api_settings(path)

    assert migrated["schema"] == SETTINGS_SCHEMA
    assert migrated["profiles"][0]["location"] == "remote"
    binding = migrated["route_bindings"]["semantic_frame"]
    assert binding["default_location"] == "remote"
    assert binding["remote_pool_id"]
    assert migrated["route_pools"][0]["location"] == "remote_approved"
    assert json.loads(path.read_text(encoding="utf-8"))["schema"].endswith(".v1")


def test_route_pool_rejects_mixed_local_and_remote_deployments(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    data = {
        "schema": SETTINGS_SCHEMA,
        "profiles": [
            _profile(
                "local-vlm",
                location="local",
                provider="local_vlm",
                base_url="http://127.0.0.1:8000/v1",
                model="local-vlm",
                capabilities=["vision"],
            ),
            _profile(
                "remote-vlm",
                location="remote",
                provider="openai_compatible",
                base_url="https://models.example.com/v1",
                model="remote-vlm",
                capabilities=["vision"],
            ),
        ],
        "route_pools": [
            {
                "id": "mixed",
                "name": "mixed",
                "location": "local_only",
                "capability": "vision",
                "deployments": ["local-vlm", "remote-vlm"],
            }
        ],
        "task_routes": {"semantic_frame": "local-vlm"},
        "route_bindings": {
            "semantic_frame": {
                "default_location": "local",
                "local_pool_id": "mixed",
                "remote_pool_id": "",
            }
        },
        "updated_at": "",
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="mix|location"):
        load_model_api_settings(path)


def test_content_addressed_route_revision_changes_with_deployment(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    profile = _profile(
        "local-vlm",
        location="local",
        provider="local_vlm",
        base_url="http://127.0.0.1:8000/v1",
        model="local-vlm-v1",
        capabilities=["vision"],
    )
    upsert_model_api_profile(
        profile,
        tasks=["semantic_frame"],
        settings_path=path,
        secrets_path=secrets,
    )
    first = resolve_model_api_route("semantic_frame", execution_location="local", settings_path=path)

    profile["model"] = "local-vlm-v2"
    upsert_model_api_profile(
        profile,
        tasks=["semantic_frame"],
        settings_path=path,
        secrets_path=secrets,
    )
    second = resolve_model_api_route("semantic_frame", execution_location="local", settings_path=path)

    assert first["route_revision"] != second["route_revision"]
    assert first["virtual_model"].startswith("vkp-local-vision-")
    assert first["virtual_model"].endswith(first["route_revision"][:12])
    assert second["deployments"][0]["model"] == "local-vlm-v2"


def test_local_and_remote_routes_remain_separate(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    upsert_model_api_profile(
        _profile(
            "remote-text",
            location="remote",
            provider="openai_compatible",
            base_url="https://models.example.com/v1",
            model="remote-text",
            capabilities=["text"],
        ),
        tasks=["text_llm"],
        settings_path=path,
        secrets_path=secrets,
    )
    upsert_model_api_profile(
        _profile(
            "local-text",
            location="local",
            provider="openai_compatible",
            base_url="http://127.0.0.1:9000/v1",
            model="local-text",
            capabilities=["text"],
        ),
        tasks=["text_llm"],
        settings_path=path,
        secrets_path=secrets,
    )

    local_route = resolve_model_api_route("text_llm", execution_location="local", settings_path=path)
    remote_route = resolve_model_api_route("text_llm", execution_location="remote", settings_path=path)

    assert local_route["deployments"][0]["id"] == "local-text"
    assert remote_route["deployments"][0]["id"] == "remote-text"
    assert local_route["route_id"] != remote_route["route_id"]

def test_route_revision_locks_capability_protocol(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    upsert_model_api_profile(
        _profile(
            "remote-asr",
            location="remote",
            provider="openai_compatible_asr",
            base_url="https://asr-adapter.example/v1",
            model="asr-model",
            capabilities=["asr"],
        ),
        tasks=["asr"],
        settings_path=path,
        secrets_path=secrets,
    )
    upsert_model_api_profile(
        _profile(
            "remote-ocr",
            location="remote",
            provider="mistral_compatible_ocr",
            base_url="https://ocr-adapter.example/v1",
            model="ocr-model",
            capabilities=["ocr"],
        ),
        tasks=["ocr"],
        settings_path=path,
        secrets_path=secrets,
    )

    asr = resolve_model_api_route("asr", execution_location="remote", settings_path=path)
    ocr = resolve_model_api_route("ocr", execution_location="remote", settings_path=path)

    assert asr["deployments"][0]["interface"] == "openai_audio_transcriptions"
    assert ocr["deployments"][0]["interface"] == "mistral_ocr"
    assert asr["route_revision"] != ocr["route_revision"]

def test_profile_edit_preserves_explicit_multi_deployment_pool(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    first = _profile(
        "remote-first",
        location="remote",
        provider="openai_compatible",
        base_url="https://first.example/v1",
        model="first",
        capabilities=["text"],
    )
    second = _profile(
        "remote-second",
        location="remote",
        provider="openai_compatible",
        base_url="https://second.example/v1",
        model="second",
        capabilities=["text"],
    )
    upsert_model_api_profile(first, tasks=["summary_rewrite"], settings_path=path, secrets_path=secrets)
    status = upsert_model_api_profile(second, tasks=["summary_rewrite"], settings_path=path, secrets_path=secrets)
    pool = dict(status["route_pools"][0])
    pool["deployments"] = ["remote-first", "remote-second"]
    replace_model_api_route_configuration(
        [pool],
        status["route_bindings"],
        settings_path=path,
        secrets_path=secrets,
    )

    first["name"] = "Remote first renamed"
    updated = upsert_model_api_profile(
        first,
        tasks=["summary_rewrite"],
        settings_path=path,
        secrets_path=secrets,
    )

    assert updated["route_pools"][0]["deployments"] == ["remote-first", "remote-second"]
    assert updated["route_bindings"]["summary_rewrite"]["remote_pool_id"] == pool["id"]

def test_proxy_ocr_requires_native_mistral_or_explicit_thin_adapter(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    unsupported = _profile(
        "remote-ocr",
        location="remote",
        provider="openai_compatible",
        base_url="https://ocr.example/v1",
        model="ocr-model",
        capabilities=["ocr"],
    )

    with pytest.raises(ValueError, match="Mistral-compatible thin adapter"):
        upsert_model_api_profile(
            unsupported,
            tasks=["ocr"],
            settings_path=settings,
            secrets_path=secrets,
        )

    accepted = upsert_model_api_profile(
        {**unsupported, "provider": "mistral_compatible_ocr"},
        tasks=["ocr"],
        settings_path=settings,
        secrets_path=secrets,
    )
    assert accepted["profiles"][0]["provider"] == "mistral_compatible_ocr"
    assert accepted["route_bindings"]["ocr"]["remote_pool_id"]

def test_route_binding_rejects_wrong_task_capability(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    data = {
        "schema": SETTINGS_SCHEMA,
        "profiles": [
            _profile(
                "remote-vision",
                location="remote",
                provider="openai_compatible",
                base_url="https://vision.example/v1",
                model="vision-model",
                capabilities=["vision"],
            )
        ],
        "route_pools": [
            {
                "id": "remote-vision",
                "name": "Remote vision",
                "location": "remote_approved",
                "capability": "vision",
                "deployments": ["remote-vision"],
            }
        ],
        "task_routes": {},
        "route_bindings": {
            "summary_rewrite": {
                "default_location": "remote",
                "local_pool_id": "",
                "remote_pool_id": "remote-vision",
            }
        },
        "updated_at": "",
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="task capability"):
        load_model_api_settings(path)

def test_route_pool_rejects_proxy_legacy_mix_and_duplicate_deployments(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    profiles = [
        _profile(
            "proxy-text",
            location="remote",
            provider="openai_compatible",
            base_url="https://proxy.example/v1",
            model="proxy",
            capabilities=["text"],
        ),
        {
            **_profile(
                "legacy-text",
                location="remote",
                provider="openai_compatible",
                base_url="https://legacy.example/v1",
                model="legacy",
                capabilities=["text"],
            ),
            "adapter_backend": "legacy",
        },
    ]
    base = {
        "schema": SETTINGS_SCHEMA,
        "profiles": profiles,
        "task_routes": {},
        "route_bindings": {
            "summary_rewrite": {
                "default_location": "remote",
                "local_pool_id": "",
                "remote_pool_id": "remote-text",
            }
        },
        "updated_at": "",
    }
    mixed = {
        **base,
        "route_pools": [
            {
                "id": "remote-text",
                "name": "Remote text",
                "location": "remote_approved",
                "capability": "text",
                "deployments": ["proxy-text", "legacy-text"],
            }
        ],
    }
    path.write_text(json.dumps(mixed), encoding="utf-8")
    with pytest.raises(ValueError, match="proxy and legacy"):
        load_model_api_settings(path)

    duplicate = {
        **base,
        "route_pools": [
            {
                "id": "remote-text",
                "name": "Remote text",
                "location": "remote_approved",
                "capability": "text",
                "deployments": ["proxy-text", "proxy-text"],
            }
        ],
    }
    path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="unique and ordered"):
        load_model_api_settings(path)

def test_enabled_profile_requires_base_url_and_model() -> None:
    profile = _profile(
        "remote-text",
        location="remote",
        provider="openai_compatible",
        base_url="https://api.example/v1",
        model="example-model",
        capabilities=["text"],
    )
    with pytest.raises(ValueError, match="require base_url"):
        validate_model_api_profile({**profile, "base_url": ""})
    with pytest.raises(ValueError, match="require model"):
        validate_model_api_profile({**profile, "model": ""})

    disabled = validate_model_api_profile(
        {**profile, "base_url": "", "model": "", "enabled": False}
    )
    assert disabled["profile"]["enabled"] is False
