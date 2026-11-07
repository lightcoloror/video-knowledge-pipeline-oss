"""Thin, fail-closed adapter from VKP route settings to the shared gateway."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .canonical_json import canonical_json_sha256
from .model_route_settings import resolve_model_route


ADAPTER_SCHEMA = "video_knowledge_pipeline.model_provider_gateway_adapter.v2"
_PROVIDER_ALIASES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "openrouter": "openrouter",
    "siliconflow": "siliconflow",
    "modelscope": "modelscope",
    "volcengine_ark": "ark",
    "groq": "groq",
    "groq_asr": "groq",
    "mistral_chat": "mistral",
    "mistral_compatible_ocr": "mistral",
}


class SharedGatewayUnavailable(RuntimeError):
    """Raised when the optional shared package is not installed."""


def _shared(module: str):
    try:
        return import_module(f"model_provider_gateway.{module}")
    except ImportError as exc:
        raise SharedGatewayUnavailable(
            "model-provider-gateway is not installed; install the VKP shared_gateway extra"
        ) from exc


def shared_vkp_adapter_contract() -> dict[str, Any]:
    return _shared("adapters").adapter_contract("vkp")


def reviewed_shared_preset(profile: dict[str, Any]) -> str:
    if str(profile.get("location") or "") != "remote":
        raise ValueError(
            "shared provider presets currently accept remote VKP profiles only"
        )
    if str(profile.get("adapter_backend") or "") != "proxy":
        raise ValueError(
            "legacy VKP providers must not silently migrate to the shared gateway"
        )
    shared_provider = _PROVIDER_ALIASES.get(str(profile.get("provider") or ""))
    if not shared_provider:
        raise ValueError("VKP provider has no reviewed shared-gateway mapping")
    catalog_row = _shared("catalog").catalog_show(shared_provider)
    base_url = str(profile.get("base_url") or "").rstrip("/")
    model = str(profile.get("model") or "")
    capabilities = set(profile.get("capabilities") or [])
    matches = [
        candidate
        for candidate in catalog_row["models"]
        if str(catalog_row["base_url"]).rstrip("/") == base_url
        and str(candidate["model"]) == model
        and capabilities <= set(candidate["capabilities"])
    ]
    if len(matches) != 1:
        raise ValueError(
            "VKP profile does not exactly match one reviewed shared-gateway preset"
        )
    expected_options = dict(matches[0].get("provider_options") or {})
    if dict(profile.get("provider_options") or {}) != expected_options:
        raise ValueError(
            "VKP provider options drifted from the reviewed shared-gateway preset"
        )
    return str(matches[0]["preset_id"])


def vkp_profile_to_shared(
    profile: dict[str, Any],
    *,
    max_calls: int = 1,
    max_cost_usd: float = 0.0,
    source_route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preset_id = reviewed_shared_preset(profile)
    profile_module = _shared("profile")
    result = profile_module.build_profile(
        preset_id,
        str(profile.get("secret_ref") or ""),
        profile_id=str(profile.get("id") or ""),
        capabilities=list(profile.get("capabilities") or []),
        timeout_seconds=int(profile.get("timeout_seconds") or 120),
        max_attempts=1,
        max_calls=max_calls,
        max_cost_usd=max_cost_usd,
        last_verified=str(profile.get("last_verified") or "") or None,
    )
    result["provenance"]["vkp_source"] = {
        "profile_id": str(profile.get("id") or ""),
        "profile_sha256": canonical_json_sha256(profile),
        "route_id": str((source_route or {}).get("route_id") or ""),
        "route_revision": str((source_route or {}).get("route_revision") or ""),
    }
    result["route_revision"] = profile_module.profile_revision(result)
    return profile_module.validate_profile(result)


def vkp_route_to_shared(
    settings: dict[str, Any],
    *,
    task: str,
    execution_location: str = "",
    fallback_enabled: bool = False,
    max_calls: int = 1,
    max_cost_usd: float = 0.0,
) -> dict[str, Any]:
    source_route = resolve_model_route(
        settings,
        task=task,
        execution_location=execution_location,
    )
    profiles = {str(row.get("id") or ""): row for row in settings.get("profiles") or []}
    shared_profiles = []
    for deployment in source_route["deployments"]:
        profile_id = str(deployment.get("id") or "")
        profile = profiles.get(profile_id)
        if not profile:
            raise ValueError(f"VKP route profile not found: {profile_id}")
        shared_profiles.append(
            vkp_profile_to_shared(
                profile,
                max_calls=max_calls,
                max_cost_usd=max_cost_usd,
                source_route=source_route,
            )
        )
    shared_route = _shared("route").build_route_plan(
        str(source_route["route_id"]),
        str(source_route["capability"]),
        shared_profiles,
        fallback_enabled=fallback_enabled,
        max_calls=max_calls,
        max_cost_usd=max_cost_usd,
    )
    return {
        "schema": ADAPTER_SCHEMA,
        "source_route": {
            "route_id": source_route["route_id"],
            "route_revision": source_route["route_revision"],
            "execution_location": source_route["execution_location"],
        },
        "shared_route": shared_route,
        "profiles": shared_profiles,
        "adapter_contract": shared_vkp_adapter_contract(),
        "secrets_exposed": False,
        "provider_execution_performed": False,
        "silent_fallback_allowed": False,
    }


def vkp_execution_request_to_shared(
    projection: dict[str, Any],
    consent: dict[str, Any],
    *,
    task: str,
    max_estimated_cost_usd: float,
    broker_receipt_hash: str,
    selected_profile_id: str = "",
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build the shared canonical request without executing or resolving VKP secrets."""

    if projection.get("schema") != ADAPTER_SCHEMA:
        raise ValueError("unsupported VKP shared-gateway projection schema")
    profiles = list(projection.get("profiles") or [])
    if selected_profile_id:
        profiles = [
            row
            for row in profiles
            if str(row.get("profile_id") or "") == selected_profile_id
        ]
    elif len(profiles) != 1:
        raise ValueError(
            "multi-deployment VKP routes require an explicit selected_profile_id"
        )
    if len(profiles) != 1:
        raise ValueError(
            "selected VKP shared-gateway profile was not found exactly once"
        )
    request = _shared("execution").build_execution_request(
        profiles[0],
        dict(projection.get("shared_route") or {}),
        consent,
        consumer_id="vkp",
        task=task,
        max_estimated_cost_usd=max_estimated_cost_usd,
        request_id=request_id,
        owner_gate_receipt_hash=broker_receipt_hash,
    )
    if request.get("owner_gate_receipt_hash") != broker_receipt_hash:
        raise ValueError(
            "shared execution request did not preserve the VKP Broker receipt hash"
        )
    return request
