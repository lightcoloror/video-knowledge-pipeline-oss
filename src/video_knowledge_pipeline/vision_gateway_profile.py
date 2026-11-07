from __future__ import annotations

from pathlib import Path
from typing import Any

from .model_api_settings import (
    public_model_api_settings_status,
    resolve_model_api_provider_config,
)
from .model_gateway import model_gateway_runtime_readiness


VISION_ROUTE_TASKS = ("semantic_frame", "temporal_sequence")
_READY_CREDENTIAL_STATUSES = frozenset({"ready", "not_required"})


def resolve_route_based_vision_gateway_profile(
    task: str,
    *,
    settings_path: str | Path | None = None,
    secrets_path: str | Path | None = None,
    gateway_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve a public Proxy route without reading or exposing a key value.

    This is intentionally a readiness resolver, not a provider probe.  It reads
    route metadata plus encrypted-secret *presence*, then probes only the VKP
    loopback gateway.  A remote model call still needs a consented execution.
    """

    task_key = _normalise_vision_task(task)
    try:
        provider_config = resolve_model_api_provider_config(
            task_key,
            settings_path=settings_path,
            secrets_path=secrets_path,
            execution_location="remote",
        )
    except (OSError, ValueError) as exc:
        return _blocked(task_key, "route_missing", str(exc))

    route_id = str(provider_config.get("route_id") or "").strip()
    route_revision = str(provider_config.get("route_revision") or "").strip()
    profile_id = str(provider_config.get("profile_id") or "").strip()
    if not route_id or not route_revision or not profile_id:
        return _blocked(
            task_key,
            "route_missing",
            "No route-based remote vision profile is configured for this task.",
        )
    if str(provider_config.get("adapter_backend") or "").strip().lower() != "proxy":
        return _blocked(
            task_key,
            "legacy_route",
            "The configured vision route is legacy; it is not a LiteLLM gateway route.",
            provider_config=provider_config,
        )

    try:
        public_settings = public_model_api_settings_status(settings_path, secrets_path)
    except (OSError, ValueError) as exc:
        return _blocked(
            task_key,
            "settings_unavailable",
            str(exc),
            provider_config=provider_config,
        )
    profiles = {
        str(row.get("id") or ""): row
        for row in public_settings.get("profiles") or []
        if isinstance(row, dict)
    }
    profile = profiles.get(profile_id) or {}
    credential_status = str(profile.get("credential_status") or "missing_api_key")
    credential_ready = credential_status in _READY_CREDENTIAL_STATUSES
    gateway = model_gateway_runtime_readiness(
        gateway_config_path=gateway_config_path,
    )
    gateway_ready = bool(gateway.get("ready"))
    status = (
        "gateway_ready"
        if credential_ready and gateway_ready
        else "gateway_credential_missing"
        if not credential_ready
        else "gateway_unavailable"
    )
    public_config = {
        **provider_config,
        "api_key": "",
        "api_key_source": "gateway_dpapi",
        "credential_status": credential_status,
        "credential_ready": credential_ready,
        "gateway_configured": True,
        "gateway_ready": gateway_ready,
        "gateway_status": str(gateway.get("status") or "unknown"),
        "gateway_host": str((gateway.get("gateway") or {}).get("host") or ""),
        "gateway_port": int((gateway.get("gateway") or {}).get("port") or 0),
    }
    return {
        "schema": "video_knowledge_pipeline.vision_gateway_profile.v1",
        "task": task_key,
        "status": status,
        "route_configured": True,
        "gateway_configured": True,
        "gateway_ready": gateway_ready,
        "credential_status": credential_status,
        "credential_ready": credential_ready,
        "route_id": route_id,
        "route_revision": route_revision,
        "virtual_model": str(provider_config.get("virtual_model") or ""),
        "profile_id": profile_id,
        "provider_config": public_config,
        "gateway": gateway,
        "remote_requests_made": False,
        "secret_values_accessed": False,
        "consent_required": True,
    }


def configured_route_based_vision_gateway_profiles(
    *,
    settings_path: str | Path | None = None,
    secrets_path: str | Path | None = None,
    gateway_config_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    return [
        resolve_route_based_vision_gateway_profile(
            task,
            settings_path=settings_path,
            secrets_path=secrets_path,
            gateway_config_path=gateway_config_path,
        )
        for task in VISION_ROUTE_TASKS
    ]


def _normalise_vision_task(value: str) -> str:
    task = str(value or "").strip().lower().replace("-", "_")
    aliases = {"vision": "semantic_frame", "temporal": "temporal_sequence"}
    task = aliases.get(task, task)
    if task not in VISION_ROUTE_TASKS:
        raise ValueError(
            "route-based vision task must be semantic_frame or temporal_sequence"
        )
    return task


def _blocked(
    task: str,
    status: str,
    error: str,
    *,
    provider_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "video_knowledge_pipeline.vision_gateway_profile.v1",
        "task": task,
        "status": status,
        "route_configured": bool(provider_config),
        "gateway_configured": bool(provider_config),
        "gateway_ready": False,
        "credential_status": "unknown",
        "credential_ready": False,
        "route_id": str((provider_config or {}).get("route_id") or ""),
        "route_revision": str((provider_config or {}).get("route_revision") or ""),
        "virtual_model": str((provider_config or {}).get("virtual_model") or ""),
        "profile_id": str((provider_config or {}).get("profile_id") or ""),
        "provider_config": dict(provider_config or {}),
        "gateway": {},
        "error": error,
        "remote_requests_made": False,
        "secret_values_accessed": False,
        "consent_required": True,
    }
