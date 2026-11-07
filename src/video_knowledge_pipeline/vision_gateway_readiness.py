from __future__ import annotations

from typing import Any

from .vision_gateway_profile import (
    VISION_ROUTE_TASKS,
    resolve_route_based_vision_gateway_profile,
)


def route_based_gateway_provider_test(
    provider_config: dict[str, Any] | None = None,
    *,
    task: str = "",
) -> dict[str, Any]:
    """Report route/gateway readiness without issuing a provider request.

    A proxy route owns its API key in the gateway process.  This function must
    therefore never fall back to ``GEMINI_API_KEY`` or try a direct provider
    smoke request merely to decide whether the route is configured.
    """
    config = dict(provider_config or {})
    task_key = str(task or config.get("task") or "").strip()
    if not task_key:
        task_key = "temporal_sequence"
    if not config.get("route_id"):
        profile = resolve_route_based_vision_gateway_profile(task_key)
        config = dict(profile.get("provider_config") or {})
    gateway_configured = bool(config.get("gateway_configured"))
    gateway_ready = bool(config.get("gateway_ready"))
    credential_ready = bool(config.get("credential_ready"))
    status = (
        "gateway_ready"
        if gateway_configured and gateway_ready and credential_ready
        else "gateway_credential_missing"
        if gateway_configured and not credential_ready
        else "gateway_unavailable"
        if gateway_configured
        else "route_missing"
    )
    error_class = "" if status == "gateway_ready" else status
    error_summary = {
        "gateway_credential_missing": "The configured gateway route has no ready DPAPI credential.",
        "gateway_unavailable": "The configured LiteLLM loopback gateway is not ready.",
        "route_missing": "No route-based remote vision profile is configured for this task.",
    }.get(status, "")
    provider = {
        "provider": str(config.get("provider") or ""),
        "base_url": str(config.get("base_url") or ""),
        "model": str(config.get("model") or ""),
        "timeout_seconds": int(config.get("timeout_seconds") or 0),
        "adapter_backend": "proxy",
        "execution_location": "remote",
        "route_id": str(config.get("route_id") or ""),
        "route_revision": str(config.get("route_revision") or ""),
        "virtual_model": str(config.get("virtual_model") or ""),
        "profile_id": str(config.get("profile_id") or ""),
        "api_key_required": False,
        "api_key_configured": credential_ready,
        "credential_status": str(config.get("credential_status") or "missing_api_key"),
        "credential_ready": credential_ready,
        "gateway_configured": gateway_configured,
        "gateway_ready": gateway_ready,
        "gateway_status": str(config.get("gateway_status") or "unknown"),
    }
    return {
        "schema": "lecture_vision_provider_test.v1",
        "status": status,
        "safe_to_execute": status == "gateway_ready",
        "error_class": error_class,
        "error_summary": error_summary,
        "provider": provider,
        "checks": [
            {
                "name": "gateway_route_readiness",
                "ok": status == "gateway_ready",
                "status": status,
                "error_class": error_class,
                "error": error_summary,
                "image_count": 0,
                "remote_requests_made": False,
            }
        ],
        "failure_diagnosis": {
            "status": error_class,
            "text_ping_ok": False,
            "image_checks_failed": 0,
            "provider_execution_verified": False,
        },
        "secrets_redacted": True,
        "remote_requests_made": False,
        "consent_required": True,
    }


def configured_gateway_vision_profiles() -> list[dict[str, Any]]:
    """Return one public route profile per supported vision task, in priority order."""
    return [resolve_route_based_vision_gateway_profile(task) for task in VISION_ROUTE_TASKS]
