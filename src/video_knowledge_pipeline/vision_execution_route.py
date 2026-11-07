from __future__ import annotations

from typing import Any
import socket
import sys
from urllib.parse import urlsplit

from .vision_gateway_profile import resolve_route_based_vision_gateway_profile


def resolve_vision_task_execution_route(
    task: str,
    *,
    provider_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one execution task to its explicit config or its configured gateway route.

    The result never substitutes a legacy provider for an absent route.  Callers
    may support legacy operation only when the caller supplied an explicit
    provider configuration.
    """
    if provider_config is not None:
        return {
            "task": task,
            "status": "explicit_provider_config",
            "provider_config": dict(provider_config),
            "provider_config_source": "explicit",
            "route_profile": {},
            "legacy_fallback_blocked": False,
        }

    local_endpoint = probe_lm_studio_endpoint()
    profile = resolve_route_based_vision_gateway_profile(task)
    route_config = profile.get("provider_config") if isinstance(profile.get("provider_config"), dict) else {}
    if profile.get("route_configured") and route_config:
        return {
            "task": task,
            "status": str(profile.get("status") or "gateway_unavailable"),
            "provider_config": dict(route_config),
            "provider_config_source": "route_based_gateway",
            "local_endpoint_check": local_endpoint,
            "route_profile": profile,
            "legacy_fallback_blocked": False,
        }

    return {
        "task": task,
        "status": str(profile.get("status") or "route_missing"),
        "provider_config": {},
        "provider_config_source": "route_missing",
        "local_endpoint_check": local_endpoint,
        "route_profile": profile,
        "legacy_fallback_blocked": True,
    }


def probe_lm_studio_endpoint() -> dict[str, Any]:
    """Only check the user-reported loopback port; never load a model or send media."""
    try:
        with socket.create_connection(("127.0.0.1", 1234), timeout=0.25):
            listening = True
    except OSError:
        listening = False
    return {"base_url": "http://127.0.0.1:1234/v1", "listening": listening,
            "service_identity_verified": False, "vision_model_verified": False}


def check_vision_invocation(*, execute: bool, route: dict[str, Any]) -> dict[str, Any]:
    """Make preview and implicit route selection visible before any vision call.

    A listening LM Studio port is a choice signal, not proof of a loaded VLM.
    Preserve explicit routes and all downstream calls/indexes/consent gates.
    """
    explicit = route.get("provider_config_source") == "explicit"
    local = route.get("local_endpoint_check") or {}
    config = route.get("provider_config") or {}
    if execute and explicit and not str(config.get("provider") or "").strip():
        raise ValueError("vision_route_config_incomplete: explicit --provider-config must name the intended provider")
    notices = []
    if not execute:
        notices.append("PREVIEW ONLY: no vision model will run. To execute, pass --execute, --provider-config, --confirm-vision-calls and --confirm-vision-indexes from the matching preflight.")
    if not explicit:
        notices.append("No explicit --provider-config: the configured gateway route is only a candidate, not a selection of LM Studio.")
        notices.append(f"Resolved provider={config.get('provider') or 'missing'}, model={config.get('model') or 'missing'}, readiness={route.get('status') or 'unknown'}.")
    choice_required = bool(not explicit and local.get("listening"))
    if choice_required:
        notices.append("A local endpoint is listening at http://127.0.0.1:1234/v1. Select it explicitly with provider=local_vlm, base_url, the actual loaded vision model and an empty/local API key, or explicitly select the intended remote route. A listening port does not verify LM Studio or vision capability.")
    for notice in notices:
        print("[vision-preflight] " + notice, file=sys.stderr)
    if execute and choice_required:
        raise ValueError("vision_route_choice_required: local endpoint detected; pass explicit --provider-config before execution. No vision calls made.")
    if explicit and str(config.get("provider") or "").lower() in {"local_vlm", "local_qwen_vl"}:
        if execute and urlsplit(str(config.get("base_url") or "")).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("local_vision_config_incomplete: local provider requires a loopback base_url")
        if not str(config.get("model") or "").strip() or not str(config.get("base_url") or "").strip():
            if execute:
                raise ValueError("local_vision_config_incomplete: explicit local route requires base_url and actual loaded model")
            notices.append("Local route is incomplete: provide base_url and the actual loaded model before execution.")
    return {"mode": "execute_requested" if execute else "preview_only",
            "provider_config_source": route.get("provider_config_source"),
            "route_resolution_status": route.get("status"),
            "route_choice_required": choice_required,
            "local_endpoint_check": local or {"status": "not_probed_explicit_route"},
            "automatic_route_switch": False, "notices": notices}
