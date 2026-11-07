from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .model_api_settings import (
    _read_secret,
    default_secrets_path,
    default_settings_path,
    load_model_api_settings,
)
from .model_provider_onboarding import provider_onboarding_definition


SCHEMA = "video_knowledge_pipeline.model_provider_catalog_probe.v1"
MAX_CATALOG_BYTES = 8 * 1024 * 1024


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError(req.full_url, code, "catalog redirects are disabled", headers, fp)


def probe_model_api_onboarding_bundle(
    provider_id: str,
    *,
    execute: bool = False,
    include_model_ids: bool = False,
    settings_path: str | Path | None = None,
    secrets_path: str | Path | None = None,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    started = time.perf_counter()
    definition = provider_onboarding_definition(provider_id)
    templates = [dict(row) for row in definition.get("profile_templates") or []]
    if not templates:
        raise ValueError("provider does not have an exact onboarding bundle")

    settings_file = _path(settings_path, default_settings_path())
    secrets_file = _path(secrets_path, default_secrets_path(settings_path=settings_file))
    settings = load_model_api_settings(settings_file)
    profiles = {str(row["id"]): dict(row) for row in settings["profiles"]}
    expected_ids = [str(row["id"]) for row in templates]
    missing_profiles = sorted(profile_id for profile_id in expected_ids if profile_id not in profiles)
    if missing_profiles:
        return _blocked(
            provider_id,
            templates,
            status="profiles_missing",
            error="exact onboarding profiles are not installed",
            started=started,
            missing_profile_ids=missing_profiles,
        )

    credential_profile_id = next(
        (
            profile_id
            for profile_id in expected_ids
            if _read_secret(profile_id, secrets_file)
        ),
        "",
    )
    if not credential_profile_id:
        return _blocked(
            provider_id,
            templates,
            status="credential_missing",
            error="no saved credential is available for this provider bundle",
            started=started,
        )

    profile = profiles[credential_profile_id]
    destination = str(definition.get("destination") or "").casefold()
    catalog_url, auth_kind = _catalog_request_contract(profile, destination=destination)
    invocation_contracts = [
        _invocation_contract(template)
        for template in templates
    ]
    base = {
        "schema": SCHEMA,
        "provider_id": str(provider_id),
        "status": "planned",
        "ok": True,
        "execute": bool(execute),
        "catalog_url": catalog_url,
        "catalog_auth": auth_kind,
        "credential_profile_id": credential_profile_id,
        "credential_configured": True,
        "catalog_entries": _catalog_entries(templates, set(), checked=False),
        "invocation_contracts": invocation_contracts,
        "network_calls": 0,
        "model_inference_calls": 0,
        "artifact_reads": 0,
        "artifact_uploads": 0,
        "secrets_exposed": False,
        "saving_authorizes_egress": False,
        "latency_ms": 0,
    }
    if not execute:
        return base

    api_key = _read_secret(credential_profile_id, secrets_file)
    headers = {"Accept": "application/json", "User-Agent": "video-knowledge-pipeline/1.0"}
    if auth_kind == "x-goog-api-key":
        headers["X-goog-api-key"] = api_key
    else:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(catalog_url, headers=headers, method="GET")
    try:
        with _open_catalog_request(request, timeout_seconds=timeout_seconds) as response:
            raw = _read_limited(response, MAX_CATALOG_BYTES)
            http_status = int(response.status)
        payload = json.loads(raw.decode("utf-8"))
        model_ids = _model_ids(payload)
    except urllib.error.HTTPError as exc:
        return _probe_failure(base, "catalog_rejected", str(exc.code), started, http_status=exc.code)
    except (socket.timeout, TimeoutError):
        return _probe_failure(base, "catalog_timeout", "catalog request timed out", started)
    except urllib.error.URLError as exc:
        return _probe_failure(base, "catalog_unavailable", str(exc.reason), started)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _probe_failure(base, "catalog_response_invalid", str(exc), started)

    entries = _catalog_entries(templates, model_ids, checked=True)
    enabled_missing = [
        row["model"]
        for row in entries
        if row["install_enabled"]
        and row["catalog_visibility_required"]
        and not row["visible"]
    ]
    base.update(
        {
            "status": "verified" if not enabled_missing else "models_missing",
            "ok": not enabled_missing,
            "http_status": http_status,
            "catalog_count": len(model_ids),
            "catalog_entries": entries,
            "enabled_models_missing": enabled_missing,
            "network_calls": 1,
            "latency_ms": _elapsed_ms(started),
        }
    )
    if include_model_ids:
        base["catalog_model_ids"] = sorted(model_ids, key=str.casefold)
    return base


def _catalog_request_contract(profile: dict[str, Any], *, destination: str) -> tuple[str, str]:
    base_url = str(profile.get("base_url") or "").strip().rstrip("/")
    parsed = urlsplit(base_url)
    host = str(parsed.hostname or "").casefold()
    if parsed.scheme != "https" or not host or host != destination:
        raise ValueError("catalog probe destination must match the reviewed HTTPS onboarding destination")
    provider = str(profile.get("provider") or "")
    litellm_provider = str(profile.get("litellm_provider") or "")
    if provider == "gemini" or litellm_provider == "gemini":
        return base_url + "/models?pageSize=1000", "x-goog-api-key"
    return base_url + "/models", "bearer"


def _invocation_contract(template: dict[str, Any]) -> dict[str, Any]:
    protocol = str(template.get("protocol") or "chat_completions")
    base_url = str(template.get("base_url") or "").rstrip("/")
    model = str(template.get("model") or "")
    litellm_provider = str(template.get("litellm_provider") or "")
    if protocol == "audio_transcriptions":
        gateway_endpoint = "/v1/audio/transcriptions"
        provider_endpoint = base_url + "/audio/transcriptions"
        payload = "multipart file + model; bounded prompt optional"
    elif protocol == "ocr":
        gateway_endpoint = "/v1/ocr"
        provider_endpoint = base_url + "/ocr"
        payload = "one document object with complete MIME data URL"
    else:
        gateway_endpoint = "/v1/chat/completions"
        if litellm_provider == "gemini":
            provider_endpoint = base_url + "/models/" + model + ":generateContent"
            payload = "Gemini contents/parts; key in X-goog-api-key header"
        else:
            provider_endpoint = base_url + "/chat/completions"
            payload = "OpenAI-compatible messages; local images become transient MIME data URLs"
    return {
        "profile_id": str(template.get("id") or ""),
        "model": model,
        "install_enabled": bool(template.get("install_enabled", True)),
        "catalog_status": str(template.get("catalog_status") or "unknown"),
        "client_transport": "loopback_litellm_proxy",
        "client_endpoint": gateway_endpoint,
        "provider_protocol": protocol,
        "provider_endpoint": provider_endpoint,
        "litellm_model": model if model.startswith(litellm_provider + "/") else litellm_provider + "/" + model,
        "payload_contract": payload,
        "direct_provider_url_accepted_from_agent": False,
    }


def _catalog_entries(templates: list[dict[str, Any]], model_ids: set[str], *, checked: bool) -> list[dict[str, Any]]:
    return [
        {
            "profile_id": str(row.get("id") or ""),
            "model": str(row.get("model") or ""),
            "install_enabled": bool(row.get("install_enabled", True)),
            "catalog_status": str(row.get("catalog_status") or "unknown"),
            "catalog_visibility_required": not str(
                row.get("catalog_status") or ""
            ).startswith("coding_plan_alias_"),
            "visible": str(row.get("model") or "") in model_ids if checked else None,
        }
        for row in templates
    ]


def _model_ids(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        raise ValueError("catalog response must be an object")
    rows = payload.get("models") if isinstance(payload.get("models"), list) else payload.get("data")
    if not isinstance(rows, list):
        raise ValueError("catalog response does not contain models or data")
    result: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = str(row.get("id") or row.get("name") or "")
        if value.startswith("models/"):
            value = value.removeprefix("models/")
        if value:
            result.add(value)
    return result


def _open_catalog_request(request: urllib.request.Request, *, timeout_seconds: int):
    opener = urllib.request.build_opener(_NoRedirectHandler())
    return opener.open(request, timeout=max(1, int(timeout_seconds)))


def _read_limited(response: Any, limit: int) -> bytes:
    data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("catalog response exceeds the size limit")
    return data


def _blocked(provider_id: str, templates: list[dict[str, Any]], *, status: str, error: str, started: float, **extra: Any) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "provider_id": str(provider_id),
        "status": status,
        "ok": False,
        "execute": False,
        "error": error,
        "catalog_entries": _catalog_entries(templates, set(), checked=False),
        "network_calls": 0,
        "model_inference_calls": 0,
        "artifact_reads": 0,
        "artifact_uploads": 0,
        "secrets_exposed": False,
        "latency_ms": _elapsed_ms(started),
        **extra,
    }


def _probe_failure(base: dict[str, Any], status: str, error: str, started: float, *, http_status: int | None = None) -> dict[str, Any]:
    return {
        **base,
        "ok": False,
        "status": status,
        "error": str(error)[:500],
        "http_status": http_status,
        "network_calls": 1,
        "latency_ms": _elapsed_ms(started),
    }


def _path(value: str | Path | None, default: Path) -> Path:
    return Path(value).expanduser().resolve() if value else default


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))
