from __future__ import annotations

import json
import re
import urllib.parse
from pathlib import Path
from typing import Any

from .canonical_json import canonical_json_sha256
from .media_capability_registry import (
    CONTROL_PLANE_DESTINATION,
    PROTOCOL,
    PROVIDER,
    UPSTREAM_COMMIT,
    media_capability,
)
from .model_connector_consent import _normalise_route_snapshot
from .storage import write_json
from .time_utils import utc_now_iso_seconds

SCHEMA = "video_knowledge_pipeline.media_route_settings.v1"
DEFAULT_FILENAME = "media-route-settings.json"
CONTROL_PLANE_BASE_URL = f"https://{CONTROL_PLANE_DESTINATION}"
DEFAULT_ROUTE_ID = "mediakit-remote-approved"
_ROUTE_ID_RE = re.compile(r"^[a-z][a-z0-9-]{2,79}$")


def default_media_route_settings_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".local" / DEFAULT_FILENAME


def load_media_route_settings(
    settings_path: str | Path | None = None,
) -> dict[str, Any]:
    path = _settings_path(settings_path)
    if not path.is_file():
        return _normalise_settings({}, settings_path=path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("media route settings must be a JSON object")
    return _normalise_settings(payload, settings_path=path)


def save_media_route_settings(
    *,
    upload_destinations: list[str] | tuple[str, ...] = (),
    route_id: str = DEFAULT_ROUTE_ID,
    max_poll_attempts: int = 12,
    poll_interval_seconds: float = 10,
    timeout_seconds: int = 900,
    settings_path: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    path = _settings_path(settings_path)
    payload = _normalise_settings(
        {
            "schema": SCHEMA,
            "provider": PROVIDER,
            "route_id": route_id,
            "control_plane_base_url": CONTROL_PLANE_BASE_URL,
            "upload_destinations": list(upload_destinations),
            "max_poll_attempts": max_poll_attempts,
            "poll_interval_seconds": poll_interval_seconds,
            "timeout_seconds": timeout_seconds,
            "updated_at": utc_now_iso_seconds(),
        },
        settings_path=path,
    )
    if write:
        write_json(path, {key: value for key, value in payload.items() if key != "settings_path"})
    return payload


def build_media_route_snapshot(
    task: str,
    *,
    settings_path: str | Path | None = None,
) -> dict[str, Any]:
    capability = media_capability(task)
    settings = load_media_route_settings(settings_path)
    route_id = f"{settings['route_id']}-{capability['task'].replace('_', '-')}"
    # The official CLI owns provider-managed local uploads. Legacy upload origins
    # are retained for migration visibility but never become arbitrary egress routes.
    destinations = [CONTROL_PLANE_BASE_URL]
    deployment = {
        "id": "volcengine-mediakit",
        "provider": PROVIDER,
        "model": capability["provider_task"],
        "base_url": CONTROL_PLANE_BASE_URL,
        "interface": PROTOCOL,
        "adapter_backend": "media_capability",
        "auth_mode": "external_environment",
        "api_key_optional": False,
        "provider_options": {
            "max_poll_attempts": settings["max_poll_attempts"],
            "poll_interval_ms": int(round(settings["poll_interval_seconds"] * 1000)),
            "source_commit": UPSTREAM_COMMIT[:12],
        },
        "required_provider_options": [],
        "environment_bindings": [
            {"param": "api_key", "env": "MEDIAKIT_API_KEY", "required": True},
        ],
        "timeout_seconds": settings["timeout_seconds"],
    }
    revision_seed = {
        "route_id": route_id,
        "execution_location": "remote",
        "destinations": destinations,
        "deployments": [deployment],
    }
    revision = canonical_json_sha256(revision_seed)
    route = _normalise_route_snapshot(
        {
            **revision_seed,
            "route_revision": revision,
            "virtual_model": f"vkp-media-{capability['task'].replace('_', '-')}-{revision[:12]}",
        }
    )
    return {
        "schema": "video_knowledge_pipeline.media_route_status.v1",
        "status": "ready_for_consent",
        "execution_ready": True,
        "task": capability["task"],
        "provider_task": capability["provider_task"],
        "route": route,
        "settings": settings,
        "credential": {
            "kind": "external_environment",
            "env": "MEDIAKIT_API_KEY",
            "value_exposed": False,
        },
        "operator_boundary": {
            "saving_configuration_authorizes_egress": False,
            "provider_managed_local_uploads": True,
            "legacy_upload_destinations_used_for_execution": False,
            "consent_v2_required": True,
            "real_execution_tool_available": bool(_mediakit_cli_status().get("available")),
            "silent_fallback_allowed": False,
        },
    }


def media_route_settings_status(
    *,
    task: str = "",
    settings_path: str | Path | None = None,
) -> dict[str, Any]:
    if task:
        return build_media_route_snapshot(task, settings_path=settings_path)
    tasks = [
        "scene_segmentation",
        "storyline",
        "highlight_detection",
        "video_ocr",
        "video_asr",
    ]
    rows = [build_media_route_snapshot(item, settings_path=settings_path) for item in tasks]
    settings = load_media_route_settings(settings_path)
    return {
        "schema": "video_knowledge_pipeline.media_route_matrix.v1",
        "status": "ready_for_consent",
        "settings": settings,
        "routes": rows,
        "route_count": len(rows),
        "execution_tools_available": bool(_mediakit_cli_status().get("available")),
        "execution_tool": _mediakit_cli_status(),
    }


def _normalise_settings(value: dict[str, Any], *, settings_path: Path) -> dict[str, Any]:
    schema = str(value.get("schema") or SCHEMA)
    if schema != SCHEMA:
        raise ValueError(f"unsupported media route settings schema: {schema}")
    provider = str(value.get("provider") or PROVIDER)
    if provider != PROVIDER:
        raise ValueError(f"media route provider must remain {PROVIDER}")
    control_plane = str(value.get("control_plane_base_url") or CONTROL_PLANE_BASE_URL).rstrip("/")
    if control_plane != CONTROL_PLANE_BASE_URL:
        raise ValueError("MediaKit control plane destination is fixed and cannot be overridden")
    route_id = str(value.get("route_id") or DEFAULT_ROUTE_ID).strip().lower()
    if not _ROUTE_ID_RE.fullmatch(route_id):
        raise ValueError("media route_id must use lowercase letters, digits, and hyphens")
    raw_destinations = value.get("upload_destinations") or []
    if not isinstance(raw_destinations, list):
        raise ValueError("upload_destinations must be a list")
    destinations: list[str] = []
    for raw in raw_destinations:
        origin = _normalise_https_origin(str(raw or ""))
        if origin == CONTROL_PLANE_BASE_URL:
            continue
        if origin not in destinations:
            destinations.append(origin)
    max_poll_attempts = int(value.get("max_poll_attempts") or 12)
    poll_interval_seconds = float(value.get("poll_interval_seconds") if value.get("poll_interval_seconds") is not None else 10)
    timeout_seconds = int(value.get("timeout_seconds") or 900)
    if not 1 <= max_poll_attempts <= 100:
        raise ValueError("max_poll_attempts must be between 1 and 100")
    if not 0 <= poll_interval_seconds <= 60:
        raise ValueError("poll_interval_seconds must be between 0 and 60")
    if not 1 <= timeout_seconds <= 3600:
        raise ValueError("timeout_seconds must be between 1 and 3600")
    return {
        "schema": SCHEMA,
        "provider": PROVIDER,
        "route_id": route_id,
        "control_plane_base_url": CONTROL_PLANE_BASE_URL,
        "upload_destinations": sorted(destinations),
        "max_poll_attempts": max_poll_attempts,
        "poll_interval_seconds": poll_interval_seconds,
        "timeout_seconds": timeout_seconds,
        "updated_at": str(value.get("updated_at") or ""),
        "settings_path": str(settings_path),
        "secrets_persisted": False,
    }


def _normalise_https_origin(value: str) -> str:
    parsed = urllib.parse.urlsplit(str(value or "").strip())
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("upload destination must be an explicit HTTPS origin")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("upload destination must not contain credentials, query, or fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError("upload destination must be an origin without a path")
    host = parsed.hostname.lower()
    rendered_host = f"[{host}]" if ":" in host else host
    port = f":{parsed.port}" if parsed.port and parsed.port != 443 else ""
    return f"https://{rendered_host}{port}"


def _settings_path(value: str | Path | None) -> Path:
    return Path(value).expanduser().resolve() if value else default_media_route_settings_path()


def _mediakit_cli_status() -> dict[str, Any]:
    # Lazy import avoids making settings inspection depend on the execution module.
    from .mediakit_cli_adapter import mediakit_cli_status

    return mediakit_cli_status()