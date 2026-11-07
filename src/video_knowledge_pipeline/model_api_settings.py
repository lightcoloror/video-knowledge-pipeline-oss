from __future__ import annotations

import base64
import ctypes
import ipaddress
import json
import os
import re
import shutil
import uuid
from ctypes import wintypes
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .media_capability_registry import media_capability_registry_status
from .model_route_settings import (
    CAPABILITIES,
    LEGACY_SETTINGS_SCHEMA,
    SETTINGS_V2_SCHEMA,
    TASK_CAPABILITIES,
    assign_profile_routes,
    delete_profile_routes,
    infer_profile_location,
    normalise_profile_capabilities,
    normalise_route_state,
    resolve_model_route,
)
from .model_provider_catalog import (
    ALLOWED_PROVIDERS,
    OCR_LITELLM_PROVIDERS,
    PROVIDER_PRESETS,
    provider_catalog_status,
    provider_preset,
    resolve_litellm_provider,
)
from .model_provider_onboarding import (
    free_screening_onboarding_status,
    key_once_onboarding_provider_ids,
    provider_onboarding_definition,
)
from .model_screening_lab import model_screening_lab_status
from .storage import write_json
from .utils import now_iso


SETTINGS_SCHEMA = SETTINGS_V2_SCHEMA
SECRETS_SCHEMA = "video_knowledge_pipeline.local_model_api_secrets.v1"
SETTINGS_ENV_VAR = "VKP_MODEL_API_SETTINGS_PATH"
SECRETS_ENV_VAR = "VKP_MODEL_API_SECRETS_PATH"

MODEL_TASKS = (
    "asr",
    "ocr",
    "document_visual",
    "semantic_frame",
    "temporal_sequence",
    "video_segment",
    "text_llm",
    "summary_rewrite",
    "transcript_correction",
)

ONLINE_SCREENING_ROUTE_PRESET_ID = "online-screening-v1"
ONLINE_SCREENING_ROUTE_TASK_PROFILES = {
    "asr": "groq-whisper-large-v3-turbo",
    "ocr": "mistral-ocr-4-0",
    "document_visual": "siliconflow-paddleocr-vl-1-5",
    "semantic_frame": "siliconflow-glm-4-1v-9b-thinking",
    "temporal_sequence": "google-gemini-3-6-flash",
    "video_segment": "google-gemini-3-6-flash",
    "text_llm": "ark-deepseek-v4-pro",
    "summary_rewrite": "ark-kimi-k2-6",
    "transcript_correction": "ark-glm-latest",
}
ONLINE_PRODUCTION_ROUTE_PRESET_ID = "online-production-existing-apis-v1"
ONLINE_PRODUCTION_ROUTE_TASK_PROFILES = {
    "asr": "groq-whisper-large-v3-turbo",
    "ocr": "mistral-ocr-4-0",
    "document_visual": "siliconflow-paddleocr-vl-1-5",
    "semantic_frame": "siliconflow-glm-4-1v-9b-thinking",
    "temporal_sequence": "google-gemini-3-6-flash",
    "video_segment": "google-gemini-3-6-flash",
    "text_llm": "google-gemini-3-6-flash",
    "summary_rewrite": "google-gemini-3-6-flash",
    "transcript_correction": "google-gemini-3-6-flash",
}

LOCAL_PRODUCTION_ROUTE_PRESET_ID = "local-production-v1"
LOCAL_PRODUCTION_ROUTE_TASK_PROFILES = {
    "document_visual": "local-lmstudio-qwen3-vl-8b",
    "semantic_frame": "local-lmstudio-qwen3-vl-8b",
    "temporal_sequence": "local-lmstudio-qwen3-vl-8b",
    "video_segment": "local-lmstudio-qwen3-vl-8b",
    "text_llm": "local-lmstudio-qwen3-5-9b",
    "summary_rewrite": "local-lmstudio-qwen3-5-9b",
    "transcript_correction": "local-lmstudio-qwen3-5-9b",
}
LOCAL_PRODUCTION_PROFILE_TEMPLATES = (
    {
        "id": "local-lmstudio-qwen3-vl-8b",
        "name": "LM Studio Qwen3-VL 8B",
        "provider": "local_qwen_vl",
        "litellm_provider": "openai",
        "adapter_backend": "builtin",
        "location": "local",
        "capabilities": ["vision"],
        "base_url": "http://127.0.0.1:1234/v1",
        "model": "qwen/qwen3-vl-8b",
        "timeout_seconds": 300,
        "enabled": True,
    },
    {
        "id": "local-lmstudio-qwen3-5-9b",
        "name": "LM Studio Qwen3.5 9B",
        "provider": "local_openai_compatible",
        "litellm_provider": "openai",
        "adapter_backend": "builtin",
        "location": "local",
        "capabilities": ["text"],
        "base_url": "http://127.0.0.1:1234/v1",
        "model": "qwen/qwen3.5-9b",
        "provider_options": {"reasoning_effort": "none", "response_format": "text", "max_tokens": 1200},
        "timeout_seconds": 300,
        "enabled": True,
    },
)

MODEL_API_ROUTE_PRESETS = {
    ONLINE_SCREENING_ROUTE_PRESET_ID: ONLINE_SCREENING_ROUTE_TASK_PROFILES,
    ONLINE_PRODUCTION_ROUTE_PRESET_ID: ONLINE_PRODUCTION_ROUTE_TASK_PROFILES,
    LOCAL_PRODUCTION_ROUTE_PRESET_ID: LOCAL_PRODUCTION_ROUTE_TASK_PROFILES,
}

ALLOWED_ADAPTER_BACKENDS = frozenset({"proxy", "legacy", "auto", "builtin", "litellm"})
PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
SECRET_FIELD_RE = re.compile(
    r"(?:api[_-]?key|(?:^|[_-])token(?:$|[_-])|secret|password|authorization)",
    re.IGNORECASE,
)
SAFE_SECRET_METADATA_FIELDS = frozenset({"secret_ref", "api_key_optional"})
PROVIDER_OPTION_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


UI_CONFIG_SCHEMA = "video_knowledge_pipeline.model_api_settings_ui.v1"


def default_ui_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "model-api-settings-ui.json"


def load_model_api_settings_ui_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = _path(config_path, default_ui_config_path())
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or str(data.get("schema") or "") != UI_CONFIG_SCHEMA:
        raise ValueError("invalid model API settings UI config")
    host = str(data.get("host") or "127.0.0.1").strip()
    if not _is_loopback_host(host):
        raise ValueError("model API settings UI host must be loopback")
    port = int(data.get("port") or 8767)
    if port < 1 or port > 65535:
        raise ValueError("model API settings UI port must be between 1 and 65535")
    path_value = "/" + str(data.get("path") or "/").strip().strip("/")
    return {"schema": UI_CONFIG_SCHEMA, "host": host, "port": port, "path": path_value}


def model_api_settings_ui_url(config_path: str | Path | None = None) -> str:
    configured = load_model_api_settings_ui_config(config_path)
    host = configured["host"]
    rendered_host = f"[{host}]" if ":" in host else host
    return f"http://{rendered_host}:{configured['port']}{configured['path']}"

class SecretStorageUnavailable(RuntimeError):
    pass


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_settings_path() -> Path:
    configured = str(os.environ.get(SETTINGS_ENV_VAR) or "").strip()
    return Path(configured).expanduser().resolve() if configured else project_root() / ".local" / "model-api-settings.json"


def default_secrets_path(*, settings_path: str | Path | None = None) -> Path:
    configured = str(os.environ.get(SECRETS_ENV_VAR) or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    settings = _path(settings_path, default_settings_path())
    return settings.with_name("model-api-secrets.json")


def load_model_api_settings(settings_path: str | Path | None = None) -> dict[str, Any]:
    path = _path(settings_path, default_settings_path())
    if not path.exists():
        return _empty_settings()
    data = json.loads(path.read_text(encoding="utf-8"))
    return _normalise_settings(data)


def configured_remote_destination_status(
    settings_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return exact HTTPS hosts from enabled remote profiles without reading secrets."""

    settings_file = _path(settings_path, default_settings_path())
    settings = load_model_api_settings(settings_file)
    destinations: set[str] = set()
    profile_ids: list[str] = []
    for row in settings["profiles"]:
        if not bool(row.get("enabled", True)):
            continue
        if str(row.get("location") or "") != "remote":
            continue
        parts = urlsplit(str(row.get("base_url") or ""))
        host = str(parts.hostname or "").strip().lower()
        if parts.scheme.lower() != "https" or not host or "*" in host:
            raise ValueError(
                f"enabled remote profile {row['id']} requires an exact HTTPS base_url"
            )
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError(
                f"enabled remote profile {row['id']} has an unsafe base_url"
            )
        loopback = host == "localhost"
        try:
            loopback = loopback or ipaddress.ip_address(host).is_loopback
        except ValueError:
            pass
        if loopback:
            raise ValueError(
                f"enabled remote profile {row['id']} cannot use a loopback host"
            )
        destinations.add(host)
        profile_ids.append(str(row["id"]))

    return {
        "schema": "video_knowledge_pipeline.configured_remote_destinations.v1",
        "settings_path": str(settings_file),
        "destinations": sorted(destinations),
        "profile_ids": sorted(profile_ids),
        "secrets_accessed": False,
        "api_keys_exposed": False,
        "remote_requests_made": False,
        "consent_still_required": True,
        "arbitrary_urls_allowed": False,
    }


def public_model_api_settings_status(
    settings_path: str | Path | None = None,
    secrets_path: str | Path | None = None,
) -> dict[str, Any]:
    settings_file = _path(settings_path, default_settings_path())
    secrets_file = _path(secrets_path, default_secrets_path(settings_path=settings_file))
    settings = load_model_api_settings(settings_file)
    secret_ids = set(_load_secret_document(secrets_file).get("items", {}))
    profiles = []
    routed_tasks: dict[str, list[str]] = {}
    for task, profile_id in settings["task_routes"].items():
        routed_tasks.setdefault(profile_id, []).append(task)
    for row in settings["profiles"]:
        auth_status = _profile_auth_status(row, secret_ids=secret_ids)
        profiles.append(
            {
                **row,
                "tasks": sorted(routed_tasks.get(row["id"], []), key=MODEL_TASKS.index),
                "api_key_configured": _profile_secret_id(row) in secret_ids,
                "credential_status": auth_status["status"],
                "auth_status": auth_status,
            }
        )
    route_status = _public_route_status(settings)
    onboarding = free_screening_onboarding_status(profiles, route_status)
    return {
        "schema": SETTINGS_SCHEMA,
        "settings_path": str(settings_file),
        "secrets_path": str(secrets_file),
        "settings_ui_url": model_api_settings_ui_url(),
        "profiles": profiles,
        "task_routes": dict(settings["task_routes"]),
        "route_bindings": {task: dict(row) for task, row in settings["route_bindings"].items()},
        "route_pools": [dict(row) for row in settings["route_pools"]],
        "route_status": route_status,
        "migrated_from": str(settings.get("migrated_from") or ""),
        "model_tasks": list(MODEL_TASKS),
        "provider_presets": [dict(row) for row in PROVIDER_PRESETS],
        "provider_catalog": provider_catalog_status(),
        "free_screening_onboarding": onboarding,
        "model_screening_lab": model_screening_lab_status(profiles, onboarding),
        "media_capability_catalog": media_capability_registry_status(),
        "secret_storage": {
            "kind": "windows_dpapi" if os.name == "nt" else "unavailable",
            "available": os.name == "nt",
            "plaintext_persisted": False,
        },
        "updated_at": settings.get("updated_at") or "",
        "execution_boundary": "Saving a profile does not authorize network egress; consent and trusted destination policy still apply.",
    }


def validate_model_api_profile(profile: dict[str, Any], tasks: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    return {
        "profile": _normalise_profile(profile, default_adapter_backend="proxy"),
        "tasks": _normalise_tasks(tasks if tasks is not None else profile.get("tasks")),
    }


def upsert_model_api_profile(
    profile: dict[str, Any],
    *,
    tasks: list[str] | tuple[str, ...] | None = None,
    api_key: str = "",
    remove_api_key: bool = False,
    settings_path: str | Path | None = None,
    secrets_path: str | Path | None = None,
) -> dict[str, Any]:
    settings_file = _path(settings_path, default_settings_path())
    secrets_file = _path(secrets_path, default_secrets_path(settings_path=settings_file))
    validated = validate_model_api_profile(profile, tasks)
    row = validated["profile"]
    selected_tasks = validated["tasks"]
    secret_id = _profile_secret_id(row)
    if remove_api_key and secret_id != row["id"]:
        raise ValueError("shared secret_ref cannot be removed through an alias profile")
    settings = load_model_api_settings(settings_file)
    rows = [existing for existing in settings["profiles"] if existing["id"] != row["id"]]
    rows.append(row)
    sorted_rows = sorted(rows, key=lambda item: (str(item.get("name") or "").casefold(), item["id"]))
    route_state = assign_profile_routes(
        settings,
        profiles=sorted_rows,
        profile_id=row["id"],
        selected_tasks=selected_tasks,
    )
    saved = {
        "schema": SETTINGS_SCHEMA,
        "profiles": sorted_rows,
        **route_state,
        "updated_at": now_iso(),
    }
    _backup_legacy_settings(settings_file)
    write_json(settings_file, saved)
    _restrict_file(settings_file)
    if remove_api_key:
        _delete_secret(secret_id, secrets_file)
    elif str(api_key or ""):
        _save_secret(secret_id, str(api_key), secrets_file)
    return public_model_api_settings_status(settings_file, secrets_file)


_ONBOARDING_CONFLICT_FIELDS = (
    "provider",
    "litellm_provider",
    "adapter_backend",
    "base_url",
    "model",
    "location",
    "capabilities",
    "rpm",
    "tpm",
    "max_parallel_requests",
)


def _onboarding_profile_from_template(template: dict[str, Any]) -> dict[str, Any]:
    candidate = {
        key: template[key]
        for key in (
            "id",
            "name",
            "provider",
            "litellm_provider",
            "provider_options",
            "adapter_backend",
            "location",
            "capabilities",
            "base_url",
            "model",
            "timeout_seconds",
            "rpm",
            "tpm",
            "max_parallel_requests",
        )
        if key in template
    }
    candidate["enabled"] = bool(template.get("install_enabled", True))
    return validate_model_api_profile(candidate, [])["profile"]


def _check_onboarding_profile_conflict(
    existing: dict[str, Any],
    candidate: dict[str, Any],
    template: dict[str, Any],
    *,
    allow_known_model_refresh: bool,
) -> dict[str, str] | None:
    changed = [
        field
        for field in _ONBOARDING_CONFLICT_FIELDS
        if existing.get(field) != candidate.get(field)
    ]
    previous_models = {str(value) for value in template.get("replaces_models") or []}
    if (
        allow_known_model_refresh
        and changed == ["model"]
        and str(existing.get("model") or "") in previous_models
    ):
        return {
            "profile_id": str(candidate["id"]),
            "from_model": str(existing.get("model") or ""),
            "to_model": str(candidate.get("model") or ""),
        }
    if changed:
        raise ValueError(
            f"onboarding profile conflict for {candidate['id']}: "
            + ", ".join(changed)
            + "; existing custom profile was not overwritten"
        )
    return None


def prepare_model_api_onboarding_bundles(
    provider_ids: list[str] | tuple[str, ...] | None = None,
    *,
    settings_path: str | Path | None = None,
    secrets_path: str | Path | None = None,
    refresh_known_models: bool = False,
) -> dict[str, Any]:
    """Prepare exact profiles while reading only encrypted credential metadata."""

    selected_ids = tuple(provider_ids or key_once_onboarding_provider_ids())
    if not selected_ids:
        raise ValueError("at least one exact onboarding provider is required")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("duplicate onboarding provider ids are not allowed")

    definitions: list[dict[str, Any]] = []
    validated_rows: list[dict[str, Any]] = []
    templates_by_id: dict[str, dict[str, Any]] = {}
    for provider_id in selected_ids:
        definition = provider_onboarding_definition(provider_id)
        templates = [
            dict(row)
            for row in definition.get("profile_templates") or []
            if isinstance(row, dict)
        ]
        if not templates:
            raise ValueError(
                f"provider {provider_id!r} requires explicit model selection "
                "and has no exact bundle"
            )
        definitions.append(definition)
        for template in templates:
            validated = _onboarding_profile_from_template(template)
            validated_rows.append(validated)
            templates_by_id[str(validated["id"])] = template

    profile_ids = [str(row["id"]) for row in validated_rows]
    if len(set(profile_ids)) != len(profile_ids):
        raise ValueError("onboarding bundles contain duplicate profile ids")

    settings_file = _path(settings_path, default_settings_path())
    secrets_file = _path(secrets_path, default_secrets_path(settings_path=settings_file))
    settings = load_model_api_settings(settings_file)
    secret_ids = set(_load_secret_document(secrets_file).get("items", {}))
    existing_by_id = {str(row["id"]): row for row in settings["profiles"]}
    provider_secret_refs: dict[tuple[str, str], str] = {}
    for existing in settings["profiles"]:
        provider = str(existing.get("provider") or "").casefold()
        base_url = str(existing.get("base_url") or "").rstrip("/")
        secret_ref = str(existing.get("secret_ref") or "")
        if (
            provider
            and base_url
            and secret_ref.startswith("dpapi:")
            and secret_ref.removeprefix("dpapi:") in secret_ids
        ):
            provider_secret_refs.setdefault((provider, base_url), secret_ref)
    reused_credential_refs: list[str] = []
    dangling_credential_refs: list[str] = []
    refreshed_models: list[dict[str, str]] = []
    for row in validated_rows:
        existing = existing_by_id.get(str(row["id"]))
        if existing is not None:
            refreshed = _check_onboarding_profile_conflict(
                existing,
                row,
                templates_by_id[str(row["id"])],
                allow_known_model_refresh=refresh_known_models,
            )
            if refreshed:
                refreshed_models.append(refreshed)
        existing_secret_ref = str((existing or {}).get("secret_ref") or "")
        secret_ref = (
            existing_secret_ref
            if existing_secret_ref.removeprefix("dpapi:") in secret_ids
            else ""
        )
        if existing_secret_ref and not secret_ref:
            dangling_credential_refs.append(str(row["id"]))
        if not secret_ref:
            provider_key = (
                str(row.get("provider") or "").casefold(),
                str(row.get("base_url") or "").rstrip("/"),
            )
            secret_ref = provider_secret_refs.get(provider_key, "")
            if secret_ref:
                reused_credential_refs.append(str(row["id"]))
        if secret_ref:
            row["secret_ref"] = secret_ref

    bundle_ids = set(profile_ids)
    profiles = [row for row in settings["profiles"] if str(row["id"]) not in bundle_ids]
    profiles.extend(validated_rows)
    profiles.sort(key=lambda row: (str(row.get("name") or "").casefold(), row["id"]))
    credential_ready = sorted(
        str(row["id"])
        for row in validated_rows
        if _profile_secret_id(row) in secret_ids
    )
    credential_missing = sorted(set(profile_ids) - set(credential_ready))
    route_state = normalise_route_state(
        profiles=profiles,
        legacy_routes={},
        route_pools=settings.get("route_pools"),
        route_bindings=settings.get("route_bindings"),
    )
    updated_at = now_iso()
    _backup_legacy_settings(settings_file)
    write_json(settings_file, {
        "schema": SETTINGS_SCHEMA,
        "profiles": profiles,
        **route_state,
        "updated_at": updated_at,
    })
    _restrict_file(settings_file)
    return {
        "schema": "video_knowledge_pipeline.model_api_onboarding_prepare.v1",
        "settings_path": str(settings_file),
        "provider_ids": [str(row["id"]) for row in definitions],
        "profile_ids": sorted(bundle_ids),
        "profile_count": len(bundle_ids),
        "route_configuration_changed": False,
        "network_calls": False,
        "known_models_refreshed": refreshed_models,
        "credential_reference_reused_profile_ids": sorted(reused_credential_refs),
        "credential_ready_profile_ids": credential_ready,
        "credential_missing_profile_ids": credential_missing,
        "dangling_credential_reference_profile_ids": sorted(dangling_credential_refs),
        "secrets_accessed": True,
        "secret_values_accessed": False,
        "secrets_decrypted": False,
        "saving_authorizes_egress": False,
        "updated_at": updated_at,
    }



def install_model_api_onboarding_bundle(
    provider_id: str,
    *,
    api_key: str,
    settings_path: str | Path | None = None,
    secrets_path: str | Path | None = None,
) -> dict[str, Any]:
    clean_secret = str(api_key or "").strip()
    if not clean_secret:
        raise ValueError("api_key is required for a key-once onboarding bundle")
    if len(clean_secret) > 16384 or any(character in clean_secret for character in "\r\n\x00"):
        raise ValueError("api_key must be a single safe value of at most 16384 characters")

    definition = provider_onboarding_definition(provider_id)
    from .model_provider_onboarding import validate_provider_onboarding_prefills

    validate_provider_onboarding_prefills()
    templates = [
        dict(row)
        for row in definition.get("profile_templates") or []
        if isinstance(row, dict)
    ]
    if not templates:
        raise ValueError("this provider requires explicit model selection and has no key-once bundle")

    validated_rows = [_onboarding_profile_from_template(template) for template in templates]
    templates_by_id = {str(template["id"]): template for template in templates}

    settings_file = _path(settings_path, default_settings_path())
    secrets_file = _path(secrets_path, default_secrets_path(settings_path=settings_file))
    settings = load_model_api_settings(settings_file)
    existing_by_id = {str(row["id"]): row for row in settings["profiles"]}
    for row in validated_rows:
        existing = existing_by_id.get(str(row["id"]))
        if existing is None:
            continue
        _check_onboarding_profile_conflict(
            existing,
            row,
            templates_by_id[str(row["id"])],
            allow_known_model_refresh=True,
        )

    bundle_ids = {str(row["id"]) for row in validated_rows}
    profiles = [
        row for row in settings["profiles"] if str(row["id"]) not in bundle_ids
    ]
    profiles.extend(validated_rows)
    profiles.sort(key=lambda row: (str(row.get("name") or "").casefold(), row["id"]))
    route_state = normalise_route_state(
        profiles=profiles,
        legacy_routes={},
        route_pools=settings.get("route_pools"),
        route_bindings=settings.get("route_bindings"),
    )
    updated_at = now_iso()
    saved = {
        "schema": SETTINGS_SCHEMA,
        "profiles": profiles,
        **route_state,
        "updated_at": updated_at,
    }

    secret_document = _load_secret_document(secrets_file)
    secret_items = dict(secret_document.get("items") or {})
    encrypted = {
        row["id"]: {
            "ciphertext": _protect_secret(clean_secret),
            "updated_at": updated_at,
        }
        for row in validated_rows
    }
    secret_items.update(encrypted)

    _backup_legacy_settings(settings_file)
    write_json(
        secrets_file,
        {
            "schema": SECRETS_SCHEMA,
            "items": secret_items,
            "updated_at": updated_at,
        },
    )
    _restrict_file(secrets_file)
    write_json(settings_file, saved)
    _restrict_file(settings_file)

    status = public_model_api_settings_status(settings_file, secrets_file)
    status["last_onboarding_install"] = {
        "provider_id": str(definition["id"]),
        "profile_ids": sorted(bundle_ids),
        "profile_count": len(bundle_ids),
        "route_configuration_changed": False,
        "network_calls": False,
        "saving_authorizes_egress": False,
        "secret_values_exposed": False,
    }
    return status

def delete_model_api_profile(
    profile_id: str,
    *,
    settings_path: str | Path | None = None,
    secrets_path: str | Path | None = None,
) -> dict[str, Any]:
    clean_id = _normalise_profile_id(profile_id)
    settings_file = _path(settings_path, default_settings_path())
    secrets_file = _path(secrets_path, default_secrets_path(settings_path=settings_file))
    settings = load_model_api_settings(settings_file)
    profiles = [row for row in settings["profiles"] if row["id"] != clean_id]
    route_state = delete_profile_routes(settings, profiles=profiles, profile_id=clean_id)
    saved = {
        "schema": SETTINGS_SCHEMA,
        "profiles": profiles,
        **route_state,
        "updated_at": now_iso(),
    }
    _backup_legacy_settings(settings_file)
    write_json(settings_file, saved)
    _restrict_file(settings_file)
    _delete_secret(clean_id, secrets_file)
    return public_model_api_settings_status(settings_file, secrets_file)

def replace_model_api_route_configuration(
    route_pools: list[dict[str, Any]],
    route_bindings: dict[str, dict[str, Any]],
    *,
    settings_path: str | Path | None = None,
    secrets_path: str | Path | None = None,
) -> dict[str, Any]:
    settings_file = _path(settings_path, default_settings_path())
    secrets_file = _path(secrets_path, default_secrets_path(settings_path=settings_file))
    settings = load_model_api_settings(settings_file)
    route_state = normalise_route_state(
        profiles=settings["profiles"],
        legacy_routes={},
        route_pools=route_pools,
        route_bindings=route_bindings,
    )
    saved = {
        "schema": SETTINGS_SCHEMA,
        "profiles": settings["profiles"],
        **route_state,
        "updated_at": now_iso(),
    }
    _backup_legacy_settings(settings_file)
    write_json(settings_file, saved)
    _restrict_file(settings_file)
    return public_model_api_settings_status(settings_file, secrets_file)


def apply_model_api_route_preset(
    preset_id: str,
    *,
    settings_path: str | Path | None = None,
    secrets_path: str | Path | None = None,
) -> dict[str, Any]:
    clean_preset = str(preset_id or "").strip().lower()
    task_profiles = MODEL_API_ROUTE_PRESETS.get(clean_preset)
    if task_profiles is None:
        raise ValueError(f"unsupported model API route preset: {preset_id!r}")

    settings_file = _path(settings_path, default_settings_path())
    secrets_file = _path(secrets_path, default_secrets_path(settings_path=settings_file))
    settings = load_model_api_settings(settings_file)
    preset_location = (
        "local" if clean_preset == LOCAL_PRODUCTION_ROUTE_PRESET_ID else "remote"
    )
    target_pool_location = (
        "local_only" if preset_location == "local" else "remote_approved"
    )
    opposite_pool_location = (
        "remote_approved" if preset_location == "local" else "local_only"
    )

    profile_rows = [dict(row) for row in settings["profiles"]]
    if clean_preset == LOCAL_PRODUCTION_ROUTE_PRESET_ID:
        existing_by_id = {str(row["id"]): row for row in profile_rows}
        installed_ids: set[str] = set()
        for template in LOCAL_PRODUCTION_PROFILE_TEMPLATES:
            candidate = _normalise_profile(template, default_adapter_backend="proxy")
            profile_id = str(candidate["id"])
            existing = existing_by_id.get(profile_id)
            if existing is not None:
                changed = [
                    field
                    for field in (
                        *_ONBOARDING_CONFLICT_FIELDS,
                        "timeout_seconds",
                        "enabled",
                    )
                    if existing.get(field) != candidate.get(field)
                ]
                if changed:
                    raise ValueError(
                        f"local production profile conflict for {profile_id}: "
                        + ", ".join(changed)
                        + "; existing custom profile was not overwritten"
                    )
            installed_ids.add(profile_id)
            existing_by_id[profile_id] = candidate
        profile_rows = sorted(
            existing_by_id.values(),
            key=lambda row: (str(row.get("name") or "").casefold(), row["id"]),
        )

    profiles = {str(row["id"]): row for row in profile_rows}
    preserved_pools = [
        dict(row)
        for row in settings.get("route_pools") or []
        if str(row.get("location") or "") == opposite_pool_location
    ]
    opposite_pool_ids = {str(row["id"]) for row in preserved_pools}
    pools = list(preserved_pools)
    pool_ids_by_deployment: dict[tuple[str, str], str] = {}
    bindings: dict[str, dict[str, str]] = {}
    destinations: set[str] = set()

    for task, profile_id in task_profiles.items():
        profile = profiles.get(profile_id)
        if not profile:
            raise ValueError(f"route preset profile not found: {profile_id}")
        if not bool(profile.get("enabled", True)):
            raise ValueError(f"route preset profile is disabled: {profile_id}")
        capability = TASK_CAPABILITIES[task]
        if capability not in profile.get("capabilities", []):
            raise ValueError(
                f"route preset profile {profile_id} does not support {capability}"
            )
        if str(profile.get("location") or "") != preset_location:
            raise ValueError(
                f"route preset profile must be {preset_location}: {profile_id}"
            )

        pool_key = (profile_id, capability)
        pool_id = pool_ids_by_deployment.get(pool_key)
        if not pool_id:
            pool_id = (
                f"pool-{clean_preset.removesuffix('-v1')}-"
                f"{profile_id}-{capability}"
            )
            pool_ids_by_deployment[pool_key] = pool_id
            pools.append(
                {
                    "id": pool_id,
                    "name": f"{profile['name']} / {capability}",
                    "location": target_pool_location,
                    "capability": capability,
                    "deployments": [profile_id],
                    "retry_policy": {
                        "max_retries": 0 if preset_location == "local" else 1,
                        "timeout_seconds": int(
                            profile.get("timeout_seconds") or 120
                        ),
                        "cooldown_seconds": (
                            0 if preset_location == "local" else 5
                        ),
                    },
                }
            )
        current = dict((settings.get("route_bindings") or {}).get(task) or {})
        opposite_key = (
            "remote_pool_id" if preset_location == "local" else "local_pool_id"
        )
        opposite_pool_id = str(current.get(opposite_key) or "")
        binding = {
            "default_location": preset_location,
            "local_pool_id": "",
            "remote_pool_id": "",
        }
        binding[f"{preset_location}_pool_id"] = pool_id
        if opposite_pool_id in opposite_pool_ids:
            binding[opposite_key] = opposite_pool_id
        bindings[task] = binding
        if preset_location == "remote":
            host = str(
                urlsplit(str(profile.get("base_url") or "")).hostname or ""
            ).lower()
            if host:
                destinations.add(host)

    route_state = normalise_route_state(
        profiles=profile_rows,
        legacy_routes={},
        route_pools=pools,
        route_bindings=bindings,
    )
    saved = {
        "schema": SETTINGS_SCHEMA,
        "profiles": profile_rows,
        **route_state,
        "updated_at": now_iso(),
    }
    _backup_legacy_settings(settings_file)
    write_json(settings_file, saved)
    _restrict_file(settings_file)
    status = public_model_api_settings_status(settings_file, secrets_file)
    status["last_route_preset"] = {
        "preset_id": clean_preset,
        "task_count": len(task_profiles),
        "pool_count": len(pool_ids_by_deployment),
        "remote_destinations": sorted(destinations),
        "single_deployment_pools": True,
        "automatic_cross_destination_fallback": False,
        "saving_authorizes_egress": False,
    }
    if clean_preset == LOCAL_PRODUCTION_ROUTE_PRESET_ID:
        status["last_route_preset"].update(
            {
                "local_media_tasks": {
                    "asr": {
                        "primary": "sensevoice",
                        "secondary": "qwen3-asr-1.7b",
                        "device": "cuda",
                    },
                    "ocr": {
                        "primary": "ebook_markdown_pipeline",
                        "engine": "rapidocr",
                        "device": "cuda",
                    },
                },
                "remote_profiles_selected": False,
                "remote_requests_made": False,
            }
        )
    return status

def resolve_model_api_route(
    task: str,
    *,
    execution_location: str = "",
    settings_path: str | Path | None = None,
) -> dict[str, Any]:
    settings_file = _path(settings_path, default_settings_path())
    settings = load_model_api_settings(settings_file)
    task_name = _normalise_task(task)
    return resolve_model_route(settings, task=task_name, execution_location=execution_location)


def resolve_model_api_provider_config(
    task: str,
    explicit: dict[str, Any] | None = None,
    *,
    settings_path: str | Path | None = None,
    secrets_path: str | Path | None = None,
    execution_location: str = "",
) -> dict[str, Any]:
    explicit_cfg = dict(explicit or {})
    settings_file = _path(settings_path, default_settings_path())
    settings = load_model_api_settings(settings_file)
    task_name = _normalise_task(task)
    try:
        route = resolve_model_route(settings, task=task_name, execution_location=execution_location)
    except ValueError:
        return explicit_cfg
    profile_id = str(route["deployments"][0]["id"])
    profile = next((row for row in settings["profiles"] if row["id"] == profile_id and row.get("enabled", True)), None)
    if not profile:
        return explicit_cfg
    explicit_provider = str(explicit_cfg.get("provider") or "").strip()
    stored_provider = str(profile.get("provider") or "").strip()
    explicit_route = bool(
        str(explicit_cfg.get("route_id") or "").strip()
        or str(explicit_cfg.get("route_revision") or "").strip()
    )
    if explicit_provider and not explicit_route:
        return explicit_cfg
    if explicit_provider and explicit_provider != stored_provider:
        return explicit_cfg
    resolved = {
        key: profile[key]
        for key in (
            "provider",
            "litellm_provider",
            "auth_mode",
            "provider_options",
            "api_key_optional",
            "environment_bindings",
            "base_url",
            "model",
            "adapter_backend",
            "timeout_seconds",
            "location",
            "rpm",
            "tpm",
            "max_parallel_requests",
            "capabilities",
        )
        if profile.get(key) not in (None, "")
    }
    secrets_file = _path(secrets_path, default_secrets_path(settings_path=settings_file))
    if str(resolved.get("adapter_backend") or "") != "proxy":
        secret = _read_secret(_profile_secret_id(profile), secrets_file)
        if secret:
            resolved["api_key"] = secret
            resolved["api_key_source"] = "local_dpapi"
    resolved.update(
        {
            "profile_id": profile_id,
            "route_id": route["route_id"],
            "route_revision": route["route_revision"],
            "virtual_model": route["virtual_model"],
            "execution_location": route["execution_location"],
        }
    )
    resolved.update(explicit_cfg)
    return resolved

def _public_route_status(settings: dict[str, Any]) -> list[dict[str, Any]]:
    checked_at = now_iso()
    gateway_health = _gateway_health_snapshot()
    allowed_raw = str(os.environ.get("VKP_MODEL_CONNECTOR_ALLOWED_DESTINATIONS") or "").strip()
    allowed = {
        item.strip().lower().rstrip("/")
        for line in allowed_raw.replace("\r", "\n").split("\n")
        for item in line.split(",")
        if item.strip()
    }
    rows: list[dict[str, Any]] = []
    for task in MODEL_TASKS:
        binding = (settings.get("route_bindings") or {}).get(task)
        if not isinstance(binding, dict):
            continue
        for location in ("local", "remote"):
            if not str(binding.get(f"{location}_pool_id") or ""):
                continue
            try:
                route = resolve_model_route(settings, task=task, execution_location=location)
            except ValueError as exc:
                rows.append(
                    {
                        "task": task,
                        "execution_location": location,
                        "status": "invalid",
                        "error": str(exc),
                        "last_checked_at": checked_at,
                    }
                )
                continue
            backends = {str(row.get("adapter_backend") or "") for row in route["deployments"]}
            health = gateway_health if "proxy" in backends else "legacy_not_probed"
            destinations = sorted(
                {
                    str(urlsplit(str(row.get("base_url") or "")).hostname or "").lower()
                    for row in route["deployments"]
                    if str(row.get("base_url") or "")
                }
            )
            if location == "local":
                allowlist_status = "not_required"
            elif not allowed:
                allowlist_status = "unknown"
            else:
                allowlist_status = "approved" if all(host in allowed for host in destinations) else "blocked"
            rows.append(
                {
                    "task": task,
                    "capability": TASK_CAPABILITIES[task],
                    "execution_location": location,
                    "is_default": str(binding.get("default_location") or "") == location,
                    "route_id": route["route_id"],
                    "route_revision": route["route_revision"],
                    "virtual_model": route["virtual_model"],
                    "deployments": [str(row.get("id") or "") for row in route["deployments"]],
                    "health_status": health,
                    "last_checked_at": checked_at,
                    "estimated_cost": "unknown",
                    "consent_required": location == "remote",
                    "allowlist_status": allowlist_status,
                    "destinations": destinations,
                }
            )
    return rows


def _gateway_health_snapshot() -> str:
    try:
        from .model_gateway import model_gateway_status

        status = model_gateway_status()
        checks = {str(row.get("key") or ""): row for row in status.get("checks") or [] if isinstance(row, dict)}
        if bool((checks.get("live_listener") or {}).get("ok")):
            return "ready" if str(status.get("http_status") or "") == "ready" else "degraded"
        return "stopped"
    except (OSError, RuntimeError, ValueError):
        return "unknown"

def _empty_settings() -> dict[str, Any]:
    return {
        "schema": SETTINGS_SCHEMA,
        "profiles": [],
        "task_routes": {},
        "route_bindings": {},
        "route_pools": [],
        "updated_at": "",
    }


def _normalise_settings(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("model API settings must be a JSON object")
    schema = str(value.get("schema") or "")
    if schema not in {SETTINGS_SCHEMA, LEGACY_SETTINGS_SCHEMA}:
        raise ValueError(f"unsupported model API settings schema: {value.get('schema')!r}")
    profiles_raw = value.get("profiles")
    routes_raw = value.get("task_routes")
    if not isinstance(profiles_raw, list) or not isinstance(routes_raw, dict):
        raise ValueError("model API settings require profiles and task_routes")
    profiles = [
        _normalise_profile(
            row,
            default_adapter_backend="legacy" if schema == LEGACY_SETTINGS_SCHEMA else "proxy",
        )
        for row in profiles_raw
    ]
    profile_ids = {row["id"] for row in profiles}
    if len(profile_ids) != len(profiles):
        raise ValueError("model API profile ids must be unique")
    legacy_routes: dict[str, str] = {}
    inline_bindings: dict[str, dict[str, Any]] = {}
    for raw_task, raw_route in routes_raw.items():
        task = _normalise_task(str(raw_task))
        if isinstance(raw_route, dict):
            inline_bindings[task] = dict(raw_route)
            continue
        profile_id = _normalise_profile_id(str(raw_route))
        if profile_id in profile_ids:
            legacy_routes[task] = profile_id
    route_bindings = value.get("route_bindings")
    if route_bindings is None and inline_bindings:
        route_bindings = inline_bindings
    authoritative_bindings = route_bindings if schema == SETTINGS_SCHEMA else None
    legacy_for_state = legacy_routes if schema == LEGACY_SETTINGS_SCHEMA or authoritative_bindings is None else {}
    state = normalise_route_state(
        profiles=profiles,
        legacy_routes=legacy_for_state,
        route_pools=value.get("route_pools") if schema == SETTINGS_SCHEMA else None,
        route_bindings=authoritative_bindings,
    )
    result = {
        "schema": SETTINGS_SCHEMA,
        "profiles": profiles,
        **state,
        "updated_at": str(value.get("updated_at") or ""),
    }
    if schema == LEGACY_SETTINGS_SCHEMA:
        result["migrated_from"] = LEGACY_SETTINGS_SCHEMA
    return result


def _backup_legacy_settings(path: Path) -> None:
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict) or str(payload.get("schema") or "") != LEGACY_SETTINGS_SCHEMA:
        return
    backup = path.with_name(path.name + ".v1.bak")
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        _restrict_file(backup)

def _normalise_profile(value: Any, *, default_adapter_backend: str = "proxy") -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("model API profile must be an object")
    for key in value:
        if str(key) not in SAFE_SECRET_METADATA_FIELDS and SECRET_FIELD_RE.search(str(key)):
            raise ValueError("API keys and other secrets must not be stored in profile fields")
    profile_id = _normalise_profile_id(str(value.get("id") or f"profile-{uuid.uuid4().hex[:12]}"))
    secret_ref = _normalise_secret_ref(value.get("secret_ref"), profile_id=profile_id)
    name = str(value.get("name") or "").strip()
    if not name or len(name) > 100:
        raise ValueError("profile name is required and must be at most 100 characters")
    provider = str(value.get("provider") or "").strip().lower().replace("-", "_")
    if provider not in ALLOWED_PROVIDERS:
        raise ValueError(f"unsupported provider: {provider!r}")
    preset = provider_preset(provider)
    provider_options = _normalise_provider_options(value.get("provider_options"), preset)
    if provider == "groq_asr":
        provider_options.setdefault("asr_timestamp_granularity", "word")
    litellm_provider = resolve_litellm_provider(
        provider,
        str(value.get("litellm_provider") or ""),
    )
    adapter_backend = str(value.get("adapter_backend") or default_adapter_backend).strip().lower()
    if adapter_backend not in ALLOWED_ADAPTER_BACKENDS:
        raise ValueError(f"unsupported adapter_backend: {adapter_backend!r}")
    base_url = _validate_base_url(str(value.get("base_url") or "").strip())
    location = infer_profile_location(str(value.get("location") or ""), base_url=base_url)
    if value.get("capabilities") is None:
        capabilities = (
            list(CAPABILITIES)
            if adapter_backend != "proxy"
            else list(preset.get("default_capabilities") or [])
        )
    else:
        capabilities = normalise_profile_capabilities(
            value.get("capabilities"), provider=provider
        )
    supported_capabilities = set(preset.get("supported_capabilities") or [])
    unsupported_capabilities = set(capabilities) - supported_capabilities
    if (
        adapter_backend == "proxy"
        and "ocr" in unsupported_capabilities
        and litellm_provider not in OCR_LITELLM_PROVIDERS
    ):
        raise ValueError(
            "proxy OCR profiles must use a LiteLLM OCR provider or an explicit "
            "Mistral-compatible thin adapter"
        )
    if adapter_backend == "proxy" and unsupported_capabilities:
        raise ValueError(
            f"provider {provider!r} does not declare capabilities: "
            + ", ".join(sorted(unsupported_capabilities))
        )
    if (
        adapter_backend == "proxy"
        and "ocr" in capabilities
        and litellm_provider not in OCR_LITELLM_PROVIDERS
    ):
        raise ValueError(
            "proxy OCR profiles must use provider=mistral or provider=mistral_compatible_ocr; "
            "unsupported OCR providers require an explicit Mistral-compatible thin adapter"
        )
    model = str(value.get("model") or "").strip()
    if len(model) > 300:
        raise ValueError("model must be at most 300 characters")
    enabled = bool(value.get("enabled", True))
    if enabled and not base_url:
        raise ValueError("enabled model API profiles require base_url")
    if enabled and not model:
        raise ValueError("enabled model API profiles require model")
    timeout_seconds = int(value.get("timeout_seconds") or 120)
    if timeout_seconds < 1 or timeout_seconds > 3600:
        raise ValueError("timeout_seconds must be between 1 and 3600")
    rpm = _optional_capacity_limit(value.get("rpm"), key="rpm", maximum=1_000_000)
    tpm = _optional_capacity_limit(
        value.get("tpm"), key="tpm", maximum=100_000_000
    )
    max_parallel_requests = _optional_capacity_limit(
        value.get("max_parallel_requests"),
        key="max_parallel_requests",
        maximum=128,
    )
    return {
        "id": profile_id,
        "name": name,
        "provider": provider,
        "litellm_provider": litellm_provider,
        "auth_mode": str(preset.get("auth_mode") or "api_key_dpapi"),
        "api_key_optional": bool(preset.get("api_key_optional")) or location == "local",
        "provider_options": provider_options,
        "required_provider_options": list(preset.get("required_provider_options") or []),
        "environment_bindings": [dict(row) for row in preset.get("environment_bindings") or []],
        "adapter_backend": adapter_backend,
        "base_url": base_url,
        "model": model,
        "location": location,
        "capabilities": capabilities,
        "secret_ref": secret_ref,
        "timeout_seconds": timeout_seconds,
        "enabled": enabled,
        **({"rpm": rpm} if rpm is not None else {}),
        **({"tpm": tpm} if tpm is not None else {}),
        **(
            {"max_parallel_requests": max_parallel_requests}
            if max_parallel_requests is not None
            else {}
        ),
    }


def _optional_capacity_limit(value: Any, *, key: str, maximum: int) -> int | None:
    if value in (None, ""):
        return None
    number = int(value)
    if number < 1 or number > maximum:
        raise ValueError(f"{key} must be between 1 and {maximum}")
    return number


def _normalise_provider_options(value: Any, preset: dict[str, Any]) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError("provider_options must be an object")
    allowed = {str(key) for key in preset.get("allowed_provider_options") or []}
    provider = str(preset.get("provider") or "")
    raw_options = dict(value)
    if provider == "siliconflow" and "thinking_mode" in raw_options:
        legacy_mode = str(raw_options.pop("thinking_mode") or "").strip().lower()
        if legacy_mode not in {"enabled", "disabled"}:
            raise ValueError(
                "SiliconFlow does not support thinking_mode=auto; use the official "
                "enable_thinking boolean"
            )
        migrated_value = legacy_mode == "enabled"
        if (
            "enable_thinking" in raw_options
            and raw_options["enable_thinking"] is not migrated_value
        ):
            raise ValueError(
                "conflicting SiliconFlow thinking options; keep only enable_thinking"
            )
        raw_options["enable_thinking"] = migrated_value
    result: dict[str, Any] = {}
    for raw_key, raw_value in raw_options.items():
        key = str(raw_key or "").strip()
        if not PROVIDER_OPTION_KEY_RE.fullmatch(key) or SECRET_FIELD_RE.search(key):
            raise ValueError("provider_options keys must be safe non-secret identifiers")
        if key not in allowed:
            raise ValueError(f"provider option is not allowed for this preset: {key}")
        if not isinstance(raw_value, (str, int, bool)) or isinstance(raw_value, float):
            raise ValueError("provider option values must be strings, integers, or booleans")
        if key == "thinking_mode":
            clean_mode = str(raw_value or "").strip().lower()
            if clean_mode not in {"enabled", "disabled", "auto"}:
                raise ValueError(
                    "thinking_mode must be one of: enabled, disabled, auto"
                )
            result[key] = clean_mode
            continue
        if key == "response_format":
            clean_format = str(raw_value or "").strip().lower()
            if clean_format not in {"json_object", "text"}:
                raise ValueError(
                    "response_format must be one of: json_object, text"
                )
            result[key] = clean_format
            continue
        if key == "asr_timestamp_granularity":
            clean_granularity = str(raw_value or "").strip().lower()
            if clean_granularity not in {"segment", "word"}:
                raise ValueError(
                    "asr_timestamp_granularity must be one of: segment, word"
                )
            result[key] = clean_granularity
            continue
        if key in {"enable_thinking", "stream"}:
            if not isinstance(raw_value, bool):
                raise ValueError(f"{key} must be a boolean")
            result[key] = raw_value
            continue
        if key in {"thinking_budget", "max_tokens"}:
            if isinstance(raw_value, bool) or not isinstance(raw_value, int):
                raise ValueError(f"{key} must be an integer")
            minimum, maximum = (
                (128, 32768) if key == "thinking_budget" else (1, 131072)
            )
            if raw_value < minimum or raw_value > maximum:
                raise ValueError(f"{key} must be between {minimum} and {maximum}")
            result[key] = raw_value
            continue
        if isinstance(raw_value, str):
            clean = raw_value.strip()
            if len(clean) > 500 or any(character in clean for character in "\r\n\x00"):
                raise ValueError("provider option string values must be at most 500 safe characters")
            if clean.startswith("os.environ/"):
                raise ValueError("provider option environment references are catalog-controlled")
            result[key] = clean
        else:
            result[key] = raw_value
    return {key: result[key] for key in sorted(result)}


def _profile_auth_status(
    profile: dict[str, Any],
    *,
    secret_ids: set[str] | frozenset[str],
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    environment = os.environ if environ is None else environ
    auth_mode = str(profile.get("auth_mode") or "api_key_dpapi")
    secret_id = _profile_secret_id(profile)
    bindings = [dict(row) for row in profile.get("environment_bindings") or [] if isinstance(row, dict)]
    required_env = sorted(
        str(row.get("env") or "") for row in bindings if row.get("required") and str(row.get("env") or "")
    )
    missing_env = [name for name in required_env if not str(environment.get(name) or "").strip()]
    if auth_mode == "external_environment":
        status = "ready" if not missing_env else "missing_environment"
    elif bool(profile.get("api_key_optional")):
        status = "not_required" if secret_id not in secret_ids else "ready"
    else:
        status = "ready" if secret_id in secret_ids else "missing_api_key"
    return {
        "status": status,
        "auth_mode": auth_mode,
        "required_environment": required_env,
        "missing_environment": missing_env,
        "secret_values_exposed": False,
    }


def _normalise_tasks(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("tasks must be a list")
    tasks = []
    for item in value:
        task = _normalise_task(str(item))
        if task not in tasks:
            tasks.append(task)
    return tasks


def _normalise_task(value: str) -> str:
    task = str(value or "").strip().lower().replace("-", "_")
    aliases = {"vision": "semantic_frame", "temporal": "temporal_sequence", "summary": "summary_rewrite", "text": "text_llm"}
    task = aliases.get(task, task)
    if task not in MODEL_TASKS:
        raise ValueError(f"unsupported model task: {value!r}")
    return task


def _normalise_profile_id(value: str) -> str:
    profile_id = str(value or "").strip().lower()
    if not PROFILE_ID_RE.fullmatch(profile_id):
        raise ValueError("profile id must contain only lowercase letters, digits, underscore, or hyphen")
    return profile_id


def _normalise_secret_ref(value: Any, *, profile_id: str) -> str:
    reference = str(value or f"dpapi:{profile_id}").strip()
    if not reference.startswith("dpapi:"):
        raise ValueError("secret_ref must use dpapi:<profile-id>")
    secret_id = _normalise_profile_id(reference.removeprefix("dpapi:"))
    return f"dpapi:{secret_id}"


def _profile_secret_id(profile: dict[str, Any]) -> str:
    profile_id = _normalise_profile_id(str(profile.get("id") or ""))
    return _normalise_secret_ref(profile.get("secret_ref"), profile_id=profile_id).removeprefix(
        "dpapi:"
    )


def _validate_base_url(value: str) -> str:
    if not value:
        return ""
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or "*" in parsed.hostname
    ):
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain credentials, query parameters, or fragments")
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise ValueError("remote base_url must use HTTPS; HTTP is allowed only for loopback hosts")
    return value.rstrip("/")


def _is_loopback_host(host: str) -> bool:
    value = str(host or "").strip().lower()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _load_secret_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema": SECRETS_SCHEMA, "items": {}, "updated_at": ""}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or str(data.get("schema") or "") != SECRETS_SCHEMA or not isinstance(data.get("items"), dict):
        raise ValueError("invalid model API secret store")
    return data


def _save_secret(profile_id: str, secret: str, path: Path) -> None:
    document = _load_secret_document(path)
    items = dict(document.get("items") or {})
    items[profile_id] = {"ciphertext": _protect_secret(secret), "updated_at": now_iso()}
    write_json(path, {"schema": SECRETS_SCHEMA, "items": items, "updated_at": now_iso()})
    _restrict_file(path)


def _read_secret(profile_id: str, path: Path) -> str:
    item = _load_secret_document(path).get("items", {}).get(profile_id)
    if not isinstance(item, dict) or not str(item.get("ciphertext") or ""):
        return ""
    return _unprotect_secret(str(item["ciphertext"]))


def _delete_secret(profile_id: str, path: Path) -> None:
    if not path.exists():
        return
    document = _load_secret_document(path)
    items = dict(document.get("items") or {})
    if profile_id not in items:
        return
    items.pop(profile_id, None)
    write_json(path, {"schema": SECRETS_SCHEMA, "items": items, "updated_at": now_iso()})
    _restrict_file(path)


def _protect_secret(secret: str) -> str:
    if os.name != "nt":
        raise SecretStorageUnavailable("Windows DPAPI is required to persist API keys")
    payload = secret.encode("utf-8")
    in_buffer = ctypes.create_string_buffer(payload)
    in_blob = _DataBlob(len(payload), ctypes.cast(in_buffer, ctypes.POINTER(ctypes.c_ubyte)))
    out_blob = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    if not crypt32.CryptProtectData(ctypes.byref(in_blob), "VKP model API key", None, None, None, 0x1, ctypes.byref(out_blob)):
        raise ctypes.WinError()
    try:
        encrypted = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        return base64.b64encode(encrypted).decode("ascii")
    finally:
        kernel32.LocalFree(out_blob.pbData)


def _unprotect_secret(ciphertext: str) -> str:
    if os.name != "nt":
        raise SecretStorageUnavailable("Windows DPAPI is required to read API keys")
    encrypted = base64.b64decode(ciphertext.encode("ascii"), validate=True)
    in_buffer = ctypes.create_string_buffer(encrypted)
    in_blob = _DataBlob(len(encrypted), ctypes.cast(in_buffer, ctypes.POINTER(ctypes.c_ubyte)))
    out_blob = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    if not crypt32.CryptUnprotectData(ctypes.byref(in_blob), None, None, None, None, 0x1, ctypes.byref(out_blob)):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData).decode("utf-8")
    finally:
        kernel32.LocalFree(out_blob.pbData)


def _restrict_file(path: Path) -> None:
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _path(value: str | Path | None, fallback: Path) -> Path:
    return Path(value).expanduser().resolve() if value is not None and str(value).strip() else fallback.resolve()
