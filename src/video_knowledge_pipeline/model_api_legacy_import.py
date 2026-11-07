from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import model_api_settings as settings_module
from .model_api_settings import (
    SECRETS_SCHEMA,
    SETTINGS_SCHEMA,
    default_secrets_path,
    default_settings_path,
    load_model_api_settings,
    validate_model_api_profile,
)
from .model_route_settings import assign_profile_routes
from .storage import replace_file_with_retry, write_json
from .utils import now_iso

IMPORT_SCHEMA = "video_knowledge_pipeline.model_api_legacy_import.v1"
REPORT_FILENAME = "model-api-legacy-import-report.json"
ARK_PROFILE_ID = "remote-ark"
LOCAL_VLM_PROFILE_ID = "local-qwen-vl"
ARK_HOST = "ark.cn-beijing.volces.com"
ARK_SECRET_NAMES = ("ARK_API_KEY", "VOLCENGINE_API_KEY", "LLM_API_KEY")
ARK_TASKS = (
    "document_visual",
    "semantic_frame",
    "temporal_sequence",
    "video_segment",
    "text_llm",
    "summary_rewrite",
    "transcript_correction",
)
LOCAL_VLM_TASKS = (
    "document_visual",
    "semantic_frame",
    "temporal_sequence",
    "video_segment",
)
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MAX_ENV_SOURCE_BYTES = 1024 * 1024
MAX_JSON_SOURCE_BYTES = 5 * 1024 * 1024


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def safe_import_legacy_model_api_settings(
    *,
    bundle_dir: str | Path,
    env_files: list[str | Path] | tuple[str | Path, ...] | None = None,
    settings_path: str | Path | None = None,
    secrets_path: str | Path | None = None,
    report_path: str | Path | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    root = project_root().resolve()
    settings_file = _path(settings_path, default_settings_path())
    secrets_file = _path(secrets_path, default_secrets_path(settings_path=settings_file))
    report_file = _path(report_path, settings_file.with_name(REPORT_FILENAME))
    try:
        discovery, candidates, secret_values = _discover_legacy_sources(
            bundle_dir=bundle_dir,
            env_files=env_files,
            allowed_root=root,
        )
        proposed = _build_settings(candidates)
        target = _inspect_target(settings_file, secrets_file, proposed, set(secret_values))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _blocked_result(
            execute=execute,
            settings_file=settings_file,
            secrets_file=secrets_file,
            report_file=report_file,
            reason="source_or_target_invalid",
            error=str(exc),
        )

    public_candidates = [_public_candidate(row) for row in candidates]
    base_result: dict[str, Any] = {
        "schema": IMPORT_SCHEMA,
        "status": "planned",
        "execute": bool(execute),
        "settings_path": str(settings_file),
        "secrets_path": str(secrets_file),
        "report_path": str(report_file),
        "sources": discovery["sources"],
        "profiles": public_candidates,
        "profile_count": len(public_candidates),
        "route_pool_count": len(proposed["route_pools"]),
        "task_binding_count": len(proposed["route_bindings"]),
        "excluded": discovery["excluded"],
        "ignored_legacy_authorizations": discovery["ignored_legacy_authorizations"],
        "manual_configuration_required": ["asr", "ocr"],
        "target": target,
        "security": {
            "preview_only_by_default": True,
            "source_files_modified": False,
            "plaintext_secrets_in_result": False,
            "plaintext_secrets_persisted": False,
            "secret_storage": "windows_dpapi",
            "old_consents_reused": False,
            "network_calls": 0,
            "remote_authorization_granted": False,
        },
        "updated_at": now_iso(),
    }
    if not candidates:
        return {**base_result, "status": "blocked", "reason": "no_safe_import_candidates"}
    if target["status"] == "already_imported":
        result = {
            **base_result,
            "status": "already_imported",
            "verification": {
                "profile_count": len(proposed["profiles"]),
                "route_pool_count": len(proposed["route_pools"]),
                "task_binding_count": len(proposed["route_bindings"]),
                "secret_ids": sorted(secret_values),
                "network_calls": 0,
            },
        }
        if execute:
            write_json(report_file, result)
            settings_module._restrict_file(report_file)
        return result
    if target["status"] != "empty":
        return {**base_result, "status": "blocked", "reason": str(target["status"])}
    if not execute:
        return base_result

    try:
        encrypted_items = {
            profile_id: {
                "ciphertext": settings_module._protect_secret(secret),
                "updated_at": now_iso(),
            }
            for profile_id, secret in secret_values.items()
        }
        saved_settings = {**proposed, "updated_at": now_iso()}
        saved_secrets = {
            "schema": SECRETS_SCHEMA,
            "items": encrypted_items,
            "updated_at": now_iso(),
        }
        verified_settings, verified_secrets = _write_settings_and_secrets(
            settings_file=settings_file,
            settings=saved_settings,
            secrets_file=secrets_file,
            secrets=saved_secrets,
            required_secret_ids=set(secret_values),
        )
    except (OSError, RuntimeError, ValueError, settings_module.SecretStorageUnavailable) as exc:
        return {
            **base_result,
            "status": "blocked",
            "reason": "write_or_verification_failed",
            "error": str(exc),
        }

    result = {
        **base_result,
        "status": "imported",
        "target": {
            **target,
            "status": "imported",
            "settings_exists": True,
            "secrets_exists": bool(encrypted_items),
            "settings_written": True,
            "secrets_written": bool(encrypted_items),
        },
        "verification": {
            "settings_schema": verified_settings["schema"],
            "profile_count": len(verified_settings["profiles"]),
            "route_pool_count": len(verified_settings["route_pools"]),
            "task_binding_count": len(verified_settings["route_bindings"]),
            "secret_ids": sorted(set(secret_values)),
            "plaintext_secret_check": "not_present_in_settings_or_report",
            "network_calls": 0,
        },
        "updated_at": now_iso(),
    }
    write_json(report_file, result)
    settings_module._restrict_file(report_file)
    return result


def _discover_legacy_sources(
    *,
    bundle_dir: str | Path,
    env_files: list[str | Path] | tuple[str | Path, ...] | None,
    allowed_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    bundle = _source_path(bundle_dir, allowed_root=allowed_root, kind="bundle directory")
    if not bundle.is_dir():
        raise ValueError(f"legacy bundle directory not found: {bundle}")
    selected_env_files = list(env_files) if env_files is not None else [
        allowed_root / ".local" / "model-connector.env",
        allowed_root / ".local" / "vision.env",
    ]
    sources: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    environment: dict[str, str] = {}
    environment_sources: dict[str, str] = {}
    allowed_env_names = set(ARK_SECRET_NAMES) | {"AGNES_API_KEY"}
    for raw_path in selected_env_files:
        path = _source_path(raw_path, allowed_root=allowed_root, kind="environment file")
        if not path.is_file():
            excluded.append({"id": path.name, "reason": "source_not_found"})
            continue
        payload = _read_source_bytes(path, max_bytes=MAX_ENV_SOURCE_BYTES)
        parsed = _parse_env(payload.decode("utf-8-sig"), allowed_names=allowed_env_names)
        recognised = sorted(parsed)
        sources.append(
            {
                "kind": "environment",
                "path": str(path),
                "sha256": _sha256(payload),
                "recognized_secret_names": recognised,
                "secret_values_exposed": False,
            }
        )
        for name, value in parsed.items():
            if name not in environment and value:
                environment[name] = value
                environment_sources[name] = str(path)

    candidates: list[dict[str, Any]] = []
    secret_values: dict[str, str] = {}
    local_path = _source_path(
        bundle / "local-vlm-serving-smoke.json",
        allowed_root=allowed_root,
        kind="local VLM source",
    )
    if local_path.is_file():
        payload = _read_source_bytes(local_path, max_bytes=MAX_JSON_SOURCE_BYTES)
        data = _load_json_object(payload, source=local_path)
        profile_data = data.get("profile")
        if not isinstance(profile_data, dict) or str(profile_data.get("provider") or "") != "local_qwen_vl":
            raise ValueError("local VLM smoke source does not describe provider=local_qwen_vl")
        validated = validate_model_api_profile(
            {
                "id": LOCAL_VLM_PROFILE_ID,
                "name": "Imported local Qwen-VL",
                "provider": "local_qwen_vl",
                "adapter_backend": "proxy",
                "location": "local",
                "capabilities": ["vision"],
                "base_url": str(profile_data.get("base_url") or ""),
                "model": str(profile_data.get("model") or ""),
                "timeout_seconds": 90,
                "enabled": True,
            },
            list(LOCAL_VLM_TASKS),
        )
        candidates.append(
            {
                **validated,
                "credential": {"present": False, "source_env_var": "", "source_path": ""},
                "evidence": str(local_path),
                "health_status": "unverified",
            }
        )
        sources.append({"kind": "local_vlm_smoke", "path": str(local_path), "sha256": _sha256(payload)})
    else:
        excluded.append({"id": LOCAL_VLM_PROFILE_ID, "reason": "local_vlm_smoke_not_found"})

    ark_path = _source_path(
        bundle / "volcengine-provider-public.json",
        allowed_root=allowed_root,
        kind="Ark provider source",
    )
    if ark_path.is_file():
        payload = _read_source_bytes(ark_path, max_bytes=MAX_JSON_SOURCE_BYTES)
        data = _load_json_object(payload, source=ark_path)
        if _contains_plaintext_secret_field(data):
            raise ValueError("public Ark provider source contains a plaintext secret field")
        if str(data.get("provider") or "") != "volcengine_coding_plan":
            raise ValueError("Ark provider source does not describe provider=volcengine_coding_plan")
        host = str(urlsplit(str(data.get("base_url") or "")).hostname or "").lower()
        if host != ARK_HOST:
            raise ValueError("Ark provider source destination is not the trusted Ark host")
        secret_name = next((name for name in ARK_SECRET_NAMES if str(environment.get(name) or "")), "")
        if not secret_name:
            excluded.append({"id": ARK_PROFILE_ID, "reason": "approved_ark_secret_not_found"})
        else:
            validated = validate_model_api_profile(
                {
                    "id": ARK_PROFILE_ID,
                    "name": "Imported Ark text and vision",
                    "provider": "volcengine_coding_plan",
                    "adapter_backend": "proxy",
                    "location": "remote",
                    "capabilities": ["text", "vision"],
                    "base_url": str(data.get("base_url") or ""),
                    "model": str(data.get("model") or ""),
                    "timeout_seconds": int(data.get("timeout_seconds") or 90),
                    "enabled": True,
                },
                list(ARK_TASKS),
            )
            candidates.append(
                {
                    **validated,
                    "credential": {
                        "present": True,
                        "source_env_var": secret_name,
                        "source_path": environment_sources[secret_name],
                    },
                    "evidence": str(ark_path),
                    "health_status": "not_probed",
                }
            )
            secret_values[ARK_PROFILE_ID] = environment[secret_name]
        sources.append({"kind": "ark_provider", "path": str(ark_path), "sha256": _sha256(payload)})
    else:
        excluded.append({"id": ARK_PROFILE_ID, "reason": "ark_provider_source_not_found"})

    if str(environment.get("AGNES_API_KEY") or ""):
        excluded.append({"id": "remote-agnes", "reason": "destination_not_in_trusted_broker_allowlist"})
    snapshot_path = bundle / "model-settings.json"
    if snapshot_path.is_file():
        excluded.append({"id": "model-settings.json", "reason": "legacy_snapshot_is_not_route_authority"})
    ignored_consents = sorted(
        str(path)
        for path in bundle.glob("*consent*.json")
        if path.is_file()
    )
    return (
        {
            "sources": sources,
            "excluded": excluded,
            "ignored_legacy_authorizations": {
                "count": len(ignored_consents),
                "paths": ignored_consents,
                "reason": "legacy consent is never imported or reused as v2 route authorization",
            },
        },
        candidates,
        secret_values,
    )


def _build_settings(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = sorted(
        [dict(row["profile"]) for row in candidates],
        key=lambda row: (str(row.get("name") or "").casefold(), str(row["id"])),
    )
    state: dict[str, Any] = {
        "schema": SETTINGS_SCHEMA,
        "profiles": profiles,
        "task_routes": {},
        "route_bindings": {},
        "route_pools": [],
        "updated_at": "",
    }
    for candidate in candidates:
        route_state = assign_profile_routes(
            state,
            profiles=profiles,
            profile_id=str(candidate["profile"]["id"]),
            selected_tasks=list(candidate["tasks"]),
        )
        state = {**state, **route_state}
    return state


def _inspect_target(
    settings_file: Path,
    secrets_file: Path,
    proposed: dict[str, Any],
    required_secret_ids: set[str],
) -> dict[str, Any]:
    current = load_model_api_settings(settings_file)
    secrets = settings_module._load_secret_document(secrets_file)
    current_secret_ids = set(secrets.get("items") or {})
    if _settings_equivalent(current, proposed) and required_secret_ids.issubset(current_secret_ids):
        return {
            "status": "already_imported",
            "settings_exists": settings_file.exists(),
            "secrets_exists": secrets_file.exists(),
        }
    settings_empty = not any(
        current.get(key)
        for key in ("profiles", "task_routes", "route_bindings", "route_pools")
    )
    secrets_empty = not current_secret_ids
    if settings_empty and secrets_empty:
        status = "empty"
    elif not settings_empty:
        status = "target_settings_not_empty"
    else:
        status = "target_secrets_not_empty"
    return {
        "status": status,
        "settings_exists": settings_file.exists(),
        "secrets_exists": secrets_file.exists(),
        "settings_empty": settings_empty,
        "secrets_empty": secrets_empty,
    }


def _write_settings_and_secrets(
    *,
    settings_file: Path,
    settings: dict[str, Any],
    secrets_file: Path,
    secrets: dict[str, Any],
    required_secret_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    originals = {
        settings_file: settings_file.read_bytes() if settings_file.exists() else None,
        secrets_file: secrets_file.read_bytes() if secrets_file.exists() else None,
    }
    try:
        write_json(secrets_file, secrets)
        settings_module._restrict_file(secrets_file)
        write_json(settings_file, settings)
        settings_module._restrict_file(settings_file)
        verified_settings = load_model_api_settings(settings_file)
        verified_secrets = settings_module._load_secret_document(secrets_file)
        if not _settings_equivalent(verified_settings, settings):
            raise RuntimeError("import verification failed for model API settings")
        if not required_secret_ids.issubset(set(verified_secrets.get("items") or {})):
            raise RuntimeError("import verification failed for model API secrets")
        return verified_settings, verified_secrets
    except Exception:
        rollback_errors: list[str] = []
        for path, payload in originals.items():
            try:
                _restore_file(path, payload)
            except OSError as exc:
                rollback_errors.append(f"{path}: {exc}")
        if rollback_errors:
            raise RuntimeError("model API import failed and rollback was incomplete: " + "; ".join(rollback_errors))
        raise


def _restore_file(path: Path, payload: bytes | None) -> None:
    if payload is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.rollback")
    try:
        temporary.write_bytes(payload)
        replace_file_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _settings_equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("schema", "profiles", "task_routes", "route_bindings", "route_pools")
    return all(left.get(key) == right.get(key) for key in keys)


def _public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    credential = dict(candidate.get("credential") or {})
    return {
        "profile": dict(candidate["profile"]),
        "tasks": list(candidate["tasks"]),
        "credential": {
            "present": bool(credential.get("present")),
            "source_env_var": str(credential.get("source_env_var") or ""),
            "source_path": str(credential.get("source_path") or ""),
            "persisted_as": "windows_dpapi" if credential.get("present") else "not_required",
        },
        "evidence": str(candidate.get("evidence") or ""),
        "health_status": str(candidate.get("health_status") or "unverified"),
    }


def _blocked_result(
    *,
    execute: bool,
    settings_file: Path,
    secrets_file: Path,
    report_file: Path,
    reason: str,
    error: str,
) -> dict[str, Any]:
    return {
        "schema": IMPORT_SCHEMA,
        "status": "blocked",
        "execute": bool(execute),
        "reason": reason,
        "error": error,
        "settings_path": str(settings_file),
        "secrets_path": str(secrets_file),
        "report_path": str(report_file),
        "security": {
            "plaintext_secrets_in_result": False,
            "plaintext_secrets_persisted": False,
            "old_consents_reused": False,
            "network_calls": 0,
        },
        "updated_at": now_iso(),
    }


def _source_path(value: str | Path, *, allowed_root: Path, kind: str) -> Path:
    path = Path(value).expanduser()
    path = path.resolve() if path.is_absolute() else (allowed_root / path).resolve()
    try:
        path.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"{kind} must stay within project root: {path}") from exc
    return path


def _path(value: str | Path | None, fallback: Path) -> Path:
    return Path(value).expanduser().resolve() if value is not None and str(value).strip() else fallback.resolve()


def _parse_env(text: str, *, allowed_names: set[str] | None = None) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"invalid environment assignment at line {line_number}")
        name, value = line.split("=", 1)
        name = name.strip()
        if not _ENV_NAME_RE.fullmatch(name):
            raise ValueError(f"invalid environment variable name at line {line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if allowed_names is None or name in allowed_names:
            result[name] = value
    return result


def _load_json_object(payload: bytes, *, source: Path) -> dict[str, Any]:
    data = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"legacy source must contain a JSON object: {source}")
    return data


def _read_source_bytes(path: Path, *, max_bytes: int) -> bytes:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"legacy source exceeds size limit: {path}")
    payload = path.read_bytes()
    if len(payload) > max_bytes:
        raise ValueError(f"legacy source exceeds size limit: {path}")
    return payload


def _contains_plaintext_secret_field(data: dict[str, Any]) -> bool:
    sensitive = {"api_key", "apikey", "token", "secret", "password", "authorization"}
    return any(str(key).strip().lower() in sensitive for key in data)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
