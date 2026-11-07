from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit


from .canonical_json import canonical_json_sha256

ROUTE_SCHEMA = "video_knowledge_pipeline.model_route.v1"
SETTINGS_V2_SCHEMA = "video_knowledge_pipeline.local_model_api_settings.v2"
LEGACY_SETTINGS_SCHEMA = "video_knowledge_pipeline.local_model_api_settings.v1"

CAPABILITIES = ("text", "vision", "asr", "ocr")
TASK_CAPABILITIES = {
    "asr": "asr",
    "ocr": "ocr",
    "document_visual": "vision",
    "semantic_frame": "vision",
    "temporal_sequence": "vision",
    "video_segment": "vision",
    "text_llm": "text",
    "summary_rewrite": "text",
    "transcript_correction": "text",
}
DEFAULT_RETRY_POLICY = {
    "max_retries": 1,
    "timeout_seconds": 120,
    "cooldown_seconds": 5,
}
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")


def infer_profile_location(value: str, *, base_url: str) -> str:
    requested = str(value or "").strip().lower().replace("-", "_")
    host = (urlsplit(str(base_url or "")).hostname or "").lower()
    inferred = "local" if host in {"localhost", "127.0.0.1", "::1"} else "remote"
    if requested:
        if requested not in {"local", "remote"}:
            raise ValueError("profile location must be local or remote")
        if requested == "local" and inferred != "local":
            raise ValueError("local profiles must use a loopback base_url")
        if requested == "remote" and inferred == "local":
            raise ValueError("remote profiles must not use a loopback base_url")
        return requested
    return inferred


def normalise_profile_capabilities(value: Any, *, provider: str) -> list[str]:
    if value is None:
        if provider == "openai_compatible_asr":
            return ["asr"]
        if provider in {"mistral", "mistral_compatible_ocr"}:
            return ["ocr"]
        if provider in {"local_qwen_vl", "local_vlm"}:
            return ["vision"]
        return ["text", "vision"]
    if not isinstance(value, (list, tuple)):
        raise ValueError("profile capabilities must be a list")
    result: list[str] = []
    for item in value:
        capability = str(item or "").strip().lower().replace("-", "_")
        if capability not in CAPABILITIES:
            raise ValueError(f"unsupported model capability: {item!r}")
        if capability not in result:
            result.append(capability)
    if not result:
        raise ValueError("profile capabilities must not be empty")
    return result


def normalise_route_state(
    *,
    profiles: list[dict[str, Any]],
    legacy_routes: dict[str, str],
    route_pools: Any = None,
    route_bindings: Any = None,
) -> dict[str, Any]:
    profile_map = {str(row["id"]): row for row in profiles}
    pools = _normalise_pools(route_pools, profile_map) if route_pools is not None else []
    bindings = _normalise_bindings(route_bindings, pools) if route_bindings is not None else {}
    pool_map = {row["id"]: row for row in pools}

    for task, profile_id in legacy_routes.items():
        profile = profile_map.get(profile_id)
        if not profile:
            continue
        capability = TASK_CAPABILITIES[task]
        pool = _ensure_single_profile_pool(pools, profile, capability)
        pool_map[pool["id"]] = pool
        binding = dict(bindings.get(task) or _empty_binding())
        location = str(profile["location"])
        binding[f"{location}_pool_id"] = pool["id"]
        if not binding.get("default_location"):
            binding["default_location"] = location
        bindings[task] = binding

    referenced: set[str] = set()
    clean_bindings: dict[str, dict[str, str]] = {}
    for task, raw in bindings.items():
        binding = dict(raw)
        local_id = str(binding.get("local_pool_id") or "")
        remote_id = str(binding.get("remote_pool_id") or "")
        if local_id:
            _require_pool_location(pool_map, local_id, "local_only")
            _require_pool_capability(pool_map, local_id, TASK_CAPABILITIES[task])
            referenced.add(local_id)
        if remote_id:
            _require_pool_location(pool_map, remote_id, "remote_approved")
            _require_pool_capability(pool_map, remote_id, TASK_CAPABILITIES[task])
            referenced.add(remote_id)
        default = str(binding.get("default_location") or "")
        if default not in {"local", "remote"}:
            default = "local" if local_id else "remote"
        if default == "local" and not local_id:
            default = "remote"
        if default == "remote" and not remote_id:
            default = "local"
        if not local_id and not remote_id:
            continue
        clean_bindings[task] = {
            "default_location": default,
            "local_pool_id": local_id,
            "remote_pool_id": remote_id,
        }

    clean_pools = [row for row in pools if row["id"] in referenced]
    legacy = _legacy_route_view(clean_bindings, clean_pools)
    return {
        "route_pools": clean_pools,
        "route_bindings": clean_bindings,
        "task_routes": legacy,
    }


def assign_profile_routes(
    settings: dict[str, Any],
    *,
    profiles: list[dict[str, Any]],
    profile_id: str,
    selected_tasks: list[str],
) -> dict[str, Any]:
    state = normalise_route_state(
        profiles=profiles,
        legacy_routes={} if isinstance(settings.get("route_bindings"), dict) else dict(settings.get("task_routes") or {}),
        route_pools=settings.get("route_pools"),
        route_bindings=settings.get("route_bindings"),
    )
    profile = next(row for row in profiles if row["id"] == profile_id)
    pools = [dict(row) for row in state["route_pools"]]
    bindings = {task: dict(row) for task, row in state["route_bindings"].items()}
    location = str(profile["location"])

    for task, binding in list(bindings.items()):
        pool_key = f"{location}_pool_id"
        pool_id = str(binding.get(pool_key) or "")
        pool = next((row for row in pools if row["id"] == pool_id), None)
        if (
            task not in selected_tasks
            and pool
            and pool.get("deployments") == [profile_id]
        ):
            binding[pool_key] = ""
            if binding.get("default_location") == location:
                binding["default_location"] = "remote" if location == "local" else "local"
            bindings[task] = binding

    for task in selected_tasks:
        capability = TASK_CAPABILITIES[task]
        binding = dict(bindings.get(task) or _empty_binding())
        pool_key = f"{location}_pool_id"
        current_pool_id = str(binding.get(pool_key) or "")
        current_pool = next((row for row in pools if row["id"] == current_pool_id), None)
        if current_pool and profile_id in current_pool.get("deployments", []):
            pool = current_pool
        else:
            pool = _ensure_single_profile_pool(pools, profile, capability)
            binding[pool_key] = pool["id"]
        binding["default_location"] = location
        bindings[task] = binding

    return normalise_route_state(
        profiles=profiles,
        legacy_routes={},
        route_pools=pools,
        route_bindings=bindings,
    )


def delete_profile_routes(settings: dict[str, Any], *, profiles: list[dict[str, Any]], profile_id: str) -> dict[str, Any]:
    pools: list[dict[str, Any]] = []
    removed_pool_ids: set[str] = set()
    for raw in settings.get("route_pools") or []:
        row = dict(raw)
        deployments = [item for item in row.get("deployments") or [] if str(item) != profile_id]
        if not deployments:
            removed_pool_ids.add(str(row.get("id") or ""))
            continue
        row["deployments"] = deployments
        pools.append(row)
    bindings: dict[str, dict[str, str]] = {}
    for task, raw in (settings.get("route_bindings") or {}).items():
        row = dict(raw)
        for key in ("local_pool_id", "remote_pool_id"):
            if str(row.get(key) or "") in removed_pool_ids:
                row[key] = ""
        bindings[str(task)] = row
    return normalise_route_state(
        profiles=profiles,
        legacy_routes={},
        route_pools=pools,
        route_bindings=bindings,
    )


def resolve_model_route(settings: dict[str, Any], *, task: str, execution_location: str = "") -> dict[str, Any]:
    bindings = settings.get("route_bindings") if isinstance(settings.get("route_bindings"), dict) else {}
    binding = bindings.get(task) if isinstance(bindings.get(task), dict) else None
    if not binding:
        raise ValueError(f"no model route configured for task: {task}")
    requested = str(execution_location or binding.get("default_location") or "").strip().lower()
    if requested not in {"local", "remote"}:
        raise ValueError("execution_location must be local or remote")
    pool_id = str(binding.get(f"{requested}_pool_id") or "")
    if not pool_id:
        raise ValueError(f"no {requested} model route configured for task: {task}")
    pool = next((row for row in settings.get("route_pools") or [] if row.get("id") == pool_id), None)
    if not isinstance(pool, dict):
        raise ValueError(f"model route pool not found: {pool_id}")
    expected_pool_location = "local_only" if requested == "local" else "remote_approved"
    if str(pool.get("location") or "") != expected_pool_location:
        raise ValueError("route binding location does not match route pool location")
    expected_capability = TASK_CAPABILITIES.get(task)
    if not expected_capability or str(pool.get("capability") or "") != expected_capability:
        raise ValueError("route binding task capability does not match route pool capability")
    profiles = {str(row["id"]): row for row in settings.get("profiles") or []}
    deployments = []
    for profile_id in pool.get("deployments") or []:
        profile = profiles.get(str(profile_id))
        if not profile or not profile.get("enabled", True):
            continue
        deployments.append(_deployment_identity(profile, capability=str(pool["capability"])))
    if not deployments:
        raise ValueError(f"model route has no enabled deployments: {pool_id}")
    snapshot = {
        "route_id": pool_id,
        "execution_location": requested,
        "pool_location": str(pool["location"]),
        "capability": str(pool["capability"]),
        "deployments": deployments,
        "retry_policy": dict(pool["retry_policy"]),
    }
    revision = canonical_json_sha256(snapshot)
    slug = re.sub(r"[^a-z0-9]+", "-", pool_id.lower()).strip("-")[:40] or "pool"
    virtual_model = f"vkp-{requested}-{pool['capability']}-{slug}-{revision[:12]}"
    return {
        "schema": ROUTE_SCHEMA,
        **snapshot,
        "route_revision": revision,
        "virtual_model": virtual_model,
    }


def _normalise_pools(value: Any, profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("route_pools must be a list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError("route pool must be an object")
        pool_id = str(raw.get("id") or "").strip().lower()
        if not _ID_RE.fullmatch(pool_id) or pool_id in seen:
            raise ValueError("route pool ids must be unique lowercase identifiers")
        seen.add(pool_id)
        location = str(raw.get("location") or "").strip().lower()
        if location not in {"local_only", "remote_approved"}:
            raise ValueError("route pool location must be local_only or remote_approved")
        capability = str(raw.get("capability") or "").strip().lower()
        if capability not in CAPABILITIES:
            raise ValueError("route pool capability is invalid")
        deployments = [str(item or "").strip().lower() for item in raw.get("deployments") or []]
        if not deployments:
            raise ValueError("route pool requires at least one deployment")
        expected_location = "local" if location == "local_only" else "remote"
        for profile_id in deployments:
            profile = profiles.get(profile_id)
            if not profile:
                raise ValueError(f"route pool deployment not found: {profile_id}")
            if profile["location"] != expected_location:
                raise ValueError("route pool cannot mix local and remote deployment locations")
            if capability not in profile["capabilities"]:
                raise ValueError(f"deployment {profile_id} does not support {capability}")
        if len(set(deployments)) != len(deployments):
            raise ValueError("route pool deployments must be unique and ordered")
        backends = {str(profiles[profile_id].get("adapter_backend") or "") for profile_id in deployments}
        if "proxy" in backends and len(backends) != 1:
            raise ValueError("route pool cannot mix proxy and legacy adapter backends")
        if len(deployments) > 1 and backends != {"proxy"}:
            raise ValueError("multi-deployment route pools require adapter_backend=proxy")
        retry = _normalise_retry_policy(raw.get("retry_policy"))
        result.append(
            {
                "id": pool_id,
                "name": str(raw.get("name") or pool_id).strip()[:100],
                "location": location,
                "capability": capability,
                "deployments": deployments,
                "retry_policy": retry,
            }
        )
    return result


def _normalise_bindings(value: Any, pools: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        raise ValueError("route_bindings must be an object")
    pool_ids = {row["id"] for row in pools}
    result: dict[str, dict[str, str]] = {}
    for task, raw in value.items():
        if task not in TASK_CAPABILITIES or not isinstance(raw, dict):
            raise ValueError("route binding is invalid")
        local_id = str(raw.get("local_pool_id") or "")
        remote_id = str(raw.get("remote_pool_id") or "")
        if local_id and local_id not in pool_ids:
            raise ValueError(f"route binding pool not found: {local_id}")
        if remote_id and remote_id not in pool_ids:
            raise ValueError(f"route binding pool not found: {remote_id}")
        result[str(task)] = {
            "default_location": str(raw.get("default_location") or ""),
            "local_pool_id": local_id,
            "remote_pool_id": remote_id,
        }
    return result


def _ensure_single_profile_pool(
    pools: list[dict[str, Any]],
    profile: dict[str, Any],
    capability: str,
) -> dict[str, Any]:
    pool_id = f"pool-{profile['id']}-{capability}"
    existing = next((row for row in pools if row["id"] == pool_id), None)
    row = {
        "id": pool_id,
        "name": f"{profile['name']} / {capability}",
        "location": "local_only" if profile["location"] == "local" else "remote_approved",
        "capability": capability,
        "deployments": [profile["id"]],
        "retry_policy": dict(DEFAULT_RETRY_POLICY),
    }
    if existing is None:
        pools.append(row)
        return row
    existing.update(row)
    return existing


def _legacy_route_view(bindings: dict[str, dict[str, str]], pools: list[dict[str, Any]]) -> dict[str, str]:
    pool_map = {row["id"]: row for row in pools}
    result: dict[str, str] = {}
    for task, binding in bindings.items():
        location = binding["default_location"]
        pool = pool_map.get(str(binding.get(f"{location}_pool_id") or ""))
        if pool and pool["deployments"]:
            result[task] = str(pool["deployments"][0])
    return result


def _deployment_identity(profile: dict[str, Any], *, capability: str) -> dict[str, Any]:
    result = {
        "id": str(profile["id"]),
        "provider": str(profile["provider"]),
        "litellm_provider": str(profile.get("litellm_provider") or ""),
        "auth_mode": str(profile.get("auth_mode") or "api_key_dpapi"),
        "api_key_optional": bool(profile.get("api_key_optional")),
        "secret_ref": str(profile.get("secret_ref") or f"dpapi:{profile['id']}"),
        "provider_options": dict(profile.get("provider_options") or {}),
        "required_provider_options": list(profile.get("required_provider_options") or []),
        "environment_bindings": [dict(row) for row in profile.get("environment_bindings") or []],
        "model": str(profile.get("model") or ""),
        "base_url": str(profile.get("base_url") or ""),
        "adapter_backend": str(profile.get("adapter_backend") or ""),
        "interface": {
            "asr": "openai_audio_transcriptions",
            "ocr": "mistral_ocr",
        }.get(capability, "openai_chat_completions"),
        "timeout_seconds": int(profile.get("timeout_seconds") or 120),
    }
    for key in ("rpm", "tpm", "max_parallel_requests"):
        if profile.get(key) not in (None, ""):
            result[key] = int(profile[key])
    return result


def _normalise_retry_policy(value: Any) -> dict[str, int]:
    raw = value if isinstance(value, dict) else {}
    result = {
        "max_retries": int(raw.get("max_retries", DEFAULT_RETRY_POLICY["max_retries"])),
        "timeout_seconds": int(raw.get("timeout_seconds", DEFAULT_RETRY_POLICY["timeout_seconds"])),
        "cooldown_seconds": int(raw.get("cooldown_seconds", DEFAULT_RETRY_POLICY["cooldown_seconds"])),
    }
    if result["max_retries"] < 0 or result["max_retries"] > 10:
        raise ValueError("max_retries must be between 0 and 10")
    if result["timeout_seconds"] < 1 or result["timeout_seconds"] > 3600:
        raise ValueError("route timeout_seconds must be between 1 and 3600")
    if result["cooldown_seconds"] < 0 or result["cooldown_seconds"] > 3600:
        raise ValueError("cooldown_seconds must be between 0 and 3600")
    return result


def _require_pool_location(pool_map: dict[str, dict[str, Any]], pool_id: str, location: str) -> None:
    pool = pool_map.get(pool_id)
    if not pool:
        raise ValueError(f"route pool not found: {pool_id}")
    if pool["location"] != location:
        raise ValueError("route binding location does not match route pool location")


def _require_pool_capability(pool_map: dict[str, dict[str, Any]], pool_id: str, capability: str) -> None:
    pool = pool_map.get(pool_id)
    if not pool:
        raise ValueError(f"route pool not found: {pool_id}")
    if str(pool.get("capability") or "") != str(capability):
        raise ValueError("route binding task capability does not match route pool capability")


def _empty_binding() -> dict[str, str]:
    return {"default_location": "", "local_pool_id": "", "remote_pool_id": ""}