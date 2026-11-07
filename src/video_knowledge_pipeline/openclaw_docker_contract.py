from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import now_iso
from .path_defaults import openclaw_compose_path, workspace_root

DEFAULT_OPENCLAW_COMPOSE = openclaw_compose_path()
DEFAULT_HOST_ROOT = str(workspace_root())
DEFAULT_CONTAINER_ROOT = "/mnt/used-by-codex"
REQUIRED_ENV = {
    "VKP_API_BASE": "http://host.docker.internal:8931",
    "VDO_API_BASE": "http://host.docker.internal:8921",
    "VKP_CONTAINER_ROOT": DEFAULT_CONTAINER_ROOT,
    "VKP_HOST_ROOT": DEFAULT_HOST_ROOT,
}


def openclaw_docker_contract_check(
    compose_path: str | Path = "",
    *,
    host_root: str = DEFAULT_HOST_ROOT,
    container_root: str = DEFAULT_CONTAINER_ROOT,
) -> dict[str, Any]:
    """Inspect whether OpenClaw compose includes the VKP/VDO path contract."""

    path = Path(compose_path).expanduser().resolve() if compose_path else DEFAULT_OPENCLAW_COMPOSE
    required_env = dict(REQUIRED_ENV)
    required_env["VKP_CONTAINER_ROOT"] = container_root
    required_env["VKP_HOST_ROOT"] = host_root
    if not path.exists():
        return {
            "schema": "video_knowledge_pipeline.openclaw_docker_contract.v1",
            "created_at": now_iso(),
            "ok": False,
            "status": "compose_not_found",
            "compose_path": str(path),
            "issues": [{"key": "compose_not_found", "message": f"OpenClaw compose not found: {path}"}],
            "recommended_override": _override_payload(host_root=host_root, container_root=container_root, required_env=required_env),
        }

    text = path.read_text(encoding="utf-8", errors="replace")
    checks = {
        "container_root_mounted": container_root in text and _host_root_seen(text, host_root),
        "vkp_api_base": "VKP_API_BASE" in text and "host.docker.internal:8931" in text,
        "vdo_api_base": "VDO_API_BASE" in text and "host.docker.internal:8921" in text,
        "vkp_container_root": "VKP_CONTAINER_ROOT" in text and container_root in text,
        "vkp_host_root": "VKP_HOST_ROOT" in text and _host_root_seen(text, host_root),
    }
    issues = [
        {"key": key, "message": _issue_message(key, host_root=host_root, container_root=container_root)}
        for key, ok in checks.items()
        if not ok
    ]
    return {
        "schema": "video_knowledge_pipeline.openclaw_docker_contract.v1",
        "created_at": now_iso(),
        "ok": not issues,
        "status": "ok" if not issues else "contract_incomplete",
        "compose_path": str(path),
        "host_root": host_root,
        "container_root": container_root,
        "checks": checks,
        "issues": issues,
        "recommended_override": _override_payload(host_root=host_root, container_root=container_root, required_env=required_env),
        "operator_boundary": {
            "kind": "read_only_check",
            "summary": "This command inspects OpenClaw Docker configuration but does not modify production compose files.",
        },
        "next_actions": ["apply_openclaw_override_manually", "restart_openclaw_after_review"] if issues else ["openclaw_docker_contract_ok"],
    }


def _host_root_seen(text: str, host_root: str) -> bool:
    normalized = host_root.replace("\\", "/")
    return host_root in text or normalized in text or "USED_BY_CODEX_DIR" in text


def _issue_message(key: str, *, host_root: str, container_root: str) -> str:
    messages = {
        "container_root_mounted": f"Missing host workspace mount: {host_root} -> {container_root}",
        "vkp_api_base": "Missing VKP_API_BASE=http://host.docker.internal:8931",
        "vdo_api_base": "Missing VDO_API_BASE=http://host.docker.internal:8921",
        "vkp_container_root": f"Missing VKP_CONTAINER_ROOT={container_root}",
        "vkp_host_root": f"Missing VKP_HOST_ROOT={host_root}",
    }
    return messages.get(key, key)


def _override_payload(*, host_root: str, container_root: str, required_env: dict[str, str]) -> dict[str, Any]:
    return {
        "path": r"examples\openclaw\docker-compose.used-by-codex.override.yml",
        "volume": f"{host_root}:{container_root}",
        "environment": required_env,
        "apply_hint": "Use as a reviewed docker compose override; do not patch production compose automatically.",
    }

