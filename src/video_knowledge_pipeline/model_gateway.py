from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .model_api_legacy_import import safe_import_legacy_model_api_settings
from .model_api_settings import (
    _load_secret_document,
    _profile_secret_id,
    _read_secret,
    _save_secret,
    default_secrets_path,
    default_settings_path,
    load_model_api_settings,
)
from .model_route_settings import resolve_model_route
from .path_defaults import port_record_path
from .storage import write_text_atomic

CONFIG_SCHEMA = "video_knowledge_pipeline.model_gateway_config.v1"
RESULT_SCHEMA = "video_knowledge_pipeline.model_gateway.v1"
MASTER_KEY_ID = "gateway-master-key"
MASTER_KEY_ENV = "VKP_LITELLM_MASTER_KEY"
RUNTIME_ONLY_PROVIDER_OPTIONS = frozenset(
    {
        "thinking_mode",
        "response_format",
        "enable_thinking",
        "thinking_budget",
        "reasoning_effort",
        "reasoning_format",
        "strip_reasoning_tags",
        "strip_json_fences",
        "max_tokens",
        "stream",
        "asr_timestamp_granularity",
    }
)
DEFAULT_PORT_RECORD = port_record_path()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_model_gateway_config_path() -> Path:
    return project_root() / "config" / "model-gateway.json"


def load_model_gateway_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = _path(config_path, default_model_gateway_config_path())
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or str(data.get("schema") or "") != CONFIG_SCHEMA:
        raise ValueError("invalid model gateway config")
    host = str(data.get("host") or "127.0.0.1").strip()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("model gateway must bind to loopback")
    port = int(data.get("port") or 18776)
    if port < 1 or port > 65535:
        raise ValueError("model gateway port must be between 1 and 65535")
    base = project_root()
    return {
        "schema": CONFIG_SCHEMA,
        "host": host,
        "port": port,
        "telemetry": bool(data.get("telemetry", False)),
        "config_path": str(_resolve_configured_path(data.get("config_path"), base / ".local" / "litellm-config.yaml")),
        "pid_path": str(_resolve_configured_path(data.get("pid_path"), base / ".local" / "model-gateway.pid")),
        "log_path": str(_resolve_configured_path(data.get("log_path"), base / ".local" / "model-gateway.log")),
        "source_path": str(path),
    }


def render_litellm_config(
    *,
    settings_path: str | Path | None = None,
    secrets_path: str | Path | None = None,
    gateway_config_path: str | Path | None = None,
    output_path: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    settings_file = _path(settings_path, default_settings_path())
    secrets_file = _path(secrets_path, default_secrets_path(settings_path=settings_file))
    gateway = load_model_gateway_config(gateway_config_path)
    destination = _path(output_path, Path(gateway["config_path"]))
    settings = load_model_api_settings(settings_file)
    configured_secret_ids = set((_load_secret_document(secrets_file).get("items") or {}).keys())
    routes, skipped_legacy_routes = _proxy_routes(settings)

    lines = ["model_list:"]
    model_count = 0
    env_names: set[str] = {MASTER_KEY_ENV}
    fallback_chains: dict[str, list[str]] = {}
    credential_blockers: list[dict[str, Any]] = []
    capacity_warnings: list[dict[str, Any]] = []
    for virtual_model in sorted(routes):
        route = routes[virtual_model]
        route_policy = dict(route.get("retry_policy") or {})
        model_names = [_route_model_name(virtual_model, index) for index, _ in enumerate(route["deployments"])]
        if len(model_names) > 1:
            fallback_chains[model_names[0]] = model_names[1:]
        for index, deployment in enumerate(route["deployments"]):
            profile_id = str(deployment["id"])
            secret_id = _profile_secret_id(deployment)
            gateway_deployment_id = _gateway_deployment_id(route, index)
            capability = str(route.get("capability") or "")
            required_capacity_keys = (
                ("rpm", "tpm", "max_parallel_requests")
                if capability in {"text", "vision"}
                else ("rpm", "max_parallel_requests")
            )
            capacity = {
                key: deployment.get(key)
                for key in required_capacity_keys
            }
            missing_capacity = [
                key for key, value in capacity.items() if value in (None, "")
            ]
            if missing_capacity:
                capacity_warnings.append(
                    {
                        "profile_id": profile_id,
                        "route_id": str(route.get("route_id") or ""),
                        "capability": str(route.get("capability") or ""),
                        "execution_location": str(route.get("execution_location") or ""),
                        "status": "capacity_policy_incomplete",
                        "missing": missing_capacity,
                        "next_action": "set_profile_capacity_limits",
                    }
                )
            env_name = _profile_key_env(profile_id)
            lines.extend(
                [
                    f"  - model_name: {_yaml(model_names[index])}",
                    "    litellm_params:",
                    f"      model: {_yaml(_litellm_model(deployment))}",
                    f"      api_base: {_yaml(deployment['base_url'])}",
                    f"      timeout: {int(deployment['timeout_seconds'])}",
                    f"      cooldown_time: {int(route_policy.get('cooldown_seconds') or 0)}",
                ]
            )
            for capacity_key in ("rpm", "tpm", "max_parallel_requests"):
                if deployment.get(capacity_key) not in (None, ""):
                    lines.append(
                        f"      {capacity_key}: {int(deployment[capacity_key])}"
                    )
            provider_options = dict(deployment.get("provider_options") or {})
            for key in sorted(provider_options):
                if key in RUNTIME_ONLY_PROVIDER_OPTIONS:
                    continue
                lines.append(f"      {key}: {_yaml(provider_options[key])}")
            required_options = [str(key) for key in deployment.get("required_provider_options") or []]
            missing_options = [key for key in required_options if provider_options.get(key) in (None, "")]
            if missing_options:
                credential_blockers.append(
                    {
                        "profile_id": profile_id,
                        "status": "missing_provider_options",
                        "missing_provider_options": missing_options,
                    }
                )
            auth_mode = str(deployment.get("auth_mode") or "api_key_dpapi")
            if auth_mode == "external_environment":
                bindings = [dict(row) for row in deployment.get("environment_bindings") or []]
                for binding in bindings:
                    param = str(binding.get("param") or "")
                    external_env = str(binding.get("env") or "")
                    env_names.add(external_env)
                    lines.append(f"      {param}: {_yaml('os.environ/' + external_env)}")
                missing_env = sorted(
                    str(row.get("env") or "")
                    for row in bindings
                    if row.get("required") and not str(os.environ.get(str(row.get("env") or "")) or "").strip()
                )
                if missing_env:
                    credential_blockers.append(
                        {
                            "profile_id": profile_id,
                            "status": "missing_environment",
                            "missing_environment": missing_env,
                        }
                    )
            elif secret_id in configured_secret_ids:
                lines.append(f"      api_key: {_yaml('os.environ/' + env_name)}")
                env_names.add(env_name)
            elif not bool(deployment.get("api_key_optional")):
                credential_blockers.append(
                    {"profile_id": profile_id, "status": "missing_api_key"}
                )
            lines.extend(
                [
                    "    model_info:",
                    f"      id: {_yaml(gateway_deployment_id)}",
                    f"      mode: {_yaml(_litellm_mode(str(route['capability'])))}",
                    f"      profile_id: {_yaml(profile_id)}",
                    f"      route_id: {_yaml(route['route_id'])}",
                    f"      route_revision: {_yaml(route['route_revision'])}",
                    f"      execution_location: {_yaml(route['execution_location'])}",
                    f"      deployment_index: {index}",
                    f"      primary_virtual_model: {_yaml(virtual_model)}",
                ]
            )
            model_count += 1
    if model_count == 0:
        lines.append("  []")
    max_fallbacks = max((len(items) for items in fallback_chains.values()), default=0)
    lines.extend(
        [
            "router_settings:",
            f"  routing_strategy: {_yaml('simple-shuffle')}",
            "  num_retries: 0",
            "  enable_pre_call_checks: true",
            f"  max_fallbacks: {max_fallbacks}",
            "  fallbacks:",
        ]
    )
    if fallback_chains:
        for primary in sorted(fallback_chains):
            lines.append(f"    - {_yaml({primary: fallback_chains[primary]})}")
    else:
        lines.append("    []")
    lines.extend(
        [
            "general_settings:",
            f"  master_key: {_yaml('os.environ/' + MASTER_KEY_ENV)}",
            "  disable_spend_logs: true",
            "  background_health_checks: false",
            "litellm_settings:",
            f"  telemetry: {str(bool(gateway['telemetry'])).lower()}",
            "  turn_off_message_logging: true",
            "  redact_user_api_key_info: true",
        ]
    )
    rendered = "\n".join(lines) + "\n"
    if write:
        write_text_atomic(destination, rendered)
        _restrict_file(destination)
    return {
        "schema": RESULT_SCHEMA,
        "status": "rendered" if write else "planned",
        "write": bool(write),
        "config_path": str(destination),
        "model_count": model_count,
        "virtual_models": sorted(routes),
        "required_env_names": sorted(env_names),
        "credential_blockers": credential_blockers,
        "capacity_warnings": capacity_warnings,
        "capacity_policy_ready": not capacity_warnings,
        "ready_for_start": model_count > 0 and not credential_blockers,
        "fallback_chains": fallback_chains,
        "skipped_legacy_routes": skipped_legacy_routes,
        "secrets_rendered": False,
        "updated_at": _now_iso(),
    }


def model_gateway_doctor(
    *,
    gateway_config_path: str | Path | None = None,
    port_record_path: str | Path | None = None,
    probe_http: bool = True,
) -> dict[str, Any]:
    gateway = load_model_gateway_config(gateway_config_path)
    host = str(gateway["host"])
    port = int(gateway["port"])
    record_path = _path(port_record_path, DEFAULT_PORT_RECORD)
    recorded, owned = _port_record_state(record_path, port)
    live = _can_connect(host, port)
    bind_available = False if live else _can_bind(host, port)
    dynamic_range = _dynamic_tcp_port_range()
    outside_dynamic_range = dynamic_range is None or not (
        int(dynamic_range["start"]) <= port <= int(dynamic_range["end"])
    )
    dynamic_detail = (
        "not_detected"
        if dynamic_range is None
        else f"{dynamic_range['start']}-{dynamic_range['end']}"
    )
    litellm_proxy_available = _optional_module_available(
        "litellm.proxy.proxy_server"
    )
    checks = [
        {"key": "loopback_host", "ok": host in {"127.0.0.1", "localhost", "::1"}, "detail": host},
        {
            "key": "litellm_proxy_module",
            "ok": litellm_proxy_available,
            "detail": "litellm.proxy.proxy_server",
            "blocker": (
                "optional_dependency_missing:litellm"
                if not litellm_proxy_available
                else ""
            ),
        },
        {
            "key": "port_record",
            "ok": recorded and owned,
            "detail": str(record_path),
            "recorded": recorded,
            "owned_by_vkp": owned,
        },
        {
            "key": "outside_dynamic_client_range",
            "ok": outside_dynamic_range,
            "detail": dynamic_detail,
        },
        {"key": "live_listener", "ok": live, "detail": f"{host}:{port}"},
        {"key": "bind_available", "ok": bind_available, "detail": f"{host}:{port}"},
    ]
    http_status = "not_probed"
    if probe_http and live:
        http_status = _probe_health(host, port)
        checks.append({"key": "http_health", "ok": http_status == "ready", "detail": http_status})
    listener_ready = live and (not probe_http or http_status == "ready")
    check_by_key = {str(row["key"]): row for row in checks}
    ready = (
        bool(check_by_key["loopback_host"]["ok"])
        and bool(check_by_key["litellm_proxy_module"]["ok"])
        and bool(check_by_key["port_record"]["ok"])
        and bool(check_by_key["outside_dynamic_client_range"]["ok"])
        and (listener_ready or bind_available)
    )
    return {
        "schema": RESULT_SCHEMA,
        "status": "ready" if ready else "blocked",
        "ready": ready,
        "gateway": gateway,
        "checks": checks,
        "dynamic_client_port_range": dynamic_range,
        "http_status": http_status,
        "remote_requests_made": False,
        "updated_at": _now_iso(),
    }


def _optional_module_available(module_name: str) -> bool:
    """Return optional dependency readiness without importing it.

    ``find_spec`` raises ``ModuleNotFoundError`` for a dotted module when its
    parent package is absent.  A clean core install must report that optional
    capability as blocked instead of crashing unrelated status/UI paths.
    """

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
        return False


def model_gateway_status(
    *,
    gateway_config_path: str | Path | None = None,
    port_record_path: str | Path | None = None,
) -> dict[str, Any]:
    return model_gateway_doctor(
        gateway_config_path=gateway_config_path,
        port_record_path=port_record_path,
        probe_http=True,
    )


def model_gateway_runtime_readiness(
    *,
    gateway_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Probe only the local Proxy runtime needed before a consent reservation.

    Unlike ``model_gateway_status`` this check does not treat an available bind
    port as ready: a live, healthy loopback listener is required. It does not
    read provider credentials or contact any remote destination.
    """

    try:
        gateway = load_model_gateway_config(gateway_config_path)
        host = str(gateway.get("host") or "").strip().lower()
        port = int(gateway.get("port") or 0)
    except (OSError, TypeError, ValueError) as exc:
        return {
            "schema": RESULT_SCHEMA,
            "status": "configuration_blocked",
            "ready": False,
            "error": str(exc),
            "remote_requests_made": False,
            "updated_at": _now_iso(),
        }
    loopback = host in {"127.0.0.1", "localhost", "::1"}
    http_status = _probe_health(host, port) if loopback and port > 0 else "not_probed"
    ready = loopback and port > 0 and http_status == "ready"
    return {
        "schema": RESULT_SCHEMA,
        "status": "ready" if ready else "gateway_unavailable",
        "ready": ready,
        "gateway": {"host": host, "port": port},
        "http_status": http_status,
        "remote_requests_made": False,
        "updated_at": _now_iso(),
    }


def start_model_gateway(
    *,
    gateway_config_path: str | Path | None = None,
    settings_path: str | Path | None = None,
    secrets_path: str | Path | None = None,
    port_record_path: str | Path | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    gateway = load_model_gateway_config(gateway_config_path)
    settings_file = _path(settings_path, default_settings_path())
    secrets_file = _path(secrets_path, default_secrets_path(settings_path=settings_file))
    try:
        render = render_litellm_config(
            settings_path=settings_file,
            secrets_path=secrets_file,
            gateway_config_path=gateway_config_path,
            output_path=gateway["config_path"],
            write=execute,
        )
    except (OSError, ValueError) as exc:
        return {
            "schema": RESULT_SCHEMA,
            "status": "configuration_blocked",
            "execute": bool(execute),
            "gateway": gateway,
            "render": {
                "model_count": 0,
                "ready_for_start": False,
                "credential_blockers": [],
            },
            "error_code": "route_render_failed",
            "error": str(exc),
            "secrets_in_command": False,
            "remote_requests_made": False,
        }
    command = _litellm_command() + [
        "--config",
        str(gateway["config_path"]),
        "--host",
        str(gateway["host"]),
        "--port",
        str(gateway["port"]),
        "--telemetry",
        "False",
    ]
    preview = {
        "schema": RESULT_SCHEMA,
        "status": "planned",
        "execute": bool(execute),
        "command": command,
        "gateway": gateway,
        "render": render,
        "secrets_in_command": False,
        "remote_requests_made": False,
    }
    if not execute:
        return preview
    if int(render.get("model_count") or 0) < 1:
        return {**preview, "status": "configuration_blocked", "error": "no proxy model routes are configured"}
    if render.get("credential_blockers"):
        return {
            **preview,
            "status": "configuration_blocked",
            "error": "one or more proxy deployments are missing credentials or required provider options",
            "credential_blockers": render["credential_blockers"],
        }
    doctor = model_gateway_doctor(
        gateway_config_path=gateway_config_path,
        port_record_path=port_record_path,
        probe_http=True,
    )
    live = next(row for row in doctor["checks"] if row["key"] == "live_listener")["ok"]
    if live:
        status = "already_running" if doctor.get("ready") else "port_blocked"
        return {**preview, "status": status, "doctor": doctor}
    bind_ok = next(row for row in doctor["checks"] if row["key"] == "bind_available")["ok"]
    record_ok = next(row for row in doctor["checks"] if row["key"] == "port_record")["ok"]
    if not bind_ok or not record_ok:
        return {**preview, "status": "port_blocked", "doctor": doctor}
    env = _runtime_environment(settings_file, secrets_file)
    log_path = Path(gateway["log_path"])
    pid_path = Path(gateway["pid_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(project_root()),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            creationflags=creationflags,
        )
    pid_path.write_text(str(process.pid), encoding="ascii")
    _restrict_file(pid_path)
    return {
        **preview,
        "status": "started",
        "pid": process.pid,
        "pid_path": str(pid_path),
        "log_path": str(log_path),
    }


def _proxy_routes(settings: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    routes: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []
    for task in sorted(settings.get("route_bindings") or {}):
        binding = settings["route_bindings"][task]
        for location in ("local", "remote"):
            if not str(binding.get(f"{location}_pool_id") or ""):
                continue
            route = resolve_model_route(settings, task=task, execution_location=location)
            backends = {str(row.get("adapter_backend") or "") for row in route["deployments"]}
            if backends == {"proxy"}:
                routes.setdefault(route["virtual_model"], route)
                continue
            if "proxy" in backends:
                raise ValueError("route cannot mix proxy and legacy adapter backends")
            skipped.append(
                {
                    "task": str(task),
                    "execution_location": location,
                    "route_id": str(route["route_id"]),
                    "reason": "explicit_legacy_route",
                }
            )
    return routes, skipped


def _route_model_name(virtual_model: str, index: int) -> str:
    return virtual_model if index == 0 else f"{virtual_model}-fallback-{index + 1}"

def _gateway_deployment_id(route: dict[str, Any], index: int) -> str:
    deployments = route.get("deployments") if isinstance(route.get("deployments"), list) else []
    if index < 0 or index >= len(deployments):
        raise IndexError("gateway deployment index is outside route deployments")
    profile_id = str(deployments[index].get("id") or "")
    seed = f"{route.get('route_revision') or ''}:{index}:{profile_id}"
    return "vkp-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def _runtime_environment(settings_path: Path, secrets_path: Path) -> dict[str, str]:
    settings = load_model_api_settings(settings_path)
    routes, _skipped = _proxy_routes(settings)
    proxy_profile_ids = {
        str(deployment["id"])
        for route in routes.values()
        for deployment in route["deployments"]
    }
    env = dict(os.environ)
    for profile in settings["profiles"]:
        profile_id = str(profile["id"])
        if profile_id not in proxy_profile_ids:
            continue
        secret = _read_secret(_profile_secret_id(profile), secrets_path)
        if secret:
            env[_profile_key_env(profile_id)] = secret
    master = _read_secret(MASTER_KEY_ID, secrets_path)
    if not master:
        master = secrets.token_urlsafe(32)
        _save_secret(MASTER_KEY_ID, master, secrets_path)
    env[MASTER_KEY_ENV] = master
    env["LITELLM_TELEMETRY"] = "False"
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    return env


def _litellm_command() -> list[str]:
    executable = shutil.which("litellm")
    if not executable and os.name == "nt":
        candidate = Path(sys.executable).parent / "Scripts" / "litellm.exe"
        if candidate.is_file():
            executable = str(candidate)
    return [executable or "litellm"]


def _litellm_model(deployment: dict[str, Any]) -> str:
    model = str(deployment.get("model") or "")
    litellm_provider = str(deployment.get("litellm_provider") or "").strip()
    if not litellm_provider:
        raise ValueError("proxy deployment is missing litellm_provider")
    prefix = litellm_provider + "/"
    return model if model.startswith(prefix) else prefix + model


def _litellm_mode(capability: str) -> str:
    return {
        "asr": "audio_transcription",
        "ocr": "ocr",
    }.get(str(capability or ""), "chat")

def _profile_key_env(profile_id: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in profile_id.upper())
    return f"VKP_LITELLM_PROFILE_{safe}_API_KEY"


def _port_record_state(path: Path, port: int) -> tuple[bool, bool]:
    if not path.is_file():
        return False, False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False, False
    port_pattern = re.compile(rf"(?<!\d){int(port)}(?!\d)")
    matching_lines = [line for line in text.splitlines() if port_pattern.search(line)]
    recorded = bool(matching_lines)
    owned = any("VKP LiteLLM Proxy" in line for line in matching_lines)
    return recorded, owned


def _can_connect(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def _can_bind(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    candidate = socket.socket(family, socket.SOCK_STREAM)
    try:
        candidate.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        candidate.close()


def _dynamic_tcp_port_range() -> dict[str, int] | None:
    if os.name != "nt":
        return None
    creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        completed = subprocess.run(
            ["netsh", "interface", "ipv4", "show", "dynamicportrange", "protocol=tcp"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=False,
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    values = [
        int(value)
        for value in re.findall(
            r"(?m)^\s*[^:\r\n]+:\s*(\d+)\s*$",
            completed.stdout or "",
        )
    ]
    if completed.returncode != 0 or len(values) < 2 or values[1] < 1:
        return None
    start, count = values[:2]
    return {"start": start, "end": start + count - 1, "count": count}
def _probe_health(host: str, port: int) -> str:
    rendered_host = f"[{host}]" if ":" in host else host
    for path in ("/health/liveliness", "/health"):
        try:
            with urllib.request.urlopen(f"http://{rendered_host}:{port}{path}", timeout=1.0) as response:
                if 200 <= int(response.status) < 300:
                    return "ready"
        except (OSError, urllib.error.URLError):
            continue
    return "unreachable"


def _yaml(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _resolve_configured_path(value: Any, fallback: Path) -> Path:
    raw = str(value or "").strip()
    if not raw:
        return fallback.resolve()
    candidate = Path(raw).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (project_root() / candidate).resolve()


def _path(value: str | Path | None, fallback: Path) -> Path:
    return Path(value).expanduser().resolve() if value is not None and str(value).strip() else fallback.resolve()


def _restrict_file(path: Path) -> None:
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _now_iso() -> str:
    from .utils import now_iso

    return now_iso()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="video-knowledge-model-gateway")
    parser.add_argument("--gateway-config", default="")
    parser.add_argument("--settings-path", default="")
    parser.add_argument("--secrets-path", default="")
    parser.add_argument("--port-record-path", default="")
    sub = parser.add_subparsers(dest="command", required=True)
    render_parser = sub.add_parser("render-config")
    render_parser.add_argument("--output", default="")
    sub.add_parser("doctor")
    sub.add_parser("status")
    import_parser = sub.add_parser("import-legacy")
    import_parser.add_argument("--bundle-dir", required=True)
    import_parser.add_argument("--env-file", action="append", default=[])
    import_parser.add_argument("--report-path", default="")
    import_parser.add_argument("--execute", action="store_true")
    start_parser = sub.add_parser("start")
    start_parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    common = {"gateway_config_path": args.gateway_config or None}
    if args.command == "import-legacy":
        result = safe_import_legacy_model_api_settings(
            bundle_dir=args.bundle_dir,
            env_files=args.env_file or None,
            settings_path=args.settings_path or None,
            secrets_path=args.secrets_path or None,
            report_path=args.report_path or None,
            execute=bool(args.execute),
        )
    elif args.command == "render-config":
        result = render_litellm_config(
            **common,
            settings_path=args.settings_path or None,
            secrets_path=args.secrets_path or None,
            output_path=args.output or None,
            write=True,
        )
    elif args.command == "doctor":
        result = model_gateway_doctor(
            **common,
            port_record_path=args.port_record_path or None,
            probe_http=False,
        )
    elif args.command == "status":
        result = model_gateway_status(
            **common,
            port_record_path=args.port_record_path or None,
        )
    else:
        result = start_model_gateway(
            **common,
            settings_path=args.settings_path or None,
            secrets_path=args.secrets_path or None,
            port_record_path=args.port_record_path or None,
            execute=bool(args.execute),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") not in {"blocked", "port_blocked", "configuration_blocked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
