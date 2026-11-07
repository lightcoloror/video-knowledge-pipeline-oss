from __future__ import annotations

import json
import socket
from pathlib import Path

from video_knowledge_pipeline import model_api_settings as settings_module
from video_knowledge_pipeline import model_gateway as gateway_module
from video_knowledge_pipeline.model_api_settings import upsert_model_api_profile
from video_knowledge_pipeline.model_gateway import (
    load_model_gateway_config,
    model_gateway_doctor,
    render_litellm_config,
    start_model_gateway,
)


def _remote_profile() -> dict[str, object]:
    return {
        "id": "remote-text",
        "name": "Remote Text",
        "provider": "openai_compatible",
        "adapter_backend": "proxy",
        "location": "remote",
        "capabilities": ["text"],
        "base_url": "https://models.example.com/v1",
        "model": "remote-text",
        "timeout_seconds": 120,
        "enabled": True,
    }


def test_gateway_config_uses_single_loopback_port_source() -> None:
    configured = load_model_gateway_config()

    assert configured["host"] == "127.0.0.1"
    assert configured["port"] == 18776
    assert configured["telemetry"] is False


def test_gateway_doctor_blocks_cleanly_when_litellm_is_not_installed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "gateway.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.model_gateway_config.v1",
                "host": "127.0.0.1",
                "port": 18776,
                "telemetry": False,
                "config_path": str(tmp_path / "generated.yaml"),
                "pid_path": str(tmp_path / "gateway.pid"),
                "log_path": str(tmp_path / "gateway.log"),
            }
        ),
        encoding="utf-8",
    )
    port_record = tmp_path / "ports.md"
    port_record.write_text("18776 VKP LiteLLM Proxy\n", encoding="utf-8")

    def missing_optional_parent(name: str):
        if name == "litellm.proxy.proxy_server":
            raise ModuleNotFoundError("No module named 'litellm'")
        return None

    monkeypatch.setattr(gateway_module.importlib.util, "find_spec", missing_optional_parent)
    monkeypatch.setattr(gateway_module, "_can_connect", lambda host, port: False)
    monkeypatch.setattr(gateway_module, "_can_bind", lambda host, port: True)
    monkeypatch.setattr(gateway_module, "_dynamic_tcp_port_range", lambda: None)

    result = model_gateway_doctor(
        gateway_config_path=config_path,
        port_record_path=port_record,
        probe_http=False,
    )

    check = next(row for row in result["checks"] if row["key"] == "litellm_proxy_module")
    assert result["status"] == "blocked"
    assert result["ready"] is False
    assert check["ok"] is False
    assert check["blocker"] == "optional_dependency_missing:litellm"
    assert result["remote_requests_made"] is False


def test_litellm_config_is_secretless_and_content_addressed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    output_path = tmp_path / "litellm.yaml"
    monkeypatch.setattr(settings_module, "_protect_secret", lambda value: "cipher:" + value.encode().hex())
    monkeypatch.setattr(settings_module, "_unprotect_secret", lambda value: bytes.fromhex(value.removeprefix("cipher:")).decode())
    secret = "never-render-this-secret"
    upsert_model_api_profile(
        _remote_profile(),
        tasks=["text_llm"],
        api_key=secret,
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    result = render_litellm_config(
        settings_path=settings_path,
        secrets_path=secrets_path,
        output_path=output_path,
        write=True,
    )

    rendered = output_path.read_text(encoding="utf-8")
    assert result["model_count"] == 1
    assert "vkp-remote-text-" in rendered
    assert "os.environ/VKP_LITELLM_PROFILE_REMOTE_TEXT_API_KEY" in rendered
    assert "os.environ/VKP_LITELLM_MASTER_KEY" in rendered
    assert secret not in rendered
    assert secret not in json.dumps(result)


def test_gateway_start_is_preview_only_by_default(tmp_path: Path, monkeypatch) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    config_path = tmp_path / "gateway.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.model_gateway_config.v1",
                "host": "127.0.0.1",
                "port": 18776,
                "telemetry": False,
                "config_path": str(tmp_path / "generated.yaml"),
                "pid_path": str(tmp_path / "gateway.pid"),
                "log_path": str(tmp_path / "gateway.log"),
            }
        ),
        encoding="utf-8",
    )
    upsert_model_api_profile(
        {
            **_remote_profile(),
            "id": "local-text",
            "name": "Local Text",
            "location": "local",
            "base_url": "http://127.0.0.1:19000/v1",
        },
        tasks=["text_llm"],
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    preview = start_model_gateway(
        gateway_config_path=config_path,
        settings_path=settings_path,
        secrets_path=secrets_path,
        execute=False,
        port_record_path=tmp_path / "ports.md",
    )

    assert preview["status"] == "planned"
    assert preview["execute"] is False
    assert preview["command"]
    assert not (tmp_path / "gateway.pid").exists()


def test_gateway_doctor_reports_live_port_without_contacting_remote(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(gateway_module, "_dynamic_tcp_port_range", lambda: None)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    config_path = tmp_path / "gateway.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.model_gateway_config.v1",
                "host": "127.0.0.1",
                "port": port,
                "telemetry": False,
                "config_path": str(tmp_path / "generated.yaml"),
                "pid_path": str(tmp_path / "gateway.pid"),
                "log_path": str(tmp_path / "gateway.log"),
            }
        ),
        encoding="utf-8",
    )
    port_record = tmp_path / "ports.md"
    port_record.write_text(f"| {port} | VKP LiteLLM Proxy |\n", encoding="utf-8")
    try:
        result = model_gateway_doctor(
            gateway_config_path=config_path,
            port_record_path=port_record,
            probe_http=False,
        )
    finally:
        listener.close()

    checks = {row["key"]: row for row in result["checks"]}
    assert checks["loopback_host"]["ok"] is True
    assert checks["port_record"]["ok"] is True
    assert checks["live_listener"]["ok"] is True
    assert result["remote_requests_made"] is False


def test_gateway_doctor_requires_exact_vkp_port_registration(tmp_path: Path) -> None:
    config_path = tmp_path / "gateway.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.model_gateway_config.v1",
                "host": "127.0.0.1",
                "port": 18776,
                "telemetry": False,
                "config_path": str(tmp_path / "generated.yaml"),
                "pid_path": str(tmp_path / "gateway.pid"),
                "log_path": str(tmp_path / "gateway.log"),
            }
        ),
        encoding="utf-8",
    )
    port_record = tmp_path / "ports.md"
    port_record.write_text("187760 unrelated\n8776 VKP LiteLLM Proxy\n", encoding="utf-8")

    blocked = model_gateway_doctor(
        gateway_config_path=config_path,
        port_record_path=port_record,
        probe_http=False,
    )
    check = next(row for row in blocked["checks"] if row["key"] == "port_record")

    assert blocked["status"] == "blocked"
    assert check["recorded"] is False
    assert check["owned_by_vkp"] is False


def test_gateway_execute_refuses_empty_proxy_configuration(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    config_path = tmp_path / "gateway.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.model_gateway_config.v1",
                "host": "127.0.0.1",
                "port": 18776,
                "telemetry": False,
                "config_path": str(tmp_path / "generated.yaml"),
                "pid_path": str(tmp_path / "gateway.pid"),
                "log_path": str(tmp_path / "gateway.log"),
            }
        ),
        encoding="utf-8",
    )

    result = start_model_gateway(
        gateway_config_path=config_path,
        settings_path=settings_path,
        secrets_path=secrets_path,
        port_record_path=tmp_path / "ports.md",
        execute=True,
    )

    assert result["status"] == "configuration_blocked"
    assert result["render"]["model_count"] == 0
    assert not (tmp_path / "gateway.pid").exists()

def test_litellm_config_marks_asr_and_ocr_modes(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    output_path = tmp_path / "litellm.yaml"
    asr_profile = {
        **_remote_profile(),
        "id": "remote-asr",
        "name": "Remote ASR",
        "adapter_backend": "proxy",
        "provider": "openai_compatible_asr",
        "base_url": "https://asr-adapter.example/v1",
        "capabilities": ["asr"],
        "model": "asr-model",
    }
    upsert_model_api_profile(
        asr_profile,
        tasks=["asr"],
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    ocr_profile = {
        **asr_profile,
        "id": "remote-ocr",
        "name": "Remote OCR",
        "provider": "mistral_compatible_ocr",
        "base_url": "https://ocr-adapter.example/v1",
        "capabilities": ["ocr"],
        "model": "ocr-model",
    }
    upsert_model_api_profile(
        ocr_profile,
        tasks=["ocr"],
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    render_litellm_config(
        settings_path=settings_path,
        secrets_path=secrets_path,
        output_path=output_path,
        write=True,
    )
    rendered = output_path.read_text(encoding="utf-8")

    assert 'mode: "audio_transcription"' in rendered
    assert 'mode: "ocr"' in rendered

def test_litellm_config_renders_ordered_fallback_and_route_policy(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    output_path = tmp_path / "litellm.yaml"
    first = _remote_profile()
    first["rpm"] = 60
    first["tpm"] = 120_000
    first["max_parallel_requests"] = 2
    second = {**_remote_profile(), "id": "remote-second", "name": "Remote second", "base_url": "https://second.example/v1", "model": "second-model"}
    upsert_model_api_profile(first, tasks=["summary_rewrite"], settings_path=settings_path, secrets_path=secrets_path)
    status = upsert_model_api_profile(second, tasks=["summary_rewrite"], settings_path=settings_path, secrets_path=secrets_path)
    pool = dict(status["route_pools"][0])
    pool["deployments"] = ["remote-text", "remote-second"]
    pool["retry_policy"] = {"max_retries": 3, "timeout_seconds": 45, "cooldown_seconds": 17}
    settings_module.replace_model_api_route_configuration(
        [pool], status["route_bindings"], settings_path=settings_path, secrets_path=secrets_path
    )
    route = settings_module.resolve_model_api_route(
        "summary_rewrite", execution_location="remote", settings_path=settings_path
    )

    result = render_litellm_config(
        settings_path=settings_path,
        secrets_path=secrets_path,
        output_path=output_path,
        write=True,
    )

    fallback_name = f"{route['virtual_model']}-fallback-2"
    rendered = output_path.read_text(encoding="utf-8")
    assert result["fallback_chains"] == {route["virtual_model"]: [fallback_name]}
    assert rendered.index(f'model_name: "{route["virtual_model"]}"') < rendered.index(f'model_name: "{fallback_name}"')
    assert rendered.count("cooldown_time: 17") == 2
    assert "num_retries: 0" in rendered
    assert "enable_pre_call_checks: true" in rendered
    assert "rpm: 60" in rendered
    assert "tpm: 120000" in rendered
    assert "max_parallel_requests: 2" in rendered
    assert "max_fallbacks: 1" in rendered
    assert "turn_off_message_logging: true" in rendered
    assert "background_health_checks: false" in rendered

    previous_revision = route["route_revision"]
    first["rpm"] = 61
    upsert_model_api_profile(
        first,
        tasks=["summary_rewrite"],
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    revised = settings_module.resolve_model_api_route(
        "summary_rewrite", execution_location="remote", settings_path=settings_path
    )
    assert revised["route_revision"] != previous_revision


def test_gateway_injects_only_proxy_profile_secrets(tmp_path: Path, monkeypatch) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    proxy = _remote_profile()
    legacy = {
        **_remote_profile(),
        "id": "legacy-text",
        "name": "Legacy text",
        "adapter_backend": "legacy",
        "base_url": "https://legacy.example/v1",
    }
    upsert_model_api_profile(proxy, tasks=["summary_rewrite"], settings_path=settings_path, secrets_path=secrets_path)
    upsert_model_api_profile(legacy, tasks=["text_llm"], settings_path=settings_path, secrets_path=secrets_path)
    calls: list[str] = []

    def fake_read(profile_id: str, path: Path) -> str:
        calls.append(profile_id)
        return f"secret-{profile_id}"

    monkeypatch.setattr(gateway_module, "_read_secret", fake_read)
    environment = gateway_module._runtime_environment(settings_path, secrets_path)

    assert "remote-text" in calls
    assert "legacy-text" not in calls
    assert gateway_module.MASTER_KEY_ID in calls
    assert "VKP_LITELLM_PROFILE_REMOTE_TEXT_API_KEY" in environment
    assert "VKP_LITELLM_PROFILE_LEGACY_TEXT_API_KEY" not in environment
    assert environment["LITELLM_TELEMETRY"] == "False"
    assert environment["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"


def test_mistral_ocr_profile_renders_native_litellm_model(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    output_path = tmp_path / "litellm.yaml"
    upsert_model_api_profile(
        {
            "id": "mistral-ocr",
            "name": "Mistral OCR",
            "provider": "mistral",
            "adapter_backend": "proxy",
            "location": "remote",
            "capabilities": ["ocr"],
            "base_url": "https://api.mistral.ai/v1",
            "model": "mistral-ocr-latest",
        },
        tasks=["ocr"],
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    render_litellm_config(
        settings_path=settings_path,
        secrets_path=secrets_path,
        output_path=output_path,
        write=True,
    )

    rendered = output_path.read_text(encoding="utf-8")
    assert 'model: "mistral/mistral-ocr-latest"' in rendered
    assert 'mode: "ocr"' in rendered

def test_gateway_execute_refuses_unregistered_port(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    config_path = tmp_path / "gateway.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.model_gateway_config.v1",
                "host": "127.0.0.1",
                "port": 18776,
                "telemetry": False,
                "config_path": str(tmp_path / "generated.yaml"),
                "pid_path": str(tmp_path / "gateway.pid"),
                "log_path": str(tmp_path / "gateway.log"),
            }
        ),
        encoding="utf-8",
    )
    upsert_model_api_profile(
        {
            **_remote_profile(),
            "id": "local-text",
            "name": "Local Text",
            "location": "local",
            "base_url": "http://127.0.0.1:19000/v1",
        },
        tasks=["text_llm"],
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    result = start_model_gateway(
        gateway_config_path=config_path,
        settings_path=settings_path,
        secrets_path=secrets_path,
        port_record_path=tmp_path / "ports.md",
        execute=True,
    )

    assert result["status"] == "port_blocked"
    assert result["render"]["model_count"] == 1
    assert not (tmp_path / "gateway.pid").exists()

def test_gateway_doctor_rejects_unhealthy_live_listener(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "gateway.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.model_gateway_config.v1",
                "host": "127.0.0.1",
                "port": 18776,
                "telemetry": False,
                "config_path": str(tmp_path / "generated.yaml"),
                "pid_path": str(tmp_path / "gateway.pid"),
                "log_path": str(tmp_path / "gateway.log"),
            }
        ),
        encoding="utf-8",
    )
    port_record = tmp_path / "ports.md"
    port_record.write_text("18776 VKP LiteLLM Proxy\n", encoding="utf-8")
    monkeypatch.setattr(gateway_module, "_can_connect", lambda host, port: True)
    monkeypatch.setattr(gateway_module, "_probe_health", lambda host, port: "unreachable")

    result = model_gateway_doctor(
        gateway_config_path=config_path,
        port_record_path=port_record,
        probe_http=True,
    )

    assert result["status"] == "blocked"
    assert result["ready"] is False
    assert result["http_status"] == "unreachable"

def test_gateway_health_probe_rejects_generic_http_404(monkeypatch) -> None:
    class Generic404Response:
        status = 404

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(
        gateway_module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: Generic404Response(),
    )

    assert gateway_module._probe_health("127.0.0.1", 18776) == "unreachable"

def test_gateway_start_rejects_unhealthy_occupied_port(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "gateway.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.model_gateway_config.v1",
                "host": "127.0.0.1",
                "port": 18776,
                "telemetry": False,
                "config_path": str(tmp_path / "generated.yaml"),
                "pid_path": str(tmp_path / "gateway.pid"),
                "log_path": str(tmp_path / "gateway.log"),
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_doctor(**kwargs):
        captured.update(kwargs)
        return {
            "ready": False,
            "checks": [
                {"key": "live_listener", "ok": True},
                {"key": "bind_available", "ok": False},
                {"key": "port_record", "ok": True},
            ],
        }

    monkeypatch.setattr(
        gateway_module,
        "render_litellm_config",
        lambda **kwargs: {"model_count": 1},
    )
    monkeypatch.setattr(gateway_module, "model_gateway_doctor", fake_doctor)
    monkeypatch.setattr(gateway_module, "_litellm_command", lambda: ["litellm"])
    monkeypatch.setattr(
        gateway_module.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Popen must not run")),
    )

    result = start_model_gateway(
        gateway_config_path=config_path,
        settings_path=tmp_path / "settings.json",
        secrets_path=tmp_path / "secrets.json",
        port_record_path=tmp_path / "ports.md",
        execute=True,
    )

    assert result["status"] == "port_blocked"
    assert captured["probe_http"] is True
    assert not (tmp_path / "gateway.pid").exists()


def test_gateway_resolves_shared_secret_ref_without_copying_ciphertext(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    output_path = tmp_path / "litellm.yaml"
    monkeypatch.setattr(settings_module, "_protect_secret", lambda value: "cipher:" + value.encode().hex())
    monkeypatch.setattr(
        settings_module,
        "_unprotect_secret",
        lambda value: bytes.fromhex(value.removeprefix("cipher:")).decode(),
    )
    secret = "shared-provider-secret"
    source = _remote_profile()
    alias = {
        **_remote_profile(),
        "id": "remote-text-alias",
        "name": "Remote Text Alias",
        "secret_ref": "dpapi:remote-text",
    }
    upsert_model_api_profile(
        source,
        tasks=[],
        api_key=secret,
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    upsert_model_api_profile(
        alias,
        tasks=["summary_rewrite"],
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    rendered = render_litellm_config(
        settings_path=settings_path,
        secrets_path=secrets_path,
        output_path=output_path,
        write=True,
    )
    environment = gateway_module._runtime_environment(settings_path, secrets_path)

    assert rendered["credential_blockers"] == []
    assert "VKP_LITELLM_PROFILE_REMOTE_TEXT_ALIAS_API_KEY" in rendered["required_env_names"]
    assert environment["VKP_LITELLM_PROFILE_REMOTE_TEXT_ALIAS_API_KEY"] == secret
    assert "VKP_LITELLM_PROFILE_REMOTE_TEXT_API_KEY" not in environment
    items = json.loads(secrets_path.read_text(encoding="utf-8"))["items"]
    assert "remote-text" in items
    assert "remote-text-alias" not in items


def test_gateway_doctor_rejects_windows_dynamic_client_port(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "gateway.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.model_gateway_config.v1",
                "host": "127.0.0.1",
                "port": 8776,
                "telemetry": False,
                "config_path": str(tmp_path / "generated.yaml"),
                "pid_path": str(tmp_path / "gateway.pid"),
                "log_path": str(tmp_path / "gateway.log"),
            }
        ),
        encoding="utf-8",
    )
    port_record = tmp_path / "ports.md"
    port_record.write_text("8776 VKP LiteLLM Proxy\n", encoding="utf-8")
    monkeypatch.setattr(
        gateway_module,
        "_dynamic_tcp_port_range",
        lambda: {"start": 1024, "end": 15000, "count": 13977},
    )
    monkeypatch.setattr(gateway_module, "_can_connect", lambda host, port: False)
    monkeypatch.setattr(gateway_module, "_can_bind", lambda host, port: True)
    monkeypatch.setattr(gateway_module, "_optional_module_available", lambda name: True)

    result = model_gateway_doctor(
        gateway_config_path=config_path,
        port_record_path=port_record,
        probe_http=False,
    )

    checks = {row["key"]: row for row in result["checks"]}
    assert result["status"] == "blocked"
    assert checks["outside_dynamic_client_range"]["ok"] is False
    assert result["dynamic_client_port_range"]["end"] == 15000


def test_gateway_doctor_accepts_registered_port_outside_dynamic_range(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "gateway.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.model_gateway_config.v1",
                "host": "127.0.0.1",
                "port": 18776,
                "telemetry": False,
                "config_path": str(tmp_path / "generated.yaml"),
                "pid_path": str(tmp_path / "gateway.pid"),
                "log_path": str(tmp_path / "gateway.log"),
            }
        ),
        encoding="utf-8",
    )
    port_record = tmp_path / "ports.md"
    port_record.write_text("18776 VKP LiteLLM Proxy\n", encoding="utf-8")
    monkeypatch.setattr(
        gateway_module,
        "_dynamic_tcp_port_range",
        lambda: {"start": 1024, "end": 15000, "count": 13977},
    )
    monkeypatch.setattr(gateway_module, "_can_connect", lambda host, port: False)
    monkeypatch.setattr(gateway_module, "_can_bind", lambda host, port: True)
    monkeypatch.setattr(gateway_module, "_optional_module_available", lambda name: True)

    result = model_gateway_doctor(
        gateway_config_path=config_path,
        port_record_path=port_record,
        probe_http=False,
    )

    checks = {row["key"]: row for row in result["checks"]}
    assert checks["outside_dynamic_client_range"]["ok"] is True
    assert result["ready"] is True


def test_gateway_start_reports_route_render_failure_without_traceback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "gateway.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.model_gateway_config.v1",
                "host": "127.0.0.1",
                "port": 18776,
                "telemetry": False,
                "config_path": str(tmp_path / "generated.yaml"),
                "pid_path": str(tmp_path / "gateway.pid"),
                "log_path": str(tmp_path / "gateway.log"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        gateway_module,
        "render_litellm_config",
        lambda **kwargs: (_ for _ in ()).throw(
            ValueError("model route has no enabled deployments")
        ),
    )
    monkeypatch.setattr(
        gateway_module.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Popen must not run")
        ),
    )

    result = start_model_gateway(
        gateway_config_path=config_path,
        settings_path=tmp_path / "settings.json",
        secrets_path=tmp_path / "secrets.json",
        port_record_path=tmp_path / "ports.md",
        execute=True,
    )

    assert result["status"] == "configuration_blocked"
    assert result["error_code"] == "route_render_failed"
    assert "no enabled deployments" in result["error"]
    assert result["remote_requests_made"] is False
    assert not (tmp_path / "gateway.pid").exists()
