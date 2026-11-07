from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from video_knowledge_pipeline import model_api_settings as settings_module
from video_knowledge_pipeline.model_api_settings import (
    ONLINE_PRODUCTION_ROUTE_PRESET_ID,
    ONLINE_PRODUCTION_ROUTE_TASK_PROFILES,
    ONLINE_SCREENING_ROUTE_PRESET_ID,
    ONLINE_SCREENING_ROUTE_TASK_PROFILES,
    apply_model_api_route_preset,
    configured_remote_destination_status,
    delete_model_api_profile,
    install_model_api_onboarding_bundle,
    load_model_api_settings,
    prepare_model_api_onboarding_bundles,
    load_model_api_settings_ui_config,
    model_api_settings_ui_url,
    public_model_api_settings_status,
    resolve_model_api_provider_config,
    upsert_model_api_profile,
    validate_model_api_profile,
)
from video_knowledge_pipeline.model_api_settings_http import _render_html, build_server
from video_knowledge_pipeline.online_model_gateway import online_model_api_call
from video_knowledge_pipeline.task_console import _render_asr_runtime_settings_panel


@pytest.fixture
def fake_secret_codec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings_module, "_protect_secret", lambda value: "cipher:" + value.encode("utf-8").hex())
    monkeypatch.setattr(settings_module, "_unprotect_secret", lambda value: bytes.fromhex(value.removeprefix("cipher:")).decode("utf-8"))

def test_online_screening_route_preset_assigns_exact_single_deployment_models(
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    prepare_model_api_onboarding_bundles(settings_path=settings_path)

    status = apply_model_api_route_preset(
        ONLINE_SCREENING_ROUTE_PRESET_ID,
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    assert status["task_routes"] == ONLINE_SCREENING_ROUTE_TASK_PROFILES
    assert len(status["route_bindings"]) == 9
    assert all(len(pool["deployments"]) == 1 for pool in status["route_pools"])
    assert all("remote-ark" not in pool["deployments"] for pool in status["route_pools"])
    assert status["last_route_preset"] == {
        "preset_id": ONLINE_SCREENING_ROUTE_PRESET_ID,
        "task_count": 9,
        "pool_count": 8,
        "remote_destinations": [
            "api.groq.com",
            "api.mistral.ai",
            "api.siliconflow.cn",
            "ark.cn-beijing.volces.com",
            "generativelanguage.googleapis.com",
        ],
        "single_deployment_pools": True,
        "automatic_cross_destination_fallback": False,
        "saving_authorizes_egress": False,
    }


def test_online_production_route_preset_reuses_existing_standard_api_profiles(
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    prepare_model_api_onboarding_bundles(settings_path=settings_path)

    status = apply_model_api_route_preset(
        ONLINE_PRODUCTION_ROUTE_PRESET_ID,
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    assert status["task_routes"] == ONLINE_PRODUCTION_ROUTE_TASK_PROFILES
    assert len(status["route_bindings"]) == 9
    assert all(len(pool["deployments"]) == 1 for pool in status["route_pools"])
    assert all(
        not any(str(profile_id).startswith("ark-") for profile_id in pool["deployments"])
        for pool in status["route_pools"]
    )
    assert status["last_route_preset"] == {
        "preset_id": ONLINE_PRODUCTION_ROUTE_PRESET_ID,
        "task_count": 9,
        "pool_count": 6,
        "remote_destinations": [
            "api.groq.com",
            "api.mistral.ai",
            "api.siliconflow.cn",
            "generativelanguage.googleapis.com",
        ],
        "single_deployment_pools": True,
        "automatic_cross_destination_fallback": False,
        "saving_authorizes_egress": False,
    }

def test_settings_ui_exposes_explicit_catalog_probe_action() -> None:
    html = _render_html("csrf-test")
    assert "data-probe-bundle" in html
    assert "/api/onboarding-catalog/" in html
    assert "execute:true" in html
    assert "只读检查 Key 与模型目录" in html


def test_settings_ui_makes_key_only_onboarding_the_primary_path() -> None:
    html = _render_html("csrf-test")

    assert "快速接入（推荐：只填写 API Key）" in html
    assert "填写 API Key 并安装预设" in html
    assert "高级配置：手工编辑单个 Provider、URL、模型与 JSON 参数" in html
    assert html.index("快速接入（推荐：只填写 API Key）") < html.index("高级配置：")
    assert "providerOptionsHint" in html
    assert "字段契约：" in html
    assert "contract_sha256" in html


def test_settings_ui_exposes_litellm_capacity_and_redacted_batch_monitor() -> None:
    html = _render_html("csrf-test")

    assert 'id="rpm"' in html
    assert 'id="tpm"' in html
    assert 'id="maxParallelRequests"' in html
    assert "三项配额直接交给 LiteLLM Router" in html
    assert 'id="modelBatches"' in html
    assert "/api/model-batches" in html
    assert "批次层不自研动态限流" in html
    assert "dependency blocked" in html



def _profile(profile_id: str = "ark-main", *, name: str = "Ark 主力") -> dict[str, object]:
    return {
        "id": profile_id,
        "name": name,
        "provider": "volcengine_coding_plan",
        "adapter_backend": "builtin",
        "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
        "model": "ark-code-latest",
        "timeout_seconds": 180,
        "enabled": True,
    }


def test_configured_remote_destinations_include_only_enabled_https_profiles(
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "settings.json"
    profiles = [
        {
            **_profile("ark-enabled"),
            "adapter_backend": "proxy",
            "location": "remote",
            "capabilities": ["text"],
        },
        {
            "id": "modelscope-enabled",
            "name": "ModelScope enabled",
            "provider": "openai_compatible",
            "litellm_provider": "openai",
            "adapter_backend": "proxy",
            "base_url": "https://api-inference.modelscope.cn/v1",
            "model": "ZhipuAI/GLM-5.2",
            "location": "remote",
            "capabilities": ["text"],
            "timeout_seconds": 120,
            "enabled": True,
        },
        {
            "id": "remote-disabled",
            "name": "Remote disabled",
            "provider": "openai_compatible",
            "litellm_provider": "openai",
            "adapter_backend": "proxy",
            "base_url": "https://disabled.example/v1",
            "model": "disabled-model",
            "location": "remote",
            "capabilities": ["text"],
            "timeout_seconds": 120,
            "enabled": False,
        },
        {
            "id": "local-enabled",
            "name": "Local enabled",
            "provider": "local_openai_compatible",
            "litellm_provider": "openai",
            "adapter_backend": "proxy",
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "local-model",
            "location": "local",
            "capabilities": ["text"],
            "timeout_seconds": 120,
            "enabled": True,
        },
    ]
    for profile in profiles:
        upsert_model_api_profile(
            profile,
            tasks=[],
            settings_path=settings_path,
            secrets_path=tmp_path / "secrets.json",
        )

    status = configured_remote_destination_status(settings_path)

    assert status["destinations"] == [
        "api-inference.modelscope.cn",
        "ark.cn-beijing.volces.com",
    ]
    assert status["profile_ids"] == ["ark-enabled", "modelscope-enabled"]
    assert status["secrets_accessed"] is False
    assert status["api_keys_exposed"] is False
    assert status["remote_requests_made"] is False
    assert status["consent_still_required"] is True
    assert status["arbitrary_urls_allowed"] is False


def test_profile_store_encrypts_secret_and_routes_tasks(
    tmp_path: Path,
    fake_secret_codec: None,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    secret = "vk-test-secret-value"

    status = upsert_model_api_profile(
        _profile(),
        tasks=["ocr", "temporal_sequence"],
        api_key=secret,
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    assert status["task_routes"] == {"ocr": "ark-main", "temporal_sequence": "ark-main"}
    assert status["profiles"][0]["api_key_configured"] is True
    assert secret not in settings_path.read_text(encoding="utf-8")
    assert secret not in secrets_path.read_text(encoding="utf-8")
    assert secret not in json.dumps(status, ensure_ascii=False)

    resolved = resolve_model_api_provider_config(
        "ocr",
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    assert resolved["provider"] == "volcengine_coding_plan"
    assert resolved["api_key"] == secret
    assert resolved["profile_id"] == "ark-main"

    explicit = resolve_model_api_provider_config(
        "ocr",
        {"provider": "gemini", "model": "gemini-test"},
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    assert explicit == {"provider": "gemini", "model": "gemini-test"}

    deleted = delete_model_api_profile("ark-main", settings_path=settings_path, secrets_path=secrets_path)
    assert deleted["profiles"] == []
    assert deleted["task_routes"] == {}


def test_profile_updates_reassign_tasks_and_blank_key_preserves_secret(
    tmp_path: Path,
    fake_secret_codec: None,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    upsert_model_api_profile(
        _profile("ark-main"),
        tasks=["ocr", "summary_rewrite"],
        api_key="ark-secret",
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    gemini = {
        "id": "gemini-vision",
        "name": "Gemini 视觉",
        "provider": "gemini",
        "adapter_backend": "auto",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-2.5-flash",
        "timeout_seconds": 120,
        "enabled": True,
    }
    status = upsert_model_api_profile(
        gemini,
        tasks=["ocr"],
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    assert status["task_routes"] == {"summary_rewrite": "ark-main", "ocr": "gemini-vision"}

    upsert_model_api_profile(
        _profile("ark-main"),
        tasks=["summary_rewrite"],
        api_key="",
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    resolved = resolve_model_api_provider_config(
        "summary_rewrite",
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    assert resolved["api_key"] == "ark-secret"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://remote.example.com/v1",
        "https://*.example.com/v1",
        "https://user:password@example.com/v1",
        "https://example.com/v1?api_key=bad",
        "file:///tmp/provider",
    ],
)
def test_profile_validation_rejects_unsafe_base_urls(base_url: str) -> None:
    profile = _profile()
    profile["base_url"] = base_url
    with pytest.raises(ValueError):
        validate_model_api_profile(profile, ["ocr"])


def test_online_gateway_uses_task_profile_without_exposing_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_secret_codec: None,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    monkeypatch.setenv("VKP_MODEL_API_SETTINGS_PATH", str(settings_path))
    monkeypatch.setenv("VKP_MODEL_API_SECRETS_PATH", str(secrets_path))
    upsert_model_api_profile(
        _profile(),
        tasks=["ocr"],
        api_key="gateway-secret",
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    result = online_model_api_call("ocr", image_paths=["D:/not-read.png"], write=False)

    provider = result["request_plan"]["provider"]
    assert provider["provider"] == "volcengine_coding_plan"
    assert provider["model"] == "ark-code-latest"
    assert result["status"] == "planned"
    assert result["execute"] is False
    assert "gateway-secret" not in json.dumps(result, ensure_ascii=False)


def test_loopback_settings_http_ui_saves_and_redacts(
    tmp_path: Path,
    fake_secret_codec: None,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    server = build_server(
        host="127.0.0.1",
        port=0,
        settings_path=settings_path,
        secrets_path=secrets_path,
        csrf_token="test-csrf-token",
        project_root_path=tmp_path,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        page = urllib.request.urlopen(base + "/", timeout=5).read().decode("utf-8")
        assert "保存到本机" in page
        assert "Windows DPAPI" in page
        assert "保存配置不等于授权外发" in page
        assert "路由池与任务默认位置" in page
        assert "LiteLLM Proxy" in page
        assert "freeOnboarding" in page
        assert "快速接入（推荐：只填写 API Key）" in page
        assert "data-install-bundle" in page
        assert "Suggested tasks" in page
        assert "Provider RPM" in page
        assert "在线模型批次运行状态" in page
        batches = json.loads(urllib.request.urlopen(base + "/api/model-batches", timeout=5).read())
        assert batches["provider_rate_limit_owner"] == "litellm_proxy"

        payload = json.dumps(
            {
                "profile": _profile(),
                "tasks": ["ocr", "text_llm"],
                "api_key": "http-secret",
                "remove_api_key": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            base + "/api/profile",
            data=payload,
            method="PUT",
            headers={"Content-Type": "application/json", "X-VKP-Settings-Token": "test-csrf-token"},
        )
        saved = json.loads(urllib.request.urlopen(request, timeout=5).read().decode("utf-8"))
        assert saved["ok"] is True
        assert saved["settings"]["profiles"][0]["api_key_configured"] is True
        assert "http-secret" not in json.dumps(saved, ensure_ascii=False)
        assert "http-secret" not in secrets_path.read_text(encoding="utf-8")
        bundle_secret = "one-key-groq-http-secret"
        bundle_request = urllib.request.Request(
            base + "/api/onboarding/groq",
            data=json.dumps({"api_key": bundle_secret}).encode("utf-8"),
            method="PUT",
            headers={"Content-Type": "application/json", "X-VKP-Settings-Token": "test-csrf-token"},
        )
        bundle_saved = json.loads(
            urllib.request.urlopen(bundle_request, timeout=5).read().decode("utf-8")
        )
        assert bundle_saved["ok"] is True
        installed_ids = {
            row["id"]
            for row in bundle_saved["settings"]["profiles"]
        }
        assert {
            "groq-qwen3-6-27b",
            "groq-whisper-large-v3-turbo",
        }.issubset(installed_ids)
        assert bundle_saved["settings"]["last_onboarding_install"]["profile_count"] == 2
        assert bundle_saved["settings"]["last_onboarding_install"]["route_configuration_changed"] is False
        assert bundle_secret not in json.dumps(bundle_saved)
        assert bundle_secret not in secrets_path.read_text(encoding="utf-8")


        bad_request = urllib.request.Request(
            base + "/api/profile",
            data=payload,
            method="PUT",
            headers={"Content-Type": "application/json", "X-VKP-Settings-Token": "wrong"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(bad_request, timeout=5)
        assert exc.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_settings_ui_config_and_generated_console_link() -> None:
    configured = load_model_api_settings_ui_config()
    assert configured["host"] == "127.0.0.1"
    assert configured["port"] == 8767
    assert model_api_settings_ui_url() == "http://127.0.0.1:8767/"

    rendered = _render_asr_runtime_settings_panel({}, [], "<bundle_dir>")
    assert "打开可保存的 API 设置界面" in rendered
    assert "http://127.0.0.1:8767/" in rendered
    assert "start-model-api-settings.ps1" in rendered
    assert "ASR Runtime 设置" in rendered


def test_public_status_returns_paths_and_never_ciphertext(tmp_path: Path) -> None:
    status = public_model_api_settings_status(tmp_path / "settings.json", tmp_path / "secrets.json")
    assert status["profiles"] == []
    assert status["secret_storage"]["plaintext_persisted"] is False
    assert status["settings_ui_url"] == "http://127.0.0.1:8767/"
    assert status["provider_catalog"]["provider_count"] >= 41
    assert status["provider_catalog"]["extension_provider"] == "litellm_native"
    assert status["provider_catalog"]["secrets_in_catalog"] is False
    gemini_preset = next(
        row for row in status["provider_presets"] if row["provider"] == "gemini"
    )
    assert gemini_preset["default_model"] == "gemini-3.6-flash"
    assert status["free_screening_onboarding"]["network_calls"] is False
    assert status["free_screening_onboarding"]["secrets_exposed"] is False
    assert status["free_screening_onboarding"]["saving_authorizes_egress"] is False
    assert len(status["free_screening_onboarding"]["entries"]) == 10


def test_new_profile_defaults_to_proxy_and_legacy_migration_stays_legacy(tmp_path: Path) -> None:
    profile = _profile()
    profile.pop("adapter_backend")
    validated = validate_model_api_profile(profile, ["summary_rewrite"])
    assert validated["profile"]["adapter_backend"] == "proxy"

    settings_path = tmp_path / "legacy.json"
    settings_path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.local_model_api_settings.v1",
                "profiles": [profile],
                "task_routes": {"summary_rewrite": "ark-main"},
                "updated_at": "",
            }
        ),
        encoding="utf-8",
    )
    migrated = public_model_api_settings_status(settings_path, tmp_path / "secrets.json")
    assert migrated["profiles"][0]["adapter_backend"] == "legacy"
    assert json.loads(settings_path.read_text(encoding="utf-8"))["schema"].endswith(".v1")

def test_settings_http_updates_ordered_route_pool_and_reports_security_state(
    tmp_path: Path,
    fake_secret_codec: None,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    first = _profile("remote-first", name="Remote first")
    first["adapter_backend"] = "proxy"
    first["capabilities"] = ["text"]
    first["location"] = "remote"
    second = _profile("remote-second", name="Remote second")
    second["adapter_backend"] = "proxy"
    second["capabilities"] = ["text"]
    second["location"] = "remote"
    upsert_model_api_profile(first, tasks=["summary_rewrite"], settings_path=settings_path, secrets_path=secrets_path)
    status = upsert_model_api_profile(second, tasks=["summary_rewrite"], settings_path=settings_path, secrets_path=secrets_path)
    pool = status["route_pools"][0]
    pool["deployments"] = ["remote-first", "remote-second"]
    server = build_server(
        host="127.0.0.1",
        port=0,
        settings_path=settings_path,
        secrets_path=secrets_path,
        csrf_token="route-token",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        payload = json.dumps(
            {
                "route_pools": [pool],
                "route_bindings": status["route_bindings"],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            base + "/api/routes",
            data=payload,
            method="PUT",
            headers={"Content-Type": "application/json", "X-VKP-Settings-Token": "route-token"},
        )
        saved = json.loads(urllib.request.urlopen(request, timeout=5).read().decode("utf-8"))["settings"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert saved["route_pools"][0]["deployments"] == ["remote-first", "remote-second"]
    route_status = next(
        row
        for row in saved["route_status"]
        if row["task"] == "summary_rewrite" and row["execution_location"] == "remote"
    )
    assert route_status["route_revision"]
    assert route_status["estimated_cost"] == "unknown"
    assert route_status["consent_required"] is True
    assert route_status["allowlist_status"] in {"unknown", "approved", "blocked"}


def test_key_once_onboarding_bundle_preserves_routes_and_hides_secret(
    tmp_path: Path,
    fake_secret_codec: None,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    baseline = upsert_model_api_profile(
        _profile(),
        tasks=["summary_rewrite"],
        api_key="existing-route-secret",
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    route_pools_before = baseline["route_pools"]
    route_bindings_before = baseline["route_bindings"]
    bundle_secret = "one-key-modelscope-secret"

    status = install_model_api_onboarding_bundle(
        "modelscope",
        api_key=bundle_secret,
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    profiles = {row["id"]: row for row in status["profiles"]}
    assert profiles["modelscope-glm-5-2"]["model"] == "ZhipuAI/GLM-5.2"
    assert profiles["modelscope-deepseek-v4-pro"]["model"] == "deepseek-ai/DeepSeek-V4-Pro"
    assert profiles["modelscope-glm-5-2"]["api_key_configured"] is True
    assert profiles["modelscope-deepseek-v4-pro"]["api_key_configured"] is True
    assert status["route_pools"] == route_pools_before
    assert status["route_bindings"] == route_bindings_before
    assert status["last_onboarding_install"] == {
        "provider_id": "modelscope",
        "profile_ids": ["modelscope-deepseek-v4-pro", "modelscope-glm-5-2"],
        "profile_count": 2,
        "route_configuration_changed": False,
        "network_calls": False,
        "saving_authorizes_egress": False,
        "secret_values_exposed": False,
    }
    assert bundle_secret not in settings_path.read_text(encoding="utf-8")
    assert bundle_secret not in secrets_path.read_text(encoding="utf-8")
    assert bundle_secret not in json.dumps(status)


def test_prepare_all_exact_onboarding_bundles_reads_only_secret_metadata(
    tmp_path: Path,
    fake_secret_codec: None,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    baseline = upsert_model_api_profile(
        _profile(),
        tasks=["summary_rewrite"],
        api_key="existing-route-secret",
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    secret_bytes_before = secrets_path.read_bytes()

    report = prepare_model_api_onboarding_bundles(
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    saved = load_model_api_settings(settings_path)
    profiles = {row["id"]: row for row in saved["profiles"]}

    assert report["provider_ids"] == [
        "siliconflow",
        "modelscope",
        "groq",
        "ark_coding_plan",
        "ark_model_api",
        "google_gemini",
        "mistral",
    ]
    assert report["profile_count"] == 29
    assert report["network_calls"] is False
    assert report["secrets_accessed"] is True
    assert report["secret_values_accessed"] is False
    assert report["secrets_decrypted"] is False
    assert report["route_configuration_changed"] is False
    assert saved["route_pools"] == baseline["route_pools"]
    assert saved["route_bindings"] == baseline["route_bindings"]
    assert secrets_path.read_bytes() == secret_bytes_before
    assert profiles["mistral-ocr-4-0"]["capabilities"] == ["ocr"]
    assert profiles["groq-whisper-large-v3-turbo"]["model"] == "whisper-large-v3-turbo"
    assert profiles["google-gemini-3-6-flash"]["model"] == "gemini-3.6-flash"
    assert profiles["google-gemini-3-5-flash-lite"]["model"] == "gemini-3.5-flash-lite"
    assert "modelscope-deepseek-v4-pro" in profiles
    assert "ark-deepseek-v4-pro" in profiles
    assert profiles["ark-api-deepseek-v4-pro"]["provider"] == "volcengine_ark"
    assert profiles["ark-api-deepseek-v4-pro"]["base_url"] == "https://ark.cn-beijing.volces.com/api/v3"


def test_key_once_ark_model_api_bundle_uses_one_entered_key_for_three_profiles(
    tmp_path: Path,
    fake_secret_codec: None,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    secret = "ark-standard-api-key"

    status = install_model_api_onboarding_bundle(
        "ark_model_api",
        api_key=secret,
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    profiles = {row["id"]: row for row in status["profiles"]}
    assert set(profiles) == {
        "ark-api-deepseek-v4-flash",
        "ark-api-deepseek-v4-pro",
        "ark-api-glm-5-2",
    }
    assert {row["model"] for row in profiles.values()} == {
        "deepseek-v4-pro-260425",
        "deepseek-v4-flash-260425",
        "glm-5-2-260617",
    }
    assert all(row["provider"] == "volcengine_ark" for row in profiles.values())
    assert all(row["litellm_provider"] == "volcengine" for row in profiles.values())
    assert all(row["base_url"] == "https://ark.cn-beijing.volces.com/api/v3" for row in profiles.values())
    assert all(row["api_key_configured"] is True for row in profiles.values())
    assert status["route_pools"] == []
    assert status["route_bindings"] == {}
    assert secret not in settings_path.read_text(encoding="utf-8")
    assert secret not in secrets_path.read_text(encoding="utf-8")
    assert secret not in json.dumps(status)


def test_key_once_ark_coding_plan_bundle_is_text_only_and_secretless(
    tmp_path: Path,
    fake_secret_codec: None,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    secret = "ark-key-once-secret"

    status = install_model_api_onboarding_bundle(
        "ark_coding_plan",
        api_key=secret,
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    profiles = {row["id"]: row for row in status["profiles"]}
    assert {row["model"] for row in profiles.values()} == {
        "doubao-seed-2.0-pro", "doubao-seed-2.0-lite", "deepseek-v4-pro", "deepseek-v4-flash", "minimax-m3",
        "minimax-m2.7", "glm-5.2", "kimi-k2.7-code", "kimi-k2.6",
        "kimi-k3",
    }
    assert all(row["provider"] == "volcengine_coding_plan" for row in profiles.values())
    assert all(row["base_url"] == "https://ark.cn-beijing.volces.com/api/coding/v3" for row in profiles.values())
    assert all(row["capabilities"] == ["text"] for row in profiles.values())
    assert all(row["api_key_configured"] is True for row in profiles.values())
    assert status["route_pools"] == []
    assert profiles["ark-minimax-m3"]["enabled"] is False
    assert profiles["ark-deepseek-v4-flash"]["provider_options"] == {
        "thinking_mode": "disabled"
    }
    assert profiles["ark-minimax-m2-7"]["enabled"] is False
    assert status["route_bindings"] == {}
    assert status["last_onboarding_install"]["profile_ids"] == [
        "ark-deepseek-v4-flash",
        "ark-deepseek-v4-pro",
        "ark-doubao-seed-2-0-lite",
        "ark-doubao-seed-2-0-pro",
        "ark-glm-latest",
        "ark-kimi-k2-6",
        "ark-kimi-k2-7-code",
        "ark-kimi-k3",
        "ark-minimax-m2-7",
        "ark-minimax-m3",
    ]
    assert status["last_onboarding_install"]["route_configuration_changed"] is False
    assert secret not in settings_path.read_text(encoding="utf-8")
    assert secret not in secrets_path.read_text(encoding="utf-8")
    assert secret not in json.dumps(status)

def test_prepare_can_refresh_only_reviewed_obsolete_models(
    tmp_path: Path,
    fake_secret_codec: None,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    legacy = {
        "id": "ark-deepseek-v4-pro",
        "name": "Ark DeepSeek V4 Pro",
        "provider": "volcengine_coding_plan",
        "litellm_provider": "openai",
        "adapter_backend": "proxy",
        "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
        "model": "deepseek-v4-pro-260425",
        "location": "remote",
        "capabilities": ["text"],
        "timeout_seconds": 120,
        "enabled": True,
    }
    upsert_model_api_profile(
        legacy,
        tasks=[],
        api_key="preserved-secret",
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    with pytest.raises(ValueError, match="existing custom profile was not overwritten"):
        prepare_model_api_onboarding_bundles(["ark_coding_plan"], settings_path=settings_path)

    result = prepare_model_api_onboarding_bundles(
        ["ark_coding_plan"],
        settings_path=settings_path,
        refresh_known_models=True,
    )
    saved = {row["id"]: row for row in load_model_api_settings(settings_path)["profiles"]}
    assert saved["ark-deepseek-v4-pro"]["model"] == "deepseek-v4-pro"
    assert result["known_models_refreshed"][0]["from_model"] == "deepseek-v4-pro-260425"
    assert secrets_path.is_file()



def test_prepare_reuses_dpapi_reference_only_for_same_provider_and_base_url(
    tmp_path: Path,
    fake_secret_codec: None,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    upsert_model_api_profile(
        _profile("ark-existing"),
        tasks=[],
        api_key="existing-ark-secret",
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    secret_bytes_before = secrets_path.read_bytes()

    report = prepare_model_api_onboarding_bundles(
        ["ark_coding_plan"],
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    saved = {row["id"]: row for row in load_model_api_settings(settings_path)["profiles"]}

    assert report["secrets_accessed"] is True
    assert report["secret_values_accessed"] is False
    assert report["secrets_decrypted"] is False
    assert set(report["credential_reference_reused_profile_ids"]) == set(report["profile_ids"])
    assert all(saved[profile_id]["secret_ref"] == "dpapi:ark-existing" for profile_id in report["profile_ids"])
    assert secrets_path.read_bytes() == secret_bytes_before


def test_prepare_never_reuses_dangling_dpapi_reference(
    tmp_path: Path,
    fake_secret_codec: None,
) -> None:
    settings_path = tmp_path / "settings.json"
    source = _profile("ark-dangling")
    upsert_model_api_profile(source, tasks=[], settings_path=settings_path)

    report = prepare_model_api_onboarding_bundles(
        ["ark_coding_plan"],
        settings_path=settings_path,
    )
    saved = {row["id"]: row for row in load_model_api_settings(settings_path)["profiles"]}

    assert report["credential_reference_reused_profile_ids"] == []
    assert report["credential_ready_profile_ids"] == []
    assert set(report["credential_missing_profile_ids"]) == set(report["profile_ids"])
    assert all(
        saved[profile_id]["secret_ref"] == f"dpapi:{profile_id}"
        for profile_id in report["profile_ids"]
    )


def test_prepare_never_reuses_reference_across_different_base_urls(
    tmp_path: Path,
    fake_secret_codec: None,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    source = _profile("ark-different-base")
    source["base_url"] = "https://ark.cn-beijing.volces.com/api/v3"
    upsert_model_api_profile(
        source,
        tasks=[],
        api_key="different-route-secret",
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    secret_bytes_before = secrets_path.read_bytes()

    report = prepare_model_api_onboarding_bundles(
        ["ark_coding_plan"],
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    saved = {row["id"]: row for row in load_model_api_settings(settings_path)["profiles"]}

    assert report["credential_reference_reused_profile_ids"] == []
    assert all(
        saved[profile_id]["secret_ref"] == f"dpapi:{profile_id}"
        for profile_id in report["profile_ids"]
    )
    assert all(
        saved[profile_id]["secret_ref"] != "dpapi:ark-different-base"
        for profile_id in report["profile_ids"]
    )
    assert secrets_path.read_bytes() == secret_bytes_before

def test_key_once_onboarding_does_not_overwrite_custom_profile(
    tmp_path: Path,
    fake_secret_codec: None,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    custom = _profile("modelscope-glm-5-2", name="Operator custom GLM route")
    custom.update(
        {
            "provider": "openai_compatible",
            "litellm_provider": "openai",
            "adapter_backend": "proxy",
            "base_url": "https://api-inference.modelscope.cn/v1",
            "model": "operator-custom-model",
            "location": "remote",
            "capabilities": ["text"],
        }
    )
    upsert_model_api_profile(custom, tasks=[], settings_path=settings_path, secrets_path=secrets_path)

    with pytest.raises(ValueError, match="existing custom profile was not overwritten"):
        install_model_api_onboarding_bundle(
            "modelscope",
            api_key="must-not-be-written",
            settings_path=settings_path,
            secrets_path=secrets_path,
        )
    assert not secrets_path.exists()


def test_key_once_onboarding_preserves_legacy_glm_5_1_profile(
    tmp_path: Path,
    fake_secret_codec: None,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    legacy = _profile("modelscope-glm-5-1", name="ModelScope GLM-5.1")
    legacy.update(
        {
            "provider": "openai_compatible",
            "litellm_provider": "openai",
            "adapter_backend": "proxy",
            "base_url": "https://api-inference.modelscope.cn/v1",
            "model": "ZhipuAI/GLM-5.1",
            "location": "remote",
            "capabilities": ["text"],
        }
    )
    upsert_model_api_profile(
        legacy,
        tasks=[],
        api_key="legacy-secret",
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    status = install_model_api_onboarding_bundle(
        "modelscope",
        api_key="new-bundle-secret",
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    profiles = {row["id"]: row for row in status["profiles"]}
    assert profiles["modelscope-glm-5-1"]["model"] == "ZhipuAI/GLM-5.1"
    assert profiles["modelscope-glm-5-2"]["model"] == "ZhipuAI/GLM-5.2"
    assert profiles["modelscope-deepseek-v4-pro"]["model"] == "deepseek-ai/DeepSeek-V4-Pro"
    assert profiles["modelscope-glm-5-1"]["api_key_configured"] is True
    assert profiles["modelscope-glm-5-2"]["api_key_configured"] is True


def test_shared_secret_ref_is_preserved_and_resolved(
    tmp_path: Path,
    fake_secret_codec: None,
) -> None:
    settings_path = tmp_path / "settings.json"
    secrets_path = tmp_path / "secrets.json"
    secret = "shared-ark-secret"
    upsert_model_api_profile(
        _profile("ark-source"),
        tasks=[],
        api_key=secret,
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    alias = {
        **_profile("ark-alias", name="Ark alias"),
        "secret_ref": "dpapi:ark-source",
    }

    status = upsert_model_api_profile(
        alias,
        tasks=["summary_rewrite"],
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    profiles = {row["id"]: row for row in status["profiles"]}
    assert profiles["ark-alias"]["secret_ref"] == "dpapi:ark-source"
    assert profiles["ark-alias"]["api_key_configured"] is True
    secret_items = json.loads(secrets_path.read_text(encoding="utf-8"))["items"]
    assert set(secret_items) == {"ark-source"}
    resolved = resolve_model_api_provider_config(
        "summary_rewrite",
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    assert resolved["profile_id"] == "ark-alias"
    assert resolved["api_key"] == secret

    with pytest.raises(ValueError, match="shared secret_ref cannot be removed"):
        upsert_model_api_profile(
            alias,
            tasks=["summary_rewrite"],
            remove_api_key=True,
            settings_path=settings_path,
            secrets_path=secrets_path,
        )


def test_secret_ref_rejects_non_dpapi_and_invalid_profile_ids() -> None:
    with pytest.raises(ValueError, match="secret_ref must use dpapi"):
        validate_model_api_profile(
            {**_profile(), "secret_ref": "env:ARK_API_KEY"},
            [],
        )
    with pytest.raises(ValueError, match="profile id must contain"):
        validate_model_api_profile(
            {**_profile(), "secret_ref": "dpapi:../ark"},
            [],
        )
