from __future__ import annotations

import json
from urllib.parse import urlsplit

from video_knowledge_pipeline.model_provider_onboarding import (
    FREE_SCREENING_PROVIDERS,
    ONBOARDING_SCHEMA,
    free_screening_onboarding_status,
    provider_onboarding_definition,
    validate_provider_onboarding_prefills,
)


def test_free_screening_catalog_is_secretless_https_metadata() -> None:
    ids = {row["id"] for row in FREE_SCREENING_PROVIDERS}
    assert ids == {
        "ark_model_api",
        "ark_coding_plan",
        "github_models",
        "google_gemini",
        "groq",
        "mistral",
        "modelscope",
        "nvidia_nim",
        "openrouter",
        "siliconflow",
    }

    for row in FREE_SCREENING_PROVIDERS:
        for field in ("account_url", "credential_url", "documentation_url"):
            parsed = urlsplit(row[field])
            assert parsed.scheme == "https"
            assert parsed.hostname
        for template in row.get("profile_templates") or []:
            assert template["enabled"] is False
            assert template["location"] == "remote"
            assert template["model"]
        if row.get("profile_templates"):
            contract = row["prefill_contract"]
            assert contract["field_authority"] == "official_provider_documentation"
            assert len(contract["contract_sha256"]) == 64
            assert contract["automatic_catalog_updates"] is False
        assert row["key_once_ready"] is bool(row.get("profile_templates"))

    rendered = json.dumps(FREE_SCREENING_PROVIDERS, sort_keys=True)
    assert "Bearer " not in rendered
    assert "sk-" not in rendered


def test_key_only_prefills_match_versioned_provider_field_contracts() -> None:
    result = validate_provider_onboarding_prefills()

    assert result == {
        "ok": True,
        "contract_count": 7,
        "template_count": 30,
        "automatic_catalog_updates": False,
    }
    siliconflow = provider_onboarding_definition("siliconflow")
    assert all(
        "thinking_mode" not in (row.get("provider_options") or {})
        for row in siliconflow["profile_templates"]
    )
    ark = provider_onboarding_definition("ark_coding_plan")
    assert all(
        "enable_thinking" not in (row.get("provider_options") or {})
        for row in ark["profile_templates"]
    )
    deepseek_flash = next(
        row for row in ark["profile_templates"] if row["id"] == "ark-deepseek-v4-flash"
    )
    assert deepseek_flash["provider_options"] == {"thinking_mode": "disabled"}



def test_key_once_bundles_pin_exact_reviewed_model_ids() -> None:
    expected = {
        "ark_coding_plan": [
            "doubao-seed-2.0-pro",
            "doubao-seed-2.0-lite",
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "minimax-m3",
            "minimax-m2.7",
            "glm-5.2",
            "kimi-k2.7-code",
            "kimi-k2.6",
            "kimi-k3",
        ],
        "ark_model_api": [
            "deepseek-v4-pro-260425",
            "deepseek-v4-flash-260425",
            "glm-5-2-260617",
        ],
        "google_gemini": ["gemini-3.6-flash", "gemini-3.5-flash-lite"],
        "modelscope": ["ZhipuAI/GLM-5.2", "deepseek-ai/DeepSeek-V4-Pro"],
        "groq": ["qwen/qwen3.6-27b", "whisper-large-v3-turbo"],
        "mistral": ["voxtral-mini-2602", "mistral-ocr-4-0"],
        "siliconflow": [
            "TeleAI/TeleSpeechASR",
            "Qwen/Qwen3.5-4B",
            "zai-org/GLM-4.5V",
            "PaddlePaddle/PaddleOCR-VL-1.5",
            "deepseek-ai/DeepSeek-V4-Pro",
            "deepseek-ai/DeepSeek-V4-Flash",
            "zai-org/GLM-5.2",
            "moonshotai/Kimi-K2.7-Code",
            "Pro/moonshotai/Kimi-K2.6",
        ],
    }
    for provider_id, model_ids in expected.items():
        definition = provider_onboarding_definition(provider_id)
        assert [row["model"] for row in definition["profile_templates"]] == model_ids

    ark_api = provider_onboarding_definition("ark_model_api")["profile_templates"]
    assert all(row["provider"] == "volcengine_ark" for row in ark_api)
    assert all(row["litellm_provider"] == "volcengine" for row in ark_api)
    assert all(row["base_url"] == "https://ark.cn-beijing.volces.com/api/v3" for row in ark_api)
    assert all(row["catalog_status"] == "verified_visible" for row in ark_api)

    ark = provider_onboarding_definition("ark_coding_plan")["profile_templates"]
    unavailable = {row["model"] for row in ark if not row["install_enabled"]}
    assert unavailable == {"minimax-m3", "minimax-m2.7"}
    assert all(
        row["catalog_status"] == "not_visible_in_account_catalog"
        for row in ark if not row["install_enabled"]
    )


    groq = provider_onboarding_definition("groq")["profile_templates"]
    qwen = next(row for row in groq if row["id"] == "groq-qwen3-6-27b")
    whisper = next(
        row for row in groq if row["id"] == "groq-whisper-large-v3-turbo"
    )
    assert whisper["litellm_provider"] == "openai"
    assert whisper["provider_options"] == {
        "asr_timestamp_granularity": "word"
    }
    assert qwen["provider_options"] == {
        "reasoning_effort": "none",
        "reasoning_format": "hidden",
        "strip_reasoning_tags": True,
        "strip_json_fences": True,
    }

def test_onboarding_starts_blocked_without_profiles_or_network_calls() -> None:
    result = free_screening_onboarding_status([], [])

    assert result["schema"] == ONBOARDING_SCHEMA
    assert result["network_calls"] is False
    assert result["secrets_exposed"] is False
    assert result["saving_authorizes_egress"] is False
    by_id = {row["id"]: row for row in result["entries"]}
    assert by_id["siliconflow"]["status"] == "not_started"
    assert by_id["siliconflow"]["blockers"] == [
        "create_profile",
        "add_credential",
        "configure_route",
    ]
    assert len(by_id["siliconflow"]["expected_profile_ids"]) == 9
    assert by_id["github_models"]["status"] == "adapter_required"
    assert by_id["github_models"]["profile_templates"] == []


def test_onboarding_reports_ready_for_consent_without_claiming_consent() -> None:
    profiles = [
        {
            "id": "google-gemini-3-6-flash",
            "provider": "gemini",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "model": "gemini-3.6-flash",
            "api_key_configured": True,
            "credential_status": "ready",
        },
        {
            "id": "google-gemini-3-5-flash-lite",
            "provider": "gemini",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "model": "gemini-3.5-flash-lite",
            "api_key_configured": True,
            "credential_status": "ready",
        }
    ]
    routes = [
        {
            "task": "text_llm",
            "execution_location": "remote",
            "deployments": ["google-gemini-3-6-flash"],
            "allowlist_status": "approved",
        }
    ]

    result = free_screening_onboarding_status(profiles, routes)
    row = next(item for item in result["entries"] if item["id"] == "google_gemini")

    assert row["status"] == "ready_for_consent"
    assert row["installed_profile_ids"] == [
        "google-gemini-3-5-flash-lite",
        "google-gemini-3-6-flash",
    ]
    assert row["profile_saved"] is True
    assert row["model_selected"] is True
    assert row["credential_configured"] is True
    assert row["route_configured"] is True
    assert row["allowlist_statuses"] == ["approved"]
    assert row["consent_status"] == "not_checked"
    assert row["blockers"] == ["create_consent"]
