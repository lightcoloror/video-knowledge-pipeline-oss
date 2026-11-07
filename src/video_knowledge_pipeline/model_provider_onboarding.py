from __future__ import annotations

import copy
from typing import Any
from urllib.parse import urlsplit


from .canonical_json import canonical_json_sha256
from .model_defaults import GEMINI_DEFAULT_MODEL, GEMINI_THROUGHPUT_MODEL

ONBOARDING_SCHEMA = "video_knowledge_pipeline.model_provider_onboarding.v2"


def _profile_template(
    profile_id: str,
    name: str,
    *,
    provider: str,
    litellm_provider: str,
    base_url: str,
    capabilities: tuple[str, ...],
    model: str = "",
    provider_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": profile_id,
        "name": name,
        "provider": provider,
        "litellm_provider": litellm_provider,
        "provider_options": dict(provider_options or {}),
        "adapter_backend": "proxy",
        "location": "remote",
        "capabilities": list(capabilities),
        "base_url": base_url,
        "model": model,
        "timeout_seconds": 120,
        "enabled": False,
    }


def _definition(
    provider_id: str,
    label: str,
    priority: str,
    capabilities: tuple[str, ...],
    recommended_tasks: tuple[str, ...],
    *,
    account_url: str,
    credential_url: str,
    documentation_url: str,
    free_tier_note: str,
    data_boundary: str,
    runtime_integration: str,
    match_providers: tuple[str, ...],
    destination: str,
    profile_template: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "id": provider_id,
        "label": label,
        "priority": priority,
        "capabilities": list(capabilities),
        "recommended_tasks": list(recommended_tasks),
        "account_url": account_url,
        "credential_url": credential_url,
        "documentation_url": documentation_url,
        "free_tier_note": free_tier_note,
        "data_boundary": data_boundary,
        "runtime_integration": runtime_integration,
        "match_providers": list(match_providers),
        "destination": destination,
        "profile_template": profile_template,
    }


FREE_SCREENING_PROVIDERS: tuple[dict[str, Any], ...] = (
    _definition(
        "siliconflow",
        "SiliconFlow",
        "P0",
        ("text", "vision", "asr"),
        ("text_llm", "semantic_frame", "temporal_sequence", "asr"),
        account_url="https://cloud.siliconflow.cn/",
        credential_url="https://cloud.siliconflow.cn/account/ak",
        documentation_url="https://docs.siliconflow.cn/cn/api-reference/audio/create-audio-transcriptions",
        free_tier_note="Credits and eligible models change; verify the live console.",
        data_boundary="Selected text, images, or explicitly consented audio go to api.siliconflow.cn; allowlist and consent remain mandatory.",
        runtime_integration="preset_ready",
        match_providers=("siliconflow",),
        destination="api.siliconflow.cn",
        profile_template=_profile_template(
            "siliconflow-screening",
            "SiliconFlow screening",
            provider="siliconflow",
            litellm_provider="openai",
            base_url="https://api.siliconflow.cn/v1",
            capabilities=("text", "vision"),
        ),
    ),
    _definition(
        "modelscope",
        "ModelScope",
        "P0",
        ("text",),
        ("text_llm", "summary_rewrite", "transcript_correction"),
        account_url="https://modelscope.cn/",
        credential_url="https://modelscope.cn/my/myaccesstoken",
        documentation_url="https://modelscope.cn/docs/model-service/API-Inference/intro",
        free_tier_note="Only use models currently marked for API-Inference; quotas change.",
        data_boundary="Selected text goes to api-inference.modelscope.cn; this draft does not claim vision, ASR, or OCR.",
        runtime_integration="openai_compatible_candidate",
        match_providers=("openai_compatible",),
        destination="api-inference.modelscope.cn",
        profile_template=_profile_template(
            "modelscope-screening",
            "ModelScope screening",
            provider="openai_compatible",
            litellm_provider="openai",
            base_url="https://api-inference.modelscope.cn/v1",
            capabilities=("text",),
        ),
    ),
    _definition(
        "groq_asr",
        "Groq ASR",
        "P0",
        ("asr",),
        ("asr",),
        account_url="https://console.groq.com/",
        credential_url="https://console.groq.com/keys",
        documentation_url="https://console.groq.com/docs/speech-to-text",
        free_tier_note="Developer limits and available models change; verify the live console.",
        data_boundary="Audio goes to api.groq.com; OCR/PPT terms are bounded prompt context only.",
        runtime_integration="preset_ready",
        match_providers=("groq_asr",),
        destination="api.groq.com",
        profile_template=_profile_template(
            "groq-asr-screening",
            "Groq ASR screening",
            provider="groq_asr",
            litellm_provider="openai",
            base_url="https://api.groq.com/openai/v1",
            capabilities=("asr",),
            model="whisper-large-v3-turbo",
            provider_options={"asr_timestamp_granularity": "word"},
        ),
    ),
    _definition(
        "openrouter",
        "OpenRouter",
        "P1",
        ("text", "vision"),
        ("text_llm", "semantic_frame", "temporal_sequence"),
        account_url="https://openrouter.ai/",
        credential_url="https://openrouter.ai/settings/keys",
        documentation_url="https://openrouter.ai/docs/quickstart",
        free_tier_note="Free labels, quotas, models, and data policies must be checked per deployment.",
        data_boundary="OpenRouter may relay content to downstream providers; consent must pin the final deployment set.",
        runtime_integration="preset_ready",
        match_providers=("openrouter",),
        destination="openrouter.ai",
        profile_template=_profile_template(
            "openrouter-screening",
            "OpenRouter screening",
            provider="openrouter",
            litellm_provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            capabilities=("text", "vision"),
        ),
    ),
    _definition(
        "nvidia_nim",
        "NVIDIA NIM / Build",
        "P1",
        ("text", "vision"),
        ("text_llm", "semantic_frame"),
        account_url="https://build.nvidia.com/",
        credential_url="https://build.nvidia.com/settings/api-keys",
        documentation_url="https://docs.api.nvidia.com/nim/reference/llm-apis",
        free_tier_note="Trial capacity and available models change; verify NVIDIA Build.",
        data_boundary="Selected text or images go to integrate.api.nvidia.com; only manifest-listed files may be sent.",
        runtime_integration="preset_ready",
        match_providers=("nvidia_nim",),
        destination="integrate.api.nvidia.com",
        profile_template=_profile_template(
            "nvidia-nim-screening",
            "NVIDIA NIM screening",
            provider="nvidia_nim",
            litellm_provider="nvidia_nim",
            base_url="https://integrate.api.nvidia.com/v1",
            capabilities=("text", "vision"),
        ),
    ),
    _definition(
        "github_models",
        "GitHub Models",
        "P1",
        ("text", "vision"),
        ("text_llm", "semantic_frame"),
        account_url="https://github.com/marketplace/models",
        credential_url="https://github.com/settings/tokens",
        documentation_url="https://docs.github.com/en/rest/models/inference",
        free_tier_note="Useful for prototypes; access tiers, quotas, and models are policy-dependent.",
        data_boundary="Requests go to models.github.ai; implement and test the fixed-header adapter before routing.",
        runtime_integration="thin_adapter_required",
        match_providers=(),
        destination="models.github.ai",
        profile_template=None,
    ),
)


def _bundle_profile(
    profile_id: str,
    name: str,
    *,
    provider: str,
    litellm_provider: str,
    base_url: str,
    capabilities: tuple[str, ...],
    model: str,
    recommended_tasks: tuple[str, ...],
    protocol: str,
    note: str = "",
    catalog_status: str = "verified_visible",
    install_enabled: bool = True,
    provider_options: dict[str, Any] | None = None,
    replaces_models: tuple[str, ...] = (),
) -> dict[str, Any]:
    row = _profile_template(
        profile_id,
        name,
        provider=provider,
        litellm_provider=litellm_provider,
        base_url=base_url,
        capabilities=capabilities,
        model=model,
    )
    row.update(
        {
            "recommended_tasks": list(recommended_tasks),
            "protocol": protocol,
            "note": note,
            "catalog_status": catalog_status,
            "install_enabled": bool(install_enabled),
            "provider_options": dict(provider_options or {}),
            "replaces_models": list(replaces_models),
        }
    )
    return row


ONBOARDING_SCHEMA = "video_knowledge_pipeline.model_provider_onboarding.v2"
_PROFILE_BUNDLES: dict[str, tuple[dict[str, Any], ...]] = {
    "ark_coding_plan": (
        _bundle_profile(
            "ark-doubao-seed-2-0-pro",
            "Ark Doubao Seed 2.0 Pro",
            provider="volcengine_coding_plan",
            litellm_provider="openai",
            base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
            capabilities=("text",),
            model="doubao-seed-2.0-pro",
            recommended_tasks=("text_llm", "summary_rewrite", "transcript_correction"),
            protocol="chat_completions",
            note="Available when the configured Coding Plan account accepts this exact model; normal consent and provider responses determine use.",
            catalog_status="coding_plan_alias_verified_by_execution",
        ),
        _bundle_profile(
            "ark-doubao-seed-2-0-lite",
            "Ark Doubao Seed 2.0 Lite",
            provider="volcengine_coding_plan",
            litellm_provider="openai",
            base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
            capabilities=("text",),
            model="doubao-seed-2.0-lite",
            recommended_tasks=("transcript_correction",),
            protocol="chat_completions",
            note="Available when the configured Coding Plan account accepts this exact model; normal consent and provider responses determine use.",
            catalog_status="coding_plan_alias_verified_by_execution",
        ),        _bundle_profile(
            "ark-deepseek-v4-pro",
            "Ark DeepSeek V4 Pro",
            provider="volcengine_coding_plan",
            litellm_provider="openai",
            base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
            capabilities=("text",),
            model="deepseek-v4-pro",
            recommended_tasks=("text_llm", "transcript_correction"),
            protocol="chat_completions",
            note="Coding Plan Model Name verified by earlier consented executions; do not substitute the dated online-inference ID.",
            catalog_status="coding_plan_alias_verified_by_execution",
            replaces_models=("deepseek-v4-pro-260425",),
        ),
        _bundle_profile(
            "ark-deepseek-v4-flash",
            "Ark DeepSeek V4 Flash",
            provider="volcengine_coding_plan",
            litellm_provider="openai",
            base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
            capabilities=("text",),
            model="deepseek-v4-flash",
            recommended_tasks=("transcript_correction",),
            protocol="chat_completions",
            provider_options={"thinking_mode": "disabled"},
            note="Coding Plan Model Name verified by earlier consented executions; never an automatic fallback.",
            catalog_status="coding_plan_alias_verified_by_execution",
            replaces_models=("deepseek-v4-flash-260425",),
        ),
        _bundle_profile(
            "ark-minimax-m3",
            "Ark MiniMax M3",
            provider="volcengine_coding_plan",
            litellm_provider="openai",
            base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
            capabilities=("text",),
            model="minimax-m3",
            recommended_tasks=("summary_rewrite",),
            protocol="chat_completions",
            provider_options={"thinking_mode": "disabled"},
            note="Not returned by this account's 2026-07-16 /models catalog; kept disabled until the account exposes an exact ID.",
            catalog_status="not_visible_in_account_catalog",
            install_enabled=False,
        ),
        _bundle_profile(
            "ark-minimax-m2-7",
            "Ark MiniMax M2.7",
            provider="volcengine_coding_plan",
            litellm_provider="openai",
            base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
            capabilities=("text",),
            model="minimax-m2.7",
            recommended_tasks=("summary_rewrite",),
            protocol="chat_completions",
            note="Not returned by this account's 2026-07-16 /models catalog; kept disabled until the account exposes an exact ID.",
            catalog_status="not_visible_in_account_catalog",
            install_enabled=False,
        ),
        _bundle_profile(
            "ark-glm-latest",
            "Ark GLM 5.2",
            provider="volcengine_coding_plan",
            litellm_provider="openai",
            base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
            capabilities=("text",),
            model="glm-5.2",
            recommended_tasks=("text_llm", "transcript_correction"),
            protocol="chat_completions",
            note="Coding Plan Model Name verified by earlier consented executions; the generic /models catalog is not authoritative for this alias.",
            catalog_status="coding_plan_alias_verified_by_execution",
            replaces_models=("glm-5-2-260617", "glm-latest"),
        ),
        _bundle_profile(
            "ark-kimi-k2-7-code",
            "Ark Kimi K2.7 Code",
            provider="volcengine_coding_plan",
            litellm_provider="openai",
            base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
            capabilities=("text",),
            model="kimi-k2.7-code",
            recommended_tasks=("text_llm", "summary_rewrite"),
            protocol="chat_completions",
            note="Coding Plan alias verified by an earlier consented Stage B execution; the generic /models catalog may omit it.",
            catalog_status="coding_plan_alias_verified_by_execution",
        ),
        _bundle_profile(
            "ark-kimi-k2-6",
            "Ark Kimi K2.6",
            provider="volcengine_coding_plan",
            litellm_provider="openai",
            base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
            capabilities=("text",),
            model="kimi-k2.6",
            recommended_tasks=("text_llm", "summary_rewrite"),
            protocol="chat_completions",
            note="Coding Plan alias verified by an earlier consented Stage B execution; the generic /models catalog may omit it.",
            catalog_status="coding_plan_alias_verified_by_execution",
            replaces_models=("kimi-k2-thinking-251104",),
        ),
        _bundle_profile(
            "ark-kimi-k3",
            "Ark Kimi K3",
            provider="volcengine_coding_plan",
            litellm_provider="openai",
            base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
            capabilities=("text",),
            model="kimi-k3",
            recommended_tasks=("text_llm", "summary_rewrite"),
            protocol="chat_completions",
            note=(
                "Coding Plan alias reported available by the operator on 2026-07-17. "
                "It is not exposed by the generic Ark /models catalog, so keep it out "
                "of production defaults until a fresh consented capability gate passes."
            ),
            catalog_status="coding_plan_alias_pending_revalidation",
        ),
    ),
    "ark_model_api": (
        _bundle_profile(
            "ark-api-deepseek-v4-pro",
            "Ark API DeepSeek V4 Pro",
            provider="volcengine_ark",
            litellm_provider="volcengine",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            capabilities=("text",),
            model="deepseek-v4-pro-260425",
            recommended_tasks=("text_llm", "transcript_correction"),
            protocol="chat_completions",
            note="标准方舟模型 API；按在线推理计费，不使用 Coding Plan 套餐。",
        ),
        _bundle_profile(
            "ark-api-deepseek-v4-flash",
            "Ark API DeepSeek V4 Flash",
            provider="volcengine_ark",
            litellm_provider="volcengine",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            capabilities=("text",),
            model="deepseek-v4-flash-260425",
            recommended_tasks=("summary_rewrite", "transcript_correction"),
            protocol="chat_completions",
            note="标准方舟模型 API；按在线推理计费，不使用 Coding Plan 套餐。",
        ),
        _bundle_profile(
            "ark-api-glm-5-2",
            "Ark API GLM 5.2",
            provider="volcengine_ark",
            litellm_provider="volcengine",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            capabilities=("text",),
            model="glm-5-2-260617",
            recommended_tasks=("text_llm", "summary_rewrite", "transcript_correction"),
            protocol="chat_completions",
            note="标准方舟模型 API；按在线推理计费，不使用 Coding Plan 套餐。",
        ),
    ),
    "google_gemini": (
        _bundle_profile(
            "google-gemini-3-6-flash",
            "Google Gemini 3.6 Flash",
            provider="gemini",
            litellm_provider="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            capabilities=("text", "vision"),
            model=GEMINI_DEFAULT_MODEL,
            recommended_tasks=(
                "text_llm", "summary_rewrite", "transcript_correction",
                "semantic_frame", "temporal_sequence", "video_segment", "document_visual",
            ),
            protocol="chat_completions",
            note="Production-quality multimodal, temporal, summary, and correction route; omit deprecated sampling fields.",
        ),
        _bundle_profile(
            "google-gemini-3-5-flash-lite",
            "Google Gemini 3.5 Flash-Lite",
            provider="gemini",
            litellm_provider="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            capabilities=("text", "vision"),
            model=GEMINI_THROUGHPUT_MODEL,
            recommended_tasks=("text_llm", "document_visual"),
            protocol="chat_completions",
            note="Optional low-cost high-throughput parsing and structured extraction route; not the default for complex temporal reasoning.",
        ),
    ),
    "siliconflow": (
        _bundle_profile(
            "siliconflow-telespeech-asr",
            "SiliconFlow TeleSpeechASR free candidate",
            provider="siliconflow",
            litellm_provider="openai",
            base_url="https://api.siliconflow.cn/v1",
            capabilities=("asr",),
            model="TeleAI/TeleSpeechASR",
            recommended_tasks=("asr",),
            protocol="audio_transcriptions",
            note="Candidate-only OpenAI-compatible transcription route; audio upload still requires an exact consent manifest and never replaces the local transcript automatically.",
        ),
        _bundle_profile(
            "siliconflow-qwen3-5-4b",
            "SiliconFlow Qwen3.5 4B free text",
            provider="siliconflow",
            litellm_provider="openai",
            base_url="https://api.siliconflow.cn/v1",
            capabilities=("text",),
            model="Qwen/Qwen3.5-4B",
            recommended_tasks=("text_llm", "summary_rewrite"),
            protocol="chat_completions",
        ),
        _bundle_profile(
            "siliconflow-glm-4-1v-9b-thinking",
            "SiliconFlow GLM-4.5V vision",
            provider="siliconflow",
            litellm_provider="openai",
            base_url="https://api.siliconflow.cn/v1",
            capabilities=("vision",),
            model="zai-org/GLM-4.5V",
            recommended_tasks=("semantic_frame",),
            protocol="chat_completions",
            note="Replaces the no-longer-visible THUDM/GLM-4.1V-9B-Thinking preset.",
            replaces_models=("THUDM/GLM-4.1V-9B-Thinking",),
        ),
        _bundle_profile(
            "siliconflow-paddleocr-vl-1-5",
            "SiliconFlow PaddleOCR-VL-1.5 document visual",
            provider="siliconflow",
            litellm_provider="openai",
            base_url="https://api.siliconflow.cn/v1",
            capabilities=("vision",),
            model="PaddlePaddle/PaddleOCR-VL-1.5",
            recommended_tasks=("document_visual",),
            protocol="document_visual_chat",
            note="Vision candidate evidence only; never silently expose it as the standard /v1/ocr route.",
        ),
        _bundle_profile(
            "siliconflow-deepseek-v4-pro",
            "SiliconFlow DeepSeek V4 Pro",
            provider="siliconflow",
            litellm_provider="openai",
            base_url="https://api.siliconflow.cn/v1",
            capabilities=("text",),
            model="deepseek-ai/DeepSeek-V4-Pro",
            recommended_tasks=("text_llm", "transcript_correction"),
            protocol="chat_completions",
            note="Same advertised model family/version as the Ark DeepSeek V4 Pro candidate; serving stacks may still differ.",
        ),
        _bundle_profile(
            "siliconflow-deepseek-v4-flash",
            "SiliconFlow DeepSeek V4 Flash",
            provider="siliconflow",
            litellm_provider="openai",
            base_url="https://api.siliconflow.cn/v1",
            capabilities=("text",),
            model="deepseek-ai/DeepSeek-V4-Flash",
            recommended_tasks=("text_llm", "transcript_correction"),
            protocol="chat_completions",

            note="SiliconFlow documents optional reasoning_effort=high|max; parity presets leave it unset so both providers receive the same common controls.",
        ),
        _bundle_profile(
            "siliconflow-glm-5-2",
            "SiliconFlow GLM-5.2",
            provider="siliconflow",
            litellm_provider="openai",
            base_url="https://api.siliconflow.cn/v1",
            capabilities=("text",),
            model="zai-org/GLM-5.2",
            recommended_tasks=("text_llm", "summary_rewrite", "transcript_correction"),
            protocol="chat_completions",
            note="Same advertised model family/version as the Ark GLM-5.2 candidate; serving stacks may still differ.",
        ),
        _bundle_profile(
            "siliconflow-kimi-k2-7-code",
            "SiliconFlow Kimi K2.7 Code",
            provider="siliconflow",
            litellm_provider="openai",
            base_url="https://api.siliconflow.cn/v1",
            capabilities=("text",),
            model="moonshotai/Kimi-K2.7-Code",
            recommended_tasks=("text_llm", "summary_rewrite"),
            protocol="chat_completions",
            note="Parity candidate for the callable Ark Coding Plan kimi-k2.7-code alias.",
        ),
        _bundle_profile(
            "siliconflow-kimi-k2-6",
            "SiliconFlow Kimi K2.6",
            provider="siliconflow",
            litellm_provider="openai",
            base_url="https://api.siliconflow.cn/v1",
            capabilities=("text",),
            model="Pro/moonshotai/Kimi-K2.6",
            recommended_tasks=("text_llm", "summary_rewrite"),
            protocol="chat_completions",
            note="Parity candidate for the callable Ark Coding Plan kimi-k2.6 alias.",
        ),
    ),
    "modelscope": (
        _bundle_profile(
            "modelscope-glm-5-2",
            "ModelScope GLM-5.2",
            provider="openai_compatible",
            litellm_provider="openai",
            base_url="https://api-inference.modelscope.cn/v1",
            capabilities=("text",),
            model="ZhipuAI/GLM-5.2",
            recommended_tasks=("text_llm", "summary_rewrite", "transcript_correction"),
            protocol="chat_completions",
        ),
        _bundle_profile(
            "modelscope-deepseek-v4-pro",
            "ModelScope DeepSeek V4 Pro",
            provider="openai_compatible",
            litellm_provider="openai",
            base_url="https://api-inference.modelscope.cn/v1",
            capabilities=("text",),
            model="deepseek-ai/DeepSeek-V4-Pro",
            recommended_tasks=("text_llm", "transcript_correction"),
            protocol="chat_completions",
        ),
    ),
    "groq": (
        _bundle_profile(
            "groq-qwen3-6-27b",
            "Groq Qwen3.6 27B",
            provider="groq",
            litellm_provider="groq",
            base_url="https://api.groq.com/openai/v1",
            capabilities=("text", "vision"),
            model="qwen/qwen3.6-27b",
            provider_options={
                "reasoning_effort": "none",
                "reasoning_format": "hidden",
                "strip_reasoning_tags": True,
                "strip_json_fences": True,
            },
            recommended_tasks=(
                "text_llm", "summary_rewrite", "transcript_correction",
                "semantic_frame", "document_visual",
            ),
            protocol="chat_completions",
            note="The provider documents a three-image maximum, so this is not an eight-frame temporal preset.",
        ),
        _bundle_profile(
            "groq-whisper-large-v3-turbo",
            "Groq Whisper Large V3 Turbo",
            provider="groq_asr",
            litellm_provider="openai",
            base_url="https://api.groq.com/openai/v1",
            capabilities=("asr",),
            model="whisper-large-v3-turbo",
            provider_options={"asr_timestamp_granularity": "word"},
            recommended_tasks=("asr",),
            protocol="audio_transcriptions",
        ),
    ),
    "mistral": (
        _bundle_profile(
            "mistral-voxtral-mini-2602",
            "Mistral Voxtral Mini Transcribe 2",
            provider="mistral_asr",
            litellm_provider="mistral",
            base_url="https://api.mistral.ai/v1",
            capabilities=("asr",),
            model="voxtral-mini-2602",
            recommended_tasks=("asr",),
            protocol="audio_transcriptions",
        ),
        _bundle_profile(
            "mistral-ocr-4-0",
            "Mistral OCR 4",
            provider="mistral",
            litellm_provider="mistral",
            base_url="https://api.mistral.ai/v1",
            capabilities=("ocr",),
            model="mistral-ocr-4-0",
            recommended_tasks=("ocr",),
            protocol="ocr",
        ),
    ),
}


def _static_definition(
    provider_id: str,
    label: str,
    *,
    capabilities: tuple[str, ...],
    recommended_tasks: tuple[str, ...],
    account_url: str,
    credential_url: str,
    documentation_url: str,
    free_tier_note: str,
    data_boundary: str,
    match_providers: tuple[str, ...],
    destination: str,
) -> dict[str, Any]:
    templates = [copy.deepcopy(row) for row in _PROFILE_BUNDLES[provider_id]]
    return {
        "id": provider_id,
        "label": label,
        "priority": "P0",
        "capabilities": list(capabilities),
        "recommended_tasks": list(recommended_tasks),
        "account_url": account_url,
        "credential_url": credential_url,
        "documentation_url": documentation_url,
        "free_tier_note": free_tier_note,
        "data_boundary": data_boundary,
        "runtime_integration": "preset_ready",
        "match_providers": list(match_providers),
        "destination": destination,
        "profile_templates": templates,
        "profile_template": copy.deepcopy(templates[0]),
        "key_once_ready": True,
    }


for _definition_row in FREE_SCREENING_PROVIDERS:
    _definition_id = str(_definition_row.get("id") or "")
    if _definition_id == "groq_asr":
        _definition_row.update(
            {
                "id": "groq",
                "label": "Groq text / vision / ASR",
                "capabilities": ["text", "vision", "asr"],
                "recommended_tasks": [
                    "text_llm", "summary_rewrite", "transcript_correction",
                    "semantic_frame", "document_visual", "asr",
                ],
                "documentation_url": "https://console.groq.com/docs/models",
                "data_boundary": "Selected text, up to three images, or audio goes to api.groq.com; temporal eight-frame routing is not preassigned.",
                "match_providers": ["groq", "groq_asr"],
            }
        )
        _definition_id = "groq"
    templates = [copy.deepcopy(row) for row in _PROFILE_BUNDLES.get(_definition_id, ())]
    _definition_row["profile_templates"] = templates
    _definition_row["profile_template"] = copy.deepcopy(templates[0]) if templates else None
    _definition_row["key_once_ready"] = bool(templates)
    if not templates and _definition_row.get("runtime_integration") != "thin_adapter_required":
        _definition_row["runtime_integration"] = "model_selection_required"

FREE_SCREENING_PROVIDERS += (
    _static_definition(
        "ark_coding_plan",
        "Ark Coding Plan multi-model pool",
        capabilities=("text",),
        recommended_tasks=("text_llm", "summary_rewrite", "transcript_correction"),
        account_url="https://www.volcengine.com/activity/codingplan",
        credential_url="https://console.volcengine.com/ark/region:ark+cn-beijing/apikey",
        documentation_url="https://www.volcengine.com/docs/82379/1925114",
        free_tier_note="Coding Plan is subscription-backed; verify the current plan and model availability in the console.",
        data_boundary="Selected text goes to ark.cn-beijing.volces.com; all saved profiles are text-only and saving does not authorize egress.",
        match_providers=("volcengine_coding_plan",),
        destination="ark.cn-beijing.volces.com",
    ),
    _static_definition(
        "ark_model_api",
        "Ark standard model API",
        capabilities=("text",),
        recommended_tasks=("text_llm", "summary_rewrite", "transcript_correction"),
        account_url="https://console.volcengine.com/ark/region:ark+cn-beijing/overview",
        credential_url="https://console.volcengine.com/ark/region:ark+cn-beijing/apikey",
        documentation_url="https://www.volcengine.com/docs/82379/1795150",
        free_tier_note="Standard Ark online inference is API-billed and does not consume Coding Plan quota.",
        data_boundary="Selected text goes to ark.cn-beijing.volces.com through the standard /api/v3 model API; saving profiles does not authorize egress.",
        match_providers=("volcengine_ark",),
        destination="ark.cn-beijing.volces.com",
    ),
    _static_definition(
        "google_gemini",
        "Google Gemini API",
        capabilities=("text", "vision"),
        recommended_tasks=(
            "text_llm", "summary_rewrite", "transcript_correction",
            "semantic_frame", "temporal_sequence", "video_segment", "document_visual",
        ),
        account_url="https://aistudio.google.com/",
        credential_url="https://aistudio.google.com/apikey",
        documentation_url="https://ai.google.dev/gemini-api/docs/latest-model",
        free_tier_note="Free-tier availability and regional eligibility must be checked in AI Studio.",
        data_boundary="Selected content goes to generativelanguage.googleapis.com; use a newly issued key if an older key was exposed.",
        match_providers=("gemini",),
        destination="generativelanguage.googleapis.com",
    ),
    _static_definition(
        "mistral",
        "Mistral Voxtral / OCR",
        capabilities=("asr", "ocr"),
        recommended_tasks=("asr", "ocr"),
        account_url="https://console.mistral.ai/",
        credential_url="https://console.mistral.ai/api-keys",
        documentation_url="https://docs.mistral.ai/models/overview",
        free_tier_note="Trial credit eligibility must be checked in Studio; OCR is priced per page when not covered by credit.",
        data_boundary="Selected audio, images, or documents go to api.mistral.ai; OCR calls remain separately countable.",
        match_providers=("mistral_asr", "mistral"),
        destination="api.mistral.ai",
    ),
)


def _prefill_contract(
    contract_id: str,
    *,
    documentation_urls: tuple[str, ...],
    expected_base_urls: tuple[str, ...],
    expected_providers: tuple[str, ...],
    expected_protocols: tuple[str, ...],
    forbidden_provider_options: tuple[str, ...] = (),
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "schema": "video_knowledge_pipeline.provider_prefill_contract.v1",
        "contract_id": contract_id,
        "last_verified_at": "2026-07-18",
        "field_authority": "official_provider_documentation",
        "model_id_authority": "reviewed_account_catalog_or_execution_evidence",
        "documentation_urls": list(documentation_urls),
        "expected_base_urls": list(expected_base_urls),
        "expected_providers": list(expected_providers),
        "expected_protocols": list(expected_protocols),
        "forbidden_provider_options": list(forbidden_provider_options),
        "automatic_catalog_updates": False,
        "saving_authorizes_egress": False,
        "implementation_reuse": {
            "protocol_gateway": "LiteLLM",
            "provider_registry_pattern": "Cline @cline/llms",
            "model_discovery_pattern": "Open WebUI",
        },
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return contract


PROVIDER_PREFILL_CONTRACTS: dict[str, dict[str, Any]] = {
    "siliconflow": _prefill_contract(
        "siliconflow-openai-compatible-2026-08-11",
        documentation_urls=(
            "https://docs.siliconflow.cn/cn/userguide/quickstart",
            "https://docs.siliconflow.cn/en/api-reference/chat-completions/chat-completions",
            "https://docs.siliconflow.cn/cn/api-reference/audio/create-audio-transcriptions",
        ),
        expected_base_urls=("https://api.siliconflow.cn/v1",),
        expected_providers=("siliconflow",),
        expected_protocols=("chat_completions", "document_visual_chat", "audio_transcriptions"),
        forbidden_provider_options=("thinking_mode",),
    ),
    "modelscope": _prefill_contract(
        "modelscope-api-inference-2026-07-18",
        documentation_urls=(
            "https://modelscope.cn/docs/model-service/API-Inference/intro",
        ),
        expected_base_urls=("https://api-inference.modelscope.cn/v1",),
        expected_providers=("openai_compatible",),
        expected_protocols=("chat_completions",),
    ),
    "groq": _prefill_contract(
        "groq-openai-compatible-2026-07-18",
        documentation_urls=(
            "https://console.groq.com/docs/models",
            "https://console.groq.com/docs/speech-to-text",
        ),
        expected_base_urls=("https://api.groq.com/openai/v1",),
        expected_providers=("groq", "groq_asr"),
        expected_protocols=("chat_completions", "audio_transcriptions"),
    ),
    "ark_coding_plan": _prefill_contract(
        "volcengine-coding-plan-openclaw-2026-07-18",
        documentation_urls=(
            "https://developer.volcengine.com/articles/7615528054736945158",
            "https://www.volcengine.com/docs/82379/1925114",
        ),
        expected_base_urls=("https://ark.cn-beijing.volces.com/api/coding/v3",),
        expected_providers=("volcengine_coding_plan",),
        expected_protocols=("chat_completions",),
        forbidden_provider_options=("enable_thinking",),
    ),
    "ark_model_api": _prefill_contract(
        "volcengine-ark-standard-api-2026-07-18",
        documentation_urls=("https://www.volcengine.com/docs/82379/1795150",),
        expected_base_urls=("https://ark.cn-beijing.volces.com/api/v3",),
        expected_providers=("volcengine_ark",),
        expected_protocols=("chat_completions",),
        forbidden_provider_options=("enable_thinking",),
    ),
    "google_gemini": _prefill_contract(
        "google-gemini-api-2026-07-22",
        documentation_urls=(
            "https://ai.google.dev/gemini-api/docs/latest-model",
            "https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash",
            "https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite",
        ),
        expected_base_urls=("https://generativelanguage.googleapis.com/v1beta",),
        expected_providers=("gemini",),
        expected_protocols=("chat_completions",),
        forbidden_provider_options=(
            "temperature",
            "top_p",
            "top_k",
            "thinking_budget",
            "candidate_count",
        ),
    ),
    "mistral": _prefill_contract(
        "mistral-voxtral-ocr-2026-07-18",
        documentation_urls=("https://docs.mistral.ai/models/overview",),
        expected_base_urls=("https://api.mistral.ai/v1",),
        expected_providers=("mistral", "mistral_asr"),
        expected_protocols=("audio_transcriptions", "ocr"),
    ),
}

for _provider_definition in FREE_SCREENING_PROVIDERS:
    _provider_id = str(_provider_definition.get("id") or "")
    _provider_definition["prefill_contract"] = copy.deepcopy(
        PROVIDER_PREFILL_CONTRACTS.get(_provider_id)
    )


def validate_provider_onboarding_prefills() -> dict[str, Any]:
    """Fail closed when a Key-only bundle drifts from its reviewed field contract."""

    checked_templates = 0
    errors: list[str] = []
    for definition in FREE_SCREENING_PROVIDERS:
        templates = [
            dict(row)
            for row in definition.get("profile_templates") or []
            if isinstance(row, dict)
        ]
        if not templates:
            continue
        provider_id = str(definition.get("id") or "")
        contract = definition.get("prefill_contract")
        if not isinstance(contract, dict):
            errors.append(f"{provider_id}: missing prefill contract")
            continue
        hash_payload = dict(contract)
        expected_hash = str(hash_payload.pop("contract_sha256", ""))
        actual_hash = canonical_json_sha256(hash_payload)
        if expected_hash != actual_hash:
            errors.append(f"{provider_id}: prefill contract hash mismatch")
        for url in contract.get("documentation_urls") or []:
            parsed = urlsplit(str(url))
            if parsed.scheme != "https" or not parsed.hostname:
                errors.append(f"{provider_id}: invalid documentation URL")
        allowed_base_urls = set(contract.get("expected_base_urls") or [])
        allowed_providers = set(contract.get("expected_providers") or [])
        allowed_protocols = set(contract.get("expected_protocols") or [])
        forbidden_options = set(contract.get("forbidden_provider_options") or [])
        for template in templates:
            checked_templates += 1
            template_id = str(template.get("id") or "")
            if str(template.get("base_url") or "") not in allowed_base_urls:
                errors.append(f"{provider_id}/{template_id}: unreviewed base_url")
            if str(template.get("provider") or "") not in allowed_providers:
                errors.append(f"{provider_id}/{template_id}: unreviewed provider")
            if str(template.get("protocol") or "") not in allowed_protocols:
                errors.append(f"{provider_id}/{template_id}: unreviewed protocol")
            option_keys = set((template.get("provider_options") or {}).keys())
            if option_keys.intersection(forbidden_options):
                errors.append(f"{provider_id}/{template_id}: forbidden provider option")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "ok": True,
        "contract_count": len(PROVIDER_PREFILL_CONTRACTS),
        "template_count": checked_templates,
        "automatic_catalog_updates": False,
    }


def provider_onboarding_definition(provider_id: str) -> dict[str, Any]:
    clean_id = str(provider_id or "").strip().lower()
    # Preserve older UI links without retaining their former filtering semantics.
    if clean_id == "ark_no_doubao":
        clean_id = "ark_coding_plan"
    definition = next(
        (row for row in FREE_SCREENING_PROVIDERS if str(row.get("id") or "") == clean_id),
        None,
    )
    if definition is None:
        raise ValueError(f"unsupported onboarding provider: {provider_id!r}")
    return copy.deepcopy(definition)


def key_once_onboarding_provider_ids() -> tuple[str, ...]:
    """Return providers whose exact profile templates can be prepared offline."""

    return tuple(
        str(row.get("id") or "")
        for row in FREE_SCREENING_PROVIDERS
        if row.get("profile_templates")
    )




def _base_free_screening_onboarding_status(
    profiles: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    route_status: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    profile_rows = [dict(row) for row in profiles if isinstance(row, dict)]
    route_rows = [dict(row) for row in route_status if isinstance(row, dict)]
    return {
        "schema": ONBOARDING_SCHEMA,
        "entries": [
            _entry_status(definition, profile_rows, route_rows)
            for definition in FREE_SCREENING_PROVIDERS
        ],
        "network_calls": False,
        "secrets_exposed": False,
        "saving_authorizes_egress": False,
        "checked_state": "local_configuration_only",
    }


def _entry_status(
    definition: dict[str, Any],
    profiles: list[dict[str, Any]],
    route_status: list[dict[str, Any]],
) -> dict[str, Any]:
    providers = set(definition.get("match_providers") or [])
    destination = str(definition.get("destination") or "").lower()
    matches = [
        row
        for row in profiles
        if str(row.get("provider") or "") in providers
        and (
            not destination
            or str(urlsplit(str(row.get("base_url") or "")).hostname or "").lower()
            == destination
        )
    ]
    profile_ids = {str(row.get("id") or "") for row in matches}
    routes = [
        row
        for row in route_status
        if profile_ids.intersection(str(item) for item in row.get("deployments") or [])
    ]
    profile_saved = bool(matches)
    model_selected = any(bool(str(row.get("model") or "").strip()) for row in matches)
    credential_configured = any(
        bool(row.get("api_key_configured"))
        or str(row.get("credential_status") or "") in {"ready", "not_required"}
        for row in matches
    )
    route_configured = bool(routes)
    allowlist_states = sorted(
        {str(row.get("allowlist_status") or "unknown") for row in routes}
    )
    allowlist_approved = bool(routes) and all(
        state == "approved" for state in allowlist_states
    )
    runtime_integration = str(definition.get("runtime_integration") or "")

    blockers: list[str] = []
    if runtime_integration == "thin_adapter_required":
        blockers.append("implement_runtime_adapter")
    if not profile_saved:
        blockers.append("create_profile")
    if not model_selected:
        blockers.append("select_model")
    if not credential_configured:
        blockers.append("add_credential")
    if not route_configured:
        blockers.append("configure_route")
    elif not allowlist_approved:
        blockers.append("approve_destination")
    else:
        blockers.append("create_consent")

    if runtime_integration == "thin_adapter_required":
        status = "adapter_required"
    elif not profile_saved:
        status = "not_started"
    elif not model_selected or not credential_configured:
        status = "profile_draft"
    elif not route_configured:
        status = "route_pending"
    elif not allowlist_approved:
        status = "allowlist_pending"
    else:
        status = "ready_for_consent"

    result = copy.deepcopy(definition)
    result.update(
        {
            "status": status,
            "matching_profile_ids": sorted(profile_ids),
            "profile_saved": profile_saved,
            "model_selected": model_selected,
            "credential_configured": credential_configured,
            "route_configured": route_configured,
            "allowlist_statuses": allowlist_states,
            "consent_status": "not_checked",
            "blockers": blockers,
        }
    )
    return result


def free_screening_onboarding_status(
    profiles: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    route_status: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    result = _base_free_screening_onboarding_status(profiles, route_status)
    profile_map = {
        str(row.get("id") or ""): dict(row)
        for row in profiles
        if isinstance(row, dict)
    }
    for entry in result["entries"]:
        templates = [dict(row) for row in entry.get("profile_templates") or []]
        expected_ids = {str(row.get("id") or "") for row in templates}
        installed_ids = expected_ids.intersection(profile_map)
        missing_ids = expected_ids - installed_ids
        inspected = [profile_map[profile_id] for profile_id in sorted(installed_ids)]
        installable = bool(expected_ids)
        profile_saved = installable and not missing_ids
        model_selected = profile_saved and all(
            bool(str(row.get("model") or "").strip()) for row in inspected
        )
        credential_configured = profile_saved and all(
            bool(row.get("api_key_configured"))
            or str(row.get("credential_status") or "") in {"ready", "not_required"}
            for row in inspected
        )
        route_configured = bool(entry.get("route_configured"))
        allowlist_approved = bool(entry.get("allowlist_statuses")) and all(
            state == "approved" for state in entry["allowlist_statuses"]
        )
        blockers: list[str] = []
        if entry.get("runtime_integration") == "thin_adapter_required":
            blockers.append("implement_runtime_adapter")
        if not installable:
            blockers.append("select_model")
        elif not profile_saved:
            blockers.append("create_profile")
        if installable and not credential_configured:
            blockers.append("add_credential")
        if installable and not route_configured:
            blockers.append("configure_route")
        elif installable and not allowlist_approved:
            blockers.append("approve_destination")
        elif installable:
            blockers.append("create_consent")

        if entry.get("runtime_integration") == "thin_adapter_required":
            status = "adapter_required"
        elif not installable:
            status = "model_selection_required"
        elif not profile_saved:
            status = "not_started"
        elif not model_selected or not credential_configured:
            status = "profile_draft"
        elif not route_configured:
            status = "profiles_ready_route_pending"
        elif not allowlist_approved:
            status = "allowlist_pending"
        else:
            status = "ready_for_consent"
        entry.update(
            {
                "status": status,
                "expected_profile_ids": sorted(expected_ids),
                "installed_profile_ids": sorted(installed_ids),
                "missing_profile_ids": sorted(missing_ids),
                "profile_saved": profile_saved,
                "model_selected": model_selected,
                "credential_configured": credential_configured,
                "blockers": blockers,
            }
        )
    result["schema"] = ONBOARDING_SCHEMA
    return result
