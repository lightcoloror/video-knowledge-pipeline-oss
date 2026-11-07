from __future__ import annotations

import re
from typing import Any

from .canonical_json import canonical_json_sha256
from .model_defaults import GEMINI_DEFAULT_MODEL


CATALOG_SCHEMA = "video_knowledge_pipeline.model_provider_catalog.v1"
LITELLM_PROVIDER_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _preset(
    provider: str,
    label: str,
    *,
    litellm_provider: str,
    capabilities: tuple[str, ...],
    default_base_url: str = "",
    default_model: str = "",
    default_capabilities: tuple[str, ...] | None = None,
    api_key_optional: bool = False,
    location: str = "remote",
    auth_mode: str = "api_key_dpapi",
    allowed_provider_options: tuple[str, ...] = (),
    required_provider_options: tuple[str, ...] | None = None,
    environment_bindings: tuple[tuple[str, str, bool], ...] = (),
    usage_scope: str = "general_model_api",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "provider": provider,
        "label": label,
        "litellm_provider": litellm_provider,
        "supported_capabilities": list(capabilities),
        "default_capabilities": list(default_capabilities or capabilities[:1]),
        "default_base_url": default_base_url,
        "default_model": default_model,
        "api_key_optional": bool(api_key_optional),
        "default_location": location,
        "auth_mode": auth_mode,
        "allowed_provider_options": list(allowed_provider_options),
        "required_provider_options": list(
            allowed_provider_options
            if required_provider_options is None
            else required_provider_options
        ),
        "environment_bindings": [
            {"param": param, "env": env, "required": bool(required)}
            for param, env, required in environment_bindings
        ],
        "usage_scope": usage_scope,
        "notes": notes,
    }


# Provider-specific protocol conversion belongs to LiteLLM. This catalog only
# declares stable routing metadata plus audited API-key or external-environment
# authentication. ``litellm_native`` is the extension point for any
# other LiteLLM provider prefix without adding another core-code branch.
PROVIDER_PRESETS: tuple[dict[str, Any], ...] = (
    _preset(
        "volcengine_coding_plan",
        "火山方舟 / Coding Plan",
        litellm_provider="openai",
        capabilities=("text", "vision"),
        default_base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        default_model="deepseek-v4-pro",
        default_capabilities=("text",),
        allowed_provider_options=(
            "thinking_mode",
            "response_format",
            "max_tokens",
            "strip_reasoning_tags",
            "strip_json_fences",
        ),
        required_provider_options=(),
        usage_scope="general_model_api",
    ),
    _preset(
        "volcengine_ark",
        "火山方舟 / Ark API",
        litellm_provider="volcengine",
        capabilities=("text", "vision"),
        default_base_url="https://ark.cn-beijing.volces.com/api/v3",
        default_capabilities=("text",),
        allowed_provider_options=(
            "thinking_mode",
            "response_format",
            "max_tokens",
            "strip_reasoning_tags",
            "strip_json_fences",
        ),
        required_provider_options=(),
        notes="标准方舟模型 API；按在线推理计费，不消耗 Coding Plan 套餐额度。",
    ),
    _preset(
        "openai",
        "OpenAI（文本 / 视觉）",
        litellm_provider="openai",
        capabilities=("text", "vision"),
        default_base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        default_capabilities=("text", "vision"),
    ),
    _preset(
        "azure_openai",
        "Azure OpenAI（API Key）",
        litellm_provider="azure",
        capabilities=("text", "vision", "asr"),
        allowed_provider_options=("api_version",),
        notes="Base URL 必须填写精确 Azure resource endpoint；api_version 为非密钥参数。",
    ),
    _preset(
        "azure_openai_entra",
        "Azure OpenAI（Entra ID）",
        litellm_provider="azure",
        capabilities=("text", "vision", "asr"),
        auth_mode="external_environment",
        allowed_provider_options=("api_version",),
        environment_bindings=(
            ("tenant_id", "AZURE_TENANT_ID", True),
            ("client_id", "AZURE_CLIENT_ID", True),
            ("client_secret", "AZURE_CLIENT_SECRET", True),
        ),
        notes="Entra 凭据只从 LiteLLM 子进程环境读取，VKP 设置文件不保存其值。",
    ),
    _preset(
        "azure_ai_ocr",
        "Azure AI OCR",
        litellm_provider="azure_ai",
        capabilities=("ocr",),
        allowed_provider_options=("api_version",),
        notes="使用 LiteLLM /v1/ocr；Base URL 必须是已审查的 Azure AI endpoint。",
    ),
    _preset(
        "vertex_ai",
        "Google Vertex AI（文本 / 视觉）",
        litellm_provider="vertex_ai",
        capabilities=("text", "vision"),
        auth_mode="external_environment",
        allowed_provider_options=("vertex_project", "vertex_location"),
        environment_bindings=(("vertex_credentials", "GOOGLE_APPLICATION_CREDENTIALS", True),),
        notes="凭据文件路径从环境读取；项目和区域作为非密钥参数保存。",
    ),
    _preset(
        "vertex_ai_ocr",
        "Google Vertex AI OCR",
        litellm_provider="vertex_ai",
        capabilities=("ocr",),
        auth_mode="external_environment",
        allowed_provider_options=("vertex_project", "vertex_location"),
        environment_bindings=(("vertex_credentials", "GOOGLE_APPLICATION_CREDENTIALS", True),),
        notes="使用 LiteLLM /v1/ocr；凭据文件路径仅从环境读取。",
    ),
    _preset(
        "bedrock",
        "AWS Bedrock（访问密钥环境变量）",
        litellm_provider="bedrock",
        capabilities=("text", "vision"),
        auth_mode="external_environment",
        allowed_provider_options=("aws_region_name",),
        environment_bindings=(
            ("aws_access_key_id", "AWS_ACCESS_KEY_ID", True),
            ("aws_secret_access_key", "AWS_SECRET_ACCESS_KEY", True),
        ),
        notes="本 profile 使用显式 AWS 环境凭据；区域作为非密钥参数保存。",
    ),
    _preset(
        "anthropic",
        "Anthropic Claude",
        litellm_provider="anthropic",
        capabilities=("text", "vision"),
        default_base_url="https://api.anthropic.com",
    ),
    _preset(
        "gemini",
        "Google Gemini API",
        litellm_provider="gemini",
        capabilities=("text", "vision"),
        default_base_url="https://generativelanguage.googleapis.com/v1beta",
        default_model=GEMINI_DEFAULT_MODEL,
        default_capabilities=("text", "vision"),
    ),
    _preset(
        "deepseek",
        "DeepSeek",
        litellm_provider="deepseek",
        capabilities=("text",),
        default_base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
    ),
    _preset(
        "dashscope",
        "阿里云百炼 / DashScope Qwen",
        litellm_provider="dashscope",
        capabilities=("text", "vision"),
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    _preset(
        "dashscope_filetrans",
        "阿里云百炼 / Qwen Audio 录音文件识别",
        # This is not OpenAI audio/transcriptions. VKP invokes the pinned
        # moys-asr-workflow async filetrans adapter after consent reservation.
        litellm_provider="dashscope",
        capabilities=("asr",),
        default_base_url="https://dashscope.aliyuncs.com",
        default_model="qwen-audio-3.0-asr-flash-filetrans",
        default_capabilities=("asr",),
        allowed_provider_options=(
            "workspace_id",
            "region",
            "language",
            "speaker_diarization",
            "poll_interval_seconds",
            "poll_timeout_seconds",
        ),
        required_provider_options=(),
        usage_scope="cloud_asr_candidate",
        notes=(
            "复用固定 moys-asr-workflow 异步 filetrans CLI；支持临时 OSS 上传、"
            "任务轮询、粤语、词级时间戳和说话人分离。不得按 OpenAI ASR 协议调用。"
        ),
    ),
    _preset(
        "openrouter",
        "OpenRouter",
        litellm_provider="openrouter",
        capabilities=("text", "vision"),
        default_base_url="https://openrouter.ai/api/v1",
    ),
    _preset(
        "groq",
        "Groq（文本 / 视觉）",
        litellm_provider="groq",
        capabilities=("text", "vision"),
        default_base_url="https://api.groq.com/openai/v1",
        default_model="qwen/qwen3.6-27b",
        default_capabilities=("text", "vision"),
        allowed_provider_options=(
            "reasoning_effort",
            "reasoning_format",
            "strip_reasoning_tags",
            "strip_json_fences",
        ),
    ),
    _preset(
        "mistral_chat",
        "Mistral（文本 / 视觉）",
        litellm_provider="mistral",
        capabilities=("text", "vision"),
        default_base_url="https://api.mistral.ai/v1",
    ),
    _preset(
        "moonshot",
        "Moonshot / Kimi",
        litellm_provider="moonshot",
        capabilities=("text", "vision"),
        default_base_url="https://api.moonshot.cn/v1",
    ),
    _preset(
        "minimax",
        "MiniMax",
        litellm_provider="minimax",
        capabilities=("text",),
        default_base_url="https://api.minimax.chat/v1",
    ),
    _preset(
        "zai",
        "智谱 GLM / Z.AI",
        litellm_provider="zai",
        capabilities=("text", "vision"),
        default_base_url="https://open.bigmodel.cn/api/paas/v4",
    ),
    _preset(
        "xai",
        "xAI",
        litellm_provider="xai",
        capabilities=("text", "vision"),
        default_base_url="https://api.x.ai/v1",
    ),
    _preset(
        "together_ai",
        "Together AI",
        litellm_provider="together_ai",
        capabilities=("text", "vision"),
        default_base_url="https://api.together.xyz/v1",
    ),
    _preset(
        "fireworks_ai",
        "Fireworks AI（文本 / 视觉）",
        litellm_provider="fireworks_ai",
        capabilities=("text", "vision"),
        default_base_url="https://api.fireworks.ai/inference/v1",
    ),
    _preset(
        "deepinfra",
        "DeepInfra",
        litellm_provider="deepinfra",
        capabilities=("text", "vision"),
        default_base_url="https://api.deepinfra.com/v1/openai",
    ),
    _preset(
        "cerebras",
        "Cerebras",
        litellm_provider="cerebras",
        capabilities=("text",),
        default_base_url="https://api.cerebras.ai/v1",
    ),
    _preset(
        "sambanova",
        "SambaNova",
        litellm_provider="sambanova",
        capabilities=("text", "vision"),
        default_base_url="https://api.sambanova.ai/v1",
    ),
    _preset(
        "perplexity",
        "Perplexity",
        litellm_provider="perplexity",
        capabilities=("text",),
        default_base_url="https://api.perplexity.ai",
    ),
    _preset(
        "nvidia_nim",
        "NVIDIA NIM",
        litellm_provider="nvidia_nim",
        capabilities=("text", "vision"),
        default_base_url="https://integrate.api.nvidia.com/v1",
    ),
    _preset(
        "siliconflow",
        "SiliconFlow（OpenAI-compatible）",
        litellm_provider="openai",
        capabilities=("text", "vision", "asr"),
        default_base_url="https://api.siliconflow.cn/v1",
        allowed_provider_options=(
            "enable_thinking",
            "thinking_budget",
            "reasoning_effort",
            "response_format",
            "max_tokens",
            "stream",
            "strip_reasoning_tags",
            "strip_json_fences",
        ),
        required_provider_options=(),
    ),
    _preset(
        "openai_compatible",
        "OpenAI-compatible / 自定义供应商",
        litellm_provider="openai",
        capabilities=("text", "vision", "asr"),
    ),
    _preset(
        "litellm_native",
        "其他 LiteLLM 原生 Provider",
        litellm_provider="",
        capabilities=("text", "vision", "asr", "ocr"),
        notes="必须显式填写 LiteLLM provider prefix；目的地仍受 route、consent 与 Broker allowlist 约束。",
    ),
    _preset(
        "openai_compatible_asr",
        "OpenAI-compatible ASR",
        litellm_provider="openai",
        capabilities=("asr",),
        default_base_url="https://api.openai.com/v1",
        default_model="gpt-4o-transcribe",
    ),
    _preset(
        "groq_asr",
        "Groq ASR",
        # Groq exposes the OpenAI-compatible transcription contract, including
        # word timestamps. LiteLLM's native Groq STT transform (through 1.86.2)
        # drops timestamp_granularities, so reuse its mature OpenAI transport.
        litellm_provider="openai",
        capabilities=("asr",),
        default_base_url="https://api.groq.com/openai/v1",
        default_model="whisper-large-v3-turbo",
        allowed_provider_options=("asr_timestamp_granularity",),
        required_provider_options=(),
        notes=(
            "Provider identity remains Groq; LiteLLM OpenAI transcription transport "
            "preserves the route-locked word timestamp request."
        ),
    ),
    _preset(
        "deepgram_asr",
        "Deepgram ASR",
        litellm_provider="deepgram",
        capabilities=("asr",),
        default_base_url="https://api.deepgram.com/v1",
        default_model="nova-3",
    ),
    _preset(
        "fireworks_asr",
        "Fireworks AI ASR",
        litellm_provider="fireworks_ai",
        capabilities=("asr",),
        default_base_url="https://api.fireworks.ai/inference/v1",
    ),
    _preset(
        "mistral_asr",
        "Mistral Voxtral ASR",
        # LiteLLM 1.81.7 does not dispatch audio transcriptions for its native
        # ``mistral`` provider. Mistral's transcription endpoint implements
        # the OpenAI-compatible /v1/audio/transcriptions contract, so keep the
        # Mistral destination/provider identity while using LiteLLM's OpenAI
        # transcription transport.
        litellm_provider="openai",
        capabilities=("asr",),
        default_base_url="https://api.mistral.ai/v1",
    ),
    _preset(
        "mistral",
        "Mistral OCR",
        litellm_provider="mistral",
        capabilities=("ocr",),
        default_base_url="https://api.mistral.ai/v1",
        default_model="mistral-ocr-latest",
    ),
    _preset(
        "mistral_compatible_ocr",
        "Mistral-compatible OCR thin adapter",
        litellm_provider="mistral",
        capabilities=("ocr",),
        default_model="mistral-ocr-latest",
    ),
    _preset(
        "local_openai_compatible",
        "本地 OpenAI-compatible 文本 / 视觉",
        litellm_provider="openai",
        capabilities=("text", "vision"),
        default_base_url="http://127.0.0.1:8000/v1",
        default_model="local-model",
        default_capabilities=("text", "vision"),
        api_key_optional=True,
        location="local",
        allowed_provider_options=("reasoning_effort", "response_format", "max_tokens"),
        required_provider_options=(),
    ),
    _preset(
        "speaches_openai_compatible",
        "本地 Speaches OpenAI-compatible ASR",
        litellm_provider="openai",
        capabilities=("asr",),
        default_base_url="http://127.0.0.1:8001/v1",
        default_model="Systran/faster-whisper-large-v3",
        api_key_optional=True,
        location="local",
        notes="Speaches 服务由操作者管理；默认与本地 VLM 使用不同端口。",
    ),
    _preset(
        "local_qwen_vl",
        "本地 Qwen-VL",
        litellm_provider="openai",
        capabilities=("vision",),
        default_base_url="http://127.0.0.1:8000/v1",
        default_model="Qwen/Qwen2.5-VL-3B-Instruct",
        api_key_optional=True,
        location="local",
    ),
    _preset(
        "local_vlm",
        "本地 OpenAI-compatible VLM",
        litellm_provider="openai",
        capabilities=("vision",),
        default_base_url="http://127.0.0.1:8000/v1",
        default_model="local-vlm",
        api_key_optional=True,
        location="local",
    ),
)

PROVIDER_MAP = {str(row["provider"]): row for row in PROVIDER_PRESETS}
ALLOWED_PROVIDERS = frozenset(PROVIDER_MAP)
OCR_LITELLM_PROVIDERS = frozenset({"mistral", "azure_ai", "vertex_ai"})


def provider_preset(provider: str) -> dict[str, Any]:
    key = str(provider or "").strip().lower().replace("-", "_")
    row = PROVIDER_MAP.get(key)
    if row is None:
        raise ValueError(f"unsupported provider: {provider!r}")
    return dict(row)


def resolve_litellm_provider(provider: str, explicit: str = "") -> str:
    row = provider_preset(provider)
    expected = str(row.get("litellm_provider") or "")
    requested = str(explicit or "").strip().lower().replace("-", "_")
    if str(row["provider"]) == "litellm_native":
        if not LITELLM_PROVIDER_RE.fullmatch(requested):
            raise ValueError("litellm_native profiles require a valid litellm_provider prefix")
        return requested
    if requested and requested != expected:
        legacy_transport = {
            "mistral_asr": "mistral",
            "groq_asr": "groq",
        }.get(str(row["provider"]))
        if requested == legacy_transport:
            return expected
        raise ValueError("litellm_provider must match the selected provider preset")
    return expected


def provider_usage_scope(provider: str) -> str:
    return str(provider_preset(provider).get("usage_scope") or "general_model_api")


def providers_for_capability(capability: str, *, location: str = "") -> list[str]:
    requested_capability = str(capability or "").strip().lower()
    if requested_capability not in {"text", "vision", "asr", "ocr"}:
        raise ValueError(f"unsupported provider capability: {capability!r}")
    requested_location = str(location or "").strip().lower()
    if requested_location not in {"", "local", "remote"}:
        raise ValueError(f"unsupported provider location: {location!r}")
    return [
        str(row["provider"])
        for row in PROVIDER_PRESETS
        if requested_capability in set(row.get("supported_capabilities") or [])
        and (
            not requested_location
            or str(row.get("default_location") or "remote") == requested_location
        )
    ]


def provider_catalog_status() -> dict[str, Any]:
    payload = [dict(row) for row in PROVIDER_PRESETS]
    revision = canonical_json_sha256(payload)
    return {
        "schema": CATALOG_SCHEMA,
        "revision": revision,
        "provider_count": len(payload),
        "providers": payload,
        "extension_provider": "litellm_native",
        "secrets_in_catalog": False,
    }
