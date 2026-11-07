from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


from .model_api_settings import public_model_api_settings_status
from .model_defaults import GEMINI_DEFAULT_MODEL
from .storage import write_text_atomic
CONFIG_ENV_VAR = "VIDEO_KNOWLEDGE_PIPELINE_CONFIG"
CONFIG_SCHEMA = "video_knowledge_pipeline.config.v1"
REQUIRED_SERVICES = {
    "review_webui": {"kind": "static_file"},
    "ebook_markdown_pipeline_http": {"kind": "http"},
    "openclaw_http": {"kind": "http"},
    "mcp": {"kind": "stdio"},
}
DEFAULT_VISION_EXECUTION = {
    "provider": "gemini",
    "model": GEMINI_DEFAULT_MODEL,
    "multimodal_limit": 19,
    "temporal_limit": 3,
    "frame_count": 8,
}
DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS = 5.0
DEFAULT_LOCAL_FRAME_BUDGET = 720
DEFAULT_LOCAL_FRAME_SAMPLING_MODE = "balanced-long-video"
LOCAL_FRAME_SAMPLING_MODES = ("balanced-long-video", "dense-local", "triage-first")
DEFAULT_EBOOK_PIPELINE = {
    "execute_default": False,
    "include_routes": ["document_visual", "mixed"],
    "limit": 36,
    "timeout_seconds": 300,
    "rapidocr_device": "auto",
    "rapidocr_cuda_device_id": 0,
}
DEFAULT_PROCESSING_PROFILES = {
    "default": "quality",
    "quality": {
        "primary_asr_preset": "sensevoice",
        "secondary_asr_preset": "qwen3-asr-1.7b",
        "secondary_fallback_preset": "qwen3-asr-0.6b",
        "forced_aligner_model": "Qwen/Qwen3-ForcedAligner-0.6B",
        "chapter_mode": "semantic",
        "text_llm_auto_execute": True,
        "data_export_allowed": False,
        "llm_preflight_call_threshold": 20,
        "llm_preflight_input_char_threshold": 120000,
        "cloud_audio_triage_only": True,
        "cloud_vision_triage_only": True,
    },
}


DEFAULT_ASR_RUNTIME = {
    "provider": "funasr_sensevoice",
    "model": "iic/SenseVoiceSmall",
    "device": "cuda_preferred",
    "compute_type": "auto",
    "vad_model": "fsmn-vad",
    "punc_model": "ct-punc",
    "spk_model": "",
    "enable_vad": True,
    "enable_itn": True,
    "enable_punctuation": True,
    "enable_diarization": False,
    "merge_vad": True,
    "merge_length_s": 15,
    "audio_preprocess": {
        "enabled": True,
        "ffmpeg_normalize": True,
        "target_sample_rate": 16000,
    },
    "openai_compatible": {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "Systran/faster-whisper-large-v3",
        "timeout_seconds": 600,
    },
}
ASR_RUNTIME_PROVIDERS = ("funasr_sensevoice", "faster_whisper", "whisperx_alignment", "speaches_openai_compatible", "custom_openai_compatible")
ASR_RUNTIME_DEVICES = ("cuda_preferred", "auto", "cuda", "cpu")
VISION_PROVIDER_PRESETS = [
    {"provider": "gemini", "label": "Gemini", "default_model": GEMINI_DEFAULT_MODEL, "default_base_url": "https://generativelanguage.googleapis.com/v1beta", "api_key_env": ["GEMINI_API_KEY"]},
    {"provider": "openai", "label": "OpenAI", "default_model": "gpt-4o-mini", "default_base_url": "https://api.openai.com/v1", "api_key_env": ["OPENAI_API_KEY"]},
    {"provider": "openai_compatible", "label": "OpenAI-compatible", "default_model": "", "default_base_url": "", "api_key_env": ["LECTURE_VISION_API_KEY", "OPENAI_API_KEY"]},
    {"provider": "volcengine_coding_plan", "label": "Volcengine Coding Plan", "default_model": "ark-code-latest", "default_base_url": "https://ark.cn-beijing.volces.com/api/coding/v3", "api_key_env": ["ARK_API_KEY", "LLM_API_KEY"], "compatible_key_env": ["OPENAI_API_KEY"], "compatible_base_url_env": "OPENAI_BASE_URL"},
    {"provider": "agnes", "label": "Agnes AI", "default_model": "agnes-1.5-flash", "default_base_url": "https://api.agnes-ai.com/v1", "api_key_env": ["AGNES_API_KEY"]},
    {"provider": "local_qwen_vl", "label": "Local Qwen-VL", "default_model": "Qwen/Qwen2.5-VL-3B-Instruct", "default_base_url": "http://127.0.0.1:8000/v1", "api_key_env": ["LOCAL_QWEN_VL_API_KEY"], "api_key_optional": True},
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return project_root() / "config" / "video-knowledge-pipeline.json"


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    raw = config_path or os.environ.get(CONFIG_ENV_VAR) or default_config_path()
    return Path(raw).expanduser().resolve()


def load_project_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = resolve_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"video knowledge config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("video knowledge config must be a JSON object")
    schema = str(data.get("schema") or "")
    if schema != CONFIG_SCHEMA:
        raise ValueError(f"unsupported video knowledge config schema: {schema}")
    services = data.get("services")
    if not isinstance(services, dict):
        raise ValueError("video knowledge config missing services object")
    return data


def config_status(config_path: str | Path | None = None) -> dict[str, Any]:
    path = resolve_config_path(config_path)
    try:
        data = load_project_config(path)
    except Exception as exc:  # noqa: BLE001 - surfaced for CLI/MCP health checks.
        return {
            "ok": False,
            "schema": CONFIG_SCHEMA,
            "config_path": str(path),
            "error": str(exc),
        }
    services = data.get("services") if isinstance(data.get("services"), dict) else {}
    validation = validate_project_config(data)
    return {
        "ok": validation["ok"],
        "schema": data.get("schema"),
        "config_path": str(path),
        "services": _sanitized_services(services),
        "service_urls": _service_urls(services),
        "vision_execution": vision_execution_profile(path),
        "ebook_pipeline": ebook_pipeline_profile(path),
        "model_api_settings": model_api_settings_status(path),
        "asr_runtime": asr_runtime_profile(path),
        "processing_profiles": processing_profiles(path),
        "validation": validation,
    }


def model_api_settings_status(config_path: str | Path | None = None) -> dict[str, Any]:
    current = vision_execution_profile(config_path)
    providers: list[dict[str, Any]] = []
    for preset in VISION_PROVIDER_PRESETS:
        env_names = [str(name) for name in preset.get("api_key_env") or []]
        providers.append({
            **preset,
            "api_key_env": env_names,
            "api_key_configured": _provider_api_key_configured(preset, env_names),
            "selected": str(current.get("provider") or "") == str(preset.get("provider") or ""),
        })
    local_store = public_model_api_settings_status()
    return {
        "schema": "video_knowledge_pipeline.model_api_settings.v1",
        "config_path": str(resolve_config_path(config_path)),
        "current_profile": current,
        "providers": providers,
        "asr_runtime": asr_runtime_profile(config_path),
        "asr_service_adapters": _asr_service_adapters(),
        "local_profile_store": local_store,
        "settings_ui_url": local_store["settings_ui_url"],
        "secret_policy": "API keys may come from user environment variables, private env files, or the local Windows-DPAPI profile store; they are never written to project config/manifests/reports.",
    }


def _asr_service_adapters() -> list[dict[str, Any]]:
    return [
        {
            "provider": "funasr_sensevoice",
            "label": "Local FunASR / SenseVoice",
            "interface": "python_runner",
            "default_model": DEFAULT_ASR_RUNTIME["model"],
            "gpu_policy": "cuda_preferred",
            "reuse_source": "FunASR/SenseVoice local runner",
        },
        {
            "provider": "speaches_openai_compatible",
            "label": "Speaches OpenAI-compatible ASR",
            "interface": "openai_audio_transcriptions",
            "default_base_url": DEFAULT_ASR_RUNTIME["openai_compatible"]["base_url"],
            "default_model": DEFAULT_ASR_RUNTIME["openai_compatible"]["model"],
            "gpu_policy": "managed_by_service",
            "reuse_source": "Speaches local ASR service contract",
        },
        {
            "provider": "faster_whisper",
            "label": "faster-whisper fallback",
            "interface": "local_python_runner",
            "default_model": "large-v3",
            "gpu_policy": "cuda_preferred",
            "reuse_source": "AI-Video-Transcriber/faster-whisper parameter pattern",
        },
        {
            "provider": "qwen3_asr",
            "label": "Qwen3-ASR quality hypothesis",
            "interface": "official_python_runner",
            "default_model": "Qwen/Qwen3-ASR-1.7B",
            "fallback_model": "Qwen/Qwen3-ASR-0.6B",
            "forced_aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
            "gpu_policy": "cuda_preferred_explicit_fallback",
            "reuse_source": "QwenLM/Qwen3-ASR official qwen-asr package",
        },
        {
            "provider": "fun_asr_nano",
            "label": "Fun-ASR-Nano benchmark challenger",
            "interface": "funasr_python_runner",
            "default_model": "FunAudioLLM/Fun-ASR-Nano-2512",
            "gpu_policy": "cuda_preferred",
            "reuse_source": "FunAudioLLM/Fun-ASR-Nano official model through FunASR",
        },
        {
            "provider": "whisperx_alignment",
            "label": "WhisperX alignment",
            "interface": "alignment_only",
            "default_model": "large-v3",
            "gpu_policy": "cuda_preferred",
            "reuse_source": "WhisperX alignment/diarization best-practice branch",
        },
    ]


def _provider_api_key_configured(preset: dict[str, Any], env_names: list[str]) -> bool:
    if any(bool(os.environ.get(name)) for name in env_names):
        return True
    if str(preset.get("provider") or "") == "volcengine_coding_plan":
        base_env = str(preset.get("compatible_base_url_env") or "")
        base = os.environ.get(base_env) or ""
        return bool(os.environ.get("OPENAI_API_KEY")) and _is_volcengine_coding_plan_base_url(base)
    return False


def _is_volcengine_coding_plan_base_url(value: str | None) -> bool:
    text = str(value or "").strip().lower()
    return "volces.com" in text and "/api/coding" in text

def processing_profiles(config_path: str | Path | None = None) -> dict[str, Any]:
    profiles = json.loads(json.dumps(DEFAULT_PROCESSING_PROFILES, ensure_ascii=False))
    try:
        data = load_project_config(config_path)
    except Exception:
        return profiles
    raw = data.get("processing_profiles") if isinstance(data.get("processing_profiles"), dict) else {}
    default_name = str(raw.get("default") or profiles["default"]).strip()
    for name, values in raw.items():
        if name == "default" or not isinstance(values, dict):
            continue
        base = dict(profiles.get(name) if isinstance(profiles.get(name), dict) else {})
        base.update(values)
        profiles[name] = base
    profiles["default"] = default_name if isinstance(profiles.get(default_name), dict) else "quality"
    return profiles


def processing_profile(name: str = "", config_path: str | Path | None = None) -> dict[str, Any]:
    profiles = processing_profiles(config_path)
    selected = str(name or profiles.get("default") or "quality").strip()
    profile = profiles.get(selected) if isinstance(profiles.get(selected), dict) else profiles.get("quality", {})
    return {"name": selected if isinstance(profiles.get(selected), dict) else "quality", **dict(profile)}


def asr_runtime_profile(config_path: str | Path | None = None) -> dict[str, Any]:
    try:
        data = load_project_config(config_path)
    except Exception:
        return _normalise_asr_runtime({})
    raw = data.get("asr_runtime") if isinstance(data.get("asr_runtime"), dict) else {}
    return _normalise_asr_runtime(raw)


def set_asr_runtime_profile(
    *,
    provider: str = "",
    model: str = "",
    device: str = "",
    compute_type: str = "",
    vad_model: str = "",
    punc_model: str = "",
    spk_model: str = "",
    enable_vad: bool | None = None,
    enable_itn: bool | None = None,
    enable_punctuation: bool | None = None,
    enable_diarization: bool | None = None,
    merge_vad: bool | None = None,
    merge_length_s: int | None = None,
    audio_preprocess: bool | None = None,
    ffmpeg_normalize: bool | None = None,
    target_sample_rate: int | None = None,
    service_base_url: str = "",
    service_model: str = "",
    service_timeout_seconds: int | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    path = resolve_config_path(config_path)
    data = load_project_config(path)
    profile = asr_runtime_profile(path)
    if provider:
        profile["provider"] = str(provider).strip()
    if model:
        profile["model"] = str(model).strip()
    if device:
        profile["device"] = str(device).strip()
    if compute_type:
        profile["compute_type"] = str(compute_type).strip()
    if vad_model != "":
        profile["vad_model"] = str(vad_model).strip()
    if punc_model != "":
        profile["punc_model"] = str(punc_model).strip()
    if spk_model != "":
        profile["spk_model"] = str(spk_model).strip()
    for key, value in {
        "enable_vad": enable_vad,
        "enable_itn": enable_itn,
        "enable_punctuation": enable_punctuation,
        "enable_diarization": enable_diarization,
        "merge_vad": merge_vad,
    }.items():
        if value is not None:
            profile[key] = bool(value)
    if merge_length_s is not None:
        profile["merge_length_s"] = max(1, int(merge_length_s))
    audio = dict(profile.get("audio_preprocess") if isinstance(profile.get("audio_preprocess"), dict) else {})
    if audio_preprocess is not None:
        audio["enabled"] = bool(audio_preprocess)
    if ffmpeg_normalize is not None:
        audio["ffmpeg_normalize"] = bool(ffmpeg_normalize)
    if target_sample_rate is not None:
        audio["target_sample_rate"] = max(8000, int(target_sample_rate))
    profile["audio_preprocess"] = audio
    service = dict(profile.get("openai_compatible") if isinstance(profile.get("openai_compatible"), dict) else {})
    if service_base_url:
        service["base_url"] = str(service_base_url).strip()
    if service_model:
        service["model"] = str(service_model).strip()
    if service_timeout_seconds is not None:
        service["timeout_seconds"] = max(30, int(service_timeout_seconds))
    profile["openai_compatible"] = service
    profile = _normalise_asr_runtime(profile)
    if _contains_secret_key(profile):
        raise ValueError("asr_runtime profile must not contain secrets")
    data["asr_runtime"] = profile
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return {
        "ok": True,
        "schema": "video_knowledge_pipeline.set_asr_runtime_profile.v1",
        "config_path": str(path),
        "asr_runtime": profile,
        "secret_policy": "API keys were not written. Persist keys with user-level environment variables or private env files.",
    }


def _normalise_asr_runtime(raw: dict[str, Any]) -> dict[str, Any]:
    profile = json.loads(json.dumps(DEFAULT_ASR_RUNTIME, ensure_ascii=False))
    if isinstance(raw, dict):
        for key in ("provider", "model", "device", "compute_type", "vad_model", "punc_model", "spk_model"):
            value = str(raw.get(key) or "").strip()
            if value or key in {"vad_model", "punc_model", "spk_model"}:
                profile[key] = value
        for key in ("enable_vad", "enable_itn", "enable_punctuation", "enable_diarization", "merge_vad"):
            if key in raw:
                profile[key] = bool(raw.get(key))
        profile["merge_length_s"] = max(1, _safe_int(raw.get("merge_length_s"), int(profile["merge_length_s"])))
        if isinstance(raw.get("audio_preprocess"), dict):
            audio = dict(profile["audio_preprocess"])
            audio.update({key: value for key, value in raw["audio_preprocess"].items() if key in {"enabled", "ffmpeg_normalize", "target_sample_rate"}})
            audio["enabled"] = bool(audio.get("enabled"))
            audio["ffmpeg_normalize"] = bool(audio.get("ffmpeg_normalize"))
            audio["target_sample_rate"] = max(8000, _safe_int(audio.get("target_sample_rate"), 16000))
            profile["audio_preprocess"] = audio
        if isinstance(raw.get("openai_compatible"), dict):
            service = dict(profile["openai_compatible"])
            for key in ("base_url", "model"):
                value = str(raw["openai_compatible"].get(key) or "").strip()
                if value:
                    service[key] = value
            service["timeout_seconds"] = max(30, _safe_int(raw["openai_compatible"].get("timeout_seconds"), int(service["timeout_seconds"])))
            profile["openai_compatible"] = service
    if profile["provider"] not in ASR_RUNTIME_PROVIDERS:
        profile["provider"] = DEFAULT_ASR_RUNTIME["provider"]
    if profile["device"] not in ASR_RUNTIME_DEVICES:
        profile["device"] = DEFAULT_ASR_RUNTIME["device"]
    return profile


def set_vision_execution_profile(
    *,
    provider: str,
    model: str = "",
    base_url: str = "",
    multimodal_limit: int | None = None,
    temporal_limit: int | None = None,
    frame_count: int | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    path = resolve_config_path(config_path)
    data = load_project_config(path)
    profile = vision_execution_profile(path)
    provider = str(provider or "").strip()
    if not provider:
        raise ValueError("provider is required")
    profile["provider"] = provider
    model_text = str(model or "").strip()
    if model_text:
        profile["model"] = model_text
    else:
        profile.pop("model", None)
    base_url_text = str(base_url or "").strip()
    if base_url_text:
        profile["base_url"] = base_url_text
    else:
        profile.pop("base_url", None)
    if multimodal_limit is not None:
        profile["multimodal_limit"] = max(0, int(multimodal_limit))
    if temporal_limit is not None:
        profile["temporal_limit"] = max(0, int(temporal_limit))
    if frame_count is not None:
        profile["frame_count"] = max(5, min(int(frame_count), 12))
    if _contains_secret_key(profile):
        raise ValueError("vision_execution profile must not contain secrets")
    data["vision_execution"] = profile
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return {
        "ok": True,
        "schema": "video_knowledge_pipeline.set_vision_profile.v1",
        "config_path": str(path),
        "vision_execution": profile,
        "model_api_settings": model_api_settings_status(path),
        "secret_policy": "API keys were not written. Persist keys with user-level environment variables.",
    }

def service_config(service_name: str, config_path: str | Path | None = None) -> dict[str, Any]:
    data = load_project_config(config_path)
    services = data.get("services") if isinstance(data.get("services"), dict) else {}
    service = services.get(service_name)
    if not isinstance(service, dict):
        raise KeyError(f"service not configured: {service_name}")
    return service


def service_url(service_name: str, config_path: str | Path | None = None) -> str:
    service = service_config(service_name, config_path)
    host = str(service.get("host") or "").strip()
    port = _int_port(service.get("port"))
    path = str(service.get("path") or "").strip()
    scheme = str(service.get("scheme") or "http").strip() or "http"
    if not host or not port:
        raise ValueError(f"service {service_name} must define host and port")
    if path and not path.startswith("/"):
        path = "/" + path
    return f"{scheme}://{host}:{port}{path}"


def runtime_config_manifest() -> dict[str, Any]:
    status = config_status()
    return {
        "schema": CONFIG_SCHEMA,
        "config_path": status.get("config_path", ""),
        "ok": bool(status.get("ok")),
        "services": status.get("services", {}),
        "service_urls": status.get("service_urls", {}),
        "vision_execution": status.get("vision_execution", {}),
        "ebook_pipeline": status.get("ebook_pipeline", {}),
        "processing_profiles": status.get("processing_profiles", {}),
        "validation": status.get("validation", {}),
    }


def vision_execution_profile(config_path: str | Path | None = None) -> dict[str, Any]:
    try:
        data = load_project_config(config_path)
    except Exception:
        return dict(DEFAULT_VISION_EXECUTION)
    raw = data.get("vision_execution") if isinstance(data.get("vision_execution"), dict) else {}
    profile = dict(DEFAULT_VISION_EXECUTION)
    for key in ("provider", "model", "base_url"):
        value = str(raw.get(key) or "").strip()
        if value:
            profile[key] = value
    for key in ("multimodal_limit", "temporal_limit", "frame_count"):
        profile[key] = max(0, _safe_int(raw.get(key), int(profile[key])))
    return profile


def ebook_pipeline_profile(config_path: str | Path | None = None) -> dict[str, Any]:
    try:
        data = load_project_config(config_path)
    except Exception:
        return dict(DEFAULT_EBOOK_PIPELINE)
    raw = data.get("ebook_pipeline") if isinstance(data.get("ebook_pipeline"), dict) else {}
    profile = dict(DEFAULT_EBOOK_PIPELINE)
    profile["execute_default"] = bool(raw.get("execute_default", profile["execute_default"]))
    routes = raw.get("include_routes")
    if isinstance(routes, list):
        parsed_routes = [str(route).strip() for route in routes if str(route).strip()]
        if parsed_routes:
            profile["include_routes"] = parsed_routes
    elif isinstance(routes, str):
        parsed_routes = [part.strip() for part in routes.split(",") if part.strip()]
        if parsed_routes:
            profile["include_routes"] = parsed_routes
    profile["limit"] = max(0, _safe_int(raw.get("limit"), int(profile["limit"])))
    profile["timeout_seconds"] = max(30, _safe_int(raw.get("timeout_seconds"), int(profile["timeout_seconds"])))
    device = str(raw.get("rapidocr_device", profile.get("rapidocr_device") or "auto")).strip().lower()
    if device in {"auto", "cuda", "gpu", "cpu", "off"}:
        profile["rapidocr_device"] = "cuda" if device == "gpu" else "cpu" if device == "off" else device
    profile["rapidocr_cuda_device_id"] = max(0, _safe_int(raw.get("rapidocr_cuda_device_id"), int(profile["rapidocr_cuda_device_id"])))
    return profile


def resolve_vision_execution_profile(
    *,
    provider_config: dict[str, Any] | None = None,
    multimodal_limit: int | None = None,
    temporal_limit: int | None = None,
    frame_count: int | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    configured = vision_execution_profile(config_path)
    configured_provider = str(configured.get("provider") or "").strip()
    merged_provider = {
        key: configured[key]
        for key in ("provider", "model", "base_url")
        if key in configured and str(configured.get(key) or "").strip()
    }
    env_provider = _env_vision_provider_config()
    if env_provider:
        requested_provider = str(env_provider.get("provider") or "").strip()
        if requested_provider and requested_provider != configured_provider:
            for inherited_key in ("model", "base_url"):
                if inherited_key not in env_provider:
                    merged_provider.pop(inherited_key, None)
        merged_provider.update(env_provider)
    if provider_config:
        requested_provider = str(provider_config.get("provider") or "").strip()
        current_provider = str(merged_provider.get("provider") or configured_provider).strip()
        if requested_provider and requested_provider != current_provider:
            for inherited_key in ("model", "base_url"):
                if inherited_key not in provider_config:
                    merged_provider.pop(inherited_key, None)
        merged_provider.update(provider_config)
    return {
        "provider_config": merged_provider,
        "multimodal_limit": _non_negative_int(multimodal_limit, int(configured.get("multimodal_limit") or 19)),
        "temporal_limit": _non_negative_int(temporal_limit, int(configured.get("temporal_limit") or 3)),
        "frame_count": max(5, min(_non_negative_int(frame_count, int(configured.get("frame_count") or 8)), 12)),
    }


def public_vision_provider_profile(provider_config: dict[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in dict(provider_config or {}).items() if not _secret_key(key)}


def validate_project_config(config: dict[str, Any]) -> dict[str, Any]:
    services = config.get("services") if isinstance(config.get("services"), dict) else {}
    issues: list[dict[str, Any]] = []
    for service_name, requirements in REQUIRED_SERVICES.items():
        service = services.get(service_name)
        if not isinstance(service, dict):
            issues.append({"service": service_name, "key": "missing_service", "message": f"missing service: {service_name}"})
            continue
        kind = requirements["kind"]
        if kind == "http":
            host = str(service.get("host") or "").strip()
            port = _safe_int_port(service.get("port"))
            path = str(service.get("path") or "").strip()
            if not host:
                issues.append({"service": service_name, "key": "missing_host", "message": f"{service_name} missing host"})
            if not port:
                issues.append({"service": service_name, "key": "missing_port", "message": f"{service_name} missing valid port"})
            if path and not path.startswith("/"):
                issues.append({"service": service_name, "key": "path_without_slash", "message": f"{service_name} path should start with /"})
        elif kind == "static_file":
            entrypoint = str(service.get("entrypoint") or "").strip()
            if not entrypoint:
                issues.append({"service": service_name, "key": "missing_entrypoint", "message": f"{service_name} missing entrypoint"})
        elif kind == "stdio":
            transport = str(service.get("transport") or "").strip()
            if transport != "stdio":
                issues.append({"service": service_name, "key": "unsupported_transport", "message": f"{service_name} transport should be stdio"})
    raw_vision = config.get("vision_execution")
    if raw_vision is not None and not isinstance(raw_vision, dict):
        issues.append({"service": "vision_execution", "key": "invalid_profile", "message": "vision_execution must be an object"})
    elif isinstance(raw_vision, dict):
        for key in ("multimodal_limit", "temporal_limit", "frame_count"):
            value = _safe_int(raw_vision.get(key), -1)
            if value < 0:
                issues.append({"service": "vision_execution", "key": f"invalid_{key}", "message": f"{key} must be a non-negative integer"})
        if _contains_secret_key(raw_vision):
            issues.append({"service": "vision_execution", "key": "secret_in_config", "message": "vision_execution must not contain secrets"})
    raw_asr = config.get("asr_runtime")
    if raw_asr is not None and not isinstance(raw_asr, dict):
        issues.append({"service": "asr_runtime", "key": "invalid_profile", "message": "asr_runtime must be an object"})
    elif isinstance(raw_asr, dict):
        provider = str(raw_asr.get("provider") or "").strip()
        if provider and provider not in ASR_RUNTIME_PROVIDERS:
            issues.append({"service": "asr_runtime", "key": "invalid_provider", "message": "asr_runtime provider is unsupported"})
        device = str(raw_asr.get("device") or "").strip()
        if device and device not in ASR_RUNTIME_DEVICES:
            issues.append({"service": "asr_runtime", "key": "invalid_device", "message": "asr_runtime device must be cuda_preferred, auto, cuda, or cpu"})
        if _contains_secret_key(raw_asr):
            issues.append({"service": "asr_runtime", "key": "secret_in_config", "message": "asr_runtime must not contain secrets"})

    raw_profiles = config.get("processing_profiles")
    if raw_profiles is not None and not isinstance(raw_profiles, dict):
        issues.append({"service": "processing_profiles", "key": "invalid_profile", "message": "processing_profiles must be an object"})
    elif isinstance(raw_profiles, dict):
        default_name = str(raw_profiles.get("default") or "").strip()
        if default_name and not isinstance(raw_profiles.get(default_name), dict):
            issues.append({"service": "processing_profiles", "key": "missing_default_profile", "message": f"processing profile {default_name} is missing"})
        if _contains_secret_key(raw_profiles):
            issues.append({"service": "processing_profiles", "key": "secret_in_config", "message": "processing_profiles must not contain secrets"})

    raw_ebook = config.get("ebook_pipeline")
    if raw_ebook is not None and not isinstance(raw_ebook, dict):
        issues.append({"service": "ebook_pipeline", "key": "invalid_profile", "message": "ebook_pipeline must be an object"})
    elif isinstance(raw_ebook, dict):
        for key in ("limit", "timeout_seconds", "rapidocr_cuda_device_id"):
            value = _safe_int(raw_ebook.get(key), -1)
            if value < 0:
                issues.append({"service": "ebook_pipeline", "key": f"invalid_{key}", "message": f"{key} must be a non-negative integer"})
        rapidocr_device = str(raw_ebook.get("rapidocr_device") or "").strip().lower()
        if rapidocr_device and rapidocr_device not in {"auto", "cuda", "gpu", "cpu", "off"}:
            issues.append({"service": "ebook_pipeline", "key": "invalid_rapidocr_device", "message": "rapidocr_device must be auto, cuda, gpu, cpu, or off"})
        routes = raw_ebook.get("include_routes")
        if routes is not None and not isinstance(routes, (list, str)):
            issues.append({"service": "ebook_pipeline", "key": "invalid_include_routes", "message": "include_routes must be a list or comma-separated string"})
        if _contains_secret_key(raw_ebook):
            issues.append({"service": "ebook_pipeline", "key": "secret_in_config", "message": "ebook_pipeline must not contain secrets"})
    return {
        "ok": not issues,
        "required_services": sorted(REQUIRED_SERVICES),
        "issue_count": len(issues),
        "issues": issues,
    }


def _sanitized_services(services: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, service in services.items():
        if not isinstance(service, dict):
            rows[str(name)] = service
            continue
        rows[str(name)] = {key: value for key, value in service.items() if "key" not in str(key).lower() and "token" not in str(key).lower()}
    return rows


def _service_urls(services: dict[str, Any]) -> dict[str, str]:
    urls: dict[str, str] = {}
    for name, service in services.items():
        if not isinstance(service, dict):
            continue
        host = str(service.get("host") or "").strip()
        port = _safe_int_port(service.get("port"))
        if not host or not port:
            continue
        path = str(service.get("path") or "").strip()
        scheme = str(service.get("scheme") or "http").strip() or "http"
        if path and not path.startswith("/"):
            path = "/" + path
        urls[str(name)] = f"{scheme}://{host}:{port}{path}"
        docker_host = str(service.get("docker_host") or "").strip()
        if docker_host:
            urls[f"{name}_docker"] = f"{scheme}://{docker_host}:{port}{path}"
    return urls


def _safe_int_port(value: Any) -> int:
    try:
        return _int_port(value)
    except ValueError:
        return 0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _contains_secret_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if "key" in lowered or "token" in lowered or "secret" in lowered:
                return True
            if _contains_secret_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_secret_key(item) for item in value)
    elif isinstance(value, str):
        lowered = value.lower()
        secret_markers = (
            "api_key=",
            "apikey=",
            "access_token=",
            "token=",
            "secret=",
            "authorization: bearer",
            "bearer sk-",
            "sk-",
        )
        return any(marker in lowered for marker in secret_markers)
    return False

def _non_negative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(0, parsed)


def _secret_key(key: Any) -> bool:
    text = str(key).lower()
    return "key" in text or "token" in text or "secret" in text


def _env_vision_provider_config() -> dict[str, str]:
    values = {
        "provider": os.environ.get("LECTURE_VISION_PROVIDER", ""),
        "model": os.environ.get("LECTURE_VISION_MODEL", ""),
        "base_url": os.environ.get("LECTURE_VISION_BASE_URL", ""),
    }
    return {key: str(value).strip() for key, value in values.items() if str(value or "").strip()}


def _int_port(value: Any) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        return 0
    if port < 1 or port > 65535:
        raise ValueError(f"invalid port: {port}")
    return port
