from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .file_hash import sha256_file as _sha256
from .path_defaults import local_model_root
from .config import load_project_config
from .model_api_settings import (
    LOCAL_PRODUCTION_ROUTE_PRESET_ID,
    LOCAL_PRODUCTION_ROUTE_TASK_PROFILES,
    apply_model_api_route_preset,
    resolve_model_api_route,
)
from .storage import write_json
from .utils import now_iso


SCHEMA = "video_knowledge_pipeline.local_production_preset.v1"


def install_local_production_preset(
    output_dir: str | Path,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Materialise an isolated, secretless, local-only VKP production preset."""

    root = Path(output_dir).expanduser().resolve()
    settings_path = root / "model-api-settings.json"
    secrets_path = root / "model-api-secrets.json"
    config_path = root / "video-knowledge-pipeline.json"
    manifest_path = root / "local-production-v1.json"
    paths = {
        "root": str(root),
        "model_api_settings": str(settings_path),
        "model_api_secrets": str(secrets_path),
        "pipeline_config": str(config_path),
        "manifest": str(manifest_path),
    }
    if not write:
        return {
            "schema": SCHEMA,
            "preset_id": LOCAL_PRODUCTION_ROUTE_PRESET_ID,
            "status": "planned",
            "write": False,
            "paths": paths,
            "remote_requests_made": False,
            "remote_destinations": [],
        }

    root.mkdir(parents=True, exist_ok=True)
    route_status = apply_model_api_route_preset(
        LOCAL_PRODUCTION_ROUTE_PRESET_ID,
        settings_path=settings_path,
        secrets_path=secrets_path,
    )
    _require_local_only_route_status(route_status)
    pipeline_config = _local_pipeline_config()
    write_json(config_path, pipeline_config)

    route_snapshots = {
        task: resolve_model_api_route(
            task,
            execution_location="local",
            settings_path=settings_path,
        )
        for task in LOCAL_PRODUCTION_ROUTE_TASK_PROFILES
    }
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "preset_id": LOCAL_PRODUCTION_ROUTE_PRESET_ID,
        "status": "installed",
        "write": True,
        "paths": paths,
        "models": {
            "asr_primary": {
                "provider": "funasr_sensevoice",
                "model": "iic/SenseVoiceSmall",
                "device": "cuda",
            },
            "asr_secondary": {
                "provider": "qwen_asr",
                "model": str(local_model_root() / "Qwen3-ASR-1.7B"),
                "device": "cuda",
            },
            "ocr": {
                "provider": "ebook_markdown_pipeline",
                "engine": "rapidocr",
                "device": "cuda",
            },
            "vision": {
                "provider": "lmstudio_openai_compatible",
                "adapter_backend": "builtin",
                "base_url": "http://127.0.0.1:1234/v1",
                "model": "qwen/qwen3-vl-8b",
            },
            "text": {
                "provider": "lmstudio_openai_compatible",
                "adapter_backend": "builtin",
                "base_url": "http://127.0.0.1:1234/v1",
                "model": "qwen/qwen3.5-9b",
                "reasoning_effort": "none",
                "response_format": "text",
                "max_tokens": 1200,
            },
        },
        "task_routes": dict(LOCAL_PRODUCTION_ROUTE_TASK_PROFILES),
        "route_revisions": {
            task: {
                "route_id": snapshot["route_id"],
                "route_revision": snapshot["route_revision"],
                "virtual_model": snapshot["virtual_model"],
            }
            for task, snapshot in route_snapshots.items()
        },
        "network_policy": {
            "execution_location": "local",
            "remote_destinations": [],
            "remote_requests_allowed": False,
            "automatic_local_remote_fallback": False,
            "loopback_destinations": ["127.0.0.1", "::1", "localhost"],
        },
        "runtime_environment": {
            "VIDEO_KNOWLEDGE_PIPELINE_CONFIG": str(config_path),
            "VKP_MODEL_API_SETTINGS_PATH": str(settings_path),
            "VKP_MODEL_API_SECRETS_PATH": str(secrets_path),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "LITELLM_TELEMETRY": "False",
        },
        "verification": {
            "installed_only": True,
            "local_services_verified": False,
            "end_to_end_verified": False,
            "remote_requests_made": False,
        },
        "artifact_hashes": {
            "model_api_settings": _sha256(settings_path),
            "pipeline_config": _sha256(config_path),
        },
        "remote_requests_made": False,
        "updated_at": now_iso(),
    }
    write_json(manifest_path, manifest)
    return manifest


def _local_pipeline_config() -> dict[str, Any]:
    config = json.loads(
        json.dumps(load_project_config(), ensure_ascii=False)
    )
    config["vision_execution"] = {
        "provider": "local_qwen_vl",
        "model": "qwen/qwen3-vl-8b",
        "base_url": "http://127.0.0.1:1234/v1",
        "multimodal_limit": 19,
        "temporal_limit": 6,
        "frame_count": 8,
    }
    config["ebook_pipeline"] = {
        "execute_default": True,
        "include_routes": ["document_visual", "mixed"],
        "limit": 36,
        "timeout_seconds": 300,
        "rapidocr_device": "cuda",
        "rapidocr_cuda_device_id": 0,
    }
    config["asr_runtime"] = {
        "provider": "funasr_sensevoice",
        "model": "iic/SenseVoiceSmall",
        "device": "cuda",
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
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "qwen/qwen3.5-9b",
            "timeout_seconds": 600,
        },
    }
    config["processing_profiles"] = {
        "default": LOCAL_PRODUCTION_ROUTE_PRESET_ID,
        LOCAL_PRODUCTION_ROUTE_PRESET_ID: {
            "primary_asr_preset": "sensevoice",
            "secondary_asr_preset": "qwen3-asr-1.7b",
            "secondary_fallback_preset": "",
            "forced_aligner_model": "Qwen/Qwen3-ForcedAligner-0.6B",
            "chapter_mode": "semantic",
            "text_llm_auto_execute": True,
            "data_export_allowed": False,
            "local_model_execution_allowed": True,
            "llm_preflight_call_threshold": 20,
            "llm_preflight_input_char_threshold": 120000,
            "cloud_audio_triage_only": False,
            "cloud_vision_triage_only": False,
        },
    }
    config["local_production"] = {
        "preset_id": LOCAL_PRODUCTION_ROUTE_PRESET_ID,
        "asr_backend": "sensevoice",
        "ocr_backend": "ebook_markdown_pipeline",
        "text_model": "qwen/qwen3.5-9b",
        "vision_model": "qwen/qwen3-vl-8b",
        "remote_requests_allowed": False,
        "automatic_local_remote_fallback": False,
    }
    return config


def _require_local_only_route_status(status: dict[str, Any]) -> None:
    profiles = [
        row for row in status.get("profiles") or [] if isinstance(row, dict)
    ]
    selected = set(LOCAL_PRODUCTION_ROUTE_TASK_PROFILES.values())
    selected_profiles = [
        row for row in profiles if str(row.get("id") or "") in selected
    ]
    if len(selected_profiles) != len(selected):
        raise ValueError("local production preset profiles are incomplete")
    for profile in selected_profiles:
        host = str(
            urlsplit(str(profile.get("base_url") or "")).hostname or ""
        ).lower()
        if str(profile.get("location") or "") != "local":
            raise ValueError("local production preset selected a remote profile")
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("local production preset selected a non-loopback URL")
    for pool in status.get("route_pools") or []:
        if str(pool.get("location") or "") != "local_only":
            raise ValueError("local production preset contains a remote pool")
