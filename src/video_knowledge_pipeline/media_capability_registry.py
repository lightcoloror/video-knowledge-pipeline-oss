from __future__ import annotations

from copy import deepcopy
from typing import Any


SCHEMA = "video_knowledge_pipeline.media_capability_registry.v1"
PROTOCOL = "mediakit_async_v1"
PROVIDER = "volcengine_mediakit"
UPSTREAM_COMMIT = "e0538f5e08150ce21d0dd5be5caeb23f5298c952"
CONTROL_PLANE_DESTINATION = "amk.cn-beijing.volces.com"


LOCAL_VIDEO_STRUCTURE_CAPABILITIES: list[dict[str, Any]] = [
    {
        "task": "shot_boundary_detection",
        "label": "镜头切分",
        "status": "ready_local",
        "implementation": "pyscenedetect_with_ffmpeg_fallback",
        "input": "local_video",
        "output": "candidate_shot_boundaries",
        "model_required": False,
        "cloud_allowed": False,
    },
    {
        "task": "semantic_scene_segmentation",
        "label": "语义场景切分",
        "status": "ready_local",
        "implementation": "semantic_chapter_evidence_fusion",
        "input": "timeline_asr_ocr_tagger_and_shot_boundaries",
        "output": "candidate_semantic_scenes",
        "model_required": False,
        "cloud_allowed": False,
    },
    {
        "task": "storyline_structure",
        "label": "剧情线与内容结构",
        "status": "ready_local_evidence",
        "implementation": "ordered_semantic_scene_storyline",
        "input": "candidate_semantic_scenes",
        "output": "candidate_storyline",
        "model_required": False,
        "cloud_allowed": False,
    },
    {
        "task": "highlight_detection",
        "label": "高光检测",
        "status": "optional_local_runtime",
        "implementation": "lighthouse_cg_detr",
        "input": "local_scene_clip_and_text_query",
        "output": "candidate_time_windows_and_saliency",
        "model_required": True,
        "default_model": "cg_detr",
        "device_policy": "gpu_required",
        "cloud_allowed": False,
        "automatic_model_download": False,
    },
    {
        "task": "general_image_tagging",
        "label": "通用图像标签",
        "status": "optional_local_runtime",
        "implementation": "recognize_anything_ram_plus",
        "input": "local_keyframes",
        "output": "candidate_multilabel_annotations",
        "model_required": True,
        "default_model": "ram_plus_swin_large_14m",
        "device_policy": "gpu_required",
        "compatibility_baselines": ["cl_tagger", "wd_eva02_large_tagger_v3"],
        "cloud_allowed": False,
        "automatic_model_download": False,
    },
]


EXCLUDED_CAPABILITIES = [
    {
        "task": "native_whole_video_understanding",
        "status": "disabled",
        "reason": "cost_latency_and_compute_policy",
    }
]


_COMMON = {
    "provider": PROVIDER,
    "location": "remote",
    "supported_locations": ["remote"],
    "async": True,
    "poll": {"method": "GET", "path_template": "/api/v1/tasks/{task_id}"},
    "protocol": PROTOCOL,
    "consent_required": True,
    "allowlist_required": True,
    "authorization_status": "not_configured",
    "execution_status": "official_cli_adapter",
    "candidate_only": True,
    "estimated_cost": "unknown",
    "control_plane_destination": CONTROL_PLANE_DESTINATION,
    "upload_destinations_status": "not_audited",
}


MEDIA_CAPABILITIES: dict[str, dict[str, Any]] = {
    "scene_segmentation": {
        **_COMMON,
        "task": "scene_segmentation",
        "label": "场景切分",
        "priority": "P0",
        "provider_task": "segment-scenes",
        "submit": {"method": "POST", "path": "/api/v1/tools/segment-scenes"},
        "artifact_types": ["video"],
        "min_artifacts": 1,
        "max_artifacts": 1,
        "parameters": [
            {"name": "enable_clip_fade", "type": "boolean", "required": False},
            {
                "name": "segment_threshold",
                "type": "number",
                "required": False,
                "minimum": 0,
                "exclusive_maximum": 100,
            },
            {"name": "min_duration", "type": "number", "required": False, "exclusive_minimum": 0},
            {"name": "max_duration", "type": "number", "required": False, "exclusive_minimum": 0},
        ],
        "normalised_output": "segments",
    },
    "storyline": {
        **_COMMON,
        "task": "storyline",
        "label": "剧情故事线分析",
        "priority": "P1",
        "provider_task": "analyze-video-storyline",
        "submit": {"method": "POST", "path": "/api/v1/tools/analyze-video-storyline"},
        "artifact_types": ["video"],
        "min_artifacts": 1,
        "max_artifacts": 30,
        "parameters": [
            {"name": "enable_snapshot", "type": "boolean", "required": False},
        ],
        "normalised_output": "clips_and_highlights",
    },
    "highlight_detection": {
        **_COMMON,
        "task": "highlight_detection",
        "label": "高光片段提取",
        "priority": "P2",
        "provider_task": "analyze-video-highlights",
        "submit": {"method": "POST", "path": "/api/v1/tools/analyze-video-highlights"},
        "artifact_types": ["video"],
        "min_artifacts": 1,
        "max_artifacts": 100,
        "parameters": [
            {
                "name": "model",
                "type": "string",
                "required": True,
                "enum": ["Miniseries", "Game"],
            },
            {
                "name": "mode",
                "type": "string",
                "required": True,
                "enum": ["StorylineCuts", "HighlightExtract"],
            },
            {"name": "minigame_info", "type": "object", "required": False, "max_bytes": 8192},
        ],
        "normalised_output": "highlights",
    },
    "video_ocr": {
        **_COMMON,
        "task": "video_ocr",
        "label": "视频连续 OCR",
        "priority": "P2",
        "provider_task": "video-ocr",
        "submit": {"method": "POST", "path": "/api/v1/tools/video-ocr"},
        "artifact_types": ["video"],
        "min_artifacts": 1,
        "max_artifacts": 1,
        "parameters": [
            {
                "name": "mode",
                "type": "string",
                "required": False,
                "enum": ["Subtitle", "Detailed"],
                "default": "Subtitle",
            },
        ],
        "normalised_output": "visual_text_segments",
    },
    "video_asr": {
        **_COMMON,
        "task": "video_asr",
        "label": "媒体服务 ASR",
        "priority": "P2",
        "provider_task": "asr-subtitles",
        "submit": {"method": "POST", "path": "/api/v1/tools/asr-subtitles"},
        "artifact_types": ["video", "audio"],
        "min_artifacts": 1,
        "max_artifacts": 1,
        "parameters": [
            {
                "name": "content_type",
                "type": "string",
                "required": False,
                "enum": ["speech", "singing"],
            },
            {
                "name": "language",
                "type": "string",
                "required": False,
                "enum": ["cmn-Hans-CN", "eng-US"],
            },
            {"name": "enable_speaker_info", "type": "boolean", "required": False},
            {"name": "enable_confidence", "type": "boolean", "required": False},
        ],
        "normalised_output": "asr_sidecar",
    },
}


def media_capability(task: str) -> dict[str, Any]:
    key = str(task or "").strip().lower().replace("-", "_")
    if key not in MEDIA_CAPABILITIES:
        expected = ", ".join(MEDIA_CAPABILITIES)
        raise ValueError(f"unsupported media capability: {task}; expected one of: {expected}")
    return deepcopy(MEDIA_CAPABILITIES[key])


def media_capability_registry_status() -> dict[str, Any]:
    rows = [deepcopy(row) for row in MEDIA_CAPABILITIES.values()]
    return {
        "schema": SCHEMA,
        "status": "execution_capable",
        "protocol": PROTOCOL,
        "provider": PROVIDER,
        "capability_count": len(rows),
        "capabilities": rows,
        "tasks": [row["task"] for row in rows],
        "local_video_structure_capabilities": deepcopy(LOCAL_VIDEO_STRUCTURE_CAPABILITIES),
        "excluded_capabilities": deepcopy(EXCLUDED_CAPABILITIES),
        "source": {
            "repository": "volcengine/mediakit-cli",
            "commit": UPSTREAM_COMMIT,
            "reuse_mode": "contract_and_schema_adaptation",
        },
        "routing": {
            "route_id": "",
            "route_revision": "",
            "remote_destination_allowlisted": True,
            "upload_destinations_audited": "provider_managed_by_official_cli",
        },
        "operator_boundary": {
            "real_remote_execution_available": True,
            "arbitrary_provider_urls_allowed": False,
            "api_keys_in_requests_allowed": False,
            "automatic_upload_allowed": "only_after_explicit_consent",
            "silent_local_remote_fallback_allowed": False,
            "timeline_writeback_allowed": False,
            "saving_configuration_authorizes_egress": False,
            "native_whole_video_understanding_enabled": False,
        },
    }
