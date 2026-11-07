from __future__ import annotations

from copy import deepcopy
from typing import Any


PROFILE_SCHEMA = "video_knowledge_pipeline.content_clip_query_profile.v1"


# Intent: select evidence, boundary, and post-cut checks by requested content.
# Decision: keep a small versioned registry over VKP's existing retrieval and
# evidence contracts instead of adding another index or model router.
# Reason: ASR is authoritative for quotes, while shots/OCR/temporal evidence are
# authoritative for visual and screen events; one universal strategy creates
# false matches and poor boundaries.
# Evidence: existing VKP VideoRAG, moment-index, technical-shot, OCR, temporal,
# word-timestamp, and clip-alignment contracts; Shot2Story task decomposition at
# ae26ac3d2f9e9a91a7fd0653bfb6a2b3cb250308.
# Effective scope: derived content-clip candidate and validation artifacts only.
_PROFILES: dict[str, dict[str, Any]] = {
    "spoken-quote-v1": {
        "purposes": ["quote"],
        "retrieval_sources": ["video_rag", "moment_index", "transcript"],
        "default_required_modalities": ["asr"],
        "default_optional_modalities": ["audio", "shot"],
        "boundary_strategy": "complete_sentence",
        "verification": ["clip_asr", "speaker", "subtitle_semantics"],
    },
    "lecture-explanation-v1": {
        "purposes": ["explanation"],
        "retrieval_sources": ["video_rag", "moment_index", "transcript", "ocr"],
        "default_required_modalities": ["asr"],
        "default_optional_modalities": ["ocr", "shot"],
        "boundary_strategy": "complete_sentence",
        "verification": ["clip_asr", "claim_terms", "sentence_boundary"],
    },
    "tutorial-step-v1": {
        "purposes": ["tutorial_step"],
        "retrieval_sources": ["video_rag", "moment_index", "transcript", "ocr", "visual"],
        "default_required_modalities": ["asr", "visual"],
        "default_optional_modalities": ["ocr", "shot"],
        "boundary_strategy": "instruction_to_stable_result",
        "verification": ["clip_asr", "visual_human_review", "result_frame"],
    },
    "visual-event-v1": {
        "purposes": ["visual_event"],
        "retrieval_sources": ["video_rag", "moment_index", "visual", "shot"],
        "default_required_modalities": ["visual", "shot"],
        "default_optional_modalities": ["asr", "ocr"],
        "boundary_strategy": "action_start_peak_end",
        "verification": ["visual_human_review", "technical_shot"],
    },
    "screen-content-v1": {
        "purposes": ["screen_text"],
        "retrieval_sources": ["video_rag", "moment_index", "ocr", "shot"],
        "default_required_modalities": ["ocr"],
        "default_optional_modalities": ["asr", "visual", "shot"],
        "boundary_strategy": "stable_screen_content",
        "verification": ["clip_ocr", "screen_stability"],
    },
    "audio-event-v1": {
        "purposes": ["audio_event"],
        "retrieval_sources": ["video_rag", "moment_index", "audio"],
        "default_required_modalities": ["audio"],
        "default_optional_modalities": ["asr", "shot"],
        "boundary_strategy": "audio_event_bounds",
        "verification": ["audio_human_review"],
    },
    "broll-v1": {
        "purposes": ["broll"],
        "retrieval_sources": ["video_rag", "moment_index", "visual", "shot"],
        "default_required_modalities": ["shot"],
        "default_optional_modalities": ["visual", "ocr", "asr"],
        "boundary_strategy": "whole_technical_shot",
        "verification": ["technical_shot", "visual_human_review", "exclusion_terms"],
    },
    "highlight-v1": {
        "purposes": ["highlight"],
        "retrieval_sources": ["video_rag", "moment_index", "asr", "visual", "audio"],
        "default_required_modalities": ["asr"],
        "default_optional_modalities": ["visual", "audio", "shot"],
        "boundary_strategy": "cause_peak_result",
        "verification": ["clip_asr", "context_complete", "multimodal_human_review"],
    },
    "story-beat-v1": {
        "purposes": ["story_beat"],
        "retrieval_sources": ["video_rag", "moment_index", "asr", "visual", "shot"],
        "default_required_modalities": ["asr"],
        "default_optional_modalities": ["visual", "ocr", "shot"],
        "boundary_strategy": "cause_change_result",
        "verification": ["clip_asr", "context_complete", "visual_human_review"],
    },
}


_PURPOSE_DEFAULTS = {
    purpose: profile_id
    for profile_id, payload in _PROFILES.items()
    for purpose in payload["purposes"]
}


def list_content_clip_query_profiles() -> list[dict[str, Any]]:
    return [get_content_clip_query_profile(profile_id) for profile_id in sorted(_PROFILES)]


def get_content_clip_query_profile(profile_id: str) -> dict[str, Any]:
    key = str(profile_id or "").strip()
    if key not in _PROFILES:
        raise ValueError(f"unknown content clip query profile: {key}")
    return {"schema": PROFILE_SCHEMA, "profile_id": key, **deepcopy(_PROFILES[key])}


def resolve_content_clip_query_profile(clip: dict[str, Any]) -> dict[str, Any]:
    explicit = str(clip.get("profile_id") or "").strip()
    purpose = str(clip.get("purpose") or "").strip()
    profile_id = explicit or _PURPOSE_DEFAULTS.get(purpose, "")
    if not profile_id:
        raise ValueError(f"no content clip query profile for purpose: {purpose}")
    profile = get_content_clip_query_profile(profile_id)
    if purpose and purpose not in profile["purposes"]:
        raise ValueError(f"profile {profile_id} does not support purpose {purpose}")
    return profile
