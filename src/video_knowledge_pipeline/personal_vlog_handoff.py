from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .personal_vlog import (
    RECIPE_SCHEMA,
    PersonalVlogBoundaryError,
    PersonalVlogDriftError,
    public_recipe_projection,
)


MVP_V4_SCHEMA = "video_creation_pipeline.vkp_edit_job.v4"


def build_mvp_v4_candidate_handoff(
    project: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a Vlog recipe into a path-redacted MVP v4 candidate contract."""

    if recipe.get("schema") != RECIPE_SCHEMA:
        raise ValueError("personal Vlog recipe schema is invalid")
    expected_project = str(project.get("project_identity_sha256") or "")
    if not expected_project or str(recipe.get("project_identity_sha256") or "") != expected_project:
        raise PersonalVlogDriftError("recipe_project_identity_mismatch")
    recipe = public_recipe_projection(recipe)
    assets = {
        str(row.get("asset_id") or ""): row
        for row in project.get("assets") or []
        if isinstance(row, Mapping) and row.get("asset_id")
    }
    referenced_ids = [
        str(row.get("source_asset_id") or "")
        for row in recipe.get("clips") or []
        if isinstance(row, Mapping)
    ]
    if not referenced_ids or any(asset_id not in assets for asset_id in referenced_ids):
        raise PersonalVlogBoundaryError("recipe_references_unknown_source_asset")

    source_assets = []
    for asset_id in dict.fromkeys(referenced_ids):
        asset = assets[asset_id]
        source_assets.append(
            {
                "source_asset_id": asset_id,
                "media_type": str(asset.get("media_type") or ""),
                "source_sha256": str(asset.get("source_sha256") or ""),
                "private_resolver": {
                    "kind": "personal_vlog_project_manifest",
                    "project_id": project.get("project_id"),
                    "source_path_exported": False,
                },
            }
        )

    candidate_clips = []
    for row in recipe.get("clips") or []:
        if not isinstance(row, Mapping):
            continue
        clip = {
            key: row.get(key)
            for key in (
                "clip_id",
                "source_asset_id",
                "media_type",
                "event_id",
                "story_role",
                "timeline_start_seconds",
                "timeline_duration_seconds",
                "hold_seconds",
                "source_range",
                "audio_policy",
                "crop_plan",
                "evidence_refs",
                "uncertainty",
            )
            if row.get(key) is not None
        }
        clip["status"] = "candidate_only"
        clip["review_required"] = True
        candidate_clips.append(clip)

    payload = {
        "schema": MVP_V4_SCHEMA,
        "contract_mode": "multi_source_personal_vlog",
        "source_project": {
            "schema": project.get("schema"),
            "project_id": project.get("project_id"),
            "project_identity_sha256": expected_project,
            "source_paths_exported": False,
            "private_project_manifest_required_for_resolution": True,
        },
        "recipe": {
            "schema": recipe.get("schema"),
            "preset": recipe.get("preset"),
            "recipe_sha256": recipe.get("recipe_sha256"),
            "aspect_ratio": (recipe.get("target") or {}).get("aspect_ratio"),
            "target_seconds": (recipe.get("target") or {}).get("target_seconds"),
        },
        "source_assets": source_assets,
        "candidate_clips": candidate_clips,
        "readiness": {
            "status": "review_required",
            "blockers": [],
            "candidate_only": True,
        },
        "consumer_contract": {
            "photo_layout": "reuse_existing",
            "smart_reframe": "reuse_existing",
            "music_planning": "reuse_existing",
            "audio_mastering": "reuse_existing",
            "ffmpeg_outlet": "single_existing_outlet_after_approval",
        },
        "operator_boundary": {
            "bridge_executes_ffmpeg": False,
            "bridge_executes_asr": False,
            "bridge_copies_sources": False,
            "automatic_render": False,
            "automatic_upload": False,
            "automatic_publish": False,
            "source_mutation": False,
            "human_approval_required": True,
        },
    }
    validate_mvp_v4_candidate_handoff(payload)
    payload["handoff_sha256"] = _sha256_json(payload)
    return payload


def validate_mvp_v4_candidate_handoff(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != MVP_V4_SCHEMA:
        raise ValueError("unsupported MVP handoff schema")
    source_project = payload.get("source_project")
    if not isinstance(source_project, Mapping) or source_project.get("source_paths_exported"):
        raise PersonalVlogBoundaryError("MVP v4 source paths must remain private")
    if not source_project.get("private_project_manifest_required_for_resolution"):
        raise PersonalVlogBoundaryError("MVP v4 must require the private project resolver")
    assets = payload.get("source_assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("MVP v4 requires source assets")
    known = set()
    for asset in assets:
        if not isinstance(asset, Mapping):
            raise ValueError("MVP v4 source asset must be an object")
        asset_id = str(asset.get("source_asset_id") or "")
        digest = str(asset.get("source_sha256") or "")
        if not asset_id or asset_id in known:
            raise ValueError("MVP v4 source asset IDs must be unique")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("MVP v4 source asset SHA-256 is invalid")
        resolver = asset.get("private_resolver")
        if not isinstance(resolver, Mapping) or resolver.get("source_path_exported"):
            raise PersonalVlogBoundaryError("MVP v4 source resolver leaked a path")
        known.add(asset_id)
    clips = payload.get("candidate_clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("MVP v4 requires candidate clips")
    for clip in clips:
        if not isinstance(clip, Mapping) or clip.get("source_asset_id") not in known:
            raise ValueError("MVP v4 clip references an unknown source asset")
        if clip.get("status") != "candidate_only" or not clip.get("review_required"):
            raise ValueError("MVP v4 clips must remain review-required candidates")
        media_type = str(clip.get("media_type") or "")
        if media_type == "image":
            hold = float(clip.get("hold_seconds") or 0)
            if hold < 2.0 or hold > 4.5:
                raise ValueError("MVP v4 photo hold must stay within 2–4.5 seconds")
        elif not isinstance(clip.get("source_range"), Mapping):
            raise ValueError("MVP v4 video/audio clips require a source range")
    operator = payload.get("operator_boundary")
    if not isinstance(operator, Mapping):
        raise ValueError("MVP v4 operator boundary is missing")
    forbidden = (
        "bridge_executes_ffmpeg",
        "bridge_executes_asr",
        "bridge_copies_sources",
        "automatic_render",
        "automatic_upload",
        "automatic_publish",
        "source_mutation",
    )
    if any(bool(operator.get(key)) for key in forbidden):
        raise PersonalVlogBoundaryError("MVP v4 bridge attempted an unauthorized action")
    if not operator.get("human_approval_required"):
        raise PersonalVlogBoundaryError("MVP v4 human approval cannot be disabled")
    _reject_private_fields(payload)


def _reject_private_fields(value: Any) -> None:
    forbidden = {
        "source_path",
        "event_root",
        "project_dir",
        "latitude",
        "longitude",
        "gps",
        "appearance_vector",
        "face_embedding",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in forbidden:
                raise PersonalVlogBoundaryError(f"MVP v4 forbidden field:{key}")
            _reject_private_fields(child)
    elif isinstance(value, list):
        for child in value:
            _reject_private_fields(child)


def _sha256_json(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
