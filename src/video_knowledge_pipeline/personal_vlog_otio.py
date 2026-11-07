from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Mapping

from .personal_vlog import PersonalVlogBoundaryError


def opentimelineio_status() -> dict[str, Any]:
    available = importlib.util.find_spec("opentimelineio") is not None
    return {
        "available": available,
        "backend": "OpenTimelineIO official Python API",
        "serialization_fallback": False,
        "status": "ready" if available else "degraded",
        "review_gap": None
        if available
        else "opentimelineio_not_installed; JSON recipe remains authoritative",
    }


def export_recipe_otio(
    recipe: Mapping[str, Any],
    output_path: str | Path,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Export through the official OTIO API or return an explicit degraded receipt."""

    status = opentimelineio_status()
    target = Path(output_path).expanduser().resolve()
    if not status["available"]:
        return {
            **status,
            "output_path": "",
            "recipe_sha256": recipe.get("recipe_sha256"),
            "write_requested": bool(write),
            "changed": False,
        }
    import opentimelineio as otio

    timeline = otio.schema.Timeline(
        name=f"VKP personal Vlog {recipe.get('preset')}",
        metadata={
            "vkp_schema": recipe.get("schema"),
            "project_id": recipe.get("project_id"),
            "recipe_sha256": recipe.get("recipe_sha256"),
            "preset": recipe.get("preset"),
            "source_paths_exported": False,
            "precise_gps_exported": False,
        },
    )
    video_track = otio.schema.Track(name="VKP candidate picture", kind=otio.schema.TrackKind.Video)
    audio_track = otio.schema.Track(name="VKP candidate audio", kind=otio.schema.TrackKind.Audio)
    rate = 24.0
    expected: list[tuple[str, str]] = []
    for row in recipe.get("clips") or []:
        if not isinstance(row, Mapping):
            continue
        asset_id = str(row.get("source_asset_id") or "")
        if not asset_id or any(token in asset_id for token in ("/", "\\", ":")):
            raise PersonalVlogBoundaryError("invalid OTIO source asset ID")
        duration = float(row.get("timeline_duration_seconds") or 0)
        if duration <= 0:
            raise ValueError("OTIO clip duration must be positive")
        source_range = row.get("source_range") if isinstance(row.get("source_range"), Mapping) else {}
        source_start = float(source_range.get("start_seconds") or 0)
        clip = otio.schema.Clip(
            name=str(row.get("clip_id") or asset_id),
            media_reference=otio.schema.ExternalReference(
                target_url=f"vkp-asset://{asset_id}",
                metadata={
                    "source_asset_id": asset_id,
                    "private_project_resolver_required": True,
                },
            ),
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(source_start * rate, rate),
                duration=otio.opentime.RationalTime(duration * rate, rate),
            ),
            metadata={
                "source_asset_id": asset_id,
                "story_role": row.get("story_role"),
                "recipe_sha256": recipe.get("recipe_sha256"),
                "candidate_only": True,
                "review_required": True,
            },
        )
        kind = str(row.get("media_type") or "")
        (audio_track if kind == "audio" else video_track).append(clip)
        expected.append((str(clip.name), asset_id))
    if len(video_track):
        timeline.tracks.append(video_track)
    if len(audio_track):
        timeline.tracks.append(audio_track)
    if not write:
        return {
            **status,
            "output_path": str(target),
            "recipe_sha256": recipe.get("recipe_sha256"),
            "clip_count": len(expected),
            "changed": False,
            "timeline": timeline,
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    otio.adapters.write_to_file(timeline, str(target))
    restored = otio.adapters.read_from_file(str(target))
    actual = [
        (str(clip.name), str(clip.metadata.get("source_asset_id") or ""))
        for track in restored.tracks
        for clip in track
        if isinstance(clip, otio.schema.Clip)
    ]
    if actual != expected:
        raise PersonalVlogBoundaryError("OTIO round-trip identity mismatch")
    return {
        **status,
        "output_path": str(target),
        "recipe_sha256": recipe.get("recipe_sha256"),
        "clip_count": len(expected),
        "round_trip_verified": True,
        "changed": True,
    }
