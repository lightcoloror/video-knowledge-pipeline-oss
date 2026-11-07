"""Source-bound media capabilities shared by preparation and downstream gates.

Only a successful local stream probe can exempt a source from visual work.
Missing frames, file extensions, failed probes and old report labels cannot.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Any

from .file_hash import sha256_file
from .models import now_iso
from .storage import read_json, write_json
from .video import _probe_with_ffprobe

SCHEMA = "video_knowledge_pipeline.media_applicability.v1"
VISUAL_CHANNELS = {
    "screen_text", "visual_frames", "structured_visual", "visual_route",
    "semantic_frame_understanding", "temporal_visual_understanding",
    "panorama_spherical_evidence",
}
VISUAL_ISSUES = {
    "missing_frame", "missing_visual_text", "missing_ocr", "low_ocr_confidence",
    "ocr_text_empty", "ocr_text_low_information", "ocr_wrapper_only",
    "screen_text_low_confidence", "structured_visual_without_frame",
    "structured_visual_without_ocr", "structured_visual_without_structure",
    "keep_image_without_frame", "missing_visual_route", "missing_visual_understanding",
    "semantic_frame_without_analysis", "temporal_sequence_without_analysis",
}


@lru_cache(maxsize=32)
def _probe_bound_bytes(path: str, digest: str) -> dict[str, Any]:
    metadata = _probe_with_ffprobe(Path(path))
    if sha256_file(Path(path)) != digest:
        raise ValueError("source_changed_during_media_probe")
    return metadata


def inspect_media_source(path: str | Path, *, expected_sha256: str = "") -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    row: dict[str, Any] = {"path": str(source), "status": "unknown", "source_bound": False}
    try:
        digest = sha256_file(source)
        row.update(sha256=digest, bytes=source.stat().st_size)
        if expected_sha256 and digest != expected_sha256.lower():
            raise ValueError("source_sha256_mismatch")
        probe = _probe_bound_bytes(str(source), digest)
        audio = probe.get("audio_stream_count")
        video = probe.get("video_stream_count")
        if type(audio) is not int or type(video) is not int or audio < 0 or video < 0:
            raise ValueError("stream_counts_not_verified")
        if not audio and not video:
            raise ValueError("no_audio_or_video_stream")
        row.update(
            status="verified", source_bound=True,
            media_kind="audio_only" if audio and not video else "video",
            audio_stream_count=audio, video_stream_count=video,
            attached_picture_count=probe.get("attached_picture_count", 0),
            duration_seconds=probe.get("duration_seconds", 0), probe="ffprobe",
        )
    except (OSError, ValueError, RuntimeError, TimeoutExpired) as exc:
        row["error"] = str(exc)
    return row


def resolve_media_applicability(
    bundle_dir: str | Path | None,
    manifest: dict[str, Any] | None = None,
    timeline: list[dict[str, Any]] | None = None,
    *, write: bool = False,
) -> dict[str, Any]:
    root = Path(bundle_dir).resolve() if bundle_dir is not None else None
    if manifest is None:
        data = read_json(root / "manifest.json") if root and (root / "manifest.json").is_file() else {}
        manifest = data if isinstance(data, dict) else {}
    if timeline is None:
        data = read_json(root / "timeline.json") if root and (root / "timeline.json").is_file() else []
        timeline = [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []
    raw_sources = manifest.get("sources")
    sources: list[dict[str, Any]] = []
    for source in raw_sources if isinstance(raw_sources, list) else []:
        if not isinstance(source, dict) or not str(source.get("path") or "").strip():
            sources.append({"status": "unknown", "source_bound": False, "error": "source_path_missing"})
            continue
        path = Path(str(source["path"])).expanduser()
        if not path.is_absolute():
            if root is None:
                sources.append({"status": "unknown", "source_bound": False, "error": "source_base_missing"})
                continue
            path = root / path
        record = inspect_media_source(path, expected_sha256=str(source.get("sha256") or ""))
        # Legacy source IDs contain the original media hash prefix. Retain this
        # binding when upgrading a bundle that predates explicit capabilities.
        source_id = str(source.get("video_id") or "")
        prefix = source_id.removeprefix("video_")
        if source_id.startswith("video_") and len(prefix) == 12 and all(c in "0123456789abcdef" for c in prefix):
            if record.get("sha256", "")[:12] != prefix:
                record.update(status="unknown", source_bound=False, error="source_id_hash_mismatch")
        sources.append(record)
    verified = bool(sources) and all(row.get("source_bound") is True for row in sources)
    audio_only = verified and all(row.get("media_kind") == "audio_only" for row in sources)
    supplemental = bool(
        manifest.get("companion_courseware_text") or manifest.get("companion_courseware")
        or manifest.get("supplementary_visual_materials") or manifest.get("visual_materials")
        or manifest.get("assets")
        or any(row.get("frame_paths") or row.get("asset_paths") or row.get("assets")
               or row.get("crop_paths") or row.get("visual_text")
               or row.get("structured_visual") or row.get("panorama_visual_understanding") for row in timeline)
    )
    not_applicable = bool(audio_only and not supplemental)
    result = {
        "schema": SCHEMA, "status": "verified" if verified else "unknown",
        "media_kind": "audio_only" if audio_only else ("video_or_mixed" if verified else "unknown"),
        "source_bound": verified, "sources": sources,
        "supplementary_visual_materials": supplemental,
        "visual_status": "not_applicable" if not_applicable else "required",
        "visual_required": not not_applicable,
        "reason": "verified_audio_only_without_visual_materials" if not_applicable else
                  ("supplementary_visual_materials_present" if audio_only else "video_or_unverified_media"),
        "checked_at": now_iso(),
    }
    if write:
        if root is None:
            raise ValueError("bundle_dir is required to persist media applicability")
        write_json(root / "media-applicability.json", result)
    return result


def audio_only_ocr_status(applicability: dict[str, Any]) -> dict[str, Any] | None:
    """Expose N/A without inventing execution counts for the video workflow."""
    if applicability.get("visual_status") != "not_applicable":
        return None
    return {
        "status": "not_applicable", "applicable": False,
        "scope": "verified_audio_only", "full_video_page_coverage_verified": None,
        "summary": {**{key: 0 for key in (
            "required", "executed", "completed", "existing_evidence", "human_confirmed_evidence",
            "failed", "blocked", "not_run", "empty", "usable_text", "suspicious", "remaining")},
            "execution_complete": None},
        "sampling_audit": {"status": "not_applicable", "full_page_coverage_verified": None},
    }


def project_timeline_for_media(
    timeline: list[dict[str, Any]], applicability: dict[str, Any],
) -> list[dict[str, Any]]:
    """Filter inapplicable *derived* obligations; never rewrite raw evidence."""
    if applicability.get("visual_status") != "not_applicable":
        return timeline
    return [
        {**row, "visual_applicability": "not_applicable", "needs_keyframe": False,
         "visual_route": "", "material_types": [t for t in row.get("material_types", []) if t not in {"table", "formula", "code", "image"}],
         "quality_issues": [issue for issue in row.get("quality_issues", []) if issue not in VISUAL_ISSUES]}
        for row in timeline
    ]
