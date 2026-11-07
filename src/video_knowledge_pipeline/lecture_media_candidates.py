from __future__ import annotations

from pathlib import Path
from typing import Any

from .video import probe_video


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}


def find_lecture_media_candidates(
    roots: list[str | Path],
    *,
    min_duration_seconds: float = 60.0,
    max_duration_seconds: float = 300.0,
    limit: int = 20,
    max_files: int = 500,
    include_rejected: bool = False,
) -> dict[str, Any]:
    """Find local video files that are plausible strict real-lecture smoke inputs."""
    normalized_roots = [Path(root).expanduser().resolve() for root in roots]
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    scanned_files = 0
    probe_errors = 0

    for root in normalized_roots:
        for path in _iter_video_files(root):
            if scanned_files >= max_files:
                break
            scanned_files += 1
            try:
                metadata = probe_video(path)
            except Exception as exc:  # pragma: no cover - exact ffprobe failures vary by host media.
                probe_errors += 1
                if include_rejected:
                    rejected.append({"path": str(path), "reason": "probe_failed", "error": str(exc)})
                continue

            row = {
                "path": metadata.path,
                "title": metadata.title,
                "duration_seconds": metadata.duration_seconds,
                "width": metadata.width,
                "height": metadata.height,
                "fps": metadata.fps,
                "sha256": metadata.sha256,
                "recommended_command": _strict_smoke_command(metadata.path),
            }
            reason = _candidate_rejection_reason(
                duration_seconds=metadata.duration_seconds,
                min_duration_seconds=min_duration_seconds,
                max_duration_seconds=max_duration_seconds,
            )
            if reason:
                if include_rejected:
                    row["reason"] = reason
                    rejected.append(row)
                continue
            candidates.append(row)
            if len(candidates) >= limit:
                break
        if scanned_files >= max_files or len(candidates) >= limit:
            break

    return {
        "ok": bool(candidates),
        "roots": [str(root) for root in normalized_roots],
        "criteria": {
            "min_duration_seconds": min_duration_seconds,
            "max_duration_seconds": max_duration_seconds,
            "limit": limit,
            "max_files": max_files,
        },
        "scanned_files": scanned_files,
        "probe_errors": probe_errors,
        "count": len(candidates),
        "candidates": candidates,
        "rejected_count": len(rejected),
        "rejected": rejected if include_rejected else [],
    }


def _iter_video_files(root: Path):
    if root.is_file():
        if root.suffix.lower() in VIDEO_EXTENSIONS:
            yield root
        return
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def _candidate_rejection_reason(
    *,
    duration_seconds: float,
    min_duration_seconds: float,
    max_duration_seconds: float,
) -> str:
    if min_duration_seconds > 0 and duration_seconds < min_duration_seconds:
        return "too_short"
    if max_duration_seconds > 0 and duration_seconds > max_duration_seconds:
        return "too_long"
    return ""


def _strict_smoke_command(media_path: str) -> str:
    return (
        ".\\scripts\\video-knowledge.ps1 smoke-lecture-e2e "
        "--strict-real-lecture --extractor peepshow --asr-mode always "
        f"--media \"{media_path}\" --work-dir .\\tmp\\lesson-e2e-real --keep"
    )

