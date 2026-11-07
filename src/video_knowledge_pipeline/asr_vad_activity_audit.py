from __future__ import annotations

from pathlib import Path
from typing import Any

from .asr_vad_chunking import read_vad_intervals
from .audio_silence_probe import (
    build_audio_silence_probe_command,
    probe_audio_silence,
)
from .interval_coverage import interval_coverage
from .models import now_iso
from .storage import write_json
from .video import probe_video, sha256_file


SCHEMA = "video_knowledge_pipeline.asr_vad_activity_audit.v1"


def audit_asr_vad_audio_activity(
    media_path: str | Path,
    vad_json: str | Path,
    *,
    output_path: str | Path | None = None,
    duration_seconds: float | None = None,
    noise_db: float = -45.0,
    minimum_silence_seconds: float = 0.5,
    minimum_uncovered_seconds: float = 2.0,
    execute: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    """Compare FunASR VAD against FFmpeg non-silent audio as candidate evidence."""

    media = Path(media_path).expanduser().resolve()
    vad_path = Path(vad_json).expanduser().resolve()
    if not media.is_file():
        raise FileNotFoundError(f"media not found: {media}")
    vad_intervals = read_vad_intervals(vad_path)
    vad_end = max(float(row["end"]) for row in vad_intervals)
    requested_duration = float(duration_seconds or 0.0)
    if requested_duration < 0:
        raise ValueError("duration_seconds must not be negative")
    duration = requested_duration or vad_end
    duration_source = "explicit" if requested_duration else "vad_max_end_preview"
    if execute and not requested_duration:
        duration = float(probe_video(media).duration_seconds)
        duration_source = "ffprobe"
    if duration <= 0:
        raise ValueError("media duration must be positive")
    target = (
        Path(output_path).expanduser().resolve()
        if output_path
        else vad_path.with_name("asr-vad-activity-audit.json")
    )
    command = build_audio_silence_probe_command(
        media,
        start=0.0,
        end=duration,
        noise_db=noise_db,
        minimum_silence_seconds=minimum_silence_seconds,
    )
    source_media = {
        "path": str(media),
        "bytes": media.stat().st_size,
        "sha256": sha256_file(media),
    }
    base: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "planned",
        "ok": True,
        "execute": bool(execute),
        "write": bool(write),
        "source_media": source_media,
        "vad_json": str(vad_path),
        "vad_sha256": sha256_file(vad_path),
        "vad_interval_count": len(vad_intervals),
        "duration_seconds": round(duration, 6),
        "duration_source": duration_source,
        "settings": {
            "noise_db": float(noise_db),
            "minimum_silence_seconds": float(minimum_silence_seconds),
            "minimum_uncovered_seconds": float(minimum_uncovered_seconds),
        },
        "probe_command": command,
        "candidate_gap_count": 0,
        "candidate_gaps": [],
        "network_call": False,
        "operator_boundary": {
            "ffmpeg_local_only": True,
            "non_silent_audio_is_not_proven_speech": True,
            "candidate_evidence_only": True,
            "vad_segments_modified": False,
            "chunk_manifest_modified": False,
            "automatic_remote_retry": False,
            "automatic_fallback": False,
        },
        "output_path": str(target),
        "updated_at": now_iso(),
    }
    if not execute:
        base["limitations"] = [
            "preview does not run FFmpeg",
            "duration ends at the final VAD interval unless duration_seconds is supplied",
        ]
        if write:
            write_json(target, base)
        return base

    probe = probe_audio_silence(
        media,
        start=0.0,
        end=duration,
        noise_db=noise_db,
        minimum_silence_seconds=minimum_silence_seconds,
        timeout_seconds=max(90, int(duration / 4) + 30),
    )
    base["audio_probe"] = probe
    if not probe.get("ok"):
        base.update({"status": "failed", "ok": False})
        if write:
            write_json(target, base)
        return base
    activity = [
        (float(row["start"]), float(row["end"]))
        for row in probe.get("activity_intervals") or []
        if isinstance(row, dict)
    ]
    coverage = interval_coverage(
        activity,
        [(float(row["start"]), float(row["end"])) for row in vad_intervals],
        minimum_gap_seconds=minimum_uncovered_seconds,
    )
    gaps = [
        {
            "candidate_id": f"audio-activity-gap-{position:04d}",
            **row,
            "reason": "non_silent_audio_without_vad_coverage",
            "evidence_type": "candidate_audio_activity",
            "candidate_only": True,
            "needs_human_or_speech_vad_confirmation": True,
        }
        for position, row in enumerate(coverage["gaps"], start=1)
    ]
    status = "review_required" if gaps else "passed"
    base.update(
        {
            "status": status,
            "ok": True,
            "vad_coverage_verified": not gaps,
            "activity_coverage": coverage,
            "candidate_gap_count": len(gaps),
            "candidate_gaps": gaps,
            "limitations": [
                "FFmpeg non-silent audio may be music, noise, or effects rather than speech",
                "candidate gaps must not be added to VAD or uploaded automatically",
            ],
            "recommended_action": "confirm candidate gaps with a speech-specific VAD or human review",
            "updated_at": now_iso(),
        }
    )
    if write:
        write_json(target, base)
    return base
