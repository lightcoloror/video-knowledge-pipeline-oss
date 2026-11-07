from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from .interval_coverage import merge_intervals
from .media_tools import local_tool_subprocess_env, resolve_media_tool


SCHEMA = "video_knowledge_pipeline.audio_silence_probe.v1"
_EVENT_PATTERN = re.compile(r"silence_(start|end):\s*([0-9.]+)")


def build_audio_silence_probe_command(
    media_path: str | Path,
    *,
    start: float,
    end: float,
    noise_db: float = -45.0,
    minimum_silence_seconds: float = 0.5,
    ffmpeg_path: str | Path | None = None,
) -> list[str]:
    media = Path(media_path).expanduser().resolve()
    start_value, end_value = _window(start, end)
    if minimum_silence_seconds <= 0:
        raise ValueError("minimum_silence_seconds must be positive")
    ffmpeg = str(ffmpeg_path or resolve_media_tool("ffmpeg") or "ffmpeg")
    return [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-ss",
        f"{start_value:.6f}",
        "-t",
        f"{end_value - start_value:.6f}",
        "-i",
        str(media),
        "-vn",
        "-af",
        f"silencedetect=noise={float(noise_db):g}dB:d={float(minimum_silence_seconds):g}",
        "-f",
        "null",
        "-",
    ]


def parse_audio_silence_output(
    output: str,
    *,
    window_start: float,
    window_end: float,
) -> dict[str, Any]:
    start_value, end_value = _window(window_start, window_end)
    silence_rows: list[tuple[float, float]] = []
    active_start: float | None = None
    for match in _EVENT_PATTERN.finditer(str(output or "")):
        kind = match.group(1)
        value = min(end_value, max(start_value, start_value + float(match.group(2))))
        if kind == "start":
            active_start = value if active_start is None else min(active_start, value)
            continue
        interval_start = start_value if active_start is None else active_start
        if value > interval_start:
            silence_rows.append((interval_start, value))
        active_start = None
    if active_start is not None and end_value > active_start:
        silence_rows.append((active_start, end_value))
    silence_intervals = merge_intervals(silence_rows)
    activity_intervals = _complement_intervals(
        silence_intervals,
        start=start_value,
        end=end_value,
    )
    return {
        "silence_starts": [round(start, 6) for start, _ in silence_intervals],
        "silence_ends": [round(end, 6) for _, end in silence_intervals],
        "silence_intervals": [
            {
                "start": round(start, 6),
                "end": round(end, 6),
                "duration_seconds": round(end - start, 6),
            }
            for start, end in silence_intervals
        ],
        "activity_intervals": [
            {
                "start": round(start, 6),
                "end": round(end, 6),
                "duration_seconds": round(end - start, 6),
            }
            for start, end in activity_intervals
        ],
    }


def probe_audio_silence(
    media_path: str | Path,
    *,
    start: float,
    end: float,
    noise_db: float = -45.0,
    minimum_silence_seconds: float = 0.5,
    timeout_seconds: int = 90,
) -> dict[str, Any]:
    media = Path(media_path).expanduser().resolve()
    start_value, end_value = _window(start, end)
    if not media.is_file():
        return _failure(media, start_value, end_value, "media_not_found")
    ffmpeg = resolve_media_tool("ffmpeg")
    if not ffmpeg:
        return _failure(media, start_value, end_value, "ffmpeg_not_available")
    command = build_audio_silence_probe_command(
        media,
        start=start_value,
        end=end_value,
        noise_db=noise_db,
        minimum_silence_seconds=minimum_silence_seconds,
        ffmpeg_path=ffmpeg,
    )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1, int(timeout_seconds)),
            check=False,
            env=local_tool_subprocess_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result = _failure(
            media,
            start_value,
            end_value,
            f"{type(exc).__name__}: {exc}",
        )
        result["command"] = command
        return result
    combined = (completed.stderr or "") + "\n" + (completed.stdout or "")
    if completed.returncode != 0:
        result = _failure(media, start_value, end_value, "ffmpeg_failed")
        result.update(
            {
                "command": command,
                "returncode": completed.returncode,
                "stderr_tail": (completed.stderr or "")[-1000:],
            }
        )
        return result
    parsed = parse_audio_silence_output(
        combined,
        window_start=start_value,
        window_end=end_value,
    )
    return {
        "schema": SCHEMA,
        "ok": True,
        "status": "completed",
        "source": "ffmpeg_silencedetect",
        "media_path": str(media),
        "window": {"start": start_value, "end": end_value},
        "settings": {
            "noise_db": float(noise_db),
            "minimum_silence_seconds": float(minimum_silence_seconds),
        },
        "command": command,
        "returncode": completed.returncode,
        **parsed,
    }


def _complement_intervals(
    intervals: list[tuple[float, float]],
    *,
    start: float,
    end: float,
) -> list[tuple[float, float]]:
    activity: list[tuple[float, float]] = []
    cursor = start
    for silence_start, silence_end in intervals:
        if silence_start > cursor:
            activity.append((cursor, silence_start))
        cursor = max(cursor, silence_end)
    if cursor < end:
        activity.append((cursor, end))
    return activity


def _window(start: float, end: float) -> tuple[float, float]:
    start_value = max(0.0, float(start))
    end_value = float(end)
    if end_value <= start_value:
        raise ValueError("audio silence probe end must be greater than start")
    return start_value, end_value


def _failure(media: Path, start: float, end: float, status: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "ok": False,
        "status": status,
        "source": "ffmpeg_silencedetect",
        "media_path": str(media),
        "window": {"start": start, "end": end},
        "silence_intervals": [],
        "activity_intervals": [],
    }
