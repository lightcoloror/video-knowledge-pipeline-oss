from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

from .file_hash import sha256_file
from .media_tools import resolve_media_tool
from .models import EvidenceSegment, TranscriptCue, VideoMetadata, new_id
from .transcript import transcript_excerpt


_END_SEEK_MARGIN_SECONDS = 0.1


def probe_video(path: str | Path) -> VideoMetadata:
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    sha = sha256_file(video_path)
    metadata = _probe_with_ffprobe(video_path)
    return VideoMetadata(
        id=f"video_{sha[:12]}",
        path=str(video_path.resolve()),
        title=video_path.stem,
        duration_seconds=float(metadata.get("duration_seconds") or 0),
        width=metadata.get("width"),
        height=metadata.get("height"),
        fps=metadata.get("fps"),
        sha256=sha,
    )


def fixed_timepoints(duration: float, interval: float, max_points: int) -> list[tuple[float, str]]:
    if max_points <= 0:
        return []
    if duration <= 0:
        return [(0.0, "fixed_interval")]
    if max_points == 1:
        return [(0.0, "fixed_interval")]
    # FFmpeg can return success without emitting a frame when seeking to the
    # container duration exactly. Keep the final sampling point inside the
    # decodable interval instead of creating an EOF-only evidence segment.
    end_margin = min(_END_SEEK_MARGIN_SECONDS, duration / 2)
    last_seek_time = max(0.0, duration - end_margin)
    interval = max(1.0, interval)
    points = []
    current = 0.0
    while current <= last_seek_time + 0.001 and len(points) < max_points:
        points.append((min(current, last_seek_time), "fixed_interval"))
        current += interval
    if points and points[-1][0] < last_seek_time and len(points) < max_points:
        points.append((last_seek_time, "fixed_interval"))
    return points


def scene_change_timepoints(video_path: str | Path, threshold: float = 0.35, max_points: int = 120) -> list[tuple[float, str]]:
    ffmpeg = resolve_media_tool("ffmpeg")
    if not ffmpeg:
        return []
    with tempfile.TemporaryDirectory() as temp_dir:
        scene_pattern = Path(temp_dir) / "scene-%04d.jpg"
        command = [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(video_path),
            "-vf",
            f"select='gt(scene,{threshold})',showinfo",
            "-vsync",
            "vfr",
            "-frames:v",
            str(max_points),
            str(scene_pattern),
        ]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        return []
    return [(time, "scene_change") for time in parse_showinfo_scene_times(result.stderr)[:max_points]]


def parse_showinfo_scene_times(stderr: str) -> list[float]:
    times: list[float] = []
    for match in re.finditer(r"pts_time:(?P<time>\d+(?:\.\d+)?)", stderr):
        value = float(match.group("time"))
        if not times or value != times[-1]:
            times.append(value)
    return times


def merge_timepoints(points: list[tuple[float, str]], window_seconds: float, duration: float, max_segments: int) -> list[tuple[float, list[str]]]:
    if not points:
        return []
    normalized = sorted((max(0.0, min(duration, point)), signal) for point, signal in points)
    merged: list[tuple[float, list[str]]] = []
    for point, signal in normalized:
        if not merged or abs(point - merged[-1][0]) > window_seconds:
            merged.append((point, [signal]))
            continue
        existing_point, signals = merged[-1]
        if signal not in signals:
            signals.append(signal)
        merged[-1] = ((existing_point + point) / 2, signals)
    return merged[:max_segments]


def build_segments(
    *,
    video_id: str,
    duration: float,
    timepoints: list[tuple[float, list[str]]],
    cues: list[TranscriptCue],
    window_seconds: float = 8.0,
) -> list[EvidenceSegment]:
    segments = []
    for midpoint, signals in timepoints:
        start = max(0.0, midpoint - window_seconds / 2)
        end = min(duration, midpoint + window_seconds / 2) if duration > 0 else midpoint + window_seconds / 2
        segments.append(
            EvidenceSegment(
                id=new_id("segment"),
                video_id=video_id,
                start=start,
                end=end,
                midpoint=midpoint,
                signals=signals,
                transcript_excerpt=transcript_excerpt(cues, start, end),
            )
        )
    return segments


def extract_segment_frames(video_path: str | Path, output_dir: str | Path, segments: list[EvidenceSegment]) -> None:
    ffmpeg = resolve_media_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found; set LECTURE_FFMPEG_DIR or FFMPEG_BINARY")
    frame_dir = Path(output_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    for index, segment in enumerate(segments, start=1):
        target = frame_dir / f"{index:03d}_{_timestamp_for_filename(segment.midpoint)}.jpg"
        command = [
            ffmpeg,
            "-y",
            "-ss",
            f"{segment.midpoint:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-pix_fmt",
            "yuvj420p",
            "-strict",
            "unofficial",
            "-q:v",
            "2",
            str(target),
        ]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"ffmpeg failed for {target}")
        if not target.is_file() or target.stat().st_size <= 0:
            raise RuntimeError(
                f"ffmpeg produced no frame for {target} at {segment.midpoint:.3f}s; "
                "the sampling point may be outside the decodable interval"
            )
        segment.frame_paths.append(str(target))


def _probe_with_ffprobe(path: Path) -> dict:
    ffprobe = resolve_media_tool("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe was not found; set LECTURE_FFMPEG_DIR or FFPROBE_BINARY")
    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    if not result.stdout:
        raise RuntimeError("ffprobe returned no JSON output")
    data = json.loads(result.stdout)
    if not isinstance(data, dict) or not isinstance(data.get("streams"), list):
        raise ValueError("ffprobe returned no valid streams inventory")
    streams = data["streams"]
    if any(not isinstance(stream, dict) or stream.get("codec_type") not in
           {"audio", "video", "subtitle", "data", "attachment"} for stream in streams):
        raise ValueError("ffprobe returned an invalid stream entry")
    if any(not isinstance(stream.get("disposition", {}), dict) for stream in streams):
        raise ValueError("ffprobe returned invalid stream disposition")
    if any(type(stream.get('disposition', {}).get('attached_pic', 0)) not in (int, bool)
           or stream.get('disposition', {}).get('attached_pic', 0) not in (0, 1)
           for stream in streams):
        raise ValueError('ffprobe returned an invalid attached-picture flag')
    visual_streams = [stream for stream in streams if stream.get("codec_type") == "video"
                      and not (stream.get("disposition") or {}).get("attached_pic")]
    video_stream = next(iter(visual_streams), {})
    # Container duration commonly follows the longest audio stream. When
    # audio extends beyond video, that value is valid for the container but
    # not decodable for frame extraction. Prefer the video stream bound.
    duration = video_stream.get("duration") or data.get("format", {}).get("duration") or 0
    return {
        "audio_stream_count": sum(stream.get("codec_type") == "audio" for stream in streams),
        "video_stream_count": len(visual_streams),
        "attached_picture_count": sum(bool((stream.get("disposition") or {}).get("attached_pic")) for stream in streams),
        "duration_seconds": float(duration),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "fps": _parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")),
    }


def _parse_fps(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            return None
        return float(numerator) / denominator_value
    return float(value)


def _timestamp_for_filename(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    return f"{millis:010d}ms"
