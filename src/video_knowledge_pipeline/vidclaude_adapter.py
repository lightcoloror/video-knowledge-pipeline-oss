from __future__ import annotations

import json
import re
from pathlib import Path

from .models import EvidenceSegment, TranscriptCue, VideoMetadata, dataclass_to_dict, new_id
from .orchestrator import graph_candidates_for_video, render_video_evidence_card
from .source_artifacts import collect_source_artifacts
from .storage import append_jsonl, ensure_project_dirs, write_json
from .transcript import transcript_excerpt
from .video import probe_video


def import_vidclaude_cache(root: str | Path, cache_dir: str | Path, *, topic: str) -> dict:
    """Import a vidclaude .vidcache/<id> folder as research video evidence."""
    paths = ensure_project_dirs(root)
    cache = Path(cache_dir)
    meta_path = cache / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"vidclaude cache missing meta.json: {cache}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    video_path = Path(str(meta.get("path") or meta.get("video_path") or meta.get("source") or ""))
    metadata = _metadata_from_meta(video_path, meta)

    transcript = _read_vidclaude_transcript(cache / "transcript.json")
    timeline = _read_json_list(cache / "timeline.json")
    evidence_md = cache / "evidence.md"
    frames = _frame_paths_from_evidence(evidence_md)
    if not frames:
        frames = sorted((cache / "frames").glob("*.jpg"))
    frame_signals = _frame_signals_from_timeline(timeline)
    segments = _segments_from_vidclaude_frames(metadata.id, metadata.duration_seconds, frames, transcript, frame_signals, timeline)

    video_dir = paths["videos"] / metadata.id
    video_dir.mkdir(parents=True, exist_ok=True)
    write_json(video_dir / "metadata.json", dataclass_to_dict(metadata))
    write_json(video_dir / "segments.json", [dataclass_to_dict(segment) for segment in segments])

    copied_evidence_path = ""
    if evidence_md.exists():
        copied_evidence = paths["notes"] / f"{metadata.id}-vidclaude-evidence.md"
        copied_evidence.write_text(evidence_md.read_text(encoding="utf-8"), encoding="utf-8")
        copied_evidence_path = str(copied_evidence)
    source_artifacts = collect_source_artifacts("vidclaude", cache, copied_paths={"evidence": copied_evidence_path})
    source_artifacts_path = video_dir / "source-artifacts.json"
    write_json(source_artifacts_path, source_artifacts)

    card = render_video_evidence_card(
        metadata=dataclass_to_dict(metadata),
        segments=[dataclass_to_dict(segment) for segment in segments],
        topic=topic,
    )
    card_path = paths["notes"] / f"{metadata.id}-vidclaude-video-evidence-card.md"
    card_path.write_text(card, encoding="utf-8")

    graph_rows = graph_candidates_for_video(
        metadata=dataclass_to_dict(metadata),
        segments=[dataclass_to_dict(segment) for segment in segments],
        topic=topic,
    )
    append_jsonl(paths["graph"], graph_rows)
    return {
        "video_id": metadata.id,
        "imported_from": str(cache),
        "segment_count": len(segments),
        "card_path": str(card_path),
        "copied_evidence_path": copied_evidence_path,
        "source_artifacts_path": str(source_artifacts_path),
        "source_artifact_count": source_artifacts["available_count"],
        "segments_path": str(video_dir / "segments.json"),
        "graph_path": str(paths["graph"]),
    }


def _read_vidclaude_transcript(path: Path) -> list[TranscriptCue]:
    cues = []
    for chunk in _read_json_list(path):
        text = str(chunk.get("text", "")).strip()
        if text:
            start = _timestamp_seconds(chunk, "start")
            end = _timestamp_seconds(chunk, "end", default=start)
            cues.append(
                TranscriptCue(
                    start=start,
                    end=end,
                    text=text,
                )
            )
    return cues


def _metadata_from_meta(video_path: Path, meta: dict) -> VideoMetadata:
    if str(video_path):
        try:
            return probe_video(video_path)
        except Exception:
            pass
    resolution = meta.get("resolution")
    width = None
    height = None
    if isinstance(resolution, list) and len(resolution) >= 2:
        width = _optional_int(resolution[0])
        height = _optional_int(resolution[1])
    elif isinstance(resolution, dict):
        width = _optional_int(resolution.get("width"))
        height = _optional_int(resolution.get("height"))
    duration = _optional_float(meta.get("duration_sec"))
    if duration is None:
        duration = _optional_float(meta.get("durationSeconds"))
    if duration is None:
        duration = _optional_float(meta.get("duration"))
    return VideoMetadata(
        id=new_id("video"),
        path=str(video_path),
        title=str(meta.get("title") or (video_path.name if str(video_path) else "vidclaude-video")),
        duration_seconds=duration or 0.0,
        width=width,
        height=height,
        fps=_optional_float(meta.get("fps")),
    )


def _segments_from_vidclaude_frames(
    video_id: str,
    duration: float,
    frames: list[Path],
    transcript: list[TranscriptCue],
    frame_signals: dict[str, list[str]],
    timeline: list[dict],
) -> list[EvidenceSegment]:
    segments = []
    for frame in frames:
        frame_id, midpoint = _parse_vidclaude_frame_name(frame)
        start = max(0.0, midpoint - 0.5)
        end = min(duration, midpoint + 0.5) if duration else midpoint + 0.5
        signals = _signals_for_frame(frame_id, start, end, frame_signals, timeline)
        segments.append(
            EvidenceSegment(
                id=new_id("segment"),
                video_id=video_id,
                start=start,
                end=end,
                midpoint=midpoint,
                signals=signals,
                transcript_excerpt=transcript_excerpt(transcript, start, end),
                frame_paths=[str(frame)],
                uncertainty="imported from vidclaude; visual content requires human review",
            )
        )
    return segments


def _frame_signals_from_timeline(timeline: list[dict]) -> dict[str, list[str]]:
    signals: dict[str, list[str]] = {}
    for event in timeline:
        if not isinstance(event, dict):
            continue
        modality = _signal_token(str(event.get("modality") or "event"))
        summary = _signal_token(str(event.get("summary") or modality or "frame"))
        for source_id in event.get("source_ids", []):
            signals.setdefault(str(source_id), [])
            signal = f"vidclaude_{modality}_{summary}"
            if signal not in signals[str(source_id)]:
                signals[str(source_id)].append(signal)
    return signals


def _signals_for_frame(
    frame_id: str,
    start: float,
    end: float,
    frame_signals: dict[str, list[str]],
    timeline: list[dict],
) -> list[str]:
    signals = list(frame_signals.get(frame_id, []))
    for event in timeline:
        if not isinstance(event, dict):
            continue
        event_start = _timestamp_seconds(event, "start")
        event_end = _timestamp_seconds(event, "end", default=event_start)
        if event_end < start or event_start > end:
            continue
        modality = _signal_token(str(event.get("modality") or "event"))
        signal = f"vidclaude_{modality}"
        if signal not in signals:
            signals.append(signal)
    if "vidclaude_frame" not in signals:
        signals.insert(0, "vidclaude_frame")
    return signals


def _parse_vidclaude_frame_name(path: Path) -> tuple[str, float]:
    match = re.search(r"frame_(?P<index>\d+)_(?P<millis>\d+)ms?", path.stem)
    if not match:
        match = re.search(r"frame_(?P<index>\d+)_(?P<millis>\d+)", path.stem)
    if not match:
        return path.stem, 0.0
    return f"f_{int(match.group('index')):04d}", int(match.group("millis")) / 1000


def _read_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _frame_paths_from_evidence(path: Path) -> list[Path]:
    if not path.exists():
        return []
    paths = []
    for match in re.finditer(r"`(?P<path>[^`]+frame_\d+_\d+\.jpg)`", path.read_text(encoding="utf-8")):
        frame_path = Path(match.group("path"))
        if frame_path.exists() and frame_path not in paths:
            paths.append(frame_path)
    return paths


def _timestamp_seconds(data: dict, prefix: str, *, default: float = 0.0) -> float:
    for key in (f"{prefix}_ms", f"{prefix}Ms", f"{prefix}_millis", f"{prefix}Millis"):
        value = _optional_float(data.get(key))
        if value is not None:
            return value / 1000
    for key in (f"{prefix}_seconds", f"{prefix}Seconds", f"{prefix}_sec", f"{prefix}Sec", prefix):
        value = _optional_float(data.get(key))
        if value is not None:
            return value
    return default


def _signal_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return token or "event"


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
