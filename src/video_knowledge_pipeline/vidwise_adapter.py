from __future__ import annotations

import json
import re
from pathlib import Path

from .models import EvidenceSegment, VideoMetadata, dataclass_to_dict, new_id
from .orchestrator import graph_candidates_for_video, render_video_evidence_card
from .source_artifacts import collect_source_artifacts
from .storage import append_jsonl, ensure_project_dirs, write_json
from .transcript import TranscriptCue, transcript_excerpt
from .video import probe_video


def import_vidwise_output(root: str | Path, output_dir: str | Path, *, topic: str) -> dict:
    """Import an existing vidwise output folder as research video evidence."""
    paths = ensure_project_dirs(root)
    output = Path(output_dir)
    video_path = output / "video.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"vidwise output missing video.mp4: {output}")

    metadata = probe_video(video_path)
    cues = _read_vidwise_transcript(output / "transcript.json")
    frame_paths = sorted((output / "frames").glob("*.png"))
    segments = _segments_from_vidwise_frames(metadata, frame_paths, cues)

    video_dir = paths["videos"] / metadata.id
    video_dir.mkdir(parents=True, exist_ok=True)
    write_json(video_dir / "metadata.json", dataclass_to_dict(metadata))
    write_json(video_dir / "segments.json", [dataclass_to_dict(segment) for segment in segments])
    source_artifacts = collect_source_artifacts("vidwise", output)
    source_artifacts_path = video_dir / "source-artifacts.json"
    write_json(source_artifacts_path, source_artifacts)

    card = render_video_evidence_card(
        metadata=dataclass_to_dict(metadata),
        segments=[dataclass_to_dict(segment) for segment in segments],
        topic=topic,
    )
    card_path = paths["notes"] / f"{metadata.id}-vidwise-video-evidence-card.md"
    card_path.write_text(card, encoding="utf-8")

    graph_rows = graph_candidates_for_video(
        metadata=dataclass_to_dict(metadata),
        segments=[dataclass_to_dict(segment) for segment in segments],
        topic=topic,
    )
    append_jsonl(paths["graph"], graph_rows)
    return {
        "video_id": metadata.id,
        "imported_from": str(output),
        "segment_count": len(segments),
        "card_path": str(card_path),
        "source_artifacts_path": str(source_artifacts_path),
        "source_artifact_count": source_artifacts["available_count"],
        "segments_path": str(video_dir / "segments.json"),
        "graph_path": str(paths["graph"]),
    }


def _read_vidwise_transcript(path: Path) -> list[TranscriptCue]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    cues = []
    for segment in data.get("segments", []):
        text = str(segment.get("text", "")).strip()
        if text:
            cues.append(
                TranscriptCue(
                    start=float(segment.get("start", 0)),
                    end=float(segment.get("end", segment.get("start", 0))),
                    text=text,
                )
            )
    return cues


def _segments_from_vidwise_frames(metadata: VideoMetadata, frame_paths: list[Path], cues: list[TranscriptCue]) -> list[EvidenceSegment]:
    segments: list[EvidenceSegment] = []
    for frame in frame_paths:
        midpoint = _seconds_from_vidwise_frame_name(frame)
        start = max(0.0, midpoint - 0.5)
        end = min(metadata.duration_seconds, midpoint + 0.5) if metadata.duration_seconds else midpoint + 0.5
        segments.append(
            EvidenceSegment(
                id=new_id("segment"),
                video_id=metadata.id,
                start=start,
                end=end,
                midpoint=midpoint,
                signals=["vidwise_frame"],
                transcript_excerpt=transcript_excerpt(cues, start, end),
                frame_paths=[str(frame)],
                uncertainty="imported from vidwise; visual content requires human review",
            )
        )
    return segments


def _seconds_from_vidwise_frame_name(path: Path) -> float:
    match = re.search(r"frame_(?P<minutes>\d+)m(?P<seconds>\d+)s", path.stem)
    if not match:
        return 0.0
    return int(match.group("minutes")) * 60 + int(match.group("seconds"))
