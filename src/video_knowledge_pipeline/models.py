from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass
class ResearchProject:
    id: str
    question: str
    created_at: str
    status: str = "open"
    notes: str = ""


@dataclass
class SourceRecord:
    id: str
    source_type: str
    title: str
    url: str = ""
    path: str = ""
    quality: str = "unknown"
    notes: str = ""
    created_at: str = field(default_factory=now_iso)


@dataclass
class SourceReview:
    id: str
    source_id: str
    quality: str
    reason: str = ""
    reviewer: str = "human"
    created_at: str = field(default_factory=now_iso)


@dataclass
class ClaimRecord:
    id: str
    text: str
    status: str = "open"
    notes: str = ""
    created_at: str = field(default_factory=now_iso)


@dataclass
class EvidenceRecord:
    id: str
    claim_id: str
    source_id: str
    quote_or_note: str
    relation: str = "unknown"
    status: str = "candidate"
    confidence: str = "unknown"
    notes: str = ""
    created_at: str = field(default_factory=now_iso)


@dataclass
class VideoMetadata:
    id: str
    path: str
    title: str
    duration_seconds: float
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    sha256: str = ""
    created_at: str = field(default_factory=now_iso)


@dataclass
class TranscriptCue:
    start: float
    end: float
    text: str
    segment_id: str = ""
    source_segment_ids: list[str] = field(default_factory=list)
    transformations: list[dict[str, Any]] = field(default_factory=list)
    # Intent: keep upstream diarization evidence available to every downstream
    # transcript stage.
    # Decision: extend the existing cue contract with optional fields instead of
    # introducing a second speaker/timeline schema.
    # Reason: MOSS already emits start/end/text/speaker, but VKP previously
    # discarded speaker identity after normalization.
    # Evidence: fixed upstream MOSS commit
    # eda4b9f13f1574765a80438c9797780a9bd48112.
    # Effective scope: backward-compatible transcript metadata; defaults preserve
    # all existing single-speaker and legacy callers.
    speaker: str = ""
    speaker_role: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceSegment:
    id: str
    video_id: str
    start: float
    end: float
    midpoint: float
    signals: list[str]
    transcript_excerpt: str = ""
    frame_paths: list[str] = field(default_factory=list)
    visual_observation: str = ""
    uncertainty: str = "visual content requires human review"
    status: str = "candidate"
    needs_human_review: bool = True


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)


def project_paths(root: str | Path) -> dict[str, Path]:
    base = Path(root)
    return {
        "root": base,
        "project": base / "research.json",
        "sources": base / "sources.jsonl",
        "source_reviews": base / "source-reviews.jsonl",
        "claims": base / "claims.jsonl",
        "evidence": base / "evidence.jsonl",
        "videos": base / "videos",
        "transcripts": base / "transcripts",
        "notes": base / "notes",
        "lecture_packages": base / "lecture-packages",
        "graph": base / "graph-candidates.jsonl",
        "reviews": base / "reviews.jsonl",
    }
