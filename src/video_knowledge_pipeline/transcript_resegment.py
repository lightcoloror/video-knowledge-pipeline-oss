from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .asr_adapter import render_srt
from .models import TranscriptCue, new_id, now_iso
from .storage import ensure_project_dirs, write_json
from .transcript import parse_transcript
from .video import probe_video


def resegment_transcript(
    workspace_dir: str | Path,
    input_path: str | Path,
    *,
    media_path: str | Path | None = None,
    duration_seconds: float = 0.0,
    target_seconds: float = 8.0,
    max_chars: int = 180,
    title: str = "",
) -> dict[str, Any]:
    """Split a coarse transcript into estimated timed cues for timeline alignment."""
    root = Path(workspace_dir).expanduser().resolve()
    source = Path(input_path).expanduser().resolve()
    cues = parse_transcript(source)
    source_data = _read_json_if_possible(source)
    duration = _duration(duration_seconds=duration_seconds, media_path=media_path, cues=cues)
    chunks = _split_cues(cues, max_chars=max(40, int(max_chars or 180)))
    if not chunks:
        raise ValueError(f"transcript contains no text: {source}")
    if duration <= 0:
        duration = max(float(target_seconds or 8.0), len(chunks) * float(target_seconds or 8.0))
    estimated = _estimate_timing(chunks, duration)

    transcript_id = new_id("transcript_reseg")
    output_dir = ensure_project_dirs(root)["transcripts"] / transcript_id
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": transcript_id,
        "title": title or _source_title(source_data) or source.stem,
        "provider": _source_provider(source_data),
        "source_path": str(source),
        "timing_estimated": True,
        "timing_estimation_method": "character_proportional_split",
        "duration_seconds": duration,
        "created_at": now_iso(),
        "segments": [{"start": cue.start, "end": cue.end, "text": cue.text} for cue in estimated],
    }
    json_path = output_dir / "normalized-transcript.json"
    srt_path = output_dir / "normalized-transcript.srt"
    write_json(json_path, payload)
    srt_path.write_text(render_srt(estimated), encoding="utf-8")
    report_path = output_dir / "resegment-report.md"
    report_path.write_text(_render_report(payload, json_path, srt_path), encoding="utf-8")
    return {
        "schema": "lecture_transcript_resegment.v1",
        "workspace_dir": str(root),
        "input_path": str(source),
        "json_path": str(json_path),
        "srt_path": str(srt_path),
        "report_path": str(report_path),
        "segment_count": len(estimated),
        "duration_seconds": duration,
        "timing_estimated": True,
    }


def _duration(*, duration_seconds: float, media_path: str | Path | None, cues: list[TranscriptCue]) -> float:
    if duration_seconds and duration_seconds > 0:
        return float(duration_seconds)
    if media_path:
        try:
            return float(probe_video(media_path).duration_seconds or 0.0)
        except Exception:
            pass
    ends = [cue.end for cue in cues if cue.end and cue.end > cue.start]
    return max(ends) if ends else 0.0


def _split_cues(cues: list[TranscriptCue], *, max_chars: int) -> list[str]:
    chunks: list[str] = []
    for cue in cues:
        text = str(cue.text or "").strip()
        if not text:
            continue
        chunks.extend(_split_text(text, max_chars=max_chars))
    return chunks


def _split_text(text: str, *, max_chars: int) -> list[str]:
    pieces = [piece.strip() for piece in re.split(r"(?<=[。！？!?；;])", text) if piece.strip()]
    if len(pieces) <= 1:
        pieces = [piece.strip() for piece in re.split(r"(?<=，|,|、|\s)", text) if piece.strip()]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
            continue
        if len(current) + len(piece) <= max_chars:
            current += piece
        else:
            chunks.extend(_force_split(current, max_chars=max_chars))
            current = piece
    if current:
        chunks.extend(_force_split(current, max_chars=max_chars))
    return [chunk for chunk in chunks if chunk.strip()]


def _force_split(text: str, *, max_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    return [text[index : index + max_chars].strip() for index in range(0, len(text), max_chars) if text[index : index + max_chars].strip()]


def _estimate_timing(chunks: list[str], duration: float) -> list[TranscriptCue]:
    total_chars = sum(max(1, len(chunk)) for chunk in chunks)
    cursor = 0.0
    cues: list[TranscriptCue] = []
    for index, chunk in enumerate(chunks):
        if index == len(chunks) - 1:
            end = duration
        else:
            end = cursor + duration * (max(1, len(chunk)) / total_chars)
        cues.append(TranscriptCue(start=cursor, end=max(cursor + 0.2, end), text=chunk))
        cursor = cues[-1].end
    return cues


def _read_json_if_possible(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _source_title(data: Any) -> str:
    return str(data.get("title") or "") if isinstance(data, dict) else ""


def _source_provider(data: Any) -> str:
    return str(data.get("provider") or "estimated_resegment") if isinstance(data, dict) else "estimated_resegment"


def _render_report(payload: dict[str, Any], json_path: Path, srt_path: Path) -> str:
    return "\n".join(
        [
            "# Transcript Resegment Report",
            "",
            f"- Source: `{payload.get('source_path', '')}`",
            f"- JSON: `{json_path}`",
            f"- SRT: `{srt_path}`",
            f"- Segments: `{len(payload.get('segments') or [])}`",
            f"- Duration seconds: `{payload.get('duration_seconds')}`",
            f"- Timing estimated: `{payload.get('timing_estimated')}`",
            f"- Method: `{payload.get('timing_estimation_method')}`",
            "",
            "> 这个输出用于修复无时间戳 ASR 的时间轴对齐；它不是精准字幕，需要人工抽查。",
            "",
        ]
    )
