from __future__ import annotations

import re
from html import unescape
from typing import Iterable

from .models import TranscriptCue


def parse_subtitle_text(text: str, *, source_snapshot=None) -> list[TranscriptCue]:
    """Parse SRT/VTT/plain subtitle text using BiliNote-style cleanup rules."""
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    if "-->" in normalized:
        return _parse_timed_subtitle(str(text) if source_snapshot is not None else normalized,
                                     source_snapshot=source_snapshot)
    return parse_plain_transcript(normalized)


def parse_plain_transcript(text: str) -> list[TranscriptCue]:
    cues: list[TranscriptCue] = []
    for index, line in enumerate(str(text or "").splitlines()):
        clean = re.sub(r"^\[[^\]]+\]\s*", "", line).strip()
        if clean:
            segment_id = f"segment-{len(cues) + 1:06d}"
            cues.append(
                TranscriptCue(
                    start=float(index),
                    end=float(index),
                    text=clean,
                    segment_id=segment_id,
                    source_segment_ids=[segment_id],
                )
            )
    return cues


def merge_transcript_segments(segments: Iterable[TranscriptCue]) -> list[TranscriptCue]:
    """Merge adjacent short cues without crossing strong punctuation boundaries."""
    merged: list[TranscriptCue] = []
    current: TranscriptCue | None = None
    for position, segment in enumerate(segments, start=1):
        text = str(segment.text or "").strip()
        if not text:
            continue
        segment_id = str(segment.segment_id or f"segment-{position:06d}")
        source_ids = list(segment.source_segment_ids or [segment_id])
        cleaned = TranscriptCue(
            start=float(segment.start),
            end=float(segment.end),
            text=text,
            segment_id=segment_id,
            source_segment_ids=source_ids,
            transformations=list(segment.transformations),
        )
        if current is None:
            current = cleaned
            continue
        gap = cleaned.start - current.end
        duration = current.end - current.start
        should_merge = (
            gap <= 1.2
            and duration <= 12
            and len(current.text) <= 80
            and not _ends_with_strong_punctuation(current.text)
        )
        if should_merge:
            current = TranscriptCue(
                start=current.start,
                end=max(current.end, cleaned.end),
                text=_join_transcript_text(current.text, cleaned.text),
                segment_id=current.segment_id,
                source_segment_ids=current.source_segment_ids + cleaned.source_segment_ids,
                transformations=current.transformations
                + cleaned.transformations
                + [
                    {
                        "type": "explicit_merge",
                        "source_segment_ids": current.source_segment_ids + cleaned.source_segment_ids,
                    }
                ],
            )
        else:
            merged.append(current)
            current = cleaned
    if current is not None:
        merged.append(current)
    return merged


def segments_to_txt(segments: Iterable[TranscriptCue]) -> str:
    lines = [f"[{format_display_timestamp(cue.start)}] {cue.text}" for cue in segments if str(cue.text or "").strip()]
    return "\n".join(lines).rstrip() + ("\n" if lines else "")


def segments_to_srt(segments: Iterable[TranscriptCue]) -> str:
    blocks: list[str] = []
    for index, cue in enumerate(segments, start=1):
        end = cue.end if cue.end > cue.start else cue.start + 2
        blocks.extend(
            [
                str(cue.segment_id or index),
                f"{format_srt_timestamp(cue.start)} --> {format_srt_timestamp(end)}",
                str(cue.text or "").strip(),
                "",
            ]
        )
    return "\n".join(blocks).rstrip() + ("\n" if blocks else "")


def parse_srt_timestamp(value: str) -> float:
    match = re.search(r"(?:(\d+):)?(\d+):(\d+)[,.](\d+)", str(value or ""))
    if not match:
        return 0.0
    hours, minutes, seconds, milliseconds = match.groups(default="0")
    millisecond_value = float(f"0.{milliseconds[:3].ljust(3, '0')}")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + millisecond_value


def format_display_timestamp(seconds: float) -> str:
    safe_seconds = max(0, int(seconds))
    hours = safe_seconds // 3600
    minutes = (safe_seconds % 3600) // 60
    remaining = safe_seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{remaining:02d}"
    return f"{minutes:02d}:{remaining:02d}"


def format_srt_timestamp(seconds: float) -> str:
    safe = max(0.0, float(seconds))
    total = int(safe)
    milliseconds = int((safe - total) * 1000)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def clean_subtitle_text(value: str) -> str:
    text = unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{\\[^}]+\}", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _subtitle_blocks(text: str):
    """Keep original ranges while adapting the existing blank-line blocks.

    A CRLF is one line terminator. Do not map cleaned text back by searching:
    repeated cue IDs/text and HTML entities make that association ambiguous.
    """
    start = None
    end = 0
    for line in re.finditer(r'[^\r\n]*(?:\r\n|\r|\n|$)', text):
        if line.start() == line.end():
            continue
        if line.group().strip():
            if start is None:
                start = line.start()
            end = line.end()
        elif start is not None:
            yield text[start:end], [start, end]
            start = None
    if start is not None:
        yield text[start:end], [start, end]


def _parse_timed_subtitle(text: str, *, source_snapshot=None) -> list[TranscriptCue]:
    cues: list[TranscriptCue] = []
    blocks = (_subtitle_blocks(text) if source_snapshot is not None else
              ((block, None) for block in re.split(r"\n\s*\n", text.strip())))
    for block, span in blocks:
        block = block.replace('\r\n', '\n').replace('\r', '\n')
        lines = [line.strip("\ufeff ").strip() for line in block.split("\n") if line.strip()]
        timing_index = next((idx for idx, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        start_raw, end_raw = [part.strip() for part in lines[timing_index].split("-->", 1)]
        end_raw = end_raw.split()[0]
        body = " ".join(clean_subtitle_text(line) for line in lines[timing_index + 1 :]).strip()
        if body:
            start = parse_srt_timestamp(start_raw)
            explicit_id = lines[timing_index - 1] if timing_index > 0 else ""
            segment_id = explicit_id or f"segment-{len(cues) + 1:06d}"
            cues.append(
                TranscriptCue(
                    start=start,
                    end=parse_srt_timestamp(end_raw),
                    text=body,
                    segment_id=segment_id,
                    source_segment_ids=[segment_id],
                    metadata={
                        'source_locators': [source_snapshot.locator(span)],
                        'source_observation_text': body,
                        'source_locator_scope': 'subtitle_block_not_original_audio',
                    } if source_snapshot is not None else {},
                )
            )
    return cues


def _ends_with_strong_punctuation(text: str) -> bool:
    return bool(re.search(r"[。！？!?；;]$", str(text or "").strip()))


def _join_transcript_text(left: str, right: str) -> str:
    left = str(left or "").strip()
    right = str(right or "").strip()
    if not left:
        return right
    if not right:
        return left
    if re.match(r"^[,，.。!?！？;；:：]", right):
        return f"{left}{right}"
    if re.search(r"[A-Za-z0-9]$", left) and re.match(r"^[A-Za-z0-9]", right):
        return f"{left} {right}"
    if re.search(r"[\u4e00-\u9fff]$", left) and re.match(r"^[\u4e00-\u9fff]", right):
        return f"{left}，{right}"
    return f"{left}{right}"
