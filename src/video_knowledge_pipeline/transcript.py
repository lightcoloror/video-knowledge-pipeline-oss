from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .bilinote_transcript_tools import parse_subtitle_text
from .models import TranscriptCue
from .transcript_speakers import (
    cue_speaker,
    cue_speaker_role,
    normalise_speaker_value,
    split_speaker_prefix,
)


TIMESTAMP_RE = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})(?P<ms>[,.]\d{1,3})?"
)


_SPEAKER_TIMESTAMP_HEADER_RE = re.compile(
    r"^\s*(?:[🟢🟣🔵🟠🟡🟤⚪⚫]\s*)?"
    r"(?P<speaker>(?:说话人\s*\d+|speaker[\s_-]*\d+|spk[\s_-]*\d+))"
    r"\s+\[?(?P<timestamp>\d{1,2}:\d{2}:\d{2}(?:[,.]\d{1,3})?)\]?\s*$",
    re.IGNORECASE,
)

def parse_timestamp(value: str) -> float:
    match = TIMESTAMP_RE.search(value.strip())
    if not match:
        raise ValueError(f"Invalid timestamp: {value}")
    hours = int(match.group("h"))
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    millis_text = (match.group("ms") or ".0").replace(",", ".")
    millis = float(millis_text)
    return hours * 3600 + minutes * 60 + seconds + millis


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total = int(seconds)
    millis = int(round((seconds - total) * 1000))
    if millis == 1000:
        total += 1
        millis = 0
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def parse_transcript(path: str | Path, *, source_snapshot: Any = None) -> list[TranscriptCue]:
    transcript_path = Path(path)
    suffix = transcript_path.suffix.lower()
    if source_snapshot is not None:
        # Parse the same immutable decoded object used to create locators, not a
        # second file read that could race. Existing callers keep their parser.
        if suffix == '.json':
            return _parse_json_transcript('', source_snapshot=source_snapshot)
        if suffix not in {'.srt', '.vtt'}:
            raise ValueError('source_locator snapshot requires JSON or timed subtitle')
    text = source_snapshot.data if source_snapshot is not None else transcript_path.read_text(encoding="utf-8-sig")
    if suffix == ".json":
        return _parse_json_transcript(text)
    if suffix in {".srt", ".vtt"}:
        # Intent: reuse Bilinote's established timestamp parser and adapt only
        # the MOSS-style speaker prefix instead of creating a second parser.
        cues = parse_subtitle_text(text, source_snapshot=source_snapshot)
        for cue in cues:
            speaker, clean_text = split_speaker_prefix(cue.text)
            cue.text = clean_text
            cue.speaker = speaker
            if speaker:
                cue.metadata = {**dict(cue.metadata), "speaker": speaker}
        return cues
    speaker_timeline = _parse_speaker_timestamp_plain_text(text)
    if speaker_timeline:
        return speaker_timeline
    return _parse_plain_text(text)


def transcript_excerpt(cues: list[TranscriptCue], start: float, end: float) -> str:
    parts = [cue.text for cue in cues if cue.end >= start and cue.start <= end]
    return " ".join(parts).strip()


def semantic_timepoints(cues: list[TranscriptCue], topic: str = "", max_points: int = 40) -> list[tuple[float, str]]:
    topic_terms = [term.lower() for term in re.findall(r"[\w\u4e00-\u9fff]+", topic) if len(term) >= 2]
    scored: list[tuple[int, float, str]] = []
    for cue in cues:
        text = cue.text.strip()
        if not text:
            continue
        lower = text.lower()
        score = 0
        if "?" in text or "？" in text:
            score += 4
        if re.search(r"\d", text):
            score += 2
        if re.search(r"结论|所以|因此|关键|重要|问题|步骤|但是|然而|对比|原因|结果|claim|because|therefore|important|key|step|problem", lower):
            score += 3
        score += min(4, len(re.findall(r"[\w\u4e00-\u9fff]+", text)) // 8)
        score += sum(3 for term in topic_terms if term in lower)
        if score > 0:
            midpoint = (cue.start + cue.end) / 2
            scored.append((score, midpoint, "transcript_semantic"))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = sorted(scored[:max_points], key=lambda item: item[1])
    return [(midpoint, signal) for _, midpoint, signal in selected]


def _parse_timed_text(text: str) -> list[TranscriptCue]:
    cues: list[TranscriptCue] = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip())
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        timing_index = next((idx for idx, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        start_text, end_text = [part.strip() for part in lines[timing_index].split("-->", 1)]
        start = parse_timestamp(start_text)
        end = parse_timestamp(end_text)
        cue_text = " ".join(lines[timing_index + 1 :]).strip()
        speaker, cue_text = split_speaker_prefix(cue_text)
        if cue_text:
            cues.append(
                TranscriptCue(
                    start=start,
                    end=end,
                    text=cue_text,
                    speaker=speaker,
                    metadata={"speaker": speaker} if speaker else {},
                )
            )
    return cues


def _parse_json_transcript(text: str, *, source_snapshot: Any = None) -> list[TranscriptCue]:
    data = source_snapshot.data if source_snapshot is not None else json.loads(text)
    selector_prefix = []
    if isinstance(data, dict):
        items = data.get("segments") or data.get("cues") or []
        selector_prefix = ['segments' if data.get('segments') else 'cues']
    else:
        items = data
    cues = []
    for index, item in enumerate(items, start=1):
        start = float(item.get("start", item.get("start_seconds", 0)))
        end = float(item.get("end", item.get("end_seconds", start)))
        cue_text = str(item.get("text", "")).strip()
        if cue_text or bool(item.get("empty_text_preserved")):
            metadata = dict(item.get("metadata") or {}) if isinstance(item.get("metadata"), dict) else {}
            # These are projection metadata, never trusted from imported JSON.
            for key in ('source_locators', 'source_observation_text', 'source_locator_scope'):
                metadata.pop(key, None)
            if source_snapshot is not None:
                metadata.update(
                    source_locators=[source_snapshot.locator([*selector_prefix, index - 1])],
                    source_observation_text=cue_text,
                    source_locator_scope='transcript_json_cue_not_original_audio',
                )
            for metric in ("no_speech_prob", "avg_logprob", "compression_ratio"):
                if metric in item and metric not in metadata:
                    metadata[metric] = item.get(metric)
            # Keep the semantic-correction source identity when a corrected
            # transcript is parsed again.  Summary/content-scope projections
            # may omit cues and renumber their visible positions, so that
            # projected position cannot safely stand in for the original ASR
            # cue index.
            if "source_segment_index" in item:
                metadata.setdefault("source_segment_index", item.get("source_segment_index"))
            if isinstance(item.get("source_segment_indexes"), list):
                metadata.setdefault(
                    "source_segment_indexes",
                    [
                        int(value)
                        for value in item.get("source_segment_indexes") or []
                        if isinstance(value, int) and value >= 0
                    ],
                )
            if (
                "source_segment_index" not in metadata
                and not metadata.get("source_segment_indexes")
            ):
                metadata["source_segment_index"] = index - 1
            # Intent: keep numeric cluster zero when normalized JSON is read again.
            # Decision: reuse the shared direct/metadata speaker resolver.
            # Reason: truthy fallback silently erased CAM++ speaker 0.
            # Evidence: FunASR 1.3.30 emits integer speaker clusters in sentence_info.
            # Effective scope: JSON transcript parsing only; speakers stay anonymous.
            speaker = cue_speaker(item)
            speaker_role = cue_speaker_role(item)
            segment_id = _json_segment_identity(item, fallback=f"segment-{index:06d}")
            source_ids = [
                str(value)
                for value in (item.get("source_segment_ids") or [segment_id])
                if str(value).strip()
            ]
            transformations = [
                dict(value)
                for value in (item.get("transformations") or [])
                if isinstance(value, dict)
            ]
            cues.append(
                TranscriptCue(
                    start=start,
                    end=end,
                    text=cue_text,
                    segment_id=segment_id,
                    source_segment_ids=source_ids,
                    transformations=transformations,
                    speaker=speaker,
                    speaker_role=speaker_role,
                    metadata=metadata,
                )
            )
    return cues


def _json_segment_identity(item: dict, *, fallback: str) -> str:
    for key in ("segment_id", "id"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return fallback


def _parse_speaker_timestamp_plain_text(text: str) -> list[TranscriptCue]:
    """Parse GetBrain-style speaker headers without changing the spoken text.

    Intent: retain the speaker and real start time already present in exported
    consultation transcripts.
    Decision: adapt MOSS's source-segment-first parser boundary to the
    line-oriented ``speaker timestamp`` contract, while reusing VKP's existing
    timestamp and speaker normalizers.
    Reason: the generic plain-text fallback otherwise turns the title, speaker
    header, and body into separate zero-duration pseudo segments.
    Evidence: the user-provided export uses ``说话人1 00:00:00`` followed by one
    or more body lines; MOSS keeps start/speaker/text as separate source fields.
    The existing ``extract_logseq_original_transcript`` remains responsible for
    the different, evaluation-only ``- 原始转录`` indented-tree contract.
    Effective scope: local TXT/Markdown parsing only. Text is copied verbatim;
    no term correction, speaker-role inference, diarization, model call, or
    external fact checking is performed.
    """

    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for source_line, raw_line in enumerate(
        text.replace("\r\n", "\n").replace("\r", "\n").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        match = _SPEAKER_TIMESTAMP_HEADER_RE.fullmatch(line)
        if match:
            if current is not None:
                records.append(current)
            current = {
                "speaker": normalise_speaker_value(match.group("speaker")),
                "start": parse_timestamp(match.group("timestamp")),
                "header": line,
                "source_line": source_line,
                "text_lines": [],
            }
            continue
        if current is not None and line:
            current["text_lines"].append(line)
    if current is not None:
        records.append(current)
    if not records:
        return []

    cues: list[TranscriptCue] = []
    for record_index, record in enumerate(records):
        cue_text = " ".join(str(value) for value in record["text_lines"]).strip()
        if not cue_text:
            continue
        start = float(record["start"])
        next_start = (
            float(records[record_index + 1]["start"])
            if record_index + 1 < len(records)
            else None
        )
        end = next_start if next_start is not None and next_start >= start else start
        segment_id = f"segment-{len(cues) + 1:06d}"
        metadata = {
            "speaker": str(record["speaker"]),
            "source_format": "speaker_timestamp_plaintext",
            "source_header": str(record["header"]),
            "source_header_line": int(record["source_line"]),
            "end_inferred_from_next_start": next_start is not None and next_start >= start,
            "end_unknown": next_start is None,
            "non_monotonic_next_start": next_start is not None and next_start < start,
        }
        cues.append(
            TranscriptCue(
                start=start,
                end=end,
                text=cue_text,
                segment_id=segment_id,
                source_segment_ids=[segment_id],
                speaker=str(record["speaker"]),
                metadata=metadata,
            )
        )
    return cues


def _parse_plain_text(text: str) -> list[TranscriptCue]:
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    return [
        TranscriptCue(
            start=float(index - 1),
            end=float(index - 1),
            text=row,
            segment_id=f"segment-{index:06d}",
            source_segment_ids=[f"segment-{index:06d}"],
        )
        for index, row in enumerate(rows, start=1)
    ]
