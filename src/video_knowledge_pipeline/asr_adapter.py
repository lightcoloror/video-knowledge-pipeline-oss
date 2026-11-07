from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .media_tools import resolve_media_tool
from .models import TranscriptCue, new_id
from .storage import ensure_project_dirs, write_json
from .transcript import format_timestamp, parse_transcript
from .transcript_speakers import (
    cue_speaker,
    cue_speaker_role,
    speaker_display_name,
    speaker_label_map,
    speaker_payload,
)


_SENSEVOICE_TAG_PATTERN = re.compile(r"<\s*\|\s*([^<>|]+?)\s*\|\s*>")


def normalize_asr_output(root: str | Path, input_path: str | Path, *, provider: str = "auto", title: str | None = None) -> dict[str, Any]:
    """Normalize external ASR output to the transcript format consumed by the pipeline."""
    paths = ensure_project_dirs(root)
    source = Path(input_path)
    cues = read_asr_cues(source, provider=provider)
    transcript_id = new_id("transcript")
    output_dir = paths["transcripts"] / transcript_id
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = {
        "id": transcript_id,
        "title": title or source.stem,
        "provider": provider,
        "source_path": str(source),
        "segments": read_asr_segment_dicts(source, provider=provider, cues=cues),
    }
    source_execution = _trusted_connector_asr_provenance(_read_asr_data(source))
    if source_execution:
        normalized["source_execution"] = source_execution
    json_path = output_dir / "normalized-transcript.json"
    srt_path = output_dir / "normalized-transcript.srt"
    write_json(json_path, normalized)
    srt_path.write_text(render_srt(cues), encoding="utf-8")
    return {
        "transcript_id": transcript_id,
        "segment_count": len(cues),
        "json_path": str(json_path),
        "srt_path": str(srt_path),
    }


def read_asr_cues(path: str | Path, *, provider: str = "auto") -> list[TranscriptCue]:
    source = Path(path)
    if source.suffix.lower() in {".srt", ".vtt", ".txt", ".md"}:
        return parse_transcript(source)
    data = _read_asr_data(path)
    provider_key = provider.lower()
    if provider_key == "auto":
        provider_key = _detect_provider(data)
    if provider_key == "generic" and source.suffix.lower() not in {".json", ".jsonl", ".ndjson"}:
        return parse_transcript(source)
    if provider_key in {"qwen3-asr", "qwen3_asr"}:
        return _segments_from_sequence(_qwen3_items(data))
    if provider_key in {"whisperx", "faster-whisper", "whisper", "openai", "moss-transcribe-diarize", "moss_transcribe_diarize", "moss"}:
        return _segments_from_sequence(_segment_items(data))
    if provider_key in {"funasr", "sensevoice"}:
        return _segments_from_funasr(data)
    if provider_key == "dolphin":
        return _segments_from_dolphin(data)
    if provider_key == "generic":
        return _segments_from_sequence(_segment_items(data))
    raise ValueError(f"unsupported ASR provider: {provider}")


def read_asr_segment_dicts(path: str | Path, *, provider: str = "auto", cues: list[TranscriptCue] | None = None) -> list[dict[str, Any]]:
    source = Path(path)
    cue_values = cues if cues is not None else read_asr_cues(source, provider=provider)
    if source.suffix.lower() in {".srt", ".vtt", ".txt", ".md"}:
        labels = speaker_label_map(cue_values)
        return [_segment_dict(cue, speaker_labels=labels) for cue in cue_values]
    data = _read_asr_data(source)
    provider_key = provider.lower()
    if provider_key == "auto":
        provider_key = _detect_provider(data)
    if provider_key in {"funasr", "sensevoice"}:
        raw_items = _funasr_items(data)
    elif provider_key == "dolphin":
        raw_items = _dolphin_items(data)
    elif provider_key in {"qwen3-asr", "qwen3_asr"}:
        raw_items = _qwen3_items(data)
    else:
        raw_items = _segment_items(data)
    segments: list[dict[str, Any]] = []
    labels = speaker_label_map(cue_values)
    for index, cue in enumerate(cue_values):
        if not cue.segment_id:
            cue.segment_id = f"segment-{index + 1:06d}"
        if not cue.source_segment_ids:
            cue.source_segment_ids = [cue.segment_id]
        item = raw_items[index] if index < len(raw_items) else {}
        segments.append(
            _segment_dict(cue, metadata=_asr_metadata(item), speaker_labels=labels)
        )
    return segments


def render_srt(cues: list[TranscriptCue]) -> str:
    blocks = []
    labels = speaker_label_map(cues)
    for index, cue in enumerate(cues, start=1):
        speaker = speaker_display_name(cue, labels)
        text = f"{speaker}：{cue.text}" if speaker else cue.text
        blocks.extend(
            [
                str(cue.segment_id or index),
                f"{_srt_timestamp(cue.start)} --> {_srt_timestamp(cue.end)}",
                text,
                "",
            ]
        )
    return "\n".join(blocks).rstrip() + "\n"


def _detect_provider(data: Any) -> str:
    if isinstance(data, dict) and (
        str(data.get("provider") or "").lower() in {"qwen3-asr", "qwen3_asr"}
        or str(data.get("schema") or "").startswith("video_knowledge_pipeline.qwen3_asr_raw_output")
    ):
        return "qwen3-asr"
    data = _unwrap_asr_container(data)
    if isinstance(data, dict):
        if str(data.get("provider") or "").lower() in {"qwen3-asr", "qwen3_asr"} or str(data.get("schema") or "").startswith("video_knowledge_pipeline.qwen3_asr_raw_output"):
            return "qwen3-asr"
        if isinstance(data.get("segments"), list):
            return "whisperx"
        if str(data.get("schema") or "") == "video_knowledge_dolphin_raw_output.v1" or str(data.get("provider") or "").lower() == "dolphin":
            return "dolphin"
        if isinstance(data.get("sentence_info"), list) or isinstance(data.get("timestamp"), list):
            return "funasr"
        if isinstance(data.get("cues"), list):
            return "generic"
    if isinstance(data, list):
        if data and isinstance(data[0], dict) and ("timestamp" in data[0] or "sentence_info" in data[0]):
            return "funasr"
        return "generic"
    return "generic"


def read_asr_segment_items(data: Any) -> list[dict[str, Any]]:
    """Normalize provider segment rows, including word-only ASR responses."""

    return _segment_items(data)


def _segment_items(data: Any) -> list[dict[str, Any]]:
    data = _unwrap_asr_container(data)
    if isinstance(data, dict):
        items = data.get("segments") or data.get("cues") or []
        if not items:
            words = data.get("words")
            if isinstance(words, list):
                return _segments_from_timed_words(words)
    elif isinstance(data, list):
        items = data
    else:
        items = []
    return [item for item in items if isinstance(item, dict)]


def _qwen3_items(data: Any, *, max_segment_seconds: float = 15.0, silence_gap_seconds: float = 0.8) -> list[dict[str, Any]]:
    """Convert Qwen3-ASR chunk results with forced-alignment timestamps to timed segments."""
    if not isinstance(data, dict):
        return _segment_items(data)
    direct_segments = data.get("segments")
    if isinstance(direct_segments, list) and direct_segments:
        return [item for item in direct_segments if isinstance(item, dict)]
    results = data.get("results")
    if not isinstance(results, list):
        return _segment_items(data)

    rows: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        timestamps = result.get("timestamps")
        if not isinstance(timestamps, list) or not timestamps:
            text = _clean_asr_text(_first_text(result))
            if text:
                offset = float(result.get("chunk_offset_seconds") or 0.0)
                rows.append({"start": offset, "end": offset, "text": text})
            continue

        rows.extend(
            _segments_from_timed_words(
                timestamps,
                max_segment_seconds=max_segment_seconds,
                silence_gap_seconds=silence_gap_seconds,
            )
        )
    return rows


def _segments_from_timed_words(
    timestamps: list[Any],
    *,
    max_segment_seconds: float = 15.0,
    silence_gap_seconds: float = 0.8,
) -> list[dict[str, Any]]:
    """Reuse the established Qwen3 aligned-word grouping for word-only ASR."""

    rows: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    segment_start = 0.0
    previous_end = 0.0
    for raw_token in timestamps:
        if not isinstance(raw_token, dict):
            continue
        token_text = str(raw_token.get("text") or raw_token.get("word") or "").strip()
        if not token_text:
            continue
        token_start, token_end = _time_range_from_item(raw_token)
        token_end = max(token_start, token_end)
        if current and (
            token_start - previous_end >= silence_gap_seconds
            or token_end - segment_start >= max_segment_seconds
        ):
            rows.append(_qwen3_segment_row(current))
            current = []
        if not current:
            segment_start = token_start
        current.append({"text": token_text, "start": token_start, "end": token_end})
        previous_end = token_end
    if current:
        rows.append(_qwen3_segment_row(current))
    return rows


def _qwen3_segment_row(tokens: list[dict[str, Any]]) -> dict[str, Any]:
    start = float(tokens[0].get("start") or 0.0)
    end = float(tokens[-1].get("end") or start)
    return {
        "start": start,
        "end": max(start, end),
        "text": "".join(str(token.get("text") or "") for token in tokens),
        "words": [
            {
                "text": str(token.get("text") or ""),
                "start": float(token.get("start") or 0.0),
                "end": float(token.get("end") or token.get("start") or 0.0),
            }
            for token in tokens
        ],
    }

def _segments_from_sequence(
    items: list[dict[str, Any]],
    *,
    milliseconds: bool = False,
    segment_id_prefix: str = "",
) -> list[TranscriptCue]:
    cues = []
    for index, item in enumerate(items, start=1):
        text = _clean_asr_text(_first_text(item))
        metadata = _asr_metadata(item)
        speaker = cue_speaker(metadata)
        speaker_role = cue_speaker_role(metadata)
        start, end = _time_range_from_item(item, milliseconds=milliseconds)
        # Preserve timed empty segments. They are evidence of an ASR gap and their
        # original position/boundaries are required by downstream quality reports.
        # Untimed empty records remain ignorable provider noise.
        if not text and not (end > start):
            continue
        source_segment_id = _segment_identity(item, fallback=f"segment-{index:06d}")
        segment_id = (
            f"{segment_id_prefix}:{source_segment_id}"
            if segment_id_prefix
            else source_segment_id
        )
        source_segment_ids = [
            str(value)
            for value in (item.get("source_segment_ids") or [source_segment_id])
            if str(value).strip()
        ]
        if segment_id_prefix:
            source_segment_ids = [
                value
                if value.startswith(f"{segment_id_prefix}:")
                else f"{segment_id_prefix}:{value}"
                for value in source_segment_ids
            ]
        cues.append(
            TranscriptCue(
                start=start,
                end=max(start, end),
                text=text,
                segment_id=segment_id,
                source_segment_ids=source_segment_ids,
                transformations=[
                    dict(value)
                    for value in (item.get("transformations") or [])
                    if isinstance(value, dict)
                ],
                speaker=speaker,
                speaker_role=speaker_role,
                metadata=metadata,
            )
        )
    return cues


def _segments_from_funasr(data: Any) -> list[TranscriptCue]:
    duration_seconds = _duration_from_asr_container(data)
    unwrapped = _unwrap_asr_container(data)
    records = unwrapped if isinstance(unwrapped, list) else [unwrapped]
    cues: list[TranscriptCue] = []
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        sentence_info = record.get("sentence_info")
        if isinstance(sentence_info, list):
            # Intent: keep every normalized cue addressable after resumable
            # long-media chunks are merged.
            # Decision: namespace per-record sentence IDs only when FunASR
            # returns multiple records; single-pass compatibility is unchanged.
            # Reason: each chunk restarts upstream IDs at ``segment-000001``.
            # Evidence: a 21-chunk production run yielded 2,214 cues but only
            # 175 unique IDs before this adapter boundary.
            # Effective scope: multi-record FunASR/SenseVoice normalization;
            # text, timestamps, raw ASR, and provider metadata are unchanged.
            record_prefix = (
                f"funasr-record-{record_index:04d}" if len(records) > 1 else ""
            )
            cues.extend(
                _segments_from_sequence(
                    sentence_info,
                    milliseconds=True,
                    segment_id_prefix=record_prefix,
                )
            )
            continue
        text = _clean_asr_text(_first_text(record))
        metadata = _asr_metadata(record)
        speaker = cue_speaker(metadata)
        speaker_role = cue_speaker_role(metadata)
        timestamps = record.get("timestamp")
        if text and isinstance(timestamps, list) and timestamps:
            start = _timestamp_value(timestamps[0], 0, milliseconds=True)
            end = _timestamp_value(timestamps[-1], 1, milliseconds=True)
            cues.append(
                TranscriptCue(
                    start=start,
                    end=max(start, end),
                    text=text,
                    speaker=speaker,
                    speaker_role=speaker_role,
                    metadata=metadata,
                )
            )
        elif text:
            start, end = _time_range_from_item(record, milliseconds=True)
            if start == 0.0 and end == 0.0:
                window = _untimed_funasr_record_window(
                    data,
                    records,
                    record_index,
                    duration_seconds=duration_seconds,
                )
                record_cues = _split_untimed_funasr_text(
                    _first_text(record),
                    duration_seconds=window[1] - window[0],
                )
                for cue in record_cues:
                    cue.speaker = speaker
                    cue.speaker_role = speaker_role
                    cue.metadata = dict(metadata)
                for cue in record_cues:
                    for transformation in cue.transformations:
                        if transformation.get("type") == "timing_estimation":
                            transformation.update(
                                {
                                    "source_window_start": window[0],
                                    "source_window_end": window[1],
                                    "source_record_index": record_index,
                                }
                            )
                cues.extend(_offset_cues(record_cues, window[0]))
            else:
                cues.append(
                    TranscriptCue(
                        start=start,
                        end=max(start, end),
                        text=text,
                        speaker=speaker,
                        speaker_role=speaker_role,
                        metadata=metadata,
                    )
                )
    return cues


def _untimed_funasr_record_window(
    container: Any,
    records: list[Any],
    record_index: int,
    *,
    duration_seconds: float,
) -> tuple[float, float]:
    """Resolve an absolute media window for an untimed chunk result.

    Intent: preserve the real media identity of text-only FunASR chunks.
    Decision: prefer the resumable runner's exact ``chunk_offset_seconds`` /
    ``chunk_end_seconds`` manifest and retain ``chunk_seconds`` as a legacy fallback.
    Reason: spreading every chunk over the full media duration restarts the
    timeline at zero and turns normal text into false low-density warnings.
    Evidence: the 2026-07-24 production bundle had 320/326 false review rows;
    the same text normalized with the existing offsets has a monotonic
    0..6259.583667 timeline and only one coarse-timing review row.
    Effective scope: untimed FunASR/SenseVoice chunk normalization only;
    legacy single-pass output keeps its historical full-duration behavior.
    """

    record = records[record_index]
    if not isinstance(record, dict) or "chunk_offset_seconds" not in record:
        return 0.0, max(0.0, duration_seconds)

    start = _duration_seconds_value(record.get("chunk_offset_seconds"))
    end_candidates: list[float] = []
    explicit_end = _duration_seconds_value(record.get("chunk_end_seconds"))
    if explicit_end > start:
        end_candidates.append(explicit_end)
    for later in records[record_index + 1 :]:
        if not isinstance(later, dict) or "chunk_offset_seconds" not in later:
            continue
        later_start = _duration_seconds_value(later.get("chunk_offset_seconds"))
        if later_start > start:
            end_candidates.append(later_start)
            break

    if isinstance(container, dict):
        chunk_seconds = _duration_seconds_value(container.get("chunk_seconds"))
        if chunk_seconds > 0:
            end_candidates.append(start + chunk_seconds)
    if duration_seconds > start:
        end_candidates.append(duration_seconds)

    end = min(end_candidates) if end_candidates else start
    return start, max(start, end)


def _offset_cues(cues: list[TranscriptCue], offset_seconds: float) -> list[TranscriptCue]:
    if offset_seconds <= 0:
        return cues
    for cue in cues:
        cue.start += offset_seconds
        cue.end += offset_seconds
    return cues

def _funasr_items(data: Any) -> list[dict[str, Any]]:
    data = _unwrap_asr_container(data)
    records = data if isinstance(data, list) else [data]
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        sentence_info = record.get("sentence_info")
        if isinstance(sentence_info, list):
            rows.extend([item for item in sentence_info if isinstance(item, dict)])
        else:
            rows.append(record)
    return rows


def _segments_from_dolphin(data: Any) -> list[TranscriptCue]:
    duration_seconds = _duration_from_asr_container(data)
    items = _dolphin_items(data)
    if items:
        cues = _segments_from_sequence(items)
        if cues:
            return cues
    result = _dolphin_result(data)
    text = ""
    if isinstance(result, dict):
        text = _clean_asr_text(_first_text(result))
    elif isinstance(result, str):
        text = _clean_asr_text(result)
    if not text and isinstance(data, dict):
        text = _clean_asr_text(_first_text(data))
    if not text:
        return []
    end = duration_seconds if duration_seconds > 0 else max(float(len(text)) / 8.0, 1.0)
    return [TranscriptCue(start=0.0, end=end, text=text)]


def _dolphin_items(data: Any) -> list[dict[str, Any]]:
    result = _dolphin_result(data)
    if isinstance(result, dict):
        for key in ("segments", "sentences", "cues"):
            rows = result.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        words = result.get("words")
        if isinstance(words, list) and words and all(isinstance(row, dict) for row in words):
            return [{"start": _time_range_from_item(words[0])[0], "end": _time_range_from_item(words[-1])[1], "text": result.get("text", ""), "words": words}]
        if _first_text(result):
            return [result]
    if isinstance(result, list):
        return [row for row in result if isinstance(row, dict)]
    return []


def _dolphin_result(data: Any) -> Any:
    if isinstance(data, dict) and "result" in data:
        return data.get("result")
    return _unwrap_asr_container(data)


def _segment_dict(
    cue: TranscriptCue,
    *,
    metadata: dict[str, Any] | None = None,
    speaker_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": cue.segment_id,
        "segment_id": cue.segment_id,
        "source_segment_ids": list(cue.source_segment_ids or ([cue.segment_id] if cue.segment_id else [])),
        "start": cue.start,
        "end": cue.end,
        "text": cue.text,
        "transformations": list(cue.transformations),
    }
    # Intent/decision/reason/evidence/effective scope are documented in
    # transcript_speakers.speaker_label_map. Keep the raw cluster ID plus a
    # reader label; never replace it with a guessed role.
    row.update(speaker_payload(cue, speaker_labels))
    if not cue.text:
        row["empty_text_preserved"] = True
    merged_metadata = {**dict(cue.metadata), **dict(metadata or {})}
    clean_metadata = {
        key: value
        for key, value in merged_metadata.items()
        if value not in ("", [], {}, None)
    }
    if clean_metadata:
        row["metadata"] = clean_metadata
    return row


def _segment_identity(item: dict[str, Any], *, fallback: str) -> str:
    for key in ("segment_id", "id"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return fallback


def _asr_metadata(item: dict[str, Any]) -> dict[str, Any]:
    raw_text = _first_text(item)
    tags = _sensevoice_tags(raw_text)
    # Intent: preserve CAM++'s numeric cluster 0 through VKP normalization.
    # Decision: reuse the shared speaker-field resolver instead of truthy ``or``.
    # Reason: zero is a valid speaker ID, not missing metadata.
    # Evidence: the real FunASR sentence_info output uses integer spk values 0/1.
    # Effective scope: ASR speaker metadata only; no role or identity inference.
    speaker = cue_speaker(item)
    word_timestamps = read_asr_word_timestamps(item)
    return {
        "speaker": speaker,
        "speaker_local_cluster": (
            "" if item.get("spk") is None else str(item.get("spk")).strip()
        ),
        "speaker_global_id": str(item.get("speaker_global_id") or "").strip(),
        "speaker_role": cue_speaker_role(item),
        "emotion": item.get("emotion") or tags.get("emotion", ""),
        "audio_events": _list_value(item.get("audio_events") or item.get("events")) + tags.get("audio_events", []),
        "language": item.get("language") or tags.get("language", ""),
        "raw_tags": tags.get("raw_tags", []),
        "alignment": "word_level" if word_timestamps else "",
        "timestamp_precision": str(item.get("timestamp_precision") or ""),
        "word_count": len(word_timestamps) if word_timestamps else 0,
        "words": word_timestamps,
    }


def read_asr_word_timestamps(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the established provider word-timestamp contract."""

    words = item.get("words")
    if not isinstance(words, list):
        return []
    rows: list[dict[str, Any]] = []
    for word in words:
        if not isinstance(word, dict):
            continue
        text = str(word.get("word") or word.get("text") or "").strip()
        if not text:
            continue
        start, end = _time_range_from_item(word)
        row: dict[str, Any] = {"word": text}
        if start or end:
            row["start"] = start
            row["end"] = max(start, end)
        score = _first_present(word, ("score", "probability", "confidence"))
        if isinstance(score, (int, float)):
            row["score"] = float(score)
        speaker = cue_speaker(word)
        if speaker:
            row["speaker"] = speaker
        rows.append(row)
    return rows


_word_timestamps = read_asr_word_timestamps


def _sensevoice_tags(text: str) -> dict[str, Any]:
    tags = _SENSEVOICE_TAG_PATTERN.findall(str(text or ""))
    languages = {"zh", "en", "ja", "ko", "yue"}
    emotions = {"happy", "sad", "angry", "neutral", "fearful", "disgusted", "surprised"}
    result = {"raw_tags": [f"<|{tag}|>" for tag in tags], "audio_events": []}
    for tag in tags:
        clean = tag.strip()
        compact = re.sub(r"\s+", "", clean)
        lowered = compact.lower()
        if lowered in languages:
            result["language"] = lowered
        elif lowered in emotions:
            result["emotion"] = lowered
        elif clean:
            result.setdefault("audio_events", []).append(compact or clean)
    return result


def _list_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _first_present(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def _read_asr_data(path: str | Path) -> Any:
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            clean = line.strip()
            if not clean:
                continue
            rows.append(json.loads(clean))
        return rows


def _unwrap_asr_container(data: Any) -> Any:
    current = data
    while isinstance(current, dict):
        connector_payload = _trusted_connector_asr_payload(current)
        if connector_payload is not None:
            current = connector_payload
            continue
        for key in ("result", "results", "data", "value"):
            value = current.get(key)
            if isinstance(value, (dict, list)):
                current = value
                break
        else:
            return current
    return current


def _trusted_connector_asr_payload(data: dict[str, Any]) -> Any | None:
    schema = str(data.get("schema") or "")
    if not schema.startswith("video_knowledge_pipeline.trusted_model_connector."):
        return None

    model_result = data.get("model_result")
    if not isinstance(model_result, dict):
        raise ValueError("trusted connector ASR execution is missing model_result")
    task = str(data.get("task") or model_result.get("task") or "").strip()
    model_type = str(model_result.get("model_type") or "").strip().lower()
    if model_type != "asr" and task not in {"cloud_asr", "local_asr_service"}:
        raise ValueError(
            f"trusted connector execution is not an ASR task: {task or 'unknown'}"
        )
    if data.get("ok") is not True:
        raise ValueError(
            "trusted connector ASR execution did not complete successfully"
        )
    if data.get("transport_ok") is False:
        raise ValueError("trusted connector ASR transport validation failed")
    if data.get("contract_ok") is False:
        raise ValueError(
            "trusted connector ASR output contract validation failed"
        )

    runtime = model_result.get("runtime_result")
    if isinstance(runtime, dict) and isinstance(
        runtime.get("raw_output"), (dict, list)
    ):
        return runtime["raw_output"]
    raw_response = model_result.get("raw_response")
    if isinstance(raw_response, (dict, list)):
        return raw_response
    raise ValueError(
        "trusted connector ASR execution is missing structured raw output"
    )


def _trusted_connector_asr_provenance(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    schema = str(data.get("schema") or "")
    if not schema.startswith("video_knowledge_pipeline.trusted_model_connector."):
        return {}
    # Reuse the fail-closed validation performed by the normalizer.
    _trusted_connector_asr_payload(data)
    model_result = data.get("model_result")
    if not isinstance(model_result, dict):
        return {}
    runtime = model_result.get("runtime_result")
    runtime = runtime if isinstance(runtime, dict) else {}
    quality = model_result.get("asr_quality")
    if not isinstance(quality, dict):
        quality = runtime.get("asr_quality")
    quality = quality if isinstance(quality, dict) else {}
    retry_plan = quality.get("retry_plan")
    retry_plan = retry_plan if isinstance(retry_plan, dict) else {}
    deployment = runtime.get("deployment")
    deployment_identity = (
        str(deployment.get("id") or deployment.get("model") or "")
        if isinstance(deployment, dict)
        else str(deployment or "")
    )
    return {
        "schema": schema,
        "task": str(data.get("task") or model_result.get("task") or ""),
        "status": str(data.get("status") or model_result.get("status") or ""),
        "transport_ok": bool(data.get("transport_ok", True)),
        "contract_ok": bool(data.get("contract_ok", True)),
        "quality_gate_passed": bool(
            data.get(
                "quality_gate_passed",
                quality.get("quality_gate_passed", False),
            )
        ),
        "production_qualified": bool(data.get("production_qualified", False)),
        "consent_id": str(
            runtime.get("consent_id") or data.get("consent_id") or ""
        ),
        "route_id": str(runtime.get("route_id") or ""),
        "route_revision": str(runtime.get("route_revision") or ""),
        "provider": str(runtime.get("provider") or ""),
        "deployment": deployment_identity,
        "asr_quality": {
            "status": str(quality.get("status") or ""),
            "segment_count": int(quality.get("segment_count") or 0),
            "passed_segment_count": int(quality.get("passed_segment_count") or 0),
            "review_segment_count": int(quality.get("review_segment_count") or 0),
            "failed_segment_count": int(quality.get("failed_segment_count") or 0),
            "retry_status": str(retry_plan.get("status") or ""),
            "requires_new_exact_consent": bool(
                retry_plan.get("requires_new_exact_consent", False)
            ),
            "review_chunks": [
                dict(row)
                for row in (quality.get("review_chunks") or [])[:50]
                if isinstance(row, dict)
            ],
            "failed_chunks": [
                dict(row)
                for row in (quality.get("failed_chunks") or [])[:50]
                if isinstance(row, dict)
            ],
        },
    }



def _first_text(item: dict[str, Any]) -> str:
    for key in ("text_postprocessed", "text", "sentence", "value", "raw_text"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def _clean_asr_text(text: str) -> str:
    without_tags = _SENSEVOICE_TAG_PATTERN.sub("", str(text or ""))
    compact = re.sub(r"\s+", " ", without_tags).strip()
    return _normalize_asr_punctuation(compact)


def _normalize_asr_punctuation(text: str) -> str:
    value = str(text or "")
    if not value:
        return ""
    value = re.sub(r"([，。！？、；：,.!?])\1+", r"\1", value)
    value = re.sub(r"[，、]+([。！？])", r"\1", value)
    value = re.sub(r"([。！？])[,，、。]+", r"\1", value)
    value = re.sub(r"([。！？])([。！？])+", r"\1", value)
    value = re.sub(r"\s*([，。！？、；：,.!?])\s*", r"\1", value)
    return value.lstrip(" ，、。")


def _split_untimed_funasr_text(text: str, *, duration_seconds: float = 0.0) -> list[TranscriptCue]:
    tag_split_ready = _SENSEVOICE_TAG_PATTERN.sub("<TAG>", str(text or ""))
    chunks = [_clean_asr_text(chunk) for chunk in re.split(r"(?:<TAG>)+", tag_split_ready)]
    chunks = [chunk for chunk in chunks if chunk]
    if not chunks:
        clean = _clean_asr_text(text)
        chunks = [clean] if clean else []
    if not chunks:
        return []
    total_chars = sum(max(len(chunk), 1) for chunk in chunks)
    if duration_seconds <= 0:
        duration_seconds = float(len(chunks) * 5)
    cues: list[TranscriptCue] = []
    cursor = 0.0
    for index, chunk in enumerate(chunks):
        if index == len(chunks) - 1:
            end = duration_seconds
        else:
            end = cursor + duration_seconds * (max(len(chunk), 1) / total_chars)
        cues.append(
            TranscriptCue(
                start=cursor,
                end=max(cursor + 0.01, end),
                text=chunk,
                transformations=[
                    {
                        "type": "timing_estimation",
                        "method": "character_proportional_within_source_window",
                        "precision": "coarse",
                    }
                ],
            )
        )
        cursor = end
    return cues


def _duration_from_asr_container(data: Any) -> float:
    if not isinstance(data, dict):
        return 0.0
    for key in ("duration", "duration_seconds", "audio_duration", "video_duration"):
        value = data.get(key)
        # The local FunASR runner writes `duration_seconds` from ffprobe. Long
        # videos routinely exceed 10,000 seconds, so the generic heuristic in
        # `_seconds` would otherwise misclassify a valid seconds value as
        # milliseconds and compress the synthesized timeline by 1,000×.
        duration = _duration_seconds_value(value) if key == "duration_seconds" else _seconds(value)
        if duration > 0:
            return duration
    input_path = str(data.get("input") or "").strip()
    if input_path:
        return _probe_media_duration(input_path)
    return 0.0


def _duration_seconds_value(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _probe_media_duration(path: str) -> float:
    media = Path(path).expanduser()
    if not media.exists():
        return 0.0
    ffprobe = resolve_media_tool("ffprobe")
    if not ffprobe:
        return 0.0
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(media),
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0.0
    if completed.returncode != 0:
        return 0.0
    try:
        return max(0.0, float(completed.stdout.strip()))
    except ValueError:
        return 0.0


def _time_range_from_item(item: dict[str, Any], *, milliseconds: bool = False) -> tuple[float, float]:
    timestamp = item.get("timestamp")
    if isinstance(timestamp, list) and timestamp:
        start = _timestamp_value(timestamp[0], 0, milliseconds=milliseconds)
        end = _timestamp_value(timestamp[-1], 1, milliseconds=milliseconds)
        return start, max(start, end)

    start_key, start_value = _first_time_entry(item, ("start", "start_seconds", "start_ms", "start_time", "begin", "begin_time"))
    end_key, end_value = _first_time_entry(item, ("end", "end_seconds", "end_ms", "end_time", "finish", "finish_time"))
    start = _coerce_time(start_value, milliseconds=milliseconds or start_key.endswith("_ms"))
    end = _coerce_time(end_value if end_value is not None else start_value, milliseconds=milliseconds or end_key.endswith("_ms"))
    return start, end


def _first_time_entry(item: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, Any]:
    for key in keys:
        if key in item and item.get(key) is not None:
            return key, item.get(key)
    return "", None


def _coerce_time(value: Any, *, milliseconds: bool = False) -> float:
    return _milliseconds(value) if milliseconds else _seconds(value)


def _seconds(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number > 10000:
        return number / 1000
    return number


def _milliseconds(value: Any) -> float:
    try:
        return float(value) / 1000
    except (TypeError, ValueError):
        return 0.0


def _timestamp_value(value: Any, index: int, *, milliseconds: bool = False) -> float:
    if isinstance(value, (list, tuple)) and len(value) > index:
        return _milliseconds(value[index]) if milliseconds else _seconds(value[index])
    return _milliseconds(value) if milliseconds else _seconds(value)


def _srt_timestamp(seconds: float) -> str:
    return format_timestamp(seconds).replace(".", ",")
