from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .content_scope import (
    apply_content_scope_to_cues,
    content_scope_replacements,
)
from .models import TranscriptCue


TRANSCRIPT_PROJECTION_SCHEMA = "video_knowledge_pipeline.transcript_projection.v1"
PROJECTION_DISPOSITIONS = (
    "retained_semantic",
    "excluded_confirmed",
    "normalized_disfluency",
    "review_gap",
)

# These tokens have no independent propositional content.  Ambiguous responses
# such as 嗯/好/对/是/不 are deliberately absent.
SAFE_STANDALONE_FILLERS = {
    "呃",
    "呃呃",
    "额",
    "额额",
    "啊",
    "啊啊",
    "哈",
    "哈哈",
    "哎",
    "诶",
    "唉",
    "哦哦",
}

_PUNCTUATION_ONLY_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)
_PURE_SOUND_EVENT_RE = re.compile(
    r"^\s*[\[【（(]\s*(?:music|noise|silence|音乐|噪声|静音|无声|背景音乐)\s*[\]】）)]\s*$",
    re.IGNORECASE,
)
_REVIEW_SOUND_EVENT_RE = re.compile(
    r"^\s*[\[【（(]\s*(?:applause|laughter|cough|footsteps|掌声|笑声|咳嗽|脚步声?|串音|多人重叠)\s*[\]】）)]\s*$",
    re.IGNORECASE,
)
_UNINTELLIGIBLE_RE = re.compile(
    r"(?:\[\s*(?:unintelligible|inaudible)\s*\]|【\s*(?:听不清|无法辨认)\s*】|\[\s*(?:听不清|无法辨认)\s*\])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WhisperQualityDecision:
    disposition: str
    reasons: tuple[str, ...]
    omission_allowed: bool


def classify_whisper_quality_signals(metadata: dict[str, Any] | None) -> WhisperQualityDecision:
    """Classify Whisper diagnostics without turning a score into deletion proof.

    A no-speech/log-probability combination is only a review candidate.  It is
    never a confirmed hallucination unless an operator records a content-scope
    interval.  Compression and low log probability likewise request retry or
    review and do not authorize omission.
    """

    values = metadata if isinstance(metadata, dict) else {}
    no_speech = _optional_float(values.get("no_speech_prob"))
    avg_logprob = _optional_float(values.get("avg_logprob"))
    compression = _optional_float(values.get("compression_ratio"))
    reasons: list[str] = []
    if no_speech is not None and avg_logprob is not None and no_speech > 0.6 and avg_logprob < -1.0:
        reasons.append("whisper_silence_hallucination_candidate")
    if compression is not None and compression > 2.4:
        reasons.append("whisper_high_compression_retry_required")
    if avg_logprob is not None and avg_logprob < -1.0:
        reasons.append("whisper_low_logprob_review_required")
    if reasons:
        return WhisperQualityDecision("review_gap", tuple(reasons), False)
    return WhisperQualityDecision("retained_semantic", (), False)


def project_transcript_cues(
    cues: Iterable[Any],
    scope: dict[str, Any],
    *,
    projection: str,
) -> tuple[list[TranscriptCue], dict[str, Any]]:
    """Build a reversible reader projection while preserving source evidence."""

    source = list(cues)
    scoped, scope_report = apply_content_scope_to_cues(source, scope, projection=projection)
    scope_excluded = {
        str(row.get("id") or ""): row
        for row in scope_report.get("excluded_items") or []
        if isinstance(row, dict)
    }
    partial = {
        str(row.get("id") or ""): row
        for row in scope_report.get("partial_overlap_items") or []
        if isinstance(row, dict)
    }
    partial_ranges = {
        (round(float(row.get("start") or 0.0), 6), round(float(row.get("end") or 0.0), 6))
        for row in partial.values()
    }
    events: list[dict[str, Any]] = []
    counts = {key: 0 for key in PROJECTION_DISPOSITIONS}
    for row in scope_excluded.values():
        events.append({**row, "action": "omit_from_projection"})
        counts["excluded_confirmed"] += 1

    result: list[TranscriptCue] = []
    previous_retained: TranscriptCue | None = None
    for position, value in enumerate(scoped, start=1):
        cue = _copy_cue(value, fallback_id=f"projection-cue-{position:06d}")
        identity = cue.segment_id or f"projection-cue-{position:06d}"
        text = cue.text.strip()
        disposition = "retained_semantic"
        reasons: list[str] = []
        action = "retain"

        if identity in partial or (round(cue.start, 6), round(cue.end, 6)) in partial_ranges:
            disposition = "review_gap"
            reasons.append("partial_scope_overlap_requires_review")

        if not text:
            disposition = "excluded_confirmed"
            reasons = ["empty_cue"]
            action = "omit_from_projection"
        elif _is_punctuation_only(text):
            disposition = "excluded_confirmed"
            reasons = ["punctuation_only_cue"]
            action = "omit_from_projection"
        elif projection != "full_transcript" and _normalized_token(text) in SAFE_STANDALONE_FILLERS:
            disposition = "normalized_disfluency"
            reasons = ["safe_standalone_filler"]
            action = "omit_from_projection"
        elif projection != "full_transcript" and _PURE_SOUND_EVENT_RE.fullmatch(text):
            disposition = "excluded_confirmed"
            reasons = ["nonsemantic_sound_event"]
            action = "omit_from_projection"
        elif projection != "full_transcript" and (
            _REVIEW_SOUND_EVENT_RE.fullmatch(text) or _is_pure_unintelligible(text)
        ):
            disposition = "review_gap"
            reasons = ["sound_or_unintelligible_context_requires_review"]
            action = "omit_from_projection"
        else:
            quality = classify_whisper_quality_signals(_quality_metadata(cue.metadata))
            if quality.disposition == "review_gap":
                disposition = "review_gap"
                reasons.extend(quality.reasons)
                # A combined silence signal is unsafe as semantic evidence, but
                # the cue remains in the full transcript and source file.
                if (
                    projection != "full_transcript"
                    and "whisper_silence_hallucination_candidate" in quality.reasons
                ):
                    action = "omit_from_projection"
            if _UNINTELLIGIBLE_RE.search(text):
                disposition = "review_gap"
                reasons.append("contains_unintelligible_marker")

        if action == "retain" and previous_retained is not None and _is_proven_duplicate(previous_retained, cue):
            disposition = "normalized_disfluency"
            reasons = ["proven_chunk_overlap_duplicate"]
            action = "omit_from_projection"

        event = {
            "id": identity,
            "start": cue.start,
            "end": cue.end,
            "disposition": disposition,
            "reasons": _dedupe(reasons) or ["semantic_content_retained"],
            "action": action,
        }
        events.append(event)
        counts[disposition] += 1
        if action == "retain":
            projected = _with_projection_metadata(cue, disposition=disposition, reasons=event["reasons"])
            result.append(projected)
            previous_retained = projected

    replacements = content_scope_replacements(scope, projection=projection)
    for position, replacement in enumerate(replacements, start=1):
        cue = TranscriptCue(
            start=float(replacement["start"]),
            end=float(replacement["end"]),
            text=str(replacement["text"]),
            segment_id=f"content-scope-replacement-{position:04d}",
            source_segment_ids=[],
            transformations=[
                {
                    "type": "operator_confirmed_projection_replacement",
                    "interval_id": replacement.get("interval_id") or "",
                    "boundary_changed": False,
                }
            ],
            metadata={
                "projection_disposition": "retained_semantic",
                "projection_reasons": ["operator_confirmed_projection_replacement"],
                "content_scope_interval_id": replacement.get("interval_id") or "",
            },
        )
        result.append(cue)
        events.append(
            {
                "id": cue.segment_id,
                "start": cue.start,
                "end": cue.end,
                "disposition": "retained_semantic",
                "reasons": ["operator_confirmed_projection_replacement"],
                "action": "insert_projection_replacement",
            }
        )
        counts["retained_semantic"] += 1

    result.sort(key=lambda cue: (cue.start, cue.end, cue.segment_id))
    report = {
        **scope_report,
        "schema": TRANSCRIPT_PROJECTION_SCHEMA,
        "projection": projection,
        "input_count": len(source),
        "output_count": len(result),
        "projection_omitted_count": sum(
            1 for event in events if event.get("action") == "omit_from_projection"
        ),
        "event_count": len(events),
        "events": events[:2000],
        "events_truncated": len(events) > 2000,
        "disposition_counts": counts,
        "replacement_count": len(replacements),
        "source_evidence_preserved": True,
        "source_mutation": False,
        "policy": {
            "low_confidence_alone_never_deletes": True,
            "ambiguous_fillers_retained": ["嗯", "好", "对", "是", "不"],
            "duplicate_requires_overlap_or_shared_source_identity": True,
            "confirmed_hallucination_requires_operator_scope": True,
        },
    }
    return result, report


def _copy_cue(value: Any, *, fallback_id: str) -> TranscriptCue:
    metadata = _mapping_value(value, "metadata")
    return TranscriptCue(
        start=_finite_float(_value(value, "start"), default=0.0),
        end=_finite_float(_value(value, "end"), default=_finite_float(_value(value, "start"), default=0.0)),
        text=str(_value(value, "text") or ""),
        segment_id=str(_value(value, "segment_id") or _value(value, "id") or fallback_id),
        source_segment_ids=[str(item) for item in (_value(value, "source_segment_ids") or []) if str(item)],
        transformations=[dict(item) for item in (_value(value, "transformations") or []) if isinstance(item, dict)],
        speaker=str(_value(value, "speaker") or ""),
        speaker_role=str(_value(value, "speaker_role") or ""),
        metadata=dict(metadata),
    )


def _with_projection_metadata(
    cue: TranscriptCue,
    *,
    disposition: str,
    reasons: list[str],
) -> TranscriptCue:
    return TranscriptCue(
        start=cue.start,
        end=cue.end,
        text=cue.text,
        segment_id=cue.segment_id,
        source_segment_ids=list(cue.source_segment_ids),
        transformations=list(cue.transformations),
        speaker=cue.speaker,
        speaker_role=cue.speaker_role,
        metadata={
            **dict(cue.metadata),
            "projection_disposition": disposition,
            "projection_reasons": list(reasons),
        },
    )


def _is_proven_duplicate(previous: TranscriptCue, current: TranscriptCue) -> bool:
    if _normalized_text(previous.text) != _normalized_text(current.text):
        return False
    if not _normalized_text(current.text):
        return False
    if previous.speaker and current.speaker and previous.speaker != current.speaker:
        return False
    overlap = min(previous.end, current.end) - max(previous.start, current.start)
    shared_source = bool(set(previous.source_segment_ids) & set(current.source_segment_ids))
    previous_locators = previous.metadata.get('source_locators') or []
    current_locators = current.metadata.get('source_locators') or []
    if previous_locators or current_locators:
        # Original block/object identities outrank reusable numeric cue labels.
        # Different ranges with repeated IDs are not proof of duplicated speech.
        shared_source = any(left == right for left in previous_locators for right in current_locators)
    return overlap > 0.001 or shared_source


def _is_punctuation_only(text: str) -> bool:
    return bool(text.strip()) and bool(_PUNCTUATION_ONLY_RE.fullmatch(text))


def _is_pure_unintelligible(text: str) -> bool:
    stripped = _UNINTELLIGIBLE_RE.sub("", text).strip(" \t\r\n，。！？,.!?;；:：-—")
    return not stripped and bool(_UNINTELLIGIBLE_RE.search(text))


def _normalized_token(text: str) -> str:
    return re.sub(r"[\s，。！？,.!?;；:：~～…]+", "", text).strip().lower()


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip().lower()


def _mapping_value(value: Any, key: str) -> dict[str, Any]:
    candidate = _value(value, key)
    return dict(candidate) if isinstance(candidate, dict) else {}


def _quality_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    for key in ("whisper", "asr_quality", "decoding"):
        nested = metadata.get(key)
        if isinstance(nested, dict):
            for metric in ("no_speech_prob", "avg_logprob", "compression_ratio"):
                if metric not in result and metric in nested:
                    result[metric] = nested[metric]
    return result


def _value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _finite_float(value: Any, *, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result
