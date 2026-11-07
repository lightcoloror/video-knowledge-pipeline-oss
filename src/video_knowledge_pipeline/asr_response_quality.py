from __future__ import annotations

import difflib
import re
from typing import Any

from .asr_adapter import read_asr_segment_items, read_asr_word_timestamps
from .interval_coverage import interval_coverage
from .text_normalization import compact_ascii_cjk as _compact_text


SCHEMA = "video_knowledge_pipeline.asr_response_quality.v1"
DEFAULT_AVG_LOGPROB_WARNING = -0.8
DEFAULT_COMPRESSION_RATIO_MAX = 2.4
DEFAULT_NO_SPEECH_PROB_MAX = 0.6
DEFAULT_TEXT_DENSITY_MIN = 0.5
DEFAULT_RETRY_OVERLAP_SECONDS = 1.5
DEFAULT_VAD_COVERAGE_GAP_SECONDS = 2.0
DEFAULT_VAD_MINIMUM_COVERAGE_RATIO = 0.95
WORD_TIMESTAMP_BOUNDARY_TOLERANCE_SECONDS = 0.5
WORD_TIMESTAMP_MONOTONIC_TOLERANCE_SECONDS = 0.05
WHISPER_WORD_ANOMALY_SOURCE = {
    "projects": ["openai/whisper", "SYSTRAN/faster-whisper"],
    "commits": [
        "04f449b8a437f1bbd3dba5c9f826aca972e7709a",
        "ed9a06cd89a93e47838f564998a6c09b655d7f43",
    ],
    "algorithm": "word_anomaly_score/is_segment_anomaly",
}
WHISPER_WORD_PROBABILITY_ANOMALY_BELOW = 0.15
WHISPER_WORD_SHORT_DURATION_SECONDS = 0.133
WHISPER_WORD_LONG_DURATION_SECONDS = 2.0
WHISPER_WORD_ANOMALY_SCORE_THRESHOLD = 3.0
WHISPER_WORD_ANOMALY_MAX_WORDS = 8
WHISPER_PUNCTUATION = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"


def assess_asr_response(
    payload: Any,
    *,
    task_instructions: str = "",
    asr_prompt: str = "",
    vad_intervals: list[dict[str, Any]] | None = None,
    media_duration_seconds: float | None = None,
    retry_overlap_seconds: float = DEFAULT_RETRY_OVERLAP_SECONDS,
) -> dict[str, Any]:
    """Assess verbose ASR output without discarding usable transcript text."""

    segments = _segments(payload)
    assessed: list[dict[str, Any]] = []
    failed_chunks: list[dict[str, Any]] = []
    review_chunks: list[dict[str, Any]] = []
    for position, raw in enumerate(segments, start=1):
        row = _assess_segment(
            raw,
            position=position,
            task_instructions=task_instructions,
            asr_prompt=asr_prompt,
        )
        assessed.append(row)
        if row["blocking"]:
            failed_chunks.append(_failed_chunk(row))
        elif row["issues"]:
            review_chunks.append(_failed_chunk(row))
    coarse_timing_density = _coarse_timing_density(
        assessed,
        vad_intervals or [],
    )
    review_chunks.extend(coarse_timing_density["review_chunks"])
    speech_coverage = _speech_coverage(
        assessed,
        vad_intervals or [],
        minimum_gap_seconds=DEFAULT_VAD_COVERAGE_GAP_SECONDS,
    )
    coverage_gaps = list(speech_coverage["gaps"])
    retry_chunks = [
        *failed_chunks,
        *(
            row
            for row in review_chunks
            if {
                "low_text_density",
                "whisper_word_anomaly",
                "coarse_timing_low_speech_text_density",
            }
            & set(row["reasons"])
        ),
        *coverage_gaps,
    ]

    retry_windows = _retry_windows(
        retry_chunks,
        vad_intervals=vad_intervals or [],
        media_duration_seconds=media_duration_seconds,
        overlap_seconds=retry_overlap_seconds,
    )
    missing_verbose_segments = not assessed
    status = (
        "failed"
        if missing_verbose_segments
        else (
            "degraded"
            if failed_chunks or coverage_gaps
            else ("review_required" if review_chunks else "passed")
        )
    )
    response_issues: list[dict[str, Any]] = []
    if missing_verbose_segments:
        response_issues.append(
            {
                "key": "verbose_segments_missing",
                "severity": "blocking",
                "detail": "ASR response did not contain segment metadata",
            }
        )
    if coverage_gaps:
        response_issues.append(
            {
                "key": "missing_speech_coverage",
                "severity": "blocking",
                "detail": (
                    f"{len(coverage_gaps)} VAD speech interval(s) have no usable "
                    "transcript coverage"
                ),
            }
        )
    return {
        "schema": SCHEMA,
        "status": status,
        "quality_gate_passed": bool(assessed)
        and not failed_chunks
        and not review_chunks
        and not coverage_gaps,
        "segment_count": len(assessed),
        "passed_segment_count": sum(not row["issues"] for row in assessed),
        "review_segment_count": len(review_chunks),
        "failed_segment_count": len(failed_chunks),
        "coverage_gap_count": len(coverage_gaps),
        "speech_coverage": speech_coverage,
        "coarse_timing_density": {
            key: value
            for key, value in coarse_timing_density.items()
            if key != "review_chunks"
        },
        "segments": assessed,
        "failed_chunks": failed_chunks,
        "review_chunks": review_chunks,
        "response_issues": response_issues,
        "retry_plan": {
            "status": "authorization_required" if retry_windows else "not_needed",
            "windows": retry_windows,
            "overlap_seconds": float(retry_overlap_seconds),
            "requires_new_exact_consent": bool(retry_windows),
            "exact_retry_artifact_hashes_required": bool(retry_windows),
            "execute": False,
            "remote_retry_allowed_without_consent": False,
            "silent_provider_fallback_allowed": False,
            "silent_location_fallback_allowed": False,
        },
        "thresholds": {
            "avg_logprob_warning_below": DEFAULT_AVG_LOGPROB_WARNING,
            "compression_ratio_block_above": DEFAULT_COMPRESSION_RATIO_MAX,
            "no_speech_prob_block_above": DEFAULT_NO_SPEECH_PROB_MAX,
            "long_segment_text_density_warning_below": DEFAULT_TEXT_DENSITY_MIN,
            "vad_uncovered_speech_gap_seconds": DEFAULT_VAD_COVERAGE_GAP_SECONDS,
            "whisper_word_probability_anomaly_below": WHISPER_WORD_PROBABILITY_ANOMALY_BELOW,
            "whisper_word_short_duration_seconds": WHISPER_WORD_SHORT_DURATION_SECONDS,
            "whisper_word_long_duration_seconds": WHISPER_WORD_LONG_DURATION_SECONDS,
            "whisper_word_anomaly_score": WHISPER_WORD_ANOMALY_SCORE_THRESHOLD,
            "whisper_word_anomaly_max_words": WHISPER_WORD_ANOMALY_MAX_WORDS,
        },
        "quality_signal_sources": [WHISPER_WORD_ANOMALY_SOURCE],
        "preservation": {
            "successful_segments_preserved": True,
            "flagged_original_text_preserved": True,
            "automatic_replacement": False,
        },
    }


def _segments(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    queue: list[dict[str, Any]] = [payload]
    visited: set[int] = set()
    wrapper_keys = (
        "model_result",
        "runtime_result",
        "raw_output",
        "raw_response",
        "response",
    )
    while queue:
        value = queue.pop(0)
        identity = id(value)
        if identity in visited:
            continue
        visited.add(identity)
        rows = value.get("segments")
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
        words = value.get("words")
        if isinstance(words, list) and words:
            rows = read_asr_segment_items(value)
            if rows:
                return rows
        for key in wrapper_keys:
            nested = value.get(key)
            if isinstance(nested, dict):
                queue.append(nested)
    return []


def _assess_segment(
    segment: dict[str, Any],
    *,
    position: int,
    task_instructions: str,
    asr_prompt: str,
) -> dict[str, Any]:
    metadata = (
        segment.get("metadata") if isinstance(segment.get("metadata"), dict) else {}
    )
    text = str(segment.get("text") or "").strip()
    start = _number(segment.get("start"), 0.0)
    end = max(start, _number(segment.get("end"), start))
    duration = max(0.0, end - start)
    avg_logprob = _optional_number(
        segment.get("avg_logprob", metadata.get("avg_logprob"))
    )
    compression_ratio = _optional_number(
        segment.get("compression_ratio", metadata.get("compression_ratio"))
    )
    no_speech_prob = _optional_number(
        segment.get("no_speech_prob", metadata.get("no_speech_prob"))
    )
    density = len(_compact_text(text)) / duration if duration > 0 else None
    timing_estimation = _timing_estimation(segment)
    timing_estimated = bool(timing_estimation)
    words = read_asr_word_timestamps(segment)
    if not words:
        words = read_asr_word_timestamps(metadata)
    coverage_evidence = _coverage_evidence(
        text,
        words,
        fallback_start=start,
        fallback_end=end,
    )
    word_anomaly = _whisper_word_anomaly_evidence(words)
    issues: list[dict[str, Any]] = []

    leaked = _instruction_overlap(text, task_instructions)
    if leaked:
        issues.append(
            {"key": "task_instruction_leak", "severity": "blocking", "detail": leaked}
        )
    prompt_leak = (
        _instruction_overlap(
            text,
            asr_prompt,
            min_candidate_chars=24,
            min_clause_chars=16,
            min_overlap_chars=24,
        )
        if len(_compact_text(asr_prompt)) >= 24
        else ""
    )
    if prompt_leak:
        issues.append(
            {"key": "asr_prompt_leak", "severity": "blocking", "detail": prompt_leak}
        )
    if avg_logprob is not None and avg_logprob < DEFAULT_AVG_LOGPROB_WARNING:
        issues.append(
            {"key": "low_average_logprob", "severity": "warning", "value": avg_logprob}
        )
    if (
        compression_ratio is not None
        and compression_ratio > DEFAULT_COMPRESSION_RATIO_MAX
    ):
        issues.append(
            {
                "key": "high_compression_ratio",
                "severity": "blocking",
                "value": compression_ratio,
            }
        )
    if (
        no_speech_prob is not None
        and no_speech_prob > DEFAULT_NO_SPEECH_PROB_MAX
        and (avg_logprob is None or avg_logprob < -0.5)
    ):
        issues.append(
            {
                "key": "probable_no_speech_decode",
                "severity": "blocking",
                "value": no_speech_prob,
            }
        )
    if (
        duration >= 10
        and density is not None
        and density < DEFAULT_TEXT_DENSITY_MIN
        and not timing_estimated
    ):
        issues.append(
            {
                "key": "low_text_density",
                "severity": "warning",
                "value": round(density, 6),
            }
        )
    if word_anomaly["status"] == "anomaly":
        issues.append(
            {
                "key": "whisper_word_anomaly",
                "severity": "warning",
                "value": word_anomaly["score"],
                "detail": "Whisper word probability/duration anomaly heuristic matched",
            }
        )
    issue_keys = {str(row["key"]) for row in issues}
    if {"low_average_logprob", "low_text_density"}.issubset(issue_keys):
        issues.append(
            {
                "key": "low_confidence_content_gap",
                "severity": "blocking",
                "detail": "low log probability and sparse text in a long segment",
            }
        )
    return {
        "segment_id": str(
            segment.get("id") or segment.get("segment_id") or f"segment-{position:04d}"
        ),
        "position": position,
        "start": start,
        "end": end,
        "duration_seconds": round(duration, 6),
        "text": text,
        "text_density_chars_per_second": round(density, 6)
        if density is not None
        else None,
        "timing_estimated": timing_estimated,
        "timing_estimation": timing_estimation,
        "avg_logprob": avg_logprob,
        "compression_ratio": compression_ratio,
        "no_speech_prob": no_speech_prob,
        "coverage_intervals": coverage_evidence["intervals"],
        "coverage_evidence": coverage_evidence["source"],
        "coverage_evidence_reason": coverage_evidence["reason"],
        "word_anomaly_evidence": word_anomaly,
        "issues": issues,
        "blocking": any(row.get("severity") == "blocking" for row in issues),
    }


def _timing_estimation(segment: dict[str, Any]) -> dict[str, Any]:
    """Identify coarse provider timing without treating it as speech duration.

    Intent: prevent silence inside an untimed source window from becoming a
    false ``low_text_density`` retry.
    Decision: consume the normalizer's existing transformation provenance
    instead of adding a second timing classifier.
    Reason: character-proportional chunk timing is useful for navigation but
    cannot measure speaking rate.
    Evidence: the corrected 2026-07-24 bundle dropped from 320/326 density
    warnings to zero while preserving the exact transcript text.
    Effective scope: text-density warnings only; VAD coverage, confidence,
    compression, prompt-leak, and word-anomaly checks remain unchanged.
    """

    metadata = (
        segment.get("metadata") if isinstance(segment.get("metadata"), dict) else {}
    )
    if segment.get("timing_estimated") is True or metadata.get("timing_estimated") is True:
        return {"type": "timing_estimation", "precision": "coarse"}
    for row in segment.get("transformations") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("type") or "").strip().lower() in {
            "timing_estimation",
            "estimated_timing",
        }:
            return dict(row)
    return {}


def _coarse_timing_density(
    assessed: list[dict[str, Any]],
    vad_intervals: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate untimed chunks against actual VAD speech seconds.

    Intent: retain a real missing-content signal after excluding synthetic
    character-proportional durations from segment density.
    Decision: group estimated cues by their recorded source window and reuse
    ``interval_coverage`` to measure source-bound VAD speech seconds.
    Reason: completely skipping density could let one character cover a
    speech-heavy five-minute chunk; using full chunk duration creates the
    opposite false positive when most of the chunk is silent.
    Evidence: 18 production chunks measure 3.425..6.172 chars/speech-second;
    faster-whisper and WhisperX likewise establish VAD windows before decode
    or alignment.
    Effective scope: estimated FunASR/SenseVoice windows with independent VAD;
    no VAD means ``not_evaluated`` and never creates an automatic retry.
    """

    groups: dict[tuple[float, float], list[dict[str, Any]]] = {}
    estimated_count = 0
    for row in assessed:
        timing = row.get("timing_estimation")
        if not isinstance(timing, dict) or not timing:
            continue
        estimated_count += 1
        start = _optional_number(timing.get("source_window_start"))
        end = _optional_number(timing.get("source_window_end"))
        if start is None or end is None or end <= start:
            continue
        groups.setdefault((start, end), []).append(row)
    if not estimated_count:
        return {
            "status": "not_applicable",
            "estimated_segment_count": 0,
            "evaluated_window_count": 0,
            "review_window_count": 0,
            "windows": [],
            "review_chunks": [],
        }
    if not vad_intervals or not groups:
        return {
            "status": "not_evaluated",
            "estimated_segment_count": estimated_count,
            "evaluated_window_count": 0,
            "review_window_count": 0,
            "reason": "source_bound_vad_or_source_window_missing",
            "windows": [],
            "review_chunks": [],
        }

    metrics: list[dict[str, Any]] = []
    review_chunks: list[dict[str, Any]] = []
    for position, ((start, end), rows) in enumerate(sorted(groups.items()), start=1):
        clipped_vad = [
            (
                max(start, _number(item.get("start"), start)),
                min(end, _number(item.get("end"), end)),
            )
            for item in vad_intervals
            if isinstance(item, dict)
            and _number(item.get("end"), 0.0) > start
            and _number(item.get("start"), 0.0) < end
        ]
        clipped_vad = [row for row in clipped_vad if row[1] > row[0]]
        coverage = interval_coverage(
            clipped_vad,
            [(start, end)],
            minimum_gap_seconds=0.0,
        )
        speech_seconds = float(coverage.get("target_seconds") or 0.0)
        text = "".join(str(row.get("text") or "") for row in rows)
        chars = len(_compact_text(text))
        density = (chars / speech_seconds) if speech_seconds > 0 else None
        needs_review = (
            density is not None and density < DEFAULT_TEXT_DENSITY_MIN
        )
        metric = {
            "source_window_start": round(start, 6),
            "source_window_end": round(end, 6),
            "segment_count": len(rows),
            "text_char_count": chars,
            "vad_speech_seconds": round(speech_seconds, 6),
            "text_chars_per_vad_speech_second": (
                round(density, 6) if density is not None else None
            ),
            "status": "review_required" if needs_review else "passed",
        }
        metrics.append(metric)
        if needs_review:
            review_chunks.append(
                {
                    "segment_id": f"coarse-window-{position:04d}",
                    "position": min(int(row["position"]) for row in rows),
                    "start": start,
                    "end": end,
                    "reasons": ["coarse_timing_low_speech_text_density"],
                    "original_text": text,
                    "preserve_original_text": True,
                }
            )
    return {
        "status": "review_required" if review_chunks else "passed",
        "estimated_segment_count": estimated_count,
        "evaluated_window_count": len(metrics),
        "review_window_count": len(review_chunks),
        "windows": metrics,
        "review_chunks": review_chunks,
    }


def _failed_chunk(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "segment_id": row["segment_id"],
        "position": row["position"],
        "start": row["start"],
        "end": row["end"],
        "reasons": [str(issue["key"]) for issue in row["issues"]],
        "original_text": row["text"],
        "preserve_original_text": True,
    }


def _retry_windows(
    failed_chunks: list[dict[str, Any]],
    *,
    vad_intervals: list[dict[str, Any]],
    media_duration_seconds: float | None,
    overlap_seconds: float,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    duration_limit = (
        max(0.0, float(media_duration_seconds))
        if media_duration_seconds is not None
        else None
    )
    clean_vad = [
        (_number(row.get("start"), 0.0), _number(row.get("end"), 0.0))
        for row in vad_intervals
        if isinstance(row, dict)
    ]
    for chunk in failed_chunks:
        source_start = float(chunk["start"])
        source_end = float(chunk["end"])
        start = max(0.0, source_start - float(overlap_seconds))
        end = source_end + float(overlap_seconds)
        is_coverage_gap = "missing_speech_coverage" in chunk.get("reasons", [])
        overlapping_vad = [
            interval
            for interval in clean_vad
            if interval[1] > start and interval[0] < end and interval[1] > interval[0]
        ]
        alignment = "provider_segment_boundary"
        if is_coverage_gap:
            alignment = "vad_coverage_gap"
        elif overlapping_vad:
            start = max(
                0.0,
                min(interval[0] for interval in overlapping_vad)
                - float(overlap_seconds),
            )
            end = max(interval[1] for interval in overlapping_vad) + float(
                overlap_seconds
            )
            alignment = "vad_boundary"
        if duration_limit is not None:
            end = min(duration_limit, end)
        windows.append(
            {
                "retry_id": f"retry-{len(windows) + 1:04d}",
                "source_segment_ids": [str(chunk["segment_id"])],
                "start": round(start, 6),
                "end": round(max(start, end), 6),
                "alignment_source": alignment,
                "reasons": list(chunk["reasons"]),
                "snippet_artifact_status": "not_created",
            }
        )
    return _merge_windows(windows)


def _merge_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for row in sorted(
        windows, key=lambda item: (float(item["start"]), float(item["end"]))
    ):
        if not merged or float(row["start"]) > float(merged[-1]["end"]):
            merged.append(dict(row))
            continue
        previous = merged[-1]
        previous["end"] = max(float(previous["end"]), float(row["end"]))
        previous["source_segment_ids"] = list(
            dict.fromkeys([*previous["source_segment_ids"], *row["source_segment_ids"]])
        )
        previous["reasons"] = list(
            dict.fromkeys([*previous["reasons"], *row["reasons"]])
        )
        if row["alignment_source"] == "vad_boundary":
            previous["alignment_source"] = "vad_boundary"
    for position, row in enumerate(merged, start=1):
        row["retry_id"] = f"retry-{position:04d}"
        row["end"] = round(float(row["end"]), 6)
    return merged


def _speech_coverage(
    assessed: list[dict[str, Any]],
    vad_intervals: list[dict[str, Any]],
    *,
    minimum_gap_seconds: float,
) -> dict[str, Any]:
    coverage = interval_coverage(
        [
            (_number(row.get("start"), 0.0), _number(row.get("end"), 0.0))
            for row in vad_intervals
            if isinstance(row, dict)
        ],
        [
            (float(interval["start"]), float(interval["end"]))
            for row in assessed
            if str(row.get("text") or "").strip() and not row.get("blocking")
            for interval in row.get("coverage_intervals") or []
        ],
        minimum_gap_seconds=minimum_gap_seconds,
        # Intent: reject materially incomplete speech without turning normal
        # word-timestamp spacing into thousands of retry windows on long media.
        # Decision: combine the established per-gap threshold with a 95%
        # relative coverage floor; do not add a fixed two-second cumulative
        # budget here.
        # Reason: a duration-independent cumulative budget necessarily fails
        # multi-hour transcripts whose individually harmless sub-two-second
        # word gaps add up, even when coverage stays above the production floor.
        # A zero-coverage set of short gaps still fails through the ratio gate,
        # while any single gap at or above the material threshold still fails.
        # Evidence: the 2026-08-31 3h46m bundle produced 3,774 sub-two-second
        # gaps, 95.046% coverage, and only one real 2.17s boundary omission.
        # Effective scope: independent VAD speech coverage only.
        minimum_coverage_ratio=DEFAULT_VAD_MINIMUM_COVERAGE_RATIO,
    )
    if not coverage["target_interval_count"]:
        return {
            "status": "not_evaluated",
            "vad_interval_count": 0,
            "speech_seconds": 0.0,
            "covered_seconds": 0.0,
            "coverage_ratio": None,
            "minimum_gap_seconds": float(minimum_gap_seconds),
            "gaps": [],
            "evidence": _coverage_evidence_summary(assessed),
        }
    uncovered_rows = (
        coverage["all_gaps"] if coverage["status"] == "degraded" else []
    )
    gaps = [
        _coverage_gap(float(row["start"]), float(row["end"]), position)
        for position, row in enumerate(uncovered_rows, start=1)
    ]
    return {
        "status": coverage["status"],
        "vad_interval_count": int(coverage["target_interval_count"]),
        "speech_seconds": float(coverage["target_seconds"]),
        "covered_seconds": float(coverage["covered_seconds"]),
        "coverage_ratio": coverage["coverage_ratio"],
        "minimum_gap_seconds": float(minimum_gap_seconds),
        "gaps": gaps,
        "evidence": _coverage_evidence_summary(assessed),
    }


def _coverage_evidence(
    text: str,
    words: list[dict[str, Any]],
    *,
    fallback_start: float,
    fallback_end: float,
) -> dict[str, Any]:
    """Prefer word timing only when its time identity is trustworthy.

    Intent: stop malformed word timing from covering speech outside its parent
    segment.
    Decision: validate bounds and monotonicity with small provider tolerances;
    invalid timing falls back to the unchanged segment bounds.
    Reason: a complete word-text match does not prove correct time alignment.
    Evidence: a 0-2 second segment with a word extending to 5 seconds hid a
    real 3-4 second VAD gap in the previous implementation.
    Effective scope: ASR completeness evidence and targeted retry planning;
    transcript text is never rewritten.
    """

    fallback = [{"start": fallback_start, "end": fallback_end}]
    intervals: list[dict[str, float]] = []
    previous_end: float | None = None
    timing_invalid = False
    for row in words:
        start = _optional_number(row.get("start"))
        end = _optional_number(row.get("end"))
        if start is None or end is None or end <= start:
            timing_invalid = True
            break
        if (
            start < fallback_start - WORD_TIMESTAMP_BOUNDARY_TOLERANCE_SECONDS
            or end > fallback_end + WORD_TIMESTAMP_BOUNDARY_TOLERANCE_SECONDS
            or (
                previous_end is not None
                and start
                < previous_end - WORD_TIMESTAMP_MONOTONIC_TOLERANCE_SECONDS
            )
        ):
            timing_invalid = True
            break
        intervals.append(
            {
                "start": max(fallback_start, start),
                "end": min(fallback_end, end),
            }
        )
        previous_end = end
    if timing_invalid:
        return {
            "intervals": fallback,
            "source": "segment_bounds",
            "reason": "word_timestamp_bounds_invalid",
        }
    if not intervals:
        return {
            "intervals": fallback,
            "source": "segment_bounds",
            "reason": "word_timestamps_missing_or_invalid",
        }
    normalized_text = _compact_text(text)
    normalized_words = _compact_text(
        "".join(str(row.get("word") or row.get("text") or "") for row in words)
    )
    if not normalized_text or normalized_words != normalized_text:
        return {
            "intervals": fallback,
            "source": "segment_bounds",
            "reason": "word_timestamp_text_incomplete",
        }
    return {
        "intervals": intervals,
        "source": "word_timestamps",
        "reason": "complete_word_text_match",
    }


def _coverage_evidence_summary(assessed: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "word_timestamp_segment_count": sum(
            row.get("coverage_evidence") == "word_timestamps" for row in assessed
        ),
        "segment_bounds_fallback_count": sum(
            row.get("coverage_evidence") == "segment_bounds" for row in assessed
        ),
        "incomplete_word_timestamp_segment_count": sum(
            row.get("coverage_evidence_reason") == "word_timestamp_text_incomplete"
            for row in assessed
        ),
        "word_timestamp_preferred_when_complete": True,
    }


def _whisper_word_anomaly_evidence(words: list[dict[str, Any]]) -> dict[str, Any]:
    """Adapt Whisper's word anomaly heuristic without treating absent scores as zero."""

    candidates = [
        row
        for row in words
        if str(row.get("word") or "").strip() not in WHISPER_PUNCTUATION
    ][:WHISPER_WORD_ANOMALY_MAX_WORDS]
    if not candidates:
        return {
            "status": "not_evaluated",
            "reason": "word_timestamps_missing",
            "word_count": 0,
            "source": WHISPER_WORD_ANOMALY_SOURCE,
        }
    if any(not _whisper_word_anomaly_input_complete(row) for row in candidates):
        return {
            "status": "not_evaluated",
            "reason": "word_confidence_or_timing_incomplete",
            "word_count": len(candidates),
            "source": WHISPER_WORD_ANOMALY_SOURCE,
        }
    scores = [_whisper_word_anomaly_score(row) for row in candidates]
    score = sum(scores)
    anomalous = score >= WHISPER_WORD_ANOMALY_SCORE_THRESHOLD or (
        score + 0.01 >= len(candidates)
    )
    return {
        "status": "anomaly" if anomalous else "passed",
        "reason": "whisper_word_anomaly_heuristic",
        "word_count": len(candidates),
        "score": round(score, 6),
        "per_word_scores": [round(value, 6) for value in scores],
        "source": WHISPER_WORD_ANOMALY_SOURCE,
    }


def _whisper_word_anomaly_input_complete(word: dict[str, Any]) -> bool:
    start = _optional_number(word.get("start"))
    end = _optional_number(word.get("end"))
    score = _optional_number(word.get("score"))
    return start is not None and end is not None and end >= start and score is not None


def _whisper_word_anomaly_score(word: dict[str, Any]) -> float:
    probability = float(word["score"])
    duration = float(word["end"]) - float(word["start"])
    score = 0.0
    if probability < WHISPER_WORD_PROBABILITY_ANOMALY_BELOW:
        score += 1.0
    if duration < WHISPER_WORD_SHORT_DURATION_SECONDS:
        score += (WHISPER_WORD_SHORT_DURATION_SECONDS - duration) * 15
    if duration > WHISPER_WORD_LONG_DURATION_SECONDS:
        score += duration - WHISPER_WORD_LONG_DURATION_SECONDS
    return score

def _coverage_gap(start: float, end: float, position: int) -> dict[str, Any]:
    return {
        "segment_id": f"vad-gap-{position:04d}",
        "position": 0,
        "start": round(float(start), 6),
        "end": round(float(end), 6),
        "duration_seconds": round(float(end) - float(start), 6),
        "reasons": ["missing_speech_coverage"],
        "original_text": "",
        "preserve_original_text": True,
    }

def _instruction_overlap(
    text: str,
    instruction: str,
    *,
    min_candidate_chars: int = 4,
    min_clause_chars: int = 6,
    min_overlap_chars: int = 6,
) -> str:
    candidate = _compact_text(text)
    source = _compact_text(instruction)
    if len(candidate) < min_candidate_chars or len(source) < min_clause_chars:
        return ""
    for raw_clause in re.split(r"[\r\n。！？!?；;，,]+", str(instruction or "")):
        clause = _compact_text(raw_clause)
        if len(clause) >= min_clause_chars and clause in candidate:
            start = candidate.find(clause)
            return candidate[start : start + len(clause)][:80]
    if candidate in source and len(candidate) >= min_overlap_chars:
        return candidate[:80]
    match = difflib.SequenceMatcher(
        a=candidate, b=source, autojunk=False
    ).find_longest_match()
    shorter = min(len(candidate), len(source))
    if match.size >= min_overlap_chars and match.size / max(1, shorter) >= 0.55:
        return candidate[match.a : match.a + match.size][:80]
    return ""


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any, default: float) -> float:
    number = _optional_number(value)
    return float(default if number is None else number)
