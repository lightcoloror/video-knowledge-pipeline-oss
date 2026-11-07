"""Fail-closed evidence audit for deciding whether one media file is redundant.

Intent:
    Turn the duplicate-media deletion checklist into an executable, read-only
    contract instead of relying on titles, durations, or operator memory.
Decision:
    Reuse VKP's FFmpeg/ffprobe resolver, transcript stability evaluator,
    transcript parser, canonical hashing, and atomic storage.  The module only
    adapts their evidence into five independent gates and never deletes media.
Reason:
    Re-encoded recordings can share a title and duration while retaining unique
    speech, slides, or provenance references.  A destructive decision therefore
    needs directional content coverage and explicit lineage checks.
Evidence:
    The 2026-07-29 insurance-course duplicate audit found one byte-near duplicate
    and five probable re-encodes; title/duration checks alone could not prove
    that the lower-quality copies contained no unique content.
Effective scope:
    Local media, transcript, visual-evidence, ASR-plan, and reference files.
    No model call, upload, registry mutation, provenance rewrite, or deletion.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import struct
import subprocess
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file
from .media_tools import local_tool_subprocess_env, resolve_media_tool
from .models import TranscriptCue, now_iso
from .numeric_normalization import number_evidence_map
from .storage import read_json, write_json, write_text_atomic
from .transcript import parse_transcript
from .transcript_stability_evaluation import evaluate_transcript_files


SCHEMA = "video_knowledge_pipeline.media_equivalence_audit.v1"
DEFAULT_AUDIO_COVERAGE = 0.995
DEFAULT_TRANSCRIPT_COVERAGE = 0.995
DEFAULT_VISUAL_COVERAGE = 1.0
DEFAULT_MAX_UNIQUE_SPEECH_SECONDS = 5.0
DEFAULT_FRAME_SIMILARITY = 0.70
DEFAULT_AUDIO_SIMILARITY = 0.90
DEFAULT_MAX_ALIGNMENT_SECONDS = 90.0
QUALITY_POLICY_PRACTICAL = "practical_course"
QUALITY_POLICY_ARCHIVAL = "archival_lossless"
QUALITY_POLICIES = (QUALITY_POLICY_PRACTICAL, QUALITY_POLICY_ARCHIVAL)
QUALITY_POLICY_SPACE_SAVING = "space_saving"
QUALITY_POLICY_ARCHIVE_LOSSLESS = "archive_lossless"
QUALITY_POLICY_ALIASES = {
    QUALITY_POLICY_SPACE_SAVING: QUALITY_POLICY_PRACTICAL,
    QUALITY_POLICY_ARCHIVE_LOSSLESS: QUALITY_POLICY_ARCHIVAL,
}
QUALITY_POLICY_INPUTS = QUALITY_POLICIES + tuple(QUALITY_POLICY_ALIASES)
DEFAULT_QUALITY_POLICY = QUALITY_POLICY_PRACTICAL
DEFAULT_USER_QUALITY_POLICY = QUALITY_POLICY_SPACE_SAVING

_TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".md",
    ".markdown",
    ".txt",
    ".yaml",
    ".yml",
    ".ps1",
}
_TEXT_KEYS = {
    "visual_text",
    "corrected_visual_text",
    "ocr_text",
    "screen_text",
    "slide_text",
    "title",
    "heading",
    "markdown",
}
_STRUCTURED_KEYS = {
    "structured_visual",
    "visual_understanding",
    "corrected_visual_understanding",
    "temporal_visual_understanding",
    "corrected_temporal_visual_understanding",
}
_HASH_KEYS = {
    "sha256",
    "artifact_sha256",
    "frame_sha256",
    "image_sha256",
    "source_sha256",
}
_TIME_START_KEYS = (
    "start",
    "start_seconds",
    "time_start",
    "timestamp_seconds",
    "seconds",
)
_TIME_END_KEYS = ("end", "end_seconds", "time_end")
_ASR_SIGNATURE_KEYS = {
    "provider",
    "preset",
    "runner",
    "backend",
    "model",
    "model_id",
    "model_revision",
    "language",
    "task",
    "batch_size",
    "batch_size_s",
    "chunk_seconds",
    "overlap_seconds",
    "vad",
    "vad_model",
    "vad_max_segment_seconds",
    "beam_size",
    "temperature",
    "compute_type",
    "device",
}
_VOLATILE_REFERENCE_NAMES = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
}


class MediaEquivalenceAuditError(ValueError):
    """Raised when the audit cannot safely interpret an input."""


def normalize_quality_policy(value: str) -> str:
    """Map user-facing storage presets onto the stable internal audit policy."""
    requested = str(value or "").strip()
    normalized = QUALITY_POLICY_ALIASES.get(requested, requested)
    if normalized not in QUALITY_POLICIES:
        raise ValueError(f"unsupported quality policy: {value}")
    return normalized


def audit_media_equivalence(
    candidate_media: str | Path,
    retained_media: str | Path,
    *,
    candidate_transcript: str | Path | None = None,
    retained_transcript: str | Path | None = None,
    candidate_asr_plan: str | Path | None = None,
    retained_asr_plan: str | Path | None = None,
    candidate_visual_evidence: Sequence[str | Path] = (),
    retained_visual_evidence: Sequence[str | Path] = (),
    reference_roots: Sequence[str | Path] = (),
    rebind_report: str | Path | None = None,
    critical_terms: Sequence[str] = (),
    quality_policy: str = DEFAULT_QUALITY_POLICY,
    audio_coverage_threshold: float = DEFAULT_AUDIO_COVERAGE,
    transcript_coverage_threshold: float = DEFAULT_TRANSCRIPT_COVERAGE,
    visual_coverage_threshold: float = DEFAULT_VISUAL_COVERAGE,
    max_unique_speech_seconds: float = DEFAULT_MAX_UNIQUE_SPEECH_SECONDS,
    min_audio_similarity: float = DEFAULT_AUDIO_SIMILARITY,
    frame_similarity_threshold: float = DEFAULT_FRAME_SIMILARITY,
    max_alignment_seconds: float = DEFAULT_MAX_ALIGNMENT_SECONDS,
) -> dict[str, Any]:
    """Run all deletion gates and return a non-destructive evidence report.

    Intent: provide one stable front door for all four user-approved hard gates.
    Decision: every gate is independent and required; unavailable evidence blocks
    deletion instead of being treated as a pass.
    Reason: audio equality cannot prove slide equality, and content equality
    cannot prove that existing Bundle/run-registry references were rebound.
    Evidence: the prior five re-encode candidates passed title/duration checks
    but still lacked transcript, visual, and provenance proof.
    Effective scope: report construction only; ``automatic_delete`` is always
    false and no input or reference file is modified.
    """

    requested_quality_policy = str(quality_policy or "").strip()
    quality_policy = normalize_quality_policy(requested_quality_policy)

    candidate = _existing_file(candidate_media, label="candidate_media")
    retained = _existing_file(retained_media, label="retained_media")
    if candidate == retained:
        raise MediaEquivalenceAuditError(
            "candidate_media and retained_media must differ"
        )

    candidate_probe = probe_media(candidate)
    retained_probe = probe_media(retained)
    audio_gate = compare_audio_content(
        candidate,
        retained,
        candidate_probe=candidate_probe,
        retained_probe=retained_probe,
        coverage_threshold=audio_coverage_threshold,
        max_unique_speech_seconds=max_unique_speech_seconds,
        min_similarity=min_audio_similarity,
        frame_similarity_threshold=frame_similarity_threshold,
        max_alignment_seconds=max_alignment_seconds,
    )
    transcript_gate = compare_transcript_content(
        candidate_transcript,
        retained_transcript,
        candidate_asr_plan=candidate_asr_plan,
        retained_asr_plan=retained_asr_plan,
        critical_terms=critical_terms,
        coverage_threshold=transcript_coverage_threshold,
        max_unique_speech_seconds=max_unique_speech_seconds,
    )
    visual_gate = compare_visual_evidence(
        candidate_visual_evidence,
        retained_visual_evidence,
        coverage_threshold=visual_coverage_threshold,
    )
    provenance_gate = audit_provenance_references(
        candidate,
        retained,
        reference_roots=reference_roots,
        rebind_report=rebind_report,
    )
    quality_gate = compare_retained_quality(
        candidate_probe, retained_probe, policy=quality_policy
    )
    gates = {
        "audio_content_containment": audio_gate,
        "transcript_content_containment": transcript_gate,
        "visual_ocr_scene_containment": visual_gate,
        "provenance_rebinding": provenance_gate,
        "retained_technical_quality": quality_gate,
    }
    content_gate_names = (
        "audio_content_containment",
        "transcript_content_containment",
        "visual_ocr_scene_containment",
        "provenance_rebinding",
    )
    content_equivalent = all(
        gates[name].get("status") == "passed" for name in content_gate_names
    )
    passed = content_equivalent and quality_gate.get("status") == "passed"
    incomplete = any(gate.get("status") == "unavailable" for gate in gates.values())
    quality_tradeoff = content_equivalent and quality_gate.get("status") != "passed"
    decision_category = (
        "content_equivalent_can_delete"
        if passed
        else "content_equivalent_quality_tradeoff"
        if quality_tradeoff
        else "evidence_incomplete"
        if incomplete
        else "possible_unique_content_or_policy_failure"
    )
    blockers = [
        {
            "gate": gate_name,
            "status": str(gate.get("status") or "failed"),
            "reasons": list(gate.get("reasons") or []),
        }
        for gate_name, gate in gates.items()
        if gate.get("status") != "passed"
    ]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "quality_policy": quality_policy,
        "requested_quality_policy": requested_quality_policy,
        "user_quality_policy": (
            QUALITY_POLICY_ARCHIVE_LOSSLESS
            if quality_policy == QUALITY_POLICY_ARCHIVAL
            else QUALITY_POLICY_SPACE_SAVING
        ),
        "decision_category": decision_category,
        "status": (
            "safe_to_delete_candidate"
            if passed
            else "quality_tradeoff_review"
            if quality_tradeoff
            else "incomplete"
            if incomplete
            else "blocked"
        ),
        "safe_to_delete": passed,
        "automatic_delete": False,
        "operator_confirmation_required": True,
        "candidate": _media_identity(candidate, candidate_probe),
        "retained": _media_identity(retained, retained_probe),
        "thresholds": {
            "candidate_audio_coverage_min": audio_coverage_threshold,
            "candidate_transcript_coverage_min": transcript_coverage_threshold,
            "candidate_visual_coverage_min": visual_coverage_threshold,
            "max_unique_speech_seconds": max_unique_speech_seconds,
            "min_audio_fingerprint_similarity": min_audio_similarity,
            "audio_frame_similarity_min": frame_similarity_threshold,
        },
        "gates": gates,
        "blockers": blockers,
        "required_actions": _required_actions(gates),
        "decision_contract": {
            "all_gates_required": True,
            "missing_evidence_fails_closed": True,
            "titles_or_similar_durations_are_not_equivalence_proof": True,
            "default_quality_policy": DEFAULT_QUALITY_POLICY,
            "default_user_quality_policy": DEFAULT_USER_QUALITY_POLICY,
            "active_quality_policy": quality_policy,
            "active_user_quality_policy": (
                QUALITY_POLICY_ARCHIVE_LOSSLESS
                if quality_policy == QUALITY_POLICY_ARCHIVAL
                else QUALITY_POLICY_SPACE_SAVING
            ),
            "quality_policy_aliases": dict(QUALITY_POLICY_ALIASES),
            "archival_lossless_available": True,
            "delete_operation_implemented": False,
        },
        "reuse": {
            "media_tools": "media_tools.resolve_media_tool/local_tool_subprocess_env",
            "audio_fingerprint": "FFmpeg chromaprint muxer",
            "speech_activity": "FFmpeg silencedetect filter",
            "transcript": "transcript_stability_evaluation.evaluate_transcript_files",
            "transcript_parser": "transcript.parse_transcript",
            "hashing": "file_hash.sha256_file/canonical_json.canonical_json_sha256",
            "storage": "storage.write_json/write_text_atomic",
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def probe_media(path: str | Path) -> dict[str, Any]:
    """Return stable technical metadata via VKP's registered ffprobe binary."""

    resolved = _existing_file(path, label="media")
    ffprobe = resolve_media_tool("ffprobe")
    if not ffprobe:
        raise MediaEquivalenceAuditError(
            "ffprobe is not available through VKP media tools"
        )
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        (
            "format=duration,bit_rate:"
            "stream=index,codec_type,codec_name,width,height,sample_rate,channels,bit_rate,avg_frame_rate"
        ),
        "-of",
        "json",
        str(resolved),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        env=local_tool_subprocess_env(),
        timeout=120,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise MediaEquivalenceAuditError(f"ffprobe failed for {resolved}: {message}")
    payload = json.loads(completed.stdout.decode("utf-8", errors="replace"))
    streams = payload.get("streams") if isinstance(payload, dict) else []
    format_row = payload.get("format") if isinstance(payload, dict) else {}
    video = next(
        (
            row
            for row in streams or []
            if isinstance(row, dict) and row.get("codec_type") == "video"
        ),
        {},
    )
    audio = next(
        (
            row
            for row in streams or []
            if isinstance(row, dict) and row.get("codec_type") == "audio"
        ),
        {},
    )
    return {
        "duration_seconds": _safe_float((format_row or {}).get("duration")),
        "container_bit_rate": _safe_int((format_row or {}).get("bit_rate")),
        "video": {
            "codec": str(video.get("codec_name") or ""),
            "width": _safe_int(video.get("width")),
            "height": _safe_int(video.get("height")),
            "bit_rate": _safe_int(video.get("bit_rate")),
            "fps": _safe_ratio(video.get("avg_frame_rate")),
        },
        "audio": {
            "codec": str(audio.get("codec_name") or ""),
            "sample_rate": _safe_int(audio.get("sample_rate")),
            "channels": _safe_int(audio.get("channels")),
            "bit_rate": _safe_int(audio.get("bit_rate")),
        },
    }


def compare_audio_content(
    candidate_media: str | Path,
    retained_media: str | Path,
    *,
    candidate_probe: Mapping[str, Any],
    retained_probe: Mapping[str, Any],
    coverage_threshold: float,
    max_unique_speech_seconds: float,
    min_similarity: float,
    frame_similarity_threshold: float,
    max_alignment_seconds: float,
) -> dict[str, Any]:
    """Compare aligned Chromaprint frames and voiced candidate intervals."""

    candidate = Path(candidate_media)
    retained = Path(retained_media)
    try:
        candidate_fp = extract_chromaprint(candidate)
        retained_fp = extract_chromaprint(retained)
        candidate_duration = float(candidate_probe.get("duration_seconds") or 0.0)
        retained_duration = float(retained_probe.get("duration_seconds") or 0.0)
        if (
            not candidate_fp
            or not retained_fp
            or candidate_duration <= 0
            or retained_duration <= 0
        ):
            raise MediaEquivalenceAuditError("audio fingerprint or duration is empty")
        voiced_intervals = detect_voiced_intervals(
            candidate, duration_seconds=candidate_duration
        )
        alignment = align_audio_fingerprints(
            candidate_fp,
            retained_fp,
            candidate_duration_seconds=candidate_duration,
            retained_duration_seconds=retained_duration,
            voiced_intervals=voiced_intervals,
            frame_similarity_threshold=frame_similarity_threshold,
            max_alignment_seconds=max_alignment_seconds,
        )
    except (
        MediaEquivalenceAuditError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        return {
            "status": "unavailable",
            "reasons": [f"audio_evidence_unavailable:{type(exc).__name__}"],
            "detail": str(exc),
        }
    reasons: list[str] = []
    if alignment["candidate_voiced_coverage"] < coverage_threshold:
        reasons.append("candidate_voiced_audio_coverage_below_threshold")
    if alignment["mean_similarity"] < min_similarity:
        reasons.append("audio_fingerprint_similarity_below_threshold")
    if alignment["longest_unmatched_voiced_run_seconds"] > max_unique_speech_seconds:
        reasons.append("candidate_has_unique_speech_over_limit")
    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "method": "ffmpeg_chromaprint_aligned_with_silencedetect",
        **alignment,
        "candidate_voiced_interval_count": len(voiced_intervals),
    }


def extract_chromaprint(path: str | Path) -> list[int]:
    """Extract raw 32-bit Chromaprint frames without creating an audio copy."""

    ffmpeg = resolve_media_tool("ffmpeg")
    if not ffmpeg:
        raise MediaEquivalenceAuditError(
            "ffmpeg is not available through VKP media tools"
        )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-f",
        "chromaprint",
        "-fp_format",
        "raw",
        "pipe:1",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        env=local_tool_subprocess_env(),
        timeout=900,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise MediaEquivalenceAuditError(f"chromaprint extraction failed: {message}")
    raw = bytes(completed.stdout)
    usable = len(raw) - (len(raw) % 4)
    return [value[0] for value in struct.iter_unpack("<I", raw[:usable])]


def detect_voiced_intervals(
    path: str | Path,
    *,
    duration_seconds: float,
    silence_db: float = -45.0,
    minimum_silence_seconds: float = 0.35,
) -> list[tuple[float, float]]:
    """Reuse FFmpeg silencedetect to delimit candidate non-silent evidence."""

    ffmpeg = resolve_media_tool("ffmpeg")
    if not ffmpeg:
        raise MediaEquivalenceAuditError(
            "ffmpeg is not available through VKP media tools"
        )
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-af",
        f"silencedetect=noise={silence_db}dB:d={minimum_silence_seconds}",
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        env=local_tool_subprocess_env(),
        timeout=900,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise MediaEquivalenceAuditError(f"silencedetect failed: {message}")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    events = [
        (match.group("kind"), float(match.group("value")))
        for match in re.finditer(
            r"silence_(?P<kind>start|end):\s*(?P<value>-?\d+(?:\.\d+)?)",
            stderr,
        )
    ]
    silent: list[tuple[float, float]] = []
    start: float | None = None
    for kind, value in events:
        if kind == "start":
            start = max(0.0, value)
        elif start is not None:
            silent.append((start, min(duration_seconds, max(start, value))))
            start = None
    if start is not None:
        silent.append((start, duration_seconds))
    voiced: list[tuple[float, float]] = []
    cursor = 0.0
    for silence_start, silence_end in sorted(silent):
        if silence_start > cursor:
            voiced.append((cursor, silence_start))
        cursor = max(cursor, silence_end)
    if cursor < duration_seconds:
        voiced.append((cursor, duration_seconds))
    return [(start, end) for start, end in voiced if end - start >= 0.05]


def align_audio_fingerprints(
    candidate: Sequence[int],
    retained: Sequence[int],
    *,
    candidate_duration_seconds: float,
    retained_duration_seconds: float,
    voiced_intervals: Sequence[tuple[float, float]],
    frame_similarity_threshold: float = DEFAULT_FRAME_SIMILARITY,
    max_alignment_seconds: float = DEFAULT_MAX_ALIGNMENT_SECONDS,
) -> dict[str, Any]:
    """Find a constant offset and measure directional voiced-frame coverage.

    The alignment is deliberately constant-offset only.  Edited or reordered
    recordings fail rather than being declared equivalent by a permissive
    dynamic-time-warp match.
    """

    if not candidate or not retained:
        raise MediaEquivalenceAuditError("fingerprints must not be empty")
    candidate_rate = len(candidate) / max(candidate_duration_seconds, 0.001)
    retained_rate = len(retained) / max(retained_duration_seconds, 0.001)
    frame_rate = (candidate_rate + retained_rate) / 2.0
    maximum_offset = min(
        max(len(candidate), len(retained)),
        max(1, int(round(max_alignment_seconds * frame_rate))),
    )
    coarse_step = max(1, int(round(frame_rate * 0.5)))
    sample_step = max(1, math.ceil(len(candidate) / 2000))
    coarse: list[tuple[float, int]] = []
    for offset in range(-maximum_offset, maximum_offset + 1, coarse_step):
        score = _offset_similarity(candidate, retained, offset, step=sample_step)
        coarse.append((score, offset))
    candidates: set[int] = set()
    for _, offset in sorted(coarse, reverse=True)[:8]:
        for refined in range(
            max(-maximum_offset, offset - coarse_step),
            min(maximum_offset, offset + coarse_step) + 1,
        ):
            candidates.add(refined)
    best_offset = max(
        candidates,
        key=lambda offset: _offset_similarity(candidate, retained, offset, step=1),
    )
    similarities: list[tuple[int, float]] = []
    for candidate_index, value in enumerate(candidate):
        retained_index = candidate_index + best_offset
        if 0 <= retained_index < len(retained):
            similarity = 1.0 - ((value ^ retained[retained_index]).bit_count() / 32.0)
        else:
            similarity = 0.0
        similarities.append((candidate_index, similarity))
    voiced_mask = [
        _time_in_intervals(
            (index + 0.5) / candidate_rate,
            voiced_intervals,
        )
        for index in range(len(candidate))
    ]
    voiced_count = sum(voiced_mask)
    matched_voiced = sum(
        voiced and similarity >= frame_similarity_threshold
        for voiced, (_, similarity) in zip(voiced_mask, similarities, strict=True)
    )
    coverage = matched_voiced / max(1, voiced_count)
    aligned_values = [similarity for _, similarity in similarities if similarity > 0.0]
    longest_run = _longest_unmatched_run_seconds(
        similarities,
        voiced_mask=voiced_mask,
        threshold=frame_similarity_threshold,
        frame_rate=candidate_rate,
    )
    return {
        "alignment_offset_fingerprint_frames": best_offset,
        "alignment_offset_seconds": round(best_offset / frame_rate, 6),
        "candidate_fingerprint_frames": len(candidate),
        "retained_fingerprint_frames": len(retained),
        "candidate_voiced_fingerprint_frames": voiced_count,
        "matched_candidate_voiced_fingerprint_frames": matched_voiced,
        "candidate_voiced_coverage": round(coverage, 8),
        "mean_similarity": round(
            sum(aligned_values) / max(1, len(aligned_values)),
            8,
        ),
        "longest_unmatched_voiced_run_seconds": round(longest_run, 6),
        "constant_offset_only": True,
    }


def compare_transcript_content(
    candidate_transcript: str | Path | None,
    retained_transcript: str | Path | None,
    *,
    candidate_asr_plan: str | Path | None,
    retained_asr_plan: str | Path | None,
    critical_terms: Sequence[str],
    coverage_threshold: float,
    max_unique_speech_seconds: float,
) -> dict[str, Any]:
    """Apply VKP transcript stability plus directional critical-content gates."""

    if not candidate_transcript or not retained_transcript:
        return {
            "status": "unavailable",
            "reasons": ["both_candidate_and_retained_transcripts_are_required"],
        }
    try:
        candidate_path = _existing_file(
            candidate_transcript, label="candidate_transcript"
        )
        retained_path = _existing_file(retained_transcript, label="retained_transcript")
        evaluation = evaluate_transcript_files(
            candidate_path,
            retained_path,
            max_normalized_reference_edit_distance=max(
                0.000001,
                1.0 - float(coverage_threshold),
            ),
            normalization_profile="strict_v1",
            require_reference_binding=False,
        )
        candidate_cues = parse_transcript(candidate_path)
        retained_cues = parse_transcript(retained_path)
        directional = _directional_transcript_coverage(candidate_cues, retained_cues)
        terms = _critical_term_evidence(candidate_cues, retained_cues, critical_terms)
        asr_config = compare_asr_plans(candidate_asr_plan, retained_asr_plan)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "status": "unavailable",
            "reasons": [f"transcript_evidence_unavailable:{type(exc).__name__}"],
            "detail": str(exc),
        }
    windows = evaluation.get("comparison_windows") or {}
    failed_windows = [
        row
        for row in windows.get("windows") or []
        if isinstance(row, dict) and not bool(row.get("passed"))
    ]
    longest_failed_window = max(
        (
            max(
                0.0,
                float(row.get("end_seconds") or 0.0)
                - float(row.get("start_seconds") or 0.0),
            )
            for row in failed_windows
        ),
        default=0.0,
    )
    longest_unique = float(directional["longest_unmatched_cue_run_seconds"])
    reasons: list[str] = []
    if directional["candidate_text_coverage"] < coverage_threshold:
        reasons.append("candidate_transcript_coverage_below_threshold")
    if longest_unique > max_unique_speech_seconds:
        reasons.append("candidate_has_unique_transcript_interval_over_limit")
    if terms["missing_numeric_token_count"]:
        reasons.append("candidate_numeric_tokens_missing_from_retained")
    if terms["missing_critical_term_count"]:
        reasons.append("candidate_critical_terms_missing_from_retained")
    if terms["missing_speaker_count"]:
        reasons.append("candidate_speakers_missing_from_retained")
    if asr_config["status"] != "passed":
        reasons.append("same_asr_configuration_not_proven")
    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "candidate_path": str(candidate_path),
        "candidate_sha256": sha256_file(candidate_path),
        "retained_path": str(retained_path),
        "retained_sha256": sha256_file(retained_path),
        "candidate_text_coverage": directional["candidate_text_coverage"],
        "longest_unmatched_cue_run_seconds": directional[
            "longest_unmatched_cue_run_seconds"
        ],
        "stability_status": str(evaluation.get("status") or "failed"),
        "normalized_reference_edit_distance": (evaluation.get("metric") or {}).get(
            "value"
        ),
        "failed_comparison_window_count": len(failed_windows),
        "longest_failed_comparison_window_seconds": round(longest_failed_window, 6),
        "critical_content": terms,
        "same_asr_configuration": asr_config,
        "reference_binding_used": False,
        "evaluation_only": True,
    }


def compare_asr_plans(
    candidate_plan: str | Path | None,
    retained_plan: str | Path | None,
) -> dict[str, Any]:
    """Prove both transcripts used the same model and decoding parameters."""

    if not candidate_plan or not retained_plan:
        return {
            "status": "unavailable",
            "reasons": ["both_asr_execution_plans_are_required"],
        }
    candidate_path = _existing_file(candidate_plan, label="candidate_asr_plan")
    retained_path = _existing_file(retained_plan, label="retained_asr_plan")
    candidate_payload = _read_json_like(candidate_path)
    retained_payload = _read_json_like(retained_path)
    candidate_signature = _asr_signature(candidate_payload)
    retained_signature = _asr_signature(retained_payload)
    if not candidate_signature or not retained_signature:
        return {
            "status": "unavailable",
            "reasons": ["asr_plan_has_no_recognized_stable_parameters"],
            "candidate_plan": str(candidate_path),
            "retained_plan": str(retained_path),
        }
    same = candidate_signature == retained_signature
    return {
        "status": "passed" if same else "failed",
        "reasons": [] if same else ["asr_execution_signature_mismatch"],
        "candidate_plan": str(candidate_path),
        "retained_plan": str(retained_path),
        "candidate_signature_sha256": canonical_json_sha256(candidate_signature),
        "retained_signature_sha256": canonical_json_sha256(retained_signature),
        "recognized_parameter_count": len(candidate_signature),
        "raw_secret_or_command_exposed": False,
    }


def compare_visual_evidence(
    candidate_sources: Sequence[str | Path],
    retained_sources: Sequence[str | Path],
    *,
    coverage_threshold: float,
) -> dict[str, Any]:
    """Compare OCR/PPT, representative-frame hashes, and scene evidence."""

    if not candidate_sources or not retained_sources:
        return {
            "status": "unavailable",
            "reasons": ["candidate_and_retained_visual_evidence_are_required"],
        }
    try:
        candidate_records = _load_visual_records(candidate_sources)
        retained_records = _load_visual_records(retained_sources)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "status": "unavailable",
            "reasons": [f"visual_evidence_unavailable:{type(exc).__name__}"],
            "detail": str(exc),
        }
    if not candidate_records or not retained_records:
        return {
            "status": "unavailable",
            "reasons": ["visual_evidence_contains_no_comparable_records"],
            "candidate_record_count": len(candidate_records),
            "retained_record_count": len(retained_records),
        }
    candidate_kinds = {
        kind for row in candidate_records for kind in row["evidence_kinds"]
    }
    retained_kinds = {
        kind for row in retained_records for kind in row["evidence_kinds"]
    }
    content_kinds = {
        "ocr_or_ppt_text",
        "structured_visual",
        "representative_frame_hash",
    }
    evidence_reasons: list[str] = []
    if "scene" not in candidate_kinds:
        evidence_reasons.append("candidate_scene_evidence_missing")
    if "scene" not in retained_kinds:
        evidence_reasons.append("retained_scene_evidence_missing")
    if not candidate_kinds & content_kinds:
        evidence_reasons.append("candidate_visual_content_evidence_missing")
    if not retained_kinds & content_kinds:
        evidence_reasons.append("retained_visual_content_evidence_missing")
    matched: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for record in candidate_records:
        score, retained_record = _best_visual_match(record, retained_records)
        if retained_record is not None and score >= 0.92:
            matched.append(
                {"candidate_id": record["id"], "match_score": round(score, 6)}
            )
        else:
            unmatched.append(
                {
                    "candidate_id": record["id"],
                    "time_range": record["time_range"],
                    "evidence_kinds": record["evidence_kinds"],
                    "best_match_score": round(score, 6),
                }
            )
    coverage = len(matched) / max(1, len(candidate_records))
    reasons = []
    if coverage < coverage_threshold:
        reasons.append("candidate_visual_evidence_coverage_below_threshold")
    if unmatched:
        reasons.append("candidate_has_unique_ocr_ppt_frame_or_scene_evidence")
    coverage_failed = bool(reasons)
    reasons.extend(evidence_reasons)
    status = (
        "failed"
        if coverage_failed
        else ("unavailable" if evidence_reasons else "passed")
    )
    return {
        "status": status,
        "reasons": reasons,
        "candidate_record_count": len(candidate_records),
        "retained_record_count": len(retained_records),
        "candidate_evidence_kinds": sorted(candidate_kinds),
        "retained_evidence_kinds": sorted(retained_kinds),
        "matched_candidate_record_count": len(matched),
        "candidate_visual_coverage": round(coverage, 8),
        "unmatched_candidate_record_count": len(unmatched),
        "unmatched_candidate_records": unmatched[:100],
        "unmatched_candidate_records_truncated": len(unmatched) > 100,
        "unmatched_candidate_records_sha256": canonical_json_sha256(unmatched),
        "candidate_sources": [
            _artifact_identity(Path(value)) for value in candidate_sources
        ],
        "retained_sources": [
            _artifact_identity(Path(value)) for value in retained_sources
        ],
        "content_text_included": False,
    }


def audit_provenance_references(
    candidate_media: str | Path,
    retained_media: str | Path,
    *,
    reference_roots: Sequence[str | Path],
    rebind_report: str | Path | None,
) -> dict[str, Any]:
    """Find path/hash references and require an explicit verified rebind report."""

    candidate = Path(candidate_media).resolve()
    retained = Path(retained_media).resolve()
    if not reference_roots:
        return {
            "status": "unavailable",
            "reasons": ["at_least_one_reference_root_is_required"],
        }
    candidate_digest = sha256_file(candidate)
    retained_digest = sha256_file(retained)
    needles = {
        str(candidate),
        json.dumps(str(candidate), ensure_ascii=False)[1:-1],
        str(candidate).replace("\\", "/"),
        candidate_digest,
    }
    references: list[dict[str, Any]] = []
    scanned_files = 0
    for root_value in reference_roots:
        root = Path(root_value).expanduser().resolve()
        if not root.exists():
            continue
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file() or path.suffix.casefold() not in _TEXT_SUFFIXES:
                continue
            if any(part in _VOLATILE_REFERENCE_NAMES for part in path.parts):
                continue
            scanned_files += 1
            try:
                text = path.read_text(encoding="utf-8-sig", errors="ignore")
            except OSError:
                continue
            matches = [needle for needle in needles if needle and needle in text]
            if matches:
                references.append(
                    {
                        "path": str(path),
                        "matched_candidate_path": any(
                            candidate.name in value for value in matches
                        ),
                        "matched_candidate_sha256": candidate_digest in matches,
                    }
                )
    rebind = _validate_rebind_report(
        rebind_report,
        candidate_sha256=candidate_digest,
        retained_sha256=retained_digest,
        references=references,
    )
    reasons: list[str] = []
    if references and rebind["status"] != "passed":
        reasons.append("candidate_is_still_referenced")
    if not references and rebind_report and rebind["status"] == "failed":
        reasons.append("invalid_provenance_rebind_report")
    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "reference_roots": [
            str(Path(value).expanduser().resolve()) for value in reference_roots
        ],
        "scanned_text_file_count": scanned_files,
        "candidate_reference_count": len(references),
        "candidate_references": references,
        "rebind": rebind,
    }


def compare_retained_quality(
    candidate_probe: Mapping[str, Any],
    retained_probe: Mapping[str, Any],
    *,
    policy: str = DEFAULT_QUALITY_POLICY,
) -> dict[str, Any]:
    """Apply practical course preservation or archival no-downgrade policy.

    Intent: save local storage by default without treating every container
    metric as equally important for lecture content.
    Decision: practical mode keeps duration, resolution, speech floors, and
    channels as hard checks; isolated fps/sample-rate tradeoffs are reported
    but do not block. Archival mode blocks any downgrade in measured stream
    metadata, including codecs and separate audio/video bit rates.
    Reason: 44.1 kHz high-bitrate speech can be better than 48 kHz low-bitrate
    speech, while 1080p lecture slides can be more useful than 720p60 motion.
    Evidence: the five real re-encode pairs exposed exactly these tradeoffs.
    Effective scope: the technical-quality gate only; all content and lineage
    gates remain mandatory and no file is deleted automatically.
    """

    if policy not in QUALITY_POLICIES:
        raise ValueError(f"unsupported quality policy: {policy}")
    candidate_video = candidate_probe.get("video") or {}
    retained_video = retained_probe.get("video") or {}
    candidate_audio = candidate_probe.get("audio") or {}
    retained_audio = retained_probe.get("audio") or {}
    candidate_duration = float(candidate_probe.get("duration_seconds") or 0.0)
    retained_duration = float(retained_probe.get("duration_seconds") or 0.0)
    candidate_fps = float(candidate_video.get("fps") or 0.0)
    retained_fps = float(retained_video.get("fps") or 0.0)
    candidate_video_bit_rate = int(candidate_video.get("bit_rate") or 0)
    retained_video_bit_rate = int(retained_video.get("bit_rate") or 0)
    candidate_audio_rate = int(candidate_audio.get("sample_rate") or 0)
    retained_audio_rate = int(retained_audio.get("sample_rate") or 0)
    candidate_audio_bit_rate = int(candidate_audio.get("bit_rate") or 0)
    retained_audio_bit_rate = int(retained_audio.get("bit_rate") or 0)

    archival_checks = {
        "duration_not_shorter": retained_duration + 0.25 >= candidate_duration,
        "pixel_count_not_lower": (
            int(retained_video.get("width") or 0)
            * int(retained_video.get("height") or 0)
            >= int(candidate_video.get("width") or 0)
            * int(candidate_video.get("height") or 0)
        ),
        "video_codec_unchanged": retained_video.get("codec")
        == candidate_video.get("codec"),
        "frame_rate_not_lower": retained_fps >= candidate_fps,
        "video_bit_rate_not_lower": retained_video_bit_rate >= candidate_video_bit_rate,
        "container_bit_rate_not_lower": int(
            retained_probe.get("container_bit_rate") or 0
        )
        >= int(candidate_probe.get("container_bit_rate") or 0),
        "audio_codec_unchanged": retained_audio.get("codec")
        == candidate_audio.get("codec"),
        "audio_sample_rate_not_lower": retained_audio_rate >= candidate_audio_rate,
        "audio_bit_rate_not_lower": retained_audio_bit_rate >= candidate_audio_bit_rate,
        "audio_channels_not_lower": int(retained_audio.get("channels") or 0)
        >= int(candidate_audio.get("channels") or 0),
    }
    frame_rate_ratio = retained_fps / max(candidate_fps, 0.000001)
    audio_bit_rate_ratio = retained_audio_bit_rate / max(candidate_audio_bit_rate, 1)
    tradeoffs = [
        name
        for name, changed in {
            "frame_rate_lower": retained_fps < candidate_fps,
            "video_bit_rate_lower": retained_video_bit_rate < candidate_video_bit_rate,
            "container_bit_rate_lower": int(
                retained_probe.get("container_bit_rate") or 0
            )
            < int(candidate_probe.get("container_bit_rate") or 0),
            "audio_sample_rate_lower": retained_audio_rate < candidate_audio_rate,
            "audio_bit_rate_lower": retained_audio_bit_rate < candidate_audio_bit_rate,
        }.items()
        if changed
    ]

    if policy == QUALITY_POLICY_ARCHIVAL:
        reasons = [name for name, passed in archival_checks.items() if not passed]
        return {
            "status": "passed" if not reasons else "failed",
            "policy": policy,
            "reasons": reasons,
            "checks": archival_checks,
            "non_blocking_tradeoffs": [],
            "strict_technical_quality_gate": True,
        }

    practical_checks = {
        "duration_not_shorter": retained_duration + 1.0 >= candidate_duration,
        "pixel_count_not_lower": archival_checks["pixel_count_not_lower"],
        "minimum_course_frame_rate": retained_fps >= min(candidate_fps, 5.0),
        "minimum_speech_sample_rate": retained_audio_rate
        >= min(candidate_audio_rate, 44_100),
        "minimum_speech_audio_bit_rate": retained_audio_bit_rate
        >= min(candidate_audio_bit_rate, 64_000),
        "audio_channels_not_lower": archival_checks["audio_channels_not_lower"],
    }
    reasons = [name for name, passed in practical_checks.items() if not passed]
    combined_material_tradeoff = (
        candidate_fps > 0
        and candidate_audio_bit_rate > 0
        and frame_rate_ratio < 0.60
        and audio_bit_rate_ratio < 0.60
    )
    if not reasons and combined_material_tradeoff:
        reasons.append("combined_motion_and_audio_quality_tradeoff")
        status = "review_required"
    else:
        status = "passed" if not reasons else "failed"
    return {
        "status": status,
        "policy": policy,
        "reasons": reasons,
        "checks": practical_checks,
        "archival_checks": archival_checks,
        "non_blocking_tradeoffs": tradeoffs if status == "passed" else [],
        "human_review_tradeoffs": tradeoffs if status == "review_required" else [],
        "metrics": {
            "frame_rate_ratio": round(frame_rate_ratio, 6),
            "audio_bit_rate_ratio": round(audio_bit_rate_ratio, 6),
            "minimum_course_frame_rate": min(candidate_fps, 5.0),
            "minimum_speech_sample_rate": min(candidate_audio_rate, 44_100),
            "minimum_speech_audio_bit_rate": min(candidate_audio_bit_rate, 64_000),
        },
        "strict_technical_quality_gate": False,
    }


def render_media_equivalence_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact operator report without transcript or OCR contents."""

    lines = [
        "# 媒体等价性删除审计",
        "",
        f"- 状态：`{report.get('status')}`",
        f"- 判定策略：`{report.get('quality_policy')}`",
        f"- 判定类别：`{report.get('decision_category')}`",
        f"- 可删除候选：`{str(bool(report.get('safe_to_delete'))).lower()}`",
        "- 自动删除：`false`",
        f"- 候选：`{(report.get('candidate') or {}).get('path', '')}`",
        f"- 保留：`{(report.get('retained') or {}).get('path', '')}`",
        "",
        "## 硬门",
        "",
    ]
    for name, gate in (report.get("gates") or {}).items():
        reasons = "、".join(str(value) for value in gate.get("reasons") or []) or "无"
        lines.append(f"- {name}：`{gate.get('status')}`；原因：{reasons}")
        tradeoffs = "、".join(
            str(value) for value in gate.get("non_blocking_tradeoffs") or []
        )
        if tradeoffs:
            lines.append(f"  - 非阻断质量取舍：{tradeoffs}")
    lines.extend(["", "## 后续动作", ""])
    actions = report.get("required_actions") or []
    lines.extend(f"- {value}" for value in actions)
    if not actions:
        lines.append("- 人工复核报告后，另行执行删除。")
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 本报告不执行删除、不改写 Bundle、不迁移引用。",
            "- 标题相似、时长相近或低码率不能单独证明内容等价。",
            "",
        ]
    )
    return "\n".join(lines)


def _directional_transcript_coverage(
    candidate: Sequence[TranscriptCue],
    retained: Sequence[TranscriptCue],
) -> dict[str, Any]:
    candidate_text = _normalize_text("".join(cue.text for cue in candidate))
    retained_text = _normalize_text("".join(cue.text for cue in retained))
    matcher = difflib.SequenceMatcher(
        None, candidate_text, retained_text, autojunk=False
    )
    matching = sum(block.size for block in matcher.get_matching_blocks())
    unmatched_flags: list[bool] = []
    for cue in candidate:
        cue_text = _normalize_text(cue.text)
        overlapping = "".join(
            other.text
            for other in retained
            if other.end >= cue.start - 2.0 and other.start <= cue.end + 2.0
        )
        ratio = (
            difflib.SequenceMatcher(
                None,
                cue_text,
                _normalize_text(overlapping),
                autojunk=False,
            ).ratio()
            if cue_text and overlapping
            else 0.0
        )
        unmatched_flags.append(ratio < 0.80)
    longest = 0.0
    run_start: float | None = None
    run_end = 0.0
    for cue, unmatched in zip(candidate, unmatched_flags, strict=True):
        if unmatched:
            if run_start is None:
                run_start = cue.start
            run_end = max(run_end, cue.end)
        elif run_start is not None:
            longest = max(longest, run_end - run_start)
            run_start = None
            run_end = 0.0
    if run_start is not None:
        longest = max(longest, run_end - run_start)
    return {
        "candidate_text_coverage": round(matching / max(1, len(candidate_text)), 8),
        "longest_unmatched_cue_run_seconds": round(max(0.0, longest), 6),
    }


def _critical_term_evidence(
    candidate: Sequence[TranscriptCue],
    retained: Sequence[TranscriptCue],
    critical_terms: Sequence[str],
) -> dict[str, Any]:
    candidate_text = _normalize_text("".join(cue.text for cue in candidate))
    retained_text = _normalize_text("".join(cue.text for cue in retained))
    numbers = sorted(number_evidence_map(candidate_text))
    retained_numbers = number_evidence_map(retained_text)
    missing_numbers = [value for value in numbers if value not in retained_numbers]
    normalized_terms = sorted(
        {
            normalized
            for value in critical_terms
            if (normalized := _normalize_text(value)) and normalized in candidate_text
        }
    )
    missing_terms = [value for value in normalized_terms if value not in retained_text]
    candidate_speakers = {cue.speaker for cue in candidate if cue.speaker}
    retained_speakers = {cue.speaker for cue in retained if cue.speaker}
    missing_speakers = candidate_speakers - retained_speakers
    return {
        "candidate_numeric_token_count": len(numbers),
        "missing_numeric_token_count": len(missing_numbers),
        "missing_numeric_token_sha256": _string_set_sha256(missing_numbers),
        "candidate_critical_term_count": len(normalized_terms),
        "missing_critical_term_count": len(missing_terms),
        "missing_critical_term_sha256": _string_set_sha256(missing_terms),
        "candidate_speaker_count": len(candidate_speakers),
        "missing_speaker_count": len(missing_speakers),
        "missing_speaker_sha256": _string_set_sha256(missing_speakers),
        "raw_values_included": False,
    }


def _load_visual_records(sources: Sequence[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_value in sources:
        source = _existing_file(source_value, label="visual_evidence")
        payload = _read_json_like(source)
        rows = _visual_rows(payload)
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            text_parts: list[str] = []
            hashes: set[str] = set()
            kinds: set[str] = set()
            for key, value in row.items():
                normalized_key = str(key).casefold()
                if normalized_key in _TEXT_KEYS:
                    text_parts.extend(_flatten_text(value))
                    kinds.add("ocr_or_ppt_text")
                elif normalized_key in _STRUCTURED_KEYS:
                    text_parts.extend(_flatten_text(value))
                    kinds.add("structured_visual")
                elif normalized_key in _HASH_KEYS and _looks_like_sha256(value):
                    hashes.add(str(value).casefold())
                    kinds.add("representative_frame_hash")
            scene_tokens = [
                str(row.get(key) or "").strip()
                for key in (
                    "scene_id",
                    "shot_id",
                    "boundary_type",
                    "transition_type",
                    "scene_type",
                )
                if str(row.get(key) or "").strip()
            ]
            if (
                not scene_tokens
                and "index" in row
                and any(key in row for key in ("start", "end", "seconds"))
            ):
                kinds.add("scene")
            if scene_tokens:
                kinds.add("scene")
            text = _normalize_text(" ".join(text_parts))
            if not text and not hashes and "scene" not in kinds:
                continue
            records.append(
                {
                    "id": str(
                        row.get("id")
                        or row.get("timeline_id")
                        or row.get("scene_id")
                        or f"{source.name}:{index:06d}"
                    ),
                    "text": text,
                    "hashes": sorted(hashes),
                    "scene_tokens": sorted(scene_tokens),
                    "time_range": _time_range(row),
                    "evidence_kinds": sorted(kinds),
                }
            )
    return records


def _visual_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in (
        "timeline",
        "items",
        "segments",
        "scenes",
        "boundaries",
        "records",
        "evidence",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return [payload]


def _best_visual_match(
    candidate: Mapping[str, Any],
    retained: Sequence[Mapping[str, Any]],
) -> tuple[float, Mapping[str, Any] | None]:
    best_score = 0.0
    best: Mapping[str, Any] | None = None
    candidate_hashes = set(candidate.get("hashes") or [])
    candidate_text = str(candidate.get("text") or "")
    candidate_scenes = set(candidate.get("scene_tokens") or [])
    candidate_kinds = set(candidate.get("evidence_kinds") or [])
    candidate_time = candidate.get("time_range") or {}
    for row in retained:
        score_parts: list[float] = []
        row_hashes = set(row.get("hashes") or [])
        row_text = str(row.get("text") or "")
        row_scenes = set(row.get("scene_tokens") or [])
        row_kinds = set(row.get("evidence_kinds") or [])
        row_time = row.get("time_range") or {}
        if candidate_hashes:
            score_parts.append(1.0 if candidate_hashes & row_hashes else 0.0)
        if candidate_text:
            score_parts.append(
                difflib.SequenceMatcher(
                    None, candidate_text, row_text, autojunk=False
                ).ratio()
            )
        if candidate_scenes:
            score_parts.append(
                len(candidate_scenes & row_scenes) / max(1, len(candidate_scenes))
            )
        if "scene" in candidate_kinds:
            candidate_start = candidate_time.get("start_seconds")
            candidate_end = candidate_time.get("end_seconds")
            row_start = row_time.get("start_seconds")
            row_end = row_time.get("end_seconds")
            deltas = [
                abs(float(left) - float(right))
                for left, right in (
                    (candidate_start, row_start),
                    (candidate_end, row_end),
                )
                if left is not None and right is not None
            ]
            score_parts.append(
                1.0 if "scene" in row_kinds and deltas and max(deltas) <= 2.0 else 0.0
            )
        score = sum(score_parts) / max(1, len(score_parts))
        if score > best_score:
            best_score = score
            best = row
    return best_score, best


def _validate_rebind_report(
    value: str | Path | None,
    *,
    candidate_sha256: str,
    retained_sha256: str,
    references: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not value:
        return {
            "status": "not_required" if not references else "unavailable",
            "reasons": [] if not references else ["rebind_report_required"],
        }
    path = _existing_file(value, label="rebind_report")
    payload = read_json(path)
    if not isinstance(payload, dict):
        return {"status": "failed", "reasons": ["rebind_report_must_be_object"]}
    checks = {
        "candidate_sha256_matches": str(
            payload.get("candidate_sha256") or ""
        ).casefold()
        == candidate_sha256.casefold(),
        "retained_sha256_matches": str(payload.get("retained_sha256") or "").casefold()
        == retained_sha256.casefold(),
        "all_references_updated": bool(payload.get("all_references_updated")),
        "validation_status_passed": str(payload.get("status") or "").casefold()
        in {"passed", "completed", "valid"},
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "path": str(path),
        "sha256": sha256_file(path),
        "checks": checks,
    }


def _asr_signature(payload: Any) -> dict[str, Any]:
    signature: dict[str, Any] = {}

    def visit(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).casefold()
                if not prefix and normalized in {
                    "availability",
                    "model_ready",
                    "model_readiness",
                }:
                    continue
                path = f"{prefix}.{normalized}" if prefix else normalized
                if normalized in _ASR_SIGNATURE_KEYS and isinstance(
                    child, (str, int, float, bool, type(None))
                ):
                    signature[path] = child
                elif isinstance(child, (dict, list)):
                    visit(child, path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, (dict, list)):
                    visit(child, f"{prefix}[{index}]")

    visit(payload)
    return signature


def _read_json_like(path: Path) -> Any:
    if path.suffix.casefold() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    return read_json(path)


def _offset_similarity(
    candidate: Sequence[int],
    retained: Sequence[int],
    offset: int,
    *,
    step: int,
) -> float:
    start = max(0, -offset)
    end = min(len(candidate), len(retained) - offset)
    if end <= start:
        return -1.0
    total = 0.0
    count = 0
    for index in range(start, end, max(1, step)):
        total += 1.0 - (
            (candidate[index] ^ retained[index + offset]).bit_count() / 32.0
        )
        count += 1
    overlap = max(0, end - start) / max(1, len(candidate))
    return (total / max(1, count)) * overlap


def _longest_unmatched_run_seconds(
    similarities: Sequence[tuple[int, float]],
    *,
    voiced_mask: Sequence[bool],
    threshold: float,
    frame_rate: float,
) -> float:
    longest = 0
    current = 0
    for voiced, (_, similarity) in zip(voiced_mask, similarities, strict=True):
        if voiced and similarity < threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest / max(frame_rate, 0.001)


def _time_in_intervals(value: float, intervals: Sequence[tuple[float, float]]) -> bool:
    return any(start <= value <= end for start, end in intervals)


def _flatten_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        parts: list[str] = []
        for child in value.values():
            parts.extend(_flatten_text(child))
        return parts
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts = []
        for child in value:
            parts.extend(_flatten_text(child))
        return parts
    return []


def _time_range(row: Mapping[str, Any]) -> dict[str, float | None]:
    start = next(
        (_safe_float(row.get(key)) for key in _TIME_START_KEYS if key in row), None
    )
    end = next(
        (_safe_float(row.get(key)) for key in _TIME_END_KEYS if key in row), None
    )
    return {"start_seconds": start, "end_seconds": end}


def _required_actions(gates: Mapping[str, Mapping[str, Any]]) -> list[str]:
    actions: list[str] = []
    mapping = {
        "audio_content_containment": "补齐或复核整段 Chromaprint/非静音覆盖证据。",
        "transcript_content_containment": "用相同本地 ASR 参数生成两版候选稿，并复核缺口、数字、术语和说话人。",
        "visual_ocr_scene_containment": "补齐两版场景、代表帧和 OCR/PPT 证据，人工复核未匹配区间。",
        "provenance_rebinding": "将现有 Bundle/run/评测引用安全迁移到保留版并生成 rebind 报告。",
        "retained_technical_quality": "按当前策略选择保留版；质量取舍项需人工确认，或切换 archival_lossless。",
    }
    for name, gate in gates.items():
        if gate.get("status") != "passed":
            actions.append(mapping[name])
    return actions


def _media_identity(path: Path, probe: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "probe": dict(probe),
    }


def _artifact_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _existing_file(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _looks_like_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(value or "").strip()))


def _string_set_sha256(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest() if payload else ""


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _safe_ratio(value: Any) -> float:
    text = str(value or "").strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            return float(numerator) / max(float(denominator), 0.000001)
        except (TypeError, ValueError):
            return 0.0
    return _safe_float(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only hard-gate audit before deleting a probable media duplicate."
    )
    parser.add_argument("candidate_media")
    parser.add_argument("retained_media")
    parser.add_argument("--candidate-transcript")
    parser.add_argument("--retained-transcript")
    parser.add_argument("--candidate-asr-plan")
    parser.add_argument("--retained-asr-plan")
    parser.add_argument("--candidate-visual-evidence", action="append", default=[])
    parser.add_argument("--retained-visual-evidence", action="append", default=[])
    parser.add_argument("--reference-root", action="append", default=[])
    parser.add_argument("--rebind-report")
    parser.add_argument("--critical-term", action="append", default=[])
    parser.add_argument(
        "--policy",
        type=normalize_quality_policy,
        default=DEFAULT_QUALITY_POLICY,
        metavar="{space_saving,archive_lossless}",
        help=(
            "Storage preset: space_saving by default, archive_lossless for no "
            "technical downgrade; legacy policy names remain accepted."
        ),
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown")
    parser.add_argument(
        "--audio-coverage-threshold", type=float, default=DEFAULT_AUDIO_COVERAGE
    )
    parser.add_argument(
        "--transcript-coverage-threshold",
        type=float,
        default=DEFAULT_TRANSCRIPT_COVERAGE,
    )
    parser.add_argument(
        "--visual-coverage-threshold", type=float, default=DEFAULT_VISUAL_COVERAGE
    )
    parser.add_argument(
        "--max-unique-speech-seconds",
        type=float,
        default=DEFAULT_MAX_UNIQUE_SPEECH_SECONDS,
    )
    parser.add_argument(
        "--min-audio-similarity", type=float, default=DEFAULT_AUDIO_SIMILARITY
    )
    parser.add_argument(
        "--audio-frame-similarity-threshold",
        type=float,
        default=DEFAULT_FRAME_SIMILARITY,
    )
    parser.add_argument(
        "--max-alignment-seconds",
        type=float,
        default=DEFAULT_MAX_ALIGNMENT_SECONDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_media_equivalence(
        args.candidate_media,
        args.retained_media,
        candidate_transcript=args.candidate_transcript,
        retained_transcript=args.retained_transcript,
        candidate_asr_plan=args.candidate_asr_plan,
        retained_asr_plan=args.retained_asr_plan,
        candidate_visual_evidence=args.candidate_visual_evidence,
        retained_visual_evidence=args.retained_visual_evidence,
        reference_roots=args.reference_root,
        rebind_report=args.rebind_report,
        critical_terms=args.critical_term,
        quality_policy=args.policy,
        audio_coverage_threshold=args.audio_coverage_threshold,
        transcript_coverage_threshold=args.transcript_coverage_threshold,
        visual_coverage_threshold=args.visual_coverage_threshold,
        max_unique_speech_seconds=args.max_unique_speech_seconds,
        min_audio_similarity=args.min_audio_similarity,
        frame_similarity_threshold=args.audio_frame_similarity_threshold,
        max_alignment_seconds=args.max_alignment_seconds,
    )
    output_json = Path(args.output_json).expanduser().resolve()
    write_json(output_json, report)
    if args.output_markdown:
        write_text_atomic(
            Path(args.output_markdown).expanduser().resolve(),
            render_media_equivalence_markdown(report),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["safe_to_delete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
