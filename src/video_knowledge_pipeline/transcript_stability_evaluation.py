from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

from .file_hash import sha256_file
from .speaker_diarization_evaluation import evaluate_speaker_diarization
from .speaker_transcription_evaluation import (
    evaluate_speaker_transcription_tokens,
)
from .storage import read_json, write_json
from .transcript import parse_transcript
from .video import probe_video


SCHEMA = "video_knowledge_pipeline.transcript_stability_evaluation.v1"
LOGSEQ_REFERENCE_SCHEMA = "video_knowledge_pipeline.logseq_original_transcript.v1"
REFERENCE_BINDING_SCHEMA = "video_knowledge_pipeline.transcript_reference_binding.v1"
NORMALIZATION_PROFILES = ("strict_v1", "content_vocal_fillers_v1")
_VOCAL_FILLER_RE = re.compile(r"[啊呃嗯哎哦]")
_DEFAULT_TOPIC_SIMILARITY = 0.12


def evaluate_transcript_stability(
    candidate: Any,
    reference: Any,
    *,
    task_instructions: str = "",
    max_normalized_reference_edit_distance: float = 0.05,
    normalization_profile: str = "strict_v1",
    require_speaker_attribution: bool = False,
    max_diarization_error_rate: float = 0.05,
    require_speaker_transcription: bool = False,
    max_cp_speaker_character_error_rate: float = 0.05,
    max_tcp_speaker_character_error_rate: float = 0.05,
    speaker_transcription_collar_seconds: float = 1.0,
) -> dict[str, Any]:
    """Compare transcript content, completeness, and optional speaker evidence.

    Intent: distinguish overall text stability, diarization timing, and words
    assigned to each anonymous speaker instead of collapsing them into one
    opaque score.
    Decision: retain the existing normalized edit-distance and pyannote DER
    gates, then optionally add MeetEval cpCER/tcpCER over the same normalized
    character evidence.
    Reason: text-only distance misses speaker swaps; DER misses words attached
    to the wrong speaker inside otherwise correct time regions.
    Evidence: pinned pyannote and MeetEval upstream suites plus VKP anonymous
    permutation, missing-runtime, privacy, and stability-gate fixtures.
    Effective scope: evaluation JSON only. References never enter ASR prompts,
    hotwords, routing, correction, or production truth.
    """

    normalization = _normalization_definition(normalization_profile)
    candidate_text = _transcript_text(candidate)
    reference_text = _transcript_text(reference)
    candidate_surface = _normalize(candidate_text, profile="strict_v1")
    reference_surface = _normalize(reference_text, profile="strict_v1")
    candidate_normalized = _normalize(candidate_text, profile=normalization_profile)
    reference_normalized = _normalize(reference_text, profile=normalization_profile)
    if not reference_normalized:
        raise ValueError("reference transcript contains no comparable text")
    distance = _levenshtein(candidate_normalized, reference_normalized)
    surface_distance = _levenshtein(candidate_surface, reference_surface)
    normalized_distance = distance / max(1, len(reference_normalized))
    length_ratio = len(candidate_normalized) / max(1, len(reference_normalized))
    normalized_length_delta = abs(len(reference_normalized) - len(candidate_normalized))
    normalized_length_deficit = max(
        0, len(reference_normalized) - len(candidate_normalized)
    )
    maximum_passing_distance = max(
        0,
        math.ceil(
            float(max_normalized_reference_edit_distance) * len(reference_normalized)
        )
        - 1,
    )
    candidate_duration = _duration(candidate)
    reference_duration = _duration(reference)
    duration_ratio = (
        candidate_duration / reference_duration
        if candidate_duration is not None
        and reference_duration
        and reference_duration > 0
        else None
    )
    prompt_leaks = _instruction_clause_matches(candidate_surface, task_instructions)
    long_form_loss = length_ratio < 0.9 or (
        duration_ratio is not None and duration_ratio < 0.9
    )
    diagnostic_statuses: list[str] = []
    if long_form_loss:
        diagnostic_statuses.append("possible_long_form_loss")
    if normalized_distance >= float(max_normalized_reference_edit_distance):
        diagnostic_statuses.append("asr_quality_distance_exceeded")
    if prompt_leaks:
        diagnostic_statuses.append("prompt_leak_detected")
    gates = {
        "normalized_reference_edit_distance": normalized_distance
        < float(max_normalized_reference_edit_distance),
        "no_prompt_leak": not prompt_leaks,
        "no_long_form_loss": not long_form_loss,
    }
    comparison_windows = _comparison_window_diagnostics(
        candidate, reference, normalization_profile=normalization_profile
    )
    speaker_attribution = evaluate_speaker_diarization(
        _timed_rows(reference),
        _timed_rows(candidate),
        max_diarization_error_rate=max_diarization_error_rate,
        required=require_speaker_attribution,
    )
    if require_speaker_attribution:
        gates["speaker_attribution"] = bool(speaker_attribution["passed"])
        if not speaker_attribution["passed"]:
            diagnostic_statuses.append(
                "speaker_attribution_" + str(speaker_attribution["status"])
            )
    speaker_transcription = evaluate_speaker_transcription_tokens(
        _speaker_token_rows(
            reference,
            normalization_profile=normalization_profile,
        ),
        _speaker_token_rows(
            candidate,
            normalization_profile=normalization_profile,
        ),
        max_cp_token_error_rate=max_cp_speaker_character_error_rate,
        max_tcp_token_error_rate=max_tcp_speaker_character_error_rate,
        collar_seconds=speaker_transcription_collar_seconds,
        token_unit="normalized_character",
        required=require_speaker_transcription,
    )
    if require_speaker_transcription:
        gates["speaker_transcription"] = bool(speaker_transcription["passed"])
        if not speaker_transcription["passed"]:
            diagnostic_statuses.append(
                "speaker_transcription_" + str(speaker_transcription["status"])
            )
    return {
        "schema": SCHEMA,
        "status": "passed" if all(gates.values()) else "failed",
        "evaluation_state": (
            "stable" if not diagnostic_statuses else diagnostic_statuses[0]
        ),
        "diagnostic_statuses": diagnostic_statuses,
        "evaluation_only": True,
        "reference_must_not_enter_prompt_hotwords_or_routing": True,
        "normalization": normalization,
        "metric": {
            "name": "normalized_reference_edit_distance",
            "normalization_profile": normalization_profile,
            "not_character_error_rate": True,
            "edit_distance": distance,
            "reference_normalized_chars": len(reference_normalized),
            "candidate_normalized_chars": len(candidate_normalized),
            "value": round(normalized_distance, 8),
            "threshold_exclusive": float(max_normalized_reference_edit_distance),
            "passed": gates["normalized_reference_edit_distance"],
        },
        "surface_metric": {
            "name": "normalized_reference_edit_distance",
            "normalization_profile": "strict_v1",
            "not_character_error_rate": True,
            "edit_distance": surface_distance,
            "reference_normalized_chars": len(reference_surface),
            "candidate_normalized_chars": len(candidate_surface),
            "value": round(surface_distance / max(1, len(reference_surface)), 8),
            "reported_even_when_not_primary": True,
        },
        "distance_lower_bound": {
            "reason": "normalized_length_delta",
            "normalized_character_length_delta": normalized_length_delta,
            "normalized_character_deficit": normalized_length_deficit,
            "relative_length_deficit": round(
                normalized_length_deficit / max(1, len(reference_normalized)), 8
            ),
            "maximum_integer_distance_that_passes": maximum_passing_distance,
            "minimum_edit_distance_reduction_required": max(
                0, distance - maximum_passing_distance
            ),
            "minimum_length_gap_reduction_required": max(
                0, normalized_length_delta - maximum_passing_distance
            ),
            "content_recovery_required": normalized_length_delta
            > maximum_passing_distance,
        },
        "completion": {
            "candidate_to_reference_text_length_ratio": round(length_ratio, 8),
            "candidate_duration_seconds": candidate_duration,
            "reference_duration_seconds": reference_duration,
            "candidate_to_reference_duration_ratio": round(duration_ratio, 8)
            if duration_ratio is not None
            else None,
            "possible_long_form_loss": long_form_loss,
            "assessment_status": "evaluated",
        },
        "prompt_leak": {
            "passed": not prompt_leaks,
            "matches": prompt_leaks,
        },
        "gates": gates,
        "comparison_windows": comparison_windows,
        "speaker_attribution": speaker_attribution,
        "speaker_transcription": speaker_transcription,
    }


def evaluate_transcript_files(
    candidate_path: str | Path,
    reference_path: str | Path,
    *,
    task_instructions: str = "",
    max_normalized_reference_edit_distance: float = 0.05,
    normalization_profile: str = "strict_v1",
    reference_binding: Mapping[str, Any] | str | Path | None = None,
    media_path: str | Path | None = None,
    media_duration_seconds: float | None = None,
    media_identity: Mapping[str, Any] | None = None,
    require_reference_binding: bool = True,
    require_speaker_attribution: bool = False,
    max_diarization_error_rate: float = 0.05,
    require_speaker_transcription: bool = False,
    max_cp_speaker_character_error_rate: float = 0.05,
    max_tcp_speaker_character_error_rate: float = 0.05,
    speaker_transcription_collar_seconds: float = 1.0,
) -> dict[str, Any]:
    """Compare saved transcripts after an exact reference identity gate.

    Intent: evaluate content, completeness, diarization and per-speaker text
    while keeping reference material outside the production pipeline.
    Decision: retain the narrow Logseq/GetBrain loaders and exact media binding,
    then delegate speaker metrics to the optional pyannote and MeetEval adapters.
    Reason: a reference transcript is valid evaluation evidence only when its
    identity is proven; neither metric may silently become ASR correction.
    Evidence: reference-binding mismatch fixtures, summary-only Markdown
    rejection, and optional-runtime speaker metric regressions.
    Effective scope: local evaluation reports. No prompt, hotword, route,
    provider, transcript or summary artifact is modified.
    """

    candidate, candidate_metadata = _load_evaluation_input(candidate_path)
    reference, reference_metadata = _load_evaluation_input(reference_path)
    if reference_binding is None:
        if require_reference_binding:
            result = _reference_binding_failure(
                ["reference_binding_required"],
                binding_source="",
            )
            result["inputs"] = _evaluation_input_metadata(
                candidate_metadata,
                reference_metadata,
            )
            return result
        binding_report: dict[str, Any] = {
            "schema": REFERENCE_BINDING_SCHEMA,
            "status": "legacy_unbound",
            "strict": False,
            "reasons": [],
            "requires_human_selection": False,
        }
    else:
        binding_payload, binding_source = _load_reference_binding(reference_binding)
        binding_report = validate_transcript_reference_binding(
            binding_payload,
            media_path=media_path,
            reference_path=reference_path,
            candidate=candidate,
            reference=reference,
            media_duration_seconds=media_duration_seconds,
            media_identity=media_identity,
            binding_source=binding_source,
        )
        if binding_report["status"] != "valid":
            result = _reference_binding_failure(
                list(binding_report.get("reasons") or []),
                binding_source=binding_source,
                binding_report=binding_report,
            )
            result["inputs"] = _evaluation_input_metadata(
                candidate_metadata,
                reference_metadata,
            )
            return result
    result = evaluate_transcript_stability(
        candidate,
        reference,
        task_instructions=task_instructions,
        max_normalized_reference_edit_distance=max_normalized_reference_edit_distance,
        normalization_profile=normalization_profile,
        require_speaker_attribution=require_speaker_attribution,
        max_diarization_error_rate=max_diarization_error_rate,
        require_speaker_transcription=require_speaker_transcription,
        max_cp_speaker_character_error_rate=max_cp_speaker_character_error_rate,
        max_tcp_speaker_character_error_rate=max_tcp_speaker_character_error_rate,
        speaker_transcription_collar_seconds=speaker_transcription_collar_seconds,
    )
    result["reference_binding"] = binding_report
    result["inputs"] = _evaluation_input_metadata(
        candidate_metadata,
        reference_metadata,
    )
    return result


def build_transcript_reference_binding(
    media_path: str | Path,
    reference_path: str | Path,
    *,
    candidate_path: str | Path | None = None,
    media_duration_seconds: float | None = None,
    media_identity: Mapping[str, Any] | None = None,
    allow_unavailable_media_identity: bool = False,
    getnote_id: str = "",
    topic_anchors: Sequence[str] = (),
    minimum_topic_similarity: float = _DEFAULT_TOPIC_SIMILARITY,
    allow_invalid: bool = False,
) -> dict[str, Any]:
    """Build one exact media -> GetNote -> reference binding.

    The binding is content addressed. Title similarity is never used to choose a
    reference. Topic anchors and opening fingerprints are validation evidence
    only, and an invalid binding is rejected unless ``allow_invalid`` is used by
    diagnostics/tests to inspect the reasons.
    """

    media = (
        Path(media_path).expanduser().absolute()
        if allow_unavailable_media_identity
        else Path(media_path).expanduser().resolve()
    )
    reference_file = Path(reference_path).expanduser().resolve()
    if not allow_unavailable_media_identity and not media.is_file():
        raise FileNotFoundError(media)
    if not reference_file.is_file():
        raise FileNotFoundError(reference_file)
    reference, reference_metadata = _load_evaluation_input(reference_file)
    candidate = None
    if candidate_path:
        candidate, _ = _load_evaluation_input(candidate_path)
    # Intent: reuse an exact ASR chunk-manifest identity for large removable-media inputs.
    # Decision: accept precomputed SHA/duration only when path, bytes, and mtime_ns still match.
    # Reason: rehashing multi-GB USB media for every incremental evaluation is slow and can fail I/O.
    # Evidence: xlong-02 raised OSError 22 after six minutes of redundant full-file hashing.
    # Effective scope: reference-binding construction; ordinary callers still probe and hash normally.
    verified_identity = _verified_precomputed_media_identity(
        media,
        media_identity,
        allow_unavailable=allow_unavailable_media_identity,
    )
    if verified_identity:
        actual_media_duration = float(verified_identity["duration_seconds"])
        actual_media_sha256 = str(verified_identity["sha256"])
        media_file_identity = {
            "bytes": int(verified_identity["bytes"]),
            "mtime_ns": int(verified_identity["mtime_ns"]),
            "source": str(verified_identity.get("source") or "precomputed_media_identity"),
        }
    else:
        supplied_duration = _positive_float(media_duration_seconds)
        if supplied_duration:
            actual_media_duration = supplied_duration
            actual_media_sha256 = sha256_file(media)
            identity_source = "supplied_duration_and_live_hash"
        else:
            probed_media = probe_video(media)
            actual_media_duration = float(probed_media.duration_seconds)
            actual_media_sha256 = str(probed_media.sha256)
            identity_source = "live_probe"
        media_stat = media.stat()
        media_file_identity = {
            "bytes": int(media_stat.st_size),
            "mtime_ns": int(media_stat.st_mtime_ns),
            "source": identity_source,
        }
    reference_markdown = _reference_markdown(reference_file)
    actual_getnote_id = _reference_getnote_id(reference_markdown) or str(getnote_id)
    reference_duration = _reference_document_duration(
        reference_markdown,
        reference,
    )
    topic = _topic_fingerprint_report(
        candidate,
        reference,
        media_title=media.stem,
        reference_markdown=reference_markdown,
        anchors=topic_anchors,
        minimum_similarity=minimum_topic_similarity,
    )
    reasons: list[str] = []
    if not actual_getnote_id:
        reasons.append("getnote_id_missing")
    if _reference_duration_mismatch(actual_media_duration, reference_duration):
        reasons.append("reference_duration_mismatch")
    if candidate is not None and not topic["passed"]:
        reasons.append("topic_fingerprint_mismatch")
    binding = {
        "schema": REFERENCE_BINDING_SCHEMA,
        "status": "active" if not reasons else "invalid",
        "video": {
            "path": str(media),
            "sha256": actual_media_sha256,
            "duration_seconds": actual_media_duration,
            "file_identity": media_file_identity,
        },
        "getnote": {"id": actual_getnote_id},
        "reference": {
            "path": str(reference_file),
            "sha256": reference_metadata["sha256"],
            "duration_seconds": reference_duration,
        },
        "topic_fingerprint": {
            "anchors": [
                str(value).strip() for value in topic_anchors if str(value).strip()
            ],
            "ngram_size": 3,
            "minimum_opening_similarity": float(minimum_topic_similarity),
            "creation_similarity": topic["opening_similarity"],
        },
        "creation_validation": {
            "status": "valid" if not reasons else "invalid",
            "reasons": reasons,
            "requires_human_selection": bool(reasons),
        },
        "policy": {
            "title_guessing_allowed": False,
            "reference_is_evaluation_only": True,
            "duration_relative_tolerance": 0.05,
            "duration_absolute_tolerance_seconds": 60.0,
        },
    }
    if reasons and not allow_invalid:
        raise ValueError(
            "reference binding is invalid: " + ", ".join(reasons)
        )
    return binding


def validate_transcript_reference_binding(
    binding: Mapping[str, Any],
    *,
    media_path: str | Path | None,
    reference_path: str | Path,
    candidate: Any,
    reference: Any,
    media_duration_seconds: float | None = None,
    media_identity: Mapping[str, Any] | None = None,
    binding_source: str = "",
) -> dict[str, Any]:
    """Validate exact identity, duration, and topic before ASR comparison."""

    reasons: list[str] = []
    if str(binding.get("schema") or "") != REFERENCE_BINDING_SCHEMA:
        reasons.append("reference_binding_schema_invalid")
    video_binding = binding.get("video")
    reference_binding = binding.get("reference")
    getnote_binding = binding.get("getnote")
    topic_binding = binding.get("topic_fingerprint")
    if not isinstance(video_binding, Mapping):
        video_binding = {}
        reasons.append("video_binding_missing")
    if not isinstance(reference_binding, Mapping):
        reference_binding = {}
        reasons.append("reference_binding_missing")
    if not isinstance(getnote_binding, Mapping):
        getnote_binding = {}
        reasons.append("getnote_binding_missing")
    if not isinstance(topic_binding, Mapping):
        topic_binding = {}
        reasons.append("topic_fingerprint_missing")

    media: Path | None = None
    actual_media_duration = 0.0
    identity_source = "live_media"
    if media_path is None:
        reasons.append("media_path_required_for_reference_binding")
    elif isinstance(media_identity, Mapping):
        media = Path(media_path).expanduser().absolute()
        verified_identity = _verified_precomputed_media_identity(
            media,
            media_identity,
            allow_unavailable=True,
        )
        expected_media_hash = str(video_binding.get("sha256") or "")
        file_identity = video_binding.get("file_identity")
        if expected_media_hash != str(verified_identity.get("sha256") or ""):
            reasons.append("video_sha256_mismatch")
        if not isinstance(file_identity, Mapping) or (
            int(file_identity.get("bytes") or 0)
            != int(verified_identity.get("bytes") or 0)
            or int(file_identity.get("mtime_ns") or 0)
            != int(verified_identity.get("mtime_ns") or 0)
        ):
            reasons.append("video_file_identity_mismatch")
        actual_media_duration = float(verified_identity["duration_seconds"])
        identity_source = str(
            verified_identity.get("source") or "precomputed_media_identity"
        )
    else:
        media = Path(media_path).expanduser().resolve()
        if not media.is_file():
            reasons.append("media_path_missing")
        else:
            expected_media_hash = str(video_binding.get("sha256") or "")
            file_identity = video_binding.get("file_identity")
            if isinstance(file_identity, Mapping):
                stat = media.stat()
                if (
                    int(file_identity.get("bytes") or 0) != int(stat.st_size)
                    or int(file_identity.get("mtime_ns") or 0) != int(stat.st_mtime_ns)
                ):
                    reasons.append("video_file_identity_mismatch")
                if len(expected_media_hash) != 64:
                    reasons.append("video_sha256_mismatch")
                actual_media_duration = _positive_float(
                    video_binding.get("duration_seconds")
                )
                if not actual_media_duration:
                    reasons.append("video_duration_missing")
            else:
                if not expected_media_hash or sha256_file(media) != expected_media_hash:
                    reasons.append("video_sha256_mismatch")
                actual_media_duration = _resolved_media_duration(
                    media,
                    supplied=media_duration_seconds,
                )

    reference_file = Path(reference_path).expanduser().resolve()
    reference_raw = reference_file.read_bytes()
    actual_reference_hash = hashlib.sha256(reference_raw).hexdigest()
    expected_reference_hash = str(reference_binding.get("sha256") or "")
    if not expected_reference_hash or actual_reference_hash != expected_reference_hash:
        reasons.append("reference_sha256_mismatch")
    reference_markdown = (
        reference_raw.decode("utf-8-sig")
        if reference_file.suffix.casefold() in {".md", ".markdown"}
        else ""
    )
    expected_getnote_id = str(getnote_binding.get("id") or "")
    actual_getnote_id = _reference_getnote_id(reference_markdown)
    if not expected_getnote_id:
        reasons.append("getnote_id_missing")
    elif actual_getnote_id != expected_getnote_id:
        reasons.append("getnote_id_mismatch")

    reference_duration = _reference_document_duration(reference_markdown, reference)
    if actual_media_duration and _reference_duration_mismatch(
        actual_media_duration,
        reference_duration,
    ):
        reasons.append("reference_duration_mismatch")
    expected_reference_duration = _positive_float(
        reference_binding.get("duration_seconds")
    )
    if expected_reference_duration and reference_duration:
        if abs(expected_reference_duration - reference_duration) > 1.0:
            reasons.append("reference_duration_changed")

    minimum_similarity = _positive_float(
        topic_binding.get("minimum_opening_similarity")
    ) or _DEFAULT_TOPIC_SIMILARITY
    anchors = topic_binding.get("anchors")
    if not isinstance(anchors, list):
        anchors = []
    topic = _topic_fingerprint_report(
        candidate,
        reference,
        media_title=media.stem if media else "",
        reference_markdown=reference_markdown,
        anchors=[str(value) for value in anchors],
        minimum_similarity=minimum_similarity,
    )
    if not topic["passed"]:
        reasons.append("topic_fingerprint_mismatch")
    reasons = list(dict.fromkeys(reasons))
    return {
        "schema": REFERENCE_BINDING_SCHEMA,
        "status": "valid" if not reasons else "invalid",
        "strict": True,
        "binding_source": binding_source,
        "media_identity_source": identity_source,
        "reasons": reasons,
        "requires_human_selection": bool(reasons),
        "identity": {
            "video_sha256_match": "video_sha256_mismatch" not in reasons,
            "reference_sha256_match": "reference_sha256_mismatch" not in reasons,
            "getnote_id_match": "getnote_id_mismatch" not in reasons
            and "getnote_id_missing" not in reasons,
        },
        "duration": {
            "media_seconds": actual_media_duration,
            "reference_seconds": reference_duration,
            "relative_tolerance": 0.05,
            "absolute_tolerance_seconds": 60.0,
            "passed": "reference_duration_mismatch" not in reasons,
        },
        "topic_fingerprint": topic,
        "title_guessing_allowed": False,
    }


def _load_reference_binding(
    value: Mapping[str, Any] | str | Path,
) -> tuple[dict[str, Any], str]:
    if isinstance(value, Mapping):
        return dict(value), "inline"
    path = Path(value).expanduser().resolve()
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("reference binding must be a JSON object")
    return payload, str(path)


def _reference_binding_failure(
    reasons: list[str],
    *,
    binding_source: str,
    binding_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = dict(binding_report or {})
    report.setdefault("schema", REFERENCE_BINDING_SCHEMA)
    report["status"] = "invalid"
    report["strict"] = True
    report["binding_source"] = binding_source
    report["reasons"] = list(dict.fromkeys(reasons))
    report["requires_human_selection"] = True
    return {
        "schema": SCHEMA,
        "status": "failed",
        "evaluation_state": "reference_binding_invalid",
        "diagnostic_statuses": ["reference_binding_invalid"],
        "evaluation_only": True,
        "reference_must_not_enter_prompt_hotwords_or_routing": True,
        "reference_binding": report,
        "metric": {
            "name": "normalized_reference_edit_distance",
            "status": "not_evaluated",
            "reason": "reference_binding_invalid",
        },
        "completion": {
            "assessment_status": "not_evaluated",
            "possible_long_form_loss": False,
        },
        "gates": {"reference_binding_valid": False},
        "comparison_windows": {
            "status": "unavailable",
            "reason": "reference_binding_invalid",
            "windows": [],
        },
    }


def _evaluation_input_metadata(
    candidate_metadata: Mapping[str, Any],
    reference_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "candidate": {**candidate_metadata, "role": "candidate"},
        "reference": {
            **reference_metadata,
            "role": "evaluation_only_reference",
            "must_not_enter_prompt_hotwords_or_routing": True,
        },
    }


def _reference_markdown(path: Path) -> str:
    if path.suffix.casefold() not in {".md", ".markdown"}:
        return ""
    return path.read_text(encoding="utf-8-sig")


def _reference_getnote_id(markdown: str) -> str:
    match = re.search(
        r"(?im)^\s*getnote-id\s*::\s*([^\s]+)\s*$",
        str(markdown or ""),
    )
    return match.group(1).strip() if match else ""


def _reference_document_duration(markdown: str, payload: Any) -> float:
    timed_duration = float(_duration(payload) or 0.0)
    if timed_duration > 0:
        return timed_duration
    text = str(markdown or "")
    match = re.search(
        r"\*\*时长\*\*[：:]\s*约?\s*(?:(\d+)\s*小时)?\s*(?:(\d+)\s*分钟)?\s*(?:(\d+(?:\.\d+)?)\s*秒)?",
        text,
    )
    if match and any(value is not None for value in match.groups()):
        hours = float(match.group(1) or 0)
        minutes = float(match.group(2) or 0)
        seconds = float(match.group(3) or 0)
        duration = hours * 3600 + minutes * 60 + seconds
        if duration > 0:
            return duration
    return float(_duration(payload) or 0.0)


def _verified_precomputed_media_identity(
    media: Path,
    value: Mapping[str, Any] | None,
    *,
    allow_unavailable: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    expected_path = (
        Path(str(value.get("path") or "")).expanduser().absolute()
        if allow_unavailable
        else Path(str(value.get("path") or "")).expanduser().resolve()
    )
    expected_sha = str(value.get("sha256") or "").strip().lower()
    expected_bytes = int(value.get("bytes") or 0)
    expected_mtime_ns = int(value.get("mtime_ns") or 0)
    expected_duration = _positive_float(value.get("duration_seconds"))
    if expected_path != media:
        raise ValueError("precomputed media identity path mismatch")
    if len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
        raise ValueError("precomputed media identity sha256 invalid")
    if not allow_unavailable:
        stat = media.stat()
        if expected_bytes != int(stat.st_size) or expected_mtime_ns != int(stat.st_mtime_ns):
            raise ValueError("precomputed media identity is stale")
    if not expected_duration:
        raise ValueError("precomputed media identity duration missing")
    return {
        "path": str(media),
        "sha256": expected_sha,
        "bytes": expected_bytes,
        "mtime_ns": expected_mtime_ns,
        "duration_seconds": expected_duration,
        "source": str(value.get("source") or "precomputed_media_identity"),
    }

def _resolved_media_duration(path: Path, *, supplied: float | None) -> float:
    value = _positive_float(supplied)
    if value:
        return value
    return float(probe_video(path).duration_seconds)


def _reference_duration_mismatch(media_duration: float, reference_duration: float) -> bool:
    if media_duration <= 0 or reference_duration <= 0:
        return True
    delta = abs(media_duration - reference_duration)
    relative = delta / max(media_duration, 1.0)
    return relative > 0.05 or delta > 60.0


def _topic_fingerprint_report(
    candidate: Any,
    reference: Any,
    *,
    media_title: str,
    reference_markdown: str,
    anchors: Sequence[str],
    minimum_similarity: float,
) -> dict[str, Any]:
    candidate_opening = _normalize(_transcript_text(candidate)[:2000])
    reference_opening = _normalize(_transcript_text(reference)[:2000])
    similarity = _ngram_jaccard(candidate_opening, reference_opening, size=3)
    candidate_scope = _normalize(media_title + candidate_opening)
    reference_scope = _normalize(reference_markdown[:6000] + reference_opening)
    normalized_anchors = [
        _normalize(value) for value in anchors if _normalize(value)
    ]
    matched = sum(
        1
        for anchor in normalized_anchors
        if anchor in candidate_scope and anchor in reference_scope
    )
    anchors_passed = not normalized_anchors or matched == len(normalized_anchors)
    similarity_passed = bool(candidate_opening and reference_opening) and (
        similarity >= float(minimum_similarity)
    )
    return {
        "passed": anchors_passed and similarity_passed,
        "opening_similarity": round(similarity, 8),
        "minimum_opening_similarity": float(minimum_similarity),
        "ngram_size": 3,
        "anchor_count": len(normalized_anchors),
        "matched_anchor_count": matched,
        "content_included": False,
    }


def _ngram_jaccard(left: str, right: str, *, size: int) -> float:
    if not left or not right:
        return 0.0
    width = max(1, int(size))
    left_grams = {
        left[index : index + width]
        for index in range(max(1, len(left) - width + 1))
    }
    right_grams = {
        right[index : index + width]
        for index in range(max(1, len(right) - width + 1))
    }
    union = left_grams | right_grams
    return len(left_grams & right_grams) / max(1, len(union))


def _positive_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed > 0 else 0.0


def extract_logseq_original_transcript(markdown: str) -> dict[str, Any]:
    """Extract only the ``原始转录`` tree from a GetBrain/Logseq page."""

    lines = str(markdown or "").splitlines()
    start_index = next(
        (index for index, line in enumerate(lines) if line.strip() == "- 原始转录"),
        None,
    )
    if start_index is None:
        raise ValueError("Logseq reference is missing the top-level 原始转录 block")
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def finish_current() -> None:
        nonlocal current
        if current is None:
            return
        text = " ".join(
            str(part).strip() for part in current.pop("_parts", []) if str(part).strip()
        )
        if text:
            current["text"] = text
            current["id"] = f"getbrain-segment-{len(segments) + 1:04d}"
            segments.append(current)
        current = None

    for line in lines[start_index + 1 :]:
        if line.startswith("- "):
            break
        if not re.match(r"^[\t ]+-\s+", line):
            continue
        body = re.sub(r"^[\t ]+-\s+", "", line, count=1).strip()
        speaker_match = re.search(r"\[(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\]\s*$", body)
        if speaker_match and ("说话人" in body or body.startswith("🟢")):
            finish_current()
            current = {
                "start": _timestamp_seconds(speaker_match.group(1)),
                "speaker": body[: speaker_match.start()].strip(),
                "_parts": [],
            }
            continue
        if current is not None and body:
            current["_parts"].append(body)
    finish_current()
    if not segments:
        raise ValueError(
            "Logseq 原始转录 block contains no timestamped transcript segments"
        )
    for index, segment in enumerate(segments[:-1]):
        segment["end"] = segments[index + 1]["start"]
    duration = max(float(segment["start"]) for segment in segments)
    return {
        "schema": LOGSEQ_REFERENCE_SCHEMA,
        "duration_seconds": duration,
        "segments": segments,
    }


def _timestamp_seconds(value: str) -> float:
    hours, minutes, seconds = str(value).split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _load_evaluation_input(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load evaluation evidence without promoting it into production truth.

    Intent: let the same blind evaluator consume JSON, Logseq trees, and the
    speaker-timestamp exports that operators actually receive.
    Decision: retain the existing narrow Logseq ``- 原始转录`` extractor,
    delegate supported local transcript text formats to ``parse_transcript``,
    and reject Markdown that has neither recognized transcript boundary.
    Reason: title/summary pseudo-segments would corrupt completeness and speaker
    metrics; reference text must still remain evaluation-only.
    Evidence: the real raw and combined GetBrain exports both parse as 433
    speaker-timestamp segments, while summary-only Markdown is not transcript
    evidence and the older Logseq fixture requires its indented-tree boundary.
    Effective scope: local evaluation input loading only. No prompt, hotword,
    routing, ASR, semantic correction, or production artifact is modified.
    """

    resolved = Path(path).expanduser().resolve()
    raw = resolved.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    suffix = resolved.suffix.casefold()
    markdown_suffixes = {".md", ".markdown"}
    decoded = raw.decode("utf-8-sig") if suffix in {
        *markdown_suffixes,
        ".txt",
        ".srt",
        ".vtt",
    } else ""
    is_logseq_original_tree = suffix in markdown_suffixes and any(
        line.strip() == "- 原始转录" for line in decoded.splitlines()
    )
    if is_logseq_original_tree:
        payload = extract_logseq_original_transcript(decoded)
        input_format = "logseq_markdown_original_transcript"
    elif decoded:
        cues = parse_transcript(resolved)
        has_speaker_timestamp_boundary = any(
            str(cue.metadata.get("source_format") or "").strip()
            == "speaker_timestamp_plaintext"
            for cue in cues
        )
        if suffix in markdown_suffixes and not has_speaker_timestamp_boundary:
            raise ValueError(
                "Markdown evaluation input must contain either a Logseq "
                "'- 原始转录' tree or recognized speaker-timestamp transcript headers"
            )
        segments = [
            {
                "id": cue.segment_id or f"segment-{index:06d}",
                "segment_id": cue.segment_id or f"segment-{index:06d}",
                "start": float(cue.start),
                "end": float(cue.end),
                "text": cue.text,
                "speaker": cue.speaker,
                "speaker_role": cue.speaker_role,
                "metadata": dict(cue.metadata),
            }
            for index, cue in enumerate(cues, start=1)
        ]
        payload = {
            "schema": "video_knowledge_pipeline.evaluation_transcript_import.v1",
            "duration_seconds": max(
                (
                    max(float(row["start"]), float(row["end"]))
                    for row in segments
                ),
                default=0.0,
            ),
            "segments": segments,
        }
        input_format = (
            "speaker_timestamp_text"
            if any(str(row.get("speaker") or "").strip() for row in segments)
            else "local_transcript_text"
        )
    else:
        payload = read_json(resolved)
        input_format = "json"
    return payload, {
        "path": str(resolved),
        "sha256": digest,
        "bytes": len(raw),
        "format": input_format,
        "segment_count": len(payload.get("segments") or [])
        if isinstance(payload, dict)
        else None,
    }


def _transcript_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        return "\n".join(_row_text(row) for row in payload if _row_text(row))
    if not isinstance(payload, dict):
        return ""
    for key in ("segments", "cues", "items", "transcript"):
        rows = payload.get(key)
        if isinstance(rows, list):
            text = "\n".join(_row_text(row) for row in rows if _row_text(row))
            if text:
                return text
    return str(payload.get("text") or payload.get("content") or "")


def _row_text(row: Any) -> str:
    if isinstance(row, str):
        return row.strip()
    if isinstance(row, dict):
        return str(
            row.get("text") or row.get("transcript") or row.get("content") or ""
        ).strip()
    return ""


def _duration(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None
    for key in ("duration", "duration_seconds"):
        try:
            value = float(payload.get(key))
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    for key in ("segments", "cues", "items", "transcript"):
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        ends: list[float] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                ends.append(float(row.get("end")))
            except (TypeError, ValueError):
                continue
        if ends:
            return max(ends)
    return None


def _comparison_window_diagnostics(
    candidate: Any,
    reference: Any,
    *,
    normalization_profile: str = "strict_v1",
) -> dict[str, Any]:
    """Return aggregate time-window metrics without reference transcript text."""

    candidate_rows = _timed_rows(candidate)
    reference_rows = _timed_rows(reference)
    if not candidate_rows or not reference_rows:
        return {
            "status": "unavailable",
            "reason": "timestamped candidate and reference segments are required",
            "windows": [],
        }
    candidate_duration = _duration(candidate) or max(
        row["end"] for row in candidate_rows
    )
    reference_duration = _duration(reference) or max(
        row["start"] for row in reference_rows
    )
    final_end = max(float(candidate_duration), float(reference_duration))
    windows: list[dict[str, Any]] = []
    for index, reference_row in enumerate(reference_rows):
        start = float(reference_row["start"])
        if index + 1 < len(reference_rows):
            end = float(reference_rows[index + 1]["start"])
        else:
            end = final_end
        if end <= start:
            continue
        selected = [
            row
            for row in candidate_rows
            if start <= (float(row["start"]) + float(row["end"])) / 2 < end
        ]
        candidate_text = _normalize(
            " ".join(str(row["text"]) for row in selected),
            profile=normalization_profile,
        )
        reference_text = _normalize(str(reference_row["text"]), profile=normalization_profile)
        if not reference_text:
            continue
        distance = _levenshtein(candidate_text, reference_text)
        value = distance / max(1, len(reference_text))
        windows.append(
            {
                "reference_segment_id": str(
                    reference_row.get("id") or f"reference-{index + 1:04d}"
                ),
                "start_seconds": start,
                "end_seconds": end,
                "candidate_segment_count": len(selected),
                "candidate_normalized_chars": len(candidate_text),
                "reference_normalized_chars": len(reference_text),
                "edit_distance": distance,
                "normalized_reference_edit_distance": round(value, 8),
                "target_below": 0.05,
                "passed": value < 0.05,
            }
        )
    ranked = sorted(
        windows,
        key=lambda row: (
            float(row["normalized_reference_edit_distance"]),
            int(row["edit_distance"]),
        ),
        reverse=True,
    )
    return {
        "status": "available",
        "content_included": False,
        "diagnostic_only": True,
        "normalization_profile": normalization_profile,
        "assignment_policy": "candidate_segment_midpoint_to_reference_speaker_window",
        "window_count": len(windows),
        "failed_window_count": sum(not bool(row["passed"]) for row in windows),
        "windows": windows,
        "highest_difference_windows": [
            {
                "reference_segment_id": row["reference_segment_id"],
                "start_seconds": row["start_seconds"],
                "end_seconds": row["end_seconds"],
                "normalized_reference_edit_distance": row[
                    "normalized_reference_edit_distance"
                ],
            }
            for row in ranked[:5]
        ],
    }


def _timed_rows(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in ("segments", "cues", "items", "transcript"):
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        timed: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                start = float(row.get("start"))
                end = float(row.get("end", start))
            except (TypeError, ValueError):
                continue
            text = _row_text(row)
            if text:
                timed.append(
                    {**row, "start": start, "end": max(start, end), "text": text}
                )
        if timed:
            return sorted(timed, key=lambda row: (row["start"], row["end"]))
    return []


def _speaker_token_rows(
    payload: Any,
    *,
    normalization_profile: str,
) -> list[dict[str, Any]]:
    """Project timed transcript rows into normalized character-token evidence.

    Intent: give MeetEval speaker-aware text evidence without duplicating the
    transcript normalizer or exposing raw text in its report.
    Decision: reuse this module's selected normalization profile and emit one
    Unicode content character per token.
    Reason: Chinese has no mandatory whitespace word boundary, while MeetEval
    explicitly accepts caller-supplied token sequences.
    Evidence: the pinned MeetEval cpWER implementation accepts lists of tokens;
    its official cpWER/tcpWER tests pass in the isolated Python 3.12 runtime.
    Effective scope: in-memory evaluation input only; no stored transcript,
    correction, prompt, or production artifact is changed.
    """

    projected: list[dict[str, Any]] = []
    for row in _timed_rows(payload):
        speaker = str(row.get("speaker") or "").strip()
        normalized = _normalize(
            str(row.get("text") or ""),
            profile=normalization_profile,
        )
        if not speaker or not normalized:
            continue
        projected.append(
            {
                "speaker": speaker,
                "start": float(row["start"]),
                "end": float(row["end"]),
                "tokens": list(normalized),
            }
        )
    return projected


def _normalization_definition(profile: str) -> dict[str, Any]:
    normalized = str(profile or "strict_v1").strip().lower()
    if normalized not in NORMALIZATION_PROFILES:
        raise ValueError(f"unsupported transcript normalization profile: {profile}")
    if normalized == "content_vocal_fillers_v1":
        return {
            "profile": normalized,
            "purpose": "semantic_content_stability_not_surface_transcript_identity",
            "symmetric": True,
            "removed_vocal_fillers": ["啊", "呃", "嗯", "哎", "哦"],
            "surface_metric_always_reported": True,
            "not_character_error_rate": True,
        }
    return {
        "profile": normalized,
        "purpose": "surface_transcript_stability",
        "symmetric": True,
        "removed_vocal_fillers": [],
        "surface_metric_always_reported": True,
        "not_character_error_rate": True,
    }


def _normalize(value: str, *, profile: str = "strict_v1") -> str:
    definition = _normalization_definition(profile)
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    if definition["profile"] == "content_vocal_fillers_v1":
        text = _VOCAL_FILLER_RE.sub("", text)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def _instruction_clause_matches(
    candidate_normalized: str, instructions: str
) -> list[str]:
    matches: list[str] = []
    for raw_clause in re.split(r"[\r\n。！？!?；;，,]+", str(instructions or "")):
        clause = _normalize(raw_clause)
        if len(clause) >= 6 and clause in candidate_normalized:
            matches.append(raw_clause.strip())
    return list(dict.fromkeys(matches))


def _levenshtein(left: str, right: str) -> int:
    """Exact Myers bit-vector Levenshtein distance using Python big integers."""

    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) > len(right):
        left, right = right, left
    width = len(left)
    mask = (1 << width) - 1
    high_bit = 1 << (width - 1)
    char_masks: dict[str, int] = {}
    for index, char in enumerate(left):
        char_masks[char] = char_masks.get(char, 0) | (1 << index)
    positive = mask
    negative = 0
    distance = width
    for char in right:
        equal = char_masks.get(char, 0)
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        positive_step = negative | ~(horizontal | positive)
        negative_step = positive & horizontal
        distance += bool(positive_step & high_bit) - bool(negative_step & high_bit)
        positive_step = ((positive_step << 1) | 1) & mask
        negative_step = (negative_step << 1) & mask
        positive = (negative_step | ~(equal | negative | positive_step)) & mask
        negative = positive_step & (equal | negative)
    return int(distance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluation-only transcript stability comparison"
    )
    parser.add_argument("candidate_path")
    parser.add_argument("reference_path")
    parser.add_argument("output_json")
    parser.add_argument("--task-instructions", default="")
    parser.add_argument("--max-distance", type=float, default=0.05)
    parser.add_argument(
        "--normalization-profile",
        choices=NORMALIZATION_PROFILES,
        default="strict_v1",
    )
    parser.add_argument("--reference-binding", default="")
    parser.add_argument("--media-path", default="")
    parser.add_argument("--media-duration-seconds", type=float, default=None)
    parser.add_argument(
        "--require-reference-binding",
        action="store_true",
        help="Reject evaluation before comparison unless an exact binding is valid",
    )
    parser.add_argument(
        "--require-speaker-attribution",
        action="store_true",
        help="Fail when timed speaker evidence is missing, unavailable, or above DER",
    )
    parser.add_argument(
        "--max-diarization-error-rate",
        type=float,
        default=0.05,
        help="Exclusive DER threshold when speaker attribution is required",
    )
    parser.add_argument(
        "--require-speaker-transcription",
        action="store_true",
        help=(
            "Fail when normalized characters are assigned to the wrong anonymous "
            "speaker under MeetEval cpCER/tcpCER"
        ),
    )
    parser.add_argument(
        "--max-cp-speaker-character-error-rate",
        type=float,
        default=0.05,
        help="Exclusive MeetEval cpCER threshold",
    )
    parser.add_argument(
        "--max-tcp-speaker-character-error-rate",
        type=float,
        default=0.05,
        help="Exclusive time-constrained MeetEval cpCER threshold",
    )
    parser.add_argument(
        "--speaker-transcription-collar-seconds",
        type=float,
        default=1.0,
        help="Timing collar used by MeetEval tcpCER",
    )
    parser.add_argument("--create-reference-binding", default="")
    parser.add_argument("--topic-anchor", action="append", default=[])
    parser.add_argument(
        "--allow-legacy-unbound",
        action="store_true",
        help="Explicit compatibility escape hatch; strict binding is the CLI default",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    binding_value: Mapping[str, Any] | str | Path | None = (
        args.reference_binding or None
    )
    if args.create_reference_binding:
        if not args.media_path:
            raise ValueError("--media-path is required to create a reference binding")
        binding_value = build_transcript_reference_binding(
            args.media_path,
            args.reference_path,
            candidate_path=args.candidate_path,
            media_duration_seconds=args.media_duration_seconds,
            topic_anchors=args.topic_anchor,
        )
        write_json(Path(args.create_reference_binding).expanduser().resolve(), binding_value)
    result = evaluate_transcript_files(
        args.candidate_path,
        args.reference_path,
        task_instructions=args.task_instructions,
        max_normalized_reference_edit_distance=args.max_distance,
        normalization_profile=args.normalization_profile,
        reference_binding=binding_value,
        media_path=args.media_path or None,
        media_duration_seconds=args.media_duration_seconds,
        require_reference_binding=(
            args.require_reference_binding or not args.allow_legacy_unbound
        ),
        require_speaker_attribution=args.require_speaker_attribution,
        max_diarization_error_rate=args.max_diarization_error_rate,
        require_speaker_transcription=args.require_speaker_transcription,
        max_cp_speaker_character_error_rate=(
            args.max_cp_speaker_character_error_rate
        ),
        max_tcp_speaker_character_error_rate=(
            args.max_tcp_speaker_character_error_rate
        ),
        speaker_transcription_collar_seconds=(
            args.speaker_transcription_collar_seconds
        ),
    )
    destination = Path(args.output_json).expanduser().resolve()
    write_json(destination, result)
    print(
        json.dumps(
            {**result, "report_path": str(destination)}, ensure_ascii=False, indent=2
        )
    )
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
