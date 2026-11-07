from __future__ import annotations

from pathlib import Path
from typing import Any

from .asr_response_quality import assess_asr_response
from .asr_response_quality import DEFAULT_VAD_COVERAGE_GAP_SECONDS, DEFAULT_VAD_MINIMUM_COVERAGE_RATIO
from .file_hash import sha256_file
from .interval_coverage import interval_coverage
from .storage import read_json
from .text_normalization import compact_ascii_cjk


SCHEMA = "video_knowledge_pipeline.transcript_source_completeness.v1"
FUNASR_SINGLE_PASS_SCHEMA = "video_knowledge_funasr_raw_output.v1"
FUNASR_CHUNKED_SCHEMA = "video_knowledge_funasr_chunked_raw_output.v1"
SILERO_CANDIDATE_SCHEMA = "video_knowledge_pipeline.silero_vad_candidate.v1"
BOUNDARY_REVIEW_SCHEMA = "video_knowledge_pipeline.asr_boundary_review.v1"
LONG_SINGLE_PASS_SECONDS = 20 * 60


def assess_transcript_source_completeness(
    bundle_dir: str | Path,
    transcript_path: str | Path,
    *,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess ASR execution lineage without equating time span with speech coverage.

    Intent: make long-form transcript completeness evidence machine-readable.
    Decision: reuse VKP's existing ASR response-quality and faster-whisper/Silero
    candidate contracts; this module only resolves lineage and combines evidence.
    Reason: a transcript can span the full media duration while silently omitting
    speech, and a successful fixed-size chunk run can still lose boundary words.
    Evidence: ``asr_response_quality.assess_asr_response`` and
    ``silero_vad_candidate`` are the existing reviewed implementations.
    Effective scope: transcript quality reporting only; no ASR output is modified.
    """

    root = Path(bundle_dir).expanduser().resolve()
    transcript = Path(transcript_path).expanduser().resolve()
    bundle_manifest = manifest if isinstance(manifest, dict) else {}
    source_path = _resolve_raw_source(root, transcript)
    base: dict[str, Any] = {
        "schema": SCHEMA,
        "applicable": False,
        "status": "not_evaluated",
        "execution_integrity": "not_evaluated",
        "speech_coverage": "not_evaluated",
        "speech_completeness_verified": False,
        "source_path": str(source_path) if source_path else "",
        "source_schema": "",
        "execution_mode": "unknown",
        "media_duration_seconds": 0.0,
        "timing_precision": "unknown",
        "estimated_timing_segment_count": 0,
        "chunk_integrity": {},
        "response_quality": {},
        "independent_vad": {},
        "boundary_review": {},
        "issues": [],
        "semantics": {
            "timeline_span_is_speech_completeness": False,
            "successful_chunks_are_speech_completeness": False,
            "canonical_transcript_modified": False,
        },
    }
    if source_path is None:
        base["reason"] = "machine_readable_asr_source_lineage_not_available"
        return base
    if not source_path.is_file():
        base.update(
            {
                "applicable": True,
                "status": "failed",
                "execution_integrity": "failed",
                "reason": "asr_source_artifact_missing",
                "issues": [
                    _finding(
                        "asr_source_artifact_missing",
                        "fail",
                        f"ASR source artifact is missing: {source_path}",
                    )
                ],
            }
        )
        return base

    payload = read_json(source_path)
    if not isinstance(payload, dict):
        base.update(
            {
                "applicable": True,
                "status": "failed",
                "execution_integrity": "failed",
                "reason": "asr_source_artifact_invalid",
                "issues": [
                    _finding(
                        "asr_source_artifact_invalid",
                        "fail",
                        "ASR source artifact must be a JSON object",
                    )
                ],
            }
        )
        return base

    schema = str(payload.get("schema") or "")
    duration = _number(payload.get("duration_seconds"))
    normalized_segments = _transcript_segments(transcript)
    estimated_timing_count = sum(
        _segment_timing_is_estimated(row) for row in normalized_segments
    )
    base.update(
        {
            "applicable": schema
            in {FUNASR_SINGLE_PASS_SCHEMA, FUNASR_CHUNKED_SCHEMA},
            "source_schema": schema,
            "media_duration_seconds": duration,
            "timing_precision": (
                "coarse_estimated"
                if estimated_timing_count
                else "provider_timed_or_unknown"
            ),
            "estimated_timing_segment_count": estimated_timing_count,
        }
    )
    if not base["applicable"]:
        base["reason"] = f"unsupported_or_external_asr_source_schema:{schema or 'missing'}"
        return base

    if schema == FUNASR_CHUNKED_SCHEMA:
        _assess_chunked_source(base, payload)
        _apply_boundary_review_evidence(
            base,
            root,
            source_path,
            payload,
            bundle_manifest,
        )
    else:
        _assess_single_pass_source(base, duration)

    vad = _load_bound_independent_vad(
        root,
        source_path,
        payload,
        bundle_manifest,
    )
    base["independent_vad"] = vad["summary"]
    if schema == FUNASR_CHUNKED_SCHEMA and vad["verified"]:
        _reconcile_verified_silent_chunks(
            base,
            payload,
            vad["segments"],
        )
    response_quality = assess_asr_response(
        {"segments": normalized_segments},
        vad_intervals=vad["segments"],
        media_duration_seconds=duration or None,
    )
    alignment_coverage = _bound_word_alignment_coverage(
        root, transcript, normalized_segments, payload, bundle_manifest, vad,
    )
    base["alignment_coverage"] = alignment_coverage
    if alignment_coverage.get("status") == "passed":
        # Retain strict timestamp measurements. Only a separately hash-bound,
        # text-identical alignment can reconcile provider-boundary gaps.
        base["primary_timestamp_speech_coverage"] = response_quality.get("speech_coverage")
        response_quality = dict(response_quality)
        response_quality["speech_coverage"] = alignment_coverage["coverage"]
        response_quality["coverage_gap_count"] = 0
        retry = dict(response_quality.get("retry_plan") or {})
        retry["windows"] = [row for row in retry.get("windows") or []
                            if set(row.get("reasons") or []) != {"missing_speech_coverage"}]
        retry["status"] = "authorization_required" if retry["windows"] else "not_needed"
        response_quality["retry_plan"] = retry
        if not response_quality.get("failed_segment_count") and not response_quality.get("review_segment_count"):
            response_quality.update(status="passed", quality_gate_passed=True)
    base["response_quality"] = {
        "status": response_quality.get("status"),
        "quality_gate_passed": response_quality.get("quality_gate_passed"),
        "segment_count": response_quality.get("segment_count"),
        "review_segment_count": response_quality.get("review_segment_count"),
        "failed_segment_count": response_quality.get("failed_segment_count"),
        "coverage_gap_count": response_quality.get("coverage_gap_count"),
        "retry_plan": response_quality.get("retry_plan"),
        "coarse_timing_density": response_quality.get("coarse_timing_density"),
        "quality_signal_sources": response_quality.get("quality_signal_sources"),
    }
    if estimated_timing_count:
        base["issues"].append(
            _finding(
                "asr_timing_estimated",
                "warning",
                (
                    f"{estimated_timing_count} normalized ASR segment(s) use "
                    "character-proportional timing within their source chunk; "
                    "timestamps are suitable for coarse navigation, not word-level alignment"
                ),
            )
        )
    if response_quality.get("failed_segment_count"):
        base["issues"].append(
            _finding(
                "asr_response_quality_failed",
                "fail",
                f"{response_quality['failed_segment_count']} normalized ASR segment(s) failed the existing response-quality gate",
            )
        )
    elif response_quality.get("review_segment_count"):
        base["issues"].append(
            _finding(
                "asr_response_quality_review_required",
                "warning",
                f"{response_quality['review_segment_count']} normalized ASR segment(s) need review",
            )
        )

    if vad["verified"]:
        if response_quality.get("coverage_gap_count"):
            base["speech_coverage"] = "independent_vad_gap_detected"
            base["issues"].append(
                _finding(
                    "independent_vad_speech_gap",
                    "fail",
                    f"{response_quality['coverage_gap_count']} independent VAD speech interval(s) lack transcript coverage",
                )
            )
        else:
            base["speech_coverage"] = "verified_by_independent_vad"
            base["speech_completeness_verified"] = True
    else:
        base["speech_coverage"] = "unverified"
        base["issues"].append(
            _finding(
                "speech_completeness_unverified",
                "warning",
                "No completed, source-bound independent Silero VAD candidate was available; timeline coverage alone is not speech completeness",
            )
        )

    fail_count = sum(row["severity"] == "fail" for row in base["issues"])
    warning_count = sum(row["severity"] == "warning" for row in base["issues"])
    base["status"] = (
        "failed"
        if fail_count
        else ("warning" if warning_count else "passed")
    )
    return base


def _segment_timing_is_estimated(segment: dict[str, Any]) -> bool:
    """Reuse normalizer provenance to expose coarse timing as a quality fact.

    Intent: keep transcript completeness and timestamp precision separate.
    Decision: read ``timing_estimation`` transformations already emitted by
    the ASR adapter.
    Reason: chunk-level coverage can support content completeness while still
    being unsuitable for exact word seek.
    Evidence: SenseVoice text-only chunk responses in the 2026-07-24 bundle.
    Effective scope: quality reporting only; transcript text and timing are
    never rewritten here.
    """

    for row in segment.get("transformations") or []:
        if isinstance(row, dict) and str(row.get("type") or "").strip().lower() in {
            "timing_estimation",
            "estimated_timing",
        }:
            return True
    return False


def _resolve_raw_source(root: Path, transcript_path: Path) -> Path | None:
    """Follow local transcript lineage until the machine ASR receipt is found.

    Intent: stop readable/corrected projections from hiding the raw ASR quality
    status behind one intermediate ``source_path`` hop.
    Decision: follow a bounded, cycle-safe chain and prefer a supported raw ASR
    schema; retain the last resolvable source for explicit unsupported reporting.
    Reason: VKP's readable transcript points to normalized-transcript.json, which
    in turn points to raw-asr-output.json. Returning the first hop made source
    completeness incorrectly report ``schema:missing``.
    Evidence: the two Cantonese interview Bundles use exactly this two-hop
    lineage and one raw receipt is degraded at chunk boundaries.
    Effective scope: read-only transcript quality lineage resolution.
    """

    queue = [transcript_path, root / "normalized-transcript.json"]
    seen: set[Path] = set()
    last_source: Path | None = None
    for _ in range(12):
        if not queue:
            break
        candidate = queue.pop(0).expanduser().resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        value = read_json(candidate)
        if not isinstance(value, dict):
            continue
        schema = str(value.get("schema") or "")
        if schema in {FUNASR_SINGLE_PASS_SCHEMA, FUNASR_CHUNKED_SCHEMA}:
            return candidate
        raw = str(value.get("source_path") or "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = candidate.parent / path
        last_source = path.resolve()
        queue.insert(0, last_source)
    return last_source


def _transcript_segments(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if not isinstance(value, dict):
        return []
    rows = value.get("segments")
    return [dict(row) for row in rows or [] if isinstance(row, dict)]


def _assess_single_pass_source(result: dict[str, Any], duration: float) -> None:
    result["execution_mode"] = "legacy_single_pass"
    result["execution_integrity"] = "reported_without_chunk_receipt"
    if duration >= LONG_SINGLE_PASS_SECONDS:
        result["issues"].append(
            _finding(
                "legacy_single_pass_long_media",
                "warning",
                f"{duration:.3f}s media used a single-pass ASR artifact without resumable chunk evidence",
            )
        )


def _assess_chunked_source(result: dict[str, Any], payload: dict[str, Any]) -> None:
    expected = _integer(payload.get("chunk_count"))
    successful_indexes = {
        _integer(value, default=-1)
        for value in payload.get("successful_chunk_indexes") or []
    }
    successful_indexes.discard(-1)
    reported_success = _integer(payload.get("successful_chunk_count"))
    successful = max(reported_success, len(successful_indexes))
    failed = _integer(payload.get("failed_chunk_count"))
    failed_rows = [
        row for row in payload.get("failed_chunks") or [] if isinstance(row, dict)
    ]
    gaps = [row for row in payload.get("gaps") or [] if isinstance(row, dict)]
    empty_chunk_indexes = _empty_chunk_indexes(payload)
    status = str(payload.get("status") or "")
    quality_status = str(payload.get("quality_status") or status)
    overlap_merge = (
        payload.get("overlap_merge")
        if isinstance(payload.get("overlap_merge"), dict)
        else {}
    )
    boundary_review_count = _integer(
        overlap_merge.get("boundary_review_required_count")
    )
    overlap = _number(
        payload.get("overlap_seconds")
        if payload.get("overlap_seconds") is not None
        else payload.get("chunk_overlap_seconds")
    )
    report_path = str(payload.get("report_path") or "").strip()
    report_exists = bool(report_path and Path(report_path).expanduser().is_file())
    # Intent: separate transport/execution completion from content-quality
    # degradation. Decision: a six-of-six completed run has passed execution
    # integrity even when overlap arbitration still needs review. Reason: the
    # latter already has its own fail-closed finding and should not be reported
    # a second time as missing chunks. Evidence: the Cantonese interview receipt
    # completed 6/6 chunks but had three boundary-review findings. Effective
    # scope: transcript source completeness diagnostics only.
    execution_complete = (
        expected > 0
        and successful == expected
        and failed == 0
        and not failed_rows
        and not gaps
        and not empty_chunk_indexes
        and status == "completed"
        and payload.get("ok") is True
    )
    result.update(
        {
            "execution_mode": "resumable_fixed_chunks",
            "execution_integrity": "passed" if execution_complete else "failed",
            "chunk_integrity": {
                "status": status,
                "quality_status": quality_status,
                "expected_chunk_count": expected,
                "successful_chunk_count": successful,
                "failed_chunk_count": failed,
                "failed_chunk_record_count": len(failed_rows),
                "gap_count": len(gaps),
                "unverified_empty_chunk_indexes": empty_chunk_indexes,
                "chunk_seconds": _number(payload.get("chunk_seconds")),
                "overlap_seconds": overlap,
                "report_path": report_path,
                "report_exists": report_exists,
                "resumed_from_checkpoint": bool(
                    payload.get("resumed_from_checkpoint")
                ),
                "boundary_review_required_count": boundary_review_count,
            },
        }
    )
    if not execution_complete:
        result["issues"].append(
            _finding(
                "asr_chunk_integrity_failed",
                "fail",
                f"chunk execution incomplete: status={status}, success={successful}/{expected}, failed={failed}, gaps={len(gaps)}, empty={empty_chunk_indexes}",
            )
        )
    if empty_chunk_indexes:
        result["issues"].append(
            _finding(
                "unverified_empty_asr_chunks",
                "fail",
                f"ASR returned no speech text/timestamps for chunk indexes {empty_chunk_indexes}; local VAD or audio-activity evidence is required before treating them as silence",
            )
        )
    if quality_status == "degraded" or boundary_review_count:
        result["issues"].append(
            _finding(
                "asr_chunk_boundary_review_required",
                "fail",
                (
                    "ASR source is degraded at chunk boundaries: "
                    f"quality_status={quality_status}, "
                    f"review_required={boundary_review_count}"
                ),
            )
        )
    if execution_complete and not report_exists:
        result["issues"].append(
            _finding(
                "asr_chunk_report_missing",
                "warning",
                "Chunked ASR completed but its machine-readable chunk report is missing",
            )
        )
    if expected > 1 and overlap <= 0:
        result["issues"].append(
            _finding(
                "asr_chunk_boundary_context_missing",
                "warning",
                "Fixed ASR chunks have no recorded overlap/context padding; boundary-word loss remains possible",
            )
        )


def _apply_boundary_review_evidence(
    result: dict[str, Any],
    root: Path,
    source_path: Path,
    payload: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    """Resolve fixed-chunk boundary warnings only with a source-bound review.

    Intent: close manually inspected overlap conflicts without rewriting the
    immutable raw ASR receipt.
    Decision: accept a separate review artifact only when its raw-source path
    and SHA-256 still match and its no-omission decisions exactly cover every
    boundary currently marked for review.
    Reason: editing ``quality_status`` in raw ASR would erase provenance, while
    trusting an unbound note would let stale review silently pass new output.
    Evidence: the 2026-08-31 fixed-chunk run retained 17 conservative review
    markers after all 46 chunks completed; one different auto-merged boundary
    contained a real omission and was repaired in the source-arbitrated layer.
    Effective scope: source-completeness interpretation only; raw ASR and
    canonical transcript text are never modified here.
    """

    overlap_merge = (
        payload.get("overlap_merge")
        if isinstance(payload.get("overlap_merge"), dict)
        else {}
    )
    expected = {
        (
            _integer(row.get("left_chunk_index"), default=-1),
            _integer(row.get("right_chunk_index"), default=-1),
        )
        for row in overlap_merge.get("boundaries") or []
        if isinstance(row, dict) and row.get("requires_human_review") is True
    }
    expected.discard((-1, -1))
    if not expected:
        result["boundary_review"] = {
            "status": "not_required",
            "expected_boundary_count": 0,
            "resolved_boundary_count": 0,
        }
        return

    candidates: list[Path] = []
    raw = str(manifest.get("asr_boundary_review_json") or "").strip()
    if raw:
        path = Path(raw).expanduser()
        candidates.append(path if path.is_absolute() else root / path)
    candidates.extend(
        [
            root / "asr-boundary-review.json",
            source_path.parent / "asr-boundary-review.json",
        ]
    )
    review_path = next((path.resolve() for path in candidates if path.is_file()), None)
    if review_path is None:
        result["boundary_review"] = {
            "status": "not_available",
            "expected_boundary_count": len(expected),
            "resolved_boundary_count": 0,
        }
        return

    review = read_json(review_path)
    source = review.get("source") if isinstance(review, dict) and isinstance(review.get("source"), dict) else {}
    source_review_path = Path(str(source.get("path") or "")).expanduser()
    source_path_matches = (
        source_review_path.is_absolute()
        and source_review_path.resolve() == source_path.resolve()
    )
    source_sha_matches = (
        str(source.get("sha256") or "").strip().lower()
        == sha256_file(source_path).lower()
    )
    decisions = review.get("decisions") if isinstance(review, dict) else []
    resolved = {
        (
            _integer(row.get("left_chunk_index"), default=-1),
            _integer(row.get("right_chunk_index"), default=-1),
        )
        for row in decisions or []
        if isinstance(row, dict)
        and str(row.get("status") or "") == "no_omission_detected"
        and str(row.get("action") or "") == "retain_merged_output"
        and row.get("missing_content_detected") is False
    }
    resolved.discard((-1, -1))
    accepted = (
        isinstance(review, dict)
        and review.get("schema") == BOUNDARY_REVIEW_SCHEMA
        and review.get("status") == "completed"
        and source_path_matches
        and source_sha_matches
        and resolved == expected
        and len(decisions or []) == len(expected)
    )
    result["boundary_review"] = {
        "status": "accepted" if accepted else "invalid_or_stale",
        "path": str(review_path),
        "source_path_matches": source_path_matches,
        "source_sha256_matches": source_sha_matches,
        "expected_boundary_count": len(expected),
        "resolved_boundary_count": len(resolved & expected),
        "exact_boundary_set_matches": resolved == expected,
    }
    if not accepted:
        return

    result["issues"] = [
        row
        for row in result.get("issues") or []
        if str(row.get("kind") or "") != "asr_chunk_boundary_review_required"
    ]
    chunk_integrity = result.get("chunk_integrity")
    if isinstance(chunk_integrity, dict):
        chunk_integrity["boundary_review_original_count"] = len(expected)
        chunk_integrity["boundary_review_resolved_count"] = len(expected)
        chunk_integrity["boundary_review_required_count"] = 0


def _load_bound_independent_vad(
    root: Path,
    source_path: Path,
    source_payload: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    candidates: list[Path] = []
    for key in (
        "silero_vad_candidate_json",
        "independent_vad_candidate_json",
    ):
        raw = str(manifest.get(key) or "").strip()
        if raw:
            path = Path(raw).expanduser()
            candidates.append(path if path.is_absolute() else root / path)
    candidates.extend(
        [
            root / "silero-vad-candidate.json",
            source_path.parent / "silero-vad-candidate.json",
        ]
    )
    path = next((item.resolve() for item in candidates if item.is_file()), None)
    if path is None:
        return {
            "verified": False,
            "segments": [],
            "summary": {
                "status": "not_available",
                "path": "",
                "source_bound": False,
            },
        }
    payload = read_json(path)
    summary: dict[str, Any] = {
        "status": "invalid",
        "path": str(path),
        "source_bound": False,
    }
    if not isinstance(payload, dict):
        return {"verified": False, "segments": [], "summary": summary}
    source_media = (
        payload.get("source_media")
        if isinstance(payload.get("source_media"), dict)
        else {}
    )
    raw_media = Path(str(source_payload.get("input") or "")).expanduser()
    candidate_media = Path(str(source_media.get("path") or "")).expanduser()
    source_bound = (
        raw_media.is_absolute()
        and candidate_media.is_absolute()
        and raw_media.resolve() == candidate_media.resolve()
        and raw_media.is_file()
        and str(source_media.get("sha256") or "").lower()
        == sha256_file(raw_media).lower()
    )
    verified = (
        payload.get("schema") == SILERO_CANDIDATE_SCHEMA
        and payload.get("status") == "completed"
        and payload.get("ok") is not False
        and payload.get("candidate_only") is True
        and source_bound
    )
    segments = payload.get("segments")
    import math
    intervals_valid = isinstance(segments, list) and all(
        isinstance(row, dict)
        and type(row.get("start")) in (int, float) and type(row.get("end")) in (int, float)
        and math.isfinite(row["start"]) and math.isfinite(row["end"])
        and 0 <= row["start"] < row["end"]
        for row in segments
    )
    verified = verified and intervals_valid
    execution_status = payload.get('status')
    execution_status = execution_status if isinstance(execution_status, str) else 'invalid_status_type'
    summary.update(
        {
            "status": "completed" if verified else (
                execution_status if execution_status in {"runtime_unavailable", "failed", "planned"}
                else "invalid_or_stale"),
            "execution_status": execution_status or 'unknown',
            "intervals_valid": intervals_valid,
            "source_bound": source_bound,
            "segment_count": len(segments) if isinstance(segments, list) else None,
            "upstream": payload.get("upstream") or {},
            "vad_settings": payload.get("vad_settings") or {},
        }
    )
    return {
        "verified": verified,
        "segments": [dict(row) for row in segments] if verified else [],
        "summary": summary,
    }


def _bound_word_alignment_coverage(
    root: Path, transcript: Path, segments: list[dict[str, Any]],
    raw_source: dict[str, Any], manifest: dict[str, Any], vad: dict[str, Any],
) -> dict[str, Any]:
    """Reconcile timestamp precision, never infer that supplied text is correct.

    Qwen's aligner config expresses time on an 80 ms grid; Silero v5/v6 uses
    512 samples at 16 kHz (32 ms). Their sum is a bounded observation collar,
    not a changed coverage floor. Both the strict and collared measurements
    are retained. Coarse/padded VAD, changed inputs, incomplete text, arbitrary
    resolutions and long gaps cannot establish this evidence.
    """
    import math
    result: dict[str, Any] = {"status": "not_available", "text_correctness_verified": False}
    settings = vad.get("summary", {}).get("vad_settings") or {}
    upstream = vad.get("summary", {}).get("upstream") or {}
    if (not vad.get("verified") or settings.get("speech_pad_ms") != 0
            or settings.get("sample_rate_hz") != 16000
            or not isinstance(settings.get("min_silence_duration_ms"), (int, float))
            or not 0 <= settings["min_silence_duration_ms"] <= 100
            or upstream.get("model") not in {"silero_v5_bundled_onnx", "silero_v6_bundled_onnx"}):
        return {**result, "reason": "unpadded_precision_vad_required"}
    transcript_hash = sha256_file(transcript)
    media = Path(str(raw_source.get("input") or "")).expanduser()
    if not media.is_absolute() or not media.is_file():
        return {**result, "reason": "source_media_missing"}
    media_hash = sha256_file(media)
    expected_text = compact_ascii_cjk("".join(str(row.get("text") or "") for row in segments))
    for entry in manifest.get("asr_alignment_sidecars") or []:
        if not isinstance(entry, dict) or entry.get("preset") != "qwen3-forced-aligner":
            continue
        try:
            path = Path(str(entry.get("path") or "")).expanduser()
            path = path if path.is_absolute() else root / path
            value = read_json(path)
            if (entry.get("transcript_sha256") != transcript_hash
                    or entry.get("sidecar_sha256") != sha256_file(path)
                    or value.get("schema") != "video_knowledge_pipeline.qwen3_forced_aligner_output.v1"
                    or value.get("status") != "completed" or value.get("ok") is not True
                    or value.get("transcript_sha256") != transcript_hash
                    or value.get("input_sha256") != media_hash
                    or Path(str(value.get("transcript_path") or "")).resolve() != transcript
                    or Path(str(value.get("input_path") or "")).resolve() != media.resolve()):
                result = {**result, "reason": "alignment_identity_invalid_or_stale"}
                continue
            resolution = value.get("timing_resolution_seconds")
            if (type(resolution) not in (int, float) or not math.isfinite(resolution)
                    or not 0 < resolution <= .1):
                result = {**result, "reason": "alignment_resolution_invalid"}
                continue
            words = value.get("words")
            if not isinstance(words, list) or not words or not expected_text:
                continue
            if compact_ascii_cjk("".join(str(row.get("text") or "") for row in words)) != expected_text:
                result = {**result, "reason": "alignment_word_text_incomplete"}
                continue
            duration = _number(raw_source.get("duration_seconds"))
            valid = []
            previous_end = 0.0
            zero_duration = 0
            for row in words:
                start, end = row.get("start"), row.get("end")
                if (type(start) not in (int, float) or type(end) not in (int, float)
                        or not math.isfinite(start) or not math.isfinite(end)
                        or start < previous_end - .001 or not 0 <= start <= end <= duration + resolution):
                    raise ValueError("alignment_word_timing_invalid")
                previous_end = end
                if end == start:
                    zero_duration += 1  # Never turn these words into broad coverage.
                else:
                    valid.append((start, end))
            target = [(row["start"], row["end"]) for row in vad["segments"]]
            options = {"minimum_gap_seconds": DEFAULT_VAD_COVERAGE_GAP_SECONDS,
                       "minimum_coverage_ratio": DEFAULT_VAD_MINIMUM_COVERAGE_RATIO}
            strict = interval_coverage(target, valid, **options)
            collar = float(resolution) + 512 / 16000
            adjusted = interval_coverage(target, [(max(0, s-collar), min(duration, e+collar))
                                                   for s,e in valid], **options)
            return {"status": adjusted["status"], "path": str(path), "sidecar_sha256": entry["sidecar_sha256"],
                    "transcript_sha256": transcript_hash, "source_media_sha256": media_hash,
                    "strict_coverage": strict, "coverage": adjusted,
                    "word_count": len(words), "zero_duration_word_count": zero_duration,
                    "aligner_resolution_seconds": resolution, "vad_frame_seconds": 512/16000,
                    "measurement_collar_seconds": collar, "thresholds_changed": False,
                    "text_correctness_verified": False}
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            result = {**result, "reason": "alignment_evidence_invalid"}
    return result


def _reconcile_verified_silent_chunks(
    result: dict[str, Any],
    payload: dict[str, Any],
    vad_segments: list[dict[str, Any]],
) -> None:
    """Use source-bound independent VAD to distinguish silence from ASR loss.

    Intent: keep empty speech chunks fail-closed without treating proven
    non-speech intervals as missing transcript.
    Decision: only reconcile ``unverified_empty_chunk`` failures when a
    source-hash-bound Silero candidate has no speech overlap with the exact
    chunk interval.
    Reason: process success is not completeness, but forcing ASR text for
    silence would fabricate content.
    Evidence: VKP already reuses faster-whisper's bundled Silero v5 model and
    records its model/input hashes in ``silero_vad_candidate.v1``.
    Effective scope: transcript quality interpretation only; raw ASR,
    checkpoints, and canonical transcript text remain unchanged.
    """

    failed_rows = [
        row for row in payload.get("failed_chunks") or [] if isinstance(row, dict)
    ]
    # Intent: keep historical chunked runs compatible with the stricter
    # empty-chunk gate without pretending that an empty ASR response is text.
    # Decision: treat legacy ``successful`` empty chunk rows as unresolved
    # silence candidates, then require the same exact source-bound Silero VAD
    # proof used for modern ``unverified_empty_chunk`` failures.
    # Reason: older runners counted empty child JSON as success, so no
    # ``failed_chunks`` row exists for the quality gate to reconcile.
    # Evidence: the 2026-07-24 production bundle records 21 successful chunks,
    # while chunks 1-3 contain neither text nor timestamps.
    # Effective scope: transcript quality interpretation only; raw ASR,
    # checkpoints, and canonical transcript text remain unchanged.
    legacy_empty_indexes = set(_empty_chunk_indexes(payload))
    if not failed_rows and not legacy_empty_indexes:
        return
    chunk_seconds = _number(payload.get("chunk_seconds"))
    speech_intervals = [
        (_number(row.get("start")), _number(row.get("end")))
        for row in vad_segments
        if isinstance(row, dict)
        and _number(row.get("end")) > _number(row.get("start"))
    ]
    verified: list[int] = []
    for row in failed_rows:
        if str(row.get("reason") or "") != "unverified_empty_chunk":
            continue
        index = _integer(row.get("chunk_index"), default=-1)
        if index < 0:
            continue
        start = _number(row.get("start"), index * chunk_seconds)
        end = _number(row.get("end"), start + chunk_seconds)
        if end <= start:
            continue
        if not any(
            speech_end > start and speech_start < end
            for speech_start, speech_end in speech_intervals
        ):
            verified.append(index)
    for index in sorted(legacy_empty_indexes):
        start = index * chunk_seconds
        end = min(
            _number(payload.get("duration_seconds"), start + chunk_seconds),
            start + chunk_seconds,
        )
        if end <= start:
            continue
        if not any(
            speech_end > start and speech_start < end
            for speech_start, speech_end in speech_intervals
        ):
            verified.append(index)
    if not verified:
        return

    verified_set = set(verified)
    expected = _integer(payload.get("chunk_count"))
    successful_indexes = {
        _integer(value, default=-1)
        for value in payload.get("successful_chunk_indexes") or []
    }
    successful_indexes.discard(-1)
    unresolved_indexes = {
        _integer(value, default=-1)
        for value in payload.get("unresolved_chunk_indexes") or []
    }
    unresolved_indexes.discard(-1)
    if not unresolved_indexes:
        unresolved_indexes = {
            _integer(row.get("chunk_index"), default=-1) for row in failed_rows
        }
        unresolved_indexes.discard(-1)
    remaining_failed = [
        row
        for row in failed_rows
        if _integer(row.get("chunk_index"), default=-1) not in verified_set
    ]
    remaining_gaps = [
        row
        for row in payload.get("gaps") or []
        if isinstance(row, dict)
        and _integer(row.get("chunk_index"), default=-1) not in verified_set
    ]
    # Legacy empty rows were already included in successful indexes. Remove
    # them first so only independently verified silence can restore coverage.
    effective_indexes = (successful_indexes - legacy_empty_indexes) | verified_set
    semantically_complete = (
        expected > 0
        and len(effective_indexes) == expected
        and not remaining_failed
        and not remaining_gaps
        and not (unresolved_indexes - verified_set)
    )

    chunk_integrity = (
        result.get("chunk_integrity")
        if isinstance(result.get("chunk_integrity"), dict)
        else {}
    )
    chunk_integrity.update(
        {
            "independent_vad_reconciled": True,
            "verified_silent_chunk_indexes": sorted(verified_set),
            "effective_successful_chunk_count": len(effective_indexes),
            "unresolved_chunk_indexes": sorted(unresolved_indexes - verified_set),
        }
    )
    result["chunk_integrity"] = chunk_integrity
    if not semantically_complete:
        return

    result["execution_integrity"] = "passed_with_verified_silence"
    result["issues"] = [
        row
        for row in result.get("issues") or []
        if row.get("kind")
        not in {"asr_chunk_integrity_failed", "unverified_empty_asr_chunks"}
    ]
    result["issues"].append(
        _finding(
            "empty_asr_chunks_verified_silence",
            "info",
            "Independent source-bound Silero VAD verified that empty ASR chunks "
            f"{sorted(verified_set)} contain no speech",
        )
    )


def _finding(kind: str, severity: str, detail: str) -> dict[str, str]:
    return {"kind": kind, "severity": severity, "detail": detail}


def _empty_chunk_indexes(payload: dict[str, Any]) -> list[int]:
    rows = payload.get("chunk_results")
    if not isinstance(rows, list):
        return []
    indexes: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        timestamps = row.get("timestamp")
        sentences = row.get("sentence_info")
        if text or (isinstance(timestamps, list) and timestamps) or (
            isinstance(sentences, list) and sentences
        ):
            continue
        index = _integer(row.get("chunk_index"), default=-1)
        if index >= 0:
            indexes.add(index)
    return sorted(indexes)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value if value is not None else default)
    except (TypeError, ValueError):
        return int(default)
