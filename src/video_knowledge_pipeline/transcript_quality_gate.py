from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Any

from .content_profile import profile_requirements
from .models import now_iso
from .quality_benchmark import transcript_quality_metrics
from .run_artifact_registry import register_bundle_run
from .storage import read_json, write_json
from .transcript import format_timestamp, parse_transcript
from .transcript_source_completeness import assess_transcript_source_completeness
from .transcript_speakers import cue_speaker

SCHEMA = "video_knowledge_pipeline.transcript_quality_gate.v1"
PUNCT = "。！？；：，,.!?;:"
HIGH_RISK_ENTITY_TYPES = {"company", "organization", "product", "tool", "person"}
KNOWN_BAD_TERMS = ["买虫", "二则一", "同意心", "明晚八点o", "明晚八点O", "更了解的价"]
KNOWN_GOOD_TERMS = ["买重", "二择一", "同理心", "明晚八点 OK", "更了解客户"]

QUALITY_LAYER_ORDER = (
    "source_integrity",
    "semantic_accuracy",
    "reader_projection",
)
SEMANTIC_ACCURACY_ISSUE_KINDS = {
    "cer_above_threshold",
    "entity_accuracy_low",
    "known_bad_term_residual",
    "number_errors",
    "overcorrection_rate_high",
    "semantic_evidence_pending",
    "unresolved_high_risk_term",
}
READER_PROJECTION_ISSUE_KINDS = {
    "colon_without_content",
    "missing_boundary_after_latin_token",
    "punctuation_artifact",
    "punctuation_density_high",
    "punctuation_density_low",
}


def run_transcript_quality_gate(
    bundle_dir: str | Path,
    *,
    input_path: str | Path | None = None,
    reference_path: str | Path | None = None,
    baseline_path: str | Path | None = None,
    min_punctuation_per_1000: float = 50.0,
    max_punctuation_per_1000: float = 140.0,
    max_cer: float = 0.18,
    min_entity_accuracy: float = 0.98,
    max_overcorrection_rate: float = 0.01,
    require_speaker_diarization: bool | None = None,
    min_speaker_count: int = 2,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    from .media_applicability import resolve_media_applicability
    applicability = resolve_media_applicability(root, manifest)
    source = _resolve_source(root, manifest, input_path)
    cues = parse_transcript(source)
    issues: list[dict[str, Any]] = []
    all_text = "\n".join(str(cue.text or "") for cue in cues)
    coverage = _audio_coverage(cues, expected_duration=_manifest_duration(manifest))
    source_completeness = assess_transcript_source_completeness(
        root,
        source,
        manifest=manifest,
    )
    for finding in source_completeness.get("issues") or []:
        if not isinstance(finding, dict):
            continue
        issues.append(
            _issue(
                str(finding.get("kind") or "asr_source_completeness"),
                str(finding.get("severity") or "warning"),
                -1,
                0,
                0,
                str(finding.get("detail") or ""),
            )
        )
    speaker_diarization = _speaker_diarization_status(
        cues,
        manifest,
        required=require_speaker_diarization,
        min_speaker_count=min_speaker_count,
    )
    if speaker_diarization["required"] and not speaker_diarization["passed"]:
        issues.append(
            _issue(
                "speaker_diarization_required",
                "fail",
                -1,
                0,
                0,
                (
                    f"speaker labels {speaker_diarization['labeled_segment_count']}/"
                    f"{speaker_diarization['spoken_segment_count']}; distinct "
                    f"{speaker_diarization['distinct_speaker_count']} < required "
                    f"{speaker_diarization['min_speaker_count']}"
                ),
            )
        )
    if coverage < 0.98 and not source_completeness.get("speech_completeness_verified"):
        issues.append(
            _issue(
                "audio_coverage_low",
                "fail",
                -1,
                0,
                0,
                f"audio coverage {coverage:.4f} < 0.98",
            )
        )
    punctuation_density = round(
        sum(ch in PUNCT for ch in all_text) * 1000 / max(1, len(all_text)), 2
    )
    if punctuation_density < float(min_punctuation_per_1000):
        issues.append(
            _issue(
                "punctuation_density_low",
                "warning",
                -1,
                0,
                0,
                f"punctuation density {punctuation_density} < {min_punctuation_per_1000}",
            )
        )
    if punctuation_density > float(max_punctuation_per_1000):
        issues.append(
            _issue(
                "punctuation_density_high",
                "warning",
                -1,
                0,
                0,
                f"punctuation density {punctuation_density} > {max_punctuation_per_1000}",
            )
        )
    for idx, cue in enumerate(cues):
        text = str(cue.text or "")
        stripped = text.strip()
        if not stripped:
            issues.append(
                _issue(
                    "empty_segment",
                    "fail",
                    idx,
                    cue.start,
                    cue.end,
                    "empty transcript segment",
                )
            )
            continue
        if _colon_without_content(stripped):
            issues.append(
                _issue(
                    "colon_without_content", "fail", idx, cue.start, cue.end, stripped
                )
            )
        if re.search(r"\bOK(?=[\u4e00-\u9fff])", stripped):
            issues.append(
                _issue(
                    "missing_boundary_after_latin_token",
                    "warning",
                    idx,
                    cue.start,
                    cue.end,
                    stripped,
                )
            )
        for pattern in ("，，", "，。", "：。", "好，的", "那，接下来"):
            if pattern in stripped:
                issues.append(
                    _issue(
                        "punctuation_artifact",
                        "warning",
                        idx,
                        cue.start,
                        cue.end,
                        pattern,
                    )
                )
        for term in KNOWN_BAD_TERMS:
            if term in stripped:
                issues.append(
                    _issue(
                        "known_bad_term_residual", "fail", idx, cue.start, cue.end, term
                    )
                )
    reference_metrics: dict[str, Any] = {}
    issues.extend(
        _unresolved_high_risk_term_issues(root, manifest, transcript_text=all_text)
    )
    semantic_evidence = _semantic_evidence_status(root, manifest)
    if semantic_evidence["required"]:
        issues.append(
            _issue(
                "semantic_evidence_pending",
                "fail",
                -1,
                0,
                0,
                f"{semantic_evidence['pending_candidate_count']} high-risk semantic candidate(s) lack independent local evidence; status={semantic_evidence['status']}",
            )
        )
    timestamp_metrics: dict[str, Any] = {}
    reference_source = _optional_source(root, reference_path)
    baseline_source = _optional_source(root, baseline_path)
    if reference_source:
        reference_cues = parse_transcript(reference_source)
        reference_text = "\n".join(str(cue.text or "") for cue in reference_cues)
        baseline_text = ""
        if baseline_source:
            baseline_text = "\n".join(
                str(cue.text or "") for cue in parse_transcript(baseline_source)
            )
        reference_metrics = transcript_quality_metrics(
            reference_text, all_text, source_text=baseline_text
        )
        timestamp_metrics = _timestamp_metrics(reference_cues, cues)
        reference_metrics.update(
            {
                "timestamp_median_error_seconds": timestamp_metrics.get("median"),
                "timestamp_p95_error_seconds": timestamp_metrics.get("p95"),
            }
        )
        if (
            reference_metrics.get("cer") is not None
            and float(reference_metrics["cer"]) > max_cer
        ):
            issues.append(
                _issue(
                    "cer_above_threshold",
                    "fail",
                    -1,
                    0,
                    0,
                    f"CER {reference_metrics['cer']} > {max_cer}",
                )
            )
        if (
            reference_metrics.get("entity_accuracy") is not None
            and float(reference_metrics["entity_accuracy"]) < min_entity_accuracy
        ):
            issues.append(
                _issue(
                    "entity_accuracy_low",
                    "fail",
                    -1,
                    0,
                    0,
                    f"entity accuracy {reference_metrics['entity_accuracy']} < {min_entity_accuracy}",
                )
            )
        if reference_metrics.get("number_error_count"):
            issues.append(
                _issue(
                    "number_errors",
                    "fail",
                    -1,
                    0,
                    0,
                    f"number errors {reference_metrics['number_error_count']}",
                )
            )
        if (
            reference_metrics.get("overcorrection_rate") is not None
            and float(reference_metrics["overcorrection_rate"])
            > max_overcorrection_rate
        ):
            issues.append(
                _issue(
                    "overcorrection_rate_high",
                    "fail",
                    -1,
                    0,
                    0,
                    f"overcorrection {reference_metrics['overcorrection_rate']} > {max_overcorrection_rate}",
                )
            )
        if (
            timestamp_metrics.get("median") is not None
            and float(timestamp_metrics["median"]) > 0.5
        ):
            issues.append(
                _issue(
                    "timestamp_median_error_high",
                    "warning",
                    -1,
                    0,
                    0,
                    f"timestamp median error {timestamp_metrics['median']} > 0.5 seconds",
                )
            )
        if (
            timestamp_metrics.get("p95") is not None
            and float(timestamp_metrics["p95"]) > 1.5
        ):
            issues.append(
                _issue(
                    "timestamp_p95_error_high",
                    "warning",
                    -1,
                    0,
                    0,
                    f"timestamp P95 error {timestamp_metrics['p95']} > 1.5 seconds",
                )
            )
    fail_count = sum(1 for row in issues if row.get("severity") == "fail")
    warning_count = sum(1 for row in issues if row.get("severity") == "warning")
    quality_layers = _quality_layer_summaries(issues)
    status = (
        "passed"
        if fail_count == 0 and warning_count == 0
        else ("warning" if fail_count == 0 else "failed")
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "media_applicability": applicability,
        "bundle_dir": str(root),
        "source_path": str(source),
        "status": status,
        "ok": fail_count == 0,
        "segment_count": len(cues),
        "issue_count": len(issues),
        "fail_count": fail_count,
        "warning_count": warning_count,
        "quality_layers": quality_layers,
        "punctuation_density_per_1000_chars": punctuation_density,
        "audio_coverage": coverage,
        "timeline_span_coverage": coverage,
        "audio_coverage_semantics": "timeline_span_only_not_speech_completeness",
        "timeline_span_gate_basis": (
            "source_bound_independent_speech_coverage" if source_completeness.get("speech_completeness_verified")
            else "timeline_span_fallback"
        ),
        "source_completeness": source_completeness,
        "speaker_diarization": speaker_diarization,
        "fidelity_policy": "source_fidelity_not_external_world_fact_check",
        "reference_path": str(reference_source) if reference_source else "",
        "baseline_path": str(baseline_source) if baseline_source else "",
        "reference_metrics": reference_metrics,
        "semantic_evidence": semantic_evidence,
        "known_bad_counts": {term: all_text.count(term) for term in KNOWN_BAD_TERMS},
        "known_good_counts": {term: all_text.count(term) for term in KNOWN_GOOD_TERMS},
        "issues": issues,
        "next_actions": _next_actions(status, issues),
        "artifacts": {
            "json": "transcript-quality-gate.json",
            "markdown": "transcript-quality-gate.md",
        },
        "updated_at": now_iso(),
    }
    if write:
        write_json(root / "transcript-quality-gate.json", result)
        (root / "transcript-quality-gate.md").write_text(
            _render_report(result), encoding="utf-8"
        )
        manifest["transcript_quality_gate_json"] = "transcript-quality-gate.json"
        manifest["transcript_quality_gate_markdown"] = "transcript-quality-gate.md"
        manifest["transcript_quality_gate_summary"] = {
            "status": status,
            "fail_count": fail_count,
            "warning_count": warning_count,
            "quality_layers": quality_layers,
            "source_completeness_status": source_completeness["status"],
            "speech_completeness_verified": source_completeness[
                "speech_completeness_verified"
            ],
            "speaker_diarization_status": speaker_diarization["status"],
            "speaker_diarization_required": speaker_diarization["required"],
            "distinct_speaker_count": speaker_diarization["distinct_speaker_count"],
            "semantic_evidence_status": semantic_evidence["status"],
            "semantic_evidence_pending_candidate_count": semantic_evidence["pending_candidate_count"],
            "updated_at": result["updated_at"],
        }
        write_json(root / "manifest.json", manifest)
    result["run_artifact"] = _register_run(root, result, write=write)
    return result


def _quality_layer_for_issue(issue: dict[str, Any]) -> str:
    kind = str(issue.get("kind") or "").strip()
    if kind in READER_PROJECTION_ISSUE_KINDS or kind.startswith("punctuation_"):
        return "reader_projection"
    if kind in SEMANTIC_ACCURACY_ISSUE_KINDS:
        return "semantic_accuracy"
    if any(
        token in kind
        for token in (
            "semantic",
            "entity",
            "number_error",
            "known_bad_term",
            "high_risk_term",
            "overcorrection",
        )
    ):
        return "semantic_accuracy"
    # Unknown upstream findings stay in source integrity. This is fail-closed:
    # new ASR/VAD/timing issues cannot silently disappear from all three
    # summaries, while the legacy flat issue list remains unchanged.
    return "source_integrity"


def _quality_layer_summaries(
    issues: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        layer: [] for layer in QUALITY_LAYER_ORDER
    }
    for row in issues:
        grouped[_quality_layer_for_issue(row)].append(row)
    result: dict[str, dict[str, Any]] = {}
    for layer in QUALITY_LAYER_ORDER:
        rows = grouped[layer]
        fail_count = sum(1 for row in rows if row.get("severity") == "fail")
        warning_count = sum(
            1 for row in rows if row.get("severity") == "warning"
        )
        issue_kind_counts: dict[str, int] = {}
        for row in rows:
            kind = str(row.get("kind") or "unknown")
            issue_kind_counts[kind] = issue_kind_counts.get(kind, 0) + 1
        status = (
            "passed"
            if fail_count == 0 and warning_count == 0
            else ("warning" if fail_count == 0 else "failed")
        )
        result[layer] = {
            "status": status,
            "ok": fail_count == 0,
            "issue_count": len(rows),
            "fail_count": fail_count,
            "warning_count": warning_count,
            "issue_kind_counts": dict(sorted(issue_kind_counts.items())),
            "empty_segment_count": issue_kind_counts.get("empty_segment", 0),
        }
    return result


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    data = read_json(path) if path.exists() else {}
    return data if isinstance(data, dict) else {}


def _resolve_source(
    root: Path, manifest: dict[str, Any], input_path: str | Path | None
) -> Path:
    if input_path:
        path = Path(input_path).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(f"transcript not found: {path}")
    for key in (
        "source_arbitrated_transcript_json",
        "human_corrected_transcript_json",
        "llm_readable_transcript_json",
        "agent_readable_transcript_json",
        "readable_transcript_json",
        "llm_corrected_transcript_json",
        "corrected_transcript_json",
        "postprocessed_transcript_json",
        "normalized_transcript_json",
        "transcript_json",
    ):
        value = str(manifest.get(key) or "").strip()
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return path.resolve()
    for name in (
        "source-arbitrated-transcript.json",
        "human-corrected-transcript.json",
        "llm-readable-transcript.json",
        "agent-readable-transcript.json",
        "readable-transcript.json",
        "llm-corrected-transcript.json",
        "corrected-transcript.json",
        "postprocessed-transcript.json",
        "normalized-transcript.json",
    ):
        path = root / name
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"no transcript sidecar found for bundle: {root}")


def _optional_source(root: Path, value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(f"transcript reference not found: {path}")
    return path.resolve()


def _speaker_diarization_status(
    cues: list[Any],
    manifest: dict[str, Any],
    *,
    required: bool | None,
    min_speaker_count: int,
) -> dict[str, Any]:
    """Assess speaker labeling without running or reimplementing diarization.

    Intent: fail closed when a multi-speaker transcript is required.
    Decision: reuse upstream speaker clusters and measure their preservation.
    Reason: an unlabeled transcript cannot meet the user's dialogue-recording
    contract even when ASR text itself is complete.
    Evidence: MOSS segments.json and WhisperX-style diarization both expose a
    segment-level speaker identity.
    Effective scope: transcript quality status only; speaker names and roles are
    never inferred here.
    """

    requirements = manifest.get("transcript_requirements")
    requirements = requirements if isinstance(requirements, dict) else {}
    declared_expected = 0
    for value in (
        requirements.get("expected_speaker_count"),
        manifest.get("expected_speaker_count"),
        manifest.get("participant_count"),
    ):
        try:
            declared_expected = max(declared_expected, int(value or 0))
        except (TypeError, ValueError):
            continue
    if required is None:
        profile_id = str(
            manifest.get("content_profile")
            or manifest.get("video_content_profile")
            or "course-or-general-v1"
        )
        try:
            profile_requires_speakers = bool(
                profile_requirements(profile_id).get("speaker_diarization_required")
            )
        except ValueError:
            # Unknown profiles never weaken explicit requirements; the profile
            # validator reports the invalid identifier separately.
            profile_requires_speakers = False
        required = bool(
            requirements.get("speaker_diarization_required")
            or manifest.get("speaker_diarization_required")
            or declared_expected > 1
            or profile_requires_speakers
        )
    minimum = max(1, int(min_speaker_count or 1), declared_expected)
    spoken = [cue for cue in cues if str(getattr(cue, "text", "") or "").strip()]
    labeled = [cue for cue in spoken if cue_speaker(cue)]
    identities = sorted({cue_speaker(cue) for cue in labeled if cue_speaker(cue)})
    spoken_duration = sum(max(0.0, float(cue.end) - float(cue.start)) for cue in spoken)
    labeled_duration = sum(max(0.0, float(cue.end) - float(cue.start)) for cue in labeled)
    label_coverage = round(len(labeled) / max(1, len(spoken)), 6)
    duration_coverage = round(labeled_duration / max(0.001, spoken_duration), 6)
    passed = not required or (
        len(labeled) == len(spoken)
        and len(identities) >= minimum
    )
    return {
        "status": "not_required" if not required else ("passed" if passed else "speaker_diarization_required"),
        "required": bool(required),
        "passed": passed,
        "min_speaker_count": minimum,
        "spoken_segment_count": len(spoken),
        "labeled_segment_count": len(labeled),
        "segment_label_coverage": label_coverage,
        "duration_label_coverage": duration_coverage,
        "distinct_speaker_count": len(identities),
        "speaker_ids": identities,
        "role_inference_required": False,
    }


def _manifest_duration(manifest: dict[str, Any]) -> float:
    for key in ("duration_seconds", "media_duration_seconds", "video_duration_seconds"):
        try:
            value = float(manifest.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0.0


def _audio_coverage(cues: list[Any], *, expected_duration: float = 0.0) -> float:
    if not cues:
        return 0.0
    first_start = min(max(0.0, float(cue.start)) for cue in cues)
    last_end = max(max(0.0, float(cue.end)) for cue in cues)
    duration = max(float(expected_duration or 0.0), last_end)
    if duration <= 0:
        return 0.0
    # Internal VAD gaps are silence, not missing transcript. Coverage measures
    # whether the transcript spans the media from its first to last cue.
    return round(max(0.0, last_end - first_start) / duration, 6)


def _timestamp_metrics(
    reference: list[Any], candidate: list[Any]
) -> dict[str, float | None]:
    if not reference or not candidate:
        return {"median": None, "p95": None}
    errors: list[float] = []
    for ref in reference:
        overlapping = [
            cue
            for cue in candidate
            if max(float(ref.start), float(cue.start))
            <= min(float(ref.end), float(cue.end))
        ]
        if overlapping:
            best = max(
                overlapping,
                key=lambda cue: (
                    min(float(ref.end), float(cue.end))
                    - max(float(ref.start), float(cue.start))
                ),
            )
        else:
            ref_mid = (float(ref.start) + float(ref.end)) / 2
            best = min(
                candidate,
                key=lambda cue: abs(
                    ((float(cue.start) + float(cue.end)) / 2) - ref_mid
                ),
            )
        errors.append(abs(float(ref.start) - float(best.start)))
    if not errors:
        return {"median": None, "p95": None}
    ordered = sorted(errors)
    p95_index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "median": round(statistics.median(errors), 6),
        "p95": round(ordered[p95_index], 6),
    }


def _unresolved_high_risk_term_issues(
    root: Path,
    manifest: dict[str, Any],
    *,
    transcript_text: str,
) -> list[dict[str, Any]]:
    value = str(manifest.get("entity_lexicon_json") or "entity-lexicon.json").strip()
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        return []
    payload = read_json(path)
    if not isinstance(payload, dict):
        return []
    rows = payload.get("unresolved_high_risk_terms")
    if not isinstance(rows, list):
        rows = [
            row
            for row in payload.get("correction_candidates") or []
            if isinstance(row, dict)
            and not bool(row.get("auto_apply_allowed"))
            and str(row.get("entity_type") or "") in HIGH_RISK_ENTITY_TYPES
        ]
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        original = str(row.get("original_text") or "").strip()
        corrected = str(row.get("corrected_text") or "").strip()
        if not original or not corrected or original == corrected:
            continue
        if original not in transcript_text:
            continue
        candidate_id = str(row.get("candidate_id") or "")
        identity = candidate_id or f"{original}->{corrected}"
        if identity in seen:
            continue
        seen.add(identity)
        try:
            index = int(row.get("timeline_index") or -1)
        except (TypeError, ValueError):
            index = -1
        issues.append(
            _issue(
                "unresolved_high_risk_term",
                "fail",
                index,
                0,
                0,
                f"{identity}: unresolved {original} -> {corrected}",
            )
        )
    return issues


def _semantic_evidence_status(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Return whether high-risk semantic candidates have independent evidence."""

    plan_name = str(manifest.get("local_targeted_asr_plan_json") or "local-targeted-asr-plan.json").strip()
    plan_path = Path(plan_name)
    if not plan_path.is_absolute():
        plan_path = root / plan_path
    plan = read_json(plan_path) if plan_path.is_file() else {}
    if isinstance(plan, dict) and plan:
        retry_plan = plan.get("retry_plan") if isinstance(plan.get("retry_plan"), dict) else {}
        selected = [
            row
            for row in plan.get("selected_candidates") or []
            if isinstance(row, dict) and _has_concrete_alternative(row)
        ]
        # Older plans may have recorded only generic number/action heuristics.
        # They are not actionable correction requests and must not turn a
        # clean transcript into a gate failure solely because a stale plan
        # happens to exist.
        pending = len(selected)
        plan_status = str(plan.get("status") or "missing")
        if pending > 0 and plan_status in {"planned", "in_progress", "degraded", "failed"}:
            return {
                "status": plan_status,
                "required": True,
                "pending_candidate_count": pending,
                "window_count": int(retry_plan.get("window_count") or pending),
                "plan_path": str(plan_path),
            }
        return {
            "status": plan_status,
            "required": False,
            "pending_candidate_count": 0,
            "window_count": pending,
            "plan_path": str(plan_path),
        }

    pack_path = root / str(manifest.get("transcript_semantic_correction_pack_json") or "transcript-semantic-correction-pack.json")
    pack = read_json(pack_path) if pack_path.is_file() else {}
    candidates = pack.get("candidates") if isinstance(pack, dict) else []
    external = {
        "secondary_asr", "platform_subtitle", "embedded_subtitle", "ocr",
        "structured_visual", "visual_understanding", "temporal_visual", "human_note",
    }
    pending = 0
    for row in candidates or []:
        if not isinstance(row, dict):
            continue
        evidence_types = {str(value) for value in row.get("evidence_source_types") or [] if str(value)}
        high_risk = str(row.get("risk_level") or "") == "high" or str(row.get("correction_type") or "") in {"number", "proper_noun", "term", "action"}
        if high_risk and _has_concrete_alternative(row) and not evidence_types.intersection(external) and str(row.get("llm_review_defer_reason") or "") == "needs_conflicting_external_evidence":
            pending += 1
    return {
        "status": "needs_local_targeted_evidence_plan" if pending else "not_required",
        "required": bool(pending),
        "pending_candidate_count": pending,
        "window_count": 0,
        "plan_path": str(plan_path),
    }


def _has_concrete_alternative(row: dict[str, Any]) -> bool:
    return bool(str(row.get("candidate_text") or row.get("suggested_text") or "").strip())


def _colon_without_content(text: str) -> bool:
    value = text.strip()
    return value.endswith(
        ("说：", "客户说：", "顾问说：", "是这样说的：", "是这样回的说：")
    )


def _issue(
    kind: str, severity: str, index: int, start: float, end: float, detail: str
) -> dict[str, Any]:
    return {
        "kind": kind,
        "severity": severity,
        "index": index,
        "start": start,
        "end": end,
        "time": f"{format_timestamp(float(start or 0))} - {format_timestamp(float(end or 0))}"
        if index >= 0
        else "global",
        "detail": detail,
    }


def _next_actions(status: str, issues: list[dict[str, Any]]) -> list[str]:
    if status == "passed":
        return ["Use this transcript for full-transcript.md and smart-summary input."]
    actions = []
    kinds = {str(row.get("kind")) for row in issues}
    if "known_bad_term_residual" in kinds or "unresolved_high_risk_term" in kinds:
        actions.append(
            "Run transcript-evidence-correction-pipeline or import semantic review notes before export."
        )
    if "semantic_evidence_pending" in kinds:
        actions.append(
            "Build local-targeted-asr-plan, extract only its clips, and obtain an independent local ASR before final export."
        )
    if "unverified_empty_asr_chunks" in kinds:
        actions.append(
            "Run local audio-activity/VAD evidence for the empty chunks, then retry only chunks with speech activity."
        )
    if "speech_completeness_unverified" in kinds:
        actions.append(
            "Run the existing faster-whisper Silero VAD candidate and bind it to this source before claiming speech completeness."
        )
    if "speaker_diarization_required" in kinds:
        actions.append(
            "Run the explicit MOSS/WhisperX-compatible diarization route, then rerun the gate; do not infer names or merge different speaker clusters."
        )
    if (
        "colon_without_content" in kinds
        or "punctuation_artifact" in kinds
        or "missing_boundary_after_latin_token" in kinds
    ):
        actions.append(
            "Run agent-readable-transcript-rewrite, then rerun transcript-quality-gate."
        )
    if "punctuation_density_low" in kinds:
        actions.append(
            "Run agent-readable-transcript-rewrite or a real LLM punctuation pass."
        )
    return actions or [
        "Inspect transcript-quality-gate.md and decide whether to accept known gaps."
    ]


def _render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Transcript Quality Gate",
        "",
        f"- Status: `{result.get('status')}`",
        f"- OK: `{result.get('ok')}`",
        f"- Source: `{result.get('source_path')}`",
        f"- Visual applicability: `{(result.get('media_applicability') or {}).get('visual_status', 'required')}` (speech checks remain required)",
        f"- Segments: `{result.get('segment_count')}`",
        f"- Punctuation density: `{result.get('punctuation_density_per_1000_chars')}` / 1000 chars",
        f"- Timeline span coverage: `{result.get('timeline_span_coverage')}` (not speech completeness)",
        f"- Source completeness: `{result.get('source_completeness') or 'not_evaluated'}`",
        f"- Speaker diarization: `{result.get('speaker_diarization') or 'not_required'}`",
        f"- Fidelity policy: `{result.get('fidelity_policy')}`",
        f"- Reference metrics: `{result.get('reference_metrics') or 'not_provided'}`",
        f"- Semantic evidence: `{result.get('semantic_evidence') or 'not_required'}`",
        f"- Fail / warning: `{result.get('fail_count')}` / `{result.get('warning_count')}`",
        "",
        "## Quality Layers",
        "",
        "| Layer | Status | Fail | Warning | Issues |",
        "|---|---:|---:|---:|---:|",
    ]
    quality_layers = result.get("quality_layers")
    quality_layers = quality_layers if isinstance(quality_layers, dict) else {}
    for layer in QUALITY_LAYER_ORDER:
        summary = quality_layers.get(layer)
        summary = summary if isinstance(summary, dict) else {}
        lines.append(
            f"| `{layer}` | `{summary.get('status') or 'not_evaluated'}` | "
            f"{int(summary.get('fail_count') or 0)} | "
            f"{int(summary.get('warning_count') or 0)} | "
            f"{int(summary.get('issue_count') or 0)} |"
        )
    lines += ["", "### Issue Counts by Layer", ""]
    for layer in QUALITY_LAYER_ORDER:
        summary = quality_layers.get(layer)
        summary = summary if isinstance(summary, dict) else {}
        counts = summary.get("issue_kind_counts")
        counts = counts if isinstance(counts, dict) else {}
        detail = ", ".join(
            f"`{kind}`={int(count)}" for kind, count in counts.items()
        ) or "none"
        lines.append(f"- `{layer}`: {detail}")
    lines += [
        "",
        "## Next Actions",
        "",
    ]
    lines.extend(f"- {item}" for item in result.get("next_actions") or [])
    lines += ["", "## Issues", ""]
    for row in result.get("issues") or []:
        lines.append(
            f"- `{row.get('severity')}` `{row.get('kind')}` `{row.get('time')}` {row.get('detail')}"
        )
    lines.append("")
    return "\n".join(lines)


def _register_run(root: Path, result: dict[str, Any], *, write: bool) -> dict[str, Any]:
    if not write:
        return {}
    artifacts = [
        value
        for value in (result.get("artifacts") or {}).values()
        if isinstance(value, str) and value
    ]
    return register_bundle_run(
        root,
        run_type="transcript-quality-gate",
        status=str(result.get("status") or "unknown"),
        title="Transcript quality gate",
        summary=f"Transcript quality gate {result.get('status')} with {result.get('fail_count')} failures and {result.get('warning_count')} warnings.",
        inputs={"source_path": result.get("source_path")},
        parameters={
            "min_punctuation_per_1000": 50.0,
            "max_punctuation_per_1000": 140.0,
        },
        artifacts=artifacts,
        failed_items=[
            row for row in result.get("issues") or [] if row.get("severity") == "fail"
        ],
        next_actions=result.get("next_actions") or [],
        operator_boundary={
            "local_only": True,
            "no_cloud_call": True,
            "does_not_modify_raw_asr": True,
        },
    )
