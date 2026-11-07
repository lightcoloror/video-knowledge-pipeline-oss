from __future__ import annotations

import difflib
import html
import json
import os
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

from .asr_runner import _resolve_python_executable, default_local_asr_device
from .audio_silence_probe import probe_audio_silence
from .media_tools import resolve_media_tool
from .models import now_iso
from .numeric_normalization import number_evidence_map as _number_evidence_map, strip_number_mentions as _strip_number_mentions
from .storage import read_json, write_json
from .transcript import format_timestamp, parse_transcript, transcript_excerpt
from .transcript_postprocess import format_asr_review_draft


MANIFEST_SCHEMA = "video_knowledge_pipeline.quality_benchmark_manifest.v1"
RUN_SCHEMA = "video_knowledge_pipeline.quality_benchmark_run.v1"
PUNCTUATION = "，。！？；：、,.!?;:"
ALIGNED_WINDOW_STRATEGY = "asr_vad_sentence_aligned_v1"
LEGACY_WINDOW_STRATEGY = "legacy_fixed_window"
SENTENCE_TERMINATORS = "。！？!?；;"


def build_quality_benchmark(
    output_dir: str | Path,
    *,
    bundle_dirs: list[str | Path],
    media_paths: list[str | Path] | None = None,
    samples_per_bundle: int = 8,
    sample_seconds: float = 60.0,
    asr_language: str = "zh",
    asr_context_hotwords: list[str] | None = None,
    execute_clips: bool = False,
    legacy_reference_manifest: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Create a stratified human-reference benchmark template from bundles."""

    out = Path(output_dir).expanduser().resolve()
    # Intent: make domain-context A/B runs reproducible without teaching the
    # candidate the expected transcript. Decision: persist only an explicit,
    # deduplicated vocabulary list in every sample. Reason: Qwen3-ASR supports
    # context natively while SenseVoice remains the unchanged fast baseline.
    # Evidence: qwen3_asr_python_runner maps the plan hotword field to the
    # upstream context argument. Effective scope: local benchmark candidates.
    context_hotwords = list(
        dict.fromkeys(
            str(value).strip()
            for value in (asr_context_hotwords or [])
            if str(value).strip()
        )
    )
    samples: list[dict[str, Any]] = []
    legacy_references = _legacy_reference_rows(legacy_reference_manifest)
    explicit_media = list(media_paths or [])
    clip_dir = out / "clips"
    for bundle_index, bundle_value in enumerate(bundle_dirs):
        bundle = Path(bundle_value).expanduser().resolve()
        manifest = _mapping(bundle / "manifest.json")
        sampling_source = _default_transcript(bundle, manifest)
        if not sampling_source:
            continue
        variant_paths = _benchmark_variant_paths(bundle, manifest, fallback=sampling_source)
        baseline_source = variant_paths["sensevoice_full_punc"] or variant_paths["sensevoice_raw"] or sampling_source
        review_draft_source = variant_paths["sensevoice_raw"] or baseline_source
        cues = parse_transcript(sampling_source)
        review_draft_cues = parse_transcript(review_draft_source)
        duration = max((float(cue.end) for cue in cues), default=0.0)
        if duration <= 0:
            continue
        count = max(1, int(samples_per_bundle or 1))
        bundle_label = _benchmark_bundle_label(bundle)
        media = _benchmark_media_path(bundle, manifest, explicit_media[bundle_index] if bundle_index < len(explicit_media) else None)
        vad_segments: list[dict[str, float]] = []
        vad_artifact = ""
        if execute_clips and media:
            vad_path = out / "vad" / f"{_safe_name(bundle_label)}.json"
            vad_segments, vad_artifact = _load_or_run_vad_segments(media, vad_path)
        for sample_index, (start, end, category) in enumerate(_stratified_windows(cues, duration=duration, count=count, sample_seconds=sample_seconds)):
            audio_boundary = {}
            if vad_segments:
                start, end, audio_boundary = _align_window_to_vad_segments(
                    vad_segments,
                    start=start,
                    end=end,
                    duration=duration,
                )
            elif execute_clips and media:
                start, end, audio_boundary = _align_window_to_audio_silence(
                    media,
                    start=start,
                    end=end,
                    duration=duration,
                )
            excerpt = transcript_excerpt(cues, start, end)
            asr_draft_text = format_asr_review_draft(
                review_draft_cues,
                start_seconds=start,
                end_seconds=end,
            ) or excerpt
            boundary_alignment = _window_boundary_metadata(cues, start=start, end=end, duration=duration)
            if audio_boundary.get("ready"):
                source = str(audio_boundary.get("source") or "ffmpeg_silencedetect")
                boundary_alignment = {
                    **boundary_alignment,
                    **audio_boundary,
                    "ready": True,
                    "status": str(audio_boundary.get("status") or "audio_boundary_aligned"),
                    "source": source,
                }
            sample_id = f"{bundle_label}-{sample_index + 1:02d}"
            legacy_row = legacy_references.get(sample_id) or {}
            legacy_reference_text = _reference_text(legacy_row) if legacy_row else ""
            legacy_start = float(legacy_row.get("start_seconds") or start) if legacy_row else start
            legacy_end = float(legacy_row.get("end_seconds") or end) if legacy_row else end
            boundary_extension_required = bool(
                legacy_reference_text
                and (abs(legacy_start - start) > 0.02 or abs(legacy_end - end) > 0.02)
            )
            clip_path = clip_dir / f"{_safe_name(sample_id)}.wav"
            clip_error = ""
            if execute_clips and media:
                clip_error = _write_audio_clip(media, clip_path, start=start, end=end)
            samples.append(
                {
                    "sample_id": sample_id,
                    "bundle_dir": str(bundle),
                    "category": category or _sample_category(excerpt),
                    "window_strategy": ALIGNED_WINDOW_STRATEGY,
                    "nominal_sample_seconds": float(sample_seconds),
                    "actual_sample_seconds": round(end - start, 3),
                    "boundary_alignment": boundary_alignment,
                    "start_seconds": round(start, 3),
                    "end_seconds": round(end, 3),
                    "media_path": str(media or ""),
                    "audio_clip_path": str(clip_path.resolve()) if clip_path.exists() else "",
                    "audio_clip_error": clip_error,
                    "asr_language": str(asr_language or "zh").strip(),
                    "context_hotwords": context_hotwords,
                    "vad_artifact_path": vad_artifact,
                    "source_transcript": str(baseline_source),
                    "reference_transcript": "",
                    "reference_text": "",
                    "asr_draft_text": asr_draft_text,
                    "asr_draft_source": str(review_draft_source),
                    "draft_source": "current_asr_transcript_punctuation_only",
                    "legacy_reference_text": legacy_reference_text,
                    "legacy_window": {"start": legacy_start, "end": legacy_end} if legacy_reference_text else {},
                    "boundary_extension_required": boundary_extension_required,
                    "reference_start_seconds": start,
                    "reference_end_seconds": end,
                    "protected_entities": _entities(excerpt),
                    "source_variants": {
                        "sensevoice_raw": str(variant_paths["sensevoice_raw"] or ""),
                        "sensevoice_full_punc": str(variant_paths["sensevoice_full_punc"] or ""),
                        "corrected_transcript": str(variant_paths["corrected_transcript"] or ""),
                    },
                    "variants": {
                        "sensevoice_raw": "" if execute_clips else str(variant_paths["sensevoice_raw"] or ""),
                        "sensevoice_full_punc": "" if execute_clips else str(variant_paths["sensevoice_full_punc"] or ""),
                        "qwen3_asr_0_6b": "",
                        "qwen3_asr_1_7b": "",
                        "fun_asr_nano": "",
                        "corrected_transcript": "" if execute_clips else str(variant_paths["corrected_transcript"] or ""),
                    },
                    "human_review_status": "needs_boundary_extension" if boundary_extension_required else "asr_prefilled_todo",
                }
            )
    result = {
        "schema": MANIFEST_SCHEMA,
        "created_at": now_iso(),
        "output_dir": str(out),
        "bundle_count": len({row["bundle_dir"] for row in samples}),
        "sample_count": len(samples),
        "audio_clip_count": sum(1 for row in samples if row.get("audio_clip_path")),
        "legacy_reference_count": sum(1 for row in samples if row.get("legacy_reference_text")),
        "window_strategy": ALIGNED_WINDOW_STRATEGY,
        "nominal_sample_seconds": float(sample_seconds),
        "asr_language": str(asr_language or "zh").strip(),
        "context_hotwords": context_hotwords,
        "window_alignment_ready": bool(samples) and all(bool((row.get("boundary_alignment") or {}).get("ready")) for row in samples),
        "candidate_variant": "qwen3_asr_1_7b",
        "reference_role": "human_evaluation_only_never_correction_evidence",
        "summary_blind_review": {
            "required": True,
            "target_mean_improvement": 0.5,
            "target_reference_readability_gap": 0.3,
            "items": _summary_blind_review_items(bundle_dirs),
        },
        "samples": samples,
        "operator_boundary": {"local_only": True, "does_not_run_asr": True, "reference_never_imported_as_evidence": True, "clips_require_explicit_execute": True},
    }
    if write:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "quality-benchmark-manifest.json", result)
        (out / "quality-benchmark-todo.md").write_text(_render_todo(result), encoding="utf-8")
        (out / "quality-benchmark-review.html").write_text(_render_review_html(result), encoding="utf-8")
    return result


def run_quality_benchmark(manifest_json: str | Path, *, output_dir: str | Path | None = None, write: bool = True) -> dict[str, Any]:
    path = Path(manifest_json).expanduser().resolve()
    manifest = read_json(path)
    if not isinstance(manifest, dict):
        raise ValueError("quality benchmark manifest must be a JSON object")
    out = Path(output_dir).expanduser().resolve() if output_dir else path.parent
    manifest_window_strategy = str(manifest.get("window_strategy") or LEGACY_WINDOW_STRATEGY)
    sample_results: list[dict[str, Any]] = []
    variant_rows: dict[str, list[dict[str, Any]]] = {}
    for sample in manifest.get("samples") or []:
        if not isinstance(sample, dict):
            continue
        reference_text = _reference_text(sample)
        source_text = _window_text(sample.get("source_transcript"), sample)
        sample_window_strategy = str(sample.get("window_strategy") or manifest_window_strategy)
        boundary_alignment = sample.get("boundary_alignment") if isinstance(sample.get("boundary_alignment"), dict) else {}
        sample_window_alignment_ready = sample_window_strategy == ALIGNED_WINDOW_STRATEGY and bool(boundary_alignment.get("ready"))
        reference_completeness = _reference_completeness(
            reference_text,
            source_text,
            window_strategy=sample_window_strategy,
            human_review_status=str(sample.get("human_review_status") or ""),
            reviewed_window_matches=_reviewed_window_matches(sample),
        )
        variant_metrics: dict[str, Any] = {}
        variant_texts: dict[str, str] = {}
        for key, variant_path in (sample.get("variants") or {}).items():
            candidate_text, candidate_cues = _window_text_and_cues(variant_path, sample)
            metrics = _metrics(
                reference_text=reference_text,
                source_text=source_text,
                candidate_text=candidate_text,
                candidate_cues=candidate_cues,
                sample=sample,
            )
            metrics["available"] = bool(candidate_text)
            variant_texts[str(key)] = candidate_text
            metrics["transcript_path"] = str(variant_path or "")
            variant_metrics[str(key)] = metrics
            if reference_completeness["ready"] and candidate_text:
                variant_rows.setdefault(str(key), []).append(metrics)
        # Intent: make the local challenger comparison useful when a benchmark
        # was seeded from an already completed production SenseVoice run.
        # Decision: prefer an independently executed SenseVoice variant, then
        # fall back to the source-bound ``asr_draft_text`` stored in the sample.
        # Reason: the Cantonese detail pack deliberately avoided rerunning the
        # same SenseVoice audio, but the previous implementation consequently
        # reported ``compared_count=0`` even though both hypotheses existed.
        # Evidence: quality_benchmark.build records ``asr_draft_source`` and
        # ``asr_draft_text`` from the current aligned window; the review UI
        # already treats that draft as candidate evidence rather than truth.
        # Effective scope: evaluation-only disagreement priority and review UI;
        # no transcript promotion, correction, fallback, or model call.
        baseline_asr_text = _baseline_asr_text(variant_texts, sample)
        disagreement = _asr_disagreement(
            baseline_asr_text,
            variant_texts.get("qwen3_asr_1_7b") or "",
        )
        sample["asr_disagreement"] = disagreement
        if variant_texts.get("qwen3_asr_1_7b"):
            sample["qwen_draft_text"] = variant_texts["qwen3_asr_1_7b"]
        review_seed = _merged_review_seed(
            str(sample.get("legacy_reference_text") or ""),
            str(sample.get("asr_draft_text") or ""),
        )
        sample["review_seed_text"] = review_seed["text"]
        sample["review_seed_metadata"] = {key: value for key, value in review_seed.items() if key != "text"}
        sample_results.append(
            {
                "sample_id": sample.get("sample_id"),
                "category": sample.get("category"),
                "reference_ready": reference_completeness["ready"],
                "reference_completeness": reference_completeness,
                "window_strategy": sample_window_strategy,
                "window_alignment_ready": sample_window_alignment_ready,
                "boundary_alignment": boundary_alignment,
                "asr_disagreement": disagreement,
                "window": {"start": sample.get("start_seconds"), "end": sample.get("end_seconds")},
                "variants": variant_metrics,
            }
        )
    disagreement_rows = [
        row["asr_disagreement"]
        for row in sample_results
        if isinstance(row.get("asr_disagreement"), dict) and row["asr_disagreement"].get("available")
    ]
    disagreement_summary = {
        "compared_count": len(disagreement_rows),
        "mean_edit_ratio": round(statistics.mean(float(row.get("edit_ratio") or 0.0) for row in disagreement_rows), 6) if disagreement_rows else None,
        "high_disagreement_count": sum(str(row.get("review_priority") or "") == "high" for row in disagreement_rows),
        "number_conflict_count": sum(bool(row.get("number_conflicts")) for row in disagreement_rows),
        "priority_sample_ids": [
            str(row.get("sample_id") or "")
            for row in sample_results
            if str((row.get("asr_disagreement") or {}).get("review_priority") or "") == "high"
        ],
    }
    aggregate = {key: _aggregate(rows) for key, rows in variant_rows.items()}
    baseline = aggregate.get("sensevoice_full_punc") or aggregate.get("sensevoice_raw") or {}
    candidate_variant = str(manifest.get("candidate_variant") or "qwen3_asr_1_7b")
    candidate = aggregate.get(candidate_variant) or {}
    reference_count = sum(1 for row in sample_results if row["reference_ready"])
    candidate_available_count = sum(
        1
        for row in sample_results
        if bool((row.get("variants") or {}).get(candidate_variant, {}).get("available"))
    )
    candidate_run_failures = [
        {
            "sample_id": str(sample.get("sample_id") or ""),
            "status": str(((sample.get("variant_runs") or {}).get(candidate_variant) or {}).get("status") or ""),
            "error": str(((sample.get("variant_runs") or {}).get(candidate_variant) or {}).get("error") or ""),
        }
        for sample in manifest.get("samples") or []
        if isinstance(sample, dict)
        and str(((sample.get("variant_runs") or {}).get(candidate_variant) or {}).get("status") or "")
        in {"failed", "blocked", "timeout", "asr_model_not_ready"}
    ]
    window_alignment_ready = bool(sample_results) and all(bool(row.get("window_alignment_ready")) for row in sample_results)
    acceptance = _acceptance(
        baseline,
        candidate,
        sample_results,
        summary_blind_review=manifest.get("summary_blind_review"),
        candidate_variant=candidate_variant,
        window_alignment_ready=window_alignment_ready,
    )
    quality_blockers: list[dict[str, Any]] = []
    if reference_count != len(sample_results):
        quality_blockers.append(
            {
                "key": "human_reference_incomplete",
                "current": reference_count,
                "required": len(sample_results),
                "next_action": "complete_boundary_aligned_human_references",
            }
        )
    if not window_alignment_ready:
        quality_blockers.append(
            {
                "key": "boundary_alignment_incomplete",
                "current": False,
                "required": True,
                "next_action": "rebuild_or_complete_asr_vad_sentence_aligned_samples",
            }
        )
    if candidate_available_count != len(sample_results):
        quality_blockers.append(
            {
                "key": "candidate_variant_missing",
                "variant": candidate_variant,
                "current": candidate_available_count,
                "required": len(sample_results),
                "failures": candidate_run_failures,
                "next_action": "execute_quality_benchmark_candidate_variant",
            }
        )
    summary_review = acceptance.get("summary_blind_review") if isinstance(acceptance.get("summary_blind_review"), dict) else {}
    if not summary_review.get("ready"):
        quality_blockers.append(
            {
                "key": "summary_blind_review_incomplete",
                "current": int(summary_review.get("completed") or 0),
                "required": int(summary_review.get("total") or 0),
                "next_action": "complete_anonymous_summary_blind_review",
            }
        )

    if not sample_results:
        status = "no_samples"
    elif reference_count != len(sample_results):
        status = "needs_human_reference"
    elif not window_alignment_ready:
        status = "needs_boundary_aligned_rebuild"
    elif candidate_available_count != len(sample_results):
        status = "needs_candidate_variant"
    else:
        status = "completed"
    result: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "manifest_json": str(path),
        "output_dir": str(out),
        "status": status,
        "ok": status == "completed",
        "sample_count": len(sample_results),
        "reference_ready_count": reference_count,
        "reference_policy": {
            "canonical_source": "completed_current_window_reference_text",
            "current_window_completed_count": sum(
                str(row.get("reference_completeness", {}).get("reason") or "") == "completed_aligned_human_review"
                for row in sample_results
            ),
            "legacy_reference_present_count": sum(
                bool(str(sample.get("legacy_reference_text") or "").strip())
                for sample in manifest.get("samples") or []
                if isinstance(sample, dict)
            ),
            "legacy_reference_used_for_scoring": False,
            "legacy_reference_status": "superseded_history_only",
        },
        "window_strategy": manifest_window_strategy,
        "window_alignment_ready": window_alignment_ready,
        "samples": sample_results,
        "variants": aggregate,
        "candidate_variant": candidate_variant,
        "candidate_available_count": candidate_available_count,
        "candidate_run_failures": candidate_run_failures,
        "candidate_required_count": len(sample_results),
        "asr_disagreement_summary": disagreement_summary,
        "summary_blind_review": manifest.get("summary_blind_review") or {},
        "acceptance": acceptance,
        "quality_blockers": quality_blockers,
        "next_actions": list(dict.fromkeys(str(row.get("next_action") or "") for row in quality_blockers if row.get("next_action"))),
        "operator_boundary": {"reference_is_evaluation_only": True, "does_not_apply_corrections": True, "no_model_calls": True},
        "updated_at": now_iso(),
    }
    if write:
        out.mkdir(parents=True, exist_ok=True)
        write_json(path, manifest)
        (path.parent / "quality-benchmark-review.html").write_text(_render_review_html(manifest), encoding="utf-8")
        write_json(out / "quality-benchmark.json", result)
        (out / "quality-benchmark.md").write_text(_render_report(result), encoding="utf-8")
        (out / "quality-benchmark.html").write_text(_render_report_html(result), encoding="utf-8")
    return result


def report_quality_benchmark(run_json: str | Path, *, output_dir: str | Path | None = None, write: bool = True) -> dict[str, Any]:
    path = Path(run_json).expanduser().resolve()
    result = read_json(path)
    if not isinstance(result, dict):
        raise ValueError("quality benchmark run must be a JSON object")
    out = Path(output_dir).expanduser().resolve() if output_dir else path.parent
    if write:
        out.mkdir(parents=True, exist_ok=True)
        (out / "quality-benchmark.md").write_text(_render_report(result), encoding="utf-8")
        (out / "quality-benchmark.html").write_text(_render_report_html(result), encoding="utf-8")
    return {**result, "report_markdown": str(out / "quality-benchmark.md"), "report_html": str(out / "quality-benchmark.html")}


def transcript_quality_metrics(reference_text: str, candidate_text: str, *, source_text: str = "", protected_entities: list[str] | None = None) -> dict[str, Any]:
    """Public metric helper reused by transcript-quality-gate."""

    return _metrics(reference_text=reference_text, source_text=source_text, candidate_text=candidate_text, candidate_cues=[], sample={"protected_entities": protected_entities or []})


def _asr_disagreement(sensevoice_text: str, qwen_text: str) -> dict[str, Any]:
    sensevoice_numbers = _number_evidence_map(sensevoice_text)
    qwen_numbers = _number_evidence_map(qwen_text)
    equivalent_numeric_forms = bool(sensevoice_numbers) and sensevoice_numbers.keys() == qwen_numbers.keys()
    comparison_sensevoice = _strip_number_mentions(sensevoice_text) if equivalent_numeric_forms else sensevoice_text
    comparison_qwen = _strip_number_mentions(qwen_text) if equivalent_numeric_forms else qwen_text
    sensevoice = _normalise_content(comparison_sensevoice)
    qwen = _normalise_content(comparison_qwen)
    if not sensevoice or not qwen:
        return {
            "available": False,
            "edit_ratio": None,
            "review_priority": "not_available",
            "number_conflicts": [],
        }
    edit_ratio = _edit_distance(sensevoice, qwen) / max(1, max(len(sensevoice), len(qwen)))
    conflict_keys = set(sensevoice_numbers).symmetric_difference(qwen_numbers)
    number_conflicts = sorted(
        {
            mention
            for key in conflict_keys
            for mention in sensevoice_numbers.get(key, set()) | qwen_numbers.get(key, set())
        }
    )
    priority = "high" if edit_ratio >= 0.15 or number_conflicts else ("medium" if edit_ratio >= 0.08 else "low")
    return {
        "available": True,
        "edit_ratio": round(edit_ratio, 6),
        "review_priority": priority,
        "number_conflicts": number_conflicts,
        "sensevoice_chars": len(sensevoice),
        "qwen_chars": len(qwen),
    }


def _baseline_asr_text(
    variant_texts: dict[str, str], sample: dict[str, Any]
) -> str:
    """Select independent SenseVoice output or the source-bound ASR draft."""

    return str(
        variant_texts.get("sensevoice_full_punc")
        or variant_texts.get("sensevoice_raw")
        or sample.get("asr_draft_text")
        or ""
    )

def _metrics(
    *,
    reference_text: str,
    source_text: str,
    candidate_text: str,
    candidate_cues: list[Any],
    sample: dict[str, Any],
) -> dict[str, Any]:
    reference = _normalise_content(reference_text)
    source = _normalise_content(source_text)
    candidate = _normalise_content(candidate_text)
    edits = _edit_distance(reference, candidate) if reference and candidate else 0
    baseline_edits = _edit_distance(reference, source) if reference and source else 0
    cer = edits / max(1, len(reference)) if reference and candidate else None
    baseline_cer = baseline_edits / max(1, len(reference)) if reference and source else None
    punctuation = _punctuation_scores(reference_text, candidate_text)
    reference_number_evidence = _number_evidence_map(reference_text)
    candidate_number_evidence = _number_evidence_map(candidate_text)
    reference_number_keys = set(reference_number_evidence)
    candidate_number_keys = set(candidate_number_evidence)
    number_miss_keys = sorted(reference_number_keys - candidate_number_keys)
    number_extra_keys = sorted(candidate_number_keys - reference_number_keys)
    number_misses = [_numeric_evidence_label(key, reference_number_evidence) for key in number_miss_keys]
    number_extras = [_numeric_evidence_label(key, candidate_number_evidence) for key in number_extra_keys]

    protected = [str(value) for value in sample.get("protected_entities") or [] if str(value)]
    derived = [value for value in _entities(reference_text) if not _number_evidence_map(value)]
    textual_entities = list(
        dict.fromkeys(
            value
            for value in [*protected, *derived]
            if not _number_evidence_map(value) and _compact(value) in _compact(reference_text)
        )
    )
    textual_entity_hits = sum(1 for value in textual_entities if _compact(value) in _compact(candidate_text))
    numeric_entity_hits = len(reference_number_keys & candidate_number_keys)
    entity_total = len(textual_entities) + len(reference_number_keys)
    entity_hits = textual_entity_hits + numeric_entity_hits
    timestamp_errors = _timestamp_errors(candidate_cues, sample)
    return {
        "char_count": len(candidate),
        "cer": round(cer, 6) if cer is not None else None,
        "baseline_cer": round(baseline_cer, 6) if baseline_cer is not None else None,
        "relative_cer_reduction": round((baseline_cer - cer) / baseline_cer, 6) if baseline_cer and cer is not None else None,
        "overcorrection_rate": round(max(0, edits - baseline_edits) / max(1, len(reference)), 6) if reference and source and candidate else None,
        "punctuation_f1": punctuation["punctuation_f1"],
        "sentence_boundary_f1": punctuation["sentence_boundary_f1"],
        "entity_accuracy": round(entity_hits / max(1, entity_total), 6) if entity_total else None,
        "entity_total": entity_total,
        "number_error_count": len(number_miss_keys) + len(number_extra_keys),
        "number_misses": number_misses,
        "number_extras": number_extras,
        "timestamp_median_error_seconds": timestamp_errors.get("median"),
        "timestamp_p95_error_seconds": timestamp_errors.get("p95"),
        "timestamp_available": bool(timestamp_errors.get("available")),
        "timestamp_status": str(timestamp_errors.get("status") or ""),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "cer",
        "relative_cer_reduction",
        "overcorrection_rate",
        "punctuation_f1",
        "sentence_boundary_f1",
        "entity_accuracy",
        "timestamp_median_error_seconds",
        "timestamp_p95_error_seconds",
    )
    result: dict[str, Any] = {"sample_count": len(rows)}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        result[key] = round(statistics.mean(values), 6) if values else None
    result["number_error_count"] = sum(int(row.get("number_error_count") or 0) for row in rows)
    result["timestamp_available_count"] = sum(bool(row.get("timestamp_available")) for row in rows)
    return result


def _acceptance(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    samples: list[dict[str, Any]],
    *,
    summary_blind_review: Any = None,
    candidate_variant: str = "qwen3_asr_1_7b",
    window_alignment_ready: bool = False,
) -> dict[str, Any]:
    checks = [
        {
            "key": "boundary_aligned_samples",
            "passed": window_alignment_ready,
            "value": window_alignment_ready,
            "target": True,
        }
    ]
    baseline_cer = baseline.get("cer")
    candidate_cer = candidate.get("cer")
    relative = ((float(baseline_cer) - float(candidate_cer)) / float(baseline_cer)) if baseline_cer and candidate_cer is not None else None
    checks.append({"key": "relative_cer_reduction", "passed": relative is not None and relative >= 0.20, "value": round(relative, 6) if relative is not None else None, "target": 0.20})
    checks.append({"key": "overcorrection_rate", "passed": candidate.get("overcorrection_rate") is not None and float(candidate["overcorrection_rate"]) <= 0.01, "value": candidate.get("overcorrection_rate"), "target": 0.01})

    baseline_entity_error = 1.0 - float(baseline.get("entity_accuracy")) if baseline.get("entity_accuracy") is not None else None
    candidate_entity_error = 1.0 - float(candidate.get("entity_accuracy")) if candidate.get("entity_accuracy") is not None else None
    entity_reduction = _error_reduction(baseline_entity_error, candidate_entity_error)
    checks.append({"key": "proper_noun_error_reduction", "passed": entity_reduction is not None and entity_reduction >= 0.30, "value": entity_reduction, "target": 0.30})

    baseline_number_error = float(baseline.get("number_error_count")) if baseline.get("number_error_count") is not None else None
    candidate_number_error = float(candidate.get("number_error_count")) if candidate.get("number_error_count") is not None else None
    number_reduction = _error_reduction(baseline_number_error, candidate_number_error)
    checks.append({"key": "number_error_reduction", "passed": number_reduction is not None and number_reduction >= 0.30, "value": number_reduction, "target": 0.30})
    checks.append({"key": "entity_accuracy", "passed": candidate.get("entity_accuracy") is not None and float(candidate["entity_accuracy"]) >= 0.98, "value": candidate.get("entity_accuracy"), "target": 0.98})
    checks.append({"key": "timestamp_median", "passed": candidate.get("timestamp_median_error_seconds") is not None and float(candidate["timestamp_median_error_seconds"]) <= 0.5, "value": candidate.get("timestamp_median_error_seconds"), "target": 0.5})
    checks.append({"key": "timestamp_p95", "passed": candidate.get("timestamp_p95_error_seconds") is not None and float(candidate["timestamp_p95_error_seconds"]) <= 1.5, "value": candidate.get("timestamp_p95_error_seconds"), "target": 1.5})

    summary = _summary_blind_acceptance(summary_blind_review)
    checks.append({"key": "summary_blind_review_improvement", "passed": summary["passed"], "value": summary["mean_improvement"], "target": summary["target_mean_improvement"]})
    ready = bool(samples) and window_alignment_ready and all(row.get("reference_ready") for row in samples) and summary["ready"] and bool(candidate)
    passed = ready and all(bool(row["passed"]) for row in checks)
    return {
        "ready": ready,
        "passed": passed,
        "candidate_variant": candidate_variant,
        "model_switch_allowed": passed,
        "decision": "switch_allowed" if passed else "keep_current_default",
        "summary_blind_review": summary,
        "checks": checks,
    }


def _error_reduction(baseline_error: float | None, candidate_error: float | None) -> float | None:
    if baseline_error is None or candidate_error is None:
        return None
    if baseline_error <= 0:
        return 1.0 if candidate_error <= 0 else 0.0
    return round((baseline_error - candidate_error) / baseline_error, 6)


def _summary_blind_acceptance(value: Any) -> dict[str, Any]:
    review = value if isinstance(value, dict) else {}
    items = [row for row in review.get("items") or [] if isinstance(row, dict)]
    completed = [row for row in items if row.get("baseline_score") is not None and row.get("candidate_score") is not None and str(row.get("review_status") or "") == "completed"]
    improvements = [float(row["candidate_score"]) - float(row["baseline_score"]) for row in completed]
    mean_improvement = round(statistics.mean(improvements), 6) if improvements else None
    reference_items = [row for row in items if row.get("reference_summary_path")]
    readability_gaps = []
    readability_margins = []
    for row in reference_items:
        candidate_dimensions = row.get("candidate_dimension_scores") if isinstance(row.get("candidate_dimension_scores"), dict) else {}
        reference_dimensions = row.get("reference_dimension_scores") if isinstance(row.get("reference_dimension_scores"), dict) else {}
        if candidate_dimensions.get("readability") is not None and reference_dimensions.get("readability") is not None:
            candidate_readability = float(candidate_dimensions["readability"])
            reference_readability = float(reference_dimensions["readability"])
            readability_gaps.append(max(0.0, reference_readability - candidate_readability))
            readability_margins.append(candidate_readability - reference_readability)
    mean_reference_gap = round(statistics.mean(readability_gaps), 6) if readability_gaps else None
    mean_candidate_margin = round(statistics.mean(readability_margins), 6) if readability_margins else None
    target = float(review.get("target_mean_improvement") or 0.5)
    reference_gap_target = float(review.get("target_reference_readability_gap") or 0.3)
    ready = bool(items) and len(completed) == len(items) and (not reference_items or len(readability_gaps) == len(reference_items))
    improvement_passed = mean_improvement is not None and mean_improvement >= target
    reference_gap_passed = not reference_items or (mean_reference_gap is not None and mean_reference_gap <= reference_gap_target)
    return {
        "ready": ready,
        "passed": ready and improvement_passed and reference_gap_passed,
        "completed": len(completed),
        "total": len(items),
        "mean_improvement": mean_improvement,
        "target_mean_improvement": target,
        "reference_readability_completed": len(readability_gaps),
        "reference_readability_total": len(reference_items),
        "mean_reference_readability_gap": mean_reference_gap,
        "reference_readability_gap_semantics": "max(0, reference_readability - candidate_readability)",
        "mean_candidate_readability_margin": mean_candidate_margin,
        "target_reference_readability_gap": reference_gap_target,
    }


def _reference_text(sample: dict[str, Any]) -> str:
    inline = str(sample.get("reference_text") or "").strip()
    if inline:
        return inline
    path_value = str(sample.get("reference_transcript") or "").strip()
    return _window_text(path_value, sample) if path_value else ""


def _reference_completeness(
    reference_text: str,
    source_text: str,
    *,
    window_strategy: str = ALIGNED_WINDOW_STRATEGY,
    human_review_status: str = "",
    reviewed_window_matches: bool = False,
) -> dict[str, Any]:
    reference_chars = len(_normalise_content(reference_text))
    source_chars = len(_normalise_content(source_text))
    ratio = reference_chars / max(1, source_chars) if source_chars else 1.0
    legacy_window = window_strategy == LEGACY_WINDOW_STRATEGY
    completed_aligned_review = (
        str(human_review_status or "").strip().lower() == "completed"
        and reviewed_window_matches
        and not legacy_window
    )
    ready = (
        bool(reference_chars)
        if legacy_window
        else bool(reference_chars) and (completed_aligned_review or not source_chars or ratio >= 0.55)
    )
    if not reference_chars:
        reason = "missing_reference"
    elif legacy_window:
        reason = "legacy_fixed_window_reference"
    elif completed_aligned_review:
        reason = "completed_aligned_human_review"
    elif ready:
        reason = "complete"
    else:
        reason = "reference_probably_partial"
    return {
        "ready": ready,
        "reference_chars": reference_chars,
        "source_chars": source_chars,
        "reference_to_source_ratio": round(ratio, 6),
        "reason": reason,
        "boundary_warning": legacy_window,
        "human_review_status": str(human_review_status or ""),
        "reviewed_window_matches": bool(reviewed_window_matches),
    }


def _reviewed_window_matches(sample: dict[str, Any], *, tolerance_seconds: float = 0.05) -> bool:
    alignment = sample.get("boundary_alignment") if isinstance(sample.get("boundary_alignment"), dict) else {}
    if not bool(alignment.get("ready")):
        return False
    reference_start = sample.get("reference_start_seconds")
    reference_end = sample.get("reference_end_seconds")
    if reference_start is None or reference_end is None:
        return False
    try:
        return (
            abs(float(reference_start) - float(sample.get("start_seconds") or 0.0)) <= tolerance_seconds
            and abs(float(reference_end) - float(sample.get("end_seconds") or 0.0)) <= tolerance_seconds
        )
    except (TypeError, ValueError):
        return False

def _window_text(path_value: Any, sample: dict[str, Any]) -> str:
    return _window_text_and_cues(path_value, sample)[0]


def _window_text_and_cues(path_value: Any, sample: dict[str, Any]) -> tuple[str, list[Any]]:
    if not path_value:
        return "", []
    path = Path(str(path_value)).expanduser()
    if not path.exists():
        return "", []
    try:
        cues = parse_transcript(path)
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace"), []
    start = float(sample.get("start_seconds") or 0.0)
    end = float(sample.get("end_seconds") or 0.0)
    selected = [cue for cue in cues if max(start, float(cue.start)) <= min(end, float(cue.end))]
    return " ".join(str(cue.text or "") for cue in selected).strip(), selected


def _timestamp_errors(cues: list[Any], sample: dict[str, Any]) -> dict[str, Any]:
    if not cues:
        return {"available": False, "status": "no_timestamp_cues", "median": None, "p95": None}
    expected_start = sample.get("reference_start_seconds")
    expected_end = sample.get("reference_end_seconds")
    if expected_start is None or expected_end is None:
        return {"available": False, "status": "reference_window_missing", "median": None, "p95": None}
    cue_start = float(cues[0].start)
    cue_end = float(cues[-1].end)
    expected_duration = max(0.0, float(expected_end) - float(expected_start))
    cue_duration = max(0.0, cue_end - cue_start)
    # Text-only ASR output has no alignment evidence. Do not turn the missing
    # capability into a large, misleading timestamp error.
    if expected_duration > 1.0 and cue_duration <= 0.05:
        return {"available": False, "status": "asr_text_only_alignment_not_run", "median": None, "p95": None}
    errors = [abs(cue_start - float(expected_start)), abs(cue_end - float(expected_end))]
    return {"available": True, "status": "measured", "median": round(statistics.median(errors), 6), "p95": round(max(errors), 6)}


def _punctuation_scores(reference: str, candidate: str) -> dict[str, float]:
    ref_positions = _boundary_positions(reference, terminal_only=False)
    cand_positions = _boundary_positions(candidate, terminal_only=False)
    ref_sentences = _boundary_positions(reference, terminal_only=True)
    cand_sentences = _boundary_positions(candidate, terminal_only=True)
    return {"punctuation_f1": _position_f1(ref_positions, cand_positions), "sentence_boundary_f1": _position_f1(ref_sentences, cand_sentences)}


def _boundary_positions(text: str, *, terminal_only: bool) -> set[int]:
    positions: set[int] = set()
    content_index = 0
    allowed = "。！？!?" if terminal_only else PUNCTUATION
    for char in str(text or ""):
        if char in allowed:
            positions.add(content_index)
        elif char not in PUNCTUATION and not char.isspace():
            content_index += 1
    return positions


def _position_f1(reference: set[int], candidate: set[int], tolerance: int = 2) -> float:
    if not reference and not candidate:
        return 1.0
    matched_candidate: set[int] = set()
    matches = 0
    for ref in sorted(reference):
        nearest = next((value for value in sorted(candidate, key=lambda item: abs(item - ref)) if value not in matched_candidate and abs(value - ref) <= tolerance), None)
        if nearest is not None:
            matched_candidate.add(nearest)
            matches += 1
    precision = matches / max(1, len(candidate))
    recall = matches / max(1, len(reference))
    return round((2 * precision * recall / (precision + recall)) if precision + recall else 0.0, 6)


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[right_index] + 1, previous[right_index - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def _normalise_content(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").lower())


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").lower())


def _numbers(value: str) -> list[str]:
    return re.findall(r"(?<![a-zA-Z])\d+(?:\.\d+)?%?", str(value or ""))

def _numeric_evidence_label(key: str, evidence: dict[str, set[str]]) -> str:
    mentions = sorted(str(value) for value in evidence.get(key) or [] if str(value))
    return mentions[0] if mentions else key


def _entities(value: str) -> list[str]:
    latin = re.findall(r"\b[A-Za-z][A-Za-z0-9_.+-]{1,30}\b", str(value or ""))
    return list(dict.fromkeys(latin + _numbers(value)))[:40]


def _sample_category(text: str) -> str:
    if _numbers(text) or re.search(r"[零一二两三四五六七八九十百千万亿]+(?:元|块|万|分钟|小时|点|岁|个|次|%|％)", text):
        return "numbers_or_amounts"
    if re.search(r"\b[A-Za-z][A-Za-z0-9_.+-]{2,}\b", text) or re.search(r"小红书|微信|抖音|明亚|趣研学|TikTok|B站|哔哩哔哩", text, re.IGNORECASE):
        return "proper_noun_or_tool"
    if len(text) > 260:
        return "long_sentence_dense_speech"
    return "general_lecture"


def _stratified_windows(cues: list[Any], *, duration: float, count: int, sample_seconds: float) -> list[tuple[float, float, str]]:
    special: dict[str, list[tuple[float, float, str]]] = {
        "numbers_or_amounts": [],
        "proper_noun_or_tool": [],
        "long_sentence_dense_speech": [],
    }
    for cue in cues:
        category = _sample_category(str(cue.text or ""))
        if category not in special:
            continue
        center = (float(cue.start) + float(cue.end)) / 2
        window = (*_aligned_sample_window(cues, center=center, duration=duration, sample_seconds=sample_seconds), category)
        if _window_is_new(window, special[category], sample_seconds=sample_seconds):
            special[category].append(window)

    uniform: list[tuple[float, float, str]] = []
    for index in range(max(count * 3, count)):
        center = duration * ((index + 0.5) / max(count * 3, count))
        start, end = _aligned_sample_window(cues, center=center, duration=duration, sample_seconds=sample_seconds)
        uniform.append((start, end, "general_lecture"))

    selected: list[tuple[float, float, str]] = []
    quotas = {
        "numbers_or_amounts": min(2, count),
        "proper_noun_or_tool": min(2, max(0, count - 2)),
        "long_sentence_dense_speech": min(1, max(0, count - 4)),
    }
    for category, quota in quotas.items():
        rows = special[category]
        if not rows or quota <= 0:
            continue
        positions = [round(index * (len(rows) - 1) / max(1, quota - 1)) for index in range(quota)] if quota > 1 else [len(rows) // 2]
        for position in positions:
            row = rows[position]
            if _window_is_new(row, selected, sample_seconds=sample_seconds):
                selected.append(row)
    remaining = max(0, count - len(selected))
    for slot in range(remaining):
        target = duration * ((slot + 0.5) / max(1, remaining))
        available = [row for row in uniform if _window_is_new(row, selected, sample_seconds=sample_seconds)]
        if not available:
            break
        selected.append(min(available, key=lambda row: abs(((row[0] + row[1]) / 2) - target)))
    for category in ("numbers_or_amounts", "proper_noun_or_tool", "long_sentence_dense_speech"):
        for row in special[category]:
            if len(selected) >= count:
                break
            if _window_is_new(row, selected, sample_seconds=max(1.0, sample_seconds / 2)):
                selected.append(row)
    return sorted(selected[:count], key=lambda item: item[0])


def _sample_window(center: float, *, duration: float, sample_seconds: float) -> tuple[float, float]:
    length = max(10.0, float(sample_seconds or 60.0))
    start = max(0.0, min(float(duration), float(center) - length / 2))
    end = min(float(duration), start + length)
    if end - start < length and duration >= length:
        start = max(0.0, end - length)
    return round(start, 3), round(end, 3)


def _aligned_sample_window(
    cues: list[Any],
    *,
    center: float,
    duration: float,
    sample_seconds: float,
    silence_gap_seconds: float = 0.45,
) -> tuple[float, float]:
    """Expand a nominal sample to complete ASR/VAD utterance boundaries."""

    nominal_start, nominal_end = _sample_window(center, duration=duration, sample_seconds=sample_seconds)
    overlapping = [
        index
        for index, cue in enumerate(cues)
        if max(nominal_start, float(cue.start)) <= min(nominal_end, float(cue.end))
    ]
    if not overlapping:
        return nominal_start, nominal_end

    first = overlapping[0]
    last = overlapping[-1]
    aligned_start = max(0.0, float(cues[first].start))
    aligned_end = min(float(duration), float(cues[last].end))
    max_seconds = max(90.0, float(sample_seconds or 60.0) * 1.5)

    while first > 0:
        previous = cues[first - 1]
        current = cues[first]
        gap = max(0.0, float(current.start) - float(previous.end))
        if gap >= silence_gap_seconds or _ends_sentence(str(previous.text or "")):
            break
        proposed_start = max(0.0, float(previous.start))
        if aligned_end - proposed_start > max_seconds:
            break
        first -= 1
        aligned_start = proposed_start

    while last + 1 < len(cues):
        current = cues[last]
        following = cues[last + 1]
        gap = max(0.0, float(following.start) - float(current.end))
        if gap >= silence_gap_seconds or _ends_sentence(str(current.text or "")):
            break
        proposed_end = min(float(duration), float(following.end))
        if proposed_end - aligned_start > max_seconds:
            break
        last += 1
        aligned_end = proposed_end

    if aligned_end - aligned_start > max_seconds:
        return nominal_start, nominal_end
    return round(aligned_start, 3), round(aligned_end, 3)


def _ends_sentence(text: str) -> bool:
    return str(text or "").rstrip().endswith(tuple(SENTENCE_TERMINATORS))


def _window_boundary_metadata(cues: list[Any], *, start: float, end: float, duration: float) -> dict[str, Any]:
    tolerance = 0.02
    start_aligned = start <= tolerance or any(abs(float(cue.start) - start) <= tolerance for cue in cues)
    end_aligned = abs(float(duration) - end) <= tolerance or any(abs(float(cue.end) - end) <= tolerance for cue in cues)
    return {
        "ready": start_aligned and end_aligned,
        "start_aligned": start_aligned,
        "end_aligned": end_aligned,
        "status": "aligned" if start_aligned and end_aligned else "unavoidable_long_cue_or_missing_boundary",
    }


def _window_is_new(row: tuple[float, float, str], selected: list[tuple[float, float, str]], *, sample_seconds: float) -> bool:
    center = (row[0] + row[1]) / 2
    return all(abs(center - ((item[0] + item[1]) / 2)) >= max(5.0, sample_seconds * 0.75) for item in selected)


def _benchmark_bundle_label(bundle: Path) -> str:
    if bundle.name == "webui-bundle" and bundle.parent.name == "local-asr-vkp":
        return bundle.parent.parent.name
    return bundle.parent.name if bundle.name == "webui-bundle" else bundle.name


def _benchmark_media_path(bundle: Path, manifest: dict[str, Any], explicit: str | Path | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.exists() else None
    source_package = _optional_path(bundle, manifest.get("source_package"))
    if source_package:
        package = _mapping(source_package)
        for source in package.get("sources") or []:
            if isinstance(source, dict) and source.get("path"):
                path = Path(str(source["path"])).expanduser()
                if path.exists():
                    return path.resolve()
    return None


def _load_or_run_vad_segments(media: Path, output: Path) -> tuple[list[dict[str, float]], str]:
    if not _media_has_audio_stream(media):
        return [], ""
    if output.exists():
        payload = read_json(output)
        rows = payload.get("segments") if isinstance(payload, dict) else []
        if isinstance(rows, list) and rows:
            return [row for row in rows if isinstance(row, dict)], str(output.resolve())

    python_executable = _resolve_python_executable()
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1])
    existing = str(env.get("PYTHONPATH") or "").strip()
    env["PYTHONPATH"] = source_root + (os.pathsep + existing if existing else "")
    ffmpeg = resolve_media_tool("ffmpeg")
    if ffmpeg:
        env["PATH"] = str(Path(ffmpeg).resolve().parent) + os.pathsep + str(env.get("PATH") or "")
        env["FFMPEG"] = str(ffmpeg)
    device = default_local_asr_device(python_executable)
    completed = subprocess.run(
        [
            python_executable,
            "-m",
            "video_knowledge_pipeline.funasr_vad_runner",
            "--input",
            str(media),
            "--output",
            str(output),
            "--device",
            device,
            "--max-single-segment-time-ms",
            "30000",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        check=False,
        env=env,
    )
    if completed.returncode != 0 or not output.exists():
        return [], ""
    payload = read_json(output)
    rows = payload.get("segments") if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return [], ""
    return [row for row in rows if isinstance(row, dict)], str(output.resolve())


def _media_has_audio_stream(media: Path) -> bool:
    ffprobe = resolve_media_tool("ffprobe")
    if not ffprobe or not media.exists():
        return False
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(media),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and bool((completed.stdout or "").strip())

def _align_window_to_vad_segments(
    segments: list[dict[str, float]],
    *,
    start: float,
    end: float,
    duration: float,
    padding_seconds: float = 0.2,
) -> tuple[float, float, dict[str, Any]]:
    overlapping = [
        row
        for row in segments
        if max(float(start), float(row.get("start") or 0.0))
        <= min(float(end), float(row.get("end") or 0.0))
    ]
    if not overlapping:
        return start, end, {"ready": False, "status": "vad_no_overlapping_speech", "source": "funasr_fsmn_vad"}
    aligned_start = max(0.0, float(overlapping[0].get("start") or start) - padding_seconds)
    aligned_end = min(float(duration), float(overlapping[-1].get("end") or end) + padding_seconds)
    if aligned_end <= aligned_start:
        return start, end, {"ready": False, "status": "vad_invalid_window", "source": "funasr_fsmn_vad"}
    return (
        round(aligned_start, 3),
        round(aligned_end, 3),
        {
            "ready": True,
            "start_aligned": True,
            "end_aligned": True,
            "status": "funasr_vad_aligned",
            "source": "funasr_fsmn_vad",
            "vad_segment_count": len(overlapping),
        },
    )

def _align_window_to_audio_silence(
    media: Path,
    *,
    start: float,
    end: float,
    duration: float,
    search_seconds: float = 90.0,
) -> tuple[float, float, dict[str, Any]]:
    """Move review-window edges to nearby real audio silences."""

    search_start = max(0.0, float(start) - search_seconds)
    search_end = min(float(duration), float(end) + search_seconds)
    probe = probe_audio_silence(
        media,
        start=search_start,
        end=search_end,
        noise_db=-45.0,
        minimum_silence_seconds=0.5,
        timeout_seconds=90,
    )
    if not probe.get("ok"):
        status = str(probe.get("status") or "audio_silence_detection_failed")
        if status in {"media_not_found", "ffmpeg_not_available"}:
            status = "audio_silence_unavailable"
        else:
            status = "audio_silence_detection_failed"
        return start, end, {"ready": False, "status": status}
    silence_starts = [float(value) for value in probe.get("silence_starts") or []]
    silence_ends = [float(value) for value in probe.get("silence_ends") or []]
    before = [value for value in silence_ends if value <= float(start)]
    after = [value for value in silence_starts if value >= float(end)]
    aligned_start = max(0.0, max(before) + 0.05) if before else float(start)
    aligned_end = min(float(duration), min(after) - 0.05) if after else float(end)
    if aligned_end <= aligned_start:
        return start, end, {"ready": False, "status": "audio_silence_invalid"}
    start_found = bool(before) or aligned_start <= 0.02
    end_found = bool(after) or abs(float(duration) - aligned_end) <= 0.02
    return (
        round(aligned_start, 3),
        round(aligned_end, 3),
        {
            "ready": start_found and end_found,
            "start_aligned": start_found,
            "end_aligned": end_found,
            "detected_silence_count": len(probe.get("silence_intervals") or []),
            "status": "audio_silence_aligned"
            if start_found and end_found
            else "partial_audio_silence_alignment",
            "source": "ffmpeg_silencedetect",
        },
    )

def _write_audio_clip(media: Path, output: Path, *, start: float, end: float) -> str:
    ffmpeg = resolve_media_tool("ffmpeg")
    if not ffmpeg:
        return "ffmpeg_not_available"
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start), "-t", str(max(0.1, end - start)), "-i", str(media), "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode == 0 and output.exists() and output.stat().st_size > 44:
        return ""
    output.unlink(missing_ok=True)
    return (completed.stderr or "audio_clip_generation_failed")[-500:]


def _safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-\u4e00-\u9fff]+", "-", str(value or "sample")).strip("-") or "sample"


def _summary_blind_review_items(bundle_dirs: list[str | Path]) -> list[dict[str, Any]]:
    items = []
    for index, bundle_value in enumerate(bundle_dirs):
        bundle = Path(bundle_value).expanduser().resolve()
        manifest = _mapping(bundle / "manifest.json")
        current = _optional_path(bundle, manifest.get("knowledge_note_smart_summary_markdown") or "exports/smart-summary.md")
        baseline = current
        candidate: Path | None = None
        reference: Path | None = None
        source_root: Path | None = None
        if bundle.parent.name.lower() == "local-asr-vkp":
            source_root = bundle.parent.parent
            baseline = _optional_path(source_root, "webui-bundle/exports/smart-summary.md") or baseline
            candidate = current
            reference = _optional_path(source_root, "getbrain-smart-summary.md")
        items.append({
            "item_id": f"summary-{index + 1:02d}",
            "bundle_dir": str(bundle),
            "baseline_summary_path": str(baseline or ""),
            "candidate_summary_path": str(candidate or ""),
            "reference_summary_path": str(reference or ""),
            "baseline_score": None,
            "candidate_score": None,
            "reference_score": None,
            "review_status": "todo",
            "blind_labels_required": True,
            "external_reference_is_evaluation_only": bool(reference),
        })
    return items


def _default_transcript(bundle: Path, manifest: dict[str, Any]) -> Path | None:
    for key in ("corrected_transcript_json", "postprocessed_transcript_json", "normalized_transcript_json", "transcript_json"):
        path = _optional_path(bundle, manifest.get(key))
        if path:
            return path
    for name in ("corrected-transcript.json", "postprocessed-transcript.json", "normalized-transcript.json"):
        path = bundle / name
        if path.exists():
            return path.resolve()
    return None


def _benchmark_variant_paths(bundle: Path, manifest: dict[str, Any], *, fallback: Path | None = None) -> dict[str, Path | None]:
    """Resolve independent ASR stages instead of labelling the final corrected file as the baseline."""

    raw = _optional_path(bundle, manifest.get("normalized_transcript_json")) or _optional_path(bundle, "normalized-transcript.json")
    full_punc = _optional_path(bundle, manifest.get("postprocessed_transcript_json")) or _optional_path(bundle, "postprocessed-transcript.json")
    corrected = _optional_path(bundle, manifest.get("corrected_transcript_json")) or _optional_path(bundle, "corrected-transcript.json")
    return {
        "sensevoice_raw": raw,
        "sensevoice_full_punc": full_punc or raw or fallback,
        "corrected_transcript": corrected or fallback,
    }

def _optional_path(root: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve() if path.exists() else None


def _legacy_reference_rows(manifest_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not manifest_path:
        return {}
    path = Path(manifest_path).expanduser().resolve()
    manifest = read_json(path) if path.exists() else {}
    if not isinstance(manifest, dict):
        return {}
    return {
        str(row.get("sample_id")): row
        for row in manifest.get("samples") or []
        if isinstance(row, dict) and row.get("sample_id")
    }


def _mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def _render_todo(result: dict[str, Any]) -> str:
    lines = ["# Quality Benchmark Human Reference TODO", "", f"- Samples: `{result.get('sample_count')}`", "- 得到大脑或其他转写只能作为比较材料；reference 必须经过人工听视频确认。", "", "| Sample | Category | Time | Reference |", "| --- | --- | --- | --- |"]
    for row in result.get("samples") or []:
        lines.append(f"| `{row.get('sample_id')}` | `{row.get('category')}` | `{format_timestamp(float(row.get('start_seconds') or 0))} - {format_timestamp(float(row.get('end_seconds') or 0))}` | `{row.get('human_review_status')}` |")
    return "\n".join(lines).rstrip() + "\n"


def _render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Quality Benchmark",
        "",
        f"- Status: {result.get('status')}",
        f"- Human references: {result.get('reference_ready_count')}/{result.get('sample_count')}",
        f"- Canonical references: {(result.get('reference_policy') or {}).get('current_window_completed_count', 0)} current-window completed reviews",
        f"- Legacy references used for scoring: {(result.get('reference_policy') or {}).get('legacy_reference_used_for_scoring', False)}",
        f"- Acceptance: {(result.get('acceptance') or {}).get('passed', False)}",
        f"- Window strategy: {result.get('window_strategy')}",
        f"- Boundary-aligned samples: {result.get('window_alignment_ready')}",
        f"- Candidate coverage: {result.get('candidate_available_count')}/{result.get('candidate_required_count')} ({result.get('candidate_variant')})",
    ]
    if result.get("status") == "needs_boundary_aligned_rebuild":
        lines.append(
            "- Warning: legacy fixed windows may cut continuous speech. Their annotations remain valid for the clips, but CER cannot authorize a model switch."
        )
    if result.get("quality_blockers"):
        lines.extend(["", "## Quality Blockers", ""])
        for blocker in result.get("quality_blockers") or []:
            lines.append(
                f"- {blocker.get('key')} current={blocker.get('current')} required={blocker.get('required')} "
                f"next={blocker.get('next_action')}"
            )
    # Intent: keep useful challenger evidence visible even before a human CER
    # reference exists. Decision: render model-to-model disagreement as review
    # triage, never as a winner score. Reason: pure Cantonese detail errors are
    # concentrated in a few windows and numbers; a blank Variants table hid
    # those actionable windows. Evidence: the 2026-08-11 interview pack had
    # 12/12 candidate outputs but 0/12 human references. Effective scope: local
    # Markdown report only; acceptance and transcript promotion remain blocked.
    disagreement = result.get("asr_disagreement_summary") or {}
    if int(disagreement.get("compared_count") or 0) > 0:
        lines.extend(
            [
                "",
                "## ASR Disagreement Triage",
                "",
                f"- Compared samples: {disagreement.get('compared_count')}",
                f"- Mean normalized edit ratio: {_fmt(disagreement.get('mean_edit_ratio'))}",
                f"- High-disagreement samples: {disagreement.get('high_disagreement_count')}",
                f"- Samples with number conflicts: {disagreement.get('number_conflict_count')}",
                "- This is review priority, not accuracy or a model-switch decision.",
                "",
                "| Sample | Time | Priority | Edit ratio | Number conflicts |",
                "| --- | --- | --- | ---: | --- |",
            ]
        )
        for row in result.get("samples") or []:
            item = row.get("asr_disagreement") or {}
            if not item.get("available"):
                continue
            window = row.get("window") or {}
            conflicts = "、".join(str(value) for value in item.get("number_conflicts") or []) or "—"
            lines.append(
                f"| {row.get('sample_id')} | {format_timestamp(float(window.get('start') or 0))} - "
                f"{format_timestamp(float(window.get('end') or 0))} | {item.get('review_priority')} | "
                f"{_fmt(item.get('edit_ratio'))} | {conflicts} |"
            )
    lines.extend(
        [
            "",
            "## Variants",
            "",
            "| Variant | Samples | CER | CER reduction | Punctuation F1 | Entity accuracy | Overcorrection | Timestamp samples |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, row in (result.get("variants") or {}).items():
        lines.append(
            f"| {key} | {row.get('sample_count')} | {_fmt(row.get('cer'))} | "
            f"{_fmt(row.get('relative_cer_reduction'))} | {_fmt(row.get('punctuation_f1'))} | "
            f"{_fmt(row.get('entity_accuracy'))} | {_fmt(row.get('overcorrection_rate'))} | {row.get('timestamp_available_count', 0)} |"
        )
    lines.extend(["", "## Acceptance", ""])
    for row in (result.get("acceptance") or {}).get("checks") or []:
        lines.append(
            f"- {'PASS' if row.get('passed') else 'FAIL'} {row.get('key')} "
            f"value={row.get('value')} target={row.get('target')}"
        )
    return "\n".join(lines).rstrip() + "\n"

def _merged_review_seed(legacy_reference: str, asr_draft: str) -> dict[str, Any]:
    legacy = str(legacy_reference or "").strip()
    draft = str(asr_draft or "").strip()
    if not legacy:
        return {"text": draft, "source": "current_clip_asr", "merge_ready": bool(draft), "match_coverage": 0.0}
    if not draft:
        return {"text": legacy, "source": "legacy_reference_only", "merge_ready": False, "match_coverage": 0.0}
    legacy_compact, _legacy_positions = _alignment_text_with_positions(legacy)
    draft_compact, draft_positions = _alignment_text_with_positions(draft)
    if not legacy_compact or not draft_compact:
        return {"text": draft, "source": "current_clip_asr_with_legacy_unmerged", "merge_ready": False, "match_coverage": 0.0}
    matcher = difflib.SequenceMatcher(a=legacy_compact, b=draft_compact, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size >= 2]
    matched_chars = sum(block.size for block in blocks)
    coverage = matched_chars / max(1, min(len(legacy_compact), len(draft_compact)))
    if not blocks or matched_chars < 8 or coverage < 0.25:
        return {
            "text": draft,
            "source": "current_clip_asr_with_legacy_unmerged",
            "merge_ready": False,
            "match_coverage": round(coverage, 6),
            "matched_chars": matched_chars,
        }
    first = blocks[0]
    last = blocks[-1]
    prefix_end = draft_positions[first.b]
    suffix_start = draft_positions[last.b + last.size - 1] + 1
    prefix = draft[:prefix_end].strip()
    suffix = draft[suffix_start:].strip()
    parts = [part for part in (prefix, legacy, suffix) if part]
    return {
        "text": "\n\n".join(parts),
        "source": "legacy_reference_plus_boundary_asr",
        "merge_ready": True,
        "match_coverage": round(coverage, 6),
        "matched_chars": matched_chars,
        "prefix_chars": len(prefix),
        "suffix_chars": len(suffix),
    }


def _alignment_text_with_positions(value: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(str(value or "")):
        if re.match(r"[0-9A-Za-z\u4e00-\u9fff]", char):
            chars.append(char.lower())
            positions.append(index)
    return "".join(chars), positions


def _render_review_html(result: dict[str, Any]) -> str:
    rows = []
    for row in result.get("samples") or []:
        clip = str(row.get("audio_clip_path") or "")
        media = str(row.get("media_path") or "")
        player = (
            f'<audio controls preload="metadata" src="{html.escape(Path(clip).as_uri())}"></audio>'
            if clip and Path(clip).exists()
            else html.escape(str(row.get("audio_clip_error") or "音频片段未生成"))
        )
        video_button = (
            f'<button class="video-jump" data-media="{html.escape(Path(media).as_uri(), quote=True)}" '
            f'data-start="{float(row.get("start_seconds") or 0):.3f}">跳转原视频时间点</button>'
            if media and Path(media).exists()
            else ""
        )
        existing_reference = str(row.get("reference_text") or "")
        legacy_reference = str(row.get("legacy_reference_text") or "")
        asr_draft = str(row.get("asr_draft_text") or "")
        qwen_draft = str(row.get("qwen_draft_text") or "")
        disagreement = row.get("asr_disagreement") if isinstance(row.get("asr_disagreement"), dict) else {}
        review_seed = _merged_review_seed(legacy_reference, asr_draft)
        stored_seed = str(row.get("review_seed_text") or "")
        seed_text = existing_reference or stored_seed or str(review_seed.get("text") or "") or legacy_reference or asr_draft
        seed_source = str((row.get("review_seed_metadata") or {}).get("source") or review_seed.get("source") or "")
        if existing_reference:
            draft_note = "<div><strong>已载入当前人工校对稿，请复核后导出。</strong></div>"
        elif seed_source == "legacy_reference_plus_boundary_asr":
            draft_note = (
                "<div><strong>已合并旧人工纠正文段与当前完整片段 ASR。</strong>"
                "新增前后文仍是 ASR 草稿，请对照音频修正后再导出。</div>"
            )
        elif seed_source == "current_clip_asr_with_legacy_unmerged":
            draft_note = (
                "<div><strong>旧人工稿与当前 ASR 对齐不足，已用当前完整片段 ASR 预填。</strong>"
                "旧人工稿保留在下方供对照，请以音频为准修改。</div>"
            )
        elif legacy_reference:
            draft_note = (
                "<div><strong>旧人工稿暂未自动合并。</strong>"
                "请结合当前音频与片段级 ASR 完成校对。</div>"
            )
        else:
            draft_note = "<div><strong>已用当前音频片段的 ASR 结果预填，请直接修改错字、标点和遗漏。</strong></div>"
        asr_panel = (
            f"<details><summary>查看当前片段级 ASR 草稿</summary><pre>{html.escape(asr_draft)}</pre></details>"
            if asr_draft
            else ""
        )
        qwen_panel = (
            f"<details><summary>查看 Qwen3-ASR 独立识别</summary><pre>{html.escape(qwen_draft)}</pre></details>"
            if qwen_draft
            else ""
        )
        legacy_panel = (
            f"<details><summary>查看旧窗口人工稿</summary><pre>{html.escape(legacy_reference)}</pre></details>"
            if legacy_reference
            else ""
        )
        disagreement_panel = ""
        if disagreement.get("available"):
            priority = html.escape(str(disagreement.get("review_priority") or ""))
            ratio = float(disagreement.get("edit_ratio") or 0.0)
            number_conflicts = "、".join(str(value) for value in disagreement.get("number_conflicts") or []) or "无"
            disagreement_panel = (
                f"<div><strong>双 ASR 分歧：</strong>优先级 {priority}；"
                f"字符差异率 {ratio:.1%}；数字冲突 {html.escape(number_conflicts)}</div>"
            )
        rows.append(
            f"<tr><td>{html.escape(str(row.get('sample_id')))}</td>"
            f"<td>{html.escape(str(row.get('category')))}</td>"
            f"<td>{html.escape(format_timestamp(float(row.get('start_seconds') or 0)))} - "
            f"{html.escape(format_timestamp(float(row.get('end_seconds') or 0)))}</td>"
            f"<td>{player}<div>{video_button}</div></td>"
            f"<td>{draft_note}{disagreement_panel}<textarea data-sample=\"{html.escape(str(row.get('sample_id')))}\" "
            f"placeholder=\"校对当前片段，修改错字、标点和遗漏\">{html.escape(seed_text)}</textarea>"
            f"{asr_panel}{qwen_panel}{legacy_panel}</td></tr>"
        )
    manifest = json.dumps(result, ensure_ascii=False).replace("</", "<\\/")
    body = (
        "<p>请以音频片段为准校对。页面不会把 ASR 草稿自动视为人工真值。"
        "播放音频片段时会暂停原视频，播放原视频时会暂停所有音频片段。</p>"
        "<video id=\"sourceVideo\" controls preload=\"metadata\" "
        "style=\"position:sticky;top:0;width:min(100%,960px);max-height:52vh;background:#000\"></video>"
        "<div><button id=\"download\">下载已审核基准清单</button></div>"
        "<table><thead><tr><th>样本</th><th>类别</th><th>时间</th><th>音视频证据</th><th>人工校对</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    body += (
        f'<script id="manifest" type="application/json">{manifest}</script>'
        '<script>'
        'const video=document.getElementById("sourceVideo");'
        'const audios=[...document.querySelectorAll("audio")];'
        'const pauseAudios=(except=null)=>audios.forEach(audio=>{if(audio!==except)audio.pause();});'
        'audios.forEach(audio=>audio.addEventListener("play",()=>{video.pause();pauseAudios(audio);}));'
        'video.addEventListener("play",()=>pauseAudios());'
        'document.querySelectorAll(".video-jump").forEach(button=>button.addEventListener("click",()=>{'
        'pauseAudios();const start=Number(button.dataset.start||0);'
        'if(video.src!==button.dataset.media){video.src=button.dataset.media;'
        'video.addEventListener("loadedmetadata",()=>{video.currentTime=start;video.play();},{once:true});}'
        'else{video.currentTime=start;video.play();}}));'
        'document.getElementById("download").addEventListener("click",()=>{'
        'const data=JSON.parse(document.getElementById("manifest").textContent);'
        'document.querySelectorAll("textarea[data-sample]").forEach(el=>{'
        'const row=data.samples.find(item=>item.sample_id===el.dataset.sample);'
        'if(row){row.reference_text=el.value.trim();row.human_review_status=row.reference_text?"completed":"todo";}});'
        'const blob=new Blob([JSON.stringify(data,null,2)],{type:"application/json"});'
        'const link=document.createElement("a");link.href=URL.createObjectURL(blob);'
        'link.download="quality-benchmark-manifest.completed.json";link.click();'
        'URL.revokeObjectURL(link.href);});'
        '</script>'
    )
    return _html_page("逐字稿质量基准人工校对", body)

def _render_report_html(result: dict[str, Any]) -> str:
    return _html_page("Quality Benchmark", f"<pre>{html.escape(_render_report(result))}</pre>")


def _html_page(title: str, body: str) -> str:
    return f"<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{html.escape(title)}</title><style>body{{font:15px/1.55 system-ui;margin:24px;color:#1f2937}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d1d5db;padding:8px;vertical-align:top}}textarea{{width:100%;min-height:120px}}pre{{white-space:pre-wrap}} </style></head><body><h1>{html.escape(title)}</h1>{body}</body></html>"


def _fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}"
