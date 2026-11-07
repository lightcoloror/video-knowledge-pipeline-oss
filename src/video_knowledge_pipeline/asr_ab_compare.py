from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import read_json, write_json
from .transcript import parse_transcript, transcript_excerpt
from .transcript_speakers import cue_speaker

SCHEMA = "video_knowledge_pipeline.asr_ab_comparison.v1"


def compare_asr_ab_sample(
    run_json: str | Path,
    *,
    reference_transcript: str | Path | None = None,
    start_seconds: float = 0.0,
    end_seconds: float = 0.0,
    write: bool = True,
) -> dict[str, Any]:
    run_path = Path(run_json).expanduser().resolve()
    run = read_json(run_path)
    if not isinstance(run, dict):
        raise ValueError("ASR A/B run JSON must be an object")
    variants = [row for row in (run.get("variants") or []) if isinstance(row, dict)]
    reference_path = Path(reference_transcript).expanduser().resolve() if reference_transcript else None
    reference_text = _excerpt_from_path(reference_path, start_seconds=start_seconds, end_seconds=end_seconds)
    speaker_requirement = _reference_speaker_requirement(
        reference_path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    rows = [_score_variant(row, reference_text=reference_text, start_seconds=start_seconds, end_seconds=end_seconds) for row in variants]
    by_key = {row["key"]: row for row in rows}
    primary = by_key.get("sensevoice_full_punc") or by_key.get("sensevoice_basic") or (rows[0] if rows else {})
    gates = _gates(rows, speaker_requirement=speaker_requirement)
    decision = _decision(rows, speaker_requirement=speaker_requirement)
    result = {
        "schema": SCHEMA,
        "run_json": str(run_path),
        "workspace_dir": str(run.get("workspace_dir") or ""),
        "sample_media_path": str(run.get("sample_media_path") or ""),
        "reference_transcript": str(reference_path) if reference_path else "",
        "reference_role": "evaluation_only_not_correction_evidence" if reference_text else "missing",
        "reference_window": {
            "start_seconds": float(start_seconds or 0.0),
            "end_seconds": float(end_seconds or 0.0),
            "enabled": bool(end_seconds and end_seconds > start_seconds),
        },
        "speaker_requirement": speaker_requirement,
        "best_reference_variant": _best_reference_variant(rows),
        "status": decision["status"],
        "primary_recommendation": primary.get("key", ""),
        "production_recommendation": decision["production_recommendation"],
        "speaker_evaluation_candidate": decision["speaker_evaluation_candidate"],
        "speaker_evaluation_candidates": decision["speaker_evaluation_candidates"],
        "speaker_candidate_selection": decision["speaker_candidate_selection"],
        "second_asr_recommendation": decision["second_asr_recommendation"],
        "cloud_asr_recommendation": decision["cloud_asr_recommendation"],
        "variants": rows,
        "gates": gates,
        "operator_boundary": {
            "does_not_promote_any_transcript": True,
            "comparison_only": True,
            "comparison_command_does_not_trigger_cloud_upload": True,
            "reference_transcript_is_evaluation_only": bool(reference_text),
            "does_not_import_reference_as_evidence": True,
            "source_run_included_cloud_upload_attempt": _source_run_included_cloud_upload(rows),
            "second_asr_requires_successful_sample_before_default": True,
        },
        "next_actions": decision["next_actions"],
        "updated_at": now_iso(),
    }
    if write:
        output_dir = run_path.parent
        json_path = output_dir / "asr-ab-comparison.json"
        md_path = output_dir / "asr-ab-comparison.md"
        write_json(json_path, result)
        md_path.write_text(_render_markdown(result), encoding="utf-8")
        result["json_path"] = str(json_path)
        result["markdown_path"] = str(md_path)
    return result


def _score_variant(row: dict[str, Any], *, reference_text: str = "", start_seconds: float = 0.0, end_seconds: float = 0.0) -> dict[str, Any]:
    key = str(row.get("key") or "")
    status = str(row.get("status") or "")
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    segment_count = int(metrics.get("segment_count") or 0)
    char_count = int(metrics.get("char_count") or 0)
    punctuation_count = int(metrics.get("punctuation_count") or 0)
    duration_seconds = float(metrics.get("duration_seconds") or 0.0)
    readability_score = _readability_score(segment_count, char_count, punctuation_count, duration_seconds)
    availability = "ready" if status == "ok" else ("not_ready" if status in {"asr_module_not_ready", "asr_model_not_ready"} else ("not_run" if status in {"preview", "planned_upload_disabled", ""} else "failed"))
    transcript_text = _excerpt_from_path(_first_existing_path(row.get("normalized_json"), row.get("normalized_srt"), row.get("raw_output_json")), start_seconds=start_seconds, end_seconds=end_seconds)
    reference_similarity = _similarity(transcript_text, reference_text) if reference_text else 0.0
    return {
        "key": key,
        "status": status,
        "availability": availability,
        "normalized_json": row.get("normalized_json", ""),
        "normalized_srt": row.get("normalized_srt", ""),
        "raw_output_json": row.get("raw_output_json", ""),
        "operator_boundary": (
            dict(row.get("operator_boundary"))
            if isinstance(row.get("operator_boundary"), dict)
            else {}
        ),
        "metrics": metrics,
        "readability_score": readability_score,
        "reference_similarity": reference_similarity,
        "reference_char_count": len(reference_text),
        "sample_text_char_count": len(transcript_text),
        "strengths": _strengths(key, status, segment_count, punctuation_count),
        "risks": _risks(row, segment_count, char_count, punctuation_count),
        "blockers": _blockers(row),
    }


def _readability_score(segment_count: int, char_count: int, punctuation_count: int, duration_seconds: float) -> float:
    if char_count <= 0:
        return 0.0
    punc_density = min(punctuation_count / max(char_count, 1) * 18.0, 4.0)
    segment_density = 0.0
    if duration_seconds > 0:
        seconds_per_segment = duration_seconds / max(segment_count, 1)
        segment_density = 3.0 if 8 <= seconds_per_segment <= 45 else (1.5 if seconds_per_segment <= 90 else 0.5)
    length_score = 2.0 if char_count >= 300 else 0.5
    return round(min(10.0, punc_density + segment_density + length_score), 2)


def _strengths(key: str, status: str, segment_count: int, punctuation_count: int) -> list[str]:
    rows: list[str] = []
    if status == "ok":
        rows.append("sample_completed")
    if key == "sensevoice_full_punc" and punctuation_count > 0:
        rows.append("punctuation_and_itn_available")
    if key.startswith("sensevoice_full_punc_campp") and status == "ok":
        rows.append("anonymous_speaker_labels_candidate")
    if key == "sensevoice_full_punc_campp_oracle_2" and status == "ok":
        rows.append("known_speaker_count_diagnostic_upper_bound")
    if key == "moss_transcribe_diarize" and status == "ok":
        rows.extend(
            [
                "model_native_speaker_attributed_transcript_candidate",
                "source_boundaries_preserved_without_postprocess",
            ]
        )
    if key == "sensevoice_basic":
        rows.append("local_primary_baseline")
    if key == "dolphin":
        rows.append("second_local_asr_candidate")
    if key == "whisperx_alignment":
        rows.append("timestamp_speaker_alignment_evidence")
    if key == "openai_cloud_asr":
        rows.append("cloud_quality_reference_candidate")
    if segment_count > 1:
        rows.append("multi_segment_output")
    return rows


def _risks(row: dict[str, Any], segment_count: int, char_count: int, punctuation_count: int) -> list[str]:
    status = str(row.get("status") or "")
    key = str(row.get("key") or "")
    rows: list[str] = []
    if status != "ok":
        rows.append(status or "not_run")
    if status == "ok" and punctuation_count == 0:
        rows.append("no_punctuation")
    if status == "ok" and segment_count <= 1:
        rows.append("single_or_missing_segments")
    if status == "ok" and char_count < 300:
        rows.append("low_text_volume")
    failure_text = " ".join(str(row.get(name) or "") for name in ("reason", "error", "stdout_tail", "stderr_tail"))
    if key == "dolphin" and status != "ok":
        audio_extract = row.get("audio_extract") if isinstance(row.get("audio_extract"), dict) else {}
        if "torchcodec" in failure_text.lower():
            rows.append("windows_torchcodec_runtime_not_ready")
            if audio_extract.get("status") == "ok":
                rows.append("audio_extract_ok_but_dolphin_runtime_failed")
        rows.append("cannot_decide_second_asr_until_dolphin_sample_succeeds")
    if key == "moss_transcribe_diarize" and status != "ok":
        rows.extend(
            [
                "moss_local_runtime_or_model_not_ready",
                "cannot_compare_speaker_quality_until_moss_sample_succeeds",
            ]
        )
    if key == "whisperx_alignment" and status == "ok" and not row.get("word_level_alignment"):
        rows.append("alignment_without_word_timestamps")
    if key == "whisperx_alignment" and status != "ok":
        rows.append("alignment_evidence_missing")
    if key == "openai_cloud_asr" and status != "ok":
        rows.append("cloud_reference_missing")
    return rows


def _blockers(row: dict[str, Any]) -> list[dict[str, Any]]:
    key = str(row.get("key") or "")
    status = str(row.get("status") or "")
    if status in {"", "ok", "preview", "planned_upload_disabled"}:
        return []
    rows: list[dict[str, Any]] = []
    failure_text = " ".join(str(row.get(name) or "") for name in ("reason", "error", "stdout_tail", "stderr_tail"))
    if key == "dolphin":
        audio_extract = row.get("audio_extract") if isinstance(row.get("audio_extract"), dict) else {}
        module_ready = row.get("module_ready") if isinstance(row.get("module_ready"), dict) else {}
        if "torchcodec" in failure_text.lower():
            rows.append(
                {
                    "code": "dolphin_torchcodec_runtime_not_ready",
                    "severity": "blocking",
                    "evidence": {
                        "audio_extract_status": audio_extract.get("status", ""),
                        "audio_path": audio_extract.get("audio_path", ""),
                        "module_ready": module_ready.get("ready", ""),
                    },
                    "fix_hint": "Fix the Dolphin/TorchCodec runtime in the ASR Python environment; WAV extraction already succeeded, so changing the input container is not enough.",
                }
            )
        elif module_ready and not module_ready.get("ready"):
            rows.append(
                {
                    "code": "dolphin_python_module_missing",
                    "severity": "blocking",
                    "evidence": {"python": module_ready.get("python", ""), "module": module_ready.get("module", "")},
                    "fix_hint": "Install or expose the Dolphin Python module in the ASR runtime before rerunning the same sample.",
                }
            )
    if key == "moss_transcribe_diarize":
        availability = (
            row.get("availability")
            if isinstance(row.get("availability"), dict)
            else {}
        )
        runtime_probe = (
            availability.get("runtime_probe")
            if isinstance(availability.get("runtime_probe"), dict)
            else {}
        )
        model_ready = (
            row.get("model_ready")
            if isinstance(row.get("model_ready"), dict)
            else {}
        )
        rows.append(
            {
                "code": "moss_runtime_or_model_not_ready",
                "severity": "blocking",
                "evidence": {
                    "command_path": availability.get("command_path", ""),
                    "runtime_ready": row.get("runtime_ready", False),
                    "runtime_blocker": runtime_probe.get("blocker", ""),
                    "model_status": model_ready.get("status", ""),
                },
                "fix_hint": "Prepare the pinned local MOSS runtime and model explicitly, then rerun this exact sample. Do not download silently or switch to another ASR provider.",
            }
        )
    if key == "whisperx_alignment":
        alignment_run = _read_optional_json(row.get("alignment_run_json"))
        plan_path = alignment_run.get("plan_path") if isinstance(alignment_run, dict) else ""
        plan = _read_optional_json(plan_path)
        availability = plan.get("availability") if isinstance(plan.get("availability"), dict) else {}
        if availability:
            rows.append(
                {
                    "code": "whisperx_command_and_module_unavailable",
                    "severity": "blocking",
                    "evidence": {
                        "command_path": availability.get("command_path", ""),
                        "module": availability.get("module", ""),
                        "module_available": availability.get("module_available", ""),
                        "python_executable": availability.get("python_executable", ""),
                    },
                    "fix_hint": "Install WhisperX into the ASR environment or place the whisperx command on PATH; use it only for alignment evidence after SenseVoice transcript exists.",
                }
            )
        else:
            rows.append(
                {
                    "code": "whisperx_alignment_missing",
                    "severity": "blocking",
                    "evidence": {"alignment_run_json": row.get("alignment_run_json", ""), "alignment_report": row.get("alignment_report", "")},
                    "fix_hint": "Inspect the WhisperX alignment report and prepare the local runtime before rerunning.",
                }
            )
    if key == "openai_cloud_asr":
        rows.append(
            {
                "code": "cloud_asr_reference_missing",
                "severity": "optional",
                "evidence": {"status": status},
                "fix_hint": "Run cloud ASR only on an explicit sample when uploading the audio is acceptable; it is quality-reference evidence, not the default path.",
            }
        )
    return rows


def _gates(
    rows: list[dict[str, Any]],
    *,
    speaker_requirement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_key = {row["key"]: row for row in rows}
    requirement = (
        speaker_requirement if isinstance(speaker_requirement, dict) else {}
    )
    required = bool(requirement.get("required"))
    minimum = max(1, int(requirement.get("min_speaker_count") or 1))
    speaker_ready_variants = []
    speaker_diagnostic_variants = []
    for row in rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        segment_count = int(metrics.get("segment_count") or 0)
        labeled_count = int(metrics.get("speaker_labeled_segment_count") or 0)
        speaker_count = int(metrics.get("speaker_count") or 0)
        complete_labels = (
            row.get("status") == "ok"
            and segment_count > 0
            and labeled_count == segment_count
            and speaker_count >= minimum
        )
        boundary = (
            row.get("operator_boundary")
            if isinstance(row.get("operator_boundary"), dict)
            else {}
        )
        if complete_labels and boundary.get("evaluation_only_known_speaker_count"):
            speaker_diagnostic_variants.append(str(row.get("key") or ""))
        elif complete_labels:
            speaker_ready_variants.append(str(row.get("key") or ""))
    return {
        "sensevoice_full_punc_ready": by_key.get("sensevoice_full_punc", {}).get("status") == "ok",
        "sensevoice_full_punc_campp_ready": by_key.get(
            "sensevoice_full_punc_campp", {}
        ).get("status")
        == "ok",
        "moss_transcribe_diarize_ready": by_key.get(
            "moss_transcribe_diarize", {}
        ).get("status")
        == "ok",
        "dolphin_sample_ready": by_key.get("dolphin", {}).get("status") == "ok",
        "whisperx_alignment_ready": by_key.get("whisperx_alignment", {}).get("status") == "ok",
        "cloud_sample_ready": by_key.get("openai_cloud_asr", {}).get("status") == "ok",
        "can_decide_second_asr": any(
            by_key.get(key, {}).get("status") == "ok"
            for key in (
                "moss_transcribe_diarize",
                "dolphin",
                "openai_cloud_asr",
            )
        ),
        "speaker_requirement_met": not required or bool(speaker_ready_variants),
        "speaker_ready_variants": speaker_ready_variants,
        "speaker_diagnostic_variants": speaker_diagnostic_variants,
    }


def _source_run_included_cloud_upload(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if row.get("key") == "openai_cloud_asr" and (row.get("status") not in {"", "preview", "planned_upload_disabled"}):
            return True
    return False


def _decision(
    rows: list[dict[str, Any]],
    *,
    speaker_requirement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Separate text readiness from speaker-attribution readiness.

    Intent: prevent a text-only ASR success from satisfying a dialogue task
    that explicitly requires different speakers to be labeled.
    Decision: reuse the reference transcript's anonymous speaker clusters and
    the existing per-variant speaker coverage metrics as a fail-closed gate.
    Reason: `speaker_count=0` can still produce readable text, but it cannot
    support a speaker-attributed transcript or downstream dialogue summary.
    Evidence: the fixed consultation sample has two reference speakers while
    the real SenseVoice baseline produced zero speaker labels.
    Effective scope: A/B recommendation status only; no transcript, reference,
    speaker role, ASR route, model cache, or production Bundle is modified.
    """

    requirement = (
        speaker_requirement if isinstance(speaker_requirement, dict) else {}
    )
    speaker_required = bool(requirement.get("required"))
    gates = _gates(rows, speaker_requirement=requirement)
    next_actions: list[str] = []
    if gates["sensevoice_full_punc_ready"]:
        status = "primary_asr_ready_second_asr_pending"
        second = "do_not_introduce_second_asr_by_default_yet"
        production = "sensevoice_full_punc"
        next_actions.append("Use sensevoice_full_punc as the local primary ASR input for postprocessed/corrected transcript.")
    else:
        status = "primary_asr_not_ready"
        second = "fix_sensevoice_full_punc_first"
        production = "blocked_until_primary_asr_ready"
        next_actions.append("Run or fix sensevoice_full_punc before comparing second ASR candidates.")
    if speaker_required:
        if not gates["speaker_requirement_met"]:
            status = "primary_text_ready_speaker_diarization_pending" if gates["sensevoice_full_punc_ready"] else "speaker_diarization_pending"
            production = "blocked_until_speaker_diarization_ready"
            next_actions.insert(
                0,
                "Do not promote the text-only baseline as a complete dialogue transcript: the evaluation reference requires anonymous speaker labels and no successful candidate currently covers every spoken segment.",
            )
        else:
            # Intent: distinguish complete anonymous labels from proven quality.
            # Decision: expose the candidate separately and keep production blocked
            # until the existing pyannote DER and MeetEval cpCER/tcpCER gates pass.
            # Reason: 100% labeled segments can still contain speaker confusion.
            # Evidence: the real CAM++ trial labeled 168/168 segments but measured
            # DER 0.24397, cpCER 0.36036 and tcpCER 0.52853 against the local window.
            # Effective scope: A/B recommendation metadata only; no transcript is
            # promoted, edited, rerun or sent to a provider.
            production = "blocked_until_speaker_quality_evaluation_passes"
            status = "speaker_candidate_ready_for_diarization_evaluation"
    dolphin = next((row for row in rows if row.get("key") == "dolphin"), {})
    dolphin_risks = " ".join(str(item) for item in (dolphin.get("risks") or []))
    if not gates["dolphin_sample_ready"]:
        if "torchcodec" in dolphin_risks.lower() or "torchcodec" in str(dolphin.get("status") or "").lower():
            next_actions.append("Keep Dolphin as an optional A/B candidate for now; the current Windows/TorchCodec runtime is not ready enough to introduce it as a second ASR source. If audio_extract_ok_but_dolphin_runtime_failed appears, fix TorchCodec/Dolphin runtime rather than changing video container extraction.")
        else:
            next_actions.append("Install/prepare Dolphin locally, then rerun the same 5-minute sample as second ASR evidence.")
    if not gates.get("whisperx_alignment_ready"):
        next_actions.append("Optionally run WhisperX on the same sample when word-level timestamps or speaker alignment need evaluation.")
    if not gates.get("sensevoice_full_punc_campp_ready"):
        next_actions.append("Prepare CAM++ explicitly, then run sensevoice_full_punc_campp on the same local sample with CUDA; do not infer speaker roles from cluster IDs.")
    else:
        next_actions.append("Evaluate the CAM++ candidate against an exact-bound reference with the existing transcript-stability-evaluation pyannote DER and MeetEval cpCER/tcpCER gates before production use.")
    if gates.get("speaker_diagnostic_variants"):
        next_actions.append("Compare the known-speaker-count diagnostic with automatic CAM++ using DER/cpCER/tcpCER. If it does not materially improve, speaker-count estimation is not the primary error source.")
    if not gates.get("moss_transcribe_diarize_ready"):
        next_actions.append("Prepare the pinned MOSS runtime and model explicitly, then run moss_transcribe_diarize on the same local sample; no silent download or provider fallback is allowed.")
    else:
        next_actions.append("Evaluate the MOSS candidate on the exact same bound reference with the existing pyannote DER and MeetEval cpCER/tcpCER gates before choosing between speaker candidates.")
    if not gates["cloud_sample_ready"]:
        next_actions.append("Optionally run cloud ASR on the same sample only when explicit upload is acceptable.")
    if gates["can_decide_second_asr"] and gates["speaker_requirement_met"]:
        status = "ready_for_human_quality_review"
        second = "compare_term_errors_before_defaulting_second_asr"
        next_actions.append("Review term/name/number errors across successful variants before enabling a second ASR in the main pipeline.")
    cloud = "optional_quality_reference_missing" if not gates["cloud_sample_ready"] else "available_for_quality_reference"
    speaker_candidates = list(gates["speaker_ready_variants"])
    speaker_candidate = speaker_candidates[0] if len(speaker_candidates) == 1 else ""
    if len(speaker_candidates) > 1:
        speaker_candidate_selection = "multiple_candidates_require_der_cpcer_tcpcer_comparison"
        next_actions.append("Multiple fully labeled speaker candidates are available; do not select by label coverage or text similarity alone. Compare DER, cpCER and tcpCER first.")
    elif speaker_candidates:
        speaker_candidate_selection = "single_candidate_pending_quality_gates"
    else:
        speaker_candidate_selection = "no_ready_candidate"
    return {
        "status": status,
        "production_recommendation": production,
        "speaker_evaluation_candidate": speaker_candidate,
        "speaker_evaluation_candidates": speaker_candidates,
        "speaker_candidate_selection": speaker_candidate_selection,
        "second_asr_recommendation": second,
        "cloud_asr_recommendation": cloud,
        "next_actions": next_actions,
    }


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# ASR A/B Comparison",
        "",
        f"- Status: `{result.get('status', '')}`",
        f"- Primary recommendation: `{result.get('primary_recommendation', '')}`",
        f"- Production recommendation: `{result.get('production_recommendation', '')}`",
        f"- Speaker evaluation candidate: `{result.get('speaker_evaluation_candidate', '')}`",
        f"- Speaker evaluation candidates: `{result.get('speaker_evaluation_candidates') or []}`",
        f"- Speaker candidate selection: `{result.get('speaker_candidate_selection', '')}`",
        f"- Speaker requirement: `{result.get('speaker_requirement') or 'not_required'}`",
        f"- Second ASR recommendation: `{result.get('second_asr_recommendation', '')}`",
        f"- Cloud ASR recommendation: `{result.get('cloud_asr_recommendation', '')}`",
        f"- Sample: `{result.get('sample_media_path', '')}`",
        f"- Reference transcript: `{result.get('reference_transcript', '')}`",
        f"- Best reference variant: `{result.get('best_reference_variant', '')}`",
        "",
        "## Variants",
        "",
        "| Variant | Status | Availability | Readability | Ref similarity | Segments | Chars | Punctuation | Speakers | Speaker duration | Risks |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result.get("variants") or []:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        lines.append(
            f"| `{row.get('key', '')}` | `{row.get('status', '')}` | `{row.get('availability', '')}` | "
            f"{row.get('readability_score', 0)} | {row.get('reference_similarity', 0)} | {metrics.get('segment_count', '')} | {metrics.get('char_count', '')} | {metrics.get('punctuation_count', '')} | {metrics.get('speaker_count', '')} | {metrics.get('speaker_labeled_duration_ratio', '')} | "
            f"{', '.join(row.get('risks') or [])} |"
        )
    lines.extend(["", "## Gates", ""])
    for key, value in (result.get("gates") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Next Actions", ""])
    for action in result.get("next_actions") or []:
        lines.append(f"- {action}")
    blocker_rows = [
        (row.get("key", ""), blocker)
        for row in (result.get("variants") or [])
        for blocker in (row.get("blockers") or [])
        if isinstance(blocker, dict)
    ]
    if blocker_rows:
        lines.extend(["", "## Blockers", ""])
        for key, blocker in blocker_rows:
            evidence = blocker.get("evidence") if isinstance(blocker.get("evidence"), dict) else {}
            evidence_text = ", ".join(f"{name}={value}" for name, value in evidence.items() if value not in {"", None})
            lines.append(f"- `{key}` / `{blocker.get('code', '')}`: {blocker.get('fix_hint', '')}")
            if evidence_text:
                lines.append(f"  Evidence: `{evidence_text}`")
    lines.extend([
        "",
        "## Boundary",
        "",
        "This comparison does not promote any transcript. It only decides whether the current sample is sufficient for choosing a second ASR path.",
    ])
    return "\n".join(lines).rstrip() + "\n"

def _best_reference_variant(rows: list[dict[str, Any]]) -> str:
    ready = [row for row in rows if row.get("status") == "ok" and float(row.get("reference_similarity") or 0.0) > 0]
    if not ready:
        return ""
    ready.sort(key=lambda row: float(row.get("reference_similarity") or 0.0), reverse=True)
    return str(ready[0].get("key") or "")


def _first_existing_path(*values: Any) -> Path | None:
    for value in values:
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if path.exists():
            return path.resolve()
    return None


def _read_optional_json(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    path = Path(str(value)).expanduser()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _excerpt_from_path(path: Path | None, *, start_seconds: float, end_seconds: float) -> str:
    if not path or not path.exists():
        return ""
    try:
        cues = parse_transcript(path)
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace")
    if end_seconds and end_seconds > start_seconds:
        return transcript_excerpt(cues, float(start_seconds), float(end_seconds))
    return " ".join(cue.text for cue in cues).strip()


def _reference_speaker_requirement(
    path: Path | None,
    *,
    start_seconds: float,
    end_seconds: float,
) -> dict[str, Any]:
    """Derive an evaluation gate from existing anonymous reference labels.

    This does not infer roles or turn a reference transcript into correction
    evidence. It only records whether the selected evaluation window is known
    to contain more than one speaker.
    """

    base = {
        "required": False,
        "min_speaker_count": 1,
        "distinct_speaker_count": 0,
        "spoken_segment_count": 0,
        "labeled_segment_count": 0,
        "source": "missing",
        "speaker_labels_are_anonymous": True,
        "speaker_roles_are_not_inferred": True,
    }
    if not path or not path.exists():
        return base
    try:
        cues = parse_transcript(path)
    except Exception:
        return {**base, "source": "reference_not_machine_readable"}
    if end_seconds and end_seconds > start_seconds:
        cues = [
            cue
            for cue in cues
            if float(cue.end) > float(start_seconds)
            and float(cue.start) < float(end_seconds)
        ]
    spoken = [cue for cue in cues if str(cue.text or "").strip()]
    labeled = [cue for cue in spoken if cue_speaker(cue)]
    speakers = sorted({cue_speaker(cue) for cue in labeled if cue_speaker(cue)})
    required = len(speakers) > 1
    return {
        **base,
        "required": required,
        "min_speaker_count": len(speakers) if required else 1,
        "distinct_speaker_count": len(speakers),
        "spoken_segment_count": len(spoken),
        "labeled_segment_count": len(labeled),
        "source": "evaluation_reference_labels",
    }


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return round(difflib.SequenceMatcher(a=_compact(left), b=_compact(right), autojunk=False).ratio(), 4)


def _compact(text: str) -> str:
    return "".join(ch for ch in str(text or "") if not ch.isspace())
