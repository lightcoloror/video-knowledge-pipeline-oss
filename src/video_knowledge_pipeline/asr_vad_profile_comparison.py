from __future__ import annotations

from pathlib import Path
from typing import Any

from .asr_vad_chunking import read_vad_intervals
from .interval_coverage import interval_coverage
from .models import now_iso
from .storage import read_json, write_json
from .file_hash import sha256_file


SCHEMA = "video_knowledge_pipeline.asr_vad_profile_comparison.v1"
LABEL_SCHEMA = "video_knowledge_pipeline.asr_vad_human_labels.v1"
_LABELS = {"speech", "non_speech", "uncertain"}


def compare_asr_vad_profiles(
    authoritative_vad_path: str | Path,
    permissive_vad_path: str | Path,
    activity_audit_path: str | Path,
    *,
    labels_path: str | Path | None = None,
    output_path: str | Path | None = None,
    minimum_support_ratio: float = 0.5,
    write: bool = True,
) -> dict[str, Any]:
    """Compare strict and permissive FSMN-VAD evidence without changing either."""

    authoritative_path = Path(authoritative_vad_path).expanduser().resolve()
    permissive_path = Path(permissive_vad_path).expanduser().resolve()
    audit_path = Path(activity_audit_path).expanduser().resolve()
    authoritative = _read_object(authoritative_path, "authoritative VAD")
    permissive = _read_object(permissive_path, "permissive VAD")
    audit = _read_object(audit_path, "activity audit")
    support_threshold = float(minimum_support_ratio)
    if not 0.0 <= support_threshold <= 1.0:
        raise ValueError("minimum_support_ratio must be between 0 and 1")

    authoritative_sha = sha256_file(authoritative_path)
    permissive_sha = sha256_file(permissive_path)
    audit_sha = sha256_file(audit_path)
    _validate_profile_pair(
        authoritative,
        permissive,
        audit,
        authoritative_sha=authoritative_sha,
    )

    authoritative_intervals = [
        (float(row["start"]), float(row["end"]))
        for row in read_vad_intervals(authoritative_path)
    ]
    permissive_intervals = [
        (float(row["start"]), float(row["end"]))
        for row in read_vad_intervals(permissive_path)
    ]
    candidates = _candidate_rows(
        audit,
        permissive_intervals,
        minimum_support_ratio=support_threshold,
    )
    labels, label_source = _read_labels(
        labels_path,
        candidate_ids={str(row["candidate_id"]) for row in candidates},
        authoritative_sha=authoritative_sha,
        permissive_sha=permissive_sha,
        audit_sha=audit_sha,
    )
    for row in candidates:
        row["human_label"] = labels.get(str(row["candidate_id"]), "unlabeled")
    human_metrics = _human_metrics(candidates)

    activity = [
        (float(row["start"]), float(row["end"]))
        for row in (audit.get("audio_probe") or {}).get("activity_intervals") or []
        if isinstance(row, dict)
    ]
    permissive_delta = interval_coverage(
        permissive_intervals,
        authoritative_intervals,
        minimum_gap_seconds=0.0,
    )
    outside_activity = (
        interval_coverage(
            permissive_intervals,
            activity,
            minimum_gap_seconds=0.0,
        )
        if activity
        else None
    )
    target = (
        Path(output_path).expanduser().resolve()
        if output_path
        else audit_path.with_name("asr-vad-profile-comparison.json")
    )
    label_template_path = (
        target.with_name("asr-vad-human-labels.template.json")
        if candidates
        else None
    )
    label_template = {
        "schema": LABEL_SCHEMA,
        "authoritative_vad_sha256": authoritative_sha,
        "permissive_vad_sha256": permissive_sha,
        "activity_audit_sha256": audit_sha,
        "allowed_labels": sorted(_LABELS),
        "labels": [
            {
                "candidate_id": str(row["candidate_id"]),
                "start": row["start"],
                "end": row["end"],
                "label": "",
                "notes": "",
            }
            for row in candidates
        ],
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": True,
        "status": human_metrics["status"],
        "authoritative_vad": {
            "path": str(authoritative_path),
            "sha256": authoritative_sha,
            "settings": authoritative["vad_settings"],
        },
        "permissive_vad": {
            "path": str(permissive_path),
            "sha256": permissive_sha,
            "settings": permissive["vad_settings"],
            "candidate_only": True,
        },
        "activity_audit": {
            "path": str(audit_path),
            "sha256": audit_sha,
            "candidate_gap_count": len(candidates),
        },
        "minimum_support_ratio": support_threshold,
        "candidate_comparisons": candidates,
        "same_model_support_counts": _support_counts(candidates),
        "permissive_delta_vs_authoritative": permissive_delta,
        "permissive_outside_audio_activity": outside_activity,
        "human_labels": {
            "path": label_source,
            "template_path": str(label_template_path) if label_template_path else "",
            "template_written": bool(write and not labels_path and label_template_path),
            **human_metrics,
        },
        "decision_boundary": {
            "same_model_second_pass_is_independent_evidence": False,
            "candidate_only": True,
            "authoritative_vad_modified": False,
            "chunk_manifest_modified": False,
            "canonical_transcript_modified": False,
            "production_default_change_allowed": False,
            "network_call": False,
        },
        "recommended_action": (
            "label every candidate as speech, non_speech, or uncertain; choose a production "
            "profile only after fixed-sample acceptance thresholds are defined"
            if candidates
            else "retain the authoritative profile; this sample has no blind-spot calibration signal"
        ),
        "output_path": str(target),
        "updated_at": now_iso(),
    }
    if write:
        if not labels_path and label_template_path:
            write_json(label_template_path, label_template)
        write_json(target, result)
    return result


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _validate_profile_pair(
    authoritative: dict[str, Any],
    permissive: dict[str, Any],
    audit: dict[str, Any],
    *,
    authoritative_sha: str,
) -> None:
    if str(audit.get("schema") or "") != "video_knowledge_pipeline.asr_vad_activity_audit.v1":
        raise ValueError("activity audit schema is not supported")
    if str(audit.get("vad_sha256") or "").lower() != authoritative_sha.lower():
        raise ValueError("activity audit does not bind the authoritative VAD hash")
    if str(audit.get("status") or "") not in {"review_required", "passed"}:
        raise ValueError("activity audit must have a completed review_required or passed status")
    if authoritative.get("candidate_only") is True:
        raise ValueError("authoritative VAD cannot be candidate-only")
    if str(authoritative.get("evidence_profile") or "authoritative") != "authoritative":
        raise ValueError("authoritative VAD evidence_profile must be authoritative")
    if permissive.get("candidate_only") is not True:
        raise ValueError("permissive VAD must be explicitly candidate-only")
    if str(permissive.get("evidence_profile") or "") != "candidate-permissive":
        raise ValueError("permissive VAD evidence_profile must be candidate-permissive")

    for field in ("input", "resolved_model", "model_revision"):
        if str(authoritative.get(field) or "") != str(permissive.get(field) or ""):
            raise ValueError(f"VAD profile pair differs in {field}")
    strict_settings = authoritative.get("vad_settings")
    loose_settings = permissive.get("vad_settings")
    if not isinstance(strict_settings, dict) or not isinstance(loose_settings, dict):
        raise ValueError("both VAD profiles must record vad_settings")
    strict_threshold = float(strict_settings.get("speech_noise_threshold"))
    loose_threshold = float(loose_settings.get("speech_noise_threshold"))
    strict_silence = int(strict_settings.get("max_end_silence_time_ms"))
    loose_silence = int(loose_settings.get("max_end_silence_time_ms"))
    if loose_threshold > strict_threshold:
        raise ValueError("candidate-permissive speech threshold must not exceed authoritative")
    if loose_silence < strict_silence:
        raise ValueError("candidate-permissive end silence must not be shorter than authoritative")
    if int(loose_settings.get("max_single_segment_time_ms")) != int(
        strict_settings.get("max_single_segment_time_ms")
    ):
        raise ValueError("VAD profile pair must keep max_single_segment_time_ms identical")


def _candidate_rows(
    audit: dict[str, Any],
    permissive_intervals: list[tuple[float, float]],
    *,
    minimum_support_ratio: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in audit.get("candidate_gaps") or []:
        if not isinstance(raw, dict):
            continue
        candidate_id = str(raw.get("candidate_id") or "").strip()
        start = float(raw.get("start"))
        end = float(raw.get("end"))
        if not candidate_id or end <= start:
            raise ValueError("activity audit candidate gap is invalid")
        coverage = interval_coverage(
            [(start, end)],
            permissive_intervals,
            minimum_gap_seconds=0.0,
        )
        ratio = float(coverage["coverage_ratio"])
        if ratio >= minimum_support_ratio:
            support_status = "same_model_permissive_supported"
        elif ratio > 0:
            support_status = "same_model_permissive_partial"
        else:
            support_status = "unresolved"
        result.append(
            {
                "candidate_id": candidate_id,
                "start": round(start, 6),
                "end": round(end, 6),
                "duration_seconds": round(end - start, 6),
                "permissive_coverage_ratio": ratio,
                "support_status": support_status,
                "candidate_only": True,
                "automatic_acceptance_allowed": False,
            }
        )
    return result


def _read_labels(
    labels_path: str | Path | None,
    *,
    candidate_ids: set[str],
    authoritative_sha: str,
    permissive_sha: str,
    audit_sha: str,
) -> tuple[dict[str, str], str]:
    if not labels_path:
        return {}, ""
    path = Path(labels_path).expanduser().resolve()
    payload = _read_object(path, "VAD human labels")
    if str(payload.get("schema") or "") != LABEL_SCHEMA:
        raise ValueError("VAD human label schema is not supported")
    expected = {
        "authoritative_vad_sha256": authoritative_sha,
        "permissive_vad_sha256": permissive_sha,
        "activity_audit_sha256": audit_sha,
    }
    for field, value in expected.items():
        if str(payload.get(field) or "").lower() != value.lower():
            raise ValueError(f"VAD human labels do not bind {field}")
    labels: dict[str, str] = {}
    for raw in payload.get("labels") or []:
        if not isinstance(raw, dict):
            continue
        candidate_id = str(raw.get("candidate_id") or "").strip()
        label = str(raw.get("label") or "").strip().lower()
        if candidate_id not in candidate_ids:
            raise ValueError(f"VAD human label references unknown candidate: {candidate_id}")
        if candidate_id in labels:
            raise ValueError(f"duplicate VAD human label: {candidate_id}")
        if label not in _LABELS:
            raise ValueError(f"unsupported VAD human label: {label}")
        labels[candidate_id] = label
    return labels, str(path)


def _human_metrics(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 0,
    }
    definite = 0
    uncertain = 0
    for row in candidates:
        label = str(row.get("human_label") or "unlabeled")
        if label == "uncertain":
            uncertain += 1
            continue
        if label not in {"speech", "non_speech"}:
            continue
        definite += 1
        predicted = row["support_status"] == "same_model_permissive_supported"
        if predicted and label == "speech":
            confusion["true_positive"] += 1
        elif predicted:
            confusion["false_positive"] += 1
        elif label == "speech":
            confusion["false_negative"] += 1
        else:
            confusion["true_negative"] += 1
    precision_denominator = confusion["true_positive"] + confusion["false_positive"]
    recall_denominator = confusion["true_positive"] + confusion["false_negative"]
    precision = (
        confusion["true_positive"] / precision_denominator
        if precision_denominator
        else None
    )
    recall = (
        confusion["true_positive"] / recall_denominator
        if recall_denominator
        else None
    )
    total = len(candidates)
    if not total:
        status = "no_candidates"
    elif definite == total:
        status = "calibration_labeled"
    elif definite or uncertain:
        status = "calibration_partially_labeled"
    else:
        status = "awaiting_human_labels"
    return {
        "status": status,
        "candidate_count": total,
        "definite_label_count": definite,
        "uncertain_label_count": uncertain,
        "unlabeled_count": total - definite - uncertain,
        "confusion_counts": confusion,
        "precision": round(precision, 6) if precision is not None else None,
        "recall": round(recall, 6) if recall is not None else None,
    }


def _support_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "same_model_permissive_supported": 0,
        "same_model_permissive_partial": 0,
        "unresolved": 0,
    }
    for row in candidates:
        counts[str(row["support_status"])] += 1
    return counts
