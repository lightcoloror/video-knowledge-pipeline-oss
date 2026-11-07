"""Independent coverage/overlap evidence and deterministic agreement sampling."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .asr_vad_chunking import read_vad_intervals
from .file_hash import sha256_file
from .interval_coverage import interval_coverage, merge_intervals
from .storage import read_json


def _binding(path: Path, payload: dict[str, Any], media: Path | None) -> dict[str, Any]:
    source = payload.get("source_media") if isinstance(payload.get("source_media"), dict) else {}
    expected = payload.get("source_media_sha256") or payload.get("media_sha256") or source.get("sha256")
    actual = sha256_file(media) if media and media.is_file() else None
    if expected and actual and expected != actual:
        raise ValueError("Review evidence belongs to different audio")
    return {"path": str(path), "sha256": sha256_file(path),
            "media_binding": "verified" if expected and actual else "unverified"}


def _intervals(rows: list[Any]) -> list[tuple[float, float]]:
    result = []
    for row in rows:
        a, b = float(row["start"]), float(row["end"])
        if not math.isfinite(a) or not math.isfinite(b) or a < 0 or b <= a:
            raise ValueError("Review evidence requires valid second-based intervals")
        result.append((a, b))
    return result


def build_review_audit(primary: list[Any], secondary: list[Any], windows: list[dict[str, Any]], *,
                       media: Path | None = None, speech_activity_json: str | Path | None = None,
                       diarization_json: str | Path | None = None, sample_size: int = 3) -> dict[str, Any]:
    coverage: dict[str, Any] = {"status": "not_assessed", "reason": "independent_activity_evidence_missing"}
    overlap: dict[str, Any] = {"status": "not_assessed", "reason": "diarization_evidence_missing"}
    risks: list[dict[str, Any]] = []
    if speech_activity_json:
        path = Path(speech_activity_json)
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError("Activity evidence must include a metadata object")
        binding = _binding(path, payload, media)
        if "speech_intervals" in payload:
            target = _intervals(payload["speech_intervals"])
        else:
            target = _intervals(read_vad_intervals(path))
        coverage = {"status": "assessed" if binding["media_binding"] == "verified" else "unverified",
                    "binding": binding, "detector": payload.get("detector", "unspecified"),
                    "evidence_kind": payload.get("evidence_kind", "speech_activity_candidate"),
                    "transcript_accuracy_proven": False}
        for name, cues in (("primary", primary), ("secondary", secondary), ("both", primary + secondary)):
            report = interval_coverage(target, [(c.start, c.end) for c in cues if str(c.text).strip()], minimum_gap_seconds=0.2)
            coverage[name] = report
        for row in coverage["both"].get("all_gaps", []):
            risks.append({"start": row["start"], "end": row["end"], "reason": "both_asr_uncovered_activity"})
    if diarization_json:
        path = Path(diarization_json)
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError("Diarization evidence must include metadata")
        binding = _binding(path, payload, media)
        rows = payload.get("speaker_diarization", payload.get("segments"))
        if not isinstance(rows, list):
            raise ValueError("Diarization evidence needs speaker_diarization or segments")
        _intervals(rows)
        events: dict[float, list[tuple[str, int]]] = {}
        for row in rows:
            speaker = row.get("speaker")
            if speaker is None or not str(speaker):
                raise ValueError("Diarization interval needs speaker identity")
            events.setdefault(float(row["start"]), []).append((str(speaker), 1))
            events.setdefault(float(row["end"]), []).append((str(speaker), -1))
        active: dict[str, int] = {}
        previous = None
        spans = []
        for time, changes in sorted(events.items()):
            if previous is not None and time > previous and sum(v > 0 for v in active.values()) >= 2:
                spans.append((previous, time))
            for speaker, change in changes:
                active[speaker] = active.get(speaker, 0) + change
            previous = time
        intervals = [{"start": a, "end": b} for a, b in merge_intervals(spans)]
        exclusive = payload.get("exclusive", False) or payload.get("diarization_type") == "exclusive"
        overlap = {"status": "not_assessed" if exclusive else "assessed" if binding["media_binding"] == "verified" else "unverified",
                   "binding": binding, "intervals": intervals, "exclusive_input": bool(exclusive),
                   "reason": "exclusive_timeline_cannot_measure_overlap" if exclusive else "regular_timeline"}
        risks.extend({**r, "reason": "overlapping_speakers"} for r in intervals)
    agreements = [w for w in windows if w.get("status") == "agreement"]
    n = min(max(0, int(sample_size)), len(agreements))
    indexes = sorted({round(i * (len(agreements) - 1) / max(1, n - 1)) for i in range(n)})
    samples = [agreements[i]["window_id"] for i in indexes]
    for window in windows:
        reasons = [r["reason"] for r in risks if min(window.get("group_end", window["end"]), r["end"]) > max(window.get("group_start", window["start"]), r["start"])]
        if window["window_id"] in samples:
            reasons.append("agreement_spot_check")
        window["review_reasons"] = sorted(set(window.get("review_reasons", []) + reasons))
        window["review_required"] = window["status"] not in {"agreement", "not_comparable_partial_scope"} or bool(reasons)
    return {"coverage": coverage, "overlap": overlap, "agreement_samples": samples,
            "agreement_sampling_status": "pending_review" if samples else "no_agreement_windows",
            "independent_risk_windows": risks, "agreement_is_not_ground_truth": True,
            "audit_complete": coverage["status"] == "assessed" and overlap["status"] == "assessed"}
