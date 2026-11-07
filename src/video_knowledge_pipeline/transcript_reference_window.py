from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from .file_hash import sha256_file
from .storage import read_json, write_json
from .transcript import parse_transcript
from .transcript_semantic_correction import (
    apply_human_confirmed_source_fidelity_decisions,
)


SCHEMA = "video_knowledge_pipeline.transcript_reference_window.v1"
RECEIPT_SCHEMA = "video_knowledge_pipeline.transcript_reference_window_receipt.v1"


def export_transcript_reference_window(
    transcript_path: str | Path,
    output_json: str | Path,
    *,
    start_seconds: float,
    end_seconds: float,
    rebase_timestamps: bool = True,
    human_corrections_json: str | Path | None = None,
    validation_scope: str = "source",
    write: bool = True,
) -> dict[str, Any]:
    """Export one bounded, speaker-preserving evaluation reference.

    Intent: make a fixed ASR A/B sample and its reference use the same time
    window without losing anonymous speaker attribution.
    Decision: adapt VKP's existing ``parse_transcript``/``TranscriptCue``
    contract and emit only a thin, content-addressed JSON artifact.
    Reason: the existing text-only excerpt helper cannot support DER, cpCER, or
    tcpCER and copying transcript parsing would create a second source of truth.
    Evidence: the user-provided GetBrain transcript parses into 433 cues and two
    anonymous speakers, while ``asr-ab-compare`` currently flattens excerpts.
    Effective scope: local, evaluation-only artifacts; no prompt, hotword,
    correction, provider route, upload, or canonical transcript is changed.
    """

    source = Path(transcript_path).expanduser().resolve()
    destination = Path(output_json).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == destination:
        raise ValueError("reference window output must differ from the source transcript")
    start = float(start_seconds)
    end = float(end_seconds)
    if start < 0:
        raise ValueError("start_seconds must be non-negative")
    if end <= start:
        raise ValueError("end_seconds must be greater than start_seconds")
    if validation_scope not in {"source", "window"}:
        raise ValueError("validation_scope must be source or window")

    cues = parse_transcript(source)
    if not cues:
        raise ValueError("source transcript contains no parseable cues")
    if validation_scope == "source":
        _require_monotonic_cues(cues)
    else:
        # Some legacy chunked transcripts contain a tiny overlap/order reset at
        # a distant chunk boundary.  A bounded review export may validate only
        # the requested window, but it must never suppress disorder inside the
        # exported evidence itself.
        _require_monotonic_cues(_window_cues(cues, start=start, end=end))
    correction_receipt: dict[str, Any] = {
        "path": "",
        "sha256": "",
        "decision_count": 0,
        "applied_count": 0,
        "source_applied_count": 0,
        "human_confirmed": False,
    }
    if human_corrections_json:
        cues, correction_receipt = _apply_human_corrections(
            cues,
            source_sha256=sha256_file(source),
            corrections_path=human_corrections_json,
        )

    segments: list[dict[str, Any]] = []
    speaker_counts: Counter[str] = Counter()
    speaker_durations: Counter[str] = Counter()
    for index, cue in enumerate(cues, start=1):
        cue_start = float(cue.start)
        cue_end = max(cue_start, float(cue.end))
        overlap_start = max(start, cue_start)
        overlap_end = min(end, cue_end)
        is_point_cue = cue_end == cue_start and start <= cue_start < end
        if overlap_end <= overlap_start and not is_point_cue:
            continue
        clipped_start = overlap_start if not is_point_cue else cue_start
        clipped_end = overlap_end if not is_point_cue else cue_start
        output_start = clipped_start - start if rebase_timestamps else clipped_start
        output_end = clipped_end - start if rebase_timestamps else clipped_end
        segment_id = str(cue.segment_id or f"segment-{index:06d}")
        source_ids = [
            str(value)
            for value in (cue.source_segment_ids or [segment_id])
            if str(value).strip()
        ]
        speaker = str(cue.speaker or "").strip()
        transformation = {
            "type": "evaluation_time_window_clip",
            "window_start_seconds": start,
            "window_end_seconds": end,
            "source_start_seconds": cue_start,
            "source_end_seconds": cue_end,
            "clipped_start_seconds": clipped_start,
            "clipped_end_seconds": clipped_end,
            "timestamps_rebased": bool(rebase_timestamps),
        }
        segments.append(
            {
                "id": segment_id,
                "segment_id": segment_id,
                "source_segment_ids": source_ids,
                "start": round(output_start, 6),
                "end": round(output_end, 6),
                "text": cue.text,
                "speaker": speaker,
                "speaker_role": str(cue.speaker_role or ""),
                "transformations": [
                    *[dict(value) for value in cue.transformations],
                    transformation,
                ],
                "metadata": {
                    **dict(cue.metadata),
                    "evaluation_only": True,
                    "source_start_seconds": cue_start,
                    "source_end_seconds": cue_end,
                },
            }
        )
        if speaker:
            speaker_counts[speaker] += 1
            speaker_durations[speaker] += max(0.0, clipped_end - clipped_start)

    if not segments:
        raise ValueError("the requested window contains no transcript cues")
    correction_receipt["applied_count"] = sum(
        int(transformation.get("correction_count") or 0)
        for segment in segments
        for transformation in segment.get("transformations", [])
        if transformation.get("type")
        == "human_confirmed_source_fidelity_correction"
    )

    window_duration = end - start
    payload = {
        "schema": SCHEMA,
        "status": "ready",
        "role": "evaluation_only_reference",
        "duration_seconds": window_duration,
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "parser": "video_knowledge_pipeline.transcript.parse_transcript",
            "cue_count": len(cues),
        },
        "human_corrections": correction_receipt,
        "window": {
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": window_duration,
            "timestamps_rebased": bool(rebase_timestamps),
            "validation_scope": validation_scope,
        },
        "speaker_evidence": {
            "speaker_count": len(speaker_counts),
            "anonymous_labels": sorted(speaker_counts),
            "segment_count_by_speaker": dict(sorted(speaker_counts.items())),
            "labeled_duration_seconds_by_speaker": {
                key: round(value, 6)
                for key, value in sorted(speaker_durations.items())
            },
        },
        "segments": segments,
        "policy": {
            "evaluation_only": True,
            "must_not_enter_prompt_hotwords_or_routing": True,
            "must_not_promote_or_correct_transcript": True,
            "speaker_labels_are_anonymous": True,
            "speaker_roles_are_not_inferred": True,
            "source_order_and_segment_identity_preserved": True,
        },
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    planned_sha256 = hashlib.sha256(serialized).hexdigest()
    if write:
        write_json(destination, payload)
        artifact_sha256 = sha256_file(destination)
    else:
        artifact_sha256 = planned_sha256

    return {
        "schema": RECEIPT_SCHEMA,
        "ok": True,
        "status": "written" if write else "preview",
        "source_path": str(source),
        "source_sha256": payload["source"]["sha256"],
        "artifact_path": str(destination),
        "artifact_sha256": artifact_sha256,
        "artifact_written": bool(write),
        "window": payload["window"],
        "segment_count": len(segments),
        "speaker_evidence": payload["speaker_evidence"],
        "human_corrections": {
            "path": correction_receipt["path"],
            "sha256": correction_receipt["sha256"],
            "decision_count": correction_receipt["decision_count"],
            "applied_count": correction_receipt["applied_count"],
            "source_applied_count": correction_receipt["source_applied_count"],
            "human_confirmed": correction_receipt["human_confirmed"],
        },
        "operator_boundary": payload["policy"],
    }


def _apply_human_corrections(
    cues: list[Any],
    *,
    source_sha256: str,
    corrections_path: str | Path,
) -> tuple[list[Any], dict[str, Any]]:
    path = Path(corrections_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("human corrections JSON must be an object")
    bound_source_sha256 = str(payload.get("source_sha256") or "")
    if not bound_source_sha256:
        raise ValueError("human corrections JSON requires source_sha256")
    if bound_source_sha256 != source_sha256:
        raise ValueError("human corrections source_sha256 does not match transcript")
    decisions = [
        dict(value)
        for value in (payload.get("decisions") or [])
        if isinstance(value, dict)
    ]
    if not decisions:
        raise ValueError("human corrections JSON contains no decisions")
    corrected_segments, applied = apply_human_confirmed_source_fidelity_decisions(
        cues,
        decisions,
    )
    if len(corrected_segments) != len(cues):
        raise ValueError("human source-fidelity corrections changed segment structure")
    corrected_cues = []
    for cue, segment in zip(cues, corrected_segments, strict=True):
        corrections = [
            dict(value)
            for value in (segment.get("semantic_corrections") or [])
            if isinstance(value, dict)
        ]
        transformations = list(cue.transformations)
        if corrections:
            transformations.append(
                {
                    "type": "human_confirmed_source_fidelity_correction",
                    "candidate_ids": [
                        str(value.get("candidate_id") or "")
                        for value in corrections
                    ],
                    "correction_count": len(corrections),
                    "source_corrections_sha256": sha256_file(path),
                }
            )
        corrected_cues.append(
            replace(
                cue,
                text=str(segment.get("text") or cue.text),
                transformations=transformations,
            )
        )
    return corrected_cues, {
        "path": str(path),
        "sha256": sha256_file(path),
        "decision_count": len(decisions),
        "applied_count": 0,
        "source_applied_count": len(applied),
        "human_confirmed": True,
    }


def _require_monotonic_cues(cues: list[Any]) -> None:
    previous_start = -1.0
    for index, cue in enumerate(cues, start=1):
        start = float(cue.start)
        end = float(cue.end)
        if start < previous_start:
            raise ValueError(f"source transcript cue order is non-monotonic at index {index}")
        if end < start:
            raise ValueError(f"source transcript cue has end before start at index {index}")
        previous_start = start


def _window_cues(cues: list[Any], *, start: float, end: float) -> list[Any]:
    result = []
    for cue in cues:
        cue_start = float(cue.start)
        cue_end = max(cue_start, float(cue.end))
        if cue_end > start and cue_start < end:
            result.append(cue)
    return result
