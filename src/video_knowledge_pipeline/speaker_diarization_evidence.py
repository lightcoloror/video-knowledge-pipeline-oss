from __future__ import annotations

import bisect
import os
import re
import shutil
import subprocess
import wave
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .file_hash import sha256_file
from .models import TranscriptCue, now_iso
from .storage import read_json, write_json
from .transcript import parse_transcript


PLAN_SCHEMA = "video_knowledge_pipeline.sherpa_onnx_speaker_diarization_plan.v1"
EVIDENCE_SCHEMA = "video_knowledge_pipeline.speaker_diarization_evidence.v1"
CANDIDATE_TRANSCRIPT_SCHEMA = (
    "video_knowledge_pipeline.speaker_assigned_transcript_candidate.v1"
)
UPSTREAM_PROJECT = "k2-fsa/sherpa-onnx"
UPSTREAM_COMMIT = "75e1fc31e747194c546787ec7b40a7e0b390dc4b"
ASSIGNMENT_REFERENCE_PROJECT = "m-bain/whisperX"
ASSIGNMENT_REFERENCE_COMMIT = "5f2f9d4320dd93a7d12f5ba2495eef7e0a5af963"
DEFAULT_COMMAND = "sherpa-onnx-offline-speaker-diarization"
DEFAULT_CLUSTER_THRESHOLD = 0.90
DEFAULT_MIN_DURATION_ON = 0.30
DEFAULT_MIN_DURATION_OFF = 0.50
DEFAULT_MIN_COVERAGE_RATIO = 0.50
DEFAULT_MIN_DOMINANCE_RATIO = 0.60

_SEGMENT_RE = re.compile(
    r"^\s*(?P<start>\d+(?:\.\d+)?)\s+--\s+"
    r"(?P<end>\d+(?:\.\d+)?)\s+"
    r"(?P<speaker>speaker_\d+)\s*$",
    re.IGNORECASE,
)


def plan_sherpa_speaker_diarization(
    bundle_dir: str | Path,
    media_path: str | Path,
    transcript_path: str | Path,
    *,
    command: str = "",
    segmentation_model: str | Path = "",
    embedding_model: str | Path = "",
    provider: str = "cuda",
    num_speakers: int | None = None,
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    min_duration_on: float = DEFAULT_MIN_DURATION_ON,
    min_duration_off: float = DEFAULT_MIN_DURATION_OFF,
    write: bool = True,
) -> dict[str, Any]:
    """Prepare sherpa-onnx diarization without executing or downloading anything.

    Intent: provide VKP with a second, fully local speaker-evidence route.
    Decision: invoke sherpa-onnx's official CLI and adapt only its interval
    output; do not reproduce segmentation, embedding, or clustering inference.
    Reason: the pinned upstream already implements the mature pipeline and
    exposes CUDA providers for both segmentation and embedding.
    Evidence: sherpa-onnx commit ``75e1fc3`` registers
    ``segmentation.provider`` and ``embedding.provider`` and emits
    ``start -- end speaker_NN`` rows.
    Effective scope: planning and readiness only. No model execution, network,
    download, transcript mutation, or CPU fallback occurs here.
    """

    root = Path(bundle_dir).expanduser().resolve()
    media = Path(media_path).expanduser().resolve()
    transcript = Path(transcript_path).expanduser().resolve()
    command_value = str(
        command
        or os.environ.get("LECTURE_SHERPA_DIARIZATION_COMMAND")
        or DEFAULT_COMMAND
    ).strip()
    segmentation = _configured_path(
        segmentation_model,
        "LECTURE_SHERPA_DIARIZATION_SEGMENTATION_MODEL",
    )
    embedding = _configured_path(
        embedding_model,
        "LECTURE_SHERPA_DIARIZATION_EMBEDDING_MODEL",
    )
    provider_key = str(provider or "cuda").strip().lower()
    if provider_key not in {"cuda", "cpu", "coreml"}:
        raise ValueError("provider must be one of: cuda, cpu, coreml")
    if num_speakers is not None and int(num_speakers) < 1:
        raise ValueError("num_speakers must be positive when supplied")
    if not 0 < float(cluster_threshold) <= 1:
        raise ValueError("cluster_threshold must be between 0 and 1")
    if min_duration_on < 0 or min_duration_off < 0:
        raise ValueError("minimum durations must be non-negative")

    command_path = _resolve_command(command_value)
    audio_probe = _wave_probe(media)
    runtime_probe = _runtime_probe(command_path)
    blockers = _plan_blockers(
        media=media,
        transcript=transcript,
        segmentation=segmentation,
        embedding=embedding,
        command_path=command_path,
        runtime_probe=runtime_probe,
        audio_probe=audio_probe,
    )
    command_argv = _command_argv(
        command_path,
        media=media,
        segmentation=segmentation,
        embedding=embedding,
        provider=provider_key,
        num_speakers=num_speakers,
        cluster_threshold=float(cluster_threshold),
        min_duration_on=float(min_duration_on),
        min_duration_off=float(min_duration_off),
    )
    output_dir = root / "transcripts" / "speaker-diarization"
    plan_path = output_dir / "sherpa-onnx-plan.json"
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "ready" if not blockers else "blocked",
        "ready": not blockers,
        "bundle_dir": str(root),
        "media": _artifact_identity(media),
        "transcript": _artifact_identity(transcript),
        "runtime": {
            "command": command_value,
            "command_path": str(command_path) if command_path else "",
            "runtime_probe": runtime_probe,
            "provider": provider_key,
            "gpu_required": provider_key == "cuda",
            "automatic_cpu_fallback": False,
        },
        "models": {
            "segmentation": _artifact_identity(segmentation),
            "embedding": _artifact_identity(embedding),
        },
        "parameters": {
            "num_speakers": int(num_speakers) if num_speakers is not None else None,
            "cluster_threshold": float(cluster_threshold),
            "min_duration_on": float(min_duration_on),
            "min_duration_off": float(min_duration_off),
        },
        "audio_probe": audio_probe,
        "command_argv": command_argv,
        "blockers": blockers,
        "upstream": {
            "project": UPSTREAM_PROJECT,
            "commit": UPSTREAM_COMMIT,
            "entrypoint": DEFAULT_COMMAND,
            "reuse_mode": "official_cli_contract",
        },
        "assignment_reference": {
            "project": ASSIGNMENT_REFERENCE_PROJECT,
            "commit": ASSIGNMENT_REFERENCE_COMMIT,
            "algorithm": "maximum_overlap_per_transcript_segment",
            "nearest_fill_enabled": False,
            "reuse_mode": "independent_adaptation",
        },
        "operator_boundary": {
            "local_only": True,
            "models_executed": False,
            "downloads_performed": False,
            "network_calls": 0,
            "candidate_evidence_only": True,
            "primary_transcript_mutated": False,
            "automatic_local_remote_fallback": False,
            "automatic_cpu_fallback": False,
        },
        "outputs": {
            "plan": str(plan_path),
            "stdout": str(output_dir / "sherpa-onnx.stdout.txt"),
            "stderr": str(output_dir / "sherpa-onnx.stderr.txt"),
            "evidence": str(output_dir / "speaker-diarization-evidence.json"),
            "candidate_transcript": str(
                output_dir / "speaker-assigned-transcript.candidate.json"
            ),
        },
        "updated_at": now_iso(),
    }
    if write:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(plan_path, plan)
    return plan


def run_sherpa_speaker_diarization_plan(
    plan_path: str | Path,
    *,
    execute: bool = False,
    timeout_seconds: int = 3600,
    min_coverage_ratio: float = DEFAULT_MIN_COVERAGE_RATIO,
    min_dominance_ratio: float = DEFAULT_MIN_DOMINANCE_RATIO,
    write: bool = True,
) -> dict[str, Any]:
    """Preview or execute one exact local sherpa-onnx diarization plan."""

    path = Path(plan_path).expanduser().resolve()
    plan = _mapping(read_json(path), "speaker diarization plan")
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"unsupported speaker diarization plan schema: {plan.get('schema')}")
    _validate_ratio(min_coverage_ratio, "min_coverage_ratio")
    _validate_ratio(min_dominance_ratio, "min_dominance_ratio")
    identity_blockers = _identity_blockers(plan)
    blockers = [str(value) for value in plan.get("blockers") or []]
    blockers.extend(identity_blockers)
    result: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "status": "preview" if not execute and not blockers else "blocked" if blockers else "running",
        "ok": not blockers,
        "execute": bool(execute),
        "plan_path": str(path),
        "blockers": blockers,
        "operator_boundary": {
            **dict(plan.get("operator_boundary") or {}),
            "models_executed": False,
            "primary_transcript_mutated": False,
        },
        "updated_at": now_iso(),
    }
    if blockers or not execute:
        return result

    argv = [str(value) for value in plan.get("command_argv") or []]
    if not argv:
        raise ValueError("speaker diarization plan contains no command argv")
    outputs = _mapping(plan.get("outputs"), "speaker diarization outputs")
    stdout_path = Path(str(outputs["stdout"])).resolve()
    stderr_path = Path(str(outputs["stderr"])).resolve()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        argv,
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(1, int(timeout_seconds)),
    )
    if write:
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        result.update(
            {
                "ok": False,
                "status": "failed",
                "returncode": int(completed.returncode),
                "blockers": ["sherpa_onnx_execution_failed"],
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
        )
        return result

    expected_provider = str((plan.get("runtime") or {}).get("provider") or "")
    if expected_provider == "cuda" and (completed.stdout or "").count('provider="cuda"') < 2:
        result.update(
            {
                "ok": False,
                "status": "failed",
                "returncode": 0,
                "blockers": ["cuda_provider_not_confirmed_by_upstream_output"],
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
        )
        return result

    intervals = parse_sherpa_diarization_output(completed.stdout or "")
    if not intervals:
        result.update(
            {
                "ok": False,
                "status": "degraded",
                "returncode": 0,
                "blockers": ["no_speaker_intervals_emitted"],
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
        )
        return result

    transcript_path = Path(str((plan.get("transcript") or {}).get("path") or "")).resolve()
    cues = parse_transcript(transcript_path)
    assignment = assign_speaker_intervals(
        cues,
        intervals,
        min_coverage_ratio=min_coverage_ratio,
        min_dominance_ratio=min_dominance_ratio,
    )
    evidence_path = Path(str(outputs["evidence"])).resolve()
    candidate_path = Path(str(outputs["candidate_transcript"])).resolve()
    evidence = {
        **result,
        "ok": True,
        "status": "needs_human_review",
        "returncode": 0,
        "blockers": [],
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "interval_count": len(intervals),
        "speaker_count": len({row["speaker"] for row in intervals}),
        "intervals": intervals,
        "assignment_summary": assignment["summary"],
        "assignment_policy": assignment["policy"],
        "provenance": {
            "upstream": plan.get("upstream"),
            "assignment_reference": plan.get("assignment_reference"),
            "media": plan.get("media"),
            "transcript": plan.get("transcript"),
            "models": plan.get("models"),
            "runtime": plan.get("runtime"),
            "parameters": plan.get("parameters"),
        },
        "artifacts": {
            "evidence": str(evidence_path),
            "candidate_transcript": str(candidate_path),
        },
        "operator_boundary": {
            **dict(result["operator_boundary"]),
            "models_executed": True,
            "candidate_evidence_only": True,
            "human_confirmation_required_before_promotion": True,
            "primary_transcript_mutated": False,
        },
        "updated_at": now_iso(),
    }
    candidate = {
        "schema": CANDIDATE_TRANSCRIPT_SCHEMA,
        "status": "needs_human_review",
        "source_transcript": plan.get("transcript"),
        "speaker_evidence_path": str(evidence_path),
        "segments": assignment["segments"],
        "summary": assignment["summary"],
        "operator_boundary": evidence["operator_boundary"],
        "updated_at": now_iso(),
    }
    if write:
        write_json(evidence_path, evidence)
        write_json(candidate_path, candidate)
    return evidence


def parse_sherpa_diarization_output(stdout: str) -> list[dict[str, Any]]:
    """Parse only the stable segment lines emitted by the official CLI."""

    intervals: list[dict[str, Any]] = []
    for line_number, line in enumerate(str(stdout or "").splitlines(), start=1):
        match = _SEGMENT_RE.fullmatch(line)
        if not match:
            continue
        start = float(match.group("start"))
        end = float(match.group("end"))
        if end <= start:
            raise ValueError(f"invalid sherpa speaker interval at line {line_number}")
        intervals.append(
            {
                "interval_id": f"speaker-interval-{len(intervals) + 1:06d}",
                "start": start,
                "end": end,
                "speaker": match.group("speaker").lower(),
                "source_line": line_number,
            }
        )
    intervals.sort(key=lambda row: (row["start"], row["end"], row["speaker"]))
    return intervals


def assign_speaker_intervals(
    cues: Iterable[TranscriptCue],
    intervals: Iterable[Mapping[str, Any]],
    *,
    min_coverage_ratio: float = DEFAULT_MIN_COVERAGE_RATIO,
    min_dominance_ratio: float = DEFAULT_MIN_DOMINANCE_RATIO,
) -> dict[str, Any]:
    """Attach dominant-overlap speaker candidates without changing cue boundaries.

    Intent: turn diarization intervals into usable VKP speaker labels.
    Decision: independently adapt WhisperX's maximum-overlap assignment and
    explicitly disable nearest-speaker filling.
    Reason: temporal overlap is evidence; nearest filling can silently label
    silence or gaps as speech from the wrong person.
    Evidence: WhisperX commit ``5f2f9d4`` assigns the speaker with the greatest
    intersection duration and exposes nearest fill as an optional behavior.
    Effective scope: candidate transcript only. Original text, IDs, ordering,
    timestamps, existing speaker labels, and source transcript remain intact.
    """

    _validate_ratio(min_coverage_ratio, "min_coverage_ratio")
    _validate_ratio(min_dominance_ratio, "min_dominance_ratio")
    normalized = [_normalise_interval(row) for row in intervals]
    normalized.sort(key=lambda row: (row["start"], row["end"], row["speaker"]))
    starts = [row["start"] for row in normalized]
    candidate_segments: list[dict[str, Any]] = []
    assigned_count = 0
    ambiguous_count = 0
    uncovered_count = 0
    existing_preserved_count = 0
    conflict_count = 0

    for position, cue in enumerate(cues, start=1):
        start = float(cue.start)
        end = float(cue.end)
        duration = max(0.0, end - start)
        right = bisect.bisect_left(starts, end)
        overlaps: dict[str, float] = {}
        evidence_ids: list[str] = []
        for interval in normalized[:right]:
            intersection = min(end, interval["end"]) - max(start, interval["start"])
            if intersection <= 0:
                continue
            speaker = interval["speaker"]
            overlaps[speaker] = overlaps.get(speaker, 0.0) + intersection
            evidence_ids.append(interval["interval_id"])
        total_overlap = sum(overlaps.values())
        dominant_speaker = ""
        dominant_overlap = 0.0
        if overlaps:
            dominant_speaker, dominant_overlap = max(
                overlaps.items(),
                key=lambda item: (item[1], item[0]),
            )
        coverage_ratio = min(1.0, total_overlap / duration) if duration > 0 else 0.0
        dominance_ratio = dominant_overlap / total_overlap if total_overlap > 0 else 0.0
        eligible = (
            bool(dominant_speaker)
            and coverage_ratio >= min_coverage_ratio
            and dominance_ratio >= min_dominance_ratio
        )
        existing_speaker = str(cue.speaker or "").strip()
        if existing_speaker:
            assigned_speaker = existing_speaker
            existing_preserved_count += 1
            if dominant_speaker and dominant_speaker.casefold() != existing_speaker.casefold():
                state = "existing_conflicts_with_candidate"
                conflict_count += 1
            else:
                state = "existing_preserved"
        elif eligible:
            assigned_speaker = dominant_speaker
            state = "candidate_assigned"
            assigned_count += 1
        elif overlaps:
            assigned_speaker = ""
            state = "ambiguous_overlap"
            ambiguous_count += 1
        else:
            assigned_speaker = ""
            state = "uncovered"
            uncovered_count += 1

        metadata = dict(cue.metadata)
        metadata["speaker_diarization_candidate"] = {
            "state": state,
            "speaker": dominant_speaker,
            "coverage_ratio": round(coverage_ratio, 6),
            "dominance_ratio": round(dominance_ratio, 6),
            "overlap_seconds_by_speaker": {
                key: round(value, 6) for key, value in sorted(overlaps.items())
            },
            "evidence_ids": evidence_ids,
            "nearest_fill_used": False,
        }
        candidate_segments.append(
            {
                "id": cue.segment_id or f"segment-{position:06d}",
                "start": start,
                "end": end,
                "text": cue.text,
                "source_segment_ids": list(cue.source_segment_ids),
                "transformations": list(cue.transformations),
                "speaker": assigned_speaker,
                "speaker_role": cue.speaker_role,
                "metadata": metadata,
            }
        )

    total = len(candidate_segments)
    return {
        "segments": candidate_segments,
        "summary": {
            "segment_count": total,
            "candidate_assigned_count": assigned_count,
            "existing_preserved_count": existing_preserved_count,
            "existing_conflict_count": conflict_count,
            "ambiguous_count": ambiguous_count,
            "uncovered_count": uncovered_count,
            "fully_labeled": total > 0
            and assigned_count + existing_preserved_count == total
            and conflict_count == 0,
        },
        "policy": {
            "method": "maximum_overlap_per_transcript_segment",
            "min_coverage_ratio": float(min_coverage_ratio),
            "min_dominance_ratio": float(min_dominance_ratio),
            "nearest_fill_enabled": False,
            "existing_speaker_overwrite_allowed": False,
            "segment_split_or_merge_allowed": False,
        },
    }


def _configured_path(value: str | Path, env_name: str) -> Path | None:
    raw = str(value or os.environ.get(env_name) or "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def _resolve_command(value: str) -> Path | None:
    direct = Path(value).expanduser()
    if direct.is_file():
        return direct.resolve()
    resolved = shutil.which(value)
    return Path(resolved).resolve() if resolved else None


def _runtime_probe(command_path: Path | None) -> dict[str, Any]:
    if command_path is None:
        return {
            "status": "command_not_found",
            "ready": False,
            "blocker": "command_not_found",
        }
    try:
        completed = subprocess.run(
            [str(command_path), "--help"],
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "status": "probe_failed",
            "ready": False,
            "blocker": f"runtime_probe_failed:{type(exc).__name__}",
        }
    combined = f"{completed.stdout}\n{completed.stderr}"
    contract_found = (
        "Offline/Non-streaming speaker diarization" in combined
        or "segmentation.pyannote-model" in combined
    )
    return {
        "status": "ready" if contract_found else "unexpected_cli_contract",
        "ready": bool(contract_found),
        "blocker": "" if contract_found else "unexpected_cli_contract",
        "returncode": int(completed.returncode),
        "official_contract_detected": bool(contract_found),
    }


def _wave_probe(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "media_not_found",
            "ready": False,
            "sample_rate": None,
            "channels": None,
        }
    if path.suffix.lower() != ".wav":
        return {
            "status": "wav_required",
            "ready": False,
            "sample_rate": None,
            "channels": None,
        }
    try:
        with wave.open(str(path), "rb") as source:
            sample_rate = int(source.getframerate())
            channels = int(source.getnchannels())
            frame_count = int(source.getnframes())
    except (OSError, wave.Error):
        return {
            "status": "invalid_wav",
            "ready": False,
            "sample_rate": None,
            "channels": None,
        }
    ready = sample_rate == 16000 and channels == 1
    return {
        "status": "ready" if ready else "requires_16khz_mono_wav",
        "ready": ready,
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": frame_count / sample_rate if sample_rate > 0 else 0.0,
    }


def _plan_blockers(
    *,
    media: Path,
    transcript: Path,
    segmentation: Path | None,
    embedding: Path | None,
    command_path: Path | None,
    runtime_probe: Mapping[str, Any],
    audio_probe: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if command_path is None:
        blockers.append("sherpa_onnx_diarization_command_not_found")
    elif not runtime_probe.get("ready"):
        blockers.append(str(runtime_probe.get("blocker") or "sherpa_runtime_not_ready"))
    if not media.is_file():
        blockers.append("media_not_found")
    elif not audio_probe.get("ready"):
        blockers.append(str(audio_probe.get("status") or "audio_not_ready"))
    if not transcript.is_file():
        blockers.append("transcript_not_found")
    if segmentation is None or not segmentation.is_file():
        blockers.append("segmentation_model_not_found")
    if embedding is None or not embedding.is_file():
        blockers.append("embedding_model_not_found")
    return blockers


def _command_argv(
    command_path: Path | None,
    *,
    media: Path,
    segmentation: Path | None,
    embedding: Path | None,
    provider: str,
    num_speakers: int | None,
    cluster_threshold: float,
    min_duration_on: float,
    min_duration_off: float,
) -> list[str]:
    if command_path is None:
        return []
    clustering = (
        f"--clustering.num-clusters={int(num_speakers)}"
        if num_speakers is not None
        else f"--clustering.cluster-threshold={cluster_threshold:.6g}"
    )
    return [
        str(command_path),
        clustering,
        f"--segmentation.pyannote-model={segmentation or ''}",
        f"--segmentation.provider={provider}",
        f"--embedding.model={embedding or ''}",
        f"--embedding.provider={provider}",
        f"--min-duration-on={min_duration_on:.6g}",
        f"--min-duration-off={min_duration_off:.6g}",
        str(media),
    ]


def _artifact_identity(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": "", "exists": False, "bytes": None, "sha256": ""}
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
        "sha256": sha256_file(path) if exists else "",
    }


def _identity_blockers(plan: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    for key in ("media", "transcript"):
        artifact = _mapping(plan.get(key), key)
        path = Path(str(artifact.get("path") or "")).resolve()
        if not path.is_file():
            blockers.append(f"{key}_not_found")
        elif str(artifact.get("sha256") or "") != sha256_file(path):
            blockers.append(f"{key}_sha256_changed")
    models = _mapping(plan.get("models"), "models")
    for key in ("segmentation", "embedding"):
        artifact = _mapping(models.get(key), key)
        path = Path(str(artifact.get("path") or "")).resolve()
        if not path.is_file():
            blockers.append(f"{key}_model_not_found")
        elif str(artifact.get("sha256") or "") != sha256_file(path):
            blockers.append(f"{key}_model_sha256_changed")
    return blockers


def _normalise_interval(row: Mapping[str, Any]) -> dict[str, Any]:
    start = float(row.get("start") or 0.0)
    end = float(row.get("end") or 0.0)
    speaker = str(row.get("speaker") or "").strip()
    interval_id = str(row.get("interval_id") or "").strip()
    if end <= start:
        raise ValueError("speaker interval end must be greater than start")
    if not speaker:
        raise ValueError("speaker interval must include a speaker")
    return {
        "interval_id": interval_id or f"speaker-interval-{start:.3f}-{end:.3f}",
        "start": start,
        "end": end,
        "speaker": speaker,
    }


def _validate_ratio(value: float, name: str) -> None:
    if not 0 <= float(value) <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def main(argv: list[str] | None = None) -> int:
    """Stable local CLI front door for the sherpa-onnx evidence adapter."""

    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="python -m video_knowledge_pipeline.speaker_diarization_evidence"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser(
        "plan",
        help="Create an exact local diarization plan without model execution or downloads.",
    )
    plan_parser.add_argument("bundle_dir")
    plan_parser.add_argument("media_path", help="Exact 16 kHz mono WAV input")
    plan_parser.add_argument("transcript_path", help="Existing timed transcript to label")
    plan_parser.add_argument("--command-path", default="")
    plan_parser.add_argument("--segmentation-model", default="")
    plan_parser.add_argument("--embedding-model", default="")
    plan_parser.add_argument(
        "--provider",
        choices=("cuda", "cpu", "coreml"),
        default="cuda",
    )
    plan_parser.add_argument("--num-speakers", type=int)
    plan_parser.add_argument("--cluster-threshold", type=float, default=0.90)
    plan_parser.add_argument("--min-duration-on", type=float, default=0.30)
    plan_parser.add_argument("--min-duration-off", type=float, default=0.50)
    plan_parser.add_argument("--no-write", action="store_true")

    run_parser = sub.add_parser(
        "run",
        help="Preview or execute an exact plan; output remains candidate evidence.",
    )
    run_parser.add_argument("plan_json")
    run_parser.add_argument("--execute", action="store_true")
    run_parser.add_argument("--timeout-seconds", type=int, default=3600)
    run_parser.add_argument("--min-coverage-ratio", type=float, default=0.50)
    run_parser.add_argument("--min-dominance-ratio", type=float, default=0.60)
    run_parser.add_argument("--no-write", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "plan":
        result = plan_sherpa_speaker_diarization(
            args.bundle_dir,
            args.media_path,
            args.transcript_path,
            command=args.command_path,
            segmentation_model=args.segmentation_model,
            embedding_model=args.embedding_model,
            provider=args.provider,
            num_speakers=args.num_speakers,
            cluster_threshold=args.cluster_threshold,
            min_duration_on=args.min_duration_on,
            min_duration_off=args.min_duration_off,
            write=not args.no_write,
        )
    else:
        result = run_sherpa_speaker_diarization_plan(
            args.plan_json,
            execute=args.execute,
            timeout_seconds=args.timeout_seconds,
            min_coverage_ratio=args.min_coverage_ratio,
            min_dominance_ratio=args.min_dominance_ratio,
            write=not args.no_write,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if bool(result.get("ok")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
