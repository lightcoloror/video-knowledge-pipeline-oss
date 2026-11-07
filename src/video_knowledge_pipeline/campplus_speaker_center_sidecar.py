from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file
from .media_tools import local_tool_subprocess_env, resolve_media_tool
from .models import now_iso
from .speaker_global_alignment import (
    align_chunk_speaker_records,
    write_alignment_artifacts,
)
from .storage import read_json, write_json


SIDECAR_SCHEMA = "video_knowledge_pipeline.campplus_speaker_center_sidecar.v1"
CANDIDATE_SCHEMA = "video_knowledge_pipeline.speaker_aligned_asr_candidate.v1"
PRIVATE_CENTERS_SCHEMA = (
    "video_knowledge_pipeline.campplus_local_speaker_centers_private.v1"
)
UPSTREAM_PROJECT = "modelscope/FunASR"
UPSTREAM_VERSION = "1.3.30"
UPSTREAM_COMMIT = "16cd165ac3946cc8c08bf845331f91fefec8e1a9"
UPSTREAM_ENTRYPOINT = "funasr.AutoModel(model=CAM++, device=cuda).generate"


def build_campplus_speaker_center_sidecar(
    chunked_output_path: str | Path,
    media_path: str | Path,
    output_dir: str | Path,
    *,
    expected_speaker_count: int | None = None,
    device: str = "cuda",
    max_evidence_seconds: float = 30.0,
    execute: bool = False,
    write: bool = True,
    clip_builder: Callable[[Path, Path, list[dict[str, Any]]], dict[str, Any]]
    | None = None,
    embedding_extractor: Callable[
        [list[Path], str], tuple[list[list[float]], dict[str, Any]]
    ]
    | None = None,
    oracle_clusterer: Callable[[list[list[float]], int], list[int]] | None = None,
) -> dict[str, Any]:
    """Recover CAM++ centers from an existing timestamped ASR result.

    Intent: repair chunk-local speaker over-splitting without repeating ASR.
    Decision: use FFmpeg only to assemble bounded evidence snippets and invoke
    the pinned FunASR CAM++ embedding entrypoint once for all local clusters.
    Reason: the combined SenseVoice+CAM++ whole-chunk route stalled after ASR
    had already succeeded, while speaker embeddings depend only on audio and
    existing timestamped local labels.
    Evidence: FunASR 1.3.30's official CAM++ test returns ``spk_embedding`` and
    VKP already consumes per-local-speaker centers in its global aligner.
    Effective scope: local candidate speaker sidecars only. Raw ASR, transcript
    text, Timeline, identities, roles, summaries, and provider routes do not
    change; no network or CPU fallback is allowed.
    """

    source = Path(chunked_output_path).expanduser().resolve()
    media = Path(media_path).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"chunked ASR output not found: {source}")
    if not media.is_file():
        raise FileNotFoundError(f"media not found: {media}")
    if str(device or "").strip().lower() != "cuda":
        raise ValueError("CAM++ speaker-center sidecar requires explicit CUDA")
    if max_evidence_seconds <= 0:
        raise ValueError("max_evidence_seconds must be positive")
    if expected_speaker_count is not None and int(expected_speaker_count) < 1:
        raise ValueError("expected_speaker_count must be positive")

    payload = read_json(source)
    if not isinstance(payload, dict):
        raise ValueError("chunked ASR output must be a JSON object")
    records = payload.get("chunk_results")
    if not isinstance(records, list) or not records:
        raise ValueError("chunked ASR output contains no chunk_results")
    rows = [dict(row) for row in records if isinstance(row, dict)]
    duration_seconds = float(payload.get("duration_seconds") or 0.0)
    evidence = _speaker_evidence_plan(
        rows,
        media_duration_seconds=duration_seconds,
        max_evidence_seconds=float(max_evidence_seconds),
    )
    if not evidence:
        raise ValueError("timestamped local speaker evidence is unavailable")
    if expected_speaker_count is not None and int(expected_speaker_count) > len(
        evidence
    ):
        raise ValueError("expected_speaker_count exceeds local speaker evidence count")

    source_sha256 = sha256_file(source)
    media_sha256 = sha256_file(media)
    source_input = str(payload.get("input") or "").strip()
    input_matches_media = not source_input or _same_path(source_input, media)
    if not input_matches_media:
        raise ValueError("media path does not match the chunked ASR source input")

    snippets_dir = root / "speaker-evidence-snippets.private"
    public_path = root / "campplus-speaker-center-sidecar.json"
    private_centers_path = root / "campplus-speaker-centers.private.json"
    candidate_path = root / "speaker-aligned-asr.candidate.json"
    plan = {
        "schema": SIDECAR_SCHEMA,
        "status": "planned",
        "execute": bool(execute),
        "candidate_only": True,
        "source": {
            "path": str(source),
            "sha256": source_sha256,
            "schema": str(payload.get("schema") or ""),
        },
        "media": {
            "path": str(media),
            "sha256": media_sha256,
            "bytes": media.stat().st_size,
            "duration_seconds": duration_seconds,
        },
        "parameters": {
            "device": "cuda",
            "expected_speaker_count": expected_speaker_count,
            "max_evidence_seconds_per_local_speaker": float(max_evidence_seconds),
            "automatic_cpu_fallback": False,
        },
        "local_speaker_evidence_count": len(evidence),
        "evidence": evidence,
        "upstream": {
            "project": UPSTREAM_PROJECT,
            "version": UPSTREAM_VERSION,
            "commit": UPSTREAM_COMMIT,
            "entrypoint": UPSTREAM_ENTRYPOINT,
            "reuse_mode": "official_model_entrypoint_plus_thin_sidecar_adapter",
        },
        "artifacts": {
            "public_sidecar": str(public_path),
            "private_centers": str(private_centers_path),
            "candidate_transcript": str(candidate_path),
            "snippet_directory": str(snippets_dir),
        },
        "privacy": {
            "biometric_embeddings_local_only": True,
            "embedding_vectors_in_public_sidecar": False,
            "speaker_roles_inferred": False,
            "person_identities_inferred": False,
            "explicit_delete_supported": True,
        },
        "operator_boundary": {
            "network_calls": 0,
            "downloads_performed": False,
            "asr_reexecuted": False,
            "raw_asr_mutated": False,
            "timeline_mutated": False,
            "summary_refreshed": False,
            "human_confirmation_required_before_promotion": True,
        },
        "updated_at": now_iso(),
    }
    if not execute:
        if write:
            root.mkdir(parents=True, exist_ok=True)
            write_json(public_path, plan)
        return plan

    root.mkdir(parents=True, exist_ok=True)
    snippets_dir.mkdir(parents=True, exist_ok=True)
    make_clip = clip_builder or _build_ffmpeg_evidence_clip
    clip_paths: list[Path] = []
    sample_evidence: list[dict[str, Any]] = []
    clip_receipts: list[dict[str, Any]] = []
    for item in evidence:
        for sample_index, window in enumerate(item["windows"]):
            clip_path = snippets_dir / (
                f"chunk-{int(item['chunk_index']):04d}-speaker-"
                f"{_safe_id(item['local_speaker_id'])}-sample-{sample_index:03d}.wav"
            )
            receipt = make_clip(media, clip_path, [dict(window)])
            if not clip_path.is_file():
                raise RuntimeError(
                    f"speaker evidence clip was not created: {clip_path}"
                )
            sample = {
                "chunk_index": int(item["chunk_index"]),
                "local_speaker_id": str(item["local_speaker_id"]),
                "source_segment_id": str(window["source_segment_id"]),
                "start": float(window["start"]),
                "end": float(window["end"]),
                "duration_seconds": float(window["duration_seconds"]),
            }
            sample_evidence.append(sample)
            clip_paths.append(clip_path)
            clip_receipts.append(
                {
                    **sample,
                    "path": str(clip_path),
                    "sha256": sha256_file(clip_path),
                    "bytes": clip_path.stat().st_size,
                    "execution": receipt,
                }
            )

    extract = embedding_extractor or _extract_campplus_embeddings
    vectors, model_evidence = extract(clip_paths, "cuda")
    if len(vectors) != len(sample_evidence):
        raise RuntimeError("CAM++ embedding count does not match speaker sample count")
    normalised_samples = [_normalise_vector(vector) for vector in vectors]
    group_vectors: dict[tuple[int, str], list[list[float]]] = {}
    for sample, vector in zip(sample_evidence, normalised_samples, strict=True):
        group_vectors.setdefault(
            (int(sample["chunk_index"]), str(sample["local_speaker_id"])), []
        ).append(vector)
    normalised = [
        _mean_vectors(
            group_vectors[(int(item["chunk_index"]), str(item["local_speaker_id"]))]
        )
        for item in evidence
    ]
    enriched = _attach_centers(rows, evidence, normalised)
    mapped, alignment_public, alignment_private = align_chunk_speaker_records(
        enriched,
        expected_speaker_count=expected_speaker_count,
        oracle_clusterer=oracle_clusterer,
    )
    candidate = {
        **payload,
        "schema": CANDIDATE_SCHEMA,
        "candidate_only": True,
        "status": "needs_human_review",
        "source_schema": str(payload.get("schema") or ""),
        "source_path": str(source),
        "source_sha256": source_sha256,
        "source_media_sha256": media_sha256,
        "speaker_alignment_source": str(public_path),
        "chunk_results": _strip_centers(mapped),
        "updated_at": now_iso(),
    }
    private_centers = {
        "schema": PRIVATE_CENTERS_SCHEMA,
        "status": "completed",
        "biometric_data": True,
        "must_remain_local": True,
        "must_not_be_committed": True,
        "source_sha256": source_sha256,
        "media_sha256": media_sha256,
        "model": model_evidence,
        "centers": [
            {
                "chunk_index": item["chunk_index"],
                "local_speaker_id": item["local_speaker_id"],
                "center": vector,
                "sample_count": len(
                    group_vectors[
                        (int(item["chunk_index"]), str(item["local_speaker_id"]))
                    ]
                ),
            }
            for item, vector in zip(evidence, normalised, strict=True)
        ],
        "samples": [
            {
                **sample,
                "center": vector,
                "evidence_clip_sha256": clip_receipts[index]["sha256"],
            }
            for index, (sample, vector) in enumerate(
                zip(sample_evidence, normalised_samples, strict=True)
            )
        ],
        "updated_at": now_iso(),
    }
    plan.update(
        {
            "status": "needs_human_review",
            "model": model_evidence,
            "clip_count": len(clip_paths),
            "embedding_granularity": "per_source_segment_sample",
            "clip_receipts": clip_receipts,
            "alignment": {
                key: value
                for key, value in alignment_public.items()
                if key not in {"mappings", "updated_at"}
            },
            "updated_at": now_iso(),
        }
    )
    if write:
        write_json(private_centers_path, private_centers)
        write_json(candidate_path, candidate)
        alignment_artifact = write_alignment_artifacts(
            candidate_path, alignment_public, alignment_private, write=True
        )
        plan["artifacts"]["alignment_public"] = alignment_artifact["artifacts"][
            "public_path"
        ]
        plan["artifacts"]["alignment_private"] = alignment_artifact["artifacts"][
            "private_path"
        ]
        plan["artifacts"]["private_centers_sha256"] = sha256_file(private_centers_path)
        plan["artifacts"]["candidate_transcript_sha256"] = sha256_file(candidate_path)
        write_json(public_path, plan)
    return plan


def _speaker_evidence_plan(
    records: list[dict[str, Any]],
    *,
    media_duration_seconds: float,
    max_evidence_seconds: float,
) -> list[dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    for row in sorted(records, key=lambda value: int(value.get("chunk_index") or 0)):
        chunk_index = int(row.get("chunk_index") or 0)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for sentence_index, sentence in enumerate(row.get("sentence_info") or []):
            if not isinstance(sentence, dict):
                continue
            local = str(
                sentence.get("spk") if sentence.get("spk") is not None else ""
            ).strip()
            if not local:
                continue
            start = max(float(sentence.get("start") or 0.0) / 1000.0, 0.0)
            end = min(
                float(sentence.get("end") or 0.0) / 1000.0,
                media_duration_seconds if media_duration_seconds > 0 else float("inf"),
            )
            if end - start < 0.5:
                continue
            grouped.setdefault(local, []).append(
                {
                    "source_segment_id": f"chunk-{chunk_index:04d}-sentence-{sentence_index:05d}",
                    "start": round(start, 6),
                    "end": round(end, 6),
                    "duration_seconds": round(end - start, 6),
                }
            )
        for local in sorted(grouped):
            remaining = float(max_evidence_seconds)
            chosen: list[dict[str, Any]] = []
            for sentence in sorted(
                grouped[local],
                key=lambda value: float(value["duration_seconds"]),
                reverse=True,
            ):
                if remaining < 0.5:
                    break
                duration = min(float(sentence["duration_seconds"]), remaining, 8.0)
                if duration < 0.5:
                    continue
                chosen.append(
                    {
                        **sentence,
                        "end": round(float(sentence["start"]) + duration, 6),
                        "duration_seconds": round(duration, 6),
                    }
                )
                remaining -= duration
            if chosen:
                chosen.sort(key=lambda value: (value["start"], value["end"]))
                planned.append(
                    {
                        "chunk_index": chunk_index,
                        "local_speaker_id": local,
                        "window_count": len(chosen),
                        "evidence_duration_seconds": round(
                            sum(float(value["duration_seconds"]) for value in chosen), 6
                        ),
                        "windows": chosen,
                    }
                )
    return planned


def _build_ffmpeg_evidence_clip(
    media: Path, output: Path, windows: list[dict[str, Any]]
) -> dict[str, Any]:
    ffmpeg = resolve_media_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg_not_available_for_speaker_evidence")
    if not windows:
        raise ValueError("speaker evidence windows are empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    filters: list[str] = []
    if len(windows) == 1:
        sources = ["0:a"]
    else:
        labels = "".join(f"[speaker_source_{index}]" for index in range(len(windows)))
        filters.append(f"[0:a]asplit={len(windows)}{labels}")
        sources = [f"speaker_source_{index}" for index in range(len(windows))]
    for index, (source, window) in enumerate(zip(sources, windows, strict=True)):
        filters.append(
            f"[{source}]atrim=start={float(window['start']):.6f}:end={float(window['end']):.6f},"
            f"asetpts=PTS-STARTPTS[speaker_clip_{index}]"
        )
    clip_labels = "".join(f"[speaker_clip_{index}]" for index in range(len(windows)))
    filters.append(f"{clip_labels}concat=n={len(windows)}:v=0:a=1[speaker_out]")
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(media),
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[speaker_out]",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        env=local_tool_subprocess_env(),
    )
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(
            f"ffmpeg speaker evidence failed: {(completed.stderr or completed.stdout)[-1000:]}"
        )
    return {
        "tool": "VKP shared FFmpeg resolver",
        "returncode": int(completed.returncode),
        "window_count": len(windows),
        "command_sha256": canonical_json_sha256(command),
    }


def _extract_campplus_embeddings(
    clips: list[Path], device: str
) -> tuple[list[list[float]], dict[str, Any]]:
    try:
        import funasr  # type: ignore  # noqa: F401
        import torch  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        return _extract_campplus_embeddings_isolated(clips, device)
    from .campplus_embedding_runner import extract_campplus_embeddings

    return extract_campplus_embeddings(clips, device)


def _extract_campplus_embeddings_isolated(
    clips: list[Path], device: str
) -> tuple[list[list[float]], dict[str, Any]]:
    root = Path(__file__).resolve().parents[2]
    configured = str(os.environ.get("LECTURE_ASR_PYTHON") or "").strip()
    python = (
        Path(configured).expanduser()
        if configured
        else root / ".conda-lecture-asr" / "python.exe"
    )
    if not python.is_file():
        raise RuntimeError("isolated FunASR Python is not ready")
    private_root = clips[0].parent.parent
    request_path = private_root / "campplus-embedding-request.private.json"
    output_path = private_root / "campplus-embedding-result.private.json"
    write_json(
        request_path,
        {
            "schema": "video_knowledge_pipeline.campplus_embedding_batch_private.v1",
            "device": device,
            "clips": [str(path) for path in clips],
        },
    )
    env = local_tool_subprocess_env()
    existing = str(env.get("PYTHONPATH") or "")
    env["PYTHONPATH"] = str(root / "src") + (os.pathsep + existing if existing else "")
    command = [
        str(python.resolve()),
        "-m",
        "video_knowledge_pipeline.campplus_embedding_runner",
        "--request",
        str(request_path),
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        env=env,
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            "isolated CAM++ embedding runner failed: "
            + (completed.stderr or completed.stdout or "no output")[-2000:]
        )
    payload = read_json(output_path)
    if not isinstance(payload, dict) or payload.get("status") != "completed":
        raise RuntimeError("isolated CAM++ embedding result is invalid")
    vectors = [_normalise_vector(value) for value in payload.get("vectors") or []]
    return vectors, dict(payload.get("model") or {})


def _attach_centers(
    records: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    vectors: list[list[float]],
) -> list[dict[str, Any]]:
    centers: dict[int, list[dict[str, Any]]] = {}
    for item, vector in zip(evidence, vectors, strict=True):
        centers.setdefault(int(item["chunk_index"]), []).append(
            {
                "local_speaker_id": str(item["local_speaker_id"]),
                "center": vector,
            }
        )
    enriched: list[dict[str, Any]] = []
    for row in records:
        copied = _json_copy(row)
        copied["_speaker_embedding_centers"] = sorted(
            centers.get(int(row.get("chunk_index") or 0), []),
            key=lambda value: value["local_speaker_id"],
        )
        enriched.append(copied)
    return enriched


def _strip_centers(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in row.items()
            if key != "_speaker_embedding_centers"
        }
        for row in records
    ]


def _normalise_vector(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("speaker embedding center must be a non-empty list")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) for item in result):
        raise ValueError("speaker embedding center contains non-finite values")
    magnitude = math.sqrt(sum(item * item for item in result))
    if magnitude <= 0:
        raise ValueError("speaker embedding center has zero magnitude")
    return [item / magnitude for item in result]


def _mean_vectors(values: list[list[float]]) -> list[float]:
    if not values:
        raise ValueError("speaker sample group is empty")
    width = len(values[0])
    if any(len(value) != width for value in values):
        raise ValueError("speaker sample vectors have inconsistent dimensions")
    return _normalise_vector(
        [sum(value[index] for value in values) / len(values) for index in range(width)]
    )


def _same_path(value: str, path: Path) -> bool:
    try:
        return Path(value).expanduser().resolve() == path
    except OSError:
        return False


def _safe_id(value: Any) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value)
    )
    return cleaned or "unknown"


def _json_copy(value: Any) -> Any:
    import json

    return json.loads(json.dumps(value, ensure_ascii=False))
