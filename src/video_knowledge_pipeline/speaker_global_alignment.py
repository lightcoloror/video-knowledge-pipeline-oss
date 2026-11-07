from __future__ import annotations

import math
import os
import shutil
import subprocess
import uuid
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .asr_local_agreement import measure_local_agreement
from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file
from .models import now_iso
from .media_tools import local_tool_subprocess_env
from .asr_runner import _resolve_python_executable
from .storage import read_json, write_json


ALIGNMENT_SCHEMA = "video_knowledge_pipeline.speaker_global_alignment.v1"
PRIVATE_ALIGNMENT_SCHEMA = (
    "video_knowledge_pipeline.speaker_global_alignment_private.v1"
)
VOICEPRINT_REGISTRY_SCHEMA = (
    "video_knowledge_pipeline.local_speaker_voiceprint_registry.v1"
)
VOICEPRINT_MATCH_SCHEMA = "video_knowledge_pipeline.speaker_voiceprint_match.v1"
UPSTREAM_PROJECT = "modelscope/FunASR"
UPSTREAM_VERSION = "1.3.30"
UPSTREAM_COMMIT = "16cd165ac3946cc8c08bf845331f91fefec8e1a9"
UPSTREAM_ENTRYPOINT = "funasr.bin.realtime_ws.HybridSpeakerTracker._map_cluster_centers"
DEFAULT_SIMILARITY_THRESHOLD = 0.60
DEFAULT_CROSS_VIDEO_THRESHOLD = 0.72
DEFAULT_OVERLAP_AGREEMENT_THRESHOLD = 0.80


def align_chunk_speaker_records(
    records: list[dict[str, Any]],
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    overlap_agreement_threshold: float = DEFAULT_OVERLAP_AGREEMENT_THRESHOLD,
    max_speakers: int = 15,
    expected_speaker_count: int | None = None,
    oracle_clusterer: Callable[[list[list[float]], int], list[int]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Map chunk-local CAM++ clusters to recording-global anonymous IDs.

    Intent: prevent one real speaker from receiving a new label at every ASR
    chunk boundary.
    Decision: adapt FunASR's tested ``HybridSpeakerTracker`` centroid mapping
    and corroborate adjacent chunks with VKP's existing overlap-agreement
    evidence; do not implement another embedding or clustering model.
    Reason: CAM++ already emits one centroid per corrected local ``spk`` label,
    while the current VKP chunk runner previously discarded those centroids.
    Evidence: pinned FunASR 1.3.30 normalizes centroids, performs cosine matching,
    prevents two active clusters sharing one ID, and incrementally updates each
    center with a bounded weight.
    Effective scope: a derived anonymous speaker sidecar and
    ``speaker_global_id`` fields only. Local ``spk`` values, transcript text,
    speaker roles, identities, media, and provider routing remain unchanged.
    """

    threshold = _probability(similarity_threshold, "similarity_threshold")
    overlap_threshold = _probability(
        overlap_agreement_threshold, "overlap_agreement_threshold"
    )
    if int(max_speakers) < 1:
        raise ValueError("max_speakers must be positive")
    if expected_speaker_count is not None:
        expected_speaker_count = int(expected_speaker_count)
        if expected_speaker_count < 1:
            raise ValueError("expected_speaker_count must be positive")
        if expected_speaker_count > int(max_speakers):
            raise ValueError("expected_speaker_count exceeds max_speakers")

    values = [_deepcopy_json(row) for row in records if isinstance(row, dict)]
    chunks = _chunk_centers(values)
    overlap_anchors = _overlap_anchors(
        values,
        agreement_threshold=overlap_threshold,
    )
    global_centers: list[list[float]] = []
    global_updates: list[int] = []
    mapping: dict[tuple[int, str], str] = {}
    mappings: list[dict[str, Any]] = []
    missing_center_chunks: list[int] = []

    # Intent: let an operator-confirmed participant count repair CAM++
    # over-splitting, including two local labels inside the same ASR chunk.
    # Decision: reuse FunASR 1.3.30's SpectralCluster oracle-count path and
    # canonicalise only its anonymous label numbering; never infer a role.
    # Reason: the upstream streaming mapper intentionally prevents two active
    # local clusters sharing one ID, which is correct online but cannot repair
    # within-chunk over-segmentation after the recording is complete.
    # Evidence: pinned ``ClusterBackend.forward`` accepts ``oracle_num`` and
    # delegates to ``SpectralCluster``; the user confirmed three participants.
    # Effective scope: derived candidate ``speaker_global_id`` fields and the
    # local-private centroid sidecar only. Transcript text and identities stay
    # unchanged and require human confirmation before promotion.
    oracle_assignments = _oracle_assignments(
        chunks,
        expected_speaker_count,
        clusterer=oracle_clusterer,
    )

    chunk_indexes = sorted({_chunk_index(row) for row in values})
    for chunk_index in chunk_indexes:
        local_centers = chunks.get(chunk_index, [])
        local_speakers = _chunk_local_speakers(values, chunk_index)
        if local_speakers and not local_centers:
            missing_center_chunks.append(chunk_index)
            continue
        used_ids: set[int] = set()
        for center_row in local_centers:
            local_id = str(center_row["local_speaker_id"])
            center = _normalise_vector(center_row["center"])
            anchor = overlap_anchors.get((chunk_index, local_id))
            best_id: int | None = None
            best_similarity = float("-inf")
            method = "funasr_cosine_center"

            if oracle_assignments is not None:
                best_id = oracle_assignments[(chunk_index, local_id)]
                method = "funasr_spectral_oracle_count"
                if best_id < len(global_centers):
                    best_similarity = _dot(center, global_centers[best_id])
                else:
                    best_similarity = 1.0

            if best_id is None and anchor is not None:
                anchored_global = mapping.get(
                    (int(anchor["previous_chunk_index"]), str(anchor["previous_local_speaker_id"]))
                )
                if anchored_global:
                    candidate = _global_number(anchored_global)
                    if candidate not in used_ids and candidate < len(global_centers):
                        anchored_similarity = _dot(center, global_centers[candidate])
                        if anchored_similarity >= threshold:
                            best_id = candidate
                            best_similarity = anchored_similarity
                            method = "overlap_anchor_plus_fun_asr_cosine"

            if best_id is None and global_centers:
                similarities = [_dot(center, known) for known in global_centers]
                candidates = sorted(
                    range(len(similarities)),
                    key=lambda index: similarities[index],
                    reverse=True,
                )
                for candidate in candidates:
                    if candidate not in used_ids:
                        best_id = candidate
                        best_similarity = similarities[candidate]
                        break

            created = False
            if oracle_assignments is not None:
                if best_id == len(global_centers):
                    global_centers.append(center)
                    global_updates.append(1)
                    created = True
                elif best_id is None or best_id > len(global_centers):
                    raise RuntimeError("oracle speaker labels are not canonical")
            elif best_id is None or best_similarity < threshold:
                if len(global_centers) < int(max_speakers):
                    best_id = len(global_centers)
                    global_centers.append(center)
                    global_updates.append(1)
                    best_similarity = 1.0
                    created = True
                    method = "new_global_center"
                else:
                    mappings.append(
                        {
                            "chunk_index": chunk_index,
                            "local_speaker_id": local_id,
                            "global_speaker_id": "",
                            "status": "unavailable",
                            "method": "max_speakers_reached",
                            "cosine_similarity": None,
                            "overlap_anchor": anchor or {},
                        }
                    )
                    continue

            assert best_id is not None
            if not created:
                count = global_updates[best_id]
                weight = 1.0 / min(count + 1, 20)
                updated = [
                    (1.0 - weight) * old + weight * new
                    for old, new in zip(global_centers[best_id], center, strict=True)
                ]
                global_centers[best_id] = _normalise_vector(updated)
                global_updates[best_id] = count + 1

            global_id = f"speaker-global-{best_id + 1:03d}"
            mapping[(chunk_index, local_id)] = global_id
            if oracle_assignments is None:
                used_ids.add(best_id)
            mappings.append(
                {
                    "chunk_index": chunk_index,
                    "local_speaker_id": local_id,
                    "global_speaker_id": global_id,
                    "status": "candidate",
                    "method": method,
                    "cosine_similarity": round(float(best_similarity), 6),
                    "overlap_anchor": anchor or {},
                }
            )

    mapped = _apply_mapping(values, mapping)
    local_speaker_count = len(
        {
            (chunk, local)
            for chunk, local in mapping
        }
    )
    mapped_sentence_count = sum(
        1
        for row in mapped
        for sentence in row.get("sentence_info") or []
        if isinstance(sentence, dict) and sentence.get("speaker_global_id")
    )
    speaker_sentence_count = sum(
        1
        for row in mapped
        for sentence in row.get("sentence_info") or []
        if isinstance(sentence, dict) and _local_speaker(sentence) != ""
    )
    if not speaker_sentence_count:
        status = "not_applicable"
    elif not global_centers:
        status = "unavailable"
    elif mapped_sentence_count < speaker_sentence_count or missing_center_chunks:
        status = "degraded"
    else:
        status = "candidate"

    source_revision = canonical_json_sha256(
        [
            {
                "chunk_index": _chunk_index(row),
                "record_index": int(row.get("record_index") or 0),
                "sentence_info": row.get("sentence_info") or [],
                "speaker_center_sha256": canonical_json_sha256(
                    row.get("_speaker_embedding_centers") or []
                ),
            }
            for row in values
        ]
    )
    public = {
        "schema": ALIGNMENT_SCHEMA,
        "status": status,
        "candidate_only": True,
        "source_revision": source_revision,
        "similarity_threshold": threshold,
        "overlap_agreement_threshold": overlap_threshold,
        "chunk_count": len(chunk_indexes),
        "chunk_local_speaker_count": local_speaker_count,
        "global_speaker_count": len(global_centers),
        "expected_speaker_count": expected_speaker_count,
        "speaker_sentence_count": speaker_sentence_count,
        "mapped_sentence_count": mapped_sentence_count,
        "missing_center_chunk_indexes": sorted(set(missing_center_chunks)),
        "mappings": mappings,
        "upstream": {
            "project": UPSTREAM_PROJECT,
            "version": UPSTREAM_VERSION,
            "commit": UPSTREAM_COMMIT,
            "entrypoint": UPSTREAM_ENTRYPOINT,
            "reuse_mode": "independent_thin_adaptation_of_tested_mapping_contract",
            "oracle_count_entrypoint": "funasr.models.campplus.cluster_backend.SpectralCluster",
        },
        "overlap_reference": {
            "module": "video_knowledge_pipeline.asr_local_agreement",
            "algorithm_sources": ["WhisperStreaming LocalAgreement", "CrispASR overlap-save"],
            "anchor_count": len(overlap_anchors),
        },
        "privacy": {
            "anonymous_ids_only": True,
            "person_identity_inferred": False,
            "role_inferred": False,
            "embedding_vectors_in_public_artifact": False,
            "cross_video_matching_automatic": False,
        },
        "updated_at": now_iso(),
    }
    private = {
        "schema": PRIVATE_ALIGNMENT_SCHEMA,
        "status": status,
        "biometric_data": True,
        "must_remain_local": True,
        "must_not_be_committed": True,
        "source_revision": source_revision,
        "embedding_model": "FunASR CAM++",
        "embedding_model_upstream": public["upstream"],
        "global_centers": [
            {
                "global_speaker_id": f"speaker-global-{index + 1:03d}",
                "center": center,
                "update_count": global_updates[index],
            }
            for index, center in enumerate(global_centers)
        ],
        "updated_at": now_iso(),
    }
    return mapped, public, private


def write_alignment_artifacts(
    output_path: str | Path,
    public: dict[str, Any],
    private: dict[str, Any],
    *,
    write: bool = True,
) -> dict[str, Any]:
    output = Path(output_path).expanduser().resolve()
    public_path = output.with_name(f"{output.stem}-speaker-global-alignment.json")
    private_path = output.with_name(
        f"{output.stem}-speaker-global-alignment.private.json"
    )
    result = _deepcopy_json(public)
    result["artifacts"] = {
        "public_path": str(public_path),
        "private_path": str(private_path),
        "private_contains_biometric_data": True,
    }
    if write:
        write_json(private_path, private)
        result["artifacts"]["private_sha256"] = sha256_file(private_path)
        write_json(public_path, result)
        result["artifacts"]["public_sha256"] = sha256_file(public_path)
    return result


def build_speaker_global_alignment(
    chunked_output_path: str | Path,
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    overlap_agreement_threshold: float = DEFAULT_OVERLAP_AGREEMENT_THRESHOLD,
    max_speakers: int = 15,
    expected_speaker_count: int | None = None,
    write: bool = True,
) -> dict[str, Any]:
    source = Path(chunked_output_path).expanduser().resolve()
    payload = read_json(source)
    if not isinstance(payload, dict):
        raise ValueError("chunked ASR output must be a JSON object")
    records = payload.get("chunk_results")
    if not isinstance(records, list):
        raise ValueError("chunked ASR output contains no chunk_results")
    mapped, public, private = align_chunk_speaker_records(
        [dict(row) for row in records if isinstance(row, dict)],
        similarity_threshold=similarity_threshold,
        overlap_agreement_threshold=overlap_agreement_threshold,
        max_speakers=max_speakers,
        expected_speaker_count=expected_speaker_count,
    )
    artifact = write_alignment_artifacts(source, public, private, write=write)
    artifact["mapped_chunk_results"] = _public_records(mapped)
    return artifact


def _oracle_assignments(
    chunks: dict[int, list[dict[str, Any]]],
    expected_speaker_count: int | None,
    *,
    clusterer: Callable[[list[list[float]], int], list[int]] | None,
) -> dict[tuple[int, str], int] | None:
    if expected_speaker_count is None:
        return None
    ordered = [
        (chunk_index, str(row["local_speaker_id"]), _normalise_vector(row["center"]))
        for chunk_index in sorted(chunks)
        for row in sorted(
            chunks[chunk_index], key=lambda value: str(value["local_speaker_id"])
        )
    ]
    if expected_speaker_count > len(ordered):
        raise ValueError("expected_speaker_count exceeds available speaker centers")
    runner = clusterer or _funasr_spectral_oracle_labels
    labels = [int(value) for value in runner([row[2] for row in ordered], expected_speaker_count)]
    if len(labels) != len(ordered):
        raise RuntimeError("FunASR oracle cluster label count mismatch")
    unique = sorted(set(labels), key=lambda label: labels.index(label))
    if len(unique) != expected_speaker_count:
        raise RuntimeError("FunASR oracle clustering did not produce expected speaker count")
    canonical = {label: index for index, label in enumerate(unique)}
    return {
        (chunk_index, local_id): canonical[label]
        for (chunk_index, local_id, _center), label in zip(ordered, labels, strict=True)
    }


def _funasr_spectral_oracle_labels(
    centers: list[list[float]], expected_speaker_count: int
) -> list[int]:
    """Call the pinned FunASR spectral implementation without a new clusterer."""

    try:
        from .campplus_oracle_cluster_runner import cluster_campplus_centers

        return cluster_campplus_centers(centers, expected_speaker_count)
    except ModuleNotFoundError:
        return _funasr_spectral_oracle_labels_isolated(
            centers, expected_speaker_count
        )


def _funasr_spectral_oracle_labels_isolated(
    centers: list[list[float]], expected_speaker_count: int
) -> list[int]:
    """Reuse the existing isolated FunASR runtime, not a second install."""

    python = Path(_resolve_python_executable()).expanduser()
    if not python.is_file():
        raise RuntimeError("funasr_cluster_backend_unavailable")
    root = Path(__file__).resolve().parents[2]
    env = local_tool_subprocess_env()
    existing = str(env.get("PYTHONPATH") or "")
    env["PYTHONPATH"] = str(root / "src") + (os.pathsep + existing if existing else "")
    configured_root = str(os.environ.get("VKP_PRIVATE_RUNTIME_DIR") or "").strip()
    candidates = [Path(configured_root).expanduser()] if configured_root else []
    candidates.extend(
        [
            root / ".local" / "private-runtime",
            root.parent / "outputs" / "vkp-private-runtime",
        ]
    )
    runtime_root: Path | None = None
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            runtime_root = candidate.resolve()
            break
        except OSError:
            continue
    if runtime_root is None:
        raise RuntimeError("no writable local-private runtime directory")
    temporary = runtime_root / f"campplus-oracle-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        request = temporary / "request.private.json"
        output = temporary / "result.private.json"
        write_json(
            request,
            {
                "schema": "video_knowledge_pipeline.campplus_oracle_cluster_private.v1",
                "biometric_data": True,
                "expected_speaker_count": expected_speaker_count,
                "centers": centers,
            },
        )
        completed = subprocess.run(
            [
                str(python.resolve()),
                "-m",
                "video_knowledge_pipeline.campplus_oracle_cluster_runner",
                "--request",
                str(request),
                "--output",
                str(output),
            ],
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env=env,
        )
        if completed.returncode != 0 or not output.is_file():
            raise RuntimeError(
                "funasr oracle clustering failed: "
                + (completed.stderr or completed.stdout or "no output")[-2000:]
            )
        payload = read_json(output)
        if not isinstance(payload, dict) or payload.get("status") != "completed":
            raise RuntimeError("funasr oracle clustering returned invalid output")
        return [int(value) for value in payload.get("labels") or []]
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def enroll_local_voiceprints(
    private_alignment_path: str | Path,
    registry_path: str | Path,
    *,
    source_id: str,
    confirm_local_biometric_storage: bool,
    write: bool = True,
) -> dict[str, Any]:
    """Explicitly enroll anonymous local centroids into a deletable registry.

    Intent: enable opt-in cross-video speaker candidates without exporting a
    biometric identifier.
    Decision: store only local anonymous centroids behind an explicit operator
    confirmation and make enrollment idempotent by source revision plus speaker.
    Reason: voice embeddings are biometric data and must not silently enter a
    normal Bundle, Git history, prompt, or online provider request.
    Evidence: the recording-global centers are already the bounded output of the
    pinned FunASR mapping contract.
    Effective scope: the explicitly supplied local registry path only.
    """

    if not confirm_local_biometric_storage:
        return {
            "schema": VOICEPRINT_REGISTRY_SCHEMA,
            "status": "blocked",
            "blockers": ["explicit_local_biometric_storage_confirmation_required"],
        }
    private_path = Path(private_alignment_path).expanduser().resolve()
    registry = Path(registry_path).expanduser().resolve()
    alignment = read_json(private_path)
    _require_private_alignment(alignment)
    current = _read_registry(registry)
    entries = list(current.get("entries") or [])
    keys = {
        (str(row.get("source_revision") or ""), str(row.get("source_global_speaker_id") or ""))
        for row in entries
        if isinstance(row, dict)
    }
    added = 0
    for row in alignment.get("global_centers") or []:
        if not isinstance(row, dict):
            continue
        global_id = str(row.get("global_speaker_id") or "")
        key = (str(alignment.get("source_revision") or ""), global_id)
        if key in keys:
            continue
        identity_seed = {"source_revision": key[0], "global_speaker_id": global_id}
        entries.append(
            {
                "voiceprint_id": f"voiceprint-{canonical_json_sha256(identity_seed)[:16]}",
                "source_id": str(source_id or "").strip(),
                "source_revision": key[0],
                "source_global_speaker_id": global_id,
                "center": _normalise_vector(row.get("center")),
                "role_label": "",
                "role_status": "unconfirmed",
                "created_at": now_iso(),
            }
        )
        keys.add(key)
        added += 1
    result = {
        "schema": VOICEPRINT_REGISTRY_SCHEMA,
        "status": "active",
        "biometric_data": True,
        "must_remain_local": True,
        "must_not_be_committed": True,
        "automatic_identity_assignment": False,
        "entries": entries,
        "updated_at": now_iso(),
    }
    if write:
        write_json(registry, result)
    return {
        "schema": VOICEPRINT_REGISTRY_SCHEMA,
        "status": "active",
        "registry_path": str(registry),
        "entry_count": len(entries),
        "added_count": added,
    }


def match_local_voiceprints(
    private_alignment_path: str | Path,
    registry_path: str | Path,
    *,
    similarity_threshold: float = DEFAULT_CROSS_VIDEO_THRESHOLD,
) -> dict[str, Any]:
    threshold = _probability(similarity_threshold, "similarity_threshold")
    private_path = Path(private_alignment_path).expanduser().resolve()
    registry = Path(registry_path).expanduser().resolve()
    alignment = read_json(private_path)
    _require_private_alignment(alignment)
    current = _read_registry(registry)
    source_revision = str(alignment.get("source_revision") or "")
    entries = [
        row
        for row in current.get("entries") or []
        if isinstance(row, dict)
        and str(row.get("source_revision") or "") != source_revision
    ]
    matches: list[dict[str, Any]] = []
    for source in alignment.get("global_centers") or []:
        if not isinstance(source, dict):
            continue
        center = _normalise_vector(source.get("center"))
        ranked = sorted(
            (
                (_dot(center, _normalise_vector(row.get("center"))), row)
                for row in entries
            ),
            key=lambda value: value[0],
            reverse=True,
        )
        best_similarity, best = ranked[0] if ranked else (float("-inf"), {})
        suspected = bool(best) and best_similarity >= threshold
        matches.append(
            {
                "source_global_speaker_id": str(source.get("global_speaker_id") or ""),
                "status": "suspected_same_speaker" if suspected else "no_match",
                "candidate_voiceprint_id": str(best.get("voiceprint_id") or "") if suspected else "",
                "cosine_similarity": round(float(best_similarity), 6) if best else None,
                "threshold": threshold,
                "identity_confirmed": False,
                "requires_human_confirmation": suspected,
            }
        )
    return {
        "schema": VOICEPRINT_MATCH_SCHEMA,
        "status": "candidate" if any(row["status"] == "suspected_same_speaker" for row in matches) else "no_match",
        "source_revision": source_revision,
        "registry_path": str(registry),
        "matches": matches,
        "privacy": {
            "local_only": True,
            "anonymous_only": True,
            "identity_assignment_automatic": False,
            "embedding_vectors_in_result": False,
        },
    }


def bind_local_voiceprint_role(
    registry_path: str | Path,
    voiceprint_id: str,
    role_label: str,
    *,
    confirm_role_binding: bool,
    write: bool = True,
) -> dict[str, Any]:
    """Bind an operator-confirmed role without inferring a real identity."""

    if not confirm_role_binding:
        return {
            "schema": VOICEPRINT_REGISTRY_SCHEMA,
            "status": "blocked",
            "blockers": ["explicit_role_binding_confirmation_required"],
        }
    label = str(role_label or "").strip()
    if not label:
        raise ValueError("role_label is required")
    registry = Path(registry_path).expanduser().resolve()
    current = _read_registry(registry)
    entries = [row for row in current.get("entries") or [] if isinstance(row, dict)]
    updated = 0
    for row in entries:
        if str(row.get("voiceprint_id") or "") == str(voiceprint_id):
            row["role_label"] = label
            row["role_status"] = "human_confirmed"
            row["role_updated_at"] = now_iso()
            updated += 1
    result = {**current, "entries": entries, "updated_at": now_iso()}
    if write and updated:
        write_json(registry, result)
    return {
        "schema": VOICEPRINT_REGISTRY_SCHEMA,
        "status": "role_bound" if updated else "not_found",
        "registry_path": str(registry),
        "updated_count": updated,
        "role_label": label if updated else "",
        "identity_inferred": False,
    }


def delete_local_voiceprint(
    registry_path: str | Path,
    voiceprint_id: str,
    *,
    confirm_delete: bool,
    write: bool = True,
) -> dict[str, Any]:
    if not confirm_delete:
        return {
            "schema": VOICEPRINT_REGISTRY_SCHEMA,
            "status": "blocked",
            "blockers": ["explicit_voiceprint_delete_confirmation_required"],
        }
    registry = Path(registry_path).expanduser().resolve()
    current = _read_registry(registry)
    original = [row for row in current.get("entries") or [] if isinstance(row, dict)]
    kept = [row for row in original if str(row.get("voiceprint_id") or "") != str(voiceprint_id)]
    result = {**current, "entries": kept, "updated_at": now_iso()}
    if write:
        write_json(registry, result)
    return {
        "schema": VOICEPRINT_REGISTRY_SCHEMA,
        "status": "deleted" if len(kept) < len(original) else "not_found",
        "registry_path": str(registry),
        "deleted_count": len(original) - len(kept),
        "entry_count": len(kept),
    }


def _chunk_centers(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for row in records:
        centers = row.get("_speaker_embedding_centers")
        if not isinstance(centers, list) or not centers:
            continue
        chunk_index = _chunk_index(row)
        parsed: list[dict[str, Any]] = []
        for index, value in enumerate(centers):
            if isinstance(value, dict):
                local_id = str(value.get("local_speaker_id", index))
                center = value.get("center")
            else:
                local_id = str(index)
                center = value
            parsed.append(
                {
                    "local_speaker_id": local_id,
                    "center": _normalise_vector(center),
                }
            )
        result[chunk_index] = parsed
    return result


def _overlap_anchors(
    records: list[dict[str, Any]],
    *,
    agreement_threshold: float,
) -> dict[tuple[int, str], dict[str, Any]]:
    by_chunk = {int(_chunk_index(row)): row for row in records}
    anchors: dict[tuple[int, str], dict[str, Any]] = {}
    for right_index in sorted(by_chunk):
        left_index = right_index - 1
        if left_index not in by_chunk:
            continue
        left = by_chunk[left_index]
        right = by_chunk[right_index]
        start = max(
            float(left.get("chunk_core_end_seconds") or 0.0),
            float(right.get("chunk_offset_seconds") or 0.0),
        )
        end = min(
            float(left.get("chunk_end_seconds") or 0.0),
            float(right.get("chunk_core_start_seconds") or 0.0),
        )
        if end <= start:
            start = float(right.get("chunk_offset_seconds") or 0.0)
            end = min(
                float(left.get("chunk_end_seconds") or 0.0),
                float(right.get("chunk_core_start_seconds") or 0.0),
            )
        if end <= start:
            continue
        left_text = _speaker_text_in_window(left, start, end)
        right_text = _speaker_text_in_window(right, start, end)
        scored: list[tuple[float, str, str, dict[str, Any]]] = []
        for left_speaker, left_value in left_text.items():
            for right_speaker, right_value in right_text.items():
                agreement = measure_local_agreement(left_value, right_value, language="zh")
                score = float(agreement.get("agreement_over_shorter") or 0.0)
                if score >= agreement_threshold:
                    scored.append((score, left_speaker, right_speaker, agreement))
        used_left: set[str] = set()
        used_right: set[str] = set()
        for score, left_speaker, right_speaker, agreement in sorted(scored, reverse=True):
            if left_speaker in used_left or right_speaker in used_right:
                continue
            anchors[(right_index, right_speaker)] = {
                "previous_chunk_index": left_index,
                "previous_local_speaker_id": left_speaker,
                "agreement_over_shorter": round(score, 6),
                "matched_unit_count": int(agreement.get("matched_unit_count") or 0),
                "overlap_start": round(start, 6),
                "overlap_end": round(end, 6),
            }
            used_left.add(left_speaker)
            used_right.add(right_speaker)
    return anchors


def _speaker_text_in_window(
    record: dict[str, Any], start: float, end: float
) -> dict[str, str]:
    rows: dict[str, list[str]] = defaultdict(list)
    for sentence in record.get("sentence_info") or []:
        if not isinstance(sentence, dict):
            continue
        sentence_start = float(sentence.get("start") or 0.0) / 1000.0
        sentence_end = float(sentence.get("end") or sentence_start) / 1000.0
        midpoint = (sentence_start + sentence_end) / 2.0
        local = _local_speaker(sentence)
        if local != "" and start <= midpoint < end:
            rows[local].append(str(sentence.get("text") or ""))
    return {key: "".join(values) for key, values in rows.items() if "".join(values)}


def _apply_mapping(
    records: list[dict[str, Any]], mapping: dict[tuple[int, str], str]
) -> list[dict[str, Any]]:
    mapped = _deepcopy_json(records)
    for record in mapped:
        chunk_index = _chunk_index(record)
        for sentence in record.get("sentence_info") or []:
            if not isinstance(sentence, dict):
                continue
            local = _local_speaker(sentence)
            global_id = mapping.get((chunk_index, local), "")
            if global_id:
                sentence["speaker_local_cluster"] = local
                sentence["speaker_global_id"] = global_id
    return mapped


def _public_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "_speaker_embedding_centers"}
        for row in records
    ]


def _chunk_local_speakers(records: list[dict[str, Any]], chunk_index: int) -> set[str]:
    return {
        _local_speaker(sentence)
        for row in records
        if _chunk_index(row) == chunk_index
        for sentence in row.get("sentence_info") or []
        if isinstance(sentence, dict) and _local_speaker(sentence) != ""
    }


def _local_speaker(value: dict[str, Any]) -> str:
    for key in ("speaker_local_cluster", "spk", "speaker", "speaker_id", "spk_id"):
        candidate = value.get(key)
        if candidate is not None and str(candidate).strip() != "":
            return str(candidate).strip()
    return ""


def _chunk_index(row: dict[str, Any]) -> int:
    try:
        return int(row.get("chunk_index") or 0)
    except (TypeError, ValueError):
        return 0


def _global_number(global_id: str) -> int:
    return int(str(global_id).rsplit("-", 1)[-1]) - 1


def _normalise_vector(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError("speaker center must be an array")
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("speaker center must be one-dimensional")
        value = value[0]
    vector = [float(item) for item in value]
    if not vector:
        raise ValueError("speaker center must not be empty")
    norm = math.sqrt(sum(item * item for item in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("speaker center norm must be positive and finite")
    return [item / norm for item in vector]


def _dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("speaker center dimensions do not match")
    return sum(a * b for a, b in zip(left, right, strict=True))


def _probability(value: float, name: str) -> float:
    result = float(value)
    if not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return result


def _require_private_alignment(value: Any) -> None:
    if not isinstance(value, dict) or value.get("schema") != PRIVATE_ALIGNMENT_SCHEMA:
        raise ValueError("unsupported private speaker alignment artifact")
    if not bool(value.get("biometric_data")) or not bool(value.get("must_remain_local")):
        raise ValueError("private speaker alignment lacks biometric privacy markers")


def _read_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema": VOICEPRINT_REGISTRY_SCHEMA,
            "status": "active",
            "biometric_data": True,
            "must_remain_local": True,
            "must_not_be_committed": True,
            "entries": [],
        }
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema") != VOICEPRINT_REGISTRY_SCHEMA:
        raise ValueError("unsupported local speaker voiceprint registry")
    return value


def _deepcopy_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _deepcopy_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deepcopy_json(item) for item in value]
    if isinstance(value, tuple):
        return [_deepcopy_json(item) for item in value]
    if hasattr(value, "tolist"):
        return _deepcopy_json(value.tolist())
    return value
