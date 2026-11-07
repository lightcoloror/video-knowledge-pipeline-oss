from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file
from .models import now_iso
from .speaker_global_alignment import _funasr_spectral_oracle_labels
from .storage import read_json, write_json


SCHEMA = "video_knowledge_pipeline.shared_session_speaker_alignment.v1"
PRIVATE_SCHEMA = "video_knowledge_pipeline.shared_session_speaker_alignment_private.v1"
ROLE_REVIEW_SCHEMA = "video_knowledge_pipeline.speaker_role_review.v1"


def build_shared_session_speaker_alignment(
    private_center_paths: list[str | Path],
    candidate_transcript_paths: list[str | Path],
    output_dir: str | Path,
    *,
    expected_speaker_count: int,
    confirm_shared_participant_set: bool,
    write: bool = True,
    oracle_clusterer: Any = None,
) -> dict[str, Any]:
    """Jointly align adjacent recordings known to share one participant set.

    Intent: preserve one anonymous speaker ID across separately recorded clips.
    Decision: reuse the pinned FunASR spectral oracle-count implementation over
    the already extracted local CAM++ centers from every supplied recording.
    Reason: pairwise voiceprint thresholds must remain conservative for
    unrelated videos, but adjacent clips from one operator-confirmed interview
    can use the stronger shared-participant-set constraint.
    Evidence: both interview files have user-confirmed participant_count=3;
    their conservative pairwise scores stayed below 0.72 and were correctly
    rejected instead of silently lowering the identity threshold.
    Effective scope: new candidate transcripts, a local role-review pack, and
    biometric private centers. Raw ASR, roles, identities, Timeline, summaries,
    and independent-video matching policy remain unchanged.
    """

    if not confirm_shared_participant_set:
        return {
            "schema": SCHEMA,
            "status": "blocked",
            "blockers": ["explicit_shared_participant_set_confirmation_required"],
        }
    if int(expected_speaker_count) < 1:
        raise ValueError("expected_speaker_count must be positive")
    centers_paths = [
        Path(value).expanduser().resolve() for value in private_center_paths
    ]
    transcript_paths = [
        Path(value).expanduser().resolve() for value in candidate_transcript_paths
    ]
    if not centers_paths or len(centers_paths) != len(transcript_paths):
        raise ValueError(
            "one private center sidecar is required per candidate transcript"
        )
    if any(not path.is_file() for path in centers_paths + transcript_paths):
        raise FileNotFoundError("shared-session speaker input is missing")

    center_payloads = [read_json(path) for path in centers_paths]
    candidates = [read_json(path) for path in transcript_paths]
    if any(not isinstance(value, dict) for value in center_payloads + candidates):
        raise ValueError("shared-session speaker inputs must be JSON objects")
    flattened: list[dict[str, Any]] = []
    for recording_index, payload in enumerate(center_payloads):
        assert isinstance(payload, dict)
        if not bool(payload.get("biometric_data")):
            raise ValueError("private center sidecar is not marked as biometric data")
        sample_rows = payload.get("samples") or []
        rows = sample_rows if sample_rows else payload.get("centers") or []
        source_kind = "source_segment_sample" if sample_rows else "local_speaker_center"
        for row in rows:
            if not isinstance(row, dict):
                continue
            flattened.append(
                {
                    "recording_index": recording_index,
                    "chunk_index": int(row.get("chunk_index") or 0),
                    "local_speaker_id": str(row.get("local_speaker_id") or ""),
                    "center": _normalise(row.get("center")),
                    "source_segment_id": str(row.get("source_segment_id") or ""),
                    "duration_seconds": float(row.get("duration_seconds") or 0.0),
                    "source_kind": source_kind,
                }
            )
    if int(expected_speaker_count) > len(flattened):
        raise ValueError("expected_speaker_count exceeds available centers")
    runner = oracle_clusterer or _funasr_spectral_oracle_labels
    labels = [
        int(value)
        for value in runner(
            [row["center"] for row in flattened], int(expected_speaker_count)
        )
    ]
    if len(labels) != len(flattened):
        raise RuntimeError("shared-session cluster label count mismatch")
    unique = sorted(set(labels), key=lambda label: labels.index(label))
    if len(unique) != int(expected_speaker_count):
        raise RuntimeError("shared-session clustering did not produce expected count")
    canonical = {label: index for index, label in enumerate(unique)}
    assignments: dict[tuple[int, int, str], str] = {}
    assignment_confidence: dict[tuple[int, int, str], float] = {}
    sample_assignments: dict[tuple[int, str], str] = {}
    mappings: list[dict[str, Any]] = []
    grouped_centers: dict[int, list[list[float]]] = {}
    votes: dict[tuple[int, int, str], dict[int, float]] = {}
    sample_counts: dict[tuple[int, int, str], int] = {}
    for row, raw_label in zip(flattened, labels, strict=True):
        label = canonical[raw_label]
        global_id = f"speaker-global-{label + 1:03d}"
        key = (
            int(row["recording_index"]),
            int(row["chunk_index"]),
            str(row["local_speaker_id"]),
        )
        grouped_centers.setdefault(label, []).append(row["center"])
        weight = max(float(row.get("duration_seconds") or 0.0), 0.5)
        distribution = votes.setdefault(key, {})
        distribution[label] = distribution.get(label, 0.0) + weight
        sample_counts[key] = sample_counts.get(key, 0) + 1
        source_segment_id = str(row.get("source_segment_id") or "")
        if source_segment_id:
            sample_assignments[(int(row["recording_index"]), source_segment_id)] = (
                global_id
            )

    for key in sorted(votes):
        distribution = votes[key]
        winner = max(distribution, key=distribution.get)
        total = sum(distribution.values())
        purity = float(distribution[winner] / total) if total else 0.0
        global_id = f"speaker-global-{winner + 1:03d}"
        assignments[key] = global_id
        assignment_confidence[key] = purity
        mappings.append(
            {
                "recording_index": key[0],
                "chunk_index": key[1],
                "local_speaker_id": key[2],
                "shared_global_speaker_id": global_id,
                "status": "candidate" if purity >= 0.67 else "needs_human_review",
                "method": "funasr_spectral_sample_oracle_count_majority_vote",
                "sample_count": sample_counts[key],
                "sample_purity": round(purity, 6),
            }
        )

    root = Path(output_dir).expanduser().resolve()
    public_path = root / "shared-session-speaker-alignment.json"
    private_path = root / "shared-session-speaker-alignment.private.json"
    review_path = root / "speaker-role-review.local.json"
    output_candidates: list[dict[str, Any]] = []
    role_samples: dict[str, list[dict[str, Any]]] = {
        f"speaker-global-{index + 1:03d}": []
        for index in range(int(expected_speaker_count))
    }
    for recording_index, payload in enumerate(candidates):
        assert isinstance(payload, dict)
        derived = _copy(payload)
        for chunk in derived.get("chunk_results") or []:
            if not isinstance(chunk, dict):
                continue
            chunk_index = int(chunk.get("chunk_index") or 0)
            for sentence_index, sentence in enumerate(chunk.get("sentence_info") or []):
                if not isinstance(sentence, dict):
                    continue
                local = str(
                    sentence.get("speaker_local_cluster")
                    if sentence.get("speaker_local_cluster") is not None
                    else sentence.get("spk")
                    if sentence.get("spk") is not None
                    else ""
                )
                source_segment_id = (
                    f"chunk-{chunk_index:04d}-sentence-{sentence_index:05d}"
                )
                global_id = sample_assignments.get(
                    (recording_index, source_segment_id),
                    assignments.get((recording_index, chunk_index, local), ""),
                )
                if not global_id:
                    continue
                previous = str(sentence.get("speaker_global_id") or "")
                if previous:
                    sentence["recording_local_speaker_global_id"] = previous
                sentence["speaker_global_id"] = global_id
                exact_sample = (
                    recording_index,
                    source_segment_id,
                ) in sample_assignments
                sentence["speaker_global_assignment"] = {
                    "method": (
                        "exact_source_segment_campplus_sample"
                        if exact_sample
                        else "local_cluster_sample_majority"
                    ),
                    "confidence": (
                        1.0
                        if exact_sample
                        else round(
                            assignment_confidence.get(
                                (recording_index, chunk_index, local), 0.0
                            ),
                            6,
                        )
                    ),
                }
                samples = role_samples[global_id]
                text = str(sentence.get("text") or "").strip()
                if text:
                    samples.append(
                        {
                            "recording_index": recording_index,
                            "start_ms": float(sentence.get("start") or 0.0),
                            "end_ms": float(sentence.get("end") or 0.0),
                            "text": text,
                        }
                    )
        derived["schema"] = (
            "video_knowledge_pipeline.shared_session_speaker_candidate.v1"
        )
        derived["candidate_only"] = True
        derived["status"] = "needs_human_role_review"
        derived["shared_session_alignment_path"] = str(public_path)
        derived["source_candidate_path"] = str(transcript_paths[recording_index])
        derived["source_candidate_sha256"] = sha256_file(
            transcript_paths[recording_index]
        )
        destination = (
            root / f"recording-{recording_index + 1:02d}-shared-speakers.candidate.json"
        )
        if write:
            root.mkdir(parents=True, exist_ok=True)
            write_json(destination, derived)
        output_candidates.append(
            {
                "recording_index": recording_index,
                "path": str(destination),
                "sha256": sha256_file(destination) if write else "",
            }
        )

    source_revision = canonical_json_sha256(
        {
            "private_centers": [sha256_file(path) for path in centers_paths],
            "candidate_transcripts": [sha256_file(path) for path in transcript_paths],
            "expected_speaker_count": int(expected_speaker_count),
        }
    )
    public = {
        "schema": SCHEMA,
        "status": "needs_human_role_review",
        "candidate_only": True,
        "shared_participant_set_confirmed": True,
        "expected_speaker_count": int(expected_speaker_count),
        "recording_count": len(candidates),
        "local_center_count": sum(
            len(payload.get("centers") or []) for payload in center_payloads
        ),
        "sample_embedding_count": len(flattened),
        "global_speaker_count": len(unique),
        "source_revision": source_revision,
        "mappings": mappings,
        "candidate_transcripts": output_candidates,
        "role_review_path": str(review_path),
        "privacy": {
            "anonymous_ids_only": True,
            "person_identity_inferred": False,
            "role_inferred": False,
            "embedding_vectors_in_public_artifact": False,
        },
        "upstream": {
            "project": "modelscope/FunASR",
            "version": "1.3.30",
            "commit": "16cd165ac3946cc8c08bf845331f91fefec8e1a9",
            "entrypoint": "funasr.models.campplus.cluster_backend.SpectralCluster",
            "embedding_granularity": (
                "per_source_segment_sample"
                if any(payload.get("samples") for payload in center_payloads)
                else "local_speaker_center"
            ),
        },
        "updated_at": now_iso(),
    }
    private = {
        "schema": PRIVATE_SCHEMA,
        "status": "completed",
        "biometric_data": True,
        "must_remain_local": True,
        "must_not_be_committed": True,
        "source_revision": source_revision,
        "global_centers": [
            {
                "global_speaker_id": f"speaker-global-{index + 1:03d}",
                "center": _mean_center(grouped_centers[index]),
                "local_center_count": len(grouped_centers[index]),
            }
            for index in range(int(expected_speaker_count))
        ],
        "updated_at": now_iso(),
    }
    role_review = {
        "schema": ROLE_REVIEW_SCHEMA,
        "status": "needs_human_review",
        "source_revision": source_revision,
        "roles_allowed": ["采访者", "客户", "家属"],
        "assignments": [
            {
                "speaker_global_id": global_id,
                "role_label": "",
                "role_status": "unconfirmed",
                "samples": _representative_samples(samples),
            }
            for global_id, samples in role_samples.items()
        ],
        "instructions": "试听或阅读样本后，为三个匿名 ID 分别绑定采访者、客户、家属；系统不得自动确认。",
        "updated_at": now_iso(),
    }
    if write:
        root.mkdir(parents=True, exist_ok=True)
        write_json(private_path, private)
        write_json(review_path, role_review)
        public["private_alignment_path"] = str(private_path)
        public["private_alignment_sha256"] = sha256_file(private_path)
        write_json(public_path, public)
    return public


def _normalise(value: Any) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError("speaker center must be a non-empty list")
    vector = [float(item) for item in value]
    if any(not math.isfinite(item) for item in vector):
        raise ValueError("speaker center contains non-finite values")
    magnitude = math.sqrt(sum(item * item for item in vector))
    if magnitude <= 0:
        raise ValueError("speaker center has zero magnitude")
    return [item / magnitude for item in vector]


def _mean_center(values: list[list[float]]) -> list[float]:
    if not values:
        raise ValueError("shared speaker center group is empty")
    return _normalise(
        [
            sum(row[index] for row in values) / len(values)
            for index in range(len(values[0]))
        ]
    )


def _representative_samples(
    samples: list[dict[str, Any]], *, per_recording: int = 4
) -> list[dict[str, Any]]:
    """Keep long examples from every recording instead of first-arrival bias."""

    by_recording: dict[int, list[dict[str, Any]]] = {}
    for row in samples:
        by_recording.setdefault(int(row.get("recording_index") or 0), []).append(row)
    selected = [
        row
        for recording in sorted(by_recording)
        for row in sorted(
            by_recording[recording],
            key=lambda value: (
                float(value.get("end_ms") or 0.0) - float(value.get("start_ms") or 0.0)
            ),
            reverse=True,
        )[: max(int(per_recording), 1)]
    ]
    return sorted(
        selected,
        key=lambda value: (
            int(value.get("recording_index") or 0),
            float(value.get("start_ms") or 0.0),
        ),
    )


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))
