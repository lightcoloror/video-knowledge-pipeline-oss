from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .file_hash import sha256_file
from .models import now_iso
from .panorama_projection import PANORAMA_VIEWPORT_PACK_SCHEMA, viewport_pack_sha256
from .panorama_session import CANONICAL_DIRECTIONS
from .storage import bundle_write_lock, read_json, write_json
from .temporal_frame_preprocess import prepare_temporal_image_probe


PANORAMA_ANALYSIS_PLAN_SCHEMA = "video_knowledge_pipeline.panorama_analysis_plan.v1"
MAX_SEGMENTS = 16
MAX_TEMPORAL_DIRECTIONS = 3
MIN_TEMPORAL_FRAMES = 5
MAX_TEMPORAL_FRAMES = 12


def plan_panorama_analysis(
    bundle_dir: str | Path,
    *,
    viewport_pack_json: str | Path,
    limit: int = MAX_SEGMENTS,
    temporal_direction_limit: int = MAX_TEMPORAL_DIRECTIONS,
    local_vlm_preflight: Mapping[str, Any] | None = None,
    provider_preflight: Mapping[str, Any] | None = None,
    interactive_session_available: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Build a bounded six-face/three-direction plan without calling any model."""

    root = Path(bundle_dir).expanduser().resolve()
    if not 1 <= int(limit) <= MAX_SEGMENTS:
        raise ValueError(f"limit must be between 1 and {MAX_SEGMENTS} segments")
    if not 0 <= int(temporal_direction_limit) <= MAX_TEMPORAL_DIRECTIONS:
        raise ValueError(f"temporal_direction_limit must be between 0 and {MAX_TEMPORAL_DIRECTIONS}")
    timeline_path = root / "timeline.json"
    manifest_path = root / "manifest.json"
    timeline = read_json(timeline_path)
    manifest = read_json(manifest_path)
    if not isinstance(timeline, list) or not isinstance(manifest, dict):
        raise ValueError("Bundle manifest/timeline contract is invalid")
    pack_path = _inside_file(root, viewport_pack_json, "viewport pack")
    pack = read_json(pack_path)
    if not isinstance(pack, dict) or pack.get("schema") != PANORAMA_VIEWPORT_PACK_SCHEMA:
        raise ValueError("viewport pack schema is invalid")
    if viewport_pack_sha256(pack) != str(pack.get("task_pack_sha256") or ""):
        raise ValueError("viewport pack hash drift")
    for row in pack.get("viewports") or []:
        if not isinstance(row, Mapping):
            raise ValueError("viewport pack row is invalid")
        path = _inside_file(root, row.get("path"), "viewport frame")
        if sha256_file(path) != str(row.get("sha256") or ""):
            raise ValueError("viewport frame hash drift")

    route = resolve_panorama_vision_route(
        local_vlm_preflight=local_vlm_preflight,
        provider_preflight=provider_preflight,
        interactive_session_available=interactive_session_available,
    )
    selected_indexes = [int(value) for value in pack.get("selected_indexes") or [] if int(value) > 0][: int(limit)]
    by_key: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in pack.get("viewports") or []:
        if not isinstance(row, Mapping):
            continue
        index = int(row.get("timeline_index") or 0)
        direction = str(row.get("direction") or "")
        if index in selected_indexes and direction in CANONICAL_DIRECTIONS:
            by_key.setdefault((index, direction), []).append(dict(row))
    tasks: list[dict[str, Any]] = []
    review_gaps: list[dict[str, Any]] = []
    analysis_root = root / "runs" / f"panorama-analysis-{str(pack.get('run_id') or 'plan')}"
    temporal_required_indexes: list[int] = []
    for index in selected_indexes:
        direction_rows = {
            direction: sorted(by_key.get((index, direction), []), key=lambda row: int(row.get("sample_index") or 0))
            for direction in CANONICAL_DIRECTIONS
        }
        if any(not direction_rows[direction] for direction in CANONICAL_DIRECTIONS):
            review_gaps.append({"index": index, "key": "panorama_six_face_pack_incomplete"})
            continue
        common_samples = set.intersection(
            *[
                {int(row.get("sample_index") or 0) for row in direction_rows[direction]}
                for direction in CANONICAL_DIRECTIONS
            ]
        )
        if not common_samples:
            review_gaps.append({"index": index, "key": "panorama_common_sample_missing"})
            continue
        semantic_sample = sorted(common_samples)[len(common_samples) // 2]
        semantic_rows = [
            next(row for row in direction_rows[direction] if int(row.get("sample_index") or 0) == semantic_sample)
            for direction in CANONICAL_DIRECTIONS
        ]
        tasks.append(
            {
                "task_id": f"panorama-semantic-{index:06d}-{semantic_sample:03d}",
                "task_type": "panorama_six_view_semantic",
                "timeline_index": index,
                "sample_index": semantic_sample,
                "frame_paths": [str((root / str(row["path"])).resolve()) for row in semantic_rows],
                "route": dict(route),
            }
        )
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for direction in CANONICAL_DIRECTIONS:
            rows = direction_rows[direction]
            paths = [str((root / str(row["path"])).resolve()) for row in rows]
            if not MIN_TEMPORAL_FRAMES <= len(paths) <= MAX_TEMPORAL_FRAMES:
                continue
            probe = prepare_temporal_image_probe(
                paths,
                output_dir=analysis_root / f"{index:04d}" / direction,
                max_images=MAX_TEMPORAL_FRAMES,
                representative_limit=2,
                role=f"panorama_{direction}_temporal_probe",
                write=write,
            )
            score = _change_score(probe)
            scored.append((score, direction, probe))
        selected_temporal = sorted(scored, key=lambda value: (-value[0], CANONICAL_DIRECTIONS.index(value[1])))[: int(temporal_direction_limit)]
        if selected_temporal:
            temporal_required_indexes.append(index)
        for score, direction, probe in selected_temporal:
            tasks.append(
                {
                    "task_id": f"panorama-temporal-{index:06d}-{direction}",
                    "task_type": "panorama_temporal_direction",
                    "timeline_index": index,
                    "direction": direction,
                    "change_score": score,
                    "frame_paths": list(probe.get("source_image_paths") or []),
                    "representative_frame_paths": list(probe.get("prepared_image_paths") or []),
                    "probe": {
                        "schema": probe.get("schema"),
                        "status": probe.get("status"),
                        "frame_manifest": probe.get("frame_manifest") or [],
                        "frame_mapping": probe.get("frame_mapping") or [],
                        "near_duplicate_cluster_count": probe.get("near_duplicate_cluster_count"),
                        "implementation": probe.get("implementation") or {},
                    },
                    "route": dict(route),
                }
            )
        if route["status"] == "unavailable":
            review_gaps.append({"index": index, "key": "panorama_visual_backend_unavailable"})

    result: dict[str, Any] = {
        "schema": PANORAMA_ANALYSIS_PLAN_SCHEMA,
        "status": "review_gap" if review_gaps else "ready",
        "bundle_dir": str(root),
        "viewport_pack_path": str(pack_path),
        "viewport_pack_sha256": str(pack.get("task_pack_sha256") or ""),
        "timeline_sha256": sha256_file(timeline_path),
        "selected_indexes": selected_indexes,
        "temporal_required_indexes": temporal_required_indexes,
        "semantic_task_count": sum(row["task_type"] == "panorama_six_view_semantic" for row in tasks),
        "temporal_task_count": sum(row["task_type"] == "panorama_temporal_direction" for row in tasks),
        "tasks": tasks,
        "review_gaps": review_gaps,
        "remote_calls_made": 0,
        "model_calls_made": 0,
        "write_performed": bool(write),
        "created_at": now_iso(),
    }
    result["plan_sha256"] = _sha256_json({key: value for key, value in result.items() if key not in {"plan_sha256", "created_at"}})
    if write:
        plan_path = analysis_root / "analysis-plan.json"
        with bundle_write_lock(root, operation="panorama_analysis_plan", timeout_seconds=1.0):
            write_json(plan_path, result)
            manifest = read_json(manifest_path)
            if not isinstance(manifest, dict):
                raise ValueError("manifest changed during panorama analysis planning")
            manifest["panorama_analysis_plan"] = {
                "schema": PANORAMA_ANALYSIS_PLAN_SCHEMA,
                "selected_indexes": selected_indexes,
                "temporal_required_indexes": temporal_required_indexes,
                "plan_sha256": result["plan_sha256"],
                "path": plan_path.relative_to(root).as_posix(),
            }
            write_json(manifest_path, manifest)
        result["plan_path"] = str(plan_path)
    else:
        result["plan_path"] = str(analysis_root / "analysis-plan.json")
    return result


def resolve_panorama_vision_route(
    *,
    cache_available: bool = False,
    local_vlm_preflight: Mapping[str, Any] | None = None,
    provider_preflight: Mapping[str, Any] | None = None,
    interactive_session_available: bool = True,
) -> dict[str, Any]:
    """Resolve an existing route without making provider requests or inventing consent."""

    if cache_available:
        return {"status": "ready", "route": "validated_cache", "unattended": True}
    local = local_vlm_preflight or {}
    local_provider = local.get("provider") if isinstance(local.get("provider"), Mapping) else {}
    if local.get("ready_to_execute") is True and (
        str(local_provider.get("provider") or "") in {"local_vlm", "lm_studio"}
        or str(local_provider.get("base_url") or "").startswith(("http://127.0.0.1", "http://localhost"))
    ):
        return {"status": "ready", "route": "local_vlm", "unattended": True}
    provider = provider_preflight or {}
    authorization = provider.get("authorization") if isinstance(provider.get("authorization"), Mapping) else {}
    consent = provider.get("consent") if isinstance(provider.get("consent"), Mapping) else {}
    consent_valid = bool(
        provider.get("standing_policy_valid") is True
        or authorization.get("confirmed") is True
        or consent.get("valid") is True
    )
    if provider.get("ready_to_execute") is True and consent_valid:
        return {"status": "ready", "route": "authorized_provider", "unattended": True}
    if interactive_session_available:
        return {"status": "ready", "route": "codex_session", "unattended": False}
    return {
        "status": "unavailable",
        "route": "review_gap",
        "unattended": False,
        "reason": "no validated cache, local VLM, authorized Provider or interactive session",
    }


def _change_score(probe: Mapping[str, Any]) -> float:
    values: list[float] = []
    for row in probe.get("frame_manifest") or []:
        if not isinstance(row, Mapping):
            continue
        rms = row.get("difference_rms_to_previous")
        changed = row.get("changed_ratio_to_previous")
        values.append(float(rms or 0.0) + float(changed or 0.0))
    return round(max(values, default=0.0), 6)


def _inside_file(root: Path, value: Any, label: str) -> Path:
    path = Path(str(value or "")).expanduser()
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{label} must be an existing file inside Bundle")
    return path


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
