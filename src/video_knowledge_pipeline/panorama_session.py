from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .file_hash import sha256_file
from .models import now_iso
from .panorama_projection import (
    FACE_DIRECTIONS,
    PANORAMA_VIEWPORT_PACK_SCHEMA,
    viewport_pack_sha256,
)
from .storage import read_json


PANORAMA_UNDERSTANDING_SCHEMA = "video_knowledge_pipeline.panorama_visual_understanding.v1"
CANONICAL_DIRECTIONS = tuple(row[0] for row in FACE_DIRECTIONS)
_CANONICAL_GEOMETRY = {row[0]: (row[1], row[2]) for row in FACE_DIRECTIONS}
_GEOMETRY_TOLERANCE = 0.001


def prepare_panorama_vision_session(
    bundle_dir: str | Path,
    *,
    viewport_pack_json: str | Path,
    timeline_index: int,
    sample_index: int = 1,
    model_requested: str = "luna",
    quality: str = "low",
    write: bool = True,
) -> dict[str, Any]:
    """Prepare one six-viewport task through the existing Codex session engine."""

    from .codex_vision_session import prepare_codex_vision_session

    return prepare_codex_vision_session(
        bundle_dir,
        kind="semantic",
        indexes=[int(timeline_index)],
        limit=1,
        max_frames=6,
        model_requested=model_requested,
        quality=quality,
        viewport_pack_json=viewport_pack_json,
        viewport_timeline_index=int(timeline_index),
        viewport_sample_index=int(sample_index),
        write=write,
    )


def prepare_panorama_temporal_session(
    bundle_dir: str | Path,
    *,
    analysis_plan_json: str | Path,
    task_id: str,
    model_requested: str = "luna",
    quality: str = "low",
    write: bool = True,
) -> dict[str, Any]:
    """Prepare one directional temporal track through the existing Codex engine."""

    from .codex_vision_session import prepare_codex_vision_session

    return prepare_codex_vision_session(
        bundle_dir,
        kind="temporal",
        limit=1,
        max_frames=12,
        model_requested=model_requested,
        quality=quality,
        panorama_analysis_plan_json=analysis_plan_json,
        panorama_task_id=task_id,
        write=write,
    )


def build_panorama_session_candidate(
    root: Path,
    timeline: list[dict[str, Any]],
    *,
    viewport_pack_json: str | Path,
    timeline_index: int,
    sample_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pack_path = Path(viewport_pack_json).expanduser()
    pack_path = pack_path.resolve() if pack_path.is_absolute() else (root / pack_path).resolve()
    _assert_file_inside(root, pack_path, label="viewport pack")
    pack = read_json(pack_path)
    if not isinstance(pack, dict) or pack.get("schema") != PANORAMA_VIEWPORT_PACK_SCHEMA:
        raise ValueError("viewport pack schema is invalid")
    expected_pack_sha256 = viewport_pack_sha256(pack)
    if str(pack.get("task_pack_sha256") or "") != expected_pack_sha256:
        raise ValueError("viewport pack hash drift")
    if int(timeline_index) < 1 or int(timeline_index) > len(timeline):
        raise ValueError("panorama timeline index is unavailable")
    if int(sample_index) < 1:
        raise ValueError("panorama sample_index must be positive")

    selected = [
        dict(row)
        for row in pack.get("viewports") or []
        if isinstance(row, Mapping)
        and int(row.get("timeline_index") or 0) == int(timeline_index)
        and int(row.get("sample_index") or 0) == int(sample_index)
    ]
    by_direction = {str(row.get("direction") or ""): row for row in selected}
    if len(selected) != 6 or len(by_direction) != 6 or set(by_direction) != set(CANONICAL_DIRECTIONS):
        raise ValueError("panorama task requires the exact six canonical viewports")

    ordered: list[dict[str, Any]] = []
    timestamps: list[float] = []
    frames: list[dict[str, Any]] = []
    for direction in CANONICAL_DIRECTIONS:
        row = by_direction[direction]
        yaw, pitch = _CANONICAL_GEOMETRY[direction]
        if not _near(row.get("yaw"), yaw) or not _near(row.get("pitch"), pitch):
            raise ValueError("panorama viewport canonical geometry mismatch")
        if not _near(row.get("roll"), 0.0) or not _near(row.get("h_fov"), 90.0) or not _near(row.get("v_fov"), 90.0):
            raise ValueError("panorama viewport canonical geometry mismatch")
        relative = Path(str(row.get("path") or ""))
        if relative.is_absolute():
            raise ValueError("panorama viewport paths must be Bundle-relative")
        path = (root / relative).resolve()
        _assert_file_inside(root, path, label="viewport")
        digest = sha256_file(path)
        if digest != str(row.get("sha256") or ""):
            raise ValueError("panorama viewport frame hash drift")
        timestamp = float(row.get("timestamp_seconds") or 0.0)
        timestamps.append(timestamp)
        public = {
            "viewport_id": str(row.get("viewport_id") or ""),
            "direction": direction,
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": 0.0,
            "h_fov": 90.0,
            "v_fov": 90.0,
            "path": str(path),
            "sha256": digest,
        }
        ordered.append(public)
        frames.append(
            {
                "frame_id": public["viewport_id"],
                "timestamp_ms": int(round(timestamp * 1000)),
                "timestamp": f"{timestamp:.3f}s",
                "filename": path.name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    if max(timestamps) - min(timestamps) > 0.001:
        raise ValueError("panorama viewport timestamps must match within 1ms")
    item = timeline[int(timeline_index) - 1]
    existing = item.get("panorama_visual_understanding")
    sessions = existing.get("sessions") if isinstance(existing, dict) else []
    if any(int(row.get("sample_index") or 0) == int(sample_index) for row in sessions or [] if isinstance(row, dict)):
        raise ValueError("panorama sample already has a formal result")
    context = {
        "schema": "video_knowledge_pipeline.panorama_viewport_context.v1",
        "viewport_pack_path": str(pack_path),
        "viewport_pack_file_sha256": sha256_file(pack_path),
        "viewport_pack_sha256": expected_pack_sha256,
        "projection_run_id": str(pack.get("run_id") or ""),
        "profile": {
            "projection": str((pack.get("profile") or {}).get("projection") or "")
            if isinstance(pack.get("profile"), Mapping)
            else ""
        },
        "timeline_index": int(timeline_index),
        "sample_index": int(sample_index),
        "timestamp_seconds": round(timestamps[0], 6),
        "viewports": ordered,
    }
    candidate = {
        "task_id": f"panorama-{int(timeline_index):06d}-{int(sample_index):03d}-{expected_pack_sha256[:12]}",
        "task_type": "panorama_six_view_semantic",
        "kind": "semantic",
        "index": int(timeline_index),
        "start": float(item.get("start") or 0.0),
        "end": float(item.get("end") or 0.0),
        "visual_route": "panorama_six_view",
        "transcript": str(item.get("transcript") or "")[:2000],
        "visual_text": str(item.get("visual_text") or "")[:2000],
        "frame_paths": [row["path"] for row in ordered],
        "frames": frames,
        "expected_frame_ids": [row["viewport_id"] for row in ordered],
        "viewport_context": context,
    }
    return candidate, context


def build_panorama_temporal_session_candidate(
    root: Path,
    timeline: list[dict[str, Any]],
    *,
    analysis_plan_json: str | Path,
    task_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind one planned direction track to exact local viewport frames."""

    plan_path = Path(analysis_plan_json).expanduser()
    plan_path = plan_path.resolve() if plan_path.is_absolute() else (root / plan_path).resolve()
    _assert_file_inside(root, plan_path, label="panorama analysis plan")
    plan = read_json(plan_path)
    if not isinstance(plan, Mapping) or plan.get("schema") != "video_knowledge_pipeline.panorama_analysis_plan.v1":
        raise ValueError("panorama analysis plan schema is invalid")
    tasks = [dict(row) for row in plan.get("tasks") or [] if isinstance(row, Mapping)]
    matches = [row for row in tasks if str(row.get("task_id") or "") == str(task_id or "")]
    if len(matches) != 1 or matches[0].get("task_type") != "panorama_temporal_direction":
        raise ValueError("panorama temporal task is missing or ambiguous")
    task = matches[0]
    index = int(task.get("timeline_index") or 0)
    if not 1 <= index <= len(timeline):
        raise ValueError("panorama temporal timeline index is unavailable")
    direction = str(task.get("direction") or "")
    if direction not in CANONICAL_DIRECTIONS:
        raise ValueError("panorama temporal direction is invalid")
    paths: list[Path] = []
    for value in task.get("frame_paths") or []:
        path = Path(str(value or "")).expanduser()
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        _assert_file_inside(root, path, label="panorama temporal frame")
        if path in paths:
            raise ValueError("panorama temporal frame set contains duplicates")
        paths.append(path)
    if not 5 <= len(paths) <= 12:
        raise ValueError("panorama temporal tasks require 5-12 ordered frames")
    item = timeline[index - 1]
    existing = item.get("panorama_temporal_understanding")
    tracks = existing.get("tracks") if isinstance(existing, Mapping) else []
    if any(str(row.get("direction") or "") == direction for row in tracks or [] if isinstance(row, Mapping)):
        raise ValueError("panorama temporal direction already has a formal result")
    start = float(item.get("start") or 0.0)
    end = float(item.get("end") or start)
    step = (end - start) / max(1, len(paths) - 1)
    frames = [
        {
            "frame_id": f"{task_id}-frame-{position:02d}",
            "timestamp_ms": int(round((start + (position - 1) * step) * 1000)),
            "timestamp": f"{start + (position - 1) * step:.3f}s",
            "filename": path.name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for position, path in enumerate(paths, start=1)
    ]
    pack_path = Path(str(plan.get("viewport_pack_path") or "")).expanduser()
    pack_path = pack_path.resolve() if pack_path.is_absolute() else (root / pack_path).resolve()
    _assert_file_inside(root, pack_path, label="viewport pack")
    viewport_pack = read_json(pack_path)
    if not isinstance(viewport_pack, Mapping):
        raise ValueError("viewport pack schema is invalid")
    context = {
        "schema": "video_knowledge_pipeline.panorama_temporal_context.v1",
        "context_type": "temporal_direction",
        "analysis_plan_path": str(plan_path),
        "analysis_plan_file_sha256": sha256_file(plan_path),
        "analysis_plan_sha256": str(plan.get("plan_sha256") or ""),
        "viewport_pack_path": str(pack_path),
        "viewport_pack_file_sha256": sha256_file(pack_path),
        "viewport_pack_sha256": str(plan.get("viewport_pack_sha256") or ""),
        "projection_run_id": str(viewport_pack.get("run_id") or ""),
        "timeline_index": index,
        "task_id": str(task_id),
        "direction": direction,
        "required_directions": sorted(
            {
                str(row.get("direction") or "")
                for row in tasks
                if row.get("task_type") == "panorama_temporal_direction"
                and int(row.get("timeline_index") or 0) == index
                and str(row.get("direction") or "") in CANONICAL_DIRECTIONS
            },
            key=CANONICAL_DIRECTIONS.index,
        ),
        "change_score": float(task.get("change_score") or 0.0),
    }
    candidate = {
        "task_id": str(task_id),
        "task_type": "panorama_temporal_direction",
        "kind": "temporal",
        "index": index,
        "start": start,
        "end": end,
        "visual_route": "panorama_temporal_direction",
        "direction": direction,
        "frame_paths": [str(path) for path in paths],
        "frames": frames,
        "expected_frame_ids": [str(frame["frame_id"]) for frame in frames],
        "viewport_context": context,
    }
    return candidate, context


def panorama_result_template(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    context = task.get("viewport_context") if isinstance(task.get("viewport_context"), Mapping) else {}
    return [
        {
            "viewport_id": str(row.get("viewport_id") or ""),
            "direction": str(row.get("direction") or ""),
            "yaw": float(row.get("yaw") or 0.0),
            "pitch": float(row.get("pitch") or 0.0),
            "roll": float(row.get("roll") or 0.0),
            "h_fov": float(row.get("h_fov") or 0.0),
            "v_fov": float(row.get("v_fov") or 0.0),
            "evidence_frame_path": str(row.get("path") or ""),
            "observed": False,
            "observation": "",
        }
        for row in context.get("viewports") or []
        if isinstance(row, Mapping)
    ]


def validate_panorama_pack(root: Path, pack: Mapping[str, Any]) -> list[dict[str, str]]:
    context = pack.get("viewport_context") if isinstance(pack.get("viewport_context"), Mapping) else None
    if context is None:
        return []
    issues: list[dict[str, str]] = []
    pack_path = Path(str(context.get("viewport_pack_path") or "")).expanduser().resolve()
    if not pack_path.is_relative_to(root) or not pack_path.is_file():
        return [{"key": "viewport_pack_missing_or_external", "message": "viewport pack is unavailable inside Bundle"}]
    if sha256_file(pack_path) != str(context.get("viewport_pack_file_sha256") or ""):
        issues.append({"key": "viewport_pack_file_sha256_mismatch", "message": "viewport pack changed after preparation"})
    current = read_json(pack_path)
    if not isinstance(current, dict) or viewport_pack_sha256(current) != str(context.get("viewport_pack_sha256") or ""):
        issues.append({"key": "viewport_pack_identity_mismatch", "message": "viewport pack identity changed after preparation"})
    if context.get("context_type") == "temporal_direction":
        plan_path = Path(str(context.get("analysis_plan_path") or "")).expanduser().resolve()
        if not plan_path.is_relative_to(root) or not plan_path.is_file():
            issues.append({"key": "panorama_analysis_plan_missing", "message": "panorama analysis plan is unavailable inside Bundle"})
        elif sha256_file(plan_path) != str(context.get("analysis_plan_file_sha256") or ""):
            issues.append({"key": "panorama_analysis_plan_sha256_mismatch", "message": "panorama analysis plan changed after preparation"})
    return issues


def validate_panorama_result_item(row: Mapping[str, Any], task: Mapping[str, Any]) -> list[dict[str, str]]:
    if str(task.get("task_type") or "") != "panorama_six_view_semantic":
        return []
    expected = panorama_result_template(task)
    actual = row.get("panorama_viewport_observations") if isinstance(row.get("panorama_viewport_observations"), list) else []
    issues: list[dict[str, str]] = []
    expected_ids = [str(value["viewport_id"]) for value in expected]
    actual_ids = [str(value.get("viewport_id") or "") for value in actual if isinstance(value, Mapping)]
    if actual_ids != expected_ids or len(actual) != len(expected):
        issues.append({"key": "panorama_viewport_set_mismatch", "message": "result must preserve the exact ordered six-viewport set"})
        return issues
    immutable = ("direction", "yaw", "pitch", "roll", "h_fov", "v_fov", "evidence_frame_path")
    for position, (received, reference) in enumerate(zip(actual, expected, strict=True), start=1):
        if not isinstance(received, Mapping):
            issues.append({"key": "panorama_viewport_observation_invalid", "message": f"viewport observation {position} must be an object"})
            continue
        if any(received.get(key) != reference.get(key) for key in immutable):
            issues.append({"key": "panorama_viewport_geometry_mismatch", "message": f"viewport observation {position} changed immutable geometry or path"})
        if received.get("observed") is not True or not str(received.get("observation") or "").strip():
            issues.append({"key": "panorama_viewport_observation_incomplete", "message": f"viewport observation {position} must be explicitly observed and described"})
    return issues


def apply_panorama_result(
    item: dict[str, Any],
    *,
    row: Mapping[str, Any],
    task: Mapping[str, Any],
    result: Mapping[str, Any],
    normalized_visual: Mapping[str, Any],
) -> dict[str, Any]:
    context = task.get("viewport_context") if isinstance(task.get("viewport_context"), Mapping) else {}
    session = {
        "session_id": str(result.get("session_id") or ""),
        "task_id": str(task.get("task_id") or ""),
        "viewport_pack_sha256": str(context.get("viewport_pack_sha256") or ""),
        "projection_run_id": str(context.get("projection_run_id") or ""),
        "sample_index": int(context.get("sample_index") or 0),
        "timestamp_seconds": float(context.get("timestamp_seconds") or 0.0),
        "evidence_level": str(row.get("evidence_level") or ""),
        "inference_basis": str(row.get("inference_basis") or ""),
        "unavailable_reason": str(row.get("unavailable_reason") or ""),
        "executor": "codex_session",
        "model_requested": str(result.get("model_requested") or ""),
        "model_actual": str(result.get("model_actual") or ""),
        "quality": str(result.get("quality") or ""),
        "viewport_observations": [dict(value) for value in row.get("panorama_viewport_observations") or [] if isinstance(value, Mapping)],
        "aggregate_visual_understanding": dict(normalized_visual),
    }
    existing = item.get("panorama_visual_understanding")
    sessions = [dict(value) for value in existing.get("sessions") or [] if isinstance(value, Mapping)] if isinstance(existing, Mapping) else []
    sessions.append(session)
    payload = {
        "schema": PANORAMA_UNDERSTANDING_SCHEMA,
        "projection": str((context.get("profile") or {}).get("projection") or ""),
        "sessions": sessions,
    }
    item["panorama_visual_understanding"] = payload
    item["panorama_visual_understanding_updated_at"] = now_iso()
    return payload


def apply_panorama_temporal_result(
    item: dict[str, Any],
    *,
    row: Mapping[str, Any],
    task: Mapping[str, Any],
    result: Mapping[str, Any],
    normalized_temporal: Mapping[str, Any],
) -> dict[str, Any]:
    context = task.get("viewport_context") if isinstance(task.get("viewport_context"), Mapping) else {}
    track = {
        "session_id": str(result.get("session_id") or ""),
        "task_id": str(task.get("task_id") or ""),
        "direction": str(context.get("direction") or task.get("direction") or ""),
        "analysis_plan_sha256": str(context.get("analysis_plan_sha256") or ""),
        "viewport_pack_sha256": str(context.get("viewport_pack_sha256") or ""),
        "change_score": float(context.get("change_score") or 0.0),
        "evidence_level": str(row.get("evidence_level") or ""),
        "inference_basis": str(row.get("inference_basis") or ""),
        "unavailable_reason": str(row.get("unavailable_reason") or ""),
        "observed_frame_count": len(row.get("observed_frame_ids") or []),
        "observed_frame_ids": [str(value) for value in row.get("observed_frame_ids") or []],
        "executor": "codex_session",
        "model_requested": str(result.get("model_requested") or ""),
        "model_actual": str(result.get("model_actual") or ""),
        "quality": str(result.get("quality") or ""),
        "temporal_visual_understanding": dict(normalized_temporal),
    }
    existing = item.get("panorama_temporal_understanding")
    tracks = [dict(value) for value in existing.get("tracks") or [] if isinstance(value, Mapping)] if isinstance(existing, Mapping) else []
    tracks.append(track)
    payload = {
        "schema": "video_knowledge_pipeline.panorama_temporal_understanding.v1",
        "required_directions": [str(value) for value in context.get("required_directions") or []],
        "tracks": tracks,
    }
    item["panorama_temporal_understanding"] = payload
    item["panorama_temporal_understanding_updated_at"] = now_iso()
    return payload


def _assert_file_inside(root: Path, path: Path, *, label: str) -> None:
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{label} must be an existing file inside the Bundle")


def _near(value: Any, expected: float) -> bool:
    try:
        return abs(float(value) - float(expected)) <= _GEOMETRY_TOLERANCE
    except (TypeError, ValueError):
        return False
