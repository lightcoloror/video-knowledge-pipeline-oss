from __future__ import annotations

import hashlib
import json
import re
import subprocess
import uuid
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any, Callable, Mapping

from .file_hash import sha256_file
from .media_tools import local_tool_subprocess_env, resolve_media_tool
from .storage import bundle_write_lock, read_json, write_json
from .temporal_frame_groups import plan_temporal_frame_candidates


PANORAMA_PROJECTION_PLAN_SCHEMA = "video_knowledge_pipeline.panorama_projection_plan.v1"
PANORAMA_VIEWPORT_PACK_SCHEMA = "video_knowledge_pipeline.panorama_viewport_pack.v1"
PANORAMA_PROJECTION_RECEIPT_SCHEMA = "video_knowledge_pipeline.panorama_projection_receipt.v1"

MIN_FRAME_COUNT = 5
MAX_FRAME_COUNT = 12
MAX_SEGMENTS = 16
MIN_FACE_SIZE = 64
MAX_FACE_SIZE = 4096

FACE_DIRECTIONS: tuple[tuple[str, float, float], ...] = (
    ("front", 0.0, 0.0),
    ("right", 90.0, 0.0),
    ("back", 180.0, 0.0),
    ("left", -90.0, 0.0),
    ("up", 0.0, 90.0),
    ("down", 0.0, -90.0),
)

CommandRunner = Callable[[list[str]], CompletedProcess[str]]


def plan_panorama_projection(
    bundle_dir: str | Path,
    *,
    profile: Mapping[str, Any],
    indexes: list[int] | None = None,
    frame_count: int = 8,
    limit: int = MAX_SEGMENTS,
    face_size: int = 512,
    window_seconds: float = 4.0,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build a bounded FFmpeg-v360 plan without writing derived files."""

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.is_file() or not timeline_path.is_file():
        raise FileNotFoundError(f"Bundle requires manifest.json and timeline.json: {root}")
    if not bool(profile.get("ready")):
        raise ValueError("panorama profile is not ready")
    frame_count = int(frame_count)
    limit = int(limit)
    face_size = int(face_size)
    if not MIN_FRAME_COUNT <= frame_count <= MAX_FRAME_COUNT:
        raise ValueError(f"frame_count must be between {MIN_FRAME_COUNT} and {MAX_FRAME_COUNT}")
    if not 1 <= limit <= MAX_SEGMENTS:
        raise ValueError(f"limit must be between 1 and {MAX_SEGMENTS} segments")
    if not MIN_FACE_SIZE <= face_size <= MAX_FACE_SIZE:
        raise ValueError(f"face_size must be between {MIN_FACE_SIZE} and {MAX_FACE_SIZE}")

    manifest = read_json(manifest_path)
    timeline_data = read_json(timeline_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain an object")
    if not isinstance(timeline_data, list):
        raise ValueError("timeline.json must contain an array")
    timeline = [dict(row) for row in timeline_data if isinstance(row, dict)]
    routes = {str(row.get("visual_route") or "unknown") for row in timeline}
    candidates = plan_temporal_frame_candidates(
        timeline,
        routes=routes,
        frame_count=frame_count,
        window_seconds=float(window_seconds),
        root=root,
    )
    requested = sorted({int(value) for value in (indexes or []) if int(value) > 0})
    if requested:
        candidates = [row for row in candidates if int(row.get("index") or 0) in set(requested)]
    candidates = candidates[:limit]
    if not candidates:
        raise ValueError("no panorama timeline segments selected")

    source_paths = {str(row.get("video_key") or "") for row in candidates if str(row.get("video_key") or "")}
    if len(source_paths) != 1:
        raise ValueError("one panorama projection plan must reference exactly one source video")
    source = Path(next(iter(source_paths))).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"panorama source video is missing: {source}")
    source_sha256 = sha256_file(source)
    declared_source_sha256 = _declared_source_sha256(manifest)
    if declared_source_sha256 and declared_source_sha256 != source_sha256:
        raise ValueError("manifest source hash does not match panorama source video")

    clean_run_id = _run_id(run_id)
    run_relative = Path("runs") / f"panorama-{clean_run_id}"
    ffmpeg = resolve_media_tool("ffmpeg") or "ffmpeg"
    viewports: list[dict[str, Any]] = []
    selected_indexes: list[int] = []
    duration_by_index = {
        int(row.get("index") or position): _positive_duration(row.get("video_duration_seconds"))
        for position, row in enumerate(timeline, start=1)
    }
    for candidate in candidates:
        timeline_index = int(candidate["index"])
        selected_indexes.append(timeline_index)
        for sample_index, raw_timestamp in enumerate(candidate.get("times") or [], start=1):
            timestamp = _safe_decode_timestamp(
                float(raw_timestamp),
                duration=duration_by_index.get(timeline_index),
            )
            for direction, yaw, pitch in FACE_DIRECTIONS:
                relative_output = (
                    run_relative
                    / "viewports"
                    / f"{timeline_index:04d}"
                    / f"sample-{sample_index:02d}-{int(round(float(timestamp) * 1000)):010d}ms"
                    / f"{direction}.jpg"
                )
                absolute_output = (root / relative_output).resolve()
                filter_value = _v360_filter(
                    profile,
                    yaw=yaw,
                    pitch=pitch,
                    face_size=face_size,
                )
                command = [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{float(timestamp):.3f}",
                    "-i",
                    str(source),
                    "-frames:v",
                    "1",
                    "-vf",
                    filter_value,
                    "-q:v",
                    "2",
                    str(absolute_output),
                ]
                identity = {
                    "timeline_index": timeline_index,
                    "sample_index": sample_index,
                    "timestamp_seconds": round(float(timestamp), 6),
                    "direction": direction,
                    "yaw": yaw,
                    "pitch": pitch,
                    "roll": 0.0,
                    "h_fov": 90.0,
                    "v_fov": 90.0,
                    "path": relative_output.as_posix(),
                    "source_sha256": source_sha256,
                }
                viewports.append(
                    {
                        **identity,
                        "viewport_id": "vp-" + _sha256_json(identity)[:24],
                        "absolute_output_path": str(absolute_output),
                        "filter": filter_value,
                        "command": command,
                    }
                )

    plan: dict[str, Any] = {
        "schema": PANORAMA_PROJECTION_PLAN_SCHEMA,
        "run_id": clean_run_id,
        "bundle_dir": str(root),
        "profile": dict(profile),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_identity_sha256": panorama_manifest_identity_sha256(
            {**manifest, "panorama_profile": dict(profile)}
        ),
        "timeline_sha256": sha256_file(timeline_path),
        "source_sha256": source_sha256,
        "source_path": str(source),
        "selected_indexes": selected_indexes,
        "segment_count": len(candidates),
        "sample_count": sum(len(row.get("times") or []) for row in candidates),
        "frame_count": frame_count,
        "face_size": face_size,
        "viewport_count": len(viewports),
        "estimated_output_bytes": len(viewports) * face_size * face_size * 3,
        "viewports": viewports,
        "write_performed": False,
    }
    plan["plan_sha256"] = _plan_sha256(plan)
    return plan


def execute_panorama_projection(
    bundle_dir: str | Path,
    plan: Mapping[str, Any],
    *,
    command_runner: CommandRunner | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Execute one validated plan and write a hash-bound immutable viewport pack."""

    root = Path(bundle_dir).expanduser().resolve()
    if str(plan.get("schema") or "") != PANORAMA_PROJECTION_PLAN_SCHEMA:
        raise ValueError("panorama projection plan schema is invalid")
    if str(plan.get("bundle_dir") or "") != str(root):
        raise ValueError("panorama projection plan Bundle path mismatch")
    if str(plan.get("plan_sha256") or "") != _plan_sha256(plan):
        raise ValueError("panorama projection plan hash drift")
    if sha256_file(root / "timeline.json") != str(plan.get("timeline_sha256") or ""):
        raise ValueError("timeline hash drift")
    current_manifest = read_json(root / "manifest.json")
    if not isinstance(current_manifest, dict):
        raise ValueError("manifest contract drift")
    expected_manifest_identity = str(plan.get("manifest_identity_sha256") or "")
    if expected_manifest_identity:
        effective_manifest = dict(current_manifest)
        effective_manifest.setdefault("panorama_profile", dict(plan.get("profile") or {}))
        if panorama_manifest_identity_sha256(effective_manifest) != expected_manifest_identity:
            raise ValueError("manifest identity drift")
    elif sha256_file(root / "manifest.json") != str(plan.get("manifest_sha256") or ""):
        raise ValueError("manifest hash drift")
    source = Path(str(plan.get("source_path") or "")).expanduser().resolve()
    if not source.is_file() or sha256_file(source) != str(plan.get("source_sha256") or ""):
        raise ValueError("source video hash drift")

    runner = command_runner or (lambda command: _run_command(command, timeout_seconds=timeout_seconds))
    run_id = str(plan.get("run_id") or "")
    run_root = (root / "runs" / f"panorama-{run_id}").resolve()
    _assert_inside(root, run_root)
    completed_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with bundle_write_lock(root, operation=f"panorama_projection:{run_id}", timeout_seconds=1.0):
        for row in plan.get("viewports") or []:
            if not isinstance(row, Mapping):
                failures.append({"key": "viewport_invalid", "message": "viewport plan row must be an object"})
                continue
            relative = Path(str(row.get("path") or ""))
            output = (root / relative).resolve()
            _assert_inside(run_root, output)
            if str(row.get("absolute_output_path") or "") != str(output):
                raise ValueError("viewport output path drift")
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [str(value) for value in row.get("command") or []]
            if not command or command[-1] != str(output):
                raise ValueError("viewport command/output mismatch")
            try:
                completed = runner(command)
            except Exception as exc:  # pragma: no cover - platform subprocess failure
                failures.append(
                    {
                        "viewport_id": str(row.get("viewport_id") or ""),
                        "key": "projection_command_failed",
                        "message": str(exc)[:500],
                    }
                )
                continue
            if completed.returncode != 0 or not output.is_file():
                failures.append(
                    {
                        "viewport_id": str(row.get("viewport_id") or ""),
                        "key": "projection_output_missing",
                        "message": str(completed.stderr or "FFmpeg did not create the viewport")[:500],
                    }
                )
                continue
            public_row = {
                key: row[key]
                for key in (
                    "viewport_id",
                    "timeline_index",
                    "sample_index",
                    "timestamp_seconds",
                    "direction",
                    "yaw",
                    "pitch",
                    "roll",
                    "h_fov",
                    "v_fov",
                    "path",
                    "source_sha256",
                    "filter",
                )
            }
            public_row["sha256"] = sha256_file(output)
            public_row["bytes"] = output.stat().st_size
            completed_rows.append(public_row)

        receipt_path = run_root / "projection-receipt.json"
        if failures or len(completed_rows) != int(plan.get("viewport_count") or 0):
            receipt = {
                "schema": PANORAMA_PROJECTION_RECEIPT_SCHEMA,
                "status": "failed",
                "run_id": run_id,
                "plan_sha256": str(plan.get("plan_sha256") or ""),
                "completed": len(completed_rows),
                "expected": int(plan.get("viewport_count") or 0),
                "failures": failures,
                "pack_written": False,
            }
            write_json(receipt_path, receipt)
            return {
                "status": "failed",
                "run_id": run_id,
                "receipt_path": str(receipt_path),
                "failures": failures,
                "pack": None,
                "pack_path": "",
            }

        coverage_identity = [
            {
                key: row[key]
                for key in (
                    "viewport_id",
                    "timeline_index",
                    "sample_index",
                    "timestamp_seconds",
                    "direction",
                    "sha256",
                )
            }
            for row in completed_rows
        ]
        pack: dict[str, Any] = {
            "schema": PANORAMA_VIEWPORT_PACK_SCHEMA,
            "run_id": run_id,
            "profile": dict(plan.get("profile") or {}),
            "plan_sha256": str(plan.get("plan_sha256") or ""),
            "manifest_sha256": str(plan.get("manifest_sha256") or ""),
            "timeline_sha256": str(plan.get("timeline_sha256") or ""),
            "source_sha256": str(plan.get("source_sha256") or ""),
            "selected_indexes": list(plan.get("selected_indexes") or []),
            "segment_count": int(plan.get("segment_count") or 0),
            "sample_count": int(plan.get("sample_count") or 0),
            "frame_count": int(plan.get("frame_count") or 0),
            "face_size": int(plan.get("face_size") or 0),
            "viewport_count": len(completed_rows),
            "viewports": completed_rows,
            "geometric_coverage_sha256": _sha256_json(coverage_identity),
        }
        pack["task_pack_sha256"] = viewport_pack_sha256(pack)
        pack_path = run_root / "viewport-pack.json"
        write_json(pack_path, pack)
        receipt = {
            "schema": PANORAMA_PROJECTION_RECEIPT_SCHEMA,
            "status": "succeeded",
            "run_id": run_id,
            "plan_sha256": str(plan.get("plan_sha256") or ""),
            "completed": len(completed_rows),
            "expected": len(completed_rows),
            "failures": [],
            "pack_written": True,
            "pack_path": pack_path.relative_to(root).as_posix(),
            "task_pack_sha256": pack["task_pack_sha256"],
        }
        write_json(receipt_path, receipt)

    return {
        "status": "succeeded",
        "run_id": run_id,
        "receipt_path": str(receipt_path),
        "pack_path": str(pack_path),
        "pack": pack,
        "failures": [],
    }


def _v360_filter(profile: Mapping[str, Any], *, yaw: float, pitch: float, face_size: int) -> str:
    projection = str(profile.get("projection") or "")
    inputs = {
        "equirectangular_mono": "e",
        "equirectangular_stereo_sbs": "e",
        "equirectangular_stereo_tb": "e",
        "cubemap_3x2": "c3x2",
        "cubemap_6x1": "c6x1",
        "cubemap_1x6": "c1x6",
        "dual_fisheye_explicit": "dfisheye",
    }
    if projection not in inputs:
        raise ValueError(f"unsupported panorama projection for v360: {projection}")
    parts = [f"input={inputs[projection]}", "output=flat"]
    if projection == "equirectangular_stereo_sbs":
        parts.append("in_stereo=sbs")
    elif projection == "equirectangular_stereo_tb":
        parts.append("in_stereo=tb")
    elif projection == "dual_fisheye_explicit":
        if str(profile.get("lens_order") or "") != "left_right":
            raise ValueError("FFmpeg dual-fisheye projection currently requires lens_order=left_right")
        parts.append(f"ih_fov={_number_text(profile.get('input_fov'))}")
    parts.extend(
        [
            f"yaw={_number_text(yaw)}",
            f"pitch={_number_text(pitch)}",
            "roll=0",
            "h_fov=90",
            "v_fov=90",
            f"w={face_size}",
            f"h={face_size}",
        ]
    )
    return "v360=" + ":".join(parts)


def panorama_manifest_identity_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash only projection-relevant manifest state, excluding lifecycle pointers."""

    profile = manifest.get("panorama_profile") if isinstance(manifest.get("panorama_profile"), Mapping) else {}
    return _sha256_json(
        {
            "schema": manifest.get("schema"),
            "source_sha256": _declared_source_sha256(manifest),
            "panorama_profile": {
                key: profile.get(key)
                for key in (
                    "schema",
                    "projection",
                    "stereo_mode",
                    "input_fov",
                    "lens_order",
                    "projection_source",
                    "profile_sha256",
                    "ready",
                )
            },
        }
    )


def _positive_duration(value: Any) -> float | None:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if duration > 0 else None


def _safe_decode_timestamp(timestamp: float, *, duration: float | None) -> float:
    """Keep the last probe away from EOF where sparse-GOP video may decode no frame."""

    value = max(0.0, float(timestamp))
    if duration is None:
        return round(value, 6)
    return round(min(value, max(0.0, float(duration) - 0.5)), 6)


def _run_command(command: list[str], *, timeout_seconds: int) -> CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(1, int(timeout_seconds)),
        check=False,
        env=local_tool_subprocess_env(),
    )


def _declared_source_sha256(manifest: Mapping[str, Any]) -> str:
    for key in ("source_sha256", "source_media_sha256", "media_sha256"):
        value = str(manifest.get(key) or "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", value):
            return value
    return ""


def _run_id(value: str | None) -> str:
    clean = str(value or uuid.uuid4().hex).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", clean):
        raise ValueError("run_id must contain only letters, numbers, dot, underscore or hyphen")
    return clean


def _plan_sha256(plan: Mapping[str, Any]) -> str:
    identity = {key: value for key, value in plan.items() if key not in {"plan_sha256", "write_performed"}}
    return _sha256_json(identity)


def viewport_pack_sha256(pack: Mapping[str, Any]) -> str:
    """Return the immutable viewport-pack identity used by downstream adapters."""
    identity = {key: value for key, value in pack.items() if key != "task_pack_sha256"}
    return _sha256_json(identity)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assert_inside(root: Path, target: Path) -> None:
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"panorama path escapes allowed root: {target}") from exc


def _number_text(value: Any) -> str:
    number = float(value or 0.0)
    return str(int(number)) if number.is_integer() else f"{number:.6f}".rstrip("0").rstrip(".")
