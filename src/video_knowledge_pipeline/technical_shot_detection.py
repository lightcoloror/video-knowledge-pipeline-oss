from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .file_hash import sha256_file
from .models import now_iso
from .scene_detection_adapter import run_scene_detection
from .shot_boundary_saved_predictions import normalize_saved_shot_boundaries
from .storage import bundle_write_lock, read_json, write_json
from .transcript import format_timestamp
from .video import probe_video


SCHEMA = "video_knowledge_pipeline.technical_shot_boundaries.v1"
BOUNDARY_SCHEMA = "video_knowledge_pipeline.technical_shot_boundary.v1"
OUTPUT_PATH = "exports/technical-shot-boundaries.json"
REVIEWED_OUTPUT_PATH = "exports/technical-shot-boundaries.reviewed.json"
MARKDOWN_PATH = "exports/technical-shot-boundaries.md"
BACKEND_SOURCE_FORMAT = {
    "autoshot": "autoshot_scenes",
    "omnishotcut": "omnishotcut_scenes",
}


def run_technical_shot_detection(
    bundle_dir: str | Path,
    *,
    backend: str = "autoshot",
    media_path: str | Path | None = None,
    predictions_json: str | Path | None = None,
    source_format: str = "",
    frame_rate: float | None = None,
    detector: str = "adaptive",
    threshold: float | None = None,
    min_scene_len: int = 15,
    source_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    strict: bool = True,
    write: bool = True,
    _runtime_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create the only technical-shot boundary contract used by pull-film v2.

    Backends are explicit. A failed backend never switches to another detector.
    Chapters and Timeline rows are never accepted as technical shots.
    """

    root = Path(bundle_dir).expanduser().resolve()
    backend_key = str(backend or "").strip().lower().replace("_", "-")
    if backend_key not in {"pyscenedetect", "autoshot", "omnishotcut", "saved"}:
        raise ValueError(
            "backend must be pyscenedetect, autoshot, omnishotcut, or saved"
        )
    manifest = _read_object(root / "manifest.json")
    media = _resolve_media(root, manifest, media_path)

    if backend_key == "pyscenedetect":
        if media is None:
            raise FileNotFoundError("media path is required for PySceneDetect")
        legacy = run_scene_detection(
            root,
            media_path=media,
            detector=detector,
            threshold=threshold,
            min_scene_len=min_scene_len,
            source_root=source_root,
            allow_fallback=not strict,
            write=False,
        )
        if legacy.get("backend") != "pyscenedetect":
            return _blocked_result(
                root,
                backend=backend_key,
                status="blocked_backend_fallback",
                reason=str(
                    legacy.get("fallback_reason") or "PySceneDetect did not run"
                ),
                strict=strict,
                write=write,
            )
        shots = _normalise_seconds_scenes(legacy.get("scenes") or [])
        execution = {
            "mode": "local_runtime",
            "backend": "pyscenedetect",
            "detector": legacy.get("detector"),
            "threshold": legacy.get("threshold"),
            "source_root": legacy.get("source_root"),
            "fallback_used": False,
        }
    else:
        payload, prediction_path, execution = _prediction_payload(
            backend=backend_key,
            media=media,
            predictions_json=predictions_json,
            frame_rate=frame_rate,
            source_root=source_root,
            checkpoint_path=checkpoint_path,
            runtime_runner=_runtime_runner,
        )
        format_key = str(
            source_format
            or BACKEND_SOURCE_FORMAT.get(backend_key)
            or payload.get("source_format")
            or ""
        ).strip().lower()
        if backend_key == "saved" and not format_key:
            raise ValueError("source_format is required for saved predictions")
        supported = {
            "autoshot_scenes",
            "omnishotcut_scenes",
            "transnetv2_scenes",
        }
        if format_key not in supported:
            raise ValueError("unsupported saved technical-shot source_format")
        _, metadata = normalize_saved_shot_boundaries(
            payload,
            source_format=format_key,
            frame_rate=frame_rate,
        )
        shots = _normalise_frame_scenes(
            payload.get("scenes") or [],
            float(metadata["frame_rate"]),
            frame_timestamps_seconds=payload.get("frame_timestamps_seconds"),
            time_offset_seconds=payload.get("time_offset_seconds", 0.0),
        )
        time_basis = (
            "vfr_frame_timestamps"
            if payload.get("frame_timestamps_seconds") is not None
            else "constant_frame_rate"
        )
        execution.update(
            {
                "source_format": format_key,
                "upstream_project": metadata["upstream_project"],
                "upstream_commit": metadata["upstream_commit"],
                "prediction_path": str(prediction_path) if prediction_path else "",
                "prediction_sha256": (
                    sha256_file(prediction_path) if prediction_path else ""
                ),
                "frame_rate": metadata["frame_rate"],
                "time_basis": time_basis,
                "time_offset_seconds": _nonnegative_float(
                    payload.get("time_offset_seconds", 0.0),
                    "time_offset_seconds",
                ),
                "fallback_used": False,
            }
        )

    if not shots:
        return _blocked_result(
            root,
            backend=backend_key,
            status="blocked_missing_technical_shots",
            reason="detector produced no valid technical shot ranges",
            strict=strict,
            write=write,
        )

    result = {
        "schema": SCHEMA,
        "status": "completed",
        "ok": True,
        "bundle_dir": str(root),
        "media": _media_provenance(media),
        "boundary_kind": "technical_shot",
        "backend": backend_key,
        "strict": bool(strict),
        "shot_count": len(shots),
        "boundary_count": max(0, len(shots) - 1),
        "shots": shots,
        "boundaries": _boundaries(shots, backend=backend_key),
        "execution": execution,
        "candidate_only": True,
        "human_confirmation_required": True,
        "timeline_mutated": False,
        "operator_boundary": {
            "local_only": True,
            "no_network_or_model_download": True,
            "no_backend_fallback": True,
            "no_chapter_or_timeline_range_fallback": True,
            "does_not_modify_media": True,
        },
        "artifacts": {"json": OUTPUT_PATH, "markdown": MARKDOWN_PATH},
        "updated_at": now_iso(),
    }
    if write:
        _write_result(root, result)
    return result


def load_verified_technical_shots(
    bundle_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load native v2 shots, or a verifiable legacy PySceneDetect artifact."""

    root = Path(bundle_dir).expanduser().resolve()
    reviewed_path = root / REVIEWED_OUTPUT_PATH
    if reviewed_path.is_file():
        payload = _read_object(reviewed_path)
        valid = (
            payload.get("schema") == SCHEMA
            and payload.get("boundary_kind") == "technical_shot"
            and payload.get("backend") == "human_review"
            and payload.get("human_confirmed") is True
            and payload.get("ok") is True
        )
        if not valid:
            return [], {
                "status": "blocked_invalid_human_review",
                "path": str(reviewed_path),
            }
        stale = _stale_review_sources(root, payload.get("source_artifacts"))
        if stale:
            return [], {
                "status": "blocked_stale_human_review",
                "path": str(reviewed_path),
                "reason": stale,
            }
        shots = _normalise_seconds_scenes(payload.get("shots") or [])
        if shots:
            return shots, _artifact_provenance(
                reviewed_path,
                payload,
                compatibility="human_reviewed_projection",
            )
    current_path = root / OUTPUT_PATH
    if current_path.is_file():
        payload = _read_object(current_path)
        valid = (
            payload.get("schema") == SCHEMA
            and payload.get("boundary_kind") == "technical_shot"
            and payload.get("ok") is True
        )
        shots = (
            _normalise_seconds_scenes(payload.get("shots") or []) if valid else []
        )
        if shots:
            return shots, _artifact_provenance(
                current_path,
                payload,
                compatibility="native_v2",
            )

    legacy_path = root / "exports" / "scene-detection.json"
    if legacy_path.is_file():
        payload = _read_object(legacy_path)
        backend = str(payload.get("backend") or "").strip().lower()
        boundary_kind = str(payload.get("boundary_kind") or "").strip().lower()
        verifiable = (
            backend == "pyscenedetect"
            and boundary_kind in {"", "shot", "technical_shot"}
        )
        shots = (
            _normalise_seconds_scenes(payload.get("scenes") or [])
            if verifiable
            else []
        )
        if shots:
            return shots, _artifact_provenance(
                legacy_path,
                payload,
                compatibility="verified_legacy_pyscenedetect",
            )

    return [], {
        "status": "blocked_missing_technical_shots",
        "reason": (
            "no valid technical_shot_boundaries.v1 or verifiable "
            "PySceneDetect artifact"
        ),
    }


def _stale_review_sources(root: Path, value: Any) -> str:
    rows = value if isinstance(value, list) else []
    if not rows:
        return "reviewed projection has no source_artifacts"
    for row in rows:
        if not isinstance(row, dict):
            return "reviewed projection source_artifacts are invalid"
        raw = str(row.get("path") or "").strip()
        expected = str(row.get("sha256") or "").strip().lower()
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        try:
            path.relative_to(root)
        except ValueError:
            return f"reviewed projection source is outside bundle: {path}"
        if not path.is_file():
            return f"reviewed projection source is missing: {path}"
        if not expected or sha256_file(path).lower() != expected:
            return f"reviewed projection source changed: {path}"
    return ""


def _prediction_payload(
    *,
    backend: str,
    media: Path | None,
    predictions_json: str | Path | None,
    frame_rate: float | None,
    source_root: str | Path | None,
    checkpoint_path: str | Path | None,
    runtime_runner: Callable[..., dict[str, Any]] | None,
) -> tuple[dict[str, Any], Path | None, dict[str, Any]]:
    if predictions_json:
        path = Path(predictions_json).expanduser().resolve()
        payload = _read_object(path)
        return payload, path, {
            "mode": "saved_prediction_import",
            "model_execution_performed": False,
        }
    if backend == "saved":
        raise ValueError("predictions_json is required when backend=saved")
    if media is None:
        raise FileNotFoundError(f"media path is required for {backend} runtime")
    if runtime_runner is None:
        from .shot_boundary_runtime import run_shot_boundary_runtime

        runtime_runner = run_shot_boundary_runtime
    payload = runtime_runner(
        backend=backend,
        media_path=media,
        source_root=source_root,
        checkpoint_path=checkpoint_path,
        frame_rate=frame_rate,
    )
    if not isinstance(payload, dict):
        raise ValueError("shot-boundary runtime must return a JSON object")
    return payload, None, {
        "mode": "local_gpu_runtime",
        "model_execution_performed": True,
    }


def _normalise_frame_scenes(
    rows: list[Any],
    fps: float,
    *,
    frame_timestamps_seconds: Any = None,
    time_offset_seconds: Any = 0.0,
) -> list[dict[str, Any]]:
    timestamps = _frame_timestamps(frame_timestamps_seconds)
    offset = _nonnegative_float(time_offset_seconds, "time_offset_seconds")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if isinstance(row, dict):
            start_frame = row.get("start_frame", row.get("start"))
            end_frame = row.get("end_frame", row.get("end"))
        elif isinstance(row, (list, tuple)) and len(row) == 2:
            start_frame, end_frame = row
        else:
            continue
        try:
            start_i, end_i = int(start_frame), int(end_frame)
        except (TypeError, ValueError):
            continue
        if start_i < 0 or end_i < start_i:
            continue
        if timestamps:
            if end_i >= len(timestamps):
                raise ValueError(
                    "frame_timestamps_seconds does not cover every scene frame"
                )
            start = timestamps[start_i]
            end = _frame_end_time(timestamps, end_i, fps=fps)
        else:
            start, end = start_i / fps, (end_i + 1) / fps
        start += offset
        end += offset
        result.append(
            _shot_row(
                index,
                start,
                end,
                start_frame=start_i,
                end_frame=end_i,
            )
        )
    return _ordered_nonoverlapping(result)


def _frame_timestamps(value: Any) -> list[float]:
    if value is None:
        return []
    if not isinstance(value, list) or not value:
        raise ValueError("frame_timestamps_seconds must be a non-empty array")
    result: list[float] = []
    previous = -1.0
    for index, raw in enumerate(value):
        timestamp = _nonnegative_float(
            raw,
            f"frame_timestamps_seconds[{index}]",
        )
        if timestamp <= previous:
            raise ValueError("frame_timestamps_seconds must be strictly increasing")
        result.append(timestamp)
        previous = timestamp
    return result


def _frame_end_time(timestamps: list[float], end_frame: int, *, fps: float) -> float:
    next_index = end_frame + 1
    if next_index < len(timestamps):
        return timestamps[next_index]
    if len(timestamps) >= 2:
        step = timestamps[-1] - timestamps[-2]
    else:
        step = 1.0 / fps
    return timestamps[end_frame] + step


def _nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative number") from exc
    if number < 0:
        raise ValueError(f"{label} must be a non-negative number")
    return number


def _normalise_seconds_scenes(rows: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        try:
            start, end = float(row.get("start")), float(row.get("end"))
        except (TypeError, ValueError):
            continue
        if start < 0 or end <= start:
            continue
        result.append(_shot_row(index, start, end))
    return _ordered_nonoverlapping(result)


def _ordered_nonoverlapping(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows.sort(key=lambda row: (float(row["start"]), float(row["end"])))
    previous_end = -1.0
    for row in rows:
        if float(row["start"]) < previous_end - 0.001:
            raise ValueError(
                "technical shot ranges must be ordered and non-overlapping"
            )
        previous_end = float(row["end"])
    return rows


def _shot_row(
    index: int,
    start: float,
    end: float,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "shot_id": f"technical-shot-{index:04d}",
        "index": index,
        "start": round(start, 6),
        "end": round(end, 6),
        "duration": round(end - start, 6),
        "start_time": format_timestamp(start),
        "end_time": format_timestamp(end),
        "boundary_kind": "technical_shot",
        **extra,
    }


def _boundaries(
    shots: list[dict[str, Any]],
    *,
    backend: str,
) -> list[dict[str, Any]]:
    return [
        {
            "schema": BOUNDARY_SCHEMA,
            "boundary_id": f"technical-boundary-{index:04d}",
            "seconds": shot["start"],
            "time": shot["start_time"],
            "backend": backend,
            "candidate_only": True,
        }
        for index, shot in enumerate(shots[1:], start=1)
    ]


def _blocked_result(
    root: Path,
    *,
    backend: str,
    status: str,
    reason: str,
    strict: bool,
    write: bool,
) -> dict[str, Any]:
    result = {
        "schema": SCHEMA,
        "status": status,
        "ok": False,
        "bundle_dir": str(root),
        "boundary_kind": "technical_shot",
        "backend": backend,
        "strict": bool(strict),
        "reason": reason,
        "shot_count": 0,
        "boundary_count": 0,
        "shots": [],
        "boundaries": [],
        "candidate_only": True,
        "timeline_mutated": False,
        "operator_boundary": {
            "no_backend_fallback": True,
            "no_chapter_or_timeline_range_fallback": True,
        },
        "artifacts": {"json": OUTPUT_PATH, "markdown": MARKDOWN_PATH},
        "updated_at": now_iso(),
    }
    if write:
        _write_result(root, result)
    return result


def _write_result(root: Path, result: dict[str, Any]) -> None:
    exports = root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    with bundle_write_lock(
        root,
        operation="technical_shot_detection",
        timeout_seconds=1.0,
    ):
        write_json(root / OUTPUT_PATH, result)
        (root / MARKDOWN_PATH).write_text(
            _render_markdown(result),
            encoding="utf-8",
        )
        manifest = _read_object(manifest_path)
        manifest["technical_shot_boundaries_json"] = OUTPUT_PATH
        manifest["technical_shot_boundaries_markdown"] = MARKDOWN_PATH
        manifest["technical_shot_summary"] = {
            "status": result["status"],
            "backend": result["backend"],
            "shot_count": result["shot_count"],
            "updated_at": result["updated_at"],
        }
        write_json(manifest_path, manifest)


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Technical Shot Boundaries",
        "",
        f"- Status: {result.get('status')}",
        f"- Backend: {result.get('backend')}",
        f"- Shot count: {result.get('shot_count', 0)}",
        "- Candidate evidence: true",
        "- Automatic fallback: false",
    ]
    if result.get("reason"):
        lines.append(f"- Blocker: {result['reason']}")
    lines.extend(["", "## Shots", ""])
    for shot in result.get("shots") or []:
        lines.append(
            f"- {shot['shot_id']} {shot['start_time']} - {shot['end_time']}"
        )
    return "\n".join(lines).rstrip() + "\n"


def _media_provenance(media: Path | None) -> dict[str, Any]:
    if media is None:
        return {"status": "not_required_for_saved_import"}
    metadata = probe_video(media)
    return {
        "status": "bound",
        "path": str(media),
        "sha256": metadata.sha256,
        "duration_seconds": metadata.duration_seconds,
        "fps": metadata.fps,
        "width": metadata.width,
        "height": metadata.height,
    }


def _artifact_provenance(
    path: Path,
    payload: dict[str, Any],
    *,
    compatibility: str,
) -> dict[str, Any]:
    return {
        "status": "loaded",
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": payload.get("schema"),
        "backend": payload.get("backend"),
        "compatibility": compatibility,
    }


def _resolve_media(
    root: Path,
    manifest: dict[str, Any],
    value: str | Path | None,
) -> Path | None:
    candidate = value or manifest.get("media_path") or manifest.get("source_path")
    if not candidate:
        return None
    path = Path(str(candidate)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve() if path.is_file() else None


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}
