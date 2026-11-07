from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .models import now_iso
from .path_defaults import tool_source_review_root
from .storage import bundle_write_lock, read_json, write_json
from .transcript import format_timestamp
from .video import scene_change_timepoints


SCHEMA = "video_knowledge_pipeline.scene_detection.v1"
DEFAULT_SOURCE_ROOT = tool_source_review_root() / "PySceneDetect"


def pyscenedetect_timepoints(
    media_path: str | Path,
    *,
    detector: str = "adaptive",
    threshold: float | None = None,
    min_scene_len: int = 15,
    max_points: int = 300,
    source_root: str | Path | None = None,
) -> tuple[list[tuple[float, str]], dict[str, Any]]:
    """Return scene boundary timepoints for the sampling orchestrator."""

    media = Path(media_path).expanduser().resolve()
    detector_name = str(detector or "adaptive").strip().lower().replace("-", "_")
    effective_threshold = float(threshold if threshold is not None else (3.0 if detector_name == "adaptive" else 27.0))
    source = Path(source_root).expanduser().resolve() if source_root else _source_root()
    try:
        api = _load_pyscenedetect(source)
        video = api["open_video"](str(media))
        manager = api["SceneManager"]()
        detector_instance = (
            api["AdaptiveDetector"](adaptive_threshold=effective_threshold, min_scene_len=max(1, int(min_scene_len)))
            if detector_name == "adaptive"
            else api["ContentDetector"](threshold=effective_threshold, min_scene_len=max(1, int(min_scene_len)))
        )
        manager.add_detector(detector_instance)
        manager.detect_scenes(video=video, show_progress=False)
        _, points = _normalise_scene_list(manager.get_scene_list(start_in_scene=True), max_points=max_points)
        return (
            [(float(row["seconds"]), "pyscenedetect_scene_boundary") for row in points],
            {
                "backend": "pyscenedetect",
                "detector": detector_name,
                "threshold": effective_threshold,
                "boundary_count": len(points),
                "fallback_reason": "",
            },
        )
    except Exception as exc:
        points = scene_change_timepoints(media, max_points=max(0, int(max_points)))
        return (
            points,
            {
                "backend": "ffmpeg_scene_fallback",
                "detector": detector_name,
                "threshold": effective_threshold,
                "boundary_count": len(points),
                "fallback_reason": f"{type(exc).__name__}: {exc}",
            },
        )

def run_scene_detection(
    bundle_dir: str | Path,
    *,
    media_path: str | Path | None = None,
    detector: str = "adaptive",
    threshold: float | None = None,
    min_scene_len: int = 15,
    max_points: int = 300,
    source_root: str | Path | None = None,
    allow_fallback: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Run PySceneDetect locally, with an explicit ffmpeg fallback."""

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    manifest = manifest if isinstance(manifest, dict) else {}
    media = _resolve_media(root, manifest, media_path)
    if media is None:
        raise FileNotFoundError("media path is required for scene detection")

    detector_name = str(detector or "adaptive").strip().lower().replace("-", "_")
    if detector_name not in {"adaptive", "content"}:
        raise ValueError("detector must be adaptive or content")
    effective_threshold = float(threshold if threshold is not None else (3.0 if detector_name == "adaptive" else 27.0))
    source = Path(source_root).expanduser().resolve() if source_root else _source_root()
    backend = "pyscenedetect"
    fallback_reason = ""
    scenes: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    try:
        api = _load_pyscenedetect(source)
        video = api["open_video"](str(media))
        manager = api["SceneManager"]()
        if detector_name == "adaptive":
            detector_instance = api["AdaptiveDetector"](adaptive_threshold=effective_threshold, min_scene_len=max(1, int(min_scene_len)))
        else:
            detector_instance = api["ContentDetector"](threshold=effective_threshold, min_scene_len=max(1, int(min_scene_len)))
        manager.add_detector(detector_instance)
        manager.detect_scenes(video=video, show_progress=False)
        scene_list = manager.get_scene_list(start_in_scene=True)
        scenes, points = _normalise_scene_list(scene_list, max_points=max_points)
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(
                f"PySceneDetect strict execution failed: {type(exc).__name__}: {exc}"
            ) from exc
        backend = "ffmpeg_scene_fallback"
        fallback_reason = f"{type(exc).__name__}: {exc}"
        fallback = scene_change_timepoints(media, max_points=max(0, int(max_points)))
        points = [
            {
                "index": index + 1,
                "seconds": round(float(seconds), 3),
                "time": format_timestamp(float(seconds)),
                "reason": reason,
            }
            for index, (seconds, reason) in enumerate(fallback)
        ]

    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "media_path": str(media),
        "status": "completed" if points else ("completed_no_boundaries" if backend == "pyscenedetect" else "fallback_no_boundaries"),
        "ok": backend == "pyscenedetect" or bool(points),
        "backend": backend,
        "detector": detector_name,
        "threshold": effective_threshold,
        "min_scene_len": max(1, int(min_scene_len)),
        "max_points": max(0, int(max_points)),
        "source_root": str(source) if source else "",
        "fallback_reason": fallback_reason,
        "scene_count": len(scenes),
        "boundary_count": len(points),
        "scenes": scenes,
        "boundaries": points,
        "operator_boundary": {
            "local_only": True,
            "no_model_or_cloud_call": True,
            "scene_boundaries_are_candidates": True,
            "does_not_replace_uniform_or_semantic_sampling": True,
            "fallback_is_reported": True,
            "fallback_allowed": bool(allow_fallback),
        },
        "artifacts": {
            "json": "exports/scene-detection.json",
            "markdown": "exports/scene-detection.md",
        },
        "updated_at": now_iso(),
    }
    if write:
        exports = root / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        with bundle_write_lock(root, operation="scene_detection", timeout_seconds=1.0):
            write_json(exports / "scene-detection.json", result)
            (exports / "scene-detection.md").write_text(_render_markdown(result), encoding="utf-8")
            write_json(
                root / "mcp-scene-detection.args.json",
                {
                    "bundle_dir": str(root),
                    "media_path": str(media),
                    "detector": detector_name,
                    "threshold": effective_threshold,
                    "min_scene_len": int(min_scene_len),
                    "max_points": int(max_points),
                    "write": True,
                },
            )
            manifest["scene_detection_json"] = "exports/scene-detection.json"
            manifest["scene_detection_markdown"] = "exports/scene-detection.md"
            manifest["mcp_scene_detection_args"] = "mcp-scene-detection.args.json"
            manifest["scene_detection_summary"] = {
                "backend": backend,
                "boundary_count": len(points),
                "fallback_reason": fallback_reason,
                "updated_at": result["updated_at"],
            }
            write_json(manifest_path, manifest)
    return result


def scene_boundary_candidates(root: str | Path) -> list[dict[str, Any]]:
    path = Path(root).expanduser().resolve() / "exports" / "scene-detection.json"
    if not path.exists():
        return []
    payload = read_json(path)
    if not isinstance(payload, dict):
        return []
    return [row for row in payload.get("boundaries") or [] if isinstance(row, dict)]


def _load_pyscenedetect(source_root: Path | None) -> dict[str, Any]:
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import AdaptiveDetector, ContentDetector
    except ImportError:
        if source_root is None or not source_root.exists():
            raise
        source_text = str(source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import AdaptiveDetector, ContentDetector
    return {
        "SceneManager": SceneManager,
        "open_video": open_video,
        "AdaptiveDetector": AdaptiveDetector,
        "ContentDetector": ContentDetector,
    }


def _normalise_scene_list(scene_list: list[Any], *, max_points: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenes: list[dict[str, Any]] = []
    for index, pair in enumerate(scene_list):
        start, end = pair
        start_seconds = float(start.get_seconds())
        end_seconds = float(end.get_seconds())
        scenes.append(
            {
                "index": index + 1,
                "start": round(start_seconds, 3),
                "end": round(end_seconds, 3),
                "start_time": format_timestamp(start_seconds),
                "end_time": format_timestamp(end_seconds),
            }
        )
    boundary_seconds = [float(row["start"]) for row in scenes[1:] if float(row["start"]) > 0]
    boundary_seconds = _uniform_cap(boundary_seconds, max_points=max_points)
    points = [
        {
            "index": index + 1,
            "seconds": round(seconds, 3),
            "time": format_timestamp(seconds),
            "reason": "pyscenedetect_scene_boundary",
        }
        for index, seconds in enumerate(boundary_seconds)
    ]
    return scenes, points


def _uniform_cap(values: list[float], *, max_points: int) -> list[float]:
    if max_points <= 0:
        return []
    if len(values) <= max_points:
        return values
    if max_points == 1:
        return [values[len(values) // 2]]
    indexes = {round(index * (len(values) - 1) / (max_points - 1)) for index in range(max_points)}
    return [values[index] for index in sorted(indexes)]


def _resolve_media(root: Path, manifest: dict[str, Any], value: str | Path | None) -> Path | None:
    candidate = value or manifest.get("media_path") or manifest.get("source_path")
    if not candidate:
        return None
    path = Path(str(candidate)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve() if path.exists() else None


def _source_root() -> Path | None:
    configured = os.environ.get("VKP_PYSCENEDETECT_SOURCE", "").strip()
    if configured:
        path = Path(configured).expanduser()
        return path.resolve() if path.exists() else None
    return DEFAULT_SOURCE_ROOT.resolve() if DEFAULT_SOURCE_ROOT.exists() else None


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Scene Detection",
        "",
        f"- Backend: {result.get('backend')}",
        f"- Detector: {result.get('detector')}",
        f"- Boundaries: {result.get('boundary_count')}",
        f"- Fallback reason: {result.get('fallback_reason')}",
        "",
        "| Index | Time | Reason |",
        "| ---: | --- | --- |",
    ]
    for row in result.get("boundaries") or []:
        lines.append(f"| {row.get('index')} | {row.get('time')} | {row.get('reason')} |")
    return "\n".join(lines).rstrip() + "\n"