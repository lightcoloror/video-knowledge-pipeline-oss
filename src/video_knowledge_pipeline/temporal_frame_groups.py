from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .powershell import quote_powershell_argument as _quote_command_part
from .config import resolve_vision_execution_profile
from .frame_recapture import _coverage_audit, _quality_audit
from .media_tools import resolve_media_tool
from .models import now_iso
from .repair_status import build_repair_status
from .storage import read_json, write_json
from .visual_integration import integrated_visual


DEFAULT_ROUTES = {"temporal_sequence", "mixed"}


def run_temporal_frame_groups(
    bundle_dir: str | Path,
    *,
    execute: bool = False,
    frame_count: int | None = None,
    window_seconds: float = 4.0,
    include_routes: list[str] | None = None,
    indexes: list[int] | None = None,
    limit: int | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Generate 5-12 ordered frame groups for temporal visual understanding."""
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"bundle missing manifest.json: {root}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"bundle missing timeline.json: {root}")
    manifest = read_json(manifest_path)
    timeline_data = read_json(timeline_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")
    if not isinstance(timeline_data, list):
        raise ValueError("timeline.json must contain a JSON array")
    timeline = [item for item in timeline_data if isinstance(item, dict)]
    profile = resolve_vision_execution_profile(temporal_limit=limit, frame_count=frame_count)
    count = int(profile["frame_count"])
    effective_limit = int(profile["temporal_limit"])
    requested_indexes = {int(value) for value in (indexes or []) if int(value) > 0}
    routes = {str(route) for route in (include_routes or sorted(DEFAULT_ROUTES)) if str(route)}
    if requested_indexes:
        # Explicit indexes are an operator/planner decision. Include their
        # current route so stale/unknown route labels cannot turn a precise
        # recapture request into a silent no-op.
        routes.update(
            str(item.get("visual_route") or "unknown")
            for position, item in enumerate(timeline, start=1)
            if (_int_value(item.get("index")) or position) in requested_indexes
        )
    candidates = plan_temporal_frame_candidates(
        timeline,
        routes=routes,
        frame_count=count,
        window_seconds=float(window_seconds or 4.0),
        root=root,
    )
    if requested_indexes:
        candidates = [candidate for candidate in candidates if _int_value(candidate.get("index")) in requested_indexes]
    if effective_limit > 0:
        candidates = candidates[:effective_limit]
    results = [_run_candidate(candidate, execute=execute, timeout_seconds=timeout_seconds) for candidate in candidates]
    updated = _backfill_timeline(timeline, results) if execute else []
    if execute:
        write_json(timeline_path, timeline)
        _sync_source_package(manifest, timeline, updated)

    summary = {
        "schema": "lecture_temporal_frame_groups_summary.v1",
        "total": len(candidates),
        "execute": execute,
        "frame_count": count,
        "window_seconds": float(window_seconds or 4.0),
        "include_routes": sorted(routes),
        "indexes": sorted(requested_indexes),
        "limit": effective_limit,
        "succeeded": sum(1 for item in results if item.get("ok")),
        "failed": sum(1 for item in results if item.get("executed") and not item.get("ok")),
        "planned": sum(1 for item in results if not item.get("executed")),
        "updated": len(updated),
        "updated_indexes": updated,
        "updated_at": now_iso(),
    }
    manifest["temporal_frame_groups"] = {
        "schema": "lecture_temporal_frame_groups.v1",
        "count": len(candidates),
        "items": results,
        "last_run": summary,
    }
    manifest["coverage"] = _coverage_audit(timeline)
    manifest["quality_audit"] = _quality_audit(timeline)
    manifest["repair_status"] = build_repair_status(manifest, timeline)
    write_json(manifest_path, manifest)
    report_path = root / "temporal-frame-groups-report.md"
    report_path.write_text(_render_report(root, results, summary), encoding="utf-8")
    return {
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "timeline_path": str(timeline_path),
        "report_path": str(report_path),
        "summary": summary,
        "items": results,
    }


def plan_temporal_frame_candidates(
    timeline: list[dict[str, Any]],
    *,
    routes: set[str],
    frame_count: int,
    window_seconds: float,
    root: Path,
) -> list[dict[str, Any]]:
    """Return the existing ordered-frame plan without running FFmpeg or writing files."""
    items: list[dict[str, Any]] = []
    for index, item in enumerate(timeline, start=1):
        route = str(item.get("visual_route") or "unknown")
        if route not in routes:
            continue
        video_key = str(item.get("video_key") or "").strip()
        if not video_key:
            continue
        start = _float_value(item.get("start"))
        end = _float_value(item.get("end"))
        duration = _float_value(item.get("video_duration_seconds"))
        midpoint = _float_value(item.get("midpoint"))
        if midpoint <= 0 and end > start:
            midpoint = (start + end) / 2
        if end <= start:
            half = max(window_seconds, 0.5) / 2
            start = max(0.0, midpoint - half)
            end = midpoint + half
        times = _sample_times(start, end, frame_count, duration=duration)
        output_dir = root / "temporal-frames" / f"{index:04d}"
        outputs = [output_dir / f"frame_{pos:02d}_{_timestamp_for_filename(time)}.jpg" for pos, time in enumerate(times, start=1)]
        items.append(
            {
                "index": index,
                "visual_route": route,
                "video_key": video_key,
                "start": start,
                "end": end,
                "frame_count": frame_count,
                "times": times,
                "output_paths": [str(path) for path in outputs],
            }
        )
    return items


def _candidates(
    timeline: list[dict[str, Any]],
    *,
    routes: set[str],
    frame_count: int,
    window_seconds: float,
    root: Path,
) -> list[dict[str, Any]]:
    """Backward-compatible private alias for existing callers."""
    return plan_temporal_frame_candidates(
        timeline,
        routes=routes,
        frame_count=frame_count,
        window_seconds=window_seconds,
        root=root,
    )


def _run_candidate(candidate: dict[str, Any], *, execute: bool, timeout_seconds: int) -> dict[str, Any]:
    ffmpeg = resolve_media_tool("ffmpeg")
    outputs = [Path(str(path)).expanduser() for path in candidate.get("output_paths") or []]
    result = {
        **candidate,
        "executed": execute,
        "ok": False,
        "ffmpeg": ffmpeg,
        "commands": [],
        "returncodes": [],
        "stderr": "",
        "exists_count": sum(1 for path in outputs if path.exists()),
    }
    for time, output in zip(candidate.get("times") or [], outputs):
        command = [ffmpeg or "ffmpeg", "-y", "-ss", f"{_float_value(time):.3f}", "-i", str(candidate.get("video_key") or ""), "-frames:v", "1", "-q:v", "2", str(output)]
        result["commands"].append(" ".join(_quote_command_part(part) for part in command))
    if not execute:
        result["ok"] = bool(ffmpeg and candidate.get("video_key") and outputs)
        return result
    if not ffmpeg:
        result["stderr"] = "ffmpeg was not found; set LECTURE_FFMPEG_DIR or FFMPEG_BINARY"
        return result
    output_parent = outputs[0].parent if outputs else None
    if output_parent:
        output_parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for time, output in zip(candidate.get("times") or [], outputs):
        command = [ffmpeg, "-y", "-ss", f"{_float_value(time):.3f}", "-i", str(candidate.get("video_key") or ""), "-frames:v", "1", "-q:v", "2", str(output)]
        try:
            completed = subprocess.run(
                command,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=int(timeout_seconds or 0) or None,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            result["returncodes"].append(None)
            errors.append(f"timeout after {timeout_seconds}s: {exc}")
            continue
        result["returncodes"].append(completed.returncode)
        if completed.returncode != 0:
            errors.append(completed.stderr.strip())
    result["exists_count"] = sum(1 for path in outputs if path.exists())
    result["ok"] = result["exists_count"] == len(outputs) and bool(outputs)
    result["stderr"] = "\n".join(error for error in errors if error)
    return result


def _backfill_timeline(timeline: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[int]:
    updated: list[int] = []
    for result in results:
        if not result.get("ok"):
            continue
        index = _int_value(result.get("index"))
        if not (1 <= index <= len(timeline)):
            continue
        item = timeline[index - 1]
        paths = [str(Path(str(path)).expanduser()) for path in result.get("output_paths") or [] if Path(str(path)).expanduser().exists()]
        if not paths:
            continue
        item["temporal_frame_paths"] = paths
        item["temporal_frame_group"] = {
            "schema": "lecture_temporal_frame_group.v1",
            "frame_paths": paths,
            "times": result.get("times") or [],
            "source": "temporal_frame_groups",
            "updated_at": now_iso(),
        }
        material_types = item.setdefault("material_types", [])
        if isinstance(material_types, list) and "temporal_sequence" not in material_types:
            material_types.append("temporal_sequence")
        item["integrated_visual"] = integrated_visual(item)
        updated.append(index)
    return updated


def _sync_source_package(manifest: dict[str, Any], timeline: list[dict[str, Any]], updated_indexes: list[int]) -> None:
    if not updated_indexes:
        return
    source = Path(str(manifest.get("source_package") or "")).expanduser()
    if not source.exists() or not source.is_file():
        return
    package = read_json(source)
    if not isinstance(package, dict) or not isinstance(package.get("timeline"), list):
        return
    for index in updated_indexes:
        if not (1 <= index <= len(timeline) and index <= len(package["timeline"])):
            continue
        target = package["timeline"][index - 1]
        source_item = timeline[index - 1]
        if not isinstance(target, dict):
            continue
        for key in ("temporal_frame_paths", "temporal_frame_group", "material_types", "integrated_visual"):
            if key in source_item:
                target[key] = source_item[key]
    package["coverage"] = _coverage_audit(package["timeline"])
    package["quality_audit"] = _quality_audit(package["timeline"])
    package["temporal_frame_groups_backfilled_at"] = now_iso()
    write_json(source, package)


def _sample_times(start: float, end: float, count: int, *, duration: float = 0.0) -> list[float]:
    start = max(0.0, start)
    end = max(start, end)
    if duration > 0:
        max_time = max(0.0, duration - 0.12)
        start = min(start, max_time)
        end = min(end, max_time)
    if count <= 1:
        return [start]
    if end == start:
        return [start for _ in range(count)]
    return [start + ((end - start) * i / (count - 1)) for i in range(count)]


def _render_report(root: Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Temporal Frame Groups Report",
        "",
        f"- Bundle: `{root}`",
        f"- Total: {summary.get('total', 0)}",
        f"- Execute: `{summary.get('execute')}`",
        f"- Updated: `{summary.get('updated', 0)}`",
        "",
    ]
    for item in results:
        lines.extend(
            [
                f"## Timeline {item.get('index')}",
                "",
                f"- Route: `{item.get('visual_route')}`",
                f"- Frames: `{item.get('exists_count', 0)}/{len(item.get('output_paths') or [])}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _timestamp_for_filename(seconds: float) -> str:
    millis = int(round(max(seconds, 0.0) * 1000))
    return f"{millis:010d}ms"
