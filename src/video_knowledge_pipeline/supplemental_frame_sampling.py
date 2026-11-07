from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import bundle_write_lock, read_json, write_json
from .vision_review_triage import vision_review_triage

SCHEMA = "video_knowledge_pipeline.supplemental_frame_sampling_plan.v1"

DEFAULT_OFFSETS = (-3.0, 0.0, 3.0, 8.0)
TEMPORAL_OFFSETS = (-6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0)


def plan_supplemental_frame_sampling(
    bundle_dir: str | Path,
    *,
    triage_json: str | Path | None = None,
    max_items: int = 0,
    max_frames_per_item: int = 4,
    include_temporal: bool = True,
    include_visual_structure: bool = True,
    include_semantic: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Plan local supplemental frame recapture from existing evidence gaps.

    This is intentionally a planner. It does not call ffmpeg, OCR, ebook tools,
    or cloud vision. When `write=True`, it writes `manifest.frame_recapture`
    items so the existing `run-frame-recapture-plan` executor can be reused.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline.json not found: {timeline_path}")

    manifest = _as_dict(read_json(manifest_path)) if manifest_path.exists() else {}
    triage = _load_or_create_triage(root, triage_json=triage_json)
    video_path = _source_video_path(root, manifest)
    candidates = _candidate_rows(
        triage,
        include_temporal=include_temporal,
        include_visual_structure=include_visual_structure,
        include_semantic=include_semantic,
    )
    if max_items and max_items > 0:
        candidates = candidates[: int(max_items)]

    max_frames_per_item = max(1, int(max_frames_per_item or 1))
    items: list[dict[str, Any]] = []
    seen: set[tuple[int, float]] = set()
    for row in candidates:
        for midpoint in _recommended_midpoints(row, max_frames=max_frames_per_item):
            index = _int(row.get("index"))
            key = (index, round(midpoint, 3))
            if key in seen:
                continue
            seen.add(key)
            item = _recapture_item(root, row, midpoint=midpoint, video_path=video_path)
            items.append(item)

    summary = {
        "total_candidates": len(candidates),
        "planned_frames": len(items),
        "has_source_video": bool(video_path),
        "cloud_vision_allowed_by_default": False,
        "write": bool(write),
        "updated_at": now_iso(),
    }
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "status": "ok" if video_path else "needs_source_video",
        "source_video_path": video_path,
        "triage_source": str(triage_json or root / "vision-review-triage.json"),
        "summary": summary,
        "items": items,
        "next_actions": _next_actions(root, has_source_video=bool(video_path), planned_frames=len(items)),
    }

    if write:
        with bundle_write_lock(root, operation="supplemental_frame_sampling_plan", timeout_seconds=1.0):
            _write_outputs(root, manifest, result)
    return result


def _load_or_create_triage(root: Path, *, triage_json: str | Path | None) -> dict[str, Any]:
    if triage_json:
        path = Path(triage_json).expanduser().resolve()
        data = read_json(path)
        return _as_dict(data)
    path = root / "vision-review-triage.json"
    if path.exists():
        return _as_dict(read_json(path))
    return vision_review_triage(root, write=False)


def _candidate_rows(
    triage: dict[str, Any], *, include_temporal: bool, include_visual_structure: bool, include_semantic: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if include_temporal:
        rows.extend(_with_action(_as_list(triage.get("temporal_recapture_candidates")), "temporal_recapture", force=True))
        rows.extend(_with_action(_as_list(triage.get("temporal_candidates")), "temporal_multimodal"))
    if include_visual_structure:
        rows.extend(_with_action(_as_list(triage.get("visual_structure_first_candidates")), "visual_structure_first"))
    if include_semantic:
        rows.extend(_with_action(_as_list(triage.get("semantic_candidates")), "semantic_multimodal"))

    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = _int(row.get("index"))
        if index <= 0:
            continue
        current = by_index.get(index)
        if current is None or _candidate_preference(row) < _candidate_preference(current):
            by_index[index] = row
    return sorted(by_index.values(), key=lambda row: (-_int(row.get("score")), _action_rank(str(row.get("recommended_action") or "")), _int(row.get("index"))))


def _with_action(rows: list[Any], fallback_action: str, *, force: bool = False) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        copy = dict(row)
        if force:
            copy["recommended_action"] = fallback_action
        else:
            copy.setdefault("recommended_action", fallback_action)
        result.append(copy)
    return result


def _action_rank(action: str) -> int:
    return {
        "temporal_recapture": 0,
        "temporal_multimodal": 1,
        "visual_structure_first": 2,
        "semantic_multimodal": 3,
    }.get(action, 9)


def _candidate_preference(row: dict[str, Any]) -> tuple[int, int]:
    return (_action_rank(str(row.get("recommended_action") or "")), -_int(row.get("score")))


def _recommended_midpoints(row: dict[str, Any], *, max_frames: int) -> list[float]:
    start = _float(row.get("start"))
    end = _float(row.get("end"))
    if end <= start:
        end = start + 4.0
    base = max(0.0, (start + end) / 2.0)
    action = str(row.get("recommended_action") or "")
    offsets = TEMPORAL_OFFSETS if action in {"temporal_multimodal", "temporal_recapture"} else DEFAULT_OFFSETS
    points = []
    for offset in offsets:
        point = max(0.0, base + offset)
        if not any(abs(point - existing) < 0.25 for existing in points):
            points.append(point)
        if len(points) >= max_frames:
            break
    return points


def _recapture_item(root: Path, row: dict[str, Any], *, midpoint: float, video_path: str) -> dict[str, Any]:
    index = _int(row.get("index"))
    action = str(row.get("recommended_action") or "")
    output_path = root / "supplemental-frames" / f"timeline-{index:04d}-{midpoint:010.3f}.jpg"
    command = f'ffmpeg -y -ss {midpoint:.3f} -i "{video_path}" -frames:v 1 "{output_path}"'
    return {
        "index": index,
        "midpoint": round(midpoint, 3),
        "video_key": video_path,
        "output_path": str(output_path),
        "ffmpeg_command": command,
        "recommended_action": action,
        "source": "supplemental_frame_sampling_plan",
        "score": _int(row.get("score")),
        "priority": row.get("priority") or "",
        "reasons": row.get("reasons") if isinstance(row.get("reasons"), list) else [],
        "visual_route": row.get("visual_route") or "",
        "cloud_vision_allowed_by_default": False,
        "transcript_excerpt": row.get("transcript_excerpt") or "",
        "visual_text_excerpt": row.get("visual_text_excerpt") or "",
    }


def _write_outputs(root: Path, manifest: dict[str, Any], result: dict[str, Any]) -> None:
    json_path = root / "supplemental-frame-sampling-plan.json"
    md_path = root / "supplemental-frame-sampling-plan.md"
    args_path = root / "mcp-supplemental-frame-sampling-plan.args.json"
    write_json(json_path, result)
    md_path.write_text(_render_markdown(result), encoding="utf-8")
    write_json(args_path, {"bundle_dir": str(root), "write": True})

    manifest["frame_recapture"] = {
        "schema": "video_knowledge_pipeline.frame_recapture.v1",
        "source": "supplemental_frame_sampling_plan",
        "planned_at": now_iso(),
        "items": result.get("items", []),
        "plan_path": str(json_path),
        "report_path": str(md_path),
    }
    supplemental = manifest.setdefault("supplemental_frame_sampling", {})
    if isinstance(supplemental, dict):
        supplemental.update(
            {
                "schema": SCHEMA,
                "plan_path": str(json_path),
                "report_path": str(md_path),
                "planned_frames": result.get("summary", {}).get("planned_frames", 0),
                "updated_at": now_iso(),
            }
        )
    write_json(root / "manifest.json", manifest)


def _render_markdown(result: dict[str, Any]) -> str:
    summary = _as_dict(result.get("summary"))
    lines = [
        "# Supplemental Frame Sampling Plan",
        "",
        f"- Bundle: `{result.get('bundle_dir', '')}`",
        f"- Status: `{result.get('status', '')}`",
        f"- Source video: `{result.get('source_video_path', '') or '(missing)'}`",
        f"- Candidates: {summary.get('total_candidates', 0)}",
        f"- Planned frames: {summary.get('planned_frames', 0)}",
        f"- Cloud vision allowed by default: `{summary.get('cloud_vision_allowed_by_default')}`",
        "",
        "## Next Actions",
        "",
    ]
    for action in _as_list(result.get("next_actions")):
        if not isinstance(action, dict):
            continue
        lines.append(f"- `{action.get('key', '')}`: {action.get('description', '')}")
        if action.get("command"):
            lines.extend(["", "```powershell", str(action.get("command")), "```", ""])
    lines.extend(["", "## Planned Frames", ""])
    items = _as_list(result.get("items"))
    if not items:
        lines.append("No supplemental frames planned.")
    for item in items[:200]:
        lines.extend(
            [
                f"### Timeline {item.get('index')} @ {item.get('midpoint')}s",
                "",
                f"- Action: `{item.get('recommended_action', '')}`",
                f"- Score: `{item.get('score', '')}` / Priority: `{item.get('priority', '')}`",
                f"- Reasons: `{', '.join(str(v) for v in _as_list(item.get('reasons')))}`",
                f"- Output: `{item.get('output_path', '')}`",
                f"- Transcript: {item.get('transcript_excerpt', '')}",
                "",
            ]
        )
    if len(items) > 200:
        lines.append(f"... {len(items) - 200} more planned frames omitted from markdown; see JSON.")
    return "\n".join(lines).rstrip() + "\n"


def _next_actions(root: Path, *, has_source_video: bool, planned_frames: int) -> list[dict[str, Any]]:
    actions = []
    if not has_source_video:
        actions.append(
            {
                "key": "set_source_video_path",
                "description": "补帧需要原视频路径。请在 manifest/source_package 中保留 media_path，或重新从 acceptance/openclaw ingest 入口生成 bundle。",
            }
        )
    if planned_frames > 0 and has_source_video:
        actions.append(
            {
                "key": "run_frame_recapture_plan",
                "description": "执行本地 ffmpeg 补帧；不会调用云模型。",
                "command": f'.\\scripts\\video-knowledge.ps1 run-frame-recapture-plan "{root}" --execute',
            }
        )
    actions.append(
        {
            "key": "rerun_triage_after_local_evidence",
            "description": "补帧、OCR/ebook、crop/OCR 完成后，再运行 vision-review-triage 重新挑选真正需要多模态的疑难点。",
            "command": f'.\\scripts\\video-knowledge.ps1 vision-review-triage "{root}" --mode triage',
        }
    )
    return actions


def _source_video_path(root: Path, manifest: dict[str, Any]) -> str:
    for key in ("media_path", "source_video_path", "video_path", "local_media_path"):
        value = str(manifest.get(key) or "").strip()
        if value:
            return value
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    for key in ("media_path", "path", "source_path", "video_path", "local_media_path"):
        value = str(source.get(key) or "").strip()
        if value:
            return value
    package_text = str(manifest.get("source_package") or "").strip()
    if package_text:
        package_path = Path(package_text).expanduser()
        if not package_path.is_absolute():
            package_path = root / package_path
        if package_path.exists():
            package = _as_dict(read_json(package_path))
            for key in ("media_path", "video_path", "local_media_path"):
                value = str(package.get(key) or "").strip()
                if value:
                    return value
            source_pkg = package.get("source") if isinstance(package.get("source"), dict) else {}
            for key in ("media_path", "path", "source_path", "video_path", "local_media_path"):
                value = str(source_pkg.get(key) or "").strip()
                if value:
                    return value
    return ""


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
