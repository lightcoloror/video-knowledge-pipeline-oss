from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any, Mapping

from .models import now_iso
from .panorama_evidence import panorama_bundle_coverage
from .panorama_session import CANONICAL_DIRECTIONS
from .storage import read_json, write_json


PANORAMA_WORKBENCH_SCHEMA = "video_knowledge_pipeline.panorama_workbench.v1"
PANORAMA_CAMERA_HINTS_SCHEMA = "video_knowledge_pipeline.panorama_camera_hints.v1"


def panorama_panel_html(summary: Mapping[str, Any]) -> str:
    if summary.get("status") == "not_required":
        return '<div class="muted">当前 Bundle 未声明全景投影；普通视频流程不受影响。</div>'
    profile = summary.get("profile") if isinstance(summary.get("profile"), Mapping) else {}
    coverage = summary.get("coverage") if isinstance(summary.get("coverage"), Mapping) else {}
    faces = summary.get("face_grid") if isinstance(summary.get("face_grid"), list) else []
    face_html = "".join(
        '<figure class="artifact" style="margin:0"><img loading="lazy" style="width:100%;aspect-ratio:1;object-fit:cover" '
        f'src="{html.escape(str(row.get("href") or ""), quote=True)}" alt="{html.escape(str(row.get("direction") or ""))}">'
        f'<figcaption>{html.escape(str(row.get("direction") or ""))}</figcaption></figure>'
        for row in faces
        if isinstance(row, Mapping) and row.get("href")
    )
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), Mapping) else {}
    links = " ".join(
        f'<a href="{html.escape(str(href), quote=True)}">{html.escape(str(key))}</a>'
        for key, href in artifacts.items()
        if href
    )
    return (
        '<div class="closure-grid">'
        f'<div class="closure-metric"><span class="muted">投影</span><strong>{html.escape(str(profile.get("projection") or "unknown"))}</strong></div>'
        f'<div class="closure-metric"><span class="muted">几何覆盖</span><strong>{int(coverage.get("geometric_complete_segments") or 0)}/{int(coverage.get("expected_segments") or 0)}</strong></div>'
        f'<div class="closure-metric"><span class="muted">语义覆盖</span><strong>{int(coverage.get("semantic_covered_segments") or 0)}</strong></div>'
        f'<div class="closure-metric"><span class="muted">时序覆盖</span><strong>{int(coverage.get("temporal_complete_segments") or 0)}/{int(coverage.get("temporal_expected_segments") or 0)}</strong></div>'
        '</div>'
        '<div class="artifact-grid" style="grid-template-columns:repeat(3,minmax(0,1fr));margin:8px 0">'
        + face_html
        + '</div><div class="muted">六面只代表几何覆盖；局部视口不能证明全局不存在。相机提示始终 candidate_only / review_required。</div>'
        + f'<div style="margin-top:8px">{links}</div>'
    )


def export_panorama_workbench(bundle_dir: str | Path, *, write: bool = True) -> dict[str, Any]:
    """Build the panorama card consumed by the existing unified workbench."""

    root = Path(bundle_dir).expanduser().resolve()
    manifest = _mapping(root / "manifest.json")
    timeline_value = read_json(root / "timeline.json")
    timeline = [dict(row) for row in timeline_value if isinstance(row, Mapping)] if isinstance(timeline_value, list) else []
    profile = manifest.get("panorama_profile") if isinstance(manifest.get("panorama_profile"), Mapping) else {}
    run_id = str(((manifest.get("panorama_projection_plan") or {}).get("run_id") if isinstance(manifest.get("panorama_projection_plan"), Mapping) else "") or "")
    run_root = root / "runs" / f"panorama-{run_id}" if run_id else root / "runs" / "panorama-unselected"
    pack = _mapping(run_root / "viewport-pack.json")
    analysis = _mapping(root / "runs" / f"panorama-analysis-{run_id}" / "analysis-plan.json")
    coverage = panorama_bundle_coverage(manifest, timeline)
    faces = _face_grid(root, pack)
    model_rows = _model_rows(timeline)
    hints = build_panorama_camera_hints(profile=profile, timeline=timeline)
    artifacts = {
        "projection_plan": _href(root, run_root / "projection-plan.json"),
        "viewport_pack": _href(root, run_root / "viewport-pack.json"),
        "analysis_plan": _href(root, root / "runs" / f"panorama-analysis-{run_id}" / "analysis-plan.json"),
        "raw_result": _manifest_href(root, manifest, "codex_vision_session_raw_result_json"),
        "validation": _manifest_href(root, manifest, "codex_vision_session_import_validation_json"),
        "audit": _href(root, root / "vision-analysis-runs.md"),
        "restore": _href(root, root / "vision-restore-plan.md"),
        "camera_hints": "exports/panorama-camera-hints.json",
    }
    result = {
        "schema": PANORAMA_WORKBENCH_SCHEMA,
        "status": coverage.get("status") if coverage.get("required") else "not_required",
        "run_id": run_id,
        "profile": {
            "projection": str(profile.get("projection") or "unknown"),
            "metadata_confidence": str(profile.get("metadata_confidence") or "unavailable"),
            "profile_sha256": str(profile.get("profile_sha256") or ""),
        },
        "coverage": coverage,
        "face_grid": faces,
        "analysis": {
            "semantic_task_count": int(analysis.get("semantic_task_count") or 0),
            "temporal_task_count": int(analysis.get("temporal_task_count") or 0),
            "review_gaps": list(analysis.get("review_gaps") or []),
        },
        "models": model_rows,
        "camera_hints": hints,
        "limitations": [
            "Six projected faces prove geometric coverage only; semantic and temporal coverage are separate.",
            "Viewport-local facts cannot be promoted to segment-global absence claims.",
            "Camera hints are candidate-only and are not a rendered or accepted camera path.",
        ],
        "artifacts": artifacts,
        "generated_at": now_iso(),
    }
    if write:
        exports = root / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        write_json(root / "panorama-workbench.json", result)
        (root / "panorama-workbench.md").write_text(_render_markdown(result), encoding="utf-8")
        write_json(exports / "panorama-camera-hints.json", hints)
        manifest["panorama_workbench_json"] = "panorama-workbench.json"
        manifest["panorama_workbench_markdown"] = "panorama-workbench.md"
        manifest["panorama_camera_hints_json"] = "exports/panorama-camera-hints.json"
        write_json(root / "manifest.json", manifest)
    return result


def build_panorama_camera_hints(
    *,
    profile: Mapping[str, Any],
    timeline: list[dict[str, Any]],
) -> dict[str, Any]:
    hints: list[dict[str, Any]] = []
    for position, item in enumerate(timeline, start=1):
        index = int(item.get("index") or position)
        panorama = item.get("panorama_visual_understanding")
        sessions = panorama.get("sessions") if isinstance(panorama, Mapping) else []
        for session in sessions or []:
            if not isinstance(session, Mapping):
                continue
            level = str(session.get("evidence_level") or "")
            if level not in {"confirmed", "inferred"}:
                continue
            ranked = sorted(
                [dict(row) for row in session.get("viewport_observations") or [] if isinstance(row, Mapping)],
                key=lambda row: (-float(row.get("salience_score") or 0.0), CANONICAL_DIRECTIONS.index(str(row.get("direction") or "front"))),
            )
            for row in ranked[:3]:
                hints.append(
                    {
                        "hint_id": _hint_id(index, session, row),
                        "timeline_index": index,
                        "timestamp_seconds": float(session.get("timestamp_seconds") or 0.0),
                        "yaw": float(row.get("yaw") or 0.0),
                        "pitch": float(row.get("pitch") or 0.0),
                        "roll": float(row.get("roll") or 0.0),
                        "h_fov": float(row.get("h_fov") or 90.0),
                        "v_fov": float(row.get("v_fov") or 90.0),
                        "direction": str(row.get("direction") or ""),
                        "score": float(row.get("salience_score") or 0.5),
                        "evidence_level": level,
                        "evidence_refs": [
                            {
                                "viewport_id": str(row.get("viewport_id") or ""),
                                "viewport_pack_sha256": str(session.get("viewport_pack_sha256") or ""),
                            }
                        ],
                        "uncertainty": "inferred visual evidence" if level == "inferred" else "candidate camera choice requires human review",
                        "status": "candidate_only",
                        "review_required": True,
                    }
                )
    payload: dict[str, Any] = {
        "schema": PANORAMA_CAMERA_HINTS_SCHEMA,
        "source_projection": str(profile.get("projection") or "unknown"),
        "contract_status": "integration_proposal",
        "candidate_only": True,
        "review_required": True,
        "render_authorized": False,
        "source_paths_exported": False,
        "hints": hints,
        "consumer": "mvp_video_existing_smart_reframe_after_owner_acceptance",
    }
    payload["camera_hints_sha256"] = _sha256_json(payload)
    return payload


def _face_grid(root: Path, pack: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in pack.get("viewports") or [] if isinstance(row, Mapping)]
    if not rows:
        return []
    first_index = min(int(row.get("timeline_index") or 0) for row in rows)
    first_sample = min(int(row.get("sample_index") or 0) for row in rows if int(row.get("timeline_index") or 0) == first_index)
    by_direction = {
        str(row.get("direction") or ""): row
        for row in rows
        if int(row.get("timeline_index") or 0) == first_index and int(row.get("sample_index") or 0) == first_sample
    }
    return [
        {
            "direction": direction,
            "href": _safe_pack_href(root, by_direction.get(direction, {}).get("path")),
            "viewport_id": str(by_direction.get(direction, {}).get("viewport_id") or ""),
            "sha256": str(by_direction.get(direction, {}).get("sha256") or ""),
        }
        for direction in CANONICAL_DIRECTIONS
    ]


def _model_rows(timeline: list[dict[str, Any]]) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in timeline:
        panorama = item.get("panorama_visual_understanding")
        sessions = panorama.get("sessions") if isinstance(panorama, Mapping) else []
        for session in sessions or []:
            if not isinstance(session, Mapping):
                continue
            key = (
                str(session.get("executor") or ""),
                str(session.get("model_actual") or ""),
                str(session.get("evidence_level") or ""),
            )
            if key not in seen:
                seen.add(key)
                values.append({"executor": key[0], "model_actual": key[1], "evidence_level": key[2]})
    return values


def _render_markdown(result: Mapping[str, Any]) -> str:
    coverage = result.get("coverage") if isinstance(result.get("coverage"), Mapping) else {}
    lines = [
        "# VKP Panorama Workbench Card",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Projection: `{(result.get('profile') or {}).get('projection', 'unknown')}`",
        f"- Geometric complete segments: `{coverage.get('geometric_complete_segments', 0)}`",
        f"- Semantic covered segments: `{coverage.get('semantic_covered_segments', 0)}`",
        f"- Temporal complete segments: `{coverage.get('temporal_complete_segments', 0)}`",
        f"- Camera hints: `{len((result.get('camera_hints') or {}).get('hints') or [])}` candidate-only",
        "",
        "## Limitations",
        "",
        *[f"- {value}" for value in result.get("limitations") or []],
        "",
        "## Artifacts",
        "",
        *[f"- `{key}`: `{value}`" for key, value in (result.get("artifacts") or {}).items() if value],
        "",
    ]
    return "\n".join(lines)


def _mapping(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _manifest_href(root: Path, manifest: Mapping[str, Any], key: str) -> str:
    value = str(manifest.get(key) or "")
    return _href(root, root / value) if value else ""


def _safe_pack_href(root: Path, value: Any) -> str:
    path = (root / str(value or "")).resolve()
    return _href(root, path)


def _href(root: Path, path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) and resolved.exists() else ""


def _hint_id(index: int, session: Mapping[str, Any], viewport: Mapping[str, Any]) -> str:
    identity = {
        "index": index,
        "session_id": session.get("session_id"),
        "viewport_id": viewport.get("viewport_id"),
        "timestamp": session.get("timestamp_seconds"),
    }
    return "camera-" + _sha256_json(identity)[:20]


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
