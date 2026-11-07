from __future__ import annotations

from pathlib import Path
from typing import Any

from .highlight_detection_adapter import run_highlight_detection
from .filmed_structure import build_filmed_structure_plan
from .models import now_iso
from .scene_detection_adapter import run_scene_detection
from .semantic_chapter_plan import build_semantic_chapter_plan
from .storage import bundle_write_lock, read_json, write_json


SCHEMA = "video_knowledge_pipeline.video_structure.v1"


def build_video_structure(
    bundle_dir: str | Path,
    *,
    media_path: str | Path | None = None,
    title: str = "",
    input_pack: str | Path | None = None,
    run_shot_detection: bool = True,
    shot_detector: str = "adaptive",
    shot_threshold: float | None = None,
    shot_source_root: str | Path | None = None,
    highlight_query: str = "找出对内容理解或后续剪辑最重要的片段",
    highlight_predictions_json: str | Path | None = None,
    content_profile: str = "course-v1",
    shot_embeddings_json: str | Path | None = None,
    story_evidence_json: str | Path | None = None,
    local_story_route_id: str = "",
    write: bool = True,
) -> dict[str, Any]:
    """Build an auditable video-structure pack without native whole-video VLM use."""

    root = Path(bundle_dir).expanduser().resolve()
    profile = str(content_profile or "course-v1").strip().lower()
    if profile == "filmed-v1":
        return _build_filmed_profile(
            root,
            media_path=media_path,
            highlight_query=highlight_query,
            highlight_predictions_json=highlight_predictions_json,
            shot_embeddings_json=shot_embeddings_json,
            story_evidence_json=story_evidence_json,
            local_story_route_id=local_story_route_id,
            write=write,
        )
    if profile != "course-v1":
        raise ValueError("content_profile must be course-v1 or filmed-v1")
    shot_result = _shot_result(
        root,
        media_path=media_path,
        enabled=run_shot_detection,
        detector=shot_detector,
        threshold=shot_threshold,
        source_root=shot_source_root,
        write=write,
    )
    semantic = build_semantic_chapter_plan(
        root,
        title=title,
        input_pack=input_pack,
        chapter_mode="semantic",
        write=write,
    )
    semantic_scenes = _semantic_scenes(semantic)
    storyline = _storyline(semantic_scenes)
    highlights = run_highlight_detection(
        root,
        query=highlight_query,
        media_path=media_path,
        predictions_json=highlight_predictions_json,
        execute=False,
        write=write,
    )
    semantic_ok = bool(semantic.get("ok"))
    status = "completed" if semantic_ok else "needs_timeline_evidence"
    result = {
        "schema": SCHEMA,
        "status": status,
        "ok": semantic_ok,
        "bundle_dir": str(root),
        "shot_boundary_detection": shot_result,
        "semantic_scene_segmentation": {
            "status": str(semantic.get("status") or "missing"),
            "scene_count": len(semantic_scenes),
            "scenes": semantic_scenes,
            "source_artifact": "exports/semantic-chapter-plan.json",
            "candidate_only": True,
        },
        "storyline_structure": {
            "status": "needs_story_evidence" if storyline else "missing_semantic_scenes",
            "item_count": len(storyline),
            "items": storyline,
            "candidate_only": True,
        },
        "highlight_detection": highlights,
        "operator_boundary": {
            "native_whole_video_understanding_enabled": False,
            "whole_video_vlm_upload_allowed": False,
            "cloud_calls_made": 0,
            "shot_detection_local_only": True,
            "semantic_scene_fusion_local_only": True,
            "storyline_is_candidate_evidence": True,
            "highlights_require_explicit_local_execution_or_saved_import": True,
            "automatic_local_remote_fallback": False,
        },
        "artifacts": {
            "json": "exports/video-structure.json",
            "markdown": "exports/video-structure.md",
            "shot_boundaries": "exports/scene-detection.json" if run_shot_detection else "",
            "semantic_scenes": "exports/semantic-chapter-plan.json",
            "highlights": "exports/highlight-detection.json",
        },
        "updated_at": now_iso(),
    }
    if write:
        _write_result(root, result)
    return result


def _build_filmed_profile(
    root: Path,
    *,
    media_path: str | Path | None,
    highlight_query: str,
    highlight_predictions_json: str | Path | None,
    shot_embeddings_json: str | Path | None,
    story_evidence_json: str | Path | None,
    local_story_route_id: str,
    write: bool,
) -> dict[str, Any]:
    filmed = build_filmed_structure_plan(
        root,
        shot_embeddings_json=shot_embeddings_json,
        story_evidence_json=story_evidence_json,
        local_story_route_id=local_story_route_id,
        write=write,
    )
    scene_plan = filmed["semantic_scene_plan"]
    beat_plan = filmed["story_beat_plan"]
    scenes = [
        {
            **row,
            "title_hint": "",
            "timeline_indexes": [],
            "evidence_counts": {
                modality: sum(1 for value in row.get("source_modalities") or [] if value == modality)
                for modality in ("asr", "ocr", "visual")
            },
        }
        for row in scene_plan.get("scenes") or []
    ]
    beats_by_scene = {
        row["scene_id"]: row for row in beat_plan.get("beats") or []
    }
    storyline = [
        {
            "index": index,
            "scene_index": scene["index"],
            "scene_id": scene["scene_id"],
            "start": scene["start"],
            "end": scene["end"],
            "role": beats_by_scene.get(scene["scene_id"], {}).get("role", "unknown"),
            "status": beats_by_scene.get(scene["scene_id"], {}).get("status", "unavailable"),
            "evidence_ids": beats_by_scene.get(scene["scene_id"], {}).get("evidence_ids", []),
            "candidate_only": True,
        }
        for index, scene in enumerate(scenes, start=1)
    ]
    highlights = run_highlight_detection(
        root,
        query=highlight_query,
        media_path=media_path,
        predictions_json=highlight_predictions_json,
        execute=False,
        write=write,
    )
    result = {
        "schema": SCHEMA,
        "status": filmed["status"],
        "ok": filmed["ok"],
        "content_profile": "filmed-v1",
        "bundle_dir": str(root),
        "shot_boundary_detection": {
            "status": scene_plan.get("shot_facts_provenance", {}).get("status", "missing"),
            "backend": "technical_shot_boundaries.v1",
            "boundary_count": max(0, sum(len(row.get("shot_ids") or []) for row in scenes) - 1),
            "candidate_only": True,
            "local_only": True,
        },
        "semantic_scene_segmentation": {
            "status": scene_plan["status"],
            "scene_count": len(scenes),
            "scenes": scenes,
            "source_artifact": "exports/semantic-scene-plan.json",
            "candidate_only": True,
        },
        "storyline_structure": {
            "status": beat_plan["status"],
            "item_count": len(storyline),
            "items": storyline,
            "position_only_role_inference": False,
            "candidate_only": True,
        },
        "highlight_detection": highlights,
        "operator_boundary": {
            "native_whole_video_understanding_enabled": False,
            "whole_video_vlm_upload_allowed": False,
            "cloud_calls_made": 0,
            "shot_detection_local_only": True,
            "semantic_scene_fusion_local_only": True,
            "storyline_is_candidate_evidence": True,
            "automatic_local_remote_fallback": False,
        },
        "artifacts": {
            "json": "exports/video-structure.json",
            "markdown": "exports/video-structure.md",
            "shot_boundaries": "exports/technical-shot-boundaries.json",
            "semantic_scenes": "exports/semantic-scene-plan.json",
            "story_beats": "exports/story-beat-plan.json",
            "highlights": "exports/highlight-detection.json",
        },
        "updated_at": now_iso(),
    }
    if write:
        _write_result(root, result)
    return result

def _shot_result(
    root: Path,
    *,
    media_path: str | Path | None,
    enabled: bool,
    detector: str,
    threshold: float | None,
    source_root: str | Path | None,
    write: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "status": "not_run",
            "backend": "",
            "boundary_count": 0,
            "candidate_only": True,
            "local_only": True,
        }
    try:
        result = run_scene_detection(
            root,
            media_path=media_path,
            detector=detector,
            threshold=threshold,
            source_root=source_root,
            write=write,
        )
        return {
            "status": result.get("status"),
            "backend": result.get("backend"),
            "boundary_count": result.get("boundary_count", 0),
            "fallback_reason": result.get("fallback_reason", ""),
            "candidate_only": True,
            "local_only": True,
        }
    except Exception as exc:
        return {
            "status": "needs_input",
            "backend": "",
            "boundary_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "candidate_only": True,
            "local_only": True,
        }


def _semantic_scenes(plan: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in plan.get("chapters") or []:
        if not isinstance(row, dict):
            continue
        result.append(
            {
                "index": int(row.get("index") or len(result) + 1),
                "start": float(row.get("start") or 0.0),
                "end": float(row.get("end") or 0.0),
                "start_time": str(row.get("start_time") or ""),
                "end_time": str(row.get("end_time") or ""),
                "title_hint": str(row.get("title_hint") or ""),
                "timeline_indexes": list(row.get("timeline_indexes") or []),
                "boundary_reasons": list(row.get("boundary_reasons") or []),
                "evidence_counts": dict(row.get("evidence_counts") or {}),
                "source": "semantic_chapter_evidence_fusion",
                "candidate_only": True,
            }
        )
    return result


def _storyline(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve scene evidence without assigning roles from list position."""

    return [
        {
            "index": position,
            "scene_index": scene["index"],
            "start": scene["start"],
            "end": scene["end"],
            "role": "unknown",
            "status": "unavailable",
            "title_hint": scene["title_hint"],
            "evidence_counts": scene["evidence_counts"],
            "evidence_ids": [],
            "missing_evidence": ["evidence-bound story beat decision"],
            "candidate_only": True,
        }
        for position, scene in enumerate(scenes, start=1)
    ]
def _write_result(root: Path, result: dict[str, Any]) -> None:
    exports = root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    with bundle_write_lock(root, operation="video_structure", timeout_seconds=1.0):
        write_json(exports / "video-structure.json", result)
        (exports / "video-structure.md").write_text(_render_markdown(result), encoding="utf-8")
        write_json(
            root / "mcp-video-structure.args.json",
            {
                "bundle_dir": str(root),
                "media_path": "",
                "title": "",
                "run_shot_detection": True,
                "highlight_query": result["highlight_detection"].get("query", ""),
                "write": True,
            },
        )
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        if isinstance(manifest, dict):
            manifest["video_structure_json"] = "exports/video-structure.json"
            manifest["video_structure_markdown"] = "exports/video-structure.md"
            manifest["mcp_video_structure_args"] = "mcp-video-structure.args.json"
            manifest["video_structure_summary"] = {
                "status": result["status"],
                "semantic_scene_count": result["semantic_scene_segmentation"]["scene_count"],
                "storyline_item_count": result["storyline_structure"]["item_count"],
                "highlight_count": result["highlight_detection"].get("highlight_count", 0),
                "native_whole_video_understanding_enabled": False,
                "updated_at": result["updated_at"],
            }
            write_json(manifest_path, manifest)


def _render_markdown(result: dict[str, Any]) -> str:
    semantic = result["semantic_scene_segmentation"]
    highlights = result["highlight_detection"]
    lines = [
        "# Video Structure",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Semantic scenes: `{semantic.get('scene_count')}`",
        f"- Storyline items: `{result['storyline_structure'].get('item_count')}`",
        f"- Highlights: `{highlights.get('highlight_count', 0)}` (`{highlights.get('status')}`)",
        "- Native whole-video understanding: `disabled`",
        "",
        "## Semantic scenes and storyline",
        "",
        "| # | Time | Role | Title hint |",
        "| ---: | --- | --- | --- |",
    ]
    roles = {int(row["scene_index"]): row["role"] for row in result["storyline_structure"].get("items") or []}
    for row in semantic.get("scenes") or []:
        lines.append(
            f"| {row.get('index')} | `{row.get('start_time')} - {row.get('end_time')}` | {roles.get(int(row.get('index') or 0), '')} | {_md(row.get('title_hint'))} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")
