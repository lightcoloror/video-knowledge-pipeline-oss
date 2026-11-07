from __future__ import annotations

import importlib.metadata
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .file_hash import sha256_file
from .models import now_iso
from .path_defaults import project_root, source_reviews_root
from .shot_language_analysis import load_shot_facts
from .storage import bundle_write_lock, read_json, write_json
from .transcript import format_timestamp


SCENE_SCHEMA = "video_knowledge_pipeline.semantic_scene_plan.v1"
BEAT_SCHEMA = "video_knowledge_pipeline.story_beat_plan.v1"
SCENE_PATH = "exports/semantic-scene-plan.json"
BEAT_PATH = "exports/story-beat-plan.json"
MARKDOWN_PATH = "exports/filmed-structure.md"
RUPTURES_VERSION = "1.1.10"
RUPTURES_COMMIT = "a3f8c437edf7d54c1a8f90aaa72638363a011765"
DEFAULT_RUPTURES_VENDOR = project_root() / ".local" / "vendor" / "ruptures-1.1.10"
DEFAULT_RUPTURES_SOURCE = source_reviews_root() / "vkp-pullfilm-v2-20260801" / "ruptures"
BEAT_ROLES = {
    "setup",
    "goal",
    "action",
    "change",
    "conflict",
    "result",
    "payoff",
    "unknown",
}


def build_filmed_structure_plan(
    bundle_dir: str | Path,
    *,
    shot_embeddings_json: str | Path | None = None,
    story_evidence_json: str | Path | None = None,
    local_story_route_id: str = "",
    penalty: float | None = None,
    min_scene_shots: int = 2,
    jump: int = 1,
    write: bool = True,
    _change_point_runner: Callable[[list[list[float]], float, int, int], list[int]] | None = None,
) -> dict[str, Any]:
    """Group shot facts with ruptures PELT and build evidence-only beats."""

    root = Path(bundle_dir).expanduser().resolve()
    facts = load_shot_facts(root)
    shots = [
        row
        for row in facts.get("shots") or []
        if isinstance(row, dict)
    ]
    if not shots:
        result = _blocked(root)
        if write:
            _write_result(root, result)
        return result

    embeddings, embedding_provenance = _load_embeddings(
        shot_embeddings_json,
        shots,
    )
    features = [
        _feature_vector(row, embeddings.get(str(row.get("shot_id") or "")))
        for row in shots
    ]
    effective_penalty = float(
        penalty
        if penalty is not None
        else max(1.0, math.log(max(len(shots), 2)) * 1.5)
    )
    runner = _change_point_runner or _run_pelt
    if len(shots) < max(2, int(min_scene_shots)) * 2:
        breakpoints = [len(shots)]
        segmentation_status = "single_scene_short_sequence"
    else:
        breakpoints = runner(
            features,
            effective_penalty,
            max(2, int(min_scene_shots)),
            max(1, int(jump)),
        )
        breakpoints = _validated_breakpoints(breakpoints, len(shots))
        segmentation_status = "completed"
    scenes = _scenes_from_breakpoints(shots, breakpoints)
    story_evidence, story_provenance = _load_story_evidence(
        story_evidence_json
    )
    beats = _story_beats(
        scenes,
        story_evidence,
        local_story_route_id=local_story_route_id,
    )
    embedding_ready = len(embeddings) == len(shots)
    scene_status = (
        "completed"
        if embedding_ready
        else "degraded_missing_bge_m3_embeddings"
    )
    scene_result = {
        "schema": SCENE_SCHEMA,
        "status": scene_status,
        "ok": bool(scenes),
        "bundle_dir": str(root),
        "method": {
            "library": "ruptures",
            "version": RUPTURES_VERSION,
            "source_commit": RUPTURES_COMMIT,
            "algorithm": "Pelt",
            "model": "rbf",
            "penalty": effective_penalty,
            "min_scene_shots": max(2, int(min_scene_shots)),
            "jump": max(1, int(jump)),
            "segmentation_status": segmentation_status,
        },
        "shot_facts_provenance": _artifact_provenance(
            root / "exports" / "shot-facts.json"
        ),
        "embedding_provenance": embedding_provenance,
        "scene_count": len(scenes),
        "scenes": scenes,
        "candidate_only": True,
        "timeline_mutated": False,
        "updated_at": now_iso(),
    }
    beat_result = {
        "schema": BEAT_SCHEMA,
        "status": (
            "completed"
            if beats and all(row["status"] != "unavailable" for row in beats)
            else "needs_local_story_evidence"
        ),
        "ok": bool(beats),
        "bundle_dir": str(root),
        "allowed_roles": sorted(BEAT_ROLES),
        "local_story_route_id": str(local_story_route_id or ""),
        "story_evidence_provenance": story_provenance,
        "beat_count": len(beats),
        "beats": beats,
        "position_only_role_inference": False,
        "candidate_only": True,
        "timeline_mutated": False,
        "updated_at": now_iso(),
    }
    result = {
        "schema": "video_knowledge_pipeline.filmed_structure_plan.v1",
        "status": (
            "completed"
            if scene_result["status"] == "completed"
            and beat_result["status"] == "completed"
            else "degraded"
        ),
        "ok": True,
        "bundle_dir": str(root),
        "content_profile": "filmed-v1",
        "semantic_scene_plan": scene_result,
        "story_beat_plan": beat_result,
        "highlight_plan": {
            "adapter": "lighthouse_cg_detr",
            "eligible_scene_ids": [
                row["scene_id"]
                for row in scenes
                if float(row["duration"]) <= 150.0
            ],
            "maximum_scene_duration_seconds": 150.0,
            "execute": False,
            "automatic_execution": False,
            "whole_video_input_allowed": False,
        },
        "operator_boundary": {
            "local_only": True,
            "no_whole_video_vlm": True,
            "no_position_only_story_roles": True,
            "no_automatic_highlight_execution": True,
            "no_remote_fallback": True,
            "timeline_mutated": False,
        },
        "artifacts": {
            "semantic_scenes": SCENE_PATH,
            "story_beats": BEAT_PATH,
            "markdown": MARKDOWN_PATH,
        },
        "updated_at": now_iso(),
    }
    if write:
        _write_result(root, result)
    return result


def _run_pelt(
    features: list[list[float]],
    penalty: float,
    min_size: int,
    jump: int,
) -> list[int]:
    vendor = DEFAULT_RUPTURES_VENDOR.resolve()
    source_available = DEFAULT_RUPTURES_SOURCE.is_dir()
    vendor_available = vendor.is_dir()
    if source_available or vendor_available:
        if not (source_available and vendor_available):
            raise RuntimeError(
                "reviewed ruptures source/vendor pair is incomplete; no implicit "
                "fallback is allowed"
            )
        source_commit = _git_commit(DEFAULT_RUPTURES_SOURCE)
        if source_commit != RUPTURES_COMMIT:
            raise RuntimeError(
                "ruptures source commit mismatch: "
                f"expected={RUPTURES_COMMIT} actual={source_commit}"
            )
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
    try:
        installed_version = importlib.metadata.version("ruptures")
        if installed_version != RUPTURES_VERSION:
            raise RuntimeError(
                "ruptures version mismatch: "
                f"expected={RUPTURES_VERSION} actual={installed_version}"
            )
        import numpy as np
        import ruptures as rpt
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "optional dependency missing: install video-knowledge-pipeline[local] "
            "to enable filmed structure segmentation"
        ) from exc

    signal = np.asarray(features, dtype=float)
    return [
        int(value)
        for value in rpt.Pelt(
            model="rbf",
            min_size=max(2, int(min_size)),
            jump=max(1, int(jump)),
        ).fit(signal).predict(pen=float(penalty))
    ]


def _feature_vector(
    shot: dict[str, Any],
    embedding: list[float] | None,
) -> list[float]:
    fields = shot.get("fields") if isinstance(shot.get("fields"), dict) else {}
    shot_type = _field_value(fields, "shot_type")
    movement = _field_value(fields, "camera_movement")
    dialogue = str(_field_value(fields, "dialogue_or_narration") or "")
    screen = str(_field_value(fields, "screen_text") or "")
    duration = max(0.0, float(shot.get("end") or 0.0) - float(shot.get("start") or 0.0))
    base = [
        min(duration / 30.0, 1.0),
        min(len(dialogue) / 400.0, 1.0),
        min(len(screen) / 200.0, 1.0),
    ]
    base.extend(
        1.0 if shot_type == value else 0.0
        for value in ("wide", "medium", "close_up", "extreme_close_up")
    )
    base.extend(
        1.0 if movement == value else 0.0
        for value in ("static", "pan_or_tilt", "tracking", "handheld", "zoom")
    )
    if embedding:
        base.extend(float(value) for value in embedding)
    return base


def _load_embeddings(
    value: str | Path | None,
    shots: list[dict[str, Any]],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    if not value:
        return {}, {
            "status": "unavailable",
            "model": "BAAI/bge-m3",
            "missing_evidence": ["shot_embeddings_json"],
        }
    path = Path(value).expanduser().resolve()
    payload = read_json(path)
    rows = payload.get("shots") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("shot embeddings must contain a shots array")
    result: dict[str, list[float]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("embedding"), list):
            continue
        result[str(row.get("shot_id") or "")] = [
            float(item) for item in row["embedding"]
        ]
    expected = {str(row.get("shot_id") or "") for row in shots}
    missing = sorted(expected - set(result))
    return result, {
        "status": "complete" if not missing else "partial",
        "path": str(path),
        "sha256": sha256_file(path),
        "model": str(payload.get("model") or "BAAI/bge-m3"),
        "missing_shot_ids": missing,
    }


def _scenes_from_breakpoints(
    shots: list[dict[str, Any]],
    breakpoints: list[int],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    start_index = 0
    for end_index in breakpoints:
        group = shots[start_index:end_index]
        if not group:
            continue
        start = float(group[0].get("start") or 0.0)
        end = float(group[-1].get("end") or start)
        ids = [str(row.get("shot_id") or "") for row in group]
        evidence_ids = sorted(
            {
                evidence
                for row in group
                for field in (row.get("fields") or {}).values()
                if isinstance(field, dict)
                for evidence in field.get("evidence_ids") or []
            }
        )
        modalities = sorted(
            {
                _source_modality(str(field.get("source") or ""))
                for row in group
                for field in (row.get("fields") or {}).values()
                if isinstance(field, dict) and field.get("status") != "unavailable"
            }
            - {""}
        )
        result.append(
            {
                "scene_id": f"semantic-scene-{len(result) + 1:04d}",
                "index": len(result) + 1,
                "start": start,
                "end": end,
                "duration": round(end - start, 6),
                "start_time": format_timestamp(start),
                "end_time": format_timestamp(end),
                "shot_ids": ids,
                "evidence_ids": evidence_ids,
                "source_modalities": modalities,
                "status": "inferred",
                "candidate_only": True,
            }
        )
        start_index = end_index
    return result


def _story_beats(
    scenes: list[dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
    *,
    local_story_route_id: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for scene in scenes:
        scene_id = scene["scene_id"]
        source = evidence.get(scene_id, {})
        role = str(source.get("role") or "unknown").strip().lower()
        cited = [str(value) for value in source.get("evidence_ids") or []]
        valid_evidence = sorted(set(cited).intersection(scene["evidence_ids"]))
        if role not in BEAT_ROLES:
            raise ValueError(f"unsupported story beat role: {role}")
        if role == "unknown" or not valid_evidence:
            status = "unavailable"
            role = "unknown"
            missing = [
                "local Qwen story-beat decision with scene evidence_ids"
            ]
            valid_evidence = []
        else:
            status = "inferred"
            missing = []
        result.append(
            {
                "beat_id": f"story-beat-{len(result) + 1:04d}",
                "scene_id": scene_id,
                "start": scene["start"],
                "end": scene["end"],
                "role": role,
                "summary": str(source.get("summary") or "") if status != "unavailable" else "",
                "status": status,
                "confidence": (
                    max(0.0, min(float(source.get("confidence") or 0.0), 1.0))
                    if status != "unavailable"
                    else 0.0
                ),
                "evidence_ids": valid_evidence,
                "missing_evidence": missing,
                "source": (
                    f"local_story_route:{local_story_route_id}"
                    if local_story_route_id
                    else "missing_local_story_route"
                ),
                "candidate_only": True,
            }
        )
    return result


def _load_story_evidence(
    value: str | Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not value:
        return {}, {
            "status": "not_provided",
            "missing_evidence": ["story_evidence_json"],
        }
    path = Path(value).expanduser().resolve()
    payload = read_json(path)
    rows = payload.get("scenes") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("story evidence must contain a scenes array")
    return (
        {
            str(row.get("scene_id") or ""): row
            for row in rows
            if isinstance(row, dict)
        },
        {
            "status": "loaded",
            "path": str(path),
            "sha256": sha256_file(path),
        },
    )


def _validated_breakpoints(values: list[int], count: int) -> list[int]:
    result = sorted({int(value) for value in values if 0 < int(value) <= count})
    if not result or result[-1] != count:
        result.append(count)
    return result


def _field_value(fields: dict[str, Any], key: str) -> Any:
    value = fields.get(key)
    return value.get("value") if isinstance(value, dict) else None


def _source_modality(source: str) -> str:
    if "transcript" in source or "audio" in source:
        return "asr"
    if "visual_text" in source or "ocr" in source:
        return "ocr"
    if "visual" in source or "frame" in source or "dinov2" in source or "flow" in source:
        return "visual"
    return ""


def _artifact_provenance(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    return {
        "status": "loaded",
        "path": str(path),
        "sha256": sha256_file(path),
    }


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "cannot verify ruptures source")
    return result.stdout.strip()


def _blocked(root: Path) -> dict[str, Any]:
    return {
        "schema": "video_knowledge_pipeline.filmed_structure_plan.v1",
        "status": "blocked_missing_shot_facts",
        "ok": False,
        "bundle_dir": str(root),
        "content_profile": "filmed-v1",
        "semantic_scene_plan": {
            "schema": SCENE_SCHEMA,
            "status": "blocked_missing_shot_facts",
            "scene_count": 0,
            "scenes": [],
        },
        "story_beat_plan": {
            "schema": BEAT_SCHEMA,
            "status": "blocked_missing_shot_facts",
            "beat_count": 0,
            "beats": [],
        },
        "operator_boundary": {
            "no_chapter_or_timeline_range_fallback": True,
            "no_position_only_story_roles": True,
            "timeline_mutated": False,
        },
        "artifacts": {
            "semantic_scenes": SCENE_PATH,
            "story_beats": BEAT_PATH,
            "markdown": MARKDOWN_PATH,
        },
        "updated_at": now_iso(),
    }


def _write_result(root: Path, result: dict[str, Any]) -> None:
    manifest_path = root / "manifest.json"
    with bundle_write_lock(root, operation="filmed_structure", timeout_seconds=1.0):
        write_json(root / SCENE_PATH, result["semantic_scene_plan"])
        write_json(root / BEAT_PATH, result["story_beat_plan"])
        (root / MARKDOWN_PATH).write_text(
            _render_markdown(result),
            encoding="utf-8",
        )
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        if isinstance(manifest, dict):
            manifest["semantic_scene_plan_json"] = SCENE_PATH
            manifest["story_beat_plan_json"] = BEAT_PATH
            manifest["filmed_structure_markdown"] = MARKDOWN_PATH
            manifest["filmed_structure_summary"] = {
                "status": result["status"],
                "scene_count": result["semantic_scene_plan"].get("scene_count", 0),
                "beat_count": result["story_beat_plan"].get("beat_count", 0),
                "updated_at": result["updated_at"],
            }
            write_json(manifest_path, manifest)


def _render_markdown(result: dict[str, Any]) -> str:
    scenes = result["semantic_scene_plan"]
    beats = {
        row["scene_id"]: row
        for row in result["story_beat_plan"].get("beats") or []
    }
    lines = [
        "# Filmed Structure",
        "",
        f"- Status: {result.get('status')}",
        f"- Scenes: {scenes.get('scene_count', 0)}",
        "- Position-only story inference: false",
        "- Whole-video VLM: disabled",
        "",
    ]
    for scene in scenes.get("scenes") or []:
        beat = beats.get(scene["scene_id"], {})
        lines.append(
            f"- {scene['scene_id']} {scene['start_time']} - {scene['end_time']}"
        )
        lines.append(f"  - story_role: {beat.get('role', 'unknown')}")
        lines.append(f"  - evidence_ids: {len(scene.get('evidence_ids') or [])}")
    return "\n".join(lines).rstrip() + "\n"
