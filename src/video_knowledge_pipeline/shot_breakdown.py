from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .file_hash import sha256_file
from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .scene_taxonomy import normalize_taxonomy_value
from .storage import bundle_write_lock, read_json, write_json
from .technical_shot_detection import load_verified_technical_shots
from .shot_language_analysis import load_shot_facts
from .shot_breakdown_exports import render_shot_breakdown_csv, render_shot_breakdown_logseq
from .transcript import format_timestamp


BREAKDOWN_SCHEMA = "video_knowledge_pipeline.shot_breakdown.v2"
FINGERPRINT_SCHEMA = "video_knowledge_pipeline.style_fingerprint.v1"
SCRIPT_SCHEMA = "video_knowledge_pipeline.imitation_script.v1"
READINESS_SCHEMA = "video_knowledge_pipeline.shot_imitation_readiness.v1"

UPSTREAM_REFERENCES = [
    {"project": "keng1304/video-breakdown", "commit": "a8188e148ed07381ee91915d78da3462474c016a"},
    {"project": "wassermanproductions/scriptbreak", "commit": "c457f02ec2f0f34bb31af5289af90dd9216297b5"},
    {"project": "Forget-C/Jellyfish", "commit": "a9678194ddf2d9be3ccbe78d4287d87d5089e123"},
    {"project": "OYYH-Apple/video-storyboard-generator", "commit": "4ccbe8abd80b9a44da43024aec11b2aa41b2bbb4"},
]


def build_shot_breakdown(
    bundle_dir: str | Path,
    *,
    title: str = "",
    reference_analysis_json: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Build candidate-only shot facts and an imitation-script handoff.

    This is a fusion/export layer.  It does not decode media, execute a model,
    call a provider, or mutate Timeline evidence.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_object(root / "manifest.json")
    timeline = _read_list(root / "timeline.json")
    imported = _load_reference_analysis(reference_analysis_json)
    reference_provenance = _reference_analysis_provenance(reference_analysis_json)
    technical_shots, technical_provenance = load_verified_technical_shots(root)
    shot_fact_pack = load_shot_facts(root)
    shot_facts_by_index = {
        int(row.get("index")): row
        for row in shot_fact_pack.get("shots") or []
        if isinstance(row, dict) and row.get("index") not in (None, "")
    }
    ranges = _shot_ranges(technical_shots)
    shots = [
        _build_shot(
            index,
            start,
            end,
            timeline,
            imported,
            structured=shot_facts_by_index.get(index, {}),
        )
        for index, (start, end) in enumerate(ranges, start=1)
    ]
    fingerprint = _style_fingerprint(shots)
    readiness = _readiness(shots)
    script = _imitation_script(shots, fingerprint, readiness)
    status = "completed" if shots else "blocked_missing_technical_shots"
    result = {
        "schema": BREAKDOWN_SCHEMA,
        "status": status,
        "ok": bool(shots),
        "title": title or str(manifest.get("title") or ""),
        "bundle_dir": str(root),
        "input_provenance": {
            "reference_analysis": reference_provenance,
            "technical_shots": technical_provenance,
            "shot_facts": _shot_fact_provenance(root, shot_fact_pack),
        },
        "shot_count": len(shots),
        "shots": shots,
        "style_fingerprint": fingerprint,
        "imitation_script": script,
        "readiness": readiness,
        "source_reuse": {
            "mode": "independent_contract_adaptation",
            "upstream_references": UPSTREAM_REFERENCES,
            "copied_upstream_source": False,
        },
        "operator_boundary": {
            "candidate_only": True,
            "human_confirmation_required_before_export": True,
            "timeline_mutated": False,
            "media_decoded": False,
            "model_calls_made": 0,
            "cloud_calls_made": 0,
            "automatic_local_remote_fallback": False,
            "native_whole_video_understanding_enabled": False,
            "chapter_or_timeline_ranges_used_as_shots": False,
        },
        "artifacts": {
            "breakdown_json": "exports/shot-breakdown.json",
            "breakdown_markdown": "exports/shot-breakdown.md",
            "breakdown_logseq_markdown": "exports/shot-breakdown.logseq.md",
            "breakdown_csv": "exports/shot-breakdown.csv",
            "style_fingerprint_json": "exports/style-fingerprint.json",
            "imitation_script_json": "exports/imitation-script.json",
            "imitation_script_markdown": "exports/imitation-script.md",
            "readiness_json": "exports/shot-imitation-readiness.json",
        },
        "updated_at": now_iso(),
    }
    if write:
        _write_outputs(root, result, reference_analysis_json=reference_analysis_json)
    return result


def _shot_ranges(technical_shots: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Return verified technical shots only.

    Chapters, semantic scenes, Timeline rows and the whole-media duration are
    intentionally excluded. They describe content structure, not camera cuts.
    """

    return _valid_ranges(technical_shots)


def _valid_ranges(items: Iterable[Any]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        start, end = _number(item.get("start")), _number(item.get("end"))
        if end > start >= 0:
            result.append((round(start, 3), round(end, 3)))
    return sorted(set(result))


def _build_shot(
    index: int,
    start: float,
    end: float,
    timeline: list[dict[str, Any]],
    imported: dict[int, dict[str, Any]],
    *,
    structured: dict[str, Any],
) -> dict[str, Any]:
    rows = [row for row in timeline if _overlaps(row, start, end)]
    imported_row = imported.get(index, {})
    transcript = _join_unique(_text(row, "corrected_transcript", "transcript", "raw_text") for row in rows)
    visual_text = _join_unique(_text(row, "corrected_visual_text", "visual_text", "ocr_text") for row in rows)
    temporal = _collect_temporal(rows)
    tags = _list_unique(value for row in rows for value in _tags(row))
    frames = _list_unique(value for row in rows for value in _frames(row))
    shot_type = _taxonomy_candidate("shot_type", rows, imported_row, "shot_type", "shot_size", "camera_shot")
    movement = _taxonomy_candidate("camera_movement", rows, imported_row, "camera_movement", "movement", "type")
    environment = _taxonomy_candidate("environment_type", rows, imported_row, "environment_type", "environment")
    edit_role = _taxonomy_candidate("edit_role", rows, imported_row, "edit_role")
    composition = _dict_candidate(rows, imported_row, "composition")
    color = _dict_candidate(rows, imported_row, "color_profile", "color")
    audio = _dict_candidate(rows, imported_row, "audio_sync", "audio")
    facts = {
        "shot_type": shot_type[0],
        "camera_movement": movement[0],
        "environment_type": environment[0],
        "edit_role": edit_role[0],
        "composition": composition[0],
        "lighting": "",
        "color_profile": color[0],
        "audio": audio[0],
        "subject_action": temporal,
        "tags": tags,
        "dialogue_or_narration": transcript,
        "screen_text": visual_text,
    }
    fact_fields = (
        dict(structured.get("fields") or {})
        if isinstance(structured, dict)
        else {}
    )
    for key in (
        "shot_type",
        "camera_movement",
        "composition",
        "lighting",
        "audio",
        "subject_action",
        "dialogue_or_narration",
        "screen_text",
    ):
        field = fact_fields.get(key)
        if isinstance(field, dict) and field.get("status") != "unavailable":
            if field.get("value") not in (None, "", [], {}):
                facts[key] = field["value"]
    frame_field = fact_fields.get("reference_frames")
    if isinstance(frame_field, dict) and frame_field.get("status") != "unavailable":
        if isinstance(frame_field.get("value"), list):
            frames = [str(value) for value in frame_field["value"] if str(value)]
    provenance = {
        "shot_type": shot_type[1],
        "camera_movement": movement[1],
        "environment_type": environment[1],
        "edit_role": edit_role[1],
        "composition": composition[1],
        "lighting": "missing",
        "color_profile": color[1],
        "audio": audio[1],
        "subject_action": "timeline.temporal_visual_understanding" if temporal else "missing",
        "tags": "timeline.tagger_annotations" if tags else "missing",
        "dialogue_or_narration": "timeline.transcript" if transcript else "missing",
        "screen_text": "timeline.visual_text" if visual_text else "missing",
    }
    for key, field in fact_fields.items():
        if key not in provenance or not isinstance(field, dict):
            continue
        if field.get("status") == "unavailable":
            continue
        if field.get("value") in (None, "", [], {}):
            continue
        source = str(field.get("source") or "unknown")
        provenance[key] = f"shot_facts.v1:{source}"
    unknown = [key for key, value in facts.items() if value in ("", "unknown", {}, [])]
    return {
        "shot_id": f"shot-{index:04d}",
        "index": index,
        "start": start,
        "end": end,
        "duration": round(end - start, 3),
        "start_time": format_timestamp(start),
        "end_time": format_timestamp(end),
        "timeline_indexes": [row.get("index") for row in rows if row.get("index") is not None],
        "reference_frames": frames,
        "facts": facts,
        "field_provenance": provenance,
        "fact_fields": fact_fields,
        "unknown_fields": unknown,
        "candidate_only": True,
        "human_confirmed": False,
    }


def _style_fingerprint(shots: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(row["duration"]) for row in shots]
    total = sum(durations)
    distributions = {
        key: _distribution(str(row["facts"].get(key) or "unknown") for row in shots)
        for key in ("shot_type", "camera_movement", "environment_type", "edit_role")
    }
    known = sum(
        1 for row in shots for key in ("shot_type", "camera_movement", "environment_type", "edit_role")
        if row["facts"].get(key) not in ("", "unknown")
    )
    possible = len(shots) * 4
    structured_total = sum(len(row.get("fact_fields") or {}) for row in shots)
    structured_known = sum(
        1
        for row in shots
        for field in (row.get("fact_fields") or {}).values()
        if isinstance(field, dict) and field.get("status") != "unavailable"
    )
    coverage = {
        "dialogue": _ratio(sum(bool(row["facts"].get("dialogue_or_narration")) for row in shots), len(shots)),
        "screen_text": _ratio(sum(bool(row["facts"].get("screen_text")) for row in shots), len(shots)),
        "music": _ratio(sum(_has_music(row["facts"].get("audio")) for row in shots), len(shots)),
    }
    return {
        "schema": FINGERPRINT_SCHEMA,
        "shot_count": len(shots),
        "total_duration_seconds": round(total, 3),
        "cuts_per_minute": round(max(0, len(shots) - 1) * 60 / total, 3) if total else 0.0,
        "duration_seconds": {
            "mean": round(statistics.mean(durations), 3) if durations else 0.0,
            "median": round(statistics.median(durations), 3) if durations else 0.0,
            "stdev": round(statistics.pstdev(durations), 3) if len(durations) > 1 else 0.0,
            "min": round(min(durations), 3) if durations else 0.0,
            "max": round(max(durations), 3) if durations else 0.0,
        },
        "distributions": distributions,
        "shot_type_transition_matrix": _transition_matrix(shots, "shot_type"),
        "coverage": coverage,
        "evidence_completeness": (
            round(structured_known / structured_total, 3)
            if structured_total
            else (round(known / possible, 3) if possible else 0.0)
        ),
        "candidate_only": True,
    }


def _readiness(shots: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for shot in shots:
        facts = shot["facts"]
        checks = [
            _check("time_boundary", shot["duration"] > 0, "镜头时间边界有效"),
            _check("reference_frame", bool(shot["reference_frames"]), "至少有一张参考帧"),
            _check("content_reference", bool(facts["dialogue_or_narration"] or facts["screen_text"] or facts["subject_action"]), "至少有一种内容证据"),
            _check("shot_type", facts["shot_type"] != "unknown", "景别已确认或有候选证据"),
            _check("camera_movement", facts["camera_movement"] != "unknown", "主运镜已确认或有候选证据"),
        ]
        rows.append({
            "shot_id": shot["shot_id"],
            "ready": all(item["ok"] for item in checks),
            "checks": checks,
            "blockers": [item["key"] for item in checks if not item["ok"]],
        })
    return {
        "schema": READINESS_SCHEMA,
        "ready": bool(rows) and all(row["ready"] for row in rows),
        "ready_count": sum(1 for row in rows if row["ready"]),
        "shot_count": len(rows),
        "shots": rows,
        "scope": "ready_for_human_imitation_script_review_not_generation",
    }


def _imitation_script(
    shots: list[dict[str, Any]],
    fingerprint: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    readiness_by_id = {row["shot_id"]: row for row in readiness["shots"]}
    rows = []
    for index, shot in enumerate(shots):
        facts = shot["facts"]
        previous_id = shots[index - 1]["shot_id"] if index else ""
        next_id = shots[index + 1]["shot_id"] if index + 1 < len(shots) else ""
        rows.append({
            "shot_id": shot["shot_id"],
            "reference_time": f"{shot['start_time']} - {shot['end_time']}",
            "recommended_duration_seconds": shot["duration"],
            "framing": facts["shot_type"],
            "primary_camera_move": facts["camera_movement"],
            "subject_and_action": facts["subject_action"] or "待确认",
            "setting": facts["environment_type"],
            "composition": facts["composition"] or "待确认",
            "lighting_and_color": facts["color_profile"] or "待确认",
            "dialogue_or_voiceover_reference": facts["dialogue_or_narration"] or "待确认",
            "screen_text_reference": facts["screen_text"],
            "audio_reference": facts["audio"] or "待确认",
            "continuity": {
                "previous_shot_id": previous_id,
                "next_shot_id": next_id,
                "require_screen_direction_review": bool(previous_id or next_id),
                "do_not_invent_identity_or_location": True,
            },
            "prompt_order": ["framing", "subject_and_action", "setting", "lighting_and_color", "primary_camera_move", "audio_reference"],
            "readiness": readiness_by_id.get(shot["shot_id"], {}),
            "candidate_only": True,
            "human_confirmed": False,
        })
    return {
        "schema": SCRIPT_SCHEMA,
        "status": "needs_review" if rows else "missing_shots",
        "shot_count": len(rows),
        "reference_style_fingerprint": fingerprint,
        "shots": rows,
        "coverage_audit": {
            "has_establishing": any(row["facts"]["edit_role"] == "establishing" for row in shots),
            "has_transition": any(row["facts"]["edit_role"] == "transition" for row in shots),
            "has_b_roll": any(row["facts"]["edit_role"] == "b_roll" for row in shots),
            "unknown_shot_language_count": sum(bool(row["unknown_fields"]) for row in shots),
        },
        "publication_allowed": False,
        "media_generation_allowed": False,
    }


def _write_outputs(root: Path, result: dict[str, Any], *, reference_analysis_json: str | Path | None) -> None:
    exports = root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    with bundle_write_lock(root, operation="shot_breakdown", timeout_seconds=1.0):
        write_json(exports / "shot-breakdown.json", result)
        write_json(exports / "style-fingerprint.json", result["style_fingerprint"])
        write_json(exports / "imitation-script.json", result["imitation_script"])
        write_json(exports / "shot-imitation-readiness.json", result["readiness"])
        (exports / "shot-breakdown.md").write_text(_render_breakdown(result), encoding="utf-8")
        (exports / "shot-breakdown.logseq.md").write_text(render_shot_breakdown_logseq(result), encoding="utf-8")
        (exports / "shot-breakdown.csv").write_text(render_shot_breakdown_csv(result), encoding="utf-8-sig")
        (exports / "imitation-script.md").write_text(_render_script(result["imitation_script"]), encoding="utf-8")
        write_json(root / "mcp-shot-breakdown.args.json", {
            "bundle_dir": str(root),
            "title": result.get("title", ""),
            "reference_analysis_json": str(reference_analysis_json or ""),
            "write": True,
        })
        manifest = _read_object(manifest_path)
        manifest.update({
            "shot_breakdown_json": "exports/shot-breakdown.json",
            "shot_breakdown_markdown": "exports/shot-breakdown.md",
            "shot_breakdown_logseq_markdown": "exports/shot-breakdown.logseq.md",
            "shot_breakdown_csv": "exports/shot-breakdown.csv",
            "style_fingerprint_json": "exports/style-fingerprint.json",
            "imitation_script_json": "exports/imitation-script.json",
            "imitation_script_markdown": "exports/imitation-script.md",
            "shot_imitation_readiness_json": "exports/shot-imitation-readiness.json",
            "mcp_shot_breakdown_args": "mcp-shot-breakdown.args.json",
            "shot_breakdown_summary": {
                "status": result["status"],
                "shot_count": result["shot_count"],
                "ready_count": result["readiness"]["ready_count"],
                "updated_at": result["updated_at"],
            },
        })
        write_json(manifest_path, manifest)
    register_bundle_run(
        root,
        run_type="shot_breakdown",
        run_id="shot-breakdown",
        status="needs_review" if result["ok"] else "needs_input",
        title="逐镜头拉片与仿拍脚本候选",
        summary=f"生成 {result['shot_count']} 个逐镜头候选；{result['readiness']['ready_count']} 个通过准备度门。",
        artifacts=[
            {"key": key, "path": path}
            for key, path in result["artifacts"].items()
        ],
        next_actions=["在 Workbench 中复核未知字段、镜头连续性与仿拍脚本。", "确认后再交给视频创作或生成工具。"],
        operator_boundary=result["operator_boundary"],
        resource_requirements={"cpu": 1, "gpu": 0, "network": 0},
        write=True,
    )


def _reference_analysis_provenance(value: str | Path | None) -> dict[str, Any]:
    if not value:
        return {"status": "not_provided"}
    path = Path(value).expanduser().resolve()
    return {
        "status": "loaded",
        "artifact_path": str(path),
        "artifact_bytes": path.stat().st_size,
        "artifact_sha256": sha256_file(path),
        "candidate_only": True,
    }


def _load_reference_analysis(value: str | Path | None) -> dict[int, dict[str, Any]]:
    if not value:
        return {}
    path = Path(value).expanduser().resolve()
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("reference analysis must be a JSON object")
    rows = payload.get("shots") or payload.get("shot_analyses") or []
    result: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        if isinstance(row, dict):
            result[int(row.get("index") or row.get("shot_index") or index)] = {
                **row,
                "_artifact_sha256": sha256_file(path),
                "_artifact_path": str(path),
            }
    return result


def _taxonomy_candidate(field: str, rows: list[dict[str, Any]], imported: dict[str, Any], *keys: str) -> tuple[str, str]:
    for source, obj in (("reference_analysis", imported), *(("timeline", row) for row in rows)):
        for candidate in _candidate_objects(obj):
            for key in keys:
                value = candidate.get(key)
                if isinstance(value, dict):
                    value = value.get("type") or value.get("value")
                if value not in (None, ""):
                    return normalize_taxonomy_value(field, value), source
    return "unknown", "missing"


def _dict_candidate(rows: list[dict[str, Any]], imported: dict[str, Any], *keys: str) -> tuple[dict[str, Any], str]:
    for source, obj in (("reference_analysis", imported), *(("timeline", row) for row in rows)):
        for candidate in _candidate_objects(obj):
            for key in keys:
                value = candidate.get(key)
                if isinstance(value, dict) and value:
                    return value, source
    return {}, "missing"


def _candidate_objects(value: dict[str, Any]) -> list[dict[str, Any]]:
    result = [value]
    for key in ("facts", "camera", "camera_motion", "visual_understanding", "integrated_visual"):
        nested = value.get(key)
        if isinstance(nested, dict):
            result.append(nested)
    return result


def _collect_temporal(rows: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for row in rows:
        for key in ("temporal_visual_understanding", "visual_understanding"):
            value = row.get(key)
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, dict):
                for subkey in ("event_sequence", "actions", "activity", "summary", "human_summary"):
                    nested = value.get(subkey)
                    if isinstance(nested, str):
                        values.append(nested)
                    elif isinstance(nested, list):
                        values.extend(str(item) for item in nested if item)
    return _list_unique(values)


def _tags(row: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key in ("tagger_tags", "tags", "general_tags"):
        value = row.get(key)
        if isinstance(value, list):
            result.extend(str(item) for item in value if item)
    for annotation in row.get("tagger_annotations") or []:
        if isinstance(annotation, dict):
            for key in ("tags_zh", "tags_en", "tags"):
                value = annotation.get(key)
                if isinstance(value, list):
                    result.extend(str(item) for item in value if item)
    return _list_unique(result)


def _frames(row: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key in ("frame_paths", "temporal_frame_paths", "evidence_frame_paths"):
        value = row.get(key)
        if isinstance(value, list):
            result.extend(str(item) for item in value if item)
    # Reuse Bundle-local copied assets. The public renderer still resolves
    # symlinks and rejects sources outside the active Bundle.
    assets = row.get("assets")
    if isinstance(assets, list):
        for asset in assets:
            if isinstance(asset, dict) and isinstance(asset.get("path"), str) and asset["path"]:
                result.append(asset["path"])
    integrated = row.get("integrated_visual")
    if isinstance(integrated, dict):
        value = integrated.get("evidence_frame_paths")
        if isinstance(value, list):
            result.extend(str(item) for item in value if item)
    return _list_unique(result)


def _overlaps(row: dict[str, Any], start: float, end: float) -> bool:
    row_start, row_end = _number(row.get("start")), _number(row.get("end"))
    return row_end > start and row_start < end


def _text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _join_unique(values: Iterable[str]) -> str:
    return "\n".join(_list_unique(values))


def _list_unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _shot_fact_provenance(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = root / "exports" / "shot-facts.json"
    if not payload or not path.is_file():
        return {"status": "not_available"}
    return {
        "status": "loaded",
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": payload.get("schema"),
        "shot_count": payload.get("shot_count", 0),
    }


def _transition_matrix(shots: list[dict[str, Any]], key: str) -> dict[str, int]:
    values = [str(row["facts"].get(key) or "unknown") for row in shots]
    return dict(Counter(f"{left}->{right}" for left, right in zip(values, values[1:])))


def _has_music(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return str(value.get("music") or "").strip().lower() not in {"", "unknown"}


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 3) if denominator else 0.0

def _distribution(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _check(key: str, ok: bool, message: str) -> dict[str, Any]:
    return {"key": key, "ok": bool(ok), "message": message}


def _number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _read_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def _read_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    value = read_json(path)
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _render_breakdown(result: dict[str, Any]) -> str:
    lines = [
        "# 逐镜头拉片",
        "",
        f"- 状态：`{result['status']}`",
        f"- 镜头数：`{result['shot_count']}`",
        f"- 仿拍准备度：`{result['readiness']['ready_count']}/{result['shot_count']}`",
        "- 所有字段均为候选证据，人工确认前不允许导出或生成。",
        "",
        "| 镜头 | 时间 | 时长 | 景别 | 运镜 | 内容证据 | 未知字段 |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for shot in result["shots"]:
        facts = shot["facts"]
        content = facts["dialogue_or_narration"] or facts["screen_text"] or ""
        lines.append(
            f"| `{shot['shot_id']}` | `{shot['start_time']} - {shot['end_time']}` | {shot['duration']:.2f}s | "
            f"{_md(facts['shot_type'])} | {_md(facts['camera_movement'])} | {_md(content[:80])} | {_md(', '.join(shot['unknown_fields']))} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_script(script: dict[str, Any]) -> str:
    lines = ["# 仿拍脚本候选", "", f"- 状态：`{script['status']}`", "- 未经人工确认，不允许生成或发布。", ""]
    for shot in script["shots"]:
        lines.extend([
            f"## {shot['shot_id']} · {shot['reference_time']}",
            "",
            f"- 建议时长：{shot['recommended_duration_seconds']:.2f} 秒",
            f"- 景别：{shot['framing']}",
            f"- 主运镜：{shot['primary_camera_move']}",
            f"- 主体与动作：{_md(shot['subject_and_action'])}",
            f"- 场景：{_md(shot['setting'])}",
            f"- 构图：{_md(shot['composition'])}",
            f"- 灯光与色彩：{_md(shot['lighting_and_color'])}",
            f"- 对白/旁白参考：{_md(shot['dialogue_or_voiceover_reference'])}",
            f"- 阻断项：{', '.join(shot['readiness'].get('blockers') or []) or '无'}",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def _md(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value or "").replace("|", "\\|").replace("\n", " ")
