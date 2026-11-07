from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable

from .general_tagger_adapter import run_general_tagger
from .models import now_iso
from .storage import read_json, write_json


SCHEMA = "video_knowledge_pipeline.temporal_tag_delta.v1"
INPUT_SCHEMA = "video_knowledge_pipeline.temporal_tag_delta_input.v1"
REPORT_SCHEMA = "video_knowledge_pipeline.temporal_tag_delta_report.v1"

DYNAMIC_TERMS = (
    "点击",
    "打开",
    "关闭",
    "切换",
    "拖动",
    "输入",
    "移动",
    "变化",
    "变成",
    "出现",
    "消失",
    "演示",
    "步骤",
    "流程",
    "click",
    "open",
    "close",
    "switch",
    "drag",
    "type",
    "move",
    "change",
    "appear",
    "disappear",
    "step",
)


def _run_temporal_tag_delta_v1(
    bundle_dir: str | Path,
    *,
    input_json: str | Path | None = None,
    execute_tagger: bool = False,
    source_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "cuda",
    prefer_language: str = "zh",
    limit: int = 0,
    min_frames: int = 3,
    write: bool = True,
    _inference_backend: Callable[[Path], dict[str, object]] | None = None,
) -> dict[str, Any]:
    """Smooth per-frame tags and decide whether temporal VLM is still required.

    Tag deltas are supporting evidence only. They never assert an action or a
    causal relationship, and they never populate temporal_visual_understanding.
    """

    root = Path(bundle_dir).expanduser().resolve()
    timeline_path = root / "timeline.json"
    if not timeline_path.is_file():
        raise FileNotFoundError(f"timeline.json not found: {timeline_path}")
    timeline = read_json(timeline_path)
    if not isinstance(timeline, list):
        raise ValueError("timeline.json must be an array")
    if int(min_frames) < 2:
        raise ValueError("min_frames must be at least 2")

    imported = _read_input(input_json) if input_json else {}
    tagger_result: dict[str, Any] = {}
    if execute_tagger:
        tagger_result = run_general_tagger(
            root,
            source_root=source_root,
            checkpoint_path=checkpoint_path,
            device=device,
            prefer_language=prefer_language,
            limit=0,
            frame_mode="continuous",
            execute=True,
            import_annotations=False,
            write=False,
            _inference_backend=_inference_backend,
        )
        if tagger_result.get("status") == "completed":
            imported = _group_annotations(tagger_result.get("annotations") or [])

    candidates = _candidate_rows(root, timeline)
    if int(limit) > 0:
        candidates = candidates[: int(limit)]
    template_path = root / "temporal-tag-delta-input-template.json"
    template = {
        "schema": INPUT_SCHEMA,
        "items": [
            {
                "index": row["index"],
                "frames": [
                    {
                        "frame_position": position,
                        "frame_path": path,
                        "tags": [],
                        "ocr_text": "",
                    }
                    for position, path in enumerate(row["frame_paths"], start=1)
                ],
            }
            for row in candidates
        ],
    }
    if write:
        write_json(template_path, template)

    items: list[dict[str, Any]] = []
    updated_indexes: list[int] = []
    for candidate in candidates:
        index = int(candidate["index"])
        item = candidate["item"]
        observations = (
            imported.get(index)
            or _existing_observations(item)
            or []
        )
        analysis = _analyze_temporal_tag_observations_v1(
            observations,
            transcript=str(item.get("transcript") or item.get("text") or ""),
            supporting_signals=_supporting_signals(item, observations),
            min_frames=min_frames,
        )
        row = {
            "index": index,
            "start": item.get("start", 0),
            "end": item.get("end", 0),
            "frame_paths": candidate["frame_paths"],
            **analysis,
        }
        items.append(row)
        if analysis["status"] == "completed":
            item["temporal_tag_delta"] = {
                key: value
                for key, value in analysis.items()
                if key not in {"status"}
            }
            item["temporal_multimodal_escalation"] = {
                "required": analysis["decision"] == "temporal_multimodal",
                "decision": analysis["decision"],
                "reasons": analysis["escalation_reasons"],
                "tag_delta_is_supporting_evidence_only": True,
                "suggested_command": (
                    f".\\scripts\\video-knowledge.ps1 run-temporal-visual-analysis "
                    f"'{root}' --index {index}"
                    if analysis["decision"] == "temporal_multimodal"
                    else ""
                ),
            }
            updated_indexes.append(index)

    completed = sum(1 for row in items if row["status"] == "completed")
    escalated = sum(
        1 for row in items if row.get("decision") == "temporal_multimodal"
    )
    coarse = sum(1 for row in items if row.get("decision") == "coarse_static_summary")
    if execute_tagger and tagger_result.get("status") != "completed":
        status = "blocked_tagger"
    elif not candidates:
        status = "not_needed"
    elif completed == len(candidates):
        status = "completed"
    elif completed:
        status = "partial"
    else:
        status = "needs_frame_tags"
    result = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "ok": status in {"completed", "not_needed"},
        "bundle_dir": str(root),
        "candidate_count": len(candidates),
        "completed_count": completed,
        "needs_frame_tags_count": len(candidates) - completed,
        "coarse_static_count": coarse,
        "temporal_multimodal_count": escalated,
        "updated_indexes": updated_indexes,
        "input_template_path": str(template_path),
        "tagger": {
            "executed": bool(execute_tagger),
            "status": str(tagger_result.get("status") or "not_executed"),
            "model": str(tagger_result.get("model") or ""),
            "frame_mode": str(tagger_result.get("frame_mode") or "continuous"),
            "error": tagger_result.get("error"),
        },
        "items": items,
        "change_record": {
            "intent": "Use low-cost continuous-frame tag changes before temporal multimodal calls.",
            "decision": "Smooth one-frame gaps/flicker, preserve tag deltas, and escalate dynamic or ambiguous segments.",
            "reason": "Stable labels describe scene contents but cannot prove actions, ordering, or causality.",
            "evidence": ["per_frame_tags", "ocr_change", "scene_boundary", "frame_change", "asr_dynamic_terms"],
            "effective_scope": "timeline.temporal_tag_delta and timeline.temporal_multimodal_escalation only",
            "rollback": "Remove those two fields and rerun the command; ASR, OCR, and temporal_visual_understanding are untouched.",
        },
        "operator_boundary": {
            "local_only": True,
            "remote_calls_made": 0,
            "automatic_model_download": False,
            "tag_delta_is_not_action_or_causality_proof": True,
            "does_not_populate_temporal_visual_understanding": True,
        },
        "updated_at": now_iso(),
    }
    report_json = root / "temporal-tag-delta.json"
    report_markdown = root / "temporal-tag-delta.md"
    result["report_json_path"] = str(report_json)
    result["report_markdown_path"] = str(report_markdown)
    if write:
        write_json(timeline_path, timeline)
        write_json(report_json, result)
        report_markdown.write_text(_render_markdown(result), encoding="utf-8")
        manifest_path = root / "manifest.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            if isinstance(manifest, dict):
                manifest["temporal_tag_delta"] = {
                    "schema": SCHEMA,
                    "status": status,
                    "completed_count": completed,
                    "coarse_static_count": coarse,
                    "temporal_multimodal_count": escalated,
                    "report_json": report_json.name,
                    "report_markdown": report_markdown.name,
                    "input_template_json": template_path.name,
                    "updated_at": result["updated_at"],
                }
                write_json(manifest_path, manifest)
    return result


def _analyze_temporal_tag_observations_v1(
    observations: list[dict[str, Any]],
    *,
    transcript: str = "",
    supporting_signals: dict[str, Any] | None = None,
    min_frames: int = 3,
) -> dict[str, Any]:
    rows = _normalise_observations(observations)
    if len(rows) < int(min_frames):
        return {
            "status": "needs_frame_tags",
            "decision": "insufficient_evidence",
            "raw_frame_count": len(rows),
            "smoothed_frame_count": 0,
            "raw_observations": rows,
            "smoothed_observations": [],
            "stable_tags": [],
            "transitions": [],
            "tag_change_score": None,
            "coarse_summary": "",
            "escalation_reasons": ["insufficient_frame_tag_observations"],
            "limitations": ["No action, order, intent, or causality may be inferred from insufficient tags."],
        }
    smoothed = _smooth_tags(rows)
    stable = _stable_tags(smoothed)
    transitions = _transitions(smoothed)
    change_score = _tag_change_score(smoothed)
    signals = dict(supporting_signals or {})
    reasons: list[str] = []
    if change_score >= 0.34:
        reasons.append("high_tag_change")
    if bool(signals.get("ocr_changed")):
        reasons.append("ocr_changed")
    if bool(signals.get("scene_boundary")):
        reasons.append("scene_boundary")
    if str(signals.get("frame_change_status") or "") in {"dynamic", "localized_motion"}:
        reasons.append(f"frame_change_{signals['frame_change_status']}")
    if _contains_dynamic_term(transcript):
        reasons.append("asr_dynamic_term")
    if not stable:
        reasons.append("no_stable_tags")
    if any(row.get("tags") == [] for row in smoothed):
        reasons.append("empty_frame_tags")
    reasons = _dedupe(reasons)
    decision = "temporal_multimodal" if reasons else "coarse_static_summary"
    return {
        "schema": SCHEMA,
        "status": "completed",
        "decision": decision,
        "raw_frame_count": len(rows),
        "smoothed_frame_count": len(smoothed),
        "raw_observations": rows,
        "smoothed_observations": smoothed,
        "stable_tags": stable,
        "transitions": transitions,
        "tag_change_score": change_score,
        "supporting_signals": signals,
        "coarse_summary": _coarse_summary(stable, transitions, decision),
        "escalation_reasons": reasons,
        "limitations": [
            "Tags identify likely visible contents, not actions, temporal order, intent, or causality.",
            "The coarse summary must not be represented as complete temporal multimodal understanding.",
        ],
    }


def _candidate_rows(root: Path, timeline: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        paths = _frame_paths(root, item)
        if len(paths) < 2:
            continue
        rows.append(
            {
                "index": _positive_int(item.get("index")) or position,
                "frame_paths": paths,
                "item": item,
            }
        )
    return rows


def _frame_paths(root: Path, item: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("temporal_frame_paths", "frame_paths", "evidence_frame_paths"):
        if isinstance(item.get(key), list):
            values.extend(item[key])
    integrated = item.get("integrated_visual")
    if isinstance(integrated, dict) and isinstance(integrated.get("evidence_frame_paths"), list):
        values.extend(integrated["evidence_frame_paths"])
    paths: list[str] = []
    seen: set[str] = set()
    for value in values:
        raw = value.get("path") if isinstance(value, dict) else value
        if not str(raw or "").strip():
            continue
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = root / path
        key = str(path.resolve()).casefold()
        if key in seen:
            continue
        seen.add(key)
        paths.append(str(path.resolve()))
    return paths


def _existing_observations(item: dict[str, Any]) -> list[dict[str, Any]]:
    for owner in (item, item.get("integrated_visual")):
        if not isinstance(owner, dict):
            continue
        for key in ("temporal_frame_tags", "tagger_frame_observations", "per_frame_tags"):
            value = owner.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, dict)]
    return []


def _supporting_signals(
    item: dict[str, Any], observations: list[dict[str, Any]]
) -> dict[str, Any]:
    ocr_values = [
        re.sub(r"\s+", "", str(row.get("ocr_text") or ""))
        for row in observations
        if isinstance(row, dict) and str(row.get("ocr_text") or "").strip()
    ]
    triage = item.get("vision_review_triage")
    triage = triage if isinstance(triage, dict) else {}
    frame_change = item.get("frame_change_evidence")
    frame_change = frame_change if isinstance(frame_change, dict) else triage.get("frame_change_evidence")
    frame_change = frame_change if isinstance(frame_change, dict) else {}
    scene = item.get("scene_boundary_evidence")
    scene = scene if isinstance(scene, dict) else triage.get("scene_boundary_evidence")
    scene = scene if isinstance(scene, dict) else {}
    return {
        "ocr_changed": len(set(ocr_values)) > 1 or bool(item.get("ocr_change_detected")),
        "scene_boundary": bool(scene.get("matched") or item.get("scene_boundary_matched")),
        "frame_change_status": str(frame_change.get("status") or item.get("frame_change_status") or "not_available"),
    }


def _normalise_observations(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, row in enumerate(value or [], start=1):
        if not isinstance(row, dict):
            continue
        tags = _normalise_tags(row.get("tags") or row.get("labels") or row.get("labels_zh") or row.get("labels_en"))
        rows.append(
            {
                "frame_position": _positive_int(row.get("frame_position")) or position,
                "frame_id": str(row.get("frame_id") or f"frame-{position:03d}"),
                "frame_path": str(row.get("frame_path") or row.get("artifact_path") or ""),
                "tags": tags,
                "ocr_text": str(row.get("ocr_text") or ""),
            }
        )
    rows.sort(key=lambda row: int(row["frame_position"]))
    return rows


def _smooth_tags(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sets = [set(row["tags"]) for row in rows]
    smoothed = [set(tags) for tags in sets]
    for position in range(1, len(sets) - 1):
        gap_fill = (sets[position - 1] & sets[position + 1]) - sets[position]
        flicker = sets[position] - sets[position - 1] - sets[position + 1]
        smoothed[position].update(gap_fill)
        smoothed[position].difference_update(flicker)
    return [
        {**row, "tags": sorted(smoothed[position])}
        for position, row in enumerate(rows)
    ]


def _stable_tags(rows: list[dict[str, Any]]) -> list[str]:
    threshold = max(2, math.ceil(len(rows) * 0.6))
    counts: dict[str, int] = {}
    for row in rows:
        for tag in set(row["tags"]):
            counts[tag] = counts.get(tag, 0) + 1
    return sorted(tag for tag, count in counts.items() if count >= threshold)


def _transitions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for previous, current in zip(rows, rows[1:]):
        before = set(previous["tags"])
        after = set(current["tags"])
        appeared = sorted(after - before)
        disappeared = sorted(before - after)
        if appeared or disappeared:
            output.append(
                {
                    "from_frame_position": previous["frame_position"],
                    "to_frame_position": current["frame_position"],
                    "appeared": appeared,
                    "disappeared": disappeared,
                }
            )
    return output


def _tag_change_score(rows: list[dict[str, Any]]) -> float:
    distances: list[float] = []
    for previous, current in zip(rows, rows[1:]):
        left = set(previous["tags"])
        right = set(current["tags"])
        union = left | right
        distances.append(0.0 if not union else 1.0 - len(left & right) / len(union))
    return round(sum(distances) / max(1, len(distances)), 4)


def _coarse_summary(
    stable: list[str], transitions: list[dict[str, Any]], decision: str
) -> str:
    stable_text = "、".join(stable[:12]) or "无稳定标签"
    if decision == "temporal_multimodal":
        return f"多帧稳定可见内容：{stable_text}；存在变化或歧义，需升级连续帧多模态理解。"
    if transitions:
        return f"多帧稳定可见内容：{stable_text}；仅记录少量标签出现/消失，不据此推断动作。"
    return f"多帧稳定可见内容：{stable_text}；标签基本稳定，未发现足以证明动作的变化。"


def _read_input(path_value: str | Path) -> dict[int, list[dict[str, Any]]]:
    path = Path(path_value).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = value.get("items") if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError("temporal tag input must contain an items array")
    result: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        index = _positive_int(row.get("index") or row.get("timeline_index"))
        frames = row.get("frames") or row.get("observations")
        if index and isinstance(frames, list):
            result[index] = [dict(frame) for frame in frames if isinstance(frame, dict)]
    return result


def _group_annotations(rows: list[Any]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        index = _positive_int(row.get("timeline_index") or row.get("index"))
        if not index:
            continue
        result.setdefault(index, []).append(
            {
                "frame_position": row.get("frame_position"),
                "frame_id": row.get("frame_id"),
                "frame_path": row.get("artifact_path"),
                "tags": row.get("tags") or [],
            }
        )
    return result


def _normalise_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        values = re.split(r"[|,，、;；\n]+", value)
    elif isinstance(value, list):
        values = value
    else:
        values = []
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        tag = str(raw or "").strip()
        key = tag.casefold()
        if not tag or key in seen:
            continue
        seen.add(key)
        output.append(tag)
    return output


def _contains_dynamic_term(value: str) -> bool:
    lowered = str(value or "").casefold()
    return any(term.casefold() in lowered for term in DYNAMIC_TERMS)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _positive_int(value: Any) -> int:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Temporal Tag Delta",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Candidates: `{result.get('candidate_count')}`",
        f"- Coarse/static: `{result.get('coarse_static_count')}`",
        f"- Escalate to temporal multimodal: `{result.get('temporal_multimodal_count')}`",
        "- Boundary: tag deltas are supporting evidence; they do not prove actions, order, intent, or causality.",
        "",
        "| Index | Frames | Change score | Decision | Reasons |",
        "|---:|---:|---:|---|---|",
    ]
    for row in result.get("items") or []:
        lines.append(
            "| {index} | {frames} | {score} | {decision} | {reasons} |".format(
                index=row.get("index"),
                frames=row.get("raw_frame_count", 0),
                score=row.get("tag_change_score") if row.get("tag_change_score") is not None else "-",
                decision=row.get("decision", ""),
                reasons=", ".join(row.get("escalation_reasons") or []) or "-",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


# v2 is deliberately implemented as a production adapter over the committed v1
# module.  Keeping the v1 helpers in place preserves historical Bundle readers;
# public execution and analysis now use the bounded, hash-bound v2 contract.
from .temporal_tag_delta_v2 import (  # noqa: E402,F401
    apply_temporal_tag_delta_restore,
    analyze_temporal_tag_observations,
    run_temporal_tag_delta,
)
