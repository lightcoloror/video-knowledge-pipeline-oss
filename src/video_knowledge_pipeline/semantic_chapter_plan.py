from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .models import now_iso
from .smart_summary_input_pack import build_smart_summary_input_pack
from .storage import bundle_write_lock, read_json, write_json
from .transcript import format_timestamp


SCHEMA = "video_knowledge_pipeline.semantic_chapter_plan.v1"
TOPIC_MARKERS = (
    "接下来",
    "下面",
    "第二",
    "第三",
    "第四",
    "最后",
    "总结",
    "回到",
    "另一个",
    "下一个",
    "案例",
    "步骤",
    "问题是",
)


def build_semantic_chapter_plan(
    bundle_dir: str | Path,
    *,
    title: str = "",
    input_pack: str | Path | None = None,
    chapter_mode: str = "semantic",
    write: bool = True,
) -> dict[str, Any]:
    """Build chapter boundaries from transcript, pauses, tags and visual changes."""

    root = Path(bundle_dir).expanduser().resolve()
    if chapter_mode not in {"semantic", "fixed"}:
        raise ValueError("chapter_mode must be semantic or fixed")
    pack = _load_pack(root, input_pack=input_pack, title=title, write=write)
    segments = [row for row in pack.get("transcript_segments") or [] if isinstance(row, dict) and _text(row)]
    duration = max((_seconds(row.get("end")) for row in segments), default=0.0)
    min_seconds, preferred_seconds, max_seconds = _chapter_limits(duration)
    scene_candidates = _scene_detection_candidates(root)
    candidates = _boundary_candidates(segments, pack, scene_candidates=scene_candidates)
    boundaries = _select_boundaries(
        candidates,
        duration=duration,
        min_seconds=min_seconds,
        preferred_seconds=preferred_seconds,
        max_seconds=max_seconds,
        fixed=chapter_mode == "fixed",
    )
    chapters = _chapters(boundaries, segments, pack)
    parts = _top_level_parts(chapters, duration)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "title": title or str(pack.get("title") or root.name),
        "chapter_mode": chapter_mode,
        "status": "completed" if chapters else "missing_transcript",
        "ok": bool(chapters),
        "duration_seconds": duration,
        "constraints": {
            "min_chapter_seconds": min_seconds,
            "preferred_chapter_seconds": preferred_seconds,
            "max_chapter_seconds": max_seconds,
        },
        "candidate_count": len(candidates),
        "boundary_count": len(boundaries),
        "chapter_count": len(chapters),
        "part_count": len(parts),
        "boundaries": boundaries,
        "chapters": chapters,
        "parts": parts,
        "source_signals": ["transcript_topic_shift", "vad_pause", "topic_markers", "tagger_labels", "ocr_or_visual_change"] + (["pyscenedetect_scene_boundary"] if scene_candidates else []),
        "operator_boundary": {
            "local_only": True,
            "no_llm_call": True,
            "semantic_plan_is_evidence_for_chapter_llm": True,
            "does_not_modify_transcript": True,
        },
        "artifacts": {"json": "exports/semantic-chapter-plan.json", "markdown": "exports/semantic-chapter-plan.md"},
        "updated_at": now_iso(),
    }
    if write:
        exports = root / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        with bundle_write_lock(root, operation="semantic_chapter_plan", timeout_seconds=1.0):
            write_json(exports / "semantic-chapter-plan.json", result)
            (exports / "semantic-chapter-plan.md").write_text(_render_markdown(result), encoding="utf-8")
            write_json(root / "mcp-semantic-chapter-plan.args.json", {"bundle_dir": str(root), "title": title, "chapter_mode": chapter_mode, "write": True})
            manifest_path = root / "manifest.json"
            manifest = read_json(manifest_path) if manifest_path.exists() else {}
            if isinstance(manifest, dict):
                manifest["semantic_chapter_plan_json"] = "exports/semantic-chapter-plan.json"
                manifest["semantic_chapter_plan_markdown"] = "exports/semantic-chapter-plan.md"
                manifest["mcp_semantic_chapter_plan_args"] = "mcp-semantic-chapter-plan.args.json"
                manifest["semantic_chapter_plan_summary"] = {"status": result["status"], "chapter_count": len(chapters), "part_count": len(parts), "updated_at": result["updated_at"]}
                write_json(manifest_path, manifest)
    return result


def _load_pack(root: Path, *, input_pack: str | Path | None, title: str, write: bool) -> dict[str, Any]:
    if input_pack:
        path = Path(input_pack).expanduser()
        if not path.is_absolute():
            path = root / path
        data = read_json(path)
        if not isinstance(data, dict):
            raise ValueError("smart-summary input pack must be a JSON object")
        return data
    path = root / "exports" / "smart-summary-input-pack.json"
    if path.exists():
        data = read_json(path)
        if isinstance(data, dict) and data.get("transcript_segments"):
            return data
    return build_smart_summary_input_pack(root, title=title, write=write)


def _chapter_limits(duration: float) -> tuple[float, float, float]:
    if duration <= 20 * 60:
        return 60.0, 180.0, 300.0
    if duration <= 90 * 60:
        return 240.0, 420.0, 600.0
    return 360.0, 600.0, 900.0


def _boundary_candidates(segments: list[dict[str, Any]], pack: dict[str, Any], *, scene_candidates: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    visual_items = ((pack.get("visual_digest") or {}).get("items") or []) if isinstance(pack.get("visual_digest"), dict) else []
    rows: list[dict[str, Any]] = []
    for index in range(1, len(segments)):
        previous = segments[index - 1]
        current = segments[index]
        time = _seconds(current.get("start"))
        gap = max(0.0, time - _seconds(previous.get("end")))
        left = _tokens(_text(previous))
        right = _tokens(_text(current))
        similarity = _jaccard(left, right)
        marker_hits = [marker for marker in TOPIC_MARKERS if marker in _text(current)[:80]]
        tag_hits = _tag_hits(current)
        visual_change = _visual_change_near(visual_items, time)
        score = (1.0 - similarity) * 2.0
        score += min(gap / 4.0, 2.0)
        score += min(len(marker_hits), 2) * 1.5
        score += min(len(tag_hits), 2) * 1.2
        score += 1.5 if visual_change else 0.0
        reasons: list[str] = []
        if 1.0 - similarity >= 0.55:
            reasons.append("transcript_topic_shift")
        if gap >= 2.0:
            reasons.append("vad_pause")
        if marker_hits:
            reasons.append("topic_marker:" + ",".join(marker_hits))
        if tag_hits:
            reasons.append("tagger:" + ",".join(tag_hits))
        if visual_change:
            reasons.append("ocr_or_visual_change")
        rows.append({"time": time, "segment_index": index, "score": round(score, 4), "reasons": reasons})
    for scene in scene_candidates or []:
        time = _seconds(scene.get("seconds"))
        if time <= 0:
            continue
        rows.append(
            {
                "time": time,
                "segment_index": _segment_index_near(segments, time),
                "score": 1.8,
                "reasons": ["pyscenedetect_scene_boundary"],
                "scene_boundary": True,
            }
        )
    return rows


def _scene_detection_candidates(root: Path) -> list[dict[str, Any]]:
    path = root / "exports" / "scene-detection.json"
    if not path.exists():
        return []
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("backend") != "pyscenedetect":
        return []
    return [row for row in payload.get("boundaries") or [] if isinstance(row, dict)]


def _segment_index_near(segments: list[dict[str, Any]], time: float) -> int:
    if not segments:
        return 0
    return min(
        range(len(segments)),
        key=lambda index: abs(_seconds(segments[index].get("start")) - time),
    )


def _select_boundaries(
    candidates: list[dict[str, Any]],
    *,
    duration: float,
    min_seconds: float,
    preferred_seconds: float,
    max_seconds: float,
    fixed: bool,
) -> list[dict[str, Any]]:
    if duration <= 0:
        return []
    selected = [{"time": 0.0, "score": 99.0, "reasons": ["video_start"]}]
    cursor = 0.0
    while duration - cursor > max_seconds:
        lower = cursor + min_seconds
        upper = min(duration, cursor + max_seconds)
        target = min(duration, cursor + preferred_seconds)
        eligible = [row for row in candidates if lower <= float(row["time"]) <= upper]
        if fixed or not eligible:
            choice = {"time": target, "score": 0.0, "reasons": ["duration_constraint"]}
        else:
            choice = max(eligible, key=lambda row: float(row["score"]) - abs(float(row["time"]) - target) / max(preferred_seconds, 1.0))
        choice = {**choice, "time": round(float(choice["time"]), 3)}
        if choice["time"] <= cursor:
            break
        selected.append(choice)
        cursor = float(choice["time"])
    selected.append({"time": round(duration, 3), "score": 99.0, "reasons": ["video_end"]})
    return selected


def _chapters(boundaries: list[dict[str, Any]], segments: list[dict[str, Any]], pack: dict[str, Any]) -> list[dict[str, Any]]:
    chapters: list[dict[str, Any]] = []
    for index in range(max(0, len(boundaries) - 1)):
        start = float(boundaries[index]["time"])
        end = float(boundaries[index + 1]["time"])
        bucket = [row for row in segments if start <= ((_seconds(row.get("start")) + _seconds(row.get("end"))) / 2) < end or (index == len(boundaries) - 2 and ((_seconds(row.get("start")) + _seconds(row.get("end"))) / 2) == end)]
        if not bucket:
            continue
        title_hint = _title_hint(bucket)
        chapters.append(
            {
                "index": len(chapters) + 1,
                "start": start,
                "end": end,
                "start_time": format_timestamp(start),
                "end_time": format_timestamp(end),
                "title_hint": title_hint,
                "segment_indexes": [int(row.get("index") or 0) for row in bucket],
                "timeline_indexes": sorted({int(row.get("timeline_index") or 0) for row in bucket if int(row.get("timeline_index") or 0) > 0}),
                "boundary_reasons": list(boundaries[index + 1].get("reasons") or []),
                "boundary_score": float(boundaries[index + 1].get("score") or 0.0),
                "evidence_counts": _evidence_counts(bucket),
            }
        )
    return chapters


def _top_level_parts(chapters: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    if duration <= 90 * 60 or not chapters:
        return []
    target = 45 * 60
    parts: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    start = float(chapters[0]["start"])
    for chapter in chapters:
        current.append(chapter)
        if float(chapter["end"]) - start >= target:
            parts.append(_part(len(parts) + 1, current))
            current = []
            start = float(chapter["end"])
    if current:
        parts.append(_part(len(parts) + 1, current))
    return parts


def _part(index: int, chapters: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "index": index,
        "start": float(chapters[0]["start"]),
        "end": float(chapters[-1]["end"]),
        "start_time": chapters[0]["start_time"],
        "end_time": chapters[-1]["end_time"],
        "chapter_indexes": [int(row["index"]) for row in chapters],
        "title_hint": chapters[0].get("title_hint") or f"第 {index} 部分",
    }


def _visual_change_near(items: list[Any], time: float) -> bool:
    nearby = [row for row in items if isinstance(row, dict) and abs(_seconds(row.get("start")) - time) <= 8.0]
    routes = {str(row.get("visual_route") or "") for row in nearby}
    headings = {str(row.get("visual_text") or row.get("structured_visual") or "")[:80] for row in nearby
                if row.get("visual_text_role") != "incidental_background"
                and str(row.get("visual_text") or row.get("structured_visual") or "").strip()}
    return len(routes) > 1 or len(headings) > 1


def _tag_hits(segment: dict[str, Any]) -> list[str]:
    evidence = segment.get("evidence_inputs") if isinstance(segment.get("evidence_inputs"), dict) else {}
    raw = evidence.get("tagger") or evidence.get("tags") or evidence.get("labels") or []
    values = raw if isinstance(raw, list) else [raw]
    return [str(value) for value in values if any(key in str(value) for key in ("重点", "话题", "步骤", "案例", "结论", "工具"))][:4]


def _title_hint(bucket: list[dict[str, Any]]) -> str:
    for row in bucket[:4]:
        text = _text(row)
        for marker in TOPIC_MARKERS:
            if marker in text:
                return _clip(text, 34)
    return _clip(_text(bucket[0]), 34) if bucket else ""


def _evidence_counts(bucket: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "transcript_segments": len(bucket),
        "tagged_segments": sum(1 for row in bucket if _tag_hits(row)),
        "visual_linked_segments": sum(1 for row in bucket if row.get("visual_digest_ref")),
    }


def _tokens(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", str(text or "").lower())
    chinese = [compact[index : index + 2] for index in range(max(0, len(compact) - 1)) if re.search(r"[\u4e00-\u9fff]", compact[index : index + 2])]
    latin = re.findall(r"[a-z0-9_+-]{2,}", compact)
    return set(chinese + latin)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _text(row: dict[str, Any]) -> str:
    return str(row.get("punctuated_text") or row.get("corrected_text") or row.get("raw_text") or "").strip()


def _seconds(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except Exception:
        return 0.0


def _overlaps(left_start: float, left_end: float, right_start: float, right_end: float) -> bool:
    return max(left_start, right_start) < min(left_end, right_end) or (right_start == left_start)


def _clip(value: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" ，。")
    return text if len(text) <= limit else text[: max(1, limit - 1)].rstrip(" ，。") + "…"


def _render_markdown(result: dict[str, Any]) -> str:
    limits = result.get("constraints") if isinstance(result.get("constraints"), dict) else {}
    lines = [
        "# Semantic Chapter Plan",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Mode: `{result.get('chapter_mode')}`",
        f"- Duration: `{format_timestamp(float(result.get('duration_seconds') or 0))}`",
        f"- Chapters / parts: `{result.get('chapter_count')}` / `{result.get('part_count')}`",
        f"- Chapter limits: `{limits.get('min_chapter_seconds')}` / `{limits.get('preferred_chapter_seconds')}` / `{limits.get('max_chapter_seconds')}` seconds",
        "",
        "## Chapters",
        "",
        "| # | Time | Title hint | Boundary reasons |",
        "| ---: | --- | --- | --- |",
    ]
    for row in result.get("chapters") or []:
        lines.append(f"| {row.get('index')} | `{row.get('start_time')} - {row.get('end_time')}` | {_md(row.get('title_hint'))} | {_md(', '.join(row.get('boundary_reasons') or []))} |")
    if result.get("parts"):
        lines.extend(["", "## Top-level Parts", ""])
        for row in result.get("parts") or []:
            lines.append(f"- `{row.get('start_time')} - {row.get('end_time')}` chapters={row.get('chapter_indexes')} {row.get('title_hint')}")
    return "\n".join(lines).rstrip() + "\n"


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")
