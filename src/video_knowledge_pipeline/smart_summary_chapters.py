from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .powershell import quote_powershell_literal as _ps_quote
from .local_media_progress import LocalMediaProgress, ProgressCallback
from .interval_coverage import closed_intervals_overlap as _overlaps
from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .semantic_chapter_plan import build_semantic_chapter_plan
from .smart_summary_input_pack import build_smart_summary_input_pack
from .storage import read_json, write_json
from .storage import read_json_object_or_empty as _read_mapping
from .term_text import apply_term_replacement_pairs, load_bundle_term_replacements
from .transcript import format_timestamp
from .citation_digest import digest_records, courseware_records, bind_citation_digest, SCHEMA as DIGEST_SCHEMA

SCHEMA = "video_knowledge_pipeline.smart_summary_chapters.v1"
TIMELINE_COVERAGE_SCHEMA = "video_knowledge_pipeline.chapter_timeline_coverage.v1"

TEACHING_TERMS = (
    "核心", "关键", "原则", "方法", "步骤", "流程", "问题", "需求", "客户", "信任", "成交", "复盘", "动作", "案例", "工具", "策略", "注意", "总结", "因为", "所以", "但是", "如果", "一定", "必须", "不要", "不能", "应该", "可以",
)
ACTION_TERMS = ("要", "需要", "可以", "先", "再", "最后", "准备", "记录", "复盘", "确认", "整理", "提问", "跟进", "判断", "避免", "建立", "设计")
EXPRESSION_TERMS = ("我", "你", "客户", "我们", "对方", "怎么", "为什么", "是不是", "能不能", "先", "不用", "不要")
SOFT_BOUNDARY_TERMS = ("首先", "第二", "第三", "第四", "最后", "接下来", "另外", "同时", "因为", "所以", "但是", "比如", "如果", "那么", "我们要", "大家要", "客户会", "客户可能", "这一步", "这个时候")
FILLERS = ("然后呢", "就是说", "也就是说", "对吧", "是不是", "对不对", "其实", "那么", "这个", "那个", "就是", "啊", "呃", "嗯", "哈")
NON_TOPIC_PATTERNS = (
    r"^(您已|已经)?(静音|解除静音|开启摄像头|关闭摄像头)$",
    r"^(开始|停止)?共享屏幕$",
    r"^(加入|离开|结束)会议$",
    r"^(会议|直播)(已经)?开始$",
    r"^网络(连接)?(不稳定|异常)$",
    r"^(正在)?录制(中)?$",
    r"^(下面|接下来)(有请|请).{0,18}(分享|发言)$",
    r"^感谢.{0,18}(分享|发言)$",
    r"^(业绩|排名|出勤)(播报|通报).{0,20}$",
)


def build_smart_summary_chapter_pack(
    bundle_dir: str | Path,
    *,
    title: str = "",
    write: bool = True,
    target_chapters: int = 8,
    max_visual_items: int = 120,
    chapter_mode: str = "semantic",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Build a chapter-level evidence layer for smart-summary generation.

    This is deterministic and local. It does not call an online model. The goal is
    to make the final smart summary consume chapter evidence instead of raw ASR
    chunks directly.
    """

    root = Path(bundle_dir).expanduser().resolve()
    progress = (
        LocalMediaProgress(
            pipeline="local_smart_summary_chapters",
            snapshot_path=root / "exports" / "smart-summary-chapters-progress.json",
            events_path=root / "exports" / "smart-summary-chapters-progress.jsonl",
            callback=progress_callback,
        )
        if write
        else None
    )
    if progress:
        progress.emit(stage="input", percent=0, message="Loading preserved smart-summary transcript input")
    pack = build_smart_summary_input_pack(root, title=title, write=write, max_visual_items=max_visual_items)
    note_title = title or str(pack.get("title") or root.name)
    segments = [row for row in pack.get("transcript_segments") or [] if isinstance(row, dict) and _segment_text(row)]
    visual_digest = pack.get("visual_digest") if isinstance(pack.get("visual_digest"), dict) else {}
    term_summary = pack.get("term_summary") if isinstance(pack.get("term_summary"), dict) else {}
    evidence_trace = pack.get("evidence_trace") if isinstance(pack.get("evidence_trace"), dict) else {}
    semantic_plan = build_semantic_chapter_plan(root, title=note_title, chapter_mode=chapter_mode, write=write)
    if progress:
        progress.emit(
            stage="chapter_plan",
            percent=45,
            current_item=len(segments),
            total_items=len(segments),
            message="Mapping preserved segments into explicit summary chapters",
        )
    chapters = _build_chapters(
        segments,
        visual_digest,
        evidence_trace=evidence_trace,
        target_chapters=max(1, int(target_chapters or 1)),
        chapter_plan=semantic_plan if chapter_mode == "semantic" else None,
    )
    duration_seconds = max((_seconds(row.get("end")) for row in segments), default=0.0)
    timeline_coverage_quality = evaluate_chapter_timeline_coverage(
        chapters,
        segments,
        duration_seconds=duration_seconds,
    )
    course_map = _course_map(note_title, chapters, term_summary, visual_digest)
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "title": note_title,
        "created_at": now_iso(),
        "transcript_source": pack.get("transcript_source") or "",
        "content_scope": pack.get("content_scope") if isinstance(pack.get("content_scope"), dict) else {},
        "chapter_mode": chapter_mode,
        "semantic_chapter_plan": str(root / "exports" / "semantic-chapter-plan.json") if chapter_mode == "semantic" else "",
        "chapter_count": len(chapters),
        "duration_seconds": duration_seconds,
        "course_map": course_map,
        "chapters": chapters,
        "timeline_coverage_quality": timeline_coverage_quality,
        "quality_notes": _quality_notes(pack, chapters, timeline_coverage_quality),
        "reader_summary_policy": {
            "chapter_map_is_final_summary": False,
            "global_reduce_required": True,
            "non_topic_noise_filtered": True,
            "non_topic_noise_categories": [
                "meeting_ui",
                "host_transition",
                "performance_announcement",
            ],
        },
        "artifacts": {
            "json": str(root / "exports" / "smart-summary-chapters.json"),
            "markdown": str(root / "exports" / "smart-summary-chapters.md"),
            "course_map_json": str(root / "exports" / "course-map.json"),
            "course_map_markdown": str(root / "exports" / "course-map.md"),
        },
        "write": bool(write),
    }
    replacements = load_bundle_term_replacements(root)
    if replacements:
        result = _apply_term_replacements_to_payload(result, replacements)
        course_map = result.get("course_map") if isinstance(result.get("course_map"), dict) else course_map
    # Raw slide words must bypass both browsing caps and term replacement.
    from .courseware_material import material_for_range
    material = pack.get("courseware_material", {"active": False})
    result["courseware_material"] = material
    timeline_path = root / "timeline.json"
    timeline = read_json(timeline_path) if timeline_path.is_file() else []
    if not isinstance(timeline, list):
        raise ValueError("citation_digest:timeline_list_required")
    for number, chapter in enumerate(result["chapters"]):
        chapter["courseware_material"] = material_for_range(material, chapter["start"], chapter["end"])
        # Raw evidence, identity and qualification must bypass terminology edits.
        chapter["evidence_trace"] = chapters[number]["evidence_trace"]
        chapter["citation_digest"] = digest_records([
            *chapters[number]["citation_digest"],
            *courseware_records(chapter["courseware_material"], timeline),
        ])
        chapter["evidence_trace"]["summary"]["citation_digest_items"] = len(chapter["citation_digest"])
    result["citation_digest_schema"] = DIGEST_SCHEMA
    result["citation_digest_binding"] = bind_citation_digest(root, result)
    if write:
        exports = root / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        write_json(exports / "smart-summary-chapters.json", result)
        (exports / "smart-summary-chapters.md").write_text(_render_chapters_markdown(result), encoding="utf-8")
        write_json(exports / "course-map.json", course_map)
        (exports / "course-map.md").write_text(_render_course_map_markdown(note_title, course_map), encoding="utf-8")
        manifest_path = root / "manifest.json"
        manifest = _read_mapping(manifest_path)
        manifest["smart_summary_chapters"] = "exports/smart-summary-chapters.json"
        manifest["smart_summary_citation_digest_schema"] = DIGEST_SCHEMA
        manifest["smart_summary_chapters_markdown"] = "exports/smart-summary-chapters.md"
        manifest["smart_summary_course_map"] = "exports/course-map.json"
        manifest["smart_summary_course_map_markdown"] = "exports/course-map.md"
        manifest["mcp_build_smart_summary_chapters_args"] = "mcp-build-smart-summary-chapters.args.json"
        write_json(root / "mcp-build-smart-summary-chapters.args.json", {"bundle_dir": str(root), "title": note_title, "write": True, "target_chapters": target_chapters, "max_visual_items": max_visual_items, "chapter_mode": chapter_mode})
        write_json(manifest_path, manifest)
        result["run_registry"] = _register_run(root, result, write=write)
        if progress:
            terminal = (
                "completed"
                if chapters and timeline_coverage_quality["passed"]
                else ("degraded" if chapters else "failed")
            )
            progress.emit(
                stage="finalize",
                percent=100,
                current_item=len(chapters),
                total_items=len(chapters),
                message=(
                    "Smart summary chapter pack completed"
                    if terminal == "completed"
                    else (
                        "Smart summary chapters need timeline coverage review"
                        if chapters
                        else "Smart summary chapter pack produced no chapters"
                    )
                ),
                status=terminal,
                output_paths=[exports / "smart-summary-chapters.json", exports / "course-map.json"],
                report_paths=[exports / "smart-summary-chapters.md"],
                details={"chapter_count": len(chapters), "source_segment_count": len(segments)},
            )
            result["progress"] = progress.artifacts()
            write_json(exports / "smart-summary-chapters.json", result)
    return result


def evaluate_chapter_timeline_coverage(
    chapters: list[dict[str, Any]],
    transcript_segments: list[dict[str, Any]],
    *,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Deterministically validate first/last coverage and timestamp ranges.

    Intent: adopt YouTube Digest's full-timeline chapter requirement as a
    machine gate. Decision: verify actual chapter timestamps against preserved
    transcript bounds rather than trust a prompt instruction. Reason: a model
    can produce plausible chapters while omitting the opening/ending or inventing
    out-of-range timestamps. Evidence: YouTube Digest 1.1.5
    ``prompts/analysis.md:12-13,43-60``. Effective scope: derived Smart Summary
    chapter readiness only; transcript and Timeline remain unchanged.
    """

    transcript_rows = [row for row in transcript_segments if isinstance(row, dict)]
    chapter_rows = [row for row in chapters if isinstance(row, dict)]
    if not transcript_rows or not chapter_rows:
        return {
            "schema": TIMELINE_COVERAGE_SCHEMA,
            "passed": False,
            "status": "missing_input",
            "checks": [
                {
                    "key": "input_present",
                    "passed": False,
                    "detail": f"chapters={len(chapter_rows)}, transcript_segments={len(transcript_rows)}",
                }
            ],
        }
    transcript_start = min(_seconds(row.get("start")) for row in transcript_rows)
    transcript_end = max(_seconds(row.get("end")) for row in transcript_rows)
    if duration_seconds is not None:
        transcript_end = max(transcript_end, _seconds(duration_seconds))
    span = max(0.0, transcript_end - transcript_start)
    tolerance = max(0.5, min(2.0, span * 0.002))
    ranges: list[tuple[float, float]] = [
        (_seconds(row.get("start")), _seconds(row.get("end"))) for row in chapter_rows
    ]
    timestamp_range_valid = all(
        start >= transcript_start - tolerance
        and end > start
        and end <= transcript_end + tolerance
        for start, end in ranges
    )
    ordered = all(
        ranges[index][0] >= ranges[index - 1][1] - tolerance
        for index in range(1, len(ranges))
    )
    first_limit = transcript_start + span * 0.25
    last_limit = transcript_start + span * 0.75
    first_covered = ranges[0][0] <= first_limit + tolerance
    last_covered = ranges[-1][1] >= last_limit - tolerance
    checks = [
        {
            "key": "timestamp_range",
            "passed": timestamp_range_valid,
            "detail": f"transcript={transcript_start:.3f}-{transcript_end:.3f}; tolerance={tolerance:.3f}",
        },
        {
            "key": "chapter_order",
            "passed": ordered,
            "detail": f"chapter_count={len(ranges)}",
        },
        {
            "key": "first_quarter_covered",
            "passed": first_covered,
            "detail": f"first_chapter_start={ranges[0][0]:.3f}; limit={first_limit:.3f}",
        },
        {
            "key": "last_quarter_covered",
            "passed": last_covered,
            "detail": f"last_chapter_end={ranges[-1][1]:.3f}; threshold={last_limit:.3f}",
        },
    ]
    passed = all(bool(row["passed"]) for row in checks)
    return {
        "schema": TIMELINE_COVERAGE_SCHEMA,
        "passed": passed,
        "status": "passed" if passed else "needs_review",
        "transcript_range": {"start": transcript_start, "end": transcript_end},
        "chapter_range": {"start": ranges[0][0], "end": ranges[-1][1]},
        "checks": checks,
    }



def attach_content_candidate_links(bundle_dir: str | Path, candidate_pack: dict[str, Any], *, write: bool = True) -> dict[str, Any]:
    """Attach content-candidate backlinks to smart-summary chapters.

    The candidate pack is generated downstream from chapter citation digests. This
    helper closes the reverse edge so chapter evidence can point back to reusable
    content material candidates without changing the extraction evidence itself.
    """

    root = Path(bundle_dir).expanduser().resolve()
    chapters_path = root / "exports" / "smart-summary-chapters.json"
    markdown_path = root / "exports" / "smart-summary-chapters.md"
    if not chapters_path.exists():
        return {
            "schema": "video_knowledge_pipeline.smart_summary_content_candidate_links.v1",
            "exists": False,
            "chapter_count": 0,
            "linked_candidate_count": 0,
            "json_path": str(chapters_path),
            "markdown_path": str(markdown_path),
            "write": bool(write),
        }
    payload = read_json(chapters_path)
    if not isinstance(payload, dict):
        return {
            "schema": "video_knowledge_pipeline.smart_summary_content_candidate_links.v1",
            "exists": False,
            "chapter_count": 0,
            "linked_candidate_count": 0,
            "json_path": str(chapters_path),
            "markdown_path": str(markdown_path),
            "write": bool(write),
        }
    chapters = payload.get("chapters") if isinstance(payload.get("chapters"), list) else []
    by_chapter: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidate_pack.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        refs = candidate.get("summary_chapter_refs") if isinstance(candidate.get("summary_chapter_refs"), list) else []
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            chapter_index = _safe_int(ref.get("chapter_index"))
            if chapter_index <= 0:
                continue
            by_chapter.setdefault(chapter_index, []).append(_compact_linked_content_candidate(candidate))
    linked_total = 0
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        chapter_index = _safe_int(chapter.get("index") or chapter.get("chapter_index"))
        links = _dedupe_candidate_links(by_chapter.get(chapter_index, []))
        chapter["linked_content_candidates"] = links
        trace = chapter.get("evidence_trace") if isinstance(chapter.get("evidence_trace"), dict) else {}
        summary = trace.get("summary") if isinstance(trace.get("summary"), dict) else {}
        summary["linked_content_candidate_count"] = len(links)
        if trace:
            trace["summary"] = summary
            chapter["evidence_trace"] = trace
        linked_total += len(links)
    payload["content_candidate_pack_linked"] = True
    payload["linked_content_candidate_count"] = linked_total
    payload["content_candidate_pack_path"] = str(root / "exports" / "content-candidate-pack.json")
    if write:
        write_json(chapters_path, payload)
        markdown_path.write_text(_render_chapters_markdown(payload), encoding="utf-8")
    return {
        "schema": "video_knowledge_pipeline.smart_summary_content_candidate_links.v1",
        "exists": True,
        "chapter_count": len([row for row in chapters if isinstance(row, dict)]),
        "linked_candidate_count": linked_total,
        "json_path": str(chapters_path),
        "markdown_path": str(markdown_path),
        "write": bool(write),
    }


def _compact_linked_content_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(candidate.get("id") or ""),
        "timeline_index": candidate.get("timeline_index"),
        "time_range": str(candidate.get("time_range") or ""),
        "candidate_types": [str(value) for value in candidate.get("candidate_types") or [] if str(value)],
        "viewpoint": _clip(candidate.get("viewpoint"), 180),
        "review_required": bool(candidate.get("review_required", True)),
        "publication_allowed": bool(candidate.get("publication_allowed", False)),
        "allowed_as_inspiration": bool(candidate.get("allowed_as_inspiration", True)),
    }


def _dedupe_candidate_links(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("id") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out[:12]

def _register_run(root: Path, result: dict[str, Any], *, write: bool) -> dict[str, Any]:
    chapters = [row for row in result.get("chapters") or [] if isinstance(row, dict)]
    failed_items: list[dict[str, Any]] = []
    if not chapters:
        failed_items.append({"id": "chapters", "reason": "missing_chapters", "detail": "No smart-summary chapters were generated."})
    coverage = result.get("timeline_coverage_quality")
    if chapters and isinstance(coverage, dict) and not bool(coverage.get("passed")):
        failed_checks = [
            str(row.get("key") or "")
            for row in coverage.get("checks") or []
            if isinstance(row, dict) and not bool(row.get("passed"))
        ]
        failed_items.append(
            {
                "id": "timeline_coverage",
                "reason": "chapter_timeline_coverage_failed",
                "detail": ", ".join(failed_checks),
            }
        )
    for chapter in chapters:
        trace = chapter.get("evidence_trace") if isinstance(chapter.get("evidence_trace"), dict) else {}
        summary = trace.get("summary") if isinstance(trace.get("summary"), dict) else {}
        if int(summary.get("review_gaps") or 0) > 0:
            failed_items.append(
                {
                    "id": chapter.get("index"),
                    "reason": "chapter_review_gaps",
                    "detail": str(chapter.get("title") or ""),
                }
            )
    if not chapters:
        status = "needs_input"
    elif failed_items:
        status = "needs_review"
    else:
        status = "completed"
    return register_bundle_run(
        root,
        run_type="smart_summary_chapter_pack",
        run_id="smart-summary-chapter-pack",
        status=status,
        title="Smart summary chapter pack",
        summary=f"Prepared {len(chapters)} chapters; {len(failed_items)} need review/input.",
        inputs={"input_pack": str(root / "exports" / "smart-summary-input-pack.json")},
        parameters={"chapter_count": len(chapters), "duration_seconds": result.get("duration_seconds")},
        artifacts=[
            {"key": "chapter_pack_json", "path": str(root / "exports" / "smart-summary-chapters.json")},
            {"key": "chapter_pack_markdown", "path": str(root / "exports" / "smart-summary-chapters.md")},
            {"key": "course_map_json", "path": str(root / "exports" / "course-map.json")},
            {"key": "course_map_markdown", "path": str(root / "exports" / "course-map.md")},
            {"key": "mcp_args", "path": str(root / "mcp-build-smart-summary-chapters.args.json")},
        ],
        failed_items=failed_items,
        retry_command=f".\\scripts\\video-knowledge.ps1 build-smart-summary-chapters {_ps_quote(str(root))}",
        next_actions=_chapter_run_next_actions(status),
        operator_boundary={
            "local_only": True,
            "no_cloud_call": True,
            "does_not_modify_raw_transcript": True,
            "purpose": "Prepare chapter-level evidence for Codex/LLM smart-summary generation.",
        },
        write=write,
    )


def _chapter_run_next_actions(status: str) -> list[str]:
    if status == "needs_input":
        return ["Build or import transcript evidence before creating smart-summary chapters."]
    if status == "needs_review":
        return ["Review chapter evidence gaps, then run smart-summary-section-workflow for rewrite planning."]
    return ["Run smart-summary-section-workflow or generate-smart-summary-with-codex to refresh final summary."]
def _build_chapters(
    segments: list[dict[str, Any]],
    visual_digest: dict[str, Any],
    *,
    evidence_trace: dict[str, Any] | None = None,
    target_chapters: int,
    chapter_plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not segments:
        return []
    planned = [row for row in (chapter_plan or {}).get("chapters") or [] if isinstance(row, dict)]
    buckets: list[list[dict[str, Any]]] = []
    if planned:
        for planned_chapter in planned:
            start = _seconds(planned_chapter.get("start"))
            end = _seconds(planned_chapter.get("end"))
            bucket = [
                row
                for row in segments
                if start <= ((_seconds(row.get("start")) + _seconds(row.get("end"))) / 2) < end
            ]
            if bucket:
                buckets.append(bucket)
    else:
        duration = max((_seconds(row.get("end")) for row in segments), default=0.0) or float(len(segments))
        target = max(1, min(target_chapters, len(segments)))
        buckets = [[] for _ in range(target)]
        for pos, row in enumerate(segments):
            start = _seconds(row.get("start"))
            bucket_index = min(target - 1, int((start / duration) * target)) if duration else min(target - 1, pos)
            buckets[bucket_index].append(row)
    chapters = []
    for idx, bucket in enumerate(buckets, start=1):
        if not bucket:
            continue
        start = _seconds(bucket[0].get("start"))
        end = _seconds(bucket[-1].get("end"))
        sentences = _ranked_sentences(bucket)
        visual_notes = _visual_notes_for_range(visual_digest, start, end)
        key_points = _ranked_snippets(bucket, TEACHING_TERMS, target=4)
        actions = _ranked_snippets(bucket, ACTION_TERMS, target=3)
        expressions = _ranked_snippets(bucket, EXPRESSION_TERMS, target=2)
        title = _chapter_title(idx, sentences, visual_notes)
        trace = _chapter_evidence_trace(evidence_trace or {}, bucket, start, end)
        citation_digest = _chapter_citation_digest(bucket, trace)
        trace["summary"]["citation_digest_items"] = len(citation_digest)
        chapters.append(
            {
                "index": idx,
                "title": title,
                "start": start,
                "end": end,
                "start_time": format_timestamp(start),
                "end_time": format_timestamp(end),
                "summary_sentences": sentences[:4],
                "visual_notes": visual_notes[:5],
                "key_points": key_points,
                "actions": actions,
                "reusable_expressions": expressions,
                "segment_count": len(bucket),
                "source_segment_ids": [
                    source_id
                    for row in bucket
                    for source_id in (row.get("source_segment_ids") or [row.get("segment_id")])
                    if source_id
                ],
                "transformations": [
                    {
                        "type": "explicit_summary_aggregation",
                        "source_segment_ids": [
                            source_id
                            for row in bucket
                            for source_id in (row.get("source_segment_ids") or [row.get("segment_id")])
                            if source_id
                        ],
                        "boundary_changed": False,
                    }
                ],
                "evidence_trace": trace,
                "citation_digest": citation_digest,
            }
        )
    return chapters



def _chapter_evidence_trace(evidence_trace: dict[str, Any], bucket: list[dict[str, Any]], start: float, end: float) -> dict[str, Any]:
    indexes = _unique_ints([_safe_int(row.get("timeline_index")) for row in bucket])
    wanted = set(indexes)
    ocr_items = _filter_trace_items(evidence_trace.get("ocr_items"), wanted, start=start, end=end)
    visual_items = _filter_trace_items(evidence_trace.get("visual_items"), wanted, start=start, end=end)
    tile_items = _filter_trace_items(evidence_trace.get("tile_items"), wanted, start=start, end=end)
    temporal_items = _filter_trace_items(evidence_trace.get("temporal_items"), wanted, start=start, end=end)
    continuous_tag_items = _filter_trace_items(evidence_trace.get("continuous_tag_items"), wanted, start=start, end=end)
    review_gaps = _filter_trace_items(evidence_trace.get("review_gaps"), wanted, start=start, end=end)
    transcript_corrections = [
        {
            "segment_id": str(row.get("segment_id") or ""),
            "source_segment_ids": [
                str(value)
                for value in (row.get("source_segment_ids") or [])
                if str(value)
            ],
            **dict(row.get("semantic_correction_provenance") or {}),
        }
        for row in bucket
        if isinstance(row.get("semantic_correction_provenance"), dict)
    ]
    moment_chunks = []
    for chunk in evidence_trace.get("moment_chunks") or []:
        if not isinstance(chunk, dict):
            continue
        chunk_indexes = [_safe_int(value) for value in (chunk.get("timeline_indexes") or [])]
        overlap = [value for value in chunk_indexes if value in wanted]
        if overlap or _overlaps(start, end, _seconds(chunk.get("start")), _seconds(chunk.get("end"))):
            moment_chunks.append({**chunk, "timeline_indexes": overlap or chunk_indexes})
    return {
        "transcript_source": evidence_trace.get("transcript_source") or "",
        "transcript_path": evidence_trace.get("transcript_path") or "",
        "transcript_sha256": evidence_trace.get("transcript_sha256") or "",
        "semantic_correction": evidence_trace.get("semantic_correction") or {},
        "transcript_corrections": transcript_corrections,
        "timeline_indexes": indexes,
        "ocr_items": ocr_items,
        "visual_items": visual_items,
        "tile_items": tile_items,
        "temporal_items": temporal_items,
        "continuous_tag_items": continuous_tag_items,
        "moment_chunks": moment_chunks,
        "review_gaps": review_gaps,
        "summary": {
            "transcript_segments": len(bucket),
            "timeline_indexes": len(indexes),
            "ocr_or_ebook_items": len(ocr_items),
            "high_res_tile_items": len(tile_items),
            "visual_understanding_items": len(visual_items),
            "temporal_understanding_items": len(temporal_items),
            "continuous_tag_weak_evidence_items": len(continuous_tag_items),
            "moment_chunks": len(moment_chunks),
            "review_gaps": len(review_gaps),
            "semantically_corrected_segments": len(transcript_corrections),
        },
    }



def _chapter_citation_digest(bucket: list[dict[str, Any]], trace: dict[str, Any]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for row in bucket:
        text = str(row.get("punctuated_text") or row.get("corrected_text") or row.get("raw_text") or "")
        if not text:
            continue
        timeline_index = _safe_int(row.get("timeline_index"))
        citations.append(
            {
                "source_type": "transcript",
                "source_locators": row.get('source_locators') or [],
                "source_observation_text": row.get('source_observation_text') or '',
                "source_context": {'scope': row.get('source_locator_scope') or 'unlocated_projection'},
                "excerpt_scope": "summary_projection_not_verbatim_source",
                "source_path": str(trace.get("transcript_path") or ""),
                "source_sha256": str(trace.get("transcript_sha256") or ""),
                "segment_id": str(row.get("segment_id") or ""),
                "source_segment_ids": [
                    str(value)
                    for value in (row.get("source_segment_ids") or [])
                    if str(value)
                ],
                "time": row.get("start_time") or format_timestamp(_seconds(row.get("start"))),
                "timeline_indexes": [timeline_index] if timeline_index else [],
                "text": text,
                "evidence_paths": [],
                "semantic_correction": (
                    dict(row.get("semantic_correction_provenance") or {})
                    if isinstance(row.get("semantic_correction_provenance"), dict)
                    else {}
                ),
            }
        )
    for chunk in trace.get("moment_chunks") or []:
        if not isinstance(chunk, dict):
            continue
        keywords = ", ".join(str(value) for value in (chunk.get("keywords") or []))
        text = keywords or f"moment chunk {chunk.get('chunk_index')}"
        citations.append(
            {
                "source_type": "moment",
                "excerpt_scope": "moment_keywords_only_not_full_observation",
                "time": f"{chunk.get('start_time')} - {chunk.get('end_time')}",
                "timeline_indexes": [int(value) for value in chunk.get("timeline_indexes") or [] if _safe_int(value)],
                "text": text,
                "evidence_paths": [str(value) for value in chunk.get("evidence_paths") or [] if str(value)],
            }
        )
    for source_type, key in (
        ("ocr_or_ebook", "ocr_items"),
        ("high_res_tile", "tile_items"),
        ("visual_understanding", "visual_items"),
        ("temporal_understanding", "temporal_items"),
        ("continuous_tag_weak_evidence", "continuous_tag_items"),
    ):
        for item in trace.get(key) or []:
            if not isinstance(item, dict):
                continue
            excerpt = str(item.get("excerpt") or "")
            if not excerpt:
                continue
            timeline_index = _safe_int(item.get("timeline_index"))
            from .citation_digest import evidence_level as citation_level
            evidence_level = citation_level({**item, "source_type": source_type})
            citation_source_type = f"{source_type}_inferred" if evidence_level == "inferred" else source_type
            citations.append(
                {
                    "source_type": citation_source_type,
                    "evidence_level": evidence_level,
                    "inference_basis": str(item.get("inference_basis") or ""),
                    "source_context": item.get("source_context") or {},
                    "source_locators": item.get("source_locators") or [],
                    "recognition_provenance": item.get("recognition_provenance") or {},
                    "time": f"{item.get('start_time')} - {item.get('end_time')}",
                    "timeline_indexes": [timeline_index] if timeline_index else [],
                    "text": excerpt,
                    "evidence_paths": [str(value) for value in item.get("evidence_paths") or [] if str(value)],
                }
            )
    for gap in trace.get("review_gaps") or []:
        if not isinstance(gap, dict):
            continue
        reasons = ", ".join(str(value) for value in gap.get("reasons") or [] if str(value))
        if not reasons:
            continue
        timeline_index = _safe_int(gap.get("timeline_index"))
        citations.append(
            {
                "source_type": "review_gap",
                "evidence_level": "unavailable" if any("seen_unavailable" in str(value) for value in gap.get("reasons") or []) else "not_seen",
                "time": f"{gap.get('start_time')} - {gap.get('end_time')}",
                "timeline_indexes": [timeline_index] if timeline_index else [],
                "text": reasons,
                "evidence_paths": [str(value) for value in gap.get("evidence_paths") or [] if str(value)],
            }
        )
    return _dedupe_citations(citations)


def _dedupe_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return digest_records(citations)

def _filter_trace_items(items: Any, wanted: set[int], *, start=None, end=None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        if _safe_int(item.get("timeline_index")) not in wanted:
            continue
        # An index is navigation, not identity. Repeated labels at another time
        # must not attach the wrong page to this chapter. Untimed legacy rows
        # remain compatible and explicitly lack precise source proof.
        if (start is not None and end is not None and 'start' in item and 'end' in item
                and not _overlaps(start, end, _seconds(item['start']), _seconds(item['end']))):
            continue
        out.append(item)
    return out



def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _unique_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
def _course_map(title: str, chapters: list[dict[str, Any]], term_summary: dict[str, Any], visual_digest: dict[str, Any]) -> dict[str, Any]:
    high_terms = []
    for row in term_summary.get("high_confidence_terms") or []:
        if isinstance(row, dict) and row.get("canonical_term"):
            high_terms.append(str(row.get("canonical_term")))
    visual_count = int(visual_digest.get("total_items_with_visual_digest") or 0)
    return {
        "title": title,
        "main_question": _main_question(title),
        "mainline": _mainline(title, chapters),
        "topics": [
            {
                "chapter_index": chapter.get("index"),
                "time_range": f"{chapter.get('start_time')} - {chapter.get('end_time')}",
                "title": chapter.get("title"),
                "role": _topic_role(chapter, len(chapters)),
            }
            for chapter in chapters
        ],
        "high_confidence_terms": _dedupe(high_terms)[:30],
        "visual_evidence_count": visual_count,
        "visual_boundary": "visual/courseware evidence included but still review-required" if visual_count else "visual evidence not reliable or not executed",
    }


def _main_question(title: str) -> str:
    clean = _clean(title)
    if clean:
        return f"这节课主要回答：{clean} 应该如何理解、拆解并落到行动？"
    return "这节课主要回答一个待整理的课程主题如何理解、拆解并落到行动。"


def _mainline(title: str, chapters: list[dict[str, Any]]) -> str:
    if not chapters:
        return f"围绕《{title}》整理课程主线。"
    first = chapters[0].get("title") or "开头问题"
    middle = chapters[len(chapters) // 2].get("title") or "中段展开"
    last = chapters[-1].get("title") or "结尾落点"
    return f"课程从“{first}”进入，中段展开“{middle}”，最后落到“{last}”。"


def _topic_role(chapter: dict[str, Any], total: int) -> str:
    index = int(chapter.get("index") or 0)
    if index <= 1:
        return "opening_context"
    if index >= total:
        return "closing_takeaway"
    if chapter.get("actions"):
        return "operation_or_method"
    if chapter.get("visual_notes"):
        return "courseware_explanation"
    return "concept_development"


def _ranked_sentences(bucket: list[dict[str, Any]]) -> list[str]:
    rows = []
    for row in bucket:
        for sentence in _split_sentences(_segment_text(row)):
            if _is_non_topic_noise(sentence):
                continue
            score = _sentence_score(sentence, TEACHING_TERMS)
            rows.append((score, sentence))
    rows.sort(key=lambda item: (-item[0], len(item[1])))
    return _dedupe([_clip(text, 150) for _, text in rows])


def _ranked_snippets(bucket: list[dict[str, Any]], keywords: tuple[str, ...], *, target: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in bucket:
        start = _seconds(row.get("start"))
        for sentence in _split_sentences(_segment_text(row)):
            if _is_non_topic_noise(sentence):
                continue
            score = _sentence_score(sentence, keywords)
            if score <= 0:
                continue
            rows.append({"time": format_timestamp(start), "seconds": start, "text": _clip(sentence, 120), "score": score})
    rows.sort(key=lambda item: (-int(item.get("score") or 0), float(item.get("seconds") or 0)))
    out = []
    seen = set()
    for row in rows:
        key = re.sub(r"\W+", "", row["text"].lower())[:36]
        if key in seen:
            continue
        seen.add(key)
        out.append({"time": row["time"], "seconds": row["seconds"], "text": row["text"]})
        if len(out) >= target:
            break
    return out


def _visual_notes_for_range(visual_digest: dict[str, Any], start: float, end: float) -> list[dict[str, Any]]:
    items = visual_digest.get("items") if isinstance(visual_digest.get("items"), list) else []
    notes: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_start = _seconds(item.get("start"))
        item_end = _seconds(item.get("end"))
        if item_end < start or item_start > end:
            continue
        text = _visual_text(item)
        if not text or _is_non_topic_noise(text):
            continue
        notes.append({"time": item.get("start_time") or format_timestamp(item_start), "timeline_index": item.get("timeline_index"), "route": item.get("visual_route") or "", "text": text})
    return notes


def _visual_text(item: dict[str, Any]) -> str:
    # Incidental packaging/signage remains in the visual evidence digest, but
    # cannot supply a spoken-content chapter title or topic boundary.
    if item.get("visual_text_role") == "incidental_background":
        return ""
    for key in ("structured_visual", "visual_text", "visual_understanding", "temporal_visual_understanding"):
        value = str(item.get(key) or "").strip()
        if value:
            headings = re.findall(r"#{1,4}\s*([^#|]{2,60})", value)
            if headings:
                return "、".join(_dedupe([_clip(h, 42) for h in headings])[:3])
            value = re.sub(r"source:\s*ebook_markdown_pipeline[；;]?", "", value)
            value = re.sub(r"type:\s*structured_visual[；;]?", "", value)
            value = re.sub(r"markdown:\s*", "", value)
            return _clip(value, 140)
    return ""


def _chapter_title(index: int, sentences: list[str], visual_notes: list[dict[str, Any]]) -> str:
    if visual_notes:
        text = str(visual_notes[0].get("text") or "").strip()
        if text and not _is_non_topic_noise(text):
            return _clip(text, 28)
    if sentences:
        for sentence in sentences:
            if not _is_non_topic_noise(sentence):
                return _clip(sentence, 28)
    return f"第 {index} 部分（主题待提炼）"


def _is_non_topic_noise(value: Any) -> bool:
    """Filter UI/host boilerplate without modifying the source transcript."""
    text = re.sub(r"[\s，,。.!！?？:：;；]+", "", str(value or "").strip())
    if not text or len(text) > 48:
        return False
    return any(re.fullmatch(pattern, text) for pattern in NON_TOPIC_PATTERNS)


def _quality_notes(
    pack: dict[str, Any],
    chapters: list[dict[str, Any]],
    timeline_coverage_quality: dict[str, Any],
) -> list[str]:
    notes = []
    if not chapters:
        notes.append("No chapters were generated; smart summary should not be treated as final.")
    if pack.get("quality_notes"):
        notes.extend(str(note) for note in pack.get("quality_notes") or [])
    if chapters and not bool(timeline_coverage_quality.get("passed")):
        failed = [
            str(row.get("key") or "")
            for row in timeline_coverage_quality.get("checks") or []
            if isinstance(row, dict) and not bool(row.get("passed"))
        ]
        notes.append(
            "Chapter timeline coverage requires review: " + ", ".join(failed)
        )
    return _dedupe(notes)


def _render_chapters_markdown(result: dict[str, Any]) -> str:
    coverage = result.get("timeline_coverage_quality") if isinstance(result.get("timeline_coverage_quality"), dict) else {}
    lines = [f"# Smart Summary Chapters: {result.get('title')}", "", f"- Created: `{result.get('created_at')}`", f"- Transcript source: `{result.get('transcript_source')}`", f"- Chapter count: `{result.get('chapter_count')}`", f"- Timeline coverage: `{coverage.get('status') or 'unknown'}`", ""]
    for chapter in result.get("chapters") or []:
        trace = chapter.get("evidence_trace") if isinstance(chapter.get("evidence_trace"), dict) else {}
        trace_summary = trace.get("summary") if isinstance(trace.get("summary"), dict) else {}
        lines.extend([f"## {chapter.get('index')}. {chapter.get('title')}", "", f"- Time: `{chapter.get('start_time')} - {chapter.get('end_time')}`", f"- Evidence: transcript=`{trace_summary.get('transcript_segments') or 0}`, OCR/ebook=`{trace_summary.get('ocr_or_ebook_items') or 0}`, high-res tile=`{trace_summary.get('high_res_tile_items') or 0}`, visual=`{trace_summary.get('visual_understanding_items') or 0}`, temporal=`{trace_summary.get('temporal_understanding_items') or 0}`, moments=`{trace_summary.get('moment_chunks') or 0}`, review gaps=`{trace_summary.get('review_gaps') or 0}`", "", "### Summary", ""])
        for sentence in chapter.get("summary_sentences") or []:
            lines.append(f"- {sentence}")
        if chapter.get("citation_digest"):
            from .citation_digest import preview_note, evidence_level
            citations = chapter["citation_digest"]
            lines.extend(["", "### Citation Digest", "", preview_note(min(24, len(citations)), len(citations)), "", "| Type / Level | Time | Timeline | Evidence | Text |", "| --- | --- | --- | --- | --- |"])
            for citation in citations[:24]:
                lines.append(
                    "| {source_type} | {time} | {timeline} | {evidence} | {text} |".format(
                        source_type=_md(str(citation.get("source_type")) + " / " + evidence_level(citation)),
                        time=_md(citation.get("time")),
                        timeline=_md(", ".join(str(value) for value in citation.get("timeline_indexes") or [])),
                        evidence=_md("; ".join(str(value) for value in citation.get("evidence_paths") or [])),
                        text=_md(_clip(citation.get("text"), 240)),
                    )
                )
        if chapter.get("linked_content_candidates"):
            lines.extend(["", "### Linked Content Candidates", "", "| ID | Time | Types | Review | Viewpoint |", "| --- | --- | --- | --- | --- |"])
            for candidate in chapter.get("linked_content_candidates") or []:
                lines.append(
                    "| {id} | {time} | {types} | {review} | {viewpoint} |".format(
                        id=_md(candidate.get("id")),
                        time=_md(candidate.get("time_range")),
                        types=_md(", ".join(str(value) for value in candidate.get("candidate_types") or [])),
                        review=_md("review_required" if candidate.get("review_required") else "review_optional"),
                        viewpoint=_md(candidate.get("viewpoint")),
                    )
                )
        if chapter.get("visual_notes"):
            lines.extend(["", "### Visual / Courseware", ""])
            for note in chapter.get("visual_notes") or []:
                lines.append(f"- `{note.get('time')}` {note.get('text')}")
        if chapter.get("key_points"):
            lines.extend(["", "### Key Points", ""])
            for row in chapter.get("key_points") or []:
                lines.append(f"- `{row.get('time')}` {row.get('text')}")
        if chapter.get("actions"):
            lines.extend(["", "### Actions", ""])
            for row in chapter.get("actions") or []:
                lines.append(f"- `{row.get('time')}` {row.get('text')}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_course_map_markdown(title: str, course_map: dict[str, Any]) -> str:
    lines = [f"# Course Map: {title}", "", f"- Main question: {course_map.get('main_question')}", f"- Mainline: {course_map.get('mainline')}", f"- Visual boundary: `{course_map.get('visual_boundary')}`", "", "## Topics", ""]
    for topic in course_map.get("topics") or []:
        lines.append(f"- `{topic.get('time_range')}` {topic.get('title')} (`{topic.get('role')}`)")
    terms = course_map.get("high_confidence_terms") or []
    if terms:
        lines.extend(["", "## High Confidence Terms", "", "- " + ", ".join(str(term) for term in terms)])
    return "\n".join(lines).rstrip() + "\n"


def _segment_text(row: dict[str, Any]) -> str:
    return _clean(row.get("punctuated_text") or row.get("corrected_text") or row.get("raw_text"))


def _split_sentences(text: str) -> list[str]:
    cleaned = _clean(text)
    if not cleaned:
        return []
    for term in SOFT_BOUNDARY_TERMS:
        cleaned = cleaned.replace(term, "。" + term)
    cleaned = re.sub(r"。+", "。", cleaned).strip("。")
    parts = re.split(r"[。！？!?；;\n]+", cleaned)
    out: list[str] = []
    for part in parts:
        value = part.strip(" ，,、：:")
        if 8 <= len(value) <= 180:
            out.append(value)
        elif len(value) > 180:
            out.extend(_fixed_chunks(value, max_chars=86))
    return _dedupe(out)


def _sentence_score(sentence: str, keywords: tuple[str, ...]) -> int:
    value = str(sentence or "")
    score = sum(2 for key in keywords if key in value)
    score += sum(1 for key in TEACHING_TERMS if key in value)
    if 18 <= len(value) <= 100:
        score += 3
    if len(value) < 10 or len(value) > 160:
        score -= 2
    return score


def _fixed_chunks(text: str, *, max_chars: int) -> list[str]:
    value = str(text or "").strip()
    chunks: list[str] = []
    while len(value) > max_chars:
        cut = max(value.rfind("，", 0, max_chars), value.rfind("、", 0, max_chars), value.rfind(" ", 0, max_chars))
        if cut < max_chars // 2:
            cut = max_chars
        chunks.append(value[:cut].strip(" ，,、"))
        value = value[cut:].strip(" ，,、")
    if value:
        chunks.append(value)
    return [chunk for chunk in chunks if len(chunk) >= 8]


def _clean(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    for filler in FILLERS:
        text = text.replace(filler, "")
    text = re.sub(r"[，,、]{2,}", "，", text)
    text = re.sub(r"^(所以|但是|然后|那么|那|比如说|注意|其实|就是|这个|这个呢|那么呢|好)+", "", text)
    return re.sub(r"\s+", " ", text).strip(" ，,、。")


def _clip(value: Any, limit: int) -> str:
    text = _clean(value)
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip(" ，,、；;") + "…"


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def _seconds(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except Exception:
        return 0.0


def _dedupe(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _apply_term_replacements_to_payload(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        return apply_term_replacement_pairs(value, replacements)
    if isinstance(value, list):
        return [_apply_term_replacements_to_payload(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _apply_term_replacements_to_payload(item, replacements) for key, item in value.items()}
    return value
