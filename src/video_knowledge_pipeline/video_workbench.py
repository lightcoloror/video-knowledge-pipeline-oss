from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .powershell import quote_powershell_literal as _ps_quote
from .models import now_iso
from .panorama_workbench import export_panorama_workbench, panorama_panel_html
from .run_artifact_registry import build_run_artifact_registry, register_bundle_run
from .storage import read_json, read_jsonl, write_json
from .storage import read_json_object_or_empty as _read_optional_object
from .task_console import _build_processing_queue, _build_subqueue_action_plan, _compact_moment_index, _load_run_registry_for_console
from .task_console import _summary_claim_review_status, _summary_claim_review_panel_html
from .task_console import _citation_digest_status, _citation_digest_panel_html
from .task_console import _summary_stage_status, _summary_stage_panel_html
from .term_correction_status import term_correction_status as _term_correction_status
from .transcript_semantic_correction import transcript_semantic_correction_status as _semantic_correction_status
from .transcript import format_timestamp
from .video_decomposition import video_decomposition_report_status
from .creator_report import creator_report_status
from .visual_evidence import evidence_level_counts, temporal_evidence_state, visual_evidence_state
from .shot_review_workbench import prepare_shot_review_workbench
from .subtitle_editor_ui import prepare_subtitle_editor
from .ui_shell_assets import install_ui_shell_assets
from .ui_surface_registry import export_ui_surface_registry, surface_artifact_cards

SCHEMA = "video_knowledge_pipeline.video_workbench.v1"


def _autopilot_bundle_summary(root: Path) -> dict[str, Any]:
    from .autopilot import autopilot_bundle_summary

    return autopilot_bundle_summary(root)


def export_video_workbench(bundle_dir: str | Path, *, write: bool = True) -> dict[str, Any]:
    """Write VKP's primary local workbench and specialist-surface navigator.

    The workbench remains a static local HTML entrypoint. It inventories and
    links existing VKP surfaces instead of replacing their domain-specific
    behavior, writeback service, or evidence contract.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline.json not found: {timeline_path}")
    manifest = read_json(manifest_path)
    timeline = read_json(timeline_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    if not isinstance(timeline, list):
        raise ValueError("timeline.json must be a JSON array")
    alignment_by_index = _timeline_alignment_by_index(root)
    transcript_arbitration = _transcript_arbitration_summary(root, manifest)
    term_correction_impact = _term_correction_impact_summary(root, manifest)
    term_correction = _term_correction_status(root)
    semantic_correction = _semantic_correction_status(root, write=False)
    transcript_arbitration["term_correction_impact"] = term_correction_impact
    arbitration_by_index = _transcript_arbitration_by_index(transcript_arbitration)
    rows = [
        _timeline_row(item, position, alignment_by_index=alignment_by_index, arbitration_by_index=arbitration_by_index)
        for position, item in enumerate(timeline, start=1)
        if isinstance(item, dict)
    ]
    subtitle_editor_status: dict[str, Any] = {"status": "not_prepared"}
    if write:
        try:
            subtitle_editor_status = prepare_subtitle_editor(root, write=True)
            manifest = read_json(manifest_path)
            if not isinstance(manifest, dict):
                raise ValueError("manifest.json must remain a JSON object after subtitle editor preparation")
        except (FileNotFoundError, ValueError) as exc:
            subtitle_editor_status = {
                "status": "blocked_missing_subtitle_editor_input",
                "reason": str(exc),
            }
    panorama = export_panorama_workbench(root, write=write)
    if write:
        refreshed_manifest = read_json(manifest_path)
        if isinstance(refreshed_manifest, dict):
            manifest = refreshed_manifest
    ui_surface_registry = export_ui_surface_registry(
        root,
        manifest,
        current_surface="video_workbench",
        write=write,
    )
    ui_shell_assets = install_ui_shell_assets(root, write=write)
    artifacts = _artifact_cards(root, manifest, ui_surface_registry=ui_surface_registry)
    review_closure = _review_closure_summary(root)
    evidence_status = _evidence_status_summary(root, [item for item in timeline if isinstance(item, dict)])
    moment_index = _compact_moment_index(_read_optional_object(root / "exports" / "video-moment-index.json"))
    video_rag_chunks = _compact_video_rag_chunks(root / "exports" / "video-rag-chunks.jsonl")
    video_rag_status = _video_rag_status(root, manifest, video_rag_chunks)
    provider_status = _provider_status_summary(root, manifest)
    content_candidates = _content_candidate_pack_summary(root, manifest, moment_index=moment_index, video_rag_chunks=video_rag_chunks)
    creative_workflow = _creative_workflow_summary(root, manifest)
    run_registry = _load_run_registry_for_console(root)
    processing_queue = _build_processing_queue(root, run_registry)
    external_reuse_status = _external_reuse_status_summary(root, manifest, run_registry, processing_queue)
    subqueue_action_plan = _build_subqueue_action_plan(processing_queue)
    shot_review = prepare_shot_review_workbench(root, write=write)
    autopilot = _autopilot_bundle_summary(root)
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "title": str(manifest.get("title") or root.name),
        "generated_at": now_iso(),
        "timeline_count": len(rows),
        "artifacts": artifacts,
        "review_closure": review_closure,
        "transcript_arbitration": transcript_arbitration,
        "term_correction_impact": term_correction_impact,
        "term_correction": term_correction,
        "semantic_correction": semantic_correction,
        "evidence_status": evidence_status,
        "moment_index": moment_index,
        "video_rag_chunks": video_rag_chunks,
        "video_rag_status": video_rag_status,
        "provider_status": provider_status,
        "content_candidates": content_candidates,
        "creative_workflow": creative_workflow,
        "run_registry": run_registry,
        "processing_queue": processing_queue,
        "external_reuse_status": external_reuse_status,
        "subqueue_action_plan": subqueue_action_plan,
        "shot_review": shot_review,
        "autopilot": autopilot,
        "summary_claim_review": _summary_claim_review_status(root),
        "summary_stage_coverage": _summary_stage_status(root),
        "citation_digest_integrity": _citation_digest_status(root),

        "panorama": panorama,
        "subtitle_editor_status": subtitle_editor_status,
        "ui_surface_registry": ui_surface_registry,
        "ui_shell_assets": ui_shell_assets,
        "paths": {
            "html": str(root / "video-workbench.html"),
            "json": str(root / "video-workbench.json"),
            "mcp_args": str(root / "mcp-video-workbench.args.json"),
            "ui_surface_registry_json": str(root / "ui-surface-registry.json"),
            "ui_surface_registry_markdown": str(root / "ui-surface-registry.md"),
        },
        "operator_boundary": {
            "static_local_html": True,
            "no_cloud_call": True,
            "no_media_processing": True,
            "purpose": "Primary local entrypoint for VKP status, timeline navigation, and specialist processing, review, editing, evaluation, and settings surfaces.",
        },
    }
    if write:
        build_run_artifact_registry(root, write=True)
        run_registry = _load_run_registry_for_console(root)
        processing_queue = _build_processing_queue(root, run_registry)
        result["run_registry"] = run_registry
        result["processing_queue"] = processing_queue
        result["external_reuse_status"] = _external_reuse_status_summary(root, manifest, run_registry, processing_queue)
        result["subqueue_action_plan"] = _build_subqueue_action_plan(processing_queue)
        write_json(root / "video-workbench.json", {**result, "timeline": rows})
        (root / "video-workbench.html").write_text(_render_workbench_html(result, rows), encoding="utf-8")
        write_json(root / "mcp-video-workbench.args.json", {"bundle_dir": str(root), "write": True})
        manifest["video_workbench_html"] = "video-workbench.html"
        manifest["video_workbench_json"] = "video-workbench.json"
        manifest["mcp_video_workbench_args"] = "mcp-video-workbench.args.json"
        manifest["ui_surface_registry_json"] = "ui-surface-registry.json"
        manifest["ui_surface_registry_markdown"] = "ui-surface-registry.md"
        manifest["video_workbench_updated_at"] = result["generated_at"]
        write_json(manifest_path, manifest)
        register_bundle_run(
            root,
            run_type="video_workbench",
            run_id="video-workbench",
            status="completed",
            title="视频知识工作台",
            summary=f"Static workbench with {len(rows)} timeline rows and {len(artifacts)} artifact links.",
            artifacts=[
                {"key": "html", "path": root / "video-workbench.html"},
                {"key": "json", "path": root / "video-workbench.json"},
                {"key": "mcp_args", "path": root / "mcp-video-workbench.args.json"},
                {"key": "ui_surface_registry_json", "path": root / "ui-surface-registry.json"},
                {"key": "ui_surface_registry_markdown", "path": root / "ui-surface-registry.md"},
            ],
            retry_command=f".\\scripts\\video-knowledge.ps1 export-video-workbench {root}",
            operator_boundary=result["operator_boundary"],
            write=True,
        )
        build_run_artifact_registry(root, write=True)
        run_registry = _load_run_registry_for_console(root)
        processing_queue = _build_processing_queue(root, run_registry)
        result["run_registry"] = run_registry
        result["processing_queue"] = processing_queue
        result["external_reuse_status"] = _external_reuse_status_summary(root, manifest, run_registry, processing_queue)
        result["subqueue_action_plan"] = _build_subqueue_action_plan(processing_queue)
        write_json(root / "video-workbench.json", {**result, "timeline": rows})
        (root / "video-workbench.html").write_text(_render_workbench_html(result, rows), encoding="utf-8")
    return result


def _timeline_row(
    item: dict[str, Any],
    position: int,
    *,
    alignment_by_index: dict[int, dict[str, Any]] | None = None,
    arbitration_by_index: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    index = _row_index(item, position)
    start = _seconds(item.get("review_start") or item.get("start"))
    end = _seconds(item.get("end"), default=start)
    text = _text(item.get("corrected_transcript") or item.get("transcript") or item.get("asr_text") or item.get("text"))
    visual = _text(item.get("visual_text") or item.get("ocr_text"))
    visual_state = visual_evidence_state(item)
    temporal_state = temporal_evidence_state(item)
    quality_issues = [str(value) for value in (item.get("quality_issues") or []) if str(value)]
    tile_targets = [value for value in (item.get("tile_review_targets") or []) if isinstance(value, dict)]
    alignment = (alignment_by_index or {}).get(index, {})
    arbitration = (arbitration_by_index or {}).get(index, {})
    evidence_flags: list[str] = []
    if alignment.get("issues"):
        evidence_flags.append("timeline_alignment_issue")
    if "needs_high_res_tile_recovery" in quality_issues:
        evidence_flags.append("needs_high_res_tile_recovery")
    if tile_targets or "tile_result_needs_review" in quality_issues:
        evidence_flags.append("tile_result_needs_review")
    if item.get("needs_human_review"):
        evidence_flags.append("needs_human_review")
    if arbitration:
        evidence_flags.append("transcript_source_conflict")
    if "inferred" in {visual_state["evidence_level"], temporal_state["evidence_level"]}:
        evidence_flags.append("inferred")
    if "unavailable" in {visual_state["evidence_level"], temporal_state["evidence_level"]}:
        evidence_flags.append("unavailable")
    return {
        "index": index,
        "start": start,
        "end": end,
        "start_time": format_timestamp(start),
        "end_time": format_timestamp(end),
        "route": _text(item.get("visual_route") or item.get("route") or "unknown"),
        "transcript": text,
        "visual_text": visual,
        "has_visual_understanding": bool(visual_state["consumable"]),
        "has_temporal_understanding": bool(temporal_state["consumable"]),
        "visual_evidence_level": visual_state["evidence_level"],
        "temporal_evidence_level": temporal_state["evidence_level"],
        "needs_human_review": bool(item.get("needs_human_review")),
        "quality_issues": quality_issues,
        "evidence_flags": evidence_flags,
        "timeline_alignment": _compact_alignment(alignment),
        "transcript_arbitration": _compact_transcript_arbitration_row(arbitration),
        "tile_review_targets": _compact_tile_targets(tile_targets),
        "frame_paths": [str(value) for value in (item.get("frame_paths") or []) if str(value)][:4],
    }


def _row_index(item: dict[str, Any], position: int) -> int:
    raw = item.get("index")
    if raw is None or raw == "":
        return position
    try:
        return int(raw)
    except Exception:
        return position


def _timeline_alignment_by_index(root: Path) -> dict[int, dict[str, Any]]:
    payload = _read_optional_object(root / "timeline-alignment-audit.json")
    rows = payload.get("items") if isinstance(payload.get("items"), list) else []
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("index"))
        except Exception:
            continue
        result[index] = row
    return result



def _transcript_arbitration_summary(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    rel = str(manifest.get("transcript_source_arbitration_json") or "transcript-source-arbitration.json")
    json_path = _bundle_path(root, rel)
    md_path = _bundle_path(root, str(manifest.get("transcript_source_arbitration_markdown") or "transcript-source-arbitration.md"))
    base = {
        "exists": json_path.exists(),
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "json_href": _relative_href(root, json_path),
        "markdown_href": _relative_href(root, md_path),
        "summary": {},
        "quality_summary": {},
        "review_rows": [],
        "changed_rows": [],
        "status": "missing_report",
    }
    if not json_path.exists():
        return base
    payload = _read_optional_object(json_path)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    quality_summary = payload.get("quality_summary") if isinstance(payload.get("quality_summary"), dict) else {}
    if not quality_summary and isinstance(summary.get("quality_summary"), dict):
        quality_summary = summary.get("quality_summary")
    summary_input_policy = quality_summary.get("summary_input_policy") if isinstance(quality_summary.get("summary_input_policy"), dict) else {}
    review_segment_refs = [_compact_transcript_arbitration_row(row) for row in (quality_summary.get("review_segment_refs") or []) if isinstance(row, dict)]
    review_rows = [_compact_transcript_arbitration_row(row) for row in (payload.get("review_rows") or []) if isinstance(row, dict)]
    if not review_segment_refs:
        review_segment_refs = review_rows
    segments = payload.get("segments") if isinstance(payload.get("segments"), list) else []
    changed_rows = [_compact_transcript_arbitration_row(row) for row in segments if isinstance(row, dict) and row.get("changed")]
    status = "needs_review" if review_rows else str(quality_summary.get("status") or ("changed" if changed_rows else "clean"))
    return {
        **base,
        "exists": True,
        "summary": summary,
        "quality_summary": quality_summary,
        "summary_input_policy": summary_input_policy,
        "review_segment_refs": review_segment_refs[:80],
        "next_commands": _transcript_arbitration_commands(root),
        "review_rows": review_rows[:80],
        "changed_rows": changed_rows[:80],
        "status": status,
    }



def _term_correction_impact_summary(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    rel = str(manifest.get("term_correction_impact_report_json") or "term-correction-impact-report.json")
    json_path = _bundle_path(root, rel)
    md_path = _bundle_path(root, str(manifest.get("term_correction_impact_report_markdown") or "term-correction-impact-report.md"))
    base = {
        "exists": json_path.exists(),
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "json_href": _relative_href(root, json_path),
        "markdown_href": _relative_href(root, md_path),
        "status": "missing_report",
        "ok": False,
        "replacement_count": 0,
        "source_alias_total": 0,
        "output_alias_total": 0,
        "final_export_alias_total": 0,
        "reduction_rate": None,
        "final_clean_rate": None,
        "terms": [],
        "next_actions": [],
    }
    if not json_path.exists():
        return base
    payload = _read_optional_object(json_path)
    terms = []
    for row in payload.get("terms") if isinstance(payload.get("terms"), list) else []:
        if not isinstance(row, dict):
            continue
        terms.append(
            {
                "alias": _text(row.get("alias")),
                "canonical": _text(row.get("canonical")),
                "source_alias_count": _safe_int(row.get("source_alias_count")),
                "output_alias_count": _safe_int(row.get("output_alias_count")),
                "resolved_in_outputs": bool(row.get("resolved_in_outputs")),
                "had_source_alias": bool(row.get("had_source_alias")),
            }
        )
    return {
        **base,
        "exists": True,
        "status": _text(payload.get("status") or "unknown"),
        "ok": bool(payload.get("ok")),
        "replacement_count": _safe_int(payload.get("replacement_count")),
        "source_alias_total": _safe_int(payload.get("source_alias_total")),
        "output_alias_total": _safe_int(payload.get("output_alias_total")),
        "final_export_alias_total": _safe_int(payload.get("final_export_alias_total")),
        "reduction_rate": payload.get("reduction_rate"),
        "final_clean_rate": payload.get("final_clean_rate"),
        "terms": terms[:30],
        "next_actions": [str(value) for value in (payload.get("next_actions") or []) if str(value)][:10],
    }
def _transcript_arbitration_commands(root: Path) -> list[str]:
    quoted = _ps_quote(str(root))
    return [
        f".\\scripts\\video-knowledge.ps1 prepare-subtitle-editor {quoted}",
        f".\\scripts\\video-knowledge.ps1 prepare-review-session {quoted} --limit 0 --group-by reason",
        f".\\scripts\\video-knowledge.ps1 transcript-source-arbitration {quoted}",
        f".\\scripts\\video-knowledge.ps1 term-correction-impact-report {quoted}",
    ]


def _transcript_arbitration_by_index(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = []
    for key in ("review_rows", "changed_rows"):
        value = payload.get(key) if isinstance(payload.get(key), list) else []
        rows.extend(row for row in value if isinstance(row, dict))
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            index = int(row.get("index"))
        except Exception:
            continue
        result[index] = row
    return result


def _compact_transcript_arbitration_row(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    alternatives = []
    for alt in (row.get("alternatives") or [])[:4]:
        if not isinstance(alt, dict):
            continue
        alternatives.append(
            {
                "source_id": _text(alt.get("source_id")),
                "source_type": _text(alt.get("source_type")),
                "text": _text(alt.get("text"))[:240],
                "score": alt.get("score"),
            }
        )
    return {
        "index": _safe_int(row.get("index")),
        "start": _seconds(row.get("start")),
        "end": _seconds(row.get("end")),
        "start_time": format_timestamp(_seconds(row.get("start"))),
        "end_time": format_timestamp(_seconds(row.get("end"))),
        "time_range": _text(row.get("time_range")) or (format_timestamp(_seconds(row.get("start"))) + " - " + format_timestamp(_seconds(row.get("end")))),
        "original_text": _text(row.get("original_text"))[:500],
        "corrected_text": _text(row.get("corrected_text") or row.get("text"))[:500],
        "chosen_source": _text(row.get("chosen_source")),
        "chosen_source_type": _text(row.get("chosen_source_type")),
        "confidence": row.get("confidence"),
        "review_reason": _text(row.get("review_reason") or row.get("reason")),
        "alternatives": alternatives,
    }


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0
def _compact_alignment(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "issues": [str(value) for value in row.get("issues") or []],
        "review_start": row.get("review_start"),
        "asr_first_start": row.get("asr_first_start"),
        "frame_time": row.get("frame_time"),
        "suggested_review_start": row.get("suggested_review_start"),
        "asr_excerpt": _text(row.get("asr_excerpt")),
    }


def _compact_tile_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for row in rows[:8]:
        targets.append(
            {
                "tile_id": _text(row.get("tile_id")),
                "confidence": row.get("confidence"),
                "reasons": [str(value) for value in row.get("reasons") or []],
                "evidence_path": _text(row.get("evidence_path") or row.get("tile_path")),
            }
        )
    return targets

def _artifact_cards(
    root: Path,
    manifest: dict[str, Any],
    *,
    ui_surface_registry: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    keys = [
        ("ui_surface_registry_markdown", "界面目录与运行边界", "ui-surface-registry.md"),
        ("ui_surface_registry_json", "界面目录 JSON", "ui-surface-registry.json"),
        ("review_closure_status", "复核关闭进度", "review-closure-status.md"),
        ("review_pack", "复核包", "review-pack.md"),
        ("timeline_alignment_audit_report", "时间轴对齐审计", "timeline-alignment-audit.md"),
        ("transcript_source_arbitration_markdown", "字幕/ASR 多源仲裁", "transcript-source-arbitration.md"),
        ("term_arbitration_codex_markdown", "Codex 术语仲裁", "term-arbitration-codex.md"),
        ("term_arbitration_codex_prompt_markdown", "Codex 术语仲裁 Prompt", "term-arbitration-codex-prompt.md"),
        ("term_arbitration_codex_result_codex_markdown", "Codex 术语回复草稿", "term-arbitration-codex-result.codex.md"),
        ("term_arbitration_codex_validation_markdown", "Codex 术语回复预检", "term-arbitration-codex-validation.md"),
        ("term_arbitration_codex_validation_json", "Codex 术语回复预检 JSON", "term-arbitration-codex-validation.json"),
        ("term_arbitration_glossary_json", "术语仲裁 Glossary", "term-arbitration-glossary.json"),
        ("term_correction_impact_report_markdown", "术语纠错影响", "term-correction-impact-report.md"),
        ("term_correction_closure_markdown", "术语纠错闭环", "term-correction-closure.md"),
        ("term_correction_closure_json", "术语纠错闭环 JSON", "term-correction-closure.json"),
        ("transcript_semantic_correction_prompt_markdown", "通用语义纠错 Prompt", "transcript-semantic-correction-prompt.md"),
        ("transcript_semantic_correction_llm_prompt_markdown", "通用语义纠错 LLM Prompt", "transcript-semantic-correction-llm-prompt.md"),
        ("transcript_semantic_correction_pack_json", "通用语义纠错证据包", "transcript-semantic-correction-pack.json"),
        ("transcript_semantic_correction_result_codex_markdown", "通用语义纠错 Codex 回复", "transcript-semantic-correction-result.codex.md"),
        ("transcript_semantic_correction_result_llm_markdown", "通用语义纠错 LLM 回复", "transcript-semantic-correction-result.llm.md"),
        ("transcript_semantic_correction_validation_markdown", "通用语义纠错预检", "transcript-semantic-correction-validation.md"),
        ("transcript_semantic_correction_closure_markdown", "通用语义纠错闭环", "transcript-semantic-correction-closure.md"),
        ("transcript_semantic_correction_impact_report_markdown", "通用语义纠错影响", "transcript-semantic-correction-impact-report.md"),
        ("transcript_semantic_correction_readable_impact_markdown", "通用语义纠错可读文件影响", "transcript-semantic-readable-impact-report.md"),
        ("transcript_semantic_correction_status_markdown", "通用语义纠错状态", "transcript-semantic-correction-status.md"),
        ("knowledge_note_smart_summary_markdown", "智能总结", "exports/smart-summary.md"),
        ("knowledge_note_full_body_markdown", "连续正文", "exports/full-body.md"),
        ("smart_summary_input_pack_markdown", "智能总结输入证据包", "exports/smart-summary-input-pack.md"),
        ("smart_summary_chapters_markdown", "智能总结章节证据包", "exports/smart-summary-chapters.md"),
        ("smart_summary_course_map_markdown", "课程地图", "exports/course-map.md"),
        ("knowledge_note_transcript_markdown", "完整逐字稿", "exports/full-transcript.md"),
        ("knowledge_note_markdown", "知识笔记", "exports/knowledge-note.md"),
        ("run_artifact_registry_report", "任务产物索引", "run-artifact-registry.md"),
        ("video_moment_index_markdown", "片段索引", "exports/video-moment-index.md"),
        ("video_rag_pack_markdown", "视频 RAG 包", "exports/video-rag-pack.md"),
        ("video_rag_search_markdown", "视频 RAG 查询", "exports/video-rag-search.md"),
        ("video_rag_service_plan_markdown", "视频 RAG 服务计划", "exports/video-rag-service-plan.md"),
        ("external_capability_pack_markdown", "外部能力复用包", "exports/external-capability-pack.md"),
        ("content_candidate_pack_markdown", "内容素材候选包", "exports/content-candidate-pack.md"),
        ("video_edit_review_pack_markdown", "剪辑交接候选与审计", "exports/video-edit-review-pack.md"),
        ("script_clip_candidate_pack_markdown", "脚本驱动的采访片段候选", "exports/script-clip-candidate-pack.md"),
        ("script_clip_alignment_check_markdown", "脚本与剪后片段一致性", "exports/script-clip-alignment-check.md"),
        ("content_clip_candidate_pack_markdown", "通用内容片段候选", "exports/content-clip-candidate-pack.md"),
        ("content_clip_alignment_check_markdown", "通用片段剪后多模态验真", "exports/content-clip-alignment-check.md"),
        ("long_video_fast_segment_plan_markdown", "超长视频快速分段审核计划", "exports/long-video-fast-segment-plan.md"),
        ("long_video_fast_segment_approved_json", "超长视频人工确认剪辑计划", "exports/long-video-fast-segment-approved.json"),
        ("long_video_fast_segment_render_receipt", "超长视频新副本渲染回执", "exports/long-video-fast-segment-render-receipt.json"),
        ("generation_contract_import_markdown", "Generation preflight and representative-frame evidence", "exports/generation-contract-import.md"),
        ("generation_contract_import_json", "Generation preflight evidence JSON", "exports/generation-contract-import.json"),
        ("previs_candidate_evidence_markdown", "3D previs candidate evidence", "exports/previs-candidate-evidence.md"),
        ("previs_candidate_evidence_json", "3D previs candidate evidence JSON", "exports/previs-candidate-evidence.json"),
        ("video_edit_artifact_validation", "Video edit artifact consistency gate", "exports/video-edit-artifact-validation.json"),
        ("video_edit_storyboard_candidates", "Storyboard 候选", "exports/storyboard.candidates.json"),
        ("shot_breakdown_markdown", "逐镜头拉片", "exports/shot-breakdown.md"),
        ("creator_report_index", "创作者候选报告与下载", "exports/creator-report/index.html"),
        ("creator_report_receipt", "创作者报告导出回执", "exports/creator-report/latest.json"),
        ("practical_report_index", "实用参考报告与下载", "exports/creator-report-practical/index.html"),
        ("practical_report_receipt", "实用参考报告导出回执", "exports/creator-report-practical/latest.json"),
        ("shot_breakdown_logseq_markdown", "逐镜头拉片 · Logseq", "exports/shot-breakdown.logseq.md"),
        ("shot_breakdown_csv", "逐镜头拉片 · CSV", "exports/shot-breakdown.csv"),
        ("style_fingerprint_json", "参考视频风格指纹", "exports/style-fingerprint.json"),
        ("imitation_script_markdown", "仿拍脚本候选", "exports/imitation-script.md"),
        ("shot_imitation_readiness_json", "仿拍准备度", "exports/shot-imitation-readiness.json"),
        ("video_decomposition_report_markdown", "证据化视频拆解", "exports/video-decomposition-report.md"),
        ("video_decomposition_report_json", "证据化视频拆解 JSON", "exports/video-decomposition-report.json"),
        ("video_decomposition_report_status_markdown", "视频拆解新鲜度", "exports/video-decomposition-report-status.md"),
        ("human_sample_eval_report", "人工抽样质量评估", "human-sample-eval.md"),
        ("vision_provider_smoke", "视觉 Provider Smoke", "vision-provider-smoke.md"),
        ("vision_provider_matrix", "视觉 Provider Matrix", "vision-provider-matrix.md"),
        ("local_vlm_serving_smoke", "本地 VLM Smoke", "local-vlm-serving-smoke.md"),
        ("codex_vision_session_task_pack_json", "Codex 视觉 Task pack", "codex-vision-sessions/not-prepared/task-pack.json"),
        ("codex_vision_session_prompt_markdown", "Codex 视觉 Prompt", "codex-vision-sessions/not-prepared/prompt.md"),
        ("codex_vision_session_result_template_json", "Codex 视觉结果模板", "codex-vision-sessions/not-prepared/result.template.json"),
        ("codex_vision_session_raw_result_json", "Codex 视觉 Raw result", "codex-vision-sessions/not-prepared/raw-result.json"),
        ("codex_vision_session_import_validation_json", "Codex 视觉验证报告", "codex-vision-sessions/not-prepared/import-validation.json"),
        ("vision_analysis_runs", "视觉分析审计", "vision-analysis-runs.md"),
        ("vision_restore_plan_markdown", "视觉恢复计划", "vision-restore-plan.md"),
        ("temporal_tag_delta_report", "连续多帧打标报告", "temporal-tag-delta.md"),
        ("temporal_tag_delta_json", "连续多帧打标 JSON", "temporal-tag-delta.json"),
        ("temporal_tag_delta_input_template", "连续打标结果模板", "temporal-tag-delta-input-template.json"),
        ("temporal_tag_delta_task_pack", "连续打标 Task pack", "runs/not-started/task-pack.json"),
        ("temporal_tag_delta_raw_result", "连续打标 Raw result", "runs/not-started/raw-result.json"),
        ("temporal_tag_delta_validation", "连续打标验证报告", "runs/not-started/validation.json"),
        ("temporal_tag_delta_run_audit", "连续打标运行审计", "runs/not-started/run.md"),
        ("temporal_tag_delta_restore", "连续打标 Restore", "runs/not-started/restore-plan.json"),
    ]
    cards = surface_artifact_cards(ui_surface_registry) if isinstance(ui_surface_registry, dict) else []
    for key, label, fallback in keys:
        value = str(manifest.get(key) or fallback)
        path = _bundle_path(root, value)
        cards.append({"key": key, "label": label, "href": _relative_href(root, path), "path": str(path), "exists": path.exists()})
    decomposition_status = video_decomposition_report_status(root, write=False)
    creator_status = creator_report_status(root)
    practical_status = creator_report_status(root, view="practical")
    for card in cards:
        if str(card.get("key") or "").startswith("video_decomposition_report"):
            card["status"] = decomposition_status["status"]
        if str(card.get("key") or "").startswith("creator_report_"):
            card["status"] = creator_status["status"]
            card["candidate_only"] = True
            card["export_status"] = creator_status.get("export_status", "missing")
        if str(card.get("key") or "").startswith("practical_report_"):
            card["status"] = practical_status["status"]
            card["candidate_only"] = True
            card["export_status"] = practical_status.get("export_status", "missing")
    return cards



EXTERNAL_REUSE_CAPABILITIES = [
    {
        "key": "time_localization",
        "label": "时间定位 / VTimeLLM",
        "source_projects": ["VTimeLLM", "VideoRAG"],
        "run_types": ["video_moment_index", "timeline_alignment_audit"],
        "artifact_keys": ["video_moment_index_markdown", "timeline_alignment_audit_report"],
        "queue_key": "timeline_rag",
        "description": "片段索引、时间定位和时间轴错位审计。",
    },
    {
        "key": "long_video_memory",
        "label": "长视频 memory / MovieChat",
        "source_projects": ["MovieChat"],
        "run_types": ["long_video_memory_pack"],
        "artifact_keys": ["long_video_memory_pack_markdown"],
        "queue_key": "timeline_rag",
        "description": "短记忆、长记忆和长视频总结输入压缩。",
    },
    {
        "key": "video_rag",
        "label": "VideoRAG 本地检索",
        "source_projects": ["VideoRAG"],
        "run_types": ["video_rag_pack", "video_rag_search", "video_rag_service"],
        "artifact_keys": ["video_rag_pack_markdown", "video_rag_search_markdown", "video_rag_service_plan_markdown"],
        "queue_key": "timeline_rag",
        "description": "JSONL/SQLite 检索包、查询报告和显式启动的本地 HTTP 服务计划。",
    },
    {
        "key": "local_vlm_adapter",
        "label": "本地 VLM adapter",
        "source_projects": ["Qwen-VL", "InternVL", "LLaVA-OneVision"],
        "run_types": ["local_vlm_serving_smoke", "local_vlm_adapter_plan", "vision_provider_smoke"],
        "artifact_keys": ["local_vlm_serving_smoke", "vision_provider_smoke", "vision_provider_matrix"],
        "queue_key": "vision",
        "description": "本地/云多模态 provider smoke 和本地 VLM 服务契约。",
    },
    {
        "key": "content_capability",
        "label": "内容素材能力包",
        "source_projects": ["vsummary", "BiliNote", "VKP content assets"],
        "run_types": ["external_capability_pack", "knowledge_note_export"],
        "artifact_keys": ["external_capability_pack_markdown", "content_candidate_pack_markdown", "knowledge_note_smart_summary_markdown"],
        "queue_key": "summary_export",
        "description": "外部能力复用包、内容素材候选和智能总结交接。",
    },
    {
        "key": "long_video_fast_segment",
        "label": "超长视频内容感知快速分段",
        "source_projects": ["videocut-kit", "FunASR VAD", "faster-whisper Silero VAD", "AutoShot", "VKP content clips"],
        "run_types": ["long_video_fast_segment_plan", "long_video_fast_segment_review", "ffmpeg_execution_receipt"],
        "artifact_keys": ["long_video_fast_segment_plan_markdown", "long_video_fast_segment_approved_json", "long_video_fast_segment_render_receipt"],
        "queue_key": "summary_export",
        "description": "按 ASR/VAD/OCR/镜头证据生成可审核的保留/删除计划；人工确认后仅输出新副本。",
    },
    {
        "key": "video_edit_handoff",
        "label": "剪辑交接 / videocut-kit",
        "source_projects": ["videocut-kit", "VKP Timeline", "VKP Smart Summary"],
        "run_types": ["video_edit_review_pack"],
        "artifact_keys": ["video_edit_review_pack_markdown", "video_edit_artifact_validation", "video_edit_storyboard_candidates"],
        "queue_key": "summary_export",
        "description": "本地边界吸附、静音恢复、产物一致性硬门、Storyboard 候选与人工确认偏好证据。",
    },
    {
        "key": "script_clip_alignment",
        "label": "脚本驱动原声候选与剪后一致性",
        "source_projects": ["VKP VideoRAG", "VKP transcript reference window", "VKP transcript editor"],
        "run_types": ["script_clip_candidate_pack", "script_clip_alignment_check"],
        "artifact_keys": ["script_clip_candidate_pack_markdown", "script_clip_alignment_check_markdown"],
        "queue_key": "summary_export",
        "description": "按脚本槽位检索采访原声，人工选择后核对说话人、剪切范围、clip-only ASR 与字幕语义。",
    },
    {
        "key": "content_clip_alignment",
        "label": "通用内容片段检索与剪后验真",
        "source_projects": ["VKP VideoRAG", "VKP technical shots", "Subtitle Edit", "moys-asr-workflow", "Shot2Story"],
        "run_types": ["content_clip_candidate_pack", "content_clip_alignment_check"],
        "artifact_keys": ["content_clip_candidate_pack_markdown", "content_clip_alignment_check_markdown"],
        "queue_key": "summary_export",
        "description": "按内容类型选择 ASR/OCR/视觉/音频/技术镜头证据，人工选段后执行 clip-only 多模态验真。",
    },
    {
        "key": "shot_breakdown_and_imitation",
        "label": "逐镜头拉片 / 仿拍脚本",
        "source_projects": ["AutoShot", "OmniShotCut", "Auto Scenes", "ruptures", "WaveSurfer.js", "VKP Timeline"],
        "run_types": ["shot_breakdown"],
        "artifact_keys": ["shot_breakdown_markdown", "shot_breakdown_logseq_markdown", "shot_breakdown_csv", "style_fingerprint_json", "imitation_script_markdown", "shot_imitation_readiness_json"],
        "queue_key": "summary_export",
        "description": "基于现有镜头、ASR/OCR、视觉和标签证据生成逐镜头事实、风格指纹、准备度和人工复核仿拍脚本。",
    },
]

EXTERNAL_REUSE_ACTION_STATUSES = {"needs_retry", "needs_review", "needs_execution", "needs_input", "failed", "error", "blocked"}


def _external_reuse_status_summary(root: Path, manifest: dict[str, Any], registry: dict[str, Any], queue: dict[str, Any]) -> dict[str, Any]:
    runs = registry.get("runs") if isinstance(registry.get("runs"), list) else []
    artifacts = {row["key"]: row for row in _artifact_cards(root, manifest)}
    rows = []
    for spec in EXTERNAL_REUSE_CAPABILITIES:
        wanted = set(spec["run_types"])
        matched = [run for run in runs if str(run.get("run_type") or "") in wanted]
        action_runs = [run for run in matched if str(run.get("status") or "") in EXTERNAL_REUSE_ACTION_STATUSES]
        status = "missing"
        if matched:
            status = "action_required" if action_runs else "ready"
        artifact_rows = [artifacts[key] for key in spec["artifact_keys"] if key in artifacts]
        retry_commands = []
        failed_count = 0
        status_counts: dict[str, int] = {}
        for run in matched:
            run_status = str(run.get("status") or "unknown")
            status_counts[run_status] = status_counts.get(run_status, 0) + 1
            failed_count += int(run.get("failed_count") or 0)
            retry = str(run.get("retry_command") or "").strip()
            if retry and retry not in retry_commands:
                retry_commands.append(retry)
        rows.append(
            {
                "key": spec["key"],
                "label": spec["label"],
                "source_projects": spec["source_projects"],
                "description": spec["description"],
                "status": status,
                "queue_key": spec["queue_key"],
                "run_types": spec["run_types"],
                "run_count": len(matched),
                "action_required": len(action_runs),
                "failed_count": failed_count,
                "status_counts": status_counts,
                "retry_commands": retry_commands,
                "artifacts": artifact_rows,
                "runs": matched[:8],
            }
        )
    return {
        "schema": "video_knowledge_pipeline.external_reuse_workbench_status.v1",
        "status": "action_required" if any(row["action_required"] for row in rows) else "ready" if any(row["run_count"] for row in rows) else "missing",
        "capability_count": len(rows),
        "ready_count": sum(1 for row in rows if row["status"] == "ready"),
        "action_required_count": sum(1 for row in rows if row["action_required"]),
        "missing_count": sum(1 for row in rows if row["status"] == "missing"),
        "capabilities": rows,
        "operator_boundary": {
            "read_only": True,
            "no_command_execution": True,
            "no_cloud_call": True,
            "no_model_server_started": True,
            "purpose": "Show external-project reuse capability runs inside the static workbench.",
        },
    }

def _review_closure_summary(root: Path) -> dict[str, Any]:
    json_path = root / "review-closure-status.json"
    md_path = root / "review-closure-status.md"
    base = {
        "exists": json_path.exists(),
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "json_href": _relative_href(root, json_path),
        "markdown_href": _relative_href(root, md_path),
        "summary": {},
        "quality_summary": {},
        "open_by_reason": {},
        "closed_by_reason": {},
        "transcript_arbitration": {"open": 0, "closed": 0, "status": "missing_report"},
    }
    if not json_path.exists():
        return base
    payload = read_json(json_path)
    if not isinstance(payload, dict):
        return {**base, "exists": False, "error": "review-closure-status.json is not an object"}
    open_by_reason = payload.get("open_by_reason") if isinstance(payload.get("open_by_reason"), dict) else {}
    closed_by_reason = payload.get("closed_by_reason") if isinstance(payload.get("closed_by_reason"), dict) else {}
    keys = ("transcript_source_conflict", "low_arbitration_confidence")
    open_arbitration = max([int(open_by_reason.get(key) or 0) for key in keys] or [0])
    closed_arbitration = max([int(closed_by_reason.get(key) or 0) for key in keys] or [0])
    status = "needs_review" if open_arbitration else "closed" if closed_arbitration else "none"
    return {
        **base,
        "exists": True,
        "summary": payload.get("summary") if isinstance(payload.get("summary"), dict) else {},
        "open_by_reason": open_by_reason,
        "closed_by_reason": closed_by_reason,
        "transcript_arbitration": {"open": open_arbitration, "closed": closed_arbitration, "status": status},
    }


def _evidence_status_summary(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    alignment_path = root / "timeline-alignment-audit.json"
    moment_path = root / "exports" / "video-moment-index.json"
    tile_targets = 0
    tile_issue_items = 0
    for row in rows:
        quality = {str(value) for value in row.get("quality_issues") or []}
        targets = row.get("tile_review_targets") if isinstance(row.get("tile_review_targets"), list) else []
        tile_targets += len([value for value in targets if isinstance(value, dict)])
        if targets or "tile_result_needs_review" in quality or "needs_high_res_tile_recovery" in quality:
            tile_issue_items += 1
    alignment = _read_optional_object(alignment_path)
    alignment_items = alignment.get("items") if isinstance(alignment.get("items"), list) else []
    alignment_issue_count = int(alignment.get("summary", {}).get("issue_count") or alignment.get("issue_count") or 0) if isinstance(alignment, dict) else 0
    if not alignment_issue_count:
        alignment_issue_count = sum(1 for item in alignment_items if isinstance(item, dict) and item.get("issues"))
    moment = _read_optional_object(moment_path)
    chunks = moment.get("chunks") if isinstance(moment.get("chunks"), list) else []
    moment_summary = moment.get("summary") if isinstance(moment.get("summary"), dict) else {}
    codex_session = _read_optional_object(root / "codex-vision-session-status.json")
    codex_session["commands"] = {
        "prepare_semantic": f".\\scripts\\video-knowledge.ps1 prepare-codex-vision-session {_ps_quote(str(root))} --kind semantic",
        "prepare_temporal": f".\\scripts\\video-knowledge.ps1 prepare-codex-vision-session {_ps_quote(str(root))} --kind temporal",
        "import_results": f".\\scripts\\video-knowledge.ps1 import-codex-vision-results {_ps_quote(str(root))} --input-json \"<result.json>\"",
    }
    temporal_tag = _read_optional_object(root / "temporal-tag-delta.json")
    temporal_tag["commands"] = {
        "run": f".\\scripts\\video-knowledge.ps1 run-temporal-tag-delta {_ps_quote(str(root))} --execute-tagger --frame-count 8 --limit 16",
        "resume": (
            f".\\scripts\\video-knowledge.ps1 run-temporal-tag-delta {_ps_quote(str(root))} --execute-tagger --frame-count 8 --limit 16 --resume {temporal_tag.get('run_id')}"
            if temporal_tag.get("run_id") else ""
        ),
    }
    return {
        "visual_evidence_levels": evidence_level_counts(rows),
        "codex_vision_session": codex_session or {"phase": "not_prepared", "status": "not_prepared"},
        "temporal_tag_delta": temporal_tag or {"status": "not_started", "outcome_counts": {}},
        "timeline_alignment": {
            "exists": alignment_path.exists(),
            "issue_count": alignment_issue_count,
            "item_count": len(alignment_items),
            "href": _relative_href(root, alignment_path.with_suffix(".md")),
            "json_href": _relative_href(root, alignment_path),
        },
        "tile_review": {
            "target_count": tile_targets,
            "item_count": tile_issue_items,
            "status": "needs_review" if tile_targets or tile_issue_items else "none",
        },
        "video_moment_index": {
            "exists": moment_path.exists(),
            "chunk_count": int(moment_summary.get("chunks") or len(chunks) or 0),
            "duration_seconds": moment_summary.get("duration_seconds", 0),
            "href": _relative_href(root, moment_path.with_suffix(".md")),
            "json_href": _relative_href(root, moment_path),
        },
    }



def _compact_video_rag_chunks(path: Path, *, limit: int = 240) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        rows = read_jsonl(path)
    except Exception:
        return []
    chunks: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        start = _seconds(metadata.get("start"))
        end = _seconds(metadata.get("end"), default=start)
        timeline_indexes = metadata.get("timeline_indexes") if isinstance(metadata.get("timeline_indexes"), list) else []
        tags = metadata.get("tags") if isinstance(metadata.get("tags"), list) else []
        keywords = metadata.get("keywords") if isinstance(metadata.get("keywords"), list) else []
        evidence_paths = metadata.get("evidence_paths") if isinstance(metadata.get("evidence_paths"), list) else []
        chunks.append(
            {
                "id": _text(row.get("id") or f"rag:{len(chunks) + 1}"),
                "chunk_kind": _text(metadata.get("chunk_kind") or "moment"),
                "text": _clip_text(_text(row.get("text")), 720),
                "start": start,
                "end": end,
                "start_time": _text(metadata.get("start_time") or format_timestamp(start)),
                "end_time": _text(metadata.get("end_time") or format_timestamp(end)),
                "timeline_indexes": [_safe_int(value) for value in timeline_indexes],
                "memory_level": _text(metadata.get("memory_level") or ""),
                "memory_id": _text(metadata.get("memory_id") or ""),
                "parent_memory_id": _text(metadata.get("parent_memory_id") or ""),
                "child_memory_ids": [_text(value) for value in metadata.get("child_memory_ids") or [] if _text(value)] if isinstance(metadata.get("child_memory_ids"), list) else [],
                "child_moment_indexes": [_safe_int(value) for value in metadata.get("child_moment_indexes") or []] if isinstance(metadata.get("child_moment_indexes"), list) else [],
                "fact_status": _text(metadata.get("fact_status") or ""),
                "tags": [_text(value) for value in tags if _text(value)],
                "keywords": [_text(value) for value in keywords if _text(value)],
                "evidence_paths": [_text(value) for value in evidence_paths if _text(value)][:8],
                "has_visual_evidence": bool(metadata.get("has_visual_evidence")),
                "has_temporal_evidence": bool(metadata.get("has_temporal_evidence")),
            }
        )
        if len(chunks) >= limit:
            break
    return chunks



def _video_rag_status(root: Path, manifest: dict[str, Any], video_rag_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    chunks_path = root / "exports" / "video-rag-chunks.jsonl"
    search_path = root / "exports" / "video-rag-search.json"
    search_report = _read_optional_object(search_path)
    sqlite_rel = _text(manifest.get("video_rag_sqlite_index") or "exports/video-rag-index.sqlite")
    sqlite_path = (root / sqlite_rel).resolve() if sqlite_rel and not Path(sqlite_rel).is_absolute() else Path(sqlite_rel).expanduser().resolve()
    operator_boundary = search_report.get("operator_boundary") if isinstance(search_report.get("operator_boundary"), dict) else {}
    return {
        "chunks_jsonl_exists": chunks_path.exists(),
        "chunk_count": len(video_rag_chunks),
        "search_report_exists": search_path.exists(),
        "search_href": _relative_href(root, search_path),
        "search_markdown_href": _relative_href(root, root / "exports" / "video-rag-search.md"),
        "retrieval_backend": _text(search_report.get("retrieval_backend") or manifest.get("video_rag_search_backend") or "keyword"),
        "requested_retrieval_backend": _text(search_report.get("requested_retrieval_backend") or ""),
        "manifest_backend": _text(manifest.get("video_rag_search_backend") or ""),
        "backend_status": _text(search_report.get("backend_status") or ("ok" if chunks_path.exists() else "missing_chunks")),
        "backend_warning": _text(search_report.get("backend_warning") or ""),
        "sqlite_index_exists": sqlite_path.exists(),
        "sqlite_index_href": _relative_href(root, sqlite_path) if sqlite_path.exists() else "",
        "no_vector_backend_started": bool(operator_boundary.get("no_vector_backend_started", True)),
        "local_only": bool(operator_boundary.get("local_only", True)),
    }

def _clip_text(value: str, limit: int = 720) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _content_candidate_pack_summary(root: Path, manifest: dict[str, Any], *, moment_index: dict[str, Any] | None = None, video_rag_chunks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    assets = manifest.get("content_assets") if isinstance(manifest.get("content_assets"), dict) else {}
    export = manifest.get("knowledge_note_export") if isinstance(manifest.get("knowledge_note_export"), dict) else {}
    if not assets and isinstance(export.get("content_assets"), dict):
        assets = export.get("content_assets")
    json_path = _bundle_path(root, str(assets.get("content_candidate_pack_path") or manifest.get("content_candidate_pack_json") or "exports/content-candidate-pack.json"))
    markdown_path = _bundle_path(root, str(assets.get("content_candidate_pack_markdown_path") or manifest.get("content_candidate_pack_markdown") or "exports/content-candidate-pack.md"))
    payload = _read_optional_object(json_path)
    human_eval = _human_sample_eval_summary_for_workbench(root, manifest)
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    compact: list[dict[str, Any]] = []
    for row in candidates:
        if not isinstance(row, dict):
            continue
        candidate = _compact_content_candidate(row)
        candidate["moment_links"] = _content_candidate_moment_links(candidate, moment_index or {}, video_rag_chunks or [])
        candidate["moment_link_count"] = len(candidate["moment_links"])
        candidate["moment_link_summary"] = _content_candidate_moment_link_summary(candidate["moment_links"])
        candidate["review_filters"] = _content_candidate_review_filters(candidate, human_eval)
        candidate["review_filter_label"] = _content_candidate_review_filter_label(candidate["review_filters"])
        compact.append(candidate)
    return {
        "exists": json_path.exists() and markdown_path.exists(),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "json_href": _relative_href(root, json_path),
        "markdown_href": _relative_href(root, markdown_path),
        "candidate_count": int(payload.get("candidate_count") or len(compact)) if payload else 0,
        "review_required": bool(payload.get("review_required")) if payload else True,
        "publication_allowed": bool(payload.get("publication_allowed")) if payload else False,
        "allowed_as_fact": bool(payload.get("allowed_as_fact")) if payload else False,
        "allowed_as_inspiration": bool(payload.get("allowed_as_inspiration")) if payload else False,
        "human_sample_eval": human_eval,
        "filter_counts": _content_candidate_filter_counts(compact),
        "citation_digest_candidate_count": int(payload.get("citation_digest_candidate_count") or sum(1 for candidate in compact if int(candidate.get("citation_count") or 0) > 0)) if payload else 0,
        "linked_moment_candidate_count": sum(1 for candidate in compact if int(candidate.get("moment_link_count") or 0) > 0),
        "candidates": compact[:24],
    }


def _human_sample_eval_summary_for_workbench(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    json_path = _bundle_path(root, str(manifest.get("human_sample_eval_json") or "human-sample-eval.json"))
    markdown_path = _bundle_path(root, str(manifest.get("human_sample_eval_report") or "human-sample-eval.md"))
    payload = _read_optional_object(json_path)
    rates = payload.get("rates") if isinstance(payload.get("rates"), dict) else {}
    exists = bool(payload) and json_path.exists()
    status = _text(payload.get("status") or ("available" if exists else "not_available"))
    labeled_rows = int(payload.get("labeled_rows") or 0) if payload else 0
    sample_count = int(payload.get("sample_count") or 0) if payload else 0
    return {
        "exists": exists,
        "status": status,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "json_href": _relative_href(root, json_path),
        "markdown_href": _relative_href(root, markdown_path),
        "sample_count": sample_count,
        "labeled_rows": labeled_rows,
        "candidate_usable_rate": _rate_number(rates.get("content_candidate_usable_rate")),
        "candidate_evidence_sufficient_rate": _rate_number(rates.get("content_candidate_evidence_sufficient_rate")),
        "multimodal_net_help_rate": _rate_number(rates.get("human_sampled_multimodal_net_help_rate")),
        "review_signal_only": True,
    }


def _rate_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 1:
        number *= 100
    return round(number, 2)


def _rate_text(value: Any) -> str:
    number = _rate_number(value)
    return "未抽样" if number is None else f"{number:.1f}%"


def _content_candidate_review_filters(candidate: dict[str, Any], human_eval: dict[str, Any]) -> list[str]:
    filters = ["all"]
    if not human_eval.get("exists") or int(human_eval.get("labeled_rows") or 0) <= 0:
        filters.append("unsampled")
    evidence_rate = human_eval.get("candidate_evidence_sufficient_rate")
    if evidence_rate is None or float(evidence_rate) < 80:
        filters.append("evidence_low")
    usable_rate = human_eval.get("candidate_usable_rate")
    if usable_rate is not None and float(usable_rate) >= 60:
        filters.append("usable")
    if int(candidate.get("citation_count") or 0) > 0:
        filters.append("citation_ready")
    else:
        filters.append("citation_missing")
    if int(candidate.get("summary_chapter_ref_count") or len(candidate.get("summary_chapter_refs") or [])) > 0:
        filters.append("chapter_linked")
    else:
        filters.append("chapter_missing")
    if int(candidate.get("moment_link_count") or 0) > 0:
        filters.append("moment_linked")
    else:
        filters.append("moment_missing")
    if not candidate.get("evidence_paths") and int(candidate.get("citation_count") or 0) <= 0:
        filters.append("evidence_low")
    return sorted(set(filters))


def _content_candidate_review_filter_label(filters: list[str]) -> str:
    labels = {
        "chapter_linked": "已关联章节",
        "chapter_missing": "缺章节",
        "moment_linked": "已关联片段",
        "moment_missing": "缺片段",
        "unsampled": "未抽样",
        "evidence_low": "证据不足",
        "usable": "可继续加工",
        "citation_ready": "有Citation",
        "citation_missing": "缺Citation",
    }
    selected = [labels[key] for key in filters if key in labels]
    return " / ".join(selected) if selected else "已抽样"


def _content_candidate_filter_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"all": len(candidates), "unsampled": 0, "evidence_low": 0, "usable": 0, "citation_ready": 0, "citation_missing": 0, "chapter_linked": 0, "chapter_missing": 0, "moment_linked": 0, "moment_missing": 0}
    for candidate in candidates:
        filters = set(candidate.get("review_filters") or [])
        for key in ("unsampled", "evidence_low", "usable", "citation_ready", "citation_missing", "chapter_linked", "chapter_missing", "moment_linked", "moment_missing"):
            if key in filters:
                counts[key] += 1
    return counts


def _compact_content_candidate(row: dict[str, Any]) -> dict[str, Any]:
    citations = _compact_content_candidate_citations(row.get("evidence_citations") if isinstance(row.get("evidence_citations"), list) else [])
    return {
        "id": _text(row.get("id")),
        "timeline_index": row.get("timeline_index"),
        "time_range": _text(row.get("time_range")),
        "candidate_types": [str(value) for value in row.get("candidate_types") or [] if str(value)],
        "viewpoint": _clip_text(_text(row.get("viewpoint")), 180),
        "case_or_example": _clip_text(_text(row.get("case_or_example")), 140),
        "evidence_paths": [str(value) for value in row.get("evidence_paths") or [] if str(value)][:4],
        "citation_digest_status": _text(row.get("citation_digest_status") or ("ready" if citations else "not_available")),
        "citation_count": len(citations),
        "citation_summary": _content_candidate_citation_summary(citations),
        "evidence_citations": citations,
        "summary_chapter_refs": _compact_summary_chapter_refs(row.get("summary_chapter_refs") if isinstance(row.get("summary_chapter_refs"), list) else []),
        "summary_chapter_ref_count": int(row.get("summary_chapter_ref_count") or 0),
    }


def _content_candidate_timeline_indexes(candidate: dict[str, Any]) -> list[int]:
    indexes: list[int] = []
    raw_index = candidate.get("timeline_index")
    if raw_index is not None:
        try:
            indexes.append(int(raw_index))
        except (TypeError, ValueError):
            pass
    for citation in candidate.get("evidence_citations") or []:
        if not isinstance(citation, dict):
            continue
        for value in citation.get("timeline_indexes") or []:
            try:
                indexes.append(int(value))
            except (TypeError, ValueError):
                continue
    seen: set[int] = set()
    result: list[int] = []
    for value in indexes:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _int_set(values: Any) -> set[int]:
    result: set[int] = set()
    if not isinstance(values, list):
        return result
    for value in values:
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            continue
    return result


def _content_candidate_moment_links(candidate: dict[str, Any], moment_index: dict[str, Any], video_rag_chunks: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    indexes = set(_content_candidate_timeline_indexes(candidate))
    if not indexes:
        return []
    links: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in moment_index.get("chunks") if isinstance(moment_index.get("chunks"), list) else []:
        if not isinstance(chunk, dict):
            continue
        chunk_indexes = _int_set(chunk.get("timeline_indexes"))
        if not indexes.intersection(chunk_indexes):
            continue
        chunk_index = chunk.get("chunk_index")
        link_id = f"moment:{chunk_index if chunk_index is not None else len(links) + 1}"
        if link_id in seen:
            continue
        seen.add(link_id)
        start = _seconds(chunk.get("start"))
        end = _seconds(chunk.get("end"), default=start)
        links.append({
            "link_type": "moment",
            "id": link_id,
            "label": f"moment #{chunk_index if chunk_index is not None else len(links) + 1}",
            "start": start,
            "end": end,
            "time": f"{_text(chunk.get('start_time') or format_timestamp(start))} - {_text(chunk.get('end_time') or format_timestamp(end))}",
            "timeline_indexes": sorted(chunk_indexes),
            "text": _clip_text(_text(chunk.get("snippet") or chunk.get("transcript_text") or chunk.get("visual_text") or chunk.get("search_text")), 100),
            "evidence_paths": [str(value) for value in chunk.get("evidence_paths") or [] if str(value)][:3],
        })
        if len(links) >= limit:
            return links
    for chunk in video_rag_chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_indexes = _int_set(chunk.get("timeline_indexes"))
        if not indexes.intersection(chunk_indexes):
            continue
        link_id = _text(chunk.get("id") or f"rag:{len(links) + 1}")
        if not link_id or link_id in seen:
            continue
        seen.add(link_id)
        kind = _text(chunk.get("chunk_kind") or "rag")
        start = _seconds(chunk.get("start"))
        end = _seconds(chunk.get("end"), default=start)
        links.append({
            "link_type": "rag",
            "id": link_id,
            "label": kind,
            "start": start,
            "end": end,
            "time": f"{_text(chunk.get('start_time') or format_timestamp(start))} - {_text(chunk.get('end_time') or format_timestamp(end))}",
            "timeline_indexes": sorted(chunk_indexes),
            "text": _clip_text(_text(chunk.get("text")), 100),
            "evidence_paths": [str(value) for value in chunk.get("evidence_paths") or [] if str(value)][:3],
        })
        if len(links) >= limit:
            break
    return links


def _content_candidate_moment_link_summary(links: list[dict[str, Any]]) -> str:
    if not links:
        return "未关联片段"
    first = links[0]
    label = _text(first.get("label") or first.get("id") or "moment")
    time = _text(first.get("time") or "")
    suffix = f" +{len(links) - 1}" if len(links) > 1 else ""
    return f"{label} {time}{suffix}".strip()


def _content_candidate_moment_links_html(links: list[dict[str, Any]]) -> str:
    if not links:
        return "-"
    parts: list[str] = []
    for idx, link in enumerate(links[:4], start=1):
        link_id = _text(link.get("id") or "")
        label = _text(link.get("label") or link_id or f"片段{idx}")
        start = _seconds(link.get("start"))
        js_id = html.escape(json.dumps(link_id), quote=True)
        parts.append(f"<button type=\"button\" class=\"mini-link\" onclick=\"selectSearchChunk({start:.3f}, {js_id})\">{html.escape(label)}</button>")
    return " ".join(parts)


def _compact_summary_chapter_refs(rows: list[Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows[:6]:
        if not isinstance(row, dict):
            continue
        try:
            chapter_index = int(row.get("chapter_index") or 0)
        except (TypeError, ValueError):
            chapter_index = 0
        if chapter_index <= 0 or chapter_index in seen:
            continue
        seen.add(chapter_index)
        compact.append(
            {
                "chapter_index": chapter_index,
                "chapter_title": _clip_text(_text(row.get("chapter_title")), 80),
                "chapter_time_range": _text(row.get("chapter_time_range")),
            }
        )
    return compact

def _compact_content_candidate_citations(rows: list[Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rows[:8]:
        if not isinstance(row, dict):
            continue
        compact.append(
            {
                "source_type": _text(row.get("source_type") or "unknown"),
                "time": _text(row.get("time") or ""),
                "timeline_indexes": [int(value) for value in row.get("timeline_indexes") or [] if str(value).isdigit()],
                "text": _clip_text(_text(row.get("text")), 140),
                "evidence_paths": [str(value) for value in row.get("evidence_paths") or [] if str(value)][:3],
            }
        )
    return compact


def _content_candidate_citation_summary(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return "未生成 Citation Digest"
    first = citations[0]
    text = _clip_text(_text(first.get("text")), 82)
    source_type = _text(first.get("source_type") or "unknown")
    time = _text(first.get("time") or "") or "unknown_time"
    suffix = f" +{len(citations) - 1}" if len(citations) > 1 else ""
    return f"{time} {source_type}: {text}{suffix}"


def _content_candidate_panel_html(summary: dict[str, Any]) -> str:
    if not summary.get("exists"):
        return '<div class="muted">尚未生成内容素材候选包。运行 export-knowledge-note 后会生成 exports/content-candidate-pack.json/md。</div>'
    candidates = summary.get("candidates") if isinstance(summary.get("candidates"), list) else []
    human_eval = summary.get("human_sample_eval") if isinstance(summary.get("human_sample_eval"), dict) else {}
    counts = summary.get("filter_counts") if isinstance(summary.get("filter_counts"), dict) else {}
    rows = []
    for candidate in candidates[:24]:
        types = ", ".join(str(value) for value in candidate.get("candidate_types") or [])
        evidence = "; ".join(str(value) for value in candidate.get("evidence_paths") or [])
        citation = str(candidate.get("citation_summary") or "")
        chapters = ", ".join(str(ref.get("chapter_index")) for ref in candidate.get("summary_chapter_refs") or []) or "-"
        moment_links = _content_candidate_moment_links_html(candidate.get("moment_links") if isinstance(candidate.get("moment_links"), list) else [])
        filters = " ".join(str(value) for value in candidate.get("review_filters") or ["all"])
        rows.append(
            '<tr class="content-candidate-row" data-review-filters="' + html.escape(filters) + '">'
            + "<td><code>" + html.escape(str(candidate.get("id") or "")) + "</code></td>"
            + "<td><code>" + html.escape(str(candidate.get("time_range") or "")) + "</code></td>"
            + "<td>" + html.escape(types) + "</td>"
            + "<td>" + html.escape(str(candidate.get("review_filter_label") or "")) + "</td>"
            + "<td>" + html.escape(str(candidate.get("viewpoint") or "")) + "</td>"
            + "<td>" + html.escape(chapters) + "</td>"
            + "<td>" + html.escape(citation) + "</td>"
            + "<td>" + moment_links + "</td>"
            + "<td>" + html.escape(evidence) + "</td>"
            + "</tr>"
        )
    table = '<div class="muted">暂无候选条目。</div>'
    if rows:
        table = '<table class="mini-table"><thead><tr><th>ID</th><th>时间</th><th>类型</th><th>抽样状态</th><th>观点</th><th>章节</th><th>Citation</th><th>关联片段</th><th>证据</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
    filter_buttons = (
        '<div class="toolbar" style="margin:8px 0">'
        '<button onclick="filterContentCandidates(\'all\')">全部 ' + html.escape(str(counts.get("all", 0))) + '</button>'
        '<button onclick="filterContentCandidates(\'unsampled\')">只看未抽样 ' + html.escape(str(counts.get("unsampled", 0))) + '</button>'
        '<button onclick="filterContentCandidates(\'evidence_low\')">抽样证据不足 ' + html.escape(str(counts.get("evidence_low", 0))) + '</button>'
        '<button onclick="filterContentCandidates(\'usable\')">可继续加工 ' + html.escape(str(counts.get("usable", 0))) + '</button>'
        '<button onclick="filterContentCandidates(\'citation_ready\')">有Citation ' + html.escape(str(counts.get("citation_ready", 0))) + '</button>'
        '<button onclick="filterContentCandidates(\'citation_missing\')">缺Citation ' + html.escape(str(counts.get("citation_missing", 0))) + '</button>'
        '<button onclick="filterContentCandidates(\'chapter_linked\')">已关联章节 ' + html.escape(str(counts.get("chapter_linked", 0))) + '</button>'
        '<button onclick="filterContentCandidates(\'chapter_missing\')">缺章节 ' + html.escape(str(counts.get("chapter_missing", 0))) + '</button>'
        '<button onclick="filterContentCandidates(\'moment_linked\')">已关联片段 ' + html.escape(str(counts.get("moment_linked", 0))) + '</button>'
        '<button onclick="filterContentCandidates(\'moment_missing\')">缺片段 ' + html.escape(str(counts.get("moment_missing", 0))) + '</button>'
        '</div><div id="contentCandidateFilterStatus" class="muted">当前显示全部候选。</div>'
    )
    sample_eval_button = '<button onclick="openArtifact(\'human_sample_eval_report\')">打开抽样评估</button>' if human_eval.get("exists") else '<button disabled>抽样评估未生成</button>'
    return (
        '<div class="closure-grid">'
        '<div class="closure-metric"><span class="muted">候选数</span><strong>' + html.escape(str(summary.get("candidate_count") or 0)) + '</strong></div>'
        '<div class="closure-metric"><span class="muted">抽样状态</span><strong>' + html.escape(str(human_eval.get("status") or "not_available")) + '</strong></div>'
        '<div class="closure-metric"><span class="muted">Citation候选</span><strong>' + html.escape(str(summary.get("citation_digest_candidate_count") or 0)) + '</strong></div>'
        '<div class="closure-metric"><span class="muted">片段互链</span><strong>' + html.escape(str(summary.get("linked_moment_candidate_count") or 0)) + '</strong></div>'
        '<div class="closure-metric"><span class="muted">候选可用率</span><strong>' + html.escape(_rate_text(human_eval.get("candidate_usable_rate"))) + '</strong></div>'
        '<div class="closure-metric"><span class="muted">证据充分率</span><strong>' + html.escape(_rate_text(human_eval.get("candidate_evidence_sufficient_rate"))) + '</strong></div>'
        '</div>'
        '<div class="muted">review_required=' + html.escape(str(bool(summary.get("review_required"))).lower())
        + ' · publication_allowed=' + html.escape(str(bool(summary.get("publication_allowed"))).lower())
        + ' · allowed_as_fact=' + html.escape(str(bool(summary.get("allowed_as_fact"))).lower())
        + ' · allowed_as_inspiration=' + html.escape(str(bool(summary.get("allowed_as_inspiration"))).lower())
        + ' · review_signal_only=' + html.escape(str(bool(human_eval.get("review_signal_only", True))).lower()) + '</div>'
        '<div style="margin:8px 0"><button onclick="openArtifact(\'content_candidate_pack_markdown\')">打开候选包</button> ' + sample_eval_button + '</div>'
        + filter_buttons
        + table
    )


def _provider_status_summary(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    smoke = _compact_vision_provider_smoke(
        root,
        manifest,
        json_key="vision_provider_smoke_json",
        markdown_key="vision_provider_smoke",
        json_fallback="vision-provider-smoke.json",
        markdown_fallback="vision-provider-smoke.md",
    )
    matrix = _compact_vision_provider_matrix(
        root,
        manifest,
        json_key="vision_provider_matrix_json",
        markdown_key="vision_provider_matrix",
        json_fallback="vision-provider-matrix.json",
        markdown_fallback="vision-provider-matrix.md",
    )
    local_vlm = _compact_local_vlm_smoke(
        root,
        manifest,
        json_key="local_vlm_serving_smoke_json",
        markdown_key="local_vlm_serving_smoke",
        json_fallback="local-vlm-serving-smoke.json",
        markdown_fallback="local-vlm-serving-smoke.md",
    )
    any_report = bool(smoke.get("exists") or matrix.get("exists") or local_vlm.get("exists"))
    ready_provider = bool(smoke.get("safe_to_execute") or matrix.get("recommended_provider"))
    local_ready = str(local_vlm.get("status") or "") == "executed_ready"
    status = "ready" if ready_provider or local_ready else "reports_available" if any_report else "missing_reports"
    return {
        "status": status,
        "has_any_report": any_report,
        "vision_provider_smoke": smoke,
        "vision_provider_matrix": matrix,
        "local_vlm_serving_smoke": local_vlm,
        "operator_boundary": {
            "read_only": True,
            "does_not_start_model_server": True,
            "does_not_call_cloud_provider": True,
            "does_not_modify_timeline": True,
            "purpose": "Show existing provider/local VLM smoke artifacts inside the static workbench.",
        },
        "next_commands": [
            f".\\scripts\\video-knowledge.ps1 vision-provider-smoke --bundle-dir {root}",
            f".\\scripts\\video-knowledge.ps1 vision-provider-matrix --providers \"local_qwen_vl,volcengine_coding_plan,gemini,openai,agnes\" --bundle-dir {root}",
            f".\\scripts\\video-knowledge.ps1 local-vlm-serving-smoke --provider local_qwen_vl --bundle-dir {root}",
        ],
    }


def _compact_vision_provider_smoke(
    root: Path,
    manifest: dict[str, Any],
    *,
    json_key: str,
    markdown_key: str,
    json_fallback: str,
    markdown_fallback: str,
) -> dict[str, Any]:
    json_path = _bundle_path(root, str(manifest.get(json_key) or json_fallback))
    markdown_path = _bundle_path(root, str(manifest.get(markdown_key) or markdown_fallback))
    payload = _read_optional_object(json_path)
    provider = payload.get("provider") if isinstance(payload.get("provider"), dict) else {}
    return {
        "exists": json_path.exists(),
        "status": _text(payload.get("status") or ("missing_report" if not json_path.exists() else "unknown")),
        "safe_to_execute": bool(payload.get("safe_to_execute")),
        "provider": _text(provider.get("provider") or provider.get("name") or provider.get("id") or ""),
        "model": _text(provider.get("model") or ""),
        "error_class": _text(payload.get("error_class") or ""),
        "error_summary": _clip_text(_text(payload.get("error_summary") or ""), 220),
        "recovery_suggestion": _clip_text(_text(payload.get("recovery_suggestion") or ""), 260),
        "json_href": _relative_href(root, json_path),
        "markdown_href": _relative_href(root, markdown_path),
    }


def _compact_vision_provider_matrix(
    root: Path,
    manifest: dict[str, Any],
    *,
    json_key: str,
    markdown_key: str,
    json_fallback: str,
    markdown_fallback: str,
) -> dict[str, Any]:
    json_path = _bundle_path(root, str(manifest.get(json_key) or json_fallback))
    markdown_path = _bundle_path(root, str(manifest.get(markdown_key) or markdown_fallback))
    payload = _read_optional_object(json_path)
    ranking = payload.get("provider_ranking") if isinstance(payload.get("provider_ranking"), list) else []
    ready_count = sum(1 for row in ranking if isinstance(row, dict) and row.get("ready"))
    requested = payload.get("providers_requested") if isinstance(payload.get("providers_requested"), list) else []
    return {
        "exists": json_path.exists(),
        "status": _text(payload.get("status") or ("missing_report" if not json_path.exists() else "unknown")),
        "recommended_provider": _text(payload.get("recommended_provider") or ""),
        "provider_count": len(requested) or len(ranking),
        "ready_count": ready_count,
        "providers_requested": [_text(value) for value in requested if _text(value)],
        "json_href": _relative_href(root, json_path),
        "markdown_href": _relative_href(root, markdown_path),
    }


def _compact_local_vlm_smoke(
    root: Path,
    manifest: dict[str, Any],
    *,
    json_key: str,
    markdown_key: str,
    json_fallback: str,
    markdown_fallback: str,
) -> dict[str, Any]:
    json_path = _bundle_path(root, str(manifest.get(json_key) or json_fallback))
    markdown_path = _bundle_path(root, str(manifest.get(markdown_key) or markdown_fallback))
    payload = _read_optional_object(json_path)
    execute = bool(payload.get("execute"))
    ok = bool(payload.get("ok"))
    status = "missing_report"
    if json_path.exists():
        status = "executed_ready" if execute and ok else "executed_failed" if execute else "plan_only"
    matrix = payload.get("capability_matrix") if isinstance(payload.get("capability_matrix"), list) else []
    capability_counts: dict[str, int] = {}
    for row in matrix:
        if not isinstance(row, dict):
            continue
        key = _text(row.get("status") or "unknown") or "unknown"
        capability_counts[key] = capability_counts.get(key, 0) + 1
    profile = payload.get("profile") if isinstance(payload.get("profile"), dict) else {}
    return {
        "exists": json_path.exists(),
        "status": status,
        "ok": ok,
        "execute": execute,
        "provider": _text(payload.get("provider") or profile.get("provider") or ""),
        "model": _text(profile.get("model") or ""),
        "base_url": _text(profile.get("base_url") or ""),
        "short_frame_group_image_count": int((payload.get("input_spec") or {}).get("short_frame_group_image_count") or 0) if isinstance(payload.get("input_spec"), dict) else 0,
        "capability_counts": capability_counts,
        "json_href": _relative_href(root, json_path),
        "markdown_href": _relative_href(root, markdown_path),
    }

def _creative_workflow_summary(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    generation_path = _bundle_path(
        root,
        str(manifest.get("generation_contract_import_json") or "exports/generation-contract-import.json"),
    )
    previs_path = _bundle_path(
        root,
        str(manifest.get("previs_candidate_evidence_json") or "exports/previs-candidate-evidence.json"),
    )
    generation = _read_optional_object(generation_path)
    previs = _read_optional_object(previs_path)
    return {
        "generation": {
            "exists": generation_path.is_file(),
            "status": str(generation.get("status") or "missing"),
            "generator": str(generation.get("generator") or ""),
            "required_capability": str(generation.get("required_capability") or ""),
            "preflight": dict(generation.get("capability_preflight") or {}),
            "technical_verification": dict(generation.get("technical_verification") or {}),
            "visual_verification": dict(generation.get("visual_verification") or {}),
            "href": _relative_href(root, generation_path),
        },
        "previs": {
            "exists": previs_path.is_file(),
            "status": str(previs.get("status") or "missing"),
            "scene": dict(previs.get("scene") or {}),
            "camera_count": len(previs.get("cameras") or []),
            "capture_count": len(previs.get("captures") or []),
            "authority_boundary": dict(previs.get("authority_boundary") or {}),
            "href": _relative_href(root, previs_path),
        },
    }

def _creative_workflow_panel_html(summary: dict[str, Any]) -> str:
    generation = summary.get("generation") if isinstance(summary.get("generation"), dict) else {}
    previs = summary.get("previs") if isinstance(summary.get("previs"), dict) else {}
    if not generation.get("exists") and not previs.get("exists"):
        return '<div class="muted">No imported generation or 3D previs contract. VKP does not install, render, or fallback implicitly.</div>'
    cards: list[str] = []
    if generation.get("exists"):
        visual = generation.get("visual_verification") if isinstance(generation.get("visual_verification"), dict) else {}
        preflight = generation.get("preflight") if isinstance(generation.get("preflight"), dict) else {}
        cards.append(
            '<div class="artifact"><strong>Generation capability and representative frames</strong>'
            f'<div class="muted">status={html.escape(str(generation.get("status") or ""))}; generator={html.escape(str(generation.get("generator") or ""))}; capability={html.escape(str(generation.get("required_capability") or ""))}</div>'
            f'<div class="muted">preflight={html.escape(str(preflight.get("ready", False)))} via {html.escape(str(preflight.get("probe_method") or ""))}; representative_frames={len(visual.get("representative_frames") or [])}</div>'
            f'<a href="{html.escape(str(generation.get("href") or ""))}">machine-readable evidence</a></div>'
        )
    if previs.get("exists"):
        boundary = previs.get("authority_boundary") if isinstance(previs.get("authority_boundary"), dict) else {}
        cards.append(
            '<div class="artifact"><strong>3D previs candidate</strong>'
            f'<div class="muted">status={html.escape(str(previs.get("status") or ""))}; cameras={html.escape(str(previs.get("camera_count") or 0))}; captures={html.escape(str(previs.get("capture_count") or 0))}</div>'
            f'<div class="muted">synthetic={html.escape(str(boundary.get("synthetic", True)))}; observed_video_fact={html.escape(str(boundary.get("observed_video_fact", False)))}</div>'
            f'<a href="{html.escape(str(previs.get("href") or ""))}">machine-readable candidate</a></div>'
        )
    return "".join(cards)


def _ui_shell_asset_href(result: dict[str, Any], key: str, fallback: str) -> str:
    shell = result.get("ui_shell_assets") if isinstance(result.get("ui_shell_assets"), dict) else {}
    assets = shell.get("assets") if isinstance(shell.get("assets"), dict) else {}
    row = assets.get(key) if isinstance(assets.get(key), dict) else {}
    return html.escape(str(row.get("href") or fallback), quote=True)


def _surface_sidebar_html(registry: dict[str, Any]) -> str:
    group_markers = {
        "overview": "01",
        "processing": "02",
        "review": "03",
        "editing": "04",
        "evaluation": "05",
        "settings": "06",
    }
    group_blocks: list[str] = []
    for group in registry.get("groups") or []:
        if not isinstance(group, dict):
            continue
        group_key = str(group.get("key") or "")
        label = html.escape(str(group.get("label") or group_key))
        ready_count = int(group.get("ready_count") or 0)
        surface_count = int(group.get("surface_count") or 0)
        links: list[str] = []
        for surface in group.get("items") or []:
            if not isinstance(surface, dict):
                continue
            surface_id = html.escape(str(surface.get("surface_id") or ""), quote=True)
            title = html.escape(str(surface.get("title") or surface_id))
            href = html.escape(str(surface.get("href") or ""), quote=True)
            command = html.escape(str(surface.get("resolved_generation_command") or ""), quote=True)
            description = html.escape(str(surface.get("description") or ""), quote=True)
            status = str(surface.get("status") or "not_generated")
            ready = bool(surface.get("exists") or surface.get("is_current"))
            active = " active" if surface.get("is_current") else ""
            muted = " vkp-nav-missing" if not ready else ""
            status_label = html.escape(str(surface.get("status_label") or status))
            stability = html.escape(str(surface.get("stability") or ""))
            links.append(
                f'<a class="nav-link{active}{muted}" href="{href}" data-surface-id="{surface_id}" '
                f'data-surface-title="{title}" data-ready="{str(ready).lower()}" data-command="{command}" '
                f'onclick="return handleSurfaceNavigation(event, this)" title="{description}">'
                '<span class="nav-link-icon"><span class="vkp-status-dot '
                + html.escape(status, quote=True)
                + '"></span></span>'
                f'<span class="nav-link-title">{title}<small>{status_label} · {stability}</small></span></a>'
            )
        marker = html.escape(group_markers.get(group_key, "--"))
        group_blocks.append(
            '<section class="vkp-nav-group">'
            f'<div class="vkp-nav-group-title"><span>{marker} / {label}</span><span>{ready_count}/{surface_count}</span></div>'
            '<div class="navbar-nav">'
            + "".join(links)
            + "</div></section>"
        )
    return "".join(group_blocks)

def _render_workbench_html(result: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    title = html.escape(str(result.get("title") or "Video Workbench"))
    data_json = json.dumps({"result": result, "timeline": rows}, ensure_ascii=False).replace("</", "<\\/")
    cards_html = "\n".join(_artifact_card_html(card) for card in result.get("artifacts") or [])
    queue_html = _queue_panel_html(result.get("processing_queue") if isinstance(result.get("processing_queue"), dict) else {})
    closure_html = _review_closure_panel_html(result.get("review_closure") if isinstance(result.get("review_closure"), dict) else {})
    arbitration_html = _transcript_arbitration_panel_html(result.get("transcript_arbitration") if isinstance(result.get("transcript_arbitration"), dict) else {})
    term_status_html = _term_correction_status_panel_html(result.get("term_correction") if isinstance(result.get("term_correction"), dict) else {})
    semantic_correction_html = _semantic_correction_panel_html(result.get("semantic_correction") if isinstance(result.get("semantic_correction"), dict) else {})
    term_impact_html = _term_correction_impact_panel_html(result.get("term_correction_impact") if isinstance(result.get("term_correction_impact"), dict) else {})
    evidence_html = _evidence_status_panel_html(result.get("evidence_status") if isinstance(result.get("evidence_status"), dict) else {})
    panorama_html = panorama_panel_html(result.get("panorama") if isinstance(result.get("panorama"), dict) else {})
    provider_html = _provider_status_panel_html(result.get("provider_status") if isinstance(result.get("provider_status"), dict) else {})
    external_reuse_html = _external_reuse_status_panel_html(result.get("external_reuse_status") if isinstance(result.get("external_reuse_status"), dict) else {})
    autopilot_html = _autopilot_panel_html(
        result.get("autopilot") if isinstance(result.get("autopilot"), dict) else {},
        Path(str(result.get("bundle_dir") or ".")).resolve(),
    )
    summary_claim_review_html = _summary_claim_review_panel_html(result.get("summary_claim_review") or {})
    summary_stage_html = _summary_stage_panel_html(result.get("summary_stage_coverage") or {})
    citation_digest_html = _citation_digest_panel_html(result.get("citation_digest_integrity") or {})
    subqueue_action_html = _subqueue_action_plan_panel_html(result.get("subqueue_action_plan") if isinstance(result.get("subqueue_action_plan"), dict) else {})
    content_candidate_html = _content_candidate_panel_html(result.get("content_candidates") if isinstance(result.get("content_candidates"), dict) else {})
    creative_workflow_html = _creative_workflow_panel_html(result.get("creative_workflow") if isinstance(result.get("creative_workflow"), dict) else {})
    shot_review = result.get("shot_review") if isinstance(result.get("shot_review"), dict) else {}
    shot_assets = shot_review.get("asset_paths") if isinstance(shot_review.get("asset_paths"), dict) else {}
    wavesurfer_href = html.escape(str(shot_assets.get("wavesurfer") or "assets/wavesurfer-7.12.11/wavesurfer.min.js"))
    regions_href = html.escape(str(shot_assets.get("regions") or "assets/wavesurfer-7.12.11/regions.min.js"))
    shot_glue_href = html.escape(str(shot_assets.get("glue") or "assets/shot-review-workbench.js"))
    moment_html = _moment_search_panel_html(
        result.get("moment_index") if isinstance(result.get("moment_index"), dict) else {},
        result.get("video_rag_chunks") if isinstance(result.get("video_rag_chunks"), list) else [],
        result.get("video_rag_status") if isinstance(result.get("video_rag_status"), dict) else {},
    )
    surface_registry = result.get("ui_surface_registry") if isinstance(result.get("ui_surface_registry"), dict) else {}
    sidebar_html = _surface_sidebar_html(surface_registry)
    surface_summary = surface_registry.get("summary") if isinstance(surface_registry.get("summary"), dict) else {}
    surface_ready = int(surface_summary.get("ready_count") or 0)
    surface_total = int(surface_registry.get("surface_count") or 0)
    surface_missing = int(surface_summary.get("not_generated_count") or 0)
    tabler_css_href = _ui_shell_asset_href(result, "tabler_css", "assets/vkp-ui-shell/tabler-1.4.0.min.css")
    tabler_js_href = _ui_shell_asset_href(result, "tabler_js", "assets/vkp-ui-shell/tabler-1.4.0.min.js")
    media_chrome_href = _ui_shell_asset_href(result, "media_chrome_js", "assets/vkp-ui-shell/media-chrome-4.19.2.iife.min.js")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} - VKP 视频知识工作台</title>
  <link rel="stylesheet" href="{tabler_css_href}">
  <style>
    :root {{ color-scheme:light; --bg:#eef1f4; --panel:#fff; --line:#ccd4dc; --ink:#14212b; --muted:#61707d; --accent:#e85d04; --rail:#101820; --rail-line:#293844; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Bahnschrift,"Microsoft YaHei UI","Segoe UI",sans-serif; background:var(--bg); color:var(--ink); }}
    .vkp-shell {{ min-height:100vh; background:var(--bg); }}
    .vkp-sidebar {{ --tblr-navbar-bg:var(--rail); --tblr-navbar-color:#d8e1e8; background:var(--rail); border-right:1px solid var(--rail-line); }}
    .vkp-sidebar .container-fluid {{ align-items:stretch; }}
    .vkp-brand {{ display:flex; align-items:center; gap:10px; padding:16px 14px; color:#fff; letter-spacing:.08em; }}
    .vkp-brand-mark {{ width:34px; height:34px; display:grid; place-items:center; border:1px solid #ff8a3d; color:#ff8a3d; font-weight:800; font-size:12px; }}
    .vkp-brand-copy strong {{ display:block; font-size:13px; }}
    .vkp-brand-copy small {{ display:block; color:#91a2af; font-size:10px; letter-spacing:.12em; }}
    .vkp-nav-scroll {{ padding:4px 10px 18px; overflow-y:auto; }}
    .vkp-nav-meta {{ margin:0 10px 14px; padding:10px; border:1px solid var(--rail-line); color:#94a6b2; font-size:10px; letter-spacing:.04em; }}
    .vkp-nav-meta strong {{ display:block; margin:3px 0 7px; color:#fff; font-size:14px; }}
    .vkp-nav-meta a {{ color:#ff9a59; text-decoration:none; }}
    .vkp-nav-meta a:hover {{ color:#fff; }}
    .vkp-nav-group {{ margin:0 0 16px; }}
    .vkp-nav-group-title {{ display:flex; justify-content:space-between; padding:0 9px 6px; color:#7f929f; font-size:10px; letter-spacing:.08em; text-transform:uppercase; }}
    .vkp-sidebar .nav-link {{ min-height:42px; gap:7px; border-left:2px solid transparent; border-radius:0; color:#c8d3da; padding:7px 9px; }}
    .vkp-sidebar .nav-link:hover {{ background:#17242e; color:#fff; }}
    .vkp-sidebar .nav-link.active {{ background:#1d2c37; border-left-color:#ff7a24; color:#fff; }}
    .vkp-sidebar .nav-link.vkp-nav-missing {{ color:#788995; }}
    .vkp-sidebar .nav-link-icon {{ min-width:14px; width:14px; }}
    .vkp-sidebar .nav-link-title {{ display:flex; flex-direction:column; min-width:0; font-size:12px; }}
    .vkp-sidebar .nav-link-title small {{ color:#7f929f; font-size:9px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .vkp-status-dot {{ width:7px; height:7px; border-radius:50%; display:inline-block; background:#647580; box-shadow:0 0 0 3px rgba(100,117,128,.14); }}
    .vkp-status-dot.current {{ background:#ff7a24; box-shadow:0 0 0 3px rgba(255,122,36,.18); }}
    .vkp-status-dot.ready {{ background:#38b983; box-shadow:0 0 0 3px rgba(56,185,131,.16); }}
    .vkp-page-wrapper {{ min-width:0; }}
    .vkp-header {{ padding:16px 22px; background:#fff; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }}
    .vkp-eyebrow {{ margin-bottom:5px; color:#9d3d00; font-size:10px; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }}
    h1 {{ margin:0 0 6px; font-size:22px; }}
    h2 {{ margin:0 0 10px; font-size:16px; }}
    button {{ border:1px solid var(--line); background:#fff; border-radius:6px; padding:7px 10px; cursor:pointer; }}
    input, select {{ border:1px solid var(--line); border-radius:6px; padding:8px; font:inherit; }}
    .vkp-workbench-grid {{ display:grid; grid-template-columns:320px minmax(0,1fr); gap:12px; padding:12px; min-height:calc(100vh - 82px); }}
    .vkp-review-workspace {{ --video-pane-width:56%; display:grid; grid-template-columns:minmax(360px,var(--video-pane-width)) 12px minmax(320px,1fr); min-width:0; min-height:0; }}
    .vkp-review-workspace > .panel {{ min-width:0; min-height:0; }}
    .vkp-layout-splitter {{ position:relative; width:12px; min-width:12px; margin:0; padding:0; border:0; border-radius:0; background:transparent; cursor:col-resize; touch-action:none; outline:none; }}
    .vkp-layout-splitter::before {{ content:""; position:absolute; inset:8px 4px; border-radius:999px; background:#a8b4be; transition:background .15s, box-shadow .15s; }}
    .vkp-layout-splitter:hover::before, .vkp-layout-splitter:focus-visible::before, .vkp-layout-splitter.dragging::before {{ background:var(--accent); box-shadow:0 0 0 3px rgba(232,93,4,.15); }}
    .vkp-layout-splitter:focus-visible {{ box-shadow:none; }}
    body.vkp-resizing {{ cursor:col-resize; user-select:none; }}
    .vkp-panel-heading {{ display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:8px; margin-bottom:10px; }}
    .vkp-panel-heading h2 {{ margin:0; }}
    .vkp-layout-tools {{ display:flex; align-items:center; gap:7px; color:var(--muted); font-size:11px; }}
    .vkp-layout-tools button {{ padding:4px 7px; font-size:11px; }}
    .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; overflow:auto; }}
    .muted {{ color:var(--muted); }}
    .artifact-grid {{ display:grid; grid-template-columns:1fr; gap:8px; }}
    .artifact {{ border:1px solid var(--line); border-radius:8px; padding:10px; background:#fff; }}
    .artifact.missing {{ opacity:.62; }}
    .queue-card {{ border:1px solid var(--line); border-radius:8px; padding:10px; background:#fff; margin:0 0 8px; cursor:pointer; }}
    .queue-card.action_required {{ border-left:5px solid #b42318; }}
    .queue-card.ready {{ border-left:5px solid #0f6b4f; }}
    .queue-card.empty {{ border-left:5px solid #98a2b3; }}
    .queue-card ul {{ margin:6px 0; padding-left:18px; }}
    .queue-card code {{ margin-top:6px; }}
    .closure-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:8px 0; }}
    .closure-metric {{ border:1px solid var(--line); border-radius:8px; padding:8px; background:#fff; }}
    .closure-metric strong {{ display:block; font-size:18px; }}
    .closure-alert {{ border-left:5px solid #b42318; }}
    .closure-ok {{ border-left:5px solid #0f6b4f; }}
    .arbitration-row {{ width:100%; text-align:left; border:1px solid var(--line); border-radius:8px; padding:8px; margin:0 0 8px; background:#fff; }}
    .arbitration-row strong {{ display:block; }}
    .toolbar {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
    media-controller.vkp-media-controller {{ display:block; width:100%; height:auto; max-width:100%; max-height:none; aspect-ratio:16/9; background:#0b1116; border-radius:8px; overflow:hidden; --media-primary-color:#ff7a24; --media-secondary-color:#0b1116; }}
    media-controller.vkp-media-controller video {{ width:100%; height:100%; max-width:100%; max-height:none; background:#0b1116; display:block; object-fit:contain; }}
    .shot-review-panel {{ margin:12px 0; border:1px solid var(--line); border-radius:8px; padding:10px; background:#fbfcfe; }}
    #shotWaveform {{ min-height:72px; border:1px solid var(--line); border-radius:6px; background:#fff; margin:8px 0; }}
    .shot-review-row {{ display:grid; grid-template-columns:minmax(170px,1fr) 1fr 1fr; gap:6px; align-items:center; border-top:1px solid var(--line); padding:6px 0; }}
    .shot-review-row.active {{ background:rgba(36,84,198,.06); }}
    .shot-review-row label {{ display:flex; gap:4px; align-items:center; font-size:12px; }}
    .shot-review-row select {{ min-width:0; width:100%; }}
    .timeline-row {{ width:100%; text-align:left; border:1px solid var(--line); border-radius:8px; padding:9px; margin:0 0 8px; background:#fff; }}
    .timeline-row.active {{ border-color:var(--accent); box-shadow:0 0 0 2px rgba(36,84,198,.12); }}
    .badge {{ display:inline-block; border:1px solid var(--line); border-radius:999px; padding:2px 7px; font-size:12px; margin:4px 4px 0 0; color:var(--muted); }}
    .search {{ width:100%; margin-bottom:10px; }}
    .detail {{ line-height:1.55; }}
    iframe {{ width:100%; height:52vh; border:1px solid var(--line); border-radius:8px; background:#fff; }}
    pre {{ white-space:pre-wrap; word-break:break-word; background:#f1f4f8; border:1px solid var(--line); border-radius:6px; padding:8px; }}
    @media (min-width:1181px) {{
      .vkp-workbench-grid {{ height:calc(100vh - 158px); min-height:640px; overflow:hidden; }}
      .vkp-workbench-grid > .panel {{ height:100%; min-height:0; }}
      .vkp-review-workspace {{ height:100%; overflow:hidden; }}
      .vkp-review-workspace > .panel {{ height:100%; }}
    }}
    @media (max-width:1180px) {{ .vkp-workbench-grid {{ grid-template-columns:1fr; }} .vkp-review-workspace {{ display:block; }} .vkp-review-workspace > .panel {{ margin-bottom:12px; }} .vkp-layout-splitter {{ display:none; }} iframe {{ height:420px; }} }}
  </style>
</head>
<body>
  <div class="page vkp-shell">
    <aside class="navbar navbar-vertical navbar-expand-lg vkp-sidebar" data-bs-theme="dark" aria-label="VKP 工作台导航">
      <div class="container-fluid">
        <a class="navbar-brand vkp-brand" href="video-workbench.html">
          <span class="vkp-brand-mark">VKP</span>
          <span class="vkp-brand-copy"><strong>LOCAL WORKBENCH</strong><small>VIDEO KNOWLEDGE PIPELINE</small></span>
        </a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#vkpSidebarNav" aria-controls="vkpSidebarNav" aria-expanded="false" aria-label="展开导航">MENU</button>
        <div class="collapse navbar-collapse" id="vkpSidebarNav">
          <div class="vkp-nav-scroll">{sidebar_html}</div>
          <div class="vkp-nav-meta">
            UI REGISTRY / V1
            <strong>{surface_ready}/{surface_total} READY · {surface_missing} MISSING</strong>
            <a href="ui-surface-registry.md">查看界面与写回边界 →</a>
          </div>
        </div>
      </div>
    </aside>
    <div class="page-wrapper vkp-page-wrapper">
      <header class="vkp-header">
        <div>
          <div class="vkp-eyebrow">VKP // LOCAL // PRIMARY SURFACE</div>
          <h1>{title}</h1>
          <div class="muted">唯一首页：集中显示状态与入口；专业编辑器和评测页保持独立。草稿只存浏览器，正式写回仍受各自 loopback / 显式 apply 边界约束。</div>
        </div>
        <div class="toolbar">
          <button onclick="openArtifact('task_console')">任务控制台</button>
          <button onclick="openArtifact('review_html')">证据审核</button>
          <button onclick="openArtifact('targeted_review_html')">精确区间审核</button>
          <button onclick="openArtifact('subtitle_editor_html')">双轨字幕编辑</button>
          <button onclick="openArtifact('model_settings')">模型设置</button>
        </div>
      </header>
      <div class="page-body">
  <main class="vkp-workbench-grid">
    <aside class="panel">
      <h2>复核闭环</h2>
      {closure_html}
      <h2 style="margin-top:16px">字幕仲裁</h2>
      {arbitration_html}
      <h2 style="margin-top:16px">术语纠错闭环</h2>
      {term_status_html}
      <h2 style="margin-top:16px">通用语义纠错</h2>
      {semantic_correction_html}
      <h2 style="margin-top:16px">术语纠错影响</h2>
      {term_impact_html}
      <h2 style="margin-top:16px">证据状态</h2>
      {evidence_html}
      <h2 style="margin-top:16px">全景球面分析</h2>
      {panorama_html}
      <h2 style="margin-top:16px">Provider / 本地 VLM</h2>
      {provider_html}
      <h2 style="margin-top:16px">小模型 Autopilot</h2>
      {autopilot_html}
      {summary_claim_review_html}
      {summary_stage_html}
      {citation_digest_html}
      <h2 style="margin-top:16px">外部复用能力</h2>
      {external_reuse_html}
      <h2 style="margin-top:16px">下一步调度</h2>
      {subqueue_action_html}
      <h2 style="margin-top:16px">处理队列</h2>
      {queue_html}
      <h2 style="margin-top:16px">内容素材候选</h2>
      {content_candidate_html}
      <h2 style="margin-top:16px">生成 / 3D 预演证据</h2>
      {creative_workflow_html}
      <h2 style="margin-top:16px">入口与产物</h2>
      <div class="artifact-grid">{cards_html}</div>
    </aside>
    <div class="vkp-review-workspace" id="reviewWorkspace">
    <section class="panel" id="videoPane">
      <div class="vkp-panel-heading">
        <h2>视频与时间轴</h2>
        <div class="vkp-layout-tools"><output id="layoutRatio">画面 56% / 文字 44%</output><button id="layoutReset" type="button" title="恢复默认的画面与文字比例">重置比例</button></div>
      </div>
      <media-controller class="vkp-media-controller">
        <video id="player" slot="media" controls playsinline></video>
        <media-loading-indicator slot="centered-chrome" noautohide></media-loading-indicator>
        <media-control-bar>
          <media-play-button></media-play-button>
          <media-seek-backward-button seek-offset="10"></media-seek-backward-button>
          <media-seek-forward-button seek-offset="10"></media-seek-forward-button>
          <media-time-range></media-time-range>
          <media-time-display show-duration></media-time-display>
          <media-mute-button></media-mute-button>
          <media-volume-range></media-volume-range>
          <media-fullscreen-button></media-fullscreen-button>
        </media-control-bar>
      </media-controller>
      <div class="toolbar" style="margin:10px 0">
        <input id="mediaFile" type="file" accept="video/*,audio/*" onchange="loadMedia(this.files[0])">
        <span id="mediaHint" class="muted">选择原视频后，点击时间轴可跳转。</span>
      </div>
      <section id="shotReviewPanel" class="shot-review-panel" hidden>
        <h2>镜头边界与镜头语言复核</h2>
        <div id="shotReviewStatus" class="muted">正在载入审核草稿……</div>
        <div id="shotWaveform"></div>
        <div class="toolbar">
          <button id="shotSplit" type="button">在播放头拆分</button>
          <button id="shotMerge" type="button">与下一镜头合并</button>
          <button id="shotDownload" type="button">下载审核 JSON</button>
          <button id="shotSave" type="button">保存到 VKP</button>
        </div>
        <div id="shotReviewList"></div>
      </section>
      {moment_html}
      <input id="filter" class="search" placeholder="筛选字幕、OCR、route、问题" oninput="renderTimeline()">
      <div id="timeline"></div>
    </section>
    <button class="vkp-layout-splitter" id="reviewSplitter" type="button" role="separator" aria-label="拖动调整视频与文字审核区域宽度" aria-orientation="vertical" aria-valuemin="0" aria-valuemax="100" aria-valuenow="56" title="拖动调整画面与文字区域；双击恢复默认比例"></button>
    <section class="panel" id="detailPane">
      <h2 id="detailTitle">当前条目</h2>
      <div id="detail" class="detail muted">点击左侧时间轴查看 transcript、OCR、多模态状态和证据帧。</div>
      <h2 style="margin-top:16px">内嵌页面</h2>
      <select id="frameSelect" onchange="openArtifact(this.value)" style="width:100%; margin-bottom:8px"></select>
      <iframe id="artifactFrame"></iframe>
    </section>
    </div>
  </main>
      </div>
    </div>
  </div>
  <script src="{tabler_js_href}"></script>
  <script src="{media_chrome_href}"></script>
  <script id="workbenchData" type="application/json">{data_json}</script>
  <script src="{wavesurfer_href}"></script>
  <script src="{regions_href}"></script>
  <script src="{shot_glue_href}"></script>
  <script>
    const DATA = JSON.parse(document.getElementById('workbenchData').textContent);
    const ARTIFACTS = DATA.result.artifacts || [];
    const ROWS = DATA.timeline || [];
    const MOMENT_INDEX = DATA.result.moment_index || {{}};
    const VIDEO_RAG_CHUNKS = Array.isArray(DATA.result.video_rag_chunks) ? DATA.result.video_rag_chunks : [];
    const VIDEO_RAG_STATUS = DATA.result.video_rag_status || {{}};
    const TRANSCRIPT_ARBITRATION = DATA.result.transcript_arbitration || {{}};
    const SHOT_REVIEW = DATA.result.shot_review || {{}};
    const QUEUE_GROUPS = ((DATA.result.processing_queue || {{}}).groups || []);
    const UI_SURFACE_REGISTRY = DATA.result.ui_surface_registry || {{}};
    let activeIndex = 0;
    let activeQueueKey = "";
    function q(id) {{ return document.getElementById(id); }}
    const VIDEO_PANE_LAYOUT_KEY = 'vkp.video-workbench.video-pane-percent.v1';
    const VIDEO_PANE_DEFAULT_PERCENT = 56;
    function reviewLayoutBounds() {{
      const workspace = q('reviewWorkspace');
      const splitter = q('reviewSplitter');
      const available = Math.max(1, (workspace?.getBoundingClientRect().width || 1) - (splitter?.offsetWidth || 12));
      const minimumVideo = Math.min(360, Math.max(220, available * .34));
      const minimumText = Math.min(320, Math.max(240, available * .28));
      return {{
        available,
        min: Math.min(50, (minimumVideo / available) * 100),
        max: Math.max(50, ((available - minimumText) / available) * 100),
      }};
    }}
    function setVideoPanePercent(value, persist = true) {{
      const workspace = q('reviewWorkspace');
      const splitter = q('reviewSplitter');
      if (!workspace || !splitter) return;
      const bounds = reviewLayoutBounds();
      const percent = Math.max(bounds.min, Math.min(bounds.max, Number(value) || VIDEO_PANE_DEFAULT_PERCENT));
      workspace.style.setProperty('--video-pane-width', `${{percent}}%`);
      splitter.setAttribute('aria-valuemin', String(Math.round(bounds.min)));
      splitter.setAttribute('aria-valuemax', String(Math.round(bounds.max)));
      splitter.setAttribute('aria-valuenow', String(Math.round(percent)));
      const ratio = q('layoutRatio');
      if (ratio) ratio.textContent = `画面 ${{Math.round(percent)}}% / 文字 ${{Math.round(100 - percent)}}%`;
      if (persist) {{ try {{ localStorage.setItem(VIDEO_PANE_LAYOUT_KEY, String(percent)); }} catch (_) {{}} }}
    }}
    function resetReviewLayout() {{ setVideoPanePercent(VIDEO_PANE_DEFAULT_PERCENT); }}
    function bindReviewLayout() {{
      const workspace = q('reviewWorkspace');
      const splitter = q('reviewSplitter');
      if (!workspace || !splitter) return;
      let saved = VIDEO_PANE_DEFAULT_PERCENT;
      try {{ saved = Number(localStorage.getItem(VIDEO_PANE_LAYOUT_KEY)) || saved; }} catch (_) {{}}
      setVideoPanePercent(saved, false);
      let dragPointer = null;
      splitter.addEventListener('pointerdown', event => {{
        event.preventDefault();
        dragPointer = event.pointerId;
        splitter.setPointerCapture?.(event.pointerId);
        splitter.classList.add('dragging');
        document.body.classList.add('vkp-resizing');
      }});
      splitter.addEventListener('pointermove', event => {{
        if (dragPointer !== event.pointerId) return;
        const rect = workspace.getBoundingClientRect();
        const bounds = reviewLayoutBounds();
        setVideoPanePercent(((event.clientX - rect.left) / bounds.available) * 100, false);
      }});
      const finishDrag = event => {{
        if (dragPointer !== event.pointerId) return;
        dragPointer = null;
        splitter.classList.remove('dragging');
        document.body.classList.remove('vkp-resizing');
        try {{ splitter.releasePointerCapture?.(event.pointerId); }} catch (_) {{}}
        setVideoPanePercent(Number(splitter.getAttribute('aria-valuenow')) || VIDEO_PANE_DEFAULT_PERCENT);
      }};
      splitter.addEventListener('pointerup', finishDrag);
      splitter.addEventListener('pointercancel', finishDrag);
      splitter.addEventListener('dblclick', resetReviewLayout);
      splitter.addEventListener('keydown', event => {{
        const current = Number(splitter.getAttribute('aria-valuenow')) || VIDEO_PANE_DEFAULT_PERCENT;
        const bounds = reviewLayoutBounds();
        let next = current;
        if (event.key === 'ArrowLeft') next -= event.shiftKey ? 10 : 2;
        else if (event.key === 'ArrowRight') next += event.shiftKey ? 10 : 2;
        else if (event.key === 'Home') next = bounds.min;
        else if (event.key === 'End') next = bounds.max;
        else return;
        event.preventDefault();
        setVideoPanePercent(next);
      }});
      q('layoutReset')?.addEventListener('click', resetReviewLayout);
      if (window.ResizeObserver) new ResizeObserver(() => setVideoPanePercent(Number(splitter.getAttribute('aria-valuenow')) || saved, false)).observe(workspace);
      else window.addEventListener('resize', () => setVideoPanePercent(Number(splitter.getAttribute('aria-valuenow')) || saved, false));
    }}
    function esc(v) {{ return String(v || '').replace(/[&<>\"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c] || c)); }}
    function loadMedia(file) {{ if (!file) return; q('player').src = URL.createObjectURL(file); q('mediaHint').textContent = '已加载：' + file.name; window.VKPShotReview?.loadMedia(file); }}
    function artifact(key) {{ return ARTIFACTS.find(row => row.key === key) || {{}}; }}
    function handleSurfaceNavigation(event, link) {{
      if (String(link?.dataset?.ready || '') === 'true') return true;
      event?.preventDefault();
      const title = link?.dataset?.surfaceTitle || '专业页面';
      const command = link?.dataset?.command || '';
      q('detailTitle').textContent = `未生成：${{title}}`;
      q('detail').innerHTML = `<p>该页面已进入统一注册表，但当前 bundle 尚未生成对应文件。</p>${{command ? `<p><strong>恢复 / 生成命令</strong></p><pre id="surfaceRecoveryCommand">${{esc(command)}}</pre><button type="button" onclick="copyText('surfaceRecoveryCommand')">复制命令</button>` : '<p class="muted">当前页面由其专用准备流程生成；注册表不会臆造或自动启动写回服务。</p>'}}`;
      return false;
    }}
    function openArtifact(key) {{
      const item = artifact(key);
      if (!item.href) return;
      if (!item.exists) {{
        q('detailTitle').textContent = `未生成：${{item.label || key}}`;
        q('detail').innerHTML = `<p>目标：${{esc(item.href)}}</p>${{item.generation_command ? `<pre id="artifactRecoveryCommand">${{esc(item.generation_command)}}</pre><button type="button" onclick="copyText('artifactRecoveryCommand')">复制命令</button>` : '<p class="muted">没有可安全自动执行的生成命令。</p>'}}`;
        return;
      }}
      q('artifactFrame').src = item.href;
      q('frameSelect').value = key;
    }}
    function renderSelect() {{ q('frameSelect').innerHTML = ARTIFACTS.map(row => `<option value="${{esc(row.key)}}">${{esc(row.label)}}${{row.exists ? '' : '（未生成）'}}</option>`).join(''); }}
    async function copyText(id) {{ const el = q(id); if (!el) return; await navigator.clipboard.writeText(el.textContent || ''); }}
    function filterContentCandidates(filter) {{
      const active = filter || 'all';
      let shown = 0;
      document.querySelectorAll('.content-candidate-row').forEach(row => {{
        const tags = String(row.dataset.reviewFilters || '').split(/\\s+/).filter(Boolean);
        const visible = active === 'all' || tags.includes(active);
        row.style.display = visible ? '' : 'none';
        if (visible) shown += 1;
      }});
      const labels = {{all:'全部候选', unsampled:'未抽样候选', evidence_low:'抽样证据不足候选', usable:'可继续加工候选', citation_ready:'有Citation候选', citation_missing:'缺Citation候选', chapter_linked:'已关联章节候选', chapter_missing:'缺章节候选', moment_linked:'已关联片段候选', moment_missing:'缺片段候选'}};
      const status = q('contentCandidateFilterStatus');
      if (status) status.textContent = `当前显示：${{labels[active] || active}}，${{shown}} 条。`;
    }}
    function setFilter(value) {{ q('filter').value = value || ''; renderTimeline(); }}
    function badges(values) {{ return (values || []).map(value => `<span class="badge">${{esc(value)}}</span>`).join(''); }}
    function alignmentDetail(row) {{ const a = row.timeline_alignment || {{}}; const issues = (a.issues || []).join('、'); if (!issues) return '无'; return `issues=${{issues}}; review_start=${{a.review_start ?? ''}}; asr_start=${{a.asr_first_start ?? ''}}; suggested=${{a.suggested_review_start ?? ''}}`; }}
    function arbitrationDetail(row) {{ const a = row.transcript_arbitration || {{}}; if (!a.index && !a.corrected_text && !a.original_text) return '无'; const alts = (a.alternatives || []).map(x => `${{x.source_id || ''}}:${{x.text || ''}}`).join(' | '); return `reason=${{a.review_reason || ''}}; confidence=${{a.confidence ?? ''}}; chosen=${{a.chosen_source || ''}}; original=${{a.original_text || ''}}; corrected=${{a.corrected_text || ''}}; alternatives=${{alts}}`; }}
    function selectArbitration(index, seconds) {{ const row = ROWS.find(item => Number(item.index) === Number(index)) || nearestRow(Number(seconds || 0)); if (row) selectRow(row.index); const player = q('player'); if (player.src) {{ player.currentTime = Math.max(0, Number(seconds || (row ? row.start : 0) || 0)); player.play().catch(() => {{}}); }} setFilter('transcript_source_conflict'); }}
    function tileDetail(row) {{ const targets = row.tile_review_targets || []; if (!targets.length) return '无'; return targets.map(t => `${{t.tile_id || 'tile'}} conf=${{t.confidence ?? ''}} reasons=${{(t.reasons || []).join('/')}} evidence=${{t.evidence_path || ''}}`).join(' | '); }}
    function selectQueue(key) {{
      const group = QUEUE_GROUPS.find(item => String(item.key || '') === String(key || ''));
      if (!group) return;
      activeQueueKey = group.key || '';
      document.querySelectorAll('.queue-card').forEach(el => el.classList.toggle('active', el.dataset.queueKey === activeQueueKey));
      const runs = Array.isArray(group.runs) ? group.runs : [];
      const failed = Array.isArray(group.failed_items_preview) ? group.failed_items_preview : [];
      const nextActions = Array.isArray(group.next_actions) ? group.next_actions : [];
      const retryCommands = Array.isArray(group.retry_commands) ? group.retry_commands : [];
      q('detailTitle').textContent = `队列：${{group.label || group.key || 'Queue'}}`;
      const retryHtml = retryCommands.length ? retryCommands.slice(0, 3).map((cmd, idx) => `<pre id="queue-detail-retry-${{idx}}">${{esc(cmd)}}</pre><button type="button" onclick="copyText('queue-detail-retry-${{idx}}')">复制重试命令 ${{idx + 1}}</button>`).join('') : '<div class="muted">无重试命令。</div>';
      const runHtml = runs.length ? runs.map(run => `<li><span class="badge">${{esc(run.status || 'unknown')}}</span> ${{esc(run.title || run.run_id || 'run')}}<div class="muted">${{esc(run.summary || '')}}</div></li>`).join('') : '<li class="muted">暂无 run。</li>';
      const failedHtml = failed.length ? failed.map(item => `<li>${{esc([item.index, item.reason, item.detail].filter(Boolean).join(' / '))}}</li>`).join('') : '<li class="muted">暂无失败项。</li>';
      const nextHtml = nextActions.length ? nextActions.map(item => `<li>${{esc(item)}}</li>`).join('') : '<li class="muted">暂无下一步。</li>';
      q('detail').innerHTML = `<p><strong>说明</strong><br>${{esc(group.description || '')}}</p>
        <p><strong>状态</strong><br><span class="badge">${{esc(group.status || '')}}</span> runs=${{esc(group.run_count || 0)}} action=${{esc(group.action_required || 0)}} failed=${{esc(group.failed_count || 0)}}</p>
        <p><strong>Runs</strong></p><ul>${{runHtml}}</ul>
        <p><strong>失败项</strong></p><ul>${{failedHtml}}</ul>
        <p><strong>下一步</strong></p><ul>${{nextHtml}}</ul>
        <p><strong>重试命令</strong></p>${{retryHtml}}`;
    }}    function allSearchChunks() {{
      const moments = (Array.isArray(MOMENT_INDEX.chunks) ? MOMENT_INDEX.chunks : []).map((row, i) => {{
        const chunkId = `moment:${{row.chunk_index ?? i}}`;
        const text = [row.search_text, row.snippet, row.transcript_text, row.visual_text, row.temporal_text, (row.keywords || []).join(' ')].filter(Boolean).join(' ');
        return {{...row, id: chunkId, chunk_kind: 'moment', text, result_label: 'moment', original_chunk_index: row.chunk_index ?? i}};
      }});
      const rag = VIDEO_RAG_CHUNKS.map((row, i) => {{
        const chunkId = String(row.id || `rag:${{i}}`);
        const text = [row.text, (row.keywords || []).join(' '), (row.tags || []).join(' '), row.chunk_kind, row.memory_level, row.memory_id, row.parent_memory_id, (row.child_memory_ids || []).join(' '), (row.child_moment_indexes || []).join(' '), row.fact_status].filter(Boolean).join(' ');
        return {{...row, id: chunkId, text, result_label: row.chunk_kind || 'rag'}};
      }});
      return moments.concat(rag);
    }}
    function renderMomentSearch() {{
      const input = q('momentSearchInput');
      const results = q('momentResults');
      if (!input || !results) return;
      const chunks = allSearchChunks();
      const query = (input.value || '').trim().toLowerCase();
      const matches = chunks.filter(row => !query || String(row.text || JSON.stringify(row)).toLowerCase().includes(query)).slice(0, 24);
      if (!chunks.length) {{ results.innerHTML = '<div class="muted">还没有生成 video-moment-index.json 或 video-rag-chunks.jsonl。</div>'; return; }}
      if (!matches.length) {{ results.innerHTML = '<div class="muted">没有匹配片段。</div>'; return; }}
      results.innerHTML = matches.map((row) => {{
        const memoryBits = [];
        if (row.memory_level || row.memory_id) memoryBits.push(`memory=${{row.memory_level || '-'}}/${{row.memory_id || '-'}}`);
        if (row.parent_memory_id) memoryBits.push(`parent=${{row.parent_memory_id}}`);
        if ((row.child_memory_ids || []).length) memoryBits.push(`child memory=${{(row.child_memory_ids || []).join(',')}}`);
        if ((row.child_moment_indexes || []).length) memoryBits.push(`child moment=${{(row.child_moment_indexes || []).join(',')}}`);
        const hierarchyLine = memoryBits.length ? `<div class="muted">层级：${{esc(memoryBits.join(' -> '))}}</div>` : '';
        const factLine = row.fact_status ? `<div class="muted">fact status: ${{esc(row.fact_status)}}</div>` : '';
        return `<button class="moment-result" data-moment-start="${{Number(row.start || 0)}}" data-search-chunk-id="${{esc(row.id || '')}}">
        <strong>${{esc(row.start_time || row.start || '')}} - ${{esc(row.end_time || row.end || '')}}</strong>
        <span class="badge">${{esc(row.result_label || row.chunk_kind || 'moment')}}</span>
        ${{row.memory_level ? `<span class="badge">${{esc(row.memory_level)}} memory</span>` : ''}}
        ${{row.has_visual_evidence ? '<span class="badge">视觉证据</span>' : ''}}
        ${{row.has_temporal_evidence ? '<span class="badge">连续证据</span>' : ''}}
        <div>${{badges((row.keywords || []).concat(row.tags || []))}}</div>
        <div class="muted">timeline: ${{esc((row.timeline_indexes || []).join(','))}}</div>
        ${{hierarchyLine}}
        ${{factLine}}
        <div>${{esc(row.snippet || row.text || '')}}</div>
        <div class="muted">证据：${{esc((row.evidence_paths || []).slice(0, 3).join(' | ') || '无')}}</div>
      </button>`;
      }}).join('');
      results.querySelectorAll('.moment-result').forEach(el => el.addEventListener('click', () => selectSearchChunk(Number(el.dataset.momentStart || 0), el.dataset.searchChunkId)));
    }}
    function selectSearchChunk(seconds, chunkId) {{
      const chunks = allSearchChunks();
      const row = chunks.find(item => String(item.id || '') === String(chunkId || '')) || chunks.find(item => Number(item.start || 0) === Number(seconds || 0)) || {{}};
      q('momentResults')?.querySelectorAll('.moment-result').forEach(el => el.classList.toggle('active', String(el.dataset.searchChunkId || '') === String(chunkId || '')));
      const start = Math.max(0, Number(row.start ?? seconds ?? 0));
      const player = q('player');
      const indexes = (row.timeline_indexes || []).map(value => Number(value));
      const matched = ROWS.find(item => indexes.includes(Number(item.index))) || nearestRow(start);
      if (matched) selectRow(matched.index);
      if (player.src) {{ player.currentTime = start; player.play().catch(() => {{}}); }}
      q('mediaHint').textContent = `已定位片段：${{row.start_time || start + 's'}} - ${{row.end_time || ''}}`;
    }}
    function selectMoment(seconds, chunkIndex) {{ selectSearchChunk(seconds, `moment:${{chunkIndex}}`); }}    function nearestRow(seconds) {{
      if (!ROWS.length) return null;
      return ROWS.reduce((best, row) => {{
        const start = Number(row.start || 0);
        const end = Number(row.end || start);
        const distance = seconds >= start && seconds <= end ? 0 : Math.min(Math.abs(seconds - start), Math.abs(seconds - end));
        return !best || distance < best.distance ? {{row, distance}} : best;
      }}, null)?.row || null;
    }}    function renderTimeline() {{
      const query = (q('filter').value || '').toLowerCase();
      const rows = ROWS.filter(row => !query || JSON.stringify(row).toLowerCase().includes(query));
      q('timeline').innerHTML = rows.map(row => `<button class="timeline-row ${{row.index === activeIndex ? 'active' : ''}}" onclick="selectRow(${{row.index}})">
        <strong>#${{row.index}} · ${{esc(row.start_time)}} - ${{esc(row.end_time)}}</strong><br>
        <span class="badge">${{esc(row.route || 'unknown')}}</span>
        ${{row.has_visual_understanding ? '<span class="badge">视觉理解</span>' : ''}}
        ${{row.has_temporal_understanding ? '<span class="badge">连续理解</span>' : ''}}
        ${{row.needs_human_review ? '<span class="badge">待人审</span>' : ''}}
        ${{badges(row.evidence_flags)}}
        <div class="muted">${{esc((row.transcript || row.visual_text || '').slice(0, 120))}}</div>
      </button>`).join('') || '<div class="muted">没有匹配时间轴。</div>';
    }}
    function selectRow(index) {{
      const row = ROWS.find(item => Number(item.index) === Number(index));
      if (!row) return;
      activeIndex = row.index;
      const player = q('player');
      if (player.src) {{ player.currentTime = Math.max(0, Number(row.start || 0)); player.play().catch(() => {{}}); }}
      q('detailTitle').textContent = `#${{row.index}} · ${{row.start_time}} - ${{row.end_time}}`;
      q('detail').innerHTML = `<p><strong>Transcript</strong><br>${{esc(row.transcript || '无')}}</p>
        <p><strong>OCR / screen text</strong><br>${{esc(row.visual_text || '无')}}</p>
        <p><strong>质量问题</strong><br>${{esc((row.quality_issues || []).join('、') || '无')}}</p>
        <p><strong>证据标记</strong><br>${{esc((row.evidence_flags || []).join('、') || '无')}}</p>
        <p><strong>时间对齐</strong><br>${{esc(alignmentDetail(row))}}</p>
        <p><strong>字幕仲裁</strong><br>${{esc(arbitrationDetail(row))}}</p>
        <p><strong>Tile 复核</strong><br>${{esc(tileDetail(row))}}</p>
        <p><strong>证据帧</strong><br>${{esc((row.frame_paths || []).join(' | ') || '无')}}</p>`;
      renderTimeline();
    }}
    function boot() {{
      bindReviewLayout();
      renderSelect();
      renderMomentSearch();
      renderTimeline();
      if (ROWS.length) selectRow(ROWS[0].index);
      if (ARTIFACTS.length) openArtifact(ARTIFACTS[0].key);
      window.VKPShotReview?.init({{shotReview: SHOT_REVIEW, player: q('player')}});
      if (window.customElements?.whenDefined) customElements.whenDefined('media-controller').then(() => q('player')?.removeAttribute('controls'));
    }}
    boot();
  </script>
</body>
</html>
"""





def _moment_search_panel_html(moment_index: dict[str, Any], video_rag_chunks: list[dict[str, Any]] | None = None, video_rag_status: dict[str, Any] | None = None) -> str:
    status = video_rag_status if isinstance(video_rag_status, dict) else {}
    chunks = moment_index.get("chunks") if isinstance(moment_index.get("chunks"), list) else []
    rag_count = len(video_rag_chunks or [])
    summary = moment_index.get("summary") if isinstance(moment_index.get("summary"), dict) else {}
    count = int(summary.get("chunks") or len(chunks) or 0)
    duration = summary.get("duration_seconds", 0)
    if moment_index.get("error"):
        return f'<div class="moment-panel"><h2>片段搜索</h2><div class="muted">片段索引不可用：{html.escape(str(moment_index.get("error")))}</div></div>'
    meta = f"{count} 个片段"
    if rag_count:
        meta += f"，RAG chunks {rag_count}"
    if duration:
        meta += f"，覆盖 {format_timestamp(_seconds(duration))}"
    backend = str(status.get("retrieval_backend") or status.get("manifest_backend") or "keyword")
    backend_status = str(status.get("backend_status") or "unknown")
    sqlite_text = "SQLite index ready" if status.get("sqlite_index_exists") else "SQLite index not built"
    vector_text = "no vector backend started" if status.get("no_vector_backend_started", True) else "vector backend status unknown"
    search_text = "search report ready" if status.get("search_report_exists") else "search report not generated"
    return f"""
      <div class="moment-panel">
        <h2>片段搜索</h2>
        <div class="muted">复用 VideoRAG-style moment index；本地搜索，不启动服务、不调用云。</div>
        <div class="muted">{html.escape(meta)}</div>
        <div class="muted">检索后端：{html.escape(backend)} / {html.escape(backend_status)}；{html.escape(sqlite_text)}；{html.escape(search_text)}；{html.escape(vector_text)}</div>
        <div class="moment-search-row"><input id="momentSearchInput" placeholder="搜索工具名、步骤、案例、价格、看屏幕" oninput="renderMomentSearch()"><button type="button" onclick="renderMomentSearch()">搜索</button></div>
        <div id="momentResults"></div>
      </div>
    """




def _format_rate_label(value: Any) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return str(value)


def _semantic_correction_panel_html(status: dict[str, Any]) -> str:
    if not status:
        return "<div class=\"muted\">尚未生成通用语义纠错状态；先运行 transcript-semantic-correction-pack。</div>"
    value = str(status.get("status") or "missing")
    residual = int(status.get("final_residual_error_total") or 0)
    readable_status = str(status.get("readable_impact_status") or "missing")
    readable_residual = int(status.get("readable_required_residual_total") or 0)
    llm_status = str(status.get("llm_draft_status") or "not_planned")
    llm_next_action = str(status.get("llm_draft_next_action") or "run_llm_draft_preview")
    llm_decision_count = int(status.get("llm_draft_decision_count") or 0)
    candidate_count = int(status.get("candidate_count") or 0)
    accepted_count = int(status.get("accepted_decision_count") or 0)
    cls = "closure-ok" if value in {"impact_passed", "no_candidates"} and residual == 0 else "closure-alert"
    commands = status.get("commands") if isinstance(status.get("commands"), dict) else {}
    next_action = str(status.get("next_action_key") or "")
    command_html = ""
    if next_action and commands.get(next_action.replace("build_pack", "pack")):
        command_html = f"<pre>{html.escape(str(commands.get(next_action)))}</pre>"
    elif commands:
        command_html = "".join(f"<pre>{html.escape(str(v))}</pre>" for v in list(commands.values())[:2] if str(v).strip())
    return f"""
    <div class=\"closure-grid\">
      <div class=\"closure-metric {cls}\"><span class=\"muted\">状态</span><strong>{html.escape(value)}</strong><small>{html.escape(next_action)}</small></div>
      <div class=\"closure-metric\"><span class=\"muted\">候选/接受</span><strong>{candidate_count} / {accepted_count}</strong></div>
      <div class=\"closure-metric {'closure-alert' if residual else 'closure-ok'}\"><span class=\"muted\">最终残留</span><strong>{residual}</strong></div>
      <div class=\"closure-metric {'closure-alert' if readable_status not in {"passed", "no_accepted_decisions"} or readable_residual else 'closure-ok'}\"><span class=\"muted\">可读文件影响</span><strong>{html.escape(readable_status)}</strong><small>residual {readable_residual}</small></div>
      <div class=\"closure-metric {'closure-ok' if llm_status in {"executed", "prompt_ready"} else 'closure-alert'}\"><span class=\"muted\">LLM/Codex 草稿</span><strong>{html.escape(llm_status)}</strong><small>{html.escape(llm_next_action)} · {llm_decision_count}</small></div>
      <div class=\"closure-metric\"><span class=\"muted\">写入边界</span><strong>本地校验</strong><small>raw ASR 不覆盖</small></div>
    </div>
    <div class=\"toolbar\"><button onclick=\"openArtifact('transcript_semantic_correction_prompt_markdown')\">Prompt</button><button onclick=\"openArtifact('transcript_semantic_correction_llm_prompt_markdown')\">LLM Prompt</button><button onclick=\"openArtifact('transcript_semantic_correction_pack_json')\">证据包</button><button onclick=\"openArtifact('transcript_semantic_correction_result_llm_markdown')\">LLM 回复</button><button onclick=\"openArtifact('transcript_semantic_correction_validation_markdown')\">预检</button><button onclick=\"openArtifact('transcript_semantic_correction_closure_markdown')\">闭环</button><button onclick=\"openArtifact('transcript_semantic_correction_impact_report_markdown')\">影响</button><button onclick=\"openArtifact('transcript_semantic_correction_readable_impact_markdown')\">可读影响</button></div>
    {command_html}
    """

def _term_correction_status_panel_html(status: dict[str, Any]) -> str:
    value = str(status.get("status") or "missing")
    accepted = int(status.get("accepted_term_count") or 0)
    validation_status = str(status.get("term_validation_status") or "missing")
    accepted_validation = int(status.get("accepted_validation_decisions") or 0)
    rejected_validation = int(status.get("rejected_validation_decisions") or 0)
    final_alias_total = int(status.get("final_export_alias_total") or 0)
    quality_passed = bool(status.get("smart_summary_quality_passed"))
    source_ok = bool(status.get("source_arbitrated_transcript_exists"))
    cls = "closure-ok" if value in {"completed", "ready"} and final_alias_total == 0 and quality_passed else "closure-alert"
    validation_needs_attention = value == "needs_codex_term_validation" or validation_status in {"invalid", "no_accepted_decisions"}
    validation_cls = "closure-alert" if validation_needs_attention else "closure-ok"
    next_action = str(status.get("next_action_key") or "")
    codex_html = _term_codex_substitute_html(status.get("codex_substitute") if isinstance(status.get("codex_substitute"), dict) else {})
    next_html = "<li class=\"muted\">暂无下一步。</li>" if not next_action else "<li>建议下一步：<code>" + html.escape(next_action) + "</code></li>"
    return f"""
      <div class=\"closure-grid\">
        <div class=\"closure-metric {cls}\"><span class=\"muted\">闭环状态</span><strong>{html.escape(value)}</strong></div>
        <div class=\"closure-metric closure-ok\"><span class=\"muted\">已接受术语</span><strong>{accepted}</strong></div>
        <div class=\"closure-metric {validation_cls}\"><span class=\"muted\">Codex预检</span><strong>{html.escape(validation_status)}</strong></div>
        <div class=\"closure-metric {validation_cls}\"><span class=\"muted\">预检接受/拒绝</span><strong>{accepted_validation}/{rejected_validation}</strong></div>
        <div class=\"closure-metric {'closure-ok' if source_ok else 'closure-alert'}\"><span class=\"muted\">纠正版转写</span><strong>{'yes' if source_ok else 'no'}</strong></div>
        <div class=\"closure-metric {cls}\"><span class=\"muted\">最终残留</span><strong>{final_alias_total}</strong></div>
        <div class=\"closure-metric {'closure-ok' if quality_passed else 'closure-alert'}\"><span class=\"muted\">智能总结质量</span><strong>{'passed' if quality_passed else 'pending'}</strong></div>
        <div class=\"closure-metric\"><span class=\"muted\">影响报告</span><strong>{html.escape(str(status.get('impact_status') or 'missing'))}</strong></div>
      </div>
      <div class=\"toolbar\"><button onclick=\"openArtifact('term_correction_closure_markdown')\">打开闭环报告</button><button onclick=\"openArtifact('term_arbitration_codex_validation_markdown')\">打开预检报告</button><button onclick=\"openArtifact('term_arbitration_glossary_json')\">打开术语词典</button></div>
      {codex_html}
      <details><summary>下一步</summary><ul>{next_html}</ul></details>
    """


def _term_codex_substitute_html(substitute: dict[str, Any]) -> str:
    if not substitute:
        return '<div class="muted">尚未生成 Codex 术语语义仲裁操作契约。</div>'
    commands = substitute.get("commands") if isinstance(substitute.get("commands"), dict) else {}
    command_rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td><code>{html.escape(str(value))}</code></td></tr>"
        for key, value in commands.items()
        if str(value).strip()
    ) or '<tr><td colspan="2" class="muted">暂无命令。</td></tr>'
    return f"""
      <details open class=\"mini-panel\"><summary>Codex 术语/工具名语义仲裁</summary>
        <p class=\"muted\">当前用 Codex 临时代替在线文本 LLM。规则草稿不能直接当结论；高置信替换必须通过预检。</p>
        <table>
          <tr><td>Prompt</td><td><code>{html.escape(str(substitute.get('prompt_markdown') or ''))}</code></td></tr>
          <tr><td>证据包</td><td><code>{html.escape(str(substitute.get('context_pack_json') or ''))}</code></td></tr>
          <tr><td>结果保存</td><td><code>{html.escape(str(substitute.get('suggested_result_markdown') or ''))}</code></td></tr>
          <tr><td>验收规则</td><td>{html.escape(str(substitute.get('acceptance_rule') or ''))}</td></tr>
        </table>
        <table><thead><tr><th>动作</th><th>命令</th></tr></thead><tbody>{command_rows}</tbody></table>
      </details>
    """

def _term_correction_impact_panel_html(impact: dict[str, Any]) -> str:
    if not impact.get("exists"):
        return "<div class=\"muted\">尚未生成 term-correction-impact-report；先运行术语纠错影响报告，确认高置信术语是否进入最终导出。</div>"
    status = str(impact.get("status") or "unknown")
    source_total = int(impact.get("source_alias_total") or 0)
    output_total = int(impact.get("output_alias_total") or 0)
    final_total = int(impact.get("final_export_alias_total") or 0)
    replacement_count = int(impact.get("replacement_count") or 0)
    ok = bool(impact.get("ok")) and final_total == 0
    cls = "closure-ok" if ok else "closure-alert"
    terms = impact.get("terms") if isinstance(impact.get("terms"), list) else []
    rows = []
    for row in terms[:8]:
        if not isinstance(row, dict):
            continue
        row_cls = "closure-ok" if row.get("resolved_in_outputs") else "closure-alert"
        rows.append(
            "<div class=\"closure-metric {row_cls}\"><span class=\"muted\">{alias}</span>"
            "<strong>{canonical}</strong><small>source {source} / output {output}</small></div>".format(
                row_cls=row_cls,
                alias=html.escape(str(row.get("alias") or "")),
                canonical=html.escape(str(row.get("canonical") or "")),
                source=int(row.get("source_alias_count") or 0),
                output=int(row.get("output_alias_count") or 0),
            )
        )
    term_rows = "".join(rows) or "<div class=\"muted\">没有术语明细。</div>"
    actions = impact.get("next_actions") if isinstance(impact.get("next_actions"), list) else []
    action_html = "".join("<li>" + html.escape(str(item)) + "</li>" for item in actions[:5])
    if not action_html:
        action_html = "<li class=\"muted\">暂无下一步。</li>"
    return f"""
      <div class=\"closure-grid\">
        <div class=\"closure-metric {cls}\"><span class=\"muted\">状态</span><strong>{html.escape(status)}</strong><small>{'clean' if ok else 'needs attention'}</small></div>
        <div class=\"closure-metric\"><span class=\"muted\">术语规则</span><strong>{replacement_count}</strong></div>
        <div class=\"closure-metric\"><span class=\"muted\">源材料错词</span><strong>{source_total}</strong></div>
        <div class=\"closure-metric {cls}\"><span class=\"muted\">最终导出残留</span><strong>{final_total}</strong><small>outputs {output_total}</small></div>
        <div class=\"closure-metric closure-ok\"><span class=\"muted\">纠错降低率</span><strong>{html.escape(_format_rate_label(impact.get('reduction_rate')))}</strong></div>
        <div class=\"closure-metric {cls}\"><span class=\"muted\">最终清洁率</span><strong>{html.escape(_format_rate_label(impact.get('final_clean_rate')))}</strong></div>
      </div>
      <div class=\"toolbar\"><button onclick=\"openArtifact('term_correction_impact_report_markdown')\">打开影响报告</button><button onclick=\"openArtifact('transcript_source_arbitration_markdown')\">打开仲裁报告</button></div>
      <div class=\"closure-grid\" style=\"margin-top:8px\">{term_rows}</div>
      <details><summary>下一步</summary><ul>{action_html}</ul></details>
    """
def _transcript_arbitration_panel_html(arbitration: dict[str, Any]) -> str:
    if not arbitration.get("exists"):
        return "<div class=\"muted\">尚未生成 transcript-source-arbitration；先运行字幕/ASR 多源仲裁。</div>"
    summary = arbitration.get("summary") if isinstance(arbitration.get("summary"), dict) else {}
    quality = arbitration.get("quality_summary") if isinstance(arbitration.get("quality_summary"), dict) else {}
    policy = arbitration.get("summary_input_policy") if isinstance(arbitration.get("summary_input_policy"), dict) else {}
    review_refs = arbitration.get("review_segment_refs") if isinstance(arbitration.get("review_segment_refs"), list) else []
    review_rows = arbitration.get("review_rows") if isinstance(arbitration.get("review_rows"), list) else []
    changed_rows = arbitration.get("changed_rows") if isinstance(arbitration.get("changed_rows"), list) else []
    rows = review_refs or review_rows or changed_rows
    quality_status = str(quality.get("status") or summary.get("quality_status") or arbitration.get("status") or "unknown")
    average_confidence = quality.get("average_confidence", summary.get("average_confidence", ""))
    high_conf_terms = quality.get("high_confidence_term_replacements", summary.get("high_confidence_term_replacements", 0))
    low_conflicts = quality.get("low_confidence_conflicts", summary.get("low_confidence_conflicts", len(review_rows)))
    metrics = f"""
      <div class=\"closure-grid\">
        <div class=\"closure-metric {'closure-alert' if review_rows or low_conflicts else 'closure-ok'}\"><span class=\"muted\">待复核冲突</span><strong>{int(low_conflicts or len(review_rows) or 0)}</strong></div>
        <div class=\"closure-metric closure-ok\"><span class=\"muted\">已改写片段</span><strong>{int(summary.get('changed_segments') or len(changed_rows) or 0)}</strong></div>
        <div class=\"closure-metric closure-ok\"><span class=\"muted\">高置信术语</span><strong>{int(high_conf_terms or 0)}</strong></div>
        <div class=\"closure-metric\"><span class=\"muted\">仲裁质量</span><strong>{html.escape(quality_status)}</strong><small>avg {html.escape(str(average_confidence))}</small></div>
        <div class="closure-metric {'closure-alert' if policy.get('must_exclude_review_segments') else 'closure-ok'}"><span class="muted">总结输入</span><strong>{html.escape(str(policy.get('mode') or 'unknown'))}</strong><small>safe {html.escape(str(quality.get('safe_segment_count', '')))}</small></div>
      </div>
    """
    command_html = "".join("<pre>" + html.escape(str(cmd)) + "</pre>" for cmd in (arbitration.get("next_commands") or [])[:3])
    policy_html = "<div class=\"muted\">策略：" + html.escape(str(policy.get("guidance") or "")) + "</div>"
    toolbar = '<div class="toolbar"><button onclick="setFilter(\'transcript_source_conflict\')">筛字幕冲突</button><button onclick="openArtifact(\'transcript_source_arbitration_markdown\')">打开仲裁报告</button><button onclick="openArtifact(\'subtitle_editor_html\')">打开双轨字幕编辑</button></div>' + policy_html + command_html
    if not rows:
        return metrics + toolbar + "<div class=\"muted\">没有发现字幕/ASR 冲突。</div>"
    cards = []
    for row in rows[:8]:
        if not isinstance(row, dict):
            continue
        alternatives = "; ".join(
            f"{alt.get('source_id', '')}:{str(alt.get('text', ''))[:80]}"
            for alt in (row.get("alternatives") or [])[:3]
            if isinstance(alt, dict)
        )
        cards.append(
            "<button class=\"arbitration-row\" onclick=\"selectArbitration({index}, {start})\">"
            "<strong>#{index} · {start_time} · {reason}</strong>"
            "<div><span class=\"badge\">{source}</span><span class=\"badge\">conf={confidence}</span></div>"
            "<div>原文：{original}</div>"
            "<div>纠正：{corrected}</div>"
            "<div class=\"muted\">候选：{alternatives}</div>"
            "</button>".format(
                index=int(row.get("index") or 0),
                start=float(row.get("start") or 0),
                start_time=html.escape(str(row.get("start_time") or "")),
                reason=html.escape(str(row.get("review_reason") or "changed")),
                source=html.escape(str(row.get("chosen_source") or "unknown_source")),
                confidence=html.escape(str(row.get("confidence") if row.get("confidence") is not None else "")),
                original=html.escape(str(row.get("original_text") or "")[:180]),
                corrected=html.escape(str(row.get("corrected_text") or "")[:180]),
                alternatives=html.escape(alternatives),
            )
        )
    return metrics + toolbar + "".join(cards)


def _review_closure_panel_html(closure: dict[str, Any]) -> str:
    if not closure.get("exists"):
        return "<div class=\"muted\">尚未生成 review-closure-status；先运行复核关闭进度命令。</div>"
    summary = closure.get("summary") if isinstance(closure.get("summary"), dict) else {}
    arbitration = closure.get("transcript_arbitration") if isinstance(closure.get("transcript_arbitration"), dict) else {}
    status = str(arbitration.get("status") or "none")
    cls = "closure-alert" if status == "needs_review" else "closure-ok" if status == "closed" else ""
    report = html.escape(str(closure.get("markdown_href") or "review-closure-status.md"))
    return f"""
      <div class=\"closure-grid\">
        <div class=\"closure-metric\"><span class=\"muted\">Open</span><strong>{int(summary.get('open') or 0)}</strong></div>
        <div class=\"closure-metric\"><span class=\"muted\">Closed</span><strong>{int(summary.get('closed') or 0)}</strong></div>
        <div class=\"closure-metric {cls}\"><span class=\"muted\">字幕仲裁待复核</span><strong>{int(arbitration.get('open') or 0)}</strong></div>
        <div class=\"closure-metric closure-ok\"><span class=\"muted\">字幕仲裁已关闭</span><strong>{int(arbitration.get('closed') or 0)}</strong></div>
      </div>
      <div class=\"toolbar\"><button onclick=\"openArtifact('review_closure_status')\">查看关闭报告</button><button onclick=\"openArtifact('subtitle_editor_html')\">打开双轨字幕编辑</button></div>
      <div class=\"muted\" style=\"margin-top:6px\">报告：<a href=\"{report}\">review-closure-status.md</a></div>
    """


def _evidence_status_panel_html(status: dict[str, Any]) -> str:
    alignment = status.get("timeline_alignment") if isinstance(status.get("timeline_alignment"), dict) else {}
    tile = status.get("tile_review") if isinstance(status.get("tile_review"), dict) else {}
    moment = status.get("video_moment_index") if isinstance(status.get("video_moment_index"), dict) else {}
    all_levels = status.get("visual_evidence_levels") if isinstance(status.get("visual_evidence_levels"), dict) else {}
    visual_levels = all_levels.get("visual") if isinstance(all_levels.get("visual"), dict) else {}
    temporal_levels = all_levels.get("temporal") if isinstance(all_levels.get("temporal"), dict) else {}
    tag_levels = all_levels.get("continuous_tag") if isinstance(all_levels.get("continuous_tag"), dict) else {}
    temporal_tag = status.get("temporal_tag_delta") if isinstance(status.get("temporal_tag_delta"), dict) else {}
    temporal_tag_commands = temporal_tag.get("commands") if isinstance(temporal_tag.get("commands"), dict) else {}
    codex_session = status.get("codex_vision_session") if isinstance(status.get("codex_vision_session"), dict) else {}
    codex_commands = codex_session.get("commands") if isinstance(codex_session.get("commands"), dict) else {}
    codex_validation = codex_session.get("validation") if isinstance(codex_session.get("validation"), dict) else {}
    codex_pending = str(codex_session.get("phase") or "") == "prepared"
    alignment_cls = "closure-alert" if int(alignment.get("issue_count") or 0) else "closure-ok" if alignment.get("exists") else ""
    tile_cls = "closure-alert" if int(tile.get("target_count") or 0) or int(tile.get("item_count") or 0) else "closure-ok"
    moment_cls = "closure-ok" if int(moment.get("chunk_count") or 0) else ""
    return f"""
      <div class=\"closure-grid\">
        <div class=\"closure-metric {alignment_cls}\"><span class=\"muted\">时间错位</span><strong>{int(alignment.get('issue_count') or 0)}</strong></div>
        <div class=\"closure-metric {tile_cls}\"><span class=\"muted\">Tile 待复核</span><strong>{int(tile.get('target_count') or tile.get('item_count') or 0)}</strong></div>
        <div class=\"closure-metric {moment_cls}\"><span class=\"muted\">片段索引</span><strong>{int(moment.get('chunk_count') or 0)}</strong></div>
        <div class=\"closure-metric\"><span class=\"muted\">覆盖时长</span><strong>{int(float(moment.get('duration_seconds') or 0))}s</strong></div>
        <div class=\"closure-metric\"><span class=\"muted\">单帧证据 C/I/U/N</span><strong>{int(visual_levels.get('confirmed') or 0)}/{int(visual_levels.get('inferred') or 0)}/{int(visual_levels.get('unavailable') or 0)}/{int(visual_levels.get('not_seen') or 0)}</strong></div>
        <div class=\"closure-metric\"><span class=\"muted\">连续证据 C/I/U/N</span><strong>{int(temporal_levels.get('confirmed') or 0)}/{int(temporal_levels.get('inferred') or 0)}/{int(temporal_levels.get('unavailable') or 0)}/{int(temporal_levels.get('not_seen') or 0)}</strong></div>
        <div class=\"closure-metric\"><span class=\"muted\">连续打标 弱/升级/缺口/未看</span><strong>{int(tag_levels.get('candidate_weak') or 0)}/{int(tag_levels.get('escalated') or 0)}/{int(tag_levels.get('unavailable') or 0)}/{int(tag_levels.get('not_seen') or 0)}</strong></div>
        <div class=\"closure-metric\"><span class=\"muted\">打标 Run / 状态</span><strong>{html.escape(str(temporal_tag.get('run_id') or 'none'))} / {html.escape(str(temporal_tag.get('status') or 'not_started'))}</strong></div>
        <div class=\"closure-metric\"><span class=\"muted\">Session</span><strong>{html.escape(str(codex_session.get('session_id') or 'not_prepared'))}</strong></div>
        <div class=\"closure-metric\"><span class=\"muted\">Codex session</span><strong>{html.escape(str(codex_session.get('phase') or 'not_prepared'))}</strong></div>
        <div class=\"closure-metric\"><span class=\"muted\">待导入 / 验证</span><strong>{'yes' if codex_pending else 'no'} / {html.escape(str(codex_validation.get('status') or 'not_validated'))}</strong></div>
        <div class=\"closure-metric\"><span class=\"muted\">实际模型</span><strong>{html.escape(str(codex_session.get('model_actual') or 'pending'))}</strong></div>
      </div>
      <div class=\"toolbar\"><pre id=\"codexPrepareSemantic\">{html.escape(str(codex_commands.get('prepare_semantic') or ''))}</pre><button type=\"button\" onclick=\"copyText('codexPrepareSemantic')\">准备 Codex 单帧视觉包</button><pre id=\"codexPrepareTemporal\">{html.escape(str(codex_commands.get('prepare_temporal') or ''))}</pre><button type=\"button\" onclick=\"copyText('codexPrepareTemporal')\">准备 Codex 连续帧包</button><pre id=\"codexImportResults\">{html.escape(str(codex_commands.get('import_results') or ''))}</pre><button type=\"button\" onclick=\"copyText('codexImportResults')\">复制导入命令</button></div>
      <div class=\"toolbar\"><pre id=\"temporalTagRun\">{html.escape(str(temporal_tag_commands.get('run') or ''))}</pre><button type=\"button\" onclick=\"copyText('temporalTagRun')\">运行连续打标</button><pre id=\"temporalTagResume\">{html.escape(str(temporal_tag_commands.get('resume') or ''))}</pre><button type=\"button\" onclick=\"copyText('temporalTagResume')\">恢复连续打标</button><button onclick=\"openArtifact('temporal_tag_delta_report')\">打标报告</button><button onclick=\"openArtifact('temporal_tag_delta_input_template')\">结果模板</button><button onclick=\"openArtifact('temporal_tag_delta_json')\">Raw / 审计</button></div>
      <div class=\"toolbar\"><button onclick=\"setFilter('timeline_alignment_issue')\">筛时间错位</button><button onclick=\"setFilter('inferred')\">筛推断证据</button><button onclick=\"setFilter('unavailable')\">筛不可判断</button><button onclick=\"setFilter('needs_high_res_tile_recovery')\">筛高分辨率Tile</button><button onclick=\"setFilter('tile_result_needs_review')\">筛 Tile 复核</button><button onclick=\"openArtifact('codex_vision_session_prompt_markdown')\">Codex Prompt</button><button onclick=\"openArtifact('codex_vision_session_result_template_json')\">结果模板</button><button onclick=\"openArtifact('codex_vision_session_raw_result_json')\">Raw result</button><button onclick=\"openArtifact('codex_vision_session_import_validation_json')\">验证报告</button><button onclick=\"openArtifact('vision_analysis_runs')\">审计</button><button onclick=\"openArtifact('vision_restore_plan_markdown')\">Restore</button></div>
    """


def _subqueue_action_plan_panel_html(plan: dict[str, Any]) -> str:
    rows = plan.get("rows") if isinstance(plan.get("rows"), list) else []
    if not rows:
        return '<div class="muted">尚未生成下一步调度单。先运行 task console 或 subqueue-action-plan。</div>'
    action_required = int(plan.get("action_required_count") or 0)
    summary = f"{int(plan.get('row_count') or len(rows))} 个子队列动作，{action_required} 个需要处理。"
    cards = []
    for idx, row in enumerate(rows[:8], start=1):
        if not isinstance(row, dict):
            continue
        key = html.escape(str(row.get("key") or f"subqueue-{idx}"), quote=True)
        label = html.escape(str(row.get("label") or row.get("subqueue_key") or "subqueue"))
        group_label = html.escape(str(row.get("group_label") or row.get("group_key") or "group"))
        status = html.escape(str(row.get("action_status") or row.get("status") or "unknown"))
        kind = html.escape(str(row.get("action_kind") or "ready_or_empty"))
        priority = html.escape(str(row.get("priority") or 90))
        reason = html.escape(str(row.get("blocked_reason") or ""))
        hint = html.escape(str(row.get("safe_execution_hint") or ""))
        machine = "machine" if row.get("machine_action_available") else "human"
        command = str(row.get("primary_command") or "").strip()
        command_html = ""
        if command:
            code_id = f"subqueue-action-primary-{idx}"
            command_html = '<pre id="' + code_id + '">' + html.escape(command) + '</pre><button type="button" onclick="copyText(\'' + code_id + '\')">复制首选命令</button>'
        cards.append(
            '<div class="queue-card ' + html.escape(machine) + '">'
            '<strong>' + group_label + ' / ' + label + '</strong> '
            '<span class="badge">' + status + '</span> <span class="badge">' + kind + '</span> <span class="badge">P' + priority + '</span>'
            '<div class="muted">key=' + key + '</div>'
            + ('<div class="muted">原因：' + reason + '</div>' if reason else '')
            + ('<div class="muted">' + hint + '</div>' if hint else '')
            + command_html
            + '</div>'
        )
    return '<div class="muted">' + html.escape(summary) + '</div>' + ''.join(cards)
def _external_reuse_status_panel_html(status: dict[str, Any]) -> str:
    rows = status.get("capabilities") if isinstance(status.get("capabilities"), list) else []
    if not rows:
        return '<div class="muted">尚未生成外部复用能力 run。先运行 external-capability-pack、video-rag-pack 或 run-artifact-registry。</div>'
    summary = (
        f"{int(status.get('capability_count') or len(rows))} 类能力，"
        f"ready {int(status.get('ready_count') or 0)}，"
        f"action {int(status.get('action_required_count') or 0)}，"
        f"missing {int(status.get('missing_count') or 0)}。"
    )
    cards = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        key = html.escape(str(row.get("key") or f"external-{idx}"))
        label = html.escape(str(row.get("label") or key))
        row_status = html.escape(str(row.get("status") or "missing"))
        sources = html.escape(", ".join(str(value) for value in row.get("source_projects") or []))
        desc = html.escape(str(row.get("description") or ""))
        queue_key = html.escape(str(row.get("queue_key") or ""), quote=True)
        counts = row.get("status_counts") if isinstance(row.get("status_counts"), dict) else {}
        count_text = html.escape(" / ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "no runs")
        artifacts = row.get("artifacts") if isinstance(row.get("artifacts"), list) else []
        artifact_buttons = []
        for artifact in artifacts[:4]:
            if not isinstance(artifact, dict):
                continue
            artifact_key = html.escape(str(artifact.get("key") or ""), quote=True)
            artifact_label = html.escape(str(artifact.get("label") or artifact.get("key") or "artifact"))
            exists = "" if artifact.get("exists") else "（未生成）"
            artifact_buttons.append(f'<button type="button" onclick="openArtifact(\'{artifact_key}\')">{artifact_label}{exists}</button>')
        retry_commands = [str(cmd) for cmd in (row.get("retry_commands") or []) if str(cmd).strip()]
        retry_html = ""
        if retry_commands:
            code_id = f"external-reuse-retry-{idx}"
            retry_html = '<pre id="' + code_id + '">' + html.escape(retry_commands[0]) + '</pre><button type="button" onclick="copyText(\'' + code_id + '\')">复制命令</button>'
        queue_button = f'<button type="button" onclick="selectQueue(\'{queue_key}\')">查看队列</button>' if queue_key else ""
        cards.append(
            '<div class="queue-card ' + row_status + '">'
            '<strong>' + label + '</strong> <span class="badge">' + row_status + '</span>'
            '<div class="muted">来源：' + sources + '</div>'
            '<div class="muted">' + desc + '</div>'
            '<div class="muted">runs=' + html.escape(str(row.get("run_count") or 0)) + ' action=' + html.escape(str(row.get("action_required") or 0)) + ' failed=' + html.escape(str(row.get("failed_count") or 0)) + '</div>'
            '<div class="muted">' + count_text + '</div>'
            '<div class="toolbar" style="margin-top:6px">' + queue_button + ''.join(artifact_buttons) + '</div>'
            + retry_html
            + '</div>'
        )
    return '<div class="muted">' + html.escape(summary) + '</div>' + ''.join(cards)


def _autopilot_panel_html(summary: dict[str, Any], root: Path) -> str:
    route = summary.get("execution_route") if isinstance(summary.get("execution_route"), dict) else {}
    work_item = summary.get("current_work_item") if isinstance(summary.get("current_work_item"), dict) else {}
    provider_wait = summary.get("provider_wait") if isinstance(summary.get("provider_wait"), dict) else {}
    local_provider = summary.get("local_vision_provider") if isinstance(summary.get("local_vision_provider"), dict) else {}
    provider = local_provider.get("provider") if isinstance(local_provider.get("provider"), dict) else {}
    paths = summary.get("paths") if isinstance(summary.get("paths"), dict) else {}
    links: list[str] = []
    for key, label in (
        ("job", "Job"),
        ("events", "Journal"),
        ("latest_raw_result", "Raw"),
        ("latest_validation", "验证"),
        ("run", "审计"),
        ("restore_plan", "Restore"),
    ):
        raw = str(paths.get(key) or "").strip()
        if raw:
            links.append(f'<a href="{html.escape(_relative_href(root, Path(raw).resolve()))}">{label}</a>')
    command = str(summary.get("next_command") or summary.get("start_command") or "")
    if provider_wait and summary.get("job_id"):
        command = f".\\scripts\\video-knowledge.ps1 autopilot-next {summary['job_id']} --retry-provider"
    gaps = summary.get("review_gaps") if isinstance(summary.get("review_gaps"), list) else []
    return (
        '<div class="artifact">'
        f'<strong>{html.escape(str(summary.get("state") or "not_started"))}</strong>'
        f'<div class="muted">runner {html.escape(str(summary.get("runner") or "-"))} · route {html.escape(str(route.get("resolved_route") or "pending"))} · gaps {len(gaps)}</div>'
        f'<div class="muted">task {html.escape(str(work_item.get("task_type") or "none"))}</div>'
        f'<div class="muted">local provider {html.escape(str(provider.get("model") or "unconfigured"))} · {html.escape(str(provider.get("base_url") or ""))} · source {html.escape(str(local_provider.get("provider_config_source") or ""))}</div>'
        f'<div class="muted">provider wait {html.escape(str(provider_wait.get("reason") or "none"))}</div>'
        f'<div>{" · ".join(links)}</div>'
        f'<code>{html.escape(command)}</code>'
        '<div class="muted">complete_draft 之后仍需人工验收；Autopilot 不写 accepted。</div>'
        '</div>'
    )


def _provider_status_panel_html(status: dict[str, Any]) -> str:
    smoke = status.get("vision_provider_smoke") if isinstance(status.get("vision_provider_smoke"), dict) else {}
    matrix = status.get("vision_provider_matrix") if isinstance(status.get("vision_provider_matrix"), dict) else {}
    local_vlm = status.get("local_vlm_serving_smoke") if isinstance(status.get("local_vlm_serving_smoke"), dict) else {}
    if not status.get("has_any_report"):
        commands = status.get("next_commands") if isinstance(status.get("next_commands"), list) else []
        command_html = "".join('<pre>' + html.escape(str(cmd)) + '</pre>' for cmd in commands[:3])
        return (
            '<div class="muted">尚未生成 provider / 本地 VLM smoke 报告。此面板只读取已有报告，不启动服务、不调用云。</div>'
            + command_html
        )
    smoke_cls = "closure-ok" if smoke.get("safe_to_execute") else "closure-alert" if smoke.get("exists") else ""
    matrix_cls = "closure-ok" if matrix.get("recommended_provider") else "closure-alert" if matrix.get("exists") else ""
    local_cls = "closure-ok" if local_vlm.get("status") == "executed_ready" else "closure-alert" if local_vlm.get("status") == "executed_failed" else ""
    provider_label = smoke.get("provider") or "unknown"
    recommended = matrix.get("recommended_provider") or "none"
    local_status = local_vlm.get("status") or "missing_report"
    capability_text = ", ".join(f"{k}:{v}" for k, v in sorted((local_vlm.get("capability_counts") or {}).items())) or "no capability matrix"
    error_text = smoke.get("error_class") or smoke.get("error_summary") or "no error summary"
    return f"""
      <div class=\"closure-grid\">
        <div class=\"closure-metric {smoke_cls}\"><span class=\"muted\">Provider smoke</span><strong>{html.escape(str(smoke.get('status') or 'missing'))}</strong></div>
        <div class=\"closure-metric {matrix_cls}\"><span class=\"muted\">推荐 provider</span><strong>{html.escape(str(recommended))}</strong></div>
        <div class=\"closure-metric {local_cls}\"><span class=\"muted\">本地 VLM</span><strong>{html.escape(str(local_status))}</strong></div>
        <div class=\"closure-metric\"><span class=\"muted\">边界</span><strong>read-only</strong></div>
      </div>
      <div class=\"muted\">provider={html.escape(str(provider_label))} · safe_to_execute={html.escape(str(bool(smoke.get('safe_to_execute'))).lower())} · matrix ready={int(matrix.get('ready_count') or 0)}/{int(matrix.get('provider_count') or 0)} · local capabilities={html.escape(capability_text)}</div>
      <div class=\"muted\">provider issue: {html.escape(str(error_text))}</div>
      <div class=\"muted\">此面板不启动 Qwen/InternVL 服务、不调用云 provider、不修改 timeline；只显示已有 smoke/matrix 产物。</div>
      <div class=\"toolbar\"><button onclick=\"openArtifact('vision_provider_smoke')\">Provider Smoke</button><button onclick=\"openArtifact('vision_provider_matrix')\">Provider Matrix</button><button onclick=\"openArtifact('local_vlm_serving_smoke')\">本地 VLM Smoke</button></div>
    """
def _queue_panel_html(queue: dict[str, Any]) -> str:
    groups = queue.get("groups") if isinstance(queue.get("groups"), list) else []
    run_count = int(queue.get("run_count") or 0)
    action_required = int(queue.get("action_required_count") or 0)
    if not groups:
        return '<div class="muted">尚未生成 run registry。先运行 task console 或 run-artifact-registry 后，这里会显示批次状态和重试命令。</div>'
    cards = [_queue_card_html(group, idx) for idx, group in enumerate(groups, start=1)]
    summary = f"{run_count} 个 run，{action_required} 个队列需要执行、重试或复核。"
    return '<div class="muted">' + html.escape(summary) + '</div>' + "\n".join(cards)


def _queue_card_html(group: dict[str, Any], idx: int) -> str:
    key = html.escape(str(group.get("key") or "queue"))
    label = html.escape(str(group.get("label") or group.get("key") or "Queue"))
    status = html.escape(str(group.get("status") or "empty"))
    desc = html.escape(str(group.get("description") or ""))
    run_count = html.escape(str(group.get("run_count") or 0))
    failed_count = html.escape(str(group.get("failed_count") or 0))
    action_required = html.escape(str(group.get("action_required") or 0))
    counts = group.get("status_counts") if isinstance(group.get("status_counts"), dict) else {}
    count_text = html.escape(" / ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "no runs")
    runs = group.get("runs") if isinstance(group.get("runs"), list) else []
    run_items = []
    for run in runs[:3]:
        if not isinstance(run, dict):
            continue
        title = html.escape(str(run.get("title") or run.get("run_id") or "run"))
        run_status = html.escape(str(run.get("status") or "unknown"))
        run_items.append('<li><span class="badge">' + run_status + '</span> ' + title + '</li>')
    failed_items = group.get("failed_items_preview") if isinstance(group.get("failed_items_preview"), list) else []
    failed = "; ".join(_queue_failed_label(item) for item in failed_items[:4] if isinstance(item, dict))
    retry_commands = [str(cmd) for cmd in (group.get("retry_commands") or []) if str(cmd).strip()]
    retry_html = ""
    if retry_commands:
        code_id = f"queue-retry-{idx}"
        retry_html = '<pre id="' + code_id + '">' + html.escape(retry_commands[0]) + '</pre><button type="button" onclick="copyText(\'' + code_id + '\')">复制重试命令</button>'
    return (
        '<div class="queue-card ' + status + '" data-queue-key="' + key + '" onclick="selectQueue(\'' + key + '\')">'
        '<strong>' + label + '</strong> <span class="badge">' + status + '</span>'
        '<div class="muted">' + desc + '</div>'
        '<div class="muted">runs: ' + run_count + ' | action: ' + action_required + ' | failed: ' + failed_count + '</div>'
        '<div class="muted">' + count_text + '</div>'
        + ('<ul>' + ''.join(run_items) + '</ul>' if run_items else '')
        + ('<div class="muted">失败项：' + html.escape(failed) + '</div>' if failed else '')
        + retry_html
        + '</div>'
    )


def _queue_failed_label(item: dict[str, Any]) -> str:
    parts = [str(item.get(key) or "").strip() for key in ("index", "reason", "detail")]
    return " / ".join(value for value in parts if value)

def _artifact_card_html(card: dict[str, Any]) -> str:
    klass = "artifact" if card.get("exists") else "artifact missing"
    label = html.escape(str(card.get("label") or card.get("key") or "artifact"))
    href = html.escape(str(card.get("href") or ""))
    status = str(card.get("status") or ("存在" if card.get("exists") else "未生成"))
    key = html.escape(str(card.get("key") or ""))
    return f'<div class="{klass}"><strong>{label}</strong><div class="muted">{html.escape(status)}</div><a href="{href}">{href}</a><div><button onclick="openArtifact(\'{key}\')">打开到右侧</button></div></div>'


def _bundle_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _relative_href(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _seconds(value: Any, *, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return max(0.0, float(value))
    except Exception:
        return default


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()
