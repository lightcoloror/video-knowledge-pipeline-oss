from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .powershell import quote_powershell_argument as _ps_quote
from .bundle_readiness import build_bundle_readiness
from .config import ebook_pipeline_profile, runtime_config_manifest
from .lecture_package import render_lecture_package_markdown, render_lecture_review_html
from .knowledge_coverage import audit_knowledge_coverage, build_knowledge_coverage, render_knowledge_coverage_markdown
from .lecture_outline import generate_lecture_outline, render_lecture_outline_markdown
from .lecture_review_queue import (
    generate_lecture_review_queue,
    render_lecture_review_anki_csv,
    render_lecture_review_queue_markdown,
    render_lecture_review_tasks_markdown,
)
from .lecture_search import generate_lecture_search_index, render_lecture_search_index_markdown
from .lecture_study_index import generate_lecture_study_cards, generate_lecture_study_index, render_lecture_study_cards_markdown, render_lecture_study_index_markdown
from .models import now_iso
from .media_applicability import project_timeline_for_media, resolve_media_applicability
from .repair_status import build_repair_status
from .review_export_context import CONTEXT_FILENAME, configured_identity, normalize_project_identity, resolve_review_export_context
from .source_artifacts import build_source_artifact_index, render_source_artifact_index_markdown
from .storage import ensure_project_dirs, read_json, write_json
from .task_console import export_task_console
from .timeline_alignment_audit import timeline_alignment_audit
from .transcript_sidecar import ensure_review_transcript_sidecar
from .video_frame_router import write_video_frame_router_input_template
from .multimodal_frame_analyzer import write_multimodal_frame_input_template
from .temporal_visual_analyzer import write_temporal_visual_input_template
from .visual_structure import _tool_statuses, _visual_structure_candidates, write_visual_structure_input_template


def _bind_canonical_transcript_manifest(
    bundle_dir: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Bind review/export transcript pointers to the bundle canonical source."""

    canonical_json = bundle_dir / "source-arbitrated-transcript.json"
    if not canonical_json.is_file():
        return manifest
    manifest["source_arbitrated_transcript_json"] = canonical_json.name
    manifest["corrected_transcript_json"] = canonical_json.name
    manifest["corrected_transcript_source"] = "source_arbitrated_transcript"
    for suffix, source_key, corrected_key in (
        ("srt", "source_arbitrated_transcript_srt", "corrected_transcript_srt"),
        ("md", "source_arbitrated_transcript_markdown", "corrected_transcript_markdown"),
    ):
        candidate = bundle_dir / f"source-arbitrated-transcript.{suffix}"
        if candidate.is_file():
            manifest[source_key] = candidate.name
            manifest[corrected_key] = candidate.name
    base = bundle_dir / "asr-evidence-adjudicated-base.json"
    if base.is_file():
        manifest["transcript_semantic_correction_base_json"] = base.name
    adjudication = bundle_dir / "asr-evidence-autoadjudication.json"
    if adjudication.is_file():
        manifest["asr_evidence_autoadjudication"] = adjudication.name
    manifest["review_transcript_canonical"] = {
        "schema": "video_knowledge_pipeline.review_transcript_canonical.v1",
        "path": canonical_json.name,
        "sha256": hashlib.sha256(canonical_json.read_bytes()).hexdigest(),
        "source": "source_arbitrated_transcript",
        "raw_timeline_transcript_is_audit_only": True,
    }
    return manifest


def export_webui_bundle(root: str | Path, output_dir: str | Path | None = None, *, target: str = "bilinote") -> dict[str, Any]:
    """Export a stable handoff bundle for BiliNote-like WebUI integration."""
    paths = ensure_project_dirs(root)
    package_path = paths["lecture_packages"] / "lecture-package.json"
    if not package_path.exists():
        raise FileNotFoundError(f"lecture package not found: {package_path}")
    package = read_json(package_path)
    if not isinstance(package, dict):
        raise ValueError("lecture package must be a JSON object")

    bundle_dir = Path(output_dir) if output_dir else paths["lecture_packages"] / "webui-bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    output_identity = configured_identity(bundle_dir)
    export_context = resolve_review_export_context(
        package,
        output_identity if output_identity is not None else resolve_review_export_context(package, root=paths["root"]),
        root=bundle_dir,
    )
    if export_context is not None:
        package = {**package, "project_identity": normalize_project_identity(export_context, allow_incomplete=True)}
        # Carry the resolved context into the bundle so the unmodified public
        # task-console naming call (one-argument review_export_script(console))
        # can resolve the same project identity without guessing a title.
        write_json(bundle_dir / CONTEXT_FILENAME, export_context)
    assets_dir = bundle_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    copied_assets = _copy_frame_assets(package, assets_dir)
    bundled_package = _package_with_webui_assets(package, copied_assets)
    bundled_package["bundle_dir"] = str(bundle_dir.resolve())
    timeline = _webui_timeline(package, copied_assets)
    frame_recapture = _frame_recapture_plan(package, bundle_dir)
    time_gap_recapture = _time_gap_recapture_plan(package, bundle_dir)
    visual_structure = _visual_structure_plan(bundle_dir, timeline)
    video_frame_router = _video_frame_router_plan(bundle_dir, timeline)
    multimodal_frame_analysis = _multimodal_frame_analysis_plan(bundle_dir, timeline)
    temporal_visual_analysis = _temporal_visual_analysis_plan(bundle_dir, timeline)
    outline = generate_lecture_outline(bundled_package)
    search_index = generate_lecture_search_index(bundled_package)
    study_index = generate_lecture_study_index(bundled_package)
    study_cards = generate_lecture_study_cards(study_index)
    study_review_queue = generate_lecture_review_queue(study_cards)
    source_artifact_index = build_source_artifact_index(
        {
            **bundled_package,
            "bundle_dir": str(bundle_dir),
            "source_package": str(package_path),
            "timeline_path": str(bundle_dir / "timeline.json"),
            "asset_manifest": str(assets_dir / "asset-manifest.json"),
        }
    )
    review_notes_path = bundle_dir / "review-notes.json"
    review_notes_template_path = bundle_dir / "review-notes.template.json"
    review_fill_guide_path = bundle_dir / "review-fill-guide.md"
    refresh_args_path = bundle_dir / "mcp-refresh-lecture-review.args.json"
    frame_recapture_args_path = bundle_dir / "mcp-run-frame-recapture.args.json"
    ocr_backfill_args_path = bundle_dir / "mcp-run-ocr-backfill.args.json"
    screen_text_recovery_args_path = bundle_dir / "mcp-run-screen-text-recovery.args.json"
    visual_structure_args_path = bundle_dir / "mcp-run-visual-structure.args.json"
    video_frame_router_args_path = bundle_dir / "mcp-run-video-frame-router.args.json"
    multimodal_frame_analysis_args_path = bundle_dir / "mcp-run-multimodal-frame-analysis.args.json"
    temporal_frame_groups_args_path = bundle_dir / "mcp-run-temporal-frame-groups.args.json"
    temporal_visual_analysis_args_path = bundle_dir / "mcp-run-temporal-visual-analysis.args.json"
    task_console_args_path = bundle_dir / "mcp-export-task-console.args.json"
    asset_repair_args_path = bundle_dir / "mcp-repair-bundle-assets.args.json"
    repair_status_args_path = bundle_dir / "mcp-refresh-repair-status.args.json"
    readiness_args_path = bundle_dir / "mcp-audit-bundle-readiness.args.json"
    next_action_args_path = bundle_dir / "mcp-bundle-next-action.args.json"
    source_artifacts_args_path = bundle_dir / "mcp-bundle-source-artifacts.args.json"
    knowledge_coverage_args_path = bundle_dir / "mcp-audit-knowledge-coverage.args.json"
    advance_args_path = bundle_dir / "mcp-bundle-advance.args.json"
    advance_log_args_path = bundle_dir / "mcp-bundle-advance-log.args.json"
    advance_queue_args_path = bundle_dir / "mcp-bundle-advance-queue.args.json"
    review_session_args_path = bundle_dir / "mcp-prepare-review-session.args.json"
    apply_review_notes_args_path = bundle_dir / "mcp-apply-review-notes.args.json"
    status_report_args_path = bundle_dir / "mcp-bundle-status-report.args.json"
    acceptance_check_args_path = bundle_dir / "mcp-acceptance-check.args.json"
    controlled_execution_check_args_path = bundle_dir / "mcp-controlled-execution-check.args.json"
    controlled_execution_smoke_args_path = bundle_dir / "mcp-controlled-execution-smoke.args.json"
    export_knowledge_note_args_path = bundle_dir / "mcp-export-knowledge-note.args.json"
    vision_acceptance_args_path = bundle_dir / "mcp-vision-acceptance-plan.args.json"
    refresh_command = _local_cli_command(
        [
            "refresh-lecture-review",
            str(root),
            str(review_notes_path),
            "--webui-output-dir",
            str(bundle_dir),
        ]
    )
    refresh_args = {
        "root": str(root),
        "review_json": str(review_notes_path),
        "webui_output_dir": str(bundle_dir),
        "target": target,
        "allow_blocked_export": False,
    }
    frame_recapture_args = {
        "bundle_dir": str(bundle_dir),
        "execute": False,
        "timeout_seconds": 30,
    }
    ocr_backfill_args = {
        "bundle_dir": str(bundle_dir),
        "execute": False,
        "language": "chi_sim",
    }
    screen_text_recovery_args = {
        "bundle_dir": str(bundle_dir),
        "execute_crops": False,
        "execute_ocr": False,
        "language": "chi_sim+eng",
        "limit": 0,
        "write": True,
    }
    ebook_profile = ebook_pipeline_profile()
    visual_structure_args = {
        "bundle_dir": str(bundle_dir),
        "execute_ebook_pipeline": bool(ebook_profile.get("execute_default")),
        "include_routes": list(ebook_profile.get("include_routes") or ["document_visual", "mixed"]),
        "timeout_seconds": int(ebook_profile.get("timeout_seconds") or 300),
        "indexes": [],
        "limit": int(ebook_profile.get("limit") or 0),
    }
    video_frame_router_args = {
        "bundle_dir": str(bundle_dir),
        "write": True,
    }
    multimodal_frame_analysis_args = {
        "bundle_dir": str(bundle_dir),
        "execute": False,
        "limit": 19,
    }
    temporal_frame_groups_args = {
        "bundle_dir": str(bundle_dir),
        "execute": False,
        "frame_count": 8,
        "window_seconds": 4.0,
        "include_routes": ["temporal_sequence", "mixed"],
        "timeout_seconds": 60,
    }
    temporal_visual_analysis_args = {
        "bundle_dir": str(bundle_dir),
        "execute": False,
        "frame_count": 8,
        "limit": 3,
    }
    asset_repair_args = {
        "bundle_dir": str(bundle_dir),
    }
    repair_status_args = {
        "bundle_dir": str(bundle_dir),
    }
    readiness_args = {
        "bundle_dir": str(bundle_dir),
        "write": True,
    }
    manifest = {
        "schema": "lecture_webui_bundle.v1",
        **({"project_identity": package["project_identity"]} if package.get("project_identity") else {}),
        "target": target,
        "title": package.get("title", "Lecture Package"),
        "created_at": now_iso(),
        "runtime_config": runtime_config_manifest(),
        "source_package": str(package_path),
        "entry_note": "note.md",
        "outline": "outline.md",
        "outline_json": "outline.json",
        "search_index": "search-index.md",
        "search_index_json": "search-index.json",
        "study_index": "study-index.md",
        "study_index_json": "study-index.json",
        "study_cards": "study-cards.md",
        "study_cards_json": "study-cards.json",
        "study_review_queue": "study-review-queue.md",
        "study_review_queue_json": "study-review-queue.json",
        "study_review_tasks": "study-review-tasks.md",
        "study_review_anki": "study-review-anki.csv",
        "source_artifacts": "source-artifacts.md",
        "source_artifacts_json": "source-artifacts.json",
        "knowledge_coverage_markdown": "knowledge-coverage.md",
        "knowledge_coverage_json": "knowledge-coverage.json",
        "review_html": "review.html",
        "task_console": "task-console.html",
        "task_console_json": "task-console.json",
        "review_notes": "review-notes.json",
        "mcp_refresh_args": "mcp-refresh-lecture-review.args.json",
        "mcp_frame_recapture_args": "mcp-run-frame-recapture.args.json",
        "mcp_ocr_backfill_args": "mcp-run-ocr-backfill.args.json",
        "mcp_screen_text_recovery_args": "mcp-run-screen-text-recovery.args.json",
        "mcp_visual_structure_args": "mcp-run-visual-structure.args.json",
        "mcp_video_frame_router_args": "mcp-run-video-frame-router.args.json",
        "mcp_multimodal_frame_analysis_args": "mcp-run-multimodal-frame-analysis.args.json",
        "mcp_temporal_frame_groups_args": "mcp-run-temporal-frame-groups.args.json",
        "mcp_temporal_visual_analysis_args": "mcp-run-temporal-visual-analysis.args.json",
        "mcp_export_task_console_args": "mcp-export-task-console.args.json",
        "mcp_asset_repair_args": "mcp-repair-bundle-assets.args.json",
        "mcp_repair_status_args": "mcp-refresh-repair-status.args.json",
        "mcp_readiness_args": "mcp-audit-bundle-readiness.args.json",
        "mcp_next_action_args": "mcp-bundle-next-action.args.json",
        "mcp_source_artifacts_args": "mcp-bundle-source-artifacts.args.json",
        "mcp_knowledge_coverage_args": "mcp-audit-knowledge-coverage.args.json",
        "mcp_advance_args": "mcp-bundle-advance.args.json",
        "mcp_advance_log_args": "mcp-bundle-advance-log.args.json",
        "mcp_advance_queue_args": "mcp-bundle-advance-queue.args.json",
        "mcp_review_session_args": "mcp-prepare-review-session.args.json",
        "mcp_apply_review_notes_args": "mcp-apply-review-notes.args.json",
        "bundle_status_report": "bundle-status.md",
        "bundle_status_report_json": "bundle-status.json",
        "mcp_status_report_args": "mcp-bundle-status-report.args.json",
        "acceptance_check": "acceptance-check.md",
        "acceptance_check_json": "acceptance-check.json",
        "mcp_acceptance_check_args": "mcp-acceptance-check.args.json",
        "review_session": "review-session.md",
        "review_session_json": "review-session.json",
        "review_notes_template": "review-notes.template.json",
        "review_fill_guide": "review-fill-guide.md",
        "controlled_execution_check": "controlled-execution-check.md",
        "controlled_execution_check_json": "controlled-execution-check.json",
        "mcp_controlled_execution_check_args": "mcp-controlled-execution-check.args.json",
        "controlled_execution_smoke": "controlled-execution-smoke.md",
        "controlled_execution_smoke_json": "controlled-execution-smoke.json",
        "mcp_controlled_execution_smoke_args": "mcp-controlled-execution-smoke.args.json",
        "knowledge_note_markdown": "exports/knowledge-note.md",
        "knowledge_note_smart_summary_markdown": "exports/smart-summary.md",
        "knowledge_note_full_body_markdown": "exports/full-body.md",
        "knowledge_note_smart_summary_codex_prompt_markdown": "exports/smart-summary-codex-prompt.md",
        "knowledge_note_transcript_markdown": "exports/full-transcript.md",
        "knowledge_note_extraction_audit_markdown": "exports/extraction-audit.md",
        "mcp_export_knowledge_note_args": "mcp-export-knowledge-note.args.json",
        "vision_acceptance_plan": "vision-acceptance-plan.md",
        "vision_acceptance_plan_json": "vision-acceptance-plan.json",
        "mcp_vision_acceptance_plan_args": "mcp-vision-acceptance-plan.args.json",
        "timeline_json": "timeline.json",
        "assets_dir": "assets",
        "asset_manifest": "assets/asset-manifest.json",
        "coverage": package.get("coverage", {}),
        "sources": package.get("sources", []),
        "assets": list(copied_assets.values()),
        "frame_recapture": frame_recapture,
        "time_gap_recapture": time_gap_recapture,
        "visual_structure": visual_structure,
        "video_frame_router": video_frame_router,
        "multimodal_frame_analysis": multimodal_frame_analysis,
        "temporal_visual_analysis": temporal_visual_analysis,
        "post_review": {
            "review_notes_path": str(review_notes_path),
            "refresh_command": refresh_command,
            "mcp_tool": "refresh_lecture_review_outputs",
            "mcp_args_path": str(refresh_args_path),
            "apply_mcp_tool": "apply_review_notes",
            "apply_mcp_args_path": str(apply_review_notes_args_path),
        },
        "repair_tools": {
            "frame_recapture": {
                "mcp_tool": "run_frame_recapture_plan",
                "mcp_args_path": str(frame_recapture_args_path),
            },
            "ocr_backfill": {
                "mcp_tool": "run_ocr_backfill",
                "mcp_args_path": str(ocr_backfill_args_path),
            },
            "screen_text_recovery": {
                "mcp_tool": "run_screen_text_recovery",
                "mcp_args_path": str(screen_text_recovery_args_path),
            },
            "visual_structure": {
                "mcp_tool": "run_visual_structure_plan",
                "mcp_args_path": str(visual_structure_args_path),
            },
            "video_frame_router": {
                "mcp_tool": "run_video_frame_router",
                "mcp_args_path": str(video_frame_router_args_path),
            },
            "multimodal_frame_analysis": {
                "mcp_tool": "run_multimodal_frame_analysis",
                "mcp_args_path": str(multimodal_frame_analysis_args_path),
            },
            "temporal_visual_analysis": {
                "mcp_tool": "run_temporal_visual_analysis",
                "mcp_args_path": str(temporal_visual_analysis_args_path),
            },
            "temporal_frame_groups": {
                "mcp_tool": "run_temporal_frame_groups",
                "mcp_args_path": str(temporal_frame_groups_args_path),
            },
            "bundle_assets": {
                "mcp_tool": "repair_bundle_assets",
                "mcp_args_path": str(asset_repair_args_path),
            },
        },
        "integration_notes": [
            "BiliNote already renders Markdown notes; load note.md as the note body.",
            "Use timeline.json for a richer timeline/review panel.",
            "Use outline.md/json as a non-summary navigation layer over the full timeline.",
            "Use search-index.md/json for deterministic local keyword lookup by humans or agents.",
            "Use study-index.md/json as a searchable no-summary learning entry point grouped by concepts, procedures, examples, formulas, tables, code, visual assets to keep, and review queue.",
            "Use study-cards.md/json as evidence-preserving draft cards for Obsidian-side review and card splitting.",
            "Use study-review-queue.md/json as the actionable spaced-review queue derived from human card fields.",
            "Use study-review-tasks.md for Obsidian Tasks import, or study-review-anki.csv for Anki import.",
            "Use source-artifacts.md/json to trace back to original vidclaude/peepshow/vidwise outputs before trusting any derived note.",
            "Use knowledge-coverage.md/json to audit speech, ebook_markdown_pipeline-backed screen text/layout, visual frames, structured visual material, time-axis gaps, and original evidence traceability.",
            "After human review, save review-notes.json and call apply_review_notes with mcp-apply-review-notes.args.json to update timeline/coverage/readiness, or run post_review.refresh_command for the older full refresh path.",
            "For agent-assisted repair, call repair_tools frame_recapture, visual_structure, video_frame_router, multimodal_frame_analysis, temporal_frame_groups, temporal_visual_analysis, or ocr_backfill fallback with the matching MCP args JSON generated beside this manifest.",
            "If review_readiness reports asset_gap, call repair_bundle_assets with mcp-repair-bundle-assets.args.json to recopy key frames from recorded source paths.",
            "After manual repair imports or direct manifest edits, call refresh_bundle_repair_status with mcp-refresh-repair-status.args.json.",
            "Before final export or agent handoff, call audit_bundle_readiness with mcp-audit-bundle-readiness.args.json.",
            "For a single agent-safe next step, call bundle_next_action with mcp-bundle-next-action.args.json.",
            "Use bundle_source_artifacts with mcp-bundle-source-artifacts.args.json to inspect the original extractor artifact index.",
            "Use audit_knowledge_coverage with mcp-audit-knowledge-coverage.args.json to inspect no-loss coverage across knowledge channels.",
            "To let an agent advance one preview-safe bundle step, call bundle_advance with mcp-bundle-advance.args.json.",
            "Read bundle_advance history with bundle_advance_log and mcp-bundle-advance-log.args.json.",
            "Use bundle_advance_queue for bounded, stalled-aware multi-step agent progress.",
            "Use prepare_review_session to create a single human/agent review entry point with review.html, review-notes.json, blockers, and post-review commands.",
            "Use bundle_status_report with mcp-bundle-status-report.args.json as the compact status dashboard for humans, UI, and agents.",
            "Use acceptance_check with mcp-acceptance-check.args.json as the final truth source for provider, review, coverage, export freshness, and next action.",
            "Use task-console.html as the lightweight human control panel for choosing ASR/OCR/triage/vision/export steps while keeping review.html as the detailed review workspace.",
            "Use export_knowledge_note with mcp-export-knowledge-note.args.json to create an Obsidian-friendly human-readable knowledge note from transcript, OCR, structured visuals, single-frame understanding, and temporal understanding.",
            "Use vision_acceptance_plan with mcp-vision-acceptance-plan.args.json before execute=true multimodal calls; it checks provider/key readiness and keeps the first real run to the 19 semantic + 3 temporal acceptance batch.",
            "Use controlled_execution_smoke with mcp-controlled-execution-smoke.args.json to exercise preflight, confirmed write, audit, and optional restore with a local fixture provider.",
            "Open review.html for the current standalone review UI until native UI integration exists.",
            "Use frame_recapture.items when structured visual content needs a replacement key frame.",
            "Use time_gap_recapture.items to inspect timeline blank ranges before trusting coverage.",
            "Use video_frame_router first to decide whether a frame is document_visual, semantic_frame, temporal_sequence, mixed, or unknown.",
            "Use multimodal_frame_analysis.items when screenshots contain objects, actions, software state, spatial relations, instructor pointing, or other non-text visual information.",
            "Use temporal_frame_groups before temporal_visual_analysis when a route needs ordered 5-12 frame evidence from the same time window.",
            "Use temporal_visual_analysis.items for mouse movement, software operations, experiments, demos, or frame-to-frame state changes.",
            "Use visual_structure.items to extract screenshot text/layout with ebook_markdown_pipeline. For document_visual this is the primary branch; for semantic_frame/temporal_sequence it runs alongside multimodal visual understanding and then integrates the results. Use ocr_backfill only as a manual OCR/Tesseract/CaptiOCR fallback.",
        ],
    }
    _bind_canonical_transcript_manifest(bundle_dir, manifest)
    ensure_review_transcript_sidecar(bundle_dir, manifest, timeline, title=str(package.get("title", "Lecture Package")), project_root=root, write=True)
    applicability = resolve_media_applicability(bundle_dir, manifest, timeline, write=True)
    manifest["media_applicability"] = applicability
    timeline = project_timeline_for_media(timeline, applicability)
    bundled_package["timeline"] = project_timeline_for_media(bundled_package.get("timeline") or [], applicability)
    manifest["repair_status"] = build_repair_status(manifest, timeline, bundle_dir=bundle_dir)
    manifest["review_readiness"] = build_bundle_readiness(manifest, timeline, bundle_dir=bundle_dir)
    manifest["knowledge_coverage"] = build_knowledge_coverage(manifest, timeline, bundle_dir=bundle_dir)
    _preserve_controlled_execution_entrypoints(bundle_dir, manifest)
    write_json(bundle_dir / "manifest.json", manifest)
    write_json(refresh_args_path, refresh_args)
    write_json(frame_recapture_args_path, frame_recapture_args)
    write_json(ocr_backfill_args_path, ocr_backfill_args)
    write_json(screen_text_recovery_args_path, screen_text_recovery_args)
    write_json(visual_structure_args_path, visual_structure_args)
    write_json(video_frame_router_args_path, video_frame_router_args)
    write_json(multimodal_frame_analysis_args_path, multimodal_frame_analysis_args)
    write_json(temporal_frame_groups_args_path, temporal_frame_groups_args)
    write_json(temporal_visual_analysis_args_path, temporal_visual_analysis_args)
    write_json(task_console_args_path, {"bundle_dir": str(bundle_dir), "write": True, "refresh": False})
    write_json(asset_repair_args_path, asset_repair_args)
    write_json(repair_status_args_path, repair_status_args)
    write_json(readiness_args_path, readiness_args)
    write_json(next_action_args_path, {"bundle_dir": str(bundle_dir), "refresh": True})
    write_json(source_artifacts_args_path, {"bundle_dir": str(bundle_dir), "refresh": False, "write": False})
    write_json(knowledge_coverage_args_path, {"bundle_dir": str(bundle_dir), "write": True})
    write_json(review_session_args_path, {"bundle_dir": str(bundle_dir), "refresh": True, "limit": 30, "offset": 0, "reason": ""})
    write_json(apply_review_notes_args_path, {"bundle_dir": str(bundle_dir), "review_json": str(review_notes_path), "write": True})
    review_template = _review_notes_template_from_timeline(timeline)
    write_json(review_notes_template_path, review_template)
    if not review_notes_path.exists():
        write_json(review_notes_path, {"schema": "lecture_review_notes.v1", "created_at": now_iso(), "reviews": []})
    review_fill_guide_path.write_text(
        _render_review_fill_guide(
            title=str(package.get("title", "Lecture Package")),
            bundle_dir=bundle_dir,
            review_html_path=bundle_dir / "review.html",
            review_notes_path=review_notes_path,
            review_notes_template_path=review_notes_template_path,
            apply_review_notes_args_path=apply_review_notes_args_path,
            review_count=len(review_template.get("reviews", [])),
        ),
        encoding="utf-8",
    )
    write_json(status_report_args_path, {"bundle_dir": str(bundle_dir), "refresh": True, "write": True})
    write_json(acceptance_check_args_path, {"bundle_dir": str(bundle_dir), "refresh": True, "write": True})
    write_json(controlled_execution_check_args_path, {"bundle_dir": str(bundle_dir), "refresh": False, "write": True})
    write_json(
        controlled_execution_smoke_args_path,
        {
            "bundle_dir": str(bundle_dir),
            "execute": False,
            "restore_after": False,
            "provider_config": {"provider": "fixture"},
            "kind": "auto",
            "frame_count": 8,
            "write": True,
        },
    )
    write_json(
        export_knowledge_note_args_path,
        {
            "bundle_dir": str(bundle_dir),
            "output_dir": str(bundle_dir / "exports"),
            "title": str(package.get("title", "Lecture Package")),
            "include_timeline": True,
            "include_full_transcript": True,
            "write": True,
        },
    )
    write_json(
        vision_acceptance_args_path,
        {
            "bundle_dir": str(bundle_dir),
            "semantic_limit": 19,
            "temporal_limit": 3,
            "frame_count": 8,
            "write": True,
        },
    )
    write_json(
        advance_args_path,
        {
            "bundle_dir": str(bundle_dir),
            "execute": False,
            "refresh_outputs": False,
            "folder": "00_Inbox/AI/课程视频知识包",
            "timeout_seconds": 30,
            "ocr_language": "chi_sim",
        },
    )
    write_json(advance_log_args_path, {"bundle_dir": str(bundle_dir)})
    write_json(
        advance_queue_args_path,
        {
            "bundle_dir": str(bundle_dir),
            "max_steps": 4,
            "execute": False,
            "refresh_outputs": False,
            "folder": "00_Inbox/AI/课程视频知识包",
            "timeout_seconds": 30,
            "ocr_language": "chi_sim",
        },
    )
    asset_manifest = _asset_manifest(copied_assets)
    write_json(assets_dir / "asset-manifest.json", asset_manifest)
    write_json(bundle_dir / "timeline.json", timeline)
    timeline_alignment = _timeline_alignment_for_review(bundle_dir, write=True)
    if timeline_alignment.get("manifest_updated"):
        refreshed_manifest = read_json(bundle_dir / "manifest.json")
        if isinstance(refreshed_manifest, dict):
            manifest = refreshed_manifest
    # The transcript sidecar is bound before the alignment audit.  Rebuild
    # coverage after the audit so a newly arrived canonical transcript is not
    # hidden behind the previous zero-cue receipt.
    audit_knowledge_coverage(bundle_dir, write=True)
    refreshed_manifest = read_json(bundle_dir / "manifest.json")
    if isinstance(refreshed_manifest, dict):
        manifest = refreshed_manifest
    write_json(bundle_dir / "outline.json", outline)
    write_json(bundle_dir / "search-index.json", search_index)
    write_json(bundle_dir / "study-index.json", study_index)
    write_json(bundle_dir / "study-cards.json", study_cards)
    write_json(bundle_dir / "study-review-queue.json", study_review_queue)
    write_json(bundle_dir / "source-artifacts.json", source_artifact_index)
    write_json(bundle_dir / "knowledge-coverage.json", manifest["knowledge_coverage"])
    (bundle_dir / "note.md").write_text(render_lecture_package_markdown(bundled_package), encoding="utf-8")
    (bundle_dir / "outline.md").write_text(render_lecture_outline_markdown(outline), encoding="utf-8")
    (bundle_dir / "search-index.md").write_text(render_lecture_search_index_markdown(search_index), encoding="utf-8")
    (bundle_dir / "study-index.md").write_text(render_lecture_study_index_markdown(study_index), encoding="utf-8")
    (bundle_dir / "study-cards.md").write_text(render_lecture_study_cards_markdown(study_cards), encoding="utf-8")
    (bundle_dir / "study-review-queue.md").write_text(render_lecture_review_queue_markdown(study_review_queue), encoding="utf-8")
    (bundle_dir / "study-review-tasks.md").write_text(render_lecture_review_tasks_markdown(study_review_queue), encoding="utf-8")
    (bundle_dir / "study-review-anki.csv").write_text(render_lecture_review_anki_csv(study_review_queue), encoding="utf-8-sig")
    (bundle_dir / "source-artifacts.md").write_text(render_source_artifact_index_markdown(source_artifact_index), encoding="utf-8")
    (bundle_dir / "knowledge-coverage.md").write_text(render_knowledge_coverage_markdown(manifest["knowledge_coverage"]), encoding="utf-8")
    html_package = {**bundled_package, "timeline": _annotate_timeline_alignment(timeline, timeline_alignment), "review_artifacts": _review_artifacts_for_render(bundle_dir, manifest)}
    (bundle_dir / "review.html").write_text(_media_review_html(html_package, applicability), encoding="utf-8")
    (bundle_dir / "README.md").write_text(_bundle_readme(manifest), encoding="utf-8")
    task_console = export_task_console(bundle_dir, write=True, refresh=False)
    return {
        "bundle_dir": str(bundle_dir),
        "manifest_path": str(bundle_dir / "manifest.json"),
        "note_path": str(bundle_dir / "note.md"),
        "outline_path": str(bundle_dir / "outline.json"),
        "outline_markdown_path": str(bundle_dir / "outline.md"),
        "search_index_path": str(bundle_dir / "search-index.json"),
        "search_index_markdown_path": str(bundle_dir / "search-index.md"),
        "study_index_path": str(bundle_dir / "study-index.json"),
        "study_index_markdown_path": str(bundle_dir / "study-index.md"),
        "study_cards_path": str(bundle_dir / "study-cards.json"),
        "study_cards_markdown_path": str(bundle_dir / "study-cards.md"),
        "study_review_queue_path": str(bundle_dir / "study-review-queue.json"),
        "study_review_queue_markdown_path": str(bundle_dir / "study-review-queue.md"),
        "study_review_tasks_markdown_path": str(bundle_dir / "study-review-tasks.md"),
        "study_review_anki_csv_path": str(bundle_dir / "study-review-anki.csv"),
        "source_artifacts_path": str(bundle_dir / "source-artifacts.json"),
        "source_artifacts_markdown_path": str(bundle_dir / "source-artifacts.md"),
        "knowledge_coverage_path": str(bundle_dir / "knowledge-coverage.json"),
        "knowledge_coverage_markdown_path": str(bundle_dir / "knowledge-coverage.md"),
        "timeline_path": str(bundle_dir / "timeline.json"),
        "review_html_path": str(bundle_dir / "review.html"),
        "task_console_html_path": str(bundle_dir / "task-console.html"),
        "task_console_json_path": str(bundle_dir / "task-console.json"),
        "mcp_export_task_console_args_path": str(task_console_args_path),
        "mcp_refresh_args_path": str(refresh_args_path),
        "mcp_frame_recapture_args_path": str(frame_recapture_args_path),
        "mcp_ocr_backfill_args_path": str(ocr_backfill_args_path),
        "mcp_visual_structure_args_path": str(visual_structure_args_path),
        "mcp_video_frame_router_args_path": str(video_frame_router_args_path),
        "mcp_multimodal_frame_analysis_args_path": str(multimodal_frame_analysis_args_path),
        "mcp_temporal_visual_analysis_args_path": str(temporal_visual_analysis_args_path),
        "mcp_asset_repair_args_path": str(asset_repair_args_path),
        "mcp_repair_status_args_path": str(repair_status_args_path),
        "mcp_readiness_args_path": str(readiness_args_path),
        "mcp_next_action_args_path": str(next_action_args_path),
        "mcp_source_artifacts_args_path": str(source_artifacts_args_path),
        "mcp_knowledge_coverage_args_path": str(knowledge_coverage_args_path),
        "mcp_advance_args_path": str(advance_args_path),
        "mcp_advance_log_args_path": str(advance_log_args_path),
        "mcp_advance_queue_args_path": str(advance_queue_args_path),
        "mcp_review_session_args_path": str(review_session_args_path),
        "mcp_apply_review_notes_args_path": str(apply_review_notes_args_path),
        "review_notes_template_path": str(review_notes_template_path),
        "review_fill_guide_path": str(review_fill_guide_path),
        "mcp_status_report_args_path": str(status_report_args_path),
        "mcp_acceptance_check_args_path": str(acceptance_check_args_path),
        "mcp_controlled_execution_check_args_path": str(controlled_execution_check_args_path),
        "mcp_controlled_execution_smoke_args_path": str(controlled_execution_smoke_args_path),
        "post_review_refresh_command": refresh_command,
        "asset_count": len(copied_assets),
        "asset_manifest_path": str(assets_dir / "asset-manifest.json"),
        "task_console": task_console,
    }


def refresh_bundle_review_html(bundle_dir: str | Path, *, write: bool = True) -> dict[str, Any]:
    """Refresh review.html from the current bundle manifest/timeline without rebuilding the bundle."""
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    review_html_path = root / "review.html"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline not found: {timeline_path}")
    manifest = read_json(manifest_path)
    timeline_data = read_json(timeline_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    if not isinstance(timeline_data, list):
        raise ValueError("timeline.json must be a JSON array")
    _bind_canonical_transcript_manifest(root, manifest)
    applicability = resolve_media_applicability(root, manifest, timeline_data, write=write)
    manifest["media_applicability"] = applicability
    refresh_identity = configured_identity(root)
    refresh_context = resolve_review_export_context(
        manifest,
        refresh_identity if refresh_identity is not None else resolve_review_export_context(manifest, root=root),
        root=root,
    )
    if refresh_context is not None:
        manifest["project_identity"] = normalize_project_identity(refresh_context, allow_incomplete=True)
    package = {
        "bundle_dir": str(root),
        "title": manifest.get("title") or root.name,
        "coverage": manifest.get("coverage") if isinstance(manifest.get("coverage"), dict) else {},
        "sources": manifest.get("sources") if isinstance(manifest.get("sources"), list) else [],
        "timeline": project_timeline_for_media([item for item in timeline_data if isinstance(item, dict)], applicability),
        "review_artifacts": _review_artifacts_for_render(root, manifest),
        **({"project_identity": manifest["project_identity"]} if manifest.get("project_identity") else {}),
    }
    timeline_alignment = _timeline_alignment_for_review(root, write=write)
    if timeline_alignment.get("manifest_updated"):
        refreshed_manifest = read_json(manifest_path)
        if isinstance(refreshed_manifest, dict):
            manifest = refreshed_manifest
            manifest["media_applicability"] = applicability
            _bind_canonical_transcript_manifest(root, manifest)
            package["review_artifacts"] = _review_artifacts_for_render(root, manifest)
    package["timeline"] = _annotate_timeline_alignment(package["timeline"], timeline_alignment)
    package["timeline"] = _annotate_transcript_semantic_candidates(root, package["timeline"])
    html_text = _media_review_html(package, applicability)
    result = {
        "schema": "lecture_review_html_refresh.v1",
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "timeline_path": str(timeline_path),
        "review_html_path": str(review_html_path),
        "timeline_items": len(package["timeline"]),
        "timeline_alignment_issue_count": int((timeline_alignment.get("summary") or {}).get("items_with_issues") or 0),
        "timeline_alignment_report": str(timeline_alignment.get("report_path") or ""),
        "write": write,
        "refreshed_at": now_iso(),
    }
    if write:
        review_html_path.write_text(html_text, encoding="utf-8")
        manifest["review_html"] = "review.html"
        manifest.setdefault("review_notes", "review-notes.json")
        manifest.setdefault("review_notes_template", "review-notes.template.json")
        manifest.setdefault("review_fill_guide", "review-fill-guide.md")
        manifest.setdefault("mcp_apply_review_notes_args", "mcp-apply-review-notes.args.json")
        _ensure_review_entrypoint_files(root, manifest, package["timeline"])
        manifest["review_html_refreshed_at"] = result["refreshed_at"]
        write_json(manifest_path, manifest)
    return result


def _media_review_html(package: dict[str, Any], applicability: dict[str, Any]) -> str:
    html = render_lecture_review_html(package)
    if applicability['visual_status'] == 'not_applicable':
        html = html.replace('<body>', '<body><p role="status">纯音频：画面、视觉模型与 OCR 不适用。语音完整性和待核词句继续单独检查。</p>', 1)
    return html


def _timeline_alignment_for_review(bundle_dir: Path, *, write: bool) -> dict[str, Any]:
    try:
        result = timeline_alignment_audit(bundle_dir, write=write)
        result["manifest_updated"] = bool(write)
        return result
    except Exception as exc:
        existing = bundle_dir / "timeline-alignment-audit.json"
        if existing.exists():
            try:
                data = read_json(existing)
                if isinstance(data, dict):
                    data.setdefault("render_warning", f"{type(exc).__name__}: {exc}")
                    data["manifest_updated"] = False
                    return data
            except Exception:
                pass
        return {
            "schema": "video_knowledge_pipeline.timeline_alignment_audit.render_error.v1",
            "bundle_dir": str(bundle_dir),
            "summary": {"items_with_issues": 0, "transcript_available": False},
            "items": [],
            "render_warning": f"{type(exc).__name__}: {exc}",
            "manifest_updated": False,
        }


def _annotate_timeline_alignment(timeline: list[dict[str, Any]], audit: dict[str, Any]) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for row in audit.get("items") or []:
        if not isinstance(row, dict) or not row.get("issues"):
            continue
        try:
            index = int(row.get("index") or 0)
        except (TypeError, ValueError):
            index = 0
        if index:
            by_index[index] = row
    annotated: list[dict[str, Any]] = []
    for position, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        copy = dict(item)
        try:
            index = int(copy.get("index") or position)
        except (TypeError, ValueError):
            index = position
        row = by_index.get(index)
        if row:
            copy["timeline_alignment"] = {
                "issues": [str(value) for value in row.get("issues") or []],
                "review_start": row.get("review_start"),
                "review_start_source": row.get("review_start_source") or copy.get("review_start_source") or "",
                "asr_first_start": row.get("asr_first_start"),
                "frame_time": row.get("frame_time"),
                "tagger_times": row.get("tagger_times") or [],
                "asr_overlap_count": row.get("asr_overlap_count") or 0,
                "asr_excerpt": row.get("asr_excerpt") or "",
                "suggested_review_start": row.get("asr_first_start"),
                "suggestion": "Preview only: if the ASR overlap is reliable, use asr_first_start as review_start after human confirmation.",
            }
        annotated.append(copy)
    return annotated


def _annotate_transcript_semantic_candidates(bundle_dir: Path, timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pack_path = bundle_dir / "transcript-semantic-correction-pack.json"
    if not pack_path.is_file():
        return timeline
    try:
        pack = read_json(pack_path)
    except (OSError, ValueError):
        return timeline
    candidates = pack.get("candidates") if isinstance(pack, dict) else []
    if not isinstance(candidates, list):
        return timeline
    applied_by_id: dict[str, dict[str, Any]] = {}
    closure_path = bundle_dir / "transcript-semantic-correction-closure.json"
    if closure_path.is_file():
        try:
            closure = read_json(closure_path)
            for row in closure.get("applied_corrections", []) if isinstance(closure, dict) else []:
                if isinstance(row, dict) and str(row.get("candidate_id") or ""):
                    applied_by_id[str(row["candidate_id"])] = row
        except (OSError, ValueError):
            pass
    annotated = [dict(item) for item in timeline]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        original = str(candidate.get("original_text") or "").strip()
        if not candidate_id or not original:
            continue
        match_index = _best_timeline_candidate_match(annotated, candidate)
        if match_index is None:
            continue
        applied = applied_by_id.get(candidate_id, {})
        row = {
            "candidate_id": candidate_id,
            "original_text": original,
            "suggested_text": str(candidate.get("candidate_text") or candidate.get("suggested_text") or ""),
            "corrected_text": str(applied.get("corrected_text") or ""),
            "time_range": str(candidate.get("time_range") or ""),
            "risk_level": str(candidate.get("risk_level") or ""),
            "reason": str(candidate.get("reason") or ""),
            "evidence_ids": [str(value) for value in candidate.get("evidence_ids") or [] if str(value).strip()],
            "applied": bool(applied),
        }
        annotated[match_index].setdefault("transcript_semantic_candidates", []).append(row)
    return annotated


def _best_timeline_candidate_match(timeline: list[dict[str, Any]], candidate: dict[str, Any]) -> int | None:
    original = str(candidate.get("original_text") or "").casefold().strip()
    candidate_start = _review_float_value(candidate.get("start"), -1.0)
    candidate_end = _review_float_value(candidate.get("end"), candidate_start)
    best: tuple[float, int] | None = None
    for position, item in enumerate(timeline):
        transcript = str(
            item.get("review_transcript_excerpt")
            or item.get("human_corrected_transcript")
            or item.get("transcript")
            or ""
        ).casefold()
        if not original or original not in transcript:
            continue
        start = _review_float_value(item.get("start"), -1.0)
        end = _review_float_value(item.get("end"), start)
        overlap = max(0.0, min(end, candidate_end) - max(start, candidate_start)) if start >= 0 and candidate_start >= 0 else 0.0
        distance = abs(((start + end) / 2.0) - ((candidate_start + candidate_end) / 2.0)) if start >= 0 and candidate_start >= 0 else 0.0
        score = (1000.0 if overlap > 0 else 0.0) + overlap - distance
        if best is None or score > best[0]:
            best = (score, position)
    return best[1] if best is not None else None


def _review_float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

def _ensure_review_entrypoint_files(bundle_dir: Path, manifest: dict[str, Any], timeline: list[dict[str, Any]]) -> None:
    review_notes = bundle_dir / str(manifest.get("review_notes") or "review-notes.json")
    template_path = bundle_dir / str(manifest.get("review_notes_template") or "review-notes.template.json")
    fill_guide_path = bundle_dir / str(manifest.get("review_fill_guide") or "review-fill-guide.md")
    apply_args_path = bundle_dir / str(manifest.get("mcp_apply_review_notes_args") or "mcp-apply-review-notes.args.json")
    if template_path.exists():
        review_template = read_json(template_path)
        if not isinstance(review_template, dict):
            review_template = _review_notes_template_from_timeline(timeline)
            write_json(template_path, review_template)
    else:
        review_template = _review_notes_template_from_timeline(timeline)
        write_json(template_path, review_template)
    if not review_notes.exists():
        write_json(review_notes, {"schema": "lecture_review_notes.v1", "created_at": now_iso(), "reviews": []})
    write_json(apply_args_path, {"bundle_dir": str(bundle_dir), "review_json": str(review_notes), "write": True})
    fill_guide_path.write_text(
        _render_review_fill_guide(
            title=str(manifest.get("title") or bundle_dir.name),
            bundle_dir=bundle_dir,
            review_html_path=bundle_dir / str(manifest.get("review_html") or "review.html"),
            review_notes_path=review_notes,
            review_notes_template_path=template_path,
            apply_review_notes_args_path=apply_args_path,
            review_count=len(review_template.get("reviews", [])),
        ),
        encoding="utf-8",
    )


def _review_artifacts_for_render(bundle_dir: Path, manifest: dict[str, Any]) -> dict[str, str]:
    review_notes = str(manifest.get("review_notes") or "review-notes.json")
    template = str(manifest.get("review_notes_template") or "review-notes.template.json")
    fill_guide = str(manifest.get("review_fill_guide") or "review-fill-guide.md")
    apply_args = str(manifest.get("mcp_apply_review_notes_args") or "mcp-apply-review-notes.args.json")
    batch_acceptance = str(manifest.get("batch_acceptance_summary_markdown") or "")
    batch_repair = str(manifest.get("batch_repair_report") or "")
    batch_human_review = str(manifest.get("batch_human_review") or "")
    return {
        "bundle_dir": str(bundle_dir),
        "media_path": str(manifest.get("multimodal_sample_review_media_path") or manifest.get("media_path") or ""),
        "source_arbitrated_transcript_json": str(manifest.get("source_arbitrated_transcript_json") or ""),
        "corrected_transcript_json": str(manifest.get("corrected_transcript_json") or ""),
        "normalized_transcript_json": str(manifest.get("normalized_transcript_json") or ""),
        "transcript_json": str(manifest.get("transcript_json") or ""),
        "qwen3_forced_alignment_json": str(manifest.get("qwen3_forced_alignment_json") or ""),
        "whisperx_alignment_transcript_json": str(manifest.get("whisperx_alignment_transcript_json") or ""),
        "review_notes": review_notes,
        "review_notes_template": template,
        "task_console": str(manifest.get("task_console") or "task-console.html"),
        "task_console_json": str(manifest.get("task_console_json") or "task-console.json"),
        "mcp_export_task_console_args": str(manifest.get("mcp_export_task_console_args") or "mcp-export-task-console.args.json"),
        "review_fill_guide": fill_guide,
        "review_pack": str(manifest.get("review_pack") or "review-pack.md"),
        "review_pack_json": str(manifest.get("review_pack_json") or "review-pack.json"),
        "review_notes_todo": str(manifest.get("review_notes_todo") or "review-notes.todo.json"),
        "review_closure_status": str(manifest.get("review_closure_status") or "review-closure-status.md"),
        "review_closure_status_json": str(manifest.get("review_closure_status_json") or "review-closure-status.json"),
        "mcp_apply_review_notes_args": apply_args,
        "mcp_next_action_args": str(manifest.get("mcp_next_action_args") or "mcp-bundle-next-action.args.json"),
        "mcp_review_closure_status_args": str(manifest.get("mcp_review_closure_status_args") or "mcp-review-closure-status.args.json"),
        "batch_acceptance_summary": batch_acceptance,
        "batch_repair_run": batch_repair,
        "batch_human_review": batch_human_review,
        "apply_command": f".\\scripts\\video-knowledge.ps1 mcp-call apply_review_notes {apply_args}",
        "review_server_command": f".\\scripts\\start-review-webui.ps1 -BundleDir '{str(bundle_dir).replace(chr(39), chr(39) * 2)}'",
        "validate_command": f".\\scripts\\video-knowledge.ps1 validate-review-notes {bundle_dir} --review-json {bundle_dir / review_notes}",
        "next_action_command": ".\\scripts\\video-knowledge.ps1 mcp-call bundle_next_action "
        + str(manifest.get("mcp_next_action_args") or "mcp-bundle-next-action.args.json"),
        "review_closure_command": ".\\scripts\\video-knowledge.ps1 mcp-call review_closure_status "
        + str(manifest.get("mcp_review_closure_status_args") or "mcp-review-closure-status.args.json"),
        "batch_repair_command": ".\\scripts\\video-knowledge.ps1 batch-repair-run <batch-acceptance-summary.json>",
    }


def _review_notes_template_from_timeline(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for position, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        index = int(item.get("index") or position)
        rows.append(
            {
                "timeline_index": index,
                "time_range": f"{_float_value(item.get('start')):.3f}-{_float_value(item.get('end')):.3f}s",
                "route": item.get("visual_route", ""),
                "reason": ", ".join(str(value) for value in item.get("quality_issues") or []),
                "suggested_status": _suggested_review_status_for_timeline_item(item),
                "evidence_frame_paths": _review_evidence_paths_for_timeline_item(item),
                "transcript_excerpt": _truncate_review_text(item.get("transcript"), 220),
                "visual_text_excerpt": _truncate_review_text(item.get("visual_text"), 220),
                "model_output_excerpt": _truncate_review_text(_review_model_output_text(item), 260),
                "tags": [],
                "comment": "",
                "corrected_transcript": "",
                "corrected_visual_text": "",
                "corrected_visual_understanding": {},
                "corrected_temporal_visual_understanding": {},
                "reviewed_at": "",
            }
        )
    return {"schema": "lecture_review_notes.v1", "created_at": now_iso(), "reviews": rows}


def _suggested_review_status_for_timeline_item(item: dict[str, Any]) -> str:
    issues = {str(value) for value in item.get("quality_issues") or []}
    if "temporal_sequence_without_analysis" in issues:
        return "corrected_temporal_visual_understanding"
    if "semantic_frame_without_analysis" in issues:
        return "corrected_visual_understanding"
    if "missing_visual_text" in issues or "ocr_text_empty" in issues:
        return "corrected_visual_text"
    retention = item.get("visual_retention") if isinstance(item.get("visual_retention"), dict) else {}
    if str(retention.get("recommendation") or "") in {"keep_image", "review_image"}:
        return "keep_image"
    if item.get("needs_human_review"):
        return "needs_human_review"
    return "accepted"


def _review_evidence_paths_for_timeline_item(item: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("frame_paths", "temporal_frame_paths", "original_frame_paths"):
        value = item.get(key)
        if isinstance(value, list):
            paths.extend(str(path) for path in value if str(path))
    for asset in item.get("assets") or []:
        if isinstance(asset, dict):
            paths.append(str(asset.get("path") or asset.get("source") or ""))
    return _dedupe([path for path in paths if path])


def _review_model_output_text(item: dict[str, Any]) -> str:
    parts = []
    for key in ("visual_understanding", "temporal_visual_understanding", "structured_visual"):
        value = item.get(key)
        if value not in (None, "", [], {}):
            parts.append(json.dumps(value, ensure_ascii=False, default=str))
    return " ".join(parts)


def _truncate_review_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _render_review_fill_guide(
    *,
    title: str,
    bundle_dir: Path,
    review_html_path: Path,
    review_notes_path: Path,
    review_notes_template_path: Path,
    apply_review_notes_args_path: Path,
    review_count: int,
) -> str:
    return "\n".join(
        [
            "# Review Notes Fill Guide",
            "",
            f"- Title: {title}",
            f"- Bundle: `{bundle_dir}`",
            f"- Review UI: {review_html_path.resolve().as_uri()}",
            f"- Template JSON: `{review_notes_template_path}`",
            f"- Write reviewed JSON to: `{review_notes_path}`",
            f"- Review rows in template: `{review_count}`",
            "",
            "## Fields",
            "",
            "- `timeline_index`: must match `timeline.json`.",
            "- `status`: use `accepted`, `keep_image`, `needs_fix`, `corrected_visual_text`, `corrected_visual_understanding`, or `corrected_temporal_visual_understanding`.",
            "- `evidence_frame_paths`: keep the frame paths that justify the review decision.",
            "- `corrected_visual_text`: use when OCR or screen text is missing/wrong.",
            "- `corrected_visual_understanding`: JSON object for single-frame visual meaning fixes.",
            "- `corrected_temporal_visual_understanding`: JSON object for multi-frame/action-sequence fixes.",
            "- `comment`: free-form human note. Do not put API keys or secrets here.",
            "",
            "## Apply",
            "",
            "Validate before importing if you edited JSON manually:",
            "",
            "```powershell",
            f".\\scripts\\video-knowledge.ps1 validate-review-notes {bundle_dir} --review-json {review_notes_path}",
            "```",
            "",
            "Apply reviewed notes through MCP args:",
            "",
            "```powershell",
            f".\\scripts\\video-knowledge.ps1 mcp-call apply_review_notes {apply_review_notes_args_path.name}",
            "```",
            "",
        ]
    )


def _copy_frame_assets(package: dict[str, Any], assets_dir: Path) -> dict[str, dict[str, str]]:
    copied: dict[str, dict[str, str]] = {}
    for item in package.get("timeline", []):
        for frame in item.get("frame_paths") or []:
            source = Path(str(frame))
            key = str(source)
            if key in copied:
                continue
            target_name = _asset_name(source, len(copied) + 1)
            target = assets_dir / target_name
            if source.exists() and source.is_file():
                shutil.copy2(source, target)
                copied[key] = {"source": key, "path": f"assets/{target_name}", "copied": "true"}
            else:
                copied[key] = {"source": key, "path": key, "copied": "false"}
    return copied


def _package_with_webui_assets(package: dict[str, Any], assets: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Return a package clone whose frame paths point at bundled assets."""
    cloned = json.loads(json.dumps(package, ensure_ascii=False))
    timeline = cloned.get("timeline") if isinstance(cloned.get("timeline"), list) else []
    for item in timeline:
        if not isinstance(item, dict):
            continue
        frames = item.get("frame_paths") if isinstance(item.get("frame_paths"), list) else []
        rewritten: list[str] = []
        originals: list[str] = []
        for frame in frames:
            original = str(frame)
            asset = assets.get(original)
            rewritten.append(str((asset or {}).get("path") or original))
            originals.append(original)
        if originals:
            item["original_frame_paths"] = originals
        item["frame_paths"] = rewritten
    return cloned


def _preserve_controlled_execution_entrypoints(bundle_dir: Path, manifest: dict[str, Any]) -> None:
    old_path = bundle_dir / "manifest.json"
    old = read_json(old_path) if old_path.exists() else {}
    if not isinstance(old, dict):
        return
    keys = [
        "vision_execution_preflight",
        "vision_execution_preflight_json",
        "mcp_vision_execution_preflight_args",
        "mcp_multimodal_frame_analysis_confirmed_args",
        "mcp_temporal_visual_analysis_confirmed_args",
        "vision_analysis_runs",
        "vision_analysis_runs_jsonl",
        "controlled_execution_check",
        "controlled_execution_check_json",
        "mcp_controlled_execution_check_args",
        "controlled_execution_smoke",
        "controlled_execution_smoke_json",
        "mcp_controlled_execution_smoke_args",
    ]
    for key in keys:
        value = old.get(key)
        if value and _bundle_relative_exists(bundle_dir, value):
            manifest[key] = value


def _bundle_relative_exists(bundle_dir: Path, value: Any) -> bool:
    path = Path(str(value))
    candidate = path if path.is_absolute() else bundle_dir / path
    return candidate.exists()


def _asset_manifest(assets: dict[str, dict[str, str]]) -> dict[str, Any]:
    rows = list(assets.values())
    copied = [row for row in rows if row.get("copied") == "true"]
    missing = [row for row in rows if row.get("copied") != "true"]
    return {
        "schema": "lecture_webui_asset_manifest.v1",
        "copied_count": len(copied),
        "missing_count": len(missing),
        "assets": rows,
    }


def _webui_timeline(package: dict[str, Any], assets: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    audit_by_index = _quality_audit_by_index(package)
    for index, item in enumerate(package.get("timeline", []), start=1):
        audit = audit_by_index.get(index, {})
        rows.append(
            {
                "index": index,
                "start": item.get("start", 0),
                "end": item.get("end", 0),
                "midpoint": item.get("midpoint", 0),
                "video_key": item.get("video_key", ""),
                "video_id": item.get("video_id", ""),
                "video_duration_seconds": item.get("video_duration_seconds", 0),
                "source_segment_ids": item.get("source_segment_ids", []),
                "transcript": item.get("transcript", ""),
                "visual_text": item.get("visual_text", ""),
                "original_transcript": item.get("original_transcript", ""),
                "original_visual_text": item.get("original_visual_text", ""),
                "structured_visual": item.get("structured_visual", []),
                "visual_route": item.get("visual_route", ""),
                "visual_route_confidence": item.get("visual_route_confidence", 0),
                "visual_route_reasons": item.get("visual_route_reasons", []),
                "visual_understanding": item.get("visual_understanding", {}),
                "temporal_visual_understanding": item.get("temporal_visual_understanding", {}),
                "visual_retention": item.get("visual_retention", {}),
                "material_types": item.get("material_types", []),
                "review_status": item.get("review_status", "pending"),
                "needs_human_review": item.get("needs_human_review", True),
                "human_review": item.get("human_review", {}),
                "evidence": item.get("evidence", {}),
                "quality_score": audit.get("score", 0),
                "quality_issues": audit.get("issues", []),
                "assets": [assets.get(str(path), {"source": str(path), "path": str(path), "copied": "false"}) for path in item.get("frame_paths") or []],
            }
        )
    return rows


def _frame_recapture_plan(package: dict[str, Any], bundle_dir: Path) -> dict[str, Any]:
    output_dir = bundle_dir / "assets" / "recapture"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_by_index = _quality_audit_by_index(package)
    items = []
    for index, item in enumerate(package.get("timeline", []), start=1):
        audit = audit_by_index.get(index, {})
        issues = audit.get("issues") if isinstance(audit.get("issues"), list) else []
        if not _needs_frame_recapture(issues):
            continue
        source = str(item.get("video_key") or "").strip()
        midpoint = _midpoint_seconds(item)
        output_path = output_dir / f"timeline-{index:04d}-{_safe_time_label(midpoint)}.jpg"
        command = _ffmpeg_frame_command(source, midpoint, output_path) if source else ""
        items.append(
            {
                "index": index,
                "start": item.get("start", 0),
                "end": item.get("end", 0),
                "midpoint": midpoint,
                "video_key": source,
                "source_exists": bool(source and Path(source).expanduser().exists()),
                "output_path": str(output_path),
                "relative_output": str(output_path.relative_to(bundle_dir)),
                "issues": issues,
                "material_types": item.get("material_types", []),
                "source_segment_ids": item.get("source_segment_ids", []),
                "ffmpeg_command": command,
            }
        )
    return {
        "schema": "lecture_frame_recapture_plan.v1",
        "count": len(items),
        "output_dir": str(output_dir),
        "items": items,
        "notes": [
            "Commands are intentionally not executed by export-webui-bundle.",
            "After capturing frames, add them to the relevant extractor/project output and rerun the lecture package/WebUI export.",
        ],
    }


def _time_gap_recapture_plan(package: dict[str, Any], bundle_dir: Path) -> dict[str, Any]:
    output_dir = bundle_dir / "assets" / "time-gaps"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = package.get("time_gap_audit") if isinstance(package.get("time_gap_audit"), dict) else {}
    gaps = audit.get("gaps") if isinstance(audit.get("gaps"), list) else []
    items = []
    for index, gap in enumerate([item for item in gaps if isinstance(item, dict)], start=1):
        source = str(gap.get("video_key") or "").strip()
        start = _float_value(gap.get("start"))
        end = _float_value(gap.get("end"))
        midpoint = max((start + end) / 2, 0.0)
        output_path = output_dir / f"gap-{index:04d}-{_safe_time_label(midpoint)}.jpg"
        command = _ffmpeg_frame_command(source, midpoint, output_path) if source else ""
        items.append(
            {
                "gap_index": index,
                "start": start,
                "end": end,
                "midpoint": midpoint,
                "duration_seconds": gap.get("duration_seconds", max(end - start, 0.0)),
                "position": gap.get("position", "gap"),
                "video_key": source,
                "source_exists": bool(source and Path(source).expanduser().exists()),
                "output_path": str(output_path),
                "output_exists": output_path.exists(),
                "relative_output": str(output_path.relative_to(bundle_dir)),
                "ffmpeg_command": command,
            }
        )
    return {
        "schema": "lecture_time_gap_recapture_plan.v1",
        "count": len(items),
        "output_dir": str(output_dir),
        "items": items,
        "notes": [
            "Commands sample the midpoint of uncovered timeline ranges.",
            "Inspect captured frames before deciding whether to rerun denser extraction or create manual timeline entries.",
        ],
    }


def _visual_structure_plan(bundle_dir: Path, timeline: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = _visual_structure_candidates(bundle_dir, timeline)
    template_path = write_visual_structure_input_template(bundle_dir, candidates)
    return {
        "schema": "lecture_visual_structure_plan.v1",
        "count": len(candidates),
        "tools": _tool_statuses(),
        "items": candidates,
        "input_template_json": str(template_path),
        "notes": [
            "This branch extracts screenshot text/layout with ebook_markdown_pipeline. It is primary for document_visual frames and auxiliary for non-document frames.",
            "Non-document visual frames still need multimodal_frame_analysis or temporal_visual_analysis; integrated_visual combines OCR/layout and visual understanding.",
        ],
    }


def _video_frame_router_plan(bundle_dir: Path, timeline: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    for index, item in enumerate(timeline, start=1):
        items.append(
            {
                "index": index,
                "start": item.get("start", 0),
                "end": item.get("end", 0),
                "visual_route": item.get("visual_route") or "unknown",
                "confidence": item.get("visual_route_confidence") or 0,
                "reasons": item.get("visual_route_reasons") or [],
                "frame_paths": item.get("frame_paths") or [],
            }
        )
    template_path = write_video_frame_router_input_template(bundle_dir, items)
    return {
        "schema": "lecture_video_frame_router.v1",
        "count": len(items),
        "items": items,
        "input_template_json": str(template_path),
        "notes": ["Run run-video-frame-router before visual analysis so document screenshots, semantic frames, and temporal sequences go to the right branch."],
    }


def _multimodal_frame_analysis_plan(bundle_dir: Path, timeline: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for index, item in enumerate(timeline, start=1):
        route = str(item.get("visual_route") or "")
        if route not in {"semantic_frame", "mixed"}:
            continue
        candidates.append(
            {
                "index": index,
                "start": item.get("start", 0),
                "end": item.get("end", 0),
                "visual_route": route,
                "frame_paths": item.get("frame_paths") or [],
                "has_visual_understanding": bool(item.get("visual_understanding")),
            }
        )
    template_path = write_multimodal_frame_input_template(bundle_dir, candidates)
    return {
        "schema": "lecture_multimodal_frame_analysis.v1",
        "count": len(candidates),
        "items": candidates,
        "input_template_json": str(template_path),
        "notes": ["Default is preview/import only. Set execute=true to call the configured OpenAI-compatible multimodal API."],
    }


def _temporal_visual_analysis_plan(bundle_dir: Path, timeline: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for index, item in enumerate(timeline, start=1):
        route = str(item.get("visual_route") or "")
        if route not in {"temporal_sequence", "mixed"}:
            continue
        candidates.append(
            {
                "index": index,
                "start": item.get("start", 0),
                "end": item.get("end", 0),
                "visual_route": route,
                "frame_paths": item.get("frame_paths") or [],
                "has_temporal_visual_understanding": bool(item.get("temporal_visual_understanding")),
            }
        )
    template_path = write_temporal_visual_input_template(bundle_dir, candidates)
    return {
        "schema": "lecture_temporal_visual_analysis.v1",
        "count": len(candidates),
        "items": candidates,
        "input_template_json": str(template_path),
        "notes": ["Use this for software operations, experiments, mouse movement, demos, and frame-to-frame state changes."],
    }


def _needs_frame_recapture(issues: list[Any]) -> bool:
    issue_set = {str(issue) for issue in issues}
    return bool(issue_set & {"missing_frame", "structured_visual_without_frame", "keep_image_without_frame"})


def _midpoint_seconds(item: dict[str, Any]) -> float:
    try:
        midpoint = float(item.get("midpoint"))
        if midpoint >= 0:
            return midpoint
    except (TypeError, ValueError):
        pass
    try:
        start = float(item.get("start") or 0)
        end = float(item.get("end") or start)
    except (TypeError, ValueError):
        return 0.0
    return max((start + end) / 2, 0.0)


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_time_label(seconds: float) -> str:
    return f"{max(seconds, 0):.3f}".replace(".", "s")


def _ffmpeg_frame_command(source: str, seconds: float, output_path: Path) -> str:
    return " ".join(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{max(seconds, 0):.3f}",
            "-i",
            _ps_quote(source),
            "-frames:v",
            "1",
            _ps_quote(str(output_path)),
        ]
    )


def _quality_audit_by_index(package: dict[str, Any]) -> dict[int, dict[str, Any]]:
    audit = package.get("quality_audit") if isinstance(package.get("quality_audit"), dict) else {}
    priority_items = audit.get("priority_items") if isinstance(audit.get("priority_items"), list) else []
    result = {}
    for item in priority_items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index", 0))
        except (TypeError, ValueError):
            continue
        if index:
            result[index] = item
    return result


def _asset_name(source: Path, index: int) -> str:
    suffix = source.suffix or ".jpg"
    stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in source.stem).strip("_")
    return f"{index:04d}-{stem or 'frame'}{suffix}"


def _bundle_readme(manifest: dict[str, Any]) -> str:
    lines = [
        f"# {manifest.get('title', 'Lecture WebUI Bundle')}",
        "",
        "This is a handoff bundle for BiliNote-like WebUI integration.",
        "",
        "## Files",
        "",
        "- `manifest.json`: stable entry point and metadata.",
        "- `note.md`: Markdown note body suitable for BiliNote-style Markdown viewers.",
        "- `outline.md` / `outline.json`: non-summary navigation outline over the full timeline.",
        "- `search-index.md` / `search-index.json`: deterministic local keyword search index.",
        "- `study-index.md` / `study-index.json`: searchable no-summary learning index.",
        "- `study-cards.md` / `study-cards.json`: evidence-preserving draft study cards.",
        "- `source-artifacts.md` / `source-artifacts.json`: traceability index for original extractor outputs.",
        "- `knowledge-coverage.md` / `knowledge-coverage.json`: no-loss coverage matrix across speech, OCR, visual frames, structured visuals, time-axis gaps, and source artifacts.",
        "- `bundle-status.md` / `bundle-status.json`: compact status dashboard for humans, UI, and agents.",
        "- `acceptance-check.md` / `acceptance-check.json`: final acceptance status across provider, review lifecycle, coverage, and export freshness.",
        "- `exports/knowledge-note.md`: evidence-rich knowledge note and audit-oriented review draft.",
        "- `exports/smart-summary.md`: Codex-assisted smart summary for direct reading and reuse.",
        "- `exports/smart-summary-codex-prompt.md`: local prompt pack for Codex-assisted rewriting without online LLM APIs.",
        "- `exports/full-transcript.md`: full transcript-only Markdown.",
        "- `exports/extraction-audit.md`: coverage/gap/review/evidence audit for checking leak risks.",
        "- `timeline.json`: structured timeline for native UI panels.",
        "- `review.html`: standalone review UI.",
        "- `review-session.md` / `review-session.json`: human/agent review handoff.",
        "- `review-notes.template.json`: fillable review template for remaining visual/OCR/temporal gaps.",
        "- `review-notes.json`: created by BiliNote/WebUI after human review.",
        "- `review-fill-guide.md`: field-level guide for filling review-notes.json safely.",
        "- `mcp-apply-review-notes.args.json`: MCP arguments for applying review notes back into timeline, coverage, and readiness.",
        "- `mcp-refresh-lecture-review.args.json`: MCP arguments for refreshing package/WebUI/Obsidian outputs after review.",
        "- `mcp-run-frame-recapture.args.json`: MCP arguments for previewing/running key-frame recapture.",
        "- `mcp-run-visual-structure.args.json`: MCP arguments for previewing/importing/running ebook_markdown_pipeline-backed screenshot text/layout extraction.",
        "- `mcp-run-ocr-backfill.args.json`: MCP arguments for fallback OCR import/CaptiOCR/Tesseract backfill.",
        "- `mcp-run-screen-text-recovery.args.json`: MCP arguments for crop-based small screen text recovery.",
        "- `mcp-refresh-repair-status.args.json`: MCP arguments for recalculating manifest repair status.",
        "- `mcp-audit-bundle-readiness.args.json`: MCP arguments for checking final export readiness.",
        "- `mcp-audit-knowledge-coverage.args.json`: MCP arguments for refreshing knowledge coverage audit.",
        "- `mcp-prepare-review-session.args.json`: MCP arguments for creating a single human/agent review-session handoff.",
        "- `mcp-bundle-status-report.args.json`: MCP arguments for refreshing the compact bundle status dashboard.",
        "- `mcp-acceptance-check.args.json`: MCP arguments for refreshing the final acceptance report.",
        "- `mcp-controlled-execution-check.args.json`: MCP arguments for checking the controlled real vision execution chain.",
        "- `mcp-controlled-execution-smoke.args.json`: MCP arguments for running a local fixture preflight/write/audit/optional-restore smoke.",
        "- `mcp-export-knowledge-note.args.json`: MCP arguments for exporting the final human-readable knowledge note.",
        "- `mcp-vision-acceptance-plan.args.json`: MCP arguments for checking the first real multimodal API acceptance batch.",
        "- `mcp-run-multimodal-frame-analysis-confirmed.args.json`: generated by vision-execution-preflight when a confirmed semantic direct-execution batch is available.",
        "- `mcp-run-temporal-visual-analysis-confirmed.args.json`: generated by vision-execution-preflight when a confirmed temporal direct-execution batch is available.",
        "- `mcp-bundle-source-artifacts.args.json`: MCP arguments for inspecting source-artifacts.md/json.",
        "- `assets/`: copied frame assets when source files are available.",
        "- `manifest.json -> frame_recapture`: ffmpeg command plan for timeline items that need replacement key frames.",
        "",
        "## After Review",
        "",
        "When BiliNote/WebUI writes `review-notes.json`, first apply notes to the bundle timeline/coverage/readiness with:",
        "",
        "```powershell",
        f".\\scripts\\video-knowledge.ps1 mcp-call apply_review_notes {manifest.get('mcp_apply_review_notes_args', '')}",
        "```",
        "",
        "For the older full package/WebUI/Obsidian refresh path, run:",
        "",
        "```powershell",
        (manifest.get("post_review") or {}).get("refresh_command", ""),
        "```",
        "",
        "Agent equivalent:",
        "",
        "```powershell",
        f".\\scripts\\video-knowledge.ps1 mcp-call {(manifest.get('post_review') or {}).get('mcp_tool', 'refresh_lecture_review_outputs')} {(manifest.get('post_review') or {}).get('mcp_args_path', '')}",
        "```",
        "",
        "After Obsidian export, validate the exported course folder with:",
        "",
        "```powershell",
        "video-knowledge mcp-call obsidian_export_status <course-folder>\\mcp-obsidian-export-status.args.json",
        "```",
        "",
        "## Agent Repair Calls",
        "",
        "```powershell",
        f".\\scripts\\video-knowledge.ps1 mcp-call run_frame_recapture_plan {manifest.get('mcp_frame_recapture_args', '')}",
        f".\\scripts\\video-knowledge.ps1 mcp-call run_visual_structure_plan {manifest.get('mcp_visual_structure_args', '')}",
        f".\\scripts\\video-knowledge.ps1 mcp-call run_ocr_backfill {manifest.get('mcp_ocr_backfill_args', '')}",
        f".\\scripts\\video-knowledge.ps1 mcp-call run_screen_text_recovery {manifest.get('mcp_screen_text_recovery_args', '')}",
        f".\\scripts\\video-knowledge.ps1 mcp-call repair_bundle_assets {manifest.get('mcp_asset_repair_args', '')}",
        f".\\scripts\\video-knowledge.ps1 mcp-call refresh_bundle_repair_status {manifest.get('mcp_repair_status_args', '')}",
        f".\\scripts\\video-knowledge.ps1 mcp-call audit_bundle_readiness {manifest.get('mcp_readiness_args', '')}",
        f".\\scripts\\video-knowledge.ps1 mcp-call audit_knowledge_coverage {manifest.get('mcp_knowledge_coverage_args', '')}",
        f".\\scripts\\video-knowledge.ps1 mcp-call bundle_next_action {manifest.get('mcp_next_action_args', '')}",
        f".\\scripts\\video-knowledge.ps1 mcp-call bundle_source_artifacts {manifest.get('mcp_source_artifacts_args', '')}",
        f".\\scripts\\video-knowledge.ps1 mcp-call bundle_advance {manifest.get('mcp_advance_args', '')}",
        f".\\scripts\\video-knowledge.ps1 mcp-call bundle_advance_log {manifest.get('mcp_advance_log_args', '')}",
        f".\\scripts\\video-knowledge.ps1 mcp-call bundle_advance_queue {manifest.get('mcp_advance_queue_args', '')}",
        f".\\scripts\\video-knowledge.ps1 mcp-call prepare_review_session {manifest.get('mcp_review_session_args', '')}",
        f".\\scripts\\video-knowledge.ps1 mcp-call apply_review_notes {manifest.get('mcp_apply_review_notes_args', '')}",
        f".\\scripts\\video-knowledge.ps1 mcp-call bundle_status_report {manifest.get('mcp_status_report_args', '')}",
        f".\\scripts\\video-knowledge.ps1 mcp-call acceptance_check {manifest.get('mcp_acceptance_check_args', '')}",
        f".\\scripts\\video-knowledge.ps1 mcp-call export_knowledge_note {manifest.get('mcp_export_knowledge_note_args', '')}",
        f".\\scripts\\video-knowledge.ps1 mcp-call vision_acceptance_plan {manifest.get('mcp_vision_acceptance_plan_args', '')}",
        "```",
        "",
        "Confirmed direct vision calls generated by `vision-execution-preflight`:",
        "",
        "```powershell",
    ]
    confirmed_lines = _confirmed_direct_vision_readme_lines(manifest)
    lines.extend(confirmed_lines or ["# Run vision-execution-preflight first; confirmed direct vision args are not generated in the initial bundle."])
    lines.extend(
        [
            "```",
            "",
        "## Integration Notes",
        "",
        ]
    )
    lines.extend(f"- {note}" for note in manifest.get("integration_notes", []))
    recapture = manifest.get("frame_recapture") if isinstance(manifest.get("frame_recapture"), dict) else {}
    items = recapture.get("items") if isinstance(recapture.get("items"), list) else []
    if items:
        lines.extend(["", "## Frame Recapture", ""])
        for item in items:
            lines.append(
                f"- timeline `{item.get('index')}` at `{item.get('midpoint')}`s -> `{item.get('relative_output', '')}`"
            )
    return "\n".join(lines).rstrip() + "\n"


def _confirmed_direct_vision_readme_lines(manifest: dict[str, Any]) -> list[str]:
    rows = []
    semantic = str(manifest.get("mcp_multimodal_frame_analysis_confirmed_args") or "").strip()
    temporal = str(manifest.get("mcp_temporal_visual_analysis_confirmed_args") or "").strip()
    if semantic:
        rows.append(f".\\scripts\\video-knowledge.ps1 mcp-call run_multimodal_frame_analysis {semantic}")
    if temporal:
        rows.append(f".\\scripts\\video-knowledge.ps1 mcp-call run_temporal_visual_analysis {temporal}")
    return rows


def _local_cli_command(args: list[str]) -> str:
    return ".\\scripts\\video-knowledge.ps1 " + " ".join(_ps_quote(arg) for arg in args)
