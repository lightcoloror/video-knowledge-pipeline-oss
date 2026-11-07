from __future__ import annotations
from .review_export_naming import review_export_script

import html
import json
from pathlib import Path
from typing import Any

from .powershell import quote_powershell_literal as _ps_quote
from .consented_model_batch import list_consented_model_batches
from .config import config_status, ebook_pipeline_profile, model_api_settings_status
from .content_asset_status import content_asset_status
from .model_api_settings import model_api_settings_ui_url
from .models import now_iso
from .storage import read_json, write_json
from .term_correction_status import term_correction_status as _term_correction_status
from .transcript_semantic_correction import transcript_semantic_correction_status as _semantic_correction_status
from .transcript_semantic_batch import transcript_semantic_repair_queue
from .timeline_alignment_audit import timeline_alignment_audit
from .video_moment_index import build_video_moment_index
from .visual_evidence import evidence_level_counts, temporal_evidence_state, visual_evidence_state

MODEL_BATCH_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _autopilot_bundle_summary(root: Path) -> dict[str, Any]:
    # Lazy import avoids bundle_next -> lecture_workflow -> task_console ->
    # autopilot initialization cycles while retaining one summary contract.
    from .autopilot import autopilot_bundle_summary

    return autopilot_bundle_summary(root)


def _summary_claim_review_status(root: Path) -> dict[str, Any]:
    from .summary_claim_review import summary_claim_review_status

    return summary_claim_review_status(root)


def _summary_stage_status(root: Path) -> dict[str, Any]:
    from .summary_stage_coverage import current_summary_stage_coverage
    return current_summary_stage_coverage(root)


def _citation_digest_status(root: Path) -> dict[str, Any]:
    from .citation_digest import audit_citation_digest
    return audit_citation_digest(root)


def _citation_digest_panel_html(status: dict[str, Any]) -> str:
    if not status.get('evaluated'):
        return ''
    esc = lambda value: html.escape(str(value), quote=True)
    levels = ' / '.join(f'{esc(k)}: {esc(v)}' for k, v in (status.get('evidence_levels') or {}).items())
    return f'''<section class="panel" id="citation-digest-integrity"><h2>引用导航完整性</h2>
      <p>{esc(status.get('status'))}；记录 {esc(status.get('record_count', 'unknown'))}；{levels}</p>
      <p>{esc(status.get('detail'))}</p><p>引用存在不等于支持结论；未声明的内容完整性仍未评估。</p>
      <p><a href="exports/smart-summary-chapters.json">完整引用 / 等级 / 精确定位</a></p></section>'''


def _summary_stage_panel_html(status: dict[str, Any]) -> str:
    if not status.get('required'):
        return ''
    esc = lambda value: html.escape(str(value), quote=True)
    stages = ('source_evidence', 'material', 'model_input', 'output')
    captured = str(status.get('input_basis', '')).startswith('captured_gateway_input')
    labels = ('已定位来源', '章节材料', '模型请求输入' if captured else '准备输入', '最终正文')
    input_note = '已绑定 Gateway 请求；仍不能证明模型逐字观察。' if captured else '准备输入不代表模型已读。'
    rows = []
    for point in status.get('points') or []:
        cells = ''.join(f'<td>{esc((point.get("stages") or {}).get(key))}</td>' for key in stages)
        rows.append(f'<tr><td>{esc(point.get("id"))}</td>{cells}<td>{esc(point.get("first_loss"))}</td></tr>')
    links = ' · '.join(f'<a href="{esc(path)}">{esc(key)}</a>' for key, path in (status.get('artifacts') or {}).items())
    headings = ''.join(f'<th>{name}</th>' for name in labels)
    errors = '<br>'.join(esc(value) for value in status.get('errors') or [])
    return f'''<section class="panel" id="summary-stage-coverage"><h2>重要信息在哪一层丢失</h2>
      <p>{esc(status.get('status'))}；材料利用率：{esc(status.get('material_utilization'))}</p>
      <p>仅评估已声明关键点；{input_note}来源一致不代表现实事实已证实。</p>
      <table><thead><tr><th>关键点</th>{headings}<th>首次缺失</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
      <p>{links}</p><p>{errors}</p></section>'''


def _summary_claim_review_panel_html(status: dict[str, Any]) -> str:
    """One existing-shell panel shared with the workbench; no second UI state."""
    if not status.get('required'):
        return ''
    esc = lambda value: html.escape(str(value), quote=True)
    task = status.get('next_task') or {}
    counts = ' / '.join(f'{esc(key)}: {esc(value)}' for key, value in (status.get('counts') or {}).items())
    commands = status.get('commands') or {}
    allowed_commands = ('prepare', 'import', 'status') if task else ('status',)
    if status.get('passed') is True:
        allowed_commands = ('resume_summary', 'status')
    command_html = ''.join(f'<p>{esc(name)}<pre>{esc(commands[name])}</pre></p>'
                           for name in allowed_commands if commands.get(name))
    links = ' · '.join(f'<a href="{esc(row["path"])}">{esc(row["label"])}</a>'
                       for row in status.get('artifacts', []))
    errors = '<br>'.join(esc(value) for value in status.get('errors', []))
    return f'''<section class="panel" id="summary-claim-review">
      <h2>摘要逐条证据审核</h2>
      <p>{esc(status.get('status', 'not_assessed'))} — {counts}</p>
      <p>当前任务：{esc(task.get('task_id', 'none'))} / {esc(task.get('phase', 'none'))}</p>
      <p>来源和解释审核不代表现实事实已证实；完整性另行评估，不能自动接受。</p>
      <p>{links}</p><p>{errors}</p>{command_html}
    </section>'''



def export_task_console(bundle_dir: str | Path, *, write: bool = True, refresh: bool = False) -> dict[str, Any]:
    """Write a lightweight operator console for an existing WebUI bundle.

    The console is intentionally static. It gives humans copyable commands and
    links into the review/export artifacts while agents can read the JSON plan.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline not found: {timeline_path}")

    manifest = _read_object(manifest_path)
    timeline_data = read_json(timeline_path)
    if not isinstance(timeline_data, list):
        raise ValueError("timeline.json must be a JSON array")
    timeline = [item for item in timeline_data if isinstance(item, dict)]

    refreshed: dict[str, Any] = {}
    if refresh and write:
        refreshed = _refresh_known_status(root)
        manifest = _read_object(manifest_path)

    moment_index = _safe_build_video_moment_index(root, write=write)
    if write and not moment_index.get("error"):
        manifest = _read_object(manifest_path)
    timeline_alignment = _safe_timeline_alignment_audit(root, write=write)
    if write and not timeline_alignment.get("error"):
        manifest = _read_object(manifest_path)

    console_json_path = root / "task-console.json"
    console_html_path = root / "task-console.html"
    model_settings_json_path = root / "model-settings.json"
    model_settings_html_path = root / "model-settings.html"
    mcp_args_path = root / "mcp-export-task-console.args.json"
    manifest_for_console = dict(manifest)
    manifest_for_console.setdefault("task_console", "task-console.html")
    manifest_for_console.setdefault("video_workbench_html", "video-workbench.html")
    manifest_for_console.setdefault("video_workbench_json", "video-workbench.json")
    manifest_for_console.setdefault("mcp_video_workbench_args", "mcp-video-workbench.args.json")
    manifest_for_console.setdefault("task_console_json", "task-console.json")
    manifest_for_console.setdefault("quality_console", "quality-console.html")
    manifest_for_console.setdefault("quality_console_json", "quality-console.json")
    manifest_for_console.setdefault("model_settings", "model-settings.html")
    manifest_for_console.setdefault("model_settings_json", "model-settings.json")
    manifest_for_console.setdefault("transcript_editor_html", "transcript-editor.html")
    manifest_for_console.setdefault("transcript_edit_session_json", "transcript-edit-session.json")
    manifest_for_console.setdefault("mcp_prepare_transcript_edit_session_args", "mcp-prepare-transcript-edit-session.args.json")
    manifest_for_console.setdefault("mcp_apply_transcript_edits_args", "mcp-apply-transcript-edits.args.json")
    manifest_for_console.setdefault("transcript_evidence_correction_pipeline_markdown", "transcript-evidence-correction-pipeline.md")
    manifest_for_console.setdefault("mcp_transcript_evidence_correction_pipeline_args", "mcp-transcript-evidence-correction-pipeline.args.json")
    manifest_for_console.setdefault("transcript_source_arbitration_markdown", "transcript-source-arbitration.md")
    manifest_for_console.setdefault("term_arbitration_codex_markdown", "term-arbitration-codex.md")
    manifest_for_console.setdefault("term_arbitration_codex_prompt_markdown", "term-arbitration-codex-prompt.md")
    manifest_for_console.setdefault("term_arbitration_codex_result_codex_markdown", "term-arbitration-codex-result.codex.md")
    manifest_for_console.setdefault("term_arbitration_codex_pack_json", "term-arbitration-codex-pack.json")
    manifest_for_console.setdefault("term_arbitration_codex_draft_json", "term-arbitration-codex-draft.json")
    manifest_for_console.setdefault("term_arbitration_codex_validation_markdown", "term-arbitration-codex-validation.md")
    manifest_for_console.setdefault("term_arbitration_codex_validation_json", "term-arbitration-codex-validation.json")
    manifest_for_console.setdefault("term_arbitration_glossary_json", "term-arbitration-glossary.json")
    manifest_for_console.setdefault("mcp_term_arbitration_codex_args", "mcp-term-arbitration-codex.args.json")
    manifest_for_console.setdefault("mcp_term_arbitration_codex_validate_args", "mcp-term-arbitration-codex-validate.args.json")
    manifest_for_console.setdefault("term_correction_impact_report_markdown", "term-correction-impact-report.md")
    manifest_for_console.setdefault("term_correction_impact_report_json", "term-correction-impact-report.json")
    manifest_for_console.setdefault("mcp_term_correction_impact_report_args", "mcp-term-correction-impact-report.args.json")
    manifest_for_console.setdefault("term_correction_closure_markdown", "term-correction-closure.md")
    manifest_for_console.setdefault("term_correction_closure_json", "term-correction-closure.json")
    manifest_for_console.setdefault("mcp_term_correction_closure_args", "mcp-term-correction-closure.args.json")
    manifest_for_console.setdefault("mcp_term_correction_closure_codex_args", "mcp-term-correction-closure-codex.args.json")
    manifest_for_console.setdefault("mcp_term_correction_status_args", "mcp-term-correction-status.args.json")
    manifest_for_console.setdefault("transcript_semantic_correction_pack_json", "transcript-semantic-correction-pack.json")
    manifest_for_console.setdefault("transcript_semantic_correction_prompt_markdown", "transcript-semantic-correction-prompt.md")
    manifest_for_console.setdefault("transcript_semantic_correction_llm_prompt_markdown", "transcript-semantic-correction-llm-prompt.md")
    manifest_for_console.setdefault("transcript_semantic_candidate_discovery_pack_json", "transcript-semantic-candidate-discovery-pack.json")
    manifest_for_console.setdefault("transcript_semantic_candidate_discovery_prompt_markdown", "transcript-semantic-candidate-discovery-prompt.md")
    manifest_for_console.setdefault("transcript_semantic_candidate_discovery_template_json", "transcript-semantic-candidate-discovery-template.json")
    manifest_for_console.setdefault("transcript_semantic_candidate_discovery_llm_prompt_markdown", "transcript-semantic-candidate-discovery-llm-prompt.md")
    manifest_for_console.setdefault("transcript_semantic_candidate_suggestions_codex_markdown", "transcript-semantic-candidate-suggestions.codex.md")
    manifest_for_console.setdefault("transcript_semantic_candidate_suggestions_llm_markdown", "transcript-semantic-candidate-suggestions.llm.md")
    manifest_for_console.setdefault("transcript_semantic_candidate_suggestions_import_json", "transcript-semantic-candidate-suggestions-import.json")
    manifest_for_console.setdefault("mcp_transcript_semantic_candidate_discovery_pack_args", "mcp-transcript-semantic-candidate-discovery-pack.args.json")
    manifest_for_console.setdefault("mcp_transcript_semantic_candidate_discovery_llm_draft_args", "mcp-transcript-semantic-candidate-discovery-llm-draft.args.json")
    manifest_for_console.setdefault("mcp_transcript_semantic_candidate_discovery_codex_draft_args", "mcp-transcript-semantic-candidate-discovery-codex-draft.args.json")
    manifest_for_console.setdefault("mcp_import_transcript_semantic_candidate_suggestions_args", "mcp-import-transcript-semantic-candidate-suggestions.args.json")
    manifest_for_console.setdefault("transcript_semantic_correction_result_codex_markdown", "transcript-semantic-correction-result.codex.md")
    manifest_for_console.setdefault("transcript_semantic_correction_result_llm_markdown", "transcript-semantic-correction-result.llm.md")
    manifest_for_console.setdefault("transcript_semantic_correction_validation_markdown", "transcript-semantic-correction-validation.md")
    manifest_for_console.setdefault("transcript_semantic_correction_closure_markdown", "transcript-semantic-correction-closure.md")
    manifest_for_console.setdefault("transcript_semantic_correction_impact_report_markdown", "transcript-semantic-correction-impact-report.md")
    manifest_for_console.setdefault("transcript_semantic_correction_readable_impact_markdown", "transcript-semantic-readable-impact-report.md")
    manifest_for_console.setdefault("transcript_semantic_summary_impact_markdown", "transcript-semantic-summary-impact-report.md")
    manifest_for_console.setdefault("transcript_semantic_correction_status_markdown", "transcript-semantic-correction-status.md")
    manifest_for_console.setdefault("mcp_transcript_semantic_correction_pack_args", "mcp-transcript-semantic-correction-pack.args.json")
    manifest_for_console.setdefault("mcp_transcript_semantic_correction_codex_draft_args", "mcp-transcript-semantic-correction-codex-draft.args.json")
    manifest_for_console.setdefault("mcp_transcript_semantic_correction_llm_draft_args", "mcp-transcript-semantic-correction-llm-draft.args.json")
    manifest_for_console.setdefault("mcp_validate_transcript_semantic_correction_args", "mcp-validate-transcript-semantic-correction.args.json")
    manifest_for_console.setdefault("mcp_transcript_semantic_correction_closure_args", "mcp-transcript-semantic-correction-closure.args.json")
    manifest_for_console.setdefault("mcp_transcript_semantic_correction_impact_report_args", "mcp-transcript-semantic-correction-impact-report.args.json")
    manifest_for_console.setdefault("mcp_transcript_semantic_readable_impact_report_args", "mcp-transcript-semantic-readable-impact-report.args.json")
    manifest_for_console.setdefault("mcp_transcript_semantic_summary_impact_report_args", "mcp-transcript-semantic-summary-impact-report.args.json")
    manifest_for_console.setdefault("mcp_transcript_semantic_correction_status_args", "mcp-transcript-semantic-correction-status.args.json")
    manifest_for_console.setdefault("transcript_semantic_repair_queue_json", "exports/transcript-semantic-repair-queue.json")
    manifest_for_console.setdefault("transcript_semantic_repair_queue_markdown", "exports/transcript-semantic-repair-queue.md")
    manifest_for_console.setdefault("mcp_transcript_semantic_repair_queue_args", "mcp-transcript-semantic-repair-queue.args.json")
    manifest_for_console.setdefault("transcript_semantic_repair_run_json", "exports/transcript-semantic-repair-run.json")
    manifest_for_console.setdefault("transcript_semantic_repair_run_markdown", "exports/transcript-semantic-repair-run.md")
    manifest_for_console.setdefault("mcp_transcript_semantic_repair_run_args", "mcp-transcript-semantic-repair-run.args.json")
    manifest_for_console.setdefault("transcript_semantic_batch_review_pack_json", "exports/transcript-semantic-batch-review-pack.json")
    manifest_for_console.setdefault("transcript_semantic_batch_review_pack_markdown", "exports/transcript-semantic-batch-review-pack.md")
    manifest_for_console.setdefault("transcript_semantic_batch_review_notes_todo_json", "exports/transcript-semantic-batch-review-notes.todo.json")
    manifest_for_console.setdefault("transcript_semantic_batch_codex_review_prompt_markdown", "exports/transcript-semantic-batch-codex-review-prompt.md")
    manifest_for_console.setdefault("transcript_semantic_batch_codex_review_draft_json", "exports/transcript-semantic-batch-review-notes.codex-draft.json")
    manifest_for_console.setdefault("transcript_semantic_batch_codex_review_draft_markdown", "exports/transcript-semantic-batch-review-notes.codex-draft.md")
    manifest_for_console.setdefault("transcript_semantic_batch_review_import_json", "exports/transcript-semantic-batch-review-import.json")
    manifest_for_console.setdefault("transcript_semantic_batch_review_import_markdown", "exports/transcript-semantic-batch-review-import.md")
    manifest_for_console.setdefault("mcp_transcript_semantic_batch_review_pack_args", "mcp-transcript-semantic-batch-review-pack.args.json")
    manifest_for_console.setdefault("mcp_transcript_semantic_batch_codex_review_draft_args", "mcp-transcript-semantic-batch-codex-review-draft.args.json")
    manifest_for_console.setdefault("mcp_transcript_semantic_batch_import_review_notes_args", "mcp-transcript-semantic-batch-import-review-notes.args.json")
    manifest_for_console.setdefault("postprocessed_transcript_markdown", "asr-transcript-postprocess.md")
    manifest_for_console.setdefault("postprocessed_transcript_json", "postprocessed-transcript.json")
    manifest_for_console.setdefault("mcp_postprocess_asr_transcript_args", "mcp-postprocess-asr-transcript.args.json")
    manifest_for_console.setdefault("readable_transcript_json", "readable-transcript.json")
    manifest_for_console.setdefault("readable_transcript_srt", "readable-transcript.srt")
    manifest_for_console.setdefault("llm_readable_transcript_json", "llm-readable-transcript.json")
    manifest_for_console.setdefault("llm_readable_transcript_srt", "llm-readable-transcript.srt")
    manifest_for_console.setdefault("llm_readable_transcript_markdown", "llm-readable-transcript.md")
    manifest_for_console.setdefault("mcp_readable_transcript_llm_polish_args", "mcp-readable-transcript-llm-polish.args.json")
    manifest_for_console.setdefault("agent_readable_transcript_json", "agent-readable-transcript.json")
    manifest_for_console.setdefault("agent_readable_transcript_markdown", "agent-readable-transcript.md")
    manifest_for_console.setdefault("agent_readable_transcript_rewrite_json", "agent-readable-transcript-rewrite.json")
    manifest_for_console.setdefault("agent_readable_transcript_rewrite_markdown", "agent-readable-transcript-rewrite.md")
    manifest_for_console.setdefault("agent_readable_transcript_task_json", "agent-readable-transcript-task.json")
    manifest_for_console.setdefault("agent_readable_transcript_task_markdown", "agent-readable-transcript-task.md")
    manifest_for_console.setdefault("transcript_quality_gate_json", "transcript-quality-gate.json")
    manifest_for_console.setdefault("transcript_quality_gate_markdown", "transcript-quality-gate.md")
    manifest_for_console.setdefault("mcp_agent_readable_transcript_rewrite_args", "mcp-agent-readable-transcript-rewrite.args.json")
    manifest_for_console.setdefault("mcp_transcript_quality_gate_args", "mcp-transcript-quality-gate.args.json")
    manifest_for_console.setdefault("mcp_transcript_source_arbitration_args", "mcp-transcript-source-arbitration.args.json")
    manifest_for_console.setdefault("mcp_export_task_console_args", "mcp-export-task-console.args.json")
    manifest_for_console.setdefault("mcp_plan_cloud_asr_args", "mcp-plan-cloud-asr.args.json")
    manifest_for_console.setdefault("mcp_run_cloud_asr_plan_args", "mcp-run-cloud-asr-plan.args.json")
    manifest_for_console.setdefault("mcp_plan_local_asr_service_args", "mcp-plan-local-asr-service.args.json")
    manifest_for_console.setdefault("mcp_run_local_asr_service_plan_args", "mcp-run-local-asr-service-plan.args.json")
    manifest_for_console.setdefault("vision_review_queue_html", "vision-review-queue.html")
    manifest_for_console.setdefault("run_artifact_registry_report", "run-artifact-registry.md")
    manifest_for_console.setdefault("mcp_run_artifact_registry_args", "mcp-run-artifact-registry.args.json")
    manifest_for_console.setdefault("multimodal_sample_review_html", "multimodal-sample-review.html")
    manifest_for_console.setdefault("multimodal_sample_review_json", "multimodal-sample-review.json")
    manifest_for_console.setdefault("multimodal_sample_review_summary_report", "multimodal-sample-review-summary.md")
    manifest_for_console.setdefault("human_sample_eval_report", "human-sample-eval.md")
    manifest_for_console.setdefault("video_moment_index", "exports/video-moment-index.json")
    manifest_for_console.setdefault("video_moment_index_markdown", "exports/video-moment-index.md")
    manifest_for_console.setdefault("mcp_video_moment_index_args", "mcp-video-moment-index.args.json")
    manifest_for_console.setdefault("long_video_memory_pack", "exports/long-video-memory-pack.json")
    manifest_for_console.setdefault("long_video_memory_pack_markdown", "exports/long-video-memory-pack.md")
    manifest_for_console.setdefault("mcp_long_video_memory_pack_args", "mcp-long-video-memory-pack.args.json")
    manifest_for_console.setdefault("video_rag_pack_markdown", "exports/video-rag-pack.md")
    manifest_for_console.setdefault("video_rag_search_markdown", "exports/video-rag-search.md")
    manifest_for_console.setdefault("video_rag_service_plan_markdown", "exports/video-rag-service-plan.md")
    manifest_for_console.setdefault("video_rag_chunks_jsonl", "exports/video-rag-chunks.jsonl")
    manifest_for_console.setdefault("mcp_video_rag_pack_args", "mcp-video-rag-pack.args.json")
    manifest_for_console.setdefault("mcp_video_rag_search_args", "mcp-video-rag-search.args.json")
    manifest_for_console.setdefault("mcp_video_rag_service_plan_args", "mcp-video-rag-service-plan.args.json")
    manifest_for_console.setdefault("external_capability_pack_markdown", "exports/external-capability-pack.md")
    manifest_for_console.setdefault("online_model_api_matrix_json", "exports/online-model-api-matrix.json")
    manifest_for_console.setdefault("online_model_api_matrix_markdown", "exports/online-model-api-matrix.md")
    manifest_for_console.setdefault("mcp_online_model_api_args", "mcp-online-model-api.args.json")
    manifest_for_console.setdefault("mcp_online_model_api_matrix_args", "mcp-online-model-api-matrix.args.json")
    manifest_for_console.setdefault("mcp_external_capability_pack_args", "mcp-external-capability-pack.args.json")
    manifest_for_console.setdefault("timeline_alignment_audit_json", "timeline-alignment-audit.json")
    manifest_for_console.setdefault("timeline_alignment_audit_report", "timeline-alignment-audit.md")
    manifest_for_console.setdefault("mcp_timeline_alignment_audit_args", "mcp-timeline-alignment-audit.args.json")
    manifest_for_console.setdefault("tile_result_import_report", "tile-result-import.md")
    manifest_for_console.setdefault("mcp_tile_result_import_build_args", "mcp-tile-result-import-build.args.json")
    manifest_for_console.setdefault("tile_result_merge_report", "tile-result-merge.md")
    manifest_for_console.setdefault("mcp_tile_result_merge_args", "mcp-tile-result-merge.args.json")
    manifest_for_console.setdefault("smart_summary_input_pack_markdown", "exports/smart-summary-input-pack.md")
    manifest_for_console.setdefault("smart_summary_chapters_markdown", "exports/smart-summary-chapters.md")
    manifest_for_console.setdefault("smart_summary_course_map_markdown", "exports/course-map.md")
    manifest_for_console.setdefault("mcp_build_smart_summary_input_pack_args", "mcp-build-smart-summary-input-pack.args.json")
    manifest_for_console.setdefault("mcp_build_smart_summary_chapters_args", "mcp-build-smart-summary-chapters.args.json")
    manifest_for_console.setdefault("smart_summary_llm_rewrite_pack_markdown", "exports/smart-summary-llm-rewrite-pack.md")
    manifest_for_console.setdefault("smart_summary_llm_rewrite_template_markdown", "exports/smart-summary.llm.todo.md")
    manifest_for_console.setdefault("smart_summary_llm_rewrite_status_json", "exports/smart-summary-llm-rewrite-status.json")
    manifest_for_console.setdefault("mcp_prepare_smart_summary_llm_rewrite_args", "mcp-prepare-smart-summary-llm-rewrite.args.json")
    manifest_for_console.setdefault("mcp_run_smart_summary_llm_rewrite_args", "mcp-run-smart-summary-llm-rewrite.args.json")
    manifest_for_console.setdefault("smart_summary_section_workflow_markdown", "exports/smart-summary-section-workflow.md")
    manifest_for_console.setdefault("smart_summary_section_editor_html", "smart-summary-section-editor.html")
    manifest_for_console.setdefault("smart_summary_section_semantic_review_notes_template", "exports/smart-summary-section-semantic-review-notes.template.json")
    manifest_for_console.setdefault("mcp_smart_summary_section_workflow_args", "mcp-smart-summary-section-workflow.args.json")
    manifest_for_console.setdefault("mcp_smart_summary_section_editor_args", "mcp-smart-summary-section-editor.args.json")
    manifest_for_console.setdefault("smart_summary_section_apply_markdown", "exports/smart-summary-section-apply.md")
    manifest_for_console.setdefault("bilinote_mind_map_prompt_pack_markdown", "exports/bilinote-mind-map-prompt-pack.md")
    manifest_for_console.setdefault("mcp_bilinote_mind_map_prompt_pack_args", "mcp-bilinote-mind-map-prompt-pack.args.json")
    manifest_for_console.setdefault("mcp_smart_summary_section_apply_args", "mcp-smart-summary-section-apply.args.json")
    manifest_for_console.setdefault("smart_summary_section_llm_rewrite_markdown", "exports/smart-summary-section-llm-rewrite.md")
    manifest_for_console.setdefault("smart_summary_section_llm_revisions_json", "exports/smart-summary-section-llm-revisions.json")
    manifest_for_console.setdefault("mcp_smart_summary_section_llm_rewrite_args", "mcp-smart-summary-section-llm-rewrite.args.json")
    status = _build_status(root, manifest_for_console, timeline)
    status["semantic_repair_queue"] = _safe_transcript_semantic_repair_queue(root, write=write)
    status["semantic_repair_run"] = _safe_transcript_semantic_repair_run(root)
    status["semantic_batch_review"] = _safe_transcript_semantic_batch_review_status(root)
    commands = _build_commands(root, manifest_for_console, status)
    artifacts = _artifact_links(root, manifest_for_console)
    run_registry = _load_run_registry_for_console(root)
    processing_queue = _build_processing_queue(root, run_registry)
    subqueue_action_plan = _build_subqueue_action_plan(processing_queue)
    model_batches = _model_batches_for_bundle(root)
    bridge = _semantic_repair_bridge(root)
    autopilot = _autopilot_bundle_summary(root)
    result = {
        "schema": "video_knowledge_pipeline.task_console.v1",
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "timeline_path": str(timeline_path),
        "task_console_json_path": str(console_json_path),
        "task_console_html_path": str(console_html_path),
        "model_settings_json_path": str(model_settings_json_path),
        "model_settings_html_path": str(model_settings_html_path),
        "mcp_args_path": str(mcp_args_path),
        "title": str(manifest.get("title") or root.name),
        "generated_at": now_iso(),
        "write": write,
        "refresh": refresh,
        "refreshed": refreshed,
        "status": status,
        "artifacts": artifacts,
        "commands": commands,
        "run_registry": run_registry,
        "processing_queue": processing_queue,
        "subqueue_action_plan": subqueue_action_plan,
        "model_batches": model_batches,
        "moment_index": _compact_moment_index(moment_index),
        "timeline_alignment": _compact_timeline_alignment(timeline_alignment),
        "media": _media_source(root, manifest, timeline),
        "model_api_settings": model_api_settings_status(),
        "bridge": bridge,
        "autopilot": autopilot,
        "summary_claim_review": _summary_claim_review_status(root),
        "summary_stage_coverage": _summary_stage_status(root),
        "citation_digest_integrity": _citation_digest_status(root),
        "operator_boundary": {
            "default_behavior": "review_and_preview_only",
            "ebook_pipeline": "local ebook_markdown_pipeline execution follows the unified ebook_pipeline config",
            "cloud_vision": "requires vision preflight and explicit execute/confirmation arguments",
            "asr": "local ASR execution is explicit; cloud ASR is not the default path",
            "download": "not handled by this console; use video-download-orchestrator through VKP/OpenClaw planning tools",
            "publication": "content material cards are review-only; publication_allowed=false",
        },
    }

    if write:
        write_json(console_json_path, result)
        write_json(model_settings_json_path, result["model_api_settings"])
        console_html_path.write_text(_render_task_console_html(result), encoding="utf-8")
        model_settings_html_path.write_text(_render_model_settings_html(result), encoding="utf-8")
        write_json(mcp_args_path, {"bundle_dir": str(root), "write": True, "refresh": False})
        write_json(root / "mcp-video-workbench.args.json", {"bundle_dir": str(root), "write": True})
        write_json(root / "mcp-run-visual-structure.args.json", _visual_structure_args(root))
        write_json(root / "mcp-video-rag-search.args.json", {"bundle_dir": str(root), "query": "", "top_k": 8, "ensure_pack": True, "write": True})
        write_json(root / "mcp-video-rag-service-plan.args.json", {"bundle_dir": str(root), "host": "127.0.0.1", "port": 8781, "write": True})
        write_json(root / "mcp-prepare-transcript-edit-session.args.json", {"bundle_dir": str(root), "write": True})
        write_json(root / "mcp-apply-transcript-edits.args.json", {"bundle_dir": str(root), "edits_json": str(root / "transcript-edits.json"), "write": True})
        write_json(root / "mcp-postprocess-asr-transcript.args.json", {"bundle_dir": str(root), "input_path": "", "target_seconds": 18.0, "max_chars": 180, "punctuation_mode": "readable", "set_corrected": True, "write": True})
        write_json(root / "mcp-readable-transcript-llm-polish.args.json", {"bundle_dir": str(root), "provider_config": {}, "input_json": "", "execute": False, "promote": False, "max_segments_per_batch": 40, "max_prompt_chars": 9000, "max_tokens": 4000, "temperature": 0, "write": True})
        write_json(root / "mcp-agent-readable-transcript-rewrite.args.json", {"bundle_dir": str(root), "input_json": "", "agent_name": "local_agent", "source_path": "", "promote": True, "write": True})
        write_json(root / "mcp-transcript-quality-gate.args.json", {"bundle_dir": str(root), "input_path": "", "min_punctuation_per_1000": 50.0, "max_punctuation_per_1000": 140.0, "write": True})
        write_json(root / "mcp-transcript-source-arbitration.args.json", {"bundle_dir": str(root), "platform_subtitle": "", "subtitle": "", "asr_json": "", "glossary_json": "", "min_confidence": 0.72, "promote": True, "write": True})
        write_json(root / "mcp-term-arbitration-codex.args.json", {"bundle_dir": str(root), "input_json": "", "max_terms": 60, "min_confidence": 0.88, "write": True})
        write_json(root / "mcp-term-arbitration-codex-validate.args.json", {"bundle_dir": str(root), "input_json": str(root / "term-arbitration-codex-result.codex.md"), "min_confidence": 0.88, "write": True})
        write_json(root / "mcp-term-correction-impact-report.args.json", {"bundle_dir": str(root), "min_confidence": 0.88, "write": True})
        write_json(root / "mcp-term-correction-closure.args.json", {"bundle_dir": str(root), "accept_draft": True, "input_json": "", "max_terms": 60, "term_min_confidence": 0.88, "transcript_min_confidence": 0.72, "generate_codex_summary": True, "write": True})
        write_json(root / "mcp-term-correction-closure-codex.args.json", {"bundle_dir": str(root), "accept_draft": False, "input_json": str(root / "term-arbitration-codex-result.codex.md"), "max_terms": 60, "term_min_confidence": 0.88, "transcript_min_confidence": 0.72, "generate_codex_summary": True, "write": True})
        write_json(root / "mcp-term-correction-status.args.json", {"bundle_dir": str(root)})
        write_json(root / "mcp-transcript-semantic-correction-pack.args.json", {"bundle_dir": str(root), "limit": 0, "write": True})
        write_json(root / "mcp-transcript-semantic-correction-codex-draft.args.json", {"bundle_dir": str(root), "input_json": str(root / "transcript-semantic-correction-pack.json"), "min_confidence": 0.88, "write": True})
        write_json(root / "mcp-transcript-semantic-correction-llm-draft.args.json", {"bundle_dir": str(root), "input_json": str(root / "transcript-semantic-correction-pack.json"), "provider_config": {}, "execute": False, "limit": 80, "min_confidence": 0.88, "write": True})
        write_json(root / "mcp-transcript-semantic-candidate-discovery-pack.args.json", {"bundle_dir": str(root), "input_json": str(root / "transcript-semantic-correction-pack.json"), "limit": 40, "write": True})
        write_json(root / "mcp-transcript-semantic-candidate-discovery-llm-draft.args.json", {"bundle_dir": str(root), "input_json": str(root / "transcript-semantic-candidate-discovery-pack.json"), "provider_config": {}, "execute": False, "limit": 40, "write": True})
        write_json(root / "mcp-transcript-semantic-candidate-discovery-codex-draft.args.json", {"bundle_dir": str(root), "input_json": str(root / "transcript-semantic-correction-pack.json"), "limit": 40, "max_suggestions": 40, "write": True})
        write_json(root / "mcp-import-transcript-semantic-candidate-suggestions.args.json", {"bundle_dir": str(root), "input_json": str(root / "transcript-semantic-candidate-suggestions.codex.md"), "write": True})
        write_json(root / "mcp-validate-transcript-semantic-correction.args.json", {"bundle_dir": str(root), "input_json": str(root / "transcript-semantic-correction-result.codex.md"), "min_confidence": 0.88, "write": True})
        write_json(root / "mcp-transcript-semantic-correction-closure.args.json", {"bundle_dir": str(root), "input_json": str(root / "transcript-semantic-correction-result.codex.md"), "min_confidence": 0.88, "auto_apply": False, "refresh_exports": True, "write": True})
        write_json(root / "mcp-transcript-semantic-correction-impact-report.args.json", {"bundle_dir": str(root), "write": True})
        write_json(root / "mcp-transcript-semantic-readable-impact-report.args.json", {"bundle_dir": str(root), "write": True})
        write_json(root / "mcp-transcript-semantic-summary-impact-report.args.json", {"bundle_dir": str(root), "summary_path": "", "baseline_summary_path": "", "write": True})
        write_json(root / "mcp-transcript-semantic-correction-status.args.json", {"bundle_dir": str(root), "write": True})
        write_json(root / "mcp-import-transcript-semantic-review-notes.args.json", {"bundle_dir": str(root), "review_json": str(root / "transcript-semantic-correction-review-notes.json"), "min_confidence": 0.88, "write": True})
        write_json(root / "mcp-transcript-semantic-repair-queue.args.json", {"batch_input": str(root), "output_dir": str(root / "exports"), "target_bundle_count": 1, "limit": 1, "write": True})
        write_json(root / "mcp-transcript-semantic-repair-run.args.json", {"batch_input": str(root), "output_dir": str(root / "exports"), "target_bundle_count": 1, "limit": 1, "execute_safe_actions": False, "max_actions": 0, "max_rounds": 1, "allow_closure": False, "allow_llm": False, "provider_config": {}, "llm_limit": 80, "write": True})
        write_json(root / "mcp-transcript-semantic-batch-review-pack.args.json", {"batch_input": str(root), "output_dir": str(root / "exports"), "target_bundle_count": 1, "limit": 1, "max_candidates_per_bundle": 0, "write": True})
        write_json(root / "mcp-transcript-semantic-batch-codex-review-draft.args.json", {"review_pack_json": str(root / "exports" / "transcript-semantic-batch-review-pack.json"), "output_dir": str(root / "exports"), "write": True})
        write_json(root / "mcp-transcript-semantic-batch-import-review-notes.args.json", {"review_json": str(root / "exports" / "transcript-semantic-batch-review-notes.todo.json"), "output_dir": str(root / "exports"), "min_confidence": 0.88, "write": True})
        write_json(root / "mcp-transcript-semantic-acceptance.args.json", {"bundle_dir": str(root), "output_dir": str(root / "exports"), "write": True})
        write_json(root / "mcp-timeline-alignment-audit.args.json", {"bundle_dir": str(root), "tolerance_seconds": 2.0, "write": True})
        write_json(root / "mcp-tile-result-import-build.args.json", {"bundle_dir": str(root), "results_dir": "", "output_json": str(root / "tile-result-import.json"), "default_source": "tile_result_import_builder", "default_confidence": 0.0, "write": True})
        write_json(root / "mcp-tile-result-merge.args.json", {"bundle_dir": str(root), "input_json": str(root / "tile-result-import.json"), "execute": False, "min_confidence": 0.65, "write": True})
        write_json(root / "mcp-build-smart-summary-input-pack.args.json", {"bundle_dir": str(root), "title": "", "write": True, "max_visual_items": 80})
        write_json(root / "mcp-build-smart-summary-chapters.args.json", {"bundle_dir": str(root), "title": "", "write": True, "target_chapters": 8, "max_visual_items": 120})
        write_json(root / "mcp-prepare-smart-summary-llm-rewrite.args.json", {"bundle_dir": str(root), "provider": "codex_manual", "write": True})
        write_json(root / "mcp-run-smart-summary-llm-rewrite.args.json", {"bundle_dir": str(root), "provider_config": {}, "execute": False, "max_input_chars": 60000, "temperature": 0, "install": True, "write": True})
        write_json(root / "mcp-smart-summary-section-workflow.args.json", {"bundle_dir": str(root), "title": "", "write": True, "target_chapters": 8})
        write_json(root / "mcp-smart-summary-section-editor.args.json", {"bundle_dir": str(root), "write": True})
        write_json(root / "mcp-smart-summary-section-llm-rewrite.args.json", {"bundle_dir": str(root), "provider_config": {}, "execute": False, "target_chapters": 8, "limit": 0, "only_needing_rewrite": False, "max_prompt_chars": 6000, "max_tokens": 1200, "temperature": 0, "install": True, "require_all_sections": True, "write": True})
        write_json(root / "mcp-bilinote-mind-map-prompt-pack.args.json", {"bundle_dir": str(root), "title": "", "max_chars": 5000, "write": True})
        write_json(root / "mcp-online-model-api.args.json", {"model_type": "text_llm", "provider_config": {}, "prompt": "", "input_text": "", "image_paths": [], "audio_path": "", "execute": False, "output_dir": str(root / "exports"), "write": True})
        write_json(root / "mcp-online-model-api-matrix.args.json", {"provider_config": {}, "output_dir": str(root / "exports"), "write": True})
        media_source = _media_path_for_args(root, manifest)
        write_json(root / "mcp-plan-local-asr-service.args.json", {"root": str(root), "media_path": media_source, "provider_config": {}, "model": "", "language": "zh", "prompt": ""})
        write_json(root / "mcp-run-local-asr-service-plan.args.json", {"plan_json": str(root / "transcripts" / "local_asr_service_PLAN_ID" / "local-asr-service-plan.json"), "provider_config": {}, "execute": False, "normalize": True, "allow_remote": False})
        manifest["task_console"] = "task-console.html"
        manifest["video_workbench_html"] = "video-workbench.html"
        manifest["video_workbench_json"] = "video-workbench.json"
        manifest["mcp_video_workbench_args"] = "mcp-video-workbench.args.json"
        manifest["task_console_json"] = "task-console.json"
        manifest["quality_console"] = "quality-console.html"
        manifest["quality_console_json"] = "quality-console.json"
        manifest["model_settings"] = "model-settings.html"
        manifest["model_settings_json"] = "model-settings.json"
        manifest["video_rag_search_markdown"] = "exports/video-rag-search.md"
        manifest["video_rag_service_plan_markdown"] = "exports/video-rag-service-plan.md"
        manifest["mcp_video_rag_search_args"] = "mcp-video-rag-search.args.json"
        manifest["mcp_video_rag_service_plan_args"] = "mcp-video-rag-service-plan.args.json"
        manifest["online_model_api_matrix_json"] = "exports/online-model-api-matrix.json"
        manifest["online_model_api_matrix_markdown"] = "exports/online-model-api-matrix.md"
        manifest["mcp_online_model_api_args"] = "mcp-online-model-api.args.json"
        manifest["mcp_online_model_api_matrix_args"] = "mcp-online-model-api-matrix.args.json"
        manifest["transcript_editor_html"] = "transcript-editor.html"
        manifest["transcript_edit_session_json"] = "transcript-edit-session.json"
        manifest["mcp_prepare_transcript_edit_session_args"] = "mcp-prepare-transcript-edit-session.args.json"
        manifest["mcp_apply_transcript_edits_args"] = "mcp-apply-transcript-edits.args.json"
        manifest["postprocessed_transcript_markdown"] = "asr-transcript-postprocess.md"
        manifest["postprocessed_transcript_json"] = "postprocessed-transcript.json"
        manifest["readable_transcript_json"] = "readable-transcript.json"
        manifest["readable_transcript_srt"] = "readable-transcript.srt"
        manifest["llm_readable_transcript_json"] = "llm-readable-transcript.json"
        manifest["llm_readable_transcript_srt"] = "llm-readable-transcript.srt"
        manifest["llm_readable_transcript_markdown"] = "llm-readable-transcript.md"
        manifest["mcp_postprocess_asr_transcript_args"] = "mcp-postprocess-asr-transcript.args.json"
        manifest["mcp_readable_transcript_llm_polish_args"] = "mcp-readable-transcript-llm-polish.args.json"
        manifest["agent_readable_transcript_json"] = "agent-readable-transcript.json"
        manifest["agent_readable_transcript_markdown"] = "agent-readable-transcript.md"
        manifest["agent_readable_transcript_rewrite_json"] = "agent-readable-transcript-rewrite.json"
        manifest["agent_readable_transcript_rewrite_markdown"] = "agent-readable-transcript-rewrite.md"
        manifest["agent_readable_transcript_task_json"] = "agent-readable-transcript-task.json"
        manifest["agent_readable_transcript_task_markdown"] = "agent-readable-transcript-task.md"
        manifest["transcript_quality_gate_json"] = "transcript-quality-gate.json"
        manifest["transcript_quality_gate_markdown"] = "transcript-quality-gate.md"
        manifest["mcp_agent_readable_transcript_rewrite_args"] = "mcp-agent-readable-transcript-rewrite.args.json"
        manifest["mcp_transcript_quality_gate_args"] = "mcp-transcript-quality-gate.args.json"
        manifest["mcp_plan_local_asr_service_args"] = "mcp-plan-local-asr-service.args.json"
        manifest["mcp_run_local_asr_service_plan_args"] = "mcp-run-local-asr-service-plan.args.json"
        manifest["transcript_source_arbitration_markdown"] = "transcript-source-arbitration.md"
        manifest["term_arbitration_codex_markdown"] = "term-arbitration-codex.md"
        manifest["term_arbitration_codex_prompt_markdown"] = "term-arbitration-codex-prompt.md"
        manifest["term_arbitration_codex_result_codex_markdown"] = "term-arbitration-codex-result.codex.md"
        manifest["term_arbitration_codex_pack_json"] = "term-arbitration-codex-pack.json"
        manifest["term_arbitration_codex_draft_json"] = "term-arbitration-codex-draft.json"
        manifest["term_arbitration_codex_validation_markdown"] = "term-arbitration-codex-validation.md"
        manifest["term_arbitration_codex_validation_json"] = "term-arbitration-codex-validation.json"
        manifest["term_arbitration_glossary_json"] = "term-arbitration-glossary.json"
        manifest["mcp_term_arbitration_codex_args"] = "mcp-term-arbitration-codex.args.json"
        manifest["mcp_term_arbitration_codex_validate_args"] = "mcp-term-arbitration-codex-validate.args.json"
        manifest["term_correction_impact_report_markdown"] = "term-correction-impact-report.md"
        manifest["term_correction_impact_report_json"] = "term-correction-impact-report.json"
        manifest["mcp_term_correction_impact_report_args"] = "mcp-term-correction-impact-report.args.json"
        manifest["term_correction_closure_markdown"] = "term-correction-closure.md"
        manifest["term_correction_closure_json"] = "term-correction-closure.json"
        manifest["mcp_term_correction_closure_args"] = "mcp-term-correction-closure.args.json"
        manifest["mcp_term_correction_closure_codex_args"] = "mcp-term-correction-closure-codex.args.json"
        manifest["mcp_term_correction_status_args"] = "mcp-term-correction-status.args.json"
        manifest["transcript_semantic_correction_pack_json"] = "transcript-semantic-correction-pack.json"
        manifest["transcript_semantic_correction_prompt_markdown"] = "transcript-semantic-correction-prompt.md"
        manifest["transcript_semantic_correction_llm_prompt_markdown"] = "transcript-semantic-correction-llm-prompt.md"
        manifest["transcript_semantic_candidate_discovery_pack_json"] = "transcript-semantic-candidate-discovery-pack.json"
        manifest["transcript_semantic_candidate_discovery_prompt_markdown"] = "transcript-semantic-candidate-discovery-prompt.md"
        manifest["transcript_semantic_candidate_discovery_template_json"] = "transcript-semantic-candidate-discovery-template.json"
        manifest["transcript_semantic_candidate_discovery_llm_prompt_markdown"] = "transcript-semantic-candidate-discovery-llm-prompt.md"
        manifest["transcript_semantic_candidate_suggestions_codex_markdown"] = "transcript-semantic-candidate-suggestions.codex.md"
        manifest["transcript_semantic_candidate_suggestions_llm_markdown"] = "transcript-semantic-candidate-suggestions.llm.md"
        manifest["transcript_semantic_candidate_suggestions_import_json"] = "transcript-semantic-candidate-suggestions-import.json"
        manifest["transcript_semantic_correction_result_codex_markdown"] = "transcript-semantic-correction-result.codex.md"
        manifest["transcript_semantic_correction_result_llm_markdown"] = "transcript-semantic-correction-result.llm.md"
        manifest["transcript_semantic_correction_validation_markdown"] = "transcript-semantic-correction-validation.md"
        manifest["transcript_semantic_correction_closure_markdown"] = "transcript-semantic-correction-closure.md"
        manifest["transcript_semantic_correction_impact_report_markdown"] = "transcript-semantic-correction-impact-report.md"
        manifest["transcript_semantic_correction_readable_impact_markdown"] = "transcript-semantic-readable-impact-report.md"
        manifest["transcript_semantic_summary_impact_markdown"] = "transcript-semantic-summary-impact-report.md"
        manifest["transcript_semantic_correction_status_markdown"] = "transcript-semantic-correction-status.md"
        manifest["mcp_transcript_semantic_correction_pack_args"] = "mcp-transcript-semantic-correction-pack.args.json"
        manifest["mcp_transcript_semantic_correction_codex_draft_args"] = "mcp-transcript-semantic-correction-codex-draft.args.json"
        manifest["mcp_transcript_semantic_correction_llm_draft_args"] = "mcp-transcript-semantic-correction-llm-draft.args.json"
        manifest["mcp_transcript_semantic_candidate_discovery_pack_args"] = "mcp-transcript-semantic-candidate-discovery-pack.args.json"
        manifest["mcp_transcript_semantic_candidate_discovery_llm_draft_args"] = "mcp-transcript-semantic-candidate-discovery-llm-draft.args.json"
        manifest["mcp_transcript_semantic_candidate_discovery_codex_draft_args"] = "mcp-transcript-semantic-candidate-discovery-codex-draft.args.json"
        manifest["mcp_import_transcript_semantic_candidate_suggestions_args"] = "mcp-import-transcript-semantic-candidate-suggestions.args.json"
        manifest["mcp_validate_transcript_semantic_correction_args"] = "mcp-validate-transcript-semantic-correction.args.json"
        manifest["mcp_transcript_semantic_correction_closure_args"] = "mcp-transcript-semantic-correction-closure.args.json"
        manifest["mcp_transcript_semantic_correction_impact_report_args"] = "mcp-transcript-semantic-correction-impact-report.args.json"
        manifest["mcp_transcript_semantic_readable_impact_report_args"] = "mcp-transcript-semantic-readable-impact-report.args.json"
        manifest["mcp_transcript_semantic_summary_impact_report_args"] = "mcp-transcript-semantic-summary-impact-report.args.json"
        manifest["mcp_transcript_semantic_correction_status_args"] = "mcp-transcript-semantic-correction-status.args.json"
        manifest["transcript_semantic_acceptance_json"] = "exports/transcript-semantic-acceptance.json"
        manifest["transcript_semantic_acceptance_markdown"] = "exports/transcript-semantic-acceptance.md"
        manifest["mcp_transcript_semantic_acceptance_args"] = "mcp-transcript-semantic-acceptance.args.json"
        manifest["mcp_import_transcript_semantic_review_notes_args"] = "mcp-import-transcript-semantic-review-notes.args.json"
        manifest["transcript_semantic_repair_queue_json"] = "exports/transcript-semantic-repair-queue.json"
        manifest["transcript_semantic_repair_queue_markdown"] = "exports/transcript-semantic-repair-queue.md"
        manifest["mcp_transcript_semantic_repair_queue_args"] = "mcp-transcript-semantic-repair-queue.args.json"
        manifest["transcript_semantic_repair_run_json"] = "exports/transcript-semantic-repair-run.json"
        manifest["transcript_semantic_repair_run_markdown"] = "exports/transcript-semantic-repair-run.md"
        manifest["mcp_transcript_semantic_repair_run_args"] = "mcp-transcript-semantic-repair-run.args.json"
        manifest["transcript_semantic_batch_review_pack_json"] = "exports/transcript-semantic-batch-review-pack.json"
        manifest["transcript_semantic_batch_review_pack_markdown"] = "exports/transcript-semantic-batch-review-pack.md"
        manifest["transcript_semantic_batch_review_notes_todo_json"] = "exports/transcript-semantic-batch-review-notes.todo.json"
        manifest["transcript_semantic_batch_codex_review_prompt_markdown"] = "exports/transcript-semantic-batch-codex-review-prompt.md"
        manifest["transcript_semantic_batch_codex_review_draft_json"] = "exports/transcript-semantic-batch-review-notes.codex-draft.json"
        manifest["transcript_semantic_batch_codex_review_draft_markdown"] = "exports/transcript-semantic-batch-review-notes.codex-draft.md"
        manifest["transcript_semantic_batch_review_import_json"] = "exports/transcript-semantic-batch-review-import.json"
        manifest["transcript_semantic_batch_review_import_markdown"] = "exports/transcript-semantic-batch-review-import.md"
        manifest["mcp_transcript_semantic_batch_review_pack_args"] = "mcp-transcript-semantic-batch-review-pack.args.json"
        manifest["mcp_transcript_semantic_batch_codex_review_draft_args"] = "mcp-transcript-semantic-batch-codex-review-draft.args.json"
        manifest["mcp_transcript_semantic_batch_import_review_notes_args"] = "mcp-transcript-semantic-batch-import-review-notes.args.json"
        manifest["mcp_transcript_source_arbitration_args"] = "mcp-transcript-source-arbitration.args.json"
        manifest["mcp_export_task_console_args"] = "mcp-export-task-console.args.json"
        manifest["timeline_alignment_audit_json"] = "timeline-alignment-audit.json"
        manifest["timeline_alignment_audit_report"] = "timeline-alignment-audit.md"
        manifest["mcp_timeline_alignment_audit_args"] = "mcp-timeline-alignment-audit.args.json"
        manifest["tile_result_import_report"] = "tile-result-import.md"
        manifest["mcp_tile_result_import_build_args"] = "mcp-tile-result-import-build.args.json"
        manifest["tile_result_merge_report"] = "tile-result-merge.md"
        manifest["mcp_tile_result_merge_args"] = "mcp-tile-result-merge.args.json"
        manifest["smart_summary_input_pack_markdown"] = "exports/smart-summary-input-pack.md"
        manifest["smart_summary_chapters_markdown"] = "exports/smart-summary-chapters.md"
        manifest["smart_summary_course_map_markdown"] = "exports/course-map.md"
        manifest["mcp_build_smart_summary_input_pack_args"] = "mcp-build-smart-summary-input-pack.args.json"
        manifest["mcp_build_smart_summary_chapters_args"] = "mcp-build-smart-summary-chapters.args.json"
        manifest["smart_summary_llm_rewrite_pack_markdown"] = "exports/smart-summary-llm-rewrite-pack.md"
        manifest["smart_summary_llm_rewrite_template_markdown"] = "exports/smart-summary.llm.todo.md"
        manifest["smart_summary_llm_rewrite_status_json"] = "exports/smart-summary-llm-rewrite-status.json"
        manifest["mcp_prepare_smart_summary_llm_rewrite_args"] = "mcp-prepare-smart-summary-llm-rewrite.args.json"
        manifest["mcp_run_smart_summary_llm_rewrite_args"] = "mcp-run-smart-summary-llm-rewrite.args.json"
        manifest["smart_summary_section_workflow_markdown"] = "exports/smart-summary-section-workflow.md"
        manifest["mcp_smart_summary_section_workflow_args"] = "mcp-smart-summary-section-workflow.args.json"
        manifest["smart_summary_section_editor_html"] = "smart-summary-section-editor.html"
        manifest["smart_summary_section_semantic_review_notes_template"] = "exports/smart-summary-section-semantic-review-notes.template.json"
        manifest["bilinote_mind_map_prompt_pack_markdown"] = "exports/bilinote-mind-map-prompt-pack.md"
        manifest["mcp_bilinote_mind_map_prompt_pack_args"] = "mcp-bilinote-mind-map-prompt-pack.args.json"
        manifest["mcp_smart_summary_section_editor_args"] = "mcp-smart-summary-section-editor.args.json"
        manifest["smart_summary_section_llm_rewrite_markdown"] = "exports/smart-summary-section-llm-rewrite.md"
        manifest["smart_summary_section_llm_revisions_json"] = "exports/smart-summary-section-llm-revisions.json"
        manifest["mcp_smart_summary_section_llm_rewrite_args"] = "mcp-smart-summary-section-llm-rewrite.args.json"
        manifest["task_console_refreshed_at"] = result["generated_at"]
        write_json(manifest_path, manifest)

    return result


def _visual_structure_args(root: Path) -> dict[str, Any]:
    ebook_profile = ebook_pipeline_profile()
    return {
        "bundle_dir": str(root),
        "execute_ebook_pipeline": bool(ebook_profile.get("execute_default")),
        "include_routes": list(ebook_profile.get("include_routes") or ["document_visual", "mixed"]),
        "timeout_seconds": int(ebook_profile.get("timeout_seconds") or 300),
        "indexes": [],
        "limit": int(ebook_profile.get("limit") or 0),
    }


def _refresh_known_status(root: Path) -> dict[str, Any]:
    refreshed: dict[str, Any] = {}
    try:
        from .knowledge_coverage import audit_knowledge_coverage

        refreshed["knowledge_coverage"] = audit_knowledge_coverage(root, write=True)
    except Exception as exc:  # noqa: BLE001
        refreshed["knowledge_coverage_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from .bundle_status import bundle_status_report

        refreshed["bundle_status"] = bundle_status_report(root, refresh=False)
    except Exception as exc:  # noqa: BLE001
        refreshed["bundle_status_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from .acceptance_check import acceptance_check

        refreshed["acceptance_check"] = acceptance_check(root, refresh=False, write=True)
    except Exception as exc:  # noqa: BLE001
        refreshed["acceptance_check_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from .review_session import review_closure_status

        refreshed["review_closure_status"] = review_closure_status(root, write=True)
    except Exception as exc:  # noqa: BLE001
        refreshed["review_closure_status_error"] = f"{type(exc).__name__}: {exc}"
    return refreshed



def _safe_transcript_semantic_repair_queue(root: Path, *, write: bool) -> dict[str, Any]:
    try:
        return transcript_semantic_repair_queue(root, output_dir=root / "exports", target_bundle_count=1, limit=1, write=write)
    except Exception as exc:  # noqa: BLE001
        return {
            "schema": "video_knowledge_pipeline.transcript_semantic_repair_queue.error.v1",
            "bundle_dir": str(root),
            "status": "error",
            "ok": False,
            "summary": {"action_required_count": 1, "machine_action_available_count": 0, "human_review_required_count": 1},
            "items": [
                {
                    "bundle_dir": str(root),
                    "action_key": "inspect_bundle",
                    "action_status": "blocked_or_failed",
                    "action_kind": "operator_review_required",
                    "machine_action_available": False,
                    "human_review_required": True,
                    "retry_command": f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-status '{root}'",
                    "reason": f"Failed to build repair queue: {type(exc).__name__}: {exc}",
                    "progress": {"step": 0, "total_steps": 7, "percent": 0},
                }
            ],
            "error": f"{type(exc).__name__}: {exc}",
            "operator_boundary": {"preview_only": True, "does_not_execute_actions": True},
        }


def _safe_transcript_semantic_repair_run(root: Path) -> dict[str, Any]:
    path = root / "exports" / "transcript-semantic-repair-run.json"
    markdown_path = root / "exports" / "transcript-semantic-repair-run.md"
    if not path.exists():
        return {
            "schema": "video_knowledge_pipeline.transcript_semantic_repair_run.missing.v1",
            "status": "missing",
            "json_path": str(path),
            "markdown_path": str(markdown_path),
            "summary": {
                "action_count": 0,
                "executed_count": 0,
                "planned_count": 0,
                "failed_count": 0,
                "operator_required_count": 0,
            },
            "executions": [],
        }
    try:
        data = read_json(path)
        if isinstance(data, dict):
            data.setdefault("json_path", str(path))
            data.setdefault("markdown_path", str(markdown_path))
            return data
    except Exception as exc:  # noqa: BLE001
        return {
            "schema": "video_knowledge_pipeline.transcript_semantic_repair_run.error.v1",
            "status": "error",
            "json_path": str(path),
            "markdown_path": str(markdown_path),
            "summary": {"action_count": 1, "executed_count": 0, "planned_count": 0, "failed_count": 1, "operator_required_count": 0},
            "executions": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "schema": "video_knowledge_pipeline.transcript_semantic_repair_run.invalid.v1",
        "status": "invalid",
        "json_path": str(path),
        "markdown_path": str(markdown_path),
        "summary": {"action_count": 1, "executed_count": 0, "planned_count": 0, "failed_count": 1, "operator_required_count": 0},
        "executions": [],
    }


def _safe_transcript_semantic_batch_review_status(root: Path) -> dict[str, Any]:
    exports = root / "exports"
    pack_path = exports / "transcript-semantic-batch-review-pack.json"
    pack_md_path = exports / "transcript-semantic-batch-review-pack.md"
    todo_path = exports / "transcript-semantic-batch-review-notes.todo.json"
    prompt_path = exports / "transcript-semantic-batch-codex-review-prompt.md"
    draft_path = exports / "transcript-semantic-batch-review-notes.codex-draft.json"
    draft_md_path = exports / "transcript-semantic-batch-review-notes.codex-draft.md"
    import_path = exports / "transcript-semantic-batch-review-import.json"
    import_md_path = exports / "transcript-semantic-batch-review-import.md"
    pack = _read_object(pack_path)
    draft = _read_object(draft_path)
    imported = _read_object(import_path)
    status = "missing_pack"
    if pack:
        status = "pack_ready"
    if draft:
        status = "codex_draft_ready"
    if imported:
        status = str(imported.get("status") or "imported")
    review_count = int(pack.get("review_item_count") or 0) if pack else 0
    by_review_status = draft.get("by_review_status") if isinstance(draft.get("by_review_status"), dict) else {}
    if draft and not by_review_status:
        by_review_status = _count_batch_review_statuses(draft.get("reviews") or [])
    editable_reviews = _batch_editable_reviews(pack, draft)
    return {
        "schema": "video_knowledge_pipeline.task_console.semantic_batch_review_status.v1",
        "status": status,
        "pack_exists": pack_path.exists(),
        "draft_exists": draft_path.exists(),
        "import_exists": import_path.exists(),
        "bundle_count": int(pack.get("bundle_count") or 0) if pack else 0,
        "review_item_count": review_count,
        "todo_review_count": _todo_review_count(todo_path),
        "draft_review_count": int(draft.get("review_count") or len(draft.get("reviews") or [])) if draft else 0,
        "draft_by_review_status": by_review_status,
        "imported_decision_count": int(imported.get("imported_decision_count") or 0) if imported else 0,
        "imported_accepted_decision_count": int(imported.get("accepted_decision_count") or 0) if imported else 0,
        "imported_review_required_count": int(imported.get("review_required_count") or 0) if imported else 0,
        "imported_closure_ready_bundle_count": int(imported.get("closure_ready_bundle_count") or 0) if imported else 0,
        "imported_open_review_bundle_count": int(imported.get("open_review_bundle_count") or 0) if imported else 0,
        "imported_bundle_count": int(imported.get("bundle_count") or 0) if imported else 0,
        "skipped_count": int(imported.get("skipped_count") or 0) if imported else 0,
        "import_by_validation_status": imported.get("by_validation_status") if isinstance(imported.get("by_validation_status"), dict) else {},
        "import_post_next_action_counts": imported.get("post_import_next_action_counts") if isinstance(imported.get("post_import_next_action_counts"), dict) else {},
        "import_next_actions": imported.get("next_actions") if isinstance(imported.get("next_actions"), list) else [],
        "editable_reviews": editable_reviews,
        "editable_review_count": len(editable_reviews),
        "editable_review_truncated": bool(pack and int(pack.get("review_item_count") or 0) > len(editable_reviews)),
        "paths": {
            "review_pack_json": str(pack_path),
            "review_pack_markdown": str(pack_md_path),
            "todo_json": str(todo_path),
            "codex_prompt_markdown": str(prompt_path),
            "codex_draft_json": str(draft_path),
            "codex_draft_markdown": str(draft_md_path),
            "import_json": str(import_path),
            "import_markdown": str(import_md_path),
        },
        "commands": {
            "build_review_pack": f".\\scripts\\video-knowledge.ps1 transcript-semantic-batch-review-pack '{root}' --target-bundle-count 1 --limit 1 --output-dir '{exports}'",
            "codex_review_draft": f".\\scripts\\video-knowledge.ps1 transcript-semantic-batch-codex-review-draft '{pack_path}' --output-dir '{exports}'",
            "import_todo_or_draft": f".\\scripts\\video-knowledge.ps1 transcript-semantic-batch-import-review-notes '{todo_path}' --output-dir '{exports}'",
            "import_codex_draft": f".\\scripts\\video-knowledge.ps1 transcript-semantic-batch-import-review-notes '{draft_path}' --output-dir '{exports}'",
        },
        "operator_boundary": {
            "no_cloud_call": True,
            "does_not_modify_raw_sources": True,
            "import_still_requires_closure": True,
            "codex_draft_is_conservative": True,
        },
    }


def _todo_review_count(path: Path) -> int:
    data = _read_object(path)
    if not data:
        return 0
    rows = data.get("reviews") if isinstance(data.get("reviews"), list) else []
    return len([row for row in rows if isinstance(row, dict)])


def _count_batch_review_statuses(rows: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        key = str(row.get("review_status") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _batch_editable_reviews(pack: dict[str, Any], draft: dict[str, Any], *, limit: int = 80) -> list[dict[str, Any]]:
    pack_items = pack.get("items") if isinstance(pack.get("items"), list) else []
    draft_rows = draft.get("reviews") if isinstance(draft.get("reviews"), list) else []
    draft_by_key = {_batch_review_key(row): row for row in draft_rows if isinstance(row, dict)}
    rows: list[dict[str, Any]] = []
    for item in pack_items:
        if not isinstance(item, dict):
            continue
        draft_row = draft_by_key.get(_batch_review_key(item), {})
        review_status = str(draft_row.get("review_status") or item.get("review_status") or "needs_more_evidence")
        rows.append({
            "review_id": str(item.get("review_id") or draft_row.get("review_id") or ""),
            "bundle_dir": str(item.get("bundle_dir") or draft_row.get("bundle_dir") or ""),
            "bundle_title": str(item.get("bundle_title") or draft_row.get("bundle_title") or ""),
            "candidate_id": str(item.get("candidate_id") or draft_row.get("candidate_id") or ""),
            "correction_type": str(item.get("correction_type") or draft_row.get("correction_type") or "ordinary_word"),
            "risk_level": str(item.get("risk_level") or draft_row.get("risk_level") or "unknown"),
            "time_range": item.get("time_range") or draft_row.get("time_range") or {},
            "original_text": str(item.get("original_text") or draft_row.get("original_text") or ""),
            "suggested_text": str(item.get("suggested_text") or draft_row.get("suggested_text") or ""),
            "context_text": str(item.get("context_text") or ""),
            "evidence": item.get("evidence") if isinstance(item.get("evidence"), list) else [],
            "evidence_ids": item.get("evidence_ids") or draft_row.get("evidence_ids") or [],
            "review_status": review_status,
            "corrected_text": str(draft_row.get("corrected_text") or ""),
            "confidence": draft_row.get("confidence", ""),
            "review_note": str(draft_row.get("review_note") or ""),
        })
        if len(rows) >= int(limit or 80):
            break
    return rows


def _batch_review_key(row: dict[str, Any]) -> str:
    review_id = str(row.get("review_id") or "").strip()
    candidate_id = str(row.get("candidate_id") or "").strip()
    bundle_dir = str(row.get("bundle_dir") or "").strip()
    return review_id or (bundle_dir + "::" + candidate_id) or candidate_id


def _batch_review_time_label(value: Any) -> str:
    if isinstance(value, dict):
        start = _coerce_seconds(value.get("start"))
        end = _coerce_seconds(value.get("end"))
        if start is not None and end is not None:
            return f"{_format_time_seconds(start)} - {_format_time_seconds(end)}"
        if start is not None:
            return _format_time_seconds(start)
        return ""
    return str(value or "").strip()


def _batch_review_start_seconds(value: Any) -> float | None:
    if isinstance(value, dict):
        return _coerce_seconds(value.get("start"))
    text = str(value or "").strip()
    if not text:
        return None
    first = text.split("-", 1)[0].strip()
    return _parse_time_seconds(first)


def _coerce_seconds(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    return _parse_time_seconds(str(value or "").strip())


def _parse_time_seconds(text: str) -> float | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    parts = raw.split(":")
    if len(parts) not in {2, 3}:
        return None
    try:
        values = [float(part) for part in parts]
    except ValueError:
        return None
    if len(values) == 2:
        minutes, seconds = values
        return max(0.0, minutes * 60 + seconds)
    hours, minutes, seconds = values
    return max(0.0, hours * 3600 + minutes * 60 + seconds)


def _format_time_seconds(seconds: float) -> str:
    total_ms = int(round(max(0.0, float(seconds)) * 1000))
    hours, rem = divmod(total_ms, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _semantic_batch_evidence_html(evidence: Any) -> str:
    rows: list[str] = []
    for ev in evidence if isinstance(evidence, list) else []:
        if not isinstance(ev, dict):
            continue
        source = html.escape(str(ev.get("source_type") or "unknown"))
        evidence_id = html.escape(str(ev.get("evidence_id") or ""))
        text = html.escape(str(ev.get("text") or "")[:360])
        path_value = str(ev.get("path") or ev.get("frame_path") or "").strip()
        path_html = '<div class="evidence-path">path: <code>' + html.escape(path_value) + '</code></div>' if path_value else ""
        timeline_index = ev.get("timeline_index")
        confidence = ev.get("confidence")
        meta_parts = []
        if evidence_id:
            meta_parts.append("id=" + evidence_id)
        if timeline_index not in {None, ""}:
            meta_parts.append("timeline=" + html.escape(str(timeline_index)))
        if confidence not in {None, ""}:
            meta_parts.append("conf=" + html.escape(str(confidence)))
        meta = '<div class="muted">' + " | ".join(meta_parts) + '</div>' if meta_parts else ""
        rows.append('<li><code>' + source + '</code> ' + meta + '<div class="snippet">' + text + '</div>' + path_html + '</li>')
    return '<ul class="semantic-evidence-list">' + "".join(rows) + '</ul>' if rows else ""

def _semantic_repair_bridge(root: Path) -> dict[str, Any]:
    status = config_status()
    service_urls = status.get("service_urls") if isinstance(status.get("service_urls"), dict) else {}
    call_url = str(service_urls.get("openclaw_http") or "")
    docker_call_url = str(service_urls.get("openclaw_http_docker") or "")
    return {
        "schema": "video_knowledge_pipeline.task_console.bridge.v1",
        "call_url": call_url,
        "docker_call_url": docker_call_url,
        "tool": "transcript_semantic_repair_run",
        "semantic_repair_run_arguments": {
            "batch_input": str(root),
            "output_dir": str(root / "exports"),
            "target_bundle_count": 1,
            "limit": 1,
            "execute_safe_actions": False,
            "max_actions": 0,
            "allow_closure": False,
            "allow_llm": False,
            "provider_config": {},
            "llm_limit": 80,
            "write": True,
        },
        "operator_boundary": {
            "default_preview_only": True,
            "execute_button_runs_local_safe_actions_only": True,
            "allow_llm": False,
            "allow_closure": False,
            "cloud_calls": "disabled_by_default",
        },
    }


def _semantic_repair_bridge_html(bridge: dict[str, Any]) -> str:
    call_url = html.escape(str(bridge.get("call_url") or ""), quote=True)
    tool = html.escape(str(bridge.get("tool") or "transcript_semantic_repair_run"))
    if not call_url:
        status = "未从统一配置读取到 bridge URL；可手动填写本机 /call URL。"
    else:
        status = f"默认调用：{call_url}"
    return f"""
      <div class="panel bridge-panel">
        <h3>一键 repair-run（本机安全动作）</h3>
        <p class="muted">通过 VKP OpenClaw HTTP bridge 调用 <code style="display:inline;padding:2px 5px">{tool}</code>。Preview 不写入；安全执行只跑本地动作，强制 <code style="display:inline;padding:2px 5px">allow_llm=false</code>、<code style="display:inline;padding:2px 5px">allow_closure=false</code>，不会调用云 LLM。</p>
        <div class="muted">{html.escape(status)}</div>
        <div class="bridge-controls">
          <label>Bridge /call URL<input id="semanticRepairBridgeUrl" value="{call_url}" placeholder="http://127.0.0.1:8931/call"></label>
          <button type="button" onclick="runSemanticRepairViaBridge(false)">预览</button>
          <button type="button" onclick="runSemanticRepairViaBridge(true)">执行下一步安全动作</button>
        </div>
        <pre id="semanticRepairBridgeOutput" class="bridge-log">尚未调用。执行后请刷新 task-console，查看新产物和状态。</pre>
      </div>
    """

def _timeline_alignment_status(root: Path) -> dict[str, Any]:
    audit = _read_object(root / "timeline-alignment-audit.json")
    summary = audit.get("summary") if isinstance(audit.get("summary"), dict) else {}
    issue_counts = summary.get("issue_counts") if isinstance(summary.get("issue_counts"), dict) else {}
    return {
        "available": bool(audit),
        "status": _timeline_alignment_status_value(audit=bool(audit), transcript_available=bool(summary.get("transcript_available")), issue_count=int(summary.get("items_with_issues") or 0)),
        "items": int(summary.get("items") or 0),
        "items_with_issues": int(summary.get("items_with_issues") or 0),
        "missing_asr_overlap": int(summary.get("missing_asr_overlap") or 0),
        "review_start_mismatch": int(summary.get("review_start_mismatch") or 0),
        "tagger_time_conflict": int(summary.get("tagger_time_conflict") or 0),
        "transcript_available": bool(summary.get("transcript_available")),
        "issue_counts": issue_counts,
        "report_path": str(root / "timeline-alignment-audit.md"),
        "json_path": str(root / "timeline-alignment-audit.json"),
    }


def _timeline_alignment_status_value(*, audit: bool, transcript_available: bool, issue_count: int) -> str:
    if not audit:
        return "not_generated"
    if not transcript_available:
        return "needs_input"
    if issue_count:
        return "needs_review"
    return "completed"

def _build_status(root: Path, manifest: dict[str, Any], timeline: list[dict[str, Any]]) -> dict[str, Any]:
    routes: dict[str, int] = {}
    counts = {
        "timeline_items": len(timeline),
        "items_with_transcript": 0,
        "items_with_corrected_transcript": 0,
        "items_with_visual_text": 0,
        "items_with_structured_visual": 0,
        "items_with_visual_understanding": 0,
        "items_with_temporal_understanding": 0,
        "items_with_tagger_annotations": 0,
        "items_with_continuous_tag_weak_evidence": 0,
        "items_with_term_candidates": 0,
        "items_needs_human_review": 0,
    }
    for item in timeline:
        route = str(item.get("visual_route") or "unknown")
        routes[route] = routes.get(route, 0) + 1
        if str(item.get("transcript") or item.get("text") or "").strip():
            counts["items_with_transcript"] += 1
        if str(item.get("corrected_transcript") or item.get("corrected_text") or "").strip():
            counts["items_with_corrected_transcript"] += 1
        if str(item.get("visual_text") or "").strip():
            counts["items_with_visual_text"] += 1
        if _non_empty(item.get("structured_visual")):
            counts["items_with_structured_visual"] += 1
        if visual_evidence_state(item)["consumable"]:
            counts["items_with_visual_understanding"] += 1
        if temporal_evidence_state(item)["consumable"]:
            counts["items_with_temporal_understanding"] += 1
        if _non_empty(item.get("tagger_annotations")) or _non_empty(item.get("tagger_tags")):
            counts["items_with_tagger_annotations"] += 1
        tag_delta = item.get("temporal_tag_delta") if isinstance(item.get("temporal_tag_delta"), dict) else {}
        if str(tag_delta.get("decision") or "") == "coarse_weak_evidence":
            counts["items_with_continuous_tag_weak_evidence"] += 1
        if _non_empty(item.get("term_candidates")):
            counts["items_with_term_candidates"] += 1
        if item.get("needs_human_review") is True:
            counts["items_needs_human_review"] += 1

    acceptance = _read_object(root / "acceptance-check.json")
    bundle_status = _read_object(root / "bundle-status.json")
    coverage = _read_object(root / "knowledge-coverage.json")
    review_closure = _read_object(root / "review-closure-status.json")
    content_status = content_asset_status(root, write=False)
    timeline_alignment = _timeline_alignment_status(root)
    term_correction = _term_correction_status(root)
    semantic_correction = _semantic_correction_status(root, write=False)
    codex_session = _read_object(root / "codex-vision-session-status.json")
    return {
        "title": str(manifest.get("title") or root.name),
        "counts": counts,
        "routes": routes,
        "acceptance_status": str(acceptance.get("status") or acceptance.get("acceptance_status") or ""),
        "bundle_status": str(bundle_status.get("status") or bundle_status.get("bundle_status") or ""),
        "coverage_status": str(coverage.get("status") or coverage.get("coverage_status") or ""),
        "review_open": _review_open_count(review_closure),
        "content_asset_status": content_status.get("status"),
        "content_asset_allowed_as_inspiration": bool(content_status.get("allowed_as_inspiration")),
        "timeline_alignment": timeline_alignment,
        "timeline_alignment_issue_count": int(timeline_alignment.get("items_with_issues") or 0),
        "term_correction": term_correction,
        "semantic_correction": semantic_correction,
        "visual_evidence_levels": evidence_level_counts(timeline),
        "codex_vision_session": codex_session or {
            "phase": "not_prepared",
            "status": "not_prepared",
            "evidence_levels": {"confirmed": 0, "inferred": 0, "unavailable": 0, "not_seen": len(timeline)},
        },
        "exports_present": {
            "knowledge_note": (root / "exports" / "knowledge-note.md").exists(),
            "smart_summary": (root / "exports" / "smart-summary.md").exists(),
            "full_transcript": (root / "exports" / "full-transcript.md").exists(),
            "extraction_audit": (root / "exports" / "extraction-audit.md").exists(),
            "content_material_card": (root / "exports" / "content-material-card.json").exists(),
        },
    }


def _build_commands(root: Path, manifest: dict[str, Any], status: dict[str, Any]) -> list[dict[str, Any]]:
    bundle = _ps_quote(str(root))
    ebook_args = _visual_structure_args(root)
    ebook_routes = ",".join(str(route) for route in ebook_args.get("include_routes") or ["document_visual", "mixed"])
    ebook_limit = int(ebook_args.get("limit") or 0)
    ebook_timeout = int(ebook_args.get("timeout_seconds") or 300)
    ebook_execute_flag = " --execute-ebook-pipeline" if ebook_args.get("execute_ebook_pipeline") else ""
    ebook_command = (
        f".\\scripts\\video-knowledge.ps1 run-visual-structure {bundle}"
        f"{ebook_execute_flag} --include-routes \\\"{ebook_routes}\\\" --limit {ebook_limit} --timeout-seconds {ebook_timeout}"
    )
    commands = [
        _command("autopilot_start", "启动小模型 Autopilot", "execution", "deterministic_controller", f".\\scripts\\video-knowledge.ps1 autopilot-start {bundle}", str(root / "runs")),
        _command("open_review", "打开审核页", "review", "manual_review", "", str(root / str(manifest.get("review_html") or "review.html"))),
        _command("video_workbench", "打开视频知识工作台", "review", "manual_review", f".\\scripts\\video-knowledge.ps1 export-video-workbench {bundle}", str(root / "video-workbench.html")),
        _command("quality_console", "Transcript and summary quality", "review", "safe", f".\\scripts\\video-knowledge.ps1 export-quality-console {bundle}", str(root / "quality-console.html")),
        _command("status", "刷新状态报告", "preview", "safe", f".\\scripts\\video-knowledge.ps1 bundle-status-report {bundle}", _manifest_path(root, manifest, "bundle_status_report")),
        _command("coverage", "刷新覆盖率审计", "preview", "safe", f".\\scripts\\video-knowledge.ps1 audit-knowledge-coverage {bundle}", _manifest_path(root, manifest, "knowledge_coverage_markdown")),
        _command("media_equivalence_space_saving", "媒体等价审计（默认节省空间）", "preview", "local_read_only", "python -m video_knowledge_pipeline.media_equivalence_audit \"<待删除媒体>\" \"<保留媒体>\" --policy space_saving --output-json " + _ps_quote(str(root / "media-equivalence-space-saving.json")) + " --output-markdown " + _ps_quote(str(root / "media-equivalence-space-saving.md")), str(root / "media-equivalence-space-saving.md")),
        _command("media_equivalence_archive_lossless", "媒体等价审计（档案级绝不降质）", "preview", "local_read_only", "python -m video_knowledge_pipeline.media_equivalence_audit \"<待删除媒体>\" \"<保留媒体>\" --policy archive_lossless --output-json " + _ps_quote(str(root / "media-equivalence-archive-lossless.json")) + " --output-markdown " + _ps_quote(str(root / "media-equivalence-archive-lossless.md")), str(root / "media-equivalence-archive-lossless.md")),
        _command("timeline_alignment", "时间轴对齐审计", "preview", "safe", f".\\scripts\\video-knowledge.ps1 timeline-alignment-audit {bundle} --tolerance-seconds 2", str(root / "timeline-alignment-audit.md")),
        _command("triage", "疑难点 triage", "preview", "safe", f".\\scripts\\video-knowledge.ps1 vision-review-triage {bundle}", str(root / "vision-review-triage.md")),
        _command("video_moment_index", "视频片段索引", "preview", "safe", f".\\scripts\\video-knowledge.ps1 video-moment-index {bundle}", str(root / "exports" / "video-moment-index.md")),
        _command("video_moment_search", "片段/术语搜索", "preview", "safe", f".\\scripts\\video-knowledge.ps1 video-moment-index {bundle} --query \"<关键词或疑难点>\"", str(root / "exports" / "video-moment-index.md")),
        _command("long_video_memory_pack", "长视频记忆包", "preview", "safe", f".\\scripts\\video-knowledge.ps1 long-video-memory-pack {bundle}", str(root / "exports" / "long-video-memory-pack.md")),
        _command("video_rag_pack", "视频 RAG 包", "preview", "safe", f".\\scripts\\video-knowledge.ps1 video-rag-pack {bundle} --query \"<问题或术语>\"", str(root / "exports" / "video-rag-pack.md")),
        _command("video_rag_search", "视频 RAG 本地查询", "preview", "safe", f".\\scripts\\video-knowledge.ps1 video-rag-search {bundle} --query \"<问题或术语>\" --top-k 8", str(root / "exports" / "video-rag-search.md")),
        _command("video_rag_service", "视频 RAG 本地服务", "preview", "local_only", f".\\scripts\\video-knowledge.ps1 video-rag-service-plan {bundle} --host 127.0.0.1 --port 8781", str(root / "exports" / "video-rag-service-plan.md")),
        _command("external_capability_pack", "外部能力复用包", "preview", "safe", f".\\scripts\\video-knowledge.ps1 external-capability-pack {bundle}", str(root / "exports" / "external-capability-pack.md")),
        _command("online_model_api_matrix", "在线模型 API 接口矩阵", "preview", "no_cloud_preview", ".\\scripts\\video-knowledge.ps1 online-model-api-matrix --output-dir " + _ps_quote(str(root / "exports")), str(root / "exports" / "online-model-api-matrix.md")),
        _command("vision_queue", "疑难点多模态队列", "preview", "safe", f".\\scripts\\video-knowledge.ps1 vision-review-queue {bundle} --min-score 10 --batch-size 10 --max-items 0", str(root / "vision-review-queue.html")),
        _command("multimodal_sample_review", "多模态抽样标注", "review", "manual_review", f".\\scripts\\video-knowledge.ps1 multimodal-sample-review {bundle} --sample-size 30", str(root / "multimodal-sample-review.html")),
        _command("validate_multimodal_sample_notes", "汇总多模态抽样标注", "review", "safe", f".\\scripts\\video-knowledge.ps1 validate-multimodal-sample-notes {bundle}", str(root / "multimodal-sample-review-summary.md")),
        _command("visual_structure", "图文截图解析", "execution" if ebook_args.get("execute_ebook_pipeline") else "preview", "local_ebook_pipeline" if ebook_args.get("execute_ebook_pipeline") else "safe", ebook_command, str(root / "visual-structure-report.md")),
        _command("screen_text_crops", "屏幕小字 crop 补救", "execution", "local_write", f".\\scripts\\video-knowledge.ps1 run-screen-text-recovery {bundle} --execute-crops --limit 30", str(root / "screen-text-recovery.md")),
        _command("tile_import", "Tile 导入包生成", "preview", "local_only", f".\\scripts\\video-knowledge.ps1 tile-result-import-build {bundle} --results-dir " + _ps_quote(str(root / "tile-results")), str(root / "tile-result-import.md")),
        _command("tile_merge", "Tile 结果合并", "review", "safe_import", f".\\scripts\\video-knowledge.ps1 tile-result-merge {bundle} --input-json " + _ps_quote(str(root / "tile-result-import.json")), str(root / "tile-result-merge.md")),
        _command("vision_preflight", "多模态执行预检", "preview", "required_before_cloud_call", f".\\scripts\\video-knowledge.ps1 vision-execution-preflight {bundle} --semantic-limit 10 --no-temporal", str(root / "vision-execution-preflight.md")),
        _command("codex_vision_semantic", "准备 Codex 单帧视觉包", "review", "interactive_local_bundle_only", f".\\scripts\\video-knowledge.ps1 prepare-codex-vision-session {bundle} --kind semantic", str(root / "codex-vision-session-status.json")),
        _command("codex_vision_temporal", "准备 Codex 连续帧包", "review", "interactive_local_bundle_only", f".\\scripts\\video-knowledge.ps1 prepare-codex-vision-session {bundle} --kind temporal", str(root / "codex-vision-session-status.json")),
        _command("codex_vision_import", "导入 Codex 视觉结果", "review", "safe_atomic_import", f".\\scripts\\video-knowledge.ps1 import-codex-vision-results {bundle} --input-json \"<result.json>\"", str(root / "codex-vision-session-status.json")),
        _command("local_vlm_smoke", "本地 Qwen/InternVL smoke", "preview", "local_only", f".\\scripts\\video-knowledge.ps1 local-vlm-serving-smoke --provider local_qwen_vl --bundle-dir {bundle}", str(root / "local-vlm-serving-smoke.md")),
        _command("semantic_vision", "多模态单帧复核", "execution", "cloud_call_requires_confirmation", f".\\scripts\\video-knowledge.ps1 run-multimodal-frame-analysis {bundle} --execute --limit 10 --confirm-vision-calls <preflight_calls> --confirm-vision-indexes \"<preflight_indexes>\"", str(root / "multimodal-frame-analysis-report.md")),
        _command("volcengine_semantic_batch", "火山单帧批处理", "execution", "visible_powershell_cloud_call", f".\\scripts\\run-volcengine-vision-batch.ps1 {bundle} -Limit 10", str(root / "vision-execution-preflight.md")),
        _command("temporal_groups", "连续片段帧组", "execution", "local_write", f".\\scripts\\video-knowledge.ps1 run-temporal-frame-groups {bundle} --execute --frame-count 8 --limit 5", str(root / "temporal-frame-groups-report.md")),
        _command("temporal_tag_delta", "连续多帧低成本打标", "execution", "local_candidate_weak_evidence", f".\\scripts\\video-knowledge.ps1 run-temporal-tag-delta {bundle} --execute-tagger --frame-count 8 --limit 16", str(root / "temporal-tag-delta.md")),
        _command("temporal_vision", "连续片段多模态复核", "execution", "cloud_call_requires_confirmation", f".\\scripts\\video-knowledge.ps1 run-temporal-visual-analysis {bundle} --execute --frame-count 8 --limit 5 --confirm-vision-calls <preflight_calls> --confirm-vision-indexes \"<preflight_indexes>\"", str(root / "temporal-visual-analysis-report.md")),
        _command("volcengine_temporal_batch", "火山连续片段批处理", "execution", "visible_powershell_cloud_call", f".\\scripts\\run-volcengine-vision-batch.ps1 {bundle} -Temporal -Limit 5 -FrameCount 8", str(root / "vision-execution-preflight.md")),
        _command("cloud_asr_plan", "云 ASR 计划（默认不上传）", "preview", "cloud_asr_preview_only", f".\\scripts\\video-knowledge.ps1 plan-cloud-asr {bundle} <media-path> --model gpt-4o-transcribe", str(root / "transcripts")),
        _command("whisperx_alignment", "WhisperX 精细时间戳计划", "preview", "local_alignment_optional", f".\\scripts\\video-knowledge.ps1 plan-whisperx-alignment {bundle} <media-path> --model large-v3", str(root / "transcripts")),
        _command("postprocess_asr_transcript", "ASR 标点/断句后处理", "execution", "local_only_transcript_cleanup", f".\\scripts\\video-knowledge.ps1 postprocess-asr-transcript {bundle}", str(root / "asr-transcript-postprocess.md")),
        _command("readable_transcript_llm_polish", "LLM 逐字稿标点/断句增强", "execution", "explicit_text_llm_transcript_polish", f".\\scripts\\video-knowledge.ps1 readable-transcript-llm-polish {bundle}", str(root / "readable-transcript-llm-polish.md")),
        _command("agent_readable_transcript_rewrite", "本地 agent 逐字稿可读化", "execution", "local_agent_no_cloud", f".\\scripts\\video-knowledge.ps1 agent-readable-transcript-rewrite {bundle} --agent-name local_agent --promote", str(root / "agent-readable-transcript-rewrite.md")),
        _command("transcript_quality_gate", "逐字稿质量门禁", "preview", "local_quality_gate", f".\\scripts\\video-knowledge.ps1 transcript-quality-gate {bundle}", str(root / "transcript-quality-gate.md")),
        _command("transcript_evidence_correction_pipeline", "证据仲裁转写纠错主链路", "execution", "llm_execute_explicit_auto_apply_explicit", f".\\scripts\\video-knowledge.ps1 transcript-evidence-correction-pipeline {bundle}", str(root / "transcript-evidence-correction-pipeline.md")),
        _command("transcript_source_arbitration", "字幕/ASR 多源仲裁", "preview", "safe", f".\\scripts\\video-knowledge.ps1 transcript-source-arbitration {bundle}", str(root / "transcript-source-arbitration.md")),
        _command("term_arbitration_codex", "Codex 术语/工具名语义仲裁", "review", "codex_llm_substitute_local_prompt", f".\\scripts\\video-knowledge.ps1 term-arbitration-codex {bundle}", str(root / "term-arbitration-codex.md")),
        _command("term_arbitration_codex_accept_draft", "接受高置信 Codex draft", "review", "codex_llm_substitute_high_confidence_only", f".\\scripts\\video-knowledge.ps1 term-arbitration-codex {bundle} --accept-draft", str(root / "term-arbitration-glossary.json")),
        _command("term_arbitration_codex_import", "导入 Codex 术语仲裁结果", "review", "safe_import", f".\\scripts\\video-knowledge.ps1 term-arbitration-codex {bundle} --input-json " + _ps_quote(str(root / "term-arbitration-codex-result.json")), str(root / "term-arbitration-glossary.json")),
        _command("term_arbitration_codex_validate", "预检 Codex 术语回复", "preview", "safe_parse_only", f".\\scripts\\video-knowledge.ps1 validate-term-arbitration-codex-result {bundle} --input-json " + _ps_quote(str(root / "term-arbitration-codex-result.codex.md")), str(root / "term-arbitration-codex-validation.md")),
        _command("term_correction_impact", "术语纠错影响检查", "preview", "safe", f".\\scripts\\video-knowledge.ps1 term-correction-impact-report {bundle}", str(root / "term-correction-impact-report.md")),
        _command("term_correction_status", "术语纠错状态", "preview", "safe", f".\\scripts\\video-knowledge.ps1 term-correction-status {bundle}", str(root / "term-correction-closure.json")),
        _command("term_correction_closure", "术语纠错闭环", "execution", "local_only_high_confidence_terms", f".\\scripts\\video-knowledge.ps1 term-correction-closure {bundle} --accept-draft", str(root / "term-correction-closure.md")),
        _command("term_correction_closure_codex_import", "导入 Codex 术语回复并闭环", "execution", "codex_reviewed_term_closure", f".\\scripts\\video-knowledge.ps1 term-correction-closure {bundle} --input-json " + _ps_quote(str(root / "term-arbitration-codex-result.codex.md")), str(root / "term-correction-closure.md")),
        _command("transcript_semantic_correction_pack", "通用转写语义纠错包", "preview", "safe", f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-pack {bundle}", str(root / "transcript-semantic-correction-pack.json")),
        _command("transcript_semantic_correction_codex_draft", "Codex 本地判读草稿", "review", "codex_llm_substitute_high_confidence_only", f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-codex-draft {bundle}", str(root / "transcript-semantic-correction-result.codex.md")),
        _command("transcript_semantic_correction_llm_draft", "LLM 语义判读计划", "preview", "llm_preview_no_cloud_by_default", f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-llm-draft {bundle} --limit 80", str(root / "transcript-semantic-correction-llm-prompt.md")),
        _command("transcript_semantic_candidate_discovery", "语义错词候选发现 Prompt", "preview", "safe_no_cloud", f".\\scripts\\video-knowledge.ps1 transcript-semantic-candidate-discovery-pack {bundle} --limit 40", str(root / "transcript-semantic-candidate-discovery-prompt.md")),
        _command("transcript_semantic_candidate_discovery_llm", "LLM 候选发现计划", "preview", "llm_preview_no_cloud_by_default", f".\\scripts\\video-knowledge.ps1 transcript-semantic-candidate-discovery-llm-draft {bundle} --limit 40", str(root / "transcript-semantic-candidate-discovery-llm-prompt.md")),
        _command("transcript_semantic_candidate_discovery_codex", "Codex 本地候选发现草稿", "review", "local_codex_no_cloud_suggestions_only", f".\\scripts\\video-knowledge.ps1 transcript-semantic-candidate-discovery-codex-draft {bundle} --limit 40 --max-suggestions 40", str(root / "transcript-semantic-candidate-suggestions.codex.md")),
        _command("import_transcript_semantic_candidate_suggestions", "导入候选发现 suggestions", "review", "safe_import_candidates_only", f".\\scripts\\video-knowledge.ps1 import-transcript-semantic-candidate-suggestions {bundle} --input-json " + _ps_quote(str(root / "transcript-semantic-candidate-suggestions.codex.md")), str(root / "transcript-semantic-candidate-suggestions-import.json")),
        _command("validate_transcript_semantic_correction", "预检通用语义纠错回复", "preview", "safe_parse_only", f".\\scripts\\video-knowledge.ps1 validate-transcript-semantic-correction {bundle} --input-json " + _ps_quote(str(root / "transcript-semantic-correction-result.codex.md")), str(root / "transcript-semantic-correction-validation.md")),
        _command("transcript_semantic_correction_closure", "通用语义纠错闭环", "execution", "codex_reviewed_semantic_closure", f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-closure {bundle} --input-json " + _ps_quote(str(root / "transcript-semantic-correction-result.codex.md")) + " --refresh-exports", str(root / "transcript-semantic-correction-closure.md")),
        _command("transcript_semantic_correction_impact", "通用语义纠错影响检查", "preview", "safe", f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-impact-report {bundle}", str(root / "transcript-semantic-correction-impact-report.md")),
        _command("transcript_semantic_readable_impact", "可读文件纠错影响检查", "preview", "safe", f".\\scripts\\video-knowledge.ps1 transcript-semantic-readable-impact-report {bundle}", str(root / "transcript-semantic-readable-impact-report.md")),
        _command("transcript_semantic_correction_status", "通用语义纠错状态", "preview", "safe", f".\\scripts\\video-knowledge.ps1 transcript-semantic-correction-status {bundle}", str(root / "transcript-semantic-correction-status.md")),
        _command("transcript_semantic_acceptance", "单视频语义纠错验收证明", "preview", "read_only", f".\\scripts\\video-knowledge.ps1 transcript-semantic-acceptance {bundle}", str(root / "exports" / "transcript-semantic-acceptance.md")),
        _command("transcript_semantic_review_notes_import", "导入章节语义复核 notes", "review", "safe_import", f".\\scripts\\video-knowledge.ps1 import-transcript-semantic-review-notes {bundle} --review-json " + _ps_quote(str(root / "transcript-semantic-correction-review-notes.json")), str(root / "transcript-semantic-correction-result.review.md")),
        _command("transcript_semantic_repair_queue", "通用语义纠错重试队列", "preview", "preview_only_no_execution", f".\\scripts\\video-knowledge.ps1 transcript-semantic-repair-queue {bundle} --target-bundle-count 1 --limit 1 --output-dir " + _ps_quote(str(root / "exports")), str(root / "exports" / "transcript-semantic-repair-queue.md")),
        _command("transcript_semantic_repair_run", "执行安全语义纠错队列", "execution", "local_safe_actions_only", f".\\scripts\\video-knowledge.ps1 transcript-semantic-repair-run {bundle} --target-bundle-count 1 --limit 1 --output-dir " + _ps_quote(str(root / "exports")) + " --execute-safe-actions", str(root / "exports" / "transcript-semantic-repair-run.md")),
        _command("transcript_semantic_batch_review_pack", "生成批量语义复核包", "review", "local_no_cloud", f".\\scripts\\video-knowledge.ps1 transcript-semantic-batch-review-pack {bundle} --target-bundle-count 1 --limit 1 --output-dir " + _ps_quote(str(root / "exports")), str(root / "exports" / "transcript-semantic-batch-review-pack.md")),
        _command("transcript_semantic_batch_codex_review_draft", "生成保守 Codex 批量复核草稿", "review", "local_codex_rules_no_cloud", ".\\scripts\\video-knowledge.ps1 transcript-semantic-batch-codex-review-draft " + _ps_quote(str(root / "exports" / "transcript-semantic-batch-review-pack.json")) + " --output-dir " + _ps_quote(str(root / "exports")), str(root / "exports" / "transcript-semantic-batch-review-notes.codex-draft.md")),
        _command("transcript_semantic_batch_import_review_notes", "导入批量语义复核 notes", "review", "safe_import", ".\\scripts\\video-knowledge.ps1 transcript-semantic-batch-import-review-notes " + _ps_quote(str(root / "exports" / "transcript-semantic-batch-review-notes.todo.json")) + " --output-dir " + _ps_quote(str(root / "exports")), str(root / "exports" / "transcript-semantic-batch-review-import.md")),
        _command("transcript_semantic_batch_import_codex_draft", "导入保守 Codex 批量草稿", "review", "safe_import", ".\\scripts\\video-knowledge.ps1 transcript-semantic-batch-import-review-notes " + _ps_quote(str(root / "exports" / "transcript-semantic-batch-review-notes.codex-draft.json")) + " --output-dir " + _ps_quote(str(root / "exports")), str(root / "exports" / "transcript-semantic-batch-review-import.md")),
        _command("resolve_terms", "名词纠错与纠正版逐字稿", "preview", "safe", f".\\scripts\\video-knowledge.ps1 resolve-terms {bundle}", str(root / "term-resolution-report.md")),
        _command("smart_summary_input_pack", "智能总结输入证据包", "preview", "safe", f".\\scripts\\video-knowledge.ps1 build-smart-summary-input-pack {bundle}", str(root / "exports" / "smart-summary-input-pack.md")),
        _command("smart_summary_chapters", "智能总结章节证据包", "preview", "safe", f".\\scripts\\video-knowledge.ps1 build-smart-summary-chapters {bundle}", str(root / "exports" / "smart-summary-chapters.md")),
        _command("smart_summary_llm_rewrite", "准备 LLM/Codex 智能总结改写包", "review", "local_no_cloud", f".\\scripts\\video-knowledge.ps1 prepare-smart-summary-llm-rewrite {bundle}", str(root / "exports" / "smart-summary-llm-rewrite-pack.md")),
        _command("smart_summary_llm_rewrite_run", "LLM 智能总结改写执行预览", "preview", "preview_only_no_cloud_without_execute", f".\\scripts\\video-knowledge.ps1 run-smart-summary-llm-rewrite {bundle}", str(root / "exports" / "smart-summary-llm-run-status.md")),
        _command("smart_summary_section_workflow", "智能总结章节工作流", "preview", "safe", f".\\scripts\\video-knowledge.ps1 smart-summary-section-workflow {bundle}", str(root / "exports" / "smart-summary-section-workflow.md")),
        _command("smart_summary_section_editor", "智能总结章节编辑器", "review", "manual_review", f".\\scripts\\video-knowledge.ps1 smart-summary-section-editor {bundle}", str(root / "smart-summary-section-editor.html")),
        _command("smart_summary_section_llm_rewrite", "章节级 LLM 智能总结改写", "preview", "preview_only_no_cloud_without_execute", f".\\scripts\\video-knowledge.ps1 run-smart-summary-section-llm-rewrite {bundle}", str(root / "exports" / "smart-summary-section-llm-rewrite.md")),
        _command("smart_summary_section_apply", "导入章节修订为智能总结", "review", "safe_import", f".\\scripts\\video-knowledge.ps1 smart-summary-section-apply {bundle} --input-json " + _ps_quote(str(root / "exports" / "smart-summary-section-todo.json")), str(root / "exports" / "smart-summary-section-apply.md")),
        _command("transcript_editor", "转录编辑器", "review", "manual_review", f".\\scripts\\video-knowledge.ps1 prepare-transcript-edit-session {bundle}", str(root / "transcript-editor.html")),
        _command("apply_transcript_edits", "导入转录人工编辑", "review", "manual_review", f".\\scripts\\video-knowledge.ps1 apply-transcript-edits {bundle} --edits-json " + _ps_quote(str(root / "transcript-edits.json")), str(root / "human-corrected-transcript.md")),
        _command("prepare_review", "生成人工复核包", "review", "manual_review", f".\\scripts\\video-knowledge.ps1 prepare-review-session {bundle} --limit 0 --group-by reason", str(root / "review-pack.md")),
        _command("review_closure", "复核关闭进度", "review", "safe", f".\\scripts\\video-knowledge.ps1 review-closure-status {bundle}", str(root / "review-closure-status.md")),
        _command("export", "导出人类可读文件", "export", "safe", f".\\scripts\\video-knowledge.ps1 export-knowledge-note {bundle}", _manifest_path(root, manifest, "knowledge_note_markdown")),
        _command("content_status", "内容素材卡状态", "export", "safe", f".\\scripts\\video-knowledge.ps1 content-asset-status {bundle}", str(root / "exports" / "content-material-card.md")),
        _command("refresh_console", "刷新任务控制台", "preview", "safe", f".\\scripts\\video-knowledge.ps1 export-task-console {bundle}", str(root / "task-console.html")),
    ]
    commands[0]["available"] = Path(commands[0]["artifact_path"]).exists()
    for command in commands[1:]:
        command["available"] = bool(command.get("command"))
    _prioritize_commands(commands, status)
    return commands


def _media_path_for_args(root: Path, manifest: dict[str, Any]) -> str:
    raw = str(manifest.get("media_path") or manifest.get("source_media_path") or "").strip()
    if not raw:
        return ""
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())

def _command(key: str, label: str, phase: str, safety: str, command: str, artifact_path: str) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "phase": phase,
        "safety": safety,
        "command": command,
        "artifact_path": artifact_path,
        "artifact_exists": Path(artifact_path).exists() if artifact_path else False,
    }


def _prioritize_commands(commands: list[dict[str, Any]], status: dict[str, Any]) -> None:
    counts = status.get("counts") if isinstance(status.get("counts"), dict) else {}
    weak_screen_text = int(counts.get("items_with_visual_text") or 0) < int(counts.get("timeline_items") or 0)
    weak_semantic = int(counts.get("items_with_visual_understanding") or 0) < int(counts.get("timeline_items") or 0)
    weak_temporal = int(counts.get("items_with_temporal_understanding") or 0) == 0
    term_status = status.get("term_correction") if isinstance(status.get("term_correction"), dict) else {}
    semantic_status = status.get("semantic_correction") if isinstance(status.get("semantic_correction"), dict) else {}
    semantic_review_required = int(semantic_status.get("review_required_count") or 0)
    term_action = str(term_status.get("next_action_key") or "")
    for command in commands:
        key = command["key"]
        command["recommended"] = key in {"video_workbench", "open_review", "status", "coverage", "triage", "video_moment_index", "video_rag_pack", "video_rag_search", "video_rag_service", "external_capability_pack", "vision_queue", "transcript_editor", "export"}
        if weak_screen_text and key in {"visual_structure", "screen_text_crops", "tile_import", "tile_merge"}:
            command["recommended"] = True
            command["reason"] = "screen_text_or_ebook_coverage_can_improve"
        if weak_semantic and key in {"vision_preflight", "codex_vision_semantic", "semantic_vision", "volcengine_semantic_batch", "vision_queue"}:
            command["recommended"] = True
            command["reason"] = "semantic_visual_understanding_can_improve"
        if weak_temporal and key in {"codex_vision_temporal", "temporal_groups", "temporal_vision", "volcengine_temporal_batch"}:
            command["recommended"] = True
            command["reason"] = "temporal_understanding_absent_or_weak"
        if int(status.get("timeline_alignment_issue_count") or 0) and key in {"timeline_alignment"}:
            command["recommended"] = True
            command["reason"] = "timeline_alignment_issues_need_review"
        if term_action and key in {term_action, "term_correction_closure"}:
            command["recommended"] = True
            command["reason"] = "term_correction_status=" + str(term_status.get("status") or "")
        if semantic_review_required > 0 and key in {"transcript_semantic_batch_review_pack", "transcript_semantic_batch_codex_review_draft", "transcript_semantic_batch_import_review_notes", "transcript_semantic_batch_import_codex_draft"}:
            command["recommended"] = True
            command["reason"] = "semantic_review_required=" + str(semantic_review_required)


def _artifact_links(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [
        ("video_workbench_html", "视频知识工作台"),
        ("review_html", "审核主界面"),
        ("knowledge_coverage_markdown", "覆盖率审计"),
        ("timeline_alignment_audit_report", "时间轴对齐审计"),
        ("tile_result_import_report", "Tile 导入包"),
        ("tile_result_merge_report", "Tile 结果合并"),
        ("acceptance_check", "验收状态"),
        ("bundle_status_report", "Bundle 状态"),
        ("knowledge_note_markdown", "知识笔记"),
        ("knowledge_note_smart_summary_markdown", "智能总结"),
        ("knowledge_note_full_body_markdown", "连续正文"),
        ("knowledge_note_smart_summary_codex_prompt_markdown", "Codex总结提示"),
        ("smart_summary_llm_rewrite_pack_markdown", "LLM/Codex 改写包"),
        ("smart_summary_llm_rewrite_template_markdown", "LLM/Codex 改写模板"),
        ("smart_summary_llm_rewrite_status_json", "LLM/Codex 改写状态"),
        ("smart_summary_section_workflow_markdown", "智能总结章节工作流"),
        ("smart_summary_section_editor_html", "智能总结章节编辑器"),
        ("smart_summary_section_semantic_review_notes_template", "章节语义复核 notes 模板"),
        ("smart_summary_section_apply_markdown", "章节修订导入"),
        ("knowledge_note_transcript_markdown", "逐字稿"),
        ("codex_vision_session_prompt_markdown", "Codex 视觉 Prompt"),
        ("codex_vision_session_result_template_json", "Codex 视觉结果模板"),
        ("codex_vision_session_raw_result_json", "Codex 视觉原始结果"),
        ("codex_vision_session_import_validation_json", "Codex 视觉导入验证"),
        ("vision_analysis_runs", "视觉执行审计"),
        ("vision_restore_plan_markdown", "视觉恢复计划"),
        ("knowledge_note_extraction_audit_markdown", "提取审计"),
        ("task_console", "任务控制台"),
        ("model_settings", "模型 API 设置"),
        ("agent_readable_transcript_rewrite_markdown", "本地 agent 逐字稿可读化"),
        ("agent_readable_transcript_task_markdown", "Agent 逐字稿改写任务包"),
        ("transcript_quality_gate_markdown", "逐字稿质量门禁"),
        ("mcp_plan_local_asr_service_args", "Speaches 本地 ASR 服务计划 MCP Args"),
        ("mcp_run_local_asr_service_plan_args", "Speaches 本地 ASR 服务执行 MCP Args"),
        ("mcp_agent_readable_transcript_rewrite_args", "Agent 逐字稿改写 MCP Args"),
        ("mcp_transcript_quality_gate_args", "逐字稿质量门禁 MCP Args"),
        ("transcript_source_arbitration_markdown", "字幕/ASR 多源仲裁"),
        ("term_arbitration_codex_markdown", "Codex 术语仲裁"),
        ("term_arbitration_codex_prompt_markdown", "Codex 术语仲裁 Prompt"),
        ("term_arbitration_codex_result_codex_markdown", "Codex 术语回复草稿"),
        ("term_arbitration_codex_validation_markdown", "Codex 术语回复预检"),
        ("term_arbitration_codex_validation_json", "Codex 术语回复预检 JSON"),
        ("term_arbitration_glossary_json", "术语仲裁 Glossary"),
        ("term_correction_impact_report_markdown", "术语纠错影响"),
        ("term_correction_closure_markdown", "术语纠错闭环"),
        ("mcp_term_arbitration_codex_validate_args", "Codex 术语预检 MCP Args"),
        ("mcp_term_correction_status_args", "术语纠错状态 MCP Args"),
        ("mcp_term_correction_closure_codex_args", "导入 Codex 术语回复 MCP Args"),
        ("transcript_semantic_correction_prompt_markdown", "通用语义纠错 Prompt"),
        ("transcript_semantic_correction_llm_prompt_markdown", "通用语义纠错 LLM Prompt"),
        ("transcript_semantic_candidate_discovery_prompt_markdown", "语义错词候选发现 Prompt"),
        ("transcript_semantic_candidate_discovery_pack_json", "语义错词候选发现包 JSON"),
        ("transcript_semantic_candidate_discovery_llm_prompt_markdown", "语义错词候选发现 LLM Prompt"),
        ("transcript_semantic_candidate_suggestions_llm_markdown", "语义错词候选发现 LLM Suggestions"),
        ("transcript_semantic_candidate_suggestions_codex_markdown", "语义错词候选发现 Codex Suggestions"),
        ("transcript_semantic_candidate_suggestions_import_json", "语义错词候选导入结果 JSON"),
        ("mcp_transcript_semantic_candidate_discovery_pack_args", "语义错词候选发现 MCP Args"),
        ("mcp_transcript_semantic_candidate_discovery_llm_draft_args", "语义错词候选发现 LLM MCP Args"),
        ("mcp_transcript_semantic_candidate_discovery_codex_draft_args", "语义错词候选发现 Codex MCP Args"),
        ("mcp_import_transcript_semantic_candidate_suggestions_args", "语义错词候选导入 MCP Args"),
        ("transcript_semantic_correction_pack_json", "通用语义纠错证据包"),
        ("transcript_semantic_correction_result_codex_markdown", "通用语义纠错 Codex 回复"),
        ("transcript_semantic_correction_result_llm_markdown", "通用语义纠错 LLM 回复"),
        ("transcript_semantic_correction_validation_markdown", "通用语义纠错预检"),
        ("transcript_semantic_correction_closure_markdown", "通用语义纠错闭环"),
        ("transcript_semantic_correction_impact_report_markdown", "通用语义纠错影响"),
        ("transcript_semantic_correction_readable_impact_markdown", "通用语义纠错可读文件影响"),
        ("transcript_semantic_correction_status_markdown", "通用语义纠错状态"),
        ("transcript_semantic_repair_queue_markdown", "通用语义纠错重试队列"),
        ("transcript_semantic_repair_queue_json", "通用语义纠错重试队列 JSON"),
        ("transcript_semantic_repair_run_markdown", "通用语义纠错安全执行"),
        ("transcript_semantic_repair_run_json", "通用语义纠错安全执行 JSON"),
        ("transcript_semantic_batch_review_pack_markdown", "通用语义纠错批量复核包"),
        ("transcript_semantic_batch_review_pack_json", "通用语义纠错批量复核包 JSON"),
        ("transcript_semantic_batch_review_notes_todo_json", "通用语义纠错批量复核 Todo"),
        ("transcript_semantic_batch_codex_review_prompt_markdown", "通用语义纠错批量 Codex Prompt"),
        ("transcript_semantic_batch_codex_review_draft_markdown", "通用语义纠错批量 Codex 草稿"),
        ("transcript_semantic_batch_codex_review_draft_json", "通用语义纠错批量 Codex 草稿 JSON"),
        ("transcript_semantic_batch_review_import_markdown", "通用语义纠错批量导入结果"),
        ("transcript_semantic_batch_review_import_json", "通用语义纠错批量导入结果 JSON"),
        ("transcript_evidence_correction_pipeline_markdown", "证据仲裁转写纠错主链路"),
        ("mcp_transcript_evidence_correction_pipeline_args", "证据仲裁纠错 MCP Args"),
        ("mcp_transcript_semantic_correction_pack_args", "通用语义纠错包 MCP Args"),
        ("mcp_transcript_semantic_correction_codex_draft_args", "Codex 语义草稿 MCP Args"),
        ("mcp_transcript_semantic_correction_llm_draft_args", "LLM 语义草稿 MCP Args"),
        ("mcp_validate_transcript_semantic_correction_args", "通用语义纠错预检 MCP Args"),
        ("mcp_transcript_semantic_correction_closure_args", "通用语义纠错闭环 MCP Args"),
        ("mcp_transcript_semantic_readable_impact_report_args", "可读文件影响 MCP Args"),
        ("transcript_editor_html", "转录编辑器"),
        ("vision_review_queue_html", "疑难点多模态队列"),
        ("run_artifact_registry_report", "任务产物索引"),
        ("multimodal_sample_review_html", "多模态抽样标注"),
        ("multimodal_sample_review_summary_report", "多模态抽样汇总"),
        ("human_sample_eval_report", "人工抽样质量评估"),
        ("video_moment_index_markdown", "视频片段索引"),
        ("long_video_memory_pack_markdown", "长视频记忆包"),
        ("video_rag_pack_markdown", "视频 RAG 包"),
        ("video_rag_search_markdown", "视频 RAG 查询"),
        ("video_rag_service_plan_markdown", "视频 RAG 服务计划"),
        ("external_capability_pack_markdown", "外部能力复用包"),
        ("bilinote_mind_map_prompt_pack_markdown", "BiliNote 脑图 Prompt 包"),
    ]
    artifacts: list[dict[str, Any]] = []
    for key, label in keys:
        path = _manifest_path(root, manifest, key)
        if path:
            artifacts.append({"key": key, "label": label, "path": path, "exists": Path(path).exists()})
    return artifacts



def _load_run_registry_for_console(root: Path) -> dict[str, Any]:
    path = root / "run-artifact-registry.json"
    if not path.exists():
        return {
            "schema": "video_knowledge_pipeline.run_artifact_registry.empty.v1",
            "bundle_dir": str(root),
            "run_count": 0,
            "status_counts": {},
            "runs": [],
            "paths": {"json": str(path), "markdown": str(root / "run-artifact-registry.md")},
        }
    value = _read_object(path)
    if not value:
        return {
            "schema": "video_knowledge_pipeline.run_artifact_registry.unreadable.v1",
            "bundle_dir": str(root),
            "run_count": 0,
            "status_counts": {},
            "runs": [],
            "paths": {"json": str(path), "markdown": str(root / "run-artifact-registry.md")},
            "error": "run-artifact-registry.json exists but could not be read as an object",
        }
    runs = [row for row in (value.get("runs") or []) if isinstance(row, dict)]
    return {
        "schema": value.get("schema", ""),
        "bundle_dir": value.get("bundle_dir", str(root)),
        "generated_at": value.get("generated_at", ""),
        "run_count": int(value.get("run_count") or len(runs)),
        "status_counts": value.get("status_counts") if isinstance(value.get("status_counts"), dict) else {},
        "runs": runs,
        "paths": value.get("paths") if isinstance(value.get("paths"), dict) else {"json": str(path), "markdown": str(root / "run-artifact-registry.md")},
        "error": value.get("error", ""),
    }

def _model_batches_for_bundle(root: Path) -> dict[str, Any]:
    try:
        result = list_consented_model_batches(MODEL_BATCH_PROJECT_ROOT, limit=200)
    except Exception as exc:
        return {
            "schema": "video_knowledge_pipeline.consented_model_batch_list.v1",
            "ok": False,
            "count": 0,
            "items": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    matched = []
    for item in result.get("items") or []:
        if not isinstance(item, dict):
            continue
        try:
            bundle = Path(str(item.get("bundle_dir") or "")).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if bundle == root:
            matched.append(item)
    return {**result, "count": len(matched), "items": matched}



QUEUE_GROUPS = [
    ("asr_transcript", "ASR / 转写", "本地 ASR、字幕导入、转写纠错和人工转录编辑。"),
    ("document_ocr", "图文 OCR / ebook", "ebook_markdown_pipeline、屏幕文字恢复、高分辨率 tile 证据。"),
    ("vision", "多模态复核", "疑难点 triage、单帧/连续片段视觉 API 或本地 VLM 复核。"),
    ("timeline_rag", "时间轴 / RAG", "时间错位审计、片段索引、VideoRAG 本地检索。"),
    ("summary_export", "总结 / 导出", "Codex/LLM 智能总结、知识笔记、内容素材卡导出。"),
    ("review", "人工审核", "review pack、review closure、人工修正导入。"),
    ("other", "其他任务", "未归类但已登记的本地 run。"),
]

ACTION_STATUSES = {"needs_retry", "needs_review", "needs_execution", "needs_input", "failed", "error", "blocked"}

QUEUE_GROUP_TOKENS = {
    "asr_transcript": (
        "asr",
        "transcript",
        "transcribe",
        "whisper",
        "whisperx",
        "sensevoice",
        "funasr",
        "subtitle",
        "caption",
        "arbitration",
        "source_arbitration",
        "transcript_source_arbitration",
        "correction",
        "transcript_correction",
        "term_correction_impact_report",
        "term_correction_impact",
        "transcript_editor",
    ),
    "document_ocr": (
        "visual-structure",
        "visual_structure",
        "visual_structure_ebook",
        "ebook",
        "ocr",
        "screen-text",
        "screen_text",
        "screen_text_recovery",
        "tile",
        "tile_result",
        "tile_result_import",
        "tile_result_merge",
        "high-res",
        "high_res",
        "high_res_tile",
        "document_visual",
    ),
    "vision": (
        "vision",
        "vision_review",
        "vision_review_queue",
        "multimodal",
        "multimodal_frame",
        "temporal_visual",
        "temporal_sequence",
        "vlm",
        "local_vlm",
        "provider",
        "frame_analysis",
        "visual_understanding",
    ),
    "timeline_rag": (
        "timeline",
        "alignment",
        "timeline_alignment",
        "moment",
        "video_moment",
        "rag",
        "video_rag",
        "memory",
        "long_video_memory",
        "recapture",
        "frame_recapture",
        "supplemental_frame",
    ),
    "summary_export": (
        "summary",
        "smart_summary",
        "section_workflow",
        "section_editor",
        "section_apply",
        "export",
        "knowledge",
        "content",
        "content_asset",
        "material_card",
        "capability_pack",
        "external_capability",
        "mind_map",
        "prompt_pack",
        "bilinote_mind_map",
    ),
    "review": (
        "review",
        "closure",
        "review_closure",
        "review_pack",
        "human_review",
        "human_sample_eval",
        "sample_review",
        "impact_report",
    ),
}


ASR_TRANSCRIPT_SUBQUEUES = [
    (
        "local_asr",
        "本地 ASR 执行",
        "FunASR/SenseVoice、faster-whisper、WhisperX 和 normalized transcript 落盘任务。",
        ("funasr", "sensevoice", "faster_whisper", "faster-whisper", "whisperx", "local_asr", "asr_runner", "asr_run", "funasr_python_runner"),
    ),
    (
        "asr_env",
        "ASR 环境 / 模型门禁",
        "asr-env-status、模型缓存、CUDA/GPU、ffmpeg 和本地 ASR 环境检查。",
        ("asr_env", "asr-env", "env_status", "model_ready", "model_not_ready", "cuda", "gpu", "ffmpeg", "install-local-asr"),
    ),
    (
        "subtitle_import",
        "字幕 / 平台字幕",
        "平台字幕、自带字幕、caption/subtitle 导入和 sidecar 字幕整理。",
        ("subtitle", "caption", "platform_subtitle", "bilibili_subtitle", "subtitle_import", "subtitle_sidecar", "caption_import"),
    ),
    (
        "source_arbitration",
        "来源仲裁",
        "ASR、自带字幕、网页简介和其他 transcript source 的冲突仲裁。",
        ("transcript_source_arbitration", "source_arbitration", "asr_source", "subtitle_source", "arbitration"),
    ),
    (
        "transcript_correction",
        "术语纠错 / 纠正版转写",
        "术语词典、错词修正、corrected transcript 和 correction pack。",
        ("transcript_correction", "correction_pack", "corrected_transcript", "term", "glossary", "terminology", "text_correction"),
    ),
    (
        "term_arbitration",
        "Codex 术语/工具名仲裁",
        "Codex-first 语义仲裁包、工具名上下文判断、glossary 导入。",
        ("term_arbitration_codex", "term-arbitration-codex", "codex terminology", "tool name", "glossary_import", "term_arbitration_codex_accept_draft", "accept-draft", "high-confidence draft", "codex_substitute_local_draft"),
    ),
    (
        "term_impact",
        "术语影响检查",
        "术语纠错影响报告，检查高置信术语是否进入纠正版逐字稿和最终人类可读导出。",
        ("term_correction_impact", "term_correction_impact_report", "impact_report", "final_export_alias", "source_alias", "term_correction_closure"),
    ),
    (
        "transcript_editor",
        "人工转录编辑",
        "transcript editor、编辑会话、人工改稿和 apply transcript edits。",
        ("transcript_editor", "transcript_edit", "transcript_edit_session", "apply_transcript", "apply_transcript_edits", "manual_transcript"),
    ),
    (
        "other_asr_transcript",
        "其他 ASR/转写",
        "未能细分到本地 ASR、环境门禁、字幕导入、来源仲裁、术语纠错或人工编辑的转写 run。",
        (),
    ),
]

DOCUMENT_OCR_SUBQUEUES = [
    (
        "ebook",
        "ebook / 整帧图文",
        "visual-structure 和 ebook_markdown_pipeline 的整帧图文解析任务。",
        ("visual_structure", "visual-structure", "ebook", "document_visual"),
    ),
    (
        "screen_text_crop",
        "屏幕文字 crop",
        "screen-text recovery、局部 crop 和轻量 OCR 补救任务。",
        ("screen_text", "screen-text", "crop", "screen_text_recovery", "ocr_backfill"),
    ),
    (
        "high_res_tile",
        "高分辨率 tile",
        "high-res tile 计划、tile 结果导入、tile merge 和 pending tile 重试。",
        ("tile", "tile_result", "tile_result_import", "tile_result_merge", "high_res", "high-res", "high_res_tile"),
    ),
    (
        "other_document_ocr",
        "其他 OCR",
        "未能细分到 ebook、crop 或 tile 的图文 OCR run。",
        (),
    ),
]


VISION_SUBQUEUES = [
    (
        "review_triage",
        "疑难点队列",
        "vision review triage、候选帧/片段队列和人工/模型复核入口。",
        ("vision_review", "vision_review_queue", "triage", "review_queue"),
    ),
    (
        "semantic_frame",
        "单帧多模态",
        "semantic frame、多模态单帧复核和 visual understanding 任务。",
        ("multimodal_frame", "semantic", "semantic_frame", "frame_analysis", "visual_understanding", "single_frame"),
    ),
    (
        "temporal_sequence",
        "连续片段多模态",
        "temporal visual、frame group 和连续变化片段复核任务。",
        ("temporal_visual", "temporal_sequence", "temporal", "frame_group", "frame-groups"),
    ),
    (
        "provider_smoke",
        "Provider / 预检",
        "vision provider smoke、provider matrix 和云调用 preflight 状态。",
        ("provider", "provider_smoke", "provider_matrix", "vision_provider", "preflight", "vision_execution_preflight"),
    ),
    (
        "local_vlm",
        "本地 VLM",
        "local VLM serving smoke、本地 Qwen/InternVL/LLaVA adapter 状态。",
        ("local_vlm", "vlm_serving", "qwen", "internvl", "llava"),
    ),
    (
        "other_vision",
        "其他视觉任务",
        "未能细分到疑难队列、单帧、连续片段、provider 或本地 VLM 的视觉 run。",
        (),
    ),
]


TIMELINE_RAG_SUBQUEUES = [
    (
        "timeline_alignment",
        "时间轴对齐",
        "timeline alignment audit、ASR/抽帧/打标时间错位和时间戳修正。",
        ("timeline_alignment", "alignment_audit", "timeline_alignment_audit", "time_alignment", "timestamp_alignment"),
    ),
    (
        "moment_index",
        "片段索引 / 时间定位",
        "video moment index、片段搜索和可跳转时间点索引。",
        ("video_moment", "moment_index", "moment_search", "video_moment_index", "video_moment_search"),
    ),
    (
        "video_rag",
        "VideoRAG / 检索",
        "video-rag pack、search、HTTP service plan 和本地视频检索入口。",
        ("video_rag", "video-rag", "rag_pack", "rag_search", "rag_service", "video_rag_search", "video_rag_service"),
    ),
    (
        "long_video_memory",
        "长视频 memory",
        "long-video memory pack、分层课程记忆和跨段上下文索引。",
        ("long_video_memory", "long-video-memory", "memory_pack", "course_memory", "long_context"),
    ),
    (
        "recapture",
        "补帧 / 重采样",
        "frame recapture、supplemental frame sampling 和疑难片段补帧。",
        ("recapture", "frame_recapture", "supplemental_frame", "supplemental_frame_sampling", "resample", "sampling"),
    ),
    (
        "other_timeline_rag",
        "其他时间轴/RAG",
        "未能细分到时间轴对齐、片段索引、VideoRAG、长视频 memory 或补帧的 run。",
        (),
    ),
]

SUMMARY_EXPORT_SUBQUEUES = [
    (
        "summary_input",
        "总结输入包",
        "smart-summary input pack、章节证据包和长视频 memory 输入准备。",
        ("smart_summary_input", "input_pack", "chapter_pack", "smart_summary_chapter", "long_video_memory", "course_map", "mind_map", "mind_map_prompt", "prompt_pack", "bilinote_mind_map"),
    ),
    (
        "section_workflow",
        "章节工作流",
        "smart-summary section workflow、章节编辑器和章节修订 TODO。",
        ("section_workflow", "section_editor", "chapter_workflow", "section_todo"),
    ),
    (
        "section_apply",
        "章节修订导入",
        "smart-summary section apply、Codex 改写结果安装和质量门禁。",
        ("section_apply", "smart_summary_codex", "codex", "quality", "apply"),
    ),
    (
        "knowledge_export",
        "知识笔记导出",
        "knowledge-note、full-transcript、smart-summary 和 extraction audit 导出。",
        ("knowledge_note", "full_transcript", "smart_summary", "extraction_audit", "export"),
    ),
    (
        "content_candidate",
        "内容素材候选",
        "content candidate、content material card、external capability pack 和素材交接。",
        ("content_candidate", "content_asset", "material_card", "capability_pack", "external_capability", "handoff"),
    ),
    (
        "other_summary_export",
        "其他总结/导出",
        "未能细分到输入包、章节工作流、章节导入、知识导出或内容素材的 summary/export run。",
        (),
    ),
]

REVIEW_SUBQUEUES = [
    (
        "review_pack",
        "复核包 / notes",
        "prepare-review-session、review pack、review notes 校验和导入准备。",
        ("prepare_review", "prepare-review", "review_pack", "review_session", "review_notes", "validate_review"),
    ),
    (
        "transcript_arbitration",
        "字幕仲裁复核",
        "transcript arbitration 低置信冲突、字幕/ASR 来源仲裁和人工确认。",
        ("transcript_arbitration", "arbitration_review", "transcript_review", "low_confidence_conflict"),
    ),
    (
        "sample_eval",
        "抽样评估",
        "multimodal sample review、human sample eval 和 impact report。",
        ("sample_review", "multimodal_sample_review", "scene_candidate_review", "human_sample_eval", "impact_report", "sample_eval"),
    ),
    (
        "closure_status",
        "复核关闭进度",
        "review closure status、closure audit 和 open/closed 进度。",
        ("review_closure", "closure_status", "review_closure_status", "closure"),
    ),
    (
        "manual_import",
        "人工导入",
        "human review import、人工修正写回和安全导入。",
        ("human_review", "manual_review", "human_import", "safe_import", "apply_review", "apply_notes"),
    ),
    (
        "other_review",
        "其他审核",
        "未能细分到复核包、字幕仲裁、抽样评估、关闭状态或人工导入的审核 run。",
        (),
    ),
]

def _build_processing_queue(root: Path, registry: dict[str, Any]) -> dict[str, Any]:
    runs = registry.get("runs") if isinstance(registry.get("runs"), list) else []
    hydrated = [_hydrate_run_row(root, row) for row in runs if isinstance(row, dict)]
    groups = {key: {"key": key, "label": label, "description": desc, "runs": [], "status_counts": {}, "failed_count": 0, "action_required": 0} for key, label, desc in QUEUE_GROUPS}
    for run in hydrated:
        key = _run_queue_group(run)
        group = groups.setdefault(key, {"key": key, "label": key, "description": "", "runs": [], "status_counts": {}, "failed_count": 0, "action_required": 0})
        status = str(run.get("status") or "unknown")
        group["status_counts"][status] = int(group["status_counts"].get(status, 0)) + 1
        group["failed_count"] = int(group.get("failed_count") or 0) + int(run.get("failed_count") or 0)
        if _run_needs_action(run):
            group["action_required"] = int(group.get("action_required") or 0) + 1
        group["runs"].append(run)
    queue_groups = []
    total_action_required = 0
    for key, label, desc in QUEUE_GROUPS:
        group = groups[key]
        group_runs = group.get("runs") if isinstance(group.get("runs"), list) else []
        action_required = int(group.get("action_required") or 0)
        total_action_required += action_required
        group["run_count"] = len(group_runs)
        group["status"] = "empty" if not group_runs else ("action_required" if action_required else "ready")
        action_runs = [run for run in group_runs if _run_needs_action(run)] or group_runs
        group["retry_commands"] = _group_retry_commands(action_runs, limit=3)
        group["next_actions"] = _group_next_actions(action_runs)
        group["failed_items_preview"] = _group_failed_items(action_runs)
        if key == "asr_transcript":
            group["subqueues"] = _asr_transcript_subqueues(group_runs)
        if key == "document_ocr":
            group["subqueues"] = _document_ocr_subqueues(group_runs)
        if key == "vision":
            group["subqueues"] = _vision_subqueues(group_runs)
        if key == "timeline_rag":
            group["subqueues"] = _timeline_rag_subqueues(group_runs)
        if key == "summary_export":
            group["subqueues"] = _summary_export_subqueues(group_runs)
        if key == "review":
            group["subqueues"] = _review_subqueues(group_runs)
        group["runs"] = group_runs[:6]
        queue_groups.append(group)
    return {
        "schema": "video_knowledge_pipeline.task_processing_queue.v1",
        "generated_from": "run_artifact_registry",
        "run_count": len(hydrated),
        "action_required_count": total_action_required,
        "groups": queue_groups,
        "operator_boundary": {
            "local_only": True,
            "no_process_started": True,
            "no_cloud_call": True,
            "purpose": "Summarise registered VKP runs into operator-visible retry/review queues.",
        },
    }


def _asr_transcript_subqueues(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {
        key: {"key": key, "label": label, "description": desc, "runs": [], "status_counts": {}, "failed_count": 0, "action_required": 0}
        for key, label, desc, _tokens in ASR_TRANSCRIPT_SUBQUEUES
    }
    for run in runs:
        key = _asr_transcript_subqueue_key(run)
        group = groups.setdefault(key, {"key": key, "label": key, "description": "", "runs": [], "status_counts": {}, "failed_count": 0, "action_required": 0})
        status = str(run.get("status") or "unknown")
        group["status_counts"][status] = int(group["status_counts"].get(status, 0)) + 1
        group["failed_count"] = int(group.get("failed_count") or 0) + int(run.get("failed_count") or 0)
        if _run_needs_action(run):
            group["action_required"] = int(group.get("action_required") or 0) + 1
        group["runs"].append(run)
    out: list[dict[str, Any]] = []
    for key, label, desc, _tokens in ASR_TRANSCRIPT_SUBQUEUES:
        group = groups[key]
        group_runs = group.get("runs") if isinstance(group.get("runs"), list) else []
        action_runs = [run for run in group_runs if _run_needs_action(run)] or group_runs
        group["run_count"] = len(group_runs)
        group["status"] = "empty" if not group_runs else ("action_required" if int(group.get("action_required") or 0) else "ready")
        group["retry_commands"] = _group_retry_commands(action_runs, limit=2)
        group["next_actions"] = _group_next_actions(action_runs)[:2]
        group["failed_items_preview"] = _group_failed_items(action_runs)[:4]
        group["run_ids"] = [str(run.get("run_id") or run.get("title") or "") for run in group_runs[:4] if str(run.get("run_id") or run.get("title") or "").strip()]
        out.append(group)
    return out


def _asr_transcript_subqueue_key(run: dict[str, Any]) -> str:
    text_parts = [str(run.get(key) or "") for key in ("run_type", "run_id", "title", "summary")]
    for item in run.get("failed_items_preview") or []:
        if isinstance(item, dict):
            text_parts.extend(str(item.get(key) or "") for key in ("reason", "detail"))
    text = " ".join(text_parts).lower()
    if "asr_env" in text or "asr-env" in text or "model_not_ready" in text or "model_ready" in text:
        return "asr_env"
    if "transcript_source_arbitration" in text or "source_arbitration" in text:
        return "source_arbitration"
    if "term_correction_impact" in text or "term-correction-impact" in text or "final_export_alias" in text:
        return "term_impact"
    if "term_arbitration_codex" in text or "term-arbitration-codex" in text or "codex terminology" in text:
        return "term_arbitration"
    if "transcript_correction" in text or "correction_pack" in text or "corrected_transcript" in text or "glossary" in text:
        return "transcript_correction"
    if "transcript_editor" in text or "transcript_edit" in text or "apply_transcript" in text:
        return "transcript_editor"
    if "subtitle" in text or "caption" in text:
        return "subtitle_import"
    if "funasr" in text or "sensevoice" in text or "whisper" in text or "local_asr" in text or "asr_runner" in text:
        return "local_asr"
    for key, _label, _desc, tokens in ASR_TRANSCRIPT_SUBQUEUES:
        if key == "other_asr_transcript":
            continue
        if any(token in text for token in tokens):
            return key
    return "other_asr_transcript"


def _document_ocr_subqueues(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {
        key: {"key": key, "label": label, "description": desc, "runs": [], "status_counts": {}, "failed_count": 0, "action_required": 0}
        for key, label, desc, _tokens in DOCUMENT_OCR_SUBQUEUES
    }
    for run in runs:
        key = _document_ocr_subqueue_key(run)
        group = groups.setdefault(key, {"key": key, "label": key, "description": "", "runs": [], "status_counts": {}, "failed_count": 0, "action_required": 0})
        status = str(run.get("status") or "unknown")
        group["status_counts"][status] = int(group["status_counts"].get(status, 0)) + 1
        group["failed_count"] = int(group.get("failed_count") or 0) + int(run.get("failed_count") or 0)
        if _run_needs_action(run):
            group["action_required"] = int(group.get("action_required") or 0) + 1
        group["runs"].append(run)
    out: list[dict[str, Any]] = []
    for key, label, desc, _tokens in DOCUMENT_OCR_SUBQUEUES:
        group = groups[key]
        group_runs = group.get("runs") if isinstance(group.get("runs"), list) else []
        action_runs = [run for run in group_runs if _run_needs_action(run)] or group_runs
        group["run_count"] = len(group_runs)
        group["status"] = "empty" if not group_runs else ("action_required" if int(group.get("action_required") or 0) else "ready")
        group["retry_commands"] = _group_retry_commands(action_runs, limit=2)
        group["next_actions"] = _group_next_actions(action_runs)[:2]
        group["failed_items_preview"] = _group_failed_items(action_runs)[:4]
        group["run_ids"] = [str(run.get("run_id") or run.get("title") or "") for run in group_runs[:4] if str(run.get("run_id") or run.get("title") or "").strip()]
        out.append(group)
    return out


def _document_ocr_subqueue_key(run: dict[str, Any]) -> str:
    text_parts = [str(run.get(key) or "") for key in ("run_type", "run_id", "title", "summary")]
    for item in run.get("failed_items_preview") or []:
        if isinstance(item, dict):
            text_parts.extend(str(item.get(key) or "") for key in ("reason", "detail"))
    text = " ".join(text_parts).lower()
    for key, _label, _desc, tokens in DOCUMENT_OCR_SUBQUEUES:
        if key == "other_document_ocr":
            continue
        if any(token in text for token in tokens):
            return key
    return "other_document_ocr"


def _vision_subqueues(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {
        key: {"key": key, "label": label, "description": desc, "runs": [], "status_counts": {}, "failed_count": 0, "action_required": 0}
        for key, label, desc, _tokens in VISION_SUBQUEUES
    }
    for run in runs:
        key = _vision_subqueue_key(run)
        group = groups.setdefault(key, {"key": key, "label": key, "description": "", "runs": [], "status_counts": {}, "failed_count": 0, "action_required": 0})
        status = str(run.get("status") or "unknown")
        group["status_counts"][status] = int(group["status_counts"].get(status, 0)) + 1
        group["failed_count"] = int(group.get("failed_count") or 0) + int(run.get("failed_count") or 0)
        if _run_needs_action(run):
            group["action_required"] = int(group.get("action_required") or 0) + 1
        group["runs"].append(run)
    out: list[dict[str, Any]] = []
    for key, label, desc, _tokens in VISION_SUBQUEUES:
        group = groups[key]
        group_runs = group.get("runs") if isinstance(group.get("runs"), list) else []
        action_runs = [run for run in group_runs if _run_needs_action(run)] or group_runs
        group["run_count"] = len(group_runs)
        group["status"] = "empty" if not group_runs else ("action_required" if int(group.get("action_required") or 0) else "ready")
        group["retry_commands"] = _group_retry_commands(action_runs, limit=2)
        group["next_actions"] = _group_next_actions(action_runs)[:2]
        group["failed_items_preview"] = _group_failed_items(action_runs)[:4]
        group["run_ids"] = [str(run.get("run_id") or run.get("title") or "") for run in group_runs[:4] if str(run.get("run_id") or run.get("title") or "").strip()]
        out.append(group)
    return out


def _vision_subqueue_key(run: dict[str, Any]) -> str:
    text_parts = [str(run.get(key) or "") for key in ("run_type", "run_id", "title", "summary")]
    for item in run.get("failed_items_preview") or []:
        if isinstance(item, dict):
            text_parts.extend(str(item.get(key) or "") for key in ("reason", "detail"))
    text = " ".join(text_parts).lower()
    if "local_vlm" in text or "vlm_serving" in text:
        return "local_vlm"
    for key, _label, _desc, tokens in VISION_SUBQUEUES:
        if key == "other_vision":
            continue
        if any(token in text for token in tokens):
            return key
    return "other_vision"


def _timeline_rag_subqueues(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {
        key: {"key": key, "label": label, "description": desc, "runs": [], "status_counts": {}, "failed_count": 0, "action_required": 0}
        for key, label, desc, _tokens in TIMELINE_RAG_SUBQUEUES
    }
    for run in runs:
        key = _timeline_rag_subqueue_key(run)
        group = groups.setdefault(key, {"key": key, "label": key, "description": "", "runs": [], "status_counts": {}, "failed_count": 0, "action_required": 0})
        status = str(run.get("status") or "unknown")
        group["status_counts"][status] = int(group["status_counts"].get(status, 0)) + 1
        group["failed_count"] = int(group.get("failed_count") or 0) + int(run.get("failed_count") or 0)
        if _run_needs_action(run):
            group["action_required"] = int(group.get("action_required") or 0) + 1
        group["runs"].append(run)
    out: list[dict[str, Any]] = []
    for key, label, desc, _tokens in TIMELINE_RAG_SUBQUEUES:
        group = groups[key]
        group_runs = group.get("runs") if isinstance(group.get("runs"), list) else []
        action_runs = [run for run in group_runs if _run_needs_action(run)] or group_runs
        group["run_count"] = len(group_runs)
        group["status"] = "empty" if not group_runs else ("action_required" if int(group.get("action_required") or 0) else "ready")
        group["retry_commands"] = _group_retry_commands(action_runs, limit=2)
        group["next_actions"] = _group_next_actions(action_runs)[:2]
        group["failed_items_preview"] = _group_failed_items(action_runs)[:4]
        group["run_ids"] = [str(run.get("run_id") or run.get("title") or "") for run in group_runs[:4] if str(run.get("run_id") or run.get("title") or "").strip()]
        out.append(group)
    return out


def _timeline_rag_subqueue_key(run: dict[str, Any]) -> str:
    text_parts = [str(run.get(key) or "") for key in ("run_type", "run_id", "title", "summary")]
    for item in run.get("failed_items_preview") or []:
        if isinstance(item, dict):
            text_parts.extend(str(item.get(key) or "") for key in ("reason", "detail"))
    text = " ".join(text_parts).lower()
    if "timeline_alignment" in text or "alignment_audit" in text or "timestamp_alignment" in text:
        return "timeline_alignment"
    if "video_moment" in text or "moment_index" in text or "moment_search" in text:
        return "moment_index"
    if "video_rag" in text or "video-rag" in text or "rag_pack" in text or "rag_search" in text or "rag_service" in text:
        return "video_rag"
    if "long_video_memory" in text or "long-video-memory" in text or "memory_pack" in text or "long_context" in text:
        return "long_video_memory"
    if "frame_recapture" in text or "supplemental_frame" in text or "recapture" in text or "resample" in text:
        return "recapture"
    for key, _label, _desc, tokens in TIMELINE_RAG_SUBQUEUES:
        if key == "other_timeline_rag":
            continue
        if any(token in text for token in tokens):
            return key
    return "other_timeline_rag"


def _summary_export_subqueues(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {
        key: {"key": key, "label": label, "description": desc, "runs": [], "status_counts": {}, "failed_count": 0, "action_required": 0}
        for key, label, desc, _tokens in SUMMARY_EXPORT_SUBQUEUES
    }
    for run in runs:
        key = _summary_export_subqueue_key(run)
        group = groups.setdefault(key, {"key": key, "label": key, "description": "", "runs": [], "status_counts": {}, "failed_count": 0, "action_required": 0})
        status = str(run.get("status") or "unknown")
        group["status_counts"][status] = int(group["status_counts"].get(status, 0)) + 1
        group["failed_count"] = int(group.get("failed_count") or 0) + int(run.get("failed_count") or 0)
        if _run_needs_action(run):
            group["action_required"] = int(group.get("action_required") or 0) + 1
        group["runs"].append(run)
    out: list[dict[str, Any]] = []
    for key, label, desc, _tokens in SUMMARY_EXPORT_SUBQUEUES:
        group = groups[key]
        group_runs = group.get("runs") if isinstance(group.get("runs"), list) else []
        action_runs = [run for run in group_runs if _run_needs_action(run)] or group_runs
        group["run_count"] = len(group_runs)
        group["status"] = "empty" if not group_runs else ("action_required" if int(group.get("action_required") or 0) else "ready")
        group["retry_commands"] = _group_retry_commands(action_runs, limit=2)
        group["next_actions"] = _group_next_actions(action_runs)[:2]
        group["failed_items_preview"] = _group_failed_items(action_runs)[:4]
        group["run_ids"] = [str(run.get("run_id") or run.get("title") or "") for run in group_runs[:4] if str(run.get("run_id") or run.get("title") or "").strip()]
        out.append(group)
    return out


def _summary_export_subqueue_key(run: dict[str, Any]) -> str:
    text_parts = [str(run.get(key) or "") for key in ("run_type", "run_id", "title", "summary")]
    for item in run.get("failed_items_preview") or []:
        if isinstance(item, dict):
            text_parts.extend(str(item.get(key) or "") for key in ("reason", "detail"))
    text = " ".join(text_parts).lower()
    if "content" in text or "material_card" in text or "capability_pack" in text:
        return "content_candidate"
    if "section_apply" in text:
        return "section_apply"
    if "section_workflow" in text or "section_editor" in text:
        return "section_workflow"
    if "input_pack" in text or "chapter_pack" in text or "smart_summary_chapter" in text or "long_video_memory" in text:
        return "summary_input"
    for key, _label, _desc, tokens in SUMMARY_EXPORT_SUBQUEUES:
        if key == "other_summary_export":
            continue
        if any(token in text for token in tokens):
            return key
    return "other_summary_export"


def _review_subqueues(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {
        key: {"key": key, "label": label, "description": desc, "runs": [], "status_counts": {}, "failed_count": 0, "action_required": 0}
        for key, label, desc, _tokens in REVIEW_SUBQUEUES
    }
    for run in runs:
        key = _review_subqueue_key(run)
        group = groups.setdefault(key, {"key": key, "label": key, "description": "", "runs": [], "status_counts": {}, "failed_count": 0, "action_required": 0})
        status = str(run.get("status") or "unknown")
        group["status_counts"][status] = int(group["status_counts"].get(status, 0)) + 1
        group["failed_count"] = int(group.get("failed_count") or 0) + int(run.get("failed_count") or 0)
        if _run_needs_action(run):
            group["action_required"] = int(group.get("action_required") or 0) + 1
        group["runs"].append(run)
    out: list[dict[str, Any]] = []
    for key, label, desc, _tokens in REVIEW_SUBQUEUES:
        group = groups[key]
        group_runs = group.get("runs") if isinstance(group.get("runs"), list) else []
        action_runs = [run for run in group_runs if _run_needs_action(run)] or group_runs
        group["run_count"] = len(group_runs)
        group["status"] = "empty" if not group_runs else ("action_required" if int(group.get("action_required") or 0) else "ready")
        group["retry_commands"] = _group_retry_commands(action_runs, limit=2)
        group["next_actions"] = _group_next_actions(action_runs)[:2]
        group["failed_items_preview"] = _group_failed_items(action_runs)[:4]
        group["run_ids"] = [str(run.get("run_id") or run.get("title") or "") for run in group_runs[:4] if str(run.get("run_id") or run.get("title") or "").strip()]
        out.append(group)
    return out


def _review_subqueue_key(run: dict[str, Any]) -> str:
    text_parts = [str(run.get(key) or "") for key in ("run_type", "run_id", "title", "summary")]
    for item in run.get("failed_items_preview") or []:
        if isinstance(item, dict):
            text_parts.extend(str(item.get(key) or "") for key in ("reason", "detail"))
    text = " ".join(text_parts).lower()
    if "closure" in text:
        return "closure_status"
    if "sample_review" in text or "human_sample_eval" in text or "impact_report" in text or "sample_eval" in text:
        return "sample_eval"
    if "transcript_arbitration" in text or "arbitration_review" in text or "transcript_review" in text or "low_confidence_conflict" in text:
        return "transcript_arbitration"
    if "human_review_import" in text or "human_import" in text or "safe_import" in text or "apply_review" in text or "apply-review" in text or "apply_notes" in text:
        return "manual_import"
    if "review_pack" in text or "review_session" in text or "prepare_review" in text or "review_notes" in text or "validate_review" in text:
        return "review_pack"
    for key, _label, _desc, tokens in REVIEW_SUBQUEUES:
        if key == "other_review":
            continue
        if any(token in text for token in tokens):
            return key
    return "other_review"

def _hydrate_run_row(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    run = dict(row)
    run_json = str(row.get("run_json") or "")
    path = Path(run_json)
    if run_json and not path.is_absolute():
        path = root / path
    detail = _read_object(path) if run_json else {}
    if detail:
        for key in ("failed_items", "next_actions", "parameters", "operator_boundary", "artifacts", "inputs"):
            if key in detail:
                run[key] = detail.get(key)
        run["summary"] = run.get("summary") or detail.get("summary", "")
        run["retry_command"] = run.get("retry_command") or detail.get("retry_command", "")
        run["failed_count"] = len(detail.get("failed_items") or [])
        run["artifact_count"] = len(detail.get("artifacts") or [])
    run["failed_items_preview"] = _compact_failed_items(run.get("failed_items") if isinstance(run.get("failed_items"), list) else [])
    return run


def _run_needs_action(run: dict[str, Any]) -> bool:
    status = str(run.get("status") or "").strip().lower()
    if status in ACTION_STATUSES:
        return True
    return int(run.get("failed_count") or 0) > 0


def _run_queue_group(run: dict[str, Any]) -> str:
    text = " ".join(str(run.get(key) or "") for key in ("run_type", "run_id", "title")).lower()
    if "term_correction_impact" in text or "term-correction-impact" in text:
        return "asr_transcript"
    if any(token in text for token in ("sample_review", "impact_report", "human_review", "arbitration_review")):
        return "review"
    for group_key, tokens in QUEUE_GROUP_TOKENS.items():
        if any(token in text for token in tokens):
            return group_key
    return "other"


def _group_retry_commands(runs: list[dict[str, Any]], *, limit: int) -> list[str]:
    values: list[str] = []
    for run in runs:
        for item in run.get("failed_items_preview") or []:
            if not isinstance(item, dict):
                continue
            for key in (
                "suggested_retry_command",
                "tile_recovery_command",
                "ebook_retry_command",
                "suggested_apply_command",
                "review_command",
                "tile_result_import_command",
                "tile_result_merge_command",
                "multimodal_triage_command",
            ):
                command = str(item.get(key) or "").strip()
                if command and command not in values:
                    values.append(command)
                    if len(values) >= limit:
                        return values
    for run in runs:
        retry = str(run.get("retry_command") or "").strip()
        if retry and retry not in values:
            values.append(retry)
            if len(values) >= limit:
                return values
    return values[:limit]

def _group_next_actions(runs: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for run in runs:
        for action in run.get("next_actions") or []:
            text = str(action or "").strip()
            if text and text not in values:
                values.append(text)
            if len(values) >= 4:
                return values
    for run in runs:
        retry = str(run.get("retry_command") or "").strip()
        if retry:
            values.append("Copy and run the retry command for " + str(run.get("title") or run.get("run_id") or "this task") + ".")
            break
    return values[:4]


def _group_failed_items(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for run in runs:
        run_id = str(run.get("run_id") or "")
        for item in run.get("failed_items_preview") or []:
            if isinstance(item, dict):
                row = dict(item)
                row.setdefault("run_id", run_id)
                items.append(row)
            if len(items) >= 8:
                return items
    return items


def _compact_failed_items(items: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items[:8]:
        if not isinstance(item, dict):
            continue
        row = {
            "index": item.get("index") or item.get("id") or item.get("item") or "",
            "reason": str(item.get("reason") or item.get("status") or ""),
            "detail": _clip_for_console(str(item.get("detail") or item.get("message") or item.get("tile_id") or ""), 140),
        }
        for key in (
            "suggested_next_tool",
            "suggested_next_reason",
            "suggested_retry_command",
            "ebook_retry_command",
            "tile_recovery_command",
            "multimodal_triage_command",
            "review_command",
            "tile_result_import_command",
            "tile_result_merge_command",
            "suggested_apply_command",
            "tile_id",
            "time_range",
            "title",
        ):
            value = item.get(key)
            if value:
                row[key] = _clip_for_console(str(value), 220) if key.endswith("reason") else str(value)
        evidence_paths = item.get("evidence_paths") if isinstance(item.get("evidence_paths"), list) else []
        if evidence_paths:
            row["evidence_path_count"] = len(evidence_paths)
            row["first_evidence_path"] = str(evidence_paths[0])
        rows.append(row)
    return rows

def _subqueue_effective_status(status_counts: dict[str, Any], fallback: str) -> str:
    for status in ("blocked", "failed", "error", "needs_input", "needs_review", "needs_retry", "needs_execution"):
        if int(status_counts.get(status) or 0) > 0:
            return status
    return fallback

def _subqueue_action_kind(status: str, failed_count: int, retry_commands: list[str], next_actions: list[Any]) -> str:
    normalized = status.strip().lower()
    if normalized in {"blocked", "failed", "error"}:
        return "blocked_or_failed"
    if normalized == "needs_input":
        return "operator_input_required"
    if normalized == "needs_review":
        return "human_review_required"
    if normalized == "needs_execution":
        return "explicit_execution_required" if retry_commands else "execution_plan_missing"
    if normalized == "needs_retry" or failed_count > 0:
        return "retry_available" if retry_commands else "retry_command_missing"
    if retry_commands and next_actions:
        return "optional_followup"
    return "ready_or_empty"


def _subqueue_priority(action_kind: str, action_required: int, failed_count: int) -> int:
    if action_kind == "blocked_or_failed":
        return 10
    if action_kind in {"operator_input_required", "human_review_required"}:
        return 20
    if action_kind in {"retry_available", "retry_command_missing"}:
        return 30
    if action_kind in {"explicit_execution_required", "execution_plan_missing"}:
        return 40
    if action_required or failed_count:
        return 50
    return 90


def _subqueue_blocked_reason(status: str, failed_items: list[Any], next_actions: list[Any]) -> str:
    normalized = status.strip().lower()
    if failed_items:
        item = failed_items[0] if isinstance(failed_items[0], dict) else {}
        reason = str(item.get("reason") or item.get("status") or "").strip()
        detail = str(item.get("detail") or item.get("message") or item.get("tile_id") or "").strip()
        if reason and detail:
            return reason + ": " + detail
        return reason or detail
    if normalized == "needs_input":
        return "missing required input"
    if normalized == "needs_review":
        return "human review required"
    if normalized in {"blocked", "failed", "error"}:
        return normalized
    if next_actions:
        return str(next_actions[0])
    return ""


def _subqueue_safe_execution_hint(action_kind: str) -> str:
    if action_kind in {"operator_input_required", "human_review_required"}:
        return "需要人工输入或审核；不要自动执行。"
    if action_kind in {"blocked_or_failed", "retry_command_missing", "execution_plan_missing"}:
        return "先查看 run.md / failed_items；不要盲目重跑。"
    if action_kind in {"retry_available", "explicit_execution_required"}:
        return "可复制命令执行；仍需遵守该命令自己的 execute/preflight/确认边界。"
    return "只读状态；无需动作。"

def _build_subqueue_action_plan(queue: dict[str, Any]) -> dict[str, Any]:
    groups = queue.get("groups") if isinstance(queue.get("groups"), list) else []
    rows: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        group_key = str(group.get("key") or "")
        group_label = str(group.get("label") or group_key)
        subqueues = group.get("subqueues") if isinstance(group.get("subqueues"), list) else []
        for subqueue in subqueues:
            if not isinstance(subqueue, dict):
                continue
            subqueue_key = str(subqueue.get("key") or "subqueue")
            status = str(subqueue.get("status") or "empty")
            action_required = int(subqueue.get("action_required") or 0)
            failed_count = int(subqueue.get("failed_count") or 0)
            retry_commands = [str(cmd) for cmd in subqueue.get("retry_commands") or [] if str(cmd).strip()]
            next_actions = subqueue.get("next_actions") if isinstance(subqueue.get("next_actions"), list) else []
            failed_items = subqueue.get("failed_items_preview") if isinstance(subqueue.get("failed_items_preview"), list) else []
            if status == "empty" and not action_required and not failed_count and not retry_commands:
                continue
            status_counts = subqueue.get("status_counts") if isinstance(subqueue.get("status_counts"), dict) else {}
            action_status = _subqueue_effective_status(status_counts, status)
            action_kind = _subqueue_action_kind(action_status, failed_count, retry_commands, next_actions)
            priority = _subqueue_priority(action_kind, action_required, failed_count)
            primary_command = retry_commands[0] if retry_commands else ""
            rows.append(
                {
                    "key": group_key + ":" + subqueue_key,
                    "group_key": group_key,
                    "group_label": group_label,
                    "subqueue_key": subqueue_key,
                    "label": str(subqueue.get("label") or subqueue_key),
                    "status": status,
                    "action_status": action_status,
                    "action_kind": action_kind,
                    "priority": priority,
                    "primary_command": primary_command,
                    "blocked_reason": _subqueue_blocked_reason(action_status, failed_items, next_actions),
                    "safe_execution_hint": _subqueue_safe_execution_hint(action_kind),
                    "machine_action_available": bool(primary_command and action_kind in {"retry_available", "explicit_execution_required"}),
                    "operator_review_required": action_kind in {"operator_input_required", "human_review_required", "blocked_or_failed", "retry_command_missing", "execution_plan_missing"},
                    "run_count": int(subqueue.get("run_count") or 0),
                    "action_required": action_required,
                    "failed_count": failed_count,
                    "retry_commands": retry_commands,
                    "command_bundle": "\n".join(retry_commands),
                    "failed_items_preview": failed_items,
                    "next_actions": next_actions,
                }
            )
    rows.sort(key=lambda row: (int(row.get("priority") or 90), str(row.get("group_key") or ""), str(row.get("subqueue_key") or "")))
    return {
        "schema": "video_knowledge_pipeline.subqueue_action_plan.v1",
        "generated_from": "processing_queue.subqueues",
        "row_count": len(rows),
        "action_required_count": sum(1 for row in rows if int(row.get("action_required") or 0) > 0),
        "rows": rows,
        "operator_boundary": {
            "no_process_started": True,
            "no_cloud_call": True,
            "purpose": "Make existing subqueue retry/review commands easier to filter and copy.",
        },
    }

def _subqueue_action_plan_html(plan: dict[str, Any]) -> str:
    rows = plan.get("rows") if isinstance(plan.get("rows"), list) else []
    if not rows:
        return '<div class="panel"><strong>子队列行动面板</strong><div class="muted">当前没有需要展示的子队列动作。</div></div>'
    cards = []
    for idx, row in enumerate(rows[:36], start=1):
        if not isinstance(row, dict):
            continue
        key = str(row.get("key") or "")
        safe_key = html.escape(key, quote=True)
        label = html.escape(str(row.get("label") or row.get("subqueue_key") or "subqueue"))
        group_label = html.escape(str(row.get("group_label") or row.get("group_key") or "group"))
        status = html.escape(str(row.get("status") or "empty"))
        action_kind = html.escape(str(row.get("action_kind") or "ready_or_empty"))
        priority = html.escape(str(row.get("priority") or 90))
        blocked_reason = html.escape(str(row.get("blocked_reason") or ""))
        safe_hint = html.escape(str(row.get("safe_execution_hint") or ""))
        run_count = html.escape(str(row.get("run_count") or 0))
        action_required = html.escape(str(row.get("action_required") or 0))
        failed_count = html.escape(str(row.get("failed_count") or 0))
        commands = str(row.get("command_bundle") or "")
        cmd_id = f"subqueue-action-bundle-{idx}"
        copy_html = ""
        if commands.strip():
            copy_html = '<code id="' + cmd_id + '">' + html.escape(commands) + '</code><button type="button" onclick="copyCommand(\'' + cmd_id + '\')">复制命令包</button>'
        reason_html = '<div class="muted">原因：' + blocked_reason + '</div>' if blocked_reason else ""
        cards.append(
            '<div class="subqueue-action ' + status + ' ' + action_kind + '">'
            + '<div><strong>' + group_label + ' / ' + label + '</strong> <span class="badge">' + status + '</span> <span class="badge">' + action_kind + '</span></div>'
            + '<div class="muted">priority: ' + priority + ' | runs: ' + run_count + ' | action: ' + action_required + ' | failed: ' + failed_count + '</div>'
            + reason_html
            + '<div class="muted">' + safe_hint + '</div>'
            + '<button type="button" onclick="filterSubqueue(\'' + safe_key + '\')">只看这个子队列</button>'
            + copy_html
            + '</div>'
        )
    summary = f"{int(plan.get('row_count') or len(rows))} 个子队列，{int(plan.get('action_required_count') or 0)} 个需要动作。"
    return '<div class="panel"><div class="action-head"><strong>子队列行动面板</strong><button type="button" onclick="filterSubqueue(\'all\')">显示全部</button></div><div class="muted">' + html.escape(summary) + '</div><div class="subqueue-action-grid">' + "".join(cards) + '</div></div>'

def _processing_queue_html(queue: dict[str, Any], root: Path) -> str:
    groups = queue.get("groups") if isinstance(queue.get("groups"), list) else []
    if not groups:
        return '<div class="panel"><strong>处理队列未生成</strong><div class="muted">刷新 task console 后会从 run-artifact-registry.json 汇总任务队列。</div></div>'
    summary = f"{int(queue.get('run_count') or 0)} 个 run，{int(queue.get('action_required_count') or 0)} 个队列需要执行、重试或复核。"
    cards = []
    for idx, group in enumerate(groups, start=1):
        cards.append(_processing_queue_card(group, idx, root))
    return '<div class="panel"><div class="muted">' + html.escape(summary) + '</div><div class="queue-grid">' + ''.join(cards) + '</div></div>'


def _processing_queue_card(group: dict[str, Any], idx: int, root: Path) -> str:
    status = html.escape(str(group.get("status") or "empty"))
    label = html.escape(str(group.get("label") or group.get("key") or "Queue"))
    desc = html.escape(str(group.get("description") or ""))
    run_count = html.escape(str(group.get("run_count") or 0))
    failed_count = html.escape(str(group.get("failed_count") or 0))
    action_required = html.escape(str(group.get("action_required") or 0))
    counts = group.get("status_counts") if isinstance(group.get("status_counts"), dict) else {}
    count_text = html.escape(" / ".join(f"{key}:{value}" for key, value in sorted(counts.items())) or "no runs")
    runs = group.get("runs") if isinstance(group.get("runs"), list) else []
    run_lines = []
    for run in runs[:4]:
        run_md = html.escape(_relative_href(Path(str(run.get("run_markdown") or "")), root)) if run.get("run_markdown") else ""
        link = f' <a href="{run_md}">run.md</a>' if run_md else ""
        run_lines.append('<li><span class="badge">' + html.escape(str(run.get("status") or "")) + '</span> ' + html.escape(str(run.get("title") or run.get("run_id") or "run")) + link + '</li>')
    next_actions = group.get("next_actions") if isinstance(group.get("next_actions"), list) else []
    action_lines = ''.join('<li>' + html.escape(str(action)) + '</li>' for action in next_actions[:3])
    failed_items = group.get("failed_items_preview") if isinstance(group.get("failed_items_preview"), list) else []
    failed_text = '; '.join(_failed_item_label(item) for item in failed_items[:4] if isinstance(item, dict))
    subqueue_title = _subqueue_title(str(group.get("key") or ""))
    subqueue_html = _queue_subqueues_html(group.get("subqueues"), root, idx, subqueue_title, str(group.get("key") or ""))
    retry_commands = [str(cmd) for cmd in group.get("retry_commands") or [] if str(cmd).strip()]
    retry_html = ""
    if retry_commands:
        cmd_id = f"queue-retry-{idx}"
        retry_html = '<code id="' + cmd_id + '">' + html.escape(retry_commands[0]) + '</code><button type="button" onclick="copyCommand(\'' + cmd_id + '\')">复制重试命令</button>'
    return (
        '<div class="queue-card ' + status + '">'
        '<div><strong>' + label + '</strong> <span class="badge">' + status + '</span></div>'
        '<div class="muted">' + desc + '</div>'
        '<div class="muted">runs: ' + run_count + ' | action: ' + action_required + ' | failed: ' + failed_count + '</div>'
        '<div class="muted">' + count_text + '</div>'
        '<ul>' + ''.join(run_lines) + '</ul>'
        + ('<div class="muted">失败项：' + html.escape(failed_text) + '</div>' if failed_text else '')
        + ('<ul>' + action_lines + '</ul>' if action_lines else '')
        + subqueue_html
        + retry_html
        + '</div>'
    )


def _subqueue_title(group_key: str) -> str:
    if group_key == "asr_transcript":
        return "ASR 子队列"
    if group_key == "vision":
        return "视觉子队列"
    if group_key == "timeline_rag":
        return "时间轴/RAG 子队列"
    if group_key == "summary_export":
        return "总结/导出子队列"
    if group_key == "review":
        return "审核子队列"
    return "OCR 子队列"


def _queue_subqueues_html(value: Any, root: Path, card_idx: int, title: str, group_key: str) -> str:
    if not isinstance(value, list) or not value:
        return ""
    rows = []
    for idx, subqueue in enumerate(value, start=1):
        if not isinstance(subqueue, dict):
            continue
        status = html.escape(str(subqueue.get("status") or "empty"))
        label = html.escape(str(subqueue.get("label") or subqueue.get("key") or "subqueue"))
        run_count = html.escape(str(subqueue.get("run_count") or 0))
        failed_count = html.escape(str(subqueue.get("failed_count") or 0))
        action_required = html.escape(str(subqueue.get("action_required") or 0))
        failed_items = subqueue.get("failed_items_preview") if isinstance(subqueue.get("failed_items_preview"), list) else []
        failed_text = "; ".join(_failed_item_label(item) for item in failed_items[:2] if isinstance(item, dict))
        retry_commands = [str(cmd) for cmd in subqueue.get("retry_commands") or [] if str(cmd).strip()]
        retry_html = ""
        if retry_commands:
            cmd_id = f"queue-subretry-{card_idx}-{idx}"
            retry_html = '<code id="' + cmd_id + '">' + html.escape(retry_commands[0]) + '</code><button type="button" onclick="copyCommand(\'' + cmd_id + '\')">复制子队列命令</button>'
        rows.append(
            '<div class="subqueue ' + status + '" data-subqueue-full-key="' + html.escape(group_key + ":" + str(subqueue.get("key") or ""), quote=True) + '">'
            + '<div><strong>' + label + '</strong> <span class="badge">' + status + '</span></div>'
            + '<div class="muted">runs: ' + run_count + ' | action: ' + action_required + ' | failed: ' + failed_count + '</div>'
            + ('<div class="muted">失败项：' + html.escape(failed_text) + '</div>' if failed_text else '')
            + retry_html
            + '</div>'
        )
    return '<div class="subqueues"><div class="muted">' + html.escape(title) + '</div>' + "".join(rows) + '</div>'


def _failed_item_label(item: dict[str, Any]) -> str:
    index = str(item.get("index") or "")
    reason = str(item.get("reason") or "")
    detail = str(item.get("detail") or "")
    next_tool = str(item.get("suggested_next_tool") or "")
    evidence_count = str(item.get("evidence_path_count") or "")
    parts = [value for value in (index, reason, detail) if value]
    if next_tool:
        parts.append("next:" + next_tool)
    if evidence_count:
        parts.append("evidence:" + evidence_count)
    return " / ".join(parts)
def _run_history_html(registry: dict[str, Any], root: Path) -> str:
    runs = registry.get("runs") if isinstance(registry.get("runs"), list) else []
    count = int(registry.get("run_count") or len(runs))
    if not runs:
        command = f".\\scripts\\video-knowledge.ps1 run-artifact-registry {_ps_quote(str(root))}"
        return "".join([
            '<div class="panel">',
            '<strong>还没有任务历史</strong>',
            '<div class="muted">生成疑难点多模态队列、ebook 批次、summary 批次后，这里会显示 run 状态、失败项和重试命令。</div>',
            f'<code id="cmd-run-registry-refresh">{html.escape(command)}</code>',
            '<button type="button" onclick="copyCommand(\'cmd-run-registry-refresh\')">复制刷新命令</button>',
            '</div>',
        ])
    counts = registry.get("status_counts") if isinstance(registry.get("status_counts"), dict) else {}
    count_text = "、".join(f"{key}: {value}" for key, value in sorted(counts.items())) or f"runs: {count}"
    cards = []
    for idx, run in enumerate(runs[:8], start=1):
        status = html.escape(str(run.get("status") or "unknown"))
        run_id = html.escape(str(run.get("run_id") or "run"))
        run_type = html.escape(str(run.get("run_type") or ""))
        title = html.escape(str(run.get("title") or run.get("run_id") or "Run"))
        summary = html.escape(str(run.get("summary") or ""))
        freshness = run.get("freshness") if isinstance(run.get("freshness"), dict) else {}
        freshness_status = html.escape(str(freshness.get("status") or "not_recorded"))
        evidence_badge = f'<span class="badge">evidence:{freshness_status}</span>'
        artifact_count = html.escape(str(run.get("artifact_count") or 0))
        failed_count = html.escape(str(run.get("failed_count") or 0))
        run_md = html.escape(_relative_href(Path(str(run.get("run_markdown") or "")), root)) if run.get("run_markdown") else ""
        retry = str(run.get("retry_command") or "")
        retry_html = ""
        if retry:
            cmd_id = f"run-retry-{idx}"
            retry_html = f'<code id="{cmd_id}">{html.escape(retry)}</code><button type="button" onclick="copyCommand(\'{cmd_id}\')">复制重试命令</button>'
        link_html = f'<a href="{run_md}">run.md</a>' if run_md else ""
        cards.append(
            f'<div class="run-card {status}"><div><strong>{title}</strong> <span class="badge">{status}</span> <span class="badge">{run_type}</span> {evidence_badge}</div>'
            f'<div class="muted">run: <code style="display:inline;padding:2px 5px">{run_id}</code> | artifacts: {artifact_count} | failed: {failed_count} | {link_html}</div>'
            f'<div class="snippet">{summary}</div>{retry_html}</div>'
        )
    return "".join([
        '<div class="panel">',
        f'<div class="muted">{html.escape(count_text)}。最近 {len(cards)} 条任务；用于查看失败项、产物和重试入口。</div>',
        '<div class="run-list">',
        "".join(cards),
        '</div></div>',
    ])
def _term_validation_guidance_html(status: dict[str, Any]) -> str:
    reasons = status.get("validation_rejection_reasons") if isinstance(status.get("validation_rejection_reasons"), list) else []
    rejected = status.get("validation_rejected_decisions") if isinstance(status.get("validation_rejected_decisions"), list) else []
    if not reasons and not rejected:
        return ""
    reason_items = "".join(
        f"<li><code>{html.escape(str(row.get('reason') or ''))}</code> x {html.escape(str(row.get('count') or 0))}</li>"
        for row in reasons[:8]
        if isinstance(row, dict)
    )
    rejected_items = "".join(
        "<li>"
        + f"<strong>{html.escape(str(row.get('canonical') or '-'))}</strong> "
        + f"<span class=\"muted\">{html.escape(', '.join(str(value) for value in row.get('rejection_reasons') or []))}</span>"
        + "</li>"
        for row in rejected[:6]
        if isinstance(row, dict)
    )
    return (
        '<section class="callout warn">'
        '<strong>Codex 术语预检需要补证据</strong>'
        '<p class="muted">高置信替换必须带语义理由、候选 ID 和证据时间线索引；否则不会进入术语闭环。</p>'
        + (f'<ul>{reason_items}</ul>' if reason_items else '')
        + (f'<div class="muted">被拒绝样例</div><ul>{rejected_items}</ul>' if rejected_items else '')
        + '</section>'
    )


def _semantic_correction_ui_summary_html(status: dict[str, Any]) -> str:
    summary = status.get("ui_summary") if isinstance(status.get("ui_summary"), dict) else {}
    if not summary:
        return ""
    export_chain = summary.get("export_chain") if isinstance(summary.get("export_chain"), dict) else {}
    cards = [
        ("闭环状态", summary.get("ui_state") or "unknown"),
        ("下一步", summary.get("next_action_key") or status.get("next_action_key") or ""),
        ("自动候选", summary.get("auto_candidate_count") or 0),
        ("需人工候选", summary.get("human_review_candidate_count") or 0),
        ("接受/拒绝", f"{int(summary.get('accepted_decision_count') or 0)} / {int(summary.get('rejected_decision_count') or 0)}"),
        ("已应用", summary.get("applied_correction_count") or 0),
        ("导入/关闭/未关", f"{int(summary.get('review_imported_count') or 0)} / {int(summary.get('review_closed_count') or 0)} / {int(summary.get('review_required_count') or 0)}"),
        ("可读/总结影响", f"{export_chain.get('readable_impact_status') or 'missing'} / {export_chain.get('summary_impact_status') or 'missing'}"),
    ]
    card_html = "".join('<div><span class="muted">' + html.escape(str(label)) + '</span><strong>' + html.escape(str(value)) + '</strong></div>' for label, value in cards)
    groups = [
        ("已接受类型", summary.get("accepted_decision_type_counts") if isinstance(summary.get("accepted_decision_type_counts"), dict) else {}),
        ("已应用类型", summary.get("applied_correction_type_counts") if isinstance(summary.get("applied_correction_type_counts"), dict) else {}),
        ("拒绝原因", summary.get("rejected_decision_reason_counts") if isinstance(summary.get("rejected_decision_reason_counts"), dict) else {}),
    ]
    group_cards = []
    for title, values in groups:
        if not values:
            continue
        items = "".join('<li><code>' + html.escape(str(key)) + '</code> x ' + html.escape(str(count)) + '</li>' for key, count in sorted(values.items()))
        group_cards.append('<div class="subpanel"><strong>' + html.escape(title) + '</strong><ul>' + items + '</ul></div>')
    preview = summary.get("applied_correction_preview") if isinstance(summary.get("applied_correction_preview"), list) else []
    preview_rows = []
    for row in preview[:8]:
        preview_rows.append(
            '<tr><td><code>' + html.escape(str(row.get("candidate_id") or "")) + '</code></td>'
            + '<td><code>' + html.escape(str(row.get("correction_type") or "")) + '</code></td>'
            + '<td>' + html.escape(str(row.get("original_text") or "")) + '</td>'
            + '<td>' + html.escape(str(row.get("corrected_text") or "")) + '</td></tr>'
        )
    preview_html = ''
    if preview_rows:
        preview_html = '<div class="subpanel"><strong>已应用预览</strong><table><thead><tr><th>ID</th><th>类型</th><th>原文</th><th>纠正</th></tr></thead><tbody>' + ''.join(preview_rows) + '</tbody></table></div>'
    return '<div class="subpanel semantic-ui-summary"><strong>通用语义纠错闭环进度摘要</strong><p class="muted">给 UI、批量队列和 OpenClaw 使用的稳定摘要：显示哪些错词候选可自动闭环，哪些仍需人工或更多证据。</p><div class="grid">' + card_html + '</div><div class="grid">' + ''.join(group_cards) + '</div>' + preview_html + '</div>'
def _semantic_correction_detail_html(status: dict[str, Any]) -> str:
    groups = [
        ("候选类型", status.get("candidate_type_counts") if isinstance(status.get("candidate_type_counts"), dict) else {}),
        ("风险等级", status.get("risk_level_counts") if isinstance(status.get("risk_level_counts"), dict) else {}),
        ("证据来源", status.get("evidence_source_counts") if isinstance(status.get("evidence_source_counts"), dict) else {}),
        ("预检拒绝原因", status.get("validation_rejection_reason_counts") if isinstance(status.get("validation_rejection_reason_counts"), dict) else {}),
    ]
    cards = []
    for title, values in groups:
        if not values:
            continue
        items = "".join(f"<li><code>{html.escape(str(key))}</code> x {html.escape(str(count))}</li>" for key, count in sorted(values.items()))
        cards.append(f"<div class=\"subpanel\"><strong>{html.escape(title)}</strong><ul>{items}</ul></div>")
    if not cards:
        return '<p class="muted">尚无候选分类、证据源或预检失败原因汇总。</p>'
    return '<div class="grid semantic-correction-detail">' + "".join(cards) + '</div>'




def _semantic_correction_source_vote_html(status: dict[str, Any]) -> str:
    summary = status.get("source_vote_summary") if isinstance(status.get("source_vote_summary"), dict) else {}
    if not summary:
        return ""
    cards = [
        ("有投票候选", summary.get("candidate_count_with_votes") or 0),
        ("来源冲突", summary.get("source_conflict_count") or 0),
        ("投票需复核", summary.get("needs_review_by_source_vote_count") or 0),
        ("候选/原文权重", f"{int(summary.get('candidate_weight_total') or 0)} / {int(summary.get('original_weight_total') or 0)}"),
    ]
    card_html = "".join('<div><span class="muted">' + html.escape(str(label)) + '</span><strong>' + html.escape(str(value)) + '</strong></div>' for label, value in cards)
    groups = []
    for title, key in (("优势方", "by_dominant_side"), ("支持候选来源", "by_candidate_support_source"), ("支持原文来源", "by_original_support_source")):
        values = summary.get(key) if isinstance(summary.get(key), dict) else {}
        if not values:
            continue
        items = "".join('<li><code>' + html.escape(str(name)) + '</code> x ' + html.escape(str(count)) + '</li>' for name, count in sorted(values.items()))
        groups.append('<div class="subpanel"><strong>' + html.escape(title) + '</strong><ul>' + items + '</ul></div>')
    rows = []
    for row in (summary.get("conflict_preview") if isinstance(summary.get("conflict_preview"), list) else [])[:10]:
        rows.append(
            '<tr><td><code>' + html.escape(str(row.get("candidate_id") or "")) + '</code></td>'
            + '<td><code>' + html.escape(str(row.get("dominant_side") or "")) + '</code></td>'
            + '<td>' + html.escape(str(row.get("candidate_weight") or 0)) + '/' + html.escape(str(row.get("original_weight") or 0)) + '</td>'
            + '<td>' + html.escape(str(row.get("original_text") or "")) + '</td>'
            + '<td>' + html.escape(str(row.get("candidate_text") or "")) + '</td>'
            + '<td>' + html.escape(', '.join(str(item) for item in row.get("supports_candidate") or [])) + '</td>'
            + '<td>' + html.escape(', '.join(str(item) for item in row.get("supports_original") or [])) + '</td></tr>'
        )
    table = ''
    if rows:
        table = '<table><thead><tr><th>ID</th><th>优势方</th><th>权重</th><th>原文</th><th>候选</th><th>支持候选</th><th>支持原文</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'
    return '<div class="subpanel semantic-source-vote"><strong>来源投票 / 字幕可靠性摘要</strong><p class="muted">平台/内嵌字幕是证据源，不是默认事实；当 OCR、视觉或人工强证据支持原文时，不会被字幕单独覆盖。</p><div class="grid">' + card_html + '</div><div class="grid">' + ''.join(groups) + '</div>' + table + '</div>'
def _semantic_correction_candidate_groups_html(status: dict[str, Any]) -> str:
    rows = status.get("candidate_group_preview") if isinstance(status.get("candidate_group_preview"), list) else []
    if not rows:
        return ""
    body = []
    for row in rows[:12]:
        types = ", ".join(str(item) for item in (row.get("correction_types") or [row.get("correction_type")]) if str(item))
        variants = ", ".join(str(item) for item in (row.get("variant_texts") or [])[:6])
        sources = ", ".join(str(item) for item in (row.get("evidence_source_types") or [])[:6])
        body.append(
            "<tr>"
            + f"<td><code>{html.escape(str(row.get('candidate_group_id') or ''))}</code></td>"
            + f"<td>{html.escape(str(row.get('canonical_hint') or ''))}</td>"
            + f"<td><code>{html.escape(types)}</code></td>"
            + f"<td><code>{html.escape(str(row.get('risk_level') or ''))}</code></td>"
            + f"<td>{int(row.get('candidate_count') or 0)}</td>"
            + f"<td>{html.escape(variants)}</td>"
            + f"<td>{html.escape(sources)}</td>"
            + "</tr>"
        )
    return '<div class="subpanel semantic-candidate-groups"><strong>候选分组预览</strong><table><thead><tr><th>Group</th><th>Canonical</th><th>类型</th><th>风险</th><th>候选</th><th>变体</th><th>证据源</th></tr></thead><tbody>' + "".join(body) + '</tbody></table></div>'



def _semantic_correction_attention_html(status: dict[str, Any]) -> str:
    rows = status.get("semantic_attention_preview") if isinstance(status.get("semantic_attention_preview"), list) else []
    if not rows:
        return ""
    body = []
    for row in rows[:12]:
        sources = ", ".join(str(item) for item in (row.get("evidence_source_types") or [])[:6])
        body.append(
            "<tr>"
            + f"<td><code>{html.escape(str(row.get('candidate_id') or ''))}</code></td>"
            + f"<td><code>{html.escape(str(row.get('correction_type') or ''))}</code></td>"
            + f"<td>{int(row.get('priority_score') or 0)}</td>"
            + f"<td>{html.escape(str(row.get('time_range') or ''))}</td>"
            + f"<td>{html.escape(str(row.get('original_text') or ''))}</td>"
            + f"<td>{html.escape(str(row.get('suggested_text') or ''))}</td>"
            + f"<td>{html.escape(str(row.get('reason') or ''))}</td>"
            + f"<td>{html.escape(sources)}</td>"
            + "</tr>"
        )
    return '<div class="subpanel semantic-attention-items"><strong>语义重点复核队列</strong><p class="muted">优先展示数字、动作、概念、普通错词和断句候选；分数越高越值得先交给 Codex/LLM/人工判读。</p><table><thead><tr><th>ID</th><th>类型</th><th>分数</th><th>时间</th><th>原文</th><th>建议</th><th>原因</th><th>证据源</th></tr></thead><tbody>' + "".join(body) + '</tbody></table></div>'



def _semantic_correction_chapter_risk_html(status: dict[str, Any]) -> str:
    rows = status.get("chapter_risk_summary") if isinstance(status.get("chapter_risk_summary"), list) else []
    if not rows:
        return ""
    body = []
    for row in rows[:20]:
        risks = row.get("risk_level_counts") if isinstance(row.get("risk_level_counts"), dict) else {}
        risk_text = ", ".join(f"{key}={value}" for key, value in sorted(risks.items())) or "none"
        high_ids = ", ".join(str(item) for item in (row.get("high_risk_candidate_ids") or [])[:8])
        body.append(
            "<tr>"
            + f"<td><code>{html.escape(str(row.get('chapter_index') or ''))}</code> {html.escape(str(row.get('chapter_title') or ''))}</td>"
            + f"<td>{html.escape(str(row.get('chapter_time_range') or ''))}</td>"
            + f"<td>{int(row.get('candidate_count') or 0)}</td>"
            + f"<td>{int(row.get('review_required_count') or 0)}</td>"
            + f"<td>{html.escape(risk_text)}</td>"
            + f"<td>{html.escape(high_ids)}</td>"
            + "</tr>"
        )
    return '<div class="subpanel semantic-chapter-risk"><strong>按章节/风险分组</strong><table><thead><tr><th>章节</th><th>时间</th><th>候选</th><th>待复核</th><th>风险</th><th>高风险候选</th></tr></thead><tbody>' + "".join(body) + '</tbody></table></div>'
def _semantic_correction_review_preview_html(status: dict[str, Any]) -> str:
    rows = status.get("review_required_preview") if isinstance(status.get("review_required_preview"), list) else []
    if not rows:
        return ""
    body = []
    for row in rows[:8]:
        reasons = ", ".join(str(item) for item in row.get("reject_reasons") or [])
        evidence = ", ".join(str(item) for item in row.get("evidence_source_types") or [])
        body.append(
            "<tr>"
            + f"<td><code>{html.escape(str(row.get('candidate_id') or ''))}</code></td>"
            + f"<td><code>{html.escape(str(row.get('correction_type') or ''))}</code></td>"
            + f"<td>{html.escape(str(row.get('time_range') or ''))}</td>"
            + f"<td>{html.escape(str(row.get('original_text') or ''))}</td>"
            + f"<td>{html.escape(str(row.get('suggested_text') or reasons or ''))}</td>"
            + f"<td>{html.escape(evidence)}</td>"
            + "</tr>"
        )
    return '<div class="subpanel"><strong>待人工复核样例</strong><table><thead><tr><th>ID</th><th>类型</th><th>时间</th><th>原文</th><th>建议/原因</th><th>证据源</th></tr></thead><tbody>' + "".join(body) + '</tbody></table></div>'
def _semantic_correction_review_closure_html(status: dict[str, Any]) -> str:
    summary = status.get("review_closure_summary") if isinstance(status.get("review_closure_summary"), dict) else {}
    if not summary or not summary.get("review_result_imported"):
        return ""
    actions = summary.get("actions") if isinstance(summary.get("actions"), dict) else {}
    action_items = "".join(f"<li><code>{html.escape(str(key))}</code> x {html.escape(str(value))}</li>" for key, value in sorted(actions.items()))
    skipped = summary.get("skipped") if isinstance(summary.get("skipped"), list) else []
    skipped_items = "".join(
        "<li>"
        + f"row <code>{html.escape(str(row.get('row_number') or ''))}</code> "
        + f"candidate <code>{html.escape(str(row.get('candidate_id') or ''))}</code>: "
        + html.escape(str(row.get("reason") or ""))
        + "</li>"
        for row in skipped[:8]
        if isinstance(row, dict)
    )
    next_action = html.escape(str(summary.get("next_action_key") or ""))
    next_command = html.escape(str(summary.get("next_action_command") or ""))
    validation_status = html.escape(str(summary.get("validation_status") or "missing"))
    return (
        '<div class="subpanel semantic-review-closure"><strong>语义纠错复核导入结果</strong>'
        + '<div class="grid">'
        + f'<div><span class="muted">已导入</span><strong>{int(summary.get("imported_review_decision_count") or 0)}</strong></div>'
        + f'<div><span class="muted">预检接受</span><strong>{int(summary.get("accepted_imported_review_decision_count") or 0)}</strong></div>'
        + f'<div><span class="muted">预检拒绝</span><strong>{int(summary.get("rejected_imported_review_decision_count") or 0)}</strong></div>'
        + f'<div><span class="muted">导入跳过</span><strong>{int(summary.get("skipped_review_note_count") or 0)}</strong></div>'
        + f'<div><span class="muted">已关闭</span><strong>{int(summary.get("closed_review_decision_count") or 0)}</strong></div>'
        + f'<div><span class="muted">仍待处理</span><strong>{int(summary.get("open_review_required_count") or 0)}</strong></div>'
        + '</div>'
        + f'<p class="muted">Validation: <code>{validation_status}</code>；下一步：<code>{next_action}</code></p>'
        + (f'<pre>{next_command}</pre>' if next_command else '')
        + (f'<ul>{action_items}</ul>' if action_items else '')
        + (f'<details><summary>跳过的 review notes</summary><ul>{skipped_items}</ul></details>' if skipped_items else '')
        + '</div>'
    )


def _semantic_correction_export_chain_html(status: dict[str, Any]) -> str:
    closure_status = str(status.get("closure_status") or "missing")
    corrected_exists = bool(status.get("corrected_transcript_exists"))
    summary_status = str(status.get("summary_impact_status") or "missing")
    if closure_status == "missing" and not corrected_exists and summary_status == "missing":
        return ""
    closure_ok = "yes" if status.get("closure_ok") else "no"
    corrected_label = "yes" if corrected_exists else "no"
    summary_ok = "yes" if status.get("summary_impact_ok") else "no"
    corrected_path = html.escape(str(status.get("corrected_transcript_path") or ""))
    summary_rate = html.escape(str(status.get("summary_absorption_rate") or 0.0))
    residual = int(status.get("summary_residual_original_total") or 0)
    closure_command = html.escape(str((status.get("commands") if isinstance(status.get("commands"), dict) else {}).get("closure") or ""))
    summary_command = html.escape(str((status.get("commands") if isinstance(status.get("commands"), dict) else {}).get("summary_impact") or ""))
    next_command = summary_command if corrected_exists and summary_status == "missing" else closure_command
    return (
        '<div class="subpanel semantic-export-chain"><strong>语义纠错导出闭环</strong>'
        + '<p class="muted">这块检查复核/预检之后有没有真正进入纠正版 transcript，并继续影响 smart-summary。</p>'
        + '<div class="grid">'
        + f'<div><span class="muted">Closure</span><strong>{html.escape(closure_status)}</strong></div>'
        + f'<div><span class="muted">Closure OK</span><strong>{closure_ok}</strong></div>'
        + f'<div><span class="muted">已应用</span><strong>{int(status.get("closure_applied_correction_count") or 0)}</strong></div>'
        + f'<div><span class="muted">改动段落</span><strong>{int(status.get("closure_changed_segment_count") or 0)}</strong></div>'
        + f'<div><span class="muted">纠正版 transcript</span><strong>{corrected_label}</strong></div>'
        + f'<div><span class="muted">Summary impact</span><strong>{html.escape(summary_status)}</strong></div>'
        + f'<div><span class="muted">Summary OK</span><strong>{summary_ok}</strong></div>'
        + f'<div><span class="muted">吸收率</span><strong>{summary_rate}</strong></div>'
        + f'<div><span class="muted">总结残留错词</span><strong>{residual}</strong></div>'
        + '</div>'
        + (f'<p class="muted">Corrected transcript: <code>{corrected_path}</code></p>' if corrected_path else '')
        + (f'<pre>{next_command}</pre>' if next_command else '')
        + '</div>'
    )

def _semantic_candidate_discovery_html(status: dict[str, Any]) -> str:
    artifacts = status.get("candidate_discovery_artifacts") if isinstance(status.get("candidate_discovery_artifacts"), dict) else {}
    if not artifacts and not str(status.get("candidate_discovery_status") or "").strip():
        return ""
    commands = status.get("commands") if isinstance(status.get("commands"), dict) else {}
    sources: list[dict[str, Any]] = []
    for label, key in [
        ("Codex suggestions", "codex_suggestions_markdown"),
        ("LLM suggestions JSON", "llm_suggestions_json"),
        ("LLM suggestions Markdown", "llm_suggestions_markdown"),
    ]:
        path_text = str(artifacts.get(key) or "").strip()
        if not path_text:
            continue
        path = Path(path_text)
        payload, error = _read_json_or_markdown_object(path)
        suggestions = payload.get("suggestions") if isinstance(payload.get("suggestions"), list) else payload.get("candidates")
        suggestions = [row for row in (suggestions or []) if isinstance(row, dict)]
        if path.exists() or suggestions or error:
            sources.append({"label": label, "path": str(path), "payload": payload, "error": error, "suggestions": suggestions})
    import_path = Path(str(artifacts.get("import_json") or "")) if str(artifacts.get("import_json") or "").strip() else None
    imported = _read_object(import_path) if import_path is not None else {}
    imported_ids = imported.get("imported_candidate_ids") if isinstance(imported.get("imported_candidate_ids"), list) else []
    skipped = imported.get("skipped") if isinstance(imported.get("skipped"), list) else []
    cards = [
        ("候选发现", status.get("candidate_discovery_status") or "not_planned"),
        ("下一步", status.get("candidate_discovery_next_action") or "run_candidate_discovery"),
        ("发现片段", status.get("candidate_discovery_segment_count") or 0),
        ("Suggestions", status.get("candidate_discovery_suggestion_count") or 0),
        ("已导入", status.get("candidate_discovery_imported_candidate_count") or 0),
        ("跳过", status.get("candidate_discovery_skipped_count") or 0),
    ]
    card_html = "".join('<div><span class="muted">' + html.escape(str(label)) + '</span><strong>' + html.escape(str(value)) + '</strong></div>' for label, value in cards)
    source_rows = []
    for source in sources:
        error = str(source.get("error") or "")
        count = len(source.get("suggestions") or [])
        source_rows.append(
            '<div><span class="muted">' + html.escape(str(source.get("label") or "")) + '</span><code>'
            + html.escape(str(source.get("path") or "")) + '</code><strong>' + html.escape(str(count)) + '</strong>'
            + (('<p class="muted">解析失败：' + html.escape(error) + '</p>') if error else '')
            + '</div>'
        )
    suggestions_html: list[str] = []
    shown = 0
    for source in sources:
        label = str(source.get("label") or "")
        for row in (source.get("suggestions") or []):
            if shown >= 8:
                break
            shown += 1
            original = html.escape(str(row.get("original_text") or row.get("span") or ""))
            candidate = html.escape(str(row.get("candidate_text") or row.get("suggested_text") or row.get("corrected_text") or ""))
            ctype = html.escape(str(row.get("correction_type") or "ordinary_word"))
            confidence = html.escape(str(row.get("confidence") if row.get("confidence") not in {None, ""} else ""))
            segment = html.escape(str(row.get("source_segment_index") if row.get("source_segment_index") is not None else row.get("segment_index") or ""))
            reason = html.escape(str(row.get("reason") or row.get("evidence_summary") or ""))
            suggestions_html.append(
                '<tr><td>' + html.escape(label) + '</td><td><code>' + segment + '</code></td><td><span class="badge">' + ctype + '</span></td>'
                + '<td>' + original + '</td><td>' + candidate + '</td><td>' + confidence + '</td><td>' + reason + '</td></tr>'
            )
    if not suggestions_html:
        suggestions_html.append('<tr><td colspan="7" class="muted">尚无可预览 suggestions；先运行候选发现 Prompt，再让 Codex/LLM 填写 suggestions。</td></tr>')
    skipped_rows = []
    for row in skipped[:6]:
        suggestion = row.get("suggestion") if isinstance(row.get("suggestion"), dict) else {}
        skipped_rows.append(
            '<tr><td>' + html.escape(str(row.get("row_number") or "")) + '</td><td>' + html.escape(str(row.get("reason") or "")) + '</td><td>'
            + html.escape(str(suggestion.get("original_text") or suggestion.get("span") or "")) + '</td><td>'
            + html.escape(str(suggestion.get("candidate_text") or suggestion.get("suggested_text") or "")) + '</td></tr>'
        )
    skipped_html = "".join(skipped_rows) or '<tr><td colspan="4" class="muted">暂无导入跳过项。</td></tr>'
    imported_html = ", ".join(html.escape(str(item)) for item in imported_ids[:12]) or "暂无"
    editor_html = _semantic_candidate_suggestions_editor_html(sources)
    import_command = str(commands.get("import_candidate_suggestions") or "").strip()
    command_html = '<pre>' + html.escape(import_command) + '</pre>' if import_command else ''
    return (
        '<div class="subpanel semantic-candidate-discovery"><strong>候选发现 suggestions 预览</strong>'
        + '<p class="muted">这里只展示候选召回结果。导入后仍只是标准候选，必须继续走 Codex/LLM/人工仲裁、validate、closure，不能直接改 transcript。</p>'
        + '<div class="grid">' + card_html + '</div>'
        + '<h3>Suggestions 来源</h3><div class="grid">' + ("".join(source_rows) or '<div class="muted">还没有 suggestions 文件。</div>') + '</div>'
        + '<h3>候选建议样例</h3><table><thead><tr><th>来源</th><th>Segment</th><th>类型</th><th>原文</th><th>建议</th><th>置信</th><th>理由</th></tr></thead><tbody>' + "".join(suggestions_html) + '</tbody></table>'
        + editor_html
        + '<h3>导入结果</h3><p class="muted">已导入 candidate ids：' + imported_html + '</p><table><thead><tr><th>行号</th><th>跳过原因</th><th>原文</th><th>建议</th></tr></thead><tbody>' + skipped_html + '</tbody></table>'
        + command_html
        + '</div>'
    )



def _semantic_candidate_suggestions_editor_html(sources: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for source_index, source in enumerate(sources, start=1):
        label = str(source.get("label") or "suggestions")
        path_text = str(source.get("path") or "")
        for suggestion_index, row in enumerate((source.get("suggestions") or []), start=1):
            if not isinstance(row, dict):
                continue
            row_id_raw = str(row.get("suggestion_id") or row.get("candidate_id") or f"suggestion-{source_index}-{suggestion_index}")
            row_id = html.escape(row_id_raw, quote=True)
            source_label = html.escape(label, quote=True)
            source_path = html.escape(path_text, quote=True)
            segment_value = row.get("source_segment_index") if row.get("source_segment_index") is not None else row.get("segment_index")
            segment = html.escape("" if segment_value is None else str(segment_value), quote=True)
            original_raw = str(row.get("original_text") or row.get("span") or "")
            candidate_raw = str(row.get("candidate_text") or row.get("suggested_text") or row.get("corrected_text") or "")
            reason_raw = str(row.get("reason") or row.get("evidence_summary") or "")
            confidence = html.escape(str(row.get("confidence") if row.get("confidence") not in {None, ""} else ""), quote=True)
            current_type = str(row.get("correction_type") or "ordinary_word")
            type_options = []
            for value, label_text in [("proper_noun", "专名/工具名"), ("concept", "概念"), ("ordinary_word", "普通错词"), ("number", "数字/金额"), ("action", "动作/步骤"), ("punctuation", "标点/断句"), ("segment_boundary", "分段边界")]:
                selected = " selected" if value == current_type else ""
                type_options.append('<option value="' + html.escape(value, quote=True) + '"' + selected + '>' + html.escape(label_text) + '</option>')
            search_text = html.escape(" ".join([label, row_id_raw, original_raw, candidate_raw, reason_raw, current_type]).lower(), quote=True)
            rows.append(
                '<div class="semantic-candidate-suggestion-row" data-suggestion-id="' + row_id + '" data-source-label="' + source_label + '" data-source-path="' + source_path + '" data-search-text="' + search_text + '">'
                + '<div class="semantic-row-head"><div><strong><code>' + row_id + '</code></strong> <span class="badge">' + html.escape(label) + '</span></div><label class="inline-check"><input type="checkbox" data-field="include" checked> 导入为候选</label></div>'
                + '<label>Segment<input data-field="source_segment_index" value="' + segment + '" placeholder="source_segment_index"></label>'
                + '<label>类型<select data-field="correction_type">' + "".join(type_options) + '</select></label>'
                + '<label>疑似原文<textarea data-field="original_text" rows="2">' + html.escape(original_raw) + '</textarea></label>'
                + '<label>建议候选<textarea data-field="candidate_text" rows="2">' + html.escape(candidate_raw) + '</textarea></label>'
                + '<label>置信度<input data-field="confidence" value="' + confidence + '" placeholder="0.0-1.0"></label>'
                + '<label>理由/证据摘要<textarea data-field="reason" rows="2">' + html.escape(reason_raw) + '</textarea></label>'
                + '</div>'
            )
    if not rows:
        return ""
    return (
        '<h3>候选建议编辑器</h3>'
        + '<div class="subpanel semantic-candidate-suggestion-editor"><p class="muted">编辑后复制或下载标准 suggestions JSON，保存为 <code style="display:inline;padding:2px 5px">transcript-semantic-candidate-suggestions.codex.md</code> 或 JSON，再运行 import-transcript-semantic-candidate-suggestions。这里只生成候选，不会 validate 或 closure。</p>'
        + '<div class="semantic-batch-filter"><label>搜索<input id="semanticCandidateSuggestionTextFilter" placeholder="来源、原文、建议、理由" oninput="filterSemanticCandidateSuggestions()"></label><label>类型<input id="semanticCandidateSuggestionTypeFilter" placeholder="proper_noun / concept" oninput="filterSemanticCandidateSuggestions()"></label></div>'
        + '<div class="semantic-review-toolbar"><button type="button" onclick="setSemanticCandidateSuggestionsIncluded(true)">可见项设为导入</button><button type="button" onclick="setSemanticCandidateSuggestionsIncluded(false)">可见项设为不导入</button><span class="muted" id="semanticCandidateSuggestionVisibleCount"></span></div>'
        + '<div class="semantic-review-toolbar"><button type="button" onclick="copySemanticCandidateSuggestions()">复制 suggestions JSON</button><button type="button" onclick="downloadSemanticCandidateSuggestions()">下载 suggestions JSON</button></div>'
        + '<div class="semantic-candidate-suggestion-list">' + "".join(rows) + '</div>'
        + '<textarea id="semanticCandidateSuggestionsOutput" class="json-output" rows="8" readonly></textarea></div>'
    )
def _read_json_or_markdown_object(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        return {}, ""
    try:
        if path.suffix.lower() == ".json":
            data = read_json(path)
            return (data if isinstance(data, dict) else {}), "" if isinstance(data, dict) else "json root is not object"
        text = path.read_text(encoding="utf-8-sig")
        try:
            data = json.loads(text)
            return (data if isinstance(data, dict) else {}), "" if isinstance(data, dict) else "json root is not object"
        except json.JSONDecodeError:
            pass
        payload_text = ""
        marker = "```json"
        if marker in text:
            start = text.find(marker) + len(marker)
            end = text.find("```", start)
            payload_text = text[start:end if end >= 0 else len(text)].strip()
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end >= start:
                payload_text = text[start:end + 1]
        if not payload_text:
            return {}, "no JSON object found"
        data = json.loads(payload_text)
        return (data if isinstance(data, dict) else {}), "" if isinstance(data, dict) else "json root is not object"
    except Exception as exc:
        return {}, str(exc)
def _semantic_correction_review_editor_html(status: dict[str, Any]) -> str:
    rows = status.get("review_required_items") if isinstance(status.get("review_required_items"), list) else []
    if not rows:
        rows = status.get("review_required_preview") if isinstance(status.get("review_required_preview"), list) else []
    if not rows:
        return ""
    body = []
    for row in rows:
        raw_correction_type = str(row.get("correction_type") or "")
        candidate_id = html.escape(str(row.get("candidate_id") or ""))
        correction_type = html.escape(raw_correction_type)
        time_range = html.escape(str(row.get("time_range") or ""))
        original = html.escape(str(row.get("original_text") or ""))
        suggested = html.escape(str(row.get("suggested_text") or ""))
        reasons = html.escape(", ".join(str(item) for item in row.get("reject_reasons") or []))
        structure_html = ""
        if raw_correction_type in {"punctuation", "segment_boundary"}:
            try:
                start_value = float(row.get("start") or 0.0)
            except Exception:
                start_value = 0.0
            try:
                end_value = float(row.get("end") or start_value)
            except Exception:
                end_value = start_value
            mid_value = start_value + max(0.0, end_value - start_value) / 2
            split_example = json.dumps(
                [
                    {"start": round(start_value, 3), "end": round(mid_value, 3), "text": "第一段纠正文"},
                    {"start": round(mid_value, 3), "end": round(end_value, 3), "text": "第二段纠正文"},
                ],
                ensure_ascii=False,
                indent=2,
            )
            segment_index = row.get("segment_index")
            merge_placeholder = "0,1"
            if isinstance(segment_index, int):
                merge_placeholder = f"{segment_index},{segment_index + 1}"
            structure_html = (
                '<details class="semantic-structure-editor"><summary>结构化断句/合并（可选）</summary>'
                + '<p class="muted">用于一个 ASR 段拆成多段，或多个短 ASR 段合并为一个语义段。留空则只做普通纠正文。</p>'
                + '<label>拆分 segments JSON<textarea data-field="segments" rows="5" placeholder="' + html.escape(split_example, quote=True) + '"></textarea></label>'
                + '<label>合并 segment indexes<input data-field="merge_segment_indexes" placeholder="' + html.escape(merge_placeholder, quote=True) + '"></label>'
                + '</details>'
            )
        body.append(
            '<div class="semantic-review-row" data-candidate-id="' + candidate_id + '" data-correction-type="' + correction_type + '">'
            + '<div><strong><code>' + candidate_id + '</code></strong> <span class="badge">' + correction_type + '</span> <span class="muted">' + time_range + '</span></div>'
            + '<label>处理状态<select data-field="status"><option value="accept_correction">接受纠正</option><option value="keep_original">保留原文</option><option value="needs_more_evidence">需要更多证据</option><option value="needs_rerun_asr">重跑 ASR</option><option value="needs_rerun_ocr">重跑 OCR/图文</option></select></label>'
            + '<label>纠正文<textarea data-field="corrected_text" rows="2">' + suggested + '</textarea></label>'
            + structure_html
            + '<label>审核备注<textarea data-field="comment" rows="2">' + reasons + '</textarea></label>'
            + '<details><summary>原文和证据摘要</summary><div class="snippet">' + original + '</div></details>'
            + '</div>'
        )
    return '<div class="subpanel semantic-review-editor"><strong>语义纠错人工编辑表单</strong><p class="muted">编辑后复制或下载 JSON，保存为 <code style="display:inline;padding:2px 5px">transcript-semantic-correction-review-notes.json</code>，再运行 import-transcript-semantic-review-notes。</p><div class="semantic-review-toolbar"><button type="button" onclick="copySemanticReviewNotes()">复制 review notes JSON</button><button type="button" onclick="downloadSemanticReviewNotes()">下载 review notes JSON</button></div><div class="semantic-review-list">' + "".join(body) + '</div><textarea id="semanticReviewNotesOutput" class="json-output" rows="8" readonly></textarea></div>'
def _semantic_batch_review_html(batch: dict[str, Any]) -> str:
    if not batch:
        return ""
    paths = batch.get("paths") if isinstance(batch.get("paths"), dict) else {}
    commands = batch.get("commands") if isinstance(batch.get("commands"), dict) else {}
    counts = batch.get("draft_by_review_status") if isinstance(batch.get("draft_by_review_status"), dict) else {}
    validation_counts = batch.get("import_by_validation_status") if isinstance(batch.get("import_by_validation_status"), dict) else {}
    next_action_counts = batch.get("import_post_next_action_counts") if isinstance(batch.get("import_post_next_action_counts"), dict) else {}
    cards = [
        ("Batch review", batch.get("status") or "missing"),
        ("Review items", batch.get("review_item_count") or 0),
        ("Todo rows", batch.get("todo_review_count") or 0),
        ("Codex draft", batch.get("draft_review_count") or 0),
        ("Imported review notes", batch.get("imported_decision_count") or 0),
        ("Accepted", batch.get("imported_accepted_decision_count") or 0),
        ("Still review open", batch.get("imported_review_required_count") or 0),
        ("Closure-ready bundles", batch.get("imported_closure_ready_bundle_count") or 0),
        ("Open-review bundles", batch.get("imported_open_review_bundle_count") or 0),
        ("Skipped", batch.get("skipped_count") or 0),
    ]
    card_html = "".join('<div><span class="muted">' + html.escape(str(label)) + '</span><strong>' + html.escape(str(value)) + '</strong></div>' for label, value in cards)
    count_html = "".join('<div><span class="muted">' + html.escape(str(key)) + '</span><strong>' + html.escape(str(value)) + '</strong></div>' for key, value in sorted(counts.items())) or '<div class="muted">尚无 Codex 草稿状态统计。</div>'
    validation_html = "".join('<div><span class="muted">' + html.escape(str(key)) + '</span><strong>' + html.escape(str(value)) + '</strong></div>' for key, value in sorted(validation_counts.items())) or '<div class="muted">尚无导入校验状态。</div>'
    next_action_html = "".join('<div><span class="muted">' + html.escape(str(key)) + '</span><strong>' + html.escape(str(value)) + '</strong></div>' for key, value in sorted(next_action_counts.items())) or '<div class="muted">尚无导入后下一步统计。</div>'
    next_actions = [str(item) for item in (batch.get("import_next_actions") if isinstance(batch.get("import_next_actions"), list) else []) if str(item).strip()]
    next_action_rows = "".join('<li><code>' + html.escape(item) + '</code></li>' for item in next_actions[:8]) or '<li class="muted">暂无导入后推荐动作。</li>'
    batch_editor_html = _semantic_batch_review_editor_html(batch)
    path_rows = "".join('<div><span class="muted">' + html.escape(str(key)) + '</span><code>' + html.escape(str(value)) + '</code></div>' for key, value in paths.items() if str(value).strip())
    command_rows = "".join(
        '<div class="command"><div><strong>' + html.escape(str(key)) + '</strong></div><code>' + html.escape(str(value)) + '</code><button type="button" onclick="navigator.clipboard.writeText(this.previousElementSibling.innerText)">复制</button></div>'
        for key, value in commands.items()
        if str(value).strip()
    )
    return (
        '<section class="panel semantic-batch-review">'
        + '<h2>通用语义纠错批量复核包</h2>'
        + '<p class="muted">用于集中处理低置信、证据不足或需要 Codex/人工判断的 ASR/字幕疑似错词。生成 pack 不调用云模型；Codex 草稿是保守本地规则，导入后仍需要 closure 才会写入纠正版 transcript。</p>'
        + '<div class="grid">' + card_html + '</div>'
        + '<h3>Codex 草稿状态</h3><div class="grid">' + count_html + '</div>'
        + '<h3>导入后校验状态</h3><div class="grid">' + validation_html + '</div>'
        + '<h3>导入后下一步</h3><div class="grid">' + next_action_html + '</div><ul>' + next_action_rows + '</ul>'
        + batch_editor_html
        + '<h3>产物路径</h3><div class="grid">' + path_rows + '</div>'
        + '<h3>操作命令</h3><div class="grid">' + command_rows + '</div>'
        + '</section>'
    )

def _semantic_batch_review_editor_html(batch: dict[str, Any]) -> str:
    rows = batch.get("editable_reviews") if isinstance(batch.get("editable_reviews"), list) else []
    if not rows:
        return ""
    truncated = bool(batch.get("editable_review_truncated"))
    body: list[str] = []
    for row in rows:
        review_id = html.escape(str(row.get("review_id") or ""), quote=True)
        bundle_dir = html.escape(str(row.get("bundle_dir") or ""), quote=True)
        candidate_id = html.escape(str(row.get("candidate_id") or ""), quote=True)
        correction_type = html.escape(str(row.get("correction_type") or "ordinary_word"), quote=True)
        evidence_ids = html.escape(json.dumps(row.get("evidence_ids") or [], ensure_ascii=False), quote=True)
        title = html.escape(str(row.get("bundle_title") or ""))
        risk = html.escape(str(row.get("risk_level") or "unknown"))
        raw_time_range = row.get("time_range")
        time_label = _batch_review_time_label(raw_time_range)
        time_range = html.escape(time_label)
        start_seconds = _batch_review_start_seconds(raw_time_range)
        start_attr = html.escape(("" if start_seconds is None else f"{start_seconds:.3f}"), quote=True)
        play_control = (
            '<button type="button" class="seek-button" onclick="seekToSemanticBatchReview(&quot;' + review_id + '&quot;, ' + f"{start_seconds:.3f}" + ')">播放此处</button>'
            if start_seconds is not None
            else '<span class="muted">无可跳转时间</span>'
        )
        original_raw = str(row.get("original_text") or "")
        context_raw = str(row.get("context_text") or "")
        original = html.escape(original_raw)
        suggested_raw = str(row.get("suggested_text") or row.get("corrected_text") or "")
        corrected = html.escape(str(row.get("corrected_text") or suggested_raw))
        note = html.escape(str(row.get("review_note") or ""))
        confidence = html.escape(str(row.get("confidence") if row.get("confidence") not in {None, ""} else ""), quote=True)
        current_status = str(row.get("review_status") or "needs_more_evidence")
        context = html.escape(context_raw)
        evidence = row.get("evidence") if isinstance(row.get("evidence"), list) else []
        evidence_html = _semantic_batch_evidence_html(evidence[:8])
        options = []
        for value, label in [("accept_correction", "接受纠正"), ("keep_original", "保留原文"), ("needs_more_evidence", "需要更多证据"), ("needs_rerun_asr", "重跑 ASR"), ("needs_rerun_ocr", "重跑 OCR/图文")]:
            selected = " selected" if value == current_status else ""
            options.append('<option value="' + value + '"' + selected + '>' + label + '</option>')
        search_text = html.escape(" ".join([str(row.get("candidate_id") or ""), str(row.get("bundle_title") or ""), original_raw, str(row.get("suggested_text") or ""), context_raw, str(row.get("review_note") or ""), time_label]).lower(), quote=True)
        original_attr = html.escape(original_raw[:240], quote=True)
        context_attr = html.escape(context_raw[:240], quote=True)
        body.append(
            '<div class="semantic-batch-review-row" data-review-id="' + review_id + '" data-bundle-dir="' + bundle_dir + '" data-candidate-id="' + candidate_id + '" data-correction-type="' + correction_type + '" data-risk-level="' + risk + '" data-review-status="' + html.escape(current_status, quote=True) + '" data-search-text="' + search_text + '" data-evidence-ids="' + evidence_ids + '" data-start-seconds="' + start_attr + '" data-time-range="' + html.escape(time_label, quote=True) + '" data-original-text="' + original_attr + '" data-context-text="' + context_attr + '">'
            + '<div class="semantic-row-head"><div><strong><code>' + candidate_id + '</code></strong> <span class="badge">' + correction_type + '</span> <span class="badge">risk=' + risk + '</span> <span class="muted">' + time_range + '</span></div>' + play_control + '</div>'
            + '<div class="muted">' + title + '</div>'
            + '<label>处理状态<select data-field="review_status">' + "".join(options) + '</select></label>'
            + '<label>纠正文<textarea data-field="corrected_text" rows="2">' + corrected + '</textarea></label>'
            + '<label>置信度<input data-field="confidence" value="' + confidence + '" placeholder="0.0-1.0"></label>'
            + '<label>审核备注<textarea data-field="review_note" rows="2">' + note + '</textarea></label>'
            + '<details><summary>上下文和证据</summary><div class="snippet"><strong>原文：</strong>' + original + '</div><div class="snippet"><strong>上下文：</strong>' + context + '</div>' + evidence_html + '</details>'
            + '</div>'
        )
    truncated_note = '<p class="muted">页面只显示前 80 条；完整候选仍在 batch review JSON 中。</p>' if truncated else ''
    filter_html = (
        '<div class="semantic-batch-filter"><label>状态<select id="semanticBatchStatusFilter" onchange="filterSemanticBatchReviews()"><option value="all">全部</option><option value="accept_correction">接受纠正</option><option value="keep_original">保留原文</option><option value="needs_more_evidence">需要更多证据</option><option value="needs_rerun_asr">重跑 ASR</option><option value="needs_rerun_ocr">重跑 OCR/图文</option></select></label>'
        + '<label>风险<select id="semanticBatchRiskFilter" onchange="filterSemanticBatchReviews()"><option value="all">全部</option><option value="high">high</option><option value="medium">medium</option><option value="low">low</option><option value="unknown">unknown</option></select></label>'
        + '<label>类型<input id="semanticBatchTypeFilter" placeholder="proper_noun / ordinary_word" oninput="filterSemanticBatchReviews()"></label>'
        + '<label>搜索<input id="semanticBatchTextFilter" placeholder="候选 ID、原文、建议、上下文" oninput="filterSemanticBatchReviews()"></label></div>'
        + '<div class="semantic-review-toolbar"><button type="button" onclick="setSemanticBatchVisibleStatus(&quot;keep_original&quot;)">可见项设为保留原文</button><button type="button" onclick="setSemanticBatchVisibleStatus(&quot;needs_more_evidence&quot;)">可见项设为需要更多证据</button><button type="button" onclick="setSemanticBatchVisibleStatus(&quot;needs_rerun_asr&quot;)">可见项设为重跑 ASR</button><span class="muted" id="semanticBatchVisibleCount"></span></div>'
    )
    return (
        '<h3>批量复核编辑器</h3>'
        + '<div class="subpanel semantic-batch-review-editor"><p class="muted">在这里逐条修改低置信候选，然后复制或下载 JSON，再运行 batch import。不会绕过 validation / closure。</p>'
        + truncated_note
        + filter_html
        + '<div class="semantic-review-toolbar"><button type="button" onclick="copySemanticBatchReviewNotes()">复制 batch review notes JSON</button><button type="button" onclick="downloadSemanticBatchReviewNotes()">下载 batch review notes JSON</button></div>'
        + '<div class="semantic-batch-review-list">' + "".join(body) + '</div>'
        + '<textarea id="semanticBatchReviewNotesOutput" class="json-output" rows="8" readonly></textarea></div>'
    )


def _semantic_repair_queue_html(queue: dict[str, Any], bridge: dict[str, Any] | None = None, latest_run: dict[str, Any] | None = None) -> str:
    if not queue:
        return '<section class="panel muted">尚未生成通用语义纠错修复/重试队列；刷新任务控制台或运行 transcript-semantic-repair-queue。</section>'
    summary = queue.get("summary") if isinstance(queue.get("summary"), dict) else {}
    items = [item for item in (queue.get("items") or []) if isinstance(item, dict)]
    cards = [
        ("队列状态", queue.get("status") or "missing"),
        ("Bundle", queue.get("bundle_count") or len(items)),
        ("需动作", summary.get("action_required_count") or 0),
        ("机器可做", summary.get("machine_action_available_count") or 0),
        ("人工复核", summary.get("human_review_required_count") or 0),
        ("Preview", "yes" if (queue.get("operator_boundary") or {}).get("preview_only", True) else "no"),
    ]
    card_html = "".join(
        '<div><span class="muted">' + html.escape(str(label)) + '</span><strong>' + html.escape(str(value)) + '</strong></div>'
        for label, value in cards
    )
    rows: list[str] = []
    for item in items[:12]:
        progress = item.get("progress") if isinstance(item.get("progress"), dict) else {}
        percent = int(progress.get("percent") or 0)
        retry = str(item.get("retry_command") or "").strip()
        command_html = (
            '<div class="command"><code>' + html.escape(retry) + '</code><button type="button" onclick="navigator.clipboard.writeText(this.previousElementSibling.innerText)">复制</button></div>'
            if retry
            else '<div class="muted">无重试命令</div>'
        )
        rows.append(
            '<div class="queue-row">'
            + '<div><strong>' + html.escape(str(item.get("action_key") or "none")) + '</strong> '
            + '<span class="badge">' + html.escape(str(item.get("action_status") or "unknown")) + '</span> '
            + '<span class="badge">' + html.escape(str(item.get("action_kind") or "")) + '</span></div>'
            + '<div class="muted">' + html.escape(str(item.get("bundle_dir") or "")) + '</div>'
            + '<div class="progress"><div style="width:' + str(max(0, min(100, percent))) + '%"></div></div>'
            + '<div class="grid"><div><span class="muted">进度</span><strong>' + str(percent) + '%</strong></div>'
            + '<div><span class="muted">语义状态</span><strong>' + html.escape(str(item.get("semantic_status") or "")) + '</strong></div>'
            + '<div><span class="muted">LLM</span><strong>' + html.escape(str(item.get("llm_draft_status") or "")) + '</strong></div>'
            + '<div><span class="muted">人工</span><strong>' + ("yes" if item.get("human_review_required") else "no") + '</strong></div></div>'
            + '<p class="muted">' + html.escape(str(item.get("reason") or "")) + '</p>'
            + command_html
            + '</div>'
        )
    if len(items) > 12:
        rows.append('<div class="muted">还有 ' + str(len(items) - 12) + ' 条未显示；打开 JSON 查看完整队列。</div>')
    rows_html = "".join(rows) or '<div class="muted">队列为空。</div>'
    artifacts = [str(queue.get("json_path") or ""), str(queue.get("markdown_path") or "")]
    artifact_html = "".join('<code>' + html.escape(path) + '</code>' for path in artifacts if path.strip())
    bridge_panel = _semantic_repair_bridge_html(bridge or {})
    latest_run_panel = _semantic_repair_latest_run_html(latest_run or {})
    return f"""
    <section class=\"panel\">
      <h2>通用语义纠错修复/重试队列</h2>
      <p class=\"muted\">默认 preview-only：这里显示每个 bundle 的下一步、进度、失败/人工分流和可复制命令；下方按钮只通过本机 bridge 执行安全本地动作，不会自动调用云 LLM。</p>
      <div class=\"grid\">{card_html}</div>
      <div class=\"grid\">{artifact_html}</div>
      {bridge_panel}
      {latest_run_panel}
      <div class=\"queue-list\">{rows_html}</div>
    </section>
    """

def _semantic_repair_latest_run_html(run: dict[str, Any]) -> str:
    if not run:
        return ""
    status = str(run.get("status") or "missing")
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    executions = [item for item in (run.get("executions") or []) if isinstance(item, dict)]
    cards = [
        ("最新运行", status),
        ("动作", summary.get("action_count") or 0),
        ("已执行", summary.get("executed_count") or 0),
        ("预览", summary.get("planned_count") or 0),
        ("失败", summary.get("failed_count") or 0),
        ("需人工", summary.get("operator_required_count") or 0),
    ]
    card_html = "".join(
        '<div><span class="muted">' + html.escape(str(label)) + '</span><strong>' + html.escape(str(value)) + '</strong></div>'
        for label, value in cards
    )
    rows = []
    for item in executions[:8]:
        reason = str(item.get("error") or item.get("reason") or item.get("result_status") or "")
        rows.append(
            '<tr>'
            + '<td><code>' + html.escape(str(item.get("action_key") or "")) + '</code></td>'
            + '<td><code>' + html.escape(str(item.get("run_status") or "")) + '</code></td>'
            + '<td>' + html.escape("yes" if item.get("executed") else "no") + '</td>'
            + '<td>' + html.escape(reason) + '</td>'
            + '</tr>'
        )
    rows_html = "".join(rows) or '<tr><td colspan="4" class="muted">还没有执行明细。</td></tr>'
    artifacts = [str(run.get("json_path") or ""), str(run.get("markdown_path") or "")]
    artifact_html = "".join('<code>' + html.escape(path) + '</code>' for path in artifacts if path.strip())
    return (
        '<div class="subpanel semantic-repair-run">'
        + '<strong>语义纠错最新执行结果</strong>'
        + '<p class="muted">刷新控制台后会读取持久化 repair-run JSON，显示上一轮安全本地动作的执行、失败、跳过和人工分流。</p>'
        + '<div class="grid">' + card_html + '</div>'
        + '<div class="grid">' + artifact_html + '</div>'
        + '<table><thead><tr><th>动作</th><th>运行状态</th><th>执行</th><th>结果/原因</th></tr></thead><tbody>' + rows_html + '</tbody></table>'
        + '</div>'
    )


def _semantic_correction_status_html(status: dict[str, Any]) -> str:
    if not status:
        return '<section class="panel muted">尚未生成通用转写语义纠错状态；先运行 transcript-semantic-correction-pack。</section>'
    commands = status.get("commands") if isinstance(status.get("commands"), dict) else {}
    artifacts = status.get("artifacts") if isinstance(status.get("artifacts"), dict) else {}
    command_rows = "".join(
        f"<div class=\"command\"><div><strong>{html.escape(str(key))}</strong></div><code>{html.escape(str(value))}</code><button type=\"button\" onclick=\"navigator.clipboard.writeText(this.previousElementSibling.innerText)\">复制</button></div>"
        for key, value in commands.items()
        if str(value).strip()
    ) or '<div class="muted">暂无命令。</div>'
    artifact_rows = "".join(
        f"<div><span class=\"muted\">{html.escape(str(key))}</span><code>{html.escape(str(value))}</code></div>"
        for key, value in artifacts.items()
        if str(value).strip()
    )
    ui_summary_rows = _semantic_correction_ui_summary_html(status)
    detail_rows = _semantic_correction_detail_html(status)
    source_vote_rows = _semantic_correction_source_vote_html(status)
    candidate_group_rows = _semantic_correction_candidate_groups_html(status)
    attention_rows = _semantic_correction_attention_html(status)
    chapter_risk_rows = _semantic_correction_chapter_risk_html(status)
    review_preview_rows = _semantic_correction_review_preview_html(status)
    candidate_discovery_rows = _semantic_candidate_discovery_html(status)
    review_closure_rows = _semantic_correction_review_closure_html(status)
    export_chain_rows = _semantic_correction_export_chain_html(status)
    review_editor_rows = _semantic_correction_review_editor_html(status)
    return f"""
    <section class=\"panel\">
      <h2>通用 ASR/字幕语义纠错</h2>
      <p class=\"muted\">覆盖工具名之外的专名、数字、动作、普通错词和断句。Codex/LLM 只判断 evidence pack；VKP 本地校验后才写入纠正版 transcript。</p>
      <div class=\"grid\">
        <div><span class=\"muted\">状态</span><strong>{html.escape(str(status.get('status') or 'missing'))}</strong></div>
        <div><span class=\"muted\">下一步</span><strong>{html.escape(str(status.get('next_action_key') or ''))}</strong></div>
        <div><span class=\"muted\">候选/接受</span><strong>{int(status.get('candidate_count') or 0)} / {int(status.get('accepted_decision_count') or 0)}</strong></div>
        <div><span class=\"muted\">候选分组</span><strong>{int(status.get('candidate_group_count') or 0)}</strong></div>
        <div><span class=\"muted\">最终残留</span><strong>{int(status.get('final_residual_error_total') or 0)}</strong></div>
        <div><span class=\"muted\">可读文件影响</span><strong>{html.escape(str(status.get('readable_impact_status') or 'missing'))}</strong></div>
        <div><span class=\"muted\">可读文件残留</span><strong>{int(status.get('readable_required_residual_total') or 0)}</strong></div>
        <div><span class=\"muted\">LLM/Codex 草稿</span><strong>{html.escape(str(status.get('llm_draft_status') or 'not_planned'))}</strong></div>
        <div><span class=\"muted\">LLM 下一步</span><strong>{html.escape(str(status.get('llm_draft_next_action') or 'run_llm_draft_preview'))}</strong></div>
        <div><span class=\"muted\">LLM 决策数</span><strong>{int(status.get('llm_draft_decision_count') or 0)}</strong></div>
        <div><span class="muted">候选发现状态</span><strong>{html.escape(str(status.get('candidate_discovery_status') or 'not_planned'))}</strong></div>
        <div><span class="muted">候选发现下一步</span><strong>{html.escape(str(status.get('candidate_discovery_next_action') or 'run_candidate_discovery'))}</strong></div>
        <div><span class="muted">发现片段/建议/导入</span><strong>{int(status.get('candidate_discovery_segment_count') or 0)} / {int(status.get('candidate_discovery_suggestion_count') or 0)} / {int(status.get('candidate_discovery_imported_candidate_count') or 0)}</strong></div>
      </div>
      <div class=\"grid\">{artifact_rows}</div>
      {ui_summary_rows}
      {detail_rows}
      {source_vote_rows}
      {candidate_discovery_rows}
      {attention_rows}
      {candidate_group_rows}
      {chapter_risk_rows}
      {review_preview_rows}
      {review_closure_rows}
      {export_chain_rows}
      {review_editor_rows}
      <div class=\"grid\">{command_rows}</div>
    </section>
    """

def _term_codex_substitute_html(substitute: dict[str, Any]) -> str:
    if not substitute:
        return '<section class="panel muted">尚未生成 Codex 术语/工具名语义仲裁操作契约。先运行 term-arbitration-codex。</section>'
    commands = substitute.get("commands") if isinstance(substitute.get("commands"), dict) else {}
    command_rows = "".join(
        f"<div class=\"command\"><div><strong>{html.escape(str(key))}</strong></div><code>{html.escape(str(value))}</code><button type=\"button\" onclick=\"navigator.clipboard.writeText(this.previousElementSibling.innerText)\">复制</button></div>"
        for key, value in commands.items()
        if str(value).strip()
    ) or '<div class="muted">暂无命令。</div>'
    return f"""
    <section class=\"panel\">
      <h2>Codex 术语/工具名语义仲裁</h2>
      <p class=\"muted\">当前用 Codex 临时代替在线文本 LLM。请把 Prompt 和证据包交给 Codex 做语义判断，保存结果后先预检，再导入闭环。</p>
      <div class=\"grid\">
        <div><span class=\"muted\">Prompt</span><code>{html.escape(str(substitute.get('prompt_markdown') or ''))}</code></div>
        <div><span class=\"muted\">证据包</span><code>{html.escape(str(substitute.get('context_pack_json') or ''))}</code></div>
        <div><span class=\"muted\">结果保存</span><code>{html.escape(str(substitute.get('suggested_result_markdown') or ''))}</code></div>
        <div><span class=\"muted\">验收规则</span><code>{html.escape(str(substitute.get('acceptance_rule') or ''))}</code></div>
      </div>
      <div class=\"grid\">{command_rows}</div>
    </section>
    """


def _codex_vision_session_html(session: dict[str, Any], root: Path) -> str:
    levels = session.get("evidence_levels") if isinstance(session.get("evidence_levels"), dict) else {}
    validation = session.get("validation") if isinstance(session.get("validation"), dict) else {}
    artifacts = session.get("artifacts") if isinstance(session.get("artifacts"), dict) else {}
    links = []
    labels = {
        "prompt_markdown": "Prompt",
        "result_template_json": "结果模板",
        "raw_result_json": "Raw result",
        "import_validation_json": "验证报告",
        "task_pack_json": "Task pack",
    }
    for key, label in labels.items():
        raw = str(artifacts.get(key) or "").strip()
        if not raw:
            continue
        path = Path(raw)
        links.append(f'<a href="{html.escape(_relative_href(path, root))}">{html.escape(label)}</a>')
    pending = str(session.get("phase") or "") == "prepared"
    return f"""
    <section class="panel">
      <h2>Codex 会话原生视觉</h2>
      <p class="muted">只使用 Bundle 内已有帧的交互式任务包；不需要 API Key，不会无人值守调用。</p>
      <div class="grid">
        <div><span class="muted">Session</span><code>{html.escape(str(session.get('session_id') or 'not_prepared'))}</code></div>
        <div><span class="muted">阶段 / 待导入</span><strong>{html.escape(str(session.get('phase') or 'not_prepared'))} / {html.escape('yes' if pending else 'no')}</strong></div>
        <div><span class="muted">验证状态</span><strong>{html.escape(str(validation.get('status') or session.get('status') or 'not_validated'))}</strong></div>
        <div><span class="muted">实际模型</span><strong>{html.escape(str(session.get('model_actual') or 'pending'))}</strong></div>
        <div><span class="muted">证据等级</span><strong>confirmed {int(levels.get('confirmed') or 0)} / inferred {int(levels.get('inferred') or 0)} / unavailable {int(levels.get('unavailable') or 0)} / not_seen {int(levels.get('not_seen') or 0)}</strong></div>
      </div>
      <div class="grid">{' · '.join(links) if links else '<span class="muted">任务包尚未准备。</span>'}</div>
    </section>
    """


def _temporal_tag_delta_html(status: dict[str, Any], root: Path) -> str:
    all_levels = status.get("visual_evidence_levels") if isinstance(status.get("visual_evidence_levels"), dict) else {}
    levels = all_levels.get("continuous_tag") if isinstance(all_levels.get("continuous_tag"), dict) else {}
    manifest = _read_object(root / "manifest.json")
    summary = manifest.get("temporal_tag_delta") if isinstance(manifest.get("temporal_tag_delta"), dict) else {}
    report = _read_object(root / "temporal-tag-delta.json")
    run_id = str(summary.get("run_id") or "")
    links: list[str] = []
    for path, label in (
        (root / "temporal-tag-delta.md", "报告"),
        (root / "temporal-tag-delta.json", "Raw/JSON"),
        (root / "temporal-tag-delta-input-template.json", "结果模板"),
        (root / str(summary.get("task_pack") or ""), "Task pack"),
        (root / str(summary.get("raw_result") or ""), "Raw result"),
        (root / str(summary.get("validation_report") or ""), "验证报告"),
        (root / str(summary.get("run_audit") or ""), "运行审计"),
        (root / str(summary.get("restore_plan") or ""), "Restore"),
    ):
        if path.is_file():
            links.append(f'<a href="{html.escape(_relative_href(path, root))}">{label}</a>')
    run_command = f'.\\scripts\\video-knowledge.ps1 run-temporal-tag-delta {_ps_quote(str(root))} --execute-tagger --frame-count 8 --limit 16'
    resume_command = f'{run_command} --resume {run_id}' if run_id else ""
    segment_rows: list[str] = []
    for row in report.get("items") or []:
        if not isinstance(row, dict):
            continue
        transitions = row.get("transitions") if isinstance(row.get("transitions"), list) else []
        appeared = sorted({str(tag) for transition in transitions if isinstance(transition, dict) for tag in transition.get("appeared") or []})
        disappeared = sorted({str(tag) for transition in transitions if isinstance(transition, dict) for tag in transition.get("disappeared") or []})
        segment_rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                int(row.get("index") or 0),
                html.escape("、".join(str(tag) for tag in row.get("stable_tags") or []) or "-"),
                html.escape("、".join(appeared) or "-"),
                html.escape("、".join(disappeared) or "-"),
                html.escape(str(row.get("decision") or "review_gap")),
                html.escape("、".join(str(reason) for reason in row.get("escalation_reasons") or []) or "-"),
            )
        )
    segment_table = (
        '<table><thead><tr><th>片段</th><th>稳定</th><th>出现</th><th>消失</th><th>决策</th><th>理由</th></tr></thead><tbody>'
        + "".join(segment_rows[:16])
        + "</tbody></table>"
        if segment_rows
        else ""
    )
    return f"""
    <section class="panel">
      <h2>连续多帧低成本打标</h2>
      <p class="muted">标签只是 candidate_weak：可说明可能出现的对象/场景，不能证明动作、顺序、意图或因果。动态与冲突片段会升级连续帧多模态。</p>
      <div class="grid">
        <div><span class="muted">状态 / Run</span><strong>{html.escape(str(summary.get('status') or 'not_started'))} / {html.escape(run_id or 'none')}</strong></div>
        <div><span class="muted">弱证据 / 升级</span><strong>{int(levels.get('candidate_weak') or 0)} / {int(levels.get('escalated') or 0)}</strong></div>
        <div><span class="muted">不可用 / 未观察</span><strong>{int(levels.get('unavailable') or 0)} / {int(levels.get('not_seen') or 0)}</strong></div>
        <div><span class="muted">实际模型</span><strong>{html.escape('、'.join(str(value) for value in summary.get('model_actual') or []) or 'pending')}</strong></div>
      </div>
      {segment_table}
      <div>{' · '.join(links) if links else '<span class="muted">尚无连续打标产物。</span>'}</div>
      <code>{html.escape(run_command)}</code>
      {f'<code>{html.escape(resume_command)}</code>' if resume_command else ''}
    </section>
    """


def _autopilot_panel_html(summary: dict[str, Any], root: Path) -> str:
    state = html.escape(str(summary.get("state") or "not_started"))
    route = summary.get("execution_route") if isinstance(summary.get("execution_route"), dict) else {}
    budget = summary.get("budget_usage") if isinstance(summary.get("budget_usage"), dict) else {}
    work_item = summary.get("current_work_item") if isinstance(summary.get("current_work_item"), dict) else {}
    provider_wait = summary.get("provider_wait") if isinstance(summary.get("provider_wait"), dict) else {}
    local_provider = summary.get("local_vision_provider") if isinstance(summary.get("local_vision_provider"), dict) else {}
    provider = local_provider.get("provider") if isinstance(local_provider.get("provider"), dict) else {}
    paths = summary.get("paths") if isinstance(summary.get("paths"), dict) else {}
    links: list[str] = []
    for key, label in (
        ("job", "Job"),
        ("events", "Journal"),
        ("latest_raw_result", "Raw result"),
        ("latest_validation", "验证报告"),
        ("run", "Run audit"),
        ("restore_plan", "Restore"),
    ):
        raw = str(paths.get(key) or "").strip()
        if raw:
            links.append(f'<a href="{html.escape(_relative_href(Path(raw), root))}">{label}</a>')
    command = str(summary.get("next_command") or summary.get("start_command") or "")
    if provider_wait and summary.get("job_id"):
        command = f".\\scripts\\video-knowledge.ps1 autopilot-next {summary['job_id']} --retry-provider"
    gaps = summary.get("review_gaps") if isinstance(summary.get("review_gaps"), list) else []
    return f"""
    <section class="panel">
      <h2>小模型 Autopilot</h2>
      <p class="muted">控制器决定下一步、重试与终止；模型每次只收到一个 Hash 绑定任务。终态是 complete_draft，绝不代写 accepted。</p>
      <div class="grid">
        <div><span class="muted">状态 / Runner</span><strong>{state} / {html.escape(str(summary.get('runner') or 'not_started'))}</strong></div>
        <div><span class="muted">实际路由</span><strong>{html.escape(str(route.get('resolved_route') or 'pending'))}</strong></div>
        <div><span class="muted">当前任务</span><strong>{html.escape(str(work_item.get('task_type') or 'none'))}</strong></div>
        <div><span class="muted">调用 / gaps</span><strong>{int(budget.get('model_calls') or 0)} / {len(gaps)}</strong></div>
        <div><span class="muted">本地视觉模型</span><strong>{html.escape(str(provider.get('model') or 'unconfigured'))}</strong></div>
        <div><span class="muted">Provider 等待</span><strong>{html.escape(str(provider_wait.get('reason') or 'none'))}</strong></div>
      </div>
      <p class="muted">{html.escape(str(provider.get('base_url') or ''))} · source={html.escape(str(local_provider.get('provider_config_source') or ''))} · 不自动启动或切换 LM Studio。</p>
      <div>{' · '.join(links) if links else '<span class="muted">尚无运行产物。</span>'}</div>
      {f'<code>{html.escape(command)}</code>' if command else ''}
    </section>
    """


def _render_task_console_html(console: dict[str, Any]) -> str:
    title = html.escape(str(console.get("title") or "Video Knowledge Task Console"))
    status = console.get("status") if isinstance(console.get("status"), dict) else {}
    model_batches = console.get("model_batches") if isinstance(console.get("model_batches"), dict) else {}
    counts = status.get("counts") if isinstance(status.get("counts"), dict) else {}
    term_status = status.get("term_correction") if isinstance(status.get("term_correction"), dict) else {}
    semantic_status = status.get("semantic_correction") if isinstance(status.get("semantic_correction"), dict) else {}
    semantic_queue = status.get("semantic_repair_queue") if isinstance(status.get("semantic_repair_queue"), dict) else {}
    semantic_run = status.get("semantic_repair_run") if isinstance(status.get("semantic_repair_run"), dict) else {}
    semantic_batch_review = status.get("semantic_batch_review") if isinstance(status.get("semantic_batch_review"), dict) else {}
    codex_vision = status.get("codex_vision_session") if isinstance(status.get("codex_vision_session"), dict) else {}
    autopilot = console.get("autopilot") if isinstance(console.get("autopilot"), dict) else {}
    cards = [
        ("时间轴", counts.get("timeline_items", 0)),
        ("ASR/字幕", counts.get("items_with_transcript", 0)),
        ("屏幕文字", counts.get("items_with_visual_text", 0)),
        ("图文结构", counts.get("items_with_structured_visual", 0)),
        ("单帧理解", counts.get("items_with_visual_understanding", 0)),
        ("连续理解", counts.get("items_with_temporal_understanding", 0)),
        ("打标器", counts.get("items_with_tagger_annotations", 0)),
        ("连续弱证据", counts.get("items_with_continuous_tag_weak_evidence", 0)),
        ("时间错位", status.get("timeline_alignment_issue_count", 0)),
        ("人审待处理", status.get("review_open", 0)),
        ("术语闭环", term_status.get("status", "missing")),
        ("Codex预检", term_status.get("term_validation_status") or "missing"),
        ("预检接受/拒绝", f"{int(term_status.get('accepted_validation_decisions') or 0)}/{int(term_status.get('rejected_validation_decisions') or 0)}"),
        ("通用纠错", semantic_status.get("status", "missing")),
        ("通用候选/接受", f"{int(semantic_status.get('candidate_count') or 0)}/{int(semantic_status.get('accepted_decision_count') or 0)}"),
        ("通用残留", int(semantic_status.get("final_residual_error_total") or 0)),
        ("LLM草稿", semantic_status.get("llm_draft_status", "not_planned")),
        ("LLM下一步", semantic_status.get("llm_draft_next_action", "run_llm_draft_preview")),
        ("纠错队列", semantic_queue.get("status", "missing")),
        ("队列动作", (semantic_queue.get("summary") if isinstance(semantic_queue.get("summary"), dict) else {}).get("action_required_count", 0)),
        ("在线批次", model_batches.get("count", 0)),
        ("在线运行中", sum(1 for item in model_batches.get("items") or [] if isinstance(item, dict) and not item.get("terminal"))),
        ("Codex视觉", codex_vision.get("phase") or "not_prepared"),
        ("视觉实际模型", codex_vision.get("model_actual") or "pending"),
        ("Autopilot", autopilot.get("state") or "not_started"),
        ("Autopilot gaps", len(autopilot.get("review_gaps") or [])),
    ]
    card_html = "\n".join(f"<div class=\"metric\"><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>" for label, value in cards)
    term_validation_guidance_html = _term_validation_guidance_html(term_status)
    term_codex_substitute_html = _term_codex_substitute_html(term_status.get("codex_substitute") if isinstance(term_status.get("codex_substitute"), dict) else {})
    semantic_correction_html = _semantic_correction_status_html(semantic_status)
    bridge = console.get("bridge") if isinstance(console.get("bridge"), dict) else {}
    semantic_batch_review_html = _semantic_batch_review_html(semantic_batch_review)
    semantic_repair_queue_html = _semantic_repair_queue_html(semantic_queue, bridge, semantic_run)
    codex_vision_html = _codex_vision_session_html(codex_vision, Path(str(console.get("bundle_dir") or ".")).expanduser().resolve())
    temporal_tag_html = _temporal_tag_delta_html(status, Path(str(console.get("bundle_dir") or ".")).expanduser().resolve())
    autopilot_html = _autopilot_panel_html(autopilot, Path(str(console.get("bundle_dir") or ".")).expanduser().resolve())
    summary_claim_review_html = _summary_claim_review_panel_html(console.get("summary_claim_review") or {})
    summary_stage_html = _summary_stage_panel_html(console.get("summary_stage_coverage") or {})
    citation_digest_html = _citation_digest_panel_html(console.get("citation_digest_integrity") or {})
    root = Path(str(console.get("bundle_dir") or ".")).expanduser().resolve()
    artifacts_html = "\n".join(_artifact_html(artifact, root) for artifact in console.get("artifacts", []) if isinstance(artifact, dict))
    run_history_html = _run_history_html(console.get("run_registry") if isinstance(console.get("run_registry"), dict) else {}, root)
    processing_queue_html = _processing_queue_html(console.get("processing_queue") if isinstance(console.get("processing_queue"), dict) else {}, root)
    subqueue_action_plan_html = _subqueue_action_plan_html(console.get("subqueue_action_plan") if isinstance(console.get("subqueue_action_plan"), dict) else {})
    command_html = "\n".join(_command_html(command) for command in console.get("commands", []) if isinstance(command, dict))
    model_batches_html = _model_batches_html(model_batches)
    moment_search_html = _moment_search_html(console.get("moment_index") if isinstance(console.get("moment_index"), dict) else {})
    moment_search_script_data = json.dumps(console.get("moment_index") or {}, ensure_ascii=False).replace("</", "<\\/")
    bridge_script_data = json.dumps(bridge, ensure_ascii=False).replace("</", "<\\/")
    raw_json = html.escape(str(Path(str(console.get("task_console_json_path") or "")).name or "task-console.json"))
    return f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{title} - VKP 任务控制台</title>
  <style>
    :root {{ color-scheme: light; --bg:#f7f8fa; --panel:#fff; --ink:#172026; --muted:#667085; --line:#d8dee8; --accent:#2557a7; --warn:#995c00; --ok:#0f6b4f; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif; color:var(--ink); background:var(--bg); }}
    header {{ padding:24px 32px 16px; background:var(--panel); border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 8px; font-size:24px; font-weight:650; }}
    h2 {{ margin:28px 0 12px; font-size:18px; }}
    main {{ max-width:1180px; margin:0 auto; padding:20px 24px 36px; }}
    .muted {{ color:var(--muted); }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(128px,1fr)); gap:10px; margin-top:16px; }}
    .metric {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; min-height:72px; }}
    .metric span {{ display:block; color:var(--muted); font-size:13px; }}
    .metric strong {{ display:block; margin-top:6px; font-size:24px; }}
    .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; margin:12px 0; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:10px; }}
    a {{ color:var(--accent); text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
    .command {{ display:grid; grid-template-columns:minmax(170px,240px) 1fr auto; gap:12px; align-items:start; }}
    code {{ display:block; white-space:pre-wrap; word-break:break-word; background:#f1f4f8; border:1px solid var(--line); padding:8px; border-radius:6px; }}
    button {{ border:1px solid var(--line); background:#fff; border-radius:6px; padding:7px 10px; cursor:pointer; }}
    .badge {{ display:inline-block; border:1px solid var(--line); border-radius:999px; padding:2px 8px; font-size:12px; color:var(--muted); margin-right:4px; }}
    .recommended {{ border-color:#a6c8ff; box-shadow:0 0 0 1px #a6c8ff inset; }}
    .safety-cloud_call_requires_confirmation {{ color:var(--warn); }}
    .safety-safe {{ color:var(--ok); }}
    .searchbar {{ display:flex; gap:8px; align-items:center; margin:10px 0 12px; }}
    .searchbar input {{ flex:1; min-width:180px; border:1px solid var(--line); border-radius:6px; padding:9px; }}
    .moment-result {{ border-top:1px solid var(--line); padding:10px 0; }}
    .moment-result:first-child {{ border-top:0; }}
    .queue-row {{ border-top:1px solid var(--line); padding:12px 0; }}
    .queue-row:first-child {{ border-top:0; }}
    .progress {{ height:8px; background:#eef2f7; border-radius:999px; overflow:hidden; margin:8px 0; }}
    .progress div {{ height:100%; background:var(--accent); }}
    .snippet {{ margin-top:6px; line-height:1.55; }}
    .video-console {{ display:grid; grid-template-columns:minmax(320px,1fr) minmax(260px,420px); gap:12px; align-items:start; }}
    .video-player {{ width:100%; aspect-ratio:16/9; min-height:260px; max-height:70vh; background:#000; border-radius:8px; display:block; }}
    .citation-panel {{ min-height:180px; line-height:1.55; }}
    .seek-button {{ margin-left:8px; padding:4px 8px; font-size:12px; }}
    .run-list {{ display:grid; grid-template-columns:1fr; gap:10px; }}
    .run-card {{ border:1px solid var(--line); border-radius:8px; padding:12px; background:#fff; }}
    .run-card.needs_retry {{ border-left:5px solid #b42318; }}
    .run-card.needs_execution {{ border-left:5px solid #995c00; }}
    .run-card.completed {{ border-left:5px solid #0f6b4f; }}
    .queue-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:10px; }}
    .queue-card {{ border:1px solid var(--line); border-radius:8px; padding:12px; background:#fff; min-height:150px; }}
    .queue-card.action_required {{ border-left:5px solid #b42318; }}
    .queue-card.ready {{ border-left:5px solid #0f6b4f; }}
    .queue-card.empty {{ border-left:5px solid #98a2b3; }}
    .subqueues {{ margin-top:10px; display:grid; gap:8px; }}
    .subqueue {{ border:1px dashed var(--line); border-radius:6px; padding:8px; background:#fbfcfe; }}
    .subqueue.action_required {{ border-left:4px solid #b42318; }}
    .subqueue.ready {{ border-left:4px solid #0f6b4f; }}
    .subqueue.empty {{ opacity:.72; }}
    .subqueue-action-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:8px; margin-top:10px; }}
    .subqueue-action {{ border:1px solid var(--line); border-radius:8px; padding:10px; background:#fff; }}
    .subqueue-action.action_required {{ border-left:5px solid #b42318; }}
    .subqueue-action.ready {{ border-left:5px solid #0f6b4f; }}
    .action-head {{ display:flex; align-items:center; justify-content:space-between; gap:10px; }}
    .subqueue.filtered-out {{ display:none; }}
    .semantic-review-list, .semantic-batch-review-list, .semantic-candidate-suggestion-list {{ display:grid; gap:10px; margin-top:10px; }}
    .semantic-review-row, .semantic-batch-review-row, .semantic-candidate-suggestion-row {{ border:1px solid var(--line); border-radius:8px; padding:10px; background:#fbfcfe; display:grid; gap:8px; }}
    .semantic-review-row label, .semantic-batch-review-row label, .semantic-candidate-suggestion-row label {{ display:grid; gap:4px; font-size:13px; color:var(--muted); }}
    .semantic-review-row select, .semantic-review-row textarea, .semantic-review-row input, .semantic-batch-review-row select, .semantic-batch-review-row textarea, .semantic-batch-review-row input, .semantic-candidate-suggestion-row select, .semantic-candidate-suggestion-row textarea, .semantic-candidate-suggestion-row input, .json-output {{ width:100%; border:1px solid var(--line); border-radius:6px; padding:8px; font:inherit; background:#fff; color:var(--ink); }}
    .semantic-review-toolbar {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; align-items:center; }}
    .semantic-batch-filter {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:8px; margin:10px 0; }}
    .semantic-batch-filter label {{ display:grid; gap:4px; font-size:13px; color:var(--muted); }}
    .semantic-batch-filter input, .semantic-batch-filter select {{ width:100%; border:1px solid var(--line); border-radius:6px; padding:8px; font:inherit; background:#fff; color:var(--ink); }}
    .semantic-batch-review-row.filtered-out, .semantic-candidate-suggestion-row.filtered-out {{ display:none; }}
    .inline-check {{ display:flex !important; grid-template-columns:none !important; align-items:center; gap:6px; color:var(--ink) !important; }}
    .inline-check input {{ width:auto !important; }}
    .semantic-batch-review-row.active {{ border-color:#2d5bd1; box-shadow:0 0 0 3px rgba(45,91,209,.12); background:#eef4ff; }}
    .semantic-row-head {{ display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap; }}
    .semantic-evidence-list {{ display:grid; gap:8px; padding-left:18px; }}
    .evidence-path {{ margin-top:4px; color:var(--muted); overflow-wrap:anywhere; }}
    .bridge-controls {{ display:grid; grid-template-columns:1fr auto auto; gap:8px; align-items:end; margin:10px 0; }}
    .bridge-controls label {{ display:grid; gap:4px; color:var(--muted); font-size:13px; }}
    .bridge-controls input {{ width:100%; border:1px solid var(--line); border-radius:6px; padding:8px; }}
    .bridge-log {{ max-height:260px; overflow:auto; }}
    .semantic-structure-editor {{ border:1px dashed var(--line); border-radius:8px; padding:8px; background:#fff; }}
    .json-output {{ margin-top:10px; font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }}
    @media (max-width:760px) {{ header {{ padding:18px; }} main {{ padding:14px; }} .command {{ grid-template-columns:1fr; }} .video-console {{ grid-template-columns:1fr; }} }}
  </style>
  {review_export_script(console)}
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class=\"muted\">VKP 任务控制台。审核主界面仍是 <code style=\"display:inline;padding:2px 5px\">review.html</code>；这里负责选择下一步、复制命令、查看产物。</div>
  </header>
  <main>
    <section class=\"metrics\">{card_html}</section>
    {autopilot_html}
    {summary_claim_review_html}
    {summary_stage_html}
    {citation_digest_html}
    {temporal_tag_html}
    {codex_vision_html}
    {term_validation_guidance_html}
    {term_codex_substitute_html}
    {semantic_correction_html}
    {semantic_batch_review_html}
      {semantic_repair_queue_html}
    <section>
      <h2>处理队列</h2>
      {subqueue_action_plan_html}
      {processing_queue_html}
    </section>
    <section>
      <h2>在线模型批次</h2>
      {model_batches_html}
    </section>
    <section>
      <h2>任务历史</h2>
      {run_history_html}
    </section>
    <section>
      <h2>关键产物</h2>
      <div class=\"grid\">{artifacts_html}</div>
    </section>
    <section>
      <h2>片段搜索</h2>
      {moment_search_html}
    </section>
    <section>
      <h2>下一步命令</h2>
      {command_html}
    </section>
    <section class=\"panel muted\">
      机器可读状态：<a href=\"{raw_json}\">{raw_json}</a>。云视觉、多模态执行和真实下载仍需要显式确认；这个页面不会自动执行命令。
    </section>
  </main>
  <script>
    const MOMENT_INDEX = {moment_search_script_data};
    const SEMANTIC_REPAIR_BRIDGE = {bridge_script_data};
    async function copyCommand(id) {{
      const el = document.getElementById(id);
      if (!el) return;
      await navigator.clipboard.writeText(el.innerText);
    }}
    function filterSubqueue(key) {{
      const rows = Array.from(document.querySelectorAll(".subqueue"));
      rows.forEach(row => {{
        const fullKey = row.getAttribute("data-subqueue-full-key") || "";
        row.classList.toggle("filtered-out", key !== "all" && fullKey !== key);
      }});
      if (key !== "all") {{
        const target = document.querySelector(`.subqueue[data-subqueue-full-key="${{CSS.escape(key)}}"]`);
        if (target) target.scrollIntoView({{ behavior:"smooth", block:"center" }});
      }}
    }}
    function searchMoments() {{
      const input = document.getElementById("momentSearchInput");
      const results = document.getElementById("momentSearchResults");
      if (!input || !results) return;
      const q = input.value.trim().toLowerCase();
      const chunks = Array.isArray(MOMENT_INDEX.chunks) ? MOMENT_INDEX.chunks : [];
      const selected = chunks.filter(row => !q || row.search_text.includes(q)).slice(0, 20);
      results.innerHTML = selected.map(renderMomentResult).join("") || "<div class=\"muted\">没有匹配片段。换一个术语、工具名或疑难点试试。</div>";
    }}
    function renderMomentResult(row) {{
      const badges = [row.has_visual_evidence ? "视觉证据" : "无视觉证据", row.has_temporal_evidence ? "连续理解" : "无连续理解"].map(v => '<span class="badge">' + escapeHtml(v) + '</span>').join("");
      const indexes = Array.isArray(row.timeline_indexes) ? row.timeline_indexes.join(",") : "";
      return '<div class="moment-result" id="moment-' + escapeHtml(row.chunk_index) + '"><strong>' + escapeHtml(row.start_time) + ' - ' + escapeHtml(row.end_time) + '</strong><button type="button" class="seek-button" onclick="seekToMoment(' + Number(row.start || 0) + ', ' + Number(row.chunk_index || 0) + ')">播放</button> ' + badges + '<div class="muted">timeline: ' + escapeHtml(indexes) + ' | 关键词：' + escapeHtml((row.keywords || []).slice(0,8).join("、")) + '</div><div class="snippet">' + escapeHtml(row.snippet || "") + '</div></div>';
    }}
    function loadConsoleVideo(input) {{ const file = input.files && input.files[0]; if (!file) return; const video = document.getElementById("consoleVideo"); video.src = URL.createObjectURL(file); video.load(); document.getElementById("consoleVideoStatus").textContent = "已加载本地视频：" + file.name; }}
    function seekToMoment(seconds, chunkIndex) {{ const chunks = Array.isArray(MOMENT_INDEX.chunks) ? MOMENT_INDEX.chunks : []; const row = chunks.find(item => Number(item.chunk_index || 0) === Number(chunkIndex || 0)) || {{}}; const video = document.getElementById("consoleVideo"); const start = Math.max(0, Number(seconds || row.start || 0)); document.querySelectorAll(".moment-result").forEach(el => el.style.background = ""); const card = document.getElementById("moment-" + chunkIndex); if (card) {{ card.style.background = "#eef4ff"; card.scrollIntoView({{behavior:"smooth", block:"nearest"}}); }} const panel = document.getElementById("citationPanel"); if (panel) panel.innerHTML = "<strong>当前 citation</strong><div><span class=\"badge\">" + escapeHtml(row.start_time || "") + " - " + escapeHtml(row.end_time || "") + "</span></div><div class=\"muted\">timeline: " + escapeHtml((row.timeline_indexes || []).join(",")) + "</div><div class=\"snippet\">" + escapeHtml(row.snippet || "") + "</div><div class=\"muted\">证据：" + escapeHtml((row.evidence_paths || []).slice(0,4).join(" | ")) + "</div>"; if (!video || !video.src) {{ document.getElementById("consoleVideoStatus").textContent = "已选中片段，但没有加载视频。请先选择本地视频文件。"; return; }} video.currentTime = start; video.play().catch(() => {{}}); document.getElementById("consoleVideoStatus").textContent = "已跳转到 " + (row.start_time || start + "s") + "。"; }}
    function seekToSemanticBatchReview(reviewId, seconds) {{
      const rows = Array.from(document.querySelectorAll(".semantic-batch-review-row"));
      const row = rows.find(item => item.getAttribute("data-review-id") === String(reviewId));
      rows.forEach(item => item.classList.remove("active"));
      const start = Math.max(0, Number(seconds || (row ? row.getAttribute("data-start-seconds") : 0) || 0));
      if (row) {{
        row.classList.add("active");
        row.scrollIntoView({{behavior:"smooth", block:"center"}});
      }}
      const panel = document.getElementById("citationPanel");
      if (panel && row) {{
        panel.innerHTML = "<strong>当前语义纠错候选</strong><div><span class=\"badge\">" + escapeHtml(row.getAttribute("data-candidate-id") || "") + "</span><span class=\"badge\">" + escapeHtml(row.getAttribute("data-correction-type") || "") + "</span></div><div class=\"muted\">" + escapeHtml(row.getAttribute("data-time-range") || "") + "</div><div class=\"snippet\">" + escapeHtml(row.getAttribute("data-original-text") || "") + "</div><div class=\"muted\">上下文：" + escapeHtml(row.getAttribute("data-context-text") || "") + "</div>";
      }}
      const status = document.getElementById("consoleVideoStatus");
      const video = document.getElementById("consoleVideo");
      if (!video || !video.src) {{
        if (status) status.textContent = "已选中语义纠错候选 " + reviewId + "，但没有加载视频。请先选择本地视频文件。";
        return;
      }}
      video.currentTime = start;
      video.play().catch(() => {{}});
      if (status) status.textContent = "已跳转到语义纠错候选 " + reviewId + " @ " + start.toFixed(3) + "s。";
    }}
    function collectSemanticReviewNotes() {{
      const rows = Array.from(document.querySelectorAll(".semantic-review-row"));
      const parseJsonArrayField = value => {{
        const text = String(value || "").trim();
        if (!text) return [];
        try {{
          const parsed = JSON.parse(text);
          return Array.isArray(parsed) ? parsed : [];
        }} catch (err) {{
          return [];
        }}
      }};
      const parseIntegerListField = value => String(value || "").split(/[，, \\t\\r\\n]+/).map(item => Number(item)).filter(item => Number.isInteger(item) && item >= 0);
      const reviews = rows.map(row => {{
        const field = name => row.querySelector(`[data-field="${{name}}"]`);
        const review = {{
          candidate_id: row.getAttribute("data-candidate-id") || "",
          correction_type: row.getAttribute("data-correction-type") || "ordinary_word",
          status: field("status") ? field("status").value : "needs_more_evidence",
          corrected_text: field("corrected_text") ? field("corrected_text").value.trim() : "",
          comment: field("comment") ? field("comment").value.trim() : "",
        }};
        const segments = parseJsonArrayField(field("segments") ? field("segments").value : "");
        if (segments.length) review.segments = segments;
        const mergeIndexes = parseIntegerListField(field("merge_segment_indexes") ? field("merge_segment_indexes").value : "");
        if (mergeIndexes.length) review.merge_segment_indexes = mergeIndexes;
        return review;
      }}).filter(row => row.candidate_id);
      const payload = {{
        schema: "lecture_review_notes.v1",
        source: "video_knowledge_pipeline.task_console.semantic_review_editor",
        reviews,
      }};
      const text = JSON.stringify(payload, null, 2);
      const output = document.getElementById("semanticReviewNotesOutput");
      if (output) output.value = text;
      return text;
    }}
    async function copySemanticReviewNotes() {{
      const text = collectSemanticReviewNotes();
      await navigator.clipboard.writeText(text);
    }}
    function downloadSemanticReviewNotes() {{
      const text = collectSemanticReviewNotes();
      const blob = new Blob([text], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = vkpReviewExportName('转录审核', JSON.parse(text));
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}
    function collectSemanticBatchReviewNotes() {{
      const rows = Array.from(document.querySelectorAll(".semantic-batch-review-row"));
      const parseEvidenceIds = value => {{
        try {{
          const parsed = JSON.parse(String(value || "[]"));
          return Array.isArray(parsed) ? parsed.map(item => String(item)).filter(Boolean) : [];
        }} catch (err) {{
          return [];
        }}
      }};
      const reviews = rows.map(row => {{
        const field = name => row.querySelector(`[data-field="${{name}}"]`);
        const confidenceRaw = field("confidence") ? field("confidence").value.trim() : "";
        const review = {{
          review_id: row.getAttribute("data-review-id") || "",
          bundle_dir: row.getAttribute("data-bundle-dir") || "",
          candidate_id: row.getAttribute("data-candidate-id") || "",
          correction_type: row.getAttribute("data-correction-type") || "ordinary_word",
          evidence_ids: parseEvidenceIds(row.getAttribute("data-evidence-ids") || "[]"),
          review_status: field("review_status") ? field("review_status").value : "needs_more_evidence",
          corrected_text: field("corrected_text") ? field("corrected_text").value.trim() : "",
          review_note: field("review_note") ? field("review_note").value.trim() : "",
        }};
        if (confidenceRaw) {{
          const value = Number(confidenceRaw);
          review.confidence = Number.isFinite(value) ? value : confidenceRaw;
        }}
        return review;
      }}).filter(row => row.bundle_dir && row.candidate_id);
      const payload = {{
        schema: "video_knowledge_pipeline.transcript_semantic_batch_review_notes.v1",
        source: "video_knowledge_pipeline.task_console.semantic_batch_review_editor",
        reviews,
      }};
      const text = JSON.stringify(payload, null, 2);
      const output = document.getElementById("semanticBatchReviewNotesOutput");
      if (output) output.value = text;
      return text;
    }}
    function filterSemanticBatchReviews() {{
      const statusFilter = document.getElementById("semanticBatchStatusFilter");
      const riskFilter = document.getElementById("semanticBatchRiskFilter");
      const typeFilter = document.getElementById("semanticBatchTypeFilter");
      const textFilter = document.getElementById("semanticBatchTextFilter");
      const statusValue = statusFilter ? String(statusFilter.value || "all") : "all";
      const riskValue = riskFilter ? String(riskFilter.value || "all") : "all";
      const typeValue = typeFilter ? String(typeFilter.value || "").trim().toLowerCase() : "";
      const textValue = textFilter ? String(textFilter.value || "").trim().toLowerCase() : "";
      const rows = Array.from(document.querySelectorAll(".semantic-batch-review-row"));
      let visible = 0;
      rows.forEach(row => {{
        const rowStatus = row.getAttribute("data-review-status") || "";
        const rowRisk = row.getAttribute("data-risk-level") || "unknown";
        const rowType = String(row.getAttribute("data-correction-type") || "").toLowerCase();
        const rowSearch = String(row.getAttribute("data-search-text") || "").toLowerCase();
        const ok = (statusValue === "all" || rowStatus === statusValue)
          && (riskValue === "all" || rowRisk === riskValue)
          && (!typeValue || rowType.includes(typeValue))
          && (!textValue || rowSearch.includes(textValue));
        row.classList.toggle("filtered-out", !ok);
        if (ok) visible += 1;
      }});
      const count = document.getElementById("semanticBatchVisibleCount");
      if (count) count.textContent = "可见 " + visible + " / " + rows.length + " 条";
    }}
    function setSemanticBatchVisibleStatus(status) {{
      const rows = Array.from(document.querySelectorAll(".semantic-batch-review-row"));
      rows.forEach(row => {{
        if (row.classList.contains("filtered-out")) return;
        const select = row.querySelector('[data-field="review_status"]');
        if (select) select.value = status;
        row.setAttribute("data-review-status", status);
      }});
      filterSemanticBatchReviews();
      collectSemanticBatchReviewNotes();
    }}
    function bindSemanticBatchReviewRows() {{
      document.querySelectorAll('.semantic-batch-review-row [data-field="review_status"]').forEach(select => {{
        select.addEventListener("change", event => {{
          const row = event.target.closest(".semantic-batch-review-row");
          if (row) row.setAttribute("data-review-status", event.target.value || "needs_more_evidence");
          filterSemanticBatchReviews();
        }});
      }});
      filterSemanticBatchReviews();
    }}
    async function copySemanticBatchReviewNotes() {{
      const text = collectSemanticBatchReviewNotes();
      await navigator.clipboard.writeText(text);
    }}
    function downloadSemanticBatchReviewNotes() {{
      const text = collectSemanticBatchReviewNotes();
      const blob = new Blob([text], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = vkpReviewExportName('转录批次审核', JSON.parse(text));
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}
    function collectSemanticCandidateSuggestions() {{
      const rows = Array.from(document.querySelectorAll(".semantic-candidate-suggestion-row"));
      const suggestions = rows.map(row => {{
        const field = name => row.querySelector(`[data-field="${{name}}"]`);
        const include = field("include");
        if (include && !include.checked) return null;
        const segmentRaw = field("source_segment_index") ? field("source_segment_index").value.trim() : "";
        const confidenceRaw = field("confidence") ? field("confidence").value.trim() : "";
        const suggestion = {{
          suggestion_id: row.getAttribute("data-suggestion-id") || "",
          source: row.getAttribute("data-source-label") || "task_console_edit",
          source_path: row.getAttribute("data-source-path") || "",
          correction_type: field("correction_type") ? field("correction_type").value : "ordinary_word",
          original_text: field("original_text") ? field("original_text").value.trim() : "",
          candidate_text: field("candidate_text") ? field("candidate_text").value.trim() : "",
          reason: field("reason") ? field("reason").value.trim() : "",
        }};
        if (segmentRaw) {{
          const segmentNumber = Number(segmentRaw);
          suggestion.source_segment_index = Number.isFinite(segmentNumber) ? segmentNumber : segmentRaw;
        }}
        if (confidenceRaw) {{
          const confidenceNumber = Number(confidenceRaw);
          suggestion.confidence = Number.isFinite(confidenceNumber) ? confidenceNumber : confidenceRaw;
        }}
        return suggestion;
      }}).filter(row => row && row.original_text && row.candidate_text);
      const payload = {{
        schema: "video_knowledge_pipeline.transcript_semantic_candidate_suggestions.v1",
        source: "video_knowledge_pipeline.task_console.semantic_candidate_suggestion_editor",
        suggestions,
        operator_boundary: {{
          suggestions_only: true,
          no_validation: true,
          no_closure: true,
          no_transcript_write: true,
        }},
      }};
      const text = JSON.stringify(payload, null, 2);
      const output = document.getElementById("semanticCandidateSuggestionsOutput");
      if (output) output.value = text;
      return text;
    }}
    function filterSemanticCandidateSuggestions() {{
      const textFilter = document.getElementById("semanticCandidateSuggestionTextFilter");
      const typeFilter = document.getElementById("semanticCandidateSuggestionTypeFilter");
      const textValue = textFilter ? String(textFilter.value || "").trim().toLowerCase() : "";
      const typeValue = typeFilter ? String(typeFilter.value || "").trim().toLowerCase() : "";
      const rows = Array.from(document.querySelectorAll(".semantic-candidate-suggestion-row"));
      let visible = 0;
      rows.forEach(row => {{
        const rowSearch = String(row.getAttribute("data-search-text") || "").toLowerCase();
        const typeSelect = row.querySelector('[data-field="correction_type"]');
        const rowType = typeSelect ? String(typeSelect.value || "").toLowerCase() : "";
        const ok = (!textValue || rowSearch.includes(textValue)) && (!typeValue || rowType.includes(typeValue));
        row.classList.toggle("filtered-out", !ok);
        if (ok) visible += 1;
      }});
      const count = document.getElementById("semanticCandidateSuggestionVisibleCount");
      if (count) count.textContent = "可见 " + visible + " / " + rows.length + " 条";
    }}
    function setSemanticCandidateSuggestionsIncluded(included) {{
      const rows = Array.from(document.querySelectorAll(".semantic-candidate-suggestion-row"));
      rows.forEach(row => {{
        if (row.classList.contains("filtered-out")) return;
        const include = row.querySelector('[data-field="include"]');
        if (include) include.checked = !!included;
      }});
      collectSemanticCandidateSuggestions();
      filterSemanticCandidateSuggestions();
    }}
    function bindSemanticCandidateSuggestionRows() {{
      document.querySelectorAll('.semantic-candidate-suggestion-row [data-field]').forEach(input => {{
        input.addEventListener("input", () => collectSemanticCandidateSuggestions());
        input.addEventListener("change", () => {{ collectSemanticCandidateSuggestions(); filterSemanticCandidateSuggestions(); }});
      }});
      filterSemanticCandidateSuggestions();
      collectSemanticCandidateSuggestions();
    }}
    async function copySemanticCandidateSuggestions() {{
      const text = collectSemanticCandidateSuggestions();
      await navigator.clipboard.writeText(text);
    }}
    function downloadSemanticCandidateSuggestions() {{
      const text = collectSemanticCandidateSuggestions();
      const blob = new Blob([text], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = vkpReviewExportName('转录候选审核', JSON.parse(text));
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }}    async function runSemanticRepairViaBridge(executeSafeActions) {{
      const urlInput = document.getElementById("semanticRepairBridgeUrl");
      const output = document.getElementById("semanticRepairBridgeOutput");
      const url = urlInput ? String(urlInput.value || "").trim() : "";
      if (!output) return;
      if (!url) {{
        output.textContent = "Bridge URL 为空。请先启动 VKP OpenClaw HTTP bridge，或填写 /call URL。";
        return;
      }}
      const args = Object.assign({{}}, SEMANTIC_REPAIR_BRIDGE.semantic_repair_run_arguments || {{}});
      args.execute_safe_actions = !!executeSafeActions;
      args.allow_llm = false;
      args.allow_closure = false;
      args.provider_config = {{}};
      args.max_actions = executeSafeActions ? 1 : 0;
      const payload = {{name: "transcript_semantic_repair_run", arguments: args}};
      output.textContent = "Calling " + url + " ...\n" + JSON.stringify(payload, null, 2);
      try {{
        const response = await fetch(url, {{method: "POST", headers: {{"Content-Type": "application/json"}}, body: JSON.stringify(payload)}});
        const text = await response.text();
        let parsed = null;
        try {{ parsed = JSON.parse(text); }} catch (err) {{ parsed = null; }}
        output.textContent = parsed ? JSON.stringify(parsed, null, 2) : text;
      }} catch (err) {{
        output.textContent = "Bridge call failed: " + (err && err.message ? err.message : String(err));
      }}
    }}
    function escapeHtml(value) {{ return String(value || "").replace(/[&<>\"]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\\"":"&quot;"}}[c] || c)); }}
    document.addEventListener("DOMContentLoaded", () => {{ searchMoments(); bindSemanticBatchReviewRows(); bindSemanticCandidateSuggestionRows(); }});
  </script>
</body>
</html>
"""


def _artifact_html(artifact: dict[str, Any], root: Path) -> str:
    label = html.escape(str(artifact.get("label") or artifact.get("key") or "artifact"))
    path = Path(str(artifact.get("path") or ""))
    href_value = _relative_href(path, root)
    href = html.escape(href_value)
    exists = "存在" if artifact.get("exists") else "未生成"
    return f"<div class=\"panel\"><strong>{label}</strong><div class=\"muted\">{html.escape(exists)}</div><a href=\"{href}\">{html.escape(href_value)}</a></div>"


def _command_html(command: dict[str, Any]) -> str:
    key = html.escape(str(command.get("key") or "command"))
    label = html.escape(str(command.get("label") or key))
    command_text = html.escape(str(command.get("command") or command.get("artifact_path") or ""))
    safety = html.escape(str(command.get("safety") or ""))
    phase = html.escape(str(command.get("phase") or ""))
    classes = "panel command"
    if command.get("recommended"):
        classes += " recommended"
    reason = html.escape(str(command.get("reason") or ""))
    copy = f"<button type=\"button\" onclick=\"copyCommand('cmd-{key}')\">复制</button>" if command.get("command") else ""
    artifact = ""
    if command.get("artifact_path"):
        path = Path(str(command.get("artifact_path")))
        artifact = f"<div class=\"muted\">产物：{html.escape(path.name)}</div>"
    return f"""<div class=\"{classes}\">
  <div><strong>{label}</strong><div><span class=\"badge\">{phase}</span><span class=\"badge safety-{safety}\">{safety}</span></div>{artifact}<div class=\"muted\">{reason}</div></div>
  <code id=\"cmd-{key}\">{command_text}</code>
  <div>{copy}</div>
</div>"""


def _relative_href(path: Path, root: Path) -> str:
    try:
        candidate = path if path.is_absolute() else root / path
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        raw = str(path.name or path)
        return raw.replace("\\", "/")





def _safe_timeline_alignment_audit(root: Path, *, write: bool) -> dict[str, Any]:
    try:
        return timeline_alignment_audit(root, write=write)
    except Exception as exc:  # noqa: BLE001
        return {
            "schema": "video_knowledge_pipeline.timeline_alignment_audit.error.v1",
            "bundle_dir": str(root),
            "error": f"{type(exc).__name__}: {exc}",
            "summary": {"items_with_issues": 0, "issue_counts": {}},
            "json_path": str(root / "timeline-alignment-audit.json"),
            "report_path": str(root / "timeline-alignment-audit.md"),
            "mcp_args_path": str(root / "mcp-timeline-alignment-audit.args.json"),
        }


def _compact_timeline_alignment(audit: dict[str, Any]) -> dict[str, Any]:
    summary = audit.get("summary") if isinstance(audit.get("summary"), dict) else {}
    issue_counts = summary.get("issue_counts") if isinstance(summary.get("issue_counts"), dict) else {}
    return {
        "schema": audit.get("schema", ""),
        "available": not bool(audit.get("error")),
        "error": audit.get("error", ""),
        "items": int(summary.get("items") or 0),
        "items_with_issues": int(summary.get("items_with_issues") or 0),
        "missing_asr_overlap": int(summary.get("missing_asr_overlap") or 0),
        "review_start_mismatch": int(summary.get("review_start_mismatch") or 0),
        "tagger_time_conflict": int(summary.get("tagger_time_conflict") or 0),
        "transcript_available": bool(summary.get("transcript_available")),
        "issue_counts": issue_counts,
        "report_path": audit.get("report_path", ""),
        "json_path": audit.get("json_path", ""),
        "mcp_args_path": audit.get("mcp_args_path", ""),
    }

def _safe_build_video_moment_index(root: Path, *, write: bool) -> dict[str, Any]:
    try:
        return build_video_moment_index(root, write=write)
    except Exception as exc:  # noqa: BLE001
        return {
            "schema": "video_knowledge_pipeline.video_moment_index.error.v1",
            "bundle_dir": str(root),
            "error": f"{type(exc).__name__}: {exc}",
            "chunks": [],
            "summary": {},
            "artifacts": {
                "json": str(root / "exports" / "video-moment-index.json"),
                "markdown": str(root / "exports" / "video-moment-index.md"),
            },
        }


def _compact_moment_index(moment_index: dict[str, Any]) -> dict[str, Any]:
    chunks = []
    for chunk in moment_index.get("chunks") or []:
        if not isinstance(chunk, dict):
            continue
        snippet_source = " ".join(
            str(chunk.get(key) or "")
            for key in ("transcript_text", "visual_text", "temporal_text")
        ).strip()
        keywords = [str(value) for value in (chunk.get("keywords") or [])[:16]]
        compact = {
            "chunk_index": chunk.get("chunk_index"),
            "start": chunk.get("start", 0.0),
            "end": chunk.get("end", 0.0),
            "start_time": str(chunk.get("start_time") or ""),
            "end_time": str(chunk.get("end_time") or ""),
            "timeline_indexes": chunk.get("timeline_indexes") or [],
            "keywords": keywords,
            "snippet": _clip_for_console(snippet_source, 520),
            "has_visual_evidence": bool(chunk.get("has_visual_evidence")),
            "has_temporal_evidence": bool(chunk.get("has_temporal_evidence")),
            "evidence_paths": [str(value) for value in (chunk.get("evidence_paths") or [])[:6]],
        }
        compact["search_text"] = " ".join(
            [
                compact["snippet"],
                " ".join(keywords),
                " ".join(str(value) for value in compact["timeline_indexes"]),
            ]
        ).lower()
        chunks.append(compact)
    return {
        "schema": moment_index.get("schema"),
        "summary": moment_index.get("summary") if isinstance(moment_index.get("summary"), dict) else {},
        "artifacts": moment_index.get("artifacts") if isinstance(moment_index.get("artifacts"), dict) else {},
        "error": moment_index.get("error"),
        "chunks": chunks[:500],
    }


def _moment_search_html(moment_index: dict[str, Any]) -> str:
    error = str(moment_index.get("error") or "").strip()
    summary = moment_index.get("summary") if isinstance(moment_index.get("summary"), dict) else {}
    chunks = moment_index.get("chunks") if isinstance(moment_index.get("chunks"), list) else []
    count = len(chunks)
    if error:
        return f'<div class="panel"><strong>片段索引暂不可用</strong><div class="muted">{html.escape(error)}</div></div>'
    meta = f"已索引 {count} 个时间窗"
    if summary:
        meta += f"，覆盖 {html.escape(str(summary.get('timeline_items') or 0))} 条 timeline"
    return "".join([
        '<div class="panel">',
        '<div class="muted">搜索术语、工具名、疑难点或字幕关键词；结果会显示对应时间范围、timeline index 和是否已有视觉/连续证据。</div>',
        '<div class="searchbar"><input id="momentSearchInput" placeholder="例如：工具名、价格、步骤、看屏幕、疑难点" oninput="searchMoments()"><button type="button" onclick="searchMoments()">搜索</button></div>',
        f'<div class="muted">{html.escape(meta)}</div>',
        '<div id="momentSearchResults"></div>',
        '</div>',
    ])


def _clip_for_console(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"

def _render_model_settings_html(console: dict[str, Any]) -> str:
    settings = console.get("model_api_settings") if isinstance(console.get("model_api_settings"), dict) else {}
    current = settings.get("current_profile") if isinstance(settings.get("current_profile"), dict) else {}
    providers = settings.get("providers") if isinstance(settings.get("providers"), list) else []
    asr_runtime = settings.get("asr_runtime") if isinstance(settings.get("asr_runtime"), dict) else {}
    asr_adapters = settings.get("asr_service_adapters") if isinstance(settings.get("asr_service_adapters"), list) else []
    title = html.escape(str(console.get("title") or "VKP"))
    config_path = html.escape(str(settings.get("config_path") or ""))
    provider_rows = "\n".join(_model_provider_row(row) for row in providers if isinstance(row, dict))
    env_commands = "\n\n".join(_model_provider_env_command(row) for row in providers if isinstance(row, dict))
    provider_options = _model_provider_options(providers, str(current.get("provider") or ""))
    current_model = html.escape(str(current.get("model") or ""))
    current_base_url = html.escape(str(current.get("base_url") or ""))
    multimodal_limit = html.escape(str(current.get("multimodal_limit") or 19))
    temporal_limit = html.escape(str(current.get("temporal_limit") or 3))
    frame_count = html.escape(str(current.get("frame_count") or 8))
    escaped_env_commands = html.escape(env_commands)
    bundle_dir_raw = str(console.get("bundle_dir") or "")
    bundle_ps = _ps_quote(bundle_dir_raw) if bundle_dir_raw else "<bundle_dir>"
    current_provider_ps = _ps_quote(str(current.get("provider") or ""))
    provider_matrix_cmd = ".\\scripts\\video-knowledge.ps1 vision-provider-matrix --providers " + _ps_quote("local_qwen_vl,volcengine_coding_plan,gemini,openai,agnes") + " --bundle-dir " + bundle_ps
    provider_smoke_cmd = ".\\scripts\\video-knowledge.ps1 vision-provider-smoke --provider " + current_provider_ps + " --bundle-dir " + bundle_ps
    preflight_cmd = ".\\scripts\\video-knowledge.ps1 vision-execution-preflight " + bundle_ps + " --semantic-limit " + str(current.get("multimodal_limit") or 10) + " --no-temporal"
    volcengine_batch_cmd = ".\\scripts\\run-volcengine-vision-batch.ps1 " + bundle_ps + " -Limit " + str(current.get("multimodal_limit") or 10)
    model_action_commands = "\n".join([
        "# 1. 检测可用 provider",
        provider_matrix_cmd,
        "",
        "# 2. 检测当前 provider",
        provider_smoke_cmd,
        "",
        "# 3. 当前 bundle 多模态预检，不发起真实执行",
        preflight_cmd,
        "",
        "# 4. 火山引擎批处理入口；加 -Execute 才会真实调用 API",
        volcengine_batch_cmd,
    ])
    escaped_model_action_commands = html.escape(model_action_commands)
    asr_settings_panel = _render_asr_runtime_settings_panel(asr_runtime, asr_adapters, bundle_ps)
    return "".join([
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<title>{title} - 模型 API 设置</title>',
        '<style>body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f7f8fa;color:#172026}header{background:#fff;border-bottom:1px solid #d8dee8;padding:24px 32px}main{max-width:1120px;margin:0 auto;padding:20px 24px 36px}.panel{background:#fff;border:1px solid #d8dee8;border-radius:8px;padding:14px;margin:12px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}label{display:block;color:#667085;font-size:13px;margin:10px 0 4px}input,select{width:100%;padding:9px;border:1px solid #d8dee8;border-radius:6px}code,pre{white-space:pre-wrap;word-break:break-word;background:#f1f4f8;border:1px solid #d8dee8;border-radius:6px;padding:8px}button{border:1px solid #d8dee8;background:#fff;border-radius:6px;padding:8px 11px;margin:8px 8px 0 0}.muted{color:#667085}.ok{color:#0f6b4f}.warn{color:#995c00}</style></head><body>',
        '<header><h1>模型 API 设置</h1><div class="muted">保存 provider/model/base_url/批量上限；API key 只放用户环境变量或私有 env 文件。</div></header><main>',
        f'<section class="panel"><strong>当前配置</strong><div class="muted">Config: <code>{config_path}</code></div></section>',
        '<section class="panel"><h2>生成持久化配置命令</h2>',
        f'<label>Provider</label><select id="provider">{provider_options}</select>',
        f'<label>Model</label><input id="model" value="{current_model}">',
        f'<label>Base URL</label><input id="baseUrl" value="{current_base_url}">',
        f'<div class="grid"><div><label>单帧批量上限</label><input id="multimodalLimit" type="number" min="0" value="{multimodal_limit}"></div><div><label>连续片段批量上限</label><input id="temporalLimit" type="number" min="0" value="{temporal_limit}"></div><div><label>连续片段帧数</label><input id="frameCount" type="number" min="5" max="12" value="{frame_count}"></div></div>',
        '<button onclick="buildSaveCommand()">生成命令</button><button onclick="copyText(\'saveCommand\')">复制</button><pre id="saveCommand"></pre></section>',
        f'<section class="panel"><h2>检测与预检命令</h2><div class="muted">这些命令用于验证 provider、生成 preflight 或准备火山批处理；默认不保存密钥，也不会从页面直接调用云端。</div><pre id="modelActionCommands">{escaped_model_action_commands}</pre><button onclick="copyText(\'modelActionCommands\')">复制全部</button></section>',
        asr_settings_panel,
        f'<section><h2>Provider 与 key 状态</h2><div class="grid">{provider_rows}</div></section>',
        f'<section class="panel"><h2>API key 用户级持久化命令</h2><div class="muted">把 &lt;paste-key&gt; 替换成真实 key，在可见 PowerShell 执行。不要提交 key。</div><pre id="envCommands">{escaped_env_commands}</pre><button onclick="copyText(\'envCommands\')">复制全部</button></section>',
        '<section class="panel muted">设置页不会发起模型调用。真实云视觉仍需 preflight 与确认参数。</section></main>',
        '<script>function psQuote(v){return "\'"+String(v||"").replaceAll("\'","\'\'")+"\'"}function buildSaveCommand(){const cmd=[".\\\\scripts\\\\video-knowledge.ps1 set-vision-profile","--provider",psQuote(provider.value),"--model",psQuote(model.value),"--base-url",psQuote(baseUrl.value),"--multimodal-limit",multimodalLimit.value||"0","--temporal-limit",temporalLimit.value||"0","--frame-count",frameCount.value||"8"];saveCommand.innerText=cmd.join(" ")}async function copyText(id){const el=document.getElementById(id);await navigator.clipboard.writeText(el.innerText)}buildSaveCommand()</script>',
        '</body></html>',
    ])


def _render_asr_runtime_settings_panel(asr_runtime: dict[str, Any], adapters: list[Any], bundle_ps: str) -> str:
    provider = html.escape(str(asr_runtime.get("provider") or "funasr_sensevoice"))
    model = html.escape(str(asr_runtime.get("model") or "iic/SenseVoiceSmall"))
    device = html.escape(str(asr_runtime.get("device") or "cuda_preferred"))
    compute_type = html.escape(str(asr_runtime.get("compute_type") or "auto"))
    vad_model = html.escape(str(asr_runtime.get("vad_model") or "fsmn-vad"))
    punc_model = html.escape(str(asr_runtime.get("punc_model") or "ct-punc"))
    spk_model = html.escape(str(asr_runtime.get("spk_model") or ""))
    merge_length = html.escape(str(asr_runtime.get("merge_length_s") or 15))
    audio = asr_runtime.get("audio_preprocess") if isinstance(asr_runtime.get("audio_preprocess"), dict) else {}
    service = asr_runtime.get("openai_compatible") if isinstance(asr_runtime.get("openai_compatible"), dict) else {}
    service_base_url = html.escape(str(service.get("base_url") or "http://127.0.0.1:8000/v1"))
    service_model = html.escape(str(service.get("model") or "Systran/faster-whisper-large-v3"))
    service_timeout = html.escape(str(service.get("timeout_seconds") or 600))
    adapter_rows = "\n".join(_asr_adapter_row(row) for row in adapters if isinstance(row, dict))
    settings_ui_url = html.escape(model_api_settings_ui_url(), quote=True)
    asr_cmd = " ".join([
        ".\\scripts\\video-knowledge.ps1 set-asr-runtime-profile",
        "--provider", provider,
        "--model", _ps_quote(str(asr_runtime.get("model") or "iic/SenseVoiceSmall")),
        "--device", device,
        "--compute-type", compute_type,
        "--vad-model", _ps_quote(str(asr_runtime.get("vad_model") or "fsmn-vad")),
        "--punc-model", _ps_quote(str(asr_runtime.get("punc_model") or "ct-punc")),
        "--spk-model", _ps_quote(str(asr_runtime.get("spk_model") or "")),
        "--enable-vad", _bool_token(asr_runtime.get("enable_vad")),
        "--enable-itn", _bool_token(asr_runtime.get("enable_itn")),
        "--enable-punctuation", _bool_token(asr_runtime.get("enable_punctuation")),
        "--enable-diarization", _bool_token(asr_runtime.get("enable_diarization")),
        "--merge-vad", _bool_token(asr_runtime.get("merge_vad")),
        "--merge-length-s", str(asr_runtime.get("merge_length_s") or 15),
        "--audio-preprocess", _bool_token(audio.get("enabled")),
        "--ffmpeg-normalize", _bool_token(audio.get("ffmpeg_normalize")),
        "--target-sample-rate", str(audio.get("target_sample_rate") or 16000),
        "--service-base-url", _ps_quote(str(service.get("base_url") or "http://127.0.0.1:8000/v1")),
        "--service-model", _ps_quote(str(service.get("model") or "Systran/faster-whisper-large-v3")),
        "--service-timeout-seconds", str(service.get("timeout_seconds") or 600),
    ])
    commands = "\n".join([
        "# ASR 环境检查",
        ".\\scripts\\video-knowledge.ps1 asr-env-status --write",
        "",
        "# 本地模型缓存检查",
        ".\\scripts\\video-knowledge.ps1 asr-model-cache-status " + bundle_ps + " --include-optional",
        "",
        "# 持久化当前 ASR runtime profile",
        asr_cmd,
    ])
    return "".join([
        '<section class="panel"><h2>在线供应商 / API 本地配置</h2>',
        '<div class="muted">供应商、模型、Base URL 与任务路由写入本机 .local；API Key 使用 Windows DPAPI 加密。保存不会授权数据外发，也不会扩大 Broker 目的地白名单。</div>',
        f'<p><a href="{settings_ui_url}" target="_blank" rel="noreferrer"><button type="button">打开可保存的 API 设置界面</button></a></p>',
        f'<div class="muted">本地地址：<code>{settings_ui_url}</code></div>',
        '<div class="muted">若页面尚未启动，请在项目目录运行：<code>.\\scripts\\start-model-api-settings.ps1</code></div>',
        '</section>',
        '<section class="panel"><h2>ASR Runtime 设置</h2>',
        '<div class="muted">复用 Speaches 的 OpenAI-compatible ASR 服务契约、Whisper-WebUI 的参数面板思路，以及 Buzz 的说话人/音频质量提示。这里只保存运行参数，不保存 key，不启动 ASR。</div>',
        '<div class="grid">',
        f'<div><label>Provider</label><input value="{provider}" readonly></div>',
        f'<div><label>Model</label><input value="{model}" readonly></div>',
        f'<div><label>Device</label><input value="{device}" readonly></div>',
        f'<div><label>Compute type</label><input value="{compute_type}" readonly></div>',
        f'<div><label>VAD model</label><input value="{vad_model}" readonly></div>',
        f'<div><label>Punctuation model</label><input value="{punc_model}" readonly></div>',
        f'<div><label>Speaker model</label><input value="{spk_model}" readonly></div>',
        f'<div><label>Merge length seconds</label><input value="{merge_length}" readonly></div>',
        f'<div><label>Speaches/OpenAI-compatible base URL</label><input value="{service_base_url}" readonly></div>',
        f'<div><label>Speaches/OpenAI-compatible model</label><input value="{service_model}" readonly></div>',
        f'<div><label>Service timeout seconds</label><input value="{service_timeout}" readonly></div>',
        '</div>',
        '<h3>ASR adapter reuse map</h3><div class="grid">', adapter_rows or '<div class="panel muted">暂无 adapter 状态。</div>', '</div>',
        '<h3>保存/检查命令</h3>',
        f'<pre id="asrRuntimeCommands">{html.escape(commands)}</pre><button onclick="copyText(\'asrRuntimeCommands\')">复制 ASR 命令</button>',
        '</section>',
    ])


def _asr_adapter_row(row: dict[str, Any]) -> str:
    label = html.escape(str(row.get("label") or row.get("provider") or ""))
    provider = html.escape(str(row.get("provider") or ""))
    interface = html.escape(str(row.get("interface") or ""))
    model = html.escape(str(row.get("default_model") or ""))
    gpu = html.escape(str(row.get("gpu_policy") or ""))
    reuse = html.escape(str(row.get("reuse_source") or ""))
    return f'<div class="panel"><strong>{label}</strong><div class="muted">provider: <code>{provider}</code></div><div class="muted">interface: <code>{interface}</code></div><div class="muted">default model: <code>{model}</code></div><div class="muted">GPU: <code>{gpu}</code></div><div class="muted">reuse: {reuse}</div></div>'


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def _model_provider_options(providers: list[Any], selected: str) -> str:
    rows = []
    for row in providers:
        if not isinstance(row, dict):
            continue
        provider = str(row.get("provider") or "")
        label = str(row.get("label") or provider)
        mark = " selected" if provider == selected else ""
        rows.append(f'<option value="{html.escape(provider)}"{mark}>{html.escape(label)}</option>')
    return "\n".join(rows)


def _model_provider_row(row: dict[str, Any]) -> str:
    label = html.escape(str(row.get("label") or row.get("provider") or ""))
    provider = html.escape(str(row.get("provider") or ""))
    model = html.escape(str(row.get("default_model") or ""))
    base_url = html.escape(str(row.get("default_base_url") or ""))
    env_names = ", ".join(str(v) for v in row.get("api_key_env") or [])
    ok = bool(row.get("api_key_configured")) or bool(row.get("api_key_optional"))
    status = "已配置/可选" if ok else "未检测到"
    cls = "ok" if ok else "warn"
    return f'<div class="panel"><strong>{label}</strong><div class="muted">provider: <code>{provider}</code></div><div class="muted">default model: <code>{model}</code></div><div class="muted">base URL: <code>{base_url}</code></div><div class="{cls}">API key: {html.escape(status)}</div><div class="muted">env: <code>{html.escape(env_names)}</code></div></div>'


def _model_provider_env_command(row: dict[str, Any]) -> str:
    provider = str(row.get("provider") or "").strip()
    env_names = [str(v).strip() for v in row.get("api_key_env") or [] if str(v).strip()]
    if not env_names:
        return f"# {provider}: no API key required by default"
    lines = [f"# {provider}", f"[Environment]::SetEnvironmentVariable('{env_names[0]}', '<paste-key>', 'User')", f"[Environment]::SetEnvironmentVariable('LECTURE_VISION_PROVIDER', '{provider}', 'User')"]
    model = str(row.get("default_model") or "").strip()
    base_url = str(row.get("default_base_url") or "").strip()
    if model:
        lines.append(f"[Environment]::SetEnvironmentVariable('LECTURE_VISION_MODEL', '{model}', 'User')")
    if base_url:
        lines.append(f"[Environment]::SetEnvironmentVariable('LECTURE_VISION_BASE_URL', '{base_url}', 'User')")
    return "\n".join(lines)



def _model_batches_html(batches: dict[str, Any]) -> str:
    rows = batches.get("items") if isinstance(batches.get("items"), list) else []
    if not rows:
        return '<div class="panel muted">当前 Bundle 没有绑定的在线模型批次。</div>'
    cards: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        summary = row.get("summary") if isinstance(row.get("summary"), dict) else {}
        allowance = row.get("consent_allowance") if isinstance(row.get("consent_allowance"), dict) else {}
        nodes = ", ".join(str(value) for value in row.get("nodes") or [])
        destinations = ", ".join(str(value) for value in row.get("destinations") or [])
        remaining_calls = allowance.get("remaining_calls")
        remaining_cost = allowance.get("remaining_estimated_cost_usd")
        heartbeat_alive = int(summary.get("heartbeat_alive") or 0)
        heartbeat_stale = int(summary.get("heartbeat_stale") or 0)
        latency_p50 = summary.get("latency_p50_ms")
        latency_p95 = summary.get("latency_p95_ms")
        request_bytes = int(summary.get("gateway_request_bytes") or 0)
        response_bytes = int(summary.get("gateway_response_bytes") or 0)
        cards.append(
            '<div class="run-card ' + html.escape(str(row.get("status") or "")) + '">'
            + '<div><strong>' + html.escape(str(row.get("job_id") or "")) + '</strong> '
            + '<span class="badge">' + html.escape(str(row.get("status") or "unknown")) + '</span></div>'
            + '<div class="muted">节点：' + html.escape(nodes or "-") + '</div>'
            + '<div class="muted">目的地：' + html.escape(destinations or "-") + '</div>'
            + '<div>完成 ' + html.escape(str(summary.get("completed") or 0))
            + ' / ' + html.escape(str(summary.get("total") or 0))
            + '；失败 ' + html.escape(str(summary.get("failed") or 0))
            + '；依赖阻断 ' + html.escape(str(summary.get("dependency_blocked") or 0)) + '</div>'
            + '<div>心跳正常 ' + html.escape(str(heartbeat_alive))
            + '；心跳过期 ' + html.escape(str(heartbeat_stale)) + '</div>'
            + '<div>延迟 P50/P95：'
            + html.escape("未知" if latency_p50 is None else f"{float(latency_p50):.0f} ms")
            + ' / ' + html.escape("未知" if latency_p95 is None else f"{float(latency_p95):.0f} ms")
            + '；网络请求/响应：' + html.escape(str(request_bytes))
            + ' / ' + html.escape(str(response_bytes)) + ' bytes</div>'
            + '<div>Consent 剩余调用：' + html.escape("未知" if remaining_calls is None else str(remaining_calls))
            + '；剩余费用上限：' + html.escape("未知" if remaining_cost is None else f"${float(remaining_cost):.6f}") + '</div>'
            + '<div class="muted">最早到期：' + html.escape(str(allowance.get("earliest_expiry") or "未知")) + '</div>'
            + '</div>'
        )
    return '<div class="run-list">' + "".join(cards) + '</div>'

def _media_source(root: Path, manifest: dict[str, Any], timeline: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: list[str] = []
    for key in ("media_path", "local_media_path", "source_media_path", "video_path", "local_video_path", "source_path", "path"):
        value = str(manifest.get(key) or "").strip()
        if value:
            candidates.append(value)
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    for key in ("media_path", "local_media_path", "source_media_path", "video_path", "local_video_path", "source_path", "path"):
        value = str(source.get(key) or "").strip()
        if value:
            candidates.append(value)
    for item in timeline[:80]:
        for key in ("media_path", "video_key", "source_video_path", "local_video_path"):
            value = str(item.get(key) or "").strip()
            if value:
                candidates.append(value)
    for raw in candidates:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        try:
            resolved = path.expanduser().resolve()
        except Exception:
            resolved = path
        if resolved.exists() and resolved.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
            try:
                src = resolved.as_uri()
            except Exception:
                src = ""
            return {"path": str(resolved), "src": src, "exists": True}
    return {"path": candidates[0] if candidates else "", "src": "", "exists": False}
def export_subqueue_action_plan(bundle_dir: str | Path, *, write: bool = True, refresh: bool = False) -> dict[str, Any]:
    """Return the copyable subqueue action plan for one WebUI bundle.

    This is a thin agent-facing wrapper around export_task_console. It does
    not execute queued work; it only exposes the queue/action metadata that
    the static task console already renders for humans.
    """

    root = Path(bundle_dir).expanduser().resolve()
    console = export_task_console(root, write=write, refresh=refresh)
    plan = dict(console.get("subqueue_action_plan") if isinstance(console.get("subqueue_action_plan"), dict) else {})
    plan.setdefault("schema", "video_knowledge_pipeline.subqueue_action_plan.v1")
    plan["bundle_dir"] = str(root)
    plan["task_console_json_path"] = str(root / "task-console.json")
    plan["task_console_html_path"] = str(root / "task-console.html")
    plan["subqueue_action_plan_json_path"] = str(root / "subqueue-action-plan.json")
    plan["mcp_args_path"] = str(root / "mcp-subqueue-action-plan.args.json")
    plan["write"] = write
    plan["refresh"] = refresh
    plan["operator_boundary"] = {
        **(plan.get("operator_boundary") if isinstance(plan.get("operator_boundary"), dict) else {}),
        "no_process_started": True,
        "no_cloud_call": True,
        "review_only": True,
    }
    if write:
        write_json(root / "subqueue-action-plan.json", plan)
        write_json(root / "mcp-subqueue-action-plan.args.json", {"bundle_dir": str(root), "write": True, "refresh": False})
        manifest_path = root / "manifest.json"
        manifest = _read_object(manifest_path)
        manifest["subqueue_action_plan_json"] = "subqueue-action-plan.json"
        manifest["mcp_subqueue_action_plan_args"] = "mcp-subqueue-action-plan.args.json"
        write_json(manifest_path, manifest)
    return plan


def _read_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = read_json(path)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _manifest_path(root: Path, manifest: dict[str, Any], key: str) -> str:
    raw = str(manifest.get(key) or "").strip()
    if not raw:
        return ""
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return str(path)


def _non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _review_open_count(review_closure: dict[str, Any]) -> int:
    for key in ("open", "open_count", "review_targets_open"):
        try:
            return int(review_closure.get(key) or 0)
        except Exception:
            continue
    summary = review_closure.get("summary") if isinstance(review_closure.get("summary"), dict) else {}
    try:
        return int(summary.get("open") or summary.get("open_count") or 0)
    except Exception:
        return 0
