from __future__ import annotations

from typing import Any

from .acceptance_check import acceptance_check as acceptance_check_impl
from .acceptance_run import run_acceptance_bundle as run_acceptance_bundle_impl
from .acceptance_run import run_acceptance_run as run_acceptance_run_impl
from .asr_ab_compare import compare_asr_ab_sample as compare_asr_ab_sample_impl
from .asr_ab_plan import plan_asr_ab_sample as plan_asr_ab_sample_impl
from .asr_ab_run import run_asr_ab_sample as run_asr_ab_sample_impl
from .asr_consensus import build_asr_consensus as build_asr_consensus_impl
from .asr_diff_adjudication import apply_asr_diff_adjudication as apply_asr_diff_adjudication_impl, build_asr_diff_adjudication as build_asr_diff_adjudication_impl
from .asr_environment import asr_environment_status
from .asr_model_cache import asr_model_cache_status as asr_model_cache_status_impl, prepare_asr_model_cache as prepare_asr_model_cache_impl
from .asr_execution import asr_smoke as asr_smoke_impl
from .asr_execution import run_asr_plan
from .asr_runner import plan_asr_run, plan_whisperx_alignment
from .whisperx_alignment import run_whisperx_alignment as run_whisperx_alignment_impl
from .batch_repair import batch_repair_run as batch_repair_run_impl
from .batch_run import batch_video_knowledge_run as batch_video_knowledge_run_impl
from .bilinote_summary_tools import build_mind_map_prompt_pack as build_mind_map_prompt_pack_impl
from .bilinote_mind_map_prompt_pack import build_bundle_mind_map_prompt_pack as build_bundle_mind_map_prompt_pack_impl
from .bundle_next import bundle_advance as bundle_advance_impl
from .bundle_next import bundle_advance_log as bundle_advance_log_impl
from .bundle_next import bundle_advance_queue as bundle_advance_queue_impl
from .bundle_next import bundle_next_action as bundle_next_action_impl
from .bundle_status import bundle_status_report as bundle_status_report_impl
from .cloud_asr import plan_cloud_asr_run as plan_cloud_asr_run_impl
from .cloud_asr import run_cloud_asr_plan as run_cloud_asr_plan_impl
from .bundle_status import controlled_execution_check as controlled_execution_check_impl
from .config import config_status as config_status_impl, DEFAULT_LOCAL_FRAME_BUDGET, DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS, DEFAULT_LOCAL_FRAME_SAMPLING_MODE
from .content_asset_batch import batch_content_asset_status as batch_content_asset_status_impl
from .content_asset_batch import content_handoff_pack as content_handoff_pack_impl
from .content_asset_status import content_asset_status as content_asset_status_impl
from .codex_vision_session import import_codex_vision_results as import_codex_vision_results_impl
from .codex_vision_session import prepare_codex_vision_session as prepare_codex_vision_session_impl
from .summary_claim_review import prepare_summary_claim_review as prepare_summary_claim_review_impl
from .summary_claim_review import import_summary_claim_review as import_summary_claim_review_impl
from .summary_claim_review import summary_claim_review_status as summary_claim_review_status_impl
from .controlled_execution_smoke import controlled_execution_smoke as controlled_execution_smoke_impl
from .creative_contract_bridge import import_generation_contracts as import_generation_contracts_impl, import_previs_candidate as import_previs_candidate_impl
from .evidence_conflict_index import build_evidence_conflict_index as build_evidence_conflict_index_impl
from .entity_lexicon import build_entity_lexicon as build_entity_lexicon_impl
from .extractor_execution import extractor_run_log as extractor_run_log_impl
from .extractor_execution import run_extractor_plan as run_extractor_plan_impl
from .external_capability_pack import build_external_capability_pack as build_external_capability_pack_impl
from .general_tagger_adapter import general_tagger_status as general_tagger_status_impl, run_general_tagger as run_general_tagger_impl
from .temporal_tag_delta import (
    apply_temporal_tag_delta_restore as apply_temporal_tag_delta_restore_impl,
    run_temporal_tag_delta as run_temporal_tag_delta_impl,
)
from .highlight_detection_adapter import run_highlight_detection as run_highlight_detection_impl
from .high_res_tile_plan import run_high_res_tile_plan as run_high_res_tile_plan_impl
from .tile_result_import_builder import build_tile_result_import as build_tile_result_import_impl
from .tile_result_merge import run_tile_result_merge as run_tile_result_merge_impl
from .knowledge_coverage import audit_knowledge_coverage as audit_knowledge_coverage_impl
from .knowledge_note_export import export_knowledge_note as export_knowledge_note_impl
from .local_video_run import prepare_local_video_run
from .local_vlm_server_adapter import local_vlm_adapter_plan as local_vlm_adapter_plan_impl
from .local_vlm_server_adapter import local_vlm_serving_smoke as local_vlm_serving_smoke_impl
from .long_video_memory_pack import build_long_video_memory_pack as build_long_video_memory_pack_impl
from .multimodal_frame_analyzer import run_multimodal_frame_analysis as run_multimodal_frame_analysis_impl
from .multimodal_frame_analyzer import vision_analysis_apply_restore as vision_analysis_apply_restore_impl
from .multimodal_frame_analyzer import vision_analysis_restore_plan as vision_analysis_restore_plan_impl
from .multimodal_frame_analyzer import vision_analysis_run_log as vision_analysis_run_log_impl
from .multimodal_sample_review import multimodal_sample_review as multimodal_sample_review_impl
from .multimodal_sample_review import validate_multimodal_sample_notes as validate_multimodal_sample_notes_impl
from .model_task_automation import run_bilinote_mind_map_model as run_bilinote_mind_map_model_impl, run_term_arbitration_model as run_term_arbitration_model_impl
from .model_task_gateway import model_task_coverage_audit as model_task_coverage_audit_impl
from .ocr_backfill import run_ocr_backfill as run_ocr_backfill_impl
from .page_metadata import import_page_metadata as import_page_metadata_impl
from .online_model_gateway import online_model_api_call as online_model_api_call_impl
from .online_model_gateway import online_model_api_matrix as online_model_api_matrix_impl
from .offline_quality_router import offline_quality_route as offline_quality_route_impl
from .openclaw_bridge_status import openclaw_bridge_status as openclaw_bridge_status_impl
from .openclaw_bridge_doctor import openclaw_bridge_doctor as openclaw_bridge_doctor_impl
from .openclaw_docker_contract import openclaw_docker_contract_check as openclaw_docker_contract_check_impl
from .openclaw_integration import openclaw_video_ingest as openclaw_video_ingest_impl
from .openclaw_integration import openclaw_video_link as openclaw_video_link_impl
from .openclaw_integration import openclaw_video_plan as openclaw_video_plan_impl
from .openclaw_live_smoke import openclaw_live_smoke as openclaw_live_smoke_impl
from .path_defaults import provider_env_file, workspace_root
from .peepshow_adapter import attach_peepshow_output_to_bundle as attach_peepshow_output_to_bundle_impl
from .punctuation_model_stage import run_punctuation_model_stage as run_punctuation_model_stage_impl
from .quality_console import export_quality_console as export_quality_console_impl
from .quality_finalize import finalize_quality_outputs as finalize_quality_outputs_impl
from .quality_benchmark import build_quality_benchmark as build_quality_benchmark_impl, report_quality_benchmark as report_quality_benchmark_impl, run_quality_benchmark as run_quality_benchmark_impl
from .quality_benchmark_arbitration import build_quality_benchmark_arbitration as build_quality_benchmark_arbitration_impl, evaluate_quality_benchmark_arbitration as evaluate_quality_benchmark_arbitration_impl
from .quality_benchmark_punctuation import run_quality_benchmark_punctuation as run_quality_benchmark_punctuation_impl
from .quality_benchmark_punctuation_agent import build_quality_benchmark_punctuation_agent_pack as build_quality_benchmark_punctuation_agent_pack_impl, evaluate_quality_benchmark_punctuation_agent as evaluate_quality_benchmark_punctuation_agent_impl
from .quality_benchmark_residual_conflicts import build_quality_benchmark_residual_conflicts as build_quality_benchmark_residual_conflicts_impl
from .quality_benchmark_variants import execute_quality_benchmark_variants as execute_quality_benchmark_variants_impl
from .summary_blind_review import apply_summary_blind_review as apply_summary_blind_review_impl, build_summary_blind_review as build_summary_blind_review_impl
from .review_attestation import create_review_attestation as create_review_attestation_impl, validate_review_attestation as validate_review_attestation_impl
from .review_session import apply_review_notes_to_bundle as apply_review_notes_to_bundle_impl
from .run_artifact_registry import build_run_artifact_registry as build_run_artifact_registry_impl
from .review_session import prepare_review_session as prepare_review_session_impl
from .review_session import review_closure_status as review_closure_status_impl
from .review_session import validate_review_notes_for_bundle as validate_review_notes_for_bundle_impl
from .screen_text_recovery import run_screen_text_recovery as run_screen_text_recovery_impl
from .scene_detection_adapter import run_scene_detection as run_scene_detection_impl
from .shot_breakdown import build_shot_breakdown as build_shot_breakdown_impl
from .creator_report import creator_report_status as creator_report_status_impl, export_creator_report as export_creator_report_impl
from .semantic_chapter_plan import build_semantic_chapter_plan as build_semantic_chapter_plan_impl
from .smart_summary_chapters import build_smart_summary_chapter_pack as build_smart_summary_chapter_pack_impl
from .smart_summary_codex import generate_smart_summary_with_codex as generate_smart_summary_with_codex_impl
from .smart_summary_codex import prepare_smart_summary_llm_rewrite as prepare_smart_summary_llm_rewrite_impl
from .smart_summary_codex import run_smart_summary_llm_rewrite as run_smart_summary_llm_rewrite_impl
from .smart_summary_codex import smart_summary_quality_check as smart_summary_quality_check_impl
from .smart_summary_input_pack import build_smart_summary_input_pack as build_smart_summary_input_pack_impl
from .smart_summary_section_apply import apply_smart_summary_sections as apply_smart_summary_sections_impl
from .smart_summary_section_editor import build_smart_summary_section_editor as build_smart_summary_section_editor_impl
from .smart_summary_section_workflow import build_smart_summary_section_workflow as build_smart_summary_section_workflow_impl
from .smart_summary_section_llm import run_smart_summary_section_llm_rewrite as run_smart_summary_section_llm_rewrite_impl
from .smart_summary_global_reduce import run_smart_summary_global_reduce as run_smart_summary_global_reduce_impl
from .summary_consistency import run_summary_consistency_check as run_summary_consistency_check_impl
from .supplemental_frame_sampling import plan_supplemental_frame_sampling as plan_supplemental_frame_sampling_impl
from .tagger_import import import_tagger_annotations as import_tagger_annotations_impl
from .targeted_visual_evidence import run_targeted_visual_evidence as run_targeted_visual_evidence_impl
from .task_console import export_subqueue_action_plan as export_subqueue_action_plan_impl
from .task_console import export_task_console as export_task_console_impl
from .temporal_frame_groups import run_temporal_frame_groups as run_temporal_frame_groups_impl
from .term_arbitration_codex import build_term_arbitration_codex_pack as build_term_arbitration_codex_pack_impl, validate_term_arbitration_codex_result as validate_term_arbitration_codex_result_impl
from .term_correction_impact import term_correction_impact_report as term_correction_impact_report_impl
from .term_correction_closure import run_term_correction_closure as run_term_correction_closure_impl
from .term_correction_status import term_correction_status as term_correction_status_impl
from .term_resolution import resolve_terms as resolve_terms_impl
from .temporal_visual_analyzer import run_temporal_visual_analysis as run_temporal_visual_analysis_impl
from .text_llm_gateway import text_llm_provider_smoke as text_llm_provider_smoke_impl
from .timeline_alignment_audit import timeline_alignment_audit as timeline_alignment_audit_impl
from .transcript_correction_pack import build_transcript_correction_pack as build_transcript_correction_pack_impl
from .transcript_semantic_batch import transcript_semantic_acceptance as transcript_semantic_acceptance_impl, transcript_semantic_batch_acceptance as transcript_semantic_batch_acceptance_impl, transcript_semantic_batch_codex_review_draft as transcript_semantic_batch_codex_review_draft_impl, transcript_semantic_batch_import_review_notes as transcript_semantic_batch_import_review_notes_impl, transcript_semantic_batch_review_pack as transcript_semantic_batch_review_pack_impl, transcript_semantic_repair_queue as transcript_semantic_repair_queue_impl, transcript_semantic_repair_run as transcript_semantic_repair_run_impl
from .transcript_semantic_summary_impact import transcript_semantic_summary_impact_report as transcript_semantic_summary_impact_report_impl
from .transcript_semantic_benchmark import run_transcript_semantic_benchmark as transcript_semantic_benchmark_impl
from .transcript_semantic_correction import (
    build_transcript_semantic_candidate_discovery_codex_draft as build_transcript_semantic_candidate_discovery_codex_draft_impl,
    build_transcript_semantic_candidate_discovery_llm_draft as build_transcript_semantic_candidate_discovery_llm_draft_impl,
    build_transcript_semantic_candidate_discovery_pack as build_transcript_semantic_candidate_discovery_pack_impl,
    build_transcript_semantic_correction_codex_draft as build_transcript_semantic_correction_codex_draft_impl,
    build_transcript_semantic_correction_llm_draft as build_transcript_semantic_correction_llm_draft_impl,
    build_transcript_semantic_correction_pack as build_transcript_semantic_correction_pack_impl,
    import_transcript_semantic_candidate_suggestions as import_transcript_semantic_candidate_suggestions_impl,
    import_transcript_semantic_review_notes as import_transcript_semantic_review_notes_impl,
    transcript_semantic_correction_closure as transcript_semantic_correction_closure_impl,
    transcript_semantic_correction_impact_report as transcript_semantic_correction_impact_report_impl,
    transcript_semantic_correction_readable_impact_report as transcript_semantic_correction_readable_impact_report_impl,
    transcript_semantic_correction_status as transcript_semantic_correction_status_impl,
    validate_transcript_semantic_correction as validate_transcript_semantic_correction_impl,
)
from .transcript_semantic_transaction import apply_transcript_semantic_restore as apply_transcript_semantic_restore_impl
from .transcript_agent_readable import run_agent_readable_transcript_rewrite as run_agent_readable_transcript_rewrite_impl
from .transcript_evidence_correction_pipeline import run_transcript_evidence_correction_pipeline as run_transcript_evidence_correction_pipeline_impl
from .transcript_main_route_status import transcript_main_route_status as transcript_main_route_status_impl
from .transcript_source_arbitration import arbitrate_transcript_sources as arbitrate_transcript_sources_impl
from .transcript_editor import apply_transcript_edits as apply_transcript_edits_impl
from .transcript_editor import prepare_transcript_edit_session as prepare_transcript_edit_session_impl
from .transcript_resegment import resegment_transcript
from .transcript_postprocess import postprocess_asr_transcript as postprocess_asr_transcript_impl
from .transcript_quality_gate import run_transcript_quality_gate as run_transcript_quality_gate_impl
from .transcript_readable_llm import run_readable_transcript_llm_polish as run_readable_transcript_llm_polish_impl
from .video_frame_router import run_video_frame_router as run_video_frame_router_impl
from .video_moment_index import build_video_moment_index as build_video_moment_index_impl
from .video_rag_pack import build_video_rag_pack as build_video_rag_pack_impl
from .video_rag_http import video_rag_service_plan as video_rag_service_plan_impl
from .volcengine_model_routing import volcengine_model_routing as volcengine_model_routing_impl
from .volcengine_model_task_matrix import run_volcengine_model_task_matrix as run_volcengine_model_task_matrix_impl
from .video_rag_search import search_video_rag as search_video_rag_impl
from .script_clip_candidate_pack import build_script_clip_candidate_pack as build_script_clip_candidate_pack_impl
from .script_clip_alignment import check_script_clip_alignment as check_script_clip_alignment_impl
from .content_clip_candidate_pack import build_content_clip_candidate_pack as build_content_clip_candidate_pack_impl
from .content_clip_alignment import check_content_clip_alignment as check_content_clip_alignment_impl
from .video_evidence_query import apply_video_evidence_confirmation as apply_video_evidence_confirmation_impl, build_video_evidence_query_plan as build_video_evidence_query_plan_impl
from .video_edit_review_pack import build_video_edit_review_pack as build_video_edit_review_pack_impl
from .video_structure import build_video_structure as build_video_structure_impl
from .video_workbench import export_video_workbench as export_video_workbench_impl
from .vdo_handoff import ingest_vdo_handoff as ingest_vdo_handoff_impl
from .vdo_handoff import vdo_handoff_plan as vdo_handoff_plan_impl
from .vision_acceptance import vision_acceptance_plan as vision_acceptance_plan_impl
from .vision_api import test_vision_provider as test_vision_provider_impl
from .vision_preflight import vision_execution_preflight as vision_execution_preflight_impl
from .vision_export_consent import create_vision_export_consent as create_vision_export_consent_impl, revoke_vision_export_consent as revoke_vision_export_consent_impl, vision_export_consent_status as vision_export_consent_status_impl
from .vision_provider_smoke import vision_provider_matrix as vision_provider_matrix_impl
from .vision_provider_smoke import vision_provider_smoke as vision_provider_smoke_impl
from .vision_review_queue import vision_review_queue as vision_review_queue_impl
from .vision_review_triage import vision_review_triage as vision_review_triage_impl
from .visual_structure import run_visual_structure_plan as run_visual_structure_plan_impl
from .visual_ab_benchmark import build_visual_ab_benchmark_plan as build_visual_ab_benchmark_plan_impl


from .ocr_route import run_ocr_route as run_ocr_route_impl

from .adaptive_asr_route import build_adaptive_asr_route as build_adaptive_asr_route_impl

def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise SystemExit("Install MCP support with: pip install -e .[mcp]") from exc

    server = FastMCP("video-knowledge-pipeline")
    @server.tool()
    def export_video_workbench_tool(bundle_dir: str, write: bool = True) -> dict[str, Any]:
        return export_video_workbench_impl(bundle_dir, write=write)

    @server.tool()
    def export_video_workbench(bundle_dir: str, write: bool = True) -> dict[str, Any]:
        return export_video_workbench_impl(bundle_dir, write=write)

    @server.tool()
    def create_review_attestation_tool(
        bundle_dir: str,
        target: str,
        artifacts: list[dict[str, str]],
        approved_by: str,
        comment: str = "",
        write: bool = True,
    ) -> dict[str, Any]:
        return create_review_attestation_impl(
            bundle_dir,
            target=target,
            artifact_paths=artifacts,
            approved_by=approved_by,
            comment=comment,
            write=write,
        )

    @server.tool()
    def review_attestation_status_tool(
        bundle_dir: str,
        target: str = "",
        attestation_path: str = "",
    ) -> dict[str, Any]:
        return validate_review_attestation_impl(
            bundle_dir,
            target=target,
            attestation_path=attestation_path or None,
        )

    @server.tool()
    def import_generation_contracts_tool(
        bundle_dir: str,
        task_path: str,
        receipt_path: str,
        validation_path: str,
        preflight_path: str = "",
        allowed_roots: list[str] | None = None,
        write: bool = True,
    ) -> dict[str, Any]:
        return import_generation_contracts_impl(
            bundle_dir,
            task_path=task_path,
            receipt_path=receipt_path,
            validation_path=validation_path,
            preflight_path=preflight_path or None,
            allowed_roots=allowed_roots,
            write=write,
        )

    @server.tool()
    def import_previs_candidate_tool(
        bundle_dir: str,
        scene_path: str,
        capture_manifest_path: str,
        validation_path: str,
        allowed_roots: list[str] | None = None,
        write: bool = True,
    ) -> dict[str, Any]:
        return import_previs_candidate_impl(
            bundle_dir,
            scene_path=scene_path,
            manifest_path=capture_manifest_path,
            validation_path=validation_path,
            allowed_roots=allowed_roots,
            write=write,
        )

    @server.tool()
    def video_edit_review_pack_tool(
        bundle_dir: str,
        decisions_json: str = "",
        tokens_json: str = "",
        silence_json: str = "",
        delete_segments_json: str = "",
        cut_segments_json: str = "",
        ai_baseline_json: str = "",
        media_path: str = "",
        reclaim_silence: bool = False,
        human_confirmed_diff: bool = False,
        review_attestation_path: str = "",
        write: bool = True,
    ) -> dict[str, Any]:
        return build_video_edit_review_pack_impl(
            bundle_dir,
            decisions_json=decisions_json or None,
            tokens_json=tokens_json or None,
            silence_json=silence_json or None,
            delete_segments_json=delete_segments_json or None,
            cut_segments_json=cut_segments_json or None,
            ai_baseline_json=ai_baseline_json or None,
            media_path=media_path or None,
            reclaim_silence=reclaim_silence,
            human_confirmed_diff=human_confirmed_diff,
            review_attestation_path=review_attestation_path or None,
            write=write,
        )

    @server.tool()
    def video_edit_review_pack(bundle_dir: str, write: bool = True) -> dict[str, Any]:
        return build_video_edit_review_pack_impl(bundle_dir, write=write)

    @server.tool()
    def timeline_alignment_audit_tool(bundle_dir: str, tolerance_seconds: float = 2.0, write: bool = True) -> dict[str, Any]:
        return timeline_alignment_audit_impl(bundle_dir, tolerance_seconds=tolerance_seconds, write=write)

    @server.tool()
    def timeline_alignment_audit(bundle_dir: str, tolerance_seconds: float = 2.0, write: bool = True) -> dict[str, Any]:
        return timeline_alignment_audit_impl(bundle_dir, tolerance_seconds=tolerance_seconds, write=write)

    @server.tool()
    def asr_env_status(venv_dir: str = "", output_dir: str = "", write: bool = False, python_version: str = "3.11") -> dict[str, Any]:
        return asr_environment_status(venv_dir, output_dir=output_dir, write=write, python_version=python_version)

    @server.tool()
    def config_status(config_path: str = "") -> dict[str, Any]:
        return config_status_impl(config_path)

    @server.tool()
    def import_page_metadata_tool(bundle_dir: str, metadata_json: str, write: bool = True) -> dict[str, Any]:
        return import_page_metadata_impl(bundle_dir, metadata_json, write=write)

    @server.tool()
    def import_page_metadata(bundle_dir: str, metadata_json: str, write: bool = True) -> dict[str, Any]:
        return import_page_metadata_impl(bundle_dir, metadata_json, write=write)

    @server.tool()
    def asr_model_cache_status(workspace_dir: str, models: list[str] | None = None, include_optional: bool = False, write: bool = True) -> dict[str, Any]:
        return asr_model_cache_status_impl(workspace_dir, models=models, include_optional=include_optional, write=write)

    @server.tool()
    def prepare_asr_model_cache(
        workspace_dir: str,
        models: list[str] | None = None,
        include_optional: bool = False,
        execute: bool = False,
        allow_download: bool = False,
        device: str = "auto",
        timeout_seconds: int = 1800,
        write: bool = True,
    ) -> dict[str, Any]:
        return prepare_asr_model_cache_impl(
            workspace_dir,
            models=models,
            include_optional=include_optional,
            execute=execute,
            allow_download=allow_download,
            device=device,
            timeout_seconds=timeout_seconds,
            write=write,
        )

    @server.tool()
    def prepare_local_video_run_tool(
        media_path: str,
        output_dir: str,
        title: str = "",
        copy_media: bool = False,
        plan_asr: bool = True,
        execute_asr: bool = False,
        asr_preset: str = "sensevoice",
        asr_model: str = "iic/SenseVoiceSmall",
        transcript_path: str | None = None,
        build_initial_bundle: bool = False,
        sample_interval: float = DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS,
        max_frames: int = DEFAULT_LOCAL_FRAME_BUDGET,
        sample_mode: str = DEFAULT_LOCAL_FRAME_SAMPLING_MODE,
        detect_scenes: bool = True,
        extract_frames: bool = True,
        timeout_seconds: int = 1800,
    ) -> dict[str, Any]:
        return prepare_local_video_run(
            media_path,
            output_dir,
            title=title,
            copy_media=copy_media,
            plan_asr=plan_asr,
            execute_asr=execute_asr,
            asr_preset=asr_preset,
            asr_model=asr_model,
            transcript_path=transcript_path,
            build_initial_bundle=build_initial_bundle,
            sample_interval=sample_interval,
            max_frames=max_frames,
            sample_mode=sample_mode,
            detect_scenes=detect_scenes,
            extract_frames=extract_frames,
            timeout_seconds=timeout_seconds,
        )

    @server.tool()
    def openclaw_video_plan(
        url_or_text: str,
        output_dir: str = "",
        vdo_root: str = "",
        vdo_output_dir: str = "",
        backend: str = "",
        write_manifests: bool = True,
        include_manifests: bool = False,
        timeout_seconds: int = 120,
    ) -> dict[str, Any]:
        return openclaw_video_plan_impl(
            url_or_text,
            output_dir=output_dir,
            vdo_root=vdo_root,
            vdo_output_dir=vdo_output_dir,
            backend=backend,
            write_manifests=write_manifests,
            include_manifests=include_manifests,
            timeout_seconds=timeout_seconds,
        )

    @server.tool()
    def openclaw_video_ingest(
        media_path: str,
        workspace: str = "",
        title: str = "",
        copy_media: bool = False,
        plan_asr: bool = True,
        execute_asr: bool = False,
        asr_preset: str = "sensevoice",
        asr_model: str = "iic/SenseVoiceSmall",
        transcript_path: str | None = None,
        build_initial_bundle: bool = True,
        sample_interval: float = DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS,
        max_frames: int = DEFAULT_LOCAL_FRAME_BUDGET,
        sample_mode: str = DEFAULT_LOCAL_FRAME_SAMPLING_MODE,
        detect_scenes: bool = True,
        extract_frames: bool = True,
        timeout_seconds: int = 1800,
    ) -> dict[str, Any]:
        return openclaw_video_ingest_impl(
            media_path,
            workspace=workspace,
            title=title,
            copy_media=copy_media,
            plan_asr=plan_asr,
            execute_asr=execute_asr,
            asr_preset=asr_preset,
            asr_model=asr_model,
            transcript_path=transcript_path,
            build_initial_bundle=build_initial_bundle,
            sample_interval=sample_interval,
            max_frames=max_frames,
            sample_mode=sample_mode,
            detect_scenes=detect_scenes,
            extract_frames=extract_frames,
            timeout_seconds=timeout_seconds,
        )

    @server.tool()
    def openclaw_video_link(
        url_or_text: str,
        output_dir: str = "",
        vdo_root: str = "",
        vdo_output_dir: str = "",
        backend: str = "",
        allow_download: bool = False,
        actor_id: str = "",
        confirm_download: bool = False,
        confirm_sensitive: bool = False,
        ingest_after_download: bool = False,
        downloaded_media_path: str = "",
        workspace: str = "",
        title: str = "",
        max_frames: int = DEFAULT_LOCAL_FRAME_BUDGET,
        sample_interval: float = DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS,
        sample_mode: str = DEFAULT_LOCAL_FRAME_SAMPLING_MODE,
        timeout_seconds: int = 1800,
    ) -> dict[str, Any]:
        return openclaw_video_link_impl(
            url_or_text,
            output_dir=output_dir,
            vdo_root=vdo_root,
            vdo_output_dir=vdo_output_dir,
            backend=backend,
            allow_download=allow_download,
            actor_id=actor_id,
            confirm_download=confirm_download,
            confirm_sensitive=confirm_sensitive,
            ingest_after_download=ingest_after_download,
            downloaded_media_path=downloaded_media_path,
            workspace=workspace,
            title=title,
            max_frames=max_frames,
            sample_interval=sample_interval,
            sample_mode=sample_mode,
            timeout_seconds=timeout_seconds,
        )

    @server.tool()
    def openclaw_bridge_status(timeout_seconds: float = 2.0, check_health: bool = True, check_task: bool = True) -> dict[str, Any]:
        return openclaw_bridge_status_impl(timeout_seconds=timeout_seconds, check_health=check_health, check_task=check_task)

    @server.tool()
    def openclaw_bridge_doctor(timeout_seconds: float = 2.0, project_root: str = "") -> dict[str, Any]:
        return openclaw_bridge_doctor_impl(timeout_seconds=timeout_seconds, project_root=project_root)

    @server.tool()
    def openclaw_live_smoke(
        bundle_dir: str = "",
        compose_path: str = "",
        host_root: str = str(workspace_root()),
        container_root: str = "/mnt/used-by-codex",
        timeout_seconds: float = 2.0,
        output_dir: str = "",
        semantic_batch_input: str = "",
        semantic_target_bundle_count: int = 3,
        semantic_limit: int = 0,
        write_report: bool = False,
    ) -> dict[str, Any]:
        return openclaw_live_smoke_impl(
            bundle_dir=bundle_dir,
            compose_path=compose_path,
            host_root=host_root,
            container_root=container_root,
            timeout_seconds=timeout_seconds,
            output_dir=output_dir,
            semantic_batch_input=semantic_batch_input,
            semantic_target_bundle_count=semantic_target_bundle_count,
            semantic_limit=semantic_limit,
            write_report=write_report,
        )

    @server.tool()
    def openclaw_docker_contract_check(compose_path: str = "", host_root: str = str(workspace_root()), container_root: str = "/mnt/used-by-codex") -> dict[str, Any]:
        return openclaw_docker_contract_check_impl(compose_path, host_root=host_root, container_root=container_root)

    @server.tool()
    def openclaw_video_from_vdo_handoff(
        manifest_path: str = "",
        summary_path: str = "",
        review_checklist_path: str = "",
        media_path: str = "",
        host_root: str = str(workspace_root()),
        container_root: str = "/mnt/used-by-codex",
        workspace: str = "",
        title: str = "",
    ) -> dict[str, Any]:
        return vdo_handoff_plan_impl(
            manifest_path=manifest_path,
            summary_path=summary_path,
            review_checklist_path=review_checklist_path,
            media_path=media_path,
            host_root=host_root,
            container_root=container_root,
            workspace=workspace,
            title=title,
        )

    @server.tool()
    def openclaw_video_ingest_vdo_handoff(
        handoff_path: str = "",
        manifest_path: str = "",
        summary_path: str = "",
        review_checklist_path: str = "",
        media_path: str = "",
        host_root: str = str(workspace_root()),
        container_root: str = "/mnt/used-by-codex",
        workspace: str = "",
        title: str = "",
        execute: bool = False,
        max_frames: int = DEFAULT_LOCAL_FRAME_BUDGET,
        sample_interval: float = DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS,
        sample_mode: str = DEFAULT_LOCAL_FRAME_SAMPLING_MODE,
    ) -> dict[str, Any]:
        return ingest_vdo_handoff_impl(
            handoff_path=handoff_path,
            manifest_path=manifest_path,
            summary_path=summary_path,
            review_checklist_path=review_checklist_path,
            media_path=media_path,
            host_root=host_root,
            container_root=container_root,
            workspace=workspace,
            title=title,
            execute=execute,
            max_frames=max_frames,
            sample_interval=sample_interval,
            sample_mode=sample_mode,
        )

    @server.tool()
    def acceptance_run_tool(
        media_path: str,
        output_dir: str,
        title: str = "",
        copy_media: bool = False,
        execute_asr: bool = False,
        asr_preset: str = "sensevoice",
        asr_model: str = "iic/SenseVoiceSmall",
        transcript_path: str | None = None,
        build_initial_bundle: bool = True,
        sample_interval: float = DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS,
        max_frames: int = DEFAULT_LOCAL_FRAME_BUDGET,
        sample_mode: str = DEFAULT_LOCAL_FRAME_SAMPLING_MODE,
        detect_scenes: bool = True,
        extract_frames: bool = True,
        execute_temporal_groups: bool = False,
        execute_vision: bool = False,
        execute_ebook_pipeline: bool = False,
        semantic_limit: int | None = None,
        temporal_limit: int | None = None,
        frame_count: int | None = None,
        provider_config: dict[str, Any] | None = None,
        confirm_vision_calls: int | None = None,
        confirm_vision_indexes: str = "",
        timeout_seconds: int = 1800,
    ) -> dict[str, Any]:
        return run_acceptance_run_impl(
            media_path,
            output_dir,
            title=title,
            copy_media=copy_media,
            execute_asr=execute_asr,
            asr_preset=asr_preset,
            asr_model=asr_model,
            transcript_path=transcript_path,
            build_initial_bundle=build_initial_bundle,
            sample_interval=sample_interval,
            max_frames=max_frames,
            sample_mode=sample_mode,
            detect_scenes=detect_scenes,
            extract_frames=extract_frames,
            execute_temporal_groups=execute_temporal_groups,
            execute_vision=execute_vision,
            execute_ebook_pipeline=execute_ebook_pipeline,
            semantic_limit=semantic_limit,
            temporal_limit=temporal_limit,
            frame_count=frame_count,
            provider_config=provider_config,
            confirm_vision_calls=confirm_vision_calls,
            confirm_vision_indexes=confirm_vision_indexes,
            timeout_seconds=timeout_seconds,
        )

    @server.tool()
    def acceptance_run(
        media_path: str,
        output_dir: str,
        title: str = "",
        copy_media: bool = False,
        execute_asr: bool = False,
        asr_preset: str = "sensevoice",
        asr_model: str = "iic/SenseVoiceSmall",
        transcript_path: str | None = None,
        build_initial_bundle: bool = True,
        sample_interval: float = DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS,
        max_frames: int = DEFAULT_LOCAL_FRAME_BUDGET,
        sample_mode: str = DEFAULT_LOCAL_FRAME_SAMPLING_MODE,
        detect_scenes: bool = True,
        extract_frames: bool = True,
        execute_temporal_groups: bool = False,
        execute_vision: bool = False,
        execute_ebook_pipeline: bool = False,
        semantic_limit: int | None = None,
        temporal_limit: int | None = None,
        frame_count: int | None = None,
        provider_config: dict[str, Any] | None = None,
        confirm_vision_calls: int | None = None,
        confirm_vision_indexes: str = "",
        timeout_seconds: int = 1800,
    ) -> dict[str, Any]:
        return run_acceptance_run_impl(
            media_path,
            output_dir,
            title=title,
            copy_media=copy_media,
            execute_asr=execute_asr,
            asr_preset=asr_preset,
            asr_model=asr_model,
            transcript_path=transcript_path,
            build_initial_bundle=build_initial_bundle,
            sample_interval=sample_interval,
            max_frames=max_frames,
            sample_mode=sample_mode,
            detect_scenes=detect_scenes,
            extract_frames=extract_frames,
            execute_temporal_groups=execute_temporal_groups,
            execute_vision=execute_vision,
            execute_ebook_pipeline=execute_ebook_pipeline,
            semantic_limit=semantic_limit,
            temporal_limit=temporal_limit,
            frame_count=frame_count,
            provider_config=provider_config,
            confirm_vision_calls=confirm_vision_calls,
            confirm_vision_indexes=confirm_vision_indexes,
            timeout_seconds=timeout_seconds,
        )

    @server.tool()
    def acceptance_bundle_run_tool(
        bundle_dir: str,
        output_dir: str = "",
        title: str = "",
        execute_temporal_groups: bool = False,
        execute_vision: bool = False,
        execute_ebook_pipeline: bool = False,
        semantic_limit: int | None = None,
        temporal_limit: int | None = None,
        frame_count: int | None = None,
        provider_config: dict[str, Any] | None = None,
        confirm_vision_calls: int | None = None,
        confirm_vision_indexes: str = "",
    ) -> dict[str, Any]:
        return run_acceptance_bundle_impl(
            bundle_dir,
            output_dir=output_dir or None,
            title=title,
            execute_temporal_groups=execute_temporal_groups,
            execute_vision=execute_vision,
            execute_ebook_pipeline=execute_ebook_pipeline,
            semantic_limit=semantic_limit,
            temporal_limit=temporal_limit,
            frame_count=frame_count,
            provider_config=provider_config,
            confirm_vision_calls=confirm_vision_calls,
            confirm_vision_indexes=confirm_vision_indexes,
        )

    @server.tool()
    def batch_video_knowledge_run_tool(
        batch_manifest: str,
        resume: bool = False,
        force_reexport: bool = False,
        execute_asr: bool = False,
        execute_temporal_groups: bool = False,
        execute_vision: bool = False,
        execute_ebook_pipeline: bool = False,
        semantic_limit: int | None = None,
        temporal_limit: int | None = None,
        frame_count: int | None = None,
        timeout_seconds: int = 1800,
        write: bool = True,
    ) -> dict[str, Any]:
        return batch_video_knowledge_run_impl(
            batch_manifest,
            resume=resume,
            force_reexport=force_reexport,
            execute_asr=execute_asr,
            execute_temporal_groups=execute_temporal_groups,
            execute_vision=execute_vision,
            execute_ebook_pipeline=execute_ebook_pipeline,
            semantic_limit=semantic_limit,
            temporal_limit=temporal_limit,
            frame_count=frame_count,
            timeout_seconds=timeout_seconds,
            write=write,
        )

    @server.tool()
    def batch_repair_run_tool(
        batch_manifest_or_summary: str,
        execute: bool = False,
        limit: int = 0,
        max_rounds: int = 1,
        allow_asr: bool = False,
        allow_vision: bool = False,
        allow_ocr: bool = False,
        write: bool = True,
    ) -> dict[str, Any]:
        return batch_repair_run_impl(
            batch_manifest_or_summary,
            execute=execute,
            limit=limit,
            max_rounds=max_rounds,
            allow_asr=allow_asr,
            allow_vision=allow_vision,
            allow_ocr=allow_ocr,
            write=write,
        )

    @server.tool()
    def batch_repair_run(
        batch_manifest_or_summary: str,
        execute: bool = False,
        limit: int = 0,
        max_rounds: int = 1,
        allow_asr: bool = False,
        allow_vision: bool = False,
        allow_ocr: bool = False,
        write: bool = True,
    ) -> dict[str, Any]:
        return batch_repair_run_impl(
            batch_manifest_or_summary,
            execute=execute,
            limit=limit,
            max_rounds=max_rounds,
            allow_asr=allow_asr,
            allow_vision=allow_vision,
            allow_ocr=allow_ocr,
            write=write,
        )

    @server.tool()
    def acceptance_bundle_run(
        bundle_dir: str,
        output_dir: str = "",
        title: str = "",
        execute_temporal_groups: bool = False,
        execute_vision: bool = False,
        execute_ebook_pipeline: bool = False,
        semantic_limit: int | None = None,
        temporal_limit: int | None = None,
        frame_count: int | None = None,
        provider_config: dict[str, Any] | None = None,
        confirm_vision_calls: int | None = None,
        confirm_vision_indexes: str = "",
    ) -> dict[str, Any]:
        return run_acceptance_bundle_impl(
            bundle_dir,
            output_dir=output_dir or None,
            title=title,
            execute_temporal_groups=execute_temporal_groups,
            execute_vision=execute_vision,
            execute_ebook_pipeline=execute_ebook_pipeline,
            semantic_limit=semantic_limit,
            temporal_limit=temporal_limit,
            frame_count=frame_count,
            provider_config=provider_config,
            confirm_vision_calls=confirm_vision_calls,
            confirm_vision_indexes=confirm_vision_indexes,
        )

    @server.tool()
    def plan_asr(
        output_dir: str,
        media_path: str,
        preset: str = "sensevoice",
        model: str = "",
        language: str = "zh",
        punc_model: str = "__default__",
        spk_model: str = "",
        hotword: str = "",
        use_itn: bool = True,
        merge_vad: bool = True,
        merge_length_s: int = 15,
        vad_max_single_segment_time_ms: int = 30000,
        transcript_path: str = "",
    ) -> dict[str, Any]:
        return plan_asr_run(
            output_dir,
            media_path,
            preset=preset,
            model=model or None,
            language=language,
            punc_model=None if punc_model == "__default__" else punc_model,
            spk_model=spk_model or None,
            hotword=hotword or None,
            use_itn=use_itn,
            merge_vad=merge_vad,
            merge_length_s=merge_length_s,
            vad_max_single_segment_time_ms=vad_max_single_segment_time_ms,
            transcript_path=transcript_path or None,
        )

    @server.tool()
    def asr_ab_sample_plan(
        workspace_dir: str,
        media_path: str,
        sample_start_seconds: float = 0.0,
        duration_seconds: float = 300.0,
        language: str = "zh",
        cloud_provider_config: dict[str, Any] | None = None,
        write: bool = True,
    ) -> dict[str, Any]:
        return plan_asr_ab_sample_impl(
            workspace_dir,
            media_path,
            sample_start_seconds=sample_start_seconds,
            duration_seconds=duration_seconds,
            language=language,
            cloud_provider_config=cloud_provider_config,
            write=write,
        )

    @server.tool()
    def asr_ab_sample_run(
        workspace_dir: str,
        media_path: str = "",
        plan_json: str = "",
        sample_start_seconds: float = 0.0,
        duration_seconds: float = 300.0,
        language: str = "zh",
        execute_sample: bool = False,
        execute_local: bool = False,
        execute_cloud: bool = False,
        cloud_provider_config: dict[str, Any] | None = None,
        variants: list[str] | None = None,
        timeout_seconds: int = 1800,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_asr_ab_sample_impl(
            workspace_dir,
            media_path,
            plan_json=plan_json,
            sample_start_seconds=sample_start_seconds,
            duration_seconds=duration_seconds,
            language=language,
            execute_sample=execute_sample,
            execute_local=execute_local,
            execute_cloud=execute_cloud,
            cloud_provider_config=cloud_provider_config,
            variants=variants,
            timeout_seconds=timeout_seconds,
            write=write,
        )

    @server.tool()
    def asr_ab_compare(run_json: str, write: bool = True) -> dict[str, Any]:
        return compare_asr_ab_sample_impl(run_json, write=write)

    @server.tool()
    def plan_whisperx_alignment_tool(output_dir: str, media_path: str, model: str = "", language: str = "zh") -> dict[str, Any]:
        return plan_whisperx_alignment(output_dir, media_path, model=model or None, language=language)

    @server.tool()
    def run_whisperx_alignment(
        output_dir: str,
        media_path: str,
        model: str = "large-v3",
        language: str = "zh",
        execute: bool = False,
        timeout_seconds: int = 1800,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_whisperx_alignment_impl(output_dir, media_path, model=model, language=language, execute=execute, timeout_seconds=timeout_seconds, write=write)

    @server.tool()
    def run_asr_plan_tool(plan_json: str, execute: bool = False, timeout_seconds: int = 1800) -> dict[str, Any]:
        return run_asr_plan(plan_json, execute=execute, timeout_seconds=timeout_seconds)

    @server.tool()
    def plan_cloud_asr_tool(workspace_dir: str, media_path: str, provider_config: dict[str, Any] | None = None, model: str = "gpt-4o-transcribe", language: str = "zh", prompt: str = "") -> dict[str, Any]:
        return plan_cloud_asr_run_impl(workspace_dir, media_path, provider_config=provider_config, model=model, language=language, prompt=prompt)

    @server.tool()
    def plan_cloud_asr(workspace_dir: str, media_path: str, provider_config: dict[str, Any] | None = None, model: str = "gpt-4o-transcribe", language: str = "zh", prompt: str = "") -> dict[str, Any]:
        return plan_cloud_asr_run_impl(workspace_dir, media_path, provider_config=provider_config, model=model, language=language, prompt=prompt)

    @server.tool()
    def run_cloud_asr_plan_tool(plan_json: str, provider_config: dict[str, Any] | None = None, execute: bool = False, normalize: bool = True) -> dict[str, Any]:
        return run_cloud_asr_plan_impl(plan_json, provider_config=provider_config, execute=execute, normalize=normalize)

    @server.tool()
    def run_cloud_asr_plan(plan_json: str, provider_config: dict[str, Any] | None = None, execute: bool = False, normalize: bool = True) -> dict[str, Any]:
        return run_cloud_asr_plan_impl(plan_json, provider_config=provider_config, execute=execute, normalize=normalize)

    @server.tool()
    def asr_smoke_tool(
        media_path: str,
        output_dir: str = "",
        preset: str = "sensevoice",
        model: str = "",
        language: str = "zh",
        duration_seconds: int = 30,
        execute: bool = True,
        timeout_seconds: int = 600,
    ) -> dict[str, Any]:
        return asr_smoke_impl(
            media_path,
            output_dir=output_dir or None,
            preset=preset,
            model=model,
            language=language,
            duration_seconds=duration_seconds,
            execute=execute,
            timeout_seconds=timeout_seconds,
        )

    @server.tool()
    def resegment_transcript_tool(
        workspace_dir: str,
        input_path: str,
        media_path: str = "",
        duration_seconds: float = 0.0,
        target_seconds: float = 8.0,
        max_chars: int = 180,
        title: str = "",
    ) -> dict[str, Any]:
        return resegment_transcript(
            workspace_dir,
            input_path,
            media_path=media_path,
            duration_seconds=duration_seconds,
            target_seconds=target_seconds,
            max_chars=max_chars,
            title=title,
        )

    @server.tool()
    def postprocess_asr_transcript_tool(
        bundle_dir: str,
        input_path: str = "",
        target_seconds: float = 18.0,
        max_chars: int = 180,
        punctuation_mode: str = "readable",
        set_corrected: bool = True,
        write: bool = True,
    ) -> dict[str, Any]:
        return postprocess_asr_transcript_impl(
            bundle_dir,
            input_path=input_path or None,
            target_seconds=target_seconds,
            max_chars=max_chars,
            punctuation_mode=punctuation_mode,
            set_corrected=set_corrected,
            write=write,
        )

    @server.tool()
    def postprocess_asr_transcript(
        bundle_dir: str,
        input_path: str = "",
        target_seconds: float = 18.0,
        max_chars: int = 180,
        punctuation_mode: str = "readable",
        set_corrected: bool = True,
        write: bool = True,
    ) -> dict[str, Any]:
        return postprocess_asr_transcript_impl(
            bundle_dir,
            input_path=input_path or None,
            target_seconds=target_seconds,
            max_chars=max_chars,
            punctuation_mode=punctuation_mode,
            set_corrected=set_corrected,
            write=write,
        )

    @server.tool()
    def readable_transcript_llm_polish(
        bundle_dir: str,
        provider_config: dict[str, Any] | None = None,
        input_json: str = "",
        execute: bool = False,
        agent_substitute: bool = False,
        agent_name: str = "local_agent",
        codex_substitute: bool = False,
        promote: bool = False,
        max_segments_per_batch: int = 40,
        max_prompt_chars: int = 9000,
        max_tokens: int = 4000,
        temperature: float = 0,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_readable_transcript_llm_polish_impl(
            bundle_dir,
            provider_config=provider_config or None,
            input_json=input_json or None,
            execute=execute,
            agent_substitute=agent_substitute or codex_substitute,
            agent_name=agent_name,
            codex_substitute=codex_substitute,
            promote=promote,
            max_segments_per_batch=max_segments_per_batch,
            max_prompt_chars=max_prompt_chars,
            max_tokens=max_tokens,
            temperature=temperature,
            write=write,
        )

    @server.tool()
    def agent_readable_transcript_rewrite(
        bundle_dir: str,
        input_json: str = "",
        agent_name: str = "local_agent",
        source_path: str = "",
        promote: bool = False,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_agent_readable_transcript_rewrite_impl(
            bundle_dir,
            input_json=input_json or None,
            agent_name=agent_name,
            source_path=source_path or None,
            promote=promote,
            write=write,
        )

    @server.tool()
    def compare_asr_variants(reference_json: str, primary: str, secondary: str, integrated: str, output_json: str = "") -> dict[str, Any]:
        from .asr_variant_evaluation import compare_asr_variants as compare
        return compare(reference_json, primary=primary, secondary=secondary, integrated=integrated, output_json=output_json or None)

    @server.tool()
    def dual_asr_review(
        bundle_dir: str, media_path: str = "", primary_transcript: str = "", secondary_transcript: str = "",
        mode: str = "important", secondary_preset: str = "qwen3-asr-1.7b", execute: bool = False,
        speech_activity_json: str = "", diarization_json: str = "", max_windows: int = 24,
        timeout_seconds: int = 1800, allow_cpu: bool = False, write: bool = True,
        recover_failed_windows: bool = True, recovery_window_seconds: int = 60,
        recovery_max_windows: int = 24,
    ) -> dict[str, Any]:
        from .dual_asr_workflow import run_dual_asr_workflow
        return run_dual_asr_workflow(bundle_dir, media_path=media_path or None,
            primary_transcript=primary_transcript or None, secondary_transcript=secondary_transcript or None,
            mode=mode, secondary_preset=secondary_preset, execute=execute,
            speech_activity_json=speech_activity_json or None, diarization_json=diarization_json or None,
            max_windows=max_windows, timeout_seconds=timeout_seconds, allow_cpu=allow_cpu, write=write,
            recover_failed_windows=recover_failed_windows, recovery_window_seconds=recovery_window_seconds,
            recovery_max_windows=recovery_max_windows)

    @server.tool()
    def plan_qwen_window_recovery(
        partial_json: str, output_dir: str, asr_plan_json: str = "", media_path: str = "",
        speech_activity_json: str = "", window_seconds: int = 60, max_windows: int = 24,
        write: bool = True,
    ) -> dict[str, Any]:
        """Prepare a bounded recovery contract; never execute an ASR model."""
        from .qwen_failed_window_recovery import plan_qwen_failed_window_recovery
        return plan_qwen_failed_window_recovery(partial_json, output_dir=output_dir,
            asr_plan_json=asr_plan_json or None, media_path=media_path or None,
            speech_activity_json=speech_activity_json or None, window_seconds=window_seconds,
            max_windows=max_windows, write=write)

    @server.tool()
    def compose_qwen_window_recovery(plan_json: str, recovered_json: str, write: bool = True) -> dict[str, Any]:
        """Validate saved recovery evidence and compose a reference; no model calls."""
        from .qwen_failed_window_recovery import compose_qwen_failed_window_recovery
        return compose_qwen_failed_window_recovery(plan_json, recovered_json=recovered_json, write=write)

    @server.tool()
    def asr_consensus(
        bundle_dir: str,
        primary_transcript: str,
        secondary_transcript: str,
        media_path: str = "",
        agreement_threshold: float = 0.86,
        execute_clips: bool = False,
        speech_activity_json: str = "",
        diarization_json: str = "",
        critical_terms: list[str] | None = None,
        agreement_sample_size: int = 3,
        secondary_scope: str = "full",
        write: bool = True,
    ) -> dict[str, Any]:
        return build_asr_consensus_impl(bundle_dir, primary_transcript=primary_transcript, secondary_transcript=secondary_transcript, media_path=media_path or None, agreement_threshold=agreement_threshold, execute_clips=execute_clips, write=write,
            speech_activity_json=speech_activity_json or None, diarization_json=diarization_json or None,
            critical_terms=critical_terms, agreement_sample_size=agreement_sample_size, secondary_scope=secondary_scope)

    @server.tool()
    def asr_diff_adjudication(
        bundle_dir: str,
        consensus_json: str = "",
        cluster_token_gap: int = 6,
        write: bool = True,
    ) -> dict[str, Any]:
        return build_asr_diff_adjudication_impl(
            bundle_dir,
            consensus_json=consensus_json or None,
            cluster_token_gap=cluster_token_gap,
            write=write,
        )

    @server.tool()
    def apply_asr_diff_adjudication(
        bundle_dir: str,
        decisions_json: str,
        pack_json: str = "",
        min_confidence: float = 0.75,
        require_evidence: bool = True,
        promote: bool = False,
        write: bool = True,
    ) -> dict[str, Any]:
        return apply_asr_diff_adjudication_impl(
            bundle_dir,
            decisions_json=decisions_json,
            pack_json=pack_json or None,
            min_confidence=min_confidence,
            require_evidence=require_evidence,
            promote=promote,
            write=write,
        )

    @server.tool()
    def quality_finalize(
        bundle_dir: str,
        provider_config: dict[str, Any] | None = None,
        execute_llm: bool = False,
        auto_from_profile: bool = False,
        quality_profile: str = "quality",
        target_chapters: int = 8,
        write: bool = True,
    ) -> dict[str, Any]:
        return finalize_quality_outputs_impl(
            bundle_dir,
            provider_config=provider_config,
            execute_llm=execute_llm,
            auto_from_profile=auto_from_profile,
            quality_profile=quality_profile,
            target_chapters=target_chapters,
            write=write,
        )

    @server.tool()
    def quality_benchmark(
        action: str,
        input_path: str,
        bundle_dirs: list[str] | None = None,
        media_paths: list[str] | None = None,
        output_dir: str = "",
        samples_per_bundle: int = 8,
        sample_seconds: float = 60.0,
        execute_clips: bool = False,
        variants: list[str] | None = None,
        execute: bool = False,
        resume: bool = True,
        retry_failed: bool = False,
        limit: int = 0,
        timeout_seconds: int = 1800,
        scores_json: str = "",
        private_json: str = "",
        manifest_json: str = "",
        entity_lexicon_json: str = "",
        decisions_json: str = "",
        primary_variant: str = "sensevoice_full_punc",
        secondary_variant: str = "qwen3_asr_1_7b",
        min_confidence: float = 0.75,
        punctuation_model: str = "ct-punc",
        punctuation_device: str = "auto",
        write: bool = True,
    ) -> dict[str, Any]:
        if action == "build":
            return build_quality_benchmark_impl(input_path, bundle_dirs=bundle_dirs or [], media_paths=media_paths or [], samples_per_bundle=samples_per_bundle, sample_seconds=sample_seconds, execute_clips=execute_clips, write=write)
        if action == "execute-variants":
            return execute_quality_benchmark_variants_impl(
                input_path,
                variants=variants,
                execute=execute,
                resume=resume,
                retry_failed=retry_failed,
                limit=limit,
                timeout_seconds=timeout_seconds,
                write=write,
            )
        if action == "run":
            return run_quality_benchmark_impl(input_path, output_dir=output_dir or None, write=write)
        if action == "build-arbitration":
            return build_quality_benchmark_arbitration_impl(
                input_path,
                output_dir=output_dir or None,
                primary_variant=primary_variant,
                secondary_variant=secondary_variant,
                write=write,
            )
        if action == "evaluate-arbitration":
            return evaluate_quality_benchmark_arbitration_impl(
                input_path,
                private_json=private_json or None,
                decisions_json=decisions_json or None,
                output_dir=output_dir or None,
                min_confidence=min_confidence,
                write=write,
            )
        if action == "build-punctuation-agent":
            return build_quality_benchmark_punctuation_agent_pack_impl(
                input_path,
                output_dir=output_dir or None,
                source_variant=primary_variant,
                write=write,
            )
        if action == "evaluate-punctuation-agent":
            if not decisions_json:
                raise ValueError("decisions_json is required for evaluate-punctuation-agent")
            return evaluate_quality_benchmark_punctuation_agent_impl(
                input_path,
                decisions_json,
                output_dir=output_dir or None,
                source_variant=primary_variant,
                write=write,
            )
        if action == "build-residual-conflicts":
            return build_quality_benchmark_residual_conflicts_impl(
                input_path,
                manifest_json=manifest_json or None,
                entity_lexicon_json=entity_lexicon_json or None,
                output_dir=output_dir or None,
                write=write,
            )
        if action == "punctuation-ab":
            return run_quality_benchmark_punctuation_impl(
                input_path,
                output_dir=output_dir or None,
                source_variant=primary_variant,
                model=punctuation_model,
                device=punctuation_device,
                execute=execute,
                write=write,
            )
        if action == "build-summary-review":
            return build_summary_blind_review_impl(input_path, output_dir=output_dir or None, write=write)
        if action == "apply-summary-review":
            if not scores_json:
                raise ValueError("scores_json is required for apply-summary-review")
            return apply_summary_blind_review_impl(input_path, scores_json, write=write)
        if action == "report":
            return report_quality_benchmark_impl(input_path, output_dir=output_dir or None, write=write)
        raise ValueError("action must be build, execute-variants, run, report, build-arbitration, evaluate-arbitration, punctuation-ab, build-punctuation-agent, evaluate-punctuation-agent, build-residual-conflicts, build-summary-review, or apply-summary-review")

    @server.tool()
    def punctuation_model_stage(
        bundle_dir: str,
        input_path: str = "",
        model: str = "ct-punc",
        device: str = "auto",
        block_chars: int = 480,
        execute: bool = False,
        promote: bool = False,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_punctuation_model_stage_impl(
            bundle_dir,
            input_path=input_path or None,
            model=model,
            device=device,
            block_chars=block_chars,
            execute=execute,
            promote=promote,
            write=write,
        )

    @server.tool()
    def offline_quality_route(
        bundle_dir: str,
        benchmark_manifest: str = "",
        output_dir: str = "",
        write: bool = True,
    ) -> dict[str, Any]:
        return offline_quality_route_impl(
            bundle_dir,
            benchmark_manifest=benchmark_manifest or None,
            output_dir=output_dir or None,
            write=write,
        )

    @server.tool()
    def transcript_quality_gate(
        bundle_dir: str,
        input_path: str = "",
        reference_path: str = "",
        baseline_path: str = "",
        min_punctuation_per_1000: float = 50.0,
        max_punctuation_per_1000: float = 140.0,
        max_cer: float = 0.18,
        min_entity_accuracy: float = 0.98,
        max_overcorrection_rate: float = 0.01,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_transcript_quality_gate_impl(
            bundle_dir,
            input_path=input_path or None,
            reference_path=reference_path or None,
            baseline_path=baseline_path or None,
            min_punctuation_per_1000=min_punctuation_per_1000,
            max_punctuation_per_1000=max_punctuation_per_1000,
            max_cer=max_cer,
            min_entity_accuracy=min_entity_accuracy,
            max_overcorrection_rate=max_overcorrection_rate,
            write=write,
        )

    @server.tool()
    def evidence_conflict_index(bundle_dir: str, input_json: str = "", limit: int = 0, write: bool = True) -> dict[str, Any]:
        return build_evidence_conflict_index_impl(bundle_dir, input_json=input_json or None, limit=limit, write=write)

    @server.tool()
    def evidence_conflict_index_tool(bundle_dir: str, input_json: str = "", limit: int = 0, write: bool = True) -> dict[str, Any]:
        return build_evidence_conflict_index_impl(bundle_dir, input_json=input_json or None, limit=limit, write=write)

    @server.tool()
    def transcript_evidence_correction_pipeline(
        bundle_dir: str,
        platform_subtitle: str = "",
        subtitle: str = "",
        asr_json: str = "",
        secondary_asr_json: str = "",
        additional_secondary_asr_json: list[str] | None = None,
        consensus_agreement_threshold: float = 0.86,
        execute_consensus_clips: bool = False,
        media_path: str = "",
        asr_mode: str = "",
        execute_local_asr: bool = False,
        speech_activity_json: str = "",
        diarization_json: str = "",
        glossary_json: str = "",
        provider_config: dict[str, Any] | None = None,
        quality_profile: str = "quality",
        execute_llm: bool = False,
        use_agent_substitute: bool = True,
        agent_name: str = "local_agent",
        use_codex_substitute: bool | None = None,
        run_readable_llm: bool = True,
        execute_readable_llm: bool = False,
        promote_readable_llm: bool = False,
        readable_max_segments_per_batch: int = 40,
        readable_max_prompt_chars: int = 9000,
        readable_max_tokens: int = 4000,
        auto_apply_high_confidence: bool = False,
        run_postprocess: bool = True,
        run_source_arbitration: bool = True,
        source_min_confidence: float = 0.86,
        semantic_min_confidence: float = 0.88,
        semantic_limit: int = 0,
        refresh_exports: bool = True,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_transcript_evidence_correction_pipeline_impl(
            bundle_dir,
            platform_subtitle=platform_subtitle or None,
            subtitle=subtitle or None,
            asr_json=asr_json or None,
            secondary_asr_json=secondary_asr_json or None,
            additional_secondary_asr_json=additional_secondary_asr_json,
            consensus_agreement_threshold=consensus_agreement_threshold,
            execute_consensus_clips=execute_consensus_clips,
            media_path=media_path or None, asr_mode=asr_mode, execute_local_asr=execute_local_asr,
            speech_activity_json=speech_activity_json or None, diarization_json=diarization_json or None,
            glossary_json=glossary_json or None,
            provider_config=provider_config,
            quality_profile=quality_profile,
            execute_llm=execute_llm,
            use_agent_substitute=use_agent_substitute,
            agent_name=agent_name,
            use_codex_substitute=use_codex_substitute,
            run_readable_llm=run_readable_llm,
            execute_readable_llm=execute_readable_llm,
            promote_readable_llm=promote_readable_llm,
            readable_max_segments_per_batch=readable_max_segments_per_batch,
            readable_max_prompt_chars=readable_max_prompt_chars,
            readable_max_tokens=readable_max_tokens,
            auto_apply_high_confidence=auto_apply_high_confidence,
            run_postprocess=run_postprocess,
            run_source_arbitration=run_source_arbitration,
            source_min_confidence=source_min_confidence,
            semantic_min_confidence=semantic_min_confidence,
            semantic_limit=semantic_limit,
            refresh_exports=refresh_exports,
            write=write,
        )

    @server.tool()
    def transcript_evidence_correction_pipeline_tool(
        bundle_dir: str,
        platform_subtitle: str = "",
        subtitle: str = "",
        asr_json: str = "",
        secondary_asr_json: str = "",
        additional_secondary_asr_json: list[str] | None = None,
        consensus_agreement_threshold: float = 0.86,
        execute_consensus_clips: bool = False,
        media_path: str = "",
        asr_mode: str = "",
        execute_local_asr: bool = False,
        speech_activity_json: str = "",
        diarization_json: str = "",
        glossary_json: str = "",
        provider_config: dict[str, Any] | None = None,
        quality_profile: str = "quality",
        execute_llm: bool = False,
        use_agent_substitute: bool = True,
        agent_name: str = "local_agent",
        use_codex_substitute: bool | None = None,
        run_readable_llm: bool = True,
        execute_readable_llm: bool = False,
        promote_readable_llm: bool = False,
        readable_max_segments_per_batch: int = 40,
        readable_max_prompt_chars: int = 9000,
        readable_max_tokens: int = 4000,
        auto_apply_high_confidence: bool = False,
        run_postprocess: bool = True,
        run_source_arbitration: bool = True,
        source_min_confidence: float = 0.86,
        semantic_min_confidence: float = 0.88,
        semantic_limit: int = 0,
        refresh_exports: bool = True,
        write: bool = True,
    ) -> dict[str, Any]:
        return transcript_evidence_correction_pipeline(
            bundle_dir=bundle_dir,
            platform_subtitle=platform_subtitle,
            subtitle=subtitle,
            asr_json=asr_json,
            secondary_asr_json=secondary_asr_json,
            additional_secondary_asr_json=additional_secondary_asr_json,
            consensus_agreement_threshold=consensus_agreement_threshold,
            execute_consensus_clips=execute_consensus_clips,
            media_path=media_path, asr_mode=asr_mode, execute_local_asr=execute_local_asr,
            speech_activity_json=speech_activity_json, diarization_json=diarization_json,
            glossary_json=glossary_json,
            provider_config=provider_config,
            quality_profile=quality_profile,
            execute_llm=execute_llm,
            use_agent_substitute=use_agent_substitute,
            agent_name=agent_name,
            use_codex_substitute=use_codex_substitute,
            run_readable_llm=run_readable_llm,
            execute_readable_llm=execute_readable_llm,
            promote_readable_llm=promote_readable_llm,
            readable_max_segments_per_batch=readable_max_segments_per_batch,
            readable_max_prompt_chars=readable_max_prompt_chars,
            readable_max_tokens=readable_max_tokens,
            auto_apply_high_confidence=auto_apply_high_confidence,
            run_postprocess=run_postprocess,
            run_source_arbitration=run_source_arbitration,
            source_min_confidence=source_min_confidence,
            semantic_min_confidence=semantic_min_confidence,
            semantic_limit=semantic_limit,
            refresh_exports=refresh_exports,
            write=write,
        )

    @server.tool()
    def transcript_main_route_status(bundle_dir: str, write: bool = True) -> dict[str, Any]:
        return transcript_main_route_status_impl(bundle_dir, write=write)

    @server.tool()
    def transcript_main_route_status_tool(bundle_dir: str, write: bool = True) -> dict[str, Any]:
        return transcript_main_route_status_impl(bundle_dir, write=write)
    @server.tool()
    def run_extractor_plan_tool(plan_json: str, extractor: str, execute: bool = False, timeout_seconds: int = 0) -> dict[str, Any]:
        return run_extractor_plan_impl(plan_json, extractor, execute=execute, timeout_seconds=timeout_seconds)

    @server.tool()
    def run_extractor_plan(plan_json: str, extractor: str, execute: bool = False, timeout_seconds: int = 0) -> dict[str, Any]:
        return run_extractor_plan_impl(plan_json, extractor, execute=execute, timeout_seconds=timeout_seconds)

    @server.tool()
    def extractor_run_log_tool(workspace_dir: str) -> dict[str, Any]:
        return extractor_run_log_impl(workspace_dir)

    @server.tool()
    def extractor_run_log(workspace_dir: str) -> dict[str, Any]:
        return extractor_run_log_impl(workspace_dir)

    @server.tool()
    def attach_peepshow_output_tool(bundle_dir: str, output_dir: str, write: bool = True) -> dict[str, Any]:
        return attach_peepshow_output_to_bundle_impl(bundle_dir, output_dir, write=write)

    @server.tool()
    def attach_peepshow_output(bundle_dir: str, output_dir: str, write: bool = True) -> dict[str, Any]:
        return attach_peepshow_output_to_bundle_impl(bundle_dir, output_dir, write=write)

    @server.tool()
    def apply_review_notes_tool(bundle_dir: str, review_json: str | None = None, write: bool = True) -> dict[str, Any]:
        return apply_review_notes_to_bundle_impl(bundle_dir, review_json=review_json, write=write)

    @server.tool()
    def apply_review_notes(bundle_dir: str, review_json: str | None = None, write: bool = True) -> dict[str, Any]:
        return apply_review_notes_to_bundle_impl(bundle_dir, review_json=review_json, write=write)

    @server.tool()
    def validate_review_notes_tool(bundle_dir: str, review_json: str | None = None) -> dict[str, Any]:
        return validate_review_notes_for_bundle_impl(bundle_dir, review_json=review_json)

    @server.tool()
    def validate_review_notes(bundle_dir: str, review_json: str | None = None) -> dict[str, Any]:
        return validate_review_notes_for_bundle_impl(bundle_dir, review_json=review_json)

    @server.tool()
    def prepare_review_session_tool(
        bundle_dir: str,
        refresh: bool = True,
        limit: int = 30,
        offset: int = 0,
        reason: str = "",
        group_by: str = "reason",
        include_closed: bool = False,
        output_prefix: str = "review-pack",
    ) -> dict[str, Any]:
        return prepare_review_session_impl(
            bundle_dir,
            refresh=refresh,
            limit=limit,
            offset=offset,
            reason=reason,
            group_by=group_by,
            include_closed=include_closed,
            output_prefix=output_prefix,
        )

    @server.tool()
    def prepare_review_session(
        bundle_dir: str,
        refresh: bool = True,
        limit: int = 30,
        offset: int = 0,
        reason: str = "",
        group_by: str = "reason",
        include_closed: bool = False,
        output_prefix: str = "review-pack",
    ) -> dict[str, Any]:
        return prepare_review_session_impl(
            bundle_dir,
            refresh=refresh,
            limit=limit,
            offset=offset,
            reason=reason,
            group_by=group_by,
            include_closed=include_closed,
            output_prefix=output_prefix,
        )

    @server.tool()
    def review_closure_status_tool(bundle_dir: str, write: bool = True) -> dict[str, Any]:
        return review_closure_status_impl(bundle_dir, write=write)

    @server.tool()
    def review_closure_status(bundle_dir: str, write: bool = True) -> dict[str, Any]:
        return review_closure_status_impl(bundle_dir, write=write)

    @server.tool()
    def import_tagger_annotations_tool(bundle_dir: str, tagger_json: str, source: str = "qinglong", write: bool = True) -> dict[str, Any]:
        return import_tagger_annotations_impl(bundle_dir, tagger_json, source=source, write=write)

    @server.tool()
    def import_tagger_annotations(bundle_dir: str, tagger_json: str, source: str = "qinglong", write: bool = True) -> dict[str, Any]:
        return import_tagger_annotations_impl(bundle_dir, tagger_json, source=source, write=write)

    @server.tool()
    def run_video_frame_router_tool(bundle_dir: str, input_json: str | None = None, write: bool = True) -> dict[str, Any]:
        return run_video_frame_router_impl(bundle_dir, input_json=input_json, write=write)

    @server.tool()
    def run_video_frame_router(bundle_dir: str, input_json: str | None = None, write: bool = True) -> dict[str, Any]:
        return run_video_frame_router_impl(bundle_dir, input_json=input_json, write=write)

    @server.tool()
    def run_ocr_backfill_tool(
        bundle_dir: str,
        input_json: str | None = None,
        execute: bool = False,
        language: str = "chi_sim+eng",
        captiocr_root: str | None = None,
        limit: int = 0,
    ) -> dict[str, Any]:
        return run_ocr_backfill_impl(bundle_dir, input_json=input_json, execute=execute, language=language, captiocr_root=captiocr_root, limit=limit)

    @server.tool()
    def run_ocr_backfill(
        bundle_dir: str,
        input_json: str | None = None,
        execute: bool = False,
        language: str = "chi_sim+eng",
        captiocr_root: str | None = None,
        limit: int = 0,
    ) -> dict[str, Any]:
        return run_ocr_backfill_impl(bundle_dir, input_json=input_json, execute=execute, language=language, captiocr_root=captiocr_root, limit=limit)

    @server.tool()
    def run_screen_text_recovery_tool(
        bundle_dir: str,
        execute_crops: bool = False,
        execute_ocr: bool = False,
        input_json: str | None = None,
        language: str = "chi_sim+eng",
        captiocr_root: str | None = None,
        limit: int = 0,
        indexes: list[int] | None = None,
        write: bool = True,
        profile: str = "default", max_regions: int = 12, max_attempts: int = 2,
    ) -> dict[str, Any]:
        return run_screen_text_recovery_impl(
            bundle_dir,
            execute_crops=execute_crops,
            execute_ocr=execute_ocr,
            input_json=input_json,
            language=language,
            captiocr_root=captiocr_root,
            limit=limit,
            indexes=indexes,
            write=write,
            profile=profile, max_regions=max_regions, max_attempts=max_attempts,
        )

    @server.tool()
    def run_screen_text_recovery(
        bundle_dir: str,
        execute_crops: bool = False,
        execute_ocr: bool = False,
        input_json: str | None = None,
        language: str = "chi_sim+eng",
        captiocr_root: str | None = None,
        limit: int = 0,
        indexes: list[int] | None = None,
        write: bool = True,
        profile: str = "default", max_regions: int = 12, max_attempts: int = 2,
    ) -> dict[str, Any]:
        return run_screen_text_recovery_impl(
            bundle_dir,
            execute_crops=execute_crops,
            execute_ocr=execute_ocr,
            input_json=input_json,
            language=language,
            captiocr_root=captiocr_root,
            limit=limit,
            indexes=indexes,
            write=write,
            profile=profile, max_regions=max_regions, max_attempts=max_attempts,
        )
    @server.tool()
    def high_res_tile_plan_tool(
        bundle_dir: str,
        execute_tiles: bool = False,
        indexes: list[int] | None = None,
        limit: int = 0,
        tile_size: int = 768,
        overlap: float = 0.12,
        max_tiles_per_image: int = 12,
        include_routes: list[str] | None = None,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_high_res_tile_plan_impl(
            bundle_dir,
            execute_tiles=execute_tiles,
            indexes=indexes,
            limit=limit,
            tile_size=tile_size,
            overlap=overlap,
            max_tiles_per_image=max_tiles_per_image,
            include_routes=include_routes,
            write=write,
        )

    @server.tool()
    def run_high_res_tile_plan(
        bundle_dir: str,
        execute_tiles: bool = False,
        indexes: list[int] | None = None,
        limit: int = 0,
        tile_size: int = 768,
        overlap: float = 0.12,
        max_tiles_per_image: int = 12,
        include_routes: list[str] | None = None,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_high_res_tile_plan_impl(
            bundle_dir,
            execute_tiles=execute_tiles,
            indexes=indexes,
            limit=limit,
            tile_size=tile_size,
            overlap=overlap,
            max_tiles_per_image=max_tiles_per_image,
            include_routes=include_routes,
            write=write,
        )

    @server.tool()
    def tile_result_import_build_tool(
        bundle_dir: str,
        results_dir: str | None = None,
        output_json: str | None = None,
        default_source: str = "tile_result_import_builder",
        default_confidence: float = 0.0,
        write: bool = True,
    ) -> dict[str, Any]:
        return build_tile_result_import_impl(
            bundle_dir,
            results_dir=results_dir,
            output_json=output_json,
            default_source=default_source,
            default_confidence=default_confidence,
            write=write,
        )

    @server.tool()
    def build_tile_result_import(
        bundle_dir: str,
        results_dir: str | None = None,
        output_json: str | None = None,
        default_source: str = "tile_result_import_builder",
        default_confidence: float = 0.0,
        write: bool = True,
    ) -> dict[str, Any]:
        return build_tile_result_import_impl(
            bundle_dir,
            results_dir=results_dir,
            output_json=output_json,
            default_source=default_source,
            default_confidence=default_confidence,
            write=write,
        )

    @server.tool()
    def tile_result_merge_tool(
        bundle_dir: str,
        input_json: str | None = None,
        execute: bool = False,
        min_confidence: float = 0.65,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_tile_result_merge_impl(
            bundle_dir,
            input_json=input_json,
            execute=execute,
            min_confidence=min_confidence,
            write=write,
        )

    @server.tool()
    def run_tile_result_merge(
        bundle_dir: str,
        input_json: str | None = None,
        execute: bool = False,
        min_confidence: float = 0.65,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_tile_result_merge_impl(
            bundle_dir,
            input_json=input_json,
            execute=execute,
            min_confidence=min_confidence,
            write=write,
        )

    @server.tool()
    def run_visual_structure_tool(
        bundle_dir: str,
        input_json: str | None = None,
        execute_ebook_pipeline: bool = False,
        include_routes: list[str] | None = None,
        timeout_seconds: int = 120,
        indexes: list[int] | None = None,
        limit: int = 0,
    ) -> dict[str, Any]:
        return run_visual_structure_plan_impl(
            bundle_dir,
            input_json=input_json,
            execute_ebook_pipeline=execute_ebook_pipeline,
            include_routes=include_routes,
            timeout_seconds=timeout_seconds,
            indexes=indexes,
            limit=limit,
        )

    @server.tool()
    def run_visual_structure_plan(
        bundle_dir: str,
        input_json: str | None = None,
        execute_ebook_pipeline: bool = False,
        include_routes: list[str] | None = None,
        timeout_seconds: int = 120,
        indexes: list[int] | None = None,
        limit: int = 0,
    ) -> dict[str, Any]:
        return run_visual_structure_plan_impl(
            bundle_dir,
            input_json=input_json,
            execute_ebook_pipeline=execute_ebook_pipeline,
            include_routes=include_routes,
            timeout_seconds=timeout_seconds,
            indexes=indexes,
            limit=limit,
        )

    @server.tool()
    def run_multimodal_frame_analysis_tool(
        bundle_dir: str,
        input_json: str | None = None,
        execute: bool = False,
        provider_config: dict[str, Any] | None = None,
        limit: int | None = None,
        indexes: list[int] | None = None,
        confirm_vision_calls: int | None = None,
        confirm_vision_indexes: str = "",
        image_probe_max_edge: int = 0,
        image_probe_jpeg_quality: int = 70,
        vision_retries: int = 1,
        vision_retry_delay_seconds: float = 0.0,
        execution_actor: str = "agent",
        export_consent: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        return run_multimodal_frame_analysis_impl(
            bundle_dir,
            input_json=input_json,
            execute=execute,
            provider_config=provider_config,
            limit=limit,
            indexes=indexes,
            confirm_vision_calls=confirm_vision_calls,
            confirm_vision_indexes=confirm_vision_indexes,
            image_probe_max_edge=image_probe_max_edge,
            image_probe_jpeg_quality=image_probe_jpeg_quality,
            vision_retries=vision_retries,
            vision_retry_delay_seconds=vision_retry_delay_seconds,
            execution_actor=execution_actor,
            export_consent=export_consent,
            max_tokens=max_tokens,
        )

    @server.tool()
    def run_multimodal_frame_analysis(
        bundle_dir: str,
        input_json: str | None = None,
        execute: bool = False,
        provider_config: dict[str, Any] | None = None,
        limit: int | None = None,
        indexes: list[int] | None = None,
        confirm_vision_calls: int | None = None,
        confirm_vision_indexes: str = "",
        image_probe_max_edge: int = 0,
        image_probe_jpeg_quality: int = 70,
        vision_retries: int = 1,
        vision_retry_delay_seconds: float = 0.0,
        execution_actor: str = "agent",
        export_consent: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        return run_multimodal_frame_analysis_impl(
            bundle_dir,
            input_json=input_json,
            execute=execute,
            provider_config=provider_config,
            limit=limit,
            indexes=indexes,
            confirm_vision_calls=confirm_vision_calls,
            confirm_vision_indexes=confirm_vision_indexes,
            image_probe_max_edge=image_probe_max_edge,
            image_probe_jpeg_quality=image_probe_jpeg_quality,
            vision_retries=vision_retries,
            vision_retry_delay_seconds=vision_retry_delay_seconds,
            execution_actor=execution_actor,
            export_consent=export_consent,
            max_tokens=max_tokens,
        )

    @server.tool()
    def run_temporal_frame_groups_tool(
        bundle_dir: str,
        execute: bool = False,
        frame_count: int | None = None,
        window_seconds: float = 4.0,
        include_routes: list[str] | None = None,
        limit: int | None = None,
        timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        return run_temporal_frame_groups_impl(
            bundle_dir,
            execute=execute,
            frame_count=frame_count,
            window_seconds=window_seconds,
            include_routes=include_routes,
            limit=limit,
            timeout_seconds=timeout_seconds,
        )

    @server.tool()
    def run_temporal_frame_groups(
        bundle_dir: str,
        execute: bool = False,
        frame_count: int | None = None,
        window_seconds: float = 4.0,
        include_routes: list[str] | None = None,
        limit: int | None = None,
        timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        return run_temporal_frame_groups_impl(
            bundle_dir,
            execute=execute,
            frame_count=frame_count,
            window_seconds=window_seconds,
            include_routes=include_routes,
            limit=limit,
            timeout_seconds=timeout_seconds,
        )

    @server.tool()
    def run_temporal_visual_analysis_tool(
        bundle_dir: str,
        input_json: str | None = None,
        execute: bool = False,
        frame_count: int | None = None,
        limit: int | None = None,
        indexes: list[int] | None = None,
        provider_config: dict[str, Any] | None = None,
        confirm_vision_calls: int | None = None,
        confirm_vision_indexes: str = "",
        image_probe_max_edge: int = 0,
        image_probe_jpeg_quality: int = 70,
        vision_retries: int = 1,
        vision_retry_delay_seconds: float = 0.0,
        execution_actor: str = "agent",
        export_consent: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        return run_temporal_visual_analysis_impl(
            bundle_dir,
            input_json=input_json,
            execute=execute,
            frame_count=frame_count,
            limit=limit,
            indexes=indexes,
            provider_config=provider_config,
            confirm_vision_calls=confirm_vision_calls,
            confirm_vision_indexes=confirm_vision_indexes,
            image_probe_max_edge=image_probe_max_edge,
            image_probe_jpeg_quality=image_probe_jpeg_quality,
            vision_retries=vision_retries,
            vision_retry_delay_seconds=vision_retry_delay_seconds,
            execution_actor=execution_actor,
            export_consent=export_consent,
            max_tokens=max_tokens,
        )

    @server.tool()
    def run_temporal_visual_analysis(
        bundle_dir: str,
        input_json: str | None = None,
        execute: bool = False,
        frame_count: int | None = None,
        limit: int | None = None,
        indexes: list[int] | None = None,
        provider_config: dict[str, Any] | None = None,
        confirm_vision_calls: int | None = None,
        confirm_vision_indexes: str = "",
        image_probe_max_edge: int = 0,
        image_probe_jpeg_quality: int = 70,
        vision_retries: int = 1,
        vision_retry_delay_seconds: float = 0.0,
        execution_actor: str = "agent",
        export_consent: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        return run_temporal_visual_analysis_impl(
            bundle_dir,
            input_json=input_json,
            execute=execute,
            frame_count=frame_count,
            limit=limit,
            indexes=indexes,
            provider_config=provider_config,
            confirm_vision_calls=confirm_vision_calls,
            confirm_vision_indexes=confirm_vision_indexes,
            image_probe_max_edge=image_probe_max_edge,
            image_probe_jpeg_quality=image_probe_jpeg_quality,
            vision_retries=vision_retries,
            vision_retry_delay_seconds=vision_retry_delay_seconds,
            execution_actor=execution_actor,
            export_consent=export_consent,
            max_tokens=max_tokens,
        )

    @server.tool()
    def adaptive_asr_route_tool(
        bundle_dir: str,
        media_path: str,
        workspace_dir: str = "",
        task_profile: str = "balanced",
        base_lexicon_json: str = "",
        include_online_plan: bool = False,
        provider_config: dict[str, Any] | None = None,
        online_model: str = "",
        language: str = "zh",
        max_hotwords: int = 80,
        max_context_chars: int = 1200,
        write: bool = True,
    ) -> dict[str, Any]:
        """Build model-independent ASR plans using OCR/entity context and explicit online gates."""
        return build_adaptive_asr_route_impl(
            bundle_dir,
            media_path,
            workspace_dir=workspace_dir or None,
            task_profile=task_profile,
            base_lexicon_json=base_lexicon_json or None,
            include_online_plan=include_online_plan,
            provider_config=provider_config,
            online_model=online_model,
            language=language,
            max_hotwords=max_hotwords,
            max_context_chars=max_context_chars,
            write=write,
        )

    @server.tool()
    def ocr_route_tool(
        bundle_dir: str,
        backend: str = "local",
        execute_local: bool = False,
        connector_result_json: str = "",
        provider_config: dict[str, Any] | None = None,
        indexes: list[int] | None = None,
        limit: int = 8,
        timeout_seconds: int = 120,
    ) -> dict[str, Any]:
        """Choose local ebook Markdown OCR or plan/import consent-gated online OCR."""
        return run_ocr_route_impl(
            bundle_dir,
            backend=backend,
            execute_local=execute_local,
            connector_result_json=connector_result_json or None,
            provider_config=provider_config,
            indexes=indexes,
            limit=limit,
            timeout_seconds=timeout_seconds,
        )

    @server.tool()
    def build_entity_lexicon_tool(
        bundle_dir: str,
        base_lexicon_json: str = "",
        phase: str = "post_asr",
        write: bool = True,
    ) -> dict[str, Any]:
        """Build phase-scoped entity terms; only pre_asr hotwords may enter ASR plans."""
        return build_entity_lexicon_impl(
            bundle_dir,
            base_lexicon_json=base_lexicon_json or None,
            phase=phase,
            write=write,
        )

    @server.tool()
    def resolve_terms_tool(
        bundle_dir: str,
        metadata_json: str | None = None,
        glossary_json: str | None = None,
        min_mentions: int = 1,
        write: bool = True,
    ) -> dict[str, Any]:
        return resolve_terms_impl(bundle_dir, metadata_json=metadata_json, glossary_json=glossary_json, min_mentions=min_mentions, write=write)

    @server.tool()
    def resolve_terms(
        bundle_dir: str,
        metadata_json: str | None = None,
        glossary_json: str | None = None,
        min_mentions: int = 1,
        write: bool = True,
    ) -> dict[str, Any]:
        return resolve_terms_impl(bundle_dir, metadata_json=metadata_json, glossary_json=glossary_json, min_mentions=min_mentions, write=write)

    @server.tool()
    def term_arbitration_codex_tool(
        bundle_dir: str,
        input_json: str | None = None,
        max_terms: int = 60,
        min_confidence: float = 0.88,
        accept_draft: bool = False,
        write: bool = True,
    ) -> dict[str, Any]:
        return build_term_arbitration_codex_pack_impl(bundle_dir, input_json=input_json, max_terms=max_terms, min_confidence=min_confidence, accept_draft=accept_draft, write=write)

    @server.tool()
    def term_arbitration_codex(
        bundle_dir: str,
        input_json: str | None = None,
        max_terms: int = 60,
        min_confidence: float = 0.88,
        accept_draft: bool = False,
        write: bool = True,
    ) -> dict[str, Any]:
        return build_term_arbitration_codex_pack_impl(bundle_dir, input_json=input_json, max_terms=max_terms, min_confidence=min_confidence, accept_draft=accept_draft, write=write)
    @server.tool()
    def validate_term_arbitration_codex_result_tool(
        bundle_dir: str,
        input_json: str,
        min_confidence: float = 0.88,
        write: bool = True,
    ) -> dict[str, Any]:
        return validate_term_arbitration_codex_result_impl(bundle_dir, input_json=input_json, min_confidence=min_confidence, write=write)

    @server.tool()
    def validate_term_arbitration_codex_result(
        bundle_dir: str,
        input_json: str,
        min_confidence: float = 0.88,
        write: bool = True,
    ) -> dict[str, Any]:
        return validate_term_arbitration_codex_result_impl(bundle_dir, input_json=input_json, min_confidence=min_confidence, write=write)
    @server.tool()
    def term_correction_impact_report_tool(
        bundle_dir: str,
        min_confidence: float = 0.88,
        write: bool = True,
    ) -> dict[str, Any]:
        return term_correction_impact_report_impl(bundle_dir, min_confidence=min_confidence, write=write)

    @server.tool()
    def term_correction_impact_report(
        bundle_dir: str,
        min_confidence: float = 0.88,
        write: bool = True,
    ) -> dict[str, Any]:
        return term_correction_impact_report_impl(bundle_dir, min_confidence=min_confidence, write=write)

    @server.tool()
    def term_correction_status_tool(bundle_dir: str) -> dict[str, Any]:
        return term_correction_status_impl(bundle_dir)

    @server.tool()
    def term_correction_status(bundle_dir: str) -> dict[str, Any]:
        return term_correction_status_impl(bundle_dir)
    @server.tool()
    def term_correction_closure_tool(
        bundle_dir: str,
        accept_draft: bool = False,
        input_json: str | None = None,
        max_terms: int = 60,
        term_min_confidence: float = 0.88,
        transcript_min_confidence: float = 0.72,
        generate_codex_summary: bool = True,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_term_correction_closure_impl(bundle_dir, accept_draft=accept_draft, input_json=input_json, max_terms=max_terms, term_min_confidence=term_min_confidence, transcript_min_confidence=transcript_min_confidence, generate_codex_summary=generate_codex_summary, write=write)

    @server.tool()
    def term_correction_closure(
        bundle_dir: str,
        accept_draft: bool = False,
        input_json: str | None = None,
        max_terms: int = 60,
        term_min_confidence: float = 0.88,
        transcript_min_confidence: float = 0.72,
        generate_codex_summary: bool = True,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_term_correction_closure_impl(bundle_dir, accept_draft=accept_draft, input_json=input_json, max_terms=max_terms, term_min_confidence=term_min_confidence, transcript_min_confidence=transcript_min_confidence, generate_codex_summary=generate_codex_summary, write=write)

    @server.tool()
    def transcript_semantic_correction_pack_tool(bundle_dir: str, limit: int = 0, write: bool = True) -> dict[str, Any]:
        return build_transcript_semantic_correction_pack_impl(bundle_dir, limit=limit, write=write)

    @server.tool()
    def transcript_semantic_correction_pack(bundle_dir: str, limit: int = 0, write: bool = True) -> dict[str, Any]:
        return build_transcript_semantic_correction_pack_impl(bundle_dir, limit=limit, write=write)

    @server.tool()
    def transcript_semantic_candidate_discovery_pack_tool(bundle_dir: str, input_json: str = "", limit: int = 40, write: bool = True) -> dict[str, Any]:
        return build_transcript_semantic_candidate_discovery_pack_impl(bundle_dir, input_json=input_json or None, limit=limit, write=write)

    @server.tool()
    def transcript_semantic_candidate_discovery_pack(bundle_dir: str, input_json: str = "", limit: int = 40, write: bool = True) -> dict[str, Any]:
        return build_transcript_semantic_candidate_discovery_pack_impl(bundle_dir, input_json=input_json or None, limit=limit, write=write)

    @server.tool()
    def transcript_semantic_candidate_discovery_codex_draft_tool(bundle_dir: str, input_json: str = "", limit: int = 40, max_suggestions: int = 40, write: bool = True) -> dict[str, Any]:
        return build_transcript_semantic_candidate_discovery_codex_draft_impl(bundle_dir, input_json=input_json or None, limit=limit, max_suggestions=max_suggestions, write=write)

    @server.tool()
    def transcript_semantic_candidate_discovery_codex_draft(bundle_dir: str, input_json: str = "", limit: int = 40, max_suggestions: int = 40, write: bool = True) -> dict[str, Any]:
        return build_transcript_semantic_candidate_discovery_codex_draft_impl(bundle_dir, input_json=input_json or None, limit=limit, max_suggestions=max_suggestions, write=write)
    @server.tool()
    def transcript_semantic_candidate_discovery_llm_draft_tool(bundle_dir: str, input_json: str = "", provider_config: dict[str, Any] | None = None, execute: bool = False, limit: int = 40, write: bool = True) -> dict[str, Any]:
        return build_transcript_semantic_candidate_discovery_llm_draft_impl(bundle_dir, input_json=input_json or None, provider_config=provider_config, execute=execute, limit=limit, write=write)

    @server.tool()
    def transcript_semantic_candidate_discovery_llm_draft(bundle_dir: str, input_json: str = "", provider_config: dict[str, Any] | None = None, execute: bool = False, limit: int = 40, write: bool = True) -> dict[str, Any]:
        return build_transcript_semantic_candidate_discovery_llm_draft_impl(bundle_dir, input_json=input_json or None, provider_config=provider_config, execute=execute, limit=limit, write=write)

    @server.tool()
    def import_transcript_semantic_candidate_suggestions_tool(bundle_dir: str, input_json: str, write: bool = True) -> dict[str, Any]:
        return import_transcript_semantic_candidate_suggestions_impl(bundle_dir, input_json=input_json, write=write)

    @server.tool()
    def import_transcript_semantic_candidate_suggestions(bundle_dir: str, input_json: str, write: bool = True) -> dict[str, Any]:
        return import_transcript_semantic_candidate_suggestions_impl(bundle_dir, input_json=input_json, write=write)

    @server.tool()
    def transcript_semantic_correction_codex_draft_tool(bundle_dir: str, input_json: str = "", min_confidence: float = 0.88, write: bool = True) -> dict[str, Any]:
        return build_transcript_semantic_correction_codex_draft_impl(bundle_dir, input_json=input_json or None, min_confidence=min_confidence, write=write)

    @server.tool()
    def transcript_semantic_correction_codex_draft(bundle_dir: str, input_json: str = "", min_confidence: float = 0.88, write: bool = True) -> dict[str, Any]:
        return build_transcript_semantic_correction_codex_draft_impl(bundle_dir, input_json=input_json or None, min_confidence=min_confidence, write=write)
    @server.tool()
    def transcript_semantic_correction_llm_draft_tool(bundle_dir: str, input_json: str = "", provider_config: dict[str, Any] | None = None, execute: bool = False, limit: int = 80, min_confidence: float = 0.88, write: bool = True) -> dict[str, Any]:
        return build_transcript_semantic_correction_llm_draft_impl(bundle_dir, input_json=input_json or None, provider_config=provider_config, execute=execute, limit=limit, min_confidence=min_confidence, write=write)

    @server.tool()
    def transcript_semantic_correction_llm_draft(bundle_dir: str, input_json: str = "", provider_config: dict[str, Any] | None = None, execute: bool = False, limit: int = 80, min_confidence: float = 0.88, write: bool = True) -> dict[str, Any]:
        return build_transcript_semantic_correction_llm_draft_impl(bundle_dir, input_json=input_json or None, provider_config=provider_config, execute=execute, limit=limit, min_confidence=min_confidence, write=write)

    @server.tool()
    def validate_transcript_semantic_correction_tool(bundle_dir: str, input_json: str, min_confidence: float = 0.88, write: bool = True) -> dict[str, Any]:
        return validate_transcript_semantic_correction_impl(bundle_dir, input_json=input_json, min_confidence=min_confidence, write=write)

    @server.tool()
    def validate_transcript_semantic_correction(bundle_dir: str, input_json: str, min_confidence: float = 0.88, write: bool = True) -> dict[str, Any]:
        return validate_transcript_semantic_correction_impl(bundle_dir, input_json=input_json, min_confidence=min_confidence, write=write)

    @server.tool()
    def import_transcript_semantic_review_notes_tool(bundle_dir: str, review_json: str = "", min_confidence: float = 0.88, write: bool = True) -> dict[str, Any]:
        return import_transcript_semantic_review_notes_impl(bundle_dir, review_json=review_json or None, min_confidence=min_confidence, write=write)

    @server.tool()
    def import_transcript_semantic_review_notes(bundle_dir: str, review_json: str = "", min_confidence: float = 0.88, write: bool = True) -> dict[str, Any]:
        return import_transcript_semantic_review_notes_impl(bundle_dir, review_json=review_json or None, min_confidence=min_confidence, write=write)
    @server.tool()
    def transcript_semantic_correction_closure_tool(bundle_dir: str, input_json: str, min_confidence: float = 0.88, auto_apply: bool = False, refresh_exports: bool = False, write: bool = True) -> dict[str, Any]:
        return transcript_semantic_correction_closure_impl(bundle_dir, input_json=input_json, min_confidence=min_confidence, auto_apply=auto_apply, refresh_exports=refresh_exports, write=write)

    @server.tool()
    def transcript_semantic_correction_closure(bundle_dir: str, input_json: str, min_confidence: float = 0.88, auto_apply: bool = False, refresh_exports: bool = False, write: bool = True) -> dict[str, Any]:
        return transcript_semantic_correction_closure_impl(bundle_dir, input_json=input_json, min_confidence=min_confidence, auto_apply=auto_apply, refresh_exports=refresh_exports, write=write)

    @server.tool()
    def transcript_semantic_correction_impact_report_tool(bundle_dir: str, write: bool = True) -> dict[str, Any]:
        return transcript_semantic_correction_impact_report_impl(bundle_dir, write=write)

    @server.tool()
    def transcript_semantic_correction_impact_report(bundle_dir: str, write: bool = True) -> dict[str, Any]:
        return transcript_semantic_correction_impact_report_impl(bundle_dir, write=write)

    @server.tool()
    def transcript_semantic_readable_impact_report_tool(bundle_dir: str, write: bool = True) -> dict[str, Any]:
        return transcript_semantic_correction_readable_impact_report_impl(bundle_dir, write=write)

    @server.tool()
    def transcript_semantic_readable_impact_report(bundle_dir: str, write: bool = True) -> dict[str, Any]:
        return transcript_semantic_correction_readable_impact_report_impl(bundle_dir, write=write)

    @server.tool()
    def transcript_semantic_summary_impact_report_tool(bundle_dir: str, summary_path: str = "", baseline_summary_path: str = "", write: bool = True) -> dict[str, Any]:
        return transcript_semantic_summary_impact_report_impl(bundle_dir, summary_path=summary_path or None, baseline_summary_path=baseline_summary_path or None, write=write)

    @server.tool()
    def transcript_semantic_summary_impact_report(bundle_dir: str, summary_path: str = "", baseline_summary_path: str = "", write: bool = True) -> dict[str, Any]:
        return transcript_semantic_summary_impact_report_impl(bundle_dir, summary_path=summary_path or None, baseline_summary_path=baseline_summary_path or None, write=write)

    @server.tool()
    def transcript_semantic_correction_status_tool(bundle_dir: str, write: bool = False) -> dict[str, Any]:
        return transcript_semantic_correction_status_impl(bundle_dir, write=write)

    @server.tool()
    def transcript_semantic_correction_status(bundle_dir: str, write: bool = False) -> dict[str, Any]:
        return transcript_semantic_correction_status_impl(bundle_dir, write=write)

    @server.tool()
    def transcript_semantic_benchmark_tool(
        gold_json: str,
        candidate_pack_json: str,
        validation_json: str,
        full_body_path: str = "",
        smart_summary_path: str = "",
        output_json: str = "",
    ) -> dict[str, Any]:
        return transcript_semantic_benchmark_impl(
            gold_json,
            candidate_pack_json,
            validation_json,
            full_body_path=full_body_path or None,
            smart_summary_path=smart_summary_path or None,
            output_json=output_json or None,
        )

    @server.tool()
    def transcript_semantic_benchmark(
        gold_json: str,
        candidate_pack_json: str,
        validation_json: str,
        full_body_path: str = "",
        smart_summary_path: str = "",
        output_json: str = "",
    ) -> dict[str, Any]:
        return transcript_semantic_benchmark_impl(
            gold_json,
            candidate_pack_json,
            validation_json,
            full_body_path=full_body_path or None,
            smart_summary_path=smart_summary_path or None,
            output_json=output_json or None,
        )

    @server.tool()
    def apply_transcript_semantic_restore_tool(
        bundle_dir: str,
        plan_json: str,
        execute: bool = False,
        confirm_transaction_id: str = "",
    ) -> dict[str, Any]:
        return apply_transcript_semantic_restore_impl(
            bundle_dir,
            plan_json,
            execute=execute,
            confirm_transaction_id=confirm_transaction_id,
        )

    @server.tool()
    def apply_transcript_semantic_restore(
        bundle_dir: str,
        plan_json: str,
        execute: bool = False,
        confirm_transaction_id: str = "",
    ) -> dict[str, Any]:
        return apply_transcript_semantic_restore_impl(
            bundle_dir,
            plan_json,
            execute=execute,
            confirm_transaction_id=confirm_transaction_id,
        )

    @server.tool()
    def transcript_semantic_acceptance_tool(bundle_dir: str, output_dir: str = "", write: bool = True) -> dict[str, Any]:
        return transcript_semantic_acceptance_impl(bundle_dir, output_dir=output_dir, write=write)

    @server.tool()
    def transcript_semantic_acceptance(bundle_dir: str, output_dir: str = "", write: bool = True) -> dict[str, Any]:
        return transcript_semantic_acceptance_impl(bundle_dir, output_dir=output_dir, write=write)
    @server.tool()
    def transcript_semantic_batch_acceptance_tool(batch_input: str, output_dir: str = "", target_bundle_count: int = 3, limit: int = 0, write: bool = True) -> dict[str, Any]:
        return transcript_semantic_batch_acceptance_impl(batch_input, output_dir=output_dir, target_bundle_count=target_bundle_count, limit=limit, write=write)

    @server.tool()
    def transcript_semantic_batch_acceptance(batch_input: str, output_dir: str = "", target_bundle_count: int = 3, limit: int = 0, write: bool = True) -> dict[str, Any]:
        return transcript_semantic_batch_acceptance_impl(batch_input, output_dir=output_dir, target_bundle_count=target_bundle_count, limit=limit, write=write)
    @server.tool()
    def transcript_semantic_repair_queue_tool(batch_input: str, output_dir: str = "", target_bundle_count: int = 3, limit: int = 0, write: bool = True) -> dict[str, Any]:
        return transcript_semantic_repair_queue_impl(batch_input, output_dir=output_dir, target_bundle_count=target_bundle_count, limit=limit, write=write)

    @server.tool()
    def transcript_semantic_repair_queue(batch_input: str, output_dir: str = "", target_bundle_count: int = 3, limit: int = 0, write: bool = True) -> dict[str, Any]:
        return transcript_semantic_repair_queue_impl(batch_input, output_dir=output_dir, target_bundle_count=target_bundle_count, limit=limit, write=write)

    @server.tool()
    def transcript_semantic_repair_run_tool(batch_input: str, output_dir: str = "", target_bundle_count: int = 3, limit: int = 0, execute_safe_actions: bool = False, max_actions: int = 0, max_rounds: int = 1, allow_closure: bool = False, allow_llm: bool = False, provider_config: dict[str, Any] | None = None, llm_limit: int = 80, write: bool = True) -> dict[str, Any]:
        return transcript_semantic_repair_run_impl(batch_input, output_dir=output_dir, target_bundle_count=target_bundle_count, limit=limit, execute_safe_actions=execute_safe_actions, max_actions=max_actions, max_rounds=max_rounds, allow_closure=allow_closure, allow_llm=allow_llm, provider_config=provider_config, llm_limit=llm_limit, write=write)

    @server.tool()
    def transcript_semantic_repair_run(batch_input: str, output_dir: str = "", target_bundle_count: int = 3, limit: int = 0, execute_safe_actions: bool = False, max_actions: int = 0, max_rounds: int = 1, allow_closure: bool = False, allow_llm: bool = False, provider_config: dict[str, Any] | None = None, llm_limit: int = 80, write: bool = True) -> dict[str, Any]:
        return transcript_semantic_repair_run_impl(batch_input, output_dir=output_dir, target_bundle_count=target_bundle_count, limit=limit, execute_safe_actions=execute_safe_actions, max_actions=max_actions, max_rounds=max_rounds, allow_closure=allow_closure, allow_llm=allow_llm, provider_config=provider_config, llm_limit=llm_limit, write=write)

    @server.tool()
    def transcript_semantic_batch_review_pack_tool(batch_input: str, output_dir: str = "", target_bundle_count: int = 3, limit: int = 0, max_candidates_per_bundle: int = 0, write: bool = True) -> dict[str, Any]:
        return transcript_semantic_batch_review_pack_impl(batch_input, output_dir=output_dir, target_bundle_count=target_bundle_count, limit=limit, max_candidates_per_bundle=max_candidates_per_bundle, write=write)

    @server.tool()
    def transcript_semantic_batch_review_pack(batch_input: str, output_dir: str = "", target_bundle_count: int = 3, limit: int = 0, max_candidates_per_bundle: int = 0, write: bool = True) -> dict[str, Any]:
        return transcript_semantic_batch_review_pack_impl(batch_input, output_dir=output_dir, target_bundle_count=target_bundle_count, limit=limit, max_candidates_per_bundle=max_candidates_per_bundle, write=write)

    @server.tool()
    def transcript_semantic_batch_import_review_notes_tool(review_json: str, output_dir: str = "", min_confidence: float = 0.88, write: bool = True) -> dict[str, Any]:
        return transcript_semantic_batch_import_review_notes_impl(review_json, output_dir=output_dir, min_confidence=min_confidence, write=write)

    @server.tool()
    def transcript_semantic_batch_import_review_notes(review_json: str, output_dir: str = "", min_confidence: float = 0.88, write: bool = True) -> dict[str, Any]:
        return transcript_semantic_batch_import_review_notes_impl(review_json, output_dir=output_dir, min_confidence=min_confidence, write=write)

    @server.tool()
    def transcript_semantic_batch_codex_review_draft_tool(review_pack_json: str, output_dir: str = "", write: bool = True) -> dict[str, Any]:
        return transcript_semantic_batch_codex_review_draft_impl(review_pack_json, output_dir=output_dir, write=write)

    @server.tool()
    def transcript_semantic_batch_codex_review_draft(review_pack_json: str, output_dir: str = "", write: bool = True) -> dict[str, Any]:
        return transcript_semantic_batch_codex_review_draft_impl(review_pack_json, output_dir=output_dir, write=write)
    @server.tool()
    def targeted_visual_evidence(
        bundle_dir: str,
        tagger_json: str = "",
        min_score: int = 3,
        limit: int = 0,
        execute_ebook: bool = False,
        execute_crops: bool = False,
        execute_ocr: bool = False,
        execute_tiles: bool = False,
        allow_online_review: bool = False,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_targeted_visual_evidence_impl(
            bundle_dir,
            tagger_json=tagger_json or None,
            min_score=min_score,
            limit=limit,
            execute_ebook=execute_ebook,
            execute_crops=execute_crops,
            execute_ocr=execute_ocr,
            execute_tiles=execute_tiles,
            allow_online_review=allow_online_review,
            write=write,
        )

    @server.tool()
    def run_temporal_tag_delta(
        bundle_dir: str,
        input_json: str = "",
        execute_tagger: bool = False,
        source_root: str = "",
        checkpoint_path: str = "",
        device: str = "cuda",
        prefer_language: str = "zh",
        indexes: str = "",
        frame_count: int = 8,
        limit: int = 16,
        min_frames: int = 5,
        resume: str = "",
        write: bool = True,
    ) -> dict[str, Any]:
        return run_temporal_tag_delta_impl(
            bundle_dir,
            input_json=input_json or None,
            execute_tagger=execute_tagger,
            source_root=source_root or None,
            checkpoint_path=checkpoint_path or None,
            device=device,
            prefer_language=prefer_language,
            indexes=indexes or None,
            frame_count=frame_count,
            limit=limit,
            min_frames=min_frames,
            resume=resume or None,
            write=write,
        )

    @server.tool()
    def temporal_tag_restore(
        bundle_dir: str,
        plan_json: str,
        execute: bool = False,
        confirm_run_id: str = "",
    ) -> dict[str, Any]:
        return apply_temporal_tag_delta_restore_impl(
            bundle_dir,
            plan_json,
            execute=execute,
            confirm_run_id=confirm_run_id,
        )

    @server.tool()
    def vision_review_triage_tool(
        bundle_dir: str,
        mode: str = "fast",
        tagger_json: str | None = None,
        semantic_limit: int | None = None,
        temporal_limit: int | None = None,
        visual_structure_limit: int | None = None,
        min_score: int = 3,
        write: bool = True,
    ) -> dict[str, Any]:
        return vision_review_triage_impl(
            bundle_dir,
            mode=mode,
            tagger_json=tagger_json,
            semantic_limit=semantic_limit,
            temporal_limit=temporal_limit,
            visual_structure_limit=visual_structure_limit,
            min_score=min_score,
            write=write,
        )

    @server.tool()
    def vision_review_triage(
        bundle_dir: str,
        mode: str = "fast",
        tagger_json: str | None = None,
        semantic_limit: int | None = None,
        temporal_limit: int | None = None,
        visual_structure_limit: int | None = None,
        min_score: int = 3,
        write: bool = True,
    ) -> dict[str, Any]:
        return vision_review_triage_impl(
            bundle_dir,
            mode=mode,
            tagger_json=tagger_json,
            semantic_limit=semantic_limit,
            temporal_limit=temporal_limit,
            visual_structure_limit=visual_structure_limit,
            min_score=min_score,
            write=write,
        )

    @server.tool()
    def plan_supplemental_frame_sampling_tool(
        bundle_dir: str,
        triage_json: str = "",
        max_items: int = 0,
        max_frames_per_item: int = 4,
        include_temporal: bool = True,
        include_visual_structure: bool = True,
        include_semantic: bool = True,
        write: bool = True,
    ) -> dict[str, Any]:
        return plan_supplemental_frame_sampling_impl(
            bundle_dir,
            triage_json=triage_json or None,
            max_items=max_items,
            max_frames_per_item=max_frames_per_item,
            include_temporal=include_temporal,
            include_visual_structure=include_visual_structure,
            include_semantic=include_semantic,
            write=write,
        )

    @server.tool()
    def plan_supplemental_frame_sampling(
        bundle_dir: str,
        triage_json: str = "",
        max_items: int = 0,
        max_frames_per_item: int = 4,
        include_temporal: bool = True,
        include_visual_structure: bool = True,
        include_semantic: bool = True,
        write: bool = True,
    ) -> dict[str, Any]:
        return plan_supplemental_frame_sampling_impl(
            bundle_dir,
            triage_json=triage_json or None,
            max_items=max_items,
            max_frames_per_item=max_frames_per_item,
            include_temporal=include_temporal,
            include_visual_structure=include_visual_structure,
            include_semantic=include_semantic,
            write=write,
        )

    @server.tool()
    def vision_review_queue_tool(
        bundle_dir: str,
        min_score: int = 10,
        batch_size: int = 10,
        max_items: int = 0,
        provider: str = "volcengine_coding_plan",
        env_file: str = str(provider_env_file()),
        refresh_triage: bool = False,
        write: bool = True,
    ) -> dict[str, Any]:
        return vision_review_queue_impl(bundle_dir, min_score=min_score, batch_size=batch_size, max_items=max_items, provider=provider, env_file=env_file, refresh_triage=refresh_triage, write=write)

    @server.tool()
    def vision_review_queue(
        bundle_dir: str,
        min_score: int = 10,
        batch_size: int = 10,
        max_items: int = 0,
        provider: str = "volcengine_coding_plan",
        env_file: str = str(provider_env_file()),
        refresh_triage: bool = False,
        write: bool = True,
    ) -> dict[str, Any]:
        return vision_review_queue_impl(bundle_dir, min_score=min_score, batch_size=batch_size, max_items=max_items, provider=provider, env_file=env_file, refresh_triage=refresh_triage, write=write)

    @server.tool()
    @server.tool()
    def run_artifact_registry_tool(bundle_dir: str, write: bool = True) -> dict[str, Any]:
        return build_run_artifact_registry_impl(bundle_dir, write=write)

    @server.tool()
    def run_artifact_registry(bundle_dir: str, write: bool = True) -> dict[str, Any]:
        return build_run_artifact_registry_impl(bundle_dir, write=write)
    @server.tool()
    def multimodal_sample_review_tool(
        bundle_dir: str,
        comparison_json: str = "",
        sample_size: int = 30,
        include_missing: bool = True,
        media_path: str = "",
        potplayer_path: str = "",
        write: bool = True,
    ) -> dict[str, Any]:
        return multimodal_sample_review_impl(bundle_dir, comparison_json=comparison_json, sample_size=sample_size, include_missing=include_missing, media_path=media_path, potplayer_path=potplayer_path, write=write)

    @server.tool()
    def multimodal_sample_review(
        bundle_dir: str,
        comparison_json: str = "",
        sample_size: int = 30,
        include_missing: bool = True,
        media_path: str = "",
        potplayer_path: str = "",
        write: bool = True,
    ) -> dict[str, Any]:
        return multimodal_sample_review_impl(bundle_dir, comparison_json=comparison_json, sample_size=sample_size, include_missing=include_missing, media_path=media_path, potplayer_path=potplayer_path, write=write)

    @server.tool()
    def validate_multimodal_sample_notes_tool(
        bundle_dir: str,
        notes_json: str = "",
        min_reviewed: int = 10,
        write: bool = True,
    ) -> dict[str, Any]:
        return validate_multimodal_sample_notes_impl(bundle_dir, notes_json=notes_json, min_reviewed=min_reviewed, write=write)

    @server.tool()
    def validate_multimodal_sample_notes(
        bundle_dir: str,
        notes_json: str = "",
        min_reviewed: int = 10,
        write: bool = True,
    ) -> dict[str, Any]:
        return validate_multimodal_sample_notes_impl(bundle_dir, notes_json=notes_json, min_reviewed=min_reviewed, write=write)

    def test_vision_provider_tool(provider_config: dict[str, Any] | None = None, image_paths: list[str] | None = None) -> dict[str, Any]:
        return test_vision_provider_impl(provider_config, image_paths=image_paths)

    @server.tool()
    def test_vision_provider(provider_config: dict[str, Any] | None = None, image_paths: list[str] | None = None) -> dict[str, Any]:
        return test_vision_provider_impl(provider_config, image_paths=image_paths)

    @server.tool()
    def vision_provider_smoke_tool(
        provider_config: dict[str, Any] | None = None,
        provider: str = "",
        model: str = "",
        base_url: str = "",
        timeout_seconds: int | None = None,
        bundle_dir: str = "",
        single_image: str = "",
        multi_image_dir: str = "",
        output_dir: str = "",
        image_probe_max_edge: int = 0,
        image_probe_jpeg_quality: int = 70,
        max_images: int = 8,
        write: bool = True,
    ) -> dict[str, Any]:
        return vision_provider_smoke_impl(
            provider_config=provider_config,
            provider=provider,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            bundle_dir=bundle_dir,
            single_image=single_image,
            multi_image_dir=multi_image_dir,
            output_dir=output_dir,
            image_probe_max_edge=image_probe_max_edge,
            image_probe_jpeg_quality=image_probe_jpeg_quality,
            max_images=max_images,
            write=write,
        )

    @server.tool()
    def vision_provider_smoke(
        provider_config: dict[str, Any] | None = None,
        provider: str = "",
        model: str = "",
        base_url: str = "",
        timeout_seconds: int | None = None,
        bundle_dir: str = "",
        single_image: str = "",
        multi_image_dir: str = "",
        output_dir: str = "",
        image_probe_max_edge: int = 0,
        image_probe_jpeg_quality: int = 70,
        max_images: int = 8,
        write: bool = True,
    ) -> dict[str, Any]:
        return vision_provider_smoke_impl(
            provider_config=provider_config,
            provider=provider,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            bundle_dir=bundle_dir,
            single_image=single_image,
            multi_image_dir=multi_image_dir,
            output_dir=output_dir,
            image_probe_max_edge=image_probe_max_edge,
            image_probe_jpeg_quality=image_probe_jpeg_quality,
            max_images=max_images,
            write=write,
        )

    @server.tool()
    def vision_provider_matrix_tool(
        providers: list[str] | None = None,
        bundle_dir: str | None = None,
        output_dir: str | None = None,
        timeout_seconds: int | None = None,
        single_image: str | None = None,
        multi_image_dir: str | None = None,
        image_probe_max_edge: int = 0,
        image_probe_jpeg_quality: int = 70,
        max_images: int = 8,
        preferred_provider: str = "",
        write: bool = True,
    ) -> dict[str, Any]:
        return vision_provider_matrix_impl(
            providers=providers,
            bundle_dir=bundle_dir,
            output_dir=output_dir,
            timeout_seconds=timeout_seconds,
            single_image=single_image,
            multi_image_dir=multi_image_dir,
            image_probe_max_edge=image_probe_max_edge,
            image_probe_jpeg_quality=image_probe_jpeg_quality,
            max_images=max_images,
            preferred_provider=preferred_provider,
            write=write,
        )

    @server.tool()
    def vision_provider_matrix(
        providers: list[str] | None = None,
        bundle_dir: str | None = None,
        output_dir: str | None = None,
        timeout_seconds: int | None = None,
        single_image: str | None = None,
        multi_image_dir: str | None = None,
        image_probe_max_edge: int = 0,
        image_probe_jpeg_quality: int = 70,
        max_images: int = 8,
        preferred_provider: str = "",
        write: bool = True,
    ) -> dict[str, Any]:
        return vision_provider_matrix_impl(
            providers=providers,
            bundle_dir=bundle_dir,
            output_dir=output_dir,
            timeout_seconds=timeout_seconds,
            single_image=single_image,
            multi_image_dir=multi_image_dir,
            image_probe_max_edge=image_probe_max_edge,
            image_probe_jpeg_quality=image_probe_jpeg_quality,
            max_images=max_images,
            preferred_provider=preferred_provider,
            write=write,
        )

    @server.tool()
    def vision_analysis_run_log_tool(bundle_dir: str) -> dict[str, Any]:
        return vision_analysis_run_log_impl(bundle_dir)

    @server.tool()
    def prepare_codex_vision_session(
        bundle_dir: str,
        kind: str,
        indexes: list[int] | None = None,
        limit: int | None = None,
        max_frames: int = 19,
        model_requested: str = "luna",
        quality: str = "low",
        executor: str = "codex_session",
        write: bool = True,
    ) -> dict[str, Any]:
        return prepare_codex_vision_session_impl(
            bundle_dir,
            kind=kind,
            indexes=indexes,
            limit=limit,
            max_frames=max_frames,
            model_requested=model_requested,
            quality=quality,
            executor=executor,
            write=write,
        )

    @server.tool()
    def prepare_summary_claim_review(bundle_dir: str, write: bool = True) -> dict[str, Any]:
        """Prepare exactly one offline task; inspect its schema, never infer review approval."""
        return prepare_summary_claim_review_impl(bundle_dir, write=write)

    @server.tool()
    def import_summary_claim_review(bundle_dir: str, input_json: str) -> dict[str, Any]:
        """Import a Bundle-local result against the immutable prepared task, with restore journal."""
        return import_summary_claim_review_impl(bundle_dir, input_json)

    @server.tool()
    def summary_claim_review_status(bundle_dir: str) -> dict[str, Any]:
        """Read-only current task, counts and gaps. Never runs a Provider or creates a task."""
        return summary_claim_review_status_impl(bundle_dir)

    @server.tool()
    def import_codex_vision_results(
        bundle_dir: str,
        input_json: str,
        write: bool = True,
    ) -> dict[str, Any]:
        return import_codex_vision_results_impl(bundle_dir, input_json=input_json, write=write)

    @server.tool()
    def vision_analysis_run_log(bundle_dir: str) -> dict[str, Any]:
        return vision_analysis_run_log_impl(bundle_dir)

    @server.tool()
    def vision_analysis_restore_plan_tool(bundle_dir: str, run_id: str = "", write: bool = True) -> dict[str, Any]:
        return vision_analysis_restore_plan_impl(bundle_dir, run_id=run_id, write=write)

    @server.tool()
    def vision_analysis_restore_plan(bundle_dir: str, run_id: str = "", write: bool = True) -> dict[str, Any]:
        return vision_analysis_restore_plan_impl(bundle_dir, run_id=run_id, write=write)

    @server.tool()
    def vision_analysis_apply_restore_tool(
        bundle_dir: str,
        plan_json: str | None = None,
        execute: bool = False,
        confirm_run_id: str = "",
    ) -> dict[str, Any]:
        return vision_analysis_apply_restore_impl(bundle_dir, plan_json=plan_json, execute=execute, confirm_run_id=confirm_run_id)

    @server.tool()
    def vision_analysis_apply_restore(
        bundle_dir: str,
        plan_json: str | None = None,
        execute: bool = False,
        confirm_run_id: str = "",
    ) -> dict[str, Any]:
        return vision_analysis_apply_restore_impl(bundle_dir, plan_json=plan_json, execute=execute, confirm_run_id=confirm_run_id)

    @server.tool()
    def vision_acceptance_plan_tool(
        bundle_dir: str,
        provider_config: dict[str, Any] | None = None,
        semantic_limit: int | None = None,
        temporal_limit: int | None = None,
        frame_count: int | None = None,
        write: bool = True,
    ) -> dict[str, Any]:
        return vision_acceptance_plan_impl(
            bundle_dir,
            provider_config=provider_config,
            semantic_limit=semantic_limit,
            temporal_limit=temporal_limit,
            frame_count=frame_count,
            write=write,
        )

    @server.tool()
    def vision_acceptance_plan(
        bundle_dir: str,
        provider_config: dict[str, Any] | None = None,
        semantic_limit: int | None = None,
        temporal_limit: int | None = None,
        frame_count: int | None = None,
        write: bool = True,
    ) -> dict[str, Any]:
        return vision_acceptance_plan_impl(
            bundle_dir,
            provider_config=provider_config,
            semantic_limit=semantic_limit,
            temporal_limit=temporal_limit,
            frame_count=frame_count,
            write=write,
        )

    @server.tool()
    def visual_ab_benchmark_plan(bundle_dir: str, limit: int = 10, min_score: int = 4, write: bool = True) -> dict[str, Any]:
        return build_visual_ab_benchmark_plan_impl(bundle_dir, limit=limit, min_score=min_score, write=write)

    @server.tool()
    def vision_export_consent_create(
        bundle_dir: str,
        provider_config: dict[str, Any] | None = None,
        semantic_indexes: list[int] | None = None,
        temporal_indexes: list[int] | None = None,
        max_calls: int | None = None,
        expires_hours: float = 24.0,
        image_max_edge: int = 512,
        image_jpeg_quality: int = 55,
        purpose: str = "targeted multimodal review",
        confirm_data_export: bool = False,
        output_path: str | None = None,
        write: bool = True,
    ) -> dict[str, Any]:
        return create_vision_export_consent_impl(
            bundle_dir, provider_config=provider_config, semantic_indexes=semantic_indexes, temporal_indexes=temporal_indexes,
            max_calls=max_calls, expires_hours=expires_hours, image_max_edge=image_max_edge, image_jpeg_quality=image_jpeg_quality,
            purpose=purpose, confirm_data_export=confirm_data_export, output_path=output_path, write=write,
        )

    @server.tool()
    def vision_export_consent_status(
        bundle_dir: str,
        consent_path: str | None = None,
        provider_config: dict[str, Any] | None = None,
        semantic_indexes: list[int] | None = None,
        temporal_indexes: list[int] | None = None,
        expected_calls: int | None = None,
        image_max_edge: int = 512,
        image_jpeg_quality: int = 55,
    ) -> dict[str, Any]:
        return vision_export_consent_status_impl(
            bundle_dir, consent_path=consent_path, provider_config=provider_config, semantic_indexes=semantic_indexes,
            temporal_indexes=temporal_indexes, expected_calls=expected_calls, image_max_edge=image_max_edge, image_jpeg_quality=image_jpeg_quality,
        )

    @server.tool()
    def vision_export_consent_revoke(bundle_dir: str, consent_path: str | None = None, write: bool = True) -> dict[str, Any]:
        return revoke_vision_export_consent_impl(bundle_dir, consent_path=consent_path, write=write)

    @server.tool()
    def vision_execution_preflight_tool(
        bundle_dir: str,
        provider_config: dict[str, Any] | None = None,
        semantic_limit: int | None = None,
        temporal_limit: int | None = None,
        frame_count: int | None = None,
        semantic_indexes: list[int] | None = None,
        temporal_indexes: list[int] | None = None,
        include_semantic: bool = True,
        include_temporal: bool = True,
        check_provider: bool = False,
        write: bool = True,
    ) -> dict[str, Any]:
        return vision_execution_preflight_impl(
            bundle_dir,
            provider_config=provider_config,
            semantic_limit=semantic_limit,
            temporal_limit=temporal_limit,
            frame_count=frame_count,
            include_semantic=include_semantic,
            include_temporal=include_temporal,
            semantic_indexes=semantic_indexes,
            temporal_indexes=temporal_indexes,
            check_provider=check_provider,
            write=write,
        )

    @server.tool()
    def vision_execution_preflight(
        bundle_dir: str,
        provider_config: dict[str, Any] | None = None,
        semantic_limit: int | None = None,
        temporal_limit: int | None = None,
        frame_count: int | None = None,
        semantic_indexes: list[int] | None = None,
        temporal_indexes: list[int] | None = None,
        include_semantic: bool = True,
        include_temporal: bool = True,
        check_provider: bool = False,
        write: bool = True,
    ) -> dict[str, Any]:
        return vision_execution_preflight_impl(
            bundle_dir,
            provider_config=provider_config,
            semantic_limit=semantic_limit,
            temporal_limit=temporal_limit,
            frame_count=frame_count,
            include_semantic=include_semantic,
            include_temporal=include_temporal,
            semantic_indexes=semantic_indexes,
            temporal_indexes=temporal_indexes,
            check_provider=check_provider,
            write=write,
        )

    @server.tool()
    def volcengine_model_task_matrix_tool(
        execute: bool = False,
        output_dir: str = "",
        models: str = "",
        tasks: str = "",
        timeout_seconds: int = 120,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_volcengine_model_task_matrix_impl(
            execute=execute,
            output_dir=output_dir or None,
            models=models,
            tasks=tasks,
            timeout_seconds=timeout_seconds,
            write=write,
        )

    @server.tool()
    def volcengine_model_task_matrix(
        execute: bool = False,
        output_dir: str = "",
        models: str = "",
        tasks: str = "",
        timeout_seconds: int = 120,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_volcengine_model_task_matrix_impl(
            execute=execute,
            output_dir=output_dir or None,
            models=models,
            tasks=tasks,
            timeout_seconds=timeout_seconds,
            write=write,
        )

    @server.tool()
    def volcengine_model_routing_tool(route: str = "tool_terms", output_dir: str = "", write: bool = True) -> dict[str, Any]:
        return volcengine_model_routing_impl(route=route, output_dir=output_dir or None, write=write)

    @server.tool()
    def volcengine_model_routing(route: str = "tool_terms", output_dir: str = "", write: bool = True) -> dict[str, Any]:
        return volcengine_model_routing_impl(route=route, output_dir=output_dir or None, write=write)
    @server.tool()
    def text_llm_provider_smoke_tool(provider_config: dict[str, Any] | None = None, execute: bool = False, prompt: str = "Reply with exactly: ok") -> dict[str, Any]:
        return text_llm_provider_smoke_impl(provider_config, execute=execute, prompt=prompt)

    @server.tool()
    def text_llm_provider_smoke(provider_config: dict[str, Any] | None = None, execute: bool = False, prompt: str = "Reply with exactly: ok") -> dict[str, Any]:
        return text_llm_provider_smoke_impl(provider_config, execute=execute, prompt=prompt)

    @server.tool()
    def online_model_api_tool(
        model_type: str,
        provider_config: dict[str, Any] | None = None,
        prompt: str = "",
        input_text: str = "",
        image_paths: list[str] | None = None,
        audio_path: str = "",
        execute: bool = False,
        output_dir: str = "",
        write: bool = True,
    ) -> dict[str, Any]:
        return online_model_api_call_impl(
            model_type,
            provider_config=provider_config,
            prompt=prompt,
            input_text=input_text,
            image_paths=image_paths or [],
            audio_path=audio_path,
            execute=execute,
            output_dir=output_dir or None,
            write=write,
        )

    @server.tool()
    def online_model_api(
        model_type: str,
        provider_config: dict[str, Any] | None = None,
        prompt: str = "",
        input_text: str = "",
        image_paths: list[str] | None = None,
        audio_path: str = "",
        execute: bool = False,
        output_dir: str = "",
        write: bool = True,
    ) -> dict[str, Any]:
        return online_model_api_call_impl(
            model_type,
            provider_config=provider_config,
            prompt=prompt,
            input_text=input_text,
            image_paths=image_paths or [],
            audio_path=audio_path,
            execute=execute,
            output_dir=output_dir or None,
            write=write,
        )

    @server.tool()
    def online_model_api_matrix_tool(provider_config: dict[str, Any] | None = None, output_dir: str = "", write: bool = True) -> dict[str, Any]:
        return online_model_api_matrix_impl(provider_config=provider_config, output_dir=output_dir or None, write=write)

    @server.tool()
    def online_model_api_matrix(provider_config: dict[str, Any] | None = None, output_dir: str = "", write: bool = True) -> dict[str, Any]:
        return online_model_api_matrix_impl(provider_config=provider_config, output_dir=output_dir or None, write=write)

    @server.tool()
    def model_task_coverage_audit_tool(output_dir: str = "", write: bool = True) -> dict[str, Any]:
        return model_task_coverage_audit_impl(output_dir=output_dir or None, write=write)

    @server.tool()
    def model_task_coverage_audit(output_dir: str = "", write: bool = True) -> dict[str, Any]:
        return model_task_coverage_audit_impl(output_dir=output_dir or None, write=write)

    @server.tool()
    def run_term_arbitration_model_tool(
        bundle_dir: str,
        provider_config: dict[str, Any] | None = None,
        execute: bool = False,
        max_terms: int = 60,
        min_confidence: float = 0.88,
        max_tokens: int = 5000,
        temperature: float = 0,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_term_arbitration_model_impl(bundle_dir, provider_config=provider_config, execute=execute, max_terms=max_terms, min_confidence=min_confidence, max_tokens=max_tokens, temperature=temperature, write=write)

    @server.tool()
    def run_term_arbitration_model(
        bundle_dir: str,
        provider_config: dict[str, Any] | None = None,
        execute: bool = False,
        max_terms: int = 60,
        min_confidence: float = 0.88,
        max_tokens: int = 5000,
        temperature: float = 0,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_term_arbitration_model_impl(bundle_dir, provider_config=provider_config, execute=execute, max_terms=max_terms, min_confidence=min_confidence, max_tokens=max_tokens, temperature=temperature, write=write)

    @server.tool()
    def run_bilinote_mind_map_model_tool(
        bundle_dir: str,
        provider_config: dict[str, Any] | None = None,
        execute: bool = False,
        title: str = "",
        max_chars: int = 5000,
        limit: int = 0,
        max_tokens: int = 4000,
        temperature: float = 0,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_bilinote_mind_map_model_impl(bundle_dir, provider_config=provider_config, execute=execute, title=title, max_chars=max_chars, limit=limit, max_tokens=max_tokens, temperature=temperature, write=write)

    @server.tool()
    def run_bilinote_mind_map_model(
        bundle_dir: str,
        provider_config: dict[str, Any] | None = None,
        execute: bool = False,
        title: str = "",
        max_chars: int = 5000,
        limit: int = 0,
        max_tokens: int = 4000,
        temperature: float = 0,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_bilinote_mind_map_model_impl(bundle_dir, provider_config=provider_config, execute=execute, title=title, max_chars=max_chars, limit=limit, max_tokens=max_tokens, temperature=temperature, write=write)
    @server.tool()
    def transcript_correction_pack_tool(
        bundle_dir: str,
        input_json: str = "",
        provider_config: dict[str, Any] | None = None,
        execute: bool = False,
        max_segments: int = 0,
        max_chunk_chars: int = 5000,
        write: bool = True,
    ) -> dict[str, Any]:
        return build_transcript_correction_pack_impl(
            bundle_dir,
            input_json=input_json or None,
            provider_config=provider_config,
            execute=execute,
            max_segments=max_segments,
            max_chunk_chars=max_chunk_chars,
            write=write,
        )

    @server.tool()
    def transcript_correction_pack(
        bundle_dir: str,
        input_json: str = "",
        provider_config: dict[str, Any] | None = None,
        execute: bool = False,
        max_segments: int = 0,
        max_chunk_chars: int = 5000,
        write: bool = True,
    ) -> dict[str, Any]:
        return build_transcript_correction_pack_impl(
            bundle_dir,
            input_json=input_json or None,
            provider_config=provider_config,
            execute=execute,
            max_segments=max_segments,
            max_chunk_chars=max_chunk_chars,
            write=write,
        )
    @server.tool()
    def bilinote_mind_map_prompt_pack_tool(
        bundle_dir: str,
        title: str = "",
        max_chars: int = 5000,
        write: bool = True,
    ) -> dict[str, Any]:
        return build_bundle_mind_map_prompt_pack_impl(bundle_dir, title=title, max_chars=max_chars, write=write)

    @server.tool()
    def bilinote_mind_map_prompt_pack(
        bundle_dir: str,
        title: str = "",
        max_chars: int = 5000,
        write: bool = True,
    ) -> dict[str, Any]:
        return build_bundle_mind_map_prompt_pack_impl(bundle_dir, title=title, max_chars=max_chars, write=write)

    @server.tool()
    def transcript_source_arbitration_tool(
        bundle_dir: str,
        platform_subtitle: str = "",
        subtitle: str = "",
        asr_json: str = "",
        glossary_json: str = "",
        min_confidence: float = 0.72,
        promote: bool = True,
        write: bool = True,
    ) -> dict[str, Any]:
        return arbitrate_transcript_sources_impl(
            bundle_dir,
            platform_subtitle=platform_subtitle or None,
            subtitle=subtitle or None,
            asr_json=asr_json or None,
            glossary_json=glossary_json or None,
            min_confidence=min_confidence,
            promote=promote,
            write=write,
        )

    @server.tool()
    def transcript_source_arbitration(
        bundle_dir: str,
        platform_subtitle: str = "",
        subtitle: str = "",
        asr_json: str = "",
        glossary_json: str = "",
        min_confidence: float = 0.72,
        promote: bool = True,
        write: bool = True,
    ) -> dict[str, Any]:
        return arbitrate_transcript_sources_impl(
            bundle_dir,
            platform_subtitle=platform_subtitle or None,
            subtitle=subtitle or None,
            asr_json=asr_json or None,
            glossary_json=glossary_json or None,
            min_confidence=min_confidence,
            promote=promote,
            write=write,
        )

    @server.tool()
    def video_moment_index_tool(
        bundle_dir: str,
        query: str = "",
        target_window_seconds: float = 300.0,
        max_chunk_chars: int = 3600,
        top_k: int = 8,
        write: bool = True,
    ) -> dict[str, Any]:
        return build_video_moment_index_impl(
            bundle_dir,
            query=query,
            target_window_seconds=target_window_seconds,
            max_chunk_chars=max_chunk_chars,
            top_k=top_k,
            write=write,
        )

    @server.tool()
    def video_moment_index(
        bundle_dir: str,
        query: str = "",
        target_window_seconds: float = 300.0,
        max_chunk_chars: int = 3600,
        top_k: int = 8,
        write: bool = True,
    ) -> dict[str, Any]:
        return build_video_moment_index_impl(
            bundle_dir,
            query=query,
            target_window_seconds=target_window_seconds,
            max_chunk_chars=max_chunk_chars,
            top_k=top_k,
            write=write,
        )

    @server.tool()
    def long_video_memory_pack_tool(
        bundle_dir: str,
        target_window_seconds: float = 300.0,
        max_chunk_chars: int = 3600,
        long_group_size: int = 6,
        write: bool = True,
    ) -> dict[str, Any]:
        return build_long_video_memory_pack_impl(
            bundle_dir,
            target_window_seconds=target_window_seconds,
            max_chunk_chars=max_chunk_chars,
            long_group_size=long_group_size,
            write=write,
        )

    @server.tool()
    def long_video_memory_pack(
        bundle_dir: str,
        target_window_seconds: float = 300.0,
        max_chunk_chars: int = 3600,
        long_group_size: int = 6,
        write: bool = True,
    ) -> dict[str, Any]:
        return build_long_video_memory_pack_impl(
            bundle_dir,
            target_window_seconds=target_window_seconds,
            max_chunk_chars=max_chunk_chars,
            long_group_size=long_group_size,
            write=write,
        )

    @server.tool()
    def video_rag_pack_tool(
        bundle_dir: str,
        query: str = "",
        target_window_seconds: float = 300.0,
        max_chunk_chars: int = 3600,
        top_k: int = 8,
        write: bool = True,
    ) -> dict[str, Any]:
        return build_video_rag_pack_impl(
            bundle_dir,
            query=query,
            target_window_seconds=target_window_seconds,
            max_chunk_chars=max_chunk_chars,
            top_k=top_k,
            write=write,
        )

    @server.tool()
    def video_rag_pack(
        bundle_dir: str,
        query: str = "",
        target_window_seconds: float = 300.0,
        max_chunk_chars: int = 3600,
        top_k: int = 8,
        write: bool = True,
    ) -> dict[str, Any]:
        return build_video_rag_pack_impl(
            bundle_dir,
            query=query,
            target_window_seconds=target_window_seconds,
            max_chunk_chars=max_chunk_chars,
            top_k=top_k,
            write=write,
        )

    @server.tool()
    def video_rag_search_tool(
        bundle_dir: str,
        query: str,
        top_k: int = 8,
        ensure_pack: bool = True,
        retrieval_backend: str = "keyword",
        write: bool = True,
    ) -> dict[str, Any]:
        return search_video_rag_impl(bundle_dir, query=query, top_k=top_k, ensure_pack=ensure_pack, retrieval_backend=retrieval_backend, write=write)

    @server.tool()
    def video_rag_search(
        bundle_dir: str,
        query: str,
        top_k: int = 8,
        ensure_pack: bool = True,
        retrieval_backend: str = "keyword",
        write: bool = True,
    ) -> dict[str, Any]:
        return search_video_rag_impl(bundle_dir, query=query, top_k=top_k, ensure_pack=ensure_pack, retrieval_backend=retrieval_backend, write=write)

    @server.tool()
    def script_clip_candidate_pack_tool(
        bundle_dir: str,
        request_json: str,
        top_k: int = 8,
        retrieval_backend: str = "keyword",
        context_seconds: float = 3.0,
        write: bool = True,
    ) -> dict[str, Any]:
        return build_script_clip_candidate_pack_impl(
            bundle_dir,
            request_json,
            top_k=top_k,
            retrieval_backend=retrieval_backend,
            context_seconds=context_seconds,
            write=write,
        )

    @server.tool()
    def script_clip_candidate_pack(
        bundle_dir: str,
        request_json: str,
        top_k: int = 8,
        retrieval_backend: str = "keyword",
        context_seconds: float = 3.0,
        write: bool = True,
    ) -> dict[str, Any]:
        return build_script_clip_candidate_pack_impl(
            bundle_dir,
            request_json,
            top_k=top_k,
            retrieval_backend=retrieval_backend,
            context_seconds=context_seconds,
            write=write,
        )

    @server.tool()
    def script_clip_alignment_check_tool(
        bundle_dir: str,
        review_notes_json: str,
        fine_cut_plan_json: str,
        candidate_pack_json: str = "",
        write: bool = True,
    ) -> dict[str, Any]:
        return check_script_clip_alignment_impl(
            bundle_dir,
            review_notes_json,
            fine_cut_plan_json,
            candidate_pack_json=candidate_pack_json or None,
            write=write,
        )

    @server.tool()
    def script_clip_alignment_check(
        bundle_dir: str,
        review_notes_json: str,
        fine_cut_plan_json: str,
        candidate_pack_json: str = "",
        write: bool = True,
    ) -> dict[str, Any]:
        return check_script_clip_alignment_impl(
            bundle_dir,
            review_notes_json,
            fine_cut_plan_json,
            candidate_pack_json=candidate_pack_json or None,
            write=write,
        )

    @server.tool()
    def content_clip_candidate_pack(
        bundle_dir: str,
        request_json: str,
        top_k: int = 8,
        retrieval_backend: str = "keyword",
        context_seconds: float = 3.0,
        write: bool = True,
    ) -> dict[str, Any]:
        return build_content_clip_candidate_pack_impl(
            bundle_dir,
            request_json,
            top_k=top_k,
            retrieval_backend=retrieval_backend,
            context_seconds=context_seconds,
            write=write,
        )

    @server.tool()
    def content_clip_alignment_check(
        bundle_dir: str,
        review_notes_json: str,
        fine_cut_plan_json: str,
        candidate_pack_json: str = "",
        write: bool = True,
    ) -> dict[str, Any]:
        return check_content_clip_alignment_impl(
            bundle_dir,
            review_notes_json,
            fine_cut_plan_json,
            candidate_pack_json=candidate_pack_json or None,
            write=write,
        )

    @server.tool()
    def video_evidence_query_plan(
        bundle_dir: str,
        query: str,
        coarse_top_k: int = 12,
        fine_top_k: int = 4,
        write: bool = True,
    ) -> dict[str, Any]:
        return build_video_evidence_query_plan_impl(
            bundle_dir,
            query=query,
            coarse_top_k=coarse_top_k,
            fine_top_k=fine_top_k,
            write=write,
        )

    @server.tool()
    def apply_video_evidence_confirmation(
        bundle_dir: str,
        decisions_json: str,
        plan_json: str = "",
        write: bool = True,
    ) -> dict[str, Any]:
        return apply_video_evidence_confirmation_impl(
            bundle_dir,
            decisions_json=decisions_json,
            plan_json=plan_json or None,
            write=write,
        )

    @server.tool()
    def external_capability_pack_tool(bundle_dir: str, query: str = "", write: bool = True) -> dict[str, Any]:
        return build_external_capability_pack_impl(bundle_dir, query=query, write=write)

    @server.tool()
    def external_capability_pack(bundle_dir: str, query: str = "", write: bool = True) -> dict[str, Any]:
        return build_external_capability_pack_impl(bundle_dir, query=query, write=write)
    @server.tool()
    def local_vlm_adapter_plan_tool(output_dir: str = "", write: bool = False) -> dict[str, Any]:
        return local_vlm_adapter_plan_impl(output_dir, write=write)

    @server.tool()
    def local_vlm_adapter_plan(output_dir: str = "", write: bool = False) -> dict[str, Any]:
        return local_vlm_adapter_plan_impl(output_dir, write=write)

    @server.tool()
    def local_vlm_serving_smoke_tool(
        provider: str = "local_qwen_vl",
        bundle_dir: str = "",
        output_dir: str = "",
        single_image: str = "",
        multi_image_dir: str = "",
        execute: bool = False,
        timeout_seconds: int = 30,
        max_images: int = 3,
        image_probe_max_edge: int = 512,
        image_probe_jpeg_quality: int = 70,
        frame_group_count: int = 8,
        write: bool = True,
    ) -> dict[str, Any]:
        return local_vlm_serving_smoke_impl(provider=provider, bundle_dir=bundle_dir, output_dir=output_dir, single_image=single_image, multi_image_dir=multi_image_dir, execute=execute, timeout_seconds=timeout_seconds, max_images=max_images, image_probe_max_edge=image_probe_max_edge, image_probe_jpeg_quality=image_probe_jpeg_quality, frame_group_count=frame_group_count, write=write)

    @server.tool()
    def local_vlm_serving_smoke(
        provider: str = "local_qwen_vl",
        bundle_dir: str = "",
        output_dir: str = "",
        single_image: str = "",
        multi_image_dir: str = "",
        execute: bool = False,
        timeout_seconds: int = 30,
        max_images: int = 3,
        image_probe_max_edge: int = 512,
        image_probe_jpeg_quality: int = 70,
        frame_group_count: int = 8,
        write: bool = True,
    ) -> dict[str, Any]:
        return local_vlm_serving_smoke_impl(provider=provider, bundle_dir=bundle_dir, output_dir=output_dir, single_image=single_image, multi_image_dir=multi_image_dir, execute=execute, timeout_seconds=timeout_seconds, max_images=max_images, image_probe_max_edge=image_probe_max_edge, image_probe_jpeg_quality=image_probe_jpeg_quality, frame_group_count=frame_group_count, write=write)
    @server.tool()
    def audit_knowledge_coverage_tool(bundle_dir: str, write: bool = True) -> dict[str, Any]:
        return audit_knowledge_coverage_impl(bundle_dir, write=write)

    @server.tool()
    def audit_knowledge_coverage(bundle_dir: str, write: bool = True) -> dict[str, Any]:
        return audit_knowledge_coverage_impl(bundle_dir, write=write)

    @server.tool()
    def export_knowledge_note_tool(
        bundle_dir: str,
        output_dir: str = "",
        title: str = "",
        include_timeline: bool = True,
        include_full_transcript: bool = True,
        write: bool = True,
    ) -> dict[str, Any]:
        return export_knowledge_note_impl(
            bundle_dir,
            output_dir=output_dir or None,
            title=title,
            include_timeline=include_timeline,
            include_full_transcript=include_full_transcript,
            write=write,
        )

    @server.tool()
    def export_knowledge_note(
        bundle_dir: str,
        output_dir: str = "",
        title: str = "",
        include_timeline: bool = True,
        include_full_transcript: bool = True,
        write: bool = True,
    ) -> dict[str, Any]:
        return export_knowledge_note_impl(
            bundle_dir,
            output_dir=output_dir or None,
            title=title,
            include_timeline=include_timeline,
            include_full_transcript=include_full_transcript,
            write=write,
        )

    @server.tool()
    def generate_smart_summary_with_codex_tool(bundle_dir: str, input_md: str = "", write: bool = True, handoff_id: str = "", agent_available: bool = False) -> dict[str, Any]:
        return generate_smart_summary_with_codex_impl(bundle_dir, input_md=input_md or None, write=write, handoff_id=handoff_id, agent_available=agent_available)

    @server.tool()
    def generate_smart_summary_with_codex(bundle_dir: str, input_md: str = "", write: bool = True, handoff_id: str = "", agent_available: bool = False) -> dict[str, Any]:
        return generate_smart_summary_with_codex_impl(bundle_dir, input_md=input_md or None, write=write, handoff_id=handoff_id, agent_available=agent_available)

    @server.tool()
    def prepare_smart_summary_llm_rewrite_tool(bundle_dir: str, provider: str = "codex_manual", write: bool = True) -> dict[str, Any]:
        return prepare_smart_summary_llm_rewrite_impl(bundle_dir, provider=provider, write=write)

    @server.tool()
    def prepare_smart_summary_llm_rewrite(bundle_dir: str, provider: str = "codex_manual", write: bool = True) -> dict[str, Any]:
        return prepare_smart_summary_llm_rewrite_impl(bundle_dir, provider=provider, write=write)
    @server.tool()
    def run_smart_summary_llm_rewrite_tool(bundle_dir: str, provider_config: dict[str, Any] | None = None, execute: bool = False, max_input_chars: int = 60000, temperature: float = 0, install: bool = True, write: bool = True) -> dict[str, Any]:
        return run_smart_summary_llm_rewrite_impl(bundle_dir, provider_config=provider_config, execute=execute, max_input_chars=max_input_chars, temperature=temperature, install=install, write=write)

    @server.tool()
    def run_smart_summary_llm_rewrite(bundle_dir: str, provider_config: dict[str, Any] | None = None, execute: bool = False, max_input_chars: int = 60000, temperature: float = 0, install: bool = True, write: bool = True) -> dict[str, Any]:
        return run_smart_summary_llm_rewrite_impl(bundle_dir, provider_config=provider_config, execute=execute, max_input_chars=max_input_chars, temperature=temperature, install=install, write=write)

    @server.tool()
    def build_smart_summary_input_pack_tool(bundle_dir: str, title: str = "", write: bool = True, max_visual_items: int = 80) -> dict[str, Any]:
        return build_smart_summary_input_pack_impl(bundle_dir, title=title, write=write, max_visual_items=max_visual_items)

    @server.tool()
    def build_smart_summary_input_pack(bundle_dir: str, title: str = "", write: bool = True, max_visual_items: int = 80) -> dict[str, Any]:
        return build_smart_summary_input_pack_impl(bundle_dir, title=title, write=write, max_visual_items=max_visual_items)

    @server.tool()
    def scene_detection(
        bundle_dir: str,
        media_path: str = "",
        detector: str = "adaptive",
        threshold: float | None = None,
        min_scene_len: int = 15,
        max_points: int = 300,
        source_root: str = "",
        write: bool = True,
    ) -> dict[str, Any]:
        return run_scene_detection_impl(
            bundle_dir,
            media_path=media_path or None,
            detector=detector,
            threshold=threshold,
            min_scene_len=min_scene_len,
            max_points=max_points,
            source_root=source_root or None,
            write=write,
        )

    @server.tool()
    def highlight_detection(
        bundle_dir: str,
        query: str,
        media_path: str = "",
        checkpoint_path: str = "",
        source_root: str = "",
        predictions_json: str = "",
        feature_name: str = "clip",
        device: str = "cuda",
        execute: bool = False,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_highlight_detection_impl(
            bundle_dir,
            query=query,
            media_path=media_path or None,
            checkpoint_path=checkpoint_path or None,
            source_root=source_root or None,
            predictions_json=predictions_json or None,
            feature_name=feature_name,
            device=device,
            execute=execute,
            write=write,
        )

    @server.tool()
    def video_structure(
        bundle_dir: str,
        media_path: str = "",
        title: str = "",
        run_shot_detection: bool = True,
        highlight_query: str = "找出对内容理解或后续剪辑最重要的片段",
        highlight_predictions_json: str = "",
        write: bool = True,
    ) -> dict[str, Any]:
        return build_video_structure_impl(
            bundle_dir,
            media_path=media_path or None,
            title=title,
            run_shot_detection=run_shot_detection,
            highlight_query=highlight_query,
            highlight_predictions_json=highlight_predictions_json or None,
            write=write,
        )

    @server.tool()
    def export_creator_report(
        bundle_dir: str, title: str = "", brief: dict[str, Any] | None = None,
        formats: list[str] | None = None, write: bool = True,
        view: str = "strict", video_type: str = "talking_head",
        interactive_html: bool = False, output_dir: str = "", ocr_results_json: str = "",
    ) -> dict[str, Any]:
        # Executable paths are operator-side CLI configuration, never MCP client input.
        return export_creator_report_impl(bundle_dir, title=title, brief=brief, formats=formats, write=write,
                                         view=view, video_type=video_type, interactive_html=interactive_html,
                                         output_dir=output_dir or None, ocr_results_json=ocr_results_json or None)

    @server.tool()
    def creator_report_status(bundle_dir: str, view: str = "strict", output_dir: str = "") -> dict[str, Any]:
        return creator_report_status_impl(bundle_dir, view=view, output_dir=output_dir or None)

    @server.tool()
    def shot_breakdown(
        bundle_dir: str,
        title: str = "",
        reference_analysis_json: str = "",
        write: bool = True,
    ) -> dict[str, Any]:
        return build_shot_breakdown_impl(
            bundle_dir,
            title=title,
            reference_analysis_json=reference_analysis_json or None,
            write=write,
        )

    @server.tool()
    def general_tagger_status(source_root: str = "", checkpoint_path: str = "") -> dict[str, Any]:
        return general_tagger_status_impl(
            source_root=source_root or None,
            checkpoint_path=checkpoint_path or None,
        )

    @server.tool()
    def run_general_tagger(
        bundle_dir: str,
        source_root: str = "",
        checkpoint_path: str = "",
        device: str = "cuda",
        prefer_language: str = "zh",
        limit: int = 0,
        frame_mode: str = "representative",
        execute: bool = False,
        import_annotations: bool = True,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_general_tagger_impl(
            bundle_dir,
            source_root=source_root or None,
            checkpoint_path=checkpoint_path or None,
            device=device,
            prefer_language=prefer_language,
            limit=limit,
            frame_mode=frame_mode,
            execute=execute,
            import_annotations=import_annotations,
            write=write,
        )

    @server.tool()
    def semantic_chapter_plan(bundle_dir: str, title: str = "", chapter_mode: str = "semantic", write: bool = True) -> dict[str, Any]:
        return build_semantic_chapter_plan_impl(bundle_dir, title=title, chapter_mode=chapter_mode, write=write)

    @server.tool()
    def build_smart_summary_chapters_tool(bundle_dir: str, title: str = "", write: bool = True, target_chapters: int = 8, max_visual_items: int = 120, chapter_mode: str = "semantic") -> dict[str, Any]:
        return build_smart_summary_chapter_pack_impl(bundle_dir, title=title, write=write, target_chapters=target_chapters, max_visual_items=max_visual_items, chapter_mode=chapter_mode)

    @server.tool()
    def build_smart_summary_chapters(bundle_dir: str, title: str = "", write: bool = True, target_chapters: int = 8, max_visual_items: int = 120, chapter_mode: str = "semantic") -> dict[str, Any]:
        return build_smart_summary_chapter_pack_impl(bundle_dir, title=title, write=write, target_chapters=target_chapters, max_visual_items=max_visual_items, chapter_mode=chapter_mode)

    @server.tool()
    def smart_summary_section_workflow_tool(bundle_dir: str, title: str = "", write: bool = True, target_chapters: int = 8) -> dict[str, Any]:
        return build_smart_summary_section_workflow_impl(bundle_dir, title=title, write=write, target_chapters=target_chapters)

    @server.tool()
    def smart_summary_section_workflow(bundle_dir: str, title: str = "", write: bool = True, target_chapters: int = 8) -> dict[str, Any]:
        return build_smart_summary_section_workflow_impl(bundle_dir, title=title, write=write, target_chapters=target_chapters)

    @server.tool()
    def smart_summary_section_editor_tool(bundle_dir: str, write: bool = True) -> dict[str, Any]:
        return build_smart_summary_section_editor_impl(bundle_dir, write=write)

    @server.tool()
    def smart_summary_section_editor(bundle_dir: str, write: bool = True) -> dict[str, Any]:
        return build_smart_summary_section_editor_impl(bundle_dir, write=write)

    @server.tool()
    def smart_summary_section_apply_tool(bundle_dir: str, input_json: str = "", write: bool = True, require_all_sections: bool = False) -> dict[str, Any]:
        return apply_smart_summary_sections_impl(bundle_dir, input_json=input_json or None, write=write, require_all_sections=require_all_sections)

    @server.tool()
    def smart_summary_section_apply(bundle_dir: str, input_json: str = "", write: bool = True, require_all_sections: bool = False) -> dict[str, Any]:
        return apply_smart_summary_sections_impl(bundle_dir, input_json=input_json or None, write=write, require_all_sections=require_all_sections)

    @server.tool()
    def run_smart_summary_section_llm_rewrite_tool(
        bundle_dir: str,
        provider_config: dict[str, Any] | None = None,
        execute: bool = False,
        auto_from_profile: bool = False,
        quality_profile: str = "quality",
        target_chapters: int = 8,
        limit: int = 0,
        section_ids: str = "",
        only_needing_rewrite: bool = False,
        max_prompt_chars: int = 6000,
        max_tokens: int = 1200,
        min_section_chars: int = 120,
        temperature: float = 0,
        install: bool = True,
        require_all_sections: bool = True,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_smart_summary_section_llm_rewrite_impl(
            bundle_dir,
            provider_config=provider_config,
            execute=execute,
            auto_from_profile=auto_from_profile,
            quality_profile=quality_profile,
            target_chapters=target_chapters,
            limit=limit,
            section_ids=section_ids,
            only_needing_rewrite=only_needing_rewrite,
            max_prompt_chars=max_prompt_chars,
            max_tokens=max_tokens,
            min_section_chars=min_section_chars,
            temperature=temperature,
            install=install,
            require_all_sections=require_all_sections,
            write=write,
        )

    @server.tool()
    def run_smart_summary_section_llm_rewrite(
        bundle_dir: str,
        provider_config: dict[str, Any] | None = None,
        execute: bool = False,
        auto_from_profile: bool = False,
        quality_profile: str = "quality",
        target_chapters: int = 8,
        limit: int = 0,
        section_ids: str = "",
        only_needing_rewrite: bool = False,
        max_prompt_chars: int = 6000,
        max_tokens: int = 1200,
        min_section_chars: int = 120,
        temperature: float = 0,
        install: bool = True,
        require_all_sections: bool = True,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_smart_summary_section_llm_rewrite_impl(
            bundle_dir,
            provider_config=provider_config,
            execute=execute,
            auto_from_profile=auto_from_profile,
            quality_profile=quality_profile,
            target_chapters=target_chapters,
            limit=limit,
            section_ids=section_ids,
            only_needing_rewrite=only_needing_rewrite,
            max_prompt_chars=max_prompt_chars,
            max_tokens=max_tokens,
            min_section_chars=min_section_chars,
            temperature=temperature,
            install=install,
            require_all_sections=require_all_sections,
            write=write,
        )

    @server.tool()
    def smart_summary_global_reduce(
        bundle_dir: str,
        provider_config: dict[str, Any] | None = None,
        execute: bool = False,
        max_input_chars: int = 60000,
        max_tokens: int = 5000,
        temperature: float = 0,
        install: bool = True,
        write: bool = True,
    ) -> dict[str, Any]:
        return run_smart_summary_global_reduce_impl(
            bundle_dir,
            provider_config=provider_config,
            execute=execute,
            max_input_chars=max_input_chars,
            max_tokens=max_tokens,
            temperature=temperature,
            install=install,
            write=write,
        )

    @server.tool()
    def summary_consistency_check(
        bundle_dir: str,
        summary_path: str = "",
        write: bool = True,
    ) -> dict[str, Any]:
        return run_summary_consistency_check_impl(
            bundle_dir,
            summary_path=summary_path or None,
            write=write,
        )

    @server.tool()
    def smart_summary_quality_check_tool(bundle_dir: str, summary_path: str = "", require_codex: bool = False, write: bool = True) -> dict[str, Any]:
        return smart_summary_quality_check_impl(bundle_dir, summary_path=summary_path or None, require_codex=require_codex, write=write)

    @server.tool()
    def smart_summary_quality_check(bundle_dir: str, summary_path: str = "", require_codex: bool = False, write: bool = True) -> dict[str, Any]:
        return smart_summary_quality_check_impl(bundle_dir, summary_path=summary_path or None, require_codex=require_codex, write=write)

    @server.tool()
    def content_asset_status_tool(bundle_dir: str, write: bool = False) -> dict[str, Any]:
        return content_asset_status_impl(bundle_dir, write=write)

    @server.tool()
    def content_asset_status(bundle_dir: str, write: bool = False) -> dict[str, Any]:
        return content_asset_status_impl(bundle_dir, write=write)

    @server.tool()
    def export_quality_console(bundle_dir: str, write: bool = True) -> dict[str, Any]:
        return export_quality_console_impl(bundle_dir, write=write)

    @server.tool()
    def export_task_console(bundle_dir: str, write: bool = True, refresh: bool = False) -> dict[str, Any]:
        return export_task_console_impl(bundle_dir, write=write, refresh=refresh)

    @server.tool()
    def export_task_console_tool(bundle_dir: str, write: bool = True, refresh: bool = False) -> dict[str, Any]:
        return export_task_console_impl(bundle_dir, write=write, refresh=refresh)

    @server.tool()
    def subqueue_action_plan(bundle_dir: str, write: bool = True, refresh: bool = False) -> dict[str, Any]:
        return export_subqueue_action_plan_impl(bundle_dir, write=write, refresh=refresh)

    @server.tool()
    def subqueue_action_plan_tool(bundle_dir: str, write: bool = True, refresh: bool = False) -> dict[str, Any]:
        return export_subqueue_action_plan_impl(bundle_dir, write=write, refresh=refresh)

    @server.tool()
    def batch_content_asset_status(batch_input: str, output_dir: str = "", write: bool = True) -> dict[str, Any]:
        return batch_content_asset_status_impl(batch_input, output_dir=output_dir, write=write)

    @server.tool()
    def content_handoff_pack(batch_input: str, output_dir: str = "", write: bool = True) -> dict[str, Any]:
        return content_handoff_pack_impl(batch_input, output_dir=output_dir, write=write)

    @server.tool()
    def bundle_status_report_tool(bundle_dir: str, refresh: bool = True) -> dict[str, Any]:
        return bundle_status_report_impl(bundle_dir, refresh=refresh)

    @server.tool()
    def bundle_status_report(bundle_dir: str, refresh: bool = True) -> dict[str, Any]:
        return bundle_status_report_impl(bundle_dir, refresh=refresh)

    @server.tool()
    def acceptance_check_tool(bundle_dir: str, refresh: bool = True, write: bool = True) -> dict[str, Any]:
        return acceptance_check_impl(bundle_dir, refresh=refresh, write=write)

    @server.tool()
    def acceptance_check(bundle_dir: str, refresh: bool = True, write: bool = True) -> dict[str, Any]:
        return acceptance_check_impl(bundle_dir, refresh=refresh, write=write)

    @server.tool()
    def controlled_execution_check_tool(bundle_dir: str, refresh: bool = False, write: bool = True) -> dict[str, Any]:
        return controlled_execution_check_impl(bundle_dir, refresh=refresh, write=write)

    @server.tool()
    def controlled_execution_check(bundle_dir: str, refresh: bool = False, write: bool = True) -> dict[str, Any]:
        return controlled_execution_check_impl(bundle_dir, refresh=refresh, write=write)

    @server.tool()
    def controlled_execution_smoke_tool(
        bundle_dir: str,
        execute: bool = False,
        restore_after: bool = False,
        provider_config: dict[str, Any] | None = None,
        kind: str = "auto",
        index: int | None = None,
        frame_count: int = 8,
        write: bool = True,
    ) -> dict[str, Any]:
        return controlled_execution_smoke_impl(
            bundle_dir,
            execute=execute,
            restore_after=restore_after,
            provider_config=provider_config,
            kind=kind,
            index=index,
            frame_count=frame_count,
            write=write,
        )

    @server.tool()
    def controlled_execution_smoke(
        bundle_dir: str,
        execute: bool = False,
        restore_after: bool = False,
        provider_config: dict[str, Any] | None = None,
        kind: str = "auto",
        index: int | None = None,
        frame_count: int = 8,
        write: bool = True,
    ) -> dict[str, Any]:
        return controlled_execution_smoke_impl(
            bundle_dir,
            execute=execute,
            restore_after=restore_after,
            provider_config=provider_config,
            kind=kind,
            index=index,
            frame_count=frame_count,
            write=write,
        )

    @server.tool()
    def bundle_next_action_tool(bundle_dir: str, refresh: bool = True) -> dict[str, Any]:
        return bundle_next_action_impl(bundle_dir, refresh=refresh)

    @server.tool()
    def bundle_next_action(bundle_dir: str, refresh: bool = True) -> dict[str, Any]:
        return bundle_next_action_impl(bundle_dir, refresh=refresh)

    @server.tool()
    def bundle_advance_tool(
        bundle_dir: str,
        execute: bool = False,
        refresh_outputs: bool = False,
        vault: str = "",
        folder: str = "00_Inbox/AI/课程视频知识包",
        timeout_seconds: int = 30,
        ocr_input_json: str | None = None,
        ocr_language: str = "chi_sim",
        captiocr_root: str | None = None,
        visual_structure_input_json: str | None = None,
        provider_config: dict[str, Any] | None = None,
        multimodal_limit: int | None = None,
        temporal_limit: int | None = None,
        frame_count: int | None = None,
        confirm_vision_calls: int | None = None,
        confirm_vision_indexes: str = "",
    ) -> dict[str, Any]:
        return bundle_advance_impl(
            bundle_dir,
            execute=execute,
            refresh_outputs=refresh_outputs,
            vault=vault or None,
            folder=folder,
            timeout_seconds=timeout_seconds,
            ocr_input_json=ocr_input_json,
            ocr_language=ocr_language,
            captiocr_root=captiocr_root,
            visual_structure_input_json=visual_structure_input_json,
            provider_config=provider_config,
            multimodal_limit=multimodal_limit,
            temporal_limit=temporal_limit,
            frame_count=frame_count,
            confirm_vision_calls=confirm_vision_calls,
            confirm_vision_indexes=confirm_vision_indexes,
        )

    @server.tool()
    def bundle_advance(
        bundle_dir: str,
        execute: bool = False,
        refresh_outputs: bool = False,
        vault: str = "",
        folder: str = "00_Inbox/AI/课程视频知识包",
        timeout_seconds: int = 30,
        ocr_input_json: str | None = None,
        ocr_language: str = "chi_sim",
        captiocr_root: str | None = None,
        visual_structure_input_json: str | None = None,
        provider_config: dict[str, Any] | None = None,
        multimodal_limit: int | None = None,
        temporal_limit: int | None = None,
        frame_count: int | None = None,
        confirm_vision_calls: int | None = None,
        confirm_vision_indexes: str = "",
    ) -> dict[str, Any]:
        return bundle_advance_impl(
            bundle_dir,
            execute=execute,
            refresh_outputs=refresh_outputs,
            vault=vault or None,
            folder=folder,
            timeout_seconds=timeout_seconds,
            ocr_input_json=ocr_input_json,
            ocr_language=ocr_language,
            captiocr_root=captiocr_root,
            visual_structure_input_json=visual_structure_input_json,
            provider_config=provider_config,
            multimodal_limit=multimodal_limit,
            temporal_limit=temporal_limit,
            frame_count=frame_count,
            confirm_vision_calls=confirm_vision_calls,
            confirm_vision_indexes=confirm_vision_indexes,
        )

    @server.tool()
    def bundle_advance_queue_tool(
        bundle_dir: str,
        max_steps: int = 4,
        execute: bool = False,
        refresh_outputs: bool = False,
        vault: str = "",
        folder: str = "00_Inbox/AI/课程视频知识包",
        timeout_seconds: int = 30,
        ocr_input_json: str | None = None,
        ocr_language: str = "chi_sim",
        captiocr_root: str | None = None,
        visual_structure_input_json: str | None = None,
        provider_config: dict[str, Any] | None = None,
        multimodal_limit: int | None = None,
        temporal_limit: int | None = None,
        frame_count: int | None = None,
        confirm_vision_calls: int | None = None,
        confirm_vision_indexes: str = "",
    ) -> dict[str, Any]:
        return bundle_advance_queue_impl(
            bundle_dir,
            max_steps=max_steps,
            execute=execute,
            refresh_outputs=refresh_outputs,
            vault=vault or None,
            folder=folder,
            timeout_seconds=timeout_seconds,
            ocr_input_json=ocr_input_json,
            ocr_language=ocr_language,
            captiocr_root=captiocr_root,
            visual_structure_input_json=visual_structure_input_json,
            provider_config=provider_config,
            multimodal_limit=multimodal_limit,
            temporal_limit=temporal_limit,
            frame_count=frame_count,
            confirm_vision_calls=confirm_vision_calls,
            confirm_vision_indexes=confirm_vision_indexes,
        )

    @server.tool()
    def bundle_advance_queue(
        bundle_dir: str,
        max_steps: int = 4,
        execute: bool = False,
        refresh_outputs: bool = False,
        vault: str = "",
        folder: str = "00_Inbox/AI/课程视频知识包",
        timeout_seconds: int = 30,
        ocr_input_json: str | None = None,
        ocr_language: str = "chi_sim",
        captiocr_root: str | None = None,
        visual_structure_input_json: str | None = None,
        provider_config: dict[str, Any] | None = None,
        multimodal_limit: int | None = None,
        temporal_limit: int | None = None,
        frame_count: int | None = None,
        confirm_vision_calls: int | None = None,
        confirm_vision_indexes: str = "",
    ) -> dict[str, Any]:
        return bundle_advance_queue_impl(
            bundle_dir,
            max_steps=max_steps,
            execute=execute,
            refresh_outputs=refresh_outputs,
            vault=vault or None,
            folder=folder,
            timeout_seconds=timeout_seconds,
            ocr_input_json=ocr_input_json,
            ocr_language=ocr_language,
            captiocr_root=captiocr_root,
            visual_structure_input_json=visual_structure_input_json,
            provider_config=provider_config,
            multimodal_limit=multimodal_limit,
            temporal_limit=temporal_limit,
            frame_count=frame_count,
            confirm_vision_calls=confirm_vision_calls,
            confirm_vision_indexes=confirm_vision_indexes,
        )

    @server.tool()
    def bundle_advance_log_tool(bundle_dir: str) -> dict[str, Any]:
        return bundle_advance_log_impl(bundle_dir)

    @server.tool()
    def bundle_advance_log(bundle_dir: str) -> dict[str, Any]:
        return bundle_advance_log_impl(bundle_dir)

    server.run()


if __name__ == "__main__":
    main()
