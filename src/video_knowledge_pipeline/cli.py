from __future__ import annotations

from .storage import read_json

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .cli_output import render_cli_result

from .acceptance_check import acceptance_check
from .path_defaults import provider_env_file, workspace_root
from .acceptance_run import run_acceptance_bundle, run_acceptance_run
from .adaptive_asr_route import build_adaptive_asr_route
from .asr_chunk_batch_status import query_asr_chunk_batch_status
from .asr_chunk_batch_submit import submit_asr_chunk_batch_workflow
from .asr_chunk_batch_workflow import (
    build_asr_chunk_batch_workflow,
    build_asr_chunk_business_workflow,
)
from .asr_chunk_batch_merge import merge_asr_chunk_batch_reports
from .asr_vad_activity_audit import audit_asr_vad_audio_activity
from .asr_vad_independent_crosscheck import (
    crosscheck_asr_vad_with_independent_candidate,
)
from .asr_vad_profile_comparison import compare_asr_vad_profiles
from .asr_ab_compare import compare_asr_ab_sample
from .asr_ab_plan import plan_asr_ab_sample
from .asr_ab_run import run_asr_ab_sample
from .asr_environment import asr_environment_status
from .asr_local_targeted_evidence import build_local_targeted_asr_evidence
from .asr_model_cache import asr_model_cache_status, prepare_asr_model_cache
from .asr_execution import asr_smoke, run_asr_plan
from .asr_adapter import normalize_asr_output
from .asr_consensus import build_asr_consensus
from .asr_secondary_evidence import close_secondary_asr_evidence
from .asr_diff_adjudication import apply_asr_diff_adjudication, build_asr_diff_adjudication
from .asr_evidence_autoadjudication import adjudicate_asr_with_independent_evidence
from .asr_runner import plan_asr_run, plan_whisperx_alignment
from .asr_retry_snippets import prepare_asr_retry_snippets
from .local_targeted_asr_execution import run_local_targeted_asr_evidence
from .local_targeted_asr_plan import build_local_targeted_asr_plan
from .asr_vad_chunking import prepare_asr_vad_chunks
from .silero_vad_candidate import run_silero_vad_candidate
from .whisperx_alignment import run_whisperx_alignment
from .asr_setup_plan import plan_asr_setup
from .batch_repair import batch_repair_run
from .batch_run import batch_video_knowledge_run
from .bilinote_summary_tools import build_mind_map_prompt_pack
from .bilinote_mind_map_prompt_pack import build_bundle_mind_map_prompt_pack
from .bundle_assets import repair_bundle_assets
from .bundle_next import bundle_advance, bundle_advance_log, bundle_advance_queue, bundle_next_action
from .bundle_readiness import audit_bundle_readiness
from .bundle_source_artifacts import bundle_source_artifacts
from .bundle_status import bundle_status_report, controlled_execution_check
from .cloud_asr import plan_cloud_asr_run, prepare_cloud_asr_audio, run_cloud_asr_plan
from .config import config_status, model_api_settings_status, set_asr_runtime_profile, set_vision_execution_profile, DEFAULT_LOCAL_FRAME_BUDGET, DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS, DEFAULT_LOCAL_FRAME_SAMPLING_MODE, LOCAL_FRAME_SAMPLING_MODES
from .content_asset_batch import batch_content_asset_status, content_handoff_pack
from .content_asset_status import content_asset_status
from .content_profile import SUPPORTED_PROFILES, apply_content_profile
from .codex_vision_session import import_codex_vision_results, prepare_codex_vision_session
from .codex_vision_executor import codex_vision_catalog, run_codex_vision
from .summary_claim_review import prepare_summary_claim_review, import_summary_claim_review, summary_claim_review_status
from .controlled_execution_smoke import controlled_execution_smoke
from .consented_model_task_cli import run_consented_model_task_cli
from .model_business_authorization import (
    create_business_child_consent,
    create_model_business_authorization_from_plan,
    validate_model_business_authorization,
)
from .trusted_model_connector_policy import TrustedModelConnectorPolicy
from .creative_contract_bridge import (
    build_material_manifest,
    import_generation_contracts,
    import_previs_candidate,
    validate_material_manifest,
)
from .evidence_conflict_index import build_evidence_conflict_index
from .entity_lexicon import build_entity_lexicon
from .extractor_execution import extractor_run_log, run_extractor_plan
from .external_capability_pack import build_external_capability_pack
from .frame_recapture import run_frame_recapture_plan
from .general_tagger_adapter import general_tagger_status, run_general_tagger
from .temporal_tag_delta import apply_temporal_tag_delta_restore, run_temporal_tag_delta
from .high_res_tile_plan import run_high_res_tile_plan
from .tile_result_import_builder import build_tile_result_import
from .tile_result_merge import run_tile_result_merge
from .knowledge_coverage import audit_knowledge_coverage
from .companion_courseware_text import import_companion_courseware_text
from .knowledge_note_export import export_knowledge_note
from .lecture_pipeline import run_ready_lecture_pipeline
from .lecture_workflow import refresh_lecture_review_outputs
from .local_video_run import prepare_local_video_run
from .local_tool_inventory import local_runtime_preflight
from .local_production_preset import install_local_production_preset
from .local_media_progress import stderr_progress_callback
from .local_asr_service_adapter import plan_local_asr_service_run, run_local_asr_service_plan
from .local_vlm_server_adapter import local_vlm_adapter_plan, local_vlm_serving_smoke
from .long_video_memory_pack import build_long_video_memory_pack
from .long_video_fast_segment import (
    apply_long_video_fast_segment_review,
    build_long_video_fast_segment_plan,
    render_long_video_fast_segment,
)
from .media_artifact_lifecycle import (
    MediaArtifactLifecycleError,
    explain_media_artifact,
    media_artifact_doctor,
    plan_media_artifact_gc,
    restore_synthetic_media_artifact,
    scan_media_artifacts,
    verify_media_artifact_plan,
)
from .small_model_video_closure import (
    VideoClosureError,
    resume_video_closure,
    run_video_closure,
    video_closure_doctor,
    video_closure_status,
)
from .media_capability_registry import media_capability_registry_status
from .highlight_detection_adapter import run_highlight_detection
from .media_connector_consent import media_connector_preflight
from .media_route_settings import media_route_settings_status
from .mcp_args_audit import (
    audit_bundle_mcp_args as audit_bundle_mcp_args_impl,
    filter_mcp_payload,
    normalise_mcp_payload,
    read_mcp_args,
)
from .multimodal_frame_analyzer import run_multimodal_frame_analysis, vision_analysis_apply_restore, vision_analysis_restore_plan, vision_analysis_run_log
from .multimodal_sample_review import multimodal_sample_review, validate_multimodal_sample_notes
from .model_api_settings import apply_model_api_route_preset, prepare_model_api_onboarding_bundles
from .model_provider_probe import probe_model_api_onboarding_bundle
from .model_task_automation import run_bilinote_mind_map_model, run_term_arbitration_model
from .model_task_gateway import model_task_coverage_audit
from .ocr_backfill import run_ocr_backfill
from .page_metadata import import_page_metadata
from .online_model_gateway import online_model_api_call, online_model_api_matrix
from .offline_quality_router import offline_quality_route
from .openclaw_bridge_status import openclaw_bridge_status
from .openclaw_bridge_doctor import openclaw_bridge_doctor
from .openclaw_docker_contract import openclaw_docker_contract_check
from .openclaw_integration import openclaw_video_ingest, openclaw_video_link, openclaw_video_plan
from .openclaw_live_smoke import openclaw_live_smoke
from .peepshow_adapter import attach_peepshow_output_to_bundle
from .punctuation_model_stage import run_punctuation_model_stage
from .quality_console import export_quality_console
from .production_artifact_gate import evaluate_production_artifact_gate
from .quality_finalize import finalize_quality_outputs
from .quality_benchmark import build_quality_benchmark, report_quality_benchmark, run_quality_benchmark
from .quality_benchmark_arbitration import build_quality_benchmark_arbitration, evaluate_quality_benchmark_arbitration
from .quality_benchmark_punctuation import run_quality_benchmark_punctuation
from .quality_benchmark_punctuation_agent import build_quality_benchmark_punctuation_agent_pack, evaluate_quality_benchmark_punctuation_agent
from .quality_benchmark_residual_conflicts import build_quality_benchmark_residual_conflicts
from .quality_benchmark_variants import execute_quality_benchmark_variants
from .summary_blind_review import apply_summary_blind_review, build_summary_blind_review
from .repair_status import refresh_bundle_repair_status
from .review_attestation import create_review_attestation, validate_review_attestation
from .review_session import apply_review_notes_to_bundle, prepare_review_session, review_closure_status, validate_review_notes_for_bundle
from .run_artifact_registry import build_run_artifact_registry
from .screen_text_recovery import run_screen_text_recovery
from .scene_detection_adapter import run_scene_detection
from .scene_candidate_evidence import build_scene_candidate_evidence
from .source_review_lineage import discover_source_review_lineage
from .campplus_speaker_center_sidecar import build_campplus_speaker_center_sidecar
from .speaker_global_alignment import (
    bind_local_voiceprint_role,
    build_speaker_global_alignment,
    delete_local_voiceprint,
    enroll_local_voiceprints,
    match_local_voiceprints,
)
from .speaker_shared_session_alignment import build_shared_session_speaker_alignment
from .technical_shot_detection import run_technical_shot_detection
from .technical_shot_fusion import fuse_technical_shot_boundaries
from .shot_breakdown import build_shot_breakdown
from .creator_report import creator_report_status, export_creator_report
from .shot_language_analysis import run_shot_language_analysis
from .shot_review import apply_shot_review_notes, shot_review_status
from .video_decomposition import (
    build_video_decomposition_report,
    compare_video_decomposition_reports,
    video_decomposition_report_status,
)
from .semantic_chapter_plan import build_semantic_chapter_plan
from .smart_summary_chapters import build_smart_summary_chapter_pack
from .smart_summary_codex import generate_smart_summary_with_codex, prepare_smart_summary_llm_rewrite, run_smart_summary_llm_rewrite, smart_summary_quality_check
from .smart_summary_input_pack import build_smart_summary_input_pack
from .smart_summary_section_apply import apply_smart_summary_sections
from .smart_summary_section_editor import build_smart_summary_section_editor
from .smart_summary_section_workflow import build_smart_summary_section_workflow
from .smart_summary_section_llm import run_smart_summary_section_llm_rewrite
from .smart_summary_global_reduce import run_smart_summary_global_reduce
from .summary_consistency import run_summary_consistency_check
from .supplemental_frame_sampling import plan_supplemental_frame_sampling
from .tagger_import import import_tagger_annotations
from .targeted_visual_evidence import run_targeted_visual_evidence
from .task_console import export_subqueue_action_plan, export_task_console
from .temporal_frame_groups import run_temporal_frame_groups
from .term_arbitration_codex import build_term_arbitration_codex_pack, validate_term_arbitration_codex_result
from .term_correction_impact import term_correction_impact_report
from .term_correction_closure import run_term_correction_closure
from .term_correction_status import term_correction_status
from .term_resolution import resolve_terms
from .temporal_visual_analyzer import run_temporal_visual_analysis
from .text_llm_gateway import text_llm_provider_smoke
from .timeline_alignment_audit import timeline_alignment_audit
from .transcript_correction_pack import build_transcript_correction_pack
from .transcript_candidate_recall_benchmark import benchmark_transcript_candidate_recall
from .transcript_reference_window import export_transcript_reference_window
from .transcript_semantic_batch import transcript_semantic_acceptance, transcript_semantic_batch_acceptance, transcript_semantic_batch_codex_review_draft, transcript_semantic_batch_import_review_notes, transcript_semantic_batch_review_pack, transcript_semantic_repair_queue, transcript_semantic_repair_run
from .transcript_semantic_summary_impact import transcript_semantic_summary_impact_report
from .transcript_semantic_benchmark import run_transcript_semantic_benchmark
from .transcript_semantic_correction import (
    build_transcript_semantic_candidate_discovery_pack,
    build_transcript_semantic_candidate_discovery_codex_draft,
    build_transcript_semantic_candidate_discovery_llm_draft,
    build_transcript_semantic_correction_codex_draft,
    build_transcript_semantic_correction_llm_draft,
    build_transcript_semantic_correction_pack,
    import_transcript_semantic_candidate_suggestions,
    import_transcript_semantic_review_notes,
    transcript_semantic_correction_closure,
    transcript_semantic_correction_impact_report,
    transcript_semantic_correction_readable_impact_report,
    transcript_semantic_correction_status,
    validate_transcript_semantic_correction,
)
from .transcript_semantic_transaction import apply_transcript_semantic_restore
from .transcript_agent_readable import run_agent_readable_transcript_rewrite
from .transcript_evidence_correction_pipeline import run_transcript_evidence_correction_pipeline
from .transcript_main_route_status import transcript_main_route_status
from .transcript_source_arbitration import arbitrate_transcript_sources
from .transcript_editor import apply_transcript_edits, prepare_transcript_edit_session
from .subtitle_editor import apply_subtitle_review, validate_subtitle_review
from .subtitle_editor_ui import prepare_subtitle_editor
from .transcript_resegment import resegment_transcript
from .transcript_postprocess import postprocess_asr_transcript
from .transcript_translation import translate_transcript_to_mandarin
from .transcript_quality_gate import run_transcript_quality_gate
from .transcript_downstream_refresh import refresh_transcript_downstream_outputs
from .transcript_readable_llm import run_readable_transcript_llm_polish
from .video_frame_router import run_video_frame_router
from .video_moment_index import build_video_moment_index
from .video_rag_pack import build_video_rag_pack
from .video_rag_http import serve_video_rag, video_rag_service_plan
from .volcengine_model_routing import volcengine_model_routing
from .volcengine_model_task_matrix import run_volcengine_model_task_matrix
from .video_rag_search import search_video_rag
from .script_clip_candidate_pack import build_script_clip_candidate_pack
from .script_clip_alignment import check_script_clip_alignment
from .content_clip_candidate_pack import build_content_clip_candidate_pack
from .content_clip_alignment import check_content_clip_alignment
from .video_evidence_query import apply_video_evidence_confirmation, build_video_evidence_query_plan
from .video_edit_review_pack import build_video_edit_review_pack
from .video_source import prepare_video_source
from .video_structure import build_video_structure
from .video_workbench import export_video_workbench
from .vdo_handoff import ingest_vdo_handoff, vdo_handoff_plan
from .vision_acceptance import vision_acceptance_plan
from .vision_api import test_vision_provider
from .vision_environment import vision_environment_status
from .vision_export_consent import create_vision_export_consent, revoke_vision_export_consent, vision_export_consent_status
from .vision_preflight import vision_execution_preflight
from .vision_provider_smoke import vision_provider_matrix, vision_provider_smoke
from .vision_review_queue import vision_review_queue
from .vision_review_triage import vision_review_triage
from .visual_structure import repair_ebook_artifact_text, run_visual_structure_plan
from .visual_structure_batch import run_visual_structure_ebook_batches
from .visual_ab_benchmark import build_visual_ab_benchmark_plan
from .webui_bridge import refresh_bundle_review_html


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    from .courseware_cli import COMMANDS as courseware_commands, main as courseware_main
    supplied = list(argv) if argv is not None else list(sys.argv[1:])
    if supplied and supplied[0] in courseware_commands:
        return courseware_main(supplied)
    parser = build_parser()
    cli_argv = list(argv) if argv is not None else list(sys.argv[1:])
    verbose = "--verbose" in cli_argv
    cli_argv = [value for value in cli_argv if value != "--verbose"]
    args = parser.parse_args(cli_argv)

    if args.command in {
        "media-artifact-doctor",
        "media-artifact-preflight",
        "media-artifact-scan",
        "media-artifact-explain",
        "media-artifact-plan-gc",
        "media-artifact-verify-plan",
        "media-artifact-restore",
    }:
        try:
            if args.command in {"media-artifact-doctor", "media-artifact-preflight"}:
                result = media_artifact_doctor(
                    args.bundle_dir or None,
                    scope_path=args.scope_path or None,
                    core_root=args.core_root or None,
                    self_media_root=args.self_media_root or None,
                    large_task=args.command == "media-artifact-preflight",
                    estimated_output_bytes=getattr(args, "estimated_output_bytes", None),
                    explicit_non_large=bool(getattr(args, "non_large", False)),
                )
            elif args.command == "media-artifact-scan":
                result = scan_media_artifacts(
                    args.bundle_dir,
                    scope_path=args.scope_path,
                    output_path=args.output_path,
                    checkpoint_dir=args.checkpoint_dir,
                    core_root=args.core_root or None,
                    max_depth=args.max_depth,
                    max_items=args.max_items,
                )
            elif args.command == "media-artifact-explain":
                result = explain_media_artifact(
                    args.manifest_path,
                    args.artifact_id,
                    core_root=args.core_root or None,
                )
            elif args.command == "media-artifact-plan-gc":
                result = plan_media_artifact_gc(
                    args.manifest_path,
                    bundle_dir=args.bundle_dir,
                    quarantine_root=args.quarantine_root,
                    output_path=args.output_path,
                    core_root=args.core_root or None,
                )
            elif args.command == "media-artifact-verify-plan":
                result = verify_media_artifact_plan(
                    args.plan_path,
                    args.manifest_path,
                    bundle_dir=args.bundle_dir,
                    quarantine_root=args.quarantine_root,
                    core_root=args.core_root or None,
                )
            else:
                result = restore_synthetic_media_artifact(
                    args.plan_path,
                    args.receipt_path,
                    bundle_dir=args.bundle_dir,
                    quarantine_root=args.quarantine_root,
                    output_path=args.output_path,
                    execute_synthetic=args.execute_synthetic,
                    core_root=args.core_root or None,
                )
        except MediaArtifactLifecycleError as exc:
            result = {
                "schema_version": "video_knowledge_pipeline.media_artifact_cli_failure.v1",
                "status": "blocked",
                "error_code": exc.error_code,
                "detail": exc.detail,
                "cleanup_mode": "plan_only",
                "filesystem_mutation_performed": False,
            }
            print(render_cli_result(result, verbose=verbose))
            return 2
        print(render_cli_result(result, verbose=verbose))
        return 2 if result.get("status") == "blocked" else 0

    if args.command in {
        "video-closure-doctor",
        "video-closure-run",
        "video-closure-resume",
        "video-closure-status",
    }:
        try:
            if args.command == "video-closure-doctor":
                result = video_closure_doctor(
                    args.bundle_dir or None,
                    mvp_root=args.mvp_root or None,
                )
            elif args.command == "video-closure-run":
                result = run_video_closure(
                    args.bundle_dir,
                    output_dir=args.output_dir or None,
                    closure_run_id=args.closure_run_id or None,
                    mvp_root=args.mvp_root or None,
                    pause_after=args.pause_after or None,
                    execution_mode=args.execution_mode,
                )
            elif args.command == "video-closure-resume":
                result = resume_video_closure(
                    args.closure_dir,
                    mvp_root=args.mvp_root or None,
                )
            else:
                result = video_closure_status(args.closure_dir)
        except VideoClosureError as exc:
            result = {
                "schema": "video_knowledge_pipeline.video_closure_cli_failure.v1",
                "status": "blocked",
                "error_code": exc.error_code,
                "detail": exc.detail,
                "auto_retry_count": 0,
                "explicit_resume_required": True,
            }
            print(render_cli_result(result, verbose=verbose))
            return 2
        print(render_cli_result(result, verbose=verbose))
        return 2 if result.get("status") == "blocked" else 0

    if args.command == "execute-consented-model-task":
        result, exit_code = run_consented_model_task_cli(
            args.consent_path,
            route_revision=args.route_revision,
            write=args.write,
        )
        print(render_cli_result(result, verbose=verbose))
        return exit_code

    if args.command == "model-business-authorization-create":
        policy = TrustedModelConnectorPolicy.from_environment(
            default_root=Path(__file__).resolve().parents[2]
        )
        result = create_model_business_authorization_from_plan(
            args.plan_path,
            confirm_data_export=args.confirm_data_export,
            output_path=args.output_path or None,
            policy=policy,
            write=not args.no_write,
        )
    elif args.command == "model-business-authorization-status":
        policy = TrustedModelConnectorPolicy.from_environment(
            default_root=Path(__file__).resolve().parents[2]
        )
        result = validate_model_business_authorization(
            args.authorization_path, policy=policy
        )
    elif args.command == "model-business-child-consent":
        policy = TrustedModelConnectorPolicy.from_environment(
            default_root=Path(__file__).resolve().parents[2]
        )
        result = create_business_child_consent(
            args.authorization_path,
            stage_id=args.stage_id,
            artifact_paths=args.artifact,
            producer=args.producer,
            input_paths=args.lineage_input,
            max_calls=args.max_calls,
            output_path=args.output_path or None,
            policy=policy,
            write=not args.no_write,
        )
    elif args.command == "media-capability-status":
        result = media_capability_registry_status()
    elif args.command == "general-tagger-status":
        result = general_tagger_status(
            source_root=args.source_root or None,
            checkpoint_path=args.checkpoint_path or None,
        )
    elif args.command == "media-route-status":
        result = media_route_settings_status(
            task=args.task,
            settings_path=args.settings_path or None,
        )
    elif args.command == "media-connector-preflight":
        result = media_connector_preflight(
            args.consent_path,
            route_revision=args.route_revision,
            expected_calls=args.expected_calls,
            settings_path=args.settings_path or None,
        )
    elif args.command == "compare-asr-variants":
        from .asr_variant_evaluation import compare_asr_variants
        result = compare_asr_variants(args.reference_json, primary=args.primary, secondary=args.secondary,
                                     integrated=args.integrated, output_json=args.output_json or None)
    elif args.command == "dual-asr-review":
        from .dual_asr_workflow import run_dual_asr_workflow
        result = run_dual_asr_workflow(
            args.bundle_dir, media_path=args.media_path or None,
            primary_transcript=args.primary_transcript or None, secondary_transcript=args.secondary_transcript or None,
            mode=args.mode, secondary_preset=args.secondary_preset, execute=args.execute,
            secondary_timestamps=args.secondary_timestamps,
            speech_activity_json=args.speech_activity_json or None, diarization_json=args.diarization_json or None,
            timeout_seconds=args.timeout_seconds, max_windows=args.max_windows, allow_cpu=args.allow_cpu,
            recover_failed_windows=not args.no_recover_failed_windows,
            recovery_window_seconds=args.recovery_window_seconds,
            recovery_max_windows=args.recovery_max_windows,
            write=not args.no_write,
        )
    elif args.command == "plan-qwen-window-recovery":
        from .qwen_failed_window_recovery import plan_qwen_failed_window_recovery
        result = plan_qwen_failed_window_recovery(
            args.partial_json, output_dir=args.output_dir,
            asr_plan_json=args.asr_plan_json or None, media_path=args.media_path or None,
            speech_activity_json=args.speech_activity_json or None,
            window_seconds=args.window_seconds, max_windows=args.max_windows, write=not args.no_write)
    elif args.command == "compose-qwen-window-recovery":
        from .qwen_failed_window_recovery import compose_qwen_failed_window_recovery
        result = compose_qwen_failed_window_recovery(
            args.plan_json, recovered_json=args.recovered_json, write=not args.no_write)
    elif args.command == "asr-consensus":
        result = build_asr_consensus(
            args.bundle_dir,
            primary_transcript=args.primary_transcript,
            secondary_transcript=args.secondary_transcript,
            media_path=args.media_path or None,
            agreement_threshold=args.agreement_threshold,
            execute_clips=args.execute_clips,
            speech_activity_json=args.speech_activity_json or None,
            diarization_json=args.diarization_json or None,
            critical_terms=args.critical_term,
            agreement_sample_size=args.agreement_sample_size,
            secondary_scope=args.secondary_scope,
            write=not args.no_write,
        )
    elif args.command == "asr-secondary-evidence":
        result = close_secondary_asr_evidence(
            args.bundle_dir,
            connector_execution=args.connector_execution,
            prepared_suite=args.prepared_suite,
            candidate_id=args.candidate_id,
            primary_transcript=args.primary_transcript or None,
            media_path=args.media_path or None,
            agreement_threshold=args.agreement_threshold,
            write=not args.no_write,
        )
    elif args.command == "asr-diff-adjudication":
        result = build_asr_diff_adjudication(
            args.bundle_dir,
            consensus_json=args.consensus_json or None,
            cluster_token_gap=args.cluster_token_gap,
            write=not args.no_write,
        )
    elif args.command == "apply-asr-diff-adjudication":
        result = apply_asr_diff_adjudication(
            args.bundle_dir,
            decisions_json=args.decisions_json,
            pack_json=args.pack_json or None,
            min_confidence=args.min_confidence,
            require_evidence=not args.allow_without_evidence,
            promote=args.promote,
            review_only=args.review_only,
            write=not args.no_write,
        )
    elif args.command == "asr-evidence-autoadjudication":
        result = adjudicate_asr_with_independent_evidence(
            args.bundle_dir,
            secondary_transcript=args.secondary_transcript,
            corroborating_transcripts=args.corroborating_transcript,
            refresh_exports=args.refresh_exports,
            write=args.write,
        )
    elif args.command == "offline-quality-route":
        result = offline_quality_route(
            args.bundle_dir,
            benchmark_manifest=args.benchmark_manifest or None,
            output_dir=args.output_dir or None,
            write=not args.no_write,
        )
    elif args.command == "punctuation-model-stage":
        result = run_punctuation_model_stage(
            args.bundle_dir,
            input_path=args.input_path or None,
            model=args.model,
            device=args.device,
            block_chars=args.block_chars,
            execute=args.execute,
            promote=args.promote,
            write=not args.no_write,
        )
    elif args.command == "quality-finalize":
        result = finalize_quality_outputs(
            args.bundle_dir,
            provider_config=_provider_config_arg(args.provider_config),
            execute_llm=args.execute_llm,
            auto_from_profile=args.auto_from_profile,
            quality_profile=args.quality_profile,
            target_chapters=args.target_chapters,
            write=not args.no_write,
        )
    elif args.command == "refresh-transcript-downstream":
        result = refresh_transcript_downstream_outputs(
            args.bundle_dir,
            canonical_before_sha256=args.canonical_before_sha256,
            canonical_after_sha256=args.canonical_after_sha256,
            reason=args.reason,
            write=not args.no_write,
        )
    elif args.command == "quality-benchmark":
        if args.action == "build":
            result = build_quality_benchmark(
                args.input_path,
                bundle_dirs=[value.strip() for value in args.bundle_dirs.split(",") if value.strip()],
                media_paths=args.media_path,
                samples_per_bundle=args.samples_per_bundle,
                sample_seconds=args.sample_seconds,
                asr_language=args.asr_language,
                asr_context_hotwords=[
                    value.strip()
                    for value in args.asr_context_hotwords.split(",")
                    if value.strip()
                ],
                execute_clips=args.execute_clips,
                legacy_reference_manifest=args.legacy_reference_manifest or None,
                write=not args.no_write,
            )
        elif args.action == "execute-variants":
            result = execute_quality_benchmark_variants(
                args.input_path,
                variants=[value.strip() for value in args.variants.split(",") if value.strip()],
                execute=args.execute,
                resume=args.resume,
                retry_failed=args.retry_failed,
                limit=args.limit,
                timeout_seconds=args.timeout_seconds,
                write=not args.no_write,
            )
        elif args.action == "run":
            result = run_quality_benchmark(args.input_path, output_dir=args.output_dir or None, write=not args.no_write)
        elif args.action == "build-arbitration":
            result = build_quality_benchmark_arbitration(
                args.input_path,
                output_dir=args.output_dir or None,
                primary_variant=args.primary_variant,
                secondary_variant=args.secondary_variant,
                write=not args.no_write,
            )
        elif args.action == "evaluate-arbitration":
            result = evaluate_quality_benchmark_arbitration(
                args.input_path,
                private_json=args.private_json or None,
                decisions_json=args.decisions_json or None,
                output_dir=args.output_dir or None,
                min_confidence=args.min_confidence,
                write=not args.no_write,
            )
        elif args.action == "build-punctuation-agent":
            result = build_quality_benchmark_punctuation_agent_pack(
                args.input_path,
                output_dir=args.output_dir or None,
                source_variant=args.primary_variant,
                write=not args.no_write,
            )
        elif args.action == "evaluate-punctuation-agent":
            if not args.decisions_json:
                raise ValueError("--decisions-json is required for evaluate-punctuation-agent")
            result = evaluate_quality_benchmark_punctuation_agent(
                args.input_path,
                args.decisions_json,
                output_dir=args.output_dir or None,
                source_variant=args.primary_variant,
                write=not args.no_write,
            )
        elif args.action == "build-residual-conflicts":
            result = build_quality_benchmark_residual_conflicts(
                args.input_path,
                manifest_json=args.manifest_json or None,
                entity_lexicon_json=args.entity_lexicon_json or None,
                output_dir=args.output_dir or None,
                write=not args.no_write,
            )
        elif args.action == "punctuation-ab":
            result = run_quality_benchmark_punctuation(
                args.input_path,
                output_dir=args.output_dir or None,
                source_variant=args.primary_variant,
                model=args.punctuation_model,
                device=args.punctuation_device,
                execute=args.execute,
                write=not args.no_write,
            )
        elif args.action == "build-summary-review":
            result = build_summary_blind_review(args.input_path, output_dir=args.output_dir or None, write=not args.no_write)
        elif args.action == "apply-summary-review":
            if not args.scores_json:
                raise ValueError("--scores-json is required for apply-summary-review")
            result = apply_summary_blind_review(args.input_path, args.scores_json, write=not args.no_write)
        else:
            result = report_quality_benchmark(args.input_path, output_dir=args.output_dir or None, write=not args.no_write)
    elif args.command == "asr-env-status":
        result = asr_environment_status(args.venv_dir, output_dir=args.output_dir, write=args.write, python_version=args.python_version)
    elif args.command == "config-status":
        result = config_status(args.config_path)
    elif args.command == "model-api-settings":
        result = model_api_settings_status(args.config_path)
    elif args.command == "model-api-onboarding-prepare":
        result = prepare_model_api_onboarding_bundles(
            args.provider or None,
            settings_path=args.settings_path or None,
            secrets_path=args.secrets_path or None,
            refresh_known_models=args.refresh_known_models,
        )
    elif args.command == "model-api-catalog-probe":
        result = probe_model_api_onboarding_bundle(
            args.provider,
            execute=args.execute,
            include_model_ids=args.include_model_ids,
            settings_path=args.settings_path or None,
            secrets_path=args.secrets_path or None,
        )
    elif args.command == "model-api-route-preset":
        result = apply_model_api_route_preset(
            args.preset,
            settings_path=args.settings_path or None,
            secrets_path=args.secrets_path or None,
        )
    elif args.command == "local-production-preset":
        result = install_local_production_preset(
            args.output_dir,
            write=not args.no_write,
        )
    elif args.command == "set-vision-profile":
        result = set_vision_execution_profile(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            multimodal_limit=args.multimodal_limit,
            temporal_limit=args.temporal_limit,
            frame_count=args.frame_count,
            config_path=args.config_path,
        )
    elif args.command == "set-asr-runtime-profile":
        result = set_asr_runtime_profile(
            provider=args.provider,
            model=args.model,
            device=args.device,
            compute_type=args.compute_type,
            vad_model=args.vad_model,
            punc_model=args.punc_model,
            spk_model=args.spk_model,
            enable_vad=_optional_bool_arg(args.enable_vad),
            enable_itn=_optional_bool_arg(args.enable_itn),
            enable_punctuation=_optional_bool_arg(args.enable_punctuation),
            enable_diarization=_optional_bool_arg(args.enable_diarization),
            merge_vad=_optional_bool_arg(args.merge_vad),
            merge_length_s=args.merge_length_s,
            audio_preprocess=_optional_bool_arg(args.audio_preprocess),
            ffmpeg_normalize=_optional_bool_arg(args.ffmpeg_normalize),
            target_sample_rate=args.target_sample_rate,
            service_base_url=args.service_base_url,
            service_model=args.service_model,
            service_timeout_seconds=args.service_timeout_seconds,
            config_path=args.config_path,
        )
    elif args.command == "plan-asr-setup":
        result = plan_asr_setup(args.workspace_dir, venv_dir=args.venv_dir)
    elif args.command == "asr-model-cache-status":
        result = asr_model_cache_status(
            args.workspace_dir,
            models=_csv_arg(args.models),
            include_optional=args.include_optional,
            write=not args.no_write,
        )
    elif args.command == "prepare-asr-model-cache":
        result = prepare_asr_model_cache(
            args.workspace_dir,
            models=_csv_arg(args.models),
            include_optional=args.include_optional,
            execute=args.execute,
            allow_download=args.allow_download,
            device=args.device,
            timeout_seconds=args.timeout_seconds,
            write=not args.no_write,
        )
    elif args.command == "adaptive-asr-route":
        result = build_adaptive_asr_route(
            args.bundle_dir,
            args.media_path,
            workspace_dir=args.workspace_dir or None,
            task_profile=args.task_profile,
            base_lexicon_json=args.base_lexicon_json or None,
            include_online_plan=args.include_online_plan,
            provider_config=_provider_config_arg(args.provider_config),
            online_model=args.online_model,
            language=args.language,
            max_hotwords=args.max_hotwords,
            max_context_chars=args.max_context_chars,
            write=not args.no_write,
        )
    elif args.command == "prepare-cloud-asr-audio":
        result = prepare_cloud_asr_audio(
            args.media_path,
            output_path=args.output_path or None,
            bitrate_kbps=args.bitrate_kbps,
            sample_rate_hz=args.sample_rate_hz,
            channels=args.channels,
            execute=args.execute,
            timeout_seconds=args.timeout_seconds,
            receipt_bundle_dir=args.receipt_bundle_dir or None,
        )
    elif args.command == "prepare-cloud-asr-chunks":
        result = prepare_asr_vad_chunks(
            args.media_path,
            args.vad_json,
            args.output_dir,
            bitrate_kbps=args.bitrate_kbps,
            sample_rate_hz=args.sample_rate_hz,
            channels=args.channels,
            max_request_seconds=args.max_request_seconds,
            context_padding_seconds=args.context_padding_seconds,
            execute=args.execute,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.command == "asr-vad-activity-audit":
        result = audit_asr_vad_audio_activity(
            args.media_path,
            args.vad_json,
            output_path=args.output_path or None,
            duration_seconds=args.duration_seconds or None,
            noise_db=args.noise_db,
            minimum_silence_seconds=args.minimum_silence_seconds,
            minimum_uncovered_seconds=args.minimum_uncovered_seconds,
            execute=args.execute,
            write=not args.no_write,
        )
    elif args.command == "silero-vad-candidate":
        result = run_silero_vad_candidate(
            args.media_path,
            output_path=args.output_path or None,
            threshold=args.threshold,
            neg_threshold=args.neg_threshold,
            min_speech_duration_ms=args.min_speech_duration_ms,
            max_speech_duration_seconds=args.max_speech_duration_seconds or None,
            min_silence_duration_ms=args.min_silence_duration_ms,
            speech_pad_ms=args.speech_pad_ms,
            execute=args.execute,
            write=not args.no_write,
        )
    elif args.command == "asr-vad-independent-crosscheck":
        result = crosscheck_asr_vad_with_independent_candidate(
            args.authoritative_vad,
            args.candidate_vad,
            activity_audit_path=args.activity_audit or None,
            output_path=args.output_path or None,
            minimum_gap_seconds=args.minimum_gap_seconds,
            write=not args.no_write,
        )
    elif args.command == "asr-vad-profile-compare":
        result = compare_asr_vad_profiles(
            args.authoritative_vad,
            args.permissive_vad,
            args.activity_audit,
            labels_path=args.labels_path or None,
            output_path=args.output_path or None,
            minimum_support_ratio=args.minimum_support_ratio,
            write=not args.no_write,
        )
    elif args.command == "asr-chunk-batch-workflow":
        result = build_asr_chunk_batch_workflow(
            args.chunk_manifest,
            args.consent_path,
            output_path=args.output_path or None,
            bundle_dir=args.bundle_dir or None,
            activity_audit_path=args.activity_audit or None,
            max_parallel_global=args.max_parallel_global,
            max_parallel_per_destination=args.max_parallel_per_destination,
            write=not args.no_write,
        )
    elif args.command == "asr-chunk-business-workflow":
        policy = TrustedModelConnectorPolicy.from_environment(
            default_root=Path(__file__).resolve().parents[2]
        )
        result = build_asr_chunk_business_workflow(
            args.chunk_manifest,
            args.authorization_path,
            stage_id=args.stage_id,
            producer=args.producer,
            lineage_input_paths=args.lineage_input,
            output_path=args.output_path or None,
            bundle_dir=args.bundle_dir or None,
            activity_audit_path=args.activity_audit or None,
            max_parallel_global=args.max_parallel_global,
            max_parallel_per_destination=args.max_parallel_per_destination,
            policy=policy,
            write=not args.no_write,
        )
    elif args.command == "asr-chunk-batch-submit":
        result = submit_asr_chunk_batch_workflow(
            args.workflow_path,
            broker_url=args.broker_url,
            execute=args.execute,
        )
    elif args.command == "asr-chunk-batch-status":
        result = query_asr_chunk_batch_status(
            args.job_id,
            broker_url=args.broker_url,
            output_path=args.output_path or None,
        )
    elif args.command == "asr-chunk-batch-merge":
        result = merge_asr_chunk_batch_reports(
            args.workflow_path,
            args.execution_report,
            batch_status_path=args.batch_status_path or None,
            output_dir=args.output_dir or None,
            title=args.title,
            prepare_alignment_plan=args.prepare_alignment_plan,
            alignment_language=args.alignment_language,
            alignment_model=args.alignment_model,
            write=not args.no_write,
        )
    elif args.command == "plan-cloud-asr":
        result = plan_cloud_asr_run(
            args.workspace_dir,
            args.media_path,
            provider_config=_provider_config_arg(args.provider_config),
            model=args.model,
            language=args.language,
            prompt=args.prompt,
        )
    elif args.command == "run-cloud-asr-plan":
        result = run_cloud_asr_plan(
            args.plan_json,
            provider_config=_provider_config_arg(args.provider_config),
            execute=args.execute,
            normalize=not args.no_normalize,
        )
    elif args.command == "prepare-video-source":
        result = prepare_video_source(args.video_or_url, args.workspace_dir, execute=args.execute)
    elif args.command == "import-page-metadata":
        result = import_page_metadata(args.bundle_dir, args.metadata_json, write=not args.no_write)
    elif args.command == "local-runtime-preflight":
        result = local_runtime_preflight(args.project_root or None)
    elif args.command == "prepare-local-video-run":
        result = prepare_local_video_run(
            args.media_path,
            args.output_dir,
            title=args.title,
            copy_media=args.copy_media,
            plan_asr=not args.no_plan_asr,
            execute_asr=args.execute_asr,
            asr_preset=args.asr_preset,
            asr_model=args.asr_model,
            transcript_path=args.transcript_path,
            build_initial_bundle=args.build_initial_bundle,
            sample_interval=args.sample_interval,
            max_frames=args.max_frames,
            sample_mode=args.sample_mode,
            detect_scenes=not args.no_scene_detect,
            extract_frames=not args.no_frame_extract,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.command == "openclaw-video-plan":
        result = openclaw_video_plan(
            args.url_or_text,
            output_dir=args.output_dir,
            vdo_root=args.vdo_root,
            vdo_output_dir=args.vdo_output_dir,
            backend=args.backend,
            write_manifests=not args.no_write_manifests,
            include_manifests=args.include_manifests,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.command == "openclaw-video-ingest":
        result = openclaw_video_ingest(
            args.media_path,
            workspace=args.workspace,
            title=args.title,
            copy_media=args.copy_media,
            plan_asr=not args.no_plan_asr,
            execute_asr=args.execute_asr,
            asr_preset=args.asr_preset,
            asr_model=args.asr_model,
            transcript_path=args.transcript_path,
            build_initial_bundle=not args.no_build_initial_bundle,
            sample_interval=args.sample_interval,
            max_frames=args.max_frames,
            sample_mode=args.sample_mode,
            detect_scenes=not args.no_scene_detect,
            extract_frames=not args.no_frame_extract,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.command == "openclaw-video-link":
        result = openclaw_video_link(
            args.url_or_text,
            output_dir=args.output_dir,
            vdo_root=args.vdo_root,
            vdo_output_dir=args.vdo_output_dir,
            backend=args.backend,
            allow_download=args.allow_download,
            actor_id=args.actor_id,
            confirm_download=args.confirm_download,
            confirm_sensitive=args.confirm_sensitive,
            ingest_after_download=args.ingest_after_download,
            downloaded_media_path=args.downloaded_media_path,
            workspace=args.workspace,
            title=args.title,
            max_frames=args.max_frames,
            sample_interval=args.sample_interval,
            sample_mode=args.sample_mode,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.command == "openclaw-bridge-status":
        result = openclaw_bridge_status(timeout_seconds=args.timeout_seconds, check_health=not args.no_health, check_task=not args.no_task)
    elif args.command == "openclaw-bridge-doctor":
        result = openclaw_bridge_doctor(timeout_seconds=args.timeout_seconds, project_root=args.project_root)
    elif args.command == "openclaw-live-smoke":
        result = openclaw_live_smoke(
            bundle_dir=args.bundle_dir,
            compose_path=args.compose_path,
            host_root=args.host_root,
            container_root=args.container_root,
            timeout_seconds=args.timeout_seconds,
            output_dir=args.output_dir,
            semantic_batch_input=args.semantic_batch_input,
            semantic_target_bundle_count=args.semantic_target_bundle_count,
            semantic_limit=args.semantic_limit,
            write_report=args.write_report,
        )
    elif args.command == "openclaw-docker-contract-check":
        result = openclaw_docker_contract_check(args.compose_path, host_root=args.host_root, container_root=args.container_root)
    elif args.command == "openclaw-video-from-vdo-handoff":
        result = vdo_handoff_plan(
            manifest_path=args.manifest_path,
            summary_path=args.summary_path,
            review_checklist_path=args.review_checklist_path,
            media_path=args.media_path,
            host_root=args.host_root,
            container_root=args.container_root,
            workspace=args.workspace,
            title=args.title,
        )
    elif args.command == "openclaw-video-ingest-vdo-handoff":
        result = ingest_vdo_handoff(
            handoff_path=args.handoff_path,
            manifest_path=args.manifest_path,
            summary_path=args.summary_path,
            review_checklist_path=args.review_checklist_path,
            media_path=args.media_path,
            host_root=args.host_root,
            container_root=args.container_root,
            workspace=args.workspace,
            title=args.title,
            execute=args.execute,
            max_frames=args.max_frames,
            sample_interval=args.sample_interval,
            sample_mode=args.sample_mode,
        )
    elif args.command == "acceptance-run":
        result = run_acceptance_run(
            args.media_path,
            args.output_dir,
            title=args.title,
            copy_media=args.copy_media,
            execute_asr=args.execute_asr,
            asr_preset=args.asr_preset,
            asr_model=args.asr_model,
            transcript_path=args.transcript_path,
            build_initial_bundle=not args.no_build_initial_bundle,
            sample_interval=args.sample_interval,
            max_frames=args.max_frames,
            sample_mode=args.sample_mode,
            detect_scenes=not args.no_scene_detect,
            extract_frames=not args.no_frame_extract,
            execute_temporal_groups=args.execute_temporal_groups,
            execute_vision=args.execute_vision,
            execute_ebook_pipeline=args.execute_ebook_pipeline,
            semantic_limit=args.semantic_limit,
            temporal_limit=args.temporal_limit,
            frame_count=args.frame_count,
            provider_config=_provider_config_arg(args.provider_config),
            confirm_vision_calls=args.confirm_vision_calls,
            confirm_vision_indexes=args.confirm_vision_indexes,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.command == "acceptance-bundle-run":
        result = run_acceptance_bundle(
            args.bundle_dir,
            output_dir=args.output_dir,
            title=args.title,
            execute_temporal_groups=args.execute_temporal_groups,
            execute_vision=args.execute_vision,
            execute_ebook_pipeline=args.execute_ebook_pipeline,
            semantic_limit=args.semantic_limit,
            temporal_limit=args.temporal_limit,
            frame_count=args.frame_count,
            provider_config=_provider_config_arg(args.provider_config),
            confirm_vision_calls=args.confirm_vision_calls,
            confirm_vision_indexes=args.confirm_vision_indexes,
        )
    elif args.command == "batch-run":
        result = batch_video_knowledge_run(
            args.batch_manifest,
            resume=args.resume,
            force_reexport=args.force_reexport,
            execute_asr=args.execute_asr,
            execute_temporal_groups=args.execute_temporal_groups,
            execute_vision=args.execute_vision,
            execute_ebook_pipeline=args.execute_ebook_pipeline,
            semantic_limit=args.semantic_limit,
            temporal_limit=args.temporal_limit,
            frame_count=args.frame_count,
            timeout_seconds=args.timeout_seconds,
            write=not args.no_write,
        )
    elif args.command == "batch-repair-run":
        result = batch_repair_run(
            args.batch_manifest_or_summary,
            execute=args.execute,
            limit=args.limit,
            max_rounds=args.max_rounds,
            allow_asr=args.allow_asr,
            allow_vision=args.allow_vision,
            allow_ocr=args.allow_ocr,
            write=not args.no_write,
        )
    elif args.command == "plan-asr":
        if args.alignment_chunk_seconds is not None and args.preset != "qwen3-forced-aligner":
            raise ValueError("--alignment-chunk-seconds is only supported by qwen3-forced-aligner")
        result = plan_asr_run(
            args.workspace_dir,
            args.media_path,
            preset=args.preset,
            language=args.language,
            model=args.model or None,
            punc_model=None if args.punc_model == "__default__" else args.punc_model,
            spk_model=args.spk_model or None,
            hotword=args.hotword or None,
            use_itn=not args.no_use_itn,
            merge_vad=not args.no_merge_vad,
            merge_length_s=args.merge_length_s,
            vad_max_single_segment_time_ms=args.vad_max_single_segment_time_ms,
            chunk_boundary_mode=args.chunk_boundary_mode,
            chunk_overlap_seconds=args.chunk_overlap_seconds,
            qwen_timestamp_basis=args.qwen_timestamp_basis,
            vad_json=args.vad_json or None,
            transcript_path=args.transcript_path or None,
            alignment_chunk_seconds=300 if args.alignment_chunk_seconds is None else args.alignment_chunk_seconds,
        )
    elif args.command == "speaker-global-align":
        result = build_speaker_global_alignment(
            args.chunked_output,
            similarity_threshold=args.similarity_threshold,
            overlap_agreement_threshold=args.overlap_agreement_threshold,
            max_speakers=args.max_speakers,
            expected_speaker_count=args.expected_speakers,
            write=not args.no_write,
        )
    elif args.command == "speaker-center-sidecar":
        result = build_campplus_speaker_center_sidecar(
            args.chunked_output,
            args.media_path,
            args.output_dir,
            expected_speaker_count=args.expected_speakers,
            device=args.device,
            max_evidence_seconds=args.max_evidence_seconds,
            execute=args.execute,
            write=not args.no_write,
        )
    elif args.command == "speaker-shared-session-align":
        result = build_shared_session_speaker_alignment(
            args.private_centers,
            args.candidate_transcripts,
            args.output_dir,
            expected_speaker_count=args.expected_speakers,
            confirm_shared_participant_set=args.confirm_shared_participant_set,
            write=not args.no_write,
        )
    elif args.command == "speaker-voiceprint":
        if args.action == "enroll":
            if not args.private_alignment or not args.source_id:
                raise ValueError("enroll requires --private-alignment and --source-id")
            result = enroll_local_voiceprints(
                args.private_alignment,
                args.registry,
                source_id=args.source_id,
                confirm_local_biometric_storage=args.confirm_local_biometric_storage,
                write=not args.no_write,
            )
        elif args.action == "match":
            if not args.private_alignment:
                raise ValueError("match requires --private-alignment")
            result = match_local_voiceprints(
                args.private_alignment,
                args.registry,
                similarity_threshold=args.similarity_threshold,
            )
        elif args.action == "bind-role":
            if not args.voiceprint_id or not args.role_label:
                raise ValueError("bind-role requires --voiceprint-id and --role-label")
            result = bind_local_voiceprint_role(
                args.registry,
                args.voiceprint_id,
                args.role_label,
                confirm_role_binding=args.confirm_role_binding,
                write=not args.no_write,
            )
        else:
            if not args.voiceprint_id:
                raise ValueError("delete requires --voiceprint-id")
            result = delete_local_voiceprint(
                args.registry,
                args.voiceprint_id,
                confirm_delete=args.confirm_delete,
                write=not args.no_write,
            )
    elif args.command == "asr-retry-snippets":
        result = prepare_asr_retry_snippets(
            args.media_path,
            args.quality_report_json,
            args.output_dir,
            ffmpeg_path=args.ffmpeg,
            execute=args.execute,
        )
    elif args.command == "plan-local-targeted-asr-evidence":
        result = build_local_targeted_asr_plan(
            args.bundle_dir,
            input_pack=args.input_pack or None,
            max_windows=args.max_windows,
            padding_seconds=args.padding_seconds,
            write=not args.no_write,
        )
    elif args.command == "run-local-targeted-asr-evidence":
        result = run_local_targeted_asr_evidence(
            args.bundle_dir,
            media_path=args.media_path or None,
            input_plan=args.input_plan or None,
            preset=args.preset,
            language=args.language,
            model=args.model or None,
            timeout_seconds=args.timeout_seconds,
            execute=args.execute,
            allow_cpu=args.allow_cpu,
            max_windows=args.max_windows,
            padding_seconds=args.padding_seconds,
            write=not args.no_write,
        )
    elif args.command == "asr-local-targeted-evidence":
        result = build_local_targeted_asr_evidence(
            args.snippet_manifest,
            args.raw_output,
            output_json=args.output_json,
            require_gpu=not args.allow_cpu,
            write=args.write,
        )
    elif args.command == "asr-ab-sample-plan":
        result = plan_asr_ab_sample(
            args.workspace_dir,
            args.media_path,
            sample_start_seconds=args.sample_start_seconds,
            duration_seconds=args.duration_seconds,
            language=args.language,
            cloud_provider_config=_provider_config_arg(args.cloud_provider_config) if args.cloud_provider_config else None,
            write=not args.no_write,
        )
    elif args.command == "asr-ab-sample-run":
        result = run_asr_ab_sample(
            args.workspace_dir,
            args.media_path or "",
            plan_json=args.plan_json or "",
            sample_start_seconds=args.sample_start_seconds,
            duration_seconds=args.duration_seconds,
            language=args.language,
            execute_sample=args.execute_sample,
            execute_local=args.execute_local,
            execute_cloud=args.execute_cloud,
            cloud_provider_config=_provider_config_arg(args.cloud_provider_config) if args.cloud_provider_config else None,
            variants=_csv_arg(args.variants),
            timeout_seconds=args.timeout_seconds,
            write=not args.no_write,
        )
    elif args.command == "asr-ab-compare":
        result = compare_asr_ab_sample(
            args.run_json,
            reference_transcript=args.reference_transcript or None,
            start_seconds=args.start_seconds,
            end_seconds=args.end_seconds,
            write=not args.no_write,
        )
    elif args.command == "transcript-reference-window":
        result = export_transcript_reference_window(
            args.transcript_path,
            args.output_json,
            start_seconds=args.start_seconds,
            end_seconds=args.end_seconds,
            rebase_timestamps=not args.absolute_timestamps,
            human_corrections_json=args.human_corrections_json or None,
            write=not args.no_write,
        )
    elif args.command == "transcript-candidate-recall-benchmark":
        result = benchmark_transcript_candidate_recall(
            args.bundle_dir,
            reference_transcript=args.reference_transcript or None,
            source_transcript=args.source_transcript or None,
            target_pairs_json=args.target_pairs_json or None,
            asr_ab_run_json=args.asr_ab_run_json or None,
            start_seconds=args.start_seconds,
            end_seconds=args.end_seconds,
            write=not args.no_write,
        )
    elif args.command == "plan-local-asr-service":
        result = plan_local_asr_service_run(
            args.workspace_dir,
            args.media_path,
            provider_config=_provider_config_arg(args.provider_config) if args.provider_config else None,
            model=args.model or "",
            language=args.language,
            prompt=args.prompt,
        )
    elif args.command == "run-local-asr-service-plan":
        result = run_local_asr_service_plan(
            args.plan_json,
            provider_config=_provider_config_arg(args.provider_config) if args.provider_config else None,
            execute=args.execute,
            normalize=not args.no_normalize,
            allow_remote=args.allow_remote,
        )
    elif args.command == "plan-whisperx-alignment":
        result = plan_whisperx_alignment(args.workspace_dir, args.media_path, language=args.language, model=args.model or None)
    elif args.command == "run-whisperx-alignment":
        result = run_whisperx_alignment(
            args.workspace_dir,
            args.media_path,
            language=args.language,
            model=args.model or "large-v3",
            execute=args.execute,
            timeout_seconds=args.timeout_seconds,
            write=not args.no_write,
        )
    elif args.command == "run-asr-plan":
        result = run_asr_plan(
            args.plan_json,
            execute=args.execute,
            timeout_seconds=args.timeout_seconds,
            resume=args.resume,
            progress_callback=stderr_progress_callback,
        )
    elif args.command == "asr-smoke":
        result = asr_smoke(
            args.media_path,
            output_dir=args.output_dir,
            preset=args.preset,
            model=args.model or "",
            language=args.language,
            duration_seconds=args.duration_seconds,
            execute=args.execute,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.command == "normalize-asr":
        result = normalize_asr_output(args.workspace_dir, args.input_path, provider=args.provider, title=args.title)
    elif args.command == "resegment-transcript":
        result = resegment_transcript(
            args.workspace_dir,
            args.input_path,
            media_path=args.media_path,
            duration_seconds=args.duration_seconds,
            target_seconds=args.target_seconds,
            max_chars=args.max_chars,
            title=args.title,
        )
    elif args.command == "postprocess-asr-transcript":
        result = postprocess_asr_transcript(
            args.bundle_dir,
            input_path=args.input_path or None,
            target_seconds=args.target_seconds,
            max_chars=args.max_chars,
            punctuation_mode=args.punctuation_mode,
            segment_policy=args.segment_policy,
            set_corrected=not args.no_set_corrected,
            write=not args.no_write,
            progress_callback=stderr_progress_callback,
        )
    elif args.command == "translate-transcript-to-mandarin":
        result = translate_transcript_to_mandarin(
            args.bundle_dir,
            source_path=args.source_path or None,
            input_json=args.input_json or None,
            batch_size=args.batch_size,
            route_id=args.route_id,
            route_revision=args.route_revision,
            direct_lmstudio=args.direct_lmstudio,
            execute=args.execute,
            write=not args.no_write,
        )
    elif args.command == "run-ready-pipeline":
        result = run_ready_lecture_pipeline(args.workspace_dir, extractor=args.extractor)
    elif args.command == "run-extractor-plan":
        result = run_extractor_plan(args.plan_json, args.extractor, execute=args.execute, timeout_seconds=args.timeout_seconds)
    elif args.command == "extractor-run-log":
        result = extractor_run_log(args.workspace_dir)
    elif args.command == "attach-peepshow-output":
        result = attach_peepshow_output_to_bundle(args.bundle_dir, args.output_dir, write=not args.no_write)
    elif args.command == "refresh-lecture-review":
        result = refresh_lecture_review_outputs(
            args.project,
            args.review_json,
            webui_output_dir=args.webui_output_dir,
            vault=args.vault,
            folder=args.folder,
            target=args.target,
            allow_blocked_export=args.allow_blocked_export,
        )
    elif args.command == "apply-review-notes":
        result = apply_review_notes_to_bundle(args.bundle_dir, review_json=args.review_json, write=not args.no_write)
    elif args.command == "validate-review-notes":
        result = validate_review_notes_for_bundle(args.bundle_dir, review_json=args.review_json)
    elif args.command == "prepare-review-session":
        result = prepare_review_session(
            args.bundle_dir,
            run_id=args.run_id,
            dependency_versions=read_json(Path(args.dependency_versions)) if args.dependency_versions else None,
            refresh=not args.no_refresh,
            limit=args.limit,
            offset=args.offset,
            reason=args.reason,
            group_by=args.group_by,
            include_closed=args.include_closed,
            output_prefix=args.output_prefix,
        )
    elif args.command == "review-closure-status":
        result = review_closure_status(args.bundle_dir, write=not args.no_write)
    elif args.command == "run-video-frame-router":
        result = run_video_frame_router(
            args.bundle_dir,
            input_json=args.input_json,
            content_profile=args.content_profile,
            write=not args.no_write,
        )
    elif args.command == "import-tagger-annotations":
        result = import_tagger_annotations(args.bundle_dir, args.tagger_json, source=args.source, write=not args.no_write)
    elif args.command == "run-general-tagger":
        result = run_general_tagger(
            args.bundle_dir,
            source_root=args.source_root or None,
            checkpoint_path=args.checkpoint_path or None,
            device=args.device,
            prefer_language=args.prefer_language,
            limit=args.limit,
            frame_mode=args.frame_mode,
            execute=args.execute,
            import_annotations=not args.no_import,
            write=not args.no_write,
        )
    elif args.command == "run-ocr-backfill":
        result = run_ocr_backfill(
            args.bundle_dir,
            input_json=args.input_json,
            execute=args.execute,
            language=args.language,
            captiocr_root=args.captiocr_root,
            limit=args.limit,
            apply_mode=args.apply_mode,
            engine=args.engine,
        )
    elif args.command == "run-screen-text-recovery":
        result = run_screen_text_recovery(
            args.bundle_dir,
            execute_crops=args.execute_crops,
            execute_ocr=args.execute_ocr,
            input_json=args.input_json,
            language=args.language,
            captiocr_root=args.captiocr_root,
            engine=args.engine,
            limit=args.limit,
            indexes=_int_csv_arg(args.indexes),
            profile=args.profile, max_regions=args.max_regions, max_attempts=args.max_attempts,
            write=not args.no_write,
        )
    elif args.command == "high-res-tile-plan":
        result = run_high_res_tile_plan(
            args.bundle_dir,
            execute_tiles=args.execute_tiles,
            indexes=_int_csv_arg(args.indexes),
            limit=args.limit,
            tile_size=args.tile_size,
            overlap=args.overlap,
            max_tiles_per_image=args.max_tiles_per_image,
            include_routes=_csv_arg(args.include_routes),
            write=not args.no_write,
        )
    elif args.command == "tile-result-import-build":
        result = build_tile_result_import(
            args.bundle_dir,
            results_dir=args.results_dir,
            output_json=args.output_json,
            default_source=args.default_source,
            default_confidence=args.default_confidence,
            write=not args.no_write,
        )
    elif args.command == "tile-result-merge":
        result = run_tile_result_merge(
            args.bundle_dir,
            input_json=args.input_json,
            execute=args.execute,
            min_confidence=args.min_confidence,
            write=not args.no_write,
        )
    elif args.command == "run-frame-recapture-plan":
        result = run_frame_recapture_plan(
            args.bundle_dir,
            execute=args.execute,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.command == "run-visual-structure":
        result = run_visual_structure_plan(
            args.bundle_dir,
            input_json=args.input_json,
            execute_ebook_pipeline=args.execute_ebook_pipeline,
            include_routes=_csv_arg(args.include_routes),
            timeout_seconds=args.timeout_seconds,
            indexes=_int_csv_arg(args.indexes),
            limit=args.limit,
        )
    elif args.command == "run-visual-structure-ebook-batches":
        result = run_visual_structure_ebook_batches(
            args.bundle_dir,
            execute=args.execute,
            include_routes=_csv_arg(args.include_routes),
            indexes=_int_csv_arg(args.indexes),
            batch_size=args.batch_size,
            timeout_seconds=args.timeout_seconds,
            resume=not args.no_resume,
            write=not args.no_write,
        )
    elif args.command == "repair-ebook-artifact-text":
        result = repair_ebook_artifact_text(args.bundle_dir, write=not args.no_write)
    elif args.command == "run-multimodal-frame-analysis":
        result = run_multimodal_frame_analysis(
            args.bundle_dir,
            input_json=args.input_json,
            execute=args.execute,
            provider_config=_provider_config_arg(args.provider_config),
            limit=args.limit,
            indexes=_int_csv_and_repeat_arg(args.indexes, args.index),
            confirm_vision_calls=args.confirm_vision_calls,
            confirm_vision_indexes=args.confirm_vision_indexes,
            image_probe_max_edge=args.image_probe_max_edge,
            image_probe_jpeg_quality=args.image_probe_jpeg_quality,
            vision_retries=args.vision_retries,
            vision_retry_delay_seconds=args.vision_retry_delay_seconds,
            execution_actor=args.execution_actor,
            export_consent=args.export_consent or None,
            max_tokens=args.max_tokens or None,
        )
    elif args.command == "run-temporal-frame-groups":
        result = run_temporal_frame_groups(
            args.bundle_dir,
            execute=args.execute,
            frame_count=args.frame_count,
            window_seconds=args.window_seconds,
            include_routes=_csv_arg(args.include_routes),
            indexes=_int_csv_arg(args.indexes),
            limit=args.limit,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.command == "run-temporal-visual-analysis":
        result = run_temporal_visual_analysis(
            args.bundle_dir,
            input_json=args.input_json,
            execute=args.execute,
            frame_count=args.frame_count,
            limit=args.limit,
            indexes=_int_csv_and_repeat_arg(args.indexes, args.index),
            provider_config=_provider_config_arg(args.provider_config),
            confirm_vision_calls=args.confirm_vision_calls,
            confirm_vision_indexes=args.confirm_vision_indexes,
            image_probe_max_edge=args.image_probe_max_edge,
            image_probe_jpeg_quality=args.image_probe_jpeg_quality,
            vision_retries=args.vision_retries,
            vision_retry_delay_seconds=args.vision_retry_delay_seconds,
            execution_actor=args.execution_actor,
            export_consent=args.export_consent or None,
            max_tokens=args.max_tokens or None,
        )
    elif args.command == "prepare-summary-claim-review":
        result = prepare_summary_claim_review(args.bundle_dir, write=not args.no_write)
    elif args.command == "import-summary-claim-review":
        result = import_summary_claim_review(args.bundle_dir, args.input_json)
    elif args.command == "summary-claim-review-status":
        result = summary_claim_review_status(args.bundle_dir)
    elif args.command == "run-codex-vision":
        result = run_codex_vision(
            args.bundle_dir, profile=args.profile_json, state_json=args.state_json,
            indexes=_int_csv_and_repeat_arg(args.indexes, args.index) or None,
            execute=args.execute, max_batches=args.max_batches, retry_failed=args.retry_failed,
        )
    elif args.command == "codex-vision-catalog":
        result = codex_vision_catalog(args.profile_json)
    elif args.command == "prepare-codex-vision-session":
        result = prepare_codex_vision_session(
            args.bundle_dir,
            kind=args.kind,
            indexes=_int_csv_and_repeat_arg(args.indexes, args.index),
            limit=args.limit,
            max_frames=args.max_frames,
            model_requested=args.model_requested,
            quality=args.quality,
            executor=args.executor,
            write=not args.no_write,
        )
    elif args.command == "import-codex-vision-results":
        result = import_codex_vision_results(
            args.bundle_dir,
            input_json=args.input_json,
            write=not args.no_write,
        )
    elif args.command == "test-vision-provider":
        result = test_vision_provider(_provider_config_arg(args.provider_config), image_paths=_csv_arg(args.image_paths) or [])
    elif args.command == "text-llm-provider-smoke":
        result = text_llm_provider_smoke(_provider_config_arg(args.provider_config), execute=args.execute, prompt=args.prompt)
    elif args.command == "online-model-api":
        result = online_model_api_call(
            args.model_type,
            provider_config=_provider_config_arg(args.provider_config),
            prompt=args.prompt,
            input_text=args.input_text,
            image_paths=_csv_arg(args.image_paths) or [],
            audio_path=args.audio_path,
            execute=args.execute,
            output_dir=args.output_dir or None,
            write=not args.no_write,
        )
    elif args.command == "online-model-api-matrix":
        result = online_model_api_matrix(
            provider_config=_provider_config_arg(args.provider_config),
            output_dir=args.output_dir or None,
            write=not args.no_write,
        )
    elif args.command == "model-task-coverage-audit":
        result = model_task_coverage_audit(output_dir=args.output_dir or None, write=not args.no_write)
    elif args.command == "run-term-arbitration-model":
        result = run_term_arbitration_model(
            args.bundle_dir, provider_config=_provider_config_arg(args.provider_config),
            execute=args.execute, max_terms=args.max_terms, min_confidence=args.min_confidence,
            max_tokens=args.max_tokens, temperature=args.temperature, write=not args.no_write,
        )
    elif args.command == "run-bilinote-mind-map-model":
        result = run_bilinote_mind_map_model(
            args.bundle_dir, provider_config=_provider_config_arg(args.provider_config), execute=args.execute,
            title=args.title, max_chars=args.max_chars, limit=args.limit, max_tokens=args.max_tokens,
            temperature=args.temperature, write=not args.no_write,
        )
    elif args.command == "volcengine-model-task-matrix":
        result = run_volcengine_model_task_matrix(
            execute=args.execute,
            output_dir=args.output_dir or None,
            models=_csv_arg(args.models),
            tasks=_csv_arg(args.tasks),
            timeout_seconds=args.timeout_seconds,
            write=not args.no_write,
        )
    elif args.command == "volcengine-model-routing":
        result = volcengine_model_routing(
            route=args.route,
            output_dir=args.output_dir or None,
            write=not args.no_write,
        )
    elif args.command == "bilinote-mind-map-prompt-pack":
        if args.bundle_dir:
            result = build_bundle_mind_map_prompt_pack(args.bundle_dir, title=args.title, max_chars=args.max_chars, write=not args.no_write)
        else:
            transcript = Path(args.transcript_path).read_text(encoding="utf-8") if args.transcript_path else args.transcript
            result = build_mind_map_prompt_pack(title=args.title, transcript=transcript, max_chars=args.max_chars)
    elif args.command == "prepare-transcript-edit-session":
        result = prepare_transcript_edit_session(args.bundle_dir, write=not args.no_write)
    elif args.command == "apply-transcript-edits":
        result = apply_transcript_edits(args.bundle_dir, edits_json=args.edits_json, write=not args.no_write)
    elif args.command == "prepare-subtitle-editor":
        result = prepare_subtitle_editor(args.bundle_dir, write=not args.no_write)
    elif args.command == "validate-subtitle-review":
        result = validate_subtitle_review(args.bundle_dir, args.review_json)
    elif args.command == "apply-subtitle-review":
        result = apply_subtitle_review(
            args.bundle_dir,
            review_json=args.review_json,
            write=not args.no_write,
        )
    elif args.command == "run-temporal-tag-delta":
        result = run_temporal_tag_delta(
            args.bundle_dir,
            input_json=args.input_json,
            execute_tagger=args.execute_tagger,
            source_root=args.source_root or None,
            checkpoint_path=args.checkpoint_path or None,
            device=args.device,
            prefer_language=args.prefer_language,
            indexes=args.indexes or None,
            frame_count=args.frame_count,
            limit=args.limit,
            min_frames=args.min_frames,
            resume=args.resume or None,
            write=not args.no_write,
        )
    elif args.command == "temporal-tag-restore":
        result = apply_temporal_tag_delta_restore(
            args.bundle_dir,
            args.plan_json,
            execute=args.execute,
            confirm_run_id=args.confirm_run_id,
        )
    elif args.command == "transcript-correction-pack":
        result = build_transcript_correction_pack(
            args.bundle_dir,
            input_json=args.input_json or None,
            provider_config=_provider_config_arg(args.provider_config),
            execute=args.execute,
            max_segments=args.max_segments,
            max_chunk_chars=args.max_chunk_chars,
            write=not args.no_write,
        )
    elif args.command == "evidence-conflict-index":
        result = build_evidence_conflict_index(args.bundle_dir, input_json=args.input_json or None, limit=args.limit, write=not args.no_write)
    elif args.command == "transcript-source-arbitration":
        result = arbitrate_transcript_sources(
            args.bundle_dir,
            platform_subtitle=args.platform_subtitle or None,
            subtitle=args.subtitle or None,
            asr_json=args.asr_json or None,
            glossary_json=args.glossary_json or None,
            min_confidence=args.min_confidence,
            promote=not args.no_promote,
            write=not args.no_write,
        )
    elif args.command == "readable-transcript-llm-polish":
        result = run_readable_transcript_llm_polish(
            args.bundle_dir,
            provider_config=_provider_config_arg(args.provider_config),
            input_json=args.input_json or None,
            execute=args.execute,
            agent_substitute=args.agent_substitute or args.codex_substitute,
            agent_name=args.agent_name,
            codex_substitute=args.codex_substitute,
            promote=args.promote,
            max_segments_per_batch=args.max_segments_per_batch,
            max_prompt_chars=args.max_prompt_chars,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            write=not args.no_write,
        )
    elif args.command == "agent-readable-transcript-rewrite":
        result = run_agent_readable_transcript_rewrite(
            args.bundle_dir,
            input_json=args.input_json or None,
            agent_name=args.agent_name,
            source_path=args.source_path or None,
            promote=args.promote,
            write=not args.no_write,
        )
    elif args.command == "transcript-quality-gate":
        result = run_transcript_quality_gate(
            args.bundle_dir,
            input_path=args.input_path or None,
            reference_path=args.reference_path or None,
            baseline_path=args.baseline_path or None,
            min_punctuation_per_1000=args.min_punctuation_per_1000,
            max_punctuation_per_1000=args.max_punctuation_per_1000,
            max_cer=args.max_cer,
            min_entity_accuracy=args.min_entity_accuracy,
            max_overcorrection_rate=args.max_overcorrection_rate,
            require_speaker_diarization=True if args.require_speaker_diarization else None,
            min_speaker_count=args.min_speaker_count,
            write=not args.no_write,
        )
    elif args.command == "transcript-evidence-correction-pipeline":
        result = run_transcript_evidence_correction_pipeline(
            args.bundle_dir,
            platform_subtitle=args.platform_subtitle or None,
            subtitle=args.subtitle or None,
            asr_json=args.asr_json or None,
            secondary_asr_json=args.secondary_asr_json or None,
            additional_secondary_asr_json=args.additional_secondary_asr_json,
            consensus_agreement_threshold=args.consensus_agreement_threshold,
            execute_consensus_clips=args.execute_consensus_clips,
            media_path=args.media_path or None,
            asr_mode=args.asr_mode,
            execute_local_asr=args.execute_local_asr,
            speech_activity_json=args.speech_activity_json or None,
            diarization_json=args.diarization_json or None,
            execute_local_targeted_asr=args.execute_local_targeted_asr,
            local_targeted_asr_preset=args.local_targeted_asr_preset,
            local_targeted_asr_model=args.local_targeted_asr_model or None,
            local_targeted_asr_timeout_seconds=args.local_targeted_asr_timeout_seconds,
            local_targeted_asr_allow_cpu=args.local_targeted_asr_allow_cpu,
            glossary_json=args.glossary_json or None,
            provider_config=_provider_config_arg(args.provider_config),
            quality_profile=args.quality_profile,
            execute_llm=args.execute_llm,
            use_agent_substitute=not (args.no_agent_substitute or args.no_codex_substitute),
            agent_name=args.agent_name,
            use_codex_substitute=None if not args.no_codex_substitute else False,
            run_readable_llm=not args.no_readable_llm,
            execute_readable_llm=args.execute_readable_llm,
            promote_readable_llm=args.promote_readable_llm,
            readable_max_segments_per_batch=args.readable_max_segments_per_batch,
            readable_max_prompt_chars=args.readable_max_prompt_chars,
            readable_max_tokens=args.readable_max_tokens,
            auto_apply_high_confidence=args.auto_apply_high_confidence,
            run_postprocess=not args.no_postprocess,
            run_source_arbitration=not args.no_source_arbitration,
            source_min_confidence=args.source_min_confidence,
            semantic_min_confidence=args.semantic_min_confidence,
            run_agent_readable_rewrite=not (args.no_agent_substitute or args.no_codex_substitute),
            semantic_limit=args.semantic_limit,
            refresh_exports=not args.no_refresh_exports,
            write=not args.no_write,
        )
    elif args.command == "transcript-main-route-status":
        result = transcript_main_route_status(args.bundle_dir, write=not args.no_write)
    elif args.command == "video-moment-index":
        result = build_video_moment_index(
            args.bundle_dir,
            query=args.query,
            target_window_seconds=args.target_window_seconds,
            max_chunk_chars=args.max_chunk_chars,
            top_k=args.top_k,
            write=not args.no_write,
        )
    elif args.command == "long-video-memory-pack":
        result = build_long_video_memory_pack(
            args.bundle_dir,
            target_window_seconds=args.target_window_seconds,
            max_chunk_chars=args.max_chunk_chars,
            long_group_size=args.long_group_size,
            write=not args.no_write,
        )
    elif args.command == "video-rag-pack":
        result = build_video_rag_pack(
            args.bundle_dir,
            query=args.query,
            target_window_seconds=args.target_window_seconds,
            max_chunk_chars=args.max_chunk_chars,
            top_k=args.top_k,
            write=not args.no_write,
        )
    elif args.command == "video-rag-search":
        result = search_video_rag(
            args.bundle_dir,
            query=args.query,
            top_k=args.top_k,
            ensure_pack=not args.no_ensure_pack,
            retrieval_backend=args.retrieval_backend,
            write=not args.no_write,
        )
    elif args.command == "script-clip-candidate-pack":
        result = build_script_clip_candidate_pack(
            args.bundle_dir,
            args.request_json,
            top_k=args.top_k,
            retrieval_backend=args.retrieval_backend,
            context_seconds=args.context_seconds,
            write=not args.no_write,
        )
    elif args.command == "script-clip-alignment-check":
        result = check_script_clip_alignment(
            args.bundle_dir,
            args.review_notes_json,
            args.fine_cut_plan_json,
            candidate_pack_json=args.candidate_pack_json or None,
            write=not args.no_write,
        )
    elif args.command == "content-clip-candidate-pack":
        result = build_content_clip_candidate_pack(
            args.bundle_dir,
            args.request_json,
            top_k=args.top_k,
            retrieval_backend=args.retrieval_backend,
            context_seconds=args.context_seconds,
            write=not args.no_write,
        )
    elif args.command == "content-clip-alignment-check":
        result = check_content_clip_alignment(
            args.bundle_dir,
            args.review_notes_json,
            args.fine_cut_plan_json,
            candidate_pack_json=args.candidate_pack_json or None,
            write=not args.no_write,
        )
    elif args.command == "long-video-fast-segment-plan":
        result = build_long_video_fast_segment_plan(
            args.bundle_dir,
            media_path=args.media_path or None,
            transcript_path=args.transcript_path or None,
            vad_path=args.vad_path or None,
            activity_audit_path=args.activity_audit_path or None,
            profile=args.profile,
            long_silence_seconds=args.long_silence_seconds,
            edge_blank_seconds=args.edge_blank_seconds,
            repeat_similarity=args.repeat_similarity,
            write=not args.no_write,
        )
    elif args.command == "apply-long-video-fast-segment-review":
        result = apply_long_video_fast_segment_review(
            args.bundle_dir,
            args.review_json,
            plan_json=args.plan_json or None,
            write=not args.no_write,
        )
    elif args.command == "render-long-video-fast-segment":
        result = render_long_video_fast_segment(
            args.bundle_dir,
            approved_json=args.approved_json or None,
            output_path=args.output_path or None,
            execute=args.execute,
            timeout_seconds=args.timeout_seconds,
            write=not args.no_write,
        )
    elif args.command == "video-evidence-query-plan":
        result = build_video_evidence_query_plan(
            args.bundle_dir,
            query=args.query,
            coarse_top_k=args.coarse_top_k,
            fine_top_k=args.fine_top_k,
            write=not args.no_write,
        )
    elif args.command == "apply-video-evidence-confirmation":
        result = apply_video_evidence_confirmation(
            args.bundle_dir,
            decisions_json=args.decisions_json,
            plan_json=args.plan_json or None,
            write=not args.no_write,
        )
    elif args.command == "video-rag-service-plan":
        result = video_rag_service_plan(args.bundle_dir, host=args.host, port=args.port, write=not args.no_write)
    elif args.command == "video-rag-serve":
        result = serve_video_rag(args.bundle_dir, host=args.host, port=args.port)
    elif args.command == "external-capability-pack":
        result = build_external_capability_pack(args.bundle_dir, query=args.query, write=not args.no_write)
    elif args.command == "vision-provider-smoke":
        result = vision_provider_smoke(
            provider_config=_provider_config_arg(args.provider_config),
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            bundle_dir=args.bundle_dir,
            single_image=args.single_image,
            multi_image_dir=args.multi_image_dir,
            output_dir=args.output_dir,
            image_probe_max_edge=args.image_probe_max_edge,
            image_probe_jpeg_quality=args.image_probe_jpeg_quality,
            max_images=args.max_images,
            write=not args.no_write,
        )
    elif args.command == "vision-provider-matrix":
        result = vision_provider_matrix(
            providers=_csv_arg(args.providers),
            bundle_dir=args.bundle_dir,
            output_dir=args.output_dir,
            timeout_seconds=args.timeout_seconds,
            single_image=args.single_image,
            multi_image_dir=args.multi_image_dir,
            image_probe_max_edge=args.image_probe_max_edge,
            image_probe_jpeg_quality=args.image_probe_jpeg_quality,
            max_images=args.max_images,
            preferred_provider=args.preferred_provider,
            write=not args.no_write,
        )
    elif args.command == "vision-env-status":
        result = vision_environment_status(
            provider=args.provider,
            model=args.model,
            write_template=args.write_template,
            template_path=args.template_path,
            overwrite=args.overwrite,
        )
    elif args.command == "vision-acceptance-plan":
        result = vision_acceptance_plan(
            args.bundle_dir,
            provider_config=_provider_config_arg(args.provider_config),
            semantic_limit=args.semantic_limit,
            temporal_limit=args.temporal_limit,
            frame_count=args.frame_count,
            write=not args.no_write,
        )
    elif args.command == "visual-ab-benchmark-plan":
        result = build_visual_ab_benchmark_plan(
            args.bundle_dir, limit=args.limit, min_score=args.min_score, write=not args.no_write
        )
    elif args.command == "vision-export-consent-create":
        result = create_vision_export_consent(
            args.bundle_dir,
            provider_config=_provider_config_arg(args.provider_config),
            semantic_indexes=_int_csv_arg(args.semantic_indexes),
            temporal_indexes=_int_csv_arg(args.temporal_indexes),
            max_calls=args.max_calls,
            expires_hours=args.expires_hours,
            image_max_edge=args.image_max_edge,
            image_jpeg_quality=args.image_jpeg_quality,
            purpose=args.purpose,
            confirm_data_export=args.confirm_data_export,
            output_path=args.output_path or None,
            write=not args.no_write,
        )
    elif args.command == "vision-export-consent-status":
        result = vision_export_consent_status(
            args.bundle_dir,
            consent_path=args.consent_path or None,
            provider_config=_provider_config_arg(args.provider_config),
            semantic_indexes=_int_csv_arg(args.semantic_indexes),
            temporal_indexes=_int_csv_arg(args.temporal_indexes),
            expected_calls=args.expected_calls,
            image_max_edge=args.image_max_edge,
            image_jpeg_quality=args.image_jpeg_quality,
        )
    elif args.command == "vision-export-consent-revoke":
        result = revoke_vision_export_consent(
            args.bundle_dir,
            consent_path=args.consent_path or None,
            write=not args.no_write,
        )
    elif args.command == "vision-execution-preflight":
        result = vision_execution_preflight(
            args.bundle_dir,
            provider_config=_provider_config_arg(args.provider_config),
            semantic_limit=args.semantic_limit,
            temporal_limit=args.temporal_limit,
            frame_count=args.frame_count,
            include_semantic=not args.no_semantic,
            include_temporal=not args.no_temporal,
            semantic_indexes=_int_csv_and_repeat_arg(
                args.semantic_indexes, args.semantic_index
            ),
            temporal_indexes=_int_csv_and_repeat_arg(
                args.temporal_indexes, args.temporal_index
            ),
            check_provider=args.check_provider,
            write=not args.no_write,
        )
    elif args.command == "timeline-alignment-audit":
        result = timeline_alignment_audit(args.bundle_dir, tolerance_seconds=args.tolerance_seconds, write=not args.no_write)
    elif args.command == "build-entity-lexicon":
        result = build_entity_lexicon(
            args.bundle_dir,
            base_lexicon_json=args.base_lexicon_json or None,
            phase=args.phase,
            write=not args.no_write,
        )
    elif args.command == "resolve-terms":
        result = resolve_terms(
            args.bundle_dir,
            metadata_json=args.metadata_json,
            glossary_json=args.glossary_json,
            min_mentions=args.min_mentions,
            write=not args.no_write,
        )
    elif args.command == "term-arbitration-codex":
        result = build_term_arbitration_codex_pack(
            args.bundle_dir,
            input_json=args.input_json or None,
            max_terms=args.max_terms,
            min_confidence=args.min_confidence,
            accept_draft=args.accept_draft,
            write=not args.no_write,
        )
    elif args.command == "validate-term-arbitration-codex-result":
        result = validate_term_arbitration_codex_result(
            args.bundle_dir,
            input_json=args.input_json,
            min_confidence=args.min_confidence,
            write=not args.no_write,
        )
    elif args.command == "term-correction-impact-report":
        result = term_correction_impact_report(
            args.bundle_dir,
            min_confidence=args.min_confidence,
            write=not args.no_write,
        )
    elif args.command == "term-correction-status":
        result = term_correction_status(args.bundle_dir)
    elif args.command == "term-correction-closure":
        result = run_term_correction_closure(
            args.bundle_dir,
            accept_draft=args.accept_draft,
            input_json=args.input_json or None,
            max_terms=args.max_terms,
            term_min_confidence=args.term_min_confidence,
            transcript_min_confidence=args.transcript_min_confidence,
            generate_codex_summary=not args.no_generate_codex_summary,
            write=not args.no_write,
        )
    elif args.command == "transcript-semantic-correction-pack":
        result = build_transcript_semantic_correction_pack(
            args.bundle_dir,
            run_id=args.run_id,
            dependency_versions=read_json(Path(args.dependency_versions)) if args.dependency_versions else None,
            limit=args.limit,
            source_mode=args.source_mode,
            write=not args.no_write,
        )
    elif args.command == "transcript-semantic-candidate-discovery-pack":
        result = build_transcript_semantic_candidate_discovery_pack(
            args.bundle_dir,
            input_json=args.input_json or None,
            limit=args.limit,
            write=not args.no_write,
        )
    elif args.command == "transcript-semantic-candidate-discovery-codex-draft":
        result = build_transcript_semantic_candidate_discovery_codex_draft(
            args.bundle_dir,
            input_json=args.input_json or None,
            limit=args.limit,
            max_suggestions=args.max_suggestions,
            write=not args.no_write,
        )
    elif args.command == "import-transcript-semantic-candidate-suggestions":
        result = import_transcript_semantic_candidate_suggestions(
            args.bundle_dir,
            input_json=args.input_json,
            write=not args.no_write,
        )
    elif args.command == "transcript-semantic-candidate-discovery-llm-draft":
        result = build_transcript_semantic_candidate_discovery_llm_draft(
            args.bundle_dir,
            input_json=args.input_json or None,
            provider_config=_provider_config_arg(args.provider_config),
            execute=args.execute,
            limit=args.limit,
            write=not args.no_write,
        )
    elif args.command == "transcript-semantic-correction-codex-draft":
        result = build_transcript_semantic_correction_codex_draft(
            args.bundle_dir,
            run_id=args.run_id,
            dependency_versions=read_json(Path(args.dependency_versions)) if args.dependency_versions else None,
            input_json=args.input_json or None,
            min_confidence=args.min_confidence,
            write=not args.no_write,
        )
    elif args.command == "transcript-semantic-correction-llm-draft":
        result = build_transcript_semantic_correction_llm_draft(
            args.bundle_dir,
            input_json=args.input_json or None,
            provider_config=_provider_config_arg(args.provider_config),
            execute=args.execute,
            limit=args.limit,
            min_confidence=args.min_confidence,
            write=not args.no_write,
            business_authorization_path=args.business_authorization or None,
        )
    elif args.command == "validate-transcript-semantic-correction":
        result = validate_transcript_semantic_correction(
            args.bundle_dir,
            input_json=args.input_json,
            min_confidence=args.min_confidence,
            write=not args.no_write,
        )
    elif args.command == "import-transcript-semantic-review-notes":
        result = import_transcript_semantic_review_notes(
            args.bundle_dir,
            review_json=args.review_json or None,
            min_confidence=args.min_confidence,
            write=not args.no_write,
        )
    elif args.command == "transcript-semantic-correction-closure":
        result = transcript_semantic_correction_closure(
            args.bundle_dir,
            input_json=args.input_json,
            min_confidence=args.min_confidence,
            auto_apply=args.auto_apply,
            refresh_exports=args.refresh_exports,
            write=not args.no_write,
        )
    elif args.command == "transcript-semantic-correction-impact-report":
        result = transcript_semantic_correction_impact_report(args.bundle_dir, write=not args.no_write)
    elif args.command == "transcript-semantic-readable-impact-report":
        result = transcript_semantic_correction_readable_impact_report(args.bundle_dir, write=not args.no_write)
    elif args.command == "transcript-semantic-summary-impact-report":
        result = transcript_semantic_summary_impact_report(
            args.bundle_dir,
            summary_path=args.summary_path or None,
            baseline_summary_path=args.baseline_summary_path or None,
            write=not args.no_write,
        )
    elif args.command == "transcript-semantic-correction-status":
        result = transcript_semantic_correction_status(args.bundle_dir, write=not args.no_write)
    elif args.command == "transcript-semantic-benchmark":
        result = run_transcript_semantic_benchmark(
            args.gold_json,
            args.candidate_pack_json,
            args.validation_json,
            full_body_path=args.full_body_path or None,
            smart_summary_path=args.smart_summary_path or None,
            output_json=args.output_json or None,
        )
    elif args.command == "transcript-semantic-restore":
        result = apply_transcript_semantic_restore(
            args.bundle_dir,
            args.plan_json,
            execute=args.execute,
            confirm_transaction_id=args.confirm_transaction_id,
        )
    elif args.command == "transcript-semantic-acceptance":
        result = transcript_semantic_acceptance(args.bundle_dir, output_dir=args.output_dir, write=not args.no_write)
    elif args.command == "transcript-semantic-batch-acceptance":
        result = transcript_semantic_batch_acceptance(
            args.batch_input,
            output_dir=args.output_dir,
            target_bundle_count=args.target_bundle_count,
            limit=args.limit,
            write=not args.no_write,
        )
    elif args.command == "transcript-semantic-repair-queue":
        result = transcript_semantic_repair_queue(
            args.batch_input,
            output_dir=args.output_dir,
            target_bundle_count=args.target_bundle_count,
            limit=args.limit,
            write=not args.no_write,
        )
    elif args.command == "transcript-semantic-repair-run":
        result = transcript_semantic_repair_run(
            args.batch_input,
            output_dir=args.output_dir,
            target_bundle_count=args.target_bundle_count,
            limit=args.limit,
            execute_safe_actions=args.execute_safe_actions,
            max_actions=args.max_actions,
            max_rounds=args.max_rounds,
            allow_closure=args.allow_closure,
            allow_llm=args.allow_llm,
            provider_config=_provider_config_arg(args.provider_config) if args.provider_config else None,
            llm_limit=args.llm_limit,
            business_authorization_path=args.business_authorization or None,
            write=not args.no_write,
        )
    elif args.command == "transcript-semantic-batch-review-pack":
        result = transcript_semantic_batch_review_pack(
            args.batch_input,
            output_dir=args.output_dir,
            target_bundle_count=args.target_bundle_count,
            limit=args.limit,
            max_candidates_per_bundle=args.max_candidates_per_bundle,
            write=not args.no_write,
        )
    elif args.command == "transcript-semantic-batch-import-review-notes":
        result = transcript_semantic_batch_import_review_notes(
            args.review_json,
            output_dir=args.output_dir,
            min_confidence=args.min_confidence,
            write=not args.no_write,
        )
    elif args.command == "transcript-semantic-batch-codex-review-draft":
        result = transcript_semantic_batch_codex_review_draft(
            args.review_pack_json,
            output_dir=args.output_dir,
            write=not args.no_write,
        )
    elif args.command == "targeted-visual-evidence":
        result = run_targeted_visual_evidence(
            args.bundle_dir,
            tagger_json=args.tagger_json or None,
            min_score=args.min_score,
            limit=args.limit,
            execute_ebook=args.execute_ebook,
            execute_crops=args.execute_crops,
            execute_ocr=args.execute_ocr,
            execute_tiles=args.execute_tiles,
            allow_online_review=args.allow_online_review,
            write=not args.no_write,
        )
    elif args.command == "vision-review-triage":
        result = vision_review_triage(
            args.bundle_dir,
            mode=args.mode,
            tagger_json=args.tagger_json,
            semantic_limit=args.semantic_limit,
            temporal_limit=args.temporal_limit,
            visual_structure_limit=args.visual_structure_limit,
            min_score=args.min_score,
            write=not args.no_write,
        )
    elif args.command == "plan-supplemental-frame-sampling":
        result = plan_supplemental_frame_sampling(
            args.bundle_dir,
            triage_json=args.triage_json,
            max_items=args.max_items,
            max_frames_per_item=args.max_frames_per_item,
            include_temporal=not args.no_temporal,
            include_visual_structure=not args.no_visual_structure,
            include_semantic=not args.no_semantic,
            write=not args.no_write,
        )
    elif args.command == "vision-review-queue":
        result = vision_review_queue(
            args.bundle_dir,
            min_score=args.min_score,
            batch_size=args.batch_size,
            max_items=args.max_items,
            provider=args.provider,
            env_file=args.env_file,
            refresh_triage=args.refresh_triage,
            write=not args.no_write,
        )
    elif args.command == "run-artifact-registry":
        result = build_run_artifact_registry(args.bundle_dir, write=not args.no_write)
    elif args.command == "review-attestation-create":
        result = create_review_attestation(
            args.bundle_dir,
            target=args.target,
            artifact_paths=_artifact_specs(args.artifact),
            approved_by=args.approved_by,
            comment=args.comment,
            write=not args.no_write,
        )
    elif args.command == "review-attestation-status":
        result = validate_review_attestation(
            args.bundle_dir,
            target=args.target,
            attestation_path=args.attestation_path or None,
        )
    elif args.command == "import-generation-contracts":
        result = import_generation_contracts(
            args.bundle_dir,
            task_path=args.task,
            receipt_path=args.receipt,
            validation_path=args.validation,
            preflight_path=args.preflight or None,
            allowed_roots=args.source_root or None,
            write=not args.no_write,
        )
    elif args.command == "import-previs-candidate":
        result = import_previs_candidate(
            args.bundle_dir,
            scene_path=args.scene,
            manifest_path=args.capture_manifest,
            validation_path=args.validation,
            allowed_roots=args.source_root or None,
            write=not args.no_write,
        )
    elif args.command == "material-manifest":
        result = build_material_manifest(
            args.bundle_dir,
            transcript_path=args.transcript or None,
            output_path=args.output or None,
            write=not args.no_write,
        )
    elif args.command == "material-manifest-validate":
        result = validate_material_manifest(
            args.bundle_dir,
            args.manifest_path or None,
            write_report=args.write_report,
        )
    elif args.command == "multimodal-sample-review":
        result = multimodal_sample_review(
            args.bundle_dir,
            comparison_json=args.comparison_json,
            sample_size=args.sample_size,
            include_missing=not args.no_missing,
            media_path=args.media_path,
            potplayer_path=args.potplayer_path,
            write=not args.no_write,
        )
    elif args.command == "validate-multimodal-sample-notes":
        result = validate_multimodal_sample_notes(
            args.bundle_dir,
            notes_json=args.notes_json,
            min_reviewed=args.min_reviewed,
            write=not args.no_write,
        )
    elif args.command == "local-vlm-adapter-plan":
        result = local_vlm_adapter_plan(args.output_dir, write=args.write)
    elif args.command == "local-vlm-serving-smoke":
        result = local_vlm_serving_smoke(
            provider=args.provider,
            bundle_dir=args.bundle_dir,
            output_dir=args.output_dir,
            single_image=args.single_image,
            multi_image_dir=args.multi_image_dir,
            execute=args.execute,
            timeout_seconds=args.timeout_seconds,
            max_images=args.max_images,
            image_probe_max_edge=args.image_probe_max_edge,
            image_probe_jpeg_quality=args.image_probe_jpeg_quality,
            frame_group_count=args.frame_group_count,
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_tokens or None,
            write=not args.no_write,
        )
    elif args.command == "audit-knowledge-coverage":
        result = audit_knowledge_coverage(args.bundle_dir, write=not args.no_write)
    elif args.command == "import-companion-courseware-text":
        result = import_companion_courseware_text(args.bundle_dir, args.source_path, title=args.title, write=not args.no_write)
    elif args.command == "export-knowledge-note":
        result = export_knowledge_note(
            args.bundle_dir,
            output_dir=args.output_dir,
            title=args.title,
            include_timeline=not args.no_timeline,
            include_full_transcript=not args.no_full_transcript,
            run_transcript_evidence_check=not args.no_transcript_evidence_check,
            write=not args.no_write,
        )
    elif args.command == "generate-smart-summary-with-codex":
        result = generate_smart_summary_with_codex(args.bundle_dir, input_md=args.input_md or None,
            handoff_id=args.handoff_id, agent_available=args.agent_available, write=not args.no_write)
    elif args.command == "prepare-smart-summary-llm-rewrite":
        result = prepare_smart_summary_llm_rewrite(args.bundle_dir, provider=args.provider, write=not args.no_write)
    elif args.command == "run-smart-summary-llm-rewrite":
        result = run_smart_summary_llm_rewrite(
            args.bundle_dir,
            provider_config=_provider_config_arg(args.provider_config),
            execute=args.execute,
            max_input_chars=args.max_input_chars,
            temperature=args.temperature,
            install=not args.no_install,
            write=not args.no_write,
        )
    elif args.command == "build-smart-summary-input-pack":
        result = build_smart_summary_input_pack(
            args.bundle_dir,
            title=args.title,
            write=not args.no_write,
            max_visual_items=args.max_visual_items,
            progress_callback=stderr_progress_callback,
        )
    elif args.command == "technical-shot-detection":
        result = run_technical_shot_detection(
            args.bundle_dir,
            backend=args.backend,
            media_path=args.media_path or None,
            predictions_json=args.predictions_json or None,
            source_format=args.source_format,
            frame_rate=args.frame_rate,
            detector=args.detector,
            threshold=args.threshold,
            min_scene_len=args.min_scene_len,
            source_root=args.source_root or None,
            checkpoint_path=args.checkpoint_path or None,
            strict=bool(args.strict or not args.allow_fallback),
            write=not args.no_write,
        )
    elif args.command == "scene-detection":
        result = run_scene_detection(
            args.bundle_dir,
            media_path=args.media_path or None,
            detector=args.detector,
            threshold=args.threshold,
            min_scene_len=args.min_scene_len,
            max_points=args.max_points,
            source_root=args.source_root or None,
            write=not args.no_write,
        )
    elif args.command == "shot-language-analysis":
        result = run_shot_language_analysis(
            args.bundle_dir,
            execution_location=args.execution_location,
            route_id=args.route_id,
            source_root=args.source_root or None,
            shot_scale_model_path=args.shot_scale_model_path or None,
            shot_type_confidence_threshold=args.shot_type_confidence_threshold,
            movement_confidence_threshold=args.movement_confidence_threshold,
            execute=args.execute,
            write=not args.no_write,
        )
    elif args.command == "shot-review-apply":
        result = apply_shot_review_notes(
            args.bundle_dir,
            args.review_notes,
            write=not args.no_write,
        )
    elif args.command == "shot-review-status":
        result = shot_review_status(args.bundle_dir)
    elif args.command == "technical-shot-fusion":
        result = fuse_technical_shot_boundaries(
            args.bundle_dir,
            args.candidate_paths,
            frame_rate=args.frame_rate,
            tolerance_frames=args.tolerance_frames,
            write=not args.no_write,
        )
    elif args.command == "scene-candidate-evidence":
        result = build_scene_candidate_evidence(
            args.bundle_dir,
            args.candidates_json,
            model_id=args.model_id,
            model_commit=args.model_commit,
            language=args.language,
            taxonomy_prompt=args.taxonomy_prompt,
            cache_format_version=args.cache_format_version,
            source_format=args.source_format,
            frame_rate=args.frame_rate,
            write=not args.no_write,
        )
    elif args.command == "highlight-detection":
        result = run_highlight_detection(
            args.bundle_dir,
            query=args.query,
            media_path=args.media_path or None,
            checkpoint_path=args.checkpoint_path or None,
            source_root=args.source_root or None,
            predictions_json=args.predictions_json or None,
            feature_name=args.feature_name,
            device=args.device,
            execute=args.execute,
            write=not args.no_write,
        )
    elif args.command == "video-structure":
        result = build_video_structure(
            args.bundle_dir,
            media_path=args.media_path or None,
            title=args.title,
            input_pack=args.input_pack or None,
            run_shot_detection=not args.no_shot_detection,
            shot_detector=args.shot_detector,
            shot_threshold=args.shot_threshold,
            shot_source_root=args.shot_source_root or None,
            highlight_query=args.highlight_query,
            highlight_predictions_json=args.highlight_predictions_json or None,
            content_profile=args.content_profile,
            shot_embeddings_json=args.shot_embeddings_json or None,
            story_evidence_json=args.story_evidence_json or None,
            local_story_route_id=args.local_story_route_id,
            write=not args.no_write,
        )
    elif args.command == "export-creator-report":
        result = export_creator_report(
            args.bundle_dir, title=args.title,
            brief=read_json(args.brief_json) if args.brief_json else None,
            formats=args.formats, pandoc=args.pandoc,
            scriptbreak_script=args.scriptbreak_script,
            scriptbreak_sha256=args.scriptbreak_sha256, write=not args.no_write,
            view=args.view, video_type=args.video_type,
            interactive_html=args.interactive_html, output_dir=args.output_dir or None,
            ocr_results_json=args.ocr_results_json or None,
        )
    elif args.command == "creator-report-status":
        result = creator_report_status(args.bundle_dir, view=args.view, output_dir=args.output_dir or None)
    elif args.command == "shot-breakdown":
        result = build_shot_breakdown(
            args.bundle_dir,
            title=args.title,
            reference_analysis_json=args.reference_analysis_json or None,
            write=not args.no_write,
        )
    elif args.command == "video-decomposition-report":
        result = build_video_decomposition_report(
            args.bundle_dir,
            title=args.title,
            write=not args.no_write,
        )
    elif args.command == "video-decomposition-status":
        result = video_decomposition_report_status(
            args.bundle_dir,
            report_path=args.report_path or None,
            write=not args.no_write,
        )
    elif args.command == "video-decomposition-compare":
        result = compare_video_decomposition_reports(
            args.report_paths,
            output_dir=args.output_dir or None,
            title=args.title,
            write=not args.no_write,
        )
    elif args.command == "semantic-chapter-plan":
        result = build_semantic_chapter_plan(args.bundle_dir, title=args.title, chapter_mode=args.chapter_mode, write=not args.no_write)
    elif args.command == "content-profile":
        result = apply_content_profile(
            args.bundle_dir,
            profile_id=args.profile,
            write=not args.no_write,
        )
    elif args.command == "source-review-lineage":
        result = discover_source_review_lineage(
            args.bundle_dir,
            search_roots=args.search_roots,
            apply=args.apply,
            write=not args.no_write,
        )
    elif args.command == "production-artifact-gate":
        result = evaluate_production_artifact_gate(
            args.bundle_dir,
            artifact_kind=args.artifact_kind,
            search_roots=args.search_roots,
            discover_prior_reviews=not args.no_discover_prior_reviews,
            write=not args.no_write,
        )
    elif args.command == "build-smart-summary-chapters":
        result = build_smart_summary_chapter_pack(
            args.bundle_dir,
            title=args.title,
            write=not args.no_write,
            target_chapters=args.target_chapters,
            max_visual_items=args.max_visual_items,
            chapter_mode=args.chapter_mode,
            progress_callback=stderr_progress_callback,
        )
    elif args.command == "smart-summary-section-workflow":
        result = build_smart_summary_section_workflow(args.bundle_dir, title=args.title, write=not args.no_write, target_chapters=args.target_chapters)
    elif args.command == "smart-summary-section-editor":
        result = build_smart_summary_section_editor(args.bundle_dir, write=not args.no_write)
    elif args.command == "smart-summary-section-apply":
        result = apply_smart_summary_sections(args.bundle_dir, input_json=args.input_json or None, write=not args.no_write, require_all_sections=args.require_all_sections)
    elif args.command == "run-smart-summary-section-llm-rewrite":
        result = run_smart_summary_section_llm_rewrite(
            args.bundle_dir,
            provider_config=_provider_config_arg(args.provider_config),
            execute=args.execute,
            auto_from_profile=args.auto_from_profile,
            quality_profile=args.quality_profile,
            target_chapters=args.target_chapters,
            limit=args.limit,
            section_ids=args.section_ids,
            only_needing_rewrite=args.only_needing_rewrite,
            max_prompt_chars=args.max_prompt_chars,
            max_tokens=args.max_tokens,
            min_section_chars=args.min_section_chars,
            temperature=args.temperature,
            install=not args.no_install,
            require_all_sections=not args.no_require_all_sections,
            write=not args.no_write,
            business_authorization_path=args.business_authorization or None,
        )
    elif args.command == "smart-summary-global-reduce":
        result = run_smart_summary_global_reduce(
            args.bundle_dir,
            provider_config=_provider_config_arg(args.provider_config),
            execute=args.execute,
            reuse_candidate=args.reuse_candidate,
            recovery_execution_report=args.recover_execution_report or None,
            max_input_chars=args.max_input_chars,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            install=not args.no_install,
            write=not args.no_write,
            business_authorization_path=args.business_authorization or None,
        )
    elif args.command == "summary-consistency-check":
        result = run_summary_consistency_check(
            args.bundle_dir,
            summary_path=args.summary_path or None,
            write=not args.no_write,
        )
    elif args.command == "smart-summary-quality-check":
        result = smart_summary_quality_check(args.bundle_dir, summary_path=args.summary_path or None, require_codex=args.require_codex, write=not args.no_write)
    elif args.command == "content-asset-status":
        result = content_asset_status(args.bundle_dir, write=not args.no_write)
    elif args.command == "batch-content-asset-status":
        result = batch_content_asset_status(args.batch_input, output_dir=args.output_dir, write=not args.no_write)
    elif args.command == "content-handoff-pack":
        result = content_handoff_pack(args.batch_input, output_dir=args.output_dir, write=not args.no_write)
    elif args.command == "export-quality-console":
        result = export_quality_console(args.bundle_dir, write=not args.no_write)
    elif args.command == "export-task-console":
        result = export_task_console(args.bundle_dir, write=not args.no_write, refresh=not args.no_refresh)
    elif args.command == "subqueue-action-plan":
        result = export_subqueue_action_plan(args.bundle_dir, write=not args.no_write, refresh=not args.no_refresh)
    elif args.command == "export-video-workbench":
        result = export_video_workbench(args.bundle_dir, write=not args.no_write)
    elif args.command == "video-edit-review-pack":
        result = build_video_edit_review_pack(
            args.bundle_dir,
            decisions_json=args.decisions_json or None,
            tokens_json=args.tokens_json or None,
            silence_json=args.silence_json or None,
            delete_segments_json=args.delete_segments_json or None,
            cut_segments_json=args.cut_segments_json or None,
            ai_baseline_json=args.ai_baseline_json or None,
            media_path=args.media_path or None,
            reclaim_silence=args.reclaim_silence,
            human_confirmed_diff=args.human_confirmed_diff,
            review_attestation_path=args.review_attestation_path or None,
            write=not args.no_write,
        )
    elif args.command == "refresh-review-html":
        result = refresh_bundle_review_html(args.bundle_dir, write=not args.no_write)
    elif args.command == "bundle-status-report":
        result = bundle_status_report(args.bundle_dir, refresh=not args.no_refresh)
    elif args.command == "acceptance-check":
        result = acceptance_check(args.bundle_dir, refresh=not args.no_refresh, write=not args.no_write)
    elif args.command == "controlled-execution-check":
        result = controlled_execution_check(args.bundle_dir, refresh=not args.no_refresh, write=not args.no_write)
    elif args.command == "controlled-execution-smoke":
        result = controlled_execution_smoke(
            args.bundle_dir,
            execute=args.execute,
            restore_after=args.restore_after,
            provider_config=_provider_config_arg(args.provider_config),
            kind=args.kind,
            index=args.index,
            frame_count=args.frame_count,
            write=not args.no_write,
        )
    elif args.command == "bundle-next-action":
        result = bundle_next_action(args.bundle_dir, refresh=not args.no_refresh)
    elif args.command == "autopilot-start":
        from .autopilot import autopilot_start

        result = autopilot_start(
            args.bundle_or_batch,
            runner=args.runner,
            execution_route=args.execution_route,
            online_derived_artifacts_allowed=not args.no_online_derived_artifacts,
        )
    elif args.command == "autopilot-next":
        from .autopilot import autopilot_next

        result = autopilot_next(args.job_id, retry_provider=args.retry_provider)
    elif args.command == "autopilot-submit":
        from .autopilot import autopilot_submit

        result = autopilot_submit(args.job_id, input_json=args.input_json)
    elif args.command == "autopilot-status":
        from .autopilot import autopilot_status

        result = autopilot_status(args.job_id)
    elif args.command == "autopilot-run":
        from .autopilot import autopilot_run

        result = autopilot_run(
            args.bundle_or_batch,
            runner=args.runner,
            execution_route=args.execution_route,
            resume=args.resume,
            retry_provider=args.retry_provider,
        )
    elif args.command == "bundle-advance":
        result = bundle_advance(
            args.bundle_dir,
            execute=args.execute,
            refresh_outputs=args.refresh_outputs,
            vault=args.vault,
            folder=args.folder,
            timeout_seconds=args.timeout_seconds,
            ocr_input_json=args.ocr_input_json,
            ocr_language=args.ocr_language,
            captiocr_root=args.captiocr_root,
            visual_structure_input_json=args.visual_structure_input_json,
            provider_config=_provider_config_arg(args.provider_config),
            multimodal_limit=args.multimodal_limit,
            temporal_limit=args.temporal_limit,
            frame_count=args.frame_count,
            confirm_vision_calls=args.confirm_vision_calls,
            confirm_vision_indexes=args.confirm_vision_indexes,
        )
    elif args.command == "bundle-advance-queue":
        result = bundle_advance_queue(
            args.bundle_dir,
            max_steps=args.max_steps,
            execute=args.execute,
            refresh_outputs=args.refresh_outputs,
            vault=args.vault,
            folder=args.folder,
            timeout_seconds=args.timeout_seconds,
            ocr_input_json=args.ocr_input_json,
            ocr_language=args.ocr_language,
            captiocr_root=args.captiocr_root,
            visual_structure_input_json=args.visual_structure_input_json,
            provider_config=_provider_config_arg(args.provider_config),
            multimodal_limit=args.multimodal_limit,
            temporal_limit=args.temporal_limit,
            frame_count=args.frame_count,
            confirm_vision_calls=args.confirm_vision_calls,
            confirm_vision_indexes=args.confirm_vision_indexes,
        )
    elif args.command == "bundle-advance-log":
        result = bundle_advance_log(args.bundle_dir)
    elif args.command == "vision-analysis-run-log":
        result = vision_analysis_run_log(args.bundle_dir)
    elif args.command == "vision-analysis-restore-plan":
        result = vision_analysis_restore_plan(args.bundle_dir, run_id=args.run_id, write=not args.no_write)
    elif args.command == "vision-analysis-apply-restore":
        result = vision_analysis_apply_restore(args.bundle_dir, plan_json=args.plan_json, execute=args.execute, confirm_run_id=args.confirm_run_id)
    elif args.command == "mcp-call":
        result = run_mcp_call(args.tool, args.args_json)
    elif args.command == "mcp-audit-bundle":
        result = audit_bundle_mcp_args(args.bundle_dir)
    else:
        parser.error(f"unknown command: {args.command}")
        return 2

    print(render_cli_result(result, verbose=verbose))
    if args.command == "run-codex-vision" and args.execute and result.get("status") != "completed":
        return 2 if result.get("status") == "paused_batch_limit" else 1
    if args.command == "export-creator-report" and result.get("status") not in {"preview", "exported_candidate"}:
        return 1
    if args.command == "creator-report-status" and result.get("status") != "fresh":
        return 1
    if args.command == "silero-vad-candidate" and args.execute and result.get("status") != "completed":
        return 1
    return 0


def audit_bundle_mcp_args(bundle_dir: str | Path) -> dict[str, Any]:
    return audit_bundle_mcp_args_impl(
        bundle_dir,
        callables=_mcp_callables(),
        tool_by_manifest_key=_MCP_TOOL_BY_MANIFEST_KEY,
        arg_aliases=_MCP_ARG_ALIASES,
    )


def run_mcp_call(tool: str, args_json: str | Path) -> dict[str, Any]:
    args_path = resolve_mcp_args_path(args_json)
    if not args_path.exists():
        raise FileNotFoundError(f"MCP args JSON not found: {args_path}")
    payload = read_mcp_args(args_path)
    callables = _mcp_callables()
    normalised = str(tool or "").strip()
    if normalised not in callables:
        return {
            "schema": "video_knowledge_pipeline.mcp_call.v1",
            "status": "unsupported_tool",
            "tool": normalised,
            "args_json": str(args_path.resolve()),
            "supported_tools": sorted(callables),
        }
    payload = normalise_mcp_payload(normalised, payload, _MCP_ARG_ALIASES)
    call_payload, ignored_args = filter_mcp_payload(callables[normalised], payload)
    result = callables[normalised](**call_payload)
    if isinstance(result, dict):
        result.setdefault("mcp_call", {"tool": normalised, "args_json": str(args_path.resolve()), "ignored_args": ignored_args})
        return result
    return {
        "schema": "video_knowledge_pipeline.mcp_call.v1",
        "status": "ok",
        "tool": normalised,
        "args_json": str(args_path.resolve()),
        "ignored_args": ignored_args,
        "result": result,
    }


def resolve_mcp_args_path(args_json: str | Path) -> Path:
    args_path = Path(args_json).expanduser()
    if args_path.exists() or args_path.is_absolute():
        return args_path
    project_root = Path(__file__).resolve().parents[2]
    matches = [path for path in project_root.rglob(args_path.name) if path.is_file()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            "Relative MCP args path is ambiguous; pass an absolute path. "
            f"Name: {args_path.name}; matches: {', '.join(str(path) for path in matches[:10])}"
        )
    return args_path


def _quality_benchmark_mcp_call(
    action: str,
    input_path: str,
    bundle_dirs: list[str] | None = None,
    media_paths: list[str] | None = None,
    output_dir: str = "",
    samples_per_bundle: int = 8,
    sample_seconds: float = 60.0,
    execute_clips: bool = False,
    legacy_reference_manifest: str = "",
    variants: list[str] | None = None,
    execute: bool = False,
    resume: bool = True,
    retry_failed: bool = False,
    limit: int = 0,
    timeout_seconds: int = 1800,
    scores_json: str = "",
    write: bool = True,
) -> dict[str, Any]:
    if action == "build":
        return build_quality_benchmark(
            input_path,
            bundle_dirs=bundle_dirs or [],
            media_paths=media_paths or [],
            samples_per_bundle=samples_per_bundle,
            sample_seconds=sample_seconds,
            execute_clips=execute_clips,
            legacy_reference_manifest=legacy_reference_manifest or None,
            write=write,
        )
    if action == "execute-variants":
        return execute_quality_benchmark_variants(
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
        return run_quality_benchmark(input_path, output_dir=output_dir or None, write=write)
    if action == "build-summary-review":
        return build_summary_blind_review(input_path, output_dir=output_dir or None, write=write)
    if action == "apply-summary-review":
        if not scores_json:
            raise ValueError("scores_json is required for apply-summary-review")
        return apply_summary_blind_review(input_path, scores_json, write=write)
    if action == "report":
        return report_quality_benchmark(input_path, output_dir=output_dir or None, write=write)
    raise ValueError("action must be build, execute-variants, run, report, build-summary-review, or apply-summary-review")


def _mcp_callables() -> dict[str, Any]:
    from .dual_asr_workflow import run_dual_asr_workflow
    from .qwen_failed_window_recovery import plan_qwen_failed_window_recovery, compose_qwen_failed_window_recovery
    from .asr_variant_evaluation import compare_asr_variants
    mapping: dict[str, Any] = {
        "refresh_lecture_review_outputs": refresh_lecture_review_outputs,
        "run_frame_recapture_plan": run_frame_recapture_plan,
        "run_visual_structure_plan": run_visual_structure_plan,
        "run_ocr_backfill": run_ocr_backfill,
        "run_screen_text_recovery": run_screen_text_recovery,
        "run_high_res_tile_plan": run_high_res_tile_plan,
        "tile_result_import_build": build_tile_result_import,
        "build_tile_result_import": build_tile_result_import,
        "tile_result_merge": run_tile_result_merge,
        "run_tile_result_merge": run_tile_result_merge,
        "run_video_frame_router": run_video_frame_router,
        "import_tagger_annotations": import_tagger_annotations,
        "run_multimodal_frame_analysis": run_multimodal_frame_analysis,
        "prepare_codex_vision_session": prepare_codex_vision_session,
        "prepare_summary_claim_review": prepare_summary_claim_review,
        "import_summary_claim_review": import_summary_claim_review,
        "summary_claim_review_status": summary_claim_review_status,
        "import_codex_vision_results": import_codex_vision_results,
        "run_temporal_frame_groups": run_temporal_frame_groups,
        "run_temporal_visual_analysis": run_temporal_visual_analysis,
        "attach_peepshow_output": attach_peepshow_output_to_bundle,
        "run_extractor_plan": run_extractor_plan,
        "extractor_run_log": extractor_run_log,
        "plan_asr": plan_asr_run,
        "plan_cloud_asr": plan_cloud_asr_run,
        "run_cloud_asr_plan": run_cloud_asr_plan,
        "plan_local_asr_service": plan_local_asr_service_run,
        "run_local_asr_service_plan": run_local_asr_service_plan,
        "online_model_api": online_model_api_call,
        "online_model_api_matrix": online_model_api_matrix,
        "model_task_coverage_audit": model_task_coverage_audit,
        "run_term_arbitration_model": run_term_arbitration_model,
        "run_bilinote_mind_map_model": run_bilinote_mind_map_model,
        "plan_whisperx_alignment": plan_whisperx_alignment,
        "postprocess_asr_transcript": postprocess_asr_transcript,
        "readable_transcript_llm_polish": run_readable_transcript_llm_polish,
        "agent_readable_transcript_rewrite": run_agent_readable_transcript_rewrite,
        "transcript_quality_gate": run_transcript_quality_gate,
        "asr_smoke": asr_smoke,
        "repair_bundle_assets": repair_bundle_assets,
        "refresh_bundle_repair_status": refresh_bundle_repair_status,
        "audit_bundle_readiness": audit_bundle_readiness,
        "audit_knowledge_coverage": audit_knowledge_coverage,
        "bundle_next_action": bundle_next_action,
        "bundle_source_artifacts": bundle_source_artifacts,
        "import_page_metadata": import_page_metadata,
        "bundle_advance": bundle_advance,
        "bundle_advance_log": bundle_advance_log,
        "bundle_advance_queue": bundle_advance_queue,
        "batch_repair_run": batch_repair_run,
        "batch_video_knowledge_run": batch_video_knowledge_run,
        "batch_run": batch_video_knowledge_run,
        "prepare_review_session": prepare_review_session,
        "review_closure_status": review_closure_status,
        "apply_review_notes": apply_review_notes_to_bundle,
        "apply_review_notes_to_bundle": apply_review_notes_to_bundle,
        "validate_review_notes": validate_review_notes_for_bundle,
        "bundle_status_report": bundle_status_report,
        "controlled_execution_check": controlled_execution_check,
        "controlled_execution_smoke": controlled_execution_smoke,
        "export_knowledge_note": export_knowledge_note,
        "generate_smart_summary_with_codex": generate_smart_summary_with_codex,
        "prepare_smart_summary_llm_rewrite": prepare_smart_summary_llm_rewrite,
        "run_smart_summary_llm_rewrite": run_smart_summary_llm_rewrite,
        "build_smart_summary_input_pack": build_smart_summary_input_pack,
        "asr_consensus": build_asr_consensus,
        "dual_asr_review": run_dual_asr_workflow,
        "plan_qwen_window_recovery": plan_qwen_failed_window_recovery,
        "compose_qwen_window_recovery": compose_qwen_failed_window_recovery,
        "compare_asr_variants": compare_asr_variants,
        "asr_diff_adjudication": build_asr_diff_adjudication,
        "apply_asr_diff_adjudication": apply_asr_diff_adjudication,
        "quality_benchmark": _quality_benchmark_mcp_call,
        "quality_finalize": finalize_quality_outputs,
        "punctuation_model_stage": run_punctuation_model_stage,
        "offline_quality_route": offline_quality_route,
        "targeted_visual_evidence": run_targeted_visual_evidence,
        "scene_detection": run_scene_detection,
        "semantic_chapter_plan": build_semantic_chapter_plan,
        "build_smart_summary_chapters": build_smart_summary_chapter_pack,
        "build_smart_summary_section_workflow": build_smart_summary_section_workflow,
        "smart_summary_section_workflow": build_smart_summary_section_workflow,
        "build_smart_summary_section_editor": build_smart_summary_section_editor,
        "smart_summary_section_editor": build_smart_summary_section_editor,
        "apply_smart_summary_sections": apply_smart_summary_sections,
        "smart_summary_section_apply": apply_smart_summary_sections,
        "smart_summary_section_llm_rewrite": run_smart_summary_section_llm_rewrite,
        "smart_summary_global_reduce": run_smart_summary_global_reduce,
        "summary_consistency_check": run_summary_consistency_check,
        "run_smart_summary_section_llm_rewrite": run_smart_summary_section_llm_rewrite,
        "smart_summary_quality_check": smart_summary_quality_check,
        "content_asset_status": content_asset_status,
        "batch_content_asset_status": batch_content_asset_status,
        "content_handoff_pack": content_handoff_pack,
        "export_quality_console": export_quality_console,
        "export_task_console": export_task_console,
        "subqueue_action_plan": export_subqueue_action_plan,
        "export_video_workbench": export_video_workbench,
        "video_edit_review_pack": build_video_edit_review_pack,
        "shot_breakdown": build_shot_breakdown,
        "export_creator_report": export_creator_report,
        "creator_report_status": creator_report_status,
        "video_decomposition_report": build_video_decomposition_report,
        "video_decomposition_status": video_decomposition_report_status,
        "video_decomposition_compare": compare_video_decomposition_reports,
        "video_moment_index": build_video_moment_index,
        "long_video_memory_pack": build_long_video_memory_pack,
        "video_rag_pack": build_video_rag_pack,
        "video_rag_search": search_video_rag,
        "script_clip_candidate_pack": build_script_clip_candidate_pack,
        "script_clip_alignment_check": check_script_clip_alignment,
        "content_clip_candidate_pack": build_content_clip_candidate_pack,
        "content_clip_alignment_check": check_content_clip_alignment,
        "long_video_fast_segment_plan": build_long_video_fast_segment_plan,
        "apply_long_video_fast_segment_review": apply_long_video_fast_segment_review,
        "render_long_video_fast_segment": render_long_video_fast_segment,
        "video_evidence_query_plan": build_video_evidence_query_plan,
        "apply_video_evidence_confirmation": apply_video_evidence_confirmation,
        "video_rag_service_plan": video_rag_service_plan,
        "bilinote_mind_map_prompt_pack": build_bundle_mind_map_prompt_pack,
        "prepare_transcript_edit_session": prepare_transcript_edit_session,
        "evidence_conflict_index": build_evidence_conflict_index,
        "transcript_evidence_correction_pipeline": run_transcript_evidence_correction_pipeline,
        "transcript_candidate_recall_benchmark": benchmark_transcript_candidate_recall,
        "transcript_semantic_correction_pack": build_transcript_semantic_correction_pack,
        "transcript_semantic_candidate_discovery_pack": build_transcript_semantic_candidate_discovery_pack,
        "transcript_semantic_candidate_discovery_llm_draft": build_transcript_semantic_candidate_discovery_llm_draft,
        "transcript_semantic_candidate_discovery_codex_draft": build_transcript_semantic_candidate_discovery_codex_draft,
        "import_transcript_semantic_candidate_suggestions": import_transcript_semantic_candidate_suggestions,
        "transcript_semantic_correction_codex_draft": build_transcript_semantic_correction_codex_draft,
        "transcript_semantic_correction_llm_draft": build_transcript_semantic_correction_llm_draft,
        "validate_transcript_semantic_correction": validate_transcript_semantic_correction,
        "transcript_semantic_correction_closure": transcript_semantic_correction_closure,
        "transcript_semantic_correction_impact_report": transcript_semantic_correction_impact_report,
        "transcript_semantic_readable_impact_report": transcript_semantic_correction_readable_impact_report,
        "transcript_semantic_summary_impact_report": transcript_semantic_summary_impact_report,
        "transcript_semantic_correction_status": transcript_semantic_correction_status,
        "transcript_semantic_benchmark": run_transcript_semantic_benchmark,
        "apply_transcript_semantic_restore": apply_transcript_semantic_restore,
        "transcript_semantic_acceptance": transcript_semantic_acceptance,
        "import_transcript_semantic_review_notes": import_transcript_semantic_review_notes,
        "transcript_semantic_batch_acceptance": transcript_semantic_batch_acceptance,
        "transcript_semantic_repair_queue": transcript_semantic_repair_queue,
        "transcript_semantic_repair_run": transcript_semantic_repair_run,
        "transcript_semantic_batch_review_pack": transcript_semantic_batch_review_pack,
        "transcript_semantic_batch_import_review_notes": transcript_semantic_batch_import_review_notes,
        "transcript_semantic_batch_codex_review_draft": transcript_semantic_batch_codex_review_draft,
        "transcript_source_arbitration": arbitrate_transcript_sources,
        "arbitrate_transcript_sources": arbitrate_transcript_sources,
        "apply_transcript_edits": apply_transcript_edits,
        "external_capability_pack": build_external_capability_pack,
        "acceptance_check": acceptance_check,
        "vision_acceptance_plan": vision_acceptance_plan,
        "vision_environment_status": vision_environment_status,
        "vision_execution_preflight": vision_execution_preflight,
        "vision_review_triage": vision_review_triage,
        "plan_supplemental_frame_sampling": plan_supplemental_frame_sampling,
        "vision_review_queue": vision_review_queue,
        "run_artifact_registry": build_run_artifact_registry,
        "multimodal_sample_review": multimodal_sample_review,
        "validate_multimodal_sample_notes": validate_multimodal_sample_notes,
        "timeline_alignment_audit": timeline_alignment_audit,
        "resolve_terms": resolve_terms,
        "term_arbitration_codex": build_term_arbitration_codex_pack,
        "validate_term_arbitration_codex_result": validate_term_arbitration_codex_result,
        "term_correction_impact_report": term_correction_impact_report,
        "term_correction_status": term_correction_status,
        "term_correction_closure": run_term_correction_closure,
        "vision_provider_smoke": vision_provider_smoke,
        "vision_provider_matrix": vision_provider_matrix,
        "local_vlm_serving_smoke": local_vlm_serving_smoke,
        "vision_analysis_run_log": vision_analysis_run_log,
        "vision_analysis_restore_plan": vision_analysis_restore_plan,
        "vision_analysis_apply_restore": vision_analysis_apply_restore,
        "test_vision_provider": test_vision_provider,
        "openclaw_bridge_status": openclaw_bridge_status,
        "openclaw_bridge_doctor": openclaw_bridge_doctor,
        "openclaw_live_smoke": openclaw_live_smoke,
        "openclaw_docker_contract_check": openclaw_docker_contract_check,
        "openclaw_video_plan": openclaw_video_plan,
        "openclaw_video_ingest": openclaw_video_ingest,
        "openclaw_video_link": openclaw_video_link,
        "openclaw_video_from_vdo_handoff": vdo_handoff_plan,
        "openclaw_video_ingest_vdo_handoff": ingest_vdo_handoff,
        "ingest_vdo_handoff": ingest_vdo_handoff,
        "vdo_handoff_plan": vdo_handoff_plan,
    }
    for name, func in list(mapping.items()):
        mapping[f"{name}_tool"] = func
    return mapping


_MCP_TOOL_BY_MANIFEST_KEY = {
    "mcp_refresh_args": "refresh_lecture_review_outputs",
    "mcp_frame_recapture_args": "run_frame_recapture_plan",
    "mcp_supplemental_frame_sampling_args": "plan_supplemental_frame_sampling",
    "mcp_ocr_backfill_args": "run_ocr_backfill",
    "mcp_screen_text_recovery_args": "run_screen_text_recovery",
    "mcp_high_res_tile_plan_args": "run_high_res_tile_plan",
    "mcp_tile_result_import_build_args": "build_tile_result_import",
    "mcp_tile_result_merge_args": "run_tile_result_merge",
    "mcp_visual_structure_args": "run_visual_structure_plan",
    "mcp_video_frame_router_args": "run_video_frame_router",
    "mcp_import_tagger_annotations_args": "import_tagger_annotations",
    "mcp_multimodal_frame_analysis_args": "run_multimodal_frame_analysis",
    "mcp_prepare_codex_vision_session_args": "prepare_codex_vision_session",
    "mcp_import_codex_vision_results_args": "import_codex_vision_results",
    "mcp_temporal_frame_groups_args": "run_temporal_frame_groups",
    "mcp_temporal_visual_analysis_args": "run_temporal_visual_analysis",
    "mcp_attach_peepshow_output_args": "attach_peepshow_output",
    "mcp_asset_repair_args": "repair_bundle_assets",
    "mcp_repair_status_args": "refresh_bundle_repair_status",
    "mcp_readiness_args": "audit_bundle_readiness",
    "mcp_knowledge_coverage_args": "audit_knowledge_coverage",
    "mcp_next_action_args": "bundle_next_action",
    "mcp_source_artifacts_args": "bundle_source_artifacts",
    "mcp_import_page_metadata_args": "import_page_metadata",
    "mcp_advance_args": "bundle_advance",
    "mcp_advance_log_args": "bundle_advance_log",
    "mcp_advance_queue_args": "bundle_advance_queue",
    "mcp_review_session_args": "prepare_review_session",
    "mcp_review_closure_status_args": "review_closure_status",
    "mcp_apply_review_notes_args": "apply_review_notes",
    "mcp_status_report_args": "bundle_status_report",
    "mcp_controlled_execution_check_args": "controlled_execution_check",
    "mcp_controlled_execution_smoke_args": "controlled_execution_smoke",
    "mcp_export_knowledge_note_args": "export_knowledge_note",
    "mcp_generate_smart_summary_with_codex_args": "generate_smart_summary_with_codex",
    "mcp_prepare_smart_summary_llm_rewrite_args": "prepare_smart_summary_llm_rewrite",
    "mcp_run_smart_summary_llm_rewrite_args": "run_smart_summary_llm_rewrite",
    "mcp_build_smart_summary_input_pack_args": "build_smart_summary_input_pack",
    "mcp_build_smart_summary_chapters_args": "build_smart_summary_chapters",
    "mcp_smart_summary_section_workflow_args": "smart_summary_section_workflow",
    "mcp_smart_summary_section_editor_args": "smart_summary_section_editor",
    "mcp_smart_summary_section_apply_args": "smart_summary_section_apply",
    "mcp_smart_summary_quality_check_args": "smart_summary_quality_check",
    "mcp_content_asset_status_args": "content_asset_status",
    "mcp_batch_content_asset_status_args": "batch_content_asset_status",
    "mcp_content_handoff_pack_args": "content_handoff_pack",
    "mcp_export_task_console_args": "export_task_console",
    "mcp_plan_cloud_asr_args": "plan_cloud_asr",
    "mcp_run_cloud_asr_plan_args": "run_cloud_asr_plan",
    "mcp_plan_local_asr_service_args": "plan_local_asr_service",
    "mcp_run_local_asr_service_plan_args": "run_local_asr_service_plan",
    "mcp_postprocess_asr_transcript_args": "postprocess_asr_transcript",
    "mcp_readable_transcript_llm_polish_args": "readable_transcript_llm_polish",
    "mcp_agent_readable_transcript_rewrite_args": "agent_readable_transcript_rewrite",
    "mcp_transcript_quality_gate_args": "transcript_quality_gate",
    "mcp_subqueue_action_plan_args": "subqueue_action_plan",
    "mcp_online_model_api_args": "online_model_api",
    "mcp_online_model_api_matrix_args": "online_model_api_matrix",
    "mcp_video_workbench_args": "export_video_workbench",
    "mcp_video_edit_review_pack_args": "video_edit_review_pack",
    "mcp_shot_breakdown_args": "shot_breakdown",
    "mcp_video_decomposition_report_args": "video_decomposition_report",
    "mcp_video_moment_index_args": "video_moment_index",
    "mcp_long_video_memory_pack_args": "long_video_memory_pack",
    "mcp_video_rag_pack_args": "video_rag_pack",
    "mcp_video_rag_search_args": "video_rag_search",
    "mcp_script_clip_candidate_pack_args": "script_clip_candidate_pack",
    "mcp_script_clip_alignment_check_args": "script_clip_alignment_check",
    "mcp_content_clip_candidate_pack_args": "content_clip_candidate_pack",
    "mcp_content_clip_alignment_check_args": "content_clip_alignment_check",
    "mcp_long_video_fast_segment_plan_args": "long_video_fast_segment_plan",
    "mcp_video_rag_service_plan_args": "video_rag_service_plan",
    "mcp_prepare_transcript_edit_session_args": "prepare_transcript_edit_session",
    "mcp_apply_transcript_edits_args": "apply_transcript_edits",
    "mcp_transcript_evidence_correction_pipeline_args": "transcript_evidence_correction_pipeline",
    "mcp_transcript_candidate_recall_benchmark_args": "transcript_candidate_recall_benchmark",
    "mcp_transcript_semantic_correction_pack_args": "transcript_semantic_correction_pack",
    "mcp_transcript_semantic_correction_codex_draft_args": "transcript_semantic_correction_codex_draft",
    "mcp_transcript_semantic_candidate_discovery_pack_args": "transcript_semantic_candidate_discovery_pack",
    "mcp_transcript_semantic_candidate_discovery_llm_draft_args": "transcript_semantic_candidate_discovery_llm_draft",
    "mcp_transcript_semantic_candidate_discovery_codex_draft_args": "transcript_semantic_candidate_discovery_codex_draft",
    "mcp_import_transcript_semantic_candidate_suggestions_args": "import_transcript_semantic_candidate_suggestions",
    "mcp_transcript_semantic_correction_llm_draft_args": "transcript_semantic_correction_llm_draft",
    "mcp_validate_transcript_semantic_correction_args": "validate_transcript_semantic_correction",
    "mcp_transcript_semantic_correction_closure_args": "transcript_semantic_correction_closure",
    "mcp_transcript_semantic_correction_impact_report_args": "transcript_semantic_correction_impact_report",
    "mcp_transcript_semantic_readable_impact_report_args": "transcript_semantic_readable_impact_report",
    "mcp_transcript_semantic_summary_impact_report_args": "transcript_semantic_summary_impact_report",
    "mcp_transcript_semantic_correction_status_args": "transcript_semantic_correction_status",
    "mcp_transcript_semantic_acceptance_args": "transcript_semantic_acceptance",
    "mcp_import_transcript_semantic_review_notes_args": "import_transcript_semantic_review_notes",
    "mcp_transcript_semantic_batch_acceptance_args": "transcript_semantic_batch_acceptance",
    "mcp_transcript_semantic_repair_queue_args": "transcript_semantic_repair_queue",
    "mcp_transcript_semantic_repair_run_args": "transcript_semantic_repair_run",
    "mcp_transcript_source_arbitration_args": "transcript_source_arbitration",
    "mcp_external_capability_pack_args": "external_capability_pack",
    "mcp_openclaw_bridge_doctor_args": "openclaw_bridge_doctor",
    "mcp_openclaw_live_smoke_args": "openclaw_live_smoke",
    "mcp_acceptance_check_args": "acceptance_check",
    "mcp_vision_acceptance_plan_args": "vision_acceptance_plan",
    "mcp_vision_execution_preflight_args": "vision_execution_preflight",
    "mcp_vision_review_queue_args": "vision_review_queue",
    "mcp_run_artifact_registry_args": "run_artifact_registry",
    "mcp_multimodal_sample_review_args": "multimodal_sample_review",
    "mcp_validate_multimodal_sample_notes_args": "validate_multimodal_sample_notes",
    "mcp_vision_review_triage_args": "vision_review_triage",
    "mcp_vision_review_triage_preflight_args": "vision_execution_preflight",
    "mcp_timeline_alignment_audit_args": "timeline_alignment_audit",
    "mcp_resolve_terms_args": "resolve_terms",
    "mcp_term_arbitration_codex_args": "term_arbitration_codex",
    "mcp_term_arbitration_codex_validate_args": "validate_term_arbitration_codex_result",
    "mcp_term_correction_impact_report_args": "term_correction_impact_report",
    "mcp_term_correction_status_args": "term_correction_status",
    "mcp_term_correction_closure_args": "term_correction_closure",
    "mcp_term_correction_closure_codex_args": "term_correction_closure",
    "mcp_vision_provider_smoke_args": "vision_provider_smoke",
    "mcp_vision_provider_matrix_args": "vision_provider_matrix",
    "mcp_multimodal_frame_analysis_confirmed_args": "run_multimodal_frame_analysis",
    "mcp_temporal_visual_analysis_confirmed_args": "run_temporal_visual_analysis",
}


_MCP_ARG_ALIASES = {
    "refresh_lecture_review_outputs": {"project": "root"},
    "refresh_lecture_review_outputs_tool": {"project": "root"},
}


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video-knowledge", description="Video-first knowledge extraction pipeline")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the complete JSON result; large results are concise by default",
    )
    sub = parser.add_subparsers(dest="command", required=True)


    consent_execute = sub.add_parser(
        "execute-consented-model-task",
        help="Execute a consent-v2 task through the trusted route and emit JSON",
    )
    consent_execute.add_argument("consent_path", nargs="?", default="")
    consent_execute.add_argument("--route-revision", default="")
    write_choice = consent_execute.add_mutually_exclusive_group()
    write_choice.add_argument("--write", dest="write", action="store_true")
    write_choice.add_argument("--no-write", dest="write", action="store_false")
    consent_execute.set_defaults(write=None)
    business_create = sub.add_parser(
        "model-business-authorization-create",
        help="Confirm one bounded video workflow; no provider call is made",
    )
    business_create.add_argument("plan_path")
    business_create.add_argument("--output-path", default="")
    business_create.add_argument("--confirm-data-export", action="store_true")
    business_create.add_argument("--no-write", action="store_true")
    business_status = sub.add_parser(
        "model-business-authorization-status",
        help="Validate one business authorization and show remaining aggregate allowance",
    )
    business_status.add_argument("authorization_path")
    business_child = sub.add_parser(
        "model-business-child-consent",
        help="Mint an exact consent v2 from a confirmed parent without another prompt",
    )
    business_child.add_argument("authorization_path")
    business_child.add_argument("--stage-id", required=True)
    business_child.add_argument("--producer", required=True)
    business_child.add_argument("--artifact", action="append", required=True)
    business_child.add_argument(
        "--lineage-input", action="append", required=True
    )
    business_child.add_argument("--max-calls", type=int, default=1)
    business_child.add_argument("--output-path", default="")
    business_child.add_argument("--no-write", action="store_true")
    sub.add_parser(
        "media-capability-status",
        help="List candidate MediaKit capabilities and authorization boundaries without network calls",
    )
    general_tagger_status_parser = sub.add_parser("general-tagger-status", help="Inspect the local RAM++ general-image tagger runtime without downloading or executing a model")
    general_tagger_status_parser.add_argument("--source-root", default="")
    general_tagger_status_parser.add_argument("--checkpoint-path", default="")
    media_route = sub.add_parser(
        "media-route-status",
        help="Inspect content-addressed MediaKit routes without network calls",
    )
    media_route.add_argument("--task", default="")
    media_route.add_argument("--settings-path", default="")
    media_preflight = sub.add_parser(
        "media-connector-preflight",
        help="Validate MediaKit consent, route revision, destinations, and credential readiness without executing",
    )
    media_preflight.add_argument("consent_path")
    media_preflight.add_argument("--route-revision", required=True)
    media_preflight.add_argument("--expected-calls", type=int, default=1)
    media_preflight.add_argument("--settings-path", default="")
    mcp_call = sub.add_parser("mcp-call", help="Run a local MCP-style tool with a JSON args file")
    mcp_call.add_argument("tool")
    mcp_call.add_argument("args_json")

    mcp_audit = sub.add_parser("mcp-audit-bundle", help="Statically audit generated MCP args files for one WebUI bundle")
    mcp_audit.add_argument("bundle_dir")

    asr_status = sub.add_parser("asr-env-status", help="Check local ASR environment status")
    asr_status.add_argument("--venv-dir", default="")
    asr_status.add_argument("--output-dir", default="")
    asr_status.add_argument("--write", action="store_true")
    asr_status.add_argument("--python-version", default="3.11")

    config = sub.add_parser("config-status", help="Show the unified project config source used by CLI, MCP, generated manifests, and reports")
    config.add_argument("--config-path", default="")

    model_settings = sub.add_parser("model-api-settings", help="Show sanitized cloud/local model provider settings for the UI")
    model_settings.add_argument("--config-path", default="")

    onboarding_prepare = sub.add_parser(
        "model-api-onboarding-prepare",
        help="Prepare exact online model profiles without decrypting or writing API keys",
    )
    onboarding_prepare.add_argument(
        "--provider",
        action="append",
        default=[],
        help="Exact onboarding provider id; repeat to select several. Omit to prepare all exact bundles.",
    )
    onboarding_prepare.add_argument("--settings-path", default="")
    onboarding_prepare.add_argument("--secrets-path", default="")
    onboarding_prepare.add_argument(
        "--refresh-known-models",
        action="store_true",
        help="Replace only reviewed obsolete preset model IDs; custom model IDs remain protected.",
    )


    catalog_probe = sub.add_parser(
        "model-api-catalog-probe",
        help="Check one saved onboarding bundle against its provider model catalog without inference or artifact access",
    )
    catalog_probe.add_argument("--provider", required=True)
    catalog_probe.add_argument("--settings-path", default="")
    catalog_probe.add_argument("--secrets-path", default="")
    catalog_probe.add_argument("--execute", action="store_true")
    catalog_probe.add_argument(
        "--include-model-ids",
        action="store_true",
        help="Include the provider public model IDs in executed probe output",
    )

    route_preset = sub.add_parser(
        "model-api-route-preset",
        help="Apply a reviewed single-deployment model route preset without authorizing network egress",
    )
    route_preset.add_argument("--preset", required=True)
    route_preset.add_argument("--settings-path", default="")
    route_preset.add_argument("--secrets-path", default="")

    local_production_preset = sub.add_parser(
        "local-production-preset",
        help="Install an isolated local-only ASR/OCR/Qwen3-VL/Qwen3.5 production preset",
    )
    local_production_preset.add_argument("output_dir")
    local_production_preset.add_argument("--no-write", action="store_true")


    set_profile = sub.add_parser("set-vision-profile", help="Persist the default vision provider/model/base URL and batch limits without storing API keys")
    set_profile.add_argument("--config-path", default="")
    set_profile.add_argument("--provider", required=True)
    set_profile.add_argument("--model", default="")
    set_profile.add_argument("--base-url", default="")
    set_profile.add_argument("--multimodal-limit", type=int)
    set_profile.add_argument("--temporal-limit", type=int)
    set_profile.add_argument("--frame-count", type=int)

    set_asr_profile = sub.add_parser("set-asr-runtime-profile", help="Persist local/OpenAI-compatible ASR runtime settings without storing API keys")
    set_asr_profile.add_argument("--config-path", default="")
    set_asr_profile.add_argument("--provider", default="")
    set_asr_profile.add_argument("--model", default="")
    set_asr_profile.add_argument("--device", default="")
    set_asr_profile.add_argument("--compute-type", default="")
    set_asr_profile.add_argument("--vad-model", default="")
    set_asr_profile.add_argument("--punc-model", default="")
    set_asr_profile.add_argument("--spk-model", default="")
    set_asr_profile.add_argument("--enable-vad", choices=["true", "false", ""], default="")
    set_asr_profile.add_argument("--enable-itn", choices=["true", "false", ""], default="")
    set_asr_profile.add_argument("--enable-punctuation", choices=["true", "false", ""], default="")
    set_asr_profile.add_argument("--enable-diarization", choices=["true", "false", ""], default="")
    set_asr_profile.add_argument("--merge-vad", choices=["true", "false", ""], default="")
    set_asr_profile.add_argument("--merge-length-s", type=int)
    set_asr_profile.add_argument("--audio-preprocess", choices=["true", "false", ""], default="")
    set_asr_profile.add_argument("--ffmpeg-normalize", choices=["true", "false", ""], default="")
    set_asr_profile.add_argument("--target-sample-rate", type=int)
    set_asr_profile.add_argument("--service-base-url", default="")
    set_asr_profile.add_argument("--service-model", default="")
    set_asr_profile.add_argument("--service-timeout-seconds", type=int)

    asr_setup = sub.add_parser("plan-asr-setup", help="Write a local SenseVoice/FunASR setup plan for a workspace")
    asr_setup.add_argument("workspace_dir")
    asr_setup.add_argument("--venv-dir", default="")

    asr_model_status = sub.add_parser("asr-model-cache-status", help="Check local ASR model caches including ct-punc/cam++ adjunct models")
    asr_model_status.add_argument("workspace_dir")
    asr_model_status.add_argument("--models", default="", help="Comma-separated models/aliases; default iic/SenseVoiceSmall,fsmn-vad,ct-punc")
    asr_model_status.add_argument("--include-optional", action="store_true", help="Also check optional cam++ speaker model")
    asr_model_status.add_argument("--no-write", action="store_true")

    asr_model_prepare = sub.add_parser("prepare-asr-model-cache", help="Preview or download/cache ASR adjunct models; execute requires explicit allow-download")
    asr_model_prepare.add_argument("workspace_dir")
    asr_model_prepare.add_argument("--models", default="", help="Comma-separated models/aliases; default iic/SenseVoiceSmall,fsmn-vad,ct-punc")
    asr_model_prepare.add_argument("--include-optional", action="store_true", help="Also prepare optional cam++ speaker model")
    asr_model_prepare.add_argument("--execute", action="store_true")
    asr_model_prepare.add_argument("--allow-download", action="store_true")
    asr_model_prepare.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    asr_model_prepare.add_argument("--timeout-seconds", type=int, default=1800)
    asr_model_prepare.add_argument("--no-write", action="store_true")

    source = sub.add_parser("prepare-video-source", help="Prepare a local video file or URL as a pipeline workspace source")
    source.add_argument("video_or_url")
    source.add_argument("workspace_dir")
    source.add_argument("--title", default="")
    source.add_argument("--execute", action="store_true", help="For URLs: execute download. For local files: copy media into workspace/source-media.")

    page_metadata = sub.add_parser("import-page-metadata", help="Import a local VDO/yt-dlp/page metadata JSON as weak source context; never fetches the page")
    page_metadata.add_argument("bundle_dir")
    page_metadata.add_argument("metadata_json")
    page_metadata.add_argument("--no-write", action="store_true")

    runtime_preflight = sub.add_parser("local-runtime-preflight", help="Inspect Python, uv, media tools, and core dependencies without executing them")
    runtime_preflight.add_argument("--project-root", default="")

    local_run = sub.add_parser("prepare-local-video-run", help="Create a human-readable run folder for one local knowledge video")
    local_run.add_argument("media_path")
    local_run.add_argument("output_dir")
    local_run.add_argument("--title", default="")
    local_run.add_argument("--copy-media", action="store_true")
    local_run.add_argument("--no-plan-asr", action="store_true")
    local_run.add_argument("--execute-asr", action="store_true")
    local_run.add_argument("--asr-preset", default="sensevoice")
    local_run.add_argument("--asr-model", default="iic/SenseVoiceSmall")
    local_run.add_argument("--transcript-path")
    local_run.add_argument("--build-initial-bundle", action="store_true")
    local_run.add_argument("--sample-interval", type=float, default=DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS)
    local_run.add_argument("--max-frames", type=int, default=DEFAULT_LOCAL_FRAME_BUDGET)
    local_run.add_argument("--sample-mode", choices=LOCAL_FRAME_SAMPLING_MODES, default=DEFAULT_LOCAL_FRAME_SAMPLING_MODE)
    local_run.add_argument("--no-scene-detect", action="store_true")
    local_run.add_argument("--no-frame-extract", action="store_true")
    local_run.add_argument("--timeout-seconds", type=int, default=1800)

    openclaw_plan = sub.add_parser("openclaw-video-plan", help="Plan a video link through video-download-orchestrator for OpenClaw")
    openclaw_plan.add_argument("url_or_text")
    openclaw_plan.add_argument("--output-dir", default="")
    openclaw_plan.add_argument("--vdo-root", default="")
    openclaw_plan.add_argument("--vdo-output-dir", default="")
    openclaw_plan.add_argument("--backend", default="")
    openclaw_plan.add_argument("--no-write-manifests", action="store_true")
    openclaw_plan.add_argument("--include-manifests", action="store_true")
    openclaw_plan.add_argument("--timeout-seconds", type=int, default=120)

    openclaw_ingest = sub.add_parser("openclaw-video-ingest", help="Prepare a local or already downloaded video for OpenClaw review")
    openclaw_ingest.add_argument("media_path")
    openclaw_ingest.add_argument("--workspace", default="")
    openclaw_ingest.add_argument("--title", default="")
    openclaw_ingest.add_argument("--copy-media", action="store_true")
    openclaw_ingest.add_argument("--no-plan-asr", action="store_true")
    openclaw_ingest.add_argument("--execute-asr", action="store_true")
    openclaw_ingest.add_argument("--asr-preset", default="sensevoice")
    openclaw_ingest.add_argument("--asr-model", default="iic/SenseVoiceSmall")
    openclaw_ingest.add_argument("--transcript-path")
    openclaw_ingest.add_argument("--no-build-initial-bundle", action="store_true")
    openclaw_ingest.add_argument("--sample-interval", type=float, default=DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS)
    openclaw_ingest.add_argument("--max-frames", type=int, default=DEFAULT_LOCAL_FRAME_BUDGET)
    openclaw_ingest.add_argument("--sample-mode", choices=LOCAL_FRAME_SAMPLING_MODES, default=DEFAULT_LOCAL_FRAME_SAMPLING_MODE)
    openclaw_ingest.add_argument("--no-scene-detect", action="store_true")
    openclaw_ingest.add_argument("--no-frame-extract", action="store_true")
    openclaw_ingest.add_argument("--timeout-seconds", type=int, default=1800)

    openclaw_link = sub.add_parser("openclaw-video-link", help="Plan a video link and optionally ingest an explicitly downloaded file")
    openclaw_link.add_argument("url_or_text")
    openclaw_link.add_argument("--output-dir", default="")
    openclaw_link.add_argument("--vdo-root", default="")
    openclaw_link.add_argument("--vdo-output-dir", default="")
    openclaw_link.add_argument("--backend", default="")
    openclaw_link.add_argument("--allow-download", action="store_true")
    openclaw_link.add_argument("--actor-id", default="")
    openclaw_link.add_argument("--confirm-download", action="store_true")
    openclaw_link.add_argument("--confirm-sensitive", action="store_true")
    openclaw_link.add_argument("--ingest-after-download", action="store_true")
    openclaw_link.add_argument("--downloaded-media-path", default="")
    openclaw_link.add_argument("--workspace", default="")
    openclaw_link.add_argument("--title", default="")
    openclaw_link.add_argument("--sample-interval", type=float, default=DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS)
    openclaw_link.add_argument("--max-frames", type=int, default=DEFAULT_LOCAL_FRAME_BUDGET)
    openclaw_link.add_argument("--sample-mode", choices=LOCAL_FRAME_SAMPLING_MODES, default=DEFAULT_LOCAL_FRAME_SAMPLING_MODE)
    openclaw_link.add_argument("--timeout-seconds", type=int, default=1800)

    openclaw_status = sub.add_parser("openclaw-bridge-status", help="Check the configured OpenClaw HTTP bridge without starting it")
    openclaw_status.add_argument("--timeout-seconds", type=float, default=2.0)
    openclaw_status.add_argument("--no-health", action="store_true", help="Only check whether the configured TCP port accepts connections")
    openclaw_status.add_argument("--no-task", action="store_true", help="Skip Windows scheduled task inspection")

    openclaw_doctor = sub.add_parser("openclaw-bridge-doctor", help="Read-only doctor report for the OpenClaw HTTP bridge lifecycle")
    openclaw_doctor.add_argument("--timeout-seconds", type=float, default=2.0)
    openclaw_doctor.add_argument("--project-root", default="")

    openclaw_smoke = sub.add_parser("openclaw-live-smoke", help="Read-only live smoke for bridge, Docker contract, and optional content card status")
    openclaw_smoke.add_argument("--bundle-dir", default="")
    openclaw_smoke.add_argument("--compose-path", default="")
    openclaw_smoke.add_argument("--host-root", default=str(workspace_root()))
    openclaw_smoke.add_argument("--container-root", default="/mnt/used-by-codex")
    openclaw_smoke.add_argument("--timeout-seconds", type=float, default=2.0)
    openclaw_smoke.add_argument("--output-dir", default="")
    openclaw_smoke.add_argument("--semantic-batch-input", default="")
    openclaw_smoke.add_argument("--semantic-target-bundle-count", type=int, default=3)
    openclaw_smoke.add_argument("--semantic-limit", type=int, default=0, help="Maximum semantic-correction bundles to inspect; 0 means all discovered")
    openclaw_smoke.add_argument("--write-report", action="store_true")

    docker_contract = sub.add_parser("openclaw-docker-contract-check", help="Read-only check for OpenClaw Docker VKP/VDO mount and env contract")
    docker_contract.add_argument("--compose-path", default="")
    docker_contract.add_argument("--host-root", default=str(workspace_root()))
    docker_contract.add_argument("--container-root", default="/mnt/used-by-codex")

    vdo_handoff = sub.add_parser("openclaw-video-from-vdo-handoff", help="Preview whether VDO artifacts are ready for VKP ingest")
    vdo_handoff.add_argument("--manifest-path", default="")
    vdo_handoff.add_argument("--summary-path", default="")
    vdo_handoff.add_argument("--review-checklist-path", default="")
    vdo_handoff.add_argument("--media-path", default="")
    vdo_handoff.add_argument("--host-root", default=str(workspace_root()))
    vdo_handoff.add_argument("--container-root", default="/mnt/used-by-codex")
    vdo_handoff.add_argument("--workspace", default="")
    vdo_handoff.add_argument("--title", default="")

    ingest_handoff = sub.add_parser("openclaw-video-ingest-vdo-handoff", help="Preview or execute VKP ingest from a ready VDO handoff")
    ingest_handoff.add_argument("--handoff-path", default="")
    ingest_handoff.add_argument("--manifest-path", default="")
    ingest_handoff.add_argument("--summary-path", default="")
    ingest_handoff.add_argument("--review-checklist-path", default="")
    ingest_handoff.add_argument("--media-path", default="")
    ingest_handoff.add_argument("--host-root", default=str(workspace_root()))
    ingest_handoff.add_argument("--container-root", default="/mnt/used-by-codex")
    ingest_handoff.add_argument("--workspace", default="")
    ingest_handoff.add_argument("--title", default="")
    ingest_handoff.add_argument("--execute", action="store_true")
    ingest_handoff.add_argument("--max-frames", type=int, default=DEFAULT_LOCAL_FRAME_BUDGET)
    ingest_handoff.add_argument("--sample-interval", type=float, default=DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS)
    ingest_handoff.add_argument("--sample-mode", choices=LOCAL_FRAME_SAMPLING_MODES, default=DEFAULT_LOCAL_FRAME_SAMPLING_MODE)

    acceptance_run = sub.add_parser("acceptance-run", help="Run the local acceptance workflow for one knowledge video")
    acceptance_run.add_argument("media_path")
    acceptance_run.add_argument("output_dir")
    acceptance_run.add_argument("--title", default="")
    acceptance_run.add_argument("--copy-media", action="store_true")
    acceptance_run.add_argument("--execute-asr", action="store_true")
    acceptance_run.add_argument("--asr-preset", default="sensevoice")
    acceptance_run.add_argument("--asr-model", default="iic/SenseVoiceSmall")
    acceptance_run.add_argument("--transcript-path")
    acceptance_run.add_argument("--no-build-initial-bundle", action="store_true")
    acceptance_run.add_argument("--sample-interval", type=float, default=DEFAULT_LOCAL_FRAME_SAMPLE_INTERVAL_SECONDS)
    acceptance_run.add_argument("--max-frames", type=int, default=DEFAULT_LOCAL_FRAME_BUDGET)
    acceptance_run.add_argument("--sample-mode", choices=LOCAL_FRAME_SAMPLING_MODES, default=DEFAULT_LOCAL_FRAME_SAMPLING_MODE)
    acceptance_run.add_argument("--no-scene-detect", action="store_true")
    acceptance_run.add_argument("--no-frame-extract", action="store_true")
    acceptance_run.add_argument("--execute-temporal-groups", action="store_true")
    acceptance_run.add_argument("--execute-vision", action="store_true")
    acceptance_run.add_argument("--execute-ebook-pipeline", action="store_true")
    acceptance_run.add_argument("--semantic-limit", type=int)
    acceptance_run.add_argument("--temporal-limit", type=int)
    acceptance_run.add_argument("--frame-count", type=int)
    acceptance_run.add_argument("--provider-config")
    acceptance_run.add_argument("--confirm-vision-calls", type=int)
    acceptance_run.add_argument("--confirm-vision-indexes", default="")
    acceptance_run.add_argument("--timeout-seconds", type=int, default=1800)

    acceptance_bundle = sub.add_parser("acceptance-bundle-run", help="Run the acceptance workflow from an existing webui-bundle")
    acceptance_bundle.add_argument("bundle_dir")
    acceptance_bundle.add_argument("--output-dir", default="")
    acceptance_bundle.add_argument("--title", default="")
    acceptance_bundle.add_argument("--execute-temporal-groups", action="store_true")
    acceptance_bundle.add_argument("--execute-vision", action="store_true")
    acceptance_bundle.add_argument("--execute-ebook-pipeline", action="store_true")
    acceptance_bundle.add_argument("--semantic-limit", type=int)
    acceptance_bundle.add_argument("--temporal-limit", type=int)
    acceptance_bundle.add_argument("--frame-count", type=int)
    acceptance_bundle.add_argument("--provider-config")
    acceptance_bundle.add_argument("--confirm-vision-calls", type=int)
    acceptance_bundle.add_argument("--confirm-vision-indexes", default="")

    batch = sub.add_parser("batch-run", help="Run a preview-safe batch manifest over multiple local videos or bundles")
    batch.add_argument("batch_manifest")
    batch.add_argument("--resume", action="store_true")
    batch.add_argument("--force-reexport", action="store_true")
    batch.add_argument("--execute-asr", action="store_true")
    batch.add_argument("--execute-temporal-groups", action="store_true")
    batch.add_argument("--execute-vision", action="store_true")
    batch.add_argument("--execute-ebook-pipeline", action="store_true")
    batch.add_argument("--semantic-limit", type=int)
    batch.add_argument("--temporal-limit", type=int)
    batch.add_argument("--frame-count", type=int)
    batch.add_argument("--timeout-seconds", type=int, default=1800)
    batch.add_argument("--no-write", action="store_true")

    repair_batch = sub.add_parser("batch-repair-run", help="Plan or execute batch next-action repair across existing bundles")
    repair_batch.add_argument("batch_manifest_or_summary")
    repair_batch.add_argument("--execute", action="store_true")
    repair_batch.add_argument("--limit", type=int, default=0)
    repair_batch.add_argument("--max-rounds", type=int, default=1)
    repair_batch.add_argument("--allow-asr", action="store_true")
    repair_batch.add_argument("--allow-vision", action="store_true")
    repair_batch.add_argument("--allow-ocr", action="store_true")
    repair_batch.add_argument("--no-write", action="store_true")

    asr_plan = sub.add_parser("plan-asr", help="Plan a local ASR run")
    asr_plan.add_argument("workspace_dir")
    asr_plan.add_argument("media_path")
    asr_plan.add_argument("--preset", default="sensevoice")
    asr_plan.add_argument("--model", default="")
    asr_plan.add_argument("--language", default="zh")
    asr_plan.add_argument("--punc-model", default="__default__", help="Use __default__ for ct-punc on FunASR/SenseVoice; empty string disables punctuation")
    asr_plan.add_argument("--spk-model", default="", help="Optional speaker model such as cam++")
    asr_plan.add_argument("--hotword", default="", help="Evidence-derived ASR hotwords; never derive from evaluation references")
    asr_plan.add_argument("--no-use-itn", action="store_true")
    asr_plan.add_argument("--no-merge-vad", action="store_true")
    asr_plan.add_argument("--merge-length-s", type=int, default=15)
    asr_plan.add_argument("--vad-max-single-segment-time-ms", type=int, default=30000)
    asr_plan.add_argument("--chunk-boundary-mode", choices=("fixed_duration", "silence_snap"), default="fixed_duration")
    asr_plan.add_argument("--chunk-overlap-seconds", type=float, default=5.0)
    asr_plan.add_argument("--qwen-timestamp-basis", choices=("chunk_relative", "media_absolute"), default="chunk_relative")
    asr_plan.add_argument("--vad-json", default="", help="Optional local speech-VAD evidence for Qwen window priority; no window is dropped")
    asr_plan.add_argument("--transcript-path", default="", help="Required by qwen3-forced-aligner; existing transcript to align")
    asr_plan.add_argument("--alignment-chunk-seconds", type=int, default=None,
                          help="Qwen3 forced alignment only: at least 30 seconds per chunk (default: 300)")

    speaker_global = sub.add_parser(
        "speaker-global-align",
        help="Map local CAM++ chunk clusters to recording-global anonymous speaker IDs",
    )
    speaker_global.add_argument("chunked_output")
    speaker_global.add_argument("--similarity-threshold", type=float, default=0.60)
    speaker_global.add_argument("--overlap-agreement-threshold", type=float, default=0.80)
    speaker_global.add_argument("--max-speakers", type=int, default=15)
    speaker_global.add_argument(
        "--expected-speakers",
        type=int,
        default=None,
        help="Operator-confirmed recording participant count; repairs over-splitting without inferring roles",
    )
    speaker_global.add_argument("--no-write", action="store_true")

    speaker_centers = sub.add_parser(
        "speaker-center-sidecar",
        help="Extract local CAM++ centers from existing timed ASR without rerunning ASR",
    )
    speaker_centers.add_argument("chunked_output")
    speaker_centers.add_argument("media_path")
    speaker_centers.add_argument("output_dir")
    speaker_centers.add_argument("--expected-speakers", type=int, default=None)
    speaker_centers.add_argument("--device", choices=("cuda",), default="cuda")
    speaker_centers.add_argument("--max-evidence-seconds", type=float, default=30.0)
    speaker_centers.add_argument("--execute", action="store_true")
    speaker_centers.add_argument("--no-write", action="store_true")

    shared_session = sub.add_parser(
        "speaker-shared-session-align",
        help="Align recordings explicitly confirmed to share one participant set",
    )
    shared_session.add_argument("output_dir")
    shared_session.add_argument("--private-centers", action="append", required=True)
    shared_session.add_argument(
        "--candidate-transcript", dest="candidate_transcripts", action="append", required=True
    )
    shared_session.add_argument("--expected-speakers", type=int, required=True)
    shared_session.add_argument("--confirm-shared-participant-set", action="store_true")
    shared_session.add_argument("--no-write", action="store_true")

    voiceprint = sub.add_parser(
        "speaker-voiceprint",
        help="Explicit local-only enroll/match/delete operations for anonymous voiceprints",
    )
    voiceprint.add_argument(
        "action", choices=("enroll", "match", "bind-role", "delete")
    )
    voiceprint.add_argument("--private-alignment", default="")
    voiceprint.add_argument("--registry", required=True)
    voiceprint.add_argument("--source-id", default="")
    voiceprint.add_argument("--voiceprint-id", default="")
    voiceprint.add_argument("--role-label", default="")
    voiceprint.add_argument("--similarity-threshold", type=float, default=0.72)
    voiceprint.add_argument("--confirm-local-biometric-storage", action="store_true")
    voiceprint.add_argument("--confirm-role-binding", action="store_true")
    voiceprint.add_argument("--confirm-delete", action="store_true")
    voiceprint.add_argument("--no-write", action="store_true")

    asr_retry = sub.add_parser("asr-retry-snippets", help="Plan or extract only ASR quality-gate retry windows")
    asr_retry.add_argument("media_path")
    asr_retry.add_argument("quality_report_json")
    asr_retry.add_argument("output_dir")
    asr_retry.add_argument("--ffmpeg", default="ffmpeg")
    asr_retry.add_argument("--execute", action="store_true")

    local_target_plan = sub.add_parser(
        "plan-local-targeted-asr-evidence",
        help="Plan bounded independent local ASR clips for unresolved factual semantic candidates",
    )
    local_target_plan.add_argument("bundle_dir")
    local_target_plan.add_argument("--input-pack", default="")
    local_target_plan.add_argument("--max-windows", type=int, default=24)
    local_target_plan.add_argument("--padding-seconds", type=float, default=3.0)
    local_target_plan.add_argument("--no-write", action="store_true")

    local_target_execution = sub.add_parser(
        "run-local-targeted-asr-evidence",
        help="Plan or execute bounded local second-ASR evidence and register it without replacing the canonical transcript",
    )
    local_target_execution.add_argument("bundle_dir")
    local_target_execution.add_argument("--media-path", default="", help="Optional existing local media path; bundle manifest is used when omitted")
    local_target_execution.add_argument("--input-plan", default="", help="Optional local-targeted-asr-plan.json path")
    local_target_execution.add_argument("--preset", default="qwen3-asr-0.6b")
    local_target_execution.add_argument("--model", default="")
    local_target_execution.add_argument("--language", default="zh")
    local_target_execution.add_argument("--timeout-seconds", type=int, default=900)
    local_target_execution.add_argument("--max-windows", type=int, default=24)
    local_target_execution.add_argument("--padding-seconds", type=float, default=3.0)
    local_target_execution.add_argument("--allow-cpu", action="store_true")
    local_target_execution.add_argument("--execute", action="store_true")
    local_target_execution.add_argument("--no-write", action="store_true")

    local_targeted = sub.add_parser(
        "asr-local-targeted-evidence",
        help="Build candidate-only global-time evidence from verified local clip ASR outputs",
    )
    local_targeted.add_argument("output_json")
    local_targeted.add_argument("--snippet-manifest", action="append", required=True)
    local_targeted.add_argument("--raw-output", action="append", required=True)
    local_targeted.add_argument("--allow-cpu", action="store_true")
    local_targeted.add_argument("--write", action="store_true")

    asr_ab = sub.add_parser("asr-ab-sample-plan", help="Plan a 5-minute ASR A/B sample across SenseVoice/CAM++, MOSS, Dolphin, and optional cloud ASR")
    asr_ab.add_argument("workspace_dir")
    asr_ab.add_argument("media_path")
    asr_ab.add_argument("--sample-start-seconds", type=float, default=0.0)
    asr_ab.add_argument("--duration-seconds", type=float, default=300.0)
    asr_ab.add_argument("--language", default="zh")
    asr_ab.add_argument("--cloud-provider-config", default="", help="Optional runtime provider config for the cloud ASR plan; secrets are not written")
    asr_ab.add_argument("--no-write", action="store_true")

    asr_ab_run = sub.add_parser("asr-ab-sample-run", help="Preview or execute a bounded ASR A/B sample; does not promote transcripts")
    asr_ab_run.add_argument("workspace_dir")
    asr_ab_run.add_argument("media_path", nargs="?", default="")
    asr_ab_run.add_argument("--plan-json", default="")
    asr_ab_run.add_argument("--sample-start-seconds", type=float, default=0.0)
    asr_ab_run.add_argument("--duration-seconds", type=float, default=300.0)
    asr_ab_run.add_argument("--language", default="zh")
    asr_ab_run.add_argument("--execute-sample", action="store_true", help="Extract the bounded local sample with ffmpeg")
    asr_ab_run.add_argument("--execute-local", action="store_true", help="Run local SenseVoice variants on the sample")
    asr_ab_run.add_argument("--execute-cloud", action="store_true", help="Upload only the extracted sample to the configured cloud ASR provider")
    asr_ab_run.add_argument("--cloud-provider-config", default="", help="Runtime provider config for cloud ASR; secrets are not written")
    asr_ab_run.add_argument("--variants", default="", help="Comma-separated subset, e.g. sensevoice_full_punc_campp,moss_transcribe_diarize")
    asr_ab_run.add_argument("--timeout-seconds", type=int, default=1800)
    asr_ab_run.add_argument("--no-write", action="store_true")

    asr_ab_compare = sub.add_parser("asr-ab-compare", help="Compare a completed ASR A/B run and recommend whether a second ASR is ready")
    asr_ab_compare.add_argument("run_json")
    asr_ab_compare.add_argument("--reference-transcript", default="", help="Evaluation-only transcript path; never imported as correction evidence")
    asr_ab_compare.add_argument("--start-seconds", type=float, default=0.0)
    asr_ab_compare.add_argument("--end-seconds", type=float, default=0.0)
    asr_ab_compare.add_argument("--no-write", action="store_true")

    transcript_reference_window = sub.add_parser(
        "transcript-reference-window",
        help="Export a bounded, speaker-preserving evaluation reference without promoting transcript text",
    )
    transcript_reference_window.add_argument("transcript_path")
    transcript_reference_window.add_argument("output_json")
    transcript_reference_window.add_argument("--start-seconds", type=float, required=True)
    transcript_reference_window.add_argument("--end-seconds", type=float, required=True)
    transcript_reference_window.add_argument(
        "--absolute-timestamps",
        action="store_true",
        help="Keep original timestamps instead of rebasing the window to zero",
    )
    transcript_reference_window.add_argument(
        "--human-corrections-json",
        default="",
        help="Exact source-SHA-bound, segment-scoped human-confirmed corrections",
    )
    transcript_reference_window.add_argument("--no-write", action="store_true")

    recall_benchmark = sub.add_parser("transcript-candidate-recall-benchmark", help="Benchmark ASR variant quality and semantic-correction candidate recall against an evaluation-only reference")
    recall_benchmark.add_argument("bundle_dir")
    recall_benchmark.add_argument("--reference-transcript", default="", help="Evaluation-only transcript path; never imported as correction evidence")
    recall_benchmark.add_argument("--source-transcript", default="", help="Optional ASR/source transcript to evaluate; defaults to raw/normalized bundle transcript")
    recall_benchmark.add_argument("--target-pairs-json", default="", help="Optional list/object of original_text -> corrected_text target pairs")
    recall_benchmark.add_argument("--asr-ab-run-json", default="", help="Optional asr-ab-sample-run.json for variant scoring")
    recall_benchmark.add_argument("--start-seconds", type=float, default=0.0)
    recall_benchmark.add_argument("--end-seconds", type=float, default=0.0)
    recall_benchmark.add_argument("--no-write", action="store_true")

    adaptive_asr = sub.add_parser(
        "adaptive-asr-route",
        help="Build local/online ASR plans from filtered pre-ASR OCR, title, and explicit lexicon terms",
    )
    adaptive_asr.add_argument("bundle_dir")
    adaptive_asr.add_argument("media_path")
    adaptive_asr.add_argument("--workspace-dir", default="")
    adaptive_asr.add_argument("--task-profile", choices=["balanced", "accuracy", "latency", "privacy", "terminology"], default="balanced")
    adaptive_asr.add_argument("--base-lexicon-json", default="")
    adaptive_asr.add_argument("--include-online-plan", action="store_true")
    adaptive_asr.add_argument("--provider-config", default="")
    adaptive_asr.add_argument("--online-model", default="")
    adaptive_asr.add_argument("--language", default="zh")
    adaptive_asr.add_argument("--max-hotwords", type=int, default=80)
    adaptive_asr.add_argument("--max-context-chars", type=int, default=1200)
    adaptive_asr.add_argument("--no-write", action="store_true")

    cloud_audio = sub.add_parser("prepare-cloud-asr-audio", help="Prepare a smaller local speech MP3 candidate before an explicitly authorised cloud ASR call")
    cloud_audio.add_argument("media_path")
    cloud_audio.add_argument("--output-path", default="")
    cloud_audio.add_argument("--bitrate-kbps", type=int, default=32)
    cloud_audio.add_argument("--sample-rate-hz", type=int, default=16000)
    cloud_audio.add_argument("--channels", type=int, default=1)
    cloud_audio.add_argument("--timeout-seconds", type=int, default=1800)
    cloud_audio.add_argument(
        "--receipt-bundle-dir",
        default="",
        help="Attach a validated FFmpeg execution receipt to this Bundle after local execution",
    )
    cloud_audio.add_argument("--execute", action="store_true", help="Run local FFmpeg only; never uploads")

    cloud_chunks = sub.add_parser(
        "prepare-cloud-asr-chunks",
        help="Prepare VAD-aligned local ASR request chunks; never calls a provider",
    )
    cloud_chunks.add_argument("media_path")
    cloud_chunks.add_argument("vad_json")
    cloud_chunks.add_argument("output_dir")
    cloud_chunks.add_argument("--bitrate-kbps", type=int, default=64)
    cloud_chunks.add_argument("--sample-rate-hz", type=int, default=16000)
    cloud_chunks.add_argument("--channels", type=int, default=1)
    cloud_chunks.add_argument("--max-request-seconds", type=float, default=180.0)
    cloud_chunks.add_argument("--context-padding-seconds", type=float, default=1.5)
    cloud_chunks.add_argument("--timeout-seconds", type=int, default=1800)
    cloud_chunks.add_argument("--execute", action="store_true", help="Run local FFmpeg only; never uploads")

    vad_activity = sub.add_parser(
        "asr-vad-activity-audit",
        help="Compare FunASR VAD with local FFmpeg non-silent audio candidate evidence",
    )
    vad_activity.add_argument("media_path")
    vad_activity.add_argument("vad_json")
    vad_activity.add_argument("--output-path", default="")
    vad_activity.add_argument("--duration-seconds", type=float, default=0.0)
    vad_activity.add_argument("--noise-db", type=float, default=-45.0)
    vad_activity.add_argument(
        "--minimum-silence-seconds", type=float, default=0.5
    )
    vad_activity.add_argument(
        "--minimum-uncovered-seconds", type=float, default=2.0
    )
    vad_activity.add_argument(
        "--execute", action="store_true", help="Run local FFprobe/FFmpeg only"
    )
    vad_activity.add_argument("--no-write", action="store_true")

    vad_compare = sub.add_parser(
        "asr-vad-profile-compare",
        help="Compare authoritative and candidate-permissive FSMN-VAD outputs locally",
    )
    vad_compare.add_argument("authoritative_vad")
    vad_compare.add_argument("permissive_vad")
    vad_compare.add_argument("activity_audit")
    vad_compare.add_argument("--labels-path", default="")
    vad_compare.add_argument("--output-path", default="")
    vad_compare.add_argument(
        "--minimum-support-ratio",
        type=float,
        default=0.5,
        help="Candidate-gap coverage needed to mark same-model permissive support",
    )
    vad_compare.add_argument("--no-write", action="store_true")

    silero_candidate = sub.add_parser(
        "silero-vad-candidate",
        help=(
            "Run the installed faster-whisper bundled Silero VAD as local "
            "candidate-only evidence"
        ),
    )
    silero_candidate.add_argument("media_path")
    silero_candidate.add_argument("--output-path", default="")
    silero_candidate.add_argument("--threshold", type=float, default=0.5)
    silero_candidate.add_argument("--neg-threshold", type=float, default=None)
    silero_candidate.add_argument("--min-speech-duration-ms", type=int, default=0)
    silero_candidate.add_argument(
        "--max-speech-duration-seconds", type=float, default=0.0
    )
    silero_candidate.add_argument(
        "--min-silence-duration-ms", type=int, default=2000
    )
    silero_candidate.add_argument("--speech-pad-ms", type=int, default=400)
    silero_candidate.add_argument(
        "--execute", action="store_true", help="Run only the installed local VAD"
    )
    silero_candidate.add_argument("--no-write", action="store_true")

    independent_vad = sub.add_parser(
        "asr-vad-independent-crosscheck",
        help=(
            "Compare authoritative FunASR VAD with candidate-only Silero evidence; "
            "never changes the transcript"
        ),
    )
    independent_vad.add_argument("authoritative_vad")
    independent_vad.add_argument("candidate_vad")
    independent_vad.add_argument("--activity-audit", default="")
    independent_vad.add_argument("--output-path", default="")
    independent_vad.add_argument("--minimum-gap-seconds", type=float, default=2.0)
    independent_vad.add_argument("--no-write", action="store_true")

    chunk_workflow = sub.add_parser(
        "asr-chunk-batch-workflow",
        help="Compile exact ASR chunk consents for the existing Broker workflow; never submits",
    )
    chunk_workflow.add_argument("chunk_manifest")
    chunk_workflow.add_argument(
        "--consent-path",
        action="append",
        required=True,
        help="Exact consent v2 path in chunk order; repeat once per chunk",
    )
    chunk_workflow.add_argument("--output-path", default="")
    chunk_workflow.add_argument("--bundle-dir", default="")
    chunk_workflow.add_argument("--activity-audit", default="")
    chunk_workflow.add_argument("--max-parallel-global", type=int, default=4)
    chunk_workflow.add_argument(
        "--max-parallel-per-destination", type=int, default=2
    )
    chunk_workflow.add_argument("--no-write", action="store_true")

    chunk_business_workflow = sub.add_parser(
        "asr-chunk-business-workflow",
        help=(
            "Reuse one confirmed business authorization to mint exact chunk "
            "consents and compile the existing Broker workflow"
        ),
    )
    chunk_business_workflow.add_argument("chunk_manifest")
    chunk_business_workflow.add_argument("authorization_path")
    chunk_business_workflow.add_argument("--stage-id", required=True)
    chunk_business_workflow.add_argument("--producer", required=True)
    chunk_business_workflow.add_argument(
        "--lineage-input",
        action="append",
        required=True,
        help="Exact parent source or prior admitted artifact; repeat as needed",
    )
    chunk_business_workflow.add_argument("--output-path", default="")
    chunk_business_workflow.add_argument("--bundle-dir", default="")
    chunk_business_workflow.add_argument("--activity-audit", default="")
    chunk_business_workflow.add_argument(
        "--max-parallel-global", type=int, default=4
    )
    chunk_business_workflow.add_argument(
        "--max-parallel-per-destination", type=int, default=2
    )
    chunk_business_workflow.add_argument("--no-write", action="store_true")

    chunk_submit = sub.add_parser(
        "asr-chunk-batch-submit",
        help="Revalidate and explicitly submit an ASR chunk workflow to the loopback Broker",
    )
    chunk_submit.add_argument("workflow_path")
    chunk_submit.add_argument(
        "--broker-url",
        default="http://127.0.0.1:8766/mcp",
        help="Loopback-only Streamable HTTP MCP endpoint",
    )
    chunk_submit.add_argument(
        "--execute", action="store_true", help="Submit to the local Broker"
    )

    chunk_status = sub.add_parser(
        "asr-chunk-batch-status",
        help="Read durable ASR chunk batch status from the loopback Broker",
    )
    chunk_status.add_argument("job_id")
    chunk_status.add_argument(
        "--broker-url",
        default="http://127.0.0.1:8766/mcp",
        help="Loopback-only Streamable HTTP MCP endpoint",
    )
    chunk_status.add_argument("--output-path", default="")

    chunk_merge = sub.add_parser(
        "asr-chunk-batch-merge",
        help="Merge saved Trusted Connector ASR chunk reports; never calls a provider",
    )
    chunk_merge.add_argument("workflow_path")
    chunk_merge.add_argument(
        "--execution-report",
        action="append",
        default=[],
        help="Saved execution report path; repeat once per available chunk",
    )
    chunk_merge.add_argument("--batch-status-path", default="")
    chunk_merge.add_argument("--output-dir", default="")
    chunk_merge.add_argument("--title", default="")
    chunk_merge.add_argument(
        "--prepare-alignment-plan",
        action="store_true",
        help="Plan the existing Qwen3 ForcedAligner route after a completed merge",
    )
    chunk_merge.add_argument("--alignment-language", default="zh")
    chunk_merge.add_argument(
        "--alignment-model",
        default="",
        help="Optional existing local Qwen3 ForcedAligner model/path override",
    )
    chunk_merge.add_argument("--no-write", action="store_true")

    cloud_asr_plan = sub.add_parser("plan-cloud-asr", help="Plan optional cloud ASR upload via online-model-api; default never uploads")
    cloud_asr_plan.add_argument("workspace_dir")
    cloud_asr_plan.add_argument("media_path")
    cloud_asr_plan.add_argument("--provider-config", help="Inline JSON or JSON file with provider/base_url/model/api_key; key is not written to the plan")
    cloud_asr_plan.add_argument("--model", default="gpt-4o-transcribe")
    cloud_asr_plan.add_argument("--language", default="zh")
    cloud_asr_plan.add_argument("--prompt", default="")

    cloud_asr_run = sub.add_parser("run-cloud-asr-plan", help="Preview or execute a planned cloud ASR call")
    cloud_asr_run.add_argument("plan_json")
    cloud_asr_run.add_argument("--provider-config", help="Runtime-only provider override; use env vars for API keys")
    cloud_asr_run.add_argument("--execute", action="store_true", help="Actually upload audio to the configured cloud ASR provider")
    cloud_asr_run.add_argument("--no-normalize", action="store_true")

    local_asr_service_plan = sub.add_parser("plan-local-asr-service", help="Plan Speaches/OpenAI-compatible local ASR service run; default never calls the service")
    local_asr_service_plan.add_argument("workspace_dir")
    local_asr_service_plan.add_argument("media_path")
    local_asr_service_plan.add_argument("--provider-config", default="", help="Inline JSON or JSON file override; secrets are not written")
    local_asr_service_plan.add_argument("--model", default="")
    local_asr_service_plan.add_argument("--language", default="zh")
    local_asr_service_plan.add_argument("--prompt", default="")

    local_asr_service_run = sub.add_parser("run-local-asr-service-plan", help="Preview or execute a planned Speaches/OpenAI-compatible ASR service call")
    local_asr_service_run.add_argument("plan_json")
    local_asr_service_run.add_argument("--provider-config", default="", help="Runtime-only provider override; use env vars for API keys")
    local_asr_service_run.add_argument("--execute", action="store_true", help="Actually send audio to the configured ASR service")
    local_asr_service_run.add_argument("--allow-remote", action="store_true", help="Allow non-localhost OpenAI-compatible ASR endpoint")
    local_asr_service_run.add_argument("--no-normalize", action="store_true")

    whisperx_plan = sub.add_parser("plan-whisperx-alignment", help="Plan a WhisperX word-level alignment enhancement run")
    whisperx_plan.add_argument("workspace_dir")
    whisperx_plan.add_argument("media_path")
    whisperx_plan.add_argument("--model", default="")
    whisperx_plan.add_argument("--language", default="zh")

    whisperx_run = sub.add_parser("run-whisperx-alignment", help="Preview or execute WhisperX as alignment evidence without replacing primary ASR")
    whisperx_run.add_argument("workspace_dir")
    whisperx_run.add_argument("media_path")
    whisperx_run.add_argument("--model", default="large-v3")
    whisperx_run.add_argument("--language", default="zh")
    whisperx_run.add_argument("--execute", action="store_true")
    whisperx_run.add_argument("--timeout-seconds", type=int, default=1800)
    whisperx_run.add_argument("--no-write", action="store_true")

    asr_run = sub.add_parser("run-asr-plan", help="Execute or preview a planned ASR command")
    asr_run.add_argument("plan_json")
    asr_run.add_argument("--execute", action="store_true")
    asr_run.add_argument("--timeout-seconds", type=int, default=1800)
    asr_run.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Ignore a matching Qwen chunk checkpoint and execute requested chunks again",
    )
    asr_run.set_defaults(resume=True)

    asr_smoke_parser = sub.add_parser("asr-smoke", help="Run or preview a local-only short ASR smoke test")
    asr_smoke_parser.add_argument("media_path")
    asr_smoke_parser.add_argument("--output-dir", default="")
    asr_smoke_parser.add_argument("--preset", default="sensevoice")
    asr_smoke_parser.add_argument("--model", default="")
    asr_smoke_parser.add_argument("--language", default="zh")
    asr_smoke_parser.add_argument("--duration-seconds", type=int, default=30)
    asr_smoke_parser.add_argument("--execute", action="store_true", dest="execute")
    asr_smoke_parser.add_argument("--no-execute", action="store_false", dest="execute")
    asr_smoke_parser.set_defaults(execute=True)
    asr_smoke_parser.add_argument("--timeout-seconds", type=int, default=600)

    normalize = sub.add_parser("normalize-asr", help="Normalize ASR JSON/SRT/TXT output to pipeline transcript JSON and SRT")
    normalize.add_argument("workspace_dir")
    normalize.add_argument("input_path")
    normalize.add_argument("--provider", default="auto")
    normalize.add_argument("--title", default="")

    variants = sub.add_parser("compare-asr-variants", help="Compare primary, secondary and integrated ASR on human-verified references")
    variants.add_argument("reference_json")
    variants.add_argument("primary")
    variants.add_argument("secondary")
    variants.add_argument("integrated")
    variants.add_argument("--output-json", default="")

    dual = sub.add_parser("dual-asr-review", help="Resume local full or targeted independent ASR and evidence review")
    dual.add_argument("bundle_dir")
    dual.add_argument("--media-path", default="")
    dual.add_argument("--primary-transcript", default="")
    dual.add_argument("--secondary-transcript", default="")
    dual.add_argument("--mode", choices=["daily", "important"], default="important")
    dual.add_argument("--secondary-preset", choices=["qwen3-asr-1.7b", "qwen3-asr-0.6b"], default="qwen3-asr-1.7b")
    dual.add_argument("--secondary-timestamps", choices=["auto", "required", "window"], default="auto",
                      help="Separate local transcription from optional forced alignment; window output never claims word timing")
    dual.add_argument("--speech-activity-json", default="")
    dual.add_argument("--diarization-json", default="")
    dual.add_argument("--execute", action="store_true")
    dual.add_argument("--allow-cpu", action="store_true")
    dual.add_argument("--max-windows", type=int, default=24)
    dual.add_argument("--timeout-seconds", type=int, default=1800)
    dual.add_argument("--no-recover-failed-windows", action="store_true",
                      help="Keep exhausted Qwen windows pending instead of creating a bounded smaller-window child plan")
    dual.add_argument("--recovery-window-seconds", type=int, default=60)
    dual.add_argument("--recovery-max-windows", type=int, default=24)
    dual.add_argument("--no-write", action="store_true")

    recovery_plan = sub.add_parser("plan-qwen-window-recovery", help="Plan bounded child windows from an exhausted Qwen checkpoint; no model calls")
    recovery_plan.add_argument("partial_json")
    recovery_plan.add_argument("output_dir")
    recovery_plan.add_argument("--asr-plan-json", default="")
    recovery_plan.add_argument("--media-path", default="")
    recovery_plan.add_argument("--speech-activity-json", default="")
    recovery_plan.add_argument("--window-seconds", type=int, default=60)
    recovery_plan.add_argument("--max-windows", type=int, default=24)
    recovery_plan.add_argument("--no-write", action="store_true")
    recovery_compose = sub.add_parser("compose-qwen-window-recovery", help="Validate and compose retained and cached child-window outputs; no model calls")
    recovery_compose.add_argument("plan_json")
    recovery_compose.add_argument("recovered_json")
    recovery_compose.add_argument("--no-write", action="store_true")

    consensus = sub.add_parser("asr-consensus", help="Align two independent ASR hypotheses and retain agreement/conflict windows")
    consensus.add_argument("bundle_dir")
    consensus.add_argument("primary_transcript")
    consensus.add_argument("secondary_transcript")
    consensus.add_argument("--media-path", default="")
    consensus.add_argument("--agreement-threshold", type=float, default=0.86)
    consensus.add_argument("--speech-activity-json", default="")
    consensus.add_argument("--diarization-json", default="")
    consensus.add_argument("--critical-term", action="append", default=[])
    consensus.add_argument("--agreement-sample-size", type=int, default=3)
    consensus.add_argument("--secondary-scope", choices=["full", "targeted"], default="full")
    consensus.add_argument("--execute-clips", action="store_true", help="Generate 8-30 second local audio evidence clips with ffmpeg")
    consensus.add_argument("--no-write", action="store_true")

    secondary_evidence = sub.add_parser(
        "asr-secondary-evidence",
        help="Validate and close one saved secondary ASR execution as local-only review evidence",
    )
    secondary_evidence.add_argument("bundle_dir")
    secondary_evidence.add_argument("connector_execution")
    secondary_evidence.add_argument("prepared_suite")
    secondary_evidence.add_argument("--candidate-id", default="")
    secondary_evidence.add_argument("--primary-transcript", default="")
    secondary_evidence.add_argument("--media-path", default="")
    secondary_evidence.add_argument("--agreement-threshold", type=float, default=0.86)
    secondary_evidence.add_argument("--no-write", action="store_true")

    diff_adjudication = sub.add_parser("asr-diff-adjudication", help="Build positioned, clustered and anonymous ASR disagreement candidates")
    diff_adjudication.add_argument("bundle_dir")
    diff_adjudication.add_argument("--consensus-json", default="")
    diff_adjudication.add_argument("--cluster-token-gap", type=int, default=6)
    diff_adjudication.add_argument("--no-write", action="store_true")

    apply_diff = sub.add_parser("apply-asr-diff-adjudication", help="Validate and apply local ASR disagreement patches")
    apply_diff.add_argument("bundle_dir")
    apply_diff.add_argument("decisions_json")
    apply_diff.add_argument("--pack-json", default="")
    apply_diff.add_argument("--min-confidence", type=float, default=0.75)
    apply_diff.add_argument("--allow-without-evidence", action="store_true")
    apply_diff.add_argument("--promote", action="store_true")
    apply_diff.add_argument("--review-only", action="store_true", help="Close every positioned difference with hash-bound evidence; never modify/promote transcript text")
    apply_diff.add_argument("--no-write", action="store_true")

    evidence_autoadjudication = sub.add_parser(
        "asr-evidence-autoadjudication",
        help="Conservatively patch canonical ASR only where independent overlapping ASR exactly corroborates a non-fact delta",
    )
    evidence_autoadjudication.add_argument("bundle_dir")
    evidence_autoadjudication.add_argument("secondary_transcript")
    evidence_autoadjudication.add_argument(
        "--corroborating-transcript",
        action="append",
        required=True,
        help="Independent transcript JSON; repeat for additional evidence sources",
    )
    evidence_autoadjudication.add_argument(
        "--write",
        action="store_true",
        help="Write the conservative patches into the canonical transcript; default is preview only",
    )
    evidence_autoadjudication.add_argument(
        "--refresh-exports",
        action="store_true",
        help="After --write, refresh full transcript, summaries, knowledge note and canonical/export integrity reports",
    )

    punctuation_stage = sub.add_parser("punctuation-model-stage", help="Run local FunASR ct-punc with a strict character lock")
    punctuation_stage.add_argument("bundle_dir")
    punctuation_stage.add_argument("--input-path", default="")
    punctuation_stage.add_argument("--model", default="ct-punc")
    punctuation_stage.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    punctuation_stage.add_argument("--block-chars", type=int, default=480)
    punctuation_stage.add_argument("--execute", action="store_true")
    punctuation_stage.add_argument("--promote", action="store_true")
    punctuation_stage.add_argument("--no-write", action="store_true")

    offline_route = sub.add_parser("offline-quality-route", help="Build local-only ASR/OCR/vision quality status and fallback proposal")
    offline_route.add_argument("bundle_dir")
    offline_route.add_argument("--benchmark-manifest", default="", help="Optional quality benchmark manifest used only to report human-review state")
    offline_route.add_argument("--output-dir", default="")
    offline_route.add_argument("--no-write", action="store_true")

    quality_finalize = sub.add_parser("quality-finalize", help="Gate corrected transcript, build semantic chapters, run chapter LLM rewrite, and validate final smart summary")
    quality_finalize.add_argument("bundle_dir")
    quality_finalize.add_argument("--provider-config", default="", help="Runtime-only provider JSON/profile; never persisted")
    quality_finalize.add_argument("--execute-llm", action="store_true", help="Actually call the configured text LLM")
    quality_finalize.add_argument("--auto-from-profile", action="store_true", help="Allow configured quality profile to execute below preflight thresholds")
    quality_finalize.add_argument("--quality-profile", default="quality")
    quality_finalize.add_argument("--target-chapters", type=int, default=8)
    quality_finalize.add_argument("--no-write", action="store_true")

    transcript_refresh = sub.add_parser("refresh-transcript-downstream", help="Invalidate stale summaries, rebuild local transcript-derived outputs, and rerun final quality gates without calling a model")
    transcript_refresh.add_argument("bundle_dir")
    transcript_refresh.add_argument("--canonical-before-sha256", default="")
    transcript_refresh.add_argument("--canonical-after-sha256", default="")
    transcript_refresh.add_argument("--reason", default="operator_refresh")
    transcript_refresh.add_argument("--no-write", action="store_true")

    benchmark = sub.add_parser("quality-benchmark", help="Build, run, or render the fixed human-reference transcript quality benchmark")
    benchmark.add_argument("action", choices=["build", "execute-variants", "run", "report", "build-arbitration", "evaluate-arbitration", "punctuation-ab", "build-punctuation-agent", "evaluate-punctuation-agent", "build-residual-conflicts", "build-summary-review", "apply-summary-review"])
    benchmark.add_argument("input_path", help="Build: output directory; run/report: manifest or run JSON")
    benchmark.add_argument("--bundle-dirs", default="", help="Comma-separated bundle directories for build")
    benchmark.add_argument("--output-dir", default="")
    benchmark.add_argument("--scores-json", default="", help="Downloaded anonymous scores JSON for apply-summary-review")
    benchmark.add_argument("--private-json", default="", help="Private source mapping for evaluate-arbitration; human reference remains in the benchmark manifest only")
    benchmark.add_argument("--manifest-json", default="", help="Benchmark manifest used only for sample-to-bundle lookup in residual conflict filtering")
    benchmark.add_argument("--entity-lexicon-json", default="", help="Explicit entity lexicon for residual conflict filtering; no bundle copy required")
    benchmark.add_argument("--decisions-json", default="", help="Optional evidence-backed anonymous arbitration decisions")
    benchmark.add_argument("--primary-variant", default="sensevoice_full_punc")
    benchmark.add_argument("--secondary-variant", default="qwen3_asr_1_7b")
    benchmark.add_argument("--min-confidence", type=float, default=0.75)
    benchmark.add_argument("--punctuation-model", default="ct-punc")
    benchmark.add_argument("--punctuation-device", choices=["auto", "cuda", "cpu"], default="auto")
    benchmark.add_argument("--media-path", action="append", default=[], help="Repeat once per bundle to provide the current local video path")
    benchmark.add_argument("--samples-per-bundle", type=int, default=8)
    benchmark.add_argument("--sample-seconds", type=float, default=60.0)
    benchmark.add_argument("--asr-language", default="zh", help="Source language for local ASR variants, for example yue for Cantonese")
    benchmark.add_argument(
        "--asr-context-hotwords",
        default="",
        help="Comma-separated evidence-derived vocabulary passed only as ASR context; never expected transcript text",
    )
    benchmark.add_argument("--execute-clips", action="store_true", help="Generate local WAV clips for human annotation")
    benchmark.add_argument("--legacy-reference-manifest", default="", help="Carry completed legacy fixed-window references into aligned samples for boundary-extension review")
    benchmark.add_argument("--variants", default="sensevoice_raw,sensevoice_full_punc,qwen3_asr_1_7b", help="Comma-separated local ASR variants for execute-variants")
    benchmark.add_argument("--execute", action="store_true", help="Actually run local ASR variants; preview is default")
    benchmark_resume = benchmark.add_mutually_exclusive_group()
    benchmark_resume.add_argument("--resume", action="store_true", dest="resume", default=True, help="Resume is the default; skip existing valid variant transcripts")
    benchmark_resume.add_argument("--no-resume", action="store_false", dest="resume", help="Replan variants even when a valid transcript already exists")
    benchmark.add_argument("--retry-failed", action="store_true", help="Retry variants previously recorded as failed")
    benchmark.add_argument("--limit", type=int, default=0, help="Maximum new variant attempts; 0 means all")
    benchmark.add_argument("--timeout-seconds", type=int, default=1800)
    benchmark.add_argument("--no-write", action="store_true")

    resegment = sub.add_parser("resegment-transcript", help="Split a coarse transcript into estimated timed cues")
    resegment.add_argument("workspace_dir")
    resegment.add_argument("input_path")
    resegment.add_argument("--media-path", default="")
    resegment.add_argument("--duration-seconds", type=float, default=0.0)
    resegment.add_argument("--target-seconds", type=float, default=8.0)
    resegment.add_argument("--max-chars", type=int, default=180)
    resegment.add_argument("--title", default="")

    postprocess = sub.add_parser("postprocess-asr-transcript", help="Preserve ASR segments while adding local punctuation; merging is explicit")
    postprocess.add_argument("bundle_dir")
    postprocess.add_argument("--input-path", default="")
    postprocess.add_argument("--target-seconds", type=float, default=18.0)
    postprocess.add_argument("--max-chars", type=int, default=180)
    postprocess.add_argument("--punctuation-mode", choices=["readable", "conservative", "preserve"], default="readable", help="readable inserts local cue-boundary commas; conservative only adds terminal punctuation; preserve keeps existing punctuation")
    postprocess.add_argument("--segment-policy", choices=["preserve", "readable_merge"], default="preserve", help="preserve keeps IDs/order/timestamps/boundaries; readable_merge explicitly enables split/merge with lineage records")
    postprocess.add_argument("--no-set-corrected", action="store_true", help="Do not promote the postprocessed transcript to manifest.corrected_transcript_*")
    postprocess.add_argument("--no-write", action="store_true")

    translate_mandarin = sub.add_parser(
        "translate-transcript-to-mandarin",
        help="Translate a Cantonese transcript into a separate Mandarin subtitle sidecar using the local text route",
    )
    translate_mandarin.add_argument("bundle_dir")
    translate_mandarin.add_argument("--source-path", default="")
    translate_mandarin.add_argument("--input-json", default="", help="Import reviewed translations instead of running a model")
    translate_mandarin.add_argument("--batch-size", type=int, default=24)
    translate_mandarin.add_argument("--route-id", default="")
    translate_mandarin.add_argument("--route-revision", default="")
    translate_mandarin.add_argument(
        "--direct-lmstudio",
        action="store_true",
        help="Explicitly use the fixed local LM Studio Qwen3.5 route at 127.0.0.1:1234; never falls back online",
    )
    translate_mandarin.add_argument("--execute", action="store_true", help="Execute only the configured local text route")
    translate_mandarin.add_argument("--no-write", action="store_true")

    readable_llm = sub.add_parser("readable-transcript-llm-polish", help="Preview/import/execute LLM punctuation and segmentation polish for the readable transcript")
    readable_llm.add_argument("bundle_dir")
    readable_llm.add_argument("--provider-config", default="", help="Runtime text LLM provider config JSON or path; secrets are not persisted")
    readable_llm.add_argument("--input-json", default="", help="Import reviewed/model JSON instead of calling a provider")
    readable_llm.add_argument("--execute", action="store_true", help="Call the configured text LLM provider")
    readable_llm.add_argument("--agent-substitute", action="store_true", help="Run the local agent-substitute punctuation/segmentation pass without a network call")
    readable_llm.add_argument("--agent-name", default="local_agent", help="Agent runtime label for local substitute mode, e.g. codex/workbuddy/opencode/hermes_agent/openclaw")
    readable_llm.add_argument("--codex-substitute", action="store_true", help="Legacy alias for --agent-substitute --agent-name codex")
    readable_llm.add_argument("--promote", action="store_true", help="Promote llm-readable-transcript.* to corrected-transcript.*")
    readable_llm.add_argument("--max-segments-per-batch", type=int, default=40)
    readable_llm.add_argument("--max-prompt-chars", type=int, default=9000)
    readable_llm.add_argument("--max-tokens", type=int, default=4000)
    readable_llm.add_argument("--temperature", type=float, default=0.0)
    readable_llm.add_argument("--no-write", action="store_true")

    agent_readable = sub.add_parser("agent-readable-transcript-rewrite", help="Run/import local agent transcript readability rewrite without cloud calls")
    agent_readable.add_argument("bundle_dir")
    agent_readable.add_argument("--input-json", default="", help="Import reviewed agent JSON with segments[index,text]")
    agent_readable.add_argument("--agent-name", default="local_agent", help="Agent runtime label, e.g. codex/workbuddy/opencode/hermes_agent/openclaw")
    agent_readable.add_argument("--source-path", default="", help="Optional source transcript sidecar path")
    agent_readable.add_argument("--promote", action="store_true", help="Promote agent-readable-transcript.* to corrected-transcript.*")
    agent_readable.add_argument("--no-write", action="store_true")

    quality_gate = sub.add_parser("transcript-quality-gate", help="Check corrected transcript for punctuation artifacts, residual ASR mistakes, and rewrite blockers")
    quality_gate.add_argument("bundle_dir")
    quality_gate.add_argument("--input-path", default="", help="Optional transcript sidecar path; defaults to corrected/agent-readable transcript")
    quality_gate.add_argument("--reference-path", default="", help="Optional human reference transcript used for evaluation only")
    quality_gate.add_argument("--baseline-path", default="", help="Optional baseline transcript for relative CER and overcorrection")
    quality_gate.add_argument("--min-punctuation-per-1000", type=float, default=50.0)
    quality_gate.add_argument("--max-punctuation-per-1000", type=float, default=140.0)
    quality_gate.add_argument("--max-cer", type=float, default=0.18)
    quality_gate.add_argument("--min-entity-accuracy", type=float, default=0.98)
    quality_gate.add_argument("--max-overcorrection-rate", type=float, default=0.01)
    quality_gate.add_argument(
        "--require-speaker-diarization",
        action="store_true",
        help="Fail when spoken segments lack speaker clusters or fewer than --min-speaker-count are present",
    )
    quality_gate.add_argument("--min-speaker-count", type=int, default=2)
    quality_gate.add_argument("--no-write", action="store_true")

    ready = sub.add_parser("run-ready-pipeline", help="Run the ready video pipeline after source/extractor artifacts exist")
    ready.add_argument("workspace_dir")
    ready.add_argument("--extractor", default="")

    extractor = sub.add_parser("run-extractor-plan", help="Preview or execute one planned visual extractor command")
    extractor.add_argument("plan_json")
    extractor.add_argument("extractor", choices=["vidclaude", "peepshow", "vidwise"])
    extractor.add_argument("--execute", action="store_true")
    extractor.add_argument("--timeout-seconds", type=int, default=0)

    extractor_log = sub.add_parser("extractor-run-log", help="Render persisted visual extractor command runs")
    extractor_log.add_argument("workspace_dir")

    attach_peepshow = sub.add_parser("attach-peepshow-output", help="Attach Peepshow output to a WebUI bundle as source evidence")
    attach_peepshow.add_argument("bundle_dir")
    attach_peepshow.add_argument("output_dir")
    attach_peepshow.add_argument("--no-write", action="store_true")

    refresh = sub.add_parser("refresh-lecture-review", help="Import review-notes.json and refresh WebUI/optional Obsidian outputs")
    refresh.add_argument("project")
    refresh.add_argument("review_json")
    refresh.add_argument("--webui-output-dir")
    refresh.add_argument("--vault", default="")
    refresh.add_argument("--folder", default="00_Inbox/AI/课程视频知识包")
    refresh.add_argument("--target", default="bilinote")
    refresh.add_argument("--allow-blocked-export", action="store_true")

    apply_review = sub.add_parser("apply-review-notes", help="Apply review-notes.json to timeline, coverage, and readiness")
    apply_review.add_argument("bundle_dir")
    apply_review.add_argument("--review-json")
    apply_review.add_argument("--no-write", action="store_true")

    validate_review = sub.add_parser("validate-review-notes", help="Validate review-notes.json without applying it")
    validate_review.add_argument("bundle_dir")
    validate_review.add_argument("--review-json")

    prepare_review = sub.add_parser("prepare-review-session", help="Prepare a human-readable review handoff for open bundle gaps")
    prepare_review.add_argument("bundle_dir")
    prepare_review.add_argument("--run-id", default=None, help="Read this exact versioned offline semantic run")
    prepare_review.add_argument("--dependency-versions", default="", help="JSON file containing the current dependency versions")
    prepare_review.add_argument("--no-refresh", action="store_true")
    prepare_review.add_argument("--limit", type=int, default=30)
    prepare_review.add_argument("--offset", type=int, default=0)
    prepare_review.add_argument("--reason", default="", help="Comma-separated target reasons to include, e.g. semantic_frame_without_analysis")
    prepare_review.add_argument("--group-by", default="reason", choices=["reason", "suggested_status", "route"])
    prepare_review.add_argument("--include-closed", action="store_true")
    prepare_review.add_argument("--output-prefix", default="review-pack")

    review_closure = sub.add_parser("review-closure-status", help="Summarize open/closed human review targets for a bundle")
    review_closure.add_argument("bundle_dir")
    review_closure.add_argument("--no-write", action="store_true")

    router = sub.add_parser("run-video-frame-router", help="Route frames into document, semantic, temporal, mixed, or unknown visual branches")
    router.add_argument("bundle_dir")
    router.add_argument("--input-json")
    router.add_argument(
        "--content-profile",
        choices=["auto", "general", "lecture-slides-v1"],
        default="auto",
        help="lecture-slides-v1 keeps static slides OCR-first and only escalates explicit non-text semantics",
    )
    router.add_argument("--no-write", action="store_true")

    tagger_import = sub.add_parser("import-tagger-annotations", help="Import Qinglong/manual tagger timeline annotations into the unified timeline")
    tagger_import.add_argument("bundle_dir")
    tagger_import.add_argument("tagger_json")
    tagger_import.add_argument("--source", default="qinglong")
    tagger_import.add_argument("--no-write", action="store_true")

    general_tagger = sub.add_parser("run-general-tagger", help="Plan or run local RAM++ tagging over bundle keyframes and import candidate evidence")
    general_tagger.add_argument("bundle_dir")
    general_tagger.add_argument("--source-root", default="")
    general_tagger.add_argument("--checkpoint-path", default="")
    general_tagger.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda", help="CUDA is required for real local-model execution; auto never falls back to CPU")
    general_tagger.add_argument("--prefer-language", choices=["zh", "en"], default="zh")
    general_tagger.add_argument("--limit", type=int, default=0)
    general_tagger.add_argument(
        "--frame-mode",
        choices=["representative", "continuous"],
        default="representative",
        help="Tag one representative frame or all ordered frames; continuous mode remains candidate-only",
    )
    general_tagger.add_argument("--execute", action="store_true")
    general_tagger.add_argument("--no-import", action="store_true")
    general_tagger.add_argument("--no-write", action="store_true")

    tag_delta = sub.add_parser(
        "run-temporal-tag-delta",
        help="Smooth continuous-frame tag changes and escalate dynamic or ambiguous segments",
    )
    tag_delta.add_argument("bundle_dir")
    tag_delta.add_argument("--input-json", help="Synthetic/manual per-frame tag observations")
    tag_delta.add_argument(
        "--execute-tagger",
        action="store_true",
        help="Run the existing local RAM++ backend over ordered frames",
    )
    tag_delta.add_argument("--source-root", default="")
    tag_delta.add_argument("--checkpoint-path", default="")
    tag_delta.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    tag_delta.add_argument("--prefer-language", choices=["zh", "en"], default="zh")
    tag_delta.add_argument("--indexes", default="", help="Comma-separated timeline indexes")
    tag_delta.add_argument("--frame-count", type=int, choices=range(5, 13), default=8)
    tag_delta.add_argument("--limit", type=int, choices=range(1, 17), default=16, help="Maximum timeline sequences in this bounded batch")
    tag_delta.add_argument("--min-frames", type=int, choices=range(5, 13), default=5)
    tag_delta.add_argument("--resume", default="", help="Reuse frame checkpoints from this run id")
    tag_delta.add_argument("--no-write", action="store_true")

    tag_restore = sub.add_parser("temporal-tag-restore", help="Dry-run or apply one hash-bound temporal tag restore plan")
    tag_restore.add_argument("bundle_dir")
    tag_restore.add_argument("--plan-json", required=True)
    tag_restore.add_argument("--execute", action="store_true")
    tag_restore.add_argument("--confirm-run-id", default="")

    ocr = sub.add_parser("run-ocr-backfill", help="Import or execute OCR backfill for bundle frames")
    ocr.add_argument("bundle_dir")
    ocr.add_argument("--input-json")
    ocr.add_argument("--execute", action="store_true")
    ocr.add_argument("--language", default="chi_sim+eng")
    ocr.add_argument("--captiocr-root")
    ocr.add_argument("--limit", type=int, default=0)
    ocr.add_argument("--apply-mode", choices=["merge", "replace_snapshot"], default="merge")
    ocr.add_argument("--engine", choices=["auto", "rapidocr"], default="auto",
                     help="Use the existing offline ebook RapidOCR owner when explicitly selected")

    screen_text = sub.add_parser("run-screen-text-recovery", help="Plan or execute targeted crop-based screen text recovery")
    screen_text.add_argument("bundle_dir")
    screen_text.add_argument("--execute-crops", action="store_true")
    screen_text.add_argument("--execute-ocr", action="store_true")
    screen_text.add_argument("--input-json")
    screen_text.add_argument("--language", default="chi_sim+eng")
    screen_text.add_argument("--captiocr-root")
    screen_text.add_argument("--engine", choices=["auto", "rapidocr"], default="auto")
    screen_text.add_argument("--limit", type=int, default=0)
    screen_text.add_argument("--indexes", help="Comma-separated timeline indexes to recover")
    screen_text.add_argument("--profile", choices=["default", "talking_head"], default="default")
    screen_text.add_argument("--max-regions", type=int, default=12)
    screen_text.add_argument("--max-attempts", type=int, default=2)
    screen_text.add_argument("--no-write", action="store_true")

    tile_plan = sub.add_parser("high-res-tile-plan", help="Plan or execute local high-resolution tiles for detail-heavy frames")
    tile_plan.add_argument("bundle_dir")
    tile_plan.add_argument("--execute-tiles", action="store_true")
    tile_plan.add_argument("--indexes", help="Comma-separated timeline indexes to tile")
    tile_plan.add_argument("--limit", type=int, default=0)
    tile_plan.add_argument("--tile-size", type=int, default=768)
    tile_plan.add_argument("--overlap", type=float, default=0.12)
    tile_plan.add_argument("--max-tiles-per-image", type=int, default=12)
    tile_plan.add_argument("--include-routes", default="")
    tile_plan.add_argument("--no-write", action="store_true")

    tile_import = sub.add_parser("tile-result-import-build", help="Build tile-result-import.json from high-res tile plan and existing OCR/VLM/human result files")
    tile_import.add_argument("bundle_dir")
    tile_import.add_argument("--results-dir")
    tile_import.add_argument("--output-json")
    tile_import.add_argument("--default-source", default="tile_result_import_builder")
    tile_import.add_argument("--default-confidence", type=float, default=0.0)
    tile_import.add_argument("--no-write", action="store_true")

    tile_merge = sub.add_parser("tile-result-merge", help="Merge high-res tile OCR/VLM/human results into visual evidence or review targets")
    tile_merge.add_argument("bundle_dir")
    tile_merge.add_argument("--input-json")
    tile_merge.add_argument("--execute", action="store_true")
    tile_merge.add_argument("--min-confidence", type=float, default=0.65)
    tile_merge.add_argument("--no-write", action="store_true")

    recapture = sub.add_parser("run-frame-recapture-plan", help="Preview or execute local ffmpeg frame recapture items from manifest.frame_recapture")
    recapture.add_argument("bundle_dir")
    recapture.add_argument("--execute", action="store_true")
    recapture.add_argument("--timeout-seconds", type=int, default=30)

    structure = sub.add_parser("run-visual-structure", help="Parse document-like screenshots through the document visual branch")
    structure.add_argument("bundle_dir")
    structure.add_argument("--input-json")
    structure.add_argument("--execute-ebook-pipeline", action="store_true")
    structure.add_argument("--include-routes", default="document_visual,mixed,semantic_frame,temporal_sequence")
    structure.add_argument("--timeout-seconds", type=int, default=120)
    structure.add_argument("--indexes", help="Comma-separated timeline indexes to process")
    structure.add_argument("--limit", type=int, default=0)

    ebook_batch = sub.add_parser("run-visual-structure-ebook-batches", help="Run ebook OCR in resumable short-lived child batches")
    ebook_batch.add_argument("bundle_dir")
    ebook_batch.add_argument("--execute", action="store_true")
    ebook_batch.add_argument("--include-routes", default="document_visual,mixed,semantic_frame,temporal_sequence")
    ebook_batch.add_argument("--indexes", help="Comma-separated timeline indexes to process")
    ebook_batch.add_argument("--batch-size", type=int, default=3)
    ebook_batch.add_argument("--timeout-seconds", type=int, default=120)
    ebook_batch.add_argument("--no-resume", action="store_true")
    ebook_batch.add_argument("--no-write", action="store_true")

    ebook_repair = sub.add_parser("repair-ebook-artifact-text", help="Repair ebook OCR text from existing UTF-8 artifacts without rerunning OCR")
    ebook_repair.add_argument("bundle_dir")
    ebook_repair.add_argument("--no-write", action="store_true")

    multimodal = sub.add_parser("run-multimodal-frame-analysis", help="Preview, import, or execute single-frame multimodal understanding")
    multimodal.add_argument("bundle_dir")
    multimodal.add_argument("--input-json")
    multimodal.add_argument("--execute", action="store_true")
    multimodal.add_argument("--provider-config", help="Inline JSON or JSON file: provider/openai_compatible/gemini/local_qwen_vl, api_key, model, base_url")
    multimodal.add_argument("--limit", type=int)
    multimodal.add_argument("--indexes")
    multimodal.add_argument(
        "--index",
        action="append",
        type=int,
        default=[],
        help="Repeatable timeline index; PowerShell-safe alternative to --indexes CSV",
    )
    multimodal.add_argument("--confirm-vision-calls", type=int)
    multimodal.add_argument("--confirm-vision-indexes", default="")
    multimodal.add_argument("--image-probe-max-edge", type=int, default=0, help="Resize images sent to provider; 0 sends originals")
    multimodal.add_argument("--image-probe-jpeg-quality", type=int, default=70)
    multimodal.add_argument("--vision-retries", type=int, default=1)
    multimodal.add_argument("--vision-retry-delay-seconds", type=float, default=0.0)
    multimodal.add_argument("--execution-actor", choices=["operator", "agent"], default="operator")
    multimodal.add_argument("--export-consent", default="", help="Scoped consent JSON required for agent execution")
    multimodal.add_argument("--max-tokens", type=int, default=0, help="Optional explicit output-token ceiling; truncation remains incomplete")

    summary_review_prepare = sub.add_parser("prepare-summary-claim-review", help="Prepare one immutable offline claim annotation/review task; no model calls")
    summary_review_prepare.add_argument("bundle_dir")
    summary_review_prepare.add_argument("--no-write", action="store_true")
    summary_review_import = sub.add_parser("import-summary-claim-review", help="Validate and atomically import an existing prepared claim-review result")
    summary_review_import.add_argument("bundle_dir")
    summary_review_import.add_argument("--input-json", required=True)
    summary_review_status = sub.add_parser("summary-claim-review-status", help="Read current evidence-review identity, next action and gaps without writes")
    summary_review_status.add_argument("bundle_dir")

    codex_auto = sub.add_parser("run-codex-vision", help="Run/resume bounded Codex CLI semantic vision; preview unless --execute")
    codex_auto.add_argument("bundle_dir")
    codex_auto.add_argument("--profile-json", help="Versioned model/effort/batch/runtime profile JSON")
    codex_auto.add_argument("--state-json", help="Existing queue state JSON; preserves the pinned model and completed batches")
    codex_auto.add_argument("--indexes", default="")
    codex_auto.add_argument("--index", action="append", type=int, default=[])
    codex_auto.add_argument("--execute", action="store_true")
    codex_auto.add_argument("--max-batches", type=int, help="Maximum new inference calls in this invocation")
    codex_auto.add_argument("--retry-failed", action="store_true", help="Explicit retry of failed/uncertain batch within profile attempt limit")
    codex_catalog = sub.add_parser("codex-vision-catalog", help="Read bundled CLI catalog; no inference, refresh or automatic model changes")
    codex_catalog.add_argument("--profile-json")

    codex_session = sub.add_parser(
        "prepare-codex-vision-session",
        help="Prepare one immutable, bundle-local interactive Codex vision task pack",
    )
    codex_session.add_argument("bundle_dir")
    codex_session.add_argument("--kind", choices=["semantic", "temporal"], required=True)
    codex_session.add_argument("--indexes", default="", help="Comma-separated timeline indexes")
    codex_session.add_argument(
        "--index",
        action="append",
        type=int,
        default=[],
        help="Repeatable timeline index; PowerShell-safe alternative to --indexes CSV",
    )
    codex_session.add_argument("--limit", type=int)
    codex_session.add_argument("--max-frames", type=int, default=19)
    codex_session.add_argument("--model-requested", default="luna")
    codex_session.add_argument("--quality", default="low")
    codex_session.add_argument("--executor", choices=["codex_session", "codex_native"], default="codex_session")
    codex_session.add_argument("--no-write", action="store_true")

    codex_import = sub.add_parser(
        "import-codex-vision-results",
        help="Validate and atomically import one interactive Codex vision result batch",
    )
    codex_import.add_argument("bundle_dir")
    codex_import.add_argument("--input-json", required=True)
    codex_import.add_argument("--no-write", action="store_true")

    groups = sub.add_parser("run-temporal-frame-groups", help="Generate 5-12 ordered frames for temporal visual candidates")
    groups.add_argument("bundle_dir")
    groups.add_argument("--execute", action="store_true")
    groups.add_argument("--frame-count", type=int)
    groups.add_argument("--window-seconds", type=float, default=4.0)
    groups.add_argument("--include-routes", default="temporal_sequence,mixed")
    groups.add_argument("--indexes", help="Comma-separated timeline indexes to generate frame groups for")
    groups.add_argument("--limit", type=int)
    groups.add_argument("--timeout-seconds", type=int, default=60)

    temporal = sub.add_parser("run-temporal-visual-analysis", help="Preview, import, or execute multi-frame temporal understanding")
    temporal.add_argument("bundle_dir")
    temporal.add_argument("--input-json")
    temporal.add_argument("--execute", action="store_true")
    temporal.add_argument("--frame-count", type=int)
    temporal.add_argument("--limit", type=int)
    temporal.add_argument("--indexes")
    temporal.add_argument(
        "--index",
        action="append",
        type=int,
        default=[],
        help="Repeatable timeline index; PowerShell-safe alternative to --indexes CSV",
    )
    temporal.add_argument("--provider-config", help="Inline JSON or JSON file with provider/base_url/model/api_key")
    temporal.add_argument("--confirm-vision-calls", type=int)
    temporal.add_argument("--confirm-vision-indexes", default="")
    temporal.add_argument("--image-probe-max-edge", type=int, default=0, help="Resize images sent to provider; 0 sends originals")
    temporal.add_argument("--image-probe-jpeg-quality", type=int, default=70)
    temporal.add_argument("--vision-retries", type=int, default=1)
    temporal.add_argument("--vision-retry-delay-seconds", type=float, default=0.0)
    temporal.add_argument("--execution-actor", choices=["operator", "agent"], default="operator")
    temporal.add_argument("--export-consent", default="", help="Scoped consent JSON required for agent execution")
    temporal.add_argument("--max-tokens", type=int, default=0, help="Optional explicit output-token ceiling; truncation remains incomplete")

    vision_test = sub.add_parser("test-vision-provider", help="Run a non-persistent provider connectivity and JSON-output test")
    vision_test.add_argument("--provider-config", help="Inline JSON or JSON file with provider/base_url/model/api_key; local_qwen_vl/local_vlm may omit api_key")
    vision_test.add_argument("--image-paths", default="", help="Comma-separated image paths for single/multi-image checks")

    text_llm_smoke = sub.add_parser("text-llm-provider-smoke", help="Plan or execute a no-secret text LLM provider smoke check")
    text_llm_smoke.add_argument("--provider-config", help="Inline JSON or JSON file with provider/base_url/model/api_key")
    text_llm_smoke.add_argument("--execute", action="store_true", help="Actually call the provider; default only prints a safe request plan")
    text_llm_smoke.add_argument("--prompt", default="Reply with exactly: ok")

    online_api = sub.add_parser("online-model-api", help="Preview or execute one unified online model API call")
    online_api.add_argument("model_type", choices=["asr", "ocr", "document_visual", "semantic_frame", "temporal_sequence", "video_segment", "text_llm", "summary_rewrite", "transcript_correction"])
    online_api.add_argument("--provider-config", help="Inline JSON or JSON file with provider/base_url/model/api_key")
    online_api.add_argument("--prompt", default="")
    online_api.add_argument("--input-text", default="")
    online_api.add_argument("--image-paths", default="", help="Comma-separated image paths for OCR/vision/video frame calls")
    online_api.add_argument("--audio-path", default="", help="Audio path for online ASR calls")
    online_api.add_argument("--execute", action="store_true", help="Actually call the provider; default only prints a safe request plan")
    online_api.add_argument("--output-dir", default="")
    online_api.add_argument("--no-write", action="store_true")

    online_matrix = sub.add_parser("online-model-api-matrix", help="List all online model API interfaces without network calls")
    online_matrix.add_argument("--provider-config", help="Inline JSON or JSON file with provider/base_url/model/api_key")
    online_matrix.add_argument("--output-dir", default="")
    online_matrix.add_argument("--no-write", action="store_true")

    model_coverage = sub.add_parser("model-task-coverage-audit", help="Audit task to model type, gateway, and provider coverage")
    model_coverage.add_argument("--output-dir", default="")
    model_coverage.add_argument("--no-write", action="store_true")

    term_model = sub.add_parser("run-term-arbitration-model", help="Preview or execute evidence-grounded terminology arbitration through the unified model task gateway")
    term_model.add_argument("bundle_dir")
    term_model.add_argument("--provider-config", help="Runtime-only provider JSON or JSON file")
    term_model.add_argument("--execute", action="store_true")
    term_model.add_argument("--max-terms", type=int, default=60)
    term_model.add_argument("--min-confidence", type=float, default=0.88)
    term_model.add_argument("--max-tokens", type=int, default=5000)
    term_model.add_argument("--temperature", type=float, default=0)
    term_model.add_argument("--no-write", action="store_true")

    mind_map_model = sub.add_parser("run-bilinote-mind-map-model", help="Preview or execute BiliNote-style mind-map chunks through the unified model task gateway")
    mind_map_model.add_argument("bundle_dir")
    mind_map_model.add_argument("--provider-config", help="Runtime-only provider JSON or JSON file")
    mind_map_model.add_argument("--execute", action="store_true")
    mind_map_model.add_argument("--title", default="")
    mind_map_model.add_argument("--max-chars", type=int, default=5000)
    mind_map_model.add_argument("--limit", type=int, default=0)
    mind_map_model.add_argument("--max-tokens", type=int, default=4000)
    mind_map_model.add_argument("--temperature", type=float, default=0)
    mind_map_model.add_argument("--no-write", action="store_true")

    volcengine_matrix = sub.add_parser("volcengine-model-task-matrix", help="Plan or execute Volcengine Coding Plan model tests mapped to VKP task types")
    volcengine_matrix.add_argument("--execute", action="store_true")
    volcengine_matrix.add_argument("--output-dir", default="")
    volcengine_matrix.add_argument("--models", default="", help="Comma-separated model filter")
    volcengine_matrix.add_argument("--tasks", default="", help="Comma-separated task_key filter")
    volcengine_matrix.add_argument("--timeout-seconds", type=int, default=120)
    volcengine_matrix.add_argument("--no-write", action="store_true")

    volcengine_routing = sub.add_parser("volcengine-model-routing", help="Show production Volcengine Coding Plan model routing without network calls")
    volcengine_routing.add_argument("--route", default="tool_terms", help="Route key, default tool_terms")
    volcengine_routing.add_argument("--output-dir", default="")
    volcengine_routing.add_argument("--no-write", action="store_true")

    mind_map_prompt = sub.add_parser("bilinote-mind-map-prompt-pack", help="Build BiliNote-style mind-map prompt messages from transcript text")
    mind_map_prompt.add_argument("--title", default="")
    mind_map_prompt.add_argument("--transcript", default="")
    mind_map_prompt.add_argument("--transcript-path", default="")
    mind_map_prompt.add_argument("--bundle-dir", default="", help="Optional bundle dir; when set, read the best transcript sidecar and write prompt pack artifacts")
    mind_map_prompt.add_argument("--max-chars", type=int, default=5000)
    mind_map_prompt.add_argument("--no-write", action="store_true")

    prepare_transcript_editor = sub.add_parser("prepare-transcript-edit-session", help="Write a static BiliNote-style transcript editor for one bundle")
    prepare_transcript_editor.add_argument("bundle_dir")
    prepare_transcript_editor.add_argument("--no-write", action="store_true")

    apply_transcript_editor = sub.add_parser("apply-transcript-edits", help="Import reviewed transcript-edits.json and write human-corrected transcript sidecars")
    apply_transcript_editor.add_argument("bundle_dir")
    apply_transcript_editor.add_argument("--edits-json", required=True)
    apply_transcript_editor.add_argument("--no-write", action="store_true")

    prepare_subtitle_editor_parser = sub.add_parser(
        "prepare-subtitle-editor",
        help="Write the standalone dual-track VKP subtitle editor for one bundle",
    )
    prepare_subtitle_editor_parser.add_argument("bundle_dir")
    prepare_subtitle_editor_parser.add_argument("--no-write", action="store_true")

    validate_subtitle_review_parser = sub.add_parser(
        "validate-subtitle-review",
        help="Validate subtitle_review_notes.v1 without changing the bundle",
    )
    validate_subtitle_review_parser.add_argument("bundle_dir")
    validate_subtitle_review_parser.add_argument("--review-json", required=True)

    apply_subtitle_review_parser = sub.add_parser(
        "apply-subtitle-review",
        help="Apply a human-confirmed dual-track subtitle review as derived sidecars",
    )
    apply_subtitle_review_parser.add_argument("bundle_dir")
    apply_subtitle_review_parser.add_argument("--review-json", required=True)
    apply_subtitle_review_parser.add_argument("--no-write", action="store_true")

    transcript_correction = sub.add_parser("transcript-correction-pack", help="Build/import/execute a BiliNote-style transcript correction pack")
    transcript_correction.add_argument("bundle_dir")
    transcript_correction.add_argument("--input-json", default="", help="Import a JSON correction response with segments[index,text]")
    transcript_correction.add_argument("--provider-config", help="Inline JSON or JSON file with provider/base_url/model/api_key")
    transcript_correction.add_argument("--execute", action="store_true", help="Actually call the configured text LLM provider; default only writes prompt pack")
    transcript_correction.add_argument("--max-segments", type=int, default=0, help="Limit segments for a small correction batch; 0 means all")
    transcript_correction.add_argument("--max-chunk-chars", type=int, default=5000)
    transcript_correction.add_argument("--no-write", action="store_true")

    conflict_index = sub.add_parser("evidence-conflict-index", help="Build the strict ASR/subtitle/OCR/tagger/vision/web conflict index for LLM arbitration")
    conflict_index.add_argument("bundle_dir")
    conflict_index.add_argument("--input-json", default="", help="Optional transcript-semantic-correction-pack.json path")
    conflict_index.add_argument("--limit", type=int, default=0)
    conflict_index.add_argument("--no-write", action="store_true")

    transcript_arbitration = sub.add_parser("transcript-source-arbitration", help="Arbitrate ASR/platform subtitles/corrected transcripts into a safer corrected transcript sidecar")
    transcript_arbitration.add_argument("bundle_dir")
    transcript_arbitration.add_argument("--platform-subtitle", default="", help="Optional platform subtitle JSON/SRT/VTT path")
    transcript_arbitration.add_argument("--subtitle", default="", help="Optional additional subtitle JSON/SRT/VTT path")
    transcript_arbitration.add_argument("--asr-json", default="", help="Optional ASR transcript JSON/SRT path to force as a source")
    transcript_arbitration.add_argument("--glossary-json", default="", help="Optional glossary aliases used as high-confidence term evidence")
    transcript_arbitration.add_argument("--min-confidence", type=float, default=0.72)
    transcript_arbitration.add_argument("--no-promote", action="store_true", help="Write source-arbitrated transcript but do not point manifest.corrected_transcript_* at it")
    transcript_arbitration.add_argument("--no-write", action="store_true")


    evidence_pipeline = sub.add_parser("transcript-evidence-correction-pipeline", help="Run the preferred ASR/subtitle/tagger/OCR/vision + LLM semantic correction chain")
    evidence_pipeline.add_argument("bundle_dir")
    evidence_pipeline.add_argument("--platform-subtitle", default="", help="Optional platform subtitle JSON/SRT/VTT path")
    evidence_pipeline.add_argument("--subtitle", default="", help="Optional additional subtitle JSON/SRT/VTT path")
    evidence_pipeline.add_argument("--asr-json", default="", help="Optional ASR transcript JSON/SRT path to force as a source")
    evidence_pipeline.add_argument("--secondary-asr-json", default="", help="Optional independent second ASR transcript used only as conflict evidence")
    evidence_pipeline.add_argument("--additional-secondary-asr-json", action="append", default=[], help="Repeatable independent ASR evidence path; registered without replacing the primary or first secondary transcript")
    evidence_pipeline.add_argument("--consensus-agreement-threshold", type=float, default=0.86)
    evidence_pipeline.add_argument("--execute-consensus-clips", action="store_true", help="Generate local 8-30 second audio review clips for dual-ASR conflicts")
    evidence_pipeline.add_argument("--asr-mode", choices=["daily", "important"], default="")
    evidence_pipeline.add_argument("--execute-local-asr", action="store_true", help="Advance local primary and independent secondary ASR; resume completed stages")
    evidence_pipeline.add_argument("--speech-activity-json", default="")
    evidence_pipeline.add_argument("--diarization-json", default="")
    evidence_pipeline.add_argument("--media-path", default="", help="Optional existing local media path; bundle manifest is used for targeted local ASR when omitted")
    evidence_pipeline.add_argument("--execute-local-targeted-asr", action="store_true", help="Extract semantic risk clips, run independent local ASR, register candidate-only evidence, and rebuild the semantic pack")
    evidence_pipeline.add_argument("--local-targeted-asr-preset", default="qwen3-asr-0.6b")
    evidence_pipeline.add_argument("--local-targeted-asr-model", default="")
    evidence_pipeline.add_argument("--local-targeted-asr-timeout-seconds", type=int, default=900)
    evidence_pipeline.add_argument("--local-targeted-asr-allow-cpu", action="store_true")
    evidence_pipeline.add_argument("--glossary-json", default="", help="Optional glossary aliases used as term evidence")
    evidence_pipeline.add_argument("--provider-config", default="", help="Inline JSON, JSON file, or provider profile; runtime-only and never persisted")
    evidence_pipeline.add_argument("--quality-profile", default="quality", help="Processing profile; quality enables provider auto-execution only when data_export_allowed=true")
    evidence_pipeline.add_argument("--execute-llm", action="store_true", help="Actually call the configured text LLM provider for semantic arbitration")
    evidence_pipeline.add_argument("--agent-name", default="local_agent", help="Agent runtime label for local substitute mode, e.g. codex/workbuddy/opencode/hermes_agent/openclaw")
    evidence_pipeline.add_argument("--no-agent-substitute", action="store_true", help="Disable the default local agent-substitute execution path and return to preview/cloud-gated behavior")
    evidence_pipeline.add_argument("--no-codex-substitute", action="store_true", help="Legacy alias for --no-agent-substitute")
    evidence_pipeline.add_argument("--no-readable-llm", action="store_true", help="Skip readable transcript LLM request planning/execution")
    evidence_pipeline.add_argument("--execute-readable-llm", action="store_true", help="Actually call the configured text LLM provider for punctuation/segmentation polish")
    evidence_pipeline.add_argument("--promote-readable-llm", action="store_true", help="Promote llm-readable-transcript.* to corrected-transcript.* before semantic arbitration")
    evidence_pipeline.add_argument("--readable-max-segments-per-batch", type=int, default=40)
    evidence_pipeline.add_argument("--readable-max-prompt-chars", type=int, default=9000)
    evidence_pipeline.add_argument("--readable-max-tokens", type=int, default=4000)
    evidence_pipeline.add_argument("--auto-apply-high-confidence", action="store_true", help="Apply locally validated high-confidence LLM decisions and refresh readable exports")
    evidence_pipeline.add_argument("--no-postprocess", action="store_true", help="Skip ASR punctuation/segmentation postprocess before source arbitration")
    evidence_pipeline.add_argument("--no-source-arbitration", action="store_true", help="Skip the local source arbitration step and only rebuild semantic correction evidence")
    evidence_pipeline.add_argument("--source-min-confidence", type=float, default=0.72)
    evidence_pipeline.add_argument("--semantic-min-confidence", type=float, default=0.88)
    evidence_pipeline.add_argument("--semantic-limit", type=int, default=80)
    evidence_pipeline.add_argument("--no-refresh-exports", action="store_true")
    evidence_pipeline.add_argument("--no-write", action="store_true")

    main_route_status = sub.add_parser("transcript-main-route-status", help="Audit whether the optimized ASR -> corrected transcript -> smart-summary route is closed for a bundle")
    main_route_status.add_argument("bundle_dir")
    main_route_status.add_argument("--no-write", action="store_true")

    moment_index = sub.add_parser("video-moment-index", help="Build a local queryable VideoRAG/VTime-style moment index from a bundle")
    moment_index.add_argument("bundle_dir")
    moment_index.add_argument("--query", default="", help="Optional query to rank relevant moments")
    moment_index.add_argument("--target-window-seconds", type=float, default=300.0)
    moment_index.add_argument("--max-chunk-chars", type=int, default=3600)
    moment_index.add_argument("--top-k", type=int, default=8)
    moment_index.add_argument("--no-write", action="store_true")

    memory_pack = sub.add_parser("long-video-memory-pack", help="Build a MovieChat-style short/long memory pack for long-video smart summary")
    memory_pack.add_argument("bundle_dir")
    memory_pack.add_argument("--target-window-seconds", type=float, default=300.0)
    memory_pack.add_argument("--max-chunk-chars", type=int, default=3600)
    memory_pack.add_argument("--long-group-size", type=int, default=6)
    memory_pack.add_argument("--no-write", action="store_true")

    video_rag = sub.add_parser("video-rag-pack", help="Build a local VideoRAG-style JSONL retrieval pack from a bundle")
    video_rag.add_argument("bundle_dir")
    video_rag.add_argument("--query", default="", help="Optional query to rank relevant retrieval chunks")
    video_rag.add_argument("--target-window-seconds", type=float, default=300.0)
    video_rag.add_argument("--max-chunk-chars", type=int, default=3600)
    video_rag.add_argument("--top-k", type=int, default=8)
    video_rag.add_argument("--no-write", action="store_true")

    video_rag_search = sub.add_parser("video-rag-search", help="Search the local VideoRAG JSONL chunks without starting a vector backend")
    video_rag_search.add_argument("bundle_dir")
    video_rag_search.add_argument("--query", required=True, help="Query term, tool name, question, or difficult point")
    video_rag_search.add_argument("--top-k", type=int, default=8)
    video_rag_search.add_argument("--retrieval-backend", default="keyword", choices=["keyword", "sqlite", "vector"])
    video_rag_search.add_argument("--no-ensure-pack", action="store_true", help="Do not build video-rag-pack automatically when JSONL is missing")
    video_rag_search.add_argument("--no-write", action="store_true")

    script_clip_candidates = sub.add_parser(
        "script-clip-candidate-pack",
        help="Retrieve local source-clip candidates for explicit script slots; human selection is always required",
    )
    script_clip_candidates.add_argument("bundle_dir")
    script_clip_candidates.add_argument("request_json")
    script_clip_candidates.add_argument("--top-k", type=int, default=8)
    script_clip_candidates.add_argument("--retrieval-backend", choices=["keyword", "sqlite"], default="keyword")
    script_clip_candidates.add_argument("--context-seconds", type=float, default=3.0)
    script_clip_candidates.add_argument("--no-write", action="store_true")

    script_clip_alignment = sub.add_parser(
        "script-clip-alignment-check",
        help="Check human-selected script quotes against fine-cut ranges, clip-only ASR, speakers, and subtitles",
    )
    script_clip_alignment.add_argument("bundle_dir")
    script_clip_alignment.add_argument("review_notes_json")
    script_clip_alignment.add_argument("fine_cut_plan_json")
    script_clip_alignment.add_argument("--candidate-pack-json", default="")
    script_clip_alignment.add_argument("--no-write", action="store_true")

    content_clip_candidates = sub.add_parser(
        "content-clip-candidate-pack",
        help="Retrieve local multimodal candidates for generic content requests; human selection is always required",
    )
    content_clip_candidates.add_argument("bundle_dir")
    content_clip_candidates.add_argument("request_json")
    content_clip_candidates.add_argument("--top-k", type=int, default=8)
    content_clip_candidates.add_argument("--retrieval-backend", choices=["keyword", "sqlite"], default="keyword")
    content_clip_candidates.add_argument("--context-seconds", type=float, default=3.0)
    content_clip_candidates.add_argument("--no-write", action="store_true")

    content_clip_alignment = sub.add_parser(
        "content-clip-alignment-check",
        help="Verify generic fine-cut content against clip-only ASR/OCR/visual/audio and boundary evidence",
    )
    content_clip_alignment.add_argument("bundle_dir")
    content_clip_alignment.add_argument("review_notes_json")
    content_clip_alignment.add_argument("fine_cut_plan_json")
    content_clip_alignment.add_argument("--candidate-pack-json", default="")
    content_clip_alignment.add_argument("--no-write", action="store_true")

    closure_doctor = sub.add_parser(
        "video-closure-doctor",
        help="Run a compact no-write doctor for the candidate-only small-model video closure",
    )
    closure_doctor.add_argument("bundle_dir", nargs="?", default="")
    closure_doctor.add_argument("--mvp-root", default="")

    closure_run = sub.add_parser(
        "video-closure-run",
        help="Run evidence-to-candidate-editing handoff and review packet with zero automatic retries",
    )
    closure_run.add_argument("bundle_dir")
    closure_run.add_argument("--output-dir", default="")
    closure_run.add_argument("--closure-run-id", default="")
    closure_run.add_argument("--mvp-root", default="")
    closure_run.add_argument("--pause-after", choices=["evidence", "segment_plan", "content_brief", "candidate_handoff"], default="")
    closure_run.add_argument("--execution-mode", choices=["candidate_only"], default="candidate_only")

    closure_resume = sub.add_parser(
        "video-closure-resume",
        help="Explicitly resume one hash-bound small-model video closure",
    )
    closure_resume.add_argument("closure_dir")
    closure_resume.add_argument("--mvp-root", default="")

    closure_status = sub.add_parser(
        "video-closure-status",
        help="Validate stage receipts and show one compact closure status",
    )
    closure_status.add_argument("closure_dir")

    long_video_plan = sub.add_parser(
        "long-video-fast-segment-plan",
        help="Build a local, review-only content-aware removal plan for a very long video",
    )
    long_video_plan.add_argument("bundle_dir")
    long_video_plan.add_argument("--media-path", default="")
    long_video_plan.add_argument("--transcript-path", default="")
    long_video_plan.add_argument("--vad-path", default="")
    long_video_plan.add_argument("--activity-audit-path", default="")
    long_video_plan.add_argument("--profile", choices=["auto", "lecture", "interview", "meeting", "tutorial", "vlog"], default="auto")
    long_video_plan.add_argument("--long-silence-seconds", type=float, default=4.0)
    long_video_plan.add_argument("--edge-blank-seconds", type=float, default=8.0)
    long_video_plan.add_argument("--repeat-similarity", type=float, default=0.92)
    long_video_plan.add_argument("--no-write", action="store_true")

    long_video_review = sub.add_parser(
        "apply-long-video-fast-segment-review",
        help="Apply explicit keep/drop decisions to the current long-video plan without editing media",
    )
    long_video_review.add_argument("bundle_dir")
    long_video_review.add_argument("review_json")
    long_video_review.add_argument("--plan-json", default="")
    long_video_review.add_argument("--no-write", action="store_true")

    long_video_render = sub.add_parser(
        "render-long-video-fast-segment",
        help="Preview or explicitly execute a new-copy-only FFmpeg render from a human-approved plan",
    )
    long_video_render.add_argument("bundle_dir")
    long_video_render.add_argument("--approved-json", default="")
    long_video_render.add_argument("--output-path", default="")
    long_video_render.add_argument("--execute", action="store_true")
    long_video_render.add_argument("--timeout-seconds", type=int, default=14400)
    long_video_render.add_argument("--no-write", action="store_true")

    evidence_query = sub.add_parser("video-evidence-query-plan", help="Build a local coarse-to-fine evidence review plan")
    evidence_query.add_argument("bundle_dir")
    evidence_query.add_argument("--query", required=True)
    evidence_query.add_argument("--coarse-top-k", type=int, default=12)
    evidence_query.add_argument("--fine-top-k", type=int, default=4)
    evidence_query.add_argument("--no-write", action="store_true")

    evidence_confirm = sub.add_parser("apply-video-evidence-confirmation", help="Apply explicit confirm/reject/needs-more-evidence decisions")
    evidence_confirm.add_argument("bundle_dir")
    evidence_confirm.add_argument("decisions_json")
    evidence_confirm.add_argument("--plan-json", default="")
    evidence_confirm.add_argument("--no-write", action="store_true")

    video_rag_service = sub.add_parser("video-rag-service-plan", help="Write a local HTTP retrieval service plan for VideoRAG chunks")
    video_rag_service.add_argument("bundle_dir")
    video_rag_service.add_argument("--host", default="127.0.0.1")
    video_rag_service.add_argument("--port", type=int, default=8781)
    video_rag_service.add_argument("--no-write", action="store_true")

    video_rag_serve = sub.add_parser("video-rag-serve", help="Start a local HTTP VideoRAG retrieval server for one bundle")
    video_rag_serve.add_argument("bundle_dir")
    video_rag_serve.add_argument("--host", default="127.0.0.1")
    video_rag_serve.add_argument("--port", type=int, default=8781)
    capability_pack = sub.add_parser("external-capability-pack", help="Bundle the reusable external-video-project capabilities for a VKP bundle")
    capability_pack.add_argument("bundle_dir")
    capability_pack.add_argument("--query", default="", help="Optional query passed into time localization and RAG packs")
    capability_pack.add_argument("--no-write", action="store_true")

    vision_smoke = sub.add_parser("vision-provider-smoke", help="Run and persist a no-secret provider text/image JSON smoke report")
    vision_smoke.add_argument("--provider-config", help="Inline JSON or JSON file with provider/base_url/model/api_key; local_qwen_vl/local_vlm may omit api_key")
    vision_smoke.add_argument("--provider", default="")
    vision_smoke.add_argument("--model", default="")
    vision_smoke.add_argument("--base-url", default="")
    vision_smoke.add_argument("--timeout-seconds", type=int)
    vision_smoke.add_argument("--bundle-dir", default="")
    vision_smoke.add_argument("--single-image", default="")
    vision_smoke.add_argument("--multi-image-dir", default="")
    vision_smoke.add_argument("--output-dir", default="")
    vision_smoke.add_argument("--image-probe-max-edge", type=int, default=0, help="Resize smoke images to this max edge before provider checks; 0 keeps originals")
    vision_smoke.add_argument("--image-probe-jpeg-quality", type=int, default=70)
    vision_smoke.add_argument("--max-images", type=int, default=8)
    vision_smoke.add_argument("--no-write", action="store_true")

    vision_matrix = sub.add_parser("vision-provider-matrix", help="Run and persist secret-safe smoke checks across candidate providers")
    vision_matrix.add_argument("--providers", default="local_qwen_vl,volcengine_coding_plan,gemini,openai,agnes", help="Comma-separated provider profiles to test")
    vision_matrix.add_argument("--timeout-seconds", type=int)
    vision_matrix.add_argument("--bundle-dir", default="")
    vision_matrix.add_argument("--single-image", default="")
    vision_matrix.add_argument("--multi-image-dir", default="")
    vision_matrix.add_argument("--output-dir", default="")
    vision_matrix.add_argument("--image-probe-max-edge", type=int, default=0, help="Resize smoke images to this max edge before provider checks; 0 keeps originals")
    vision_matrix.add_argument("--image-probe-jpeg-quality", type=int, default=70)
    vision_matrix.add_argument("--max-images", type=int, default=8)
    vision_matrix.add_argument("--preferred-provider", default="", help="Tie-breaker preference after key and image JSON checks pass")
    vision_matrix.add_argument("--no-write", action="store_true")

    vision_env = sub.add_parser("vision-env-status", help="Inspect or write a no-secret local vision provider env template")
    vision_env.add_argument("--provider", default="")
    vision_env.add_argument("--model", default="")
    vision_env.add_argument("--write-template", action="store_true")
    vision_env.add_argument("--template-path", default="")
    vision_env.add_argument("--overwrite", action="store_true")

    acceptance = sub.add_parser("vision-acceptance-plan", help="Write a no-secret runbook for the first real multimodal API acceptance run")
    acceptance.add_argument("bundle_dir")
    acceptance.add_argument("--provider-config", help="Inline JSON or JSON file with provider/base_url/model/api_key")
    acceptance.add_argument("--semantic-limit", type=int)
    acceptance.add_argument("--temporal-limit", type=int)
    acceptance.add_argument("--frame-count", type=int)
    acceptance.add_argument("--no-write", action="store_true")

    visual_ab = sub.add_parser("visual-ab-benchmark-plan", help="Select a full-duration A/B/C sample without calling an online model")
    visual_ab.add_argument("bundle_dir")
    visual_ab.add_argument("--limit", type=int, default=10)
    visual_ab.add_argument("--min-score", type=int, default=4)
    visual_ab.add_argument("--no-write", action="store_true")

    consent_create = sub.add_parser("vision-export-consent-create", help="Create a scoped, expiring authorisation for selected frames sent by an agent")
    consent_create.add_argument("bundle_dir")
    consent_create.add_argument("--provider-config", default="")
    consent_create.add_argument("--semantic-indexes", default="")
    consent_create.add_argument("--temporal-indexes", default="")
    consent_create.add_argument("--max-calls", type=int)
    consent_create.add_argument("--expires-hours", type=float, default=24.0)
    consent_create.add_argument("--image-max-edge", type=int, default=512)
    consent_create.add_argument("--image-jpeg-quality", type=int, default=55)
    consent_create.add_argument("--purpose", default="targeted multimodal review")
    consent_create.add_argument("--confirm-data-export", action="store_true")
    consent_create.add_argument("--output-path", default="")
    consent_create.add_argument("--no-write", action="store_true")

    consent_status = sub.add_parser("vision-export-consent-status", help="Validate a scoped vision export consent without calling a provider")
    consent_status.add_argument("bundle_dir")
    consent_status.add_argument("--consent-path", default="")
    consent_status.add_argument("--provider-config", default="")
    consent_status.add_argument("--semantic-indexes", default="")
    consent_status.add_argument("--temporal-indexes", default="")
    consent_status.add_argument("--expected-calls", type=int)
    consent_status.add_argument("--image-max-edge", type=int, default=512)
    consent_status.add_argument("--image-jpeg-quality", type=int, default=55)

    consent_revoke = sub.add_parser("vision-export-consent-revoke", help="Revoke a scoped vision export consent")
    consent_revoke.add_argument("bundle_dir")
    consent_revoke.add_argument("--consent-path", default="")
    consent_revoke.add_argument("--no-write", action="store_true")

    preflight = sub.add_parser("vision-execution-preflight", help="Check provider readiness, candidate scale, write fields, and restore chain before real vision execution")
    preflight.add_argument("bundle_dir")
    preflight.add_argument("--provider-config", help="Inline JSON or JSON file with provider/base_url/model/api_key; local_qwen_vl/local_vlm may omit api_key")
    preflight.add_argument("--semantic-limit", type=int)
    preflight.add_argument("--temporal-limit", type=int)
    preflight.add_argument("--frame-count", type=int)
    preflight.add_argument("--semantic-indexes", default="")
    preflight.add_argument("--temporal-indexes", default="")
    preflight.add_argument("--semantic-index", action="append", type=int, default=[])
    preflight.add_argument("--temporal-index", action="append", type=int, default=[])
    preflight.add_argument("--no-semantic", action="store_true")
    preflight.add_argument("--no-temporal", action="store_true")
    preflight.add_argument("--check-provider", action="store_true", help="Run provider smoke checks and block if text/image JSON checks fail")
    preflight.add_argument("--no-write", action="store_true")

    timeline_alignment = sub.add_parser("timeline-alignment-audit", help="Audit ASR/frame/tagger/review timestamp alignment without modifying timeline")
    timeline_alignment.add_argument("bundle_dir")
    timeline_alignment.add_argument("--tolerance-seconds", type=float, default=2.0)
    timeline_alignment.add_argument("--no-write", action="store_true")

    entity_lexicon = sub.add_parser("build-entity-lexicon", help="Build Chinese entity aliases, pinyin evidence, and dynamic ASR hotwords without applying corrections")
    entity_lexicon.add_argument("bundle_dir")
    entity_lexicon.add_argument("--base-lexicon-json", default="")
    entity_lexicon.add_argument("--phase", choices=["pre_asr", "post_asr"], default="post_asr")
    entity_lexicon.add_argument("--no-write", action="store_true")

    terms = sub.add_parser("resolve-terms", help="Resolve likely canonical terminology across ASR, subtitles, OCR, visual evidence, tagger data, and metadata")
    terms.add_argument("bundle_dir")
    terms.add_argument("--metadata-json", default="")
    terms.add_argument("--glossary-json", default="")
    terms.add_argument("--min-mentions", type=int, default=1)
    terms.add_argument("--no-write", action="store_true")

    term_codex = sub.add_parser("term-arbitration-codex", help="Build/import a Codex-reviewed terminology arbitration pack for tool names and ASR/OCR conflicts")
    term_codex.add_argument("bundle_dir")
    term_codex.add_argument("--input-json", default="", help="Reviewed Codex/LLM JSON decisions to import")
    term_codex.add_argument("--max-terms", type=int, default=60)
    term_codex.add_argument("--min-confidence", type=float, default=0.88)
    term_codex.add_argument("--accept-draft", action="store_true", help="Import only high-confidence local Codex-substitute draft decisions instead of waiting for reviewed input JSON")
    term_codex.add_argument("--no-write", action="store_true")

    validate_term = sub.add_parser("validate-term-arbitration-codex-result", help="Validate a Codex/LLM terminology arbitration JSON or Markdown response before importing")
    validate_term.add_argument("bundle_dir")
    validate_term.add_argument("--input-json", required=True, help="Codex/LLM JSON or Markdown response containing JSON decisions")
    validate_term.add_argument("--min-confidence", type=float, default=0.88)
    validate_term.add_argument("--no-write", action="store_true")

    term_impact = sub.add_parser("term-correction-impact-report", help="Measure whether reviewed term corrections reached corrected transcript and final exports")
    term_impact.add_argument("bundle_dir")
    term_impact.add_argument("--min-confidence", type=float, default=0.88)
    term_impact.add_argument("--no-write", action="store_true")

    term_status = sub.add_parser("term-correction-status", help="Read current terminology correction status and next action without modifying the bundle")
    term_status.add_argument("bundle_dir")

    term_closure = sub.add_parser("term-correction-closure", help="Run local Codex-substitute terminology closure through transcript arbitration, impact check, and exports")
    term_closure.add_argument("bundle_dir")
    term_closure.add_argument("--accept-draft", action="store_true", help="Accept only high-confidence local Codex-substitute draft term decisions")
    term_closure.add_argument("--input-json", default="", help="Codex/LLM reviewed JSON or Markdown response to import before transcript arbitration")
    term_closure.add_argument("--max-terms", type=int, default=60)
    term_closure.add_argument("--term-min-confidence", type=float, default=0.88)
    term_closure.add_argument("--transcript-min-confidence", type=float, default=0.72)
    term_closure.add_argument("--no-generate-codex-summary", action="store_true")
    term_closure.add_argument("--no-write", action="store_true")

    sem_pack = sub.add_parser("transcript-semantic-correction-pack", help="Build a general ASR/subtitle semantic correction evidence pack")
    sem_pack.add_argument("bundle_dir")
    sem_pack.add_argument("--run-id", default=None)
    sem_pack.add_argument("--dependency-versions", default="")
    sem_pack.add_argument("--limit", type=int, default=0)
    sem_pack.add_argument(
        "--source-mode",
        choices=["raw", "canonical"],
        default="raw",
        help="Use raw error evidence (default) or the current canonical transcript",
    )
    sem_pack.add_argument("--no-write", action="store_true")

    sem_discovery = sub.add_parser("transcript-semantic-candidate-discovery-pack", help="Build a Codex/LLM prompt to discover missed ASR/subtitle semantic correction candidates")
    sem_discovery.add_argument("bundle_dir")
    sem_discovery.add_argument("--input-json", default="", help="Optional transcript-semantic-correction-pack.json path")
    sem_discovery.add_argument("--limit", type=int, default=40)
    sem_discovery.add_argument("--no-write", action="store_true")

    sem_candidate_discovery_codex = sub.add_parser("transcript-semantic-candidate-discovery-codex-draft", help="Generate local Codex-substitute candidate suggestions from discovery pack")
    sem_candidate_discovery_codex.add_argument("bundle_dir")
    sem_candidate_discovery_codex.add_argument("--input-json", default="", help="Optional transcript-semantic-correction-pack.json path")
    sem_candidate_discovery_codex.add_argument("--limit", type=int, default=40)
    sem_candidate_discovery_codex.add_argument("--max-suggestions", type=int, default=40)
    sem_candidate_discovery_codex.add_argument("--no-write", action="store_true")

    sem_candidate_import = sub.add_parser("import-transcript-semantic-candidate-suggestions", help="Import Codex/LLM-discovered suspicious spans as normal semantic correction candidates")
    sem_candidate_import.add_argument("bundle_dir")
    sem_candidate_import.add_argument("--input-json", required=True)
    sem_candidate_import.add_argument("--no-write", action="store_true")

    sem_candidate_discovery_llm = sub.add_parser("transcript-semantic-candidate-discovery-llm-draft", help="Plan or execute text LLM discovery of missed ASR/subtitle semantic correction candidates")
    sem_candidate_discovery_llm.add_argument("bundle_dir")
    sem_candidate_discovery_llm.add_argument("--input-json", default="", help="Optional transcript-semantic-correction-pack.json path")
    sem_candidate_discovery_llm.add_argument("--provider-config", default="", help="Inline JSON, JSON file, or provider profile; only used with --execute")
    sem_candidate_discovery_llm.add_argument("--execute", action="store_true", help="Actually call configured text LLM provider; default only writes prompt/request plan")
    sem_candidate_discovery_llm.add_argument("--limit", type=int, default=40)
    sem_candidate_discovery_llm.add_argument("--no-write", action="store_true")

    sem_codex_draft = sub.add_parser("transcript-semantic-correction-codex-draft", help="Generate a conservative local Codex-substitute semantic correction result draft")
    sem_codex_draft.add_argument("bundle_dir")
    sem_codex_draft.add_argument("--run-id", default=None)
    sem_codex_draft.add_argument("--dependency-versions", default="")
    sem_codex_draft.add_argument("--input-json", default="", help="Optional transcript-semantic-correction-pack.json path")
    sem_codex_draft.add_argument("--min-confidence", type=float, default=0.88)
    sem_codex_draft.add_argument("--no-write", action="store_true")


    sem_llm_draft = sub.add_parser("transcript-semantic-correction-llm-draft", help="Plan or execute text LLM semantic correction review over transcript candidates")
    sem_llm_draft.add_argument("bundle_dir")
    sem_llm_draft.add_argument("--input-json", default="", help="Optional transcript-semantic-correction-pack.json path")
    sem_llm_draft.add_argument("--provider-config", default="", help="Inline JSON, JSON file, or provider profile; only used with --execute")
    sem_llm_draft.add_argument("--execute", action="store_true", help="Actually call configured text LLM provider; default only writes prompt/request plan")
    sem_llm_draft.add_argument("--limit", type=int, default=80)
    sem_llm_draft.add_argument("--min-confidence", type=float, default=0.88)
    sem_llm_draft.add_argument("--business-authorization", default="", help="Active parent authorization; required for remote Proxy execution")
    sem_llm_draft.add_argument("--no-write", action="store_true")
    sem_validate = sub.add_parser("validate-transcript-semantic-correction", help="Validate Codex/LLM semantic correction JSON or Markdown response")
    sem_validate.add_argument("bundle_dir")
    sem_validate.add_argument("--input-json", required=True)
    sem_validate.add_argument("--min-confidence", type=float, default=0.88)
    sem_validate.add_argument("--no-write", action="store_true")

    sem_review_import = sub.add_parser("import-transcript-semantic-review-notes", help="Convert human semantic correction review notes into validated correction result JSON")
    sem_review_import.add_argument("bundle_dir")
    sem_review_import.add_argument("--review-json", default="")
    sem_review_import.add_argument("--min-confidence", type=float, default=0.88)
    sem_review_import.add_argument("--no-write", action="store_true")
    sem_closure = sub.add_parser("transcript-semantic-correction-closure", help="Apply validated semantic corrections to corrected transcript sidecar")
    sem_closure.add_argument("bundle_dir")
    sem_closure.add_argument("--input-json", required=True)
    sem_closure.add_argument("--min-confidence", type=float, default=0.88)
    sem_closure.add_argument("--auto-apply", action="store_true")
    sem_closure.add_argument("--refresh-exports", action="store_true", help="After writing the corrected transcript, refresh full-transcript/smart-summary and semantic impact reports")
    sem_closure.add_argument("--no-write", action="store_true")

    sem_impact = sub.add_parser("transcript-semantic-correction-impact-report", help="Check semantic correction residuals in corrected transcript and final exports")
    sem_impact.add_argument("bundle_dir")
    sem_impact.add_argument("--no-write", action="store_true")


    sem_readable_impact = sub.add_parser("transcript-semantic-readable-impact-report", help="Report whether accepted semantic corrections reached full-transcript and smart-summary")
    sem_readable_impact.add_argument("bundle_dir")
    sem_readable_impact.add_argument("--no-write", action="store_true")
    sem_summary_impact = sub.add_parser("transcript-semantic-summary-impact-report", help="Report whether smart-summary visibly absorbs accepted semantic corrections")
    sem_summary_impact.add_argument("bundle_dir")
    sem_summary_impact.add_argument("--summary-path", default="")
    sem_summary_impact.add_argument("--baseline-summary-path", default="")
    sem_summary_impact.add_argument("--no-write", action="store_true")
    sem_status = sub.add_parser("transcript-semantic-correction-status", help="Read general ASR/subtitle semantic correction status")
    sem_status.add_argument("bundle_dir")
    sem_status.add_argument("--no-write", action="store_true")

    sem_benchmark = sub.add_parser(
        "transcript-semantic-benchmark",
        help="Evaluate a de-identified semantic-correction run against a local synthetic gold set",
    )
    sem_benchmark.add_argument("gold_json")
    sem_benchmark.add_argument("candidate_pack_json")
    sem_benchmark.add_argument("validation_json")
    sem_benchmark.add_argument("--full-body-path", default="")
    sem_benchmark.add_argument("--smart-summary-path", default="")
    sem_benchmark.add_argument("--output-json", default="")

    sem_restore = sub.add_parser("transcript-semantic-restore", help="Preview or apply a hash-bound semantic transcript restore plan")
    sem_restore.add_argument("bundle_dir")
    sem_restore.add_argument("--plan-json", required=True)
    sem_restore.add_argument("--execute", action="store_true", help="Apply the restore; default is preview only")
    sem_restore.add_argument("--confirm-transaction-id", default="", help="Exact transaction ID required with --execute")

    sem_acceptance = sub.add_parser("transcript-semantic-acceptance", help="Read-only single-bundle semantic correction acceptance proof")
    sem_acceptance.add_argument("bundle_dir")
    sem_acceptance.add_argument("--output-dir", default="")
    sem_acceptance.add_argument("--no-write", action="store_true")

    sem_batch = sub.add_parser("transcript-semantic-batch-acceptance", help="Summarize semantic correction acceptance across bundles")
    sem_batch.add_argument("batch_input")
    sem_batch.add_argument("--output-dir", default="")
    sem_batch.add_argument("--target-bundle-count", type=int, default=3)
    sem_batch.add_argument("--limit", type=int, default=0, help="Maximum bundles to inspect; 0 means all discovered")
    sem_batch.add_argument("--no-write", action="store_true")
    sem_queue = sub.add_parser("transcript-semantic-repair-queue", help="Build a preview-only retry queue for transcript semantic correction across bundles")
    sem_queue.add_argument("batch_input")
    sem_queue.add_argument("--output-dir", default="")
    sem_queue.add_argument("--target-bundle-count", type=int, default=3)
    sem_queue.add_argument("--limit", type=int, default=0, help="Maximum bundles to inspect; 0 means all discovered")
    sem_queue.add_argument("--no-write", action="store_true")
    sem_run = sub.add_parser("transcript-semantic-repair-run", help="Preview or execute safe local semantic-correction repair actions from the queue")
    sem_run.add_argument("batch_input")
    sem_run.add_argument("--output-dir", default="")
    sem_run.add_argument("--target-bundle-count", type=int, default=3)
    sem_run.add_argument("--limit", type=int, default=0, help="Maximum bundles to inspect; 0 means all discovered")
    sem_run.add_argument("--max-actions", type=int, default=0, help="Maximum queued actions to process; 0 means all actionable rows")
    sem_run.add_argument("--max-rounds", type=int, default=1, help="Maximum safe repair rounds when --execute-safe-actions is set; default is 1")
    sem_run.add_argument("--execute-safe-actions", action="store_true", help="Run local safe actions; default is preview only")
    sem_run.add_argument("--allow-closure", action="store_true", help="Allow validated closure writes to source-arbitrated transcript sidecars")
    sem_run.add_argument("--allow-llm", action="store_true", help="Allow queued text LLM review actions to call the configured provider")
    sem_run.add_argument("--provider-config", default="", help="Inline JSON, JSON file, or provider profile for --allow-llm execution")
    sem_run.add_argument("--llm-limit", type=int, default=80, help="Maximum semantic candidates sent per queued LLM draft action")
    sem_run.add_argument("--business-authorization", default="", help="One active parent authorization reused across bound Bundles")
    sem_run.add_argument("--no-write", action="store_true")
    sem_batch_review = sub.add_parser("transcript-semantic-batch-review-pack", help="Build a cross-bundle semantic correction review pack and todo JSON")
    sem_batch_review.add_argument("batch_input")
    sem_batch_review.add_argument("--output-dir", default="")
    sem_batch_review.add_argument("--target-bundle-count", type=int, default=3)
    sem_batch_review.add_argument("--limit", type=int, default=0, help="Maximum bundles to inspect; 0 means all discovered")
    sem_batch_review.add_argument("--max-candidates-per-bundle", type=int, default=0, help="Maximum review candidates per bundle; 0 means all")
    sem_batch_review.add_argument("--no-write", action="store_true")
    sem_batch_import = sub.add_parser("transcript-semantic-batch-import-review-notes", help="Import cross-bundle semantic review notes into each bundle")
    sem_batch_import.add_argument("review_json")
    sem_batch_import.add_argument("--output-dir", default="")
    sem_batch_import.add_argument("--min-confidence", type=float, default=0.88)
    sem_batch_import.add_argument("--no-write", action="store_true")
    sem_batch_codex = sub.add_parser("transcript-semantic-batch-codex-review-draft", help="Generate a conservative local Codex-substitute review notes draft from a batch review pack")
    sem_batch_codex.add_argument("review_pack_json")
    sem_batch_codex.add_argument("--output-dir", default="")
    sem_batch_codex.add_argument("--no-write", action="store_true")
    targeted_visual = sub.add_parser("targeted-visual-evidence", help="Run local-first ebook/crop/tile routing and expose only unresolved indexes for online or human review")
    targeted_visual.add_argument("bundle_dir")
    targeted_visual.add_argument("--tagger-json", default="")
    targeted_visual.add_argument("--min-score", type=int, default=3)
    targeted_visual.add_argument("--limit", type=int, default=0)
    targeted_visual.add_argument("--execute-ebook", action="store_true")
    targeted_visual.add_argument("--execute-crops", action="store_true")
    targeted_visual.add_argument("--execute-ocr", action="store_true")
    targeted_visual.add_argument("--execute-tiles", action="store_true")
    targeted_visual.add_argument("--allow-online-review", action="store_true", help="Only mark unresolved indexes eligible; actual vision call still requires preflight")
    targeted_visual.add_argument("--no-write", action="store_true")

    triage = sub.add_parser("vision-review-triage", help="Plan full or triaged ebook/multimodal review indexes from ASR, OCR, routes, and frame evidence")
    triage.add_argument("bundle_dir")
    triage.add_argument("--mode", choices=["fast", "triage", "full"], default="fast", help="fast selects a distributed production subset; triage selects all risky items; full selects all frame items")
    triage.add_argument("--tagger-json", help="Qinglong/manual tagger JSON; supports index or timestamp alignment")
    triage.add_argument("--semantic-limit", type=int)
    triage.add_argument("--temporal-limit", type=int)
    triage.add_argument("--visual-structure-limit", type=int)
    triage.add_argument("--min-score", type=int, default=3)
    triage.add_argument("--no-write", action="store_true")

    supplemental = sub.add_parser("plan-supplemental-frame-sampling", help="Plan local supplemental frame recapture from triage candidates; does not execute ffmpeg or cloud vision")
    supplemental.add_argument("bundle_dir")
    supplemental.add_argument("--triage-json", default="")
    supplemental.add_argument("--max-items", type=int, default=0, help="Maximum triage candidates to plan; 0 means all")
    supplemental.add_argument("--max-frames-per-item", type=int, default=4, help="Local recapture frames per selected timeline item")
    supplemental.add_argument("--no-temporal", action="store_true")
    supplemental.add_argument("--no-visual-structure", action="store_true")
    supplemental.add_argument("--no-semantic", action="store_true")
    supplemental.add_argument("--no-write", action="store_true")

    queue = sub.add_parser("vision-review-queue", help="Build a retryable batched multimodal queue from triage candidates")
    queue.add_argument("bundle_dir")
    queue.add_argument("--min-score", type=int, default=10, help="Only queue triage candidates at or above this score")
    queue.add_argument("--batch-size", type=int, default=10, help="How many frames/indexes per batch")
    queue.add_argument("--max-items", type=int, default=0, help="Maximum queued items; 0 means all matching candidates")
    queue.add_argument("--provider", default="volcengine_coding_plan")
    queue.add_argument("--env-file", default=str(provider_env_file()))
    queue.add_argument("--refresh-triage", action="store_true")
    queue.add_argument("--no-write", action="store_true")


    run_registry = sub.add_parser("run-artifact-registry", help="Refresh the local run/artifact registry for one WebUI bundle")
    run_registry.add_argument("bundle_dir")
    run_registry.add_argument("--no-write", action="store_true")

    attestation_create = sub.add_parser("review-attestation-create", help="Create an immutable content-bound review attestation for bundle artifacts")
    attestation_create.add_argument("bundle_dir")
    attestation_create.add_argument("--target", required=True)
    attestation_create.add_argument("--artifact", action="append", required=True, help="Repeat role=path for every exact dependency artifact")
    attestation_create.add_argument("--approved-by", required=True)
    attestation_create.add_argument("--comment", default="")
    attestation_create.add_argument("--no-write", action="store_true")

    attestation_status = sub.add_parser("review-attestation-status", help="Validate the current or exact review attestation against current bundle artifacts")
    attestation_status.add_argument("bundle_dir")
    attestation_status.add_argument("--target", default="")
    attestation_status.add_argument("--attestation-path", default="")

    generation_import = sub.add_parser("import-generation-contracts", help="Import fixed-upstream generation task, receipt, validation, preflight, and representative-frame evidence")
    generation_import.add_argument("bundle_dir")
    generation_import.add_argument("--task", required=True)
    generation_import.add_argument("--receipt", required=True)
    generation_import.add_argument("--validation", required=True)
    generation_import.add_argument("--preflight", default="", help="Exact preflight JSON; defaults to the task-bound path")
    generation_import.add_argument("--source-root", action="append", default=[], help="Allowed local source root; repeat for external contract directories")
    generation_import.add_argument("--no-write", action="store_true")

    previs_import = sub.add_parser("import-previs-candidate", help="Import fixed-upstream 3D previs captures as synthetic candidate evidence")
    previs_import.add_argument("bundle_dir")
    previs_import.add_argument("--scene", required=True)
    previs_import.add_argument("--capture-manifest", required=True)
    previs_import.add_argument("--validation", required=True)
    previs_import.add_argument("--source-root", action="append", default=[], help="Allowed local source root; repeat for external contract directories")
    previs_import.add_argument("--no-write", action="store_true")

    material_manifest = sub.add_parser(
        "material-manifest",
        help="Project local transcript, keyframes, temporal evidence, and Bundle metadata into material-manifest.v1",
    )
    material_manifest.add_argument("bundle_dir")
    material_manifest.add_argument("--transcript", default="", help="Optional exact transcript path; defaults to VKP canonical transcript selection")
    material_manifest.add_argument("--output", default="", help="Optional Bundle-local output path; defaults to exports/material-manifest.v1.json")
    material_manifest.add_argument("--no-write", action="store_true")

    material_validate = sub.add_parser(
        "material-manifest-validate",
        help="Validate material-manifest.v1 schema, hashes, source order, local artifacts, and Bundle freshness",
    )
    material_validate.add_argument("bundle_dir")
    material_validate.add_argument("--manifest-path", default="", help="Defaults to exports/material-manifest.v1.json")
    material_validate.add_argument("--write-report", action="store_true", help="Write exports/material-manifest-validation.json")


    sample_review = sub.add_parser("multimodal-sample-review", help="Build a static UI for human sampling of multimodal accuracy impact")
    sample_review.add_argument("bundle_dir")
    sample_review.add_argument("--comparison-json", default="")
    sample_review.add_argument("--sample-size", type=int, default=30)
    sample_review.add_argument("--no-missing", action="store_true", help="Do not include missing visual-understanding samples")
    sample_review.add_argument("--media-path", default="", help="Original video path for PotPlayer timestamp review commands")
    sample_review.add_argument("--potplayer-path", default="", help="Optional PotPlayer executable path used by generated jump commands")
    sample_review.add_argument("--no-write", action="store_true")

    sample_notes = sub.add_parser("validate-multimodal-sample-notes", help="Validate and summarize human labels exported from multimodal-sample-review.html")
    sample_notes.add_argument("bundle_dir")
    sample_notes.add_argument("--notes-json", default="")
    sample_notes.add_argument("--min-reviewed", type=int, default=10)
    sample_notes.add_argument("--no-write", action="store_true")
    local_vlm = sub.add_parser("local-vlm-adapter-plan", help="Inspect planned local VLM adapter contracts without launching models")
    local_vlm.add_argument("--output-dir", default="")
    local_vlm.add_argument("--write", action="store_true")

    vlm_smoke = sub.add_parser("local-vlm-serving-smoke", help="Plan or run a minimal smoke check against a local Qwen/InternVL OpenAI-compatible server")
    vlm_smoke.add_argument("--provider", default="local_qwen_vl")
    vlm_smoke.add_argument("--bundle-dir", default="")
    vlm_smoke.add_argument("--output-dir", default="")
    vlm_smoke.add_argument("--single-image", default="")
    vlm_smoke.add_argument("--multi-image-dir", default="")
    vlm_smoke.add_argument("--execute", action="store_true", help="Actually call the already-running local VLM server")
    vlm_smoke.add_argument("--timeout-seconds", type=int, default=30)
    vlm_smoke.add_argument("--max-images", type=int, default=3)
    vlm_smoke.add_argument("--image-probe-max-edge", type=int, default=512)
    vlm_smoke.add_argument("--image-probe-jpeg-quality", type=int, default=70)
    vlm_smoke.add_argument("--frame-group-count", type=int, default=8)
    vlm_smoke.add_argument("--base-url", default="", help="Explicit loopback OpenAI-compatible URL; LM Studio commonly uses http://127.0.0.1:1234/v1")
    vlm_smoke.add_argument("--model", default="", help="Explicit locally loaded model id; never read from or written to pipeline config")
    vlm_smoke.add_argument("--max-tokens", type=int, default=0)
    vlm_smoke.add_argument("--no-write", action="store_true")

    coverage = sub.add_parser("audit-knowledge-coverage", help="Audit no-loss knowledge-channel coverage")
    coverage.add_argument("bundle_dir")
    companion_courseware = sub.add_parser("import-companion-courseware-text", help="Import local companion courseware text; it is not video-frame OCR")
    companion_courseware.add_argument("bundle_dir")
    companion_courseware.add_argument("source_path")
    companion_courseware.add_argument("--title", default="")
    companion_courseware.add_argument("--no-write", action="store_true")
    coverage.add_argument("--no-write", action="store_true")

    export = sub.add_parser("export-knowledge-note", help="Export a human-readable Obsidian-friendly knowledge note")
    export.add_argument("bundle_dir")
    export.add_argument("--output-dir", default="")
    export.add_argument("--title", default="")
    export.add_argument("--no-timeline", action="store_true")
    export.add_argument("--no-full-transcript", action="store_true")
    export.add_argument("--no-transcript-evidence-check", action="store_true", help="Skip the default local transcript evidence correction preflight before export")
    export.add_argument("--no-write", action="store_true")

    codex_summary = sub.add_parser("generate-smart-summary-with-codex", help="Generate or install and validate a local Codex-style final smart summary")
    codex_summary.add_argument("bundle_dir")
    codex_summary.add_argument("--input-md", default="", help="Validate and register a candidate; install only after all formal gates pass")
    codex_summary.add_argument("--handoff-id", default="", help="Bind the candidate to the prepared Codex handoff and its input hashes")
    codex_summary.add_argument("--agent-available", action="store_true", help="Current authorized Agent can write the candidate; does not start a model or thread")
    codex_summary.add_argument("--no-write", action="store_true")

    llm_rewrite = sub.add_parser("prepare-smart-summary-llm-rewrite", help="Prepare a real LLM/Codex rewrite handoff pack for final smart-summary generation")
    llm_rewrite.add_argument("bundle_dir")
    llm_rewrite.add_argument("--provider", default="codex_manual", help="Rewrite provider profile; default codex_manual writes a prompt and waits for Codex output")
    llm_rewrite.add_argument("--no-write", action="store_true")

    llm_run = sub.add_parser("run-smart-summary-llm-rewrite", help="Execute the OpenAI-compatible text LLM rewrite layer for final smart-summary generation")
    llm_run.add_argument("bundle_dir")
    llm_run.add_argument("--provider-config", default="", help="Inline JSON or JSON file with provider/base_url/model/api_key; API key is runtime-only")
    llm_run.add_argument("--execute", action="store_true", help="Actually call the configured text LLM provider")
    llm_run.add_argument("--max-input-chars", type=int, default=60000)
    llm_run.add_argument("--temperature", type=float, default=0)
    llm_run.add_argument("--no-install", action="store_true", help="Only write smart-summary.llm.md; do not install through the quality gate")
    llm_run.add_argument("--no-write", action="store_true")

    summary_pack = sub.add_parser("build-smart-summary-input-pack", help="Build corrected transcript, term, and visual evidence input pack for Codex smart-summary rewriting")
    summary_pack.add_argument("bundle_dir")
    summary_pack.add_argument("--title", default="")
    summary_pack.add_argument("--max-visual-items", type=int, default=80)
    summary_pack.add_argument("--no-write", action="store_true")

    technical_shots = sub.add_parser("technical-shot-detection", help="Build strict technical-shot candidates from one explicit local backend or saved predictions")
    technical_shots.add_argument("bundle_dir")
    technical_shots.add_argument("--backend", choices=["pyscenedetect", "autoshot", "omnishotcut", "saved"], default="autoshot")
    technical_shots.add_argument("--media-path", default="")
    technical_shots.add_argument("--predictions-json", default="")
    technical_shots.add_argument("--source-format", choices=["", "autoshot_scenes", "omnishotcut_scenes", "transnetv2_scenes"], default="")
    technical_shots.add_argument("--frame-rate", type=float, default=None)
    technical_shots.add_argument("--detector", choices=["adaptive", "content"], default="adaptive")
    technical_shots.add_argument("--threshold", type=float, default=None)
    technical_shots.add_argument("--min-scene-len", type=int, default=15)
    technical_shots.add_argument("--source-root", default="")
    technical_shots.add_argument("--checkpoint-path", default="")
    technical_shots.add_argument("--allow-fallback", action="store_true", help="Allow the explicitly reported legacy ffmpeg fallback; strict mode is the default")
    technical_shots.add_argument("--strict", action="store_true", help="Explicitly require the verified backend and reject all fallback (already the default)")
    technical_shots.add_argument("--no-write", action="store_true")

    scene_detection = sub.add_parser("scene-detection", help="Run local PySceneDetect with an explicit ffmpeg fallback")
    scene_detection.add_argument("bundle_dir")
    scene_detection.add_argument("--media-path", default="")
    scene_detection.add_argument("--detector", choices=["adaptive", "content"], default="adaptive")
    scene_detection.add_argument("--threshold", type=float, default=None)
    scene_detection.add_argument("--min-scene-len", type=int, default=15)
    scene_detection.add_argument("--max-points", type=int, default=300)
    scene_detection.add_argument("--source-root", default="")
    scene_detection.add_argument("--no-write", action="store_true")

    shot_language = sub.add_parser("shot-language-analysis", help="Build evidence-bound per-shot facts with fixed local Auto Scenes adapters")
    shot_language.add_argument("bundle_dir")
    shot_language.add_argument("--execution-location", choices=["local"], default="local")
    shot_language.add_argument("--route-id", default="", help="Explicit local VLM escalation route; no automatic execution")
    shot_language.add_argument("--source-root", default="")
    shot_language.add_argument("--shot-scale-model-path", default="")
    shot_language.add_argument("--shot-type-confidence-threshold", type=float, default=0.65)
    shot_language.add_argument("--movement-confidence-threshold", type=float, default=0.65)
    shot_language.add_argument("--execute", action="store_true", help="Run fixed local GPU analyzers; never downloads models or falls back remotely")
    shot_language.add_argument("--no-write", action="store_true")

    shot_review_apply = sub.add_parser("shot-review-apply", help="Validate and formally apply hash-bound shot review notes as a derived projection")
    shot_review_apply.add_argument("bundle_dir")
    shot_review_apply.add_argument("review_notes")
    shot_review_apply.add_argument("--no-write", action="store_true")

    shot_review_status_parser = sub.add_parser("shot-review-status", help="Check whether the current derived shot review is active or stale")
    shot_review_status_parser.add_argument("bundle_dir")

    shot_fusion = sub.add_parser("technical-shot-fusion", help="Cluster explicit detector candidates within a frame tolerance without selecting disputed cuts")
    shot_fusion.add_argument("bundle_dir")
    shot_fusion.add_argument("candidate_paths", nargs="+")
    shot_fusion.add_argument("--frame-rate", type=float, required=True)
    shot_fusion.add_argument("--tolerance-frames", type=int, default=2)
    shot_fusion.add_argument("--no-write", action="store_true")

    scene_candidates = sub.add_parser("scene-candidate-evidence", help="Import provenance-locked scene boundaries as candidate evidence without changing Timeline")
    scene_candidates.add_argument("bundle_dir")
    scene_candidates.add_argument("candidates_json")
    scene_candidates.add_argument("--model-id", required=True)
    scene_candidates.add_argument("--model-commit", default="unversioned")
    scene_candidates.add_argument("--language", default="und")
    scene_candidates.add_argument("--taxonomy-prompt", required=True)
    scene_candidates.add_argument("--cache-format-version", default="scene-candidate-cache.v1")
    scene_candidates.add_argument(
        "--source-format",
        choices=["generic", "transnetv2_scenes", "autoshot_scenes"],
        default="generic",
    )
    scene_candidates.add_argument(
        "--frame-rate",
        type=float,
        default=None,
        help="Required for saved frame-index scenes unless fps/frame_rate is present in JSON",
    )
    scene_candidates.add_argument("--no-write", action="store_true")

    highlight = sub.add_parser("highlight-detection", help="Plan, run, or import local Lighthouse CG-DETR highlight detection")
    highlight.add_argument("bundle_dir")
    highlight.add_argument("--query", required=True)
    highlight.add_argument("--media-path", default="")
    highlight.add_argument("--checkpoint-path", default="")
    highlight.add_argument("--source-root", default="")
    highlight.add_argument("--predictions-json", default="")
    highlight.add_argument("--feature-name", choices=["clip", "clip_slowfast", "clip_slowfast_pann"], default="clip")
    highlight.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda", help="CUDA is required for real local-model execution; auto never falls back to CPU")
    highlight.add_argument("--execute", action="store_true")
    highlight.add_argument("--no-write", action="store_true")

    video_structure = sub.add_parser("video-structure", help="Build shot, semantic-scene, storyline, and optional highlight artifacts without whole-video VLM use")
    video_structure.add_argument("bundle_dir")
    video_structure.add_argument("--media-path", default="")
    video_structure.add_argument("--title", default="")
    video_structure.add_argument("--input-pack", default="")
    video_structure.add_argument("--no-shot-detection", action="store_true")
    video_structure.add_argument("--shot-detector", choices=["adaptive", "content"], default="adaptive")
    video_structure.add_argument("--shot-threshold", type=float, default=None)
    video_structure.add_argument("--shot-source-root", default="")
    video_structure.add_argument("--highlight-query", default="找出对内容理解或后续剪辑最重要的片段")
    video_structure.add_argument("--highlight-predictions-json", default="")
    video_structure.add_argument("--content-profile", choices=["course-v1", "filmed-v1"], default="course-v1")
    video_structure.add_argument("--shot-embeddings-json", default="")
    video_structure.add_argument("--story-evidence-json", default="")
    video_structure.add_argument("--local-story-route-id", default="")
    video_structure.add_argument("--no-write", action="store_true")

    creator_report = sub.add_parser("export-creator-report", help="Export a candidate creator report from existing Bundle evidence; no models/downloads")
    creator_report.add_argument("bundle_dir")
    creator_report.add_argument("--title", default="")
    creator_report.add_argument("--view", choices=["strict", "practical"], default="strict", help="Practical reference view preserves strict acceptance separately")
    creator_report.add_argument("--video-type", choices=["talking_head", "tutorial", "visual_story", "general"], default="talking_head", help="Explicit display type; automatic classification is deferred")
    creator_report.add_argument("--brief-json", default="", help="Creative brief with product, audience, goal, resources")
    creator_report.add_argument("--formats", nargs="+", choices=["md", "html", "docx", "pdf", "srt", "csv"], default=["md", "html"])
    creator_report.add_argument("--interactive-html", action="store_true",
                                help="Export an opt-in local-file player with timestamp navigation; static reading HTML remains the default")
    creator_report.add_argument("--output-dir", default="",
                                help="Explicit isolated report destination; source Bundle remains read-only")
    creator_report.add_argument("--ocr-results-json", default="",
                                help="Read-only cached OCR block projection, bound into this report's source fingerprint")
    creator_report.add_argument("--pandoc", default="", help="Operator-configured Pandoc executable for DOCX")
    creator_report.add_argument("--scriptbreak-script", default="", help="Reviewed upstream ScriptBreak MCP script; optional template execution")
    creator_report.add_argument("--scriptbreak-sha256", default="", help="Required content hash when executing ScriptBreak")
    creator_report.add_argument("--no-write", action="store_true")
    creator_status = sub.add_parser("creator-report-status", help="Check creator report evidence freshness and exported file hashes")
    creator_status.add_argument("bundle_dir")
    creator_status.add_argument("--view", choices=["strict", "practical"], default="strict")
    creator_status.add_argument("--output-dir", default="")

    shot_breakdown = sub.add_parser("shot-breakdown", help="Build candidate-only shot facts, style fingerprint, readiness, and imitation-script artifacts")
    shot_breakdown.add_argument("bundle_dir")
    shot_breakdown.add_argument("--title", default="")
    shot_breakdown.add_argument("--reference-analysis-json", default="", help="Optional saved local shot-analysis JSON; never executes the upstream runtime")
    shot_breakdown.add_argument("--no-write", action="store_true")

    decomposition = sub.add_parser("video-decomposition-report", help="Build a read-only evidence-bound five-layer decomposition report")
    decomposition.add_argument("bundle_dir")
    decomposition.add_argument("--title", default="")
    decomposition.add_argument("--no-write", action="store_true")

    decomposition_status = sub.add_parser("video-decomposition-status", help="Validate decomposition report hashes and input freshness")
    decomposition_status.add_argument("bundle_dir")
    decomposition_status.add_argument("--report-path", default="")
    decomposition_status.add_argument("--no-write", action="store_true")

    decomposition_compare = sub.add_parser("video-decomposition-compare", help="Compare two or more decomposition reports on one dimension matrix")
    decomposition_compare.add_argument("report_paths", nargs="+")
    decomposition_compare.add_argument("--output-dir", default="")
    decomposition_compare.add_argument("--title", default="")
    decomposition_compare.add_argument("--no-write", action="store_true")

    semantic_chapters = sub.add_parser("semantic-chapter-plan", help="Build semantic chapter boundaries from transcript, pause, tagger, OCR, and visual evidence")
    semantic_chapters.add_argument("bundle_dir")
    semantic_chapters.add_argument("--title", default="")
    semantic_chapters.add_argument("--chapter-mode", choices=["semantic", "fixed"], default="semantic")
    semantic_chapters.add_argument("--no-write", action="store_true")

    content_profile = sub.add_parser("content-profile", help="Apply a validated content profile without inferring review approval")
    content_profile.add_argument("bundle_dir")
    content_profile.add_argument("--profile", choices=SUPPORTED_PROFILES, required=True)
    content_profile.add_argument("--no-write", action="store_true")

    source_review_lineage = sub.add_parser("source-review-lineage", help="Discover exact same-source human review artifacts and optionally bind their immutable lineage")
    source_review_lineage.add_argument("bundle_dir")
    source_review_lineage.add_argument("--search-root", dest="search_roots", action="append", default=[])
    source_review_lineage.add_argument("--apply", action="store_true", help="Bind the best exact-source review lineage; never copies or overwrites transcript text")
    source_review_lineage.add_argument("--no-write", action="store_true")

    production_artifact_gate = sub.add_parser("production-artifact-gate", help="Evaluate the hard gate for formal transcript/summary artifacts")
    production_artifact_gate.add_argument("bundle_dir")
    production_artifact_gate.add_argument("--artifact-kind", default="smart_summary")
    production_artifact_gate.add_argument("--search-root", dest="search_roots", action="append", default=[])
    production_artifact_gate.add_argument("--no-discover-prior-reviews", action="store_true")
    production_artifact_gate.add_argument("--no-write", action="store_true")

    summary_chapters = sub.add_parser("build-smart-summary-chapters", help="Build chapter-level evidence and course map for smart-summary generation")
    summary_chapters.add_argument("bundle_dir")
    summary_chapters.add_argument("--title", default="")
    summary_chapters.add_argument("--target-chapters", type=int, default=8)
    summary_chapters.add_argument("--max-visual-items", type=int, default=120)
    summary_chapters.add_argument("--chapter-mode", choices=["semantic", "fixed"], default="semantic")
    summary_chapters.add_argument("--no-write", action="store_true")

    summary_section = sub.add_parser("smart-summary-section-workflow", help="Build section-level smart-summary rewrite and quality workflow")
    summary_section.add_argument("bundle_dir")
    summary_section.add_argument("--title", default="")
    summary_section.add_argument("--target-chapters", type=int, default=8)
    summary_section.add_argument("--no-write", action="store_true")

    summary_section_editor = sub.add_parser("smart-summary-section-editor", help="Build a static editor for smart-summary section revisions")
    summary_section_editor.add_argument("bundle_dir")
    summary_section_editor.add_argument("--no-write", action="store_true")
    summary_section_apply = sub.add_parser("smart-summary-section-apply", help="Install section-level smart-summary rewrites as a Codex summary candidate")
    summary_section_apply.add_argument("bundle_dir")
    summary_section_apply.add_argument("--input-json", default="")
    summary_section_apply.add_argument("--require-all-sections", action="store_true")
    summary_section_apply.add_argument("--no-write", action="store_true")

    summary_section_llm = sub.add_parser("run-smart-summary-section-llm-rewrite", help="Rewrite smart-summary one chapter at a time with an OpenAI-compatible text LLM")
    summary_section_llm.add_argument("bundle_dir")
    summary_section_llm.add_argument("--provider-config", default="", help="Inline JSON or JSON file with provider/base_url/model/api_key; API key is runtime-only")
    summary_section_llm.add_argument("--execute", action="store_true", help="Actually call the configured text LLM provider")
    summary_section_llm.add_argument("--business-authorization", default="", help="Active model business authorization JSON; required for remote Proxy execution")
    summary_section_llm.add_argument("--auto-from-profile", action="store_true", help="Auto-execute when quality profile allows data export and batch thresholds pass")
    summary_section_llm.add_argument("--quality-profile", default="quality")
    summary_section_llm.add_argument("--target-chapters", type=int, default=8)
    summary_section_llm.add_argument("--limit", type=int, default=0, help="Limit selected chapters for preview or small batches; 0 means all")
    summary_section_llm.add_argument("--section-ids", default="", help="Comma-separated section ids to rewrite, for example chapter-0005")
    summary_section_llm.add_argument("--only-needing-rewrite", action="store_true")
    summary_section_llm.add_argument("--max-prompt-chars", type=int, default=6000)
    summary_section_llm.add_argument("--max-tokens", type=int, default=1200)
    summary_section_llm.add_argument("--min-section-chars", type=int, default=120, help="Reject section LLM outputs below this compact character count")
    summary_section_llm.add_argument("--temperature", type=float, default=0)
    summary_section_llm.add_argument("--no-install", action="store_true", help="Only write section LLM revisions; do not install aggregate summary")
    summary_section_llm.add_argument("--no-require-all-sections", action="store_true", help="Allow installing a partial aggregate when using --limit")
    summary_section_llm.add_argument("--no-write", action="store_true")

    global_reduce = sub.add_parser("smart-summary-global-reduce", help="Reduce complete semantic chapter Map outputs into one final summary")
    global_reduce.add_argument("bundle_dir")
    global_reduce.add_argument("--provider-config", default="")
    global_reduce.add_argument("--execute", action="store_true")
    global_reduce.add_argument("--reuse-candidate", action="store_true", help="Reuse the persisted Reduce candidate without another model call")
    global_reduce.add_argument(
        "--recover-execution-report",
        default="",
        help="Validate and install a completed local Broker execution report without another model call",
    )
    global_reduce.add_argument(
        "--business-authorization",
        default="",
        help="Active parent business authorization used to derive one exact child consent and Broker reservation",
    )
    global_reduce.add_argument("--max-input-chars", type=int, default=60000)
    global_reduce.add_argument("--max-tokens", type=int, default=5000)
    global_reduce.add_argument("--temperature", type=float, default=0)
    global_reduce.add_argument("--no-install", action="store_true")
    global_reduce.add_argument("--no-write", action="store_true")

    summary_consistency = sub.add_parser("summary-consistency-check", help="Check chapter-to-summary entity, number, and event consistency")
    summary_consistency.add_argument("bundle_dir")
    summary_consistency.add_argument("--summary-path", default="")
    summary_consistency.add_argument("--no-write", action="store_true")

    smart_quality = sub.add_parser("smart-summary-quality-check", help="Check whether smart-summary.md is a final readable summary rather than a draft")
    smart_quality.add_argument("bundle_dir")
    smart_quality.add_argument("--summary-path", default="")
    smart_quality.add_argument("--require-codex", action="store_true")
    smart_quality.add_argument("--no-write", action="store_true")

    content_asset = sub.add_parser("content-asset-status", help="Report whether exported content material cards are ready for review-only downstream use")
    content_asset.add_argument("bundle_dir")
    content_asset.add_argument("--no-write", action="store_true")

    batch_content = sub.add_parser("batch-content-asset-status", help="Summarize content material card readiness for one or more bundles")
    batch_content.add_argument("batch_input")
    batch_content.add_argument("--output-dir", default="")
    batch_content.add_argument("--no-write", action="store_true")

    handoff_pack = sub.add_parser("content-handoff-pack", help="Generate a review-only handoff pack from ready content material cards")
    handoff_pack.add_argument("batch_input")
    handoff_pack.add_argument("--output-dir", default="")
    handoff_pack.add_argument("--no-write", action="store_true")

    quality_console = sub.add_parser("export-quality-console", help="Write the static Phase 17 transcript and summary quality console")
    quality_console.add_argument("bundle_dir")
    quality_console.add_argument("--no-write", action="store_true")

    task_console = sub.add_parser("export-task-console", help="Write a static task console for one WebUI bundle")
    task_console.add_argument("bundle_dir")
    task_console.add_argument("--no-refresh", action="store_true", help="Do not refresh status reports before rendering the console")
    task_console.add_argument("--no-write", action="store_true")

    subqueue_plan = sub.add_parser("subqueue-action-plan", help="Write or preview the copyable subqueue action plan for one WebUI bundle")
    subqueue_plan.add_argument("bundle_dir")
    subqueue_plan.add_argument("--no-refresh", action="store_true", help="Do not refresh status reports before reading the action plan")
    subqueue_plan.add_argument("--no-write", action="store_true")

    workbench = sub.add_parser("export-video-workbench", help="Write a unified static video workbench for task/review/transcript/summary surfaces")
    workbench.add_argument("bundle_dir")
    workbench.add_argument("--no-write", action="store_true")

    edit_pack = sub.add_parser("video-edit-review-pack", help="Build a local-only storyboard, boundary, validation, and preference evidence handoff for the existing editing pipeline")
    edit_pack.add_argument("bundle_dir")
    edit_pack.add_argument("--decisions-json", default="")
    edit_pack.add_argument("--tokens-json", default="")
    edit_pack.add_argument("--silence-json", default="")
    edit_pack.add_argument("--delete-segments-json", default="")
    edit_pack.add_argument("--cut-segments-json", default="")
    edit_pack.add_argument("--ai-baseline-json", default="")
    edit_pack.add_argument("--media-path", default="")
    edit_pack.add_argument("--reclaim-silence", action="store_true", help="Locally decode audio energy with ffmpeg and recover silence swallowed by ASR timestamps")
    edit_pack.add_argument("--human-confirmed-diff", action="store_true", help="Mark this one baseline/final difference as human-confirmed evidence; never auto-promotes preferences")
    edit_pack.add_argument("--no-write", action="store_true")

    edit_pack.add_argument("--review-attestation-path", default="", help="Exact VKP review attestation; current video-edit-handoff attestation is used when omitted")
    refresh_html = sub.add_parser("refresh-review-html", help="Refresh review.html from the current bundle manifest/timeline without rebuilding the bundle")
    refresh_html.add_argument("bundle_dir")
    refresh_html.add_argument("--no-write", action="store_true")

    status = sub.add_parser("bundle-status-report", help="Write a compact status report for a review bundle")
    status.add_argument("bundle_dir")
    status.add_argument("--no-refresh", action="store_true")

    acceptance_check_parser = sub.add_parser("acceptance-check", help="Write a unified acceptance status for a review bundle")
    acceptance_check_parser.add_argument("bundle_dir")
    acceptance_check_parser.add_argument("--no-refresh", action="store_true")
    acceptance_check_parser.add_argument("--no-write", action="store_true")

    controlled = sub.add_parser("controlled-execution-check", help="Check whether the vision execution control chain is ready")
    controlled.add_argument("bundle_dir")
    controlled.add_argument("--no-refresh", action="store_true")
    controlled.add_argument("--no-write", action="store_true")

    smoke = sub.add_parser("controlled-execution-smoke", help="Run a one-item controlled vision execution smoke with optional restore")
    smoke.add_argument("bundle_dir")
    smoke.add_argument("--execute", action="store_true")
    smoke.add_argument("--restore-after", action="store_true")
    smoke.add_argument("--provider-config", default="fixture", help="Provider name, inline JSON, or JSON file. Defaults to fixture.")
    smoke.add_argument("--kind", choices=["auto", "semantic", "temporal"], default="auto")
    smoke.add_argument("--index", type=int)
    smoke.add_argument("--frame-count", type=int, default=8)
    smoke.add_argument("--no-write", action="store_true")

    next_action = sub.add_parser("bundle-next-action", help="Show the next safe action for a review bundle")
    next_action.add_argument("bundle_dir")
    next_action.add_argument("--no-refresh", action="store_true")

    lifecycle_doctor = sub.add_parser("media-artifact-doctor", help="Validate the read-only VKP media lifecycle contracts and optional Bundle scope")
    _add_media_artifact_doctor_args(lifecycle_doctor)

    lifecycle_preflight = sub.add_parser("media-artifact-preflight", help="Validate output-aware disk headroom before a declared large task, or explicitly declare a non-large operation")
    _add_media_artifact_doctor_args(lifecycle_preflight, preflight=True)

    lifecycle_scan = sub.add_parser("media-artifact-scan", help="Scan only explicit bounded artifact roots and write a plan-only manifest")
    lifecycle_scan.add_argument("bundle_dir")
    lifecycle_scan.add_argument("--scope-path", required=True)
    lifecycle_scan.add_argument("--output-path", required=True)
    lifecycle_scan.add_argument("--checkpoint-dir", required=True)
    lifecycle_scan.add_argument("--max-depth", type=int, default=4)
    lifecycle_scan.add_argument("--max-items", type=int, default=1000)
    lifecycle_scan.add_argument("--core-root", default="")

    lifecycle_explain = sub.add_parser("media-artifact-explain", help="Explain project and core gates for one manifest artifact without authorizing cleanup")
    lifecycle_explain.add_argument("manifest_path")
    lifecycle_explain.add_argument("artifact_id")
    lifecycle_explain.add_argument("--core-root", default="")

    lifecycle_plan = sub.add_parser("media-artifact-plan-gc", help="Write a verified plan-only GC candidate plan; never applies it")
    lifecycle_plan.add_argument("manifest_path")
    lifecycle_plan.add_argument("--bundle-dir", required=True)
    lifecycle_plan.add_argument("--quarantine-root", required=True)
    lifecycle_plan.add_argument("--output-path", required=True)
    lifecycle_plan.add_argument("--core-root", default="")

    lifecycle_verify = sub.add_parser("media-artifact-verify-plan", help="Revalidate plan, paths, identities, hashes and hardlinks without mutation")
    lifecycle_verify.add_argument("plan_path")
    lifecycle_verify.add_argument("manifest_path")
    lifecycle_verify.add_argument("--bundle-dir", required=True)
    lifecycle_verify.add_argument("--quarantine-root", required=True)
    lifecycle_verify.add_argument("--core-root", default="")

    lifecycle_restore = sub.add_parser("media-artifact-restore", help="Restore only a marked synthetic fixture through the shared core")
    lifecycle_restore.add_argument("plan_path")
    lifecycle_restore.add_argument("receipt_path")
    lifecycle_restore.add_argument("--bundle-dir", required=True)
    lifecycle_restore.add_argument("--quarantine-root", required=True)
    lifecycle_restore.add_argument("--output-path", required=True)
    lifecycle_restore.add_argument("--execute-synthetic", action="store_true")
    lifecycle_restore.add_argument("--core-root", default="")

    autopilot_start_parser = sub.add_parser("autopilot-start", help="Create a deterministic small-model Autopilot job")
    _add_autopilot_start_args(autopilot_start_parser)

    autopilot_next_parser = sub.add_parser("autopilot-next", help="Advance deterministic work and lease at most one model task")
    autopilot_next_parser.add_argument("job_id")
    autopilot_next_parser.add_argument(
        "--retry-provider",
        action="store_true",
        help="Explicitly retry the same hash-bound loopback provider task after fixing LM Studio",
    )

    autopilot_submit_parser = sub.add_parser("autopilot-submit", help="Atomically validate and import one leased Autopilot result")
    autopilot_submit_parser.add_argument("job_id")
    autopilot_submit_parser.add_argument("--input-json", required=True)

    autopilot_status_parser = sub.add_parser("autopilot-status", help="Show Autopilot state, route, budget, gaps, and current lease")
    autopilot_status_parser.add_argument("job_id")

    autopilot_run_parser = sub.add_parser("autopilot-run", help="Run deterministic Autopilot steps until a model task or terminal state")
    _add_autopilot_start_args(autopilot_run_parser)
    autopilot_run_parser.add_argument("--resume", default="", help="Resume an existing job id instead of creating a new job")
    autopilot_run_parser.add_argument(
        "--retry-provider",
        action="store_true",
        help="When resuming, explicitly retry a waiting loopback provider task",
    )

    advance = sub.add_parser("bundle-advance", help="Run or preview one safe bundle advance step")
    _add_bundle_advance_args(advance)

    advance_queue = sub.add_parser("bundle-advance-queue", help="Run or preview safe bundle advance steps until blocked or capped")
    _add_bundle_advance_args(advance_queue)
    advance_queue.add_argument("--max-steps", type=int, default=4)

    advance_log = sub.add_parser("bundle-advance-log", help="Render the persisted bundle advance history")
    advance_log.add_argument("bundle_dir")

    vision_log = sub.add_parser("vision-analysis-run-log", help="Read persisted vision execution audits and timeline diffs")
    vision_log.add_argument("bundle_dir")

    vision_restore = sub.add_parser("vision-analysis-restore-plan", help="Write a human-review restore plan from a vision execution audit run")
    vision_restore.add_argument("bundle_dir")
    vision_restore.add_argument("--run-id", default="")
    vision_restore.add_argument("--no-write", action="store_true")

    vision_apply_restore = sub.add_parser("vision-analysis-apply-restore", help="Dry-run or apply a reviewed vision restore plan")
    vision_apply_restore.add_argument("bundle_dir")
    vision_apply_restore.add_argument("--plan-json")
    vision_apply_restore.add_argument("--execute", action="store_true")
    vision_apply_restore.add_argument("--confirm-run-id", default="")
    return parser


def _add_bundle_advance_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("bundle_dir")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--refresh-outputs", action="store_true")
    parser.add_argument("--vault", default="")
    parser.add_argument("--folder", default="00_Inbox/AI/课程视频知识包")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--ocr-input-json")
    parser.add_argument("--ocr-language", default="chi_sim")
    parser.add_argument("--captiocr-root")
    parser.add_argument("--visual-structure-input-json")
    parser.add_argument("--provider-config", help="Inline JSON or JSON file for vision provider execution")
    parser.add_argument("--multimodal-limit", type=int)
    parser.add_argument("--temporal-limit", type=int)
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--confirm-vision-calls", type=int)
    parser.add_argument("--confirm-vision-indexes", default="")


def _add_media_artifact_doctor_args(parser: argparse.ArgumentParser, *, preflight: bool = False) -> None:
    parser.add_argument("bundle_dir", **({} if preflight else {"nargs": "?", "default": ""}))
    parser.add_argument("--scope-path", default="")
    parser.add_argument("--core-root", default="")
    parser.add_argument("--self-media-root", default="")
    if preflight:
        disk_mode = parser.add_mutually_exclusive_group(required=True)
        disk_mode.add_argument(
            "--estimated-output-bytes",
            type=int,
            help="Positive estimated output bytes; VKP reserves 1.25x this value plus the shared hard reserve",
        )
        disk_mode.add_argument(
            "--non-large",
            action="store_true",
            help="Explicitly declare that this operation is not a large-output task",
        )


def _add_autopilot_start_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("bundle_or_batch")
    parser.add_argument("--runner", choices=["agent-session", "background"], default="agent-session")
    parser.add_argument(
        "--execution-route",
        choices=["auto", "hybrid", "local-only", "online-only"],
        default="auto",
    )
    parser.add_argument(
        "--no-online-derived-artifacts",
        action="store_true",
        help="Disallow online use even for bounded, Hash-bound Bundle-derived task artifacts.",
    )


def _optional_bool_arg(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return str(value).strip().lower() == "true"


def _json_arg(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    path = Path(value).expanduser()
    data = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("JSON argument must be an object")
    return data


def _provider_config_arg(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    text = str(value).strip()
    path = Path(text).expanduser()
    if path.exists() or text.startswith("{"):
        return _json_arg(text)
    return {"provider": text}


def _csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _int_csv_arg(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _int_csv_and_repeat_arg(
    csv_value: str | None, repeated: list[int] | None
) -> list[int] | None:
    values = (_int_csv_arg(csv_value) or []) + (repeated or [])
    unique = list(dict.fromkeys(int(value) for value in values if int(value) > 0))
    return unique or None


def _artifact_specs(values: list[str] | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for value in values or []:
        role, separator, path = str(value or "").partition("=")
        if not separator or not role.strip() or not path.strip():
            raise ValueError("--artifact must use role=path")
        rows.append({"role": role.strip(), "path": path.strip()})
    if not rows:
        raise ValueError("at least one --artifact role=path is required")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
