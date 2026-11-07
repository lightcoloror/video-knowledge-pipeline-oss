from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

from .asr_consensus import build_asr_consensus
from .config import processing_profile
from .local_targeted_asr_execution import run_local_targeted_asr_evidence
from .local_targeted_asr_plan import build_local_targeted_asr_plan
from .evidence_conflict_index import build_evidence_conflict_index
from .entity_lexicon import build_entity_lexicon
from .knowledge_note_export import export_knowledge_note
from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .storage import bundle_write_lock, read_json, write_json
from .transcript_postprocess import postprocess_asr_transcript
from .transcript_agent_readable import run_agent_readable_transcript_rewrite
from .transcript_quality_gate import run_transcript_quality_gate
from .transcript_readable_llm import run_readable_transcript_llm_polish
from .transcript_semantic_correction import (
    build_transcript_semantic_correction_codex_draft,
    build_transcript_semantic_correction_llm_draft,
    build_transcript_semantic_correction_pack,
    transcript_semantic_correction_status,
    transcript_semantic_correction_closure,
    validate_transcript_semantic_correction,
)
from .transcript_source_arbitration import arbitrate_transcript_sources
from .text_llm_gateway import resolve_text_provider_config
from .vision_api import provider_requires_api_key

SCHEMA = "video_knowledge_pipeline.transcript_evidence_correction_pipeline.v1"


def run_transcript_evidence_correction_pipeline(
    bundle_dir: str | Path,
    *,
    platform_subtitle: str | Path | None = None,
    subtitle: str | Path | None = None,
    asr_json: str | Path | None = None,
    secondary_asr_json: str | Path | None = None,
    additional_secondary_asr_json: list[str | Path] | None = None,
    consensus_agreement_threshold: float = 0.86,
    execute_consensus_clips: bool = False,
    media_path: str | Path | None = None,
    asr_mode: str = "",
    execute_local_asr: bool = False,
    speech_activity_json: str | Path | None = None,
    diarization_json: str | Path | None = None,
    execute_local_targeted_asr: bool = False,
    local_targeted_asr_preset: str = "qwen3-asr-0.6b",
    local_targeted_asr_model: str | None = None,
    local_targeted_asr_timeout_seconds: int = 900,
    local_targeted_asr_allow_cpu: bool = False,
    glossary_json: str | Path | None = None,
    provider_config: dict[str, Any] | None = None,
    quality_profile: str = "",
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
    postprocess_set_corrected: bool = True,
    run_source_arbitration: bool = True,
    source_arbitration_promote: bool = True,
    source_min_confidence: float = 0.72,
    semantic_min_confidence: float = 0.88,
    semantic_limit: int = 80,
    materialise_corrected_alias: bool = True,
    run_agent_readable_rewrite: bool = True,
    refresh_exports: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Run the preferred ASR/subtitle/visual-evidence semantic correction chain.

    The chain is deliberately preview-first for online LLM steps:
    local source arbitration and evidence pack generation can run immediately,
    while cloud/text LLM execution and final auto-apply require explicit flags.
    """

    root = Path(bundle_dir).expanduser().resolve()
    if use_codex_substitute is not None:
        use_agent_substitute = bool(use_codex_substitute)
    substitute_agent_name = _normalise_agent_name(agent_name, legacy_codex=bool(use_codex_substitute))
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    dual_result: dict[str, Any] | None = None
    selected_mode = asr_mode or ("important" if quality_profile == "quality" else "")
    if execute_local_asr and not selected_mode:
        selected_mode = "important"
    if selected_mode:
        from .dual_asr_workflow import run_dual_asr_workflow
        profile = processing_profile(quality_profile or "quality")
        dual_result = run_dual_asr_workflow(
            root, media_path=media_path, primary_transcript=_primary_asr_for_consensus(root, asr_json),
            secondary_transcript=secondary_asr_json, mode=selected_mode,
            secondary_preset=profile.get("secondary_asr_preset", "qwen3-asr-1.7b"),
            execute=execute_local_asr, timeout_seconds=local_targeted_asr_timeout_seconds,
            allow_cpu=local_targeted_asr_allow_cpu, speech_activity_json=speech_activity_json,
            diarization_json=diarization_json, write=write,
        )
        if not dual_result["ok"]:
            # Do not polish/promote a draft and label the requested quality mode complete.
            pending = {"schema": SCHEMA, "bundle_dir": str(root), "status": dual_result["status"],
                       "ok": False, "dual_asr_execution": dual_result,
                       "next_actions": ["Resume the saved local dual-ASR command before downstream correction."],
                       "canonical_transcript_modified": False}
            if write:
                with bundle_write_lock(root, operation="transcript_evidence_correction_pipeline", timeout_seconds=1):
                    write_json(root / "transcript-evidence-correction-pipeline.json", pending)
            return pending
        asr_json = dual_result["primary_transcript"]
        secondary_asr_json = dual_result["secondary_transcript"]
        speech_activity_json = dual_result.get("speech_activity_json") or speech_activity_json
    secondary_asr_sources = [
        value
        for value in [secondary_asr_json, *(additional_secondary_asr_json or [])]
        if value
    ]
    registered_secondary_asr_sources = _register_secondary_asr_sources(root, secondary_asr_sources) if write and secondary_asr_sources else []


    steps: list[dict[str, Any]] = []
    consensus_result: dict[str, Any] | None = None
    postprocess_result: dict[str, Any] | None = None
    source_result: dict[str, Any] | None = None
    entity_lexicon_result: dict[str, Any] | None = None
    pack_result: dict[str, Any] | None = None
    local_targeted_asr_plan_result: dict[str, Any] | None = None
    local_targeted_asr_execution_result: dict[str, Any] | None = None
    conflict_index_result: dict[str, Any] | None = None
    conflict_llm_pack_result: dict[str, Any] | None = None
    readable_llm_result: dict[str, Any] | None = None
    llm_result: dict[str, Any] | None = None
    codex_result: dict[str, Any] | None = None
    validation_result: dict[str, Any] | None = None
    closure_result: dict[str, Any] | None = None
    export_result: dict[str, Any] | None = None
    alias_result: dict[str, Any] | None = None
    agent_readable_result: dict[str, Any] | None = None
    quality_gate_result: dict[str, Any] | None = None

    if secondary_asr_json:
        primary_consensus_input = _primary_asr_for_consensus(root, asr_json)
        if primary_consensus_input:
            consensus_result = build_asr_consensus(
                root,
                primary_transcript=primary_consensus_input,
                secondary_transcript=secondary_asr_json,
                media_path=media_path,
                speech_activity_json=speech_activity_json,
                diarization_json=diarization_json,
                secondary_scope="targeted" if selected_mode == "daily" else "full",
                agreement_threshold=consensus_agreement_threshold,
                execute_clips=execute_consensus_clips,
                write=write,
            )
        else:
            consensus_result = {
                "status": "blocked_primary_asr_missing",
                "ok": False,
                "conflict_count": 0,
                "error": "primary ASR transcript not found for consensus",
            }
        steps.append(_step("asr_consensus", consensus_result))

    if run_postprocess:
        try:
            postprocess_result = postprocess_asr_transcript(root, input_path=asr_json, set_corrected=postprocess_set_corrected, write=write)
        except FileNotFoundError as exc:
            postprocess_result = {"status": "skipped_no_asr_input", "ok": False, "error": str(exc)}
        steps.append(_step("asr_postprocess", postprocess_result))

    entity_lexicon_result = build_entity_lexicon(
        root,
        base_lexicon_json=glossary_json,
        write=write,
    )
    steps.append(_step("entity_lexicon", entity_lexicon_result))

    if run_source_arbitration:
        arbitration_asr_json = asr_json
        if arbitration_asr_json is None and isinstance(postprocess_result, dict):
            artifacts = postprocess_result.get("artifacts") if isinstance(postprocess_result.get("artifacts"), dict) else {}
            postprocessed_json = artifacts.get("json")
            if postprocessed_json:
                arbitration_asr_json = postprocessed_json
        source_result = arbitrate_transcript_sources(
            root,
            platform_subtitle=platform_subtitle,
            subtitle=subtitle,
            asr_json=arbitration_asr_json,
            glossary_json=glossary_json,
            min_confidence=source_min_confidence,
            promote=source_arbitration_promote,
            write=write,
        )
        steps.append(_step("source_arbitration", source_result))

    pack_result = build_transcript_semantic_correction_pack(root, limit=0, write=write)
    steps.append(_step("semantic_correction_pack", pack_result))

    # Candidate discovery must create a bounded, independent local evidence
    # queue for factual risks. A single ASR hypothesis is not sufficient
    # evidence for names, numbers, or terms.
    local_targeted_asr_plan_result = build_local_targeted_asr_plan(
        root,
        input_pack=pack_result.get("pack_json") or root / "transcript-semantic-correction-pack.json",
        pack_data=pack_result if not write else None,
        write=write,
    )
    steps.append(_step("local_targeted_asr_plan", local_targeted_asr_plan_result))

    if execute_local_targeted_asr:
        if not write:
            raise ValueError("execute_local_targeted_asr=True requires write=True")
        local_targeted_asr_execution_result = run_local_targeted_asr_evidence(
            root,
            media_path=media_path,
            input_plan=root / "local-targeted-asr-plan.json",
            preset=local_targeted_asr_preset,
            language="zh",
            model=local_targeted_asr_model or None,
            timeout_seconds=local_targeted_asr_timeout_seconds,
            execute=True,
            allow_cpu=local_targeted_asr_allow_cpu,
            write=True,
        )
        steps.append(_step("local_targeted_asr_execution", local_targeted_asr_execution_result))
        if local_targeted_asr_execution_result.get("status") == "completed":
            # Candidate-only evidence is registered as a secondary ASR source.
            # Rebuild semantic views without promoting it into the canonical text.
            pack_result = build_transcript_semantic_correction_pack(root, limit=0, write=write)
            steps.append(_step("semantic_correction_pack_after_local_targeted_asr", pack_result))
            local_targeted_asr_plan_result = build_local_targeted_asr_plan(
                root,
                input_pack=pack_result.get("pack_json") or root / "transcript-semantic-correction-pack.json",
                write=write,
            )
            steps.append(_step("local_targeted_asr_plan_after_execution", local_targeted_asr_plan_result))

    conflict_index_result = build_evidence_conflict_index(
        root,
        input_json=pack_result.get("pack_json") or root / "transcript-semantic-correction-pack.json",
        pack_data=pack_result if not write else None,
        limit=semantic_limit,
        write=write,
    )
    steps.append(_step("evidence_conflict_index", conflict_index_result))

    candidate_count = int((conflict_index_result or {}).get("llm_arbitration_count") or 0)
    semantic_candidate_count = int((pack_result or {}).get("candidate_count") or 0)
    targeted_window_count = int(((local_targeted_asr_plan_result or {}).get("retry_plan") or {}).get("window_count") or 0)
    profile_execution, runtime_provider_config = _quality_profile_execution(
        root,
        profile_name=quality_profile,
        provider_config=provider_config,
        candidate_count=candidate_count,
        readable_max_prompt_chars=readable_max_prompt_chars,
        explicit_execute=bool(execute_llm or execute_readable_llm),
    )
    if quality_profile:
        use_agent_substitute = False
        if profile_execution.get("auto_execute"):
            provider_config = runtime_provider_config
            execute_llm = True
            execute_readable_llm = True
            promote_readable_llm = True
            auto_apply_high_confidence = True
    conflict_llm_pack_result = _write_conflict_llm_pack(root, pack_result or {}, conflict_index_result or {}, write=write)
    steps.append(_step("evidence_conflict_llm_pack", conflict_llm_pack_result))
    llm_input_json = conflict_llm_pack_result.get("pack_json") or pack_result.get("pack_json") or str(root / "transcript-semantic-correction-pack.json")
    if candidate_count > 0:
        if use_agent_substitute and not execute_llm:
            codex_result = build_transcript_semantic_correction_codex_draft(
                root,
                input_json=llm_input_json,
                min_confidence=semantic_min_confidence,
                write=write,
            )
            steps.append(_step("codex_semantic_review", codex_result))
            result_json = str(codex_result.get("result_markdown") or root / "transcript-semantic-correction-result.codex.md")
            if codex_result.get("status") == "draft_ready":
                validation_result = validate_transcript_semantic_correction(
                    root,
                    input_json=result_json,
                    min_confidence=semantic_min_confidence,
                    write=write,
                )
                steps.append(_step("semantic_validation", validation_result))
                closure_result = transcript_semantic_correction_closure(
                    root,
                    input_json=result_json,
                    min_confidence=semantic_min_confidence,
                    auto_apply=True,
                    refresh_exports=refresh_exports,
                    write=write,
                )
                steps.append(_step("semantic_closure", closure_result))
        else:
            llm_result = build_transcript_semantic_correction_llm_draft(
                root,
                input_json=llm_input_json,
                provider_config=provider_config,
                execute=execute_llm,
                limit=semantic_limit,
                min_confidence=semantic_min_confidence,
                write=write,
            )
            steps.append(_step("llm_semantic_review", llm_result))
            if execute_llm and llm_result.get("status") == "executed":
                result_json = str(llm_result.get("result_json") or root / "transcript-semantic-correction-result.llm.json")
                validation_result = validate_transcript_semantic_correction(
                    root,
                    input_json=result_json,
                    min_confidence=semantic_min_confidence,
                    write=write,
                )
                steps.append(_step("semantic_validation", validation_result))
                if auto_apply_high_confidence:
                    closure_result = transcript_semantic_correction_closure(
                        root,
                        input_json=result_json,
                        min_confidence=semantic_min_confidence,
                        auto_apply=True,
                        refresh_exports=refresh_exports,
                        write=write,
                    )
                    steps.append(_step("semantic_closure", closure_result))

    if write:
        if materialise_corrected_alias:
            alias_result = _materialise_corrected_transcript_alias(root)
        else:
            alias_result = {"status": "skipped", "ok": True, "reason": "disabled_by_caller"}
        steps.append(_step("corrected_transcript_alias", alias_result))
        if run_readable_llm:
            readable_llm_result = run_readable_transcript_llm_polish(
                root,
                provider_config=provider_config,
                execute=execute_readable_llm,
                agent_substitute=bool(use_agent_substitute and not execute_readable_llm),
                agent_name=substitute_agent_name,
                codex_substitute=bool(use_codex_substitute and use_agent_substitute and not execute_readable_llm),
                promote=bool(promote_readable_llm or (use_agent_substitute and not execute_readable_llm)),
                max_segments_per_batch=readable_max_segments_per_batch,
                max_prompt_chars=readable_max_prompt_chars,
                max_tokens=readable_max_tokens,
                write=True,
            )
            steps.append(_step("readable_transcript_llm_polish", readable_llm_result))
        real_readable_promoted = bool(readable_llm_result and readable_llm_result.get("status") in {"executed", "imported"} and readable_llm_result.get("promote"))
        if run_agent_readable_rewrite and not real_readable_promoted:
            agent_readable_result = run_agent_readable_transcript_rewrite(root, agent_name=substitute_agent_name, promote=True, write=True)
        else:
            agent_readable_result = {"status": "skipped", "ok": True, "reason": "disabled_by_caller"}
        steps.append(_step("agent_readable_transcript_rewrite", agent_readable_result))
        quality_gate_result = run_transcript_quality_gate(root, write=True)
        steps.append(_step("transcript_quality_gate", quality_gate_result))
        if refresh_exports and _should_refresh_exports(closure_result, source_result, readable_llm_result):
            export_result = export_knowledge_note(root, run_transcript_evidence_check=False, write=True)
            steps.append(_step("export_knowledge_note", export_result))

    semantic_status_snapshot = transcript_semantic_correction_status(root, write=False)
    candidate_discovery_status = str(
        semantic_status_snapshot.get("candidate_discovery_status") or "not_planned"
    )
    candidate_statistics = _candidate_statistics(
        pack_result or {}, conflict_index_result or {}, validation_result or {}
    )
    corrected_transcript_exists = (root / "corrected-transcript.json").is_file()
    pipeline_status = _pipeline_status(
        candidate_count=candidate_count,
        semantic_candidate_count=semantic_candidate_count,
        targeted_window_count=targeted_window_count,
        execute_llm=execute_llm,
        use_agent_substitute=use_agent_substitute,
        agent_name=substitute_agent_name,
        use_codex_substitute=bool(use_codex_substitute) if use_codex_substitute is not None else bool(use_agent_substitute and substitute_agent_name == "codex"),
        codex_result=codex_result,
        llm_result=llm_result,
        validation_result=validation_result,
        closure_result=closure_result,
        auto_apply_high_confidence=auto_apply_high_confidence,
        candidate_discovery_status=candidate_discovery_status,
        candidate_statistics=candidate_statistics,
        corrected_transcript_exists=corrected_transcript_exists,
        quality_gate_result=quality_gate_result,
    )
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "quality_profile": quality_profile or "legacy",
        "dual_asr_execution": dual_result or {
            "status": "dual_inputs_compared" if consensus_result and consensus_result.get("ok") else "single_model_only",
            "dual_full_transcription_complete": False,
            "provenance": "legacy_inputs_not_execution_attested",
        },
        "secondary_asr_json": str(secondary_asr_json or "" ),
        "consensus_agreement_threshold": float(consensus_agreement_threshold),
        "additional_secondary_asr_json": [str(value) for value in additional_secondary_asr_json or []],
        "registered_secondary_asr_sources": registered_secondary_asr_sources,
        "execute_consensus_clips": bool(execute_consensus_clips),
        "media_path": str(media_path or ""),
        "execute_local_targeted_asr": bool(execute_local_targeted_asr),
        "local_targeted_asr_preset": local_targeted_asr_preset,
        "local_targeted_asr_model": str(local_targeted_asr_model or ""),
        "local_targeted_asr_timeout_seconds": int(local_targeted_asr_timeout_seconds or 0),
        "local_targeted_asr_allow_cpu": bool(local_targeted_asr_allow_cpu),
        "quality_profile_execution": profile_execution,
        "status": pipeline_status,
        "ok": _pipeline_ok(
            pipeline_status,
            candidate_discovery_status=candidate_discovery_status,
            candidate_statistics=candidate_statistics,
            corrected_transcript_exists=corrected_transcript_exists,
            quality_gate_result=quality_gate_result,
        ),
        "candidate_discovery_status": candidate_discovery_status,
        "candidate_statistics": candidate_statistics,
        "discovered_candidate_count": candidate_statistics["discovered_candidate_count"],
        "validated_candidate_count": candidate_statistics["validated_candidate_count"],
        "unvalidated_candidate_count": candidate_statistics["unvalidated_candidate_count"],
        "deferred_candidate_count": candidate_statistics["deferred_candidate_count"],
        "review_required_candidate_count": candidate_statistics["review_required_candidate_count"],
        "review_required_count": candidate_statistics["review_required_candidate_count"],
        "validation_review_required_count": candidate_statistics["validation_review_required_count"],
        "corrected_transcript_exists": corrected_transcript_exists,
        "asr_consensus": _summary(consensus_result),
        "asr_postprocess": _summary(postprocess_result),
        "source_arbitration": _summary(source_result),
        "entity_lexicon": _summary(entity_lexicon_result),
        "readable_llm_polish": _summary(readable_llm_result),
        "semantic_pack": _summary(pack_result),
        "local_targeted_asr_plan": _summary(local_targeted_asr_plan_result),
        "local_targeted_asr_execution": _summary(local_targeted_asr_execution_result),
        "evidence_conflict_index": _summary(conflict_index_result),
        "evidence_conflict_llm_pack": _summary(conflict_llm_pack_result),
        "llm_review": _summary(llm_result),
        "agent_review": _summary(codex_result),
        "codex_review": _summary(codex_result),
        "validation": _summary(validation_result),
        "closure": _summary(closure_result),
        "corrected_transcript_alias": alias_result or {},
        "agent_readable_transcript_rewrite": _summary(agent_readable_result),
        "transcript_quality_gate": _summary(quality_gate_result),
        "export": _summary(export_result),
        "steps": steps,
        "artifacts": {
            "pipeline_json": str(root / "transcript-evidence-correction-pipeline.json"),
            "pipeline_markdown": str(root / "transcript-evidence-correction-pipeline.md"),
            "asr_consensus_json": str(root / "asr-consensus.json"),
            "postprocessed_transcript_json": str(root / "postprocessed-transcript.json"),
            "llm_readable_transcript_json": str(root / "llm-readable-transcript.json"),
            "evidence_conflict_index_json": str(root / "evidence-conflict-index.json"),
            "evidence_conflict_llm_pack_json": str(root / "evidence-conflict-llm-pack.json"),
            "source_arbitrated_transcript_json": str(root / "source-arbitrated-transcript.json"),
            "entity_lexicon_json": str(root / "entity-lexicon.json"),
            "entity_hotwords": str(root / "entity-hotwords.txt"),
            "local_targeted_asr_plan_json": str(root / "local-targeted-asr-plan.json"),
            "local_targeted_asr_plan_markdown": str(root / "local-targeted-asr-plan.md"),
            "local_targeted_asr_execution_json": str(root / "local-targeted-asr-execution.json"),
            "local_targeted_asr_evidence_json": str(root / "local-targeted-asr-evidence.json"),
            "corrected_transcript_json": str(root / "corrected-transcript.json"),
            "agent_readable_transcript_json": str(root / "agent-readable-transcript.json"),
            "transcript_quality_gate_json": str(root / "transcript-quality-gate.json"),
            "full_transcript": str(root / "exports" / "full-transcript.md"),
            "smart_summary": str(root / "exports" / "smart-summary.md"),
            "mcp_args": str(root / "mcp-transcript-evidence-correction-pipeline.args.json"),
        },
        "next_actions": _next_actions(
            candidate_count=candidate_count,
            semantic_candidate_count=semantic_candidate_count,
            targeted_window_count=targeted_window_count,
            execute_llm=execute_llm,
            use_agent_substitute=use_agent_substitute,
            agent_name=substitute_agent_name,
            use_codex_substitute=bool(use_codex_substitute) if use_codex_substitute is not None else bool(use_agent_substitute and substitute_agent_name == "codex"),
            codex_result=codex_result,
            llm_result=llm_result,
            validation_result=validation_result,
            closure_result=closure_result,
            auto_apply_high_confidence=auto_apply_high_confidence,
            candidate_discovery_status=candidate_discovery_status,
            candidate_statistics=candidate_statistics,
        ),
        "operator_boundary": {
            "local_source_arbitration_can_write": True,
            "online_llm_requires_execute_llm": True,
            "agent_substitute_default": bool(use_agent_substitute),
            "agent_substitute_name": substitute_agent_name,
            "agent_substitute_local_only": True,
            "agent_substitute_is_heuristic_draft": True,
            "agent_substitute_is_semantic_llm": False,
            "quality_profile_execution": profile_execution,
            "agent_substitute_supported_agents": ["codex", "workbuddy", "opencode", "hermes_agent", "openclaw", "custom_local_agent"],
            "codex_substitute_legacy_alias": True,
            "readable_llm_requires_execute_readable_llm": not bool(use_agent_substitute),
            "readable_llm_promote_requires_promote_readable_llm": not bool(use_agent_substitute),
            "auto_apply_requires_auto_apply_high_confidence": not bool(use_agent_substitute),
            "does_not_modify_raw_asr_or_subtitle_sources": True,
            "secondary_asr_never_directly_promoted": True,
            "consensus_clips_require_explicit_execute": True,
            "local_targeted_asr_requires_explicit_execute": True,
            "local_targeted_asr_evidence_rebuilds_semantic_pack": bool(execute_local_targeted_asr),
            "provider_config_runtime_only": True,
            "low_confidence_or_high_risk_conflicts_require_review": True,
        },
        "updated_at": now_iso(),
    }

    if write:
        _write_pipeline_outputs(root, result, semantic_limit=semantic_limit, semantic_min_confidence=semantic_min_confidence)
        register_bundle_run(
            root,
            run_type="transcript_evidence_correction_pipeline",
            status=result["status"],
            title="转写证据仲裁纠错链路",
            summary=f"{candidate_count} semantic candidates; status={result['status']}.",
            inputs={
                "platform_subtitle": str(platform_subtitle or ""),
                "subtitle": str(subtitle or ""),
                "asr_json": str(asr_json or ""),
                "media_path": str(media_path or ""),
                "glossary_json": str(glossary_json or ""),
            },
            parameters={
                "quality_profile": quality_profile or "legacy",
                "secondary_asr_json": str(secondary_asr_json or ""),
                "consensus_agreement_threshold": float(consensus_agreement_threshold),
                "additional_secondary_asr_json": [str(value) for value in additional_secondary_asr_json or []],
                "execute_consensus_clips": bool(execute_consensus_clips),
                "execute_llm": bool(execute_llm),
                "execute_local_targeted_asr": bool(execute_local_targeted_asr),
                "local_targeted_asr_preset": local_targeted_asr_preset,
                "local_targeted_asr_model": str(local_targeted_asr_model or ""),
                "local_targeted_asr_timeout_seconds": int(local_targeted_asr_timeout_seconds or 0),
                "local_targeted_asr_allow_cpu": bool(local_targeted_asr_allow_cpu),
                "use_agent_substitute": bool(use_agent_substitute),
                "agent_name": substitute_agent_name,
                "use_codex_substitute": bool(use_codex_substitute) if use_codex_substitute is not None else bool(use_agent_substitute and substitute_agent_name == "codex"),
                "run_readable_llm": bool(run_readable_llm),
                "execute_readable_llm": bool(execute_readable_llm),
                "promote_readable_llm": bool(promote_readable_llm),
                "readable_max_segments_per_batch": int(readable_max_segments_per_batch or 0),
                "readable_max_prompt_chars": int(readable_max_prompt_chars or 0),
                "readable_max_tokens": int(readable_max_tokens or 0),
                "auto_apply_high_confidence": bool(auto_apply_high_confidence),
                "run_postprocess": bool(run_postprocess),
                "source_min_confidence": source_min_confidence,
                "semantic_min_confidence": semantic_min_confidence,
                "semantic_limit": int(semantic_limit or 0),
                "refresh_exports": bool(refresh_exports),
            },
            artifacts=[
                {"key": "pipeline", "path": root / "transcript-evidence-correction-pipeline.md"},
                {"key": "postprocessed_transcript", "path": root / "postprocessed-transcript.json"},
                {"key": "llm_readable_transcript", "path": root / "llm-readable-transcript.json"},
                {"key": "evidence_conflict_index", "path": root / "evidence-conflict-index.json"},
                {"key": "evidence_conflict_llm_pack", "path": root / "evidence-conflict-llm-pack.json"},
                {"key": "corrected_transcript", "path": root / "corrected-transcript.json"},
                {"key": "semantic_pack", "path": root / "transcript-semantic-correction-pack.json"},
                {"key": "full_transcript", "path": root / "exports" / "full-transcript.md"},
                {"key": "smart_summary", "path": root / "exports" / "smart-summary.md"},
                {"key": "mcp_args", "path": root / "mcp-transcript-evidence-correction-pipeline.args.json"},
            ],
            next_actions=result["next_actions"],
            operator_boundary=result["operator_boundary"],
            write=True,
        )
    return result




def _primary_asr_for_consensus(root: Path, explicit: str | Path | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        path = path if path.is_absolute() else root / path
        return path.resolve() if path.exists() else None
    manifest = _read_manifest(root)
    for key in ("normalized_transcript_json", "asr_transcript_json", "postprocessed_transcript_json", "transcript_json"):
        value = str(manifest.get(key) or "").strip()
        if not value:
            continue
        path = _bundle_path(root, value)
        if path.exists():
            return path.resolve()
    for name in ("normalized-transcript.json", "postprocessed-transcript.json"):
        path = root / name
        if path.exists():
            return path.resolve()
    return None

def _quality_profile_execution(
    root: Path,
    *,
    profile_name: str,
    provider_config: dict[str, Any] | None,
    candidate_count: int,
    readable_max_prompt_chars: int,
    explicit_execute: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not profile_name:
        return {"status": "legacy_explicit_flags", "auto_execute": False}, provider_config
    profile = processing_profile(profile_name)
    runtime = resolve_text_provider_config(provider_config)
    provider_ready = bool(runtime.get("model") and runtime.get("base_url")) and (not provider_requires_api_key(runtime) or bool(runtime.get("api_key")))
    transcript_chars = _estimated_transcript_chars(root)
    readable_calls = int(math.ceil(transcript_chars / max(1000, int(readable_max_prompt_chars or 9000)))) if transcript_chars else 0
    estimated_calls = readable_calls + max(0, int(candidate_count or 0))
    call_threshold = max(1, int(profile.get("llm_preflight_call_threshold") or 20))
    char_threshold = max(1000, int(profile.get("llm_preflight_input_char_threshold") or 120000))
    data_export_allowed = bool(profile.get("data_export_allowed"))
    auto_requested = bool(profile.get("text_llm_auto_execute")) and not explicit_execute
    over_threshold = estimated_calls > call_threshold or transcript_chars > char_threshold
    auto_execute = bool(auto_requested and data_export_allowed and provider_ready and not over_threshold)
    if explicit_execute:
        status = "explicit_execution"
    elif not data_export_allowed:
        status = "data_export_not_allowed"
    elif not provider_ready:
        status = "provider_not_ready"
    elif over_threshold:
        status = "batch_preflight_required"
    elif auto_execute:
        status = "auto_execution_ready"
    else:
        status = "auto_execution_disabled"
    public_provider = {key: value for key, value in runtime.items() if key not in {"api_key", "token", "secret"}}
    public_provider["api_key_configured"] = bool(runtime.get("api_key"))
    return {
        "status": status,
        "profile": profile.get("name") or profile_name,
        "auto_execute": auto_execute,
        "data_export_allowed": data_export_allowed,
        "provider_ready": provider_ready,
        "provider": public_provider,
        "estimated_calls": estimated_calls,
        "estimated_input_chars": transcript_chars,
        "call_threshold": call_threshold,
        "input_char_threshold": char_threshold,
        "preflight_required": over_threshold,
    }, runtime


def _estimated_transcript_chars(root: Path) -> int:
    manifest = _read_manifest(root)
    for key in ("source_arbitrated_transcript_json", "postprocessed_transcript_json", "normalized_transcript_json", "transcript_json"):
        value = str(manifest.get(key) or "").strip()
        if not value:
            continue
        path = _bundle_path(root, value)
        if path.exists():
            try:
                return len(path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
    return 0


def _normalise_agent_name(agent_name: str, *, legacy_codex: bool = False) -> str:
    name = str(agent_name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not name or name == "local_agent":
        return "codex" if legacy_codex else "local_agent"
    aliases = {"hermes": "hermes_agent", "hermesagent": "hermes_agent", "open_code": "opencode", "open_claw": "openclaw"}
    return aliases.get(name, name)


def _write_conflict_llm_pack(root: Path, pack_result: dict[str, Any], conflict_index: dict[str, Any], *, write: bool) -> dict[str, Any]:
    pack_path = _bundle_path(root, pack_result.get("pack_json") or "transcript-semantic-correction-pack.json")
    try:
        pack = read_json(pack_path)
    except Exception as exc:
        return {"status": "missing_semantic_pack", "ok": False, "error": str(exc), "pack_json": "", "candidate_count": 0}
    if not isinstance(pack, dict):
        return {"status": "invalid_semantic_pack", "ok": False, "pack_json": str(pack_path), "candidate_count": 0}
    conflict_ids = {str(row.get("candidate_id") or "") for row in (conflict_index.get("conflicts") or []) if isinstance(row, dict)}
    original_candidates = [row for row in (pack.get("candidates") or []) if isinstance(row, dict)]
    filtered_candidates = [row for row in original_candidates if str(row.get("candidate_id") or "") in conflict_ids]
    filtered = dict(pack)
    filtered["schema"] = "video_knowledge_pipeline.evidence_conflict_llm_pack.v1"
    filtered["source_pack_json"] = str(pack_path)
    filtered["source_conflict_index_json"] = str(root / "evidence-conflict-index.json")
    filtered["status"] = "conflict_pack_ready" if filtered_candidates else "no_real_conflicts"
    filtered["candidate_count"] = len(filtered_candidates)
    filtered["total_source_candidate_count"] = len(original_candidates)
    filtered["candidates"] = filtered_candidates
    filtered["candidate_groups"] = []
    filtered["operator_boundary"] = {
        "local_only": True,
        "no_cloud_call": True,
        "llm_pack_contains_only_evidence_conflicts": True,
        "heuristic_risks_without_external_evidence_excluded": True,
    }
    output = root / "evidence-conflict-llm-pack.json"
    if write:
        write_json(output, filtered)
        manifest = _read_manifest(root)
        manifest["evidence_conflict_llm_pack_json"] = "evidence-conflict-llm-pack.json"
        manifest["evidence_conflict_llm_pack_summary"] = {
            "status": filtered["status"],
            "candidate_count": len(filtered_candidates),
            "total_source_candidate_count": len(original_candidates),
            "updated_at": now_iso(),
        }
        write_json(root / "manifest.json", manifest)
    return {
        "status": filtered["status"],
        "ok": True,
        "pack_json": str(output),
        "candidate_count": len(filtered_candidates),
        "total_source_candidate_count": len(original_candidates),
        "excluded_candidate_count": max(0, len(original_candidates) - len(filtered_candidates)),
    }


def _write_pipeline_outputs(root: Path, result: dict[str, Any], *, semantic_limit: int, semantic_min_confidence: float) -> None:
    with bundle_write_lock(root, operation="transcript_evidence_correction_pipeline", timeout_seconds=1.0):
        write_json(root / "transcript-evidence-correction-pipeline.json", result)
        (root / "transcript-evidence-correction-pipeline.md").write_text(_render_markdown(result), encoding="utf-8")
        write_json(
            root / "mcp-transcript-evidence-correction-pipeline.args.json",
            {
                "bundle_dir": str(root),
                "provider_config": {},
                "execute_llm": False,
                "use_agent_substitute": True,
                "agent_name": "local_agent",
                "use_codex_substitute": False,
                "run_readable_llm": True,
                "execute_readable_llm": False,
                "promote_readable_llm": False,
                "readable_max_segments_per_batch": 40,
                "readable_max_prompt_chars": 9000,
                "readable_max_tokens": 4000,
                "auto_apply_high_confidence": False,
                "run_postprocess": True,
                "source_min_confidence": 0.72,
                "semantic_min_confidence": semantic_min_confidence,
                "semantic_limit": int(semantic_limit or 0),
                "refresh_exports": True,
                "write": True,
            },
        )
        manifest = _read_manifest(root)
        manifest["transcript_evidence_correction_pipeline_json"] = "transcript-evidence-correction-pipeline.json"
        manifest["transcript_evidence_correction_pipeline_markdown"] = "transcript-evidence-correction-pipeline.md"
        manifest["transcript_evidence_correction_pipeline_summary"] = {
            "status": result.get("status"),
            "ok": result.get("ok"),
            "candidate_count": (result.get("semantic_pack") or {}).get("candidate_count"),
            "candidate_discovery_status": result.get("candidate_discovery_status"),
            "discovered_candidate_count": result.get("discovered_candidate_count"),
            "validated_candidate_count": result.get("validated_candidate_count"),
            "unvalidated_candidate_count": result.get("unvalidated_candidate_count"),
            "deferred_candidate_count": result.get("deferred_candidate_count"),
            "review_required_candidate_count": result.get("review_required_candidate_count"),
            "validation_review_required_count": result.get("validation_review_required_count"),
            "corrected_transcript_exists": result.get("corrected_transcript_exists"),
            "conflict_count": (result.get("evidence_conflict_index") or {}).get("conflict_count"),
            "llm_status": (result.get("llm_review") or {}).get("status"),
            "agent_status": (result.get("agent_review") or {}).get("status"),
            "agent_name": (result.get("operator_boundary") or {}).get("agent_substitute_name"),
            "codex_status": (result.get("codex_review") or {}).get("status"),
            "readable_llm_status": (result.get("readable_llm_polish") or {}).get("status"),
            "closure_status": (result.get("closure") or {}).get("status"),
            "updated_at": result.get("updated_at"),
        }
        manifest["mcp_transcript_evidence_correction_pipeline_args"] = "mcp-transcript-evidence-correction-pipeline.args.json"
        write_json(root / "manifest.json", manifest)


def _materialise_corrected_transcript_alias(root: Path) -> dict[str, Any]:
    manifest = _read_manifest(root)
    source_json = _bundle_path(root, manifest.get("corrected_transcript_json") or manifest.get("source_arbitrated_transcript_json") or "source-arbitrated-transcript.json")
    source_srt = _bundle_path(root, manifest.get("corrected_transcript_srt") or manifest.get("source_arbitrated_transcript_srt") or "source-arbitrated-transcript.srt")
    source_md = _bundle_path(root, manifest.get("corrected_transcript_markdown") or manifest.get("source_arbitrated_transcript_markdown") or "source-arbitrated-transcript.md")
    copied: list[str] = []
    if source_json.exists() and source_json.name != "corrected-transcript.json":
        shutil.copyfile(source_json, root / "corrected-transcript.json")
        copied.append("corrected-transcript.json")
    if source_srt.exists() and source_srt.name != "corrected-transcript.srt":
        shutil.copyfile(source_srt, root / "corrected-transcript.srt")
        copied.append("corrected-transcript.srt")
    if source_md.exists() and source_md.name != "corrected-transcript.md":
        shutil.copyfile(source_md, root / "corrected-transcript.md")
        copied.append("corrected-transcript.md")
    if (root / "corrected-transcript.json").exists():
        manifest["corrected_transcript_json"] = "corrected-transcript.json"
        manifest["transcript_json"] = "corrected-transcript.json"
    if (root / "corrected-transcript.srt").exists():
        manifest["corrected_transcript_srt"] = "corrected-transcript.srt"
        manifest["transcript_srt"] = "corrected-transcript.srt"
    if (root / "corrected-transcript.md").exists():
        manifest["corrected_transcript_markdown"] = "corrected-transcript.md"
    write_json(root / "manifest.json", manifest)
    return {
        "status": "completed" if (root / "corrected-transcript.json").exists() else "missing_corrected_source",
        "ok": (root / "corrected-transcript.json").exists(),
        "copied": copied,
        "corrected_transcript_json": str(root / "corrected-transcript.json"),
    }


def _pipeline_status(
    *,
    candidate_count: int,
    semantic_candidate_count: int,
    targeted_window_count: int,
    execute_llm: bool,
    use_agent_substitute: bool,
    agent_name: str,
    use_codex_substitute: bool,
    codex_result: dict[str, Any] | None,
    llm_result: dict[str, Any] | None,
    validation_result: dict[str, Any] | None,
    closure_result: dict[str, Any] | None,
    auto_apply_high_confidence: bool,
    candidate_discovery_status: str = "unknown",
    candidate_statistics: dict[str, Any] | None = None,
    corrected_transcript_exists: bool | None = None,
    quality_gate_result: dict[str, Any] | None = None,
) -> str:
    if targeted_window_count > 0:
        return "needs_local_targeted_asr_evidence"
    stats = candidate_statistics if isinstance(candidate_statistics, dict) else {}
    unresolved_count = int(stats.get("unresolved_candidate_count") or 0)
    if _candidate_discovery_requires_work(candidate_discovery_status):
        return "needs_candidate_discovery_or_review"
    if unresolved_count > 0:
        return "needs_candidate_discovery_or_review"
    if isinstance(quality_gate_result, dict) and quality_gate_result:
        if not bool(quality_gate_result.get("ok")):
            return "complete_with_review_gaps"
    if corrected_transcript_exists is False:
        return "complete_with_review_gaps"
    if semantic_candidate_count > 0 and candidate_count <= 0:
        return "complete_with_review_gaps"
    if candidate_count <= 0:
        return "completed_no_semantic_candidates"
    if not execute_llm:
        if use_agent_substitute:
            if closure_result:
                return str(closure_result.get("status") or "codex_closure_unknown")
            if codex_result:
                return str(codex_result.get("status") or "codex_substitute_completed")
            return "codex_substitute_not_run"
        return "needs_llm_execution"
    if not llm_result or llm_result.get("status") != "executed":
        return "llm_execution_failed_or_incomplete"
    if not validation_result:
        return "needs_validation"
    if int(validation_result.get("accepted_decision_count") or 0) <= 0:
        if int(validation_result.get("arbitrated_no_change_count") or 0) > 0 and int(validation_result.get("rejected_decision_count") or 0) == 0:
            return "completed_no_llm_changes"
        return "no_safe_llm_decisions"
    if not auto_apply_high_confidence:
        return "needs_auto_apply_or_review"
    if not closure_result:
        return "needs_closure"
    return str(closure_result.get("status") or "closure_unknown")


_CANDIDATE_DISCOVERY_COMPLETE_STATUSES = {
    "imported",
    "no_candidates_imported",
    "no_segments_selected",
    "no_suggestions",
}


def _candidate_discovery_requires_work(status: str) -> bool:
    value = str(status or "unknown").strip().lower()
    if value in {"", "unknown"}:
        # Preserve callers using the pre-v2 private helper contract. The public
        # pipeline always supplies the actual discovery status.
        return False
    return value not in _CANDIDATE_DISCOVERY_COMPLETE_STATUSES


def _candidate_statistics(
    pack_result: dict[str, Any] | None,
    conflict_index_result: dict[str, Any] | None,
    validation_result: dict[str, Any] | None,
) -> dict[str, int]:
    """Separate discovery, validation, deferral, and review accounting.

    Validation's historic ``review_required_count`` counts only rejected model
    decisions. It does not include candidates deferred before model review. The
    latter remain visible here as candidate-level review debt instead of being
    mistaken for a clean zero.
    """

    pack = pack_result if isinstance(pack_result, dict) else {}
    conflict_index = (
        conflict_index_result if isinstance(conflict_index_result, dict) else {}
    )
    validation = validation_result if isinstance(validation_result, dict) else {}
    candidates = [
        row for row in (pack.get("candidates") or []) if isinstance(row, dict)
    ]
    candidate_ids = {
        str(row.get("candidate_id") or f"candidate-{idx}")
        for idx, row in enumerate(candidates)
    }

    def decision_ids(key: str) -> set[str]:
        return {
            str(row.get("candidate_id") or "")
            for row in (validation.get(key) or [])
            if isinstance(row, dict) and str(row.get("candidate_id") or "")
        }

    accepted_ids = decision_ids("accepted_decisions")
    no_change_ids = decision_ids("arbitrated_no_change_decisions")
    rejected_ids = decision_ids("rejected_decisions")
    review_ids = {
        str(row.get("candidate_id") or "")
        for row in (validation.get("review_rows") or [])
        if isinstance(row, dict) and str(row.get("candidate_id") or "")
    }
    validated_ids = (accepted_ids | no_change_ids) & candidate_ids
    validated_count = max(
        len(validated_ids),
        int(validation.get("accepted_decision_count") or 0)
        + int(validation.get("arbitrated_no_change_count") or 0),
    )
    validated_count = min(len(candidate_ids), validated_count)
    deferred_ids = {
        str(row.get("candidate_id") or f"candidate-{idx}")
        for idx, row in enumerate(candidates)
        if row.get("llm_review_eligible") is False
        or bool(str(row.get("llm_review_defer_reason") or "").strip())
    }
    candidate_review_ids = {
        str(row.get("candidate_id") or f"candidate-{idx}")
        for idx, row in enumerate(candidates)
        if bool(row.get("needs_human_review"))
    }
    validation_review_count = max(
        len(review_ids), int(validation.get("review_required_count") or 0)
    )
    review_required_count = max(
        len(candidate_review_ids | deferred_ids | review_ids | rejected_ids),
        validation_review_count,
    )
    discovered_count = len(candidate_ids)
    unvalidated_count = max(0, discovered_count - validated_count)
    unresolved_count = max(
        unvalidated_count,
        len(deferred_ids),
        review_required_count,
        int(validation.get("rejected_decision_count") or 0),
    )
    return {
        "discovered_candidate_count": discovered_count,
        "actionable_conflict_count": int(
            conflict_index.get("llm_arbitration_count")
            or conflict_index.get("conflict_count")
            or 0
        ),
        "validated_candidate_count": validated_count,
        "unvalidated_candidate_count": unvalidated_count,
        "deferred_candidate_count": len(deferred_ids),
        "review_required_candidate_count": review_required_count,
        "validation_review_required_count": validation_review_count,
        "rejected_decision_count": int(
            validation.get("rejected_decision_count") or len(rejected_ids)
        ),
        "unresolved_candidate_count": unresolved_count,
    }


def _pipeline_ok(
    status: str,
    *,
    candidate_discovery_status: str = "unknown",
    candidate_statistics: dict[str, Any] | None = None,
    corrected_transcript_exists: bool | None = None,
    quality_gate_result: dict[str, Any] | None = None,
) -> bool:
    if _candidate_discovery_requires_work(candidate_discovery_status):
        return False
    stats = candidate_statistics if isinstance(candidate_statistics, dict) else {}
    if int(stats.get("unresolved_candidate_count") or 0) > 0:
        return False
    if corrected_transcript_exists is False:
        return False
    if isinstance(quality_gate_result, dict) and quality_gate_result:
        if not bool(quality_gate_result.get("ok")):
            return False
    return str(status or "") in {
        "accepted",
        "accepted_with_no_change",
        "arbitrated_no_change",
        "completed",
        "completed_no_llm_changes",
        "completed_no_semantic_candidates",
        "completed_no_text_changes",
        "no_candidates",
        "no_eligible_candidates",
    }


def _should_refresh_exports(source: dict[str, Any] | None, source_arbitration: dict[str, Any] | None, readable_llm: dict[str, Any] | None = None) -> bool:
    if isinstance(source, dict) and source.get("status") in {"completed", "completed_no_text_changes"}:
        return True
    if isinstance(readable_llm, dict) and readable_llm.get("status") in {"executed", "imported", "codex_substitute_executed", "agent_substitute_executed"} and readable_llm.get("promote"):
        return True
    summary = source_arbitration.get("summary") if isinstance(source_arbitration, dict) else {}
    return int((summary or {}).get("changed_segments") or 0) > 0


def _next_actions(
    *,
    candidate_count: int,
    semantic_candidate_count: int,
    targeted_window_count: int,
    execute_llm: bool,
    use_agent_substitute: bool,
    agent_name: str,
    use_codex_substitute: bool,
    codex_result: dict[str, Any] | None,
    llm_result: dict[str, Any] | None,
    validation_result: dict[str, Any] | None,
    closure_result: dict[str, Any] | None,
    auto_apply_high_confidence: bool,
    candidate_discovery_status: str = "unknown",
    candidate_statistics: dict[str, Any] | None = None,
) -> list[str]:
    if targeted_window_count > 0:
        return [
            "Run plan-local-targeted-asr-evidence and asr-retry-snippets --execute to extract only the planned local clips.",
            "Run a distinct local ASR preset on each clip, then register the verified output with asr-local-targeted-evidence.",
            "Rerun transcript-evidence-correction-pipeline after independent local evidence is available; do not promote the current text as final yet.",
        ]
    if _candidate_discovery_requires_work(candidate_discovery_status):
        return [
            "Run the bounded transcript semantic candidate-discovery pass before claiming correction completion.",
            "Review or validate deferred candidates; zero cross-source conflicts is not evidence that the transcript is semantically clean.",
        ]
    stats = candidate_statistics if isinstance(candidate_statistics, dict) else {}
    if int(stats.get("unresolved_candidate_count") or 0) > 0:
        return [
            "Resolve, validate, or explicitly retain the outstanding semantic candidates before final acceptance.",
            "Keep deferred and unvalidated counts visible as review gaps; do not treat validation_review_required_count alone as total review debt.",
        ]
    if semantic_candidate_count > 0 and candidate_count <= 0:
        return ["No actionable cross-source conflict exists after semantic screening; retain the semantic pack for review and refresh exports only after any human correction."]
    if candidate_count <= 0:
        return ["Run export-knowledge-note if the transcript changed and readable exports are stale."]
    if not execute_llm:
        if use_agent_substitute:
            label = agent_name or "local_agent"
            if closure_result and closure_result.get("status") in {"completed", "completed_no_text_changes"}:
                return [f"Use exports/full-transcript.md and exports/smart-summary.md as local agent-substitute outputs ({label}), or rerun with --execute-llm for online arbitration."]
            if codex_result and codex_result.get("status") == "no_safe_draft_decisions":
                return [f"Local agent substitute ({label}) found no safe automatic changes; add stronger subtitle/OCR/visual evidence or explicitly run --execute-llm."]
            return ["Inspect transcript-semantic-correction-result.codex.md and rerun closure if needed; online LLM remains optional with --execute-llm."]
        return ["Run transcript-evidence-correction-pipeline again with --execute-llm and runtime provider_config to let an online LLM judge semantic candidates."]
    if not llm_result or llm_result.get("status") != "executed":
        return ["Check transcript-semantic-correction-llm-prompt.md and provider settings, then retry with a small semantic_limit."]
    if not validation_result or int(validation_result.get("accepted_decision_count") or 0) <= 0:
        if validation_result and int(validation_result.get("arbitrated_no_change_count") or 0) > 0 and int(validation_result.get("rejected_decision_count") or 0) == 0:
            return ["LLM arbitration confirmed no safe text changes for this batch; build stronger conflict candidates or export current transcript."]
        return ["Review transcript-semantic-correction-review.md or provide stronger subtitle/OCR/visual evidence before applying corrections."]
    if not auto_apply_high_confidence:
        return ["Rerun with --auto-apply-high-confidence after reviewing validation, or import human review notes before closure."]
    if closure_result and closure_result.get("status") in {"completed", "completed_no_text_changes"}:
        return ["Use exports/full-transcript.md and exports/smart-summary.md as the refreshed human-readable outputs."]
    return ["Inspect transcript-semantic-correction-closure.md for unmatched accepted decisions."]


def _summary(result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    keys = [
        "status",
        "ok",
        "source_count",
        "candidate_count",
        "eligible_candidate_count",
        "selected_candidate_count",
        "window_count",
        "conflict_count",
        "llm_arbitration_count",
        "postprocessed_segment_count",
        "polished_segment_count",
        "batch_count",
        "agent_substitute",
        "agent_substitute_name",
        "codex_substitute",
        "total_candidate_count",
        "decision_count",
        "review_suggestion_count",
        "accepted_decision_count",
        "arbitrated_no_change_count",
        "rejected_decision_count",
        "fail_count",
        "warning_count",
        "quality_layers",
        "applied_correction_count",
        "changed_segment_count",
        "result_json",
        "pack_json",
        "error",
    ]
    return {key: result.get(key) for key in keys if key in result}


def _step(name: str, result: dict[str, Any] | None) -> dict[str, Any]:
    summary = _summary(result)
    status = str(summary.get("status") or "unknown")
    ok_value = summary.get("ok")
    if ok_value is None:
        ok_value = status in {"completed", "completed_no_text_changes", "pack_ready", "planned", "executed", "accepted"}
    return {"name": name, "status": status, "ok": bool(ok_value), "summary": summary}


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# 转写证据仲裁纠错链路",
        "",
        f"- Bundle: `{result.get('bundle_dir', '')}`",
        f"- Status: `{result.get('status', '')}`",
        f"- OK: `{result.get('ok', False)}`",
        f"- Candidate discovery: `{result.get('candidate_discovery_status', '')}`",
        f"- Candidates discovered / validated / unvalidated / deferred / review required: "
        f"`{result.get('discovered_candidate_count', 0)}` / "
        f"`{result.get('validated_candidate_count', 0)}` / "
        f"`{result.get('unvalidated_candidate_count', 0)}` / "
        f"`{result.get('deferred_candidate_count', 0)}` / "
        f"`{result.get('review_required_candidate_count', 0)}`",
        f"- Validation-only review required (legacy meaning): "
        f"`{result.get('validation_review_required_count', 0)}`",
        f"- Corrected transcript exists: `{result.get('corrected_transcript_exists', False)}`",
        f"- Updated: `{result.get('updated_at', '')}`",
        "",
        "## 链路",
        "",
        "```text",
        "SenseVoice 原始 ASR",
        "+ 自带字幕/本地抓取字幕/网页上下文",
        "+ 青龙打标器时间轴/重点/话题/画面状态",
        "+ OCR/ebook/多模态证据",
        "+ 在线 LLM 标点/断句后处理（显式 execute_readable_llm）",
        "+ 在线 LLM 语义仲裁（只审真实冲突）",
        "-> source-arbitrated-transcript.json",
        "-> llm-readable-transcript.json（可选：只做标点/断句/段落，不改事实）",
        "-> corrected-transcript.json",
        "-> full-transcript.md",
        "-> smart-summary.md",
        "```",
        "",
        "## 步骤状态",
        "",
        "| Step | Status | OK |",
        "| --- | --- | --- |",
    ]
    for step in result.get("steps") or []:
        if not isinstance(step, dict):
            continue
        lines.append(f"| `{step.get('name', '')}` | `{step.get('status', '')}` | `{step.get('ok', False)}` |")
    lines.extend(["", "## 下一步", ""])
    for action in result.get("next_actions") or []:
        lines.append(f"- {action}")
    lines.extend(["", "## 产物", "", "| Artifact | Path |", "| --- | --- |"])
    for key, value in (result.get("artifacts") or {}).items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## 边界", ""])
    for key, value in (result.get("operator_boundary") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines).rstrip() + "\n"


def _register_secondary_asr_sources(root: Path, sources: list[str | Path]) -> list[str]:
    existing_manifest = _read_manifest(root)
    existing = existing_manifest.get("asr_secondary_transcripts")
    values = existing if isinstance(existing, list) else ([existing] if existing else [])
    legacy = existing_manifest.get("asr_secondary_transcript")
    if legacy and legacy not in values:
        values = [legacy, *values]
    registered: list[str] = []
    for value in [*values, *sources]:
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        resolved = str(path.resolve())
        if resolved not in registered:
            registered.append(resolved)
    if not registered:
        return []
    with bundle_write_lock(root, operation="register_secondary_asr_sources", timeout_seconds=1.0):
        manifest = _read_manifest(root)
        manifest["asr_secondary_transcripts"] = registered
        if not manifest.get("asr_secondary_transcript"):
            manifest["asr_secondary_transcript"] = registered[0]
        write_json(root / "manifest.json", manifest)
    return registered


def _read_manifest(root: Path) -> dict[str, Any]:
    value = read_json(root / "manifest.json")
    return value if isinstance(value, dict) else {}


def _bundle_path(root: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else root / path
