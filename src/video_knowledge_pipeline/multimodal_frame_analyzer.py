from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .config import resolve_vision_execution_profile
from .frame_recapture import _coverage_audit, _quality_audit
from .knowledge_coverage import audit_knowledge_coverage
from .markdown_text import markdown_table_cell as _md_cell
from .model_task_gateway import model_task_api_call
from .model_runtime_client import authorise_consented_remote_runtime
from .models import now_iso
from .powershell import quote_powershell_literal as _quote_ps_path
from .repair_status import build_repair_status
from .run_artifact_registry import register_bundle_run
from .storage import append_jsonl, read_json, read_jsonl, write_json
from .video_frame_router import _frame_paths
from .vision_api import (
    parse_model_json,
    provider_requires_api_key,
    resolve_provider_config,
)
from .vision_execution_route import resolve_vision_task_execution_route, check_vision_invocation
from .vision_batch_continuation import write_semantic_continuation
from .vision_export_consent import validate_vision_export_consent, vision_export_consent_image_limits
from .vision_preflight import vision_execution_preflight
from .visual_integration import integrated_visual
from .visual_evidence import evidence_index_sets
from .evidence_lineage import call_visual_with_provenance
from .vlm_preprocess import prepare_image_probe


def call_vision_model(*, provider_config, prompt, image_paths, allowed_roots=None, max_tokens=None):
    kwargs = {
        "provider_config": provider_config,
        "prompt": prompt,
        "image_paths": image_paths,
        "allowed_roots": allowed_roots,
        "execute": True,
        "write": False,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return model_task_api_call(
        "multimodal_frame_analysis",
        **kwargs,
    )


def call_vision_model_with_broker_reservation(*, provider_config, prompt, image_paths, allowed_roots=None, max_tokens=None):
    """Invoke one remote proxy request only inside the consent-bound runtime grant."""
    config = dict(provider_config or {})
    is_remote_proxy = (
        str(config.get("adapter_backend") or "").strip().lower() == "proxy"
        and str(config.get("execution_location") or "").strip().lower() == "remote"
    )
    if not is_remote_proxy:
        kwargs = {"provider_config": config, "prompt": prompt, "image_paths": image_paths, "allowed_roots": allowed_roots}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return call_vision_model(**kwargs)
    with authorise_consented_remote_runtime(
        consent_id=str(config.get("consent_id") or ""),
        route_revision=str(config.get("route_revision") or ""),
        max_calls=1,
    ):
        kwargs = {"provider_config": config, "prompt": prompt, "image_paths": image_paths, "allowed_roots": allowed_roots}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return call_vision_model(**kwargs)

def run_multimodal_frame_analysis(
    bundle_dir: str | Path,
    *,
    execute: bool = False,
    provider_config: dict[str, Any] | None = None,
    input_json: str | Path | None = None,
    limit: int | None = None,
    indexes: list[int] | None = None,
    confirm_vision_calls: int | None = None,
    confirm_vision_indexes: str = "",
    image_probe_max_edge: int = 0,
    image_probe_jpeg_quality: int = 70,
    vision_retries: int = 1,
    vision_retry_delay_seconds: float = 0.0,
    execution_actor: str = "operator",
    export_consent: str | Path | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"bundle missing manifest.json: {root}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")
    route_resolution = resolve_vision_task_execution_route("semantic_frame", provider_config=provider_config)
    invocation_check = check_vision_invocation(execute=execute, route=route_resolution)
    if execute and route_resolution.get("legacy_fallback_blocked"):
        raise ValueError("No configured single-frame gateway route; legacy provider fallback is blocked for execution")
    effective_provider_config = provider_config
    if provider_config is None and not route_resolution.get("legacy_fallback_blocked"):
        effective_provider_config = dict(route_resolution.get("provider_config") or {})
    profile = resolve_vision_execution_profile(
        provider_config=effective_provider_config,
        multimodal_limit=limit,
    )
    effective_limit = int(profile["multimodal_limit"])
    timeline = _read_timeline(root)
    before_timeline = snapshot_timeline_for_vision_diff(timeline)
    all_candidates = _candidates(root, timeline, explicit_indexes=indexes)
    candidates = _select_candidates(all_candidates, limit=effective_limit, indexes=indexes)
    template_path = write_multimodal_frame_input_template(root, candidates)

    imported = _read_import(input_json, "visual_understanding") if input_json else []
    applied = _apply_imports(timeline, imported, field="visual_understanding") if imported else []
    results = []
    cfg = resolve_provider_config(profile["provider_config"])
    effective_image_probe_max_edge = int(image_probe_max_edge or 0)
    effective_image_probe_jpeg_quality = int(image_probe_jpeg_quality or 70)
    if execute and str(execution_actor or "operator").strip().lower() == "agent" and export_consent:
        consent_limits = vision_export_consent_image_limits(export_consent)
        effective_image_probe_max_edge = int(consent_limits["image_max_edge"])
        effective_image_probe_jpeg_quality = int(consent_limits["image_jpeg_quality"])
    gate, execution_control = _execution_control(
        root,
        execute,
        cfg,
        semantic_limit=effective_limit,
        temporal_limit=0,
        frame_count=int(profile["frame_count"]),
        include_semantic=True,
        include_temporal=False,
        semantic_indexes=indexes,
        confirm_vision_calls=confirm_vision_calls,
        confirm_vision_indexes=confirm_vision_indexes,
        image_probe_max_edge=effective_image_probe_max_edge,
        image_probe_jpeg_quality=effective_image_probe_jpeg_quality,
        execution_actor=execution_actor,
        export_consent=export_consent,
    )
    if (
        execute
        and not gate
        and candidates
        and str(cfg.get("adapter_backend") or "").strip().lower() == "proxy"
        and str(cfg.get("execution_location") or "").strip().lower() == "remote"
    ):
        consent_status = execution_control.get("export_consent") if isinstance(execution_control.get("export_consent"), dict) else {}
        consent_id = str(consent_status.get("consent_id") or "")
        if not consent_status.get("valid") or not consent_id:
            gate = {"status": "vision_export_consent_required", "error": "vision_export_consent_invalid"}
        else:
            cfg = {**cfg, "consent_id": consent_id}

    batch_abort_error = ""
    progress = _new_vision_batch_progress(
        root,
        kind="semantic_frame",
        candidates=candidates,
        execute=execute,
        gate=gate,
    )
    for position, candidate in enumerate(candidates, start=1):
        _write_vision_batch_progress(
            root,
            progress,
            results=results,
            current_index=_int_value(candidate.get("index")),
            current_position=position,
            status="running" if execute and not gate else "blocked" if gate else "planned",
        )
        result = {
            "index": candidate["index"],
            "visual_route": candidate.get("visual_route"),
            "frame_paths": candidate.get("frame_paths", []),
            "prompt": _prompt(candidate),
            "executed": bool(execute and not gate),
            "ok": False,
            "error": str(gate.get("error") or "") if gate else "",
            "request_mode": "single_frame",
            "max_images_per_request": 1,
            "request_max_tokens": int(max_tokens) if max_tokens is not None else None,
            "request_max_tokens_omitted": max_tokens is None,
            "finish_reason": "",
            "truncated": False,
            "complete": False,
        }
        if execute and not gate and batch_abort_error:
            result["executed"] = False
            result["error"] = batch_abort_error
            result["batch_aborted"] = True
            results.append(result)
            _write_vision_batch_progress(
                root,
                progress,
                results=results,
                current_index=_int_value(candidate.get("index")),
                current_position=position,
                status="running",
            )
            continue
        if execute and not gate and candidate.get("frame_paths"):
            item_started = time.perf_counter()
            original_image_paths = [_resolve_frame(root, path) for path in candidate.get("frame_paths", [])[:1]]
            image_probe = prepare_image_probe(
                original_image_paths,
                output_dir=root / "vision-analysis-image-probes" / "semantic" / str(candidate["index"]),
                max_edge=effective_image_probe_max_edge,
                jpeg_quality=effective_image_probe_jpeg_quality,
            )
            sent_image_paths = [str(path) for path in image_probe.get("image_paths") or original_image_paths if str(path)]
            call_kwargs: dict[str, Any] = {"allowed_roots": [str(root)]}
            if max_tokens is not None:
                call_kwargs["max_tokens"] = max_tokens
            response = call_visual_with_provenance(
                root=root, candidate=candidate, original_image_paths=original_image_paths,
                sent_image_paths=sent_image_paths, prompt=str(result["prompt"]), provider_config=cfg,
                invoke=lambda: call_vision_model_with_retries(
                    provider_config=cfg,
                    prompt=str(result["prompt"]),
                    image_paths=sent_image_paths,
                    attempts=vision_retries,
                    delay_seconds=vision_retry_delay_seconds,
                    call_model=call_vision_model_with_broker_reservation,
                    call_kwargs=call_kwargs,
                ),
            )
            truncated = bool(response.get("truncated"))
            complete = bool(response.get("complete", response.get("ok") and not truncated)) and not truncated
            response_error = str(response.get("error") or "")
            if truncated and not response_error:
                response_error = "model_output_truncated"
            result.update(
                {
                    "ok": bool(response.get("ok")) and complete,
                    "status": "truncated" if truncated else str(response.get("status") or ("ok" if response.get("ok") else "failed")),
                    "error": response_error,
                    "raw_content": response.get("content", ""),
                    "executed": bool(response.get("executed", True)),
                    "recognition_provenance": response.get("recognition_provenance", {}),
                    "model_actual": response.get("model_actual", ""),
                    "model_requested": response.get("model_requested", ""),
                    "sent_image_paths": sent_image_paths,
                    "sent_image_count": len(sent_image_paths),
                    "image_probe": image_probe,
                    "attempts": response.get("attempts", []),
                    "attempt_count": response.get("attempt_count", 1),
                    "request_max_tokens": response.get("request_max_tokens", max_tokens),
                    "request_max_tokens_omitted": bool(response.get("request_max_tokens_omitted", max_tokens is None)),
                    "finish_reason": str(response.get("finish_reason") or ""),
                    "response_chars": len(str(response.get("content") or "")),
                    "elapsed_ms": round((time.perf_counter() - item_started) * 1000.0, 3),
                    "truncated": truncated,
                    "complete": complete,
                }
            )
            if response.get("ok") and complete:
                understanding = _normalise_visual_understanding(parse_model_json(str(response.get("content") or "")), candidate)
                understanding.update(recognition_provenance=response['recognition_provenance'],
                                     executor='provider_api', model_actual=response['model_actual'],
                                     model_requested=response['model_requested'])
                _apply_single(timeline, int(candidate["index"]), "visual_understanding", understanding)
                applied.append(int(candidate["index"]))
                result["visual_understanding"] = understanding
            elif _is_batch_terminal_vision_error(str(response.get("error") or "")):
                batch_abort_error = _batch_abort_error(str(response.get("error") or ""))
                result["batch_abort_trigger"] = True
        results.append(result)
        _write_vision_batch_progress(
            root,
            progress,
            results=results,
            current_index=_int_value(candidate.get("index")),
            current_position=position,
            status="running" if execute and not gate else "blocked" if gate else "planned",
        )

    run_status = str(gate.get("status") or "ok") if gate else "ok"
    if batch_abort_error:
        run_status = "vision_batch_aborted"
    elif execute and not gate:
        run_status = _vision_batch_execution_status(results)
    elif not execute and not imported:
        run_status = "preview_only"
    evidence_indexes = evidence_index_sets(timeline)
    run_complete_indexes = sorted(
        {
            int(item.get("index"))
            for item in results
            if item.get("complete") and _int_value(item.get("index")) > 0
        }
    )
    run_truncated_indexes = sorted(
        {
            int(item.get("index"))
            for item in results
            if item.get("truncated") and _int_value(item.get("index")) > 0
        }
    )
    run_failed_indexes = sorted(
        {
            int(item.get("index"))
            for item in results
            if item.get("executed") and not item.get("ok") and _int_value(item.get("index")) > 0
        }
    )
    timeline_complete_indexes = evidence_indexes["visual_model_complete"]
    export_consumable_indexes = evidence_indexes["visual_export_consumable"]
    summary = {
        "schema": "lecture_multimodal_frame_analysis_summary.v1",
        "total": len(all_candidates),
        "selected": len(candidates),
        "limit": effective_limit,
        "indexes": [int(index) for index in indexes or []],
        "execute": execute,
        "status": run_status,
        "invocation_check": invocation_check,
        "error": batch_abort_error or (str(gate.get("error") or "") if gate else ""),
        "preflight_path": str(gate.get("preflight_path") or "") if gate else "",
        "preflight_json_path": str(gate.get("preflight_json_path") or "") if gate else "",
        "expected_api_calls": gate.get("expected_api_calls") if gate and "expected_api_calls" in gate else None,
        "expected_indexes": str(gate.get("expected_indexes") or "") if gate else "",
        "execution_control": execution_control,
        "image_probe_max_edge": effective_image_probe_max_edge,
        "image_probe_jpeg_quality": effective_image_probe_jpeg_quality,
        "vision_retries": int(vision_retries or 1),
        "vision_retry_delay_seconds": float(vision_retry_delay_seconds or 0),
        "request_max_tokens": int(max_tokens) if max_tokens is not None else None,
        "request_max_tokens_omitted": max_tokens is None,
        "request_mode": "single_frame",
        "max_images_per_request": 1,
        "complete_count": sum(1 for item in results if item.get("complete")),
        "truncated_count": sum(1 for item in results if item.get("truncated")),
        "failed_count": sum(1 for item in results if item.get("executed") and not item.get("ok")),
        "run_complete_indexes": run_complete_indexes,
        "run_truncated_indexes": run_truncated_indexes,
        "run_failed_indexes": run_failed_indexes,
        "timeline_complete_count": len(timeline_complete_indexes),
        "timeline_complete_indexes": timeline_complete_indexes,
        "export_consumable_count": len(export_consumable_indexes),
        "export_consumable_indexes": export_consumable_indexes,
        "run_complete_not_consumable_indexes": sorted(
            set(run_complete_indexes) - set(export_consumable_indexes)
        ),
        "imported": len(imported),
        "updated": len(set(applied)),
        "provider": {
            "provider": cfg.get("provider"), "base_url": cfg.get("base_url"), "model": cfg.get("model"),
            "api_key_configured": bool(cfg.get("api_key")) or bool(cfg.get("credential_ready")),
            "route_id": cfg.get("route_id"), "route_revision": cfg.get("route_revision"),
            "virtual_model": cfg.get("virtual_model"), "profile_id": cfg.get("profile_id"),
            "adapter_backend": cfg.get("adapter_backend"),
            "provider_config_source": str(route_resolution.get("provider_config_source") or ""),
            "route_resolution_status": str(route_resolution.get("status") or ""),
        },
        "updated_at": now_iso(),
    }
    summary["continuation"] = write_semantic_continuation(
        root,
        pending_indexes=[item["index"] for item in _candidates(root, timeline, explicit_indexes=indexes)],
        batch_size=effective_limit,
        summary=summary,
        execution_actor=str(execution_actor or "operator").strip().lower(),
    )
    progress_path = _write_vision_batch_progress(
        root,
        progress,
        results=results,
        current_index=0,
        current_position=len(candidates),
        status=run_status,
    )
    summary["progress_path"] = str(progress_path)
    manifest["multimodal_frame_analysis"] = {
        "schema": "lecture_multimodal_frame_analysis.v1",
        "count": len(all_candidates),
        "selected_count": len(candidates),
        "items": candidates,
        "input_template_json": str(template_path),
        "last_run": summary,
    }
    if applied:
        write_json(timeline_path, timeline)
        _sync_source_package(manifest, timeline, "visual_understanding")
    manifest["coverage"] = _coverage_audit(timeline)
    manifest["quality_audit"] = _quality_audit(timeline)
    manifest["repair_status"] = build_repair_status(manifest, timeline)
    timeline_diff = timeline_vision_diff(before_timeline, timeline, applied)
    report_path = root / "multimodal-frame-analysis-report.md"
    report_path.write_text(_render_report(root, candidates, summary, template_path, "Multimodal Frame Analysis", results=results), encoding="utf-8")
    run_audit = write_vision_analysis_run_audit(
        root,
        kind="semantic_frame",
        summary=summary,
        results=results,
        report_path=report_path,
        template_path=template_path,
        timeline_diff=timeline_diff,
    )
    manifest["vision_analysis_runs"] = "vision-analysis-runs.md"
    manifest["vision_analysis_runs_jsonl"] = "vision-analysis-runs.jsonl"
    write_json(manifest_path, manifest)
    post_run_refresh = _refresh_post_vision_outputs(root, updated_count=int(summary.get("updated") or 0))
    restore_hint = build_vision_restore_hint(root, run_audit)
    run_registry = _register_vision_analysis_run(
        root,
        run_type="multimodal_frame_analysis",
        run_id="multimodal-frame-analysis",
        title="Multimodal frame analysis",
        command_name="run-multimodal-frame-analysis",
        summary=summary,
        candidates=candidates,
        results=results,
        report_path=report_path,
        template_path=template_path,
        run_audit=run_audit,
    )
    return {
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "timeline_path": str(timeline_path),
        "report_path": str(report_path),
        "input_template_json": str(template_path),
        "run_audit": run_audit,
        "run_registry": run_registry,
        "post_run_refresh": post_run_refresh,
        "progress_path": str(progress_path),
        "vision_restore_hint": restore_hint,
        "summary": summary,
        "items": results,
    }


def _register_vision_analysis_run(
    root: Path,
    *,
    run_type: str,
    run_id: str,
    title: str,
    command_name: str,
    summary: dict[str, Any],
    candidates: list[dict[str, Any]],
    results: list[dict[str, Any]],
    report_path: str | Path,
    template_path: str | Path,
    run_audit: dict[str, Any],
) -> dict[str, Any]:
    selected_count = int(summary.get("selected") if summary.get("selected") is not None else summary.get("total") or len(candidates))
    status = _vision_run_registry_status(summary, results, selected_count)
    failed_items = _vision_run_failed_items(summary, results, execute=bool(summary.get("execute")))
    selected_indexes = [int(row.get("index")) for row in candidates if isinstance(row, dict) and _int_value(row.get("index"))]
    retry_indexes = [str(row.get("index")) for row in failed_items if row.get("index")]
    retry_suffix = f" --indexes {','.join(retry_indexes)}" if retry_indexes else ""
    retry_command = f".\\scripts\\video-knowledge.ps1 {command_name} {_quote_ps_path(root)}{retry_suffix}"
    artifacts = [
        {"key": "report", "path": str(report_path), "description": "Human-readable vision analysis report."},
        {"key": "input_template", "path": str(template_path), "description": "JSON template for manual/model import."},
        {"key": "vision_analysis_runs", "path": str(root / "vision-analysis-runs.md")},
        {"key": "vision_analysis_runs_jsonl", "path": str(root / "vision-analysis-runs.jsonl")},
    ]
    if run_audit.get("markdown_path"):
        artifacts.append({"key": "run_audit_markdown", "path": str(run_audit.get("markdown_path"))})
    if run_audit.get("jsonl_path"):
        artifacts.append({"key": "run_audit_jsonl", "path": str(run_audit.get("jsonl_path"))})
    return register_bundle_run(
        root,
        run_type=run_type,
        run_id=run_id,
        status=status,
        title=title,
        summary=(
            f"Selected {selected_count} items; execute={bool(summary.get('execute'))}; "
            f"updated={int(summary.get('updated') or 0)}; status={summary.get('status') or 'ok'}."
        ),
        inputs={
            "selected_indexes": selected_indexes,
            "requested_indexes": summary.get("indexes") or [],
        },
        parameters={
            "execute": bool(summary.get("execute")),
            "limit": int(summary.get("limit") or 0),
            "frame_count": int(summary.get("frame_count") or 0),
            "image_probe_max_edge": int(summary.get("image_probe_max_edge") or 0),
            "image_probe_jpeg_quality": int(summary.get("image_probe_jpeg_quality") or 0),
            "vision_retries": int(summary.get("vision_retries") or 1),
            "provider": (summary.get("provider") or {}).get("provider") if isinstance(summary.get("provider"), dict) else "",
            "model": (summary.get("provider") or {}).get("model") if isinstance(summary.get("provider"), dict) else "",
        },
        artifacts=artifacts,
        failed_items=failed_items,
        retry_command=retry_command,
        next_actions=_vision_run_next_actions(status, command_name) + (
            [f"Pending {summary['continuation']['pending_count']} frames; next task: {summary['continuation']['task_path']}"]
            if (summary.get("continuation") or {}).get("pending_count") else []
        ),
        operator_boundary={
            "preview_first": True,
            "cloud_call_requires_execute": True,
            "cloud_call_requires_confirmation": True,
            "api_key_not_recorded": True,
            "no_download": True,
            "no_publish": True,
            "purpose": "Expose vision batch run status, artifacts, failures, and retry commands to VKP workbench/task console.",
        },
        write=True,
    )


def _vision_run_registry_status(summary: dict[str, Any], results: list[dict[str, Any]], selected_count: int) -> str:
    if selected_count <= 0:
        return "not_needed"
    gate_status = str(summary.get("status") or "ok")
    if gate_status in {"partial_failure", "failed", "vision_batch_aborted"}:
        return "needs_retry"
    if gate_status not in {"", "ok", "preview_only"}:
        return "needs_input"
    if not bool(summary.get("execute")) and int(summary.get("imported") or 0) <= 0:
        return "needs_execution"
    if any(str(row.get("error") or "") for row in results if isinstance(row, dict)):
        return "needs_retry"
    if (summary.get("continuation") or {}).get("pending_count"):
        return "needs_execution"
    if int(summary.get("updated") or 0) > 0 or int(summary.get("imported") or 0) > 0:
        return "completed"
    if bool(summary.get("execute")):
        return "needs_retry"
    return "needs_execution"


def _vision_batch_execution_status(results: list[dict[str, Any]]) -> str:
    blocked = [row for row in results if isinstance(row, dict) and row.get('status') == 'input_identity_blocked']
    executed = [row for row in results if isinstance(row, dict) and row.get("executed")]
    if not executed:
        return "input_identity_blocked" if blocked else "not_needed"
    failed_count = sum(1 for row in executed if not row.get("ok")) + len(blocked)
    complete_count = sum(1 for row in executed if row.get("complete"))
    if failed_count <= 0:
        return "ok"
    return "partial_failure" if complete_count > 0 else "failed"


def _vision_run_failed_items(summary: dict[str, Any], results: list[dict[str, Any]], *, execute: bool) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    gate_status = str(summary.get("status") or "ok")
    gate_error = str(summary.get("error") or "")
    if gate_status not in {"", "ok", "preview_only"}:
        failed.append({"item": "preflight", "reason": gate_status, "detail": gate_error or "Vision execution gate is not satisfied."})
    for row in results:
        if not isinstance(row, dict):
            continue
        index = row.get("index")
        error = str(row.get("error") or "")
        if error:
            failed.append({"index": index, "reason": "vision_error", "detail": error})
        elif execute and not row.get("executed"):
            failed.append({"index": index, "reason": "not_executed", "detail": "Selected vision item was not executed."})
    return failed


def _vision_run_next_actions(status: str, command_name: str) -> list[str]:
    if status == "needs_execution":
        return [f"Run vision-execution-preflight, then execute {command_name} with confirmed calls/indexes."]
    if status == "needs_input":
        return ["Resolve the preflight gate: provider key, confirmation arguments, or selected indexes."]
    if status == "needs_retry":
        return ["Inspect failed_items and rerun only failed indexes, or send them to human review."]
    if status == "not_needed":
        return ["No selected vision candidates for this run."]
    return ["Refresh video workbench/task console to inspect updated visual evidence."]



def call_vision_model_with_retries(
    *,
    provider_config: dict[str, Any],
    prompt: str,
    image_paths: list[str],
    attempts: int = 1,
    delay_seconds: float = 0.0,
    call_kwargs: dict[str, Any] | None = None,
    call_model: Any | None = None,
) -> dict[str, Any]:
    max_attempts = max(1, int(attempts or 1))
    rows: list[dict[str, Any]] = []
    response: dict[str, Any] = {"ok": False, "error": "not_attempted", "content": ""}
    model_caller = call_model or call_vision_model
    for attempt in range(1, max_attempts + 1):
        response = model_caller(provider_config=provider_config, prompt=prompt, image_paths=image_paths, **dict(call_kwargs or {}))
        error = str(response.get("error") or "")
        rows.append({"attempt": attempt, "ok": bool(response.get("ok")), "error": error})
        if response.get("ok") or not _is_retryable_vision_error(error) or attempt >= max_attempts:
            break
        if float(delay_seconds or 0) > 0:
            retry_delay = float(delay_seconds or 0) * (2 ** (attempt - 1))
            rows[-1]["retry_delay_seconds"] = retry_delay
            time.sleep(retry_delay)
    response = dict(response)
    response["attempts"] = rows
    response["attempt_count"] = len(rows)
    response["retried"] = len(rows) > 1
    return response


def _is_retryable_vision_error(error: str) -> bool:
    lowered = str(error or "").lower()
    return any(
        marker in lowered
        for marker in [
            "unexpected_eof",
            "eof occurred",
            "ssl",
            "tls",
            "timed out",
            "timeout",
            "connection reset",
            "temporarily unavailable",
            "remote end closed",
        ]
    )


def _is_batch_terminal_vision_error(error: str) -> bool:
    """Errors that make further media uploads in this batch unsafe or futile."""
    lowered = str(error or "").lower()
    return any(
        marker in lowered
        for marker in (
            "user location is not supported",
            "failed_precondition",
            "quota exceeded",
            "rate limit",
            "ratelimiterror",
            "invalid api key",
            "authentication",
            "permission denied",
            "forbidden",
        )
    )


def _batch_abort_error(error: str) -> str:
    lowered = str(error or "").lower()
    if "user location is not supported" in lowered or "failed_precondition" in lowered:
        return "vision_batch_aborted_provider_location_unsupported"
    if "quota exceeded" in lowered or "rate limit" in lowered or "ratelimiterror" in lowered:
        return "vision_batch_aborted_provider_rate_limited"
    if "invalid api key" in lowered or "authentication" in lowered or "permission denied" in lowered or "forbidden" in lowered:
        return "vision_batch_aborted_provider_authentication_failed"
    return "vision_batch_aborted_provider_terminal_error"


def build_vision_restore_hint(bundle_dir: str | Path, run_audit: dict[str, Any]) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    record = run_audit.get("record") if isinstance(run_audit.get("record"), dict) else {}
    run_id = str(record.get("run_id") or "")
    updated_count = int(record.get("updated_count") or 0)
    timeline_diff_count = int(record.get("timeline_diff_count") or 0)
    base = {
        "run_id": run_id,
        "kind": str(record.get("kind") or ""),
        "updated_count": updated_count,
        "timeline_diff_count": timeline_diff_count,
        "audit_jsonl_path": str(run_audit.get("jsonl_path") or ""),
        "audit_markdown_path": str(run_audit.get("markdown_path") or ""),
    }
    if not run_id:
        return {**base, "status": "not_needed", "reason": "missing_run_id"}
    if updated_count <= 0 and timeline_diff_count <= 0:
        return {**base, "status": "not_needed", "reason": "no_timeline_updates"}
    restore_plan_command = f'python -m video_knowledge_pipeline.cli vision-analysis-restore-plan "{root}" --run-id {run_id}'
    restore_apply_dry_run_command = (
        f'python -m video_knowledge_pipeline.cli vision-analysis-apply-restore "{root}" '
        f'--plan-json "{root / "vision-restore-plan.json"}"'
    )
    restore_apply_execute_command = f"{restore_apply_dry_run_command} --execute --confirm-run-id {run_id}"
    return {
        **base,
        "status": "ready",
        "restore_plan_json_path": str(root / "vision-restore-plan.json"),
        "restore_plan_markdown_path": str(root / "vision-restore-plan.md"),
        "restore_plan_command": restore_plan_command,
        "restore_apply_dry_run_command": restore_apply_dry_run_command,
        "restore_apply_execute_command": restore_apply_execute_command,
    }


def _refresh_post_vision_outputs(root: Path, *, updated_count: int) -> dict[str, Any]:
    if updated_count <= 0:
        return {"status": "skipped", "reason": "no_timeline_updates"}
    result: dict[str, Any] = {"status": "ok", "updated_count": updated_count}
    coverage = audit_knowledge_coverage(root, write=True)
    result["knowledge_coverage_path"] = coverage.get("coverage_path", "")
    result["knowledge_coverage_markdown_path"] = coverage.get("coverage_markdown_path", "")
    # Import lazily to avoid a module cycle: bundle_status -> bundle_next -> multimodal_frame_analyzer.
    from .bundle_status import bundle_status_report, controlled_execution_check

    status = bundle_status_report(root, refresh=False, write=True)
    controlled = controlled_execution_check(root, refresh=False, write=True)
    result["bundle_status_report_path"] = status.get("report_path", "")
    result["bundle_status_report_markdown_path"] = status.get("report_markdown_path", "")
    result["controlled_execution_check_path"] = controlled.get("report_path", "")
    result["controlled_execution_check_markdown_path"] = controlled.get("report_markdown_path", "")
    result["ready_for_real_vision_execution"] = bool(controlled.get("ready_for_real_vision_execution"))
    freshness_receipt = {
        "schema": "video_knowledge_pipeline.knowledge_export_freshness_receipt.v1",
        "status": "stale",
        "reason": "timeline_visual_evidence_updated_after_last_export",
        "updated_timeline_item_count": int(updated_count),
        "automatic_export_performed": False,
        "recovery_command": f".\\scripts\\video-knowledge.ps1 export-knowledge-note {_quote_ps_path(root)}",
        "written_at": now_iso(),
    }
    freshness_path = root / "knowledge-export-freshness.json"
    write_json(freshness_path, freshness_receipt)
    result["knowledge_export_freshness"] = "stale"
    result["knowledge_export_freshness_path"] = str(freshness_path)
    result["knowledge_export_recovery_command"] = freshness_receipt["recovery_command"]
    return result


def _new_vision_batch_progress(
    root: Path,
    *,
    kind: str,
    candidates: list[dict[str, Any]],
    execute: bool,
    gate: dict[str, Any] | None,
) -> dict[str, Any]:
    progress = {
        "schema": "video_knowledge_pipeline.vision_batch_progress.v1",
        "kind": kind,
        "bundle_dir": str(root),
        "execute": bool(execute),
        "status": "blocked" if gate else "running" if execute else "planned",
        "total": len(candidates),
        "selected_indexes": [
            _int_value(row.get("index"))
            for row in candidates
            if isinstance(row, dict) and _int_value(row.get("index")) > 0
        ],
        "current_index": 0,
        "current_position": 0,
        "processed_count": 0,
        "complete_count": 0,
        "truncated_count": 0,
        "failed_count": 0,
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "elapsed_seconds": 0.0,
        "_started_perf_counter": time.perf_counter(),
    }
    _write_vision_batch_progress(root, progress, results=[], status=str(progress["status"]))
    return progress


def _write_vision_batch_progress(
    root: Path,
    progress: dict[str, Any],
    *,
    results: list[dict[str, Any]],
    current_index: int = 0,
    current_position: int = 0,
    status: str,
) -> Path:
    progress.update(
        {
            "status": str(status or "unknown"),
            "current_index": max(0, int(current_index or 0)),
            "current_position": max(0, int(current_position or 0)),
            "processed_count": len(results),
            "complete_count": sum(1 for row in results if row.get("complete")),
            "truncated_count": sum(1 for row in results if row.get("truncated")),
            "failed_count": sum(
                1 for row in results if row.get("executed") and not row.get("ok")
            ),
            "updated_at": now_iso(),
            "elapsed_seconds": round(
                time.perf_counter() - float(progress.get("_started_perf_counter") or time.perf_counter()),
                3,
            ),
        }
    )
    kind = str(progress.get("kind") or "vision").replace("_", "-")
    path = root / f"{kind}-analysis-progress.json"
    write_json(path, {key: value for key, value in progress.items() if not key.startswith("_")})
    if progress.get("execute"):
        print(
            "[vkp][vision-progress] "
            f"kind={progress.get('kind')} status={progress.get('status')} "
            f"position={progress.get('current_position')}/{progress.get('total')} "
            f"index={progress.get('current_index') or '-'} "
            f"complete={progress.get('complete_count')} "
            f"truncated={progress.get('truncated_count')} failed={progress.get('failed_count')}",
            flush=True,
        )
    return path


def vision_analysis_run_log(bundle_dir: str | Path) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    jsonl_path = root / "vision-analysis-runs.jsonl"
    markdown_path = root / "vision-analysis-runs.md"
    rows = read_jsonl(jsonl_path) if jsonl_path.exists() else []
    reconciliation = _vision_run_log_reconciliation(root, rows)
    return {
        "bundle_dir": str(root),
        "jsonl_path": str(jsonl_path),
        "markdown_path": str(markdown_path),
        "count": len(rows),
        "last_run": rows[-1] if rows else {},
        "reconciliation": reconciliation,
        "runs": rows,
    }


def _vision_run_log_reconciliation(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    timeline = _read_timeline(root) if (root / "timeline.json").is_file() else []
    evidence_indexes = evidence_index_sets(timeline)
    completed_events: list[int] = []
    for row in rows:
        if not isinstance(row, dict) or str(row.get("kind") or "") != "semantic_frame":
            continue
        completed_events.extend(_record_complete_indexes(row))
    completed_unique = sorted(set(completed_events))
    timeline_complete = evidence_indexes["visual_model_complete"]
    export_consumable = evidence_indexes["visual_export_consumable"]
    return {
        "schema": "video_knowledge_pipeline.vision_run_reconciliation.v1",
        "run_complete_event_count": len(completed_events),
        "run_complete_unique_count": len(completed_unique),
        "run_complete_unique_indexes": completed_unique,
        "duplicate_run_complete_event_count": len(completed_events) - len(completed_unique),
        "timeline_complete_count": len(timeline_complete),
        "timeline_complete_indexes": timeline_complete,
        "export_consumable_count": len(export_consumable),
        "export_consumable_indexes": export_consumable,
        "run_complete_not_in_timeline_indexes": sorted(
            set(completed_unique) - set(timeline_complete)
        ),
        "timeline_complete_without_logged_run_indexes": sorted(
            set(timeline_complete) - set(completed_unique)
        ),
    }


def _record_complete_indexes(row: dict[str, Any]) -> list[int]:
    explicit = [
        int(value)
        for value in row.get("complete_indexes") or []
        if _int_value(value) > 0
    ]
    if explicit:
        return explicit
    diffs = [
        int(value.get("index"))
        for value in row.get("timeline_diff") or []
        if isinstance(value, dict) and _int_value(value.get("index")) > 0
    ]
    if diffs:
        return diffs
    selected = [
        int(value)
        for value in row.get("candidate_indexes") or []
        if _int_value(value) > 0
    ]
    if int(row.get("complete_count") or 0) == len(selected):
        return selected
    return []


def vision_analysis_restore_plan(
    bundle_dir: str | Path,
    *,
    run_id: str = "",
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    rows = read_jsonl(root / "vision-analysis-runs.jsonl")
    selected = _select_vision_run(rows, run_id=run_id)
    timeline = _read_timeline(root)
    operations = _restore_operations(selected, timeline) if selected else []
    restorable = [operation for operation in operations if operation.get("restorable")]
    missing = [operation for operation in operations if not operation.get("restorable")]
    plan = {
        "schema": "lecture_vision_restore_plan.v1",
        "bundle_dir": str(root),
        "run_id": str(selected.get("run_id") or "") if selected else "",
        "kind": str(selected.get("kind") or "") if selected else "",
        "created_at": str(selected.get("created_at") or "") if selected else "",
        "status": "ready" if restorable and not missing else "partial" if restorable else "not_restorable",
        "write": bool(write),
        "operations_count": len(operations),
        "restorable_count": len(restorable),
        "not_restorable_count": len(missing),
        "operations": operations,
        "notes": [
            "This is a human-review restore plan. It does not modify timeline.json.",
            "Only sanitized controlled vision fields are restorable. Prompts, API keys, and raw model responses are never stored.",
        ],
    }
    json_path = root / "vision-restore-plan.json"
    markdown_path = root / "vision-restore-plan.md"
    if write:
        write_json(json_path, plan)
        markdown_path.write_text(_render_vision_restore_plan_markdown(root, plan), encoding="utf-8")
    plan["json_path"] = str(json_path)
    plan["markdown_path"] = str(markdown_path)
    return plan


def vision_analysis_apply_restore(
    bundle_dir: str | Path,
    *,
    plan_json: str | Path | None = None,
    execute: bool = False,
    confirm_run_id: str = "",
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"bundle missing manifest.json: {root}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")
    plan_path = Path(plan_json).expanduser().resolve() if plan_json else root / "vision-restore-plan.json"
    plan = read_json(plan_path) if plan_path.exists() else {}
    if not isinstance(plan, dict):
        raise ValueError("restore plan JSON must contain an object")
    timeline = _read_timeline(root)
    run_id = str(plan.get("run_id") or "")
    gate = _restore_execution_gate(plan, execute=execute, confirm_run_id=confirm_run_id)
    operations = plan.get("operations") if isinstance(plan.get("operations"), list) else []
    applied = _apply_restore_operations(timeline, operations) if execute and not gate else []
    if execute and not gate:
        write_json(timeline_path, timeline)
        _sync_source_package_restore(manifest, timeline, applied)
        manifest["coverage"] = _coverage_audit(timeline)
        manifest["quality_audit"] = _quality_audit(timeline)
        manifest["repair_status"] = build_repair_status(manifest, timeline)
        manifest["vision_restore_runs"] = "vision-restore-runs.md"
        manifest["vision_restore_runs_jsonl"] = "vision-restore-runs.jsonl"
        write_json(manifest_path, manifest)
    summary = {
        "schema": "lecture_vision_restore_apply_summary.v1",
        "bundle_dir": str(root),
        "plan_json": str(plan_path),
        "run_id": run_id,
        "execute": bool(execute),
        "status": str(gate.get("status") or "ok") if gate else "ok",
        "error": str(gate.get("error") or "") if gate else "",
        "operations_count": len(operations),
        "applied_count": len(applied),
        "applied": applied,
        "updated_at": now_iso(),
    }
    audit = write_vision_restore_apply_audit(root, summary=summary)
    return {
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "timeline_path": str(timeline_path),
        "plan_json": str(plan_path),
        "summary": summary,
        "audit": audit,
    }


def write_multimodal_frame_input_template(root: str | Path, candidates: list[dict[str, Any]]) -> Path:
    path = Path(root) / "multimodal-frame-analysis-input-template.json"
    payload = {
        "schema": "lecture_multimodal_frame_analysis_input.v1",
        "items": [
            {
                "index": item.get("index"),
                "visual_understanding": {
                    "objects": [],
                    "actions": [],
                    "interface_state": "",
                    "spatial_relations": [],
                    "instructor_focus": "",
                    "non_text_information": [],
                    "confidence": 0.0,
                    "keep_image_reason": "",
                    "evidence_frame_paths": item.get("frame_paths", []),
                },
            }
            for item in candidates
        ],
    }
    write_json(path, payload)
    return path


def _candidates(
    root: Path,
    timeline: list[dict[str, Any]],
    *,
    explicit_indexes: list[int] | None = None,
) -> list[dict[str, Any]]:
    items = []
    explicitly_requested = {int(value) for value in explicit_indexes or [] if int(value) > 0}
    for index, item in enumerate(timeline, start=1):
        route = str(item.get("visual_route") or "")
        if route not in {"semantic_frame", "mixed"} and index not in explicitly_requested:
            continue
        if _has_valid_understanding(item.get("visual_understanding")):
            continue
        frame_paths = [_resolve_frame(root, path) for path in _frame_paths(item)]
        items.append(
            {
                "index": index,
                "start": item.get("start", 0),
                "end": item.get("end", 0),
                "visual_route": route,
                "frame_paths": frame_paths,
                "transcript": str(item.get("transcript") or ""),
                "visual_text": str(item.get("visual_text") or ""),
                "existing_visual_understanding": item.get("visual_understanding") or {},
            }
        )
    return items


def _select_candidates(candidates: list[dict[str, Any]], *, limit: int = 0, indexes: list[int] | None = None) -> list[dict[str, Any]]:
    selected = candidates
    wanted = {int(index) for index in indexes or [] if int(index) > 0}
    if wanted:
        selected = [item for item in selected if int(item.get("index") or 0) in wanted]
    max_items = max(0, int(limit or 0))
    if max_items:
        selected = selected[:max_items]
    return selected


def _execution_gate(
    root: Path,
    execute: bool,
    cfg: dict[str, Any],
    *,
    semantic_limit: int,
    temporal_limit: int,
    frame_count: int,
    include_semantic: bool,
    include_temporal: bool,
    semantic_indexes: list[int] | None = None,
    temporal_indexes: list[int] | None = None,
    confirm_vision_calls: int | None = None,
    confirm_vision_indexes: str = "",
    image_probe_max_edge: int = 512,
    image_probe_jpeg_quality: int = 55,
    execution_actor: str = "operator",
    export_consent: str | Path | None = None,
) -> dict[str, Any]:
    return _execution_control(
        root,
        execute,
        cfg,
        semantic_limit=semantic_limit,
        temporal_limit=temporal_limit,
        frame_count=frame_count,
        include_semantic=include_semantic,
        include_temporal=include_temporal,
        semantic_indexes=semantic_indexes,
        temporal_indexes=temporal_indexes,
        confirm_vision_calls=confirm_vision_calls,
        confirm_vision_indexes=confirm_vision_indexes,
        image_probe_max_edge=image_probe_max_edge,
        image_probe_jpeg_quality=image_probe_jpeg_quality,
        execution_actor=execution_actor,
        export_consent=export_consent,
    )[0]


def _execution_control(
    root: Path,
    execute: bool,
    cfg: dict[str, Any],
    *,
    semantic_limit: int,
    temporal_limit: int,
    frame_count: int,
    include_semantic: bool,
    include_temporal: bool,
    semantic_indexes: list[int] | None = None,
    temporal_indexes: list[int] | None = None,
    confirm_vision_calls: int | None = None,
    confirm_vision_indexes: str = "",
    image_probe_max_edge: int = 512,
    image_probe_jpeg_quality: int = 55,
    execution_actor: str = "operator",
    export_consent: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not execute:
        return {}, {
            "schema": "lecture_vision_execution_control.v1",
            "execute": False,
            "preflight_required": False,
            "status": "preview",
            "confirmed": False,
        }
    preflight = vision_execution_preflight(
        root,
        provider_config=cfg,
        semantic_limit=semantic_limit,
        temporal_limit=temporal_limit,
        frame_count=frame_count,
        include_semantic=include_semantic,
        include_temporal=include_temporal,
        semantic_indexes=semantic_indexes,
        temporal_indexes=temporal_indexes,
        write=True,
    )
    expected_calls = int(preflight.get("expected_api_calls") or 0)
    expected_indexes = _preflight_selected_index_string(preflight)
    received_indexes = str(confirm_vision_indexes or "")
    actor = str(execution_actor or "operator").strip().lower()
    if actor not in {"operator", "agent"}:
        actor = "agent"
    selected = preflight.get("selected_indexes") if isinstance(preflight.get("selected_indexes"), dict) else {}
    semantic_selected = [int(value) for value in selected.get("semantic") or []]
    temporal_selected = [int(value) for value in selected.get("temporal") or []]
    consent_status = (
        validate_vision_export_consent(
            export_consent,
            bundle_dir=root,
            provider_config=cfg,
            semantic_indexes=semantic_selected,
            temporal_indexes=temporal_selected,
            expected_calls=expected_calls,
            image_max_edge=image_probe_max_edge,
            image_jpeg_quality=image_probe_jpeg_quality,
        )
        if export_consent
        else {"status": "not_provided", "valid": False, "blockers": []}
    )
    control = {
        "schema": "lecture_vision_execution_control.v1",
        "execute": True,
        "preflight_required": True,
        "preflight_path": str(preflight.get("preflight_path") or ""),
        "preflight_json_path": str(preflight.get("preflight_json_path") or ""),
        "ready_to_execute": bool(preflight.get("ready_to_execute")),
        "expected_api_calls": expected_calls,
        "expected_indexes": expected_indexes,
        "received_confirm_vision_calls": confirm_vision_calls,
        "received_confirm_vision_indexes": received_indexes,
        "execution_actor": actor,
        "export_consent": consent_status,
        "platform_policy_may_still_block": True,
        "confirmed": False,
        "status": "",
    }
    if provider_requires_api_key(cfg) and not cfg.get("api_key"):
        control["status"] = "vision_provider_not_ready"
        control["error"] = "missing_api_key"
        return {
            "status": "vision_provider_not_ready",
            "error": "missing_api_key",
            "message": "Set the provider API key environment variable or pass provider_config with api_key, then retry with execute=true.",
            "preflight_path": str(preflight.get("preflight_path") or ""),
            "preflight_json_path": str(preflight.get("preflight_json_path") or ""),
        }, control
    if not preflight.get("ready_to_execute"):
        blockers = preflight.get("blockers") if isinstance(preflight.get("blockers"), list) else []
        control["status"] = "vision_preflight_blocked"
        control["error"] = ",".join(str(item.get("key") or "") for item in blockers if isinstance(item, dict))
        return {
            "status": "vision_preflight_blocked",
            "error": ",".join(str(item.get("key") or "") for item in blockers if isinstance(item, dict)),
            "message": "Inspect vision-execution-preflight.md, fix blockers, then retry execute=true.",
            "preflight_path": str(preflight.get("preflight_path") or ""),
            "preflight_json_path": str(preflight.get("preflight_json_path") or ""),
        }, control
    if actor == "agent" and not consent_status.get("valid"):
        control["status"] = "vision_export_consent_required"
        control["error"] = "vision_export_consent_invalid"
        return {
            "status": "vision_export_consent_required",
            "error": "vision_export_consent_invalid",
            "message": "Agent execution requires an active bundle/provider/index-scoped export consent. This project gate cannot override a platform-level data export refusal.",
            "consent_status": consent_status,
            "expected_api_calls": expected_calls,
            "expected_indexes": expected_indexes,
            "fallbacks": _vision_execution_fallbacks(root, preflight),
        }, control
    calls_ok = confirm_vision_calls is not None and int(confirm_vision_calls) == expected_calls
    indexes_ok = received_indexes.strip() == expected_indexes
    if calls_ok and indexes_ok:
        control["status"] = "confirmed"
        control["confirmed"] = True
        return {}, control
    control["status"] = "vision_confirmation_required"
    control["error"] = "confirm_vision_mismatch"
    return {
        "status": "vision_confirmation_required",
        "error": "confirm_vision_mismatch",
        "message": "Retry with confirm_vision_calls and confirm_vision_indexes exactly matching the preflight output.",
        "preflight_path": str(preflight.get("preflight_path") or ""),
        "preflight_json_path": str(preflight.get("preflight_json_path") or ""),
        "expected_api_calls": expected_calls,
        "expected_indexes": expected_indexes,
    }, control


def _vision_execution_fallbacks(root: Path, preflight: dict[str, Any]) -> dict[str, str]:
    commands = preflight.get("commands") if isinstance(preflight.get("commands"), dict) else {}
    manual = str(commands.get("confirmed_run_semantic") or commands.get("confirmed_run_temporal") or "")
    return {
        "visible_powershell": manual,
        "local_vlm": f"Set LECTURE_VISION_PROVIDER=local_vlm and rerun the same preflighted indexes for {root}",
        "note": "Use a fallback only when the agent platform blocks the already-consented network export; do not broaden the frame scope.",
    }


def _preflight_selected_index_string(preflight: dict[str, Any]) -> str:
    selected = preflight.get("selected_indexes") if isinstance(preflight.get("selected_indexes"), dict) else {}
    indexes: list[int] = []
    for key in ("semantic", "temporal"):
        for value in selected.get(key) or []:
            try:
                indexes.append(int(value))
            except (TypeError, ValueError):
                continue
    return ",".join(str(index) for index in indexes)


def _has_mapping(value: object) -> bool:
    return isinstance(value, dict) and any(item not in (None, "", [], {}) for item in value.values())


def _has_valid_understanding(value: object) -> bool:
    if not _has_mapping(value):
        return False
    if not isinstance(value, dict):
        return False
    if value.get("parse_failed") is True:
        return False
    if str(value.get("validation_status") or "").strip().lower() == "incomplete":
        return False
    return True


def _prompt(candidate: dict[str, Any]) -> str:
    return (
        "你在处理知识类讲解视频。请真正观察截图，不要只总结字幕/OCR。"
        "输出 JSON，字段包括 objects, actions, interface_state, spatial_relations, "
        "instructor_focus, non_text_information, confidence, keep_image_reason。"
        "必须保留截图证据路径，能降维成文字的信息写清楚，必须保留图片/表格/图像关系的说明原因。\n\n"
        f"时间轴: {candidate.get('start')} - {candidate.get('end')}\n"
        f"字幕: {candidate.get('transcript')}\n"
        f"OCR: {candidate.get('visual_text')}\n"
        f"截图: {candidate.get('frame_paths')}"
    )


def _read_import(input_json: str | Path, field: str) -> list[dict[str, Any]]:
    data = read_json(Path(input_json).expanduser())
    rows = data.get("items") or data.get("results") or data.get(field) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("analysis input JSON must be a list or an object with items/results")
    return [row for row in rows if isinstance(row, dict)]


def _apply_imports(timeline: list[dict[str, Any]], rows: list[dict[str, Any]], *, field: str) -> list[int]:
    updated = []
    for row in rows:
        index = _int_value(row.get("index"))
        if not (1 <= index <= len(timeline)):
            continue
        payload = row.get(field) if isinstance(row.get(field), dict) else row
        candidate = {"frame_paths": _frame_paths(timeline[index - 1])}
        _apply_single(timeline, index, field, _normalise_visual_understanding(payload, candidate))
        updated.append(index)
    return updated


def _normalise_visual_understanding(payload: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result = {
        "schema": "lecture_visual_understanding.v1",
        "objects": _list(payload.get("objects")),
        "actions": _list(payload.get("actions")),
        "interface_state": str(payload.get("interface_state") or payload.get("ui_state") or ""),
        "spatial_relations": _list(payload.get("spatial_relations")),
        "instructor_focus": str(payload.get("instructor_focus") or payload.get("teacher_focus") or ""),
        "non_text_information": _list(payload.get("non_text_information") or payload.get("visual_information")),
        "confidence": _float_value(payload.get("confidence")),
        "keep_image_reason": str(payload.get("keep_image_reason") or payload.get("image_retention_reason") or ""),
        "evidence_frame_paths": _normalise_evidence_frame_paths(payload.get("evidence_frame_paths"), candidate),
        "source": str(payload.get("source") or "multimodal_frame_analyzer"),
        "executor": str(payload.get("executor") or ""),
        "model_requested": str(payload.get("model_requested") or ""),
        "model_actual": str(payload.get("model_actual") or ""),
        "evidence_level": str(payload.get("evidence_level") or ""),
        "parse_failed": bool(payload.get("_parse_failed")),
        "raw_model_output": str(payload.get("raw_content") or "") if payload.get("_parse_failed") else "",
        "updated_at": now_iso(),
    }
    issues = _visual_understanding_issues(result)
    result["validation_status"] = "incomplete" if issues else "ok"
    result["validation_issues"] = issues
    return result


def _normalise_evidence_frame_paths(values: Any, candidate: dict[str, Any]) -> list[str]:
    candidate_paths = [str(path).strip() for path in _list(candidate.get("frame_paths")) if str(path).strip()]
    raw_values = [str(path).strip() for path in _list(values) if str(path).strip()]
    if not raw_values:
        return candidate_paths

    lookup: dict[str, str] = {}
    for path in candidate_paths:
        name = Path(path).name
        lookup[path] = path
        lookup[name] = path
        lookup[name.lower()] = path

    resolved: list[str] = []
    for value in raw_values:
        name = Path(value).name
        mapped = lookup.get(value) or lookup.get(name) or lookup.get(name.lower()) or value
        if mapped not in resolved:
            resolved.append(mapped)
    return resolved or candidate_paths


def _apply_single(timeline: list[dict[str, Any]], index: int, field: str, payload: dict[str, Any]) -> None:
    item = timeline[index - 1]
    item[field] = payload
    item[f"{field}_updated_at"] = now_iso()
    issues = [
        issue
        for issue in item.get("quality_issues", [])
        if issue not in {"missing_visual_understanding", "semantic_frame_without_analysis"}
    ]
    if payload.get("parse_failed") and "model_output_parse_failed" not in issues:
        issues.append("model_output_parse_failed")
    if payload.get("validation_status") == "incomplete":
        issue_key = "visual_understanding_incomplete" if field == "visual_understanding" else "temporal_understanding_incomplete"
        if issue_key not in issues:
            issues.append(issue_key)
    item["quality_issues"] = issues
    item["integrated_visual"] = integrated_visual(item)


def _visual_understanding_issues(payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if payload.get("parse_failed"):
        issues.append("parse_failed")
    has_content = bool(
        _list(payload.get("objects"))
        or _list(payload.get("actions"))
        or str(payload.get("interface_state") or "").strip()
        or _list(payload.get("spatial_relations"))
        or _list(payload.get("non_text_information"))
        or str(payload.get("instructor_focus") or "").strip()
    )
    if not has_content:
        issues.append("missing_visual_content")
    if not _list(payload.get("evidence_frame_paths")):
        issues.append("missing_evidence_frame_paths")
    return issues


def _sync_source_package(manifest: dict[str, Any], timeline: list[dict[str, Any]], field: str) -> None:
    source = Path(str(manifest.get("source_package") or "")).expanduser()
    if not source.exists() or not source.is_file():
        return
    package = read_json(source)
    if not isinstance(package, dict) or not isinstance(package.get("timeline"), list):
        return
    for index, item in enumerate(package["timeline"], start=1):
        if index <= len(timeline) and isinstance(item, dict) and timeline[index - 1].get(field):
            item[field] = timeline[index - 1][field]
            item[f"{field}_updated_at"] = timeline[index - 1].get(f"{field}_updated_at")
            item["integrated_visual"] = timeline[index - 1].get("integrated_visual") or integrated_visual(timeline[index - 1])
    package["coverage"] = _coverage_audit(package["timeline"])
    package["quality_audit"] = _quality_audit(package["timeline"])
    write_json(source, package)


def _render_report(
    root: Path,
    candidates: list[dict[str, Any]],
    summary: dict[str, Any],
    template_path: Path,
    title: str,
    *,
    results: list[dict[str, Any]] | None = None,
) -> str:
    result_by_index = {
        int(_int_value(item.get("index"))): item
        for item in results or []
        if isinstance(item, dict) and _int_value(item.get("index")) > 0
    }
    lines = [
        f"# {title} Report",
        "",
        f"- Bundle: `{root}`",
        f"- Candidates: {len(candidates)}",
        f"- Template: `{template_path}`",
        f"- Provider: `{(summary.get('provider') or {}).get('provider', '')}` / `{(summary.get('provider') or {}).get('model', '')}`",
        f"- Execute: `{summary.get('execute', False)}`",
        f"- Status: `{summary.get('status', 'ok')}`",
        f"- Mode: `{(summary.get('invocation_check') or {}).get('mode', '')}`",
        f"- Route source: `{(summary.get('provider') or {}).get('provider_config_source', '')}`",
        f"- Route readiness: `{(summary.get('provider') or {}).get('route_resolution_status', '')}`",
        *[f"> {_md_cell(notice)}" for notice in (summary.get('invocation_check') or {}).get('notices', [])],
        f"- Error: `{summary.get('error', '')}`",
        *([
            f"- Remaining semantic frames: {(summary.get('continuation') or {}).get('pending_count')}",
            f"- Next batch task: `{summary['continuation']['task_path']}`",
            f"> {summary['continuation']['instruction']}",
            "", "```powershell",
            summary['continuation']['preflight_command'],
            summary['continuation']['execute_command_template'], "```",
        ] if (summary.get('continuation') or {}).get('pending_count') else []),
        "",
        "## 本次结果审核表",
        "",
        "| Index | Time | Route | Frame | Executed | OK | Validation | Issues | Transcript | OCR / visual text | Model understanding | Keep image reason | Confidence | Evidence |",
        "|---:|---|---|---|---:|---:|---|---|---|---|---|---|---:|---|",
    ]
    for item in candidates:
        result = result_by_index.get(int(_int_value(item.get("index"))), {})
        understanding = _result_understanding(result) or item.get("existing_visual_understanding") or item.get("existing_temporal_visual_understanding") or {}
        validation = _understanding_validation(understanding, result)
        lines.append(
            "| {index} | {time_range} | `{route}` | {frame} | {executed} | {ok} | `{validation}` | {issues} | {transcript} | {visual_text} | {analysis} | {keep_reason} | {confidence} | {evidence} |".format(
                index=item.get("index"),
                time_range=_md_cell(f"{item.get('start', '')}-{item.get('end', '')}"),
                route=_md_cell(str(item.get("visual_route") or "")),
                frame=_md_cell(Path(str((item.get("frame_paths") or [""])[0])).name if item.get("frame_paths") else ""),
                executed=bool(result.get("executed")) if result else False,
                ok=bool(result.get("ok")) if result else False,
                validation=validation["status"],
                issues=_md_cell(", ".join(validation["issues"])),
                transcript=_md_cell(_short_text(item.get("transcript"))),
                visual_text=_md_cell(_short_text(item.get("visual_text"))),
                analysis=_md_cell(_short_text(_understanding_summary(understanding))),
                keep_reason=_md_cell(_short_text((understanding or {}).get("keep_image_reason"))),
                confidence=_confidence_text((understanding or {}).get("confidence")),
                evidence=_md_cell(_short_text(_evidence_summary(understanding, item))),
            )
        )
    lines.extend([""])
    for item in candidates:
        result = result_by_index.get(int(_int_value(item.get("index"))), {})
        understanding = _result_understanding(result) or item.get("existing_visual_understanding") or item.get("existing_temporal_visual_understanding") or {}
        validation = _understanding_validation(understanding, result)
        lines.extend(
            [
                f"## Timeline {item.get('index')}",
                "",
                f"- Route: `{item.get('visual_route', '')}`",
                f"- Executed: `{bool(result.get('executed')) if result else False}`",
                f"- OK: `{bool(result.get('ok')) if result else False}`",
                f"- Error: `{result.get('error', '')}`",
                f"- Validation: `{validation['status']}`",
                f"- Validation issues: `{', '.join(validation['issues'])}`",
                f"- Frames: `{len(item.get('frame_paths') or [])}`",
                "",
                "### Transcript",
                "",
                str(item.get("transcript") or ""),
                "",
                "### OCR / Visual Text",
                "",
                str(item.get("visual_text") or ""),
                "",
                "### Model Understanding",
                "",
                "```json",
                _json_preview(understanding),
                "```",
                "",
            ]
        )
        for path in item.get("frame_paths") or []:
            lines.append(f"- Frame: `{path}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_vision_analysis_run_audit(
    root: str | Path,
    *,
    kind: str,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    report_path: str | Path,
    template_path: str | Path,
    timeline_diff: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    bundle_root = Path(root).expanduser().resolve()
    jsonl_path = bundle_root / "vision-analysis-runs.jsonl"
    markdown_path = bundle_root / "vision-analysis-runs.md"
    record = _vision_analysis_run_record(
        bundle_root,
        kind=kind,
        summary=summary,
        results=results,
        report_path=Path(report_path),
        template_path=Path(template_path),
        timeline_diff=timeline_diff or [],
    )
    append_jsonl(jsonl_path, [record])
    rows = read_jsonl(jsonl_path)
    markdown_path.write_text(_render_vision_analysis_runs_markdown(bundle_root, rows), encoding="utf-8")
    return {
        "record": record,
        "jsonl_path": str(jsonl_path),
        "markdown_path": str(markdown_path),
        "count": len(rows),
    }


def _single_reported_model(results: list[dict[str, Any]]) -> str:
    models = {str(row.get('model_actual') or '') for row in results if row.get('executed')}
    return next(iter(models)) if len(models) == 1 else ''


def _vision_analysis_run_record(
    root: Path,
    *,
    kind: str,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    report_path: Path,
    template_path: Path,
    timeline_diff: list[dict[str, Any]],
) -> dict[str, Any]:
    created_at = now_iso()
    provider = summary.get("provider") if isinstance(summary.get("provider"), dict) else {}
    execution_control = summary.get("execution_control") if isinstance(summary.get("execution_control"), dict) else {}
    evidence_levels = {"confirmed": 0, "inferred": 0, "unavailable": 0, "not_seen": 0}
    for item in results:
        understanding = _result_understanding(item)
        level = str(item.get("evidence_level") or understanding.get("evidence_level") or "").strip().lower()
        if level in {"confirmed", "inferred", "unavailable"}:
            evidence_levels[level] += 1
        else:
            evidence_levels["not_seen"] += 1
    return {
        "schema": "lecture_vision_analysis_run.v1",
        "run_id": f"{kind}-{created_at.replace(':', '').replace('-', '').replace('T', '-')}",
        "created_at": created_at,
        "kind": kind,
        "execute": bool(summary.get("execute")),
        "status": str(summary.get("status") or "ok"),
        "error": str(summary.get("error") or ""),
        "executor": str(summary.get("executor") or provider.get("provider") or ""),
        "model_requested": str(summary.get("model_requested") or provider.get("model") or ""),
        "model_actual": str(summary.get("model_actual") or _single_reported_model(results)),
        "quality": str(summary.get("quality") or ""),
        "execution_mode": str(summary.get("execution_mode") or "provider_api"),
        "provider_health_check": str(summary.get("provider_health_check") or ""),
        "task_pack_sha256": str(summary.get("task_pack_sha256") or ""),
        "task_pack_path": str(summary.get("task_pack_path") or template_path),
        "raw_result_sha256": str(summary.get("raw_result_sha256") or ""),
        "raw_result_path": str(summary.get("raw_result_path") or ""),
        "evidence_levels": evidence_levels,
        "provider": {
            "provider": provider.get("provider", ""),
            "base_url": provider.get("base_url", ""),
            "model": provider.get("model", ""),
            "api_key_configured": bool(provider.get("api_key_configured")),
        },
        "limit": summary.get("limit", 0),
        "frame_count": summary.get("frame_count", 1),
        "selected_count": len(results),
        "executed_count": sum(1 for item in results if item.get("executed")),
        "ok_count": sum(1 for item in results if item.get("ok")),
        "failed_count": sum(1 for item in results if item.get("executed") and not item.get("ok")),
        "skipped_count": sum(1 for item in results if not item.get("executed")),
        "parse_failed_count": sum(1 for item in results if _result_understanding(item).get("parse_failed")),
        "incomplete_count": sum(1 for item in results if _result_understanding(item).get("validation_status") == "incomplete"),
        "complete_count": sum(1 for item in results if item.get("complete")),
        "truncated_count": sum(1 for item in results if item.get("truncated")),
        "complete_indexes": [
            int(item.get("index"))
            for item in results
            if item.get("complete") and _int_value(item.get("index")) > 0
        ],
        "truncated_indexes": [
            int(item.get("index"))
            for item in results
            if item.get("truncated") and _int_value(item.get("index")) > 0
        ],
        "failed_indexes": [
            int(item.get("index"))
            for item in results
            if item.get("executed") and not item.get("ok") and _int_value(item.get("index")) > 0
        ],
        "updated_count": int(summary.get("updated") or 0),
        "timeline_diff_count": len(timeline_diff),
        "timeline_diff": timeline_diff,
        "candidate_indexes": [int(_int_value(item.get("index"))) for item in results if _int_value(item.get("index")) > 0],
        "execution_control": _compact_execution_control(execution_control),
        "report_path": str(report_path),
        "input_template_json": str(template_path),
        "bundle_dir": str(root),
    }


def _render_vision_analysis_runs_markdown(root: Path, rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Vision Analysis Runs",
        "",
        f"- Bundle: `{root}`",
        f"- Runs: `{len(rows)}`",
        "",
        "| Time | Kind | Execute | Status | Provider | Selected | OK | Failed | Parse failed | Incomplete | Updated | Changed | Report |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        provider = row.get("provider") if isinstance(row.get("provider"), dict) else {}
        provider_text = f"{provider.get('provider', '')}/{provider.get('model', '')}"
        lines.append(
            "| {time} | `{kind}` | {execute} | `{status}` | `{provider}` | {selected} | {ok} | {failed} | {parse_failed} | {incomplete} | {updated} | {changed} | `{report}` |".format(
                time=row.get("created_at", ""),
                kind=row.get("kind", ""),
                execute=bool(row.get("execute")),
                status=row.get("status", ""),
                provider=_md_cell(provider_text),
                selected=row.get("selected_count", 0),
                ok=row.get("ok_count", 0),
                failed=row.get("failed_count", 0),
                parse_failed=row.get("parse_failed_count", 0),
                incomplete=row.get("incomplete_count", 0),
                updated=row.get("updated_count", 0),
                changed=row.get("timeline_diff_count", 0),
                report=_md_cell(str(row.get("report_path") or "")),
            )
        )
    if any(row.get("timeline_diff") for row in rows):
        lines.extend(["", "## Timeline Diff", ""])
        for row in rows:
            diffs = row.get("timeline_diff") if isinstance(row.get("timeline_diff"), list) else []
            if not diffs:
                continue
            lines.extend([f"### {row.get('created_at', '')} `{row.get('kind', '')}`", ""])
            for diff in diffs:
                fields = ", ".join(str(field) for field in diff.get("changed_fields") or [])
                lines.append(f"- Timeline `{diff.get('index')}` changed fields: `{_md_cell(fields)}`")
    if any(row.get("execution_control") for row in rows):
        lines.extend(["", "## Execution Control", ""])
        for row in rows:
            control = row.get("execution_control") if isinstance(row.get("execution_control"), dict) else {}
            if not control:
                continue
            lines.append(
                "- `{run_id}` status `{status}`, confirmed `{confirmed}`, expected `{expected}` / `{indexes}`, preflight `{preflight}`".format(
                    run_id=_md_cell(str(row.get("run_id") or "")),
                    status=_md_cell(str(control.get("status") or "")),
                    confirmed=bool(control.get("confirmed")),
                    expected=control.get("expected_api_calls", ""),
                    indexes=_md_cell(str(control.get("expected_indexes") or "")),
                    preflight=_md_cell(str(control.get("preflight_path") or "")),
                )
            )
    return "\n".join(lines).rstrip() + "\n"


def _compact_execution_control(value: dict[str, Any]) -> dict[str, Any]:
    if not value:
        return {}
    keep = [
        "schema",
        "execute",
        "preflight_required",
        "status",
        "error",
        "confirmed",
        "ready_to_execute",
        "expected_api_calls",
        "expected_indexes",
        "received_confirm_vision_calls",
        "received_confirm_vision_indexes",
        "execution_actor",
        "export_consent",
        "platform_policy_may_still_block",
        "preflight_path",
        "preflight_json_path",
    ]
    return {key: value[key] for key in keep if key in value}


VISION_DIFF_FIELDS = (
    "visual_understanding",
    "visual_understanding_updated_at",
    "temporal_visual_understanding",
    "temporal_visual_understanding_updated_at",
    "panorama_visual_understanding",
    "panorama_visual_understanding_updated_at",
    "quality_issues",
    "integrated_visual",
)

VISION_DIFF_DROP_KEYS = {
    "api_key",
    "prompt",
    "raw_content",
    "raw_model_output",
    "response",
}


def snapshot_timeline_for_vision_diff(timeline: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {index: _snapshot_vision_fields(item) for index, item in enumerate(timeline, start=1)}


def timeline_vision_diff(
    before: dict[int, dict[str, Any]],
    after_timeline: list[dict[str, Any]],
    indexes: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index in indexes:
        index = int(index)
        if index in seen or not (1 <= index <= len(after_timeline)):
            continue
        seen.add(index)
        before_item = before.get(index, {})
        after_item = _snapshot_vision_fields(after_timeline[index - 1])
        changes: list[dict[str, Any]] = []
        for field in VISION_DIFF_FIELDS:
            before_value = before_item.get(field)
            after_value = after_item.get(field)
            if before_value == after_value:
                continue
            changes.append(
                {
                    "field": field,
                    "before_empty": not _has_diff_value(before_value),
                    "after_empty": not _has_diff_value(after_value),
                    "before_value": before_value,
                    "after_value": after_value,
                    "before": _diff_preview(before_value),
                    "after": _diff_preview(after_value),
                }
            )
        if changes:
            rows.append(
                {
                    "index": index,
                    "changed_fields": [change["field"] for change in changes],
                    "changes": changes,
                }
            )
    return rows


def _snapshot_vision_fields(item: dict[str, Any]) -> dict[str, Any]:
    return {field: _sanitize_diff_value(item.get(field)) for field in VISION_DIFF_FIELDS if field in item}


def _sanitize_diff_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_diff_value(child)
            for key, child in value.items()
            if str(key).lower() not in VISION_DIFF_DROP_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_diff_value(child) for child in value]
    return value


def _has_diff_value(value: Any) -> bool:
    if value is None or value == "" or value == {} or value == []:
        return False
    return True


def _diff_preview(value: Any, limit: int = 600) -> str:
    if value is None:
        return ""
    import json

    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _select_vision_run(rows: list[dict[str, Any]], *, run_id: str = "") -> dict[str, Any]:
    candidates = [row for row in rows if isinstance(row, dict)]
    if run_id:
        for row in candidates:
            if str(row.get("run_id") or "") == run_id:
                return row
        return {}
    return candidates[-1] if candidates else {}


def _restore_operations(run: dict[str, Any], timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    for diff in run.get("timeline_diff") or []:
        if not isinstance(diff, dict):
            continue
        index = _int_value(diff.get("index"))
        if not (1 <= index <= len(timeline)):
            operations.append(
                {
                    "index": index,
                    "restorable": False,
                    "reason": "timeline_index_missing",
                    "changes": [],
                }
            )
            continue
        changes = []
        for change in diff.get("changes") or []:
            if not isinstance(change, dict):
                continue
            field = str(change.get("field") or "")
            before_value_present = "before_value" in change
            before_empty = bool(change.get("before_empty"))
            changes.append(
                {
                    "field": field,
                    "restorable": bool(field in VISION_DIFF_FIELDS and before_value_present),
                    "action": "remove_field" if before_empty else "set_field",
                    "current": _diff_preview(_sanitize_diff_value(timeline[index - 1].get(field))),
                    "restore_to": _diff_preview(change.get("before_value")),
                    "before_empty": before_empty,
                    "after_empty": bool(change.get("after_empty")),
                    "restore_value": change.get("before_value") if before_value_present else None,
                }
            )
        operations.append(
            {
                "index": index,
                "restorable": bool(changes) and all(change.get("restorable") for change in changes),
                "reason": "" if changes else "no_changes",
                "changed_fields": [change.get("field") for change in changes],
                "changes": changes,
            }
        )
    return operations


def _render_vision_restore_plan_markdown(root: Path, plan: dict[str, Any]) -> str:
    lines = [
        "# Vision Restore Plan",
        "",
        f"- Bundle: `{root}`",
        f"- Run ID: `{plan.get('run_id', '')}`",
        f"- Kind: `{plan.get('kind', '')}`",
        f"- Status: `{plan.get('status', '')}`",
        f"- Restorable: `{plan.get('restorable_count', 0)}`",
        f"- Not restorable: `{plan.get('not_restorable_count', 0)}`",
        "",
        "This file is for human review only. It does not modify `timeline.json`.",
        "",
        "| Index | Restorable | Fields | Reason |",
        "|---:|---:|---|---|",
    ]
    for operation in plan.get("operations") or []:
        fields = ", ".join(str(field) for field in operation.get("changed_fields") or [])
        lines.append(
            "| {index} | {restorable} | `{fields}` | `{reason}` |".format(
                index=operation.get("index", 0),
                restorable=bool(operation.get("restorable")),
                fields=_md_cell(fields),
                reason=_md_cell(str(operation.get("reason") or "")),
            )
        )
    lines.append("")
    for operation in plan.get("operations") or []:
        lines.extend([f"## Timeline {operation.get('index', 0)}", ""])
        for change in operation.get("changes") or []:
            lines.extend(
                [
                    f"### `{change.get('field', '')}`",
                    "",
                    f"- Restorable: `{bool(change.get('restorable'))}`",
                    f"- Action: `{change.get('action', '')}`",
                    f"- Current: `{_md_cell(str(change.get('current') or ''))}`",
                    f"- Restore to: `{_md_cell(str(change.get('restore_to') or ''))}`",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _restore_execution_gate(plan: dict[str, Any], *, execute: bool, confirm_run_id: str) -> dict[str, str]:
    if not execute:
        return {}
    run_id = str(plan.get("run_id") or "")
    if not run_id:
        return {"status": "restore_not_ready", "error": "missing_run_id"}
    if str(plan.get("status") or "") not in {"ready", "partial"}:
        return {"status": "restore_not_ready", "error": "plan_not_restorable"}
    if confirm_run_id != run_id:
        return {"status": "restore_confirmation_required", "error": "confirm_run_id_mismatch"}
    if not plan.get("operations"):
        return {"status": "restore_not_ready", "error": "no_restore_operations"}
    return {}


def _apply_restore_operations(timeline: list[dict[str, Any]], operations: list[Any]) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    for operation in operations:
        if not isinstance(operation, dict) or not operation.get("restorable"):
            continue
        index = _int_value(operation.get("index"))
        if not (1 <= index <= len(timeline)):
            continue
        item = timeline[index - 1]
        fields: list[str] = []
        for change in operation.get("changes") or []:
            if not isinstance(change, dict) or not change.get("restorable"):
                continue
            field = str(change.get("field") or "")
            if field not in VISION_DIFF_FIELDS:
                continue
            if change.get("action") == "remove_field" or change.get("before_empty"):
                item.pop(field, None)
            else:
                item[field] = change.get("restore_value")
            fields.append(field)
        if fields:
            applied.append({"index": index, "fields": fields})
    return applied


def _sync_source_package_restore(manifest: dict[str, Any], timeline: list[dict[str, Any]], applied: list[dict[str, Any]]) -> None:
    source = Path(str(manifest.get("source_package") or "")).expanduser()
    if not source.exists() or not source.is_file():
        return
    package = read_json(source)
    if not isinstance(package, dict) or not isinstance(package.get("timeline"), list):
        return
    for operation in applied:
        index = _int_value(operation.get("index"))
        if not (1 <= index <= len(timeline)) or index > len(package["timeline"]):
            continue
        item = package["timeline"][index - 1]
        if not isinstance(item, dict):
            continue
        source_item = timeline[index - 1]
        for field in operation.get("fields") or []:
            if field in source_item:
                item[field] = source_item[field]
            else:
                item.pop(field, None)
    package["coverage"] = _coverage_audit(package["timeline"])
    package["quality_audit"] = _quality_audit(package["timeline"])
    write_json(source, package)


def write_vision_restore_apply_audit(root: str | Path, *, summary: dict[str, Any]) -> dict[str, Any]:
    bundle_root = Path(root).expanduser().resolve()
    jsonl_path = bundle_root / "vision-restore-runs.jsonl"
    markdown_path = bundle_root / "vision-restore-runs.md"
    record = {
        "schema": "lecture_vision_restore_apply_run.v1",
        "created_at": now_iso(),
        "run_id": str(summary.get("run_id") or ""),
        "execute": bool(summary.get("execute")),
        "status": str(summary.get("status") or "ok"),
        "error": str(summary.get("error") or ""),
        "operations_count": int(summary.get("operations_count") or 0),
        "applied_count": int(summary.get("applied_count") or 0),
        "applied": summary.get("applied") if isinstance(summary.get("applied"), list) else [],
        "plan_json": str(summary.get("plan_json") or ""),
        "bundle_dir": str(bundle_root),
    }
    append_jsonl(jsonl_path, [record])
    rows = read_jsonl(jsonl_path)
    markdown_path.write_text(_render_vision_restore_runs_markdown(bundle_root, rows), encoding="utf-8")
    return {"record": record, "jsonl_path": str(jsonl_path), "markdown_path": str(markdown_path), "count": len(rows)}


def _render_vision_restore_runs_markdown(root: Path, rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Vision Restore Runs",
        "",
        f"- Bundle: `{root}`",
        f"- Runs: `{len(rows)}`",
        "",
        "| Time | Execute | Status | Run ID | Operations | Applied | Error |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {time} | {execute} | `{status}` | `{run_id}` | {operations} | {applied} | `{error}` |".format(
                time=row.get("created_at", ""),
                execute=bool(row.get("execute")),
                status=_md_cell(str(row.get("status") or "")),
                run_id=_md_cell(str(row.get("run_id") or "")),
                operations=row.get("operations_count", 0),
                applied=row.get("applied_count", 0),
                error=_md_cell(str(row.get("error") or "")),
            )
        )
    if any(row.get("applied") for row in rows):
        lines.extend(["", "## Applied Fields", ""])
        for row in rows:
            applied = row.get("applied") if isinstance(row.get("applied"), list) else []
            if not applied:
                continue
            lines.append(f"### {row.get('created_at', '')} `{row.get('run_id', '')}`")
            for item in applied:
                fields = ", ".join(str(field) for field in item.get("fields") or [])
                lines.append(f"- Timeline `{item.get('index')}`: `{_md_cell(fields)}`")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _read_timeline(root: Path) -> list[dict[str, Any]]:
    path = root / "timeline.json"
    data = read_json(path) if path.exists() else []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def _resolve_frame(root: Path, path: str) -> str:
    frame = Path(path).expanduser()
    if frame.is_absolute():
        return str(frame)
    return str((root / frame).resolve())


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    return [value]


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _short_text(value: Any, limit: int = 160) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "..."




def _confidence_text(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return ""


def _result_understanding(result: dict[str, Any]) -> dict[str, Any]:
    for key in ("visual_understanding", "temporal_visual_understanding"):
        value = result.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _understanding_validation(understanding: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    if understanding:
        issues = [str(issue) for issue in understanding.get("validation_issues") or []]
        return {"status": str(understanding.get("validation_status") or "ok"), "issues": issues}
    error = str(result.get("error") or "").strip()
    return {"status": "not_available" if not error else "error", "issues": [error] if error else []}


def _understanding_summary(understanding: dict[str, Any]) -> str:
    if not isinstance(understanding, dict) or not understanding:
        return ""
    parts: list[str] = []
    for key in (
        "objects",
        "actions",
        "interface_state",
        "spatial_relations",
        "instructor_focus",
        "non_text_information",
        "event_sequence",
        "state_changes",
        "operation_steps",
        "causal_links",
        "possible_missing_points",
        "keep_image_reason",
    ):
        value = understanding.get(key)
        if value:
            parts.append(f"{key}: {value}")
    return "; ".join(parts)


def _evidence_summary(understanding: dict[str, Any], candidate: dict[str, Any]) -> str:
    evidence = understanding.get("evidence_frame_paths") if isinstance(understanding, dict) else []
    if not isinstance(evidence, list) or not evidence:
        evidence = candidate.get("frame_paths") if isinstance(candidate.get("frame_paths"), list) else []
    return f"{len(evidence)} frame(s)"


def _json_preview(value: Any) -> str:
    if not value:
        return "{}"
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)
