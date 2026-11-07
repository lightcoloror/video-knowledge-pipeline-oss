from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .config import resolve_vision_execution_profile
from .frame_recapture import _coverage_audit, _quality_audit
from .models import now_iso
from .model_task_gateway import model_task_api_call
from .multimodal_frame_analyzer import (
    _execution_control,
    _float_value,
    _int_value,
    _list,
    _normalise_evidence_frame_paths,
    _register_vision_analysis_run,
    _read_import,
    _read_timeline,
    _refresh_post_vision_outputs,
    _new_vision_batch_progress,
    _render_report,
    _resolve_frame,
    _has_mapping,
    build_vision_restore_hint,
    call_vision_model_with_retries,
    snapshot_timeline_for_vision_diff,
    timeline_vision_diff,
    write_vision_analysis_run_audit,
    _vision_batch_execution_status,
    _write_vision_batch_progress,
)
from .model_runtime_client import authorise_consented_remote_runtime
from .repair_status import build_repair_status
from .storage import read_json, write_json
from .video_frame_router import _frame_paths
from .visual_integration import integrated_visual
from .visual_evidence import evidence_index_sets
from .evidence_lineage import call_visual_with_provenance
from .vision_execution_route import resolve_vision_task_execution_route, check_vision_invocation
from .vision_export_consent import vision_export_consent_image_limits
from .vision_api import parse_model_json, resolve_provider_config
from .vlm_preprocess import prepare_image_probe
from .temporal_frame_preprocess import build_temporal_frame_manifest, prepare_temporal_image_probe


def call_vision_model(
    *, provider_config, prompt, image_paths, allowed_roots=None, max_tokens=None
):
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
    return model_task_api_call("temporal_visual_analysis", **kwargs)


def call_vision_model_with_broker_reservation(
    *, provider_config, prompt, image_paths, allowed_roots=None, max_tokens=None
):
    """Invoke one Proxy request only within the consent-bound runtime grant."""
    config = dict(provider_config or {})
    is_remote_proxy = (
        str(config.get("adapter_backend") or "").strip().lower() == "proxy"
        and str(config.get("execution_location") or "").strip().lower() == "remote"
    )
    call_kwargs = {
        "provider_config": config,
        "prompt": prompt,
        "image_paths": image_paths,
        "allowed_roots": allowed_roots,
    }
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens
    if not is_remote_proxy:
        return call_vision_model(**call_kwargs)
    with authorise_consented_remote_runtime(
        consent_id=str(config.get("consent_id") or ""),
        route_revision=str(config.get("route_revision") or ""),
        max_calls=1,
    ):
        return call_vision_model(**call_kwargs)


def run_temporal_visual_analysis(
    bundle_dir: str | Path,
    *,
    execute: bool = False,
    frame_count: int | None = None,
    limit: int | None = None,
    indexes: list[int] | None = None,
    provider_config: dict[str, Any] | None = None,
    input_json: str | Path | None = None,
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
    route_resolution = resolve_vision_task_execution_route("temporal_sequence", provider_config=provider_config)
    invocation_check = check_vision_invocation(execute=execute, route=route_resolution)
    if execute and route_resolution.get("legacy_fallback_blocked"):
        raise ValueError("No configured temporal gateway route; legacy provider fallback is blocked for execution")
    effective_provider_config = provider_config
    if provider_config is None and not route_resolution.get("legacy_fallback_blocked"):
        effective_provider_config = dict(route_resolution.get("provider_config") or {})
    profile = resolve_vision_execution_profile(
        provider_config=effective_provider_config,
        temporal_limit=limit,
        frame_count=frame_count,
    )
    effective_frame_count = int(profile["frame_count"])
    effective_limit = int(profile["temporal_limit"])
    timeline = _read_timeline(root)
    before_timeline = snapshot_timeline_for_vision_diff(timeline)
    candidates = _select_candidates(_candidates(root, timeline, frame_count=effective_frame_count), limit=effective_limit, indexes=indexes)
    template_path = write_temporal_visual_input_template(root, candidates)

    imported = _read_import(input_json, "temporal_visual_understanding") if input_json else []
    applied = _apply_imports(timeline, imported) if imported else []
    cfg = resolve_provider_config(profile["provider_config"])
    use_local_temporal_preprocess = str(cfg.get("execution_location") or "").strip().lower() == "local"
    for candidate in candidates:
        candidate["strict_frame_contract"] = use_local_temporal_preprocess
        if not candidate.get("frame_manifest"):
            candidate["frame_manifest"] = build_temporal_frame_manifest(candidate.get("frame_paths") or [])
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
        semantic_limit=0,
        temporal_limit=effective_limit,
        frame_count=effective_frame_count,
        include_semantic=False,
        include_temporal=True,
        temporal_indexes=indexes,
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

    results = []
    progress = _new_vision_batch_progress(
        root,
        kind="temporal_sequence",
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
            "finish_reason": "",
            "truncated": False,
            "complete": False,
            "request_max_tokens": int(max_tokens) if max_tokens is not None else None,
            "request_max_tokens_omitted": max_tokens is None,
        }
        if execute and not gate and candidate.get("frame_paths"):
            original_image_paths = [str(path) for path in candidate.get("frame_paths", []) if str(path)]
            preprocess_started = time.perf_counter()
            probe_kwargs = {
                "output_dir": root / "vision-analysis-image-probes" / "temporal" / str(candidate["index"]),
                "max_edge": effective_image_probe_max_edge,
                "jpeg_quality": effective_image_probe_jpeg_quality,
            }
            if use_local_temporal_preprocess:
                image_probe = prepare_temporal_image_probe(
                    original_image_paths,
                    **probe_kwargs,
                    use_contact_sheet=True,
                    representative_limit=2,
                )
            else:
                image_probe = prepare_image_probe(original_image_paths, **probe_kwargs)
            preprocess_ms = round((time.perf_counter() - preprocess_started) * 1000.0, 3)
            sent_image_paths = [str(path) for path in image_probe.get("image_paths") or original_image_paths if str(path)]
            model_started = time.perf_counter()
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
            model_call_ms = round((time.perf_counter() - model_started) * 1000.0, 3)
            truncated = bool(response.get("truncated"))
            complete = bool(
                response.get("complete", response.get("ok") and not truncated)
            ) and not truncated
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
                    "image_probe": image_probe,
                    "frame_input": _compact_temporal_probe(image_probe, original_image_paths),
                    "timing": {
                        "preprocess_ms": preprocess_ms,
                        "model_call_ms": model_call_ms,
                        "provider_latency_ms": response.get("latency_ms"),
                        "total_ms": round(preprocess_ms + model_call_ms, 3),
                    },
                    "attempts": response.get("attempts", []),
                    "attempt_count": response.get("attempt_count", 1),
                    "finish_reason": str(response.get("finish_reason") or ""),
                    "request_max_tokens": response.get("request_max_tokens", max_tokens),
                    "request_max_tokens_omitted": bool(
                        response.get("request_max_tokens_omitted", max_tokens is None)
                    ),
                    "response_chars": len(str(response.get("content") or "")),
                    "truncated": truncated,
                    "complete": complete,
                }
            )
            if response.get("ok") and complete:
                normalise_candidate = {**candidate, "frame_input": result["frame_input"]}
                understanding = _normalise_temporal_understanding(parse_model_json(str(response.get("content") or "")), normalise_candidate)
                understanding.update(recognition_provenance=response['recognition_provenance'],
                                     executor='provider_api', model_actual=response['model_actual'],
                                     model_requested=response['model_requested'])
                _apply_single(timeline, int(candidate["index"]), understanding)
                applied.append(int(candidate["index"]))
                result["temporal_visual_understanding"] = understanding
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
    if execute and not gate:
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
    timeline_complete_indexes = evidence_indexes["temporal_model_complete"]
    export_consumable_indexes = evidence_indexes["temporal_export_consumable"]
    summary = {
        "schema": "lecture_temporal_visual_analysis_summary.v1",
        "total": len(candidates),
        "execute": execute,
        "status": run_status,
        "invocation_check": invocation_check,
        "error": str(gate.get("error") or "") if gate else "",
        "preflight_path": str(gate.get("preflight_path") or "") if gate else "",
        "preflight_json_path": str(gate.get("preflight_json_path") or "") if gate else "",
        "expected_api_calls": gate.get("expected_api_calls") if gate and "expected_api_calls" in gate else None,
        "expected_indexes": str(gate.get("expected_indexes") or "") if gate else "",
        "execution_control": execution_control,
        "frame_count": effective_frame_count,
        "limit": effective_limit,
        "image_probe_max_edge": effective_image_probe_max_edge,
        "image_probe_jpeg_quality": effective_image_probe_jpeg_quality,
        "vision_retries": int(vision_retries or 1),
        "vision_retry_delay_seconds": float(vision_retry_delay_seconds or 0),
        "request_max_tokens": int(max_tokens) if max_tokens is not None else None,
        "request_max_tokens_omitted": max_tokens is None,
        "timing": _temporal_timing_summary(results),
        "frame_input": _temporal_frame_input_summary(results),
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
        "indexes": [int(index) for index in indexes or []],
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
    progress_path = _write_vision_batch_progress(
        root,
        progress,
        results=results,
        current_index=0,
        current_position=len(candidates),
        status=run_status,
    )
    summary["progress_path"] = str(progress_path)
    manifest["temporal_visual_analysis"] = {
        "schema": "lecture_temporal_visual_analysis.v1",
        "count": len(candidates),
        "items": candidates,
        "input_template_json": str(template_path),
        "last_run": summary,
    }
    if applied:
        write_json(timeline_path, timeline)
        _sync_source_package(manifest, timeline)
    manifest["coverage"] = _coverage_audit(timeline)
    manifest["quality_audit"] = _quality_audit(timeline)
    manifest["repair_status"] = build_repair_status(manifest, timeline)
    timeline_diff = timeline_vision_diff(before_timeline, timeline, applied)
    report_path = root / "temporal-visual-analysis-report.md"
    report_path.write_text(_render_report(root, candidates, summary, template_path, "Temporal Visual Analysis", results=results), encoding="utf-8")
    run_audit = write_vision_analysis_run_audit(
        root,
        kind="temporal_sequence",
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
        run_type="temporal_visual_analysis",
        run_id="temporal-visual-analysis",
        title="Temporal visual analysis",
        command_name="run-temporal-visual-analysis",
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


def write_temporal_visual_input_template(root: str | Path, candidates: list[dict[str, Any]]) -> Path:
    path = Path(root) / "temporal-visual-analysis-input-template.json"
    payload = {
        "schema": "lecture_temporal_visual_analysis_input.v2",
        "items": [
            {
                "index": item.get("index"),
                "temporal_visual_understanding": {
                    "event_sequence": [],
                    "state_changes": [],
                    "operation_steps": [],
                    "causal_links": [],
                    "possible_missing_points": [],
                    "expected_frame_count": len(item.get("frame_manifest") or []),
                    "expected_frame_ids": [row.get("frame_id") for row in item.get("frame_manifest") or []],
                    "observed_frame_count": 0,
                    "observed_frame_ids": [],
                    "per_frame_observations": [],
                    "confidence": 0.0,
                    "evidence_frame_paths": item.get("frame_paths", []),
                },
            }
            for item in candidates
        ],
    }
    write_json(path, payload)
    return path


def _candidates(root: Path, timeline: list[dict[str, Any]], *, frame_count: int) -> list[dict[str, Any]]:
    items = []
    for index, item in enumerate(timeline, start=1):
        route = str(item.get("visual_route") or "")
        if route not in {"temporal_sequence", "mixed"}:
            continue
        if _has_mapping(item.get("temporal_visual_understanding")):
            continue
        frames = [_resolve_frame(root, path) for path in _explicit_temporal_frame_paths(item)]
        if len(frames) < 2:
            continue
        sampled_frames = _sample_frames(frames, frame_count)
        items.append(
            {
                "index": index,
                "start": item.get("start", 0),
                "end": item.get("end", 0),
                "visual_route": route,
                "frame_paths": sampled_frames,
                "frame_manifest": build_temporal_frame_manifest(sampled_frames),
                "transcript": str(item.get("transcript") or ""),
                "visual_text": str(item.get("visual_text") or ""),
                "existing_temporal_visual_understanding": item.get("temporal_visual_understanding") or {},
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


def _prompt(candidate: dict[str, Any]) -> str:
    manifest = candidate.get("frame_manifest") or build_temporal_frame_manifest(candidate.get("frame_paths") or [])
    safe_manifest = [
        {"frame_id": row.get("frame_id"), "timestamp": row.get("timestamp"), "filename": row.get("filename")}
        for row in manifest
    ]
    expected_frame_count = len(safe_manifest)
    return (
        "你正在分析知识类讲课视频的一组按时间排序的画面。\n"
        "系统提供的帧数、帧 ID 和时间戳是事实，不要由模型重新猜测或改写。\n"
        f"expected_frame_count: {expected_frame_count}\n"
        f"frame_manifest: {json.dumps(safe_manifest, ensure_ascii=False)}\n"
        "必须逐一观察每个 frame_id；所有自然语言中的帧数必须与 expected_frame_count 一致。"
        "例如输入为 8 帧时，不要写成“四帧”或“若干帧”。\n"
        "只返回 JSON，对象必须包含 event_sequence, state_changes, operation_steps, causal_links, "
        "possible_missing_points, confidence, observed_frame_count, observed_frame_ids, "
        "per_frame_observations, evidence_frame_paths。\n"
        "observed_frame_ids 必须只使用 frame_manifest 中的 ID；per_frame_observations 必须覆盖每个 ID。\n"
        "OCR 文本负责标题、正文、数字、表格和术语；视觉模型负责可见物体、动作、空间关系和跨帧变化。\n"
        "没有 frame_id 或 evidence_frame_paths 支持时，不得补写人物身份、产品信息、数字或细小文字。\n"
        "若没有变化，明确写 static_sequence；若不确定，放入 possible_missing_points。\n\n"
        f"时间范围: {candidate.get('start')} - {candidate.get('end')}\n"
        f"口语/字幕: {candidate.get('transcript')}\n"
        f"OCR: {candidate.get('visual_text')}\n"
        f"原始帧路径: {candidate.get('frame_paths')}"
    )


def _apply_imports(timeline: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[int]:
    updated = []
    for row in rows:
        index = _int_value(row.get("index"))
        if not (1 <= index <= len(timeline)):
            continue
        payload = row.get("temporal_visual_understanding") if isinstance(row.get("temporal_visual_understanding"), dict) else row
        frame_paths = _temporal_frame_paths(timeline[index - 1])
        candidate = {
            "frame_paths": frame_paths,
            "frame_manifest": build_temporal_frame_manifest(frame_paths),
        }
        _apply_single(timeline, index, _normalise_temporal_understanding(payload, candidate))
        updated.append(index)
    return updated


def _normalise_temporal_understanding(payload: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    frame_manifest = candidate.get("frame_manifest") or build_temporal_frame_manifest(candidate.get("frame_paths") or [])
    expected_frame_ids = [str(row.get("frame_id") or "") for row in frame_manifest if str(row.get("frame_id") or "")]
    reported_evidence = [str(path) for path in _list(payload.get("evidence_frame_paths")) if str(path)]
    observed_frame_ids = _normalise_observed_frame_ids(payload.get("observed_frame_ids"), reported_evidence, frame_manifest)
    observed_frame_count = _int_value(payload.get("observed_frame_count")) or len(observed_frame_ids)
    per_frame_observations = [dict(row) for row in _list(payload.get("per_frame_observations")) if isinstance(row, dict)]
    result = {
        "schema": "lecture_temporal_visual_understanding.v2",
        "event_sequence": _list(payload.get("event_sequence") or payload.get("events")),
        "state_changes": _list(payload.get("state_changes")),
        "operation_steps": _list(payload.get("operation_steps") or payload.get("steps")),
        "causal_links": _list(payload.get("causal_links")),
        "possible_missing_points": _list(payload.get("possible_missing_points") or payload.get("missing_risks")),
        "confidence": _float_value(payload.get("confidence")),
        "evidence_frame_paths": _normalise_evidence_frame_paths(payload.get("evidence_frame_paths"), candidate),
        "model_reported_evidence_frame_paths": reported_evidence,
        "expected_frame_count": len(expected_frame_ids),
        "expected_frame_ids": expected_frame_ids,
        "observed_frame_count": observed_frame_count,
        "observed_frame_ids": observed_frame_ids,
        "per_frame_observations": per_frame_observations,
        "frame_manifest": [
            {"frame_id": row.get("frame_id"), "timestamp": row.get("timestamp"), "filename": row.get("filename")}
            for row in frame_manifest
        ],
        "frame_input": candidate.get("frame_input") if isinstance(candidate.get("frame_input"), dict) else {},
        "strict_frame_contract": bool(candidate.get("strict_frame_contract")),
        "frame_count_claims": _frame_count_claims(payload),
        "source": str(payload.get("source") or "temporal_visual_analyzer"),
        "executor": str(payload.get("executor") or ""),
        "model_requested": str(payload.get("model_requested") or ""),
        "model_actual": str(payload.get("model_actual") or ""),
        "evidence_level": str(payload.get("evidence_level") or ""),
        "parse_failed": bool(payload.get("_parse_failed")),
        "raw_model_output": str(payload.get("raw_content") or "") if payload.get("_parse_failed") else "",
        "updated_at": now_iso(),
    }
    issues = _temporal_understanding_issues(result)
    result["validation_status"] = "incomplete" if issues else "ok"
    result["validation_issues"] = issues
    return result


def _apply_single(timeline: list[dict[str, Any]], index: int, payload: dict[str, Any]) -> None:
    item = timeline[index - 1]
    item["temporal_visual_understanding"] = payload
    item["temporal_visual_understanding_updated_at"] = now_iso()
    issues = [
        issue
        for issue in item.get("quality_issues", [])
        if issue not in {"missing_visual_understanding", "temporal_sequence_without_analysis"}
    ]
    if payload.get("parse_failed") and "model_output_parse_failed" not in issues:
        issues.append("model_output_parse_failed")
    if payload.get("validation_status") == "incomplete" and "temporal_understanding_incomplete" not in issues:
        issues.append("temporal_understanding_incomplete")
    item["quality_issues"] = issues
    item["integrated_visual"] = integrated_visual(item)


def _temporal_understanding_issues(payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if payload.get("parse_failed"):
        issues.append("parse_failed")
    has_content = bool(
        _list(payload.get("event_sequence"))
        or _list(payload.get("state_changes"))
        or _list(payload.get("operation_steps"))
        or _list(payload.get("causal_links"))
    )
    if not has_content:
        issues.append("missing_temporal_content")
    if not _list(payload.get("evidence_frame_paths")):
        issues.append("missing_evidence_frame_paths")
    expected_count = _int_value(payload.get("expected_frame_count"))
    observed_count = _int_value(payload.get("observed_frame_count"))
    expected_ids = {str(value) for value in _list(payload.get("expected_frame_ids")) if str(value)}
    observed_ids = {str(value) for value in _list(payload.get("observed_frame_ids")) if str(value)}
    claims = {_int_value(value) for value in _list(payload.get("frame_count_claims")) if _int_value(value) > 0}
    if expected_count and any(claim != expected_count for claim in claims):
        issues.append("frame_count_claim_mismatch")
    if payload.get("strict_frame_contract"):
        if not _list(payload.get("model_reported_evidence_frame_paths")):
            issues.append("model_evidence_frame_paths_missing")
        if observed_count != expected_count:
            issues.append("observed_frame_count_mismatch")
        if observed_ids != expected_ids:
            issues.append("observed_frame_ids_mismatch")
        observation_ids = {
            str(row.get("frame_id") or "")
            for row in _list(payload.get("per_frame_observations"))
            if isinstance(row, dict) and str(row.get("frame_id") or "")
        }
        if observation_ids != expected_ids:
            issues.append("per_frame_observations_incomplete")
    return issues


def _normalise_observed_frame_ids(values: Any, evidence_paths: list[str], manifest: list[dict[str, Any]]) -> list[str]:
    expected = {str(row.get("frame_id") or "") for row in manifest}
    result = [str(value).upper() for value in _list(values) if str(value).upper() in expected]
    if result:
        return list(dict.fromkeys(result))
    by_name = {str(row.get("filename") or "").lower(): str(row.get("frame_id") or "") for row in manifest}
    return list(
        dict.fromkeys(
            by_name.get(Path(path).name.lower(), "")
            for path in evidence_paths
            if by_name.get(Path(path).name.lower(), "")
        )
    )


def _frame_count_claims(payload: dict[str, Any]) -> list[int]:
    values: list[str] = []
    for key in ("event_sequence", "state_changes", "operation_steps", "causal_links", "possible_missing_points"):
        values.extend(str(value) for value in _list(payload.get(key)) if str(value))
    claims: list[int] = []
    chinese_numbers = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    for value in values:
        claims.extend(int(match) for match in re.findall(r"(?<![A-Za-z0-9])(\d{1,2})\s*(?:帧|张(?:图|图片)?)", value))
        claims.extend(chinese_numbers[match] for match in re.findall(r"([一二两三四五六七八九十])\s*(?:帧|张(?:图|图片)?)", value))
    return list(dict.fromkeys(claims))


def _compact_temporal_probe(image_probe: dict[str, Any], original_paths: list[str]) -> dict[str, Any]:
    return {
        "schema": str(image_probe.get("schema") or ""),
        "status": str(image_probe.get("status") or ""),
        "sent_strategy": str(image_probe.get("sent_strategy") or "ordered_frames"),
        "original_frame_count": int(image_probe.get("original_frame_count") or len(original_paths)),
        "representative_frame_count": int(image_probe.get("representative_frame_count") or len(original_paths)),
        "sent_image_count": int(image_probe.get("sent_image_count") or len(image_probe.get("image_paths") or original_paths)),
        "total_source_bytes": int(image_probe.get("total_source_bytes") or 0),
        "total_prepared_bytes": int(image_probe.get("total_prepared_bytes") or image_probe.get("total_probe_bytes") or 0),
        "byte_reduction_ratio": float(image_probe.get("byte_reduction_ratio") or 0.0),
        "contact_sheet_path": str(image_probe.get("contact_sheet_path") or ""),
        "frame_mapping": image_probe.get("frame_mapping") if isinstance(image_probe.get("frame_mapping"), list) else [],
        "implementation": image_probe.get("implementation") if isinstance(image_probe.get("implementation"), dict) else {},
    }


def _temporal_timing_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    timings = [row.get("timing") for row in results if isinstance(row.get("timing"), dict)]
    return {
        "measured_count": len(timings),
        "preprocess_ms_total": round(sum(float(row.get("preprocess_ms") or 0) for row in timings), 3),
        "model_call_ms_total": round(sum(float(row.get("model_call_ms") or 0) for row in timings), 3),
        "total_ms": round(sum(float(row.get("total_ms") or 0) for row in timings), 3),
    }


def _temporal_frame_input_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row.get("frame_input") for row in results if isinstance(row.get("frame_input"), dict)]
    return {
        "measured_count": len(rows),
        "original_frame_count": sum(int(row.get("original_frame_count") or 0) for row in rows),
        "representative_frame_count": sum(int(row.get("representative_frame_count") or 0) for row in rows),
        "sent_image_count": sum(int(row.get("sent_image_count") or 0) for row in rows),
        "total_source_bytes": sum(int(row.get("total_source_bytes") or 0) for row in rows),
        "total_prepared_bytes": sum(int(row.get("total_prepared_bytes") or 0) for row in rows),
    }


def _sync_source_package(manifest: dict[str, Any], timeline: list[dict[str, Any]]) -> None:
    source = Path(str(manifest.get("source_package") or "")).expanduser()
    if not source.exists() or not source.is_file():
        return
    package = read_json(source)
    if not isinstance(package, dict) or not isinstance(package.get("timeline"), list):
        return
    for index, item in enumerate(package["timeline"], start=1):
        if index <= len(timeline) and isinstance(item, dict) and timeline[index - 1].get("temporal_visual_understanding"):
            item["temporal_visual_understanding"] = timeline[index - 1]["temporal_visual_understanding"]
            item["temporal_visual_understanding_updated_at"] = timeline[index - 1].get("temporal_visual_understanding_updated_at")
            item["integrated_visual"] = timeline[index - 1].get("integrated_visual") or integrated_visual(timeline[index - 1])
    package["coverage"] = _coverage_audit(package["timeline"])
    package["quality_audit"] = _quality_audit(package["timeline"])
    write_json(source, package)


def _sample_frames(frames: list[str], count: int) -> list[str]:
    if len(frames) <= count:
        return frames
    if count <= 1:
        return frames[:1]
    indexes = [round(i * (len(frames) - 1) / (count - 1)) for i in range(count)]
    result = []
    for index in indexes:
        value = frames[int(index)]
        if value not in result:
            result.append(value)
    return result


def _temporal_frame_paths(item: dict[str, Any]) -> list[str]:
    values = _explicit_temporal_frame_paths(item)
    if values:
        return values
    return _frame_paths(item)


def _explicit_temporal_frame_paths(item: dict[str, Any]) -> list[str]:
    values = item.get("temporal_frame_paths") if isinstance(item.get("temporal_frame_paths"), list) else []
    if values:
        return [str(path) for path in values if str(path)]
    group = item.get("temporal_frame_group") if isinstance(item.get("temporal_frame_group"), dict) else {}
    values = group.get("frame_paths") if isinstance(group.get("frame_paths"), list) else []
    if values:
        return [str(path) for path in values if str(path)]
    return []
