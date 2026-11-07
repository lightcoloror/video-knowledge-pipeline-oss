from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from .canonical_json import canonical_json_sha256

from .powershell import quote_powershell_literal as _powershell_quote
from .asr_adapter import read_asr_segment_dicts, render_srt
from .asr_chunk_batch_workflow import SCHEMA as WORKFLOW_SCHEMA
from .asr_local_agreement import (
    measure_local_agreement,
    measure_timestamped_local_agreement,
)
from .asr_response_quality import assess_asr_response
from .asr_runner import plan_asr_run
from .asr_vad_chunking import read_vad_intervals
from .consented_model_batch import SCHEMA as BATCH_SCHEMA
from .models import TranscriptCue, now_iso
from .run_artifact_registry import register_bundle_run
from .storage import read_json, write_json
from .text_normalization import compact_ascii_cjk as _compact_text
from .file_hash import sha256_file as _file_sha256
from .interval_coverage import interval_intersection_over_shorter


SCHEMA = "video_knowledge_pipeline.asr_chunk_batch_merge.v1"
TRANSCRIPT_SCHEMA = "video_knowledge_pipeline.chunked_asr_transcript.v1"
BOUNDARY_OVERLAP_RATIO = 0.6


def merge_asr_chunk_batch_reports(
    workflow_path: str | Path,
    execution_reports: Sequence[str | Path] | None = None,
    *,
    batch_status_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    title: str = "",
    prepare_alignment_plan: bool = False,
    alignment_language: str = "zh",
    alignment_model: str = "",
    alignment_planner: Callable[..., dict[str, Any]] = plan_asr_run,
    write: bool = True,
) -> dict[str, Any]:
    """Merge saved chunk execution reports without calling a provider."""

    workflow_file = Path(workflow_path).expanduser().resolve()
    if prepare_alignment_plan and not write:
        raise ValueError("prepare_alignment_plan requires write=True")
    if not workflow_file.is_file():
        raise FileNotFoundError(f"ASR chunk workflow not found: {workflow_file}")
    workflow = _object(read_json(workflow_file), "ASR chunk workflow")
    nodes = _validate_workflow(workflow)
    report_paths, batch_status = _resolve_report_paths(
        execution_reports or [],
        batch_status_path=batch_status_path,
        expected_node_ids={str(row["node_id"]) for row in nodes},
    )
    report_by_consent, report_failures = _reports_by_consent(report_paths)
    source_media = _workflow_source_media(workflow)
    merged_candidates: list[dict[str, Any]] = []
    chunk_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = list(report_failures)
    excluded_padding: list[dict[str, Any]] = []

    expected_consent_ids = {str(row["consent_id"]) for row in nodes}
    for extra_consent_id, (_, extra_path) in report_by_consent.items():
        if extra_consent_id not in expected_consent_ids:
            failures.append(
                {
                    "execution_report": str(extra_path),
                    "consent_id": extra_consent_id,
                    "error": "execution_report_not_in_workflow",
                }
            )

    for index, node in enumerate(nodes):
        is_last = index == len(nodes) - 1
        consent_id = str(node["consent_id"])
        report_entry = report_by_consent.get(consent_id)
        if report_entry is None:
            failure = {
                "node_id": str(node["node_id"]),
                "chunk_id": str(node["chunk_id"]),
                "consent_id": consent_id,
                "error": "missing_execution_report",
            }
            failures.append(failure)
            chunk_results.append({**failure, "status": "failed", "segment_count": 0})
            continue
        report, report_path = report_entry
        try:
            _validate_execution_report(report, report_path=report_path, node=node)
            local_segments = read_asr_segment_dicts(report_path, provider="openai")
            shifted, excluded = _shift_chunk_segments(
                local_segments,
                node=node,
                report_path=report_path,
                is_last=is_last,
            )
            merged_candidates.extend(shifted)
            excluded_padding.extend(excluded)
            chunk_quality_passed = bool(report.get("quality_gate_passed"))
            chunk_results.append(
                {
                    "node_id": str(node["node_id"]),
                    "chunk_id": str(node["chunk_id"]),
                    "consent_id": consent_id,
                    "status": "completed"
                    if chunk_quality_passed
                    else "review_required",
                    "segment_count": len(shifted),
                    "excluded_padding_segment_count": len(excluded),
                    "quality_gate_passed": chunk_quality_passed,
                    "production_qualified": bool(report.get("production_qualified")),
                    "execution_report": str(report_path),
                    "execution_report_sha256": _file_sha256(report_path),
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve other successful chunks.
            failure = {
                "node_id": str(node["node_id"]),
                "chunk_id": str(node["chunk_id"]),
                "consent_id": consent_id,
                "execution_report": str(report_path),
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            chunk_results.append({**failure, "status": "failed", "segment_count": 0})

    deduplicated, exact_deduplications = _deduplicate_exact_segments(
        merged_candidates
    )
    deduplicated.sort(
        key=lambda row: (
            float(row.get("start") or 0),
            float(row.get("end") or 0),
            int(row.get("chunk_position") or 0),
        )
    )
    boundary_conflicts = _boundary_conflicts(
        deduplicated,
        language=alignment_language,
    )
    vad_intervals = _workflow_vad_intervals(workflow)
    media_duration = max(
        [float(row.get("core_end") or 0) for row in nodes] + [0.0]
    )
    quality = assess_asr_response(
        {"segments": deduplicated},
        vad_intervals=vad_intervals,
        media_duration_seconds=media_duration,
    )
    reviewed_chunks = [
        row for row in chunk_results if row.get("status") == "review_required"
    ]
    if not deduplicated:
        status = "failed"
    elif failures or quality.get("status") in {"failed", "degraded"}:
        status = "degraded"
    elif reviewed_chunks or boundary_conflicts or quality.get("status") == "review_required":
        status = "review_required"
    else:
        status = "completed"

    target_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else workflow_file.parent / "asr-chunk-merge"
    )
    transcript_path = target_dir / "asr-chunk-merged-transcript.json"
    srt_path = target_dir / "asr-chunk-merged-transcript.srt"
    report_path = target_dir / "asr-chunk-merge-report.json"
    alignment_advisory = _alignment_advisory(status)
    transcript = {
        "schema": TRANSCRIPT_SCHEMA,
        "title": str(title or workflow_file.stem),
        "provider": "trusted_connector_chunk_batch",
        "source_workflow": str(workflow_file),
        "source_workflow_sha256": _file_sha256(workflow_file),
        "source_media": source_media,
        "segments": deduplicated,
        "chunk_merge": {
            "status": status,
            "chunk_count": len(nodes),
            "completed_chunk_count": sum(
                row.get("status") != "failed" for row in chunk_results
            ),
            "failed_chunk_count": sum(
                row.get("status") == "failed" for row in chunk_results
            ),
            "review_chunk_count": len(reviewed_chunks),
            "exact_deduplication_count": len(exact_deduplications),
            "boundary_conflict_count": len(boundary_conflicts),
            "padding_segment_excluded_count": len(excluded_padding),
            "quality_gate_passed": bool(quality.get("quality_gate_passed"))
            and not failures
            and not boundary_conflicts,
        },
        "created_at": now_iso(),
    }
    cues = [
        TranscriptCue(
            start=float(row.get("start") or 0),
            end=float(row.get("end") or row.get("start") or 0),
            text=str(row.get("text") or ""),
            segment_id=str(row.get("segment_id") or row.get("id") or ""),
            source_segment_ids=[
                str(value) for value in row.get("source_segment_ids") or []
            ],
            transformations=[
                dict(value)
                for value in row.get("transformations") or []
                if isinstance(value, dict)
            ],
        )
        for row in deduplicated
    ]
    result = {
        "schema": SCHEMA,
        "status": status,
        "ok": status == "completed",
        "production_qualified": status == "completed",
        "write": bool(write),
        "workflow_path": str(workflow_file),
        "workflow_sha256": _file_sha256(workflow_file),
        "source_media": source_media,
        "batch_status_path": str(batch_status.get("path") or ""),
        "batch_job_id": str(batch_status.get("job_id") or ""),
        "expected_chunk_count": len(nodes),
        "successful_chunk_count": sum(
            row.get("status") != "failed" for row in chunk_results
        ),
        "failed_chunk_count": sum(
            row.get("status") == "failed" for row in chunk_results
        ),
        "review_chunk_count": len(reviewed_chunks),
        "segment_count": len(deduplicated),
        "exact_deduplication_count": len(exact_deduplications),
        "boundary_conflict_count": len(boundary_conflicts),
        "excluded_padding_segment_count": len(excluded_padding),
        "chunk_results": chunk_results,
        "failures": failures,
        "exact_deduplications": exact_deduplications,
        "boundary_conflicts": boundary_conflicts,
        "excluded_padding_segments": excluded_padding,
        "asr_quality": quality,
        "transcript_path": str(transcript_path),
        "srt_path": str(srt_path),
        "report_path": str(report_path),
        "alignment_advisory": alignment_advisory,
        "operator_boundary": {
            "provider_call_performed": False,
            "batch_submitted": False,
            "execution_reports_read_only": True,
            "raw_chunk_outputs_modified": False,
            "canonical_transcript_modified": False,
            "fuzzy_text_merge_performed": False,
            "exact_duplicate_removal_only": True,
            "automatic_retry": False,
            "automatic_fallback": False,
            "alignment_execution_performed": False,
            "canonical_text_replaced_by_alignment": False,
        },
        "updated_at": now_iso(),
    }
    if write:
        target_dir.mkdir(parents=True, exist_ok=True)
        write_json(transcript_path, transcript)
        srt_path.write_text(render_srt(cues), encoding="utf-8")
        if prepare_alignment_plan:
            try:
                result["alignment_advisory"] = _prepare_alignment_plan(
                    workflow,
                    source_media=source_media,
                    transcript_path=transcript_path,
                    merge_status=status,
                    language=alignment_language,
                    model=alignment_model,
                    planner=alignment_planner,
                )
                result["operator_boundary"]["alignment_plan_prepared"] = True
            except Exception as exc:  # noqa: BLE001 - alignment is optional evidence.
                result["alignment_advisory"] = {
                    **_alignment_advisory(status),
                    "status": "plan_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                result["operator_boundary"]["alignment_plan_prepared"] = False
        else:
            result["operator_boundary"]["alignment_plan_prepared"] = False
        result["run_registry"] = _register_merge_run(
            workflow,
            result=result,
            transcript_path=transcript_path,
            srt_path=srt_path,
            report_path=report_path,
            target_dir=target_dir,
        )
        write_json(report_path, result)
    return result


def _register_merge_run(
    workflow: dict[str, Any],
    *,
    result: dict[str, Any],
    transcript_path: Path,
    srt_path: Path,
    report_path: Path,
    target_dir: Path,
) -> dict[str, Any]:
    bundle = Path(str(workflow.get("bundle_dir") or "")).expanduser().resolve()
    if not (bundle / "manifest.json").is_file() or not (bundle / "timeline.json").is_file():
        return {}
    status = str(result.get("status") or "failed")
    retry_command = ""
    next_actions: list[str]
    if status == "completed":
        next_actions = [
            "Optionally prepare the existing qwen3-forced-aligner plan.",
            "Refresh Task Console to inspect the merged transcript artifacts.",
        ]
    else:
        source_media = result.get("source_media")
        source_media = source_media if isinstance(source_media, dict) else {}
        media_path = str(source_media.get("path") or "")
        retry_dir = target_dir / "retry-snippets"
        retry_command = (
            ".\\scripts\\video-knowledge.ps1 asr-retry-snippets "
            f"{_powershell_quote(media_path)} "
            f"{_powershell_quote(str(report_path))} "
            f"{_powershell_quote(str(retry_dir))}"
        )
        next_actions = [
            "Inspect failed chunks and boundary conflicts in the merge report.",
            "Prepare only the reported local retry snippets; remote execution still needs exact consent.",
        ]
    run = register_bundle_run(
        bundle,
        run_type="asr_chunk_merge",
        run_id=f"asr-chunk-merge-{str(result.get('workflow_sha256') or '')[:12]}",
        status=status,
        title="VAD-aligned ASR chunk merge",
        summary=(
            f"{int(result.get('successful_chunk_count') or 0)}/"
            f"{int(result.get('expected_chunk_count') or 0)} chunks retained; "
            f"{int(result.get('failed_chunk_count') or 0)} failed."
        ),
        inputs={
            "workflow_path": str(result.get("workflow_path") or ""),
            "workflow_sha256": str(result.get("workflow_sha256") or ""),
            "batch_job_id": str(result.get("batch_job_id") or ""),
            "merge_report": str(report_path),
        },
        parameters={
            "exact_duplicate_removal_only": True,
            "fuzzy_text_merge_performed": False,
            "canonical_transcript_modified": False,
        },
        artifacts=[
            {
                "key": "chunked_asr_transcript",
                "path": str(transcript_path),
                "description": "Text-preserving merged ASR segments.",
            },
            {
                "key": "chunked_asr_srt",
                "path": str(srt_path),
                "description": "Merged SRT using preserved segment boundaries.",
            },
        ],
        failed_items=[
            {
                "id": str(row.get("node_id") or row.get("chunk_id") or ""),
                "reason": "asr_chunk_merge_failed",
                "detail": str(row.get("error") or ""),
            }
            for row in result.get("failures") or []
            if isinstance(row, dict)
        ],
        retry_command=retry_command,
        next_actions=next_actions,
        operator_boundary={
            "local_only": True,
            "provider_call_performed": False,
            "canonical_transcript_modified": False,
            "automatic_retry": False,
            "automatic_fallback": False,
        },
        resource_requirements={"cpu": 1, "network": 0},
        write=True,
    )
    paths = run.get("paths") if isinstance(run.get("paths"), dict) else {}
    return {
        "run_id": str(run.get("run_id") or ""),
        "status": str(run.get("status") or ""),
        "run_json": str(paths.get("run_json") or ""),
        "run_markdown": str(paths.get("run_markdown") or ""),
        "registry_json": str(bundle / "run-artifact-registry.json"),
    }


def _alignment_advisory(merge_status: str) -> dict[str, Any]:
    return {
        "status": "available" if merge_status == "completed" else "blocked_by_merge_status",
        "preferred_preset": "qwen3-forced-aligner",
        "purpose": "text_preserving_timestamp_alignment",
        "whisperx_role": "optional_word_timestamp_or_speaker_evidence",
        "plan_prepared": False,
        "execution_performed": False,
        "canonical_transcript_modified": False,
    }


def _prepare_alignment_plan(
    workflow: dict[str, Any],
    *,
    source_media: dict[str, Any],
    transcript_path: Path,
    merge_status: str,
    language: str,
    model: str,
    planner: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    if merge_status != "completed":
        raise ValueError("forced-alignment plan requires a completed chunk merge")
    root = Path(str(workflow.get("bundle_dir") or "")).expanduser().resolve()
    if not (root / "manifest.json").is_file() or not (root / "timeline.json").is_file():
        raise ValueError("forced-alignment plan requires the workflow Bundle directory")
    media = Path(str(source_media.get("path") or "")).expanduser().resolve()
    if not media.is_file():
        raise FileNotFoundError(f"forced-alignment source media not found: {media}")
    if media.stat().st_size != int(source_media.get("bytes") or -1):
        raise ValueError("forced-alignment source media byte count changed")
    if _file_sha256(media) != str(source_media.get("sha256") or ""):
        raise ValueError("forced-alignment source media SHA-256 changed")
    plan = planner(
        root,
        media,
        preset="qwen3-forced-aligner",
        language=str(language or "zh"),
        model=str(model or "").strip() or None,
        transcript_path=transcript_path,
    )
    if not isinstance(plan, dict) or not str(plan.get("plan_path") or "").strip():
        raise ValueError("qwen3-forced-aligner planner returned no plan_path")
    return {
        **_alignment_advisory(merge_status),
        "status": "planned",
        "plan_prepared": True,
        "plan_path": str(plan["plan_path"]),
        "preset": "qwen3-forced-aligner",
        "language": str(language or "zh"),
        "model": str(plan.get("model") or model or ""),
        "source_media_sha256": str(source_media.get("sha256") or ""),
        "transcript_sha256": _file_sha256(transcript_path),
    }


def _validate_workflow(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    if workflow.get("schema") != WORKFLOW_SCHEMA or workflow.get("status") != "ready":
        raise ValueError("unsupported or non-ready ASR chunk workflow")
    nodes = [
        dict(row) for row in workflow.get("nodes") or [] if isinstance(row, dict)
    ]
    if not nodes or int(workflow.get("chunk_count") or -1) != len(nodes):
        raise ValueError("ASR chunk workflow nodes are missing or inconsistent")
    manifest_path = (
        Path(str(workflow.get("chunk_manifest") or "")).expanduser().resolve()
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"ASR chunk manifest not found: {manifest_path}")
    if _file_sha256(manifest_path) != str(workflow.get("chunk_manifest_sha256") or ""):
        raise ValueError("ASR chunk manifest changed after workflow compilation")
    for node in nodes:
        artifact_path = Path(str(node.get("artifact_path") or "")).expanduser().resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(f"ASR chunk artifact not found: {artifact_path}")
        if artifact_path.stat().st_size != int(node.get("artifact_bytes") or -1):
            raise ValueError(
                f"ASR chunk artifact byte count changed after workflow compilation: {artifact_path}"
            )
        if _file_sha256(artifact_path) != str(node.get("artifact_sha256") or ""):
            raise ValueError(
                f"ASR chunk artifact SHA-256 changed after workflow compilation: {artifact_path}"
            )
    activity_audit_sha256 = str(workflow.get("activity_audit_sha256") or "")
    if activity_audit_sha256:
        activity_audit = workflow.get("activity_audit")
        activity_audit = activity_audit if isinstance(activity_audit, dict) else {}
        activity_audit_path = (
            Path(str(activity_audit.get("path") or "")).expanduser().resolve()
        )
        if not activity_audit_path.is_file():
            raise FileNotFoundError(
                f"ASR VAD activity audit not found: {activity_audit_path}"
            )
        if _file_sha256(activity_audit_path) != activity_audit_sha256:
            raise ValueError("ASR VAD activity audit changed after workflow compilation")
    submission = (
        workflow.get("submission")
        if isinstance(workflow.get("submission"), dict)
        else {}
    )
    arguments = (
        submission.get("arguments")
        if isinstance(submission.get("arguments"), dict)
        else {}
    )
    identity = {
        "chunk_manifest_sha256": str(workflow.get("chunk_manifest_sha256") or ""),
        "bundle_dir": str(workflow.get("bundle_dir") or ""),
        "nodes": [
            {
                "id": str(row.get("node_id") or ""),
                "consent_sha256": str(row.get("consent_sha256") or ""),
                "artifact_sha256": str(row.get("artifact_sha256") or ""),
                "route_revision": str(row.get("route_revision") or ""),
                "destination": str(row.get("destination") or ""),
            }
            for row in nodes
        ],
        "max_parallel_global": int(arguments.get("max_parallel_global") or 0),
        "max_parallel_per_destination": int(
            arguments.get("max_parallel_per_destination") or 0
        ),
    }
    if activity_audit_sha256:
        identity["activity_audit_sha256"] = activity_audit_sha256
    if _payload_sha256(identity) != str(workflow.get("workflow_sha256") or ""):
        raise ValueError("ASR chunk workflow identity changed after compilation")
    return nodes


def _resolve_report_paths(
    execution_reports: Sequence[str | Path],
    *,
    batch_status_path: str | Path | None,
    expected_node_ids: set[str],
) -> tuple[list[Path], dict[str, Any]]:
    if execution_reports and batch_status_path:
        raise ValueError("use execution_reports or batch_status_path, not both")
    if not execution_reports and not batch_status_path:
        raise ValueError("execution reports or a batch status artifact are required")
    if execution_reports:
        return [Path(value).expanduser().resolve() for value in execution_reports], {}
    path = Path(str(batch_status_path)).expanduser().resolve()
    payload = _object(read_json(path), "batch status")
    if payload.get("schema") != BATCH_SCHEMA or not payload.get("terminal"):
        raise ValueError("batch status must be a terminal consented model batch")
    items = [dict(row) for row in payload.get("items") or [] if isinstance(row, dict)]
    actual_node_ids = {str(row.get("node_id") or "") for row in items}
    if actual_node_ids != expected_node_ids:
        raise ValueError("batch status nodes do not match the ASR chunk workflow")
    paths = [
        Path(str(row.get("execution_report") or "")).expanduser().resolve()
        for row in items
        if str(row.get("execution_report") or "").strip()
    ]
    return paths, {
        "path": str(path),
        "job_id": str(payload.get("job_id") or ""),
        "status": str(payload.get("status") or ""),
    }


def _reports_by_consent(
    report_paths: list[Path],
) -> tuple[dict[str, tuple[dict[str, Any], Path]], list[dict[str, Any]]]:
    by_consent: dict[str, tuple[dict[str, Any], Path]] = {}
    failures: list[dict[str, Any]] = []
    for path in report_paths:
        if not path.is_file():
            failures.append(
                {"execution_report": str(path), "error": "execution_report_missing"}
            )
            continue
        try:
            payload = _object(read_json(path), f"execution report {path}")
        except Exception as exc:  # noqa: BLE001 - report all invalid files.
            failures.append(
                {
                    "execution_report": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        consent_id = str(payload.get("consent_id") or "").strip()
        if not consent_id:
            failures.append(
                {"execution_report": str(path), "error": "consent_id_missing"}
            )
            continue
        if consent_id in by_consent:
            failures.append(
                {
                    "execution_report": str(path),
                    "consent_id": consent_id,
                    "error": "duplicate_execution_report_for_consent",
                }
            )
            continue
        by_consent[consent_id] = (payload, path)
    return by_consent, failures


def _validate_execution_report(
    report: dict[str, Any],
    *,
    report_path: Path,
    node: dict[str, Any],
) -> None:
    schema = str(report.get("schema") or "")
    if not schema.startswith("video_knowledge_pipeline.trusted_model_connector."):
        raise ValueError("execution report is not a Trusted Connector result")
    if report.get("ok") is not True or report.get("transport_ok") is False:
        raise ValueError("chunk execution did not complete with valid transport")
    if report.get("contract_ok") is False:
        raise ValueError("chunk execution output contract failed")
    if str(report.get("task") or "") != "cloud_asr":
        raise ValueError("chunk execution task is not cloud_asr")
    if str(report.get("consent_id") or "") != str(node.get("consent_id") or ""):
        raise ValueError("chunk execution consent_id does not match workflow")
    artifact_paths = [
        str(Path(str(value)).expanduser().resolve()).casefold()
        for value in report.get("artifact_paths") or []
        if str(value).strip()
    ]
    if artifact_paths != [str(Path(str(node["artifact_path"])).resolve()).casefold()]:
        raise ValueError("chunk execution artifact path does not match workflow")
    manifest = (
        report.get("upload_manifest")
        if isinstance(report.get("upload_manifest"), dict)
        else {}
    )
    files = [dict(row) for row in manifest.get("files") or [] if isinstance(row, dict)]
    if len(files) != 1:
        raise ValueError("chunk execution upload manifest must contain one artifact")
    if int(files[0].get("bytes") or -1) != int(node["artifact_bytes"]):
        raise ValueError("chunk execution artifact bytes do not match workflow")
    if str(files[0].get("sha256") or "") != str(node["artifact_sha256"]):
        raise ValueError("chunk execution artifact SHA-256 does not match workflow")
    route = report.get("route") if isinstance(report.get("route"), dict) else {}
    if str(route.get("route_revision") or "") != str(node["route_revision"]):
        raise ValueError("chunk execution route_revision does not match workflow")
    if not report_path.is_file():
        raise FileNotFoundError(f"execution report not found: {report_path}")


def _shift_chunk_segments(
    segments: list[dict[str, Any]],
    *,
    node: dict[str, Any],
    report_path: Path,
    is_last: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    offset = float(node["artifact_start"])
    core_start = float(node["core_start"])
    core_end = float(node["core_end"])
    shifted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for position, source in enumerate(segments, start=1):
        local_start = float(source.get("start") or 0)
        local_end = max(local_start, float(source.get("end") or local_start))
        absolute_start = offset + local_start
        absolute_end = offset + local_end
        midpoint = (absolute_start + absolute_end) / 2
        belongs = core_start <= midpoint < core_end or (
            is_last and core_start <= midpoint <= core_end
        )
        source_id = str(
            source.get("segment_id") or source.get("id") or f"segment-{position:06d}"
        )
        merged_id = f"{node['chunk_id']}:{source_id}"
        if not belongs:
            excluded.append(
                {
                    "segment_id": merged_id,
                    "chunk_id": str(node["chunk_id"]),
                    "absolute_start": round(absolute_start, 6),
                    "absolute_end": round(absolute_end, 6),
                    "reason": "padding_context_midpoint_outside_core",
                    "text_sha256": _text_sha256(str(source.get("text") or "")),
                }
            )
            continue
        row = json.loads(json.dumps(source, ensure_ascii=False))
        source_ids = [
            str(value)
            for value in source.get("source_segment_ids") or [source_id]
            if str(value).strip()
        ]
        row.update(
            {
                "id": merged_id,
                "segment_id": merged_id,
                "source_segment_ids": [
                    f"{node['chunk_id']}:{value}" for value in source_ids
                ],
                "start": round(absolute_start, 6),
                "end": round(absolute_end, 6),
                "chunk_id": str(node["chunk_id"]),
                "chunk_position": int(node["chunk_position"]),
                "chunk_segment_id": source_id,
                "local_start": round(local_start, 6),
                "local_end": round(local_end, 6),
                "time_offset_seconds": round(offset, 6),
                "consent_id": str(node["consent_id"]),
                "execution_report": str(report_path),
                "execution_report_sha256": _file_sha256(report_path),
                "_core_margin": round(
                    min(midpoint - core_start, core_end - midpoint), 6
                ),
            }
        )
        row["metadata"] = _shift_word_timestamps(row.get("metadata"), offset)
        transformations = [
            dict(value)
            for value in row.get("transformations") or []
            if isinstance(value, dict)
        ]
        transformations.append(
            {
                "type": "chunk_time_offset",
                "chunk_id": str(node["chunk_id"]),
                "artifact_start": round(offset, 6),
                "timestamp_boundary_preserved": True,
            }
        )
        row["transformations"] = transformations
        shifted.append(row)
    return shifted, excluded


def _shift_word_timestamps(value: Any, offset: float) -> dict[str, Any]:
    metadata = dict(value) if isinstance(value, dict) else {}
    words = [dict(row) for row in metadata.get("words") or [] if isinstance(row, dict)]
    for row in words:
        if isinstance(row.get("start"), (int, float)):
            row["start"] = round(float(row["start"]) + offset, 6)
        if isinstance(row.get("end"), (int, float)):
            row["end"] = round(float(row["end"]) + offset, 6)
    if words:
        metadata["words"] = words
    return metadata


def _deduplicate_exact_segments(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for candidate in sorted(
        rows,
        key=lambda row: (
            float(row.get("start") or 0),
            float(row.get("end") or 0),
            int(row.get("chunk_position") or 0),
        ),
    ):
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(kept)
                if existing.get("chunk_id") != candidate.get("chunk_id")
                and _compact_text(existing.get("text")) == _compact_text(candidate.get("text"))
                and _compact_text(candidate.get("text"))
                and _time_overlap_ratio(existing, candidate) >= BOUNDARY_OVERLAP_RATIO
            ),
            None,
        )
        if duplicate_index is None:
            kept.append(candidate)
            continue
        existing = kept[duplicate_index]
        existing_margin = float(existing.get("_core_margin") or 0)
        candidate_margin = float(candidate.get("_core_margin") or 0)
        winner, loser = (
            (candidate, existing)
            if candidate_margin > existing_margin
            else (existing, candidate)
        )
        if winner is candidate:
            kept[duplicate_index] = candidate
        winner.setdefault("deduplicated_source_segment_ids", []).append(
            str(loser.get("segment_id") or "")
        )
        removed.append(
            {
                "kept_segment_id": str(winner.get("segment_id") or ""),
                "removed_segment_id": str(loser.get("segment_id") or ""),
                "rule": "exact_text_and_time_overlap",
                "time_overlap_ratio": round(_time_overlap_ratio(winner, loser), 6),
            }
        )
    for row in kept:
        row.pop("_core_margin", None)
    return kept, removed


def _boundary_conflicts(
    rows: list[dict[str, Any]],
    *,
    language: str = "",
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            if float(right.get("start") or 0) >= float(left.get("end") or 0):
                break
            if left.get("chunk_id") == right.get("chunk_id"):
                continue
            ratio = _time_overlap_ratio(left, right)
            left_text = _compact_text(left.get("text"))
            right_text = _compact_text(right.get("text"))
            if (
                ratio >= BOUNDARY_OVERLAP_RATIO
                and left_text
                and right_text
                and left_text != right_text
            ):
                conflicts.append(
                    {
                        "left_segment_id": str(left.get("segment_id") or ""),
                        "right_segment_id": str(right.get("segment_id") or ""),
                        "time_overlap_ratio": round(ratio, 6),
                        "left_text": str(left.get("text") or ""),
                        "right_text": str(right.get("text") or ""),
                        "local_agreement": measure_local_agreement(
                            left.get("text"),
                            right.get("text"),
                            language=language,
                        ),
                        "timestamped_local_agreement": measure_timestamped_local_agreement(
                            _word_rows(left),
                            _word_rows(right),
                            overlap_start=max(
                                float(left.get("start") or 0),
                                float(right.get("start") or 0),
                            ),
                            overlap_end=min(
                                float(left.get("end") or 0),
                                float(right.get("end") or 0),
                            ),
                        ),
                        "requires_human_review": True,
                    }
                )
    return conflicts


def _word_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = (
        row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    )
    return [
        dict(value)
        for value in metadata.get("words") or []
        if isinstance(value, dict)
    ]

def _time_overlap_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_start = float(left.get("start") or 0)
    left_end = float(left.get("end") or left_start)
    right_start = float(right.get("start") or 0)
    right_end = float(right.get("end") or right_start)
    return interval_intersection_over_shorter(
        left_start,
        left_end,
        right_start,
        right_end,
    )


def _workflow_vad_intervals(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_path = Path(str(workflow.get("chunk_manifest") or "")).expanduser().resolve()
    manifest = _object(read_json(manifest_path), "ASR chunk manifest")
    vad_path = Path(str(manifest.get("vad_json") or "")).expanduser().resolve()
    expected_hash = str(manifest.get("vad_sha256") or "")
    if not vad_path.is_file() or not expected_hash:
        return []
    if _file_sha256(vad_path) != expected_hash:
        raise ValueError("VAD JSON changed after ASR chunk preparation")
    return read_vad_intervals(vad_path)


def _workflow_source_media(workflow: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(workflow.get("chunk_manifest") or "")).expanduser().resolve()
    manifest = _object(read_json(manifest_path), "ASR chunk manifest")
    path = str(manifest.get("source_path") or "").strip()
    sha256 = str(manifest.get("source_sha256") or "").strip()
    byte_count = manifest.get("source_bytes")
    if not path or not sha256 or isinstance(byte_count, bool):
        return {}
    if not isinstance(byte_count, int) or byte_count < 0:
        return {}
    return {"path": path, "bytes": byte_count, "sha256": sha256}


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _text_sha256(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _payload_sha256(value: Any) -> str:
    return canonical_json_sha256(value)
