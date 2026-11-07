from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

import psutil
from jsonschema import Draft202012Validator
from transitions import Machine

from .artifact_validation import sha256_file
from .autopilot_vision_executor import (
    ProviderVisionTargetsExhausted,
    execute_provider_vision_work_item,
    import_provider_vision_result,
    local_vision_provider_snapshot,
    prepare_provider_vision_pack,
    validate_provider_vision_result,
)
from .bundle_next import bundle_advance, bundle_next_action
from .temporal_tag_delta import run_temporal_tag_delta
from .canonical_json import canonical_json_sha256
from .codex_vision_session import import_codex_vision_results, prepare_codex_vision_session
from .models import now_iso
from .panorama_cli import panorama_prepare, panorama_run
from .panorama_session import prepare_panorama_temporal_session, prepare_panorama_vision_session
from .run_artifact_registry import build_run_artifact_registry, register_bundle_run
from .storage import append_jsonl, bundle_write_lock, read_json, write_json
from .transcript_semantic_correction import (
    build_transcript_semantic_candidate_discovery_pack,
    build_transcript_semantic_correction_llm_draft,
    import_transcript_semantic_candidate_suggestions,
    transcript_semantic_correction_closure,
    validate_transcript_semantic_model_output,
)

JOB_SCHEMA = "video_knowledge_pipeline.autopilot_job.v1"
BATCH_SCHEMA = "video_knowledge_pipeline.autopilot_batch.v1"
WORK_ITEM_SCHEMA = "video_knowledge_pipeline.autopilot_work_item.v1"
SUBMISSION_SCHEMA = "video_knowledge_pipeline.autopilot_submission.v1"
RECEIPT_SCHEMA = "video_knowledge_pipeline.autopilot_step_receipt.v1"

MAX_WORK_ITEM_BYTES = 8 * 1024
MAX_SUBMISSION_BYTES = 256 * 1024
MAX_DETERMINISTIC_STEPS = 16
MAX_TRANSITIONS = 2_048
LEASE_SECONDS = 15 * 60

STATES = (
    "created",
    "planning",
    "executing",
    "waiting_result",
    "repairing",
    "finalizing",
    "complete_draft",
    "blocked",
    "cancelled",
)
TERMINAL_STATES = {"complete_draft", "blocked", "cancelled"}

TRANSITIONS = (
    ("plan", "created", "planning"),
    ("execute", "planning", "executing"),
    ("execute", "repairing", "executing"),
    ("resume", "waiting_result", "executing"),
    ("wait_result", "executing", "waiting_result"),
    ("repair", "executing", "repairing"),
    ("repair", "waiting_result", "repairing"),
    ("finalize", "planning", "finalizing"),
    ("finalize", "executing", "finalizing"),
    ("finalize", "repairing", "finalizing"),
    ("complete", "finalizing", "complete_draft"),
    ("block", "created", "blocked"),
    ("block", "planning", "blocked"),
    ("block", "executing", "blocked"),
    ("block", "waiting_result", "blocked"),
    ("block", "repairing", "blocked"),
    ("block", "finalizing", "blocked"),
    ("cancel", "created", "cancelled"),
    ("cancel", "planning", "cancelled"),
    ("cancel", "executing", "cancelled"),
    ("cancel", "waiting_result", "cancelled"),
    ("cancel", "repairing", "cancelled"),
    ("cancel", "finalizing", "cancelled"),
)

_MODEL_TOOLS = {"run_multimodal_frame_analysis", "run_temporal_visual_analysis"}
_SEMANTIC_MODEL_ACTIONS = {"semantic_candidate_discovery", "semantic_correction_review"}
_SEMANTIC_TASK_TYPES = {
    "transcript_semantic_candidate_discovery",
    "transcript_semantic_correction_review",
}
_MACHINE_STATUSES = {"repair_pending", "coverage_blocked", "semantic_correction_pending", "panorama_pending"}


class AutopilotError(RuntimeError):
    """Base error for fail-closed Autopilot operations."""


class AutopilotSubmissionError(AutopilotError):
    """Raised when a result cannot be bound to the current leased work item."""


@dataclass(frozen=True)
class ResourceSnapshot:
    sampled_at: str
    cpu_percent: float | None
    available_memory_bytes: int | None
    memory_available_ratio: float | None
    gpu_utilization_percent: float | None
    gpu_free_memory_ratio: float | None
    heavy_local_lease_active: bool
    issues: tuple[str, ...] = ()

    @property
    def busy_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.cpu_percent is not None and self.cpu_percent >= 70:
            reasons.append("cpu_at_or_above_70_percent")
        if self.available_memory_bytes is not None and self.available_memory_bytes < 8 * 1024**3:
            reasons.append("available_memory_below_8_gib")
        if self.memory_available_ratio is not None and self.memory_available_ratio < 0.20:
            reasons.append("available_memory_below_20_percent")
        if self.gpu_utilization_percent is not None and self.gpu_utilization_percent >= 70:
            reasons.append("gpu_at_or_above_70_percent")
        if self.gpu_free_memory_ratio is not None and self.gpu_free_memory_ratio < 0.25:
            reasons.append("gpu_free_memory_below_25_percent")
        if self.heavy_local_lease_active:
            reasons.append("vkp_heavy_local_lease_active")
        return reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampled_at": self.sampled_at,
            "sampled_at_unix": time.time(),
            "cpu_percent": self.cpu_percent,
            "available_memory_bytes": self.available_memory_bytes,
            "memory_available_ratio": self.memory_available_ratio,
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "gpu_free_memory_ratio": self.gpu_free_memory_ratio,
            "heavy_local_lease_active": self.heavy_local_lease_active,
            "busy": bool(self.busy_reasons),
            "busy_reasons": self.busy_reasons,
            "issues": list(self.issues),
        }


def autopilot_start(
    bundle_or_batch: str | Path,
    *,
    runner: str = "agent-session",
    execution_route: str = "auto",
    job_root: str | Path | None = None,
    online_derived_artifacts_allowed: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Create a deterministic Autopilot job without executing model work."""

    runner_name = _choice(runner, {"agent-session", "background"}, "runner")
    requested_route = _choice(
        execution_route,
        {"auto", "hybrid", "local-only", "online-only"},
        "execution_route",
    )
    source = Path(bundle_or_batch).expanduser().resolve()
    if source.is_file():
        batch = _read_batch_manifest(source)
        return _start_batch(
            source,
            batch,
            runner=runner_name,
            execution_route=requested_route,
            job_root=job_root,
            online_derived_artifacts_allowed=online_derived_artifacts_allowed,
            write=write,
        )
    return _start_bundle(
        source,
        runner=runner_name,
        execution_route=requested_route,
        job_root=job_root,
        online_derived_artifacts_allowed=online_derived_artifacts_allowed,
        write=write,
    )


def autopilot_next(
    job_id: str,
    *,
    job_root: str | Path | None = None,
    max_deterministic_steps: int = MAX_DETERMINISTIC_STEPS,
    background_executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    retry_provider: bool = False,
) -> dict[str, Any]:
    """Advance deterministic work and return at most one leased model task."""

    pointer = _load_pointer(job_id, job_root=job_root)
    if pointer.get("kind") == "batch":
        return _next_batch(
            pointer,
            job_root=job_root,
            background_executor=background_executor,
            retry_provider=retry_provider,
        )
    root, job_path = _pointer_paths(pointer)
    with bundle_write_lock(root, operation="autopilot_next", timeout_seconds=1.0):
        job = _read_job(job_path)
        if job["state"] in TERMINAL_STATES:
            if job["state"] == "complete_draft":
                # Older jobs stopped at ASR/preview completion. Re-enter only
                # the local summary handoff/export tail, never model execution.
                if not _await_summary_agent(root, job):
                    from .workflow_completion import finalize_readable_delivery
                    job["completion"] = finalize_readable_delivery(root)
                _save_job(job_path, job)
                _sync_run_registry(root, job)
            status = _public_status(job)
            if retry_provider:
                status["retry_provider"] = {
                    "status": "not_applicable",
                    "mutated": False,
                    "reason": "--retry-provider 只重试仍持有 provider_wait 租约的模型任务；complete_draft 没有待重试 Provider 调用。",
                    "next_action": "先完成 review gap 中的确定性恢复动作，再显式启动新的 Autopilot job。",
                }
            return status
        current = job.get("current_work_item")
        if isinstance(current, dict):
            if _lease_valid(current):
                provider_wait = current.get("provider_wait") if isinstance(current.get("provider_wait"), dict) else {}
                if provider_wait and retry_provider and job.get("runner") == "background":
                    executor = background_executor or _default_background_executor(job)
                    if executor is None:
                        return _next_response(job, work_item=current)
                    current.pop("provider_wait", None)
                    _event(job, "provider_retry_requested", task_id=current.get("task_id", ""))
                    _save_job(job_path, job)
                    submission = executor(current)
                    return autopilot_submit(job["job_id"], submission=submission, job_root=job_root)
                return _next_response(job, work_item=current)
            _record_gap(job, "expired_model_lease", "模型任务租约已过期，已生成新租约。", action=current)
            job["current_work_item"] = None
            if job["state"] == "waiting_result":
                _move(job, "resume")

        limit = max(1, min(int(max_deterministic_steps or 1), MAX_DETERMINISTIC_STEPS))
        for _ in range(limit):
            _assert_transition_budget(job)
            if job["state"] == "created":
                _move(job, "plan")
                _event(job, "planning_started")
                continue
            if job["state"] == "planning":
                _move(job, "execute")
                continue
            if job["state"] == "repairing":
                _move(job, "execute")
                continue
            if job["state"] == "finalizing":
                if _await_summary_agent(root, job):
                    _save_job(job_path, job)
                    _sync_run_registry(root, job)
                    return _public_status(job)
                from .workflow_completion import finalize_readable_delivery
                job["completion"] = finalize_readable_delivery(root)
                _move(job, "complete")
                _event(job, "complete_draft", review_gap_count=len(job["review_gaps"]))
                _save_job(job_path, job)
                _sync_run_registry(root, job)
                return _public_status(job)
            if job["state"] != "executing":
                break

            try:
                next_result = bundle_next_action(root, refresh=True)
            except (FileNotFoundError, PermissionError, ValueError) as exc:
                _hard_block(job, "bundle_unreadable", str(exc))
                _save_job(job_path, job)
                _sync_run_registry(root, job)
                return _public_status(job)
            action = next_result.get("next_action") if isinstance(next_result.get("next_action"), dict) else {}
            job["completion"] = next_result.get("completion") or {}
            if action.get("kind") == "smart_summary_handoff":
                _await_summary_agent(root, job)
                _save_job(job_path, job)
                _sync_run_registry(root, job)
                return _public_status(job)
            fingerprint = _action_fingerprint(root, action)
            if _fingerprint_is_resolved(job, fingerprint):
                if str(action.get("kind") or "") == "panorama_review_gap":
                    next_result = bundle_next_action(root, refresh=True, include_panorama=False)
                    action = next_result.get("next_action") if isinstance(next_result.get("next_action"), dict) else {}
                    fingerprint = _action_fingerprint(root, action)
                else:
                    _move(job, "finalize")
                    continue

            if str(action.get("kind") or "") == "panorama_blocker":
                _hard_block(job, str(action.get("key") or "panorama_blocked"), str(action.get("reason") or "全景投影身份不明确"))
                _save_job(job_path, job)
                _sync_run_registry(root, job)
                return _public_status(job)

            if str(action.get("kind") or "") == "panorama_review_gap":
                _record_gap(
                    job,
                    str(action.get("key") or "panorama_review_gap"),
                    str(action.get("reason") or "全景视觉证据不可用"),
                    action=action,
                )
                _mark_fingerprint_resolved(job, fingerprint, "review_gap")
                _event(job, "panorama_review_gap_recorded", key=action.get("key", ""))
                continue

            if _is_completion_action(action):
                refresh_args = root / "mcp-refresh-lecture-review.args.json"
                if refresh_args.is_file():
                    refreshed = bundle_advance(root, execute=False, refresh_outputs=True)
                    _event(job, "final_outputs_refreshed", advanced=bool(refreshed.get("advanced")))
                else:
                    _record_gap(
                        job,
                        "export_refresh_args_missing",
                        "Bundle 已处于 ready，但缺少本地导出刷新参数；保留为人工复核缺口。",
                        action=action,
                    )
                _move(job, "finalize")
                continue

            if _needs_model_task(action):
                budget_blocker = _model_budget_blocker(job)
                if budget_blocker:
                    _record_gap(job, "model_budget_exhausted", budget_blocker, action=action)
                    _mark_fingerprint_resolved(job, fingerprint, "review_gap")
                    _event(job, "model_execution_skipped", reason="budget_exhausted")
                    continue
                effective_background_executor = background_executor or _default_background_executor(job)
                interactive_agent_task = (
                    _is_semantic_model_action(action)
                    or _is_summary_review_action(action)
                    or str(action.get("mcp_tool") or "") in {
                        "prepare_panorama_vision_session", "prepare_panorama_temporal_session"
                    }
                )
                if (
                    job["runner"] == "background"
                    and effective_background_executor is None
                    and not interactive_agent_task
                ):
                    _record_gap(
                        job,
                        "background_executor_not_authorized",
                        "没有可用且只指向 loopback 的本地视觉执行路由；Autopilot 未发送任何派生帧或文本。",
                        action=action,
                    )
                    _mark_fingerprint_resolved(job, fingerprint, "review_gap")
                    _event(job, "background_execution_skipped", reason="executor_not_authorized")
                    continue
                try:
                    work_item = _prepare_work_item(root, job, action, fingerprint=fingerprint)
                except ProviderVisionTargetsExhausted as exc:
                    _record_gap(
                        job,
                        "provider_vision_targets_exhausted",
                        str(exc),
                        action=action,
                    )
                    _mark_fingerprint_resolved(job, fingerprint, "review_gap")
                    _event(job, "provider_vision_targets_exhausted", detail=str(exc)[:1000])
                    _move(job, "finalize")
                    continue
                job["current_work_item"] = work_item
                _move(job, "wait_result")
                _event(job, "work_item_leased", task_id=work_item["task_id"], task_type=work_item["task_type"])
                _save_job(job_path, job)
                _write_work_item(root, job, work_item)
                _sync_run_registry(root, job)
                if (
                    job["runner"] == "background"
                    and effective_background_executor is not None
                    and not interactive_agent_task
                ):
                    submission = effective_background_executor(work_item)
                    _save_job(job_path, job)
                    return autopilot_submit(job["job_id"], submission=submission, job_root=job_root)
                return _next_response(job, work_item=work_item)

            if _requires_human(action):
                _record_gap(job, "human_review_required", str(action.get("reason") or "需要人工复核"), action=action)
                _mark_fingerprint_resolved(job, fingerprint, "review_gap")
                _move(job, "finalize")
                continue

            advanced = _run_deterministic_action(root, action)
            _record_action_attempt(job, fingerprint, advanced)
            if advanced:
                _event(job, "deterministic_action_completed", key=action.get("key", ""))
                continue
            if _fingerprint_attempts(job, fingerprint) >= 2:
                job["no_progress"] = {
                    "status": "no-progress", "action": _bounded_action(action),
                    "reason": "The same task produced no evidence/count/content change; status timestamps are not progress.",
                    "next_action": action.get("command") or action.get("reason") or "Inspect the recorded blocker before retrying.",
                    "attempts": _fingerprint_attempts(job, fingerprint),
                }
                _record_gap(job, "deterministic_action_stalled", str(action.get("reason") or "动作未产生新产物"), action=action)
                _mark_fingerprint_resolved(job, fingerprint, "review_gap")
                _move(job, "finalize")
                continue
            _move(job, "repair")

        _save_job(job_path, job)
        _sync_run_registry(root, job)
        return _next_response(job)


def autopilot_submit(
    job_id: str,
    *,
    input_json: str | Path | None = None,
    submission: Mapping[str, Any] | None = None,
    job_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a leased result completely, then import it once under Bundle lock."""

    pointer = _load_pointer(job_id, job_root=job_root)
    if pointer.get("kind") != "bundle":
        raise AutopilotSubmissionError("submit requires a child Bundle job, not a batch job")
    root, job_path = _pointer_paths(pointer)
    payload, raw_bytes, input_path, parse_issues = _submission_payload(input_json=input_json, submission=submission)
    run_dir = job_path.parent
    raw_dir = run_dir / "raw-results"
    validation_dir = run_dir / "validation"
    raw_path = raw_dir / f"submission-{time.time_ns()}.json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw_bytes)

    with bundle_write_lock(root, operation="autopilot_submit", timeout_seconds=1.0):
        job = _read_job(job_path)
        work_item = job.get("current_work_item") if isinstance(job.get("current_work_item"), dict) else {}
        issues = [*parse_issues, *_validate_submission(root, job, work_item, payload, raw_bytes)]
        if issues:
            receipt = _validation_receipt(job, payload, raw_path, input_path, status="rejected", issues=issues)
            validation_path = validation_dir / f"rejected-{time.time_ns()}.json"
            write_json(validation_path, receipt)
            _event(job, "submission_rejected", issues=issues, raw_result_path=str(raw_path))
            _record_submission_rejection(job, work_item)
            _save_job(job_path, job)
            _sync_run_registry(root, job)
            return {**receipt, "job": _public_status(job), "validation_path": str(validation_path)}

        task_type = str(work_item.get("task_type") or "")
        import_result: dict[str, Any] = {}
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        result_status = str(result.get("status") or "")
        if task_type == "provider_vision_session" and result_status == "retryable":
            provider_call_count = max(1, int(result.get("provider_call_count") or 1))
            _record_model_call_budget(job, work_item, provider_call_count)
            work_item["provider_wait"] = {
                "reason": str(result.get("reason") or "local_provider_failed")[:200],
                "detail": str(result.get("detail") or "")[:4000],
                "recovery": str(result.get("recovery") or "")[:1000],
                "provider_call_count": provider_call_count,
                "created_at": now_iso(),
            }
            receipt = _validation_receipt(
                job,
                payload,
                raw_path,
                input_path,
                status="waiting_provider",
                issues=[],
            )
            validation_path = validation_dir / f"waiting-provider-{time.time_ns()}.json"
            write_json(validation_path, receipt)
            _event(
                job,
                "provider_waiting",
                task_id=work_item.get("task_id", ""),
                reason=work_item["provider_wait"]["reason"],
                raw_result_path=str(raw_path),
            )
            _save_job(job_path, job)
            _sync_run_registry(root, job)
            return {**receipt, "job": _public_status(job), "validation_path": str(validation_path)}
        if task_type == "provider_vision_session":
            issues.extend(validate_provider_vision_result(work_item, result))
        if task_type == "codex_vision_session":
            import_input = raw_dir / f"codex-result-{work_item['task_id']}.json"
            write_json(import_input, result)
            import_result = import_codex_vision_results(root, input_json=import_input, write=True)
            if str(import_result.get("status") or "") not in {"imported", "succeeded", "complete"}:
                issues = [{"key": "codex_import_rejected", "detail": str(import_result.get("status") or "rejected")}]
        elif task_type == "provider_vision_session":
            if result_status == "succeeded" and not issues:
                import_result = import_provider_vision_result(root, result=result, output_dir=raw_dir)
                if str(import_result.get("status") or "") != "imported":
                    issues = [{"key": "provider_vision_import_rejected", "detail": str(import_result.get("status") or "rejected")}]
            elif result_status in {"unavailable", "review_gap"}:
                import_result = {
                    "status": "skipped_review_gap",
                    "reason": str(result.get("reason") or "model_result_unavailable")[:200],
                    "timeline_changed": False,
                }
            elif not issues:
                issues = [{"key": "provider_result_status_invalid", "detail": result_status or "missing"}]
        elif task_type == "summary_claim_review":
            from .summary_claim_review import import_summary_claim_review
            import_input = raw_dir / f"claim-review-{work_item['task_id']}.json"
            write_json(import_input, result)
            import_result = import_summary_claim_review(root, import_input)
            if not import_result.get('ok'):
                issues = [{'key': 'summary_claim_review_import_rejected',
                           'detail': str(import_result.get('errors') or 'rejected')[:1000]}]
        elif task_type in _SEMANTIC_TASK_TYPES:
            import_result, semantic_issues, result_status = _import_semantic_submission(
                root,
                work_item,
                result,
                raw_dir,
            )
            issues.extend(semantic_issues)
        else:
            status = result_status
            if status not in {"succeeded", "unavailable", "review_gap"}:
                issues = [{"key": "result_status_invalid", "detail": "expected succeeded, unavailable, or review_gap"}]

        if issues:
            receipt = _validation_receipt(job, payload, raw_path, input_path, status="rejected", issues=issues)
            validation_path = validation_dir / f"rejected-{time.time_ns()}.json"
            write_json(validation_path, receipt)
            _event(job, "submission_import_rejected", issues=issues, raw_result_path=str(raw_path))
            _record_submission_rejection(job, work_item)
            _save_job(job_path, job)
            _sync_run_registry(root, job)
            return {**receipt, "job": _public_status(job), "validation_path": str(validation_path), "import_result": import_result}

        fingerprint = str(work_item.get("action_fingerprint") or "")
        if task_type not in _SEMANTIC_TASK_TYPES:
            result_status = str(result.get("status") or "succeeded")
        if result_status in {"unavailable", "review_gap"}:
            _record_gap(
                job,
                str(
                    result.get("reason")
                    or (
                        "semantic_correction_review_gap"
                        if task_type in _SEMANTIC_TASK_TYPES
                        else "model_result_unavailable"
                    )
                ),
                str(
                    result.get("detail")
                    or result.get("unavailable_reason")
                    or import_result.get("status")
                    or "模型已处理但无法可靠判断"
                ),
                action=work_item.get("action") if isinstance(work_item.get("action"), dict) else {},
            )
        if task_type == "transcript_semantic_candidate_discovery":
            _record_semantic_discovery_progress(job, work_item)
            # Candidate import changes the authoritative pack.  Any prior
            # run-local adjudication cursor is stale and must be rebuilt.
            job["semantic_source_pack_sha256"] = ""
            job["semantic_adjudicated_candidate_ids"] = []
        elif task_type == "transcript_semantic_correction_review":
            semantic_task = work_item.get("semantic_task") if isinstance(work_item.get("semantic_task"), dict) else {}
            adjudicated = {
                str(value)
                for value in job.get("semantic_adjudicated_candidate_ids") or []
                if str(value)
            }
            adjudicated.update(
                str(value)
                for value in semantic_task.get("selected_candidate_ids") or []
                if str(value)
            )
            job["semantic_adjudicated_candidate_ids"] = sorted(adjudicated)
            job["semantic_source_pack_sha256"] = str(
                semantic_task.get("source_pack_sha256") or job.get("semantic_source_pack_sha256") or ""
            )
        if task_type == "provider_vision_session":
            _record_provider_vision_target_outcomes(
                job,
                work_item,
                status=result_status,
                reason=str(result.get("reason") or ""),
                detail=str(result.get("detail") or result.get("unavailable_reason") or ""),
                provider_call_count=int(result.get("provider_call_count") or 0),
            )
        if not (
            task_type == "provider_vision_session"
            and result_status in {"unavailable", "review_gap"}
        ):
            _mark_fingerprint_resolved(job, fingerprint, result_status)
        job["current_work_item"] = None
        _move(job, "resume")
        provider_call_count = max(1, int(result.get("provider_call_count") or 1))
        _record_model_call_budget(job, work_item, provider_call_count)
        receipt = _validation_receipt(job, payload, raw_path, input_path, status="succeeded", issues=[])
        receipt["import_result"] = import_result
        validation_path = validation_dir / f"accepted-{time.time_ns()}.json"
        write_json(validation_path, receipt)
        _event(job, "submission_accepted", task_id=work_item["task_id"], raw_result_path=str(raw_path))
        _save_job(job_path, job)
        _sync_run_registry(root, job)
        return {**receipt, "job": _public_status(job), "validation_path": str(validation_path)}


def autopilot_status(job_id: str, *, job_root: str | Path | None = None) -> dict[str, Any]:
    pointer = _load_pointer(job_id, job_root=job_root)
    if pointer.get("kind") == "batch":
        return _batch_status(pointer, job_root=job_root)
    _, job_path = _pointer_paths(pointer)
    return _public_status(_read_job(job_path))


def autopilot_bundle_summary(bundle_dir: str | Path) -> dict[str, Any]:
    """Read the latest Bundle-local Autopilot job for static UI surfaces."""

    root = Path(bundle_dir).expanduser().resolve()
    runs_dir = root / "runs"
    paths = (
        sorted(
            runs_dir.glob("autopilot-*/autopilot-job.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        if runs_dir.is_dir()
        else []
    )
    if not paths:
        bundle = "'" + str(root).replace("'", "''") + "'"
        return {
            "status": "not_started",
            "state": "not_started",
            "start_command": f".\\scripts\\video-knowledge.ps1 autopilot-start {bundle}",
            "accepted_written": False,
            "paths": {},
        }
    job = _read_job(paths[0])
    summary = _public_status(job)
    run_dir = paths[0].parent
    raw_dir = run_dir / "raw-results"
    validation_dir = run_dir / "validation"
    raw = sorted(raw_dir.glob("*.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True) if raw_dir.is_dir() else []
    validation = (
        sorted(validation_dir.glob("*.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True)
        if validation_dir.is_dir()
        else []
    )
    summary["paths"] = {
        **dict(summary.get("paths") or {}),
        "run": str(run_dir / "run.json"),
        "latest_raw_result": str(raw[0]) if raw else "",
        "latest_validation": str(validation[0]) if validation else "",
    }
    summary["next_command"] = f".\\scripts\\video-knowledge.ps1 autopilot-next {job['job_id']}"
    summary["status_command"] = f".\\scripts\\video-knowledge.ps1 autopilot-status {job['job_id']}"
    return summary


def autopilot_run(
    bundle_or_batch: str | Path,
    *,
    runner: str = "agent-session",
    execution_route: str = "auto",
    resume: str = "",
    job_root: str | Path | None = None,
    background_executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    retry_provider: bool = False,
) -> dict[str, Any]:
    started = autopilot_status(resume, job_root=job_root) if resume else autopilot_start(
        bundle_or_batch,
        runner=runner,
        execution_route=execution_route,
        job_root=job_root,
    )
    job_id = str(started.get("job_id") or "")
    retried_provider_tasks: set[str] = set()
    for _ in range(MAX_TRANSITIONS):
        result = autopilot_next(
            job_id,
            job_root=job_root,
            background_executor=background_executor,
            retry_provider=retry_provider,
        )
        if result.get("state") in TERMINAL_STATES or result.get("status") in {
                "waiting_model", "awaiting-agent", "review-required", "blocked"}:
            return result
        if result.get("status") == "waiting_provider":
            retry_key = _loopback_provider_retry_key(result) if retry_provider else ""
            if retry_key and retry_key not in retried_provider_tasks:
                retried_provider_tasks.add(retry_key)
                continue
            return result
    raise AutopilotError("autopilot transition budget exhausted")


def _record_model_call_budget(job: dict[str, Any], work_item: Mapping[str, Any], provider_call_count: int) -> None:
    count = max(1, int(provider_call_count or 1))
    job["budget_usage"]["model_calls"] += count
    route = str(work_item.get("model_policy", {}).get("resolved_route") or "")
    if route.startswith("online"):
        job["budget_usage"]["online_calls"] += count
    else:
        job["budget_usage"]["local_calls"] += count


def _loopback_provider_retry_key(result: Mapping[str, Any]) -> str:
    """Return the leased task id only for a fail-closed loopback provider wait.

    ``autopilot-run --retry-provider`` is an explicit run-level authorization to
    retry each local leased task once.  The in-memory set in ``autopilot_run``
    prevents a slow or unavailable provider from turning that authorization into
    an unbounded loop.  Remote routes are never eligible here.
    """

    job = result.get("job") if isinstance(result.get("job"), Mapping) else result
    if str(job.get("runner") or "") != "background":
        return ""
    current = job.get("current_work_item")
    if not isinstance(current, Mapping) or str(current.get("task_type") or "") != "provider_vision_session":
        return ""
    if not isinstance(current.get("provider_wait"), Mapping):
        return ""
    snapshot = job.get("local_vision_provider")
    if not isinstance(snapshot, Mapping) or bool(snapshot.get("remote_fallback_allowed")):
        return ""
    provider = snapshot.get("provider")
    if not isinstance(provider, Mapping):
        return ""
    base_url = str(provider.get("base_url") or "").strip().lower()
    if not (
        base_url.startswith("http://127.0.0.1")
        or base_url.startswith("http://localhost")
        or base_url.startswith("http://[::1]")
    ):
        return ""
    return str(current.get("task_id") or result.get("task_id") or "")


def resource_snapshot(*, heavy_local_lease: str | Path | None = None) -> ResourceSnapshot:
    issues: list[str] = []
    try:
        cpu = float(psutil.cpu_percent(interval=None))
    except Exception as exc:  # pragma: no cover - platform probe failure
        cpu = None
        issues.append(f"cpu_probe_failed:{type(exc).__name__}")
    try:
        memory = psutil.virtual_memory()
        available = int(memory.available)
        ratio = float(memory.available / memory.total) if memory.total else None
    except Exception as exc:  # pragma: no cover - platform probe failure
        available = None
        ratio = None
        issues.append(f"memory_probe_failed:{type(exc).__name__}")
    gpu_util, gpu_free_ratio, gpu_issues = _gpu_snapshot()
    issues.extend(gpu_issues)
    lease_active = bool(heavy_local_lease and Path(heavy_local_lease).expanduser().exists())
    return ResourceSnapshot(
        sampled_at=now_iso(),
        cpu_percent=cpu,
        available_memory_bytes=available,
        memory_available_ratio=ratio,
        gpu_utilization_percent=gpu_util,
        gpu_free_memory_ratio=gpu_free_ratio,
        heavy_local_lease_active=lease_active,
        issues=tuple(issues),
    )


def resolve_execution_route(
    requested: str,
    *,
    duration_seconds: float = 0.0,
    estimated_local_seconds: float = 0.0,
    resource_samples: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    requested_route = _choice(requested, {"auto", "hybrid", "local-only", "online-only"}, "execution_route")
    samples = list(resource_samples or [])[-2:]
    busy = len(samples) == 2 and all(bool(row.get("busy")) for row in samples)
    reasons = sorted({str(reason) for row in samples for reason in row.get("busy_reasons") or []})
    long_video = float(duration_seconds or 0) >= 90 * 60 or float(estimated_local_seconds or 0) > 120 * 60
    if requested_route in {"local-only", "online-only", "hybrid"}:
        resolved = requested_route
        basis = "explicit"
    elif busy or long_video:
        resolved = "online-heavy"
        basis = "resource_busy" if busy else "long_video"
    else:
        resolved = "hybrid"
        basis = "default"
    return {
        "requested_route": requested_route,
        "resolved_route": resolved,
        "basis": basis,
        "long_video": long_video,
        "resource_busy": busy,
        "busy_reasons": reasons,
    }


def _start_bundle(
    root: Path,
    *,
    runner: str,
    execution_route: str,
    job_root: str | Path | None,
    online_derived_artifacts_allowed: bool,
    write: bool,
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.is_file() or not timeline_path.is_file():
        missing = [str(path) for path in (manifest_path, timeline_path) if not path.is_file()]
        raise FileNotFoundError("Autopilot requires an existing Bundle manifest and timeline: " + ", ".join(missing))
    job_id = uuid4().hex
    run_id = f"autopilot-{job_id}"
    run_dir = root / "runs" / run_id
    job_path = run_dir / "autopilot-job.json"
    pointer_path = _job_root(job_root) / f"{job_id}.json"
    duration = _bundle_duration_seconds(root)
    samples = [resource_snapshot(heavy_local_lease=root / ".autopilot-heavy-local.lock").to_dict()]
    route = resolve_execution_route(execution_route, duration_seconds=duration, resource_samples=samples)
    policy = {
        "schema": "video_knowledge_pipeline.autopilot_policy_snapshot.v1",
        "created_at": now_iso(),
        "immutable": True,
        "online_derived_artifacts_allowed": bool(online_derived_artifacts_allowed),
        "allowed_online_artifacts": ["bundle_task_pack", "extracted_frame", "bounded_audio_snippet", "bounded_text_snippet"],
        "forbidden_online_artifacts": ["source_video", "full_original_audio", "bundle_external_path", "unregistered_file"],
        "local_vision_provider": local_vision_provider_snapshot(),
        "budgets": {
            "local_calls_per_video": 512,
            "local_attempts_per_task": 6,
            "local_max_seconds": 24 * 60 * 60,
            "online_calls_per_video": 64,
            "online_attempts_per_task": 2,
            "online_estimated_usd": 3.0,
            "global_online_concurrency": 4,
            "per_destination_online_concurrency": 2,
            "max_state_transitions": MAX_TRANSITIONS,
        },
    }
    job: dict[str, Any] = {
        "schema": JOB_SCHEMA,
        "job_id": job_id,
        "run_id": run_id,
        "bundle_dir": str(root),
        "job_path": str(job_path),
        "state": "created",
        "state_revision": 0,
        "transition_count": 0,
        "runner": runner,
        "execution_route": route,
        "policy_snapshot": policy,
        "resource_samples": samples,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "heartbeat_at": now_iso(),
        "bundle_identity_at_start": "",
        "current_work_item": None,
        "review_gaps": [],
        "provider_vision_target_outcomes": {},
        "semantic_candidate_discovery": {},
        "action_history": {},
        "resolved_action_fingerprints": {},
        "budget_usage": {"model_calls": 0, "local_calls": 0, "online_calls": 0, "estimated_online_usd": 0.0},
        "events_path": str(run_dir / "events.jsonl"),
        "restore_plan_path": str(run_dir / "restore-plan.json"),
    }
    if write:
        with bundle_write_lock(root, operation="autopilot_start", timeout_seconds=1.0):
            register_bundle_run(
                root,
                run_type="autopilot",
                run_id=run_id,
                status="planned",
                title="VKP small-model Autopilot",
                summary="Deterministic controller; accepted remains an explicit human action.",
                inputs={"manifest": str(manifest_path), "timeline": str(timeline_path)},
                parameters={"runner": runner, "execution_route": route, "policy_snapshot": policy},
                artifacts=[{"key": "autopilot_job", "path": str(job_path)}],
                operator_boundary={"does_not_write_accepted": True, "source_media_immutable": True},
                write=True,
            )
            job["bundle_identity_at_start"] = _bundle_identity(root)
            write_json(Path(job["restore_plan_path"]), _initial_restore_plan(root, job))
            _event(job, "job_created")
            _save_job(job_path, job)
            write_json(
                pointer_path,
                {"schema": "video_knowledge_pipeline.autopilot_job_pointer.v1", "kind": "bundle", "job_id": job_id, "bundle_dir": str(root), "job_path": str(job_path)},
            )
    return _public_status(job)


def _start_batch(
    source: Path,
    batch: dict[str, Any],
    *,
    runner: str,
    execution_route: str,
    job_root: str | Path | None,
    online_derived_artifacts_allowed: bool,
    write: bool,
) -> dict[str, Any]:
    batch_id = uuid4().hex
    children: list[str] = []
    for value in batch["bundles"]:
        path = Path(str(value)).expanduser()
        root = path.resolve() if path.is_absolute() else (source.parent / path).resolve()
        child = _start_bundle(
            root,
            runner=runner,
            execution_route=execution_route,
            job_root=job_root,
            online_derived_artifacts_allowed=online_derived_artifacts_allowed,
            write=write,
        )
        children.append(str(child["job_id"]))
    pointer_path = _job_root(job_root) / f"{batch_id}.json"
    pointer = {
        "schema": BATCH_SCHEMA,
        "kind": "batch",
        "job_id": batch_id,
        "batch_manifest": str(source),
        "children": children,
        "created_at": now_iso(),
    }
    if write:
        write_json(pointer_path, pointer)
    return _batch_status(pointer, job_root=job_root)


def _read_batch_manifest(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("bundles"), list) or not value["bundles"]:
        raise ValueError("batch manifest must be an object with a non-empty bundles list")
    return value


def _next_batch(
    pointer: dict[str, Any],
    *,
    job_root: str | Path | None,
    background_executor: Callable[[dict[str, Any]], dict[str, Any]] | None,
    retry_provider: bool = False,
) -> dict[str, Any]:
    for child_id in pointer.get("children") or []:
        status = autopilot_status(str(child_id), job_root=job_root)
        if status.get("state") not in TERMINAL_STATES or (
            status.get("state") == "complete_draft" and not (status.get("completion") or {}).get("completed")):
            child = autopilot_next(
                str(child_id),
                job_root=job_root,
                background_executor=background_executor,
                retry_provider=retry_provider,
            )
            return {**_batch_status(pointer, job_root=job_root), "active_child": child}
    return _batch_status(pointer, job_root=job_root)


def _batch_status(pointer: dict[str, Any], *, job_root: str | Path | None) -> dict[str, Any]:
    children = [autopilot_status(str(job_id), job_root=job_root) for job_id in pointer.get("children") or []]
    counts: dict[str, int] = {}
    for child in children:
        state = str(child.get("state") or "unknown")
        counts[state] = counts.get(state, 0) + 1
    if children and all(child.get("state") == "complete_draft" for child in children):
        state = "complete_draft"
    elif any(child.get("state") == "blocked" for child in children):
        state = "partially_blocked"
    else:
        state = "executing"
    complete = bool(children) and all((child.get("completion") or {}).get("completed") is True for child in children)
    status = "completed" if complete else "awaiting-agent" if any(child.get("status") == "awaiting-agent" for child in children) else "review-required" if state == "complete_draft" else state
    return {"schema": BATCH_SCHEMA, "job_id": pointer["job_id"], "state": state,
            "status": status, "completed": complete, "status_counts": counts, "children": children}


def _prepare_semantic_pending_pack(
    root: Path,
    job: dict[str, Any],
) -> tuple[Path, Path, str]:
    """Create a run-local view containing only not-yet-adjudicated candidates."""

    source_path = (root / "transcript-semantic-correction-pack.json").resolve()
    source = read_json(source_path)
    if not isinstance(source, dict) or not isinstance(source.get("candidates"), list):
        raise AutopilotError("semantic correction source pack is missing or invalid")
    source_sha256 = sha256_file(source_path)
    if str(job.get("semantic_source_pack_sha256") or "") != source_sha256:
        job["semantic_source_pack_sha256"] = source_sha256
        job["semantic_adjudicated_candidate_ids"] = []
    adjudicated = {
        str(value)
        for value in job.get("semantic_adjudicated_candidate_ids") or []
        if str(value)
    }
    adjudicated.update(_semantic_artifact_decision_ids(root))
    pending = [
        dict(row)
        for row in source.get("candidates") or []
        if isinstance(row, dict)
        and str(row.get("candidate_id") or "")
        and str(row.get("candidate_id") or "") not in adjudicated
    ]
    view = dict(source)
    view.update(
        {
            "pack_profile": "autopilot_pending_view_v1",
            "source_pack_json": str(source_path),
            "source_pack_sha256": source_sha256,
            "candidate_count": len(pending),
            "candidates": pending,
            "candidate_groups": [],
            "operator_boundary": {
                **(
                    source.get("operator_boundary")
                    if isinstance(source.get("operator_boundary"), dict)
                    else {}
                ),
                "run_local_pending_view": True,
                "does_not_replace_source_pack": True,
            },
        }
    )
    run_dir = Path(str(job["job_path"])).resolve().parent
    path = run_dir / "semantic-correction-pending-pack.json"
    write_json(path, view)
    return path, source_path, source_sha256


def _semantic_artifact_decision_ids(root: Path) -> set[str]:
    decided: set[str] = set()
    for path in (
        root / "transcript-semantic-correction-validation.json",
        root / "transcript-semantic-correction-decision-ledger.json",
    ):
        if not path.is_file():
            continue
        value = read_json(path)
        if not isinstance(value, dict):
            continue
        for row in value.get("decisions") or []:
            if isinstance(row, dict) and str(row.get("candidate_id") or ""):
                decided.add(str(row["candidate_id"]))
    return decided


def _semantic_discovery_cursor(
    job: dict[str, Any],
    preview: Mapping[str, Any],
) -> int:
    """Return the current page cursor, resetting progress on transcript drift."""

    current_sha256 = str(preview.get("transcript_sha256") or "")
    preview_cursor = max(0, int(preview.get("cursor") or 0))
    state = job.get("semantic_candidate_discovery")
    if not isinstance(state, dict):
        state = {}
    stored_sha256 = str(state.get("transcript_sha256") or "")
    if stored_sha256 and current_sha256 and stored_sha256 != current_sha256:
        state = {
            "transcript_sha256": current_sha256,
            "next_cursor": 0,
            "complete": False,
            "coverage": {},
            "batches": [],
            "hash_drift_reset": True,
            "previous_transcript_sha256": stored_sha256,
            "updated_at": now_iso(),
        }
        job["semantic_candidate_discovery"] = state
        return 0
    state.setdefault("transcript_sha256", current_sha256)
    state.setdefault("next_cursor", preview_cursor)
    state.setdefault("complete", False)
    state.setdefault("coverage", {})
    state.setdefault("batches", [])
    job["semantic_candidate_discovery"] = state
    return preview_cursor


def _record_semantic_discovery_progress(
    job: dict[str, Any],
    work_item: Mapping[str, Any],
) -> None:
    """Persist one successfully imported discovery page in the Autopilot job."""

    task = work_item.get("semantic_task") if isinstance(work_item.get("semantic_task"), dict) else {}
    transcript_sha256 = str(task.get("transcript_sha256") or "")
    state = job.get("semantic_candidate_discovery")
    if not isinstance(state, dict) or (
        str(state.get("transcript_sha256") or "")
        and str(state.get("transcript_sha256") or "") != transcript_sha256
    ):
        state = {
            "transcript_sha256": transcript_sha256,
            "next_cursor": 0,
            "complete": False,
            "coverage": {},
            "batches": [],
        }
    coverage = task.get("coverage") if isinstance(task.get("coverage"), dict) else {}
    retained_count = int(coverage.get("retained_cue_count") or 0)
    covered_count = int(coverage.get("covered_retained_cue_count") or 0)
    next_cursor_value = task.get("next_cursor")
    declared_complete = bool(task.get("complete"))
    coverage_complete = covered_count >= retained_count
    complete = declared_complete and next_cursor_value is None and coverage_complete
    batch = {
        "cursor": max(0, int(task.get("cursor") or 0)),
        "next_cursor": None if complete else (
            max(0, int(next_cursor_value)) if next_cursor_value is not None else None
        ),
        "complete": complete,
        "coverage": coverage,
        "selected_source_indexes": [
            int(value)
            for value in task.get("selected_source_indexes") or []
            if str(value).lstrip("-").isdigit()
        ],
        "discovery_pack_sha256": str(task.get("discovery_pack_sha256") or ""),
        "imported_at": now_iso(),
    }
    batches = [row for row in state.get("batches") or [] if isinstance(row, dict)]
    if not any(
        str(row.get("discovery_pack_sha256") or "")
        and str(row.get("discovery_pack_sha256") or "") == batch["discovery_pack_sha256"]
        for row in batches
    ):
        batches.append(batch)
    state.update(
        {
            "transcript_sha256": transcript_sha256,
            "next_cursor": batch["next_cursor"],
            "complete": complete,
            "coverage": coverage,
            "last_discovery_pack_sha256": batch["discovery_pack_sha256"],
            "batches": batches[-256:],
            "updated_at": now_iso(),
        }
    )
    job["semantic_candidate_discovery"] = state


def _prepare_work_item(root: Path, job: dict[str, Any], action: dict[str, Any], *, fingerprint: str) -> dict[str, Any]:
    task_id = uuid4().hex
    lease_id = uuid4().hex
    refs: list[dict[str, Any]] = []
    task_type = "structured_model_task"
    instructions = str(action.get("reason") or action.get("label") or "完成当前 VKP 模型任务")
    output_schema: dict[str, Any] = {
        "type": "object",
        "required": ["status"],
        "properties": {
            "status": {"enum": ["succeeded", "unavailable", "review_gap"]},
            "reason": {"type": "string", "maxLength": 200},
            "detail": {"type": "string", "maxLength": 4000},
            "model_actual": {"type": "string", "maxLength": 200},
        },
        "additionalProperties": True,
    }
    semantic_task: dict[str, Any] = {}
    action_key = str(action.get("key") or "")
    panorama_tool = str(action.get("mcp_tool") or "")
    if _is_summary_review_action(action):
        from .summary_claim_review import prepare_summary_claim_review
        pack = prepare_summary_claim_review(root)
        if not pack.get('task_id'):
            raise AutopilotError('summary review has no current task; refresh the projected action')
        task_type = 'summary_claim_review'
        instructions = ('读取 summary_claim_review_task 引用的完整单任务包，按其中 output_schema 提交 result；'
                        '必须保留真实 model_actual 和 executor_run_id，review 与 annotate 使用不同声明运行；'
                        '只观察本任务来源，不调用 Provider 或自行接受结论。')
        refs.append(_input_ref(root, root / f'runs/summary-claim-review/tasks/{pack["task_id"]}.json',
                               role='summary_claim_review_task'))
        # Like Codex vision, schema lives in the hash-bound immutable task and
        # is validated by its existing atomic importer; no external ref resolver.
        output_schema = {'$ref': 'video_knowledge_pipeline.summary_claim_review_result.v1'}
    elif action_key == "semantic_candidate_discovery":
        discovery_state = job.get("semantic_candidate_discovery") if isinstance(job.get("semantic_candidate_discovery"), dict) else {}
        requested_cursor = (
            int(discovery_state.get("next_cursor"))
            if discovery_state and not discovery_state.get("complete") and discovery_state.get("next_cursor") is not None
            else int(action.get("discovery_cursor") or 0)
        )
        preview = build_transcript_semantic_candidate_discovery_pack(
            root,
            limit=8,
            cursor=requested_cursor,
            write=False,
        )
        cursor = _semantic_discovery_cursor(job, preview)
        if cursor != int(preview.get("cursor") or 0):
            preview = build_transcript_semantic_candidate_discovery_pack(
                root,
                limit=8,
                cursor=cursor,
                write=False,
            )
        prepared = build_transcript_semantic_candidate_discovery_pack(
            root,
            limit=8,
            cursor=cursor,
            write=True,
        )
        artifacts = prepared.get("artifacts") if isinstance(prepared.get("artifacts"), dict) else {}
        pack_path = Path(
            str(
                artifacts.get("candidate_discovery_pack_json")
                or root / "transcript-semantic-candidate-discovery-pack.json"
            )
        ).resolve()
        prompt_path = Path(
            str(
                artifacts.get("candidate_discovery_prompt_markdown")
                or root / "transcript-semantic-candidate-discovery-prompt.md"
            )
        ).resolve()
        template_path = (root / "transcript-semantic-candidate-discovery-template.json").resolve()
        refs.append(_input_ref(root, pack_path, role="semantic_candidate_discovery_pack"))
        refs.append(_input_ref(root, prompt_path, role="semantic_candidate_discovery_prompt"))
        if template_path.is_file():
            refs.append(_input_ref(root, template_path, role="semantic_candidate_discovery_template"))
        transcript_path = Path(str(prepared.get("transcript_path") or "")).resolve()
        if transcript_path.is_file() and transcript_path.is_relative_to(root):
            refs.append(_input_ref(root, transcript_path, role="semantic_candidate_discovery_transcript"))
        task_type = "transcript_semantic_candidate_discovery"
        instructions = (
            "只扫描任务包中的最多 8 个有界转写窗口，返回疑似候选；不得直接改写转写、"
            "不得把建议当作已确认纠错。"
        )
        output_schema = _semantic_candidate_discovery_output_schema()
        semantic_task = {
            "segment_count": min(8, int(prepared.get("segment_count") or 0)),
            "max_suggestions": 8,
            "cursor": int(prepared.get("cursor") or 0),
            "next_cursor": prepared.get("next_cursor"),
            "complete": bool(prepared.get("complete")),
            "coverage": prepared.get("coverage") if isinstance(prepared.get("coverage"), dict) else {},
            "selected_source_indexes": [
                int(value)
                for value in prepared.get("selected_source_indexes") or []
                if str(value).lstrip("-").isdigit()
            ],
            "transcript_sha256": str(prepared.get("transcript_sha256") or ""),
            "discovery_pack_sha256": sha256_file(pack_path),
        }
    elif action_key == "semantic_correction_review":
        pending_pack_path, source_pack_path, source_pack_sha256 = _prepare_semantic_pending_pack(
            root,
            job,
        )
        prepared = build_transcript_semantic_correction_llm_draft(
            root,
            input_json=pending_pack_path,
            execute=False,
            limit=8,
            min_confidence=0.88,
            write=True,
        )
        gateway_pack_path = Path(str(prepared.get("gateway_pack_json") or root / "transcript-semantic-correction-gateway-pack.json")).resolve()
        prompt_path = Path(str(prepared.get("prompt_markdown") or root / "transcript-semantic-correction-llm-prompt.md")).resolve()
        refs.append(_input_ref(root, source_pack_path, role="semantic_correction_source_pack"))
        refs.append(_input_ref(root, pending_pack_path, role="semantic_correction_pending_pack"))
        refs.append(_input_ref(root, gateway_pack_path, role="semantic_correction_gateway_pack"))
        refs.append(_input_ref(root, prompt_path, role="semantic_correction_prompt"))
        selection = prepared.get("candidate_selection") if isinstance(prepared.get("candidate_selection"), dict) else {}
        selected_ids = [
            str(value)
            for value in selection.get("selected_candidate_ids") or []
            if str(value)
        ][:8]
        task_type = "transcript_semantic_correction_review"
        instructions = (
            "只裁决 compact gateway pack 中列出的候选，并为每个候选返回一个决定。"
            "所有结果仍须通过 VKP 现有强校验器；高风险或证据不足必须标 needs_human_review。"
        )
        output_schema = _semantic_correction_review_output_schema()
        semantic_task = {
            "selected_candidate_count": len(selected_ids),
            "selected_candidate_ids": selected_ids,
            "min_confidence": 0.88,
            "source_pack_sha256": source_pack_sha256,
        }
    elif (
        job.get("runner") == "background"
        and _local_background_ready(job)
        and panorama_tool not in {"prepare_panorama_vision_session", "prepare_panorama_temporal_session"}
        and (
            str(action.get("mcp_tool") or "") in _MODEL_TOOLS
            or str(action.get("kind") or "") == "codex_session"
        )
    ):
        snapshot = job["policy_snapshot"]["local_vision_provider"]
        provider_action = {
            **action,
            "excluded_indexes": _provider_vision_excluded_indexes(job, action),
            # Interactive Codex packs are immutable snapshots.  When background
            # Autopilot consumes the same action later, their indexes may all be
            # complete already; allow the adapter to choose the next currently
            # missing Bundle-owned frame without widening ordinary explicit runs.
            "allow_stale_index_fallback": str(action.get("kind") or "") == "codex_session",
        }
        pack = prepare_provider_vision_pack(
            root,
            run_dir=Path(job["job_path"]).parent,
            task_id=task_id,
            action=provider_action,
            provider_snapshot=snapshot,
        )
        pack_path = Path(str(pack["task_pack_path"])).resolve()
        refs.append(_input_ref(root, pack_path, role="provider_task_pack"))
        task_type = "provider_vision_session"
        instructions = "由 VKP 控制器通过现有 Provider Gateway 顺序执行；每次只发送一张 Bundle 派生帧，不允许远程回退。"
        output_schema = {
            "type": "object",
            "required": ["schema", "status"],
            "properties": {
                "schema": {"const": "video_knowledge_pipeline.autopilot_provider_vision_result.v1"},
                "status": {"enum": ["succeeded", "retryable", "unavailable", "review_gap"]},
                "reason": {"type": "string", "maxLength": 200},
                "detail": {"type": "string", "maxLength": 4000},
                "model_requested": {"type": "string", "maxLength": 200},
                "model_actual": {"type": "string", "maxLength": 200},
                "provider_call_count": {"type": "integer", "minimum": 0, "maximum": 512},
                "items": {"type": "array", "maxItems": 19},
            },
            "additionalProperties": True,
        }
    elif panorama_tool in {"prepare_panorama_vision_session", "prepare_panorama_temporal_session"}:
        args_path = Path(str(action.get("mcp_args_path") or ""))
        args = read_json(args_path) if args_path.is_file() else {}
        if not isinstance(args, dict):
            raise AutopilotError("panorama work item arguments are invalid")
        if panorama_tool == "prepare_panorama_vision_session":
            pack = prepare_panorama_vision_session(
                root,
                viewport_pack_json=str(args.get("viewport_pack_json") or ""),
                timeline_index=int(args.get("timeline_index") or 0),
                sample_index=int(args.get("sample_index") or 0),
                model_requested="luna",
                quality="low",
                write=True,
            )
        else:
            pack = prepare_panorama_temporal_session(
                root,
                analysis_plan_json=str(args.get("analysis_plan_json") or ""),
                task_id=str(args.get("task_id") or ""),
                model_requested="luna",
                quality="low",
                write=True,
            )
        if pack.get("status") != "ready_for_interactive_analysis":
            raise AutopilotError("Panorama Codex task pack is blocked: " + json.dumps(pack.get("blockers") or [], ensure_ascii=False))
        task_type = "codex_vision_session"
        instructions = (
            "逐一观察六个规范方向视口并填写全景结果模板；不得把局部视野冒充全局事实。"
            if panorama_tool == "prepare_panorama_vision_session"
            else "按顺序观察当前一个方向的全部连续帧；逐帧记录后再描述变化，不得推断未观察方向。"
        )
        for key in ("task_pack_path", "prompt_path", "result_template_path"):
            path = Path(str(pack[key])).resolve()
            refs.append(_input_ref(root, path, role=key))
        frames = [
            frame
            for item in pack.get("items") or []
            if isinstance(item, dict)
            for frame in item.get("frames") or []
            if isinstance(frame, dict)
        ]
        for position, frame in enumerate(frames, start=1):
            refs.append(_input_ref(root, Path(str(frame.get("path") or "")).resolve(), role=f"panorama_frame_{position:02d}"))
        output_schema = {"$ref": "video_knowledge_pipeline.codex_vision_result.v1"}
    elif str(action.get("kind") or "") == "codex_session" or str(action.get("mcp_tool") or "") in _MODEL_TOOLS:
        kind = "temporal" if str(action.get("mcp_tool") or "") == "run_temporal_visual_analysis" or "temporal" in str(action.get("key") or "") else "semantic"
        indexes = _codex_indexes(action)
        pack = prepare_codex_vision_session(root, kind=kind, indexes=indexes or None, model_requested="luna", quality="low", write=True)
        if pack.get("status") != "ready_for_interactive_analysis":
            raise AutopilotError("Codex vision task pack is blocked: " + json.dumps(pack.get("blockers") or [], ensure_ascii=False))
        task_type = "codex_vision_session"
        instructions = "逐项观察任务包引用的 Bundle 内帧，填写 result template；必须报告真实 model_actual，不得臆测未观察内容。"
        for key in ("task_pack_path", "prompt_path", "result_template_path"):
            path = Path(str(pack[key])).resolve()
            refs.append(_input_ref(root, path, role=key))
        output_schema = {"$ref": "video_knowledge_pipeline.codex_vision_result.v1"}
    else:
        args_path = str(action.get("mcp_args_path") or "").strip()
        if args_path and Path(args_path).is_file():
            refs.append(_input_ref(root, Path(args_path).resolve(), role="action_args"))
    refs.extend(_identity_refs(root))
    route = _refresh_route(job, root)
    provider_snapshot = job.get("policy_snapshot", {}).get("local_vision_provider") if task_type == "provider_vision_session" else {}
    requested_model = str((provider_snapshot.get("provider") or {}).get("model") or "luna") if isinstance(provider_snapshot, dict) else "luna"
    work_item = {
        "schema": WORK_ITEM_SCHEMA,
        "job_id": job["job_id"],
        "bundle_identity_sha256": _bundle_identity(root),
        "state_revision": job["state_revision"] + 1,
        "task_id": task_id,
        "lease_id": lease_id,
        "lease_expires_at_unix": time.time() + LEASE_SECONDS,
        "task_type": task_type,
        "instructions": instructions,
        "input_refs": refs,
        "output_schema": output_schema,
        "acceptance": ["job/revision/task/lease/hash must match", "one complete result only", "model_actual must be truthful"],
        "model_policy": {
            "requested_model": requested_model,
            "quality": "low",
            "requested_route": job["execution_route"]["requested_route"],
            "resolved_route": route["resolved_route"],
            "online_derived_artifacts_allowed": bool(job["policy_snapshot"]["online_derived_artifacts_allowed"]),
            "executor": "provider_gateway" if task_type == "provider_vision_session" else "agent_session",
            "execution_mode": (
                "background_local_loopback"
                if task_type == "provider_vision_session"
                else (
                    "interactive_text_evidence_review"
                    if task_type in _SEMANTIC_TASK_TYPES or task_type == 'summary_claim_review'
                    else "interactive_multimodal"
                )
            ),
            "provider_health_check": "not_applicable" if task_type in _SEMANTIC_TASK_TYPES or task_type == 'summary_claim_review' else "required_or_interactive",
            "remote_fallback_allowed": False if task_type in _SEMANTIC_TASK_TYPES or task_type in {'provider_vision_session', 'summary_claim_review'} else None,
        },
        "submit_command": f".\\scripts\\video-knowledge.ps1 autopilot-submit {job['job_id']} --input-json <result.json>",
        "action": _bounded_action(action),
        "action_fingerprint": fingerprint,
    }
    if semantic_task:
        work_item["semantic_task"] = semantic_task
        work_item["acceptance"] = [
            *work_item["acceptance"],
            "all selected candidate ids must be returned exactly once",
            "existing semantic validator and closure remain authoritative",
        ]
    payload = _canonical_bytes(work_item)
    if len(payload) > MAX_WORK_ITEM_BYTES:
        work_item["action"] = {key: action.get(key) for key in ("status", "kind", "key", "mcp_tool") if action.get(key) not in (None, "")}
        work_item["instructions"] = work_item["instructions"][:1000]
        payload = _canonical_bytes(work_item)
    if len(payload) > MAX_WORK_ITEM_BYTES:
        raise AutopilotError(f"work item exceeds {MAX_WORK_ITEM_BYTES} bytes")
    return work_item


def _semantic_candidate_discovery_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["schema", "source", "suggestions", "model_actual"],
        "properties": {
            "schema": {"const": "video_knowledge_pipeline.transcript_semantic_candidate_suggestions.v1"},
            "source": {"type": "string", "minLength": 1, "maxLength": 200},
            "model_actual": {"type": "string", "minLength": 1, "maxLength": 200},
            "suggestions": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "required": [
                        "source_segment_index",
                        "start",
                        "end",
                        "correction_type",
                        "original_text",
                        "candidate_text",
                        "reason",
                        "confidence",
                        "evidence_summary",
                        "needs_human_review",
                    ],
                    "properties": {
                        "source_segment_index": {"type": "integer", "minimum": 0},
                        "start": {"type": "number", "minimum": 0},
                        "end": {"type": "number", "minimum": 0},
                        "correction_type": {"type": "string", "minLength": 1, "maxLength": 80},
                        "original_text": {"type": "string", "minLength": 1, "maxLength": 500},
                        "candidate_text": {"type": "string", "maxLength": 500},
                        "reason": {"type": "string", "minLength": 1, "maxLength": 1000},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence_summary": {"type": "string", "maxLength": 1500},
                        "needs_human_review": {"type": "boolean"},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }


def _semantic_correction_review_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["schema", "source", "decisions", "candidate_selection", "model_actual"],
        "properties": {
            "schema": {"const": "video_knowledge_pipeline.transcript_semantic_correction_result.v1"},
            "source": {"type": "string", "minLength": 1, "maxLength": 200},
            "model_actual": {"type": "string", "minLength": 1, "maxLength": 200},
            "candidate_selection": {
                "type": "object",
                "required": ["selected_candidate_count", "selected_candidate_ids"],
                "properties": {
                    "selected_candidate_count": {"type": "integer", "minimum": 0, "maximum": 8},
                    "selected_candidate_ids": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {"type": "string", "minLength": 1, "maxLength": 120},
                    },
                },
                "additionalProperties": True,
            },
            "decisions": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "required": [
                        "candidate_id",
                        "action",
                        "correction_type",
                        "original_text",
                        "corrected_text",
                        "confidence",
                        "rationale",
                        "evidence_ids",
                        "human_confirmed",
                        "needs_human_review",
                    ],
                    "properties": {
                        "candidate_id": {"type": "string", "minLength": 1, "maxLength": 120},
                        "action": {
                            "enum": ["replace", "keep_original", "needs_human_review", "review", "reject"]
                        },
                        "correction_type": {"type": "string", "minLength": 1, "maxLength": 80},
                        "original_text": {"type": "string", "minLength": 1, "maxLength": 500},
                        "corrected_text": {"type": "string", "maxLength": 500},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "rationale": {"type": "string", "minLength": 1, "maxLength": 1500},
                        "evidence_ids": {
                            "type": "array",
                            "maxItems": 20,
                            "items": {"type": "string", "minLength": 1, "maxLength": 160},
                        },
                        "human_confirmed": {"type": "boolean"},
                        "needs_human_review": {"type": "boolean"},
                    },
                    "additionalProperties": False,
                },
            },
            "review_suggestions": {"type": "array", "maxItems": 8},
        },
        "additionalProperties": False,
    }


def _validate_submission(root: Path, job: dict[str, Any], work_item: dict[str, Any], payload: dict[str, Any], raw_bytes: bytes) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if len(raw_bytes) > MAX_SUBMISSION_BYTES:
        issues.append({"key": "submission_too_large", "detail": f"maximum is {MAX_SUBMISSION_BYTES} bytes"})
    if job.get("state") != "waiting_result" or not work_item:
        issues.append({"key": "no_waiting_task", "detail": "job has no current leased task"})
        return issues
    allowed_keys = {
        "schema",
        "job_id",
        "state_revision",
        "task_id",
        "lease_id",
        "bundle_identity_sha256",
        "result",
    }
    extra_keys = sorted(set(payload) - allowed_keys)
    if extra_keys:
        issues.append({"key": "extra_submission_fields", "detail": ",".join(extra_keys)})
    expected = {
        "schema": SUBMISSION_SCHEMA,
        "job_id": job.get("job_id"),
        "state_revision": job.get("state_revision"),
        "task_id": work_item.get("task_id"),
        "lease_id": work_item.get("lease_id"),
        "bundle_identity_sha256": work_item.get("bundle_identity_sha256"),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            issues.append({"key": f"{key}_mismatch", "detail": f"expected {value!r}"})
    if not _lease_valid(work_item):
        issues.append({"key": "lease_expired", "detail": "work item lease expired"})
    if str(work_item.get("bundle_identity_sha256") or "") != _bundle_identity(root):
        issues.append({"key": "bundle_identity_drift", "detail": "manifest or timeline changed while the result was pending"})
    for ref in work_item.get("input_refs") or []:
        if not isinstance(ref, dict):
            issues.append({"key": "input_ref_invalid", "detail": "input reference must be an object"})
            continue
        path = Path(str(ref.get("path") or "")).resolve()
        if not path.is_file() or sha256_file(path) != str(ref.get("sha256") or ""):
            issues.append({"key": "input_hash_drift", "detail": str(path)})
    result = payload.get("result")
    if not isinstance(result, dict):
        issues.append({"key": "result_missing", "detail": "submission result must be an object"})
    elif work_item.get("task_type") not in {"codex_vision_session", "summary_claim_review"}:
        for error in Draft202012Validator(work_item["output_schema"]).iter_errors(result):
            issues.append({"key": "result_schema_invalid", "detail": error.message[:500]})
    return issues


def _import_semantic_submission(
    root: Path,
    work_item: Mapping[str, Any],
    result: dict[str, Any],
    raw_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, str]], str]:
    """Import one semantic result through the existing validators and closure."""

    task_type = str(work_item.get("task_type") or "")
    task_id = str(work_item.get("task_id") or "semantic")
    if task_type == "transcript_semantic_candidate_discovery":
        input_path = raw_dir / f"semantic-candidate-discovery-{task_id}.json"
        write_json(input_path, result)
        preview = import_transcript_semantic_candidate_suggestions(
            root,
            input_json=input_path,
            write=False,
        )
        invalid_skips = [
            row
            for row in preview.get("skipped") or []
            if isinstance(row, dict) and str(row.get("reason") or "") != "duplicate_candidate"
        ]
        if invalid_skips:
            return (
                {"status": "candidate_import_preflight_rejected", "preview": preview},
                [
                    {
                        "key": "semantic_candidate_suggestion_invalid",
                        "detail": str(row.get("reason") or "invalid suggestion")[:500],
                    }
                    for row in invalid_skips[:8]
                ],
                "rejected",
            )
        imported = import_transcript_semantic_candidate_suggestions(
            root,
            input_json=input_path,
            write=True,
        )
        if str(imported.get("status") or "") not in {"imported", "no_candidates_imported"}:
            return (
                imported,
                [
                    {
                        "key": "semantic_candidate_import_rejected",
                        "detail": str(imported.get("status") or "rejected")[:500],
                    }
                ],
                "rejected",
            )
        return imported, [], "succeeded"

    if task_type != "transcript_semantic_correction_review":
        return {}, [{"key": "semantic_task_type_invalid", "detail": task_type}], "rejected"

    source_ref = next(
        (
            row
            for row in work_item.get("input_refs") or []
            if isinstance(row, dict) and row.get("role") == "semantic_correction_source_pack"
        ),
        {},
    )
    source_pack = read_json(Path(str(source_ref.get("path") or "")))
    if not isinstance(source_pack, dict):
        return {}, [{"key": "semantic_source_pack_invalid", "detail": "source pack is not an object"}], "rejected"
    semantic_task = work_item.get("semantic_task") if isinstance(work_item.get("semantic_task"), dict) else {}
    expected_ids = [str(value) for value in semantic_task.get("selected_candidate_ids") or [] if str(value)]
    decisions = [row for row in result.get("decisions") or [] if isinstance(row, dict)]
    decision_ids = [str(row.get("candidate_id") or "") for row in decisions]
    if sorted(decision_ids) != sorted(expected_ids) or len(set(decision_ids)) != len(decision_ids):
        return (
            {"status": "candidate_set_mismatch"},
            [
                {
                    "key": "semantic_candidate_set_mismatch",
                    "detail": f"expected={expected_ids!r}; actual={decision_ids!r}"[:500],
                }
            ],
            "rejected",
        )
    selection = result.get("candidate_selection") if isinstance(result.get("candidate_selection"), dict) else {}
    selected_ids = [str(value) for value in selection.get("selected_candidate_ids") or [] if str(value)]
    if sorted(selected_ids) != sorted(expected_ids) or int(selection.get("selected_candidate_count") or 0) != len(expected_ids):
        return (
            {"status": "candidate_selection_mismatch"},
            [{"key": "semantic_candidate_selection_mismatch", "detail": "candidate_selection must echo the leased subset"}],
            "rejected",
        )
    strong = validate_transcript_semantic_model_output(
        result,
        source_pack,
        min_confidence=float(semantic_task.get("min_confidence") or 0.88),
        allow_empty_decisions=not expected_ids,
    )
    explicit_review_gap = False
    if not strong.get("quality_gate_passed"):
        explicit_review_gap = _semantic_validation_only_explicit_review_gaps(result, strong)
        if not explicit_review_gap:
            validation_issues = [
                row
                for row in [
                    *(strong.get("contract_issues") or []),
                    *(strong.get("quality_issues") or []),
                ]
                if isinstance(row, dict)
            ]
            if not validation_issues:
                validation_issues = [
                    {
                        "key": "semantic_strong_validation_failed",
                        "detail": str(strong.get("status") or "rejected")[:500],
                    }
                ]
            return (
                {"status": "strong_validation_rejected", "validation": strong},
                [
                    {
                        "key": str(row.get("key") or "semantic_strong_validation_failed"),
                        "detail": str(row.get("detail") or "")[:500],
                    }
                    for row in validation_issues[:16]
                ],
                "rejected",
            )
    input_path = raw_dir / f"semantic-review-{task_id}.json"
    write_json(input_path, result)
    closure = transcript_semantic_correction_closure(
        root,
        input_json=input_path,
        min_confidence=float(semantic_task.get("min_confidence") or 0.88),
        auto_apply=True,
        refresh_exports=False,
        write=True,
    )
    closure["strong_validation"] = strong
    closure_status = str(closure.get("status") or "")
    if closure_status in {"completed", "completed_no_text_changes"}:
        return closure, [], "review_gap" if explicit_review_gap else "succeeded"
    if closure_status in {"no_safe_decisions", "no_matching_segments"}:
        return closure, [], "review_gap"
    return (
        closure,
        [{"key": "semantic_closure_rejected", "detail": closure_status or "unknown closure status"}],
        "rejected",
    )


def _semantic_validation_only_explicit_review_gaps(
    result: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> bool:
    if not validation.get("contract_ok"):
        return False
    decisions = [row for row in result.get("decisions") or [] if isinstance(row, dict)]
    if not decisions:
        return False
    bad_indexes: set[int] = set()
    for issue in validation.get("quality_issues") or []:
        if not isinstance(issue, dict):
            return False
        detail = str(issue.get("detail") or "")
        if not detail.startswith("decisions[") or "]" not in detail:
            return False
        try:
            bad_indexes.add(int(detail.split("[", 1)[1].split("]", 1)[0]))
        except ValueError:
            return False
    if not bad_indexes:
        return False
    review_actions = {"needs_human_review", "review", "reject"}
    return all(
        0 <= index < len(decisions)
        and str(decisions[index].get("action") or "") in review_actions
        and bool(decisions[index].get("needs_human_review"))
        for index in bad_indexes
    )


def _submission_payload(
    *, input_json: str | Path | None, submission: Mapping[str, Any] | None
) -> tuple[dict[str, Any], bytes, str, list[dict[str, str]]]:
    if (input_json is None) == (submission is None):
        raise ValueError("provide exactly one of input_json or submission")
    if input_json is not None:
        path = Path(input_json).expanduser().resolve()
        raw = path.read_bytes()
        input_path = str(path)
        try:
            value = json.loads(raw.decode("utf-8"))
            parse_issues: list[dict[str, str]] = []
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            value = {}
            parse_issues = [{"key": "input_json_invalid", "detail": str(exc)[:500]}]
    else:
        value = dict(submission or {})
        raw = _canonical_bytes(value)
        input_path = "<mapping>"
        parse_issues = []
    if not isinstance(value, dict):
        parse_issues.append({"key": "submission_not_object", "detail": "submission must be a JSON object"})
        value = {}
    return value, raw, input_path, parse_issues


def _run_deterministic_action(root: Path, action: dict[str, Any]) -> bool:
    if action.get('status') == 'summary_claim_review_ready':
        # Explicit deterministic reuse only; never regenerate a model answer.
        return bool(bundle_advance(root, execute=True, refresh_outputs=False).get('advanced'))
    if str(action.get("status") or "") not in _MACHINE_STATUSES:
        return False
    from .workflow_completion import semantic_progress_signature
    before = semantic_progress_signature(root)
    if str(action.get("kind") or "") == "panorama":
        result = _run_panorama_action(root, action)
    elif str(action.get("mcp_tool") or "") == "run_temporal_tag_delta" or str(action.get("key") or "") == "temporal_tag_delta":
        result = run_temporal_tag_delta(
            root,
            execute_tagger=True,
            frame_count=8,
            limit=16,
            min_frames=5,
            write=True,
        )
    else:
        result = bundle_advance(root, execute=False, refresh_outputs=False)
    after = semantic_progress_signature(root)
    return bool(result.get("advanced") or result.get("status") in {"completed", "partial", "needs_frame_tags", "planned", "ready_for_analysis", "succeeded"}) and before != after


def _run_panorama_action(root: Path, action: dict[str, Any]) -> dict[str, Any]:
    key = str(action.get("key") or "")
    manifest = read_json(root / "manifest.json")
    if not isinstance(manifest, dict):
        raise AutopilotError("manifest.json changed before panorama deterministic action")
    profile = manifest.get("panorama_profile") if isinstance(manifest.get("panorama_profile"), dict) else {}
    if key == "panorama_prepare":
        digest = str(profile.get("profile_sha256") or "")
        return panorama_prepare(
            root,
            projection=str(profile.get("projection") or ""),
            input_fov=profile.get("input_fov"),
            lens_order=str(profile.get("lens_order") or "") or None,
            run_id=f"autopilot-{digest[:12] or 'panorama'}",
            write=True,
        )
    run_id = str((manifest.get("panorama_projection_plan") or {}).get("run_id") or "")
    if not run_id:
        raise AutopilotError("panorama projection run_id is missing")
    if key == "panorama_projection_execute":
        return panorama_run(root, run_id=run_id, execute_projection=True)
    if key == "panorama_analysis_triage":
        return panorama_run(root, run_id=run_id, execute_projection=False)
    raise AutopilotError(f"unsupported panorama deterministic action: {key}")


def _needs_model_task(action: dict[str, Any]) -> bool:
    return (
        _is_semantic_model_action(action)
        or _is_summary_review_action(action)
        or str(action.get("kind") or "") == "codex_session"
        or str(action.get("mcp_tool") or "") in _MODEL_TOOLS
    )


def _is_semantic_model_action(action: Mapping[str, Any]) -> bool:
    return str(action.get("key") or "") in _SEMANTIC_MODEL_ACTIONS


def _is_summary_review_action(action: Mapping[str, Any]) -> bool:
    return (action.get('status') == 'summary_claim_review_pending'
            and action.get('mcp_tool') == 'prepare_summary_claim_review')


def _requires_human(action: dict[str, Any]) -> bool:
    return bool(action.get("human_required")) or str(action.get("kind") or "") in {"human_review", "provider_repair"}


def _is_completion_action(action: dict[str, Any]) -> bool:
    return str(action.get("status") or "") == "ready" or str(action.get("kind") or "") == "export"


def _move(job: dict[str, Any], trigger: str) -> None:
    if job["state"] in TERMINAL_STATES:
        raise AutopilotError(f"terminal state cannot transition: {job['state']}")
    machine = Machine(states=list(STATES), initial=job["state"], auto_transitions=False)
    for name, source, dest in TRANSITIONS:
        machine.add_transition(name, source, dest)
    try:
        getattr(machine, trigger)()
    except Exception as exc:
        raise AutopilotError(f"illegal state transition {job['state']} --{trigger}--> ?") from exc
    job["state"] = str(machine.state)
    job["state_revision"] += 1
    job["transition_count"] += 1
    job["updated_at"] = now_iso()
    job["heartbeat_at"] = now_iso()
    _assert_transition_budget(job)


def _assert_transition_budget(job: dict[str, Any]) -> None:
    if int(job.get("transition_count") or 0) > MAX_TRANSITIONS:
        raise AutopilotError("state transition budget exhausted")


def _hard_block(job: dict[str, Any], key: str, detail: str) -> None:
    job.setdefault("hard_blockers", []).append({"key": key, "detail": detail, "created_at": now_iso()})
    if job["state"] != "blocked":
        _move(job, "block")


def _event(job: dict[str, Any], event: str, **details: Any) -> None:
    append_jsonl(
        Path(job["events_path"]),
        [{"schema": "video_knowledge_pipeline.autopilot_event.v1", "created_at": now_iso(), "job_id": job["job_id"], "state": job["state"], "state_revision": job["state_revision"], "event": event, **details}],
    )


def _record_gap(job: dict[str, Any], key: str, detail: str, *, action: Mapping[str, Any]) -> None:
    row = {"key": key, "detail": detail[:4000], "action": _bounded_action(action), "created_at": now_iso()}
    if row not in job["review_gaps"]:
        job["review_gaps"].append(row)


def _provider_vision_kind(action: Mapping[str, Any]) -> str:
    text = " ".join(str(action.get(key) or "") for key in ("mcp_tool", "key", "kind")).lower()
    return "temporal" if "temporal" in text else "semantic"


def _provider_vision_excluded_indexes(job: Mapping[str, Any], action: Mapping[str, Any]) -> list[int]:
    kind = _provider_vision_kind(action)
    outcomes = job.get("provider_vision_target_outcomes")
    if not isinstance(outcomes, dict):
        return []
    indexes: list[int] = []
    for key, row in outcomes.items():
        if not str(key).startswith(f"{kind}:") or not isinstance(row, dict):
            continue
        if str(row.get("status") or "") not in {"unavailable", "review_gap"}:
            continue
        try:
            indexes.append(int(row.get("index") or str(key).split(":", 1)[1]))
        except (TypeError, ValueError):
            continue
    return sorted({index for index in indexes if index > 0})


def _record_provider_vision_target_outcomes(
    job: dict[str, Any],
    work_item: Mapping[str, Any],
    *,
    status: str,
    reason: str,
    detail: str,
    provider_call_count: int,
) -> None:
    pack_path = next(
        (
            Path(str(row.get("path") or "")).expanduser().resolve()
            for row in work_item.get("input_refs") or []
            if isinstance(row, dict) and row.get("role") == "provider_task_pack"
        ),
        None,
    )
    if pack_path is None or not pack_path.is_file():
        return
    pack = read_json(pack_path)
    if not isinstance(pack, dict):
        return
    kind = str(pack.get("kind") or "semantic")
    outcomes = job.setdefault("provider_vision_target_outcomes", {})
    for item in pack.get("items") or []:
        if not isinstance(item, dict):
            continue
        index = int(item.get("index") or 0)
        if index <= 0:
            continue
        frames = [frame for frame in item.get("frames") or [] if isinstance(frame, dict)]
        outcomes[f"{kind}:{index}"] = {
            "kind": kind,
            "index": index,
            "status": str(status or "review_gap"),
            "reason": str(reason or "")[:200],
            "detail": str(detail or "")[:1000],
            "provider_call_count": max(0, int(provider_call_count or 0)),
            "frame_ids": [str(frame.get("frame_id") or "") for frame in frames],
            "frame_sha256s": [str(frame.get("sha256") or "") for frame in frames],
            "updated_at": now_iso(),
        }


def _record_action_attempt(job: dict[str, Any], fingerprint: str, changed: bool) -> None:
    row = job["action_history"].setdefault(fingerprint, {"attempts": 0, "changed": 0})
    row["attempts"] += 1
    row["changed"] += int(bool(changed))


def _record_submission_rejection(job: dict[str, Any], work_item: Mapping[str, Any]) -> None:
    task_id = str(work_item.get("task_id") or "")
    if not task_id:
        return
    counts = job.setdefault("submission_rejections", {})
    counts[task_id] = int(counts.get(task_id) or 0) + 1
    route = str((work_item.get("model_policy") or {}).get("resolved_route") or "")
    limit = 2 if route.startswith("online") else 6
    if counts[task_id] < limit:
        return
    _record_gap(
        job,
        "model_submission_retry_exhausted",
        f"任务连续 {limit} 次未通过原子校验，已停止原路径并转人工复核缺口。",
        action=work_item.get("action") if isinstance(work_item.get("action"), dict) else {},
    )
    _mark_fingerprint_resolved(job, str(work_item.get("action_fingerprint") or ""), "review_gap")
    job["current_work_item"] = None
    if job.get("state") == "waiting_result":
        _move(job, "resume")


def _fingerprint_attempts(job: dict[str, Any], fingerprint: str) -> int:
    return int(job.get("action_history", {}).get(fingerprint, {}).get("attempts") or 0)


def _mark_fingerprint_resolved(job: dict[str, Any], fingerprint: str, status: str) -> None:
    if fingerprint:
        job["resolved_action_fingerprints"][fingerprint] = {"status": status, "resolved_at": now_iso()}


def _fingerprint_is_resolved(job: dict[str, Any], fingerprint: str) -> bool:
    return bool(fingerprint and fingerprint in job.get("resolved_action_fingerprints", {}))


def _action_fingerprint(root: Path, action: Mapping[str, Any]) -> str:
    refs = []
    args_path = str(action.get("mcp_args_path") or "")
    if args_path and Path(args_path).is_file():
        refs.append({"path": str(Path(args_path).resolve()), "sha256": sha256_file(Path(args_path))})
    from .workflow_completion import semantic_progress_signature
    identity = {
        "bundle_identity": semantic_progress_signature(root),
        "action": {key: action.get(key) for key in ("status", "kind", "key", "mcp_tool")},
        "input_refs": refs,
    }
    if _is_summary_review_action(action):
        # Manifest/timeline do not change between annotation and review. The
        # existing review pack identity distinguishes phases and source shards.
        task = action.get('next_task') or {}
        identity['summary_review_task'] = {key: task.get(key) for key in ('task_id', 'task_pack_sha256')}
    return canonical_json_sha256(identity)


def _bundle_identity(root: Path) -> str:
    manifest = root / "manifest.json"
    timeline = root / "timeline.json"
    if not manifest.is_file() or not timeline.is_file():
        raise FileNotFoundError("Bundle manifest or timeline is missing")
    return canonical_json_sha256({"bundle_dir": str(root), "manifest_sha256": sha256_file(manifest), "timeline_sha256": sha256_file(timeline)})


def _identity_refs(root: Path) -> list[dict[str, Any]]:
    return [_input_ref(root, root / "manifest.json", role="manifest"), _input_ref(root, root / "timeline.json", role="timeline")]


def _input_ref(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise AutopilotError(f"input reference must be an existing Bundle file: {resolved}")
    return {"role": role, "path": str(resolved), "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def _codex_indexes(action: Mapping[str, Any]) -> list[int]:
    path = Path(str(action.get("mcp_args_path") or ""))
    if not path.is_file():
        return []
    value = read_json(path)
    if not isinstance(value, dict):
        return []
    indexes: list[int] = []
    for item in value.get("indexes") or []:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if index > 0:
            indexes.append(index)
    return indexes


def _refresh_route(job: dict[str, Any], root: Path) -> dict[str, Any]:
    samples = list(job.get("resource_samples") or [])
    latest_unix = float(samples[-1].get("sampled_at_unix") or 0) if samples and isinstance(samples[-1], dict) else 0.0
    if time.time() - latest_unix >= 30:
        samples.append(resource_snapshot(heavy_local_lease=root / ".autopilot-heavy-local.lock").to_dict())
    job["resource_samples"] = samples[-20:]
    route = resolve_execution_route(
        str(job["execution_route"]["requested_route"]),
        duration_seconds=_bundle_duration_seconds(root),
        resource_samples=job["resource_samples"],
    )
    previous = job.get("execution_route") if isinstance(job.get("execution_route"), dict) else {}
    recent = job["resource_samples"][-2:]
    two_healthy = len(recent) == 2 and all(not bool(row.get("busy")) for row in recent if isinstance(row, dict))
    if previous.get("basis") == "resource_busy" and previous.get("resolved_route") == "online-heavy" and not two_healthy:
        route = {**previous, "hysteresis_waiting_for_two_healthy_samples": True}
    job["execution_route"] = route
    return route


def _model_budget_blocker(job: Mapping[str, Any]) -> str:
    usage = job.get("budget_usage") if isinstance(job.get("budget_usage"), dict) else {}
    route = job.get("execution_route") if isinstance(job.get("execution_route"), dict) else {}
    resolved = str(route.get("resolved_route") or "hybrid")
    if resolved in {"online-only", "online-heavy"} and int(usage.get("online_calls") or 0) >= 64:
        return "online call budget reached 64 calls for this video"
    if resolved not in {"online-only", "online-heavy"} and int(usage.get("local_calls") or 0) >= 512:
        return "local call budget reached 512 calls for this video"
    if float(usage.get("estimated_online_usd") or 0.0) >= 3.0:
        return "estimated online cost budget reached USD 3"
    return ""


def _bundle_duration_seconds(root: Path) -> float:
    try:
        value = read_json(root / "timeline.json")
    except Exception:
        return 0.0
    rows = value if isinstance(value, list) else []
    return max((float(row.get("end") or 0) for row in rows if isinstance(row, dict)), default=0.0)


def _gpu_snapshot() -> tuple[float | None, float | None, list[str]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None, None, ["nvidia_smi_not_found"]
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=utilization.gpu,memory.free,memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        rows = [line.strip().split(",") for line in completed.stdout.splitlines() if line.strip()]
        parsed = [(float(row[0]), float(row[1]), float(row[2])) for row in rows if len(row) >= 3]
        if not parsed:
            return None, None, ["nvidia_smi_empty"]
        util = max(row[0] for row in parsed)
        free_ratio = min((row[1] / row[2]) for row in parsed if row[2] > 0)
        return util, free_ratio, []
    except Exception as exc:  # pragma: no cover - hardware-specific
        return None, None, [f"nvidia_smi_failed:{type(exc).__name__}"]


def _write_work_item(root: Path, job: dict[str, Any], work_item: dict[str, Any]) -> None:
    path = root / "runs" / job["run_id"] / "work-items" / f"{work_item['task_id']}.json"
    write_json(path, work_item)


def _validation_receipt(job: dict[str, Any], payload: dict[str, Any], raw_path: Path, input_path: str, *, status: str, issues: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "changed": status == "succeeded",
        "job_id": job["job_id"],
        "state": job["state"],
        "state_revision": job["state_revision"],
        "task_id": payload.get("task_id", ""),
        "issues": issues,
        "artifacts": {"raw_result": str(raw_path)},
        "input_path": input_path,
        "input_hashes": {"raw_result_sha256": sha256_file(raw_path)},
        "next_action": (
            "autopilot_next"
            if status == "succeeded"
            else "autopilot_next --retry-provider"
            if status == "waiting_provider"
            else "repair and resubmit the same leased task"
        ),
        "created_at": now_iso(),
    }


def _initial_restore_plan(root: Path, job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "video_knowledge_pipeline.autopilot_restore_plan.v1",
        "job_id": job["job_id"],
        "bundle_dir": str(root),
        "strategy": "Reuse per-action existing restore plans; remove only Autopilot coordinator artifacts after explicit human approval.",
        "automatic_apply": False,
        "accepted_state_untouched": True,
        "created_paths": [str(Path(job["job_path"]).parent)],
    }


def _sync_run_registry(root: Path, job: dict[str, Any]) -> None:
    run_path = root / "runs" / job["run_id"] / "run.json"
    if not run_path.is_file():
        return
    run = read_json(run_path)
    if not isinstance(run, dict):
        return
    state = str(job["state"])
    run["status"] = ("completed" if state == "complete_draft" and (job.get("completion") or {}).get("completed") is True
                     else "needs_input" if state == "complete_draft" or job.get("summary_handoff")
                     else "failed" if state == "blocked" else "cancelled" if state == "cancelled" else "running")
    run["summary"] = f"Autopilot state={state}; review_gaps={len(job['review_gaps'])}; accepted is unchanged."
    run["updated_at"] = now_iso()
    run["parameters"] = {**dict(run.get("parameters") or {}), "state": state, "state_revision": job["state_revision"], "budget_usage": job["budget_usage"]}
    run["failed_items"] = list(job.get("hard_blockers") or [])
    write_json(run_path, run)
    # Rebuilding the registry also refreshes manifest pointers.  While a model
    # task is leased, manifest/timeline bytes are part of its immutable input
    # identity, so defer that benign mutation until the lease is resolved.
    if not isinstance(job.get("current_work_item"), dict):
        build_run_artifact_registry(root, write=True)


def _save_job(path: Path, job: dict[str, Any]) -> None:
    job["updated_at"] = now_iso()
    job["heartbeat_at"] = now_iso()
    write_json(path, job)


def _read_job(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema") != JOB_SCHEMA:
        raise AutopilotError(f"invalid Autopilot job: {path}")
    return value


def _job_root(value: str | Path | None) -> Path:
    raw = value or os.environ.get("VKP_AUTOPILOT_JOB_ROOT") or (Path.cwd() / ".local" / "autopilot-jobs")
    path = Path(raw).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_pointer(job_id: str, *, job_root: str | Path | None) -> dict[str, Any]:
    safe = str(job_id or "").strip()
    if not safe or not safe.isalnum():
        raise AutopilotError("job_id must be alphanumeric")
    path = _job_root(job_root) / f"{safe}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Autopilot job pointer not found: {path}")
    value = read_json(path)
    if not isinstance(value, dict) or value.get("job_id") != safe:
        raise AutopilotError(f"invalid Autopilot pointer: {path}")
    return value


def _pointer_paths(pointer: Mapping[str, Any]) -> tuple[Path, Path]:
    root = Path(str(pointer.get("bundle_dir") or "")).resolve()
    job_path = Path(str(pointer.get("job_path") or "")).resolve()
    expected = root / "runs"
    if not root.is_dir() or not job_path.is_relative_to(expected):
        raise AutopilotError("job pointer escapes its Bundle runs directory")
    return root, job_path


def _public_status(job: Mapping[str, Any]) -> dict[str, Any]:
    work_item = job.get("current_work_item") if isinstance(job.get("current_work_item"), dict) else None
    provider_wait = work_item.get("provider_wait") if isinstance(work_item, dict) and isinstance(work_item.get("provider_wait"), dict) else {}
    completion = job.get("completion") or {}
    handoff = job.get("summary_handoff") or {}
    if job["state"] == "complete_draft" or handoff:
        from .workflow_completion import workflow_completion_status
        from .smart_summary_handoff import handoff_status
        root = Path(str(job["bundle_dir"]))
        completion = workflow_completion_status(root)
        current = handoff_status(root)
        if current.get("status") != "missing":
            handoff = current
    status = "waiting_provider" if provider_wait else "waiting_model" if work_item else str(job["state"])
    if handoff.get("status") in {"awaiting-agent", "review-required", "blocked"}:
        status = handoff["status"]
    elif job["state"] == "complete_draft" and completion.get("completed") is not True:
        status = completion.get("status") or "review-required"
    return {
        "schema": JOB_SCHEMA,
        "job_id": job["job_id"],
        "bundle_dir": job["bundle_dir"],
        "state": job["state"],
        "status": status,
        "completion": completion,
        "summary_handoff": handoff,
        "no_progress": job.get("no_progress") or {},
        "state_revision": job["state_revision"],
        "runner": job["runner"],
        "execution_route": job["execution_route"],
        "budget_usage": job["budget_usage"],
        "review_gaps": list(job.get("review_gaps") or []),
        "provider_vision_target_outcomes": dict(job.get("provider_vision_target_outcomes") or {}),
        "hard_blockers": list(job.get("hard_blockers") or []),
        "current_work_item": work_item,
        "provider_wait": provider_wait,
        "local_vision_provider": dict(job.get("policy_snapshot", {}).get("local_vision_provider") or {}),
        "accepted_written": False,
        "paths": {"job": job["job_path"], "events": job["events_path"], "restore_plan": job["restore_plan_path"]},
        "updated_at": job["updated_at"],
    }


def _next_response(job: Mapping[str, Any], *, work_item: dict[str, Any] | None = None) -> dict[str, Any]:
    return {**_public_status(job), "work_item": work_item, "deterministic_step_limit": MAX_DETERMINISTIC_STEPS}


def _await_summary_agent(root: Path, job: dict[str, Any]) -> bool:
    from .smart_summary_handoff import summary_handoff_action
    action = summary_handoff_action(root)
    if not action:
        job.pop("summary_handoff", None)
        return False
    from .smart_summary_codex import generate_smart_summary_with_codex
    result = generate_smart_summary_with_codex(root, agent_available=job.get("runner") == "agent-session")
    handoff = result.get("handoff") or result.get("candidate") or {
        "status": result.get("workflow_status", "blocked"), "next_actions": result.get("next_actions", [])}
    job["summary_handoff"] = handoff
    from .workflow_completion import workflow_completion_status
    job["completion"] = workflow_completion_status(root)
    return True


def _bounded_action(action: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("status", "kind", "key", "label", "reason", "mcp_tool", "mcp_args_path", "human_required"):
        value = action.get(key)
        if isinstance(value, str):
            result[key] = value[:2000]
        elif value is not None:
            result[key] = value
    return result


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _lease_valid(work_item: Mapping[str, Any]) -> bool:
    return float(work_item.get("lease_expires_at_unix") or 0) > time.time()


def _choice(value: str, allowed: set[str], name: str) -> str:
    text = str(value or "").strip().lower()
    if text not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(allowed))}")
    return text


def _local_background_ready(job: Mapping[str, Any]) -> bool:
    route = job.get("execution_route") if isinstance(job.get("execution_route"), dict) else {}
    if str(route.get("resolved_route") or "") == "online-only":
        return False
    policy = job.get("policy_snapshot") if isinstance(job.get("policy_snapshot"), dict) else {}
    snapshot = policy.get("local_vision_provider") if isinstance(policy.get("local_vision_provider"), dict) else {}
    return bool(snapshot.get("ready_to_attempt"))


def _default_background_executor(job: Mapping[str, Any]) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
    if job.get("runner") != "background" or not _local_background_ready(job):
        return None
    return execute_provider_vision_work_item
