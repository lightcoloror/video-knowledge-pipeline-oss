from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from .artifact_validation import sha256_file
from .canonical_json import canonical_json_sha256
from .config import resolve_vision_execution_profile, vision_execution_profile
from .model_api_settings import resolve_model_api_provider_config
from .model_task_gateway import model_task_api_call
from .multimodal_frame_analyzer import (
    _candidates as semantic_candidates,
    _prompt as semantic_prompt,
    _select_candidates as select_semantic_candidates,
    run_multimodal_frame_analysis,
)
from .storage import read_json, write_json
from .temporal_frame_preprocess import build_temporal_frame_manifest
from .temporal_visual_analyzer import (
    _candidates as temporal_candidates,
    _prompt as temporal_prompt,
    _select_candidates as select_temporal_candidates,
    run_temporal_visual_analysis,
)
from .vision_api import parse_model_json, resolve_provider_config


PACK_SCHEMA = "video_knowledge_pipeline.autopilot_provider_vision_pack.v1"
RESULT_SCHEMA = "video_knowledge_pipeline.autopilot_provider_vision_result.v1"
LOCAL_PROVIDER_SNAPSHOT_SCHEMA = "video_knowledge_pipeline.autopilot_local_provider_snapshot.v1"
SUBMISSION_SCHEMA = "video_knowledge_pipeline.autopilot_submission.v1"
AUTOPILOT_PROVIDER_TIMEOUT_SECONDS = 60
AUTOPILOT_VISION_MAX_TOKENS = 4096
AUTOPILOT_CONTENT_REPAIR_RETRIES = 1

_LOCAL_PROVIDERS = {"local_qwen_vl", "local_vlm"}
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


class ProviderVisionTargetsExhausted(ValueError):
    """Raised when every currently missing provider target already has a gap receipt."""


def local_vision_provider_snapshot() -> dict[str, Any]:
    """Resolve one secretless, immutable local vision route for Autopilot.

    An explicitly configured local model route wins.  The project vision profile
    remains the compatibility fallback because existing VKP installations already
    use it for LM Studio.  Neither branch probes or mutates the provider service.
    """

    route_config: dict[str, Any] = {}
    source = ""
    issue = ""
    try:
        route_config = resolve_model_api_provider_config(
            "semantic_frame",
            execution_location="local",
        )
        if route_config:
            source = "model_api_local_route"
    except (OSError, ValueError) as exc:
        issue = f"local_route_unavailable:{type(exc).__name__}"

    if not route_config:
        configured = vision_execution_profile()
        route_config = {
            key: configured.get(key)
            for key in ("provider", "model", "base_url")
            if configured.get(key) not in (None, "")
        }
        route_config.update(
            {
                "adapter_backend": "builtin",
                "execution_location": "local",
                "api_key_optional": True,
            }
        )
        source = "project_vision_execution"

    resolved = resolve_provider_config(route_config)
    public = {
        key: resolved.get(key)
        for key in (
            "provider",
            "base_url",
            "model",
            "timeout_seconds",
            "adapter_backend",
            "api_key_optional",
            "execution_location",
            "route_id",
            "route_revision",
            "virtual_model",
            "profile_id",
            "capabilities",
        )
        if resolved.get(key) not in (None, "", [], {})
    }
    try:
        configured_timeout = max(1, int(public.get("timeout_seconds") or AUTOPILOT_PROVIDER_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        configured_timeout = AUTOPILOT_PROVIDER_TIMEOUT_SECONDS
    public["timeout_seconds"] = min(configured_timeout, AUTOPILOT_PROVIDER_TIMEOUT_SECONDS)
    provider = str(public.get("provider") or "").strip().lower()
    base_url = str(public.get("base_url") or "").strip()
    model = str(public.get("model") or "").strip()
    loopback = _is_loopback_url(base_url)
    ready_to_attempt = provider in _LOCAL_PROVIDERS and loopback and bool(model)
    blockers: list[dict[str, str]] = []
    if provider not in _LOCAL_PROVIDERS:
        blockers.append({"key": "local_vision_provider_required", "detail": provider or "missing"})
    if not loopback:
        blockers.append({"key": "loopback_base_url_required", "detail": base_url or "missing"})
    if not model:
        blockers.append({"key": "local_vision_model_required", "detail": "model is empty"})
    snapshot = {
        "schema": LOCAL_PROVIDER_SNAPSHOT_SCHEMA,
        "status": "ready_to_attempt" if ready_to_attempt else "blocked",
        "ready_to_attempt": ready_to_attempt,
        "provider_config_source": source,
        "provider": public,
        "blockers": blockers,
        "issues": [issue] if issue else [],
        "service_started": False,
        "service_probed": False,
        "provider_request_made": False,
        "remote_fallback_allowed": False,
    }
    snapshot["snapshot_sha256"] = canonical_json_sha256(snapshot)
    return snapshot


def prepare_provider_vision_pack(
    bundle_dir: str | Path,
    *,
    run_dir: str | Path,
    task_id: str,
    action: Mapping[str, Any],
    provider_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a hash-bound provider pack from Bundle-owned extracted frames."""

    root = Path(bundle_dir).expanduser().resolve()
    timeline_path = root / "timeline.json"
    manifest_path = root / "manifest.json"
    timeline_data = read_json(timeline_path)
    timeline = [dict(row) for row in timeline_data if isinstance(row, dict)] if isinstance(timeline_data, list) else []
    indexes, limit, frame_count = _action_selection(action)
    excluded_indexes = _action_excluded_indexes(action)
    allow_stale_index_fallback = bool(action.get("allow_stale_index_fallback"))
    stale_index_fallback_used = False
    kind = _action_kind(action)
    if kind == "semantic":
        available = [
            row
            for row in semantic_candidates(root, timeline, explicit_indexes=indexes)
            if int(row.get("index") or 0) not in excluded_indexes
        ]
        candidates = select_semantic_candidates(
            available,
            limit=limit,
            indexes=indexes or None,
        )
        if not candidates and indexes and allow_stale_index_fallback:
            candidates = select_semantic_candidates(available, limit=limit, indexes=None)
            stale_index_fallback_used = bool(candidates)
        # One semantic frame per background work item is the durable checkpoint
        # boundary.  Local VLM latency is variable, so a 19-call synchronous
        # lease can outlive an agent/terminal command and strand the job before
        # any validated result is imported.
        candidates = candidates[:1]
    else:
        available = [
            row
            for row in temporal_candidates(root, timeline, frame_count=frame_count)
            if int(row.get("index") or 0) not in excluded_indexes
        ]
        candidates = select_temporal_candidates(
            available,
            limit=1,
            indexes=indexes or None,
        )[:1]
        if not candidates and indexes and allow_stale_index_fallback:
            candidates = select_temporal_candidates(available, limit=1, indexes=None)[:1]
            stale_index_fallback_used = bool(candidates)
    if not candidates:
        if excluded_indexes:
            raise ProviderVisionTargetsExhausted(
                f"all available {kind} vision candidates already have explicit outcomes; "
                f"excluded_indexes={sorted(excluded_indexes)}"
            )
        raise ValueError(f"no Bundle-owned {kind} vision candidates are available")

    items: list[dict[str, Any]] = []
    for candidate in candidates:
        frame_paths = [Path(str(value)).expanduser().resolve() for value in candidate.get("frame_paths") or []]
        if kind == "semantic":
            frame_paths = frame_paths[:1]
        if not frame_paths:
            raise ValueError(f"vision candidate {candidate.get('index')} has no extracted frame")
        frames = [_frame_identity(root, path) for path in frame_paths]
        frame_manifest = build_temporal_frame_manifest([str(path) for path in frame_paths]) if kind == "temporal" else []
        if kind == "temporal":
            if not 5 <= len(frames) <= 12:
                raise ValueError(f"temporal provider candidate {candidate.get('index')} requires 5-12 extracted frames")
            manifest_by_name = {str(row.get("filename") or ""): row for row in frame_manifest}
            for frame in frames:
                row = manifest_by_name.get(Path(str(frame["path"])).name, {})
                frame["frame_id"] = str(row.get("frame_id") or frame["frame_id"])
        prompt = temporal_prompt({**candidate, "frame_manifest": frame_manifest}) if kind == "temporal" else semantic_prompt(candidate)
        items.append(
            {
                "index": int(candidate.get("index") or 0),
                "kind": kind,
                "start": float(candidate.get("start") or 0),
                "end": float(candidate.get("end") or 0),
                "prompt": prompt,
                "frames": frames,
                "frame_manifest": frame_manifest,
            }
        )

    pack = {
        "schema": PACK_SCHEMA,
        "task_id": str(task_id),
        "kind": kind,
        "bundle_dir": str(root),
        "manifest_sha256": sha256_file(manifest_path),
        "timeline_sha256": sha256_file(timeline_path),
        "provider_snapshot_sha256": str(provider_snapshot.get("snapshot_sha256") or ""),
        "provider": dict(provider_snapshot.get("provider") or {}),
        "request_contract": {
            "max_images_per_request": 1,
            "automatic_retry_count": 0,
            "content_repair_retry_count": AUTOPILOT_CONTENT_REPAIR_RETRIES,
            "max_tokens_per_call": AUTOPILOT_VISION_MAX_TOKENS,
            "remote_fallback_allowed": False,
            "source_video_read_allowed": False,
        },
        "selection_contract": {
            "requested_indexes": indexes,
            "selected_indexes": [int(item["index"]) for item in items],
            "stale_index_fallback_allowed": allow_stale_index_fallback,
            "stale_index_fallback_used": stale_index_fallback_used,
        },
        "items": items,
    }
    pack["task_pack_sha256"] = _pack_sha256(pack)
    path = Path(run_dir).expanduser().resolve() / "provider-tasks" / f"{task_id}.json"
    write_json(path, pack)
    return {**pack, "task_pack_path": str(path)}


def execute_provider_vision_work_item(
    work_item: Mapping[str, Any],
    *,
    call_model: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute a local provider pack sequentially, one extracted frame per call."""

    try:
        pack_path = _work_item_pack_path(work_item)
    except ValueError as exc:
        return _submission(work_item, _retryable("provider_task_pack_missing", str(exc)))
    pack = read_json(pack_path)
    if not isinstance(pack, dict) or pack.get("schema") != PACK_SCHEMA:
        return _submission(work_item, _retryable("provider_task_pack_invalid", str(pack_path)))
    issues = _validate_pack(pack_path, pack, work_item)
    if issues:
        return _submission(work_item, _retryable("provider_task_pack_drift", "; ".join(issues)))

    provider = resolve_provider_config(dict(pack.get("provider") or {}))
    if not _is_loopback_url(str(provider.get("base_url") or "")):
        return _submission(work_item, _retryable("non_loopback_provider_blocked", str(provider.get("base_url") or "")))
    expected_model = str(provider.get("model") or "").strip()
    if not expected_model:
        return _submission(work_item, _retryable("provider_model_missing", "Configure the LM Studio model id."))

    caller = call_model or model_task_api_call
    kind = str(pack.get("kind") or "")
    imported_items: list[dict[str, Any]] = []
    call_receipts: list[dict[str, Any]] = []
    actual_models: list[str] = []
    total_calls = 0
    for item in pack.get("items") or []:
        if not isinstance(item, dict):
            continue
        if kind == "semantic":
            frame = item["frames"][0]
            payload, receipts, actual_model = _call_frame_with_content_repair(
                caller,
                provider=provider,
                expected_model=expected_model,
                index=int(item["index"]),
                frame_id=str(frame["frame_id"]),
                prompt=str(item.get("prompt") or ""),
                frame_path=str(frame["path"]),
                bundle_dir=str(pack["bundle_dir"]),
                require_semantic_content=True,
            )
            total_calls += len(receipts)
            call_receipts.extend(receipts)
            if payload is None:
                receipt = receipts[-1] if receipts else {}
                result = _content_review_gap(
                    str(receipt.get("reason") or "model_visual_content_empty"),
                    str(receipt.get("detail") or "LM Studio did not return usable visual JSON."),
                    calls=call_receipts,
                    provider_call_count=total_calls,
                ) if _is_content_gap(str(receipt["reason"])) else _retryable(
                    str(receipt.get("reason") or "local_provider_failed"),
                    str(receipt.get("detail") or "Local provider call failed."),
                    calls=call_receipts,
                    provider_call_count=total_calls,
                )
                return _submission(
                    work_item,
                    result,
                )
            actual_models.append(actual_model)
            imported_items.append(
                {
                    "index": int(item["index"]),
                    "visual_understanding": {
                        **payload,
                        "evidence_frame_paths": [str(frame["path"])],
                        "source": "autopilot_local_provider",
                        "executor": "provider_gateway",
                        "model_requested": expected_model,
                        "model_actual": actual_model,
                        "evidence_level": "confirmed",
                    },
                }
            )
        else:
            imported, receipts, models = _execute_temporal_item(
                caller,
                provider=provider,
                expected_model=expected_model,
                item=item,
                bundle_dir=str(pack["bundle_dir"]),
            )
            call_receipts.extend(receipts)
            total_calls += len(receipts)
            if imported is None:
                failed = next((row for row in receipts if not row.get("ok")), {})
                reason = str(failed.get("reason") or "temporal_provider_failed")
                detail = str(failed.get("detail") or "A sequential frame request failed.")
                result = _content_review_gap(
                    reason,
                    detail,
                    calls=call_receipts,
                    provider_call_count=total_calls,
                ) if _is_content_gap(reason) else _retryable(
                    reason,
                    detail,
                    calls=call_receipts,
                    provider_call_count=total_calls,
                )
                return _submission(
                    work_item,
                    result,
                )
            actual_models.extend(models)
            imported_items.append(imported)

    unique_models = list(dict.fromkeys(actual_models))
    if not unique_models:
        return _submission(work_item, _retryable("model_identity_missing", "Provider response omitted model identity.", calls=call_receipts, provider_call_count=total_calls))
    result = {
        "schema": RESULT_SCHEMA,
        "status": "succeeded",
        "kind": kind,
        "task_pack_sha256": str(pack.get("task_pack_sha256") or ""),
        "provider_snapshot_sha256": str(pack.get("provider_snapshot_sha256") or ""),
        "model_requested": expected_model,
        "model_actual": unique_models[0] if len(unique_models) == 1 else ",".join(unique_models),
        "provider_call_count": total_calls,
        "request_mode": "single_frame_sequential",
        "max_images_per_request": 1,
        "items": imported_items,
        "calls": call_receipts,
    }
    return _submission(work_item, result)


def import_provider_vision_result(
    bundle_dir: str | Path,
    *,
    result: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Reuse the existing vision normalisers, audits, refreshes, and restore chain."""

    root = Path(bundle_dir).expanduser().resolve()
    kind = str(result.get("kind") or "")
    items = [dict(row) for row in result.get("items") or [] if isinstance(row, dict)]
    indexes = [int(row.get("index") or 0) for row in items if int(row.get("index") or 0) > 0]
    import_path = Path(output_dir).expanduser().resolve() / f"provider-import-{str(result.get('task_pack_sha256') or '')[:12]}.json"
    write_json(import_path, {"schema": RESULT_SCHEMA, "items": items})
    provider_config = {
        "provider": "local_vlm",
        "base_url": "local://already-executed-provider-import",
        "model": str(result.get("model_actual") or result.get("model_requested") or ""),
    }
    if kind == "semantic":
        imported = run_multimodal_frame_analysis(
            root,
            execute=False,
            provider_config=provider_config,
            input_json=import_path,
            limit=len(indexes),
            indexes=indexes,
        )
    elif kind == "temporal":
        frame_count = max(
            (
                len((row.get("temporal_visual_understanding") or {}).get("evidence_frame_paths") or [])
                for row in items
                if isinstance(row.get("temporal_visual_understanding"), dict)
            ),
            default=8,
        )
        imported = run_temporal_visual_analysis(
            root,
            execute=False,
            provider_config=provider_config,
            input_json=import_path,
            frame_count=frame_count,
            limit=1,
            indexes=indexes,
        )
    else:
        raise ValueError(f"unsupported provider vision result kind: {kind}")
    summary = imported.get("summary") if isinstance(imported.get("summary"), dict) else {}
    updated = int(summary.get("updated") or 0)
    return {
        "status": "imported" if updated == len(indexes) and indexes else "rejected",
        "updated_count": updated,
        "expected_count": len(indexes),
        "indexes": indexes,
        "import_path": str(import_path),
        "analysis": imported,
    }


def validate_provider_vision_result(
    work_item: Mapping[str, Any],
    result: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Validate result identity and exact item/frame coverage before any import."""

    status = str(result.get("status") or "")
    if status != "succeeded":
        return []
    issues: list[dict[str, str]] = []
    try:
        pack_path = _work_item_pack_path(work_item)
        pack = read_json(pack_path)
    except (OSError, ValueError) as exc:
        return [{"key": "provider_task_pack_unavailable", "detail": str(exc)[:500]}]
    if not isinstance(pack, dict):
        return [{"key": "provider_task_pack_invalid", "detail": "pack must be an object"}]
    if str(result.get("schema") or "") != RESULT_SCHEMA:
        issues.append({"key": "provider_result_schema_invalid", "detail": str(result.get("schema") or "")})
    for key in ("task_pack_sha256", "provider_snapshot_sha256"):
        if str(result.get(key) or "") != str(pack.get(key) or ""):
            issues.append({"key": f"{key}_mismatch", "detail": "result identity differs from provider pack"})
    expected_model = str((pack.get("provider") or {}).get("model") or "")
    actual_model = str(result.get("model_actual") or "")
    if not actual_model:
        issues.append({"key": "model_actual_missing", "detail": "provider result must report the actual model"})
    elif _normalise_model(actual_model) != _normalise_model(expected_model):
        issues.append({"key": "model_actual_mismatch", "detail": f"expected {expected_model!r}; received {actual_model!r}"})
    if int(result.get("provider_call_count") or 0) <= 0:
        issues.append({"key": "provider_call_count_invalid", "detail": "successful provider result requires at least one call"})

    expected_items = {int(row.get("index") or 0): row for row in pack.get("items") or [] if isinstance(row, dict)}
    result_rows = [row for row in result.get("items") or [] if isinstance(row, dict)]
    result_indexes = [int(row.get("index") or 0) for row in result_rows]
    if len(result_indexes) != len(set(result_indexes)):
        issues.append({"key": "provider_result_duplicate_index", "detail": str(result_indexes)})
    if set(result_indexes) != set(expected_items):
        issues.append({"key": "provider_result_index_set_mismatch", "detail": f"expected {sorted(expected_items)}; received {sorted(result_indexes)}"})
    kind = str(pack.get("kind") or "")
    for row in result_rows:
        index = int(row.get("index") or 0)
        expected = expected_items.get(index) or {}
        expected_paths = {str(frame.get("path") or "") for frame in expected.get("frames") or [] if isinstance(frame, dict)}
        field = "visual_understanding" if kind == "semantic" else "temporal_visual_understanding"
        payload = row.get(field) if isinstance(row.get(field), dict) else {}
        evidence_paths = {str(value) for value in payload.get("evidence_frame_paths") or [] if str(value)}
        if evidence_paths != expected_paths:
            issues.append({"key": "provider_evidence_path_mismatch", "detail": f"index={index}"})
        if kind == "temporal":
            expected_ids = {str(frame.get("frame_id") or "") for frame in expected.get("frames") or [] if isinstance(frame, dict)}
            observed_ids = {str(value) for value in payload.get("observed_frame_ids") or [] if str(value)}
            observation_ids = {
                str(item.get("frame_id") or "")
                for item in payload.get("per_frame_observations") or []
                if isinstance(item, dict) and str(item.get("frame_id") or "")
            }
            if observed_ids != expected_ids:
                issues.append({"key": "provider_observed_frame_ids_mismatch", "detail": f"index={index}"})
            if observation_ids != expected_ids:
                issues.append({"key": "provider_per_frame_observations_incomplete", "detail": f"index={index}"})
    return issues


def _execute_temporal_item(
    caller: Callable[..., dict[str, Any]],
    *,
    provider: dict[str, Any],
    expected_model: str,
    item: Mapping[str, Any],
    bundle_dir: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    observations: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    models: list[str] = []
    previous = ""
    frames = [dict(row) for row in item.get("frames") or [] if isinstance(row, dict)]
    manifest = [dict(row) for row in item.get("frame_manifest") or [] if isinstance(row, dict)]
    manifest_by_name = {str(row.get("filename") or ""): row for row in manifest}
    for position, frame in enumerate(frames, start=1):
        manifest_row = manifest_by_name.get(Path(str(frame.get("path") or "")).name, {})
        prompt = (
            "Observe exactly this one frame from an ordered video sequence. Return JSON with objects, actions, "
            "interface_state, non_text_information, confidence, and a short observation. Do not infer causality.\n"
            f"Sequence position: {position}/{len(frames)}; frame_id={frame.get('frame_id')}; "
            f"timestamp={manifest_row.get('timestamp')}.\nPrevious observed state: {previous or 'none'}.\n"
            f"Segment context: {str(item.get('prompt') or '')[:1800]}"
        )
        payload, frame_receipts, actual_model = _call_frame_with_content_repair(
            caller,
            provider=provider,
            expected_model=expected_model,
            index=int(item["index"]),
            frame_id=str(frame["frame_id"]),
            prompt=prompt,
            frame_path=str(frame["path"]),
            bundle_dir=bundle_dir,
            require_semantic_content=False,
        )
        receipts.extend(frame_receipts)
        if payload is None:
            return None, receipts, models
        models.append(actual_model)
        observation = _observation_text(payload)
        observations.append(
            {
                "frame_id": str(frame["frame_id"]),
                "timestamp": manifest_row.get("timestamp"),
                "frame_path": str(frame["path"]),
                "observation": observation,
                "objects": _string_list(payload.get("objects")),
                "actions": _string_list(payload.get("actions")),
                "interface_state": str(payload.get("interface_state") or payload.get("ui_state") or ""),
                "confidence": _confidence(payload.get("confidence")),
            }
        )
        previous = observation[:600]

    state_changes = _state_changes(observations)
    frame_ids = [str(row["frame_id"]) for row in observations]
    frame_paths = [str(row["frame_path"]) for row in observations]
    payload = {
        "event_sequence": [f"{row['frame_id']} @ {row.get('timestamp')}: {row['observation']}" for row in observations],
        "state_changes": state_changes or ["static_sequence: no reliable visible state change was established"],
        "operation_steps": [],
        "causal_links": [],
        "possible_missing_points": ["Sequential single-frame aggregation cannot by itself prove hidden motion, intent, or causality."],
        "confidence": min((_confidence(row.get("confidence")) for row in observations), default=0.0),
        "evidence_frame_paths": frame_paths,
        "observed_frame_count": len(frame_ids),
        "observed_frame_ids": frame_ids,
        "per_frame_observations": observations,
        "source": "autopilot_local_provider_single_frame_sequence",
        "executor": "provider_gateway",
        "model_requested": expected_model,
        "model_actual": models[0] if models else "",
        "evidence_level": "confirmed",
    }
    return {"index": int(item["index"]), "temporal_visual_understanding": payload}, receipts, models


def _call_one_frame(
    caller: Callable[..., dict[str, Any]],
    *,
    provider: dict[str, Any],
    prompt: str,
    frame_path: str,
    bundle_dir: str,
    max_tokens: int = AUTOPILOT_VISION_MAX_TOKENS,
) -> dict[str, Any]:
    try:
        return caller(
            "multimodal_frame_analysis",
            provider_config=provider,
            prompt=prompt,
            image_paths=[frame_path],
            execute=True,
            max_tokens=max_tokens,
            max_retries=0,
            allowed_roots=[bundle_dir],
            write=False,
        )
    except Exception as exc:  # Provider boundary: preserve the leased task for explicit recovery.
        return {
            "ok": False,
            "status": "failed",
            "complete": False,
            "error": f"{type(exc).__name__}: {exc}"[:1000],
            "content": "",
        }


def _call_frame_with_content_repair(
    caller: Callable[..., dict[str, Any]],
    *,
    provider: dict[str, Any],
    expected_model: str,
    index: int,
    frame_id: str,
    prompt: str,
    frame_path: str,
    bundle_dir: str,
    require_semantic_content: bool,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
    """Retry only incomplete model content once; transport failures remain explicit waits."""

    receipts: list[dict[str, Any]] = []
    prompts = [prompt, _compact_retry_prompt(prompt)]
    for attempt, request_prompt in enumerate(prompts, start=1):
        response = _call_one_frame(
            caller,
            provider=provider,
            prompt=request_prompt,
            frame_path=frame_path,
            bundle_dir=bundle_dir,
        )
        receipt = _call_receipt(
            response,
            expected_model=expected_model,
            index=index,
            frame_id=frame_id,
        )
        receipt.update(
            {
                "attempt": attempt,
                "request_mode": "primary" if attempt == 1 else "compact_content_repair",
                "request_max_tokens": AUTOPILOT_VISION_MAX_TOKENS,
            }
        )
        receipts.append(receipt)
        if not receipt["ok"]:
            if attempt == 1 and _is_content_gap(str(receipt.get("reason") or "")):
                continue
            return None, receipts, ""

        payload = parse_model_json(str(response.get("content") or ""))
        if payload.get("_parse_failed"):
            receipt.update(
                {
                    "ok": False,
                    "reason": "model_json_invalid",
                    "detail": "LM Studio did not return a complete JSON object.",
                }
            )
            if attempt == 1:
                continue
            return None, receipts, ""

        payload = _normalise_semantic_provider_payload(payload)
        has_content = _has_semantic_visual_content(payload)
        if not require_semantic_content:
            has_content = has_content or bool(str(payload.get("observation") or payload.get("summary") or "").strip())
        if not has_content:
            receipt.update(
                {
                    "ok": False,
                    "reason": "model_visual_content_empty",
                    "detail": "LM Studio returned JSON but no usable visual content.",
                }
            )
            if attempt == 1:
                continue
            return None, receipts, ""
        return payload, receipts, str(receipt["model_actual"])
    return None, receipts, ""


def _compact_retry_prompt(original_prompt: str) -> str:
    context = str(original_prompt or "")[-800:]
    return (
        "Observe only the image. Return compact JSON without markdown or explanation. "
        "Use at most 3 objects, 2 actions, and 3 non_text_information strings; keep each string under 60 characters. "
        "Required fields: objects, actions, interface_state, spatial_relations, instructor_focus, "
        "non_text_information, confidence, keep_image_reason. Do not infer unseen facts.\n"
        f"Bounded context: {context}"
    )


def _call_receipt(response: Mapping[str, Any], *, expected_model: str, index: int, frame_id: str) -> dict[str, Any]:
    actual = _actual_model(response)
    ok = bool(response.get("ok")) and bool(response.get("complete", not response.get("truncated"))) and not bool(response.get("truncated"))
    reason = ""
    detail = ""
    if not ok:
        finish_reason = str(response.get("finish_reason") or "").strip().lower()
        if bool(response.get("truncated")) or finish_reason in {"length", "max_tokens", "max_token", "token_limit"}:
            reason = "provider_output_truncated"
            detail = f"Provider output was truncated (finish_reason={finish_reason or 'unknown'})."[:1000]
        else:
            reason = _provider_failure_reason(str(response.get("error") or response.get("status") or "provider_failed"))
            detail = str(response.get("error") or response.get("status") or "provider call failed")[:1000]
    elif not actual:
        ok = False
        reason = "model_identity_missing"
        detail = "Provider response omitted the actual model id."
    elif _normalise_model(actual) != _normalise_model(expected_model):
        ok = False
        reason = "loaded_model_mismatch"
        detail = f"expected {expected_model!r}; provider reported {actual!r}"
    return {
        "index": index,
        "frame_id": frame_id,
        "ok": ok,
        "reason": reason,
        "detail": detail,
        "model_actual": actual,
        "finish_reason": str(response.get("finish_reason") or ""),
        "truncated": bool(response.get("truncated")),
        "content_sha256": canonical_json_sha256(str(response.get("content") or "")),
    }


def _actual_model(response: Mapping[str, Any]) -> str:
    direct = str(response.get("model_actual") or "").strip()
    if direct:
        return direct
    raw = response.get("raw_response")
    if isinstance(raw, dict):
        return str(raw.get("model") or "").strip()
    runtime = response.get("runtime_result")
    if isinstance(runtime, dict):
        raw_output = runtime.get("raw_output")
        if isinstance(raw_output, dict):
            return str(raw_output.get("model") or "").strip()
    return ""


def _validate_pack(pack_path: Path, pack: Mapping[str, Any], work_item: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    if str(pack.get("task_id") or "") != str(work_item.get("task_id") or ""):
        issues.append("task_id_mismatch")
    if _pack_sha256(pack) != str(pack.get("task_pack_sha256") or ""):
        issues.append("task_pack_hash_mismatch")
    if sha256_file(pack_path) != _ref_sha256(work_item, pack_path):
        issues.append("task_pack_file_hash_mismatch")
    root = Path(str(pack.get("bundle_dir") or "")).expanduser().resolve()
    if not root.is_dir():
        issues.append("bundle_missing")
        return issues
    if sha256_file(root / "manifest.json") != str(pack.get("manifest_sha256") or ""):
        issues.append("manifest_hash_drift")
    if sha256_file(root / "timeline.json") != str(pack.get("timeline_sha256") or ""):
        issues.append("timeline_hash_drift")
    for item in pack.get("items") or []:
        if not isinstance(item, dict):
            issues.append("item_invalid")
            continue
        for frame in item.get("frames") or []:
            if not isinstance(frame, dict):
                issues.append("frame_invalid")
                continue
            path = Path(str(frame.get("path") or "")).expanduser().resolve()
            if not path.is_file() or not path.is_relative_to(root):
                issues.append(f"frame_path_invalid:{path}")
            elif sha256_file(path) != str(frame.get("sha256") or ""):
                issues.append(f"frame_hash_drift:{path}")
    return issues


def _frame_identity(root: Path, path: Path) -> dict[str, Any]:
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError(f"provider frame must be an existing Bundle-owned file: {path}")
    return {
        "frame_id": canonical_json_sha256({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})[:16].upper(),
        "path": str(path),
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _action_selection(action: Mapping[str, Any]) -> tuple[list[int], int, int]:
    value: dict[str, Any] = {}
    path = Path(str(action.get("mcp_args_path") or ""))
    if path.is_file():
        raw = read_json(path)
        if isinstance(raw, dict):
            value = raw
    indexes: list[int] = []
    for raw_index in value.get("indexes") or []:
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if index > 0:
            indexes.append(index)
    profile = resolve_vision_execution_profile()
    limit = max(1, min(int(value.get("limit") or profile["multimodal_limit"] or 19), 19))
    frame_count = max(5, min(int(value.get("frame_count") or profile["frame_count"] or 8), 12))
    return indexes, limit, frame_count


def _action_excluded_indexes(action: Mapping[str, Any]) -> set[int]:
    excluded: set[int] = set()
    for value in action.get("excluded_indexes") or []:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index > 0:
            excluded.add(index)
    return excluded


def _action_kind(action: Mapping[str, Any]) -> str:
    text = " ".join(str(action.get(key) or "") for key in ("mcp_tool", "key", "kind")).lower()
    return "temporal" if "temporal" in text else "semantic"


def _work_item_pack_path(work_item: Mapping[str, Any]) -> Path:
    for row in work_item.get("input_refs") or []:
        if isinstance(row, dict) and row.get("role") == "provider_task_pack":
            return Path(str(row.get("path") or "")).expanduser().resolve()
    raise ValueError("provider work item is missing provider_task_pack")


def _ref_sha256(work_item: Mapping[str, Any], path: Path) -> str:
    for row in work_item.get("input_refs") or []:
        if isinstance(row, dict) and Path(str(row.get("path") or "")).expanduser().resolve() == path:
            return str(row.get("sha256") or "")
    return ""


def _pack_sha256(pack: Mapping[str, Any]) -> str:
    return canonical_json_sha256({key: value for key, value in pack.items() if key not in {"task_pack_sha256", "task_pack_path"}})


def _submission(work_item: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SUBMISSION_SCHEMA,
        "job_id": work_item.get("job_id"),
        "bundle_identity_sha256": work_item.get("bundle_identity_sha256"),
        "state_revision": work_item.get("state_revision"),
        "task_id": work_item.get("task_id"),
        "lease_id": work_item.get("lease_id"),
        "result": dict(result),
    }


def _retryable(
    reason: str,
    detail: str,
    *,
    calls: list[dict[str, Any]] | None = None,
    provider_call_count: int = 0,
) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "status": "retryable",
        "reason": str(reason)[:200],
        "detail": str(detail)[:4000],
        "provider_call_count": int(provider_call_count),
        "calls": list(calls or []),
        "recovery": "Start/load the configured LM Studio model, then run autopilot-next <job-id> --retry-provider.",
    }


def _content_review_gap(
    reason: str,
    detail: str,
    *,
    calls: list[dict[str, Any]] | None = None,
    provider_call_count: int = 0,
) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "status": "review_gap",
        "reason": str(reason)[:200],
        "detail": str(detail)[:4000],
        "provider_call_count": int(provider_call_count),
        "calls": list(calls or []),
    }


def _is_content_gap(reason: str) -> bool:
    return str(reason or "") in {
        "provider_output_truncated",
        "model_output_truncated",
        "model_json_invalid",
        "model_visual_content_empty",
    }


def _provider_failure_reason(error: str) -> str:
    lowered = str(error or "").lower()
    if any(marker in lowered for marker in ("connection refused", "actively refused", "urlopen error", "winerror 10061")):
        return "local_provider_unavailable"
    if "timed out" in lowered or "timeout" in lowered:
        return "local_provider_timeout"
    if "not found" in lowered and "model" in lowered:
        return "loaded_model_mismatch"
    if "truncat" in lowered or "max_token" in lowered:
        return "model_output_truncated"
    return "local_provider_failed"


def _normalise_model(value: str) -> str:
    return str(value or "").strip().lower().replace("\\", "/")


def _is_loopback_url(value: str) -> bool:
    parsed = urlsplit(str(value or "").strip())
    try:
        port = parsed.port
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and str(parsed.hostname or "").lower() in _LOOPBACK_HOSTS and port is not None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value or "").strip() else []


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _observation_text(payload: Mapping[str, Any]) -> str:
    direct = str(payload.get("observation") or payload.get("summary") or "").strip()
    if direct:
        return direct[:1000]
    parts = [
        str(payload.get("interface_state") or payload.get("ui_state") or "").strip(),
        ", ".join(_string_list(payload.get("objects"))),
        ", ".join(_string_list(payload.get("actions"))),
        ", ".join(_string_list(payload.get("non_text_information"))),
    ]
    return "; ".join(part for part in parts if part)[:1000] or "visible state unavailable"


def _normalise_semantic_provider_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalised = dict(payload)
    if _has_semantic_visual_content(normalised):
        return normalised
    rows = [row for row in normalised.get("result") or [] if isinstance(row, dict)]
    if not rows:
        return normalised
    objects = list(
        dict.fromkeys(
            str(row.get("object_type") or row.get("type") or row.get("name") or "").strip()
            for row in rows
            if str(row.get("object_type") or row.get("type") or row.get("name") or "").strip()
        )
    )
    descriptions = []
    for row in rows:
        kind = str(row.get("object_type") or row.get("type") or row.get("name") or "").strip()
        description = str(row.get("description") or row.get("text") or row.get("content") or "").strip()
        if description:
            descriptions.append(f"{kind}: {description}" if kind else description)
    normalised.update(
        {
            "objects": objects,
            "actions": _string_list(normalised.get("actions")),
            "interface_state": str(normalised.get("interface_state") or ""),
            "spatial_relations": _string_list(normalised.get("spatial_relations")),
            "instructor_focus": str(normalised.get("instructor_focus") or ""),
            "non_text_information": list(dict.fromkeys(descriptions)),
            "confidence": _confidence(normalised.get("confidence") or 0.6),
            "keep_image_reason": str(normalised.get("keep_image_reason") or "保留画面中的结构化视觉对象及其位置/说明。"),
            "provider_output_shape": "result_item_list",
        }
    )
    return normalised


def _has_semantic_visual_content(payload: Mapping[str, Any]) -> bool:
    return bool(
        _string_list(payload.get("objects"))
        or _string_list(payload.get("actions"))
        or str(payload.get("interface_state") or "").strip()
        or _string_list(payload.get("non_text_information"))
    )


def _state_changes(observations: list[dict[str, Any]]) -> list[str]:
    changes: list[str] = []
    for before, after in zip(observations, observations[1:]):
        before_state = str(before.get("interface_state") or "").strip()
        after_state = str(after.get("interface_state") or "").strip()
        before_objects = set(_string_list(before.get("objects")))
        after_objects = set(_string_list(after.get("objects")))
        appeared = sorted(after_objects - before_objects)
        disappeared = sorted(before_objects - after_objects)
        parts: list[str] = []
        if before_state and after_state and before_state != after_state:
            parts.append(f"interface_state changed from {before_state!r} to {after_state!r}")
        if appeared:
            parts.append("objects appeared: " + ", ".join(appeared))
        if disappeared:
            parts.append("objects disappeared: " + ", ".join(disappeared))
        if parts:
            changes.append(f"{before['frame_id']} -> {after['frame_id']}: " + "; ".join(parts))
    return changes
