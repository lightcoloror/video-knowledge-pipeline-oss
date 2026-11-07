from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

from .artifact_freshness import canonical_json_sha256
from .file_hash import sha256_file
from .general_tagger_adapter import bundle_frame_records, run_general_tagger
from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .storage import bundle_write_lock, read_json, write_json, write_text_atomic
from .vision_review_triage import TAGGER_PRODUCER_FIELDS
from .powershell import quote_powershell_literal as _ps_quote

SCHEMA = "video_knowledge_pipeline.temporal_tag_delta.v2"
INPUT_SCHEMA_V1 = "video_knowledge_pipeline.temporal_tag_delta_input.v1"
SEQUENCE_SCHEMA = "video_knowledge_pipeline.temporal_tag_sequence.v2"
REPORT_SCHEMA = "video_knowledge_pipeline.temporal_tag_delta_report.v2"
VALIDATION_SCHEMA = "video_knowledge_pipeline.temporal_tag_delta_validation.v2"
RESTORE_SCHEMA = "video_knowledge_pipeline.temporal_tag_delta_restore.v1"
DEFAULT_FRAME_COUNT, MIN_FRAME_COUNT, MAX_FRAME_COUNT = 8, 5, 12
DEFAULT_SEQUENCE_LIMIT = 16
STATIC_CHANGE_THRESHOLD, STATIC_NONEMPTY_COVERAGE = 0.34, 0.80

DYNAMIC_TERMS = (
    "点击", "打开", "关闭", "切换", "拖动", "输入", "移动", "变化", "变成",
    "出现", "消失", "演示", "步骤", "流程", "click", "open", "close",
    "switch", "drag", "type", "move", "change", "appear", "disappear", "step",
)


def run_temporal_tag_delta(
    bundle_dir: str | Path,
    *,
    input_json: str | Path | None = None,
    execute_tagger: bool = False,
    source_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "cuda",
    prefer_language: str = "zh",
    indexes: str | Iterable[int] | None = None,
    frame_count: int = DEFAULT_FRAME_COUNT,
    limit: int = DEFAULT_SEQUENCE_LIMIT,
    min_frames: int = MIN_FRAME_COUNT,
    resume: str | None = None,
    write: bool = True,
    _inference_backend: Callable[[Path], dict[str, object]] | None = None,
) -> dict[str, Any]:
    """Create bounded candidate-only continuous tag evidence from Bundle frames."""
    root = Path(bundle_dir).expanduser().resolve()
    timeline_path, manifest_path = root / "timeline.json", root / "manifest.json"
    if not timeline_path.is_file():
        raise FileNotFoundError(f"timeline.json not found: {timeline_path}")
    if not MIN_FRAME_COUNT <= int(frame_count) <= MAX_FRAME_COUNT:
        raise ValueError("frame_count must be between 5 and 12")
    if not MIN_FRAME_COUNT <= int(min_frames) <= int(frame_count):
        raise ValueError("min_frames must be between 5 and frame_count")
    if not 1 <= int(limit) <= DEFAULT_SEQUENCE_LIMIT:
        raise ValueError("limit must be between 1 and 16 timeline sequences")

    before_text = timeline_path.read_text(encoding="utf-8")
    timeline = json.loads(before_text)
    if not isinstance(timeline, list):
        raise ValueError("timeline.json must be an array")
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be an object when present")
    requested = _parse_indexes(indexes)
    all_candidates = _candidate_rows(root, timeline, indexes=requested, frame_count=int(frame_count))
    candidates = all_candidates[: int(limit)]
    remaining_indexes = [int(row["index"]) for row in all_candidates[int(limit) :]]
    selected_indexes = [int(row["index"]) for row in candidates]
    run_id = _safe_run_id(resume) if resume else f"temporal-tag-{uuid4().hex[:12]}"
    pack = _build_sequence_pack(
        root,
        candidates,
        run_id=run_id,
        timeline_sha256=sha256_file(timeline_path),
        manifest_sha256=sha256_file(manifest_path) if manifest_path.is_file() else "",
        frame_count=int(frame_count),
        sequence_limit=int(limit),
    )

    imported: dict[int, list[dict[str, Any]]] = {}
    validation = {"schema": VALIDATION_SCHEMA, "status": "not_applicable", "errors": []}
    raw_input = ""
    if input_json:
        input_path = Path(input_json).expanduser().resolve()
        raw_input = input_path.read_text(encoding="utf-8-sig")
        imported, validation = _validate_input_payload(json.loads(raw_input), pack)
        if validation["status"] == "rejected":
            return _rejected_result(root, run_id, pack, validation, raw_input, write=write)

    tagger_result: dict[str, Any] = {}
    if execute_tagger:
        selected_paths = [
            frame["absolute_path"] for sequence in pack["sequences"] for frame in sequence["frames"]
        ]
        tagger_result = run_general_tagger(
            root,
            source_root=source_root,
            checkpoint_path=checkpoint_path,
            device=device,
            prefer_language=prefer_language,
            limit=0,
            frame_mode="continuous",
            execute=True,
            import_annotations=False,
            write=False,
            timeline_indexes=selected_indexes,
            per_timeline_limit=int(frame_count),
            selected_artifact_paths=selected_paths,
            cache_dir=root / "runs" / run_id / "frame-cache",
            run_id=run_id,
            _inference_backend=_inference_backend,
        )
        imported = _group_annotations(tagger_result.get("annotations") or [])
        validation = _validate_runtime_observations(imported, pack)

    template = _input_template(pack)
    if not input_json and not execute_tagger and not any(
        _existing_observations(candidate["item"]) for candidate in candidates
    ):
        return _prepared_result(root, run_id, pack, template, remaining_indexes=remaining_indexes,
                                min_frames=int(min_frames), write=write)
    if not input_json and not execute_tagger:
        validation = {
            "schema": VALIDATION_SCHEMA,
            "status": "accepted_legacy_unverified",
            "errors": [],
            "evidence_level_ceiling": "unavailable",
        }
    items: list[dict[str, Any]] = []
    planned_updates: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        index, item = int(candidate["index"]), candidate["item"]
        observations = imported.get(index) or _existing_observations(item)
        observations = [
            {
                **row,
                "executor": str(row.get("executor") or ("ram_plus_general" if execute_tagger else "external_import")),
                "model_actual": str(row.get('model_actual') or '') if 'recognition_binding' in row else str(
                    row.get("model_actual")
                    or row.get("model")
                    or tagger_result.get("model")
                    or ("legacy_external_unverified" if validation.get("status") == "accepted_legacy_unverified" else "")
                ),
                "model_revision": str(row.get("model_revision") or tagger_result.get("model_revision") or ""),
                "run_id": str(row.get("run_id") or run_id),
            }
            for row in observations
        ]
        analysis = analyze_temporal_tag_observations(
            observations,
            transcript=str(item.get("transcript") or item.get("text") or ""),
            supporting_signals=_supporting_signals(item, observations),
            min_frames=int(min_frames),
            expected_frame_count=len(candidate["frames"]),
        )
        analysis.update(
            sequence_id=candidate["sequence_id"],
            frame_manifest_sha256=candidate["sequence_sha256"],
            executor=str((observations[0] if observations else {}).get("executor") or "not_executed"),
            model_actual=str((observations[0] if observations else {}).get("model_actual") or ""),
            model_revision=str((observations[0] if observations else {}).get("model_revision") or ""),
            run_id=run_id,
        )
        # This is a derived description of the recorded frame recognitions,
        # not another model call or an independent observation.
        from .evidence_lineage import project_recognition_provenance
        provenance_rows = [row for row in observations if isinstance(row.get('recognition_provenance'), dict)]
        if provenance_rows:
            models = {str(row.get('model_actual') or '') for row in observations}
            executors = {str(row.get('executor') or '') for row in observations}
            problems = [problem for row in provenance_rows
                        for problem in row['recognition_provenance'].get('problems', [])]
            if len(provenance_rows) != len(observations):
                problems.append('partial_frame_recognition_lineage')
            analysis['recognition_provenance'] = project_recognition_provenance({
                'recognition_provenance': {
                    'frame_sha256s': [row.get('frame_sha256', '') for row in observations],
                    'model_actual': next(iter(models)) if len(models) == 1 else '',
                    'executor': next(iter(executors)) if len(executors) == 1 else '',
                    'problems': problems}}, selected_field=['temporal_tag_delta'],
                text=analysis.get('coarse_summary', ''))
            analysis['model_actual'] = next(iter(models)) if len(models) == 1 else ''
        if validation.get("status") == "accepted_legacy_unverified":
            analysis["status"] = "review_gap"
            analysis["decision"] = "review_gap"
            analysis["evidence_level"] = "unavailable"
            analysis["escalation_reasons"] = _dedupe(
                list(analysis.get("escalation_reasons") or []) + ["legacy_v1_identity_unverified"]
            )
            analysis["limitations"] = _dedupe(
                list(analysis.get("limitations") or [])
                + ["Historical v1 observations remain readable but cannot satisfy the v2 frame identity gate."]
            )
        if isinstance(item.get("temporal_visual_understanding"), dict) and item["temporal_visual_understanding"]:
            analysis["decision"] = "audit_only_formal_temporal_result"
            analysis["limitations"] = _dedupe(
                list(analysis.get("limitations") or [])
                + ["A formal temporal result exists; tag evidence is audit-only."]
            )
        row = {
            "index": index,
            "start": item.get("start", 0),
            "end": item.get("end", 0),
            "frame_paths": [frame["absolute_path"] for frame in candidate["frames"]],
            **analysis,
        }
        items.append(row)
        planned_updates[index] = {
            "temporal_tag_delta": {key: value for key, value in analysis.items() if key != "status"},
            "temporal_multimodal_escalation": _escalation_payload(root, index, analysis),
        }

    counts = _outcome_counts(items)
    gap_count = counts.get("review_gap", 0)
    if execute_tagger and tagger_result.get("status") not in {"completed", ""}:
        status = "needs_frame_tags"
    elif not candidates:
        status = "not_needed"
    elif gap_count == len(items):
        status = "needs_frame_tags"
    elif gap_count:
        status = "partial"
    else:
        status = "completed"
    run_dir = root / "runs" / run_id
    result: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "ok": status in {"completed", "not_needed"},
        "run_id": run_id,
        "bundle_dir": str(root),
        "candidate_count": len(candidates),
        "completed_count": len(items) - gap_count,
        "needs_frame_tags_count": gap_count,
        "coarse_static_count": counts.get("coarse_weak_evidence", 0),
        "temporal_multimodal_count": counts.get("temporal_multimodal_required", 0),
        "review_gap_count": gap_count,
        "outcome_counts": counts,
        "selected_indexes": selected_indexes,
        "remaining_indexes": remaining_indexes,
        "updated_indexes": selected_indexes if write else [],
        "processed_frame_count": sum(len(value) for value in imported.values()),
        "sequence_pack": pack,
        "input_template_path": str(root / "temporal-tag-delta-input-template.json"),
        "task_pack_path": str(run_dir / "task-pack.json"),
        "mcp_args_path": str(root / "mcp-run-temporal-tag-delta.args.json"),
        "validation_report_path": str(run_dir / "validation.json"),
        "raw_result_path": str(run_dir / "raw-result.json") if raw_input else "",
        "restore_plan": str(run_dir / "restore-plan.json"),
        "report_json_path": str(root / "temporal-tag-delta.json"),
        "report_markdown_path": str(root / "temporal-tag-delta.md"),
        "tagger": {
            "executed": bool(execute_tagger),
            "status": str(tagger_result.get("status") or "not_executed"),
            "model": str(tagger_result.get("model") or ""),
            "model_revision": str(tagger_result.get("model_revision") or ""),
            "cache_hit_count": int(tagger_result.get("cache_hit_count") or 0),
            "frame_mode": "continuous",
            "error": tagger_result.get("error"),
        },
        "items": items,
        "next_actions": _next_actions(root, items, run_id, remaining_indexes=remaining_indexes),
        "change_record": _change_record(),
        "operator_boundary": _operator_boundary(),
        "updated_at": now_iso(),
    }
    if remaining_indexes:
        result["next_actions"].append(
            f'.\\scripts\\video-knowledge.ps1 run-temporal-tag-delta "{root}" '
            f'--indexes {",".join(str(value) for value in remaining_indexes)} --limit 16'
        )
    if write:
        _commit_result(
            root,
            timeline_path,
            manifest_path,
            timeline,
            manifest,
            before_text,
            planned_updates,
            result,
            template,
            validation,
            raw_input,
        )
        result["post_update_refresh"] = _refresh_post_tag_outputs(root, updated_count=len(planned_updates))
        write_json(Path(result["report_json_path"]), result)
        write_text_atomic(Path(result["report_markdown_path"]), _render_markdown(result))
        register_bundle_run(
            root,
            run_type="temporal_tag_delta",
            run_id=run_id,
            status=status,
            title="Continuous frame tag delta",
            summary=str(counts),
            inputs={
                "timeline_sha256": pack["timeline_sha256"],
                "manifest_sha256": pack["manifest_sha256"],
                "bundle_identity_sha256": pack["bundle_identity_sha256"],
            },
            parameters={"indexes": selected_indexes, "frame_count": frame_count, "limit": limit},
            artifacts=[
                result["report_json_path"], result["report_markdown_path"],
                result["input_template_path"], result["task_pack_path"], result["mcp_args_path"],
                result["validation_report_path"], result["restore_plan"],
                *([result["raw_result_path"]] if result["raw_result_path"] else []),
            ],
            failed_items=[
                {"index": row["index"], "reason": "review_gap", "detail": ", ".join(row.get("escalation_reasons") or [])}
                for row in items if row.get("decision") == "review_gap"
            ],
            retry_command=f'.\\scripts\\video-knowledge.ps1 run-temporal-tag-delta "{root}" --resume {run_id}',
            next_actions=result["next_actions"],
            operator_boundary=_operator_boundary(),
            resource_requirements={"gpu": 1 if execute_tagger else 0, "network": 0},
            write=True,
        )
    return result


def analyze_temporal_tag_observations(
    observations: list[dict[str, Any]],
    *,
    transcript: str = "",
    supporting_signals: dict[str, Any] | None = None,
    min_frames: int = 3,
    expected_frame_count: int | None = None,
) -> dict[str, Any]:
    rows = _normalise_observations(observations)
    expected = int(expected_frame_count or len(rows))
    duplicate_ids = len({row["frame_id"] for row in rows}) != len(rows)
    if len(rows) < int(min_frames) or len(rows) != expected or duplicate_ids:
        tagged = sum(bool(row["tags"]) for row in rows)
        return {
            "schema": SCHEMA,
            "status": "review_gap",
            "decision": "review_gap",
            "evidence_level": "unavailable",
            "raw_frame_count": len(rows),
            "observed_frame_count": len(rows),
            "expected_frame_count": expected,
            "tagged_frame_count": tagged,
            "nonempty_coverage": round(tagged / max(1, expected), 4),
            "smoothed_frame_count": 0,
            "raw_observations": rows,
            "smoothed_observations": [],
            "stable_tags": [],
            "transitions": [],
            "tag_change_score": None,
            "coarse_summary": "",
            "escalation_reasons": ["incomplete_frame_observation"],
            "limitations": ["Incomplete frame observations cannot support a coarse description."],
        }
    signals = dict(supporting_signals or {})
    smoothed = _smooth_tags(rows)
    stable, transitions = _stable_tags(smoothed), _transitions(smoothed)
    change_score = _tag_change_score(smoothed)
    tagged = sum(bool(row["tags"]) for row in smoothed)
    coverage = round(tagged / max(1, len(smoothed)), 4)
    reasons: list[str] = []
    if change_score >= STATIC_CHANGE_THRESHOLD:
        reasons.append("high_tag_change")
    if signals.get("ocr_changed"):
        reasons.append("ocr_changed")
    if signals.get("scene_boundary"):
        reasons.append("scene_boundary")
    if signals.get("cross_channel_conflict"):
        reasons.append("cross_channel_conflict")
    if str(signals.get("frame_change_status") or "") in {"dynamic", "localized_motion"}:
        reasons.append(f"frame_change_{signals['frame_change_status']}")
    if _contains_dynamic_term(transcript):
        reasons.append("asr_dynamic_term")
    if not stable:
        reasons.append("no_stable_tags")
    if coverage < STATIC_NONEMPTY_COVERAGE:
        reasons.append("insufficient_nonempty_tag_coverage")
    reasons = _dedupe(reasons)
    decision = "temporal_multimodal_required" if reasons else "coarse_weak_evidence"
    return {
        "schema": SCHEMA,
        "status": "completed",
        "decision": decision,
        "evidence_level": "candidate_weak",
        "raw_frame_count": len(rows),
        "observed_frame_count": len(rows),
        "expected_frame_count": expected,
        "tagged_frame_count": tagged,
        "nonempty_coverage": coverage,
        "smoothed_frame_count": len(smoothed),
        "raw_observations": rows,
        "smoothed_observations": smoothed,
        "stable_tags": stable,
        "transitions": transitions,
        "tag_change_score": change_score,
        "supporting_signals": signals,
        "coarse_summary": _coarse_summary(stable, transitions, decision),
        "escalation_reasons": reasons,
        "limitations": [
            "Tags identify likely visible contents, not actions, temporal order, intent, or causality.",
            "Use uncertainty wording such as 画面标签显示、可能、疑似、似乎 or 看起来.",
        ],
    }


def apply_temporal_tag_delta_restore(
    bundle_dir: str | Path,
    plan_json: str | Path,
    *,
    execute: bool = False,
    confirm_run_id: str = "",
) -> dict[str, Any]:
    root, plan_path = Path(bundle_dir).expanduser().resolve(), Path(plan_json).expanduser().resolve()
    plan = read_json(plan_path)
    if not isinstance(plan, dict) or plan.get("schema") != RESTORE_SCHEMA:
        raise ValueError("invalid temporal tag restore plan")
    run_id, blockers = str(plan.get("run_id") or ""), []
    if execute and confirm_run_id != run_id:
        blockers.append("confirm_run_id_mismatch")
    timeline_path = root / "timeline.json"
    backup_path = Path(str(plan.get("timeline_backup_path") or "")).resolve()
    if not timeline_path.is_file() or sha256_file(timeline_path) != plan.get("timeline_after_sha256"):
        blockers.append("timeline_drift")
    if not backup_path.is_file() or sha256_file(backup_path) != plan.get("timeline_before_sha256"):
        blockers.append("restore_backup_drift")
    status = "blocked" if blockers else "ready"
    if execute and not blockers:
        with bundle_write_lock(root, operation="temporal_tag_delta_restore", timeout_seconds=2.0):
            write_text_atomic(timeline_path, backup_path.read_text(encoding="utf-8"))
        status = "restored"
    return {
        "schema": "video_knowledge_pipeline.temporal_tag_delta_restore_receipt.v1",
        "status": status,
        "run_id": run_id,
        "execute": bool(execute),
        "blockers": blockers,
        "timeline_path": str(timeline_path),
        "restored_sha256": sha256_file(timeline_path) if timeline_path.is_file() else "",
        "operator_boundary": _operator_boundary(),
        "updated_at": now_iso(),
    }


def _candidate_rows(root: Path, timeline: list[Any], *, indexes: list[int], frame_count: int) -> list[dict[str, Any]]:
    records = bundle_frame_records(
        root, timeline, frame_mode="continuous", timeline_indexes=indexes, per_timeline_limit=frame_count
    )
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(int(row["timeline_index"]), []).append(row)
    output: list[dict[str, Any]] = []
    for position, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        index = _positive_int(item.get("index")) or position
        if indexes and index not in indexes:
            continue
        frames = grouped.get(index) or []
        if len(frames) < 2:
            continue
        frame_rows = [
            {
                "frame_position": int(frame["frame_position"]),
                "frame_id": str(frame["frame_id"]),
                "source_frame_id": str(frame.get("source_frame_id") or ""),
                "timestamp": float(frame["timestamp"]),
                "absolute_path": str(frame["frame_path"]),
                "relative_path": str(Path(frame["frame_path"]).relative_to(root)),
                "sha256": sha256_file(frame["frame_path"]),
            }
            for frame in frames
        ]
        identity = {
            "index": index,
            "start": float(item.get("start") or 0),
            "end": float(item.get("end") or 0),
            "frames": [
                {key: frame[key] for key in ("frame_position", "frame_id", "timestamp", "relative_path", "sha256")}
                for frame in frame_rows
            ],
        }
        digest = canonical_json_sha256(identity)
        output.append(
            {
                "index": index,
                "item": item,
                "start": identity["start"],
                "end": identity["end"],
                "sequence_id": f"timeline-{index:06d}-{digest[:12]}",
                "sequence_sha256": digest,
                "frames": frame_rows,
            }
        )
    return output


def _build_sequence_pack(
    root: Path,
    candidates: list[dict[str, Any]],
    *,
    run_id: str,
    timeline_sha256: str,
    manifest_sha256: str,
    frame_count: int,
    sequence_limit: int,
) -> dict[str, Any]:
    sequences = [
        {
            "index": row["index"], "sequence_id": row["sequence_id"], "start": row["start"],
            "end": row["end"], "sequence_sha256": row["sequence_sha256"],
            "frames": [dict(frame) for frame in row["frames"]],
        }
        for row in candidates
    ]
    identity = canonical_json_sha256(
        {
            "bundle_dir": str(root), "timeline_sha256": timeline_sha256,
            "manifest_sha256": manifest_sha256,
            "sequence_sha256s": [row["sequence_sha256"] for row in sequences],
        }
    )
    pack = {
        "schema": SEQUENCE_SCHEMA,
        "run_id": run_id,
        "bundle_dir": str(root),
        "timeline_sha256": timeline_sha256,
        "manifest_sha256": manifest_sha256,
        "bundle_identity_sha256": identity,
        "frame_count": int(frame_count),
        "sequence_limit": int(sequence_limit),
        "sequences": sequences,
        "limitations": ["Candidate weak evidence only; no action, order, intent, or causality claims."],
    }
    pack["task_pack_sha256"] = canonical_json_sha256(pack)
    return pack


def _validate_input_payload(payload: Any, expected: dict[str, Any]) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    if isinstance(payload, dict) and payload.get("schema") == INPUT_SCHEMA_V1:
        return _read_legacy_input(payload), {
            "schema": VALIDATION_SCHEMA, "status": "accepted_legacy_unverified", "errors": [],
            "evidence_level_ceiling": "candidate_weak",
        }
    errors: list[str] = []
    if not isinstance(payload, dict) or payload.get("schema") != SEQUENCE_SCHEMA:
        return {}, {"schema": VALIDATION_SCHEMA, "status": "rejected", "errors": ["unsupported_input_schema"]}
    for key in ("run_id", "timeline_sha256", "manifest_sha256", "bundle_identity_sha256", "task_pack_sha256"):
        if payload.get(key) != expected.get(key):
            errors.append(f"{key}_mismatch")
    expected_sequences, actual_sequences = expected.get("sequences") or [], payload.get("sequences") or []
    result_metadata = payload.get("result_metadata") if isinstance(payload.get("result_metadata"), dict) else {}
    executor = str(result_metadata.get("executor") or "").strip()
    model_actual = str(result_metadata.get("model_actual") or "").strip()
    model_revision = str(result_metadata.get("model_revision") or "").strip()
    if not executor:
        errors.append("result_executor_missing")
    if not model_actual:
        errors.append("result_model_actual_missing")
    if any(len(value) > 200 for value in (executor, model_actual, model_revision)):
        errors.append("result_metadata_too_long")
    if len(actual_sequences) != len(expected_sequences):
        errors.append("sequence_set_mismatch")
    observations: dict[int, list[dict[str, Any]]] = {}
    for expected_sequence, actual_sequence in zip(expected_sequences, actual_sequences):
        if not isinstance(actual_sequence, dict):
            errors.append("invalid_sequence")
            continue
        for key in ("index", "sequence_id", "start", "end", "sequence_sha256"):
            if actual_sequence.get(key) != expected_sequence.get(key):
                errors.append(f"sequence_{key}_mismatch")
        expected_frames, actual_frames = expected_sequence.get("frames") or [], actual_sequence.get("frames") or []
        if len(actual_frames) != len(expected_frames):
            errors.append("frame_set_mismatch")
            continue
        rows: list[dict[str, Any]] = []
        for expected_frame, actual_frame in zip(expected_frames, actual_frames):
            if not isinstance(actual_frame, dict):
                errors.append("invalid_frame")
                continue
            for key in ("frame_position", "frame_id", "timestamp", "absolute_path", "relative_path", "sha256"):
                if actual_frame.get(key) != expected_frame.get(key):
                    errors.append(f"frame_{key}_mismatch")
            tags = actual_frame.get("tags") or []
            if not isinstance(tags, list) or len(tags) > 100 or any(len(str(tag)) > 200 for tag in tags):
                errors.append("frame_tags_invalid_or_too_long")
            if len(str(actual_frame.get("ocr_text") or "")) > 4000:
                errors.append("frame_ocr_text_too_long")
            rows.append(
                {
                    "frame_position": actual_frame.get("frame_position"), "frame_id": actual_frame.get("frame_id"),
                    "frame_path": actual_frame.get("absolute_path"), "timestamp": actual_frame.get("timestamp"),
                    "frame_sha256": actual_frame.get("sha256"),
                    "tags": actual_frame.get("tags") or [], "ocr_text": actual_frame.get("ocr_text") or "",
                    "confidence": actual_frame.get("confidence"),
                    "tag_confidences": actual_frame.get("tag_confidences") or {},
                    "boundary_before": bool(actual_frame.get("boundary_before")),
                    "executor": executor,
                    "model_actual": model_actual,
                    "model_revision": model_revision,
                    "run_id": str(expected.get("run_id") or ""),
                }
            )
        observations[int(expected_sequence.get("index") or 0)] = rows
    return observations, {
        "schema": VALIDATION_SCHEMA,
        "status": "rejected" if errors else "accepted",
        "errors": _dedupe(errors),
    }


def _validate_runtime_observations(observations: dict[int, list[dict[str, Any]]], pack: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    for sequence in pack.get("sequences") or []:
        index = int(sequence["index"])
        expected_ids = [frame["frame_id"] for frame in sequence["frames"]]
        actual_ids = [row.get("frame_id") for row in observations.get(index, [])]
        if actual_ids != expected_ids:
            errors.append(f"runtime_frame_set_mismatch:{index}")
    return {"schema": VALIDATION_SCHEMA, "status": "rejected" if errors else "accepted", "errors": errors}


def _commit_result(
    root: Path,
    timeline_path: Path,
    manifest_path: Path,
    timeline: list[Any],
    manifest: dict[str, Any],
    before_text: str,
    updates: dict[int, dict[str, Any]],
    result: dict[str, Any],
    template: dict[str, Any],
    validation: dict[str, Any],
    raw_input: str,
) -> None:
    run_id = result["run_id"]
    run_dir = root / "runs" / run_id
    backup_path, restore_path = run_dir / "timeline.before.json", Path(result["restore_plan"])
    with bundle_write_lock(root, operation="temporal_tag_delta_v2", timeout_seconds=2.0):
        if sha256_file(timeline_path) != result["sequence_pack"]["timeline_sha256"]:
            raise RuntimeError("timeline_hash_drift_before_commit")
        expected_manifest_sha = str(result["sequence_pack"].get("manifest_sha256") or "")
        actual_manifest_sha = sha256_file(manifest_path) if manifest_path.is_file() else ""
        if actual_manifest_sha != expected_manifest_sha:
            raise RuntimeError("manifest_hash_drift_before_commit")
        for sequence in result["sequence_pack"].get("sequences") or []:
            for frame in sequence.get("frames") or []:
                frame_path = Path(str(frame.get("absolute_path") or "")).resolve()
                try:
                    frame_path.relative_to(root)
                except ValueError as exc:
                    raise RuntimeError("frame_path_outside_bundle_before_commit") from exc
                if not frame_path.is_file() or sha256_file(frame_path) != frame.get("sha256"):
                    raise RuntimeError(f"frame_hash_drift_before_commit:{frame.get('frame_id')}")
        write_text_atomic(backup_path, before_text)
        for position, item in enumerate(timeline, start=1):
            if isinstance(item, dict):
                update = updates.get(_positive_int(item.get("index")) or position)
                if update:
                    item.update(deepcopy(update))
        write_json(timeline_path, timeline)
        restore = {
            "schema": RESTORE_SCHEMA,
            "run_id": run_id,
            "bundle_dir": str(root),
            "timeline_path": str(timeline_path),
            "timeline_backup_path": str(backup_path),
            "timeline_before_sha256": sha256_file(backup_path),
            "timeline_after_sha256": sha256_file(timeline_path),
            "updated_indexes": result["selected_indexes"],
            "created_at": now_iso(),
        }
        write_json(restore_path, restore)
        write_json(Path(result["input_template_path"]), template)
        write_json(Path(result["task_pack_path"]), result["sequence_pack"])
        write_json(
            Path(result["mcp_args_path"]),
            {
                "bundle_dir": str(root),
                "indexes": ",".join(str(value) for value in result["selected_indexes"]),
                "frame_count": int(result["sequence_pack"].get("frame_count") or DEFAULT_FRAME_COUNT),
                "limit": len(result["selected_indexes"]),
                "min_frames": MIN_FRAME_COUNT,
                "execute_tagger": False,
                "write": True,
            },
        )
        write_json(Path(result["validation_report_path"]), validation)
        if raw_input and result.get("raw_result_path"):
            write_text_atomic(Path(result["raw_result_path"]), raw_input)
        write_json(Path(result["report_json_path"]), result)
        write_text_atomic(Path(result["report_markdown_path"]), _render_markdown(result))
        manifest["temporal_tag_delta"] = {
            "schema": SCHEMA, "status": result["status"], "run_id": run_id,
            "outcome_counts": result["outcome_counts"],
            "model_actual": _dedupe([str(row.get("model_actual") or "") for row in result.get("items") or []]),
            "report_json": Path(result["report_json_path"]).name,
            "report_markdown": Path(result["report_markdown_path"]).name,
            "input_template_json": Path(result["input_template_path"]).name,
            "task_pack": str(Path(result["task_pack_path"]).relative_to(root)),
            "mcp_args_json": Path(result["mcp_args_path"]).name,
            "validation_report": str(Path(result["validation_report_path"]).relative_to(root)),
            "raw_result": str(Path(result["raw_result_path"]).relative_to(root)) if result.get("raw_result_path") else "",
            "run_audit": str((root / "runs" / run_id / "run.md").relative_to(root)),
            "restore_plan": str(restore_path.relative_to(root)), "updated_at": result["updated_at"],
        }
        manifest.update(
            {
                "temporal_tag_delta_report": Path(result["report_markdown_path"]).name,
                "temporal_tag_delta_json": Path(result["report_json_path"]).name,
                "temporal_tag_delta_input_template": Path(result["input_template_path"]).name,
                "temporal_tag_delta_task_pack": manifest["temporal_tag_delta"]["task_pack"],
                "temporal_tag_delta_raw_result": manifest["temporal_tag_delta"]["raw_result"],
                "temporal_tag_delta_validation": manifest["temporal_tag_delta"]["validation_report"],
                "temporal_tag_delta_run_audit": manifest["temporal_tag_delta"]["run_audit"],
                "temporal_tag_delta_restore": manifest["temporal_tag_delta"]["restore_plan"],
            }
        )
        write_json(manifest_path, manifest)


def _prepared_result(
    root: Path,
    run_id: str,
    pack: dict[str, Any],
    template: dict[str, Any],
    *,
    remaining_indexes: list[int],
    min_frames: int,
    write: bool,
) -> dict[str, Any]:
    template_path = root / "temporal-tag-delta-input-template.json"
    task_pack_path = root / "runs" / run_id / "task-pack.json"
    validation_path = root / "runs" / run_id / "validation.json"
    report_path = root / "runs" / run_id / "prepare-report.json"
    selected = [int(row["index"]) for row in pack.get("sequences") or []]
    submit_args = {"bundle_dir": str(root), "input_json": str(template_path), "resume": run_id,
                   "indexes": selected, "frame_count": int(pack["frame_count"]),
                   "limit": int(pack["sequence_limit"]), "min_frames": min_frames, "write": True}
    result = {
        "schema": REPORT_SCHEMA,
        "status": "prepared",
        "ok": True,
        "run_id": run_id,
        "bundle_dir": str(root),
        "selected_indexes": [int(row["index"]) for row in pack.get("sequences") or []],
        "remaining_indexes": list(remaining_indexes),
        "processed_frame_count": 0,
        "outcome_counts": {},
        "sequence_pack": pack,
        "input_template_path": str(template_path),
        "task_pack_path": str(task_pack_path),
        "validation_report_path": str(validation_path),
        "prepare_report_path": str(report_path),
        "submit_command": (
            f'.\\scripts\\video-knowledge.ps1 run-temporal-tag-delta {_ps_quote(root)} '
            f'--input-json {_ps_quote(template_path)} --resume {_ps_quote(run_id)} '
            f'--indexes {_ps_quote(",".join(str(value) for value in selected))} '
            f'--frame-count {pack["frame_count"]} --limit {pack["sequence_limit"]} '
            f'--min-frames {min_frames}'
        ),
        "submit_args": submit_args,
        "next_actions": ["Fill tags in the immutable task template, then import it with the same run_id."],
        "change_record": _change_record(),
        "operator_boundary": _operator_boundary(),
        "updated_at": now_iso(),
    }
    if write:
        write_json(task_pack_path, pack)
        write_json(template_path, template)
        write_json(
            validation_path,
            {
                "schema": VALIDATION_SCHEMA,
                "status": "prepared_zero_business_state_change",
                "errors": [],
                "timeline_sha256": pack["timeline_sha256"],
                "manifest_sha256": pack["manifest_sha256"],
            },
        )
        write_json(report_path, result)
    return result


def _refresh_post_tag_outputs(root: Path, *, updated_count: int) -> dict[str, Any]:
    """Refresh shared local audits without invoking models or exporting knowledge."""
    if updated_count <= 0:
        return {"status": "skipped", "reason": "no_timeline_updates"}
    from .bundle_readiness import audit_bundle_readiness
    from .knowledge_coverage import audit_knowledge_coverage

    coverage = audit_knowledge_coverage(root, write=True)
    readiness = audit_bundle_readiness(root, write=True)
    freshness_path = root / "knowledge-export-freshness.json"
    freshness_receipt = {
        "schema": "video_knowledge_pipeline.knowledge_export_freshness_receipt.v1",
        "status": "stale",
        "reason": "timeline_continuous_tag_weak_evidence_updated_after_last_export",
        "updated_timeline_item_count": int(updated_count),
        "automatic_export_performed": False,
        "recovery_command": f'.\\scripts\\video-knowledge.ps1 export-knowledge-note "{root}"',
        "written_at": now_iso(),
    }
    write_json(freshness_path, freshness_receipt)
    return {
        "status": "ok",
        "knowledge_coverage_path": str(coverage.get("coverage_path") or ""),
        "review_readiness_status": str((readiness.get("review_readiness") or {}).get("status") or ""),
        "knowledge_export_freshness": "stale",
        "knowledge_export_freshness_path": str(freshness_path),
    }


def _rejected_result(root: Path, run_id: str, pack: dict[str, Any], validation: dict[str, Any], raw_input: str, *, write: bool) -> dict[str, Any]:
    run_dir, attempt_id = root / "runs" / run_id, uuid4().hex[:12]
    raw_path = run_dir / f"raw-result-rejected-{attempt_id}.json"
    validation_path = run_dir / f"validation-rejected-{attempt_id}.json"
    audit_path = run_dir / f"rejection-audit-{attempt_id}.json"
    result = {
        "schema": REPORT_SCHEMA, "status": "rejected", "ok": False, "run_id": run_id,
        "bundle_dir": str(root),
        "selected_indexes": [int(row["index"]) for row in pack.get("sequences") or []],
        "processed_frame_count": 0, "outcome_counts": {}, "sequence_pack": pack,
        "validation_report_path": str(validation_path), "raw_result_path": str(raw_path),
        "audit_path": str(audit_path),
        "next_actions": ["Regenerate the input from the current immutable sequence pack."],
        "operator_boundary": _operator_boundary(), "updated_at": now_iso(),
    }
    if write:
        write_text_atomic(raw_path, raw_input)
        write_json(validation_path, validation)
        write_json(
            audit_path,
            {
                "schema": "video_knowledge_pipeline.temporal_tag_delta_rejection_audit.v1",
                "status": "rejected_zero_business_state_change",
                "run_id": run_id,
                "task_pack_sha256": pack.get("task_pack_sha256"),
                "timeline_sha256": pack.get("timeline_sha256"),
                "manifest_sha256": pack.get("manifest_sha256"),
                "raw_result_path": str(raw_path),
                "validation_report_path": str(validation_path),
                "errors": list(validation.get("errors") or []),
                "timeline_mutation": False,
                "manifest_mutation": False,
                "written_at": now_iso(),
            },
        )
    return result


def _input_template(pack: dict[str, Any]) -> dict[str, Any]:
    template = deepcopy(pack)
    template["result_metadata"] = {"executor": "", "model_actual": "", "model_revision": ""}
    for sequence in template.get("sequences") or []:
        for frame in sequence.get("frames") or []:
            frame.update(tags=[], ocr_text="", confidence=None, tag_confidences={})
    return template


def _read_legacy_input(value: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    rows = value.get("items")
    if not isinstance(rows, list):
        raise ValueError("temporal tag input must contain an items array")
    result: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if isinstance(row, dict):
            index, frames = _positive_int(row.get("index") or row.get("timeline_index")), row.get("frames") or row.get("observations")
            if index and isinstance(frames, list):
                result[index] = [dict(frame) for frame in frames if isinstance(frame, dict)]
    return result


def _existing_observations(item: dict[str, Any]) -> list[dict[str, Any]]:
    for owner in (item, item.get("integrated_visual")):
        if isinstance(owner, dict):
            for key in ("temporal_frame_tags", "tagger_frame_observations", "per_frame_tags"):
                value = owner.get(key)
                if isinstance(value, list):
                    return [dict(row) for row in value if isinstance(row, dict)]
    return []


def _group_annotations(rows: list[Any]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        index = _positive_int(row.get("timeline_index") or row.get("index"))
        if index:
            result.setdefault(index, []).append(
                {
                    "frame_position": row.get("frame_position"), "frame_id": row.get("frame_id"),
                    "frame_path": row.get("artifact_path"), "timestamp": row.get("timestamp"),
                    "frame_sha256": row.get("artifact_sha256") or "",
                    "tags": row.get("tags") or [], "confidence": row.get("confidence"),
                    "tag_confidences": row.get("tag_confidences") or {},
                    "executor": row.get("executor") or row.get("source") or "",
                    "model_actual": row.get("model_actual") or row.get("model") or "",
                    "model_revision": row.get("model_revision") or "",
                    "run_id": row.get("run_id") or "",
                    **{key: deepcopy(row[key]) for key in TAGGER_PRODUCER_FIELDS if key in row},
                }
            )
    return result


def _supporting_signals(item: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
    ocr = [re.sub(r"\s+", "", str(row.get("ocr_text") or "")) for row in observations if str(row.get("ocr_text") or "").strip()]
    triage = item.get("vision_review_triage") if isinstance(item.get("vision_review_triage"), dict) else {}
    frame_change = item.get("frame_change_evidence")
    frame_change = frame_change if isinstance(frame_change, dict) else triage.get("frame_change_evidence")
    frame_change = frame_change if isinstance(frame_change, dict) else {}
    scene = item.get("scene_boundary_evidence")
    scene = scene if isinstance(scene, dict) else triage.get("scene_boundary_evidence")
    scene = scene if isinstance(scene, dict) else {}
    return {
        "ocr_changed": len(set(ocr)) > 1 or bool(item.get("ocr_change_detected")),
        "scene_boundary": bool(scene.get("matched") or item.get("scene_boundary_matched")),
        "frame_change_status": str(frame_change.get("status") or item.get("frame_change_status") or "not_available"),
        "cross_channel_conflict": bool(item.get("visual_cross_channel_conflict")),
    }


def _normalise_observations(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, row in enumerate(value or [], start=1):
        if isinstance(row, dict):
            rows.append(
                {
                    "frame_position": _positive_int(row.get("frame_position")) or position,
                    "frame_id": str(row.get("frame_id") or f"frame-{position:03d}"),
                    "frame_path": str(row.get("frame_path") or row.get("artifact_path") or ""),
                    "timestamp": _float_or_none(row.get("timestamp")),
                    "frame_sha256": str(row.get("frame_sha256") or row.get("artifact_sha256") or ""),
                    "tags": _normalise_tags(row.get("tags") or row.get("labels") or row.get("labels_zh") or row.get("labels_en")),
                    "ocr_text": str(row.get("ocr_text") or ""),
                    "confidence": _float_or_none(row.get("confidence")),
                    "tag_confidences": dict(row.get("tag_confidences") or {}) if isinstance(row.get("tag_confidences"), dict) else {},
                    "boundary_before": bool(row.get("boundary_before")),
                    "preserve_transient": bool(row.get("preserve_transient")),
                    "executor": str(row.get("executor") or ""),
                    "model_actual": str(row.get("model_actual") or row.get("model") or ""),
                    "model_revision": str(row.get("model_revision") or ""),
                    "run_id": str(row.get("run_id") or ""),
                    **{key: deepcopy(row[key]) for key in TAGGER_PRODUCER_FIELDS if key in row},
                }
            )
    rows.sort(key=lambda row: int(row["frame_position"]))
    return rows


def _smooth_tags(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sets, smoothed = [set(row["tags"]) for row in rows], [set(row["tags"]) for row in rows]
    for position in range(1, len(sets) - 1):
        if rows[position].get("boundary_before") or rows[position + 1].get("boundary_before"):
            continue
        if _neighbor_support(rows[position - 1]) and _neighbor_support(rows[position + 1]):
            smoothed[position].update((sets[position - 1] & sets[position + 1]) - sets[position])
        if not _transient_supported(rows, position):
            smoothed[position].difference_update(sets[position] - sets[position - 1] - sets[position + 1])
    return [{**row, "tags": sorted(smoothed[position])} for position, row in enumerate(rows)]


def _neighbor_support(row: dict[str, Any]) -> bool:
    return row.get("confidence") is None or float(row["confidence"]) >= 0.5


def _transient_supported(rows: list[dict[str, Any]], position: int) -> bool:
    row = rows[position]
    if row.get("preserve_transient") or (row.get("confidence") is not None and float(row["confidence"]) >= 0.75):
        return True
    current = re.sub(r"\s+", "", str(row.get("ocr_text") or ""))
    if not current:
        return False
    left = re.sub(r"\s+", "", str(rows[position - 1].get("ocr_text") or ""))
    right = re.sub(r"\s+", "", str(rows[position + 1].get("ocr_text") or ""))
    return current not in {left, right}


def _stable_tags(rows: list[dict[str, Any]]) -> list[str]:
    threshold, counts = max(2, math.ceil(len(rows) * 0.6)), {}
    for row in rows:
        for tag in set(row["tags"]):
            counts[tag] = counts.get(tag, 0) + 1
    return sorted(tag for tag, count in counts.items() if count >= threshold)


def _transitions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for previous, current in zip(rows, rows[1:]):
        before, after = set(previous["tags"]), set(current["tags"])
        appeared, disappeared = sorted(after - before), sorted(before - after)
        if appeared or disappeared:
            output.append(
                {
                    "from_frame_position": previous["frame_position"], "to_frame_position": current["frame_position"],
                    "from_frame_id": previous["frame_id"], "to_frame_id": current["frame_id"],
                    "appeared": appeared, "disappeared": disappeared, "support": "observed_tag_delta_only",
                }
            )
    return output


def _tag_change_score(rows: list[dict[str, Any]]) -> float:
    distances: list[float] = []
    for previous, current in zip(rows, rows[1:]):
        left, right = set(previous["tags"]), set(current["tags"])
        union = left | right
        distances.append(0.0 if not union else 1.0 - len(left & right) / len(union))
    return round(sum(distances) / max(1, len(distances)), 4)


def _coarse_summary(stable: list[str], transitions: list[dict[str, Any]], decision: str) -> str:
    stable_text = "、".join(stable[:12]) or "无稳定标签"
    if decision == "temporal_multimodal_required":
        return f"画面标签显示可能包含：{stable_text}；存在变化或歧义，需升级连续帧多模态理解。"
    if transitions:
        return f"画面标签显示可能持续包含：{stable_text}；仅记录少量标签变化，不据此推断动作。"
    return f"画面标签显示可能持续包含：{stable_text}；标签基本稳定，但不据此推断动作或因果。"


def _escalation_payload(root: Path, index: int, analysis: dict[str, Any]) -> dict[str, Any]:
    required = analysis.get("decision") == "temporal_multimodal_required"
    review_gap = analysis.get("decision") == "review_gap"
    return {
        "required": required,
        "decision": analysis.get("decision"),
        "reasons": list(analysis.get("escalation_reasons") or []),
        "fallback_route": (
            "existing_temporal_visual_analysis"
            if required
            else "local_retag_or_review_gap"
            if review_gap
            else "none"
        ),
        "tag_delta_is_supporting_evidence_only": True,
        "suggested_command": (
            f'.\\scripts\\video-knowledge.ps1 run-temporal-visual-analysis "{root}" --index {index}'
            if required
            else f'.\\scripts\\video-knowledge.ps1 run-temporal-tag-delta "{root}" --indexes {index} --execute-tagger'
            if review_gap
            else ""
        ),
    }


def _next_actions(
    root: Path,
    items: list[dict[str, Any]],
    run_id: str,
    *,
    remaining_indexes: list[int],
) -> list[str]:
    actions: list[str] = []
    for row in items:
        if row.get("decision") == "temporal_multimodal_required":
            actions.append(f'.\\scripts\\video-knowledge.ps1 run-temporal-visual-analysis "{root}" --index {row["index"]}')
        elif row.get("decision") == "review_gap":
            actions.append(f'.\\scripts\\video-knowledge.ps1 run-temporal-tag-delta "{root}" --indexes {row["index"]} --resume {run_id} --execute-tagger')
    if remaining_indexes:
        actions.append(
            f'.\\scripts\\video-knowledge.ps1 run-temporal-tag-delta "{root}" '
            f'--indexes {",".join(str(value) for value in remaining_indexes)} --limit 16 --execute-tagger'
        )
    return _dedupe(actions)


def _outcome_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in items:
        decision = str(row.get("decision") or "review_gap")
        counts[decision] = counts.get(decision, 0) + 1
    return counts


def _parse_indexes(value: str | Iterable[int] | None) -> list[int]:
    if value is None:
        return []
    raw_values: Iterable[Any] = re.split(r"[,;\s]+", value.strip()) if isinstance(value, str) else value
    result: list[int] = []
    for raw in raw_values:
        if raw in {None, ""}:
            continue
        index = int(raw)
        if index <= 0:
            raise ValueError("indexes must contain positive integers")
        if index not in result:
            result.append(index)
    return result


def _normalise_tags(value: Any) -> list[str]:
    values = re.split(r"[|,，、;；\n]+", value) if isinstance(value, str) else value if isinstance(value, list) else []
    output, seen = [], set()
    for raw in values:
        tag, key = str(raw or "").strip(), str(raw or "").strip().casefold()
        if tag and key not in seen:
            seen.add(key)
            output.append(tag)
    return output


def _contains_dynamic_term(value: str) -> bool:
    lowered = str(value or "").casefold()
    return any(term.casefold() in lowered for term in DYNAMIC_TERMS)


def _change_record() -> dict[str, Any]:
    return {
        "intent": "Use bounded continuous-frame tags as low-cost weak evidence before temporal multimodal calls.",
        "decision": "Bind observations to immutable Bundle frames, apply multi-signal gates, and fail closed on drift.",
        "reason": "Object labels cannot prove actions, order, intent, or causality.",
        "evidence": ["frame_sha256", "tag_delta", "ocr_change", "scene_boundary", "frame_change", "asr_dynamic_terms"],
        "effective_scope": "timeline.temporal_tag_delta and timeline.temporal_multimodal_escalation only",
        "rollback": "Apply the per-run restore plan; source frames, ASR, OCR, human facts, and formal temporal results remain untouched.",
    }


def _operator_boundary() -> dict[str, Any]:
    return {
        "local_only": True, "remote_calls_made": 0, "automatic_model_download": False,
        "automatic_remote_fallback": False, "tag_delta_is_not_action_or_causality_proof": True,
        "does_not_populate_temporal_visual_understanding": True, "source_video_read": False,
    }


def _safe_run_id(value: str | None) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "")).strip("-._")
    if not text:
        raise ValueError("resume run id is invalid")
    return text


def _positive_int(value: Any) -> int:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Temporal Tag Delta v2", "", f"- Status: `{result.get('status')}`",
        f"- Run ID: `{result.get('run_id')}`", f"- Candidates: `{result.get('candidate_count')}`",
        f"- Coarse weak evidence: `{result.get('coarse_static_count')}`",
        f"- Escalate: `{result.get('temporal_multimodal_count')}`",
        f"- Review gaps: `{result.get('review_gap_count')}`",
        "- Boundary: candidate weak evidence only; no action, order, intent, or causality claims.", "",
        "| Index | Frames | Coverage | Change | Decision | Reasons |", "|---:|---:|---:|---:|---|---|",
    ]
    for row in result.get("items") or []:
        lines.append(
            "| {index} | {frames} | {coverage} | {score} | {decision} | {reasons} |".format(
                index=row.get("index"), frames=row.get("observed_frame_count", 0),
                coverage=row.get("nonempty_coverage", 0),
                score=row.get("tag_change_score") if row.get("tag_change_score") is not None else "-",
                decision=row.get("decision", ""), reasons=", ".join(row.get("escalation_reasons") or []) or "-",
            )
        )
    return "\n".join(lines).rstrip() + "\n"
