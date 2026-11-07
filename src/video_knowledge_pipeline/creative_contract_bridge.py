from __future__ import annotations

import json
import math
import mimetypes
import re
from pathlib import Path
from typing import Any, Iterable

import jsonschema

from .artifact_freshness import build_dependency_snapshot, canonical_json_sha256
from .artifact_freshness import validate_dependency_snapshot
from .artifact_validation import artifact_evidence, normalise_allowed_roots, validated_local_file
from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .smart_summary_input_pack import select_canonical_transcript_path
from .storage import read_json, write_json


GENERATION_TASK_SCHEMA = "video_creation_pipeline.generation_task.v1"
GENERATION_RECEIPT_SCHEMA = "video_creation_pipeline.generation_receipt.v1"
GENERATION_VALIDATION_SCHEMA = "video_creation_pipeline.generation_validation.v1"
CAPABILITY_PREFLIGHT_SCHEMA = "video_creation_pipeline.capability_preflight.v1"
PREVIS_SCENE_SCHEMA = "video_creation_pipeline.previs_scene.v1"
PREVIS_CAPTURE_MANIFEST_SCHEMA = "video_creation_pipeline.previs_capture_manifest.v1"
PREVIS_CAPTURE_VALIDATION_SCHEMA = "video_creation_pipeline.previs_capture_validation.v1"

GENERATION_IMPORT_SCHEMA = "video_knowledge_pipeline.generation_contract_import.v1"
PREVIS_EVIDENCE_SCHEMA = "video_knowledge_pipeline.previs_candidate_evidence.v1"
MATERIAL_MANIFEST_SCHEMA = "material-manifest.v1"
MATERIAL_MANIFEST_SCHEMA_VERSION = "1.0"
MATERIAL_MANIFEST_NATIVE_CONTRACT = "video_knowledge_pipeline.material_reference.v1"
MATERIAL_MANIFEST_VALIDATION_SCHEMA = "video_knowledge_pipeline.material_manifest_validation.v1"


def build_material_manifest(
    bundle_dir: str | Path,
    *,
    transcript_path: str | Path | None = None,
    output_path: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Project one VKP Bundle into the shared material-manifest.v1 facade.

    The facade contains only content-addressed local references and video-temporal
    ordering. It never copies transcript text, invents document-node order, or
    grants consent/execution authority.
    """

    root = _bundle_root(bundle_dir)
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    manifest = read_json(manifest_path)
    timeline = read_json(timeline_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    if not isinstance(timeline, list):
        raise ValueError("timeline.json must be a JSON array")
    if not timeline:
        raise ValueError("material manifest requires at least one timeline item")

    transcript = _material_transcript_path(root, manifest, transcript_path)
    timeline_items, frame_paths = _material_timeline_items(root, timeline)
    if not frame_paths:
        raise ValueError("material manifest requires at least one local keyframe")

    inputs: list[dict[str, Any]] = [
        {"role": "bundle_manifest", "path": manifest_path},
        {"role": "timeline", "path": timeline_path},
        {"role": "canonical_transcript", "path": transcript},
    ]
    inputs.extend(
        {"role": f"keyframe_{position:06d}", "path": path}
        for position, path in enumerate(frame_paths, start=1)
    )
    dependency_snapshot = build_dependency_snapshot(
        root,
        subject="material-manifest.v1",
        inputs=inputs,
        producer_schema=MATERIAL_MANIFEST_NATIVE_CONTRACT,
    )
    # Creation time is deliberately excluded: identical Bundle inputs must
    # produce byte-identical manifests on repeated runs.
    dependency_snapshot.pop("created_at", None)

    identity: dict[str, Any] = {
        "schema": MATERIAL_MANIFEST_SCHEMA,
        "schema_version": MATERIAL_MANIFEST_SCHEMA_VERSION,
        "native_contract": MATERIAL_MANIFEST_NATIVE_CONTRACT,
        "producer": "video-knowledge-pipeline",
        "bundle": {
            "title": str(manifest.get("title") or manifest.get("source_title") or root.name),
            "native_schema": str(manifest.get("schema") or ""),
            "manifest": _material_artifact_reference(root, manifest_path, kind="bundle_metadata"),
            "timeline": _material_artifact_reference(root, timeline_path, kind="time_evidence"),
            "timeline_item_count": len(timeline_items),
        },
        "transcript": {
            "kind": "transcript",
            "artifact": _material_artifact_reference(root, transcript, kind="transcript"),
            "content_embedded": False,
        },
        "timeline_items": timeline_items,
        "dependency_snapshot": dependency_snapshot,
        "authority_boundary": {
            "metadata_only": True,
            "execution_authorized": False,
            "external_io_performed": False,
            "grants_provider_consent": False,
            "grants_review_approval": False,
            "mutates_timeline": False,
            "document_node_order_claimed": False,
            "source_order_semantics": "video_temporal_only",
        },
    }
    manifest_sha256 = canonical_json_sha256(identity)
    result = {
        **identity,
        "manifest_id": f"vkp-material-{manifest_sha256[:20]}",
        "manifest_sha256": manifest_sha256,
    }
    _validate_material_manifest_payload(root, result, require_fresh=True)
    if write:
        target = _material_output_path(root, output_path)
        write_json(target, result)
    return result


def validate_material_manifest(
    bundle_dir: str | Path,
    manifest_path: str | Path | None = None,
    *,
    write_report: bool = False,
) -> dict[str, Any]:
    """Fail closed when a material manifest is invalid, drifted, or stale."""

    root = _bundle_root(bundle_dir)
    source = _material_output_path(root, manifest_path)
    payload = read_json(source)
    if not isinstance(payload, dict):
        raise ValueError("material manifest must be a JSON object")
    report = _validate_material_manifest_payload(root, payload, require_fresh=True)
    if write_report:
        write_json(root / "exports" / "material-manifest-validation.json", report)
    return report


def _validate_material_manifest_payload(
    root: Path,
    payload: dict[str, Any],
    *,
    require_fresh: bool,
) -> dict[str, Any]:
    schema = _material_manifest_json_schema()
    try:
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"material manifest schema validation failed: {exc.message}") from exc

    identity = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_id", "manifest_sha256"}
    }
    expected_hash = canonical_json_sha256(identity)
    supplied_hash = str(payload.get("manifest_sha256") or "")
    if supplied_hash != expected_hash:
        raise ValueError("material manifest hash drift detected")
    if payload.get("manifest_id") != f"vkp-material-{expected_hash[:20]}":
        raise ValueError("material manifest id does not bind its exact hash")

    items = payload.get("timeline_items") if isinstance(payload.get("timeline_items"), list) else []
    bundle = payload.get("bundle") if isinstance(payload.get("bundle"), dict) else {}
    if int(bundle.get("timeline_item_count") or 0) != len(items):
        raise ValueError("material manifest timeline item count drift detected")
    source_orders = [int(row.get("source_order") or 0) for row in items if isinstance(row, dict)]
    if source_orders != list(range(1, len(items) + 1)):
        raise ValueError("material manifest source order is invalid")
    timeline_indexes = [int(row.get("timeline_index") or 0) for row in items if isinstance(row, dict)]
    if len(timeline_indexes) != len(set(timeline_indexes)):
        raise ValueError("material manifest timeline indexes must be unique")
    starts = [int((row.get("time_range") or {}).get("start_ms") or 0) for row in items]
    if starts != sorted(starts):
        raise ValueError("material manifest temporal source order is invalid")

    frame_ids: list[str] = []
    keyframe_count = 0
    for row in items:
        for frame in row.get("keyframes") or []:
            keyframe_count += 1
            frame_ids.append(str(frame.get("frame_id") or ""))
            if (frame.get("timestamp_ms") is None) != (frame.get("timing_status") == "range_only"):
                raise ValueError("material manifest keyframe timing status is inconsistent")
    if keyframe_count == 0:
        raise ValueError("material manifest requires at least one keyframe")
    if not all(frame_ids) or len(frame_ids) != len(set(frame_ids)):
        raise ValueError("material manifest frame ids must be unique and non-empty")

    snapshot = payload.get("dependency_snapshot")
    freshness = validate_dependency_snapshot(root, snapshot if isinstance(snapshot, dict) else {})
    if require_fresh and freshness.get("status") != "fresh":
        raise ValueError(f"material manifest bundle is not fresh: {freshness.get('status')}")

    artifact_rows = _material_manifest_artifact_rows(payload)
    artifact_paths = {str(reference.get("path") or "") for _label, reference in artifact_rows}
    snapshot_paths = {
        str(row.get("path") or "")
        for row in (snapshot.get("inputs") or [])
        if isinstance(row, dict)
    } if isinstance(snapshot, dict) else set()
    if artifact_paths != snapshot_paths:
        raise ValueError("material manifest dependency snapshot does not bind every artifact")
    for label, reference in artifact_rows:
        path = _material_reference_path(root, reference, label=label)
        evidence = artifact_evidence(path)
        if int(reference.get("bytes") or -1) != int(evidence["bytes"]):
            raise ValueError(f"{label} byte length drift detected")
        if str(reference.get("sha256") or "") != str(evidence["sha256"]):
            raise ValueError(f"{label} hash drift detected")

    for row in items:
        time_range = row["time_range"]
        start_ms = int(time_range["start_ms"])
        end_ms = int(time_range["end_ms"])
        for frame in row.get("keyframes") or []:
            timestamp_ms = frame.get("timestamp_ms")
            if timestamp_ms is not None and not start_ms <= int(timestamp_ms) <= end_ms:
                raise ValueError("material manifest keyframe timestamp is outside its timeline range")

    return {
        "schema": MATERIAL_MANIFEST_VALIDATION_SCHEMA,
        "status": "valid",
        "passed": True,
        "manifest_id": payload["manifest_id"],
        "manifest_sha256": supplied_hash,
        "freshness": freshness,
        "timeline_item_count": len(items),
        "keyframe_count": keyframe_count,
        "transcript_content_embedded": False,
        "metadata_authorizes_execution": False,
    }


def _material_transcript_path(
    root: Path,
    manifest: dict[str, Any],
    transcript_path: str | Path | None,
) -> Path:
    selected = transcript_path or select_canonical_transcript_path(root, manifest)
    if not selected:
        raise FileNotFoundError("material manifest requires a canonical transcript")
    return _material_bundle_file(root, selected, label="canonical transcript")


def _material_timeline_items(
    root: Path,
    timeline: list[Any],
) -> tuple[list[dict[str, Any]], list[Path]]:
    items: list[dict[str, Any]] = []
    dependency_frames: list[Path] = []
    previous_start = -1
    seen_indexes: set[int] = set()
    for source_order, raw in enumerate(timeline, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"timeline item {source_order} must be a JSON object")
        timeline_index = _material_positive_int(raw.get("index"), fallback=source_order)
        if timeline_index in seen_indexes:
            raise ValueError("timeline indexes must be unique")
        seen_indexes.add(timeline_index)
        start_ms = _material_seconds_to_ms(raw.get("start"), label=f"timeline {timeline_index} start")
        end_ms = _material_seconds_to_ms(raw.get("end"), label=f"timeline {timeline_index} end")
        if end_ms < start_ms:
            raise ValueError(f"timeline {timeline_index} end precedes start")
        if start_ms < previous_start:
            raise ValueError("timeline source order must be chronological")
        previous_start = start_ms

        frame_values = _material_frame_values(raw)
        keyframes: list[dict[str, Any]] = []
        for frame_position, value in enumerate(frame_values, start=1):
            frame_path = _material_bundle_file(
                root,
                value,
                label=f"timeline {timeline_index} keyframe {frame_position}",
            )
            timestamp_ms = _material_frame_timestamp_ms(raw, frame_path, frame_position, len(frame_values))
            if timestamp_ms is not None and not start_ms <= timestamp_ms <= end_ms:
                raise ValueError(
                    f"timeline {timeline_index} keyframe timestamp {timestamp_ms}ms is outside {start_ms}-{end_ms}ms"
                )
            keyframes.append(
                {
                    "frame_id": f"timeline-{timeline_index:06d}-frame-{frame_position:03d}",
                    "artifact": _material_artifact_reference(root, frame_path, kind="keyframe"),
                    "timestamp_ms": timestamp_ms,
                    "timing_status": "exact" if timestamp_ms is not None else "range_only",
                }
            )
            dependency_frames.append(frame_path)
        keyframes.sort(
            key=lambda row: (
                row["timestamp_ms"] is None,
                int(row["timestamp_ms"] or 0),
                str((row.get("artifact") or {}).get("path") or ""),
            )
        )
        evidence_ids = [str(value).strip() for value in raw.get("evidence_ids") or [] if str(value).strip()]
        if not evidence_ids:
            evidence_ids = [f"timeline:{timeline_index}"]
        items.append(
            {
                "source_order": source_order,
                "timeline_index": timeline_index,
                "time_range": {"start_ms": start_ms, "end_ms": end_ms},
                "evidence_ids": evidence_ids,
                "keyframes": keyframes,
            }
        )
    unique_frames = list(dict.fromkeys(dependency_frames))
    return items, unique_frames


def _material_frame_values(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    single = str(row.get("frame_path") or "").strip()
    if single:
        values.append(single)
    for key in ("frame_paths", "temporal_frame_paths"):
        for value in row.get(key) or []:
            text = str(value or "").strip()
            if text:
                values.append(text)
    for asset in row.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        if str(asset.get("copied") or "").lower() not in {"true", "1"}:
            continue
        text = str(asset.get("path") or "").strip()
        if text:
            values.append(text)
    return list(dict.fromkeys(values))


def _material_frame_timestamp_ms(
    row: dict[str, Any],
    path: Path,
    position: int,
    frame_count: int,
) -> int | None:
    timestamps = row.get("frame_timestamps_ms")
    if isinstance(timestamps, list) and position <= len(timestamps):
        return _material_nonnegative_int(timestamps[position - 1], label="frame timestamp")
    if isinstance(timestamps, dict):
        for key in (str(path), path.name, path.as_posix()):
            if key in timestamps:
                return _material_nonnegative_int(timestamps[key], label="frame timestamp")
    if frame_count == 1 and row.get("timestamp_ms") is not None:
        return _material_nonnegative_int(row.get("timestamp_ms"), label="frame timestamp")
    match = re.search(r"(?P<timestamp>\d+)ms(?=\.[^.]+$|$)", path.name, flags=re.IGNORECASE)
    if match:
        return int(match.group("timestamp"))
    return None


def _material_artifact_reference(root: Path, path: Path, *, kind: str) -> dict[str, Any]:
    evidence = artifact_evidence(path)
    return {
        "kind": kind,
        "path": path.relative_to(root).as_posix(),
        "bytes": int(evidence["bytes"]),
        "sha256": str(evidence["sha256"]),
        "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    }


def _material_manifest_artifact_rows(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    bundle = payload.get("bundle") if isinstance(payload.get("bundle"), dict) else {}
    transcript = payload.get("transcript") if isinstance(payload.get("transcript"), dict) else {}
    rows: list[tuple[str, dict[str, Any]]] = [
        ("bundle manifest", bundle.get("manifest") or {}),
        ("timeline", bundle.get("timeline") or {}),
        ("canonical transcript", transcript.get("artifact") or {}),
    ]
    for item in payload.get("timeline_items") or []:
        for frame in item.get("keyframes") or []:
            rows.append((f"keyframe {frame.get('frame_id')}", frame.get("artifact") or {}))
    return rows


def _material_reference_path(root: Path, reference: dict[str, Any], *, label: str) -> Path:
    return _material_bundle_file(root, str(reference.get("path") or ""), label=label)


def _material_bundle_file(root: Path, value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return validated_local_file(path, label=label, allowed_roots=(root,))


def _material_output_path(root: Path, value: str | Path | None) -> Path:
    path = Path(value).expanduser() if value else root / "exports" / "material-manifest.v1.json"
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise ValueError(f"material manifest output is outside bundle: {resolved}")
    return resolved


def _material_seconds_to_ms(value: Any, *, label: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric seconds") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return int(round(number * 1000))


def _material_positive_int(value: Any, *, fallback: int) -> int:
    if value is None or value == "":
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeline index must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError("timeline index must be a positive integer")
    return parsed


def _material_nonnegative_int(value: Any, *, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return parsed


def _material_manifest_json_schema() -> dict[str, Any]:
    path = Path(__file__).parent / "schemas" / "material-manifest.v1.schema.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"material manifest schema is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("material manifest schema must be a JSON object")
    return value


def import_generation_contracts(
    bundle_dir: str | Path,
    *,
    task_path: str | Path,
    receipt_path: str | Path,
    validation_path: str | Path,
    preflight_path: str | Path | None = None,
    allowed_roots: Iterable[str | Path] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Import already-validated video-creation generation contracts.

    VKP intentionally does not reproduce the upstream executor or verifier.
    The fixed video-creation CLI remains responsible for creating and fully
    verifying the contracts; this bridge checks their bindings and authority
    boundary before copying them into the Bundle as candidate evidence.
    """

    root = _bundle_root(bundle_dir)
    roots = _source_roots(root, allowed_roots)
    task_source, task = _read_contract(task_path, "generation task", GENERATION_TASK_SCHEMA, roots)
    receipt_source, receipt = _read_contract(receipt_path, "generation receipt", GENERATION_RECEIPT_SCHEMA, roots)
    validation_source, validation = _read_contract(
        validation_path, "generation validation", GENERATION_VALIDATION_SCHEMA, roots
    )
    _validate_generation_binding(task, receipt, validation)
    _validate_generation_artifacts(receipt, roots)

    preflight_reference = task.get("capability_preflight")
    if not isinstance(preflight_reference, dict):
        raise ValueError("generation task lacks capability_preflight evidence")
    resolved_preflight = preflight_path or str(preflight_reference.get("path") or "")
    preflight_source, preflight = _read_contract(
        resolved_preflight, "capability preflight", CAPABILITY_PREFLIGHT_SCHEMA, roots
    )
    _validate_reference(preflight_source, preflight_reference, "capability preflight")
    capability = str(task.get("required_capability") or "")
    capability_result = _capability_result(preflight, capability)

    imported = _import_contract_set(
        root,
        {
            "generation-task": (task_source, task),
            "generation-receipt": (receipt_source, receipt),
            "generation-validation": (validation_source, validation),
            "capability-preflight": (preflight_source, preflight),
        },
    )
    snapshot = build_dependency_snapshot(
        root,
        subject="generation-contract-import",
        inputs=[{"role": role, "path": path} for role, path in imported.items()],
        source_run_id=str(task.get("task_id") or ""),
        producer_schema=GENERATION_VALIDATION_SCHEMA,
    )
    representative_frames = _representative_frames(receipt)
    status = str(receipt.get("status") or "")
    result: dict[str, Any] = {
        "schema": GENERATION_IMPORT_SCHEMA,
        "generated_at": now_iso(),
        "bundle_dir": str(root),
        "status": "completed" if status == "completed" else "degraded",
        "upstream_status": status,
        "task_id": str(task.get("task_id") or ""),
        "task_sha256": str(task.get("task_sha256") or ""),
        "generator": str(task.get("generator") or ""),
        "required_capability": capability,
        "capability_preflight": capability_result,
        "technical_verification": dict(receipt.get("technical_verification") or {}),
        "visual_verification": {
            "inspected": bool((receipt.get("visual_verification") or {}).get("inspected")),
            "blank": bool((receipt.get("visual_verification") or {}).get("blank")),
            "black": bool((receipt.get("visual_verification") or {}).get("black")),
            "wrong_composition": bool((receipt.get("visual_verification") or {}).get("wrong_composition")),
            "representative_frames": representative_frames,
        },
        "validation": dict(validation),
        "imported_contracts": {role: str(path.relative_to(root)) for role, path in imported.items()},
        "dependency_snapshot": snapshot,
        "authority_boundary": {
            "candidate_only": True,
            "mutates_vkp_timeline": False,
            "publishing_allowed": False,
            "silent_fallback_allowed": False,
            "upstream_verifier": "video-creative-contracts verify-generation-receipt",
        },
    }
    if write:
        output_json = root / "exports" / "generation-contract-import.json"
        output_md = root / "exports" / "generation-contract-import.md"
        snapshot_path = root / "exports" / "generation-contract-dependency-snapshot.json"
        write_json(output_json, result)
        output_md.write_text(_generation_markdown(result), encoding="utf-8")
        write_json(snapshot_path, snapshot)
        _update_manifest(
            root,
            {
                "generation_contract_import_json": "exports/generation-contract-import.json",
                "generation_contract_import_markdown": "exports/generation-contract-import.md",
                "generation_contract_dependency_snapshot": "exports/generation-contract-dependency-snapshot.json",
            },
        )
        run_status = "completed" if status == "completed" else "needs_review"
        register_bundle_run(
            root,
            run_type="generation_contract_import",
            run_id=f"generation-{str(task.get('task_id') or 'candidate')}",
            status=run_status,
            title="生成能力预检与代表帧证据",
            summary=(
                f"Imported {task.get('generator')} candidate; capability {capability} is ready; "
                f"representative frames: {len(representative_frames)}."
            ),
            inputs={"task_id": task.get("task_id"), "task_sha256": task.get("task_sha256")},
            parameters={
                "generator": task.get("generator"),
                "required_capability": capability,
                "probe_method": capability_result.get("probe_method"),
                "candidate_only": True,
            },
            artifacts=[output_json, output_md, snapshot_path, *imported.values()],
            failed_items=(
                []
                if status == "completed"
                else [{"id": task.get("task_id"), "reason": status, "detail": receipt.get("error", "")}]
            ),
            retry_command="video-creative-contracts verify-generation-receipt --task <task.json> --receipt <receipt.json>",
            next_actions=["在 Workbench 检查代表帧与构图；候选结果不得直接发布或覆盖 Timeline。"],
            operator_boundary=result["authority_boundary"],
            dependency_snapshot=snapshot,
            write=True,
        )
        result["paths"] = {"json": str(output_json), "markdown": str(output_md), "snapshot": str(snapshot_path)}
    return result


def import_previs_candidate(
    bundle_dir: str | Path,
    *,
    scene_path: str | Path,
    manifest_path: str | Path,
    validation_path: str | Path,
    allowed_roots: Iterable[str | Path] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Import fixed-upstream 3D previs captures as synthetic candidate evidence."""

    root = _bundle_root(bundle_dir)
    roots = _source_roots(root, allowed_roots)
    scene_source, scene = _read_contract(scene_path, "previs scene", PREVIS_SCENE_SCHEMA, roots)
    capture_source, capture_manifest = _read_contract(
        manifest_path, "previs capture manifest", PREVIS_CAPTURE_MANIFEST_SCHEMA, roots
    )
    validation_source, validation = _read_contract(
        validation_path, "previs capture validation", PREVIS_CAPTURE_VALIDATION_SCHEMA, roots
    )
    cameras, captures = _validate_previs_binding(scene, capture_manifest, validation, roots)
    imported = _import_contract_set(
        root,
        {
            "previs-scene": (scene_source, scene),
            "previs-capture-manifest": (capture_source, capture_manifest),
            "previs-capture-validation": (validation_source, validation),
        },
    )
    snapshot = build_dependency_snapshot(
        root,
        subject="previs-candidate-evidence",
        inputs=[{"role": role, "path": path} for role, path in imported.items()],
        source_run_id=str((scene.get("scene") or {}).get("scene_id") or ""),
        producer_schema=PREVIS_CAPTURE_VALIDATION_SCHEMA,
    )
    result: dict[str, Any] = {
        "schema": PREVIS_EVIDENCE_SCHEMA,
        "generated_at": now_iso(),
        "bundle_dir": str(root),
        "status": "needs_review",
        "scene": dict(scene.get("scene") or {}),
        "active_camera_id": str(scene.get("active_camera_id") or ""),
        "cameras": cameras,
        "captures": captures,
        "validation": dict(validation),
        "imported_contracts": {role: str(path.relative_to(root)) for role, path in imported.items()},
        "dependency_snapshot": snapshot,
        "authority_boundary": {
            "synthetic": True,
            "candidate_only": True,
            "observed_video_fact": False,
            "mutates_vkp_timeline": False,
            "final_render_claimed": False,
            "human_confirmation_required": True,
            "upstream_verifier": "video-creative-contracts verify-previs-captures",
        },
    }
    if write:
        output_json = root / "exports" / "previs-candidate-evidence.json"
        output_md = root / "exports" / "previs-candidate-evidence.md"
        snapshot_path = root / "exports" / "previs-candidate-dependency-snapshot.json"
        write_json(output_json, result)
        output_md.write_text(_previs_markdown(result), encoding="utf-8")
        write_json(snapshot_path, snapshot)
        _update_manifest(
            root,
            {
                "previs_candidate_evidence_json": "exports/previs-candidate-evidence.json",
                "previs_candidate_evidence_markdown": "exports/previs-candidate-evidence.md",
                "previs_candidate_dependency_snapshot": "exports/previs-candidate-dependency-snapshot.json",
            },
        )
        scene_id = str((scene.get("scene") or {}).get("scene_id") or "candidate")
        register_bundle_run(
            root,
            run_type="previs_candidate_import",
            run_id=f"previs-{scene_id}",
            status="needs_review",
            title="3D 预演候选证据",
            summary=f"Imported {len(captures)} synthetic capture(s) from {len(cameras)} camera(s); human review required.",
            inputs={"scene_id": scene_id, "previs_scene_sha256": scene.get("previs_scene_sha256")},
            parameters={"synthetic": True, "candidate_only": True, "observed_video_fact": False},
            artifacts=[output_json, output_md, snapshot_path, *imported.values()],
            retry_command="video-creative-contracts verify-previs-captures --scene <scene.json> --manifest <captures.json>",
            next_actions=["人工确认相机意图与构图后，方可作为 temporal visual candidate；禁止写回观察事实。"],
            operator_boundary=result["authority_boundary"],
            dependency_snapshot=snapshot,
            write=True,
        )
        result["paths"] = {"json": str(output_json), "markdown": str(output_md), "snapshot": str(snapshot_path)}
    return result


def _validate_generation_binding(
    task: dict[str, Any], receipt: dict[str, Any], validation: dict[str, Any]
) -> None:
    if task.get("task_sha256") != _payload_sha256(task, "task_sha256"):
        raise ValueError("generation task integrity check failed")
    policy = task.get("execution_policy") if isinstance(task.get("execution_policy"), dict) else {}
    if (
        policy.get("claim_after_preflight") is not True
        or policy.get("implicit_install_allowed") is not False
        or policy.get("silent_generator_fallback") is not False
    ):
        raise ValueError("generation task execution policy is unsafe")
    completion = task.get("completion_requirements") if isinstance(task.get("completion_requirements"), dict) else {}
    if (
        completion.get("technical_probe_required") is not True
        or completion.get("representative_frame_review_required") is not True
        or completion.get("acceptance_status") != "candidate_only"
    ):
        raise ValueError("generation task completion requirements are incomplete")
    if receipt.get("task_id") != task.get("task_id") or receipt.get("task_sha256") != task.get("task_sha256"):
        raise ValueError("generation receipt does not bind the exact task")
    if receipt.get("generator") != task.get("generator"):
        raise ValueError("generation receipt generator differs from the task")
    if receipt.get("published") is not False or receipt.get("fallback_used") is not False:
        raise ValueError("generation receipt reports publishing or fallback")
    status = str(receipt.get("status") or "")
    if status not in {"completed", "failed", "cancelled"}:
        raise ValueError("generation receipt status is invalid")
    if validation.get("valid") is not True:
        raise ValueError("upstream generation validation did not pass")
    for field in ("task_id", "task_sha256", "status"):
        if validation.get(field) != (task.get(field) if field != "status" else receipt.get("status")):
            raise ValueError(f"generation validation {field} binding differs")
    if validation.get("acceptance_status") != "candidate_only":
        raise ValueError("generation validation is not candidate-only")
    if status == "completed":
        technical = receipt.get("technical_verification") if isinstance(receipt.get("technical_verification"), dict) else {}
        visual = receipt.get("visual_verification") if isinstance(receipt.get("visual_verification"), dict) else {}
        if not str(technical.get("probe_method") or "").strip():
            raise ValueError("completed generation lacks a technical probe")
        if visual.get("inspected") is not True:
            raise ValueError("completed generation lacks visual inspection")
        if any(visual.get(field) is not False for field in ("blank", "black", "wrong_composition")):
            raise ValueError("completed generation failed visual inspection")
        frames = visual.get("representative_frames") if isinstance(visual.get("representative_frames"), list) else []
        if not frames or validation.get("representative_frame_count") != len(frames):
            raise ValueError("representative frame evidence is incomplete")
    elif not str(receipt.get("error") or "").strip():
        raise ValueError("failed or cancelled generation lacks a visible reason")


def _capability_result(preflight: dict[str, Any], capability: str) -> dict[str, Any]:
    rows = preflight.get("capabilities") if isinstance(preflight.get("capabilities"), list) else []
    matches = [row for row in rows if isinstance(row, dict) and row.get("id") == capability]
    if len(matches) != 1:
        raise ValueError(f"capability preflight must contain exactly one {capability} result")
    row = dict(matches[0])
    if row.get("ready") is not True or row.get("installation_performed") is not False:
        raise ValueError(f"required capability is not preflight-ready: {capability}")
    if str(row.get("probe_method") or "") not in {
        "existing_runtime",
        "registered_tool_health_check",
        "provider_capability_receipt",
        "manual_operator_verification",
    }:
        raise ValueError("capability preflight probe method is not auditable")
    return row


def _validate_generation_artifacts(
    receipt: dict[str, Any], roots: tuple[Path, ...]
) -> None:
    if str(receipt.get("status") or "") != "completed":
        return
    output = receipt.get("output") if isinstance(receipt.get("output"), dict) else {}
    output_path = validated_local_file(
        str(output.get("path") or ""), label="generated output", allowed_roots=roots
    )
    _validate_reference(output_path, output, "generated output")
    for index, frame in enumerate(_representative_frames(receipt), start=1):
        path = validated_local_file(
            str(frame.get("path") or ""),
            label=f"representative frame {index}",
            allowed_roots=roots,
        )
        _validate_reference(path, frame, f"representative frame {index}")

def _validate_previs_binding(
    scene: dict[str, Any],
    manifest: dict[str, Any],
    validation: dict[str, Any],
    roots: tuple[Path, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if scene.get("previs_scene_sha256") != _payload_sha256(scene, "previs_scene_sha256"):
        raise ValueError("previs scene integrity check failed")
    boundary = scene.get("authority_boundary") if isinstance(scene.get("authority_boundary"), dict) else {}
    if (
        boundary.get("derived_artifact") is not True
        or boundary.get("browser_local_storage_authoritative") is not False
        or boundary.get("vkp_timeline_mutated") is not False
        or boundary.get("final_render_claimed") is not False
        or boundary.get("acceptance_status") != "candidate_only"
    ):
        raise ValueError("previs authority boundary is invalid")
    if manifest.get("previs_scene_sha256") != scene.get("previs_scene_sha256"):
        raise ValueError("capture manifest does not bind the exact previs scene")
    if manifest.get("acceptance_status") != "candidate_only" or manifest.get("final_render_claimed") is not False:
        raise ValueError("previs capture manifest exceeds candidate authority")
    if validation.get("valid") is not True or validation.get("previs_scene_sha256") != scene.get("previs_scene_sha256"):
        raise ValueError("previs validation does not bind the exact scene")
    if validation.get("acceptance_status") != "candidate_only":
        raise ValueError("previs validation is not candidate-only")
    cameras = [dict(row) for row in scene.get("cameras") or [] if isinstance(row, dict)]
    if not cameras:
        raise ValueError("previs scene requires cameras")
    camera_by_id = {str(row.get("camera_id") or ""): row for row in cameras}
    if len(camera_by_id) != len(cameras) or not all(camera_by_id):
        raise ValueError("previs camera ids must be unique and non-empty")
    captures = [dict(row) for row in manifest.get("captures") or [] if isinstance(row, dict)]
    if not captures or validation.get("capture_count") != len(captures):
        raise ValueError("previs capture evidence is incomplete")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for capture in captures:
        capture_id = str(capture.get("capture_id") or "")
        if not capture_id or capture_id in seen:
            raise ValueError("previs capture ids must be unique and non-empty")
        seen.add(capture_id)
        camera = camera_by_id.get(str(capture.get("camera_id") or ""))
        if camera is None:
            raise ValueError(f"previs capture references unknown camera: {capture_id}")
        for key, expected in (
            ("fov", camera.get("fov")),
            ("position", camera.get("position")),
            ("target", camera.get("target")),
            ("width", (camera.get("frame") or {}).get("width")),
            ("height", (camera.get("frame") or {}).get("height")),
        ):
            if capture.get(key) != expected:
                raise ValueError(f"previs capture camera metadata drifted: {capture_id}/{key}")
        artifact = capture.get("artifact") if isinstance(capture.get("artifact"), dict) else {}
        artifact_path = validated_local_file(
            str(artifact.get("path") or ""), label=f"previs capture {capture_id}", allowed_roots=roots
        )
        _validate_reference(artifact_path, artifact, f"previs capture {capture_id}")
        normalized.append({**capture, "artifact": dict(artifact), "synthetic": True, "observed_video_fact": False})
    return cameras, normalized


def _representative_frames(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    visual = receipt.get("visual_verification") if isinstance(receipt.get("visual_verification"), dict) else {}
    return [dict(row) for row in visual.get("representative_frames") or [] if isinstance(row, dict)]


def _read_contract(
    source: str | Path,
    label: str,
    schema: str,
    roots: tuple[Path, ...],
) -> tuple[Path, dict[str, Any]]:
    path = validated_local_file(source, label=label, allowed_roots=roots)
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    if payload.get("schema") != schema:
        raise ValueError(f"unsupported {label} schema: {payload.get('schema')}")
    return path, payload


def _validate_reference(path: Path, reference: dict[str, Any], label: str) -> None:
    evidence = artifact_evidence(path)
    expected = {
        "bytes": int(evidence["bytes"]),
        "sha256": str(evidence["sha256"]),
    }
    if reference.get("bytes") != expected["bytes"] or reference.get("sha256") != expected["sha256"]:
        raise ValueError(f"{label} changed after the upstream contract was created")
    if reference.get("content_kind") == "json":
        value = read_json(path)
        if reference.get("canonical_sha256") != canonical_json_sha256(value):
            raise ValueError(f"{label} canonical JSON changed after contract creation")


def _import_contract_set(
    root: Path, contracts: dict[str, tuple[Path, dict[str, Any]]]
) -> dict[str, Path]:
    destination = root / "imports" / "video-creation-contracts"
    imported: dict[str, Path] = {}
    for role, (_source, payload) in contracts.items():
        digest = canonical_json_sha256(payload)
        output = destination / f"{role}-{digest[:12]}.json"
        write_json(output, payload)
        imported[role] = output
    return imported


def _payload_sha256(payload: dict[str, Any], hash_field: str) -> str:
    value = dict(payload)
    value.pop(hash_field, None)
    return canonical_json_sha256(value)


def _source_roots(root: Path, values: Iterable[str | Path] | None) -> tuple[Path, ...]:
    if values is None:
        return (root,)
    return normalise_allowed_roots(list(values), default_root=root)


def _bundle_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    if not (root / "manifest.json").is_file() or not (root / "timeline.json").is_file():
        raise FileNotFoundError(f"VKP Bundle requires manifest.json and timeline.json: {root}")
    return root


def _update_manifest(root: Path, values: dict[str, Any]) -> None:
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    manifest.update(values)
    write_json(manifest_path, manifest)


def _generation_markdown(result: dict[str, Any]) -> str:
    capability = result.get("capability_preflight") or {}
    visual = result.get("visual_verification") or {}
    technical = result.get("technical_verification") or {}
    lines = [
        "# Generation Contract Import",
        "",
        f"- Status: `{result.get('status', '')}`",
        f"- Task: `{result.get('task_id', '')}`",
        f"- Generator: `{result.get('generator', '')}`",
        f"- Capability: `{result.get('required_capability', '')}`",
        f"- Preflight ready: `{capability.get('ready', False)}` via `{capability.get('probe_method', '')}`",
        f"- Technical probe: `{technical.get('probe_method', '')}`",
        f"- Representative frames: `{len(visual.get('representative_frames') or [])}`",
        "",
        "> Candidate evidence only. No Timeline mutation, publication, implicit install, or silent fallback is authorized.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _previs_markdown(result: dict[str, Any]) -> str:
    scene = result.get("scene") or {}
    lines = [
        "# 3D Previs Candidate Evidence",
        "",
        f"- Status: `{result.get('status', '')}`",
        f"- Scene: `{scene.get('scene_id', '')}`",
        f"- Cameras: `{len(result.get('cameras') or [])}`",
        f"- Captures: `{len(result.get('captures') or [])}`",
        "",
        "> Synthetic candidate only. It is not an observed fact from the source video and cannot mutate Timeline without human confirmation.",
    ]
    return "\n".join(lines).rstrip() + "\n"
