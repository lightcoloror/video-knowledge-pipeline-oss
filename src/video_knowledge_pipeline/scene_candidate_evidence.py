from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file as _sha256_file
from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .scene_taxonomy import explain_quality_for_shot_type, normalize_scene_tags
from .shot_boundary_saved_predictions import normalize_saved_shot_boundaries
from .storage import bundle_write_lock, read_json, write_json
from .transcript import format_timestamp


SCHEMA = "video_knowledge_pipeline.scene_candidate_evidence.v1"
CANDIDATE_SCHEMA = "video_knowledge_pipeline.candidate_scene_boundary.v1"
SHOT_CANDIDATE_SCHEMA = "video_knowledge_pipeline.candidate_shot_boundary.v1"
REVIEW_SCHEMA = "video_knowledge_pipeline.scene_candidate_review_notes.v1"
UPSTREAM_PROJECT = "TuanAMV/Auto-scenes-extraction"
UPSTREAM_COMMIT = "2c34db3520e1319292bb456a0e610a0ef195e78b"
ADAPTER_VERSION = "v1"


def build_scene_candidate_evidence(
    bundle_dir: str | Path,
    candidates_json: str | Path,
    *,
    model_id: str,
    model_commit: str = "unversioned",
    language: str = "und",
    taxonomy_prompt: str | Path,
    cache_format_version: str = "scene-candidate-cache.v1",
    source_format: str = "generic",
    frame_rate: float | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Import scene-boundary candidates without changing the VKP Timeline."""

    root = Path(bundle_dir).expanduser().resolve()
    source_path = Path(candidates_json).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"scene candidate input not found: {source_path}")
    if not str(model_id or "").strip():
        raise ValueError("model_id is required for scene candidate provenance")
    if not str(language or "").strip():
        raise ValueError("language is required for scene candidate provenance")
    taxonomy_hash, taxonomy_source = _taxonomy_prompt_identity(taxonomy_prompt)
    payload = read_json(source_path)
    format_key = str(source_format or "generic").strip().lower()
    if format_key == "generic":
        source_rows = _candidate_rows(payload)
        format_metadata = {
            "source_format": "generic",
            "boundary_kind": "scene",
            "frame_rate": None,
            "prediction_threshold": None,
            "upstream_project": UPSTREAM_PROJECT,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_api": "candidate_boundary_rows",
            "source_scene_count": None,
            "boundary_count": len(source_rows),
            "model_execution_performed": False,
        }
    else:
        source_rows, format_metadata = normalize_saved_shot_boundaries(
            payload,
            source_format=format_key,
            frame_rate=frame_rate,
        )
    if not source_rows:
        raise ValueError("scene candidate input contains no boundary rows")

    manifest_path = root / "manifest.json"
    manifest = _object(read_json(manifest_path), "bundle manifest") if manifest_path.exists() else {}
    duration = float(manifest.get("duration_seconds") or 0.0)
    source_sha256 = _sha256_file(source_path)
    timeline_path = root / "timeline.json"
    timeline_sha256 = _sha256_file(timeline_path) if timeline_path.is_file() else ""
    provenance = {
        "adapter": "vkp_scene_candidate_evidence",
        "adapter_version": ADAPTER_VERSION,
        "upstream_reference": {
            "project": format_metadata["upstream_project"],
            "commit": format_metadata["upstream_commit"],
            "api": format_metadata["upstream_api"],
            "reuse_mode": (
                "saved_prediction_contract"
                if format_metadata["source_format"] != "generic"
                else "independent_adaptation"
            ),
        },
        "source_artifact": str(source_path),
        "source_artifact_sha256": source_sha256,
        "model_id": str(model_id).strip(),
        "model_commit": str(model_commit or "unversioned").strip(),
        "language": str(language).strip(),
        "taxonomy_prompt_sha256": taxonomy_hash,
        "taxonomy_prompt_source": taxonomy_source,
        "record_count": len(source_rows),
        "cache_format_version": str(cache_format_version or "scene-candidate-cache.v1"),
        "source_format": format_metadata["source_format"],
        "boundary_kind": format_metadata["boundary_kind"],
        "frame_rate": format_metadata["frame_rate"],
        "prediction_threshold": format_metadata["prediction_threshold"],
        "source_scene_count": format_metadata["source_scene_count"],
        "model_execution_performed": format_metadata["model_execution_performed"],
    }
    provenance["cache_identity_sha256"] = _sha256_json(
        {
            "model_id": provenance["model_id"],
            "model_commit": provenance["model_commit"],
            "language": provenance["language"],
            "taxonomy_prompt_sha256": taxonomy_hash,
            "record_count": len(source_rows),
            "cache_format_version": provenance["cache_format_version"],
            "source_format": provenance["source_format"],
            "boundary_kind": provenance["boundary_kind"],
            "frame_rate": provenance["frame_rate"],
            "prediction_threshold": provenance["prediction_threshold"],
        }
    )
    candidates = [
        _normalize_candidate(
            row,
            position=position,
            duration=duration,
            provenance=provenance,
        )
        for position, row in enumerate(source_rows, start=1)
    ]
    result = {
        "schema": SCHEMA,
        "status": "needs_human_review",
        "ok": True,
        "bundle_dir": str(root),
        "candidate_count": len(candidates),
        "candidate_kind": provenance["boundary_kind"],
        "candidates": candidates,
        "provenance": provenance,
        "sample_gate": {
            "status": "needs_human_review",
            "sample_size": min(5, len(candidates)),
            "full_batch_or_export_allowed": False,
            "review_artifact": "scene-candidate-review.todo.json",
            "minimal_slice_repair_only": True,
        },
        "timeline_invariant": {
            "timeline_path": str(timeline_path) if timeline_path.exists() else "",
            "timeline_sha256": timeline_sha256,
            "timeline_modified": False,
        },
        "operator_boundary": {
            "candidate_evidence_only": True,
            "saved_prediction_model_executed_by_vkp": False,
            "timeline_overwrite_allowed": False,
            "automatic_adjacent_merge_allowed": False,
            "merge_requires_asr_ocr_visual_provenance": True,
            "merge_requires_human_confirmation": True,
            "remote_calls_made": 0,
            "automatic_local_remote_fallback": False,
        },
        "artifacts": {
            "json": "exports/scene-candidate-evidence.json",
            "markdown": "exports/scene-candidate-evidence.md",
            "review_todo": "scene-candidate-review.todo.json",
        },
        "updated_at": now_iso(),
    }
    if write:
        _write_result(root, manifest, result)
        result["run_artifact"] = register_bundle_run(
            root,
            run_type="scene_candidate_review",
            run_id="scene-candidate-review",
            status="needs_input",
            title="Scene candidate first-sample review",
            summary=f"Imported {len(candidates)} candidate boundaries; export remains blocked.",
            inputs={
                "candidates_json": str(source_path),
                "source_artifact_sha256": source_sha256,
                "cache_identity_sha256": provenance["cache_identity_sha256"],
            },
            parameters={
                "model_id": provenance["model_id"],
                "model_commit": provenance["model_commit"],
                "language": provenance["language"],
                "taxonomy_prompt_sha256": taxonomy_hash,
                "source_format": provenance["source_format"],
                "frame_rate": provenance["frame_rate"],
                "prediction_threshold": provenance["prediction_threshold"],
            },
            artifacts=[
                {"key": "scene_candidate_evidence", "path": "exports/scene-candidate-evidence.json"},
                {"key": "scene_candidate_review_todo", "path": "scene-candidate-review.todo.json"},
            ],
            failed_items=[
                {
                    "id": candidate["candidate_id"],
                    "reason": "pending_first_sample_review",
                    "detail": "Candidate cannot enter export until a human confirms it.",
                }
                for candidate in candidates[: min(5, len(candidates))]
            ],
            retry_command=(
                ".\\scripts\\video-knowledge.ps1 scene-candidate-evidence "
                f"'{root}' '{source_path}' --model-id '{provenance['model_id']}' "
                f"--model-commit '{provenance['model_commit']}' --language '{provenance['language']}' "
                f"--taxonomy-prompt '{taxonomy_source}' "
                f"--source-format '{provenance['source_format']}'"
                + (
                    f" --frame-rate {provenance['frame_rate']}"
                    if provenance["frame_rate"] is not None
                    else ""
                )
            ),
            next_actions=[
                "Review scene-candidate-review.todo.json.",
                "Keep accepted boundaries as candidate evidence; do not overwrite Timeline.",
            ],
            operator_boundary=result["operator_boundary"],
            resource_requirements={"cpu": 1, "gpu": 0, "network": 0},
            write=True,
        )
    return result


def _normalize_candidate(
    row: dict[str, Any],
    *,
    position: int,
    duration: float,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    seconds = _candidate_seconds(row)
    if seconds < 0:
        raise ValueError(f"scene candidate time must be non-negative: row {position}")
    if duration > 0 and seconds > duration:
        raise ValueError(f"scene candidate exceeds bundle duration: row {position}")
    score = row.get("score", row.get("similarity"))
    normalized_score = None if score in (None, "") else float(score)
    if normalized_score is not None and not 0 <= normalized_score <= 1:
        raise ValueError(f"scene candidate score must be between 0 and 1: row {position}")
    raw_tags = row.get("taxonomy") if isinstance(row.get("taxonomy"), dict) else row.get("tags")
    taxonomy = normalize_scene_tags(raw_tags if isinstance(raw_tags, dict) else {})
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    shot_value = taxonomy["raw"].get("shot_type")
    quality_explanation = (
        explain_quality_for_shot_type(quality, shot_value)
        if quality and shot_value not in (None, "")
        else {}
    )
    evidence_refs = {
        key: [str(value) for value in row.get(f"{key}_evidence_ids") or []]
        for key in ("asr", "ocr", "visual")
    }
    seed = {
        "seconds": round(seconds, 6),
        "source_artifact_sha256": provenance["source_artifact_sha256"],
        "cache_identity_sha256": provenance["cache_identity_sha256"],
        "position": position,
    }
    return {
        "schema": (
            SHOT_CANDIDATE_SCHEMA
            if provenance["boundary_kind"] == "shot"
            else CANDIDATE_SCHEMA
        ),
        "candidate_id": (
            f"{'shot' if provenance['boundary_kind'] == 'shot' else 'scene'}"
            f"-boundary-{_sha256_json(seed)[:16]}"
        ),
        "index": position,
        "seconds": round(seconds, 6),
        "time": format_timestamp(seconds),
        "score": normalized_score,
        "reason": str(row.get("reason") or "external_scene_boundary_candidate"),
        "source_coordinates": {
            key: row[key]
            for key in ("source_scene_index", "start_frame", "end_frame")
            if key in row
        },
        "taxonomy": taxonomy,
        "shot_quality_explanation": quality_explanation,
        "evidence_refs": evidence_refs,
        "candidate_only": True,
        "timeline_mutated": False,
        "export_eligible": False,
        "requires_human_confirmation": True,
    }


def _candidate_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = (
            payload.get("candidate_scene_boundaries")
            or payload.get("boundaries")
            or payload.get("candidates")
            or []
        )
    else:
        rows = []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _candidate_seconds(row: dict[str, Any]) -> float:
    for key in ("seconds", "time_seconds", "start_seconds", "start"):
        if row.get(key) not in (None, ""):
            return float(row[key])
    raise ValueError("scene candidate row is missing seconds")


def _taxonomy_prompt_identity(value: str | Path) -> tuple[str, str]:
    text = str(value or "").strip()
    if not text:
        raise ValueError("taxonomy_prompt is required for cache provenance")
    path = Path(text).expanduser()
    if path.is_file():
        resolved = path.resolve()
        return _sha256_file(resolved), str(resolved)
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), "inline_text"


def _write_result(root: Path, manifest: dict[str, Any], result: dict[str, Any]) -> None:
    exports = root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    json_path = exports / "scene-candidate-evidence.json"
    markdown_path = exports / "scene-candidate-evidence.md"
    review_path = root / "scene-candidate-review.todo.json"
    result_sha256 = _sha256_json(result)
    review = {
        "schema": REVIEW_SCHEMA,
        "scene_candidate_pack_sha256": result_sha256,
        "status": "draft",
        "reviews": [
            {
                "candidate_id": row["candidate_id"],
                "decision": "pending",
                "notes": "",
            }
            for row in result["candidates"][: result["sample_gate"]["sample_size"]]
        ],
        "operator_boundary": {
            "draft_does_not_authorize_export": True,
            "human_confirmation_required": True,
        },
    }
    with bundle_write_lock(root, operation="scene_candidate_evidence", timeout_seconds=1.0):
        write_json(json_path, result)
        markdown_path.write_text(_render_markdown(result), encoding="utf-8")
        write_json(review_path, review)
        manifest["scene_candidate_evidence_json"] = "exports/scene-candidate-evidence.json"
        manifest["scene_candidate_evidence_markdown"] = "exports/scene-candidate-evidence.md"
        manifest["scene_candidate_review_todo"] = "scene-candidate-review.todo.json"
        manifest["scene_candidate_status"] = result["status"]
        write_json(root / "manifest.json", manifest)


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Scene Candidate Evidence",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Candidates: `{result.get('candidate_count')}`",
        f"- Model: `{result['provenance'].get('model_id')}`",
        f"- Model commit: `{result['provenance'].get('model_commit')}`",
        f"- Language: `{result['provenance'].get('language')}`",
        f"- Cache identity: `{result['provenance'].get('cache_identity_sha256')}`",
        "- Timeline overwrite: `blocked`",
        "- Adjacent merge: `blocked until ASR/OCR/visual provenance and human confirmation`",
        "",
        "| Candidate | Time | Score | Review |",
        "| --- | --- | ---: | --- |",
    ]
    for row in result.get("candidates") or []:
        lines.append(
            f"| `{row.get('candidate_id')}` | `{row.get('time')}` | {row.get('score')} | pending |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def _sha256_json(value: Any) -> str:
    return canonical_json_sha256(value)
