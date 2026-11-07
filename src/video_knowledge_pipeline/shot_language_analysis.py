from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .file_hash import sha256_file
from .models import EvidenceSegment, now_iso
from .scene_taxonomy import normalize_taxonomy_value
from .shot_language_auto_scenes import (
    AUTO_SCENES_COMMIT,
    AutoScenesShotLanguageRuntime,
)
from .storage import bundle_write_lock, read_json, write_json
from .technical_shot_detection import load_verified_technical_shots
from .temporal_frame_groups import _sample_times
from .transcript import format_timestamp
from .video import extract_segment_frames


SCHEMA = "video_knowledge_pipeline.shot_facts.v1"
OUTPUT_PATH = "exports/shot-facts.json"
MARKDOWN_PATH = "exports/shot-facts.md"
_STATUSES = {"confirmed", "inferred", "unavailable"}
_SHOT_TYPES = {
    "\u5927\u7279\u5199": "extreme_close_up",
    "\u7279\u5199": "close_up",
    "\u4e2d\u666f": "medium",
    "\u5168\u666f": "wide",
    "\u8fdc\u666f": "wide",
}
_MOVEMENTS = {
    "\u56fa\u5b9a\u955c\u5934": "static",
    "\u6447\u955c\u5934": "pan_or_tilt",
    "\u79fb\u955c\u5934": "tracking",
    "\u8ddf\u955c\u5934": "tracking",
    "\u63a8\u955c\u5934": "zoom",
    "\u62c9\u955c\u5934": "zoom",
}


def run_shot_language_analysis(
    bundle_dir: str | Path,
    *,
    execution_location: str = "local",
    route_id: str = "",
    source_root: str | Path | None = None,
    shot_scale_model_path: str | Path | None = None,
    shot_type_confidence_threshold: float = 0.65,
    movement_confidence_threshold: float = 0.65,
    execute: bool = False,
    write: bool = True,
    _shot_type_analyzer: Callable[[str], dict[str, Any]] | None = None,
    _movement_analyzer: Callable[[list[str]], dict[str, Any]] | None = None,
    _frame_extractor: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Build evidence-bound shot facts without implicit model fallback."""

    root = Path(bundle_dir).expanduser().resolve()
    if str(execution_location or "").strip().lower() != "local":
        raise ValueError(
            "execution_location must be local; remote escalation requires "
            "the VKP provider gateway and consent"
        )
    shots, provenance = load_verified_technical_shots(root)
    if not shots:
        result = _blocked(root, provenance, route_id)
        if write:
            _write_result(root, result)
        return result

    timeline = _read_list(root / "timeline.json")
    review_corrections, review_provenance = _load_review_corrections(root, provenance)
    media, media_provenance = (
        _verified_detection_media(provenance)
        if execute
        else (None, {"status": "not_checked", "reason": "execute_false"})
    )
    runtime = None
    errors: dict[str, str] = {}
    if execute and (_shot_type_analyzer is None or _movement_analyzer is None):
        try:
            runtime = AutoScenesShotLanguageRuntime(
                source_root=source_root,
                model_path=shot_scale_model_path,
            )
        except Exception as exc:
            errors["runtime"] = f"{type(exc).__name__}: {exc}"

    rows_out: list[dict[str, Any]] = []
    escalations: list[dict[str, Any]] = []
    for shot in shots:
        shot_id = str(shot.get("shot_id") or "")
        start = float(shot.get("start") or 0.0)
        end = float(shot.get("end") or 0.0)
        rows = [row for row in timeline if _overlaps(row, start, end)]
        frames = _frame_paths(root, rows)
        if execute and write and media is not None:
            try:
                extracted = _extract_technical_shot_frames(
                    root,
                    media,
                    shot_id=shot_id,
                    start=start,
                    end=end,
                    extractor=_frame_extractor or extract_segment_frames,
                )
                frames = _unique_paths([*extracted, *frames])
            except Exception as exc:
                errors[f"{shot_id}:frame_extraction"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        evidence_ids = _evidence_ids(rows, frames)
        prepared = _prepare_frames(root, shot_id, frames, write=write)

        shot_result: dict[str, Any] = {}
        movement_result: dict[str, Any] = {}
        if execute and frames:
            shot_fn = _shot_type_analyzer
            move_fn = _movement_analyzer
            if runtime is not None:
                shot_fn = shot_fn or runtime.analyze_shot_type
                move_fn = move_fn or runtime.analyze_movement
            if shot_fn:
                try:
                    representative = (
                        prepared.get("representative_source_paths") or frames
                    )[0]
                    shot_result = shot_fn(str(representative))
                except Exception as exc:
                    errors[f"{shot_id}:shot_type"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
            if move_fn:
                try:
                    movement_result = move_fn(frames)
                except Exception as exc:
                    errors[f"{shot_id}:camera_movement"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

        fields = _build_fields(
            rows,
            frames,
            evidence_ids,
            shot_result,
            movement_result,
            shot_type_threshold=float(shot_type_confidence_threshold),
            movement_threshold=float(movement_confidence_threshold),
            execute=execute,
        )
        fields = _apply_review_corrections(
            fields,
            shot_id,
            review_corrections,
            review_provenance,
        )
        weak = [
            key
            for key in ("shot_type", "camera_movement", "composition", "lighting")
            if fields[key]["status"] != "confirmed"
        ]
        if weak:
            escalations.append(
                {
                    "shot_id": shot_id,
                    "fields": weak,
                    "route_id": str(route_id or ""),
                    "status": (
                        "ready_for_explicit_local_vlm"
                        if route_id
                        else "needs_explicit_local_vlm_route"
                    ),
                    "automatic_execution": False,
                    "image_paths": list(
                        prepared.get("prepared_image_paths") or frames
                    ),
                }
            )
        rows_out.append(
            {
                "shot_id": shot_id,
                "index": int(shot.get("index") or len(rows_out) + 1),
                "start": start,
                "end": end,
                "start_time": format_timestamp(start),
                "end_time": format_timestamp(end),
                "timeline_indexes": [
                    int(row["index"])
                    for row in rows
                    if row.get("index") not in (None, "")
                ],
                "frame_manifest": prepared.get("frame_manifest") or [],
                "frame_mapping": prepared.get("frame_mapping") or [],
                "contact_sheet_path": str(
                    prepared.get("contact_sheet_path") or ""
                ),
                "fields": fields,
                "candidate_only": True,
                "human_confirmed": False,
            }
        )
    if runtime is not None:
        runtime.close()

    counts = {
        status: sum(
            1
            for row in rows_out
            for field in row["fields"].values()
            if field["status"] == status
        )
        for status in sorted(_STATUSES)
    }
    result = {
        "schema": SCHEMA,
        "status": "completed" if counts["unavailable"] == 0 else "degraded",
        "ok": bool(rows_out),
        "bundle_dir": str(root),
        "execution_location": "local",
        "route_id": str(route_id or ""),
        "execute": bool(execute),
        "technical_shot_provenance": provenance,
        "media_provenance": media_provenance,
        "shot_review_provenance": review_provenance,
        "shot_count": len(rows_out),
        "shots": rows_out,
        "summary": {
            **{f"{key}_field_count": value for key, value in counts.items()},
            "local_vlm_escalation_count": len(escalations),
        },
        "local_vlm_escalations": escalations,
        "runtime_errors": errors,
        "source_reuse": {
            "project": "VideoClipper/Auto-Scenes-Extraction",
            "commit": AUTO_SCENES_COMMIT,
            "apis": [
                "DinoV2ShotClassifier.analyze",
                "OpticalFlowAnalyzer.analyze_frames",
                "Pillow temporal contact sheet",
            ],
            "copied_upstream_inference_code": False,
        },
        "operator_boundary": {
            "local_only": True,
            "gpu_required_for_shot_scale": True,
            "no_model_download": True,
            "no_network_call": True,
            "no_automatic_local_vlm_execution": True,
            "no_remote_fallback": True,
            "timeline_mutated": False,
            "technical_media_hash_verified": (
                media_provenance.get("status") == "verified"
            ),
        },
        "artifacts": {"json": OUTPUT_PATH, "markdown": MARKDOWN_PATH},
        "updated_at": now_iso(),
    }
    if write:
        _write_result(root, result)
    return result


def load_shot_facts(bundle_dir: str | Path) -> dict[str, Any]:
    path = Path(bundle_dir).expanduser().resolve() / OUTPUT_PATH
    if not path.is_file():
        return {}
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        return {}
    provenance = payload.get("technical_shot_provenance")
    if isinstance(provenance, dict):
        raw = str(provenance.get("path") or "").strip()
        expected = str(provenance.get("sha256") or "").strip().lower()
        if raw and expected:
            source = Path(raw).expanduser().resolve()
            if not source.is_file() or sha256_file(source).lower() != expected:
                raise RuntimeError(
                    "shot_facts.v1 is stale because its technical-shot source changed"
                )
    return payload


def _load_review_corrections(
    root: Path,
    technical_provenance: dict[str, Any],
) -> tuple[dict[tuple[str, str], Any], dict[str, Any]]:
    applied_path = root / "exports" / "shot-review-applied.json"
    reviewed_path = root / "exports" / "technical-shot-boundaries.reviewed.json"
    if not applied_path.is_file() or not reviewed_path.is_file():
        return {}, {"status": "not_applied"}
    applied = read_json(applied_path)
    reviewed = read_json(reviewed_path)
    if not isinstance(applied, dict) or not isinstance(reviewed, dict):
        return {}, {"status": "invalid"}
    review_id = str(applied.get("review_id") or "")
    active_source = Path(str(technical_provenance.get("path") or "")).resolve()
    if (
        not review_id
        or review_id != str(reviewed.get("review_id") or "")
        or active_source != reviewed_path.resolve()
    ):
        return {}, {"status": "not_active_for_current_technical_shots"}
    result: dict[tuple[str, str], Any] = {}
    for row in applied.get("field_corrections") or []:
        if not isinstance(row, dict):
            continue
        shot_id = str(row.get("shot_id") or "")
        field = str(row.get("field") or "")
        if shot_id and field:
            result[(shot_id, field)] = row.get("value")
    return result, {
        "status": "active",
        "review_id": review_id,
        "path": str(applied_path),
        "sha256": sha256_file(applied_path),
        "field_correction_count": len(result),
    }


def _apply_review_corrections(
    fields: dict[str, dict[str, Any]],
    shot_id: str,
    corrections: dict[tuple[str, str], Any],
    provenance: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    review_id = str(provenance.get("review_id") or "")
    for field in ("shot_type", "camera_movement", "composition", "lighting"):
        key = (shot_id, field)
        if key not in corrections:
            continue
        fields[field] = _field(
            corrections[key],
            "confirmed",
            1.0,
            [f"human-review:{review_id}"],
            "human_review.shot_review_notes.v1",
            [],
        )
    return fields


def _build_fields(
    rows: list[dict[str, Any]],
    frames: list[str],
    evidence_ids: list[str],
    shot_result: dict[str, Any],
    movement_result: dict[str, Any],
    *,
    shot_type_threshold: float,
    movement_threshold: float,
    execute: bool,
) -> dict[str, dict[str, Any]]:
    raw_shot = str(
        shot_result.get("\u666f\u522b")
        or shot_result.get("shot_type")
        or ""
    )
    shot_value = _SHOT_TYPES.get(
        raw_shot,
        normalize_taxonomy_value("shot_type", raw_shot),
    )
    raw_move = str(
        movement_result.get("\u955c\u5934\u8fd0\u52a8")
        or movement_result.get("camera_movement")
        or ""
    )
    movement_value = _MOVEMENTS.get(
        raw_move,
        normalize_taxonomy_value("camera_movement", raw_move),
    )
    transcript = _join(
        _text(row, "corrected_transcript", "transcript", "asr_text", "text")
        for row in rows
    )
    screen_text = _join(
        _text(row, "corrected_visual_text", "visual_text", "ocr_text")
        for row in rows
    )
    actions = _temporal_actions(rows)
    composition = _visual_dict(rows, "composition")
    lighting = _visual_text(rows, "lighting", "light")
    music = _join(
        str(row.get("music") or row.get("bgm") or "")
        for row in rows
    )
    return {
        "shot_type": _model_field(
            shot_value,
            shot_result.get("confidence"),
            shot_type_threshold,
            evidence_ids,
            "auto_scenes.dinov2_shot_classifier",
            execute,
        ),
        "camera_movement": _model_field(
            movement_value,
            movement_result.get("confidence"),
            movement_threshold,
            evidence_ids,
            "auto_scenes.optical_flow_analyzer",
            execute,
        ),
        "composition": _direct(
            composition,
            evidence_ids,
            "timeline.visual_understanding.composition",
            inferred=True,
        ),
        "lighting": _direct(
            lighting,
            evidence_ids,
            "timeline.visual_understanding.lighting",
            inferred=True,
        ),
        "subject_action": _direct(
            actions,
            evidence_ids,
            "timeline.temporal_visual_understanding",
            inferred=True,
        ),
        "dialogue_or_narration": _direct(
            transcript,
            evidence_ids,
            "timeline.transcript",
        ),
        "screen_text": _direct(
            screen_text,
            evidence_ids,
            "timeline.visual_text",
        ),
        "audio": _direct(
            {
                "dialogue_present": bool(transcript),
                "music": music or "unknown",
                "sound_effects": "unknown",
            }
            if transcript or music
            else None,
            evidence_ids,
            "timeline.audio_and_transcript",
        ),
        "reference_frames": _direct(
            frames,
            evidence_ids,
            "bundle.frame_paths",
        ),
    }


def _model_field(
    value: str,
    raw_confidence: Any,
    threshold: float,
    evidence_ids: list[str],
    source: str,
    executed: bool,
) -> dict[str, Any]:
    confidence = _confidence(raw_confidence)
    if not executed or value in {"", "unknown"}:
        return _field(
            None,
            "unavailable",
            0.0,
            [],
            source,
            ["local model result"],
        )
    gaps = (
        []
        if confidence >= threshold
        else [f"confidence {confidence:.4f} below threshold {threshold:.4f}"]
    )
    return _field(
        value,
        "inferred",
        confidence,
        evidence_ids,
        source,
        gaps,
    )


def _direct(
    value: Any,
    evidence_ids: list[str],
    source: str,
    *,
    inferred: bool = False,
) -> dict[str, Any]:
    if value in (None, "", [], {}):
        return _field(
            None,
            "unavailable",
            0.0,
            [],
            source,
            [f"{source} evidence"],
        )
    return _field(
        value,
        "inferred" if inferred else "confirmed",
        1.0,
        evidence_ids,
        source,
        [],
    )


def _field(
    value: Any,
    status: str,
    confidence: float,
    evidence_ids: list[str],
    source: str,
    missing_evidence: list[str],
) -> dict[str, Any]:
    if status not in _STATUSES:
        raise ValueError(f"invalid shot-fact status: {status}")
    if status == "unavailable":
        value, evidence_ids = None, []
    return {
        "value": value,
        "status": status,
        "confidence": round(max(0.0, min(float(confidence), 1.0)), 4),
        "evidence_ids": sorted(set(evidence_ids)),
        "source": source,
        "missing_evidence": list(missing_evidence),
    }


def _verified_detection_media(
    technical_provenance: dict[str, Any],
) -> tuple[Path | None, dict[str, Any]]:
    raw_artifact = str(technical_provenance.get("path") or "").strip()
    if not raw_artifact:
        return None, {"status": "not_bound"}
    artifact = Path(raw_artifact).expanduser().resolve()
    if not artifact.is_file():
        return None, {"status": "technical_artifact_missing", "path": str(artifact)}
    payload = read_json(artifact)
    media = payload.get("media") if isinstance(payload, dict) else None
    if not isinstance(media, dict):
        return None, {"status": "not_bound"}
    raw_media = str(media.get("path") or "").strip()
    expected = str(media.get("sha256") or "").strip().lower()
    if not raw_media or not expected:
        return None, {"status": "not_bound"}
    try:
        path = Path(raw_media).expanduser().resolve()
        if not path.is_file():
            return None, {"status": "media_missing", "path": str(path)}
        actual = sha256_file(path).lower()
    except OSError as exc:
        return None, {
            "status": "media_io_error",
            "path": raw_media,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if actual != expected:
        return None, {
            "status": "media_hash_mismatch",
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
        }
    return path, {
        "status": "verified",
        "path": str(path),
        "sha256": actual,
        "source": "technical_shot_boundaries.v1.media",
    }


def _extract_technical_shot_frames(
    root: Path,
    media: Path,
    *,
    shot_id: str,
    start: float,
    end: float,
    extractor: Callable[..., None],
) -> list[str]:
    duration = max(0.0, end - start)
    inset = min(0.12, duration / 4.0)
    sample_start = min(end, start + inset)
    sample_end = max(sample_start, end - inset)
    times = _sample_times(sample_start, sample_end, 3)
    segments = [
        EvidenceSegment(
            id=f"{shot_id}-representative-{index:02d}",
            video_id="technical-shot-media",
            start=start,
            end=end,
            midpoint=seconds,
            signals=["technical_shot_representative"],
        )
        for index, seconds in enumerate(times, start=1)
    ]
    extractor(
        media,
        root / "exports" / "shot-language" / shot_id / "source-frames",
        segments,
    )
    return _unique_paths(
        [path for segment in segments for path in segment.frame_paths]
    )


def _unique_paths(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _prepare_frames(
    root: Path,
    shot_id: str,
    frames: list[str],
    *,
    write: bool,
) -> dict[str, Any]:
    try:
        from .temporal_frame_preprocess import (
            build_temporal_frame_manifest,
            prepare_temporal_image_probe,
        )
    except ImportError:
        build_temporal_frame_manifest = _basic_frame_manifest
        prepare_temporal_image_probe = None
    manifest = build_temporal_frame_manifest(frames)
    if not frames or not write or prepare_temporal_image_probe is None:
        return {
            "frame_manifest": manifest,
            "frame_mapping": [
                {
                    "frame_id": row["frame_id"],
                    "source_index": row["source_index"],
                    "representative_frame_id": row["frame_id"],
                }
                for row in manifest
            ],
            "representative_source_paths": frames[:3],
            "prepared_image_paths": frames[:3],
            "contact_sheet_path": "",
        }
    prepared = prepare_temporal_image_probe(
        frames,
        output_dir=root / "exports" / "shot-language" / shot_id,
        max_edge=1280,
        jpeg_quality=80,
        role=f"shot_language_{shot_id}",
        use_contact_sheet=True,
        representative_limit=3,
    )
    reps = {
        str(row.get("representative_frame_id") or "")
        for row in prepared.get("frame_mapping") or []
    }
    by_id = {
        str(row.get("frame_id") or ""): str(row.get("source_path") or "")
        for row in prepared.get("frame_manifest") or []
    }
    prepared["representative_source_paths"] = [
        by_id[value] for value in sorted(reps) if by_id.get(value)
    ]
    return prepared


def _basic_frame_manifest(source_paths: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "frame_id": f"F{index:02d}",
            "source_index": index,
            "filename": Path(value).name,
            "source_path": str(value),
            "timestamp_ms": None,
            "timestamp": "",
            "bytes": Path(value).stat().st_size if Path(value).is_file() else 0,
        }
        for index, value in enumerate(source_paths, start=1)
    ]

def _frame_paths(root: Path, rows: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for row in rows:
        values = _path_values(row.get("frame_paths"))
        values.extend(_path_values(row.get("temporal_frame_paths")))
        temporal_group = row.get("temporal_frame_group")
        if isinstance(temporal_group, dict):
            values.extend(_path_values(temporal_group.get("frame_paths")))
        for asset in row.get("assets") or []:
            if isinstance(asset, dict) and asset.get("path"):
                values.append(str(asset["path"]))
        for value in values:
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = root / path
            try:
                path = path.resolve()
                path.relative_to(root)
            except (OSError, ValueError):
                continue
            if path.is_file() and str(path) not in result:
                result.append(str(path))
    return result


def _path_values(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return []


def _evidence_ids(
    rows: list[dict[str, Any]],
    frames: list[str],
) -> list[str]:
    ids = [
        f"timeline:{int(row['index'])}"
        for row in rows
        if row.get("index") not in (None, "")
    ]
    ids.extend(
        f"frame:{sha256_file(Path(path))[:16]}"
        for path in frames
        if Path(path).is_file()
    )
    return ids


def _overlaps(row: dict[str, Any], start: float, end: float) -> bool:
    try:
        row_start = float(row.get("start") or 0.0)
        row_end = float(row.get("end") or row_start)
    except (TypeError, ValueError):
        return False
    return row_end > start and row_start < end


def _text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if row.get(key) not in (None, ""):
            return str(row[key]).strip()
    return ""


def _join(values: Any) -> str:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return "\n".join(result)


def _temporal_actions(rows: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for row in rows:
        value = row.get("temporal_visual_understanding")
        if not isinstance(value, dict):
            continue
        for key in ("event_sequence", "summary", "changes"):
            raw = value.get(key)
            items = raw if isinstance(raw, list) else [raw]
            for item in items:
                text = str(item or "").strip()
                if text and text not in result:
                    result.append(text)
    return result


def _visual_dict(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    for row in rows:
        value = row.get("visual_understanding")
        if isinstance(value, dict) and isinstance(value.get(key), dict):
            return dict(value[key])
    return {}


def _visual_text(
    rows: list[dict[str, Any]],
    *keys: str,
) -> str:
    for row in rows:
        value = row.get("visual_understanding")
        if isinstance(value, dict):
            for key in keys:
                text = str(value.get(key) or "").strip()
                if text:
                    return text
    return ""


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _read_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    value = read_json(path)
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _blocked(
    root: Path,
    provenance: dict[str, Any],
    route_id: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "blocked_missing_technical_shots",
        "ok": False,
        "bundle_dir": str(root),
        "execution_location": "local",
        "route_id": str(route_id or ""),
        "shot_count": 0,
        "shots": [],
        "technical_shot_provenance": provenance,
        "local_vlm_escalations": [],
        "operator_boundary": {
            "no_chapter_or_timeline_range_fallback": True,
            "no_remote_fallback": True,
            "timeline_mutated": False,
        },
        "artifacts": {"json": OUTPUT_PATH, "markdown": MARKDOWN_PATH},
        "updated_at": now_iso(),
    }


def _write_result(root: Path, result: dict[str, Any]) -> None:
    manifest_path = root / "manifest.json"
    with bundle_write_lock(
        root,
        operation="shot_language_analysis",
        timeout_seconds=1.0,
    ):
        write_json(root / OUTPUT_PATH, result)
        (root / MARKDOWN_PATH).write_text(
            _render_markdown(result),
            encoding="utf-8",
        )
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        if isinstance(manifest, dict):
            manifest["shot_facts_json"] = OUTPUT_PATH
            manifest["shot_facts_markdown"] = MARKDOWN_PATH
            manifest["shot_facts_summary"] = {
                "status": result["status"],
                "shot_count": result["shot_count"],
                "updated_at": result["updated_at"],
            }
            write_json(manifest_path, manifest)


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Shot Facts",
        "",
        f"- Status: {result.get('status')}",
        f"- Shot count: {result.get('shot_count', 0)}",
        "- Candidate only: true",
        "- Automatic remote fallback: false",
        "",
    ]
    for row in result.get("shots") or []:
        lines.append(
            f"- {row['shot_id']} {row['start_time']} - {row['end_time']}"
        )
        for key, field in row["fields"].items():
            value = field.get("value")
            lines.append(
                f"  - {key}: "
                f"{value if value is not None else 'unavailable'} "
                f"({field['status']}, {field['confidence']})"
            )
    return "\n".join(lines).rstrip() + "\n"
