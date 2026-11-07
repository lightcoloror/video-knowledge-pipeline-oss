from __future__ import annotations

import math
import subprocess
from array import array
from pathlib import Path
from typing import Any, Iterable

from .powershell import quote_powershell_literal as _ps_quote
from .artifact_freshness import build_dependency_snapshot
from .models import now_iso
from .review_attestation import validate_review_attestation
from .run_artifact_registry import register_bundle_run
from .storage import read_json, write_json
from .transcript import format_timestamp

SCHEMA = "video_knowledge_pipeline.video_edit_review_pack.v1"
BOUNDARY_SCHEMA = "video_knowledge_pipeline.video_edit_boundary_refinement.v1"
VALIDATION_SCHEMA = "video_knowledge_pipeline.video_edit_artifact_validation.v1"
PREFERENCE_SCHEMA = "video_knowledge_pipeline.video_edit_preference_evidence.v1"

PAD = 0.03
PAD_ONSET = 0.05
GUARD = 0.12
GUARD_MIN = 0.02
TOL = 0.001
SEGMENT_TOL = 0.02


def build_video_edit_review_pack(
    bundle_dir: str | Path,
    *,
    decisions_json: str | Path | None = None,
    tokens_json: str | Path | None = None,
    silence_json: str | Path | None = None,
    delete_segments_json: str | Path | None = None,
    cut_segments_json: str | Path | None = None,
    ai_baseline_json: str | Path | None = None,
    media_path: str | Path | None = None,
    reclaim_silence: bool = False,
    human_confirmed_diff: bool = False,
    review_attestation_path: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Build a local-only handoff from VKP evidence to a video editing lane.

    This function never edits media.  It emits candidate storyboard scenes,
    boundary-refined delete decisions, validation evidence, and optional
    human-confirmed preference evidence for an existing single FFmpeg/render
    pipeline to consume after operator review.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    if not timeline_path.is_file():
        raise FileNotFoundError(f"timeline.json not found: {timeline_path}")
    manifest = _read_object(manifest_path)
    timeline = read_json(timeline_path)
    if not isinstance(timeline, list):
        raise ValueError("timeline.json must be a JSON array")

    resolved = {
        "decisions": _first_existing(
            root,
            decisions_json,
            ["video-edit/edit.decisions.json", "edit.decisions.json"],
        ),
        "tokens": _first_existing(
            root,
            tokens_json,
            ["video-edit/transcript.tokens.json", "transcript.tokens.json"],
        ),
        "silence": _first_existing(
            root,
            silence_json,
            [
                "video-edit/silence.reclaimed.json",
                "silence.reclaimed.json",
                "video-edit/silence.json",
                "silence.json",
            ],
        ),
        "delete_segments": _first_existing(
            root,
            delete_segments_json,
            ["video-edit/delete_segments.json", "delete_segments.json"],
        ),
        "cut_segments": _first_existing(
            root,
            cut_segments_json,
            ["video-edit/cut.segments.json", "cut.segments.json"],
        ),
        "ai_baseline": _first_existing(
            root, ai_baseline_json, ["video-edit/ai_baseline.json", "ai_baseline.json"]
        ),
    }
    tokens_payload = _read_list(resolved["tokens"])
    tokens = spoken_tokens(tokens_payload)
    silences = build_token_silences(tokens_payload)
    if resolved["silence"]:
        silences = merge_silences(silences, _read_list(resolved["silence"]))

    reclaim_result = {
        "requested": bool(reclaim_silence),
        "executed": False,
        "status": "not_requested",
        "reclaimed_count": 0,
        "merged": silences,
    }
    if reclaim_silence:
        if not media_path:
            reclaim_result["status"] = "missing_media_path"
        elif not tokens:
            reclaim_result["status"] = "missing_tokens"
        else:
            media = _resolve_input_path(root, media_path)
            if not media or not media.is_file():
                reclaim_result["status"] = "media_not_found"
            else:
                reclaim_result = reclaim_silence_from_media(
                    media, tokens, base_silence=silences
                )
                silences = reclaim_result["merged"]

    decisions = _read_list(resolved["decisions"])
    decisions_confirmed = _edit_decisions_confirmed(decisions)
    boundary = (
        refine_edit_boundaries(decisions, tokens=tokens, silences=silences)
        if decisions
        else _empty_boundary()
    )
    delete_segments = _read_list(resolved["delete_segments"])
    cut_segments = _read_list(resolved["cut_segments"])
    duration = _timeline_duration(timeline)
    validation = validate_edit_artifacts(
        decisions,
        delete_segments,
        cut_segments=cut_segments,
        duration=duration,
        tokens=tokens,
        silences=silences,
    )

    chapters = _read_chapters(root, manifest)
    arbitration = _read_arbitration(root, manifest)
    scenes = build_storyboard_candidates(
        timeline, chapters=chapters, arbitration=arbitration
    )

    baseline = _read_list(resolved["ai_baseline"])
    preference = build_preference_evidence(
        baseline,
        decisions,
        tokens=tokens,
        video_id=str(manifest.get("title") or root.name),
        human_confirmed=human_confirmed_diff,
    )
    dependency_inputs: list[dict[str, Any]] = [
        {"role": "timeline", "path": timeline_path},
    ]
    for role, path in resolved.items():
        if path and path.is_file():
            dependency_inputs.append({"role": role, "path": path})
    dependency_snapshot = build_dependency_snapshot(
        root,
        subject="video-edit-handoff",
        inputs=dependency_inputs,
        source_run_id="video-edit-review-pack",
        producer_schema=SCHEMA,
    )
    review_attestation = validate_review_attestation(
        root,
        target="video-edit-handoff",
        attestation_path=review_attestation_path,
        expected_snapshot=dependency_snapshot,
    )
    blockers = _blockers(
        scenes=scenes,
        decisions_present=bool(decisions),
        decisions_confirmed=decisions_confirmed,
        delete_segments_present=bool(delete_segments),
        validation=validation,
        reclaim_result=reclaim_result,
        review_attestation=review_attestation,
    )
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "title": str(manifest.get("title") or root.name),
        "generated_at": now_iso(),
        "storyboard_scene_count": len(scenes),
        "storyboard_scenes": scenes,
        "boundary_refinement": boundary,
        "artifact_validation": validation,
        "silence_reclamation": reclaim_result,
        "preference_evidence": preference,
        "dependency_snapshot": dependency_snapshot,
        "review_attestation": review_attestation,
        "edit_decisions_human_confirmed": decisions_confirmed,
        "storyboard_review_required": any(
            not bool(scene.get("confirmed")) for scene in scenes
        ),
        "blockers": blockers,
        "ready_for_single_ffmpeg": not blockers
        and bool(decisions)
        and bool(delete_segments),
        "operator_boundary": {
            "local_only": True,
            "candidate_only": True,
            "media_edited": False,
            "cloud_call": False,
            "human_confirmation_required": True,
            "single_existing_render_pipeline_required": True,
            "preference_auto_write": False,
        },
        "inputs": {key: str(value) if value else "" for key, value in resolved.items()},
        "paths": {
            "json": str(root / "exports" / "video-edit-review-pack.json"),
            "markdown": str(root / "exports" / "video-edit-review-pack.md"),
            "storyboard": str(root / "exports" / "storyboard.candidates.json"),
            "refined_decisions": str(root / "exports" / "edit.decisions.refined.json"),
            "validation": str(root / "exports" / "video-edit-artifact-validation.json"),
            "preference_evidence": str(
                root / "exports" / "video-edit-preference-evidence.json"
            ),
            "dependency_snapshot": str(
                root / "exports" / "video-edit-dependency-snapshot.json"
            ),
            "mcp_args": str(root / "mcp-video-edit-review-pack.args.json"),
        },
    }
    if write:
        exports = root / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        write_json(exports / "video-edit-review-pack.json", result)
        (exports / "video-edit-review-pack.md").write_text(
            _render_markdown(result), encoding="utf-8"
        )
        write_json(exports / "storyboard.candidates.json", scenes)
        write_json(
            exports / "edit.decisions.refined.json", boundary.get("decisions") or []
        )
        write_json(exports / "video-edit-artifact-validation.json", validation)
        write_json(exports / "video-edit-preference-evidence.json", preference)
        write_json(exports / "video-edit-dependency-snapshot.json", dependency_snapshot)
        write_json(
            root / "mcp-video-edit-review-pack.args.json",
            {
                "bundle_dir": str(root),
                "decisions_json": str(resolved["decisions"] or ""),
                "tokens_json": str(resolved["tokens"] or ""),
                "silence_json": str(resolved["silence"] or ""),
                "delete_segments_json": str(resolved["delete_segments"] or ""),
                "cut_segments_json": str(resolved["cut_segments"] or ""),
                "ai_baseline_json": str(resolved["ai_baseline"] or ""),
                "media_path": str(_resolve_input_path(root, media_path) or ""),
                "reclaim_silence": False,
                "human_confirmed_diff": False,
                "review_attestation_path": str(review_attestation_path or ""),
                "write": True,
            },
        )
        manifest.update(
            {
                "video_edit_review_pack_json": "exports/video-edit-review-pack.json",
                "video_edit_review_pack_markdown": "exports/video-edit-review-pack.md",
                "video_edit_storyboard_candidates": "exports/storyboard.candidates.json",
                "video_edit_refined_decisions": "exports/edit.decisions.refined.json",
                "video_edit_artifact_validation": "exports/video-edit-artifact-validation.json",
                "video_edit_preference_evidence": "exports/video-edit-preference-evidence.json",
                "video_edit_dependency_snapshot": "exports/video-edit-dependency-snapshot.json",
                "mcp_video_edit_review_pack_args": "mcp-video-edit-review-pack.args.json",
                "video_edit_review_pack_updated_at": result["generated_at"],
            }
        )
        write_json(manifest_path, manifest)
        run = _register_run(root, result)
        result["run_registry"] = run
        write_json(exports / "video-edit-review-pack.json", result)
    return result


def spoken_tokens(reference: Any) -> list[dict[str, Any]]:
    if not isinstance(reference, list):
        return []
    return [
        item
        for item in reference
        if isinstance(item, dict)
        and not item.get("isGap")
        and _finite(item.get("start"))
        and _finite(item.get("end"))
        and float(item["end"]) > float(item["start"])
    ]


def build_token_silences(reference: Any) -> list[dict[str, float]]:
    tokens = spoken_tokens(reference)
    if not tokens:
        return []
    tokens = sorted(tokens, key=lambda row: float(row["start"]))
    out: list[dict[str, float]] = []
    first_start = float(tokens[0]["start"])
    if first_start > 0:
        out.append({"start": 0.0, "end": first_start})
    for previous, current in zip(tokens, tokens[1:]):
        start = float(previous["end"])
        end = float(current["start"])
        if end > start:
            out.append({"start": start, "end": end})
    return out


def merge_silences(*groups: Iterable[dict[str, Any]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for group in groups:
        for row in group or []:
            if (
                not isinstance(row, dict)
                or not _finite(row.get("start"))
                or not _finite(row.get("end"))
            ):
                continue
            start, end = float(row["start"]), float(row["end"])
            if end > start:
                rows.append({"start": start, "end": end})
    rows.sort(key=lambda row: row["start"])
    merged: list[dict[str, float]] = []
    for row in rows:
        if merged and row["start"] <= merged[-1]["end"] + TOL:
            merged[-1]["end"] = max(merged[-1]["end"], row["end"])
        else:
            merged.append(dict(row))
    return merged


def refine_edit_boundaries(
    decisions: list[dict[str, Any]],
    *,
    tokens: list[dict[str, Any]],
    silences: list[dict[str, Any]],
    max_sliver: float = 0.25,
) -> dict[str, Any]:
    refined: list[dict[str, Any]] = []
    snapped = 0
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        row = dict(decision)
        if str(row.get("action") or "delete") != "delete" or not silences:
            refined.append(row)
            continue
        if not _finite(row.get("start")) or not _finite(row.get("end")):
            refined.append(row)
            continue
        start, end = float(row["start"]), float(row["end"])
        new_start = (
            start
            if row.get("lock_start")
            else _snap_point(start, True, silences, tokens)
        )
        new_end = (
            end if row.get("lock_end") else _snap_point(end, False, silences, tokens)
        )
        if new_end - new_start < 0.02:
            new_start, new_end = start, end
        new_start, new_end = _r2(new_start), _r2(new_end)
        row.update(
            {"start": new_start, "end": new_end, "orig_start": start, "orig_end": end}
        )
        if new_start != start or new_end != end:
            snapped += 1
        refined.append(row)
    absorbed = _absorb_silent_slivers(refined, tokens, max_sliver=max_sliver)
    return {
        "schema": BOUNDARY_SCHEMA,
        "decision_count": len(refined),
        "snapped_count": snapped,
        "absorbed_sliver_count": absorbed,
        "silence_count": len(silences),
        "token_count": len(tokens),
        "decisions": refined,
        "source_design": "videocut-kit snap_boundaries.js concepts adapted to VKP Python contracts",
    }


def reclaim_silence_from_envelope(
    envelope_db: list[float],
    words: list[dict[str, Any]],
    *,
    base_silence: list[dict[str, Any]] | None = None,
    frame_ms: int = 10,
    smooth_frames: int = 2,
    expand_seconds: float = 0.30,
    offset_db: float = 12.0,
    min_reclaim: float = 0.10,
    merge_gap: float = 0.02,
) -> dict[str, Any]:
    if not envelope_db:
        return {
            "reclaimed_count": 0,
            "reclaimed": [],
            "merged": merge_silences(base_silence or []),
        }
    env = _max_pool([float(value) for value in envelope_db], smooth_frames)
    frame_seconds = frame_ms / 1000.0
    last = len(env) - 1

    def frame_index(seconds: float) -> int:
        return max(0, min(last, round(seconds / frame_seconds)))

    spoken = spoken_tokens(words)
    reclaimed: list[dict[str, float]] = []
    for left, right in zip(spoken, spoken[1:]):
        lo = frame_index(max(float(left["start"]), float(left["end"]) - expand_seconds))
        hi = frame_index(
            min(float(right["end"]), float(right["start"]) + expand_seconds)
        )
        if hi - lo < 2:
            continue
        threshold = max(env[lo : hi + 1]) - offset_db
        best_start = best_end = current_start = -1
        for index in range(lo, hi + 1):
            if env[index] < threshold:
                if current_start < 0:
                    current_start = index
                if index - current_start > best_end - best_start:
                    best_start, best_end = current_start, index
            else:
                current_start = -1
        if best_start >= 0 and (best_end - best_start) * frame_seconds >= min_reclaim:
            reclaimed.append(
                {
                    "start": _r2(best_start * frame_seconds),
                    "end": _r2(best_end * frame_seconds),
                }
            )
    return {
        "reclaimed_count": len(reclaimed),
        "reclaimed": reclaimed,
        "merged": _merge_with_gap([*(base_silence or []), *reclaimed], merge_gap),
    }


def reclaim_silence_from_media(
    media_path: str | Path,
    words: list[dict[str, Any]],
    *,
    base_silence: list[dict[str, Any]] | None = None,
    sample_rate: int = 8000,
    frame_ms: int = 10,
) -> dict[str, Any]:
    media = Path(media_path).expanduser().resolve()
    command = [
        "ffmpeg",
        "-nostats",
        "-hide_banner",
        "-loglevel",
        "error",
        "-vn",
        "-i",
        str(media),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    completed = subprocess.run(
        command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    samples = array("h")
    samples.frombytes(completed.stdout)
    window = max(1, round(sample_rate * frame_ms / 1000))
    envelope: list[float] = []
    for offset in range(0, len(samples), window):
        frame = samples[offset : offset + window]
        if not frame:
            continue
        mean_square = sum((sample / 32768.0) ** 2 for sample in frame) / len(frame)
        rms = math.sqrt(mean_square)
        envelope.append(20 * math.log10(rms) if rms > 0 else -90.0)
    result = reclaim_silence_from_envelope(
        envelope, words, base_silence=base_silence, frame_ms=frame_ms
    )
    result.update(
        {
            "requested": True,
            "executed": True,
            "status": "completed",
            "media_path": str(media),
            "sample_rate": sample_rate,
            "frame_ms": frame_ms,
        }
    )
    return result


def normalize_segments(
    items: Iterable[dict[str, Any]], *, merge_gap: float = 0.001
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for item in items or []:
        if (
            not isinstance(item, dict)
            or not _finite(item.get("start"))
            or not _finite(item.get("end"))
        ):
            continue
        start, end = float(item["start"]), float(item["end"])
        if end > start:
            rows.append({"start": start, "end": end})
    rows.sort(key=lambda row: row["start"])
    merged: list[dict[str, float]] = []
    for row in rows:
        if not merged or row["start"] > merged[-1]["end"] + merge_gap:
            merged.append(dict(row))
        else:
            merged[-1]["end"] = max(merged[-1]["end"], row["end"])
    return [{"start": _r2(row["start"]), "end": _r2(row["end"])} for row in merged]


def validate_edit_artifacts(
    decisions: list[dict[str, Any]],
    delete_segments: list[dict[str, Any]],
    *,
    cut_segments: list[dict[str, Any]] | None = None,
    duration: float = 0.0,
    tokens: list[dict[str, Any]] | None = None,
    silences: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    active = normalize_segments(
        row
        for row in decisions
        if isinstance(row, dict) and str(row.get("action") or "") == "delete"
    )
    exported = normalize_segments(delete_segments)
    issues: list[dict[str, Any]] = []
    mode = "raw"
    expected = active
    if not _same_segments(exported, active):
        if tokens is not None and silences is not None:
            refined = refine_edit_boundaries(
                [
                    row
                    for row in decisions
                    if isinstance(row, dict)
                    and str(row.get("action") or "") == "delete"
                ],
                tokens=tokens,
                silences=silences,
            )
            expected = normalize_segments(refined["decisions"])
            mode = "boundary_refined"
        if not _same_segments(exported, expected):
            issues.append(
                {
                    "reason": "delete_segments_mismatch",
                    "detail": f"exported={exported!r}; expected={expected!r}",
                }
            )
    normalized_cut = normalize_segments(cut_segments or [])
    if normalized_cut and duration > 0:
        coverage = sorted(
            [
                *(dict(row, kind="delete") for row in exported),
                *(dict(row, kind="keep") for row in normalized_cut),
            ],
            key=lambda row: row["start"],
        )
        cursor = 0.0
        for row in coverage:
            if row["start"] > cursor + SEGMENT_TOL:
                issues.append(
                    {
                        "reason": "timeline_coverage_gap",
                        "detail": f"{cursor:.2f}-{row['start']:.2f}",
                    }
                )
            if row["start"] < cursor - SEGMENT_TOL:
                issues.append(
                    {
                        "reason": "timeline_coverage_overlap",
                        "detail": f"at {row['start']:.2f}",
                    }
                )
            cursor = max(cursor, row["end"])
        if cursor < duration - SEGMENT_TOL:
            issues.append(
                {
                    "reason": "timeline_coverage_gap",
                    "detail": f"{cursor:.2f}-{duration:.2f}",
                }
            )
    if decisions and not delete_segments:
        issues.append(
            {
                "reason": "missing_delete_segments",
                "detail": "edit decisions exist but delete_segments.json is missing or empty",
            }
        )
    return {
        "schema": VALIDATION_SCHEMA,
        "ok": not issues,
        "status": "passed" if not issues else "blocked",
        "comparison_mode": mode,
        "active_delete_segments": active,
        "expected_delete_segments": expected,
        "exported_delete_segments": exported,
        "cut_segments": normalized_cut,
        "issues": issues,
        "hard_gate": True,
    }


def build_storyboard_candidates(
    timeline: list[Any],
    *,
    chapters: list[dict[str, Any]] | None = None,
    arbitration: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = [row for row in timeline if isinstance(row, dict)]
    chapters = chapters or []
    arbitration = arbitration or {}
    if chapters:
        return [
            _chapter_scene(chapter, rows, arbitration)
            for chapter in chapters
            if _valid_range(chapter)
        ]
    return [
        _timeline_scene(row, position, arbitration)
        for position, row in enumerate(rows, start=1)
        if _valid_range(row)
    ]


def build_preference_evidence(
    baseline: list[dict[str, Any]],
    final: list[dict[str, Any]],
    *,
    tokens: list[dict[str, Any]],
    video_id: str,
    human_confirmed: bool,
) -> dict[str, Any]:
    base_delete = [
        row
        for row in baseline
        if isinstance(row, dict) and str(row.get("action") or "") == "delete"
    ]
    final_delete = [
        row
        for row in final
        if isinstance(row, dict) and str(row.get("action") or "") == "delete"
    ]
    added = [
        row
        for row in final_delete
        if not any(_overlaps(row, candidate) for candidate in base_delete)
    ]
    restored = [
        row
        for row in base_delete
        if not any(_overlaps(row, candidate) for candidate in final_delete)
    ]
    rows = [
        _preference_row("user_added_delete", row, tokens, video_id, human_confirmed)
        for row in added
    ] + [
        _preference_row("user_restored_keep", row, tokens, video_id, human_confirmed)
        for row in restored
    ]
    return {
        "schema": PREFERENCE_SCHEMA,
        "video_id": video_id,
        "human_confirmed": bool(human_confirmed),
        "baseline_delete_count": len(base_delete),
        "final_delete_count": len(final_delete),
        "difference_count": len(rows),
        "eligible_difference_count": len(rows) if human_confirmed else 0,
        "differences": rows,
        "automatic_preference_write": False,
        "promotion_rule": "Keep observing until >=3 independent videos or an explicit user hardening decision.",
    }


def _chapter_scene(
    chapter: dict[str, Any],
    timeline: list[dict[str, Any]],
    arbitration: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    start, end = float(chapter["start"]), float(chapter["end"])
    covered = [
        row for row in timeline if _midpoint(row) >= start and _midpoint(row) < end
    ]
    indexes = [
        _row_index(row, position) for position, row in enumerate(covered, start=1)
    ]
    temporal = any(
        _mapping(row.get("temporal_visual_understanding")) for row in covered
    )
    visual = any(
        _mapping(row.get("visual_understanding")) or _text(row.get("visual_text"))
        for row in covered
    )
    evidence_paths = _unique_strings(
        value
        for row in covered
        for value in [
            row.get("frame_path"),
            *(row.get("frame_paths") or []),
            *(row.get("temporal_frame_paths") or []),
        ]
    )
    say = " ".join(
        str(value) for value in chapter.get("summary_sentences") or [] if str(value)
    ).strip()
    return {
        "id": f"scene-{int(chapter.get('index') or len(indexes) or 1):03d}",
        "start": start,
        "end": end,
        "start_time": format_timestamp(start),
        "end_time": format_timestamp(end),
        "captionIds": [],
        "timeline_indexes": indexes,
        "say": say or str(chapter.get("title") or ""),
        "screen": "broll" if temporal or visual else "facecam",
        "asset": "",
        "asset_candidates": evidence_paths,
        "clip": {"in": 0.0, "out": None},
        "fit": "auto",
        "media": {"spatial": "cover", "zoom": 1.0, "anchors": [], "htmlFile": ""},
        "note": str(chapter.get("title") or ""),
        "confirmed": False,
        "source": "vkp_smart_summary_candidate",
        "evidence": {
            "temporal_visual": temporal,
            "visual": visual,
            "asr_ocr_arbitration": [
                _compact_arbitration(arbitration.get(index, {}))
                for index in indexes
                if index in arbitration
            ],
            "citation_digest": chapter.get("citation_digest") or [],
        },
    }


def _timeline_scene(
    row: dict[str, Any], position: int, arbitration: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    index = _row_index(row, position)
    start, end = float(row["start"]), float(row["end"])
    temporal = _mapping(row.get("temporal_visual_understanding"))
    visual = _mapping(row.get("visual_understanding")) or bool(
        _text(row.get("visual_text"))
    )
    evidence_paths = _unique_strings(
        [
            row.get("frame_path"),
            *(row.get("frame_paths") or []),
            *(row.get("temporal_frame_paths") or []),
        ]
    )
    return {
        "id": f"scene-{index:03d}",
        "start": start,
        "end": end,
        "start_time": format_timestamp(start),
        "end_time": format_timestamp(end),
        "captionIds": [],
        "timeline_indexes": [index],
        "say": _text(
            row.get("corrected_transcript")
            or row.get("transcript")
            or row.get("asr_text")
            or row.get("text")
        ),
        "screen": "broll" if temporal or visual else "facecam",
        "asset": "",
        "asset_candidates": evidence_paths,
        "clip": {"in": 0.0, "out": None},
        "fit": "auto",
        "media": {"spatial": "cover", "zoom": 1.0, "anchors": [], "htmlFile": ""},
        "note": "Timeline evidence candidate",
        "confirmed": False,
        "source": "vkp_timeline_candidate",
        "evidence": {
            "temporal_visual": bool(temporal),
            "visual": bool(visual),
            "asr_ocr_arbitration": _compact_arbitration(arbitration.get(index, {})),
            "quality_issues": [
                str(value) for value in row.get("quality_issues") or [] if str(value)
            ],
        },
    }


def _snap_point(
    value: float,
    is_start: bool,
    silences: list[dict[str, Any]],
    tokens: list[dict[str, Any]],
) -> float:
    for silence in silences:
        start, end = float(silence["start"]), float(silence["end"])
        if value >= start - TOL and value <= end + TOL:
            if is_start:
                target = min(value, end - _guard_for(start, end, PAD))
                return min(max(target, start + PAD), end)
            target = max(value, start + _guard_for(start, end, PAD_ONSET))
            return max(min(target, end - PAD_ONSET), start)
    inside = next(
        (
            token
            for token in tokens
            if float(token["start"]) + TOL < value < float(token["end"]) - TOL
        ),
        None,
    )
    if not inside:
        return value
    token_start, token_end = float(inside["start"]), float(inside["end"])
    clip = min(0.08, (token_end - token_start) / 3)
    if is_start and token_end - value < clip:
        return token_end
    if not is_start and value - token_start < clip:
        return token_start
    if is_start:
        before = [row for row in silences if float(row["end"]) <= token_start + TOL]
        silence = before[-1] if before else None
        if silence and float(silence["end"]) >= token_start - TOL:
            start, end = float(silence["start"]), float(silence["end"])
            target = min(token_start, end - _guard_for(start, end, PAD))
            return min(max(target, start + PAD), end)
        return token_start
    silence = next(
        (row for row in silences if float(row["start"]) >= token_end - TOL), None
    )
    if silence and float(silence["start"]) <= token_end + TOL:
        start, end = float(silence["start"]), float(silence["end"])
        target = max(token_end, start + _guard_for(start, end, PAD_ONSET))
        return max(min(target, end - PAD_ONSET), start)
    return token_end


def _absorb_silent_slivers(
    decisions: list[dict[str, Any]], tokens: list[dict[str, Any]], *, max_sliver: float
) -> int:
    deletes = sorted(
        [
            row
            for row in decisions
            if str(row.get("action") or "delete") == "delete" and _valid_range(row)
        ],
        key=lambda row: float(row["start"]),
    )

    def has_token(start: float, end: float) -> bool:
        return any(
            start < (float(token["start"]) + float(token["end"])) / 2 < end
            for token in tokens
        )

    absorbed = 0
    if (
        deletes
        and 0 < float(deletes[0]["start"]) < max_sliver
        and not has_token(0.0, float(deletes[0]["start"]))
    ):
        deletes[0]["start"] = 0.0
        absorbed += 1
    for left, right in zip(deletes, deletes[1:]):
        gap_start, gap_end = float(left["end"]), float(right["start"])
        if 0 < gap_end - gap_start < max_sliver and not has_token(gap_start, gap_end):
            left["end"] = gap_end
            absorbed += 1
    return absorbed


def _guard_for(start: float, end: float, pad: float) -> float:
    room = end - start - pad
    if room <= GUARD_MIN:
        return max(0.0, room)
    return min(GUARD, max(GUARD_MIN, room * 0.6))


def _max_pool(values: list[float], half_width: int) -> list[float]:
    return [
        max(
            values[
                max(0, index - half_width) : min(len(values), index + half_width + 1)
            ]
        )
        for index in range(len(values))
    ]


def _merge_with_gap(rows: list[dict[str, Any]], gap: float) -> list[dict[str, float]]:
    valid = [
        {"start": float(row["start"]), "end": float(row["end"])}
        for row in rows
        if isinstance(row, dict)
        and _finite(row.get("start"))
        and _finite(row.get("end"))
        and float(row["end"]) > float(row["start"])
    ]
    valid.sort(key=lambda row: row["start"])
    merged: list[dict[str, float]] = []
    for row in valid:
        if merged and row["start"] <= merged[-1]["end"] + gap:
            merged[-1]["end"] = max(merged[-1]["end"], row["end"])
        else:
            merged.append(dict(row))
    return merged


def _same_segments(left: list[dict[str, float]], right: list[dict[str, float]]) -> bool:
    return len(left) == len(right) and all(
        abs(a["start"] - b["start"]) <= SEGMENT_TOL
        and abs(a["end"] - b["end"]) <= SEGMENT_TOL
        for a, b in zip(left, right)
    )


def _preference_row(
    direction: str,
    row: dict[str, Any],
    tokens: list[dict[str, Any]],
    video_id: str,
    confirmed: bool,
) -> dict[str, Any]:
    start, end = float(row.get("start") or 0.0), float(row.get("end") or 0.0)
    text = (
        "".join(
            _text(
                token.get("corrected_text")
                or token.get("original_text")
                or token.get("text")
            )
            for token in tokens
            if float(token["start"]) >= start - 0.15
            and float(token["start"]) < end + 0.30
        )
        or "(silence)"
    )
    return {
        "direction": direction,
        "start": _r2(start),
        "end": _r2(end),
        "kind": str(row.get("kind") or ""),
        "reason": str(row.get("reason") or ""),
        "text": text,
        "evidence_videos": [video_id],
        "sample_count": 1,
        "status": "observing",
        "human_confirmed": bool(confirmed),
        "eligible_for_learning": bool(confirmed),
        "promotion_eligible": False,
    }


def _read_chapters(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw = str(
        manifest.get("smart_summary_chapters") or "exports/smart-summary-chapters.json"
    )
    path = _resolve_input_path(root, raw)
    payload = _read_object(path) if path else {}
    return [row for row in payload.get("chapters") or [] if isinstance(row, dict)]


def _read_arbitration(
    root: Path, manifest: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    raw = str(
        manifest.get("transcript_source_arbitration_json")
        or "transcript-source-arbitration.json"
    )
    path = _resolve_input_path(root, raw)
    payload = _read_object(path) if path else {}
    rows = payload.get("items") or payload.get("review_rows") or []
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            result[int(row.get("index") or row.get("timeline_index"))] = row
        except (TypeError, ValueError):
            continue
    return result


def _compact_arbitration(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict) or not row:
        return {}
    return {
        "status": str(row.get("status") or row.get("decision") or ""),
        "selected_source": str(row.get("selected_source") or row.get("source") or ""),
        "needs_review": bool(row.get("needs_review")),
        "conflict_types": [
            str(value)
            for value in row.get("conflict_types") or row.get("issues") or []
            if str(value)
        ],
    }


def _edit_decisions_confirmed(decisions: list[dict[str, Any]]) -> bool:
    active = [
        row
        for row in decisions
        if isinstance(row, dict)
        and str(row.get("action") or "")
        in {"delete", "keep", "speed_up", "hold_screen"}
    ]
    return bool(active) and all(
        bool(row.get("confirmed")) or str(row.get("source") or "") == "user"
        for row in active
    )


def _blockers(
    *,
    scenes: list[dict[str, Any]],
    decisions_present: bool,
    decisions_confirmed: bool,
    delete_segments_present: bool,
    validation: dict[str, Any],
    reclaim_result: dict[str, Any],
    review_attestation: dict[str, Any],
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    if not scenes:
        blockers.append(
            {
                "reason": "missing_storyboard_candidates",
                "detail": "No valid Timeline or Smart Summary ranges were found.",
            }
        )
    if not decisions_present:
        blockers.append(
            {
                "reason": "missing_edit_decisions",
                "detail": "Candidate pack is available, but no reviewed edit decisions were supplied.",
            }
        )
    if decisions_present and not decisions_confirmed:
        blockers.append(
            {
                "reason": "unconfirmed_edit_decisions",
                "detail": "Every active edit decision must be user-sourced or carry confirmed=true.",
            }
        )
    if decisions_present and not delete_segments_present:
        blockers.append(
            {
                "reason": "missing_delete_segments",
                "detail": "Export delete segments before render handoff.",
            }
        )
    for issue in validation.get("issues") or []:
        blockers.append(
            {
                "reason": str(issue.get("reason") or "artifact_validation_failed"),
                "detail": str(issue.get("detail") or ""),
            }
        )
    if reclaim_result.get("requested") and reclaim_result.get("status") != "completed":
        blockers.append(
            {
                "reason": "silence_reclamation_incomplete",
                "detail": str(reclaim_result.get("status") or "unknown"),
            }
        )
    if review_attestation.get("status") != "valid":
        blockers.append(
            {
                "reason": "review_attestation_not_current",
                "detail": str(review_attestation.get("status") or "missing"),
            }
        )
    return blockers


def _register_run(root: Path, result: dict[str, Any]) -> dict[str, Any]:
    blockers = result.get("blockers") or []
    if not result.get("storyboard_scene_count"):
        status = "needs_input"
    elif blockers or bool(result.get("storyboard_review_required")):
        status = "needs_review"
    else:
        status = "completed"
    return register_bundle_run(
        root,
        run_type="video_edit_review_pack",
        run_id="video-edit-review-pack",
        status=status,
        title="Video edit review pack",
        summary=f"Prepared {result.get('storyboard_scene_count', 0)} storyboard candidates; {len(blockers)} blockers remain.",
        inputs=result.get("inputs") or {},
        parameters={
            "ready_for_single_ffmpeg": bool(result.get("ready_for_single_ffmpeg")),
            "snapped_count": int(
                (result.get("boundary_refinement") or {}).get("snapped_count") or 0
            ),
            "preference_difference_count": int(
                (result.get("preference_evidence") or {}).get("difference_count") or 0
            ),
        },
        artifacts=[
            {
                "key": "review_pack_json",
                "path": root / "exports" / "video-edit-review-pack.json",
            },
            {
                "key": "review_pack_markdown",
                "path": root / "exports" / "video-edit-review-pack.md",
            },
            {
                "key": "storyboard_candidates",
                "path": root / "exports" / "storyboard.candidates.json",
            },
            {
                "key": "refined_decisions",
                "path": root / "exports" / "edit.decisions.refined.json",
            },
            {
                "key": "artifact_validation",
                "path": root / "exports" / "video-edit-artifact-validation.json",
            },
            {
                "key": "preference_evidence",
                "path": root / "exports" / "video-edit-preference-evidence.json",
            },
        ],
        failed_items=[{"id": index + 1, **row} for index, row in enumerate(blockers)],
        retry_command=f".\\scripts\\video-knowledge.ps1 video-edit-review-pack {_ps_quote(root)}",
        next_actions=[
            "Review storyboard candidates and edit decisions in the existing Workbench.",
            "Run the existing single FFmpeg/render pipeline only after artifact_validation.ok=true and human confirmation.",
        ],
        operator_boundary=result.get("operator_boundary") or {},
        dependency_snapshot=result.get("dependency_snapshot") or {},
        write=True,
    )


def _render_markdown(result: dict[str, Any]) -> str:
    validation = result.get("artifact_validation") or {}
    boundary = result.get("boundary_refinement") or {}
    preference = result.get("preference_evidence") or {}
    lines = [
        "# Video Edit Review Pack",
        "",
        f"- Bundle: `{result.get('bundle_dir', '')}`",
        f"- Storyboard candidates: `{result.get('storyboard_scene_count', 0)}`",
        f"- Boundary snapped: `{boundary.get('snapped_count', 0)}`",
        f"- Silent slivers absorbed: `{boundary.get('absorbed_sliver_count', 0)}`",
        f"- Artifact validation: `{validation.get('status', 'unknown')}`",
        f"- Preference differences: `{preference.get('difference_count', 0)}`",
        f"- Ready for existing single FFmpeg pipeline: `{result.get('ready_for_single_ffmpeg', False)}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = result.get("blockers") or []
    lines.extend(
        f"- `{row.get('reason', '')}`: {row.get('detail', '')}" for row in blockers
    )
    if not blockers:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Storyboard Candidates",
            "",
            "| Scene | Time | Screen | Source | Confirmed |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for scene in result.get("storyboard_scenes") or []:
        lines.append(
            f"| `{scene.get('id', '')}` | `{scene.get('start_time', '')} - {scene.get('end_time', '')}` | "
            f"`{scene.get('screen', '')}` | `{scene.get('source', '')}` | `{scene.get('confirmed', False)}` |"
        )
    lines.extend(
        [
            "",
            "## Safety Boundary",
            "",
            "This pack is candidate and audit evidence only. It does not edit media, start a second render pipeline, call a cloud model, publish output, or write global preferences.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _first_existing(
    root: Path, explicit: str | Path | None, defaults: list[str]
) -> Path | None:
    if explicit:
        path = _resolve_input_path(root, explicit)
        if not path or not path.is_file():
            raise FileNotFoundError(f"input JSON not found: {path}")
        return path
    for value in defaults:
        path = _resolve_input_path(root, value)
        if path and path.is_file():
            return path
    return None


def _resolve_input_path(root: Path, value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _read_object(path: Path | None) -> dict[str, Any]:
    if not path or not path.is_file():
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def _read_list(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.is_file():
        return []
    value = read_json(path)
    return (
        [row for row in value if isinstance(row, dict)]
        if isinstance(value, list)
        else []
    )


def _timeline_duration(timeline: list[Any]) -> float:
    return max(
        (
            float(row.get("end") or 0.0)
            for row in timeline
            if isinstance(row, dict) and _finite(row.get("end"))
        ),
        default=0.0,
    )


def _row_index(row: dict[str, Any], position: int) -> int:
    try:
        return int(row.get("index") if row.get("index") is not None else position)
    except (TypeError, ValueError):
        return position


def _midpoint(row: dict[str, Any]) -> float:
    if not _valid_range(row):
        return -1.0
    return (float(row["start"]) + float(row["end"])) / 2


def _valid_range(row: dict[str, Any]) -> bool:
    return (
        _finite(row.get("start"))
        and _finite(row.get("end"))
        and float(row["end"]) > float(row["start"])
    )


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _unique_strings(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = _text(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not _valid_range(left) or not _valid_range(right):
        return False
    return float(left["start"]) < float(right["end"]) and float(left["end"]) > float(
        right["start"]
    )


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _r2(value: float) -> float:
    return round(float(value) + 1e-12, 2)


def _empty_boundary() -> dict[str, Any]:
    return {
        "schema": BOUNDARY_SCHEMA,
        "decision_count": 0,
        "snapped_count": 0,
        "absorbed_sliver_count": 0,
        "silence_count": 0,
        "token_count": 0,
        "decisions": [],
    }
