from __future__ import annotations

from pathlib import Path
from typing import Any

import jsonschema

from .canonical_json import canonical_json_sha256
from .content_clip_boundary import build_content_clip_boundary
from .content_clip_query_profiles import resolve_content_clip_query_profile
from .file_hash import sha256_file
from .run_artifact_registry import register_bundle_run
from .script_clip_candidate_pack import (
    _artifact_ref,
    _build_slot,
    _operator_boundary,
    _payload_sha,
    _read_object,
    _resolve_transcript_path,
    _validate_request_artifacts,
)
from .storage import bundle_write_lock, read_json, write_json, write_text_atomic
from .technical_shot_detection import load_verified_technical_shots
from .transcript import parse_transcript


REQUEST_SCHEMA = "video_knowledge_pipeline.content_clip_request.v1"
PACK_SCHEMA = "video_knowledge_pipeline.content_clip_candidate_pack.v1"
REVIEW_SCHEMA = "video_knowledge_pipeline.content_clip_review_notes.v1"

PACK_PATH = "exports/content-clip-candidate-pack.json"
PACK_MARKDOWN_PATH = "exports/content-clip-candidate-pack.md"
REVIEW_TODO_PATH = "content-clip-review-notes.todo.json"
MCP_ARGS_PATH = "mcp-content-clip-candidate-pack.args.json"


def adapt_script_clip_request_to_content(request: dict[str, Any]) -> dict[str, Any]:
    """Map the legacy interview-slot contract without changing its source file."""

    if request.get("schema") != "video_knowledge_pipeline.script_clip_request.v1":
        raise ValueError("legacy request must use video_knowledge_pipeline.script_clip_request.v1")
    clips: list[dict[str, Any]] = []
    for slot in request.get("slots") or []:
        if not isinstance(slot, dict):
            continue
        time_ranges = []
        window = slot.get("preferred_window") if isinstance(slot.get("preferred_window"), dict) else None
        if window:
            time_ranges.append({"start": float(window.get("start") or 0.0), "end": float(window.get("end") or 0.0)})
        clips.append(
            {
                "clip_id": str(slot.get("slot_id") or ""),
                "purpose": "quote",
                "profile_id": "spoken-quote-v1",
                "query": str(slot.get("expected_quote") or " ".join(slot.get("search_queries") or [])),
                "match_modes": ["quote", "semantic"],
                "required": bool(slot.get("required", True)),
                "state": str(slot.get("state") or "planned"),
                "required_modalities": ["asr"],
                "optional_modalities": ["audio", "shot"],
                "must_include": [],
                "must_exclude": [],
                "speaker_constraints": {
                    "allowed_roles": [str(value) for value in slot.get("required_speaker_roles") or [] if str(value)],
                    "excluded_roles": [str(value) for value in slot.get("excluded_speaker_roles") or [] if str(value)],
                },
                "duration": {"minimum_seconds": 0.5, "preferred_seconds": 20.0, "maximum_seconds": 120.0},
                "boundary_policy": "complete_sentence",
                "source_scope": {"video_ids": [], "time_ranges": time_ranges},
                "legacy_binding": {
                    "story_segment_ref": str(slot.get("story_segment_ref") or ""),
                    "episode_binding": str(slot.get("episode_binding") or slot.get("story_segment_ref") or ""),
                    "subtitle_candidate": str(slot.get("subtitle_candidate") or ""),
                },
            }
        )
    return {
        "schema": REQUEST_SCHEMA,
        "request_id": str(request.get("request_id") or ""),
        "source_transcript": dict(request.get("source_transcript") or {}),
        "clips": clips,
        "compatibility": {"adapted_from": "video_knowledge_pipeline.script_clip_request.v1"},
        "operator_boundary": dict(request.get("operator_boundary") or {}),
    }


def build_content_clip_candidate_pack(
    bundle_dir: str | Path,
    request_json: str | Path,
    *,
    top_k: int = 8,
    retrieval_backend: str = "keyword",
    context_seconds: float = 3.0,
    write: bool = True,
) -> dict[str, Any]:
    """Build a multimodal, review-only candidate pack for arbitrary content.

    Intent: generalize interview quote retrieval to explanations, screen text,
    visual/audio events, B-roll, highlights, and story beats. Decision: adapt
    every request into the proven script-slot retrieval engine, then enrich its
    candidates from existing Timeline and technical-shot evidence. Reason: this
    reuses the current VideoRAG/moment-index/transcript stack and avoids a second
    index or media pipeline. Evidence: script_clip_candidate_pack.v1 real Bundle
    dry-run plus local source reviews of VideoRAG, AutoShot, Shot2Story,
    WhisperX, and moys-asr-workflow. Effective scope: derived review artifacts;
    no model call, upload, cut, Timeline write, or automatic approval.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    request_path = Path(request_json).expanduser().resolve()
    manifest = _read_object(manifest_path, label="bundle manifest")
    raw_request = _read_object(request_path, label="content clip request")
    if raw_request.get("schema") == "video_knowledge_pipeline.script_clip_request.v1":
        _validate_request_artifacts(raw_request, request_path=request_path)
        request = adapt_script_clip_request_to_content(raw_request)
    else:
        request = raw_request
    _validate_schema(request, "content-clip-request.v1.schema.json")
    _validate_clips(request)
    if retrieval_backend not in {"keyword", "sqlite"}:
        raise ValueError("content clip retrieval_backend must be keyword or sqlite; no implicit vector fallback")
    if retrieval_backend == "sqlite" and not write:
        raise ValueError("sqlite retrieval writes a rebuildable local index; use keyword for no-write")

    transcript_path = _resolve_transcript_path(root, manifest, request, request_path)
    cues = parse_transcript(transcript_path)
    if not cues:
        raise ValueError("source transcript contains no parseable cues")
    timeline = read_json(root / "timeline.json") if (root / "timeline.json").is_file() else []
    timeline_rows = [row for row in timeline if isinstance(row, dict)] if isinstance(timeline, list) else []
    shots, shot_provenance = load_verified_technical_shots(root)
    transcript_metadata = _transcript_segment_metadata(transcript_path)
    media_end = max(
        [float(cue.end) for cue in cues]
        + [float(row.get("end") or row.get("end_seconds") or 0.0) for row in shots]
        + [float(row.get("end") or row.get("end_seconds") or 0.0) for row in timeline_rows]
        + [0.0]
    )

    clip_rows: list[dict[str, Any]] = []
    for clip in request["clips"]:
        _validate_source_scope(root, manifest, clip)
        profile = resolve_content_clip_query_profile(clip)
        required_modalities = _unique_text(clip.get("required_modalities") or profile["default_required_modalities"])
        optional_modalities = _unique_text(clip.get("optional_modalities") or profile["default_optional_modalities"])
        slot = _script_slot_projection(clip)
        slot_result = _build_slot(
            root,
            slot,
            transcript_path=transcript_path,
            cues=cues,
            top_k=max(1, int(top_k)),
            retrieval_backend=retrieval_backend,
            context_seconds=max(0.0, float(context_seconds)),
        )
        scope_ranges = [row for row in (clip.get("source_scope") or {}).get("time_ranges") or [] if isinstance(row, dict)]
        candidates = []
        for candidate in slot_result["candidates"]:
            if scope_ranges and not _candidate_in_scope(candidate, scope_ranges):
                continue
            if not _passes_hard_constraints(clip, candidate, timeline_rows):
                continue
            candidates.append(
                _enrich_candidate(
                    clip,
                    candidate,
                    profile=profile,
                    required_modalities=required_modalities,
                    optional_modalities=optional_modalities,
                    timeline_rows=timeline_rows,
                    technical_shots=shots,
                    media_end=media_end,
                    transcript_metadata=transcript_metadata,
                    scope_ranges=scope_ranges,
                )
            )
        candidates.sort(key=_candidate_sort_key)
        for rank, candidate in enumerate(candidates, start=1):
            candidate["rank"] = rank
            candidate["ranking"]["within_clip_percentile"] = round(1.0 if len(candidates) == 1 else 1.0 - (rank - 1) / (len(candidates) - 1), 6)
        clip_rows.append(
            {
                "clip_id": str(clip["clip_id"]),
                "purpose": str(clip["purpose"]),
                "profile": profile,
                "required": bool(clip.get("required", True)),
                "excluded": str(clip.get("state") or "planned") == "excluded",
                "query": str(clip["query"]),
                "match_modes": list(clip["match_modes"]),
                "required_modalities": required_modalities,
                "optional_modalities": optional_modalities,
                "must_include": _unique_text(clip.get("must_include") or []),
                "must_exclude": _unique_text(clip.get("must_exclude") or []),
                "speaker_constraints": dict(clip.get("speaker_constraints") or {}),
                "duration": dict(clip["duration"]),
                "boundary_policy": str(clip.get("boundary_policy") or profile["boundary_strategy"]),
                "search_status": slot_result["search_status"],
                "candidates": candidates,
            }
        )

    missing = [row["clip_id"] for row in clip_rows if row["required"] and not row["excluded"] and not row["candidates"]]
    missing_evidence = [
        {"clip_id": row["clip_id"], "candidate_id": candidate["candidate_id"], "modalities": candidate["missing_required_modalities"]}
        for row in clip_rows
        for candidate in row["candidates"]
        if candidate["missing_required_modalities"]
    ]
    source_artifacts = [_artifact_ref(request_path, role="content_clip_request"), _artifact_ref(transcript_path, role="canonical_or_corrected_transcript")]
    timeline_path = root / "timeline.json"
    if timeline_path.is_file():
        source_artifacts.append(_artifact_ref(timeline_path, role="timeline_evidence"))
    shot_path = Path(str(shot_provenance.get("path") or ""))
    if shot_path.is_file():
        source_artifacts.append(_artifact_ref(shot_path, role="technical_shot_evidence"))
    result: dict[str, Any] = {
        "schema": PACK_SCHEMA,
        "status": "needs_candidate_selection" if any(row["candidates"] for row in clip_rows) else "needs_search",
        "request_id": str(request.get("request_id") or ""),
        "bundle_dir": str(root),
        "source_artifacts": source_artifacts,
        "technical_shot_provenance": shot_provenance,
        "parameters": {"top_k": max(1, int(top_k)), "retrieval_backend": retrieval_backend, "context_seconds": max(0.0, float(context_seconds))},
        "summary": {
            "clip_count": len(clip_rows),
            "candidate_count": sum(len(row["candidates"]) for row in clip_rows),
            "missing_required_clip_ids": missing,
            "candidate_missing_required_evidence_count": len(missing_evidence),
            "candidate_missing_required_evidence": missing_evidence,
        },
        "clips": clip_rows,
        "review_required": True,
        "publication_allowed": False,
        "operator_boundary": {**_operator_boundary(), "automatic_media_cut": False, "automatic_model_escalation": False},
        "artifacts": {"json": str(root / PACK_PATH), "markdown": str(root / PACK_MARKDOWN_PATH), "review_todo": str(root / REVIEW_TODO_PATH), "mcp_args": str(root / MCP_ARGS_PATH)},
    }
    result["pack_sha256"] = _payload_sha(result)
    _validate_schema(result, "content-clip-candidate-pack.v1.schema.json")
    review = _review_template(result)
    _validate_schema(review, "content-clip-review-notes.v1.schema.json")
    if write:
        with bundle_write_lock(root, operation="content_clip_candidate_pack", timeout_seconds=1.0):
            write_json(root / PACK_PATH, result)
            review["candidate_pack"]["artifact_sha256"] = sha256_file(root / PACK_PATH)
            write_text_atomic(root / PACK_MARKDOWN_PATH, _render_markdown(result))
            write_json(root / REVIEW_TODO_PATH, review)
            write_json(root / MCP_ARGS_PATH, {"bundle_dir": str(root), "request_json": str(request_path), "top_k": max(1, int(top_k)), "retrieval_backend": retrieval_backend, "context_seconds": max(0.0, float(context_seconds)), "write": True})
            current = _read_object(manifest_path, label="bundle manifest")
            current.update({"content_clip_candidate_pack_json": PACK_PATH, "content_clip_candidate_pack_markdown": PACK_MARKDOWN_PATH, "content_clip_review_notes_todo": REVIEW_TODO_PATH, "mcp_content_clip_candidate_pack_args": MCP_ARGS_PATH})
            write_json(manifest_path, current)
        register_bundle_run(
            root,
            run_type="content_clip_candidate_pack",
            run_id="content-clip-candidate-pack",
            status="needs_review" if result["summary"]["candidate_count"] else "needs_input",
            title="通用内容片段候选包",
            summary=f"{len(clip_rows)} requests, {result['summary']['candidate_count']} candidates; human selection required.",
            inputs={"request": source_artifacts[0], "transcript": source_artifacts[1]},
            parameters=result["parameters"],
            artifacts=[{"key": "candidate_pack", "path": root / PACK_PATH}, {"key": "candidate_pack_markdown", "path": root / PACK_MARKDOWN_PATH}, {"key": "review_todo", "path": root / REVIEW_TODO_PATH}, {"key": "mcp_args", "path": root / MCP_ARGS_PATH}],
            failed_items=[{"id": clip_id, "reason": "missing_required_clip", "detail": "No local candidate was found."} for clip_id in missing],
            retry_command=f'.\\scripts\\video-knowledge.ps1 content-clip-candidate-pack "{root}" "{request_path}"',
            next_actions=["Select a candidate, confirm multimodal evidence and boundaries, then run content-clip-alignment-check after fine cutting."],
            operator_boundary=result["operator_boundary"],
            write=True,
        )
    return result


def _script_slot_projection(clip: dict[str, Any]) -> dict[str, Any]:
    scope = clip.get("source_scope") if isinstance(clip.get("source_scope"), dict) else {}
    ranges = [row for row in scope.get("time_ranges") or [] if isinstance(row, dict)]
    legacy = clip.get("legacy_binding") if isinstance(clip.get("legacy_binding"), dict) else {}
    query_terms = _unique_text([clip.get("query"), *(clip.get("must_include") or [])])
    return {
        "slot_id": str(clip.get("clip_id") or ""),
        "state": str(clip.get("state") or "planned"),
        "story_segment_ref": str(legacy.get("story_segment_ref") or ""),
        "episode_binding": str(legacy.get("episode_binding") or clip.get("clip_id") or ""),
        "required": bool(clip.get("required", True)),
        "search_queries": query_terms,
        "expected_quote": str(clip.get("query") or ""),
        "subtitle_candidate": str(legacy.get("subtitle_candidate") or ""),
        "preferred_window": ranges[0] if ranges else None,
        "required_speaker_roles": list((clip.get("speaker_constraints") or {}).get("allowed_roles") or []),
        "excluded_speaker_roles": list((clip.get("speaker_constraints") or {}).get("excluded_roles") or []),
    }


def _enrich_candidate(
    clip: dict[str, Any],
    candidate: dict[str, Any],
    *,
    profile: dict[str, Any],
    required_modalities: list[str],
    optional_modalities: list[str],
    timeline_rows: list[dict[str, Any]],
    technical_shots: list[dict[str, Any]],
    media_end: float,
    transcript_metadata: dict[str, dict[str, Any]],
    scope_ranges: list[dict[str, Any]],
) -> dict[str, Any]:
    start = float(candidate["source_time_range"]["start"])
    end = float(candidate["source_time_range"]["end"])
    overlapping_timeline = [row for row in timeline_rows if _overlaps(row, start, end)]
    enriched_segments = []
    for segment in candidate.get("transcript_segments") or []:
        segment_id = str(segment.get("segment_id") or "")
        enriched_segments.append({**segment, **dict(transcript_metadata.get(segment_id) or {})})
    candidate = {**candidate, "transcript_segments": enriched_segments}
    evidence = _modality_evidence(candidate, overlapping_timeline, technical_shots, start=start, end=end)
    boundary = build_content_clip_boundary(
        semantic_start=start,
        semantic_end=end,
        transcript_segments=enriched_segments,
        technical_shots=technical_shots,
        timeline_rows=overlapping_timeline,
        boundary_strategy=str(clip.get("boundary_policy") or profile["boundary_strategy"]),
        duration=dict(clip["duration"]),
        media_end=media_end,
    )
    if scope_ranges:
        _clamp_boundary_to_scope(boundary, scope_ranges)
    missing = [name for name in required_modalities if evidence.get(name, {}).get("status") == "unavailable"]
    exact_terms = [term for term in clip.get("must_include") or [] if _normalise(term) and _normalise(term) in _normalise(candidate.get("snippet"))]
    origin = str(candidate.get("retrieval", {}).get("origin") or "")
    evidence_count = sum(1 for name in set([*required_modalities, *optional_modalities]) if evidence.get(name, {}).get("status") != "unavailable")
    tier = 0 if origin == "script_preferred_window" else 1 if "quote" in (clip.get("match_modes") or []) and len(exact_terms) == len(clip.get("must_include") or []) else 2 if evidence_count >= 2 else 3
    generic_id = "content-clip-candidate-" + canonical_json_sha256({"clip_id": clip["clip_id"], "start": start, "end": end, "segments": candidate.get("source_segment_ids") or []})[:16]
    return {
        **candidate,
        "candidate_id": generic_id,
        "clip_id": str(clip["clip_id"]),
        "modality_evidence": evidence,
        "missing_required_modalities": missing,
        "eligibility_status": "missing_required_evidence" if missing else "needs_boundary_review" if boundary["human_boundary_review_required"] else "eligible",
        "boundary": boundary,
        "ranking": {"evidence_tier": tier, "evidence_count": evidence_count, "source_rank": int(candidate.get("rank") or 0), "within_clip_percentile": 0.0, "raw_scores_compared_across_retrievers": False},
    }


def _modality_evidence(candidate: dict[str, Any], timeline: list[dict[str, Any]], shots: list[dict[str, Any]], *, start: float, end: float) -> dict[str, Any]:
    ocr_rows = [row for row in timeline if any(_text(row.get(key)) for key in ("visual_text", "ocr_text", "screen_text", "structured_visual"))]
    visual_rows = [row for row in timeline if any(row.get(key) for key in ("visual_understanding", "temporal_visual_understanding", "visual_observation", "frame_paths"))]
    audio_rows = [row for row in timeline if row.get("audio_events") or row.get("audio_event")]
    segment_audio_ids = [
        str(row.get("segment_id") or "")
        for row in candidate.get("transcript_segments") or []
        if row.get("audio_events")
    ]
    shot_rows = [row for row in shots if _overlaps(row, start, end)]
    return {
        "asr": _evidence_status(bool(candidate.get("transcript_segments")), candidate.get("source_segment_ids") or []),
        "ocr": _evidence_status(bool(ocr_rows), [_timeline_id(row) for row in ocr_rows]),
        "visual": _evidence_status(bool(visual_rows), [_timeline_id(row) for row in visual_rows]),
        "audio": _evidence_status(bool(audio_rows or segment_audio_ids), [*[_timeline_id(row) for row in audio_rows], *segment_audio_ids]),
        "shot": _evidence_status(bool(shot_rows), [str(row.get("shot_id") or row.get("id") or "") for row in shot_rows]),
    }


def _evidence_status(available: bool, ids: list[Any]) -> dict[str, Any]:
    return {"status": "confirmed" if available else "unavailable", "evidence_ids": _unique_text(ids), "missing_evidence": [] if available else ["no_existing_bundle_evidence"]}


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, int, int, int]:
    ranking = candidate["ranking"]
    return (int(ranking["evidence_tier"]), len(candidate["missing_required_modalities"]), -int(ranking["evidence_count"]), int(ranking["source_rank"]))


def _review_template(pack: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": REVIEW_SCHEMA,
        "review_status": "draft",
        "review_id": "",
        "candidate_pack": {"path": PACK_PATH, "pack_sha256": pack["pack_sha256"], "artifact_sha256": ""},
        "clips": [
            {
                "clip_id": row["clip_id"],
                "selected_candidate_id": "",
                "fine_cut_order": None,
                "fine_cut_output": "",
                "approved_window": None,
                "approved_source_segment_ids": [],
                "expected_speaker_role": "",
                "expected_speaker_ids": [],
                "excluded_speaker_roles": list(row["speaker_constraints"].get("excluded_roles") or []),
                "excluded_speaker_ids": [],
                "approved_clip_text": "",
                "subtitle_text": "",
                "clip_evidence": {name: {"path": "", "sha256": ""} for name in ("asr", "ocr", "visual", "audio")},
                "modality_reviews": {name: "not_reviewed" for name in ("asr", "ocr", "visual", "audio", "shot")},
                "human_confirmed": False,
                "notes": "",
            }
            for row in pack["clips"]
        ],
        "operator_boundary": {**_operator_boundary(), "draft_is_not_formal_approval": True, "human_confirmed_required_for_alignment": True},
    }


def _validate_clips(request: dict[str, Any]) -> None:
    seen: set[str] = set()
    for clip in request["clips"]:
        clip_id = str(clip["clip_id"])
        if clip_id in seen:
            raise ValueError(f"content clip request contains duplicate clip_id: {clip_id}")
        seen.add(clip_id)
        duration = clip["duration"]
        minimum = float(duration.get("minimum_seconds") or 0.0)
        preferred = float(duration.get("preferred_seconds") or 0.0)
        maximum = float(duration.get("maximum_seconds") or 0.0)
        if not (0.0 <= minimum <= preferred <= maximum):
            raise ValueError(f"content clip duration must satisfy minimum <= preferred <= maximum: {clip_id}")
        for row in (clip.get("source_scope") or {}).get("time_ranges") or []:
            if float(row.get("end") or 0.0) <= float(row.get("start") or 0.0):
                raise ValueError(f"content clip source time range must have end > start: {clip_id}")
        resolve_content_clip_query_profile(clip)


def _validate_source_scope(root: Path, manifest: dict[str, Any], clip: dict[str, Any]) -> None:
    scope = clip.get("source_scope") if isinstance(clip.get("source_scope"), dict) else {}
    allowed = {str(value).strip() for value in scope.get("video_ids") or [] if str(value).strip()}
    if not allowed:
        return
    identities = {
        root.name,
        str(manifest.get("video_id") or "").strip(),
        str(manifest.get("id") or "").strip(),
        str(manifest.get("title") or "").strip(),
    }
    if not allowed.intersection(identities):
        raise ValueError(f"content clip source_scope excludes this Bundle: {clip.get('clip_id')}")


def _candidate_in_scope(candidate: dict[str, Any], ranges: list[dict[str, Any]]) -> bool:
    value = candidate.get("source_time_range") if isinstance(candidate.get("source_time_range"), dict) else {}
    start, end = float(value.get("start") or 0.0), float(value.get("end") or 0.0)
    return any(end > float(row.get("start") or 0.0) and start < float(row.get("end") or 0.0) for row in ranges)


def _passes_hard_constraints(clip: dict[str, Any], candidate: dict[str, Any], timeline_rows: list[dict[str, Any]]) -> bool:
    value = candidate.get("source_time_range") if isinstance(candidate.get("source_time_range"), dict) else {}
    start, end = float(value.get("start") or 0.0), float(value.get("end") or 0.0)
    timeline = [row for row in timeline_rows if _overlaps(row, start, end)]
    searchable = _normalise(" ".join([str(candidate.get("snippet") or ""), *[_text(row) for row in timeline]]))
    if any(_normalise(term) and _normalise(term) not in searchable for term in clip.get("must_include") or []):
        return False
    if any(_normalise(term) and _normalise(term) in searchable for term in clip.get("must_exclude") or []):
        return False
    source_segments = [row for row in candidate.get("transcript_segments") or [] if isinstance(row, dict) and _overlaps(row, start, end)]
    roles = {str(row.get("speaker_role") or "") for row in source_segments if str(row.get("speaker_role") or "")}
    speakers = {str(row.get("speaker") or "") for row in source_segments if str(row.get("speaker") or "")}
    constraints = clip.get("speaker_constraints") if isinstance(clip.get("speaker_constraints"), dict) else {}
    allowed_roles = {str(value) for value in constraints.get("allowed_roles") or [] if str(value)}
    allowed_speakers = {str(value) for value in constraints.get("allowed_speaker_ids") or [] if str(value)}
    excluded_roles = {str(value) for value in constraints.get("excluded_roles") or [] if str(value)}
    excluded_speakers = {str(value) for value in constraints.get("excluded_speaker_ids") or [] if str(value)}
    if excluded_roles.intersection(roles) or excluded_speakers.intersection(speakers):
        return False
    if allowed_roles and roles and not allowed_roles.intersection(roles):
        return False
    if allowed_speakers and speakers and not allowed_speakers.intersection(speakers):
        return False
    return True


def _clamp_boundary_to_scope(boundary: dict[str, Any], ranges: list[dict[str, Any]]) -> None:
    semantic = boundary.get("semantic_match_range") if isinstance(boundary.get("semantic_match_range"), dict) else {}
    start, end = float(semantic.get("start") or 0.0), float(semantic.get("end") or 0.0)
    scope = next((row for row in ranges if end > float(row.get("start") or 0.0) and start < float(row.get("end") or 0.0)), None)
    if scope is None:
        return
    lower, upper = float(scope.get("start") or 0.0), float(scope.get("end") or 0.0)
    changed = False
    for key in ("recommended_cut_range", "safe_extension_range"):
        value = boundary.get(key) if isinstance(boundary.get(key), dict) else {}
        bounded_start = max(lower, float(value.get("start") or lower))
        bounded_end = min(upper, float(value.get("end") or upper))
        if bounded_start != float(value.get("start") or 0.0) or bounded_end != float(value.get("end") or 0.0):
            changed = True
        boundary[key] = _time_range(bounded_start, max(bounded_start, bounded_end))
    if changed:
        boundary.setdefault("boundary_reason", []).append("clamped_to_explicit_source_scope")


def _transcript_segment_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix.lower() != ".json":
        return {}
    payload = read_json(path)
    rows = payload.get("segments") or payload.get("cues") or [] if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        segment_id = str(row.get("segment_id") or row.get("id") or f"segment-{index:06d}")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        words = row.get("words") if isinstance(row.get("words"), list) else metadata.get("words") if isinstance(metadata.get("words"), list) else []
        audio_events = row.get("audio_events") or row.get("events") or metadata.get("audio_events") or []
        result[segment_id] = {
            "words": [dict(value) for value in words if isinstance(value, dict)],
            "audio_events": [str(value) for value in audio_events if str(value)] if isinstance(audio_events, list) else [],
        }
    return result


def _validate_schema(payload: dict[str, Any], filename: str) -> None:
    schema = read_json(Path(__file__).with_name("schemas") / filename)
    jsonschema.validate(payload, schema)


def _overlaps(row: dict[str, Any], start: float, end: float) -> bool:
    row_start = float(row.get("start") or row.get("start_seconds") or 0.0)
    row_end = max(row_start, float(row.get("end") or row.get("end_seconds") or row_start))
    return row_end > start and row_start < end


def _timeline_id(row: dict[str, Any]) -> str:
    return f"timeline-{row.get('index')}" if row.get("index") is not None else ""


def _text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_text(item) for item in value)
    return str(value or "").strip()


def _normalise(value: Any) -> str:
    return "".join(character.lower() for character in str(value or "") if character.isalnum())


def _time_range(start: float, end: float) -> dict[str, Any]:
    from .transcript import format_timestamp

    return {"start": round(start, 6), "end": round(end, 6), "start_time": format_timestamp(start), "end_time": format_timestamp(end)}


def _unique_text(values: Any) -> list[str]:
    result: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _render_markdown(pack: dict[str, Any]) -> str:
    lines = [
        "# 通用内容片段候选包",
        "",
        f"- Request: `{pack.get('request_id', '')}`",
        f"- Status: `{pack.get('status', '')}`",
        f"- Clips: `{pack.get('summary', {}).get('clip_count', 0)}`",
        f"- Candidates: `{pack.get('summary', {}).get('candidate_count', 0)}`",
        "- 边界：本地产生候选和派生定界；必须人工选择；不得自动剪切或发布。",
        "",
    ]
    for clip in pack.get("clips") or []:
        lines.extend([f"## {clip.get('clip_id')} · {clip.get('profile', {}).get('profile_id')}", "", f"- Query: {clip.get('query')}", f"- Required modalities: `{', '.join(clip.get('required_modalities') or [])}`", f"- Search status: `{clip.get('search_status')}`", ""])
        for candidate in clip.get("candidates") or []:
            cut = candidate.get("boundary", {}).get("recommended_cut_range") or {}
            lines.extend([f"### #{candidate.get('rank')} `{candidate.get('candidate_id')}`", "", f"- 推荐范围：`{cut.get('start_time')}` — `{cut.get('end_time')}`", f"- Eligibility: `{candidate.get('eligibility_status')}`", f"- Missing required modalities: `{', '.join(candidate.get('missing_required_modalities') or []) or 'none'}`", f"- Evidence tier: `{candidate.get('ranking', {}).get('evidence_tier')}`（不同检索器原始分数不跨源比较）", "", str(candidate.get("snippet") or ""), ""])
        if not clip.get("candidates"):
            lines.extend(["没有候选；请补查询、时间范围或相应模态证据。", ""])
    return "\n".join(lines).rstrip() + "\n"
