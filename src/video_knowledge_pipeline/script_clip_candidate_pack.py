from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import jsonschema
from rapidfuzz import fuzz

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file
from .run_artifact_registry import register_bundle_run
from .storage import bundle_write_lock, read_json, write_json, write_text_atomic
from .transcript import format_timestamp, parse_transcript
from .transcript_reference_window import export_transcript_reference_window
from .video_moment_index import build_video_moment_index
from .video_rag_search import search_video_rag


REQUEST_SCHEMA = "video_knowledge_pipeline.script_clip_request.v1"
PACK_SCHEMA = "video_knowledge_pipeline.script_clip_candidate_pack.v1"
REVIEW_SCHEMA = "video_knowledge_pipeline.script_clip_review_notes.v1"

PACK_PATH = "exports/script-clip-candidate-pack.json"
PACK_MARKDOWN_PATH = "exports/script-clip-candidate-pack.md"
REVIEW_TODO_PATH = "script-clip-review-notes.todo.json"
MCP_ARGS_PATH = "mcp-script-clip-candidate-pack.args.json"


def build_script_clip_candidate_pack(
    bundle_dir: str | Path,
    request_json: str | Path,
    *,
    top_k: int = 8,
    retrieval_backend: str = "keyword",
    context_seconds: float = 3.0,
    write: bool = True,
) -> dict[str, Any]:
    """Build a local, review-only script-slot to source-clip candidate pack.

    Intent: let the reviewed script drive source-clip retrieval without making
    the script, search results, or machine speaker labels authoritative.
    Decision: orchestrate the existing VideoRAG search, moment index,
    transcript parser, and reference-window exporter; add only slot-level
    ranking, lineage, and review artifacts.
    Reason: those modules already own retrieval, temporal evidence, speaker
    preservation, and run registration. Reimplementing them would create a
    second index and a second transcript parser.
    Evidence: ``video_rag_search.v1``, ``video_moment_index.v1``, and
    ``transcript_reference_window.v1`` are stable local VKP contracts.
    Effective scope: derived Bundle review files only. No media, Timeline,
    canonical transcript, upstream script, or publication state is changed.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    request_path = Path(request_json).expanduser().resolve()
    if retrieval_backend not in {"keyword", "sqlite"}:
        raise ValueError("script clip retrieval_backend must be keyword or sqlite; no implicit vector fallback")
    if retrieval_backend == "sqlite" and not write:
        raise ValueError("sqlite retrieval writes a rebuildable local index; use keyword for a no-write preview")
    request = _read_object(request_path, label="script clip request")
    _validate_schema(request, "script-clip-request.v1.schema.json")
    _validate_request_artifacts(request, request_path=request_path)
    _validate_request_slots(request)

    manifest = _read_object(manifest_path, label="bundle manifest")
    transcript_path = _resolve_transcript_path(root, manifest, request, request_path)
    cues = parse_transcript(transcript_path)
    if not cues:
        raise ValueError("source transcript contains no parseable cues")
    slots: list[dict[str, Any]] = []
    for request_slot in request.get("slots") or []:
        slots.append(
            _build_slot(
                root,
                request_slot,
                transcript_path=transcript_path,
                cues=cues,
                top_k=max(1, int(top_k)),
                retrieval_backend=retrieval_backend,
                context_seconds=max(0.0, float(context_seconds)),
            )
        )

    source_artifacts = [
        _artifact_ref(request_path, role="script_clip_request"),
        _artifact_ref(transcript_path, role="canonical_or_corrected_transcript"),
    ]
    script_ref = request.get("script") if isinstance(request.get("script"), dict) else {}
    script_path = _resolve_declared_path(script_ref.get("path"), request_path.parent)
    if script_path and script_path.is_file():
        source_artifacts.append(_artifact_ref(script_path, role="reviewed_script"))
    upstream_ref = request.get("upstream_handoff") if isinstance(request.get("upstream_handoff"), dict) else {}
    upstream_path = _resolve_declared_path(upstream_ref.get("path"), request_path.parent)
    if upstream_path and upstream_path.is_file():
        source_artifacts.append(_artifact_ref(upstream_path, role="content_studio_handoff"))

    missing_required = [row["slot_id"] for row in slots if row["required"] and not row["excluded"] and not row["candidates"]]
    not_searched = [row["slot_id"] for row in slots if row["search_status"] == "candidate_not_searched"]
    payload: dict[str, Any] = {
        "schema": PACK_SCHEMA,
        "status": "needs_candidate_selection" if any(row["candidates"] for row in slots) else "needs_search",
        "request_id": str(request.get("request_id") or ""),
        "bundle_dir": str(root),
        "source_artifacts": source_artifacts,
        "parameters": {
            "top_k": max(1, int(top_k)),
            "retrieval_backend": str(retrieval_backend),
            "context_seconds": max(0.0, float(context_seconds)),
        },
        "summary": {
            "slot_count": len(slots),
            "required_slot_count": sum(1 for row in slots if row["required"] and not row["excluded"]),
            "candidate_count": sum(len(row["candidates"]) for row in slots),
            "missing_required_slot_ids": missing_required,
            "candidate_not_searched_slot_ids": not_searched,
            "speaker_role_unresolved_candidate_count": sum(
                1
                for row in slots
                for candidate in row["candidates"]
                if candidate["speaker_evidence"]["role_status"] == "unresolved"
            ),
        },
        "slots": slots,
        "review_required": True,
        "publication_allowed": False,
        "operator_boundary": _operator_boundary(),
        "artifacts": {
            "json": str(root / PACK_PATH),
            "markdown": str(root / PACK_MARKDOWN_PATH),
            "review_todo": str(root / REVIEW_TODO_PATH),
            "mcp_args": str(root / MCP_ARGS_PATH),
        },
    }
    payload["pack_sha256"] = _payload_sha(payload)
    _validate_schema(payload, "script-clip-candidate-pack.v1.schema.json")
    review_todo = _review_template(payload)
    _validate_schema(review_todo, "script-clip-review-notes.v1.schema.json")

    if write:
        with bundle_write_lock(root, operation="script_clip_candidate_pack", timeout_seconds=1.0):
            write_json(root / PACK_PATH, payload)
            review_todo["candidate_pack"]["artifact_sha256"] = sha256_file(root / PACK_PATH)
            write_text_atomic(root / PACK_MARKDOWN_PATH, _render_pack_markdown(payload))
            write_json(root / REVIEW_TODO_PATH, review_todo)
            write_json(
                root / MCP_ARGS_PATH,
                {
                    "bundle_dir": str(root),
                    "request_json": str(request_path),
                    "top_k": max(1, int(top_k)),
                    "retrieval_backend": str(retrieval_backend),
                    "context_seconds": max(0.0, float(context_seconds)),
                    "write": True,
                },
            )
            current_manifest = _read_object(manifest_path, label="bundle manifest")
            current_manifest.update(
                {
                    "script_clip_candidate_pack_json": PACK_PATH,
                    "script_clip_candidate_pack_markdown": PACK_MARKDOWN_PATH,
                    "script_clip_review_notes_todo": REVIEW_TODO_PATH,
                    "mcp_script_clip_candidate_pack_args": MCP_ARGS_PATH,
                }
            )
            write_json(manifest_path, current_manifest)
        failed_items = [
            {"id": slot_id, "reason": "candidate_not_searched", "detail": "The slot has no explicit local retrieval query."}
            for slot_id in not_searched
        ] + [
            {"id": slot_id, "reason": "missing_required_slot", "detail": "No candidate was found for the required slot."}
            for slot_id in missing_required
        ]
        register_bundle_run(
            root,
            run_type="script_clip_candidate_pack",
            run_id="script-clip-candidate-pack",
            status="needs_review" if payload["summary"]["candidate_count"] else "needs_input",
            title="脚本驱动的采访片段候选包",
            summary=f"{len(slots)} slots, {payload['summary']['candidate_count']} local candidates; human selection required.",
            inputs={"request": source_artifacts[0], "transcript": source_artifacts[1]},
            parameters=payload["parameters"],
            artifacts=[
                {"key": "candidate_pack", "path": root / PACK_PATH},
                {"key": "candidate_pack_markdown", "path": root / PACK_MARKDOWN_PATH},
                {"key": "review_todo", "path": root / REVIEW_TODO_PATH},
                {"key": "mcp_args", "path": root / MCP_ARGS_PATH},
            ],
            failed_items=failed_items,
            retry_command=f'.\\scripts\\video-knowledge.ps1 script-clip-candidate-pack "{root}" "{request_path}"',
            next_actions=["Open script-clip-review-notes.todo.json, select candidates, verify speaker roles and quotes, then mark review_status=human_confirmed."],
            operator_boundary=payload["operator_boundary"],
            write=True,
        )
    return payload


def _build_slot(
    root: Path,
    slot: dict[str, Any],
    *,
    transcript_path: Path,
    cues: list[Any],
    top_k: int,
    retrieval_backend: str,
    context_seconds: float,
) -> dict[str, Any]:
    slot_id = str(slot.get("slot_id") or "").strip()
    excluded = str(slot.get("state") or "planned") == "excluded"
    required = bool(slot.get("required", True))
    queries = _unique_text(slot.get("search_queries") or [])
    raw_candidates: list[dict[str, Any]] = []
    preferred = slot.get("preferred_window") if isinstance(slot.get("preferred_window"), dict) else {}
    if preferred and not excluded:
        raw_candidates.append(
            {
                "origin": "script_preferred_window",
                "query": str(slot.get("expected_quote") or (queries[0] if queries else "")),
                "score": 1.0,
                "start": float(preferred.get("start") or 0.0),
                "end": float(preferred.get("end") or 0.0),
                "coarse_start": float(preferred.get("start") or 0.0),
                "coarse_end": float(preferred.get("end") or 0.0),
                "snippet": str(slot.get("expected_quote") or ""),
                "evidence_paths": [],
                "timeline_indexes": [],
                "source_hit_id": "preferred-window",
            }
        )
    for query in queries:
        rag = search_video_rag(
            root,
            query=query,
            top_k=top_k,
            ensure_pack=False,
            retrieval_backend=retrieval_backend,
            write=False,
        )
        moment = build_video_moment_index(root, query=query, top_k=top_k, write=False)
        for hit in rag.get("hits") or []:
            if isinstance(hit, dict):
                raw_candidates.append(_normalise_hit(hit, query=query, origin="video_rag_search", cues=cues, slot=slot))
        for hit in moment.get("query_hits") or []:
            if isinstance(hit, dict):
                raw_candidates.append(_normalise_hit(hit, query=query, origin="video_moment_index", cues=cues, slot=slot))

    candidates: list[dict[str, Any]] = []
    raw_candidates.sort(key=_raw_candidate_sort_key)
    for raw in raw_candidates:
        if float(raw.get("end") or 0.0) <= float(raw.get("start") or 0.0):
            continue
        if any(_same_window(raw, existing) for existing in candidates):
            continue
        start = max(0.0, float(raw["start"]))
        end = max(start, float(raw["end"]))
        context_start = max(0.0, start - context_seconds)
        context_end = end + context_seconds
        segments = _segments_for_window(cues, context_start, context_end)
        if not segments:
            continue
        source_transcript_text = _joined_text(segments)
        receipt = export_transcript_reference_window(
            transcript_path,
            root / "exports" / f".{slot_id}-candidate-reference-window.preview.json",
            start_seconds=context_start,
            end_seconds=context_end,
            rebase_timestamps=False,
            validation_scope="window",
            write=False,
        )
        speakers = sorted({str(row.get("speaker") or "") for row in segments if str(row.get("speaker") or "")})
        roles = sorted({str(row.get("speaker_role") or "") for row in segments if str(row.get("speaker_role") or "")})
        roles_by_speaker: dict[str, set[str]] = {}
        for row in segments:
            speaker = str(row.get("speaker") or "").strip()
            role = str(row.get("speaker_role") or "").strip()
            if speaker:
                roles_by_speaker.setdefault(speaker, set())
                if role:
                    roles_by_speaker[speaker].add(role)
        roles_resolved = bool(speakers) and all(len(roles_by_speaker.get(speaker, set())) == 1 for speaker in speakers)
        candidate_id = "script-clip-candidate-" + canonical_json_sha256(
            {"slot_id": slot_id, "start": round(start, 3), "end": round(end, 3), "segments": [row["segment_id"] for row in segments]}
        )[:16]
        candidates.append(
            {
                "candidate_id": candidate_id,
                "slot_id": slot_id,
                "rank": 0,
                "retrieval": {
                    "origin": str(raw.get("origin") or ""),
                    "query": str(raw.get("query") or ""),
                    "score": round(float(raw.get("score") or 0.0), 4),
                    "source_hit_id": str(raw.get("source_hit_id") or ""),
                    "hit_snippet": str(raw.get("snippet") or "")[:700],
                    "coarse_time_range": _time_range(float(raw.get("coarse_start") or start), float(raw.get("coarse_end") or end)),
                },
                "source_time_range": _time_range(start, end),
                "context_time_range": _time_range(context_start, context_end),
                "snippet": source_transcript_text[:700],
                "transcript_segments": segments,
                "source_segment_ids": _unique_text(value for row in segments for value in row.get("source_segment_ids") or []),
                "speaker_evidence": {
                    "anonymous_speaker_ids": speakers,
                    "explicit_roles": roles,
                    "speaker_count": len(speakers),
                    "role_status": "resolved" if roles_resolved else "unresolved",
                    "role_inference_performed": False,
                },
                "evidence_paths": _unique_text(raw.get("evidence_paths") or []),
                "timeline_indexes": [int(value) for value in raw.get("timeline_indexes") or [] if str(value).isdigit()],
                "reference_window_receipt": {
                    "schema": receipt["schema"],
                    "artifact_sha256": receipt["artifact_sha256"],
                    "window": receipt["window"],
                    "segment_count": receipt["segment_count"],
                    "artifact_written": False,
                },
                "review_required": True,
                "publication_allowed": False,
            }
        )
        if len(candidates) >= top_k:
            break
    for rank, row in enumerate(candidates, start=1):
        row["rank"] = rank
    if excluded:
        search_status = "excluded"
        candidates = []
    elif not queries and not preferred:
        search_status = "candidate_not_searched"
    else:
        search_status = "searched_candidates_found" if candidates else "searched_no_candidate"
    return {
        "slot_id": slot_id,
        "state": str(slot.get("state") or "planned"),
        "story_segment_ref": str(slot.get("story_segment_ref") or ""),
        "episode_binding": str(slot.get("episode_binding") or slot.get("story_segment_ref") or ""),
        "required": required,
        "excluded": excluded,
        "expected_quote": str(slot.get("expected_quote") or ""),
        "subtitle_candidate": str(slot.get("subtitle_candidate") or ""),
        "required_speaker_roles": _unique_text(slot.get("required_speaker_roles") or []),
        "excluded_speaker_roles": _unique_text(slot.get("excluded_speaker_roles") or []),
        "search_queries": queries,
        "search_status": search_status,
        "candidates": candidates,
    }


def _normalise_hit(hit: dict[str, Any], *, query: str, origin: str, cues: list[Any], slot: dict[str, Any]) -> dict[str, Any]:
    coarse_start = float(hit.get("start") or 0.0)
    coarse_end = float(hit.get("end") or coarse_start)
    target = str(slot.get("expected_quote") or query)
    start, end = _refine_time_range(cues, coarse_start, coarse_end, target)
    return {
        "origin": origin,
        "query": query,
        "score": float(hit.get("score") or 0.0),
        "start": start,
        "end": end,
        "coarse_start": coarse_start,
        "coarse_end": coarse_end,
        "snippet": str(hit.get("snippet") or ""),
        "evidence_paths": hit.get("evidence_paths") if isinstance(hit.get("evidence_paths"), list) else [],
        "timeline_indexes": hit.get("timeline_indexes") if isinstance(hit.get("timeline_indexes"), list) else [],
        "source_hit_id": str(hit.get("chunk_id") or hit.get("chunk_index") or ""),
    }


def _refine_time_range(cues: list[Any], coarse_start: float, coarse_end: float, target: str) -> tuple[float, float]:
    bounded = [cue for cue in cues if float(cue.end) >= coarse_start and float(cue.start) <= coarse_end]
    if not bounded:
        return coarse_start, coarse_end
    cleaned_target = _normalise_text(target)
    ranked: list[tuple[float, Any]] = []
    for cue in bounded:
        score = fuzz.partial_ratio(cleaned_target, _normalise_text(cue.text)) if cleaned_target else 0.0
        ranked.append((float(score), cue))
    ranked.sort(key=lambda row: (-row[0], float(row[1].start)))
    best = ranked[0][1]
    start = max(coarse_start, float(best.start) - 1.5)
    end = min(coarse_end, max(float(best.end), float(best.start) + 1.0) + 1.5)
    return (start, end)


def _segments_for_window(cues: list[Any], start: float, end: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, cue in enumerate(cues, start=1):
        cue_start = float(cue.start)
        cue_end = max(cue_start, float(cue.end))
        if cue_end <= start or cue_start >= end:
            continue
        segment_id = str(cue.segment_id or f"segment-{index:06d}")
        rows.append(
            {
                "segment_id": segment_id,
                "source_segment_ids": [str(value) for value in (cue.source_segment_ids or [segment_id]) if str(value)],
                "start": round(cue_start, 6),
                "end": round(cue_end, 6),
                "text": str(cue.text or ""),
                "speaker": str(cue.speaker or ""),
                "speaker_role": str(cue.speaker_role or ""),
            }
        )
    return rows


def _review_template(pack: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": REVIEW_SCHEMA,
        "review_status": "draft",
        "review_id": "",
        "candidate_pack": {
            "path": PACK_PATH,
            "pack_sha256": pack["pack_sha256"],
            "artifact_sha256": "",
        },
        "slots": [
            {
                "slot_id": row["slot_id"],
                "episode_binding": row["episode_binding"],
                "selected_candidate_id": "",
                "fine_cut_order": None,
                "fine_cut_output": "",
                "clip_transcript_path": "",
                "clip_transcript_sha256": "",
                "approved_quote": row["expected_quote"],
                "approved_clip_text": "",
                "approved_window": None,
                "approved_source_segment_ids": [],
                "expected_speaker_role": "",
                "expected_speaker_ids": [],
                "excluded_speaker_roles": row["excluded_speaker_roles"],
                "excluded_speaker_ids": [],
                "label": "customer_quote" if row["required"] and not row["excluded"] else "excluded",
                "subtitle_text": row["subtitle_candidate"],
                "human_confirmed": False,
                "notes": "",
            }
            for row in pack["slots"]
        ],
        "operator_boundary": {
            **_operator_boundary(),
            "draft_is_not_formal_approval": True,
            "human_confirmed_required_for_alignment": True,
        },
    }


def _validate_request_artifacts(request: dict[str, Any], *, request_path: Path) -> None:
    for key in ("script", "upstream_handoff"):
        ref = request.get(key)
        if not isinstance(ref, dict) or not ref:
            continue
        path = _resolve_declared_path(ref.get("path"), request_path.parent)
        if not path or not path.is_file():
            raise FileNotFoundError(f"{key} artifact not found: {path}")
        expected = str(ref.get("sha256") or "").lower()
        if expected and sha256_file(path).lower() != expected:
            raise ValueError(f"{key} artifact SHA-256 changed: {path}")


def _validate_request_slots(request: dict[str, Any]) -> None:
    seen: set[str] = set()
    for slot in request.get("slots") or []:
        slot_id = str(slot.get("slot_id") or "").strip()
        if slot_id in seen:
            raise ValueError(f"script clip request contains duplicate slot_id: {slot_id}")
        seen.add(slot_id)
        window = slot.get("preferred_window")
        if isinstance(window, dict) and float(window.get("end") or 0.0) <= float(window.get("start") or 0.0):
            raise ValueError(f"script clip preferred_window must have end > start: {slot_id}")


def _resolve_transcript_path(root: Path, manifest: dict[str, Any], request: dict[str, Any], request_path: Path) -> Path:
    declared = request.get("source_transcript") if isinstance(request.get("source_transcript"), dict) else {}
    declared_path = _resolve_declared_path(declared.get("path"), request_path.parent)
    candidates: list[Path] = []
    if declared_path:
        candidates.append(declared_path)
    for key in ("human_corrected_transcript_json", "corrected_transcript", "normalized_transcript", "knowledge_note_transcript_markdown"):
        value = str(manifest.get(key) or "").strip()
        if value:
            path = Path(value).expanduser()
            candidates.append((root / path).resolve() if not path.is_absolute() else path.resolve())
    candidates.extend(
        [
            root / "corrected-transcript.json",
            root / "normalized-transcript.json",
            root / "exports-final" / "full-transcript.md",
            root / "exports" / "full-transcript.md",
        ]
    )
    transcript = next((path.resolve() for path in candidates if path.is_file()), None)
    if transcript is None:
        raise FileNotFoundError("no canonical/corrected transcript was found for script clip retrieval")
    expected = str(declared.get("sha256") or "").lower()
    if expected and sha256_file(transcript).lower() != expected:
        raise ValueError(f"source transcript SHA-256 changed: {transcript}")
    return transcript


def _validate_schema(payload: dict[str, Any], filename: str) -> None:
    schema_path = Path(__file__).with_name("schemas") / filename
    schema = read_json(schema_path)
    jsonschema.validate(payload, schema)


def _artifact_ref(path: Path, *, role: str) -> dict[str, Any]:
    return {"role": role, "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _resolve_declared_path(value: Any, base: Path) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _payload_sha(payload: dict[str, Any]) -> str:
    return canonical_json_sha256({key: value for key, value in payload.items() if key != "pack_sha256"})


def _time_range(start: float, end: float) -> dict[str, Any]:
    return {
        "start": round(max(0.0, start), 6),
        "end": round(max(max(0.0, start), end), 6),
        "start_time": format_timestamp(start),
        "end_time": format_timestamp(end),
    }


def _same_window(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_start, left_end = float(left.get("start") or 0.0), float(left.get("end") or 0.0)
    right_range = right.get("source_time_range") if isinstance(right.get("source_time_range"), dict) else right
    right_start, right_end = float(right_range.get("start") or 0.0), float(right_range.get("end") or 0.0)
    overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    shorter = max(0.001, min(left_end - left_start, right_end - right_start))
    return overlap / shorter >= 0.8


def _raw_candidate_sort_key(row: dict[str, Any]) -> tuple[int, float, float]:
    # Scores from VideoRAG and the moment index are not calibrated to the same
    # scale.  An explicit, hash-bound script time window is therefore reviewed
    # first; local retrieval results remain ordered within their own evidence
    # tier instead of competing on incomparable raw scores.
    priority = {
        "script_preferred_window": 0,
        "video_rag_search": 1,
        "video_moment_index": 2,
    }.get(str(row.get("origin") or ""), 3)
    return (priority, -float(row.get("score") or 0.0), float(row.get("start") or 0.0))


def _joined_text(rows: list[dict[str, Any]]) -> str:
    return " ".join(str(row.get("text") or "").strip() for row in rows if str(row.get("text") or "").strip())


def _normalise_text(value: Any) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(value or "").lower())


def _unique_text(values: Any) -> list[str]:
    result: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _operator_boundary() -> dict[str, Any]:
    return {
        "local_only": True,
        "review_only": True,
        "publication_allowed": False,
        "external_provider_called": False,
        "media_uploaded": False,
        "identity_inference_performed": False,
        "canonical_transcript_mutated": False,
        "timeline_mutated": False,
        "source_media_mutated": False,
        "upstream_script_mutated": False,
        "automatic_candidate_approval": False,
    }


def _render_pack_markdown(pack: dict[str, Any]) -> str:
    lines = [
        "# 脚本驱动的采访片段候选包",
        "",
        f"- Request: `{pack.get('request_id', '')}`",
        f"- Status: `{pack.get('status', '')}`",
        f"- Slots: `{pack.get('summary', {}).get('slot_count', 0)}`",
        f"- Candidates: `{pack.get('summary', {}).get('candidate_count', 0)}`",
        "- 边界：仅本地检索候选；必须人工选择；不得据此发布。",
        "",
    ]
    for slot in pack.get("slots") or []:
        lines.extend(
            [
                f"## {slot.get('slot_id')} · {slot.get('story_segment_ref') or '-'}",
                "",
                f"- Search status: `{slot.get('search_status')}`",
                f"- Required: `{str(bool(slot.get('required'))).lower()}`",
                f"- Expected quote: {slot.get('expected_quote') or '(未提供)' }",
                "",
            ]
        )
        for candidate in slot.get("candidates") or []:
            time_range = candidate.get("source_time_range") or {}
            speakers = candidate.get("speaker_evidence") or {}
            lines.extend(
                [
                    f"### #{candidate.get('rank')} `{candidate.get('candidate_id')}`",
                    "",
                    f"- 时间：`{time_range.get('start_time')}` — `{time_range.get('end_time')}`",
                    f"- 来源：`{candidate.get('retrieval', {}).get('origin')}`",
                    f"- 分数：`{candidate.get('retrieval', {}).get('score')}`",
                    f"- 匿名说话人：`{', '.join(speakers.get('anonymous_speaker_ids') or []) or 'unknown'}`",
                    f"- 角色状态：`{speakers.get('role_status')}`（不自动推断真人角色）",
                    "",
                    str(candidate.get("snippet") or ""),
                    "",
                ]
            )
        if not slot.get("candidates"):
            lines.extend(["没有候选；请补检索词或明确时间窗。", ""])
    return "\n".join(lines).rstrip() + "\n"
