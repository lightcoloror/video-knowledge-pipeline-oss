from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from typing import Any

from .models import now_iso
from .external_reuse_run_artifacts import ps_quote, register_external_reuse_run
from .storage import read_json, write_json, write_text_atomic
from .storage import read_json_object_or_empty
from .long_video_memory_pack import build_long_video_memory_pack
from .video_moment_index import build_video_moment_index, _moment_observations
from .evidence_locator import JsonEvidenceSnapshot, verify_evidence_locators
from .canonical_json import canonical_json_sha256

SCHEMA = "video_knowledge_pipeline.video_rag_pack.v1"
OBSERVATION_SCHEMA = "video_knowledge_pipeline.rag_observation_projection.v1"


def build_video_rag_pack(
    bundle_dir: str | Path,
    *,
    query: str = "",
    target_window_seconds: float = 300.0,
    max_chunk_chars: int = 3600,
    top_k: int = 8,
    write: bool = True,
) -> dict[str, Any]:
    """Build a local VideoRAG-style evidence package from VKP moments.

    This intentionally does not introduce vector DB or graph dependencies. It
    creates stable JSONL retrieval units with transcript, visual text, temporal
    understanding, timestamps, and evidence paths so another local RAG layer can
    index them later without re-reading the whole bundle.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")

    moment_index = build_video_moment_index(
        root,
        query=query,
        target_window_seconds=target_window_seconds,
        max_chunk_chars=max_chunk_chars,
        top_k=top_k,
        write=write,
    )
    moment_chunks = [_rag_chunk(chunk, root=root) for chunk in moment_index.get("chunks") or [] if isinstance(chunk, dict)]
    supplemental_chunks = _supplemental_chunks(
        root,
        manifest=manifest,
        target_window_seconds=target_window_seconds,
        max_chunk_chars=max_chunk_chars,
        write=write,
    )
    chunks = [*moment_chunks, *supplemental_chunks]
    for chunk in chunks:
        metadata = chunk['metadata']
        if 'source_observations' in metadata:
            metadata['observation_schema'] = OBSERVATION_SCHEMA
            metadata['retrieval_text_sha256'] = canonical_json_sha256(chunk['text'])
    observation_admission = _validate_observation_sources(root, chunks)
    retrieved = _retrieved(moment_index, chunks, top_k=top_k)
    exports = root / "exports"
    json_path = exports / "video-rag-pack.json"
    markdown_path = exports / "video-rag-pack.md"
    chunks_jsonl_path = exports / "video-rag-chunks.jsonl"
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "title": str(manifest.get("title") or root.name),
        "created_at": now_iso(),
        "inspired_by": [
            "VideoRAG: segment retrieval with transcript plus visual captions",
            "VTimeLLM: timestamps as first-class retrieval metadata",
        ],
        "parameters": {
            "query": query,
            "target_window_seconds": target_window_seconds,
            "max_chunk_chars": max_chunk_chars,
            "top_k": top_k,
        },
        "summary": {
            "chunks": len(chunks),
            "moment_chunks": len(moment_chunks),
            "visual_evidence_chunks": sum(1 for row in chunks if row["metadata"].get("chunk_kind") == "visual_evidence"),
            "review_gap_chunks": sum(1 for row in chunks if row["metadata"].get("chunk_kind") == "review_gap"),
            "content_asset_chunks": sum(1 for row in chunks if row["metadata"].get("chunk_kind") == "content_asset"),
            "content_candidate_chunks": sum(1 for row in chunks if row["metadata"].get("chunk_kind") == "content_candidate"),
            "short_memory_chunks": sum(1 for row in chunks if row["metadata"].get("chunk_kind") == "short_memory"),
            "chapter_memory_chunks": sum(1 for row in chunks if row["metadata"].get("chunk_kind") == "chapter_memory"),
            "theme_memory_chunks": sum(1 for row in chunks if row["metadata"].get("chunk_kind") == "theme_memory"),
            "duration_seconds": moment_index.get("summary", {}).get("duration_seconds", 0.0),
            "chunks_with_visual_evidence": sum(1 for row in chunks if row["metadata"]["has_visual_evidence"]),
            "chunks_with_temporal_evidence": sum(1 for row in chunks if row["metadata"]["has_temporal_evidence"]),
            "chunks_by_kind": _chunks_by_kind(chunks),
            "retrieved": len(retrieved),
        },
        "retrieval_units": chunks,
        "observation_admission": observation_admission,
        "retrieved_moments": retrieved,
        "artifacts": {
            "json": str(json_path),
            "markdown": str(markdown_path),
            "chunks_jsonl": str(chunks_jsonl_path),
        },
        "operator_boundary": {
            "local_only": True,
            "no_vector_backend_started": True,
            "no_cloud_model_call": True,
            "evidence_paths_required": True,
            "multi_granularity_jsonl": True,
        },
        "write": bool(write),
    }
    if write:
        exports.mkdir(parents=True, exist_ok=True)
        write_json(json_path, result)
        write_text_atomic(
            chunks_jsonl_path,
            "\n".join(json.dumps(row, ensure_ascii=False) for row in chunks) + ("\n" if chunks else ""),
        )
        write_text_atomic(markdown_path, _render_markdown(result))
        manifest["video_rag_pack"] = "exports/video-rag-pack.json"
        manifest["video_rag_pack_markdown"] = "exports/video-rag-pack.md"
        manifest["video_rag_chunks_jsonl"] = "exports/video-rag-chunks.jsonl"
        manifest["mcp_video_rag_pack_args"] = "mcp-video-rag-pack.args.json"
        write_json(
            root / "mcp-video-rag-pack.args.json",
            {
                "bundle_dir": str(root),
                "query": query,
                "target_window_seconds": target_window_seconds,
                "max_chunk_chars": max_chunk_chars,
                "top_k": top_k,
                "write": True,
            },
        )
        write_json(manifest_path, manifest)
        failed_items = [] if chunks else [{"id": "video_rag_chunks", "reason": "no_retrieval_units", "detail": "VideoRAG pack produced no retrieval units."}]
        register_external_reuse_run(
            root,
            run_type="video_rag_pack",
            title="VideoRAG pack",
            result=result,
            status="needs_input" if failed_items else "completed",
            failed_items=failed_items,
            retry_command=f".\\scripts\\video-knowledge.ps1 video-rag-pack {ps_quote(root)} --query \"<问题或术语>\"",
            next_actions=[] if not failed_items else ["Build timeline/moment index, then rerun video-rag-pack."],
            write=True,
        )
    return result


def _rag_chunk(chunk: dict[str, Any], *, root: Path) -> dict[str, Any]:
    chunk_index = int(chunk.get("chunk_index") or 0)
    transcript = _text(chunk.get("transcript_text"))
    observations = [row for rows in chunk.get('source_observations_by_kind', {}).values() for row in rows]
    visual = _clip(_visual_evidence_text([row for row in observations
        if row['source_kind'] == 'visual' and row['fact_status'] != 'review_gap_not_fact']), 8000)
    temporal = _clip(_visual_evidence_text([row for row in observations
        if row['source_kind'] == 'temporal_visual' and row['fact_status'] != 'review_gap_not_fact']), 6000)
    return {
        "id": f"{root.name}:moment:{chunk_index:04d}",
        "text": _join_nonempty(
            [
                f"Transcript: {transcript}" if transcript else "",
                f"Visual: {visual}" if visual else "",
                f"Temporal: {temporal}" if temporal else "",
            ]
        ),
        "metadata": {
            "source_observations": observations,
            "display_scope": "retrieval_excerpt_not_complete_source",
            "source_locators_by_kind": chunk.get('source_locators_by_kind', {}),
            "evidence_scope": "derived_context_not_independent_source",
            "chunk_kind": "moment",
            "chunk_index": chunk_index,
            "start": float(chunk.get("start") or 0.0),
            "end": float(chunk.get("end") or 0.0),
            "start_time": str(chunk.get("start_time") or ""),
            "end_time": str(chunk.get("end_time") or ""),
            "timeline_indexes": chunk.get("timeline_indexes") if isinstance(chunk.get("timeline_indexes"), list) else [],
            "visual_routes": chunk.get("visual_routes") if isinstance(chunk.get("visual_routes"), dict) else {},
            "tags": chunk.get("tags") if isinstance(chunk.get("tags"), list) else [],
            "keywords": chunk.get("keywords") if isinstance(chunk.get("keywords"), list) else [],
            "has_visual_evidence": bool(visual),
            "has_temporal_evidence": bool(temporal),
            "evidence_paths": chunk.get("evidence_paths") if isinstance(chunk.get("evidence_paths"), list) else [],
        },
    }



def _supplemental_chunks(
    root: Path,
    *,
    manifest: dict[str, Any],
    target_window_seconds: float,
    max_chunk_chars: int,
    write: bool,
) -> list[dict[str, Any]]:
    snapshot = JsonEvidenceSnapshot.read(root, 'timeline.json')
    if not isinstance(snapshot.data, list):
        raise ValueError('timeline.json must be a JSON array')
    timeline = snapshot.data
    index_counts = Counter(_int(item.get('index')) for item in timeline if isinstance(item, dict))
    chunks: list[dict[str, Any]] = []
    for position, item in enumerate(timeline):
        if not isinstance(item, dict):
            continue
        observations = _moment_observations(item, _int(item.get('index')), snapshot=snapshot,
                                            source_position=position)
        suffix = f':array-{position}' if index_counts[_int(item.get('index'))] > 1 else ''
        chunks.extend(_timeline_item_chunks(root, item, observations=observations, identity_suffix=suffix))
    chunks.extend(
        _long_video_memory_chunks(
            root,
            manifest=manifest,
            target_window_seconds=target_window_seconds,
            max_chunk_chars=max_chunk_chars,
            write=write,
        )
    )
    chunks.extend(_content_candidate_chunks(root, manifest=manifest))
    chunks.extend(_content_asset_chunks(root, manifest=manifest))
    return chunks

def _load_timeline(root: Path) -> list[dict[str, Any]]:
    path = root / "timeline.json"
    if not path.exists():
        return []
    try:
        data = read_json(path)
    except Exception:
        return []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def _timeline_item_chunks(root: Path, item: dict[str, Any], *, observations: list[dict],
                          identity_suffix: str = '') -> list[dict[str, Any]]:
    index = _int(item.get("index"))
    if index <= 0:
        return []
    chunks: list[dict[str, Any]] = []
    admitted = [row for row in observations if row['fact_status'] != 'review_gap_not_fact']
    gaps = [row for row in observations if row['fact_status'] == 'review_gap_not_fact']
    visual_text = _visual_evidence_text(admitted)
    evidence_paths = _evidence_paths(item)
    start = _float(item.get("start"))
    end = _float(item.get("end"))
    if visual_text:
        chunks.append(
            _make_chunk(
                root,
                kind="visual_evidence",
                key=f"timeline:{index:04d}:visual{identity_suffix}",
                text=visual_text,
                start=start,
                end=end,
                timeline_indexes=[index],
                evidence_paths=evidence_paths,
                tags=[*_string_list(item.get("tags")), "visual_evidence"],
                keywords=_keywords_from_text(visual_text),
                has_visual=True,
                has_temporal=any(r['source_kind'] == 'temporal_visual' for r in admitted),
                extra_metadata={'source_observations': admitted,
                                'evidence_scope': 'derived_context_not_independent_source'},
            )
        )
    gap_text = _review_gap_text(item)
    if gaps:
        gap_text = _join_nonempty([gap_text, 'Observed but unavailable or truncated; not a visual fact.'])
    if gap_text:
        chunks.append(
            _make_chunk(
                root,
                kind="review_gap",
                key=f"timeline:{index:04d}:review-gap{identity_suffix}",
                text=gap_text,
                start=start,
                end=end,
                timeline_indexes=[index],
                evidence_paths=evidence_paths,
                tags=[*_string_list(item.get("tags")), "review_gap"],
                keywords=_keywords_from_text(gap_text),
                has_visual=bool(evidence_paths),
                has_temporal=False,
                extra_metadata={'source_observations': gaps, 'fact_status': 'review_gap_not_fact'},
            )
        )
    return chunks


def _make_chunk(
    root: Path,
    *,
    kind: str,
    key: str,
    text: str,
    start: float = 0.0,
    end: float = 0.0,
    timeline_indexes: list[int] | None = None,
    evidence_paths: list[str] | None = None,
    tags: list[str] | None = None,
    keywords: list[str] | None = None,
    has_visual: bool = False,
    has_temporal: bool = False,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    safe_key = key.replace(" ", "-").replace(":", ":")
    metadata = {
        "chunk_kind": kind,
        "start": start,
        "end": end,
        "start_time": _timestamp(start),
        "end_time": _timestamp(end),
        "timeline_indexes": timeline_indexes or [],
        "visual_routes": {},
        "tags": _dedupe(tags or []),
        "keywords": _dedupe(keywords or []),
        "has_visual_evidence": bool(has_visual),
        "has_temporal_evidence": bool(has_temporal),
        "evidence_paths": _dedupe(evidence_paths or []),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "id": f"{root.name}:{kind}:{safe_key}",
        "text": _text(text),
        "metadata": metadata,
    }

def _long_video_memory_chunks(
    root: Path,
    *,
    manifest: dict[str, Any],
    target_window_seconds: float,
    max_chunk_chars: int,
    write: bool,
) -> list[dict[str, Any]]:
    memory_pack = _load_or_build_long_video_memory_pack(
        root,
        manifest=manifest,
        target_window_seconds=target_window_seconds,
        max_chunk_chars=max_chunk_chars,
        write=write,
    )
    if not memory_pack:
        return []
    short_memories = [row for row in memory_pack.get("short_memories") or [] if isinstance(row, dict)]
    long_memories = [row for row in memory_pack.get("long_memories") or [] if isinstance(row, dict)]
    short_by_id = {str(row.get("memory_id") or ""): row for row in short_memories}
    chunks: list[dict[str, Any]] = []

    for row in short_memories:
        memory_id = str(row.get("memory_id") or "").strip()
        text = _join_nonempty(
            [
                f"Short memory {memory_id}: {row.get('topic_hint') or ''}",
                f"Summary: {'; '.join(_string_list(row.get('summary_bullets')))}",
                f"Visual notes: {'; '.join(_string_list(row.get('visual_bullets')))}",
                f"Keywords: {', '.join(_string_list(row.get('keywords')))}",
            ]
        )
        if not text:
            continue
        chunk_index = _int(row.get("chunk_index"))
        chunks.append(
            _make_chunk(
                root,
                kind="short_memory",
                key=memory_id or f"short:{len(chunks) + 1:04d}",
                text=text,
                start=_float(row.get("start")),
                end=_float(row.get("end")),
                timeline_indexes=[_int(value) for value in row.get("timeline_indexes") or [] if _int(value) > 0],
                evidence_paths=_string_list(row.get("evidence_paths")),
                tags=["long_video_memory", "short_memory"],
                keywords=[*_string_list(row.get("keywords")), str(row.get("topic_hint") or "")],
                has_visual=bool(row.get("has_visual_evidence")),
                has_temporal=bool(row.get("has_temporal_evidence")),
                extra_metadata={
                    "memory_level": "short",
                    "memory_id": memory_id,
                    "parent_memory_id": "",
                    "child_memory_ids": [],
                    "child_moment_indexes": [chunk_index] if chunk_index > 0 else [],
                },
            )
        )

    for row in long_memories:
        memory_id = str(row.get("long_memory_id") or "").strip()
        child_ids = _string_list(row.get("short_memory_ids"))
        child_rows = [short_by_id[value] for value in child_ids if value in short_by_id]
        timeline_indexes = _dedupe_ints(
            [
                _int(value)
                for child in child_rows
                for value in (child.get("timeline_indexes") or [])
                if _int(value) > 0
            ]
        )
        child_moment_indexes = _dedupe_ints(
            [_int(child.get("chunk_index")) for child in child_rows if _int(child.get("chunk_index")) > 0]
        )
        evidence_paths = _dedupe(
            [path for child in child_rows for path in _string_list(child.get("evidence_paths"))]
        )
        chapter_text = _join_nonempty(
            [
                f"Chapter memory {memory_id}: {row.get('topic_hint') or ''}",
                f"Key points: {'; '.join(_string_list(row.get('merged_bullets')))}",
                f"Keywords: {', '.join(_string_list(row.get('keywords')))}",
                f"Child short memories: {', '.join(child_ids)}",
            ]
        )
        if chapter_text:
            chunks.append(
                _make_chunk(
                    root,
                    kind="chapter_memory",
                    key=memory_id or f"chapter:{len(chunks) + 1:04d}",
                    text=chapter_text,
                    start=_float(row.get("start")),
                    end=_float(row.get("end")),
                    timeline_indexes=timeline_indexes,
                    evidence_paths=evidence_paths,
                    tags=["long_video_memory", "chapter_memory"],
                    keywords=[*_string_list(row.get("keywords")), str(row.get("topic_hint") or "")],
                    has_visual=_int(row.get("visual_evidence_count")) > 0,
                    has_temporal=_int(row.get("temporal_evidence_count")) > 0,
                    extra_metadata={
                        "memory_level": "chapter",
                        "memory_id": memory_id,
                        "child_memory_ids": child_ids,
                        "child_moment_indexes": child_moment_indexes,
                        "memory_child_count": len(child_ids),
                    },
                )
            )
        theme_text = _join_nonempty(
            [
                f"Theme memory {memory_id}: {row.get('topic_hint') or ''}",
                f"Course-flow points: {'; '.join(_string_list(row.get('merged_bullets'))[:4])}",
            ]
        )
        if theme_text:
            chunks.append(
                _make_chunk(
                    root,
                    kind="theme_memory",
                    key=f"{memory_id}:theme" if memory_id else f"theme:{len(chunks) + 1:04d}",
                    text=theme_text,
                    start=_float(row.get("start")),
                    end=_float(row.get("end")),
                    timeline_indexes=timeline_indexes,
                    evidence_paths=evidence_paths,
                    tags=["long_video_memory", "theme_memory"],
                    keywords=[*_string_list(row.get("keywords")), str(row.get("topic_hint") or "")],
                    has_visual=_int(row.get("visual_evidence_count")) > 0,
                    has_temporal=_int(row.get("temporal_evidence_count")) > 0,
                    extra_metadata={
                        "memory_level": "theme",
                        "memory_id": f"{memory_id}:theme" if memory_id else "",
                        "parent_memory_id": memory_id,
                        "child_memory_ids": child_ids,
                        "child_moment_indexes": child_moment_indexes,
                    },
                )
            )

    final_map = memory_pack.get("final_memory_map") if isinstance(memory_pack.get("final_memory_map"), dict) else {}
    for row in final_map.get("low_confidence_boundaries") or []:
        if not isinstance(row, dict):
            continue
        memory_id = str(row.get("memory_id") or "").strip()
        short = short_by_id.get(memory_id, {})
        chunk_index = _int(short.get("chunk_index"))
        chunks.append(
            _make_chunk(
                root,
                kind="memory_boundary",
                key=memory_id or f"boundary:{len(chunks) + 1:04d}",
                text=_join_nonempty(
                    [
                        f"Memory boundary {memory_id}",
                        f"Time: {row.get('time') or ''}",
                        f"Reason: {row.get('reason') or ''}",
                    ]
                ),
                start=_float(short.get("start")),
                end=_float(short.get("end")),
                timeline_indexes=[_int(value) for value in short.get("timeline_indexes") or [] if _int(value) > 0],
                evidence_paths=_string_list(short.get("evidence_paths")),
                tags=["long_video_memory", "memory_boundary", "review_gap"],
                keywords=[str(row.get("reason") or ""), memory_id],
                has_visual=False,
                has_temporal=False,
                extra_metadata={
                    "memory_level": "boundary",
                    "memory_id": memory_id,
                    "fact_status": "memory_boundary_review_gap",
                    "child_moment_indexes": [chunk_index] if chunk_index > 0 else [],
                },
            )
        )
    return chunks


def _load_or_build_long_video_memory_pack(
    root: Path,
    *,
    manifest: dict[str, Any],
    target_window_seconds: float,
    max_chunk_chars: int,
    write: bool,
) -> dict[str, Any]:
    raw = str(manifest.get("long_video_memory_pack") or "").strip()
    candidates = [_resolve_bundle_path(root, raw)] if raw else []
    candidates.append(root / "exports" / "long-video-memory-pack.json")
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = read_json(path)
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    try:
        return build_long_video_memory_pack(
            root,
            target_window_seconds=target_window_seconds,
            max_chunk_chars=max_chunk_chars,
            write=write,
        )
    except Exception:
        return {}

def _visual_evidence_text(observations: list[dict]) -> str:
    """Render the admitted observations; never reinterpret raw model payloads."""
    return _join_nonempty([f"{row['observation_kind']} [{row['evidence_level']}]: {row['snippet']}"
                          for row in observations])


def _validate_observation_sources(root: Path, chunks: list[dict]) -> dict:
    # Reuse final-summary admission instead of a second provenance verifier.
    from .smart_summary_global_reduce import _dedupe_evidence_refs
    from .summary_citation_audit import build_citation_index
    for chunk in chunks:
        metadata = chunk.get('metadata', {})
        if 'source_observations' in metadata or 'observation_schema' in metadata:
            if (metadata.get('observation_schema') != OBSERVATION_SCHEMA or
                    not isinstance(metadata.get('source_observations'), list)):
                raise ValueError('invalid_rag_observation_schema')
            if metadata.get('retrieval_text_sha256') != canonical_json_sha256(chunk.get('text')):
                raise ValueError('rag_retrieval_text_identity_drift')
    refs = _dedupe_evidence_refs([ref for chunk in chunks
        for ref in chunk.get('metadata', {}).get('source_observations', [])])
    index = build_citation_index({'sections': [{'evidence_refs': refs}]})
    if index.errors:
        raise ValueError('invalid_rag_observation:' + ';'.join(index.errors))
    verification = verify_evidence_locators(root, [loc for ref in refs for loc in ref['source_locators']])
    return dict(verification, observation_count=len(refs),
                evidence_scope='timeline_visual_observations_only',
                other_derived_content='not_assessed', independent_source_count=0,
                legacy_or_other_chunk_count=sum('observation_schema' not in r.get('metadata', {}) for r in chunks))


def _review_gap_text(item: dict[str, Any]) -> str:
    issues = _string_list(item.get("quality_issues"))
    tile_targets = item.get("tile_review_targets") if isinstance(item.get("tile_review_targets"), list) else []
    needs_review = bool(item.get("needs_human_review"))
    status = str(item.get("review_status") or "").strip()
    if not issues and not tile_targets and not needs_review:
        return ""
    parts = [
        f"Review status: {status}" if status else "",
        f"Quality issues: {', '.join(issues)}" if issues else "",
        f"Needs human review: {needs_review}" if needs_review else "",
        f"Transcript: {_text(item.get('corrected_transcript') or item.get('transcript'))}",
    ]
    for target in tile_targets:
        if isinstance(target, dict):
            parts.append(
                "Tile target: "
                + "; ".join(
                    value
                    for value in [
                        f"tile_id={target.get('tile_id')}" if target.get("tile_id") else "",
                        f"confidence={target.get('confidence')}" if target.get("confidence") is not None else "",
                        f"reasons={','.join(_string_list(target.get('reasons')))}" if target.get("reasons") else "",
                        f"evidence={target.get('evidence_path') or target.get('tile_path') or ''}" if (target.get("evidence_path") or target.get("tile_path")) else "",
                    ]
                    if value
                )
            )
    return _join_nonempty(parts)


def _content_candidate_chunks(root: Path, *, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    pack = _load_content_candidate_pack(root, manifest)
    candidates = [row for row in pack.get("candidates") or [] if isinstance(row, dict)] if pack else []
    pack_term_snapshot = _term_snapshot_from_payload(pack)
    chunks: list[dict[str, Any]] = []
    for position, row in enumerate(candidates, start=1):
        candidate_id = str(row.get("id") or f"candidate-{position:03d}").strip()
        timeline_index = _int(row.get("timeline_index"))
        start = _float(row.get("start"))
        end = _float(row.get("end"))
        candidate_types = _string_list(row.get("candidate_types"))
        citations = row.get("evidence_citations") if isinstance(row.get("evidence_citations"), list) else []
        citation_text = _content_candidate_citation_text(citations)
        evidence_paths = _dedupe([*_string_list(row.get("evidence_paths")), *_content_candidate_citation_paths(citations)])
        summary_chapter_refs = row.get("summary_chapter_refs") if isinstance(row.get("summary_chapter_refs"), list) else []
        term_snapshot = _term_snapshot_from_payload(row) or pack_term_snapshot
        text = _join_nonempty(
            [
                f"Content candidate {candidate_id}",
                f"Types: {', '.join(candidate_types)}" if candidate_types else "",
                f"Viewpoint: {row.get('viewpoint') or ''}",
                f"Case or example: {row.get('case_or_example') or ''}",
                f"Reusable quote: {row.get('reusable_quote') or ''}",
                f"Fact status: {row.get('fact_status') or ''}",
                f"Term validation: {term_snapshot.get('term_validation_status') or 'missing'}" if term_snapshot else "",
                f"Citation evidence: {citation_text}" if citation_text else "",
            ]
        )
        if not text:
            continue
        chunks.append(
            _make_chunk(
                root,
                kind="content_candidate",
                key=candidate_id,
                text=text,
                start=start,
                end=end,
                timeline_indexes=[timeline_index] if timeline_index > 0 else [],
                evidence_paths=evidence_paths,
                tags=["content_candidate", *candidate_types],
                keywords=_keywords_from_text(" ".join([text, " ".join(candidate_types)])),
                has_visual=bool(evidence_paths or citations),
                has_temporal=False,
                extra_metadata={
                    "candidate_id": candidate_id,
                    "candidate_types": candidate_types,
                    "fact_status": str(row.get("fact_status") or "needs_review"),
                    "review_required": bool(row.get("review_required", True)),
                    "publication_allowed": bool(row.get("publication_allowed")),
                    "allowed_as_fact": bool(row.get("allowed_as_fact")),
                    "citation_count": len(citations),
                    "summary_chapter_refs": summary_chapter_refs,
                    "summary_chapter_ref_count": len(summary_chapter_refs),
                    "content_candidate_pack_path": str(_content_candidate_pack_path(root, manifest) or ""),
                    **_term_metadata(term_snapshot),
                },
            )
        )
    return chunks



def _term_snapshot_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("term_correction") if isinstance(payload.get("term_correction"), dict) else payload
    status = str(raw.get("status") or raw.get("term_correction_status") or "").strip()
    validation = str(raw.get("term_validation_status") or "").strip()
    if not status and not validation:
        return {}
    return {
        "term_correction_status": status or "missing",
        "term_validation_status": validation or "missing",
        "accepted_validation_decisions": _int(raw.get("accepted_validation_decisions")),
        "rejected_validation_decisions": _int(raw.get("rejected_validation_decisions")),
        "accepted_term_count": _int(raw.get("accepted_term_count")),
        "source_arbitrated_transcript_exists": bool(raw.get("source_arbitrated_transcript_exists")),
        "final_export_alias_total": _int(raw.get("final_export_alias_total")),
    }


def _term_snapshot_from_asset(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        return {}
    try:
        value = read_json(path)
    except Exception:
        return {}
    return _term_snapshot_from_payload(value) if isinstance(value, dict) else {}


def _term_metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
    if not snapshot:
        return {}
    return {
        "term_correction_status": snapshot.get("term_correction_status", "missing"),
        "term_validation_status": snapshot.get("term_validation_status", "missing"),
        "accepted_validation_decisions": int(snapshot.get("accepted_validation_decisions") or 0),
        "rejected_validation_decisions": int(snapshot.get("rejected_validation_decisions") or 0),
        "accepted_term_count": int(snapshot.get("accepted_term_count") or 0),
        "source_arbitrated_transcript_exists": bool(snapshot.get("source_arbitrated_transcript_exists")),
        "final_export_alias_total": int(snapshot.get("final_export_alias_total") or 0),
    }

def _load_content_candidate_pack(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = _content_candidate_pack_path(root, manifest)
    if path is None or not path.exists():
        return {}
    return read_json_object_or_empty(path)


def _content_candidate_pack_path(root: Path, manifest: dict[str, Any]) -> Path | None:
    assets = manifest.get("content_assets") if isinstance(manifest.get("content_assets"), dict) else {}
    candidates = []
    for raw in (manifest.get("content_candidate_pack_json"), assets.get("content_candidate_pack_path"), "exports/content-candidate-pack.json"):
        value = str(raw or "").strip()
        if value:
            candidates.append(_resolve_bundle_path(root, value))
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else None


def _content_candidate_citation_text(citations: list[Any], *, limit: int = 900) -> str:
    parts: list[str] = []
    for row in citations:
        if not isinstance(row, dict):
            continue
        source_type = str(row.get("source_type") or row.get("type") or "").strip()
        time_range = str(row.get("time_range") or row.get("time") or "").strip()
        text = _payload_text(row.get("text") or row.get("summary") or row.get("evidence") or "")
        if not text:
            continue
        prefix = " / ".join(value for value in (source_type, time_range) if value)
        parts.append((prefix + ": " if prefix else "") + text)
    return _clip("; ".join(parts), limit)


def _content_candidate_citation_paths(citations: list[Any]) -> list[str]:
    paths: list[str] = []
    for row in citations:
        if not isinstance(row, dict):
            continue
        for key in ("evidence_path", "path", "frame_path", "source_path"):
            value = str(row.get(key) or "").strip()
            if value:
                paths.append(value)
        values = row.get("evidence_paths")
        if isinstance(values, list):
            paths.extend(str(value).strip() for value in values if str(value).strip())
    return _dedupe(paths)

def _content_asset_chunks(root: Path, *, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    assets = manifest.get("content_assets") if isinstance(manifest.get("content_assets"), dict) else {}
    keys = [
        "smart_summary_path",
        "key_segments_path",
        "short_video_script_drafts_path",
        "highlight_post_drafts_path",
        "content_material_card_path",
        "content_material_card_markdown_path",
    ]
    chunks: list[dict[str, Any]] = []
    for key in keys:
        raw = str(assets.get(key) or "").strip()
        if not raw:
            continue
        path = _resolve_bundle_path(root, raw)
        text = _read_asset_text(path)
        term_snapshot = _term_snapshot_from_asset(path)
        if not text:
            continue
        chunks.append(
            _make_chunk(
                root,
                kind="content_asset",
                key=key,
                text=f"Content asset {key}: {text}",
                evidence_paths=[str(path)],
                tags=["content_asset", key],
                keywords=_keywords_from_text(text),
                has_visual=False,
                has_temporal=False,
                extra_metadata={"content_asset_key": key, **_term_metadata(term_snapshot)},
            )
        )
    return chunks


def _payload_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, dict):
        pieces: list[str] = []
        for key in ("summary", "screen_text", "markdown", "text", "objects", "actions", "operation_steps", "events", "corrected_text", "comment"):
            if key in value:
                pieces.append(_payload_text(value.get(key)))
        if not pieces:
            pieces.extend(_payload_text(v) for v in value.values())
        return _text(" ".join(piece for piece in pieces if piece))
    if isinstance(value, list):
        return _text(" ".join(_payload_text(item) for item in value))
    return _text(value)


def _evidence_paths(item: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("frame_path", "source_path"):
        value = str(item.get(key) or "").strip()
        if value:
            paths.append(value)
    for key in ("frame_paths", "temporal_frame_paths", "evidence_paths", "crop_paths"):
        values = item.get(key)
        if isinstance(values, list):
            paths.extend(str(value).strip() for value in values if str(value).strip())
    for asset in item.get("assets") or []:
        if isinstance(asset, dict):
            path = str(asset.get("path") or asset.get("copied_path") or "").strip()
            if path:
                paths.append(path)
    for target in item.get("tile_review_targets") or []:
        if isinstance(target, dict):
            path = str(target.get("evidence_path") or target.get("tile_path") or "").strip()
            if path:
                paths.append(path)
    return _dedupe(paths)


def _resolve_bundle_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / value


def _read_asset_text(path: Path, *, limit: int = 1800) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return _clip(text, limit)


def _keywords_from_text(value: str, *, limit: int = 24) -> list[str]:
    text = _text(value)
    if not text:
        return []
    words: list[str] = []
    for raw in text.replace("/", " ").replace("|", " ").split():
        token = raw.strip("`*_:#，。；：、,.!?()[]{}\"'")
        if len(token) >= 2:
            words.append(token[:40])
    return _dedupe(words)[:limit]


def _chunks_by_kind(chunks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in chunks:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        kind = str(metadata.get("chunk_kind") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value is None:
        return []
    return [str(value)]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append(text)
    return rows


def _dedupe_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    rows: list[int] = []
    for value in values:
        number = _int(value)
        if number <= 0 or number in seen:
            continue
        seen.add(number)
        rows.append(number)
    return rows

def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _timestamp(seconds: float) -> str:
    value = max(0.0, float(seconds or 0.0))
    millis = int(round((value - int(value)) * 1000))
    whole = int(value)
    if millis >= 1000:
        whole += 1
        millis = 0
    minutes, sec = divmod(whole, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{sec:02d}.{millis:03d}"

def _retrieved(moment_index: dict[str, Any], chunks: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    hits = moment_index.get("query_hits") if isinstance(moment_index.get("query_hits"), list) else []
    by_chunk = {int(row["metadata"].get("chunk_index") or 0): row for row in chunks if isinstance(row.get("metadata"), dict) and row["metadata"].get("chunk_index") is not None}
    if hits:
        rows = []
        for hit in hits[: max(0, top_k)]:
            if not isinstance(hit, dict):
                continue
            chunk = by_chunk.get(int(hit.get("chunk_index") or 0))
            if not chunk:
                continue
            rows.append(
                {
                    "chunk_id": chunk["id"],
                    "score": hit.get("score", 0.0),
                    "matched_terms": hit.get("matched_terms") or [],
                    "start_time": chunk["metadata"]["start_time"],
                    "end_time": chunk["metadata"]["end_time"],
                    "timeline_indexes": chunk["metadata"]["timeline_indexes"],
                    "snippet": hit.get("snippet") or _clip(chunk["text"], 420),
                    "evidence_paths": chunk["metadata"]["evidence_paths"][:12],
                }
            )
        return rows
    return [
        {
            "chunk_id": row["id"],
            "score": 0.0,
            "matched_terms": [],
            "start_time": row["metadata"]["start_time"],
            "end_time": row["metadata"]["end_time"],
            "timeline_indexes": row["metadata"]["timeline_indexes"],
            "snippet": _clip(row["text"], 420),
            "evidence_paths": row["metadata"]["evidence_paths"][:12],
        }
        for row in chunks[: max(0, top_k)]
    ]


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# VideoRAG Pack",
        "",
        f"- Created: `{result.get('created_at')}`",
        f"- Title: {result.get('title')}",
        f"- Chunks: `{result.get('summary', {}).get('chunks', 0)}`",
        f"- Query: `{result.get('parameters', {}).get('query') or ''}`",
        f"- JSONL: `{result.get('artifacts', {}).get('chunks_jsonl')}`",
        "",
        "## Retrieved Moments",
        "",
    ]
    for row in result.get("retrieved_moments") or []:
        lines.extend(
            [
                f"### `{row.get('start_time')}` - `{row.get('end_time')}`",
                "",
                f"- Chunk: `{row.get('chunk_id')}`",
                f"- Score: `{row.get('score')}`",
                f"- Timeline indexes: `{', '.join(str(v) for v in row.get('timeline_indexes') or [])}`",
                f"- Matched terms: `{', '.join(str(v) for v in row.get('matched_terms') or [])}`",
                "",
                str(row.get("snippet") or ""),
                "",
                "Evidence:",
            ]
        )
        for path in row.get("evidence_paths") or []:
            lines.append(f"- `{path}`")
        lines.append("")
    if not result.get("retrieved_moments"):
        lines.append("（暂无检索结果。）")
    return "\n".join(lines).rstrip() + "\n"


def _join_nonempty(values: list[str]) -> str:
    return "\n\n".join(value.strip() for value in values if value and value.strip())


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _clip(value: str, limit: int) -> str:
    text = _text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."
