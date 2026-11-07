from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import read_json, read_jsonl, write_json
from .external_reuse_run_artifacts import ps_quote, register_external_reuse_run
from .video_rag_pack import build_video_rag_pack, _validate_observation_sources

SCHEMA = "video_knowledge_pipeline.video_rag_search.v1"
SUPPORTED_BACKENDS = {"keyword", "sqlite", "vector"}


def search_video_rag(
    bundle_dir: str | Path,
    *,
    query: str,
    top_k: int = 8,
    ensure_pack: bool = True,
    retrieval_backend: str = "keyword",
    write: bool = True,
) -> dict[str, Any]:
    """Search VKP's local VideoRAG chunks.

    ``keyword`` remains the default and reads ``exports/video-rag-chunks.jsonl``
    directly. ``sqlite`` builds a local stdlib SQLite index and then applies the
    same transparent lexical ranking. ``vector`` is reserved as an explicit
    future backend and never starts a vector service here.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")

    backend = _normalise_backend(retrieval_backend)
    chunks_path = root / "exports" / "video-rag-chunks.jsonl"
    sqlite_path = root / "exports" / "video-rag-index.sqlite"
    pack_built = False
    if ensure_pack and not chunks_path.exists():
        build_video_rag_pack(root, query=query, top_k=top_k, write=True)
        pack_built = True
    source_chunks = [row for row in read_jsonl(chunks_path) if isinstance(row, dict)] if chunks_path.exists() else []
    observation_admission = _validate_observation_sources(root, source_chunks)
    sqlite_index_built = False
    backend_status = "ok"
    backend_warning = ""
    if backend == "sqlite":
        sqlite_index_built = _write_sqlite_index(sqlite_path, source_chunks)
        chunks = _load_sqlite_chunks(sqlite_path, query=query) if sqlite_path.exists() else source_chunks
    elif backend == "vector":
        backend_status = "not_implemented"
        backend_warning = "vector backend is reserved for a future optional adapter; falling back to keyword without starting a service"
        backend = "keyword"
        chunks = source_chunks
    else:
        chunks = source_chunks
    hits = _rank_chunks(chunks, query=query, top_k=top_k)

    json_path = root / "exports" / "video-rag-search.json"
    markdown_path = root / "exports" / "video-rag-search.md"
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "title": str(manifest.get("title") or root.name),
        "created_at": now_iso(),
        "query": query,
        "top_k": int(top_k or 8),
        "retrieval_backend": backend,
        "requested_retrieval_backend": retrieval_backend,
        "backend_status": backend_status,
        "backend_warning": backend_warning,
        "summary": {
            "chunks_loaded": len(chunks),
            "source_chunks_loaded": len(source_chunks),
            "hits": len(hits),
            "pack_built": pack_built,
            "chunks_jsonl_exists": chunks_path.exists(),
            "sqlite_index_exists": sqlite_path.exists(),
            "sqlite_index_built": sqlite_index_built,
        },
        "hits": hits,
        "observation_admission": observation_admission,
        "artifacts": {
            "json": str(json_path),
            "markdown": str(markdown_path),
            "chunks_jsonl": str(chunks_path),
            "sqlite_index": str(sqlite_path),
        },
        "operator_boundary": {
            "local_only": True,
            "no_cloud_model_call": True,
            "no_vector_backend_started": True,
            "read_only_search": True,
            "sqlite_backend_optional": True,
        },
        "write": bool(write),
    }
    if write:
        write_json(json_path, result)
        markdown_path.write_text(_render_markdown(result), encoding="utf-8")
        manifest["video_rag_search"] = "exports/video-rag-search.json"
        manifest["video_rag_search_markdown"] = "exports/video-rag-search.md"
        manifest["video_rag_search_backend"] = backend
        if sqlite_path.exists():
            manifest["video_rag_sqlite_index"] = "exports/video-rag-index.sqlite"
        manifest["mcp_video_rag_search_args"] = "mcp-video-rag-search.args.json"
        write_json(
            root / "mcp-video-rag-search.args.json",
            {
                "bundle_dir": str(root),
                "query": query,
                "top_k": top_k,
                "ensure_pack": True,
                "retrieval_backend": backend,
                "write": True,
            },
        )
        write_json(manifest_path, manifest)
        failed_items = []
        if result["summary"].get("source_chunks_loaded", 0) == 0:
            failed_items.append({"id": "video_rag_chunks", "reason": "no_chunks_loaded", "detail": "No VideoRAG chunks were available for search."})
        if backend_status != "ok":
            failed_items.append({"id": "retrieval_backend", "reason": backend_status, "detail": backend_warning})
        register_external_reuse_run(
            root,
            run_type="video_rag_search",
            title="VideoRAG local search",
            result=result,
            status="needs_input" if any(item.get("reason") == "no_chunks_loaded" for item in failed_items) else "needs_review" if failed_items else "completed",
            failed_items=failed_items,
            retry_command=f".\\scripts\\video-knowledge.ps1 video-rag-search {ps_quote(root)} --query {ps_quote(query or '<问题或术语>')} --top-k {int(top_k or 8)} --retrieval-backend {backend}",
            next_actions=[] if not failed_items else ["Build video-rag-pack first or choose keyword/sqlite backend, then rerun video-rag-search."],
            write=True,
        )
    return result


def _normalise_backend(value: str) -> str:
    backend = str(value or "keyword").strip().lower().replace("-", "_")
    if backend in {"jsonl", "lexical", "local_keyword"}:
        return "keyword"
    if backend in {"sqlite3", "sqlite_lexical", "local_sqlite"}:
        return "sqlite"
    if backend in {"vector", "vectors", "embedding", "embeddings"}:
        return "vector"
    return backend if backend in SUPPORTED_BACKENDS else "keyword"


def _write_sqlite_index(sqlite_path: Path, chunks: list[dict[str, Any]]) -> bool:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, chunk_kind TEXT, text TEXT, metadata_json TEXT, start REAL, end REAL)"
        )
        conn.execute("DELETE FROM chunks")
        rows = []
        for chunk in chunks:
            metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            rows.append(
                (
                    str(chunk.get("id") or ""),
                    str(metadata.get("chunk_kind") or "moment"),
                    str(chunk.get("text") or ""),
                    json.dumps(metadata, ensure_ascii=False),
                    float(metadata.get("start") or 0.0),
                    float(metadata.get("end") or 0.0),
                )
            )
        conn.executemany("INSERT OR REPLACE INTO chunks (id, chunk_kind, text, metadata_json, start, end) VALUES (?, ?, ?, ?, ?, ?)", rows)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_kind_start ON chunks(chunk_kind, start)")
        conn.commit()
    return True


def _load_sqlite_chunks(sqlite_path: Path, *, query: str) -> list[dict[str, Any]]:
    query_tokens = sorted(_token_set(query))[:8]
    where = ""
    params: list[str] = []
    if query_tokens:
        clauses = []
        for token in query_tokens:
            clauses.append("(text LIKE ? OR metadata_json LIKE ?)")
            like = f"%{token}%"
            params.extend([like, like])
        where = "WHERE " + " OR ".join(clauses)
    sql = f"SELECT id, text, metadata_json FROM chunks {where} ORDER BY start ASC LIMIT 500"
    with sqlite3.connect(sqlite_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        if not rows and where:
            rows = conn.execute("SELECT id, text, metadata_json FROM chunks ORDER BY start ASC LIMIT 500").fetchall()
    chunks: list[dict[str, Any]] = []
    for chunk_id, text, metadata_json in rows:
        try:
            metadata = json.loads(metadata_json or "{}")
        except json.JSONDecodeError:
            metadata = {}
        chunks.append({"id": chunk_id, "text": text, "metadata": metadata})
    return chunks


def _rank_chunks(chunks: list[dict[str, Any]], *, query: str, top_k: int) -> list[dict[str, Any]]:
    query_tokens = _token_set(query)
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        text = str(chunk.get("text") or "")
        keywords = [str(value) for value in metadata.get("keywords") or []]
        tags = [str(value) for value in metadata.get("tags") or []]
        haystack = " ".join([text, " ".join(keywords), " ".join(tags)])
        tokens = _token_set(haystack)
        overlap = query_tokens.intersection(tokens) if query_tokens else set()
        keyword_overlap = query_tokens.intersection(_token_set(" ".join(keywords))) if query_tokens else set()
        tag_overlap = query_tokens.intersection(_token_set(" ".join(tags))) if query_tokens else set()
        if query_tokens and not overlap:
            continue
        score = _score(query_tokens, overlap, keyword_overlap, tag_overlap, metadata)
        rows.append(
            {
                "chunk_id": str(chunk.get("id") or ""),
                "chunk_kind": str(metadata.get("chunk_kind") or "moment"),
                "score": round(score, 4),
                "matched_terms": sorted(overlap)[:40],
                "matched_keywords": sorted(keyword_overlap)[:20],
                "matched_tags": sorted(tag_overlap)[:20],
                "start": float(metadata.get("start") or 0.0),
                "end": float(metadata.get("end") or 0.0),
                "start_time": str(metadata.get("start_time") or ""),
                "end_time": str(metadata.get("end_time") or ""),
                "timeline_indexes": metadata.get("timeline_indexes") if isinstance(metadata.get("timeline_indexes"), list) else [],
                "memory_level": str(metadata.get("memory_level") or ""),
                "memory_id": str(metadata.get("memory_id") or ""),
                "parent_memory_id": str(metadata.get("parent_memory_id") or ""),
                "child_memory_ids": metadata.get("child_memory_ids") if isinstance(metadata.get("child_memory_ids"), list) else [],
                "child_moment_indexes": metadata.get("child_moment_indexes") if isinstance(metadata.get("child_moment_indexes"), list) else [],
                "fact_status": str(metadata.get("fact_status") or ""),
                "source_observations": metadata.get('source_observations', []),
                "evidence_scope": metadata.get('evidence_scope', 'legacy_or_other_derived_not_assessed'),
                "candidate_id": str(metadata.get("candidate_id") or ""),
                "candidate_types": metadata.get("candidate_types") if isinstance(metadata.get("candidate_types"), list) else [],
                "content_asset_key": str(metadata.get("content_asset_key") or ""),
                "term_correction_status": str(metadata.get("term_correction_status") or ""),
                "term_validation_status": str(metadata.get("term_validation_status") or ""),
                "accepted_validation_decisions": int(metadata.get("accepted_validation_decisions") or 0),
                "rejected_validation_decisions": int(metadata.get("rejected_validation_decisions") or 0),
                "accepted_term_count": int(metadata.get("accepted_term_count") or 0),
                "summary_chapter_refs": metadata.get("summary_chapter_refs") if isinstance(metadata.get("summary_chapter_refs"), list) else [],
                "visual_routes": metadata.get("visual_routes") if isinstance(metadata.get("visual_routes"), dict) else {},
                "has_visual_evidence": bool(metadata.get("has_visual_evidence")),
                "has_temporal_evidence": bool(metadata.get("has_temporal_evidence")),
                "snippet": _best_snippet(text, query_tokens),
                "evidence_paths": (metadata.get("evidence_paths") if isinstance(metadata.get("evidence_paths"), list) else [])[:16],
            }
        )
    rows.sort(key=lambda row: (-float(row.get("score") or 0.0), float(row.get("start") or 0.0)))
    return rows[: max(0, int(top_k or 8))]


def _score(
    query_tokens: set[str],
    overlap: set[str],
    keyword_overlap: set[str],
    tag_overlap: set[str],
    metadata: dict[str, Any],
) -> float:
    if not query_tokens:
        return 0.1
    base = len(overlap) / max(1, len(query_tokens))
    keyword_bonus = 0.18 * len(keyword_overlap)
    tag_bonus = 0.12 * len(tag_overlap)
    visual_bonus = 0.18 if metadata.get("has_visual_evidence") else 0.0
    temporal_bonus = 0.18 if metadata.get("has_temporal_evidence") else 0.0
    chunk_kind = str(metadata.get("chunk_kind") or "")
    review_gap_bonus = 0.12 if chunk_kind == "review_gap" else 0.0
    content_asset_bonus = 0.08 if chunk_kind in {"content_asset", "content_candidate"} else 0.0
    memory_bonus = 0.16 if chunk_kind in {"chapter_memory", "theme_memory"} else 0.08 if chunk_kind in {"short_memory", "memory_boundary"} else 0.0
    duration = max(0.0, float(metadata.get("end") or 0.0) - float(metadata.get("start") or 0.0))
    duration_penalty = min(0.18, math.log1p(duration) / 100.0)
    return base + keyword_bonus + tag_bonus + visual_bonus + temporal_bonus + review_gap_bonus + content_asset_bonus + memory_bonus - duration_penalty


def _best_snippet(text: str, query_tokens: set[str], *, limit: int = 520) -> str:
    cleaned = _clean(text)
    if not cleaned:
        return ""
    if not query_tokens:
        return _clip(cleaned, limit)
    sentences = re.split(r"(?<=[。！？!?；;])\s*", cleaned)
    ranked = []
    for sentence in sentences:
        tokens = _token_set(sentence)
        ranked.append((len(tokens.intersection(query_tokens)), -len(sentence), sentence))
    ranked.sort(reverse=True)
    return _clip(ranked[0][2] if ranked else cleaned, limit)


def _token_set(value: str) -> set[str]:
    text = str(value or "").lower()
    ascii_tokens = re.findall(r"[a-z0-9][a-z0-9_+.-]{1,}", text)
    chinese_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    short_chinese: list[str] = []
    for token in chinese_tokens:
        if len(token) <= 6:
            short_chinese.append(token)
        else:
            short_chinese.extend(token[index : index + 2] for index in range(0, len(token) - 1))
            short_chinese.extend(token[index : index + 3] for index in range(0, len(token) - 2))
    return {token for token in ascii_tokens + short_chinese if token.strip()}


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# VideoRAG Search",
        "",
        f"- Created: `{result.get('created_at')}`",
        f"- Query: `{result.get('query')}`",
        f"- Retrieval backend: `{result.get('retrieval_backend')}`",
        f"- Backend status: `{result.get('backend_status')}`",
        f"- Chunks loaded: `{result.get('summary', {}).get('chunks_loaded', 0)}`",
        f"- Hits: `{result.get('summary', {}).get('hits', 0)}`",
        f"- SQLite index: `{result.get('summary', {}).get('sqlite_index_exists', False)}`",
        "- Boundary: local retrieval only; no cloud model call and no vector backend started.",
        "",
    ]
    if result.get("backend_warning"):
        lines.extend(["## Backend Warning", "", str(result.get("backend_warning") or ""), ""])
    for hit in result.get("hits") or []:
        lines.extend(
            [
                f"## `{hit.get('start_time')}` - `{hit.get('end_time')}`",
                "",
                f"- Score: `{hit.get('score')}`",
                f"- Chunk: `{hit.get('chunk_id')}`",
                f"- Chunk kind: `{hit.get('chunk_kind')}`",
                f"- Memory: `{hit.get('memory_level') or '-'} / {hit.get('memory_id') or '-'}`",
                f"- Parent memory: `{hit.get('parent_memory_id') or '-'}`",
                f"- Child memories: `{', '.join(str(value) for value in hit.get('child_memory_ids') or [])}`",
                f"- Child moments: `{', '.join(str(value) for value in hit.get('child_moment_indexes') or [])}`",
                f"- Fact status: `{hit.get('fact_status') or '-'}`",
                f"- Content asset key: `{hit.get('content_asset_key') or '-'}`",
                f"- Term correction status: `{hit.get('term_correction_status') or 'missing'}`",
                f"- Codex term validation: `{hit.get('term_validation_status') or 'missing'}`",
                f"- Term validation accepted/rejected: `{int(hit.get('accepted_validation_decisions') or 0)}/{int(hit.get('rejected_validation_decisions') or 0)}`",
                f"- Matched terms: `{', '.join(str(value) for value in hit.get('matched_terms') or [])}`",
                f"- Visual evidence: `{hit.get('has_visual_evidence')}`",
                f"- Temporal evidence: `{hit.get('has_temporal_evidence')}`",
                "",
                str(hit.get("snippet") or ""),
                "",
                "Evidence:",
            ]
        )
        for path in hit.get("evidence_paths") or []:
            lines.append(f"- `{path}`")
        lines.append("")
    if not result.get("hits"):
        lines.append("No hits. Try a different term, tool name, or timestamp-related keyword.")
    return "\n".join(lines).rstrip() + "\n"


def _clean(value: str) -> str:
    return " ".join(str(value or "").split())


def _clip(value: str, limit: int) -> str:
    text = _clean(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."
