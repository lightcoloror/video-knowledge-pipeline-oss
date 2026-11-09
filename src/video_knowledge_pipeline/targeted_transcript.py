"""Opt-in transcript revision and non-destructive cut preview in targeted review.

Reuses the existing Tiptap adapter, VKP source/hash-bound loopback review,
and Hyperaudio's kept-section/playback model. This is a draft, not a render.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .storage import read_json, write_json
from .targeted_pacing import load_pacing_candidates, validate_pacing_review

SOURCE_SCHEMA = "vkp-targeted-transcript-source.v1"
DRAFT_SCHEMA = "vkp-targeted-transcript-review.draft.v1"
DRAFT_FILENAME = "targeted-transcript-review.draft.json"
ASSET_ROOT = Path(__file__).resolve().parent / "static" / "targeted-transcript-editor"
ASSETS = {"editor.bundle.js", "editor.bundle.css", "editor.bundle.js.LEGAL.txt"}
_DRAFT_WRITE_LOCK = threading.RLock()


class TranscriptDraftConflict(ValueError):
    pass


def transcript_draft_revision(draft: dict[str, Any] | None) -> str:
    if not draft:
        return ""
    return hashlib.sha256(json.dumps(draft["review"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError("transcript_time_invalid")
    return round(float(value), 6)


def _utf16_size(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def normalize_transcript_source(value: Any, sources: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("transcript_edit_schema_invalid")
    chapters = value.get("chapters")
    if not isinstance(chapters, list) or not 1 <= len(chapters) <= 100:
        raise ValueError("transcript_chapters_invalid")
    result = []
    seen: set[str] = set()
    for raw in chapters:
        if not isinstance(raw, dict):
            raise ValueError("transcript_chapter_invalid")
        number = str(raw.get("number") or "")
        if not number.isdigit() or len(number) != 2 or number in seen:
            raise ValueError("transcript_chapter_number_invalid")
        seen.add(number)
        matches = [s for s in sources if s["sha256"] == raw.get("source_sha256") and s["label"] == raw.get("file")]
        if len(matches) != 1:
            raise ValueError(f"transcript_source_binding_invalid:{number}")
        source = matches[0]
        segments = raw.get("segments")
        if not isinstance(segments, list) or not 1 <= len(segments) <= 10000:
            raise ValueError("transcript_segments_invalid")
        rows = []
        seen_segments: set[str] = set()
        previous_start = 0.0
        for seg in segments:
            if not isinstance(seg, dict):
                raise ValueError("transcript_segment_invalid")
            text, ident = str(seg.get("text") or ""), str(seg.get("id") or "")
            start, end = _number(seg.get("start")), _number(seg.get("end"))
            if not ident or ident in seen_segments or not text or len(text) > 10000:
                raise ValueError("transcript_segment_identity_invalid")
            if not previous_start <= start < end <= source["duration_seconds"] + 0.05:
                raise ValueError("transcript_segment_time_invalid")
            previous_start = start
            seen_segments.add(ident)
            timings = []
            previous_char = 0
            for token in seg.get("timings") or []:
                if not isinstance(token, dict):
                    raise ValueError("transcript_character_invalid")
                a, b = token.get("start_char"), token.get("end_char")
                if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int) or not isinstance(b, int):
                    raise ValueError("transcript_character_index_invalid")
                ta, tb = _number(token.get("start")), _number(token.get("end"))
                if not (previous_char <= a < b <= _utf16_size(text) and max(0, start - 0.1) <= ta <= tb <= min(source["duration_seconds"], end + 0.1)):
                    raise ValueError("transcript_character_time_invalid")
                selected = text.encode("utf-16-le")[a * 2:b * 2].decode("utf-16-le")
                if selected != token.get("text"):
                    raise ValueError("transcript_character_text_mismatch")
                previous_char = b
                timings.append({"start_char": a, "end_char": b, "text": selected, "start": ta, "end": tb,
                                "source_timestamp_index": int(token.get("source_timestamp_index", -1))})
            preselection = seg.get("preselection", "keep")
            if preselection not in {"keep", "delete", "review_required"}:
                raise ValueError("transcript_preselection_invalid")
            rows.append({"id": ident, "start": start, "end": end, "text": text, "timings": timings,
                         "preselection": preselection, "preselection_reason": str(seg.get("preselection_reason") or ""),
                         "script_match": str(seg.get("script_match") or "")})
        text_hash = hashlib.sha256("".join(s["text"] for s in rows).encode()).hexdigest()
        if raw.get("source_text_hash") != text_hash:
            raise ValueError("transcript_original_text_hash_mismatch")
        result.append({"number": number, "source_id": source["source_id"], "source_sha256": source["sha256"],
                       "file": source["label"], "duration": source["duration_seconds"], "source_text_hash": text_hash,
                       "script": str(raw.get("script") or ""), "segments": rows})
    return {"schema_version": SOURCE_SCHEMA, "default_view": "transcript", "script_reference": str(value.get("script_reference") or ""), "chapters": result}


def _validate_original_document(doc: dict[str, Any], source: dict[str, Any]) -> None:
    """Validate the preserved original projection, without reimplementing editing.

    Tiptap's insertion/deletion/sourceOrigin marks already retain the original.
    Read those marks and require exact source identity and order before saving.
    """
    segments = source["segments"]
    ids = {source["number"] + ":" + s["id"]: i for i, s in enumerate(segments)}
    collected = ["" for _ in segments]
    previous = -1

    def visit(node: Any, depth: int = 0) -> None:
        nonlocal previous
        if not isinstance(node, dict) or depth > 4 or node.get("type") not in {"doc", "paragraph", "text"}:
            raise ValueError("transcript_review_doc_node_invalid")
        if node["type"] == "text":
            marks = node.get("marks") or []
            if not isinstance(node.get("text"), str) or not isinstance(marks, list):
                raise ValueError("transcript_review_doc_text_invalid")
            if any(not isinstance(m, dict) or m.get("type") not in {"sourceOrigin", "insertion", "deletion"} for m in marks):
                raise ValueError("transcript_review_doc_mark_invalid")
            if any(m["type"] == "insertion" for m in marks):
                return
            origin = next((m.get("attrs", {}).get("sourceId") for m in marks if m["type"] == "sourceOrigin"), None)
            index = ids.get(origin, -1)
            if index < 0 or index < previous:
                raise ValueError("transcript_review_doc_source_invalid")
            previous = index
            collected[index] += node["text"]
        else:
            content = node.get("content") or []
            if not isinstance(content, list):
                raise ValueError("transcript_review_doc_content_invalid")
            for child in content:
                visit(child, depth + 1)

    visit(doc)
    if collected != [s["text"] for s in segments]:
        raise ValueError("transcript_review_doc_original_changed")


def _validate_subtitle_layout(chapter: dict[str, Any], source: dict[str, Any]) -> None:
    attrs = chapter["doc"].get("attrs") or {}
    if not isinstance(attrs, dict):
        raise ValueError("transcript_subtitle_layout_invalid")
    edits = attrs.get("subtitleEdits")
    if edits is None:
        return
    if not isinstance(edits, list) or len(edits) > 5000:
        raise ValueError("transcript_subtitle_layout_invalid")
    total = sum(_utf16_size(s["text"]) for s in source["segments"])
    revisions = {r.get("change_id"): r for r in chapter.get("revisions", []) if isinstance(r, dict)}
    seen: set[str] = set()
    for edit in edits:
        if not isinstance(edit, dict):
            raise ValueError("transcript_subtitle_layout_invalid")
        ident, action = edit.get("id"), edit.get("action")
        if not isinstance(ident, str) or not ident.startswith("subtitle:") or len(ident) > 120 or ident in seen or action not in {"merge_previous", "merge_next", "split"}:
            raise ValueError("transcript_subtitle_layout_invalid")
        seen.add(ident)
        a, at, b = edit.get("scope_start"), edit.get("at"), edit.get("scope_end")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (a, at, b)) or not 0 <= a < at < b <= total:
            raise ValueError("transcript_subtitle_layout_anchor_invalid")
        if any(not isinstance(edit.get(k), str) or not 0 < len(edit[k]) <= 30000 for k in ("before", "after")):
            raise ValueError("transcript_subtitle_layout_invalid")
        offset = 0
        for segment in source["segments"]:
            size = _utf16_size(segment["text"])
            if offset <= at < offset + size:
                local = at - offset
                if local and action == "split":
                    tokens = segment.get("timings") or []
                    previous = [t for t in tokens if t["end_char"] <= local]
                    following = [t for t in tokens if t["start_char"] >= local]
                    if any(t["start_char"] < local < t["end_char"] for t in tokens) or not previous or not following or following[0]["start"] < previous[-1]["end"]:
                        raise ValueError("transcript_subtitle_layout_native_time_required")
                break
            offset += size
        revision = revisions.get(ident)
        if not revision or revision.get("effect") != "transcript_only" or revision.get("kind") != "subtitle_structure" or revision.get("subtitle_layout") != edit:
            raise ValueError("transcript_subtitle_layout_decision_invalid")
    if any(r.get("kind") == "subtitle_structure" and r.get("change_id") not in seen for r in revisions.values()):
        raise ValueError("transcript_subtitle_layout_decision_invalid")


def compose_transcript_draft(pack: dict[str, Any], request: dict[str, Any], pacing_candidates=None) -> dict[str, Any]:
    if request.get("pack_payload_sha256") != pack.get("pack_payload_sha256"):
        raise ValueError("transcript_review_pack_changed")
    review = request.get("review")
    if not isinstance(review, dict) or review.get("schema") != "rwe.transcript-review.v3":
        raise ValueError("transcript_review_document_invalid")
    validate_pacing_review(review.get('pacing_review'), pacing_candidates)
    expected = {c["number"]: c for c in pack.get("transcript_edit", {}).get("chapters", [])}
    supplied = review.get("chapters")
    if not isinstance(supplied, list) or len(supplied) != len(expected):
        raise ValueError("transcript_review_source_count_mismatch")
    seen: set[str] = set()
    for chapter in supplied:
        if not isinstance(chapter, dict):
            raise ValueError("transcript_review_chapter_invalid")
        number = chapter.get("chapter")
        source = expected.get(number)
        if not source or number in seen or source["source_text_hash"] != chapter.get("source_hash") or source["file"] != chapter.get("file"):
            raise ValueError("transcript_review_source_hash_mismatch")
        seen.add(number)
        if not isinstance(chapter.get("doc"), dict) or chapter["doc"].get("type") != "doc":
            raise ValueError("transcript_review_doc_invalid")
        _validate_original_document(chapter["doc"], source)
        _validate_subtitle_layout(chapter, source)
        base = str(chapter.get("original_text") or "").replace("\n", "")
        if hashlib.sha256(base.encode()).hexdigest() != source["source_text_hash"]:
            raise ValueError("transcript_review_original_changed")
        revisions = chapter.get("revisions") or []
        if not isinstance(revisions, list):
            raise ValueError("transcript_review_revisions_invalid")
        for revision in revisions:
            if not isinstance(revision, dict):
                raise ValueError("transcript_review_revision_invalid")
            if revision.get("effect") not in {"cut_media", "transcript_only", "review_required"}:
                raise ValueError("transcript_review_effect_invalid")
            if any(not isinstance(revision.get(k), str) or not revision[k].strip() for k in ("intent", "decision", "reason", "evidence", "effective_scope")):
                raise ValueError("transcript_review_explanation_required")
    return {"schema_version": DRAFT_SCHEMA, "status": "draft_only", "pack_payload_sha256": pack["pack_payload_sha256"],
            "saved_at": datetime.now().astimezone().isoformat(), "review": review, "media_modified": False, "publication_allowed": False}


def save_transcript_draft(root: Path, pack: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    with _DRAFT_WRITE_LOCK:
        current = load_transcript_draft(root, pack)
        if str(request.get("base_revision_sha256") or "") != transcript_draft_revision(current):
            raise TranscriptDraftConflict("VKP 草稿已由其他页面更新；本机修改保留，请重新载入最新草稿后再保存。")
        result = compose_transcript_draft(pack, request, load_pacing_candidates(root, pack))
        result["revision_sha256"] = transcript_draft_revision(result)
        write_json(root / DRAFT_FILENAME, result)
        return result


def load_transcript_draft(root: Path, pack: dict[str, Any]) -> dict[str, Any] | None:
    path = root / DRAFT_FILENAME
    if not path.is_file():
        return None
    saved = read_json(path)
    if saved.get("pack_payload_sha256") != pack["pack_payload_sha256"]:
        return None
    validated = compose_transcript_draft(pack, saved, load_pacing_candidates(root, pack))
    validated["saved_at"] = saved.get("saved_at", validated["saved_at"])
    validated["revision_sha256"] = transcript_draft_revision(validated)
    return validated


def render_transcript_editor(pack: dict[str, Any], root: Path, token: str) -> str:
    data = pack["transcript_edit"]
    saved = load_transcript_draft(root, pack)
    from .review_export_context import resolve_review_export_context
    export_context = resolve_review_export_context(pack,root=root)
    if export_context and export_context.get('pack_payload_sha256') != pack['pack_payload_sha256']:
        raise ValueError('transcript_export_context_pack_changed')
    config = {"storageKey": "vkp-transcript-review:" + pack["pack_payload_sha256"], "mediaUrl": "/targeted-media", "exportContext": export_context,
              "saveUrl": "/api/targeted-review/transcript/save", "token": token, "packHash": pack["pack_payload_sha256"],
              "savedDraft": (saved or {}).get("review"), "savedDraftRevision": transcript_draft_revision(saved), "pacingCandidates": load_pacing_candidates(root, pack)}
    safe = lambda v: json.dumps(v, ensure_ascii=False).replace("</", "<\\/")
    bootstrap = "<script>window.RWE_TRANSCRIPT_DATA=" + safe(data) + ";window.VKP_TRANSCRIPT_CONTEXT=" + safe(config) + ";</script>"
    text = (ASSET_ROOT / "index.template.html").read_text(encoding="utf-8")
    text = text.replace('<script src="transcript-editor/data.js"></script>', bootstrap)
    text = text.replace('href="transcript-editor/', 'href="/targeted-transcript-assets/').replace('src="transcript-editor/', 'src="/targeted-transcript-assets/')
    return text


def add_transcript_panel(html: str, pack: dict[str, Any]) -> str:
    if not pack.get("transcript_edit"):
        return html
    tabs = '<nav style="display:flex;gap:8px"><button id="exactReviewTab" onclick="toggleTranscriptReview(false)">精确区间审核</button><button id="transcriptReviewTab" class="primary" onclick="toggleTranscriptReview(true)">逐字稿剪辑预览</button></nav>'
    html = html.replace('</header>', tabs + '</header>', 1).replace('<main>', '<main id="exactReviewMain">', 1)
    html = html.replace('</main>', '</main><iframe id="transcriptReviewFrame" title="逐字稿修订与剪辑预览" style="display:none;width:100%;height:calc(100vh - 74px);border:0;background:#eef0f3"></iframe>', 1)
    script = r'''<script>
    function toggleTranscriptReview(show) {
      document.getElementById('exactReviewMain').style.display=show?'none':'';
      const frame=document.getElementById('transcriptReviewFrame');
      frame.style.display=show?'block':'none';
      document.getElementById('transcriptReviewTab').classList.toggle('primary',show);
      document.getElementById('exactReviewTab').classList.toggle('primary',!show);
      if(show){document.getElementById('player').pause();if(!frame.getAttribute('src'))frame.src='/targeted-transcript-editor';}
      else if(frame.contentWindow?.rweReview?.editor){frame.contentDocument.getElementById('player')?.pause();}
    }
    let transcriptImportPending=false,transcriptImportLastId=null;
    window.addEventListener('message',async event=>{
      if(event.source!==window.opener || event.data?.type!=='VKP_IMPORT_TRANSCRIPT_REVIEW')return;
      if(event.data.migration_id && event.data.migration_id===transcriptImportLastId){event.source.postMessage({type:'VKP_TRANSCRIPT_REVIEW_IMPORTED'},event.origin==='null'?'*':event.origin);return;}
      if(transcriptImportPending)return;
      transcriptImportPending=true;
      toggleTranscriptReview(true);
      const frame=document.getElementById('transcriptReviewFrame');
      for(let i=0;i<50&&!frame.contentWindow?.rweReview;i++)await new Promise(r=>setTimeout(r,100));
      try{frame.contentWindow.rweReview.importReview(event.data.review);transcriptImportLastId=event.data.migration_id;event.source.postMessage({type:'VKP_TRANSCRIPT_REVIEW_IMPORTED'},event.origin==='null'?'*':event.origin);}
      catch(error){document.getElementById('summary').textContent='导入原文稿修订失败：'+error.message;}
      finally{transcriptImportPending=false;}
    });
    document.addEventListener('DOMContentLoaded',()=>toggleTranscriptReview(new URLSearchParams(location.search).get('view')!=='exact'));
    </script>'''
    return html.replace('</body>', script + '</body>', 1)
