from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .powershell import quote_powershell_literal as _ps_quote
from .bilinote_summary_tools import apply_transcript_corrections, correction_stats, parse_transcript_correction_json
from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .storage import read_json, write_json
from .transcript import format_timestamp
from .transcript_correction_pack import _load_best_transcript, _render_srt

SCHEMA = "video_knowledge_pipeline.transcript_edit_session.v1"
EDITS_SCHEMA = "video_knowledge_pipeline.transcript_edit_notes.v1"
CORRECTED_SCHEMA = "video_knowledge_pipeline.human_corrected_transcript.v1"


def prepare_transcript_edit_session(bundle_dir: str | Path, *, write: bool = True) -> dict[str, Any]:
    """Build a BiliNote-style static transcript editor for one bundle."""

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    cues = _load_best_transcript(root, manifest)
    segments = [_segment_from_cue(cue, index) for index, cue in enumerate(cues)]
    _attach_arbitration_reviews(root, manifest, segments)
    session_path = root / "transcript-edit-session.json"
    html_path = root / "transcript-editor.html"
    template_path = root / "transcript-edits.template.json"
    mcp_args_path = root / "mcp-prepare-transcript-edit-session.args.json"
    apply_args_path = root / "mcp-apply-transcript-edits.args.json"
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "title": str(manifest.get("title") or root.name),
        "created_at": now_iso(),
        "summary": {"segments": len(segments), "source": _transcript_source_hint(manifest)},
        "segments": segments,
        "artifacts": {
            "json": str(session_path),
            "html": str(html_path),
            "template": str(template_path),
            "mcp_prepare_args": str(mcp_args_path),
            "mcp_apply_args": str(apply_args_path),
        },
        "operator_boundary": {
            "static_editor_only": True,
            "no_llm_call": True,
            "no_timeline_write": True,
            "edits_apply_only_after_explicit_import": True,
        },
        "write": bool(write),
    }
    if write:
        write_json(session_path, result)
        html_path.write_text(_render_editor_html(result), encoding="utf-8")
        write_json(template_path, {"schema": EDITS_SCHEMA, "bundle_dir": str(root), "segments": _template_segments(segments)})
        write_json(mcp_args_path, {"bundle_dir": str(root), "write": True})
        write_json(apply_args_path, {"bundle_dir": str(root), "edits_json": str(root / "transcript-edits.json"), "write": True})
        manifest["transcript_edit_session_json"] = "transcript-edit-session.json"
        manifest["transcript_editor_html"] = "transcript-editor.html"
        manifest["transcript_edits_template_json"] = "transcript-edits.template.json"
        manifest["mcp_prepare_transcript_edit_session_args"] = "mcp-prepare-transcript-edit-session.args.json"
        manifest["mcp_apply_transcript_edits_args"] = "mcp-apply-transcript-edits.args.json"
        write_json(manifest_path, manifest)
    result["run_registry"] = _register_prepare_run(root, result, write=write)
    return result


def apply_transcript_edits(bundle_dir: str | Path, *, edits_json: str | Path, write: bool = True) -> dict[str, Any]:
    """Import reviewed transcript edits and write human-corrected transcript sidecars."""

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    cues = _load_best_transcript(root, manifest)
    original_segments = [_segment_from_cue(cue, index) for index, cue in enumerate(cues)]
    payload = _load_edits_payload(edits_json)
    corrected = apply_transcript_corrections(original_segments, payload)
    stats = correction_stats(original_segments, corrected)
    corrected_payload = {
        "schema": CORRECTED_SCHEMA,
        "bundle_dir": str(root),
        "source": "human_transcript_editor",
        "created_at": now_iso(),
        "summary": stats,
        "segments": corrected,
    }
    result = {
        "schema": "video_knowledge_pipeline.apply_transcript_edits.v1",
        "bundle_dir": str(root),
        "edits_json": str(Path(edits_json).expanduser().resolve()),
        "summary": stats,
        "corrected_transcript": {
            "json_path": str(root / "human-corrected-transcript.json"),
            "srt_path": str(root / "human-corrected-transcript.srt"),
            "markdown_path": str(root / "human-corrected-transcript.md"),
        },
        "operator_boundary": {
            "human_reviewed_import": True,
            "no_llm_call": True,
            "timeline_not_mutated": True,
        },
        "write": bool(write),
    }
    if write:
        write_json(root / "human-corrected-transcript.json", corrected_payload)
        (root / "human-corrected-transcript.srt").write_text(_render_srt(corrected), encoding="utf-8")
        (root / "human-corrected-transcript.md").write_text(_render_corrected_markdown(corrected_payload), encoding="utf-8")
        manifest["human_corrected_transcript_json"] = "human-corrected-transcript.json"
        manifest["human_corrected_transcript_srt"] = "human-corrected-transcript.srt"
        manifest["human_corrected_transcript_markdown"] = "human-corrected-transcript.md"
        manifest["corrected_transcript_json"] = "human-corrected-transcript.json"
        manifest["corrected_transcript_srt"] = "human-corrected-transcript.srt"
        manifest["corrected_transcript_markdown"] = "human-corrected-transcript.md"
        manifest["transcript_edit_summary"] = stats
        write_json(manifest_path, manifest)
    result["run_registry"] = _register_apply_run(root, result, write=write)
    return result


def _register_prepare_run(root: Path, result: dict[str, Any], *, write: bool) -> dict[str, Any]:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    segment_count = int(summary.get("segments") or 0)
    failed_items: list[dict[str, Any]] = []
    if segment_count <= 0:
        failed_items.append({"id": "transcript", "reason": "transcript_missing", "detail": "No transcript segments are available for manual editing."})
    status = "needs_execution" if not write else "needs_input"
    return register_bundle_run(
        root,
        run_type="prepare_transcript_edit_session",
        run_id="prepare-transcript-edit-session",
        status=status,
        title="Prepare transcript edit session",
        summary=f"Prepared a static transcript editor for {segment_count} segment(s); waiting for reviewed transcript-edits.json.",
        inputs={"bundle_dir": str(root), "transcript_source": str(summary.get("source") or "")},
        parameters={"write": bool(write), "segment_count": segment_count},
        artifacts=[
            {"key": "session_json", "path": root / "transcript-edit-session.json"},
            {"key": "editor_html", "path": root / "transcript-editor.html"},
            {"key": "edits_template", "path": root / "transcript-edits.template.json"},
            {"key": "prepare_mcp_args", "path": root / "mcp-prepare-transcript-edit-session.args.json"},
            {"key": "apply_mcp_args", "path": root / "mcp-apply-transcript-edits.args.json"},
        ],
        failed_items=failed_items,
        retry_command=f".\\scripts\\video-knowledge.ps1 prepare-transcript-edit-session {_ps_quote(str(root))}",
        next_actions=_prepare_run_next_actions(segment_count),
        operator_boundary={
            "local_only": True,
            "no_llm_call": True,
            "static_editor_only": True,
            "no_timeline_write": True,
            "apply_requires_explicit_reviewed_json": True,
            "source_reuse": "PrideWood/BiliNote transcript row editing and video-linked review workflow.",
        },
        write=write,
    )


def _register_apply_run(root: Path, result: dict[str, Any], *, write: bool) -> dict[str, Any]:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    corrected = int(summary.get("corrected_segments") or 0)
    total = int(summary.get("segments") or 0)
    failed_items: list[dict[str, Any]] = []
    if total <= 0:
        failed_items.append({"id": "transcript", "reason": "transcript_missing", "detail": "No original transcript segments were available for applying edits."})
    if total > 0 and corrected <= 0:
        failed_items.append({"id": "edits", "reason": "no_human_edits_applied", "detail": "The reviewed edits did not change any transcript segment."})
    if not write:
        status = "needs_execution"
    elif total <= 0:
        status = "needs_input"
    elif failed_items:
        status = "needs_review"
    else:
        status = "completed"
    edits_json = str(result.get("edits_json") or root / "transcript-edits.json")
    return register_bundle_run(
        root,
        run_type="apply_transcript_edits",
        run_id="apply-transcript-edits",
        status=status,
        title="Apply transcript edits",
        summary=f"Imported reviewed transcript edits: {corrected}/{total} segment(s) changed.",
        inputs={"bundle_dir": str(root), "edits_json": edits_json},
        parameters={"write": bool(write), "segments": total, "corrected_segments": corrected},
        artifacts=[
            {"key": "human_corrected_json", "path": root / "human-corrected-transcript.json"},
            {"key": "human_corrected_srt", "path": root / "human-corrected-transcript.srt"},
            {"key": "human_corrected_markdown", "path": root / "human-corrected-transcript.md"},
        ],
        failed_items=failed_items,
        retry_command=f".\\scripts\\video-knowledge.ps1 apply-transcript-edits {_ps_quote(str(root))} --edits-json {_ps_quote(edits_json)}",
        next_actions=_apply_run_next_actions(status),
        operator_boundary={
            "local_only": True,
            "no_llm_call": True,
            "human_reviewed_import": True,
            "timeline_not_mutated": True,
            "promotes_corrected_transcript_sidecars": True,
        },
        write=write,
    )


def _prepare_run_next_actions(segment_count: int) -> list[str]:
    if segment_count <= 0:
        return ["Add or generate a transcript sidecar, then rerun prepare-transcript-edit-session."]
    return ["Open transcript-editor.html, save reviewed transcript-edits.json, then run apply-transcript-edits."]


def _apply_run_next_actions(status: str) -> list[str]:
    if status == "completed":
        return ["Use human-corrected-transcript.* as the corrected transcript source for export and summary generation."]
    if status == "needs_review":
        return ["Review transcript-edits.json; no segment changed, so decide whether this is acceptable or needs another edit pass."]
    if status == "needs_input":
        return ["Prepare transcript-edit-session and provide a reviewed transcript-edits.json before applying edits."]
    return ["Rerun apply-transcript-edits with --edits-json after confirming the reviewed edits file."]


def _segment_from_cue(cue: Any, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "start": float(getattr(cue, "start", 0.0)),
        "end": float(getattr(cue, "end", 0.0)),
        "timestamp": format_timestamp(float(getattr(cue, "start", 0.0))),
        "end_timestamp": format_timestamp(float(getattr(cue, "end", 0.0))),
        "text": str(getattr(cue, "text", "") or ""),
    }


def _template_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": int(row.get("index") or 0),
            "timestamp": str(row.get("timestamp") or ""),
            "text": str(row.get("text") or ""),
            "corrected_text": str(row.get("text") or ""),
            "status": "unchanged",
            "note": "",
        }
        for row in segments
    ]



def _attach_arbitration_reviews(root: Path, manifest: dict[str, Any], segments: list[dict[str, Any]]) -> None:
    path = _arbitration_json_path(root, manifest)
    if path is None:
        return
    payload = read_json(path)
    if not isinstance(payload, dict):
        return
    review_rows = payload.get("review_rows") if isinstance(payload.get("review_rows"), list) else []
    by_index: dict[int, dict[str, Any]] = {}
    for row in review_rows:
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("index") or 0)
        except Exception:
            continue
        by_index[index] = {
            "segment_index": index,
            "confidence": row.get("confidence"),
            "review_reason": row.get("review_reason") or "",
            "chosen_source": row.get("chosen_source") or "",
            "chosen_source_type": row.get("chosen_source_type") or "",
            "original_text": row.get("original_text") or row.get("raw_text") or "",
            "corrected_text": row.get("corrected_text") or row.get("text") or "",
            "alternatives": _compact_arbitration_alternatives(row),
            "report_path": str(path),
        }
    for segment in segments:
        try:
            index = int(segment.get("index") or 0)
        except Exception:
            continue
        if index in by_index:
            segment["arbitration_review"] = by_index[index]
            segment["review_flags"] = [*segment.get("review_flags", []), "transcript_source_conflict"]


def _compact_arbitration_alternatives(row: dict[str, Any]) -> list[dict[str, Any]]:
    values = row.get("alternatives") if isinstance(row.get("alternatives"), list) else []
    alternatives: list[dict[str, Any]] = []
    for value in values[:8]:
        if not isinstance(value, dict):
            continue
        alternatives.append(
            {
                "source_id": str(value.get("source_id") or ""),
                "source_type": str(value.get("source_type") or ""),
                "text": str(value.get("text") or value.get("raw_text") or ""),
                "score": value.get("score"),
                "overlap": value.get("overlap"),
                "similarity_to_base": value.get("similarity_to_base"),
            }
        )
    return alternatives


def _arbitration_json_path(root: Path, manifest: dict[str, Any]) -> Path | None:
    candidates: list[Path] = []
    raw = str(manifest.get("transcript_source_arbitration_json") or "").strip()
    if raw:
        path = Path(raw).expanduser()
        candidates.append(path if path.is_absolute() else root / path)
    candidates.append(root / "transcript-source-arbitration.json")
    for path in candidates:
        if path.exists():
            return path
    return None

def _load_edits_payload(path: str | Path) -> dict[str, Any]:
    edit_path = Path(path).expanduser().resolve()
    text = edit_path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = parse_transcript_correction_json(text)
    if not isinstance(data, dict):
        raise ValueError("transcript edits must be a JSON object")
    rows = []
    for row in data.get("segments") or []:
        if not isinstance(row, dict):
            continue
        text_value = str(row.get("corrected_text") or row.get("text") or "").strip()
        if text_value:
            rows.append({"index": int(row.get("index") or 0), "text": text_value})
    return {"segments": rows}


def _render_editor_html(session: dict[str, Any]) -> str:
    title = html.escape(str(session.get("title") or "Transcript Editor"))
    rows_json = json.dumps(session.get("segments") or [], ensure_ascii=False).replace("</", "<\\/")
    template_name = html.escape(Path(str(session.get("artifacts", {}).get("template") or "transcript-edits.template.json")).name)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} - Transcript Editor</title>
  <style>
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f7f8fa; color:#172026; }}
    header {{ padding:22px 30px; background:#fff; border-bottom:1px solid #d8dee8; }}
    main {{ max-width:1180px; margin:0 auto; padding:18px 24px 40px; }}
    .panel {{ background:#fff; border:1px solid #d8dee8; border-radius:8px; padding:14px; margin:12px 0; }}
    .muted {{ color:#667085; }}
    .toolbar {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
    input {{ padding:9px; border:1px solid #d8dee8; border-radius:6px; min-width:260px; }}
    button {{ border:1px solid #d8dee8; background:#fff; border-radius:6px; padding:8px 11px; cursor:pointer; }}
    .row {{ display:grid; grid-template-columns:110px minmax(220px,1fr) minmax(260px,1.2fr); gap:10px; padding:10px 0; border-top:1px solid #e7ebf0; }}
    .row:first-child {{ border-top:0; }}
    textarea {{ width:100%; min-height:74px; resize:vertical; border:1px solid #d8dee8; border-radius:6px; padding:8px; font-family:inherit; line-height:1.5; }}
    .changed textarea {{ border-color:#d59b32; background:#fffaf0; }}
    .conflict {{ border-left:5px solid #b42318; padding-left:8px; }}
    .conflict-box {{ background:#fff4f2; border:1px solid #fecdca; border-radius:6px; padding:8px; margin-top:6px; }}
    .alt {{ margin-top:4px; font-size:13px; }}
    code, pre {{ white-space:pre-wrap; word-break:break-word; background:#f1f4f8; border:1px solid #d8dee8; border-radius:6px; padding:8px; }}
    @media (max-width:860px) {{ .row {{ grid-template-columns:1fr; }} main {{ padding:14px; }} }}
  </style>
</head>
<body>
  <header><h1>{title}</h1><div class="muted">旧版兼容入口：逐段校正 ASR/字幕，导出 JSON 后再用 CLI/MCP 显式导入。建议改用 <a href="subtitle-editor.html">新版双轨字幕编辑器</a>，获得播放器、波形、粤语/普通话双轨与显式 VKP 写回。</div></header>
  <main>
    <section class="panel toolbar"><input id="filter" placeholder="搜索转录、术语或时间戳" oninput="renderRows()"><button onclick="markAllUnchanged()">全部标记未改</button><button onclick="exportEdits()">生成 edits JSON</button><button onclick="copyEdits()">复制 JSON</button><a href="{template_name}"><button type="button">打开模板</button></a></section>
    <section id="rows" class="panel"></section>
    <section class="panel"><h2>导出的 transcript-edits.json</h2><div class="muted">保存为 bundle 下 <code style="display:inline;padding:2px 5px">transcript-edits.json</code> 后，运行 apply-transcript-edits。</div><textarea id="exportBox" style="min-height:260px"></textarea></section>
  </main>
<script>
const SEGMENTS = {rows_json};
function q(id) {{ return document.getElementById(id); }}
function esc(v) {{ return String(v || '').replace(/[&<>\"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c] || c)); }}
function renderRows() {{
  const query = (q('filter').value || '').toLowerCase();
  const rows = SEGMENTS.filter(row => !query || JSON.stringify(row).toLowerCase().includes(query) || String(row.timestamp || '').includes(query));
  q('rows').innerHTML = rows.map(row => `<div class="row ${{row.arbitration_review ? 'conflict' : ''}}" id="row-${{row.index}}"><div><strong>#${{row.index}}</strong><div class="muted">${{esc(row.timestamp)}} - ${{esc(row.end_timestamp)}}</div>${{arbitrationBadge(row)}}</div><div><div class="muted">原文</div><textarea readonly>${{esc(row.text)}}</textarea>${{arbitrationBox(row)}}</div><div><div class="muted">校正文</div><textarea data-index="${{row.index}}" oninput="markChanged(${{row.index}})">${{esc(row.text)}}</textarea></div></div>`).join('') || '<div class="muted">没有匹配段落。</div>';
}}
function arbitrationBadge(row) {{ if (!row.arbitration_review) return ''; return `<div class="muted">仲裁待复核 · conf=${{esc(row.arbitration_review.confidence || '')}} · ${{esc(row.arbitration_review.review_reason || 'source conflict')}}</div>`; }}
function arbitrationBox(row) {{ const review = row.arbitration_review; if (!review) return ''; const alts = (review.alternatives || []).slice(0,4).map(alt => `<div class="alt"><strong>${{esc(alt.source_id || alt.source_type || 'source')}}</strong> score=${{esc(alt.score || '')}}<br>${{esc(alt.text || '')}}</div>`).join(''); return `<div class="conflict-box"><strong>字幕/ASR 仲裁冲突</strong><div class="muted">chosen: ${{esc(review.chosen_source || '')}} · reason: ${{esc(review.review_reason || '')}}</div><div class="muted">原始：${{esc(review.original_text || '')}}</div><div class="muted">建议：${{esc(review.corrected_text || '')}}</div>${{alts}}</div>`; }}
function markChanged(index) {{ const el = q('row-' + index); if (el) el.classList.add('changed'); }}
function markAllUnchanged() {{ document.querySelectorAll('.row').forEach(el => el.classList.remove('changed')); }}
function exportEdits() {{
  const segments = Array.from(document.querySelectorAll('textarea[data-index]')).map(el => {{
    const index = Number(el.dataset.index || 0);
    const original = SEGMENTS.find(row => Number(row.index) === index) || {{}};
    const corrected = el.value.trim();
    return {{ index, timestamp: original.timestamp || '', text: original.text || '', corrected_text: corrected, status: corrected && corrected !== original.text ? 'corrected' : 'unchanged', note: '' }};
  }});
  q('exportBox').value = JSON.stringify({{schema:'video_knowledge_pipeline.transcript_edit_notes.v1', segments}}, null, 2);
}}
async function copyEdits() {{ if (!q('exportBox').value) exportEdits(); await navigator.clipboard.writeText(q('exportBox').value); }}
document.addEventListener('DOMContentLoaded', () => {{ renderRows(); exportEdits(); }});
</script>
</body>
</html>
"""


def _render_corrected_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines = [
        "# Human Corrected Transcript",
        "",
        f"- Source: `{payload.get('source')}`",
        f"- Segments: `{summary.get('segments', 0)}`",
        f"- Corrected segments: `{summary.get('corrected_segments', 0)}`",
        f"- Indexes preserved: `{summary.get('indexes_preserved')}`",
        "",
    ]
    for segment in payload.get("segments") or []:
        marker = " changed" if segment.get("changed") else ""
        lines.extend([f"## {segment.get('index')}. {segment.get('timestamp')}{marker}", "", str(segment.get("corrected_text") or segment.get("text") or "").strip(), ""])
        if segment.get("changed"):
            lines.extend(["原文：", "", str(segment.get("raw_text") or "").strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _transcript_source_hint(manifest: dict[str, Any]) -> str:
    for key in ("corrected_transcript_json", "human_corrected_transcript_json", "llm_corrected_transcript_json", "normalized_transcript_json", "transcript_json"):
        if manifest.get(key):
            return key
    return "timeline_or_unknown"
