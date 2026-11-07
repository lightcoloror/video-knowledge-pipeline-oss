from __future__ import annotations
from .review_export_naming import review_export_script

import html
import json
from pathlib import Path
from typing import Any

from .powershell import quote_powershell_literal as _ps_quote
from .models import now_iso
from .run_artifact_registry import register_bundle_run
from .smart_summary_section_workflow import build_smart_summary_section_workflow
from .storage import read_json, write_json
from .term_correction_status import term_correction_status as _term_correction_status
from .transcript import format_timestamp
from .transcript_correction_pack import _load_best_transcript

SCHEMA = "video_knowledge_pipeline.smart_summary_section_editor.v1"
REVISION_SCHEMA = "video_knowledge_pipeline.smart_summary_section_revisions.v1"
SEMANTIC_REVIEW_NOTES_SCHEMA = "video_knowledge_pipeline.transcript_semantic_review_notes.v1"


def build_smart_summary_section_editor(bundle_dir: str | Path, *, write: bool = True) -> dict[str, Any]:
    """Build a static BiliNote-style section editor for smart-summary revisions.

    The page is intentionally local and static: it shows video/transcript/evidence
    next to section rewrite textareas, then downloads a revision JSON that must be
    imported through `smart-summary-section-apply`.
    """

    root = Path(bundle_dir).expanduser().resolve()
    exports = root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    manifest = _read_mapping(root / "manifest.json")
    workflow = _load_workflow(root, write=write)
    todo = _read_mapping(exports / "smart-summary-section-todo.json")
    sections = _editor_sections(root, manifest, workflow, todo)
    term_correction = _term_correction_status(root)
    transcript_source_decision = _smart_summary_input_transcript_source(root)
    title = str(workflow.get("title") or manifest.get("title") or root.name)
    html_path = root / "smart-summary-section-editor.html"
    json_path = root / "smart-summary-section-editor.json"
    template_path = exports / "smart-summary-section-revisions.template.json"
    semantic_review_template_path = exports / "smart-summary-section-semantic-review-notes.template.json"
    args_path = root / "mcp-smart-summary-section-editor.args.json"
    apply_args_path = root / "mcp-smart-summary-section-apply.args.json"
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "title": title,
        "created_at": now_iso(),
        "section_count": len(sections),
        "sections_needing_rewrite": sum(1 for row in sections if row.get("status") != "ready"),
        "term_correction": term_correction,
        "transcript_source_decision": transcript_source_decision,
        "sections": sections,
        "media": _media_info(root, manifest),
        "artifacts": {
            "json": str(json_path),
            "html": str(html_path),
            "revision_template": str(template_path),
            "semantic_review_notes_template": str(semantic_review_template_path),
            "mcp_editor_args": str(args_path),
            "mcp_apply_args": str(apply_args_path),
        },
        "operator_boundary": {
            "static_editor_only": True,
            "no_cloud_call": True,
            "no_direct_writeback": True,
            "apply_requires_cli_or_mcp": True,
            "purpose": "Edit smart-summary sections with transcript and evidence context, then export a reviewed revision JSON.",
        },
        "write": bool(write),
    }
    if write:
        write_json(json_path, result)
        html_path.write_text(_render_editor_html(result), encoding="utf-8")
        write_json(template_path, _revision_template(root, sections))
        write_json(semantic_review_template_path, _semantic_review_notes_template(root, sections))
        write_json(args_path, {"bundle_dir": str(root), "write": True})
        write_json(apply_args_path, {"bundle_dir": str(root), "input_json": str(root / "smart-summary-section-revisions.json"), "write": True, "require_all_sections": False})
        manifest.update(
            {
                "smart_summary_section_editor": "smart-summary-section-editor.json",
                "smart_summary_section_editor_html": "smart-summary-section-editor.html",
                "smart_summary_section_revisions_template": "exports/smart-summary-section-revisions.template.json",
                "smart_summary_section_semantic_review_notes_template": "exports/smart-summary-section-semantic-review-notes.template.json",
                "mcp_smart_summary_section_editor_args": "mcp-smart-summary-section-editor.args.json",
                "mcp_smart_summary_section_apply_args": "mcp-smart-summary-section-apply.args.json",
            }
        )
        write_json(root / "manifest.json", manifest)
        result["run_artifact"] = _register_run(root, result, write=True)
        write_json(json_path, result)
    return result


def _load_workflow(root: Path, *, write: bool) -> dict[str, Any]:
    path = root / "exports" / "smart-summary-section-workflow.json"
    if path.exists():
        value = read_json(path)
        if isinstance(value, dict) and isinstance(value.get("sections"), list):
            return value
    return build_smart_summary_section_workflow(root, write=write)


def _editor_sections(root: Path, manifest: dict[str, Any], workflow: dict[str, Any], todo: dict[str, Any]) -> list[dict[str, Any]]:
    todo_by_id = {
        str(row.get("section_id") or ""): row
        for row in todo.get("rows") or []
        if isinstance(row, dict) and row.get("section_id")
    }
    cues = _load_transcript_rows(root, manifest)
    sections: list[dict[str, Any]] = []
    for section in workflow.get("sections") or []:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("section_id") or "").strip()
        start = float(section.get("start") or 0.0)
        end = float(section.get("end") or 0.0)
        todo_row = todo_by_id.get(section_id, {})
        sections.append(
            {
                "section_id": section_id,
                "chapter_index": section.get("chapter_index"),
                "title": str(section.get("title") or section_id),
                "start": start,
                "end": end,
                "start_time": str(section.get("start_time") or format_timestamp(start)),
                "end_time": str(section.get("end_time") or format_timestamp(end)),
                "status": str(section.get("status") or ""),
                "reasons": section.get("reasons") or [],
                "rewrite_prompt": str(section.get("rewrite_prompt") or todo_row.get("rewrite_prompt") or ""),
                "draft_markdown": str(todo_row.get("draft_markdown") or ""),
                "evidence": section.get("evidence") if isinstance(section.get("evidence"), dict) else {},
                "citations": section.get("citations") or (section.get("evidence") if isinstance(section.get("evidence"), dict) else {}).get("citations") or [],
                "semantic_correction_items": section.get("semantic_correction_items") or (section.get("evidence") if isinstance(section.get("evidence"), dict) else {}).get("semantic_correction_items") or [],
                "transcript_excerpt": _transcript_excerpt(cues, start, end),
            }
        )
    return sections


def _load_transcript_rows(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        cues = _load_best_transcript(root, manifest)
    except Exception:
        cues = []
    rows = []
    for index, cue in enumerate(cues):
        start = float(getattr(cue, "start", 0.0))
        end = float(getattr(cue, "end", start))
        text = str(getattr(cue, "text", "") or "").strip()
        if text:
            rows.append({"index": index, "start": start, "end": end, "time": format_timestamp(start), "text": text})
    return rows


def _transcript_excerpt(cues: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    pad = 3.0
    rows = [row for row in cues if float(row.get("end") or 0.0) >= start - pad and float(row.get("start") or 0.0) <= end + pad]
    return rows[:40]


def _media_info(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    raw = str(manifest.get("media_path") or manifest.get("source_path") or "").strip()
    path = Path(raw).expanduser() if raw else Path()
    if raw and not path.is_absolute():
        path = (root / path).resolve()
    return {"path": str(path) if raw else "", "exists": bool(raw and path.exists()), "file_name": path.name if raw else ""}



def _smart_summary_input_transcript_source(root: Path) -> dict[str, Any]:
    pack = _read_mapping(root / "exports" / "smart-summary-input-pack.json")
    decision = pack.get("transcript_source_decision") if isinstance(pack.get("transcript_source_decision"), dict) else {}
    if decision:
        return {
            **decision,
            "available": True,
            "input_pack_path": str(root / "exports" / "smart-summary-input-pack.json"),
            "transcript_source": str(pack.get("transcript_source") or ""),
        }
    return {
        "available": False,
        "selected_label": "unknown",
        "selected_path": "",
        "uses_corrected_transcript": False,
        "priority": "input_pack_missing",
        "priority_reason": "Run build-smart-summary-input-pack to record which transcript source the summary should use.",
        "input_pack_path": str(root / "exports" / "smart-summary-input-pack.json"),
    }


def _transcript_source_panel_html(decision: dict[str, Any]) -> str:
    available = bool(decision.get("available"))
    uses_corrected = bool(decision.get("uses_corrected_transcript"))
    cls = "status-ok" if available and uses_corrected else "status-warn"
    selected = html.escape(str(decision.get("selected_label") or "unknown"))
    priority = html.escape(str(decision.get("priority") or "unknown"))
    reason = html.escape(str(decision.get("priority_reason") or ""))
    selected_path = html.escape(str(decision.get("selected_path") or ""))
    raw = html.escape(str(decision.get("raw_asr_path") or ""))
    return f"""
      <div class=\"status-grid\">
        <div class=\"status-card {cls}\"><span class=\"muted\">选中来源</span><strong>{selected}</strong></div>
        <div class=\"status-card {cls}\"><span class=\"muted\">使用纠正版</span><strong>{'yes' if uses_corrected else 'no'}</strong></div>
        <div class=\"status-card\"><span class=\"muted\">优先级</span><strong>{priority}</strong></div>
        <div class=\"status-card {'status-ok' if available else 'status-warn'}\"><span class=\"muted\">输入包</span><strong>{'ready' if available else 'missing'}</strong></div>
      </div>
      <div class=\"muted\">{reason}</div>
      <div class=\"muted\">selected: <code>{selected_path}</code></div>
      <div class=\"muted\">raw ASR: <code>{raw or 'none'}</code></div>
    """

def _term_correction_panel_html(status: dict[str, Any]) -> str:
    value = str(status.get("status") or "missing")
    accepted = int(status.get("accepted_term_count") or 0)
    final_alias_total = int(status.get("final_export_alias_total") or 0)
    quality_passed = bool(status.get("smart_summary_quality_passed"))
    source_ok = bool(status.get("source_arbitrated_transcript_exists"))
    ok = value in {"completed", "ready"} and final_alias_total == 0 and quality_passed
    cls = "status-ok" if ok else "status-warn"
    next_action = str(status.get("next_action_key") or "")
    next_line = "暂无下一步。" if not next_action else "建议先运行：" + next_action
    return f"""
      <div class=\"status-grid\">
        <div class=\"status-card {cls}\"><span class=\"muted\">状态</span><strong>{html.escape(value)}</strong></div>
        <div class=\"status-card status-ok\"><span class=\"muted\">已接受术语</span><strong>{accepted}</strong></div>
        <div class=\"status-card {'status-ok' if source_ok else 'status-warn'}\"><span class=\"muted\">纠正版转写</span><strong>{'yes' if source_ok else 'no'}</strong></div>
        <div class=\"status-card {cls}\"><span class=\"muted\">最终残留</span><strong>{final_alias_total}</strong></div>
        <div class=\"status-card {'status-ok' if quality_passed else 'status-warn'}\"><span class=\"muted\">总结质量</span><strong>{'passed' if quality_passed else 'pending'}</strong></div>
        <div class=\"status-card\"><span class=\"muted\">影响报告</span><strong>{html.escape(str(status.get('impact_status') or 'missing'))}</strong></div>
      </div>
      <div class=\"muted\">{html.escape(next_line)}</div>
    """

def _revision_template(root: Path, sections: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": REVISION_SCHEMA,
        "bundle_dir": str(root),
        "created_at": now_iso(),
        "rows": [
            {
                "section_id": row.get("section_id"),
                "title": row.get("title"),
                "time_range": f"{row.get('start_time')} - {row.get('end_time')}",
                "status": "todo" if row.get("status") != "ready" else "optional",
                "final_markdown": "",
                "review_note": "",
            }
            for row in sections
        ],
    }


def _semantic_review_notes_template(root: Path, sections: list[dict[str, Any]]) -> dict[str, Any]:
    reviews: list[dict[str, Any]] = []
    seen: set[str] = set()
    for section in sections:
        section_id = str(section.get("section_id") or "").strip()
        title = str(section.get("title") or section_id).strip()
        for item in section.get("semantic_correction_items") or []:
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("candidate_id") or "").strip()
            if not candidate_id or candidate_id in seen:
                continue
            seen.add(candidate_id)
            suggested = str(item.get("candidate_text") or item.get("suggested_text") or "").strip()
            reason = str(item.get("reason") or "").strip()
            reviews.append(
                {
                    "candidate_id": candidate_id,
                    "section_id": section_id,
                    "section_title": title,
                    "time_range": item.get("time_range") or f"{section.get('start_time')} - {section.get('end_time')}",
                    "correction_type": item.get("correction_type") or "ordinary_word",
                    "risk_level": item.get("risk_level") or "",
                    "status": "needs_more_evidence",
                    "original_text": item.get("original_text") or "",
                    "corrected_text": suggested,
                    "suggested_text": suggested,
                    "confidence": 0.0,
                    "review_note": "Generated from smart-summary section editor semantic candidate; inspect video/transcript/OCR/vision evidence before accepting. " + reason,
                    "semantic_attention": bool(item.get("semantic_attention")),
                    "source": "smart_summary_section_editor",
                    "human_confirmed": False,
                }
            )
    return {
        "schema": SEMANTIC_REVIEW_NOTES_SCHEMA,
        "bundle_dir": str(root),
        "created_at": now_iso(),
        "source": "smart_summary_section_editor",
        "review_mode": "template_needs_human_decision",
        "instructions": [
            "Edit each row status before import: accept_correction, keep_original, needs_more_evidence, needs_rerun_asr, or needs_rerun_ocr.",
            "Rows default to needs_more_evidence and are not safe to apply until reviewed.",
            "Import through import-transcript-semantic-review-notes before closure.",
        ],
        "reviews": reviews,
    }


def _render_editor_html(result: dict[str, Any]) -> str:
    title = html.escape(str(result.get("title") or "Smart Summary Section Editor"))
    data_json = json.dumps(result, ensure_ascii=False).replace("</", "<\\/")
    term_status_html = _term_correction_panel_html(result.get("term_correction") if isinstance(result.get("term_correction"), dict) else {})
    transcript_source_html = _transcript_source_panel_html(result.get("transcript_source_decision") if isinstance(result.get("transcript_source_decision"), dict) else {})
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} - Smart Summary Section Editor</title>
  <style>
    :root {{ color-scheme: light; --line:#d8dee8; --muted:#667085; --bg:#f6f7f9; --panel:#fff; --ink:#172026; --accent:#2754c5; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ padding:18px 24px; background:#fff; border-bottom:1px solid var(--line); display:flex; gap:16px; justify-content:space-between; align-items:flex-start; }}
    h1 {{ font-size:20px; margin:0 0 6px; }}
    h2 {{ font-size:15px; margin:0 0 10px; }}
    h3 {{ font-size:14px; margin:12px 0 8px; }}
    button {{ border:1px solid var(--line); background:#fff; border-radius:6px; padding:8px 10px; cursor:pointer; }}
    button.primary {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
    input, textarea, select {{ border:1px solid var(--line); border-radius:6px; padding:8px; font:inherit; }}
    textarea {{ width:100%; min-height:240px; resize:vertical; line-height:1.55; font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }}
    video {{ width:100%; max-height:360px; background:#111; border-radius:8px; }}
    main {{ display:grid; grid-template-columns:300px minmax(360px,1fr) minmax(360px,1fr); gap:12px; padding:12px; min-height:calc(100vh - 78px); }}
    .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; overflow:auto; }}
    .muted {{ color:var(--muted); }}
    .toolbar {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
    .section-row {{ width:100%; text-align:left; margin:0 0 8px; border-color:var(--line); }}
    .section-row.active {{ border-color:var(--accent); box-shadow:0 0 0 2px rgba(39,84,197,.12); }}
    .badge {{ display:inline-block; border:1px solid var(--line); border-radius:999px; padding:2px 7px; font-size:12px; margin:3px 4px 0 0; }}
    .badge.warn {{ border-color:#d59b32; background:#fff7e6; }}
    .status-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:0 0 12px; }}
    .status-card {{ border:1px solid var(--line); border-radius:8px; padding:8px; background:#fff; }}
    .status-card strong {{ display:block; font-size:16px; }}
    .status-ok {{ border-left:5px solid #0f6b4f; }}
    .status-warn {{ border-left:5px solid #b42318; }}
    .evidence-list, .transcript-list {{ max-height:280px; overflow:auto; border:1px solid var(--line); border-radius:6px; padding:8px; background:#fbfcfe; }}
    .transcript-line {{ margin:0 0 8px; }}
    pre {{ white-space:pre-wrap; word-break:break-word; background:#f1f4f8; border:1px solid var(--line); border-radius:6px; padding:8px; max-height:220px; overflow:auto; }}
    code {{ background:#f1f4f8; border-radius:4px; padding:2px 4px; }}
    @media (max-width:1100px) {{ main {{ grid-template-columns:1fr; }} video {{ max-height:300px; }} }}
  </style>
  {review_export_script(result)}
</head>
<body>
  <header>
    <div>
      <h1>{title}</h1>
      <div class="muted">智能总结章节编辑器：同屏查看视频、逐字稿、章节证据，导出 revision JSON 后用 <code>smart-summary-section-apply</code> 导入。</div>
    </div>
    <div class="toolbar">
      <button onclick="downloadRevisions()" class="primary">下载修订 JSON</button>
      <button onclick="copyRevisions()">复制 JSON</button>
      <button onclick="copyApplyCommand()">复制 apply 命令</button>
      <button onclick="downloadSemanticReviewNotes()">下载语义复核 JSON</button>
      <button onclick="copySemanticReviewNotes()">复制语义复核 JSON</button>
      <button onclick="copySemanticReviewImportCommand()">复制语义复核预检命令</button>
    </div>
  </header>
  <main>
    <section class="panel">
      <h2>术语纠错闭环</h2>
      {term_status_html}
      <h2>章节队列</h2>
      <input id="filter" placeholder="筛选章节/原因/文本" oninput="renderSectionList()" style="width:100%; margin-bottom:10px">
      <div id="sectionList"></div>
    </section>
    <section class="panel">
      <h2 id="sectionTitle">章节</h2>
      <div class="toolbar" style="margin-bottom:8px">
        <button onclick="jumpVideoToSection()">跳到本节开始</button>
        <button onclick="fillFromPrompt()">把提示放入编辑框</button>
      </div>
      <video id="player" controls></video>
      <div class="toolbar" style="margin:8px 0">
        <input id="mediaFile" type="file" accept="video/*,audio/*" onchange="loadLocalMedia(this.files[0])">
        <span class="muted" id="mediaHint"></span>
      </div>
      <h3>逐字稿片段</h3>
      <div id="transcriptBox" class="transcript-list"></div>
      <h3>证据摘要</h3>
      <div id="evidenceBox" class="evidence-list"></div>
    </section>
    <section class="panel">
      <h2>章节修订</h2>
      <div class="muted" id="revisionHint"></div>
      <textarea id="editor" oninput="saveCurrentDraft()" placeholder="在这里写这个章节最终要进入 smart-summary.md 的 Markdown 小节。"></textarea>
      <h3>重写提示</h3>
      <pre id="promptBox"></pre>
      <h3>导入命令</h3>
      <pre id="applyCommand"></pre>
    </section>
  </main>
<script id="smartSummaryEditorData" type="application/json">{data_json}</script>
<script>
const DATA = JSON.parse(document.getElementById('smartSummaryEditorData').textContent);
const STORAGE_KEY = 'vkp-smart-summary-section-editor:' + DATA.bundle_dir;
let active = 0;
let drafts = loadDrafts();
function q(id) {{ return document.getElementById(id); }}
function esc(v) {{ return String(v || '').replace(/[&<>\"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c] || c)); }}
function loadDrafts() {{ try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'); }} catch (_) {{ return {{}}; }} }}
function persistDrafts() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(drafts)); }}
function sectionDraft(row) {{ return drafts[row.section_id] || row.draft_markdown || ''; }}
function renderSectionList() {{
  const query = (q('filter').value || '').toLowerCase();
  const rows = (DATA.sections || []).filter(row => !query || JSON.stringify(row).toLowerCase().includes(query));
  q('sectionList').innerHTML = rows.map((row) => {{
    const idx = DATA.sections.indexOf(row);
    const klass = idx === active ? 'section-row active' : 'section-row';
    const warn = row.status !== 'ready' ? ' warn' : '';
    return `<button class="${{klass}}" onclick="selectSection(${{idx}})">
      <strong>${{esc(row.title)}}</strong><br>
      <span class="muted">${{esc(row.start_time)}} - ${{esc(row.end_time)}}</span><br>
      <span class="badge${{warn}}">${{esc(row.status || 'unknown')}}</span>
      ${{(row.reasons || []).slice(0,3).map(r => `<span class="badge warn">${{esc(r)}}</span>`).join('')}}
    </button>`;
  }}).join('') || '<div class="muted">没有匹配章节。</div>';
}}
function selectSection(index) {{
  saveCurrentDraft();
  active = Math.max(0, Math.min(index, (DATA.sections || []).length - 1));
  const row = DATA.sections[active] || {{}};
  q('sectionTitle').textContent = `${{row.title || row.section_id}} · ${{row.start_time || ''}} - ${{row.end_time || ''}}`;
  q('editor').value = sectionDraft(row);
  q('promptBox').textContent = row.rewrite_prompt || '';
  q('revisionHint').textContent = `section_id=${{row.section_id || ''}} · status=${{row.status || ''}}`;
  q('transcriptBox').innerHTML = (row.transcript_excerpt || []).map(t => `<div class="transcript-line"><strong>${{esc(t.time)}}</strong> ${{esc(t.text)}}</div>`).join('') || '<div class="muted">本节没有可用逐字稿片段。</div>';
  q('evidenceBox').innerHTML = renderEvidence(row.evidence || {{}});
  q('applyCommand').textContent = applyCommand();
  renderSectionList();
}}
function renderEvidence(evidence) {{
  const keys = ['summary_sentences','key_points','actions','reusable_expressions','visual_notes'];
  const chunks = keys.map(key => {{
    const values = Array.isArray(evidence[key]) ? evidence[key] : [];
    if (!values.length) return '';
    return `<h3>${{esc(key)}}</h3><ul>${{values.map(v => `<li>${{esc(v)}}</li>`).join('')}}</ul>`;
  }}).filter(Boolean);
  const row = (DATA.sections || [])[active] || {{}};
  const citations = Array.isArray(row.citations) ? row.citations : (Array.isArray(evidence.citations) ? evidence.citations : []);
  if (citations.length) {{
    chunks.push(`<h3>citations</h3><ul>${{citations.map(c => `<li><strong>${{esc(c.citation_id || '')}}</strong> <span class="badge">${{esc(c.source || '')}}</span><span class="badge">${{esc(c.chunk_kind || '')}}</span>${{c.fact_status ? `<span class="badge warn">${{esc(c.fact_status)}}</span>` : ''}} ${{esc(c.time_range || '')}}<br><span class="muted">timeline=${{esc((c.timeline_indexes || []).join(','))}}</span><br>${{esc(c.snippet || '')}}${{(c.evidence_paths || []).length ? `<br><span class="muted">evidence: ${{esc((c.evidence_paths || []).slice(0,3).join(' | '))}}</span>` : ''}}</li>`).join('')}}</ul>`);
  }}
  const semanticItems = Array.isArray(row.semantic_correction_items) ? row.semantic_correction_items : (Array.isArray(evidence.semantic_correction_items) ? evidence.semantic_correction_items : []);
  if (semanticItems.length) {{
    chunks.push(`<h3>ASR/字幕语义纠错候选</h3><ul>${{semanticItems.map(item => `<li><strong>${{esc(item.candidate_id || '')}}</strong> <span class="badge warn">${{esc(item.correction_type || '')}}</span><span class="badge warn">${{esc(item.risk_level || '')}}</span>${{item.semantic_attention ? '<span class="badge warn">semantic attention</span>' : ''}}<br><span class="muted">${{esc(item.time_range || '')}}</span><br>原文：${{esc(item.original_text || '')}}<br>建议：${{esc(item.candidate_text || item.suggested_text || '')}}<br><span class="muted">reason=${{esc(item.reason || '')}}</span></li>`).join('')}}</ul>`);
  }}
  return chunks.join('') || '<div class="muted">没有章节证据摘要。请先运行 smart-summary-section-workflow。</div>';
}}
function saveCurrentDraft() {{
  const row = (DATA.sections || [])[active];
  if (!row) return;
  drafts[row.section_id] = q('editor').value;
  persistDrafts();
}}
function fillFromPrompt() {{
  const row = (DATA.sections || [])[active] || {{}};
  if (!q('editor').value.trim()) {{
    q('editor').value = `## ${{row.title || row.section_id}}\\n\\n<!-- 基于下方重写提示改写，不要复制提示本身。 -->\\n`;
    saveCurrentDraft();
  }}
}}
function loadLocalMedia(file) {{
  if (!file) return;
  q('player').src = URL.createObjectURL(file);
  q('mediaHint').textContent = file.name;
}}
function jumpVideoToSection() {{
  const row = (DATA.sections || [])[active] || {{}};
  const player = q('player');
  player.currentTime = Math.max(0, Number(row.start || 0));
  player.play().catch(() => {{}});
}}
function revisionPayload() {{
  saveCurrentDraft();
  return {{
    schema: '{REVISION_SCHEMA}',
    bundle_dir: DATA.bundle_dir,
    created_at: new Date().toISOString(),
    rows: (DATA.sections || []).map(row => ({{
      section_id: row.section_id,
      title: row.title,
      time_range: `${{row.start_time}} - ${{row.end_time}}`,
      status: drafts[row.section_id] && drafts[row.section_id].trim() ? 'reviewed' : 'todo',
      final_markdown: drafts[row.section_id] || '',
      review_note: ''
    }}))
  }};
}}
function downloadRevisions() {{
  const blob = new Blob([JSON.stringify(revisionPayload(), null, 2)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = vkpReviewExportName('总结审核', revisionPayload());
  a.click();
}}
async function copyRevisions() {{ await navigator.clipboard.writeText(JSON.stringify(revisionPayload(), null, 2)); }}
function semanticReviewNotesPayload() {{
  const seen = new Set();
  const reviews = [];
  (DATA.sections || []).forEach(row => {{
    const evidence = row.evidence || {{}};
    const items = Array.isArray(row.semantic_correction_items) ? row.semantic_correction_items : (Array.isArray(evidence.semantic_correction_items) ? evidence.semantic_correction_items : []);
    items.forEach(item => {{
      const candidateId = String(item.candidate_id || '').trim();
      if (!candidateId || seen.has(candidateId)) return;
      seen.add(candidateId);
      const suggested = String(item.candidate_text || item.suggested_text || '').trim();
      reviews.push({{
        candidate_id: candidateId,
        section_id: row.section_id || '',
        section_title: row.title || '',
        time_range: item.time_range || `${{row.start_time || ''}} - ${{row.end_time || ''}}`,
        correction_type: item.correction_type || 'ordinary_word',
        risk_level: item.risk_level || '',
        status: 'needs_more_evidence',
        original_text: item.original_text || '',
        corrected_text: suggested,
        suggested_text: suggested,
        confidence: 0,
        review_note: 'Generated from smart-summary section editor semantic candidate; inspect video/transcript/OCR/vision evidence before accepting. ' + (item.reason || ''),
        semantic_attention: !!item.semantic_attention,
        source: 'smart_summary_section_editor',
        human_confirmed: false
      }});
    }});
  }});
  return {{
    schema: '{SEMANTIC_REVIEW_NOTES_SCHEMA}',
    bundle_dir: DATA.bundle_dir,
    created_at: new Date().toISOString(),
    source: 'smart_summary_section_editor',
    review_mode: 'template_needs_human_decision',
    instructions: [
      'Edit each row status before import: accept_correction, keep_original, needs_more_evidence, needs_rerun_asr, or needs_rerun_ocr.',
      'Rows default to needs_more_evidence and are not safe to apply until reviewed.',
      'Import through import-transcript-semantic-review-notes before closure.'
    ],
    reviews
  }};
}}
function downloadSemanticReviewNotes() {{
  const blob = new Blob([JSON.stringify(semanticReviewNotesPayload(), null, 2)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = vkpReviewExportName('转录审核', semanticReviewNotesPayload());
  a.click();
}}
async function copySemanticReviewNotes() {{ await navigator.clipboard.writeText(JSON.stringify(semanticReviewNotesPayload(), null, 2)); }}
function semanticReviewImportCommand() {{ return `.\\scripts\\video-knowledge.ps1 import-transcript-semantic-review-notes "${{DATA.bundle_dir}}" --review-json "${{DATA.bundle_dir}}\\transcript-semantic-correction-review-notes.json"`; }}
async function copySemanticReviewImportCommand() {{ await navigator.clipboard.writeText(semanticReviewImportCommand()); }}
function applyCommand() {{ return `.\\\\scripts\\\\video-knowledge.ps1 smart-summary-section-apply "${{DATA.bundle_dir}}" --input-json "${{DATA.bundle_dir}}\\\\smart-summary-section-revisions.json"`; }}
async function copyApplyCommand() {{ await navigator.clipboard.writeText(applyCommand()); }}
function boot() {{
  const media = DATA.media || {{}};
  q('mediaHint').textContent = media.path ? `媒体路径：${{media.path}}。浏览器通常需要手动选择同一视频文件。` : '没有 media_path，请手动选择视频文件。';
  selectSection(0);
}}
boot();
</script>
</body>
</html>
"""


def _register_run(root: Path, result: dict[str, Any], *, write: bool) -> dict[str, Any]:
    return register_bundle_run(
        root,
        run_type="smart_summary_section_editor",
        run_id="smart-summary-section-editor",
        status="completed",
        title="Smart summary section editor",
        summary=f"Generated static editor for {result.get('section_count', 0)} smart-summary sections.",
        artifacts=[
            {"key": "editor_html", "path": result.get("artifacts", {}).get("html")},
            {"key": "editor_json", "path": result.get("artifacts", {}).get("json")},
            {"key": "revision_template", "path": result.get("artifacts", {}).get("revision_template")},
            {"key": "mcp_args", "path": result.get("artifacts", {}).get("mcp_editor_args")},
        ],
        retry_command=f".\\scripts\\video-knowledge.ps1 smart-summary-section-editor {_ps_quote(str(root))}",
        next_actions=[
            "Open smart-summary-section-editor.html, edit sections, and download smart-summary-section-revisions.json.",
            "Run smart-summary-section-apply with the reviewed revision JSON.",
        ],
        operator_boundary=result.get("operator_boundary") if isinstance(result.get("operator_boundary"), dict) else {},
        write=write,
    )


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}
