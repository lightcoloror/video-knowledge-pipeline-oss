"""One shared, explicit project naming/download bridge for every review page.

Merged contract for STAGE2:
- When an explicit project identity is configured (topic/video/id + Asia/Shanghai),
  use the source-bound builder and the first-party generated browser adapter.
- When no identity is configured but the page is a legacy/public producer that
  supplies a top-level title, retain the public self-contained naming fallback so
  existing offline/download behaviour and its filename contract keep working.
- When neither is available, keep the current fail-closed behaviour: the page
  loads, but an export attempt fails instead of guessing a camera filename.

The public fallback script is the public baseline implementation, retained so the
public-only no-identity fix is not lost; the bound branch is the current explicit
identity behaviour. Both are emitted as a single script with safe escaping.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .review_export_context import resolve_review_export_context

ROWS_FIELDS = (
    'index',
    'timeline_index',
    'review_id',
    'candidate_id',
    'section_id',
    'source_id',
    'bundle_dir',
    'start',
    'end',
    'start_seconds',
    'end_seconds',
    'start_time',
    'end_time',
    'review_start_seconds',
    'review_end_seconds',
    'time_range',
    'time',
)


def _collect_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect only identity/time locators, never the whole transcript twice."""
    rows: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            row = {key: value[key] for key in ROWS_FIELDS if key in value}
            if row:
                rows.append(row)
            for item in value.values():
                if isinstance(item, (dict, list)):
                    visit(item)

    visit(context)
    return rows


def _public_payload(context: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the public fallback context exactly as the public implementation did."""
    rows = _collect_rows(context)
    sources = context.get('sources') or []
    sources = sources if isinstance(sources, list) else []
    media = context.get('media') or {}
    names = [
        str(source.get('label') or source.get('path') or source.get('source_path') or '')
        for source in sources
        if isinstance(source, dict)
    ]
    video_names = [name for name in names if re.search(r'\.(mp4|mkv|mov|ts|webm|mp3|wav|m4a)$', name, re.I)]
    title = str(
        next(iter(video_names), '')
        or context.get('source_path')
        or (media.get('path') if isinstance(media, dict) else '')
        or context.get('title')
        or next((name for name in names if name), '')
        or context.get('bundle_dir')
        or '未命名视频'
    )
    source_ids = {
        source.get('source_id')
        for source in sources
        if isinstance(source, dict) and source.get('source_id')
    }
    video_count = max(len(set(video_names)), len(source_ids))
    return {'title': title, 'video_count': video_count, 'rows': rows}


def _bound_script(context: dict[str, Any], identity: dict[str, Any] | None) -> str:
    rows = _collect_rows(context)
    chapters = [
        {'chapter': chapter['number'], 'file': chapter['file'], 'source_hash': chapter['source_text_hash']}
        for chapter in (context.get('transcript_edit') or {}).get('chapters', [])
    ]
    config = {
        'context': identity,
        'chapters': chapters,
        'requestIds': [row['review_id'] for row in context.get('requests', [])],
        'rows': rows,
    }
    data = json.dumps(config, ensure_ascii=False).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    module = Path(__file__).resolve().parent / 'static/targeted-transcript-editor/export-naming-browser.js'
    script = module.read_text(encoding='utf-8').replace('</', '<\\/')
    return '<script id="vkp-review-export-naming">\n' + script + '\n' + BRIDGE.replace('__EXPORT_CONFIG__', data) + '\n</script>'


def _public_script(context: dict[str, Any]) -> str:
    data = json.dumps(_public_payload(context), ensure_ascii=False).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    return '<script>\n' + PUBLIC_FALLBACK_SCRIPT.replace('__EXPORT_CONTEXT__', data) + '\n</script>'


_UNSET = object()


def _complete_identity(identity: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return an identity only when all binding fields are explicitly present."""
    if not isinstance(identity, dict):
        return None
    if not all(str(identity.get(key) or '').strip() for key in ('topic_title', 'video_title', 'video_id')):
        return None
    if identity.get('time_zone') != 'Asia/Shanghai':
        return None
    return identity


def _has_identity_claim(
    context: dict[str, Any],
    project_context: dict[str, Any] | None,
    *,
    explicit_call: bool,
    resolved: dict[str, Any] | None,
) -> bool:
    """True when a caller claims (possibly incomplete) project identity.

    An incomplete claim is not permission to fall back to legacy filename
    guessing; it is a strict, fail-closed export path instead.
    """
    if explicit_call and project_context is not None:
        return True
    if resolved is not None:
        return True
    return bool(
        isinstance(context, dict)
        and (context.get('project_identity') or context.get('review_export_context'))
    )


def review_export_script(context: dict[str, Any], project_context: Any = _UNSET) -> str:
    """Emit the export-naming script with an explicit compatibility dispatch.

    The argument is deliberately a sentinel so an omitted argument (legacy
    one-argument callers) is distinguishable from an explicit ``None``
    (current two-argument callers). Explicit calls are strict: a missing or
    incomplete identity must block the export, never guess a filename. Legacy
    one-argument calls keep the public fallback only when no identity claim
    exists; a resolved or claimed identity takes the bound route.
    """
    explicit_call = project_context is not _UNSET
    resolved = resolve_review_export_context(context, None if not explicit_call else project_context)
    identity = _complete_identity(resolved)
    if identity is not None:
        return _bound_script(context, identity)
    if explicit_call or _has_identity_claim(context, project_context, explicit_call=explicit_call, resolved=resolved):
        # Claimed-but-incomplete identity, or a current explicit call without
        # identity: keep the page loadable but fail the export honestly.
        return _bound_script(context, None)
    # Legacy one-argument behavior with no identity claim at all.
    return _public_script(context)



PUBLIC_FALLBACK_SCRIPT = r"""(() => {
  const context = __EXPORT_CONTEXT__;
  let previousStamp = '', sequence = 0;
  const safe = value => String(value || '').split(/[\\/]/).pop()
    .replace(/\.(mp4|mkv|mov|ts|webm|mp3|wav|m4a)$/i, '')
    .replace(/[<>:"/\\|?*\u0000-\u001f]/g, '-').replace(/[. ]+$/g, '').slice(0, 72) || '未命名视频';
  function seconds(value) {
    if (typeof value === 'number') return Number.isFinite(value) && value >= 0 ? value : null;
    if (typeof value !== 'string' || !value.trim()) return null;
    const parts = value.trim().split(':');
    if (!parts.every(p => /^\d+(?:\.\d+)?$/.test(p))) return null;
    return parts.reduce((n, p) => n * 60 + Number(p), 0);
  }
  function interval(row) {
    let a = seconds(row.review_start_seconds ?? row.start_seconds ?? row.start ?? row.start_time);
    let b = seconds(row.review_end_seconds ?? row.end_seconds ?? row.end ?? row.end_time);
    if ((a === null || b === null) && (row.time_range || row.time)) {
      const pair = String(row.time_range || row.time).split(/\s*[-–—]\s*/);
      if (pair.length === 2) [a, b] = pair.map(seconds);
    }
    return a !== null && b !== null && b >= a ? [a, b] : null;
  }
  const clock = s => {
    const ms = Math.round(s * 1000), pad = n => String(n).padStart(2, '0');
    return `${pad(Math.floor(ms/3600000))}h${pad(Math.floor(ms/60000)%60)}m${pad(Math.floor(ms/1000)%60)}s` + (ms%1000 ? String(ms%1000).padStart(3,'0') : '');
  };
  window.vkpReviewExportName = (kind, payload = {}, now = new Date()) => {
    const keys = ['review_id', 'candidate_id', 'section_id', 'index', 'timeline_index'];
    const selected = payload.decisions ?? payload.reviews ?? payload.rows ?? payload.items ?? payload.suggestions;
    let rows = Array.isArray(selected) ? selected : context.rows;
    const bundles = new Set(rows.map(r => r.bundle_dir).filter(Boolean));
    const videoCount = Math.max(context.video_count, bundles.size);
    rows = rows.map(row => {
      if (interval(row)) return row;
      const key = keys.find(k => row[k] !== undefined && row[k] !== null);
      return key ? context.rows.find(r => String(r[key] ?? (key === 'timeline_index' ? r.index : undefined)) === String(row[key]) && (!row.bundle_dir || r.bundle_dir === row.bundle_dir) && interval(r)) || row : row;
    });
    const intervals = rows.map(interval).filter(Boolean).sort((a,b) => a[0]-b[0]);
    let scope = '范围见JSON';
    if (rows.length && intervals.length === rows.length) {
      const merged = [];
      for (const span of intervals) {
        const last = merged[merged.length-1];
        if (last && span[0] <= last[1]) last[1] = Math.max(last[1], span[1]);
        else merged.push([...span]);
      }
      scope = merged.length === 1 ? `${clock(merged[0][0])}-${clock(merged[0][1])}` : `多片段-${rows.length}项`;
    } else if (Array.isArray(selected) && !selected.length) scope = '空审核-0项';
    if (payload.review_scope === 'full_video') scope = '全视频';
    if (videoCount > 1) scope = `多视频-${rows.length}项`;
    const iso = new Date(now.getTime()+8*3600000).toISOString();
    const stamp = iso.slice(0,10).replace(/-/g,'')+'-'+iso.slice(11,23).replace(/[:.]/g,'');
    sequence = stamp === previousStamp ? sequence + 1 : 0; previousStamp = stamp;
    const title = videoCount > 1 ? `多视频-${videoCount}个-${safe(context.title)}` : safe(context.title);
    return `${title}__${safe(kind)}__${scope}__${stamp}${sequence ? '-'+sequence : ''}.json`;
  };
})();"""


BRIDGE = r"""(() => {
  const config=__EXPORT_CONFIG__;
  let previousName='',sameNameCount=0;
  const feedback=(text,error=false)=>{
    let box=document.getElementById('saveStatus')||document.getElementById('vkpReviewExportStatus');
    if(!box){box=document.createElement('p');box.id='vkpReviewExportStatus';box.setAttribute('role','status');document.body.prepend(box);}
    box.textContent=text;box.style.color=error?'#9c2f25':'';
  };
  const prepare=(kind,value,now=new Date())=>{
    const result=value.schema_version==='vkp-targeted-review-decisions.draft.v1'
      ? VKPProjectReviewExport.prepareExactReviewExport({...value,exported_at:value.exported_at||now.toISOString()},config.context,config.chapters,config.requestIds)
      : VKPProjectReviewExport.prepareProjectReviewExport(kind,value,config.context,config.rows,now);
    const base=result.filename;sameNameCount=base===previousName?sameNameCount+1:0;previousName=base;
    if(sameNameCount){result.filename=base.replace(/\.json$/,`_${sameNameCount}.json`);result.payload.export_identity.filename=result.filename;}
    return result;
  };
  window.vkpPrepareReviewDownload=prepare;
  window.vkpPrepareExactReviewDownload=(value,now=new Date())=>prepare('精确审核结果',value,now);
  window.vkpReviewExportName=(kind,value={},now=new Date())=>prepare(kind,value,now).filename;
  window.vkpDownloadReview=(kind,value)=>{
    try{
      const result=prepare(kind,value);const blob=new Blob([JSON.stringify(result.payload,null,2)],{type:'application/json'});
      const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=result.filename;link.click();URL.revokeObjectURL(url);
      feedback('已导出：'+result.filename);return result;
    }catch(error){feedback('导出失败：'+(error.message||error),true);return null;}
  };
  if((!config.context?.topic_title||!config.context?.video_title||!config.context?.video_id)&&typeof document!=='undefined')document.addEventListener('DOMContentLoaded',()=>feedback('导出前请在项目配置中补充选题、视频名称和视频标识。',true));
})();"""
