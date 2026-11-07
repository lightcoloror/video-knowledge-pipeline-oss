"""Self-contained download naming for offline and served review pages."""
from __future__ import annotations

import json
import re
from typing import Any


def review_export_script(context: dict[str, Any]) -> str:
    # Include only identity and time locators, never the entire transcript twice.
    rows: list[dict[str, Any]] = []
    fields = ('index', 'timeline_index', 'review_id', 'candidate_id', 'section_id', 'source_id', 'bundle_dir',
              'start', 'end', 'start_seconds', 'end_seconds', 'start_time', 'end_time',
              'review_start_seconds', 'review_end_seconds', 'time_range', 'time')

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            row = {key: value[key] for key in fields if key in value}
            if row:
                rows.append(row)
            for item in value.values():
                if isinstance(item, (dict, list)):
                    visit(item)

    visit(context)
    sources = context.get('sources') or []
    sources = sources if isinstance(sources, list) else []
    media = context.get('media') or {}
    names = [str(s.get('label') or s.get('path') or s.get('source_path') or '')
             for s in sources if isinstance(s, dict)]
    video_names = [name for name in names if re.search(r'\.(mp4|mkv|mov|ts|webm|mp3|wav|m4a)$', name, re.I)]
    title = str(next(iter(video_names), '') or context.get('source_path')
                or (media.get('path') if isinstance(media, dict) else '')
                or context.get('title') or next((n for n in names if n), '') or context.get('bundle_dir') or '未命名视频')
    # Only explicit source IDs identify multiple videos. Other sources can be
    # transcript/OCR documents belonging to the same video.
    video_count = max(len(set(video_names)), len({s.get('source_id') for s in sources if isinstance(s, dict) and s.get('source_id')}))
    payload = {'title': title, 'video_count': video_count, 'rows': rows}
    data = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    return '<script>\n' + SCRIPT.replace('__EXPORT_CONTEXT__', data) + '\n</script>'


SCRIPT = r"""(() => {
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
