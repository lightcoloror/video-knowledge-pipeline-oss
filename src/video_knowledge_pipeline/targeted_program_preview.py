"""Optional, hash-bound continuous render inside the existing exact-review UI.

Reuses the native media element and ReviewHandler's byte-range serving. It does
not compile edits, save decisions or replace source-bound transcript editing.
"""
from __future__ import annotations

import hashlib
import html
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

FILENAME = 'targeted-program-preview.json'


def transcript_render_revision(review: dict[str, Any]) -> str:
    """Hash render-affecting saved data, excluding export clocks and derived totals."""
    projection = {key: review.get(key) for key in
                  ('manual_time_ranges', 'pacing_review', 'media_cut_candidates')}
    projection['chapters'] = [
        {**{key: chapter.get(key) for key in
            ('chapter', 'file', 'source_hash', 'doc', 'revisions', 'proposed_text')},
         'sections': chapter.get('preview', {}).get('sections'),
         'display_subtitles': chapter.get('preview', {}).get('displaySubtitles')}
        for chapter in review.get('chapters', [])]
    return hashlib.sha256(json.dumps(projection, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def _sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@lru_cache(maxsize=8)
def _verified_media(path: str, size: int, mtime_ns: int, expected: str) -> bool:
    return _sha(Path(path)) == expected


def load_program_preview(bundle: Path, pack: dict[str, Any]) -> dict[str, Any] | None:
    config_path = bundle / FILENAME
    if not config_path.is_file():
        return None
    data = json.loads(config_path.read_text(encoding='utf-8-sig'))
    if data.get('schema') != 'vkp.targeted-program-preview.v1':
        raise ValueError('unsupported continuous preview binding')
    if data.get('pack_payload_sha256') != pack.get('pack_payload_sha256'):
        raise ValueError('审核包已变化，需要重新生成完整预览')
    for name, expected in data.get('review_bindings', {}).items():
        if name not in {'targeted-review-pack.json', 'targeted-transcript-review.draft.json',
                        'targeted-transcript-pacing.candidates.json'}:
            raise ValueError('invalid continuous preview review binding')
        if name == 'targeted-transcript-review.draft.json' and data.get('transcript_render_sha256'):
            draft = json.loads((bundle / name).read_text(encoding='utf-8-sig'))
            current = transcript_render_revision(draft['review'])
            expected = data['transcript_render_sha256']
        else:
            current = _sha(bundle / name)
        if current != expected:
            raise ValueError('字幕或剪辑修订已变化，需要重新生成完整预览')
    if not {'targeted-review-pack.json', 'targeted-transcript-review.draft.json'}.issubset(data.get('review_bindings', {})):
        raise ValueError('continuous preview requires a saved transcript binding')
    root = Path(data['artifact_root']).resolve()
    media = Path(data['video']['path'])
    if not media.is_absolute():
        media = bundle / media
    media = media.resolve()
    if root == Path(root.anchor) or not media.is_relative_to(root) or media.suffix.lower() != '.mp4':
        raise ValueError('continuous preview media is outside its declared artifact root')
    stat = media.stat()
    if not _verified_media(str(media), stat.st_size, stat.st_mtime_ns, data['video']['sha256']):
        raise ValueError('完整预览文件已变化，需要重新绑定')
    if not 0 < float(data['duration_s']) < 24 * 3600:
        raise ValueError('invalid continuous preview duration')
    return {**data, 'media_path': media}


def render_program_preview(data: dict[str, Any]) -> str:
    title = html.escape(str(data.get('title') or '完整成片预览'))
    chapters = [{'start': float(row['start']), 'title': str(row.get('title') or '第'+str(row['chapter'])+'段')}
                for row in data.get('chapters', [])]
    chapter_json = json.dumps(chapters, ensure_ascii=False).replace('</', '<\\/')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>完整成片预览</title><style>
body{margin:0;background:#f6f5ef;color:#233b43;font:16px system-ui}main{max-width:1100px;margin:auto;padding:20px}
.layout{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap}video{background:#111;max-width:100%;height:min(78vh,960px);width:auto}
aside{max-width:420px}button{display:block;margin:8px 0;padding:9px 14px;cursor:pointer}p{line-height:1.7}
</style><main><h2>'''+title+'''</h2><div class="layout"><video id="programPlayer" controls playsinline preload="metadata" src="/targeted-program-media"></video>
<aside><p>这是连续编码的候选预览，包含当前字幕、卡片、动画和人像布局。可直接从头到尾回听，或跳转章节。</p>
<p>继续在“逐字稿修订”中修改、保存后，本预览需要重新生成。此入口不改变已有修订或审核决定。</p>
<div id="chapters"></div><p id="status" role="status"></p></aside></div></main>
<script>const player=document.getElementById('programPlayer');const chapters='''+chapter_json+''';
chapters.forEach(c=>{const b=document.createElement('button');b.textContent=c.title;b.onclick=()=>{player.currentTime=c.start;player.play()};document.getElementById('chapters').append(b)});
window.addEventListener('message',e=>{if(e.origin===location.origin&&e.data?.type==='vkp-pause-program')player.pause()});
async function verify(){try{const r=await fetch('/api/targeted-review/program/status',{cache:'no-store'});if(!r.ok){player.pause();player.removeAttribute('src');player.load();document.getElementById('status').textContent='修订或预览文件已变化，请重新生成完整预览。'}}catch(e){player.pause();document.getElementById('status').textContent='连接已中断，请刷新后继续。'}}
player.addEventListener('play',verify);document.addEventListener('visibilitychange',()=>{if(document.hidden)player.pause()});
</script></html>'''


def add_program_panel(text: str) -> str:
    """Extend the existing transcript tab controller; preserve its save/import logic."""
    # The program route must not mount the hidden editor, which may auto-save a
    # normalized export on initialization even when the author did not edit it.
    text = text.replace("new URLSearchParams(location.search).get('view')!=='exact'",
                        "!['exact','program'].includes(new URLSearchParams(location.search).get('view'))", 1)
    panel = '''<section id="programPreviewPanel" hidden><iframe id="programPreviewFrame" title="完整成片预览" style="width:100%;height:92vh;border:0"></iframe></section>
<script>
document.addEventListener('DOMContentLoaded',()=>{
const exact=document.getElementById('exactReviewMain'),transcript=document.getElementById('transcriptReviewFrame');
const panel=document.getElementById('programPreviewPanel'),frame=document.getElementById('programPreviewFrame');
const tab=document.createElement('button');tab.id='programPreviewTab';tab.textContent='完整成片预览';
const sibling=document.getElementById('transcriptReviewTab');if(!sibling)return;sibling.after(tab);
const original=window.toggleTranscriptReview;
const hide=()=>{panel.hidden=true;tab.classList.remove('primary');frame.contentWindow?.postMessage({type:'vkp-pause-program'},location.origin)};
window.toggleTranscriptReview=function(show){hide();return original(show)};
function show(){original(false);document.getElementById('player')?.pause();
exact.style.display='none';transcript.style.display='none';
transcript.contentWindow?.postMessage({type:'vkp-pause-program'},location.origin);
try{transcript.contentDocument?.querySelectorAll('video,audio').forEach(p=>p.pause())}catch(e){}
panel.hidden=false;tab.classList.add('primary');document.getElementById('exactReviewTab')?.classList.remove('primary');
document.getElementById('transcriptReviewTab')?.classList.remove('primary');
if(!frame.getAttribute('src'))frame.src='/targeted-program-preview';}
tab.onclick=show;if(new URLSearchParams(location.search).get('view')==='program')show();
});</script>'''
    return text.replace('</body>', panel+'</body>', 1)
