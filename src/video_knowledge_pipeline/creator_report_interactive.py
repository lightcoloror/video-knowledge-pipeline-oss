"""Explicit local playback view reusing the workbench's seek/play behavior.

Static reading exports keep their existing CSP. This opt-in document uses only
hashed bundled JavaScript, data images and a user-selected local blob; it cannot
fetch arbitrary media URLs, read filesystem paths, or send data over a network.
"""
from __future__ import annotations

import base64
import hashlib
import html
import math
import re
from typing import Any

from markdown_it import MarkdownIt

from .creator_report_renderers import CSS
from .transcript import parse_timestamp

_CLOCK = re.compile(r"(?<!\d)(\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?)(?!\d)")
_SCRIPT = r"""(() => {
  'use strict';
  const player = document.getElementById('reportPlayer');
  const file = document.getElementById('reportMediaFile');
  const message = document.getElementById('reportPlayerStatus');
  let blob = null;
  const say = (text) => { message.textContent = text; };
  file.addEventListener('change', () => {
    if (blob) { URL.revokeObjectURL(blob); blob = null; }
    player.removeAttribute('src'); player.load();
    const selected = file.files && file.files[0];
    if (!selected) { say('尚未选择原片；文字和图片仍可阅读。'); return; }
    if (selected.type && !/^(video|audio)\//.test(selected.type)) {
      say('所选文件不是可识别的音视频，请重新选择原片。'); return;
    }
    blob = URL.createObjectURL(selected); player.src = blob; player.load();
    say('已选择本地文件，正在读取；文件不会上传。请确认它是本报告对应的原片。');
  });
  player.addEventListener('loadedmetadata', () => {
    say(Number.isFinite(player.duration) && player.duration > 0
      ? '所选媒体已载入（未核验身份）；确认是对应原片后，可点击章节、字幕时间或代表图回看。'
      : '无法读取媒体时长；请更换可解码的原片。');
  });
  player.addEventListener('error', () => say('本地媒体无法解码或已不可用，请重新选择原片。'));
  document.addEventListener('click', (event) => {
    const button = event.target.closest('[data-seek]');
    if (!button) return;
    const target = Number(button.dataset.seek);
    if (!Number.isFinite(target) || target < 0) { say('该时间点无效，未跳转。'); return; }
    if (!player.currentSrc || player.readyState < 1 || !Number.isFinite(player.duration)) {
      say('请先选择并载入本报告对应的原片。'); return;
    }
    if (target > player.duration) { say('该时间超过所选媒体时长，请检查是否选错原片。'); return; }
    try {
      player.currentTime = target;
      // Reuse shot-review-workbench's currentTime + play contract. A browser
      // autoplay refusal leaves the seek intact and exposes a manual action.
      player.play().catch(() => say('已跳转；浏览器未自动播放，请点击播放器播放。'));
      say('已定位到 ' + target.toFixed(3) + ' 秒。');
    } catch (_) { say('该媒体不支持跳转到此时间，请检查文件。'); }
  });
  window.addEventListener('pagehide', () => { if (blob) URL.revokeObjectURL(blob); });
})();"""


def interactive_html_document(markdown: str, title: str, report: dict[str, Any]) -> str:
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    def text_rule(tokens, index, options, env):
        raw, last, parts = tokens[index].content, 0, []
        for match in _CLOCK.finditer(raw):
            parts.append(html.escape(raw[last:match.start()]))
            seconds = parse_timestamp(match[1])
            if math.isfinite(seconds) and seconds >= 0:
                parts.append(f'<button class="seek-time" type="button" data-seek="{seconds:.3f}">{html.escape(match[1])}</button>')
            else:
                parts.append(html.escape(match[1]))
            last = match.end()
        return "".join([*parts, html.escape(raw[last:])])
    md.renderer.rules["text"] = text_rule
    image_times = {}
    for preview in report.get("chapter_previews", {}).values():
        image_times[preview["data_uri"]] = preview["time"]
    for shot in report.get("shots", []):
        preview = report.get("previews", {}).get(shot.get("shot_id"))
        if preview:
            image_times.setdefault(preview["data_uri"], shot["start"])
    image_rule = md.renderer.rules["image"]
    def seek_image(tokens, index, options, env):
        value = image_rule(tokens, index, options, env)
        seconds = image_times.get(tokens[index].attrGet("src"))
        if seconds is None:
            return value
        return f'<button class="seek-frame" type="button" data-seek="{float(seconds):.3f}" aria-label="回看代表帧对应时间">{value}</button>'
    md.renderer.rules["image"] = seek_image
    body = md.render(markdown)
    script_hash = base64.b64encode(hashlib.sha256(_SCRIPT.encode()).digest()).decode()
    csp = ("default-src 'none'; img-src data:; style-src 'unsafe-inline'; "
           f"script-src 'sha256-{script_hash}'; media-src blob:; connect-src 'none'; "
           "base-uri 'none'; form-action 'none'; object-src 'none'")
    controls = ('<section class="playback"><strong>本地原片回看</strong>'
        '<p>这是参考报告。请选择对应原片，文件仅在浏览器本地播放，不会上传；文字和图片无需原片也可阅读。</p>'
        '<label>选择本地原片 <input id="reportMediaFile" type="file" accept="video/*,audio/*"></label>'
        '<video id="reportPlayer" controls preload="metadata"></video>'
        '<p id="reportPlayerStatus" role="status" aria-live="polite">尚未选择原片；点击时间前请先选片。</p></section>')
    extra_css = '.playback{position:sticky;top:0;background:#fff;border:1px solid #ddd;padding:12px;z-index:2}.playback video{max-height:280px;width:100%}.seek-time{font:inherit;color:#175d8d;background:none;border:0;text-decoration:underline;cursor:pointer;padding:0}.seek-frame{display:block;max-width:100%;border:0;background:none;margin:auto;cursor:pointer}.seek-frame img{margin:0}'
    return ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f'<meta http-equiv="Content-Security-Policy" content="{html.escape(csp, quote=True)}">'
            f"<title>{html.escape(title)}</title><style>{CSS}{extra_css}</style><body>"
            f"{controls}{body}<script>{_SCRIPT}</script></body></html>")
