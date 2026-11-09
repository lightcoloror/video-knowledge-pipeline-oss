from __future__ import annotations

import json
from .review_export_naming import review_export_script
from typing import Any


def render_targeted_review_html(pack: dict[str, Any], export_context: dict[str, Any] | None = None) -> str:
    data = json.dumps(pack, ensure_ascii=False).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__PACK_DATA__", data).replace("</head>", review_export_script(pack, export_context) + "</head>", 1)
    if pack.get("transcript_edit"):
        from .targeted_transcript import add_transcript_panel
        html = add_transcript_panel(html, pack)
    return html


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>VKP 精确区间审核</title>
  <style>
    @font-face { font-family:"ReviewYouYuan"; src:local("YouYuan"),local("幼圆"); font-display:swap; }
    :root { color-scheme: light; --bg:#f5f6f2; --panel:#fff; --ink:#202726; --muted:#64706d; --line:#d8dfdc; --accent:#246b70; --warn:#9c5b12; --bad:#a43b32; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:"Segoe UI","Microsoft YaHei",sans-serif; }
    header { position:sticky; top:0; z-index:5; display:flex; align-items:center; justify-content:space-between; gap:16px; padding:14px 20px; border-bottom:1px solid var(--line); background:rgba(245,246,242,.96); }
    header h1 { margin:0; font-size:20px; }
    header .summary { color:var(--muted); font-size:14px; }
    main { display:grid; grid-template-columns:minmax(270px,340px) minmax(460px,1fr) minmax(300px,390px); gap:14px; padding:14px; min-height:calc(100vh - 64px); }
    .panel { min-width:0; border:1px solid var(--line); border-radius:10px; background:var(--panel); overflow:hidden; }
    .panel-title { padding:12px 14px; border-bottom:1px solid var(--line); font-weight:700; }
    .toolbar { display:flex; flex-wrap:wrap; gap:7px; padding:10px; border-bottom:1px solid var(--line); }
    button,input,textarea { font:inherit; }
    button { border:1px solid var(--line); border-radius:7px; padding:8px 10px; background:#fff; color:var(--ink); cursor:pointer; }
    button:hover,button.active { border-color:var(--accent); background:#eaf3f2; }
    button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
    button.bad.active { background:#f8e8e5; border-color:var(--bad); }
    .queue { max-height:calc(100vh - 180px); overflow:auto; padding:8px; }
    .queue-item { display:block; width:100%; margin:0 0 7px; padding:10px; text-align:left; }
    .queue-item strong { display:block; margin-bottom:4px; }
    .chips { display:flex; flex-wrap:wrap; gap:5px; }
    .chip { padding:2px 7px; border-radius:999px; background:#edf2f0; color:var(--muted); font-size:12px; }
    .queue-item[data-decision="pass"] { border-left:5px solid #398750; }
    .queue-item[data-decision="needs_rework"] { border-left:5px solid var(--bad); }
    .queue-item[data-decision="uncertain"] { border-left:5px solid var(--warn); }
    .viewer { padding:12px; }
    .video-shell { position:relative; display:grid; place-items:center; min-height:360px; border-radius:9px; background:#111; overflow:hidden; }
    video { display:block; max-width:100%; width:auto; height:min(68vh,720px); background:#111; }
    .player-tools { display:flex; flex-wrap:wrap; align-items:center; gap:8px; padding-top:10px; }
    .player-status { flex:1; min-width:240px; color:var(--muted); font-size:13px; }
    .time-readout { color:var(--ink); font-size:13px; font-variant-numeric:tabular-nums; font-weight:700; white-space:nowrap; }
    .review-timeline { display:grid; grid-template-columns:auto minmax(72px,1fr) auto auto; align-items:center; gap:8px; padding-top:9px; color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }
    .review-timeline input { width:100%; accent-color:var(--accent); cursor:pointer; }
    .subtitle-preview-overlay { position:absolute; z-index:2; left:50%; bottom:clamp(44px,8%,64px); width:min(94%,42vh,860px); transform:translateX(-50%); pointer-events:none; color:#fff9e8; font-size:clamp(17px,2.2vw,28px); font-weight:800; line-height:1.42; text-align:center; white-space:pre-line; -webkit-text-stroke:.55px rgba(20,28,28,.9); text-shadow:0 2px 3px rgba(0,0,0,.95),0 0 8px rgba(0,0,0,.9); }
    .subtitle-preview-overlay.layered { bottom:auto; transform:none; width:auto; text-shadow:none; -webkit-text-stroke:0; }
    .review-cue { position:absolute; left:5%; width:90%; text-align:center; line-height:1.25; white-space:pre-line; font-weight:800; color:white; paint-order:stroke fill; -webkit-text-stroke:var(--cue-stroke,1px) #292429; text-shadow:0 2px 3px #000b; overflow-wrap:anywhere; }
    .review-cue[data-layer="dialogue"] { bottom:20%; }
    .review-cue[data-layer="age"] { top:7%; left:6%; width:auto; max-width:88%; text-align:left; background:#242126c9; border-radius:8px; padding:.3em .55em; font-weight:600; -webkit-text-stroke:0; }
    .review-cue[data-layer="context"] { top:17%; }
    .review-cue[data-speaker="child"] { font-family:"ReviewYouYuan","YouYuan","幼圆",sans-serif; font-weight:900; letter-spacing:.025em; color:#ff80ae; -webkit-text-stroke:var(--child-stroke,2px) #fff; text-shadow:none; filter:drop-shadow(0 2px 0 #793b62) drop-shadow(1px 0 0 #793b62) drop-shadow(-1px 0 0 #793b62) drop-shadow(0 4px 3px #31223866); }
    .review-cue[data-speaker="child"][data-emphasis="true"][data-motion="bounce"] { color:#ffc542; filter:drop-shadow(0 3px 0 #7545a0) drop-shadow(2px 0 0 #7545a0) drop-shadow(-2px 0 0 #7545a0) drop-shadow(0 5px 2px #31223866); }
    .review-cue[data-speaker="child"][data-emphasis="true"][data-motion="love"] { color:#ff6ea8; }
    .cue-letter { display:inline-block; white-space:pre; }
    .review-cue[data-speaker="father"] { color:#fff; font-weight:650; }
    .review-cue[data-layer="dialogue"][data-speaker="father"] { bottom:11%; }
    .cue-animated { display:inline-block; transform-origin:50% 70%; }
    .cue-heart { color:#ff9bb0; display:inline-block; margin:0 .16em; font-size:.65em; -webkit-text-stroke:0; }
    .timing-tools { margin-top:10px; padding:10px; border:1px solid var(--line); border-radius:8px; background:#fafaf8; }
    .timing-tools .timing-row { display:flex; gap:6px; flex-wrap:wrap; margin-top:7px; align-items:center; }
    .timing-tools input,.timing-tools select { padding:7px; border:1px solid var(--line); border-radius:6px; min-width:0; }
    #timingText { flex:1; }
    .speed-control { display:inline-flex; align-items:center; gap:6px; color:var(--muted); font-size:13px; }
    .speed-control select { padding:7px 8px; border:1px solid var(--line); border-radius:7px; background:#fff; color:var(--ink); font:inherit; }
    .player-error { margin-top:9px; padding:10px 12px; border:1px solid #ead29d; border-radius:8px; color:#70430d; background:#fff8e8; line-height:1.5; }
    .review-form { padding:14px; }
    .review-form h2 { margin:0 0 8px; font-size:21px; }
    .meta { color:var(--muted); font-size:13px; overflow-wrap:anywhere; }
    .question { margin:16px 0; padding:14px; border-left:4px solid var(--accent); background:#eef5f4; font-size:18px; line-height:1.55; }
    .purpose,.reference,.clarification { margin:10px 0; padding:10px; border:1px solid var(--line); border-radius:8px; line-height:1.5; }
    .clarification { background:#fff8e8; border-color:#ead29d; }
    .reference { white-space:pre-wrap; background:#fafaf8; }
    .decision-row { display:grid; grid-template-columns:repeat(3,1fr); gap:7px; margin:15px 0; }
    textarea { width:100%; min-height:110px; resize:vertical; padding:10px; border:1px solid var(--line); border-radius:8px; }
    .save-row { display:flex; gap:8px; align-items:center; margin-top:12px; }
    #reviewer { width:130px; padding:8px; border:1px solid var(--line); border-radius:7px; }
    .hidden { display:none !important; }
    @media (max-width:1050px) { main { grid-template-columns:300px 1fr; } .detail-panel { grid-column:1/-1; } }
    @media (max-width:600px) { .review-timeline { grid-template-columns:auto minmax(72px,1fr) auto; } .review-timeline .time-readout { grid-column:1/-1; justify-self:end; } }
  </style>
</head>
<body>
  <header><h1>VKP 精确区间审核</h1><div id="summary" class="summary"></div></header>
  <main>
    <section class="panel">
      <div class="panel-title">审核清单</div>
      <div id="filters" class="toolbar"></div>
      <div id="queue" class="queue"></div>
    </section>
    <section class="panel viewer">
      <div id="videoShell" class="video-shell">
        <video id="player" controls preload="auto"></video>
        <div id="subtitlePreviewOverlay" class="subtitle-preview-overlay hidden" aria-live="off"></div>
      </div>
      <div class="review-timeline" aria-label="审核区间或全篇时间轴">
        <span id="reviewStartLabel">00:00:00.000</span>
        <input id="reviewTimeline" type="range" min="0" max="1" step="0.001" value="0" aria-label="拖动定位原片审核时间" oninput="seekReviewRange(this.value)">
        <span id="reviewEndLabel">00:00:00.000</span>
        <span id="currentTimeReadout" class="time-readout" aria-live="off">当前 —</span>
      </div>
      <div class="player-tools">
        <button id="playbackButton" class="primary" onclick="togglePlayback()" aria-label="播放审核区间">▶ 播放审核区间</button>
        <button id="fullPlaybackButton" onclick="toggleFullPlayback()" aria-label="从头播放全篇">▶ 从头播放全篇</button>
        <button onclick="replayCurrent()">从头重播本区间</button>
        <label class="speed-control">播放速度
          <select id="playbackRate" aria-label="播放速度" onchange="setPlaybackRate(this.value)">
            <option value="0.5">0.5× 慢放</option>
            <option value="0.75">0.75× 慢放</option>
            <option value="1" selected>1× 正常</option>
            <option value="1.25">1.25×</option>
            <option value="1.5">1.5×</option>
            <option value="2">2×</option>
          </select>
        </label>
        <button onclick="stepFrame(-1)">上一帧</button>
        <button onclick="stepFrame(1)">下一帧</button>
        <span id="playerStatus" class="player-status">审核播放到区间末尾暂停；全篇播放不截停。</span>
      </div>
      <div id="playerError" class="player-error hidden" role="status" aria-live="polite"></div>
      <div class="timing-tools">
        <strong>在原片上标记一句对白</strong>
        <div class="meta">播放或拖动到对应位置，记起点、终点，填写内容后加入备注。多次重复请逐句记录，最后保存审核结果。</div>
        <div class="timing-row">
          <button onclick="captureDialogueTime('start')">记起点</button>
          <button onclick="captureDialogueTime('end')">记终点</button>
          <span id="timingMarks" class="meta">起点 — ／ 终点 —</span>
        </div>
        <div class="timing-row">
          <select id="timingSpeaker" aria-label="这句是谁说的"><option value="child">孩子</option><option value="father">爸爸</option><option value="unknown">不确定</option></select>
          <input id="timingText" aria-label="这句对白" placeholder="填写实际听到的对白" maxlength="300">
          <button onclick="appendDialogueTiming()">加入备注</button>
          <label><input id="showSubtitlePreview" type="checkbox" checked onchange="updateSubtitlePreview()">显示样式字幕</label>
        </div>
        <div id="timingFeedback" class="meta" role="status"></div>
      </div>
    </section>
    <section class="panel detail-panel">
      <div class="panel-title">本次只判断一个问题</div>
      <div class="review-form">
        <h2 id="reviewTitle">尚未选择</h2>
        <div id="reviewMeta" class="meta"></div>
        <div id="reviewQuestion" class="question">从左侧选择审核项。</div>
        <div id="decisionGuidance" class="clarification"><strong>按钮怎么选：</strong>选择与当前问题相符的结论。</div>
        <div id="reviewClarification" class="clarification hidden"></div>
        <div id="reviewPurpose" class="purpose hidden"></div>
        <div id="referenceText" class="reference hidden"></div>
        <div id="decisionButtons" class="decision-row"></div>
        <textarea id="note" placeholder="可选：若有新内容，请写大概时间和内容；若不确定，也可说明原因。"></textarea>
        <div class="save-row">
          <input id="reviewer" value="黄炳灿" aria-label="审核人">
          <button class="primary" onclick="saveToVkp()">保存审核结果</button>
          <button onclick="downloadResults()">下载 JSON</button>
        </div>
        <div id="saveStatus" class="meta"></div>
      </div>
    </section>
  </main>
  <script>
    const PACK = __PACK_DATA__;
    const SOURCES = Object.fromEntries(PACK.sources.map(row => [row.source_id, row]));
    const STORAGE_KEY = `vkp-targeted-review:${PACK.pack_payload_sha256}`;
    const state = { current: -1, filter: 'all', decisions: {}, activeSource: '' };
    const requestedReviewId = new URLSearchParams(location.search).get('review');
    const requestedReviewIndex = PACK.requests.findIndex(row => row.review_id === requestedReviewId);
    if (requestedReviewIndex >= 0) state.filter = PACK.requests[requestedReviewIndex].review_lane;
    let playbackRate = 1;
    let fullPlaybackMode = false;
    let fullPlaybackRequest = 0;
    let timeDisplayFrame = 0;
    let dialogueMarks = {start:null, end:null};
    PACK.requests.forEach(row => { state.decisions[row.review_id] = { decision: row.status || 'pending', note: '' }; });

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));
    }
    function laneLabel(value) {
      return ({audio:'声音',subtitle:'字幕',picture:'画面/动效',privacy:'隐私',compliance:'合规',publication:'发布观感',mixed:'综合'}[value] || value);
    }
    function modeLabel(value) {
      return ({normal:'正常播放',audio_only:'听声音（画面和进度条保留）',muted:'静音看画面',frame_step:'逐帧检查'}[value] || value);
    }
    function positioningLabel(row) {
      if (row.time_positioning_granularity === 'sentence_segment') return '句段级定位（词级时间不可用，已回退）';
      if (row.time_positioning_granularity === 'word') return '词级重点定位';
      return '时间区间定位';
    }
    function preciseTimecode(seconds) {
      const value = Number(seconds);
      if (!Number.isFinite(value) || value < 0) return '未知';
      const milliseconds = Math.round(value * 1000);
      const hours = Math.floor(milliseconds / 3600000);
      const minutes = Math.floor(milliseconds / 60000) % 60;
      const secs = Math.floor(milliseconds / 1000) % 60;
      const fraction = String(milliseconds % 1000).padStart(3, '0');
      return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${fraction}`;
    }
    function updateCurrentTimeReadout() {
      const player = document.getElementById('player');
      const row = currentRow();
      const timeline = document.getElementById('reviewTimeline');
      if (row) {
        const bounds = activeTimelineBounds(row);
        const currentTime = Number.isFinite(player.currentTime) ? player.currentTime : bounds.start;
        timeline.min = String(bounds.start);
        timeline.max = String(bounds.end);
        timeline.value = String(Math.max(bounds.start, Math.min(bounds.end, currentTime)));
        timeline.setAttribute('aria-label', fullPlaybackMode ? '拖动定位全篇时间' : '拖动定位原片审核时间');
        document.getElementById('reviewStartLabel').textContent = preciseTimecode(bounds.start);
        document.getElementById('reviewEndLabel').textContent = preciseTimecode(bounds.end);
      }
      document.getElementById('currentTimeReadout').textContent = row
        ? `当前 ${preciseTimecode(player.currentTime)}`
        : '当前 —';
      updateSubtitlePreview();
    }
    function updateSubtitlePreview() {
      const overlay = document.getElementById('subtitlePreviewOverlay');
      const row = currentRow();
      const player = document.getElementById('player');
      const currentTime = Number(player.currentTime);
      const cues = row && Number.isFinite(currentTime) && document.getElementById('showSubtitlePreview').checked
        ? (row.subtitle_cues || []).filter(item => currentTime >= item.start_seconds && currentTime < item.end_seconds)
        : [];
      const layered = !!row && (row.subtitle_cues || []).some(c => c.layer || c.speaker || c.motion || c.font_size_1080);
      overlay.classList.toggle('layered', layered);
      overlay.classList.toggle('hidden', !cues.length);
      if (!layered) {
        overlay.removeAttribute('style');
        overlay.textContent = cues[0]?.text || '';
        delete overlay.dataset.cueKey;
        return;
      }
      // Anchor to the displayed video rectangle, not the surrounding black shell.
      const shellRect = document.getElementById('videoShell').getBoundingClientRect();
      const rect = player.getBoundingClientRect();
      overlay.style.left = `${rect.left-shellRect.left}px`;
      overlay.style.top = `${rect.top-shellRect.top}px`;
      overlay.style.width = `${rect.width}px`;
      overlay.style.height = `${rect.height}px`;
      const key = `${row.review_id}:${JSON.stringify(cues)}`;
      if (overlay.dataset.cueKey !== key) {
        overlay.replaceChildren();
        cues.forEach(cue => {
          const box = document.createElement('div');
          box.className = 'review-cue';
          box.dataset.layer = cue.layer || 'dialogue';
          box.dataset.speaker = cue.speaker || 'neutral';
          box.dataset.motion = cue.motion || 'none';
          box.dataset.emphasis = cue.emphasis_reason ? 'true' : 'false';
          box.title = cue.emphasis_reason || '';
          const span = document.createElement('span');
          span.className = 'cue-animated';
          if (cue.speaker === 'child') {
            Array.from(cue.text).forEach(character => {
              const letter = document.createElement('span');
              letter.className = 'cue-letter'; letter.textContent = character;
              span.appendChild(letter);
            });
          } else span.textContent = cue.text;
          box.appendChild(span);
          if (cue.motion === 'love') {
            const heart = document.createElement('span');
            heart.className = 'cue-heart'; heart.textContent = '♥';
            span.appendChild(heart);
          }
          overlay.appendChild(box);
        });
        overlay.dataset.cueKey = key;
      }
      cues.forEach((cue, i) => {
        const box = overlay.children[i];
        const size = cue.font_size_1080 || (cue.layer === 'age' ? 48 : cue.speaker === 'child' ? 96 : 76);
        box.style.fontSize = `${size * rect.width / 1080}px`;
        box.style.setProperty('--cue-stroke', `${Math.max(.65, 4 * rect.width / 1080)}px`);
        box.style.setProperty('--child-stroke', `${Math.max(1.3, 9 * rect.width / 1080)}px`);
        const t = Math.max(0, currentTime - cue.start_seconds);
        const motion = cue.motion || 'none';
        let scale = 1, y = 0, rotation = 0;
        if (motion !== 'none') {
          const entrance = Math.min(1,t/.42);
          const overshoot = 1 + 2.4*Math.pow(entrance-1,3) + 1.4*Math.pow(entrance-1,2);
          scale = .66 + .34*overshoot;
          if (motion === 'bounce') {
            scale *= 1 + .08*Math.sin(t*12);
            y = -Math.abs(Math.sin(t*9))*35*rect.width/1080;
            rotation = Math.sin(t*11)*4;
          } else if (motion === 'love') {
            scale *= 1 + .035*Math.sin(t*6);
            rotation = Math.sin(t*4)*3;
          } else {
            y = -Math.sin(t*5)*12*rect.width/1080;
          }
        }
        box.firstElementChild.style.transform = `translateY(${y}px) scale(${scale}) rotate(${rotation}deg)`;
        box.querySelectorAll('.cue-letter').forEach((letter,j) => {
          const wave = motion === 'none' ? 0 : Math.sin(t*(motion==='bounce'?11:6)-j*.65)*(motion==='love'?12:motion==='bounce'?10:4)*rect.width/1080;
          letter.style.transform = `translateY(${wave}px)`;
        });
      });
    }
    window.addEventListener('resize', updateSubtitlePreview);
    function captureDialogueTime(which) {
      const player = document.getElementById('player');
      if (!currentRow() || !Number.isFinite(player.currentTime)) return;
      player.pause();
      dialogueMarks[which] = Math.round(player.currentTime*1000)/1000;
      document.getElementById('timingMarks').textContent = `起点 ${dialogueMarks.start === null ? '—' : preciseTimecode(dialogueMarks.start)} ／ 终点 ${dialogueMarks.end === null ? '—' : preciseTimecode(dialogueMarks.end)}`;
    }
    function appendDialogueTiming() {
      const row = currentRow();
      const text = document.getElementById('timingText').value.trim();
      const feedback = document.getElementById('timingFeedback');
      const {start, end} = dialogueMarks;
      if (!row || start === null || end === null || end <= start || !text || start < row.review_start_seconds || end > row.review_end_seconds) {
        feedback.textContent = '请标记当前审核区间内有效的起止时间，并填写对白；终点须晚于起点。';
        return;
      }
      const annotation = {speaker:document.getElementById('timingSpeaker').value, start_seconds:start, end_seconds:end, text};
      const note = document.getElementById('note');
      note.value = `${note.value}${note.value ? '\n' : ''}[对白定位] ${JSON.stringify(annotation)}`;
      persistCurrentNote();
      dialogueMarks = {start:null, end:null};
      document.getElementById('timingMarks').textContent = '起点 — ／ 终点 —';
      document.getElementById('timingText').value = '';
      feedback.textContent = '已加入本项备注草稿；可继续标下一句，完成后点“保存审核结果”。';
    }
    function refreshTimeDisplayWhilePlaying() {
      const player = document.getElementById('player');
      cancelAnimationFrame(timeDisplayFrame);
      if (player.paused) return;
      const tick = () => {
        updateCurrentTimeReadout();
        if (!player.paused) timeDisplayFrame = requestAnimationFrame(tick);
      };
      timeDisplayFrame = requestAnimationFrame(tick);
    }
    function seekReviewRange(value) {
      const row = currentRow();
      if (!row) return;
      const bounds = activeTimelineBounds(row);
      const time = Math.max(bounds.start, Math.min(bounds.end, Number(value)));
      document.getElementById('player').currentTime = time;
      updateCurrentTimeReadout();
    }
    function activeTimelineBounds(row) {
      if (fullPlaybackMode) {
        const source = SOURCES[row.source_id];
        const playerDuration = Number(document.getElementById('player').duration);
        const sourceDuration = Number(source?.duration_seconds);
        const duration = Number.isFinite(playerDuration) && playerDuration > 0 ? playerDuration : sourceDuration;
        if (Number.isFinite(duration) && duration > 0) return {start: 0, end: duration};
      }
      return {start: row.review_start_seconds, end: row.review_end_seconds};
    }
    function decisionLabels(row) {
      const supplied = row?.decision_labels || {};
      return {
        pass: supplied.pass || '无需补充',
        needs_rework: supplied.needs_rework || '有新内容要补充',
        uncertain: supplied.uncertain || '看不清／不确定'
      };
    }
    function renderDecisionControls(row) {
      const labels = decisionLabels(row);
      document.getElementById('decisionButtons').innerHTML = [
        ['pass', labels.pass, ''],
        ['needs_rework', labels.needs_rework, 'bad'],
        ['uncertain', labels.uncertain, '']
      ].map(([decision, label, css]) => `<button class="${css}" data-decision="${decision}" onclick="setDecision('${decision}')">${escapeHtml(label)}</button>`).join('');
      const custom = Boolean(row?.decision_labels);
      document.getElementById('decisionGuidance').textContent = custom
        ? `请选择最符合本题的结论：${labels.pass}／${labels.needs_rework}／${labels.uncertain}。必要时在备注补充细节。`
        : '按钮怎么选：无需补充表示自动记录已经够用；有新内容要补充表示有漏掉、读错或证据缺失而需要返工；看不清／不确定表示看完本段仍无法可靠判断。';
    }
    function reviewClarification(row, index) {
      const question = String(row.review_question || '');
      const ordinal = question.match(/第\s*(\d+)\s*段/);
      if (ordinal && index > 0) {
        const previous = PACK.requests[index - 1];
        return `说明：“第${ordinal[1]}段”是内部时间线编号。这里指本清单上一项 ${previous.timecode_start}–${previous.timecode_end}。请判断当前片段是否新增了上一项尚未出现的可读事实；只是同一画面延续则无需补充。`;
      }
      if (/抄袭|洗稿/.test(question)) {
        return '说明：这里只判断画面中是否有可读的对照证据，不要求你作法律或平台意义上的“抄袭”裁定。';
      }
      if (/缺可解码帧/.test(question)) {
        return '说明：如果播放器或画面无法解码、看不到本该核对的内容，选“有新内容要补充”，备注“缺可解码帧”，以便返工。';
      }
      return '';
    }
    function visibleRequests() {
      return PACK.requests.map((row,index) => ({row,index})).filter(({row}) => state.filter === 'all' || row.review_lane === state.filter);
    }
    function renderFilters() {
      const lanes = ['all', ...new Set(PACK.requests.map(row => row.review_lane))];
      document.getElementById('filters').innerHTML = lanes.map(lane => `<button class="${state.filter===lane?'active':''}" onclick="setFilter('${lane}')">${lane==='all'?'全部':laneLabel(lane)}</button>`).join('');
    }
    function renderQueue() {
      const rows = visibleRequests();
      document.getElementById('queue').innerHTML = rows.map(({row,index}) => {
        const decision = state.decisions[row.review_id]?.decision || 'pending';
        return `<button class="queue-item ${state.current===index?'active':''}" data-decision="${decision}" onclick="selectRequest(${index}, true)"><strong>${escapeHtml(row.review_question)}</strong><div class="chips"><span class="chip">${laneLabel(row.review_lane)}</span><span class="chip">${row.timecode_start}–${row.timecode_end}</span><span class="chip">${decision}</span></div></button>`;
      }).join('') || '<div class="meta">当前筛选没有审核项。</div>';
      const done = Object.values(state.decisions).filter(row => row.decision !== 'pending').length;
      document.getElementById('summary').textContent = `${done}/${PACK.requests.length} 已处理 · ${PACK.sources.length} 个审核媒体源 · 原始文件不修改`;
    }
    function setFilter(value) { state.filter = value; renderFilters(); renderQueue(); }
    function sourceUrl(source) {
      const api = window.VKP_TARGETED_REVIEW_API;
      return api ? `${api.media_url}?source_id=${encodeURIComponent(source.source_id)}` : `file:///${source.path.replace(/\\/g,'/')}`;
    }
    function unsupportedCodecMessage(source, player) {
      const codec = String(source.video_codec || '').toLowerCase();
      if (codec !== 'hevc' && codec !== 'h265' && codec !== 'h.265') return '';
      const supported = ['hvc1', 'hev1'].some(tag => player.canPlayType(`video/mp4; codecs="${tag}"`));
      return supported ? '' : `当前浏览器不支持源视频编码 HEVC/H.265（${source.video_codec}），因此无法播放这段审核素材。请为本项准备 H.264/AVC 浏览器审核代理，并保留问题中标注的原片时间码；原始文件未修改。`;
    }
    function showPlaybackError(source, errorCode) {
      const codec = source?.video_codec || '未知';
      const prefix = String(codec).toLowerCase() === 'hevc'
        ? `当前浏览器无法解码源视频 HEVC/H.265（${codec}）。`
        : `媒体播放失败（浏览器错误 ${errorCode || '未知'}，视频编码 ${codec}）。`;
      document.getElementById('playerError').textContent = `${prefix}请为本审核项准备 H.264/AVC 浏览器审核代理，并保留问题中标注的原片时间码；原始文件未修改。`;
      document.getElementById('playerError').classList.remove('hidden');
      document.getElementById('playerStatus').textContent = '无法播放源媒体；请查看上方兼容性说明。';
    }
    function clearPlaybackError() {
      const error = document.getElementById('playerError');
      error.textContent = '';
      error.classList.add('hidden');
    }
    function applyPlaybackMode(row) {
      const player = document.getElementById('player');
      player.muted = ['muted','frame_step'].includes(row.playback_mode);
      player.playbackRate = playbackRate;
      if (row.playback_mode === 'frame_step') player.pause();
    }
    function setPlaybackRate(value) {
      const offeredRates = [0.5,0.75,1,1.25,1.5,2];
      const selectedRate = Number(value);
      playbackRate = offeredRates.includes(selectedRate) ? selectedRate : 1;
      document.getElementById('playbackRate').value = String(playbackRate);
      document.getElementById('player').playbackRate = playbackRate;
      const row = currentRow();
      if (row) {
        const player = document.getElementById('player');
        const stateText = fullPlaybackMode
          ? (player.ended ? '全篇已播放完成' : player.paused ? '全篇已暂停' : '全篇连续播放中')
          : (player.paused ? '已定位，使用下方播放按钮' : '正在播放，到点自动暂停');
        const playbackScope = fullPlaybackMode ? '全篇' : `${row.timecode_start}–${row.timecode_end}`;
        document.getElementById('playerStatus').textContent = `${modeLabel(row.playback_mode)} · ${playbackRate}× · ${playbackScope} · ${stateText}`;
      }
    }
    function updatePlaybackButton(playing) {
      const button = document.getElementById('playbackButton');
      const segmentPlaying = playing && !fullPlaybackMode;
      button.textContent = segmentPlaying ? '❚❚ 暂停审核' : '▶ 播放审核区间';
      button.setAttribute('aria-label', segmentPlaying ? '暂停审核' : '播放审核区间');
    }
    function updateFullPlaybackButton() {
      const player = document.getElementById('player');
      const button = document.getElementById('fullPlaybackButton');
      const label = !fullPlaybackMode
        ? '▶ 从头播放全篇'
        : player.ended ? '↻ 从头播放全篇' : player.paused ? '▶ 继续播放全篇' : '❚❚ 暂停全篇';
      button.textContent = label;
      button.setAttribute('aria-label', label.replace(/^[^ ]+\s*/, ''));
    }
    function playWholeSource(row, restart = false) {
      const player = document.getElementById('player');
      const requestId = ++fullPlaybackRequest;
      const startPlayback = () => {
        if (requestId !== fullPlaybackRequest || !fullPlaybackMode || currentRow()?.review_id !== row.review_id) return;
        applyPlaybackMode(row);
        if (restart || player.ended) player.currentTime = 0;
        updateCurrentTimeReadout();
        player.play().then(() => {
          document.getElementById('playerStatus').textContent = `${modeLabel(row.playback_mode)} · ${playbackRate}× · 全篇连续播放中`;
          updateFullPlaybackButton();
        }).catch(() => {
          document.getElementById('playerStatus').textContent = `全篇已定位 ${preciseTimecode(player.currentTime)} · 若未自动播放，请点“继续播放全篇”`;
          updateFullPlaybackButton();
        });
      };
      if (player.readyState >= 1) {
        startPlayback();
      } else {
        document.getElementById('playerStatus').textContent = '已切换到全篇时间轴，等待媒体加载…';
        player.addEventListener('loadedmetadata', startPlayback, {once: true});
      }
    }
    function toggleFullPlayback() {
      const row = currentRow();
      if (!row) return;
      const player = document.getElementById('player');
      if (!fullPlaybackMode) {
        fullPlaybackMode = true;
        updateCurrentTimeReadout();
        updateFullPlaybackButton();
        playWholeSource(row, true);
        return;
      }
      if (!player.paused) {
        player.pause();
        return;
      }
      const bounds = activeTimelineBounds(row);
      const restart = player.ended || player.currentTime >= bounds.end;
      playWholeSource(row, restart);
    }
    function playFromCurrentPosition(row) {
      const player = document.getElementById('player');
      if (row.playback_mode === 'frame_step') return;
      applyPlaybackMode(row);
      player.play().then(() => {
        document.getElementById('playerStatus').textContent = `${modeLabel(row.playback_mode)} · ${playbackRate}× · ${row.timecode_start}–${row.timecode_end} · 正在播放，到点自动暂停`;
      }).catch(() => {
        document.getElementById('playerStatus').textContent = `${modeLabel(row.playback_mode)} · 已定位 ${row.timecode_start}–${row.timecode_end} · 若未自动播放，请点“播放审核区间”`;
        updatePlaybackButton(false);
      });
    }
    function seekAndPlay(row) {
      const player = document.getElementById('player');
      fullPlaybackMode = false;
      fullPlaybackRequest += 1;
      applyPlaybackMode(row);
      player.currentTime = row.review_start_seconds;
      updateCurrentTimeReadout();
      updateFullPlaybackButton();
      playFromCurrentPosition(row);
    }
    function showReviewStart(row, autoplay = false) {
      const player = document.getElementById('player');
      fullPlaybackMode = false;
      fullPlaybackRequest += 1;
      applyPlaybackMode(row);
      player.pause();
      player.currentTime = row.review_start_seconds;
      updateCurrentTimeReadout();
      updateFullPlaybackButton();
      document.getElementById('playerStatus').textContent = `${modeLabel(row.playback_mode)} · ${row.timecode_start}–${row.timecode_end} · 已定位，使用下方播放按钮`;
      updatePlaybackButton(false);
      if (autoplay) playFromCurrentPosition(row);
    }
    function selectRequest(index, autoplay = false) {
      dialogueMarks = {start:null, end:null};
      document.getElementById('timingMarks').textContent = '起点 — ／ 终点 —';
      document.getElementById('timingText').value = '';
      document.getElementById('timingFeedback').textContent = '';
      persistCurrentNote();
      state.current = index;
      fullPlaybackMode = false;
      fullPlaybackRequest += 1;
      updateFullPlaybackButton();
      const row = PACK.requests[index];
      const source = SOURCES[row.source_id];
      const player = document.getElementById('player');
      player.onerror = () => showPlaybackError(source, player.error?.code);
      document.getElementById('reviewTitle').textContent = `${laneLabel(row.review_lane)}审核`;
      document.getElementById('reviewMeta').textContent = `${source.label} · ${row.workflow_stage} · 回放区间 ${row.timecode_start}–${row.timecode_end} · 重点判断 ${preciseTimecode(row.issue_start_seconds)}–${preciseTimecode(row.issue_end_seconds)} · ${positioningLabel(row)} · ${modeLabel(row.playback_mode)}`;
      document.getElementById('reviewQuestion').textContent = row.review_question;
      renderDecisionControls(row);
      setOptionalText('reviewClarification', reviewClarification(row, index));
      setOptionalText('reviewPurpose', row.review_purpose === row.review_question ? '' : row.review_purpose);
      setOptionalText('referenceText', row.reference_text ? `参考说明（来源与确认状态见下文）\n\n${row.reference_text}` : '');
      document.getElementById('note').value = state.decisions[row.review_id]?.note || '';
      refreshDecisionButtons();
      renderQueue();
      const compatibilityMessage = unsupportedCodecMessage(source, player);
      if (compatibilityMessage) {
        player.pause();
        player.removeAttribute('src');
        player.load();
        state.activeSource = '';
        document.getElementById('playerError').textContent = compatibilityMessage;
        document.getElementById('playerError').classList.remove('hidden');
        document.getElementById('playerStatus').textContent = '源视频编码与当前浏览器不兼容。';
        return;
      }
      clearPlaybackError();
      if (state.activeSource !== source.source_id) {
        state.activeSource = source.source_id;
        player.src = sourceUrl(source);
        player.addEventListener('loadedmetadata', () => showReviewStart(row, autoplay), {once:true});
        player.load();
      } else {
        showReviewStart(row, autoplay);
      }
    }
    function setOptionalText(id, value) {
      const node = document.getElementById(id);
      node.textContent = value;
      node.classList.toggle('hidden', !value);
    }
    function currentRow() { return state.current >= 0 ? PACK.requests[state.current] : null; }
    function persistCurrentNote() {
      const row = currentRow();
      if (!row) return;
      state.decisions[row.review_id].note = document.getElementById('note').value;
      persistDraft();
    }
    function setDecision(value) {
      const row = currentRow();
      if (!row) return;
      state.decisions[row.review_id] = {decision:value, note:document.getElementById('note').value};
      persistDraft();
      refreshDecisionButtons();
      renderQueue();
    }
    function refreshDecisionButtons() {
      const row = currentRow();
      const value = row ? state.decisions[row.review_id]?.decision : 'pending';
      document.querySelectorAll('[data-decision]').forEach(button => {
        if (button.classList.contains('queue-item')) return;
        button.classList.toggle('active', button.dataset.decision === value);
      });
    }
    function persistDraft() {
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify({decisions:state.decisions, reviewer:document.getElementById('reviewer').value})); } catch (_) {}
    }
    function restoreDraft() {
      try {
        const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
        if (saved?.decisions) Object.assign(state.decisions, saved.decisions);
        if (saved?.reviewer) document.getElementById('reviewer').value = saved.reviewer;
      } catch (_) {}
    }
    function decisionRows() {
      persistCurrentNote();
      return PACK.requests.map(row => ({review_id:row.review_id, decision:state.decisions[row.review_id]?.decision || 'pending', note:state.decisions[row.review_id]?.note || ''}));
    }
    function replayCurrent() { const row=currentRow(); if (row) seekAndPlay(row); }
    function togglePlayback() {
      const row = currentRow();
      if (!row) return;
      const player = document.getElementById('player');
      if (fullPlaybackMode) {
        seekAndPlay(row);
        return;
      }
      if (!player.paused) { player.pause(); return; }
      if (player.currentTime < row.review_start_seconds || player.currentTime >= row.review_end_seconds) {
        player.currentTime = row.review_start_seconds;
      }
      updateCurrentTimeReadout();
      playFromCurrentPosition(row);
    }
    function stepFrame(direction) {
      const row=currentRow(); if (!row) return;
      const source=SOURCES[row.source_id]; const player=document.getElementById('player');
      const bounds=activeTimelineBounds(row);
      player.pause(); player.currentTime=Math.max(bounds.start,Math.min(bounds.end,player.currentTime + direction/(source.fps || 30)));
    }
    document.getElementById('player').addEventListener('timeupdate', event => {
      const row=currentRow(); if (!row) return;
      updateCurrentTimeReadout();
      if (fullPlaybackMode) {
        updateFullPlaybackButton();
        return;
      }
      if (event.target.currentTime >= row.review_end_seconds) {
        const source=SOURCES[row.source_id];
        event.target.pause();
        event.target.currentTime=Math.max(row.review_start_seconds, row.review_end_seconds - 1/(source.fps || 30));
        updateCurrentTimeReadout();
        document.getElementById('playerStatus').textContent=`${modeLabel(row.playback_mode)} · ${row.timecode_start}–${row.timecode_end} · 已到区间末尾，保留最后画面`;
      }
    });
    document.getElementById('player').addEventListener('ended', () => {
      if (!fullPlaybackMode) return;
      updateCurrentTimeReadout();
      updateFullPlaybackButton();
      document.getElementById('playerStatus').textContent = `全篇播放完成 · ${preciseTimecode(document.getElementById('player').duration)}`;
    });
    document.getElementById('player').addEventListener('seeked', updateCurrentTimeReadout);
    document.getElementById('player').addEventListener('loadedmetadata', updateCurrentTimeReadout);
    document.getElementById('player').addEventListener('play', () => {
      updatePlaybackButton(true);
      updateFullPlaybackButton();
      refreshTimeDisplayWhilePlaying();
    });
    document.getElementById('player').addEventListener('ratechange', event => {
      const offeredRates = [0.5,0.75,1,1.25,1.5,2];
      const actualRate = event.currentTarget.playbackRate;
      if (!offeredRates.includes(actualRate)) return;
      playbackRate = actualRate;
      document.getElementById('playbackRate').value = String(actualRate);
      const row = currentRow();
      if (row) {
        const stateText = fullPlaybackMode
          ? (event.currentTarget.ended ? '全篇已播放完成' : event.currentTarget.paused ? '全篇已暂停' : '全篇连续播放中')
          : (event.currentTarget.paused ? '已定位，使用下方播放按钮' : '正在播放，到点自动暂停');
        const playbackScope = fullPlaybackMode ? '全篇' : `${row.timecode_start}–${row.timecode_end}`;
        document.getElementById('playerStatus').textContent = `${modeLabel(row.playback_mode)} · ${actualRate}× · ${playbackScope} · ${stateText}`;
      }
    });
    document.getElementById('player').addEventListener('pause', () => {
      cancelAnimationFrame(timeDisplayFrame);
      updateCurrentTimeReadout();
      updatePlaybackButton(false);
      updateFullPlaybackButton();
    });
    document.getElementById('note').addEventListener('input', persistCurrentNote);
    document.getElementById('reviewer').addEventListener('input', persistDraft);

    async function saveToVkp() {
      const api=window.VKP_TARGETED_REVIEW_API;
      if (!api) { document.getElementById('saveStatus').textContent='静态打开模式不能写回，请下载 JSON，或通过本地 VKP 服务打开。'; return; }
      const payload={pack_payload_sha256:PACK.pack_payload_sha256, reviewer:document.getElementById('reviewer').value, decisions:decisionRows()};
      try {
        const response=await fetch(api.save_url,{method:'POST',headers:{'Content-Type':'application/json','X-VKP-Review-Token':api.token},body:JSON.stringify(payload)});
        const result=await response.json(); if(!response.ok||!result.ok) throw new Error(result.error||`HTTP ${response.status}`);
        document.getElementById('saveStatus').textContent=`已保存到 VKP：${result.status}`;
      } catch(error) { document.getElementById('saveStatus').textContent=`保存失败：${error.message||error}`; }
    }
    function downloadResults() {
      const value={schema_version:'vkp-targeted-review-decisions.draft.v1',pack_payload_sha256:PACK.pack_payload_sha256,exported_at:new Date().toISOString(),reviewer:document.getElementById('reviewer').value,decisions:decisionRows()};
      try {
        const prepared=window.vkpPrepareExactReviewDownload ? window.vkpPrepareExactReviewDownload(value) : {filename:vkpReviewExportName('综合审核',value),payload:value};
        const blob=new Blob([JSON.stringify(prepared.payload,null,2)],{type:'application/json'});
        const link=document.createElement('a'); link.href=URL.createObjectURL(blob); link.download=prepared.filename; link.click(); URL.revokeObjectURL(link.href);
        document.getElementById('saveStatus').textContent='已导出：'+prepared.filename;
      } catch(error) { document.getElementById('saveStatus').textContent='导出失败：'+(error.message||error); }
    }
    async function restoreServerState() {
      const api=window.VKP_TARGETED_REVIEW_API; if(!api && !PACK.live_visual_state) return;
      try {
        const result=api ? await (await fetch(api.status_url, {cache:'no-store'})).json() : {live_visual_state:PACK.live_visual_state};
        const live=result.live_visual_state;
        if(live && (live.status!=='review_required' || !live.pack_evidence_current)) {
          document.getElementById('player').pause();
          document.getElementById('queue').textContent=live.status==='no_candidates' ? '当前没有需要人工复核的视觉项。' : '证据已更新；此页为历史审核包，请打开新版本。';
          document.getElementById('reviewTitle').textContent='最新视觉审核状态';
          document.getElementById('reviewQuestion').textContent=live.status==='no_candidates' ? '当前无须人工处理视觉项。这不代表总体验收通过。' : '历史卡片已停用，旧决定仍然保留。';
          document.getElementById('reviewMeta').textContent=`检查时间：${live.checked_at||''} · 总体验收：${live.acceptance_status||'未知'}`;
          const ocr=live.ocr_execution||{};
          document.getElementById('summary').textContent=`视觉候选 ${live.indexes?.length??'未知'} 项 · OCR 共 ${ocr.required??'?'} 帧：本次执行 ${ocr.executed??0}，已有证据 ${ocr.existing_evidence??0}，待处理 ${ocr.remaining??'?'}，空结果 ${ocr.empty??0}`;
          document.querySelector('.viewer').classList.add('hidden');
          document.querySelector('main').style.gridTemplateColumns='1fr';
          document.querySelectorAll('.decision-row button,.save-row button').forEach(button=>button.disabled=true);
          document.getElementById('referenceText').classList.add('hidden');
          document.getElementById('reviewPurpose').classList.add('hidden');
          state.current=-1; return;
        }
        const receipt=result.decisions;
        if(receipt?.pack_payload_sha256===PACK.pack_payload_sha256) {
          receipt.decisions.forEach(row => { if(state.decisions[row.review_id]) state.decisions[row.review_id]={decision:row.decision,note:row.note||''}; });
          if(receipt.reviewer && receipt.reviewer!=='待填写') document.getElementById('reviewer').value=receipt.reviewer;
          renderQueue(); if(state.current>=0) selectRequest(state.current);
        }
        if(state.current < 0 && requestedReviewIndex >= 0) selectRequest(requestedReviewIndex);
        if(state.current < 0) selectRequest(0);
      } catch(_) {}
    }
    restoreDraft(); renderFilters(); renderQueue();
    if(!PACK.live_visual_state && !window.VKP_TARGETED_REVIEW_API) selectRequest(requestedReviewIndex >= 0 ? requestedReviewIndex : 0);
    restoreServerState();
  </script>
</body>
</html>
"""
