from __future__ import annotations

import json
from .review_export_naming import review_export_script
from typing import Any


def render_targeted_review_html(pack: dict[str, Any]) -> str:
    data = json.dumps(pack, ensure_ascii=False).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__PACK_DATA__", data).replace("</head>", review_export_script(pack) + "</head>", 1)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>VKP 精确区间审核</title>
  <style>
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
    .audio-mask { display:none; position:absolute; inset:0; place-items:center; padding:30px; color:#fff; background:#17201f; text-align:center; font-size:20px; pointer-events:none; }
    .video-shell.audio-only .audio-mask { display:grid; }
    .video-shell.audio-only video { opacity:0; }
    .player-tools { display:flex; flex-wrap:wrap; align-items:center; gap:8px; padding-top:10px; }
    .player-status { flex:1; min-width:240px; color:var(--muted); font-size:13px; }
    .review-form { padding:14px; }
    .review-form h2 { margin:0 0 8px; font-size:21px; }
    .meta { color:var(--muted); font-size:13px; overflow-wrap:anywhere; }
    .question { margin:16px 0; padding:14px; border-left:4px solid var(--accent); background:#eef5f4; font-size:18px; line-height:1.55; }
    .purpose,.reference { margin:10px 0; padding:10px; border:1px solid var(--line); border-radius:8px; line-height:1.5; }
    .reference { white-space:pre-wrap; background:#fafaf8; }
    .decision-row { display:grid; grid-template-columns:repeat(3,1fr); gap:7px; margin:15px 0; }
    textarea { width:100%; min-height:110px; resize:vertical; padding:10px; border:1px solid var(--line); border-radius:8px; }
    .save-row { display:flex; gap:8px; align-items:center; margin-top:12px; }
    #reviewer { width:130px; padding:8px; border:1px solid var(--line); border-radius:7px; }
    .hidden { display:none !important; }
    @media (max-width:1050px) { main { grid-template-columns:300px 1fr; } .detail-panel { grid-column:1/-1; } }
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
        <video id="player" controls preload="metadata"></video>
        <div class="audio-mask">本项是“只听声音”审核。画面已临时隐藏，避免视觉信息干扰。</div>
      </div>
      <div class="player-tools">
        <button onclick="replayCurrent()">重播本区间</button>
        <button onclick="stepFrame(-1)">上一帧</button>
        <button onclick="stepFrame(1)">下一帧</button>
        <span id="playerStatus" class="player-status">选择一项后自动跳转，并在结束秒暂停。</span>
      </div>
    </section>
    <section class="panel detail-panel">
      <div class="panel-title">本次只判断一个问题</div>
      <div class="review-form">
        <h2 id="reviewTitle">尚未选择</h2>
        <div id="reviewMeta" class="meta"></div>
        <div id="reviewQuestion" class="question">从左侧选择审核项。</div>
        <div id="reviewPurpose" class="purpose hidden"></div>
        <div id="referenceText" class="reference hidden"></div>
        <div class="decision-row">
          <button data-decision="pass" onclick="setDecision('pass')">通过</button>
          <button class="bad" data-decision="needs_rework" onclick="setDecision('needs_rework')">需要修改</button>
          <button data-decision="uncertain" onclick="setDecision('uncertain')">不确定</button>
        </div>
        <textarea id="note" placeholder="可选：记录具体问题。需要修改时建议写清楚时间点和现象。"></textarea>
        <div class="save-row">
          <input id="reviewer" value="" aria-label="审核人">
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
    PACK.requests.forEach(row => { state.decisions[row.review_id] = { decision: row.status || 'pending', note: '' }; });

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));
    }
    function laneLabel(value) {
      return ({audio:'声音',subtitle:'字幕',picture:'画面/动效',privacy:'隐私',compliance:'合规',publication:'发布观感',mixed:'综合'}[value] || value);
    }
    function modeLabel(value) {
      return ({normal:'正常播放',audio_only:'只听声音',muted:'静音看画面',frame_step:'逐帧检查'}[value] || value);
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
        return `<button class="queue-item ${state.current===index?'active':''}" data-decision="${decision}" onclick="selectRequest(${index})"><strong>${escapeHtml(row.review_question)}</strong><div class="chips"><span class="chip">${laneLabel(row.review_lane)}</span><span class="chip">${row.timecode_start}–${row.timecode_end}</span><span class="chip">${decision}</span></div></button>`;
      }).join('') || '<div class="meta">当前筛选没有审核项。</div>';
      const done = Object.values(state.decisions).filter(row => row.decision !== 'pending').length;
      document.getElementById('summary').textContent = `${done}/${PACK.requests.length} 已处理 · ${PACK.sources.length} 个原视频 · 不生成审核片段`;
    }
    function setFilter(value) { state.filter = value; renderFilters(); renderQueue(); }
    function sourceUrl(source) {
      const api = window.VKP_TARGETED_REVIEW_API;
      return api ? `${api.media_url}?source_id=${encodeURIComponent(source.source_id)}` : `file:///${source.path.replace(/\\/g,'/')}`;
    }
    function applyPlaybackMode(row) {
      const player = document.getElementById('player');
      const shell = document.getElementById('videoShell');
      shell.classList.toggle('audio-only', row.playback_mode === 'audio_only');
      player.muted = ['muted','frame_step'].includes(row.playback_mode);
      if (row.playback_mode === 'frame_step') player.pause();
    }
    function seekAndPlay(row) {
      const player = document.getElementById('player');
      applyPlaybackMode(row);
      player.currentTime = row.review_start_seconds;
      if (row.playback_mode !== 'frame_step') player.play().catch(() => {});
      document.getElementById('playerStatus').textContent = `${modeLabel(row.playback_mode)} · ${row.timecode_start}–${row.timecode_end} · 到点自动暂停`;
    }
    function selectRequest(index) {
      persistCurrentNote();
      state.current = index;
      const row = PACK.requests[index];
      const source = SOURCES[row.source_id];
      const player = document.getElementById('player');
      document.getElementById('reviewTitle').textContent = `${laneLabel(row.review_lane)}审核`;
      document.getElementById('reviewMeta').textContent = `${source.label} · ${row.workflow_stage} · ${row.timecode_start}–${row.timecode_end} · ${modeLabel(row.playback_mode)}`;
      document.getElementById('reviewQuestion').textContent = row.review_question;
      setOptionalText('reviewPurpose', row.review_purpose === row.review_question ? '' : row.review_purpose);
      setOptionalText('referenceText', row.reference_text || '');
      document.getElementById('note').value = state.decisions[row.review_id]?.note || '';
      refreshDecisionButtons();
      renderQueue();
      if (state.activeSource !== source.source_id) {
        state.activeSource = source.source_id;
        player.src = sourceUrl(source);
        player.addEventListener('loadedmetadata', () => seekAndPlay(row), {once:true});
        player.load();
      } else {
        seekAndPlay(row);
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
    function stepFrame(direction) {
      const row=currentRow(); if (!row) return;
      const source=SOURCES[row.source_id]; const player=document.getElementById('player');
      player.pause(); player.currentTime=Math.max(row.review_start_seconds,Math.min(row.review_end_seconds,player.currentTime + direction/(source.fps || 30)));
    }
    document.getElementById('player').addEventListener('timeupdate', event => {
      const row=currentRow(); if (!row) return;
      if (event.target.currentTime >= row.review_end_seconds) { event.target.pause(); event.target.currentTime=row.review_end_seconds; }
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
      const value={schema_version:'vkp-targeted-review-decisions.draft.v1',pack_payload_sha256:PACK.pack_payload_sha256,reviewer:document.getElementById('reviewer').value,decisions:decisionRows()};
      const blob=new Blob([JSON.stringify(value,null,2)],{type:'application/json'});
      const link=document.createElement('a'); link.href=URL.createObjectURL(blob); link.download=vkpReviewExportName('综合审核', value); link.click(); URL.revokeObjectURL(link.href);
    }
    async function restoreServerState() {
      const api=window.VKP_TARGETED_REVIEW_API; if(!api) return;
      try {
        const response=await fetch(api.status_url); const result=await response.json();
        const receipt=result.decisions;
        if(receipt?.pack_payload_sha256===PACK.pack_payload_sha256) {
          receipt.decisions.forEach(row => { if(state.decisions[row.review_id]) state.decisions[row.review_id]={decision:row.decision,note:row.note||''}; });
          if(receipt.reviewer && receipt.reviewer!=='待填写') document.getElementById('reviewer').value=receipt.reviewer;
          renderQueue(); if(state.current>=0) selectRequest(state.current);
        }
      } catch(_) {}
    }
    restoreDraft(); renderFilters(); renderQueue(); selectRequest(0); restoreServerState();
  </script>
</body>
</html>
"""
