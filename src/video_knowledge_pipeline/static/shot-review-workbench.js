(function () {
  'use strict'

  let state = null
  let player = null
  let regionsPlugin = null
  let wavesurfer = null
  let renderingRegions = false
  let activeShot = 0

  const q = (id) => document.getElementById(id)
  const copy = (value) => JSON.parse(JSON.stringify(value))
  const esc = (value) => String(value ?? '').replace(/[&<>\"]/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;',
  }[char] || char))

  function draftKey() {
    const template = state?.template || {}
    return `vkp-shot-review:${template.bundle_dir || 'bundle'}:${template.source_revision || 'none'}`
  }

  function loadDraft(template) {
    try {
      const saved = JSON.parse(localStorage.getItem(draftKey()) || 'null')
      if (saved?.schema === template.schema && saved?.source_revision === template.source_revision) return saved
    } catch (_) {}
    return copy(template)
  }

  function saveDraft(message = '草稿已自动保存；尚未写回 VKP。') {
    if (!state?.draft) return
    localStorage.setItem(draftKey(), JSON.stringify(state.draft))
    const status = q('shotReviewStatus')
    if (status) status.textContent = message
  }

  function reindexShots() {
    state.draft.shots.forEach((shot, index) => {
      shot.shot_id = `technical-shot-${String(index + 1).padStart(4, '0')}`
      shot.index = index + 1
    })
    const valid = new Set(state.draft.shots.map((shot) => shot.shot_id))
    state.draft.field_corrections = (state.draft.field_corrections || []).filter((row) => valid.has(row.shot_id))
  }

  function correctionValue(shotId, field) {
    return (state.draft.field_corrections || []).find((row) => row.shot_id === shotId && row.field === field)?.value || ''
  }

  function setCorrection(shotId, field, value) {
    const rows = state.draft.field_corrections || (state.draft.field_corrections = [])
    const index = rows.findIndex((row) => row.shot_id === shotId && row.field === field)
    if (!value) {
      if (index >= 0) rows.splice(index, 1)
    } else if (index >= 0) {
      rows[index].value = value
    } else {
      rows.push({ shot_id: shotId, field, value })
    }
    saveDraft()
  }

  function renderShotList() {
    const host = q('shotReviewList')
    if (!host || !state?.draft) return
    const shotTypes = ['', 'extreme_close_up', 'close_up', 'medium', 'wide', 'unknown']
    const movements = ['', 'static', 'pan_or_tilt', 'tracking', 'handheld', 'zoom', 'unknown']
    host.innerHTML = state.draft.shots.map((shot, index) => {
      const selectedType = correctionValue(shot.shot_id, 'shot_type')
      const selectedMovement = correctionValue(shot.shot_id, 'camera_movement')
      const options = (values, selected) => values.map((value) => `<option value="${esc(value)}" ${value === selected ? 'selected' : ''}>${esc(value || '保持候选值')}</option>`).join('')
      return `<div class="shot-review-row ${index === activeShot ? 'active' : ''}" data-shot-index="${index}">
        <button type="button" data-shot-action="jump" data-shot-index="${index}">#${index + 1} · ${shot.start.toFixed(3)}–${shot.end.toFixed(3)}s</button>
        <label>景别 <select data-shot-field="shot_type" data-shot-index="${index}">${options(shotTypes, selectedType)}</select></label>
        <label>运镜 <select data-shot-field="camera_movement" data-shot-index="${index}">${options(movements, selectedMovement)}</select></label>
      </div>`
    }).join('')
    host.querySelectorAll('[data-shot-action="jump"]').forEach((button) => button.addEventListener('click', () => {
      activeShot = Number(button.dataset.shotIndex || 0)
      const shot = state.draft.shots[activeShot]
      if (player && shot) { player.currentTime = shot.start; player.play().catch(() => {}) }
      renderShotList()
    }))
    host.querySelectorAll('[data-shot-field]').forEach((select) => select.addEventListener('change', () => {
      const shot = state.draft.shots[Number(select.dataset.shotIndex || 0)]
      if (shot) setCorrection(shot.shot_id, select.dataset.shotField, select.value)
    }))
  }

  function renderRegions() {
    if (!regionsPlugin || !state?.draft || renderingRegions) return
    renderingRegions = true
    regionsPlugin.clearRegions()
    state.draft.shots.forEach((shot, index) => regionsPlugin.addRegion({
      id: `shot-region-${index}`,
      start: Number(shot.start),
      end: Number(shot.end),
      drag: false,
      resize: true,
      content: `#${index + 1}`,
      color: index % 2 ? 'rgba(36,84,198,.18)' : 'rgba(15,107,79,.18)',
    }))
    renderingRegions = false
  }

  function updateFromRegion(region, side) {
    if (renderingRegions || !state?.draft) return
    const index = Number(String(region.id || '').replace('shot-region-', ''))
    const shot = state.draft.shots[index]
    if (!shot) return
    if (side === 'start' && index > 0) {
      const value = Math.max(state.draft.shots[index - 1].start + 0.01, Math.min(region.start, shot.end - 0.01))
      state.draft.shots[index - 1].end = value
      shot.start = value
    } else if (side === 'end' && index < state.draft.shots.length - 1) {
      const value = Math.max(shot.start + 0.01, Math.min(region.end, state.draft.shots[index + 1].end - 0.01))
      shot.end = value
      state.draft.shots[index + 1].start = value
    } else {
      shot.start = Number(region.start)
      shot.end = Number(region.end)
    }
    saveDraft()
    renderShotList()
    renderRegions()
  }

  function splitActive() {
    const shot = state?.draft?.shots?.[activeShot]
    const at = Number(player?.currentTime || 0)
    if (!shot || at <= shot.start + 0.05 || at >= shot.end - 0.05) return saveDraft('播放头必须位于当前镜头内部，且距边界至少 0.05 秒。')
    const left = { ...copy(shot), end: at }
    const right = { ...copy(shot), start: at }
    state.draft.shots.splice(activeShot, 1, left, right)
    reindexShots()
    saveDraft()
    renderShotList()
    renderRegions()
  }

  function mergeNext() {
    const shots = state?.draft?.shots || []
    if (activeShot < 0 || activeShot >= shots.length - 1) return saveDraft('当前镜头没有下一镜头可合并。')
    const first = shots[activeShot]
    const second = shots[activeShot + 1]
    first.end = second.end
    first.source_shot_ids = Array.from(new Set([...(first.source_shot_ids || []), ...(second.source_shot_ids || [])]))
    shots.splice(activeShot + 1, 1)
    reindexShots()
    saveDraft()
    renderShotList()
    renderRegions()
  }

  function downloadDraft() {
    const payload = preparePayload(false)
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = 'shot-review-notes.json'
    link.click()
    URL.revokeObjectURL(link.href)
  }

  function preparePayload(formal) {
    const payload = copy(state.draft)
    payload.review_status = formal ? 'human_confirmed' : 'draft'
    payload.review_id = payload.review_id || `shot-review-${Date.now()}`
    payload.reviewed_at = formal ? new Date().toISOString() : ''
    return payload
  }

  async function saveToVKP() {
    const api = window.VKP_SHOT_REVIEW_API
    if (!api?.apply_url || !api?.token) return saveDraft('当前是静态页面：已保留草稿。请从 loopback review 服务打开 Workbench 后再“保存到 VKP”。')
    const response = await fetch(api.apply_url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-VKP-Review-Token': api.token },
      body: JSON.stringify({ bundle_revision: api.bundle_revision, shot_review_notes: preparePayload(true) }),
    })
    const result = await response.json()
    if (!response.ok || !result.ok) return saveDraft(`写回失败：${result.error || result.status || response.status}`)
    localStorage.removeItem(draftKey())
    q('shotReviewStatus').textContent = '已正式保存为 VKP 派生镜头投影；请重新运行 shot-language-analysis。'
  }

  function initWaveSurfer(file) {
    if (!window.WaveSurfer || !window.WaveSurfer.Regions || !q('shotWaveform')) return
    if (!wavesurfer) {
      regionsPlugin = window.WaveSurfer.Regions.create()
      wavesurfer = window.WaveSurfer.create({
        container: '#shotWaveform', plugins: [regionsPlugin],
        waveColor: '#98a2b3', progressColor: '#2454c6', height: 72,
        interact: true,
      })
      regionsPlugin.on('region-clicked', (region, event) => {
        event.stopPropagation()
        activeShot = Number(String(region.id || '').replace('shot-region-', ''))
        if (player) player.currentTime = region.start
        renderShotList()
      })
      regionsPlugin.on('region-updated', updateFromRegion)
      wavesurfer.on('ready', renderRegions)
      wavesurfer.on('interaction', (seconds) => { if (player) player.currentTime = Number(seconds || 0) })
      player?.addEventListener('timeupdate', () => {
        if (wavesurfer && Number.isFinite(player.currentTime)) wavesurfer.setTime(player.currentTime)
      })
      const duration = Math.max(0.001, ...state.draft.shots.map((shot) => Number(shot.end || 0)))
      wavesurfer.load('', [[0, 0]], duration).catch((error) => saveDraft(`镜头时间轴载入失败：${error.message || error}`))
    }
    if (file) wavesurfer.loadBlob(file).catch((error) => saveDraft(`波形载入失败：${error.message || error}`))
  }

  window.VKPShotReview = {
    init(options) {
      state = options?.shotReview || null
      player = options?.player || q('player')
      const api = window.VKP_SHOT_REVIEW_API
      if (player && api?.media_url && !player.getAttribute('src')) player.src = api.media_url
      const panel = q('shotReviewPanel')
      if (!state?.ok || !state?.template?.shots?.length) {
        if (panel) panel.hidden = false
        if (q('shotReviewStatus')) q('shotReviewStatus').textContent = `镜头审核未就绪：${state?.status || 'missing technical shots'}`
        return
      }
      state.draft = loadDraft(state.template)
      if (panel) panel.hidden = false
      renderShotList()
      q('shotSplit')?.addEventListener('click', splitActive)
      q('shotMerge')?.addEventListener('click', mergeNext)
      q('shotDownload')?.addEventListener('click', downloadDraft)
      q('shotSave')?.addEventListener('click', saveToVKP)
      saveDraft('已加载草稿；修改会自动保存在本浏览器，尚未写回 VKP。')
      initWaveSurfer()
    },
    loadMedia(file) { initWaveSurfer(file) },
  }
})()
