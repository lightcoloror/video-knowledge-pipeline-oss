/*
 * VKP adapter layered after the unmodified moys-asr-workflow editor.
 * It reuses the upstream playback/waveform/editing algorithms and only adds
 * dual-track lineage, local draft persistence, and explicit VKP apply.
 */
(function installVkpSubtitleAdapter() {
  'use strict';

  const cfg = SERVER_CONFIG && SERVER_CONFIG.vkp;
  if (!cfg) return;

  const originalSegments = Array.isArray(cfg.projection?.segments)
    ? cfg.projection.segments : [];
  const draftKey = `vkp:subtitle-editor:${cfg.bundleId}:${cfg.projection.projection_sha256}`;
  let lastDraftJson = '';
  let formallyApplied = false;
  let translationGeneration = 0;
  let translationAbortController = null;
  let translationObserver = null;
  let transcriptMode = 'bilingual';
  const pendingTranslationIds = new Set();
  const timestampNotes = new Map();

  function normalizedSegment(segment, index) {
    const sourceIds = Array.isArray(segment.source_segment_ids) && segment.source_segment_ids.length
      ? segment.source_segment_ids.map(String)
      : [String(segment.segment_id || `segment-${index + 1}`)];
    const lineageIds = Array.isArray(segment.source_lineage_ids) && segment.source_lineage_ids.length
      ? segment.source_lineage_ids.map(String) : [...sourceIds];
    return {
      segment_id: String(segment.segment_id || `review-${index + 1}`),
      source_segment_ids: sourceIds,
      source_lineage_ids: lineageIds,
      start_ms: Math.round(Number(segment.start || 0)),
      end_ms: Math.round(Number(segment.end || 0)),
      speaker_global_id: String(segment.speaker_global_id || segment.speaker || ''),
      speaker_role: String(segment.speaker_role || ''),
      source_text: String(segment.text || '').trim(),
      mandarin_text: String(segment.mandarin_text || '').trim(),
      mandarin_loaded: segment.mandarin_loaded !== false,
      translation_available: Boolean(segment.translation_available),
      words: Array.isArray(segment.items)
        ? segment.items.map((item) => ({
            text: String(item.text || ''),
            start_ms: Math.round(Number(item.start || 0)),
            end_ms: Math.round(Number(item.end || item.start || 0)),
            ...(item.speaker ? { speaker_global_id: String(item.speaker) } : {}),
          }))
        : [],
      evidence_ids: Array.isArray(segment.evidence_ids) ? segment.evidence_ids.map(String) : [],
      disabled: Boolean(segment.disabled),
      needs_translation_review: Boolean(segment.needs_translation_review),
      timing_status: String(segment.timing_status || 'ready'),
    };
  }

  function reviewNotes(humanConfirmed) {
    const currentSegmentIds = new Set(DATA.segments.map((segment) => String(segment.segment_id)));
    return {
      schema: 'video_knowledge_pipeline.subtitle_review_notes.v1',
      projection_sha256: cfg.projection.projection_sha256,
      source_sha256: cfg.projection.source_sha256,
      segments: DATA.segments.map(normalizedSegment),
      timestamp_notes: [...timestampNotes.values()]
        .filter((note) => currentSegmentIds.has(String(note.segment_id)))
        .map((note) => ({ ...note })),
      gap_remove: DATA.gap_remove ? JSON.parse(JSON.stringify(DATA.gap_remove)) : null,
      human_confirmed: Boolean(humanConfirmed),
    };
  }

  function setState(text, state) {
    const el = document.getElementById('vkp-review-state');
    if (!el) return;
    el.textContent = text;
    el.dataset.state = state;
  }

  function markDraft() {
    formallyApplied = false;
    setState('草稿已自动保存 · 尚未写回 VKP', 'draft');
  }

  function persistDraft() {
    const json = JSON.stringify(reviewNotes(false));
    if (json === lastDraftJson) return;
    localStorage.setItem(draftKey, json);
    lastDraftJson = json;
    markDraft();
  }

  function restoreDraft() {
    const raw = localStorage.getItem(draftKey);
    if (!raw) return;
    try {
      const draft = JSON.parse(raw);
      if (draft.projection_sha256 !== cfg.projection.projection_sha256) {
        setState('旧草稿与当前 Bundle 冲突，已禁止覆盖', 'conflict');
        return;
      }
      if (!Array.isArray(draft.segments) || !draft.segments.length) return;
      DATA.segments = draft.segments.map((segment) => ({
        start: segment.start_ms,
        end: segment.end_ms,
        text: segment.source_text,
        mandarin_text: segment.mandarin_text || '',
        mandarin_loaded: segment.mandarin_loaded !== false,
        translation_available: Boolean(segment.translation_available || segment.mandarin_text),
        segment_id: segment.segment_id,
        source_segment_ids: segment.source_segment_ids,
        source_lineage_ids: segment.source_lineage_ids || segment.source_segment_ids,
        speaker: segment.speaker_global_id || null,
        speaker_global_id: segment.speaker_global_id || '',
        speaker_role: segment.speaker_role || '',
        evidence_ids: segment.evidence_ids || [],
        items: (segment.words || []).map((word) => ({
          text: word.text,
          start: word.start_ms,
          end: word.end_ms,
          ...(word.speaker_global_id ? { speaker: word.speaker_global_id } : {}),
        })),
        disabled: Boolean(segment.disabled),
        needs_translation_review: Boolean(segment.needs_translation_review),
        timing_status: String(segment.timing_status || 'ready'),
        _dirty: true,
      }));
      timestampNotes.clear();
      (draft.timestamp_notes || []).forEach((note) => {
        if (note && note.note_id && note.segment_id) timestampNotes.set(String(note.note_id), { ...note });
      });
      lastDraftJson = raw;
      renderAll();
      setState('已恢复本地草稿 · 尚未写回 VKP', 'draft');
    } catch (error) {
      setState(`草稿损坏：${error.message || error}`, 'error');
    }
  }

  function currentSegment() {
    return Number.isInteger(currentCuePanelIdx) && currentCuePanelIdx >= 0
      ? DATA.segments[currentCuePanelIdx] : null;
  }

  function sourceQuoteForSegment(segment) {
    if (!segment) return '';
    const wanted = new Set((segment.source_lineage_ids || segment.source_segment_ids || []).map(String));
    return originalSegments
      .filter((source) => (source.source_lineage_ids || source.source_segment_ids || [])
        .some((lineage) => wanted.has(String(lineage))))
      .map((source) => String(source.source_text || '').trim())
      .filter(Boolean)
      .join(' ');
  }

  function bumpTranslationGeneration() {
    translationGeneration += 1;
    translationAbortController?.abort();
    translationAbortController = null;
    pendingTranslationIds.clear();
  }

  let translationFlushTimer = null;
  let translationRequestInFlight = false;

  function queueTranslation(segmentId) {
    if (!cfg.lazyTranslation || transcriptMode === 'original') return;
    const segment = DATA.segments.find((row) => String(row.segment_id) === String(segmentId));
    if (!segment || segment.mandarin_loaded !== false || !segment.translation_available) return;
    pendingTranslationIds.add(String(segmentId));
    if (translationFlushTimer) return;
    translationFlushTimer = window.setTimeout(flushTranslationQueue, 30);
  }

  async function flushTranslationQueue() {
    translationFlushTimer = null;
    if (translationRequestInFlight || !pendingTranslationIds.size) return;
    const ids = [...pendingTranslationIds].slice(0, 4);
    ids.forEach((id) => pendingTranslationIds.delete(id));
    const generation = translationGeneration;
    const params = new URLSearchParams({
      projection_sha256: cfg.projection.projection_sha256,
      generation: String(generation),
    });
    ids.forEach((id) => params.append('segment_id', id));
    translationRequestInFlight = true;
    translationAbortController = new AbortController();
    try {
      const response = await fetch(`${cfg.translationUrl}?${params}`, {
        signal: translationAbortController.signal,
        headers: { Accept: 'application/json' },
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      if (generation !== translationGeneration || Number(payload.generation) !== generation) return;
      (payload.segments || []).forEach((row) => {
        const segment = DATA.segments.find((candidate) => String(candidate.segment_id) === String(row.segment_id));
        if (!segment) return;
        segment.mandarin_text = String(row.text || '');
        segment.mandarin_loaded = true;
        segment.translation_available = row.status === 'ready';
      });
      syncMandarinEditor();
      refreshMandarinPreviews();
    } catch (error) {
      if (error?.name !== 'AbortError' && generation === translationGeneration) {
        ids.forEach((id) => {
          const segment = DATA.segments.find((candidate) => String(candidate.segment_id) === id);
          if (segment) segment.translation_load_error = String(error.message || error);
        });
        refreshMandarinPreviews();
      }
    } finally {
      translationRequestInFlight = false;
      translationAbortController = null;
      if (pendingTranslationIds.size) translationFlushTimer = window.setTimeout(flushTranslationQueue, 0);
    }
  }

  function installTranslationObserver() {
    translationObserver?.disconnect();
    translationObserver = null;
    if (!cfg.lazyTranslation || transcriptMode === 'original' || !('IntersectionObserver' in window)) return;
    const generation = translationGeneration;
    translationObserver = new IntersectionObserver((entries) => {
      if (generation !== translationGeneration) return;
      entries.filter((entry) => entry.isIntersecting).forEach((entry) => {
        const index = Number(entry.target.dataset.idx);
        const segment = DATA.segments[index];
        if (segment) queueTranslation(segment.segment_id);
      });
    }, { rootMargin: '240px 0px' });
    document.querySelectorAll('.cue[data-idx]').forEach((cue) => translationObserver.observe(cue));
  }

  function setTranscriptMode(mode) {
    if (!['original', 'mandarin', 'bilingual'].includes(mode)) return;
    bumpTranslationGeneration();
    transcriptMode = mode;
    document.body.dataset.vkpTranscriptMode = mode;
    document.querySelectorAll('[data-vkp-transcript-mode]').forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.vkpTranscriptMode === mode));
    });
    installTranslationObserver();
    syncMandarinEditor();
    refreshMandarinPreviews();
  }

  function syncMandarinEditor() {
    const input = document.getElementById('vkp-mandarin-text');
    const warning = document.getElementById('vkp-translation-review');
    if (!input) return;
    const segment = currentSegment();
    input.disabled = !segment;
    if (segment?.mandarin_loaded === false) queueTranslation(segment.segment_id);
    if (document.activeElement !== input) input.value = segment?.mandarin_text || '';
    if (warning) {
      warning.hidden = !segment?.needs_translation_review;
      warning.textContent = segment?.needs_translation_review
        ? '此段在原文拆分后仍需人工确认普通话拆分点。' : '';
    }
  }

  function refreshMandarinPreviews() {
    document.querySelectorAll('.cue[data-idx]').forEach((cue) => {
      const index = Number(cue.dataset.idx);
      const segment = DATA.segments[index];
      const anchor = cue.querySelector('.text') || cue.querySelector('.cue-text');
      if (!anchor || !segment) return;
      let preview = cue.querySelector('.vkp-mandarin-preview');
      if (!preview) {
        preview = document.createElement('div');
        preview.className = 'vkp-mandarin-preview';
        anchor.insertAdjacentElement('afterend', preview);
        preview.addEventListener('click', () => queueTranslation(DATA.segments[Number(cue.dataset.idx)]?.segment_id));
      }
      preview.dataset.translationState = segment.mandarin_loaded === false
        ? (segment.translation_load_error ? 'error' : 'loading')
        : (segment.mandarin_text ? 'ready' : 'missing');
      preview.textContent = segment.mandarin_loaded === false
        ? (segment.translation_load_error ? '翻译加载失败，点击重试' : '翻译将在进入视口时加载…')
        : (segment.mandarin_text || '（暂无普通话翻译）');
    });
  }

  function syncTimestampNoteEditor() {
    const segment = currentSegment();
    const original = document.getElementById('vkp-note-original');
    const polished = document.getElementById('vkp-note-polished');
    const noteText = document.getElementById('vkp-note-text');
    const deleteButton = document.getElementById('vkp-note-delete');
    if (!original || !polished || !noteText) return;
    const noteId = segment ? `note:${segment.segment_id}` : '';
    const note = noteId ? timestampNotes.get(noteId) : null;
    const quote = sourceQuoteForSegment(segment);
    original.textContent = quote || '当前段没有可绑定的原始证据';
    polished.disabled = noteText.disabled = !segment || !quote;
    if (document.activeElement !== polished) polished.value = note?.polished_quote || '';
    if (document.activeElement !== noteText) noteText.value = note?.note_text || '';
    if (deleteButton) deleteButton.disabled = !note;
  }

  function updateTimestampNote() {
    const segment = currentSegment();
    if (!segment) return;
    const originalQuote = sourceQuoteForSegment(segment);
    if (!originalQuote) return;
    const noteId = `note:${segment.segment_id}`;
    const polishedQuote = String(document.getElementById('vkp-note-polished')?.value || '').trim();
    const noteText = String(document.getElementById('vkp-note-text')?.value || '').trim();
    if (!polishedQuote && !noteText) {
      timestampNotes.delete(noteId);
    } else {
      timestampNotes.set(noteId, {
        note_id: noteId,
        segment_id: String(segment.segment_id),
        source_segment_ids: [...(segment.source_segment_ids || [])],
        timestamp_ms: Math.round(Number(segment.start || 0)),
        original_quote: originalQuote,
        polished_quote: polishedQuote,
        note_text: noteText,
      });
    }
    persistDraft();
    syncTimestampNoteEditor();
  }

  function installChrome() {
    // Upstream has hidden panel-local <header> elements. Anchor the VKP bar to
    // the top-level title so it never lands inside a closed workspace panel.
    const header = document.querySelector('body > h1') || document.body.firstElementChild;
    const bar = document.createElement('div');
    bar.className = 'vkp-review-bar';
    bar.innerHTML = `
      <strong>VKP 双轨字幕审核</strong>
      <span class="vkp-transcript-modes" role="group" aria-label="逐字稿显示模式">
        <button type="button" data-vkp-transcript-mode="original">原文</button>
        <button type="button" data-vkp-transcript-mode="mandarin">中文</button>
        <button type="button" data-vkp-transcript-mode="bilingual" aria-pressed="true">双语</button>
      </span>
      <span class="vkp-boundary-note">${cfg.projection.timing_review?.status === 'needs_review'
        ? `检测到 ${cfg.projection.timing_review.overlap_count} 处原始时间重叠，必须校正后才能正式写回；`
        : ''}原始 ASR / Timeline 不会被覆盖；静音区与剪辑导出仅生成计划</span>
      <span class="vkp-spacer"></span>
      <span class="vkp-review-state" id="vkp-review-state" data-state="clean">尚无草稿</span>
      <button type="button" id="vkp-validate-review">校验草稿</button>
      <button type="button" id="vkp-apply-review">保存到 VKP</button>`;
    header.insertAdjacentElement('afterend', bar);
    if (!cfg.csrfToken) {
      bar.querySelector('#vkp-validate-review').disabled = true;
      bar.querySelector('#vkp-apply-review').disabled = true;
      setState('静态页面仅保存草稿；请通过 VKP Review Server 正式写回', 'draft');
    }
    if (cfg.approvedStickerOnly) {
      const rootButton = document.getElementById('sticker-root-btn');
      const rootInput = document.getElementById('sticker-root-input');
      const pickButton = document.getElementById('sticker-root-pick');
      const confirmButton = document.getElementById('sticker-root-confirm');
      if (rootButton) {
        rootButton.disabled = true;
        rootButton.title = 'VKP 仅允许 Bundle 内 stickers/ 目录中的已审核静态资源';
      }
      if (rootInput) {
        rootInput.disabled = true;
        rootInput.placeholder = '仅允许 Bundle/stickers/';
      }
      if (pickButton) pickButton.disabled = true;
      if (confirmButton) confirmButton.disabled = true;
    }

    const source = document.getElementById('cue-panel-text');
    if (source) {
      const sourceLabel = document.createElement('label');
      sourceLabel.className = 'vkp-track-label';
      sourceLabel.textContent = `粤语原文（${cfg.projection.tracks.source.language}）`;
      source.insertAdjacentElement('beforebegin', sourceLabel);
      const wrap = document.createElement('div');
      wrap.className = 'vkp-mandarin-wrap';
      wrap.innerHTML = `
        <label class="vkp-track-label" for="vkp-mandarin-text">普通话翻译（zh-CN）</label>
        <textarea id="vkp-mandarin-text" rows="3" placeholder="翻译轨缺失时保持为空，不会自动伪造"></textarea>
        <div class="vkp-translation-review" id="vkp-translation-review" hidden></div>`;
      source.parentElement.appendChild(wrap);
      const input = wrap.querySelector('textarea');
      input.addEventListener('input', () => {
        const segment = currentSegment();
        if (!segment) return;
        segment.mandarin_text = input.value;
        segment.mandarin_loaded = true;
        segment.translation_available = Boolean(input.value.trim());
        segment.needs_translation_review = false;
        segment._dirty = true;
        persistDraft();
        refreshMandarinPreviews();
      });
      const noteWrap = document.createElement('div');
      noteWrap.className = 'vkp-timestamp-note-wrap';
      noteWrap.innerHTML = `
        <label class="vkp-track-label">时间戳笔记（派生内容，不覆盖原话）</label>
        <div class="vkp-note-original" id="vkp-note-original"></div>
        <textarea id="vkp-note-polished" rows="2" placeholder="可选：润色后的摘录"></textarea>
        <textarea id="vkp-note-text" rows="3" placeholder="写下与当前时间点绑定的笔记"></textarea>
        <button type="button" id="vkp-note-delete">删除本段笔记</button>`;
      source.parentElement.appendChild(noteWrap);
      noteWrap.querySelector('#vkp-note-polished').addEventListener('input', updateTimestampNote);
      noteWrap.querySelector('#vkp-note-text').addEventListener('input', updateTimestampNote);
      noteWrap.querySelector('#vkp-note-delete').addEventListener('click', () => {
        const segment = currentSegment();
        if (segment) timestampNotes.delete(`note:${segment.segment_id}`);
        persistDraft();
        syncTimestampNoteEditor();
      });
    }
    document.querySelectorAll('[data-vkp-transcript-mode]').forEach((button) => {
      button.addEventListener('click', () => setTranscriptMode(button.dataset.vkpTranscriptMode));
    });
  }

  async function postReview(path, humanConfirmed) {
    persistDraft();
    const response = await fetch(path, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-VKP-Review-Token': String(cfg.csrfToken || ''),
      },
      body: JSON.stringify({
        bundle_revision: cfg.projection.bundle_revision,
        subtitle_review_notes: reviewNotes(humanConfirmed),
      }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    return payload;
  }

  async function validateDraft() {
    try {
      const result = await postReview(cfg.validateUrl, true);
      setState(`草稿校验通过 · ${result.summary?.reviewed_segments || DATA.segments.length} 段`, 'draft');
    } catch (error) {
      setState(`校验失败：${error.message || error}`, 'error');
    }
  }

  async function applyReview() {
    if (!confirm('正式写回 VKP？原始 ASR 和 Timeline 不会修改，下游总结将标记为待刷新。')) return;
    try {
      const result = await postReview(cfg.applyUrl, true);
      formallyApplied = true;
      localStorage.removeItem(draftKey);
      lastDraftJson = JSON.stringify(reviewNotes(false));
      setState(`已写回 VKP · ${result.summary?.source_text_changes || 0} 处原文修改`, 'applied');
    } catch (error) {
      setState(`写回失败：${error.message || error}`, 'error');
    }
  }

  const upstreamSetCurrentCuePanelIndex = setCurrentCuePanelIndex;
  setCurrentCuePanelIndex = function vkpSetCurrentCuePanelIndex(index) {
    upstreamSetCurrentCuePanelIndex(index);
    syncMandarinEditor();
    syncTimestampNoteEditor();
  };

  const upstreamRenderAll = renderAll;
  renderAll = function vkpRenderAll() {
    upstreamRenderAll();
    syncMandarinEditor();
    syncTimestampNoteEditor();
    refreshMandarinPreviews();
    installTranslationObserver();
  };

  const upstreamBuildJson = buildJson;
  buildJson = function vkpBuildJson() {
    const payload = JSON.parse(upstreamBuildJson());
    payload.vkp_projection = {
      schema: cfg.projection.schema,
      projection_sha256: cfg.projection.projection_sha256,
      source_sha256: cfg.projection.source_sha256,
    };
    payload.segments.forEach((row, index) => {
      const source = DATA.segments[index] || {};
      row.segment_id = source.segment_id;
      row.source_segment_ids = source.source_segment_ids || [];
      row.source_lineage_ids = source.source_lineage_ids || row.source_segment_ids;
      row.mandarin_text = source.mandarin_text || '';
      row.speaker_global_id = source.speaker_global_id || source.speaker || '';
      row.speaker_role = source.speaker_role || '';
      row.evidence_ids = source.evidence_ids || [];
      if (source.needs_translation_review) row.needs_translation_review = true;
    });
    return JSON.stringify(payload, null, 2);
  };

  const upstreamSplitAtCursor = splitAtCursor;
  splitAtCursor = function vkpSplitAtCursor() {
    if (!editingState) return;
    const index = editingState.idx;
    const before = DATA.segments[index];
    const sourceIds = [...(before.source_segment_ids || [before.segment_id])];
    const lineageIds = [...(before.source_lineage_ids || sourceIds)];
    const translation = String(before.mandarin_text || '');
    if (before.mandarin_loaded === false) {
      queueTranslation(before.segment_id);
      flashHint('请等待当前段普通话翻译加载后再拆分');
      return;
    }
    const translationInput = document.getElementById('vkp-mandarin-text');
    const translationCursor = translationInput?.selectionStart ?? 0;
    if (translation && (translationCursor <= 0 || translationCursor >= translation.length)) {
      before.needs_translation_review = true;
    }
    upstreamSplitAtCursor();
    if (DATA.segments.length < index + 2) return;
    const left = DATA.segments[index];
    const right = DATA.segments[index + 1];
    left.segment_id = `${before.segment_id || sourceIds[0]}:a`;
    right.segment_id = `${before.segment_id || sourceIds[0]}:b`;
    left.source_segment_ids = sourceIds;
    right.source_segment_ids = sourceIds;
    left.source_lineage_ids = lineageIds;
    right.source_lineage_ids = lineageIds;
    left.speaker = right.speaker = before.speaker || null;
    left.speaker_global_id = right.speaker_global_id = before.speaker_global_id || before.speaker || '';
    left.speaker_role = right.speaker_role = before.speaker_role || '';
    left.evidence_ids = right.evidence_ids = [...(before.evidence_ids || [])];
    left.timing_status = right.timing_status = 'human_adjusted';
    if (translation && translationCursor > 0 && translationCursor < translation.length) {
      left.mandarin_text = translation.slice(0, translationCursor).trim();
      right.mandarin_text = translation.slice(translationCursor).trim();
      left.needs_translation_review = right.needs_translation_review = false;
    } else {
      left.mandarin_text = translation;
      right.mandarin_text = '';
      left.needs_translation_review = right.needs_translation_review = Boolean(translation);
    }
    persistDraft();
    renderAll();
  };

  const upstreamMergeSegments = mergeSegments;
  mergeSegments = function vkpMergeSegments(indexes) {
    const sorted = [...new Set(indexes)].sort((a, b) => a - b);
    const segments = sorted.map((index) => DATA.segments[index]).filter(Boolean);
    if (segments.some((segment) => segment.mandarin_loaded === false)) {
      segments.forEach((segment) => queueTranslation(segment.segment_id));
      flashHint('请等待所选段普通话翻译加载后再合并');
      return;
    }
    const speakers = new Set(segments.map((segment) => segment.speaker_global_id || segment.speaker || '').filter(Boolean));
    if (speakers.size > 1) {
      flashHint('不同全局说话人的字幕禁止自动合并');
      return;
    }
    const sourceIds = [];
    const lineageIds = [];
    const seenLineage = new Set();
    segments.forEach((segment) => {
      const segmentSources = segment.source_segment_ids || [segment.segment_id];
      const segmentLineages = segment.source_lineage_ids || segmentSources;
      segmentLineages.forEach((lineageId, sourceIndex) => {
        const lineage = String(lineageId);
        if (seenLineage.has(lineage)) return;
        seenLineage.add(lineage);
        lineageIds.push(lineage);
        sourceIds.push(String(segmentSources[sourceIndex] || segmentSources[0] || lineage));
      });
    });
    const mandarinText = segments.map((segment) => segment.mandarin_text || '').filter(Boolean).join(EDITOR_SETTINGS.mergeJoinText || '');
    const evidenceIds = [...new Set(segments.flatMap((segment) => segment.evidence_ids || []))];
    upstreamMergeSegments(sorted);
    const merged = DATA.segments[sorted[0]];
    if (!merged) return;
    merged.segment_id = `human-merge:${lineageIds.join('+')}`;
    merged.source_segment_ids = sourceIds;
    merged.source_lineage_ids = lineageIds;
    merged.mandarin_text = mandarinText;
    merged.speaker_global_id = segments[0]?.speaker_global_id || segments[0]?.speaker || '';
    merged.speaker_role = segments[0]?.speaker_role || '';
    merged.evidence_ids = evidenceIds;
    merged.timing_status = 'human_adjusted';
    merged.needs_translation_review = segments.some((segment) => segment.needs_translation_review);
    persistDraft();
    renderAll();
  };

  installChrome();
  restoreDraft();
  setTranscriptMode('bilingual');
  renderAll();
  document.getElementById('vkp-validate-review')?.addEventListener('click', validateDraft);
  document.getElementById('vkp-apply-review')?.addEventListener('click', applyReview);
  document.addEventListener('input', () => window.setTimeout(persistDraft, 0));
  document.addEventListener('change', () => window.setTimeout(persistDraft, 0));
  window.setInterval(() => {
    if (!formallyApplied) persistDraft();
    syncMandarinEditor();
    refreshMandarinPreviews();
  }, 1500);
})();
