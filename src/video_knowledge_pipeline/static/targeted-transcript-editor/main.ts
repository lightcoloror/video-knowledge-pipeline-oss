// Adapted from tiptap-track-changes demo/src/main.ts and demo/src/sidebar.ts.
// Project-specific code adds ASR source references, five-field explanations, persistence and local media playback.
import { Extension } from '@tiptap/core';
import { Plugin, PluginKey } from '@tiptap/pm/state';
import { Decoration, DecorationSet } from '@tiptap/pm/view';
import { createReviewEditor, initialDocument, groupedReview, ensureMetadata, reviewPayload, anchorsForRange, time, EFFECTS, trackedDelete, trackedReplace, getBaseText,
  type Chapter, type ChapterReview, type RevisionMeta, type Effect } from './model';
import './styles.css';
import { buildPreviewPlan, CutPreviewController, editedTime, sourceTime } from './preview';
import { applySubtitleAction, removeSubtitleAction, subtitleCues, subtitleSelection, validateSubtitleLayout, type LayoutEdit } from './subtitle-layout';
import { initialPacingReview, validatePacingReview, pacingAuditionCandidates, type PacingPacket, type PauseDecision } from './pacing-review';
import {prepareReviewExport,exportContextHint,type ReviewExportContext} from './export-naming';
import {buildWholeTimeline,locateWholeTime,wholePosition,wholeTimeLabel,type WholeTimeline} from './whole-preview';

declare global { interface Window { RWE_TRANSCRIPT_DATA: { chapters: Chapter[]; script_reference: string }; rweReview: any; VKP_TRANSCRIPT_CONTEXT?: any } }
const chapters = window.RWE_TRANSCRIPT_DATA.chapters;
const vkp = window.VKP_TRANSCRIPT_CONTEXT;
const exportContext:ReviewExportContext|undefined=vkp?.exportContext||window.RWE_TRANSCRIPT_EXPORT_CONTEXT;
const pacingPacket: PacingPacket | undefined = vkp?.pacingCandidates || undefined;
let pacingReview = initialPacingReview(pacingPacket);
let draftBaseRevision = vkp?.savedDraftRevision || '';
const storageKey = vkp?.storageKey || 'rwe-transcript-review-v3';
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const selector = $<HTMLSelectElement>('chapter');
const video = $<HTMLVideoElement>('player');
let book: Record<string, ChapterReview> = {};
let legacy: Record<string, string> = {}, legacyRanges: Record<string, any[]> = {};
let current = chapters[0], editor: ReturnType<typeof createReviewEditor>;
let renderTimer: ReturnType<typeof setTimeout> | undefined;
let clipEnd: number | null = null, mediaReady = false, pendingTime: number | null = null;
let activeRevision: string | null = null;
let restoreNotice = '';
let localDraftTime = 0;
let legacyDraft = false;
let serverTimer: ReturnType<typeof setTimeout> | undefined, writingServer = false, draftVersion = 0;
let wholeTimeline:WholeTimeline={clips:[],duration:0},wholeDirty=true,wholeMode=true,wholeEnded=false,wholeRequest=0;
let wholePending:{request:number;chapter:string;sourceTime:number;globalTime:number;autoplay:boolean}|null=null;
const preview = new CutPreviewController(video, updatePreviewUI, completeWholeChapter);
try { const saved = JSON.parse(localStorage.getItem(storageKey) || 'null'); if (saved?.schema === 'rwe.transcript-review.v3') { book = saved.chapters || {}; legacyRanges = saved.manual_time_ranges || {}; localDraftTime = Date.parse(saved.updated_at || '') || 0; legacyDraft = !Object.hasOwn(saved, 'base_revision_sha256'); if (legacyDraft && vkp?.savedDraft) localStorage.setItem(`${storageKey}:pre-version:${saved.updated_at || 'unknown'}`, JSON.stringify(saved)); else if (!legacyDraft) draftBaseRevision = saved.base_revision_sha256; } } catch { restoreNotice = '本机存档无法读取；可从已导出的复核 JSON 恢复。'; }
try { legacy = JSON.parse(localStorage.getItem('rwe-caption-review-v1') || '{}'); if (!Object.keys(legacyRanges).length) legacyRanges = JSON.parse(localStorage.getItem('rwe-caption-delete-ranges-v1') || '{}'); } catch {}
try { const saved = JSON.parse(localStorage.getItem(storageKey) || 'null'); if (saved?.pacing_review) { validatePacingReview(saved.pacing_review, pacingPacket); pacingReview = saved.pacing_review; } } catch { restoreNotice = '气口候选版本已变化；旧记录仍保留，请核对新版候选。'; }
function notice(text: string, error = false) { $('save-status').textContent = text; $('save-status').classList.toggle('error', error); }
function metadata() { return book[current.number].meta; }
function activeGroups() { return groupedReview(editor, current); }
function storeCurrent() {
  if (!editor || (editor.storage as any).rweCompositionAdapter?.active) return;
  book[current.number] = { source_hash: current.source_text_hash, doc: editor.getJSON(), meta: metadata() };
  try { localStorage.setItem(storageKey, JSON.stringify({ schema: 'rwe.transcript-review.v3', updated_at: new Date().toISOString(), base_revision_sha256: draftBaseRevision, chapters: book, manual_time_ranges: legacyRanges, pacing_review: pacingReview })); notice('已自动保存到本机'); }
  catch { notice('本机保存失败，请导出复核 JSON 保留修改。', true); }
  draftVersion++;
  if (vkp?.saveUrl && !writingServer) { clearTimeout(serverTimer); serverTimer = setTimeout(saveVkpDraft, 600); }
}
async function saveVkpDraft() {
  if (writingServer || (editor.storage as any).rweCompositionAdapter?.active) return;
  writingServer = true; const review = exportReview(), version = draftVersion;
  try {
    const response = await fetch(vkp.saveUrl, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-VKP-Review-Token': vkp.token }, body: JSON.stringify({ pack_payload_sha256: vkp.packHash, base_revision_sha256: draftBaseRevision, review }) });
    const result = await response.json(); if (!response.ok || !result.ok) throw Error(result.error || 'VKP 保存失败');
    draftBaseRevision = result.revision_sha256;
    const local = JSON.parse(localStorage.getItem(storageKey) || 'null');
    if (local?.schema === 'rwe.transcript-review.v3') { local.base_revision_sha256 = draftBaseRevision; localStorage.setItem(storageKey, JSON.stringify(local)); }
    notice('已保存到 VKP · 文稿修订草稿');
  } catch (error) { notice(`${(error as Error).message}；本机记录保留，可导出 JSON。`, true); }
  finally { writingServer = false; if (draftVersion > version) serverTimer = setTimeout(saveVkpDraft, 600); }
}

// Upstream grouping drives these inline links; no custom revision engine is introduced.
const RevisionLinks = Extension.create({
  name: 'revisionLinks',
  addProseMirrorPlugins() { return [new Plugin({ key: new PluginKey('rweRevisionLinks'), props: {
    decorations(state) {
      const groups = groupedReview({ state } as any, current);
      const links = groups.map((g, i) => Decoration.widget(Math.min(g.to, state.doc.content.size), () => {
        const button = document.createElement('button'); button.type = 'button'; button.className = 'revision-link'; button.contentEditable = 'false';
        button.textContent = `修订 ${i + 1} ↗`; button.setAttribute('aria-label', `查看修订 ${i + 1} 的意图、决策、理由、证据与生效范围`);
        button.onmousedown = e => e.preventDefault(); button.onclick = () => focusRevision(g.id);
        return button;
      // ProseMirror reuses widget DOM for equal keys. Include the visible ordinal so
      // inserting/removing an earlier revision refreshes its label and aria-label.
      // g.id still owns the review data and click target; renumbering never changes that identity.
      }, { key: `rwe-link:${g.id}:${i + 1}`, side: 1, stopEvent: () => true }));
      const starts = subtitleCues({ state } as any, current).map((c, i) => Decoration.widget(c.from, () => {
        const marker = document.createElement('button'); marker.type = 'button'; marker.className = 'subtitle-start'; marker.contentEditable = 'false';
        marker.textContent = '│'; marker.title = `字幕 ${i + 1}：${c.text}`; marker.setAttribute('aria-label', `定位字幕 ${i + 1}：${c.text}`);
        marker.onmousedown = event => event.preventDefault(); marker.onclick = () => { editor.commands.setTextSelection(c.from); editor.view.focus(); updateSelection(); };
        return marker;
      }, { key: `subtitle-start:${c.id}:${i}:${c.from}:${c.text}`, side: -1, stopEvent: () => true }));
      return DecorationSet.create(state.doc, [...links, ...starts]);
    },
  } })]; },
});

function loadChapter(number: string, save = true) {
  if (editor) { if (save) storeCurrent(); editor.destroy(); }
  current = chapters.find(c => c.number === number) || chapters[0]; selector.value = current.number;
  const saved = book[current.number];
  if (saved && saved.source_hash !== current.source_text_hash) {
    try { localStorage.setItem(`${storageKey}:archived:${current.number}:${saved.source_hash}`, JSON.stringify(saved)); restoreNotice = '源识别文本版本已变化；旧段落存档已另存到本机，本段将使用当前原文。'; }
    catch { restoreNotice = '原始文本版本不匹配，旧存档备份失败；请保留之前导出的复核 JSON。'; }
  }
  const usable = saved?.source_hash === current.source_text_hash;
  book[current.number] = usable ? saved : { source_hash: current.source_text_hash, doc: initialDocument(current, legacy), meta: {} };
  editor = createReviewEditor({ element: $('editor'), content: book[current.number].doc, extraExtensions: [RevisionLinks],
    onUpdate() { clearTimeout(renderTimer); renderTimer = setTimeout(() => { if (!editor.isDestroyed) sync(); }, 150); },
    onSelectionUpdate() { updateSelection(); },
  });
  activeRevision = null; clipEnd = null; mediaReady = false; pendingTime = null;
  video.src = vkp?.mediaUrl ? `${vkp.mediaUrl}?source_id=${encodeURIComponent(current.source_id!)}` : `../sources/talking-head/${current.file}`; video.load();
  $('clip-file').textContent = `第 ${current.number} 段 · ${current.file}`;
  $('script-reference').textContent = current.script;
  $('document-title').textContent = `第 ${current.number} 段 · 口播逐字稿`;
  sync(false);
  if (restoreNotice) { notice(restoreNotice, true); restoreNotice = ''; }
}
function updateSelection() {
  if (!editor) return;
  const { from, to } = editor.state.selection;
  const selected = editor.state.doc.textBetween(from, to, ' ');
  $('selection-status').textContent = from === to ? '拖选任意文字；Delete / Backspace 删除，直接输入即可修正' : `已选 ${selected.length} 字：${selected.slice(0, 42)}${selected.length > 42 ? '…' : ''}`;
  ($('delete-selection') as HTMLButtonElement).disabled = from === to;
  ($('replace-selection') as HTMLButtonElement).disabled = from === to;
  const selection = subtitleSelection(editor, current);
  ($('subtitle-merge-previous') as HTMLButtonElement).disabled = !selection.previous;
  ($('subtitle-merge-next') as HTMLButtonElement).disabled = !selection.next;
  ($('subtitle-split') as HTMLButtonElement).disabled = !selection.split;
  $('subtitle-selection').textContent = selection.cue ? `当前字幕 ${selection.index + 1}/${selection.cues.length} · ${time(selection.cue.start)}–${time(selection.cue.end)}：${selection.cue.text}` : '当前没有保留的字幕。';
}
function sync(save = true) {
  if ((editor.storage as any).rweCompositionAdapter?.active) return;
  const groups = ensureMetadata(editor, current, metadata());
  refreshPreview();
  $('revision-count').textContent = `${groups.length} 处修订 · ${groups.filter(g => metadata()[g.id].status === 'confirmed').length} 处已确认`;
  renderCards(); renderPacingCards(); updateSelection(); if (save) storeCurrent();
}
function playAnchors(anchors: ReturnType<typeof anchorsForRange>) {
  if (!anchors.length) { notice('这处文稿结构修改没有可回听的原片区间。'); return; }
  cancelWhole();
  const start = Math.min(...anchors.map(a => a.envelope_start_s)), end = Math.max(...anchors.map(a => a.envelope_end_s));
  clipEnd = end;
  preview.setEnabled(false); $<HTMLInputElement>('preview-enabled').checked = false;
  $('media-status').textContent = `原声回听 ${time(start)}–${time(end)}；已临时关闭字幕区间预览`;
  if (!mediaReady) pendingTime = start;
  else { video.currentTime = start; video.play().catch(() => notice('请点播放器的播放按钮。')); }
}
function refreshPreview() {
  if (!editor || (editor.storage as any).rweCompositionAdapter?.active) return;
  wholeDirty=true;
  preview.setPlan(buildPreviewPlan(editor, current, metadata(), mediaReady ? video.duration : current.duration || 0, legacyRanges[current.number] || [], pacingAuditionCandidates(pacingPacket, pacingReview, current.number),pacingPacket?.policy));
  updateWholeUI();
}
function updatePreviewUI() {
  if (!editor) return;
  const plan = preview.plan;
  const pauseWarnings=plan.warnings.filter(w=>w.startsWith('气口 '));
  const timingWarnings=plan.warnings.length-pauseWarnings.length;
  $('preview-summary').textContent = `保留 ${time(plan.editedDuration)} / ${plan.sections.length} 段 · 跳过 ${plan.removedSeconds.toFixed(2)} 秒${timingWarnings ? ` · ${timingWarnings} 处字词时间不足，按字幕短语范围保留` : ''}${pauseWarnings.length ? ` · ${pauseWarnings.length} 处气口仍偏长，需回听复核（悬停查看原片区间）` : ''}`;
  $('preview-summary').title=plan.warnings.join('\n');
  $('edited-playhead').textContent = `${time(editedTime(plan.sections, video.currentTime))} / ${time(plan.editedDuration)}`;
  const slider = $<HTMLInputElement>('edited-timeline'); slider.max = String(plan.editedDuration); slider.value = String(editedTime(plan.sections, video.currentTime));
  const subtitle = plan.displaySubtitles?.find(s => s.start <= video.currentTime && video.currentTime < s.end);
  $('subtitle-overlay').textContent = subtitle?.text || '';
  if(preview.enabled) $('media-status').textContent = pacingReview?.enabled ? '气口候选试听 · 待逐处回听确认' : '仅播放保留／修正字幕对应区间';
  updateWholeUI();
}
function wholePlan(){
  if(wholeDirty&&editor){wholeTimeline=buildWholeTimeline(chapterPayloads());wholeDirty=false;}
  return wholeTimeline;
}
function updateWholeUI(){
  if(!editor)return;
  const plan=wholePlan(),position=wholeEnded?plan.duration:wholePending?.globalTime??wholePosition(plan,current.number,video.currentTime);
  $('whole-preview-summary').textContent=`${plan.clips.length} 段素材 · 整条 ${wholeTimeLabel(plan.duration)} · ${pacingReview?.enabled?'含气口压缩候选':'审核删留版'}`;
  const slider=$<HTMLInputElement>('whole-edited-timeline');slider.max=String(plan.duration);slider.value=String(position);slider.disabled=!plan.clips.length;
  $('whole-edited-playhead').textContent=`${wholeTimeLabel(position)} / ${wholeTimeLabel(plan.duration)}`;
  $('whole-preview-toggle').textContent=wholeMode&&!video.paused?'暂停全片':wholeEnded?'重新回听全片':'继续全片';
  $('whole-preview-status').textContent=wholePending?`正在接续第 ${wholePending.chapter} 段…`:wholeEnded?'整条净剪已播放结束':wholeMode?`${video.paused?'全片已暂停':'全片连续回听中'} · 当前第 ${current.number} 段，结束后自动接下一段`:'当前为本段复核；点“从头连续回听全片”听整条';
}
function cancelWhole(){wholeMode=false;wholeEnded=false;wholePending=null;wholeRequest++;updateWholeUI();}
function finishWholeSeek(){
  const pending=wholePending;
  if(!pending||!mediaReady||pending.request!==wholeRequest||pending.chapter!==current.number)return;
  video.currentTime=pending.sourceTime;wholePending=null;
  if(pending.autoplay)video.play().catch(()=>{notice('全片已定位，请点“继续全片”。');updateWholeUI();});
  updateWholeUI();
}
function seekWhole(value:number,autoplay:boolean){
  const plan=wholePlan(),target=locateWholeTime(plan,value);
  if(!target){video.pause();notice('没有保留的口播区间可供回听。');return;}
  if((editor.storage as any).rweCompositionAdapter?.active){notice('请先完成当前输入法选字，再开始全片回听。');return;}
  wholeMode=true;wholeEnded=false;clipEnd=null;
  wholePending={request:++wholeRequest,...target,globalTime:Math.max(0,Math.min(plan.duration,value)),autoplay};
  video.pause();
  $<HTMLInputElement>('preview-enabled').checked=true;
  if(current.number!==target.chapter)loadChapter(target.chapter);
  preview.setEnabled(true);finishWholeSeek();updateWholeUI();
}
function completeWholeChapter(){
  if(!wholeMode||wholePending||clipEnd!==null)return false;
  const plan=wholePlan(),index=plan.clips.findIndex(c=>c.chapter===current.number),next=plan.clips[index+1];
  if(next){seekWhole(next.start,true);return true;}
  wholeEnded=true;video.pause();updateWholeUI();return false;
}
function focusRevision(id: string) {
  activeRevision = id;
  document.querySelectorAll('.revision-card').forEach(el => el.classList.toggle('active', (el as HTMLElement).dataset.id === id));
  document.getElementById(`card-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  const group = activeGroups().find(g => g.id === id);
  if (group) { editor.commands.setTextSelection({ from: group.from, to: group.to }); editor.view.focus(); }
}
function field(card: HTMLElement, label: string, value: string, change?: (s: string) => void) {
  const wrap = document.createElement('label'); wrap.className = 'meta-field';
  const title = document.createElement('span'); title.textContent = label; wrap.append(title);
  if (change) { const input = document.createElement('textarea'); input.rows = label === '证据' ? 3 : 2; input.value = value; input.setAttribute('aria-label', label); input.onchange = () => change(input.value); wrap.append(input); }
  else { const text = document.createElement('span'); text.className = 'scope'; text.textContent = value; wrap.append(text); }
  card.append(wrap);
}
function renderCards() {
  const panel = $('review-cards'); panel.replaceChildren(); const groups = activeGroups();
  if (!groups.length) { const empty = document.createElement('p'); empty.className = 'empty'; empty.textContent = '本段还没有修订。直接在左侧选中文字删除或输入修正内容。'; panel.append(empty); }
  groups.forEach((g, i) => {
    const meta = metadata()[g.id]; if (!meta) return;
    const card = document.createElement('article'); card.className = 'revision-card'; card.id = `card-${g.id}`; card.dataset.id = g.id;
    if (activeRevision === g.id) card.classList.add('active');
    const heading = document.createElement('button'); heading.type = 'button'; heading.className = 'card-title'; heading.textContent = `修订 ${i + 1} · ${g.layout ? '断句修正' : g.kind === 'replacement' ? '修正' : g.kind === 'deletion' ? '删除' : g.kind === 'insertion' ? '补充' : '段落结构'} · ${meta.status === 'confirmed' ? '已确认' : '待确认'}`;
    heading.onclick = () => focusRevision(g.id); card.append(heading);
    const sample = document.createElement('div'); sample.className = 'change-sample';
    if (g.deleted) { const old = document.createElement('del'); old.textContent = g.deleted; sample.append(old); }
    if (g.inserted) { const corrected = document.createElement('ins'); corrected.textContent = g.inserted; sample.append(corrected); }
    if (g.layout) sample.textContent = `调整前：${g.layout.before}\n调整后：${g.layout.after}`;
    card.append(sample);
    const update = (key: keyof RevisionMeta, value: any) => { (meta as any)[key] = value; meta.status = 'proposed'; delete meta.confirmed_at; heading.textContent = heading.textContent!.replace('已确认', '待确认'); confirm.disabled = false; confirm.textContent = '确认本处'; storeCurrent(); const gs = activeGroups(); $('revision-count').textContent = `${gs.length} 处修订 · ${gs.filter(g => metadata()[g.id]?.status === 'confirmed').length} 处已确认`; };
    field(card, '意图', meta.intent, v => update('intent', v));
    const decision = document.createElement('label'); decision.className = 'meta-field';
    const label = document.createElement('span'); label.textContent = '决策'; decision.append(label);
    const effect = document.createElement('select'); effect.setAttribute('aria-label', '决策');
    for (const [value, text] of Object.entries(EFFECTS)) { const o = document.createElement('option'); o.value = value; o.textContent = text; o.selected = meta.effect === value; effect.append(o); }
    if (g.layout) { effect.disabled = true; effect.title = '断句修正只改变字幕显示，保留原声。'; }
    effect.onchange = () => { meta.effect = effect.value as Effect; meta.decision = EFFECTS[meta.effect]; meta.status = 'proposed'; delete meta.confirmed_at; storeCurrent(); sync(false); };
    decision.append(effect); card.append(decision);
    field(card, '理由', meta.reason, v => update('reason', v));
    field(card, '证据', meta.evidence, v => update('evidence', v));
    field(card, '生效范围', meta.effective_scope);
    const buttons = document.createElement('div'); buttons.className = 'card-actions';
    const play = document.createElement('button'); play.textContent = '回听原声'; play.onclick = () => playAnchors(g.anchors);
    const confirm = document.createElement('button'); confirm.textContent = meta.status === 'confirmed' ? '已确认' : '确认本处'; confirm.disabled = meta.status === 'confirmed';
    confirm.onclick = () => { meta.status = 'confirmed'; meta.confirmed_at = new Date().toISOString(); storeCurrent(); sync(false); };
    const reject = document.createElement('button'); reject.textContent = g.layout ? '恢复原断句' : '恢复原文'; reject.onclick = () => { if (g.layout) removeSubtitleAction(editor, g.id); else editor.commands.rejectChange(g.id); sync(); };
    buttons.append(play, confirm, reject); card.append(buttons); panel.append(card);
  });
}

function renderPacingCards() {
  $('pacing-review-panel').hidden = !pacingPacket;
  if (!pacingPacket || !pacingReview) return;
  $<HTMLInputElement>('pacing-enabled').checked = pacingReview.enabled;
  const rows = pacingPacket.candidates.filter(c=>c.chapter===current.number), panel = $('pacing-cards'); panel.replaceChildren();
  const policy=pacingPacket.policy;
  $('pacing-summary').textContent = `${policy.name} · 本段 ${rows.length} 处候选，${pacingReview.enabled?`${preview.plan.pacing_audition_ids?.length||0}处参与试听`:'试听已关闭'} · 句间 ≥${policy.between_sentences_compress_from_s.toFixed(2)} 秒建议留 ${policy.between_sentences_preserve_s.toFixed(2)} 秒，句内 ≥${policy.within_sentence_compress_from_s.toFixed(2)} 秒建议留 ${policy.within_sentence_preserve_s.toFixed(2)} 秒。当前为候选对比试听，确认前不作为正式剪辑。`;
  for (const c of rows) {
    const get = (): PauseDecision => pacingReview!.decisions[c.id] || {action:'keep_pending_review'};
    const applicable=!pacingReview.enabled||get().action==='keep'||!!preview.plan.pacing_audition_ids?.includes(c.id);
    const card = document.createElement('details'); card.className = 'pacing-card'; card.dataset.pacingId = c.id;
    const summary = document.createElement('summary'); summary.textContent = `${time(c.pause_start)}–${time(c.pause_end)} · ${(c.pause_end-c.pause_start).toFixed(2)} → ${c.preserve_s.toFixed(2)}秒 · ${!applicable?'相邻原声已删，当前不应用':get().action==='compress'?'已确认':get().action==='keep'?'保留原长':'待回听'}`; card.append(summary);
    field(card,'意图',c.intent); field(card,'决策',get().action==='compress'?'已回听确认压缩':get().action==='keep'?'保留原气口':c.decision);
    field(card,'理由',c.reason); field(card,'证据',c.evidence); field(card,'生效范围',c.effective_scope);
    const buttons = document.createElement('div'); buttons.className='card-actions';
    const listen = document.createElement('button'); listen.textContent='回听原声'; listen.onclick=()=>playAnchors([{envelope_start_s:Math.max(0,c.protect_start-1.2),envelope_end_s:Math.min(current.duration||c.protect_end+1.2,c.protect_end+1.2)} as any]);
    const audition = document.createElement('button'); audition.textContent='试听缩短后'; audition.disabled=!applicable; audition.onclick=()=>{
      cancelWhole(); pacingReview!.enabled=true; refreshPreview(); $<HTMLInputElement>('pacing-enabled').checked=true;
      clipEnd=Math.min(current.duration||c.protect_end+1.2,c.protect_end+1.2); preview.setEnabled(true); $<HTMLInputElement>('preview-enabled').checked=true;
      const start=Math.max(0,c.protect_start-1.2); if(!mediaReady) pendingTime=start; else {video.currentTime=start;video.play().catch(()=>notice('请点播放器播放按钮。'));} storeCurrent();
    };
    const keep = document.createElement('button'); keep.textContent=get().action==='keep'?'恢复候选':'保留原长'; keep.onclick=()=>{pacingReview!.decisions[c.id]={action:get().action==='keep'?'keep_pending_review':'keep'};refreshPreview();renderPacingCards();storeCurrent();};
    buttons.append(listen,audition,keep);card.append(buttons);
    const confirm = document.createElement('button'); confirm.textContent='确认本处压缩';
    const ready=()=>{const d=get();confirm.disabled=!applicable||!(d.audio_checked&&d.context_checked&&d.semantic_role==='ordinary'&&d.evidence?.trim())||d.action==='compress';};
    const update=(patch:Partial<PauseDecision>)=>{pacingReview!.decisions[c.id]={...get(),...patch,action:'keep_pending_review',confirmed_at:undefined};ready();storeCurrent();};
    for(const [key,text] of [['audio_checked','已回听，词尾和起音完整'],['context_checked','已核对为普通空隙，无强调、情绪或有意义的呼吸']] as const){const label=document.createElement('label');label.className='pacing-check';const input=document.createElement('input');input.type='checkbox';input.checked=!!get()[key];input.onchange=()=>update({...{[key]:input.checked},...(key==='context_checked'?{semantic_role:input.checked?'ordinary':'unknown'}:{})});label.append(input,document.createTextNode(text));card.append(label);}
    field(card,'回听说明',get().evidence||'',v=>update({evidence:v}));
    confirm.onclick=()=>{const d=get();validatePacingReview({...pacingReview!,decisions:{...pacingReview!.decisions,[c.id]:{...d,action:'compress'}}},pacingPacket);pacingReview!.decisions[c.id]={...d,action:'compress',confirmed_at:new Date().toISOString()};refreshPreview();renderPacingCards();storeCurrent();};
    ready();card.append(confirm);panel.append(card);
  }
}

// Persistence/export follows the upstream demo's document snapshot pattern, with immutable source hashes added.
function chapterPayloads() {
  return chapters.map(ch => {
    const stored = book[ch.number];
    if (ch.number === current.number) return { ...reviewPayload(ch, editor, metadata()), preview: preview.plan };
    const temp = createReviewEditor({ element: document.createElement('div'), content: stored?.doc || initialDocument(ch, legacy) });
    const value = reviewPayload(ch, temp, stored?.meta || {}); const plan = buildPreviewPlan(temp, ch, stored?.meta || {}, ch.duration || 0, legacyRanges[ch.number] || [], pacingAuditionCandidates(pacingPacket, pacingReview, ch.number),pacingPacket?.policy); temp.destroy(); return { ...value, preview: plan };
  });
}
function exportReview() {
  storeCurrent();
  const payload=chapterPayloads();
  return { schema: 'rwe.transcript-review.v3', exported_at: new Date().toISOString(), preview_basis: 'retained_subtitle_intervals.v1', preview_boundary_policy: 'trim_deleted_edge_words.v1', script_reference: window.RWE_TRANSCRIPT_DATA.script_reference,
    text_index_unit: 'UTF-16 code units; start inclusive, end exclusive',
    chapters: payload, manual_time_ranges: legacyRanges, pacing_review: pacingReview,
    media_cut_candidates: payload.flatMap(ch => ch.revisions.filter(r => r.effect === 'cut_media').map(r => ({ chapter: ch.chapter, file: ch.file, change_id: r.change_id, confirmed: r.status === 'confirmed', source_anchors: r.source_anchors, timing_status: 'preview_from_native_asr_timing; human_cut_boundary_review_required' }))),
  };
}
function download(value: unknown, name: string) { const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' })); const a = document.createElement('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
function importReview(value: any) {
  if (value.schema !== 'rwe.transcript-review.v3' || !Array.isArray(value.chapters)) throw new Error('请选择本页导出的 v3 复核 JSON。');
  validatePacingReview(value.pacing_review, pacingPacket);
  const imported: Record<string, ChapterReview> = {};
  for (const item of value.chapters) {
    const ch = chapters.find(c => c.number === item.chapter);
    if (!ch || ch.source_text_hash !== item.source_hash) throw new Error(`第 ${item.chapter} 段的原始识别文本版本不匹配，已停止导入。`);
    const temp = createReviewEditor({ element: document.createElement('div'), content: item.doc });
    try { validateSubtitleLayout(temp, ch); } catch (error) { temp.destroy(); throw error; }
    const base = getBaseText(temp).replace(/\n/g, ''); temp.destroy();
    if (base !== ch.segments.map(s => s.text).join('')) throw new Error(`第 ${ch.number} 段的原始文字记录不完整，已停止导入。`);
    const meta: Record<string, RevisionMeta> = {};
    for (const r of item.revisions || []) { if (!r.change_id || !Object.hasOwn(EFFECTS, r.effect)) throw new Error('复核决策字段无效。'); meta[r.change_id] = { intent: String(r.intent || ''), decision: EFFECTS[r.effect as Effect], reason: String(r.reason || ''), evidence: String(r.evidence || ''), effective_scope: String(r.effective_scope || ''), effect: r.effect, status: r.status === 'confirmed' ? 'confirmed' : 'proposed', origin: r.origin || 'imported_review', confirmed_at: r.confirmed_at, content_fingerprint: r.content_fingerprint }; }
    imported[ch.number] = { source_hash: item.source_hash, doc: item.doc, meta };
  }
  book = imported; legacyRanges = value.manual_time_ranges || legacyRanges; if(value.pacing_review)pacingReview=value.pacing_review; loadChapter(current.number, false); storeCurrent(); notice('已导入复核记录。');
}

for (const ch of chapters) { const o = document.createElement('option'); o.value = ch.number; o.textContent = `第 ${ch.number} 段`; selector.append(o); }
selector.onchange = () => { cancelWhole(); if ((editor.storage as any).rweCompositionAdapter?.active) setTimeout(() => loadChapter(selector.value), 70); else loadChapter(selector.value); };
for (const id of ['delete-selection', 'undo', 'redo', 'play-selection', 'subtitle-merge-previous', 'subtitle-merge-next', 'subtitle-split']) $(id).onmousedown = e => e.preventDefault();
for (const [id, action] of Object.entries({ 'subtitle-merge-previous': 'merge_previous', 'subtitle-merge-next': 'merge_next', 'subtitle-split': 'split' })) $(id).onclick = () => {
  if ((editor.storage as any).rweCompositionAdapter?.active) { notice('请先完成或取消当前输入法选字，再调整字幕断句。'); return; }
  try { applySubtitleAction(editor, current, action as LayoutEdit['action']); editor.view.focus(); sync(); }
  catch (error) { notice((error as Error).message, true); }
};
$('delete-selection').onclick = () => { if (trackedDelete(editor)) { editor.view.focus(); sync(); } };
$('undo').onclick = () => { editor.commands.undo(); editor.view.focus(); sync(); };
$('redo').onclick = () => { editor.commands.redo(); editor.view.focus(); sync(); };
$('play-selection').onclick = () => { const { from, to } = editor.state.selection; playAnchors(anchorsForRange(editor, current, from, to || from)); };
$('replace-selection').onclick = () => { const input = $<HTMLInputElement>('replacement'); if (!input.value) { input.focus(); return; } if (trackedReplace(editor, input.value)) { input.value = ''; editor.view.focus(); sync(); } };
$('export-identity-hint').textContent=exportContextHint(exportContext,chapters.length);
$('export-review').onclick = () => { if ((editor.storage as any).rweCompositionAdapter?.active) { notice('请先完成或取消当前输入法选字，再导出。'); return; } try {const result=prepareReviewExport(exportReview(),exportContext);download(result.payload,result.filename);$('export-identity-hint').textContent=`已导出：${result.filename}`;($('export-review') as HTMLButtonElement).dataset.lastExportFilename=result.filename;notice('已按选题、视频和导出时间命名。');}catch(error){notice((error as Error).message,true);} };
$('import-review').onclick = () => $<HTMLInputElement>('import-file').click();
$<HTMLInputElement>('import-file').onchange = async e => { const file = (e.target as HTMLInputElement).files?.[0]; if (!file) return; try { importReview(JSON.parse(await file.text())); } catch (err) { notice((err as Error).message, true); } finally { (e.target as HTMLInputElement).value = ''; } };
video.addEventListener('loadedmetadata', () => { mediaReady = true; refreshPreview(); finishWholeSeek(); if (pendingTime !== null) { video.currentTime = pendingTime; pendingTime = null; video.play().catch(() => {}); } });
video.addEventListener('error',()=>{if(wholePending){wholePending=null;wholeMode=false;notice(`第 ${current.number} 段加载失败，全片回听已暂停，修订保留。`,true);updateWholeUI();}});
video.addEventListener('play',()=>{if(wholeMode&&wholeEnded&&!wholePending)seekWhole(0,true);});
video.addEventListener('timeupdate', () => { $('playhead').textContent = time(video.currentTime); if (clipEnd !== null && video.currentTime >= clipEnd) { video.pause(); clipEnd = null; } });
window.addEventListener('beforeunload', () => storeCurrent());
$<HTMLInputElement>('preview-enabled').onchange = e => {if(!(e.target as HTMLInputElement).checked)cancelWhole();preview.setEnabled((e.target as HTMLInputElement).checked);};
$<HTMLInputElement>('pacing-enabled').onchange = e => {if(!pacingReview)return;pacingReview.enabled=(e.target as HTMLInputElement).checked;refreshPreview();storeCurrent();};
$('preview-start').textContent='仅回听本段';
$('preview-start').onclick = () => { cancelWhole();clipEnd = null; $<HTMLInputElement>('preview-enabled').checked = true; preview.setEnabled(true); video.currentTime = preview.plan.sections[0]?.start || 0; video.play().catch(() => notice('请点播放器播放按钮。')); };
$<HTMLInputElement>('edited-timeline').oninput = e => { cancelWhole();video.currentTime = sourceTime(preview.plan.sections, Number((e.target as HTMLInputElement).value)); };
$('whole-preview-start').onclick=()=>{wholeDirty=true;seekWhole(0,true);};
$('whole-preview-toggle').onclick=()=>{if(wholeMode&&!video.paused){video.pause();updateWholeUI();}else seekWhole(wholeEnded?0:wholePosition(wholePlan(),current.number,video.currentTime),true);};
$<HTMLInputElement>('whole-edited-timeline').oninput=e=>seekWhole(Number((e.target as HTMLInputElement).value),!video.paused);
window.rweReview = { get editor() { return editor; }, get current() { return current; }, get preview() { return preview; }, get wholePreview(){return {timeline:wholePlan(),enabled:wholeMode,ended:wholeEnded,pending:wholePending};},exportReview, importReview, loadChapter };
new ResizeObserver(() => document.documentElement.style.setProperty('--review-controls-height', `${$('editor-controls').getBoundingClientRect().height + 12}px`)).observe($('editor-controls'));
loadChapter(chapters[0].number, false);
if (vkp?.savedDraft && (legacyDraft || !localStorage.getItem(storageKey) || (Date.parse(vkp.savedDraft.exported_at || '') || 0) > localDraftTime)) { try { draftBaseRevision = vkp.savedDraftRevision || ''; importReview(vkp.savedDraft); notice(legacyDraft ? '已恢复 VKP 最新草稿；旧版本机存档已另存保留。' : '已恢复较新的 VKP 文稿修订草稿。'); } catch (error) { notice((error as Error).message, true); } }
