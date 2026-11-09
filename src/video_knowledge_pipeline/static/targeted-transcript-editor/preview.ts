// VKP bridge to Hyperaudio's kept-section and playback skipping model.
// Source: hyperaudio-lite-editor/js/editor-audio-cut.js, b8bd4728f6d6914a37350d119cf52d4c29eb9968.
import { computePlayableSections } from './vendor/hyperaudio/playable-sections.js';
import { groupedReview, originRuns, type Chapter, type RevisionMeta } from './model';
import type { Editor } from '@tiptap/core';
import { layoutEdits, subtitleCues } from './subtitle-layout';
import { residualPauseWarnings, type PauseCandidate } from './pacing-review';

export type Section = { start: number; end: number };
export type SpeechBoundaryPolicy = {name?:string;word_tail_guard_s?:number;between_sentences_compress_from_s:number;within_sentence_compress_from_s?:number};
export type PreviewPlan = { sections: Section[]; cuts: Section[]; removedSeconds: number; editedDuration: number; warnings: string[]; subtitles: Record<string, string>; pacing_audition_ids?: string[]; displaySubtitles?: { id: string; start: number; end: number; text: string }[]; basis?: 'retained_subtitle_intervals.v1'; boundary_policy?: 'trim_deleted_edge_words.v1'; speech_boundary_guard?:{policy:string;tail_s:number;short_gap_threshold_s:number;tails:Section[];short_gaps:Section[]} };
export function timedTranscript(words: { start: number; end: number; text?: string; struck: boolean }[]) {
  // Hyperaudio consumes timed DOM spans. A small read-only facade adapts the existing ASR model.
  return { querySelectorAll() { return words.map(w => ({ getAttribute(name: string) { return String(Math.round((name === 'data-m' ? w.start : w.end - w.start) * 1000)); }, style: { textDecoration: w.struck ? 'line-through' : '' }, textContent: w.text || '' })); } };
}
function subtractCuts(cuts: Section[], duration: number): Section[] {
  // Hyperaudio computePlayableSections second pass, adapted only to expose manual time ranges too.
  cuts.sort((a, b) => a.start - b.start); const sections: Section[] = []; let cursor = 0;
  cuts.forEach(cut => { const start = Math.max(cut.start, cursor); if (start >= cut.end) { cursor = Math.max(cursor, cut.end); return; } if (start > cursor) sections.push({ start: cursor, end: start }); cursor = cut.end; });
  if (cursor < duration) sections.push({ start: cursor, end: duration }); return sections;
}
export function complement(sections: Section[], duration: number): Section[] {
  const cuts: Section[] = []; let cursor = 0;
  for (const s of sections) { if (s.start > cursor) cuts.push({ start: cursor, end: s.start }); cursor = Math.max(cursor, s.end); }
  if (cursor < duration) cuts.push({ start: cursor, end: duration }); return cuts;
}
export function buildPreviewPlan(editor: Editor, chapter: Chapter, meta: Record<string, RevisionMeta>, duration: number, manual: any[] = [], pauses: PauseCandidate[] = [], boundary?:SpeechBoundaryPolicy): PreviewPlan {
  const groups = groupedReview(editor, chapter), warnings: string[] = [];
  const ranges = groups.filter(g => !['structure', 'subtitle_structure'].includes(g.kind) && meta[g.id]?.effect === 'cut_media').flatMap(g => g.anchors);
  const words = chapter.segments.flatMap(s => (s.timings || []).map(t => ({ ...t, segment_id: s.id, text: t.text, struck: ranges.some(a => a.segment_id === s.id && a.start_char < t.end_char && a.end_char > t.start_char) })));
  const coarse: Section[] = [];
  for (const anchor of ranges) {
    const s = chapter.segments.find(x => x.id === anchor.segment_id); if (!s) continue;
    if (s.timings?.some(t => t.start_char < anchor.end_char && t.end_char > anchor.start_char)) continue;
    if (!/[\p{L}\p{N}]/u.test(anchor.original_text)) continue; // punctuation alone is never an audio deletion
    if (anchor.start_char === 0 && anchor.end_char === s.text.length) coarse.push({ start: s.start, end: s.end });
    else warnings.push(`${s.id} 的局部删词缺少字词时间，暂保留原声`);
  }
  const knownDuration = Number.isFinite(duration) && duration > 0 ? duration : chapter.duration || 0;
  const native = knownDuration > 0 ? computePlayableSections(timedTranscript(words), { player: { duration: knownDuration }, removeGapsEnabled: false }).filter((s: Section) => s.end > s.start) : [];
  const manualCuts = manual.filter(r => Number.isFinite(r.start) && Number.isFinite(r.end) && r.start >= 0 && r.end > r.start && r.end <= knownDuration).map(r => ({ start: r.start, end: r.end }));
  const subtitles = Object.fromEntries(chapter.segments.map(s => [s.id, '']));
  const fallback = new Map(groups.map(g => [g.id, g.anchors[0]?.source_id]));
  const runs = originRuns(editor, chapter);
  editor.state.doc.descendants((node, pos) => {
    if (!node.isText || node.marks.some(m => m.type.name === 'deletion')) return;
    const change = node.marks.find(m => m.type.name === 'insertion')?.attrs.changeId;
    let id = node.marks.find(m => m.type.name === 'sourceOrigin')?.attrs.sourceId || fallback.get(change);
    if (!id) id = runs.find(r => r.to >= pos)?.id || runs.at(-1)?.id;
    const segment = chapter.segments.find(s => id === `${chapter.number}:${s.id}`);
    if (segment) subtitles[segment.id] += node.text || '';
  });
  // Positive coverage is the final retained/corrected subtitle cues, not the
  // full media. Preserve each cue's natural speech envelope: native character
  // pairs may mark only short parts of a syllable, so their gaps are not silence.
  // The same Hyperaudio subtraction pass computes the coverage complement,
  // including overlapping/nested cues; intersect it with the existing cuts.
  // A transcript-only correction may remove an entire punctuation-only cue.
  // Its audio must remain exactly as before; only explicit cut_media owns cuts.
  const transcriptOnlyAudio = new Set(groups.filter(g => meta[g.id]?.effect === 'transcript_only' && g.deleted).flatMap(g => g.anchors.map(a => a.segment_id)));
  const tails:Section[]=[],shortGaps:Section[]=[];
  const tailGuard=Number.isFinite(boundary?.word_tail_guard_s)?Math.max(0,boundary!.word_tail_guard_s!):0;
  const shortGap=tailGuard>0&&Number.isFinite(boundary?.between_sentences_compress_from_s)?boundary!.between_sentences_compress_from_s:0;
  const envelopes=chapter.segments.map((s,index)=>{
    if(!subtitles[s.id].trim()&&!transcriptOnlyAudio.has(s.id))return undefined;
    const spoken = words.filter(w => w.segment_id === s.id && w.end > w.start && /[\p{L}\p{N}]/u.test(w.text));
    const kept = spoken.filter(w => !w.struck);
    if (spoken.length && !kept.length) return undefined;
    // Removing an edge word removes its adjacent outer pause as well. Keep the
    // original envelope for unchanged/corrected edges and natural internal speech.
    const start = spoken[0]?.struck ? kept[0].start : s.start;
    let end = spoken.at(-1)?.struck ? kept.at(-1)!.end : s.end;
    const last=kept.at(-1);
    if(last&&tailGuard>0){
      // Native ASR ends are alignment anchors, not a proved phonetic release.
      // Reuse the active MVP tail guard, capped before the next spoken token.
      // Explicit native, coarse and manual deletions are still subtracted below.
      const next=words.filter(w=>w!==last&&w.end>w.start&&/[\p{L}\p{N}]/u.test(w.text)&&w.start>=last.end-1e-7).reduce((n,w)=>Math.min(n,w.start),knownDuration);
      const guarded=Math.min(knownDuration,next,Math.max(end,last.end+tailGuard));
      if(guarded>end){tails.push({start:end,end:guarded});end=guarded;}
    }
    return {index,start:Math.max(0,start),end:Math.min(knownDuration,end),spoken,kept};
  }).filter((s):s is NonNullable<typeof s>=>!!s&&s.end>s.start);
  const coverage:Section[]=envelopes.map(({start,end})=>({start,end}));
  for(const [index,left] of envelopes.entries()){
    const right=envelopes[index+1];if(!right||right.index!==left.index+1)continue;
    // Only adjacent intact speech edges own this short natural gap. Never
    // bridge a deleted cue, a removed suffix/prefix, or unrelated pre-roll.
    if(!left.kept.length||!right.kept.length||left.kept.at(-1)!==left.spoken.at(-1)||right.kept[0]!==right.spoken[0])continue;
    const a=left.kept.at(-1)!.end,b=right.kept[0].start;
    if(shortGap>0&&b>a&&b-a<shortGap-1e-7){shortGaps.push({start:a,end:b});coverage.push({start:a,end:b});}
  }
  const activePauses = pauses.filter(p => p.chapter === chapter.number && p.source_hash === chapter.source_text_hash && [p.left,p.right].every(a=>{const matches=words.filter(w=>w.segment_id===a.segment_id&&w.start_char<a.end_char&&w.end_char>a.start_char);return matches.length>0&&matches.every(w=>!w.struck);}));
  // Preserve the policy's two handles even when the old subtitle envelopes
  // omitted part of a native word tail. Explicit lexical/manual cuts still win.
  for (const p of activePauses) coverage.push({start:p.protect_start,end:p.start},{start:p.end,end:p.protect_end});
  const outsideSubtitles = subtractCuts(coverage, knownDuration);
  const sections = subtractCuts([...outsideSubtitles, ...complement(native, knownDuration), ...coarse, ...manualCuts, ...activePauses.map(p=>({start:p.start,end:p.end}))], knownDuration);
  const cuts = complement(sections, knownDuration);
  warnings.push(...residualPauseWarnings(activePauses,sections,boundary));
  const displaySubtitles = layoutEdits(editor).length ? subtitleCues(editor, chapter).map(({ id, start, end, text }) => ({ id, start, end, text })) : chapter.segments.map(s => ({ id: s.id, start: s.start, end: s.end, text: subtitles[s.id] }));
  return { basis: 'retained_subtitle_intervals.v1', boundary_policy: 'trim_deleted_edge_words.v1', sections, cuts, removedSeconds: cuts.reduce((n, s) => n + s.end - s.start, 0), editedDuration: sections.reduce((n, s) => n + s.end - s.start, 0), warnings, subtitles, displaySubtitles, pacing_audition_ids: activePauses.map(p=>p.id),...(tailGuard>0?{speech_boundary_guard:{policy:boundary?.name||'MVP word-tail guard',tail_s:tailGuard,short_gap_threshold_s:shortGap,tails,short_gaps:shortGaps}}:{}) };
}
export function editedTime(sections: Section[], sourceTime: number) { return sections.reduce((n, s) => n + Math.max(0, Math.min(sourceTime, s.end) - s.start), 0); }
export function sourceTime(sections: Section[], edited: number) { let left = Math.max(0, edited); for (const s of sections) { if (left < s.end - s.start) return s.start + left; left -= s.end - s.start; } return sections.at(-1)?.end || 0; }

export class CutPreviewController {
  enabled = true; plan: PreviewPlan = { sections: [], cuts: [], removedSeconds: 0, editedDuration: 0, warnings: [], subtitles: {} };
  private frame = 0; private lastSeek = -1; private completed=false;
  constructor(readonly player: HTMLVideoElement, readonly update: () => void, readonly onComplete?: () => boolean) {
    player.addEventListener('play', () => { this.check(); this.start(); });
    player.addEventListener('pause', () => { cancelAnimationFrame(this.frame); this.frame = 0; this.update(); });
    player.addEventListener('seeked', () => { this.check(); this.update(); });
    player.addEventListener('timeupdate', () => { this.check(); this.update(); });
    player.addEventListener('ended', () => { this.check(); this.update(); });
  }
  setPlan(plan: PreviewPlan) { this.plan = plan; this.lastSeek = -1; this.completed=false; this.check(); this.update(); }
  setEnabled(enabled: boolean) { this.enabled = enabled; this.lastSeek = -1; this.check(); this.update(); }
  private start() { cancelAnimationFrame(this.frame); const tick = () => { this.check(); this.update(); if (!this.player.paused) this.frame = requestAnimationFrame(tick); else this.frame = 0; }; this.frame = requestAnimationFrame(tick); }
  check() {
    // Adapted from Hyperaudio checkStrikeThrus/ensureSkipListeners: RAF plus seeked/timeupdate.
    // Land exactly on the next kept start, preserving its first 50ms as well.
    if (!this.enabled || this.player.readyState < 1 || !Number.isFinite(this.player.duration)) return;
    const sections = this.plan.sections, t = this.player.currentTime;
    if(sections.length&&t<sections.at(-1)!.end)this.completed=false;
    if (!sections.length) { if(this.complete())return; this.player.pause(); if (t < this.player.duration) this.seek(this.player.duration); return; }
    if (t < sections[0].start) { this.seek(sections[0].start); return; }
    for (let i = 0; i < sections.length - 1; i++) { if (t >= sections[i].end && t < sections[i + 1].start) { this.seek(sections[i + 1].start); return; } }
    const last = sections.at(-1)!;
    if (t >= last.end) { if(this.complete())return; if(last.end<this.player.duration){this.player.pause();this.seek(this.player.duration);} }
  }
  private complete(){if(this.completed)return false;this.completed=true;return this.onComplete?.()===true;}
  private seek(to: number) { if (Math.abs(this.player.currentTime - to) > 0.0005 && (this.lastSeek !== to || !this.player.seeking)) { this.lastSeek = to; this.player.currentTime = to; } }
}
