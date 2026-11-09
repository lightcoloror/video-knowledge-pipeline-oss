import { afterEach, describe, expect, it, vi } from 'vitest';
import { createReviewEditor, initialDocument, trackedDelete, trackedReplace, ensureMetadata, groupedReview, type Chapter } from '../model';
import { buildPreviewPlan, CutPreviewController, editedTime, sourceTime, timedTranscript } from '../preview';
import { computePlayableSections } from '../vendor/hyperaudio/playable-sections.js';

// Synthetic native pairs deliberately leave pauses between characters.
const chapter: Chapter = { number: '01', file: 'synthetic.mp4', duration: 20, source_text_hash: 'fixture', script: '', segments: [
  { id: 's1', start: 2, end: 8, text: '正常错词，继续正常。', preselection: 'keep', preselection_reason: '', script_match: '', timings: [
    { text: '正常', start_char: 0, end_char: 2, start: 2, end: 2.8 },
    { text: '错', start_char: 2, end_char: 3, start: 3, end: 3.3 },
    { text: '词', start_char: 3, end_char: 4, start: 3.8, end: 4 },
    { text: '继续正常', start_char: 5, end_char: 9, start: 5, end: 8 },
  ] },
  { id: 's2', start: 10, end: 15, text: '另一句正常。', preselection: 'keep', preselection_reason: '', script_match: '', timings: [
    { text: '另一句正常', start_char: 0, end_char: 5, start: 10, end: 15 },
  ] },
] };
const editors: ReturnType<typeof createReviewEditor>[] = [];
const subtitleCoverage = [{start:2,end:8},{start:10,end:15}];
const outsideSubtitles = [{start:0,end:2},{start:8,end:10},{start:15,end:20}];
function make(ch = chapter, content = initialDocument(ch)) { const e = createReviewEditor({ element: document.createElement('div'), content }); editors.push(e); return e; }
function select(e: ReturnType<typeof make>, text: string) { let from = -1; e.state.doc.descendants((n,p) => { if (n.isText && n.text?.includes(text) && from < 0) from = p + n.text.indexOf(text); }); if (from < 0) throw Error(text); e.commands.setTextSelection({from, to: from + text.length}); }
function plan(e: ReturnType<typeof make>, ch = chapter, meta: any = {}) { ensureMetadata(e,ch,meta); return buildPreviewPlan(e,ch,meta,20); }
afterEach(() => { editors.splice(0).forEach(e => e.destroy()); vi.restoreAllMocks(); });

describe('native-timed transcript cut preview', () => {
  it('cuts only the selected wrong word, including its internal pause', () => {
    const e=make(); select(e,'错词'); trackedDelete(e); const p=plan(e);
    expect(p.sections).toEqual([{start:2,end:3},{start:4,end:8},{start:10,end:15}]); expect(p.editedDuration).toBe(10);
    expect(p.subtitles.s1).toBe('正常，继续正常。');
    expect(groupedReview(e,chapter)[0].anchors[0].native_ranges).toHaveLength(2);
  });
  it('corrects subtitles and preserves their original audio intervals', () => {
    const e=make(); select(e,'错词'); trackedReplace(e,'正确词'); const p=plan(e);
    expect(p.sections).toEqual(subtitleCoverage); expect(p.editedDuration).toBe(11);
    expect(p.subtitles.s1).toBe('正常正确词，继续正常。');
  });
  it('recomputes cuts when a decision is changed to transcript only', () => {
    const e=make(); select(e,'错词'); trackedDelete(e); const meta:any={}; expect(plan(e,chapter,meta).sections).toHaveLength(3);
    meta[groupedReview(e,chapter)[0].id].effect='transcript_only'; expect(plan(e,chapter,meta).sections).toEqual(subtitleCoverage);
  });
  it('undo and redo restore the preview together with the document', () => {
    const e=make(); select(e,'错词'); trackedDelete(e); const deleted=plan(e).sections; expect(deleted).toHaveLength(3);
    e.commands.undo(); expect(plan(e).sections).toEqual(subtitleCoverage); e.commands.redo(); expect(plan(e).sections).toEqual(deleted);
  });
  it('deleting punctuation never changes the subtitle audio intervals', () => { const e=make(); select(e,'，'); trackedDelete(e); expect(plan(e).sections).toEqual(subtitleCoverage); });
  it('keeps sound with a warning if a partial lexical deletion has no native timing', () => {
    const ch=structuredClone(chapter); ch.segments[0].timings=[]; const e=make(ch); select(e,'错词'); trackedDelete(e);
    const p=plan(e,ch); expect(p.sections).toEqual(subtitleCoverage); expect(p.warnings).toHaveLength(1);
  });
  it('uses a known phrase interval only for a whole untimed phrase deletion', () => {
    const ch=structuredClone(chapter); ch.segments[0].timings=[]; const e=make(ch); select(e,ch.segments[0].text); trackedDelete(e);
    expect(plan(e,ch).sections).toEqual([{start:10,end:15}]);
  });
  it('unions overlapping manual and native cuts without double counting', () => {
    const e=make(); select(e,'错词'); trackedDelete(e); const meta={}; ensureMetadata(e,chapter,meta);
    const p=buildPreviewPlan(e,chapter,meta,20,[{start:3.5,end:6},{start:2.9,end:3.2}]);
    expect(p.sections).toEqual([{start:2,end:2.9},{start:6,end:8},{start:10,end:15}]); expect(p.removedSeconds).toBeCloseTo(12.1);
  });
  it('maps the edited seek bar at a cut boundary to the next kept start', () => {
    const sections=[{start:0,end:3},{start:4,end:20}]; expect(editedTime(sections,3.6)).toBe(3);
    expect(sourceTime(sections,3)).toBe(4); expect(sourceTime(sections,19)).toBe(20);
  });
  it('does not turn a paragraph boundary revision into a paragraph audio deletion', () => {
    const content={type:'doc',content:chapter.segments.map(s=>({type:'paragraph',content:[{type:'text',text:s.text,marks:[{type:'sourceOrigin',attrs:{sourceId:'01:'+s.id}}]}]}))};
    const e=make(chapter,content); const boundary=e.state.doc.child(0).nodeSize-1;
    e.commands.setTextSelection({from:boundary,to:boundary+2}); trackedDelete(e);
    expect(plan(e).sections).toEqual(subtitleCoverage);
  });
  it('starts from retained subtitle coverage even without a single text deletion', () => {
    const p=plan(make());expect(p.basis).toBe('retained_subtitle_intervals.v1');
    expect(p.sections).toEqual(subtitleCoverage);expect(p.cuts).toEqual(outsideSubtitles);expect(p.editedDuration).toBe(11);
  });
  it('keeps natural sound between character pairs inside a retained subtitle', () => {
    const p=plan(make());expect(p.sections.some(s=>s.start<=2.8&&s.end>=3)).toBe(true);
    expect(p.sections.some(s=>s.start<=4&&s.end>=5)).toBe(true);
  });
  it('a whole corrected subtitle inherits its entire original audio interval', () => {
    const e=make();select(e,chapter.segments[0].text);trackedReplace(e,'这是一句修正后的字幕。');
    const p=plan(e);expect(p.sections).toEqual(subtitleCoverage);expect(p.subtitles.s1).toBe('这是一句修正后的字幕。');
  });
  it('removes an entirely deleted subtitle envelope and its surrounding empty time', () => {
    const e=make();select(e,chapter.segments[0].text);trackedDelete(e);
    expect(plan(e).sections).toEqual([{start:10,end:15}]);
  });
  it('an all-deleted transcript has no playable interval or full-source fallback', () => {
    const e=make();for(const s of chapter.segments){select(e,s.text);trackedDelete(e);}
    const p=plan(e);expect(p.sections).toEqual([]);expect(p.editedDuration).toBe(0);expect(p.cuts).toEqual([{start:0,end:20}]);
  });
  it('unions nested subtitle intervals without cutting a still-covered word', () => {
    const ch=structuredClone(chapter);Object.assign(ch.segments[1],{start:3,end:4});
    ch.segments[1].timings=[{text:'另一句正常',start_char:0,end_char:5,start:3,end:4}];
    expect(plan(make(ch),ch).sections).toEqual([{start:2,end:8}]);
  });
  it('trims a deleted prefix through its following gap to the first retained word', () => {
    const e=make();select(e,'正常');trackedDelete(e);const p=plan(e);
    expect(p.sections).toEqual([{start:3,end:8},{start:10,end:15}]);expect(p.boundary_policy).toBe('trim_deleted_edge_words.v1');
  });
  it('trims a deleted suffix and its preceding gap back to the last retained word', () => {
    const e=make();select(e,'继续正常。');trackedDelete(e);
    expect(plan(e).sections).toEqual([{start:2,end:4},{start:10,end:15}]);
  });
  it('removes long original outer margins when both text edges are deleted', () => {
    const ch=structuredClone(chapter);ch.segments=ch.segments.slice(0,1);ch.segments[0].start=0;ch.segments[0].end=20;
    const e=make(ch);select(e,'正常');trackedDelete(e);select(e,'继续正常。');trackedDelete(e);
    const p=plan(e,ch);expect(p.sections).toEqual([{start:3,end:4}]);expect(p.editedDuration).toBe(1);
  });
  it('a corrected prefix keeps its original sound and envelope', () => {
    const e=make();select(e,'正常');trackedReplace(e,'正确');expect(plan(e).sections).toEqual(subtitleCoverage);
  });
  it('a corrected suffix keeps its original sound and envelope', () => {
    const e=make();select(e,'继续正常。');trackedReplace(e,'继续表达。');expect(plan(e).sections).toEqual(subtitleCoverage);
  });
  it('undo and redo recompute the trimmed head boundary', () => {
    const e=make();select(e,'正常');trackedDelete(e);const trimmed=plan(e).sections;
    e.commands.undo();expect(plan(e).sections).toEqual(subtitleCoverage);e.commands.redo();expect(plan(e).sections).toEqual(trimmed);
  });
  it('deleting only final punctuation does not shorten a spoken edge', () => {
    const e=make();select(e,'。');trackedDelete(e);expect(plan(e).sections).toEqual(subtitleCoverage);
  });
});

describe('active MVP spoken-tail protection',()=>{
  const boundary={name:'natural_conversational_v3',word_tail_guard_s:.15,between_sentences_compress_from_s:.9};
  function guarded(e:ReturnType<typeof make>,ch=chapter,manual:any[]=[]){const meta={};ensureMetadata(e,ch,meta);return buildPreviewPlan(e,ch,meta,20,manual,[],boundary);}
  it('keeps the release after a native ASR sentence-end anchor',()=>{const p=guarded(make());expect(p.sections).toEqual([{start:2,end:8.15},{start:10,end:15.15}]);expect(p.speech_boundary_guard?.tail_s).toBe(.15);});
  it('keeps a short natural gap between two adjacent retained subtitles',()=>{const ch=structuredClone(chapter);ch.segments[1].start=8.24;ch.segments[1].timings![0].start=8.24;const p=guarded(make(ch),ch);expect(p.sections).toEqual([{start:2,end:15.15}]);expect(p.speech_boundary_guard?.short_gaps).toEqual([{start:8,end:8.24}]);});
  it('does not bring back a deleted subtitle to bridge a gap',()=>{const e=make();select(e,chapter.segments[0].text);trackedDelete(e);expect(guarded(e).sections).toEqual([{start:10,end:15.15}]);});
  it('protects the last retained word when a suffix was deleted',()=>{const e=make();select(e,'继续正常。');trackedDelete(e);expect(guarded(e).sections).toEqual([{start:2,end:4.15},{start:10,end:15.15}]);});
  it('caps a protected tail before the next explicitly deleted token',()=>{const ch=structuredClone(chapter);ch.segments[0].timings![3].start=4.08;const e=make(ch);select(e,'继续正常。');trackedDelete(e);expect(guarded(e,ch).sections[0]).toEqual({start:2,end:4.08});});
  it('keeps explicit manual cuts authoritative over the tail expansion',()=>{expect(guarded(make(),chapter,[{start:8.05,end:8.3}]).sections[0]).toEqual({start:2,end:8.05});});
  it('keeps corrected subtitle sound with the same protected edge',()=>{const e=make();select(e,'继续正常。');trackedReplace(e,'继续表达。');const p=guarded(e);expect(p.sections).toEqual([{start:2,end:8.15},{start:10,end:15.15}]);expect(p.subtitles.s1).toBe('正常错词，继续表达。');});
  it('cannot restore an entirely deleted program',()=>{const e=make();for(const s of chapter.segments){select(e,s.text);trackedDelete(e);}expect(guarded(e).sections).toEqual([]);});
});

describe('reused Hyperaudio shipped regression cases', () => {
  function sections(rows:number[][],duration=10,gaps=false) {return computePlayableSections(timedTranscript(rows.map(([m,d,struck])=>({start:m/1000,end:(m+d)/1000,struck:!!struck}))),{player:{duration},removeGapsEnabled:gaps});}
  it('one struck word keeps both neighboring phrases',()=>expect(sections([[1000,500],[3000,500,1],[6000,500]])).toEqual([{start:0,end:3},{start:3.5,end:10}]));
  it('a struck run removes the pauses between the struck words',()=>expect(sections([[1000,500],[3000,200,1],[4000,200,1],[6000,500]])).toEqual([{start:0,end:3},{start:4.2,end:10}]));
  it('takes the maximum end for overlapping struck words',()=>expect(sections([[1000,500],[3000,1000,1],[3200,300,1],[6000,500]])).toEqual([{start:0,end:3},{start:4,end:10}]));
  it('keeps all sound when there are no deletions and gaps are off',()=>expect(sections([[1000,500],[2000,500]])).toEqual([{start:0,end:10}]));
});

describe('preview playback controller adapted from Hyperaudio', () => {
  function player() { const p=new EventTarget() as any; Object.assign(p,{currentTime:0,duration:20,readyState:4,seeking:false,paused:true,pause:vi.fn()}); vi.stubGlobal('requestAnimationFrame',vi.fn(()=>1)); vi.stubGlobal('cancelAnimationFrame',vi.fn()); return p; }
  const pplan={sections:[{start:2,end:3},{start:4,end:18}],cuts:[{start:0,end:2},{start:3,end:4},{start:18,end:20}],removedSeconds:5,editedDuration:15,warnings:[],subtitles:{}};
  it('skips leading/deleted intervals and lands exactly on the next kept word',()=>{const p=player(),c=new CutPreviewController(p,()=>{}); c.setPlan(pplan); expect(p.currentTime).toBe(2); p.currentTime=3.2;p.dispatchEvent(new Event('timeupdate'));expect(p.currentTime).toBe(4);});
  it('allows original-sound checking while skip is switched off',()=>{const p=player(),c=new CutPreviewController(p,()=>{});c.setPlan(pplan);c.setEnabled(false);p.currentTime=3.2;p.dispatchEvent(new Event('seeked'));expect(p.currentTime).toBe(3.2);});
  it('stops after the final kept interval and handles an all-cut source',()=>{const p=player(),c=new CutPreviewController(p,()=>{});c.setPlan(pplan);p.currentTime=18;c.check();expect(p.currentTime).toBe(20);expect(p.pause).toHaveBeenCalled();c.setPlan({...pplan,sections:[]});p.currentTime=0;c.check();expect(p.currentTime).toBe(20);});
});
