import { afterEach, describe, expect, it } from 'vitest';
import { createReviewEditor, initialDocument, ensureMetadata, reviewPayload, trackedDelete, trackedReplace, type Chapter } from '../model';
import { applySubtitleAction, removeSubtitleAction, subtitleCues, subtitleSelection, layoutEdits, validateSubtitleLayout } from '../subtitle-layout';
import { buildPreviewPlan } from '../preview';

const chapter: Chapter = { number: '03', file: 'synthetic.mp4', duration: 10, source_text_hash: 'fixture', script: '', segments: [
  { id: 's1', start: 1, end: 4, text: '都要记清楚。', preselection: 'keep', preselection_reason: '', script_match: '', timings: [
    { text: '都要', start_char: 0, end_char: 2, start: 1, end: 1.5 }, { text: '记', start_char: 2, end_char: 3, start: 2, end: 2.3 }, { text: '清楚', start_char: 3, end_char: 5, start: 2.5, end: 3.5 }] },
  { id: 's2', start: 4.2, end: 5, text: '来源，', preselection: 'keep', preselection_reason: '', script_match: '', timings: [{ text: '来源', start_char: 0, end_char: 2, start: 4.2, end: 4.8 }] },
  { id: 's3', start: 5.5, end: 8, text: '报告数字。', preselection: 'keep', preselection_reason: '', script_match: '', timings: [{ text: '报告', start_char: 0, end_char: 2, start: 5.5, end: 6.2 }, { text: '数字', start_char: 2, end_char: 4, start: 7, end: 7.8 }] },
] };
const editors: ReturnType<typeof createReviewEditor>[] = [];
function make(ch = chapter, doc = initialDocument(ch)) { const e = createReviewEditor({ element: document.createElement('div'), content: doc }); editors.push(e); return e; }
function caret(e: ReturnType<typeof make>, text: string, offset = 0) { let pos = -1; e.state.doc.descendants((n, p) => { if (pos < 0 && n.isText && n.text!.includes(text)) pos = p + n.text!.indexOf(text) + offset; }); if (pos < 0) throw Error(text); e.commands.setTextSelection(pos); return pos; }
function plan(e: ReturnType<typeof make>, ch = chapter, meta: any = {}) { ensureMetadata(e, ch, meta); return buildPreviewPlan(e, ch, meta, 10); }
function media(p: ReturnType<typeof plan>) { return { sections: p.sections, cuts: p.cuts, duration: p.editedDuration, warnings: p.warnings }; }
afterEach(() => editors.splice(0).forEach(e => e.destroy()));

describe('source-bound subtitle layout over MAWE and ProseMirror', () => {
  it('rejects corrupted imported boundaries before they can enter the review book', () => {
    const e = make(); caret(e, '来源'); applySubtitleAction(e, chapter, 'merge_previous'); expect(() => validateSubtitleLayout(e, chapter)).not.toThrow();
    const doc = e.getJSON(); doc.attrs!.subtitleEdits[0].at = -1; const invalid = make(chapter, doc); expect(() => validateSubtitleLayout(invalid, chapter)).toThrow();
  });
  it('merges the next cue and changes only display, preserving every media interval', () => {
    const e = make(), before = plan(e), original = reviewPayload(chapter, e, {}); caret(e, '记清楚');
    applySubtitleAction(e, chapter, 'merge_next'); const after = plan(e);
    expect(after.displaySubtitles![0]).toMatchObject({ start: 1, end: 5, text: '都要记清楚。来源，' });
    expect(media(after)).toEqual(media(before)); expect(reviewPayload(chapter, e, {}).original_text).toBe(original.original_text);
  });
  it('merges previous based on the caret and retains punctuation verbatim', () => {
    const e = make(); caret(e, '来源'); applySubtitleAction(e, chapter, 'merge_previous');
    expect(subtitleCues(e, chapter).map(c => c.text)).toEqual(['都要记清楚。来源，', '报告数字。']);
  });
  it('splits a merged cue back at its original source boundary', () => {
    const e = make(); caret(e, '来源'); applySubtitleAction(e, chapter, 'merge_previous'); caret(e, '来源');
    expect(subtitleSelection(e, chapter).split).toBe(true); applySubtitleAction(e, chapter, 'split');
    expect(subtitleCues(e, chapter).map(c => c.text)).toEqual(['都要记清楚。', '来源，', '报告数字。']);
  });
  it('splits inside a cue at native timing without modifying sound coverage', () => {
    const e = make(), before = plan(e); caret(e, '记清楚'); applySubtitleAction(e, chapter, 'split');
    expect(subtitleCues(e, chapter).slice(0, 2).map(c => [c.text, c.start, c.end])).toEqual([['都要', 1, 2], ['记清楚。', 2, 4]]);
    expect(media(plan(e))).toEqual(media(before));
  });
  it('does not split inside one unsplittable native multi-character token', () => {
    const e = make(); caret(e, '清楚', 1); expect(subtitleSelection(e, chapter).split).toBe(false);
    expect(() => applySubtitleAction(e, chapter, 'split')).toThrow(); expect(layoutEdits(e)).toHaveLength(0);
  });
  it('does not guess a timestamp when native alignment is absent', () => {
    const ch = structuredClone(chapter); ch.segments[0].timings = []; const e = make(ch); caret(e, '记清楚');
    expect(() => applySubtitleAction(e, ch, 'split')).toThrow(); expect(layoutEdits(e)).toHaveLength(0);
  });
  it('rejects range selections and first/last merge edges without adding revisions', () => {
    const e = make(); const p = caret(e, '都要'); expect(subtitleSelection(e, chapter).previous).toBe(false);
    expect(() => applySubtitleAction(e, chapter, 'merge_previous')).toThrow();
    e.commands.setTextSelection({ from: p, to: p + 2 }); expect(() => applySubtitleAction(e, chapter, 'merge_next')).toThrow();
    caret(e, '报告'); expect(subtitleSelection(e, chapter).next).toBe(false); expect(() => applySubtitleAction(e, chapter, 'merge_next')).toThrow();
  });
  it('undoes and redoes layout with the existing editor history', () => {
    const e = make(); caret(e, '来源'); applySubtitleAction(e, chapter, 'merge_previous'); expect(subtitleCues(e, chapter)).toHaveLength(2);
    e.commands.undo(); expect(subtitleCues(e, chapter)).toHaveLength(3); e.commands.redo(); expect(subtitleCues(e, chapter)).toHaveLength(2);
  });
  it('keeps existing confirmed lexical revisions and all five fields unchanged', () => {
    const e = make(), p = caret(e, '。'); e.commands.setTextSelection({ from: p, to: p + 1 }); trackedDelete(e);
    const meta: any = {}; ensureMetadata(e, chapter, meta); const id = Object.keys(meta)[0]; meta[id].effect = 'transcript_only'; meta[id].status = 'confirmed'; meta[id].confirmed_at = '2026-10-26';
    const before = reviewPayload(chapter, e, meta).revisions[0]; caret(e, '来源'); applySubtitleAction(e, chapter, 'merge_previous');
    const after = reviewPayload(chapter, e, meta); expect(after.revisions.find(r => r.change_id === id)).toEqual(before);
    const layout = after.revisions.find(r => r.kind === 'subtitle_structure')!; expect(layout.effect).toBe('transcript_only');
    for (const key of ['intent', 'decision', 'reason', 'evidence', 'effective_scope']) expect((layout as any)[key]).toBeTruthy();
    expect(subtitleCues(e, chapter)[0].text).toBe('都要记清楚来源，');
  });
  it('keeps corrected text in the proper cue and disables splits inside inserted text', () => {
    const e = make(), p = caret(e, '清楚'); e.commands.setTextSelection({ from: p, to: p + 2 }); trackedReplace(e, '明白');
    caret(e, '来源'); applySubtitleAction(e, chapter, 'merge_previous'); expect(subtitleCues(e, chapter)[0].text).toBe('都要记明白。来源，');
    caret(e, '明白', 1); expect(subtitleSelection(e, chapter).split).toBe(false);
  });
  it('restores one layout decision without restoring deleted text', () => {
    const e = make(), p = caret(e, '。'); e.commands.setTextSelection({ from: p, to: p + 1 }); trackedDelete(e);
    caret(e, '来源'); const op = applySubtitleAction(e, chapter, 'merge_previous'); removeSubtitleAction(e, op.id);
    expect(subtitleCues(e, chapter)[0].text).toBe('都要记清楚'); expect(reviewPayload(chapter, e, {}).revisions).toHaveLength(1);
  });
  it('round-trips the grouping in the same document snapshot and v3 payload', () => {
    const e = make(); caret(e, '来源'); applySubtitleAction(e, chapter, 'merge_previous'); const payload = reviewPayload(chapter, e, {});
    const restored = make(chapter, JSON.parse(JSON.stringify(payload.doc))); const meta = Object.fromEntries(payload.revisions.map(r => [r.change_id, { ...r }]));
    expect(subtitleCues(restored, chapter)).toEqual(subtitleCues(e, chapter)); expect(reviewPayload(chapter, restored, meta)).toEqual(payload);
  });
  it('does not split a surrogate pair or a punctuation-only side', () => {
    const ch = structuredClone(chapter); ch.segments[0].text = '你😀好吗。'; ch.segments[0].timings = [{text:'你',start_char:0,end_char:1,start:1,end:1.2},{text:'😀',start_char:1,end_char:3,start:1.5,end:2},{text:'好吗',start_char:3,end_char:5,start:2.5,end:3.5}];
    const e = make(ch); caret(e, '😀', 1); expect(subtitleSelection(e, ch).split).toBe(false); caret(e, '。'); expect(subtitleSelection(e, ch).split).toBe(false);
  });
  it('merges across a deleted empty cue without losing either surviving subtitle', () => {
    const e = make(), p = caret(e, '来源，'); e.commands.setTextSelection({from:p,to:p+3}); trackedDelete(e); caret(e, '记清楚'); applySubtitleAction(e, chapter, 'merge_next');
    expect(subtitleCues(e, chapter).map(c => c.text)).toEqual(['都要记清楚。报告数字。']);
  });
});
