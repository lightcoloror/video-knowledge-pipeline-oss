import { afterEach, describe, expect, it } from 'vitest';
import { createReviewEditor, initialDocument, trackedDelete, trackedReplace, groupedReview, ensureMetadata, getBaseText, getResultText, reviewPayload, originRuns, type Chapter } from '../model';

const chapter: Chapter = { number: '01', file: 'fixture.mp4', source_text_hash: 'fixture-hash', script: '', segments: [
  { id: 's1', start: 2, end: 8, text: '这是一句正常的话，里面错词需要检查，后面继续正常。', preselection: 'keep', preselection_reason: '', script_match: '' },
  { id: 's2', start: 9, end: 18, text: '第二段也有正常文字，结尾还有重复重复，需要确认。', preselection: 'keep', preselection_reason: '', script_match: '' },
] };
const originals = chapter.segments.map(s => s.text).join('');
const editors: ReturnType<typeof createReviewEditor>[] = [];
function make(content = initialDocument(chapter)) { const el = document.createElement('div'); document.body.append(el); const e = createReviewEditor({ element: el, content }); editors.push(e); return e; }
function position(e: ReturnType<typeof make>, text: string) { let at = -1; e.state.doc.descendants((n, p) => { if (n.isText && at < 0 && n.text!.includes(text)) at = p + n.text!.indexOf(text); }); if (at < 0) throw Error(`Not found: ${text}`); return at; }
afterEach(() => { for (const e of editors.splice(0)) e.destroy(); document.body.replaceChildren(); });

describe('ASR source adaptation using the unmodified upstream revision engine', () => {
  it('selects a wrong word inside a normal sentence, preserving both sides and the original', () => {
    const e = make(); const from = position(e, '错词'); e.commands.setTextSelection({ from, to: from + 2 }); expect(trackedDelete(e)).toBe(true);
    const g = groupedReview(e, chapter)[0]; expect(g.deleted).toBe('错词'); expect(g.anchors).toHaveLength(1);
    expect(g.anchors[0]).toMatchObject({ segment_id: 's1', start_char: chapter.segments[0].text.indexOf('错词'), end_char: chapter.segments[0].text.indexOf('错词') + 2, original_text: '错词', timing_status: 'word_alignment_required' });
    expect(getBaseText(e).replace(/\n/g, '')).toBe(originals); expect(getResultText(e)).toContain('里面需要检查');
  });
  it('shows the old word and correction adjacent, with the same upstream revision id', () => {
    const e = make(); const from = position(e, '错词'); e.commands.setTextSelection({ from, to: from + 2 }); expect(trackedReplace(e, '正确词')).toBe(true);
    const [g] = groupedReview(e, chapter); expect(g).toMatchObject({ kind: 'replacement', deleted: '错词', inserted: '正确词' });
    expect(e.getHTML()).toMatch(/<del[^>]*>.*错词.*<\/del><ins/s); expect(getBaseText(e).replace(/\n/g, '')).toBe(originals);
    expect(getResultText(e)).toContain('里面正确词需要检查');
  });
  it('keeps exact original-character offsets after an earlier correction inserts more characters', () => {
    const e = make(); let at = position(e, '错词'); e.commands.setTextSelection({ from: at, to: at + 2 }); trackedReplace(e, '一段更长的正确表达');
    at = position(e, '继续正常'); e.commands.setTextSelection({ from: at, to: at + 2 }); trackedDelete(e);
    const g = groupedReview(e, chapter).find(g => g.deleted === '继续')!;
    expect(g.anchors[0].start_char).toBe(chapter.segments[0].text.indexOf('继续'));
    expect(getBaseText(e).replace(/\n/g, '')).toBe(originals);
  });
  it('supports selection across normal paragraphs and retains original source text', () => {
    const e = make(); const from = position(e, '后面继续正常'); const to = position(e, '第二段也有') + 5;
    e.commands.setTextSelection({ from, to }); expect(trackedDelete(e)).toBe(true);
    const groups = groupedReview(e, chapter); expect(groups.flatMap(g => g.anchors).map(a => a.segment_id)).toEqual(expect.arrayContaining(['s1', 's2']));
    expect(getBaseText(e).replace(/\n/g, '')).toBe(originals);
  });
  it('undoes and redoes a character-level deletion without losing source anchors', () => {
    const e = make(); const at = position(e, '错词'); e.commands.setTextSelection({ from: at, to: at + 2 }); trackedDelete(e);
    expect(e.commands.undo()).toBe(true); expect(groupedReview(e, chapter)).toHaveLength(0);
    expect(e.commands.redo()).toBe(true); expect(groupedReview(e, chapter)[0].anchors[0].original_text).toBe('错词');
  });
  it('exports all five explanation fields and distinguishes transcript correction from media deletion', () => {
    const e = make(); const at = position(e, '错词'); e.commands.setTextSelection({ from: at, to: at + 2 }); trackedReplace(e, '正确词');
    const meta = {}; ensureMetadata(e, chapter, meta); const p = reviewPayload(chapter, e, meta);
    expect(p.revisions[0].effect).toBe('transcript_only');
    for (const key of ['intent', 'decision', 'reason', 'evidence', 'effective_scope']) expect((p.revisions[0] as any)[key]).toBeTruthy();
  });
  it('round-trips the rich document and source mapping without accepting away the deletion trace', () => {
    const e = make(); const at = position(e, '错词'); e.commands.setTextSelection({ from: at, to: at + 2 }); trackedDelete(e);
    const restored = make(JSON.parse(JSON.stringify(e.getJSON()))); expect(groupedReview(restored, chapter)[0].anchors).toEqual(groupedReview(e, chapter)[0].anchors);
    expect(restored.getHTML()).toContain('<del'); expect(getBaseText(restored).replace(/\n/g, '')).toBe(originals);
  });
  it('restores the original after rejecting a replacement', () => {
    const e = make(); const at = position(e, '错词'); e.commands.setTextSelection({ from: at, to: at + 2 }); trackedReplace(e, '正确词');
    e.commands.rejectChange(groupedReview(e, chapter)[0].id); expect(groupedReview(e, chapter)).toHaveLength(0); expect(getResultText(e).replace(/\n/g, '')).toBe(originals);
  });
  it('imports legacy manual keep overrides without silently applying a preset deletion', () => {
    const ch = structuredClone(chapter); ch.segments[0].preselection = 'delete';
    const kept = make(initialDocument(ch, { '01:s1': 'keep' })); expect(groupedReview(kept, ch)).toHaveLength(0);
    const deleted = make(initialDocument(ch)); expect(groupedReview(deleted, ch)[0].deleted).toBe(ch.segments[0].text);
  });
  it('renders correction text containing markup literally rather than executing it', () => {
    const e = make(); const at = position(e, '错词'); e.commands.setTextSelection({ from: at, to: at + 2 }); trackedReplace(e, '<img src=x onerror=alert(1)>');
    expect(e.view.dom.querySelector('img')).toBeNull(); expect(getResultText(e)).toContain('<img'); expect(originRuns(e, chapter).length).toBeGreaterThan(0);
  });
  it('requires confirmation again when a previously confirmed correction is extended', () => {
    const e = make(); const at = position(e, '错词'); e.commands.setTextSelection({ from: at, to: at + 2 }); trackedReplace(e, '正确词');
    const meta: any = {}; const [g] = ensureMetadata(e, chapter, meta); meta[g.id].status = 'confirmed'; meta[g.id].confirmed_at = new Date().toISOString();
    e.commands.setTextSelection(g.to); e.view.someProp('handleTextInput', fn => fn(e.view, g.to, g.to, '补充', () => e.state.tr));
    ensureMetadata(e, chapter, meta); expect(meta[g.id].status).toBe('proposed'); expect(meta[g.id].confirmed_at).toBeUndefined();
  });
});
