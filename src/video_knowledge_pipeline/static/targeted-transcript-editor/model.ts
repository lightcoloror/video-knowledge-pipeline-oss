// RWE adapter for sungkhum/tiptap-track-changes, commit 753eef717f7ffaed0cb0e451ecb8b93f5ddaca22.
// The upstream editor, change marks, input handling, undo and grouping are reused unchanged.
import { Editor, Mark, mergeAttributes } from '@tiptap/core';
import Document from '@tiptap/extension-document';
import Paragraph from '@tiptap/extension-paragraph';
import Text from '@tiptap/extension-text';
import History from '@tiptap/extension-history';
import { TrackChangesExtension, getGroupedChanges, getBaseText, getResultText } from './vendor/tiptap-track-changes/src/index';
import { CompositionAdapter } from './ime-adapter';
import { subtitleLayoutGroups, type LayoutEdit } from './subtitle-layout';

export { getGroupedChanges, getBaseText, getResultText };
export type NativeTiming = { start_char: number; end_char: number; text: string; start: number; end: number; source_timestamp_index?: number };
export type Segment = { id: string; start: number; end: number; text: string; preselection: string; preselection_reason: string; script_match: string; timings?: NativeTiming[] };
export type Chapter = { number: string; file: string; source_id?: string; duration?: number; source_text_hash: string; script: string; segments: Segment[] };
export type Effect = 'cut_media' | 'transcript_only' | 'review_required';
export type RevisionMeta = { intent: string; decision: string; reason: string; evidence: string; effective_scope: string; effect: Effect; status: 'proposed' | 'confirmed'; origin: string; confirmed_at?: string; content_fingerprint?: string };
export type SourceAnchor = { source_id: string; chapter: string; file: string; segment_id: string; start_char: number; end_char: number; original_text: string; envelope_start_s: number; envelope_end_s: number; timing_status: string; native_ranges?: NativeTiming[] };
export type ChapterReview = { source_hash: string; doc: any; meta: Record<string, RevisionMeta> };
export const EFFECTS: Record<Effect, string> = { cut_media: '剪掉对应原声片段', transcript_only: '仅修正识别文字，保留原声', review_required: '待回听后决定' };

const Origin = Mark.create({
  name: 'sourceOrigin', inclusive: false,
  addAttributes() { return { sourceId: { default: null } }; },
  parseHTML() { return [{ tag: 'span[data-source-id]' }]; },
  renderHTML({ HTMLAttributes }) { return ['span', mergeAttributes({ 'data-source-id': HTMLAttributes.sourceId }), 0]; },
});
const SourceParagraph = Paragraph.extend({
  addAttributes() { return { ...this.parent?.(), sourceTime: { default: null, renderHTML: a => a.sourceTime ? { 'data-time': a.sourceTime } : {} } }; },
});
const ReviewDocument = Document.extend({ addAttributes() { return { subtitleEdits: { default: null, rendered: false } }; } });
export function time(t: number) { return `${String(Math.floor(t / 60)).padStart(2, '0')}:${(t % 60).toFixed(2).padStart(5, '0')}`; }
export function sourceId(ch: Chapter, s: Segment) { return `${ch.number}:${s.id}`; }

// Adaptation of the upstream demo content loader. ASR text stays verbatim; only paragraph grouping changes.
export function initialDocument(ch: Chapter, legacy: Record<string, string> = {}) {
  const content: any[] = []; let group: Segment[] = [];
  function flush() {
    if (!group.length) return;
    content.push({ type: 'paragraph', attrs: { sourceTime: `${time(group[0].start)}–${time(group.at(-1)!.end)}` }, content: group.map(s => {
      const marks: any[] = [{ type: 'sourceOrigin', attrs: { sourceId: sourceId(ch, s) } }];
      if ((legacy[sourceId(ch, s)] || s.preselection) === 'delete') marks.push({ type: 'deletion', attrs: {
        changeId: `preset:${sourceId(ch, s)}`, authorId: 'preselection', authorName: '稿文预选', authorColor: '#a43131', timestamp: '2026-10-26T00:00:00Z',
      } });
      return { type: 'text', text: s.text, marks };
    }) });
    group = [];
  }
  for (const s of ch.segments) {
    group.push(s);
    const length = group.reduce((n, x) => n + x.text.length, 0);
    if ((/[。？！?！]$/.test(s.text) && length >= 24) || length >= 100 || group.length >= 7) flush();
  }
  flush(); return { type: 'doc', content };
}

export function createReviewEditor(options: { element?: HTMLElement; content: any; extraExtensions?: any[]; onUpdate?: (p: any) => void; onSelectionUpdate?: (p: any) => void }) {
  // Adapted from upstream demo/src/main.ts; revision engine and history are upstream modules.
  return new Editor({ element: options.element, extensions: [ReviewDocument, SourceParagraph, Text, History, Origin, CompositionAdapter.configure({ onFinish: () => options.onUpdate?.({}) }),
    TrackChangesExtension.configure({ author: { id: 'local-reviewer', name: '本人复核', color: '#22634d' }, mode: 'suggest' }), ...(options.extraExtensions || [])],
    content: options.content, onUpdate: options.onUpdate || (() => {}), onSelectionUpdate: options.onSelectionUpdate || (() => {}),
    editorProps: { attributes: { 'aria-label': '口播逐字稿修订编辑器', spellcheck: 'false' } },
  });
}
export function trackedDelete(editor: Editor) {
  if (editor.state.selection.empty) return false;
  let handled = false;
  editor.view.someProp('handleKeyDown', fn => { handled = !!fn(editor.view, new KeyboardEvent('keydown', { key: 'Delete', bubbles: true })); return handled; });
  return handled;
}
export function trackedReplace(editor: Editor, replacement: string) {
  const { from, to } = editor.state.selection; if (from === to || !replacement) return false;
  let handled = false;
  editor.view.someProp('handleTextInput', fn => { handled = !!fn(editor.view, from, to, replacement, () => editor.state.tr.insertText(replacement, from, to)); return handled; });
  return handled;
}

// Original-text source positions, derived from upstream's preserved deletion/insertion marks.
// We never infer word times by proportionally dividing phrase timestamps.
export function originRuns(editor: Editor, ch: Chapter) {
  const offsets = new Map<string, number>(); const runs: { from: number; to: number; id: string; offset: number }[] = [];
  editor.state.doc.descendants((node, pos) => {
    if (!node.isText || node.marks.some(m => m.type.name === 'insertion')) return;
    const id = node.marks.find(m => m.type.name === 'sourceOrigin')?.attrs.sourceId;
    if (!id || !id.startsWith(ch.number + ':')) return;
    const offset = offsets.get(id) || 0; runs.push({ from: pos, to: pos + node.nodeSize, id, offset }); offsets.set(id, offset + node.nodeSize);
  }); return runs;
}
export function anchorsForRange(editor: Editor, ch: Chapter, from: number, to: number): SourceAnchor[] {
  const result: SourceAnchor[] = []; const runs = originRuns(editor, ch);
  const matching = runs.filter(r => r.from < to && r.to > from);
  if (!matching.length) { const near = runs.find(r => r.to >= from) || runs.at(-1); if (near) matching.push({ ...near, from: Math.min(near.to, Math.max(near.from, from)), to: Math.min(near.to, Math.max(near.from, from)), offset: near.offset + Math.max(0, Math.min(from - near.from, near.to - near.from)) }); }
  for (const r of matching) {
    const s = ch.segments.find(x => sourceId(ch, x) === r.id); if (!s) continue;
    const a = r.offset + Math.max(0, from - r.from), b = r.offset + Math.min(r.to - r.from, Math.max(0, to - r.from));
    const start = Math.min(s.text.length, a), end = Math.max(start, Math.min(s.text.length, b));
    const previous = result.at(-1);
    if (previous?.source_id === r.id && previous.end_char === start) { previous.end_char = end; previous.original_text = s.text.slice(previous.start_char, end); }
    else result.push({ source_id: r.id, chapter: ch.number, file: ch.file, segment_id: s.id, start_char: start, end_char: end, original_text: s.text.slice(start, end), envelope_start_s: s.start, envelope_end_s: s.end, timing_status: start === 0 && end === s.text.length ? 'phrase_interval_review_required' : 'word_alignment_required' });
  }
  for (const anchor of result) {
    const seg = ch.segments.find(s => s.id === anchor.segment_id);
    const ranges = seg?.timings?.filter(t => t.start_char < anchor.end_char && t.end_char > anchor.start_char);
    if (ranges?.length) { anchor.native_ranges = ranges; anchor.timing_status = 'native_asr_character_timing'; }
  }
  return result;
}
export function groupedReview(editor: Editor, ch: Chapter) {
  const textGroups = [...getGroupedChanges(editor)].map(([id, changes]) => {
    // Upstream also reports paragraph-boundary edits as deletions. A boundary is
    // not the whole paragraph's audio: only actual inline deletion marks own audio cuts.
    const deleted = changes.filter(x => x.type === 'deletion' && editor.state.doc.nodeAt(x.from)?.isText).sort((a, b) => a.from - b.from);
    const inserted = changes.filter(x => x.type === 'insertion' && editor.state.doc.nodeAt(x.from)?.isText).sort((a, b) => a.from - b.from);
    const from = Math.min(...changes.map(x => x.from)), to = Math.max(...changes.map(x => x.to));
    const anchors = deleted.length ? deleted.flatMap(x => anchorsForRange(editor, ch, x.from, x.to)) : anchorsForRange(editor, ch, from, from);
    return { id, changes, from, to, deleted: deleted.map(x => x.text).join(''), inserted: inserted.map(x => x.text).join(''), kind: deleted.length ? (inserted.length ? 'replacement' : 'deletion') : (inserted.length ? 'insertion' : 'structure'), anchors, layout: undefined as LayoutEdit | undefined };
  });
  return [...textGroups, ...subtitleLayoutGroups(editor, ch).map(g => ({ ...g, anchors: g.anchors as SourceAnchor[] }))].sort((a, b) => a.from - b.from);
}
export function ensureMetadata(editor: Editor, ch: Chapter, meta: Record<string, RevisionMeta>) {
  const groups = groupedReview(editor, ch);
  for (const g of groups) {
    const first = g.anchors[0]; const s = ch.segments.find(x => sourceId(ch, x) === first?.source_id);
    const fingerprint = JSON.stringify([g.deleted, g.inserted, g.anchors.map(a => [a.source_id, a.start_char, a.end_char])]);
    if (!meta[g.id]) {
      const effect: Effect = g.kind === 'deletion' ? 'cut_media' : g.kind === 'structure' ? 'review_required' : 'transcript_only';
      meta[g.id] = { intent: g.kind === 'deletion' ? '删除口误、重读或冗余候选' : '修正识别文字并保留修订痕迹', decision: EFFECTS[effect], reason: g.id.startsWith('preset:') ? (s?.preselection_reason || '按旧页复核选择迁入') : '直接在逐字稿中提出修改；可在此补充判断理由。',
        evidence: `${ch.file}；原识别文字“${g.deleted || s?.text || ''}”。${s?.script_match ? `口播稿 v0.6 对照：${s.script_match}` : '原声回听状态待确认。'}`, effective_scope: '', effect, status: 'proposed', origin: g.id.startsWith('preset:') ? 'legacy_preselection' : 'manual_text_edit' };
    }
    if (g.layout && meta[g.id].origin !== 'subtitle_layout') Object.assign(meta[g.id], { intent: g.layout.action === 'split' ? '在光标处修正字幕断句' : '合并相邻字幕，修正错误断句',
      reason: '字幕显示分组与实际语义不一致；只调整字幕边界，不改变文字修订和原声播放区间。',
      evidence: `${ch.file}；原文哈希 ${ch.source_text_hash}；原字幕“${g.layout.before}”；调整后“${g.layout.after}”；原始字符偏移 ${g.layout.at}。`, origin: 'subtitle_layout' });
    if (meta[g.id].content_fingerprint && meta[g.id].content_fingerprint !== fingerprint) { meta[g.id].status = 'proposed'; delete meta[g.id].confirmed_at; }
    meta[g.id].content_fingerprint = fingerprint;
    meta[g.id].effective_scope = g.anchors.map(a => `第${ch.number}段 ${a.segment_id} 字符[${a.start_char},${a.end_char})；原片 ${time(a.envelope_start_s)}–${time(a.envelope_end_s)}（短语范围）${a.native_ranges?.length ? `；字词 ${time(a.native_ranges[0].start)}–${time(a.native_ranges.at(-1)!.end)}` : ''}`).join('；') || `第${ch.number}段文稿结构`;
  }
  return groups;
}
export function reviewPayload(ch: Chapter, editor: Editor, meta: Record<string, RevisionMeta>) {
  const groups = ensureMetadata(editor, ch, meta);
  return { chapter: ch.number, file: ch.file, source_hash: ch.source_text_hash, doc: editor.getJSON(), original_text: getBaseText(editor), proposed_text: getResultText(editor),
    revisions: groups.map(g => ({ change_id: g.id, kind: g.kind, original: g.deleted, corrected: g.inserted, ...meta[g.id], source_anchors: g.anchors, ...(g.layout ? { subtitle_layout: g.layout } : {}) })),
  };
}
