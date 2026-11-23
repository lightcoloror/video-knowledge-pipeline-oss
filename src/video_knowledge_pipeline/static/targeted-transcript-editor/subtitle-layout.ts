// Subtitle grouping adapter over MAWE cue operations and ProseMirror's
// reversible document-attribute steps. Source text and media plans stay intact.
import type { Editor } from '@tiptap/core';
import { closeHistory } from '@tiptap/pm/history';
import type { Chapter } from './model';
import { mergeSubtitleCues, splitSubtitleText } from './vendor/mawe/subtitle-operations';

export type LayoutEdit = { id: string; action: 'merge_previous' | 'merge_next' | 'split'; at: number; scope_start: number; scope_end: number; before: string; after: string };
export type SubtitleUnit = { text: string; pos: number; to: number; at: number; inserted: boolean };
export type SubtitleCue = { id: string; start: number; end: number; text: string; at: number; end_at: number; from: number; to: number; units: SubtitleUnit[] };
export const layoutEdits = (editor: Editor): LayoutEdit[] => editor.state.doc.attrs.subtitleEdits || [];
export function validateSubtitleLayout(editor: Editor, ch: Chapter) {
  const edits = layoutEdits(editor), total = ch.segments.reduce((n, s) => n + s.text.length, 0), ids = new Set<string>();
  if (!Array.isArray(edits) || edits.length > 5000) throw Error('字幕断句记录格式无效。');
  for (const op of edits) {
    if (!op || typeof op.id !== 'string' || !op.id.startsWith('subtitle:') || ids.has(op.id) || !['split', 'merge_previous', 'merge_next'].includes(op.action)
      || ![op.at, op.scope_start, op.scope_end].every(Number.isInteger) || !(0 <= op.scope_start && op.scope_start < op.at && op.at < op.scope_end && op.scope_end <= total)
      || typeof op.before !== 'string' || typeof op.after !== 'string' || boundaryTime(ch, op.at, 'start') === null) throw Error('字幕断句记录的原文锚点或字词时间无效。');
    ids.add(op.id);
  }
}
export function sourceOffsets(ch: Chapter) {
  let offset = 0;
  return ch.segments.map(segment => { const start = offset; offset += segment.text.length; return { segment, start, end: offset }; });
}
export function subtitleUnits(editor: Editor, ch: Chapter): SubtitleUnit[] {
  const sources = sourceOffsets(ch), offsets = new Map<string, number>();
  const original: { pos: number; to: number; at: number; deleted: boolean; change?: string; text: string }[] = [];
  const inserted: { pos: number; text: string; change: string }[] = [];
  editor.state.doc.descendants((node, pos) => {
    if (!node.isText) return;
    const insertion = node.marks.find(m => m.type.name === 'insertion');
    if (insertion) { inserted.push({ pos, text: node.text!, change: insertion.attrs.changeId }); return; }
    const id = node.marks.find(m => m.type.name === 'sourceOrigin')?.attrs.sourceId;
    const source = sources.find(s => `${ch.number}:${s.segment.id}` === id); if (!source) return;
    let offset = offsets.get(id) || 0;
    const deletion = node.marks.find(m => m.type.name === 'deletion');
    for (const text of node.text!) { original.push({ pos, to: pos + text.length, at: source.start + offset, deleted: !!deletion, change: deletion?.attrs.changeId, text }); offset += text.length; pos += text.length; }
    offsets.set(id, offset);
  });
  const units: SubtitleUnit[] = original.filter(u => !u.deleted).map(u => ({ ...u, inserted: false }));
  for (const run of inserted) {
    const replaced = original.find(u => u.change === run.change);
    const next = original.find(u => u.to > run.pos), previous = original.filter(u => u.to <= run.pos).at(-1);
    const at = replaced?.at ?? next?.at ?? previous?.at ?? 0;
    let pos = run.pos;
    for (const text of run.text) { units.push({ text, pos, to: pos + text.length, at, inserted: true }); pos += text.length; }
  }
  return units.sort((a, b) => a.pos - b.pos);
}
export function boundaryTime(ch: Chapter, at: number, edge: 'start' | 'end'): number | null {
  const sources = sourceOffsets(ch), total = sources.at(-1)?.end || 0;
  if (at === total) return sources.at(-1)?.segment.end ?? null;
  const source = sources.find(s => s.start <= at && at < s.end); if (!source) return null;
  if (at === source.start) return edge === 'end' && source.start > 0 ? sources[sources.indexOf(source) - 1].segment.end : source.segment.start;
  const offset = at - source.start, tokens = source.segment.timings || [];
  if (tokens.some(t => t.start_char < offset && offset < t.end_char)) return null;
  const next = tokens.find(t => t.start_char >= offset), previous = tokens.filter(t => t.end_char <= offset).at(-1);
  return next && previous && next.start >= previous.end ? next.start : null;
}
export function subtitleCues(editor: Editor, ch: Chapter): SubtitleCue[] {
  const sources = sourceOffsets(ch), total = sources.at(-1)?.end || 0;
  const boundaries = new Set(sources.map(s => s.start)); boundaries.add(total);
  for (const op of layoutEdits(editor)) {
    if (op.action === 'split') boundaries.add(op.at);
    else for (const at of boundaries) if (op.scope_start < at && at < op.scope_end) boundaries.delete(at);
  }
  const points = [...boundaries].sort((a, b) => a - b), units = subtitleUnits(editor, ch), cues: SubtitleCue[] = [];
  for (let i = 0; i < points.length - 1; i++) {
    const at = points[i], end_at = points[i + 1], selected = units.filter(u => at <= u.at && u.at < end_at);
    const text = selected.map(u => u.text).join(''); if (!text.trim()) continue;
    const start = boundaryTime(ch, at, 'start'), end = boundaryTime(ch, end_at, 'end');
    if (start === null || end === null) continue;
    cues.push({ id: `subtitle:${at}:${end_at}`, start, end, text, at, end_at, from: selected[0].pos, to: selected.at(-1)!.to, units: selected });
  }
  return cues;
}
export function subtitleSelection(editor: Editor, ch: Chapter) {
  const { from, to } = editor.state.selection, cues = subtitleCues(editor, ch);
  const cue = cues.find(c => c.units.some(u => u.pos <= from && from < u.to)) || cues.find(c => c.from >= from) || cues.at(-1);
  const index = cue ? cues.indexOf(cue) : -1, unit = cue?.units.find(u => u.pos >= from);
  const inside = cue && unit && unit.pos === from && !unit.inserted && from > cue.from && from < cue.to;
  let split = !!inside && boundaryTime(ch, unit!.at, 'start') !== null;
  if (split && cue) { try { splitSubtitleText(cue.text, cue.units.filter(u => u.pos < from).reduce((n, u) => n + u.text.length, 0)); } catch { split = false; } }
  return { cues, cue, index, previous: from === to && index > 0, next: from === to && index >= 0 && index < cues.length - 1, split: from === to && split, unit };
}
function setEdits(editor: Editor, edits: LayoutEdit[]) {
  editor.view.dispatch(closeHistory(editor.state.tr));
  editor.view.dispatch(editor.state.tr.setDocAttribute('subtitleEdits', edits));
  editor.view.dispatch(closeHistory(editor.state.tr));
}
export function applySubtitleAction(editor: Editor, ch: Chapter, action: LayoutEdit['action']) {
  const state = subtitleSelection(editor, ch), cue = state.cue;
  if (!editor.state.selection.empty) throw Error('请先在一条字幕内单击放置光标。');
  if (!cue) throw Error('当前没有可处理的字幕。');
  let at: number, scope_start = cue.at, scope_end = cue.end_at, before: string, after: string;
  if (action === 'split') {
    if (!state.split || !state.unit) throw Error('请把光标放在字幕内部的原生字词边界；修正文字内部或时间不足处暂不分句。');
    at = state.unit.at;
    const offset = cue.units.filter(u => u.pos < editor.state.selection.from).reduce((n, u) => n + u.text.length, 0);
    const split = splitSubtitleText(cue.text, offset); before = cue.text; after = `${split.leftText} ｜ ${split.rightText}`;
  } else {
    const neighbor = state.cues[state.index + (action === 'merge_previous' ? -1 : 1)];
    if (!neighbor) throw Error(action === 'merge_previous' ? '已经是第一条字幕。' : '已经是最后一条字幕。');
    const ordered = action === 'merge_previous' ? [neighbor, cue] : [cue, neighbor];
    scope_start = ordered[0].at; scope_end = ordered[1].end_at; at = ordered[1].at;
    before = ordered.map(c => c.text).join(' ｜ '); after = mergeSubtitleCues(ordered).text;
  }
  const op: LayoutEdit = { id: `subtitle:${crypto.randomUUID()}`, action, at, scope_start, scope_end, before, after };
  setEdits(editor, [...layoutEdits(editor), op]); return op;
}
export function removeSubtitleAction(editor: Editor, id: string) { setEdits(editor, layoutEdits(editor).filter(op => op.id !== id)); }
export function subtitleLayoutGroups(editor: Editor, ch: Chapter) {
  const sources = sourceOffsets(ch), units = subtitleUnits(editor, ch);
  return layoutEdits(editor).map(op => {
    const nearby = units.find(u => u.at >= op.at) || units.at(-1);
    const anchors = sources.filter(s => s.start < op.scope_end && s.end > op.scope_start).map(s => {
      const start = Math.max(0, op.scope_start - s.start), end = Math.min(s.segment.text.length, op.scope_end - s.start);
      return { source_id: `${ch.number}:${s.segment.id}`, chapter: ch.number, file: ch.file, segment_id: s.segment.id, start_char: start, end_char: end,
        original_text: s.segment.text.slice(start, end), envelope_start_s: s.segment.start, envelope_end_s: s.segment.end, timing_status: 'subtitle_layout_only' };
    });
    return { id: op.id, changes: [] as any[], from: nearby?.pos || 1, to: nearby?.pos || 1, deleted: '', inserted: '', kind: 'subtitle_structure', anchors, layout: op };
  });
}
