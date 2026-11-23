// Orchestration over the unmodified Tiptap revision engine and Python's
// canonical ASR punctuation plan. No source text is accepted away or rewritten.
import { closeHistory } from '@tiptap/pm/history';
import { trackedDelete, groupedReview, ensureMetadata, getBaseText, EFFECTS, type Chapter, type RevisionMeta } from './model';
import type { Editor } from '@tiptap/core';

export type PunctuationPlan = { chapter: string; source_hash: string; original_flat: string; removals: { start_char: number; end_char: number; text: string }[]; protected_user_characters_skipped: number };
export function visibleTextIndex(editor: Editor) {
  let text = ''; const units: { pos: number; protected: boolean }[] = [];
  editor.state.doc.descendants((node, pos) => {
    if (!node.isText || node.marks.some(m => m.type.name === 'deletion')) return;
    const protectedText = node.marks.some(m => m.type.name === 'insertion');
    for (let i = 0; i < node.text!.length; i++) units.push({ pos: pos + i, protected: protectedText });
    text += node.text;
  });
  return { text, units };
}
export function applyPunctuationPlan(editor: Editor, chapter: Chapter, meta: Record<string, RevisionMeta>, plan: PunctuationPlan) {
  const index = visibleTextIndex(editor), original = getBaseText(editor);
  if (chapter.number !== plan.chapter || chapter.source_text_hash !== plan.source_hash || index.text !== plan.original_flat) throw Error('标点清理前的文稿已变化，停止应用。');
  const author = { ...editor.storage.trackChanges.author };
  const existing = new Set(groupedReview(editor, chapter).map(g => g.id));
  const ranges: { from: number; to: number; text: string }[] = [];
  for (const removal of plan.removals) {
    if (removal.end_char !== removal.start_char + 1 || index.text.slice(removal.start_char, removal.end_char) !== removal.text || !/^[，。！？、；：,!?;]$/u.test(removal.text)) throw Error('清理计划包含非标点改动。');
    const unit = index.units[removal.start_char];
    if (!unit || unit.protected) throw Error('清理计划触及已有人工修订，停止应用。');
    const previous = ranges.at(-1);
    if (previous && previous.to === unit.pos) { previous.to++; previous.text += removal.text; }
    else ranges.push({ from: unit.pos, to: unit.pos + 1, text: removal.text });
  }
  editor.view.dispatch(closeHistory(editor.state.tr));
  try {
    for (const [i, range] of [...ranges].reverse().entries()) {
      editor.commands.setTrackChangesAuthor({ id: `punctuation-cleanup:${chapter.number}:${Date.now()}:${i}`, name: '标点清理', color: '#22634d' });
      editor.commands.setTextSelection({ from: range.from, to: range.to });
      if (!trackedDelete(editor)) throw Error('上游修订引擎未能应用标点修改。');
      const groups = ensureMetadata(editor, chapter, meta);
      for (const g of groups) {
        if (existing.has(g.id) || meta[g.id].origin === 'punctuation_cleanup') continue;
        Object.assign(meta[g.id], { effect: 'transcript_only', decision: EFFECTS.transcript_only, intent: '清理重复、多余的机器标点',
          reason: '原识别标点叠加或跨段拼接造成重复；仅修字幕，不剪原声。',
          evidence: `${chapter.file}；原文“${g.deleted}”；规则 deduplicate_asr_punctuation.v1；原文哈希 ${chapter.source_text_hash}；已有人工修改未覆盖。`,
          origin: 'punctuation_cleanup', status: 'proposed' });
      }
    }
  } finally { editor.commands.setTrackChangesAuthor(author); editor.view.dispatch(closeHistory(editor.state.tr)); }
  if (getBaseText(editor) !== original) throw Error('原始识别文本发生变化，停止应用。');
  return groupedReview(editor, chapter).filter(g => !existing.has(g.id)).map(g => g.id);
}
