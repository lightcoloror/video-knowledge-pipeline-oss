/*
 * Public-source curation marker, 2026-11-10.
 * Existing local VKP adaptation is retained below; this marker does not modify,
 * replace or re-date that adaptation.
 * Upstream base: Moyf/moys-asr-workflow,
 * commit 949bc84058cdae1d9c021c50203e6d2742f9392c.
 * Original upstream copyright and license text (vendor/mawe/LICENSE) remain
 * retained.
 * No new authorship is claimed. The date above is the public-source curation
 * date only; it is not an inferred historical modification date.
 */

// Adapted from Moyf/moys-asr-workflow 949bc84058cdae1d9c021c50203e6d2742f9392c:
// web/editor-utils.js joinSegmentTexts; web/editor.js mergeContiguousIndices,
// splitAtCursor. License is adjacent. VKP preserves punctuation and omits the
// upstream proportional-time fallback; immutable source cues are never spliced.
export function joinSegmentTexts(segments: { text: string }[], separator = '') {
  return segments.map(segment => String(segment?.text || '')).join(separator);
}
export function mergeSubtitleCues<T extends { start: number; end: number; text: string }>(segments: T[]) {
  return { start: segments[0].start, end: segments[segments.length - 1].end, text: joinSegmentTexts(segments) };
}
export function splitSubtitleText(text: string, cursorOffset: number) {
  if (cursorOffset <= 0 || cursorOffset >= text.length) throw Error('光标必须在当前字幕内部。');
  const leftText = text.slice(0, cursorOffset), rightText = text.slice(cursorOffset);
  if (!/[\p{L}\p{N}\p{S}]/u.test(leftText) || !/[\p{L}\p{N}\p{S}]/u.test(rightText)) throw Error('分句两侧都需要保留正文。');
  return { leftText, rightText };
}
