// punctuation-cleanup-cli.ts
import { readFileSync, writeFileSync } from "node:fs";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";

// model.ts
import { Editor, Mark as Mark4, mergeAttributes as mergeAttributes4 } from "@tiptap/core";
import Document from "@tiptap/extension-document";
import Paragraph from "@tiptap/extension-paragraph";
import Text from "@tiptap/extension-text";
import History from "@tiptap/extension-history";

// vendor/tiptap-track-changes/src/extension.ts
import { Extension } from "@tiptap/core";

// vendor/tiptap-track-changes/src/marks/insertion.ts
import { Mark, mergeAttributes } from "@tiptap/core";

// vendor/tiptap-track-changes/src/utils.ts
function sanitizeCSSValue(value) {
  if (!value || typeof value !== "string") return null;
  const sanitized = value.replace(/[;{}()<>]/g, "").trim();
  return sanitized || null;
}
var counter = 0;
function generateChangeId() {
  counter++;
  return `ch_${Date.now()}_${counter}_${Math.random().toString(36).slice(2, 8)}`;
}
var _segmenter = null;
function getSegmenter() {
  if (_segmenter) return _segmenter;
  if (typeof Intl !== "undefined" && typeof Intl.Segmenter === "function") {
    _segmenter = new Intl.Segmenter(void 0, { granularity: "grapheme" });
    return _segmenter;
  }
  return null;
}
function lastGraphemeClusterLength(text) {
  if (text.length === 0) return 0;
  const segmenter = getSegmenter();
  if (segmenter) {
    let last;
    for (const seg of segmenter.segment(text)) {
      last = seg.segment;
    }
    if (last !== void 0) {
      return last.length;
    }
  }
  return lastClusterLengthFallback(text);
}
function firstGraphemeClusterLength(text) {
  if (text.length === 0) return 0;
  const segmenter = getSegmenter();
  if (segmenter) {
    for (const seg of segmenter.segment(text)) {
      return seg.segment.length;
    }
  }
  return firstClusterLengthFallback(text);
}
function isCombiningMark(code) {
  return code >= 768 && code <= 879 || // Combining Diacritical Marks
  code >= 1155 && code <= 1161 || // Cyrillic combining marks
  code >= 1425 && code <= 1469 || // Hebrew points
  code >= 1471 && code <= 1479 || // Hebrew points (continued)
  code >= 1552 && code <= 1562 || // Arabic combining marks
  code >= 1611 && code <= 1631 || // Arabic tashkeel
  code === 1648 || // Arabic superscript alef
  code >= 1750 && code <= 1773 || // Arabic marks
  code === 1809 || // Syriac
  code >= 1840 && code <= 1866 || // Syriac marks
  code >= 1958 && code <= 1968 || // Thaana
  code >= 2259 && code <= 2303 || // Arabic Extended-A combining marks
  code >= 2304 && code <= 2307 || // Devanagari
  code >= 2362 && code <= 2383 || // Devanagari vowel signs + virama
  code >= 2385 && code <= 2391 || // Devanagari stress marks
  code >= 2402 && code <= 2403 || // Devanagari vowel signs
  code >= 2433 && code <= 2435 || // Bengali
  code >= 2492 && code <= 2509 || // Bengali signs + virama
  code >= 2530 && code <= 2531 || // Bengali vowel signs
  code >= 2561 && code <= 2563 || // Gurmukhi
  code >= 2620 && code <= 2637 || // Gurmukhi signs
  code >= 2689 && code <= 2691 || // Gujarati
  code >= 2748 && code <= 2765 || // Gujarati signs
  code >= 2817 && code <= 2819 || // Oriya
  code >= 2876 && code <= 2893 || // Oriya signs
  code === 2946 || // Tamil
  code >= 3006 && code <= 3021 || // Tamil vowel signs + virama
  code >= 3072 && code <= 3076 || // Telugu
  code >= 3134 && code <= 3149 || // Telugu signs
  code >= 3201 && code <= 3203 || // Kannada
  code >= 3260 && code <= 3277 || // Kannada signs
  code >= 3328 && code <= 3331 || // Malayalam
  code >= 3387 && code <= 3405 || // Malayalam signs
  code === 3633 || // Thai sara am/combining
  code >= 3636 && code <= 3642 || // Thai combining vowels/marks
  code >= 3655 && code <= 3662 || // Thai tone marks
  code === 3761 || // Lao combining
  code >= 3764 && code <= 3772 || // Lao combining vowels
  code >= 3784 && code <= 3789 || // Lao tone marks
  code >= 3864 && code <= 3865 || // Tibetan
  code >= 3893 && code <= 3897 || // Tibetan marks
  code >= 3953 && code <= 3972 || // Tibetan vowel signs
  code >= 3974 && code <= 3975 || // Tibetan marks
  code >= 3981 && code <= 4028 || // Tibetan subjoined consonants
  code >= 4139 && code <= 4158 || // Myanmar vowel signs + medials
  code >= 4182 && code <= 4185 || // Myanmar extensions
  code >= 4190 && code <= 4192 || // Myanmar extensions
  code >= 4194 && code <= 4196 || // Myanmar extensions
  code >= 4199 && code <= 4205 || // Myanmar extensions
  code >= 4209 && code <= 4212 || // Myanmar extensions
  code >= 4226 && code <= 4237 || // Myanmar extensions
  code >= 4957 && code <= 4959 || // Ethiopic combining
  code >= 6068 && code <= 6099 || // Khmer vowel signs + coeng
  code === 6109 || // Khmer sign atthacan
  code >= 6432 && code <= 6443 || // Limbu
  code >= 6448 && code <= 6459 || // Limbu
  code >= 6832 && code <= 6911 || // Combining Diacritical Marks Extended
  code >= 7616 && code <= 7679 || // Combining Diacritical Marks Supplement
  code === 8204 || // Zero Width Non-Joiner (ZWNJ)
  code === 8205 || // Zero Width Joiner (ZWJ)
  code >= 8400 && code <= 8447 || // Combining marks for symbols
  code >= 11744 && code <= 11775 || // Cyrillic Extended-B combining
  code >= 42607 && code <= 42610 || // Cyrillic Extended-B combining marks
  code >= 43232 && code <= 43255 || // Devanagari Extended combining marks
  code >= 65056 && code <= 65071;
}
function lastClusterLengthFallback(text) {
  let len = 0;
  let i = text.length;
  while (i > 0) {
    const code = text.charCodeAt(i - 1);
    if (code >= 56320 && code <= 57343 && i >= 2) {
      const high = text.charCodeAt(i - 2);
      if (high >= 55296 && high <= 56319) {
        const cp = (high - 55296) * 1024 + (code - 56320) + 65536;
        if (len === 0) {
          return 2;
        }
        if (!isCombiningMark(cp)) {
          len += 2;
          break;
        }
        len += 2;
        i -= 2;
        continue;
      }
    }
    if (len > 0 && !isCombiningMark(code)) {
      break;
    }
    len += 1;
    i -= 1;
    if (!isCombiningMark(code)) {
      break;
    }
  }
  return Math.max(len, 1);
}
function firstClusterLengthFallback(text) {
  let len = 0;
  let i = 0;
  const firstCode = text.codePointAt(0);
  if (firstCode === void 0) return 1;
  if (firstCode > 65535) {
    len = 2;
    i = 2;
  } else {
    len = 1;
    i = 1;
  }
  while (i < text.length) {
    const cp = text.codePointAt(i);
    if (cp === void 0) break;
    const charLen = cp > 65535 ? 2 : 1;
    if (!isCombiningMark(cp)) break;
    len += charLen;
    i += charLen;
  }
  return len;
}

// vendor/tiptap-track-changes/src/marks/insertion.ts
var InsertionMark = Mark.create({
  name: "insertion",
  addOptions() {
    return {
      HTMLAttributes: {}
    };
  },
  addAttributes() {
    return {
      changeId: { default: null },
      authorId: { default: null },
      authorName: { default: null },
      authorColor: { default: null },
      timestamp: { default: null }
    };
  },
  inclusive: false,
  parseHTML() {
    return [{ tag: "ins[data-change-id]" }];
  },
  renderHTML({ HTMLAttributes }) {
    const safeColor = sanitizeCSSValue(HTMLAttributes.authorColor);
    return [
      "ins",
      mergeAttributes(this.options.HTMLAttributes, {
        "data-change-id": HTMLAttributes.changeId,
        "data-author-id": HTMLAttributes.authorId,
        "data-author-name": HTMLAttributes.authorName,
        ...safeColor ? { style: `--author-color: ${safeColor};` } : {}
      }),
      0
    ];
  }
});

// vendor/tiptap-track-changes/src/marks/deletion.ts
import { Mark as Mark2, mergeAttributes as mergeAttributes2 } from "@tiptap/core";
var DeletionMark = Mark2.create({
  name: "deletion",
  addOptions() {
    return {
      HTMLAttributes: {}
    };
  },
  addAttributes() {
    return {
      changeId: { default: null },
      authorId: { default: null },
      authorName: { default: null },
      authorColor: { default: null },
      timestamp: { default: null }
    };
  },
  inclusive: false,
  parseHTML() {
    return [{ tag: "del[data-change-id]" }];
  },
  renderHTML({ HTMLAttributes }) {
    const safeColor = sanitizeCSSValue(HTMLAttributes.authorColor);
    return [
      "del",
      mergeAttributes2(this.options.HTMLAttributes, {
        "data-change-id": HTMLAttributes.changeId,
        "data-author-id": HTMLAttributes.authorId,
        "data-author-name": HTMLAttributes.authorName,
        ...safeColor ? { style: `--author-color: ${safeColor};` } : {}
      }),
      0
    ];
  }
});

// vendor/tiptap-track-changes/src/marks/format-change.ts
import { Mark as Mark3, mergeAttributes as mergeAttributes3 } from "@tiptap/core";
var FormatChangeMark = Mark3.create({
  name: "formatChange",
  // Allow multiple formatChange marks on the same text (e.g., bold added + italic removed)
  excludes: "",
  addOptions() {
    return {
      HTMLAttributes: {}
    };
  },
  addAttributes() {
    return {
      changeId: { default: null },
      authorId: { default: null },
      authorName: { default: null },
      authorColor: { default: null },
      timestamp: { default: null },
      formatAdded: { default: null },
      formatRemoved: { default: null }
    };
  },
  inclusive: false,
  parseHTML() {
    return [{ tag: "span[data-format-change]" }];
  },
  renderHTML({ HTMLAttributes }) {
    const title = [
      HTMLAttributes.formatAdded ? `Added: ${HTMLAttributes.formatAdded}` : "",
      HTMLAttributes.formatRemoved ? `Removed: ${HTMLAttributes.formatRemoved}` : ""
    ].filter(Boolean).join(", ");
    const safeColor = sanitizeCSSValue(HTMLAttributes.authorColor);
    return [
      "span",
      mergeAttributes3(this.options.HTMLAttributes, {
        "data-format-change": "true",
        "data-change-id": HTMLAttributes.changeId,
        "data-author-id": HTMLAttributes.authorId,
        "data-format-added": HTMLAttributes.formatAdded,
        "data-format-removed": HTMLAttributes.formatRemoved,
        title,
        ...safeColor ? { style: `border-bottom: 2px dotted ${safeColor};` } : {}
      }),
      0
    ];
  }
});

// vendor/tiptap-track-changes/src/suggest-mode-plugin.ts
import { Plugin, PluginKey, TextSelection } from "@tiptap/pm/state";
import { AddMarkStep, RemoveMarkStep } from "@tiptap/pm/transform";
var _isHistoryTransaction;
try {
  const historyModule = Function('return typeof require !== "undefined" ? require : undefined')()?.("@tiptap/pm/history");
  if (historyModule?.isHistoryTransaction) {
    _isHistoryTransaction = historyModule.isHistoryTransaction;
  }
} catch {
}
function isHistoryTransaction(tr) {
  if (_isHistoryTransaction) return _isHistoryTransaction(tr);
  return false;
}
function createNodeTracking(author, type, changeId) {
  return {
    changeId: changeId ?? generateChangeId(),
    authorId: author.id,
    authorName: author.name,
    authorColor: author.color,
    timestamp: (/* @__PURE__ */ new Date()).toISOString(),
    originalType: type
  };
}
var trackChangesPluginKey = new PluginKey("trackChanges");
function createChangeAttrs(author, changeId) {
  return {
    changeId: changeId ?? generateChangeId(),
    authorId: author.id,
    authorName: author.name,
    authorColor: author.color,
    timestamp: (/* @__PURE__ */ new Date()).toISOString()
  };
}
function markRangeAsDeleted(doc, tr, from, to, deletionMark) {
  doc.nodesBetween(from, to, (node, pos) => {
    if (!node.isText) return;
    const nodeFrom = Math.max(from, pos);
    const nodeTo = Math.min(to, pos + node.nodeSize);
    const hasDeletion = node.marks.some((m) => m.type.name === "deletion");
    if (!hasDeletion) {
      tr.addMark(nodeFrom, nodeTo, deletionMark);
    }
  });
}
function getAdjacentInsertionMark(view, pos, authorId) {
  if (pos <= 0) return null;
  const $pos = view.state.doc.resolve(pos);
  const nodeBefore = $pos.nodeBefore;
  if (nodeBefore && nodeBefore.isText) {
    const insertionMark = nodeBefore.marks.find(
      (m) => m.type.name === "insertion" && m.attrs.authorId === authorId
    );
    if (insertionMark) {
      return insertionMark;
    }
  }
  return null;
}
function handleTextInsert(view, from, to, text, author) {
  const { state } = view;
  const schema = state.schema;
  const tr = state.tr;
  const normalizedText = text.normalize("NFC");
  if (from !== to) {
    const changeId = generateChangeId();
    const deletionMark = schema.marks.deletion.create(
      createChangeAttrs(author, changeId)
    );
    const insertionMark = schema.marks.insertion.create(
      createChangeAttrs(author, changeId)
    );
    markRangeAsDeleted(state.doc, tr, from, to, deletionMark);
    tr.insertText(normalizedText, to);
    tr.addMark(to, to + normalizedText.length, insertionMark);
    tr.setSelection(TextSelection.create(tr.doc, to + normalizedText.length));
  } else {
    const existingMark = getAdjacentInsertionMark(view, from, author.id);
    const insertionMark = existingMark ?? schema.marks.insertion.create(
      createChangeAttrs(author)
    );
    tr.insertText(normalizedText, from);
    tr.addMark(from, from + normalizedText.length, insertionMark);
    tr.setSelection(TextSelection.create(tr.doc, from + normalizedText.length));
  }
  tr.setMeta(trackChangesPluginKey, { handled: true });
  view.dispatch(tr);
  return true;
}
function handleBackspace(view, author) {
  const { state } = view;
  const { selection } = state;
  if (!selection.empty) {
    return handleSelectionDelete(view, author);
  }
  const $from = selection.$from;
  if ($from.parentOffset === 0) {
    return handleBlockJoinBackward(view, author);
  }
  const textBefore = $from.parent.textBetween(0, $from.parentOffset);
  const clusterLen = lastGraphemeClusterLength(textBefore);
  const deleteFrom = $from.pos - clusterLen;
  const deleteTo = $from.pos;
  return applyDeletion(view, deleteFrom, deleteTo, author);
}
function handleDelete(view, author) {
  const { state } = view;
  const { selection } = state;
  if (!selection.empty) {
    return handleSelectionDelete(view, author);
  }
  const $to = selection.$to;
  const parentSize = $to.parent.content.size;
  if ($to.parentOffset >= parentSize) {
    return handleBlockJoinForward(view, author);
  }
  const textAfter = $to.parent.textBetween(
    $to.parentOffset,
    parentSize
  );
  const clusterLen = firstGraphemeClusterLength(textAfter);
  const deleteFrom = $to.pos;
  const deleteTo = $to.pos + clusterLen;
  return applyDeletion(view, deleteFrom, deleteTo, author, "forward");
}
function handleBlockJoinBackward(view, author) {
  const { state } = view;
  const { selection } = state;
  const $from = selection.$from;
  const blockPos = $from.before($from.depth);
  if (blockPos <= 0) return false;
  const node = state.doc.nodeAt(blockPos);
  if (!node) return false;
  if (node.attrs.dataTracked) return true;
  const tr = state.tr;
  const tracking = createNodeTracking(author, "boundaryDeleted");
  tr.setNodeMarkup(blockPos, void 0, { ...node.attrs, dataTracked: tracking });
  tr.setMeta(trackChangesPluginKey, { handled: true });
  view.dispatch(tr);
  return true;
}
function handleBlockJoinForward(view, author) {
  const { state } = view;
  const { selection } = state;
  const $to = selection.$to;
  const afterPos = $to.after($to.depth);
  if (afterPos >= state.doc.content.size) return false;
  const nextNode = state.doc.nodeAt(afterPos);
  if (!nextNode || !nextNode.isBlock) return false;
  if (nextNode.attrs.dataTracked) return true;
  const tr = state.tr;
  const tracking = createNodeTracking(author, "boundaryDeleted");
  tr.setNodeMarkup(afterPos, void 0, { ...nextNode.attrs, dataTracked: tracking });
  tr.setMeta(trackChangesPluginKey, { handled: true });
  view.dispatch(tr);
  return true;
}
function handleEnter(view, author) {
  const { state } = view;
  const { selection } = state;
  const tr = state.tr;
  const schema = state.schema;
  if (!selection.empty) {
    const { from, to } = selection;
    const changeId = generateChangeId();
    const deletionMark = schema.marks.deletion.create(
      createChangeAttrs(author, changeId)
    );
    let hasOwnInsertions = false;
    state.doc.nodesBetween(from, to, (node) => {
      if (node.isText) {
        const ownInsertion = node.marks.find(
          (m) => m.type.name === "insertion" && m.attrs.authorId === author.id
        );
        if (ownInsertion) hasOwnInsertions = true;
      }
    });
    if (hasOwnInsertions) {
      const ranges = [];
      state.doc.nodesBetween(from, to, (node, pos) => {
        if (!node.isText) return;
        const nodeFrom = Math.max(from, pos);
        const nodeTo = Math.min(to, pos + node.nodeSize);
        const ownInsertion = node.marks.find(
          (m) => m.type.name === "insertion" && m.attrs.authorId === author.id
        );
        const hasDeletion = node.marks.some((m) => m.type.name === "deletion");
        ranges.push({ from: nodeFrom, to: nodeTo, isOwnInsertion: !!ownInsertion, isAlreadyDeleted: hasDeletion });
      });
      for (let i = ranges.length - 1; i >= 0; i--) {
        const range = ranges[i];
        if (range.isOwnInsertion) {
          tr.delete(range.from, range.to);
        } else if (!range.isAlreadyDeleted) {
          tr.addMark(range.from, range.to, deletionMark);
        }
      }
    } else {
      markRangeAsDeleted(state.doc, tr, from, to, deletionMark);
    }
    const splitPos = tr.mapping.map(from);
    tr.split(splitPos);
    const newParaPos = splitPos + 1;
    const newNode = tr.doc.nodeAt(newParaPos);
    if (newNode) {
      const tracking = createNodeTracking(author, "paragraphInserted");
      tr.setNodeMarkup(newParaPos, void 0, { ...newNode.attrs, dataTracked: tracking });
    }
  } else {
    const pos = selection.from;
    const $pos = state.doc.resolve(pos);
    if ($pos.parentOffset === 0) {
      tr.split(pos);
      const $splitPos = tr.doc.resolve(pos);
      const emptyParaPos = $splitPos.before($splitPos.depth);
      const emptyNode = tr.doc.nodeAt(emptyParaPos);
      if (emptyNode && !emptyNode.attrs.dataTracked) {
        const tracking = createNodeTracking(author, "paragraphInserted");
        tr.setNodeMarkup(emptyParaPos, void 0, { ...emptyNode.attrs, dataTracked: tracking });
      }
    } else {
      tr.split(pos);
      const newParaPos = pos + 1;
      const newNode = tr.doc.nodeAt(newParaPos);
      if (newNode) {
        const tracking = createNodeTracking(author, "paragraphInserted");
        tr.setNodeMarkup(newParaPos, void 0, { ...newNode.attrs, dataTracked: tracking });
      }
    }
  }
  try {
    const newCursorPos = selection.empty ? selection.from + 2 : tr.mapping.map(selection.from) + 2;
    tr.setSelection(TextSelection.create(tr.doc, newCursorPos));
  } catch {
  }
  tr.setMeta(trackChangesPluginKey, { handled: true });
  view.dispatch(tr);
  return true;
}
function handleSelectionDelete(view, author) {
  const { state } = view;
  const { from, to } = state.selection;
  const tr = state.tr;
  const deletionMark = state.schema.marks.deletion.create(
    createChangeAttrs(author)
  );
  let hasOwnInsertions = false;
  state.doc.nodesBetween(from, to, (node, pos) => {
    if (node.isText) {
      const ownInsertion = node.marks.find(
        (m) => m.type.name === "insertion" && m.attrs.authorId === author.id
      );
      if (ownInsertion) {
        hasOwnInsertions = true;
      }
    }
  });
  if (hasOwnInsertions) {
    const ranges = [];
    state.doc.nodesBetween(from, to, (node, pos) => {
      if (!node.isText) return;
      const nodeFrom = Math.max(from, pos);
      const nodeTo = Math.min(to, pos + node.nodeSize);
      const ownInsertion = node.marks.find(
        (m) => m.type.name === "insertion" && m.attrs.authorId === author.id
      );
      const hasDeletion = node.marks.some((m) => m.type.name === "deletion");
      ranges.push({
        from: nodeFrom,
        to: nodeTo,
        isOwnInsertion: !!ownInsertion,
        isAlreadyDeleted: hasDeletion
      });
    });
    for (let i = ranges.length - 1; i >= 0; i--) {
      const range = ranges[i];
      if (range.isOwnInsertion) {
        tr.delete(range.from, range.to);
      } else if (!range.isAlreadyDeleted) {
        tr.addMark(range.from, range.to, deletionMark);
      }
    }
  } else {
    markRangeAsDeleted(state.doc, tr, from, to, deletionMark);
  }
  const $selFrom = state.doc.resolve(from);
  const $selTo = state.doc.resolve(to);
  if ($selFrom.depth > 0 && $selTo.depth > 0) {
    const startBlockEnd = $selFrom.end($selFrom.depth);
    if (to > startBlockEnd + 1) {
      state.doc.nodesBetween(from, to, (node, pos) => {
        if (!node.isBlock || !node.isTextblock) return;
        if (pos <= $selFrom.before($selFrom.depth)) return;
        if (!node.attrs.dataTracked) {
          const mappedPos = tr.mapping.map(pos);
          const mappedNode = tr.doc.nodeAt(mappedPos);
          if (mappedNode) {
            const tracking = createNodeTracking(author, "boundaryDeleted");
            tr.setNodeMarkup(mappedPos, void 0, { ...mappedNode.attrs, dataTracked: tracking });
          }
        }
      });
    }
  }
  tr.setSelection(TextSelection.create(tr.doc, tr.mapping.map(from)));
  tr.setMeta(trackChangesPluginKey, { handled: true });
  view.dispatch(tr);
  return true;
}
function handleMultiLinePaste(view, from, to, text, author) {
  const { state } = view;
  const schema = state.schema;
  const tr = state.tr;
  const lines = text.split("\n");
  if (from !== to) {
    const changeId = generateChangeId();
    const deletionMark = schema.marks.deletion.create(
      createChangeAttrs(author, changeId)
    );
    markRangeAsDeleted(state.doc, tr, from, to, deletionMark);
  }
  const insertPos = to;
  const firstLine = lines[0].normalize("NFC");
  if (firstLine) {
    const lineMark = schema.marks.insertion.create(createChangeAttrs(author));
    tr.insertText(firstLine, insertPos);
    tr.addMark(insertPos, insertPos + firstLine.length, lineMark);
  }
  let cursor = insertPos + firstLine.length;
  for (let i = 1; i < lines.length; i++) {
    tr.split(cursor);
    const newParaPos = cursor + 1;
    const newNode = tr.doc.nodeAt(newParaPos);
    if (newNode) {
      const tracking = {
        changeId: generateChangeId(),
        authorId: author.id,
        authorName: author.name,
        authorColor: author.color,
        timestamp: (/* @__PURE__ */ new Date()).toISOString(),
        originalType: "paragraphInserted"
      };
      tr.setNodeMarkup(newParaPos, void 0, { ...newNode.attrs, dataTracked: tracking });
    }
    const line = lines[i].normalize("NFC");
    const textPos = newParaPos + 1;
    if (line) {
      const lineMark = schema.marks.insertion.create(createChangeAttrs(author));
      tr.insertText(line, textPos);
      tr.addMark(textPos, textPos + line.length, lineMark);
      cursor = textPos + line.length;
    } else {
      cursor = textPos;
    }
  }
  try {
    tr.setSelection(TextSelection.create(tr.doc, cursor));
  } catch {
  }
  tr.setMeta(trackChangesPluginKey, { handled: true });
  view.dispatch(tr);
  return true;
}
function applyDeletion(view, deleteFrom, deleteTo, author, direction = "backward") {
  const { state } = view;
  const tr = state.tr;
  let isOwnInsertion = false;
  let isAlreadyDeleted = false;
  state.doc.nodesBetween(deleteFrom, deleteTo, (node) => {
    if (node.isText) {
      const ownInsertion = node.marks.find(
        (m) => m.type.name === "insertion" && m.attrs.authorId === author.id
      );
      if (ownInsertion) {
        isOwnInsertion = true;
      }
      const deletion = node.marks.find((m) => m.type.name === "deletion");
      if (deletion) {
        isAlreadyDeleted = true;
      }
    }
  });
  if (isOwnInsertion) {
    const ranges = [];
    state.doc.nodesBetween(deleteFrom, deleteTo, (node, pos) => {
      if (!node.isText) return;
      const nodeFrom = Math.max(deleteFrom, pos);
      const nodeTo = Math.min(deleteTo, pos + node.nodeSize);
      const ownIns = node.marks.find(
        (m) => m.type.name === "insertion" && m.attrs.authorId === author.id
      );
      const hasDeletion = node.marks.some((m) => m.type.name === "deletion");
      ranges.push({ from: nodeFrom, to: nodeTo, isOwn: !!ownIns, isAlreadyDeleted: hasDeletion });
    });
    for (let i = ranges.length - 1; i >= 0; i--) {
      const range = ranges[i];
      if (range.isOwn) {
        tr.delete(range.from, range.to);
      } else if (!range.isAlreadyDeleted) {
        const deletionMark2 = state.schema.marks.deletion.create(
          createChangeAttrs(author)
        );
        tr.addMark(range.from, range.to, deletionMark2);
      }
    }
    tr.setMeta(trackChangesPluginKey, { handled: true });
    view.dispatch(tr);
    return true;
  }
  if (isAlreadyDeleted) {
    const skipPos = findDeletionBoundary(
      view,
      direction === "forward" ? deleteTo : deleteFrom,
      direction
    );
    tr.setSelection(TextSelection.create(tr.doc, skipPos));
    view.dispatch(tr);
    return true;
  }
  const deletionMark = state.schema.marks.deletion.create(
    createChangeAttrs(author)
  );
  tr.addMark(deleteFrom, deleteTo, deletionMark);
  tr.setSelection(TextSelection.create(tr.doc, deleteFrom));
  tr.setMeta(trackChangesPluginKey, { handled: true });
  view.dispatch(tr);
  return true;
}
function findDeletionBoundary(view, pos, direction) {
  const { doc } = view.state;
  let currentPos = pos;
  if (direction === "backward") {
    while (currentPos > 0) {
      const $pos = doc.resolve(currentPos);
      const nodeBefore = $pos.nodeBefore;
      if (!nodeBefore || !nodeBefore.isText) break;
      const hasDeletion = nodeBefore.marks.some(
        (m) => m.type.name === "deletion"
      );
      if (!hasDeletion) break;
      currentPos -= nodeBefore.nodeSize;
    }
  } else {
    const docSize = doc.content.size;
    while (currentPos < docSize) {
      const $pos = doc.resolve(currentPos);
      const nodeAfter = $pos.nodeAfter;
      if (!nodeAfter || !nodeAfter.isText) break;
      const hasDeletion = nodeAfter.marks.some(
        (m) => m.type.name === "deletion"
      );
      if (!hasDeletion) break;
      currentPos += nodeAfter.nodeSize;
    }
  }
  return currentPos;
}
function createSuggestModePlugin(options) {
  return new Plugin({
    key: trackChangesPluginKey,
    props: {
      editable: () => options.getMode() !== "view",
      handleTextInput(view, from, to, text) {
        if (options.getMode() !== "suggest") return false;
        if (view.composing) return false;
        return handleTextInsert(view, from, to, text, options.getAuthor());
      },
      handleKeyDown(view, event) {
        if (options.getMode() !== "suggest") return false;
        if (view.composing) return false;
        if (event.key === "Backspace") {
          return handleBackspace(view, options.getAuthor());
        }
        if (event.key === "Delete") {
          return handleDelete(view, options.getAuthor());
        }
        if (event.key === "Enter" && !event.shiftKey) {
          return handleEnter(view, options.getAuthor());
        }
        return false;
      },
      handlePaste(view, event, _slice) {
        if (options.getMode() !== "suggest") return false;
        let text = event.clipboardData?.getData("text/plain");
        if (!text) {
          const html = event.clipboardData?.getData("text/html");
          if (html) {
            const parsed = new DOMParser().parseFromString(html, "text/html");
            text = parsed.body.textContent || "";
          }
        }
        if (!text) return false;
        const { from, to } = view.state.selection;
        if (text.includes("\n")) {
          return handleMultiLinePaste(view, from, to, text, options.getAuthor());
        }
        return handleTextInsert(view, from, to, text, options.getAuthor());
      }
    },
    // Safety net: catch text insertions that bypass handleTextInput
    // (e.g., IME composition for Khmer, Thai, CJK, etc.) and add insertion marks
    appendTransaction(transactions, _oldState, newState) {
      if (options.getMode() !== "suggest") return null;
      const isHandled = transactions.some((t) => t.getMeta(trackChangesPluginKey));
      if (isHandled) return null;
      const isHistoryTr = transactions.some((t) => isHistoryTransaction(t));
      if (isHistoryTr) return null;
      const trackMarkNames = /* @__PURE__ */ new Set(["insertion", "deletion", "formatChange"]);
      const formatSteps = [];
      for (const transaction of transactions) {
        for (const step of transaction.steps) {
          if (step instanceof AddMarkStep) {
            const markStep = step;
            if (!trackMarkNames.has(markStep.mark.type.name)) {
              formatSteps.push({
                from: markStep.from,
                to: markStep.to,
                formatAdded: markStep.mark.type.name,
                formatRemoved: null
              });
            }
          } else if (step instanceof RemoveMarkStep) {
            const markStep = step;
            if (!trackMarkNames.has(markStep.mark.type.name)) {
              formatSteps.push({
                from: markStep.from,
                to: markStep.to,
                formatAdded: null,
                formatRemoved: markStep.mark.type.name
              });
            }
          }
        }
      }
      if (formatSteps.length > 0) {
        const tr2 = newState.tr;
        const author2 = options.getAuthor();
        for (const fs of formatSteps) {
          const formatChangeMark = newState.schema.marks.formatChange.create({
            ...createChangeAttrs(author2),
            formatAdded: fs.formatAdded,
            formatRemoved: fs.formatRemoved
          });
          tr2.addMark(fs.from, fs.to, formatChangeMark);
        }
        tr2.setMeta(trackChangesPluginKey, { handled: true });
        return tr2;
      }
      const isUserInput = transactions.some((t) => t.getMeta("uiEvent") || t.getMeta("composition"));
      if (!isUserInput) return null;
      if (!transactions.some((t) => t.docChanged)) return null;
      const changedRanges = [];
      for (let i = 0; i < transactions.length; i++) {
        const transaction = transactions[i];
        if (!transaction.docChanged) continue;
        transaction.steps.forEach((step, stepIndex) => {
          step.getMap().forEach(
            (_oldStart, _oldEnd, newStart, newEnd) => {
              if (newEnd <= newStart) return;
              let from = newStart;
              let to = newEnd;
              for (let s = stepIndex + 1; s < transaction.steps.length; s++) {
                from = transaction.steps[s].getMap().map(from, 1);
                to = transaction.steps[s].getMap().map(to, -1);
              }
              for (let j = i + 1; j < transactions.length; j++) {
                from = transactions[j].mapping.map(from, 1);
                to = transactions[j].mapping.map(to, -1);
              }
              if (to > from) {
                changedRanges.push({ from, to });
              }
            }
          );
        });
      }
      if (changedRanges.length === 0) return null;
      const tr = newState.tr;
      let hasChanges = false;
      const author = options.getAuthor();
      for (const range of changedRanges) {
        const clampedTo = Math.min(range.to, newState.doc.content.size);
        if (range.from >= clampedTo) continue;
        newState.doc.nodesBetween(range.from, clampedTo, (node, pos) => {
          if (!node.isText) return;
          const hasTrackMark = node.marks.some(
            (m) => m.type.name === "insertion" || m.type.name === "deletion"
          );
          if (hasTrackMark) return;
          const nodeFrom = Math.max(range.from, pos);
          const nodeTo = Math.min(clampedTo, pos + node.nodeSize);
          if (nodeTo > nodeFrom) {
            let insertionMark = null;
            if (nodeFrom > 0) {
              const $pos = newState.doc.resolve(nodeFrom);
              const nodeBefore = $pos.nodeBefore;
              if (nodeBefore?.isText) {
                insertionMark = nodeBefore.marks.find(
                  (m) => m.type.name === "insertion" && m.attrs.authorId === author.id
                ) ?? null;
              }
            }
            if (!insertionMark) {
              insertionMark = newState.schema.marks.insertion.create(
                createChangeAttrs(author)
              );
            }
            tr.addMark(nodeFrom, nodeTo, insertionMark);
            hasChanges = true;
          }
        });
      }
      if (hasChanges) {
        tr.setMeta(trackChangesPluginKey, { handled: true });
        return tr;
      }
      return null;
    }
  });
}

// vendor/tiptap-track-changes/src/commands.ts
import { TextSelection as TextSelection2 } from "@tiptap/pm/state";
import { canJoin } from "@tiptap/pm/transform";
var trackChangesCommands = {
  setSuggestMode: () => ({ editor, tr, dispatch }) => {
    editor.storage.trackChanges.mode = "suggest";
    if (dispatch) {
      tr.setMeta("addToHistory", false);
      dispatch(tr);
    }
    return true;
  },
  setEditMode: () => ({ editor, tr, dispatch }) => {
    editor.storage.trackChanges.mode = "edit";
    if (dispatch) {
      tr.setMeta("addToHistory", false);
      dispatch(tr);
    }
    return true;
  },
  setViewMode: () => ({ editor, tr, dispatch }) => {
    editor.storage.trackChanges.mode = "view";
    if (dispatch) {
      tr.setMeta("addToHistory", false);
      dispatch(tr);
    }
    return true;
  },
  setTrackChangesMode: (mode) => ({ editor, tr, dispatch }) => {
    editor.storage.trackChanges.mode = mode;
    if (dispatch) {
      tr.setMeta("addToHistory", false);
      dispatch(tr);
    }
    return true;
  },
  setTrackChangesAuthor: (author) => ({ editor }) => {
    editor.storage.trackChanges.author = author;
    return true;
  },
  acceptChange: (changeId) => ({ state, dispatch, editor }) => {
    if (!dispatch) return true;
    const tr = state.tr;
    const deletionRanges = [];
    const insertionUnmarkRanges = [];
    const formatChangeRanges = [];
    const dataTrackedNodes = [];
    state.doc.descendants((node, pos) => {
      if (!node.isText) {
        if (node.isBlock && node.attrs.dataTracked) {
          const tracking = node.attrs.dataTracked;
          if (tracking.changeId === changeId) {
            dataTrackedNodes.push({ pos, tracking });
          }
        }
        return;
      }
      for (const mark of node.marks) {
        if (mark.attrs.changeId !== changeId) continue;
        const nodeFrom = pos;
        const nodeTo = pos + node.nodeSize;
        if (mark.type.name === "insertion") {
          insertionUnmarkRanges.push({ from: nodeFrom, to: nodeTo });
        } else if (mark.type.name === "deletion") {
          deletionRanges.push({ from: nodeFrom, to: nodeTo });
        } else if (mark.type.name === "formatChange") {
          formatChangeRanges.push({ from: nodeFrom, to: nodeTo });
        }
      }
    });
    const sortedDataTracked = dataTrackedNodes.sort((a, b) => b.pos - a.pos);
    for (const { pos, tracking } of sortedDataTracked) {
      const mappedPos = tr.mapping.map(pos);
      const node = tr.doc.nodeAt(mappedPos);
      if (!node) continue;
      if (tracking.originalType === "paragraphInserted") {
        tr.setNodeMarkup(mappedPos, void 0, { ...node.attrs, dataTracked: null });
      } else if (tracking.originalType === "boundaryDeleted") {
        tr.setNodeMarkup(mappedPos, void 0, { ...node.attrs, dataTracked: null });
        if (canJoin(tr.doc, mappedPos)) {
          tr.join(mappedPos);
        }
      } else {
        tr.setNodeMarkup(mappedPos, void 0, { ...node.attrs, dataTracked: null });
      }
    }
    for (const range of insertionUnmarkRanges) {
      const mappedFrom = tr.mapping.map(range.from);
      const mappedTo = tr.mapping.map(range.to);
      tr.removeMark(mappedFrom, mappedTo, state.schema.marks.insertion);
    }
    for (const range of formatChangeRanges) {
      const mappedFrom = tr.mapping.map(range.from);
      const mappedTo = tr.mapping.map(range.to);
      tr.removeMark(mappedFrom, mappedTo, state.schema.marks.formatChange);
    }
    const sortedDeletions = deletionRanges.sort((a, b) => b.from - a.from);
    for (const range of sortedDeletions) {
      const mappedFrom = tr.mapping.map(range.from);
      const mappedTo = tr.mapping.map(range.to);
      tr.delete(mappedFrom, mappedTo);
    }
    if (insertionUnmarkRanges.length > 0 || deletionRanges.length > 0 || formatChangeRanges.length > 0 || dataTrackedNodes.length > 0) {
      tr.setMeta(trackChangesPluginKey, { handled: true });
      dispatch(tr);
      editor.storage.trackChanges.onStatusChange?.(changeId, "accepted");
    }
    return true;
  },
  rejectChange: (changeId) => ({ state, dispatch, editor }) => {
    if (!dispatch) return true;
    const tr = state.tr;
    const insertionDeleteRanges = [];
    const deletionUnmarkRanges = [];
    const formatChangeRanges = [];
    const dataTrackedNodes = [];
    state.doc.descendants((node, pos) => {
      if (!node.isText) {
        if (node.isBlock && node.attrs.dataTracked) {
          const tracking = node.attrs.dataTracked;
          if (tracking.changeId === changeId) {
            dataTrackedNodes.push({ pos, tracking });
          }
        }
        return;
      }
      for (const mark of node.marks) {
        if (mark.attrs.changeId !== changeId) continue;
        const nodeFrom = pos;
        const nodeTo = pos + node.nodeSize;
        if (mark.type.name === "insertion") {
          insertionDeleteRanges.push({ from: nodeFrom, to: nodeTo });
        } else if (mark.type.name === "deletion") {
          deletionUnmarkRanges.push({ from: nodeFrom, to: nodeTo });
        } else if (mark.type.name === "formatChange") {
          formatChangeRanges.push({
            from: nodeFrom,
            to: nodeTo,
            formatAdded: mark.attrs.formatAdded,
            formatRemoved: mark.attrs.formatRemoved
          });
        }
      }
    });
    for (const range of deletionUnmarkRanges) {
      const mappedFrom = tr.mapping.map(range.from);
      const mappedTo = tr.mapping.map(range.to);
      tr.removeMark(mappedFrom, mappedTo, state.schema.marks.deletion);
    }
    const sortedDataTracked = dataTrackedNodes.sort((a, b) => b.pos - a.pos);
    for (const { pos, tracking } of sortedDataTracked) {
      const mappedPos = tr.mapping.map(pos);
      const node = tr.doc.nodeAt(mappedPos);
      if (!node) continue;
      if (tracking.originalType === "paragraphInserted") {
        tr.setNodeMarkup(mappedPos, void 0, { ...node.attrs, dataTracked: null });
        if (canJoin(tr.doc, mappedPos)) {
          tr.join(mappedPos);
        }
      } else if (tracking.originalType === "boundaryDeleted") {
        tr.setNodeMarkup(mappedPos, void 0, { ...node.attrs, dataTracked: null });
      } else {
        const originalType = state.schema.nodes[tracking.originalType];
        if (originalType) {
          const originalAttrs = tracking.originalAttrs ?? {};
          tr.setNodeMarkup(mappedPos, originalType, {
            ...node.attrs,
            ...originalAttrs,
            dataTracked: null
          });
        }
      }
    }
    for (const range of formatChangeRanges) {
      const mappedFrom = tr.mapping.map(range.from);
      const mappedTo = tr.mapping.map(range.to);
      tr.removeMark(mappedFrom, mappedTo, state.schema.marks.formatChange);
      if (range.formatAdded) {
        const markType = state.schema.marks[range.formatAdded];
        if (markType) {
          tr.removeMark(mappedFrom, mappedTo, markType);
        }
      }
      if (range.formatRemoved) {
        const markType = state.schema.marks[range.formatRemoved];
        if (markType) {
          tr.addMark(mappedFrom, mappedTo, markType.create());
        }
      }
    }
    const sortedDeletions = insertionDeleteRanges.sort(
      (a, b) => b.from - a.from
    );
    for (const range of sortedDeletions) {
      const mappedFrom = tr.mapping.map(range.from);
      const mappedTo = tr.mapping.map(range.to);
      tr.delete(mappedFrom, mappedTo);
    }
    if (insertionDeleteRanges.length > 0 || deletionUnmarkRanges.length > 0 || formatChangeRanges.length > 0 || dataTrackedNodes.length > 0) {
      tr.setMeta(trackChangesPluginKey, { handled: true });
      dispatch(tr);
      editor.storage.trackChanges.onStatusChange?.(changeId, "rejected");
    }
    return true;
  },
  acceptAll: () => ({ state, dispatch, editor }) => {
    if (!dispatch) return true;
    const tr = state.tr;
    const deletionRanges = [];
    const boundaryDeletePositions = [];
    state.doc.descendants((node, pos) => {
      if (node.isBlock && node.attrs.dataTracked) {
        const tracking = node.attrs.dataTracked;
        if (tracking.originalType === "boundaryDeleted") {
          boundaryDeletePositions.push(pos);
        }
        const mappedPos = tr.mapping.map(pos);
        const mappedNode = tr.doc.nodeAt(mappedPos);
        if (mappedNode) {
          tr.setNodeMarkup(mappedPos, void 0, { ...mappedNode.attrs, dataTracked: null });
        }
      }
      if (!node.isText) return;
      for (const mark of node.marks) {
        const mappedFrom = tr.mapping.map(pos);
        const mappedTo = tr.mapping.map(pos + node.nodeSize);
        if (mark.type.name === "insertion") {
          tr.removeMark(mappedFrom, mappedTo, state.schema.marks.insertion);
        }
        if (mark.type.name === "deletion") {
          deletionRanges.push({
            from: pos,
            to: pos + node.nodeSize
          });
        }
        if (mark.type.name === "formatChange") {
          tr.removeMark(mappedFrom, mappedTo, state.schema.marks.formatChange);
        }
      }
    });
    const sortedBoundaryDeletes = boundaryDeletePositions.sort((a, b) => b - a);
    for (const pos of sortedBoundaryDeletes) {
      const mappedPos = tr.mapping.map(pos);
      if (canJoin(tr.doc, mappedPos)) {
        tr.join(mappedPos);
      }
    }
    const sortedDeletions = deletionRanges.sort(
      (a, b) => b.from - a.from
    );
    for (const range of sortedDeletions) {
      const mappedFrom = tr.mapping.map(range.from);
      const mappedTo = tr.mapping.map(range.to);
      tr.delete(mappedFrom, mappedTo);
    }
    if (deletionRanges.length > 0 || tr.steps.length > 0) {
      const newDoc = tr.doc;
      const maxPos = newDoc.content.size - 1;
      const safePos = Math.min(
        state.selection.from,
        maxPos > 0 ? maxPos : 1
      );
      try {
        tr.setSelection(TextSelection2.create(newDoc, safePos));
      } catch {
        tr.setSelection(TextSelection2.create(newDoc, 1));
      }
      tr.setMeta(trackChangesPluginKey, { handled: true });
      dispatch(tr);
    }
    return true;
  },
  rejectAll: () => ({ state, dispatch, editor }) => {
    if (!dispatch) return true;
    const tr = state.tr;
    const insertionDeleteRanges = [];
    const paragraphInsertedPositions = [];
    state.doc.descendants((node, pos) => {
      if (node.isBlock && node.attrs.dataTracked) {
        const tracking = node.attrs.dataTracked;
        if (tracking.originalType === "paragraphInserted") {
          paragraphInsertedPositions.push(pos);
        } else if (tracking.originalType === "boundaryDeleted") {
          const mappedPos = tr.mapping.map(pos);
          const mappedNode = tr.doc.nodeAt(mappedPos);
          if (mappedNode) {
            tr.setNodeMarkup(mappedPos, void 0, { ...mappedNode.attrs, dataTracked: null });
          }
        } else {
          const originalType = state.schema.nodes[tracking.originalType];
          if (originalType) {
            const mappedPos = tr.mapping.map(pos);
            const mappedNode = tr.doc.nodeAt(mappedPos);
            if (mappedNode) {
              tr.setNodeMarkup(mappedPos, originalType, {
                ...mappedNode.attrs,
                ...tracking.originalAttrs ?? {},
                dataTracked: null
              });
            }
          }
        }
      }
      if (!node.isText) return;
      for (const mark of node.marks) {
        const mappedFrom = tr.mapping.map(pos);
        const mappedTo = tr.mapping.map(pos + node.nodeSize);
        if (mark.type.name === "deletion") {
          tr.removeMark(mappedFrom, mappedTo, state.schema.marks.deletion);
        }
        if (mark.type.name === "insertion") {
          insertionDeleteRanges.push({
            from: pos,
            to: pos + node.nodeSize
          });
        }
        if (mark.type.name === "formatChange") {
          if (mark.attrs.formatAdded) {
            const markType = state.schema.marks[mark.attrs.formatAdded];
            if (markType) {
              tr.removeMark(mappedFrom, mappedTo, markType);
            }
          }
          if (mark.attrs.formatRemoved) {
            const markType = state.schema.marks[mark.attrs.formatRemoved];
            if (markType) {
              tr.addMark(mappedFrom, mappedTo, markType.create());
            }
          }
          tr.removeMark(mappedFrom, mappedTo, state.schema.marks.formatChange);
        }
      }
    });
    const sortedParagraphInserts = paragraphInsertedPositions.sort((a, b) => b - a);
    for (const pos of sortedParagraphInserts) {
      const mappedPos = tr.mapping.map(pos);
      const node = tr.doc.nodeAt(mappedPos);
      if (node) {
        tr.setNodeMarkup(mappedPos, void 0, { ...node.attrs, dataTracked: null });
      }
      if (canJoin(tr.doc, mappedPos)) {
        tr.join(mappedPos);
      }
    }
    const sortedDeletions = insertionDeleteRanges.sort(
      (a, b) => b.from - a.from
    );
    for (const range of sortedDeletions) {
      const mappedFrom = tr.mapping.map(range.from);
      const mappedTo = tr.mapping.map(range.to);
      tr.delete(mappedFrom, mappedTo);
    }
    if (insertionDeleteRanges.length > 0 || tr.steps.length > 0) {
      const newDoc = tr.doc;
      const maxPos = newDoc.content.size - 1;
      const safePos = Math.min(
        state.selection.from,
        maxPos > 0 ? maxPos : 1
      );
      try {
        tr.setSelection(TextSelection2.create(newDoc, safePos));
      } catch {
        tr.setSelection(TextSelection2.create(newDoc, 1));
      }
      tr.setMeta(trackChangesPluginKey, { handled: true });
      dispatch(tr);
    }
    return true;
  },
  trackSetNode: (typeName, attrs) => ({ state, dispatch, editor }) => {
    if (!dispatch) return true;
    const mode = editor.storage.trackChanges.mode;
    if (mode === "view") return false;
    if (mode !== "suggest") {
      const { $from: $from2 } = state.selection;
      const pos2 = $from2.before($from2.depth);
      const node2 = state.doc.nodeAt(pos2);
      if (!node2) return false;
      const nodeType2 = state.schema.nodes[typeName];
      if (!nodeType2) return false;
      const tr2 = state.tr;
      tr2.setNodeMarkup(pos2, nodeType2, { ...node2.attrs, ...attrs });
      dispatch(tr2);
      return true;
    }
    const { $from } = state.selection;
    const pos = $from.before($from.depth);
    const node = state.doc.nodeAt(pos);
    if (!node) return false;
    const nodeType = state.schema.nodes[typeName];
    if (!nodeType) return false;
    if (node.type.name === typeName) return false;
    const author = editor.storage.trackChanges.author;
    const tracking = {
      changeId: generateChangeId(),
      authorId: author.id,
      authorName: author.name,
      authorColor: author.color,
      timestamp: (/* @__PURE__ */ new Date()).toISOString(),
      originalType: node.type.name,
      originalAttrs: { ...node.attrs }
    };
    if (node.attrs.dataTracked) {
      const existing = node.attrs.dataTracked;
      if (existing.originalType === typeName) {
        const tr2 = state.tr;
        tr2.setNodeMarkup(pos, nodeType, {
          ...existing.originalAttrs ?? {},
          ...attrs,
          dataTracked: null
        });
        dispatch(tr2);
        return true;
      }
      tracking.originalType = existing.originalType;
      tracking.originalAttrs = existing.originalAttrs;
      tracking.changeId = existing.changeId;
    }
    const tr = state.tr;
    tr.setNodeMarkup(pos, nodeType, {
      ...node.attrs,
      ...attrs,
      dataTracked: tracking
    });
    dispatch(tr);
    return true;
  }
};

// vendor/tiptap-track-changes/src/extension.ts
var TrackChangesExtension = Extension.create({
  name: "trackChanges",
  addOptions() {
    return {
      author: {
        id: "anonymous",
        name: "Anonymous",
        color: "#6b7280"
      },
      mode: "edit",
      onStatusChange: void 0,
      additionalBlockTypes: []
    };
  },
  addStorage() {
    return {
      mode: this.options.mode ?? "edit",
      author: this.options.author,
      onStatusChange: this.options.onStatusChange
    };
  },
  addGlobalAttributes() {
    return [
      {
        // Add dataTracked attribute to all block-level nodes for node-level
        // change tracking (e.g., paragraph -> heading type changes)
        types: [.../* @__PURE__ */ new Set([
          "paragraph",
          "heading",
          "blockquote",
          "codeBlock",
          "listItem",
          "bulletList",
          "orderedList",
          ...this.options.additionalBlockTypes ?? []
        ])],
        attributes: {
          dataTracked: {
            default: null,
            parseHTML: (element) => {
              const val = element.getAttribute("data-tracked");
              if (!val) return null;
              try {
                return JSON.parse(val);
              } catch {
                return null;
              }
            },
            renderHTML: (attributes) => {
              if (!attributes.dataTracked) return {};
              return { "data-tracked": JSON.stringify(attributes.dataTracked) };
            }
          }
        }
      }
    ];
  },
  addExtensions() {
    return [InsertionMark, DeletionMark, FormatChangeMark];
  },
  addCommands() {
    return trackChangesCommands;
  },
  addProseMirrorPlugins() {
    const storage = this.storage;
    return [
      createSuggestModePlugin({
        getMode: () => storage.mode,
        getAuthor: () => storage.author
      })
    ];
  }
});

// vendor/tiptap-track-changes/src/helpers.ts
function getTrackedChanges(editor) {
  const changes = [];
  const seen = /* @__PURE__ */ new Set();
  editor.state.doc.descendants((node, pos) => {
    if (node.isBlock && node.attrs.dataTracked) {
      const tracking = node.attrs.dataTracked;
      const key = `${tracking.changeId}-dataTracked-${pos}`;
      if (!seen.has(key)) {
        seen.add(key);
        let type;
        if (tracking.originalType === "paragraphInserted") {
          type = "insertion";
        } else if (tracking.originalType === "boundaryDeleted") {
          type = "deletion";
        } else {
          type = "nodeChange";
        }
        changes.push({
          changeId: tracking.changeId,
          type,
          authorId: tracking.authorId,
          authorName: tracking.authorName,
          authorColor: tracking.authorColor,
          timestamp: tracking.timestamp,
          from: pos,
          to: pos + node.nodeSize,
          text: node.textContent ?? ""
        });
      }
    }
    if (!node.isText) return;
    for (const mark of node.marks) {
      if (mark.type.name !== "insertion" && mark.type.name !== "deletion" && mark.type.name !== "formatChange") {
        continue;
      }
      const changeId = mark.attrs.changeId;
      const key = `${changeId}-${mark.type.name}-${pos}`;
      if (seen.has(key)) continue;
      seen.add(key);
      changes.push({
        changeId,
        type: mark.type.name,
        authorId: mark.attrs.authorId,
        authorName: mark.attrs.authorName,
        authorColor: mark.attrs.authorColor,
        timestamp: mark.attrs.timestamp,
        from: pos,
        to: pos + node.nodeSize,
        text: node.text ?? "",
        ...mark.type.name === "formatChange" ? {
          formatAdded: mark.attrs.formatAdded ?? void 0,
          formatRemoved: mark.attrs.formatRemoved ?? void 0
        } : {}
      });
    }
  });
  return changes;
}
function getGroupedChanges(editor) {
  const changes = getTrackedChanges(editor);
  const groups = /* @__PURE__ */ new Map();
  for (const change of changes) {
    const group = groups.get(change.changeId);
    if (group) {
      group.push(change);
    } else {
      groups.set(change.changeId, [change]);
    }
  }
  return groups;
}
function getBaseText(editor) {
  let text = "";
  let isFirstBlock = true;
  editor.state.doc.descendants((node) => {
    if (node.isBlock && node.isTextblock) {
      const tracking = node.attrs.dataTracked;
      if (tracking?.originalType === "paragraphInserted") {
        return;
      }
      if (!isFirstBlock) {
        text += "\n";
      }
      isFirstBlock = false;
      return;
    }
    if (!node.isText) return;
    const hasInsertion = node.marks.some(
      (m) => m.type.name === "insertion"
    );
    if (hasInsertion) return;
    text += node.text ?? "";
  });
  return text;
}
function getResultText(editor) {
  let text = "";
  let isFirstBlock = true;
  editor.state.doc.descendants((node) => {
    if (node.isBlock && node.isTextblock) {
      const tracking = node.attrs.dataTracked;
      if (tracking?.originalType === "boundaryDeleted") {
        return;
      }
      if (!isFirstBlock) {
        text += "\n";
      }
      isFirstBlock = false;
      return;
    }
    if (!node.isText) return;
    const hasDeletion = node.marks.some(
      (m) => m.type.name === "deletion"
    );
    if (hasDeletion) return;
    text += node.text ?? "";
  });
  return text;
}

// ime-adapter.ts
import { Extension as Extension2 } from "@tiptap/core";
import { Plugin as Plugin2, PluginKey as PluginKey2 } from "@tiptap/pm/state";
var CompositionAdapter = Extension2.create({
  name: "rweCompositionAdapter",
  priority: 1100,
  addOptions() {
    return { onFinish: () => {
    } };
  },
  addStorage() {
    return { active: false };
  },
  addProseMirrorPlugins() {
    const editor = this.editor, storage = this.storage, onFinish = this.options.onFinish;
    let snapshot = null, previousMode = "suggest";
    return [new Plugin2({ key: new PluginKey2("rweCompositionAdapter"), props: { handleDOMEvents: {
      compositionstart(view) {
        if (storage.active) return false;
        snapshot = view.state;
        previousMode = editor.storage.trackChanges.mode;
        storage.active = true;
        editor.storage.trackChanges.mode = "edit";
        return false;
      },
      compositionend(view, event) {
        if (!snapshot) return false;
        const before = snapshot;
        snapshot = null;
        const text = event.data || "";
        setTimeout(() => {
          if (editor.isDestroyed) return;
          view.updateState(before);
          editor.storage.trackChanges.mode = previousMode;
          storage.active = false;
          if (text) {
            const { from, to } = before.selection;
            view.someProp("handleTextInput", (fn) => fn(view, from, to, text, () => view.state.tr));
          }
          onFinish();
        }, 35);
        return false;
      }
    } } })];
  }
});

// subtitle-layout.ts
import { closeHistory } from "@tiptap/pm/history";
var layoutEdits = (editor) => editor.state.doc.attrs.subtitleEdits || [];
function sourceOffsets(ch) {
  let offset = 0;
  return ch.segments.map((segment) => {
    const start = offset;
    offset += segment.text.length;
    return { segment, start, end: offset };
  });
}
function subtitleUnits(editor, ch) {
  const sources = sourceOffsets(ch), offsets = /* @__PURE__ */ new Map();
  const original = [];
  const inserted = [];
  editor.state.doc.descendants((node, pos) => {
    if (!node.isText) return;
    const insertion = node.marks.find((m) => m.type.name === "insertion");
    if (insertion) {
      inserted.push({ pos, text: node.text, change: insertion.attrs.changeId });
      return;
    }
    const id = node.marks.find((m) => m.type.name === "sourceOrigin")?.attrs.sourceId;
    const source = sources.find((s) => `${ch.number}:${s.segment.id}` === id);
    if (!source) return;
    let offset = offsets.get(id) || 0;
    const deletion = node.marks.find((m) => m.type.name === "deletion");
    for (const text of node.text) {
      original.push({ pos, to: pos + text.length, at: source.start + offset, deleted: !!deletion, change: deletion?.attrs.changeId, text });
      offset += text.length;
      pos += text.length;
    }
    offsets.set(id, offset);
  });
  const units = original.filter((u) => !u.deleted).map((u) => ({ ...u, inserted: false }));
  for (const run of inserted) {
    const replaced = original.find((u) => u.change === run.change);
    const next = original.find((u) => u.to > run.pos), previous = original.filter((u) => u.to <= run.pos).at(-1);
    const at = replaced?.at ?? next?.at ?? previous?.at ?? 0;
    let pos = run.pos;
    for (const text of run.text) {
      units.push({ text, pos, to: pos + text.length, at, inserted: true });
      pos += text.length;
    }
  }
  return units.sort((a, b) => a.pos - b.pos);
}
function boundaryTime(ch, at, edge) {
  const sources = sourceOffsets(ch), total = sources.at(-1)?.end || 0;
  if (at === total) return sources.at(-1)?.segment.end ?? null;
  const source = sources.find((s) => s.start <= at && at < s.end);
  if (!source) return null;
  if (at === source.start) return edge === "end" && source.start > 0 ? sources[sources.indexOf(source) - 1].segment.end : source.segment.start;
  const offset = at - source.start, tokens = source.segment.timings || [];
  if (tokens.some((t) => t.start_char < offset && offset < t.end_char)) return null;
  const next = tokens.find((t) => t.start_char >= offset), previous = tokens.filter((t) => t.end_char <= offset).at(-1);
  return next && previous && next.start >= previous.end ? next.start : null;
}
function subtitleCues(editor, ch) {
  const sources = sourceOffsets(ch), total = sources.at(-1)?.end || 0;
  const boundaries = new Set(sources.map((s) => s.start));
  boundaries.add(total);
  for (const op of layoutEdits(editor)) {
    if (op.action === "split") boundaries.add(op.at);
    else for (const at of boundaries) if (op.scope_start < at && at < op.scope_end) boundaries.delete(at);
  }
  const points = [...boundaries].sort((a, b) => a - b), units = subtitleUnits(editor, ch), cues = [];
  for (let i = 0; i < points.length - 1; i++) {
    const at = points[i], end_at = points[i + 1], selected = units.filter((u) => at <= u.at && u.at < end_at);
    const text = selected.map((u) => u.text).join("");
    if (!text.trim()) continue;
    const start = boundaryTime(ch, at, "start"), end = boundaryTime(ch, end_at, "end");
    if (start === null || end === null) continue;
    cues.push({ id: `subtitle:${at}:${end_at}`, start, end, text, at, end_at, from: selected[0].pos, to: selected.at(-1).to, units: selected });
  }
  return cues;
}
function subtitleLayoutGroups(editor, ch) {
  const sources = sourceOffsets(ch), units = subtitleUnits(editor, ch);
  return layoutEdits(editor).map((op) => {
    const nearby = units.find((u) => u.at >= op.at) || units.at(-1);
    const anchors = sources.filter((s) => s.start < op.scope_end && s.end > op.scope_start).map((s) => {
      const start = Math.max(0, op.scope_start - s.start), end = Math.min(s.segment.text.length, op.scope_end - s.start);
      return {
        source_id: `${ch.number}:${s.segment.id}`,
        chapter: ch.number,
        file: ch.file,
        segment_id: s.segment.id,
        start_char: start,
        end_char: end,
        original_text: s.segment.text.slice(start, end),
        envelope_start_s: s.segment.start,
        envelope_end_s: s.segment.end,
        timing_status: "subtitle_layout_only"
      };
    });
    return { id: op.id, changes: [], from: nearby?.pos || 1, to: nearby?.pos || 1, deleted: "", inserted: "", kind: "subtitle_structure", anchors, layout: op };
  });
}

// model.ts
var EFFECTS = { cut_media: "\u526A\u6389\u5BF9\u5E94\u539F\u58F0\u7247\u6BB5", transcript_only: "\u4EC5\u4FEE\u6B63\u8BC6\u522B\u6587\u5B57\uFF0C\u4FDD\u7559\u539F\u58F0", review_required: "\u5F85\u56DE\u542C\u540E\u51B3\u5B9A" };
var Origin = Mark4.create({
  name: "sourceOrigin",
  inclusive: false,
  addAttributes() {
    return { sourceId: { default: null } };
  },
  parseHTML() {
    return [{ tag: "span[data-source-id]" }];
  },
  renderHTML({ HTMLAttributes }) {
    return ["span", mergeAttributes4({ "data-source-id": HTMLAttributes.sourceId }), 0];
  }
});
var SourceParagraph = Paragraph.extend({
  addAttributes() {
    return { ...this.parent?.(), sourceTime: { default: null, renderHTML: (a) => a.sourceTime ? { "data-time": a.sourceTime } : {} } };
  }
});
var ReviewDocument = Document.extend({ addAttributes() {
  return { subtitleEdits: { default: null, rendered: false } };
} });
function time(t) {
  return `${String(Math.floor(t / 60)).padStart(2, "0")}:${(t % 60).toFixed(2).padStart(5, "0")}`;
}
function sourceId(ch, s) {
  return `${ch.number}:${s.id}`;
}
function createReviewEditor(options) {
  return new Editor({
    element: options.element,
    extensions: [
      ReviewDocument,
      SourceParagraph,
      Text,
      History,
      Origin,
      CompositionAdapter.configure({ onFinish: () => options.onUpdate?.({}) }),
      TrackChangesExtension.configure({ author: { id: "local-reviewer", name: "\u672C\u4EBA\u590D\u6838", color: "#22634d" }, mode: "suggest" }),
      ...options.extraExtensions || []
    ],
    content: options.content,
    onUpdate: options.onUpdate || (() => {
    }),
    onSelectionUpdate: options.onSelectionUpdate || (() => {
    }),
    editorProps: { attributes: { "aria-label": "\u53E3\u64AD\u9010\u5B57\u7A3F\u4FEE\u8BA2\u7F16\u8F91\u5668", spellcheck: "false" } }
  });
}
function trackedDelete(editor) {
  if (editor.state.selection.empty) return false;
  let handled = false;
  editor.view.someProp("handleKeyDown", (fn) => {
    handled = !!fn(editor.view, new KeyboardEvent("keydown", { key: "Delete", bubbles: true }));
    return handled;
  });
  return handled;
}
function originRuns(editor, ch) {
  const offsets = /* @__PURE__ */ new Map();
  const runs = [];
  editor.state.doc.descendants((node, pos) => {
    if (!node.isText || node.marks.some((m) => m.type.name === "insertion")) return;
    const id = node.marks.find((m) => m.type.name === "sourceOrigin")?.attrs.sourceId;
    if (!id || !id.startsWith(ch.number + ":")) return;
    const offset = offsets.get(id) || 0;
    runs.push({ from: pos, to: pos + node.nodeSize, id, offset });
    offsets.set(id, offset + node.nodeSize);
  });
  return runs;
}
function anchorsForRange(editor, ch, from, to) {
  const result = [];
  const runs = originRuns(editor, ch);
  const matching = runs.filter((r) => r.from < to && r.to > from);
  if (!matching.length) {
    const near = runs.find((r) => r.to >= from) || runs.at(-1);
    if (near) matching.push({ ...near, from: Math.min(near.to, Math.max(near.from, from)), to: Math.min(near.to, Math.max(near.from, from)), offset: near.offset + Math.max(0, Math.min(from - near.from, near.to - near.from)) });
  }
  for (const r of matching) {
    const s = ch.segments.find((x) => sourceId(ch, x) === r.id);
    if (!s) continue;
    const a = r.offset + Math.max(0, from - r.from), b = r.offset + Math.min(r.to - r.from, Math.max(0, to - r.from));
    const start = Math.min(s.text.length, a), end = Math.max(start, Math.min(s.text.length, b));
    const previous = result.at(-1);
    if (previous?.source_id === r.id && previous.end_char === start) {
      previous.end_char = end;
      previous.original_text = s.text.slice(previous.start_char, end);
    } else result.push({ source_id: r.id, chapter: ch.number, file: ch.file, segment_id: s.id, start_char: start, end_char: end, original_text: s.text.slice(start, end), envelope_start_s: s.start, envelope_end_s: s.end, timing_status: start === 0 && end === s.text.length ? "phrase_interval_review_required" : "word_alignment_required" });
  }
  for (const anchor of result) {
    const seg = ch.segments.find((s) => s.id === anchor.segment_id);
    const ranges = seg?.timings?.filter((t) => t.start_char < anchor.end_char && t.end_char > anchor.start_char);
    if (ranges?.length) {
      anchor.native_ranges = ranges;
      anchor.timing_status = "native_asr_character_timing";
    }
  }
  return result;
}
function groupedReview(editor, ch) {
  const textGroups = [...getGroupedChanges(editor)].map(([id, changes]) => {
    const deleted = changes.filter((x) => x.type === "deletion" && editor.state.doc.nodeAt(x.from)?.isText).sort((a, b) => a.from - b.from);
    const inserted = changes.filter((x) => x.type === "insertion" && editor.state.doc.nodeAt(x.from)?.isText).sort((a, b) => a.from - b.from);
    const from = Math.min(...changes.map((x) => x.from)), to = Math.max(...changes.map((x) => x.to));
    const anchors = deleted.length ? deleted.flatMap((x) => anchorsForRange(editor, ch, x.from, x.to)) : anchorsForRange(editor, ch, from, from);
    return { id, changes, from, to, deleted: deleted.map((x) => x.text).join(""), inserted: inserted.map((x) => x.text).join(""), kind: deleted.length ? inserted.length ? "replacement" : "deletion" : inserted.length ? "insertion" : "structure", anchors, layout: void 0 };
  });
  return [...textGroups, ...subtitleLayoutGroups(editor, ch).map((g) => ({ ...g, anchors: g.anchors }))].sort((a, b) => a.from - b.from);
}
function ensureMetadata(editor, ch, meta) {
  const groups = groupedReview(editor, ch);
  for (const g of groups) {
    const first = g.anchors[0];
    const s = ch.segments.find((x) => sourceId(ch, x) === first?.source_id);
    const fingerprint = JSON.stringify([g.deleted, g.inserted, g.anchors.map((a) => [a.source_id, a.start_char, a.end_char])]);
    if (!meta[g.id]) {
      const effect = g.kind === "deletion" ? "cut_media" : g.kind === "structure" ? "review_required" : "transcript_only";
      meta[g.id] = {
        intent: g.kind === "deletion" ? "\u5220\u9664\u53E3\u8BEF\u3001\u91CD\u8BFB\u6216\u5197\u4F59\u5019\u9009" : "\u4FEE\u6B63\u8BC6\u522B\u6587\u5B57\u5E76\u4FDD\u7559\u4FEE\u8BA2\u75D5\u8FF9",
        decision: EFFECTS[effect],
        reason: g.id.startsWith("preset:") ? s?.preselection_reason || "\u6309\u65E7\u9875\u590D\u6838\u9009\u62E9\u8FC1\u5165" : "\u76F4\u63A5\u5728\u9010\u5B57\u7A3F\u4E2D\u63D0\u51FA\u4FEE\u6539\uFF1B\u53EF\u5728\u6B64\u8865\u5145\u5224\u65AD\u7406\u7531\u3002",
        evidence: `${ch.file}\uFF1B\u539F\u8BC6\u522B\u6587\u5B57\u201C${g.deleted || s?.text || ""}\u201D\u3002${s?.script_match ? `\u53E3\u64AD\u7A3F v0.6 \u5BF9\u7167\uFF1A${s.script_match}` : "\u539F\u58F0\u56DE\u542C\u72B6\u6001\u5F85\u786E\u8BA4\u3002"}`,
        effective_scope: "",
        effect,
        status: "proposed",
        origin: g.id.startsWith("preset:") ? "legacy_preselection" : "manual_text_edit"
      };
    }
    if (g.layout && meta[g.id].origin !== "subtitle_layout") Object.assign(meta[g.id], {
      intent: g.layout.action === "split" ? "\u5728\u5149\u6807\u5904\u4FEE\u6B63\u5B57\u5E55\u65AD\u53E5" : "\u5408\u5E76\u76F8\u90BB\u5B57\u5E55\uFF0C\u4FEE\u6B63\u9519\u8BEF\u65AD\u53E5",
      reason: "\u5B57\u5E55\u663E\u793A\u5206\u7EC4\u4E0E\u5B9E\u9645\u8BED\u4E49\u4E0D\u4E00\u81F4\uFF1B\u53EA\u8C03\u6574\u5B57\u5E55\u8FB9\u754C\uFF0C\u4E0D\u6539\u53D8\u6587\u5B57\u4FEE\u8BA2\u548C\u539F\u58F0\u64AD\u653E\u533A\u95F4\u3002",
      evidence: `${ch.file}\uFF1B\u539F\u6587\u54C8\u5E0C ${ch.source_text_hash}\uFF1B\u539F\u5B57\u5E55\u201C${g.layout.before}\u201D\uFF1B\u8C03\u6574\u540E\u201C${g.layout.after}\u201D\uFF1B\u539F\u59CB\u5B57\u7B26\u504F\u79FB ${g.layout.at}\u3002`,
      origin: "subtitle_layout"
    });
    if (meta[g.id].content_fingerprint && meta[g.id].content_fingerprint !== fingerprint) {
      meta[g.id].status = "proposed";
      delete meta[g.id].confirmed_at;
    }
    meta[g.id].content_fingerprint = fingerprint;
    meta[g.id].effective_scope = g.anchors.map((a) => `\u7B2C${ch.number}\u6BB5 ${a.segment_id} \u5B57\u7B26[${a.start_char},${a.end_char})\uFF1B\u539F\u7247 ${time(a.envelope_start_s)}\u2013${time(a.envelope_end_s)}\uFF08\u77ED\u8BED\u8303\u56F4\uFF09${a.native_ranges?.length ? `\uFF1B\u5B57\u8BCD ${time(a.native_ranges[0].start)}\u2013${time(a.native_ranges.at(-1).end)}` : ""}`).join("\uFF1B") || `\u7B2C${ch.number}\u6BB5\u6587\u7A3F\u7ED3\u6784`;
  }
  return groups;
}
function reviewPayload(ch, editor, meta) {
  const groups = ensureMetadata(editor, ch, meta);
  return {
    chapter: ch.number,
    file: ch.file,
    source_hash: ch.source_text_hash,
    doc: editor.getJSON(),
    original_text: getBaseText(editor),
    proposed_text: getResultText(editor),
    revisions: groups.map((g) => ({ change_id: g.id, kind: g.kind, original: g.deleted, corrected: g.inserted, ...meta[g.id], source_anchors: g.anchors, ...g.layout ? { subtitle_layout: g.layout } : {} }))
  };
}

// vendor/hyperaudio/playable-sections.js
function isStruck(el) {
  return el.style.textDecoration.includes("line-through");
}
function computePlayableSections(transcript, options = {}) {
  const wordEls = transcript.querySelectorAll("[data-m]");
  const player = options.player;
  const removeGapsEnabled = !!options.removeGapsEnabled, gapThreshold = options.gapThreshold ?? 0.5, gapBuffer = options.gapBuffer ?? 0.1;
  const duration = player && !isNaN(player.duration) && player.duration > 0 ? player.duration : Infinity;
  if (wordEls.length === 0) {
    return [{ start: 0, end: duration }];
  }
  const words = Array.from(wordEls).map((el) => {
    const start = parseInt(el.getAttribute("data-m")) / 1e3;
    return {
      start,
      end: start + (parseInt(el.getAttribute("data-d")) || 0) / 1e3,
      struck: isStruck(el),
      text: el.textContent
    };
  });
  const cuts = [];
  let prevKeptEnd = null;
  let prevKeptWord = null;
  let pendingStruck = [];
  const struckRunCut = (struckSpans) => {
    if (struckSpans.length === 0) return null;
    let end = struckSpans[0].end;
    struckSpans.forEach((s) => {
      if (s.end > end) end = s.end;
    });
    return { start: struckSpans[0].start, end };
  };
  const TRUNCATION_RATIO = 0.4;
  const syllableCount = (text) => {
    const w = String(text || "").toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/[^a-z]/g, "");
    if (w === "") return 0;
    const groups = w.match(/[aeiouy]+/g);
    let n = groups ? groups.length : 1;
    if (w.endsWith("e") && n > 1) n -= 1;
    return Math.max(1, n);
  };
  const spokenRate = (word) => {
    if (!word) return null;
    const syllables = syllableCount(word.text);
    const seconds = word.end - word.start;
    if (syllables < 1 || seconds <= 0) return null;
    return seconds / syllables;
  };
  const rates = words.map(spokenRate).filter((r) => r !== null).sort((a, b) => a - b);
  const medianRate = rates.length >= 20 ? rates[Math.floor(rates.length / 2)] : null;
  const looksTruncated = (word) => {
    if (medianRate === null) return false;
    const syllables = syllableCount(word ? word.text : "");
    if (syllables < 2) return false;
    const rate = spokenRate(word);
    return rate !== null && rate < medianRate * TRUNCATION_RATIO;
  };
  const flushRegion = (regionStart2, regionEnd, struckSpans, bufferLeft, bufferRight, precedingWord) => {
    const run = struckRunCut(struckSpans);
    if (removeGapsEnabled && regionStart2 !== null && !looksTruncated(precedingWord)) {
      const struckTotal = struckSpans.reduce((sum, s) => sum + (s.end - s.start), 0);
      const effectivePause = regionEnd - regionStart2 - struckTotal;
      if (effectivePause > gapThreshold) {
        const cutStart = Math.min(regionStart2 + (bufferLeft ? gapBuffer : 0), run !== null ? run.start : Infinity);
        const cutEnd = Math.max(regionEnd - (bufferRight ? gapBuffer : 0), run !== null ? run.end : -Infinity);
        if (cutEnd > cutStart) {
          cuts.push({ start: cutStart, end: cutEnd });
          return;
        }
      }
    }
    if (run !== null) cuts.push(run);
  };
  const firstKeptIndex = words.findIndex((w) => !w.struck);
  if (firstKeptIndex !== -1 && words[firstKeptIndex].start > 0) {
    const leadingStruck = words.slice(0, firstKeptIndex).filter((w) => w.struck);
    flushRegion(0, words[firstKeptIndex].start, leadingStruck, false, true);
    words.slice(0, firstKeptIndex).forEach((w) => {
      w.struck = false;
      w.consumed = true;
    });
  }
  words.forEach((word) => {
    if (word.consumed) return;
    if (word.struck) {
      pendingStruck.push(word);
      return;
    }
    if (pendingStruck.length > 0 || prevKeptEnd !== null) {
      const regionStart2 = prevKeptEnd !== null ? prevKeptEnd : pendingStruck.length ? pendingStruck[0].start : null;
      if (regionStart2 !== null && word.start > regionStart2) {
        flushRegion(regionStart2, word.start, pendingStruck, prevKeptEnd !== null, true, prevKeptWord);
      } else {
        const run = struckRunCut(pendingStruck);
        if (run !== null) cuts.push(run);
      }
    }
    pendingStruck = [];
    prevKeptEnd = word.end;
    prevKeptWord = word;
  });
  const regionStart = prevKeptEnd !== null ? prevKeptEnd : pendingStruck.length > 0 ? pendingStruck[0].start : null;
  if (regionStart !== null) {
    const struckEnd = pendingStruck.length > 0 ? pendingStruck[pendingStruck.length - 1].end : regionStart;
    const regionEnd = Number.isFinite(duration) ? Math.max(duration, struckEnd) : struckEnd;
    if (regionEnd > regionStart) {
      flushRegion(regionStart, regionEnd, pendingStruck, prevKeptEnd !== null, false, prevKeptWord);
    }
  }
  cuts.sort((a, b) => a.start - b.start);
  const sections = [];
  let cursor = 0;
  cuts.forEach((cut) => {
    const start = Math.max(cut.start, cursor);
    if (start >= cut.end) {
      cursor = Math.max(cursor, cut.end);
      return;
    }
    if (start > cursor) sections.push({ start: cursor, end: start });
    cursor = cut.end;
  });
  if (cursor < duration) sections.push({ start: cursor, end: duration });
  if (sections.length === 0) sections.push({ start: duration, end: duration });
  return sections;
}

// preview.ts
function timedTranscript(words) {
  return { querySelectorAll() {
    return words.map((w) => ({ getAttribute(name) {
      return String(Math.round((name === "data-m" ? w.start : w.end - w.start) * 1e3));
    }, style: { textDecoration: w.struck ? "line-through" : "" }, textContent: w.text || "" }));
  } };
}
function subtractCuts(cuts, duration) {
  cuts.sort((a, b) => a.start - b.start);
  const sections = [];
  let cursor = 0;
  cuts.forEach((cut) => {
    const start = Math.max(cut.start, cursor);
    if (start >= cut.end) {
      cursor = Math.max(cursor, cut.end);
      return;
    }
    if (start > cursor) sections.push({ start: cursor, end: start });
    cursor = cut.end;
  });
  if (cursor < duration) sections.push({ start: cursor, end: duration });
  return sections;
}
function complement(sections, duration) {
  const cuts = [];
  let cursor = 0;
  for (const s of sections) {
    if (s.start > cursor) cuts.push({ start: cursor, end: s.start });
    cursor = Math.max(cursor, s.end);
  }
  if (cursor < duration) cuts.push({ start: cursor, end: duration });
  return cuts;
}
function buildPreviewPlan(editor, chapter, meta, duration, manual = [], pauses = []) {
  const groups = groupedReview(editor, chapter), warnings = [];
  const ranges = groups.filter((g) => !["structure", "subtitle_structure"].includes(g.kind) && meta[g.id]?.effect === "cut_media").flatMap((g) => g.anchors);
  const words = chapter.segments.flatMap((s) => (s.timings || []).map((t) => ({ ...t, segment_id: s.id, text: t.text, struck: ranges.some((a) => a.segment_id === s.id && a.start_char < t.end_char && a.end_char > t.start_char) })));
  const coarse = [];
  for (const anchor of ranges) {
    const s = chapter.segments.find((x) => x.id === anchor.segment_id);
    if (!s) continue;
    if (s.timings?.some((t) => t.start_char < anchor.end_char && t.end_char > anchor.start_char)) continue;
    if (!/[\p{L}\p{N}]/u.test(anchor.original_text)) continue;
    if (anchor.start_char === 0 && anchor.end_char === s.text.length) coarse.push({ start: s.start, end: s.end });
    else warnings.push(`${s.id} \u7684\u5C40\u90E8\u5220\u8BCD\u7F3A\u5C11\u5B57\u8BCD\u65F6\u95F4\uFF0C\u6682\u4FDD\u7559\u539F\u58F0`);
  }
  const knownDuration = Number.isFinite(duration) && duration > 0 ? duration : chapter.duration || 0;
  const native = knownDuration > 0 ? computePlayableSections(timedTranscript(words), { player: { duration: knownDuration }, removeGapsEnabled: false }).filter((s) => s.end > s.start) : [];
  const manualCuts = manual.filter((r) => Number.isFinite(r.start) && Number.isFinite(r.end) && r.start >= 0 && r.end > r.start && r.end <= knownDuration).map((r) => ({ start: r.start, end: r.end }));
  const subtitles = Object.fromEntries(chapter.segments.map((s) => [s.id, ""]));
  const fallback = new Map(groups.map((g) => [g.id, g.anchors[0]?.source_id]));
  const runs = originRuns(editor, chapter);
  editor.state.doc.descendants((node, pos) => {
    if (!node.isText || node.marks.some((m) => m.type.name === "deletion")) return;
    const change = node.marks.find((m) => m.type.name === "insertion")?.attrs.changeId;
    let id = node.marks.find((m) => m.type.name === "sourceOrigin")?.attrs.sourceId || fallback.get(change);
    if (!id) id = runs.find((r) => r.to >= pos)?.id || runs.at(-1)?.id;
    const segment = chapter.segments.find((s) => id === `${chapter.number}:${s.id}`);
    if (segment) subtitles[segment.id] += node.text || "";
  });
  const transcriptOnlyAudio = new Set(groups.filter((g) => meta[g.id]?.effect === "transcript_only" && g.deleted).flatMap((g) => g.anchors.map((a) => a.segment_id)));
  const coverage = chapter.segments.filter((s) => subtitles[s.id].trim() || transcriptOnlyAudio.has(s.id)).flatMap((s) => {
    const spoken = words.filter((w) => w.segment_id === s.id && w.end > w.start && /[\p{L}\p{N}]/u.test(w.text));
    const kept = spoken.filter((w) => !w.struck);
    if (spoken.length && !kept.length) return [];
    const start = spoken[0]?.struck ? kept[0].start : s.start;
    const end = spoken.at(-1)?.struck ? kept.at(-1).end : s.end;
    return [{ start: Math.max(0, start), end: Math.min(knownDuration, end) }].filter((t) => t.end > t.start);
  });
  const activePauses = pauses.filter((p) => p.chapter === chapter.number && p.source_hash === chapter.source_text_hash && [p.left, p.right].every((a) => {
    const matches = words.filter((w) => w.segment_id === a.segment_id && w.start_char < a.end_char && w.end_char > a.start_char);
    return matches.length > 0 && matches.every((w) => !w.struck);
  }));
  for (const p of activePauses) coverage.push({ start: p.protect_start, end: p.start }, { start: p.end, end: p.protect_end });
  const outsideSubtitles = subtractCuts(coverage, knownDuration);
  const sections = subtractCuts([...outsideSubtitles, ...complement(native, knownDuration), ...coarse, ...manualCuts, ...activePauses.map((p) => ({ start: p.start, end: p.end }))], knownDuration);
  const cuts = complement(sections, knownDuration);
  const displaySubtitles = layoutEdits(editor).length ? subtitleCues(editor, chapter).map(({ id, start, end, text }) => ({ id, start, end, text })) : chapter.segments.map((s) => ({ id: s.id, start: s.start, end: s.end, text: subtitles[s.id] }));
  return { basis: "retained_subtitle_intervals.v1", boundary_policy: "trim_deleted_edge_words.v1", sections, cuts, removedSeconds: cuts.reduce((n, s) => n + s.end - s.start, 0), editedDuration: sections.reduce((n, s) => n + s.end - s.start, 0), warnings, subtitles, displaySubtitles, pacing_audition_ids: activePauses.map((p) => p.id) };
}

// punctuation-cleanup.ts
import { closeHistory as closeHistory2 } from "@tiptap/pm/history";
function visibleTextIndex(editor) {
  let text = "";
  const units = [];
  editor.state.doc.descendants((node, pos) => {
    if (!node.isText || node.marks.some((m) => m.type.name === "deletion")) return;
    const protectedText = node.marks.some((m) => m.type.name === "insertion");
    for (let i = 0; i < node.text.length; i++) units.push({ pos: pos + i, protected: protectedText });
    text += node.text;
  });
  return { text, units };
}
function applyPunctuationPlan(editor, chapter, meta, plan) {
  const index = visibleTextIndex(editor), original = getBaseText(editor);
  if (chapter.number !== plan.chapter || chapter.source_text_hash !== plan.source_hash || index.text !== plan.original_flat) throw Error("\u6807\u70B9\u6E05\u7406\u524D\u7684\u6587\u7A3F\u5DF2\u53D8\u5316\uFF0C\u505C\u6B62\u5E94\u7528\u3002");
  const author = { ...editor.storage.trackChanges.author };
  const existing = new Set(groupedReview(editor, chapter).map((g) => g.id));
  const ranges = [];
  for (const removal of plan.removals) {
    if (removal.end_char !== removal.start_char + 1 || index.text.slice(removal.start_char, removal.end_char) !== removal.text || !/^[，。！？、；：,!?;]$/u.test(removal.text)) throw Error("\u6E05\u7406\u8BA1\u5212\u5305\u542B\u975E\u6807\u70B9\u6539\u52A8\u3002");
    const unit = index.units[removal.start_char];
    if (!unit || unit.protected) throw Error("\u6E05\u7406\u8BA1\u5212\u89E6\u53CA\u5DF2\u6709\u4EBA\u5DE5\u4FEE\u8BA2\uFF0C\u505C\u6B62\u5E94\u7528\u3002");
    const previous = ranges.at(-1);
    if (previous && previous.to === unit.pos) {
      previous.to++;
      previous.text += removal.text;
    } else ranges.push({ from: unit.pos, to: unit.pos + 1, text: removal.text });
  }
  editor.view.dispatch(closeHistory2(editor.state.tr));
  try {
    for (const [i, range] of [...ranges].reverse().entries()) {
      editor.commands.setTrackChangesAuthor({ id: `punctuation-cleanup:${chapter.number}:${Date.now()}:${i}`, name: "\u6807\u70B9\u6E05\u7406", color: "#22634d" });
      editor.commands.setTextSelection({ from: range.from, to: range.to });
      if (!trackedDelete(editor)) throw Error("\u4E0A\u6E38\u4FEE\u8BA2\u5F15\u64CE\u672A\u80FD\u5E94\u7528\u6807\u70B9\u4FEE\u6539\u3002");
      const groups = ensureMetadata(editor, chapter, meta);
      for (const g of groups) {
        if (existing.has(g.id) || meta[g.id].origin === "punctuation_cleanup") continue;
        Object.assign(meta[g.id], {
          effect: "transcript_only",
          decision: EFFECTS.transcript_only,
          intent: "\u6E05\u7406\u91CD\u590D\u3001\u591A\u4F59\u7684\u673A\u5668\u6807\u70B9",
          reason: "\u539F\u8BC6\u522B\u6807\u70B9\u53E0\u52A0\u6216\u8DE8\u6BB5\u62FC\u63A5\u9020\u6210\u91CD\u590D\uFF1B\u4EC5\u4FEE\u5B57\u5E55\uFF0C\u4E0D\u526A\u539F\u58F0\u3002",
          evidence: `${chapter.file}\uFF1B\u539F\u6587\u201C${g.deleted}\u201D\uFF1B\u89C4\u5219 deduplicate_asr_punctuation.v1\uFF1B\u539F\u6587\u54C8\u5E0C ${chapter.source_text_hash}\uFF1B\u5DF2\u6709\u4EBA\u5DE5\u4FEE\u6539\u672A\u8986\u76D6\u3002`,
          origin: "punctuation_cleanup",
          status: "proposed"
        });
      }
    }
  } finally {
    editor.commands.setTrackChangesAuthor(author);
    editor.view.dispatch(closeHistory2(editor.state.tr));
  }
  if (getBaseText(editor) !== original) throw Error("\u539F\u59CB\u8BC6\u522B\u6587\u672C\u53D1\u751F\u53D8\u5316\uFF0C\u505C\u6B62\u5E94\u7528\u3002");
  return groupedReview(editor, chapter).filter((g) => !existing.has(g.id)).map((g) => g.id);
}

// punctuation-cleanup-cli.ts
var dom = new JSDOM("<!doctype html><body></body>", { url: "http://127.0.0.1" });
for (const key of ["window", "document", "navigator", "Node", "HTMLElement", "MutationObserver", "getComputedStyle", "KeyboardEvent", "Range"]) Object.defineProperty(globalThis, key, { value: dom.window[key], configurable: true });
if (!Range.prototype.getClientRects) Range.prototype.getClientRects = () => [];
if (!Range.prototype.getBoundingClientRect) Range.prototype.getBoundingClientRect = () => ({ top: 0, left: 0, bottom: 0, right: 0, width: 0, height: 0 });
var input = JSON.parse(readFileSync(process.argv[2], "utf8"));
var review = structuredClone(input.review);
var evidence = [];
var preservedKeys = ["change_id", "kind", "original", "corrected", "intent", "decision", "reason", "evidence", "effective_scope", "effect", "status", "origin", "confirmed_at", "content_fingerprint", "source_anchors"];
for (const chapter of input.chapters) {
  const saved = review.chapters.find((c) => c.chapter === chapter.number);
  const editor = createReviewEditor({ element: document.createElement("div"), content: saved.doc });
  const meta = Object.fromEntries(saved.revisions.map((r) => [r.change_id, { ...r }]));
  ensureMetadata(editor, chapter, meta);
  const baseline = reviewPayload(chapter, editor, meta), before = buildPreviewPlan(editor, chapter, meta, chapter.duration, review.manual_time_ranges?.[chapter.number] || []);
  const plan = input.plan.chapters.find((c) => c.chapter === chapter.number);
  const added = applyPunctuationPlan(editor, chapter, meta, plan);
  const after = reviewPayload(chapter, editor, meta), preview = buildPreviewPlan(editor, chapter, meta, chapter.duration, review.manual_time_ranges?.[chapter.number] || []);
  for (const old of baseline.revisions) {
    const current = after.revisions.find((r) => r.change_id === old.change_id);
    assert.ok(current, "existing revision disappeared");
    for (const key of preservedKeys) assert.deepEqual(current[key], old[key], `existing revision ${old.change_id}: ${key}`);
  }
  assert.deepEqual(preview.sections, before.sections, "audio keep ranges changed");
  assert.deepEqual(preview.cuts, before.cuts, "audio cuts changed");
  assert.equal(preview.editedDuration, before.editedDuration);
  assert.equal(after.original_text, baseline.original_text);
  assert.ok(after.revisions.filter((r) => added.includes(r.change_id)).every((r) => r.effect === "transcript_only"));
  Object.assign(saved, after, { preview });
  evidence.push({ chapter: chapter.number, added_revisions: added.length, removed_punctuation: plan.removals.length, protected_user_characters_skipped: plan.protected_user_characters_skipped, existing_revisions_preserved: true, media_ranges_unchanged: true, before_flat: plan.original_flat, after_flat: visibleTextIndex(editor).text });
  editor.destroy();
}
review.exported_at = (/* @__PURE__ */ new Date()).toISOString();
review.punctuation_cleanup = { policy: input.plan.policy, effect: "transcript_only", media_modified: false };
writeFileSync(process.argv[3], JSON.stringify({ review, evidence }, null, 2));
console.log(JSON.stringify({ chapters: evidence.length, added_revisions: evidence.reduce((n, c) => n + c.added_revisions, 0), removed_punctuation: evidence.reduce((n, c) => n + c.removed_punctuation, 0), media_modified: false }));
