// Composition adapter using upstream edit/suggest modes, document snapshots and handleTextInput.
// Native IME completes without tracked-mark DOM changes. Restore the pre-composition EditorState,
// then replay the committed text once through the upstream revision handler.
import { Extension } from '@tiptap/core';
import { Plugin, PluginKey, type EditorState } from '@tiptap/pm/state';

export const CompositionAdapter = Extension.create({
  name: 'rweCompositionAdapter', priority: 1100,
  addOptions() { return { onFinish: () => {} }; },
  addStorage() { return { active: false }; },
  addProseMirrorPlugins() {
    const editor = this.editor, storage = this.storage, onFinish = this.options.onFinish;
    let snapshot: EditorState | null = null, previousMode = 'suggest';
    return [new Plugin({ key: new PluginKey('rweCompositionAdapter'), props: { handleDOMEvents: {
      compositionstart(view) {
        if (storage.active) return false;
        snapshot = view.state; previousMode = editor.storage.trackChanges.mode;
        storage.active = true;
        editor.storage.trackChanges.mode = 'edit';
        return false;
      },
      compositionend(view, event) {
        if (!snapshot) return false;
        const before = snapshot; snapshot = null;
        const text = (event as CompositionEvent).data || '';
        setTimeout(() => {
          if (editor.isDestroyed) return;
          view.updateState(before);
          editor.storage.trackChanges.mode = previousMode as any;
          storage.active = false;
          if (text) {
            const { from, to } = before.selection;
            view.someProp('handleTextInput', fn => fn(view, from, to, text, () => view.state.tr));
          }
          onFinish();
        }, 35);
        return false;
      },
    } } })];
  },
});
