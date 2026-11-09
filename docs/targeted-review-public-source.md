# Targeted review and transcript editor — public source notes

This document describes the targeted-review surface as it exists in this source
overlay. It is descriptive source documentation, not a readiness or approval
claim.

## Overlay scope

This overlay carries the accepted targeted-review source together with the
public metadata and notices needed to distribute it. It does not edit, retype or
rewrite production Python, TypeScript, JavaScript, JSON Schema or test source;
executable source bodies are retained from the accepted source snapshot. Five
adapted vendor files additionally receive prepend-only source-curation comments.
For publication, CRLF-to-LF text normalization resolves the unchanged Git
whitespace check; Python/test AST and JSON values remain identical. Three
trailing spaces in the retained Hyperaudio licensing texts were also removed,
with every license word preserved. The raw tested/upstream snapshots are
retained separately. These text changes do not constitute a
new runtime test or a frontend rebuild.

## Entrypoints

- `video-knowledge-targeted-review prepare <request.json> <output-dir>` — create
  a targeted review pack, its HTML page, its initial decisions file and, when a
  project identity is configured, a bound export context.
- `video-knowledge-targeted-review serve <bundle-dir> [--host 127.0.0.1]
  [--port 0] [--open]` — open the loopback-only review server.
- `video-knowledge-review <bundle-dir>` — the existing public base review server;
  unchanged and still available for review/workbench/subtitle surfaces.
- `python -m video_knowledge_pipeline.review_export_context configure <root>
  --topic-title ... --video-title ... --video-id ...` — configure the explicit
  project identity used for export naming; `status` reports the current value.

## Modules and purpose

- `targeted_review.py` — request/source/hash binding, exact time ranges, cue and
  presentation validation, pack integrity, decision receipt, loopback routes.
- `targeted_review_ui.py` — single-file review page (queue, player, timeline,
  subtitle preview, dialogue marking, decision form).
- `targeted_transcript.py` — optional transcript source normalization, preserved
  original validation, revision-CAS draft persistence, editor page assembly.
- `targeted_pacing.py` — optional read-only pacing-candidate verification; a
  pending suggestion never becomes an approved media edit.
- `targeted_program_preview.py` — optional hash-bound continuous preview; it does
  not compile edits, save decisions or replace transcript editing.
- `targeted_visual_evidence.py` — local-first visual evidence orchestration over
  the existing triage/OCR/tile helpers; no online call is made during preview.
- `visual_evidence.py` — the existing public definitions are retained unchanged;
  the accepted source adds the bounded multimodal-document support helper for
  the optional visual-evidence path. The complete module differs from the public
  baseline; this is not a claim that its entire file is unchanged.
- `review_export_context.py` / `review_export_naming.py` — explicit project
  identity and export naming. With an identity bound, names include topic, video,
  video id, Beijing time and pack/source bindings. Legacy public pages without an
  identity keep their own filename fallback; a targeted pack without identity
  stays available but blocks export rather than guessing a camera filename.
- `webui_bridge.py` — the existing public web-UI bridge, which includes a bounded
  project-identity merge in two top-level functions so the existing web UI can
  pass the explicit review-export identity. It adds no new bridge, service, route
  or renderer.
- `review_http.py` — the existing public base review server, reused unchanged.

## Dependencies

- Python: the standard library, the already-declared `jsonschema` dependency and
  the existing public package modules. No new Python dependency, version or
  script entrypoint is introduced by this feature.
- Frontend: `@tiptap/*` 3.20.1, `esbuild` 0.27.3, `vitest` 3.2.4, `jsdom`
  26.1.0, `typescript` 5.9.3, pinned by the feature `package.json` /
  `package-lock.json`. These declarations are newly exposed source metadata in
  this overlay; no dependency install was performed and the npm tree is not
  vendored.
- Vendored source: Hyperaudio (AGPL-3.0 default, with a non-commercial-only MIT
  note), Mawe/Moys subtitle operations (AGPL-3.0), and tiptap-track-changes (a
  modified local source derived from the 0.2.1 upstream base, MIT with original
  copyright retained). See `NOTICE` and `THIRD_PARTY_NOTICES.md`; all retained
  license texts must stay beside the code.

## Source build (documented, not executed)

Run inside `src/video_knowledge_pipeline/static/targeted-transcript-editor/`
after an authorized install:

```sh
npm install                       # uses the committed package-lock.json
node build.mjs                    # editor.bundle.js + CSS + sourcemap
node build-export-naming.mjs      # export-naming-cli.mjs + export-naming-browser.js
node build-punctuation-cleanup.mjs
npx vitest run --maxWorkers=2     # jsdom test suite
```

These commands were **not executed** for this overlay, no dependency install was
performed, and no generated asset was rebuilt. The existing generated
naming/punctuation helpers retain their source/build mapping
(`export-naming-browser.ts` / `export-naming-cli.ts` via
`build-export-naming.mjs`; `punctuation-cleanup-cli.ts` via
`build-punctuation-cleanup.mjs`; the editor entry via `build.mjs`), but rebuild
reproducibility is **unverified**.

## Packaging

`pyproject.toml` keeps the current public Python metadata unchanged and unions
only the accepted package-data patterns for
`static/targeted-transcript-editor/*` at four depths, so the editor template,
generated naming bundle, vendored source and retained licenses can be shipped.
No Python dependency, optional group, version, entrypoint or other project
semantic is changed. The compiled editor bundle itself is a build artifact; it
is not fabricated here.

## Explicit not-validated conditions

- No installed wheel or package build was produced or validated, no `npm
  install` was run, no `node build*.mjs` was run, no `vitest`/jsdom suite was run
  and no full pytest/full-suite result is claimed. Browser/IME behaviour,
  playable-media decode and file playback were not exercised.
- The compiled editor bundle (`editor.bundle.js`, `editor.bundle.css`, maps and
  `editor.bundle.js.LEGAL.txt`) is not present and was not built or checked.
  Until an authorized build runs, the optional transcript editor cannot be served
  successfully; it must fail honestly, the `ASSETS` guard must stay, and missing
  assets are not fabricated.
- The `webui_bridge.py` project-identity merge and the `visual_evidence.py`
  support helper are included as source, but their integration/compat tests are
  **NOTRUN**, not passed, and no web-UI runtime behaviour is claimed.
- The transcript editor's tiptap revision engine is a modified local source
  derived from the 0.2.1 upstream base; it is not the unchanged official release
  and no npm, compiled, binary or blanket legal clearance is claimed.
- The metadata-writing phase made no runtime, build, package, network,
  loopback, npm, browser or model call. The subsequent limited Python checks
  below do not establish those untested capabilities.

## Test data

This feature's tests use synthetic dictionaries and synthetic bytes only. No real
review pack, font, media, transcript, runtime STATE, private configuration,
history, health data or customer data is included.

## Limited source verification

The assembled source was imported with Python 3.13.5. One command ran eight
unchanged, explicitly selected in-memory tests from `test_targeted_transcript.py`:
normalization/idempotence; source binding; character/time validation; draft-only
status; pack/original drift; document-original preservation; insertions; and
required revision explanations. All eight passed, with no failure or skip.
Six additional Python emitter checks covered explicit/empty identity dispatch,
stale pack rejection, script escaping and retained UI emitters. They did not
execute JavaScript or a browser. Twenty selected Python files parsed as AST.

All supplied original tests remain included. The other tests, filesystem draft
persistence/concurrency, npm/vitest, browser/IME, installed packaging and actual
media/server/provider execution remain unverified. An earlier separate pacing
file-suite attempt had zero passes, one permission error and three unrun cases;
it was not rerun or converted into a pass by these in-memory checks. This source
increment supplies no real review decisions, consent, clinical conclusion or
publication permission.
