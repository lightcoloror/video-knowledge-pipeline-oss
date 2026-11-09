# Creator Report Player Offline Test Harness

`docs/report-player-offline-harness.md` documents the offline Node.js CommonJS test harness in `tests/creator_report_player_harness.cjs`, licensed under AGPL-3.0-only as part of the Video Knowledge Pipeline (VKP). The harness uses Node.js standard built-ins (`node:fs`, `node:vm`, `node:assert/strict`) exclusively, requiring no external packages, browsers, or network services.

## Invocation and Architecture

The harness accepts a path to trusted player JavaScript. For this acceptance, the Owner extracted the exact `_SCRIPT` string from the existing public `src/video_knowledge_pipeline/creator_report_interactive.py` via AST `literal_eval`, without importing or rendering that Python module. Pass the resulting trusted script path:

```bash
node tests/creator_report_player_harness.cjs "path/to/extracted-player.js"
```

The existing public Python test `test_player_actual_javascript_events_without_browser` imports `_SCRIPT`, writes it to a temporary `player.js`, invokes this harness and requires `player behavior assertions passed` with exit 0. That full Python test module and its broader dependencies were not executed in this acceptance.

## Event Simulation and Assertions

Using in-memory DOM and media stubs, the existing assertions cover these cases:

- Initial seek guard before media loading
- Non-media MIME type rejection
- Metadata loading with unverified identity notices
- Valid timestamp seeking and play promise resolution
- Out-of-bounds duration and NaN seek rejections
- Autoplay refusal handling with manual action prompts
- Decoder error notifications
- Object URL cleanup on reselection and `window.pagehide`
- Infinite or missing duration handling
- Empty file selection fallbacks

Owner verification used existing Node v22.17.1: `--check` passed, and the unchanged harness passed against the byte-identical public `_SCRIPT`. A missing-harness baseline returned MODULE_NOT_FOUND. A separate bad script disabled only the over-duration guard; the harness then failed with ERR_ASSERTION as expected. Neither bad copy nor generated script is distributed in this batch.

## Verification Boundaries

`vm.runInNewContext` executes trusted code against stubs; it is not a security sandbox and must not evaluate untrusted input. Mock files and URL objects do not decode video streams, verify file authenticity, or enforce browser Content Security Policies. Full browser rendering and end-to-end playback remain unverified.
