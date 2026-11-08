# Public dependency provenance

The public edition retains source credits, upstream revisions, distribution
boundaries and license copies. Its [third-party notices](../THIRD_PARTY_NOTICES.md)
separate vendored/adapted code from optional external runtimes and reference-only
designs. The historical engineering ledger is not bundled; this index does not
claim to reproduce that ledger or newly verify all upstream checkouts.

## Public source and notice locations

| Distributed or adapted component | Retained public evidence |
| --- | --- |
| WaveSurfer.js 7.12.11 and Regions | [Upstream BSD license](../src/video_knowledge_pipeline/static/wavesurfer-7.12.11/LICENSE), beside the vendored assets |
| Moyf/moys-asr-workflow editor subset | [Pinned upstream and modification record](../src/video_knowledge_pipeline/static/moys-subtitle-editor/UPSTREAM.md); excluded runtime/server/provider components remain excluded |
| RapidOCR resize arithmetic | [Apache-2.0 license copy](../LICENSES/RapidOCR-Apache-2.0.txt); the notices give the exact upstream source file and revision |
| Public context-adapter test fixture | [First-party source/adaptation record](../CONTEXT-ADAPTER-PROVENANCE.json) and [retained Apache-2.0 license](../LICENSES/MVP-test-fixture-Apache-2.0.txt) |
| Subtitle coverage adaptation | [Source and synthetic-test record](../SUBTITLE-COVERAGE-PROVENANCE.json) |
| VKP UI shell | [Source/projection metadata](../src/video_knowledge_pipeline/static/vkp-ui-shell/PROVENANCE.json) |

The fixed revisions in the notices are historical source-review evidence, not
proof that a reader has installed the same runtime or has permission to process
particular media. Models, weights, external APIs, datasets and independently
installed packages retain their own terms. No additional source, model, font,
binary, customer media or private engineering notes are added by this repair.

The [AGPL project license](../LICENSE) covers the project's own code within its
stated scope; it does not replace those upstream terms. [README](../README.md)
documents the supported source edition. Offline/synthetic tests are separate
from actual ASR/OCR, evidence authenticity, consent and publication acceptance.

Added 2026-11-08 by Codex (exact runtime model not independently verified), using
the already published notices and retained files only; no new upstream audit.
