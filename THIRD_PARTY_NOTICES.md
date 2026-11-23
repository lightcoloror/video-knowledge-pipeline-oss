# Third-party notices

Updated: 2026-08-15 21:30:35 +08:00 by Codex (GPT-5.6 Sol)

VKP's own source is licensed under `AGPL-3.0-only`. That license does not replace the licenses of third-party packages, browser assets, independently installed tools, model weights, datasets, or hosted APIs.

This document is an attribution and boundary record, not legal advice. The public source and retained-license locations are indexed in [public dependency provenance](docs/public-dependency-provenance.md). The separate historical engineering ledger is not distributed in this public edition.

## Distributed in this repository

| Component | Version / source | License | Distribution boundary |
|---|---|---|---|
| WaveSurfer.js and Regions plugin | 7.12.11, `katspaugh/wavesurfer.js` | BSD-3-Clause | Minified browser assets are vendored under `src/video_knowledge_pipeline/static/wavesurfer-7.12.11/`; the full upstream license is retained beside them. |
| RapidOCR detector resize arithmetic | `RapidAI/RapidOCR`, commit `7b2d368ac62f635094d2562a9e973f7b5eec083f`, `python/rapidocr/ch_ppocr_det/utils.py` | Apache-2.0 | Only the small resize arithmetic is adapted in `courseware_detector_budget.py` for an allocation-free preflight; source attribution and modification notice are in that module. The upstream license is retained at `LICENSES/RapidOCR-Apache-2.0.txt`. Detector/OCR runtime and model weights are not bundled. |
| Moyf/moys-asr-workflow web editor | v1.3.1, `949bc84058cdae1d9c021c50203e6d2742f9392c` | AGPL-3.0-only | Eight root `web/` assets are vendored under `src/video_knowledge_pipeline/static/moys-subtitle-editor/`. VKP adds two explicit adapter slots and separate VKP CSS/JS; launcher, provider/API-key UI, server, model code, and FFmpeg orchestration are excluded. |

No model weights, customer media, production transcripts, or evaluation datasets are distributed in this repository.

## Declared Python dependencies

| Component | Constraint / fixed source | License | Role |
|---|---|---|---|
| jieba | `>=0.42.1,<1` | MIT | Deterministic Chinese tokenization. |
| jsonschema | `>=4.25,<5` | MIT | Local JSON Schema validation. |
| markdown-it-py | `>=3.0` | MIT | Markdown parsing for local review/evaluation. |
| RapidFuzz | `>=3.14.5,<4` | MIT | Candidate matching and review metrics. |
| MCP Python SDK | `>=1.0.0` optional extra | MIT | Optional MCP front door. |
| LiteLLM Proxy | `>=1.81.7,<2` optional extra | MIT for the open-source distribution used by VKP; LiteLLM enterprise files/features have separate terms | Loopback model gateway. VKP does not redistribute LiteLLM or enable enterprise-only code. |
| ruptures | `1.1.10` optional local extra | BSD-2-Clause | PELT change-point candidate generation. |
| pyannote.metrics | `>=4.1,<5` optional evaluation extra | MIT | Speaker diarization evaluation only. |
| MeetEval | commit `184ff17eb77fd6db4aba27a9e303a6a3edb09364` optional evaluation extra | MIT | Speaker-attributed transcript metrics only. |

The portable developer workflow also uses independently installed tooling. `astral-sh/uv` (source-reviewed commit `f1a42680ff5272232d65748acf338b19778dde24`, Apache-2.0 OR MIT) provides the isolated Python environment/bootstrap contract. `go-task/task` (source-reviewed commit `1868ad29698bb336ba76f54bbd9d711c2fa08e8d`, MIT) interprets the local-only `Taskfile.yml`. Neither tool is vendored into the VKP wheel; the Taskfile contains no remote includes and the offline tasks do not use Task's insecure remote feature.

The package manager installs these dependencies separately. Their own distributions contain the authoritative license text and transitive notices.

## Reviewed or adapted upstream designs

The following are not vendored source trees. VKP either calls an independently managed runtime, consumes a saved-output contract, or independently adapts a bounded algorithm/contract. Fixed revisions and integration categories are recorded below. Detailed historical engineering notes are outside this public edition and are not implied to be public evidence.

| Upstream | Fixed evidence | License verified from the pinned local source | VKP boundary |
|---|---|---|---|
| ModelScope FunASR | `516c4f770496a5cbb89c8e2e447211bbb7b0db71`; reviewed release `16cd165ac3946cc8c08bf845331f91fefec8e1a9` | MIT | Independently installed ASR/VAD/punctuation/speaker runtime; no weights bundled. |
| OpenMOSS MOSS-Transcribe-Diarize | `eda4b9f13f1574765a80438c9797780a9bd48112` | Apache-2.0 | Optional CLI adapter; no runtime or weights bundled. |
| Qwen3-ASR | `7c6daf77a2421100f5fb066495372c00129d39ff` | Apache-2.0 | Optional independently installed Cantonese/forced-alignment runtime; VKP reuses the public `language` and `context` interfaces. No source tree or weights are bundled. |
| lightcoloror/model-provider-gateway | public provenance baseline `6575c8089d5f28ae159a0c01cedfa3320db7d7b3`; runtime compatibility range `>=0.3,<0.4` | AGPL-3.0-only | Optional shared profile/route/manifest/gate/receipt package. Runtime admission uses adapter Schema, capabilities, consent and safety semantics rather than an unrelated exact repository HEAD. VKP does not bundle credentials, models, vectors, or provider data. |
| SYSTRAN faster-whisper | `ed9a06cd89a93e47838f564998a6c09b655d7f43` | MIT | Optional local VAD/quality evidence; not a bundled model. |
| OpenAI Whisper | `04f449b8a437f1bbd3dba5c9f826aca972e7709a` | MIT | Quality-heuristic reference/adaptation only. |
| zarazhangrui/YouTube Digest | v1.1.5, `d03e1f61e017b032159ffd1821cac6e7693ce0c7` | MIT | Design reference for stable-ID bilingual transcript batching, viewport-lazy translation, timestamp notes, and time-coverage gates. No upstream browser-extension or Provider source is copied; Supadata, fixed DeepSeek, Key UI, and Chrome integration are excluded. |
| Subtitle Edit | `1517bb5c23e1c4072ea829edbc8d08e27cf79289` | MIT | Silence-boundary chunking semantics adapted into VKP's existing manifest. |
| UFAL SimulStreaming | `077ea37d5ab4ff98bc567e4507f140dc4e5d5ad6` | MIT (`LICENCE.txt`) | Longest-common-prefix agreement contract, candidate evidence only. |
| UFAL whisper_streaming | `6da90b44b7e50d79695e68166d2a2c7609c75abb` | MIT | Timestamped overlap-evidence contract only. |
| AutoShot | `77c82ff826a9301bb173d9be786297a49d73d081` | MIT | Optional saved predictions/GPU candidate route; no checkpoint bundled. |
| TransNetV2 | `85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed` | MIT | Optional saved-prediction candidate only. |
| Shot2Story | `ae26ac3d2f9e9a91a7fd0653bfb6a2b3cb250308` | Apache-2.0 for code per upstream README; text annotations are CC BY-NC-SA 4.0 | Task/evaluation architecture reference only; code, dataset, annotations, and model are not included. |
| NarratoAI | `0a5dcf5f21f7f40ca77bc38ea6d1d3fd52e32c26` | MIT | Two-pass FFmpeg loudness-normalization design adapted behind candidate-only gates. |
| ll-video-decomposer | `8b4d57ce0dc8475751c372c8dc49c1088cee1e69` | MIT | Evidence/report contract adapted as a read-only derived projection. |
| pyannote.metrics | `e8000509ee06331ef3e0fec08fa3605af834efbb` | MIT | Optional local evaluation dependency. |
| RapidFuzz | `edf9f3c2d016c878dae1511301f8b4a501bba871` | MIT | Direct package use for review metrics. |
| jieba | `1e20c89b66f56c9301b0feed211733ffaa1bd72a` | MIT | Direct package use for deterministic tokenization. |

## Reference-only and rejected code

- `WEIFENG2333/VideoCaptioner` at `95842ecb5618c0b6a548a336bdfb0eb859bdb501` is GPL-3.0. VKP reviewed its architecture but copied no source; it is not a runtime or dependency.
- Projects with absent, proprietary, conflicting, or unverified licenses remain reference-only or rejected in the separate historical engineering notes. Their code must not be copied into VKP without a new license review.
- AGPL licensing of VKP does not convert a reference-only checkout, independent executable, hosted API, model, or dataset into VKP-owned code.

## Models and datasets

Model code licenses and model-weight licenses are separate. VKP does not grant rights to SenseVoice, CAM++, MOSS, Qwen, Gemini, DeepSeek, OCR, embedding, tagging, highlight-detection, or other weights/services. Operators must review the exact model card, provider terms, acceptable-use policy, and territorial/data-processing rules before use.

Dataset annotations and media also retain their own licenses and consent requirements. AutoShot, ClipShots, MovieNet, Video-MME, SlideVQA, internal course videos, and customer recordings are not included. Tests in this repository use synthetic or de-identified fixtures only.


The public context-adapter integration test adapts the synthetic predecessor/
purpose fixture pattern from first-party mvp-video-pipeline-oss v0.8.1
(commit 9f55b12864df642f08fcbd96df15acb0f97652d5), Apache-2.0.
The license copy is LICENSES/MVP-test-fixture-Apache-2.0.txt. Source snapshots,
public adaptations and test changes are identified in CONTEXT-ADAPTER-PROVENANCE.json.
No external source history, actual media or private workflow receipt is included.

## Targeted review / transcript editor additions

These components are part of the optional targeted-review feature and are
distributed as source with their retained license texts under
`src/video_knowledge_pipeline/static/targeted-transcript-editor/vendor/`.

| Component | Source identity | License | Distribution boundary |
|---|---|---|---|
| Hyperaudio kept-section / audio-cut algorithm | Adapted from the Hyperaudio Project `editor-audio-cut.js`, upstream base commit `b8bd4728f6d6914a37350d119cf52d4c29eb9968` as recorded in the source comment | AGPL-3.0 default (`vendor/hyperaudio/LICENSE`); an MIT text is also retained (`LICENSE-MIT`) but `vendor/hyperaudio/LICENSES` limits that offer to non-commercial entities | Only the bounded kept-section/playback skipping logic is adapted; original license texts and the modification note are retained. The MIT text is not promoted to a general MIT permission. |
| Mawe / Moys subtitle operations | Adapted from `Moyf/moys-asr-workflow`, upstream base commit `949bc84058cdae1d9c021c50203e6d2742f9392c` as recorded in the source comment | AGPL-3.0 (`vendor/mawe/LICENSE`) | Small join/merge/split primitives only; punctuation is preserved and the upstream proportional-time fallback is omitted. |
| tiptap-track-changes | `sungkhum/tiptap-track-changes` 0.2.1 base commit `25ca36690db5b4977e6d5db5911f9d1eebf6485f`, MIT (`vendor/tiptap-track-changes/LICENSE`, `package.json`) | MIT, Copyright (c) 2025 sungkhum | A modified local source: the vendored tree retains the upstream structure but three source files (`src/commands.ts`, `src/suggest-mode-plugin.ts`, `src/types.ts`) differ from that base, so it is not a byte-identical official release. VKP's adaptations add non-history mode-change transactions, view-mode editability, storage typing/callback and unused-name adjustments. Original copyright and MIT text are retained. |

Frontend npm packages (`@tiptap/*` 3.20.1, `esbuild` 0.27.3, `vitest` 3.2.4,
`jsdom` 26.1.0, `typescript` 5.9.3 and their transitive tree) are declared and
locked but are not vendored into this repository. The declarations are newly
exposed source metadata; no dependency install was performed for this overlay.
The lockfile is metadata, not an installed-license or reproducible-build proof.
The compiled editor bundle and maps are not asserted to be present, and
legal-comment output from a build was not produced or validated in this source
review.

The targeted-review UI references a locally installed font family (`YouYuan` /
`幼圆`) by CSS `local(...)`; no font binary is distributed. No model weights,
media, real review packs, transcripts, private configuration, runtime STATE,
history or customer data are added by this feature.

This section was appended as a public-source curation on 2026-11-10. That date
is the curation date only; it is not a claimed original modification date and it
asserts no new authorship.
