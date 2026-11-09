# Video Knowledge Pipeline

Local-first video transcription, timestamped evidence, material manifests and
review tooling. The public source edition retains AGPL-3.0-only and the
component notices. It includes code and synthetic tests. Model weights,
third-party runtimes, private captures and machine configuration are excluded.

## Install and run the synthetic contract

```sh
git clone https://github.com/lightcoloror/video-knowledge-pipeline-oss.git
cd video-knowledge-pipeline-oss
python -m venv .venv
# Activate this environment using your shell's activation command.
python -m pip install ".[test]"
video-knowledge --help
video-knowledge-portable --project-root . doctor
video-knowledge-portable --project-root . smoke --output-dir .local/synthetic-smoke
video-knowledge-portable --project-root . validate-smoke --run-root .local/synthetic-smoke
python -m pytest -q tests/test_portable.py tests/test_quick_health.py tests/test_path_defaults.py
```

The explicit project root supplies the clone's discovery metadata, lock and
synthetic fixture to the installed Python code. The synthetic smoke checks
evidence and material contracts. It does not process real video or establish
ASR, OCR, speaker identification, model quality or publication readiness.

Optional extras configure local, VAD, online, evaluation and shared Gateway
integrations. The Gateway's uv source is pinned to the reviewed public
repository. External executables, providers and model licenses require their
own configuration. The source test collection includes optional integration
and deployment tests beyond the focused portable regression command above.

See [AGENT_DISCOVERY.md](AGENT_DISCOVERY.md), [LICENSE](LICENSE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSES](LICENSES).

## Public portable contract metadata

Version 0.1.1 replaces the old private bundle locator with the public
`codex-local-governance-oss` v0.1.1 distribution. The exact root bundle digest
is pinned in `portable-contract.lock.json`. Supply its local
`contracts/portable/v1/portable-contract-bundle.v1.json` to the existing
`doctor --contract-bundle <file> --require-contract-bundle` command.
No download or execution is performed by this reference. Bundle verification
checks the root metadata bytes and required IDs; it does not establish native
runtime interoperability, evidence truth, author approval or publication.
The bundle's optional native review Owners remain undistributed and blocked;
its frozen intake schema is not a replacement review implementation.

The current public bundle is commit
`ba0bca6632979a35881583d531f4637593e2e66c`; root digest
`cf97adbb27a97bee1fb846954721f671e1a922a2a2f3b7e130f234cdac652994`.
Only package version metadata changes in Python; runtime behavior is unchanged.

## Subtitle coverage contract (0.1.2)

The existing first-party `subtitle_coverage` module is included for the main
workflow's video delivery consumer. It reuses existing interval coverage and
media probing; it does not implement ASR, rewrite subtitles or render media.
Source/output fingerprints and full-timeline human review remain required.
A successful ASR execution or mechanically complete timeline is not a human
coverage pass. Tests use invented source hashes, timestamps and review states.
See SUBTITLE-COVERAGE-PROVENANCE.json for working-snapshot origins and changes.

This public module reports label/quality assertions separately from actual local
file checks. Malformed meaningful words/timestamps with no sentence fallback
raise an error instead of silently dropping speech. A valid sentence fallback
remains available. Reviewers need the complete audit including omissions and
evidence context; a cropped prompt alone does not establish a human review.
Focused checks: 50 passed. Main synthetic composition: 95 passed, one browser
regression not run, 58 subtests. See SUBTITLE-COVERAGE-ACCEPTANCE.md.

## Native context ASR decision adapter (0.1.3)

The existing registered-runtime adapter is now available as installed command
`video-knowledge-context decide-asr` and source shim
`python scripts/context_manifest_consumer.py decide-asr`. Set
`SELF_MEDIA_SYSTEM_ROOT` to a compatible explicit local public main workflow
checkout (tested: `self-media-creation-system-oss` commit
`98a85593f5d306575580af29687a9521a8ae7eb1`). No download, provider or ASR runs.
The installed adapter imports that operator-selected main workflow to reuse
its native handoff validation and decision policy; the main workflow is not
bundled into VKP or silently replaced by a stub.

Supply `--handoff`, `--audio` and `--output`; optionally supply
`--previous-receipt` and repeat `--changed-window START-END`. Native purpose,
previous output and context hashes remain required. The audio argument must
match a hash-bound media artifact in the validated context. Derived audio can
be separately bound with schema role `source_media` and a unique recipe
`source_role`; the original video's required recipe role remains intact.
An unchanged audio hash cannot request a changed-window rerun, and is rejected
before creating a new decision output. This writes a decision receipt, never
an ASR transcript, a human approval or a completed business step.

Eight synthetic tests exercise the actual public main predecessor chain and
purpose/hash gates. Fake audio bytes are used only for hashing. See
`CONTEXT-ADAPTER-PROVENANCE.json` and `CONTEXT-ADAPTER-ACCEPTANCE.md` for the
origin, implementation changes, failed fixture attempts and coverage limits.

## Offline fixture tools (optional local tooling)

Two optional local scripts build and audit synthetic offline acceptance fixtures.
They add no runtime dependency and do not change the installer, command-line
entrypoints, model gateways or any existing module.

- `scripts/build-offline-acceptance-fixtures.py` retains the original fixture
  builder helpers and `video_knowledge_pipeline.offline_fixture_set.v1` contract.
  By default it generates three tiny (2 s) first-party PCM tone tracks with the
  standard library `wave`/`math` modules and draws synthetic ASCII cards with
  Pillow, then muxes them with local FFmpeg. A tone is not speech and establishes
  no ASR quality. Operators can declare local speech media explicitly with
  `--audio-spec`: each declaration supplies a local path, expected SHA-256 and a
  provenance/use-rights statement, and every declaration is validated before any
  output is written or rendered. An explicitly supplied licensed font can be
  passed with `--card-font`; the original Chinese card text is available only as
  an explicit `--card-language zh` mode that requires that font, and the default
  reads no system fonts. Tone duration/frequency are bounded to small documented
  fixture limits before any allocation. No audio, fonts, binaries or model
  weights are bundled.
- `scripts/export-offline-frame-manifest.py` retains the original frame-manifest
  parsing, row and hash semantics and the
  `video_knowledge_pipeline.frame_manifest.v1` schema. Frame references resolve
  against the bundle root and must stay inside the resolved root; traversal,
  absolute-outside, foreign-drive/UNC and symlink escapes are blocked before any
  read. Missing, blocked and non-regular references stay visible and make the
  exit status nonzero. The fixed `timeline.json`/`manifest.json` members are
  resolved, contained and required to be regular files before any byte read, so
  an outside-root link or non-regular member fails before open while an absent
  optional `manifest.json` stays allowed. Source media keeps the original
  `video_key` /
  `media_path` / `source_media_path` / `video_path` selection order, but only
  regular resolved paths contained by the bundle root are read and hashed;
  same-root contained values may be autodiscovered, while outside values stay
  unread unless the operator passes `--source-media` explicitly; the manifest
  records the actual authorization instead of claiming explicit-only reads.
  Permission/runtime errors propagate instead of being hidden, and
  both output destinations are validated before any write so symlink escapes,
  nonregular paths and unrelated existing artifacts are refused; a non-regular
  destination is refused even when its JSON was parsed earlier, and the pure
  builder result reports destination validation as not checked until the writing
  path reports its actual validated status. `manifest.json` is only updated with
  `--register`.
- `examples/offline_fixture_roundtrip.py` is a small source-checkout example that
  builds the three tone/card MP4 fixtures, decodes one real frame per MP4 with
  local FFmpeg, writes a minimal synthetic bundle timeline, and exports the frame
  manifest through the same exporter. Report paths are always relative to a
  documented output-root or bundle-root base, are re-resolved and re-hashed before
  writing, and nonempty output roots are refused.
- `docs/offline-fixture-tools.md` documents CLI usage, output ownership, the
  staged verification plan and the honest boundary: no ASR/OCR, no normalized
  transcript, no human review, no knowledge note and no model-quality proof.
- `OFFLINE_FIXTURE_TOOLS_PROVENANCE.json` records the fixed upstream commit,
  blobs and SHA-256 source evidence for the two adapted scripts, the unmodified
  third runner's held-out disposition and the unverified rights status of
  upstream reference audio.

A completed fixture round-trip is a fixture self-check. It is not speech, ASR,
OCR or model-quality evidence, not a human review or knowledge note, and not the
full 11-repository workflow. The public tree adds no sample media, fonts,
binaries or model weights. Thread, timeout and file-protocol settings are argv
requests, not measured OS resource peaks or proven network blocking.

## Review-export identity (standalone)

The existing first-party `video_knowledge_pipeline.review_export_context` module is
available as a standalone source module. It records an explicit topic/video identity
(`vkp.review-export-context.v1`) through
`python -m video_knowledge_pipeline.review_export_context configure/status` and
through the direct Python API. Identity fields are caller-supplied metadata; they do not prove media ownership
or authenticity.

Focused synthetic checks live in `tests/test_review_export_identity_standalone.py`.
Verification for this delta is source-only: the authoring run stopped at a
temporary-directory permission error, so no passing-suite claim is made.
Synthetic checks are not end-to-end, browser, media, consumer-bridge, editor or
human-review evidence. The root AGPL-3.0-only license governs distribution.

See [the standalone review-export identity guide](docs/review-export-identity.md) for commands, conditional bindings and verification limits.

### Tested-Entry PowerShell Wrapper

The repository includes `scripts/run-tested-vkp.ps1`, an entrypoint guard and forwarder for `src/video_knowledge_pipeline/tested_entry.py`. Under its legacy "tested" name, the wrapper validates software identity and drift (Git HEAD, interpreter SHA256, and managed source snapshots) against an operator-local binding file before dispatching native commands.

The wrapper provides software integrity verification rather than empirical test execution or sandbox isolation. Binding files are operator-local and not distributed. See [docs/tested-entry-wrapper.md](docs/tested-entry-wrapper.md) and [tests/test_tested_entry.py](tests/test_tested_entry.py) for option models, binding lifecycle details, and verification boundaries under AGPL-3.0-only.
