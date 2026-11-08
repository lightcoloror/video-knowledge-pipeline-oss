# Offline fixture tools

Two optional local scripts build and audit small synthetic offline acceptance
fixtures for VKP. They reuse the project's existing fixture-contract and
frame-manifest shapes, add no runtime dependency and do not change the
installer, CLI, model gateways or any existing module. The third supplied
offline-test runner is intentionally not adapted or distributed here; see
[`OFFLINE_FIXTURE_TOOLS_PROVENANCE.json`](../OFFLINE_FIXTURE_TOOLS_PROVENANCE.json).

## What these tools are not

- Default media is a **first-party generated PCM tone**, not speech. A tone is
  not an ASR sample and establishes no ASR, speaker, caption or speech-quality
  result.
- The scripts do not create normalized transcripts, OCR/visual evidence, human
  reviews, knowledge notes or `review.html` outputs. The fixture contract lists
  those downstream artifacts only as **targets that are not proven outputs**.
- A SHA-256 value checks the bytes that were read. It does not prove source
  authenticity, factual truth, licensing or speech correctness.
- The frame-manifest exporter does not run `ffmpeg` and cannot prove which
  process created a frame. Extractor provenance is reported as unknown.
- A successful fixture round-trip is not the full 11-repository workflow.
- Thread limits, timeouts and local-file protocol requests are **argv settings**.
  They are not measured OS thread/CPU/RAM peaks and they do not prove that every
  network path is blocked.

No audio, font, binary, codec, model weight or real media file is bundled.

## Provenance and license

Both adapted scripts are public copies of first-party VKP source under
AGPL-3.0-only. Fixed commit, blob, byte and SHA-256 evidence is recorded in
`OFFLINE_FIXTURE_TOOLS_PROVENANCE.json`: project source commit
`d8541dc19247e37e4d6bc6a3a1e1689a2a98b6f0` on public parent
`e272209e161d5480a3510d829100ab8ae4f31c59`.

The original upstream reference audio paths used by the private engineering
snapshot (transcribe-critic, FunASR and Docling test files) are **unverified
rights, not licensed fixtures**. They are not copied, not read by these tools and
not required by any default path.

## Fixture builder

```sh
python scripts/build-offline-acceptance-fixtures.py OUTPUT_DIR
```

`OUTPUT_DIR` must be a fresh explicit directory owned by this run. A nonempty
existing directory is refused, never cleaned or overwritten.

Output layout:

```text
OUTPUT_DIR/
  fixture-contract.json            # video_knowledge_pipeline.offline_fixture_set.v1
  generated-audio/                 # first-party PCM tone WAVs for tone fixtures
  talking-head-synthetic/
  slides-screen-synthetic/
  mixed-visual-synthetic/          # cards + a tiny MP4 each
```

The original three fixture IDs and types are preserved:
`talking-head-synthetic` (`talking_video`), `slides-screen-synthetic`
(`slides_or_screen_video`) and `mixed-visual-synthetic`
(`mixed_visual_video`).

Options:

| Option | Meaning |
|---|---|
| `--tone-seconds` | Generated tone duration per tone fixture (default 2 s; bounded 0.25–5 s). |
| `--tone-frequency` | Generated tone frequency in Hz (default 440; bounded 20–4000 Hz). |
| `--audio-spec PATH` | Optional operator-declared local speech media JSON. |
| `--card-font PATH` | Optional explicitly licensed local font file. |
| `--card-language {ascii,zh}` | Card text language. Default `ascii`; `zh` restores the original Chinese card text and requires `--card-font`. |

Tone duration and frequency are validated against those small documented fixture
bounds **before** any output, audio probe or sample allocation. This prevents an
arbitrarily large finite setting from exhausting memory. Reported durations use
0.001 s precision; a duration that would round to zero is rejected rather than
silently returned as zero.

Default cards are synthetic ASCII and the default font is Pillow's bundled
default; the default path never searches host font directories. Passing
`--card-font` is the only way the builder opens a font file, and the file must be
a regular local file supplied by the operator. The original Chinese card text is
retained as an explicit optional `--card-language zh` mode; it requires an
operator-supplied font and is refused otherwise. No host font is read.

### Operator speech media specification

```json
{
  "audio": [
    {
      "fixture_id": "talking-head-synthetic",
      "path": "D:/operator-approved/speech.wav",
      "expected_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "operator_provenance": "recorded for this acceptance run by the operator",
      "use_rights_statement": "operator declares local use rights for this file"
    }
  ]
}
```

Every entry is validated before any output is written or rendered:

1. shapes, known original fixture ID, no duplicates, nonempty local path,
   64-hex expected SHA-256, nonempty provenance and use-rights statement, and a
   regular local file for **all** entries;
2. SHA-256 of **all** declared files must match their declarations;
3. duration probing for **all** declared files must be finite and positive.

A defect in a later entry therefore fails before earlier media is probed or
rendered. Declared provenance and use rights are recorded in the manifest but are
not independently verified; the declaration is not a license audit, and it does
not prove speech correctness. Fixtures without a declaration keep the generated
tone. No upstream path is discovered or looked up automatically.

The manifest labels, per fixture: `speech_reference` (`none` for tone),
`asr_reference` (`none`), quality reference (false), media/timing/hash scope,
`expected_artifacts_status` (`targets_not_proven_outputs`), and explicit false
claims for transcript/OCR/review/knowledge-note creation.

The top-level `generated_media` aggregate is derived from the fixtures that were
actually built: card language/font, audio kinds, generated-tone and
operator-declared counts, and the actual per-fixture audio durations. Mixed or
all-operator runs therefore do not claim a tone-only set. In an all-operator run
the tone fields are `null`. `operator_supplied_local_inputs` is true only when
operator audio or a font was actually supplied. `speech_reference` states
`none`, all-operator, or a mixed count without claiming verification.

### FFmpeg and ffprobe handling

The builder uses local FFmpeg/ffprobe only. Its FFmpeg argv requests per-input
decoder threads, encoder threads, `-filter_threads` and
`-filter_complex_threads` values of 2, with bounded timeouts (120 s FFmpeg, 60 s
ffprobe) and one media child at a time. These are argv requests, not measured OS
thread/CPU/RAM peaks.

ffprobe is invoked with options supported by the supplied build: it does not use
`-nostdin`, its stdin is closed with `subprocess.DEVNULL`, and its input protocol
is restricted with `-protocol_whitelist file`. That is a local-file request, not
proof of network blocking.

Concatenation is expressed with generated relative card basenames such as
`file 'card-01.png'` under `-safe 1`, so arbitrary absolute or quoted paths are
not written into the ffconcat file. Commands are argument lists (never shell
strings) and media inputs request the local `file` protocol only. A missing
FFmpeg/ffprobe is an error. Failures leave partial output as evidence and exit
nonzero; the tool never prints `ok` for a failed build.

## Frame-manifest exporter

```sh
python scripts/export-offline-frame-manifest.py BUNDLE_DIR
```

The bundle must contain `timeline.json` (a JSON array) and may contain
`manifest.json` (a JSON object). Both fixed members are resolved against the
bundle root, required to stay inside the resolved root and required to be regular
files **before any byte is read**. An outside-root input symlink or junction is
refused before `_read`; a directory, FIFO or other non-regular member is refused
before `open`; an absent optional `manifest.json` remains allowed. Resolution,
`stat` and read errors such as `PermissionError`/`EACCES`/`EPERM` propagate and
stop the export instead of becoming a missing or blocked row. The exporter writes
`BUNDLE_DIR/frame-manifest.json` using the original
`video_knowledge_pipeline.frame_manifest.v1` fields and the original row fields
(`timeline_index`, `start_seconds`, `end_seconds`, `role`, `path`, `exists`,
`sha256`, `evidence_status`), plus audit fields for blocked reasons, occurrences
and counts.

Reference handling:

- Empty or non-string references are invalid; an empty reference is never
  resolved to the bundle directory.
- Relative references resolve against the bundle root, never the ambient working
  directory.
- Every frame reference must stay inside the resolved bundle root. `..`
  traversal, absolute outside-root, foreign-drive/UNC and resolved symlink
  escapes are reported as `blocked` before any byte is read or hashed.
- Only regular files are hashed. Directories and other non-regular paths are
  `blocked`; missing paths are `missing`. Empty, missing, blocked and directory
  cases stay visible in `frames`, `incomplete_locations` and the counts.
- Deduplication is case-sensitive: `frames/A.PNG` and `frames/a.png` remain
  distinct rows. When one file appears in several timeline entries, one row keeps
  the first primary context and records every occurrence (timeline index and
  start/end) in `occurrences`.
- A timeline with no frame references, a malformed timeline/manifest shape or
  malformed JSON fails before any write and exits 2.
- An incomplete export still writes the visible manifest and exits 1 with
  `ok: false`; it never reports full integrity.

Only genuinely malformed input is converted to an input error. `PermissionError`
and other runtime/filesystem errors propagate and stop before later probes or
writes; they are not hidden as blocked or missing rows. The exporter's file
probes use an explicit `stat` and ignore only not-found/not-a-directory cases.
The fixed `timeline.json`/`manifest.json` members are resolved, contained and
checked as regular files before their first byte read.

Source media:

- The original `timeline.json` `video_key` first, then `manifest.json`
  `media_path` / `source_media_path` / `video_path` selection order is retained.
  Relative values resolve against the bundle root.
- Only regular resolved paths contained by the bundle root are read and hashed.
  Outside, foreign-drive/UNC, traversal and non-regular values are recorded as
  blocked/missing candidates and are never read.
- Outside-bundle source media is read only when the operator passes
  `--source-media PATH` explicitly. That declared file must be a regular local
  file and is hashed.
- Same-root bundle-reported media is intentionally discovered and hashed without
  `--source-media`; that is contained autodiscovery, not an outside-root read.
  The `operator_boundary` records both capabilities
  (`bundle_root_source_media_autodiscovery` and
  `outside_root_source_media_requires_explicit_operator_path`) and the actual
  `source_media_authorization` used by this export
  (`bundle_root_contained_autodiscovery`, `explicit_operator_path_inside_bundle_root`,
  `explicit_operator_path_outside_bundle_root`, or `none`). The same string is
  recorded under `source_media.authorization`. No field claims that all source
  media reads are explicit-only.
- Frame-only bundles need no source-media declaration; discovery is optional.
- This exporter does not probe `ffmpeg`. `extractor` is recorded as
  `unknown`/`not_probed`; a version probe would not prove frame extraction
  provenance anyway.

Output destinations:

- Both intended destinations are validated before any write.
- A symlink destination, a destination that resolves outside the bundle root, a
  non-regular destination, or an unrelated existing `frame-manifest.json`
  (different schema) is refused with no write.
- An existing destination must be a regular file even when its JSON bytes were
  already parsed earlier; parsing is not regular-file proof. The regular-file
  check is unconditional and is not bypassed by a "known valid" flag.
- A pre-existing owned frame manifest with the same schema is a legitimate
  refresh and is replaced.
- `manifest.json` is unchanged by default. `--register` explicitly updates its
  `frame_manifest` pointer while preserving other keys, and only for a complete
  export.
- A pure `build_frame_manifest` result reports
  `output_destinations_validated: false` with
  `output_destinations_validation: not_checked_pure_computation`; destination
  checks happen in the writing path. Only after those checks succeed does the
  written manifest report `output_destinations_validated: true` (with
  `validated_before_write`) and the queued `frame-manifest.json` write proceeds.
- These checks reduce accidental clobbering but are not a TOCTOU or OS-sandbox
  guarantee; a concurrent filesystem change remains a known limitation. Hard
  links and junction/reparse-point races are not proven here.

Exit codes: `0` complete, `1` incomplete (missing/blocked frames), `2` invalid
input/unsafe destination.

## Reproducible source-checkout example

```sh
python examples/offline_fixture_roundtrip.py OUT_DIR
```

The example refuses a nonempty `OUT_DIR`, builds all three tiny tone/card MP4
fixtures through the builder, decodes at least one real frame per generated MP4
with local FFmpeg into `OUT_DIR/bundle/frames`, writes a minimal synthetic
`timeline.json` referencing those decoded frames relative to the bundle root,
and exports `frame-manifest.json` through the same exporter.

The example's FFmpeg argv requests decoder, encoder and filter thread limits of 2
with a 120 s per-child timeout and closes stdin; these are argv requests, not
measured peaks. Its decode input requests the local `file` protocol only.

`roundtrip-report.json` uses explicit bases (`output_root`, `fixture_root`,
`bundle_root`): every media/frame/artifact path is reported relative to a
documented base, and the example re-resolves and re-hashes each reported
reference before writing the report. The bundle's small `manifest.json` does not
invent a schema and does not claim the sibling `fixtures/fixture-contract.json`
is inside the bundle. No SRT/ASR captions, OCR output or knowledge notes are
produced. Reproduce from a source checkout with:

```sh
python -m venv .venv
# Activate this environment using your shell's activation command.
python -m pip install ".[test]"
python examples/offline_fixture_roundtrip.py .local/offline-fixture-roundtrip
```

Local FFmpeg and ffprobe with `libx264`/AAC support are required for the example
and the builder. No download, service, model or network call is made.

## Tests

`tests/test_offline_fixture_tools_synthetic.py` is a stdlib `unittest` suite. It
exercises the real builder/exporter validation, containment, source-selection,
row, hash, aggregate and manifest logic with in-memory fakes: no real
FFmpeg/ffprobe/subprocess, socket, DNS, temporary files or directory fixtures,
and no whole-package imports. Run it from the candidate root:

```sh
python -m unittest discover -s tests -p "test_offline_fixture_tools_synthetic.py" -v
```

Coverage includes default tone/no-host-font behavior and optional Chinese cards
with a supplied font; tone bounds and rounded-zero duration rejection; declared
audio/hash preflight before any write; an invalid second declaration; ffprobe
argv without `-nostdin` plus DEVNULL stdin and file-protocol requests;
decoder/encoder/filter thread argv; safe concat names; aggregate provenance for
mixed and all-operator runs; injected `EACCES`/`EPERM` propagation; malformed and
empty bundles; missing/blocked/directory/empty references;
traversal/absolute/foreign-drive/UNC/resolved-escape containment; restored
same-root source-media discovery and outside-root non-reading; relative-root
resolution; exact hash/row/occurrence counts; case-sensitive distinct paths;
output-destination validation with mocked symlink/escape/unrelated cases;
explicit source media; opt-in registration and no input mutation.

Additional regression coverage for the pre-read input checks and destination
metadata: an outside-root `timeline.json`/`manifest.json` resolved link fails
before the corresponding read; a non-regular required or optional bundle member
is refused before `open`; an absent optional `manifest.json` is still accepted; a
contained resolved regular member is read through its resolved path; injected
`PermissionError` from input resolution or `stat` propagates instead of becoming
missing; a non-regular existing destination is refused even with
`known_valid_existing=True`; both destinations are validated before the first
write; and pure results report destination validation as not checked while the
writing path reports the validated status before the write.

Source-media authorization coverage: same-root autodiscovery reports
`bundle_root_contained_autodiscovery` while both capability fields stay true and
no explicit-only claim exists; explicit outside/inside operator paths report their
own authorization; and a frame-only bundle reports `none`.

The suite's synthetic roots are built from the native absolute anchor
(`Path("C:/")` on Windows, `Path("/")` on POSIX) so the same in-memory fakes and
expected strings are platform-portable. Windows drive, drive-relative and UNC
strings remain real lexical-validator cases; a separate pair of tests exercises
`PurePosixPath`/`PureWindowsPath` lexical forms. This is source-portability
coverage, not a claim that the suite ran on Linux.

The suite also supports a controlled confinement mutant: deleting **only** the
resolved frame-root containment rejection must make the plain outside-root
reference test fail under the same suite, with no real file read. The exact
mutant is confirmed separately by the Owner; this document does not claim a
mutant run in the public tree.

## Staged verification and limits

1. Source/static stage: syntax/AST/compile checks, hashes and source review only;
   no candidate module import, test, example or media execution.
2. Synthetic unit stage (separately admitted): run the pure unittest suite and
   the confinement mutant above; no real media or process execution.
3. Media round-trip stage (separately admitted): run the public example on fresh
   output with constrained local FFmpeg/ffprobe, then inspect the generated MP4
   streams/durations, decoded PNG hashes and exported frame manifest.

Known limits: no transaction or TOCTOU guarantee is claimed between validation
and use; symlink/junction races and OS sandbox guarantees are not proven by these
tools; thread/timeout/file-protocol argv settings are not measured resource or
network proof; hash checks do not establish authenticity; the generated tone is
not speech and the ASCII/Chinese cards are not OCR ground truth; and a completed
round-trip remains a fixture self-check, not an ASR/OCR/model/business result or
the full workflow.
