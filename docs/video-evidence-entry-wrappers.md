# Video Evidence PowerShell Entry Wrappers

`docs/video-evidence-entry-wrappers.md` documents three optional source-checkout PowerShell entry wrappers under `scripts/`, licensed under AGPL-3.0 as part of the Video Knowledge Pipeline (VKP). These wrappers forward command-line arguments to existing public backend Python modules:

- `scripts/ocr-route.ps1` -> `video_knowledge_pipeline.ocr_route`
- `scripts/speaker-diarization-evidence.ps1` -> `video_knowledge_pipeline.speaker_diarization_evidence`
- `scripts/video-knowledge-local-media-contract.ps1` -> `video_knowledge_pipeline.local_media_contracts`

## Runtime Environment and Invocation

The wrappers execute `python` from the operator's existing `PATH`, overwrite process `PYTHONPATH` with the local repository `src/` directory, set `PYTHONIOENCODING=utf-8`, and preserve native `$LASTEXITCODE`. They do not isolate environments, select Python interpreters, or bypass system execution policies. If PowerShell script execution is restricted by environment policy, do not bypass or disable security controls.

Inspect the backend CLI usage through `--help` from a dependency-ready source checkout:

```powershell
.\scripts\ocr-route.ps1 --help
.\scripts\speaker-diarization-evidence.ps1 --help
.\scripts\video-knowledge-local-media-contract.ps1 --help
```

## Backend Capabilities

- **OCR Routing** (`ocr_route`): Dispatches item processing with positional `bundle_dir`, `--backend`, `--execute-local`, `--connector-result-json`, `--provider-config`, `--indexes`, `--limit`, and `--timeout-seconds`.
- **Speaker Diarization Evidence** (`speaker_diarization_evidence`): Generates candidate evidence using `plan` and `run` subcommands (`run` accepts `plan_json`, `--execute`, `--timeout-seconds`, `--min-coverage-ratio`, `--min-dominance-ratio`, `--no-write`). Produces candidate evidence, not identity validation.
- **Local Media Contracts** (`local_media_contracts`): Builds/checks declared receipt and import metadata via `speech-receipt`, `ffmpeg-receipt`, `rough-cut-import`, and `provider-status`. Requires `--request` (except `provider-status`) with opt-in `--write`. Declared metadata checks do not prove empirical tool execution.

Existing backend candidate-evidence regression test source is in `tests/test_speaker_diarization_evidence.py`. That is separate from this wrapper batch; it was not rerun here, and the wrappers do not certify historical test results.

## Operational Boundaries

Resolving repository roots with `-LiteralPath` avoids wildcard misinterpretation within the wrappers, but does not sandbox the pipeline or guarantee filesystem portability. Backend commands may write local plans, receipts, or import artifacts; they are not read-only. No machine learning models, media files, or binary runtimes are distributed. Owner checks under PowerShell 5.1.26100.4652 Desktop: three nonexecuting ParseInput checks reported no syntax errors; all 17 other top-level statements remained identical. The only code change is LiteralPath root resolution, plus one repaired documentation comment. None of the three wrappers, Python modules, OCR/ASR/ONNX tools, native argument forwarding, or wrapper runtime behavior under PowerShell 5.1 were exercised by these checks. Those behaviors remain unverified.

These thin launchers do not grant provider consent, execution or publication authority. Use the existing backend rules and inspect candidate results. External dependencies and model files are operator-supplied; this source batch bundles or installs none of them.
