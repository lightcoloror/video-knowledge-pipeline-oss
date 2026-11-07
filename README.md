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
