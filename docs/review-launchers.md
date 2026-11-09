# VKP Review Launchers

## Overview

This suite provides three PowerShell entry helpers for generating, refreshing, and opening local review artifacts:

- `scripts/open-review.ps1`: Refreshes or opens an existing bundle’s `review.html` or `multimodal-sample-review.html`.
- `scripts/open-task-console.ps1`: Exports and opens `task-console.html` alongside `review.html`.
- `scripts/open-targeted-review.ps1`: Prepares and serves an interactive loopback review session.

These scripts are first-party source under [AGPL-3.0-only](../LICENSE), adapted from commit `d8541dc19247e37e4d6bc6a3a1e1689a2a98b6f0` with portable `-LiteralPath` and operator-configurable bundle roots.

## 1. `scripts/open-review.ps1`

Opens or refreshes the standard WebUI review bundle.

### Parameters
- `-BundleDir <path>`: Explicit bundle directory containing `manifest.json` and `timeline.json`. When provided, automatic bundle discovery is skipped.
- `-AdditionalBundleRoots <string[]>`: Optional extra search roots scanned when `-BundleDir` is omitted (replaces hardcoded drive paths). Uses `-LiteralPath` and ignores empty entries.
- `-NoRefresh`: Skips running `refresh-review-html` before opening.
- `-SampleReview`: Generates a `multimodal_sample_review` page with `--sample-size <int>` (default 48), regardless of `-NoRefresh`.
- `-NoOpen`: Suppresses launching the default browser. This switch alone does not suppress generation or refresh; those calls depend on the other options.
- `-PrintJson`: Outputs structured status JSON to stdout.

### Example
```powershell
powershell.exe -NoProfile -File scripts/open-review.ps1 `
  -BundleDir "<path-to-webui-bundle>" -NoOpen -PrintJson
```

*Discovery across custom roots:*
```powershell
powershell.exe -NoProfile -File scripts/open-review.ps1 `
  -AdditionalBundleRoots "artifacts/bundles" -NoOpen
```

## 2. `scripts/open-task-console.ps1`

Generates and opens the task management console.

### Parameters & Historical Behavior
- `-BundleDir <path>`: Explicit bundle path. If omitted, searches repo roots and `-AdditionalBundleRoots`.
- `-AdditionalBundleRoots <string[]>`: Optional additional search roots for bundle discovery.
- `-NoRefresh`: Forwards `--no-refresh` to the inner `export-task-console` command.
- **Unconditional Refresh Notice:** Regardless of `-NoRefresh`, the launcher unconditionally runs `refresh-review-html` at the end to ensure `review.html` and `task-console.html` remain mutually linked. Files may still be written when `-NoOpen` or `-NoRefresh` is specified.
- `-NoOpen`: Prevents launching the browser.
- `-PrintJson`: Outputs JSON status to stdout. Fallback paths are used if stdout JSON parsing encounters large outputs.

### Example
```powershell
powershell.exe -NoProfile -File scripts/open-task-console.ps1 `
  -BundleDir "<path-to-webui-bundle>" -NoOpen
```

## 3. `scripts/open-targeted-review.ps1`

Prepares and starts a local loopback HTTP server for targeted transcript reviews.

### Parameters & Execution
- `-RequestJson <path>`: (Mandatory) Path to targeted review request JSON.
- `-OutputDir <path>`: (Mandatory) Destination directory for prepared review data.
- `-Port <int>`: Server port (defaults to `0` for OS-allocated dynamic port).
- `-NoOpen`: Suppresses passing `--open` to the server, but **does not suppress starting the server**.
- **Interpreter Selection:** Invokes `python` directly from `PATH`, whereas the first two scripts delegate through `video-knowledge.ps1` (which may select a repo `.venv` interpreter).

### Example
```powershell
powershell.exe -NoProfile -File scripts/open-targeted-review.ps1 `
  -RequestJson "<path-to-request.json>" -OutputDir "<path-to-output-dir>" -Port 8080 -NoOpen
```

## Operational Boundaries & Verification Status

- **Side Effects & Environment:** The scripts may modify process environment variables (`$env:PYTHONPATH`, `$env:PYTHONIOENCODING`) and exit with `$LASTEXITCODE`. Call them from a dedicated PowerShell session if environment isolation is required.
- **Network & Provider Reads:** Forwarded commands may read local provider configurations or access network services allowed by backend policies; no offline certificate is provided.
- **Unverified Runtime:** File existence checks confirm artifact presence, not HTML validity, media playability, or human review approval. No browser, server, or pytest execution occurred during this publication batch.

## Source adaptation and checks

The added `AdditionalBundleRoots` parameter is last in the two existing parameter lists, preserving their earlier ordering. With no explicit bundle, those launchers retain repository-relative `real-tests` and `openclaw-runs` discovery plus any nonblank additional roots, then select the latest matching bundle as before. Relative extra roots use the caller’s working directory. The targeted launcher has no such option.

The scripts are checkout sources, not new installed commands or registry routes. JSON output paths can contain local runtime identifiers; status flags record requested operations, not proof a browser displayed a usable page. The task-console HTML existence guard does not certify its JSON, media or source evidence.

Only source-text parsing and static comparison are intended for this release; no copied launcher, real bundle, environment configuration, browser or HTTP server was executed. A separate PowerShell 7.5.2 in-memory array comparison checked empty/null/blank and valid extra-root values; it is not Windows PowerShell 5.1 or launcher execution proof. Existing backend features/assets/optional dependencies and author permissions remain separate.
