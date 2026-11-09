# Local Operator UI PowerShell Entry Wrappers

`docs/operator-ui-entry-wrappers.md` documents two first-party Windows PowerShell entry wrappers under `scripts/`, licensed under AGPL-3.0-only as part of the Video Knowledge Pipeline (VKP):

- `scripts/start-review-webui.ps1` -> `video_knowledge_pipeline.review_http`
- `scripts/start-model-api-settings.ps1` -> `video_knowledge_pipeline.model_api_settings_http`

## Review WebUI Entry Wrapper

`start-review-webui.ps1` launches the local bundle inspection and editorial review server. It requires a mandatory `-BundleDir` parameter that must resolve to an existing exported directory containing `manifest.json` and `timeline.json`.

By default, the script binds to loopback (`127.0.0.1`), selects an ephemeral port when `-Port 0` is passed, refreshes review HTML on startup, and automatically launches the system browser. Operators can suppress browser launch with `-NoOpen` and request no refresh with `-NoRefresh`. The following relative fictional-bundle example was not executed; the two filename checks alone do not validate the full bundle contract:

```powershell
powershell.exe -NoProfile -File .\scripts\start-review-webui.ps1 -BundleDir "runs/sample-bundle" -Port 0 -NoOpen -NoRefresh
```

The review interface serves on loopback only. Draft edits remain in browser storage until explicitly saved back to the bundle. Bundle contents may include production transcripts or media metadata and should be treated as sensitive.

## Model API Settings WebUI Entry Wrapper

`start-model-api-settings.ps1` provides a loopback HTTP interface for managing local provider configurations and Windows DPAPI-encrypted API credentials. It accepts optional `-Port`, `-SettingsPath`, and `-SecretsPath` parameters:

```powershell
powershell.exe -NoProfile -File .\scripts\start-model-api-settings.ps1 -Port 8767
```

The wrapper forwards `-Port` only when greater than zero; otherwise the backend applies its own default/environment selection. Nonempty `-SettingsPath` and `-SecretsPath` are forwarded as supplied. Startup and UI actions can read or write local settings or secret stores; the wrapper is not a no-write operation. Code review did not inspect any real settings, API keys or secret files.

Unlike the review wrapper, the settings server does not automatically launch a browser. It serves until interrupted (Ctrl+C). Binding to loopback restricts network exposure but does not constitute an operating-system sandbox or authorization to expose endpoints across a local network.

## Verification Status and Operational Boundaries

Use a suitable existing Python on `PATH` with the project's existing dependencies in a source checkout. This publication installs nothing and starts no service. Settings operations remain governed by the backend and the operator's actual authorization.

Both wrappers resolve repository roots with `-LiteralPath` to prevent errors with bracketed directory paths, configure UTF-8 encoding, set `PYTHONPATH`, and append `exit $LASTEXITCODE` to propagate Python startup failures.

Windows PowerShell 5.1 nonexecuting parsing found no errors in the two original or candidate scripts, and both full parameter blocks were unchanged. Four top-level wrapper controls using an in-process `python` function stub passed for argument/root forwarding and exit codes 0/37, including bracketed paths; a missing-bundle check rejected before the stub was reached. A private bad copy with only the explicit exit removed reported false success (0) for stub code 37, as expected. These are stub controls, not actual Python backend or UI execution. No actual HTTP servers, browser sessions, DPAPI credentials, provider tests, or media processing were executed during this review.
