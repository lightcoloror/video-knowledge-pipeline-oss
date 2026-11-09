# Consent CLI PowerShell Entry Wrappers

`docs/consent-cli-wrappers.md` documents two optional, source-only PowerShell entry wrappers under `scripts/`, under the repository's existing AGPL-3.0-only license, providing thin front-door access to public consent modules:

- `scripts/media-connector-consent.ps1` -> `video_knowledge_pipeline.media_connector_consent`
- `scripts/model-connector-consent.ps1` -> `video_knowledge_pipeline.model_connector_consent`

## Execution and Environment

The wrappers resolve the repository root via `-LiteralPath` (safely handling directory paths containing literal brackets), set `$env:PYTHONPATH` to `$root/src`, configure `PYTHONIOENCODING=utf-8`, and forward command arguments `@args` to `python` on `PATH`. Both scripts enforce a non-zero exit guard (`if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }`).

These are source-checkout entry points, not a pinned interpreter binding. The operator must already have an appropriate PATH-selected Python and VKP dependencies. PYTHONPATH is overwritten, not appended. Separate processes keep the assignments out of the calling interactive session. Usage-only examples (not executed by this acceptance):

```powershell
powershell.exe -NoProfile -File .\scripts\media-connector-consent.ps1 --help
powershell.exe -NoProfile -File .\scripts\model-connector-consent.ps1 --help
```

## Verification Status and Safety Boundaries

Owner static analysis confirmed 0 AST parse errors under Windows PowerShell 5.1 Desktop. In synthetic tests using prepended stub functions, 4 cases (2 wrappers across exit codes 0 and 37) verified argument forwarding, UTF-8 encoding, literal bracket path resolution, and exit code propagation. A separate control copy omitted only the model exit guard and returned host code 0 for stub code 37, demonstrating the specific false-success regression. The fixture prepended a Python-named PowerShell function to the byte-identical candidate body; it did not test real external interpreter binding.

These tests validated PowerShell wrapper contracts only; no actual Python modules, consent registries, external AI models, network sockets, or media connectors were executed. Underlying CLI actions may write consent records, reserve attempts or lead to provider execution. These launchers add no execution authority; a stored consent record does not authenticate a native user's approval. Apply the existing policy and authorization boundaries before protected operations. No credentials, private configuration, media, runtime binaries or new dependencies are distributed in this batch. Installed wheels, cross-platform environments, and live policy enforcement remain unverified.
