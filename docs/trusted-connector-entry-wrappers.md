# Trusted model connector PowerShell entry wrappers

These two first-party Windows source-checkout launchers use the existing public VKP backends. The repository license is AGPL-3.0-only. No additional dependency, configuration sample containing secrets, or service binary is distributed by this source increment.

- `scripts/start-trusted-model-connector-with-env.ps1` launches `video_knowledge_pipeline.trusted_model_connector_mcp` over stdio.
- `scripts/start-trusted-capability-broker-http.ps1` launches `video_knowledge_pipeline.trusted_model_connector_remote_mcp` with a loopback HTTP endpoint.

## Environment and prerequisites

Use a separate PowerShell process and an appropriate existing Python on PATH with the backend's dependencies already available. This guide does not install them. The launchers set PYTHONPATH to this checkout's `src` and UTF-8 output encoding. They are source-checkout entrypoints, not a packaged standalone runtime.

Both read these optional operator-trusted files in order: `.local/model-connector.env`, `.local/video-knowledge.env`, `.local/vision.env`. They accept simple `NAME=value` or `$env:NAME=value`, trim values, strip one enclosing matching quote pair, and set the current process environment. Later definitions replace earlier ones. Comment, empty and nonmatching lines are ignored. Values are not evaluated as PowerShell expressions.

These files can change the child process environment and module/provider configuration. Trust them as operator configuration; this parser and a separate process are not a security sandbox. This publication did not read any real instance of these files.

## Stdio entry

The stdio wrapper has no formal parameter block and forwards its arguments to the backend. The reviewed backend entry does not implement `--help` or `--dry-run`; do not use those as safety probes.

Example only, not executed during acceptance:

```powershell
powershell.exe -NoProfile -File .\scripts\start-trusted-model-connector-with-env.ps1
```

## HTTP entry

The default bind arguments are `127.0.0.1`, port `8766`, and path `/mcp`. `-AllowedRoot` resolves an existing literal path; otherwise the checkout root is exported. The source computes an additional optional sibling output-root array, but exports only the single resolved root to `VKP_MODEL_CONNECTOR_ALLOWED_ROOTS`. This behavior is preserved; the unused array's intended policy is unknown.

The default `-AllowedDestination` is `ark.cn-beijing.volces.com`. `-IncludeConfiguredRemoteDestinations` is enabled by default. The wrapper invokes the existing `model_destination_allowlist` module using the explicitly selected settings file, or `.local/model-api-settings.json`, then combines the returned destinations with explicit ones in a case-insensitive set and exports sorted values. A nonzero helper exit or malformed JSON stops before server startup. This is configuration plumbing; it is not independent proof of complete destination or consent enforcement.

An explicit `-ModelSettingsPath` is forwarded for runtime route selection too. `-ModelSecretsPath` resolves and forwards an existing operator-selected store path. The wrapper does not itself decrypt the store, validate credentials, or establish how backend key protection works. No real settings or store was opened during acceptance.

Example only, not executed during acceptance:

```powershell
powershell.exe -NoProfile -File .\scripts\start-trusted-capability-broker-http.ps1 -Port 8766 -Path "/mcp"
```

Loopback binding describes inbound HTTP listening. Exposed capability tools may make outbound provider requests when actually operated, subject to the backend's separate policy and consent checks. Publishing these sources does not authorize such requests or prove their enforcement. Starting a backend is not a no-write operation guarantee.

## Changes and bounded checks

Only repository-root resolution gains `-LiteralPath`, retaining PathInfo semantics, and the final native Python call gains `exit $LASTEXITCODE`. Parameter blocks, defaults, environment parsing, helper-failure handling and the actual single-root export remain unchanged.

Owner checks on Windows PowerShell 5.1 parsed both originals and both candidates with zero errors and identical parameter blocks. Six synthetic wrapper cases passed: stdio and HTTP final exits 0/37, helper failure, and malformed helper JSON. Bracket/space paths, environment precedence, quote stripping without interpolation, opaque stdio argument forwarding, explicit settings/store paths and actual single-root/destination exports were checked with function stubs. An expected-failing private bad copy lacking final exit forwarding returned false success for a stub failure, demonstrating the exit check's sensitivity.

The first harness run had an incorrect empty-string expectation for an absent environment variable; PowerShell serialized it as null. Owner corrected the harness only, retained the first error/logs, and ran the corrected checks. Candidate source did not change for that correction.

Gemini assisted source review. No operational MCP/HTTP backend, DPAPI access, remote provider call, real media processing or real configuration loading was performed. Function stubs do not prove service startup, transport compatibility, credential protection, consent correctness, sandbox security or provider health. The HTTP backend's main entry and dependency hashes were checked; this was not a full backend security audit.
