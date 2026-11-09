# Model Gateway PowerShell Entry Wrappers

`docs/model-gateway-entry-wrappers.md` documents two optional, source-only PowerShell entry wrappers under `scripts/`, licensed under AGPL-3.0-only as part of the Video Knowledge Pipeline (VKP). They provide direct checkout entry points to existing public backend Python modules:

- `scripts/video-knowledge-model-gateway.ps1` -> `video_knowledge_pipeline.model_gateway`
- `scripts/video-knowledge-model-smoke-readiness.ps1` -> `video_knowledge_pipeline.model_gateway_smoke_readiness`

## Invocation and Environment

The wrappers capture remaining CLI arguments via `ValueFromRemainingArguments`, resolve repository paths using `-LiteralPath` with `.Path` string conversion, overwrite `PYTHONPATH` with `$root/src`, and enforce `PYTHONIOENCODING=utf-8`. The gateway script also sets `LITELLM_TELEMETRY=False` and `LITELLM_LOCAL_MODEL_COST_MAP=True` (these existing environment flags are retained; their runtime effects and comprehensive cost or privacy properties were not tested). Both scripts preserve `$LASTEXITCODE`.

To avoid mutating caller process environments or altering exit handling, run wrappers in a separate PowerShell process:

```powershell
powershell.exe -NoProfile -File .\scripts\video-knowledge-model-gateway.ps1 --help
powershell.exe -NoProfile -File .\scripts\video-knowledge-model-smoke-readiness.ps1 --help
```

The wrapper selects `python` through the operator's PATH. A suitable interpreter and existing VKP dependencies must already be available; this batch provides no pinned private venv, new environment or installation. The help examples above are usage instructions and were not executed by this acceptance.

## Operational Roles and Verification Boundaries

- **Gateway**: May persist configuration, import legacy settings, or bind a local HTTP service.
- **Readiness Diagnostic**: Inspects settings, consent metadata, and port records, writing diagnostic reports. These tools are not read-only.

Owner static `ParseFile` analysis verified syntax across both scripts on Windows PowerShell 5.1 without errors, confirming 12 other EndBlock statements remain unchanged. The two root-resolution statements contain the only changes; parameter blocks and all other source bytes are preserved. However, neither wrapper nor backend Python module, HTTP gateway, AI provider, network operation or readiness check was executed. Actual argument forwarding, exit behavior, PATH binding and LiteLLM effects remain unverified. The wrappers do not grant model execution authorization or certify provider availability. Wheel distribution and cross-platform workflows remain unverified.
