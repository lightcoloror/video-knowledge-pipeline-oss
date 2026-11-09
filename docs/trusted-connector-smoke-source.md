# Trusted-model connector smoke (source)

`scripts/smoke-trusted-model-connector.py` is an optional, operator-run source smoke for the existing stdio MCP trusted-model connector. It is provided as a source entry, not evidence of runtime acceptance.

The script uses the MCP Python client to start `scripts/start-trusted-model-connector-with-env.ps1` as a child process over stdio. It initializes the session, lists the advertised tools, and calls only `model_connector_capabilities`. It then prints a single JSON object and exits `0` when all five expected tool names are present, no advertised name contains `create` or `confirm`, and the capabilities call reports `isError` false; otherwise that completed check returns `2`. Import, child-process or protocol exceptions may instead terminate before this result is produced. The JSON object reports `ok`, `tools`, `missing_tools`, `forbidden_tools` and `capabilities_call_error`, so a failed check is identifiable without parsing logs. The script does not retry and does not modify the connector or its configuration.

The child launcher is invoked as `powershell.exe -NoProfile -File` with the launcher path, so the local PowerShell execution policy applies; the script does not request a policy bypass. The launcher reads operator-local `.local` environment files when present before starting `video_knowledge_pipeline.trusted_model_connector_mcp`, so operator settings or credential context may exist outside the repository.

`mcp>=1.0.0` is an existing optional dependency in the public `pyproject.toml`; this source does not install, pin or verify it. A successful capabilities call only proves the MCP call returned without the protocol error flag; it does not verify capability contents, safety or consent. Listing tools does not invoke them. Review the source before running it; real use launches the connector process. This script calls only the named capability tool; underlying SDK/server effects were not runtime-audited. License: AGPL-3.0-only.

## Verification limit

Nonexecuting AST/compile and exact-diff checks verified only removal of the two adjacent policy-bypass arguments; the remaining source and guard AST is unchanged. The candidate was never imported or invoked. Two checker defects were retained before its final static pass. Actual MCP SDK import, child launch, tool inventory, capability payload, policy behavior, providers and consent/security enforcement remain unverified. No new dependency was installed. External SDK/PowerShell payloads, credentials, real settings/media and private history are not bundled.
