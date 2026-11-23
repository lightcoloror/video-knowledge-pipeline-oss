# Streamable HTTP MCP Capability Probe Source Guide

## 1. Purpose and Scope
This document specifies the operational contracts for `scripts/smoke-trusted-capability-broker-http.py` under the AGPL-3.0-only license within first-party VKP tooling. The 35-line script provides source for a small connection probe that attempts Streamable HTTP transport, session initialization, tool discovery and capability invocation against an external Model Context Protocol (MCP) endpoint without distributing the MCP SDK, local servers, or credentials.

## 2. Probe Lifecycle and Output Contract
The probe executes a sequential asynchronous workflow:
- Connects to the target URL (defaulting to `http://127.0.0.1:8766/mcp`, configurable via `--url`), initializes a `ClientSession`, sorts discovered tool names, and invokes `model_connector_capabilities` with empty arguments.
- **Status Semantics**: Emits `status: "passed"` solely if the awaited calls return without a propagated exception. The script does not evaluate `result.isError`, validate capability schema payloads, check tool permissions, or assert specific toolsets. This field records completion of those calls; it is not security authorization or business capability compliance, and it does not establish a live endpoint in this source-only batch.
- **URL Handling**: The output echoes the complete provided URL without redaction; operators must avoid placing private access tokens or sensitive query parameters in the target address.

## 3. Dependencies and Verification Limits
- **Prerequisites**: Requires an independently installed `mcp` Python SDK and a pre-existing authorized server. Remote endpoints are not guaranteed to be side-effect-free.
- **Verification Status**: All 35 lines remain strictly **UNEXECUTED (NOT RUN)**. Owner static checks and one actual stdlib ast.parse/compile of the source passed. The module and MCP SDK were not imported, no endpoint was contacted, and no runtime or capability test was run.
The existing public project metadata contains broad MCP dependency declarations; their presence does not establish compatibility with the required Streamable HTTP API. This increment does not change those requirements or distribute SDK/server code. The original 35 lines are byte-identical to the selected first-party source blob. Actual invocation requires separate authorization and a compatible operator-selected endpoint; the emitted URL and content-block count are not evidence of real capability acceptance.
