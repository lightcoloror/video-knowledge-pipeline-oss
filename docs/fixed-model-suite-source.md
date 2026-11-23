# Fixed Model Candidate Benchmark Suite Source Guide

## 1. Purpose and Scope
This document specifies the operational and governance contracts for `scripts/run-model-candidate-fixed-suite.ps1` under the AGPL-3.0-only license as part of first-party VKP tooling. The script orchestrates fixed-sample model candidate benchmark workflows, parameterizing local gateway execution, per-candidate consent generation, and contract validation. In this release, two legacy project-local path defaults were replaced with mandatory required-path assertions, leaving all other parameter positions, logic, functions, and guards unchanged. No candidate benchmark suites, DPAPI secret stores, operational credentials, or model weights are distributed.

## 2. Operator Inputs and Parameter Contracts
The adapted source requires explicit input paths; it does not establish filesystem ownership or authorize execution:
- **`-PreparedSuite`**: Required string specifying the path to the prepared benchmark suite JSON. It checks path existence, reads JSON, and compares the schema and status fields with video_knowledge_pipeline.model_candidate_fixed_suite_prepared.v1 and ready_for_operator_consent. This is not full JSON Schema validation or a file-type/ownership proof.
- **`-SecretsPath`**: Required string targeting the operator-controlled DPAPI secrets store.
- **Optional Controls**: Supports `-CandidateIds` for filtering validated unique candidate identifiers, `-PortRecordPath` for port registry resolution, `-RunOutputPath` (defaulting to `fixed-suite-run.json` in the suite folder), and the `-Execute` switch.

## 3. Plan Mode versus Execute Mode
The script separates inspection from execution:
- **Plan Mode (Default)**: When `-Execute` is omitted, the script checks path existence, reads suite metadata, validates candidate selections, and configures console UTF-8 encodings. It does not read secrets store contents, spawn subprocesses, transmit network payloads, generate consent files, or write output reports.
- **Execute Mode**: Supplying `-Execute` requires an existing explicit port record containing the port and VKP LiteLLM Proxy marker, and rejects an occupied localhost TCP port. The text match does not prove authoritative registration or ownership. It launches `video_knowledge_pipeline.model_gateway`, waits for readiness, creates consent artifacts with `--confirm-data-export` and execution budgets, invokes candidate front doors, checks output contracts, and writes the final summary.

## 4. Governance, Cleanup, and Verification Boundaries
- **Process Cleanup Mechanics**: The `Stop-OwnedGateway` helper terminates the gateway process ID recorded in the PID file. Because it lacks process-generation identity checks, stale or untrusted PID paths remain an operational risk.
- **Exit Code Semantics**: The explicit final exit 1 is selected when failed_count > 0, including recorded candidate exceptions in setup, gateway, consent, execution or validation; it is not limited to transport failures. Candidates completing transport that fail quality gates yield status `completed_with_unqualified_results` under `exit 0`. Exit code 0 does not represent quality acceptance.
- **Verification Status**: All 443 lines remain strictly **UNEXECUTED (NOT RUN)**. Presence of referenced Python modules in the repository tree does not guarantee runtime or CLI compatibility.

One actual immutable stdlib source-diff check returned source_delta_pass=true,
the exact candidate hash, all 443 lines, preserved six parameter positions and
CHECK_EXIT=0. It did not parse or execute PowerShell. No original model suite,
gateway, consent, client, network, credential store, port/PID record or material
was run or read by that check. PowerShell syntax and runtime remain unverified.
The source supplies confirm-data-export to consent creation when Execute is
selected; per-candidate budget, export, retry and route fields are declarations,
not independent evidence of downstream enforcement or current user consent.
Cleanup exceptions and process environment restoration on all failure paths
are also untested. This increment distributes no operator input, configuration,
media, weight, credential or private history; AGPL-3.0-only covers its first-party
source, not those unprovided materials or external dependencies. The guide used
full-source Owner facts and selected exact excerpts, not an independent complete
runtime audit. Separate scoped authorization and verification are required
before any real operator execution.
