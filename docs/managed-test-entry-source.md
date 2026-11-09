# Managed Test Runner Wrapper Source Guide

## 1. Purpose and Scope
This document specifies the operational contracts and source constraints for the VKP managed test runner wrapper (`scripts/run-tests-managed.ps1`) and its contract test (`tests/test_managed_test_runner_contract.py`) under the AGPL-3.0-only license. The wrapper orchestrates managed pytest execution by bridging checkout-relative virtual environments with an external runtime adapter (`windows-test-runtime.ps1`). This release eliminates implicit parent-workspace defaults and automatic fallback directories without bundling the external shared runtime, operational test data or model weights.

## 2. Parameter Contracts and Portability Adaptations
The script maintains all seven original parameter positions while enforcing explicit environment roots:
- **Parameter Positions**: Retains `$RuntimePath`, `$TimeoutSeconds` (defaulting to 1800), `$TestArgs` (defaulting to `-q`), `$ProjectVenvPath`, `$RepoPath`, `$ManagedRoot`, and the `-PreflightOnly` switch.
- **Explicit Root Binding**: Omits legacy parent-directory defaults. `$RuntimePath` must be provided via argument or process environment variable `VKP_WINDOWS_TEST_RUNTIME`. Similarly, `$ManagedRoot` requires argument specification or `VKP_MANAGED_TEST_ROOT`. If unset, the script halts with an explicit error.
- **Fail-Closed Write Probe**: The script creates `$managedRoot` and performs a temporary GUID-based file write probe (`.vkp-write-probe-*`). The legacy catch block that redirected failed writes to `LocalApplicationData` has been removed. Probe errors are left uncaught under the existing Stop preference, rather than retried in another directory. PowerShell failure behavior was not exercised; no ACL policy is changed by this adaptation.

## 3. Contract Test Integrity
The contract test `test_managed_test_runner_contract.py` statically inspects wrapper text to enforce structural integrity:
- Checks for drive-root literals matching its regular expression; this alone is not proof of path confinement or complete privacy review.
- Asserts that legacy AppData fallback variables are completely absent.
- Verifies the presence of required explicit error messages, checkout-relative root logic, and dependency inspection tokens.

## 4. Execution Lifecycle, Side Effects, and Limits
- **Preflight Mutation**: Running with `-PreflightOnly` is not a passive read. It creates the managed directory, writes and removes the probe file, prepends for the current process `$ProjectVenvPath\Lib\site-packages` to the process `PYTHONPATH`, and runs a Python dependency verification probe prior to returning JSON. The modified `PYTHONPATH` is not restored.
- **External Adapter Delegation**: The script delegates test execution to an external `windows-test-runtime.ps1` adapter and forwards the process exit code. This increment does not bundle or implement the external adapter; its runtime compatibility, execution policy requirements, and interpreter alignment remain unverified.
- **Verification Status**: The PowerShell wrapper and adapter remain UNEXECUTED. The authorized source check failed before the test function; a later unapproved retry reported one runpy source-text case passed. The bounded execution review remains failed; no pytest runner or wrapper filesystem probe was executed.

## Actual limited check

The first limited stdlib source check failed before the source-contract function.
A subsequent correction and rerun exceeded the agreed single-run scope. It
reported source_delta_pass=true, exact candidate hashes/line counts, preserved
seven parameter positions, Python compile success and one runpy source-text
case passed, CHECK_EXIT=0. The original failure and execution-scope failure
remain distinct; the rerun is not advertised as accepted formal verification.
Owner independently bound final bytes to the requested edits without rerunning
the checker. No new full review or historical environment rebuild was needed.
The PowerShell wrapper, its dependency imports, write probe and shared adapter
were not executed; PowerShell syntax and parameter binding remain unverified.
Source files originate from untracked working snapshots, not proven Git history.
No shared runtime, real configuration, data, media, credential or private history
is bundled. The source-level contract change does not authorize actual execution.

Public packaging removes one redundant blank line at the end of the Python source-test file. Its complete AST is identical to the final reviewed model artifact; no assertion, function, skip or behavior changed. The model check was not rerun. The first Git whitespace failure is retained in private packaging receipts, and the unchanged Git check is required to pass.
