# Tested-Entry PowerShell Wrapper

## Overview

`scripts/run-tested-vkp.ps1` is a PowerShell wrapper and forwarder for the repository's version-bound entrypoint mechanism, implemented by `src/video_knowledge_pipeline/tested_entry.py` with source tests in [tests/test_tested_entry.py](../tests/test_tested_entry.py).

The canonical wrapper script is first-party source under [AGPL-3.0-only](../LICENSE) (commit `d8541dc19247e37e4d6bc6a3a1e1689a2a98b6f0`, 3440 bytes LF, SHA256 `6b2c2623d9a3dd1a62d9180785fadf3ba37c96524fd31f626d78af870531f771`).

### Scope and Naming Notice
- **Legacy Name:** "Tested" is historical nomenclature. The binding and proof mechanisms verify software identity and drift detection; they do **not** run test suites, attest historical tests, or certify media, models, or content quality.
- **Contract Field:** `quality_gate_bypassed: false` reflects schema intent, not empirical quality assurance.
- **Permissions:** The wrapper does not restrict the forwarded native command, which runs with the operator's ambient permissions and may perform network or model operations.

## Option Model and Argument Parsing

The wrapper parses leading options only, allowing native arguments to pass untouched:
- **`--` Separator:** Explicitly terminates wrapper option parsing; all subsequent tokens are passed directly to the native Python command.
- **`-Vkp_Tested_Version_Path <path>`:** Path to operator binding JSON (or set via `$env:VKP_TESTED_VERSION_PATH`). Defaults to `.local/tested-version.json`.
- **`-EntryPoint <name>`:** Declared entrypoint name; defaults to `cli`.
- **`-CreateBinding`:** Switches to creation mode. Requires an explicit existing absolute `-Python` path and an explicit expected Git SHA in `-ExpectedHead`. Rejects any trailing native command.
- **`-Python` / `-ExpectedHead`:** Valid only with `-CreateBinding`. Wrapper `-Python` is a leading wrapper option; a native `--python` can follow the native command or the explicit `--` separator.

## Binding Lifecycle & Drift Detection

1. **Creation:** Reads Git HEAD (`.git`), exact Python binary metadata, managed source files, and dependency snapshots. It writes an operator binding (`video_knowledge_pipeline.tested_entry.v1`). Existing files are **never** overwritten; re-binding requires an explicit new path. In-tree bindings must reside in `.local` (git-ignored).
2. **Pre-flight Validation:** Before launching Python, the wrapper checks binding schema, verifies `source_root` matches the repository checkout, ensures the bound Python binary exists by absolute path, and verifies its SHA256 executable hash. PATH fallback is disabled.
3. **Deep Validation:** The Python adapter re-validates Git HEAD, runtime interpreter identity, re-derived source file hashes, and dependency metadata before dispatching.

## Usage Examples

### 1. Create an Operator Binding

```powershell
powershell.exe -NoProfile -File scripts/run-tested-vkp.ps1 -CreateBinding `
  -Python "<absolute-python.exe>" -ExpectedHead "<actual-fixed-checkout-HEAD>"
```

*Optional custom binding destination:*
```powershell
powershell.exe -NoProfile -File scripts/run-tested-vkp.ps1 -CreateBinding `
  -Vkp_Tested_Version_Path "<operator-binding.json>" `
  -Python "<absolute-python.exe>" -ExpectedHead "<actual-fixed-checkout-HEAD>"
```

### 2. Verify Binding Drift (`--version-proof`)

Outputs JSON proof confirming software identity and dependency state without executing native CLI logic. `--version-proof` must be the sole native argument:

```powershell
powershell.exe -NoProfile -File scripts/run-tested-vkp.ps1 `
  -Vkp_Tested_Version_Path "<operator-binding.json>" -- --version-proof
```

### 3. Forward to Native CLI

```powershell
powershell.exe -NoProfile -File scripts/run-tested-vkp.ps1 `
  -Vkp_Tested_Version_Path "<operator-binding.json>" -EntryPoint cli -- --help
```

## Operational Boundaries & Verification Status

- **Operator-Local Data:** Binding files contain local filesystem paths and environment hashes; keep them local and do not commit them as public fixtures.
- **No Isolation Guarantee:** The wrapper is not an OS sandbox. Child Python processes manage their own environment.
- **Unverified Runtime Status:** Native quoting, argument escaping, and Windows PowerShell 5.1 runtime behavior remain unverified. No native execution, test suite run, or service startup was executed for this publication batch.

## This source publication

The canonical Git-blob wrapper bytes are unchanged; the working copy differed only by CRLF line endings. PowerShell 7.5.2 Core `ParseInput` reported zero syntax errors without invoking the script. This is not Windows PowerShell 5.1 or native argument/execution verification. No binding was read or created, no version proof or forwarded command was run, and the existing public Python tests were not rerun.

The wrapper is supplied in the checkout, not installed as a wheel command. The examples use a separate PowerShell process; the wrapper contains `exit` statements. Replace placeholders before use. Creation/proof are designed to use the existing metadata-only bootstrap, while a forwarded command may import providers, process media or start services under its own authorization.

The managed-source scan follows the existing backend policy and exclusions; it is not a promise to cover every file or sanitize an arbitrary operator directory. A binding can contain local source/interpreter/distribution paths. It is an operator-controlled snapshot, not a signed test attestation or additional execution/publishing permission. Preserve an old binding and choose a fresh explicit path when re-binding is intended.
