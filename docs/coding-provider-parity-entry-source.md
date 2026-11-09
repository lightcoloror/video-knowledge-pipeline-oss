# Coding Provider Parity CLI Entry Guide

Script: `scripts/run-coding-provider-parity.ps1`

## Overview
This document provides reference usage for the AGPL-3.0-only PowerShell wrapper `scripts/run-coding-provider-parity.ps1`. The wrapper manages execution arguments for the Python module `video_knowledge_pipeline.coding_tool_provider_parity`. It does not forward `--execution-client`, so preparation uses the backend's native OpenAI-compatible default. Other backend adapter options are outside this entry's surface.

## Environment & Execution Model
- **Interpreter**: Dispatches to bare `python` on the current system `PATH`. Automatic virtual environment (venv) detection, selection, or activation is not provided.
- **Environment Variables**: Sets process-level `$env:PYTHONPATH` to `<repo-root>/src` and `$env:PYTHONIOENCODING` to `utf-8`. Existing process environment variables are not restored upon script termination.
- **Exit Code Forwarding**: Propagates `$LASTEXITCODE` from Python. In the supplied public main, exit 0 is selected for `ready_for_operator_consent`, `ready_for_operator_execution`, `completed` or `ready_for_human_review`; other returned statuses select 2, while caught `OSError`, `ValueError` or `subprocess.SubprocessError` select 3. A status string is not independent proof of a successful or authorized remote operation.

## Request Profiles
The `-RequestProfile` parameter accepts one of three profile identifiers:
1. `common_fields_v1` (Default): Measures standard provider-default serving behavior with shared fields.
2. `content_quality_v1`: Obtains complete answers for content-quality pairing.
3. `capability_ceiling_v1`: Declares an intent to measure completion without a VKP output-token/thinking budget; provider-side limits remain unknown.

*(Note: Profile timeout and budget limits are internal constants, not commercial pricing quotes or service availability guarantees).*

## Actions and CLI Arguments
The script supports six actions via `-Action`:

1. **`prepare`** (Default):
   - Optional: `-OutputDir <path>`, `-RequestProfile <id>`.
   - Generates plan artifacts. Not guaranteed read-only; handlers may inspect local settings or write output.
2. **`create-consents`**:
   - Required: `-PlanPath <path>`.
   - Optional: `-ConfirmDataExport` (switch forwarding `--confirm-data-export`).
3. **`execute`**:
   - Required: `-PlanPath <path>`, `-ConsentIndexPath <path>`, `-CandidateId <id>`.
   - Optional: `-OperatorConfirmNetwork` (switch forwarding `--operator-confirm-network`).
4. **`execute-all`**:
   - Required: `-PlanPath <path>`, `-ConsentIndexPath <path>`.
   - Optional: `-OperatorConfirmNetwork` (switch forwarding `--operator-confirm-network`).
5. **`recover-interrupted`**:
   - Required: `-PlanPath <path>`, `-ConsentIndexPath <path>`, `-CandidateId <id>`, `-Reason <text>`.
6. **`compare`**:
   - Required: `-PlanPath <path>`.
   - Optional: `-OutputDir <path>`.

## Governance & Safety Boundaries
- **Operator Switches**: The `-ConfirmDataExport` and `-OperatorConfirmNetwork` switches forward flags to Python; they do not constitute external human consent or network sandboxing.
- **Self-Reported Status**: Receipts containing `"remote_requests_made": false` are self-authored runtime outputs, not external verification of network inactivity.
- **Operational Status**: Remote provider availability, model performance, account status, and parity metrics remain completely unverified.

## Non-Interactive Invocation Examples
```powershell
# Prepare plan with a specific request profile
powershell.exe -NoProfile -NonInteractive -File scripts/run-coding-provider-parity.ps1 -Action prepare -RequestProfile common_fields_v1 -OutputDir "<output_dir>"

# Execute a candidate with network confirmation
powershell.exe -NoProfile -NonInteractive -File scripts/run-coding-provider-parity.ps1 -Action execute -PlanPath "<plan_path>" -ConsentIndexPath "<consent_path>" -CandidateId "<candidate_id>" -OperatorConfirmNetwork
```

## Verification status
The wrapper statement text is unchanged; only the published copy's CRLF line endings are normalized to LF. Original and normalized copies passed nonexecuting Windows PowerShell 5.1 parsing. Six synthetic argument lists were accepted by an extracted pure public argparse builder, and the three profile identifiers matched. No wrapper invocation, full backend import, action handler, settings/secrets read, plan generation, consent creation, remote request or comparison run occurred. This is source/CLI-contract evidence only, not provider/runtime/quality or billing verification. Process environment assignments are scoped to whichever PowerShell process runs the script; the separate-process examples avoid carrying those assignments back into the calling process.
