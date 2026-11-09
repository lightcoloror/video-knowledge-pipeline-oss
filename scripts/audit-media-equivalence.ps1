param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

# Intent: expose the read-only duplicate-media hard gates through a stable local
# front door. Decision: delegate to the Python module rather than duplicating
# media, ASR, or provenance logic in PowerShell. Reason: one implementation keeps
# CLI and tests aligned. Evidence: media_equivalence_audit.py owns the contract.
# Effective scope: local report generation only; no deletion is implemented.
$root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONIOENCODING = "utf-8"
& python -m video_knowledge_pipeline.media_equivalence_audit @RemainingArgs
exit $LASTEXITCODE
