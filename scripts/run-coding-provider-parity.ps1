param(
    [ValidateSet("prepare", "create-consents", "execute", "execute-all", "recover-interrupted", "compare")]
    [string]$Action = "prepare",
    [string]$PlanPath = "",
    [string]$ConsentIndexPath = "",
    [string]$CandidateId = "",
    [string]$Reason = "",
    [string]$OutputDir = "",
    [ValidateSet("common_fields_v1", "content_quality_v1", "capability_ceiling_v1")]
    [string]$RequestProfile = "common_fields_v1",
    [switch]$ConfirmDataExport,
    [switch]$OperatorConfirmNetwork
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONIOENCODING = "utf-8"
$module = "video_knowledge_pipeline.coding_tool_provider_parity"

if ($Action -eq "prepare") {
    $arguments = @("-m", $module, "prepare")
    if ($OutputDir) {
        $arguments += @("--output-dir", $OutputDir)
    }
    $arguments += @("--request-profile", $RequestProfile)
} elseif ($Action -eq "create-consents") {
    if (-not $PlanPath) {
        throw "PlanPath is required for create-consents"
    }
    $arguments = @("-m", $module, "create-consents", $PlanPath)
    if ($ConfirmDataExport) {
        $arguments += "--confirm-data-export"
    }
} elseif ($Action -eq "execute" -or $Action -eq "execute-all") {
    if (-not $PlanPath -or -not $ConsentIndexPath) {
        throw "PlanPath and ConsentIndexPath are required for execute"
    }
    $arguments = @(
        "-m", $module, $Action, $PlanPath, $ConsentIndexPath
    )
    if ($Action -eq "execute") {
        if (-not $CandidateId) {
            throw "CandidateId is required for execute"
        }
        $arguments += @("--candidate-id", $CandidateId)
    }
    if ($OperatorConfirmNetwork) {
        $arguments += "--operator-confirm-network"
    }
} elseif ($Action -eq "recover-interrupted") {
    if (-not $PlanPath -or -not $ConsentIndexPath) {
        throw "PlanPath and ConsentIndexPath are required for recover-interrupted"
    }
    if (-not $CandidateId -or -not $Reason) {
        throw "CandidateId and Reason are required for recover-interrupted"
    }
    $arguments = @(
        "-m", $module, "recover-interrupted", $PlanPath, $ConsentIndexPath,
        "--candidate-id", $CandidateId,
        "--reason", $Reason
    )
} else {
    if (-not $PlanPath) {
        throw "PlanPath is required for compare"
    }
    $arguments = @("-m", $module, "compare", $PlanPath)
    if ($OutputDir) {
        $arguments += @("--output-dir", $OutputDir)
    }
}

python @arguments
exit $LASTEXITCODE
