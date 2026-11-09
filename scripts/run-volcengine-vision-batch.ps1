param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$BundleDir,
    [string]$EnvFile = $env:VKP_PROVIDER_ENV_FILE,
    [int]$Limit = 10,
    [string]$Indexes = "",
    [int]$TimeoutSeconds = 90,
    [int]$ImageProbeMaxEdge = 512,
    [int]$ImageProbeJpegQuality = 55,
    [int]$VisionRetries = 3,
    [double]$VisionRetryDelaySeconds = 5,
    [switch]$Temporal,
    [int]$FrameCount = 8,
    [switch]$Execute,
    [ValidateSet("operator", "agent")]
    [string]$ExecutionActor = "operator",
    [string]$ExportConsent = "",
    [switch]$NoRefresh
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$BundleDir = (Resolve-Path -LiteralPath $BundleDir).Path

function Import-VisionEnv {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Vision env file not found: $Path"
    }
    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) {
            continue
        }
        $parts = $trimmed.Split("=", 2)
        $name = $parts[0].Trim()
        $value = $parts[1].Trim()
        if ($value.Length -ge 2) {
            $quote = $value.Substring(0, 1)
            if (($quote -eq '"' -or $quote -eq "'") -and $value.EndsWith($quote)) {
                $value = $value.Substring(1, $value.Length - 2)
            }
        }
        if ($name -match "^[A-Za-z_][A-Za-z0-9_]*$") {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
    if ($env:OPENAI_API_KEY -and -not $env:LLM_API_KEY) { $env:LLM_API_KEY = $env:OPENAI_API_KEY }
    if ($env:OPENAI_BASE_URL -and -not $env:LLM_BASE_URL) { $env:LLM_BASE_URL = $env:OPENAI_BASE_URL }
    if ($env:MODEL_NAME -and -not $env:LLM_MODEL) { $env:LLM_MODEL = $env:MODEL_NAME }
    $env:LECTURE_VISION_PROVIDER = "volcengine_coding_plan"
}

function Invoke-Vkp {
    param([string[]]$Arguments)
    & (Join-Path $PSScriptRoot "video-knowledge.ps1") @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "video-knowledge.ps1 failed: $($Arguments -join ' ')"
    }
}

function Quote-Arg {
    param([string]$Value)
    if ($Value -match "^[A-Za-z0-9_./:\-]+$") {
        return $Value
    }
    return "'" + ($Value -replace "'", "''") + "'"
}

function Write-RunCommand {
    param(
        [string]$Path,
        [string[]]$Arguments
    )
    $command = ".\scripts\video-knowledge.ps1 " + (($Arguments | ForEach-Object { Quote-Arg $_ }) -join " ")
    [System.IO.File]::WriteAllText($Path, $command + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
}

Import-VisionEnv -Path $EnvFile

$providerConfigPath = Join-Path $BundleDir "volcengine-provider-config.json"
$providerConfigJson = @{
    provider = "volcengine_coding_plan"
    timeout_seconds = $TimeoutSeconds
} | ConvertTo-Json -Depth 4
[System.IO.File]::WriteAllText($providerConfigPath, $providerConfigJson, [System.Text.UTF8Encoding]::new($false))
$preflightArgs = @("vision-execution-preflight", $BundleDir, "--provider-config", $providerConfigPath)
if ($Temporal) {
    $preflightArgs += @("--no-semantic", "--temporal-limit", "$Limit", "--frame-count", "$FrameCount")
    if ($Indexes) { $preflightArgs += @("--temporal-indexes", $Indexes) }
} else {
    $preflightArgs += @("--semantic-limit", "$Limit", "--no-temporal")
    if ($Indexes) { $preflightArgs += @("--semantic-indexes", $Indexes) }
}

$preflightRaw = Invoke-Vkp -Arguments $preflightArgs
$preflight = ($preflightRaw | Out-String) | ConvertFrom-Json

$confirmCalls = [int]$preflight.confirmation.confirm_vision_calls
$confirmIndexes = [string]$preflight.confirmation.confirm_vision_indexes

Write-Host "Volcengine vision preflight ready: $($preflight.ready_to_execute)"
Write-Host "Provider: $($preflight.provider.provider) / $($preflight.provider.model)"
Write-Host "Confirm calls: $confirmCalls"
Write-Host "Confirm indexes: $confirmIndexes"
Write-Host "Index mode: timeline item indexes, not zero-based JSON list positions"
$confirmedArgsName = "mcp-run-multimodal-frame-analysis-confirmed.args.json"
if ($Temporal) {
    $confirmedArgsName = "mcp-run-temporal-visual-analysis-confirmed.args.json"
}
Write-Host "Confirmed args: $BundleDir\$confirmedArgsName"

if (-not $preflight.ready_to_execute) {
    Write-Host "Preflight is not ready. Inspect: $BundleDir\vision-execution-preflight.md"
    exit 2
}

if (-not $Execute) {
    Write-Host "Preview only. Re-run with -Execute in a visible PowerShell to send frames to Volcengine."
    exit 0
}

if ($Temporal) {
    $runArgs = @(
        "run-temporal-visual-analysis", $BundleDir,
        "--execute",
        "--limit", "$Limit",
        "--frame-count", "$FrameCount",
        "--provider-config", $providerConfigPath,
        "--confirm-vision-calls", "$confirmCalls",
        "--confirm-vision-indexes", $confirmIndexes,
        "--image-probe-max-edge", "$ImageProbeMaxEdge",
        "--image-probe-jpeg-quality", "$ImageProbeJpegQuality",
        "--vision-retries", "$VisionRetries",
        "--vision-retry-delay-seconds", "$VisionRetryDelaySeconds"
    )
    if ($Indexes) { $runArgs += @("--indexes", $Indexes) }
} else {
    $runArgs = @(
        "run-multimodal-frame-analysis", $BundleDir,
        "--execute",
        "--limit", "$Limit",
        "--provider-config", $providerConfigPath,
        "--confirm-vision-calls", "$confirmCalls",
        "--confirm-vision-indexes", $confirmIndexes,
        "--image-probe-max-edge", "$ImageProbeMaxEdge",
        "--image-probe-jpeg-quality", "$ImageProbeJpegQuality",
        "--vision-retries", "$VisionRetries",
        "--vision-retry-delay-seconds", "$VisionRetryDelaySeconds"
    )
    if ($Indexes) { $runArgs += @("--indexes", $Indexes) }
}

$runArgs += @("--execution-actor", $ExecutionActor)
if ($ExportConsent) {
    $resolvedConsent = (Resolve-Path -LiteralPath $ExportConsent).Path
    $runArgs += @("--export-consent", $resolvedConsent)
}

$runCommandPath = Join-Path $BundleDir "volcengine-last-run-command.ps1"
Write-RunCommand -Path $runCommandPath -Arguments $runArgs
Write-Host "Run command saved: $runCommandPath"
Write-Host "Starting Volcengine vision execution..."
Invoke-Vkp -Arguments $runArgs | Out-Host
Write-Host "Volcengine vision execution finished. Refreshing bundle artifacts..."

if (-not $NoRefresh) {
    Invoke-Vkp -Arguments @("audit-knowledge-coverage", $BundleDir) | Out-Host
    Invoke-Vkp -Arguments @("acceptance-check", $BundleDir) | Out-Host
    Invoke-Vkp -Arguments @("bundle-status-report", $BundleDir) | Out-Host
    Invoke-Vkp -Arguments @("export-knowledge-note", $BundleDir) | Out-Host
    Invoke-Vkp -Arguments @("export-task-console", $BundleDir) | Out-Host
    Invoke-Vkp -Arguments @("refresh-review-html", $BundleDir) | Out-Host
}

Write-Host "Volcengine vision batch completed."
