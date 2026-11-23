param(
    [string]$PreparedSuite = "",
    [string]$SecretsPath = "",
    [string]$PortRecordPath = "",
    [string]$CandidateIds = "",
    [string]$RunOutputPath = "",
    [switch]$Execute
)

$ErrorActionPreference = "Stop"
$Utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $Utf8NoBom
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom

function Test-TcpPort {
    param([string]$HostName, [int]$Port)
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $pending = $client.BeginConnect($HostName, $Port, $null, $null)
        if (-not $pending.AsyncWaitHandle.WaitOne(300)) {
            return $false
        }
        $client.EndConnect($pending)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Stop-OwnedGateway {
    param([string]$GatewayConfigPath)
    if (-not (Test-Path -LiteralPath $GatewayConfigPath)) {
        return
    }
    $gateway = Get-Content -Raw -Encoding utf8 -LiteralPath $GatewayConfigPath | ConvertFrom-Json
    if (-not (Test-Path -LiteralPath $gateway.pid_path)) {
        return
    }
    $processId = [int](Get-Content -Raw -LiteralPath $gateway.pid_path)
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($process) {
        Stop-Process -Id $processId -Force
        [void]$process.WaitForExit(10000)
    }
    Remove-Item -LiteralPath $gateway.pid_path -Force -ErrorAction SilentlyContinue
    $deadline = [DateTime]::UtcNow.AddSeconds(15)
    while ((Test-TcpPort -HostName $gateway.host -Port $gateway.port) -and [DateTime]::UtcNow -lt $deadline) {
        Start-Sleep -Milliseconds 250
    }
}

function Wait-GatewayReady {
    param([string]$HostName, [int]$Port)
    $deadline = [DateTime]::UtcNow.AddSeconds(45)
    $lastError = ""
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri "http://${HostName}:$Port/health/liveliness"
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) {
                return
            }
        }
        catch {
            $lastError = $_.Exception.Message
        }
        Start-Sleep -Milliseconds 500
    }
    throw "Benchmark gateway did not become ready: $lastError"
}

function Restore-ProcessEnvironmentVariable {
    param([string]$Name, [bool]$HadValue, [AllowNull()][string]$Value)
    if ($HadValue) {
        [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
    }
    else {
        Remove-Item -LiteralPath "Env:$Name" -ErrorAction SilentlyContinue
    }
}

function Resolve-PortRecordPath {
    param([string]$RequestedPath, [int]$Port)
    if (-not $RequestedPath) {
        throw "-PortRecordPath is required with -Execute; pass the reviewed port record explicitly"
    }

    $resolved = [System.IO.Path]::GetFullPath($RequestedPath)
    if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
        throw "Port record not found: $resolved"
    }
    return $resolved
}

function Resolve-CandidateOutputContract {
    param([object]$Candidate)
    if ($Candidate.output_contract_path -and (Test-Path -LiteralPath $Candidate.output_contract_path -PathType Leaf)) {
        return [string]$Candidate.output_contract_path
    }
    if ($Candidate.output_contract) {
        return ($Candidate.output_contract | ConvertTo-Json -Compress -Depth 20)
    }
    return (@{
        format = if ($Candidate.expected_format) { [string]$Candidate.expected_format } else { "any" }
    } | ConvertTo-Json -Compress)
}

function Get-LatestExecutionReport {
    param([string]$CandidateRoot, [DateTime]$NotBeforeUtc)
    $runRoot = Join-Path $CandidateRoot "model-connector-runs"
    if (-not (Test-Path -LiteralPath $runRoot -PathType Container)) {
        return $null
    }
    $threshold = $NotBeforeUtc.AddSeconds(-5)
    return Get-ChildItem -LiteralPath $runRoot -Recurse -Filter "connector-execution.json" -File |
        Where-Object { $_.LastWriteTimeUtc -ge $threshold } |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
}
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $PreparedSuite) {
    throw "-PreparedSuite is required; pass the reviewed prepared-suite JSON explicitly."
}
if (-not $SecretsPath) {
    throw "-SecretsPath is required; pass the operator-owned DPAPI secrets-store path explicitly."
}
$PreparedSuite = [System.IO.Path]::GetFullPath($PreparedSuite)
$SecretsPath = [System.IO.Path]::GetFullPath($SecretsPath)

if (-not (Test-Path -LiteralPath $PreparedSuite)) {
    throw "Prepared suite not found: $PreparedSuite"
}
if (-not (Test-Path -LiteralPath $SecretsPath)) {
    throw "DPAPI secrets store not found: $SecretsPath"
}
$suite = Get-Content -Raw -Encoding utf8 -LiteralPath $PreparedSuite | ConvertFrom-Json
if ($suite.schema -ne "video_knowledge_pipeline.model_candidate_fixed_suite_prepared.v1") {
    throw "Invalid prepared suite schema"
}
if ($suite.status -ne "ready_for_operator_consent") {
    throw "Prepared suite is not executable: status=$([string]$suite.status)"
}
$selectedCandidates = @($suite.candidates)
$requestedCandidateIds = @()
if ($CandidateIds) {
    $requestedCandidateIds = @($CandidateIds.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    if (@($requestedCandidateIds | Sort-Object -Unique).Count -ne $requestedCandidateIds.Count) {
        throw "CandidateIds contains duplicates"
    }
    $candidateById = @{}
    foreach ($candidate in $suite.candidates) {
        $candidateById[[string]$candidate.candidate_id] = $candidate
    }
    $missingCandidateIds = @($requestedCandidateIds | Where-Object { -not $candidateById.ContainsKey($_) })
    if ($missingCandidateIds.Count -gt 0) {
        throw "Unknown candidate ids: $($missingCandidateIds -join ', ')"
    }
    $selectedCandidates = @($requestedCandidateIds | ForEach-Object { $candidateById[$_] })
}
if ($selectedCandidates.Count -eq 0) {
    throw "No candidates selected"
}
$runDirectory = Split-Path -Parent $PreparedSuite
if (-not $RunOutputPath) {
    $RunOutputPath = Join-Path $runDirectory "fixed-suite-run.json"
}
$RunOutputPath = [System.IO.Path]::GetFullPath($RunOutputPath)
$port = [int]$suite.gateway.port
$summary = [ordered]@{
    schema = "video_knowledge_pipeline.model_candidate_fixed_suite_run.v1"
    execute = [bool]$Execute
    candidate_count = $selectedCandidates.Count
    suite_candidate_count = @($suite.candidates).Count
    selected_candidate_ids = @($selectedCandidates | ForEach-Object { $_.candidate_id })
    gateway_port = $port
    port_record_path = if ($PortRecordPath) { [System.IO.Path]::GetFullPath($PortRecordPath) } else { "" }
    run_path = $RunOutputPath
    completed_count = 0
    failed_count = 0
    contract_passed_count = 0
    quality_passed_count = 0
    results = @()
    operator_boundary = [ordered]@{
        requires_explicit_execute = $true
        exact_consent_per_candidate = $true
        no_cross_candidate_fallback = $true
        no_default_route_changes = $true
    }
}

if (-not $Execute) {
    $summary.status = "planned"
    $summary | ConvertTo-Json -Depth 8
    return
}

$PortRecordPath = Resolve-PortRecordPath -RequestedPath $PortRecordPath -Port $port
$summary.port_record_path = $PortRecordPath
$matchingPortLine = Get-Content -Encoding utf8 -LiteralPath $PortRecordPath | Where-Object {
    $_ -match "(?<!\d)$port(?!\d)" -and $_ -match "VKP LiteLLM Proxy"
}
if (-not $matchingPortLine) {
    throw "Port $port is not registered to VKP LiteLLM Proxy in $PortRecordPath"
}
if (Test-TcpPort -HostName "127.0.0.1" -Port $port) {
    throw "Benchmark port $port is already in use; refusing to stop an unknown listener"
}

$trackedEnvironment = @(
    "PYTHONPATH",
    "VKP_MODEL_API_SETTINGS_PATH",
    "VKP_MODEL_API_SECRETS_PATH",
    "VKP_MODEL_CONNECTOR_ALLOWED_ROOTS",
    "VKP_MODEL_CONNECTOR_ALLOWED_DESTINATIONS"
)
$previousEnvironment = @{}
foreach ($name in $trackedEnvironment) {
    $previousEnvironment[$name] = @{
        had_value = Test-Path -LiteralPath "Env:$name"
        value = [Environment]::GetEnvironmentVariable($name, "Process")
    }
}

Push-Location $RepoRoot
try {
    $env:PYTHONPATH = Join-Path $RepoRoot "src"
    $env:VKP_MODEL_API_SECRETS_PATH = $SecretsPath
    $env:VKP_MODEL_CONNECTOR_ALLOWED_ROOTS = $RepoRoot
    foreach ($candidate in $selectedCandidates) {
        $gatewayStarted = $false
        $candidateResult = [ordered]@{
            candidate_id = $candidate.candidate_id
            case_id = $candidate.case_id
            profile_id = $candidate.profile_id
            model = $candidate.model
            provider_response_model = ""
            route_revision = $candidate.route_revision
            status = "pending"
            consent_path = $candidate.consent_path
            execution_report_path = ""
            execution_exit_code = $null
            ok = $false
            transport_ok = $false
            contract_ok = $false
            quality_gate_passed = $false
            outcome_status = "pending"
            applied_aliases = @()
            contract_issues = @()
            quality_issues = @()
            latency_ms = $null
            error = ""
        }
        try {
            $env:VKP_MODEL_API_SETTINGS_PATH = $candidate.settings_path
            $env:VKP_MODEL_CONNECTOR_ALLOWED_DESTINATIONS = $candidate.destination
            if (Test-TcpPort -HostName "127.0.0.1" -Port $port) {
                throw "Benchmark port $port became occupied before candidate start"
            }

            $gatewayArgs = @(
                "-m", "video_knowledge_pipeline.model_gateway",
                "--gateway-config", $candidate.gateway_config_path,
                "--settings-path", $candidate.settings_path,
                "--secrets-path", $SecretsPath,
                "--port-record-path", $PortRecordPath,
                "start", "--execute"
            )
            $gatewayOutput = & python @gatewayArgs
            if ($LASTEXITCODE -ne 0) {
                throw "Candidate gateway start failed with exit code $LASTEXITCODE"
            }
            $gatewayStatus = ConvertFrom-Json -InputObject ($gatewayOutput -join "`n")
            if ($gatewayStatus.status -ne "started") {
                throw "Candidate gateway was not started: $($gatewayStatus.status)"
            }
            $gatewayStarted = $true
            Wait-GatewayReady -HostName "127.0.0.1" -Port $port

            $contractValue = Resolve-CandidateOutputContract -Candidate $candidate
            $consentArgs = @(
                "-m", "video_knowledge_pipeline.model_connector_consent",
                "create", $RepoRoot,
                "--task", $candidate.connector_task,
                "--route-id", $candidate.route_id,
                "--route-revision", $candidate.route_revision,
                "--settings-path", $candidate.settings_path,
                "--instructions", $candidate.instructions,
                "--output-contract", $contractValue,
                "--purpose", "VKP fixed-sample candidate quality benchmark",
                "--expires-hours", ([string]$candidate.expires_hours),
                "--max-calls", ([string]$candidate.max_calls),
                "--max-estimated-cost-usd", ([string]$candidate.max_estimated_cost_usd),
                "--max-cost-per-call-usd", ([string]$candidate.max_cost_per_call_usd),
                "--max-retries-per-call", ([string]$candidate.max_retries_per_call),
                "--confirm-data-export",
                "--output-path", $candidate.consent_path
            )
            if ($candidate.asr_prompt) {
                $consentArgs += @("--asr-prompt", [string]$candidate.asr_prompt)
            }
            foreach ($artifact in $candidate.artifacts) {
                $consentArgs += @("--artifact", $artifact.path)
            }
            & python @consentArgs | Out-Null
            if ($LASTEXITCODE -ne 0) {
                throw "Consent creation failed with exit code $LASTEXITCODE"
            }

            $executionStartedUtc = [DateTime]::UtcNow
            $executeArgs = @(
                "execute-consented-model-task",
                $candidate.consent_path,
                "--route-revision", $candidate.route_revision,
                "--write"
            )
            $executionResultLines = & (Join-Path $RepoRoot "scripts\video-knowledge.ps1") @executeArgs
            $candidateResult.execution_exit_code = $LASTEXITCODE
            $frontDoorResult = $null
            if ($executionResultLines) {
                try {
                    $frontDoorResult = ConvertFrom-Json -InputObject ($executionResultLines -join "`n")
                }
                catch {
                    $frontDoorResult = $null
                }
            }
            $candidateRoot = Split-Path -Parent $candidate.consent_path
            $reportFile = Get-LatestExecutionReport -CandidateRoot $candidateRoot -NotBeforeUtc $executionStartedUtc
            if (-not $reportFile) {
                $frontDoorDetail = ""
                if ($frontDoorResult) {
                    $frontDoorMessage = [string]$frontDoorResult.error
                    if (-not $frontDoorMessage -and @($frontDoorResult.consent.blockers).Count -gt 0) {
                        $frontDoorMessage = [string]$frontDoorResult.consent.blockers[0].message
                    }
                    $frontDoorDetail = "; front-door status=$([string]$frontDoorResult.status)"
                    if ($frontDoorMessage) {
                        $frontDoorDetail += "; message=$frontDoorMessage"
                    }
                }
                throw "Candidate execution report was not written; exit code $($candidateResult.execution_exit_code)$frontDoorDetail"
            }
            $execution = Get-Content -Raw -Encoding utf8 -LiteralPath $reportFile.FullName | ConvertFrom-Json
            $candidateResult.execution_report_path = $reportFile.FullName
            $candidateResult.ok = [bool]$execution.ok
            $candidateResult.status = [string]$execution.status
            $runtime = $execution.model_result.runtime_result
            if ($execution.model_result.schema -eq "video_knowledge_pipeline.model_runtime_result.v1") {
                $runtime = $execution.model_result
            }
            elseif (-not $runtime -and @($execution.model_result.calls).Count -gt 0) {
                $runtime = $execution.model_result.calls[0].runtime_result
            }
            if ($runtime) {
                $candidateResult.latency_ms = $runtime.latency_ms
                $candidateResult.provider_response_model = [string]$runtime.response.model
            }
            $candidateResult.transport_ok = $candidateResult.ok
            $contractValue = Resolve-CandidateOutputContract -Candidate $candidate
            $contractArgs = @(
                "-m", "video_knowledge_pipeline.model_output_contracts",
                $reportFile.FullName,
                "--contract", $contractValue
            )
            $contractOutput = & python @contractArgs
            $contractExitCode = $LASTEXITCODE
            try {
                $contractResult = ConvertFrom-Json -InputObject ($contractOutput -join [Environment]::NewLine)
            }
            catch {
                throw "Output contract validator returned invalid JSON; exit code $contractExitCode"
            }
            $candidateResult.contract_ok = [bool]$contractResult.contract_ok
            $candidateResult.quality_gate_passed = [bool]$contractResult.quality_gate_passed
            $candidateResult.outcome_status = [string]$contractResult.status
            $candidateResult.applied_aliases = @($contractResult.applied_aliases)
            $candidateResult.contract_issues = @($contractResult.contract_issues)
            $candidateResult.quality_issues = @($contractResult.quality_issues)
            if (-not $candidateResult.ok) {
                $candidateResult.error = [string]$execution.model_result.error
                if (-not $candidateResult.error -and $runtime) {
                    $candidateResult.error = [string]$runtime.error
                }
                if (-not $candidateResult.error) {
                    $candidateResult.error = "Candidate execution failed with exit code $($candidateResult.execution_exit_code)"
                }
            }
            if ($candidateResult.ok) {
                $summary.completed_count++
            }
            else {
                $summary.failed_count++
            }
            if ($candidateResult.contract_ok) {
                $summary.contract_passed_count++
            }
            if ($candidateResult.quality_gate_passed) {
                $summary.quality_passed_count++
            }
        }
        catch {
            $candidateResult.status = "failed"
            $candidateResult.error = $_.Exception.Message
            $summary.failed_count++
        }
        finally {
            if ($gatewayStarted) {
                Stop-OwnedGateway -GatewayConfigPath $candidate.gateway_config_path
            }
            $summary.results += [pscustomobject]$candidateResult
        }
    }
}
finally {
    foreach ($name in $trackedEnvironment) {
        $previous = $previousEnvironment[$name]
        Restore-ProcessEnvironmentVariable -Name $name -HadValue $previous.had_value -Value $previous.value
    }
    Pop-Location
}

$summary.status = if ($summary.failed_count -gt 0) {
    "partial"
}
elseif ($summary.quality_passed_count -lt $summary.candidate_count) {
    "completed_with_unqualified_results"
}
else {
    "completed"
}
[System.IO.File]::WriteAllText(
    $RunOutputPath,
    ($summary | ConvertTo-Json -Depth 10),
    $Utf8NoBom
)
$summary | ConvertTo-Json -Depth 10
if ($summary.failed_count -gt 0) {
    exit 1
}
