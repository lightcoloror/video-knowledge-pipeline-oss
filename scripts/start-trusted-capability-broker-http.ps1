param(
    [int]$Port = 8766,
    [string]$Path = "/mcp",
    [string]$AllowedRoot = "",
    [string[]]$AllowedDestination = @("ark.cn-beijing.volces.com"),
    [switch]$IncludeConfiguredRemoteDestinations = $true,
    [string]$ModelSettingsPath = "",
    [string]$ModelSecretsPath = ""
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = Join-Path $root "src"
$envFiles = @(
    (Join-Path $root ".local\model-connector.env"),
    (Join-Path $root ".local\video-knowledge.env"),
    (Join-Path $root ".local\vision.env")
)

foreach ($envFile in $envFiles) {
    if (-not (Test-Path -LiteralPath $envFile)) {
        continue
    }
    foreach ($line in Get-Content -LiteralPath $envFile) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }
        if ($trimmed -notmatch '^(?:\$env:)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
            continue
        }
        $name = $Matches[1]
        $value = $Matches[2].Trim()
        if ($value.Length -ge 2) {
            $quote = $value.Substring(0, 1)
            if (($quote -eq '"' -or $quote -eq "'") -and $value.EndsWith($quote)) {
                $value = $value.Substring(1, $value.Length - 2)
            }
        }
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

$resolvedAllowedRoot = if ($AllowedRoot) {
    (Resolve-Path -LiteralPath $AllowedRoot).Path
} else {
    $root.Path
}
$allowedRoots = @($resolvedAllowedRoot)
$managedOutputRoot = Join-Path (Split-Path -Parent $root.Path) "video-knowledge-output"
if (-not $AllowedRoot -and (Test-Path -LiteralPath $managedOutputRoot -PathType Container)) {
    $allowedRoots += (Resolve-Path -LiteralPath $managedOutputRoot).Path
}

$destinationSet = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
foreach ($destination in $AllowedDestination) {
    $clean = [string]$destination
    if ($clean.Trim()) {
        [void]$destinationSet.Add($clean.Trim().ToLowerInvariant())
    }
}

if ($IncludeConfiguredRemoteDestinations) {
    $settingsFile = if ($ModelSettingsPath) {
        (Resolve-Path -LiteralPath $ModelSettingsPath).Path
    } else {
        (Join-Path $root ".local\model-api-settings.json")
    }
    $arguments = @(
        "-m",
        "video_knowledge_pipeline.model_destination_allowlist",
        "--settings-path",
        $settingsFile
    )
    $rawStatus = & python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to derive configured remote model destinations."
    }
    $destinationStatus = ConvertFrom-Json -InputObject ($rawStatus -join "`n")
    foreach ($destination in @($destinationStatus.destinations)) {
        [void]$destinationSet.Add(([string]$destination).ToLowerInvariant())
    }
}
elseif ($ModelSettingsPath) {
    $settingsFile = (Resolve-Path -LiteralPath $ModelSettingsPath).Path
}

if ($ModelSettingsPath) {
    # The reviewed settings file must drive both the allowlist and runtime route
    # resolution. Otherwise consent v2 is validated against the default route.
    $env:VKP_MODEL_API_SETTINGS_PATH = $settingsFile
}
if ($ModelSecretsPath) {
    # Temporary settings may live beside non-secret artifacts. Reuse the
    # operator-selected DPAPI store instead of guessing a sibling file.
    $env:VKP_MODEL_API_SECRETS_PATH = (Resolve-Path -LiteralPath $ModelSecretsPath).Path
}
[string[]]$destinations = @($destinationSet)
[Array]::Sort($destinations, [System.StringComparer]::OrdinalIgnoreCase)
$env:VKP_MODEL_CONNECTOR_ALLOWED_ROOTS = $resolvedAllowedRoot
$env:VKP_MODEL_CONNECTOR_ALLOWED_DESTINATIONS = ($destinations -join ",")
$env:VKP_MODEL_CONNECTOR_HOST = "127.0.0.1"
$env:VKP_MODEL_CONNECTOR_PORT = [string]$Port
$env:VKP_MODEL_CONNECTOR_PATH = $Path
$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONIOENCODING = "utf-8"

python -m video_knowledge_pipeline.trusted_model_connector_remote_mcp `
    --host 127.0.0.1 `
    --port $Port `
    --path $Path
exit $LASTEXITCODE
