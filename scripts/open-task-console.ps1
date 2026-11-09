param(
    [string]$BundleDir = "",
    [switch]$NoRefresh,
    [switch]$NoOpen,
    [switch]$PrintJson,
    [string[]]$AdditionalBundleRoots = @()
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

function Test-VkpBundle {
    param([string]$Path)
    if (-not $Path) { return $false }
    return (Test-Path -LiteralPath (Join-Path $Path "manifest.json")) -and (Test-Path -LiteralPath (Join-Path $Path "timeline.json"))
}

function Find-LatestVkpBundle {
    $searchRoots = @(
        (Join-Path $root "real-tests"),
        (Join-Path $root "openclaw-runs")
    ) + (@($AdditionalBundleRoots) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }) |
        Where-Object { Test-Path -LiteralPath $_ } | Select-Object -Unique

    $candidates = New-Object System.Collections.Generic.List[object]
    foreach ($searchRoot in $searchRoots) {
        Get-ChildItem -LiteralPath $searchRoot -Directory -Recurse -Filter "webui-bundle" -ErrorAction SilentlyContinue | ForEach-Object {
            if (Test-VkpBundle $_.FullName) {
                $manifest = Join-Path $_.FullName "manifest.json"
                $timeline = Join-Path $_.FullName "timeline.json"
                $latestWrite = @((Get-Item -LiteralPath $manifest).LastWriteTime, (Get-Item -LiteralPath $timeline).LastWriteTime, $_.LastWriteTime) | Sort-Object -Descending | Select-Object -First 1
                $candidates.Add([pscustomobject]@{ Path = $_.FullName; LastWriteTime = $latestWrite }) | Out-Null
            }
        }
    }

    if (Test-VkpBundle $root) {
        $candidates.Add([pscustomobject]@{ Path = $root; LastWriteTime = (Get-Item -LiteralPath (Join-Path $root "manifest.json")).LastWriteTime }) | Out-Null
    }

    $latest = $candidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest) { return $latest.Path }
    return ""
}

if (-not $BundleDir) {
    $BundleDir = Find-LatestVkpBundle
}

if (-not $BundleDir) {
    throw "No VKP webui-bundle found. Pass -BundleDir D:\path\to\webui-bundle or run prepare-local-video-run/acceptance-run first."
}

$BundleDir = (Resolve-Path -LiteralPath $BundleDir).Path
if (-not (Test-VkpBundle $BundleDir)) {
    throw "Not a VKP webui-bundle: $BundleDir. Expected manifest.json and timeline.json."
}

$argsList = @("export-task-console", $BundleDir)
if ($NoRefresh) {
    $argsList += "--no-refresh"
}

$cliOutput = & (Join-Path $PSScriptRoot "video-knowledge.ps1") @argsList
if ($LASTEXITCODE -ne 0) {
    throw "export-task-console failed with exit code $LASTEXITCODE"
}

# The CLI can return a very large JSON object after refresh. Windows PowerShell
# may fail to parse that stdout even when the command succeeded, so the launcher
# treats the generated files as the source of truth.
$result = $null
try {
    $result = ($cliOutput | Out-String) | ConvertFrom-Json
} catch {
    $result = $null
}

$consolePath = Join-Path $BundleDir "task-console.html"
$consoleJsonPath = Join-Path $BundleDir "task-console.json"
if ($result -and [string]$result.task_console_html_path) {
    $consolePath = [string]$result.task_console_html_path
}
if ($result -and [string]$result.task_console_json_path) {
    $consoleJsonPath = [string]$result.task_console_json_path
}
if (-not (Test-Path -LiteralPath $consolePath)) {
    throw "task-console.html was not generated: $consolePath"
}

# Keep review.html and task-console.html mutually reachable for existing bundles.
$null = & (Join-Path $PSScriptRoot "video-knowledge.ps1") @("refresh-review-html", $BundleDir)
if ($LASTEXITCODE -ne 0) {
    throw "refresh-review-html failed with exit code $LASTEXITCODE"
}

if (-not $NoOpen) {
    Start-Process -FilePath $consolePath | Out-Null
}

$response = [pscustomobject]@{
    ok = $true
    bundle_dir = $BundleDir
    task_console_html_path = $consolePath
    task_console_json_path = $consoleJsonPath
    opened = -not $NoOpen
    refreshed = -not $NoRefresh
}

if ($PrintJson) {
    $response | ConvertTo-Json -Depth 6
} else {
    Write-Host "VKP task console ready: $consolePath"
    if (-not $NoOpen) { Write-Host "Opened in the default browser." }
}
