param(
    [string]$BundleDir = "",
    [switch]$NoRefresh,
    [switch]$NoOpen,
    [switch]$PrintJson,
    [switch]$SampleReview,
    [int]$SampleSize = 48,
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

if ($SampleReview) {
    $argsList = @("multimodal-sample-review", $BundleDir, "--sample-size", [string]$SampleSize)
    $null = & (Join-Path $PSScriptRoot "video-knowledge.ps1") @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "multimodal-sample-review failed with exit code $LASTEXITCODE"
    }
    $targetPath = Join-Path $BundleDir "multimodal-sample-review.html"
    $targetKind = "multimodal_sample_review"
} else {
    if (-not $NoRefresh) {
        $null = & (Join-Path $PSScriptRoot "video-knowledge.ps1") @("refresh-review-html", $BundleDir)
        if ($LASTEXITCODE -ne 0) {
            throw "refresh-review-html failed with exit code $LASTEXITCODE"
        }
    }
    $targetPath = Join-Path $BundleDir "review.html"
    $targetKind = "review"
}

if (-not (Test-Path -LiteralPath $targetPath)) {
    throw "Review page was not generated: $targetPath"
}

if (-not $NoOpen) {
    Start-Process -FilePath $targetPath | Out-Null
}

$response = [pscustomobject]@{
    ok = $true
    bundle_dir = $BundleDir
    target = $targetKind
    review_html_path = $targetPath
    opened = -not $NoOpen
    refreshed = (-not $NoRefresh) -and (-not $SampleReview)
    sample_size = if ($SampleReview) { $SampleSize } else { $null }
}

if ($PrintJson) {
    $response | ConvertTo-Json -Depth 6
} else {
    Write-Host "VKP review page ready: $targetPath"
    if (-not $NoOpen) { Write-Host "Opened in the default browser." }
}
