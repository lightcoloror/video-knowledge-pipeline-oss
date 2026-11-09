param(
    [Parameter(Mandatory = $true)]
    [string]$BundleDir,
    [int]$Port = 0,
    [switch]$NoRefresh,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$bundle = (Resolve-Path -LiteralPath $BundleDir).Path
if (-not (Test-Path -LiteralPath (Join-Path $bundle "manifest.json")) -or
    -not (Test-Path -LiteralPath (Join-Path $bundle "timeline.json"))) {
    throw "Not a VKP webui-bundle: $bundle"
}

$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONIOENCODING = "utf-8"
$arguments = @("-m", "video_knowledge_pipeline.review_http", $bundle, "--host", "127.0.0.1", "--port", [string]$Port)
if ($NoRefresh) {
    $arguments += "--no-refresh"
}
if (-not $NoOpen) {
    $arguments += "--open"
}

python @arguments
exit $LASTEXITCODE
