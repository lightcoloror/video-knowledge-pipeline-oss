param(
    [Parameter(Mandatory = $true)][string]$RequestJson,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [int]$Port = 0,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONIOENCODING = "utf-8"

python -m video_knowledge_pipeline.targeted_review prepare $RequestJson $OutputDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$arguments = @("-m", "video_knowledge_pipeline.targeted_review", "serve", $OutputDir, "--port", [string]$Port)
if (-not $NoOpen) { $arguments += "--open" }
python @arguments
exit $LASTEXITCODE
