$ErrorActionPreference = "Stop"

$root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONIOENCODING = "utf-8"

python -m video_knowledge_pipeline.local_media_contracts @args
$exitCode = $LASTEXITCODE
exit $exitCode
