$ErrorActionPreference = "Stop"
$root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONIOENCODING = "utf-8"
python -m video_knowledge_pipeline.model_connector_consent @args
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
