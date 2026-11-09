param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONIOENCODING = "utf-8"
$env:LITELLM_TELEMETRY = "False"
$env:LITELLM_LOCAL_MODEL_COST_MAP = "True"
& python -m video_knowledge_pipeline.model_gateway @Arguments
exit $LASTEXITCODE
