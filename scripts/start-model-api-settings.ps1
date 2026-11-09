param(
    [int]$Port = 0,
    [string]$SettingsPath = "",
    [string]$SecretsPath = ""
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONIOENCODING = "utf-8"

$arguments = @("-m", "video_knowledge_pipeline.model_api_settings_http", "--host", "127.0.0.1")
if ($Port -gt 0) {
    $arguments += @("--port", [string]$Port)
}
if ($SettingsPath) {
    $arguments += @("--settings-path", $SettingsPath)
}
if ($SecretsPath) {
    $arguments += @("--secrets-path", $SecretsPath)
}

python @arguments
exit $LASTEXITCODE
