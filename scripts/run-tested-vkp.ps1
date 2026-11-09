$ErrorActionPreference = 'Stop'
# Parse only leading wrapper options. PowerShell's normal named-parameter
# binding would otherwise consume native options such as --python appearing
# after the VKP command. Remaining arguments are passed as the original array.
$bindingOverride = $env:VKP_TESTED_VERSION_PATH
$entryPoint = 'cli'
$createBinding = $false
$bindingPython = ''
$expectedHead = ''
$wrapperArgs = @($args)
$position = 0
:wrapperOptions while ($position -lt $wrapperArgs.Count) {
    $option = ([string]$wrapperArgs[$position]).ToLowerInvariant()
    if ($option -eq '--') { $position++; break }
    if ($option -eq '-createbinding') { $createBinding = $true; $position++; continue }
    if ($option -notin @('-vkp_tested_version_path', '-entrypoint', '-python', '-expectedhead')) { break }
    if ($position + 1 -ge $wrapperArgs.Count) { throw "Missing value for wrapper option $option" }
    $value = [string]$wrapperArgs[$position + 1]
    switch ($option) {
        '-vkp_tested_version_path' { $bindingOverride = $value }
        '-entrypoint' { $entryPoint = $value }
        '-python' { $bindingPython = $value }
        '-expectedhead' { $expectedHead = $value }
    }
    $position += 2
}
[string[]]$VkpArgs = if ($position -lt $wrapperArgs.Count) { @($wrapperArgs[$position..($wrapperArgs.Count - 1)]) } else { @() }
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$entry = Join-Path $root 'src/video_knowledge_pipeline/tested_entry.py'
$bindingPath = if ($bindingOverride) {
    [IO.Path]::GetFullPath($bindingOverride)
} else {
    Join-Path $root '.local/tested-version.json'
}
if ($createBinding) {
    if (-not $bindingPython -or -not $expectedHead -or $VkpArgs.Count -gt 0) {
        throw 'Creation requires -Python and -ExpectedHead and accepts no native command.'
    }
    if (-not [IO.Path]::IsPathRooted($bindingPython) -or -not (Test-Path -LiteralPath $bindingPython -PathType Leaf)) {
        throw 'An explicit existing Python path is required; PATH fallback is disabled.'
    }
    & $bindingPython -B $entry --create-binding --source-root $root --python $bindingPython --expected-head $expectedHead --binding $bindingPath
    exit $LASTEXITCODE
}
if ($bindingPython -or $expectedHead) {
    throw '-Python/-ExpectedHead may only be used to create a new operator binding.'
}
if (-not (Test-Path -LiteralPath $bindingPath -PathType Leaf)) {
    throw 'The local tested-version binding is missing; no interpreter fallback will be selected.'
}
$binding = Get-Content -Raw -LiteralPath $bindingPath | ConvertFrom-Json
if ($binding.schema -ne 'video_knowledge_pipeline.tested_entry.v1' -or $binding.source_root -ne $root) {
    throw 'The binding schema/source checkout does not match this entrypoint.'
}
$pythonPath = [string]$binding.python.path
if (-not [IO.Path]::IsPathRooted($pythonPath) -or -not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw 'The bound interpreter is unavailable; PATH fallback is disabled.'
}
if ((Get-FileHash -LiteralPath $pythonPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $binding.python.executable_sha256) {
    throw 'The bound interpreter binary changed.'
}
if ($VkpArgs.Count -eq 1 -and $VkpArgs[0] -eq '--version-proof') {
    & $pythonPath -B $entry --binding $bindingPath --entry-point $entryPoint --version-proof
} else {
    & $pythonPath -B $entry --binding $bindingPath --entry-point $entryPoint -- @VkpArgs
}
exit $LASTEXITCODE
