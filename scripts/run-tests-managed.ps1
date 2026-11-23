[CmdletBinding()]
param(
    [string]$RuntimePath = '',
    [double]$TimeoutSeconds = 1800,
    [string[]]$TestArgs = @('-q'),
    [string]$ProjectVenvPath = '',
    [string]$RepoPath = '',
    [string]$ManagedRoot = '',
    [switch]$PreflightOnly
)
$ErrorActionPreference = 'Stop'
$scriptRoot = Split-Path -Parent $PSCommandPath
$repo = if ($RepoPath) {
    [IO.Path]::GetFullPath($RepoPath)
} else {
    [IO.Path]::GetFullPath((Join-Path $scriptRoot '..'))
}
if (-not $ProjectVenvPath) {
    $ProjectVenvPath = Join-Path $repo '.venv'
}
if (-not $RuntimePath) {
    $runtimeFromEnvironment = [Environment]::GetEnvironmentVariable('VKP_WINDOWS_TEST_RUNTIME', 'Process')
    $RuntimePath = if ($runtimeFromEnvironment) {
        $runtimeFromEnvironment
    } else {
        throw 'Pass -RuntimePath or set VKP_WINDOWS_TEST_RUNTIME explicitly.'
    }
}
if (-not (Test-Path -LiteralPath $RuntimePath -PathType Leaf)) {
    throw "Managed Windows test runtime not found: $RuntimePath. Pass -RuntimePath or set VKP_WINDOWS_TEST_RUNTIME."
}
if (-not $ManagedRoot) {
    $managedRootFromEnvironment = [Environment]::GetEnvironmentVariable('VKP_MANAGED_TEST_ROOT', 'Process')
    $ManagedRoot = if ($managedRootFromEnvironment) {
        $managedRootFromEnvironment
    } else {
        throw 'Pass -ManagedRoot or set VKP_MANAGED_TEST_ROOT explicitly.'
    }
}
$managedRoot = [IO.Path]::GetFullPath($ManagedRoot)
New-Item -ItemType Directory -Path $managedRoot -Force | Out-Null
$managedProbe = Join-Path $managedRoot ('.vkp-write-probe-' + [Guid]::NewGuid().ToString('N'))
[IO.File]::WriteAllText($managedProbe, 'probe')
Remove-Item -LiteralPath $managedProbe -Force -ErrorAction Stop
$vkpSitePackages = Join-Path $ProjectVenvPath 'Lib\site-packages'
if (-not (Test-Path -LiteralPath $vkpSitePackages -PathType Container)) {
    throw "VKP venv site-packages not found: $vkpSitePackages"
}
$vkpPriorPythonPath = [Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process')
$env:PYTHONPATH = if ($vkpPriorPythonPath) {
    "$vkpSitePackages$([IO.Path]::PathSeparator)$vkpPriorPythonPath"
} else {
    $vkpSitePackages
}
# Intent: run pytest through the ACL-safe shared runtime while binding the
# project's already-installed runtime dependencies. This does not install or
# mutate packages; it fails before collection when the managed environment is
# incomplete.
python -c "import pytest, rapidfuzz, transitions, psutil"
if ($LASTEXITCODE -ne 0) {
    throw 'Managed VKP test dependencies are incomplete: pytest/rapidfuzz/transitions/psutil are required.'
}
if ($PreflightOnly) {
    [ordered]@{
        status = 'ready'
        repo = $repo
        runtime_path = [IO.Path]::GetFullPath($RuntimePath)
        managed_root = $managedRoot
        project_venv = [IO.Path]::GetFullPath($ProjectVenvPath)
        dependencies = @('pytest', 'rapidfuzz', 'transitions', 'psutil')
    } | ConvertTo-Json -Depth 4
    exit 0
}
& $RuntimePath adapter -Purpose 'video-knowledge-pipeline-tests' -Adapter pytest -WorkingDirectory $repo -ManagedRoot $managedRoot -TimeoutSeconds $TimeoutSeconds -ArgumentsJson ($TestArgs | ConvertTo-Json -Compress)
exit $LASTEXITCODE
