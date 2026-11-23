$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$project = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $PSScriptRoot "start-trusted-capability-tunnel.ps1"
$runtimeDirectory = if ($env:VKP_TUNNEL_RUNTIME_DIRECTORY) {
  $env:VKP_TUNNEL_RUNTIME_DIRECTORY
} else {
  Join-Path $project ".local\tunnel-client\runtime"
}
$logFile = Join-Path $runtimeDirectory "vkp-local-8081.log"
$pidFile = Join-Path $runtimeDirectory "vkp-local-8081.pid"

if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
  throw "The secure tunnel launcher is missing: $launcher"
}

New-Item -ItemType Directory -Force -Path $runtimeDirectory | Out-Null

$previousHealth = $env:HEALTH_LISTEN_ADDR
$previousLog = $env:LOG_FILE
$previousPid = $env:PID_FILE
try {
  $env:HEALTH_LISTEN_ADDR = "127.0.0.1:8081"
  $env:LOG_FILE = $logFile
  $env:PID_FILE = $pidFile
  & $launcher @args
  exit $LASTEXITCODE
}
finally {
  if ($null -eq $previousHealth) {
    Remove-Item Env:HEALTH_LISTEN_ADDR -ErrorAction SilentlyContinue
  }
  else {
    $env:HEALTH_LISTEN_ADDR = $previousHealth
  }
  if ($null -eq $previousLog) {
    Remove-Item Env:LOG_FILE -ErrorAction SilentlyContinue
  }
  else {
    $env:LOG_FILE = $previousLog
  }
  if ($null -eq $previousPid) {
    Remove-Item Env:PID_FILE -ErrorAction SilentlyContinue
  }
  else {
    $env:PID_FILE = $previousPid
  }
}
