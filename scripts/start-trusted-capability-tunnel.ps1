[CmdletBinding()]
param(
  [switch]$CredentialStatus,
  [switch]$ForgetCredential,
  [string]$ClientPath = '',
  [string]$ProfilePath = '',
  [string]$CredentialDirectory = ''
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$project = Split-Path -Parent $PSScriptRoot
$client = if ($ClientPath) { $ClientPath } else { [Environment]::GetEnvironmentVariable("VKP_TUNNEL_CLIENT_PATH", "Process") }
$profile = if ($ProfilePath) { $ProfilePath } else { [Environment]::GetEnvironmentVariable("VKP_TUNNEL_PROFILE_PATH", "Process") }
$secretDirectory = if ($CredentialDirectory) {
  $CredentialDirectory
} elseif ($env:VKP_TUNNEL_CREDENTIAL_DIRECTORY) {
  $env:VKP_TUNNEL_CREDENTIAL_DIRECTORY
} else {
  Join-Path $project ".local\tunnel-client\secrets"
}
$credentialFile = Join-Path $secretDirectory "vkp-tunnel-runtime.credential.xml"
$credentialUser = "VKP_TUNNEL_RUNTIME"

function Set-CurrentUserOnlyAcl {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )

  $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
  $acl = New-Object System.Security.AccessControl.FileSecurity
  $acl.SetOwner($identity)
  $acl.SetAccessRuleProtection($true, $false)
  $rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    $identity,
    [System.Security.AccessControl.FileSystemRights]::FullControl,
    [System.Security.AccessControl.AccessControlType]::Allow
  )
  [void]$acl.AddAccessRule($rule)
  Set-Acl -LiteralPath $Path -AclObject $acl
}

function Save-RuntimeCredential {
  param(
    [Parameter(Mandatory = $true)]
    [System.Security.SecureString]$Secret
  )

  New-Item -ItemType Directory -Force -Path $secretDirectory | Out-Null
  $credential = New-Object System.Management.Automation.PSCredential(
    $credentialUser,
    $Secret
  )
  $temporaryFile = "$credentialFile.$([Guid]::NewGuid().ToString('N')).tmp"
  try {
    $credential | Export-Clixml -LiteralPath $temporaryFile
    Set-CurrentUserOnlyAcl -Path $temporaryFile
    Move-Item -LiteralPath $temporaryFile -Destination $credentialFile -Force
  }
  finally {
    if (Test-Path -LiteralPath $temporaryFile) {
      Remove-Item -LiteralPath $temporaryFile -Force
    }
  }
}

function Load-RuntimeCredential {
  if (-not (Test-Path -LiteralPath $credentialFile -PathType Leaf)) {
    return $null
  }
  $credential = Import-Clixml -LiteralPath $credentialFile
  if ($credential -isnot [System.Management.Automation.PSCredential]) {
    throw "The saved tunnel credential has an invalid format."
  }
  if ($credential.UserName -ne $credentialUser) {
    throw "The saved tunnel credential has an invalid identity marker."
  }
  return $credential
}

if ($ForgetCredential) {
  if (Test-Path -LiteralPath $credentialFile) {
    Remove-Item -LiteralPath $credentialFile -Force
  }
  [pscustomobject]@{
    ok = $true
    credential_saved = $false
    credential_path = $credentialFile
  } | ConvertTo-Json -Compress
  exit 0
}

if ($CredentialStatus) {
  [pscustomobject]@{
    ok = $true
    credential_saved = (Test-Path -LiteralPath $credentialFile -PathType Leaf)
    storage = "Windows DPAPI via Export-Clixml"
    scope = "current Windows user on this machine"
    credential_path = $credentialFile
  } | ConvertTo-Json -Compress
  exit 0
}

if ([string]::IsNullOrWhiteSpace($client)) {
  throw "Pass -ClientPath or set VKP_TUNNEL_CLIENT_PATH."
}
if ([string]::IsNullOrWhiteSpace($profile)) {
  throw "Pass -ProfilePath or set VKP_TUNNEL_PROFILE_PATH."
}

if (-not (Test-Path -LiteralPath $client -PathType Leaf)) {
  throw "Tunnel client is missing: $client"
}
if (-not (Test-Path -LiteralPath $profile -PathType Leaf)) {
  throw "Tunnel profile is missing: $profile"
}

$injectedByScript = $false
try {
  if (-not $env:CONTROL_PLANE_API_KEY -and -not $env:OPENAI_API_KEY) {
    $credential = Load-RuntimeCredential
    if ($null -eq $credential) {
      $secure = Read-Host "OpenAI Tunnel runtime API key (saved with Windows DPAPI)" -AsSecureString
      $credential = New-Object System.Management.Automation.PSCredential(
        $credentialUser,
        $secure
      )
      $plainForValidation = $credential.GetNetworkCredential().Password
      try {
        if ([string]::IsNullOrWhiteSpace($plainForValidation)) {
          throw "The tunnel runtime API key cannot be empty."
        }
      }
      finally {
        $plainForValidation = $null
      }
      Save-RuntimeCredential -Secret $secure
      Write-Host "Saved the tunnel runtime credential with Windows DPAPI."
    }

    $env:CONTROL_PLANE_API_KEY = $credential.GetNetworkCredential().Password
    $injectedByScript = $true
  }

  & $client run --profile-file $profile
  exit $LASTEXITCODE
}
finally {
  if ($injectedByScript) {
    Remove-Item Env:CONTROL_PLANE_API_KEY -ErrorAction SilentlyContinue
  }
}
