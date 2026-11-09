param(
    [string]$VenvDir = "",
    [string]$PythonCommand = "python",
    [string]$CondaCommand = "conda",
    [string]$PythonVersion = "3.11",
    [string]$FunASRVersion = "1.3.30",
    [switch]$CreateVenv,
    [switch]$CreateCondaEnv,
    [switch]$InstallFunASR,
    [switch]$InstallCudaTorch,
    [switch]$InstallWhisperX,
    [switch]$InstallFasterWhisper,
    [switch]$AllowModelDownload
)

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
if (-not $VenvDir) {
    $VenvDir = Join-Path $projectRoot ".venv-lecture-asr"
}
$VenvDir = [System.IO.Path]::GetFullPath($VenvDir)

$venvPython = Join-Path $VenvDir "Scripts\python.exe"
$condaPython = Join-Path $VenvDir "python.exe"
$managedPython = if (Test-Path -LiteralPath $venvPython) { $venvPython } elseif (Test-Path -LiteralPath $condaPython) { $condaPython } else { $venvPython }
$venvBin = Join-Path $VenvDir "Scripts"
$funasrCommand = Join-Path $venvBin "funasr.exe"
$whisperxCommand = Join-Path $venvBin "whisperx.exe"
$fasterWhisperCommand = Join-Path $venvBin "faster-whisper.exe"
$venvExists = (Test-Path -LiteralPath $venvPython) -or (Test-Path -LiteralPath $condaPython)

$createCommand = ".\scripts\install-local-asr-env.ps1 -VenvDir `"$VenvDir`" -CreateVenv"
$createCondaCommand = ".\scripts\install-local-asr-env.ps1 -VenvDir `"$VenvDir`" -CreateCondaEnv -PythonVersion $PythonVersion"
$installFunASRCommand = ".\scripts\install-local-asr-env.ps1 -VenvDir `"$VenvDir`" -InstallFunASR"
$installCudaTorchCommand = ".\scripts\install-local-asr-env.ps1 -VenvDir `"$VenvDir`" -InstallCudaTorch"
$installWhisperXCommand = ".\scripts\install-local-asr-env.ps1 -VenvDir `"$VenvDir`" -InstallWhisperX"
$installFasterWhisperCommand = ".\scripts\install-local-asr-env.ps1 -VenvDir `"$VenvDir`" -InstallFasterWhisper"
$modelDownloadAllowed = [bool]$AllowModelDownload
if ($modelDownloadAllowed) {
    $env:LECTURE_ASR_ALLOW_MODEL_DOWNLOAD = "1"
}

if ($CreateVenv -and -not $venvExists) {
    & $PythonCommand -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create lecture ASR venv at $VenvDir"
    }
    $venvExists = Test-Path -LiteralPath $venvPython
    $managedPython = if (Test-Path -LiteralPath $venvPython) { $venvPython } elseif (Test-Path -LiteralPath $condaPython) { $condaPython } else { $venvPython }
}

if ($CreateCondaEnv -and -not $venvExists) {
    & $CondaCommand create -y -p $VenvDir "python=$PythonVersion"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create lecture ASR conda env at $VenvDir"
    }
    $venvExists = (Test-Path -LiteralPath $venvPython) -or (Test-Path -LiteralPath $condaPython)
    $managedPython = if (Test-Path -LiteralPath $venvPython) { $venvPython } elseif (Test-Path -LiteralPath $condaPython) { $condaPython } else { $venvPython }
}

function Invoke-PipInstall {
    param([string[]]$Packages)
    if (-not $venvExists) {
        throw "Lecture ASR environment not found under $VenvDir. Run with -CreateVenv or -CreateCondaEnv first."
    }
    & $managedPython -m pip install @Packages
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install ASR packages: $($Packages -join ' ')"
    }
}


function Invoke-PipInstallIndex {
    param([string[]]$Packages, [string]$IndexUrl)
    if (-not $venvExists) {
        throw "Lecture ASR environment not found under $VenvDir. Run with -CreateVenv or -CreateCondaEnv first."
    }
    & $managedPython -m pip install --upgrade --force-reinstall @Packages --index-url $IndexUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install packages from ${IndexUrl}: $($Packages -join ' ')"
    }
}
if ($InstallFunASR) {
    # Intent: keep SenseVoice + CAM++ on the upstream release that fixes timestamp alignment.
    # Decision: install the reviewed FunASR release instead of an unbounded latest version.
    # Reason: FunASR 1.3.9 reproduced the official #2706 None-boundary failure on VKP audio.
    # Evidence: 1.3.30 official timestamp/diarization unittest suite passed 13/13 locally.
    # Effective scope: newly installed local ASR environments; existing environments are not mutated.
    Invoke-PipInstall -Packages @("funasr==$FunASRVersion", "modelscope")
}
if ($InstallCudaTorch) {
    Invoke-PipInstallIndex -Packages @("torch==2.11.0+cu128", "torchaudio==2.11.0+cu128") -IndexUrl "https://download.pytorch.org/whl/cu128"
}

if ($InstallWhisperX) {
    Invoke-PipInstall -Packages @("whisperx")
}
if ($InstallFasterWhisper) {
    Invoke-PipInstall -Packages @("faster-whisper")
}

$venvExists = (Test-Path -LiteralPath $venvPython) -or (Test-Path -LiteralPath $condaPython)
$managedPython = if (Test-Path -LiteralPath $venvPython) { $venvPython } elseif (Test-Path -LiteralPath $condaPython) { $condaPython } else { $venvPython }

function Get-PythonVersionInfo {
    if (-not $venvExists) {
        return [pscustomobject]@{
            executable = $managedPython
            version = ""
            major = $null
            minor = $null
            asr_recommended = $false
            warning = "environment not created"
        }
    }
    $json = & $managedPython -c "import json, sys; print(json.dumps({'version': sys.version.split()[0], 'major': sys.version_info.major, 'minor': sys.version_info.minor}))"
    $parsed = $json | ConvertFrom-Json
    $recommended = [bool]($parsed.major -eq 3 -and $parsed.minor -ge 10 -and $parsed.minor -le 12)
    [pscustomobject]@{
        executable = $managedPython
        version = $parsed.version
        major = $parsed.major
        minor = $parsed.minor
        asr_recommended = $recommended
        warning = if ($recommended) { "" } else { "FunASR/WhisperX dependencies are more reliable on Python 3.10-3.12; Python $($parsed.version) may fail to build binary dependencies on Windows." }
    }
}

$pythonInfo = Get-PythonVersionInfo

function Test-Module {
    param([string]$ModuleName)
    if (-not $venvExists) {
        return [pscustomobject]@{
            module = $ModuleName
            available = $false
            returncode = $null
        }
    }
    & $managedPython -c "import importlib.util, sys; raise SystemExit(0 if importlib.util.find_spec('$ModuleName') else 1)" *> $null
    return [pscustomobject]@{
        module = $ModuleName
        available = [bool]($LASTEXITCODE -eq 0)
        returncode = $LASTEXITCODE
    }
}

$tools = @(
    [pscustomobject]@{
        name = "funasr"
        role = "primary Chinese ASR and SenseVoice runner"
        module = (Test-Module -ModuleName "funasr")
        command = $funasrCommand
        command_exists = [bool](Test-Path -LiteralPath $funasrCommand)
        env_command = "LECTURE_FUNASR_COMMAND"
        install_command = $installFunASRCommand
    },
    [pscustomobject]@{
        name = "sensevoice"
        role = "SenseVoice through FunASR CLI"
        module = (Test-Module -ModuleName "funasr")
        command = $funasrCommand
        command_exists = [bool](Test-Path -LiteralPath $funasrCommand)
        env_command = "LECTURE_FUNASR_COMMAND"
        install_command = $installFunASRCommand
    },
    [pscustomobject]@{
        name = "whisperx"
        role = "alignment-capable multilingual fallback"
        module = (Test-Module -ModuleName "whisperx")
        command = $whisperxCommand
        command_exists = [bool](Test-Path -LiteralPath $whisperxCommand)
        env_command = "LECTURE_WHISPERX_COMMAND"
        install_command = $installWhisperXCommand
    },
    [pscustomobject]@{
        name = "faster-whisper"
        role = "lighter local Whisper fallback"
        module = (Test-Module -ModuleName "faster_whisper")
        command = $fasterWhisperCommand
        command_exists = [bool](Test-Path -LiteralPath $fasterWhisperCommand)
        env_command = "LECTURE_FASTER_WHISPER_COMMAND"
        install_command = $installFasterWhisperCommand
    }
)

$availableTools = @($tools | Where-Object { $_.module.available -or $_.command_exists })
$envSnippet = @(
    "`$env:LECTURE_ASR_BIN_DIR=`"$venvBin`"",
    "`$env:LECTURE_FUNASR_COMMAND=`"$funasrCommand`"",
    "`$env:LECTURE_WHISPERX_COMMAND=`"$whisperxCommand`"",
    "`$env:LECTURE_FASTER_WHISPER_COMMAND=`"$fasterWhisperCommand`"",
    "`$env:LECTURE_ASR_ALLOW_MODEL_DOWNLOAD=`"$([int]$modelDownloadAllowed)`""
)
$nextActions = if (-not $venvExists) {
    @("run create_conda_command for Python 3.11, then install FunASR or another ASR package")
}
elseif (-not $pythonInfo.asr_recommended) {
    @("recreate the ASR environment with create_conda_command because the current Python version is not recommended for FunASR on Windows")
}
elseif ($availableTools.Count -eq 0) {
    @("run install_funasr_command for the recommended Chinese ASR path")
}
else {
    @("apply env_snippet before running plan-asr or BiliNote lecture extraction")
}

[pscustomobject]@{
    ok = [bool]($venvExists -and $availableTools.Count -gt 0)
    venv_exists = [bool]$venvExists
    venv_dir = if (Test-Path -LiteralPath $VenvDir) { (Resolve-Path -LiteralPath $VenvDir).Path } else { $VenvDir }
    venv_python = $managedPython
    python = $pythonInfo
    venv_bin = $venvBin
    tools = $tools
    available_tools = @($availableTools | ForEach-Object { $_.name })
    recommended_order = @("funasr", "sensevoice", "whisperx", "faster-whisper")
    funasr_required_version = $FunASRVersion
    model_download_allowed = $modelDownloadAllowed
    privacy = "Local ASR runs on this machine. Audio is not uploaded by this pipeline; first-run model download is controlled by LECTURE_ASR_ALLOW_MODEL_DOWNLOAD."
    expected_disk_usage = "SenseVoiceSmall roughly 1-2 GB; Whisper large models can require several GB."
    create_command = $createCommand
    create_conda_command = $createCondaCommand
    install_funasr_command = $installFunASRCommand
    install_cuda_torch_command = $installCudaTorchCommand
    install_whisperx_command = $installWhisperXCommand
    install_faster_whisper_command = $installFasterWhisperCommand
    env_snippet = $envSnippet
    next_actions = @($nextActions)
} | ConvertTo-Json -Depth 8
