$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")

# The registry health probe must not pay the cost of loading or inspecting the
# ASR/model stack.  Keep this dispatch before environment-file parsing and the
# remote-destination allowlist; detailed capability diagnostics remain under
# `asr-env-status`.
if ($args.Count -gt 0 -and $args[0] -eq "quick-health") {
    $env:PYTHONPATH = Join-Path $root "src"
    $env:PYTHONIOENCODING = "utf-8"
    $quickArgs = @()
    if ($args.Count -gt 1) {
        $quickArgs = $args[1..($args.Count - 1)]
    }
    python -m video_knowledge_pipeline.quick_health @quickArgs
    $exitCode = $LASTEXITCODE
    exit $exitCode
}

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$pythonCommand = if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $venvPython
} else {
    "python"
}

# Hash-bound result recovery is metadata-only and needs no Provider environment.
if ($args.Count -gt 0 -and $args[0] -eq "versioned-semantic-run") {
    $env:PYTHONPATH = Join-Path $root "src"
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    $runArgs = @()
    if ($args.Count -gt 1) { $runArgs = $args[1..($args.Count - 1)] }
    & $pythonCommand -m video_knowledge_pipeline.versioned_semantic_run @runArgs
    exit $LASTEXITCODE
}

# Version-selected consumers are read-only with respect to the source bundle.
if ($args.Count -gt 0 -and $args[0] -in @("prepare-review-session", "transcript-semantic-correction-pack", "transcript-semantic-correction-codex-draft") -and $args -contains "--run-id") {
    $env:PYTHONPATH = Join-Path $root "src"
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    & $pythonCommand -m video_knowledge_pipeline.cli @args
    exit $LASTEXITCODE
}

if ($args.Count -gt 0 -and $args[0] -eq "run-artifact-recovery") {
    $env:PYTHONPATH = Join-Path $root "src"
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    $recoveryArgs = @()
    if ($args.Count -gt 1) { $recoveryArgs = $args[1..($args.Count - 1)] }
    & $pythonCommand -m video_knowledge_pipeline.run_artifact_recovery @recoveryArgs
    exit $LASTEXITCODE
}

# These project-local control commands are deliberately offline, candidate-only
# or plan-only. Route them
# through the repository venv before parsing provider environment files so the
# command gets the project's validated Python dependencies without reading
# unrelated API settings or widening execution authority.
$offlineProjectCommands = @(
    "export-creator-report",
    "creator-report-status",
    "video-closure-doctor",
    "silero-vad-candidate",
    "asr-vad-independent-crosscheck",
    "video-closure-run",
    "video-closure-resume",
    "video-closure-status",
    "media-artifact-doctor",
    "media-artifact-preflight",
    "media-artifact-scan",
    "media-artifact-explain",
    "media-artifact-plan-gc",
    "media-artifact-verify-plan",
    "media-artifact-restore",
    "transcript-semantic-restore",
    "prepare-summary-claim-review",
    "import-summary-claim-review",
    "summary-claim-review-status"
)
if ($args.Count -gt 0 -and $args[0] -in $offlineProjectCommands) {
    $env:PYTHONPATH = Join-Path $root "src"
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    & $pythonCommand -m video_knowledge_pipeline.cli @args
    $exitCode = $LASTEXITCODE
    exit $exitCode
}

foreach ($envFile in @((Join-Path $root ".local\video-knowledge.env"), (Join-Path $root ".local\vision.env"))) {
    if (Test-Path -LiteralPath $envFile) {
        foreach ($line in Get-Content -LiteralPath $envFile) {
            $trimmed = $line.Trim()
            if (-not $trimmed -or $trimmed.StartsWith("#")) {
                continue
            }
            if ($trimmed -match '^(?:\$env:)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
                $name = $Matches[1]
                $value = $Matches[2].Trim()
                if ($value.Length -ge 2) {
                    $quote = $value.Substring(0, 1)
                    if (($quote -eq '"' -or $quote -eq "'") -and $value.EndsWith($quote)) {
                        $value = $value.Substring(1, $value.Length - 2)
                    }
                }
                if ($name -like "LECTURE_*" -or $name -in @("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "AGNES_API_KEY", "ARK_API_KEY", "VOLCENGINE_API_KEY", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LOCAL_QWEN_VL_API_KEY", "LOCAL_QWEN_VL_BASE_URL", "LOCAL_QWEN_VL_MODEL", "LOCAL_VLM_API_KEY", "LOCAL_VLM_BASE_URL", "LOCAL_VLM_MODEL")) {
                    [Environment]::SetEnvironmentVariable($name, $value, "Process")
                }
            }
        }
    }
}
$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONIOENCODING = "utf-8"
$env:LITELLM_TELEMETRY = "False"
$env:LITELLM_LOCAL_MODEL_COST_MAP = "True"

# Keep the CLI's consent front door aligned with the Broker. The reviewed
# model settings are the single source of configured remote destinations;
# individual plans and consent v2 still narrow this set to exact routes.
if (-not $env:VKP_MODEL_CONNECTOR_ALLOWED_ROOTS) {
    $roots = @([string]$root)
    $managedOutputRoot = Join-Path (Split-Path -Parent $root) "video-knowledge-output"
    if (Test-Path -LiteralPath $managedOutputRoot -PathType Container) {
        $roots += (Resolve-Path -LiteralPath $managedOutputRoot).Path
    }
    $env:VKP_MODEL_CONNECTOR_ALLOWED_ROOTS = ($roots -join [IO.Path]::PathSeparator)
}
if (-not $env:VKP_MODEL_CONNECTOR_ALLOWED_DESTINATIONS) {
    $settingsPath = if ($env:VKP_MODEL_API_SETTINGS_PATH) {
        $env:VKP_MODEL_API_SETTINGS_PATH
    } else {
        Join-Path $root ".local\model-api-settings.json"
    }
    if (Test-Path -LiteralPath $settingsPath) {
        $rawStatus = & $pythonCommand -m video_knowledge_pipeline.model_destination_allowlist `
            --settings-path $settingsPath
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to derive configured remote model destinations."
        }
        $destinationStatus = ConvertFrom-Json -InputObject ($rawStatus -join "`n")
        $env:VKP_MODEL_CONNECTOR_ALLOWED_DESTINATIONS = `
            (@($destinationStatus.destinations) -join ",")
    }
}
& $pythonCommand -m video_knowledge_pipeline.cli @args
$exitCode = $LASTEXITCODE
exit $exitCode
