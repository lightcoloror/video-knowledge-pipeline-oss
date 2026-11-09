$ErrorActionPreference = "Stop"
$root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONIOENCODING = "utf-8"

# Intent: expose the pinned sherpa-onnx adapter through a stable local front
# door without modifying the busy shared video-knowledge.ps1 dispatcher.
# Decision: forward arguments to the module CLI; explicit model/command paths
# remain part of the exact plan.
# Reason: this avoids a second execution service and preserves fail-closed
# preview/execute behavior.
# Evidence: the module CLI and offline candidate-evidence regressions are
# documented in docs/video-evidence-entry-wrappers.md.
# Effective scope: local planning/execution only; no download, upload, fallback,
# provider call, or primary transcript mutation.
python -m video_knowledge_pipeline.speaker_diarization_evidence @args
$exitCode = $LASTEXITCODE
exit $exitCode
