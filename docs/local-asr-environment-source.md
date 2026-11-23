# Local ASR Environment Provisioning and Diagnostics Source Guide

## 1. Purpose and Scope
This document specifies the operational contracts and diagnostic behavior for `scripts/install-local-asr-env.ps1` under the AGPL-3.0-only license as part of first-party VKP tooling. The script manages Windows local Python virtual environments, selective ASR package installation, and tool diagnostics for Chinese and multilingual transcription engines. No model weights, precompiled binaries, third-party packages, or customer audio files are distributed within this source release.

## 2. Parameter Switches and Execution Modes
The script accepts parameters for directory targeting (`$VenvDir`, defaulting to `.venv-lecture-asr` at the project root), Python selection, and operational switches:
- **Environment Creation**: The `-CreateVenv` switch initializes standard Python virtual environments, while `-CreateCondaEnv` targets Conda installations (defaulting to Python 3.11).
- **Package Installation Switches**:
  - `-InstallFunASR`: Requests installation of `funasr` at the configurable FunASRVersion (default 1.3.30) and `modelscope` via pip.
  - `-InstallCudaTorch`: Executes pip with `--upgrade --force-reinstall` targeting CUDA wheels (`torch==2.11.0+cu128`, `torchaudio==2.11.0+cu128`) from the official repository `https://download.pytorch.org/whl/cu128`.
  - `-InstallWhisperX` and `-InstallFasterWhisper`: Installs respective fallback ASR packages.
- **Diagnostic Probing (No-Install Mode)**: Running without installation flags does not represent a passive static file check. If an environment exists, the script immediately invokes Python subprocesses to interrogate version details and evaluate module presence via `importlib.util.find_spec` for `funasr`, `whisperx`, and `faster_whisper`.
- **Environment Mutation**: When install switches are specified against an existing environment, existing packages may be changed, especially by CUDA force-reinstall. The historical non-mutation comment must not be read as an unconditional guarantee; this review does not establish successful installs or side-effect-free Python startup.

## 3. Governance and Subprocess Boundaries
- **Model Download Governance**: Passing `-AllowModelDownload` sets `$env:LECTURE_ASR_ALLOW_MODEL_DOWNLOAD = "1"`. However, omitting or passing false does not clear an inherited variable from the parent session. The reported boolean does not enforce an OS-level egress sandbox or guarantee model repository availability.
- **Subprocess Diagnostic Exit Handling**: The Python version probe executes a one-line JSON print without validating `$LASTEXITCODE` prior to `ConvertFrom-Json`, so return-code handling at that probe is not explicitly guarded. Overall failure behavior has not been tested; JSON parsing or host behavior may still raise errors.
- **Historical Claims**: Mentions of 13/13 test passes and version pins reflect historical snapshots, not active verifications of external package indexes.

## 4. Verification Status and Limits
All installation steps, conda/pip commands, module probes, and network downloads remain strictly **UNEXECUTED (NOT RUN)**. Review is restricted to full 231-line source inspection. Compatibility across external hardware, CUDA runtimes, and upstream packages remains unverified.

The complete 231-line script is already LF and byte-identical to the selected
tracked first-party source blob; no code, guard, parameter order or URL changed.
PowerShell syntax parsing and all target/runtime tests were NOT RUN. No
environment, dependency, package index, audio, model, credential or configuration
was accessed by executing this script. Python reliability and model-size
estimates in the source are also historical, unverified statements.
AGPL-3.0-only covers this first-party source increment, not third-party packages,
models or weights. Publishing source is not authorization to execute its
installer. Explicit VenvDir can select a location outside the project, and path
parameters do not create an OS sandbox. Effective inherited download state and
downstream package/provider behavior require separate verification before use.
