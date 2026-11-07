from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_import_does_not_require_autopilot_runtime_dependency() -> None:
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo / "src") + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import builtins; "
                "original_import = builtins.__import__; "
                "builtins.__import__ = lambda name, *args, **kwargs: "
                "(_ for _ in ()).throw(ModuleNotFoundError('blocked transitions')) "
                "if name.split('.', 1)[0] == 'transitions' "
                "else original_import(name, *args, **kwargs); "
                "import video_knowledge_pipeline.cli"
            ),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_video_knowledge_wrapper_routes_all_full_cli_commands_through_project_python() -> None:
    repo = Path(__file__).resolve().parents[1]
    wrapper = (repo / "scripts" / "video-knowledge.ps1").read_text(encoding="utf-8")

    assert '$venvPython = Join-Path $root ".venv\\Scripts\\python.exe"' in wrapper
    assert "python -m video_knowledge_pipeline.cli @args" not in wrapper
    assert wrapper.count(
        "& $pythonCommand -m video_knowledge_pipeline.cli @args"
    ) >= 2
    assert (
        "& $pythonCommand -m video_knowledge_pipeline.model_destination_allowlist"
        in wrapper
    )


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell wrapper is Windows-only")
def test_video_knowledge_wrapper_propagates_cli_failure() -> None:
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("Windows PowerShell is unavailable")
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["VKP_MODEL_CONNECTOR_ALLOWED_DESTINATIONS"] = "test.invalid"

    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(repo / "scripts" / "video-knowledge.ps1"),
            "definitely-not-a-vkp-command",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=False,
    )

    assert completed.returncode != 0


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell wrapper is Windows-only")
def test_video_knowledge_wrapper_loads_prepare_command_from_project_venv() -> None:
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if not powershell:
        pytest.skip("PowerShell is unavailable")
    repo = Path(__file__).resolve().parents[1]
    venv_python = repo / ".venv" / "Scripts" / "python.exe"
    if not venv_python.is_file():
        pytest.skip("Project venv is unavailable")
    env = dict(os.environ)
    env["VKP_MODEL_CONNECTOR_ALLOWED_DESTINATIONS"] = "test.invalid"

    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(repo / "scripts" / "video-knowledge.ps1"),
            "prepare-local-video-run",
            "--help",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage: video-knowledge prepare-local-video-run" in completed.stdout
    assert "ModuleNotFoundError" not in completed.stderr
