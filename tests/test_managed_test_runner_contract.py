from __future__ import annotations

from pathlib import Path
import re


RUNNER = Path(__file__).parents[1] / "scripts" / "run-tests-managed.ps1"


def test_managed_runner_is_checkout_relative_and_uses_explicit_roots() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert not re.search(r"[A-Za-z]:[\\/]", text)
    assert "Split-Path -Parent $PSCommandPath" in text
    assert "Join-Path $repo '.venv'" in text
    assert "VKP_WINDOWS_TEST_RUNTIME" in text
    assert "VKP_MANAGED_TEST_ROOT" in text
    assert "GetFolderPath('LocalApplicationData')" not in text
    assert "$fallbackManagedRoot" not in text
    assert "Pass -RuntimePath or set VKP_WINDOWS_TEST_RUNTIME explicitly." in text
    assert "Pass -ManagedRoot or set VKP_MANAGED_TEST_ROOT explicitly." in text
    assert "[switch]$PreflightOnly" in text
