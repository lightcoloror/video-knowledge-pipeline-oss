from __future__ import annotations

from pathlib import Path


def test_explicit_model_settings_path_drives_broker_runtime_route() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root / "scripts" / "start-trusted-capability-broker-http.ps1"
    ).read_text(encoding="utf-8-sig")

    assert "$env:VKP_MODEL_API_SETTINGS_PATH = $settingsFile" in script
    assert "if ($ModelSettingsPath)" in script


def test_explicit_model_secrets_path_is_forwarded_to_broker_runtime() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root / "scripts" / "start-trusted-capability-broker-http.ps1"
    ).read_text(encoding="utf-8-sig")

    assert '[string]$ModelSecretsPath = ""' in script
    assert "if ($ModelSecretsPath)" in script
    assert "$env:VKP_MODEL_API_SECRETS_PATH =" in script
    assert "Resolve-Path -LiteralPath $ModelSecretsPath" in script
