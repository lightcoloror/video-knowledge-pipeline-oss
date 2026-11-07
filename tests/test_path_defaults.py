from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline.path_defaults import (
    openclaw_compose_path,
    port_record_path,
    provider_env_file,
    workspace_root,
)


def test_public_path_defaults_are_environment_configurable(monkeypatch) -> None:
    monkeypatch.setenv("VKP_WORKSPACE_ROOT", str(Path("workspace-root")))
    monkeypatch.setenv("VKP_PROVIDER_ENV_FILE", str(Path("secrets") / "provider.env"))
    monkeypatch.setenv("VKP_PORT_RECORD_PATH", str(Path("state") / "ports.md"))
    monkeypatch.setenv("VKP_OPENCLAW_COMPOSE_PATH", str(Path("openclaw") / "compose.yml"))

    assert workspace_root().name == "workspace-root"
    assert provider_env_file().name == "provider.env"
    assert port_record_path().name == "ports.md"
    assert openclaw_compose_path().name == "compose.yml"


def test_public_path_defaults_do_not_embed_a_windows_user_profile() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "video_knowledge_pipeline" / "path_defaults.py").read_text(
        encoding="utf-8"
    )
    assert "c:\\users\\" not in source.casefold()
    assert "d:\\used-by-codex" not in source.casefold()