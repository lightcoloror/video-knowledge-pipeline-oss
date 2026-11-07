from pathlib import Path
from typing import Any

import video_knowledge_pipeline.dolphin_python_runner as dolphin_runner
import video_knowledge_pipeline.model_api_legacy_import as legacy_import
import video_knowledge_pipeline.model_gateway as model_gateway
import video_knowledge_pipeline.punctuation_model_runner as punctuation_runner
import video_knowledge_pipeline.stage_cache as stage_cache
import video_knowledge_pipeline.storage as storage


def test_dolphin_output_delegates_to_shared_storage_owner(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, object]] = []
    monkeypatch.setattr(
        dolphin_runner,
        "write_json",
        lambda path, payload: calls.append((path, payload)),
    )
    output = tmp_path / "dolphin.json"
    payload: dict[str, Any] = {"ok": True, "segments": []}

    dolphin_runner._write_output(output, payload)

    assert calls == [(output, payload)]


def test_punctuation_failure_delegates_to_shared_storage_owner(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, object]] = []
    monkeypatch.setattr(
        punctuation_runner,
        "write_json",
        lambda path, payload: calls.append((path, payload)),
    )
    output = tmp_path / "punctuation.json"

    result = punctuation_runner._write_failure(output, "fixture", "expected")

    assert calls == [(output, result)]

def test_model_gateway_config_delegates_to_shared_text_owner(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        model_gateway,
        "write_text_atomic",
        lambda path, payload: calls.append((path, payload)),
    )
    monkeypatch.setattr(model_gateway, "_restrict_file", lambda _path: None)
    output = tmp_path / "litellm.yaml"

    result = model_gateway.render_litellm_config(
        settings_path=tmp_path / "settings.json",
        secrets_path=tmp_path / "secrets.json",
        output_path=output,
        write=True,
    )

    assert result["status"] == "rendered"
    assert len(calls) == 1
    assert calls[0][0] == output
    assert calls[0][1].startswith("model_list:\n")

def test_binary_replace_frontdoors_share_storage_owner() -> None:
    assert legacy_import.replace_file_with_retry is storage.replace_file_with_retry
    assert stage_cache.replace_file_with_retry is storage.replace_file_with_retry
    assert storage._replace_file_with_retry is storage.replace_file_with_retry


def test_shared_replace_retries_windows_permission_errors(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[Path, Path]] = []

    def fake_replace(source: Path, target: Path) -> None:
        calls.append((source, target))
        if len(calls) < 3:
            raise PermissionError("fixture")

    monkeypatch.setattr(storage.os, "replace", fake_replace)
    monkeypatch.setattr(storage.time, "sleep", lambda _seconds: None)
    source = tmp_path / "source.tmp"
    target = tmp_path / "target.bin"

    storage.replace_file_with_retry(source, target, attempts=3)

    assert calls == [(source, target), (source, target), (source, target)]
