from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline import model_api_legacy_import as legacy_import
from video_knowledge_pipeline import model_api_settings as settings_module
from video_knowledge_pipeline.model_gateway import main as model_gateway_main
from video_knowledge_pipeline.storage import write_json as real_write_json


SECRET = "legacy-secret-must-never-appear"


def _legacy_sources(root: Path, *, malformed_ark: bool = False) -> Path:
    bundle = root / "bundle"
    bundle.mkdir(parents=True)
    (root / ".local").mkdir(parents=True)
    (root / ".local" / "model-connector.env").write_text(
        f"ARK_API_KEY={SECRET}\n",
        encoding="utf-8",
    )
    (root / ".local" / "vision.env").write_text(
        "LECTURE_VISION_PROVIDER=agnes\nAGNES_API_KEY=agnes-secret\n",
        encoding="utf-8",
    )
    if malformed_ark:
        (bundle / "volcengine-provider-public.json").write_text("{", encoding="utf-8")
    else:
        (bundle / "volcengine-provider-public.json").write_text(
            json.dumps(
                {
                    "provider": "volcengine_coding_plan",
                    "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
                    "model": "ark-code-latest",
                    "timeout_seconds": 90,
                }
            ),
            encoding="utf-8",
        )
    (bundle / "local-vlm-serving-smoke.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.local_vlm_serving_smoke.v1",
                "execute": False,
                "provider": "local_qwen_vl",
                "profile": {
                    "provider": "local_qwen_vl",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "model": "Qwen/Qwen2.5-VL-3B-Instruct",
                },
            }
        ),
        encoding="utf-8",
    )
    (bundle / "model-settings.json").write_text("{}", encoding="utf-8")
    (bundle / "vision-export-consent.json").write_text("{}", encoding="utf-8")
    return bundle


def _configure_test_root(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(legacy_import, "project_root", lambda: root)
    monkeypatch.setattr(settings_module, "_protect_secret", lambda value: "cipher:" + value.encode().hex())


def test_safe_import_preview_is_read_only_and_redacted(tmp_path: Path, monkeypatch) -> None:
    bundle = _legacy_sources(tmp_path)
    _configure_test_root(monkeypatch, tmp_path)
    settings_path = tmp_path / "target" / "settings.json"
    secrets_path = tmp_path / "target" / "secrets.json"

    result = legacy_import.safe_import_legacy_model_api_settings(
        bundle_dir=bundle,
        settings_path=settings_path,
        secrets_path=secrets_path,
    )

    rendered = json.dumps(result, ensure_ascii=False)
    assert result["status"] == "planned"
    assert result["profile_count"] == 2
    assert result["route_pool_count"] == 3
    assert result["task_binding_count"] == 7
    assert result["security"]["old_consents_reused"] is False
    assert result["security"]["network_calls"] == 0
    assert SECRET not in rendered
    assert "agnes-secret" not in rendered
    assert any(row["id"] == "remote-agnes" for row in result["excluded"])
    assert result["ignored_legacy_authorizations"]["count"] == 1
    assert not settings_path.exists()
    assert not secrets_path.exists()


def test_safe_import_executes_once_with_dpapi_only_and_remote_default(tmp_path: Path, monkeypatch) -> None:
    bundle = _legacy_sources(tmp_path)
    _configure_test_root(monkeypatch, tmp_path)
    settings_path = tmp_path / "target" / "settings.json"
    secrets_path = tmp_path / "target" / "secrets.json"
    source_hashes = {path: path.read_bytes() for path in bundle.iterdir() if path.is_file()}

    result = legacy_import.safe_import_legacy_model_api_settings(
        bundle_dir=bundle,
        settings_path=settings_path,
        secrets_path=secrets_path,
        execute=True,
    )

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    secrets = json.loads(secrets_path.read_text(encoding="utf-8"))
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert result["status"] == "imported"
    assert settings["schema"] == "video_knowledge_pipeline.local_model_api_settings.v2"
    assert {row["id"] for row in settings["profiles"]} == {"remote-ark", "local-qwen-vl"}
    assert settings["route_bindings"]["semantic_frame"]["default_location"] == "remote"
    assert settings["route_bindings"]["semantic_frame"]["local_pool_id"]
    assert settings["route_bindings"]["semantic_frame"]["remote_pool_id"]
    assert settings["route_bindings"]["text_llm"]["default_location"] == "remote"
    assert secrets["items"]["remote-ark"]["ciphertext"].startswith("cipher:")
    assert SECRET not in settings_path.read_text(encoding="utf-8")
    assert SECRET not in report
    assert "agnes-secret" not in report
    assert all(path.read_bytes() == payload for path, payload in source_hashes.items())
    assert result["verification"]["network_calls"] == 0

    repeated = legacy_import.safe_import_legacy_model_api_settings(
        bundle_dir=bundle,
        settings_path=settings_path,
        secrets_path=secrets_path,
        execute=True,
    )
    assert repeated["status"] == "already_imported"
    assert json.loads(Path(repeated["report_path"]).read_text(encoding="utf-8"))["status"] == "already_imported"


def test_safe_import_refuses_to_overwrite_nonmatching_v2_target(tmp_path: Path, monkeypatch) -> None:
    bundle = _legacy_sources(tmp_path)
    _configure_test_root(monkeypatch, tmp_path)
    settings_path = tmp_path / "target" / "settings.json"
    secrets_path = tmp_path / "target" / "secrets.json"
    imported = legacy_import.safe_import_legacy_model_api_settings(
        bundle_dir=bundle,
        settings_path=settings_path,
        secrets_path=secrets_path,
        execute=True,
    )
    assert imported["status"] == "imported"
    data = json.loads(settings_path.read_text(encoding="utf-8"))
    data["profiles"][0]["model"] = "operator-edited-model"
    settings_path.write_text(json.dumps(data), encoding="utf-8")
    before = settings_path.read_bytes()

    blocked = legacy_import.safe_import_legacy_model_api_settings(
        bundle_dir=bundle,
        settings_path=settings_path,
        secrets_path=secrets_path,
        execute=True,
    )

    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "target_settings_not_empty"
    assert settings_path.read_bytes() == before


def test_safe_import_fails_closed_on_malformed_source(tmp_path: Path, monkeypatch) -> None:
    bundle = _legacy_sources(tmp_path, malformed_ark=True)
    _configure_test_root(monkeypatch, tmp_path)
    settings_path = tmp_path / "target" / "settings.json"

    result = legacy_import.safe_import_legacy_model_api_settings(
        bundle_dir=bundle,
        settings_path=settings_path,
        execute=True,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "source_or_target_invalid"
    assert SECRET not in json.dumps(result)
    assert not settings_path.exists()


def test_safe_import_rolls_back_both_target_files_on_write_failure(tmp_path: Path, monkeypatch) -> None:
    bundle = _legacy_sources(tmp_path)
    _configure_test_root(monkeypatch, tmp_path)
    settings_path = tmp_path / "target" / "settings.json"
    secrets_path = tmp_path / "target" / "secrets.json"
    calls = 0

    def failing_write(path: Path, data: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated settings write failure")
        real_write_json(path, data)

    monkeypatch.setattr(legacy_import, "write_json", failing_write)
    result = legacy_import.safe_import_legacy_model_api_settings(
        bundle_dir=bundle,
        settings_path=settings_path,
        secrets_path=secrets_path,
        execute=True,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "write_or_verification_failed"
    assert not settings_path.exists()
    assert not secrets_path.exists()


def test_safe_import_rolls_back_both_target_files_on_verification_failure(tmp_path: Path, monkeypatch) -> None:
    bundle = _legacy_sources(tmp_path)
    _configure_test_root(monkeypatch, tmp_path)
    settings_path = tmp_path / "target" / "settings.json"
    secrets_path = tmp_path / "target" / "secrets.json"
    original_load = legacy_import.load_model_api_settings
    calls = 0

    def failing_verification(path: Path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("simulated verification failure")
        return original_load(path)

    monkeypatch.setattr(legacy_import, "load_model_api_settings", failing_verification)
    result = legacy_import.safe_import_legacy_model_api_settings(
        bundle_dir=bundle,
        settings_path=settings_path,
        secrets_path=secrets_path,
        execute=True,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "write_or_verification_failed"
    assert not settings_path.exists()
    assert not secrets_path.exists()

def test_model_gateway_import_legacy_cli_defaults_to_preview(tmp_path: Path, monkeypatch, capsys) -> None:
    bundle = _legacy_sources(tmp_path)
    _configure_test_root(monkeypatch, tmp_path)
    settings_path = tmp_path / "target" / "settings.json"

    exit_code = model_gateway_main(
        [
            "--settings-path",
            str(settings_path),
            "import-legacy",
            "--bundle-dir",
            str(bundle),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"status": "planned"' in output
    assert SECRET not in output
    assert not settings_path.exists()
