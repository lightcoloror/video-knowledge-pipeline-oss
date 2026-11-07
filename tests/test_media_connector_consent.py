from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.cli import build_parser
from video_knowledge_pipeline.media_connector_consent import (
    create_media_connector_consent,
    media_connector_consent_status,
    reserve_media_connector_attempt,
)
from video_knowledge_pipeline.media_route_settings import (
    CONTROL_PLANE_BASE_URL,
    build_media_route_snapshot,
    load_media_route_settings,
    media_route_settings_status,
    save_media_route_settings,
)
from video_knowledge_pipeline.model_connector_consent import (
    SCHEMA_V2,
    _payload_sha256,
    validate_model_connector_consent,
)
from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.trusted_model_connector import execute_consented_model_task
from video_knowledge_pipeline.trusted_model_connector_policy import TrustedModelConnectorPolicy


def _media(tmp_path: Path) -> Path:
    path = tmp_path / "lesson.mp4"
    path.write_bytes(b"fake-video-for-consent")
    return path


def _settings(tmp_path: Path, destination: str = "https://upload.example") -> Path:
    path = tmp_path / "media-route-settings.json"
    save_media_route_settings(
        upload_destinations=[destination],
        settings_path=path,
        max_poll_attempts=3,
        poll_interval_seconds=0,
        timeout_seconds=30,
    )
    return path


def _policy(tmp_path: Path, *destinations: str) -> TrustedModelConnectorPolicy:
    allowed = {
        "amk.cn-beijing.volces.com",
        "upload.example",
        *destinations,
    }
    return TrustedModelConnectorPolicy(
        allowed_roots=(tmp_path.resolve(),),
        allowed_destinations=frozenset(allowed),
    )


def _consent(
    tmp_path: Path,
    *,
    settings_path: Path,
    policy: TrustedModelConnectorPolicy,
    max_calls: int = 1,
) -> dict[str, object]:
    return create_media_connector_consent(
        tmp_path,
        task="scene_segmentation",
        artifact_paths=[_media(tmp_path)],
        settings_path=settings_path,
        max_calls=max_calls,
        max_estimated_cost_usd=0.2,
        max_cost_per_call_usd=0.1 if max_calls > 1 else 0.2,
        confirm_data_export=True,
        policy=policy,
    )


def test_media_route_is_content_addressed_and_secretless(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    first = build_media_route_snapshot("scene_segmentation", settings_path=settings)
    second = build_media_route_snapshot("scene_segmentation", settings_path=settings)

    assert first["execution_ready"] is True
    assert first["route"]["route_revision"] == second["route"]["route_revision"]
    assert first["route"]["virtual_model"].startswith("vkp-media-scene-segmentation-")
    assert first["route"]["destinations"] == [
        CONTROL_PLANE_BASE_URL,
    ]
    deployment = first["route"]["deployments"][0]
    assert deployment["interface"] == "mediakit_async_v1"
    assert deployment["environment_bindings"] == [
        {"param": "api_key", "env": "MEDIAKIT_API_KEY", "required": True}
    ]
    serialised = json.dumps(first, ensure_ascii=False)
    assert "Bearer " not in serialised
    assert first["settings"]["secrets_persisted"] is False
    assert first["credential"]["value_exposed"] is False

    save_media_route_settings(
        upload_destinations=["https://other-upload.example"],
        settings_path=settings,
        max_poll_attempts=3,
        poll_interval_seconds=0,
        timeout_seconds=30,
    )
    changed = build_media_route_snapshot("scene_segmentation", settings_path=settings)
    assert changed["route"]["route_revision"] == first["route"]["route_revision"]
    assert changed["route"]["route_snapshot_sha256"] == first["route"]["route_snapshot_sha256"]


def test_media_route_requires_audited_https_upload_origins(tmp_path: Path) -> None:
    missing = media_route_settings_status(
        task="scene_segmentation",
        settings_path=tmp_path / "missing.json",
    )
    assert missing["status"] == "ready_for_consent"
    assert missing["execution_ready"] is True

    with pytest.raises(ValueError, match="HTTPS origin"):
        save_media_route_settings(
            upload_destinations=["http://upload.example"],
            settings_path=tmp_path / "bad-http.json",
        )
    with pytest.raises(ValueError, match="without a path"):
        save_media_route_settings(
            upload_destinations=["https://upload.example/private/path"],
            settings_path=tmp_path / "bad-path.json",
        )
    bad_control_plane = tmp_path / "bad-control-plane.json"
    bad_control_plane.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.media_route_settings.v1",
                "provider": "volcengine_mediakit",
                "route_id": "mediakit-remote-approved",
                "control_plane_base_url": "https://arbitrary.example",
                "upload_destinations": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fixed"):
        load_media_route_settings(bad_control_plane)


def test_media_consent_reuses_v2_manifest_route_and_destination_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    policy = _policy(tmp_path)
    consent = _consent(tmp_path, settings_path=settings, policy=policy)
    route = build_media_route_snapshot("scene_segmentation", settings_path=settings)["route"]

    assert consent["schema"] == SCHEMA_V2
    assert consent["task"] == "scene_segmentation"
    assert consent["model_type"] == "media_service"
    assert consent["authorized_destinations"] == [
        CONTROL_PLANE_BASE_URL,
    ]
    assert consent["upload_manifest"]["files"] == consent["artifacts"]
    assert consent["operator_confirmation"]["exact_manifest_sha256"]
    assert consent["route"]["route_revision"] == route["route_revision"]

    monkeypatch.delenv("MEDIAKIT_API_KEY", raising=False)
    missing_credential = media_connector_consent_status(
        consent["consent_path"],
        expected_route_revision=route["route_revision"],
        settings_path=settings,
        policy=policy,
    )
    assert missing_credential["valid"] is True
    assert missing_credential["ready_for_execution"] is False
    assert any(row["key"] == "mediakit_credential_missing" for row in missing_credential["blockers"])

    monkeypatch.setenv("MEDIAKIT_API_KEY", "test-only-not-persisted")
    monkeypatch.setattr(
        "video_knowledge_pipeline.media_connector_consent.mediakit_cli_status",
        lambda: {"available": True, "command": "fixture-mediakit-cli", "install_command": "fixture"},
    )
    ready = media_connector_consent_status(
        consent["consent_path"],
        expected_route_revision=route["route_revision"],
        settings_path=settings,
        policy=policy,
    )
    assert ready["status"] == "ready_for_execution"
    assert ready["ready_for_execution"] is True
    assert ready["credential"]["value_exposed"] is False
    assert "test-only-not-persisted" not in json.dumps(ready)
    assert ready["network_calls_made"] == 0
    assert ready["operator_boundary"]["execute_tool_available"] is True


def test_legacy_destination_settings_do_not_change_official_cli_consent_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    policy = _policy(tmp_path, "other-upload.example")
    consent = _consent(tmp_path, settings_path=settings, policy=policy)
    original_revision = consent["route"]["route_revision"]
    save_media_route_settings(
        upload_destinations=["https://other-upload.example"],
        settings_path=settings,
        max_poll_attempts=3,
        poll_interval_seconds=0,
        timeout_seconds=30,
    )
    monkeypatch.setenv("MEDIAKIT_API_KEY", "test-only")
    monkeypatch.setattr(
        "video_knowledge_pipeline.media_connector_consent.mediakit_cli_status",
        lambda: {"available": True, "command": "fixture-mediakit-cli", "install_command": "fixture"},
    )

    result = media_connector_consent_status(
        consent["consent_path"],
        expected_route_revision=original_revision,
        settings_path=settings,
        policy=policy,
    )

    assert result["valid"] is True
    assert result["ready_for_execution"] is True
    assert result["network_calls_made"] == 0


def test_authorized_destination_tamper_is_detected_even_with_rehashed_payload(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    policy = _policy(tmp_path, "evil.example")
    consent = _consent(tmp_path, settings_path=settings, policy=policy)
    path = Path(str(consent["consent_path"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["authorized_destinations"] = [
        CONTROL_PLANE_BASE_URL,
        "https://evil.example",
    ]
    payload["consent_sha256"] = _payload_sha256(payload)
    write_json(path, payload)
    route = build_media_route_snapshot("scene_segmentation", settings_path=settings)["route"]

    status = validate_model_connector_consent(
        path,
        route_snapshot=route,
        expected_route_revision=route["route_revision"],
        expected_task="scene_segmentation",
    )

    assert status["valid"] is False
    assert any(
        row["key"] == "consent_authorized_destinations_mismatch"
        for row in status["blockers"]
    )


def test_media_reservation_is_atomic_and_model_execute_front_door_cannot_consume_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    policy = _policy(tmp_path)
    consent = _consent(
        tmp_path,
        settings_path=settings,
        policy=policy,
        max_calls=1,
    )
    route_revision = consent["route"]["route_revision"]

    model_result = execute_consented_model_task(
        consent["consent_path"],
        expected_route_revision=route_revision,
        write=False,
    )
    assert model_result["status"] == "mediakit_cli_unavailable"
    untouched = json.loads(Path(str(consent["consent_path"])).read_text(encoding="utf-8"))
    assert untouched["usage"]["calls_attempted"] == 0

    monkeypatch.setenv("MEDIAKIT_API_KEY", "test-only")
    first = reserve_media_connector_attempt(
        consent["consent_path"],
        expected_route_revision=route_revision,
        settings_path=settings,
        policy=policy,
    )
    second = reserve_media_connector_attempt(
        consent["consent_path"],
        expected_route_revision=route_revision,
        settings_path=settings,
        policy=policy,
    )
    assert first["reserved"] is True
    assert second["reserved"] is False
    assert any(row["key"] == "consent_call_limit_exceeded" for row in second["blockers"])


def test_media_control_surfaces_are_read_only_and_cli_registered() -> None:
    parser = build_parser()
    route_args = parser.parse_args(["media-route-status", "--task", "scene_segmentation"])
    preflight_args = parser.parse_args(
        [
            "media-connector-preflight",
            "consent.json",
            "--route-revision",
            "revision",
        ]
    )
    source = Path(
        "src/video_knowledge_pipeline/trusted_model_connector_remote_mcp.py"
    ).read_text(encoding="utf-8")

    assert route_args.command == "media-route-status"
    assert preflight_args.command == "media-connector-preflight"
    assert "def media_connector_preflight_tool(" in source
    assert "execute_consented_media" not in source
