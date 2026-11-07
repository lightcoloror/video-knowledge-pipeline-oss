from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from video_knowledge_pipeline.cli import build_parser
from video_knowledge_pipeline.media_async_client import execute_loopback_media_task
from video_knowledge_pipeline.media_capability_registry import (
    UPSTREAM_COMMIT,
    media_capability_registry_status,
)
from video_knowledge_pipeline.media_task_protocol import build_media_task_plan
from video_knowledge_pipeline.model_api_settings import public_model_api_settings_status
from video_knowledge_pipeline.model_api_settings_http import _render_html
from video_knowledge_pipeline.model_task_gateway import model_task_api_call
from video_knowledge_pipeline.trusted_model_connector import trusted_model_connector_capabilities


class _MediaHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []
    poll_payloads: list[dict[str, Any]] = []
    submit_status_code = 200

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        type(self).requests.append(
            {
                "method": "POST",
                "path": self.path,
                "protocol": self.headers.get("X-VKP-Media-Protocol"),
                "body": body,
            }
        )
        if type(self).submit_status_code != 200:
            self._send(
                {"error": {"message": "provider unavailable"}},
                status=type(self).submit_status_code,
            )
            return
        self._send(
            {
                "success": True,
                "task_id": "task-1",
                "request_id": "request-1",
                "status": "queued",
            }
        )

    def do_GET(self) -> None:  # noqa: N802
        type(self).requests.append(
            {
                "method": "GET",
                "path": self.path,
                "protocol": self.headers.get("X-VKP-Media-Protocol"),
                "body": b"",
            }
        )
        payload = (
            type(self).poll_payloads.pop(0)
            if type(self).poll_payloads
            else {"task_id": "task-1", "status": "running"}
        )
        self._send(payload)

    def _send(self, payload: dict[str, Any], *, status: int = 200) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


def _server(
    *,
    poll_payloads: list[dict[str, Any]] | None = None,
    submit_status_code: int = 200,
) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    _MediaHandler.requests = []
    _MediaHandler.poll_payloads = list(poll_payloads or [])
    _MediaHandler.submit_status_code = submit_status_code
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MediaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}"


def _plan(tmp_path: Path, *, location: str = "remote") -> dict[str, Any]:
    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"fake-mp4-content")
    return build_media_task_plan(
        "scene_segmentation",
        execution_location=location,
        route_id="media-remote-approved" if location == "remote" else "",
        route_revision="sha256:test-revision" if location == "remote" else "",
        artifact_paths=[media],
        consent_id="test-consent" if location == "remote" else "",
        parameters={"segment_threshold": 10, "min_duration": 3, "max_duration": 30},
        allowed_roots=[tmp_path],
    )


def test_media_registry_has_fixed_execution_capabilities() -> None:
    result = media_capability_registry_status()

    assert result["status"] == "execution_capable"
    assert result["capability_count"] == 5
    assert result["source"]["commit"] == UPSTREAM_COMMIT
    assert result["tasks"] == [
        "scene_segmentation",
        "storyline",
        "highlight_detection",
        "video_ocr",
        "video_asr",
    ]
    assert {row["submit"]["path"] for row in result["capabilities"]} == {
        "/api/v1/tools/segment-scenes",
        "/api/v1/tools/analyze-video-storyline",
        "/api/v1/tools/analyze-video-highlights",
        "/api/v1/tools/video-ocr",
        "/api/v1/tools/asr-subtitles",
    }
    assert all(row["candidate_only"] is True for row in result["capabilities"])
    assert all(row["authorization_status"] == "not_configured" for row in result["capabilities"])
    assert result["operator_boundary"]["real_remote_execution_available"] is True


def test_media_plan_locks_local_artifact_hash_and_rejects_provider_inputs(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    assert plan["schema_version"] == "mediakit_async_v1"
    assert plan["artifact_hashes"][0] == plan["artifacts"][0]["sha256"]
    assert plan["operator_boundary"]["contains_provider_urls"] is False
    assert plan["provider_artifact_refs_present"] is False

    media = tmp_path / "another.mp4"
    media.write_bytes(b"content")
    with pytest.raises(ValueError, match="unsupported media parameters"):
        build_media_task_plan(
            "scene_segmentation",
            execution_location="remote",
            route_id="route",
            route_revision="revision",
            artifact_paths=[media],
            parameters={"video_url": "https://example.invalid/video.mp4"},
            allowed_roots=[tmp_path],
        )
    with pytest.raises(ValueError, match="hash mismatch"):
        build_media_task_plan(
            "scene_segmentation",
            execution_location="remote",
            route_id="route",
            route_revision="revision",
            artifact_paths=[media],
            artifact_hashes=["0" * 64],
            allowed_roots=[tmp_path],
        )


def test_media_plan_rejects_non_media_and_outside_root(tmp_path: Path) -> None:
    document = tmp_path / "notes.txt"
    document.write_text("not media", encoding="utf-8")
    with pytest.raises(ValueError, match="audio or video"):
        build_media_task_plan(
            "video_ocr",
            execution_location="remote",
            route_id="route",
            route_revision="revision",
            artifact_paths=[document],
            allowed_roots=[tmp_path],
        )
    outside = tmp_path.parent / "outside.mp4"
    outside.write_bytes(b"outside")
    try:
        with pytest.raises(ValueError, match="outside allowed roots"):
            build_media_task_plan(
                "scene_segmentation",
                execution_location="remote",
                route_id="route",
                route_revision="revision",
                artifact_paths=[outside],
                allowed_roots=[tmp_path],
            )
    finally:
        outside.unlink(missing_ok=True)


def test_fake_loopback_submit_poll_success_is_test_only_and_secretless(tmp_path: Path) -> None:
    server, thread, base_url = _server(
        poll_payloads=[
            {"task_id": "task-1", "request_id": "request-1", "status": "running"},
            {
                "task_id": "task-1",
                "request_id": "request-1",
                "status": "completed",
                "result": {
                    "segments": [{"start": 0, "end": 3, "score": 0.9}],
                    "preview_url": "https://signed.invalid/private?token=secret",
                    "access_token": "secret",
                },
            },
        ]
    )
    plan = _plan(tmp_path)
    try:
        result = execute_loopback_media_task(
            plan,
            loopback_base_url=base_url,
            max_poll_attempts=3,
            poll_interval_seconds=0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result["ok"] is True
    assert result["status"] == "succeeded"
    assert result["provider_status"] == "completed"
    assert result["content"]["segments"][0]["end"] == 3
    assert "preview_url" not in result["content"]
    assert "access_token" not in result["content"]
    assert result["evidence"][-1]["candidate_only"] is True
    assert result["writeback"] == {
        "timeline_updated": False,
        "bundle_updated": False,
        "smart_summary_updated": False,
    }
    assert result["network_audit"] == {
        "transport": "fake_loopback",
        "requests_made": 3,
        "remote_requests_made": False,
        "fallback_attempted": False,
    }
    assert [row["path"] for row in _MediaHandler.requests] == [
        "/api/v1/tools/segment-scenes",
        "/api/v1/tasks/task-1",
        "/api/v1/tasks/task-1",
    ]
    submitted = json.loads(_MediaHandler.requests[0]["body"])
    assert submitted["artifact_manifest"][0]["sha256"] == plan["artifact_hashes"][0]
    assert plan["artifact_paths"][0] not in _MediaHandler.requests[0]["body"].decode("utf-8")
    assert all(row["protocol"] == "mediakit_async_v1" for row in _MediaHandler.requests)


@pytest.mark.parametrize(
    ("provider_status", "expected_status"),
    [("failed", "failed"), ("canceled", "cancelled"), ("cancelled", "cancelled")],
)
def test_fake_loopback_normalises_failure_and_cancelled_terminal_states(
    tmp_path: Path,
    provider_status: str,
    expected_status: str,
) -> None:
    server, thread, base_url = _server(
        poll_payloads=[
            {
                "task_id": "task-1",
                "status": provider_status,
                "error": {"message": "terminal"},
                "result": {"segments": [{"start": 1, "end": 2}]},
            }
        ]
    )
    try:
        result = execute_loopback_media_task(
            _plan(tmp_path),
            loopback_base_url=base_url,
            max_poll_attempts=1,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result["ok"] is False
    assert result["status"] == expected_status
    assert result["terminal"] is True
    assert result["content"]["segments"]
    assert result["evidence"][-1]["partial"] is True
    assert result["error"]["code"] in {"provider_terminal_failure", "provider_cancelled"}


@pytest.mark.parametrize("status_code", [429, 503])
def test_fake_loopback_http_failures_are_bounded_and_retryable(
    tmp_path: Path,
    status_code: int,
) -> None:
    server, thread, base_url = _server(submit_status_code=status_code)
    try:
        result = execute_loopback_media_task(_plan(tmp_path), loopback_base_url=base_url)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result["status"] == "failed"
    assert result["error"]["code"] == f"provider_http_{status_code}"
    assert result["error"]["retryable"] is True
    assert result["network_audit"]["requests_made"] == 1


def test_fake_loopback_poll_attempt_exhaustion_returns_timeout(tmp_path: Path) -> None:
    server, thread, base_url = _server(
        poll_payloads=[
            {"task_id": "task-1", "status": "running"},
            {"task_id": "task-1", "status": "running", "result": {"segments": [{"start": 0}]}}
        ]
    )
    try:
        result = execute_loopback_media_task(
            _plan(tmp_path),
            loopback_base_url=base_url,
            max_poll_attempts=2,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result["status"] == "timeout"
    assert result["error"]["code"] == "media_poll_attempts_exhausted"
    assert result["content"]["segments"]
    assert result["network_audit"]["requests_made"] == 3


def test_non_loopback_and_local_only_routes_block_before_socket(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def forbidden_urlopen(*args: object, **kwargs: object) -> None:
        calls.append("called")
        raise AssertionError("socket must not be called")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden_urlopen)
    remote = execute_loopback_media_task(
        _plan(tmp_path),
        loopback_base_url="https://amk.cn-beijing.volces.com",
    )
    local = execute_loopback_media_task(
        _plan(tmp_path, location="local"),
        loopback_base_url="http://127.0.0.1:65535",
    )

    assert remote["error"]["code"] == "loopback_test_transport_requires_loopback"
    assert local["error"]["code"] == "local_media_capability_unavailable"
    assert remote["network_audit"]["requests_made"] == 0
    assert local["network_audit"]["requests_made"] == 0
    assert calls == []


def test_tampered_provider_path_blocks_before_socket(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def forbidden_urlopen(*args: object, **kwargs: object) -> None:
        calls.append("called")
        raise AssertionError("socket must not be called")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden_urlopen)
    plan = _plan(tmp_path)
    plan["submit"] = {"method": "POST", "path": "/api/v1/tools/unreviewed-task"}

    result = execute_loopback_media_task(
        plan,
        loopback_base_url="http://127.0.0.1:65535",
    )

    assert result["error"]["code"] == "invalid_media_route_snapshot"
    assert result["network_audit"]["requests_made"] == 0
    assert calls == []


def test_media_catalog_is_discoverable_but_not_authorized_in_ui_broker_and_cli(
    tmp_path: Path,
) -> None:
    status = public_model_api_settings_status(
        tmp_path / "settings.json",
        tmp_path / "secrets.json",
    )
    broker = trusted_model_connector_capabilities()
    html = _render_html("test-token")
    cli = build_parser().parse_args(["media-capability-status"])

    assert status["media_capability_catalog"]["capability_count"] == 5
    assert broker["media_capability_catalog"]["routing"]["remote_destination_allowlisted"] is True
    assert "MediaKit 远程媒体能力" in html
    assert "每次执行仍需绑定文件哈希" in html
    assert cli.command == "media-capability-status"


def test_native_whole_video_model_route_is_provider_capability(monkeypatch) -> None:
    from video_knowledge_pipeline import online_model_gateway

    monkeypatch.setattr(online_model_gateway, "call_gemini_video", lambda **_: {"ok": True, "error": "", "content": "ok"})
    result = model_task_api_call(
        "native_video_segment",
        provider_config={"provider": "gemini", "model": "gemini-3.6-flash"},
        video_path="fixture.mp4",
        execute=True,
    )

    assert result["ok"] is True
    assert result["request_plan"]["interface"] == "gemini_files_api"
