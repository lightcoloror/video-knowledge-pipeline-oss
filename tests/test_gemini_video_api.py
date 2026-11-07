from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline import gemini_video_api


def test_gemini_video_files_api_upload_generate_and_delete(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"video-bytes")
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(gemini_video_api, "_begin_resumable_upload", lambda **_: "https://upload.example/session")
    monkeypatch.setattr(
        gemini_video_api,
        "_upload_file",
        lambda *_args, **_kwargs: {"file": {"name": "files/abc", "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc", "state": "ACTIVE"}},
    )
    monkeypatch.setattr(
        gemini_video_api,
        "_wait_for_active_file",
        lambda **_: ({"name": "files/abc", "state": "ACTIVE"}, 0),
    )

    def fake_request(method: str, url: str, payload, **_kwargs):
        calls.append((method, url))
        if method == "POST":
            return {"candidates": [{"content": {"parts": [{"text": "video summary"}]}}]}
        return {}

    monkeypatch.setattr(gemini_video_api, "_request_json", fake_request)
    result = gemini_video_api.call_gemini_video(
        provider_config={"provider": "gemini", "api_key": "test-key", "model": "gemini-3.6-flash"},
        prompt="summarise",
        video_path=str(video),
    )

    assert result["ok"] is True
    assert result["content"] == "video summary"
    assert result["network_audit"]["source_artifact_uploaded"] is True
    assert result["network_audit"]["provider_file_deleted"] is True
    assert calls[-1] == ("DELETE", "https://generativelanguage.googleapis.com/v1beta/files/abc")


def test_gemini_video_does_not_fallback_to_another_provider(tmp_path: Path) -> None:
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"video-bytes")
    result = gemini_video_api.call_gemini_video(
        provider_config={"provider": "volcengine_coding_plan", "api_key": "test-key"},
        prompt="summarise",
        video_path=str(video),
    )
    assert result == {"ok": False, "error": "provider_video_capability_unavailable", "content": ""}


def test_gemini_resumable_upload_url_is_destination_locked() -> None:
    assert gemini_video_api._allowed_upload_url(
        "https://generativelanguage.googleapis.com/upload/v1beta/files?upload_id=fixture"
    ) is True
    assert gemini_video_api._allowed_upload_url(
        "https://unapproved.example/upload/session"
    ) is False


def test_gemini_video_delete_failure_is_not_reported_as_success(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"video-bytes")
    monkeypatch.setattr(gemini_video_api, "_begin_resumable_upload", lambda **_: "https://generativelanguage.googleapis.com/upload/session")
    monkeypatch.setattr(
        gemini_video_api,
        "_upload_file",
        lambda *_args, **_kwargs: {
            "file": {
                "name": "files/abc",
                "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc",
                "state": "ACTIVE",
            }
        },
    )
    monkeypatch.setattr(
        gemini_video_api,
        "_wait_for_active_file",
        lambda **_: ({"name": "files/abc", "state": "ACTIVE"}, 0),
    )

    def fake_request(method: str, _url: str, _payload, **_kwargs):
        if method == "DELETE":
            raise OSError("delete failed")
        return {"candidates": [{"content": {"parts": [{"text": "video summary"}]}}]}

    monkeypatch.setattr(gemini_video_api, "_request_json", fake_request)
    result = gemini_video_api.call_gemini_video(
        provider_config={"provider": "gemini", "api_key": "test-key", "model": "gemini-3.6-flash"},
        prompt="summarise",
        video_path=str(video),
    )

    assert result["ok"] is False
    assert result["error"] == "provider_file_delete_failed"
    assert result["network_audit"]["provider_file_delete_failed"] is True
