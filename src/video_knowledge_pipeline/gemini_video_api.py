from __future__ import annotations

import json
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .model_defaults import GEMINI_DEFAULT_MODEL
from .vision_api import provider_requires_api_key


FILE_POLL_ATTEMPTS = 60
FILE_POLL_INTERVAL_SECONDS = 2.0
ALLOWED_UPLOAD_HOSTS = frozenset({"generativelanguage.googleapis.com"})


def call_gemini_video(
    *,
    provider_config: dict[str, Any],
    prompt: str,
    video_path: str,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Use Gemini Files API for one already-authorised local video.

    The caller owns consent and hash verification. This function owns only the
    provider transaction: upload the file to Gemini, generate content, then delete
    Gemini's temporary file without trying a different provider on failure.
    """
    cfg = dict(provider_config or {})
    if str(cfg.get("provider") or "").strip().lower() != "gemini":
        return _failure("provider_video_capability_unavailable")
    if provider_requires_api_key(cfg) and not str(cfg.get("api_key") or "").strip():
        return _failure("missing_api_key")
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        return _failure("video_not_found")
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    if not mime.startswith("video/"):
        return _failure("unsupported_video_mime")
    base_url = _gemini_base_url(str(cfg.get("base_url") or ""))
    if not base_url:
        return _failure("unsupported_gemini_endpoint")
    api_key = str(cfg.get("api_key") or "")
    timeout = max(1, min(3600, int(cfg.get("timeout_seconds") or 120)))
    file_name = ""
    outcome: dict[str, Any] | None = None
    audit = {
        "provider": "gemini",
        "transport": "gemini_files_api",
        "source_artifact_uploaded": False,
        "provider_file_deleted": False,
        "requests_made": 0,
        "fallback_attempted": False,
    }
    try:
        upload_url = _begin_resumable_upload(
            base_url=base_url,
            api_key=api_key,
            path=path,
            mime=mime,
            timeout=timeout,
        )
        audit["requests_made"] += 1
        uploaded = _upload_file(upload_url, path=path, mime=mime, timeout=timeout)
        audit["requests_made"] += 1
        file_data = uploaded.get("file") if isinstance(uploaded.get("file"), dict) else uploaded
        file_name = str(file_data.get("name") or "")
        file_uri = str(file_data.get("uri") or "")
        if not file_name or not file_uri:
            return _failure("gemini_files_upload_invalid_response", raw_response=uploaded, audit=audit)
        audit["source_artifact_uploaded"] = True
        file_data, poll_count = _wait_for_active_file(
            base_url=base_url,
            api_key=api_key,
            file_name=file_name,
            initial=file_data,
            timeout=timeout,
            sleep=sleep,
        )
        audit["requests_made"] += poll_count
        state = str(file_data.get("state") or "").upper()
        if state != "ACTIVE":
            return _failure("gemini_file_processing_failed", raw_response=file_data, audit=audit)
        model = str(cfg.get("model") or GEMINI_DEFAULT_MODEL)
        response = _request_json(
            "POST",
            f"{base_url}/models/{urllib.parse.quote(model, safe='')}:generateContent",
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": str(prompt or "")},
                            {"file_data": {"mime_type": mime, "file_uri": file_uri}},
                        ],
                    }
                ]
            },
            headers={"X-goog-api-key": api_key, "Content-Type": "application/json"},
            timeout=timeout,
        )
        audit["requests_made"] += 1
        content = _gemini_text(response)
        if not content:
            return _failure("empty_content", raw_response=response, audit=audit)
        outcome = {
            "ok": True,
            "error": "",
            "content": content,
            "raw_response": response,
            "network_audit": audit,
        }
        return outcome
    except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        return _failure(f"gemini_video_request_failed:{type(exc).__name__}", audit=audit)
    finally:
        if file_name:
            try:
                _request_json(
                    "DELETE",
                    f"{base_url}/{file_name}",
                    None,
                    headers={"X-goog-api-key": api_key},
                    timeout=timeout,
                )
                audit["requests_made"] += 1
                audit["provider_file_deleted"] = True
            except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
                audit["provider_file_delete_failed"] = True
                if outcome is not None and outcome.get("ok"):
                    outcome.update(
                        {
                            "ok": False,
                            "error": "provider_file_delete_failed",
                            "content": "",
                        }
                    )


def _gemini_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(str(value or "https://generativelanguage.googleapis.com/v1beta").rstrip("/"))
    if parsed.scheme != "https" or parsed.hostname != "generativelanguage.googleapis.com":
        return ""
    path = parsed.path.rstrip("/") or "/v1beta"
    if not path.startswith("/v1"):
        return ""
    return f"https://generativelanguage.googleapis.com{path}"


def _begin_resumable_upload(*, base_url: str, api_key: str, path: Path, mime: str, timeout: int) -> str:
    upload_url = base_url.replace("/v1", "/upload/v1", 1) + "/files"
    request = urllib.request.Request(
        upload_url,
        data=json.dumps({"file": {"display_name": path.name}}, ensure_ascii=False).encode("utf-8"),
        headers={
            "X-goog-api-key": api_key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(path.stat().st_size),
            "X-Goog-Upload-Header-Content-Type": mime,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = str(response.headers.get("X-Goog-Upload-URL") or "").strip()
    if not _allowed_upload_url(result):
        raise ValueError("Gemini did not return an HTTPS resumable upload URL")
    return result


def _allowed_upload_url(value: str) -> bool:
    parsed = urllib.parse.urlsplit(str(value or ""))
    return parsed.scheme == "https" and str(parsed.hostname or "").lower() in ALLOWED_UPLOAD_HOSTS


def _upload_file(upload_url: str, *, path: Path, mime: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        upload_url,
        data=path.read_bytes(),
        headers={
            "Content-Length": str(path.stat().st_size),
            "Content-Type": mime,
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip() else {}


def _wait_for_active_file(
    *,
    base_url: str,
    api_key: str,
    file_name: str,
    initial: dict[str, Any],
    timeout: int,
    sleep: Callable[[float], None],
) -> tuple[dict[str, Any], int]:
    current = dict(initial)
    polls = 0
    for _ in range(FILE_POLL_ATTEMPTS):
        state = str(current.get("state") or "ACTIVE").upper()
        if state in {"ACTIVE", "FAILED"}:
            return current, polls
        sleep(FILE_POLL_INTERVAL_SECONDS)
        current = _request_json(
            "GET",
            f"{base_url}/{file_name}",
            None,
            headers={"X-goog-api-key": api_key},
            timeout=timeout,
        )
        polls += 1
    return current, polls


def _request_json(method: str, url: str, payload: dict[str, Any] | None, *, headers: dict[str, str], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip() else {}


def _gemini_text(payload: dict[str, Any]) -> str:
    try:
        parts = payload["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return ""
    return "\n".join(str(part.get("text") or "") for part in parts if isinstance(part, dict)).strip()


def _failure(error: str, *, raw_response: Any = None, audit: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": False, "error": error, "content": ""}
    if raw_response is not None:
        result["raw_response"] = raw_response
    if audit is not None:
        result["network_audit"] = audit
    return result
