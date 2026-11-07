from __future__ import annotations

import ipaddress
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from .media_capability_registry import media_capability
from .media_capability_registry import PROTOCOL
from .media_task_protocol import (
    media_status_is_terminal,
    normalise_media_task_result,
)


@dataclass
class MediaHTTPError(Exception):
    status_code: int
    payload: dict[str, Any] | None = None
    body: str = ""

    def __str__(self) -> str:
        return f"HTTP {self.status_code}"


def execute_loopback_media_task(
    plan: dict[str, Any],
    *,
    loopback_base_url: str,
    max_poll_attempts: int = 6,
    poll_interval_seconds: float = 0,
    timeout_seconds: float = 30,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    started = time.perf_counter()
    if str(plan.get("schema_version") or "") != PROTOCOL:
        return _failure(
            plan,
            code="invalid_media_plan",
            message=f"media plan schema_version must be {PROTOCOL}",
            started=started,
        )
    if str(plan.get("execution_location") or "") == "local":
        return _failure(
            plan,
            code="local_media_capability_unavailable",
            message="MediaKit capabilities are remote-only; VKP will not fallback to a remote pool",
            started=started,
        )
    if str(plan.get("execution_location") or "") != "remote":
        return _failure(
            plan,
            code="invalid_execution_location",
            message="media task execution_location must be local or remote",
            started=started,
        )
    if not _is_loopback_base_url(loopback_base_url):
        return _failure(
            plan,
            code="loopback_test_transport_requires_loopback",
            message="execute_loopback_media_task is a test fixture; production MediaKit execution uses the consented official CLI adapter",
            started=started,
        )
    try:
        attempts = int(max_poll_attempts)
        interval = float(poll_interval_seconds)
        timeout = float(timeout_seconds)
    except (TypeError, ValueError) as exc:
        return _failure(plan, code="invalid_poll_policy", message=str(exc), started=started)
    if attempts < 0 or attempts > 100:
        return _failure(
            plan,
            code="invalid_poll_policy",
            message="max_poll_attempts must be between 0 and 100",
            started=started,
        )
    if interval < 0 or interval > 60 or timeout <= 0 or timeout > 3600:
        return _failure(
            plan,
            code="invalid_poll_policy",
            message="poll interval and timeout exceed bounded policy",
            started=started,
        )
    submit = plan.get("submit") if isinstance(plan.get("submit"), dict) else {}
    poll = plan.get("poll") if isinstance(plan.get("poll"), dict) else {}
    try:
        capability = media_capability(str(plan.get("task") or ""))
    except ValueError as exc:
        return _failure(plan, code="invalid_media_route_snapshot", message=str(exc), started=started)
    expected_submit = capability["submit"]
    expected_poll = capability["poll"]
    if (
        str(plan.get("provider_task") or "") != str(capability["provider_task"])
        or submit != expected_submit
        or poll != expected_poll
    ):
        return _failure(
            plan,
            code="invalid_media_route_snapshot",
            message="media plan provider task, submit path, or poll path differs from the fixed registry",
            started=started,
        )
    submit_path = str(submit["path"])
    poll_template = str(poll["path_template"])
    deadline = time.monotonic() + timeout
    requests_made = 0
    try:
        payload = _request_json(
            "POST",
            _url(loopback_base_url, submit_path),
            _submit_payload(plan),
            timeout=_remaining(deadline),
        )
        requests_made += 1
    except MediaHTTPError as exc:
        requests_made += 1
        return _http_failure(plan, exc, started=started, request_count=requests_made)
    except (TimeoutError, socket.timeout):
        requests_made += 1
        return _failure(
            plan,
            code="media_submit_timeout",
            message="fake loopback media submit timed out",
            forced_status="timeout",
            started=started,
            request_count=requests_made,
        )
    except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
        requests_made += 1
        return _failure(
            plan,
            code="loopback_media_service_unavailable",
            message=str(exc),
            started=started,
            request_count=requests_made,
        )
    if payload.get("success") is False:
        return normalise_media_task_result(
            plan,
            payload,
            forced_status="failed",
            error={"code": "provider_business_failure", "details": payload.get("error")},
            latency_ms=_latency_ms(started),
            request_count=requests_made,
        )
    task_id = str(payload.get("task_id") or "").strip()
    if not task_id:
        return _failure(
            plan,
            code="invalid_provider_response",
            message="media submit response did not include task_id",
            started=started,
            request_count=requests_made,
        )
    submit_result = normalise_media_task_result(
        plan,
        payload,
        latency_ms=_latency_ms(started),
        request_count=requests_made,
    )
    if media_status_is_terminal(str(submit_result["status"])) or attempts == 0:
        return _terminal_error_result(plan, payload, submit_result, started, requests_made)
    last_payload = payload
    for _ in range(attempts):
        if interval:
            sleep(interval)
        try:
            remaining = _remaining(deadline)
            poll_path = poll_template.replace(
                "{task_id}",
                urllib.parse.quote(task_id, safe=""),
            )
            last_payload = _request_json(
                "GET",
                _url(loopback_base_url, poll_path),
                None,
                timeout=remaining,
            )
            requests_made += 1
        except MediaHTTPError as exc:
            requests_made += 1
            return _http_failure(plan, exc, started=started, request_count=requests_made)
        except (TimeoutError, socket.timeout):
            requests_made += 1
            return _failure(
                plan,
                code="media_poll_timeout",
                message="fake loopback media poll timed out",
                forced_status="timeout",
                provider_payload=last_payload,
                started=started,
                request_count=requests_made,
            )
        except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
            requests_made += 1
            return _failure(
                plan,
                code="loopback_media_service_unavailable",
                message=str(exc),
                provider_payload=last_payload,
                started=started,
                request_count=requests_made,
            )
        result = normalise_media_task_result(
            plan,
            last_payload,
            latency_ms=_latency_ms(started),
            request_count=requests_made,
        )
        if media_status_is_terminal(str(result["status"])):
            return _terminal_error_result(plan, last_payload, result, started, requests_made)
    return normalise_media_task_result(
        plan,
        last_payload,
        forced_status="timeout",
        error={
            "code": "media_poll_attempts_exhausted",
            "message": "media task did not reach a terminal state within the bounded poll policy",
        },
        latency_ms=_latency_ms(started),
        request_count=requests_made,
    )


def _terminal_error_result(
    plan: dict[str, Any],
    payload: dict[str, Any],
    result: dict[str, Any],
    started: float,
    request_count: int,
) -> dict[str, Any]:
    status = str(result.get("status") or "")
    if status not in {"failed", "cancelled"}:
        return result
    code = "provider_terminal_failure" if status == "failed" else "provider_cancelled"
    return normalise_media_task_result(
        plan,
        payload,
        forced_status=status,
        error={
            "code": code,
            "message": f"media provider reached terminal status {status}",
            "details": payload.get("error"),
        },
        latency_ms=_latency_ms(started),
        request_count=request_count,
    )


def _submit_payload(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL,
        "task": str(plan.get("provider_task") or ""),
        "artifact_manifest": [
            {
                "sha256": str(row.get("sha256") or ""),
                "bytes": int(row.get("bytes") or 0),
                "data_type": str(row.get("data_type") or ""),
            }
            for row in plan.get("artifacts") or []
            if isinstance(row, dict)
        ],
        "parameters": dict(plan.get("parameters") or {}),
        "candidate_only": True,
    }


def _request_json(method: str, url: str, payload: dict[str, Any] | None, *, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-VKP-Media-Protocol": PROTOCOL,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
        parsed = _parse_json_object(body, allow_empty=True)
        raise MediaHTTPError(exc.code, parsed, body.decode("utf-8", errors="replace")) from exc
    return _parse_json_object(body)


def _parse_json_object(body: bytes, *, allow_empty: bool = False) -> dict[str, Any]:
    if not body and allow_empty:
        return {}
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("media provider response must be a JSON object")
    return payload


def _http_failure(
    plan: dict[str, Any],
    exc: MediaHTTPError,
    *,
    started: float,
    request_count: int,
) -> dict[str, Any]:
    retryable = exc.status_code == 429 or exc.status_code >= 500
    return _failure(
        plan,
        code=f"provider_http_{exc.status_code}",
        message=str(exc),
        details=exc.payload or exc.body,
        retryable=retryable,
        started=started,
        request_count=request_count,
    )


def _failure(
    plan: dict[str, Any],
    *,
    code: str,
    message: str,
    started: float,
    forced_status: str = "failed",
    provider_payload: dict[str, Any] | None = None,
    details: Any = None,
    retryable: bool = False,
    request_count: int = 0,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "retryable": bool(retryable),
    }
    if details not in (None, "", {}):
        error["details"] = details
    return normalise_media_task_result(
        plan,
        provider_payload or {},
        forced_status=forced_status,
        error=error,
        latency_ms=_latency_ms(started),
        request_count=request_count,
    )


def _is_loopback_base_url(value: str) -> bool:
    parsed = urllib.parse.urlsplit(str(value or "").strip())
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
        return False
    host = parsed.hostname.lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _url(base_url: str, path: str) -> str:
    return str(base_url).rstrip("/") + "/" + str(path).lstrip("/")


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("media task deadline exceeded")
    return remaining


def _latency_ms(started: float) -> int:
    return max(0, int(round((time.perf_counter() - started) * 1000)))
