from __future__ import annotations

import base64
from contextlib import contextmanager
from contextvars import ContextVar
import json
import mimetypes
import re
import socket
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .model_defaults import gemini_omits_legacy_sampling_parameters

from .artifact_validation import (
    DEFAULT_ALLOWED_ROOTS_ENV,
    artifact_evidence,
    normalise_allowed_roots,
    validated_local_file,
)
from .model_api_settings import (
    _read_secret,
    default_secrets_path,
    default_settings_path,
    resolve_model_api_route,
)
from .model_gateway import MASTER_KEY_ID, _gateway_deployment_id, load_model_gateway_config



SCHEMA = "video_knowledge_pipeline.model_runtime_result.v1"
TEXT_TASKS = frozenset({"text_llm", "summary_rewrite", "transcript_correction"})
VISION_TASKS = frozenset({"document_visual", "semantic_frame", "temporal_sequence", "video_segment"})
RUNTIME_TASKS = TEXT_TASKS | VISION_TASKS | {"asr", "ocr"}
ALLOWED_ROOTS_ENV = DEFAULT_ALLOWED_ROOTS_ENV
ASR_PROMPT_MAX_CHARS = 4000
_REMOTE_EXECUTION_GRANT: ContextVar[dict[str, Any] | None] = ContextVar(
    "vkp_remote_model_execution_grant",
    default=None,
)


@contextmanager
def authorise_consented_remote_runtime(
    *,
    consent_id: str,
    route_revision: str,
    max_calls: int,
):
    identifier = str(consent_id or "").strip()
    revision = str(route_revision or "").strip()
    calls = int(max_calls)
    if not identifier or not revision or calls < 1:
        raise ValueError("remote runtime grant requires consent_id, route_revision, and positive max_calls")
    token = _REMOTE_EXECUTION_GRANT.set(
        {"consent_id": identifier, "route_revision": revision, "remaining_calls": calls}
    )
    try:
        yield
    finally:
        _REMOTE_EXECUTION_GRANT.reset(token)




# Compatibility alias for the original Broker-only internal import.
_authorise_consented_remote_runtime = authorise_consented_remote_runtime
def _consume_remote_execution_grant(*, consent_id: str, route_revision: str) -> str:
    grant = _REMOTE_EXECUTION_GRANT.get()
    if not isinstance(grant, dict):
        return "remote proxy execution is allowed only inside a validated Broker consent reservation"
    if str(grant.get("consent_id") or "") != str(consent_id or ""):
        return "remote runtime grant consent_id does not match the request"
    if str(grant.get("route_revision") or "") != str(route_revision or ""):
        return "remote runtime grant route_revision does not match the configured route"
    remaining = int(grant.get("remaining_calls") or 0)
    if remaining < 1:
        return "remote runtime grant call allowance is exhausted"
    grant["remaining_calls"] = remaining - 1
    return ""


def consume_consented_remote_runtime_grant(
    *, consent_id: str, route_revision: str
) -> str:
    """Consume one already-reserved Broker call without exposing grant state.

    This public thin wrapper lets audited non-proxy owner adapters participate in
    the same consent reservation contract as the LiteLLM runtime client.  It does
    not create authority and returns the existing fail-closed error text when no
    matching reservation is active.
    """

    return _consume_remote_execution_grant(
        consent_id=consent_id,
        route_revision=route_revision,
    )


def model_runtime_request(
    task: str,
    *,
    execution_location: str = "",
    route_id: str = "",
    route_revision: str = "",
    text: str = "",
    image_paths: list[str | Path] | None = None,
    audio_path: str | Path = "",
    document_path: str | Path = "",
    prompt: str = "",
    messages: list[dict[str, Any]] | None = None,
    temperature: float = 0,
    response_format: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    max_retries: int | None = None,
    consent_id: str = "",
    execute: bool = False,
    settings_path: str | Path | None = None,
    secrets_path: str | Path | None = None,
    gateway_config_path: str | Path | None = None,
    allowed_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> dict[str, Any]:
    task_key = _normalise_task(task)
    started = time.perf_counter()
    try:
        route = resolve_model_api_route(
            task_key,
            execution_location=execution_location,
            settings_path=settings_path,
        )
        _require_route_identity(route, route_id=route_id, route_revision=route_revision)
        runtime_options = _route_runtime_options(route)
        if max_tokens is not None:
            requested_max_tokens = _bounded_runtime_integer("max_tokens", max_tokens)
            locked_max_tokens = runtime_options.get("max_tokens")
            if (
                locked_max_tokens is not None
                and requested_max_tokens != locked_max_tokens
            ):
                raise ValueError("max_tokens does not match the route-locked value")
            runtime_options["max_tokens"] = requested_max_tokens
        _effective_response_format(response_format, runtime_options)
        retry_policy = dict(route.get("retry_policy") or {})
        route_retry_limit = _bounded_retry_count(
            retry_policy.get("max_retries") or 0
        )
        retry_limit = route_retry_limit if max_retries is None else _bounded_retry_count(max_retries)
        if retry_limit > route_retry_limit:
            raise ValueError(
                "max_retries cannot exceed the route-locked retry policy"
            )
        roots = _normalise_allowed_roots(allowed_roots)
        inputs = _validate_inputs(
            task_key,
            text=text,
            image_paths=image_paths or [],
            audio_path=audio_path,
            document_path=document_path,
            allowed_roots=roots,
        )
        _validate_messages(task_key, messages)
        gateway = load_model_gateway_config(gateway_config_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return _failure(
            task_key,
            status="invalid_runtime_request",
            error=str(exc),
            started=started,
            execution_location=str(execution_location or ""),
            route_id=str(route_id or ""),
            route_revision=str(route_revision or ""),
            consent_id=consent_id,
        )

    endpoint = _endpoint(task_key, gateway)
    result = _base_result(route, task_key, endpoint=endpoint, evidence=inputs["evidence"], consent_id=consent_id)
    if not execute:
        result.update(
            {
                "ok": True,
                "status": "planned",
                "latency_ms": _elapsed_ms(started),
                "request": {
                    "endpoint": endpoint,
                    "image_count": len(inputs["images"]),
                    "has_audio": bool(inputs["audio"]),
                    "has_document": bool(inputs["document"]),
                    "text_chars": len(str(text or "")),
                    "prompt_chars": min(len(str(prompt or "")), ASR_PROMPT_MAX_CHARS) if task_key == "asr" else len(str(prompt or "")),
                    "prompt_truncated": task_key == "asr" and len(str(prompt or "")) > ASR_PROMPT_MAX_CHARS,
                    "transient_data_urls": task_key in VISION_TASKS or task_key == "ocr",
                    "runtime_options": {
                        **dict(runtime_options),
                        "max_retries": retry_limit,
                    },
                },
            }
        )
        return result

    if str(route.get("execution_location") or "") == "remote":
        grant_error = _consume_remote_execution_grant(
            consent_id=consent_id,
            route_revision=str(route.get("route_revision") or ""),
        )
        if grant_error:
            result.update(
                {
                    "ok": False,
                    "status": "remote_consent_required",
                    "error": grant_error,
                    "latency_ms": _elapsed_ms(started),
                    "remote_requests_made": False,
                }
            )
            return result

    settings_file = Path(settings_path).expanduser().resolve() if settings_path else default_settings_path()
    secrets_file = (
        Path(secrets_path).expanduser().resolve()
        if secrets_path
        else default_secrets_path(settings_path=settings_file)
    )
    api_key = _gateway_api_key(secrets_file)
    if not api_key:
        result.update(
            {
                "ok": False,
                "status": "gateway_credentials_unavailable",
                "error": "LiteLLM gateway master key is not available in the local secret store",
                "latency_ms": _elapsed_ms(started),
            }
        )
        return result

    try:
        body, content_type = _request_body(
            task_key,
            route=route,
            inputs=inputs,
            prompt=prompt,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
            runtime_options=runtime_options,
        )
        result["network_accounting"] = {
            "scope": "vkp_to_loopback_gateway_payload",
            "gateway_request_bytes": len(body),
            "gateway_response_bytes": 0,
            "source_artifact_bytes": sum(
                int(row.get("bytes") or 0)
                for row in inputs.get("evidence") or []
                if isinstance(row, dict)
            ),
            "provider_wire_bytes_exact": False,
            "provider_wire_note": (
                "LiteLLM may re-serialise the request before provider delivery; "
                "these byte counts are exact only for the VKP-to-loopback-gateway payload."
            ),
        }
        timeout_seconds = int(retry_policy.get("timeout_seconds") or 120)
        request = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": content_type,
                "Accept": "text/event-stream" if runtime_options.get("stream") else "application/json",
                "x-litellm-timeout": str(timeout_seconds),
                "x-litellm-num-retries": str(retry_limit),
            },
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload, response_headers, http_status, response_payload_bytes = _read_gateway_response(
                response,
                streamed=bool(runtime_options.get("stream")),
            )
            result["network_accounting"]["gateway_response_bytes"] = response_payload_bytes
    except urllib.error.HTTPError as exc:
        return _http_failure(result, exc.code, started=started, detail=_http_error_detail(exc))
    except (socket.timeout, TimeoutError) as exc:
        result.update({"ok": False, "status": "gateway_timeout", "error": str(exc), "latency_ms": _elapsed_ms(started)})
        return result
    except urllib.error.URLError as exc:
        unavailable_status = (
            "local_gateway_unavailable"
            if str(result.get("execution_location") or "") == "local"
            else "gateway_unavailable"
        )
        result.update(
            {
                "ok": False,
                "status": unavailable_status,
                "error": str(exc.reason),
                "latency_ms": _elapsed_ms(started),
                "remote_requests_made": False if unavailable_status == "local_gateway_unavailable" else None,
            }
        )
        return result
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result.update({"ok": False, "status": "gateway_response_invalid", "error": str(exc), "latency_ms": _elapsed_ms(started)})
        return result

    deployment = _response_deployment(route, payload, response_headers)
    content, content_sanitization = _content(
        task_key,
        payload,
        runtime_options=runtime_options,
    )
    result.update(
        {
            "ok": 200 <= http_status < 300,
            "status": "completed" if 200 <= http_status < 300 else "gateway_error",
            "deployment": deployment,
            "provider": str(deployment.get("provider") or ""),
            "latency_ms": _elapsed_ms(started),
            "usage": _usage(payload),
            "estimated_cost": _estimated_cost(payload, response_headers),
            "content": content,
            "content_sanitization": content_sanitization,
            "raw_output": payload if task_key == "asr" else {},
            "request_options": {
                **dict(runtime_options),
                "max_retries": retry_limit,
            },
            "response": _response_metadata(
                payload,
                http_status,
                response_headers,
                streamed=bool(runtime_options.get("stream")),
            ),
        }
    )
    return result


def _normalise_task(value: str) -> str:
    task = str(value or "").strip().lower().replace("-", "_")
    aliases = {"text": "text_llm", "summary": "summary_rewrite", "vision": "semantic_frame", "temporal": "temporal_sequence"}
    task = aliases.get(task, task)
    if task not in RUNTIME_TASKS:
        raise ValueError(f"unsupported runtime task: {value!r}")
    return task


def _validate_messages(task: str, messages: list[dict[str, Any]] | None) -> None:
    if messages is None:
        return
    if task not in TEXT_TASKS:
        raise ValueError("structured messages are accepted only for text tasks")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    for row in messages:
        if not isinstance(row, dict) or not str(row.get("role") or "").strip():
            raise ValueError("each message requires a role")
        if not isinstance(row.get("content"), str):
            raise ValueError("runtime text messages must contain text only; image inputs require local image_paths")

def _require_route_identity(route: dict[str, Any], *, route_id: str, route_revision: str) -> None:
    if route_id and str(route.get("route_id") or "") != str(route_id):
        raise ValueError("configured route_id differs from requested route_id")
    if route_revision and str(route.get("route_revision") or "") != str(route_revision):
        raise ValueError("configured route_revision differs from requested route_revision")


def _normalise_allowed_roots(values: list[str | Path] | tuple[str | Path, ...] | None) -> tuple[Path, ...]:
    return normalise_allowed_roots(values, env_var=ALLOWED_ROOTS_ENV)


def _validated_path(value: str | Path, *, label: str, allowed_roots: tuple[Path, ...]) -> Path:
    return validated_local_file(value, label=label, allowed_roots=allowed_roots)


def _validate_inputs(
    task: str,
    *,
    text: str,
    image_paths: list[str | Path],
    audio_path: str | Path,
    document_path: str | Path,
    allowed_roots: tuple[Path, ...],
) -> dict[str, Any]:
    images: list[Path] = []
    audio: Path | None = None
    document: Path | None = None
    if task in VISION_TASKS:
        if not image_paths:
            raise ValueError("image_paths are required for vision tasks")
        for value in image_paths:
            path = _validated_path(value, label="image_path", allowed_roots=allowed_roots)
            mime = mimetypes.guess_type(str(path))[0] or ""
            if not mime.startswith("image/"):
                raise ValueError(f"image_path has unsupported MIME type: {path}")
            images.append(path)
    elif task == "asr":
        audio = _validated_path(audio_path, label="audio_path", allowed_roots=allowed_roots)
    elif task == "ocr":
        candidates = [value for value in image_paths if str(value).strip()]
        if document_path:
            candidates.append(document_path)
        if len(candidates) != 1:
            raise ValueError("OCR requires exactly one image or document per call")
        document = _validated_path(candidates[0], label="document_path", allowed_roots=allowed_roots)
    evidence_paths = images + ([audio] if audio else []) + ([document] if document else [])
    return {
        "text": str(text or ""),
        "images": images,
        "audio": audio,
        "document": document,
        "evidence": [_evidence(path) for path in evidence_paths],
    }


def _evidence(path: Path) -> dict[str, Any]:
    return artifact_evidence(path)


def _endpoint(task: str, gateway: dict[str, Any]) -> str:
    host = str(gateway["host"])
    rendered_host = f"[{host}]" if ":" in host else host
    suffix = "/v1/audio/transcriptions" if task == "asr" else ("/v1/ocr" if task == "ocr" else "/v1/chat/completions")
    return f"http://{rendered_host}:{int(gateway['port'])}{suffix}"


def _request_body(
    task: str,
    *,
    route: dict[str, Any],
    inputs: dict[str, Any],
    prompt: str,
    messages: list[dict[str, Any]] | None,
    temperature: float,
    response_format: dict[str, Any] | None,
    runtime_options: dict[str, Any],
) -> tuple[bytes, str]:
    model = str(route["virtual_model"])
    if task == "asr":
        fields = {
            "model": model,
            "response_format": "verbose_json",
        }
        # Provider-specific transport remains route-locked. Native Groq STT in
        # LiteLLM 1.81.7-1.86.2 rejects this field; the reviewed Groq preset
        # therefore uses LiteLLM's OpenAI-compatible transcription transport.
        if _asr_timestamp_granularities_supported(route):
            fields["timestamp_granularities[]"] = str(
                runtime_options.get("asr_timestamp_granularity") or "segment"
            )
        if prompt:
            fields["prompt"] = str(prompt)[:ASR_PROMPT_MAX_CHARS]
        return _multipart_form_data(fields, file_path=inputs["audio"])
    if task == "ocr":
        path = inputs["document"]
        mime, data_url = _data_url(path)
        key = "image_url" if mime.startswith("image/") else "document_url"
        payload = {"model": model, "document": {"type": key, key: data_url}}
        return json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json"
    if task in VISION_TASKS:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": _combined_prompt(prompt, inputs["text"])}
        ]
        for path in inputs["images"]:
            _mime, data_url = _data_url(path)
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        request_messages = [{"role": "user", "content": content}]
    else:
        request_messages = list(
            messages
            or [{"role": "user", "content": _combined_prompt(prompt, inputs["text"])}]
        )
    payload: dict[str, Any] = {
        "model": model,
        "messages": request_messages,
    }
    if not _route_omits_legacy_sampling_parameters(route):
        payload["temperature"] = float(temperature)
    effective_response_format = _effective_response_format(
        response_format, runtime_options
    )
    if effective_response_format:
        payload["response_format"] = effective_response_format
    if "max_tokens" in runtime_options:
        payload["max_tokens"] = int(runtime_options["max_tokens"])
    if "stream" in runtime_options:
        payload["stream"] = bool(runtime_options["stream"])
    extra_body: dict[str, Any] = {}
    thinking_mode = str(runtime_options.get("thinking_mode") or "")
    if thinking_mode:
        extra_body["thinking"] = {"type": thinking_mode}
    if "enable_thinking" in runtime_options:
        extra_body["enable_thinking"] = bool(runtime_options["enable_thinking"])
    if "thinking_budget" in runtime_options:
        extra_body["thinking_budget"] = int(runtime_options["thinking_budget"])
    if "reasoning_effort" in runtime_options:
        extra_body["reasoning_effort"] = str(runtime_options["reasoning_effort"])
    if "reasoning_format" in runtime_options:
        extra_body["reasoning_format"] = str(runtime_options["reasoning_format"])
    if extra_body:
        # The generic OpenAI-compatible LiteLLM adapter calls the OpenAI SDK.
        # Provider extensions must therefore be merged into the raw request
        # body instead of being passed as unsupported SDK keyword arguments.
        payload["extra_body"] = extra_body
    return json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json"


def _asr_timestamp_granularities_supported(route: dict[str, Any]) -> bool:
    for deployment in route.get("deployments") or []:
        if not isinstance(deployment, dict):
            continue
        provider = str(deployment.get("provider") or "").strip().lower()
        transport = str(deployment.get("litellm_provider") or "").strip().lower()
        if transport == "groq" or (provider == "groq_asr" and not transport):
            return False
    return True


def _route_omits_legacy_sampling_parameters(route: dict[str, Any]) -> bool:
    return any(
        str(row.get("provider") or "").lower() == "gemini"
        and gemini_omits_legacy_sampling_parameters(row.get("model"))
        for row in route.get("deployments") or []
        if isinstance(row, dict)
    )


_RUNTIME_PROVIDER_OPTION_KEYS = (
    "asr_timestamp_granularity",
    "thinking_mode",
    "response_format",
    "enable_thinking",
    "thinking_budget",
    "reasoning_effort",
    "reasoning_format",
    "strip_reasoning_tags",
    "strip_json_fences",
    "max_tokens",
    "stream",
)


def _route_runtime_options(route: dict[str, Any]) -> dict[str, Any]:
    deployments = [
        dict(row) for row in route.get("deployments") or [] if isinstance(row, dict)
    ]
    if not deployments:
        return {}
    result: dict[str, Any] = {}
    for key in _RUNTIME_PROVIDER_OPTION_KEYS:
        values = [
            (
                key in dict(row.get("provider_options") or {}),
                dict(row.get("provider_options") or {}).get(key),
            )
            for row in deployments
        ]
        present = [value for configured, value in values if configured]
        if not present:
            continue
        if len(present) != len(deployments) or any(value != present[0] for value in present[1:]):
            raise ValueError(
                f"{key} must be identical across every deployment in a route"
            )
        result[key] = _validated_runtime_option(key, present[0])
    return result


def _effective_response_format(
    response_format: dict[str, Any] | None,
    runtime_options: dict[str, Any],
) -> dict[str, Any] | None:
    route_response_format = str(runtime_options.get("response_format") or "")
    locked = {"type": route_response_format} if route_response_format else None
    if response_format and locked and response_format != locked:
        raise ValueError("response_format does not match the route-locked value")
    return response_format or locked


def _validated_runtime_option(key: str, value: Any) -> Any:
    if key == "asr_timestamp_granularity":
        granularity = str(value or "").strip().lower()
        if granularity not in {"segment", "word"}:
            raise ValueError("route contains an unsupported ASR timestamp granularity")
        return granularity
    if key == "response_format":
        response_format = str(value or "").strip().lower()
        if response_format not in {"json_object", "text"}:
            raise ValueError("route contains an unsupported response_format")
        return response_format
    if key == "thinking_mode":
        mode = str(value or "").strip().lower()
        if mode not in {"enabled", "disabled", "auto"}:
            raise ValueError("route contains an unsupported thinking_mode")
        return mode
    if key == "reasoning_effort":
        effort = str(value or "").strip().lower()
        if effort not in {"none", "default", "low", "medium", "high", "max"}:
            raise ValueError("route contains an unsupported reasoning_effort")
        return effort
    if key == "reasoning_format":
        format_name = str(value or "").strip().lower()
        if format_name not in {"hidden", "raw", "parsed"}:
            raise ValueError("route contains an unsupported reasoning_format")
        return format_name
    if key in {
        "enable_thinking",
        "stream",
        "strip_reasoning_tags",
        "strip_json_fences",
    }:
        if not isinstance(value, bool):
            raise ValueError(f"route contains a non-boolean {key}")
        return value
    return _bounded_runtime_integer(key, value)


def _bounded_runtime_integer(key: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    minimum, maximum = (128, 32768) if key == "thinking_budget" else (1, 131072)
    if value < minimum or value > maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _bounded_retry_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_retries must be an integer")
    if value < 0 or value > 10:
        raise ValueError("max_retries must be between 0 and 10")
    return value


def _read_gateway_response(
    response: Any,
    *,
    streamed: bool,
) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    headers = dict(response.headers.items())
    http_status = int(response.status)
    if streamed:
        payload, response_payload_bytes = _streaming_chat_payload(response)
    else:
        raw_bytes = response.read()
        response_payload_bytes = len(raw_bytes)
        raw = raw_bytes.decode("utf-8", errors="replace")
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            raise ValueError("model gateway response must be a JSON object")
    return payload, headers, http_status, response_payload_bytes


def _streaming_chat_payload(response: Any) -> tuple[dict[str, Any], int]:
    event_count = 0
    response_payload_bytes = 0
    content_parts: list[str] = []
    response_id = ""
    response_model = ""
    created: Any = None
    finish_reason: Any = None
    usage: dict[str, Any] = {}
    role = "assistant"
    for raw_line in response:
        response_payload_bytes += len(raw_line)
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        if data == "[DONE]":
            break
        event = json.loads(data)
        if not isinstance(event, dict):
            raise ValueError("model gateway stream event must be a JSON object")
        event_count += 1
        response_id = str(event.get("id") or response_id)
        response_model = str(event.get("model") or response_model)
        created = event.get("created", created)
        if isinstance(event.get("usage"), dict):
            usage = dict(event["usage"])
        choices = event.get("choices") if isinstance(event.get("choices"), list) else []
        if not choices or not isinstance(choices[0], dict):
            continue
        choice = choices[0]
        finish_reason = choice.get("finish_reason", finish_reason)
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        message = (
            choice.get("message") if isinstance(choice.get("message"), dict) else {}
        )
        role = str(delta.get("role") or message.get("role") or role)
        for source in (delta, message):
            content = source.get("content")
            if isinstance(content, str) and content:
                content_parts.append(content)

    if event_count == 0:
        raise ValueError("model gateway stream did not contain any data events")
    response_message: dict[str, Any] = {
        "role": role,
        "content": "".join(content_parts),
    }

    payload: dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion",
        "model": response_model,
        "choices": [
            {
                "index": 0,
                "message": response_message,
                "finish_reason": finish_reason,
            }
        ],
    }
    if created is not None:
        payload["created"] = created
    if usage:
        payload["usage"] = usage
    return payload, response_payload_bytes


def _combined_prompt(prompt: str, text: str) -> str:
    clean_prompt = str(prompt or "").strip()
    clean_text = str(text or "").strip()
    if clean_prompt and clean_text:
        return f"{clean_prompt}\n\nInput/context:\n{clean_text}"
    return clean_prompt or clean_text


def _data_url(path: Path) -> tuple[str, str]:
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return mime, f"data:{mime};base64,{encoded}"


def _multipart_form_data(fields: dict[str, str], *, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----vkp-runtime-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'.encode("utf-8"),
            f"Content-Type: {mime}\r\n\r\n".encode(),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _gateway_api_key(secrets_path: Path) -> str:
    return str(_read_secret(MASTER_KEY_ID, secrets_path) or "")


def _response_deployment(
    route: dict[str, Any],
    payload: dict[str, Any],
    headers: dict[str, Any],
) -> dict[str, Any]:
    deployments = [dict(row) for row in route.get("deployments") or [] if isinstance(row, dict)]
    litellm_model_id = _header_value(headers, "x-litellm-model-id")
    if litellm_model_id:
        for index, row in enumerate(deployments):
            gateway_id = _gateway_deployment_id(route, index)
            if litellm_model_id == gateway_id:
                return {**row, "gateway_deployment_id": gateway_id, "selection": "litellm_model_id"}
    response_model = str(payload.get("model") or "")
    model_matches = [
        row
        for row in deployments
        if response_model and response_model in {str(row.get("model") or ""), str(row.get("id") or "")}
    ]
    if len(model_matches) == 1:
        return {**model_matches[0], "selection": "unique_response_model"}
    if len(deployments) == 1:
        return {**deployments[0], "selection": "singleton_route"}
    return {
        "id": "",
        "provider": "",
        "model": response_model,
        "gateway_deployment_id": litellm_model_id,
        "selection": "gateway_not_disclosed",
    }


def _header_value(headers: dict[str, Any], name: str) -> str:
    expected = str(name or "").casefold()
    for key, value in headers.items():
        if str(key).casefold() == expected:
            return str(value or "")
    return ""

def _usage(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("usage") or payload.get("usage_info") or {}
    return dict(value) if isinstance(value, dict) else {}


def _estimated_cost(payload: dict[str, Any], headers: dict[str, Any]) -> float | None:
    candidates = [payload.get("response_cost"), headers.get("x-litellm-response-cost"), headers.get("X-LiteLLM-Response-Cost")]
    for value in candidates:
        try:
            if value not in (None, ""):
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _content(
    task: str,
    payload: dict[str, Any],
    *,
    runtime_options: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    options = dict(runtime_options or {})
    sanitization = {
        "reasoning_tags_removed": 0,
        "json_fence_removed": False,
        "reasoning_tags_remaining": False,
    }
    if task == "asr":
        return str(payload.get("text") or ""), sanitization
    if task == "ocr":
        return {
            "pages": payload.get("pages") or [],
            "document_annotation": payload.get("document_annotation"),
        }, sanitization
    choices = payload.get("choices") if isinstance(payload.get("choices"), list) else []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") if isinstance(choices[0].get("message"), dict) else {}
        value = message.get("content", "")
    else:
        value = str(payload.get("content") or "")
    if not isinstance(value, str):
        return value, sanitization
    return _sanitise_chat_content(value, options=options)


_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>.*?</think\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
_THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", flags=re.IGNORECASE)
_JSON_FENCE_RE = re.compile(
    r"\A\s*\x60{3}(?:json)?\s*(.*?)\s*\x60{3}\s*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


def _sanitise_chat_content(
    value: str,
    *,
    options: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    cleaned = str(value)
    reasoning_tags_removed = 0
    if bool(options.get("strip_reasoning_tags")):
        cleaned, reasoning_tags_removed = _THINK_BLOCK_RE.subn("", cleaned)
        cleaned = cleaned.strip()

    json_fence_removed = False
    if bool(options.get("strip_json_fences")):
        fenced = _JSON_FENCE_RE.fullmatch(cleaned)
        if fenced:
            candidate = fenced.group(1).strip()
            try:
                json.loads(candidate)
            except json.JSONDecodeError:
                pass
            else:
                cleaned = candidate
                json_fence_removed = True

    return cleaned, {
        "reasoning_tags_removed": reasoning_tags_removed,
        "json_fence_removed": json_fence_removed,
        "reasoning_tags_remaining": bool(_THINK_TAG_RE.search(cleaned)),
    }


def _response_metadata(
    payload: dict[str, Any],
    http_status: int,
    headers: dict[str, Any],
    *,
    streamed: bool = False,
) -> dict[str, Any]:
    return {
        "http_status": http_status,
        "id": str(payload.get("id") or ""),
        "object": str(payload.get("object") or ""),
        "model": str(payload.get("model") or ""),
        "litellm_model_id": _header_value(headers, "x-litellm-model-id"),
        "litellm_model_group": _header_value(headers, "x-litellm-model-group"),
        "streamed": bool(streamed),
    }

def _base_result(route: dict[str, Any], task: str, *, endpoint: str, evidence: list[dict[str, Any]], consent_id: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "ok": False,
        "status": "planned",
        "task": task,
        "execution_location": str(route["execution_location"]),
        "route_id": str(route["route_id"]),
        "route_revision": str(route["route_revision"]),
        "virtual_model": str(route["virtual_model"]),
        "deployment": {},
        "provider": "",
        "latency_ms": 0,
        "usage": {},
        "estimated_cost": None,
        "content": "",
        "evidence": evidence,
        "consent_id": str(consent_id or ""),
        "gateway": {"endpoint": endpoint, "transport": "loopback_http"},
        "secrets_redacted": True,
        "transient_encodings_persisted": False,
        "remote_requests_made": False if str(route["execution_location"]) == "local" else None,
    }


def _failure(
    task: str,
    *,
    status: str,
    error: str,
    started: float,
    execution_location: str,
    route_id: str,
    route_revision: str,
    consent_id: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "ok": False,
        "status": status,
        "task": task,
        "execution_location": execution_location,
        "route_id": route_id,
        "route_revision": route_revision,
        "virtual_model": "",
        "deployment": {},
        "provider": "",
        "latency_ms": _elapsed_ms(started),
        "usage": {},
        "estimated_cost": None,
        "content": "",
        "evidence": [],
        "consent_id": str(consent_id or ""),
        "error": error,
        "secrets_redacted": True,
        "transient_encodings_persisted": False,
        "remote_requests_made": False if str(execution_location) == "local" else None,
    }


def _http_failure(result: dict[str, Any], code: int, *, started: float, detail: str) -> dict[str, Any]:
    status = "rate_limited" if code == 429 else ("provider_unavailable" if code >= 500 else "gateway_rejected")
    result.update({"ok": False, "status": status, "error": detail or f"HTTP {code}", "http_status": int(code), "latency_ms": _elapsed_ms(started)})
    return result


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read(4096).decode("utf-8", errors="replace")
        payload = json.loads(raw)
        if isinstance(payload, dict):
            detail = payload.get("error")
            if isinstance(detail, dict):
                return str(detail.get("message") or f"HTTP {exc.code}")
            if detail:
                return str(detail)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return f"HTTP {exc.code}"


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))
