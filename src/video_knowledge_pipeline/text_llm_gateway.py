from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .model_json import extract_json_document as extract_json_document
from .vision_api import provider_requires_api_key, redact_url_secrets, resolve_provider_config

SCHEMA = "video_knowledge_pipeline.text_llm_gateway.v1"


def resolve_text_provider_config(provider_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve text LLM provider config using VKP's existing model profile rules.

    This intentionally reuses the same provider/env/config surface as the vision
    API layer so model settings do not drift. API keys stay in env or explicit
    runtime args and should not be persisted.
    """

    raw_cfg = dict(provider_config or {})
    cfg = resolve_provider_config(provider_config)
    resolved = {
        **cfg,
        "base_url": resolve_openai_compatible_api_base_url(str(cfg.get("base_url") or "")),
        "interface": "openai_chat_completions",
    }
    provider_options = raw_cfg.get("provider_options")
    if isinstance(provider_options, dict):
        extra_body = dict(resolved.get("extra_body") or {})
        for key in (
            "enable_thinking",
            "thinking_budget",
            "reasoning_effort",
            "reasoning_format",
        ):
            if key in provider_options:
                extra_body[key] = provider_options[key]
        if extra_body:
            resolved["extra_body"] = extra_body
        if "response_format" in provider_options:
            resolved["response_format_override"] = _normalize_response_format(provider_options["response_format"])
        if "max_tokens" in provider_options:
            resolved["max_tokens_cap"] = max(1, int(provider_options["max_tokens"]))
    for key in ("extra_body", "thinking"):
        if key in raw_cfg:
            resolved[key] = raw_cfg[key]
    if raw_cfg.get("disable_thinking") is True and "thinking" not in resolved:
        resolved["thinking"] = {"type": "disabled"}
    return resolved


def _normalize_response_format(value: Any) -> dict[str, Any]:
    result = dict(value) if isinstance(value, dict) else {"type": str(value)}
    kind = result.get("type")
    if kind not in {"text", "json_object", "json_schema"}:
        raise ValueError("response_format must be text, json_object, or json_schema")
    if kind == "json_schema":
        spec = result.get("json_schema")
        if not isinstance(spec, dict) or not str(spec.get("name") or "").strip() or not isinstance(spec.get("schema"), dict):
            raise ValueError("json_schema response_format requires a name and a schema object")
        from jsonschema.validators import validator_for
        validator_for(spec["schema"]).check_schema(spec["schema"])
    return result


def _effective_response_format(cfg: dict[str, Any], requested: dict[str, Any] | None) -> dict[str, Any] | None:
    override = cfg.get("response_format_override")
    if override is not None:
        return _normalize_response_format(override)
    if not requested:
        return requested
    result = _normalize_response_format(requested)
    # Local servers can return JSON text without implementing OpenAI's json_object
    # mode. Schema/task validation remains the responsibility of the caller.
    endpoint = urllib.parse.urlsplit(str(cfg.get("base_url") or ""))
    local_endpoint = endpoint.hostname in {"localhost", "127.0.0.1", "::1"}
    local_profile = cfg.get("provider") in {"local_vlm", "local_qwen_vl"}
    if result["type"] == "json_object" and local_endpoint and (local_profile or endpoint.port == 1234):
        return {"type": "text"}
    return result


def text_llm_provider_smoke(
    provider_config: dict[str, Any] | None = None,
    *,
    execute: bool = False,
    prompt: str = "Reply with exactly: ok",
) -> dict[str, Any]:
    cfg = resolve_text_provider_config(provider_config)
    public_cfg = _public_provider_config(cfg)
    plan = {
        "schema": SCHEMA,
        "provider": public_cfg,
        "execute": bool(execute),
        "request_plan": {
            "url": redact_url_secrets(openai_compatible_chat_completions_url(cfg)),
            "model": cfg.get("model"),
            "message_count": 1,
            "temperature": 0,
        },
        "source_reuse": {
            "project": "alpha03123/vsummary",
            "commit_reviewed": "1b8ac39",
            "reused_patterns": [
                "OpenAI-compatible base URL normalization",
                "JSON response extraction from raw/fenced/balanced text",
                "provider smoke ping shape",
            ],
        },
        "secrets_redacted": True,
    }
    if not execute:
        return {**plan, "ok": True, "status": "planned", "next_action": "rerun with execute=true after provider credentials are configured"}
    result = call_openai_compatible_text(
        provider_config=cfg,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=16,
    )
    return {
        **plan,
        "ok": bool(result.get("ok")),
        "status": "ok" if result.get("ok") else str(result.get("error") or "failed"),
        "content": result.get("content", ""),
        "error": result.get("error", ""),
    }


def call_openai_compatible_text(
    *,
    provider_config: dict[str, Any],
    messages: list[dict[str, Any]],
    temperature: float = 0,
    response_format: dict[str, Any] | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    cfg = resolve_text_provider_config(provider_config)
    if provider_requires_api_key(cfg) and not cfg.get("api_key"):
        return {"ok": False, "error": "missing_api_key", "content": ""}
    body = build_openai_compatible_text_body(
        cfg=cfg,
        messages=messages,
        temperature=temperature,
        response_format=response_format,
        max_tokens=max_tokens,
    )
    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg.get('api_key')}"
    request = urllib.request.Request(
        openai_compatible_chat_completions_url(cfg),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=int(cfg.get("timeout_seconds") or 60)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = _http_error_detail(exc)
        error_class = "response_format_unsupported" if exc.code == 400 and "response_format" in detail else "http_error"
        return {"ok": False, "error": detail, "error_class": error_class, "http_status": exc.code, "content": ""}
    except Exception as exc:  # pragma: no cover - network is optional and normally gated.
        return {"ok": False, "error": str(exc), "content": ""}
    content = ""
    finish_reason = ""
    reasoning_chars = 0
    try:
        choice = payload["choices"][0]
        message = choice.get("message") if isinstance(choice, dict) else {}
        if not isinstance(message, dict):
            message = {}
        content = str(message.get("content") or "")
        finish_reason = str(choice.get("finish_reason") or "") if isinstance(choice, dict) else ""
        reasoning_chars = len(str(message.get("reasoning_content") or ""))
    except (KeyError, IndexError, TypeError):
        content = json.dumps(payload, ensure_ascii=False)
    if not content.strip():
        detail = f"empty_content; finish_reason={finish_reason or 'unknown'}"
        if reasoning_chars:
            detail = f"empty_content_reasoning_only; finish_reason={finish_reason or 'unknown'}; reasoning_chars={reasoning_chars}"
        return {"ok": False, "error": detail, "content": ""}
    return {"ok": True, "error": "", "content": content, "raw_response": payload}

def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    """Return bounded provider diagnostics without reflecting request content or secrets."""

    detail = ""
    try:
        raw = exc.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            detail = str(error.get("message") or error.get("type") or error.get("code") or "")
        elif error:
            detail = str(error)
    except Exception:
        detail = ""
    detail = re.sub(r"[\r\n\x00-\x1f]+", " ", detail).strip()[:1000]
    prefix = f"http_{int(getattr(exc, 'code', 0) or 0)}"
    return f"{prefix}: {detail}" if detail else prefix


def build_openai_compatible_text_body(
    *,
    cfg: dict[str, Any],
    messages: list[dict[str, Any]],
    temperature: float,
    response_format: dict[str, Any] | None,
    max_tokens: int | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": cfg.get("model"),
        "messages": messages,
        "temperature": temperature,
    }
    effective_response_format = _effective_response_format(cfg, response_format)
    if isinstance(effective_response_format, dict) and effective_response_format:
        body["response_format"] = effective_response_format
    effective_max_tokens = max_tokens
    max_tokens_cap = cfg.get("max_tokens_cap")
    if max_tokens_cap is not None:
        cap = max(1, int(max_tokens_cap))
        effective_max_tokens = cap if effective_max_tokens is None else min(cap, max(1, int(effective_max_tokens)))
    if effective_max_tokens is not None:
        body["max_tokens"] = max(1, int(effective_max_tokens))
    extra_body = cfg.get("extra_body")
    if isinstance(extra_body, dict):
        body.update(extra_body)
    thinking = cfg.get("thinking")
    if isinstance(thinking, dict):
        body["thinking"] = thinking
    return body


# Backward-compatible private alias for existing internal callers/tests.
_build_openai_compatible_text_body = build_openai_compatible_text_body


# Adapted from alpha03123/vsummary src/backend/shared/llm/base_url.py.
def normalize_provider_base_url(value: str) -> str:
    normalized = str(value or "").strip().rstrip("/")
    if not normalized:
        return normalized
    for endpoint in ("/chat/completions", "/responses", "/completions"):
        if normalized.endswith(endpoint):
            return normalized[: -len(endpoint)].rstrip("/")
    return normalized


# Adapted from alpha03123/vsummary src/backend/shared/llm/base_url.py.
def resolve_openai_compatible_api_base_url(value: str) -> str:
    normalized = normalize_provider_base_url(value)
    parsed = urllib.parse.urlparse(normalized)
    if not parsed.scheme or not parsed.netloc:
        return normalized
    path = parsed.path.rstrip("/")
    if path.endswith("/api/coding/v3"):
        return f"{parsed.scheme}://{parsed.netloc}{path}"
    if not re.search(r"/v\d+$", path):
        path = f"{path}/v1" if path else "/v1"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def openai_compatible_chat_completions_url(provider_config: dict[str, Any]) -> str:
    base_url = resolve_openai_compatible_api_base_url(str(provider_config.get("base_url") or "https://api.openai.com/v1"))
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/api/coding/v3"):
        return f"{base_url}/chat/completions"
    return f"{base_url.rstrip('/')}/chat/completions"


def _public_provider_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": cfg.get("provider", ""),
        "model": cfg.get("model", ""),
        "base_url": redact_url_secrets(str(cfg.get("base_url") or "")),
        "api_key_required": provider_requires_api_key(cfg),
        "api_key_configured": bool(cfg.get("api_key")),
        "interface": cfg.get("interface", "openai_chat_completions"),
    }
