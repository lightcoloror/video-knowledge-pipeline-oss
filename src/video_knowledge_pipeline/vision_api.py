from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .config import vision_execution_profile
from .model_json import extract_json_document
from .model_defaults import GEMINI_DEFAULT_MODEL, gemini_omits_legacy_sampling_parameters


def resolve_provider_config(provider_config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(provider_config or {})
    proxy_backend = str(cfg.get("adapter_backend") or "").strip().lower() == "proxy"
    configured = vision_execution_profile()
    configured_provider = _provider_name(str(configured.get("provider") or ""))
    env_provider_raw = str(os.environ.get("LECTURE_VISION_PROVIDER") or "").strip()
    env_provider = _provider_name(env_provider_raw) if env_provider_raw else ""
    explicit_provider = _provider_name(str(cfg.get("provider") or "")) if cfg.get("provider") else ""
    provider = _provider_name(str(cfg.get("provider") or os.environ.get("LECTURE_VISION_PROVIDER") or configured.get("provider") or "openai_compatible"))
    profile = _provider_profile(provider)
    openai_compatible_base_url = os.environ.get("OPENAI_BASE_URL") or ""
    volcengine_openai_compatible = _is_volcengine_coding_plan_base_url(openai_compatible_base_url)
    local_provider = provider in {"local_qwen_vl", "local_vlm"}
    if local_provider and not proxy_backend:
        # An explicit empty local key is intentional; never inherit a cloud key.
        api_key = cfg.get("api_key") if "api_key" in cfg else (
            os.environ.get("LOCAL_QWEN_VL_API_KEY") or os.environ.get("LOCAL_VLM_API_KEY") or ""
        )
    else:
        api_key = "" if proxy_backend else (cfg.get("api_key") or os.environ.get("LECTURE_VISION_API_KEY"))
    if not api_key and not proxy_backend and not local_provider:
        if provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        elif provider == "volcengine_coding_plan":
            api_key = (
                os.environ.get("ARK_API_KEY")
                or os.environ.get("VOLCENGINE_API_KEY")
                or os.environ.get("LLM_API_KEY")
                or (os.environ.get("OPENAI_API_KEY") if volcengine_openai_compatible else "")
            )
        elif provider == "agnes":
            api_key = os.environ.get("AGNES_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
    return {
        "provider": provider,
        "base_url": str(
            cfg.get("base_url")
            or _provider_scoped_env_value("LECTURE_VISION_BASE_URL", provider=provider, env_provider=env_provider, explicit_provider=explicit_provider)
            or _configured_provider_value(configured, configured_provider, provider, "base_url")
            or (os.environ.get("LLM_BASE_URL") if provider == "volcengine_coding_plan" else "")
            or (openai_compatible_base_url if provider == "volcengine_coding_plan" and volcengine_openai_compatible else "")
            or (
                os.environ.get("LOCAL_QWEN_VL_BASE_URL") or os.environ.get("LOCAL_VLM_BASE_URL")
                if provider in {"local_qwen_vl", "local_vlm"}
                else ""
            )
            or profile["base_url"]
        ),
        "api_key": _normalise_api_key(api_key),
        "model": str(
            cfg.get("model")
            or _provider_scoped_env_value("LECTURE_VISION_MODEL", provider=provider, env_provider=env_provider, explicit_provider=explicit_provider)
            or _configured_provider_value(configured, configured_provider, provider, "model")
            or (os.environ.get("LLM_MODEL") if provider == "volcengine_coding_plan" else "")
            or (
                os.environ.get("LOCAL_QWEN_VL_MODEL") or os.environ.get("LOCAL_VLM_MODEL")
                if provider in {"local_qwen_vl", "local_vlm"}
                else ""
            )
            or profile["model"]
        ),
        "max_tokens": _positive_int_or_none(cfg.get("max_tokens")),
        "timeout_seconds": int(cfg.get("timeout_seconds") or os.environ.get("LECTURE_VISION_TIMEOUT_SECONDS") or 60),
        "adapter_backend": str(cfg.get("adapter_backend") or ""),
        "api_key_optional": bool(cfg.get("api_key_optional")),
        "provider_options": dict(cfg.get("provider_options") or {}),
        "location": str(cfg.get("location") or ""),
        "execution_location": str(cfg.get("execution_location") or ""),
        "route_id": str(cfg.get("route_id") or ""),
        "route_revision": str(cfg.get("route_revision") or ""),
        "virtual_model": str(cfg.get("virtual_model") or ""),
        "profile_id": str(cfg.get("profile_id") or ""),
        "credential_status": str(cfg.get("credential_status") or ""),
        "credential_ready": bool(cfg.get("credential_ready")),
        "gateway_configured": bool(cfg.get("gateway_configured")),
        "gateway_ready": bool(cfg.get("gateway_ready")),
        "gateway_status": str(cfg.get("gateway_status") or ""),
        "provider_config_source": str(cfg.get("provider_config_source") or ""),
        "consent_id": str(cfg.get("consent_id") or ""),
        "capabilities": list(cfg.get("capabilities") or []),
    }


def call_vision_model(
    *,
    provider_config: dict[str, Any],
    prompt: str,
    image_paths: list[str],
    max_tokens: int | None = None,
) -> dict[str, Any]:
    provider = _provider_name(str(provider_config.get("provider") or "openai_compatible"))
    if provider == "fixture":
        result = call_fixture_vision(provider_config=provider_config, prompt=prompt, image_paths=image_paths)
        return _with_completion_metadata(result, max_tokens=max_tokens, finish_reason="stop")
    if provider == "gemini":
        result = call_gemini_vision(provider_config=provider_config, prompt=prompt, image_paths=image_paths)
        return _with_completion_metadata(result, max_tokens=max_tokens, finish_reason=str(result.get("finish_reason") or ""))
    return call_openai_compatible_vision(
        provider_config=provider_config,
        prompt=prompt,
        image_paths=image_paths,
        max_tokens=max_tokens,
    )


def test_vision_provider(
    provider_config: dict[str, Any] | None = None,
    *,
    image_paths: list[str] | None = None,
) -> dict[str, Any]:
    cfg = resolve_provider_config(provider_config)
    checks: list[dict[str, Any]] = []
    checks.append(_vision_check(cfg, "text_ping", "Return only JSON: {\"ok\": true}", []))
    frames = [str(path) for path in image_paths or [] if str(path)]
    if frames:
        checks.append(_vision_check(cfg, "single_image_json", "Return JSON describing visible text and non-text visual information.", frames[:1]))
    if len(frames) >= 2:
        checks.append(_vision_check(cfg, "multi_image_json", "Return JSON describing changes across these ordered frames.", frames[:8]))
    return {
        "schema": "lecture_vision_provider_test.v1",
        "provider": _public_provider_config(cfg),
        "checks": checks,
        "ok": all(check.get("ok") for check in checks),
        "status": _provider_test_status(checks),
        "safe_to_execute": all(check.get("ok") for check in checks),
        "error_class": _provider_error_class(checks),
        "error_summary": _provider_error_summary(checks),
        "failure_diagnosis": _provider_failure_diagnosis(checks),
        "secrets_redacted": True,
    }


def call_openai_compatible_vision(
    *,
    provider_config: dict[str, Any],
    prompt: str,
    image_paths: list[str],
    max_tokens: int | None = None,
) -> dict[str, Any]:
    if provider_requires_api_key(provider_config) and not provider_config.get("api_key"):
        return _with_completion_metadata(
            {"ok": False, "error": "missing_api_key", "content": ""},
            max_tokens=max_tokens,
            finish_reason="",
        )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in image_paths:
        data_url = _image_data_url(path)
        if data_url:
            content.append({"type": "image_url", "image_url": {"url": data_url}})
    body = {
        "model": provider_config.get("model"),
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
    }
    if max_tokens is not None and int(max_tokens) > 0:
        body["max_tokens"] = int(max_tokens)
    headers = {
        "Content-Type": "application/json",
    }
    if provider_config.get("api_key"):
        headers["Authorization"] = f"Bearer {provider_config.get('api_key')}"
    request = urllib.request.Request(
        _openai_compatible_chat_completions_url(provider_config),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=int(provider_config.get("timeout_seconds") or 60)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # pragma: no cover - network is optional and normally disabled in tests.
        return _with_completion_metadata(
            {"ok": False, "error": str(exc), "content": ""},
            max_tokens=max_tokens,
            finish_reason="",
        )
    content_text = ""
    finish_reason = ""
    reasoning_chars = 0
    try:
        choice = payload["choices"][0]
        message = choice.get("message") if isinstance(choice, dict) else {}
        if not isinstance(message, dict):
            message = {}
        content_text = str(message.get("content") or "")
        finish_reason = str(choice.get("finish_reason") or "") if isinstance(choice, dict) else ""
        reasoning_chars = len(str(message.get("reasoning_content") or ""))
    except (KeyError, IndexError, TypeError):
        content_text = json.dumps(payload, ensure_ascii=False)
    if not content_text.strip():
        detail = f"empty_content; finish_reason={finish_reason or 'unknown'}"
        if reasoning_chars:
            detail = f"empty_content_reasoning_only; finish_reason={finish_reason or 'unknown'}; reasoning_chars={reasoning_chars}"
        return _with_completion_metadata(
            {"ok": False, "error": detail, "content": "", "raw_response": payload},
            max_tokens=max_tokens,
            finish_reason=finish_reason,
        )
    return _with_completion_metadata(
        {"ok": True, "error": "", "content": content_text, "raw_response": payload},
        max_tokens=max_tokens,
        finish_reason=finish_reason,
    )


def call_fixture_vision(
    *,
    provider_config: dict[str, Any],
    prompt: str,
    image_paths: list[str],
) -> dict[str, Any]:
    """Deterministic local provider for controlled execution smoke tests."""
    model = str(provider_config.get("model") or "fixture-vision")
    frame_paths = [str(path) for path in image_paths if str(path)]
    if len(frame_paths) >= 2:
        payload = {
            "event_sequence": [
                f"Observed ordered frame {index + 1}: {Path(path).name}" for index, path in enumerate(frame_paths)
            ],
            "state_changes": ["Fixture provider detected a controlled temporal frame sequence."],
            "operation_steps": ["Inspect frames in timestamp order.", "Record visible state changes with evidence paths."],
            "causal_links": ["Earlier frames provide context for later visible state changes."],
            "possible_missing_points": [],
            "confidence": 0.5,
            "evidence_frame_paths": frame_paths,
            "source": f"{model}:fixture",
        }
    else:
        payload = {
            "objects": ["fixture-observed-frame"],
            "actions": ["controlled local vision execution"],
            "interface_state": "Fixture provider returned deterministic visual understanding without network access.",
            "spatial_relations": ["Evidence frame path is preserved for human review."],
            "instructor_focus": "Not inferred by fixture provider.",
            "non_text_information": ["This local fixture proves the execution, audit, refresh, and restore chain."],
            "confidence": 0.5,
            "keep_image_reason": "Retain the frame as evidence for the controlled execution smoke test.",
            "evidence_frame_paths": frame_paths,
            "source": f"{model}:fixture",
        }
    return {"ok": True, "error": "", "content": json.dumps(payload, ensure_ascii=False), "raw_response": payload}


def call_gemini_vision(
    *,
    provider_config: dict[str, Any],
    prompt: str,
    image_paths: list[str],
) -> dict[str, Any]:
    if provider_requires_api_key(provider_config) and not provider_config.get("api_key"):
        return {"ok": False, "error": "missing_api_key", "content": ""}
    parts: list[dict[str, Any]] = [{"text": prompt}]
    for path in image_paths:
        inline_data = _image_inline_data(path)
        if inline_data:
            parts.append({"inline_data": inline_data})
    body = {"contents": [{"role": "user", "parts": parts}]}
    if not gemini_omits_legacy_sampling_parameters(provider_config.get("model")):
        body["generationConfig"] = {"temperature": 0}
    request = urllib.request.Request(
        _gemini_endpoint(provider_config),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-goog-api-key": str(provider_config.get("api_key") or ""),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=int(provider_config.get("timeout_seconds") or 60)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # pragma: no cover - network is optional and normally disabled in tests.
        return {"ok": False, "error": str(exc), "content": ""}
    content_text = ""
    try:
        parts = payload["candidates"][0]["content"]["parts"]
        content_text = "\n".join(str(part.get("text") or "") for part in parts if isinstance(part, dict)).strip()
    except (KeyError, IndexError, TypeError):
        content_text = json.dumps(payload, ensure_ascii=False)
    return {"ok": True, "error": "", "content": content_text, "raw_response": payload}


def parse_model_json(text: str) -> dict[str, Any]:
    try:
        data = extract_json_document(text)
    except ValueError:
        return {"_parse_failed": True, "summary": text, "raw_content": text}
    return data if isinstance(data, dict) else {"result": data}


def _image_data_url(path: str) -> str:
    image_path = Path(path).expanduser()
    if not image_path.exists() or not image_path.is_file():
        return ""
    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _image_inline_data(path: str) -> dict[str, str]:
    image_path = Path(path).expanduser()
    if not image_path.exists() or not image_path.is_file():
        return {}
    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {"mime_type": mime, "data": encoded}


def _gemini_endpoint(provider_config: dict[str, Any]) -> str:
    base_url = str(provider_config.get("base_url") or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    model = str(provider_config.get("model") or GEMINI_DEFAULT_MODEL)
    if ":generateContent" in base_url:
        return base_url
    return f"{base_url}/models/{urllib.parse.quote(model)}:generateContent"


def _provider_name(value: str) -> str:
    normalised = value.strip().lower().replace("-", "_")
    if normalised in {"fixture", "local_fixture", "mock", "test_fixture"}:
        return "fixture"
    if normalised in {"gemini", "google", "google_gemini"}:
        return "gemini"
    if normalised in {"openai", "openai_vision"}:
        return "openai"
    if normalised in {"agnes", "agnes_ai"}:
        return "agnes"
    if normalised in {"volcengine", "volcengine_coding_plan", "ark", "ark_coding_plan", "coding_plan", "huoshan", "huoshan_ark", "火山", "火山方舟"}:
        return "volcengine_coding_plan"
    if normalised in {"local_qwen", "local_qwen_vl", "qwen_local", "qwen_vl_local", "qwen2_5_vl_local", "qwen25_vl_local"}:
        return "local_qwen_vl"
    if normalised in {"local_vlm", "local_openai_compatible", "localhost_vlm"}:
        return "local_vlm"
    if normalised in {"custom", "custom_openai_compatible", "openai_compatible"}:
        return "custom_openai_compatible" if normalised.startswith("custom") else "openai_compatible"
    return "openai_compatible"


def _is_volcengine_coding_plan_base_url(value: str | None) -> bool:
    parsed = urllib.parse.urlparse(str(value or "").strip())
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    return host.endswith("volces.com") and "/api/coding" in path

def _provider_profile(provider: str) -> dict[str, str]:
    profiles = {
        "fixture": {"base_url": "local://fixture", "model": "fixture-vision"},
        "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta", "model": GEMINI_DEFAULT_MODEL},
        "openai": {"base_url": "https://api.openai.com/v1/chat/completions", "model": "gpt-4o-mini"},
        "openai_compatible": {"base_url": "https://api.openai.com/v1/chat/completions", "model": "gpt-4o-mini"},
        "custom_openai_compatible": {"base_url": "https://api.openai.com/v1/chat/completions", "model": "gpt-4o-mini"},
        "agnes": {"base_url": "https://apihub.agnes-ai.com/v1", "model": "agnes-1.5-flash"},
        "volcengine_coding_plan": {"base_url": "https://ark.cn-beijing.volces.com/api/coding/v3", "model": "ark-code-latest"},
        "local_qwen_vl": {"base_url": "http://127.0.0.1:8000/v1", "model": "Qwen/Qwen2.5-VL-3B-Instruct"},
        "local_vlm": {"base_url": "http://127.0.0.1:8000/v1", "model": "local-vlm"},
    }
    return profiles.get(provider, profiles["openai_compatible"])


def _openai_compatible_chat_completions_url(provider_config: dict[str, Any]) -> str:
    base_url = str(provider_config.get("base_url") or "https://api.openai.com/v1/chat/completions").rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/api/coding/v3"):
        return f"{base_url}/chat/completions"
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


def _configured_provider_value(configured: dict[str, Any], configured_provider: str, provider: str, key: str) -> str:
    if configured_provider != provider:
        return ""
    return str(configured.get(key) or "").strip()


def _provider_scoped_env_value(env_name: str, *, provider: str, env_provider: str, explicit_provider: str) -> str:
    if explicit_provider and env_provider and explicit_provider != env_provider:
        return ""
    if env_provider and env_provider != provider:
        return ""
    return str(os.environ.get(env_name) or "").strip()


def _vision_check(cfg: dict[str, Any], name: str, prompt: str, image_paths: list[str]) -> dict[str, Any]:
    image_payload = _image_payload_summary(image_paths)
    if provider_requires_api_key(cfg) and not cfg.get("api_key"):
        return {
            "name": name,
            "ok": False,
            "status": "missing_api_key",
            "error": "missing_api_key",
            "error_class": "missing_api_key",
            "image_count": len(image_paths),
            "image_payload": image_payload,
        }
    configured_max_tokens = _positive_int_or_none(cfg.get("max_tokens"))
    if configured_max_tokens is None:
        response = call_vision_model(provider_config=cfg, prompt=prompt, image_paths=image_paths)
    else:
        response = call_vision_model(
            provider_config=cfg,
            prompt=prompt,
            image_paths=image_paths,
            max_tokens=configured_max_tokens,
        )
    parsed = parse_model_json(str(response.get("content") or ""))
    truncated = bool(response.get("truncated"))
    status = "truncated" if truncated else _check_status(response=response, parsed=parsed)
    error = str(
        response.get("error")
        or ("model_output_truncated" if truncated else "model_output_parse_failed" if parsed.get("_parse_failed") else "")
    )
    complete = bool(response.get("ok")) and not parsed.get("_parse_failed") and not truncated
    return {
        "name": name,
        "ok": complete,
        "status": status,
        "error": error,
        "error_class": "model_output_truncated" if truncated else _classify_provider_error(error, status=status),
        "request_max_tokens": response.get("request_max_tokens"),
        "request_max_tokens_omitted": bool(response.get("request_max_tokens_omitted", True)),
        "finish_reason": str(response.get("finish_reason") or ""),
        "truncated": truncated,
        "complete": complete,
        "image_count": len(image_paths),
        "image_payload": image_payload,
        "parsed_preview": {key: parsed.get(key) for key in list(parsed)[:6] if key != "raw_content"},
    }


def _public_provider_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": cfg.get("provider"),
        "base_url": redact_url_secrets(str(cfg.get("base_url") or "")),
        "model": cfg.get("model"),
        "max_tokens": cfg.get("max_tokens"),
        "api_key_required": provider_requires_api_key(cfg),
        "api_key_configured": bool(cfg.get("api_key")),
        "timeout_seconds": cfg.get("timeout_seconds"),
    }


def provider_runtime_diagnostics(provider_config: dict[str, Any]) -> dict[str, Any]:
    """Return secret-safe endpoint and environment diagnostics for provider recovery."""
    cfg = resolve_provider_config(provider_config)
    base_url = str(cfg.get("base_url") or "")
    parsed = urllib.parse.urlparse(base_url)
    request_url = _provider_request_url(cfg)
    return {
        "provider": cfg.get("provider"),
        "model": cfg.get("model"),
        "base_url": redact_url_secrets(base_url),
        "base_url_scheme": parsed.scheme,
        "base_url_host": parsed.netloc,
        "request_url": redact_url_secrets(request_url),
        "endpoint_kind": _provider_endpoint_kind(cfg),
        "timeout_seconds": cfg.get("timeout_seconds"),
        "api_key_required": provider_requires_api_key(cfg),
        "api_key_configured": bool(cfg.get("api_key")),
        "proxy_env": {
            "HTTP_PROXY": bool(os.environ.get("HTTP_PROXY")),
            "HTTPS_PROXY": bool(os.environ.get("HTTPS_PROXY")),
            "ALL_PROXY": bool(os.environ.get("ALL_PROXY")),
            "NO_PROXY": bool(os.environ.get("NO_PROXY")),
        },
        "secrets_redacted": True,
    }


def provider_requires_api_key(provider_config: dict[str, Any]) -> bool:
    if str(provider_config.get("adapter_backend") or "").strip().lower() == "proxy":
        return False
    provider = _provider_name(str(provider_config.get("provider") or ""))
    return provider not in {"fixture", "local_qwen_vl", "local_vlm"}


def _normalise_api_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"<your key>", "<your-key>", "<api key>", "<api-key>", "your-key", "your api key", "..."}:
        return ""
    return text


def _check_status(*, response: dict[str, Any], parsed: dict[str, Any]) -> str:
    if response.get("ok") and not parsed.get("_parse_failed"):
        return "ok"
    error = str(response.get("error") or "")
    if error:
        return _classify_provider_error(error, status="")
    if parsed.get("_parse_failed"):
        return "model_output_parse_failed"
    return "provider_failed"


def _provider_test_status(checks: list[dict[str, Any]]) -> str:
    if all(check.get("ok") for check in checks):
        return "ok"
    image_status = _image_failure_status(checks)
    if image_status:
        return image_status
    for check in checks:
        status = str(check.get("status") or "")
        if status and status != "ok":
            return status
    return "provider_failed"


def _provider_error_class(checks: list[dict[str, Any]]) -> str:
    image_status = _image_failure_status(checks)
    if image_status:
        return image_status
    for check in checks:
        if check.get("ok"):
            continue
        error_class = str(check.get("error_class") or "")
        if error_class:
            return error_class
    return ""


def _provider_error_summary(checks: list[dict[str, Any]]) -> str:
    for check in checks:
        if check.get("ok"):
            continue
        error = str(check.get("error") or check.get("status") or "")
        if error:
            return error
    return ""


def _provider_failure_diagnosis(checks: list[dict[str, Any]]) -> dict[str, Any]:
    text_check = next((check for check in checks if str(check.get("name") or "") == "text_ping"), {})
    image_checks = [check for check in checks if int(check.get("image_count") or 0) > 0]
    failed_image_checks = [check for check in image_checks if not check.get("ok")]
    status = _image_failure_status(checks) or _provider_error_class(checks) or "ok"
    total_payload_bytes = 0
    for check in image_checks:
        payload = check.get("image_payload") if isinstance(check.get("image_payload"), dict) else {}
        total_payload_bytes = max(total_payload_bytes, int(payload.get("total_bytes") or 0))
    return {
        "status": status,
        "text_ping_ok": bool(text_check.get("ok")),
        "image_checks_run": len(image_checks),
        "image_checks_failed": len(failed_image_checks),
        "failed_image_check_names": [str(check.get("name") or "") for check in failed_image_checks],
        "max_image_payload_bytes": total_payload_bytes,
        "likely_causes": _failure_likely_causes(status),
    }


def _image_failure_status(checks: list[dict[str, Any]]) -> str:
    text_ok = any(str(check.get("name") or "") == "text_ping" and check.get("ok") for check in checks)
    if not text_ok:
        return ""
    image_failures = [check for check in checks if int(check.get("image_count") or 0) > 0 and not check.get("ok")]
    if not image_failures:
        return ""
    classes = {str(check.get("error_class") or check.get("status") or "") for check in image_failures}
    if "provider_payload_too_large" in classes:
        return "text_only_ok_image_payload_too_large"
    if "provider_image_not_supported" in classes:
        return "text_only_ok_image_not_supported"
    if "provider_unreachable" in classes or "provider_transport_error" in classes:
        return "text_only_ok_image_timeout"
    if "model_output_parse_failed" in classes:
        return "text_only_ok_image_parse_failed"
    return "text_only_ok_image_failed"


def _failure_likely_causes(status: str) -> list[str]:
    if status == "text_only_ok_image_timeout":
        return [
            "Text endpoint/auth works, but image or multi-image requests timed out.",
            "Increase timeout_seconds, reduce image count/size, or verify the provider/model supports vision payloads.",
        ]
    if status == "text_only_ok_image_not_supported":
        return ["The selected model or endpoint appears to reject image payloads; switch to a vision-capable model/profile."]
    if status == "text_only_ok_image_payload_too_large":
        return ["Image payload is likely too large; reduce resolution, frame count, or JPEG quality before retrying."]
    if status == "text_only_ok_image_parse_failed":
        return ["The provider responded to image requests but did not return valid JSON; adjust prompt or enable JSON repair/import."]
    if status == "missing_api_key":
        return ["Set the provider-specific API key in the local environment or explicit provider config."]
    if status == "ok":
        return []
    return ["Check provider base_url, model, proxy/network, and provider-specific vision support."]


def _image_payload_summary(image_paths: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total = 0
    for path in image_paths:
        candidate = Path(path).expanduser()
        size = candidate.stat().st_size if candidate.exists() and candidate.is_file() else 0
        total += size
        rows.append({"path": str(path), "bytes": size})
    return {"image_count": len(image_paths), "total_bytes": total, "images": rows[:12]}


def _classify_provider_error(error: str, *, status: str = "") -> str:
    if status in {"missing_api_key", "model_output_parse_failed"}:
        return status
    lowered = str(error or "").lower()
    if not lowered:
        return status or "provider_failed"
    if "missing_api_key" in lowered:
        return "missing_api_key"
    if "timed out" in lowered or "timeout" in lowered or "read operation timed out" in lowered:
        return "provider_unreachable"
    if "getaddrinfo" in lowered or "name or service not known" in lowered or "nodename nor servname" in lowered or "no such host" in lowered:
        return "provider_dns_failed"
    if "tunnel connection failed" in lowered or "proxy" in lowered:
        return "provider_proxy_failed"
    if "connection refused" in lowered or "actively refused" in lowered:
        return "provider_connection_refused"
    if "ssl" in lowered or "tls" in lowered or "eof" in lowered or "transport" in lowered or "connection reset" in lowered:
        return "provider_transport_error"
    if "http error 401" in lowered or "unauthorized" in lowered or "forbidden" in lowered or "invalid api key" in lowered:
        return "provider_auth_failed"
    if "http error 429" in lowered or "rate limit" in lowered or "quota" in lowered:
        return "provider_rate_limited"
    if "http error 413" in lowered or "payload too large" in lowered or "request entity too large" in lowered or "content too large" in lowered:
        return "provider_payload_too_large"
    if "image not supported" in lowered or "does not support image" in lowered or "doesn't support image" in lowered or "unsupported image" in lowered or "invalid image_url" in lowered:
        return "provider_image_not_supported"
    return status or "provider_failed"


def _with_completion_metadata(
    result: dict[str, Any],
    *,
    max_tokens: int | None,
    finish_reason: str,
) -> dict[str, Any]:
    request_max_tokens = _positive_int_or_none(max_tokens)
    normalised_finish = str(finish_reason or "").strip().lower()
    truncated = normalised_finish in {"length", "max_tokens", "max_token", "token_limit"}
    value = dict(result)
    value.update(
        {
            "request_max_tokens": request_max_tokens,
            "request_max_tokens_omitted": request_max_tokens is None,
            "finish_reason": str(finish_reason or ""),
            "truncated": truncated,
            "complete": bool(value.get("ok")) and not truncated,
        }
    )
    return value


def _positive_int_or_none(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _provider_request_url(cfg: dict[str, Any]) -> str:
    provider = _provider_name(str(cfg.get("provider") or ""))
    if provider == "fixture":
        return "local://fixture"
    if provider == "gemini":
        return _gemini_endpoint(cfg)
    return _openai_compatible_chat_completions_url(cfg)


def _provider_endpoint_kind(cfg: dict[str, Any]) -> str:
    provider = _provider_name(str(cfg.get("provider") or ""))
    if provider == "gemini":
        return "gemini_generate_content"
    if provider == "fixture":
        return "local_fixture"
    return "openai_chat_completions"


def redact_url_secrets(value: str) -> str:
    if not value:
        return ""
    parsed = urllib.parse.urlsplit(value)
    if not parsed.query:
        return value
    safe_pairs: list[tuple[str, str]] = []
    secret_keys = {"key", "api_key", "apikey", "token", "access_token", "authorization"}
    for key, val in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
        safe_pairs.append((key, "<redacted>" if key.lower() in secret_keys else val))
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(safe_pairs), parsed.fragment))
