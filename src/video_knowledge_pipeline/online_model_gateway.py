from __future__ import annotations

import json
import base64
import mimetypes
import os
import uuid
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .models import now_iso
from .model_defaults import gemini_omits_legacy_sampling_parameters
from .storage import write_json
from .text_llm_gateway import call_openai_compatible_text, openai_compatible_chat_completions_url, resolve_text_provider_config, _effective_response_format
from .vision_api import call_vision_model, provider_requires_api_key, redact_url_secrets, resolve_provider_config
from .gemini_video_api import call_gemini_video

from .model_api_settings import resolve_model_api_provider_config
from .model_runtime_client import model_runtime_request
from .dashscope_filetrans_adapter import call_dashscope_filetrans_asr
SCHEMA = "video_knowledge_pipeline.online_model_api.v1"

MODEL_TYPES = (
    "asr",
    "ocr",
    "document_visual",
    "semantic_frame",
    "temporal_sequence",
    "video_segment",
    "text_llm",
    "summary_rewrite",
    "transcript_correction",
)

VISION_MODEL_TYPES = {"ocr", "document_visual", "semantic_frame", "temporal_sequence"}
TEXT_MODEL_TYPES = {"text_llm", "summary_rewrite", "transcript_correction"}
ASR_MODEL_TYPES = {"asr"}

DEFAULT_PROMPTS = {
    "ocr": "Extract all visible screen text. Preserve headings, lists, numbers, tables, code, formulas, and uncertain text. Return JSON with text_blocks, tables, formulas, code_blocks, confidence, evidence_frame_paths, and uncertainties.",
    "document_visual": "Parse this screenshot as a document/courseware page. Return Markdown-ready structure: title, headings, body text, tables, formulas, code, layout_notes, keep_image_reason, confidence, and evidence_frame_paths.",
    "semantic_frame": "Describe the non-text visual information in this frame. Include objects, actions, interface state, spatial relations, instructor focus, non_text_information, confidence, keep_image_reason, and evidence_frame_paths. Return JSON.",
    "temporal_sequence": "These frames are ordered by time. Describe event_sequence, state_changes, operation_steps, causal_links, possible_missing_points, confidence, and evidence_frame_paths. Return JSON.",
    "video_segment": "These frames represent a short video segment. Explain what changes over time, what is demonstrated, what screen state matters, what may be missed by OCR/ASR, and evidence_frame_paths. Return JSON.",
    "text_llm": "Answer using the provided text. Return concise Markdown unless JSON is requested.",
    "summary_rewrite": "Rewrite the provided video evidence into a polished hierarchical Chinese smart summary. Preserve timestamps, uncertainty boundaries, and review notes. Do not invent visual facts. Return Markdown.",
    "transcript_correction": "Review the transcript against evidence. Identify likely ASR/subtitle errors, proposed corrections, confidence, rationale, and evidence references. Return JSON.",
}


def online_model_api_call(
    model_type: str,
    *,
    provider_config: dict[str, Any] | None = None,
    prompt: str = "",
    input_text: str = "",
    messages: list[dict[str, Any]] | None = None,
    image_paths: list[str] | None = None,
    audio_path: str = "",
    video_path: str = "",
    execute: bool = False,
    temperature: float = 0,
    response_format: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    max_retries: int | None = None,
    output_dir: str | Path | None = None,
    allowed_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    kind = _normalise_model_type(model_type)
    provider_config = resolve_model_api_provider_config(kind, provider_config)
    images = [str(Path(path).expanduser()) for path in image_paths or [] if str(path).strip()]
    audio = str(Path(audio_path).expanduser()) if str(audio_path).strip() else ""
    video = str(Path(video_path).expanduser()) if str(video_path).strip() else ""
    prompt_text = str(prompt or DEFAULT_PROMPTS.get(kind) or "").strip()
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "model_type": kind,
        "execute": bool(execute),
        "status": "planned",
        "ok": True,
        "request_plan": _request_plan(kind, provider_config=provider_config, prompt=prompt_text, input_text=input_text, messages=messages, image_paths=images, audio_path=audio, video_path=video),
        "adapter_reuse": _adapter_reuse_plan(provider_config),
        "operator_boundary": {
            "default_preview_only": True,
            "execute_required_for_network_call": True,
            "provider_config_runtime_or_local_profile": True,
            "local_profile_secrets": "windows_dpapi",
            "secrets_redacted": True,
        },
        "updated_at": now_iso(),
    }
    if kind == "video_segment":
        if not video:
            result.update({"ok": False, "status": "missing_video", "error": "video_path required for video_segment"})
            return _write_result(result, output_dir=output_dir, write=write)
        if not execute:
            result["next_action"] = "rerun with --execute after video-specific route consent and data-boundary approval are configured"
            return _write_result(result, output_dir=output_dir, write=write)
        direct_config = dict(provider_config)
        direct_config.pop("adapter_backend", None)
        call = call_gemini_video(
            provider_config=resolve_provider_config(direct_config),
            prompt=_vision_prompt(kind, prompt_text, input_text),
            video_path=video,
        )
        result.update(_call_result(call))
        return _write_result(result, output_dir=output_dir, write=write)
    if _adapter_backend(provider_config) == "proxy":
        if not execute:
            result["next_action"] = "rerun with --execute after route consent and data-boundary approval are configured"
            return _write_result(result, output_dir=output_dir, write=write)
        runtime = model_runtime_request(
            kind,
            execution_location=str(provider_config.get("execution_location") or ""),
            route_id=str(provider_config.get("route_id") or ""),
            route_revision=str(provider_config.get("route_revision") or ""),
            text=str(input_text or ""),
            image_paths=[] if kind == "ocr" else images,
            audio_path=audio,
            document_path=images[0] if kind == "ocr" and len(images) == 1 else "",
            prompt=prompt_text,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
            max_tokens=max_tokens,
            max_retries=max_retries,
            consent_id=str(provider_config.get("consent_id") or ""),
            execute=execute,
            allowed_roots=allowed_roots,
        )
        result.update(
            {
                "ok": bool(runtime.get("ok")),
                "status": str(runtime.get("status") or "failed"),
                "content": runtime.get("content", ""),
                "raw_response": runtime.get("raw_output") if kind == "asr" else None,
                "error": str(runtime.get("error") or ""),
                "runtime_result": runtime,
                "fallback_from": "",
                "secrets_redacted": True,
            }
        )
        return _write_result(result, output_dir=output_dir, write=write)
    if not execute:
        result["next_action"] = "rerun with --execute after credentials and data-boundary approval are configured"
        return _write_result(result, output_dir=output_dir, write=write)
    if kind in VISION_MODEL_TYPES:
        if not images:
            result.update({"ok": False, "status": "missing_images", "error": "image_paths required for this model_type"})
            return _write_result(result, output_dir=output_dir, write=write)
        cfg = resolve_provider_config(provider_config)
        if _should_use_litellm(cfg):
            call = call_litellm_chat(
                provider_config=cfg,
                messages=_vision_messages(kind, prompt_text, input_text, images),
                temperature=0,
                max_tokens=max_tokens,
            )
            if _should_fallback_from_litellm(cfg, call):
                call = _with_litellm_fallback_metadata(
                    _call_vision_with_optional_tokens(
                        provider_config=cfg,
                        prompt=_vision_prompt(kind, prompt_text, input_text),
                        image_paths=images,
                        max_tokens=max_tokens,
                    ),
                    litellm_call=call,
                )
        else:
            call = _call_vision_with_optional_tokens(
                provider_config=cfg,
                prompt=_vision_prompt(kind, prompt_text, input_text),
                image_paths=images,
                max_tokens=max_tokens,
            )
        result.update(_call_result(call))
    elif kind in TEXT_MODEL_TYPES:
        cfg = resolve_text_provider_config(provider_config)
        response_format = _effective_response_format(cfg, response_format)
        request_messages = list(messages or [{"role": "user", "content": _text_prompt(kind, prompt_text, input_text)}])
        if _should_use_litellm(cfg):
            call = call_litellm_chat(provider_config=cfg, messages=request_messages, temperature=temperature, response_format=response_format, max_tokens=max_tokens)
            if _should_fallback_from_litellm(cfg, call):
                call = _with_litellm_fallback_metadata(
                    call_openai_compatible_text(provider_config=cfg, messages=request_messages, temperature=temperature, response_format=response_format, max_tokens=max_tokens),
                    litellm_call=call,
                )
        else:
            call = call_openai_compatible_text(provider_config=cfg, messages=request_messages, temperature=temperature, response_format=response_format, max_tokens=max_tokens)
        result.update(_call_result(call))
    elif kind in ASR_MODEL_TYPES:
        if not audio:
            result.update({"ok": False, "status": "missing_audio", "error": "audio_path required for asr"})
            return _write_result(result, output_dir=output_dir, write=write)
        cfg = resolve_asr_provider_config(provider_config)
        if str(cfg.get("provider") or "").strip().lower() == "dashscope_filetrans":
            call = call_dashscope_filetrans_asr(
                provider_config=cfg,
                audio_path=audio,
                prompt=prompt_text,
            )
        elif _should_use_litellm(cfg):
            call = call_litellm_asr(provider_config=cfg, audio_path=audio, prompt=prompt_text)
            if _should_fallback_from_litellm(cfg, call):
                call = _with_litellm_fallback_metadata(
                    call_openai_compatible_asr(provider_config=cfg, audio_path=audio, prompt=prompt_text),
                    litellm_call=call,
                )
        else:
            call = call_openai_compatible_asr(provider_config=cfg, audio_path=audio, prompt=prompt_text)
        result.update(_call_result(call))
    else:  # pragma: no cover - guarded by _normalise_model_type.
        result.update({"ok": False, "status": "unsupported_model_type", "error": kind})
    return _write_result(result, output_dir=output_dir, write=write)


def online_model_api_matrix(*, provider_config: dict[str, Any] | None = None, output_dir: str | Path | None = None, write: bool = True) -> dict[str, Any]:
    rows = []
    for kind in MODEL_TYPES:
        rows.append(_request_plan(kind, provider_config=provider_config, prompt=DEFAULT_PROMPTS.get(kind, ""), input_text="", messages=None, image_paths=[], audio_path="", video_path=""))
    result = {
        "schema": "video_knowledge_pipeline.online_model_api_matrix.v1",
        "model_types": list(MODEL_TYPES),
        "providers": rows,
        "status": "planned",
        "ok": True,
        "secrets_redacted": True,
        "adapter_reuse": _adapter_reuse_plan(provider_config),
        "updated_at": now_iso(),
    }
    return _write_result(result, output_dir=output_dir, write=write, basename="online-model-api-matrix")


def call_litellm_chat(
    *,
    provider_config: dict[str, Any],
    messages: list[dict[str, Any]],
    temperature: float = 0,
    response_format: dict[str, Any] | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    try:
        import litellm  # type: ignore[import-not-found]
    except ImportError:
        return _completion_result({"ok": False, "error": "litellm_not_installed", "content": ""}, max_tokens=max_tokens)
    cfg = dict(provider_config or {})
    kwargs: dict[str, Any] = {
        "model": cfg.get("model"),
        "messages": messages,
    }
    if not (
        str(cfg.get("provider") or "").lower() == "gemini"
        and gemini_omits_legacy_sampling_parameters(cfg.get("model"))
    ):
        kwargs["temperature"] = temperature
    if cfg.get("api_key"):
        kwargs["api_key"] = cfg.get("api_key")
    if cfg.get("base_url"):
        kwargs["api_base"] = cfg.get("base_url")
    if cfg.get("timeout_seconds"):
        kwargs["timeout"] = int(cfg.get("timeout_seconds") or 60)
    if cfg.get("custom_llm_provider"):
        kwargs["custom_llm_provider"] = cfg.get("custom_llm_provider")
    if response_format:
        kwargs["response_format"] = response_format
    if max_tokens:
        kwargs["max_tokens"] = int(max_tokens)
    try:
        response = litellm.completion(**kwargs)
    except Exception as exc:  # pragma: no cover - optional network/provider path.
        return _completion_result({"ok": False, "error": str(exc), "content": ""}, max_tokens=max_tokens)
    payload = _safe_litellm_response(response)
    try:
        choice = payload["choices"][0]
        message = choice.get("message") if isinstance(choice, dict) else {}
        if not isinstance(message, dict):
            message = {}
        content = str(message.get("content") or "")
        finish_reason = str(choice.get("finish_reason") or "") if isinstance(choice, dict) else ""
        reasoning_chars = len(str(message.get("reasoning_content") or ""))
    except (KeyError, IndexError, TypeError):
        content = ""
        finish_reason = ""
        reasoning_chars = 0
    if not content.strip():
        detail = f"empty_content; finish_reason={finish_reason or 'unknown'}"
        if reasoning_chars:
            detail = f"empty_content_reasoning_only; finish_reason={finish_reason or 'unknown'}; reasoning_chars={reasoning_chars}"
        return _completion_result(
            {"ok": False, "error": detail, "content": "", "raw_response": payload},
            max_tokens=max_tokens,
            finish_reason=finish_reason,
        )
    return _completion_result(
        {"ok": True, "error": "", "content": content, "raw_response": payload},
        max_tokens=max_tokens,
        finish_reason=finish_reason,
    )


def call_litellm_asr(*, provider_config: dict[str, Any], audio_path: str, prompt: str = "") -> dict[str, Any]:
    try:
        import litellm  # type: ignore[import-not-found]
    except ImportError:
        return {"ok": False, "error": "litellm_not_installed", "content": ""}
    if not hasattr(litellm, "transcription"):
        return {"ok": False, "error": "litellm_transcription_not_available", "content": ""}
    path = Path(audio_path).expanduser().resolve()
    if not path.exists():
        return {"ok": False, "error": f"audio_not_found: {path}", "content": ""}
    cfg = dict(provider_config or {})
    kwargs: dict[str, Any] = {"model": cfg.get("model") or "gpt-4o-transcribe"}
    if cfg.get("api_key"):
        kwargs["api_key"] = cfg.get("api_key")
    if cfg.get("base_url"):
        kwargs["api_base"] = cfg.get("base_url")
    if cfg.get("language"):
        kwargs["language"] = cfg.get("language")
    if prompt:
        kwargs["prompt"] = prompt
    try:
        with path.open("rb") as audio_file:
            response = litellm.transcription(file=audio_file, **kwargs)
    except Exception as exc:  # pragma: no cover - optional network/provider path.
        return {"ok": False, "error": str(exc), "content": ""}
    payload = _safe_litellm_response(response)
    content = payload.get("text") if isinstance(payload, dict) else str(payload)
    return {"ok": True, "error": "", "content": str(content or ""), "raw_response": payload}


def call_openai_compatible_asr(*, provider_config: dict[str, Any], audio_path: str, prompt: str = "") -> dict[str, Any]:
    cfg = resolve_asr_provider_config(provider_config)
    if provider_requires_api_key(cfg) and not cfg.get("api_key"):
        return {"ok": False, "error": "missing_api_key", "content": ""}
    path = Path(audio_path).expanduser().resolve()
    if not path.exists():
        return {"ok": False, "error": f"audio_not_found: {path}", "content": ""}
    fields: dict[str, str] = {"model": str(cfg.get("model") or "gpt-4o-transcribe"), "response_format": str(cfg.get("response_format") or "verbose_json")}
    language = str(cfg.get("language") or os.environ.get("LECTURE_ASR_API_LANGUAGE") or "").strip()
    if language:
        fields["language"] = language
    if prompt:
        fields["prompt"] = prompt
    data, content_type = _multipart_form_data(fields, file_field="file", file_path=path)
    headers = {"Content-Type": content_type}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg.get('api_key')}"
    request = urllib.request.Request(asr_transcriptions_url(cfg), data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=int(cfg.get("timeout_seconds") or 120)) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - network is optional and gated.
        return {"ok": False, "error": str(exc), "content": ""}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"text": raw}
    content = payload.get("text") if isinstance(payload, dict) else raw
    return {"ok": True, "error": "", "content": str(content or ""), "raw_response": payload}


def resolve_asr_provider_config(provider_config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(provider_config or {})
    proxy_backend = str(cfg.get("adapter_backend") or "").strip().lower() == "proxy"
    provider = str(cfg.get("provider") or os.environ.get("LECTURE_ASR_API_PROVIDER") or "openai_compatible_asr").strip()
    api_key = "" if proxy_backend else (
        cfg.get("api_key")
        or os.environ.get("LECTURE_ASR_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("LLM_API_KEY")
    )
    base_url = str(cfg.get("base_url") or os.environ.get("LECTURE_ASR_API_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")
    model = str(cfg.get("model") or os.environ.get("LECTURE_ASR_API_MODEL") or "gpt-4o-transcribe")
    return {
        "provider": provider,
        "base_url": base_url.rstrip("/"),
        "api_key": _normalise_api_key(api_key),
        "model": model,
        "timeout_seconds": int(cfg.get("timeout_seconds") or os.environ.get("LECTURE_ASR_API_TIMEOUT_SECONDS") or 120),
        "response_format": str(cfg.get("response_format") or os.environ.get("LECTURE_ASR_API_RESPONSE_FORMAT") or "verbose_json"),
        "language": str(cfg.get("language") or os.environ.get("LECTURE_ASR_API_LANGUAGE") or ""),
        "adapter_backend": str(cfg.get("adapter_backend") or ""),
        "location": str(cfg.get("location") or ""),
        "execution_location": str(cfg.get("execution_location") or ""),
        "route_id": str(cfg.get("route_id") or ""),
        "route_revision": str(cfg.get("route_revision") or ""),
        "virtual_model": str(cfg.get("virtual_model") or ""),
        "profile_id": str(cfg.get("profile_id") or ""),
        "consent_id": str(cfg.get("consent_id") or ""),
        "capabilities": list(cfg.get("capabilities") or []),
        "provider_options": dict(cfg.get("provider_options") or {}),
    }

def asr_transcriptions_url(provider_config: dict[str, Any]) -> str:
    base = str(provider_config.get("base_url") or "https://api.openai.com/v1").rstrip("/")
    if base.endswith("/audio/transcriptions"):
        return base
    parsed = urllib.parse.urlparse(base)
    path = parsed.path.rstrip("/")
    if path:
        return f"{base}/audio/transcriptions"
    if parsed.scheme and parsed.netloc:
        return f"{base}/v1/audio/transcriptions"
    return f"{base}/audio/transcriptions"


def _request_plan(
    kind: str,
    *,
    provider_config: dict[str, Any] | None,
    prompt: str,
    input_text: str,
    messages: list[dict[str, Any]] | None,
    image_paths: list[str],
    audio_path: str,
    video_path: str,
) -> dict[str, Any]:
    if kind == "video_segment":
        cfg = resolve_provider_config({**dict(provider_config or {}), "adapter_backend": ""})
        return {
            "model_type": kind,
            "interface": "gemini_files_api",
            "adapter_backend": "gemini_files_api",
            "provider": _public_vision_config(cfg),
            "video_path": video_path,
            "video_exists": Path(video_path).expanduser().is_file() if video_path else False,
            "prompt_chars": len(prompt or ""),
            "provider_managed_file_upload": True,
        }
    if kind in VISION_MODEL_TYPES:
        cfg = resolve_provider_config(provider_config)
        return {
            "model_type": kind,
            "interface": "vision_chat_completions_or_gemini",
            "adapter_backend": _adapter_backend(cfg),
            "provider": _public_vision_config(cfg),
            "image_count": len(image_paths),
            "prompt_chars": len(prompt or ""),
            "input_text_chars": len(input_text or ""),
        }
    if kind in TEXT_MODEL_TYPES:
        cfg = resolve_text_provider_config(provider_config)
        return {
            "model_type": kind,
            "interface": "openai_chat_completions",
            "adapter_backend": _adapter_backend(cfg),
            "provider": _public_text_config(cfg),
            "url": redact_url_secrets(openai_compatible_chat_completions_url(cfg)),
            "message_count": len(messages or []) or 1,
            "prompt_chars": len(prompt or ""),
            "input_text_chars": len(input_text or ""),
        }
    cfg = resolve_asr_provider_config(provider_config)
    is_dashscope_filetrans = (
        str(cfg.get("provider") or "").strip().lower() == "dashscope_filetrans"
    )
    return {
        "model_type": kind,
        "interface": (
            "dashscope_async_filetrans"
            if is_dashscope_filetrans
            else "openai_audio_transcriptions"
        ),
        "adapter_backend": (
            "moys_asr_workflow_fixed_cli"
            if is_dashscope_filetrans
            else _adapter_backend(cfg)
        ),
        "provider": _public_asr_config(cfg),
        "url": (
            redact_url_secrets(str(cfg.get("base_url") or ""))
            if is_dashscope_filetrans
            else redact_url_secrets(asr_transcriptions_url(cfg))
        ),
        "audio_path": audio_path,
        "audio_exists": Path(audio_path).expanduser().exists() if audio_path else False,
        "prompt_chars": len(prompt or ""),
    }


def _adapter_backend(provider_config: dict[str, Any] | None) -> str:
    cfg = dict(provider_config or {})
    backend = str(cfg.get("adapter_backend") or cfg.get("backend") or os.environ.get("VKP_ONLINE_MODEL_BACKEND") or "legacy").strip().lower()
    if backend in {"proxy", "litellm_proxy", "gateway"}:
        return "proxy"
    if backend == "legacy":
        return "legacy"
    if backend in {"litellm", "lite_llm"}:
        return "litellm"
    if backend in {"urllib", "builtin", "native"}:
        return "builtin"
    if backend in {"auto", "auto_litellm_then_builtin"}:
        return "auto_litellm_then_builtin"
    return "legacy"


def _adapter_reuse_plan(provider_config: dict[str, Any] | None) -> dict[str, Any]:
    backend = _adapter_backend(provider_config)
    return {
        "preferred_open_source_backend": "LiteLLM Proxy",
        "backend": backend,
        "fallback_backend": (
            "built_in_openai_compatible_urllib"
            if backend == "auto_litellm_then_builtin"
            else "none"
        ),
        "legacy_adapter": (
            "built_in_openai_compatible_urllib"
            if backend in {"legacy", "builtin"}
            else ""
        ),
        "why": "Provider dialects, OpenAI-compatible chat/vision, transcription, retries/proxy hooks, and future routing should be delegated to a mature adapter when installed.",
        "install_optional_extra": "pip install -e .[online]",
    }


def _should_use_litellm(provider_config: dict[str, Any] | None) -> bool:
    backend = _adapter_backend(provider_config)
    if backend in {"builtin", "legacy", "proxy"}:
        return False
    if backend == "litellm":
        return True
    provider = str((provider_config or {}).get("provider") or "").strip().lower()
    if provider in {"local_qwen_vl", "local_vlm"}:
        # Local OpenAI-compatible servers already expose the exact contract VKP
        # needs. Auto-detecting them through embedded LiteLLM can discard final
        # content for otherwise valid Qwen/LM Studio responses. Explicit
        # adapter_backend=litellm or proxy still opts into those routes.
        return False
    try:
        import litellm  # noqa: F401  # type: ignore[import-not-found]
    except ImportError:
        return False
    return True

def _should_fallback_from_litellm(provider_config: dict[str, Any], call: dict[str, Any]) -> bool:
    return _adapter_backend(provider_config) == "auto_litellm_then_builtin" and not bool(call.get("ok"))


def _with_litellm_fallback_metadata(fallback_call: dict[str, Any], *, litellm_call: dict[str, Any]) -> dict[str, Any]:
    result = dict(fallback_call or {})
    result["fallback_from"] = "litellm"
    result["litellm_error"] = str(litellm_call.get("error") or "")
    return result

def _vision_messages(kind: str, prompt: str, input_text: str, image_paths: list[str]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": _vision_prompt(kind, prompt, input_text)}]
    for image_path in image_paths:
        path = Path(image_path).expanduser().resolve()
        mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
    return [{"role": "user", "content": content}]


def _safe_litellm_response(response: Any) -> Any:
    for attr in ("model_dump", "dict"):
        method = getattr(response, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:
                pass
    try:
        return json.loads(json.dumps(response, default=str))
    except Exception:
        return str(response)


def _call_result(call: dict[str, Any]) -> dict[str, Any]:
    ok = bool(call.get("ok"))
    return {
        "ok": ok,
        "status": str(
            call.get("status") or ("ok" if ok else call.get("error") or "failed")
        ),
        "content": call.get("content", ""),
        "error": call.get("error", ""),
        "error_class": call.get("error_class", ""),
        "http_status": call.get("http_status"),
        "raw_response": call.get("raw_response"),
        "fallback_from": call.get("fallback_from", ""),
        "litellm_error": call.get("litellm_error", ""),
        "request_max_tokens": call.get("request_max_tokens"),
        "request_max_tokens_omitted": bool(call.get("request_max_tokens_omitted", True)),
        "finish_reason": str(call.get("finish_reason") or ""),
        "truncated": bool(call.get("truncated")),
        "complete": bool(call.get("complete", ok and not call.get("truncated"))),
        "secrets_redacted": True,
        "remote_requests_made": bool(call.get("remote_requests_made", ok)),
    }


def _call_vision_with_optional_tokens(
    *,
    provider_config: dict[str, Any],
    prompt: str,
    image_paths: list[str],
    max_tokens: int | None,
) -> dict[str, Any]:
    if max_tokens is None:
        return call_vision_model(provider_config=provider_config, prompt=prompt, image_paths=image_paths)
    return call_vision_model(
        provider_config=provider_config,
        prompt=prompt,
        image_paths=image_paths,
        max_tokens=max_tokens,
    )


def _completion_result(
    result: dict[str, Any],
    *,
    max_tokens: int | None,
    finish_reason: str = "",
) -> dict[str, Any]:
    request_max_tokens = int(max_tokens) if max_tokens is not None and int(max_tokens) > 0 else None
    normalised_finish = str(finish_reason or "").strip().lower()
    truncated = normalised_finish in {"length", "max_tokens", "max_token", "token_limit"}
    return {
        **result,
        "request_max_tokens": request_max_tokens,
        "request_max_tokens_omitted": request_max_tokens is None,
        "finish_reason": str(finish_reason or ""),
        "truncated": truncated,
        "complete": bool(result.get("ok")) and not truncated,
    }


def _write_result(result: dict[str, Any], *, output_dir: str | Path | None, write: bool, basename: str = "online-model-api") -> dict[str, Any]:
    if not write or not output_dir:
        return result
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{basename}.json"
    md_path = out / f"{basename}.md"
    write_json(json_path, result)
    md_path.write_text(_render_markdown(result), encoding="utf-8")
    result["artifacts"] = {"json": str(json_path), "markdown": str(md_path)}
    return result


def _render_markdown(result: dict[str, Any]) -> str:
    lines = ["# Online Model API", "", f"- Schema: `{result.get('schema')}`", f"- Status: `{result.get('status')}`", f"- OK: `{result.get('ok')}`", f"- Model type: `{result.get('model_type', '')}`", "", "## Request Plan", "", "```json", json.dumps(result.get("request_plan") or result.get("providers") or {}, ensure_ascii=False, indent=2), "```"]
    if result.get("content"):
        lines.extend(["", "## Content", "", str(result.get("content"))[:4000]])
    if result.get("error"):
        lines.extend(["", "## Error", "", str(result.get("error"))])
    return "\n".join(lines).rstrip() + "\n"


def _vision_prompt(kind: str, prompt: str, input_text: str) -> str:
    base = prompt or DEFAULT_PROMPTS.get(kind, "")
    if input_text:
        return f"{base}\n\nContext text/evidence:\n{input_text}"
    return base


def _text_prompt(kind: str, prompt: str, input_text: str) -> str:
    base = prompt or DEFAULT_PROMPTS.get(kind, "")
    return f"{base}\n\nInput:\n{input_text}" if input_text else base


def _normalise_model_type(value: str) -> str:
    key = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "speech": "asr",
        "speech_asr": "asr",
        "audio_transcription": "asr",
        "screen_text": "ocr",
        "layout": "document_visual",
        "doc_visual": "document_visual",
        "multimodal": "semantic_frame",
        "vision": "semantic_frame",
        "temporal": "temporal_sequence",
        "video": "video_segment",
        "llm": "text_llm",
        "text": "text_llm",
        "summary": "summary_rewrite",
    }
    key = aliases.get(key, key)
    if key not in MODEL_TYPES:
        raise ValueError(f"unsupported model_type={value!r}; expected one of {', '.join(MODEL_TYPES)}")
    return key


def _public_vision_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {"provider": cfg.get("provider", ""), "model": cfg.get("model", ""), "base_url": redact_url_secrets(str(cfg.get("base_url") or "")), "api_key_required": provider_requires_api_key(cfg), "api_key_configured": bool(cfg.get("api_key"))}


def _public_text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {"provider": cfg.get("provider", ""), "model": cfg.get("model", ""), "base_url": redact_url_secrets(str(cfg.get("base_url") or "")), "api_key_required": provider_requires_api_key(cfg), "api_key_configured": bool(cfg.get("api_key")), "interface": "openai_chat_completions"}


def _public_asr_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {"provider": cfg.get("provider", ""), "model": cfg.get("model", ""), "base_url": redact_url_secrets(str(cfg.get("base_url") or "")), "api_key_required": provider_requires_api_key(cfg), "api_key_configured": bool(cfg.get("api_key")), "response_format": cfg.get("response_format", "verbose_json"), "language": cfg.get("language", "")}


def _multipart_form_data(fields: dict[str, str], *, file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----vkp-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")
    mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    chunks.append(f"--{boundary}\r\n".encode("utf-8"))
    chunks.append(f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'.encode("utf-8"))
    chunks.append(f"Content-Type: {mime}\r\n\r\n".encode("utf-8"))
    chunks.append(file_path.read_bytes())
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _normalise_api_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.startswith("${") or text.startswith("<"):
        return ""
    return text
