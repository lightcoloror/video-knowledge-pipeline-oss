from __future__ import annotations

import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from video_knowledge_pipeline import model_runtime_client
from video_knowledge_pipeline.model_api_settings import resolve_model_api_route, upsert_model_api_profile
from video_knowledge_pipeline.model_runtime_client import model_runtime_request


class _FakeGatewayHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, object]] = []
    chat_content = "gateway result"

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        type(self).requests.append(
            {
                "path": self.path,
                "content_type": self.headers.get("Content-Type", ""),
                "authorization": self.headers.get("Authorization", ""),
                "accept": self.headers.get("Accept", ""),
                "timeout": self.headers.get("x-litellm-timeout", ""),
                "num_retries": self.headers.get("x-litellm-num-retries", ""),
                "body": body,
            }
        )
        request_payload = {}
        if self.headers.get("Content-Type") == "application/json":
            request_payload = json.loads(body)
        if self.path == "/v1/audio/transcriptions":
            payload = {"text": "transcribed", "model": "remote-model", "usage": {"seconds": 1}}
        elif self.path == "/v1/ocr":
            payload = {
                "object": "ocr",
                "model": "remote-model",
                "pages": [{"index": 0, "markdown": "# OCR text", "dimensions": {"width": 10, "height": 10}}],
                "usage_info": {"pages_processed": 1},
            }
        elif request_payload.get("stream") is True:
            events = [
                {
                    "id": "chat-stream-test",
                    "object": "chat.completion.chunk",
                    "model": "remote-model",
                    "choices": [
                        {"index": 0, "delta": {"reasoning_content": "transient reasoning"}}
                    ],
                },
                {
                    "id": "chat-stream-test",
                    "object": "chat.completion.chunk",
                    "model": "remote-model",
                    "choices": [
                        {"index": 0, "delta": {"content": "gateway stream "}}
                    ],
                },
                {
                    "id": "chat-stream-test",
                    "object": "chat.completion.chunk",
                    "model": "remote-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "result"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 3,
                        "total_tokens": 7,
                    },
                },
            ]
            encoded = b"".join(
                f"data: {json.dumps(event)}\n\n".encode("utf-8") for event in events
            ) + b"data: [DONE]\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("x-litellm-response-cost", "0.002")
            self.end_headers()
            self.wfile.write(encoded)
            self.wfile.flush()
            return
        else:
            payload = {
                "id": "chat-test",
                "object": "chat.completion",
                "model": "remote-model",
                "choices": [{"message": {"content": type(self).chat_content}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("x-litellm-response-cost", "0.001")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


def _gateway(tmp_path: Path) -> tuple[ThreadingHTTPServer, threading.Thread, Path]:
    _FakeGatewayHandler.requests = []
    _FakeGatewayHandler.chat_content = "gateway result"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeGatewayHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = tmp_path / "model-gateway.json"
    config.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.model_gateway_config.v1",
                "host": "127.0.0.1",
                "port": server.server_port,
                "telemetry": False,
                "config_path": str(tmp_path / "litellm.yaml"),
                "pid_path": str(tmp_path / "gateway.pid"),
                "log_path": str(tmp_path / "gateway.log"),
            }
        ),
        encoding="utf-8",
    )
    return server, thread, config


def _settings(tmp_path: Path) -> tuple[Path, Path]:
    settings = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    upsert_model_api_profile(
        {
            "id": "remote-main",
            "name": "Remote main",
            "provider": "openai_compatible",
            "adapter_backend": "proxy",
            "base_url": "https://provider.example/v1",
            "model": "remote-model",
            "location": "remote",
            "capabilities": ["text", "vision", "asr"],
        },
        tasks=["text_llm", "summary_rewrite", "semantic_frame", "temporal_sequence", "asr"],
        settings_path=settings,
        secrets_path=secrets,
    )
    upsert_model_api_profile(
        {
            "id": "remote-ocr",
            "name": "Remote OCR",
            "provider": "mistral_compatible_ocr",
            "adapter_backend": "proxy",
            "base_url": "https://ocr-adapter.example/v1",
            "model": "mistral-ocr-latest",
            "location": "remote",
            "capabilities": ["ocr"],
        },
        tasks=["ocr"],
        settings_path=settings,
        secrets_path=secrets,
    )
    return settings, secrets


def _consented_runtime_request(task: str, *, settings_path: Path, **kwargs: object) -> dict[str, object]:
    route = resolve_model_api_route(task, execution_location="remote", settings_path=settings_path)
    consent_id = "test-consent"
    with model_runtime_client._authorise_consented_remote_runtime(
        consent_id=consent_id,
        route_revision=str(route["route_revision"]),
        max_calls=1,
    ):
        return model_runtime_request(
            task,
            settings_path=settings_path,
            route_revision=str(route["route_revision"]),
            consent_id=consent_id,
            **kwargs,
        )


def test_new_gemini_proxy_request_omits_deprecated_temperature() -> None:
    body, content_type = model_runtime_client._request_body(
        "text_llm",
        route={
            "virtual_model": "vkp-remote-text-gemini-test",
            "deployments": [
                {
                    "provider": "gemini",
                    "model": "gemini-3.6-flash",
                    "provider_options": {},
                }
            ],
        },
        inputs={"text": "test", "images": [], "audio": "", "document": ""},
        prompt="",
        messages=None,
        temperature=0,
        response_format=None,
        runtime_options={},
    )

    assert content_type == "application/json"
    payload = json.loads(body)
    assert payload["model"] == "vkp-remote-text-gemini-test"
    assert "temperature" not in payload


def test_groq_asr_request_omits_unsupported_timestamp_granularities(tmp_path: Path) -> None:
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"audio")

    body, content_type = model_runtime_client._request_body(
        "asr",
        route={
            "virtual_model": "vkp-remote-asr-groq-test",
            "deployments": [
                {
                    "provider": "groq_asr",
                    "litellm_provider": "groq",
                    "model": "whisper-large-v3-turbo",
                }
            ],
        },
        inputs={"text": "", "images": [], "audio": audio, "document": ""},
        prompt="hot words",
        messages=None,
        temperature=0,
        response_format=None,
        runtime_options={},
    )

    assert content_type.startswith("multipart/form-data;")
    assert b'name="response_format"' in body
    assert b"verbose_json" in body
    assert b'timestamp_granularities[]' not in body
    assert b'name="prompt"' in body

def test_groq_openai_transport_requests_route_locked_word_timestamps(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"audio")

    body, content_type = model_runtime_client._request_body(
        "asr",
        route={
            "virtual_model": "vkp-remote-asr-groq-word-test",
            "deployments": [
                {
                    "provider": "groq_asr",
                    "litellm_provider": "openai",
                    "model": "whisper-large-v3-turbo",
                }
            ],
        },
        inputs={"text": "", "images": [], "audio": audio, "document": ""},
        prompt="hot words",
        messages=None,
        temperature=0,
        response_format=None,
        runtime_options={"asr_timestamp_granularity": "word"},
    )

    assert content_type.startswith("multipart/form-data;")
    assert b'name="timestamp_granularities[]"' in body
    assert b"word" in body

def test_runtime_preview_exposes_route_without_network(tmp_path: Path) -> None:
    settings, secrets = _settings(tmp_path)
    gateway = tmp_path / "gateway.json"
    gateway.write_text(
        json.dumps({"schema": "video_knowledge_pipeline.model_gateway_config.v1", "host": "127.0.0.1", "port": 8776}),
        encoding="utf-8",
    )

    result = model_runtime_request(
        "summary_rewrite",
        execution_location="remote",
        text="source",
        execute=False,
        settings_path=settings,
        secrets_path=secrets,
        gateway_config_path=gateway,
        allowed_roots=[tmp_path],
    )

    assert result["ok"] is True
    assert result["status"] == "planned"
    assert result["route_revision"]
    assert result["virtual_model"].startswith("vkp-remote-text-")
    assert result["gateway"]["transport"] == "loopback_http"


def test_runtime_fake_proxy_covers_text_multi_image_asr_and_ocr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings, secrets = _settings(tmp_path)
    server, thread, gateway = _gateway(tmp_path)
    monkeypatch.setattr(model_runtime_client, "_gateway_api_key", lambda path: "local-gateway-key")
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.jpg"
    audio = tmp_path / "speech.wav"
    document = tmp_path / "slide.pdf"
    image_a.write_bytes(b"png-one")
    image_b.write_bytes(b"jpeg-two")
    audio.write_bytes(b"wave-data")
    document.write_bytes(b"%PDF-fixture")
    try:
        text_result = _consented_runtime_request(
            "summary_rewrite",
            execution_location="remote",
            text="source text",
            prompt="summarise",
            execute=True,
            settings_path=settings,
            secrets_path=secrets,
            gateway_config_path=gateway,
            allowed_roots=[tmp_path],
        )
        vision_result = _consented_runtime_request(
            "temporal_sequence",
            execution_location="remote",
            image_paths=[image_a, image_b],
            prompt="sequence",
            execute=True,
            settings_path=settings,
            secrets_path=secrets,
            gateway_config_path=gateway,
            allowed_roots=[tmp_path],
        )
        asr_result = _consented_runtime_request(
            "asr",
            execution_location="remote",
            audio_path=audio,
            prompt="hot words",
            execute=True,
            settings_path=settings,
            secrets_path=secrets,
            gateway_config_path=gateway,
            allowed_roots=[tmp_path],
        )
        ocr_result = _consented_runtime_request(
            "ocr",
            execution_location="remote",
            document_path=document,
            execute=True,
            settings_path=settings,
            secrets_path=secrets,
            gateway_config_path=gateway,
            allowed_roots=[tmp_path],
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert text_result["content"] == "gateway result"
    assert text_result["usage"]["total_tokens"] == 5
    assert text_result["estimated_cost"] == 0.001
    assert vision_result["ok"] is True
    assert len(vision_result["evidence"]) == 2
    assert asr_result["content"] == "transcribed"
    assert asr_result["raw_output"]["text"] == "transcribed"
    assert ocr_result["content"]["pages"][0]["markdown"] == "# OCR text"
    assert [row["path"] for row in _FakeGatewayHandler.requests] == [
        "/v1/chat/completions",
        "/v1/chat/completions",
        "/v1/audio/transcriptions",
        "/v1/ocr",
    ]
    assert {row["timeout"] for row in _FakeGatewayHandler.requests} == {"120"}
    assert {row["num_retries"] for row in _FakeGatewayHandler.requests} == {"1"}
    asr_body = _FakeGatewayHandler.requests[2]["body"]
    assert b'name="response_format"' in asr_body
    assert b"verbose_json" in asr_body
    assert b'name="timestamp_granularities[]"' in asr_body
    assert b"segment" in asr_body
    assert b'name="prompt"' in asr_body
    assert b"hot words" in asr_body
    vision_payload = json.loads(_FakeGatewayHandler.requests[1]["body"])
    image_urls = [part["image_url"]["url"] for part in vision_payload["messages"][0]["content"][1:]]
    assert image_urls[0].startswith("data:image/png;base64,")
    assert image_urls[1].startswith("data:image/jpeg;base64,")
    ocr_payload = json.loads(_FakeGatewayHandler.requests[3]["body"])
    assert ocr_payload["document"]["document_url"].startswith("data:application/pdf;base64,")
    serialized_results = json.dumps([text_result, vision_result, asr_result, ocr_result], ensure_ascii=False)
    assert ";base64," not in serialized_results
    assert "local-gateway-key" not in serialized_results


def test_runtime_can_lower_but_not_raise_route_retry_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings, secrets = _settings(tmp_path)
    server, thread, gateway = _gateway(tmp_path)
    monkeypatch.setattr(
        model_runtime_client,
        "_gateway_api_key",
        lambda path: "local-gateway-key",
    )
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"wave")
    try:
        result = _consented_runtime_request(
            "asr",
            execution_location="remote",
            audio_path=audio,
            prompt="明亚保险 Excel",
            max_retries=0,
            execute=True,
            settings_path=settings,
            secrets_path=secrets,
            gateway_config_path=gateway,
            allowed_roots=[tmp_path],
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result["ok"] is True
    assert result["request_options"]["max_retries"] == 0
    assert _FakeGatewayHandler.requests[0]["num_retries"] == "0"

    route = resolve_model_api_route(
        "asr",
        execution_location="remote",
        settings_path=settings,
    )
    raised = model_runtime_request(
        "asr",
        execution_location="remote",
        route_revision=str(route["route_revision"]),
        audio_path=audio,
        max_retries=2,
        execute=False,
        settings_path=settings,
        secrets_path=secrets,
        gateway_config_path=gateway,
        allowed_roots=[tmp_path],
    )
    assert raised["status"] == "invalid_runtime_request"
    assert "cannot exceed" in raised["error"]




def test_runtime_streams_siliconflow_with_route_locked_limits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    upsert_model_api_profile(
        {
            "id": "siliconflow-qwen",
            "name": "SiliconFlow Qwen",
            "provider": "siliconflow",
            "adapter_backend": "proxy",
            "base_url": "https://api.siliconflow.cn/v1",
            "model": "Qwen/Qwen3.5-397B-A17B",
            "location": "remote",
            "capabilities": ["text"],
            "provider_options": {
                "enable_thinking": False,
                "thinking_budget": 256,
                "max_tokens": 256,
                "stream": True,
            },
        },
        tasks=["summary_rewrite"],
        settings_path=settings,
        secrets_path=secrets,
    )
    server, thread, gateway = _gateway(tmp_path)
    monkeypatch.setattr(
        model_runtime_client,
        "_gateway_api_key",
        lambda path: "local-gateway-key",
    )
    try:
        result = _consented_runtime_request(
            "summary_rewrite",
            execution_location="remote",
            text="source text",
            prompt="summarise",
            execute=True,
            settings_path=settings,
            secrets_path=secrets,
            gateway_config_path=gateway,
            allowed_roots=[tmp_path],
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result["ok"] is True
    assert result["content"] == "gateway stream result"
    assert result["usage"]["total_tokens"] == 7
    assert result["estimated_cost"] == 0.002
    assert result["response"]["streamed"] is True
    assert result["request_options"] == {
        "enable_thinking": False,
        "thinking_budget": 256,
        "max_tokens": 256,
        "stream": True,
        "max_retries": 1,
    }
    request = _FakeGatewayHandler.requests[0]
    payload = json.loads(request["body"])
    assert request["accept"] == "text/event-stream"
    assert payload["stream"] is True
    assert payload["max_tokens"] == 256
    assert payload["extra_body"] == {
        "enable_thinking": False,
        "thinking_budget": 256,
    }
    assert "transient reasoning" not in json.dumps(result)

def test_runtime_passes_route_locked_thinking_disabled_to_proxy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    upsert_model_api_profile(
        {
            "id": "ark-api-glm-5-2",
            "name": "Ark API GLM 5.2",
            "provider": "volcengine_ark",
            "adapter_backend": "proxy",
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "model": "glm-5-2-260617",
            "location": "remote",
            "capabilities": ["text"],
            "provider_options": {
                "thinking_mode": "disabled",
                "response_format": "json_object",
                "max_tokens": 128,
            },
        },
        tasks=["summary_rewrite"],
        settings_path=settings,
        secrets_path=secrets,
    )
    server, thread, gateway = _gateway(tmp_path)
    monkeypatch.setattr(model_runtime_client, "_gateway_api_key", lambda path: "local-gateway-key")
    try:
        result = _consented_runtime_request(
            "summary_rewrite",
            execution_location="remote",
            text="source text",
            prompt="summarise",
            execute=True,
            settings_path=settings,
            secrets_path=secrets,
            gateway_config_path=gateway,
            allowed_roots=[tmp_path],
        )
        response_format_mismatch = _consented_runtime_request(
            "summary_rewrite",
            execution_location="remote",
            text="source text",
            prompt="summarise",
            response_format={"type": "text"},
            execute=True,
            settings_path=settings,
            secrets_path=secrets,
            gateway_config_path=gateway,
            allowed_roots=[tmp_path],
        )
        max_tokens_mismatch = _consented_runtime_request(
            "summary_rewrite",
            execution_location="remote",
            text="source text",
            prompt="summarise",
            max_tokens=64,
            execute=True,
            settings_path=settings,
            secrets_path=secrets,
            gateway_config_path=gateway,
            allowed_roots=[tmp_path],
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result["ok"] is True
    assert response_format_mismatch["status"] == "invalid_runtime_request"
    assert "response_format does not match" in response_format_mismatch["error"]
    assert max_tokens_mismatch["status"] == "invalid_runtime_request"
    assert "max_tokens does not match" in max_tokens_mismatch["error"]
    assert len(_FakeGatewayHandler.requests) == 1
    payload = json.loads(_FakeGatewayHandler.requests[0]["body"])
    assert payload["extra_body"] == {"thinking": {"type": "disabled"}}
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_tokens"] == 128
    assert result["request_options"] == {
        "thinking_mode": "disabled",
        "response_format": "json_object",
        "max_tokens": 128,
        "max_retries": 1,
    }
    assert "thinking" not in payload
    assert "allowed_openai_params" not in payload


def test_runtime_allows_coding_plan_for_consented_text_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    upsert_model_api_profile(
        {
            "id": "coding-plan-only",
            "name": "Coding Plan only",
            "provider": "volcengine_coding_plan",
            "adapter_backend": "proxy",
            "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
            "model": "deepseek-v4-pro",
            "location": "remote",
            "capabilities": ["text"],
        },
        tasks=["summary_rewrite"],
        settings_path=settings,
        secrets_path=secrets,
    )
    gateway = tmp_path / "gateway.json"
    gateway.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.model_gateway_config.v1",
                "host": "127.0.0.1",
                "port": 8776,
            }
        ),
        encoding="utf-8",
    )
    network_called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal network_called
        network_called = True
        raise AssertionError("network must be blocked before urlopen")

    monkeypatch.setattr(model_runtime_client.urllib.request, "urlopen", forbidden)
    result = _consented_runtime_request(
        "summary_rewrite",
        execution_location="remote",
        text="source",
        execute=True,
        settings_path=settings,
        secrets_path=secrets,
        gateway_config_path=gateway,
        allowed_roots=[tmp_path],
    )

    assert result["status"] != "provider_usage_scope_blocked"
    assert result["remote_requests_made"] is None
    assert "blocked_deployments" not in result
    assert network_called is False


def test_runtime_disables_qwen_reasoning_and_sanitizes_structured_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    upsert_model_api_profile(
        {
            "id": "groq-qwen3-6-27b",
            "name": "Groq Qwen3.6 27B",
            "provider": "groq",
            "litellm_provider": "groq",
            "adapter_backend": "proxy",
            "base_url": "https://api.groq.com/openai/v1",
            "model": "qwen/qwen3.6-27b",
            "location": "remote",
            "capabilities": ["vision"],
            "provider_options": {
                "reasoning_effort": "none",
                "reasoning_format": "hidden",
                "strip_reasoning_tags": True,
                "strip_json_fences": True,
            },
        },
        tasks=["document_visual"],
        settings_path=settings,
        secrets_path=secrets,
    )
    server, thread, gateway = _gateway(tmp_path)
    monkeypatch.setattr(
        model_runtime_client,
        "_gateway_api_key",
        lambda path: "local-gateway-key",
    )
    _FakeGatewayHandler.chat_content = (
        '<think>private chain</think>\n```json\n{"title":"PPT"}\n```'
    )
    image = tmp_path / "slide.png"
    image.write_bytes(b"png")
    try:
        result = _consented_runtime_request(
            "document_visual",
            execution_location="remote",
            image_paths=[image],
            prompt="return JSON",
            execute=True,
            settings_path=settings,
            secrets_path=secrets,
            gateway_config_path=gateway,
            allowed_roots=[tmp_path],
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result["ok"] is True
    assert result["content"] == '{"title":"PPT"}'
    assert result["content_sanitization"] == {
        "reasoning_tags_removed": 1,
        "json_fence_removed": True,
        "reasoning_tags_remaining": False,
    }
    payload = json.loads(_FakeGatewayHandler.requests[0]["body"])
    assert payload["extra_body"] == {
        "reasoning_effort": "none",
        "reasoning_format": "hidden",
    }
    assert "strip_reasoning_tags" not in json.dumps(payload)


def test_runtime_route_mismatch_blocks_before_loopback_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings, secrets = _settings(tmp_path)
    server, thread, gateway = _gateway(tmp_path)
    monkeypatch.setattr(model_runtime_client, "_gateway_api_key", lambda path: "local-gateway-key")
    try:
        result = model_runtime_request(
            "summary_rewrite",
            execution_location="remote",
            route_revision="stale-revision",
            text="source",
            execute=True,
            settings_path=settings,
            secrets_path=secrets,
            gateway_config_path=gateway,
            allowed_roots=[tmp_path],
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result["status"] == "invalid_runtime_request"
    assert "route_revision" in result["error"]
    assert _FakeGatewayHandler.requests == []


@pytest.mark.parametrize(
    "image_path",
    ["data:image/png;base64,AAAA", "https://example.com/frame.png", "AAAA"],
)
def test_runtime_rejects_non_local_or_missing_image_inputs(tmp_path: Path, image_path: str) -> None:
    settings, secrets = _settings(tmp_path)
    gateway = tmp_path / "gateway.json"
    gateway.write_text(
        json.dumps({"schema": "video_knowledge_pipeline.model_gateway_config.v1", "host": "127.0.0.1", "port": 8776}),
        encoding="utf-8",
    )

    result = model_runtime_request(
        "semantic_frame",
        execution_location="remote",
        image_paths=[image_path],
        execute=False,
        settings_path=settings,
        secrets_path=secrets,
        gateway_config_path=gateway,
        allowed_roots=[tmp_path],
    )

    assert result["status"] == "invalid_runtime_request"


def test_runtime_rejects_path_outside_allowed_roots(tmp_path: Path) -> None:
    settings, secrets = _settings(tmp_path)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"png")

    result = model_runtime_request(
        "semantic_frame",
        execution_location="remote",
        image_paths=[outside],
        execute=False,
        settings_path=settings,
        secrets_path=secrets,
        allowed_roots=[allowed],
    )

    assert result["status"] == "invalid_runtime_request"
    assert "outside allowed roots" in result["error"]


def test_runtime_uses_content_addressed_revision_from_settings(tmp_path: Path) -> None:
    settings, secrets = _settings(tmp_path)
    route = resolve_model_api_route("summary_rewrite", execution_location="remote", settings_path=settings)

    result = model_runtime_request(
        "summary_rewrite",
        execution_location="remote",
        route_id=route["route_id"],
        route_revision=route["route_revision"],
        text="source",
        execute=False,
        settings_path=settings,
        secrets_path=secrets,
        allowed_roots=[tmp_path],
    )

    assert result["route_id"] == route["route_id"]
    assert result["route_revision"] == route["route_revision"]

def test_runtime_rejects_multimodal_message_payloads(tmp_path: Path) -> None:
    settings, secrets = _settings(tmp_path)

    result = model_runtime_request(
        "summary_rewrite",
        execution_location="remote",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                ],
            }
        ],
        execute=False,
        settings_path=settings,
        secrets_path=secrets,
        allowed_roots=[tmp_path],
    )

    assert result["status"] == "invalid_runtime_request"
    assert "local image_paths" in result["error"]

def test_asr_prompt_is_bounded_in_preview(tmp_path: Path) -> None:
    settings, secrets = _settings(tmp_path)
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"wave")
    gateway = tmp_path / "gateway.json"
    gateway.write_text(
        json.dumps({"schema": "video_knowledge_pipeline.model_gateway_config.v1", "host": "127.0.0.1", "port": 8776}),
        encoding="utf-8",
    )

    result = model_runtime_request(
        "asr",
        execution_location="remote",
        audio_path=audio,
        prompt="词" * 5000,
        execute=False,
        settings_path=settings,
        secrets_path=secrets,
        gateway_config_path=gateway,
        allowed_roots=[tmp_path],
    )

    assert result["request"]["prompt_chars"] == 4000
    assert result["request"]["prompt_truncated"] is True

def test_remote_proxy_execute_without_broker_grant_is_blocked_before_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings, secrets = _settings(tmp_path)
    gateway = tmp_path / "gateway.json"
    gateway.write_text(
        json.dumps({"schema": "video_knowledge_pipeline.model_gateway_config.v1", "host": "127.0.0.1", "port": 8776}),
        encoding="utf-8",
    )
    network_called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal network_called
        network_called = True
        raise AssertionError("network must be blocked before urlopen")

    monkeypatch.setattr(model_runtime_client.urllib.request, "urlopen", forbidden)
    result = model_runtime_request(
        "summary_rewrite",
        execution_location="remote",
        text="source",
        consent_id="caller-supplied-is-not-a-grant",
        execute=True,
        settings_path=settings,
        secrets_path=secrets,
        gateway_config_path=gateway,
        allowed_roots=[tmp_path],
    )

    assert result["status"] == "remote_consent_required"
    assert result["remote_requests_made"] is False
    assert network_called is False

def test_local_only_gateway_failure_reports_zero_remote_requests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.json"
    secrets = tmp_path / "secrets.json"
    upsert_model_api_profile(
        {
            "id": "local-text",
            "name": "Local text",
            "provider": "openai_compatible",
            "adapter_backend": "proxy",
            "base_url": "http://127.0.0.1:9901/v1",
            "model": "local-model",
            "location": "local",
            "capabilities": ["text"],
        },
        tasks=["summary_rewrite"],
        settings_path=settings,
        secrets_path=secrets,
    )
    gateway = tmp_path / "model-gateway.json"
    gateway.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.model_gateway_config.v1",
                "host": "127.0.0.1",
                "port": 8776,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_runtime_client, "_gateway_api_key", lambda path: "local-gateway-key")

    def unavailable(*args: object, **kwargs: object) -> object:
        raise urllib.error.URLError("loopback proxy stopped")

    monkeypatch.setattr(model_runtime_client.urllib.request, "urlopen", unavailable)
    result = model_runtime_request(
        "summary_rewrite",
        execution_location="local",
        text="source",
        execute=True,
        settings_path=settings,
        secrets_path=secrets,
        gateway_config_path=gateway,
        allowed_roots=[tmp_path],
    )

    assert result["status"] == "local_gateway_unavailable"
    assert result["remote_requests_made"] is False
    assert result["gateway"]["endpoint"].startswith("http://127.0.0.1:8776/")

def test_response_deployment_uses_litellm_model_id_for_same_model_fallback() -> None:
    route = {
        "route_revision": "a" * 64,
        "deployments": [
            {"id": "first", "provider": "openai_compatible", "model": "shared-model"},
            {"id": "second", "provider": "openai_compatible", "model": "shared-model"},
        ],
    }
    selected_id = model_runtime_client._gateway_deployment_id(route, 1)

    selected = model_runtime_client._response_deployment(
        route,
        {"model": "shared-model"},
        {"X-LiteLLM-Model-ID": selected_id},
    )
    undisclosed = model_runtime_client._response_deployment(
        route,
        {"model": "shared-model"},
        {},
    )

    assert selected["id"] == "second"
    assert selected["selection"] == "litellm_model_id"
    assert undisclosed["id"] == ""
    assert undisclosed["selection"] == "gateway_not_disclosed"
