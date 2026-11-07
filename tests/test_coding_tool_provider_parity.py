from __future__ import annotations

import json
import shutil
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import video_knowledge_pipeline.coding_tool_provider_parity as parity
import video_knowledge_pipeline.trusted_model_connector_remote_mcp as remote_mcp
from video_knowledge_pipeline.model_api_settings import SECRETS_SCHEMA, SETTINGS_SCHEMA
from video_knowledge_pipeline.storage import read_json, write_json


def _profile(profile_id: str, provider: str, base_url: str, model: str) -> dict[str, object]:
    return {
        "id": profile_id,
        "name": profile_id,
        "provider": provider,
        "litellm_provider": "openai",
        "provider_options": {},
        "adapter_backend": "proxy",
        "location": "remote",
        "capabilities": ["text"],
        "base_url": base_url,
        "model": model,
        "secret_ref": f"dpapi:{profile_id}",
        "timeout_seconds": 120,
        "enabled": True,
    }


def _stores(root: Path) -> tuple[Path, Path, Path]:
    profiles = []
    for pair in parity.MODEL_PAIRS:
        profiles.append(
            _profile(
                pair["ark_profile_id"],
                "volcengine_coding_plan",
                parity.OFFICIAL_REFERENCES["volcengine_coding_plan"]["base_url"],
                pair["ark_model"],
            )
        )
        profiles.append(
            _profile(
                pair["siliconflow_profile_id"],
                "siliconflow",
                parity.OFFICIAL_REFERENCES["siliconflow"]["base_url"],
                pair["siliconflow_model"],
            )
        )
    settings = root / "settings.json"
    secrets = root / "secrets.json"
    artifact = root / "atomic-quota-fixture.txt"
    write_json(
        settings,
        {
            "schema": SETTINGS_SCHEMA,
            "profiles": profiles,
            "task_routes": {},
            "route_pools": [],
            "route_bindings": {},
            "updated_at": "2026-07-18T00:00:00+00:00",
        },
    )
    write_json(
        secrets,
        {
            "schema": SECRETS_SCHEMA,
            "items": {
                str(row["id"]): {"ciphertext": "test-only-not-decrypted"}
                for row in profiles
            },
            "updated_at": "2026-07-18T00:00:00+00:00",
        },
    )
    artifact.write_text("fixed synthetic code sample", encoding="utf-8")
    return settings, secrets, artifact


def test_prepare_locks_ten_exact_native_candidates_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parity, "project_root", lambda: tmp_path)
    settings, secrets, artifact = _stores(tmp_path)

    result = parity.prepare_coding_tool_provider_parity(
        settings_path=settings,
        secrets_path=secrets,
        artifact_path=artifact,
        output_dir=tmp_path / "out",
    )

    assert result["status"] == "ready_for_operator_consent"
    assert result["candidate_count"] == 10
    assert result["operator_boundary"]["remote_requests_made"] == 0
    assert result["comparison_contract"]["fallbacks"] == []
    assert result["comparison_contract"]["external_retry_count"] == 0
    assert result["comparison_contract"]["execution_client"] == parity.NATIVE_EXECUTION_CLIENT
    assert result["comparison_contract"]["tool_surface"] is False
    assert result["comparison_contract"]["reasoning_mode_scope"] == (
        "provider_default_without_vendor_specific_override"
    )
    assert result["comparison_contract"]["same_weights_assumed"] is False
    assert {row["api"] for row in result["candidates"]} == {"openai_chat_completions"}
    assert {row["destination"] for row in result["candidates"]} == set(
        parity.EXPECTED_DESTINATIONS
    )
    assert all(len(row["route"]["deployments"]) == 1 for row in result["candidates"])
    saved = read_json(Path(result["artifacts"]["plan"]))
    assert saved["plan_sha256"] == parity._payload_sha256(saved)


def test_prepare_rejects_model_alias_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parity, "project_root", lambda: tmp_path)
    settings, secrets, artifact = _stores(tmp_path)
    document = read_json(settings)
    document["profiles"][0]["model"] = "auto"
    write_json(settings, document)

    result = parity.prepare_coding_tool_provider_parity(
        settings_path=settings,
        secrets_path=secrets,
        artifact_path=artifact,
        output_dir=tmp_path / "out",
    )

    assert result["status"] == "incomplete"
    assert any(row["key"] == "profile_contract_mismatch" for row in result["blockers"])


def test_content_quality_profile_locks_complete_output_request_into_route_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parity, "project_root", lambda: tmp_path)
    settings, secrets, artifact = _stores(tmp_path)

    common = parity.prepare_coding_tool_provider_parity(
        settings_path=settings,
        secrets_path=secrets,
        artifact_path=artifact,
        output_dir=tmp_path / "common",
    )
    quality = parity.prepare_coding_tool_provider_parity(
        settings_path=settings,
        secrets_path=secrets,
        artifact_path=artifact,
        output_dir=tmp_path / "quality",
        request_profile_id=parity.CONTENT_QUALITY_PROFILE,
    )

    contract = quality["comparison_contract"]
    assert contract["request_profile_id"] == parity.CONTENT_QUALITY_PROFILE
    assert contract["max_tokens"] == 16384
    assert contract["streaming"] is True
    assert contract["timeout_seconds"] == 300
    assert quality["upload_manifest"]["max_total_cost_usd"] == 0.8
    for common_candidate, quality_candidate in zip(
        common["candidates"], quality["candidates"], strict=True
    ):
        deployment = quality_candidate["route"]["deployments"][0]
        assert deployment["provider_options"] == {
            "max_tokens": 16384,
            "stream": True,
            "temperature": 0,
        }
        assert deployment["timeout_seconds"] == 300
        assert quality_candidate["route"]["route_revision"] != common_candidate["route"][
            "route_revision"
        ]


def test_content_quality_consents_use_new_cost_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parity, "project_root", lambda: tmp_path)
    settings, secrets, artifact = _stores(tmp_path)
    plan = parity.prepare_coding_tool_provider_parity(
        settings_path=settings,
        secrets_path=secrets,
        artifact_path=artifact,
        output_dir=tmp_path / "quality",
        request_profile_id=parity.CONTENT_QUALITY_PROFILE,
    )
    index = parity.create_coding_tool_provider_parity_consents(
        plan["artifacts"]["plan"], confirm_data_export=True
    )

    for row in index["consents"]:
        consent = read_json(Path(row["consent_path"]))
        assert consent["scope"]["max_estimated_cost_usd"] == 0.08
        assert consent["authorized_deployments"][0]["provider_options"][
            "max_tokens"
        ] == 16384


def test_capability_ceiling_profile_omits_client_token_and_thinking_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parity, "project_root", lambda: tmp_path)
    settings, secrets, artifact = _stores(tmp_path)

    result = parity.prepare_coding_tool_provider_parity(
        settings_path=settings,
        secrets_path=secrets,
        artifact_path=artifact,
        output_dir=tmp_path / "ceiling",
        request_profile_id=parity.CAPABILITY_CEILING_PROFILE,
    )

    contract = result["comparison_contract"]
    assert contract["request_profile_id"] == parity.CAPABILITY_CEILING_PROFILE
    assert contract["max_tokens"] is None
    assert contract["streaming"] is True
    assert contract["timeout_seconds"] == 900
    assert contract["unverified_provider_fields_sent"] == []
    assert result["upload_manifest"]["max_total_cost_usd"] == 2.0
    for candidate in result["candidates"]:
        deployment = candidate["route"]["deployments"][0]
        assert deployment["provider_options"] == {
            "stream": True,
            "temperature": 0,
        }
        assert deployment["timeout_seconds"] == 900
        assert parity.CAPABILITY_CEILING_PROFILE in candidate["route"]["route_id"]


def test_consent_creation_is_ten_candidate_specific_and_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parity, "project_root", lambda: tmp_path)
    settings, secrets, artifact = _stores(tmp_path)
    plan = parity.prepare_coding_tool_provider_parity(
        settings_path=settings,
        secrets_path=secrets,
        artifact_path=artifact,
        output_dir=tmp_path / "out",
    )

    with pytest.raises(ValueError, match="confirm-data-export"):
        parity.create_coding_tool_provider_parity_consents(
            plan["artifacts"]["plan"], confirm_data_export=False
        )
    index = parity.create_coding_tool_provider_parity_consents(
        plan["artifacts"]["plan"], confirm_data_export=True
    )

    assert index["consent_count"] == 10
    assert index["operator_boundary"]["remote_requests_made"] == 0
    for row in index["consents"]:
        consent = read_json(Path(row["consent_path"]))
        assert consent["task"] == "provider_task_benchmark"
        assert consent["scope"]["max_calls"] == 1
        assert consent["scope"]["max_estimated_cost_usd"] == 0.02
        assert len(consent["authorized_deployments"]) == 1
        assert consent["upload_manifest"]["files"][0]["sha256"] == plan["artifact"]["sha256"]


def test_native_candidate_execution_uses_no_openclaw_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parity, "project_root", lambda: tmp_path)
    settings, secrets, artifact = _stores(tmp_path)
    plan = parity.prepare_coding_tool_provider_parity(
        settings_path=settings,
        secrets_path=secrets,
        artifact_path=artifact,
        output_dir=tmp_path / "out",
    )
    index = parity.create_coding_tool_provider_parity_consents(
        plan["artifacts"]["plan"], confirm_data_export=True
    )
    saved_plan = read_json(Path(plan["artifacts"]["plan"]))
    candidate = saved_plan["candidates"][0]
    parsed = {
        "bug_class": "check-then-act race",
        "explanation": "BEGIN IMMEDIATE serializes writers while preserving schema and signature.",
        "patch": "+ conn.execute('BEGIN IMMEDIATE')\n+ conn.rollback()",
        "tests": ["multiprocessing workers assert accepted count is at most limit"],
        "tradeoffs": ["writers serialize briefly"],
    }
    monkeypatch.setattr(parity, "_read_secret", lambda *_args: "secret-not-logged")
    monkeypatch.setattr(
        parity,
        "_resolve_openclaw_command",
        lambda *_args: (_ for _ in ()).throw(AssertionError("OpenClaw must not be resolved")),
    )

    def fake_native(**kwargs: object) -> dict[str, object]:
        assert kwargs["candidate"] == candidate
        assert kwargs["api_key"] == "secret-not-logged"
        return {
            "ok": True,
            "status": "completed",
            "content": json.dumps(parsed, ensure_ascii=False),
            "usage": {"total_tokens": 20},
            "estimated_cost": 0.001,
            "audit": {
                "external_request_count": 1,
                "request_model": candidate["model"],
                "provider_response_model": candidate["model"],
                "upstream_status": 200,
            },
            "response": {"http_status": 200},
        }

    monkeypatch.setattr(parity, "_call_native_openai_compatible_once", fake_native)
    result = parity.execute_coding_tool_provider_parity_candidate(
        plan["artifacts"]["plan"],
        index["consent_index_path"],
        candidate_id=str(candidate["candidate_id"]),
        operator_confirm_network=True,
    )

    assert result["ok"] is True
    assert result["execution_client"] == parity.NATIVE_EXECUTION_CLIENT
    assert result["openclaw"] == {}
    assert result["operator_boundary"]["tool_surface"] is False
    assert result["operator_boundary"]["external_request_count"] == 1
    assert "secret-not-logged" not in json.dumps(result)
    consent = read_json(Path(index["consents"][0]["consent_path"]))
    assert consent["usage"]["calls_attempted"] == 1
    assert consent["usage"]["calls_completed"] == 1
    assert consent["usage"]["cost_reported_usd"] == 0.001


def test_consent_entrypoint_recovers_only_the_exact_locked_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parity, "project_root", lambda: tmp_path)
    settings, secrets, artifact = _stores(tmp_path)
    plan = parity.prepare_coding_tool_provider_parity(
        settings_path=settings,
        secrets_path=secrets,
        artifact_path=artifact,
        output_dir=tmp_path / "out",
    )
    index = parity.create_coding_tool_provider_parity_consents(
        plan["artifacts"]["plan"], confirm_data_export=True
    )
    consent_row = index["consents"][0]
    captured: dict[str, object] = {}

    def fake_execute(
        plan_path: str | Path,
        consent_index_path: str | Path,
        *,
        candidate_id: str,
        operator_confirm_network: bool,
        openclaw_command: str = "openclaw",
    ) -> dict[str, object]:
        captured.update(
            plan_path=str(Path(plan_path).resolve()),
            consent_index_path=str(Path(consent_index_path).resolve()),
            candidate_id=candidate_id,
            operator_confirm_network=operator_confirm_network,
            openclaw_command=openclaw_command,
        )
        return {"ok": True, "candidate_id": candidate_id}

    monkeypatch.setattr(
        parity, "execute_coding_tool_provider_parity_candidate", fake_execute
    )
    result = parity.execute_coding_tool_provider_parity_consent(
        consent_row["consent_path"],
        expected_route_revision=consent_row["route_revision"],
    )

    assert result == {"ok": True, "candidate_id": captured["candidate_id"]}
    assert captured["candidate_id"] == "deepseek-v4-pro--ark"
    assert captured["operator_confirm_network"] is True
    assert captured["plan_path"] == str(Path(plan["artifacts"]["plan"]).resolve())
    assert captured["consent_index_path"] == str(
        Path(index["consent_index_path"]).resolve()
    )

    with pytest.raises(ValueError, match="route revision"):
        parity.execute_coding_tool_provider_parity_consent(
            consent_row["consent_path"], expected_route_revision="stale"
        )
    with pytest.raises(ValueError, match="write=true"):
        parity.execute_coding_tool_provider_parity_consent(
            consent_row["consent_path"], write=False
        )


def _fake_execution(candidate: dict[str, object], *, ok: bool = True) -> dict[str, object]:
    parsed = {
        "bug_class": "check-then-act race",
        "explanation": "BEGIN IMMEDIATE serializes the transaction.",
        "patch": "+ conn.execute('BEGIN IMMEDIATE')",
        "tests": ["multiprocessing workers respect limit"],
        "tradeoffs": ["writers serialize"],
    }
    result = {
        "schema": parity.EXECUTION_SCHEMA,
        "ok": ok,
        "status": "completed" if ok else "provider_failed",
        "candidate_id": candidate["candidate_id"],
        "pair_id": candidate["pair_id"],
        "side": candidate["side"],
        "provider": candidate["provider"],
        "requested_model": candidate["model"],
        "model_identity": {
            "status": "exact",
            "provider_response_model": candidate["model"],
        },
        "route": candidate["route"],
        "latency_ms": 10,
        "content": json.dumps(parsed, ensure_ascii=False),
        "assessment": {
            "score": 100,
            "quality_gate_passed": ok,
            "parsed_json": parsed,
        },
        "audit": {
            "external_request_count": 1,
            "request_model": candidate["model"],
        },
    }
    write_json(Path(str(candidate["result_path"])), result)
    return {**result, "result_path": str(candidate["result_path"])}


def test_execute_all_is_sequential_resumable_and_has_no_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parity, "project_root", lambda: tmp_path)
    settings, secrets, artifact = _stores(tmp_path)
    plan = parity.prepare_coding_tool_provider_parity(
        settings_path=settings,
        secrets_path=secrets,
        artifact_path=artifact,
        output_dir=tmp_path / "out",
    )
    index = parity.create_coding_tool_provider_parity_consents(
        plan["artifacts"]["plan"], confirm_data_export=True
    )
    saved_plan = read_json(Path(plan["artifacts"]["plan"]))
    first = saved_plan["candidates"][0]
    _fake_execution(first)
    calls: list[str] = []

    def fake_execute(
        _plan_path: str | Path,
        _index_path: str | Path,
        *,
        candidate_id: str,
        operator_confirm_network: bool,
        openclaw_command: str,
    ) -> dict[str, object]:
        assert operator_confirm_network is True
        assert openclaw_command == "openclaw-test"
        calls.append(candidate_id)
        return _fake_execution(parity._candidate(saved_plan, candidate_id))

    monkeypatch.setattr(
        parity, "execute_coding_tool_provider_parity_candidate", fake_execute
    )
    result = parity.execute_all_coding_tool_provider_parity_candidates(
        plan["artifacts"]["plan"],
        index["consent_index_path"],
        operator_confirm_network=True,
        openclaw_command="openclaw-test",
    )

    assert result["status"] == "completed"
    assert result["candidate_count"] == 10
    assert result["attempted_this_run"] == 9
    assert result["reused_existing"] == 1
    assert result["external_requests_this_run"] == 9
    assert calls == [row["candidate_id"] for row in saved_plan["candidates"][1:]]
    assert result["operator_boundary"]["fallbacks"] == []
    assert result["operator_boundary"]["provider_model_or_url_overrides_accepted"] is False
    assert result["comparison"]["ready_pair_count"] == 5
    comparison = read_json(Path(result["artifacts"]["json"]))
    assert all(
        row["output_difference"]["classification"] == "exact_output_match"
        for row in comparison["pairs"]
    )
    assert all("content" not in row for pair in comparison["pairs"] for row in pair["candidates"])
    progress = read_json(Path(result["artifacts"]["batch_execution"]))
    assert progress["status"] == "completed"


def test_execute_all_rejects_tampered_existing_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parity, "project_root", lambda: tmp_path)
    settings, secrets, artifact = _stores(tmp_path)
    plan = parity.prepare_coding_tool_provider_parity(
        settings_path=settings,
        secrets_path=secrets,
        artifact_path=artifact,
        output_dir=tmp_path / "out",
    )
    index = parity.create_coding_tool_provider_parity_consents(
        plan["artifacts"]["plan"], confirm_data_export=True
    )
    saved_plan = read_json(Path(plan["artifacts"]["plan"]))
    first = saved_plan["candidates"][0]
    tampered = _fake_execution(first)
    tampered["requested_model"] = "auto"
    write_json(Path(str(first["result_path"])), tampered)

    with pytest.raises(ValueError, match="requested_model"):
        parity.execute_all_coding_tool_provider_parity_candidates(
            plan["artifacts"]["plan"],
            index["consent_index_path"],
            operator_confirm_network=True,
        )


class _UpstreamHandler(BaseHTTPRequestHandler):
    count = 0
    last_request: dict[str, object] = {}

    def do_POST(self) -> None:  # noqa: N802
        type(self).count += 1
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        request = json.loads(body)
        type(self).last_request = dict(request)
        if request.get("stream"):
            events = [
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": '{"ok":true}'},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "model": request["model"],
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            ]
            response = (
                "".join(
                    "data: " + json.dumps(event, separators=(",", ":")) + "\n\n"
                    for event in events
                )
                + "data: [DONE]\n\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("X-Request-Id", "request-test")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
            return
        response = json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": request["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"ok":true}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Request-Id", "request-test")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def test_audit_proxy_forwards_only_one_external_request() -> None:
    _UpstreamHandler.count = 0
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
    body = json.dumps({"model": "locked-model", "messages": []}).encode()
    try:
        with parity.SingleRequestAuditProxy(
            upstream_base_url=upstream_url,
            expected_model="locked-model",
            api_key="secret-not-logged",
            allow_http_upstream_for_tests=True,
        ) as proxy:
            first = urllib.request.urlopen(
                urllib.request.Request(
                    f"{proxy.base_url}/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
            )
            assert first.status == 200
            with pytest.raises(urllib.error.HTTPError) as repeated:
                urllib.request.urlopen(
                    urllib.request.Request(
                        f"{proxy.base_url}/chat/completions",
                        data=body,
                        headers={"Content-Type": "application/json"},
                    )
                )
            assert repeated.value.code == 400
            audit = proxy.audit_snapshot()
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    assert _UpstreamHandler.count == 1
    assert audit["external_request_count"] == 1
    assert audit["blocked_repeat_requests"] == 1
    assert audit["request_model"] == "locked-model"
    assert audit["provider_response_model"] == "locked-model"
    assert audit["upstream_request_ids"] == {"x-request-id": "request-test"}
    assert "secret-not-logged" not in json.dumps(audit)


def test_native_client_uses_existing_openai_contract_and_one_exact_request() -> None:
    _UpstreamHandler.count = 0
    _UpstreamHandler.last_request = {}
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
    candidate = {
        "model": "locked-model",
        "base_url": base_url,
        "destination": "127.0.0.1",
    }
    try:
        result = parity._call_native_openai_compatible_once(
            candidate=candidate,
            api_key="secret-not-logged",
            prompt="Return one JSON object.",
            timeout_seconds=10,
            allow_http_upstream_for_tests=True,
        )
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    assert result["ok"] is True
    assert result["content"] == '{"ok":true}'
    assert _UpstreamHandler.count == 1
    assert _UpstreamHandler.last_request["model"] == "locked-model"
    assert _UpstreamHandler.last_request["stream"] is False
    assert _UpstreamHandler.last_request["temperature"] == 0
    assert _UpstreamHandler.last_request["max_tokens"] == 1024
    assert "thinking" not in _UpstreamHandler.last_request
    assert "thinking_mode" not in _UpstreamHandler.last_request
    assert "enable_thinking" not in _UpstreamHandler.last_request
    assert result["audit"]["external_request_count"] == 1
    assert result["audit"]["request_model"] == "locked-model"
    assert result["audit"]["provider_response_model"] == "locked-model"
    assert "secret-not-logged" not in json.dumps(result)


def test_native_content_quality_request_streams_with_larger_locked_budget() -> None:
    _UpstreamHandler.count = 0
    _UpstreamHandler.last_request = {}
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
    candidate = {
        "model": "locked-model",
        "base_url": base_url,
        "destination": "127.0.0.1",
    }
    try:
        result = parity._call_native_openai_compatible_once(
            candidate=candidate,
            api_key="secret-not-logged",
            prompt="Return one JSON object.",
            timeout_seconds=10,
            request_profile=parity.REQUEST_PROFILES[parity.CONTENT_QUALITY_PROFILE],
            allow_http_upstream_for_tests=True,
        )
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    assert result["ok"] is True
    assert result["content"] == '{"ok":true}'
    assert _UpstreamHandler.count == 1
    assert _UpstreamHandler.last_request["stream"] is True
    assert _UpstreamHandler.last_request["max_tokens"] == 16384
    assert _UpstreamHandler.last_request["temperature"] == 0
    assert "thinking" not in _UpstreamHandler.last_request
    assert "thinking_mode" not in _UpstreamHandler.last_request
    assert "enable_thinking" not in _UpstreamHandler.last_request
    assert result["audit"]["request_profile_id"] == parity.CONTENT_QUALITY_PROFILE
    assert result["response"]["finish_reason"] == "stop"


def test_native_capability_ceiling_request_omits_model_output_limit() -> None:
    _UpstreamHandler.count = 0
    _UpstreamHandler.last_request = {}
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
    candidate = {
        "model": "locked-model",
        "base_url": base_url,
        "destination": "127.0.0.1",
    }
    try:
        result = parity._call_native_openai_compatible_once(
            candidate=candidate,
            api_key="secret-not-logged",
            prompt="Return one JSON object.",
            timeout_seconds=10,
            request_profile=parity.REQUEST_PROFILES[
                parity.CAPABILITY_CEILING_PROFILE
            ],
            allow_http_upstream_for_tests=True,
        )
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    assert result["ok"] is True
    assert _UpstreamHandler.last_request["stream"] is True
    assert "max_tokens" not in _UpstreamHandler.last_request
    assert "thinking" not in _UpstreamHandler.last_request
    assert "thinking_mode" not in _UpstreamHandler.last_request
    assert "enable_thinking" not in _UpstreamHandler.last_request
    assert "thinking_budget" not in _UpstreamHandler.last_request
    assert result["audit"]["request_max_tokens"] is None
    assert result["audit"]["request_max_tokens_omitted"] is True


class _HeartbeatStreamHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        started = time.monotonic()
        try:
            while time.monotonic() - started < 3:
                event = {
                    "model": "locked-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"reasoning_content": "still working"},
                            "finish_reason": None,
                        }
                    ],
                }
                self.wfile.write(
                    (
                        "data: "
                        + json.dumps(event, separators=(",", ":"))
                        + "\n\n"
                    ).encode()
                )
                self.wfile.flush()
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def log_message(self, _format: str, *_args: object) -> None:
        return


def test_native_client_enforces_total_wall_clock_with_sse_heartbeats() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _HeartbeatStreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    candidate = {
        "model": "locked-model",
        "base_url": f"http://127.0.0.1:{upstream.server_address[1]}/v1",
        "destination": "127.0.0.1",
    }
    started = time.monotonic()
    try:
        result = parity._call_native_openai_compatible_once(
            candidate=candidate,
            api_key="secret-not-logged",
            prompt="Return one JSON object.",
            timeout_seconds=1,
            request_profile=parity.REQUEST_PROFILES[
                parity.CAPABILITY_CEILING_PROFILE
            ],
            allow_http_upstream_for_tests=True,
        )
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    assert time.monotonic() - started < 3
    assert result["ok"] is False
    assert result["audit"]["external_request_count"] == 1
    assert result["audit"]["wall_clock_timeout_seconds"] == 1
    assert result["audit"]["wall_clock_timeout_exceeded"] is True
    assert result["response"]["error_type"] == "TimeoutError"


def test_audit_snapshot_survives_runner_failure_after_external_request() -> None:
    _UpstreamHandler.count = 0
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
    body = json.dumps({"model": "locked-model", "messages": []}).encode()
    proxy = parity.SingleRequestAuditProxy(
        upstream_base_url=upstream_url,
        expected_model="locked-model",
        api_key="secret-not-logged",
        allow_http_upstream_for_tests=True,
    )
    try:
        with pytest.raises(RuntimeError, match="simulated runner failure"):
            with proxy:
                response = urllib.request.urlopen(
                    urllib.request.Request(
                        f"{proxy.base_url}/chat/completions",
                        data=body,
                        headers={"Content-Type": "application/json"},
                    )
                )
                assert response.status == 200
                raise RuntimeError("simulated runner failure")
        audit = proxy.audit_snapshot()
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    assert _UpstreamHandler.count == 1
    assert audit["external_request_count"] == 1
    assert audit["request_model"] == "locked-model"
    assert "secret-not-logged" not in json.dumps(audit)


@pytest.mark.skipif(shutil.which("openclaw") is None, reason="OpenClaw is not installed")
def test_installed_openclaw_uses_exact_model_through_loopback_contract(
    tmp_path: Path,
) -> None:
    _UpstreamHandler.count = 0
    _UpstreamHandler.last_request = {}
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
    proxy = parity.SingleRequestAuditProxy(
        upstream_base_url=upstream_url,
        expected_model="locked-model",
        api_key="secret-not-logged",
        allow_http_upstream_for_tests=True,
    )
    try:
        with proxy:
            config_path, state_dir, workspace = parity._write_openclaw_config(
                tmp_path,
                candidate={"model": "locked-model"},
                proxy_base_url=proxy.base_url,
            )
            result = parity._run_openclaw(
                str(shutil.which("openclaw")),
                config_path=config_path,
                state_dir=state_dir,
                workspace=workspace,
                prompt="Return one JSON object with ok set to true and do not use tools.",
                timeout_seconds=30,
                redactions=("secret-not-logged",),
            )
            audit = proxy.audit_snapshot()
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    write_json(tmp_path / "openclaw-result.json", result)
    assert result["returncode"] == 0, result
    assert result["phase"] == "agent"
    assert _UpstreamHandler.count == 1
    assert audit["external_request_count"] == 1
    assert audit["blocked_repeat_requests"] == 0
    assert audit["request_model"] == "locked-model"
    assert audit["provider_response_model"] == "locked-model"
    assert _UpstreamHandler.last_request.get("stream") is True
    assert parity._openclaw_content(result["json"]) == '{"ok":true}', result
    assert "secret-not-logged" not in json.dumps(result)


def test_assessment_rewards_atomic_cross_process_fix_and_rejects_think_leak() -> None:
    content = json.dumps(
        {
            "bug_class": "check-then-act race",
            "explanation": "BEGIN IMMEDIATE serializes the transaction and rollback cleans failures while preserving schema and signature.",
            "patch": "@@\n+    conn.execute('BEGIN IMMEDIATE')\n+    conn.rollback()",
            "tests": ["multiprocessing concurrent workers; assert sum(results) == limit"],
            "tradeoffs": ["writers serialize briefly"],
        }
    )

    passed = parity.assess_coding_parity_output(content)
    leaked = parity.assess_coding_parity_output(f"<think>hidden</think>{content}")
    fenced = parity.assess_coding_parity_output(f"```json\n{content}\n```")
    wrong_shape = json.loads(content)
    wrong_shape["tradeoffs"] = "writers serialize briefly"
    wrong_shape_result = parity.assess_coding_parity_output(json.dumps(wrong_shape))
    missing_commit = json.loads(content)
    missing_commit["patch"] = (
        "@@\n+    cur = conn.execute('INSERT INTO quotas VALUES (?, ?) "
        "ON CONFLICT(user_id) DO UPDATE SET used = used + 1')"
    )
    missing_commit_result = parity.assess_coding_parity_output(
        json.dumps(missing_commit)
    )

    assert passed["quality_gate_passed"] is True
    assert passed["score"] == 100
    assert leaked["quality_gate_passed"] is False
    assert fenced["checks"]["json_object"] is True
    assert fenced["checks"]["no_forbidden_markers"] is False
    assert fenced["quality_gate_passed"] is False
    assert wrong_shape_result["checks"]["output_contract_shape"] is False
    assert wrong_shape_result["quality_gate_passed"] is False
    assert missing_commit_result["checks"]["success_path_persisted"] is False
    assert missing_commit_result["quality_gate_passed"] is False
    assert missing_commit_result["score"] == 80


def test_openclaw_command_is_resolved_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shim = tmp_path / "openclaw.CMD"
    shim.write_text("@echo off\r\n", encoding="utf-8")
    monkeypatch.setattr(parity.shutil, "which", lambda value: str(shim))

    assert parity._resolve_openclaw_command("openclaw") == str(shim.resolve())
    monkeypatch.setattr(parity.shutil, "which", lambda value: None)
    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        parity._resolve_openclaw_command("missing-openclaw")


def test_capability_ceiling_consent_selects_background_execution(
    tmp_path: Path,
) -> None:
    output = tmp_path / "suite"
    consent = output / "candidate" / "consent.v2.json"
    consent.parent.mkdir(parents=True)
    consent.write_text("{}", encoding="utf-8")
    write_json(
        output / "parity-plan.json",
        {
            "comparison_contract": {
                "request_profile_id": parity.CAPABILITY_CEILING_PROFILE
            }
        },
    )

    assert remote_mcp._parity_plan_uses_background_execution(consent) is True

    write_json(
        output / "parity-plan.json",
        {
            "comparison_contract": {
                "request_profile_id": parity.COMMON_FIELDS_PROFILE
            }
        },
    )
    assert remote_mcp._parity_plan_uses_background_execution(consent) is False


def test_background_submissions_are_serial_within_one_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "suite"
    first = output / "first" / "consent.v2.json"
    second = output / "second" / "consent.v2.json"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    started = {
        "first": threading.Event(),
        "second": threading.Event(),
    }
    release = {
        "first": threading.Event(),
        "second": threading.Event(),
    }
    order: list[str] = []
    active = 0
    max_active = 0
    state_lock = threading.Lock()

    def fake_execute(
        consent_path: Path,
        _revision: str,
        _write: bool,
    ) -> dict[str, object]:
        nonlocal active, max_active
        candidate_id = consent_path.parent.name
        with state_lock:
            active += 1
            max_active = max(max_active, active)
            order.append(candidate_id)
        started[candidate_id].set()
        assert release[candidate_id].wait(timeout=5)
        with state_lock:
            active -= 1
        return {"status": "completed", "candidate_id": candidate_id}

    monkeypatch.setattr(remote_mcp, "_execute_parity_background", fake_execute)
    with remote_mcp._PARITY_BACKGROUND_LOCK:
        remote_mcp._PARITY_BACKGROUND_FUTURES.clear()
        executors = list(remote_mcp._PARITY_BACKGROUND_EXECUTORS.values())
        remote_mcp._PARITY_BACKGROUND_EXECUTORS.clear()
    for executor in executors:
        executor.shutdown(wait=True, cancel_futures=True)

    consent = {"authorized_destinations": ["https://api.siliconflow.cn"]}
    try:
        first_result = remote_mcp._submit_parity_background(
            first,
            consent=consent,
            expected_route_revision="revision-1",
            write=True,
        )
        second_result = remote_mcp._submit_parity_background(
            second,
            consent=consent,
            expected_route_revision="revision-2",
            write=True,
        )
        assert first_result["status"] == "accepted"
        assert second_result["status"] == "accepted"
        assert started["first"].wait(timeout=2)
        assert started["second"].wait(timeout=0.1) is False
        release["first"].set()
        assert started["second"].wait(timeout=2)
        release["second"].set()
        futures = list(remote_mcp._PARITY_BACKGROUND_FUTURES.values())
        for future in futures:
            assert future.result(timeout=2)["status"] == "completed"
    finally:
        release["first"].set()
        release["second"].set()
        with remote_mcp._PARITY_BACKGROUND_LOCK:
            executors = list(remote_mcp._PARITY_BACKGROUND_EXECUTORS.values())
            remote_mcp._PARITY_BACKGROUND_EXECUTORS.clear()
            remote_mcp._PARITY_BACKGROUND_FUTURES.clear()
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)

    assert order == ["first", "second"]
    assert max_active == 1


def test_background_submissions_parallelize_different_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "suite"
    first = output / "first" / "consent.v2.json"
    second = output / "second" / "consent.v2.json"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    started = {
        "first": threading.Event(),
        "second": threading.Event(),
    }
    release = threading.Event()

    def fake_execute(
        consent_path: Path,
        _revision: str,
        _write: bool,
    ) -> dict[str, object]:
        candidate_id = consent_path.parent.name
        started[candidate_id].set()
        assert release.wait(timeout=5)
        return {"status": "completed", "candidate_id": candidate_id}

    monkeypatch.setattr(remote_mcp, "_execute_parity_background", fake_execute)
    with remote_mcp._PARITY_BACKGROUND_LOCK:
        remote_mcp._PARITY_BACKGROUND_FUTURES.clear()
        executors = list(remote_mcp._PARITY_BACKGROUND_EXECUTORS.values())
        remote_mcp._PARITY_BACKGROUND_EXECUTORS.clear()
    for executor in executors:
        executor.shutdown(wait=True, cancel_futures=True)

    try:
        first_result = remote_mcp._submit_parity_background(
            first,
            consent={"authorized_destinations": ["https://ark.example"]},
            expected_route_revision="revision-1",
            write=True,
        )
        second_result = remote_mcp._submit_parity_background(
            second,
            consent={"authorized_destinations": ["https://siliconflow.example"]},
            expected_route_revision="revision-2",
            write=True,
        )
        assert first_result["status"] == "accepted"
        assert second_result["status"] == "accepted"
        assert started["first"].wait(timeout=2)
        assert started["second"].wait(timeout=2)
        release.set()
        futures = list(remote_mcp._PARITY_BACKGROUND_FUTURES.values())
        for future in futures:
            assert future.result(timeout=2)["status"] == "completed"
    finally:
        release.set()
        with remote_mcp._PARITY_BACKGROUND_LOCK:
            executors = list(remote_mcp._PARITY_BACKGROUND_EXECUTORS.values())
            remote_mcp._PARITY_BACKGROUND_EXECUTORS.clear()
            remote_mcp._PARITY_BACKGROUND_FUTURES.clear()
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)


def test_recover_interrupted_candidate_never_reserves_a_second_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parity, "project_root", lambda: tmp_path)
    settings, secrets, artifact = _stores(tmp_path)
    plan = parity.prepare_coding_tool_provider_parity(
        settings_path=settings,
        secrets_path=secrets,
        artifact_path=artifact,
        output_dir=tmp_path / "out",
        request_profile_id=parity.CAPABILITY_CEILING_PROFILE,
    )
    index = parity.create_coding_tool_provider_parity_consents(
        plan["artifacts"]["plan"],
        confirm_data_export=True,
    )
    saved_plan = read_json(Path(plan["artifacts"]["plan"]))
    candidate = saved_plan["candidates"][0]
    consent_path = Path(candidate["consent_path"])
    reservation = parity.reserve_model_connector_attempt(
        consent_path,
        route_snapshot=candidate["route"],
        expected_route_revision=candidate["route"]["route_revision"],
        expected_task="provider_task_benchmark",
        expected_calls=1,
    )
    assert reservation["reserved"] is True

    result = parity.recover_interrupted_coding_tool_provider_parity_candidate(
        plan["artifacts"]["plan"],
        index["consent_index_path"],
        candidate_id=candidate["candidate_id"],
        reason="wall_clock_timeout_after_broker_restart",
    )
    consent = read_json(consent_path)

    assert result["status"] == "runner_failed"
    assert result["error"]["type"] == "WallClockTimeout"
    assert result["audit"]["external_request_count"] == 1
    assert result["audit"]["wall_clock_timeout_exceeded"] is True
    assert consent["usage"]["calls_attempted"] == 1
    assert consent["usage"]["calls_completed"] == 0
    assert consent["usage"]["cost_unreported_calls"] == 1

    reused = parity.recover_interrupted_coding_tool_provider_parity_candidate(
        plan["artifacts"]["plan"],
        index["consent_index_path"],
        candidate_id=candidate["candidate_id"],
        reason="wall_clock_timeout_after_broker_restart",
    )
    assert reused["recovered"] is False
    assert read_json(consent_path)["usage"]["calls_attempted"] == 1
