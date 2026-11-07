from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.model_connector_consent import (
    create_model_connector_consent,
    validate_model_connector_consent,
)
from video_knowledge_pipeline.trusted_model_connector import (
    execute_consented_bundle_vision,
    execute_consented_model_task,
    trusted_model_connector_capabilities,
)


PROVIDER = {
    "provider": "custom_openai_compatible",
    "base_url": "https://example.invalid/v1",
    "model": "test-model",
}


def test_text_consent_locks_artifact_provider_task_and_call_limit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "transcript.md"
    source.write_text("A corrected transcript.", encoding="utf-8")
    consent = create_model_connector_consent(
        tmp_path,
        task="smart_summary_rewrite",
        artifact_paths=[source],
        provider_config=PROVIDER,
        instructions="Create a structured summary.",
        max_calls=1,
        max_estimated_cost_usd=1.0,
        confirm_data_export=True,
    )
    status = validate_model_connector_consent(
        consent["consent_path"], provider_config=PROVIDER
    )
    assert status["valid"] is True
    assert status["remaining_calls"] == 1
    assert status["artifacts"][0]["data_type"] == "text"

    wrong_provider = validate_model_connector_consent(
        consent["consent_path"],
        provider_config={**PROVIDER, "model": "other-model"},
    )
    assert wrong_provider["valid"] is False
    assert any(
        row["key"] == "consent_provider_model_mismatch"
        for row in wrong_provider["blockers"]
    )


def test_artifact_change_invalidates_consent(tmp_path: Path) -> None:
    source = tmp_path / "transcript.txt"
    source.write_text("before", encoding="utf-8")
    consent = create_model_connector_consent(
        tmp_path,
        task="transcript_readable_polish",
        artifact_paths=[source],
        provider_config=PROVIDER,
        max_estimated_cost_usd=1.0,
        confirm_data_export=True,
    )
    source.write_text("after", encoding="utf-8")
    status = validate_model_connector_consent(
        consent["consent_path"], provider_config=PROVIDER
    )
    assert status["valid"] is False
    assert any(row["key"] == "artifact_changed" for row in status["blockers"])


def test_execute_consent_reads_only_authorised_text_and_consumes_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "chapter.md"
    source.write_text("Chapter evidence", encoding="utf-8")
    consent = create_model_connector_consent(
        tmp_path,
        task="smart_summary_section_rewrite",
        artifact_paths=[source],
        provider_config=PROVIDER,
        instructions="Summarise this chapter.",
        max_calls=1,
        max_estimated_cost_usd=1.0,
        confirm_data_export=True,
    )
    captured: dict[str, object] = {}

    def fake_call(task: str, **kwargs: object) -> dict[str, object]:
        captured.update({"task": task, **kwargs})
        return {"ok": True, "status": "completed", "response": {"content": "Summary"}}

    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.model_task_api_call",
        fake_call,
    )
    result = execute_consented_model_task(
        consent["consent_path"], provider_config=PROVIDER
    )
    assert result["ok"] is True
    assert result["transport_ok"] is True
    assert result["contract_ok"] is True
    assert result["quality_gate_passed"] is True
    assert result["production_qualified"] is True
    assert captured["task"] == "smart_summary_section_rewrite"
    assert "Chapter evidence" in str(captured["input_text"])
    assert captured["prompt"] == "Summarise this chapter."
    assert result["usage"]["calls_attempted"] == 1
    assert result["usage"]["calls_completed"] == 1
    assert Path(result["artifacts"]["execution_report"]).is_file()

    second = execute_consented_model_task(
        consent["consent_path"], provider_config=PROVIDER
    )
    assert second["status"] == "consent_required"
    assert any(
        row["key"] == "consent_call_limit_exceeded"
        for row in second["consent"]["blockers"]
    )


def test_global_reduce_forwards_prepared_messages_and_generation_controls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    messages = [
        {"role": "system", "content": "Return one mature summary as JSON."},
        {"role": "user", "content": "Use only the supplied chapter fact packs."},
    ]
    source = tmp_path / "smart-summary-global-reduce.json"
    source.write_text(
        json.dumps(
            {
                "schema": (
                    "video_knowledge_pipeline."
                    "smart_summary_global_reduce_request.v1"
                ),
                "task": "smart_summary_global_reduce",
                "messages": messages,
                "generation_parameters": {
                    "temperature": 0.0,
                    "max_tokens": 5000,
                    "response_format": "json_object",
                },
                "output_contract": {
                    "format": "json",
                    "required_keys": {"schema": "string"},
                },
            }
        ),
        encoding="utf-8",
    )
    consent = create_model_connector_consent(
        tmp_path,
        task="smart_summary_global_reduce",
        artifact_paths=[source],
        provider_config=PROVIDER,
        instructions="Execute the prepared Reduce request.",
        output_contract={
            "format": "json",
            "required_keys": {"schema": "string"},
        },
        max_calls=1,
        max_estimated_cost_usd=1.0,
        confirm_data_export=True,
    )
    captured: dict[str, object] = {}

    def fake_call(task: str, **kwargs: object) -> dict[str, object]:
        captured.update({"task": task, **kwargs})
        return {
            "ok": True,
            "status": "completed",
            "content": {
                "schema": "video_knowledge_pipeline.smart_summary_reader_plan.v1"
            },
        }

    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.model_task_api_call",
        fake_call,
    )
    result = execute_consented_model_task(
        consent["consent_path"], provider_config=PROVIDER
    )

    assert result["ok"] is True
    assert captured["task"] == "smart_summary_global_reduce"
    assert captured["messages"] == messages
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 5000
    assert captured["response_format"] == {"type": "json_object"}
    assert "input_text" not in captured
    assert Path(result["artifacts"]["execution_report"]).is_file()


def test_explicit_retry_limit_reserves_worst_case_provider_attempts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "chapter.md"
    source.write_text("Chapter evidence", encoding="utf-8")
    consent = create_model_connector_consent(
        tmp_path,
        task="smart_summary_section_rewrite",
        artifact_paths=[source],
        provider_config=PROVIDER,
        instructions="Summarise this chapter.",
        max_calls=2,
        max_estimated_cost_usd=2.0,
        max_retries_per_call=1,
        confirm_data_export=True,
    )
    captured: dict[str, object] = {}

    def fake_call(task: str, **kwargs: object) -> dict[str, object]:
        captured.update({"task": task, **kwargs})
        return {"ok": True, "status": "completed", "response": {"content": "Summary"}}

    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.model_task_api_call",
        fake_call,
    )
    result = execute_consented_model_task(
        consent["consent_path"], provider_config=PROVIDER
    )

    assert captured["max_retries"] == 1
    assert result["cost_control"]["provider_attempt_cap"] == 2
    assert result["usage"]["calls_attempted"] == 2
    assert result["usage"]["calls_completed"] == 1

def test_route_retry_conflict_does_not_reserve_consent_or_call_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "chapter.md"
    source.write_text("Chapter evidence", encoding="utf-8")
    consent = create_model_connector_consent(
        tmp_path,
        task="smart_summary_section_rewrite",
        artifact_paths=[source],
        provider_config=PROVIDER,
        max_calls=2,
        max_estimated_cost_usd=2.0,
        max_retries_per_call=1,
        confirm_data_export=True,
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector._resolve_execution_route",
        lambda *args, **kwargs: (
            dict(PROVIDER),
            {"retry_policy": {"max_retries": 0}},
        ),
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.model_task_api_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("provider must not be called")
        ),
    )

    result = execute_consented_model_task(
        consent["consent_path"], provider_config=PROVIDER
    )
    stored = json.loads(Path(consent["consent_path"]).read_text(encoding="utf-8"))

    assert result["status"] == "retry_policy_conflict"
    assert result["remote_requests_made"] is False
    assert result["consent_reserved"] is False
    assert stored["usage"]["calls_attempted"] == 0

def test_consent_locked_output_contract_can_fail_after_transport_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "chapter.md"
    source.write_text("Chapter evidence", encoding="utf-8")
    consent = create_model_connector_consent(
        tmp_path,
        task="smart_summary_section_rewrite",
        artifact_paths=[source],
        provider_config=PROVIDER,
        output_contract={
            "format": "json",
            "required_keys": {"title": "string"},
        },
        max_estimated_cost_usd=1.0,
        confirm_data_export=True,
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.model_task_api_call",
        lambda *args, **kwargs: {
            "ok": True,
            "status": "completed",
            "content": {"summary": "transport succeeded"},
        },
    )

    result = execute_consented_model_task(
        consent["consent_path"],
        provider_config=PROVIDER,
        write=False,
    )

    assert result["ok"] is True
    assert result["transport_ok"] is True
    assert result["contract_ok"] is False
    assert result["quality_gate_passed"] is False
    assert result["production_qualified"] is False
    assert result["output_validation"]["contract_issues"] == [
        {"key": "missing_required_key", "detail": "title"}
    ]


def test_semantic_pack_consent_is_hardened_and_deep_validation_blocks_bad_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pack = tmp_path / "transcript-semantic-correction-pack.json"
    pack.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.transcript_semantic_correction_pack.v1",
                "candidates": [
                    {
                        "candidate_id": "semcorr-1",
                        "correction_type": "ordinary_word",
                        "original_text": "和",
                        "evidence": [
                            {
                                "evidence_id": "asr-1",
                                "source_type": "asr_or_subtitle",
                                "text": "和",
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    consent = create_model_connector_consent(
        tmp_path,
        task="transcript_correction_pack",
        artifact_paths=[pack],
        provider_config=PROVIDER,
        instructions="Review candidates.",
        output_contract={
            "format": "json",
            "required_keys": {
                "schema": "string",
                "source": "string",
                "decisions": "array",
            },
        },
        max_estimated_cost_usd=1.0,
        confirm_data_export=True,
    )
    assert "VKP_STRICT_TRANSCRIPT_SEMANTIC_CORRECTION_V1" in consent["instructions"]
    assert consent["output_contract"]["array_item_contracts"]["decisions"][
        "required_keys"
    ]["confidence"] == "number"

    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.model_task_api_call",
        lambda *args, **kwargs: {
            "ok": True,
            "status": "completed",
            "content": {
                "schema": "video_knowledge_pipeline.transcript_semantic_correction_result.v1",
                "source": "online_llm_review",
                "decisions": [
                    {
                        "candidate_id": "semcorr-1",
                        "original_text": "and",
                        "accept": False,
                        "needs_human_review": True,
                        "reason": "uncertain",
                    }
                ],
            },
        },
    )

    result = execute_consented_model_task(
        consent["consent_path"],
        provider_config=PROVIDER,
        write=False,
    )

    assert result["ok"] is True
    assert result["transport_ok"] is True
    assert result["contract_ok"] is False
    assert result["quality_gate_passed"] is False
    assert result["production_qualified"] is False
    deep = result["output_validation"]["task_specific_validation"]
    assert deep["rejected_decision_count"] == 1
    assert any(row["key"] == "original_text_mismatch" for row in deep["quality_issues"])


def test_connector_allows_coding_plan_for_consented_text_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "chapter.md"
    source.write_text("Chapter evidence", encoding="utf-8")
    provider = {
        "provider": "volcengine_coding_plan",
        "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
        "model": "deepseek-v4-pro",
    }
    consent = create_model_connector_consent(
        tmp_path,
        task="smart_summary_rewrite",
        artifact_paths=[source],
        provider_config=provider,
        max_calls=1,
        max_estimated_cost_usd=1.0,
        confirm_data_export=True,
    )
    called = False

    def forbidden(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("provider call must be blocked")

    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.model_task_api_call",
        forbidden,
    )
    result = execute_consented_model_task(
        consent["consent_path"],
        provider_config=provider,
        write=False,
    )
    status = validate_model_connector_consent(
        consent["consent_path"],
        provider_config=provider,
    )

    assert result["status"] != "provider_usage_scope_blocked"
    assert "blocked_deployments" not in result
    assert status["usage"]["calls_attempted"] == 1
    assert called is True


def test_connector_blocks_unhealthy_proxy_before_consent_reservation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "chapter.md"
    source.write_text("Chapter evidence", encoding="utf-8")
    route = {
        "route_id": "pool-remote-text",
        "route_revision": "a" * 64,
        "virtual_model": "vkp-remote-text-pool-remote-text-aaaaaaaaaaaa",
        "execution_location": "remote",
        "deployments": [
            {
                "id": "remote-a",
                "provider": "openai_compatible",
                "model": "model-a",
                "base_url": "https://example.invalid/v1",
                "interface": "openai_compatible",
                "adapter_backend": "proxy",
            }
        ],
    }
    consent = create_model_connector_consent(
        tmp_path,
        task="smart_summary_rewrite",
        artifact_paths=[source],
        route_snapshot=route,
        max_calls=1,
        max_estimated_cost_usd=1.0,
        confirm_data_export=True,
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector._resolve_execution_route",
        lambda *args, **kwargs: (
            {
                "provider": "openai_compatible",
                "base_url": "https://example.invalid/v1",
                "model": "model-a",
            },
            route,
        ),
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.model_gateway_runtime_readiness",
        lambda: {
            "status": "gateway_unavailable",
            "ready": False,
            "remote_requests_made": False,
        },
    )
    called = False

    def forbidden(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("model call must not run while the local Proxy is unavailable")

    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.model_task_api_call",
        forbidden,
    )
    result = execute_consented_model_task(
        consent["consent_path"],
        expected_route_revision=str(route["route_revision"]),
        write=False,
    )
    status = validate_model_connector_consent(
        consent["consent_path"],
        route_snapshot=route,
        expected_route_revision=str(route["route_revision"]),
    )
    assert result["status"] == "gateway_unavailable"
    assert result["remote_requests_made"] is False
    assert result["consent_reserved"] is False
    assert status["usage"]["calls_attempted"] == 0
    assert called is False


def test_connector_rejects_inline_secrets(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("text", encoding="utf-8")
    with pytest.raises(ValueError, match="must not include secrets"):
        create_model_connector_consent(
            tmp_path,
            task="smart_summary_rewrite",
            artifact_paths=[source],
            provider_config={**PROVIDER, "api_key": "not-allowed"},
            max_estimated_cost_usd=1.0,
            confirm_data_export=True,
        )


def test_saved_route_redacts_dpapi_runtime_secret_before_policy_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from video_knowledge_pipeline.trusted_model_connector import (
        _resolve_execution_route,
    )

    consent_path = tmp_path / "consent.json"
    consent_path.write_text(
        json.dumps(
            {
                "task": "temporal_visual_analysis",
                "model_type": "temporal_sequence",
            }
        ),
        encoding="utf-8",
    )
    route = {
        "route_id": "remote-vision",
        "route_revision": "a" * 64,
        "virtual_model": "vkp-remote-vision",
        "execution_location": "remote",
        "deployments": [],
    }
    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.resolve_model_api_route",
        lambda *args, **kwargs: route,
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.resolve_model_api_provider_config",
        lambda *args, **kwargs: {
            "provider": "gemini",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "model": "gemini-3.5-flash",
            "adapter_backend": "legacy",
            "api_key": "locally-decrypted-runtime-value",
            "api_key_source": "local_dpapi",
        },
    )

    config, resolved_route = _resolve_execution_route(
        consent_path,
        provider_config=None,
        expected_route_revision="a" * 64,
    )

    assert resolved_route == route
    assert config["provider"] == "gemini"
    assert "api_key" not in config
    assert "api_key_source" not in config


def test_consent_enforces_task_modality(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-a-real-image")
    with pytest.raises(ValueError, match="text tasks accept text artifacts only"):
        create_model_connector_consent(
            tmp_path,
            task="smart_summary_rewrite",
            artifact_paths=[image],
            provider_config=PROVIDER,
            max_estimated_cost_usd=1.0,
            confirm_data_export=True,
        )


def test_capabilities_cover_text_vision_and_asr() -> None:
    result = trusted_model_connector_capabilities()
    assert result["status"] == "ready"
    assert {"text", "image", "multi_image", "audio"}.issubset(set(result["modalities"]))
    assert result["consent_contract"]["mcp_can_create_consent"] is False
    assert result["consent_contract"]["temporal_group_call_reservation"] is True
    assert result["consent_contract"]["explicit_per_file_upload_manifest"] is True
    assert result["consent_contract"]["atomic_call_and_cost_reservation"] is True
    assert result["consent_contract"]["v1_remote_execution_allowed"] is False
    assert result["provider_catalog"]["provider_count"] >= 30
    tasks = {row["task"]: row for row in result["tasks"]}
    assert "groq_asr" in tasks["cloud_asr"]["provider_profile_ids"]
    assert "openai_compatible" in tasks["cloud_asr"]["provider_profile_ids"]
    assert "speaches_openai_compatible" in tasks["cloud_asr"]["provider_profile_ids"]
    assert tasks["online_ocr"]["catalog_capability"] == "ocr"
    assert "mistral" in tasks["online_ocr"]["provider_profile_ids"]
    assert "openai" in tasks["smart_summary_rewrite"]["provider_profile_ids"]
    assert (
        result["operator_boundary"]["does_not_override_agent_platform_policy"] is True
    )
    assert result["operator_boundary"]["automatic_publish_allowed"] is False
    assert result["operator_boundary"]["unlisted_file_upload_allowed"] is False
    assert result["operator_boundary"]["silent_local_cloud_fallback_allowed"] is False


def test_bundle_vision_reuses_existing_analyzers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_runner(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"ok": True, "status": "completed"}

    monkeypatch.setattr(
        "video_knowledge_pipeline.trusted_model_connector.run_multimodal_frame_analysis",
        fake_runner,
    )
    result = execute_consented_bundle_vision(
        tmp_path,
        mode="semantic",
        indexes=[5, 3, 5],
        export_consent=tmp_path / "vision-export-consent.json",
        provider_config=PROVIDER,
    )
    assert result["ok"] is True
    assert captured["indexes"] == [3, 5]
    assert captured["confirm_vision_calls"] == 2
    assert captured["confirm_vision_indexes"] == "3,5"
    assert captured["execution_actor"] == "trusted_model_connector"


def test_connector_mcp_does_not_expose_consent_creation() -> None:
    source = Path(
        "src/video_knowledge_pipeline/trusted_model_connector_mcp.py"
    ).read_text(encoding="utf-8")
    assert "create_model_connector_consent" not in source
    assert "confirm_data_export" not in source
