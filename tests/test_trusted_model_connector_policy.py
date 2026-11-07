from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.model_connector_consent import (
    create_model_connector_consent,
)
from video_knowledge_pipeline.trusted_model_connector_policy import (
    TrustedModelConnectorPolicy,
)
from video_knowledge_pipeline.trusted_model_connector_remote_mcp import (
    _locked_route_revision,
    _normalise_http_path,
    _require_loopback_host,
    build_server,
)


LOCAL_PROVIDER = {
    "provider": "local_qwen_vl",
    "base_url": "http://127.0.0.1:8000/v1",
    "model": "Qwen/Qwen2.5-VL-3B-Instruct",
}
REMOTE_PROVIDER = {
    "provider": "volcengine_coding_plan",
    "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
    "model": "doubao-seed-2.0-pro",
}


def test_policy_allows_only_configured_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    inside = allowed / "consent.json"
    inside.write_text("{}", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    policy = TrustedModelConnectorPolicy((allowed.resolve(),), frozenset())

    assert policy.require_path(inside, label="inside") == inside.resolve()
    with pytest.raises(ValueError, match="outside VKP_MODEL_CONNECTOR_ALLOWED_ROOTS"):
        policy.require_path(outside, label="outside")


def test_policy_allows_local_models_without_remote_allowlist(tmp_path: Path) -> None:
    policy = TrustedModelConnectorPolicy((tmp_path.resolve(),), frozenset())
    provider = policy.require_provider_destination(
        "multimodal_frame_analysis", LOCAL_PROVIDER
    )
    assert provider["base_url"] == LOCAL_PROVIDER["base_url"]


def test_policy_requires_explicit_remote_destination(tmp_path: Path) -> None:
    blocked = TrustedModelConnectorPolicy((tmp_path.resolve(),), frozenset())
    with pytest.raises(ValueError, match="provider destination is not allowlisted"):
        blocked.require_provider_destination(
            "temporal_visual_analysis", REMOTE_PROVIDER
        )

    allowed = TrustedModelConnectorPolicy(
        (tmp_path.resolve(),),
        frozenset({"ark.cn-beijing.volces.com"}),
    )
    provider = allowed.require_provider_destination(
        "temporal_visual_analysis", REMOTE_PROVIDER
    )
    assert provider["model"] == "doubao-seed-2.0-pro"


def test_policy_checks_every_consent_artifact_root(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("sensitive", encoding="utf-8")
    consent = root / "consent.json"
    consent.write_text(
        json.dumps(
            {
                "task": "smart_summary_rewrite",
                "artifacts": [{"path": str(outside)}],
            }
        ),
        encoding="utf-8",
    )
    policy = TrustedModelConnectorPolicy(
        (root.resolve(),),
        frozenset({"ark.cn-beijing.volces.com"}),
    )
    with pytest.raises(ValueError, match="consent artifact is outside"):
        policy.require_consent_scope(consent, provider_config=REMOTE_PROVIDER)


def test_http_server_declares_read_and_execution_annotations(tmp_path: Path) -> None:
    policy = TrustedModelConnectorPolicy((tmp_path.resolve(),), frozenset())
    server = build_server(policy=policy, host="127.0.0.1", port=8766)
    tools = asyncio.run(server.list_tools())
    rows = {tool.name: tool for tool in tools}
    assert rows["model_connector_capabilities"].annotations.readOnlyHint is True
    assert rows["model_connector_consent_status"].annotations.openWorldHint is False
    assert rows["execute_consented_model_task_tool"].annotations.readOnlyHint is False
    assert rows["execute_consented_temporal_vision"].annotations.openWorldHint is True


def test_http_transport_rejects_non_loopback_bindings() -> None:
    _require_loopback_host("127.0.0.1")
    _require_loopback_host("::1")
    with pytest.raises(ValueError, match="loopback"):
        _require_loopback_host("0.0.0.0")
    assert _normalise_http_path("/mcp/") == "/mcp"
    with pytest.raises(ValueError, match="absolute path"):
        _normalise_http_path("mcp")


def test_mcp_execution_uses_consent_locked_route_revision_when_argument_is_omitted() -> None:
    consent = {"route": {"route_revision": "locked-revision"}}

    assert _locked_route_revision(consent) == "locked-revision"
    assert _locked_route_revision(consent, "locked-revision") == "locked-revision"


def test_mcp_execution_rejects_route_revision_drift() -> None:
    consent = {"route": {"route_revision": "locked-revision"}}

    with pytest.raises(ValueError, match="differs"):
        _locked_route_revision(consent, "stale-revision")
    with pytest.raises(ValueError, match="not locked"):
        _locked_route_revision({})


def test_policy_checks_every_v2_consent_destination(tmp_path: Path) -> None:
    artifact = tmp_path / "source.md"
    artifact.write_text("approved", encoding="utf-8")
    consent = tmp_path / "consent.json"
    consent.write_text(
        json.dumps(
            {
                "task": "smart_summary_rewrite",
                "authorized_deployments": [
                    {
                        "base_url": "https://a.example/v1",
                        "provider": "openai_compatible",
                    },
                    {
                        "base_url": "https://b.example/v1",
                        "provider": "openai_compatible",
                    },
                ],
                "artifacts": [{"path": str(artifact)}],
            }
        ),
        encoding="utf-8",
    )
    policy = TrustedModelConnectorPolicy(
        (tmp_path.resolve(),),
        frozenset({"a.example"}),
    )

    with pytest.raises(ValueError, match="b.example"):
        policy.require_consent_scope(consent)


def test_http_server_exposes_local_only_execution_tool(tmp_path: Path) -> None:
    policy = TrustedModelConnectorPolicy((tmp_path.resolve(),), frozenset())
    server = build_server(policy=policy, host="127.0.0.1", port=8766)
    tools = asyncio.run(server.list_tools())
    rows = {tool.name: tool for tool in tools}

    assert rows["execute_local_model_task_tool"].annotations.openWorldHint is True
    assert (
        "provider_config"
        not in rows["execute_consented_model_task_tool"].inputSchema["properties"]
    )
    assert (
        "route_revision"
        in rows["execute_consented_model_task_tool"].inputSchema["properties"]
    )
    assert (
        "route_revision"
        not in rows["execute_consented_model_task_tool"].inputSchema.get("required", [])
    )
    for tool_name in (
        "execute_consented_semantic_vision",
        "execute_consented_temporal_vision",
    ):
        properties = rows[tool_name].inputSchema["properties"]
        assert "provider_config" not in properties
        assert "route_revision" in properties


def test_policy_requires_complete_v2_contract_for_execution(tmp_path: Path) -> None:
    artifact = tmp_path / "source.md"
    artifact.write_text("approved", encoding="utf-8")
    consent = create_model_connector_consent(
        tmp_path,
        task="smart_summary_rewrite",
        artifact_paths=[artifact],
        provider_config={
            "provider": "custom_openai_compatible",
            "base_url": "https://example.invalid/v1",
            "model": "test-model",
        },
        max_estimated_cost_usd=0.1,
        confirm_data_export=True,
    )
    policy = TrustedModelConnectorPolicy(
        (tmp_path.resolve(),),
        frozenset({"example.invalid"}),
    )

    payload = policy.require_consent_scope(
        consent["consent_path"],
        require_execution_contract=True,
    )
    assert payload["upload_manifest"]["files"] == payload["artifacts"]

    consent_path = Path(consent["consent_path"])
    broken = json.loads(consent_path.read_text(encoding="utf-8"))
    broken.pop("operator_confirmation")
    consent_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="operator confirmation"):
        policy.require_consent_scope(
            consent_path,
            require_execution_contract=True,
        )
