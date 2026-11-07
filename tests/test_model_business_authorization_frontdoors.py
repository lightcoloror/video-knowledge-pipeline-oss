from __future__ import annotations

import asyncio
import json
from pathlib import Path

from video_knowledge_pipeline.cli import main
from video_knowledge_pipeline.consented_model_batch import ConsentedModelBatchManager
from video_knowledge_pipeline.trusted_model_connector_policy import (
    ALLOWED_DESTINATIONS_ENV,
    ALLOWED_ROOTS_ENV,
    TrustedModelConnectorPolicy,
)
from video_knowledge_pipeline.trusted_model_connector_remote_mcp import build_server


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    (bundle / "timeline.json").write_text("[]", encoding="utf-8")
    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.model_business_authorization_plan.v1",
                "root_dir": str(tmp_path),
                "bundle_dir": str(bundle),
                "source_paths": [str(source)],
                "purpose": "one visible confirmation",
                "expires_hours": 12,
                "scope": {
                    "max_calls": 1,
                    "max_estimated_cost_usd": 0.1,
                },
                "stages": [
                    {
                        "id": "summary",
                        "task": "smart_summary_rewrite",
                        "route_snapshot": {
                            "route_id": "summary-route",
                            "route_revision": "b" * 64,
                            "virtual_model": "vkp-remote-summary-test",
                            "execution_location": "remote",
                            "deployments": [
                                {
                                    "provider": "fixture",
                                    "model": "fixture-model",
                                    "base_url": "https://api.example/v1",
                                    "interface": "openai_compatible",
                                }
                            ],
                        },
                        "allowed_producers": ["smart_summary_input_pack"],
                        "max_calls": 1,
                        "max_estimated_cost_usd": 0.1,
                        "max_cost_per_call_usd": 0.1,
                        "max_retries_per_call": 0,
                        "max_artifacts": 1,
                        "max_total_bytes": 4096,
                        "max_artifacts_per_child": 1,
                        "max_bytes_per_child": 4096,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return plan, bundle, source


def test_cli_confirms_once_then_creates_child_without_confirmation_flag(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    plan, bundle, source = _fixture(tmp_path)
    authorization = tmp_path / "authorization.json"
    monkeypatch.setenv(ALLOWED_ROOTS_ENV, str(tmp_path))
    monkeypatch.setenv(ALLOWED_DESTINATIONS_ENV, "api.example")

    assert (
        main(
            [
                "model-business-authorization-create",
                str(plan),
                "--output-path",
                str(authorization),
                "--confirm-data-export",
            ]
        )
        == 0
    )
    created = json.loads(capsys.readouterr().out)
    assert created["status"] == "active"
    derived = bundle / "summary-input.json"
    derived.write_text("{}", encoding="utf-8")

    assert (
        main(
            [
                "model-business-child-consent",
                str(authorization),
                "--stage-id",
                "summary",
                "--producer",
                "smart_summary_input_pack",
                "--artifact",
                str(derived),
                "--lineage-input",
                str(source),
            ]
        )
        == 0
    )
    child = json.loads(capsys.readouterr().out)
    assert child["new_user_confirmation_required"] is False
    assert Path(child["consent_path"]).is_file()


def test_broker_exposes_narrow_child_consent_tool_without_provider_override(
    tmp_path: Path,
) -> None:
    policy = TrustedModelConnectorPolicy(
        (tmp_path.resolve(),), frozenset({"api.example"})
    )
    manager = ConsentedModelBatchManager(project_root=tmp_path, policy=policy)
    server = build_server(policy=policy, batch_manager=manager)
    rows = {tool.name: tool for tool in asyncio.run(server.list_tools())}

    status = rows["model_business_authorization_status_tool"]
    child = rows["create_business_child_consent_tool"]
    assert status.annotations.readOnlyHint is True
    assert child.annotations.readOnlyHint is False
    assert child.annotations.openWorldHint is False
    assert {
        "authorization_path",
        "stage_id",
        "artifact_paths",
        "producer",
        "lineage_input_paths",
    } <= set(child.inputSchema["required"])
    assert "provider_config" not in child.inputSchema["properties"]
    assert "base_url" not in child.inputSchema["properties"]
    assert "api_key" not in child.inputSchema["properties"]
