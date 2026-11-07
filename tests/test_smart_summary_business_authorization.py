from __future__ import annotations

import hashlib
import json
from pathlib import Path

import video_knowledge_pipeline.smart_summary_section_llm as section_llm
from video_knowledge_pipeline.model_business_authorization import (
    create_model_business_authorization,
    validate_model_business_authorization,
)
from video_knowledge_pipeline.trusted_model_connector_policy import TrustedModelConnectorPolicy


def _bundle(root: Path) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    transcript = bundle / "source-arbitrated-transcript.json"
    transcript.write_text(
        json.dumps(
            {"segments": [{"start": 0, "end": 12, "text": "课程讲解客户画像、信任建立与后续行动。"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "manifest.json").write_text(
        json.dumps(
            {"schema": "lecture_webui_bundle.v1", "title": "授权摘要", "source_arbitrated_transcript_json": transcript.name},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (bundle / "timeline.json").write_text(
        json.dumps(
            [{"index": 1, "start": 0, "end": 12, "transcript": "课程讲解客户画像、信任建立与后续行动。", "visual_route": "semantic_frame"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return bundle


def _remote_config() -> dict[str, str]:
    return {
        "provider": "openai_compatible",
        "adapter_backend": "proxy",
        "execution_location": "remote",
        "base_url": "http://127.0.0.1:18776/v1",
        "model": "vkp-remote-text-route",
        "route_id": "pool-remote-text",
        "route_revision": "a" * 64,
    }


def _patch_route_resolution(monkeypatch) -> None:
    monkeypatch.setattr(section_llm, "resolve_model_api_provider_config", lambda _task, explicit=None: dict(explicit or {}))
    monkeypatch.setattr(section_llm, "resolve_text_provider_config", lambda config: dict(config))


def test_remote_proxy_requires_business_authorization_before_any_direct_call(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)
    _patch_route_resolution(monkeypatch)

    def fail_direct(*args, **kwargs):
        raise AssertionError("remote proxy must not use the direct summary adapter")

    monkeypatch.setattr(section_llm, "call_openai_compatible_text", fail_direct)

    result = section_llm.run_smart_summary_section_llm_rewrite(
        bundle,
        provider_config=_remote_config(),
        execute=True,
        limit=1,
    )

    assert result["status"] == "business_authorization_required"
    assert result["ok"] is False


def test_remote_proxy_uses_business_child_consent_and_connector_reservation(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)
    _patch_route_resolution(monkeypatch)
    transcript = bundle / "source-arbitrated-transcript.json"
    authorization = bundle / "business-authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "stages": [
                    {
                        "id": "summary-sections",
                        "task": "smart_summary_section_rewrite",
                        "allowed_producers": ["smart_summary_input_pack"],
                        "route_snapshot": {"route_id": "pool-remote-text", "route_revision": "a" * 64},
                    }
                ],
                "sources": [{"path": str(transcript)}],
                "admissions": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(section_llm, "validate_model_business_authorization", lambda _path: {"valid": True, "bundle_dir": str(bundle)})

    def fake_child(path, **kwargs):
        captured["authorization_path"] = path
        captured.update(kwargs)
        return {"status": "child_consent_created", "consent_path": str(bundle / "child-consent.json"), "consent_id": "child-1", "route_revision": "a" * 64}

    def fake_connector(consent_path, **kwargs):
        captured["connector_consent_path"] = consent_path
        captured["connector_kwargs"] = kwargs
        return {
            "ok": True,
            "status": "completed",
            "model_result": {"content": "### 课程核心\n\n本节说明客户画像与信任建立的关键方法，并给出后续行动建议。\n\n- 关键动作：先确认客户顾虑，再记录下一步跟进。\n- 证据边界：不编造未出现的画面。"},
        }

    monkeypatch.setattr(section_llm, "create_business_child_consent", fake_child)
    monkeypatch.setattr(section_llm, "execute_consented_model_task", fake_connector)
    monkeypatch.setattr(section_llm, "call_openai_compatible_text", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("direct adapter is forbidden")))

    result = section_llm.run_smart_summary_section_llm_rewrite(
        bundle,
        provider_config=_remote_config(),
        execute=True,
        business_authorization_path=authorization,
        limit=1,
        install=False,
        require_all_sections=False,
        min_section_chars=40,
    )

    assert result["status"] == "completed"
    assert captured["stage_id"] == "summary-sections"
    assert captured["producer"] == "smart_summary_input_pack"
    assert captured["connector_kwargs"] == {"expected_route_revision": "a" * 64, "write": True}
    assert Path(captured["artifact_paths"][0]).is_file()
    assert result["calls"][0]["connector_status"] == "completed"


def test_summary_context_accepts_second_bundle_bound_by_one_parent(tmp_path: Path, monkeypatch) -> None:
    first_bundle = _bundle(tmp_path)
    second_root = tmp_path / "second"
    second_root.mkdir()
    second_bundle = _bundle(second_root)
    first_transcript = first_bundle / "source-arbitrated-transcript.json"
    second_transcript = second_bundle / "source-arbitrated-transcript.json"
    authorization = tmp_path / "batch-business-authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "stages": [{"id": "summary-sections", "task": "smart_summary_section_rewrite", "allowed_producers": ["smart_summary_input_pack"], "route_snapshot": {"route_id": "pool-remote-text", "route_revision": "a" * 64}}],
                "sources": [{"path": str(first_transcript)}, {"path": str(second_transcript)}],
                "admissions": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        section_llm,
        "validate_model_business_authorization",
        lambda _path: {"valid": True, "bundle_dir": str(first_bundle), "bundle_dirs": [str(first_bundle), str(second_bundle)]},
    )

    context = section_llm._prepare_business_summary_context(
        second_bundle, _remote_config(), authorization
    )

    assert context["stage_id"] == "summary-sections"
    assert context["lineage_input_paths"] == [str(second_transcript)]


def test_summary_preflight_keeps_parent_active_when_generated_pack_is_refreshed(
    tmp_path: Path, monkeypatch
) -> None:
    """A local rewrite preflight may refresh pack diagnostics, not its evidence lineage."""
    bundle = _bundle(tmp_path)
    transcript = bundle / "source-arbitrated-transcript.json"
    pack = bundle / "exports" / "smart-summary-input-pack.json"
    pack.parent.mkdir()

    def write_pack(created_at: str) -> None:
        pack.write_text(
            json.dumps(
                {
                    "schema": "video_knowledge_pipeline.smart_summary_input_pack.v1",
                    "created_at": created_at,
                    "transcript_source": str(transcript),
                    "transcript_source_sha256": hashlib.sha256(
                        transcript.read_bytes()
                    ).hexdigest(),
                    "quality": {"preflight_snapshot": created_at},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    write_pack("before-parent-authorization")
    authorization = tmp_path / "summary-parent.json"
    policy = TrustedModelConnectorPolicy(
        (tmp_path.resolve(),), frozenset({"api.example"})
    )
    create_model_business_authorization(
        tmp_path,
        bundle_dir=bundle,
        source_paths=[transcript, pack],
        stages=[
            {
                "id": "summary-sections",
                "task": "smart_summary_section_rewrite",
                "route_snapshot": {
                    "route_id": "pool-remote-text",
                    "route_revision": "a" * 64,
                    "virtual_model": "vkp-remote-text-route",
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
                "instructions": "Return Chinese Markdown.",
            }
        ],
        purpose="summary preflight stable pack regression",
        max_calls=1,
        max_estimated_cost_usd=0.1,
        confirm_data_export=True,
        output_path=authorization,
        policy=policy,
    )

    # Mimic the workflow's local quality/workflow refresh before remote admission.
    write_pack("rebuilt-by-preflight")
    _patch_route_resolution(monkeypatch)
    result = section_llm.run_smart_summary_section_llm_rewrite(
        bundle,
        provider_config=_remote_config(),
        execute=True,
        section_ids=["section-not-present"],
        business_authorization_path=authorization,
        install=False,
        require_all_sections=False,
    )

    assert result["selected_section_count"] == 0
    assert result["business_authorization"]["execution_mode"] == "business_child_consent"
    # No section was selected, therefore this run has no model call to make.
    assert result["calls"] == []
    assert validate_model_business_authorization(authorization, policy=policy)["valid"] is True
