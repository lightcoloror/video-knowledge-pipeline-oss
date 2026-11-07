from __future__ import annotations

import json
from pathlib import Path

import video_knowledge_pipeline.transcript_semantic_correction as semantic
import video_knowledge_pipeline.transcript_semantic_batch as semantic_batch
from video_knowledge_pipeline.model_business_authorization import (
    create_model_business_authorization,
)
from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.transcript_semantic_correction import (
    build_transcript_semantic_correction_llm_draft,
    build_transcript_semantic_correction_pack,
    validate_transcript_semantic_correction,
)
from video_knowledge_pipeline.trusted_model_connector_policy import (
    TrustedModelConnectorPolicy,
)


def _bundle(root: Path) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    write_json(
        bundle / "manifest.json",
        {"normalized_transcript_json": "normalized-transcript.json"},
    )
    write_json(
        bundle / "normalized-transcript.json",
        {"segments": [{"start": 0, "end": 4, "text": "这里讲 titok 平台"}]},
    )
    write_json(
        bundle / "timeline.json",
        [{"index": 0, "start": 0, "end": 4, "visual_text": "TikTok platform"}],
    )
    return bundle


def test_remote_semantic_correction_uses_parent_child_consent(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = _bundle(tmp_path)
    build_transcript_semantic_correction_pack(bundle, write=True)
    transcript = bundle / "normalized-transcript.json"
    pack = bundle / "transcript-semantic-correction-pack.json"
    authorization = tmp_path / "semantic-parent.json"
    policy = TrustedModelConnectorPolicy(
        (tmp_path.resolve(),), frozenset({"api.example"})
    )
    route = {
        "route_id": "remote-transcript-correction",
        "route_revision": "b" * 64,
        "virtual_model": "vkp-remote-transcript-correction",
        "execution_location": "remote",
        "deployments": [
            {
                "provider": "fixture",
                "model": "fixture-model",
                "base_url": "https://api.example/v1",
                "interface": "openai_compatible",
            }
        ],
    }
    create_model_business_authorization(
        tmp_path,
        bundle_dir=bundle,
        source_paths=[transcript, pack],
        stages=[
            {
                "id": "semantic-correction",
                "task": "transcript_semantic_correction",
                "route_snapshot": route,
                "allowed_producers": [
                    "transcript_semantic_correction_gateway_pack"
                ],
                "max_calls": 1,
                "max_estimated_cost_usd": 0.1,
                "max_cost_per_call_usd": 0.1,
                "max_retries_per_call": 0,
                "max_artifacts": 1,
                "max_total_bytes": 524288,
                "max_artifacts_per_child": 1,
                "max_bytes_per_child": 524288,
                "instructions": "Return strict JSON.",
            }
        ],
        purpose="semantic correction remote fixture",
        max_calls=1,
        max_estimated_cost_usd=0.1,
        confirm_data_export=True,
        output_path=authorization,
        policy=policy,
    )
    cfg = {
        "provider": "openai_compatible",
        "adapter_backend": "proxy",
        "execution_location": "remote",
        "base_url": "http://127.0.0.1:18776/v1",
        "model": "vkp-remote-transcript-correction",
        "route_id": route["route_id"],
        "route_revision": route["route_revision"],
    }
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        semantic, "resolve_model_api_provider_config", lambda _task, _explicit: dict(cfg)
    )
    monkeypatch.setattr(
        semantic,
        "create_business_child_consent",
        lambda path, **kwargs: captured.update({"path": path, **kwargs})
        or {
            "status": "child_consent_created",
            "consent_path": str(bundle / "child-consent.json"),
            "consent_id": "child-1",
            "route_revision": route["route_revision"],
            "admission_id": "admission-1",
        },
    )
    monkeypatch.setattr(
        semantic,
        "execute_consented_model_task",
        lambda _path, **_kwargs: {
            "ok": True,
            "status": "completed",
            "model_result": {"content": json.dumps({"decisions": []})},
        },
    )
    monkeypatch.setattr(
        semantic,
        "model_task_api_call",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("remote proxy must use a child consent")
        ),
    )

    result = build_transcript_semantic_correction_llm_draft(
        bundle,
        execute=True,
        limit=1,
        business_authorization_path=authorization,
        write=True,
    )

    assert result["business_authorization"]["execution_mode"] == "business_child_consent"
    assert result["business_authorization"]["connector_status"] == "completed"
    assert captured["stage_id"] == "semantic-correction"
    assert captured["producer"] == "transcript_semantic_correction_gateway_pack"
    assert captured["input_paths"] == [str(transcript), str(pack)]


def test_batch_llm_action_forwards_one_parent_authorization(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = _bundle(tmp_path)
    parent = tmp_path / "shared-parent.json"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        semantic,
        "build_transcript_semantic_correction_llm_draft",
        lambda root, **kwargs: captured.update({"root": root, **kwargs})
        or {"ok": True, "status": "completed"},
    )

    result = semantic_batch._execute_safe_queue_action(
        bundle,
        "execute_llm_or_use_codex",
        allow_closure=False,
        allow_llm=True,
        provider_config=None,
        llm_limit=17,
        business_authorization_path=parent,
    )

    assert result["status"] == "completed"
    assert captured["root"] == bundle
    assert captured["business_authorization_path"] == parent
    assert captured["provider_config"] is None
    assert captured["limit"] == 17


def test_empty_decisions_are_qualified_only_for_empty_eligible_selection() -> None:
    pack = {
        "schema": semantic.PACK_SCHEMA,
        "candidates": [],
        "candidate_selection": {"selected_candidate_count": 0},
    }
    payload = {
        "schema": semantic.RESULT_SCHEMA,
        "source": "text_llm_semantic_review",
        "decisions": [],
    }

    allowed = semantic.validate_transcript_semantic_model_output(
        payload, pack, allow_empty_decisions=True
    )
    blocked = semantic.validate_transcript_semantic_model_output(payload, pack)

    assert allowed["quality_gate_passed"] is True
    assert blocked["quality_gate_passed"] is False
    assert blocked["quality_issues"][0]["key"] == "semantic_decisions_empty"


def test_downstream_validation_accepts_explicit_empty_eligible_selection(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    build_transcript_semantic_correction_pack(bundle, write=True)
    result_path = bundle / "empty-result.json"
    write_json(
        result_path,
        {
            "schema": semantic.RESULT_SCHEMA,
            "source": "text_llm_semantic_review",
            "decisions": [],
            "candidate_selection": {"selected_candidate_count": 0},
        },
    )

    result = validate_transcript_semantic_correction(bundle, input_json=result_path)

    assert result["status"] == "no_eligible_candidates"
    assert result["ok"] is True
