from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.model_business_authorization import (
    create_business_child_consent,
    create_model_business_authorization,
    find_reusable_model_business_authorization,
    validate_model_business_authorization,
)
from video_knowledge_pipeline.model_connector_consent import (
    validate_model_connector_consent,
)
from video_knowledge_pipeline.trusted_model_connector_policy import (
    TrustedModelConnectorPolicy,
)


def _bundle(root: Path) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text('{"title":"fixture"}', encoding="utf-8")
    (bundle / "timeline.json").write_text("[]", encoding="utf-8")
    return bundle


def _route() -> dict[str, object]:
    return {
        "route_id": "summary-route",
        "route_revision": "a" * 64,
        "virtual_model": "vkp-remote-summary-fixture",
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


def _stage(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "id": "summary",
        "task": "smart_summary_rewrite",
        "route_snapshot": _route(),
        "allowed_producers": ["smart_summary_input_pack"],
        "max_calls": 2,
        "max_estimated_cost_usd": 0.2,
        "max_cost_per_call_usd": 0.1,
        "max_retries_per_call": 0,
        "max_artifacts": 2,
        "max_total_bytes": 4096,
        "max_artifacts_per_child": 1,
        "max_bytes_per_child": 2048,
        "instructions": "Return Chinese Markdown.",
    }
    result.update(overrides)
    return result


def _authorization(
    root: Path,
    *,
    confirmed: bool = True,
    stage: dict[str, object] | None = None,
) -> tuple[Path, Path, Path, TrustedModelConnectorPolicy]:
    bundle = _bundle(root)
    source = root / "source.txt"
    source.write_text("exact source", encoding="utf-8")
    policy = TrustedModelConnectorPolicy(
        (root.resolve(),), frozenset({"api.example"})
    )
    path = root / "business-authorization.json"
    create_model_business_authorization(
        root,
        bundle_dir=bundle,
        source_paths=[source],
        stages=[stage or _stage()],
        purpose="one video online workflow",
        max_calls=2,
        max_estimated_cost_usd=0.2,
        confirm_data_export=confirmed,
        output_path=path,
        policy=policy,
    )
    return path, bundle, source, policy


def test_one_business_confirmation_mints_exact_child_consent_v2(
    tmp_path: Path,
) -> None:
    authorization, bundle, source, policy = _authorization(tmp_path)
    derived = bundle / "summary-input.json"
    derived.write_text('{"section":1}', encoding="utf-8")

    result = create_business_child_consent(
        authorization,
        stage_id="summary",
        artifact_paths=[derived],
        producer="smart_summary_input_pack",
        input_paths=[source],
        max_calls=1,
        policy=policy,
    )

    assert result["ok"] is True
    assert result["user_confirmation_reused"] is True
    assert result["new_user_confirmation_required"] is False
    child_path = Path(result["consent_path"])
    child = json.loads(child_path.read_text(encoding="utf-8"))
    assert child["schema"] == "video_knowledge_pipeline.model_connector_consent.v2"
    assert child["operator_confirmation"]["confirmation_method"] == "parent_business_authorization"
    assert child["artifacts"][0]["sha256"]
    assert child["upload_manifest"]["files"] == child["artifacts"]
    assert validate_model_connector_consent(
        child_path,
        route_snapshot=_route(),
        expected_route_revision="a" * 64,
    )["valid"] is True
    assert policy.require_consent_scope(
        child_path, require_execution_contract=True
    )["consent_id"] == child["consent_id"]
    parent_status = validate_model_business_authorization(
        authorization, policy=policy
    )
    assert parent_status["remaining_calls"] == 1
    assert parent_status["usage"]["artifacts_admitted"] == 1


def test_explicit_business_retry_limit_flows_to_child_consent(tmp_path: Path) -> None:
    authorization, bundle, source, policy = _authorization(
        tmp_path,
        stage=_stage(max_retries_per_call=1),
    )
    derived = bundle / "summary-input.json"
    derived.write_text('{"section": 1}', encoding="utf-8")

    result = create_business_child_consent(
        authorization,
        stage_id="summary",
        artifact_paths=[derived],
        producer="smart_summary_input_pack",
        input_paths=[source],
        max_calls=2,
        policy=policy,
    )

    child = json.loads(Path(result["consent_path"]).read_text(encoding="utf-8"))
    parent = json.loads(authorization.read_text(encoding="utf-8"))
    assert parent["scope"]["automatic_retry_allowed"] is True
    assert child["scope"]["max_retries_per_call"] == 1
    assert child["scope"]["max_calls"] == 2

def test_child_consent_creation_is_idempotent_for_same_lineage(tmp_path: Path) -> None:
    authorization, bundle, source, policy = _authorization(tmp_path)
    derived = bundle / "summary-input.json"
    derived.write_text('{"section":1}', encoding="utf-8")
    first = create_business_child_consent(
        authorization,
        stage_id="summary",
        artifact_paths=[derived],
        producer="smart_summary_input_pack",
        input_paths=[source],
        policy=policy,
    )
    second = create_business_child_consent(
        authorization,
        stage_id="summary",
        artifact_paths=[derived],
        producer="smart_summary_input_pack",
        input_paths=[source],
        policy=policy,
    )

    assert second["status"] == "existing_child_consent"
    assert second["consent_path"] == first["consent_path"]
    parent = json.loads(authorization.read_text(encoding="utf-8"))
    assert len(parent["admissions"]) == 1
    assert parent["usage"]["calls_authorized"] == 1


def test_unlinked_or_out_of_bundle_artifacts_are_rejected(tmp_path: Path) -> None:
    authorization, bundle, _source, policy = _authorization(tmp_path)
    derived = bundle / "summary-input.json"
    derived.write_text("{}", encoding="utf-8")
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("not an authorized lineage input", encoding="utf-8")

    with pytest.raises(ValueError, match="lineage"):
        create_business_child_consent(
            authorization,
            stage_id="summary",
            artifact_paths=[derived],
            producer="smart_summary_input_pack",
            input_paths=[unrelated],
            policy=policy,
        )
    with pytest.raises(ValueError, match="inside the bound Bundle"):
        create_business_child_consent(
            authorization,
            stage_id="summary",
            artifact_paths=[unrelated],
            producer="smart_summary_input_pack",
            input_paths=[_source],
            policy=policy,
        )


def test_parent_budget_and_stage_producer_are_hard_limits(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("exact source", encoding="utf-8")
    policy = TrustedModelConnectorPolicy(
        (tmp_path.resolve(),), frozenset({"api.example"})
    )
    authorization = tmp_path / "business-authorization.json"
    create_model_business_authorization(
        tmp_path,
        bundle_dir=bundle,
        source_paths=[source],
        stages=[_stage(max_calls=1, max_estimated_cost_usd=0.1)],
        purpose="one call",
        max_calls=1,
        max_estimated_cost_usd=0.1,
        confirm_data_export=True,
        output_path=authorization,
        policy=policy,
    )
    first = bundle / "first.txt"
    second = bundle / "second.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    with pytest.raises(ValueError, match="producer"):
        create_business_child_consent(
            authorization,
            stage_id="summary",
            artifact_paths=[first],
            producer="arbitrary_agent",
            input_paths=[source],
            policy=policy,
        )
    create_business_child_consent(
        authorization,
        stage_id="summary",
        artifact_paths=[first],
        producer="smart_summary_input_pack",
        input_paths=[source],
        policy=policy,
    )
    with pytest.raises(ValueError, match="call limit"):
        create_business_child_consent(
            authorization,
            stage_id="summary",
            artifact_paths=[second],
            producer="smart_summary_input_pack",
            input_paths=[source],
            policy=policy,
        )


def test_source_change_or_missing_confirmation_blocks_children(tmp_path: Path) -> None:
    authorization, bundle, source, policy = _authorization(tmp_path)
    source.write_text("changed", encoding="utf-8")
    status = validate_model_business_authorization(authorization, policy=policy)
    assert status["valid"] is False
    assert any(row["key"] == "authorization_source_changed" for row in status["blockers"])

    other_root = tmp_path / "other"
    other_root.mkdir()
    pending, pending_bundle, pending_source, pending_policy = _authorization(
        other_root, confirmed=False
    )
    derived = pending_bundle / "summary-input.json"
    derived.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not confirmed"):
        create_business_child_consent(
            pending,
            stage_id="summary",
            artifact_paths=[derived],
            producer="smart_summary_input_pack",
            input_paths=[pending_source],
            policy=pending_policy,
        )

def test_one_parent_authorization_covers_two_bundles_with_six_exact_children(tmp_path: Path) -> None:
    first_bundle = _bundle(tmp_path)
    second_bundle = tmp_path / "bundle-two"
    second_bundle.mkdir()
    (second_bundle / "manifest.json").write_text('{"title":"second"}', encoding="utf-8")
    (second_bundle / "timeline.json").write_text("[]", encoding="utf-8")
    first_source = first_bundle / "transcript.txt"
    first_courseware = first_bundle / "companion-courseware.md"
    second_source = second_bundle / "transcript.txt"
    second_courseware = second_bundle / "companion-courseware.md"
    for path, text in ((first_source, "first transcript"), (first_courseware, "first courseware"), (second_source, "second transcript"), (second_courseware, "second courseware")):
        path.write_text(text, encoding="utf-8")
    policy = TrustedModelConnectorPolicy((tmp_path.resolve(),), frozenset({"api.example"}))
    authorization = tmp_path / "batch-business-authorization.json"
    create_model_business_authorization(
        tmp_path,
        bundle_dir=first_bundle,
        bundle_dirs=[second_bundle],
        source_paths=[first_source, first_courseware, second_source, second_courseware],
        stages=[_stage(max_calls=6, max_estimated_cost_usd=0.6, max_artifacts=6, max_total_bytes=8192)],
        purpose="two videos with companion courseware and six chapter summaries",
        max_calls=6,
        max_estimated_cost_usd=0.6,
        confirm_data_export=True,
        output_path=authorization,
        policy=policy,
    )

    for bundle, transcript, courseware in ((first_bundle, first_source, first_courseware), (second_bundle, second_source, second_courseware)):
        for chapter in range(1, 4):
            request = bundle / f"summary-request-{chapter}.json"
            request.write_text(json.dumps({"chapter": chapter}), encoding="utf-8")
            child = create_business_child_consent(
                authorization,
                stage_id="summary",
                artifact_paths=[request],
                producer="smart_summary_input_pack",
                input_paths=[transcript, courseware],
                policy=policy,
            )
            assert child["new_user_confirmation_required"] is False
            payload = json.loads(Path(child["consent_path"]).read_text(encoding="utf-8"))
            assert payload["operator_confirmation"]["parent_authorization"]["bundle_path"] == str(bundle)

    status = validate_model_business_authorization(authorization, policy=policy)
    assert status["valid"] is True
    assert status["bundle_dirs"] == [str(first_bundle), str(second_bundle)]

def test_batch_reuse_matcher_requires_exact_bound_bundle_source_route_and_capacity(tmp_path: Path) -> None:
    first_bundle = _bundle(tmp_path)
    second_bundle = tmp_path / "bundle-two"
    second_bundle.mkdir()
    (second_bundle / "manifest.json").write_text("{}", encoding="utf-8")
    (second_bundle / "timeline.json").write_text("[]", encoding="utf-8")
    first_source = first_bundle / "transcript.txt"
    second_source = second_bundle / "transcript.txt"
    courseware = second_bundle / "companion-courseware.md"
    for path, text in ((first_source, "first"), (second_source, "second"), (courseware, "courseware")):
        path.write_text(text, encoding="utf-8")
    policy = TrustedModelConnectorPolicy((tmp_path.resolve(),), frozenset({"api.example"}))
    authorization = tmp_path / "batch.json"
    create_model_business_authorization(
        tmp_path,
        bundle_dir=first_bundle,
        bundle_dirs=[second_bundle],
        source_paths=[first_source, second_source, courseware],
        stages=[_stage(max_calls=6, max_estimated_cost_usd=0.6)],
        purpose="batch",
        max_calls=6,
        max_estimated_cost_usd=0.6,
        confirm_data_export=True,
        output_path=authorization,
        policy=policy,
    )
    reusable = find_reusable_model_business_authorization(
        [authorization], bundle_dir=second_bundle, source_paths=[second_source, courseware], task="smart_summary_rewrite", route_id="summary-route", route_revision="a" * 64, producer="smart_summary_input_pack", required_calls=6, policy=policy,
    )
    assert reusable["status"] == "reusable"
    courseware.write_text("changed", encoding="utf-8")
    blocked = find_reusable_model_business_authorization(
        [authorization], bundle_dir=second_bundle, source_paths=[second_source, courseware], task="smart_summary_rewrite", route_id="summary-route", route_revision="a" * 64, producer="smart_summary_input_pack", required_calls=1, policy=policy,
    )
    assert blocked["status"] == "scope_expansion_required"


def test_summary_input_pack_source_uses_stable_transcript_lineage(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    transcript = bundle / "normalized-transcript.json"
    transcript.write_text('{"segments":[{"text":"stable transcript"}]}', encoding="utf-8")
    companion = bundle / "companion-courseware.md"
    companion.write_text("stable courseware", encoding="utf-8")
    exports = bundle / "exports"
    exports.mkdir()
    pack = exports / "smart-summary-input-pack.json"

    def write_pack(*, created_at: str, transcript_hash: str | None = None) -> None:
        pack.write_text(
            json.dumps(
                {
                    "schema": "video_knowledge_pipeline.smart_summary_input_pack.v1",
                    "created_at": created_at,
                    "transcript_source": str(transcript),
                    "transcript_source_sha256": transcript_hash or hashlib.sha256(transcript.read_bytes()).hexdigest(),
                    "companion_courseware": {"bundle_copy_path": str(companion)},
                    "quality": {"volatile": created_at},
                }
            ),
            encoding="utf-8",
        )

    write_pack(created_at="first")
    policy = TrustedModelConnectorPolicy((tmp_path.resolve(),), frozenset({"api.example"}))
    authorization = tmp_path / "summary-parent.json"
    create_model_business_authorization(
        tmp_path,
        bundle_dir=bundle,
        source_paths=[transcript, companion, pack],
        stages=[_stage()],
        purpose="summary input pack preflight",
        max_calls=2,
        max_estimated_cost_usd=0.2,
        confirm_data_export=True,
        output_path=authorization,
        policy=policy,
    )

    write_pack(created_at="rebuilt-before-preflight")
    assert validate_model_business_authorization(authorization, policy=policy)["valid"] is True

    write_pack(created_at="wrong-lineage", transcript_hash="f" * 64)
    status = validate_model_business_authorization(authorization, policy=policy)
    assert status["valid"] is False
    assert any(row["key"] == "authorization_source_changed" for row in status["blockers"])
