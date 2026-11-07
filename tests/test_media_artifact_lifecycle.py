from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
from collections import namedtuple
from pathlib import Path

import pytest

from video_knowledge_pipeline.media_artifact_lifecycle import (
    ALLOWED_CANDIDATE_KINDS,
    CORE_CONTRACT_FILES,
    MINIMUM_FREE_BYTES,
    OUTPUT_HEADROOM_MULTIPLIER,
    PROTECTED_CLASSES,
    WARNING_FREE_BYTES,
    MediaArtifactLifecycleError,
    _load_core,
    _resolve_core_root,
    explain_media_artifact,
    media_artifact_doctor,
    plan_media_artifact_gc,
    restore_synthetic_media_artifact,
    scan_media_artifacts,
    verify_media_artifact_plan,
)


REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "media-artifact-lifecycle"
CORE = REPO.parent / "artifact-lifecycle-core"
SELF_MEDIA = REPO.parent / "self-media-creation-system"


def _copy_fixture(tmp_path: Path) -> Path:
    bundle = tmp_path / "media-artifact-fixture"
    shutil.copytree(FIXTURE, bundle)
    return bundle


def _paths(bundle: Path) -> tuple[Path, Path, Path, Path]:
    lifecycle = bundle / ".lifecycle" / "runtime"
    lifecycle.mkdir(parents=True, exist_ok=True)
    quarantine = bundle / ".fixture-quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    return (
        lifecycle / "manifest.json",
        lifecycle / "plan.json",
        lifecycle / "checkpoints",
        quarantine,
    )


def _scan(bundle: Path) -> tuple[dict, Path, Path, Path]:
    manifest, plan, checkpoints, quarantine = _paths(bundle)
    result = scan_media_artifacts(
        bundle,
        scope_path=".lifecycle/media-artifact-scope.v1.json",
        output_path=manifest,
        checkpoint_dir=checkpoints,
        core_root=CORE,
        max_depth=4,
        max_items=100,
    )
    assert result["status"] == "passed"
    return result, manifest, plan, quarantine


def _plan(bundle: Path) -> tuple[dict, Path, Path, Path]:
    _, manifest, plan, quarantine = _scan(bundle)
    core = _load_core(_resolve_core_root(CORE))
    original_git_status = core.git_status
    core.git_status = lambda _root: "clean"
    try:
        result = plan_media_artifact_gc(
            manifest,
            bundle_dir=bundle,
            quarantine_root=quarantine,
            output_path=plan,
            core_root=CORE,
        )
    finally:
        core.git_status = original_git_status
    return result, manifest, plan, quarantine


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_doctor_contracts_and_output_aware_disk_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _copy_fixture(tmp_path)
    ready = media_artifact_doctor(
        bundle,
        scope_path=".lifecycle/media-artifact-scope.v1.json",
        core_root=CORE,
        self_media_root=SELF_MEDIA,
    )
    assert ready["status"] == "ready"
    assert ready["lifecycle_profile"] == "media"
    assert ready["cleanup_mode"] == "plan_only"
    assert ready["automatic_purge"] is False
    assert ready["scope"]["compatibility"]["loaded_through_core_compatibility_view"] is True
    assert ready["core_contract"]["files"][0]["path"] == CORE_CONTRACT_FILES[0]
    assert ready["media_profile"]["minimum_free_bytes"] == MINIMUM_FREE_BYTES
    assert ready["media_profile"]["warning_free_bytes"] == WARNING_FREE_BYTES
    assert ready["media_profile"]["output_headroom_multiplier"] == OUTPUT_HEADROOM_MULTIPLIER
    assert "minimum_free_ratio" not in ready["media_profile"]

    core = _load_core(_resolve_core_root(CORE))
    disk_usage = namedtuple("disk_usage", "total used free")
    two_gib = 2 * 1024**3
    total = 100 * 1024**3
    monkeypatch.setattr(core.shutil, "disk_usage", lambda _path: disk_usage(total, total - two_gib, two_gib))
    warning = media_artifact_doctor(
        bundle,
        scope_path=".lifecycle/media-artifact-scope.v1.json",
        core_root=CORE,
        self_media_root=SELF_MEDIA,
        large_task=True,
        estimated_output_bytes=128 * 1024**2,
    )
    assert warning["status"] == "ready"
    assert warning["reason_codes"] == ["low_disk_allowed"]
    assert warning["blocking_reason_codes"] == []
    assert warning["warnings"] == ["low_disk_allowed"]
    assert warning["large_task_preflight"]["warning_only"] is True
    assert warning["large_task_preflight"]["disk"]["required_free_bytes"] == 704643072

    blocked = media_artifact_doctor(
        bundle,
        scope_path=".lifecycle/media-artifact-scope.v1.json",
        core_root=CORE,
        self_media_root=SELF_MEDIA,
        large_task=True,
        estimated_output_bytes=two_gib,
    )
    assert blocked["status"] == "blocked"
    assert blocked["reason_codes"] == ["blocked_low_disk"]
    assert blocked["large_task_preflight"]["disk"]["required_free_bytes"] == 3 * 1024**3
    assert blocked["large_task_preflight"]["starts_task"] is False


def test_large_preflight_requires_estimate_or_explicit_non_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _copy_fixture(tmp_path)
    missing = media_artifact_doctor(
        bundle,
        scope_path=".lifecycle/media-artifact-scope.v1.json",
        core_root=CORE,
        self_media_root=SELF_MEDIA,
        large_task=True,
    )
    assert missing["status"] == "blocked"
    assert missing["reason_codes"] == ["estimated_output_bytes_required"]
    zero = media_artifact_doctor(
        bundle,
        scope_path=".lifecycle/media-artifact-scope.v1.json",
        core_root=CORE,
        self_media_root=SELF_MEDIA,
        large_task=True,
        estimated_output_bytes=0,
    )
    assert zero["reason_codes"] == ["estimated_output_bytes_must_be_positive_integer"]
    non_integer = media_artifact_doctor(
        bundle,
        scope_path=".lifecycle/media-artifact-scope.v1.json",
        core_root=CORE,
        self_media_root=SELF_MEDIA,
        large_task=True,
        estimated_output_bytes=1.5,  # type: ignore[arg-type] - runtime API must fail closed for non-integer bytes.
    )
    assert non_integer["reason_codes"] == ["estimated_output_bytes_must_be_positive_integer"]

    core = _load_core(_resolve_core_root(CORE))
    monkeypatch.setattr(core.shutil, "disk_usage", lambda _path: pytest.fail("non-large declaration must not probe disk"))
    non_large = media_artifact_doctor(
        bundle,
        scope_path=".lifecycle/media-artifact-scope.v1.json",
        core_root=CORE,
        self_media_root=SELF_MEDIA,
        large_task=True,
        explicit_non_large=True,
    )
    assert non_large["status"] == "ready"
    assert non_large["large_task_preflight"]["operation_class"] == "non_large_declared"
    assert non_large["large_task_preflight"]["disk"] is None

    conflict = media_artifact_doctor(
        bundle,
        scope_path=".lifecycle/media-artifact-scope.v1.json",
        core_root=CORE,
        self_media_root=SELF_MEDIA,
        large_task=True,
        estimated_output_bytes=1,
        explicit_non_large=True,
    )
    assert conflict["status"] == "blocked"
    assert conflict["reason_codes"] == ["disk_preflight_mode_conflict"]


def test_preflight_cli_requires_exactly_one_output_classification() -> None:
    source_path = REPO / "src" / "video_knowledge_pipeline" / "cli.py"
    parsed = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    helper = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "_add_media_artifact_doctor_args")
    namespace = {"argparse": argparse}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source_path), "exec"), namespace)  # noqa: S102 - executes one parsed local helper only.
    parser = argparse.ArgumentParser()
    namespace["_add_media_artifact_doctor_args"](parser, preflight=True)
    estimated = parser.parse_args(["fixture", "--estimated-output-bytes", "134217728"])
    assert estimated.estimated_output_bytes == 134217728
    assert estimated.non_large is False
    non_large = parser.parse_args(["fixture", "--non-large"])
    assert non_large.non_large is True
    assert non_large.estimated_output_bytes is None
    with pytest.raises(SystemExit):
        parser.parse_args(["fixture"])
    with pytest.raises(SystemExit):
        parser.parse_args(["fixture", "--estimated-output-bytes", "1", "--non-large"])


def test_scope_disk_policy_is_required_and_legacy_ratio_is_not_accepted(tmp_path: Path) -> None:
    bundle = _copy_fixture(tmp_path)
    scope_path = bundle / ".lifecycle" / "media-artifact-scope.v1.json"
    scope = _read(scope_path)
    scope["disk_policy"] = {
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "warning_free_bytes": WARNING_FREE_BYTES,
        "output_headroom_multiplier": OUTPUT_HEADROOM_MULTIPLIER,
        "minimum_free_ratio": 0.15,
    }
    _write(scope_path, scope)
    blocked = media_artifact_doctor(
        bundle,
        scope_path=".lifecycle/media-artifact-scope.v1.json",
        core_root=CORE,
        self_media_root=SELF_MEDIA,
    )
    assert blocked["status"] == "blocked"
    assert blocked["reason_codes"] == ["SCOPE_SCHEMA_INVALID"]


def test_upstream_contract_hash_drift_blocks_before_dynamic_import(tmp_path: Path) -> None:
    bundle = _copy_fixture(tmp_path)
    copied_core = tmp_path / "artifact-lifecycle-core"
    shutil.copytree(CORE, copied_core, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    core_source = copied_core / "artifact_lifecycle_core.py"
    core_source.write_text(core_source.read_text(encoding="utf-8") + "\n# synthetic drift\n", encoding="utf-8")
    blocked = media_artifact_doctor(
        bundle,
        scope_path=".lifecycle/media-artifact-scope.v1.json",
        core_root=copied_core,
        self_media_root=SELF_MEDIA,
    )
    assert blocked["status"] == "blocked"
    assert blocked["reason_codes"] == ["CORE_CONTRACT_PIN_DRIFT"]


def test_scan_is_explicit_bounded_complete_and_idempotent(tmp_path: Path) -> None:
    bundle = _copy_fixture(tmp_path)
    before = {path.relative_to(bundle).as_posix(): path.read_bytes() for path in bundle.rglob("*") if path.is_file()}
    first, manifest_path, _, _ = _scan(bundle)
    second = scan_media_artifacts(
        bundle,
        scope_path=".lifecycle/media-artifact-scope.v1.json",
        output_path=manifest_path,
        checkpoint_dir=manifest_path.parent / "checkpoints",
        core_root=CORE,
        max_depth=4,
        max_items=100,
    )
    assert first["manifest_sha256"] == second["manifest_sha256"]
    manifest = _read(manifest_path)
    assert manifest["scan"]["explicit_roots"] == ["artifacts"]
    assert manifest["scan"]["whole_workspace_scan"] is False
    assert manifest["scan"]["observed_items"] < 100
    assert len(manifest["artifacts"]) == 14
    assert {item.get("candidate_kind") for item in manifest["artifacts"] if item.get("candidate_kind")} == ALLOWED_CANDIDATE_KINDS
    assert {item.get("protection_class") for item in manifest["artifacts"] if item.get("protection_class")} == PROTECTED_CLASSES
    for item in manifest["artifacts"]:
        for field in (
            "closure_run_id",
            "producer_tool_id",
            "owner_lane",
            "digest",
            "artifact_role",
            "depends_on",
            "referenced_by",
            "rebuild_proof",
            "minimum_keep_until",
            "sensitivity",
            "pin",
        ):
            assert field in item
    after = {path.relative_to(bundle).as_posix(): path.read_bytes() for path in bundle.rglob("*") if path.is_file() and ".lifecycle/runtime/" not in path.relative_to(bundle).as_posix()}
    assert after == before


def test_plan_only_protects_authoritative_and_exposes_six_candidates(tmp_path: Path) -> None:
    bundle = _copy_fixture(tmp_path)
    result, manifest_path, plan_path, quarantine = _plan(bundle)
    assert result["eligible_count"] == 6
    assert result["blocked_count"] == 8
    assert result["authorization_state"] == "not_authorized"
    plan = _read(plan_path)
    eligible = {item["artifact_id"] for item in plan["entries"] if item["eligible"]}
    assert eligible == {
        "candidate-chunk",
        "candidate-frame",
        "candidate-proxy",
        "candidate-failed-render",
        "candidate-duplicate-transcode",
        "candidate-superseded-temp",
    }
    for item in plan["entries"]:
        if item["artifact_id"].startswith("protected-"):
            assert not item["eligible"]
            assert any(value.startswith("permanent_protection:") for value in item["blocking_gates"])
    explained = explain_media_artifact(manifest_path, "protected-final", core_root=CORE)
    assert "permanent_protection:final_deliverable" in explained["blocking_gates"]
    verified = verify_media_artifact_plan(
        plan_path,
        manifest_path,
        bundle_dir=bundle,
        quarantine_root=quarantine,
        core_root=CORE,
    )
    assert verified["status"] == "passed"
    assert verified["filesystem_mutation_performed"] is False


def test_dirty_git_gate_remains_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _copy_fixture(tmp_path)
    _, manifest_path, plan_path, quarantine = _scan(bundle)
    core = _load_core(_resolve_core_root(CORE))
    monkeypatch.setattr(core, "git_status", lambda _root: "dirty")
    result = plan_media_artifact_gc(
        manifest_path,
        bundle_dir=bundle,
        quarantine_root=quarantine,
        output_path=plan_path,
        core_root=CORE,
    )
    assert result["eligible_count"] == 0
    plan = _read(plan_path)
    assert all("git_dirty" in item["blocking_gates"] for item in plan["entries"])


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (lambda scope: scope["artifacts"][0]["referenced_by"].append("human-review"), "references_not_cleared"),
        (lambda scope: scope["artifacts"][0].update({"rebuild_proof": {"status": "unknown", "command": None, "input_hashes": [], "dependency_lock": None, "verification_ref": None}}), "rebuild_proof_incomplete"),
        (lambda scope: scope["active_run"].update({"state": "active", "owner_state": "active", "dispatch_state": "active", "process_ids": [4242]}), "active_process_not_cleared"),
    ],
)
def test_references_proof_and_active_run_block_candidates(tmp_path: Path, mutator, expected: str) -> None:
    bundle = _copy_fixture(tmp_path)
    scope_path = bundle / ".lifecycle" / "media-artifact-scope.v1.json"
    scope = _read(scope_path)
    mutator(scope)
    _write(scope_path, scope)
    _, _, plan_path, _ = _plan(bundle)
    candidate = next(item for item in _read(plan_path)["entries"] if item["artifact_id"] == "candidate-chunk")
    assert expected in candidate["blocking_gates"]
    assert candidate["eligible"] is False


def test_plan_and_artifact_drift_are_rejected(tmp_path: Path) -> None:
    bundle = _copy_fixture(tmp_path)
    _, manifest_path, plan_path, quarantine = _plan(bundle)
    plan = _read(plan_path)
    plan["boundary"] += " drift"
    _write(plan_path, plan)
    drifted_plan = verify_media_artifact_plan(
        plan_path,
        manifest_path,
        bundle_dir=bundle,
        quarantine_root=quarantine,
        core_root=CORE,
    )
    assert drifted_plan["status"] == "blocked"
    assert "plan_hash_drift" in drifted_plan["errors"]

    plan_media_artifact_gc(
        manifest_path,
        bundle_dir=bundle,
        quarantine_root=quarantine,
        output_path=plan_path,
        core_root=CORE,
    )
    (bundle / "artifacts" / "chunks" / "chunk-0001.fixture").write_text("identity drift\n", encoding="utf-8")
    drifted_file = verify_media_artifact_plan(
        plan_path,
        manifest_path,
        bundle_dir=bundle,
        quarantine_root=quarantine,
        core_root=CORE,
    )
    assert drifted_file["status"] == "blocked"
    assert any("candidate-chunk" in error for error in drifted_file["errors"])


def test_scope_hash_depth_item_and_metadata_overwrite_guards(tmp_path: Path) -> None:
    bundle = _copy_fixture(tmp_path)
    _, manifest_path, plan_path, quarantine = _plan(bundle)
    scope_path = bundle / ".lifecycle" / "media-artifact-scope.v1.json"
    scope_path.write_text(scope_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(MediaArtifactLifecycleError, match="SCOPE_HASH_DRIFT"):
        plan_media_artifact_gc(
            manifest_path,
            bundle_dir=bundle,
            quarantine_root=quarantine,
            output_path=plan_path,
            core_root=CORE,
        )

    fresh = _copy_fixture(tmp_path / "fresh")
    with pytest.raises(MediaArtifactLifecycleError, match="SCAN_ITEM_LIMIT_EXCEEDED"):
        scan_media_artifacts(
            fresh,
            scope_path=".lifecycle/media-artifact-scope.v1.json",
            output_path=fresh / ".lifecycle" / "manifest.json",
            checkpoint_dir=fresh / ".lifecycle" / "checkpoints",
            core_root=CORE,
            max_depth=4,
            max_items=1,
        )
    with pytest.raises(MediaArtifactLifecycleError, match="METADATA_DESTINATION_INSIDE_ARTIFACT_ROOT"):
        scan_media_artifacts(
            fresh,
            scope_path=".lifecycle/media-artifact-scope.v1.json",
            output_path=fresh / "artifacts" / "manifest.json",
            checkpoint_dir=fresh / ".lifecycle" / "checkpoints",
            core_root=CORE,
            max_depth=4,
            max_items=100,
        )


def test_reparse_cross_volume_and_hardlink_coverage_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _copy_fixture(tmp_path)
    core = _load_core(_resolve_core_root(CORE))
    original_reparse = core._is_reparse
    blocked_path = bundle / "artifacts" / "frames"
    monkeypatch.setattr(core, "_is_reparse", lambda path: Path(path) == blocked_path or original_reparse(Path(path)))
    with pytest.raises(MediaArtifactLifecycleError, match="REPARSE_POINT_BLOCKED"):
        _scan(bundle)
    monkeypatch.setattr(core, "_is_reparse", original_reparse)

    _, manifest_path, plan_path, quarantine = _scan(bundle)
    original_volume = core.volume_identity
    monkeypatch.setattr(core, "volume_identity", lambda path: 2 if Path(path) == quarantine else 1)
    with pytest.raises(MediaArtifactLifecycleError, match="CROSS_VOLUME_BLOCKED"):
        plan_media_artifact_gc(
            manifest_path,
            bundle_dir=bundle,
            quarantine_root=quarantine,
            output_path=plan_path,
            core_root=CORE,
        )
    monkeypatch.setattr(core, "volume_identity", original_volume)

    hardlink_bundle = _copy_fixture(tmp_path / "hardlink")
    source = hardlink_bundle / "artifacts" / "chunks" / "chunk-0001.fixture"
    outside = hardlink_bundle / "undeclared-hardlink.fixture"
    try:
        os.link(source, outside)
    except OSError:
        pytest.skip("hardlinks are unavailable on this test volume")
    _, _, hardlink_plan, _ = _plan(hardlink_bundle)
    entry = next(item for item in _read(hardlink_plan)["entries"] if item["artifact_id"] == "candidate-chunk")
    assert "hardlink_coverage_incomplete" in entry["blocking_gates"]
    assert entry["eligible"] is False


def test_synthetic_restore_uses_core_and_real_or_repeat_restore_is_blocked(tmp_path: Path) -> None:
    bundle = _copy_fixture(tmp_path)
    _, manifest_path, plan_path, quarantine = _plan(bundle)
    core = _load_core(_resolve_core_root(CORE))
    plan = _read(plan_path)
    entry = next(item for item in plan["entries"] if item["artifact_id"] == "candidate-chunk")
    source = Path(entry["source_path"])
    quarantined = Path(entry["quarantine_path"])
    quarantined.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, quarantined)
    receipt = core.attach_hash(
        {
            "schema_version": core.RECEIPT_SCHEMA_VERSION,
            "operation": "quarantine",
            "status": "completed",
            "plan_sha256": plan["plan_sha256"],
            "manifest_sha256": _read(manifest_path)["manifest_sha256"],
            "closure_run_id": plan["closure_run_id"],
            "completed_at": "2026-08-28T00:00:00Z",
            "items": [
                {
                    "artifact_id": entry["artifact_id"],
                    "from": str(source),
                    "to": str(quarantined),
                    "digest": entry["digest"],
                }
            ],
            "filesystem_mutation_scope": "synthetic_fixture_test_only",
        },
        "receipt_sha256",
    )
    receipt_path = bundle / ".lifecycle" / "runtime" / "quarantine-receipt.json"
    _write(receipt_path, receipt)
    preview = restore_synthetic_media_artifact(
        plan_path,
        receipt_path,
        bundle_dir=bundle,
        quarantine_root=quarantine,
        output_path=bundle / ".lifecycle" / "runtime" / "restore-receipt.json",
        core_root=CORE,
    )
    assert preview["status"] == "ready_to_restore_synthetic"
    assert not source.exists()
    restored = restore_synthetic_media_artifact(
        plan_path,
        receipt_path,
        bundle_dir=bundle,
        quarantine_root=quarantine,
        output_path=bundle / ".lifecycle" / "runtime" / "restore-receipt.json",
        execute_synthetic=True,
        core_root=CORE,
    )
    assert restored["restored_count"] == 1
    assert restored["mutation_scope"] == "synthetic_fixture_only"
    assert source.is_file() and not quarantined.exists()
    with pytest.raises(MediaArtifactLifecycleError):
        restore_synthetic_media_artifact(
            plan_path,
            receipt_path,
            bundle_dir=bundle,
            quarantine_root=quarantine,
            output_path=bundle / ".lifecycle" / "runtime" / "restore-repeat.json",
            execute_synthetic=True,
            core_root=CORE,
        )

    unmarked = _copy_fixture(tmp_path / "unmarked")
    (unmarked / ".vkp-synthetic-artifact-fixture.json").unlink()
    with pytest.raises(MediaArtifactLifecycleError, match="SYNTHETIC_MARKER_REQUIRED"):
        restore_synthetic_media_artifact(
            plan_path,
            receipt_path,
            bundle_dir=unmarked,
            quarantine_root=quarantine,
            output_path=unmarked / "restore.json",
            execute_synthetic=True,
            core_root=CORE,
        )


def test_cli_surface_has_no_apply_or_purge_command() -> None:
    source = (REPO / "src" / "video_knowledge_pipeline" / "cli.py").read_text(encoding="utf-8")
    for name in (
        "media-artifact-doctor",
        "media-artifact-preflight",
        "media-artifact-scan",
        "media-artifact-explain",
        "media-artifact-plan-gc",
        "media-artifact-verify-plan",
        "media-artifact-restore",
    ):
        assert f'add_parser("{name}"' in source
    assert 'add_parser("media-artifact-apply-quarantine"' not in source
    assert 'add_parser("media-artifact-purge"' not in source
