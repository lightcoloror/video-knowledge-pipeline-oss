from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import jsonschema
import pytest

from video_knowledge_pipeline import portable
from portable_test_runtime import portable_test_directory


ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_BUNDLE_ENV = os.environ.get("VKP_PORTABLE_CONTRACT_BUNDLE", "").strip()
CONTRACT_BUNDLE = Path(_CONTRACT_BUNDLE_ENV) if _CONTRACT_BUNDLE_ENV else None


def test_portable_doctor_is_schema_valid_idempotent_and_path_neutral() -> None:
    first = portable.portable_doctor(ROOT)
    second = portable.portable_doctor(ROOT)
    schema = _read_json(ROOT / "src/video_knowledge_pipeline/schemas/portable-doctor.v1.schema.json")

    assert first == second
    jsonschema.validate(first, schema)
    assert first["ok"] is True
    assert first["project"]["root"] == "."
    assert first["boundaries"] == {
        "network_access": False,
        "provider_called": False,
        "secret_read": False,
        "media_uploaded": False,
        "service_started": False,
        "execution_authorized": False,
        "silent_fallback": False,
    }
    assert not _absolute_paths(first)


def test_portable_doctor_distinguishes_windows_linux_and_macos_evidence() -> None:
    windows = portable.portable_doctor(ROOT, platform_name="windows")
    linux = portable.portable_doctor(ROOT, platform_name="linux")
    macos = portable.portable_doctor(ROOT, platform_name="macos")

    current = portable._normalise_platform(sys.platform)
    assert windows["platform"] == "windows"
    assert linux["platform"] == "linux"
    assert macos["platform"] == "macos"
    assert windows["platform_matrix"][current]["runtime_evidence"] == "verified_current_host"
    for result in (windows, linux, macos):
        for platform_name, evidence in result["platform_matrix"].items():
            if platform_name != current and platform_name != "macos":
                assert evidence["runtime_evidence"] in {
                    "ci_matrix_candidate",
                    "static_contract_and_ci_matrix_only",
                }


@pytest.mark.skipif(
    CONTRACT_BUNDLE is None,
    reason="set VKP_PORTABLE_CONTRACT_BUNDLE to verify the external distribution bundle",
)
def test_portable_doctor_verifies_exact_shared_contract_bundle() -> None:
    assert CONTRACT_BUNDLE is not None
    result = portable.portable_doctor(
        ROOT,
        contract_bundle=CONTRACT_BUNDLE,
        require_contract_bundle=True,
    )

    assert result["ok"] is True
    assert result["portable_contract_bundle"]["status"] == "verified"
    assert result["portable_contract_bundle"]["required_contracts_present"] is True


def test_portable_doctor_fails_closed_for_missing_required_contract_bundle() -> None:
    with portable_test_directory("missing-contract") as tmp_path:
        result = portable.portable_doctor(
            ROOT,
            contract_bundle=tmp_path / "missing.json",
            require_contract_bundle=True,
        )

    assert result["ok"] is False
    assert "portable_contract_bundle" in result["failed_checks"]


def test_portable_doctor_fails_closed_for_manifest_path_and_hash_drift() -> None:
    with portable_test_directory("manifest-drift") as tmp_path:
        project = _copy_portable_project(tmp_path)
        manifest_path = project / "agent-tool-manifest.v1.json"
        manifest = _read_json(manifest_path)
        manifest["provenance"]["unsafe_path"] = "C:\\Users\\example\\private"
        _write_json(manifest_path, manifest)
        _refresh_lock_hash(project, "agent-tool-manifest.v1.json")

        result = portable.portable_doctor(project)

    assert result["ok"] is False
    assert "portable_paths" in result["failed_checks"]
    assert "lock_hashes" not in result["failed_checks"]


def test_portable_doctor_fails_closed_for_stale_lock() -> None:
    with portable_test_directory("stale-lock") as tmp_path:
        project = _copy_portable_project(tmp_path)
        (project / "Taskfile.yml").write_text("version: '3'\ntasks: {}\n", encoding="utf-8")

        result = portable.portable_doctor(project)

    assert result["ok"] is False
    assert "lock_hashes" in result["failed_checks"]


def test_gateway_git_commit_is_provenance_not_compatibility_gate() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    manifest = _read_json(ROOT / "agent-tool-manifest.v1.json")

    assert "model-provider-gateway>=0.3,<0.4" in pyproject
    assert "model-provider-gateway.git@" not in pyproject
    assert manifest["provenance"]["gateway_version_range"] == ">=0.3,<0.4"
    assert manifest["provenance"]["source_commit_role"] == "informational_public_baseline_not_runtime_gate"


def test_portable_smoke_is_hash_bound_and_idempotent() -> None:
    with portable_test_directory("smoke-idempotent") as tmp_path:
        run_root = tmp_path / "portable-smoke"

        first = portable.run_portable_smoke(run_root, project_root=ROOT)
        first_bytes = (run_root / "portable-smoke-receipt.json").read_bytes()
        second = portable.run_portable_smoke(run_root, project_root=ROOT)
        second_bytes = (run_root / "portable-smoke-receipt.json").read_bytes()
        validation = portable.validate_portable_smoke(run_root, project_root=ROOT)

    assert first == second
    assert first_bytes == second_bytes
    assert first["material_manifest"]["status"] == "valid"
    assert validation["passed"] is True
    assert first["boundaries"]["external_provider_called"] is False
    assert first["boundaries"]["real_media_read"] is False


def test_portable_smoke_rejects_corrupt_prior_state() -> None:
    with portable_test_directory("smoke-corrupt") as tmp_path:
        run_root = tmp_path / "portable-smoke"
        portable.run_portable_smoke(run_root, project_root=ROOT)
        (run_root / "synthetic-video-bundle/timeline.json").write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="state drift"):
            portable.run_portable_smoke(run_root, project_root=ROOT)
        with pytest.raises(ValueError, match="artifact drift"):
            portable.validate_portable_smoke(run_root, project_root=ROOT)


def test_portable_smoke_requires_explicit_compatible_gateway_mock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        portable,
        "_gateway_contract_status",
        lambda: {
            "status": "blocked_optional_dependency",
            "version": "",
            "required_version": ">=0.3,<0.4",
            "contract_schema": "model_provider_gateway.adapter_contract.v2",
        },
    )

    with portable_test_directory("gateway-mock") as tmp_path:
        with pytest.raises(RuntimeError, match="loopback mock is unavailable"):
            portable.run_portable_smoke(
                tmp_path / "required",
                project_root=ROOT,
                require_gateway_mock=True,
            )
        receipt = portable.run_portable_smoke(tmp_path / "optional", project_root=ROOT)
        assert receipt["gateway_mock"]["status"] == "blocked_optional_dependency"


def test_capability_truth_does_not_promote_unverified_or_online_capabilities() -> None:
    truth = portable.portable_doctor(ROOT)["capability_truth"]

    assert truth["cantonese_asr"]["production_ready"] is False
    assert truth["embedding"]["production_ready"] is False
    assert truth["panorama_analysis"] == {
        "status": "local_trial",
        "projection_backend": "ffmpeg_v360_optional_runtime_not_evaluated",
        "py360convert": "optional_oracle_not_core_runtime",
        "equilib": "optional_gpu_worker_not_core_runtime",
        "real_media_proof": False,
    }
    assert truth["speaker_identity"]["identity_inference_allowed"] is False
    assert truth["online_provider"]["status"] == "blocked_missing_explicit_consent"
    assert truth["tts_and_digital_avatar"]["status"] == "paused_not_promoted"
    assert truth["secret_storage"] == {
        "status": "shared_gateway_contract",
        "cross_platform": ["env", "keyring"],
        "windows_legacy_adapter": "dpapi",
        "secret_read_by_doctor": False,
    }


def test_taskfile_and_ci_example_keep_offline_execution_explicit() -> None:
    taskfile = (ROOT / "Taskfile.yml").read_text(encoding="utf-8")
    workflow = (ROOT / "ci/portable.yml.example").read_text(encoding="utf-8")

    assert "--offline --no-sync" in taskfile
    assert "video-knowledge-portable doctor" in taskfile
    assert "windows-latest" in workflow and "ubuntu-latest" in workflow
    assert "--offline --no-sync" in workflow
    assert "require-gateway-mock" not in workflow


def test_lock_bound_files_have_cross_platform_lf_checkout_policy() -> None:
    attributes = set((ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines())
    lock = _read_json(ROOT / "portable-contract.lock.json")

    for row in lock["artifacts"]:
        assert f"{row['path']} text eol=lf" in attributes


def _copy_portable_project(tmp_path: Path) -> Path:
    destination = tmp_path / "project"
    paths = [
        ".gitattributes",
        "AGENT_DISCOVERY.md",
        "Taskfile.yml",
        "agent-tool-manifest.v1.json",
        "portable-contract.lock.json",
        "pyproject.toml",
        "portability/pyproject.toml",
        "config/video-knowledge-pipeline.json",
        "src/video_knowledge_pipeline/__init__.py",
        "src/video_knowledge_pipeline/cli.py",
        "src/video_knowledge_pipeline/portable.py",
        "src/video_knowledge_pipeline/quick_health.py",
        "src/video_knowledge_pipeline/schemas/agent-tool-manifest.v1.schema.json",
        "src/video_knowledge_pipeline/schemas/portable-doctor.v1.schema.json",
        "src/video_knowledge_pipeline/schemas/portable-smoke-receipt.v1.schema.json",
        "src/video_knowledge_pipeline/schemas/quick-health.v1.schema.json",
        "src/video_knowledge_pipeline/fixtures/portable/synthetic-video-bundle.v1.json",
    ]
    # Include every locked adapter added by a reviewed public increment.
    paths.extend(row["path"] for row in _read_json(ROOT / "portable-contract.lock.json")["artifacts"]
                 if row["path"] not in paths)
    for relative in paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return destination


def _refresh_lock_hash(project: Path, relative: str) -> None:
    lock_path = project / "portable-contract.lock.json"
    lock = _read_json(lock_path)
    for row in lock["artifacts"]:
        if row["path"] == relative:
            row["sha256"] = _sha256(project / relative)
            break
    else:
        raise AssertionError(f"lock does not bind {relative}")
    _write_json(lock_path, lock)


def _absolute_paths(value: object) -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            hits.extend(_absolute_paths(item))
    elif isinstance(value, list):
        for item in value:
            hits.extend(_absolute_paths(item))
    elif isinstance(value, str) and (":\\" in value or value.startswith(("/home/", "/Users/"))):
        hits.append(value)
    return hits


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
