"""Clone-portable, fail-closed discovery and synthetic smoke front door.

This module owns no provider protocol, ASR runtime, model cache, or workflow
state. It validates repository metadata, reuses VKP's material-manifest
producer, and optionally calls the shared Gateway's ephemeral loopback mock.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import jsonschema

from .file_hash import sha256_file as _sha256
from .quick_health import build_quick_health
from .storage import read_json, write_json, write_text_atomic


DOCTOR_SCHEMA = "video_knowledge_pipeline.portable_doctor.v1"
SMOKE_SCHEMA = "video_knowledge_pipeline.portable_smoke_receipt.v1"
LOCK_SCHEMA = "video_knowledge_pipeline.portable_contract_lock.v1"
GATEWAY_VERSION_RANGE = ">=0.3,<0.4"
REQUIRED_CONTRACTS = {"material-manifest.v1", "evidence-bundle.v1", "review-packet.v1"}
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def portable_doctor(
    project_root: str | Path | None = None,
    *,
    platform_name: str | None = None,
    contract_bundle: str | Path | None = None,
    require_contract_bundle: bool = False,
) -> dict[str, Any]:
    """Validate clone-local discovery without loading model/provider stacks."""

    root = _project_root(project_root)
    target_platform = _normalise_platform(platform_name or sys.platform)
    current_platform = _normalise_platform(sys.platform)
    checks: list[dict[str, str]] = []

    def check(check_id: str, ok: bool, detail: str) -> None:
        checks.append({"check_id": check_id, "status": "passed" if ok else "failed", "detail": detail})

    paths = _portable_paths(root)
    for key, path in paths.items():
        check(f"required_path:{key}", path.is_file(), f"{path.relative_to(root).as_posix()} must be readable")

    manifest = _read_object(paths["manifest"])
    manifest_schema = _read_object(paths["manifest_schema"])
    lock = _read_object(paths["lock"])
    doctor_schema = _read_object(paths["doctor_schema"])
    smoke_schema = _read_object(paths["smoke_schema"])
    check("manifest_schema", _schema_valid(manifest, manifest_schema), "agent manifest must match shared v1")
    check("doctor_schema", _schema_declares(doctor_schema, DOCTOR_SCHEMA), "portable doctor schema must be v1")
    check("smoke_schema", _schema_declares(smoke_schema, SMOKE_SCHEMA), "portable smoke schema must be v1")
    check("lock_schema", lock.get("schema") == LOCK_SCHEMA, "portable lock schema must be v1")
    check("lock_hashes", _lock_hashes_match(root, lock), "lock-bound clone artifacts must match SHA-256")
    check("portable_paths", not _absolute_path_values(manifest, lock), "portable metadata must not contain host paths")
    check(
        "manifest_authority",
        manifest.get("notTrustedAsInstruction") is True and manifest.get("executionAuthority") == "none_metadata_only",
        "agent metadata must never grant execution authority",
    )
    check(
        "semantic_gateway_range",
        _gateway_dependency_is_semantic(paths["pyproject"]),
        "shared Gateway compatibility must use a 0.3.x semantic range",
    )
    quick = build_quick_health(root)
    check("quick_health", quick.get("ok") is True, "bounded quick health must pass without model stacks")

    bundle_status = _contract_bundle_status(contract_bundle, lock)
    if require_contract_bundle:
        check("portable_contract_bundle", bundle_status["status"] == "verified", "exact contract bundle required")
    elif bundle_status["status"] == "invalid":
        check("portable_contract_bundle", False, "supplied contract bundle must match the lock")

    failed = [row["check_id"] for row in checks if row["status"] == "failed"]
    gateway = _gateway_contract_status()
    return {
        "schema": DOCTOR_SCHEMA,
        "schema_version": "1.0",
        "ok": not failed,
        "status": "ready" if not failed else "not_ready",
        "project": {
            "name": "video-knowledge-pipeline",
            "version": _project_version(paths["pyproject"]),
            "root": ".",
        },
        "platform": target_platform,
        "platform_matrix": _platform_matrix(current_platform),
        "manifest": {
            "status": "valid" if _schema_valid(manifest, manifest_schema) else "invalid",
            "path": "agent-tool-manifest.v1.json",
            "sha256": _sha256(paths["manifest"]) if paths["manifest"].is_file() else "",
            "execution_authority": "none_metadata_only",
        },
        "portable_contract_bundle": bundle_status,
        "gateway": gateway,
        "capability_truth": _capability_truth(gateway),
        "checks": checks,
        "failed_checks": failed,
        "boundaries": {
            "network_access": False,
            "provider_called": False,
            "secret_read": False,
            "media_uploaded": False,
            "service_started": False,
            "execution_authorized": False,
            "silent_fallback": False,
        },
    }


def run_portable_smoke(
    output_dir: str | Path,
    *,
    project_root: str | Path | None = None,
    require_gateway_mock: bool = False,
) -> dict[str, Any]:
    """Run deterministic synthetic video-evidence and Gateway mock checks."""

    root = _project_root(project_root)
    if not portable_doctor(root)["ok"]:
        raise ValueError("portable doctor is not ready")
    destination = Path(output_dir).expanduser().resolve()
    fixture_path = _portable_paths(root)["fixture"]
    fixture = _read_object(fixture_path)
    if fixture.get("fixture_schema") != "video_knowledge_pipeline.material_manifest_fixture.v1":
        raise ValueError("portable synthetic fixture schema is unsupported")
    bundle = destination / "synthetic-video-bundle"
    _materialise_fixture(bundle, fixture)

    from .creative_contract_bridge import build_material_manifest, validate_material_manifest

    material = build_material_manifest(bundle)
    validation = validate_material_manifest(bundle)
    gateway = _run_gateway_mock()
    if require_gateway_mock and gateway["status"] != "ready_loopback_mock":
        raise RuntimeError("compatible model-provider-gateway loopback mock is unavailable")

    artifacts = [
        _artifact_ref(destination, role, path)
        for role, path in (
            ("bundle_manifest", bundle / "manifest.json"),
            ("timeline", bundle / "timeline.json"),
            ("transcript", bundle / "corrected-transcript.json"),
            ("material_manifest", bundle / "exports" / "material-manifest.v1.json"),
        )
    ]
    identity = {
        "schema": SMOKE_SCHEMA,
        "schema_version": "1.0",
        "status": "completed",
        "fixture_sha256": _sha256(fixture_path),
        "material_manifest": {
            "status": validation["status"],
            "manifest_id": material["manifest_id"],
            "manifest_sha256": material["manifest_sha256"],
            "timeline_item_count": validation["timeline_item_count"],
            "keyframe_count": validation["keyframe_count"],
        },
        "gateway_mock": gateway,
        "artifacts": artifacts,
        "capability_truth": _capability_truth(gateway),
        "boundaries": {
            "synthetic_fixture_only": True,
            "network_access": False,
            "external_provider_called": False,
            "secret_read": False,
            "real_media_read": False,
            "media_uploaded": False,
            "long_lived_service_started": False,
            "execution_authorized": False,
            "silent_fallback": False,
        },
    }
    receipt = {**identity, "receipt_sha256": _canonical_sha256(identity)}
    _validate_smoke_payload(destination, receipt, root=root)
    write_json(destination / "portable-smoke-receipt.json", receipt)
    return receipt


def validate_portable_smoke(run_root: str | Path, *, project_root: str | Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    destination = Path(run_root).expanduser().resolve()
    receipt = _read_object(destination / "portable-smoke-receipt.json")
    _validate_smoke_payload(destination, receipt, root=root)
    return {
        "schema": "video_knowledge_pipeline.portable_smoke_validation.v1",
        "status": "valid",
        "passed": True,
        "receipt_sha256": receipt["receipt_sha256"],
        "artifact_count": len(receipt["artifacts"]),
        "metadata_authorizes_execution": False,
    }


def _validate_smoke_payload(destination: Path, receipt: dict[str, Any], *, root: Path) -> None:
    schema = _read_object(_portable_paths(root)["smoke_schema"])
    try:
        jsonschema.validate(receipt, schema)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"portable smoke schema validation failed: {exc.message}") from exc
    identity = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _canonical_sha256(identity):
        raise ValueError("portable smoke receipt hash drift detected")
    for row in receipt.get("artifacts") or []:
        path = (destination / str(row.get("path") or "")).resolve()
        if not path.is_relative_to(destination):
            raise ValueError("portable smoke artifact escapes the run root")
        if not path.is_file():
            raise FileNotFoundError(f"portable smoke artifact is missing: {row.get('path')}")
        if int(row.get("bytes") or -1) != path.stat().st_size or row.get("sha256") != _sha256(path):
            raise ValueError(f"portable smoke artifact drift detected: {row.get('path')}")
    from .creative_contract_bridge import validate_material_manifest

    validate_material_manifest(destination / "synthetic-video-bundle")


def _materialise_fixture(bundle: Path, fixture: dict[str, Any]) -> None:
    for row in fixture.get("files") or []:
        if not isinstance(row, dict):
            raise ValueError("portable fixture file row must be an object")
        relative = Path(str(row.get("path") or ""))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("portable fixture path must stay relative")
        target = bundle / relative
        if "json" in row:
            expected = json.dumps(row["json"], ensure_ascii=False, indent=2)
        elif "text" in row:
            expected = str(row["text"])
        else:
            raise ValueError("portable fixture row requires json or text content")
        if target.exists():
            try:
                current = target.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ValueError(f"portable smoke state is unreadable: {relative.as_posix()}") from exc
            if current != expected:
                raise ValueError(f"portable smoke state drift detected: {relative.as_posix()}")
            continue
        write_text_atomic(target, expected)


def _run_gateway_mock() -> dict[str, Any]:
    status = _gateway_contract_status()
    if status["status"] != "compatible_installed":
        return {**status, "external_network": False, "provider_called": False}
    adapters = importlib.import_module("model_provider_gateway.adapters")
    mock_server = importlib.import_module("model_provider_gateway.mock_server")
    contract = adapters.adapter_contract("vkp")
    if contract.get("schema") != "model_provider_gateway.adapter_contract.v2":
        raise ValueError("shared Gateway adapter schema is incompatible")
    if any(
        task.get("control_policy") != "vkp_broker"
        for task in contract.get("tasks", {}).values()
        if isinstance(task, dict) and task.get("execution_status") == "enabled"
    ):
        raise ValueError("shared Gateway VKP task bypasses the Broker policy")
    result = mock_server.mock_smoke()
    if result != {"ok": True, "status": 200, "content": "mock-ok"}:
        raise RuntimeError("shared Gateway loopback mock failed")
    return {
        "status": "ready_loopback_mock",
        "contract_schema": contract["schema"],
        "version": status["version"],
        "response": "mock-ok",
        "external_network": False,
        "provider_called": False,
    }


def _gateway_contract_status() -> dict[str, Any]:
    try:
        package = importlib.import_module("model_provider_gateway")
        adapters = importlib.import_module("model_provider_gateway.adapters")
        version = str(getattr(package, "__version__", ""))
        contract = adapters.adapter_contract("vkp")
    except (ImportError, AttributeError, ValueError):
        return {
            "status": "blocked_optional_dependency",
            "version": "",
            "required_version": GATEWAY_VERSION_RANGE,
            "contract_schema": "model_provider_gateway.adapter_contract.v2",
        }
    compatible = _version_in_gateway_range(version) and contract.get("schema") == "model_provider_gateway.adapter_contract.v2"
    return {
        "status": "compatible_installed" if compatible else "blocked_incompatible_contract",
        "version": version,
        "required_version": GATEWAY_VERSION_RANGE,
        "contract_schema": str(contract.get("schema") or ""),
    }


def _capability_truth(gateway: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "manifest_discovery": {"status": "ready", "execution_authority": False},
        "synthetic_video_understanding": {"status": "ready_offline_fixture", "real_media_proof": False},
        "panorama_analysis": {
            "status": "local_trial",
            "projection_backend": "ffmpeg_v360_optional_runtime_not_evaluated",
            "py360convert": "optional_oracle_not_core_runtime",
            "equilib": "optional_gpu_worker_not_core_runtime",
            "real_media_proof": False,
        },
        "gateway_mock": {"status": gateway["status"], "current_online": False},
        "secret_storage": {
            "status": "shared_gateway_contract",
            "cross_platform": ["env", "keyring"],
            "windows_legacy_adapter": "dpapi",
            "secret_read_by_doctor": False,
        },
        "local_asr": {"status": "optional_runtime_not_evaluated", "production_ready": False},
        "cantonese_asr": {"status": "candidate_only_human_truth_missing", "production_ready": False},
        "embedding": {"status": "candidate_only_runtime_not_verified", "production_ready": False},
        "speaker_identity": {"status": "anonymous_candidates_only", "identity_inference_allowed": False},
        "online_provider": {"status": "blocked_missing_explicit_consent", "current_online": False},
        "tts_and_digital_avatar": {"status": "paused_not_promoted", "production_ready": False},
    }


def _portable_paths(root: Path) -> dict[str, Path]:
    package = root / "src" / "video_knowledge_pipeline"
    return {
        "manifest": root / "agent-tool-manifest.v1.json",
        "manifest_schema": package / "schemas" / "agent-tool-manifest.v1.schema.json",
        "doctor_schema": package / "schemas" / "portable-doctor.v1.schema.json",
        "smoke_schema": package / "schemas" / "portable-smoke-receipt.v1.schema.json",
        "lock": root / "portable-contract.lock.json",
        "fixture": package / "fixtures" / "portable" / "synthetic-video-bundle.v1.json",
        "pyproject": root / "pyproject.toml",
        "taskfile": root / "Taskfile.yml",
        "uv_project": root / "portability" / "pyproject.toml",
        "discovery": root / "AGENT_DISCOVERY.md",
    }


def _contract_bundle_status(source: str | Path | None, lock: dict[str, Any]) -> dict[str, Any]:
    expected = str((lock.get("portable_bundle") or {}).get("sha256") or "")
    if source is None:
        return {
            "status": "external_not_resolved",
            "required_for_metadata_discovery": False,
            "required_for_contract_import": True,
            "expected_sha256": expected,
        }
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        return {"status": "invalid", "reason": "missing_contract_bundle", "expected_sha256": expected}
    payload = _read_object(path)
    contracts = {str(row.get("contract_id") or "") for row in payload.get("contracts") or [] if isinstance(row, dict)}
    valid = (
        payload.get("schema_version") == "portable-contract-bundle.v1"
        and payload.get("bundle_version") == "1.0.0"
        and REQUIRED_CONTRACTS <= contracts
        and _sha256(path) == expected
    )
    return {
        "status": "verified" if valid else "invalid",
        "expected_sha256": expected,
        "actual_sha256": _sha256(path),
        "required_contracts_present": REQUIRED_CONTRACTS <= contracts,
    }


def _lock_hashes_match(root: Path, lock: dict[str, Any]) -> bool:
    artifacts = lock.get("artifacts") if isinstance(lock.get("artifacts"), list) else []
    if not artifacts:
        return False
    for row in artifacts:
        if not isinstance(row, dict):
            return False
        path = (root / str(row.get("path") or "")).resolve()
        if not path.is_relative_to(root) or not path.is_file() or _sha256(path) != row.get("sha256"):
            return False
    return True


def _gateway_dependency_is_semantic(pyproject_path: Path) -> bool:
    try:
        with pyproject_path.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return False
    extras = ((payload.get("project") or {}).get("optional-dependencies") or {}).get("shared_gateway") or []
    return extras == ["model-provider-gateway>=0.3,<0.4"]


def _absolute_path_values(*payloads: dict[str, Any]) -> list[str]:
    hits: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and (
            _WINDOWS_ABSOLUTE.match(value) or value.startswith(("/home/", "/Users/", "\\\\"))
        ):
            hits.append(value)

    for payload in payloads:
        visit(payload)
    return hits


def _platform_matrix(current: str) -> dict[str, dict[str, Any]]:
    return {
        "windows": {
            "contract": "supported",
            "runtime_evidence": "verified_current_host" if current == "windows" else "ci_matrix_candidate",
        },
        "linux": {
            "contract": "supported",
            "runtime_evidence": "verified_current_host" if current == "linux" else "static_contract_and_ci_matrix_only",
        },
        "macos": {"contract": "supported", "runtime_evidence": "not_run"},
    }


def _normalise_platform(value: str) -> str:
    lowered = value.strip().lower()
    if lowered.startswith("win"):
        return "windows"
    if lowered.startswith("linux"):
        return "linux"
    if lowered.startswith(("darwin", "mac")):
        return "macos"
    raise ValueError(f"unsupported portability platform: {value}")


def _version_in_gateway_range(value: str) -> bool:
    try:
        major, minor, *_ = (int(part) for part in value.split("."))
    except (TypeError, ValueError):
        return False
    return major == 0 and minor == 3


def _project_root(value: str | Path | None) -> Path:
    return Path(value).expanduser().resolve() if value is not None else Path(__file__).resolve().parents[2]


def _project_version(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            return str((tomllib.load(handle).get("project") or {}).get("version") or "")
    except (OSError, tomllib.TOMLDecodeError):
        return ""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _schema_valid(payload: dict[str, Any], schema: dict[str, Any]) -> bool:
    try:
        jsonschema.validate(payload, schema)
    except (jsonschema.ValidationError, jsonschema.SchemaError):
        return False
    return bool(payload) and bool(schema)


def _schema_declares(schema: dict[str, Any], expected: str) -> bool:
    return (schema.get("properties") or {}).get("schema", {}).get("const") == expected


def _artifact_ref(root: Path, role: str, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("portable smoke artifact escapes the run root")
    return {
        "role": role,
        "path": resolved.relative_to(root).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _canonical_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VKP clone-portable discovery and synthetic offline smoke")
    parser.add_argument("--project-root", default="")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--platform", default="")
    doctor.add_argument("--contract-bundle", default="")
    doctor.add_argument("--require-contract-bundle", action="store_true")
    doctor.add_argument("--pretty", action="store_true")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--require-gateway-mock", action="store_true")
    smoke.add_argument("--pretty", action="store_true")
    validate = sub.add_parser("validate-smoke")
    validate.add_argument("--run-root", required=True)
    validate.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root or None
    try:
        if args.command == "doctor":
            result = portable_doctor(
                root,
                platform_name=args.platform or None,
                contract_bundle=args.contract_bundle or None,
                require_contract_bundle=args.require_contract_bundle,
            )
        elif args.command == "smoke":
            result = run_portable_smoke(
                args.output_dir,
                project_root=root,
                require_gateway_mock=args.require_gateway_mock,
            )
        else:
            result = validate_portable_smoke(args.run_root, project_root=root)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(json.dumps({"ok": False, "status": "failed", "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if result.get("ok", result.get("passed", result.get("status") == "completed")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
