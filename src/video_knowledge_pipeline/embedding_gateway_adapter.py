"""Candidate-only VKP adapter for Gateway's canonical local embedding owner."""

from __future__ import annotations

import json
import mimetypes
from importlib import metadata
from pathlib import Path
from typing import Any

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file
from .model_provider_gateway_adapter import SharedGatewayUnavailable, _shared
from .storage import bundle_write_lock, write_json


EMBEDDING_RECEIPT_SCHEMA = "video_knowledge_pipeline.embedding_candidate_receipt.v1"
EXPECTED_GATEWAY_CONTRACT = {
    "owner_plan_schema": "model_provider_gateway.owner_capability_plan.v1",
    "owner_receipt_schema": "model_provider_gateway.owner_capability_receipt.v1",
    "owner_input_manifest_schema": "model_provider_gateway.owner_input_manifest.v1",
    "owner_gate_schema": "model_provider_gateway.owner_capability_gate.v1",
    "owner_run_report_schema": "model_provider_gateway.owner_runtime_run_report.v1",
}


class EmbeddingGatewaySemanticIncompatible(RuntimeError):
    """Raised when the installed owner-capability schemas drift."""


class EmbeddingOwnerExecutionBlocked(RuntimeError):
    """Carries one stable fail-closed local owner category."""

    def __init__(self, error_class: str) -> None:
        super().__init__(error_class)
        self.error_class = error_class


def run_embedding_candidate(
    *,
    artifact_paths: list[str | Path],
    purpose: str,
    request_id: str,
    receipt_path: str | Path,
    embedding_python: str | Path | None,
    embedding_model: str | Path | None,
    owner_output_dir: str | Path | None = None,
    mime_types: list[str] | None = None,
    data_classification: str = "synthetic_fixture",
    embedding_device: str = "cpu",
    expected_dimensions: int = 384,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Execute one explicit local owner route and return metadata-only candidates."""
    if not request_id or not purpose:
        raise ValueError("embedding request_id and purpose are required")
    if data_classification not in {"synthetic_fixture", "public_fixture"}:
        raise ValueError("VKP embedding accepts only synthetic or public fixtures")
    paths = [Path(value).expanduser().resolve() for value in artifact_paths]
    identities = _artifact_identities(paths, mime_types=mime_types)
    runtime_binding = _runtime_binding(
        embedding_python=embedding_python,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
        expected_dimensions=expected_dimensions,
        timeout_seconds=timeout_seconds,
    )
    fingerprint = canonical_json_sha256(
        {
            "consumer_id": "vkp",
            "request_id": request_id,
            "purpose": purpose,
            "data_classification": data_classification,
            "artifacts": identities,
            "runtime_binding": runtime_binding,
            "execution_mode": "gateway_owner_run_local",
        }
    )
    target = Path(receipt_path).expanduser().resolve()
    run_root = (
        Path(owner_output_dir).expanduser().resolve()
        if owner_output_dir is not None
        else target.parent / f"{target.stem}.owner-run"
    )
    gateway: dict[str, Any] | None = None
    with bundle_write_lock(
        target.parent,
        operation="vkp_embedding_candidate",
        timeout_seconds=5,
        lock_name=f".{target.name}.lock",
        stale_after_seconds=120,
    ):
        existing = _read_existing(target, fingerprint=fingerprint, artifact_paths=paths)
        if existing is not None:
            return existing
        try:
            _require_local_runtime(runtime_binding)
            gateway = _load_gateway_api()
            manifest = gateway["build_owner_input_manifest"](
                paths,
                purpose=purpose,
                data_classification=data_classification,
                mime_types=mime_types,
            )
            gateway["validate_owner_input_manifest"](manifest, artifact_paths=paths)
            bundle = gateway["execute_owner_capability"](
                "embedding",
                "sentence-transformers-local",
                paths,
                output_dir=run_root,
                consumer_id="vkp",
                purpose=purpose,
                data_classification=data_classification,
                runtime={
                    "embedding_python": str(
                        Path(embedding_python).expanduser().resolve()
                    ),
                    "embedding_model": str(
                        Path(embedding_model).expanduser().resolve()
                    ),
                    "embedding_device": embedding_device,
                    "model_revision": runtime_binding["model_revision"],
                },
                timeout_seconds=timeout_seconds,
            )
            _validate_owner_bundle(
                bundle,
                manifest=manifest,
                artifact_paths=paths,
                runtime_binding=runtime_binding,
                gateway=gateway,
            )
            receipt = _candidate_receipt(
                request_id=request_id,
                fingerprint=fingerprint,
                bundle=bundle,
                runtime_binding=runtime_binding,
                gateway=gateway,
            )
            validate_embedding_candidate_receipt(receipt, artifact_paths=paths)
        except Exception as exc:  # Reduce local runtime detail to a stable safe class.
            receipt = _blocked_receipt(
                request_id=request_id,
                fingerprint=fingerprint,
                error_class=_safe_error_class(exc),
                gateway=gateway,
            )
        write_json(target, receipt)
        return receipt


def validate_embedding_candidate_receipt(
    receipt: dict[str, Any],
    *,
    artifact_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    if receipt.get("schema") != EMBEDDING_RECEIPT_SCHEMA:
        raise EmbeddingGatewaySemanticIncompatible("VKP embedding receipt schema drift")
    if receipt.get("automatic_fallback_allowed") is not False:
        raise ValueError("VKP embedding fallback must remain disabled")
    if any(
        key in receipt for key in ("vectors", "embedding_vectors", "verified_facts")
    ):
        raise ValueError(
            "VKP embedding receipt may not contain vectors or verified facts"
        )
    if receipt.get("status") == "blocked":
        if (
            receipt.get("ok") is not False
            or receipt.get("candidate_refs")
            or receipt.get("execution_performed") is not False
        ):
            raise ValueError(
                "blocked VKP embedding receipt exposed execution or candidates"
            )
        return receipt
    if receipt.get("status") != "candidate" or receipt.get("ok") is not True:
        raise ValueError("VKP embedding receipt must be candidate or blocked")
    if artifact_paths is None:
        raise ValueError(
            "artifact paths are required to validate VKP embedding input hashes"
        )

    gateway = _load_gateway_api()
    manifest = receipt.get("input_manifest")
    plan = receipt.get("owner_plan")
    gate = receipt.get("owner_gate")
    owner = receipt.get("owner_receipt")
    run_receipt = receipt.get("owner_run_receipt")
    binding = receipt.get("runtime_binding")
    if not all(
        isinstance(value, dict)
        for value in (manifest, plan, gate, owner, run_receipt, binding)
    ):
        raise ValueError("VKP embedding owner bundle is incomplete")
    paths = [Path(value).expanduser().resolve() for value in artifact_paths]
    gateway["validate_owner_input_manifest"](manifest, artifact_paths=paths)
    try:
        gateway["validate_owner_gate"](gate, plan=plan, manifest=manifest)
    except ValueError as exc:
        if str(exc) == "consent_invalid":
            raise ValueError(
                "VKP embedding route or owner gate drift detected"
            ) from exc
        raise
    gateway["validate_owner_capability_receipt"](owner, plan=plan)
    if not (
        plan.get("consumer_id") == "vkp"
        and plan.get("capability") == "embedding"
        and plan.get("adapter_id") == "sentence-transformers-local"
        and plan.get("input_manifest_hash") == manifest.get("manifest_revision")
        and plan.get("automatic_retry_allowed") is False
        and plan.get("automatic_fallback_allowed") is False
    ):
        raise ValueError("VKP embedding route drift detected")
    route = receipt.get("route_receipt") or {}
    consent = receipt.get("local_owner_gate") or {}
    if not (
        route.get("plan_id") == plan.get("plan_id")
        and route.get("plan_revision") == plan.get("plan_revision")
        and route.get("owner_gate_receipt_hash") == gate.get("gate_revision")
        and consent.get("status") == "granted_local"
        and consent.get("local_execution_authorized") is True
        and consent.get("provider_call_authorized") is False
        and consent.get("external_io_authorized") is False
        and consent.get("owner_gate_receipt_hash") == gate.get("gate_revision")
        and consent.get("input_manifest_hash") == manifest.get("manifest_revision")
    ):
        raise ValueError("VKP embedding route or owner gate drift detected")
    runtime = run_receipt.get("runtime") or {}
    if not (
        owner.get("status") == "completed"
        and owner.get("provider_called") is False
        and owner.get("external_io_performed") is False
        and owner.get("automatic_fallback_used") is False
        and run_receipt.get("status") == "completed"
        and run_receipt.get("receipt_hash") == gateway["sha256_json"](owner)
        and runtime.get("provider_called") is False
        and runtime.get("external_io_performed") is False
        and runtime.get("automatic_retry_count") == 0
        and runtime.get("automatic_fallback_used") is False
    ):
        raise ValueError("VKP embedding owner receipt drift detected")
    output = owner.get("output") or {}
    candidates = receipt.get("candidate_refs") or []
    evidence = receipt.get("evidence_refs") or []
    expected_candidate = {**output, "candidate_status": "candidate_only"}
    expected_evidence = [
        {**row, "candidate_status": "candidate_only"} for row in manifest["artifacts"]
    ]
    if candidates != [expected_candidate] or evidence != expected_evidence:
        raise ValueError("VKP embedding candidate reference drift detected")
    if output.get("dimensions") != binding.get("expected_dimensions"):
        raise ValueError("VKP embedding dimension drift detected")
    if output.get("model_revision") != binding.get("model_revision"):
        raise ValueError("VKP embedding model hash drift detected")
    if not (
        receipt.get("execution_performed") is True
        and receipt.get("provider_called") is False
        and receipt.get("external_io_performed") is False
        and receipt.get("candidate_status") == "candidate_only"
        and receipt.get("human_review_required") is True
    ):
        raise ValueError("VKP embedding candidate semantics drift detected")
    return receipt


def _load_gateway_api() -> dict[str, Any]:
    owner = _shared("owner_execution")
    control = _shared("capability_control")
    canonical = _shared("canonical")
    observed = {
        "owner_plan_schema": control.OWNER_PLAN_SCHEMA,
        "owner_receipt_schema": control.OWNER_RECEIPT_SCHEMA,
        "owner_input_manifest_schema": owner.OWNER_INPUT_MANIFEST_SCHEMA,
        "owner_gate_schema": owner.OWNER_GATE_SCHEMA,
        "owner_run_report_schema": owner.OWNER_RUN_REPORT_SCHEMA,
    }
    if observed != EXPECTED_GATEWAY_CONTRACT:
        raise EmbeddingGatewaySemanticIncompatible(
            "Gateway owner-capability schema drift"
        )
    try:
        package_version = metadata.version("model-provider-gateway")
    except metadata.PackageNotFoundError:
        package_version = "workspace"
    return {
        "build_owner_input_manifest": owner.build_owner_input_manifest,
        "validate_owner_input_manifest": owner.validate_owner_input_manifest,
        "validate_owner_gate": owner.validate_owner_gate,
        "execute_owner_capability": owner.execute_owner_capability,
        "validate_owner_capability_receipt": control.validate_owner_capability_receipt,
        "sha256_json": canonical.sha256_json,
        "semantic_contract": observed,
        "package_version": package_version,
    }


def _validate_owner_bundle(
    bundle: dict[str, Any],
    *,
    manifest: dict[str, Any],
    artifact_paths: list[Path],
    runtime_binding: dict[str, Any],
    gateway: dict[str, Any],
) -> None:
    if bundle.get("manifest") != manifest:
        raise EmbeddingOwnerExecutionBlocked("input_hash_drift")
    plan = bundle.get("plan")
    gate = bundle.get("gate")
    owner = bundle.get("receipt")
    report = bundle.get("report")
    if not all(isinstance(value, dict) for value in (plan, gate, owner, report)):
        raise EmbeddingOwnerExecutionBlocked("owner_bundle_incomplete")
    gateway["validate_owner_input_manifest"](manifest, artifact_paths=artifact_paths)
    gateway["validate_owner_gate"](gate, plan=plan, manifest=manifest)
    gateway["validate_owner_capability_receipt"](owner, plan=plan)
    if owner.get("status") != "completed" or report.get("status") != "completed":
        error = owner.get("error") or report.get("error") or {}
        raise EmbeddingOwnerExecutionBlocked(
            str(error.get("category") or "owner_execution_failed")
        )
    output = owner.get("output") or {}
    if output.get("dimensions") != runtime_binding.get("expected_dimensions"):
        raise EmbeddingOwnerExecutionBlocked("dimension_drift")
    if output.get("item_count") != len(artifact_paths):
        raise EmbeddingOwnerExecutionBlocked("item_count_drift")
    if output.get("model_revision") != runtime_binding.get("model_revision"):
        raise EmbeddingOwnerExecutionBlocked("model_hash_drift")
    if report.get("receipt_hash") != gateway["sha256_json"](owner):
        raise EmbeddingOwnerExecutionBlocked("receipt_hash_drift")
    runtime = report.get("runtime") or {}
    if not (
        runtime.get("provider_called") is False
        and runtime.get("external_io_performed") is False
        and runtime.get("automatic_retry_count") == 0
        and runtime.get("automatic_fallback_used") is False
    ):
        raise EmbeddingOwnerExecutionBlocked("unauthorized_fallback")


def _candidate_receipt(
    *,
    request_id: str,
    fingerprint: str,
    bundle: dict[str, Any],
    runtime_binding: dict[str, Any],
    gateway: dict[str, Any],
) -> dict[str, Any]:
    manifest = bundle["manifest"]
    plan = bundle["plan"]
    gate = bundle["gate"]
    owner = bundle["receipt"]
    report = bundle["report"]
    output = owner["output"]
    runtime = report["runtime"]
    return {
        "schema": EMBEDDING_RECEIPT_SCHEMA,
        "ok": True,
        "status": "candidate",
        "consumer_id": "vkp",
        "task": "embedding_candidate",
        "request_id": request_id,
        "request_fingerprint": fingerprint,
        "gateway_contract": {
            **gateway["semantic_contract"],
            "package_version": gateway["package_version"],
            "compatibility_basis": "semantic_schema_not_git_identity",
            "git_commit_is_compatibility_gate": False,
        },
        "profile_ref": {
            "mode": "owner_adapter",
            "capability": "embedding",
            "adapter_id": "sentence-transformers-local",
        },
        "route_receipt": {
            "plan_id": plan["plan_id"],
            "plan_revision": plan["plan_revision"],
            "owner_gate_receipt_hash": gate["gate_revision"],
        },
        "local_owner_gate": {
            "status": "granted_local",
            "local_execution_authorized": True,
            "provider_call_authorized": False,
            "external_io_authorized": False,
            "owner_gate_receipt_hash": gate["gate_revision"],
            "input_manifest_hash": manifest["manifest_revision"],
        },
        "input_manifest": manifest,
        "owner_plan": plan,
        "owner_gate": gate,
        "owner_receipt": owner,
        "owner_run_receipt": {
            "schema": report["schema"],
            "status": report["status"],
            "capability": report["capability"],
            "adapter_id": report["adapter_id"],
            "plan_revision": report["plan_revision"],
            "input_manifest_hash": report["input_manifest_hash"],
            "gate_revision": report["gate_revision"],
            "receipt_hash": report["receipt_hash"],
            "runtime": {
                "execution_location": runtime["execution_location"],
                "runtime_id": runtime["runtime_id"],
                "provider_called": runtime["provider_called"],
                "external_io_performed": runtime["external_io_performed"],
                "automatic_retry_count": runtime["automatic_retry_count"],
                "automatic_fallback_used": runtime["automatic_fallback_used"],
            },
            "output_refs": report["output_refs"],
            "error": report["error"],
        },
        "runtime_binding": runtime_binding,
        "evidence_refs": [
            {**row, "candidate_status": "candidate_only"}
            for row in manifest["artifacts"]
        ],
        "candidate_refs": [{**output, "candidate_status": "candidate_only"}],
        "capability_truth": {
            "contract_ready": True,
            "runtime_ready": True,
            "fresh_e2e_pass": True,
            "real_usage_proof": False,
            "current_online": False,
        },
        "execution_performed": True,
        "provider_called": False,
        "external_io_performed": False,
        "candidate_status": "candidate_only",
        "human_review_required": True,
        "automatic_fallback_allowed": False,
        "automatic_retry_allowed": False,
    }


def _blocked_receipt(
    *,
    request_id: str,
    fingerprint: str,
    error_class: str,
    gateway: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": EMBEDDING_RECEIPT_SCHEMA,
        "ok": False,
        "status": "blocked",
        "consumer_id": "vkp",
        "task": "embedding_candidate",
        "request_id": request_id,
        "request_fingerprint": fingerprint,
        "gateway_contract": dict(
            (gateway or {}).get("semantic_contract") or EXPECTED_GATEWAY_CONTRACT
        ),
        "error_class": error_class,
        "error_message": "VKP embedding candidate failed closed",
        "candidate_refs": [],
        "execution_performed": False,
        "provider_called": False,
        "external_io_performed": False,
        "candidate_status": "unavailable",
        "human_review_required": True,
        "automatic_fallback_allowed": False,
        "automatic_retry_allowed": False,
    }


def _read_existing(
    path: Path,
    *,
    fingerprint: str,
    artifact_paths: list[Path],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("existing VKP embedding receipt is unreadable") from exc
    if value.get("request_fingerprint") != fingerprint:
        raise ValueError("VKP embedding request drift detected")
    return validate_embedding_candidate_receipt(value, artifact_paths=artifact_paths)


def _artifact_identities(
    paths: list[Path], *, mime_types: list[str] | None
) -> list[dict[str, Any]]:
    if not paths:
        raise ValueError("VKP embedding requires at least one artifact")
    if mime_types is not None and len(mime_types) != len(paths):
        raise ValueError("mime_types must match artifact_paths")
    rows = []
    for index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(f"embedding artifact not found: {path}")
        mime_type = (
            mime_types[index]
            if mime_types is not None
            else (mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        )
        rows.append(
            {
                "artifact_id": f"artifact-{index + 1:04d}",
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "mime_type": mime_type.lower(),
            }
        )
    return rows


def _runtime_binding(
    *,
    embedding_python: str | Path | None,
    embedding_model: str | Path | None,
    embedding_device: str,
    expected_dimensions: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    python_path = (
        Path(embedding_python).expanduser().resolve() if embedding_python else None
    )
    model_path = (
        Path(embedding_model).expanduser().resolve() if embedding_model else None
    )
    modules_path = model_path / "modules.json" if model_path else None
    return {
        "runtime_path_provided": embedding_python is not None,
        "runtime_sha256": sha256_file(python_path)
        if python_path and python_path.is_file()
        else None,
        "model_path_provided": embedding_model is not None,
        "model_revision": sha256_file(modules_path)
        if modules_path and modules_path.is_file()
        else None,
        "model_source": "explicit_local_files_only",
        "device": embedding_device,
        "expected_dimensions": int(expected_dimensions),
        "timeout_seconds": int(timeout_seconds),
    }


def _require_local_runtime(binding: dict[str, Any]) -> None:
    if not binding.get("runtime_path_provided") or not binding.get("runtime_sha256"):
        raise EmbeddingOwnerExecutionBlocked("runtime_missing")
    if not binding.get("model_path_provided") or not binding.get("model_revision"):
        raise EmbeddingOwnerExecutionBlocked("model_missing")
    if binding.get("device") not in {"cpu", "cuda"}:
        raise EmbeddingOwnerExecutionBlocked("runtime_device_invalid")
    if int(binding.get("expected_dimensions") or 0) <= 0:
        raise EmbeddingOwnerExecutionBlocked("dimension_contract_invalid")
    if int(binding.get("timeout_seconds") or 0) <= 0:
        raise EmbeddingOwnerExecutionBlocked("timeout_contract_invalid")


def _safe_error_class(exc: Exception) -> str:
    if isinstance(exc, EmbeddingOwnerExecutionBlocked):
        return exc.error_class
    if isinstance(exc, SharedGatewayUnavailable):
        return "gateway_unavailable"
    if isinstance(exc, EmbeddingGatewaySemanticIncompatible):
        return "gateway_semantic_incompatible"
    if isinstance(exc, TimeoutError):
        return "timeout"
    text = str(exc).lower()
    if "fallback" in text:
        return "unauthorized_fallback"
    if "dimension" in text:
        return "dimension_drift"
    if "model" in text and ("missing" in text or "not found" in text):
        return "model_missing"
    if "runtime" in text and ("missing" in text or "not found" in text):
        return "runtime_missing"
    if "hash" in text:
        return "input_or_receipt_hash_drift"
    if "route" in text or "gate" in text or "revision" in text:
        return "route_drift"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    return "embedding_contract_blocked"
