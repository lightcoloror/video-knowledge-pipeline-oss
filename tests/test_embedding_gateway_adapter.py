from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import pytest

from video_knowledge_pipeline import embedding_gateway_adapter
from video_knowledge_pipeline.embedding_gateway_adapter import (
    EMBEDDING_RECEIPT_SCHEMA,
    run_embedding_candidate,
    validate_embedding_candidate_receipt,
)
from video_knowledge_pipeline.model_provider_gateway_adapter import (
    SharedGatewayUnavailable,
)


def _artifact(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "synthetic-public-candidate.json"
    path.write_text('{"candidate":"video knowledge only"}', encoding="utf-8")
    return path


def _model_contract(root: Path) -> Path:
    model = root / "synthetic-model-contract"
    model.mkdir(parents=True, exist_ok=True)
    (model / "modules.json").write_text("{}", encoding="utf-8")
    return model


def _completed_gateway(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, list[dict]]:
    gateway = embedding_gateway_adapter._load_gateway_api()
    calls: list[dict] = []

    def completed(
        capability,
        adapter_id,
        artifact_paths,
        *,
        output_dir,
        consumer_id,
        purpose,
        data_classification,
        runtime,
        timeout_seconds,
    ):
        from model_provider_gateway.capability_control import (
            build_owner_capability_plan,
        )
        from model_provider_gateway.owner_execution import build_owner_gate

        calls.append(
            {
                "capability": capability,
                "adapter_id": adapter_id,
                "consumer_id": consumer_id,
                "runtime": dict(runtime),
                "timeout_seconds": timeout_seconds,
            }
        )
        manifest = gateway["build_owner_input_manifest"](
            artifact_paths,
            purpose=purpose,
            data_classification=data_classification,
        )
        plan = build_owner_capability_plan(
            capability,
            adapter_id,
            consumer_id=consumer_id,
            input_manifest_hash=manifest["manifest_revision"],
            options={
                "timeout_seconds": timeout_seconds,
                "no_retry": True,
                "no_fallback": True,
            },
        )
        gate = build_owner_gate(plan, manifest)
        output = {
            "kind": "embedding_manifest",
            "vectors_sha256": "a" * 64,
            "dimensions": 384,
            "item_count": len(artifact_paths),
            "model_revision": runtime["model_revision"],
        }
        owner = {
            "schema": "model_provider_gateway.owner_capability_receipt.v1",
            "status": "completed",
            "plan_id": plan["plan_id"],
            "plan_revision": plan["plan_revision"],
            "consumer_id": consumer_id,
            "capability": capability,
            "adapter_id": adapter_id,
            "input_manifest_hash": manifest["manifest_revision"],
            "owner_gate_receipt_hash": gate["gate_revision"],
            "external_io_performed": False,
            "provider_called": False,
            "automatic_fallback_used": False,
            "output": output,
            "error": None,
        }
        gateway["validate_owner_capability_receipt"](owner, plan=plan)
        report = {
            "schema": "model_provider_gateway.owner_runtime_run_report.v1",
            "status": "completed",
            "capability": capability,
            "adapter_id": adapter_id,
            "plan_revision": plan["plan_revision"],
            "input_manifest_hash": manifest["manifest_revision"],
            "gate_revision": gate["gate_revision"],
            "receipt_hash": gateway["sha256_json"](owner),
            "runtime": {
                "execution_location": "local",
                "runtime_id": "synthetic-contract-only",
                "latency_ms": 7,
                "provider_called": False,
                "external_io_performed": False,
                "automatic_retry_count": 0,
                "automatic_fallback_used": False,
            },
            "output_refs": ["embedding-vectors.json"],
            "error": None,
        }
        return {
            "manifest": manifest,
            "plan": plan,
            "gate": gate,
            "receipt": owner,
            "report": report,
        }

    gateway["execute_owner_capability"] = completed
    monkeypatch.setattr(embedding_gateway_adapter, "_load_gateway_api", lambda: gateway)
    return gateway, calls


@pytest.fixture
def contract_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict, list[dict]]:
    artifact = _artifact(tmp_path)
    model = _model_contract(tmp_path)
    _, calls = _completed_gateway(monkeypatch)
    receipt_path = tmp_path / "candidate.json"
    kwargs = {
        "artifact_paths": [artifact],
        "purpose": "Synthetic VKP contract candidate.",
        "request_id": "vkp-embedding-contract",
        "receipt_path": receipt_path,
        "embedding_python": sys.executable,
        "embedding_model": model,
        "owner_output_dir": tmp_path / "owner-run",
    }
    first = run_embedding_candidate(**kwargs)
    first_bytes = receipt_path.read_bytes()
    second = run_embedding_candidate(**kwargs)
    assert first == second
    assert first_bytes == receipt_path.read_bytes()
    assert len(calls) == 1
    return artifact, first, calls


def test_candidate_is_thin_route_bound_and_idempotent(contract_candidate) -> None:
    artifact, receipt, calls = contract_candidate

    assert receipt["schema"] == EMBEDDING_RECEIPT_SCHEMA
    assert receipt["status"] == "candidate"
    assert receipt["owner_receipt"]["status"] == "completed"
    assert receipt["candidate_refs"][0]["dimensions"] == 384
    assert receipt["candidate_status"] == "candidate_only"
    assert (
        receipt["gateway_contract"]["compatibility_basis"]
        == "semantic_schema_not_git_identity"
    )
    assert receipt["gateway_contract"]["git_commit_is_compatibility_gate"] is False
    assert receipt["provider_called"] is False
    assert receipt["external_io_performed"] is False
    assert all("vectors" not in row for row in receipt["candidate_refs"])
    assert calls[0]["consumer_id"] == "vkp"
    assert calls[0]["adapter_id"] == "sentence-transformers-local"
    assert (
        validate_embedding_candidate_receipt(receipt, artifact_paths=[artifact])
        is receipt
    )


def test_gateway_runtime_and_model_missing_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(tmp_path)
    model = _model_contract(tmp_path)
    monkeypatch.setattr(
        embedding_gateway_adapter,
        "_load_gateway_api",
        lambda: (_ for _ in ()).throw(
            SharedGatewayUnavailable("synthetic unavailable")
        ),
    )
    unavailable = run_embedding_candidate(
        artifact_paths=[artifact],
        purpose="Synthetic gateway unavailable.",
        request_id="vkp-gateway-unavailable",
        receipt_path=tmp_path / "unavailable.json",
        embedding_python=sys.executable,
        embedding_model=model,
    )
    runtime_missing = run_embedding_candidate(
        artifact_paths=[artifact],
        purpose="Synthetic runtime missing.",
        request_id="vkp-runtime-missing",
        receipt_path=tmp_path / "runtime-missing.json",
        embedding_python=tmp_path / "missing-python.exe",
        embedding_model=model,
    )
    model_missing = run_embedding_candidate(
        artifact_paths=[artifact],
        purpose="Synthetic model missing.",
        request_id="vkp-model-missing",
        receipt_path=tmp_path / "model-missing.json",
        embedding_python=sys.executable,
        embedding_model=tmp_path / "missing-model",
    )

    assert unavailable["error_class"] == "gateway_unavailable"
    assert runtime_missing["error_class"] == "runtime_missing"
    assert model_missing["error_class"] == "model_missing"
    for value in (unavailable, runtime_missing, model_missing):
        assert value["status"] == "blocked"
        assert value["candidate_refs"] == []
        assert value["automatic_fallback_allowed"] is False


def test_timeout_fails_closed_without_retry_or_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(tmp_path)
    model = _model_contract(tmp_path)
    gateway = embedding_gateway_adapter._load_gateway_api()

    def timeout(*_args, **_kwargs):
        raise TimeoutError("synthetic owner execution timed out")

    gateway["execute_owner_capability"] = timeout
    monkeypatch.setattr(embedding_gateway_adapter, "_load_gateway_api", lambda: gateway)
    result = run_embedding_candidate(
        artifact_paths=[artifact],
        purpose="Synthetic timeout.",
        request_id="vkp-embedding-timeout",
        receipt_path=tmp_path / "timeout.json",
        embedding_python=sys.executable,
        embedding_model=model,
        timeout_seconds=1,
    )

    assert result["status"] == "blocked"
    assert result["error_class"] == "timeout"
    assert result["automatic_retry_allowed"] is False
    assert result["automatic_fallback_allowed"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["owner_run_receipt"].__setitem__(
                "receipt_hash", "0" * 64
            ),
            "owner receipt drift",
        ),
        (
            lambda value: value["owner_plan"].__setitem__("plan_revision", "0" * 64),
            "was expected|drift",
        ),
        (
            lambda value: value["runtime_binding"].__setitem__(
                "expected_dimensions", 768
            ),
            "dimension drift",
        ),
        (
            lambda value: value["owner_receipt"].__setitem__(
                "automatic_fallback_used", True
            ),
            "False was expected|fallback",
        ),
        (
            lambda value: value.__setitem__("embedding_vectors", [[0.1]]),
            "vectors or verified facts",
        ),
    ],
)
def test_receipt_hash_route_dimension_and_fallback_drift_fail_closed(
    contract_candidate,
    mutation,
    message: str,
) -> None:
    artifact, receipt, _ = contract_candidate
    drifted = copy.deepcopy(receipt)
    mutation(drifted)

    with pytest.raises(Exception, match=message):
        validate_embedding_candidate_receipt(drifted, artifact_paths=[artifact])


def test_input_hash_drift_fails_closed(contract_candidate) -> None:
    artifact, receipt, _ = contract_candidate
    artifact.write_text('{"candidate":"drifted"}', encoding="utf-8")

    with pytest.raises(ValueError, match="input_hash_mismatch"):
        validate_embedding_candidate_receipt(receipt, artifact_paths=[artifact])


def test_real_local_embedding_twice_when_runtime_is_available(tmp_path: Path) -> None:
    python_path = Path(os.environ.get("MPG_EMBEDDING_PYTHON", ""))
    model_path = Path(os.environ.get("MPG_EMBEDDING_MODEL_PATH", ""))
    if not python_path.is_file() or not (model_path / "modules.json").is_file():
        pytest.skip("explicit local embedding runtime/model are not configured")
    artifact = _artifact(tmp_path)
    common = {
        "artifact_paths": [artifact],
        "purpose": "Synthetic real VKP embedding.",
        "request_id": "vkp-real-embedding-twice",
        "embedding_python": python_path,
        "embedding_model": model_path,
    }
    first = run_embedding_candidate(
        **common,
        receipt_path=tmp_path / "real-a.json",
        owner_output_dir=tmp_path / "real-owner-a",
    )
    if first.get("status") == "blocked":
        pytest.skip(f"current isolated runtime blocker: {first.get('error_class')}")
    second = run_embedding_candidate(
        **common,
        receipt_path=tmp_path / "real-b.json",
        owner_output_dir=tmp_path / "real-owner-b",
    )

    assert first == second
    assert (tmp_path / "real-a.json").read_bytes() == (
        tmp_path / "real-b.json"
    ).read_bytes()
    assert first["candidate_refs"][0]["dimensions"] == 384
    assert first["provider_called"] is False
    assert first["external_io_performed"] is False
