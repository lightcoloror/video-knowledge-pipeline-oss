"""Run with the fixed workbench contract package; no server or model calls."""
# ruff: noqa: F811 -- pytest fixture import is intentionally visible to this module.
from copy import deepcopy
import pytest
pytest.importorskip("workbench.result_contract")
from workbench.result_contract import ExecutionBinding, evaluate
from workbench.vkp_adapter import VkpAdapter
from video_knowledge_pipeline.server_workbench_port import WorkbenchVkpPort, project_workbench_result
from video_knowledge_pipeline.canonical_json import canonical_json_sha256
from video_knowledge_pipeline.file_hash import sha256_file
from test_server_api_execution import case  # noqa: F401


@pytest.fixture
def wire(case):
    for name in ("media", "probe"):
        path = case["inputs"] / (name + ".txt")
        path.write_text("synthetic owner evidence " + name, encoding="utf-8")
        case["plan"]["inputs"].append({"id": name, "path": path.name, "sha256": sha256_file(path)})
    refs = {r["id"]: r for r in case["plan"]["inputs"]}
    binding = {"job_id": "a" * 32, "media_sha256": refs["media"]["sha256"], "probe_sha256": refs["probe"]["sha256"],
        "duration_seconds": 17.0, "vkp_commit": case["release"]["git_head"],
        "vkp_source_sha256": case["release"]["source"]["sha256"], "config_sha256": canonical_json_sha256(case["plan"]),
        "media_kind": "video", "has_speech": True}
    native = {"schema": "lecture_acceptance_check.v1", "status": "incomplete", "summary": {},
              "blockers": [{"key": "screen_text", "detail": "synthetic uncertainty"}]}
    gates = [{"key": k, "state": "needs_review" if k == "ocr" else "passed", "message": "synthetic evidence",
              "evidence": ["report.html#" + k]} for k in ["transcript", "summary", "ocr", "shots", "visual", "temporal"]]
    calls = []
    def result_factory(result):
        calls.append("native_projection")
        return {"native_acceptance": native, "gates": gates, "attempt": 1}
    port = WorkbenchVkpPort(case["service"], owner="account-a", input_root=case["inputs"], plan=case["plan"],
        lease_guard=lambda: True, result_factory=result_factory)
    return case, binding, native, gates, port, calls


def test_actual_owner_adapter_start_resume_result_and_restart(wire):
    case, binding, native, gates, port, calls = wire
    adapter = VkpAdapter(ExecutionBinding.model_validate(binding), port=port)
    first = adapter.start("1" * 32)
    assert port.start(binding=binding, operation_id="1" * 32)["run_id"] == first["run_id"]
    assert adapter.resume("2" * 32)["state"] == "succeeded"
    receipt = adapter.result()
    assert evaluate(receipt)["mode"] == "draft" and not evaluate(receipt)["settle_points"]
    assert receipt.native.candidate_only and not receipt.native.publication_allowed
    restored = VkpAdapter(ExecutionBinding.model_validate(binding), port=port, run_id=first["run_id"])
    restored.resume("3" * 32)
    assert restored.result() == receipt and calls == ["native_projection"]
    assert len(case["calls"]) == 2


@pytest.mark.parametrize("field", ["vkp_commit", "vkp_source_sha256", "media_sha256", "probe_sha256", "config_sha256"])
def test_owner_binding_drift_rejected_without_calls(wire, field):
    case, binding, native, gates, port, calls = wire
    wrong = deepcopy(binding)
    wrong[field] = "b" * len(wrong[field])
    with pytest.raises(ValueError, match="binding"):
        port.start(binding=wrong, operation_id="1" * 32)
    assert not case["calls"]


def test_owner_cancel_does_not_claim_provider_billing_stopped(wire):
    case, binding, native, gates, port, calls = wire
    adapter = VkpAdapter(ExecutionBinding.model_validate(binding), port=port)
    adapter.start("1" * 32)
    assert adapter.cancel("2" * 32)["provider_cancellation"] == "not_confirmed"
    assert adapter.resume("3" * 32)["state"] == "cancelled" and not case["calls"]


def test_native_failure_cannot_be_hidden_by_passed_gate(wire):
    case, binding, native, gates, port, calls = wire
    gates[2]["state"] = "passed"
    with pytest.raises(ValueError, match="contradicts"):
        project_workbench_result(binding, native, gates, 1)


def test_owner_contract_missing_or_unknown_gate_rejected(wire):
    case, binding, native, gates, port, calls = wire
    with pytest.raises(ValueError):
        project_workbench_result(binding, native, gates[:-1], 1)
