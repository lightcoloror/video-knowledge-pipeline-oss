"""Thin implementation of the workbench-owned VkpPort, imported on demand."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Callable

from .canonical_json import canonical_json_sha256
from .server_execution import ServerExecution
from .storage import bundle_write_lock, read_json, write_json


def project_workbench_result(binding: dict, native: dict, gates: list, attempt: int) -> dict:
    """Reuse owner validation/evaluation. Never turn native failure into passed."""
    from workbench.result_contract import ResultReceipt
    if not isinstance(native, dict) or native.get("schema") != "lecture_acceptance_check.v1":
        raise ValueError("native acceptance snapshot required")
    formal = (native.get("status") == "complete"
              and native.get("summary", {}).get("smart_summary_production_ready") is True
              and not native.get("blockers") and native.get("candidate_only") is not True
              and native.get("publication_allowed") is not False and native.get("formal_completed") is not False)
    gate_map = {g.get("key"): g for g in gates if isinstance(g, dict)}
    channels = {"transcript_quality_failed": "transcript", "transcript_semantic_high_risk_review": "transcript",
                "smart_summary_quality_failed": "summary", "screen_text": "ocr"}
    for item in native.get("blockers", []):
        key = channels.get(item.get("key"))
        if key and gate_map.get(key, {}).get("state") == "passed":
            raise ValueError("gate contradicts native blocker: " + key)
    receipt = ResultReceipt.model_validate({"schema": "video_workbench.result.v1", "binding": binding,
        "attempt": attempt, "media": {"kind": binding["media_kind"], "has_speech": binding["has_speech"],
        "probe_sha256": binding["probe_sha256"]}, "native": {
            "acceptance": "passed" if formal else "failed" if native.get("status") == "failed" else "incomplete",
            "candidate_only": not formal, "publication_allowed": formal}, "gates": gates})
    return receipt.model_dump()


class WorkbenchVkpPort:
    """Host authenticates/adopts the binding; this port starts no worker loop.

    result_factory captures native acceptance and six evidence-backed gates.
    It must not call a model. Draft pricing/N/A/file rules stay with the owner.
    """

    def __init__(self, service: ServerExecution, *, owner: str, input_root: str | Path, plan: dict,
                 lease_guard: Callable[[], bool], result_factory: Callable[[dict], dict],
                 reconcile: Callable[[str], dict | None] | None = None):
        self.service, self.owner = service, owner
        self.input_root, self.plan = Path(input_root).resolve(), plan
        self.lease_guard, self.result_factory, self.reconcile = lease_guard, result_factory, reconcile

    def _binding(self, value: dict) -> dict:
        from workbench.result_contract import ExecutionBinding
        b = ExecutionBinding.model_validate(value).model_dump()
        release = self.service.release
        refs = {r["id"]: r for r in self.plan["inputs"]}
        if (b["vkp_commit"] != release["git_head"] or b["vkp_source_sha256"] != release["source"]["sha256"]
                or b["config_sha256"] != canonical_json_sha256(self.plan)
                or refs.get("media", {}).get("sha256") != b["media_sha256"]
                or refs.get("probe", {}).get("sha256") != b["probe_sha256"]):
            raise ValueError("workbench binding differs from sealed source/config/media/probe")
        return b

    def _read_binding(self, run_id: str) -> dict:
        root = self.service._root(self.owner, run_id)
        value = read_json(root / "workbench-binding.json")
        return self._binding(value)

    def _response(self, run_id: str, *, state: dict | None = None) -> dict:
        state = state or self.service.status(self.owner, run_id)
        native = state["status"]
        wire = {"complete_draft": "succeeded", "cancelled": "cancelled", "failed": "failed",
                "blocked": "failed", "queued": "queued", "paused": "queued"}.get(native, "running")
        return {"run_id": run_id, "binding": self._read_binding(run_id), "state": wire,
                "native_execution_state": native, "execution_safety_v1": False,
                "provider_cancellation": state.get("provider_cancellation", "not_confirmed")}

    def start(self, *, binding: dict, operation_id: str) -> dict:
        if not re.fullmatch(r"[a-f0-9]{32}", str(operation_id)):
            raise ValueError("stable operation_id required")
        b = self._binding(binding)
        operation = canonical_json_sha256({"job": b["job_id"], "operation": operation_id})
        state = self.service.start(self.owner, operation, input_root=self.input_root, plan=self.plan)
        root = self.service._root(self.owner, state["job_id"])
        with bundle_write_lock(root, operation="pin_workbench_binding", stale_after_seconds=300):
            target = root / "workbench-binding.json"
            if target.exists():
                if read_json(target) != b:
                    raise ValueError("immutable workbench binding drift")
            else:
                write_json(target, b)
        return self._response(state["job_id"], state=state)

    def status(self, *, run_id: str) -> dict:
        return self._response(run_id)

    def cancel(self, *, run_id: str, operation_id: str) -> dict:
        self._read_binding(run_id)
        state = self.service.cancel(self.owner, run_id)
        return self._response(run_id, state=state)

    def resume(self, *, run_id: str, binding: dict, operation_id: str) -> dict:
        if self._binding(binding) != self._read_binding(run_id):
            raise ValueError("resume binding drift")
        state = self.service.resume(self.owner, run_id, lease_guard=self.lease_guard, max_steps=32, reconcile=self.reconcile)
        if state["status"] == "complete_draft":
            root = self.service._root(self.owner, run_id)
            with bundle_write_lock(root, operation="project_workbench_result", stale_after_seconds=300):
                target = root / "workbench-result.json"
                if not target.exists():
                    material = self.result_factory(self.service.result(self.owner, run_id))
                    native = material["native_acceptance"]
                    receipt = project_workbench_result(binding, native, material["gates"], material["attempt"])
                    payload = {"binding": binding, "native_acceptance": native, "result_receipt": receipt}
                    write_json(target, {"payload": payload, "sha256": canonical_json_sha256(payload)})
        return self._response(run_id, state=state)

    def result(self, *, run_id: str) -> dict:
        response = self._response(run_id)
        if response["state"] != "succeeded":
            return response
        root = self.service._root(self.owner, run_id)
        envelope = read_json(root / "workbench-result.json")
        payload = envelope["payload"]
        if canonical_json_sha256(payload) != envelope["sha256"] or payload["binding"] != response["binding"]:
            raise ValueError("workbench result receipt drift")
        response["result_receipt"] = payload["result_receipt"]
        return response
