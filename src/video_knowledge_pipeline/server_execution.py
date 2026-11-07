"""Durable, bounded server API adapter over native storage/hash/lock contracts.

No downloads, local models, Codex, process launcher or provider client. The host
registers reviewed guarded API backends; unavailable ASR/OCR routes block rather
than migrating to a text model or local pipeline. Deterministic/prepared evidence
enters as hash-bound artifacts from its existing owner.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Callable
from uuid import uuid4

from jsonschema import Draft202012Validator

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file
from .server_gateway import ApiStageBackend, ServerAdmissionRejected
from .server_release import verify_server_release
from .storage import bundle_write_lock, read_json, write_json

SCHEMA = "video_knowledge_pipeline.server_job.v1"
PLAN_SCHEMA = "video_knowledge_pipeline.server_plan.v1"
STAGE_CAPABILITIES = {
    "primary_asr": {"asr"}, "secondary_asr": {"asr"}, "alignment": {"alignment"},
    "ocr": {"ocr", "vision"}, "vision": {"vision"}, "temporal": {"vision"},
    "semantic": {"text"}, "summary": {"text"},
}
TERMINAL = {"complete_draft", "cancelled", "failed"}


def _job_lock(root: Path, *, operation: str):
    return bundle_write_lock(root, operation=operation, stale_after_seconds=300)


def _relative(root: Path, name: str) -> Path:
    parts = PurePosixPath(name).parts if isinstance(name, str) else ()
    if (not parts or PurePosixPath(name).is_absolute() or "\\" in name or ":" in name
            or any(x in {".", ".."} for x in parts) or name != "/".join(parts)):
        raise ValueError("artifact path must be a normalized relative POSIX path")
    path = root.joinpath(*parts)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes its owner root")
    return path


def _save(root: Path, state: dict[str, Any]) -> None:
    payload = {k: v for k, v in state.items() if k != "state_sha256"}
    state["state_sha256"] = canonical_json_sha256(payload)
    write_json(root / "state.json", state)


def _load(root: Path) -> dict[str, Any]:
    state = read_json(root / "state.json")
    if not isinstance(state, dict) or state.get("schema") != SCHEMA:
        raise ValueError("invalid server job")
    schema = json.loads((Path(__file__).parent / "schemas/server-job.v1.schema.json").read_text(encoding="utf-8"))
    try:
        Draft202012Validator(schema).validate(state)
    except Exception as exc:
        raise ValueError("invalid server job state structure") from exc
    if canonical_json_sha256({k: v for k, v in state.items() if k != "state_sha256"}) != state.get("state_sha256"):
        raise ValueError("job state drift")
    return state


def _validate_plan(plan: dict[str, Any]) -> None:
    schema = json.loads((Path(__file__).parent / "schemas/server-plan.v1.schema.json").read_text(encoding="utf-8"))
    try:
        Draft202012Validator(schema).validate(plan)
    except Exception as exc:
        raise ValueError("server plan JSON Schema validation failed") from exc
    if (not isinstance(plan, dict) or set(plan) != {"schema", "inputs", "stages"}
            or plan.get("schema") != PLAN_SCHEMA or not isinstance(plan["inputs"], list)
            or not isinstance(plan["stages"], list) or not 1 <= len(plan["stages"]) <= 32):
        raise ValueError("invalid bounded server plan")
    seen = set()
    for row in plan["inputs"]:
        if not isinstance(row, dict) or set(row) != {"id", "path", "sha256"}:
            raise ValueError("invalid input reference")
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", str(row["id"])) or row["id"] in seen:
            raise ValueError("duplicate/invalid input id")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])):
            raise ValueError("input SHA-256 required")
        seen.add(row["id"])
    stage_ids = set()
    for stage in plan["stages"]:
        if not isinstance(stage, dict) or stage.get("id") not in STAGE_CAPABILITIES:
            raise ValueError("unknown stage")
        if stage["id"] in seen or stage["id"] in stage_ids:
            raise ValueError("duplicate stage id")
        if not isinstance(stage.get("input_ids"), list) or not stage["input_ids"] or not set(stage["input_ids"]) <= seen:
            raise ValueError("stage dependencies must already exist")
        if stage.get("kind") == "artifact":
            if set(stage) != {"id", "kind", "input_ids"}:
                raise ValueError("artifact stage accepts references only")
        elif stage.get("kind") == "api":
            if set(stage) != {"id", "kind", "input_ids", "backend", "backend_revision", "output_schema"}:
                raise ValueError("invalid API stage")
            if not isinstance(stage["backend"], str) or not re.fullmatch(r"[0-9a-f]{64}", str(stage["backend_revision"])):
                raise ValueError("API backend must be explicitly pinned")
            schema = stage["output_schema"]
            if not isinstance(schema, dict) or schema.get("type") != "object":
                raise ValueError("object output schema required")
            # Output validation must never retrieve a remote JSON Schema.
            def check_refs(value):
                if isinstance(value, dict):
                    if any(k in value and not str(value[k]).startswith("#") for k in ("$ref", "$dynamicRef", "$recursiveRef")):
                        raise ValueError("external schema reference forbidden")
                    for child in value.values():
                        check_refs(child)
                elif isinstance(value, list):
                    for child in value:
                        check_refs(child)
            check_refs(schema)
            Draft202012Validator.check_schema(schema)
        else:
            raise ValueError("server plan only accepts API calls or owner-produced artifacts")
        stage_ids.add(stage["id"])
        seen.add(stage["id"])


class ServerExecution:
    """In-process start/status/cancel/resume/result contract; host owns HTTP/auth.

    All paid calls require a host lease guard and an ApiStageBackend's guarded
    gateway closure. An unknown call is never automatically repeated. This is
    not a claim that the workbench's entire execution_safety_v1 is implemented.
    """

    def __init__(self, workspace: str | Path, *, source_root: str | Path, release: dict[str, Any],
                 backends: dict[str, ApiStageBackend] | None = None):
        self.workspace = Path(workspace).resolve()
        self.source_root = Path(source_root).resolve()
        self.release = deepcopy(release)
        self.backends = dict(backends or {})

    def _root(self, owner: str, job: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", str(owner)) or not re.fullmatch(r"[0-9a-f]{64}", str(job)):
            raise ValueError("invalid owner/job identity")
        root = self.workspace / owner / job
        if root.is_symlink() or not root.resolve().is_relative_to(self.workspace):
            raise ValueError("job root escapes workspace")
        return root

    def _check(self, root: Path) -> dict[str, Any]:
        proof = verify_server_release(self.source_root, self.release)
        state = _load(root)
        if state["owner_id"] != root.parent.name or state["job_id"] != root.name:
            raise ValueError("job owner identity drift")
        if state["release_sha256"] != proof["release_sha256"]:
            raise ValueError("job release drift")
        plan = read_json(root / "plan.json")
        if canonical_json_sha256(plan) != state["plan_sha256"]:
            raise ValueError("job plan drift")
        for ref in plan["inputs"]:
            path = _relative(Path(state["input_root"]), ref["path"])
            if not path.is_file() or sha256_file(path) != ref["sha256"]:
                raise ValueError("job input drift")
        for stage in state["stages"].values():
            for kind in ("receipt", "pack", "diagnostic"):
                if stage.get(kind):
                    path = _relative(root, stage[kind])
                    if not path.is_file() or sha256_file(path) != stage[kind + "_sha256"]:
                        raise ValueError("stage " + kind + " drift")
        return state

    def preflight(self, plan: dict[str, Any]) -> dict[str, Any]:
        _validate_plan(plan)
        blockers = []
        for stage in plan["stages"]:
            if stage["kind"] != "api":
                continue
            backend = self.backends.get(stage["backend"])
            try:
                if backend is None:
                    raise ValueError("missing")
                backend.validate()
                if (backend.revision != stage["backend_revision"]
                        or backend.descriptor["capability"] not in STAGE_CAPABILITIES[stage["id"]]):
                    raise ValueError("drift or unsupported stage capability")
            except ValueError:
                blockers.append({"stage": stage["id"], "code": "API_backend_unavailable_or_drifted"})
        return {"ready": not blockers, "blockers": blockers, "model_calls": 0,
                "execution_mode": "server_api_only", "execution_safety_v1": False}  # gitleaks:allow status field

    def start(self, owner: str, operation_id: str, *, input_root: str | Path, plan: dict[str, Any]) -> dict[str, Any]:
        preflight = self.preflight(plan)
        proof = verify_server_release(self.source_root, self.release)
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", operation_id):
            raise ValueError("stable operation_id required")
        job = canonical_json_sha256({"owner": owner, "operation_id": operation_id})
        root = self._root(owner, job)
        input_root = Path(input_root).resolve()
        if root.is_relative_to(input_root) or input_root.is_relative_to(root):
            raise ValueError("input evidence and server job output must not overlap")
        for ref in plan["inputs"]:
            path = _relative(input_root, ref["path"])
            if not path.is_file() or sha256_file(path) != ref["sha256"]:
                raise ValueError("input missing or drifted")
        with _job_lock(root, operation="server_start"):
            if (root / "state.json").exists():
                state = self._check(root)
                if state["plan_sha256"] != canonical_json_sha256(plan) or state["input_root"] != str(input_root):
                    raise ValueError("operation_id conflicts with previous input/configuration")
                return state
            write_json(root / "plan.json", plan)
            state = {"schema": SCHEMA, "owner_id": owner, "job_id": job, "operation_id": operation_id,
                     "input_root": str(input_root), "plan_sha256": canonical_json_sha256(plan),
                     "release_sha256": proof["release_sha256"], "status": "queued" if preflight["ready"] else "blocked",
                     "preflight": preflight, "cancel_requested": False,
                     "stages": {s["id"]: {"status": "queued", "execution_kind": s["kind"]} for s in plan["stages"]},
                     "execution_safety_v1": False, "candidate_only": True, "formal_completed": False}
            _save(root, state)
        return state

    def status(self, owner: str, job: str) -> dict[str, Any]:
        return self._check(self._root(owner, job))

    def cancel(self, owner: str, job: str) -> dict[str, Any]:
        root = self._root(owner, job)
        with _job_lock(root, operation="server_cancel"):
            # A source/input drift must prevent execution, not prevent stopping.
            state = _load(root)
            if state["owner_id"] != owner or state["job_id"] != job:
                raise ValueError("job owner identity drift")
            if state["status"] not in TERMINAL:
                state.update(status="cancelled", cancel_requested=True, provider_cancellation="not_confirmed")
                _save(root, state)
            return state

    def _context(self, root: Path, state: dict[str, Any], stage: dict[str, Any], request_id: str) -> dict[str, Any]:
        plan = read_json(root / "plan.json")
        refs = {r["id"]: {**r, "absolute_path": str(_relative(Path(state["input_root"]), r["path"]))} for r in plan["inputs"]}
        for key, row in state["stages"].items():
            if row.get("receipt"):
                refs[key] = {"id": key, "sha256": row["receipt_sha256"], "absolute_path": str(_relative(root, row["receipt"]))}
        return {"owner_id": state["owner_id"], "job_id": state["job_id"], "stage": stage["id"],
                "request_id": request_id, "inputs": [refs[k] for k in stage["input_ids"]]}

    def resume(self, owner: str, job: str, *, lease_guard: Callable[[], bool], max_steps: int = 8,
               reconcile: Callable[[str], dict[str, Any] | None] | None = None) -> dict[str, Any]:
        if not callable(lease_guard) or type(max_steps) is not int or not 1 <= max_steps <= 32:
            raise ValueError("lease guard and bounded max_steps required")
        root = self._root(owner, job)
        for _ in range(max_steps):
            with _job_lock(root, operation="server_resume"):
                state = self._check(root)
                if state["status"] in TERMINAL:
                    return state
                if lease_guard() is not True:
                    state.update(status="paused", stop_reason="lease_invalid")
                    _save(root, state)
                    return state
                plan = read_json(root / "plan.json")
                preflight = self.preflight(plan)
                if not preflight["ready"]:
                    state.update(status="blocked", preflight=preflight, stop_reason="API_backend_unavailable_or_drifted")
                    _save(root, state)
                    return state
                stage = next((s for s in plan["stages"] if state["stages"][s["id"]]["status"] != "completed"), None)
                if stage is None:
                    state["status"] = "complete_draft"
                    _save(root, state)
                    return state
                row = state["stages"][stage["id"]]
                request_id = canonical_json_sha256({"job": job, "plan": state["plan_sha256"], "stage": stage["id"]})
                context = self._context(root, state, stage, request_id)
                if stage["kind"] == "artifact":
                    self._record(root, state, stage, {"kind": "owner_artifact", "inputs": context["inputs"],
                        "native_execution_status": "not_attested_by_server_adapter"})
                    continue
                backend = self.backends.get(stage["backend"])
                if (backend is None or backend.revision != stage["backend_revision"]
                        or backend.descriptor.get("capability") not in STAGE_CAPABILITIES[stage["id"]]):
                    state.update(status="blocked", stop_reason="API_backend_unavailable_or_drifted")
                    _save(root, state)
                    return state
                # A dispatch journal survives crashes before provider/local receipt completion.
                uncertain = row["status"] in {"dispatching", "reconciliation_required"}
                if uncertain and reconcile is None:
                    state.update(status="reconciliation_required", stop_reason="unknown_call_outcome")
                    _save(root, state)
                    return state
                try:
                    pack = read_json(_relative(root, row["pack"])) if uncertain else backend.prepare(context)
                except Exception as exc:
                    state.update(status="blocked", stop_reason="request_preparation:" + type(exc).__name__)
                    _save(root, state)
                    return state
                request_hash = canonical_json_sha256(pack["request"])
                if uncertain and row.get("request_sha256") != request_hash:
                    raise ValueError("reconciliation request drift")
                if not uncertain:
                    pack_path = root / "requests" / (stage["id"] + ".json")
                    write_json(pack_path, pack)
                    row.update(status="dispatching", request_id=request_id, request_sha256=request_hash,
                               pack=pack_path.relative_to(root).as_posix(), pack_sha256=sha256_file(pack_path))
                    state["status"] = "running"
                    _save(root, state)
            # No lock held during provider work: cancellation can be persisted.
            # A recovered process only queries a durable receipt; never repeats the call.
            raw = None
            try:
                if lease_guard() is not True:
                    raise RuntimeError("lease_invalid_before_call")
                raw = reconcile(request_id) if uncertain else backend.guarded_call(pack, context)
                if raw is None:
                    raise RuntimeError("unknown_call_outcome")
                backend.validate_result(raw, pack)
                failed = not raw["ok"] or raw["status"] != "completed"
                output = None if failed else backend.parse(raw, pack, stage["output_schema"])
            except Exception as exc:
                with _job_lock(root, operation="server_call_unknown"):
                    current = self._check(root)
                    diagnostic = root / "diagnostics" / (stage["id"] + "-" + uuid4().hex + ".json")
                    evidence = {"error_class": type(exc).__name__, "request_id": request_id, "validated": False}
                    if isinstance(raw, dict):
                        from .model_provider_gateway_adapter import _shared
                        try:
                            _shared("audit").assert_secretless(raw)
                            evidence["unvalidated_gateway_result"] = raw
                        except ValueError:
                            evidence["unvalidated_result_omitted"] = "secretless_audit_failed"
                    write_json(diagnostic, evidence)
                    current["stages"][stage["id"]].update(
                        diagnostic=diagnostic.relative_to(root).as_posix(), diagnostic_sha256=sha256_file(diagnostic))
                    if current["status"] != "cancelled":
                        current.update(status="blocked" if isinstance(exc, ServerAdmissionRejected) else "reconciliation_required",
                                       stop_reason=type(exc).__name__)
                        current["stages"][stage["id"]]["status"] = "reconciliation_required"
                    _save(root, current)
                return self.status(owner, job)
            with _job_lock(root, operation="server_call_receipt"):
                current = self._check(root)
                cancelled = current["status"] == "cancelled" or lease_guard() is not True
                self._record(root, current, stage, {"kind": "guarded_gateway", "request_id": request_id,
                    "request_sha256": request_hash, "gateway_result": raw, "output": output}, cancelled=cancelled)
                if failed and not cancelled:
                    current.update(status="failed", stop_reason="gateway_" + raw["status"])
                    current["stages"][stage["id"]]["status"] = "failed"
                    _save(root, current)
                    return current
                if cancelled:
                    return self.status(owner, job)
        state = self.status(owner, job)
        with _job_lock(root, operation="server_step_boundary"):
            state = self._check(root)
            if state["status"] not in TERMINAL:
                state["status"] = "complete_draft" if all(row["status"] == "completed" for row in state["stages"].values()) else "paused"
                state["stop_reason"] = "step_budget_reached"
                _save(root, state)
        return state

    def _record(self, root: Path, state: dict[str, Any], stage: dict[str, Any], result: dict[str, Any], *, cancelled: bool = False) -> None:
        path = root / "receipts" / (stage["id"] + ".json")
        if path.exists():
            raise ValueError("immutable stage receipt already exists")
        write_json(path, {"schema": "video_knowledge_pipeline.server_stage.v1", "stage": stage["id"],
                          "plan_sha256": state["plan_sha256"], "release_sha256": state["release_sha256"], **result})
        state["stages"][stage["id"]].update(status="completed", receipt=path.relative_to(root).as_posix(), receipt_sha256=sha256_file(path))
        state["status"] = "cancelled" if state["cancel_requested"] else "paused" if cancelled else "running"
        if cancelled:
            state["stop_reason"] = "cancelled_or_lease_lost_after_call; provider_cost_may_exist"
        _save(root, state)

    def result(self, owner: str, job: str) -> dict[str, Any]:
        root = self._root(owner, job)
        state = self._check(root)
        return {"job": state, "receipts": {key: read_json(_relative(root, row["receipt"]))
                for key, row in state["stages"].items() if row.get("receipt")},
                "formal_completed": False, "publication_allowed": False,
                "delivery_mapping_required": True}
