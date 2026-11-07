"""Synthetic source/input/gateway fixtures: never import local models or Codex."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from model_provider_gateway.consent import build_consent_plan
from model_provider_gateway.execution import build_execution_request
from model_provider_gateway.profile import build_profile
from model_provider_gateway.route import build_route_plan

from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.server_execution import PLAN_SCHEMA, ServerExecution
from video_knowledge_pipeline.server_gateway import ApiStageBackend
from video_knowledge_pipeline.server_release import create_server_release, verify_server_release
from video_knowledge_pipeline.storage import read_json, write_json

OUTPUT = {"type": "object", "properties": {"text": {"type": "string", "minLength": 1}},
          "required": ["text"], "additionalProperties": False}


def gateway_result(pack, *, text='{"text":"synthetic summary"}', status="completed"):
    q = pack["request"]
    return {"schema": "model_provider_gateway.execution_result.v1", "ok": status == "completed", "status": status,
            **{k: q[k] for k in ("request_id", "provider_id", "model", "capability", "consent_hash", "input_manifest_hash")},
            "route_hash": q["route_revision"], "usage": {"status": "unknown", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            "cost": {"status": "unknown", "estimated_usd": None, "reserved_ceiling_usd": q["max_estimated_cost_usd"]},
            "latency_ms": 1, "error_class": None, "error_message": None, "retryable": False,
            "artifact_refs": [], "output": {"text": text} if status == "completed" else None,
            "redacted_audit": {"consumer_id": "vkp", "task": q["task"], "profile_id": q["profile_id"],
                "transport": "synthetic_fixture", "input_paths_logged": False, "input_content_logged": False,
                "secret_logged": False, "automatic_fallback_used": False}}


@pytest.fixture
def case(tmp_path):
    source = tmp_path / "source"
    for name in ("src", "scripts", ".git"):
        (source / name).mkdir(parents=True)
    (source / ".git/HEAD").write_text("a" * 40, encoding="ascii")
    (source / "src/synthetic.py").write_text("value = 1\n", encoding="utf-8")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "transcript.txt").write_text("synthetic transcript; no real media", encoding="utf-8")
    profile = build_profile("gemini-3-6-flash", "env:SYNTHETIC_UNUSED_KEY", capabilities=["text"],
                            max_calls=4, max_cost_usd=1, last_verified="2026-11-01T00:00:00Z")
    route = build_route_plan("synthetic-text", "text", [profile], max_calls=4, max_cost_usd=1)
    calls, saved = [], {}

    def prepare(context):
        consent = build_consent_plan(route, [Path(context["inputs"][0]["absolute_path"])], capability="text",
            purpose="Synthetic server API contract test", data_type="synthetic_text", max_calls=4, max_cost_usd=1)
        request = build_execution_request(profile, route, consent, consumer_id="vkp", task="text",
            max_estimated_cost_usd=0.1, request_id=context["request_id"], owner_gate_receipt_hash="a" * 64)
        return {"request": request, "profile": profile, "route": route, "consent": consent}

    def guarded(pack, context=None):
        calls.append(pack["request"]["request_id"])
        result = gateway_result(pack)
        saved[calls[-1]] = result
        return result

    backend = ApiStageBackend({"mode": "api", "location": "remote", "fallback": False,
        "capability": "text", "provider_id": profile["provider_id"], "model": profile["model"],
        "profile_revision": profile["route_revision"], "route_revision": route["route_revision"]}, prepare, guarded)
    # Warm the same metadata-only gateway modules before taking the runtime seal.
    backend.prepare({"request_id": "warmup-request", "inputs": [{"absolute_path": str(inputs / "transcript.txt")}]})
    release = create_server_release(source, tmp_path / "release.json")
    service = ServerExecution(tmp_path / "jobs", source_root=source, release=release, backends={"text": backend})
    plan = {"schema": PLAN_SCHEMA, "inputs": [{"id": "transcript", "path": "transcript.txt", "sha256": sha256_file(inputs / "transcript.txt")}],
            "stages": [{"id": "primary_asr", "kind": "artifact", "input_ids": ["transcript"]},
                       {"id": "semantic", "kind": "api", "input_ids": ["primary_asr"], "backend": "text",
                        "backend_revision": backend.revision, "output_schema": deepcopy(OUTPUT)},
                       {"id": "summary", "kind": "api", "input_ids": ["semantic"], "backend": "text",
                        "backend_revision": backend.revision, "output_schema": deepcopy(OUTPUT)}]}
    return {"service": service, "plan": plan, "inputs": inputs, "source": source, "release": release,
            "backend": backend, "calls": calls, "saved": saved, "tmp": tmp_path}


def start(case, owner="account-a", operation="operation-1"):
    return case["service"].start(owner, operation, input_root=case["inputs"], plan=case["plan"])


def test_complete_resume_and_idempotent_start_reuse_receipts(case):
    service = case["service"]
    first = start(case)
    assert first == start(case)
    paused = service.resume("account-a", first["job_id"], lease_guard=lambda: True, max_steps=1)
    assert paused["status"] == "paused" and case["calls"] == []
    # New service instance represents a process restart; completed artifacts persist.
    restored = ServerExecution(service.workspace, source_root=service.source_root, release=service.release, backends=service.backends)
    done = restored.resume("account-a", first["job_id"], lease_guard=lambda: True)
    assert done["status"] == "complete_draft" and len(set(case["calls"])) == 2
    assert restored.resume("account-a", first["job_id"], lease_guard=lambda: True) == done
    result = restored.result("account-a", first["job_id"])
    assert len(result["receipts"]) == 3 and result["formal_completed"] is False
    assert result["receipts"]["summary"]["gateway_result"]["cost"]["estimated_usd"] is None
    assert done["execution_safety_v1"] is False


@pytest.mark.parametrize("field", ["input", "plan", "source", "dependency", "receipt", "request_pack"])
def test_drift_blocks_without_repeating_calls(case, monkeypatch, field):
    job = start(case)["job_id"]
    service = case["service"]
    service.resume("account-a", job, lease_guard=lambda: True)
    root = service.workspace / "account-a" / job
    if field == "input":
        (case["inputs"] / "transcript.txt").write_text("changed", encoding="utf-8")
    elif field == "plan":
        p = read_json(root / "plan.json")
        p["stages"].pop()
        write_json(root / "plan.json", p)
    elif field == "source":
        (case["source"] / "src/new.py").write_text("drift = True\n", encoding="utf-8")
    elif field == "dependency":
        monkeypatch.setattr("video_knowledge_pipeline.server_release.dependency_snapshot", lambda: {"sha256": "0" * 64})
    elif field == "receipt":
        (root / "receipts/summary.json").write_text("{}", encoding="utf-8")
    else:
        (root / "requests/summary.json").write_text("{}", encoding="utf-8")
    before = list(case["calls"])
    with pytest.raises(ValueError, match="drift"):
        service.resume("account-a", job, lease_guard=lambda: True)
    assert before == case["calls"]


def test_unknown_response_is_reconciled_without_new_call(case):
    service, backend = case["service"], case["backend"]
    def lost(pack, context):
        backend.guarded_call(pack, context)
        raise TimeoutError("response lost after supplier completion")
    service.backends["text"] = ApiStageBackend(backend.descriptor, backend.prepare_request, lost)
    job = start(case)["job_id"]
    state = service.resume("account-a", job, lease_guard=lambda: True)
    assert state["status"] == "reconciliation_required" and len(case["calls"]) == 1
    assert service.resume("account-a", job, lease_guard=lambda: True)["status"] == "reconciliation_required"
    assert len(case["calls"]) == 1
    service.backends["text"] = backend
    state = service.resume("account-a", job, lease_guard=lambda: True, reconcile=case["saved"].get)
    assert state["status"] == "complete_draft" and len(case["calls"]) == 2


@pytest.mark.parametrize("when", ["before", "during", "lease_before", "lease_after"])
def test_cancellation_and_lease_loss_stop_later_calls_preserve_known_cost(case, when):
    service, backend = case["service"], case["backend"]
    job = start(case)["job_id"]
    valid = [when != "lease_before"]
    if when == "before":
        service.cancel("account-a", job)
    if when in {"during", "lease_after"}:
        def guarded(pack, context):
            raw = backend.guarded_call(pack, context)
            if when == "during":
                service.cancel("account-a", job)
            else:
                valid[0] = False
            return raw
        service.backends["text"] = ApiStageBackend(backend.descriptor, backend.prepare_request, guarded)
    result = service.resume("account-a", job, lease_guard=lambda: valid[0])
    assert len(case["calls"]) == (1 if when in {"during", "lease_after"} else 0)
    assert result["status"] in {"cancelled", "paused"}
    if when == "during":
        assert result["provider_cancellation"] == "not_confirmed"
        assert service.result("account-a", job)["receipts"]["semantic"]["gateway_result"]["cost"]["status"] == "unknown"


@pytest.mark.parametrize("change", ["missing", "local", "codex", "revision", "late_asr"])
def test_entire_graph_preflight_blocks_unsupported_routes_before_calls(case, change):
    service = case["service"]
    if change == "missing":
        service.backends = {}
    elif change in {"local", "codex"}:
        case["backend"].descriptor["mode"] = change
    elif change == "revision":
        case["plan"]["stages"][1]["backend_revision"] = "0" * 64
    else:
        # Even a late unavailable ASR stage stops the earlier paid text stages.
        stage = deepcopy(case["plan"]["stages"][1])
        stage.update(id="secondary_asr", input_ids=["summary"])
        case["plan"]["stages"].append(stage)
    state = start(case)
    assert state["status"] == "blocked" and not state["preflight"]["ready"]
    assert service.resume("account-a", state["job_id"], lease_guard=lambda: True)["status"] == "blocked"
    assert not case["calls"]


@pytest.mark.parametrize("mutation", ["request_id", "model", "route_hash", "input_manifest_hash", "invalid_json", "invalid_output"])
def test_unbound_or_invalid_gateway_response_never_completes(case, mutation):
    backend = case["backend"]
    def bad(pack, context):
        raw = backend.guarded_call(pack, context)
        if mutation == "invalid_json":
            raw["output"]["text"] = "not JSON"
        elif mutation == "invalid_output":
            raw["output"]["text"] = "{}"
        else:
            raw[mutation] = "wrong"
        return raw
    case["service"].backends["text"] = ApiStageBackend(backend.descriptor, backend.prepare_request, bad)
    job = start(case)["job_id"]
    result = case["service"].resume("account-a", job, lease_guard=lambda: True)
    assert result["status"] == "reconciliation_required" and len(case["calls"]) == 1
    assert result["formal_completed"] is False
    assert result["stages"]["semantic"]["diagnostic_sha256"]


def test_known_gateway_failure_is_not_unknown_or_automatically_retried(case):
    backend = case["backend"]
    def failed(pack, context):
        case["calls"].append(pack["request"]["request_id"])
        return gateway_result(pack, status="failed")
    case["service"].backends["text"] = ApiStageBackend(backend.descriptor, backend.prepare_request, failed)
    job = start(case)["job_id"]
    assert case["service"].resume("account-a", job, lease_guard=lambda: True)["status"] == "failed"
    assert case["service"].resume("account-a", job, lease_guard=lambda: True)["status"] == "failed"
    assert len(case["calls"]) == 1


def test_owner_operation_isolation_and_conflict(case):
    first = start(case)
    second = start(case, owner="account-b")
    assert first["job_id"] != second["job_id"]
    with pytest.raises(FileNotFoundError):
        case["service"].status("account-b", first["job_id"])
    case["plan"]["stages"].pop()
    with pytest.raises(ValueError, match="conflicts"):
        start(case)


@pytest.mark.parametrize("path", ["../transcript.txt", "/tmp/file", "D:/file", "a\\b", "a/../b"])
def test_input_path_escape_rejected_before_job_creation(case, path):
    case["plan"]["inputs"][0]["path"] = path
    with pytest.raises(ValueError, match="relative"):
        start(case)
    assert not case["service"].workspace.exists()


@pytest.mark.parametrize("keyword", ["$ref", "$dynamicRef", "$recursiveRef"])
def test_external_schema_reference_rejected(case, keyword):
    case["plan"]["stages"][1]["output_schema"]["properties"]["text"] = {keyword: "https://invalid.example/schema"}
    with pytest.raises(ValueError, match="external schema"):
        start(case)


def test_seal_content_and_git_head_are_verified_and_not_overwritten(case):
    assert verify_server_release(case["source"], case["release"])["verified"] is True
    with pytest.raises(ValueError, match="preserve"):
        create_server_release(case["source"], case["tmp"] / "release.json")
    changed = deepcopy(case["release"])
    changed["git_head"] = "b" * 40
    with pytest.raises(ValueError, match="document drift"):
        verify_server_release(case["source"], changed)
    (case["source"] / ".git/HEAD").write_text("b" * 40, encoding="ascii")
    with pytest.raises(ValueError, match="HEAD drift"):
        verify_server_release(case["source"], case["release"])


def test_cancellation_remains_possible_after_source_drift(case):
    job = start(case)["job_id"]
    (case["source"] / "src/new.py").write_text("drift = True\n", encoding="utf-8")
    assert case["service"].cancel("account-a", job)["status"] == "cancelled"
    assert not case["calls"]


def test_dead_old_job_lock_recovery_reuses_native_guard(case, monkeypatch):
    import os
    import time
    service = case["service"]
    job = start(case)["job_id"]
    root = service.workspace / "account-a" / job
    path = root / ".bundle-write.lock"
    write_json(path, {"pid": 999999999, "lock_id": "synthetic-crashed-owner"})
    old = time.time() - 600
    os.utime(path, (old, old))
    monkeypatch.setattr("video_knowledge_pipeline.storage._process_is_alive", lambda pid: False)
    assert service.resume("account-a", job, lease_guard=lambda: True)["status"] == "complete_draft"


def test_machine_contract_and_plan_schema_are_executable_offline(case, capsys):
    from video_knowledge_pipeline.server_contract import main, server_api_contract
    from jsonschema import Draft202012Validator
    contract = server_api_contract()
    Draft202012Validator(contract["plan_json_schema"]).validate(case["plan"])
    assert set(contract["operations"]) == {"start", "status", "cancel", "resume", "result"}
    assert main(["contract"]) == 0
    assert json.loads(capsys.readouterr().out)["execution_safety_v1"] is False


def test_secret_bearing_pack_is_rejected_before_dispatch(case):
    backend = case["backend"]
    def unsafe(context):
        pack = backend.prepare_request(context)
        pack["profile"]["api_key"] = "synthetic-plaintext-key"
        return pack
    case["service"].backends["text"] = ApiStageBackend(backend.descriptor, unsafe, backend.guarded_call)
    job = start(case)["job_id"]
    state = case["service"].resume("account-a", job, lease_guard=lambda: True)
    assert state["status"] == "blocked" and state["stop_reason"].startswith("request_preparation:")
    assert not case["calls"]


def test_status_and_result_are_readonly(case):
    service = case["service"]
    job = start(case)["job_id"]
    service.resume("account-a", job, lease_guard=lambda: True)
    root = service.workspace / "account-a" / job
    def snapshot():
        return {p.relative_to(root).as_posix(): (sha256_file(p), p.stat().st_mtime_ns)
                for p in root.rglob("*") if p.is_file()}
    before = snapshot()
    service.status("account-a", job)
    service.result("account-a", job)
    assert snapshot() == before


def test_vision_uses_the_same_guarded_contract_without_local_executor(case):
    import base64
    path = case["inputs"] / "frame.png"
    path.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="))
    profile = build_profile("gemini-3-6-flash", "env:SYNTHETIC_UNUSED_KEY", capabilities=["vision"],
        max_calls=1, max_cost_usd=1, last_verified="2026-11-01T00:00:00Z")
    route = build_route_plan("synthetic-vision", "vision", [profile], max_calls=1, max_cost_usd=1)
    def prepare(context):
        consent = build_consent_plan(route, [path], capability="vision", purpose="Synthetic frame contract",
            data_type="synthetic_image", max_calls=1, max_cost_usd=1)
        request = build_execution_request(profile, route, consent, consumer_id="vkp", task="visual",
            max_estimated_cost_usd=0.1, request_id=context["request_id"], owner_gate_receipt_hash="a" * 64)
        return {"request": request, "profile": profile, "route": route, "consent": consent}
    calls = []
    def guarded(pack, context):
        calls.append(context["stage"])
        return gateway_result(pack)
    backend = ApiStageBackend({"mode": "api", "location": "remote", "fallback": False, "capability": "vision",
        "provider_id": profile["provider_id"], "model": profile["model"], "profile_revision": profile["route_revision"],
        "route_revision": route["route_revision"]}, prepare, guarded)
    case["service"].backends = {"vision": backend}
    case["plan"] = {"schema": PLAN_SCHEMA, "inputs": [{"id": "frame", "path": "frame.png", "sha256": sha256_file(path)}],
        "stages": [{"id": "vision", "kind": "api", "input_ids": ["frame"], "backend": "vision",
                    "backend_revision": backend.revision, "output_schema": deepcopy(OUTPUT)}]}
    job = start(case)["job_id"]
    assert case["service"].resume("account-a", job, lease_guard=lambda: True)["status"] == "complete_draft"
    assert calls == ["vision"]
