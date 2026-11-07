from __future__ import annotations

import json
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_knowledge_pipeline import codex_vision_executor as worker
from video_knowledge_pipeline.codex_vision_session import import_codex_vision_results, prepare_codex_vision_session
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.storage import read_json, write_json
from test_codex_vision_session import _bundle


@pytest.fixture
def runtime(monkeypatch):
    calls = []
    monkeypatch.setattr(worker, "_resolve_binary", lambda _: Path(sys.executable))
    monkeypatch.setattr(worker, "_probe_capability", lambda _: {"source": "synthetic_not_access_proof"})

    def execute(binary, config, pack, directory):
        calls.append(pack["selected_indexes"])
        template = read_json(Path(pack["result_template_path"]))
        items = template["items"]
        for item in items:
            item["observed_frame_ids"] = item["expected_frame_ids"]
            item["evidence_level"] = "confirmed"
            item["visual_understanding"].update(objects=["synthetic slide"], confidence=0.8,
                                                interface_state="a visible title")
        write_json(directory / "response.json", {"items": items})
        write_json(directory / "output.schema.json", worker._output_schema(template))
        receipt = {"schema": worker.RECEIPT_SCHEMA, "status": "succeeded", "exit_code": 0,
                   "turn_completed": True, "tool_call_count": 0,
                   "model_actual": config["model"], "model_requested": config["model"],
                   "reasoning_effort": config["reasoning_effort"],
                   "model_actual_basis": worker.MODEL_BASIS, "server_model_attested": False,
                   "task_pack_sha256": pack["task_pack_sha256"],
                   "response_sha256": sha256_file(directory / "response.json")}
        write_json(directory / "execution.json", receipt)
        return receipt

    monkeypatch.setattr(worker, "_execute_batch", execute)
    return calls, execute


def test_preview_is_offline_and_does_not_write(tmp_path, monkeypatch):
    root = _bundle(tmp_path)
    before = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    monkeypatch.setattr(worker, "_resolve_binary", lambda _: pytest.fail("preview started runtime"))
    preview = worker.run_codex_vision(root)
    assert preview["external_io"] is False and preview["execute"] is False
    assert preview["profile"]["model"] == "gpt-6-luna"
    assert preview["profile"]["reasoning_effort"] == "low"
    assert len(preview["tasks"]) == 3
    assert before == {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_serial_batches_resume_and_completed_rerun_never_call_model(tmp_path, runtime):
    calls, _ = runtime
    root = _bundle(tmp_path, semantic_count=7)
    state = worker.run_codex_vision(root, execute=True, max_batches=1)
    assert state["status"] == "paused_batch_limit"
    assert state["counts"] == {"pending": 4, "imported": 3}
    assert calls == [[1, 2, 3]]
    assert len(state["batches"]) == 1  # no stale prepared next batch at pause
    assert "--state-json" in state["next_action"]
    state = worker.run_codex_vision(root, state_json=state["state_json"], execute=True)
    assert state["status"] == "completed" and state["counts"]["imported"] == 7
    assert calls == [[1, 2, 3], [4, 5, 6], [7]]
    again = worker.run_codex_vision(root, state_json=state["state_json"], execute=True)
    assert again["status"] == "completed" and len(calls) == 3
    first = read_json(root / "timeline.json")[0]["visual_understanding"]
    assert first["executor"] == "codex_cli" and first["execution_mode"] == "automated_multimodal"
    assert first["recognition_provenance"]["executor"] == "codex_cli"
    assert first["execution_receipt"]["sha256"]
    audit = [json.loads(line) for line in (root / "vision-analysis-runs.jsonl").read_text().splitlines()]
    assert all(row["executor"] == "codex_cli" and row["execution_mode"] == "automated_multimodal" for row in audit)


def test_legacy_uncalled_pack_is_archived_and_reprepared_without_recalling_imported(tmp_path, runtime):
    calls, _ = runtime
    root = _bundle(tmp_path, semantic_count=4)
    state = worker.run_codex_vision(root, execute=True, max_batches=1)
    pack = prepare_codex_vision_session(root, indexes=[4], executor="codex_cli",
        model_requested=state["profile"]["model"], quality=state["profile"]["reasoning_effort"])
    old_path = Path(pack["task_pack_path"])
    old_bytes = old_path.read_bytes()
    state["batches"].append({"indexes": [4], "pack_path": str(old_path), "status": "prepared", "attempts": []})
    write_json(Path(state["state_json"]), state)
    manifest = read_json(root / "manifest.json")
    manifest["asr_pointer_only"] = "new-comparison.json"
    write_json(root / "manifest.json", manifest)
    resumed = worker.run_codex_vision(root, state_json=state["state_json"], execute=True)
    assert resumed["status"] == "completed" and calls == [[1, 2, 3], [4]]
    batch = resumed["batches"][-1]
    assert batch["pack_path"] != str(old_path) and old_path.read_bytes() == old_bytes
    assert batch["preparation_history"][0]["status"] == "superseded_before_first_call"


def test_uncalled_pack_with_different_model_is_not_rebound(tmp_path, runtime):
    calls, _ = runtime
    root = _bundle(tmp_path, semantic_count=4)
    state = worker.run_codex_vision(root, execute=True, max_batches=1)
    pack = prepare_codex_vision_session(root, indexes=[4], executor="codex_cli",
        model_requested="different-model", quality=state["profile"]["reasoning_effort"])
    state["batches"].append({"indexes": [4], "pack_path": pack["task_pack_path"], "status": "prepared", "attempts": []})
    write_json(Path(state["state_json"]), state)
    manifest = read_json(root / "manifest.json")
    manifest["unrelated_update"] = True
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="preparation identity invalid"):
        worker.run_codex_vision(root, state_json=state["state_json"], execute=True)
    assert calls == [[1, 2, 3]]


def test_called_stale_pack_cannot_be_rebound_for_retry(tmp_path, runtime, monkeypatch):
    root = _bundle(tmp_path)
    monkeypatch.setattr(worker, "_execute_batch", lambda *args: {"status": "failed", "exit_code": 1})
    state = worker.run_codex_vision(root, execute=True)
    manifest = read_json(root / "manifest.json")
    manifest["unrelated_update"] = True
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="task_pack_drift"):
        worker.run_codex_vision(root, state_json=state["state_json"], execute=True, retry_failed=True)
    assert len(read_json(Path(state["state_json"]))["batches"][0]["attempts"]) == 1


def test_profile_is_replaceable_but_pinned_within_resume(tmp_path, runtime):
    root = _bundle(tmp_path, semantic_count=4)
    state = worker.run_codex_vision(root, execute=True, max_batches=1)
    with pytest.raises(ValueError, match="profile_drift"):
        worker.run_codex_vision(root, execute=True, state_json=state["state_json"], profile={"model": "future-model"})
    other = _bundle(tmp_path / "other")
    changed = worker.run_codex_vision(other, execute=True, profile={"model": "future-model", "revision": "2"})
    assert changed["status"] == "completed" and changed["profile"]["model"] == "future-model"


@pytest.mark.parametrize("mutation", ["frame", "text", "time", "output"])
def test_drift_blocks_resume_before_more_calls(tmp_path, runtime, mutation):
    calls, _ = runtime
    root = _bundle(tmp_path, semantic_count=4)
    state = worker.run_codex_vision(root, execute=True, max_batches=1)
    rows = read_json(root / "timeline.json")
    if mutation == "frame":
        (root / rows[3]["assets"][0]["path"]).write_bytes(b"changed")
    else:
        if mutation == "text":
            rows[3]["transcript"] += "new"
        elif mutation == "time":
            rows[3]["end"] += 1
        else:
            rows[0]["visual_understanding"]["objects"] = ["changed"]
        write_json(root / "timeline.json", rows)
    with pytest.raises(ValueError, match="drift"):
        worker.run_codex_vision(root, execute=True, state_json=state["state_json"])
    assert calls == [[1, 2, 3]]


def test_failed_call_requires_explicit_retry_and_does_not_write_timeline(tmp_path, runtime, monkeypatch):
    calls, execute = runtime
    root = _bundle(tmp_path)
    before = (root / "timeline.json").read_bytes()
    monkeypatch.setattr(worker, "_execute_batch", lambda *args: {"status": "failed", "exit_code": 1, "error": "quota"})
    state = worker.run_codex_vision(root, execute=True)
    assert state["status"] == "needs_retry" and (root / "timeline.json").read_bytes() == before
    state = worker.run_codex_vision(root, execute=True, state_json=state["state_json"])
    assert len(state["batches"][0]["attempts"]) == 1 and calls == []
    monkeypatch.setattr(worker, "_execute_batch", execute)
    state = worker.run_codex_vision(root, execute=True, state_json=state["state_json"], retry_failed=True)
    assert state["status"] == "completed" and calls == [[1, 2, 3]]


def test_interrupted_process_with_durable_exit_is_collected_without_recall(tmp_path, runtime, monkeypatch):
    calls, _ = runtime
    root = _bundle(tmp_path)
    collect = worker._collect_result
    monkeypatch.setattr(worker, "_collect_result", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        worker.run_codex_vision(root, execute=True)
    path = next((root / "codex-vision-auto").glob("*/state.json"))
    state = read_json(path)
    state["batches"][0]["status"] = "executing"
    write_json(path, state)
    monkeypatch.setattr(worker, "_collect_result", collect)
    resumed = worker.run_codex_vision(root, execute=True, state_json=path)
    assert resumed["status"] == "completed" and len(calls) == 1


def test_import_crash_reconciles_durable_commit_without_duplicate_call(tmp_path, runtime, monkeypatch):
    calls, _ = runtime
    root = _bundle(tmp_path)
    native_import = worker.import_codex_vision_results

    def imported_then_crashed(*args, **kwargs):
        native_import(*args, **kwargs)
        raise KeyboardInterrupt()

    monkeypatch.setattr(worker, "import_codex_vision_results", imported_then_crashed)
    with pytest.raises(KeyboardInterrupt):
        worker.run_codex_vision(root, execute=True)
    path = next((root / "codex-vision-auto").glob("*/state.json"))
    monkeypatch.setattr(worker, "import_codex_vision_results", native_import)
    resumed = worker.run_codex_vision(root, execute=True, state_json=path)
    assert resumed["status"] == "completed" and len(calls) == 1


@pytest.mark.parametrize("mutation", ["partial", "wrong_frame", "schema", "receipt", "receipt_type"])
def test_invalid_output_cannot_enter_timeline(tmp_path, runtime, monkeypatch, mutation):
    _, execute = runtime
    root = _bundle(tmp_path)
    before = (root / "timeline.json").read_bytes()

    def bad(*args):
        receipt = execute(*args)
        directory = args[-1]
        response = read_json(directory / "response.json")
        if mutation == "partial":
            response["items"].pop()
        elif mutation == "wrong_frame":
            response["items"][0]["observed_frame_ids"] = ["not-attached"]
        elif mutation == "schema":
            response["items"] = "not an array"
        elif mutation == "receipt":
            receipt["model_actual"] = "forged"
        write_json(directory / "response.json", response)
        receipt["response_sha256"] = sha256_file(directory / "response.json")
        write_json(directory / "execution.json", [] if mutation == "receipt_type" else receipt)
        return receipt

    monkeypatch.setattr(worker, "_execute_batch", bad)
    state = worker.run_codex_vision(root, execute=True)
    assert state["status"] == "needs_retry"
    assert (root / "timeline.json").read_bytes() == before


def test_auto_import_requires_execution_receipt(tmp_path):
    root = _bundle(tmp_path, semantic_count=1)
    pack = prepare_codex_vision_session(root, executor="codex_cli", indexes=[1])
    result = read_json(Path(pack["result_template_path"]))
    result["model_actual"] = "gpt-6-luna"
    item = result["items"][0]
    item.update(evidence_level="confirmed", observed_frame_ids=item["expected_frame_ids"])
    item["visual_understanding"]["objects"] = ["slide"]
    path = tmp_path / "fake.json"
    write_json(path, result)
    outcome = import_codex_vision_results(root, input_json=path)
    assert "automated_execution_invalid" in {issue["key"] for issue in outcome["validation_issues"]}


@pytest.mark.parametrize("override", [{"batch_size": True}, {"batch_size": 20}, {"model": ""},
                                      {"reasoning_effort": "bad"}, {"max_attempts": 0}, {"extra": 1}])
def test_invalid_profiles_are_rejected(override):
    with pytest.raises(ValueError):
        worker.load_codex_vision_profile(override)


def test_command_uses_native_cli_schema_images_and_scoped_settings(tmp_path):
    command = worker._command(Path(sys.executable), worker.load_codex_vision_profile(), ["a frame.jpg"], tmp_path)
    assert command[:4] == [sys.executable, "exec", "--model", "gpt-6-luna"]
    assert 'model_reasoning_effort="low"' in command
    assert 'model_provider="openai"' in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--output-schema" in command and "--ephemeral" in command and "a frame.jpg" in command
    assert not any("bypass" in arg or arg == "--ignore-user-config" for arg in command)


def test_new_cli_routes_are_registered():
    from video_knowledge_pipeline.cli import build_parser

    args = build_parser().parse_args(["run-codex-vision", "bundle", "--index", "2", "--execute", "--max-batches", "1"])
    assert args.index == [2] and args.execute and args.max_batches == 1
    assert build_parser().parse_args(["codex-vision-catalog"]).command == "codex-vision-catalog"


@pytest.mark.parametrize("mutation", ["task_not_object", "invalid_status", "pack_escape", "attempt_escape"])
def test_malformed_state_is_rejected_without_losing_pending_counts(tmp_path, runtime, mutation):
    calls, _ = runtime
    root = _bundle(tmp_path, semantic_count=4)
    state = worker.run_codex_vision(root, execute=True, max_batches=1)
    path = Path(state["state_json"])
    if mutation == "task_not_object":
        state["tasks"].append("not a task")
    elif mutation == "invalid_status":
        state["tasks"][3]["status"] = []
    elif mutation == "pack_escape":
        state["batches"][0]["pack_path"] = str(tmp_path / "outside.json")
    else:
        state["batches"][0]["attempts"][0]["directory"] = str(tmp_path)
    write_json(path, state)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="queue"):
        worker.run_codex_vision(root, execute=True, state_json=path)
    assert path.read_bytes() == before and len(calls) == 1


def test_uncertain_call_is_not_automatically_repeated(tmp_path, runtime, monkeypatch):
    root = _bundle(tmp_path)
    monkeypatch.setattr(worker, "_execute_batch", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        worker.run_codex_vision(root, execute=True)
    path = next((root / "codex-vision-auto").glob("*/state.json"))
    resumed = worker.run_codex_vision(root, execute=True, state_json=path)
    assert resumed["status"] == "needs_retry"
    assert len(resumed["batches"][0]["attempts"]) == 1
    assert "unknown" in resumed["error"]


def test_retry_budget_is_not_reset_by_restarting(tmp_path, runtime, monkeypatch):
    root = _bundle(tmp_path)
    monkeypatch.setattr(worker, "_execute_batch", lambda *args: {"status": "failed", "exit_code": 1})
    state = worker.run_codex_vision(root, execute=True)
    for _ in range(3):
        state = worker.run_codex_vision(root, execute=True, state_json=state["state_json"], retry_failed=True)
    assert state["status"] == "needs_retry"
    assert len(state["batches"][0]["attempts"]) == 2


@pytest.mark.parametrize("entry", [None, {"slug": "gpt-6-luna", "input_modalities": ["text"]},
    {"slug": "gpt-6-luna", "input_modalities": ["image"], "supported_reasoning_levels": [{"effort": "high"}]}])
def test_catalog_capability_failure_is_not_access_or_model_substitution(tmp_path, monkeypatch, entry):
    monkeypatch.setattr(worker, "codex_vision_catalog", lambda _: {"source": "test", "models": [entry] if entry else []})
    with pytest.raises(ValueError):
        worker._probe_capability(worker.load_codex_vision_profile())


@pytest.mark.parametrize("mode", ["success", "feature_notice", "client_error", "nonzero", "timeout", "missing_response", "tool", "failed_event", "bad_json", "event_list", "bad_item"])
def test_process_exit_events_and_timeout_are_required_for_success(tmp_path, monkeypatch, mode):
    root = _bundle(tmp_path)
    pack = prepare_codex_vision_session(root, indexes=[1], executor="codex_cli")
    directory = root / "codex-vision-auto" / "process-test"
    directory.mkdir(parents=True)
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="codex-cli synthetic", stderr=""))
    killed = []

    class Process:
        done = mode != "timeout"

        def __init__(self, command, **kwargs):
            assert kwargs["shell"] is False
            assert kwargs["stdin"].read().startswith(b"You are a bounded visual annotation worker.")
            kwargs["stdin"].seek(0)
            events = [{"type": "turn.completed"}]
            if mode == "tool":
                events.append({"type": "item.completed", "item": {"type": "mcp_tool_call"}})
            if mode == "failed_event":
                events.append({"type": "turn.failed"})
            if mode in {"feature_notice", "client_error"}:
                events.append({"type": "item.completed", "item": {"type": "error", "message":
                    "Under-development features enabled: local_thread_store_compression." if mode == "feature_notice" else "model request failed"}})
            if mode == "event_list":
                events.append([])
            if mode == "bad_item":
                events.append({"type": "item.completed", "item": []})
            kwargs["stdout"].write("\n".join(json.dumps(e) for e in events).encode())
            if mode == "bad_json":
                kwargs["stdout"].write(b"\nnot-json")
            if mode != "missing_response":
                write_json(directory / "response.json", {"items": []})

        def poll(self):
            return (1 if mode in {"nonzero", "timeout"} else 0) if self.done else None

        def wait(self, **kwargs):
            return self.poll()

        def kill(self):
            self.done = True
            killed.append(True)

    monkeypatch.setattr(worker.subprocess, "Popen", Process)
    ticks = itertools.count(0, 250)
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(ticks))
    result = worker._execute_batch(Path(sys.executable), worker.load_codex_vision_profile(), pack, directory)
    succeeded = mode in {"success", "feature_notice"}
    assert result["status"] == ("succeeded" if succeeded else "failed")
    if mode == "timeout":
        assert killed and result["error"] == "hard_timeout"
    if not succeeded:
        assert result["model_actual"] == ""


def test_committed_result_changed_before_queue_commit_cannot_be_adopted(tmp_path, runtime, monkeypatch):
    calls, _ = runtime
    root = _bundle(tmp_path)
    native_import = worker.import_codex_vision_results

    def imported_then_crashed(*args, **kwargs):
        native_import(*args, **kwargs)
        raise KeyboardInterrupt()

    monkeypatch.setattr(worker, "import_codex_vision_results", imported_then_crashed)
    with pytest.raises(KeyboardInterrupt):
        worker.run_codex_vision(root, execute=True)
    state_path = next((root / "codex-vision-auto").glob("*/state.json"))
    rows = read_json(root / "timeline.json")
    rows[0]["visual_understanding"]["objects"] = ["changed after native import"]
    write_json(root / "timeline.json", rows)
    before = (root / "timeline.json").read_bytes()
    with pytest.raises(ValueError, match="import_output_drift"):
        worker.run_codex_vision(root, execute=True, state_json=state_path)
    assert len(calls) == 1 and (root / "timeline.json").read_bytes() == before


def test_non_object_execution_receipt_is_a_validation_issue(tmp_path):
    root = _bundle(tmp_path)
    path = root / "codex-vision-auto" / "receipt.json"
    write_json(path, [])
    result = {"execution_receipt": {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}}
    issues = worker.validate_execution_receipt(root, result, {})
    assert issues[0]["key"] == "automated_execution_invalid"
