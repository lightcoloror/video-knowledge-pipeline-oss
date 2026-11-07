"""Offline reproductions of local video-run feedback; never call real providers."""
import builtins
import io
import json
import urllib.error
from pathlib import Path

import pytest

from video_knowledge_pipeline import screen_text_recovery as recovery
from video_knowledge_pipeline import text_llm_gateway as text
from video_knowledge_pipeline.model_task_gateway import model_task_api_call
from video_knowledge_pipeline.vision_batch_continuation import write_semantic_continuation


LOCAL = {"provider": "local_vlm", "base_url": "http://127.0.0.1:1234/v1", "model": "fixture", "api_key": ""}


def _block_pillow(monkeypatch):
    original = builtins.__import__
    def without_pillow(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ModuleNotFoundError("No module named 'PIL'")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_pillow)


@pytest.mark.parametrize("options", [{"execute_crops": True}, {"execute_ocr": True}, {"execute_crops": True, "execute_ocr": True}])
def test_missing_pillow_blocks_before_plan_import_or_any_write(tmp_path, monkeypatch, options):
    for name, body in [("manifest.json", "{}"), ("timeline.json", "[]")]:
        (tmp_path / name).write_text(body, encoding="utf-8")
    _block_pillow(monkeypatch)
    def forbidden(*args, **kwargs):
        pytest.fail("Planner or OCR must not run after dependency failure")
    monkeypatch.setattr(recovery, "run_ocr_backfill", forbidden)
    monkeypatch.setattr(recovery, "_run_captiocr_candidates", forbidden)
    with pytest.raises(RuntimeError, match="screen_text_recovery_blocked_dependency"):
        recovery.run_screen_text_recovery(tmp_path, **options)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json", "timeline.json"]
    assert (tmp_path / "timeline.json").read_text() == "[]"


def test_missing_pillow_preview_reports_real_runtime(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "timeline.json").write_text("[]")
    _block_pillow(monkeypatch)
    result = recovery.run_screen_text_recovery(tmp_path)
    check = result["runtime_preflight"]
    assert check["status"] == "blocked_dependency"
    assert check["python_executable"] in check["install_command"]
    assert "blocked_dependency" in Path(result["report_path"]).read_text(encoding="utf-8")


class Response:
    def __init__(self, content='{"decisions": []}'):
        self.content = content
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self):
        return json.dumps({"choices": [{"message": {"content": self.content}, "finish_reason": "stop"}]}).encode()


@pytest.mark.parametrize("mode", ["default", "text", "json_schema"])
def test_task_gateway_sends_supported_format_and_keeps_json_validation(monkeypatch, mode):
    config = dict(LOCAL)
    expected = {"type": "text"}
    if mode == "text":
        config["provider_options"] = {"response_format": "text"}
    elif mode == "json_schema":
        expected = {"type": "json_schema", "json_schema": {"name": "decision", "strict": True, "schema": {
            "type": "object", "properties": {"decisions": {"type": "array", "items": {"type": "string"}}},
            "required": ["decisions"], "additionalProperties": False}}}
        config["provider_options"] = {"response_format": expected}
    sent = []
    def fake_open(request, **kwargs):
        sent.append(json.loads(request.data))
        assert request.get_header("Authorization") is None
        return Response()
    monkeypatch.setattr(text.urllib.request, "urlopen", fake_open)
    result = model_task_api_call("transcript_semantic_correction", provider_config=config,
        messages=[{"role": "user", "content": 'Return JSON with a decisions array.'}],
        execute=True, response_format={"type": "json_object"})
    assert result["ok"] and len(sent) == 1
    assert sent[0]["response_format"] == expected
    assert text.extract_json_document(result["content"], require_object=True) == {"decisions": []}
    with pytest.raises(ValueError):
        text.extract_json_document("not a decision JSON", require_object=True)


@pytest.mark.parametrize("value", ["json_schema", "invalid", {"type": "json_schema", "json_schema": {"name": "missing"}}])
def test_invalid_format_configuration_rejected_without_request(value, monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("Bad configuration must fail before HTTP")
    monkeypatch.setattr(text.urllib.request, "urlopen", forbidden)
    with pytest.raises(ValueError):
        text.call_openai_compatible_text(provider_config={**LOCAL, "provider_options": {"response_format": value}}, messages=[])


def test_format_http_400_survives_task_gateway_without_retry_or_key_error(monkeypatch):
    requests = []
    def reject(request, **kwargs):
        requests.append(request)
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {},
            io.BytesIO(b'{"error":{"message":"response_format.type must be json_schema or text"}}'))
    monkeypatch.setattr(text.urllib.request, "urlopen", reject)
    result = model_task_api_call("transcript_semantic_correction", provider_config=LOCAL,
        messages=[{"role": "user", "content": "synthetic"}], execute=True, response_format={"type": "json_object"})
    assert not result["ok"] and len(requests) == 1
    assert result["http_status"] == 400 and result["error_class"] == "response_format_unsupported"
    assert result["error"].startswith("http_400:")
    assert "missing_api_key" not in result["error"]


def test_remote_default_and_explicit_local_override_are_preserved():
    remote = text.resolve_text_provider_config({"provider": "openai_compatible", "base_url": "https://example.invalid/v1"})
    assert text._effective_response_format(remote, {"type": "json_object"}) == {"type": "json_object"}
    explicit = text.resolve_text_provider_config({**LOCAL, "provider_options": {"response_format": "json_object"}})
    assert text._effective_response_format(explicit, {"type": "json_object"}) == {"type": "json_object"}


def test_explicit_litellm_text_path_uses_same_endpoint_format(monkeypatch):
    from video_knowledge_pipeline import online_model_gateway as online
    captured = []
    def fake_litellm(**kwargs):
        captured.append(kwargs)
        return {"ok": True, "content": '{"decisions": []}'}
    monkeypatch.setattr(online, "call_litellm_chat", fake_litellm)
    result = model_task_api_call("transcript_semantic_correction",
        provider_config={**LOCAL, "adapter_backend": "litellm"}, execute=True,
        messages=[{"role": "user", "content": "Return JSON"}], response_format={"type": "json_object"})
    assert result["ok"] and captured[0]["response_format"] == {"type": "text"}


@pytest.mark.parametrize("completed,next_end", [(19, 38), (38, 57), (57, 76), (76, 80)])
def test_80_frame_batches_persist_exact_next_indexes(tmp_path, completed, next_end):
    result = write_semantic_continuation(tmp_path, pending_indexes=list(range(completed + 1, 81)),
        batch_size=19, summary={"status": "ok"}, execution_actor="agent")
    stored = json.loads(Path(result["task_path"]).read_text(encoding="utf-8"))
    assert stored["next_indexes"] == list(range(completed + 1, next_end + 1))
    assert stored["pending_count"] == 80 - completed
    assert stored["status"] == "needs_execution" and stored["automatic_execution"] is False
    assert "--provider-config $visionConfig" in stored["execute_command_template"]
    assert "--execution-actor agent --export-consent $visionConsent" in stored["execute_command_template"]
    assert f"--confirm-vision-calls {next_end - completed}" in stored["execute_command_template"]


def test_complete_replaces_stale_task_and_failed_batch_does_not_claim_progress(tmp_path):
    blocked = write_semantic_continuation(tmp_path, pending_indexes=[19, 20], batch_size=19,
        summary={"status": "partial_failure", "run_failed_indexes": [19]})
    assert blocked["status"] == "blocked_current_batch" and blocked["failed_indexes"] == [19]
    done = write_semantic_continuation(tmp_path, pending_indexes=[], batch_size=19, summary={"status": "ok"})
    stored = json.loads(Path(done["task_path"]).read_text())
    assert stored["status"] == "complete" and not stored["execute_command_template"]


def test_bundle_queue_exposes_next_confirmation_instead_of_reusing_previous(monkeypatch, tmp_path):
    from video_knowledge_pipeline import bundle_next as bn
    task = {"pending_count": 61, "next_indexes": list(range(20, 39))}
    calls = []
    def step(*args, **kwargs):
        calls.append(kwargs)
        return {"advanced": True, "action_result": {"summary": {"continuation": task}}, "before": {}, "after": {},
            "continuation_path": str(tmp_path / "semantic-frame-next-batch.json"), "pending_count": 61}
    monkeypatch.setattr(bn, "bundle_advance", step)
    monkeypatch.setattr(bn, "bundle_next_action", lambda *a, **k: {})
    monkeypatch.setattr(bn, "bundle_advance_log", lambda *a, **k: {})
    result = bn.bundle_advance_queue(tmp_path, execute=True, provider_config=LOCAL, max_steps=5)
    assert len(calls) == 1 and result["stop_reason"] == "next_batch_confirmation_required"
    assert result["continuation"] == task
    from video_knowledge_pipeline.cli_output import render_cli_result
    compact = json.loads(render_cli_result(result, inline_limit=1))
    assert compact["counts"]["pending_count"] == 61
    assert compact["paths"]["continuation_path"].endswith("semantic-frame-next-batch.json")


def test_real_analyzer_preview_persists_remaining_without_model_call(tmp_path, monkeypatch):
    from video_knowledge_pipeline import multimodal_frame_analyzer as m
    (tmp_path / "manifest.json").write_text("{}")
    timeline = [{"start": i, "end": i+1, "visual_route": "semantic_frame", "frame_paths": []} for i in range(80)]
    for item in timeline[:19]:
        item["visual_understanding"] = {"objects": ["synthetic screen"], "confidence": 0.9}
    (tmp_path / "timeline.json").write_text(json.dumps(timeline))
    def forbidden(*a, **k):
        pytest.fail("Preview must not execute a model")
    monkeypatch.setattr(m, "call_vision_model_with_retries", forbidden)
    result = m.run_multimodal_frame_analysis(tmp_path, provider_config=LOCAL, limit=19)
    plan = result["summary"]["continuation"]
    assert plan["pending_count"] == 61 and plan["next_indexes"] == list(range(20, 39))
    assert result["run_registry"]["status"] == "needs_execution"
    assert "Remaining semantic frames: 61" in Path(result["report_path"]).read_text(encoding="utf-8")


def test_successful_batch_stays_pending_until_final_batch(tmp_path, monkeypatch):
    from video_knowledge_pipeline import multimodal_frame_analyzer as m
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "frame.jpg").write_bytes(b"offline synthetic frame")
    timeline = [{"start": i, "end": i+1, "visual_route": "semantic_frame", "frame_paths": ["frame.jpg"]} for i in range(80)]
    (tmp_path / "timeline.json").write_text(json.dumps(timeline))
    calls = []
    monkeypatch.setattr(m, "_execution_control", lambda *a, **k: ({}, {"status": "confirmed", "confirmed": True}))
    monkeypatch.setattr(m, "prepare_image_probe", lambda paths, **kw: {"image_paths": paths})
    monkeypatch.setattr(m, "_refresh_post_vision_outputs", lambda *a, **kw: {"status": "test_stub"})
    def model(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "content": json.dumps({"objects": ["synthetic screen"], "actions": ["display"],
            "confidence": 0.9, "evidence_frame_paths": ["frame.jpg"]}), "attempt_count": 1, "attempts": []}
    monkeypatch.setattr(m, "call_vision_model_with_retries", model)
    for done in (19, 38, 57, 76, 80):
        result = m.run_multimodal_frame_analysis(tmp_path, provider_config=LOCAL, limit=19, execute=True)
        assert len(calls) == done
        assert result["summary"]["continuation"]["pending_count"] == 80 - done
        assert result["run_registry"]["status"] == ("completed" if done == 80 else "needs_execution")
    task = json.loads((tmp_path / "semantic-frame-next-batch.json").read_text())
    assert task["status"] == "complete" and task["next_indexes"] == []
