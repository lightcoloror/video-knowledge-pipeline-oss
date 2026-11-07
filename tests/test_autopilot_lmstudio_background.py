from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from video_knowledge_pipeline import autopilot
from video_knowledge_pipeline import autopilot_vision_executor as provider_executor


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _semantic_bundle(tmp_path: Path, *, count: int = 1) -> Path:
    root = tmp_path / "bundle"
    assets = root / "assets"
    assets.mkdir(parents=True)
    _write_json(root / "manifest.json", {"schema": "lecture_webui_bundle.v1", "title": "Synthetic"})
    timeline = []
    for index in range(1, count + 1):
        frame = assets / f"{index:04d}-{index:03d}_{index * 500:010d}ms.jpg"
        frame.write_bytes(f"synthetic-semantic-frame-{index}".encode("utf-8"))
        timeline.append(
            {
                "index": index,
                "start": float(index - 1),
                "end": float(index),
                "visual_route": "semantic_frame",
                "transcript": f"synthetic transcript {index}",
                "visual_text": f"synthetic OCR {index}",
                "visual_understanding": {},
                "quality_issues": ["missing_visual_understanding"],
                "assets": [{"path": f"assets/{frame.name}", "copied": "true"}],
            }
        )
    _write_json(
        root / "timeline.json",
        timeline,
    )
    return root


def _temporal_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "temporal-bundle"
    assets = root / "assets"
    assets.mkdir(parents=True)
    frames = []
    for index in range(5):
        frame = assets / f"0001-{index + 1:03d}_{index * 500:010d}ms.jpg"
        frame.write_bytes(f"synthetic-temporal-{index}".encode("utf-8"))
        frames.append(str(frame))
    _write_json(root / "manifest.json", {"schema": "lecture_webui_bundle.v1", "title": "Synthetic temporal"})
    _write_json(
        root / "timeline.json",
        [
            {
                "index": 1,
                "start": 0.0,
                "end": 2.5,
                "visual_route": "temporal_sequence",
                "transcript": "synthetic operation",
                "visual_text": "step",
                "temporal_frame_paths": frames,
                "temporal_visual_understanding": {},
                "quality_issues": ["temporal_sequence_without_analysis"],
            }
        ],
    )
    return root


def _action(*, temporal: bool = False) -> dict[str, object]:
    return {
        "status": "coverage_blocked",
        "kind": "knowledge_coverage",
        "key": "temporal_visual_analysis" if temporal else "multimodal_frame_analysis",
        "label": "视觉理解",
        "reason": "需要观察合成帧",
        "mcp_tool": "run_temporal_visual_analysis" if temporal else "run_multimodal_frame_analysis",
        "mcp_args_path": "",
        "human_required": False,
    }


def _fake_success(calls: list[dict[str, object]], *, model: str = "qwen/qwen3-vl-8b"):
    def call(task: str, **kwargs: object) -> dict[str, object]:
        calls.append({"task": task, **kwargs})
        count = len(calls)
        return {
            "ok": True,
            "status": "ok",
            "complete": True,
            "truncated": False,
            "finish_reason": "stop",
            "content": json.dumps(
                {
                    "objects": [f"screen-{count}"],
                    "actions": [],
                    "interface_state": f"state-{count}",
                    "spatial_relations": [],
                    "instructor_focus": "screen",
                    "non_text_information": ["synthetic"],
                    "confidence": 0.8,
                    "keep_image_reason": "evidence",
                }
            ),
            "raw_response": {"model": model},
        }

    return call


def _next_result(action: dict[str, object]) -> dict[str, object]:
    return {"status": action["status"], "next_action": action}


def test_background_local_route_calls_lmstudio_adapter_and_imports_one_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _semantic_bundle(tmp_path)
    job_root = tmp_path / "jobs"
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *_args, **_kwargs: _next_result(_action()))
    monkeypatch.setattr(provider_executor, "model_task_api_call", _fake_success(calls))

    started = autopilot.autopilot_start(
        bundle,
        runner="background",
        execution_route="local-only",
        job_root=job_root,
    )
    result = autopilot.autopilot_next(started["job_id"], job_root=job_root)

    assert result["status"] == "succeeded"
    assert len(calls) == 1
    assert calls[0]["execute"] is True
    assert len(calls[0]["image_paths"]) == 1
    assert calls[0]["provider_config"]["base_url"] == "http://127.0.0.1:1234/v1"
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    understanding = timeline[0]["visual_understanding"]
    assert understanding["executor"] == "provider_gateway"
    assert understanding["model_actual"] == "qwen/qwen3-vl-8b"
    assert understanding["evidence_level"] == "confirmed"


def test_background_semantic_work_item_is_one_frame_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _semantic_bundle(tmp_path, count=3)
    job_root = tmp_path / "jobs"
    calls: list[dict[str, object]] = []
    leased: list[dict[str, object]] = []
    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *_args, **_kwargs: _next_result(_action()))
    monkeypatch.setattr(provider_executor, "model_task_api_call", _fake_success(calls))

    def execute(work_item: dict[str, object]) -> dict[str, object]:
        leased.append(work_item)
        return provider_executor.execute_provider_vision_work_item(work_item)

    monkeypatch.setattr(autopilot, "execute_provider_vision_work_item", execute)
    started = autopilot.autopilot_start(
        bundle,
        runner="background",
        execution_route="local-only",
        job_root=job_root,
    )

    result = autopilot.autopilot_next(started["job_id"], job_root=job_root)

    assert result["status"] == "succeeded"
    assert len(calls) == 1
    pack_ref = next(row for row in leased[0]["input_refs"] if row["role"] == "provider_task_pack")
    pack = json.loads(Path(pack_ref["path"]).read_text(encoding="utf-8"))
    assert len(pack["items"]) == 1
    assert len(pack["items"][0]["frames"]) == 1


def test_background_local_route_converts_stale_codex_session_to_provider_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _semantic_bundle(tmp_path)
    job_root = tmp_path / "jobs"
    calls: list[dict[str, object]] = []
    leased: list[dict[str, object]] = []
    stale_interactive_action = {
        "status": "interactive_session_ready",
        "kind": "codex_session",
        "key": "prepare_codex_semantic_vision_session",
        "label": "准备 Codex 单帧视觉包",
        "reason": "旧 Provider preflight 仍然阻塞",
        "mcp_tool": "prepare_codex_vision_session",
        "mcp_args_path": "",
        "human_required": True,
    }
    monkeypatch.setattr(
        autopilot,
        "bundle_next_action",
        lambda *_args, **_kwargs: _next_result(stale_interactive_action),
    )
    monkeypatch.setattr(provider_executor, "model_task_api_call", _fake_success(calls))

    def execute(work_item: dict[str, object]) -> dict[str, object]:
        leased.append(work_item)
        return provider_executor.execute_provider_vision_work_item(work_item)

    monkeypatch.setattr(autopilot, "execute_provider_vision_work_item", execute)

    started = autopilot.autopilot_start(
        bundle,
        runner="background",
        execution_route="local-only",
        job_root=job_root,
    )
    result = autopilot.autopilot_next(started["job_id"], job_root=job_root)

    assert result["status"] == "succeeded"
    assert len(calls) == 1
    assert leased[0]["task_type"] == "provider_vision_session"
    assert any(
        row.get("role") == "provider_task_pack"
        for row in leased[0]["input_refs"]
    )


def test_background_local_route_advances_beyond_stale_codex_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _semantic_bundle(tmp_path, count=2)
    job_root = tmp_path / "jobs"
    calls: list[dict[str, object]] = []
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    timeline[0]["visual_understanding"] = {
        "objects": ["already complete"],
        "evidence_level": "confirmed",
    }
    _write_json(bundle / "timeline.json", timeline)
    stale_args = bundle / "mcp-prepare-codex-vision-session.args.json"
    _write_json(stale_args, {"kind": "semantic", "indexes": [1]})
    stale_action = {
        "status": "interactive_session_ready",
        "kind": "codex_session",
        "key": "prepare_codex_semantic_vision_session",
        "label": "准备 Codex 单帧视觉包",
        "reason": "旧会话包仍绑定已完成的 index 1",
        "mcp_tool": "prepare_codex_vision_session",
        "mcp_args_path": str(stale_args),
        "human_required": True,
    }
    monkeypatch.setattr(
        autopilot,
        "bundle_next_action",
        lambda *_args, **_kwargs: _next_result(stale_action),
    )
    monkeypatch.setattr(provider_executor, "model_task_api_call", _fake_success(calls))

    started = autopilot.autopilot_start(
        bundle,
        runner="background",
        execution_route="local-only",
        job_root=job_root,
    )
    result = autopilot.autopilot_next(started["job_id"], job_root=job_root)

    assert result["status"] == "succeeded"
    assert len(calls) == 1
    assert Path(str(calls[0]["image_paths"][0])).name.startswith("0002-")
    updated = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert updated[1]["visual_understanding"]["evidence_level"] == "confirmed"


def test_provider_executor_missing_pack_returns_retryable_submission_without_call() -> None:
    calls: list[dict[str, object]] = []

    def fail_if_called(*_args: object, **_kwargs: object) -> dict[str, object]:
        calls.append({"called": True})
        raise AssertionError("provider must not be called without a hash-bound task pack")

    work_item = {
        "job_id": "job",
        "bundle_identity_sha256": "a" * 64,
        "state_revision": 3,
        "task_id": "task",
        "lease_id": "lease",
        "input_refs": [],
    }

    submission = provider_executor.execute_provider_vision_work_item(
        work_item,
        call_model=fail_if_called,
    )

    assert submission["result"]["status"] == "retryable"
    assert submission["result"]["reason"] == "provider_task_pack_missing"
    assert calls == []


def test_provider_unavailable_waits_without_implicit_retry_then_explicitly_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _semantic_bundle(tmp_path)
    job_root = tmp_path / "jobs"
    calls: list[dict[str, object]] = []
    ready = {"value": False}

    def call(task: str, **kwargs: object) -> dict[str, object]:
        calls.append({"task": task, **kwargs})
        if not ready["value"]:
            return {"ok": False, "status": "failed", "complete": False, "error": "[WinError 10061] connection refused"}
        return _fake_success([])(task, **kwargs)

    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *_args, **_kwargs: _next_result(_action()))
    monkeypatch.setattr(provider_executor, "model_task_api_call", call)
    before = (bundle / "timeline.json").read_bytes()
    started = autopilot.autopilot_start(bundle, runner="background", execution_route="local-only", job_root=job_root)

    waiting = autopilot.autopilot_next(started["job_id"], job_root=job_root)
    assert waiting["status"] == "waiting_provider"
    assert waiting["job"]["provider_wait"]["reason"] == "local_provider_unavailable"
    assert (bundle / "timeline.json").read_bytes() == before
    assert len(calls) == 1

    unchanged = autopilot.autopilot_next(started["job_id"], job_root=job_root)
    assert unchanged["status"] == "waiting_provider"
    assert len(calls) == 1

    ready["value"] = True
    resumed = autopilot.autopilot_next(started["job_id"], job_root=job_root, retry_provider=True)
    assert resumed["status"] == "succeeded"
    assert len(calls) == 2


def test_autopilot_run_explicit_retry_recovers_one_new_local_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _semantic_bundle(tmp_path, count=2)
    job_root = tmp_path / "jobs"
    calls: list[dict[str, object]] = []

    def call(task: str, **kwargs: object) -> dict[str, object]:
        calls.append({"task": task, **kwargs})
        if len(calls) == 2:
            return {
                "ok": False,
                "status": "failed",
                "complete": False,
                "error": "TimeoutError: timed out",
            }
        return _fake_success([])(task, **kwargs)

    def next_action(root: Path, **_kwargs: object) -> dict[str, object]:
        timeline = json.loads((Path(root) / "timeline.json").read_text(encoding="utf-8"))
        if all(bool(row.get("visual_understanding")) for row in timeline):
            return {
                "status": "ready",
                "next_action": {
                    "status": "ready",
                    "kind": "export",
                    "key": "refresh_review_outputs",
                    "mcp_tool": "refresh_lecture_review_outputs",
                },
            }
        return _next_result(_action())

    monkeypatch.setattr(autopilot, "bundle_next_action", next_action)
    monkeypatch.setattr(provider_executor, "model_task_api_call", call)

    result = autopilot.autopilot_run(
        bundle,
        runner="background",
        execution_route="local-only",
        job_root=job_root,
        retry_provider=True,
    )

    assert result["state"] == "complete_draft"
    assert len(calls) == 3
    assert result["budget_usage"]["local_calls"] == 3
    assert result["provider_vision_target_outcomes"]["semantic:1"]["status"] == "succeeded"
    assert result["provider_vision_target_outcomes"]["semantic:2"]["status"] == "succeeded"


def test_loaded_model_mismatch_waits_and_preserves_timeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _semantic_bundle(tmp_path)
    job_root = tmp_path / "jobs"
    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *_args, **_kwargs: _next_result(_action()))
    monkeypatch.setattr(provider_executor, "model_task_api_call", _fake_success([], model="qwen/qwen3.5-9b"))
    before = (bundle / "timeline.json").read_bytes()
    started = autopilot.autopilot_start(bundle, runner="background", execution_route="local-only", job_root=job_root)

    waiting = autopilot.autopilot_next(started["job_id"], job_root=job_root)

    assert waiting["status"] == "waiting_provider"
    assert waiting["job"]["provider_wait"]["reason"] == "loaded_model_mismatch"
    assert (bundle / "timeline.json").read_bytes() == before


def test_truncated_local_output_retries_once_with_compact_contract_and_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _semantic_bundle(tmp_path)
    job_root = tmp_path / "jobs"
    calls: list[dict[str, object]] = []

    def truncated_then_compact(*_args: object, **kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            return {
                "ok": True,
                "status": "ok",
                "complete": False,
                "truncated": True,
                "finish_reason": "length",
                "content": '{"objects":["partial"',
                "raw_response": {"model": "qwen/qwen3-vl-8b"},
            }
        return {
            "ok": True,
            "status": "ok",
            "complete": True,
            "truncated": False,
            "finish_reason": "stop",
            "content": json.dumps(
                {
                    "objects": ["compact-screen"],
                    "actions": [],
                    "interface_state": "static slide",
                    "spatial_relations": [],
                    "instructor_focus": "screen",
                    "non_text_information": ["compact retry succeeded"],
                    "confidence": 0.8,
                    "keep_image_reason": "evidence",
                }
            ),
            "raw_response": {"model": "qwen/qwen3-vl-8b"},
        }

    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *_args, **_kwargs: _next_result(_action()))
    monkeypatch.setattr(provider_executor, "model_task_api_call", truncated_then_compact)
    started = autopilot.autopilot_start(bundle, runner="background", execution_route="local-only", job_root=job_root)

    result = autopilot.autopilot_next(started["job_id"], job_root=job_root)

    assert result["status"] == "succeeded"
    assert result["job"]["current_work_item"] is None
    assert result["job"]["provider_wait"] == {}
    assert result["job"]["budget_usage"]["local_calls"] == 2
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 4096
    assert calls[1]["max_tokens"] == 4096
    assert "compact JSON" in str(calls[1]["prompt"])
    assert not any(gap["key"] == "provider_output_truncated" for gap in result["job"]["review_gaps"])
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert timeline[0]["visual_understanding"]["objects"] == ["compact-screen"]


def test_persistent_truncation_skips_failed_index_and_does_not_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _semantic_bundle(tmp_path, count=2)
    job_root = tmp_path / "jobs"
    calls: list[dict[str, object]] = []
    leased_indexes: list[int] = []

    def truncated(*_args: object, **kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {
            "ok": True,
            "status": "ok",
            "complete": False,
            "truncated": True,
            "finish_reason": "length",
            "content": '{"objects":["partial"',
            "raw_response": {"model": "qwen/qwen3-vl-8b"},
        }

    def execute(work_item: dict[str, object]) -> dict[str, object]:
        pack_ref = next(row for row in work_item["input_refs"] if row["role"] == "provider_task_pack")
        pack = json.loads(Path(pack_ref["path"]).read_text(encoding="utf-8"))
        leased_indexes.append(int(pack["items"][0]["index"]))
        return provider_executor.execute_provider_vision_work_item(work_item)

    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *_args, **_kwargs: _next_result(_action()))
    monkeypatch.setattr(provider_executor, "model_task_api_call", truncated)
    monkeypatch.setattr(autopilot, "execute_provider_vision_work_item", execute)
    before = (bundle / "timeline.json").read_bytes()
    started = autopilot.autopilot_start(bundle, runner="background", execution_route="local-only", job_root=job_root)

    first = autopilot.autopilot_next(started["job_id"], job_root=job_root)
    second = autopilot.autopilot_next(started["job_id"], job_root=job_root)
    final = autopilot.autopilot_next(started["job_id"], job_root=job_root)

    assert first["status"] == "succeeded"
    assert second["status"] == "succeeded"
    assert final["state"] == "complete_draft"
    assert leased_indexes == [1, 2]
    assert len(calls) == 4
    assert final["provider_vision_target_outcomes"]["semantic:1"]["status"] == "review_gap"
    assert final["provider_vision_target_outcomes"]["semantic:2"]["status"] == "review_gap"
    assert (bundle / "timeline.json").read_bytes() == before


def test_local_result_item_list_is_normalised_to_nonempty_visual_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _semantic_bundle(tmp_path)
    job_root = tmp_path / "jobs"

    def result_items(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "ok": True,
            "status": "ok",
            "complete": True,
            "truncated": False,
            "finish_reason": "stop",
            "content": json.dumps(
                {
                    "result": [
                        {"object_type": "会议标题", "description": "画面中央显示全省大早会标题。"},
                        {"object_type": "日期", "description": "标题下方显示 2026.08.31。"},
                    ]
                },
                ensure_ascii=False,
            ),
            "raw_response": {"model": "qwen/qwen3-vl-8b"},
        }

    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *_args, **_kwargs: _next_result(_action()))
    monkeypatch.setattr(provider_executor, "model_task_api_call", result_items)
    started = autopilot.autopilot_start(bundle, runner="background", execution_route="local-only", job_root=job_root)

    result = autopilot.autopilot_next(started["job_id"], job_root=job_root)

    assert result["status"] == "succeeded"
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    understanding = timeline[0]["visual_understanding"]
    assert "会议标题" in understanding["objects"]
    assert any("全省大早会" in item for item in understanding["non_text_information"])
    assert understanding["validation_status"] == "ok"


def test_empty_local_visual_json_becomes_review_gap_without_timeline_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _semantic_bundle(tmp_path)
    job_root = tmp_path / "jobs"

    def empty_result(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "ok": True,
            "status": "ok",
            "complete": True,
            "truncated": False,
            "finish_reason": "stop",
            "content": "{}",
            "raw_response": {"model": "qwen/qwen3-vl-8b"},
        }

    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *_args, **_kwargs: _next_result(_action()))
    monkeypatch.setattr(provider_executor, "model_task_api_call", empty_result)
    before = (bundle / "timeline.json").read_bytes()
    started = autopilot.autopilot_start(bundle, runner="background", execution_route="local-only", job_root=job_root)

    result = autopilot.autopilot_next(started["job_id"], job_root=job_root)

    assert result["status"] == "succeeded"
    assert result["job"]["current_work_item"] is None
    assert any(gap["key"] == "model_visual_content_empty" for gap in result["job"]["review_gaps"])
    assert (bundle / "timeline.json").read_bytes() == before


def test_provider_exception_becomes_explicit_wait_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _semantic_bundle(tmp_path)
    job_root = tmp_path / "jobs"

    def raise_at_provider_boundary(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("synthetic provider crash")

    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *_args, **_kwargs: _next_result(_action()))
    monkeypatch.setattr(provider_executor, "model_task_api_call", raise_at_provider_boundary)
    before = (bundle / "timeline.json").read_bytes()
    started = autopilot.autopilot_start(bundle, runner="background", execution_route="local-only", job_root=job_root)

    waiting = autopilot.autopilot_next(started["job_id"], job_root=job_root)

    assert waiting["status"] == "waiting_provider"
    assert waiting["job"]["provider_wait"]["reason"] == "local_provider_failed"
    assert "synthetic provider crash" in waiting["job"]["provider_wait"]["detail"]
    assert (bundle / "timeline.json").read_bytes() == before


def test_temporal_background_uses_five_single_frame_calls_and_aggregates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _temporal_bundle(tmp_path)
    job_root = tmp_path / "jobs"
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *_args, **_kwargs: _next_result(_action(temporal=True)))
    monkeypatch.setattr(provider_executor, "model_task_api_call", _fake_success(calls))
    started = autopilot.autopilot_start(bundle, runner="background", execution_route="local-only", job_root=job_root)

    result = autopilot.autopilot_next(started["job_id"], job_root=job_root)

    assert result["status"] == "succeeded"
    assert len(calls) == 5
    assert all(len(row["image_paths"]) == 1 for row in calls)
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    understanding = timeline[0]["temporal_visual_understanding"]
    assert understanding["observed_frame_count"] == 5
    assert len(understanding["per_frame_observations"]) == 5
    assert understanding["validation_status"] == "ok"
    assert understanding["causal_links"] == []
    assert understanding["executor"] == "provider_gateway"


def test_cli_and_mcp_keep_four_tools_with_explicit_provider_retry() -> None:
    from video_knowledge_pipeline.cli import build_parser
    from video_knowledge_pipeline.autopilot_mcp import build_server

    parsed = build_parser().parse_args(["autopilot-next", "abc", "--retry-provider"])
    assert parsed.retry_provider is True
    server = build_server()
    tools = asyncio.run(server.list_tools())
    names = {getattr(tool, "name", "") for tool in tools}
    assert names == {"autopilot_start", "autopilot_next", "autopilot_submit", "autopilot_status"}


def test_remote_or_malformed_provider_route_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_executor, "resolve_model_api_provider_config", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        provider_executor,
        "vision_execution_profile",
        lambda: {
            "provider": "local_vlm",
            "model": "qwen/qwen3-vl-8b",
            "base_url": "https://remote.example.invalid/v1",
        },
    )
    remote = provider_executor.local_vision_provider_snapshot()
    assert remote["status"] == "blocked"
    assert remote["ready_to_attempt"] is False
    assert remote["remote_fallback_allowed"] is False
    assert any(item["key"] == "loopback_base_url_required" for item in remote["blockers"])

    monkeypatch.setattr(
        provider_executor,
        "vision_execution_profile",
        lambda: {
            "provider": "local_vlm",
            "model": "qwen/qwen3-vl-8b",
            "base_url": "http://127.0.0.1:not-a-port/v1",
        },
    )
    malformed = provider_executor.local_vision_provider_snapshot()
    assert malformed["status"] == "blocked"
    assert malformed["ready_to_attempt"] is False


def test_autopilot_local_provider_timeout_is_bounded_below_agent_command_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        provider_executor,
        "resolve_model_api_provider_config",
        lambda *_args, **_kwargs: {
            "provider": "local_vlm",
            "model": "qwen/qwen3-vl-8b",
            "base_url": "http://127.0.0.1:1234/v1",
            "timeout_seconds": 90,
        },
    )

    snapshot = provider_executor.local_vision_provider_snapshot()

    assert snapshot["ready_to_attempt"] is True
    assert snapshot["provider"]["timeout_seconds"] == 60
