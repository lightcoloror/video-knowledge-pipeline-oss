from __future__ import annotations

import json
import asyncio
from pathlib import Path

import pytest

from video_knowledge_pipeline import autopilot
from video_knowledge_pipeline.autopilot import (
    MAX_WORK_ITEM_BYTES,
    SUBMISSION_SCHEMA,
    autopilot_bundle_summary,
    autopilot_next,
    autopilot_start,
    autopilot_status,
    autopilot_submit,
    resolve_execution_route,
)
from video_knowledge_pipeline.cli import build_parser
from video_knowledge_pipeline.autopilot_mcp import build_server


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _bundle(tmp_path: Path, *, name: str = "bundle") -> Path:
    root = tmp_path / name
    assets = root / "assets"
    assets.mkdir(parents=True)
    frame = assets / "0001-001_0000000500ms.jpg"
    frame.write_bytes(b"synthetic-frame")
    _write_json(root / "manifest.json", {"schema": "lecture_webui_bundle.v1", "title": "Synthetic"})
    _write_json(
        root / "timeline.json",
        [
            {
                "index": 1,
                "start": 0.0,
                "end": 1.0,
                "visual_route": "semantic_frame",
                "transcript": "synthetic transcript",
                "visual_text": "synthetic OCR",
                "visual_understanding": {},
                "temporal_visual_understanding": {},
                "quality_issues": ["missing_visual_understanding"],
                "assets": [{"path": f"assets/{frame.name}", "copied": "true"}],
            }
        ],
    )
    return root


def _model_action() -> dict[str, object]:
    return {
        "status": "coverage_blocked",
        "kind": "knowledge_coverage",
        "key": "multimodal_frame_analysis",
        "label": "视觉理解",
        "reason": "需要观察合成帧",
        "mcp_tool": "run_multimodal_frame_analysis",
        "mcp_args_path": "",
        "human_required": False,
    }


def _next_result(action: dict[str, object]) -> dict[str, object]:
    return {"status": action["status"], "next_action": action}


def _fill_codex_result(work_item: dict[str, object]) -> dict[str, object]:
    refs = [row for row in work_item["input_refs"] if isinstance(row, dict)]
    template_path = Path(next(str(row["path"]) for row in refs if row.get("role") == "result_template_path"))
    result = json.loads(template_path.read_text(encoding="utf-8"))
    result["model_actual"] = "gpt-5.6-luna"
    for row in result["items"]:
        row["evidence_level"] = "confirmed"
        row["observed_frame_ids"] = list(row["expected_frame_ids"])
        row["visual_understanding"] = {
            "objects": ["synthetic slide"],
            "actions": ["presenting"],
            "interface_state": "slide visible",
            "spatial_relations": ["text above image"],
            "instructor_focus": "slide",
            "non_text_information": ["blue background"],
            "confidence": 0.8,
            "keep_image_reason": "layout evidence",
            "evidence_frame_paths": list(row["frame_paths"]),
        }
    return result


def _submission(work_item: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    return {
        "schema": SUBMISSION_SCHEMA,
        "job_id": work_item["job_id"],
        "bundle_identity_sha256": work_item["bundle_identity_sha256"],
        "state_revision": work_item["state_revision"],
        "task_id": work_item["task_id"],
        "lease_id": work_item["lease_id"],
        "result": result,
    }


def _fill_provider_result(work_item: dict[str, object]) -> dict[str, object]:
    refs = [row for row in work_item["input_refs"] if isinstance(row, dict)]
    pack_path = Path(next(str(row["path"]) for row in refs if row.get("role") == "provider_task_pack"))
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    model = str(pack["provider"]["model"])
    items = []
    for task in pack["items"]:
        frame_paths = [str(row["path"]) for row in task["frames"]]
        items.append(
            {
                "index": task["index"],
                "visual_understanding": {
                    "objects": ["synthetic slide"],
                    "actions": [],
                    "interface_state": "slide visible",
                    "spatial_relations": ["text above image"],
                    "instructor_focus": "slide",
                    "non_text_information": ["blue background"],
                    "confidence": 0.8,
                    "keep_image_reason": "layout evidence",
                    "evidence_frame_paths": frame_paths,
                    "source": "synthetic_provider",
                    "executor": "provider_gateway",
                    "model_requested": model,
                    "model_actual": model,
                    "evidence_level": "confirmed",
                },
            }
        )
    return {
        "schema": "video_knowledge_pipeline.autopilot_provider_vision_result.v1",
        "status": "succeeded",
        "kind": pack["kind"],
        "task_pack_sha256": pack["task_pack_sha256"],
        "provider_snapshot_sha256": pack["provider_snapshot_sha256"],
        "model_requested": model,
        "model_actual": model,
        "provider_call_count": len(items),
        "request_mode": "single_frame_sequential",
        "max_images_per_request": 1,
        "items": items,
        "calls": [],
    }


def test_ready_bundle_reaches_complete_draft_but_never_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _bundle(tmp_path)
    job_root = tmp_path / "job-pointers"
    ready = {"status": "ready", "kind": "export", "key": "refresh_review_outputs", "mcp_tool": "refresh_lecture_review_outputs"}
    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *_args, **_kwargs: _next_result(ready))

    started = autopilot_start(bundle, job_root=job_root)
    finished = autopilot_next(started["job_id"], job_root=job_root)

    assert finished["state"] == "complete_draft"
    assert finished["accepted_written"] is False
    assert autopilot_status(started["job_id"], job_root=job_root)["state"] == "complete_draft"

    retry = autopilot_next(started["job_id"], job_root=job_root, retry_provider=True)

    assert retry["state"] == "complete_draft"
    assert retry["retry_provider"]["status"] == "not_applicable"
    assert retry["retry_provider"]["mutated"] is False


def test_one_hash_bound_work_item_then_atomic_codex_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _bundle(tmp_path)
    job_root = tmp_path / "job-pointers"
    actions = [_model_action()]
    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *_args, **_kwargs: _next_result(actions[-1]))
    started = autopilot_start(bundle, job_root=job_root)

    waiting = autopilot_next(started["job_id"], job_root=job_root)
    work_item = waiting["work_item"]

    assert waiting["status"] == "waiting_model"
    assert work_item["task_type"] == "codex_vision_session"
    assert len(json.dumps(work_item, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) <= MAX_WORK_ITEM_BYTES
    assert {row["role"] for row in work_item["input_refs"]} >= {"manifest", "timeline", "task_pack_path", "result_template_path"}

    result = _fill_codex_result(work_item)
    receipt = autopilot_submit(started["job_id"], submission=_submission(work_item, result), job_root=job_root)

    assert receipt["status"] == "succeeded"
    assert receipt["import_result"]["status"] == "imported"
    assert autopilot_status(started["job_id"], job_root=job_root)["state"] == "executing"
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert timeline[0]["visual_understanding"]["evidence_level"] == "confirmed"

    ready = {"status": "ready", "kind": "export", "key": "refresh_review_outputs", "mcp_tool": "refresh_lecture_review_outputs"}
    actions.append(ready)
    finished = autopilot_next(started["job_id"], job_root=job_root)
    assert finished["state"] == "complete_draft"


def test_stale_revision_is_rejected_without_timeline_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _bundle(tmp_path)
    job_root = tmp_path / "job-pointers"
    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *_args, **_kwargs: _next_result(_model_action()))
    started = autopilot_start(bundle, job_root=job_root)
    waiting = autopilot_next(started["job_id"], job_root=job_root)
    work_item = waiting["work_item"]
    before = (bundle / "timeline.json").read_bytes()
    payload = _submission(work_item, _fill_codex_result(work_item))
    payload["state_revision"] = int(payload["state_revision"]) - 1

    receipt = autopilot_submit(started["job_id"], submission=payload, job_root=job_root)

    assert receipt["status"] == "rejected"
    assert "state_revision_mismatch" in {row["key"] for row in receipt["issues"]}
    assert (bundle / "timeline.json").read_bytes() == before
    assert autopilot_status(started["job_id"], job_root=job_root)["status"] == "waiting_model"
    assert Path(receipt["artifacts"]["raw_result"]).is_file()


def test_input_hash_drift_and_duplicate_or_old_lease_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _bundle(tmp_path)
    job_root = tmp_path / "job-pointers"
    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *_args, **_kwargs: _next_result(_model_action()))
    started = autopilot_start(bundle, job_root=job_root)
    work_item = autopilot_next(started["job_id"], job_root=job_root)["work_item"]
    payload = _submission(work_item, _fill_codex_result(work_item))
    Path(next(row["path"] for row in work_item["input_refs"] if row["role"] == "prompt_path")).write_text("drift", encoding="utf-8")
    payload["lease_id"] = "old-lease"

    receipt = autopilot_submit(started["job_id"], submission=payload, job_root=job_root)

    keys = {row["key"] for row in receipt["issues"]}
    assert {"lease_id_mismatch", "input_hash_drift"} <= keys
    assert receipt["changed"] is False


def test_route_policy_uses_two_busy_samples_and_long_video() -> None:
    busy = {"busy": True, "busy_reasons": ["cpu_at_or_above_70_percent"]}
    assert resolve_execution_route("auto", resource_samples=[busy])["resolved_route"] == "hybrid"
    assert resolve_execution_route("auto", resource_samples=[busy, busy])["resolved_route"] == "online-heavy"
    assert resolve_execution_route("auto", duration_seconds=90 * 60)["basis"] == "long_video"
    assert resolve_execution_route("local-only", duration_seconds=100 * 60)["resolved_route"] == "local-only"


def test_background_without_explicit_executor_finishes_with_review_gap_and_no_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    job_root = tmp_path / "job-pointers"
    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *_args, **_kwargs: _next_result(_model_action()))
    prepared = []
    monkeypatch.setattr(autopilot, "prepare_codex_vision_session", lambda *_args, **_kwargs: prepared.append(True))
    started = autopilot_start(bundle, runner="background", execution_route="online-only", job_root=job_root)

    result = autopilot_next(started["job_id"], job_root=job_root)

    assert result["state"] == "complete_draft"
    assert result["review_gaps"][0]["key"] == "background_executor_not_authorized"
    assert prepared == []
    assert result["budget_usage"]["model_calls"] == 0


def test_background_executor_uses_same_work_item_and_atomic_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    job_root = tmp_path / "job-pointers"
    actions = [_model_action()]
    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *_args, **_kwargs: _next_result(actions[-1]))
    started = autopilot_start(bundle, runner="background", job_root=job_root)

    result = autopilot_next(
        started["job_id"],
        job_root=job_root,
        background_executor=lambda work_item: _submission(work_item, _fill_provider_result(work_item)),
    )

    assert result["status"] == "succeeded"
    assert result["import_result"]["status"] == "imported"
    assert autopilot_status(started["job_id"], job_root=job_root)["state"] == "executing"


def test_batch_status_keeps_bundle_business_content_out_of_coordinator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = _bundle(tmp_path, name="one")
    second = _bundle(tmp_path, name="two")
    batch_path = tmp_path / "batch.json"
    _write_json(batch_path, {"bundles": [str(first), str(second)]})
    job_root = tmp_path / "job-pointers"
    ready = {"status": "ready", "kind": "export", "key": "refresh_review_outputs", "mcp_tool": "refresh_lecture_review_outputs"}
    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *_args, **_kwargs: _next_result(ready))

    batch = autopilot_start(batch_path, job_root=job_root)
    pointer = json.loads((job_root / f"{batch['job_id']}.json").read_text(encoding="utf-8"))

    assert len(batch["children"]) == 2
    assert set(pointer) == {"schema", "kind", "job_id", "batch_manifest", "children", "created_at"}
    assert "timeline" not in json.dumps(pointer)


def test_cli_exposes_exact_autopilot_commands_and_ui_summary(tmp_path: Path) -> None:
    parser = build_parser()
    assert parser.parse_args(["autopilot-start", "bundle"]).runner == "agent-session"
    assert parser.parse_args(["autopilot-next", "abc"]).job_id == "abc"
    assert parser.parse_args(["autopilot-submit", "abc", "--input-json", "result.json"]).input_json == "result.json"
    assert parser.parse_args(["autopilot-status", "abc"]).job_id == "abc"
    bundle = _bundle(tmp_path)
    summary = autopilot_bundle_summary(bundle)
    assert summary["state"] == "not_started"
    assert "autopilot-start" in summary["start_command"]


def test_autopilot_mcp_profile_exposes_only_four_tools() -> None:
    server = build_server()
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {
        "autopilot_start",
        "autopilot_next",
        "autopilot_submit",
        "autopilot_status",
    }


def test_invalid_json_is_preserved_and_rejected_without_business_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    job_root = tmp_path / "job-pointers"
    monkeypatch.setattr(autopilot, "bundle_next_action", lambda *_args, **_kwargs: _next_result(_model_action()))
    started = autopilot_start(bundle, job_root=job_root)
    autopilot_next(started["job_id"], job_root=job_root)
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    before = (bundle / "timeline.json").read_bytes()

    receipt = autopilot_submit(started["job_id"], input_json=malformed, job_root=job_root)

    assert receipt["status"] == "rejected"
    assert "input_json_invalid" in {row["key"] for row in receipt["issues"]}
    assert Path(receipt["artifacts"]["raw_result"]).read_bytes() == malformed.read_bytes()
    assert (bundle / "timeline.json").read_bytes() == before
