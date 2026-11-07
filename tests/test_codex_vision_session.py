from __future__ import annotations

import json
import inspect
from pathlib import Path

import pytest

from video_knowledge_pipeline import mcp_server
from video_knowledge_pipeline.bundle_next import bundle_next_action
from video_knowledge_pipeline.cli import _mcp_callables, build_parser
from video_knowledge_pipeline.codex_vision_session import (
    import_codex_vision_results,
    prepare_codex_vision_session,
)
from video_knowledge_pipeline.multimodal_frame_analyzer import (
    vision_analysis_apply_restore,
    vision_analysis_restore_plan,
)
from video_knowledge_pipeline.smart_summary_chapters import _chapter_citation_digest
from video_knowledge_pipeline.smart_summary_codex import _inferred_visual_language_gate
from video_knowledge_pipeline.smart_summary_input_pack import (
    _compact_evidence_item,
    _review_gap_reasons,
)
from video_knowledge_pipeline.task_console import _build_commands
from video_knowledge_pipeline.video_workbench import _evidence_status_panel_html, _timeline_row
from video_knowledge_pipeline.visual_evidence import evidence_level_counts, visual_evidence_state
from video_knowledge_pipeline.vision_preflight import vision_execution_preflight


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _bundle(tmp_path: Path, *, semantic_count: int = 3, temporal: bool = False) -> Path:
    root = tmp_path / "bundle"
    assets = root / "assets"
    assets.mkdir(parents=True)
    timeline: list[dict[str, object]] = []
    for index in range(1, semantic_count + 1):
        frame = assets / f"{index:04d}-{index:03d}_{index * 1000:010d}ms.jpg"
        frame.write_bytes(b"synthetic-jpeg-" + str(index).encode("ascii"))
        timeline.append(
            {
                "index": index,
                "start": float(index - 1),
                "end": float(index),
                "midpoint": float(index) - 0.5,
                "visual_route": "semantic_frame",
                "transcript": f"synthetic transcript {index}",
                "visual_text": f"synthetic OCR {index}",
                "visual_understanding": {},
                "temporal_visual_understanding": {},
                "quality_issues": ["semantic_frame_without_analysis", "missing_visual_understanding"],
                "assets": [{"path": f"assets/{frame.name}", "copied": "true"}],
            }
        )
    if temporal:
        temporal_paths = []
        for position in range(5):
            frame = assets / f"temporal_{position:02d}_{5000 + position * 250:010d}ms.jpg"
            frame.write_bytes(b"synthetic-temporal-" + str(position).encode("ascii"))
            temporal_paths.append(f"assets/{frame.name}")
        timeline.append(
            {
                "index": semantic_count + 1,
                "start": 5.0,
                "end": 7.0,
                "visual_route": "temporal_sequence",
                "transcript": "synthetic operation",
                "visual_text": "button",
                "visual_understanding": {},
                "temporal_visual_understanding": {},
                "quality_issues": ["temporal_sequence_without_analysis", "missing_visual_understanding"],
                "temporal_frame_paths": temporal_paths,
                "assets": [{"path": temporal_paths[0], "copied": "true"}],
            }
        )
    _write_json(root / "manifest.json", {"schema": "lecture_webui_bundle.v1", "title": "Synthetic"})
    _write_json(root / "timeline.json", timeline)
    return root


def _fill_semantic_results(template_path: Path, levels: list[str]) -> Path:
    result = json.loads(template_path.read_text(encoding="utf-8"))
    result["model_actual"] = "gpt-5.6-sol"
    for ordinal, (row, level) in enumerate(zip(result["items"], levels, strict=True), start=1):
        row["evidence_level"] = level
        row["observed_frame_ids"] = list(row["expected_frame_ids"])
        if level == "unavailable":
            row["unavailable_reason"] = "frame inspected but the detail is unreadable"
            continue
        row["visual_understanding"] = {
            "objects": ["slide"],
            "actions": ["presenting"],
            "interface_state": f"title page {ordinal}",
            "spatial_relations": ["title above speaker"],
            "instructor_focus": "title",
            "non_text_information": ["blue layout"],
            "confidence": 0.8,
            "keep_image_reason": "layout matters",
            "evidence_frame_paths": list(row["frame_paths"]),
        }
        if level == "inferred":
            row["inference_basis"] = "speaker position and slide composition"
    output = template_path.with_name("result.json")
    _write_json(output, result)
    return output


def test_prepare_semantic_session_is_local_hash_bound_and_capped(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, semantic_count=23)

    prepared = prepare_codex_vision_session(bundle, kind="semantic", write=True)

    assert prepared["status"] == "ready_for_interactive_analysis"
    assert prepared["executor"] == "codex_session"
    assert prepared["model_requested"] == "luna"
    assert prepared["quality"] == "low"
    assert len(prepared["items"]) == 19
    assert prepared["manifest_sha256"]
    assert prepared["timeline_sha256"]
    assert prepared["bundle_identity_sha256"]
    assert prepared["task_pack_sha256"]
    assert all(len(frame["sha256"]) == 64 for row in prepared["items"] for frame in row["frames"])
    assert all(Path(path).is_relative_to(bundle) for row in prepared["items"] for path in row["frame_paths"])
    template = json.loads(Path(prepared["result_template_path"]).read_text(encoding="utf-8"))
    assert template["task_pack_sha256"] == prepared["task_pack_sha256"]
    assert template["model_actual"] == ""


def test_explicit_semantic_indexes_over_batch_capacity_are_not_silently_truncated(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, semantic_count=23)

    prepared = prepare_codex_vision_session(bundle, kind="semantic", indexes=list(range(1, 22)))

    assert prepared["status"] == "blocked"
    assert len(prepared["selected_indexes"]) == 19
    assert "semantic_batch_capacity_exceeded" in {row["key"] for row in prepared["blockers"]}


def test_preflight_keeps_api_blocked_but_exposes_interactive_session(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    preflight = vision_execution_preflight(
        bundle,
        provider_config={"provider": "openai_compatible", "model": "luna"},
        include_temporal=False,
        write=False,
    )

    assert preflight["ready_to_execute"] is False
    assert "missing_api_key" in {row["key"] for row in preflight["blockers"]}
    assert preflight["interactive_session"]["ready_to_prepare"] is True
    assert preflight["interactive_session"]["api_key_required"] is False
    assert preflight["interactive_session"]["provider_health_check"] == "not_applicable"
    assert preflight["interactive_session"]["unattended_execution"] is False
    assert preflight["interactive_session"]["mcp_args"]["prepare_semantic"]["kind"] == "semantic"
    assert "--kind semantic" in preflight["interactive_session"]["commands"]["prepare_semantic"]


def test_bundle_next_prefers_codex_pack_after_provider_preflight_is_blocked(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    vision_execution_preflight(
        bundle,
        provider_config={"provider": "openai_compatible", "model": "luna"},
        include_temporal=False,
        write=True,
    )

    action = bundle_next_action(bundle)["next_action"]

    assert action["kind"] == "codex_session"
    assert action["mcp_tool"] == "prepare_codex_vision_session"
    assert action["unattended_execution"] is False
    assert "prepare-codex-vision-session" in action["command"]


def test_import_preserves_requested_and_actual_model_and_evidence_levels(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    prepared = prepare_codex_vision_session(bundle, kind="semantic", indexes=[1, 2, 3])
    original_frames = {
        path: Path(path).read_bytes()
        for row in prepared["items"]
        for path in row["frame_paths"]
    }
    input_json = _fill_semantic_results(
        Path(prepared["result_template_path"]),
        ["confirmed", "inferred", "unavailable"],
    )

    imported = import_codex_vision_results(bundle, input_json=input_json)

    assert imported["status"] == "imported"
    assert imported["updated_count"] == 3
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert timeline[0]["visual_understanding"]["evidence_level"] == "confirmed"
    assert timeline[1]["visual_understanding"]["evidence_level"] == "inferred"
    assert timeline[1]["visual_understanding"]["inference_basis"]
    assert timeline[2]["visual_understanding"]["evidence_level"] == "unavailable"
    assert timeline[2]["visual_understanding"]["observation_status"] == "seen_unavailable"
    audit = imported["run_audit"]["record"]
    assert audit["executor"] == "codex_session"
    assert audit["model_requested"] == "luna"
    assert audit["model_actual"] == "gpt-5.6-sol"
    assert audit["quality"] == "low"
    assert audit["execution_mode"] == "interactive_multimodal"
    assert audit["provider_health_check"] == "not_applicable"
    assert Path(imported["raw_result_path"]).read_text(encoding="utf-8") == input_json.read_text(encoding="utf-8")
    assert all(Path(path).read_bytes() == value for path, value in original_frames.items())
    assert imported["vision_restore_hint"]["status"] == "ready"


def test_rejected_invalid_json_preserves_original_bytes_exactly(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, semantic_count=1)
    input_json = tmp_path / "invalid-result.json"
    original = b"\xef\xbb\xbf{\r\n  not-json\r\n"
    input_json.write_bytes(original)

    rejected = import_codex_vision_results(bundle, input_json=input_json)

    assert rejected["status"] == "rejected"
    assert Path(rejected["raw_result_path"]).read_bytes() == original


@pytest.mark.parametrize(
    "mutation",
    ["timeline", "manifest", "frame", "task_pack", "duplicate", "partial", "extra", "time", "nan", "evidence", "unavailable_evidence", "long", "missing_model"],
)
def test_import_validation_is_atomic_and_fail_closed(tmp_path: Path, mutation: str) -> None:
    bundle = _bundle(tmp_path)
    prepared = prepare_codex_vision_session(bundle, kind="semantic", indexes=[1, 2])
    input_json = _fill_semantic_results(Path(prepared["result_template_path"]), ["confirmed", "confirmed"])
    result = json.loads(input_json.read_text(encoding="utf-8"))
    if mutation == "timeline":
        timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
        timeline[0]["transcript"] = "changed after prepare"
        _write_json(bundle / "timeline.json", timeline)
    elif mutation == "manifest":
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        manifest["changed_after_prepare"] = True
        _write_json(bundle / "manifest.json", manifest)
    elif mutation == "frame":
        Path(prepared["items"][0]["frame_paths"][0]).write_bytes(b"changed")
    elif mutation == "task_pack":
        task_pack = json.loads(Path(prepared["task_pack_path"]).read_text(encoding="utf-8"))
        task_pack["question"] = "changed"
        _write_json(Path(prepared["task_pack_path"]), task_pack)
    elif mutation == "duplicate":
        result["items"].append(dict(result["items"][0]))
        _write_json(input_json, result)
    elif mutation == "partial":
        result["items"] = result["items"][:1]
        _write_json(input_json, result)
    elif mutation == "extra":
        extra = dict(result["items"][0])
        extra["task_id"] = "semantic-999999"
        extra["index"] = 999999
        result["items"].append(extra)
        _write_json(input_json, result)
    elif mutation == "time":
        result["items"][0]["end"] += 0.002
        _write_json(input_json, result)
    elif mutation == "nan":
        result["items"][0]["end"] = float("nan")
        _write_json(input_json, result)
    elif mutation == "evidence":
        result["items"][0]["visual_understanding"]["evidence_frame_paths"] = [str(tmp_path / "external.jpg")]
        _write_json(input_json, result)
    elif mutation == "unavailable_evidence":
        result["items"][0]["evidence_level"] = "unavailable"
        result["items"][0]["unavailable_reason"] = "inspected but unreadable"
        result["items"][0]["visual_understanding"]["evidence_frame_paths"] = [str(tmp_path / "external.jpg")]
        _write_json(input_json, result)
    elif mutation == "long":
        result["items"][0]["visual_understanding"]["interface_state"] = "x" * 2001
        _write_json(input_json, result)
    elif mutation == "missing_model":
        result["model_actual"] = ""
        _write_json(input_json, result)
    before = (bundle / "timeline.json").read_bytes()

    rejected = import_codex_vision_results(bundle, input_json=input_json)

    assert rejected["status"] == "rejected"
    assert rejected["updated_count"] == 0
    assert rejected["validation_issues"]
    assert (bundle / "timeline.json").read_bytes() == before
    assert Path(rejected["raw_result_path"]).is_file()


def test_temporal_session_requires_one_complete_ordered_group(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, semantic_count=1, temporal=True)
    prepared = prepare_codex_vision_session(bundle, kind="temporal", indexes=[2])
    assert len(prepared["items"]) == 1
    assert len(prepared["items"][0]["frames"]) == 5
    result = json.loads(Path(prepared["result_template_path"]).read_text(encoding="utf-8"))
    result["model_actual"] = "gpt-5.6-sol"
    row = result["items"][0]
    row["evidence_level"] = "confirmed"
    row["observed_frame_ids"] = list(row["expected_frame_ids"])
    row["temporal_visual_understanding"] = {
        "event_sequence": ["button idle", "button activated"],
        "state_changes": ["idle to active"],
        "operation_steps": ["click button"],
        "causal_links": ["click causes activation"],
        "possible_missing_points": [],
        "confidence": 0.9,
        "observed_frame_count": 5,
        "observed_frame_ids": list(row["expected_frame_ids"]),
        "per_frame_observations": [
            {"frame_id": frame_id, "observation": f"observed {frame_id}"}
            for frame_id in row["expected_frame_ids"]
        ],
        "evidence_frame_paths": list(row["frame_paths"]),
    }
    input_json = Path(prepared["result_template_path"]).with_name("temporal-result.json")
    _write_json(input_json, result)

    imported = import_codex_vision_results(bundle, input_json=input_json)

    assert imported["status"] == "imported"
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assert timeline[1]["temporal_visual_understanding"]["observed_frame_count"] == 5


def test_cli_exposes_codex_session_commands_and_repeatable_indexes() -> None:
    parser = build_parser()
    prepared = parser.parse_args(
        ["prepare-codex-vision-session", "bundle", "--kind", "semantic", "--index", "2", "--index", "5"]
    )
    imported = parser.parse_args(
        ["import-codex-vision-results", "bundle", "--input-json", "result.json"]
    )

    assert prepared.index == [2, 5]
    assert prepared.model_requested == "luna"
    assert prepared.quality == "low"
    assert imported.input_json == "result.json"


def test_mcp_exposes_named_codex_session_tools() -> None:
    callables = _mcp_callables()
    source = inspect.getsource(mcp_server.main)

    assert callables["prepare_codex_vision_session"] is prepare_codex_vision_session
    assert callables["import_codex_vision_results"] is import_codex_vision_results
    assert "def prepare_codex_vision_session(" in source
    assert "def import_codex_vision_results(" in source


def test_prepare_alias_exact_indexes_and_blocked_existing_human_result(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, semantic_count=4)
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    timeline[2]["human_corrected_visual_understanding"] = {
        "objects": ["human reviewed slide"],
        "evidence_frame_paths": [timeline[2]["assets"][0]["path"]],
    }
    _write_json(bundle / "timeline.json", timeline)

    prepared = prepare_codex_vision_session(bundle, kind="semantic", indexes=[2, 3], executor="codex_native")

    assert prepared["executor"] == "codex_session"
    assert prepared["status"] == "blocked"
    assert prepared["selected_indexes"] == [2]
    assert "existing_result" in {row["key"] for row in prepared["blockers"]}
    result = json.loads(Path(prepared["result_template_path"]).read_text(encoding="utf-8"))
    result["model_actual"] = "gpt-5.6-luna"
    row = result["items"][0]
    row["evidence_level"] = "confirmed"
    row["observed_frame_ids"] = list(row["expected_frame_ids"])
    row["visual_understanding"]["objects"] = ["slide"]
    input_json = Path(prepared["result_template_path"]).with_name("blocked-result.json")
    _write_json(input_json, result)
    before = (bundle / "timeline.json").read_bytes()
    rejected = import_codex_vision_results(bundle, input_json=input_json)
    assert rejected["status"] == "rejected"
    assert (bundle / "timeline.json").read_bytes() == before


def test_temporal_partial_per_frame_observation_is_rejected(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, semantic_count=1, temporal=True)
    prepared = prepare_codex_vision_session(bundle, kind="temporal", indexes=[2])
    result = json.loads(Path(prepared["result_template_path"]).read_text(encoding="utf-8"))
    result["model_actual"] = "gpt-5.6-sol"
    row = result["items"][0]
    row["evidence_level"] = "confirmed"
    row["observed_frame_ids"] = list(row["expected_frame_ids"])
    row["temporal_visual_understanding"].update(
        {
            "event_sequence": ["state changes"],
            "observed_frame_count": len(row["expected_frame_ids"]),
            "observed_frame_ids": list(row["expected_frame_ids"]),
            "per_frame_observations": [
                {"frame_id": frame_id, "observation": "seen"}
                for frame_id in row["expected_frame_ids"][:-1]
            ],
        }
    )
    input_json = Path(prepared["result_template_path"]).with_name("partial-temporal.json")
    _write_json(input_json, result)
    rejected = import_codex_vision_results(bundle, input_json=input_json)
    assert rejected["status"] == "rejected"
    assert "per_frame_observations_incomplete" in {issue["key"] for issue in rejected["validation_issues"]}


def test_temporal_unavailable_preserves_complete_per_frame_observation(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, semantic_count=1, temporal=True)
    prepared = prepare_codex_vision_session(bundle, kind="temporal", indexes=[2])
    result = json.loads(Path(prepared["result_template_path"]).read_text(encoding="utf-8"))
    result["model_actual"] = "gpt-5.6-sol"
    row = result["items"][0]
    frame_ids = list(row["expected_frame_ids"])
    observations = [{"frame_id": frame_id, "observation": "inspected but ambiguous"} for frame_id in frame_ids]
    row["evidence_level"] = "unavailable"
    row["unavailable_reason"] = "the state change remains ambiguous"
    row["observed_frame_ids"] = frame_ids
    row["temporal_visual_understanding"].update(
        {
            "observed_frame_count": len(frame_ids),
            "observed_frame_ids": frame_ids,
            "per_frame_observations": observations,
        }
    )
    input_json = Path(prepared["result_template_path"]).with_name("unavailable-temporal.json")
    _write_json(input_json, result)

    imported = import_codex_vision_results(bundle, input_json=input_json)

    assert imported["status"] == "imported"
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    understanding = timeline[1]["temporal_visual_understanding"]
    assert understanding["evidence_level"] == "unavailable"
    assert understanding["per_frame_observations"] == observations


def test_temporal_candidates_advance_one_immutable_group_per_round(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, semantic_count=0, temporal=True)
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    assets = bundle / "assets"
    for group_index in (2, 3):
        frame_paths = []
        for position in range(5):
            frame = assets / f"temporal_{group_index}_{position:02d}_{group_index * 10000 + position * 250:010d}ms.jpg"
            frame.write_bytes(f"synthetic-group-{group_index}-{position}".encode("ascii"))
            frame_paths.append(f"assets/{frame.name}")
        timeline.append(
            {
                "index": group_index,
                "start": float(group_index * 2),
                "end": float(group_index * 2 + 1),
                "visual_route": "temporal_sequence",
                "temporal_frame_paths": frame_paths,
                "quality_issues": ["temporal_sequence_without_analysis"],
            }
        )
    _write_json(bundle / "timeline.json", timeline)

    selected_rounds = []
    for _ in range(3):
        prepared = prepare_codex_vision_session(bundle, kind="temporal")
        selected_rounds.append(prepared["selected_indexes"])
        result = json.loads(Path(prepared["result_template_path"]).read_text(encoding="utf-8"))
        result["model_actual"] = "gpt-5.6-sol"
        row = result["items"][0]
        frame_ids = list(row["expected_frame_ids"])
        row["evidence_level"] = "confirmed"
        row["observed_frame_ids"] = frame_ids
        row["temporal_visual_understanding"].update(
            {
                "event_sequence": ["one visible state transition"],
                "observed_frame_count": len(frame_ids),
                "observed_frame_ids": frame_ids,
                "per_frame_observations": [
                    {"frame_id": frame_id, "observation": "observed"}
                    for frame_id in frame_ids
                ],
            }
        )
        input_json = Path(prepared["result_template_path"]).with_name("result.json")
        _write_json(input_json, result)
        imported = import_codex_vision_results(bundle, input_json=input_json)
        assert imported["status"] == "imported"

    assert selected_rounds == [[1], [2], [3]]


def test_restore_undoes_codex_batch_without_touching_raw_or_human_fields(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, semantic_count=1)
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    timeline[0]["raw_asr"] = {"text": "immutable raw ASR"}
    timeline[0]["human_review_notes"] = "keep this note"
    _write_json(bundle / "timeline.json", timeline)
    prepared = prepare_codex_vision_session(bundle, kind="semantic", indexes=[1])
    input_json = _fill_semantic_results(Path(prepared["result_template_path"]), ["confirmed"])
    imported = import_codex_vision_results(bundle, input_json=input_json)
    run_id = imported["run_audit"]["record"]["run_id"]
    plan = vision_analysis_restore_plan(bundle, run_id=run_id)
    restored = vision_analysis_apply_restore(bundle, plan_json=plan["json_path"], execute=True, confirm_run_id=run_id)
    after = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))[0]
    assert restored["summary"]["status"] == "ok"
    assert restored["summary"]["applied_count"] == 1
    assert not after.get("visual_understanding")
    assert after["raw_asr"] == {"text": "immutable raw ASR"}
    assert after["human_review_notes"] == "keep this note"


def test_evidence_levels_drive_summary_trace_quality_gate_and_ui(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, semantic_count=3)
    prepared = prepare_codex_vision_session(bundle, kind="semantic", indexes=[1, 2, 3])
    input_json = _fill_semantic_results(Path(prepared["result_template_path"]), ["confirmed", "inferred", "unavailable"])
    imported = import_codex_vision_results(bundle, input_json=input_json)
    assert imported["status"] == "imported"
    timeline = json.loads((bundle / "timeline.json").read_text(encoding="utf-8"))
    levels = evidence_level_counts(timeline)["visual"]
    assert levels == {"confirmed": 1, "inferred": 1, "unavailable": 1, "not_seen": 0}
    assert visual_evidence_state(timeline[2])["consumable"] is False
    inferred = _compact_evidence_item(2, timeline[1], kind="visual_understanding")
    assert inferred["evidence_level"] == "inferred"
    assert "推断" in inferred["excerpt"] and "可能" in inferred["excerpt"]
    assert any("seen_unavailable" in reason for reason in _review_gap_reasons(timeline[2]))
    digest = _chapter_citation_digest([], {"visual_items": [inferred], "review_gaps": []})
    assert digest[0]["source_type"] == "visual_understanding_inferred"
    assert digest[0]["evidence_level"] == "inferred"

    claim = timeline[1]["visual_understanding"]["interface_state"]
    failed = _inferred_visual_language_gate(bundle, f"## 分段总结\n\n{claim}\n\n## 待复核点\n\n- 无")
    passed = _inferred_visual_language_gate(bundle, f"## 分段总结\n\n推断：画面可能显示{claim}\n\n## 待复核点\n\n- 无")
    assert failed["passed"] is False
    assert passed["passed"] is True

    row = _timeline_row(timeline[2], 3)
    assert row["has_visual_understanding"] is False
    assert row["visual_evidence_level"] == "unavailable"
    assert "unavailable" in row["evidence_flags"]
    panel = _evidence_status_panel_html(
        {
            "visual_evidence_levels": {"visual": levels, "temporal": {"confirmed": 0, "inferred": 0, "unavailable": 0, "not_seen": 3}},
            "codex_vision_session": {"session_id": "session-123", "phase": "imported", "model_actual": "gpt-5.6-sol", "validation": {"status": "imported"}, "commands": {"prepare_semantic": "prepare semantic", "prepare_temporal": "prepare temporal", "import_results": "import"}},
        }
    )
    assert "准备 Codex 单帧视觉包" in panel
    assert "session-123" in panel
    assert "Raw result" in panel
    assert "gpt-5.6-sol" in panel

    legacy = visual_evidence_state({"visual_understanding": {"summary": "legacy result without evidence_level"}})
    assert legacy["evidence_level"] == "confirmed"
    assert legacy["consumable"] is True


def test_task_console_exposes_both_prepare_commands(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, semantic_count=1)
    commands = _build_commands(bundle, {}, {"counts": {"timeline_items": 1}})
    by_key = {row["key"]: row for row in commands}
    assert "prepare-codex-vision-session" in by_key["codex_vision_semantic"]["command"]
    assert "--kind temporal" in by_key["codex_vision_temporal"]["command"]
    assert "import-codex-vision-results" in by_key["codex_vision_import"]["command"]
